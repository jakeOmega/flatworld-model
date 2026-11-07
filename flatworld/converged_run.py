"""
flatworld.converged_run -- the M3.3/M3.4 time-averaged-convergence harness for the fused AD core.

The fused core (``stage_ad_core.StageADCore``) is deeply turbulent (M2a Ra~3.7e21), so its pointwise
``run()`` drift stop is UNREACHABLE (the shakeout proved driftMax floors at a turbulent ~few*1e-4 while
the pointwise tol=1e-5 never fires). "Converged" must therefore be a TIME-AVERAGED mean-state criterion,
and the averaging window is set by the internal-variability NOISE FLOOR -- so M3.3 (one converged base
run) and M3.4 (the noise floor) are ONE measurement (M3_SUBTASKS.md; design: M33_M34_DESIGN.md).

THE ONE ESTIMATOR (tied across stop / floor / pin): the converged T(theta) is the time-average, over the
statistically-steady tail, of the masked near-surface air-temperature profile T_atm[:,-1] (theta<=conv).
  * the M3.4 FLOOR eps = the sampling uncertainty of that estimator = sigma1 / sqrt(N_eff), N_eff =
    T_tail/IAT, sigma1 on the EXPONENTIAL-DETRENDED tail (per-theta);
  * the M3.3 STOP = drive the residual transient below the floor via the PROJECTED remaining change to the
    asymptote (T_inf from a slow-exponential fit), stopping at offset < eps/3 (NOT a local slope, which
    trips early on an exponential approach);
  * M3.5's closure-sensitivity = the spread of that SAME estimator across the closure knobs, vs eps.

This module holds the pure-function estimator (shared by the live monitor and the offline analyzer so
they cannot drift), the live ``MeanStateMonitor`` (used for the cheaper M3.5 sweep once eps is known),
the run-to-guard driver, the offline ``analyze_convergence``, the M3.4 ``noise_floor``, and
``converged_diagnostics`` (physicality + the box-tested conservation re-gate + external cross-checks +
the -4C-ribbon enthalpy audit). Built test-first (tests/test_converged_run.py).
"""
from __future__ import annotations
from . import thermo as _qmax_thermo
from collections import namedtuple
import time
import numpy as np
from . import constants as C
from . import convergence as conv
from . import _ckpt_fingerprint as _ckfp                       # checkpoint config/world fingerprint (resume guard)
from . import radiation as rad
from .stage_ad_core import bulk_air_sea_sst
from .device import to_host

ExpFit = namedtuple("ExpFit", "T_inf A tau ok extrapolating span")
Floor = namedtuple("Floor", "eps sigma1 ess iat n")
ConvResult = namedtuple("ConvResult",
                        "converged tau T_inf A tailmean eps bias max_ratio tinf_shift "
                        "extrapolating tail_start_idx found stable")
Trajectory = namedtuple("Trajectory", "iters sim_time profiles sst theta_masked max_psi stop_reason summary")
FloorReport = namedtuple("FloorReport", "eps eps_block iat ess blocks_ok n")
BetweenMember = namedtuple("BetweenMember", "eps rel_uncertainty m")


def _merge_hist_rows(prev, row):
    """Concatenate a new history row (dict of (1, ...) arrays) onto a loaded history dict, tolerating
    SCHEMA DRIFT in both directions: a key new to ``row`` gets a NaN-padded history (adding a
    diagnostic channel mid-campaign must not break old files), and a key only in ``prev`` is KEPT,
    NaN-padded forward (``save_checkpoint`` overwrites the file, so dropping/renaming a channel must
    not erase its accumulated record). A prev-only key whose leading dim is not the history length
    (static metadata, e.g. ``theta_deg``) is carried through unchanged. Renderers must tolerate NaN
    rows (matplotlib gaps them). Shared by the in-run profhist appender and the supervisor-side
    ``_rad_diags`` history."""
    if prev is None or "sim_day" not in prev:
        return dict(row)
    n_prev = len(np.atleast_1d(prev["sim_day"]))
    n_row = len(np.atleast_1d(row["sim_day"]))
    out = {}
    for k, v in row.items():
        if k in prev:
            ph = np.atleast_1d(prev[k]).reshape(-1, *v.shape[1:])
        else:
            ph = np.full((n_prev, *v.shape[1:]), np.nan)
        out[k] = np.concatenate([ph, v], axis=0)
    for k in prev.keys() - row.keys():
        pv = np.atleast_1d(prev[k])
        if pv.shape[0] == n_prev:                                # a history channel -> pad forward
            out[k] = np.concatenate([pv, np.full((n_row, *pv.shape[1:]), np.nan)], axis=0)
        else:                                                    # static metadata -> carry as-is
            out[k] = prev[k]
    return out


# ======================================================================================
# PURE-FUNCTION ESTIMATOR (shared by the live monitor and the offline analyzer)
# ======================================================================================
def integrated_autocorr_time(x):
    """Integrated autocorrelation time ``tau`` (in SAMPLE units) of a 1-D series ``x``, via Sokal's
    adaptive windowing: tau = 1 + 2*sum_{k=1}^{M} rho(k), with M the smallest window s.t. M >= c*tau(M)
    (c=5) -- the window truncates the noisy large-lag tail before it blows the sum up. The effective
    sample size is then ESS = N/tau and the std of the sample mean is sigma*sqrt(tau/N). For a stationary
    AR(1) with coefficient phi the answer is (1+phi)/(1-phi); for white noise it is 1. Returns (tau, M)."""
    x = np.asarray(x, float)
    n = x.size
    if n < 16:
        return float("nan"), 0
    x = x - x.mean()
    var = float(np.dot(x, x) / n)
    if var <= 0:
        return 1.0, 0
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    f = np.fft.rfft(x, nfft)
    acf = np.fft.irfft(f * np.conj(f), nfft)[:n].real / (n * var)   # normalized autocovariance, O(n log n)
    tau = 1.0
    c = 5.0
    M = n - 1
    for k in range(1, n):
        tau += 2.0 * acf[k]
        if k >= c * tau:
            M = k
            break
    return float(max(tau, 1.0)), M


def fit_exponential_approach(t, y, extrap_frac=0.6):
    """Fit the slow-exponential approach ``y = T_inf - A*exp(-t/tau)`` to a scalar time series (the TREND;
    M33_M34_DESIGN.md). ``t`` in sim-seconds (or any consistent unit), ``y`` the channel to fit (use the
    WET/ocean sub-mask SST mean -- the isolated slow mode -- NOT a wet+dry blend, which is a sum of two
    exponentials). Returns an ``ExpFit(T_inf, A, tau, ok, extrapolating, span)``:
      * ``extrapolating`` = tau > extrap_frac*span -- the fit has not seen the asymptote, so T_inf is an
        EXTRAPOLATION and tau is a lower bound (the box-test caution: trust only what the data resolves);
      * ``ok`` = the fit converged to finite params.
    Used both for the trend stop and (with ``extrapolating``) to decide whether the run is long enough."""
    from scipy.optimize import curve_fit
    t = np.asarray(t, float); y = np.asarray(y, float)
    span = float(t[-1] - t[0]) if t.size > 1 else 0.0

    def model(tt, T_inf, A, tau):
        return T_inf - A * np.exp(-tt / np.maximum(tau, 1e-9))

    try:
        p0 = [y[-1], y[-1] - y[0], max(span / 3.0, 1.0)]
        popt, _ = curve_fit(model, t, y, p0=p0, maxfev=20000,
                            bounds=([-1e4, -1e4, 1e-6], [1e4, 1e4, 1e12]))
        T_inf, A, tau = (float(v) for v in popt)
        ok = np.all(np.isfinite(popt))
    except Exception:
        return ExpFit(float(np.mean(y)), 0.0, float("inf"), False, True, span)
    return ExpFit(T_inf, A, tau, bool(ok), bool(tau > extrap_frac * span), span)


def sampling_floor(resid):
    """The M3.4 noise floor ``eps`` = the sampling uncertainty of the TIME-MEAN of a (stationary,
    EXPONENTIAL-DETRENDED) series ``resid``. For a correlated series the variance of the mean is inflated
    over the iid sigma^2/N by the integrated autocorrelation time: Var(mean) = sigma^2 * IAT / N, so
        eps = sigma1 * sqrt(IAT / N) = sigma1 / sqrt(ESS),  ESS = N / IAT.
    (The iid floor sigma1/sqrt(N) under-estimates by sqrt(IAT) -- the trap the review flags.) ``resid``
    MUST be detrended first (the slow approach is not internal variability; it would inflate sigma1).
    Returns ``Floor(eps, sigma1, ess, iat, n)``."""
    resid = np.asarray(resid, float)
    n = resid.size
    sigma1 = float(np.std(resid, ddof=1)) if n > 1 else 0.0
    iat, _ = integrated_autocorr_time(resid)
    if not np.isfinite(iat) or iat <= 0:
        iat = 1.0
    ess = n / iat
    eps = sigma1 / np.sqrt(max(ess, 1e-30))
    return Floor(float(eps), sigma1, float(ess), float(iat), int(n))


def noise_floor(resid, n_blocks=8):
    """The M3.4 WITHIN-RUN floor of the tail-mean of a (detrended, stationary) series ``resid``, by the
    two numerical routes that must agree when the blocks are long enough (the '(a≈b) window-sizing
    self-check', M33_M34_DESIGN.md):
      * route (a) IAT/ESS: ``eps = sigma1/sqrt(ESS)`` (``sampling_floor``);
      * route (b) BLOCK resampling: split into ``n_blocks`` contiguous blocks, ``eps_block`` =
        std(block-means)/sqrt(n_blocks) -- the floor of the full-tail mean from the block scatter.
    ``blocks_ok`` flags whether the block length exceeds ~5*IAT (else a≈b is NOT a valid check -- report
    it, do not read a bogus agreement as passing). Returns ``FloorReport(eps, eps_block, iat, ess,
    blocks_ok, n)``."""
    x = np.asarray(resid, float)
    n = x.size
    fl = sampling_floor(x)
    block_len = n // max(n_blocks, 1)
    if block_len >= 2 and n_blocks >= 2:
        bmeans = np.array([x[i * block_len:(i + 1) * block_len].mean() for i in range(n_blocks)])
        eps_block = float(np.std(bmeans, ddof=1)) / np.sqrt(n_blocks)
    else:
        eps_block = float("nan")
    blocks_ok = bool(block_len > 5.0 * fl.iat)
    return FloorReport(fl.eps, eps_block, fl.iat, fl.ess, blocks_ok, n)


def between_member_floor(tailmeans):
    """Route (c), the genuinely INDEPENDENT floor (M33_M34_DESIGN.md): the between-member std of the
    converged tail-mean profiles across a perturbed-IC ensemble (blind to autocorrelation, so it catches
    an under-sized within-run IAT). ``tailmeans`` is (M, ntheta). Returns ``BetweenMember(eps[ntheta],
    rel_uncertainty, m)`` -- ``eps`` per-theta, ``rel_uncertainty`` ~ 1/sqrt(2(M-1)) the std estimate's
    own uncertainty (it shrinks with ensemble size; report the floor WITH it)."""
    T = np.asarray(tailmeans, float)
    m = T.shape[0]
    eps = np.std(T, axis=0, ddof=1) if m > 1 else np.zeros(T.shape[1])
    rel = 1.0 / np.sqrt(2.0 * max(m - 1, 1))
    return BetweenMember(eps, float(rel), int(m))


def heat_limb_enthalpy_audit(core, dt_ref=1.0):
    """The -4C-RIBBON audit (M33_M34_DESIGN.md): ``_advect_heat`` advects POTENTIAL temperature, so it
    conserves ``int(theta_pot*dm)`` exactly but NOT ``int(cp*T*dm)``. The reframe (advisor, this session):
    ``int(cp*T*dm)`` IS the column total potential energy (TPE = internal + geopotential, hydrostatic), so
    its change under advection is NOT a mere "bookkeeping artifact" -- it is the REAL APE->KE conversion the
    overturning cell performs. The concern is what happens to that KE: the core has NO frictional-dissipation
    HEATING term (grep-verified -- ``vorticity_tendency`` removes KE/vorticity via the Smagorinsky eddy
    viscosity, but ``_thermo_substep`` heats ``T_atm`` ONLY via radiation, surface sensible heat, and
    condensation latent heat -- nothing returns dissipated KE to internal energy). So APE->KE->dissipated-and-
    LOST would be a real cold sink. CONTRARY to an earlier framing, ``rel_imbalance`` (= the domain TOA
    residual int(SW_abs - OLR)dx / gross, ``energy_conservation``) DOES see such a sink and is in fact the
    DECISIVE diagnostic: with the column closed (T_surf + ocean ML both prognostic, finite heat capacity --
    verified) a true steady state forces ``int(SW_abs-OLR)dx = D_loss``, so a real dissipation sink makes
    ``rel_imbalance`` PLATEAU at ~= ``|enthalpy_rate_frac|`` (NOT -> 0). The two are equal-and-opposite at
    convergence IFF the sink is real. A SECOND, numerical contribution also inflates the snapshot change:
    upwind theta_pot advection implicitly diffuses theta_pot; a RESOLUTION SCAN separates it (numerical part
    shrinks with nz; real APE->KE-minus-dissipation sink persists and shows up in rel_imbalance).

    Reports a dt-INDEPENDENT RATE, not a per-substep change: ``DeltaH`` from one ``_advect_heat`` substep is
    EXACTLY linear in dt (the tendency is dt-free), so ``DeltaH/dt`` is dt-independent -- which is the ONLY
    form comparable across the live path (dt = the CFL-limited dt_dyn) and the analyze path (where
    ``_last_dt_dyn`` is NOT restored from a checkpoint == 0). The headline is ``enthalpy_rate_frac`` = the
    APE->KE conversion RATE as a FRACTION of the gross absorbed-SW throughput. Judge: ``<< 1`` => the sink
    perturbs T(theta) negligibly and the cold ribbon is robust regardless; NOT ``<< 1`` (the ~7% preview)
    => the missing-dissipation sink is material, the cold-ribbon MAGNITUDE is suspect, and the resolution
    scan (check b) must separate the numerical-diffusion inflation from the real sink before the number is
    trusted. Applies one substep at ``dt_ref`` and RESTORES the state (non-mutating)."""
    c = core.cfg
    dt_ref = float(dt_ref) if dt_ref else 1.0
    exner = (core.p_mid[None, :] / C.P_SURF) ** C.KAPPA_POISSON
    H0 = float(np.sum(core.dm[None, :] * C.CP_AIR * core.T_atm) * core.dx)        # int cp*T dm dx
    M0 = float(np.sum(core.dm[None, :] * (core.T_atm / exner)) * core.dx)         # int theta_pot dm dx
    T_save = core.T_atm.copy()
    core._z = core._hydrostatic_z(core.T_atm)
    u, _ = core.velocities()
    core._advect_heat(u, dt_ref)
    H1 = float(np.sum(core.dm[None, :] * C.CP_AIR * core.T_atm) * core.dx)
    M1 = float(np.sum(core.dm[None, :] * (core.T_atm / exner)) * core.dx)
    core.T_atm = T_save                                                          # non-mutating: restore
    dH_dt = (H1 - H0) / dt_ref                                                   # spurious cp*T source RATE
    q = np.clip(core.q, 0.0, _qmax_thermo.Q_MAX)
    out = rad.column_radiation(core.p_edge[None, :], core.T_atm, q, core.T_surf, core.S, core._albedo(),
                               co2_ppm=c.co2_ppm, k_wv=c.k_wv_lw, tau_wellmixed=c.tau_lw_wellmixed)
    gross = float(np.sum((core.S - out["planetary_reflected"]) * core.dx))       # radiative throughput [W/m]
    return dict(enthalpy_rate_frac=dH_dt / (gross + 1e-30),     # APE->KE conversion rate / radiative throughput
                d_theta_mass_rel=(M1 - M0) / (abs(M0) + 1e-30),  # ~0 by construction (theta-mass conserved)
                d_enthalpy_rate_W_per_m=dH_dt, gross_absorbed_W_per_m=gross, enthalpy=H0, dt_ref=dt_ref)


def converged_diagnostics(core, *, conv_theta_max=None, ribbon_lo=44.0, ribbon_hi=52.0,
                          lineage_air_T=None):
    """The M3.3 acceptance bundle on a (converged) core: PHYSICALITY (single cell, poleward quiescence),
    the tier-1 CONSERVATION re-gate (energy/mass/water), the tier-4 EXTERNAL strength cross-checks
    (subsidence rate vs M2a -0.25 cm/s; column SST vs the independent bulk air-sea root), the OFF-LINEAGE
    shift vs an optional heat-off lineage, and the heat-limb ENTHALPY AUDIT. Composes the core's own
    methods (each separately tested) + ``heat_limb_enthalpy_audit``. The box-test (theta_max 48<->72) is an
    ORCHESTRATION concern (two cores) handled by the production driver, not here. Returns a flat dict."""
    c = core.cfg
    theta = core.theta_deg
    tmax = conv_theta_max if conv_theta_max is not None else (c.conv_theta_max or c.theta_max_deg)
    cmask = theta <= tmax
    rmask = (theta >= ribbon_lo) & (theta <= ribbon_hi)
    d = core.diagnostics()
    pq = core.poleward_quiescence()
    ec = core.energy_conservation(); mc = core.mass_conservation(); wc = core.water_conservation()
    sd = core.subsidence_diagnostics()
    # tier-4: column SST vs the independent bulk air-sea root (same wetness branch; only SW+LW sum matters).
    # Use the core's OWN radiation wrapper (core._radiate) so the surface drivers (surf_abs_sw, D_surf) carry
    # the SAME all-on physics the live surface equilibrated under (clouds / lw_window / sw_diffuse / sw_bands).
    # A bare gray column_radiation call here would reconstruct CLEAR-SKY drivers, leaving a spurious ~10 K
    # bulk-vs-live SST residual under the all-on default (ALLON_CAMPAIGN.md).
    q = np.clip(core.q, 0.0, _qmax_thermo.Q_MAX)
    out = core._radiate(q)
    # pass the DOWNWELLING drivers (surf_abs_sw, D_surf), NOT out["surf_net"] -- the latter already folds
    # in -sigma*Ts^4, which the root-finder must solve for, so passing it double-counts surface LW emission
    # and collapses the root toward the T_lo floor (the canonical call, cf. test_stage_ad_core.py).
    sst_root = bulk_air_sea_sst(out["surf_abs_sw"], out["D_surf"], core.T_atm[:, -1], q[:, -1], c.wind,
                                wetness=core._wetness())
    sst_resid = np.abs(core.T_surf - sst_root)
    res = dict(
        # physicality
        poleward_quiescence_ratio=pq["ratio"], surf_sign_changes=int(d["surf_sign_changes"]),
        psi_single_signed=bool(d["psi_single_signed"]), max_abs_psi=float(d["max_abs_psi"]),
        rising_theta=float(d["rising_theta"]), finite=bool(core._state_is_finite()),
        # tier-1 conservation
        energy_rel_imbalance=ec["rel_imbalance"], mass_rel_div_deep=mc["rel_div_deep"],
        water_closure_cum=wc["closure_cum"],
        # tier-4 external strength cross-checks
        w_sub_ribbon_cm_s=float(np.mean(sd["w_sub_cm_s"][rmask])) if rmask.any() else float("nan"),
        sst_vs_bulk_root_max_K=float(np.max(sst_resid[cmask])),     # max over the mask (hot-end dominated)
        sst_vs_bulk_root_ribbon_K=float(np.max(sst_resid[rmask])) if rmask.any() else float("nan"),
        # the deliverable curve + the enthalpy audit
        theta_deg=theta, air_T_C=d["T_air_surf_C"], sst_C=d["T_surf_C"],
        ribbon_air_T_C=float(np.mean(d["T_air_surf_C"][rmask])) if rmask.any() else float("nan"),
        enthalpy_audit=heat_limb_enthalpy_audit(core),
    )
    if lineage_air_T is not None:
        dT = d["T_air_surf_C"][cmask] - np.asarray(lineage_air_T)   # heat-on minus heat-off, over cmask
        r_in_c = rmask[cmask]                                       # ribbon columns within the masked set
        res["off_lineage_max_dT_C"] = float(np.max(np.abs(dT)))
        res["off_lineage_ribbon_dT_C"] = float(np.mean(dT[r_in_c])) if r_in_c.any() else float("nan")
    return res


def project_asymptote(t, profiles, tau):
    """Per-theta asymptote ``T_inf(theta)`` and amplitude ``A(theta)`` at a FIXED shared ``tau`` (the slow
    thermal mode is ~global; M33_M34_DESIGN.md). With tau fixed, ``y_theta(t) = T_inf(theta) -
    A(theta)*exp(-t/tau)`` is LINEAR in [T_inf(theta), A(theta)], so a single least-squares solve recovers
    every column at once (far more robust than nx independent nonlinear fits of the turbulent per-column
    series). ``profiles`` is (Nt, ntheta). Returns (T_inf[ntheta], A[ntheta])."""
    t = np.asarray(t, float); P = np.asarray(profiles, float)
    X = np.column_stack([np.ones_like(t), -np.exp(-t / max(tau, 1e-9))])   # (Nt, 2): [T_inf, A]
    coef, *_ = np.linalg.lstsq(X, P, rcond=None)                           # (2, ntheta)
    return coef[0], coef[1]


def analyze_convergence(t, profiles, tau_channel=None, *, eps_frac=1.0 / 3.0, burn=0.4,
                        stability_tol=3.0):
    """The composed asymptote-projection STOP (M33_M34_DESIGN.md). Decide whether the masked-profile
    trajectory ``profiles`` (Nt, ntheta) at times ``t`` has settled, by driving the residual TRANSIENT
    below the sampling floor -- NOT a local slope (which trips early on an exponential approach):

      1. discard the first ``burn`` fraction (the coupled spin-up / fast modes), so the slow channel is
         isolated -- mirrors fitting tau on the WET sub-mask (the clean slow mode);
      2. fit the slow timescale ``tau`` on the (post-burn) ``tau_channel`` (default: the kept-window mean);
         if ``tau`` is not RESOLVED within the kept span (``extrapolating``: tau > 0.6*span) the asymptote
         is an extrapolation -> NOT converged (the two-timescale guard);
      3. project per-theta ``T_inf`` at fixed ``tau`` (project_asymptote) on the kept window;
      4. sweep the tail-start; the tail-mean's residual-transient BIAS = |T_inf - tailmean| must fall
         below ``eps_frac`` (=1/3) of the per-theta sampling floor ``eps`` (so the transient is subdominant
         to the floor -- reconciles the stop with the between-member agreement gate, which is ~eps);
      5. require ``T_inf`` STABLE across the kept window's halves (within ``stability_tol``*eps) -- the
         box-test analog that catches an unresolved slow mode the single-exp fit under-projects.

    Returns a ``ConvResult``. ``converged`` = a qualifying tail exists AND not extrapolating AND stable.
    The reported ``tailmean``/``eps`` are the chosen tail's (the longest qualifying tail -> tightest eps);
    if none qualifies they are the whole kept window's (diagnostic). The SAME estimator the live
    ``MeanStateMonitor`` uses for M3.5, so they cannot drift."""
    t = np.asarray(t, float); P = np.asarray(profiles, float)
    Nt, ntheta = P.shape
    span = float(t[-1] - t[0]); t0 = float(t[0])
    keep = t >= t0 + burn * span
    tw = t[keep]; Pw = P[keep]
    if tw.size < 32:                                   # too short to analyze
        zero = np.zeros(ntheta)
        return ConvResult(False, float("nan"), zero, zero, P.mean(0), zero, zero, float("inf"),
                          zero, True, 0, False, False)
    channel = Pw.mean(axis=1) if tau_channel is None else np.asarray(tau_channel, float)[keep]
    fit = fit_exponential_approach(tw, channel)
    tau = fit.tau
    T_inf, A = project_asymptote(tw, Pw, tau)
    # T_inf stability across the kept window's halves (same fixed tau)
    h = tw >= (tw[0] + 0.5 * (tw[-1] - tw[0]))
    T_inf_h, _ = project_asymptote(tw[h], Pw[h], tau)
    tinf_shift = np.abs(T_inf - T_inf_h)

    def _eval(i):
        tt = tw[i:]; tail = Pw[i:]
        decay = float(np.mean(np.exp(-tt / max(tau, 1e-9))))     # <exp(-t/tau)> over the tail
        trend = T_inf[None, :] - A[None, :] * np.exp(-tt[:, None] / max(tau, 1e-9))
        resid = tail - trend
        eps = np.array([sampling_floor(resid[:, j]).eps for j in range(ntheta)])
        # BIAS = the MODEL-projected residual transient still averaged into the tail-mean (= the advisor's
        # "projected remaining change to the asymptote"), NOT |T_inf - tailmean| -- the latter also charges
        # the projection NOISE in T_inf, which would spuriously block convergence once decayed.
        bias = np.abs(A) * decay
        tailmean = tail.mean(axis=0)
        ratio = float(np.max(bias / np.maximum(eps, 1e-30)))
        return dict(i=i, ratio=ratio, tailmean=tailmean, eps=eps, bias=bias)

    Nw = tw.size
    best = None
    for fr in np.linspace(0.0, 0.7, 15):               # ascending => first hit is the LONGEST valid tail
        i = int(np.searchsorted(tw, tw[0] + fr * (tw[-1] - tw[0])))
        if Nw - i < 16:
            continue
        cand = _eval(i)
        if cand["ratio"] < eps_frac:
            best = cand
            break
    found = best is not None
    if not found:
        best = _eval(0)                                # diagnostic: the whole kept window
    stable = float(np.max(tinf_shift / np.maximum(best["eps"], 1e-30))) < stability_tol
    converged = bool(found and (not fit.extrapolating) and stable)
    return ConvResult(converged, float(tau), T_inf, A, best["tailmean"], best["eps"], best["bias"],
                      best["ratio"], tinf_shift, bool(fit.extrapolating), int(best["i"]), bool(found),
                      bool(stable))


# ======================================================================================
# RUN-TO-GUARD DRIVER (M3.3/M3.4 execution model: run to a generous guard, LOG the masked-profile
# trajectory + checkpoint, then analyze OFFLINE -- ``eps`` is co-measured with the stop, so a live
# early-termination would have to know ``eps`` it does not yet have; M33_M34_DESIGN.md). Reuses the
# tested guard helpers (``convergence.save_checkpoint``/``load_checkpoint``, the core's
# ``_state_is_finite``/``state``/``load_state``) so the safety guarantees match ``StageADCore.run``.
# ======================================================================================
def run_to_steady_state(core, *, sim_time_target, conv_theta_max=None, wet_thresh=0.5,
                        sample_every=200, checkpoint_path=None, checkpoint_every=20000, log_path=None,
                        profhist_path=None, profhist_min_days=25.0,
                        max_iter=20_000_000, max_seconds=20 * 3600.0, verbose=True):
    """Drive ``core`` (a coupled ``StageADCore``) to a generous SIM-TIME guard, sampling the masked
    near-surface air-T(theta) PROFILE + the wet/ocean-sub-mask SST channel (the clean slow channel for the
    tau fit) every ``sample_every`` steps into a checkpointed trajectory for OFFLINE ``analyze_convergence``.

    Resumable: with a ``checkpoint_path`` it resumes the core state + sim-time, and with a ``log_path`` it
    re-loads + appends the trajectory -- so a run that outlives the session continues from disk (the box
    does not persist detached processes; the CHECKPOINT is the persistence). A NON-FINITE state aborts
    immediately via ``FloatingPointError`` WITHOUT checkpointing (preserve the last good resume state) --
    the M3 cost guard carried from ``StageADCore.run``. Stops at ``sim_time_target`` (the converged-tail
    budget), or the ``max_iter``/``max_seconds`` backstops. Returns a ``Trajectory``.

    ``profhist_path``: optional convergence-watch history (.npz). At checkpoint time -- AFTER the
    run-critical checkpoint+trajectory writes, inside a swallow-all guard so it can never kill a run --
    append the FULL-DOMAIN profiles {T_surf, lowest-layer air-T, resolved surface wind u_surf, and the
    precip MEAN since the last append (accumulated EVERY step: convective precip is intermittent and the
    dt is CFL-adaptive, so a subsample would be a biased estimator)} vs theta. Appends are time-gated to
    >= ``profhist_min_days`` sim-days apart (coupled-phase checkpoints land every ~1-2 sim-days; ungated
    history would bloat). Rendering is a separate single process (``_ckpt_plots`` -- no matplotlib here)."""
    c = core.cfg
    theta = core.theta_deg
    tmax = conv_theta_max if conv_theta_max is not None else (c.conv_theta_max or c.theta_max_deg)
    cmask = theta <= tmax
    theta_masked = theta[cmask]

    iters, sim_t, profs, ssts, psis = [], [], [], [], []
    sim_time = 0.0
    # resume the core + sim-time from a checkpoint, and the trajectory from its log
    if checkpoint_path:
        st, meta = conv.load_checkpoint(checkpoint_path)
        if st is not None:
            core.load_state(st)
            _ckfp.validate_resume(meta, core.cfg, mode=core.cfg.resume_validate, context="converged_run")
            sim_time = float(meta.get("sim_time", 0.0)) if meta else 0.0
            if verbose:
                print(f"[converged_run] resumed at iter {core.iter}, sim {sim_time/86400:.1f} d")
    if log_path:
        prev, _ = conv.load_checkpoint(log_path)            # the log is an npz with the same loader
        if prev is not None and "sim_time" in prev:
            iters = list(np.atleast_1d(prev["iters"])); sim_t = list(np.atleast_1d(prev["sim_time"]))
            profs = list(np.atleast_2d(prev["profiles"])); ssts = list(np.atleast_1d(prev["sst"]))
            psis = list(np.atleast_1d(prev["max_psi"]))

    def _sample():
        prof = (core.T_atm[:, -1][cmask] - C.T_FREEZE).copy()
        wet = core._wetness()
        wmask = cmask & (wet > wet_thresh)
        sst = float(np.mean(core.T_surf[wmask] if wmask.any() else core.T_surf[cmask])) - C.T_FREEZE
        iters.append(int(core.iter)); sim_t.append(float(sim_time))
        profs.append(prof); ssts.append(sst)
        psis.append(float(np.abs(np.asarray(core.psi)).max()))

    def _save_log():
        if log_path:
            conv.save_checkpoint(log_path, dict(iters=np.array(iters), sim_time=np.array(sim_t),
                                                profiles=np.array(profs), sst=np.array(ssts),
                                                max_psi=np.array(psis), theta_masked=theta_masked))

    # --- profhist: per-step precip accumulator + time-gated profile-history appends (convergence watch) ---
    precip_acc = np.zeros(c.nx)         # sum of _precip*dt [kg/m^2] since the last append
    flash_acc = np.zeros(c.nx)          # sum of _surf_flash*dt (surface flash-boil, PR #298; flag-gated:
    #   the ``precip`` channel keeps its meaning -- flux LEAVING the atmosphere, pre-flash -- and the
    #   flash is a SEPARATE channel so downstream liquid consumers (_build_bl_precip_clim -> geology)
    #   can subtract it; flag OFF the channel is not even written, so existing artifacts are unchanged)
    precip_acc_t = 0.0                  # accumulated sim-seconds since the last append
    prof_last_day = -np.inf
    if profhist_path:
        prev_h, _ = conv.load_checkpoint(profhist_path)
        if prev_h is not None and "sim_day" in prev_h:
            prof_last_day = float(np.atleast_1d(prev_h["sim_day"])[-1])

    def _profhist_append():
        """Append one profile row; NEVER raises (plot data is expendable, the run is not)."""
        nonlocal precip_acc, flash_acc, precip_acc_t, prof_last_day
        sim_day = sim_time / 86400.0
        if not profhist_path or (sim_day - prof_last_day) < profhist_min_days:
            return
        try:
            u_surf = (to_host(core._u_surf) if core._u_surf is not None
                      else np.zeros(c.nx))                      # None only during a no-dynamics (RCE) phase
            #     Phase D: _u_surf may live on device (GPU) -> to_host brings it back (no-op on CPU)
            precip_mean = (precip_acc / precip_acc_t * 86400.0 if precip_acc_t > 0.0
                           else np.zeros(c.nx))                  # [mm/day], mean since the last append
            prev_h, _ = conv.load_checkpoint(profhist_path)
            row = dict(sim_day=np.array([sim_day]),
                       T_surf=np.asarray(core.T_surf)[None, :] - C.T_FREEZE,
                       T_air=np.asarray(core.T_atm)[None, :, -1] - C.T_FREEZE,
                       u_surf=u_surf[None, :], precip=precip_mean[None, :])
            if c.surf_flash_boil:
                # surface flash-boil (PR #298): the mean flashed flux [mm/day], same window as
                # ``precip`` -- surface-reaching LIQUID = precip - surf_flash. Only written on
                # flag-ON runs (_merge_hist_rows NaN-pads it across mixed campaigns).
                row["surf_flash"] = (flash_acc / precip_acc_t * 86400.0 if precip_acc_t > 0.0
                                     else np.zeros(c.nx))[None, :]
            # radiation/cloud channels from the step's OWN cached radiation (core._last_rad: zero
            # extra cost, fresh within dt_rad) -- same key names as the _rad_diags offline recompute,
            # so future runs carry them natively at profhist density.
            out = core._last_rad
            if out is not None:
                qc = np.clip(core.q, 0.0, _qmax_thermo.Q_MAX)
                S_in = np.asarray(core.S_sw_in, dtype=float)
                lit = S_in > 1e-6
                zeros = np.zeros(c.nx)
                extra = dict(
                    OLR=np.asarray(out["OLR"]),
                    toa_net=(S_in - np.asarray(out["planetary_reflected"])) - np.asarray(out["OLR"]),
                    cloud_cover=np.asarray(out.get("cloud_cover", zeros)),
                    cloud_sw_albedo=np.asarray(out.get("cloud_sw_albedo", zeros)),
                    sw_transmit=np.where(lit, np.asarray(out["surf_down_sw"])
                                         / np.where(lit, S_in, 1.0), np.nan),
                    albedo_planet=np.where(lit, np.asarray(out["planetary_reflected"])
                                           / np.where(lit, S_in, 1.0), np.nan),
                    albedo_surf=np.asarray(core._albedo()),
                    tau_lw_col=np.asarray(out["dtau_lw"]).sum(axis=-1),
                    tau_lw_cloud_col=(np.asarray(out["dtau_lw_cloud"]).sum(axis=-1)
                                      if "dtau_lw_cloud" in out else zeros),
                    tau_win_col=(np.asarray(out["dtau_win"]).sum(axis=-1)
                                 if "dtau_win" in out else zeros),
                    tau_sw_col=np.asarray(out["dtau_sw"]).sum(axis=-1),
                    vapor_path=np.sum(qc * core.dm[None, :], axis=1))
                row.update({k: np.asarray(v, dtype=float)[None, :] for k, v in extra.items()})
            row = _merge_hist_rows(prev_h, row)
            row["theta_deg"] = theta
            conv.save_checkpoint(profhist_path, row)
            precip_acc = np.zeros(c.nx); flash_acc = np.zeros(c.nx); precip_acc_t = 0.0; prof_last_day = sim_day
        except Exception as e:  # noqa: BLE001 -- the watch hook must never kill the run
            if verbose:
                print(f"[converged_run] profhist append failed (ignored): {type(e).__name__}: {e}")

    t0 = time.time()
    stop_reason = None
    while True:
        core.step()
        dt_step = core._last_dt_dyn if core.run_dynamics else c.dt
        sim_time += dt_step
        if profhist_path:
            precip_acc += np.asarray(core._precip) * dt_step    # every step: unbiased time-mean (see docstring)
            if c.surf_flash_boil:                               # flag-gated (see flash_acc note above)
                flash_acc += np.asarray(core._surf_flash) * dt_step
            precip_acc_t += dt_step
        if not core._state_is_finite():
            # blow-up: abort WITHOUT checkpoint (do not poison the last good resume state)
            msg = f"NON-FINITE state at iter {core.iter} (CFL/closure blow-up); aborting WITHOUT checkpoint."
            if verbose:
                print(f"[converged_run] {msg}")
            raise FloatingPointError(msg)
        if core.iter % sample_every == 0:
            _sample()
        if checkpoint_path and core.iter % checkpoint_every == 0:
            conv.save_checkpoint(checkpoint_path, core.state(),
                                 {"iter": core.iter, "sim_time": sim_time,
                                  "ad_fingerprint": core.fingerprint()})
            _save_log()
            _profhist_append()                          # AFTER the run-critical writes; never raises
        if sim_time >= sim_time_target:
            stop_reason = f"sim_time target reached ({sim_time/86400:.1f} d)"
        elif core.iter >= max_iter:
            stop_reason = f"GUARD: max_iter={max_iter}"
        elif (time.time() - t0) >= max_seconds:
            stop_reason = f"GUARD: max_seconds={max_seconds:.0f}"
        if stop_reason:
            if core.iter % sample_every != 0:
                _sample()                                   # always capture the final state
            if checkpoint_path:
                conv.save_checkpoint(checkpoint_path, core.state(),
                                     {"iter": core.iter, "sim_time": sim_time,
                                      "ad_fingerprint": core.fingerprint()})
            _save_log()
            _profhist_append()
            break
    if verbose:
        print(f"[converged_run] STOP: {stop_reason}  ({time.time()-t0:.0f}s, iter {core.iter}, "
              f"{len(sim_t)} samples)")
    summary = dict(stop_reason=stop_reason, n_iter=int(core.iter), sim_days=sim_time / 86400.0,
                   wall_s=time.time() - t0, n_samples=len(sim_t))
    return Trajectory(np.array(iters), np.array(sim_t), np.array(profs), np.array(ssts),
                      theta_masked, np.array(psis), stop_reason, summary)
