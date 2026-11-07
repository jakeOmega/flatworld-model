"""
flatworld.stage_ad_column -- the AD core's module-level column physics (#289 Part 2 split).

The single-column physics helpers of the fused anelastic core, MOVED VERBATIM out of
``stage_ad_core.py`` (issue #289 Part 2 -- a move-only split; zero token changes inside any def
body): surface wetness and albedo, the enthalpy-conserving moist convective adjustment (numpy
reference + the lazy column-parallel numba kernel), the fused prognostic-microphysics/re-evap
numba kernel with its salt-CCN/cond-tau parameter packing and scratch, the independent
single-column RCE anchor, the bulk air-sea SST cross-check, and the radiative-subsidence rate.
Unless a docstring says otherwise, thermodynamic arrays are on the (nx, nz) PRESSURE grid,
A-convention: vertical axis LAST, TOP(0) -> SURFACE(nz-1).

The lazily-built numba kernel caches ``_CONV_NUMBA_KERNEL`` and ``_MICRO_NUMBA_KERNEL`` (and the
in-place ``_MICRO_SCRATCH``) live in THIS module's namespace now; ``stage_ad_core`` forwards
reads of the rebound caches dynamically (module ``__getattr__``), so
``stage_ad_core._MICRO_NUMBA_KERNEL`` stays a LIVE view, never an import-time snapshot.

Back-compat: ``stage_ad_core`` re-exports every name here (public and underscore), so existing
``from flatworld.stage_ad_core import X`` / ``stage_ad_core.X`` references keep working; new code
should import from this module.
"""
from __future__ import annotations
import warnings
import numpy as np
from . import constants as C
from . import thermo as th
from . import radiation as rad
from .device import get_xp
from .stage_ad_kernels import hydrostatic_height


def wetness_of_Tsurf(T_surf, sst_wet_hi=C.SST_WET_HI, sst_dry=C.SST_DRY):
    """Evaporation efficiency in [0,1] (Stage A's ``_wetness``, made a pure function): open ocean (1)
    ramping to a dry salt flat (0) as SST crosses the liquid-water limit. This is what caps the
    equatorial latent runaway -- over a dry surface there is no evaporation, so the hot zone reaches a
    bounded dry-rock radiative balance instead of an unbounded greenhouse (Komabayashi-Ingersoll).

    DEVICE/dtype (Phase D): array-module-generic + fp32-gate-following via ``th._F`` -- a numpy fp64 input
    is the verbatim ``np.asarray(x, float64)`` (BYTE-IDENTICAL), a CuPy input stays on the GPU, and under
    ``set_thermo_dtype(fp32)`` it runs single precision (so the genuine-fp32 surface path has no residual
    fp64 sub-path -- the swap-reconcile in ``_thermo_substep`` only catches state, not these leaves)."""
    xp = th._xp_of(T_surf)
    Tc = th._F(T_surf, xp) - C.T_FREEZE
    return xp.clip((sst_dry - Tc) / (sst_dry - sst_wet_hi), 0.0, 1.0)


# The sigma (= p/P_SURF) ceiling defining the ALOFT layer for the resolved aloft-moisture-column
# diagnostic (``StageADCore._aloft_moisture_column``). Aloft = LOW sigma = the upper-tropospheric branch
# the single overturning cell carries poleward to dew-point at the ribbon. 0.5 = the upper half of the
# column (above the moist boundary layer; the moisture there is what is available to be lofted and
# rained out on the return branch). The column total is the absolute-magnitude replacement for the
# legacy normalize-to-shape F(theta) supply (cascade.build_aloft_supply_target). The exact threshold is
# a parameter choice (a named softness): the integral's MAGNITUDE moves with it, so the diagnostic also
# carries ``sigma`` for a downstream re-integration at a different ceiling.
ALOFT_SIGMA_THRESH = 0.5


def surface_albedo(T_surf, wetness, albedo_ocean=C.ALBEDO_OCEAN, albedo_dryland=C.ALBEDO_DRYLAND,
                   albedo_ice=C.ALBEDO_ICE):
    """Surface albedo (Stage A's ``_albedo``, made a pure function): ocean<->salt-flat by wetness, then
    brightening to ice at the cold edge (sea-ice ramp over T_surf -1.8..-3.8 C: f_ice=0 at 271.35 K =
    seawater freezing, f_ice=1 at 269.35 K). A fixed surface INPUT, not validated
    physics, so it is shared verbatim by the core and the RCE anchor (no independence lost).

    DEVICE/dtype (Phase D): array-module-generic + fp32-gate-following via ``th._F`` (numpy fp64 in ->
    byte-identical; CuPy stays on the GPU; fp32 gate -> single precision). ``wetness`` and ``T_surf`` are
    routed through ONE ``xp`` (the mixed-input rule) so a device ``T_surf`` + a device ``wetness`` compose."""
    xp = th._xp_of(T_surf, wetness)
    wet = th._F(wetness, xp)
    base = albedo_ocean * wet + albedo_dryland * (1.0 - wet)
    f_ice = xp.clip((271.35 - th._F(T_surf, xp)) / 2.0, 0.0, 1.0)
    return base + (albedo_ice - base) * f_ice


def moist_convective_adjust(T_atm, dm, z, p_mid, max_sweeps=8, q=None,
                            closure="q_homogenize", rh_surf=0.8, rh_top=0.2, freeze_gamma=False,
                            fixed_topology=False, device_kernel=None, device_aux=None):
    """Moist convective adjustment (Stage A's Manabe-style scheme, made a pure function operating IN
    PLACE on the 2-D field ``T_atm`` (nx, nz), TOP->SURFACE). Remove super-(moist-)adiabatic layers by
    enthalpy-conserving pairwise mixing, vectorized over the nx columns; ``z`` (nx, nz) the layer-center
    heights, ``dm`` (nz,) the layer masses, ``p_mid`` (nz,) the layer pressures.

    ENTHALPY CONSERVED per column EXACTLY: the pairwise mix sets
        T_k^new = (m_k T_k + m_{k-1} T_{k-1} + m_{k-1} Gamma dz) / (m_k + m_{k-1}),  T_{k-1}^new = T_k^new
        - Gamma dz, so m_k T_k^new + m_{k-1} T_{k-1}^new == m_k T_k + m_{k-1} T_{k-1} (the moist-adiabat
    offset cancels). This is the tier-1-flavoured enthalpy-conservation gate of M2b.3.

    MOISTURE TRANSPORT (``q`` (nx, nz), optional, mixed IN PLACE): convection also transports water, so
    when a layer pair is convectively mixed its (mass-weighted) water is redistributed -- WATER CONSERVED
    per column exactly. This is the sub-grid VERTICAL moisture transport (the resolved flow only carries q
    cross-track), and it is what makes the column humidity well-posed: without it, with no vertical
    transport, the upper-tropospheric q would be set by the initial condition rather than by surface
    evaporation lofted by convection (the classic single-column closure).

    MOISTURE CLOSURE (``closure``, CLEARSKY_SW_PLAN Stage 3) -- HOW a convecting pair redistributes its
    (conserved) water. The ARG default "q_homogenize" keeps the pre-Stage-3 gate byte-identical; the
    CONFIG/production default is "rh_detrain" (StageADConfig.convective_closure, canonical):
      * "q_homogenize" (arg default, byte-identical to the pre-Stage-3 scheme): SPECIFIC humidity is
        homogenized, q_k = q_{k-1} = (m_k q_k + m_{k-1} q_{k-1})/(m_k + m_{k-1}). Because qsat FALLS with
        height, uniform q makes RH = q/qsat RISE with height -> the constant-q upper-tropospheric SLUG
        that saturates only at the cold top (CLEARSKY_SW_FINDINGS root cause #1: it piles water aloft
        where it most inflates the greenhouse and deposits the near-IR SW heating high in the column. NB
        clear-sky SURFACE SW transmission is Beer-Lambert in TOTAL-path WVP and altitude-INDEPENDENT, so
        the slug's SW harm is the heating-DEPOSITION altitude, NOT surface darkening per unit WVP -- the
        Stage-3 finding; the looser "darkens the surface" reading is wrong).
      * "rh_detrain" (production default): redistribute the pair's water toward the Manabe fixed-RELATIVE-
        humidity target RH_t(sigma) = rh_top + (rh_surf - rh_top)*sigma (sigma = p/p_surf, DECREASING with
        height) -- the established Manabe-Wetherald fixed-RH profile, already the model's reference RH
        structure (the same RH(sigma) the RCE anchor seeds q with), so NOT a tuned number. (MODELING
        PREMISE, not a derived profile: fixed-RH is a whole-COLUMN closure used here as the intra-PAIR
        redistribution target; a real detraining plume sets free-tropospheric RH by entrainment +
        microphysics, not a prescribed linear-in-sigma shape -- the comparison that adopted it as the
        default is CLEARSKY_SW_FINDINGS.md.) With g_j =
        RH_t(sigma_j)*qsat(T_j, p_j), set q_j = A*g_j using the single normalization A = W/(m_k g_k +
        m_{k-1} g_{k-1}), W the pair water. This conserves the pair water EXACTLY (linear rescaling:
        m_k(A g_k)+m_{k-1}(A g_{k-1}) = A*denom = W) and gives RH = A*RH_t -- FALLING with height, NO
        constant-q slug. Iterated over a connected convecting segment it converges to a single A* over the
        segment (the same positive-averaging fixed point as q_homogenize, which is the g_j==1 special
        case), so column water is conserved to machine precision and the M2b.3 theorem still holds. If g
        underflows to 0 at an extreme-cold pair it falls back to the homogenized mean (still conserving).

    PERFORMANCE (issue #99): this sweep loop is the single largest cost of the coupled fleet step
    (~2/3 of step wall-clock at nx=48/nz=18, dominated by numpy dispatch overhead on the small per-pair
    arrays -- the Step-0 profile). The body below is the dispatch-lean restructure of the original:
    pair thicknesses hoisted out of the sweeps (``z`` is fixed for the call), the repeated ``Gamma*dz``
    product reused where the original's grouping allows, and the boolean fancy-index scatter/gathers
    replaced by whole-column ``np.where`` merges (selection of the SAME values -- stable columns are
    rewritten with their own bits). Every value chain is unchanged: same predicates, same multiply
    groupings (note ``mk1 * Gamma * dz`` keeps its left-to-right association, which differs from
    ``mk1 * Gdz`` in floating point), and the ``th.moist_adiabat_lapse_rate`` / ``th.qsat`` leaves are
    called on the SAME full-column arrays (transcendental inputs untouched -- subsetting them could
    flip SIMD/scalar code paths). BIT-IDENTICAL to the original, gated by
    ``tests/test_stage_ad_core.py`` against a verbatim reference copy. The sweep order itself is
    load-bearing (pair k-1 reads the layer pair k just wrote -- Gauss-Seidel), so the k loop cannot be
    batched without changing the algorithm; and in the coupled fleet regime EVERY pair is active in
    EVERY sweep up to the ``max_sweeps`` cap (measured, issue #99), so skip-stable-pair memoization
    would win nothing -- the remaining cost is the per-pair thermo-leaf calls themselves.

    FROZEN-GAMMA SWEEPS (``freeze_gamma``, issue #99 -- flag-gated, DEFAULT OFF = byte-identical): the
    remaining wall is exactly those leaf calls, and ~3/4 of the Gamma evaluations are re-derivations of
    a slowly-moving field: within one call, sweeps 2..max re-evaluate ``moist_adiabat_lapse_rate`` on
    temperatures the sweep itself only nudged. With ``freeze_gamma=True`` each pair's Gamma is computed
    LIVE in sweep 1 (Gauss-Seidel, exactly as the default) and HELD for the later sweeps -- the #86
    ``dt_rad`` pattern (hold a slowly-varying expensive field across fast iterations; the held value is
    the model's OWN lapse rate, one sweep old). The enthalpy and water conservation theorems above are
    Gamma-INDEPENDENT (the pairwise mix conserves both for ANY Gamma), so the flag changes only WHICH
    moist adiabat the pair relaxes toward in sweeps >= 2, by O(dGamma/dT * dT_sweep) -- and every call
    re-derives sweep-1 Gammas fresh, so nothing stales across steps. Measured at the fleet point:
    ``max_sweeps``x fewer Gamma evaluations (the cap's worth -> coupled step ~1.16x), and the ON
    trajectory CONVERGES to the OFF state (the early transient differs by ~2 K, then shrinks to
    < 0.1 K over 150 coupled steps -- gated by ``tests/test_stage_ad_core.py``; adoption needs the
    fleet A/B, per the #81/#85/#86 precedent).

    FIXED-TOPOLOGY (``fixed_topology``, GPU CUDA-graph capture prerequisite -- flag-gated, DEFAULT OFF =
    byte-identical): drop the two DATA-DEPENDENT early-exits so the launch sequence is CONSTANT every call
    (a captured graph needs a fixed op chain): (1) the inner ``if not unstable.any(): continue`` (the
    per-pair skip) and (2) the outer ``if not changed.any(): break`` (the sweep-count skip). Both are
    BYTE-IDENTICAL when taken vs not: the ``np.where(unstable, new, old)`` merges below select ``old`` for
    every stable column, so a skipped pair / a skipped sweep instead REWRITES each stable column with its
    OWN bits (same IEEE values, same memory) -- and an all-stable sweep leaves the field untouched, so the
    extra sweeps after convergence are a sequence of self-writes that change nothing. (The computed-but-
    discarded ``Tk_new`` / ``A`` etc. are selected away by the where elementwise, so even an inf/NaN in a
    would-be-skipped column cannot leak.) ON, the per-call cost is always ``max_sweeps`` x (nz-1) pairs;
    OFF, the original adaptive skips. Verified bit-identical by ``tests/test_stage_ad_core.py`` (the
    fixed-topology-vs-adaptive equivalence on a stable/inversion column, where the exits WOULD fire).

    Returns T_atm.

    DEVICE (Phase D): array-module-generic via ``get_xp`` (module ``xpm``) -- numpy in is BYTE-IDENTICAL
    (the verbatim ``np.where``/``np.zeros``), CuPy in runs the Gauss-Seidel sweep on the GPU. The k-loop
    stays SEQUENTIAL (pair k-1 reads what pair k wrote -- Gauss-Seidel); CUDA-graph capture (Phase E)
    amortizes the per-pair kernel launches, so no parallel-scan rewrite is needed (the Phase-D bar is
    correctness, not throughput of the slowest 1-D piece)."""
    xpm = get_xp(T_atm)
    nx, nz = T_atm.shape
    detrain = (closure == "rh_detrain") and (q is not None)
    if detrain:
        RHt = rh_top + (rh_surf - rh_top) * (th._F(p_mid, xpm) / C.P_SURF)   # Manabe RH target (falls w/ z); th._F follows the thermo dtype (fp32 gate)
    # FUSED DEVICE KERNEL (Phase E profile follow-up, flag-gated via ``device_kernel``; default None =
    # verbatim Python loop, BYTE-IDENTICAL). The Python loop is max_sweeps x (nz-1) SEQUENTIAL Gauss-Seidel
    # iterations, each issuing many tiny (nx,) CuPy kernels -> ~5700 launches that the CUDA graph cannot
    # collapse (84% of the captured step). The columns are INDEPENDENT, so one thread per column collapses
    # them into ONE kernel (``_gpu_kernels.convadj_cols``), mirroring the numba prange-over-columns kernel
    # exactly -> round-off-equivalent (the leaf transcendentals reassociate), gated by the climate-match.
    # The kernel needs T_atm/q IN PLACE and all float args in ITS dtype + C-contiguous; q must be present
    # (the kernel always writes q -- the convadjust call site always passes q on the device/capture path).
    if device_kernel is not None and q is not None:
        kdt = T_atm.dtype                                      # the kernel was built for T_atm's dtype
        T_c = xpm.ascontiguousarray(T_atm)
        q_c = xpm.ascontiguousarray(q)
        z_c = xpm.ascontiguousarray(z.astype(kdt, copy=False))
        # STATIC aux hoist (capture hardening, PR #198 review item 1): ``p_c``/``dm_c``/``RHt_c`` depend
        # only on call-fixed inputs (``p_mid``/``dm``/the Manabe RH target, all unchanged step-to-step), so
        # the shipped rebuild made a FRESH pooled (nz,) allocation each captured step -- the captured graph
        # then leaned on CuPy memory-pool address determinism to replay them. ``device_aux`` is the core's
        # WRITE-ONCE-IN-WARMUP persistent scratch (a dict keyed by C-precision; ``StageADCore._convadj_aux``):
        # the buffers are filled with the EXACT current expressions on the first (eager warmup) call and
        # reused thereafter, so the kernel reads the SAME bits -> climate-match exactly unchanged, and the
        # static aux no longer relies on pool determinism. ``device_aux=None`` (the host/parity-probe path)
        # falls back to the shipped fresh rebuild (byte-identical). ``z_c``/``T_c``/``q_c`` are
        # state-derived (fresh each step regardless -- ``_hydrostatic_z`` already allocates inside the
        # captured graph), so they stay covered by the >=60-step replay-vs-replay bit-identity proof.
        aux = device_aux.get(kdt) if device_aux is not None else None
        if aux is None or aux["RHt"].shape[0] != nz:           # first call (warmup) / shape change: fill once
            p_c = xpm.ascontiguousarray(p_mid.astype(kdt, copy=False)).copy()
            dm_c = xpm.ascontiguousarray(dm.astype(kdt, copy=False)).copy()
            RHt_c = (xpm.ascontiguousarray(RHt.astype(kdt, copy=False)).copy() if detrain
                     else xpm.zeros(nz, dtype=kdt))
            if device_aux is not None:
                device_aux[kdt] = {"p": p_c, "dm": dm_c, "RHt": RHt_c}
        else:                                                  # reuse the persistent buffers (write once)
            p_c, dm_c, RHt_c = aux["p"], aux["dm"], aux["RHt"]
        from . import _gpu_kernels as gk
        gk.convadj_columns_device(device_kernel, T_c, q_c, dm_c, z_c, p_c, RHt_c,
                                  int(max_sweeps), 1 if detrain else 0)
        # write results back IN PLACE (the call site holds T_atm/q as the live state buffers). When
        # ascontiguousarray returned the SAME object (already contiguous), these are no-op self-copies.
        if T_c is not T_atm:
            T_atm[...] = T_c
        if q_c is not q:
            q[...] = q_c
        return T_atm
    dz_pair = z[:, :-1] - z[:, 1:]                             # >0 pair thicknesses (z fixed for the call)
    Gammas = [None] * nz if freeze_gamma else None             # per-pair sweep-1 Gamma cache (flag ON only)
    for sweep in range(max_sweeps):
        changed = xpm.zeros(nx, bool)
        for k in range(nz - 1, 0, -1):                         # pair (k below/higher-p, k-1 above)
            Tk = T_atm[:, k]; Tk1 = T_atm[:, k - 1]            # views; all reads precede the writes below
            if freeze_gamma and sweep > 0:
                Gamma = Gammas[k]                              # held sweep-1 lapse rate (flag ON)
            else:
                Gamma = th.moist_adiabat_lapse_rate(Tk, p_mid[k])  # K/m (live, Gauss-Seidel)
                if freeze_gamma:
                    Gammas[k] = Gamma
            dz = dz_pair[:, k - 1]                              # == z[:, k-1] - z[:, k] (k-1 is higher)
            Gdz = Gamma * dz
            unstable = Tk1 < (Tk - Gdz) - 1e-9
            if not fixed_topology and not unstable.any():
                continue                                       # adaptive skip (OFF); ON: fall through (the
                #     where-merges below rewrite each stable column with its OWN bits -> byte-identical)
            mk, mk1 = dm[k], dm[k - 1]
            Ctot = mk * Tk + mk1 * Tk1
            Tk_new = (Ctot + mk1 * Gamma * dz) / (mk + mk1)    # keep the original (mk1*Gamma)*dz grouping
            Tk1_new = Tk_new - Gdz
            T_atm[:, k] = xpm.where(unstable, Tk_new, Tk)
            T_atm[:, k - 1] = xpm.where(unstable, Tk1_new, Tk1)
            if q is not None:
                qk = q[:, k]; qk1 = q[:, k - 1]
                if not detrain:                                # --- q_homogenize (DEFAULT, byte-identical) ---
                    q_mix = (mk * qk + mk1 * qk1) / (mk + mk1)                  # water-conserving homogenize
                    q[:, k] = xpm.where(unstable, q_mix, qk)
                    q[:, k - 1] = xpm.where(unstable, q_mix, qk1)
                else:                                          # --- rh_detrain (Stage 3): RH-shaped, conserving ---
                    W = mk * qk + mk1 * qk1                                     # pair water (conserved)
                    gk = RHt[k] * th.qsat(Tk_new, p_mid[k])                     # RH-target q at k (below)
                    gk1 = RHt[k - 1] * th.qsat(Tk1_new, p_mid[k - 1])           # ... at k-1 (above): both smaller
                    denom = mk * gk + mk1 * gk1
                    pos = denom > 0.0                                           # g underflows only at extreme cold
                    A = xpm.where(pos, W / xpm.where(pos, denom, 1.0), 0.0)     # normalization conserving W
                    homog = W / (mk + mk1)                                      # degenerate-cold fallback (homogenize)
                    q[:, k] = xpm.where(unstable, xpm.where(pos, A * gk, homog), qk)
                    q[:, k - 1] = xpm.where(unstable, xpm.where(pos, A * gk1, homog), qk1)
            changed |= unstable
        if not fixed_topology and not changed.any():
            break                                              # adaptive sweep-count (OFF); ON: always run
            #     max_sweeps -- extra all-stable sweeps are self-writes (no change) -> byte-identical
    return T_atm


# ======================================================================================
# COLUMN-PARALLEL convective adjustment (CPU-fuse, issue #99 / branch claude/cpu-fuse-multithread).
#   The numpy ``moist_convective_adjust`` above is vectorized ACROSS columns but its k-loop is a
#   Gauss-Seidel sweep (pair k-1 reads what pair k just wrote -> sequential in k). The nx COLUMNS are
#   INDEPENDENT, so a numba @njit(parallel=True) kernel with prange OVER COLUMNS threads it perfectly
#   AND fuses the per-pair thermo leaves (qsat/lapse-rate) into one kernel (no temporaries / numpy
#   dispatch). It reproduces the numpy reference's per-column Gauss-Seidel sweep EXACTLY (same predicate,
#   same enthalpy/water-conserving mix, same multiply groupings) to round-off (~1e-13 K, NOT bit-identical
#   -- the leaf transcendentals reassociate -> a different but equally-valid floating-point path).
#
#   This is a COMPARED-VARIANT speedup gated behind a default-OFF flag (COSMO_CONVADJUST_PARALLEL): the
#   default numpy path stays untouched and byte-identical. The ON path is wired at the core's call site
#   (``_thermo_substep``) and FALLS BACK to numpy whenever the kernel cannot reproduce the numpy path
#   bit-for-round-off: freeze_gamma (kernel always recomputes Gamma live), q is None (kernel always writes
#   q), and non-fp64 precision (the kernel's fp specialization would diverge past round-off). See
#   archive/_cpufuse_prod_RESULTS.md and tests/test_stage_ad_core.py (convadjust-parallel equivalence).
#
#   numba is imported LAZILY (only when the flag is ON), so a plain/default run never imports it and the
#   module has no hard numba dependency. The kernel + leaf helpers are built once on first use and cached
#   (numba @njit cache=True), so the JIT cost is paid once per process.
# ======================================================================================
_CONV_NUMBA_KERNEL = None   # built lazily on first ON call


def _build_conv_numba_kernel():
    """Compile (once) the prange-over-columns convective-adjustment kernel. Constants are pulled to
    module floats so the njit kernel closes over plain numbers (no attribute access in the hot loop)."""
    from numba import njit, prange   # lazy: only imported when COSMO_CONVADJUST_PARALLEL is ON

    T_FREEZE = float(C.T_FREEZE); EPS_MW = float(C.EPS_MW); L_VAP = float(C.L_VAP)
    R_DRY = float(C.R_DRY); CP_AIR = float(C.CP_AIR); G_ACCEL = float(C.G_ACCEL)

    @njit(inline="always", cache=True)
    def _esat_s(T):
        # mirrors thermo.esat: Tetens over water (Tc>=0) / over ice (Tc<0); same coefficients
        Tc = T - T_FREEZE
        if Tc >= 0.0:
            return 611.2 * np.exp(17.67 * Tc / (Tc + 243.5))
        return 611.2 * np.exp(21.87 * Tc / (Tc + 265.5))

    @njit(inline="always", cache=True)
    def _qsat_s(T, p):
        e = _esat_s(T)
        cap = 0.999 * p
        if e > cap:
            e = cap
        return EPS_MW * e / (p - (1.0 - EPS_MW) * e)

    @njit(inline="always", cache=True)
    def _gamma_s(T, p):
        # mirrors thermo.moist_adiabat_lapse_rate (K/m)
        qs = _qsat_s(T, p)
        num = 1.0 + L_VAP * qs / (R_DRY * T)
        den = 1.0 + L_VAP * L_VAP * qs * EPS_MW / (CP_AIR * R_DRY * T * T)
        return (G_ACCEL / CP_AIR) * num / den

    @njit(parallel=True, cache=True)
    def _conv_adjust_numba(T_atm, q, dm, z, p_mid, max_sweeps, RHt, detrain):
        nx, nz = T_atm.shape
        for i in prange(nx):                                   # COLUMNS ARE INDEPENDENT -> parallel
            for sweep in range(max_sweeps):
                changed = False
                for k in range(nz - 1, 0, -1):                 # top->surface Gauss-Seidel (sequential in k)
                    Tk = T_atm[i, k]; Tk1 = T_atm[i, k - 1]
                    Gamma = _gamma_s(Tk, p_mid[k])
                    dz = z[i, k - 1] - z[i, k]
                    Gdz = Gamma * dz
                    if Tk1 < (Tk - Gdz) - 1e-9:
                        mk = dm[k]; mk1 = dm[k - 1]
                        Ctot = mk * Tk + mk1 * Tk1
                        Tk_new = (Ctot + mk1 * Gamma * dz) / (mk + mk1)   # keep (mk1*Gamma)*dz grouping
                        Tk1_new = Tk_new - Gdz
                        T_atm[i, k] = Tk_new
                        T_atm[i, k - 1] = Tk1_new
                        qk = q[i, k]; qk1 = q[i, k - 1]
                        if not detrain:                        # q_homogenize
                            q_mix = (mk * qk + mk1 * qk1) / (mk + mk1)
                            q[i, k] = q_mix
                            q[i, k - 1] = q_mix
                        else:                                  # rh_detrain (production default)
                            W = mk * qk + mk1 * qk1
                            gk = RHt[k] * _qsat_s(Tk_new, p_mid[k])
                            gk1 = RHt[k - 1] * _qsat_s(Tk1_new, p_mid[k - 1])
                            denom = mk * gk + mk1 * gk1
                            if denom > 0.0:
                                A = W / denom
                                q[i, k] = A * gk
                                q[i, k - 1] = A * gk1
                            else:
                                homog = W / (mk + mk1)
                                q[i, k] = homog
                                q[i, k - 1] = homog
                        changed = True
                if not changed:
                    break
        return T_atm

    return _conv_adjust_numba


def moist_convective_adjust_parallel(T_atm, dm, z, p_mid, max_sweeps, q, closure, rh_surf, rh_top):
    """Column-parallel (numba prange) drop-in for ``moist_convective_adjust``'s ON path. Operates IN PLACE
    on ``T_atm`` (nx, nz) and ``q`` (nx, nz). REQUIRES q is not None and fp64 (the caller guards this and
    falls back to numpy otherwise). Reproduces the numpy reference to round-off (~1e-13 K), not bit-for-bit.
    ``z`` (nx, nz), ``dm`` (nz,), ``p_mid`` (nz,). Both closures (rh_detrain / q_homogenize) covered."""
    global _CONV_NUMBA_KERNEL
    if _CONV_NUMBA_KERNEL is None:
        _CONV_NUMBA_KERNEL = _build_conv_numba_kernel()
    detrain = (closure == "rh_detrain")
    RHt = rh_top + (rh_surf - rh_top) * (np.asarray(p_mid, dtype=np.float64) / C.P_SURF)
    # numba needs contiguous fp64 arrays; T_atm/z/q come from the core as fp64 C-contiguous already, but
    # ascontiguousarray is a no-op cost when so and a safety net when a view is passed.
    _CONV_NUMBA_KERNEL(T_atm, q, np.ascontiguousarray(dm, dtype=np.float64),
                       np.ascontiguousarray(z, dtype=np.float64),
                       np.ascontiguousarray(p_mid, dtype=np.float64),
                       int(max_sweeps), RHt, detrain)
    return T_atm


# ======================================================================================
# ONE-PASS SEGMENT MIXER -- the CONVERGED convective adjustment (ROADMAP PHY-17 step 4).
# --------------------------------------------------------------------------------------
#   ``moist_convective_adjust`` above relaxes a column toward the moist adiabat by PAIRWISE
#   Gauss-Seidel sweeps, and production caps that at ``conv_max_sweeps`` = 4. On the canonical
#   144-level boundary-layer-refined grid the cap is far short of the fixed point: one call on the
#   run17 checkpoint's map-band columns leaves 35.8 / 29.7 / 18.1 / 3.6 / 0.0 K of a 43.1 K
#   potential-temperature deficit after 4 / 16 / 64 / 256 / 1,024 sweeps (``flatworld/OPEN_ISSUES.md``,
#   PHY-17), because a pairwise sweep propagates information one layer pair per sweep and the
#   convergence is therefore quadratic in the depth of the unstable layer, in LEVELS.
#
#   This routine reaches the SAME fixed point in a bounded number of whole-column passes. It is a
#   different ALGORITHM for the same operator, not a different physics: the predicate, the moist
#   adiabat, the enthalpy theorem and the ``rh_detrain`` water target are the sweep's, verbatim.
#
#   DEFAULT OFF (``StageADConfig.conv_segment_mixer`` / ``COSMO_CONV_SEGMENT_MIXER``); the sweep path
#   above is untouched, so an OFF run is BYTE-IDENTICAL.
# ======================================================================================
class SegmentMixTruncationWarning(RuntimeWarning):
    """Columns came back from ``moist_convective_segment_mix`` with their active set still growing at
    ``max_passes`` -- a TRUNCATED adjustment, not a converged one.

    THE MESSAGE TEXT IS FIXED AND THE NUMBERS LIVE ON THE INSTANCE, deliberately. Python's default
    warning filter keys its once-per-location suppression on the message TEXT, so a count formatted into
    the string defeats the dedupe entirely -- and the production caller runs this routine 30 thermo
    substeps per step, so a persistently truncating run would emit thousands of near-identical lines and
    train the reader to filter them out, which is the failure this diagnostic exists to prevent. Read
    ``n_truncated`` / ``n_columns`` / ``max_passes`` off the warning object, or ``passes_out`` for the
    per-column detail (negative = truncated)."""
    _TEXT = ("moist_convective_segment_mix: columns hit max_passes with their active set still growing "
             "-- the adjustment is TRUNCATED, not converged. Counts are on this warning object "
             "(n_truncated, n_columns, max_passes) and per column in passes_out. Raise "
             "conv_segment_max_passes and re-measure.")

    def __init__(self, n_truncated, n_columns, max_passes):
        self.n_truncated = int(n_truncated)
        self.n_columns = int(n_columns)
        self.max_passes = int(max_passes)
        super().__init__(self._TEXT)


_SEGMIX_TOL = 1e-9        # K -- the pairwise instability tolerance, VERBATIM from moist_convective_adjust
_SEGMIX_SETTLED_ULP = 4.0  # units in the last place of the STATE's dtype: below this a pass has stopped
#   moving the profile *as far as its own arithmetic can tell*. Not a physical tolerance and not tunable
#   in kelvin -- it is 4 x np.finfo(dtype).eps x |T|, so fp64 reads ~3e-13 K at 300 K (three orders below
#   the 1e-9 K instability tolerance, hence inert: rule (i) always fires first) and fp32 reads ~1.4e-4 K
#   (above the ~2e-5 K single-precision chatter, hence the single-precision exit). See stopping rule (ii).


def moist_convective_segment_mix(T_atm, dm, z, p_mid, q=None, closure="q_homogenize",
                                 rh_surf=0.8, rh_top=0.2, max_passes=16, gamma_iters=8,
                                 fixed_topology=False, device_kernel=None, device_aux=None,
                                 passes_out=None):
    """CONVERGED moist convective adjustment in a bounded number of passes (ROADMAP PHY-17 step 4):
    find each contiguous statically unstable SEGMENT of a column and mix the WHOLE segment to the moist
    adiabat in one operation, instead of relaxing pair by pair. Operates IN PLACE on ``T_atm`` (nx, nz)
    and (optionally) ``q`` (nx, nz), TOP(0) -> SURFACE(nz-1); ``z`` (nx, nz) layer-center heights,
    ``dm`` (nz,) layer masses, ``p_mid`` (nz,) layer pressures. Returns T_atm.

    THE STATE IT COMPUTES. The sweep's fixed point is a partition of the column into maximal MIXED
    BLOCKS. Inside a block the pairwise predicate is exactly marginal,

        T_{k-1} = T_k - Gamma(T_k, p_k) * (z_{k-1} - z_k)                     [the moist adiabat]

    with Gamma = ``th.moist_adiabat_lapse_rate`` evaluated at the LOWER layer's OWN (converged)
    temperature -- the same place the sweep evaluates it -- and the block's enthalpy Sum_j m_j T_j is
    the enthalpy it started with. Between blocks the profile is strictly stable. This routine solves
    that system directly.

    THE ALGORITHM, per column (identical in the numpy body below, in the CUDA kernel
    ``_gpu_kernels.segmix_cols``, and in the scalar reference ``_ref_segment_mix`` the tests gate it
    against).

      * A PERSISTENT set of ACTIVE pairs, grown monotonically. Pass 1 flags every pair the sweep's own
        predicate calls unstable, ``T_{k-1} < T_k - Gamma*dz - 1e-9``. A pair, once active, STAYS
        active. That is what makes one pass equivalent to a converged sweep rather than to four of
        them: mixing a block warms its top and cools its bottom, which leaves the block's own interior
        pairs exactly MARGINAL -- a fresh predicate scan would read them stable and forget the block,
        and the mixer would then crawl upward one pair per pass exactly as Gauss-Seidel does. (Both
        schemes only ever GROW their mixed set: a neighbour's mix raises the upper member of the pair
        below it and lowers the lower member of the pair above it, and both moves are destabilizing.)
      * MIX every maximal run of active pairs. A run of active pairs k = j..b spans layers j-1..b.
      * RESCAN. If the mix left any pair unstable -- the pair just above a block's warmed top or just
        below its cooled bottom, newly destabilized; or an interior pair still off the adiabat by more
        than the tolerance because the Gamma iteration below has not fully converged -- add it to the
        active set and mix the (enlarged) block again. STOPPING RULE, per column, either of:
          (i) NO pair of the column is unstable -- the sweep's OWN stopping rule, so the mixer and a
              converged sweep stop on the same test; or
          (ii) the ACTIVE SET HAS NOT GROWN for two consecutive passes -- the block partition is final --
              AND the last Gamma iterate moved the profile by less than 4 units in the last place of the
              state's dtype. This second rule is what makes the SINGLE-PRECISION path terminate: the
              ``-1e-9 K`` tolerance in (i) is a NO-OP in fp32 (the unit in the last place at ~300 K is
              ~3e-5 K), so the predicate chatters on the exactly-marginal interior pairs forever and (i)
              alone runs every column to ``max_passes``, at any ``max_passes``. Both conjuncts were put
              there by measurement: without the second, a still-converging fp64 column exits at a 6.8e-8 K
              residual instead of 2.3e-10 K; with a BITWISE second conjunct instead of the ULP one,
              nothing ever exits at fp32, because the iteration limit-cycles in the last place. In fp64
              rule (ii) is inert -- 4 ULP is ~3e-13 K at 300 K, three orders below (i)'s tolerance, so
              (i) always fires first.
        Bounded by ``max_passes``; the measured requirement, per region of the canonical world, is in
        ``PHY17_SEGMENT_MIXER_RESULTS.md``. EXHAUSTING THE BOUND WITH THE ACTIVE SET STILL GROWING is a
        genuinely truncated adjustment (the state returned is NOT the fixed point) and is REPORTED: the
        numpy path raises a ``SegmentMixTruncationWarning`` (a ``RuntimeWarning``) whose counts ride on
        the OBJECT rather than in its text -- so Python's once-per-location dedupe actually collapses it
        across the 30 thermo substeps of a step -- and both paths record it per column in ``passes_out``
        (below). Exhausting the bound under (ii) cannot happen: (ii) is itself an exit.

    MIXING ONE BLOCK j..b. Write T_i = T_b + d_i with d_b = 0 and d_{i-1} = d_i - Gamma_i*dz_i, so the
    profile shape is fixed by the Gammas and the single unknown T_b. Enthalpy conservation then gives
    T_b in CLOSED FORM,

        T_b = (H - Sum_i m_i d_i) / Sum_i m_i,        H = the block's pre-mix Sum_i m_i T_i,

    which conserves Sum_i m_i T_i EXACTLY (to round-off) for ANY Gammas -- the same Gamma-INDEPENDENT
    enthalpy theorem the pairwise mix has, and for the same reason: the adiabat offsets cancel.
    Gamma itself depends on the temperature it is evaluated at, so the profile is the fixed point of a
    short iteration: take the Gammas from the current T, solve the line above, re-evaluate the Gammas at
    the new T, re-solve, ``gamma_iters`` times. MEASURED contraction on the canonical probe columns:
    2e-3 to 5e-2 per iterate. (The textbook estimate -- dGamma/dT ~ -1e-4 K/m per K times the block's
    mass-weighted depth, ~1/25 -- is an UPPER BOUND the iteration beats comfortably, because iterate 0
    already lands close.) Enthalpy is conserved at EVERY iterate, converged or not.

    WATER (``q``, optional, mixed in place; ``closure``). Exactly the target
    ``moist_convective_adjust``'s docstring derives as the pairwise scheme's own segment-wise fixed
    point: with g_j = RH_t(sigma_j) * qsat(T_j^new, p_j) and W the block's pre-mix water Sum_j m_j q_j,

        q_j = A* g_j,      A* = W / Sum_j m_j g_j                              ["rh_detrain"]

    -- ONE normalization over the connected block, so the block water is conserved EXACTLY (a pure
    linear rescaling) and RH = A*.RH_t falls with height. ``closure="q_homogenize"`` is the g_j == 1
    special case, q_j = W / Sum_j m_j. If the denominator underflows to zero -- every layer of the block
    so cold that qsat is EXACTLY zero -- it falls back to the homogenized mean, still conserving.
      WHERE THAT IS, MEASURED (not the same answer at the two precisions, and the difference matters
    because the device path is the single-precision one): in fp64 qsat is exactly 0 at and below 15 K,
    which no realistic column reaches. In binary32 esat is exactly 0 at and below 53.7 K and SUBNORMAL
    (so flushed to zero under nvcc's ``-ftz=true``) up to 57.8 K -- and the canonical run17 state's
    coldest air cell is 48.45 K. So on the fp32 kernel this branch is LIVE on today's canonical state,
    at the cold top of the far field. The water involved is vanishing, but the branch must conserve, and
    ``test_segment_mixer_cold_block_takes_the_rh_detrain_underflow_fallback`` gates that it does.
      DIFFERENCE FROM THE SWEEP, in that one regime, DOCUMENTED NOT CHASED: iterate the pairwise
    rh_detrain update across a block that contains a cold slab of all-g==0 layers and the slab acts as a
    ONE-WAY VALVE in both directions -- at the pair (g>0 above, g==0 below) all the pair's water moves
    up, at the pair (g==0 above, g>0 below) all of it moves down. The sweep's limit therefore splits the
    block's water into two independent pools with DIFFERENT normalizations either side of the slab,
    while the single block-wide A* redistributes across it. Both schemes give the slab itself q = 0, and
    both conserve the block's water exactly; the difference is O(q) in the layers AROUND the slab.

    HEIGHTS ARE FROZEN for the call (``z`` is read once and never rehydrated after mixing), as in
    ``moist_convective_adjust``.

    FIXED-TOPOLOGY (``fixed_topology``, the CUDA-graph-capture prerequisite; default OFF = the adaptive
    exit): drop the data-dependent ``break`` so the launch sequence is CONSTANT every call. BIT-IDENTICAL
    either way -- a pass over a column with no unstable pair left selects that column's OWN bits through
    the ``np.where`` merges below, so the extra passes are self-writes. (The device kernel's per-column
    early-out is the same value-neutral shape ``convadj_cols`` uses.)

    DEVICE (``device_kernel``): the fused one-thread-per-column ``_gpu_kernels.segmix_cols`` RawKernel,
    mirroring this body op-for-op -> round-off-equivalent, not bit-identical (the leaf transcendentals
    reassociate). The kernel always writes ``q``, so it requires q is not None -- the core's call site
    guarantees it. ``device_aux`` is the core's persistent static-aux dict (p/dm/RHt), the
    ``moist_convective_adjust`` capture-hardening pattern.

    ``passes_out`` (optional (nx,) int, written in place, both paths; must be a DEVICE array on the
    device path, since the copy comes off the device buffer): per column, the number of passes it used --
    0 is legitimate (nothing to mix) and only a NEGATIVE value means truncation, i.e. it hit
    ``max_passes`` with its active set still growing.
    On the device this is the core's persistent aux buffer -- allocated once in warmup, so reading it is
    a batch-boundary diagnostic, never a per-step sync, and the launch sequence is unchanged under
    CUDA-graph capture.

    HOST BODY IS NUMPY-ONLY, deliberately: the segment decomposition uses ``ufunc.accumulate``, which
    CuPy does not implement. A device array without a ``device_kernel`` raises rather than silently
    falling back (the core's dispatch builds the kernel whenever the state is on device). Its internal
    arithmetic (the block sums, the offsets, the enthalpy solve) is float64 while the thermo leaves
    follow the usual dtype gate, and the result is written back into ``T_atm``/``q`` at their own dtype:
    the host body is the REFERENCE path, and the genuine single-precision path is the device kernel."""
    xpm = get_xp(T_atm)
    nx, nz = T_atm.shape
    detrain = (closure == "rh_detrain") and (q is not None)
    if detrain:
        RHt = rh_top + (rh_surf - rh_top) * (th._F(p_mid, xpm) / C.P_SURF)   # Manabe RH target (falls w/ z)
    if device_kernel is not None and q is not None:
        # --- FUSED DEVICE KERNEL: one thread per column, the whole pass loop in-thread (the
        #     convadj_cols contract: T_atm/q IN PLACE, every float arg in the kernel's dtype and
        #     C-contiguous, the static aux hoisted into the core's write-once-in-warmup buffers). ---
        kdt = T_atm.dtype                                      # the kernel was built for T_atm's dtype
        T_c = xpm.ascontiguousarray(T_atm)
        q_c = xpm.ascontiguousarray(q)
        z_c = xpm.ascontiguousarray(z.astype(kdt, copy=False))
        aux = device_aux.get(kdt) if device_aux is not None else None
        if aux is None or aux["RHt"].shape[0] != nz or aux["status"].shape[0] != nx:
            p_c = xpm.ascontiguousarray(p_mid.astype(kdt, copy=False)).copy()   # first call / shape change
            dm_c = xpm.ascontiguousarray(dm.astype(kdt, copy=False)).copy()
            RHt_c = (xpm.ascontiguousarray(RHt.astype(kdt, copy=False)).copy() if detrain
                     else xpm.zeros(nz, dtype=kdt))
            st_c = xpm.zeros(nx, dtype=np.int32)               # per-column passes-used (see passes_out)
            if device_aux is not None:
                device_aux[kdt] = {"p": p_c, "dm": dm_c, "RHt": RHt_c, "status": st_c}
        else:
            p_c, dm_c, RHt_c, st_c = aux["p"], aux["dm"], aux["RHt"], aux["status"]
        from . import _gpu_kernels as gk
        # (``fixed_topology`` is not passed: the device path is ONE kernel launch either way, and the
        #  kernel's per-column early-out is value-neutral, so there is no host-side topology to fix.)
        gk.segmix_columns_device(device_kernel, T_c, q_c, dm_c, z_c, p_c, RHt_c,
                                 int(max_passes), int(gamma_iters), 1 if detrain else 0, st_c)
        if passes_out is not None:
            passes_out[...] = st_c                             # the caller's own buffer (a device copy)
        if T_c is not T_atm:
            T_atm[...] = T_c
        if q_c is not q:
            q[...] = q_c
        return T_atm
    if not isinstance(T_atm, np.ndarray):
        raise TypeError("moist_convective_segment_mix's host body is numpy-only (the segment "
                        "decomposition uses ufunc.accumulate, absent in CuPy); pass device_kernel "
                        "for a device array.")
    if int(gamma_iters) < 1:
        raise ValueError(f"gamma_iters must be >= 1 (the profile solve runs at least once); got {gamma_iters}")
    if int(max_passes) < 1:
        # 0 would silently turn the convective adjustment OFF (reachable via COSMO_CONV_SEGMENT_MAX_PASSES=0),
        # which is exactly the silent-truncation failure this routine exists to remove. Refuse it.
        raise ValueError(f"max_passes must be >= 1 (the scan-and-mix pass runs at least once); got {max_passes}")
    p_b = np.asarray(p_mid, float)[None, :]                    # (1, nz) layer pressures
    dm_b = np.asarray(dm, float)[None, :]                      # (1, nz) layer masses
    dm_f = np.broadcast_to(dm_b, (nx, nz))
    dzk = np.zeros((nx, nz), float)
    dzk[:, 1:] = z[:, :-1] - z[:, 1:]                          # dzk[:, k] = z[:, k-1] - z[:, k] > 0 (pair k)
    kidx = np.arange(nz)[None, :]
    act = np.zeros((nx, nz), bool)                             # act[:, k]: pair k is in the mixed set
    hi_idx = starts = ordinal = None
    eps_T = float(np.finfo(T_atm.dtype).eps)                   # the STATE's granularity (see rule (ii))
    n_act_prev = np.full(nx, -1, np.int64)                     # active-pair count at the previous pass
    stall = np.zeros(nx, np.int64)                             # consecutive passes with no NEW active pair
    done = np.zeros(nx, bool)                                  # exited by (i) no unstable pair or (ii) stall
    used = np.zeros(nx, np.int64)                              # passes each column used (see passes_out)

    def _gsum(x):
        """Per-block sum of ``x`` (nx, nz), broadcast back to every layer of its block.

        ``np.add.reduceat`` over the flattened field, NOT a difference of whole-column cumulative sums:
        the cumulative form cancels catastrophically where a block's sum is far smaller than the column's
        running total, which is exactly the dry far field (at latitude 85.6 deg of the canonical run a
        block holding 5.6e-14 kg/m^2 of water sits under a column total of 1.4e-3, and the cumulative
        form got that block 1.5e-6 wrong in relative terms -> a 9e-9 relative column-water leak). A
        ``reduceat`` sums each block on its OWN, so nothing outside the block enters its arithmetic --
        the same footing the scalar reference and the device kernel have, and it keeps the water
        conservation at round-off everywhere. (Every column's layer 0 starts a block, so no block ever
        straddles two columns in the flattened order.)"""
        return np.add.reduceat(x.ravel(), starts)[ordinal].reshape(nx, nz)

    for _pass in range(int(max_passes)):
        # --- SCAN: the sweep's own predicate on the current column, Gamma at the lower layer ---
        Gam = th.moist_adiabat_lapse_rate(T_atm, p_b)          # (nx, nz) K/m, layer i as a pair's LOWER member
        unst = np.zeros((nx, nz), bool)
        unst[:, 1:] = T_atm[:, :-1] < (T_atm[:, 1:] - Gam[:, 1:] * dzk[:, 1:]) - _SEGMIX_TOL
        act |= unst & ~done[:, None]                           # MONOTONE: a pair, once active, stays active
        n_act = act.sum(axis=1)                                # for stopping rule (ii), applied after the mix
        stall = np.where(n_act > n_act_prev, 0, stall + 1)
        n_act_prev = n_act
        newly = ~done & ~unst.any(axis=1)                      # STOPPING RULE (i): the sweep's own test
        used[newly] = _pass                                    # a converged column did no work this pass
        done |= newly
        work = ~done                                           # (nx,) columns still being mixed
        if not fixed_topology and not work.any():
            break                                              # converged (ON: run every pass; self-writes)
        # --- BLOCK BOUNDS: maximal runs of active pairs. Pair k joins layers k-1 and k, so layer i
        #     starts a block iff pair i is inactive and ends one iff pair i+1 is. ---
        brk = ~act                                             # brk[:, 0] is True (pair 0 is the dummy)
        lo_idx = np.maximum.accumulate(np.where(brk, kidx, 0), axis=1)
        brk_end = np.empty((nx, nz), bool)
        brk_end[:, :-1] = ~act[:, 1:]
        brk_end[:, -1] = True
        hi_idx = np.minimum.accumulate(np.where(brk_end, kidx, nz - 1)[:, ::-1], axis=1)[:, ::-1]
        in_seg = hi_idx > lo_idx                               # a 1-layer block is not a mixed block
        flat_start = brk.ravel()                               # block starts in the flattened (nx*nz) order
        starts = np.flatnonzero(flat_start)
        ordinal = np.cumsum(flat_start) - 1                    # each layer -> its block's ordinal
        M = _gsum(dm_f)                                        # block mass
        H = _gsum(dm_b * T_atm)                                # block enthalpy/cp -- the CONSERVED target
        # --- PROFILE: the Gamma fixed point, enthalpy-conserving at every iterate ---
        T_new = T_prev_it = T_atm
        for _it in range(int(gamma_iters)):
            T_prev_it = T_new                                  # for rule (ii): has the profile solve settled?
            if _it:
                Gam = th.moist_adiabat_lapse_rate(T_new, p_b)  # iterate 0 reuses the scan's Gamma
            # The offsets DO come from a whole-column cumulative sum (unlike the block sums above): the
            # quantity differenced is a temperature drop, O(100 K) against a column total of O(1000 K),
            # so the cancellation costs ~1e-13 K -- four orders inside the 1e-9 K tolerance the loop
            # stops on, and it cannot leak conservation (T_bot below absorbs any d exactly).
            E = np.cumsum(Gam * dzk, axis=1)
            d = E - np.take_along_axis(E, hi_idx, 1)           # d_i = -sum_{l=i+1..b} Gamma_l dz_l (d_b = 0)
            T_new = np.where(in_seg, (H - _gsum(dm_b * d)) / M + d, T_atm)
        sel = in_seg & work[:, None]                           # converged columns keep their OWN bits
        settled = ~np.any(sel & (np.abs(T_new - T_prev_it)     # rule (ii): the profile solve has settled
                                 > _SEGMIX_SETTLED_ULP * eps_T * np.abs(T_new)), axis=1)
        if q is not None:
            W = _gsum(dm_b * q)                                # block water (pre-mix) -- CONSERVED
            homog = W / M
            if detrain:
                g = RHt[None, :] * th.qsat(T_new, p_b)         # RH-target q at the NEW temperatures
                denom = _gsum(dm_b * g)
                pos = denom > 0.0                              # g underflows only at extreme cold
                A = np.where(pos, W / np.where(pos, denom, 1.0), 0.0)
                q[...] = np.where(sel, np.where(pos, A * g, homog), q)
            else:
                q[...] = np.where(sel, homog, q)
        T_atm[...] = np.where(sel, T_new, T_atm)
        used[work] = _pass + 1
        # --- STOPPING RULE (ii): the active set has not grown for two consecutive passes (the block
        #     PARTITION is final) AND the last Gamma iterate moved the profile by less than 4 units in
        #     the last place of the state's own dtype (the PROFILE has settled as far as this arithmetic
        #     can tell). Both conjuncts are needed. The second is MEASURED, not precautionary: with the
        #     growth test alone, column 2 of the synthetic gate case exits in fp64 at a 6.8e-8 K residual
        #     instead of the 2.3e-10 K it reaches when allowed to run -- its partition was final while
        #     the Gamma fixed point was still converging. And the bar has to be a ULP count, not a
        #     bitwise "unchanged": at single precision the iteration LIMIT-CYCLES in the last place and
        #     never repeats itself exactly, so a bitwise test fires never and every column runs the bound
        #     at any bound (measured both ways). Settled profile + still-chattering predicate, because
        #     the 1e-9 K tolerance sits far below the arithmetic's resolution, IS the saturated state,
        #     and exiting it is what rule (ii) is for.
        done |= work & (stall >= 2) & settled
    # --- TRUNCATION DIAGNOSTIC. A column still live here exhausted ``max_passes`` with its active set
    #     STILL GROWING -- the block partition had not settled, so the state returned is NOT the fixed
    #     point (a built 60-level case returns 25 K from converged, 22 K still unstable, at the shipped
    #     bound of 48, and converges at 96). The predicate is deliberately "still growing", NOT "bound
    #     hit": under stopping rule (ii) every fp32 column exits AT the bound's neighbourhood with a
    #     final partition, so a bound-hit warning would fire on every single-precision call and train
    #     the reader to ignore it. Silence here is the defect PHY-17 exists to record (conv_max_sweeps=4
    #     shipped in June 2026 and went unnoticed until 2026-09-14); this routine does not repeat it.
    truncated = ~done
    if truncated.any():
        # ONE warning per call site, with a FIXED message: see SegmentMixTruncationWarning for why the
        # counts ride on the object instead of in the text.
        warnings.warn(SegmentMixTruncationWarning(int(truncated.sum()), nx, int(max_passes)),
                      stacklevel=2)
    if passes_out is not None:
        passes_out[...] = np.where(truncated, -used, used)     # negative = truncated at the bound
    return T_atm


# ======================================================================================
# CPU-fuse #4 (thermo_other): column-PARALLEL (numba prange) MICROPHYSICS + BELOW-CLOUD REEVAP.
# --------------------------------------------------------------------------------------
#   After convadjust+radiation+advect were threaded, ``thermo_other`` (the residual column physics inside
#   ``_thermo_substep``) is the dominant un-accelerated numpy limb. Direct sub-profiling
#   (cosmology_models/archive/_cpufuse_thermo_subprofile.py): the SEB is (nx,)-cheap (~0.8 ms), ``_hydrostatic_z`` ~3 ms;
#   the two real poles are ``_prognostic_microphysics`` and ``_below_cloud_reevap`` -- both (nx,nz) numpy that
#   the columns make INDEPENDENT. ``_below_cloud_reevap`` is called INSIDE the microphysics (the falling-rain
#   sweep), so they FUSE into ONE prange-over-columns kernel: each thread runs one column's
#   condensation->autoconv->ceiling->downward-reevap pass, computing ``qsat`` ONCE (the numpy path computes it
#   twice -- micro + reevap -- so the fusion is a real win beyond the threading).
#
#   This is a COMPARED-VARIANT speedup gated behind a default-OFF flag (COSMO_THERMO_OTHER_PARALLEL): the
#   default numpy ``_prognostic_microphysics`` / ``_below_cloud_reevap`` stay byte-identical. It reproduces the
#   numpy reference to ROUND-OFF (~1e-13, NOT bit-identical -- ``esat``'s ``exp`` reassociates between numpy and
#   numba's libm). It covers ONLY the CANONICAL prognostic-cloud combination (qc_pdf_cloud, qc_dqsat_adjust,
#   qc_ice_phase, qc_retained_ceiling, qc_radiative_cap_decouple, qc_reevaporation, kessler autoconv,
#   precip_reevap, equator_norain_cells==0, fp64); the caller FALLS BACK to the numpy path for any other config
#   (so e.g. the single-pass non-prognostic core and the diagnostic suppression tests are untouched).
#
#   numba is imported LAZILY (only when the flag is ON); the kernel is built once and cached (@njit cache=True).
# ======================================================================================
_MICRO_NUMBA_KERNEL = None   # built lazily on first ON call


def _build_micro_numba_kernel():
    """Compile (once) the fused prange-over-columns microphysics+reevap kernel for the CANONICAL config.
    Mirrors ``_prognostic_microphysics`` + ``_below_cloud_reevap`` exactly (same predicates, same multiply
    groupings) to round-off. Constants close over plain module floats (no attribute access in the hot loop)."""
    from numba import njit, prange   # lazy: only imported when COSMO_THERMO_OTHER_PARALLEL is ON

    T_FREEZE = float(C.T_FREEZE); EPS_MW = float(C.EPS_MW)
    L_VAP = float(C.L_VAP); L_SUB = float(C.L_SUB); L_FUS = float(C.L_FUS)
    R_DRY = float(C.R_DRY); CP_AIR = float(C.CP_AIR)
    IWC_PLAUSIBLE = float(rad.CLOUD_IWC_PLAUSIBLE); LWC_PLAUSIBLE = float(rad.CLOUD_LWC_PLAUSIBLE)
    PDF_WIDTH_FLOOR = float(rad.CLOUD_PDF_WIDTH_FLOOR)

    @njit(inline="always", cache=True)
    def _esat_s(T):
        Tc = T - T_FREEZE
        if Tc >= 0.0:
            return 611.2 * np.exp(17.67 * Tc / (Tc + 243.5))
        return 611.2 * np.exp(21.87 * Tc / (Tc + 265.5))

    @njit(inline="always", cache=True)
    def _qsat_s(T, p):
        e = _esat_s(T)
        cap = 0.999 * p
        if e > cap:
            e = cap
        return EPS_MW * e / (p - (1.0 - EPS_MW) * e)

    @njit(inline="always", cache=True)
    def _desat_dT_s(T):
        # thermo.desat_dT: exact derivative of the Bolton/ice esat (consistent with _esat_s)
        Tc = T - T_FREEZE
        e = _esat_s(T)
        if Tc >= 0.0:
            return e * 17.67 * 243.5 / ((Tc + 243.5) * (Tc + 243.5))
        return e * 21.87 * 265.5 / ((Tc + 265.5) * (Tc + 265.5))

    @njit(inline="always", cache=True)
    def _dqsat_dT_s(T, p):
        # thermo.dqsat_dT: 0 in the supercritical (clamped) branch, else eps p de/dT / denom^2
        e = _esat_s(T)
        cap = 0.999 * p
        if e >= cap:
            return 0.0
        denom = p - (1.0 - EPS_MW) * e
        return EPS_MW * p * _desat_dT_s(T) / (denom * denom)

    @njit(inline="always", cache=True)
    def _T_boil_s(p):
        # thermo.T_boil: closed-form inverse of the warm-branch Bolton esat at e = 0.999 p, so
        # T >= _T_boil_s(p) <=> the boiling mask _esat_s(T) >= 0.999 p (warm branch; #294)
        y = np.log(0.999 * p / 611.2)
        return T_FREEZE + 243.5 * y / (17.67 - y)

    @njit(parallel=True, cache=True)
    def _micro_numba(T_atm, q, q_c, dm, p_mid, dt, rh_crit, autoconv_tau, autoconv_crit,
                     autoconv_tau_ice, autoconv_crit_ice, cloud_f_min, reevap_rate,
                     cond_tau, warm_factor,          # #280 finite-tau cond (0=off) / #237 salt (nx,) warm-autoconv
                     virga_flash_boil, boil_suppress,   # #294 boiling-layer ints (0/0 = shipped arithmetic)
                     src_ice, precip_qc_buf, precip_out):
        # ONE column per prange iteration: the full canonical microphysics (_prognostic_microphysics) FUSED
        # with the below-cloud reevap sweep (_below_cloud_reevap, precip_reevap ON). Writes T_atm, q, q_c IN
        # PLACE and the per-column surface precip into precip_out. ``src_ice``/``precip_qc_buf`` are (nx,nz)
        # scratch (the production ice mask + the per-layer precip source carried from phase 1 to phase 2).
        # (nx,nz) C-contiguous fp64; src_ice bool.
        nx, nz = T_atm.shape
        for i in prange(nx):                                   # COLUMNS INDEPENDENT -> parallel
            # ---- PHASE 1: saturation-adjustment microphysics (canonical: pdf source, dqsat denom, ice phase,
            #               kessler autoconv, decoupled retained ceiling, reevaporation) ----
            for k in range(nz):
                T = T_atm[i, k]; p = p_mid[k]
                ice = T < T_FREEZE                              # PRE-heating production ice (used by reevap)
                src_ice[i, k] = ice
                L_eff = L_SUB if ice else L_VAP
                qs = _qsat_s(T, p)
                denom = 1.0 + (L_eff / CP_AIR) * _dqsat_dT_s(T, p)
                # PDF partition on total water q+q_c (rad.cloud_pdf_partition): f and qc_eq from one Q1
                q_t = q[i, k] + q_c[i, k]
                delta = (1.0 - rh_crit) * qs
                if delta < PDF_WIDTH_FLOOR:
                    delta = PDF_WIDTH_FLOOR
                Q1 = (q_t - qs) / delta
                f = 0.5 * (Q1 + 1.0)
                if f < 0.0:
                    f = 0.0
                elif f > 1.0:
                    f = 1.0
                if Q1 >= 1.0:
                    qc_eq = q_t - qs
                elif Q1 <= -1.0:
                    qc_eq = 0.0
                else:
                    qc_eq = 0.25 * delta * (1.0 + Q1) * (1.0 + Q1)
                if qc_eq < 0.0:
                    qc_eq = 0.0
                # #294 (boil_suppress): no equilibrium condensate above boiling -- qc_eq -> 0 in a boiling
                # layer, so net below kills the condensation source AND drains suspended q_c to vapour
                # (denom == 1 there: the clamped dqsat_dT is 0). boiling also zeroes precip_demand below.
                boiling = False
                if boil_suppress != 0 and _esat_s(T) >= 0.999 * p:
                    qc_eq = 0.0
                    boiling = True
                net = (qc_eq - q_c[i, k]) / denom
                cond = net if net > 0.0 else 0.0
                evap_demand = (-net) if net < 0.0 else 0.0     # qc_reevaporation ON
                if cond_tau > 0.0:                             # #280 finite-tau saturation adjustment: ONE frac
                    cfrac = dt / cond_tau                      # scales both signs of ``net`` (conjugate exchange
                    if cfrac > 1.0:                            # preserved); tau=0 skips -> shipped instantaneous.
                        cfrac = 1.0
                    cond *= cfrac
                    evap_demand *= cfrac
                qc_avail = q_c[i, k] + cond
                # autoconversion (kessler, phase-split): max(qc_avail-crit,0)/tau * dt
                if ice:
                    tau = autoconv_tau_ice; crit = autoconv_crit_ice
                else:
                    tau = autoconv_tau; crit = autoconv_crit
                pa = qc_avail - crit
                precip_demand = (pa / tau * dt) if pa > 0.0 else 0.0
                if not ice:                                    # #237 salt warm-rain autoconv DEMAND scaling
                    precip_demand *= warm_factor[i]            # (N_d/N_ref)^beta (LIQUID only); 1.0 when salt off
                # retained-water ceiling (decoupled PLAUSIBLE bound; cap only where f >= f_min)
                rho_air = p / (R_DRY * T)
                lwc_ceiling = IWC_PLAUSIBLE if ice else LWC_PLAUSIBLE
                rho_den = rho_air if rho_air > 1e-9 else 1e-9
                qc_ceiling = lwc_ceiling * f / rho_den
                if f >= cloud_f_min:
                    ce = qc_avail - qc_ceiling
                    if ce > 0.0:
                        precip_demand += ce
                if boiling:
                    precip_demand = 0.0             # #294: no precip forms in a boiling layer
                # coupled-sink positivity limiter: reevap first, precip takes the rest
                evap = evap_demand if evap_demand < qc_avail else qc_avail
                rem = qc_avail - evap
                precip_qc = precip_demand if precip_demand < rem else rem
                q_c[i, k] = qc_avail - evap - precip_qc
                q[i, k] = q[i, k] - cond + evap
                T_atm[i, k] = T + L_eff * (cond - evap) / CP_AIR
                precip_qc_buf[i, k] = precip_qc                # carry the precip SOURCE [kg/kg] into phase 2
            # ---- PHASE 2: below-cloud reevap sweep (_below_cloud_reevap, precip_reevap ON). The falling precip
            #               descends through the (post-heating) column; snow (src_ice) and rain are two downward
            #               mass fluxes Fs/Fr with a melt term, re-evaporated flux- AND deficit-limited. q and
            #               T_atm are updated in place by the per-layer dq/dT; the surface flux Fs+Fr -> precip.
            Fs = 0.0; Fr = 0.0
            for k in range(nz):
                T = T_atm[i, k]; p = p_mid[k]
                qs = _qsat_s(T, p)
                warm = T >= T_FREEZE
                L_layer = L_VAP if warm else L_SUB
                denom = 1.0 + (L_layer / CP_AIR) * _dqsat_dT_s(T, p)
                deficit = qs - q[i, k]
                if deficit < 0.0:
                    deficit = 0.0
                rate = reevap_rate * deficit * dm[k]            # deficit-proportional rate [kg/m^2/s]
                sat_cap = deficit / denom * dm[k] / dt          # anti-oversaturation cap
                evap_cap = rate if rate < sat_cap else sat_cap
                if virga_flash_boil != 0 and _esat_s(T) >= 0.999 * p:
                    # #294 flash-boil virga: a boiling layer cannot hold falling liquid -- FLOOR the cap at
                    # the ENERGY cap (full-flux flash down to the layer's own boiling point), keeping the
                    # ordinary deficit-driven rate as the floor's other arm (PR #296 review: a pure replace
                    # PINS a rain-cooled layer at T_boil into permanent pass-through). Replaces the #295
                    # sat_cap bound (wrong above boiling: ~0 in a steam-saturated layer).
                    bc = T - _T_boil_s(p)
                    if bc < 0.0:
                        bc = 0.0
                    bc = bc * CP_AIR * dm[k] / (L_VAP * dt)
                    if bc > evap_cap:
                        evap_cap = bc
                src = precip_qc_buf[i, k] * dm[k] / dt          # per-layer precip SOURCE flux [kg/m^2/s]
                if src_ice[i, k]:                               # production phase: snow vs rain source
                    Fs += src
                else:
                    Fr += src
                if warm:                                        # snow melts to rain (absorbs L_FUS)
                    melt = Fs
                    Fs = 0.0; Fr += melt
                    T_atm[i, k] = T_atm[i, k] - L_FUS * (melt * dt / dm[k]) / CP_AIR
                Ftot = Fs + Fr
                E = Ftot if Ftot < evap_cap else evap_cap       # total reevap, flux- AND deficit-limited
                fr = E / (Ftot if Ftot > 1e-30 else 1e-30)      # split by falling snow/rain fractions
                Es = fr * Fs; Er = fr * Fr
                Fs -= Es; Fr -= Er
                q[i, k] = q[i, k] + (Es + Er) * dt / dm[k]      # reevaporated vapour rejoins q
                T_atm[i, k] = T_atm[i, k] - (L_SUB * Es + L_VAP * Er) * dt / (dm[k] * CP_AIR)
            precip_out[i] = Fs + Fr                             # surface-reaching precip [kg/m^2/s]

    return _micro_numba


_MICRO_SCRATCH = {}   # per-(nx,nz) reused scratch (src_ice bool, precip_qc_buf fp64) so the ON path allocates once
#   THREAD-SAFETY: this is a process-global mutable buffer keyed by (nx,nz). It is safe for the intended use
#   (the prange threads INSIDE one kernel call write disjoint columns of the SAME buffer; one core steps
#   sequentially), but it would RACE if two same-shape cores ran the ON microphysics concurrently in threads
#   in ONE process (they'd share the (nx,nz) scratch). The cascade/production driver runs one core per step,
#   so this does not arise; a future in-process multi-core ON path must key the scratch per-core (or pass it in).


def _micro_fuse_salt_cond_params(core, xp, dtype):
    """The (cond_tau, apply_warm, warm_factor) the fused micro kernels need to cover the salt (#237 N_d
    warm-rain autoconv) + finite-tau condensation (#280) route -- the configs that used to fall to the
    un-fused generic ``_prognostic_microphysics``.

      * ``cond_tau`` (scalar): ``c.cond_tau`` when >0 and NOT ``cond_tau_salt`` (0 -> the kernels skip the
        frac, byte-identical). The per-column ``cond_tau_salt`` variant is NOT handled here -- the
        ``_micro_canonical`` gate keeps it on the generic path.
      * ``apply_warm`` (bool/int): salt N_d warm-autoconv active.
      * ``warm_factor`` ((nx,) xp, ``dtype``): ``(N_d/N_ref)^beta`` when active, else all-ones (never read).
        CACHED on the core (``_micro_warm``) so the buffer address is STABLE across CUDA-graph replays; the
        ``**beta`` pow is done ONCE here at build/warmup time -- never inside the captured step.

    Cache staleness is caught BY CONSTRUCTION, not by convention: the cache entry holds a REFERENCE to the
    ``N_d`` array it was built from, and a rebuild fires whenever that array is no longer identical to
    ``core.N_d`` (``_build_salt_ccn_field`` always *reassigns* ``self.N_d`` to a fresh object -- and so would
    any direct ``core.N_d = ...``). Holding the reference (rather than an ``id()``) sidesteps the id-reuse
    hazard. The rebuild only ever fires OUTSIDE capture (``N_d`` is never reassigned mid-captured-step), so the
    returned buffer stays address-stable across replays. ``_build_salt_ccn_field`` also clears the cache, but
    that is now belt-and-suspenders -- correctness no longer depends on it."""
    c = core.cfg
    nx = core.T_atm.shape[0]
    cond_tau = float(c.cond_tau) if (c.cond_tau > 0.0 and not c.cond_tau_salt) else 0.0
    apply_warm = bool(c.salt_ccn and c.salt_ccn_autoconv and getattr(core, "N_d", None) is not None)
    cache = getattr(core, "_micro_warm", None)
    if cache is None:
        cache = {}
        core._micro_warm = cache
    ckey = (str(np.dtype(dtype)), apply_warm)
    nd_src = core.N_d if apply_warm else None                              # the array warm must track (None -> unused)
    entry = cache.get(ckey)                                                # (warm, nd_src_at_build) or None
    warm = entry[0] if entry is not None else None
    if warm is None or warm.shape != (nx,) or entry[1] is not nd_src:      # rebuild on first use / regrid / N_d swap
        if apply_warm:
            nd = th._F(core.N_d, xp)                                       # working-dtype (nx,), device-safe
            warm = xp.ascontiguousarray((nd / c.salt_ccn_nd_ref) ** c.salt_ccn_autoconv_beta, dtype=dtype)
        else:
            warm = xp.ones(nx, dtype=dtype)                                # never read (apply_warm 0); a valid arg
        cache[ckey] = (warm, nd_src)
    return cond_tau, (1 if apply_warm else 0), warm


def prognostic_microphysics_reevap_parallel(core, q, dt):
    """Column-parallel (numba prange) drop-in for the CANONICAL ``_prognostic_microphysics`` (precip_reevap ON)
    path: the fused saturation-adjustment microphysics + below-cloud reevap. Updates ``core.T_atm``,
    ``core.q_c`` and ``core._precip`` IN PLACE and returns the updated vapour ``q`` (matching the numpy method's
    contract). Reproduces the numpy reference to round-off (~1e-13), not bit-for-bit. The caller guards the
    config (canonical combo, fp64) and falls back to numpy otherwise."""
    global _MICRO_NUMBA_KERNEL
    if _MICRO_NUMBA_KERNEL is None:
        _MICRO_NUMBA_KERNEL = _build_micro_numba_kernel()
    c = core.cfg
    nx, nz = core.T_atm.shape
    key = (nx, nz)
    sc = _MICRO_SCRATCH.get(key)
    if sc is None:
        sc = (np.empty((nx, nz), dtype=np.bool_), np.empty((nx, nz), dtype=np.float64))
        _MICRO_SCRATCH[key] = sc
    src_ice, precip_qc_buf = sc
    q = np.ascontiguousarray(q, dtype=np.float64)              # phase-1/2 write q in place; return it
    T_atm = np.ascontiguousarray(core.T_atm, dtype=np.float64)
    q_c = np.ascontiguousarray(core.q_c, dtype=np.float64)
    precip_out = np.empty(nx, dtype=np.float64)
    # #280 finite-tau cond (scalar; 0 -> skipped) + #237 salt warm-autoconv factor (nx,); ones/0 when inactive
    # -> byte-identical to the shipped canonical kernel (cond_tau branch skipped, warm_factor==1 exact).
    cond_tau_k, _apply_warm, warm_k = _micro_fuse_salt_cond_params(core, np, np.float64)
    _MICRO_NUMBA_KERNEL(T_atm, q, q_c, np.ascontiguousarray(core.dm, dtype=np.float64),
                        np.ascontiguousarray(core.p_mid, dtype=np.float64), float(dt),
                        float(c.cloud_rh_crit), float(c.qc_autoconv_tau), float(c.qc_autoconv_crit),
                        float(c.qc_autoconv_tau_ice), float(c.qc_autoconv_crit_ice), float(c.cloud_f_min),
                        float(c.precip_reevap_rate), float(cond_tau_k), warm_k,
                        int(c.virga_flash_boil), int(c.cond_boil_suppress),   # #294 (0/0 = shipped arithmetic)
                        src_ice, precip_qc_buf, precip_out)
    core.T_atm = T_atm
    core.q_c = q_c
    core._precip = precip_out
    return q


def single_column_rce(S, T_surf, *, nz=30, p_top=5.0e3, co2_ppm=400.0, k_wv=rad.K_WV_LW,
                      tau_wellmixed=rad.TAU_LW_WELLMIXED, wind=6.0, rh_surf=0.8, rh_top=0.2,
                      conv_max_sweeps=8, dt=7200.0, max_iter=8000, tol=2e-5, q_init_rh=True,
                      closure="q_homogenize"):
    """The INDEPENDENT tier-2/3 RCE anchor: relax ONE atmospheric column to radiative-convective
    equilibrium at a PRESCRIBED surface temperature ``T_surf`` and day-mean insolation ``S``, with the
    SAME leaf physics as the core (radiation.column_radiation, the moist convective adjustment, prognostic
    moisture: surface evaporation toward absolute qsat over A's wetness mask, condensation of the
    supersaturation to precip with the latent heat released to the air) but a DIFFERENT relaxation loop --
    NOT ``StageADCore.step`` (the roadmap's "a standalone radiative-convective relaxation, not the
    solver's own loop"). The core's psi==0 (no-flow) single-column limit must reproduce this to tier-2
    tolerance; disagreement is a wiring bug in the core's assembly (the point of an independent anchor).

    Fixed-SST RCE (the classic Manabe set-up): the surface is a prescribed boundary condition (the
    surface ENERGY balance that makes T_surf prognostic is M2b.4), so the atmosphere -- low heat capacity
    -- relaxes quickly. Returns a dict (T_atm (nz,) TOP->SURFACE, q, OLR, toa_net, p_mid, z, n_iter)."""
    p_edge = np.linspace(p_top, C.P_SURF, nz + 1)
    p_mid = 0.5 * (p_edge[:-1] + p_edge[1:])
    dm = np.diff(p_edge) / C.G_ACCEL
    C_layer = dm * C.CP_AIR
    sigma = p_mid / C.P_SURF
    wet = float(wetness_of_Tsurf(T_surf))
    alb = float(surface_albedo(T_surf, wet))
    z_init = -8000.0 * np.log(sigma)
    T = np.maximum(float(T_surf) - 6.5e-3 * z_init, 130.0)
    RHprof = rh_top + (rh_surf - rh_top) * sigma
    q = np.minimum(wet * RHprof * th.qsat(T, p_mid), 0.5) if q_init_rh else np.zeros(nz)
    Ck = C.RHO_AIR0 * C.CP_AIR * 1.2e-3 * wind
    Ce = C.RHO_AIR0 * C.L_VAP * 1.2e-3 * wind
    qs_surf = float(th.qsat(np.asarray(float(T_surf)), C.P_SURF))
    out = None
    for it in range(max_iter):
        T_old = T.copy()
        z = hydrostatic_height(T, p_edge)
        out = rad.column_radiation(p_edge, T, q, float(T_surf), float(S), alb, co2_ppm=co2_ppm,
                                   k_wv=k_wv, tau_wellmixed=tau_wellmixed)
        dT = dt * out["H"] / C_layer
        dT[-1] += dt * Ck * (float(T_surf) - T[-1]) / C_layer[-1]    # surface sensible heat -> lowest layer
        T = T + dT
        # evaporation: surface latent flux moistens the lowest layer (absolute footing, wetness-gated)
        LE = wet * Ce * max(qs_surf - q[-1], 0.0)
        q[-1] += (LE / C.L_VAP) * dt / dm[-1]
        # moist convective adjustment: mixes T (to the moist adiabat) AND lofts q (vertical transport),
        # with the selectable moisture closure (CLEARSKY_SW_PLAN Stage 3; default q_homogenize)
        T2, q2 = T[None, :].copy(), q[None, :].copy()
        moist_convective_adjust(T2, dm, z[None, :], p_mid, conv_max_sweeps, q=q2,
                                closure=closure, rh_surf=rh_surf, rh_top=rh_top)
        T, q = T2[0], q2[0]
        # condensation of supersaturation -> precip + latent heat to the air (energy-consistent on-grid)
        cond = np.maximum(q - th.qsat(T, p_mid), 0.0)
        q = q - cond
        T = T + C.L_VAP * cond / C.CP_AIR
        if np.max(np.abs(T - T_old)) < tol:
            break
    return dict(T_atm=T, q=q, OLR=float(out["OLR"]), toa_net=float(out["toa_net"]),
                p_mid=p_mid, z=z, n_iter=it + 1, wet=wet, albedo=alb)


# ======================================================================================
# THE SURFACE + BOUNDARY-LAYER KERNEL (M2b.4): the standalone bulk air-sea SST anchor and the
# radiative-subsidence rate. Module-level pure functions, like the dynamical and column-thermo
# kernels, so the SAME leaf physics is shared by ``StageADCore`` and its external cross-checks.
# ======================================================================================
def bulk_air_sea_sst(SW_abs, LW_down, T_air, q_air, wind, *, wetness=None, eps_surf=1.0, Ch=1.2e-3,
                     Ce=1.2e-3, rho=C.RHO_AIR0, T_lo=150.0, T_hi=900.0, n_bisect=100):
    """The INDEPENDENT tier-4 cross-method anchor for the surface solver: the steady **bulk air-sea
    energy-balance** SST that satisfies, column by column,

        SW_abs + LW_down  ==  eps*sigma*T_s^4  +  SH(T_s)  +  LE(T_s)

    with the SAME bulk-aerodynamic fluxes the core's surface step uses --
        SH = rho*cp*Ch*wind*(T_s - T_air),
        LE = wetness * rho*L_v*Ce*wind * max(qsat(T_s, P_SURF) - q_air, 0).
    Inputs are the column's radiative drivers (``SW_abs`` the absorbed shortwave at the surface and
    ``LW_down`` the downwelling longwave -- both READ from the core's radiation, so albedo is already
    folded into ``SW_abs``) and its near-surface air state (``T_air``, ``q_air``, ``wind``). This is
    *cross-method* (not the core's loop): the core INTEGRATES the surface ODE
    C dT_s/dt = (SW_abs + LW_down - sigma T^4 - SH - LE) to steady state, while this ALGEBRAICALLY
    root-finds the same balance -- so agreement validates that the integrator actually reaches the
    energy-balance root (catches a wrong sign, a wrong heat-to-surface, a non-converging semi-implicit
    step), independent of the heat capacity / time stepping.

    ``wetness``: the evaporation efficiency to use for LE. The wetness ramp `wetness_of_Tsurf(T_s)` is
    MULTIVALUED in the surface balance -- it admits a cool **wet** (evaporating) branch AND a hot **dry**
    salt-flat branch (the Komabayashi-Ingersoll runaway-greenhouse hysteresis), and which branch the core
    sits on is set by its spin-up history, not by the instantaneous energy balance. So the cross-check
    must compare on the SAME branch: pass the core's converged ``wetness`` (then the balance is
    single-valued and the residual below is monotone in T_s, so the bisection finds THE root the core
    sits at). With ``wetness=None`` it falls back to the self-consistent `wetness_of_Tsurf(T_s)` (the
    multivalued version -- the bisection then returns the *coolest* / wet branch). Scalar or array.

    DEVICE/dtype (Phase D): array-module-generic + fp32-gate-following via ``th._F`` -- numpy fp64 in is
    byte-identical, a CuPy input runs the vectorized bisection on the GPU. All array data routed through ONE
    ``xp`` (the mixed-input rule). On the coupled hot path this is reached only via ``_budget_E_w`` when
    ``budget_wet_fraction`` is on; the bisection is a fixed ``n_bisect``-count vectorized sweep (no host root-
    find), so it is capture-clean and device-resident."""
    xp = th._xp_of(SW_abs, LW_down, T_air, q_air, wetness if wetness is not None else SW_abs)
    SW_abs = th._F(SW_abs, xp); LW_down = th._F(LW_down, xp)
    T_air = th._F(T_air, xp); q_air = th._F(q_air, xp)
    Ck = rho * C.CP_AIR * Ch * wind
    Ce_ = rho * C.L_VAP * Ce * wind

    def residual(T):
        SH = Ck * (T - T_air)
        wet = wetness_of_Tsurf(T) if wetness is None else th._F(wetness, xp)
        LE = wet * Ce_ * xp.maximum(th.qsat(T, C.P_SURF) - q_air, 0.0)
        return SW_abs + LW_down - eps_surf * C.SIGMA_SB * T**4 - SH - LE

    lo = xp.full_like(SW_abs, T_lo); hi = xp.full_like(SW_abs, T_hi)
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        pos = residual(mid) > 0.0            # residual>0 means losses too small -> warmer root
        lo = xp.where(pos, mid, lo)
        hi = xp.where(pos, hi, mid)
    return 0.5 * (lo + hi)


def radiative_subsidence_rate(Q_rad_K_day, dthetadz_K_km):
    """The M2a radiative-subsidence balance `w_sub = Q_rad / (d theta_pot/dz)` (the external anchor of
    `archive/M2A_RESULTS.md`: w_sub = -0.25 cm/s at the ribbon). In a radiatively-cooling, stably-stratified
    free troposphere the steady balance is adiabatic-subsidence-warming == radiative cooling,
    w * d theta/dz = Q_rad, so a NEGATIVE Q_rad (cooling) over a POSITIVE stability gives w < 0
    (subsidence). ``Q_rad_K_day`` [K/day] and ``dthetadz_K_km`` [K/km]; returns w_sub in **cm/s**
    (the M2a table's unit). Pure arithmetic -- the velocity is set by the column's own radiation +
    stability, no free U anchor (that is the whole point of the M2a probe).

    NOTE: no guard against a near-zero ``dthetadz_K_km`` -- the balance is only defined on a stably
    stratified layer (the subsiding free troposphere, the regime ``subsidence_diagnostics`` extracts and
    the M2b.4 test asserts `stab > 0`). On a near-neutral or unstable column `w_sub` is meaningless (the
    subsidence-balance assumption fails); add a stability floor / regime mask before reusing it there."""
    Q = np.asarray(Q_rad_K_day, float) / 86400.0          # K/s
    dthdz = np.asarray(dthetadz_K_km, float) / 1000.0     # K/m
    return (Q / dthdz) * 100.0                             # m/s -> cm/s
