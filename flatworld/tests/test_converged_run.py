"""TDD tests for the M3.3/M3.4 time-averaged-convergence harness (``flatworld.converged_run``).

The harness's RISK is the ESTIMATOR (M33_M34_DESIGN.md): a stop that fires early, or a noise floor that
is under-sized, would silently corrupt the M3.5 pin verdict. So these tests pin the estimator against
SYNTHETIC signals with KNOWN answers (the review's "verify, not assert"):
  * integrated_autocorr_time recovers a known AR(1) tau;
  * the exponential-approach fit recovers (T_inf, tau) and flags an extrapolating/flat fit;
  * the sampling floor eps recovers sigma/sqrt(ESS) for white noise AND inflates correctly for AR(1);
  * the asymptote-projection STOP does NOT fire while the remaining-to-asymptote is above the floor
    (the trap), DOES fire once decayed into the floor, and is NOT fooled by a fast+slow two-timescale
    signal whose slow mode still carries > eps after the fast mode dies.
All synthetic, deterministic (seeded); no StageADCore run needed for the estimator gate.
"""
import numpy as np
import pytest

from flatworld import constants as C
from flatworld.stage_ad_core import StageADCore, StageADConfig


def _frozen_core(**kw):
    """A tiny core stepping a FROZEN no-op state (run_dynamics + couple_thermo off): dt == cfg.dt,
    state finite and unchanging -- fast and deterministic for the driver's loop/guard/checkpoint tests."""
    c = StageADConfig(nx=10, nz=12, theta_max_deg=55.0, dt=7200.0, **kw)
    m = StageADCore(c)
    m.run_dynamics = False
    m.couple_thermo = False
    return m


def _ar1(phi, n, seed=0):
    """A stationary AR(1) series x_t = phi x_{t-1} + eps_t, eps~N(0,1). Theoretical integrated
    autocorrelation time (tau = 1 + 2*sum_k rho(k), rho(k)=phi^k) is (1+phi)/(1-phi)."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n + 2000)
    x = np.zeros(n + 2000)
    for i in range(1, x.size):
        x[i] = phi * x[i - 1] + e[i]
    return x[2000:]                                   # drop the burn-in


def test_integrated_autocorr_time_recovers_known_ar1():
    from flatworld.converged_run import integrated_autocorr_time
    phi = 0.8
    tau_theory = (1 + phi) / (1 - phi)               # = 9.0
    tau, M = integrated_autocorr_time(_ar1(phi, 120_000, seed=1))
    assert abs(tau - tau_theory) / tau_theory < 0.15, f"tau={tau:.2f} vs theory {tau_theory:.2f}"
    assert 0 < M < 120_000


def test_integrated_autocorr_time_white_noise_is_one():
    from flatworld.converged_run import integrated_autocorr_time
    rng = np.random.default_rng(2)
    tau, _ = integrated_autocorr_time(rng.standard_normal(50_000))
    assert abs(tau - 1.0) < 0.3                       # white noise has tau == 1


# ------------------------------------------------------------------ exponential-approach fit (the trend)
def test_fit_exponential_approach_recovers_tau_and_asymptote():
    from flatworld.converged_run import fit_exponential_approach
    rng = np.random.default_rng(3)
    tau_true, T_inf_true, A_true = 20.0, 40.0, 30.0
    t = np.linspace(0.0, 200.0, 600)                  # span = 10 tau
    y = T_inf_true - A_true * np.exp(-t / tau_true) + 0.05 * rng.standard_normal(t.size)
    fit = fit_exponential_approach(t, y)
    assert fit.ok and not fit.extrapolating
    assert abs(fit.tau - tau_true) / tau_true < 0.10
    assert abs(fit.T_inf - T_inf_true) < 0.2


def test_fit_exponential_approach_flags_extrapolation_when_span_too_short():
    from flatworld.converged_run import fit_exponential_approach
    tau_true = 100.0
    t = np.linspace(0.0, 40.0, 200)                   # span = 0.4 tau: only the early ramp is seen
    y = 40.0 - 30.0 * np.exp(-t / tau_true)
    fit = fit_exponential_approach(t, y)
    assert fit.extrapolating                          # tau >~ span => T_inf is an extrapolation, flag it


# ------------------------------------------------------------------ per-theta asymptote at fixed tau
def test_project_asymptote_recovers_per_theta_Tinf_at_fixed_tau():
    from flatworld.converged_run import project_asymptote
    rng = np.random.default_rng(4)
    tau = 25.0
    t = np.linspace(0.0, 250.0, 500)
    T_inf = np.array([10.0, 20.0, 30.0, 25.0])        # 4 "columns" with distinct asymptotes
    A = np.array([5.0, 8.0, 12.0, 6.0])
    profiles = T_inf[None, :] - A[None, :] * np.exp(-t[:, None] / tau) \
        + 0.02 * rng.standard_normal((t.size, 4))
    Tinf_hat, A_hat = project_asymptote(t, profiles, tau)
    assert np.max(np.abs(Tinf_hat - T_inf)) < 0.1     # per-theta asymptote recovered at fixed (shared) tau


# ------------------------------------------------------------------ the sampling floor eps (the M3.4 floor)
def test_sampling_floor_predicts_empirical_spread_of_the_mean_for_correlated_series():
    """The GOLD test (review's 'verify N_eff, not assert'): eps from a SINGLE correlated series must
    predict the EMPIRICAL std of the tail-mean across many INDEPENDENT realizations. White-noise sigma/√N
    would under-estimate it by √IAT; the IAT-based ESS must recover it."""
    from flatworld.converged_run import sampling_floor
    phi, N, M = 0.8, 4000, 400
    means = np.array([_ar1(phi, N, seed=100 + s).mean() for s in range(M)])
    empirical_eps = float(np.std(means, ddof=1))      # the true sampling uncertainty of the mean
    fl = sampling_floor(_ar1(phi, N, seed=1))         # estimated from ONE series
    assert abs(fl.eps - empirical_eps) / empirical_eps < 0.25, f"eps={fl.eps:.4f} vs {empirical_eps:.4f}"
    # and it must BEAT the naive iid floor (which ignores correlation -> under-estimates by ~√IAT≈3)
    naive = np.std(_ar1(phi, N, seed=1), ddof=1) / np.sqrt(N)
    assert fl.eps > 2.0 * naive                       # IAT inflation is real and captured


def test_sampling_floor_white_noise_is_sigma_over_sqrt_N():
    from flatworld.converged_run import sampling_floor
    rng = np.random.default_rng(7)
    x = rng.standard_normal(8000)
    fl = sampling_floor(x)
    assert abs(fl.iat - 1.0) < 0.4
    assert abs(fl.eps - 1.0 / np.sqrt(8000)) / (1.0 / np.sqrt(8000)) < 0.3


# ------------------------------------------------------------------ the composed asymptote-projection STOP
def _profiles(t, T_inf, amps, taus, noise, seed, ntheta=3):
    """Synthetic window-mean profile trajectory: sum of exponentials toward T_inf + white noise."""
    rng = np.random.default_rng(seed)
    y = np.full((t.size, ntheta), float(T_inf))
    for A, tau in zip(amps, taus):
        y -= A * np.exp(-t / tau)[:, None]
    return y + noise * rng.standard_normal((t.size, ntheta))


def test_stop_does_NOT_fire_while_remaining_offset_above_floor():
    """THE TRAP (the advisor's correction): a slow-exponential approach whose remaining-to-asymptote is
    STILL ABOVE the floor must NOT be declared converged -- even though the per-window slope is already
    small. Span = 2*tau: the tail-mean is biased by ~A*exp(-1.8) >> eps."""
    from flatworld.converged_run import analyze_convergence
    tau = 20.0
    t = np.linspace(0.0, 40.0, 400)
    prof = _profiles(t, 40.0, [30.0], [tau], noise=0.5, seed=11)
    res = analyze_convergence(t, prof)
    assert not res.converged


def test_stop_DOES_fire_once_decayed_into_the_floor():
    """The complement: a slow exponential run long past the asymptote (span = 13*tau) IS converged, and
    the reported tail-mean is within a few eps of the true T_inf."""
    from flatworld.converged_run import analyze_convergence
    tau = 20.0
    t = np.linspace(0.0, 260.0, 2600)
    prof = _profiles(t, 40.0, [30.0], [tau], noise=2.0, seed=12)
    res = analyze_convergence(t, prof)
    assert res.converged
    assert np.max(np.abs(res.tailmean - 40.0)) < 0.1      # central value not biased beyond its error bar


def test_stop_is_NOT_fooled_by_a_fast_plus_slow_two_timescale_signal():
    """THE TWO-TIMESCALE TRAP (calibrated to the real ~10x C_ocean/C_rock ratio): after the FAST mode dies
    the SLOW mode still carries > eps, and a naive late-fit would declare victory. Span = 6*tau_fast =
    0.6*tau_slow: fast remaining ~A*exp(-6)=0.05 (gone), slow remaining ~A*exp(-0.6)=11 (huge). The slow
    tau is UNRESOLVED within the span (extrapolating) and T_inf is not stable -> must NOT converge."""
    from flatworld.converged_run import analyze_convergence
    tau_fast, tau_slow = 20.0, 200.0                      # the ~10x split
    t = np.linspace(0.0, 120.0, 1200)
    prof = _profiles(t, 40.0, [20.0, 20.0], [tau_fast, tau_slow], noise=0.5, seed=13)
    res = analyze_convergence(t, prof)
    assert not res.converged


def test_two_timescale_signal_DOES_converge_once_slow_mode_resolved():
    """Sanity complement: the SAME fast+slow signal run long past the SLOW timescale converges."""
    from flatworld.converged_run import analyze_convergence
    tau_fast, tau_slow = 20.0, 200.0
    t = np.linspace(0.0, 2600.0, 5200)                    # span = 13*tau_slow
    prof = _profiles(t, 40.0, [20.0, 20.0], [tau_fast, tau_slow], noise=2.0, seed=14)
    res = analyze_convergence(t, prof)
    assert res.converged
    assert np.max(np.abs(res.tailmean - 40.0)) < 0.15


# ------------------------------------------------------------------ the run-to-guard driver
def test_run_to_steady_state_logs_trajectory_and_stops_at_sim_time_guard():
    from flatworld.converged_run import run_to_steady_state
    m = _frozen_core()
    target = 5 * 7200.0                               # dt==7200 (run_dynamics off) => 5 steps of sim-time
    traj = run_to_steady_state(m, sim_time_target=target, sample_every=1, checkpoint_every=10**9,
                               max_iter=100, verbose=False)
    assert traj.stop_reason.startswith("sim_time")    # stopped on the sim-time guard, not max_iter
    assert traj.sim_time[-1] >= target
    assert traj.profiles.ndim == 2 and traj.profiles.shape[0] == traj.sim_time.size
    assert np.all(np.diff(traj.sim_time) > 0)         # monotonically increasing sampled sim-time
    assert traj.profiles.shape[1] == int(np.sum(m.theta_deg <= m.cfg.theta_max_deg))  # masked profile width


def test_run_to_steady_state_aborts_on_nonfinite_without_poisoning_checkpoint(tmp_path):
    """Carry the M3 cost guard to the driver: a mid-run blow-up raises FloatingPointError and does NOT
    overwrite the last good checkpoint (a poisoned resume state is worse than a wasted run)."""
    from flatworld import convergence as conv
    from flatworld.converged_run import run_to_steady_state
    ckpt = str(tmp_path / "drv.npz")
    m = _frozen_core(check_every=2)
    good_T = m.T_atm.copy()
    orig, calls = m.step, {"n": 0}
    def poisoned():
        orig(); calls["n"] += 1
        if calls["n"] == 5:
            m.T_atm[0, 0] = np.inf
    m.step = poisoned
    with pytest.raises(FloatingPointError):
        run_to_steady_state(m, sim_time_target=1e30, sample_every=1, checkpoint_path=ckpt,
                            checkpoint_every=2, max_iter=100, verbose=False)
    st, _ = conv.load_checkpoint(ckpt)
    assert st is not None and np.all(np.isfinite(st["T_atm"])) and np.allclose(st["T_atm"], good_T)


def test_run_to_steady_state_resumes_sim_time_and_iter_from_checkpoint(tmp_path):
    from flatworld.converged_run import run_to_steady_state
    ckpt = str(tmp_path / "res.npz")
    log = str(tmp_path / "res_traj.npz")
    m1 = _frozen_core()
    run_to_steady_state(m1, sim_time_target=4 * 7200.0, sample_every=1, checkpoint_path=ckpt,
                        log_path=log, checkpoint_every=2, max_iter=100, verbose=False)
    iter1 = m1.iter
    # a FRESH core resumes from the checkpoint and continues to a larger target
    m2 = _frozen_core()
    traj = run_to_steady_state(m2, sim_time_target=9 * 7200.0, sample_every=1, checkpoint_path=ckpt,
                               log_path=log, checkpoint_every=2, max_iter=100, verbose=False)
    assert m2.iter > iter1                            # continued, did not restart from 0
    assert traj.sim_time[-1] >= 9 * 7200.0
    assert np.all(np.diff(traj.sim_time) > 0)         # the resumed trajectory is contiguous in sim-time


# ------------------------------------------------------------------ the M3.4 noise floor (routes a/b/c)
def test_noise_floor_route_a_iat_and_route_b_blocks_agree_when_blocks_exceed_iat():
    """Routes (a) IAT/ESS and (b) block-resampling are two numerical routes to the SAME within-run floor;
    they agree BY CONSTRUCTION when block length >> IAT (the design's '(a≈b) window-sizing self-check')."""
    from flatworld.converged_run import noise_floor
    rep = noise_floor(_ar1(0.8, 6000, seed=21), n_blocks=8)   # block len 750 >> IAT≈9
    assert rep.blocks_ok
    assert abs(rep.eps_block - rep.eps) / rep.eps < 0.30      # the two within-run routes agree


def test_noise_floor_flags_when_blocks_are_shorter_than_iat():
    """When the blocks are too short relative to IAT (a≈b is NOT valid), the report must FLAG it -- so a
    bogus agreement isn't read as a passing self-check (the differential diagnosis the review asked for)."""
    from flatworld.converged_run import noise_floor
    rep = noise_floor(_ar1(0.9, 4000, seed=22), n_blocks=500)  # IAT≈19, block len 8 << IAT
    assert not rep.blocks_ok


def test_between_member_floor_recovers_known_spread_and_reports_its_uncertainty():
    """Route (c), the INDEPENDENT estimate: the between-member std of the converged tail-means recovers a
    known injected spread, and reports its OWN sampling uncertainty (it shrinks with ensemble size)."""
    from flatworld.converged_run import between_member_floor
    rng = np.random.default_rng(23)
    M, ntheta, sigma_true = 200, 4, 0.5
    truth = np.array([10.0, 20.0, 15.0, 5.0])
    tailmeans = truth[None, :] + sigma_true * rng.standard_normal((M, ntheta))
    bm = between_member_floor(tailmeans)
    assert np.max(np.abs(bm.eps - sigma_true)) < 0.15         # per-theta spread recovered
    assert bm.rel_uncertainty < 0.2 and bm.rel_uncertainty > 0.0  # M=200 -> ~1/sqrt(2*199) ~ 0.05


# ------------------------------------------------------------------ the heat-limb enthalpy audit (-4C ribbon)
def test_heat_limb_enthalpy_audit_is_nonmutating_and_zero_for_uniform_theta():
    """The -4C-ribbon audit: _advect_heat conserves int(theta_pot*dm) but not exactly int(cp*T*dm) (the
    Exner factor varies between layers). For a UNIFORM potential temperature the advection moves nothing,
    so BOTH the theta-mass and the enthalpy change must be ~0 -- and the audit must be NON-mutating."""
    from flatworld.converged_run import heat_limb_enthalpy_audit
    m = StageADConfig(nx=12, nz=14, theta_max_deg=60.0)
    core = StageADCore(m)
    # impose a real divergence-free single-cell flow (psi=0 on all walls) and a UNIFORM potential temp
    X = core.x[None, :] / core.x.max(); Z = core.z[:, None] / core.z.max()
    core.psi = 5.0e4 * np.sin(np.pi * X) * np.sin(np.pi * Z)
    core._last_dt_dyn = 100.0
    exner = (core.p_mid / C.P_SURF) ** C.KAPPA_POISSON
    core.T_atm = (300.0 * exner)[None, :] * np.ones((core.cfg.nx, 1))   # uniform theta = 300 K
    T_before = core.T_atm.copy()
    audit = heat_limb_enthalpy_audit(core)
    assert abs(audit["d_theta_mass_rel"]) < 1e-10            # theta-mass invariant (uniform-invariance)
    assert abs(audit["enthalpy_rate_frac"]) < 1e-10        # enthalpy too (nothing moves)
    assert np.array_equal(core.T_atm, T_before)             # NON-mutating (restores the state)


def test_enthalpy_audit_rate_is_dt_independent():
    """Finding 2 (review): the audit must report a dt-INDEPENDENT rate. DeltaH from one heat-advection
    substep is EXACTLY linear in dt, so the RATE ( DeltaH/dt, and its fraction of radiative throughput) is
    dt-independent -- which is what makes it comparable across the live path (dt=dt_dyn) and the analyze
    path (where _last_dt_dyn is unrestored == 0). A per-substep relative change would scale ~120x between
    dt_dyn and cfg.dt."""
    from flatworld.converged_run import heat_limb_enthalpy_audit
    core = StageADCore(StageADConfig(nx=12, nz=14, theta_max_deg=60.0))
    X = core.x[None, :] / core.x.max(); Z = core.z[:, None] / core.z.max()
    core.psi = 5.0e4 * np.sin(np.pi * X) * np.sin(np.pi * Z)   # a real divergence-free flow
    # T_atm is NON-uniform across theta (warm equator -> cool pole), so the heat limb moves enthalpy
    a1 = heat_limb_enthalpy_audit(core, dt_ref=1.0)
    a2 = heat_limb_enthalpy_audit(core, dt_ref=37.0)
    assert a1["enthalpy_rate_frac"] != 0.0                  # the limb actually moves enthalpy here
    rel = abs(a1["enthalpy_rate_frac"] - a2["enthalpy_rate_frac"]) / abs(a1["enthalpy_rate_frac"])
    assert rel < 1e-6                                       # the RATE is dt-independent (linear in dt)


def test_converged_diagnostics_sst_cross_check_physical_on_reloaded_core(tmp_path):
    """Finding 1 (review): the SST cross-check must pass the DOWNWELLING drivers (surf_abs_sw, D_surf) to
    bulk_air_sea_sst, NOT out['surf_net'] (which already folds in -sigma*Ts^4) -- the latter double-counts
    surface LW emission and collapses the root toward the T_lo floor (~150 K -> ~60+ K residual). At a
    no-flow RCE the surface IS at the bulk air-sea balance, so the residual is SMALL. Also exercises the
    production analyze path: checkpoint -> reload into a fresh core -> converged_diagnostics."""
    from flatworld import convergence as conv
    from flatworld.converged_run import converged_diagnostics
    cfg = StageADConfig(nx=12, nz=14, theta_max_deg=60.0, conv_theta_max=52.0)
    core = StageADCore(cfg)
    core.run_dynamics = False                              # no-flow RCE: T_surf relaxes to the surf balance
    for _ in range(2000):
        core.step()
    ckpt = str(tmp_path / "c.npz"); conv.save_checkpoint(ckpt, core.state(), {"iter": core.iter})
    core2 = StageADCore(cfg); st, _ = conv.load_checkpoint(ckpt); core2.load_state(st)
    d = converged_diagnostics(core2, conv_theta_max=52.0, ribbon_lo=44.0, ribbon_hi=50.0)
    assert d["sst_vs_bulk_root_ribbon_K"] < 3.0            # ~0.x with correct args; ~60+ with the bug
