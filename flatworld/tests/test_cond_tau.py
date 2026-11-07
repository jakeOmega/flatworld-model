"""Regression gate for the finite-tau condensation saturation-adjustment (cfg.cond_tau, issue #280).

The conv_tau treatment applied to the OTHER instantaneous thermo-sub-cycle operator: the saturation
adjustment (single-pass supersaturation rain-out with prognostic_condensate OFF; the PDF/grid-mean
q_c relaxation inside _prognostic_microphysics with it ON). Locks the load-bearing claims, mirroring
test_conv_tau.py: (1) default-OFF is byte-identical and never executes the cond_tau code, (2) tau small
enough that frac = min(dt/tau, 1) = 1 recovers the instantaneous scheme, (3) a partial frac has a real
effect, (4) the scaled exchange stays water-conserving (the reservoir budget closes), (5) the point of
the fix: the climate's thermo-sub-cycle (n_sub) sensitivity shrinks with cond_tau ON, and (6) constant
cond_tau>0 is now THREADED into the fused numba/device kernels (was route-to-generic) and reproduces the
generic path to round-off; only the per-column cond_tau_salt stays generic-routed. CPU/fp64, no GPU/donor
needed.
"""
import numpy as np
import pytest

from flatworld.stage_ad_core import StageADCore, StageADConfig


def _core(**kw):
    return StageADCore(StageADConfig(nx=24, nz=16, **kw))


def _prog_core(**kw):
    # prognostic-condensate column with the canonical sub-flags at their defaults (pdf source, dqsat
    # denom, ice phase, reevaporation, retained ceiling); a lowered autoconv crit so rain actually flows.
    return StageADCore(StageADConfig(nx=24, nz=16, prognostic_condensate=True,
                                     qc_autoconv_crit=5.0e-4, **kw))


def _run(core, n):
    for _ in range(n):
        core.step()
    return (np.array(core.T_atm, float).copy(), np.array(core.q, float).copy(),
            np.array(core.T_surf, float).copy())


def test_cond_tau_off_is_inert_and_byte_identical():
    """cond_tau=0 (default): bit-identical to a run that never set cond_tau, on BOTH condensation paths
    (single-pass core and prognostic microphysics) -- the default-OFF guarantee against future drift."""
    a = _core(cond_tau=0.0); Ta, qa, Tsa = _run(a, 20)
    b = _core();             Tb, qb, Tsb = _run(b, 20)   # default cond_tau is 0.0
    assert np.array_equal(Ta, Tb) and np.array_equal(qa, qb) and np.array_equal(Tsa, Tsb)
    ap = _prog_core(cond_tau=0.0); Tap, qap, _ = _run(ap, 20)
    bp = _prog_core();             Tbp, qbp, _ = _run(bp, 20)
    assert np.array_equal(Tap, Tbp) and np.array_equal(qap, qbp)
    assert np.array_equal(ap.q_c, bp.q_c)


def test_cond_tau_frac1_recovers_instantaneous():
    """tau <= thermo_dt => frac = min(dt/tau,1) = 1 => the full adjustment, recovering the shipped
    instantaneous scheme (the multiply by exactly 1.0 is IEEE-exact, so equal to the bit)."""
    off = _core(cond_tau=0.0);  To, qo, Tso = _run(off, 20)
    on  = _core(cond_tau=1e-3); Tn, qn, Tsn = _run(on, 20)   # frac saturates at 1
    assert np.array_equal(To, Tn) and np.array_equal(qo, qn) and np.array_equal(Tso, Tsn)
    offp = _prog_core(cond_tau=0.0);  Top, qop, _ = _run(offp, 20)
    onp  = _prog_core(cond_tau=1e-3); Tnp, qnp, _ = _run(onp, 20)
    assert np.array_equal(Top, Tnp) and np.array_equal(qop, qnp)
    assert np.array_equal(offp.q_c, onp.q_c)


def test_cond_tau_has_a_real_effect_when_partial():
    """A finite tau >> thermo_dt (frac < 1) must MOVE the climate away from the instantaneous result on
    both paths -- otherwise the scaling would be a silent no-op."""
    off = _core(cond_tau=0.0);    To, _, _ = _run(off, 30)
    on  = _core(cond_tau=1.0e5);  Tn, _, _ = _run(on, 30)    # frac = dt/1e5 << 1
    assert np.max(np.abs(To - Tn)) > 1e-3, "cond_tau>0 should change the single-pass trajectory"
    offp = _prog_core(cond_tau=0.0);    Tp, _, _ = _run(offp, 30)
    onp  = _prog_core(cond_tau=1.0e5);  Tq, _, _ = _run(onp, 30)
    assert np.max(np.abs(Tp - Tq)) > 1e-3, "cond_tau>0 should change the prognostic trajectory"


def test_cond_tau_water_budget_closes():
    """The scaled exchange stays exactly water-conserving: cond and evap_demand carry the SAME frac (one
    signed adjustment), so d/dt INTEGRAL (q + q_c) dm = E - P still closes to round-off with cond_tau ON
    (the test_cloud_prognostic reservoir-budget check, run at a partial frac)."""
    m = StageADCore(StageADConfig(nx=20, nz=18, theta_max_deg=55.0, prognostic_condensate=True,
                                  qc_autoconv_crit=5.0e-4, cond_tau=3.0e3))
    m.run_dynamics = False
    m.couple_surface = False                                    # fixed SST -> bounded q (no clip artifact)
    m.T_surf[:] = 292.0
    for _ in range(20):                                         # spin out the initial transient
        m.step()
    m.reset_water_budget()
    tw0 = float(np.sum((m.q + m.q_c) * m.dm[None, :]) * m.dx)   # area-weighted total water [kg/m]
    for _ in range(40):
        m.step()
    tw1 = float(np.sum((m.q + m.q_c) * m.dm[None, :]) * m.dx)
    wb = m.water_conservation()
    lhs = tw1 - tw0                                             # storage change
    rhs = wb["cum_evap"] - wb["cum_precip"]                     # E - P
    assert abs(lhs - rhs) / (abs(rhs) + 1e-12) < 1e-6           # reservoir budget closes at partial frac
    assert np.all(m.q_c >= 0.0)                                 # positivity limiter intact under scaling


def test_cond_tau_makes_the_operator_dt_convergent():
    """The claim the flag exists for (#280): sub-cycling the INSTANTANEOUS saturation adjustment against
    dt-scaled forcing is not dt-convergent, and the finite-tau split fixes exactly that. Isolate the
    operator: sub-cycle _prognostic_microphysics ALONE over a fixed window against fixed forcing (steady
    cooling that drives supersaturation + a steady moisture source), on a refinement ladder n = 4..64
    substeps. With cond_tau=0 the rung-to-rung state differences decay SLOWLY (sub-first-order: measured
    ratios ~0.83 -> 0.65); with a finite cond_tau (partial frac at every rung) they halve with each dt
    halving (clean O(dt)) and end an order of magnitude smaller. (A full-core n_sub climate comparison is
    NOT lockable at test scale -- other operators dominate there; the production inner-zone confirmation
    is the staged GPU run, mirroring how conv_tau's own empirical confirmation was staged.)"""
    import flatworld.thermo as th

    def _ladder(tau, window=3600.0, rungs=(4, 8, 16, 32, 64)):
        diffs = []
        prev = None
        for n in rungs:
            m = StageADCore(StageADConfig(nx=8, nz=16, prognostic_condensate=True,
                                          qc_autoconv_crit=5.0e-4, cond_tau=tau))
            rng = np.random.default_rng(1)
            m.T_atm = 285.0 + 20.0 * np.linspace(1, 0, 16)[None, :] + rng.uniform(-2, 2, (8, 16))
            m.q = 0.95 * th.qsat(m.T_atm, m.p_mid[None, :])      # near-saturated start
            m.q_c[:] = 0.0
            dt = window / n
            cool = 30.0 / 86400.0                                # [K/s] steady 30 K/day cooling
            moist = 2.0e-8                                       # [kg/kg/s] steady moisture source
            for _ in range(n):
                m.T_atm = m.T_atm - cool * dt
                m.q = m._prognostic_microphysics(m.q + moist * dt, dt)
            if prev is not None:
                diffs.append(float(np.max(np.abs(m.T_atm - prev))))
            prev = m.T_atm.copy()
        return diffs                                             # |T(2n) - T(n)| per rung pair

    d_off = _ladder(0.0)
    d_on = _ladder(3.0e3)                                        # frac = dt/tau < 1 at every rung
    assert all(a < b for a, b in zip(d_on, d_off)), (
        f"finite cond_tau should tighten every rung (on {d_on} vs off {d_off})")
    assert d_on[-1] * 5.0 < d_off[-1], (
        f"at the finest rung the finite-tau operator should be far more converged "
        f"(on {d_on[-1]:.3e} K vs off {d_off[-1]:.3e} K)")
    assert d_on[-1] / d_on[-2] < 0.6, (
        f"finite cond_tau should converge ~first order (measured rung ratio {d_on[-1]/d_on[-2]:.2f})")


def test_cond_tau_is_fused_and_matches_generic_roundoff():
    """cond_tau>0 (constant tau) is now THREADED into the fused numba/device microphysics kernels
    (``_micro_fuse_salt_cond_params`` -> a ``cond_tau`` scalar the kernel applies as frac=min(dt/tau,1)),
    so with thermo_other_parallel ON the fused kernel ENGAGES and reproduces the plain generic
    ``_prognostic_microphysics`` to ROUND-OFF (~1e-13, NOT bit-identical -- esat's exp reassociates between
    numpy and numba's libm). SUPERSEDES the old route-around gate (cond_tau used to be EXCLUDED from the
    fuse and forced to the generic path); only the per-column ``cond_tau_salt`` stays generic-routed."""
    pytest.importorskip("numba")
    import flatworld.stage_ad_core as adc
    gen = _prog_core(cond_tau=3.0e3, precip_reevap=True)
    par = _prog_core(cond_tau=3.0e3, precip_reevap=True, thermo_other_parallel=True)
    Tg, qg, _ = _run(gen, 15)
    Tp, qp, _ = _run(par, 15)
    assert adc._MICRO_NUMBA_KERNEL is not None                    # the fused @njit kernel engaged (not bypassed)
    assert np.max(np.abs(Tg - Tp)) < 1e-9                         # fused tracks generic to round-off
    assert np.max(np.abs(qg - qp)) < 1e-11
    assert np.max(np.abs(gen.q_c - par.q_c)) < 1e-11


# ======================================================================================================
# ROADMAP GPU-7 code-side extensions: the salt/CCN-coupled tau(x) (cond_tau_salt) and the single-pass
# sub-grid PDF-tail condensation target (cond_pdf_tail). Same gate philosophy as above: default-OFF
# byte-identity, exact zero-points, real effect when armed, conservation under the new scaling, and the
# physical-anchor helper's magnitude.
# ======================================================================================================

def _salt_core(**kw):
    """Prognostic core with the #237 salt-CCN master ON (N_d defaults to the clean reference, so every
    salt limb is at its exact no-op zero-point unless a loaded N_d is prescribed)."""
    kw.setdefault("salt_ccn", True)
    return _prog_core(**kw)


def _loaded_band_kw(nd=1.0e4):
    """A band-masked LOADED N_d field: continental-polluted nd inside the default 40-60 deg window,
    the clean reference outside -- so a run has both loaded and clean columns to compare."""
    return dict(salt_ccn=True, salt_ccn_nd=nd, salt_ccn_band_only=True)


def test_cond_tau_salt_config_validation():
    """cond_tau_salt needs the N_d field (salt_ccn) and a clean-air reference tau (cond_tau > 0);
    cond_pdf_tail is single-pass-only (the prognostic PDF source is already tail-based) and needs the
    PDF cover it unifies with (qc_pdf_cloud)."""
    with pytest.raises(ValueError, match="cond_tau_salt requires salt_ccn"):
        StageADCore(StageADConfig(nx=8, nz=12, cond_tau=3.0e3, cond_tau_salt=True))
    with pytest.raises(ValueError, match="cond_tau_salt requires cond_tau > 0"):
        StageADCore(StageADConfig(nx=8, nz=12, prognostic_condensate=True, salt_ccn=True,
                                  cond_tau=0.0, cond_tau_salt=True))
    with pytest.raises(ValueError, match="SINGLE-PASS core only"):
        StageADCore(StageADConfig(nx=8, nz=12, prognostic_condensate=True, cond_pdf_tail=True))
    with pytest.raises(ValueError, match="cond_pdf_tail requires qc_pdf_cloud"):
        StageADCore(StageADConfig(nx=8, nz=12, qc_pdf_cloud=False, cond_pdf_tail=True))


def test_cond_tau_salt_clean_field_is_bit_identical():
    """The #237 clean zero-point, inherited: with N_d == N_ref everywhere, (N_ref/N_d)^gamma == 1.0
    exactly, so the salt-coupled tau(x) run is BIT-identical to the constant-tau run (same salt_ccn
    master state on both sides, so the comparison isolates the tau coupling)."""
    a = _salt_core(cond_tau=3.0e3);                     Ta, qa, Tsa = _run(a, 20)
    b = _salt_core(cond_tau=3.0e3, cond_tau_salt=True); Tb, qb, Tsb = _run(b, 20)
    assert np.array_equal(Ta, Tb) and np.array_equal(qa, qb) and np.array_equal(Tsa, Tsb)
    assert np.array_equal(a.q_c, b.q_c)


def test_cond_tau_salt_default_off_is_byte_identical():
    """cond_tau_salt=False (default) with a LOADED band field: bit-identical to a run that never set the
    flag -- the default-OFF guarantee (the loaded N_d exercises the other salt limbs on both sides)."""
    a = _prog_core(cond_tau=3.0e3, **_loaded_band_kw());                      Ta, qa, _ = _run(a, 15)
    b = _prog_core(cond_tau=3.0e3, cond_tau_salt=False, **_loaded_band_kw()); Tb, qb, _ = _run(b, 15)
    assert np.array_equal(Ta, Tb) and np.array_equal(qa, qb)
    assert np.array_equal(a.q_c, b.q_c)


def _operator_once(core, dt=1800.0):
    """One isolated _prognostic_microphysics call on a controlled near-saturated cooling state (the
    operator-level pattern of the dt-convergence ladder above); returns the resulting q_c."""
    import flatworld.thermo as th
    rng = np.random.default_rng(7)
    core.T_atm = 285.0 + 20.0 * np.linspace(1, 0, core.cfg.nz)[None, :] \
        + rng.uniform(-1, 1, (core.cfg.nx, core.cfg.nz))
    core.q = 1.05 * th.qsat(core.T_atm, core.p_mid[None, :])     # uniformly supersaturated -> cond > 0
    core.q_c[:] = 0.0
    core._prognostic_microphysics(core.q.copy(), dt)
    return np.array(core.q_c, float).copy()


def test_cond_tau_salt_loaded_columns_relax_faster_clean_columns_bitwise_unchanged():
    """The point of the coupling: where the salt loads N_d, tau(x) shrinks and frac grows, so one
    operator call condenses MORE toward the equilibrium than the constant clean-air tau does; where the
    field is clean (N_d == N_ref) the columns are bit-identical to the constant-tau operator."""
    const = _prog_core(cond_tau=1.0e5, **_loaded_band_kw())
    salt = _prog_core(cond_tau=1.0e5, cond_tau_salt=True, **_loaded_band_kw())
    qc_const = _operator_once(const)
    qc_salt = _operator_once(salt)
    loaded = np.asarray(const.N_d, float) > const.cfg.salt_ccn_nd_ref        # the band-masked columns
    assert loaded.any() and (~loaded).any(), "fixture must span loaded AND clean columns"
    assert np.array_equal(qc_const[~loaded], qc_salt[~loaded])               # clean columns: exact no-op
    assert np.all(qc_salt[loaded] >= qc_const[loaded])                       # loaded: faster relaxation
    assert np.max(qc_salt[loaded] - qc_const[loaded]) > 0.0                  # ...strictly somewhere


def test_cond_tau_salt_exponent_orders_the_relaxation():
    """gamma orders tau(x) at a loaded N_d: gamma=1 (fixed-r_bar) shrinks tau more than the default 2/3
    (fixed-LWC), which shrinks it more than the constant tau -- so one operator call condenses in that
    order. (N_d/N_ref = 100 here: frac = dt/tau stays < 1 for all three at dt=1800, tau_ref=1e5.)"""
    base = dict(cond_tau=1.0e5, salt_ccn=True, salt_ccn_nd=1.0e4, salt_ccn_band_only=True)
    qc_const = _operator_once(_prog_core(**base))
    qc_g23 = _operator_once(_prog_core(cond_tau_salt=True, **base))                          # gamma=2/3
    qc_g1 = _operator_once(_prog_core(cond_tau_salt=True, cond_tau_salt_exp=1.0, **base))    # gamma=1
    core = _prog_core(**base)
    loaded = np.asarray(core.N_d, float) > core.cfg.salt_ccn_nd_ref
    assert np.all(qc_g1[loaded] >= qc_g23[loaded]) and np.max(qc_g1[loaded] - qc_g23[loaded]) > 0.0
    assert np.all(qc_g23[loaded] >= qc_const[loaded]) and np.max(qc_g23[loaded] - qc_const[loaded]) > 0.0


def test_cond_tau_salt_water_budget_closes():
    """Conservation under the SPATIALLY-VARYING frac: the exchange is a per-cell q<->q_c conjugate pair
    (the same per-column frac scales cond and evap_demand), so the reservoir budget
    d/dt INTEGRAL (q + q_c) dm = E - P still closes to round-off with a loaded band field."""
    m = StageADCore(StageADConfig(nx=20, nz=18, theta_max_deg=55.0, prognostic_condensate=True,
                                  qc_autoconv_crit=5.0e-4, cond_tau=3.0e3, cond_tau_salt=True,
                                  salt_ccn=True, salt_ccn_nd=1.0e4, salt_ccn_band_only=True))
    m.run_dynamics = False
    m.couple_surface = False
    m.T_surf[:] = 292.0
    for _ in range(20):
        m.step()
    m.reset_water_budget()
    tw0 = float(np.sum((m.q + m.q_c) * m.dm[None, :]) * m.dx)
    for _ in range(40):
        m.step()
    tw1 = float(np.sum((m.q + m.q_c) * m.dm[None, :]) * m.dx)
    wb = m.water_conservation()
    lhs = tw1 - tw0
    rhs = wb["cum_evap"] - wb["cum_precip"]
    assert abs(lhs - rhs) / (abs(rhs) + 1e-12) < 1e-6
    assert np.all(m.q_c >= 0.0)


def test_cond_tau_salt_operator_stays_dt_convergent():
    """Two-rung dt sanity (new-physics gate): the finite-tau operator's O(dt) convergence survives the
    spatially-varying tau(x) -- rung-to-rung differences keep shrinking on a refinement ladder with a
    loaded band field (a term that only acts at one dt would be a discretization artifact)."""
    import flatworld.thermo as th

    def _ladder(rungs=(8, 16, 32), window=3600.0):
        diffs, prev = [], None
        for n in rungs:
            # cond_tau=3e4 keeps the LOADED tau (3e4/(100)^(2/3) ~ 1.4e3 s) above every rung's dt, so
            # frac < 1 everywhere on the ladder (a frac saturated at 1 would be the instantaneous,
            # sub-first-order regime and defeat the convergence measurement).
            m = StageADCore(StageADConfig(nx=8, nz=16, prognostic_condensate=True,
                                          qc_autoconv_crit=5.0e-4, cond_tau=3.0e4, cond_tau_salt=True,
                                          salt_ccn=True, salt_ccn_nd=1.0e4, salt_ccn_band_only=True,
                                          salt_ccn_band_theta_lo=20.0, salt_ccn_band_theta_hi=60.0))
            rng = np.random.default_rng(1)
            m.T_atm = 285.0 + 20.0 * np.linspace(1, 0, 16)[None, :] + rng.uniform(-2, 2, (8, 16))
            m.q = 0.95 * th.qsat(m.T_atm, m.p_mid[None, :])
            m.q_c[:] = 0.0
            dt = window / n
            cool = 30.0 / 86400.0
            moist = 2.0e-8
            for _ in range(n):
                m.T_atm = m.T_atm - cool * dt
                m.q = m._prognostic_microphysics(m.q + moist * dt, dt)
            if prev is not None:
                diffs.append(float(np.max(np.abs(m.T_atm - prev))))
            prev = m.T_atm.copy()
        return diffs

    d = _ladder()
    assert d[1] < d[0], f"salt-coupled tau(x) operator must keep converging under dt refinement ({d})"
    assert d[1] / d[0] < 0.7, f"expected ~first-order rung shrinkage, measured ratio {d[1]/d[0]:.2f}"


def test_cond_tau_salt_routes_around_the_fused_kernel():
    """The salt coupling inherits cond_tau>0's route-to-generic (and salt_ccn_autoconv's): with the
    numba thermo path armed the result must equal the plain generic path bit-for-bit."""
    pytest.importorskip("numba")
    gen = _prog_core(cond_tau=3.0e3, cond_tau_salt=True, precip_reevap=True, **_loaded_band_kw())
    par = _prog_core(cond_tau=3.0e3, cond_tau_salt=True, precip_reevap=True,
                     thermo_other_parallel=True, **_loaded_band_kw())
    Tg, qg, _ = _run(gen, 15)
    Tp, qp, _ = _run(par, 15)
    assert np.array_equal(Tg, Tp) and np.array_equal(qg, qp)
    assert np.array_equal(gen.q_c, par.q_c)


def test_phase_relaxation_tau_magnitudes():
    """The physical anchor (thermo.phase_relaxation_tau = 1/(4 pi D N r_bar)): a few seconds at the
    clean-maritime 100 cm^-3, tens of seconds in very clean air, sub-second when salt-loaded --
    monotone decreasing in N_d. (This is what makes tau=0 the sub-physical fast extreme at the
    canonical 1 s thermo substep.)"""
    import flatworld.thermo as th
    t100 = float(th.phase_relaxation_tau(100.0))
    t10 = float(th.phase_relaxation_tau(10.0))
    t1e4 = float(th.phase_relaxation_tau(1.0e4))
    assert 1.0 < t100 < 10.0, f"expected a few seconds at 100 cm^-3, got {t100:.2f} s"
    assert 10.0 < t10 < 100.0, f"expected tens of seconds at 10 cm^-3, got {t10:.2f} s"
    assert t1e4 < 0.1 < t100 < t10                       # monotone; salt-loaded is sub-physical-fast
    assert abs(t100 * 100.0 - t10 * 10.0) / (t10 * 10.0) < 1e-12   # exact 1/N scaling at fixed r_bar


# ---- the single-pass sub-grid PDF-tail condensation target (cond_pdf_tail) --------------------------

def _tail_core(q_frac, cond_pdf_tail, **kw):
    """Single-pass core prepared for an isolated condensation comparison: dynamics off, convection off
    (conv_max_sweeps=0), COLD fixed surface (qsat(T_s) < q so evaporation is exactly zero), uniform
    q = q_frac * qsat, and a 1 s step so the pre-condensation radiative drift is negligible (the Q1
    regime set by q_frac still holds at the condensation line). The column is set to a controlled WARM
    profile (265..290 K) so qsat >> the PDF width floor everywhere -- on the default initial state the
    ~130 K test-grid top layers have 0.2*qsat < CLOUD_PDF_WIDTH_FLOOR, and a floored Delta makes even
    q = 1.5*qsat a PARTIAL-cloud cell (Q1 << 1), which is floor behavior, not the closure's."""
    import flatworld.thermo as th
    m = _core(cond_pdf_tail=cond_pdf_tail, conv_max_sweeps=0, dt=1.0, **kw)
    m.run_dynamics = False
    m.couple_surface = False
    m.T_surf[:] = 250.0
    m.T_atm = np.tile(265.0 + 25.0 * np.linspace(0.0, 1.0, m.cfg.nz), (m.cfg.nx, 1))
    m.q = q_frac * th.qsat(m.T_atm, m.p_mid[None, :])
    return m


def test_cond_pdf_tail_reduces_to_grid_mean_when_overcast():
    """Q1 >= 1 (the whole sub-grid distribution saturated): the tail first moment IS the grid-mean
    excess (qc_eq = q_t - qsat), so the flag must be BIT-identical there -- the consistent-continuation
    claim, locked at the bit level. q = 1.5*qsat keeps every cell overcast through one step's radiative
    drift (rh_crit=0.8 => the overcast threshold is q >= 1.2*qsat)."""
    off = _tail_core(1.5, cond_pdf_tail=False); off.step()
    on = _tail_core(1.5, cond_pdf_tail=True);   on.step()
    assert np.array_equal(np.array(off.T_atm), np.array(on.T_atm))
    assert np.array_equal(np.array(off.q), np.array(on.q))
    assert np.array_equal(np.array(off._precip), np.array(on._precip))


def test_cond_pdf_tail_drizzles_where_the_grid_mean_is_silent():
    """The unification's payoff: at grid-mean RH 0.9 (sub-saturated mean, but Q1 = -0.5 > -1 so the PDF
    has a saturated tail and the radiation already reports cover there) the grid-mean adjustment
    condenses NOTHING while the tail target drizzles the tail excess -- cover and condensate now agree
    that the box is partly cloudy. (RH 0.85 leaves >1 K of radiative-drift margin below saturation.)"""
    off = _tail_core(0.85, cond_pdf_tail=False); off.step()
    on = _tail_core(0.85, cond_pdf_tail=True);   on.step()
    assert float(np.sum(np.array(off._precip))) == 0.0, "grid-mean path must be silent at RH 0.85"
    assert float(np.sum(np.array(on._precip))) > 0.0, "the PDF tail must drizzle at RH 0.85"


def test_cond_pdf_tail_cond_tau_still_sets_the_rate():
    """The clean split the roadmap names: the PDF tail sets the TARGET, cond_tau the RATE -- a partial
    frac scales the first-step tail drizzle down by exactly min(dt/tau,1) relative to instantaneous."""
    inst = _tail_core(0.9, cond_pdf_tail=True);                 inst.step()
    slow = _tail_core(0.9, cond_pdf_tail=True, cond_tau=1.0e6); slow.step()
    p_inst = float(np.sum(np.array(inst._precip)))
    p_slow = float(np.sum(np.array(slow._precip)))
    assert 0.0 < p_slow < p_inst
    frac = inst.cfg.dt / 1.0e6                                   # thermo_n_sub=1 -> dt_sub = cfg.dt
    assert abs(p_slow - frac * p_inst) / (frac * p_inst) < 1e-9  # linear first step from q_c-free init
