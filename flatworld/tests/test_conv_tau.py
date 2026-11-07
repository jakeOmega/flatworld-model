"""Regression gate for the Betts-Miller finite-tau convective relaxation (cfg.conv_tau).

Locks the three load-bearing claims of the conv_tau fix (PR #214 review): (1) default-OFF is byte-identical
and never touches the BM code, (2) tau small enough that frac=1 recovers the instantaneous scheme, (3) the
fractional blend conserves column enthalpy and water exactly (the property the fix rests on). CPU/fp64,
no GPU/donor needed.
"""
import numpy as np
import pytest

from flatworld.stage_ad_core import StageADCore, StageADConfig, moist_convective_adjust


def _core(**kw):
    return StageADCore(StageADConfig(nx=24, nz=16, **kw))


def _run(core, n):
    for _ in range(n):
        core.step()
    return np.array(core.T_atm, float).copy(), np.array(core.q, float).copy(), np.array(core.T_surf, float).copy()


def test_conv_tau_off_is_inert_and_byte_identical():
    """conv_tau=0 (default): the BM snapshot buffers are never allocated, and the state is bit-identical to
    a run that never set conv_tau -- the default-OFF guarantee against future drift."""
    a = _core(conv_tau=0.0); Ta, qa, Tsa = _run(a, 20)
    b = _core();             Tb, qb, Tsb = _run(b, 20)   # default conv_tau is 0.0
    assert getattr(a, "_conv_Tpre", None) is None, "OFF path must not allocate the BM snapshot buffer"
    assert np.array_equal(Ta, Tb) and np.array_equal(qa, qb) and np.array_equal(Tsa, Tsb)


def test_conv_tau_frac1_recovers_instantaneous():
    """tau <= thermo_dt => frac = min(dt/tau,1) = 1 => the blend is the FULL adjustment, recovering the
    shipped instantaneous scheme. (Default thermo_n_sub=1 => thermo_dt = dt_dyn, so a tiny tau gives frac=1.)
    Equal to round-off (the conv_tau>0 path does a -=/*=/+= round-trip the OFF path skips)."""
    off = _core(conv_tau=0.0);  To, qo, Tso = _run(off, 20)
    on  = _core(conv_tau=1e-3); Tn, qn, Tsn = _run(on, 20)   # frac saturates at 1
    assert np.allclose(To, Tn, atol=1e-9, rtol=0) and np.allclose(qo, qn, atol=1e-12, rtol=0)
    assert np.allclose(Tso, Tsn, atol=1e-9, rtol=0)


def test_conv_tau_has_a_real_effect_when_partial():
    """A finite tau >> thermo_dt (frac<1) must MOVE the climate away from the instantaneous (conv_tau=0)
    result -- otherwise the blend would be a silent no-op."""
    off = _core(conv_tau=0.0);    To, _, _ = _run(off, 30)
    on  = _core(conv_tau=1.0e5);  Tn, _, _ = _run(on, 30)    # frac = dt_dyn/1e5 << 1
    assert np.max(np.abs(To - Tn)) > 1e-3, "conv_tau>0 should change the trajectory (frac<1 is partial)"


def test_conv_tau_blend_conserves_enthalpy_and_water():
    """The load-bearing conservation claim: moist_convective_adjust conserves column enthalpy (sum dm*T) and
    water (sum dm*q) exactly, and the linear blend X_pre + frac*(X_adj - X_pre) -- one scalar frac applied
    uniformly across layers -- preserves any linear column integral. Build a superadiabatic, moist column,
    adjust it, blend at several frac, and assert both integrals are unchanged to round-off."""
    c = _core()
    dm = np.array(c.dm, float)                                  # (nz,) layer masses
    z = np.array(c._hydrostatic_z(c.T_atm), float)              # (nx, nz)
    p_mid = np.array(c.p_mid, float)
    nx, nz = c.cfg.nx, c.cfg.nz
    rng = np.random.default_rng(0)
    # a strongly super-adiabatic, moist column (so the adjustment actually fires and moves T and q)
    T_pre = 300.0 - 6.0e-3 * (z.max() - z) + rng.uniform(-8.0, 8.0, size=(nx, nz))
    q_pre = np.clip(0.01 * np.exp(-(z.max() - z) / 3000.0) + rng.uniform(-1e-3, 3e-3, size=(nx, nz)), 0.0, 0.05)
    Ipre_T = (dm[None] * T_pre).sum(axis=1)
    Ipre_q = (dm[None] * q_pre).sum(axis=1)
    T_adj = T_pre.copy(); q_adj = q_pre.copy()
    moist_convective_adjust(T_adj, dm, z, p_mid, 8, q=q_adj, closure="rh_detrain")
    # the adjustment itself conserves both integrals
    assert np.allclose((dm[None] * T_adj).sum(axis=1), Ipre_T, atol=1e-6, rtol=0)
    assert np.allclose((dm[None] * q_adj).sum(axis=1), Ipre_q, atol=1e-10, rtol=0)
    assert np.max(np.abs(T_adj - T_pre)) > 1.0 and np.min(q_adj) >= 0.0   # it actually moved T; no neg water
    # the blend preserves the integrals at every frac, and never makes negative water (convex combo)
    for frac in (0.01, 0.25, 0.5, 0.83, 1.0):
        T_bm = T_pre + frac * (T_adj - T_pre)
        q_bm = q_pre + frac * (q_adj - q_pre)
        assert np.allclose((dm[None] * T_bm).sum(axis=1), Ipre_T, atol=1e-6, rtol=0)
        assert np.allclose((dm[None] * q_bm).sum(axis=1), Ipre_q, atol=1e-10, rtol=0)
        assert q_bm.min() >= 0.0
