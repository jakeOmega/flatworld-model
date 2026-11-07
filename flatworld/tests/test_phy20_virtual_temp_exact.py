"""ROADMAP **PHY-20** -- the exact SPECIFIC-HUMIDITY density temperature, behind ``virtual_temp_exact``.

Terminology, on first use.
  q_v   VAPOUR specific humidity [kg/kg] -- mass of vapour per unit mass of TOTAL (moist) air.
  r_v   vapour MIXING RATIO [kg/kg]      -- mass of vapour per unit mass of DRY air; r = q/(1 - q_v - q_c).
  q_c   condensed-water (cloud + rain) specific humidity [kg/kg]; "condensate loading" is its weight.
  eps   ``constants.EPS_MW`` = R_dry/R_vap ~ 0.622, the molar-mass ratio; 1/eps - 1 ~ 0.608.
  T_v / T_rho  virtual / density temperature [K]: the temperature dry air would need to match the moist
        parcel's density at the same pressure.
  AD core  the fused anelastic dynamics core (``stage_ad_core.StageADCore``), the default backbone.

THE DEFECT. ``thermo.virtual_temperature`` computes ``T (1 + q/eps)/(1 + q + q_c)``. That is the EXACT
density temperature for MIXING RATIOS -- but this model's humidities are SPECIFIC (``thermo.qsat``
returns ``eps e/(p - (1-eps) e)`` = rho_v/rho). For specific humidities the exact form is
``T (1 + (1/eps - 1) q_v - q_c)``. The two are the SAME function of the same parcel reached through
different variables, so the default path is the right function at the wrong variable and always
UNDER-states the vapour buoyancy.

This file is the merge gate for the fix: default OFF, byte-identical OFF, the identity and the gap
pinned symbolically (sympy lifted with ``sp.lambdify``, never a parallel hand-estimate), the flag
reaching every AD-core buoyancy branch, the fingerprint recording it, the documented warm-start route
working on a BUILT core, and an ON smoke.
"""
import hashlib
import json
import warnings

import numpy as np
import pytest

from flatworld import constants as C
from flatworld import thermo as th
from flatworld.stage_ad_core import StageADCore, StageADConfig

EPS = float(C.EPS_MW)
CVAP = 1.0 / EPS - 1.0                 # ~0.608, the exact specific-humidity vapour coefficient


# --------------------------------------------------------------------------- symbolic ground truth
@pytest.fixture(scope="module")
def sym():
    """The two closed forms and their difference, lifted out of sympy with ``sp.lambdify``.

    Nothing in this file compares the code against a remembered number: the reference values come from
    the symbolic expressions below, evaluated here (OPERATIONS.md 3.5, the latent-coefficient-bug rule).
    """
    import sympy as sp
    T, qv, qc = sp.symbols("T q_v q_c", positive=True)
    e = sp.Float(EPS)
    ratio = T * (1 + qv / e) / (1 + qv + qc)                  # exact for MIXING ratios
    exact = T * (1 + (1 / e - 1) * qv - qc)                   # exact for SPECIFIC humidities
    # the gap at q_c = 0 simplifies to T (1/eps - 1) q^2/(1 + q): O(q^2), so the forms agree to O(q)
    gap_closed = sp.simplify((exact - ratio).subs(qc, 0))
    assert sp.simplify(gap_closed - T * (1 / e - 1) * qv**2 / (1 + qv)) == 0
    return {
        "ratio": sp.lambdify((T, qv, qc), ratio, "numpy"),
        "exact": sp.lambdify((T, qv, qc), exact, "numpy"),
        "gap": sp.lambdify((T, qv), gap_closed, "numpy"),
    }


def test_the_exact_branch_matches_the_symbolic_closed_form(sym):
    """Units: [K] * (1 + [kg/kg]*[-] - [kg/kg]) = [K]. Checked against sympy, not against arithmetic
    retyped from the docstring."""
    rng = np.random.default_rng(3)
    T = rng.uniform(220.0, 360.0, 500)
    q_v = rng.uniform(1e-5, 0.35, 500)
    q_c = rng.uniform(0.0, 8e-3, 500)
    got = np.asarray(th.virtual_temperature(T, q_v, q_c, exact=True), float)
    assert np.allclose(got, sym["exact"](T, q_v, q_c), rtol=1e-14, atol=0.0)
    # ... and the OFF branch is still exactly the ratio form (no accidental drift in the default)
    off = np.asarray(th.virtual_temperature(T, q_v, q_c), float)
    assert np.allclose(off, sym["ratio"](T, q_v, q_c), rtol=1e-14, atol=0.0)


def test_the_mixing_ratio_identity_maps_one_form_onto_the_other_exactly(sym):
    """THE limiting case, and the whole of PHY-20: the ratio form fed the parcel's MIXING ratios equals
    the exact form fed the same parcel's SPECIFIC humidities, identically -- with condensate as well as
    without. So the linear form is not a dilute approximation of the ratio form; they are one function."""
    for T in (250.0, 300.0, 330.0, 360.0):
        for q_v in (1e-4, 0.01, 0.05, 0.10, 0.2964):
            for q_c in (0.0, 1e-3, 5e-3):
                q_d = 1.0 - q_v - q_c                          # dry-air mass fraction
                r_v, r_c = q_v / q_d, q_c / q_d                # the SAME parcel, in mixing ratios
                lhs = float(th.virtual_temperature(T, r_v, r_c))           # ratio form at r
                rhs = float(th.virtual_temperature(T, q_v, q_c, exact=True))
                assert lhs == pytest.approx(rhs, rel=1e-13)
                assert lhs == pytest.approx(float(sym["exact"](T, q_v, q_c)), rel=1e-13)


def test_the_two_forms_agree_at_q_zero_and_part_only_at_second_order(sym):
    """q -> 0: both forms are exactly T (a dry parcel has no vapour buoyancy, whatever the variable), and
    they share their first derivative -- the gap is O(q^2), which is why the defect hid."""
    for T in (280.0, 330.0):
        assert float(th.virtual_temperature(T, 0.0, exact=True)) == T
        assert float(th.virtual_temperature(T, 0.0)) == T
        # first-order agreement: the gap falls as q^2, so halving q quarters it (ratio -> 4)
        g1 = float(sym["gap"](T, 1e-3))
        g2 = float(sym["gap"](T, 5e-4))
        assert g1 / g2 == pytest.approx(4.0, rel=2e-3)


def test_the_gap_is_the_documented_size_and_always_an_understatement(sym):
    """The two humidities ROADMAP PHY-20 and PHY19_AXIS_CONVECTION 2.6 quote, cross-checked against the
    gates already in ``test_phy19_axis_concentration.py`` -- same numbers, now measured on the MODEL
    function rather than on the read-only instrument."""
    T = 330.0
    for q_v, expect in ((0.10, 1.82), (0.2964, 13.60)):
        gap = float(th.virtual_temperature(T, q_v, exact=True) - th.virtual_temperature(T, q_v))
        assert gap == pytest.approx(float(sym["gap"](T, q_v)), rel=1e-12)
        assert gap == pytest.approx(expect, abs=0.01)
    # it is about 9 % of the whole vapour increment at q = 0.10 ...
    incr = float(th.virtual_temperature(T, 0.10, exact=True)) - T
    assert float(sym["gap"](T, 0.10)) / incr == pytest.approx(0.091, abs=0.005)
    # ... and it is strictly positive everywhere (the default path is ALWAYS the less buoyant one)
    rng = np.random.default_rng(11)
    q = rng.uniform(1e-6, 0.4, 2000)
    Ts = rng.uniform(200.0, 400.0, 2000)
    d = np.asarray(th.virtual_temperature(Ts, q, exact=True), float) - np.asarray(
        th.virtual_temperature(Ts, q), float)
    assert np.all(d > 0.0)


def test_the_model_function_agrees_with_the_readonly_phy19_instrument():
    """``phy19_axis_concentration.density_temperature`` measured the gap; this function now fixes it.
    They must be the same two formulas, or the note's numbers stop describing the code."""
    import importlib.util
    import pathlib
    p = pathlib.Path(__file__).resolve().parents[2] / "phy19_axis_concentration.py"
    spec = importlib.util.spec_from_file_location("_phy19_instr", p)
    phy19 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phy19)
    for T in (300.0, 330.0):
        for q in (0.02, 0.10, 0.2964):
            for qc in (0.0, 2e-3):
                assert float(th.virtual_temperature(T, q, qc, exact=True)) == pytest.approx(
                    float(phy19.density_temperature(T, q, qc, form="exact")), rel=1e-15)


def test_condensate_loading_is_the_same_minus_T_q_cond_under_either_form():
    """The loading term is ``-T q_cond`` exactly, whichever vapour form is selected -- so PHY-19 step 2
    and PHY-20 are separable, and ``density_temperature(..., exact=True)`` is the FULLY exact form."""
    rng = np.random.default_rng(5)
    T = rng.uniform(250.0, 340.0, 300)
    q_v = rng.uniform(1e-4, 0.15, 300)
    q_cond = rng.uniform(1e-5, 6e-3, 300)
    for exact in (False, True):
        dry = np.asarray(th.density_temperature(T, q_v, 0.0, exact=exact), float)
        wet = np.asarray(th.density_temperature(T, q_v, q_cond, exact=exact), float)
        assert np.allclose(dry - wet, T * q_cond, rtol=1e-12, atol=1e-12)
    # with the flag ON the composition IS the single closed form T(1 + 0.608 q_v - q_cond)
    assert np.allclose(np.asarray(th.density_temperature(T, q_v, q_cond, exact=True), float),
                       T * (1.0 + CVAP * q_v - q_cond), rtol=1e-14, atol=0.0)


# ------------------------------------------------------------------------------ the flag on the core
def _state_hash(core):
    h = hashlib.sha256()
    for name in ("T_atm", "q", "q_c", "q_r", "b", "psi", "omega", "T_surf", "W_surf"):
        v = getattr(core, name, None)
        if v is None:
            continue
        h.update(name.encode())
        h.update(np.ascontiguousarray(np.asarray(v, dtype=np.float64)).tobytes())
    return h.hexdigest()


def _stepped_hash(flag, steps=5, seed=11, **kw):
    """Step a small toy world and hash the WHOLE prognostic state (not just T)."""
    cfg_kw = dict(nx=12, nz=10, theta_max_deg=80.0, **kw)
    if flag is not None:
        cfg_kw["virtual_temp_exact"] = flag
    rng = np.random.default_rng(seed)
    core = StageADCore(StageADConfig(**cfg_kw))
    core.T_atm = core.T_atm + 0.7 * rng.standard_normal(core.T_atm.shape)
    core.q = np.clip(core.q + 1e-4 * rng.standard_normal(core.q.shape), 0.0, 0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(steps):
            core.step()
    return _state_hash(core)


def test_the_flag_defaults_off_and_off_is_byte_identical(monkeypatch):
    """The merge bar. The flag defaults False, and a five-step run with it ABSENT from the config is
    bit-for-bit a run with it explicitly False -- over the whole prognostic state, not a visual check.
    (The cross-checkout half of the claim -- identical to ``origin/main`` -- is
    ``cosmology_models/phy20_virtual_temp_exact_off_control.py``, run in both worktrees.)"""
    monkeypatch.delenv("COSMO_VIRTUAL_TEMP_EXACT", raising=False)
    assert StageADConfig(nx=8, nz=6).virtual_temp_exact is False
    assert _stepped_hash(None) == _stepped_hash(False)


def test_the_env_key_reaches_the_config(monkeypatch):
    monkeypatch.setenv("COSMO_VIRTUAL_TEMP_EXACT", "1")
    assert StageADConfig(nx=8, nz=6).virtual_temp_exact is True
    monkeypatch.setenv("COSMO_VIRTUAL_TEMP_EXACT", "0")
    assert StageADConfig(nx=8, nz=6).virtual_temp_exact is False


def test_the_fingerprint_records_the_key():
    """A checkpoint must not be able to hide which side of the switch it ran on. The fingerprint's
    physics set is INVERTED (everything not named volatile is physics), so this rides that for free --
    which is exactly why it needs a test rather than a claim."""
    core = StageADCore(StageADConfig(nx=12, nz=8, virtual_temp_exact=True))
    fp = json.loads(core.fingerprint())
    assert fp["config"]["virtual_temp_exact"] is True
    from flatworld import _ckpt_fingerprint as ckfp
    assert "virtual_temp_exact" not in ckfp.VOLATILE_FIELDS      # i.e. a difference is REPORTED


def _buoyancy(virtual_temp_exact, **kw):
    """One ``_update_buoyancy`` call on a seeded moist column; returns b (nz, nx) as a host array."""
    rng = np.random.default_rng(4)
    core = StageADCore(StageADConfig(nx=16, nz=12, theta_max_deg=80.0,
                                     virtual_temp_exact=virtual_temp_exact, **kw))
    core.T_atm = core.T_atm + 2.0 * rng.standard_normal(core.T_atm.shape)
    core.q = np.clip(np.abs(rng.standard_normal(core.q.shape)) * 0.05, 0.0, 0.3)
    if getattr(core, "q_c", None) is not None:
        core.q_c = np.abs(rng.standard_normal(np.shape(core.q_c))) * 1.0e-3
    if getattr(core, "q_r", None) is not None:
        core.q_r = np.abs(rng.standard_normal(np.shape(core.q_r))) * 3.0e-4
    core._z = core._hydrostatic_z(core.T_atm)
    core._update_buoyancy()
    return np.asarray(core.b, dtype=np.float64).copy()


def test_the_flag_reaches_the_vapour_only_buoyancy_branch_and_raises_it():
    """ON must be NON-VACUOUS in the core, and in the physically-forced direction: the exact form is the
    MORE buoyant one, so b (which is g(T_rho - T_ref)/T_ref) can only go up."""
    off, on = _buoyancy(False), _buoyancy(True)
    assert np.max(np.abs(on - off)) > 1e-6, "the flag did not reach _update_buoyancy"
    assert np.all(on >= off - 1e-15), "the exact form must never LOWER the buoyancy"


def test_the_flag_reaches_the_condensate_loading_branch_too():
    """The loading branch calls ``density_temperature``; the flag must be threaded there as well, or ON
    would silently revert to the ratio form whenever PHY-19 step 2 is enabled."""
    kw = dict(prognostic_condensate=True, prognostic_rain=True, condensate_loading=True)
    off, on = _buoyancy(False, **kw), _buoyancy(True, **kw)
    assert np.max(np.abs(on - off)) > 1e-6
    assert np.all(on >= off - 1e-15)


def test_the_flag_is_inert_with_vapour_buoyancy_off():
    """Orthogonality: both forms are exactly T at q_v = 0, and with ``virtual_buoyancy`` OFF no vapour
    term is formed at all -- so the flag must be bit-for-bit a no-op, loading on or off."""
    for kw in ({}, dict(prognostic_condensate=True, prognostic_rain=True, condensate_loading=True)):
        off = _buoyancy(False, virtual_buoyancy=False, **kw)
        on = _buoyancy(True, virtual_buoyancy=False, **kw)
        assert np.array_equal(off, on)


def test_a_post_build_flag_change_takes_effect():
    """The documented warm-start route: a COSMO_* key the run17 checkpoint never recorded is (correctly)
    refused by the checkpoint world guard, so a canonical warm start does
    ``core = build_core_with_world(...)`` then ``core.cfg.virtual_temp_exact = True``. That works only
    because ``_update_buoyancy`` RE-READS the flag every call -- nothing is baked at construction."""
    rng = np.random.default_rng(4)
    core = StageADCore(StageADConfig(nx=16, nz=12, theta_max_deg=80.0))
    assert core.cfg.virtual_temp_exact is False
    core.T_atm = core.T_atm + 2.0 * rng.standard_normal(core.T_atm.shape)
    core.q = np.clip(np.abs(rng.standard_normal(core.q.shape)) * 0.05, 0.0, 0.3)
    core._z = core._hydrostatic_z(core.T_atm)
    core._update_buoyancy()
    b_off = np.asarray(core.b, float).copy()
    core.cfg.virtual_temp_exact = True            # exactly the line the config comment documents
    core._update_buoyancy()
    b_on = np.asarray(core.b, float).copy()
    assert np.max(np.abs(b_on - b_off)) > 1e-6 and np.all(b_on >= b_off - 1e-15)


def test_the_on_path_smoke_runs_and_stays_finite():
    """ON smoke: the flag is not just a formula, it feeds the dynamics -- five coupled steps must run,
    stay finite, and stay climatically sane (the column must not detonate)."""
    rng = np.random.default_rng(2)
    core = StageADCore(StageADConfig(nx=16, nz=12, theta_max_deg=80.0, virtual_temp_exact=True,
                                     prognostic_condensate=True, clouds=True))
    core.T_atm = core.T_atm + 0.7 * rng.standard_normal(core.T_atm.shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(5):
            core.step()
    for name in ("T_atm", "q", "b", "psi", "omega", "T_surf"):
        v = np.asarray(getattr(core, name), float)
        assert np.all(np.isfinite(v)), f"{name} went non-finite with virtual_temp_exact ON"
    Tc = np.asarray(core.T_atm, float) - C.T_FREEZE
    assert -150.0 < float(Tc.mean()) < 200.0, f"insane mean T_atm {float(Tc.mean()):.1f} C"


def test_the_symbolic_crosscheck_covers_the_exact_form():
    """``plume_floft.symbolic_crosscheck`` is the module's standing sympy-vs-numeric guard; the new
    branch has to be in it, not only in this file."""
    from flatworld import plume_floft as pf
    out = pf.symbolic_crosscheck()
    assert out["virtual_temperature_exact"] < 1e-12
    assert out["density_temperature_exact"] < 1e-12
