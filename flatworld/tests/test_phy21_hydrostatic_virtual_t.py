"""ROADMAP **PHY-21** -- the density temperature in the hypsometric bridge, behind ``hydrostatic_virtual_t``.

Terminology, on first use.
  q_v   VAPOUR specific humidity [kg/kg] -- mass of vapour per unit mass of TOTAL (moist) air.
  q_c / q_r   cloud / rain condensed-water specific humidity [kg/kg]; q_cond = q_c + q_r.
  eps   ``constants.EPS_MW`` = R_dry/R_vap ~ 0.622, the molar-mass ratio; 1/eps - 1 ~ 0.608.
  T_v / T_rho  virtual / density temperature [K]: the temperature DRY air would need to match the moist
        parcel's density at the same pressure. p = rho R_d T_rho is the moist ideal gas law.
  hypsometric relation   dz = (R_d T / g) dln(p_below/p_above), the thickness of a pressure layer.
  AD core  the fused anelastic dynamics core (``stage_ad_core.StageADCore``), the default backbone.

THE DEFECT. ``StageADCore._hydrostatic_z`` integrates the hypsometric relation on the DRY temperature
``T_atm``. The relation is exact only on ``T_rho``, so every layer of a moist column comes out
UNDER-THICK by exactly the factor ``T/T_rho = 1/(1 + 0.608 q_v - q_cond)``. Those heights are the ONE
bridge between the thermo (pressure) grid and the dynamics (height) grid, with THREE in-step consumers:
``_update_buoyancy`` maps T and q onto the height grid through them; the resolved winds come back
through the same map; and the convective adjustment's lapse-rate test measures its pair thickness
``dz_pair`` off them. So a moist column's thermo state lands too LOW on the dynamics grid AND its
convective criterion is tested over too thin a layer. It is moist-column-selective, hence a horizontal
gradient error and not an inert offset.

It is an OMISSION, not one of the anelastic approximations: those live in the DRY reference state
(``rho_ref``/``T_ref``/``p_ref``, dry by documented design), whereas ``_hydrostatic_z`` is a diagnostic
of the live column's actual geometry. The same package already integrates this relation on T_v where it
was worked through (``plume_floft._env_height_of_p``). BUT that framing is only half the story and the
smaller half: the buoyancy is referenced to ``T_ref(z)`` = ``p_ref(z)`` while the bridge places levels
where the LIVE column puts them, a prior departure ~7.4x larger on the canonical state (ROADMAP
PHY-22). This flag is the correct COMPLETION of the live map, not the closing of a self-contained
omission.

This file is the merge gate for the fix: default OFF, byte-identical OFF, the limiting cases pinned
symbolically (sympy lifted with ``sp.lambdify``, never a parallel hand-estimate), the flag composing
correctly with PHY-20 ``virtual_temp_exact`` and PHY-19 ``condensate_loading``, the fingerprint
recording it, the documented warm-start route working on a BUILT core, and an ON smoke.
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
    """The hypsometric thickness and its moist correction, lifted out of sympy with ``sp.lambdify``.

    Nothing in this file compares the code against a remembered number: the reference values come from
    the symbolic expressions below, evaluated here (OPERATIONS.md 3.5, the latent-coefficient-bug rule).
    """
    import sympy as sp
    T, qv, qc, g_, Rd, dlnp = sp.symbols("T q_v q_cond g R_d dlnp", positive=True)
    Trho = T * (1 + (1 / sp.Float(EPS) - 1) * qv - qc)          # exact density T for SPECIFIC humidities
    dz_dry = Rd * T / g_ * dlnp                                  # what the shipped bridge integrates
    dz_moist = Rd * Trho / g_ * dlnp                             # what the relation actually says
    # LIMIT 1 -- identity at q = 0: the correction vanishes exactly.
    assert sp.simplify((dz_moist - dz_dry).subs({qv: 0, qc: 0})) == 0
    # LIMIT 2 -- the thickness RATIO is T_rho/T, independent of R_d, g and dln p (they cancel).
    ratio = sp.simplify(dz_moist / dz_dry)
    assert sp.simplify(ratio - Trho / T) == 0
    assert ratio.free_symbols <= {qv, qc}                        # R_d, g, dlnp, T all cancel
    # LIMIT 3 -- an ISOTHERMAL moist column of uniform q has the closed-form height
    #            z(p) = (R_d T_rho/g) ln(p_s/p), i.e. the dry answer scaled by T_rho/T.
    p_s, p = sp.symbols("p_s p", positive=True)
    z_iso = Rd * Trho / g_ * sp.log(p_s / p)
    return {
        "ratio": sp.lambdify((qv, qc), ratio, "numpy"),
        "Trho": sp.lambdify((T, qv, qc), Trho, "numpy"),
        "z_iso": sp.lambdify((Rd, T, qv, qc, g_, p_s, p), z_iso, "numpy"),
    }


def test_the_thickness_ratio_is_exactly_the_density_temperature_ratio(sym):
    """LIMIT 2, on the code: with a UNIFORM q the whole column scales by one number, T_rho/T, and that
    number must match the symbolic ratio to round-off. R_d, g and dln p cancel out of it, so this pins
    the moist factor alone and cannot be passed by an accidentally-rescaled column."""
    for q in (0.0, 0.01, 0.07, 0.20):
        off = _heights(False, q=q)
        on = _heights(True, q=q, virtual_temp_exact=True)
        r = on[off > 0.0] / off[off > 0.0]
        assert np.allclose(r, sym["ratio"](q, 0.0), rtol=1e-12), f"q={q}"


def test_the_correction_vanishes_identically_at_q_zero(sym):
    """LIMIT 1, on the code, and at the strictest bar available: a DRY column must come out
    BIT-for-bit identical with the flag ON, because ``Tsrc`` reduces to exactly ``T_atm``."""
    assert sym["ratio"](0.0, 0.0) == 1.0
    for exact in (False, True):
        off = _heights(False, q=0.0)
        on = _heights(True, q=0.0, virtual_temp_exact=exact)
        assert np.array_equal(off, on)


def test_the_isothermal_moist_column_matches_the_closed_form(sym):
    """LIMIT 3: on an ISOTHERMAL column of uniform q the hypsometric integral has a closed form,
    ``z(p) = (R_d T_rho/g) ln(p_s/p)``. The core's own edge-averaging is exact for a constant profile,
    so the layer-centre heights must match that expression (evaluated through sympy) to round-off."""
    T0, q = 300.0, 0.08
    core = _core(True, virtual_temp_exact=True)
    core.T_atm = np.full_like(core.T_atm, T0)
    core.q = np.full_like(core.q, q)
    z = np.asarray(core._hydrostatic_z(core.T_atm), float)       # (nx, nz) TOP->SURFACE [m]
    p_edge = np.asarray(core.p_edge, float)
    z_edge_ref = sym["z_iso"](float(C.R_DRY), T0, q, 0.0, float(C.G_ACCEL),
                              float(C.P_SURF), p_edge)           # heights of the INTERFACES [m]
    # the core's surface interface is pinned to z = 0, so compare against the same datum
    z_edge_ref = z_edge_ref - z_edge_ref[-1]
    z_ref = 0.5 * (z_edge_ref[:-1] + z_edge_ref[1:])
    assert np.allclose(z, z_ref[None, :], rtol=1e-12, atol=1e-9)


def test_the_moist_column_is_thicker_and_only_the_moist_part_moves(sym):
    """Direction and selectivity -- the two properties the defect is about. Moist columns must come out
    THICKER (they were under-thick), monotonically in q, and a dry column in the same call must not
    move at all: the error is moist-column-selective, i.e. a horizontal gradient, not an offset."""
    core = _core(True, virtual_temp_exact=True)
    q = np.zeros_like(core.q)
    q[1::2, :] = 0.10                                            # every other column moist
    core.q = q
    z_on = np.asarray(core._hydrostatic_z(core.T_atm), float)
    core.cfg.hydrostatic_virtual_t = False
    z_off = np.asarray(core._hydrostatic_z(core.T_atm), float)
    dz = z_on - z_off
    assert np.array_equal(dz[0::2], np.zeros_like(dz[0::2])), "a DRY column moved"
    assert np.all(dz[1::2, :-1] > 0.0), "a MOIST column did not get thicker"
    # monotone in q, at the column top
    tops = [float(np.max(_heights(True, q=qq, virtual_temp_exact=True)[:, 0])) for qq in (0.0, 0.05, 0.15)]
    assert tops[0] < tops[1] < tops[2]


def test_the_condensate_gate_follows_condensate_loading():
    """Composition with PHY-19 step 2. The suspended cloud+rain mass is carried in the thickness only
    when ``condensate_loading`` is ON -- deliberately, so the two arms stay separable -- and when it is
    ON it must make the column THINNER than vapour alone (condensate is weight, not buoyancy)."""
    kw = dict(prognostic_condensate=True, prognostic_rain=True, virtual_temp_exact=True)
    z_vap = _heights(True, q=0.10, q_cond=2.0e-3, condensate_loading=False, **kw)
    z_load = _heights(True, q=0.10, q_cond=2.0e-3, condensate_loading=True, **kw)
    assert np.all(z_load[:, :-1] < z_vap[:, :-1]), "loading must reduce the thickness"
    # ... and it must be ABSENT when the loading flag is off, even though q_c/q_r are populated
    z_vap_noc = _heights(True, q=0.10, q_cond=0.0, condensate_loading=False, **kw)
    assert np.array_equal(z_vap, z_vap_noc)


def test_the_formula_follows_virtual_temp_exact(sym):
    """Composition with PHY-20. The bridge must use the SAME density-temperature formula the buoyancy
    uses, so that the two can never disagree: with ``virtual_temp_exact`` ON the thickness ratio is the
    exact specific-humidity one, and with it OFF it is the legacy ratio form -- and the exact form is
    the thicker of the two (it is the more buoyant one)."""
    q = 0.12
    off = _heights(False, q=q)
    r_legacy = _heights(True, q=q, virtual_temp_exact=False)[off > 0] / off[off > 0]
    r_exact = _heights(True, q=q, virtual_temp_exact=True)[off > 0] / off[off > 0]
    assert np.allclose(r_exact, sym["ratio"](q, 0.0), rtol=1e-12)
    assert np.allclose(r_legacy, float(th.virtual_temperature(1.0, q)), rtol=1e-12)
    assert r_exact.mean() > r_legacy.mean()


# ------------------------------------------------------------------------------ the flag on the core
def _core(flag, q=None, q_cond=None, **kw):
    cfg_kw = dict(nx=16, nz=12, theta_max_deg=80.0)
    cfg_kw.update(kw)                                 # callers may override nx/nz
    if flag is not None:
        cfg_kw["hydrostatic_virtual_t"] = flag
    core = StageADCore(StageADConfig(**cfg_kw))
    if q is not None:
        core.q = np.full_like(core.q, q)
    if q_cond is not None and getattr(core, "q_c", None) is not None:
        core.q_c = np.full_like(core.q_c, q_cond)
        if getattr(core, "q_r", None) is not None:
            core.q_r = np.zeros_like(core.q_r)
    return core


def _heights(flag, **kw):
    core = _core(flag, **kw)
    return np.asarray(core._hydrostatic_z(core.T_atm), float)


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
        cfg_kw["hydrostatic_virtual_t"] = flag
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
    ``cosmology_models/phy21_hydrostatic_virtual_t_off_control.py``, run in an ``origin/main`` worktree
    and in this one.)"""
    monkeypatch.delenv("COSMO_HYDROSTATIC_VIRTUAL_T", raising=False)
    assert StageADConfig(nx=8, nz=6).hydrostatic_virtual_t is False
    assert _stepped_hash(None) == _stepped_hash(False)


def test_the_env_key_reaches_the_config(monkeypatch):
    monkeypatch.setenv("COSMO_HYDROSTATIC_VIRTUAL_T", "1")
    assert StageADConfig(nx=8, nz=6).hydrostatic_virtual_t is True
    monkeypatch.setenv("COSMO_HYDROSTATIC_VIRTUAL_T", "0")
    assert StageADConfig(nx=8, nz=6).hydrostatic_virtual_t is False


def test_the_fingerprint_records_the_key():
    """A checkpoint must not be able to hide which side of the switch it ran on. The fingerprint's
    physics set is INVERTED (everything not named volatile is physics), so this rides that for free --
    which is exactly why it needs a test rather than a claim."""
    core = StageADCore(StageADConfig(nx=12, nz=8, hydrostatic_virtual_t=True))
    fp = json.loads(core.fingerprint())
    assert fp["config"]["hydrostatic_virtual_t"] is True
    from flatworld import _ckpt_fingerprint as ckfp
    assert "hydrostatic_virtual_t" not in ckfp.VOLATILE_FIELDS    # i.e. a difference is REPORTED


def _buoyancy(flag, **kw):
    """One ``_update_buoyancy`` call on a seeded moist column; returns b (nz, nx) as a host array.
    ``_z`` is refreshed through ``_hydrostatic_z`` first, exactly as the step does."""
    rng = np.random.default_rng(4)
    core = StageADCore(StageADConfig(nx=16, nz=12, theta_max_deg=80.0,
                                     hydrostatic_virtual_t=flag, **kw))
    core.T_atm = core.T_atm + 2.0 * rng.standard_normal(core.T_atm.shape)
    core.q = np.clip(np.abs(rng.standard_normal(core.q.shape)) * 0.05, 0.0, 0.3)
    if getattr(core, "q_c", None) is not None:
        core.q_c = np.abs(rng.standard_normal(np.shape(core.q_c))) * 1.0e-3
    if getattr(core, "q_r", None) is not None:
        core.q_r = np.abs(rng.standard_normal(np.shape(core.q_r))) * 3.0e-4
    core._z = core._hydrostatic_z(core.T_atm)
    core._update_buoyancy()
    return np.asarray(core.b, dtype=np.float64).copy()


def test_the_flag_reaches_the_dynamics_buoyancy():
    """ON must be NON-VACUOUS where it matters: ``_hydrostatic_z`` is only a bridge, so the claim is
    that moving the bridge moves the buoyancy the dynamics integrates."""
    off, on = _buoyancy(False), _buoyancy(True)
    assert np.max(np.abs(on - off)) > 1e-6, "the flag did not reach the buoyancy"


def test_the_flag_still_bites_with_vapour_buoyancy_off():
    """UNLIKE PHY-20, this term is NOT inert when ``virtual_buoyancy`` is OFF: it corrects the column's
    GEOMETRY, so a T-only buoyancy is still sampled at the wrong heights. This pins that the two flags
    are genuinely different knobs and that one cannot be used to reason about the other."""
    off = _buoyancy(False, virtual_buoyancy=False)
    on = _buoyancy(True, virtual_buoyancy=False)
    assert np.max(np.abs(on - off)) > 1e-6


def test_a_post_build_flag_change_takes_effect():
    """Turn-on route (b), the ``build_core_with_world`` ANALYSIS rebuild: that loader calls
    ``assert_env_matches``, which correctly refuses a live COSMO_* key the run17 checkpoint's world
    never recorded, so the rebuild turns the flag on with ``core.cfg.hydrostatic_virtual_t = True``.
    That works only because ``_hydrostatic_z`` RE-READS the flag every call -- nothing is baked at
    construction. (Route (a), a production arm, uses the env key, which route ``test_the_env_key...``
    covers.)"""
    rng = np.random.default_rng(4)
    core = StageADCore(StageADConfig(nx=16, nz=12, theta_max_deg=80.0))
    assert core.cfg.hydrostatic_virtual_t is False
    core.q = np.clip(np.abs(rng.standard_normal(core.q.shape)) * 0.05, 0.0, 0.3)
    z_off = np.asarray(core._hydrostatic_z(core.T_atm), float).copy()
    core.cfg.hydrostatic_virtual_t = True         # exactly the line the config comment documents
    z_on = np.asarray(core._hydrostatic_z(core.T_atm), float).copy()
    assert np.max(np.abs(z_on - z_off)) > 1.0     # metres


def test_the_on_path_smoke_runs_and_stays_finite():
    """ON smoke: the flag is not just a formula, it feeds the dynamics through the grid bridge -- five
    coupled steps must run, stay finite, and stay climatically sane (the column must not detonate)."""
    rng = np.random.default_rng(2)
    core = StageADCore(StageADConfig(nx=16, nz=12, theta_max_deg=80.0, hydrostatic_virtual_t=True,
                                     prognostic_condensate=True, clouds=True))
    core.T_atm = core.T_atm + 0.7 * rng.standard_normal(core.T_atm.shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(5):
            core.step()
    for name in ("T_atm", "q", "b", "psi", "omega", "T_surf"):
        v = np.asarray(getattr(core, name), float)
        assert np.all(np.isfinite(v)), f"{name} went non-finite with hydrostatic_virtual_t ON"
    Tc = np.asarray(core.T_atm, float) - C.T_FREEZE
    assert -150.0 < float(Tc.mean()) < 200.0, f"insane mean T_atm {float(Tc.mean()):.1f} C"


def _advection_core(flag, seed=7):
    """A toy core carrying a STEADY, vigorous overturning cell and a moist column, with ``_z`` built
    through the flag. The flow comes from ``seed_warm_edge_buoyancy`` + ``run`` -- the same recipe
    ``test_moisture_advected_by_resolved_flow`` uses -- because a couple of coupled steps from rest
    leaves a transport too weak for a non-vacuity bar to mean anything."""
    rng = np.random.default_rng(seed)
    core = StageADCore(StageADConfig(nx=24, nz=14, theta_max_deg=60.0, nu_b=40.0,
                                     tol=3e-3, check_every=200, patience=3, max_iter=30000,
                                     hydrostatic_virtual_t=flag, virtual_temp_exact=True))
    core.seed_warm_edge_buoyancy(dT_edge=40.0)        # held b; a steady single cell
    core.run(verbose=False)
    core.q = np.clip(np.abs(rng.standard_normal(core.q.shape)) * 0.04, 0.0, 0.3)
    core._z = core._hydrostatic_z(core.T_atm)         # the bridge under test
    return core


@pytest.mark.parametrize("flag", [False, True])
def test_the_resolved_advection_still_conserves_with_the_flag_on(flag):
    """PR #841 review F9(a). The bridge feeds `u_p` into `_divfree_mass_flux`, so a bad height map could
    in principle break the mass-consistent transport. It cannot -- the flux is built on the FIXED layer
    mass `dm = dp/g`, so the column integrals are conserved by construction whatever `z_p` is -- but
    that is an argument, and this measures it ON as well as OFF.

    `Σ q·dm` through `_advect_moisture` and `Σ θ·dm` through `_advect_heat` (θ = T(P_SURF/p)^κ, the
    adiabatic invariant that limb rearranges) must both hold to round-off. NON-VACUITY is checked in the
    same call: a transport that moved nothing would pass conservation trivially.
    """
    core = _advection_core(flag)
    u, _w = core.velocities()
    dm = np.asarray(core.dm, float)[None, :]

    q0 = np.asarray(core.q, float).copy()
    water0 = float(np.sum(q0 * dm))
    core._advect_moisture(u, 600.0)
    q1 = np.asarray(core.q, float)
    water1 = float(np.sum(q1 * dm))
    assert np.max(np.abs(q1 - q0)) > 1e-9, "the moisture transport was vacuous -- conservation is empty"
    assert abs(water1 - water0) <= 1e-12 * abs(water0)

    kap = float(C.R_DRY / C.CP_AIR)
    pfac = (float(C.P_SURF) / np.asarray(core.p_mid, float)[None, :]) ** kap
    th0 = np.asarray(core.T_atm, float) * pfac
    heat0 = float(np.sum(th0 * dm))
    core._advect_heat(u, 600.0)
    th1 = np.asarray(core.T_atm, float) * pfac
    heat1 = float(np.sum(th1 * dm))
    assert np.max(np.abs(th1 - th0)) > 1e-9, "the heat transport was vacuous"
    assert abs(heat1 - heat0) <= 1e-9 * abs(heat0)


def test_the_clamp_behaviour_is_pinned_in_both_directions():
    """PR #841 review F9(b) + F4. ON raises `z_p`, so it pushes thermo layers UP relative to the fixed
    dynamics lid: the REVERSE map (`_height_to_pressure`, thermo layers above `self.z[-1]`, which
    `np.interp` clamps to the lid's u/w) must weakly GAIN clamped layers, while the FORWARD map's top
    clamp (dynamics levels above the column's highest layer centre) must weakly LOSE them -- the column
    got taller. This is the non-locality the sizing instrument audits on the canonical state; here it is
    pinned as a property of the algebra on a toy world, in BOTH directions rather than one."""
    core = _core(True, q=0.25, nz=24, virtual_temp_exact=True)
    z_on = np.asarray(core._hydrostatic_z(core.T_atm), float)
    core.cfg.hydrostatic_virtual_t = False
    z_off = np.asarray(core._hydrostatic_z(core.T_atm), float)
    zgrid = np.asarray(core.z, float)
    lid = float(zgrid[-1])
    rev_off = (z_off > lid).sum(axis=1)
    rev_on = (z_on > lid).sum(axis=1)
    assert np.all(rev_on >= rev_off), "ON must not REDUCE the reverse map's lid clamping"
    fwd_off = (zgrid[None, :] > z_off[:, 0][:, None]).sum(axis=1)
    fwd_on = (zgrid[None, :] > z_on[:, 0][:, None]).sum(axis=1)
    assert np.all(fwd_on <= fwd_off), "ON must not INCREASE the forward map's top clamping"
    assert int(rev_on.sum()) > int(rev_off.sum()), "non-vacuous: some layer must cross the lid"


def test_the_convective_trigger_can_only_loosen():
    """PR #841 review F5. `_z` has a THIRD in-step consumer besides the buoyancy placement and the u/w
    sampling: the convective adjustment's lapse-rate test, `unstable = T[k-1] < T[k] - Γ·dz_pair`
    (`stage_ad_column`). ON thickens every pair, which raises the `Γ·dz` threshold, so the effect there
    is ONE-SIGNED: pairs can only stop being flagged unstable, never start. Pinned here because it is
    the model's dominant column-thermodynamics lever and the PR's first edition omitted it."""
    core = _core(True, q=0.12, virtual_temp_exact=True)
    rng = np.random.default_rng(5)
    core.T_atm = core.T_atm + 3.0 * rng.standard_normal(core.T_atm.shape)   # make pairs unstable
    T = np.asarray(core.T_atm, float)
    z_on = np.asarray(core._hydrostatic_z(T), float)
    core.cfg.hydrostatic_virtual_t = False
    z_off = np.asarray(core._hydrostatic_z(T), float)
    p_mid = np.asarray(core.p_mid, float)
    G = np.stack([np.asarray(th.moist_adiabat_lapse_rate(T[:, k], float(p_mid[k])), float)
                  for k in range(core.cfg.nz)], axis=1)
    def unstable(z):
        dzp = z[:, :-1] - z[:, 1:]
        return T[:, :-1] < (T[:, 1:] - G[:, 1:] * dzp) - 1e-9
    u_off, u_on = unstable(z_off), unstable(z_on)
    assert np.all(z_on[:, :-1] - z_on[:, 1:] >= z_off[:, :-1] - z_off[:, 1:]), "pairs must not thin"
    assert not np.any(~u_off & u_on), "a pair FLIPPED ON -- the effect is supposed to be one-signed"
    assert int((u_off & ~u_on).sum()) > 0, "non-vacuous: some pair must flip off on this state"


def test_the_heights_stay_monotone_and_the_surface_datum_is_pinned():
    """A grid-bridge term must not break the invariants ``_pressure_to_height`` relies on: the column's
    layer-centre heights stay STRICTLY DECREASING top->surface (the interpolation samples must be
    monotone) and the surface interface stays at z = 0 (the datum is pinned, not floated)."""
    z = _heights(True, q=0.15, virtual_temp_exact=True)
    assert np.all(np.diff(z, axis=1) < 0.0)
    assert np.all(z[:, -1] > 0.0)                 # the surface LAYER CENTRE is half a layer up
    dry = _heights(False, q=0.15)
    assert np.all(z[:, -1] > dry[:, -1])
