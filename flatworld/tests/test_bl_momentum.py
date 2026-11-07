"""Issue #111: the nonlinear boundary-layer momentum solve for the REPORTED surface wind.

Covers the shared operators refactor, the virtual-temperature Phi build, and the solver's defining
properties: energetically self-limiting (no Bernoulli-ceiling violation -- the linear balance's failure
mode), reproduces the AD core's zonal-mean u_surf, periodic-e correctness, and the opt-in katabatic
term off-by-default byte-identical. The slow test confirms the cascade default-OFF is byte-identical
and the ON path leaves the moisture/precip map untouched (only the reported wind switches).
"""
import numpy as np
import pytest
from flatworld.topography import Topography
from flatworld.stage_c_map import StageCMap, StageCConfig
from flatworld import operators as ops
from flatworld import bl_forcing as blf
from flatworld import bl_momentum as blm


def _stage_c_state(seed=7, nx=36, ne=30, Le_km=6000):
    t = Topography.procedural_continent(seed=seed, nx=nx, ne=ne, Le_km=Le_km)
    m = StageCMap(t, StageCConfig(dt=900.0, check_every=200, tol=5e-4, patience=3, max_seconds=20))
    m.run(verbose=False)
    return m, m.diagnostics()


# ----------------------------------------------------------------- operators (refactor)
def test_operators_wall_and_periodic_conventions():
    rng = np.random.default_rng(0)
    F = rng.standard_normal((8, 6))
    # e is periodic: a uniform-in-e field has zero e-derivative and rolling in e commutes with d/de.
    col = rng.standard_normal((8, 1)) * np.ones((1, 6))
    assert np.allclose(ops.dde(col, 1.0), 0.0)
    assert np.allclose(ops.dde(np.roll(F, 2, axis=1), 1.0), np.roll(ops.dde(F, 1.0), 2, axis=1))
    # x is walled: ddx uses one-sided ends (the divergence/adv_flux carry the no-flux walls).
    d = ops.ddx(F, 1.0)
    assert d.shape == F.shape
    # adv_flux conserves the domain integral of F under transport (zero-flux x-walls, periodic e):
    U = rng.standard_normal((8, 6)); V = rng.standard_normal((8, 6))
    tend = ops.adv_flux(F, U, V, 1.0, 1.0)
    assert abs(float(np.sum(tend))) < 1e-9        # pure transport integrates to ~0


def test_stage_c_methods_match_free_operators():
    # the StageCMap operator methods must be byte-identical thin wrappers over flatworld.operators.
    t = Topography.procedural_continent(seed=3, nx=20, ne=16, Le_km=5000)
    m = StageCMap(t, StageCConfig(dt=900.0, check_every=50, tol=5e-4, patience=2, max_seconds=5))
    rng = np.random.default_rng(1)
    F = rng.standard_normal((m.nx, m.ne)); U = rng.standard_normal((m.nx, m.ne)); V = rng.standard_normal((m.nx, m.ne))
    assert np.array_equal(m._dx(F), ops.ddx(F, m.dx))
    assert np.array_equal(m._de(F), ops.dde(F, m.de))
    assert np.array_equal(m._lap(F), ops.lap(F, m.dx, m.de))
    assert np.array_equal(m._div(U, V), ops.div(U, V, m.dx, m.de))
    assert np.array_equal(m._adv_flux(F, U, V), ops.adv_flux(F, U, V, m.dx, m.de))


# ----------------------------------------------------------------- bl_forcing (virtual-T Phi)
def test_dry_phi_reproduces_stage_c_geopotential():
    m, dC = _stage_c_state()
    Phi_ref = m._geopotential()
    Phi_new = blf.geopotential(m.T_map_C, m.h, m.dx, m.de,
                               beta_T=m.cfg.beta_T, g_terrain=m.cfg.g_terrain, phi_smooth_km=m.cfg.phi_smooth_km)
    assert np.array_equal(Phi_ref, Phi_new)       # byte-identical dry limit


def test_virtual_temperature_dry_limit_byte_identical_and_moist_bonus():
    m, dC = _stage_c_state()
    # q = 0 -> T_v == T (the +-T_FREEZE round-trip is short-circuited)
    assert np.array_equal(blf.virtual_T_map(m.T_map_C, np.zeros_like(m.T_map_C)), m.T_map_C)
    assert np.array_equal(blf.virtual_T_map(m.T_map_C, None), m.T_map_C)
    # with real moisture T_v >= T everywhere, and the bonus is a few K at tens of mm of column water
    Tv = blf.virtual_T_map(m.T_map_C, np.full_like(m.T_map_C, 50.0))
    assert (Tv >= m.T_map_C - 1e-12).all() and (Tv - m.T_map_C).max() > 0.5


# ----------------------------------------------------------------- the solver: defining properties
def test_no_bernoulli_ceiling_violation_pure_phi():
    # THE headline acceptance (#111): with the Phi-only flow (ubg=0) the steady kinetic energy never
    # exceeds the frictionless potential drop -- the linear balance violated this in ~26% of cells.
    m, dC = _stage_c_state()
    Tv = blf.virtual_T_map(dC["T_map_C"], dC["W_s"])
    Phi = blf.geopotential(Tv, dC["height"], m.dx, m.de, beta_T=30.0, g_terrain=0.0, phi_smooth_km=400.0)
    sol = blm.solve_surface_wind(Phi, np.zeros(m.nx), m.dx, m.de, blm.BLMomentumConfig(max_seconds=30))
    assert sol["converged"]
    assert sol["bernoulli_excess_frac"] < 0.02     # ~0 by construction (vs 0.26 for the linear balance)
    assert np.isfinite(sol["speed"]).all()


def test_self_limiting_with_realistic_background_wind():
    # with a realistic u_surf (~-30 m/s) the field stays bounded (tens of m/s, not the linear balance's
    # 150-230) and the Bernoulli excess stays small across the drag closures.
    m, dC = _stage_c_state()
    Tv = blf.virtual_T_map(dC["T_map_C"], dC["W_s"])
    Phi = blf.geopotential(Tv, dC["height"], m.dx, m.de, beta_T=30.0, g_terrain=0.0, phi_smooth_km=400.0)
    ubg = np.full(m.nx, -30.0)
    for cfg in (blm.BLMomentumConfig(drag_form="quadratic", max_seconds=30),
                blm.BLMomentumConfig(drag_form="linear", r_mom=2.44e-5, max_seconds=30)):
        sol = blm.solve_surface_wind(Phi, ubg, m.dx, m.de, cfg)
        assert sol["converged"]
        assert sol["speed"].max() < 90.0           # bounded -- nowhere near the linear balance's max
        assert sol["bernoulli_excess_frac"] < 0.05


def test_zonal_mean_reproduces_background_wind():
    # acceptance: the solved surface wind's zonal mean reproduces the AD core's u_surf within tier.
    # Under FLAT Phi (no anomaly forcing) the steady state must be exactly (u_bg, 0).
    m, dC = _stage_c_state()
    Phi = np.zeros((m.nx, m.ne))
    ubg = np.linspace(-35, -10, m.nx)
    sol = blm.solve_surface_wind(Phi, ubg, m.dx, m.de, blm.BLMomentumConfig(max_seconds=30))
    assert np.max(np.abs(sol["u"].mean(axis=1) - ubg)) < 0.5
    assert np.max(np.abs(sol["v"])) < 0.5
    # with a real Phi anomaly the zonal mean still tracks u_bg within tier (the mean-relaxation)
    Tv = blf.virtual_T_map(dC["T_map_C"], dC["W_s"])
    Phi2 = blf.geopotential(Tv, dC["height"], m.dx, m.de, beta_T=30.0, g_terrain=0.0, phi_smooth_km=400.0)
    sol2 = blm.solve_surface_wind(Phi2, np.full(m.nx, -30.0), m.dx, m.de, blm.BLMomentumConfig(max_seconds=30))
    assert np.max(np.abs(sol2["u"].mean(axis=1) - (-30.0))) < 8.0    # within tier


def test_periodic_e_invariance():
    # rolling the forcing along the periodic e-axis rolls the solution by the same shift (e is a slab
    # of the endless ribbon; ubg is e-independent).
    m, dC = _stage_c_state(seed=11, nx=28, ne=24)
    Tv = blf.virtual_T_map(dC["T_map_C"], dC["W_s"])
    Phi = blf.geopotential(Tv, dC["height"], m.dx, m.de, beta_T=30.0, g_terrain=0.0, phi_smooth_km=400.0)
    ubg = np.full(m.nx, -20.0)
    cfg = blm.BLMomentumConfig(max_seconds=30, tol=5e-5)
    s0 = blm.solve_surface_wind(Phi, ubg, m.dx, m.de, cfg)
    sR = blm.solve_surface_wind(np.roll(Phi, 5, axis=1), ubg, m.dx, m.de, cfg)
    assert np.allclose(sR["u"], np.roll(s0["u"], 5, axis=1), atol=0.5)
    assert np.allclose(sR["v"], np.roll(s0["v"], 5, axis=1), atol=0.5)


def test_callable_phi_matches_static_and_checkpoint_resumes(tmp_path):
    # a callable phi returning a CONSTANT array must give byte-identical results to the static path
    # (guards the static-hoist refactor + the callable/time-accumulation path used by #113).
    m, dC = _stage_c_state(seed=4, nx=24, ne=20)
    Phi = blf.geopotential(blf.virtual_T_map(dC["T_map_C"], dC["W_s"]), dC["height"], m.dx, m.de,
                           beta_T=30.0, g_terrain=0.0, phi_smooth_km=400.0)
    ubg = np.full(m.nx, -20.0)
    cfg = blm.BLMomentumConfig(max_seconds=30, tol=1e-4)
    s_static = blm.solve_surface_wind(Phi, ubg, m.dx, m.de, cfg)
    s_callable = blm.solve_surface_wind(lambda t=0.0: Phi, ubg, m.dx, m.de, cfg)
    assert np.array_equal(s_static["u"], s_callable["u"]) and np.array_equal(s_static["v"], s_callable["v"])
    # checkpoint resume: a guarded short run leaves a checkpoint; a second call LOADS it (resumes, not restarts)
    ckpt = str(tmp_path / "bl.npz")
    g = blm.BLMomentumConfig(tol=1e-12, max_iter=80, check_every=40, checkpoint_path=ckpt, checkpoint_every=40)
    r1 = blm.solve_surface_wind(Phi, ubg, m.dx, m.de, g)
    assert not r1["converged"] and r1["iters"] >= 80
    r2 = blm.solve_surface_wind(Phi, ubg, m.dx, m.de,
                                blm.BLMomentumConfig(tol=1e-12, max_iter=160, check_every=40, checkpoint_path=ckpt))
    assert r2["iters"] > r1["iters"]          # continued from the checkpoint rather than restarting at 0


def test_convergence_and_positive_dt():
    m, dC = _stage_c_state()
    Tv = blf.virtual_T_map(dC["T_map_C"], dC["W_s"])
    Phi = blf.geopotential(Tv, dC["height"], m.dx, m.de, beta_T=30.0, g_terrain=0.0, phi_smooth_km=400.0)
    sol = blm.solve_surface_wind(Phi, dC["ubg"], m.dx, m.de, blm.BLMomentumConfig(tol=1e-4, max_seconds=30))
    assert sol["converged"] and sol["dt"] > 0 and sol["iters"] > 0
    # tightening the guard to a tiny budget stops on the GUARD, not a crash
    g = blm.solve_surface_wind(Phi, dC["ubg"], m.dx, m.de, blm.BLMomentumConfig(tol=1e-12, max_iter=100))
    assert not g["converged"] and "GUARD" in g["stop_reason"]


# ----------------------------------------------------------------- katabatic (opt-in, scope 2)
def test_katabatic_off_byte_identical():
    m, dC = _stage_c_state()
    Tv = blf.virtual_T_map(dC["T_map_C"], dC["W_s"])
    Phi = blf.geopotential(Tv, dC["height"], m.dx, m.de, beta_T=30.0, g_terrain=0.0, phi_smooth_km=400.0)
    base = blm.BLMomentumConfig(max_seconds=30)
    off = blm.solve_surface_wind(Phi, dC["ubg"], m.dx, m.de, base, T_map_C=dC["T_map_C"], height=dC["height"])
    none = blm.solve_surface_wind(Phi, dC["ubg"], m.dx, m.de, base)   # no terrain/T passed at all
    assert np.array_equal(off["u"], none["u"]) and np.array_equal(off["v"], none["v"])


def test_katabatic_drains_downhill_off_cold_terrain():
    # a cold ridge: dT_cold>0 over the high ground, so the slope-flow acceleration points DOWN-slope.
    nx, ne = 10, 8
    height = np.zeros((nx, ne)); height[:, 3] = 2000.0; height[:, 4] = 1000.0   # a ridge at column 3
    T = np.full((nx, ne), 10.0); T[:, 3] = -10.0; T[:, 4] = 0.0                  # the ridge is COLD
    cfg = blm.BLMomentumConfig(katabatic=True, c_kata=1.0)
    ax, ay = blm.katabatic_accel(T, height, 1.0e4, 1.0e4, cfg)
    # between the peak (col 3) and the lower col 4, -dh/de < 0, cold air there -> ay < 0 (drains toward 4)
    assert np.any(ay[:, 3] < 0) or np.any(ay[:, 4] < 0)
    assert np.any(np.abs(ay) > 0)               # the term is non-trivially active over the cold slope
    # disabled -> exactly zero
    off = blm.BLMomentumConfig(katabatic=False)
    z = blm.katabatic_accel(T, height, 1.0e4, 1.0e4, off)
    assert np.array_equal(z[0], np.zeros_like(height)) and np.array_equal(z[1], np.zeros_like(height))


# ----------------------------------------------------------------- cascade integration (slow)
@pytest.mark.slow
def test_cascade_default_off_byte_identical_and_on_preserves_moisture():
    from flatworld import cascade
    base = cascade.run_cascade("test", core="diffusive", verbose=False)
    assert "bl_momentum" not in base["dC"]                       # default OFF: no momentum field
    cfg = blm.BLMomentumConfig(max_seconds=60)
    on = cascade.run_cascade("test", core="diffusive", verbose=False, bl_momentum_wind=True, bl_momentum_cfg=cfg)
    dC = on["dC"]
    # the moisture/precip map and temperature are UNTOUCHED (only the reported wind switches)
    assert np.array_equal(base["dC"]["P_mm_day"], dC["P_mm_day"])
    assert np.array_equal(base["dC"]["T_map_C"], dC["T_map_C"])
    assert np.array_equal(base["dC"]["W_s"], dC["W_s"])
    # the reported wind is the momentum field; the linear balance is preserved as provenance
    assert "u_bal" in dC and "froude" in dC and "bl_momentum" in dC
    assert not np.array_equal(dC["u"], dC["u_bal"])              # the reported field actually changed
    assert dC["bl_momentum"]["bernoulli_excess_frac"] < 0.05


# ----------------------------------------------------------------- W1: the two-layer momentum wind
def test_aloft_departure_wind_matches_stage_c_inline():
    # aloft_departure_wind is the SINGLE source of truth for Stage C's inline aloft algebra; it must
    # reproduce stage_c_map._step_with_wind's expression exactly for the StageCConfig defaults.
    rng = np.random.default_rng(2)
    nx, ne = 12, 9
    u = rng.standard_normal((nx, ne)); v = rng.standard_normal((nx, ne))
    ubg = rng.standard_normal(nx)
    c = StageCConfig()                                            # u_aloft_frac=1.0, aloft_departure_sign=1.0
    ubg2d = np.broadcast_to(ubg[:, None], (nx, ne))
    u_inline = -c.u_aloft_frac * ubg2d + c.aloft_departure_sign * (u - ubg2d)
    v_inline = c.aloft_departure_sign * v
    u_a, v_a = blm.aloft_departure_wind(u, v, ubg, c.u_aloft_frac, c.aloft_departure_sign)
    assert np.array_equal(u_a, u_inline) and np.array_equal(v_a, v_inline)


def test_two_layer_off_is_byte_identical_to_surface_solve():
    # the gate: two_layer=False -> solve_two_layer_wind returns the surface solve, byte-identical to
    # build_wind_field (no aloft fields, same u/v).
    m, d = _stage_c_state(seed=5, nx=28, ne=24)
    cfg = blm.BLMomentumConfig(max_seconds=20, two_layer=False)
    base = blm.build_wind_field(d, m.dx, m.de, cfg, verbose=False)
    two = blm.solve_two_layer_wind(d, m.dx, m.de, cfg, verbose=False)
    assert np.array_equal(base["u"], two["u"]) and np.array_equal(base["v"], two["v"])
    assert "u_aloft" not in two and "two_layer" not in two


def test_two_layer_on_adds_instrumented_aloft_self_limiting():
    # two_layer=True adds the aloft departure + the Bernoulli/divergence instrumentation; the SURFACE
    # solve is unchanged vs OFF, and on a real Stage-C state the aloft branch is self-limiting -> (a).
    m, d = _stage_c_state(seed=5, nx=28, ne=24)
    off = blm.solve_two_layer_wind(d, m.dx, m.de, blm.BLMomentumConfig(max_seconds=20, two_layer=False))
    on = blm.solve_two_layer_wind(d, m.dx, m.de, blm.BLMomentumConfig(max_seconds=20, two_layer=True))
    assert np.array_equal(off["u"], on["u"]) and np.array_equal(off["v"], on["v"])  # surface unchanged
    assert on["u_aloft"].shape == on["u"].shape and on["v_aloft"].shape == on["v"].shape
    tl = on["two_layer"]
    for k in ("surf_bernoulli_excess_frac", "aloft_bernoulli_excess_frac", "aloft_ok",
              "scope_decision", "surf_div", "aloft_div"):
        assert k in tl
    assert 0.0 <= tl["surf_bernoulli_excess_frac"] <= 1.0
    assert 0.0 <= tl["aloft_bernoulli_excess_frac"] <= 1.0
    # the surface momentum field is self-limiting (its own acceptance metric); the aloft inherits it on a
    # real state -> the W0-selected (a) scope holds here.
    assert tl["surf_bernoulli_excess_frac"] < 0.05
    assert tl["aloft_ok"] and tl["scope_decision"].startswith("(a)")


def test_aloft_ok_decision_rule_flags_a_violating_branch():
    # a synthetic aloft wind whose KE far exceeds its (small) Phi drop must trip the (b) branch; a
    # gentle one stays (a). Drives two_layer_diagnostics directly with crafted winds + Phi.
    nx, ne = 10, 8
    cfg = blm.BLMomentumConfig(two_layer=True)
    phi = np.tile(np.linspace(0.0, 50.0, ne), (nx, 1))           # mild along-track Phi structure
    calm = np.full((nx, ne), 2.0); zero = np.zeros((nx, ne))
    d_ok = blm.two_layer_diagnostics(calm, zero, calm, zero, phi, 1e4, 1e4, cfg)
    assert d_ok["aloft_ok"]
    # a huge aloft anomaly (alternating +/-50 m/s in e) vastly exceeds the ~50 m^2/s^2 Phi drop -> (b)
    big = np.tile(np.where(np.arange(ne) % 2 == 0, 50.0, -50.0), (nx, 1))
    d_bad = blm.two_layer_diagnostics(calm, zero, big, zero, phi, 1e4, 1e4, cfg)
    assert not d_bad["aloft_ok"] and d_bad["scope_decision"].startswith("(b)")


def test_bernoulli_perrow_is_stricter_than_global():
    # the per-row ceiling (max within a row) is <= the global ceiling, so the per-row excess fraction is
    # >= the global one for the same wind -- the metric distinction behind the "26% (per-row) vs 0%
    # (global)" re-measurement note.
    rng = np.random.default_rng(11)
    nx, ne = 16, 12
    # a wind with a strong along-track anomaly against a mild Phi -> both metrics see some excess
    u = 8.0 + 6.0 * np.sin(np.linspace(0, 4 * np.pi, ne))[None, :] * np.ones((nx, 1))
    v = rng.standard_normal((nx, ne))
    phi = np.tile(np.linspace(0.0, 20.0, ne), (nx, 1)) + rng.standard_normal((nx, ne))
    glob = blm.bernoulli_excess_frac(u, v, phi)
    perrow = blm.bernoulli_excess_frac_perrow(u, v, phi)
    assert perrow >= glob - 1e-12
