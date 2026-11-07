"""W2 (DIAGNOSTIC_WIND_ROADMAP) -- advect the two-layer moisture on the SELF-LIMITING momentum wind.

These cover the W2 deliverable: the coupled solver wiring (momentum_advection), the re-derived wind-Phi
smoothing length (the cap-saturated 400 km dissolves), the re-derived moisture CFL (sized to the momentum
field's actual peak, no cap), and the W2 gates (default-OFF byte-identical; ON runs stable, closes the
water budget, no W_s/W_a runaway, band placement preserved). The full canonical-nsub30 box-test is staged
in cosmology_models/diagnostic_wind_w2.py (W4 consumes it)."""
import numpy as np
import pytest
from flatworld.topography import Topography
from flatworld.stage_c_map import StageCMap, StageCConfig, frictional_smoothing_km
from flatworld import bl_momentum as blm


# ---- the re-derived momentum smoothing length (W2 risk item: don't inherit the cap-saturated 400 km) ----
def test_momentum_smoothing_km_replaces_cap_saturated_400km():
    # quadratic drag (the default arm): L = H_bl/C_D, INDEPENDENT of V_char (the cap-saturation ambiguity
    # dissolves structurally), and DISTINCT from the diagnostic balance's wind_cap/r_fric = 400 km.
    mc = blm.BLMomentumConfig()                                  # quadratic, H_bl=1000, C_D=1.2e-3
    L = blm.momentum_smoothing_km(mc)
    assert abs(L - mc.H_bl / mc.C_D / 1e3) < 1e-9               # closed form
    assert abs(L - 833.3333) < 1e-3                            # 833 km (~2x the diagnostic 400)
    assert abs(L - frictional_smoothing_km()) > 100.0          # genuinely NOT the cap-saturated 400 km
    # quadratic length is V_char-independent: same answer whatever speed you feed
    assert blm.momentum_smoothing_km(mc, V_char=5.0) == blm.momentum_smoothing_km(mc, V_char=50.0) == L
    # linear drag: L = V_char/r_mom (needs the measured momentum speed)
    mcl = blm.BLMomentumConfig(drag_form="linear", r_mom=6.1e-6)
    assert abs(blm.momentum_smoothing_km(mcl, V_char=19.0) - 19.0 / 6.1e-6 / 1e3) < 1e-6


def test_momentum_phi_smooth_resolution_and_override():
    # None -> re-derived from the momentum drag closure; an explicit float is honored verbatim.
    t = Topography.procedural_continent(seed=3, nx=32, ne=32, Le_km=6000, width_km=3500, center_theta=48)
    m_auto = StageCMap(t, StageCConfig(momentum_advection=True))
    assert abs(m_auto._mom_cfg.phi_smooth_km - blm.momentum_smoothing_km(blm.BLMomentumConfig())) < 1e-9
    m_set = StageCMap(t, StageCConfig(momentum_advection=True, momentum_phi_smooth_km=321.0))
    assert m_set._mom_cfg.phi_smooth_km == 321.0
    # OFF -> no in-loop momentum config built at all (default path untouched)
    assert StageCMap(t, StageCConfig())._mom_cfg is None


# ---- the CFL re-derivation: sized to the momentum field's ACTUAL peak (no cap), not wind_cap ----
def test_momentum_cfl_relaxes_when_gentle_and_subcycles_when_fast():
    t = Topography.procedural_continent(seed=3, nx=40, ne=40, Le_km=6000, width_km=3500, center_theta=48)
    c = StageCConfig(dt=900.0, momentum_advection=True, momentum_courant_target=0.5)
    m = StageCMap(t, c)
    dx, de = m.dx, m.de
    # a GENTLE wind (the canonical case: momentum peak << cap) -> Courant sub-target -> dt unchanged (no shrink)
    u = np.full((m.nx, m.ne), -2.0); v = np.full((m.nx, m.ne), 2.0)
    assert m._momentum_cfl_dt(u, v) == c.dt
    # a FAST ALONG-TRACK wind (binds over the SMALLER spacing de, the directional limit) -> dt SHRINKS so the
    # Courant <= target. (Along-track because the dominant aloft speed is cross-track, which advects over the
    # much larger dx here -- the directional CFL's whole point: the binding direction is de, not |V|.)
    v_fast = 5.0 * c.momentum_courant_target * de / c.dt        # along-track Courant ~5 at the full dt
    uf = np.zeros((m.nx, m.ne)); vf = np.full((m.nx, m.ne), v_fast)
    dt_fast = m._momentum_cfl_dt(uf, vf)
    assert dt_fast < c.dt
    # the resulting DIRECTIONAL Courant is at/under target in BOTH directions, over BOTH layers
    u_al, v_al = blm.aloft_departure_wind(uf, vf, m.ubg, c.u_aloft_frac, c.aloft_departure_sign)
    umax = max(float(np.max(np.abs(uf))), float(np.max(np.abs(u_al))))
    vmax = max(float(np.max(np.abs(vf))), float(np.max(np.abs(v_al))))
    assert umax * dt_fast / dx <= c.momentum_courant_target + 1e-9
    assert vmax * dt_fast / de <= c.momentum_courant_target + 1e-9


def test_momentum_cfl_accounts_for_the_faster_aloft_layer():
    # the aloft departure u_aloft = u - 2*ubg (at frac=sign=1) can EXCEED the surface speed, so the CFL must
    # bound the aloft layer too -- a surface-only CFL would under-resolve the aloft cross-track advection. A
    # large cross-track wind is used to make the dx-direction limit BIND despite dx >> de on this coarse grid.
    t = Topography.procedural_continent(seed=3, nx=40, ne=40, Le_km=6000, width_km=3500, center_theta=48)
    c = StageCConfig(dt=900.0, momentum_advection=True, momentum_courant_target=0.5)
    m = StageCMap(t, c)
    ubg2d = np.broadcast_to(m.ubg[:, None], (m.nx, m.ne))
    u = np.abs(ubg2d) * 400.0; v = np.zeros((m.nx, m.ne))       # cross-track; large enough to bind the dx-CFL
    u_al, _ = blm.aloft_departure_wind(u, v, m.ubg, c.u_aloft_frac, c.aloft_departure_sign)
    assert float(np.max(np.abs(u_al))) > float(np.max(np.abs(u)))   # aloft genuinely faster here
    dt = m._momentum_cfl_dt(u, v)
    assert dt < c.dt                                               # the (faster) aloft cross-track wind shrank dt
    # dt is sized to the ALOFT (faster) cross-track speed over dx, not just the surface -> aloft Courant <= target
    assert float(np.max(np.abs(u_al))) * dt / m.dx <= c.momentum_courant_target + 1e-9


# ---- W2 gate: default OFF is byte-identical ----
def test_momentum_advection_default_off_is_byte_identical():
    # the flag defaults OFF, and an explicit False must reproduce the default path bit-for-bit (the W2 swap is
    # fully gated). Driven by a FIXED step count so the comparison is deterministic.
    assert StageCConfig().momentum_advection is False
    t = Topography.procedural_continent(seed=5, nx=40, ne=40, Le_km=6000, width_km=3500, center_theta=48)
    m_def = StageCMap(t, StageCConfig())
    m_off = StageCMap(t, StageCConfig(momentum_advection=False))
    for _ in range(300):
        m_def.step(); m_off.step()
    assert np.array_equal(m_def.P, m_off.P)
    assert np.array_equal(m_def.W_s, m_off.W_s) and np.array_equal(m_def.W_a, m_off.W_a)


def test_momentum_advection_on_changes_the_map_and_uses_the_momentum_wind():
    # ON must (a) change the precip map vs the diagnostic balance and (b) leave the REPORTED wind equal to the
    # momentum surface field (step() advects on it, so self.u/self.v hold it -- not the balance wind).
    t = Topography.procedural_continent(seed=5, nx=40, ne=40, Le_km=6000, width_km=3500, center_theta=48)
    m_bal = StageCMap(t, StageCConfig())
    m_mom = StageCMap(t, StageCConfig(momentum_advection=True, momentum_update_every=25))
    for _ in range(300):
        m_bal.step(); m_mom.step()
    assert not np.allclose(m_bal.P, m_mom.P)                     # the swap moves the deliverable
    # the reported wind is the cached momentum field (within the last sub-step)
    assert np.array_equal(m_mom.u, m_mom._u_mom) and np.array_equal(m_mom.v, m_mom._v_mom)
    # ... and that field is NOT the diagnostic balance wind
    u_bal, v_bal = m_mom._balanced_wind()
    assert not np.allclose(m_mom.u, u_bal)


# ---- W2 gate: ON runs stable, closes the water budget, no W_s/W_a runaway ----
@pytest.fixture(scope="module")
def momentum_map():
    t = Topography.procedural_continent(seed=3, nx=64, ne=64, Le_km=6000, width_km=3500, center_theta=48)
    m = StageCMap(t, StageCConfig(dt=900.0, check_every=300, tol=8e-4, patience=3, max_seconds=90,
                                  momentum_advection=True, momentum_update_every=50))
    m.run(verbose=False)
    return m


def test_momentum_advection_stable_and_physical(momentum_map):
    d = momentum_map.diagnostics()
    assert np.isfinite(d["P_mm_day"]).all() and np.isfinite(d["T_map_C"]).all()
    assert d["P_mm_day"].min() >= 0
    assert np.isfinite(momentum_map.W_s).all() and np.isfinite(momentum_map.W_a).all()


def test_momentum_advection_closes_water_budget(momentum_map):
    # the load-bearing gate (a magnitude fix must not break conservation): the swapped advecting wind still
    # closes int(E) == int(P) at steady state.
    d = momentum_map.diagnostics()
    assert 0.85 < d["water_closure_cum"] < 1.15
    assert 0.85 < d["water_closure_inst"] < 1.15


def test_momentum_advection_no_moisture_runaway(momentum_map):
    # no W_s/W_a runaway at the convergence nodes (the explicit roadmap W2 gate). The momentum wind is gentler
    # than the cap, so the columns stay BOUNDED -- well under a generous physical ceiling (a runaway would be
    # 1e3-1e5 mm; cf. the verified ungated-drain runaway W_s ~ 3e5 mm in OPEN_ISSUES).
    assert momentum_map.W_s.max() < 500.0
    assert momentum_map.W_a.max() < 500.0


# ---- W2 gate: band placement preserved (box-test-lite; the full canonical box-test is staged) ----
def test_momentum_advection_preserves_band_placement():
    # "A magnitude fix moves the pattern" risk: swapping the advecting wind must NOT walk the rain band off the
    # circulation-set peak. On the SAME world the momentum-advection zonal-mean peak latitude must stay within a
    # small tolerance of the balance-wind peak (W0 measured Delta=0.0 deg on canonical nsub30). Here we assert the
    # peak ROW is within +/-2 rows on a fixed-step pair (the in-session placement gate; W4 box-tests on canonical).
    t = Topography.procedural_continent(seed=3, nx=64, ne=64, Le_km=6000, width_km=3500, center_theta=48)
    m_bal = StageCMap(t, StageCConfig(dt=900.0, check_every=300, tol=8e-4, patience=3, max_seconds=90))
    m_mom = StageCMap(t, StageCConfig(dt=900.0, check_every=300, tol=8e-4, patience=3, max_seconds=90,
                                      momentum_advection=True, momentum_update_every=50))
    m_bal.run(verbose=False); m_mom.run(verbose=False)
    ip_bal = int(np.argmax(m_bal.P.mean(axis=1)))
    ip_mom = int(np.argmax(m_mom.P.mean(axis=1)))
    assert abs(ip_bal - ip_mom) <= 2                            # peak placement preserved (rate moves, not place)


# ---- the W3 fusion is now wired (momentum_advection + prognostic_phi); behavioral tests live in
#      test_stage_c_momentum_w3.py. Here we only confirm the constructor ACCEPTS the combination (the W2
#      rejection is retired) and still builds the in-loop momentum config. ----
def test_momentum_advection_accepts_prognostic_phi_w3_fusion():
    t = Topography.procedural_continent(seed=3, nx=24, ne=24, Le_km=6000, width_km=3500, center_theta=48)
    m = StageCMap(t, StageCConfig(momentum_advection=True, prognostic_phi=True))   # no longer raises (W3)
    assert m._mom_cfg is not None                                # the momentum solve is still set up
