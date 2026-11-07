"""Stage C tests: topography input + the dynamical 2-D map (two-layer moisture, closed water budget,
topology-driven asymmetry, maritime buffering)."""
import numpy as np
import pytest
from flatworld.topography import Topography
from flatworld.stage_c_map import StageCMap, StageCConfig, frictional_smoothing_km, V_CHAR_BL


# ---- physically-derived geopotential smoothing length (Refinement 2) ----
def test_phi_smooth_is_the_derived_frictional_length_not_a_free_fit():
    # Pinning test: phi_smooth_km must EQUAL the derived frictional/advective adjustment length
    # L_fric = V_char / r_fric (the f=0 smoothing scale), NOT an arbitrary swept number. This guards
    # against the cosmetic-de-skew trap -- the smoothing length is set by the friction/advection physics
    # (at the OPERATIVE, cap-saturated band/ribbon wind), never tuned until the ribbon-sea skew vanishes.
    cfg = StageCConfig()
    L = frictional_smoothing_km()                          # V_CHAR_BL / r_fric in km
    assert L == V_CHAR_BL / 1.0e-4 / 1.0e3                 # the closed form (no hidden fudge)
    assert abs(L - 400.0) < 1e-9                           # 400 km from the operative V_char = wind_cap, r=1e-4/s
    assert abs(V_CHAR_BL - cfg.wind_cap) < 1e-9            # the operative wind IS the cap (saturated, measured)
    assert cfg.phi_smooth_km == L                          # the config default IS the derived frictional scale
    # and it is tied to the cap/drag physics, not an arbitrary number: L_fric == wind_cap / r_fric (km)
    assert abs(cfg.phi_smooth_km - cfg.wind_cap / cfg.r_fric / 1.0e3) < 1e-9
    # issue #88 item 4: overriding wind_cap or r_fric RE-DERIVES phi_smooth_km (the fixed point stays in sync
    # instead of de-syncing against a once-computed default); an explicit phi_smooth_km is honored verbatim.
    cfg2 = StageCConfig(wind_cap=20.0, r_fric=2.0e-4)
    assert abs(cfg2.phi_smooth_km - 20.0 / 2.0e-4 / 1.0e3) < 1e-9       # re-derived = 100 km
    assert StageCConfig(phi_smooth_km=137.0).phi_smooth_km == 137.0     # explicit override honored as-is


def test_config_stage_c_resyncs_phi_smooth_on_wind_cap_or_r_fric_sweep():
    # PR #95 review: config.stage_c uses dataclasses.replace, which re-passes the ALREADY-RESOLVED
    # phi_smooth_km from the base profile config -- so without a guard a wind_cap/r_fric sweep through this
    # production entry point would silently keep the stale 400 km (the exact de-sync issue #88 item 4 targets,
    # but via the documented sweep path -- direct StageCConfig() construction alone did not cover it).
    from flatworld import config as cfgmod
    assert abs(cfgmod.stage_c("production").phi_smooth_km - 400.0) < 1e-9          # default still 400 km
    swept_r = cfgmod.stage_c("production", r_fric=2.0e-4)
    assert abs(swept_r.phi_smooth_km - swept_r.wind_cap / 2.0e-4 / 1.0e3) < 1e-9   # re-derived (=200 km), not 400
    swept_w = cfgmod.stage_c("test", wind_cap=20.0)
    assert abs(swept_w.phi_smooth_km - 20.0 / swept_w.r_fric / 1.0e3) < 1e-9       # re-derived (=200 km), not 400
    # an explicit phi_smooth_km through stage_c is still honored verbatim, even alongside a r_fric sweep
    assert cfgmod.stage_c("production", r_fric=2.0e-4, phi_smooth_km=321.0).phi_smooth_km == 321.0


def test_production_ships_B_supply_on_total_reduction_test_and_default_stay_off():
    # J1 default-flip (2026-06-19): B (resolved_supply) is ON in PRODUCTION as total-reduction, normalized by
    # the frozen all-sea E_ref. The "test" profile and the bare StageCConfig default stay OFF (E_ref is
    # canonical-world-specific, and the byte-identical/standalone paths must be preserved). C stays OFF.
    from flatworld import config as cfgmod
    from flatworld.stage_c_map import StageCConfig
    p = cfgmod.stage_c("production")
    assert p.resolved_supply is True
    assert p.supply_zonal_mixing == 0.0
    assert p.supply_ref_export == cfgmod.SUPPLY_EREF_FROZEN > 0.0    # total-reduction (not B1 redistribute)
    assert p.prognostic_phi is False                                 # C secondary -> stays off
    assert cfgmod.stage_c("test").resolved_supply is False           # test profile off (no valid E_ref)
    assert StageCConfig().resolved_supply is False                   # bare default off (byte-identical paths)


# ---- terrain-geopotential recalibration (2026-06-17): g_terrain double-counted the orographic high ----
def test_terrain_geopotential_does_not_saturate_land_winds():
    """The terrain term g_terrain*g*h must NOT dominate the BL geopotential and peg the diagnostic wind at
    wind_cap over land. The orographic high is ALREADY in the lapse-cooled T_map (via -beta_T*(T-Tref)); the
    old explicit g_terrain=0.6 double-counted it (~30x, full g*h scaling), so even moderate (~2.5 km) relief
    saturated the cap. With the principled default (g_terrain=0) the land wind is physical and orographic
    steering survives through the thermal term. (Stage C wind is f=0 frictional, magnitude diagnostic-soft.)"""
    t = Topography.procedural_continent(seed=7, nx=48, ne=48, Le_km=6000)
    assert np.median(t.height[t.land]) > 1500.0              # the topo really has moderate relief (test has teeth)
    m = StageCMap(t, StageCConfig())                          # default g_terrain (now 0.0)
    assert m.cfg.g_terrain == 0.0
    u, v = m._balanced_wind()
    sp = np.hypot(np.asarray(u), np.asarray(v)); land = t.land
    assert np.median(sp[land]) < 0.5 * m.cfg.wind_cap         # land wind physical, not pegged at the cap
    assert (sp[land] >= m.cfg.wind_cap - 0.5).mean() < 0.05   # ~no land cells saturate the cap
    # and the OLD g_terrain=0.6 DID peg the cap over the same moderate relief (locks WHY the default changed)
    u2, v2 = StageCMap(t, StageCConfig(g_terrain=0.6))._balanced_wind()
    assert np.median(np.hypot(np.asarray(u2), np.asarray(v2))[land]) >= 0.9 * m.cfg.wind_cap


# ---- topography ----
def test_topography_uniform_in_distance():
    t = Topography.procedural_continent(seed=1, nx=40, ne=40, Le_km=5000)
    assert np.allclose(np.diff(t.x), t.dx)             # uniform in physical distance
    dth = np.diff(t.theta_deg)
    assert dth.max() / dth.min() > 1.5                 # non-uniform in theta
    assert t.land.any() and not t.land.all()           # a real coastline
    assert t.height[t.land].max() > 0 and np.all(t.height[~t.land] == 0)


def test_topography_distance_to_coast():
    t = Topography.procedural_continent(seed=2, nx=50, ne=50)
    d = t.distance_to_coast_km()
    assert d[t.land].max() > 0 and np.all(d[~t.land] == 0)


# ---- f_loft_of_theta override (the plume-anchored convective/stratiform split, OPEN_ISSUES Brief 2) ----
def test_f_loft_of_theta_overrides_scalar_per_row():
    t = Topography.procedural_continent(seed=3, nx=40, ne=40, Le_km=6000)
    # a callable that returns a recognizable per-theta ramp
    fn = lambda th: np.clip(0.3 + 0.01 * (np.asarray(th, float) - 40.0), 0.0, 1.0)
    m = StageCMap(t, StageCConfig(f_loft_of_theta=fn))
    assert m.f_loft.shape == (m.nx,)                        # per cross-track row, not a scalar
    assert np.allclose(m.f_loft, fn(t.theta_deg))           # the callable set the split
    # default (no callable) -> the scalar broadcast
    m0 = StageCMap(t, StageCConfig(f_loft=0.5))
    assert np.allclose(m0.f_loft, 0.5)


def test_f_loft_of_theta_changes_conv_stratiform_partition():
    # raising f_loft (more lofted) must REDUCE convective rain-in-place and feed more aloft supply
    t = Topography.procedural_continent(seed=4, nx=40, ne=40, Le_km=6000)
    base = dict(dt=900.0, check_every=300, tol=5e-4, patience=3, max_seconds=30)
    lo = StageCMap(t, StageCConfig(f_loft_of_theta=lambda th: np.full_like(np.asarray(th, float), 0.2), **base))
    hi = StageCMap(t, StageCConfig(f_loft_of_theta=lambda th: np.full_like(np.asarray(th, float), 0.8), **base))
    lo.run(verbose=False); hi.run(verbose=False)
    # more lofting -> more aloft water carried (W_a higher), and the budget still closes
    assert hi.W_a.mean() > lo.W_a.mean()
    assert 0.7 < hi.diagnostics()["water_closure_cum"] < 1.3


# ---- external topology INPUT (alternative to procedural generation) ----
def test_from_arrays_wraps_and_validates_external_topology():
    # the input path: take arrays (here borrowed from a procedural grid, standing in for an external
    # geophysics model's output), wrap with from_arrays, and run Stage C over them unchanged.
    base = Topography.procedural_continent(seed=7, nx=40, ne=40, Le_km=6000)
    topo = Topography.from_arrays(base.x, base.e, base.land, base.height)
    assert topo.meta["kind"] == "external"                 # tagged as supplied, not procedural
    assert np.array_equal(topo.land, base.land) and np.allclose(topo.height, base.height)
    m = StageCMap(topo, StageCConfig(dt=900.0, check_every=300, tol=5e-4, patience=3, max_seconds=30))
    m.run(verbose=False)
    d = m.diagnostics()
    assert np.isfinite(d["P_mm_day"]).all() and d["P_mm_day"].min() >= 0


def test_from_arrays_rejects_grids_stage_c_cannot_use():
    base = Topography.procedural_continent(seed=7, nx=30, ne=30, Le_km=6000)
    # non-uniform x (Stage C reads a single dx) -> rejected
    x_bad = base.x.copy(); x_bad[5] += 0.5 * base.dx
    with pytest.raises(ValueError):
        Topography.from_arrays(x_bad, base.e, base.land, base.height)
    # mismatched field shape -> rejected
    with pytest.raises(ValueError):
        Topography.from_arrays(base.x, base.e, base.land[:, :10], base.height)
    # opting out of validation lets a deliberately odd grid through
    odd = Topography.from_arrays(x_bad, base.e, base.land, base.height, validate=False)
    assert odd.land.shape == (30, 30)


def test_topography_save_load_roundtrip(tmp_path):
    base = Topography.procedural_continent(seed=8, nx=30, ne=30, Le_km=6000)
    p = str(tmp_path / "topo.npz")
    base.save(p)
    got = Topography.load(p)
    assert np.allclose(got.x, base.x) and np.allclose(got.e, base.e)
    assert np.array_equal(got.land, base.land) and np.allclose(got.height, base.height)
    assert got.meta["source"] == p


def test_resolve_topo_input_paths():
    # the cascade's topology selector: default generates, supplied topology is taken as-is.
    from flatworld.cascade import _resolve_topo
    from flatworld.topography import TopoGrid
    gen = _resolve_topo(None, "test", 3, None)             # default -> procedural
    assert isinstance(gen, TopoGrid) and gen.meta["kind"] == "procedural"
    base = Topography.procedural_continent(seed=1, nx=30, ne=30, Le_km=6000)
    assert _resolve_topo(base, "test", 0, None) is base    # a TopoGrid passes straight through (validated)
    viadict = _resolve_topo(dict(x=base.x, e=base.e, land=base.land, height=base.height), "test", 0, None)
    assert isinstance(viadict, TopoGrid) and viadict.meta["kind"] == "external"
    with pytest.raises(ValueError):                        # cannot combine procedural kwargs with a topology
        _resolve_topo(base, "test", 0, {"nx": 40})
    with pytest.raises(TypeError):                         # an unsupported type is rejected
        _resolve_topo(12345, "test", 0, None)


# ---- drain subsidence gate (Refinement 1: convective drain suppressed under large-scale subsidence) ----
def test_drain_suppressed_under_subsidence():
    # Pinning test: the LOCAL convective rain must be SUPPRESSED on the map where the single cell subsides
    # (the per-row subsidence weight -> 0), EVEN over a warm surface (T_map > conv_sst_C), and ALLOWED where
    # it ascends. This is the physical content of Refinement 1 -- deep convection is capped from above by
    # descending air, so the warm subsiding sea's drained moisture becomes surface-vapor export to the
    # equatorial pathway (vapor_export, a budget sink) rather than rained locally on the map. (The drain SINK
    # on W_s is left ungated to bound W_s -- gating it to zero runs the surface layer away; see step().)
    topo = Topography.procedural_continent(seed=3, nx=40, ne=40, Le_km=6000, width_km=4000, center_theta=45)
    th = topo.theta_deg
    th_mid = 0.5 * (th.min() + th.max())
    warm = lambda t: np.full_like(np.asarray(t, float), 45.0)         # warm sea everywhere -> conv_active True
    subs = lambda t: (np.asarray(t, float) <= th_mid).astype(float)   # ASCEND equatorward (1), SUBSIDE poleward (0)
    noband = lambda t: np.zeros_like(np.asarray(t, float))            # F=0: no aloft stratiform supply to confound
    # f_loft=0 so the convective drain rains entirely IN PLACE (P_conv=drain) and nothing detrains aloft --
    # this isolates the DRAIN: there is no aloft layer to carry moisture poleward and rain it as stratiform
    # over the subsiding rows (which would be physically correct but would mask the drain-gating signal).
    base = dict(sst_C_of_theta=warm, aloft_supply_of_theta=noband, f_loft=0.0, dt=900.0,
                check_every=300, tol=5e-4, patience=3, max_seconds=60)
    m = StageCMap(topo, StageCConfig(subsidence_of_theta=subs, **base))
    asc = th <= th_mid; sub = th > th_mid
    # the gate is built correctly: 1 over ascending rows, 0 over subsiding rows
    assert np.allclose(m.subs_gate[asc], 1.0) and np.allclose(m.subs_gate[sub], 0.0)
    m.run(verbose=False)
    P = m.diagnostics()["P_mm_day"]; sea = ~topo.land
    asc_sea = (asc[:, None] & sea); sub_sea = (sub[:, None] & sea)
    assert P[asc_sea].mean() > 1.0                                    # warm ASCENDING sea convects
    assert P[sub_sea].max() < 1e-3                                    # warm SUBSIDING sea is gated dry
    # turning the gate OFF (drain_subsidence_gate=False) restores convection over the subsiding rows
    m2 = StageCMap(topo, StageCConfig(subsidence_of_theta=subs, drain_subsidence_gate=False, **base))
    assert np.allclose(m2.subs_gate, 1.0)                            # disabled -> no-op (all ones)
    m2.run(verbose=False)
    assert m2.diagnostics()["P_mm_day"][sub_sea].mean() > 1.0         # without the gate, the warm sea rains


def test_drain_gate_budget_closes_and_Ws_bounded():
    # Pinning test (Refinement 1/1b): with an ACTIVE subsidence gate the water budget must still close and
    # W_s must stay BOUNDED. The drained moisture under subsidence is routed to surface-vapor export (a budget
    # sink), with the drain left ungated so it bounds W_s. The rejected alternatives (gate the drain to zero,
    # or advect out a theta~28 edge outflow) RUN AWAY at the diagnostic wind's convergence nodes (W_s ~3e5 mm,
    # closure -> ~0.6-0.87) -- this test would catch a regression back to either.
    topo = Topography.procedural_continent(seed=3, nx=60, ne=40, Le_km=6000, width_km=3500, center_theta=45)
    sst = lambda th: np.interp(th, [28, 36, 44, 51, 62], [52, 45, 33, 12, -18])   # wet ocean spanning the band/ribbon
    # subside equatorward of ~37 and poleward of ~44 (the inner sea + ribbon), ascend in the band branch
    subs = lambda t: ((np.asarray(t, float) >= 37.0) & (np.asarray(t, float) <= 44.0)).astype(float)
    band = lambda th: np.interp(th, [28, 36, 44, 62], [8, 50, 20, 6])
    m = StageCMap(topo, StageCConfig(sst_C_of_theta=sst, aloft_supply_of_theta=band, subsidence_of_theta=subs,
                                     dt=900.0, check_every=300, tol=5e-4, patience=3, max_seconds=60))
    m.run(verbose=False)
    d = m.diagnostics()
    assert 0.85 < d["water_closure_cum"] < 1.15           # budget closes despite the gate (no runaway / no loss)
    assert m.W_s.max() < 5.0e4                             # W_s BOUNDED (a runaway hits ~3e5+; the drain bounds it)
    assert d["vapor_export_mm_day"] > 0.0                  # the export channel is actually carrying moisture


def test_subsidence_gate_default_is_noop_standalone():
    # No-regression guard: with no subsidence_of_theta (the standalone/test default) the drain gate is a
    # NO-OP (subs_gate == 1 everywhere), so step() is bit-identical to the pre-Refinement-1 code.
    topo = Topography.procedural_continent(seed=3, nx=40, ne=40, Le_km=6000, width_km=3500, center_theta=48)
    m = StageCMap(topo, StageCConfig())
    assert np.allclose(m.subs_gate, 1.0)


@pytest.fixture(scope="module")
def relaxed_map():
    topo = Topography.procedural_continent(seed=3, nx=70, ne=70, Le_km=6000, width_km=3500, center_theta=48)
    m = StageCMap(topo, StageCConfig(dt=900.0, check_every=300, tol=5e-4, patience=3, max_seconds=60))
    m.run(verbose=False)
    return m


def test_runs_stable_and_physical(relaxed_map):
    d = relaxed_map.diagnostics()
    assert np.isfinite(d["P_mm_day"]).all() and np.isfinite(d["T_map_C"]).all()
    assert d["P_mm_day"].min() >= 0
    assert np.abs(d["u"]).max() <= 40.1 and np.abs(d["v"]).max() <= 40.1   # wind cap respected


def test_closed_water_budget(relaxed_map):
    # cumulative sources vs sinks must balance (conservation): closure -> 1
    d = relaxed_map.diagnostics()
    assert 0.85 < d["water_closure_cum"] < 1.15
    # issue #88 item 2: the INSTANTANEOUS closure now counts the F(theta) aloft-band source S_band too, so at
    # steady state it agrees with the cumulative (~1.0) instead of sitting structurally ~1.6-1.7 (the old
    # missing-source reading the prior c5 misattributed to EMA lag).
    assert 0.85 < d["water_closure_inst"] < 1.15


def test_topology_breaks_alongtrack_symmetry(relaxed_map):
    # the POINT of a 2-D map: precip varies along-track (e) where topology does -- not a 1-D profile
    d = relaxed_map.diagnostics()
    P = d["P_mm_day"]
    th = d["theta_deg"]
    rib = (th >= 43) & (th <= 51)
    along_track_std = P[rib].std(axis=1)               # variation along e in each ribbon row
    assert np.mean(along_track_std) > 1.0              # real along-track structure


def test_two_layer_moisture_is_poleward_aloft(relaxed_map):
    # the fix: the aloft layer carries moisture POLEWARD relative to the surface layer (which advects it
    # equatorward). The robust, node-independent signature is the CONTRAST between the layers: the surface
    # moisture flux <u*W_s> is equatorward (negative; surface wind ubg<0) while the aloft flux <u_aloft*W_a>
    # is far more poleward, so (aloft - surface) is large and positive. (A bare "aloft flux > 0" is NOT a
    # clean invariant: pre-Brief-1, the rate-limited rainout let W_a pile up super-saturated at cold-edge
    # convergence nodes -- ~28% of cells >3x Wa_sat -- and those nodes alone made the crude domain-mean
    # aloft flux positive; with the capacity-limited rainout (Brief 1) holding W_a<=Wa_sat the warm-source
    # equatorward recirculation sets the sign of the raw domain mean, so we test the layer CONTRAST, which
    # is the actual two-layer claim and is robust across seeds and capacity factors. OPEN_ISSUES c2/Brief 1.)
    m = relaxed_map
    ubg2d = np.broadcast_to(m.ubg[:, None], (m.nx, m.ne))
    u_aloft = -m.cfg.u_aloft_frac * ubg2d + (m.u - ubg2d)
    aloft_flux = float(np.mean(u_aloft * m.W_a))      # mean cross-track aloft moisture flux
    surf_flux = float(np.mean(m.u * m.W_s))           # mean cross-track surface moisture flux
    assert surf_flux < 0                              # surface layer advects moisture EQUATORWARD
    assert aloft_flux - surf_flux > 10.0              # aloft layer is substantially MORE poleward


def test_aloft_departure_sign_default_is_plus_one_and_flips_the_pattern():
    # issue #88 item 1: the aloft-departure sign is a FLAGGED knob defaulting to +1 (the shipped choice).
    # +1 (or unset) must be byte-identical; -1 (the return-branch vent) must change the precip map. Driven by
    # a FIXED step count (not the time-guarded run()) so the +1==default comparison is deterministic.
    assert StageCConfig().aloft_departure_sign == 1.0                 # shipped default
    t = Topography.procedural_continent(seed=5, nx=40, ne=40, Le_km=6000, width_km=3500, center_theta=48)
    m_def = StageCMap(t, StageCConfig())
    m_pos = StageCMap(t, StageCConfig(aloft_departure_sign=+1.0))
    m_neg = StageCMap(t, StageCConfig(aloft_departure_sign=-1.0))
    for _ in range(400):
        m_def.step(); m_pos.step(); m_neg.step()
    assert np.array_equal(m_def.P, m_pos.P)                           # +1 == unset default (byte-identical)
    assert not np.allclose(m_def.P, m_neg.P)                          # -1 changes the precip pattern


def test_maritime_buffering_sea_holds_SST(relaxed_map):
    # sea cells carry the (buffered) SST backbone; land interior is cooler/continental at the same lat
    m = relaxed_map; d = m.diagnostics()
    T = d["T_map_C"]; th = m.theta_deg
    row = np.argmin(np.abs(th - 49))                   # a ribbon-latitude row
    if m.sea[row].any() and m.land[row].any():
        assert T[row][m.sea[row]].mean() >= T[row][m.land[row]].mean() - 1.0   # sea no colder than land mean


def test_box_size_zonalmean_robust():
    # widening the along-track box must not change the ZONAL-MEAN ribbon rain much (a climate-set
    # quantity), even though per-continent detail (geometry-set) changes. CAVEAT that sets the test
    # design: the procedural continent is RE-GENERATED for each Le (the generator reseeds with the
    # domain), so a SINGLE-seed box pair is dominated by continent-realization noise, not domain
    # starvation -- single-seed pairs swing 15-47%, which is why the old tolerance had to be a
    # near-meaningless 0.6. The robust, physical test is whether the SEED-AVERAGED zonal mean is
    # box-independent ABOVE the converged box floor (Le ~ 12,000 km; below it the box starves -- see
    # OPEN_ISSUES box-size sweep). Averaging over continents at two above-floor extents, run to their
    # own convergence, the drift is ~0.13 (measured) -- a real constraint.
    def ribbon_mean(ne, Le, seed):
        topo = Topography.procedural_continent(seed=seed, nx=48, ne=ne, Le_km=Le, width_km=3000, center_theta=48)
        m = StageCMap(topo, StageCConfig(dt=900.0, check_every=300, tol=8e-4, max_seconds=120))
        m.run(verbose=False)
        d = m.diagnostics(); th = d["theta_deg"]; rib = (th >= 43) & (th <= 51)
        return d["P_mm_day"][rib].mean()
    seeds = (0, 5, 11)
    p1 = np.mean([ribbon_mean(160, 12000, s) for s in seeds])   # at the box floor
    p2 = np.mean([ribbon_mean(240, 18000, s) for s in seeds])   # 1.5x, still above the floor
    # tol RE-FOUNDED 0.10 -> 0.20 with the supply saturation cap (2026-07-18, wa_supply_sat_cap): the
    # old 0.057-0.10 robustness was mostly the ARTIFICIAL smoothness of the uncapped supply pump -- on
    # this standalone fixture ~75% of the ribbon P was the analytic (F - Wsat)/tau term (46.8 of ~47
    # mm/day uncapped vs ~12 capped), which is grid/box-trivial by construction. The capped statistic
    # is the genuine flow-set rain, whose grid/box character carries the documented del^2(Phi)
    # convergence-node noise (module header; the reason Stage-C mm/day is diagnostic-soft). Measured
    # capped pair: 0.105. The gate still catches the 2-cell-era gross failures (0.3-0.6).
    assert abs(p2 - p1) / max(p1, 1.0) < 0.20          # seed-averaged zonal mean is box-independent above floor


def test_resolution_sensitivity_zonalmean():
    # refining the grid must not strongly change the ZONAL-MEAN ribbon climate (a converged solution
    # is grid-independent). Same continent (same seed/extent), two resolutions IN THE CONVERGED REGIME:
    # the coarse end (nx=50) is still under-resolved (nx50->nx90 moves ~29%), so we compare two
    # already-fine grids where the solution has converged -- nx=90 -> nx=130 moves only ~10% (measured),
    # a genuine grid-independence constraint (vs the old lax 0.6 that the 2-cell-era band slipped under).
    def ribbon_mean(nx, ne):
        topo = Topography.procedural_continent(seed=4, nx=nx, ne=ne, Le_km=6000, width_km=3000, center_theta=48)
        m = StageCMap(topo, StageCConfig(dt=900.0, check_every=300, tol=8e-4, max_seconds=60))
        m.run(verbose=False)
        d = m.diagnostics(); th = d["theta_deg"]; rib = (th >= 43) & (th <= 51)
        return d["P_mm_day"][rib].mean()
    p_coarse = ribbon_mean(90, 90)
    p_fine = ribbon_mean(130, 130)                      # ~1.4x finer, both in the converged regime
    # tol RE-FOUNDED 0.10 -> 0.30 with the supply saturation cap (2026-07-18, wa_supply_sat_cap): the
    # old 0.064-0.10 "grid convergence" was mostly the uncapped supply pump -- an analytic
    # (F - Wsat)/tau profile that is grid-trivial (uncapped this fixture reads 44.5-46.8 mm/day and
    # "converges" to 5%; capped it reads the real flow-set rain, 9.5-14.2 across nx 90/130/170/210/250,
    # oscillating +/-20% with no trend -- the documented del^2(Phi) convergence-node character, the
    # named wall behind "Stage-C mm/day is diagnostic-soft"). This gate now locks the ORDER of the
    # dynamical rain across grids (catches the 2-cell-era 0.3-0.6 gross failures), not a precision the
    # model does not have.
    assert abs(p_fine - p_coarse) / max(p_coarse, 1.0) < 0.30  # zonal-mean grid-robust at the soft tier


def test_checkpoint_resume_reproduces_straightthrough():
    # Checkpoint/resume must reproduce a straight-through run bit-for-bit. This also guards the choice
    # to persist only the prognostic moisture (W_s, W_a, P) and NOT the diagnostic wind (u, v), which
    # step() recomputes from the static geopotential before use -- so dropping u/v changes nothing.
    import os, tempfile
    from flatworld import convergence as conv
    topo = Topography.procedural_continent(seed=3, nx=40, ne=40, Le_km=6000, width_km=3500, center_theta=48)
    cfg = StageCConfig(dt=900.0)
    A = StageCMap(topo, cfg)                            # straight through: 400 steps
    for _ in range(400):
        A.step()
    B = StageCMap(topo, cfg)                            # checkpoint at 200 ...
    for _ in range(200):
        B.step()
    ck = os.path.join(tempfile.gettempdir(), "flatworld_stagec_ckpt_test.npz")
    conv.save_checkpoint(ck, B.state(), {"iter": B.iter})
    C = StageCMap(topo, cfg)                            # ... fresh object resumes and runs 200 more
    st, _ = conv.load_checkpoint(ck)
    C.load_state(st)
    assert C.iter == 200
    for _ in range(200):
        C.step()
    os.remove(ck)
    assert C.iter == A.iter == 400
    for a, c in ((A.W_a, C.W_a), (A.W_s, C.W_s), (A.P, C.P)):
        assert np.array_equal(a, c)                    # bit-for-bit
    # issue #88 item 4: the water-budget accumulators + export EMA now persist too, so a resumed run reports
    # closure over the WHOLE run (not just the post-resume window) -- they must match the straight-through run.
    assert C._cum_E == A._cum_E and C._cum_P == A._cum_P and C.vapor_export == A.vapor_export


def test_checkpoint_tolerates_legacy_u_v_keys():
    # Older checkpoints also stored the (now dropped) diagnostic wind u, v; load_state must ignore them.
    topo = Topography.procedural_continent(seed=2, nx=40, ne=40, Le_km=6000)
    m = StageCMap(topo, StageCConfig(dt=900.0))
    for _ in range(50):
        m.step()
    legacy = dict(u=m.u, v=m.v, W_s=m.W_s, W_a=m.W_a, P=m.P, iter=m.iter)
    m2 = StageCMap(topo, StageCConfig(dt=900.0))
    m2.load_state(legacy)
    assert m2.iter == m.iter and np.array_equal(m2.W_a, m.W_a)


def test_standalone_default_wetness_is_unity():
    # c2 fix, no-regression guard: the standalone default SST backbone never exceeds 55 C, so A's
    # 55->70 C wetness ramp is identically 1 -> the salt-flat evaporation gate is a NO-OP for the
    # default/test configs (E_sea is bit-identical to the pre-fix code). The gate only bites in the
    # cascade, where A's hot-zone SST reaches 70-126 C and the ramp falls to 0.
    topo = Topography.procedural_continent(seed=3, nx=60, ne=40, Le_km=6000, width_km=3500, center_theta=48)
    m = StageCMap(topo, StageCConfig())
    assert np.allclose(m.wetness, 1.0)


def test_evaporation_gated_by_wetness_over_saltflat():
    # c2 fix: Stage C's sea-surface evaporation must be gated by A's wetness ramp -> ZERO over the
    # salt-flat hot zone (wetness->0 where SST>70 C), even though those cells are topographically 'sea'
    # with boiling SST. Checks the source term with the model's own formula, and that no blow-up forms.
    from flatworld.stage_c_map import _wsat_column
    from flatworld import constants as C
    topo = Topography.procedural_continent(seed=3, nx=60, ne=40, Le_km=6000, width_km=3500, center_theta=40)
    sst = lambda th: np.interp(th, [28, 36, 40, 45, 51, 62], [95, 78, 50, 30, 5, -20])     # salt flat -> wet ocean
    wet = lambda th: np.clip((70.0 - sst(th)) / 15.0, 0.0, 1.0)                            # A's 55->70 C ramp
    m = StageCMap(topo, StageCConfig(sst_C_of_theta=sst, wetness_of_theta=wet, dt=900.0))
    dry = m.wetness <= 1e-9                                            # salt-flat rows (ramped to 0)
    wetrows = m.wetness > 0.99                                         # wet-ocean rows
    assert dry.any() and wetrows.any()                                # backbone spans salt flat -> ocean
    # evaporation source evaluated with the model's formula (a few steps of spin-up first)
    for _ in range(200):
        m.step()
    Wsat_sfc = _wsat_column(m.T_map_C + C.T_FREEZE)
    E_sea = m.wet_col * np.where(m.sea, np.maximum(Wsat_sfc - m.W_s, 0.0) / m.cfg.tau_evap, 0.0)
    assert np.all(E_sea[dry] == 0.0)                                  # NO evaporation over the salt flat
    if m.sea[wetrows].any():
        assert E_sea[wetrows][m.sea[wetrows]].max() > 0.0            # the wet ocean DOES evaporate
    # the salt-flat sea cells stay ~dry (no spurious open-ocean source) and produce no blow-up
    assert m.W_s[dry][m.sea[dry]].mean() < 1.0 if m.sea[dry].any() else True
    assert m.diagnostics()["P_mm_day"].max() < 5000.0                 # no 1e4+ hot-zone spike


def test_capacity_limited_rainout_no_supersaturation(relaxed_map):
    # Brief 1 (OPEN_ISSUES): the capacity-limited aloft rainout must prevent a convergence/detrainment
    # node from piling the aloft layer up super-saturated (the c2 residual: W_a -> many*Wa_sat). After
    # convergence, no meaningful-capacity cell (Wa_sat > 10 mm; the benign cold-edge tiny-capacity
    # artifact is excluded, as the c2 measurement did) may hold more than ~3x its saturation. With the
    # frozen default wa_capacity_factor=1.0 the ceiling pins W_a <= Wa_sat each step, so the ratio is ~1.
    m = relaxed_map
    assert m.cfg.wa_capacity_factor == 1.0                 # the frozen principled default (rain to Wa_sat)
    Wa_sat = m.Wsat_aloft[:, None]                         # (nx,1) aloft saturation capacity
    meaningful = np.broadcast_to(Wa_sat > 10.0, m.W_a.shape)
    ratio = (m.W_a / np.maximum(Wa_sat, 1e-9))[meaningful]
    assert ratio.size > 0                                  # some cells have real capacity
    assert ratio.max() <= 3.0                              # NO super-saturation node survives


def test_capacity_factor_caps_aloft_water():
    # The knob does what it says: a larger wa_capacity_factor permits a (proportionally) higher aloft
    # column before the ceiling fires, and the ceiling is respected at convergence. A node-prone config
    # (strong band supply over a warm sea, short tau_supply) drives W_a hard against the ceiling.
    topo = Topography.procedural_continent(seed=3, nx=60, ne=40, Le_km=6000, width_km=3500, center_theta=48)
    big_supply = lambda th: np.full_like(th, 200.0)        # oversupply: forces the ceiling to bite
    def maxratio(fac):
        m = StageCMap(topo, StageCConfig(aloft_supply_of_theta=big_supply, wa_capacity_factor=fac,
                                         dt=900.0, check_every=300, tol=5e-4, patience=3, max_seconds=60))
        m.run(verbose=False)
        Wa_sat = m.Wsat_aloft[:, None]
        meaningful = np.broadcast_to(Wa_sat > 10.0, m.W_a.shape)
        return float((m.W_a / np.maximum(Wa_sat, 1e-9))[meaningful].max())
    r1 = maxratio(1.0); r2 = maxratio(2.0)
    assert r1 <= 1.05 and r2 <= 2.05                       # ceiling respected (small overshoot from the rate sub-step)
    assert r2 > r1 + 0.5                                   # the knob genuinely raises the held capacity


# ---- resolved diurnal cycle in the temperature map (Issue #21) ----
# CHARACTERIZATION TESTS vs CONTRACT TESTS -- read before "fixing" a failure in this section.
# Some tests below pin an EMERGENT physical result (a sign or a rough magnitude that falls out of the surface
# energy balance), NOT an invariant. They are tagged "CHARACTERIZATION". If a code change breaks one: STOP
# and sanity-check the change -- but do NOT assume the change must therefore be wrong, and NEVER tune the
# model, or loosen/retarget a threshold, just to make it pass. Real physics (a new mechanism, a fixed bug, a
# different forcing or constant) can LEGITIMATELY move the sign/magnitude; if that is what happened, re-derive
# the expected value and UPDATE the test to the new physics, with a note on why it moved. These tests exist to
# make such a move VISIBLE and deliberate -- not to freeze a number the model must be bent back to.
# (The OTHER tests here -- the algebraic no-regression, callable-override, skew, and frost-logic checks -- are
# CONTRACTS: they encode intended behavior and should hold by construction, so a break there is a real
# regression to fix in the code, not a finding.)
def test_diurnal_cycle_sea_flat_land_swings():
    # CHARACTERIZATION (the land-swing magnitude is emergent from the standalone surface solve; see the
    # section note above -- careful, don't tune-to-pass). Sea ~0 swing and the T_min<=T_map<=T_max ordering
    # are contracts; the land swing is the emergent part.
    # The core of Issue #21: the (x,e) map now carries a diurnal range. Sea is maritime-buffered (~0 swing,
    # huge mixed-layer heat capacity); land genuinely swings (tens of K); and T_min <= T_map <= T_max.
    topo = Topography.procedural_continent(seed=3, nx=60, ne=60, Le_km=6000, width_km=3500, center_theta=48)
    m = StageCMap(topo, StageCConfig())
    assert np.all(m.T_min_C <= m.T_map_C + 1e-9) and np.all(m.T_map_C <= m.T_max_C + 1e-9)
    assert m.dT_diurnal_C[m.sea].max() <= m.cfg.diurnal_range_sea_K + 1e-9   # sea: the small marine swing (~1 K)
    assert m.dT_diurnal_C[m.land].mean() > 8.0            # land genuinely swings (tens of K) -- the new content


def test_diurnal_interior_swings_more_than_coast():
    # CHARACTERIZATION (interior>coast swing is an emergent heat-capacity result; see the section note above --
    # careful, don't tune-to-pass).
    # continentality is now a SWING, not a fixed mean offset: a dry interior (low thermal inertia, C_ARID_LAND)
    # swings MORE than the damp coast (C_MOIST_LAND) -- the heat-capacity law. Split land by MEDIAN distance-to-
    # coast (robust to the grid spacing, which can exceed a fixed km threshold on a coarse grid).
    topo = Topography.procedural_continent(seed=3, nx=80, ne=70, Le_km=8000, width_km=5000, center_theta=46)
    m = StageCMap(topo, StageCConfig())
    dc = topo.distance_to_coast_km(); med = np.median(dc[m.land])
    near = m.land & (dc <= med); far = m.land & (dc > med)
    assert near.any() and far.any()
    assert m.dT_diurnal_C[far].mean() > m.dT_diurnal_C[near].mean() + 2.0     # swing grows inland


def test_diurnal_daymean_is_legacy_continentality_model_standalone():
    # CONTRACT (deliberate invariant, NOT a characterization): the standalone day-mean is meant to stay
    # algebraically identical to the prior model, so a break here IS a real regression -- unless you
    # intentionally changed the standalone default, in which case update this with that intent stated.
    # NO-REGRESSION on the validated day-MEAN: with the standalone default interior mean = coast -
    # continentality_K, the day-mean map is ALGEBRAICALLY the prior model
    #   T = coast - lapse*h - continentality_K*(1 - exp(-d_coast/L_cont)),
    # so the swing (T_min/T_max) is purely ADDITIVE -- the classification backbone is unchanged. (The cascade
    # supersedes the interior mean with the surface-EB solve, which on a seasonless world runs WARMER; that is
    # the intended physical change, exercised in the cascade tests, not here.)
    # 2026-07-18 INTENT UPDATE: the standalone default now ALSO applies maritime moderation
    # (maritime_L_km=610, deliberate -- the sharp shoreline step was the bug, see
    # test_maritime_coast.py which pins the ON algebra exactly). The legacy identity below is the
    # contract for the maritime_L_km=0 escape.
    topo = Topography.procedural_continent(seed=5, nx=50, ne=50, Le_km=6000, width_km=3000, center_theta=48)
    m = StageCMap(topo, StageCConfig(maritime_L_km=0.0))
    c = m.cfg
    legacy = np.where(m.sea, m.SST_C[:, None],
                      m.Tland_C[:, None] - c.lapse_land * m.h
                      - c.continentality_K * (1.0 - np.exp(-m.d_coast / c.L_cont_km)))
    assert np.allclose(m.T_map_C, legacy, atol=1e-9)


def test_diurnal_backbone_callables_override_defaults():
    # the cascade path: supplied diurnal-range callables (from surface.solve_periodic_surface) override the
    # standalone solve defaults, and the per-cell range blends coast<->interior by continentality.
    topo = Topography.procedural_continent(seed=3, nx=60, ne=50, Le_km=8000, width_km=5000, center_theta=46)
    dT_c = lambda t: np.full_like(np.asarray(t, float), 10.0)
    dT_i = lambda t: np.full_like(np.asarray(t, float), 50.0)
    # maritime_L_km=0 isolates the coast<->interior blend (maritime moderation would pull coastal
    # ranges below the 10 K endpoint toward the sea's ~0; its algebra is pinned in test_maritime_coast)
    m = StageCMap(topo, StageCConfig(maritime_L_km=0.0,
                                     dT_diurnal_coast_of_theta=dT_c, dT_diurnal_interior_of_theta=dT_i))
    assert np.allclose(m.dT_coast, 10.0) and np.allclose(m.dT_int, 50.0)      # callables override the solve default
    land = m.land
    assert m.dT_diurnal_C[land].min() >= 10.0 - 1e-6 and m.dT_diurnal_C[land].max() <= 50.0 + 1e-6  # within endpoints
    dc = topo.distance_to_coast_km(); med = np.median(dc[land])
    near = land & (dc <= med); far = land & (dc > med)
    assert m.dT_diurnal_C[far].mean() > m.dT_diurnal_C[near].mean()           # range blends coast(10)->interior(50) inland


def test_diurnal_skew_places_minmax_asymmetrically():
    # skew = fraction of the diurnal range that sits ABOVE the day-mean. skew=0.75 -> the hot noon spike is
    # 0.75*range above the mean and the night only 0.25*range below it (arid land's noon-heavy asymmetry).
    topo = Topography.procedural_continent(seed=3, nx=40, ne=40, Le_km=6000, width_km=4000, center_theta=48)
    flat = lambda t: np.full_like(np.asarray(t, float), 40.0)
    # maritime_L_km=0 AND maritime_range_L_km=0: isolate the skew placement from the coastal range
    # damping (test_maritime_coast). Both are needed now that the range moderation is genuinely
    # decoupled from the day-mean one rather than nested inside it.
    m = StageCMap(topo, StageCConfig(maritime_L_km=0.0, maritime_range_L_km=0.0,
                                     dT_diurnal_coast_of_theta=flat, dT_diurnal_interior_of_theta=flat,
                                     diurnal_skew_coast=0.75, diurnal_skew_interior=0.75))
    land = m.land
    above = (m.T_max_C - m.T_map_C)[land]; below = (m.T_map_C - m.T_min_C)[land]
    assert np.allclose(above, 0.75 * 40.0, atol=1e-6) and np.allclose(below, 0.25 * 40.0, atol=1e-6)


def test_frost_diagnostics_flag_freezing_nights():
    # The frost-limit information the day-mean omits: land that is livable in the day-MEAN (>=0) but whose
    # NIGHT (T_min) dips below 0. A cool mean (+5 C) with a big swing makes every land cell frost-prone.
    topo = Topography.procedural_continent(seed=3, nx=50, ne=40, Le_km=6000, width_km=3000, center_theta=50)
    cool = lambda t: np.full_like(np.asarray(t, float), 5.0)
    swing = lambda t: np.full_like(np.asarray(t, float), 30.0)
    # lapse_land=0 isolates the frost LOGIC from the procedural continent's mountains (whose altitude lapse
    # would otherwise drag the day-mean below 0 -> perennially frozen rather than frost-prone);
    # maritime_L_km=0 likewise isolates it from coastal moderation (which would damp coastal nights
    # above 0 -- the intended physics, pinned in test_maritime_coast, but not the logic under test).
    m = StageCMap(topo, StageCConfig(Tland_C_of_theta=cool, Tland_interior_C_of_theta=cool, lapse_land=0.0,
                                     maritime_L_km=0.0,
                                     dT_diurnal_coast_of_theta=swing, dT_diurnal_interior_of_theta=swing,
                                     diurnal_skew_coast=0.6, diurnal_skew_interior=0.6))
    d = m.diagnostics()
    # mean +5 (livable), night = 5 - 0.4*30 = -7 (< 0) -> all land is frost-prone, none perennially frozen
    assert d["frost_prone_frac"] > 0.95
    assert d["frost_frac"] >= d["frost_prone_frac"]
    assert d["dT_sea_mean"] <= m.cfg.diurnal_range_sea_K + 1e-9 and d["dT_land_mean"] > 25.0


# ---- screen-level (2 m) AIR diurnal range: boundary-layer damping of the surface range (Issue #21 follow-up) ----
# CONTRACTS (intended-by-construction): the air range is a damped, day-mean-anchored version of the surface
# range, so dT_air<=dT_surf, the air night min is lifted above the skin min, T_map_C is untouched, and the
# chosen damping has g_interior<g_coast. A break in any of these is a real regression, not a characterization.
def test_air_range_smaller_than_surface_range():
    # boundary-layer damping g in (0,1] -> the air swing never exceeds the surface (skin) swing, on all land.
    topo = Topography.procedural_continent(seed=3, nx=60, ne=60, Le_km=6000, width_km=3500, center_theta=48)
    m = StageCMap(topo, StageCConfig())
    land = m.land
    assert np.all(m.dT_air_C[land] <= m.dT_diurnal_C[land] + 1e-9)
    # OVER WATER THE ORDERING IS REVERSED, by design: air<=skin is a LAND boundary-layer argument (a deep
    # convective BL mixes the surface flux through a large air mass). The ocean buffers the SKIN, not the
    # air -- a 50 m mixed layer swings ~0.1 K while the 2 m marine air above it swings ~1 K on direct SW
    # absorption and daytime BL deepening. The two are now separate config fields; asserting air<=skin on
    # sea only passed while one field wrongly drove both (PR #389 review round 2, finding 3).
    assert np.all(m.dT_air_C[m.sea] >= m.dT_diurnal_C[m.sea] - 1e-9)
    assert np.allclose(m.dT_diurnal_C[m.sea], m.cfg.diurnal_range_sea_surface_K)
    assert np.allclose(m.dT_air_C[m.sea], m.cfg.diurnal_range_sea_K)


def test_air_min_warmer_than_surface_min():
    # the night lift (the frost asymmetry): a shallow stable layer decouples the air from the cooling skin,
    # so the screen-level air min sits AT OR ABOVE the surface min -- which is why air frost is rarer.
    topo = Topography.procedural_continent(seed=3, nx=60, ne=60, Le_km=6000, width_km=3500, center_theta=48)
    m = StageCMap(topo, StageCConfig())
    land = m.land                                                           # a LAND invariant (see below)
    assert np.all(m.T_air_min_C[land] >= m.T_min_C[land] - 1e-9)
    assert np.all(m.T_air_max_C[land] <= m.T_max_C[land] + 1e-9)            # and the day max is damped down
    # Over sea the marine air swings MORE than the buffered SST skin, so the air extremes sit OUTSIDE the
    # skin ones there -- the opposite ordering, and physically right (see test_air_range_smaller_than_
    # surface_range). The frost argument this invariant supports is about land anyway.


def test_air_cycle_inside_skin_under_wide_g_sweep():
    # the night-lift / day-cap ordering is enforced BY CONSTRUCTION (the air cycle is clamped inside the skin
    # cycle), so it cannot be swept out of existence (review fix): even an undamped g=1.0 on a hot-skewed
    # surface -- where the symmetric air min/max would otherwise punch THROUGH the skin extremes -- still
    # yields T_air_min>=T_min and T_air_max<=T_max. This is the case the thin parameter margin would break.
    topo = Topography.procedural_continent(seed=3, nx=60, ne=60, Le_km=6000, width_km=3500, center_theta=48)
    for g in (0.5, 0.84, 1.0):                                              # spanning past the ~0.84 coast ceiling
        m = StageCMap(topo, StageCConfig(bl_damp_g_coast=g, bl_damp_g_interior=g,
                                         diurnal_skew_coast=0.72, diurnal_skew_interior=0.72))
        land = m.land                                                       # LAND invariant (sea is reversed)
        assert np.all(m.T_air_min_C[land] >= m.T_min_C[land] - 1e-9), f"night lift broke at g={g}"
        assert np.all(m.T_air_max_C[land] <= m.T_max_C[land] + 1e-9), f"day cap broke at g={g}"
        assert np.all(m.dT_air_C[land] <= m.dT_diurnal_C[land] + 1e-9)


def test_air_range_interior_exceeds_coast():
    # the interior SURFACE swing is so much larger than the coast's that interior air > coast air even though
    # the interior is damped MORE (this does NOT pin the damping direction -- see the dedicated test below).
    topo = Topography.procedural_continent(seed=3, nx=80, ne=70, Le_km=8000, width_km=5000, center_theta=46)
    m = StageCMap(topo, StageCConfig())
    dc = topo.distance_to_coast_km(); med = np.median(dc[m.land])
    near = m.land & (dc <= med); far = m.land & (dc > med)
    assert near.any() and far.any()
    assert m.dT_air_C[far].mean() > m.dT_air_C[near].mean()


def test_interior_damps_more_than_coast():
    # the physical direction, guarded directly: the air/skin RANGE RATIO g is smaller in the arid interior
    # (bare-soil skin decoupling) than at the moist, well-coupled coast -- g_interior < g_coast.
    topo = Topography.procedural_continent(seed=3, nx=80, ne=70, Le_km=8000, width_km=5000, center_theta=46)
    m = StageCMap(topo, StageCConfig())
    dc = topo.distance_to_coast_km(); med = np.median(dc[m.land])
    near = m.land & (dc <= med); far = m.land & (dc > med)
    ratio = np.where(m.dT_diurnal_C > 1e-6, m.dT_air_C / np.maximum(m.dT_diurnal_C, 1e-6), 1.0)
    assert ratio[far].mean() < ratio[near].mean()                          # interior ratio smaller = damps more


def test_air_daymean_unchanged_and_midpoint_anchored():
    # the real no-regression invariant: the deliverable day-mean T_map_C is BIT-IDENTICAL to the legacy
    # continentality model (the air cycle only narrows the swing about the same mean). The midpoint check is a
    # CONSTRUCTION check guarded on the symmetric default air_skew=0.5 (it is not itself a regression test).
    # 2026-07-18: pinned to the maritime_L_km=0 escape -- the standalone default now carries maritime
    # moderation (intended; ON algebra pinned in test_maritime_coast). The invariant under test (the air
    # cycle narrows the swing about an UNCHANGED day-mean) is construction-true under either setting.
    topo = Topography.procedural_continent(seed=5, nx=50, ne=50, Le_km=6000, width_km=3000, center_theta=48)
    m = StageCMap(topo, StageCConfig(maritime_L_km=0.0))
    c = m.cfg
    legacy = np.where(m.sea, m.SST_C[:, None],
                      m.Tland_C[:, None] - c.lapse_land * m.h
                      - c.continentality_K * (1.0 - np.exp(-m.d_coast / c.L_cont_km)))
    assert np.allclose(m.T_map_C, legacy, atol=1e-9)                       # day-mean deliverable unchanged
    mid = 0.5 * (m.T_air_min_C + m.T_air_max_C)
    assert np.allclose(mid[m.land], m.T_map_C[m.land], atol=1e-9)          # symmetric default -> centre at Tmean
    assert np.allclose(m.dT_air_C[m.sea], c.diurnal_range_sea_K)           # sea air swing = the small marine value


def test_air_frost_subset_of_surface_frost():
    # diagnostics now report AIR frost (frost_frac/frost_prone_frac) AND surface frost (surf_frost_*). Because
    # the air night min is lifted above the skin min, air frost is a SUBSET of surface frost: air <= surf.
    topo = Topography.procedural_continent(seed=3, nx=50, ne=40, Le_km=6000, width_km=3000, center_theta=50)
    cool = lambda t: np.full_like(np.asarray(t, float), 5.0)
    swing = lambda t: np.full_like(np.asarray(t, float), 30.0)
    m = StageCMap(topo, StageCConfig(Tland_C_of_theta=cool, Tland_interior_C_of_theta=cool, lapse_land=0.0,
                                     dT_diurnal_coast_of_theta=swing, dT_diurnal_interior_of_theta=swing,
                                     diurnal_skew_coast=0.6, diurnal_skew_interior=0.6))
    d = m.diagnostics()
    assert d["frost_frac"] <= d["surf_frost_frac"] + 1e-12
    assert d["frost_prone_frac"] <= d["surf_frost_prone_frac"] + 1e-12
    assert d["surf_frost_frac"] > 0.0                                      # the config does frost (sanity)


def test_phi_smoothing_is_periodic_along_track():
    # Issue #82: everything downstream of Phi (_de/_lap/_adv_flux) is np.roll-PERIODIC in e, so the
    # frictional Gaussian smoothing must be periodic along e too ("wrap"), clamped only at the
    # cross-track walls. A clamped-e smooth leaves the land/sea contrast across the e-seam unsmoothed,
    # and the periodic derivative then differences the raw jump -- the grid-scale del^2(Phi) ascent
    # spike the smoothing exists to prevent, reborn at the seam (a measured 2-4x spurious precip
    # stripe on unpadded topologies). The decisive invariant: periodic operators are along-track
    # TRANSLATION-EQUIVARIANT -- rolling the inputs must roll Phi. Any clamped-e mode breaks this at
    # the seam (the pre-fix mode="nearest" fails this test).
    t = Topography.procedural_continent(seed=1, nx=30, ne=40, Le_km=6000)
    m = StageCMap(t, StageCConfig())
    Phi0 = m._geopotential()
    k = 13                                                  # any nontrivial along-track shift
    m.T_map_C = np.roll(m.T_map_C, k, axis=1)
    m.h = np.roll(m.h, k, axis=1)
    Phi_rolled = m._geopotential()
    assert np.allclose(np.roll(Phi0, k, axis=1), Phi_rolled, atol=1e-9)


# ---- critical-RH (sub-grid) stratiform onset: unpin RH_aloft from 1 (OPEN_ISSUES 2026-06-27) ----
def test_wa_supersat_reduces_to_hard_saturation_at_rhcrit_1():
    # At wa_rh_crit=1.0 the PDF supersaturation tail must equal the hard-saturation excess max(W-Wsat,0)
    # EXACTLY (the byte-identical default that preserves the prior closure).
    from flatworld.stage_c_map import _wa_supersat
    Wsat = np.full(7, 30.0)
    W = np.array([0.0, 15.0, 24.0, 29.999, 30.0, 33.0, 60.0])
    assert np.array_equal(_wa_supersat(W, Wsat, 1.0), np.maximum(W - Wsat, 0.0))
    assert np.array_equal(_wa_supersat(W, Wsat, 1.5), np.maximum(W - Wsat, 0.0))   # clamped >=1 too


def test_wa_supersat_fires_below_saturation_partial_and_monotone():
    # The whole point of the fix: at wa_rh_crit<1 the tail is POSITIVE for grid-mean RH>rh_crit (i.e. BEFORE
    # saturation), it is >= the hard excess everywhere (rains more, never less -> pulls RH down), and after
    # one capacity adjustment to the Wa_sat ceiling the column sits BELOW saturation with an amount that
    # depends on how supersaturated it was (the supply-dependent structure that gives RH its longitude wiggle).
    from flatworld.stage_c_map import _wa_supersat
    Wsat = 30.0; rh = 0.8
    grid = np.linspace(0.0, 1.4 * Wsat, 400)
    tail = _wa_supersat(grid, np.full_like(grid, Wsat), rh)
    onset = rh * Wsat                                                   # = Wsat - Delta, Delta=(1-rh)Wsat
    assert np.all(tail[grid < onset - 1e-9] == 0.0)                     # sub-rh_crit: no rain
    fires = grid[(grid > onset) & (grid < Wsat)]                        # rh_crit < RH < 1 : partial rainout
    assert _wa_supersat(fires, np.full_like(fires, Wsat), rh).min() > 0.0
    assert np.all(tail >= np.maximum(grid - Wsat, 0.0) - 1e-12)         # >= hard excess everywhere
    assert np.all(np.diff(tail) >= -1e-12)                             # monotone non-decreasing in W
    # capacity adjustment W -> W - qc_eq lands at or below saturation, strictly below for Q1<1:
    W_after = grid - tail
    assert np.all(W_after <= Wsat + 1e-9)
    Q1 = (grid - Wsat) / ((1.0 - rh) * Wsat)
    assert np.all(W_after[(Q1 > -1) & (Q1 < 1) & (grid > onset)] < Wsat - 1e-9)


def test_wa_rh_crit_unpins_rh_aloft_longitude_structure():
    # End-to-end: a critical-RH onset (wa_rh_crit=0.8) must UNPIN RH_aloft relative to the hard closure
    # (wa_rh_crit=1.0, which pins RH=1 -> the latitude-locked cloud-boundary symptom) and restore along-track
    # variance on the previously-pinned cells, while staying physically BOUNDED (RH<=1, the hard clip). Same
    # topography/seed/supply -- only the closure differs. (Iteration cap, not wall-clock, for determinism.)
    t = Topography.procedural_continent(seed=11, nx=44, ne=44, Le_km=6000)
    base = dict(dt=900.0, check_every=300, tol=5e-4, patience=3, max_iter=8000)
    hard = StageCMap(t, StageCConfig(wa_rh_crit=1.0, **base)); hard.run(verbose=False)
    soft = StageCMap(t, StageCConfig(wa_rh_crit=0.8, **base)); soft.run(verbose=False)
    rh_hard = hard.W_a / hard.Wsat_aloft[:, None]
    rh_soft = soft.W_a / soft.Wsat_aloft[:, None]
    pinned = rh_hard >= 1.0 - 1e-6
    assert pinned.mean() > 0.4                                          # the hard-closure pinning symptom is present
    assert rh_hard[pinned].std() < 1e-9                                 # ...and longitude-flat (==1) there
    # FIX: the critical-RH onset lowers the saturated fraction and restores along-track structure on those cells.
    assert (rh_soft >= 1.0 - 1e-6).mean() < pinned.mean() - 0.1         # materially fewer cells pinned
    assert rh_soft[pinned].std() > 1e-2                                 # previously-flat cells now vary along e
    assert rh_soft.max() <= 1.0 + 1e-9                                  # BOUNDED: the hard clip prevents super-saturation
    assert 0.7 < soft.diagnostics()["water_closure_cum"] < 1.3         # water budget still closes


def test_wa_rh_crit_converged_field_is_timestep_independent():
    # The critical-RH path evaluates all aloft tendencies from the START-OF-STEP column (dW_a/dt =
    # supply - qc(W_a)/tau_cond, a genuine O(dt) tendency), so the CONVERGED RH_aloft is dt-INDEPENDENT --
    # there is no dt in reality (PR #231 review). Run two timesteps to the SAME fixed sim-time (matched
    # iteration counts, early-stop disabled) so the comparison is not biased by the per-step convergence rate.
    t = Topography.procedural_continent(seed=11, nx=40, ne=40, Le_km=6000)
    SIMT = 2.7e6                                                        # simulated seconds, identical for both dt
    def field(dt):
        nit = int(SIMT / dt)
        cfg = StageCConfig(wa_rh_crit=0.8, tau_cond=1000.0, dt=dt, check_every=nit,
                           tol=0.0, patience=10**9, max_iter=nit)
        m = StageCMap(t, cfg); m.run(verbose=False)
        return m.W_a / m.Wsat_aloft[:, None]
    rh_hi, rh_lo = field(900.0), field(450.0)                          # 2x timestep refinement
    # the converged field barely moves under a 2x dt change: level stable AND pattern preserved.
    assert abs(rh_hi.mean() - rh_lo.mean()) < 5e-3                     # absolute RH level dt-independent
    assert np.corrcoef(rh_hi.ravel(), rh_lo.ravel())[0, 1] > 0.999     # full 2-D field pattern dt-independent
    # contrast: the OLD within-step /dt ceiling (wa_rh_crit kept <1 but with the legacy closure) drifted the
    # level by ~0.05 over the same 2x dt -- this asserts the restructure removed that.


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
