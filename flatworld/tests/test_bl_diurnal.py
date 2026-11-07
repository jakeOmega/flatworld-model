"""
Tests for flatworld.bl_diurnal -- the diurnal sea/land-breeze wind solve (issue #113).

The deliverable physics is the diurnally-resolved boundary-layer momentum field: afternoon wind
maxima, calm mornings, and coastal sea-breeze/land-breeze reversals, built by time-stepping the #111
nonlinear BL momentum solver through the lantern (painted) day instead of solving for steady state.

The load-bearing invariants the tests pin:
  * the reconstructed diurnal temperature cycle has day-mean == T_map_C EXACTLY (mean climate untouched);
  * the time-varying forcing's day-mean == the #111 climatological Phi EXACTLY (so the day-mean wind can
    match #111's steady solve -- acceptance criterion 3 -- structurally, not by luck);
  * sea cells hold flat (the breeze is a land-driven anomaly);
  * the breeze reverses sign across the day at a coast (onshore by day, offshore/weaker by night) -- the
    firm sign/shape deliverable (magnitudes are smoothed-scale soft).
"""
import numpy as np
import pytest

from flatworld import bl_diurnal as bld
from flatworld import bl_momentum as blm
from flatworld import bl_forcing as blf
from flatworld import geometry as g


def _synthetic_dC(nx=10, ne=16, coast_at=None, ubg=-5.0):
    """A small (x,e) Stage-C-like diagnostics dict with a straight coast: left half land, right half sea.

    Land carries a noon-heavy diurnal swing (T_max-T_mean=+6, T_mean-T_min=8); sea is ~flat. The
    along-track seam at e=coast_at is the coastline the sea breeze should respond to. ``ubg`` is the
    background cross-track wind: the default weak -5 keeps the breeze ANOMALY visible (for the reversal
    test); pass a cascade-like strong -30 for the consistency gate (where the breeze is a small fraction
    of the mean, as on the real wind-swept band)."""
    coast_at = ne // 2 if coast_at is None else coast_at
    land = np.zeros((nx, ne), bool)
    land[:, :coast_at] = True
    T_map = np.full((nx, ne), 20.0)
    T_min = np.where(land, T_map - 8.0, T_map - 0.05)
    T_max = np.where(land, T_map + 6.0, T_map + 0.05)
    return dict(
        T_map_C=T_map, T_min_C=T_min, T_max_C=T_max, land=land,
        W_s=np.full((nx, ne), 5.0), height=np.zeros((nx, ne)),
        ubg=np.full(nx, float(ubg)),
    )


def _forcing(dC, dcfg=None):
    dcfg = dcfg or bld.BLDiurnalConfig(n_phase=8)
    dx = de = 1.0e5  # 100 km cells
    S_of_theta = lambda th: 400.0  # noqa: E731  (only the normalized day-SHAPE is used)
    return bld.build_diurnal_forcing(dC, dx, de, dcfg, S_of_theta, clock=g.CLOCK)


# ---- 1. day-mean of the reconstructed temperature cycle is EXACTLY T_map_C (mean climate untouched) ----
def test_reconstructed_cycle_day_mean_equals_T_map():
    dC = _synthetic_dC()
    f = _forcing(dC)
    day_mean = f["T_cycle_C"].mean(axis=0)
    assert np.allclose(day_mean, dC["T_map_C"], atol=1e-9)


# ---- 2. sea cells hold flat through the day (breeze is a purely land-driven anomaly) ----
def test_sea_cells_flat_over_the_day():
    dC = _synthetic_dC()
    f = _forcing(dC)
    sea = ~dC["land"]
    swing = f["T_cycle_C"].max(axis=0) - f["T_cycle_C"].min(axis=0)
    assert np.all(swing[sea] < 1e-9)


# ---- 3. the land diurnal RANGE matches T_max_C - T_min_C per cell (amplitude faithful) ----
def test_land_diurnal_range_matches_T_max_minus_T_min():
    dC = _synthetic_dC()
    f = _forcing(dC)
    land = dC["land"]
    rng = f["T_cycle_C"].max(axis=0) - f["T_cycle_C"].min(axis=0)
    target = dC["T_max_C"] - dC["T_min_C"]
    assert np.allclose(rng[land], target[land], rtol=1e-6, atol=1e-6)


# ---- 4. the time-varying forcing's day-mean == the #111 climatological Phi EXACTLY ----
def test_forcing_day_mean_equals_climatological_phi():
    dC = _synthetic_dC()
    f = _forcing(dC)
    phi_day_mean = f["phi"].mean(axis=0)
    assert np.allclose(phi_day_mean, f["phi_clim"], atol=1e-9)


# ---- 5. phi_of(t) is circular in t mod T_day (phi_of(0) == phi_of(T_day) == first slice) ----
def test_phi_of_is_circular():
    dC = _synthetic_dC()
    f = _forcing(dC)
    T_day = g.CLOCK.T_day
    assert np.allclose(f["phi_of"](0.0), f["phi"][0], atol=1e-12)
    assert np.allclose(f["phi_of"](0.0), f["phi_of"](T_day), atol=1e-12)
    # midway between samples is the linear interpolant of the two bracketing slices
    dt_phase = T_day / f["phase_t"].size
    mid = f["phi_of"](0.5 * dt_phase)
    assert np.allclose(mid, 0.5 * (f["phi"][0] + f["phi"][1]), atol=1e-9)


# ---- 6. the shared diurnal shape peaks in the AFTERNOON (heat-capacity lag), not at noon ----
def test_diurnal_shape_peaks_in_the_afternoon():
    dC = _synthetic_dC()
    f = _forcing(dC)
    # phase_t is hours-from-noon (t=0 noon); the warmest land phase should lag noon (afternoon peak)
    land = dC["land"]
    land_speed_proxy = f["T_cycle_C"][:, land].mean(axis=1)
    k_peak = int(np.argmax(land_speed_proxy))
    t_peak_h = f["phase_t"][k_peak] / 3600.0
    # circular hours from noon: a genuine afternoon peak sits at +1..+8 h, not at 0 (noon)
    assert 0.5 < t_peak_h < 8.0


# ============================ the periodic-integration driver ============================

def _run_driver(dC, dcfg=None, **kw):
    dcfg = dcfg or bld.BLDiurnalConfig(n_phase=8, max_days=40, max_seconds=120, period_tol=3e-3)
    dx = de = 1.0e5
    f = _forcing(dC, dcfg)
    sol = bld.solve_diurnal_wind(f["phi_of"], dC["ubg"], dx, de, dcfg, f["phase_t"],
                                 T_map_C=dC["T_map_C"], height=dC["height"], **kw)
    return f, sol


# ---- 7. the driver records EXACTLY n_phase wind slices, sampled at the phase times ----
def test_driver_records_exactly_n_phase_samples_at_phase_t():
    dC = _synthetic_dC()
    dcfg = bld.BLDiurnalConfig(n_phase=8, max_days=12, max_seconds=120)
    f, sol = _run_driver(dC, dcfg)
    assert sol["speed_phase"].shape == (8,) + dC["T_map_C"].shape
    assert sol["u_phase"].shape == sol["v_phase"].shape == sol["speed_phase"].shape
    assert np.allclose(sol["phase_t"], f["phase_t"], atol=1e-6)


# ---- 8. the driver reaches a PERIODIC cycle within the day guard ----
def test_driver_reaches_periodicity():
    dC = _synthetic_dC()
    f, sol = _run_driver(dC)
    assert sol["converged"] is True
    assert sol["days"] <= sol["days_guard"]


# ---- 9. day-mean wind matches #111's steady solve within tier (acceptance criterion 3) ----
def test_day_mean_wind_matches_steady_within_tier():
    # the realistic regime: a strong background wind (cascade-like ~-30, the band blows ~-34) so the breeze
    # is a small fraction of the mean. The day-mean VELOCITY (not scalar speed -- speed rectifies upward by
    # the breeze oscillation, a genuine finding) reproduces the steady solve to within tier.
    dC = _synthetic_dC(ubg=-30.0)
    f, sol = _run_driver(dC)
    assert sol["rectification_frac"] < 0.08   # day-mean velocity ~ steady (acceptance 3, realistic band)
    # pattern preserved: the mean flow is not restructured by the diurnal cycle
    svec = np.concatenate([sol["steady_u"].ravel(), sol["steady_v"].ravel()])
    dvec = np.concatenate([sol["day_mean_u"].ravel(), sol["day_mean_v"].ravel()])
    assert np.corrcoef(dvec, svec)[0, 1] > 0.99


# ---- 9b. the breeze rectifies the day-mean SPEED upward in the weak-mean limit (a reportable finding) ----
def test_breeze_rectifies_mean_speed_in_weak_background():
    dC = _synthetic_dC(ubg=-5.0)
    f, sol = _run_driver(dC)
    # mean speed exceeds the steady speed (afternoon peaks + calm nights average above the steady value)
    assert sol["day_mean_speed"].mean() > sol["steady_speed"].mean()


# ---- 10. the driver is reproducible (no hidden randomness/time dependence) ----
def test_driver_is_reproducible():
    dC = _synthetic_dC()
    dcfg = bld.BLDiurnalConfig(n_phase=8, max_days=10, max_seconds=120)
    _, s1 = _run_driver(dC, dcfg)
    _, s2 = _run_driver(dC, dcfg)
    assert np.array_equal(s1["speed_phase"], s2["speed_phase"])


# ============================ diagnostics: breeze + per-Koppen ratios ============================

# ---- 11. the onshore normal points sea -> land (cross-shore sign convention) ----
def test_onshore_normal_points_seaward_to_landward():
    dC = _synthetic_dC()  # land at e < ne/2, sea at e >= ne/2
    nx_on, ne_on = bld.onshore_normal(dC["land"])
    # the coast varies along e, so the normal is in the e-direction; onshore (toward land at low e)
    # means the e-component is negative on the coastal cells
    land = dC["land"]
    coast_band = land & (~np.roll(land, -1, axis=1))  # land cells with sea immediately to +e
    assert np.all(ne_on[coast_band] < 0)
    assert np.allclose(nx_on[coast_band], 0.0, atol=1e-6)


# ---- 12. the sea breeze REVERSES across the day at the coast (firm sign/shape deliverable) ----
def test_sea_breeze_reverses_sign_at_coast():
    dC = _synthetic_dC()
    f, sol = _run_driver(dC)
    diag = bld.diurnal_diagnostics(sol, dC, dx=1.0e5, de=1.0e5)
    aft = diag["afternoon_cross_shore"]  # >0 = onshore (sea breeze)
    nig = diag["night_cross_shore"]
    coast = diag["coast_band"]
    # by day the coastal flow is onshore (sea breeze); it reverses toward offshore by night
    assert aft[coast].mean() > 0.0
    assert aft[coast].mean() > nig[coast].mean()


# ---- 13. breeze amplitude is reported per Koppen class, and the dipole ratio is well-posed ----
def test_breeze_amplitude_by_koppen_and_dipole_keys():
    # strong (cascade-like) background: the breeze is a small fraction of the mean, so the day/night SPEED
    # ratio cancels to ~1 in the spatial mean (the real signal is the signed orientation dipole, not this).
    dC = _synthetic_dC(ubg=-30.0)
    from flatworld import classify
    koppen = np.where(dC["land"], classify.IDX["BWh"], 0)
    f, sol = _run_driver(dC)
    diag = bld.diurnal_diagnostics(sol, dC, koppen=koppen, dx=1.0e5, de=1.0e5)
    # the meaningful per-class number is the breeze AMPLITUDE (not a day/night ratio, which averages the
    # orientation dipole to ~1); it must be present and positive (a real breeze)
    assert "BWh" in diag["breeze_amp_by_koppen"]
    assert diag["breeze_amp_by_koppen"]["BWh"] > 0.0
    # the orientation-dipole spatial mean is well-posed and ~1 (the +/-cos contributions cancel)
    assert np.isfinite(diag["daynight_ratio_spatial_mean"])
    assert 0.9 < diag["daynight_ratio_spatial_mean"] < 1.1
    # the breeze peak times are within a day
    assert 0.0 <= diag["sea_breeze_peak_hour"] < 24.0


# ---- 13b. the sea-breeze intrusion km scales by the PHYSICAL spacing of the INLAND axis (dx != de) ----
def test_penetration_scales_by_physical_inland_axis_when_dx_ne_de():
    # A coast whose inland normal runs along x (land = low-x rows, sea = high-x rows): the intrusion is
    # measured along x, so it must scale with dx -- NOT de, and NOT the isotropic cell count. With a sharp
    # 1/e drop after 3 inland cells, the penetration is 3 cells worth of dx. The pre-fix code multiplied an
    # isotropic cell count by de, which on this anisotropic grid would report ~4x too far.
    nx, ne = 20, 8
    land = np.zeros((nx, ne), bool)
    land[:12, :] = True                                        # rows 0..11 land, 12..19 sea; normal || x
    dist_cells = np.where(land, 12 - np.arange(nx)[:, None], 0)   # inland cell-distance to the coast
    flow = np.zeros((nx, ne))
    flow[dist_cells == 1] = 1.0; flow[dist_cells == 2] = 0.9; flow[dist_cells == 3] = 0.8
    flow[dist_cells >= 4] = 0.3                                # 0.3 < 1/e of the coastal 1.0 (=0.368) -> drop
    dx_km, de_km = 50.0, 200.0                                 # deliberately anisotropic (de = 4 x dx)
    pen_km, saturated = bld._penetration_km(flow, land, dx_km, de_km)
    assert pen_km == pytest.approx(3 * dx_km, rel=0.05)        # 150 km = 3 inland cells x dx (axis 0)
    assert pen_km < de_km                                      # the pre-fix cell-count x de bug gave ~600 km
    assert saturated is False                                  # decayed before the inland edge
    # same geometry, but make axis-0 (the inland axis) coarser (50 -> 200 km): the reported km tracks the
    # INLAND-axis spacing (150 -> 600), confirming it is the physical inland distance, not isotropic cells.
    pen_km_coarse, _ = bld._penetration_km(flow, land, 200.0, 50.0)
    assert pen_km_coarse == pytest.approx(3 * 200.0, rel=0.05)


# ---- 14. the cascade fails FAST on bl_diurnal_wind=True without lantern_day (premise guard) ----
def test_cascade_requires_lantern_day_for_diurnal_wind():
    from flatworld import cascade
    with pytest.raises(ValueError, match="lantern_day"):
        cascade.run_cascade("test", core="ad", lantern_day=False, bl_diurnal_wind=True, verbose=False)


# ============================ cascade integration (slow) ============================
@pytest.mark.slow
def test_cascade_diurnal_off_byte_identical_and_on_attaches_diagnostics():
    from flatworld import cascade, config
    base = cascade.run_cascade("test", core="ad", lantern_day=True, verbose=False)
    assert "bl_diurnal" not in base["dC"]                          # default OFF: no diurnal fields
    dcfg = config.bl_diurnal("test", max_days=30, max_seconds=180)
    on = cascade.run_cascade("test", core="ad", lantern_day=True, verbose=False,
                             bl_diurnal_wind=True, bl_diurnal_cfg=dcfg)
    dC = on["dC"]
    # the deliverable maps are UNTOUCHED -- the diurnal solve is additional diagnostics only
    assert np.array_equal(base["dC"]["u"], dC["u"])
    assert np.array_equal(base["dC"]["v"], dC["v"])
    assert np.array_equal(base["dC"]["P_mm_day"], dC["P_mm_day"])
    assert np.array_equal(base["dC"]["T_map_C"], dC["T_map_C"])
    # diagnostics attached
    assert "bl_diurnal" in dC and "diurnal_breeze_amp_by_koppen" in dC
    assert dC["diurnal_speed_phase"].shape[0] == dcfg.n_phase
    assert "breeze_reversal_frac" in dC["bl_diurnal"]
    # consistency gate (acceptance 3): day-mean velocity ~ #111 steady on the strong-background band
    assert dC["bl_diurnal"]["rectification_frac"] < 0.10
