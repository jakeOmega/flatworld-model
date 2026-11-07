"""Phase-C diagnostics wishlist: additive, read-only enrichment of the AD core's ``diagnostics()`` and
``stage_c_inputs()`` (resolved aloft layer; cloud water path / column optical depth; clear-sky surface
SW + diffuse fraction). These are EXPOSURE only -- they must not perturb the stepping loop or change any
existing diagnostic field. Fast: low resolution, a handful of steps.

Motivation (Phase-C team): Stage C currently fakes the aloft layer (T_aloft = SST - 6.5K/km*4km) and
recovers Wsat_aloft from a fixed lapse. Exposing the RESOLVED aloft T/q and the saturation column
``Wsat_aloft(theta)`` directly lets Stage C key off the real profile. We only EXPOSE here; the Stage-C
consumption is out of scope.
"""
import numpy as np
import pytest

from flatworld.stage_ad_core import (StageADCore, StageADConfig, ALOFT_SIGMA_THRESH,
                                     RH_ALOFT_WSAT_MIN_MM)
from flatworld import thermo as th
from flatworld import radiation as rad


def _core(clouds=True, prognostic=False, nx=20, nz=18, steps=3, **kw):
    """A small clouds-on core, stepped a few times so q/q_c/clouds are non-trivial."""
    cfg = StageADConfig(nx=nx, nz=nz, theta_max_deg=60.0, clouds=clouds,
                        prognostic_condensate=prognostic, **kw)
    m = StageADCore(cfg)
    m.run_dynamics = False          # thermo-only is enough to populate q / clouds; keeps it fast & stable
    for _ in range(steps):
        m.step()
    return m


# ----------------------------------------------------------------- resolved aloft (item 1)
def test_aloft_fields_present_and_shaped():
    m = _core()
    d = m.diagnostics()
    for k in ("T_aloft_C", "q_aloft", "Wsat_aloft_mm", "RH_aloft"):
        assert k in d, f"missing aloft field {k!r}"
        assert np.asarray(d[k]).shape == (m.cfg.nx,), f"{k} wrong shape"
    # T/q/Wsat are the CONSUMED fields: finite everywhere, always. RH_aloft is a REPORTED ratio and is
    # deliberately NaN where its denominator is degenerate (PHY-15) -- so it is finite exactly on the
    # columns with a real saturation column, which in this warm test world (theta_max 60) is all of them.
    for k in ("T_aloft_C", "q_aloft", "Wsat_aloft_mm"):
        assert np.all(np.isfinite(d[k])), f"{k} not finite"
    honest = np.asarray(d["Wsat_aloft_mm"]) >= RH_ALOFT_WSAT_MIN_MM
    assert honest.all(), "test world unexpectedly has a degenerate saturation column"
    assert np.all(np.isfinite(np.asarray(d["RH_aloft"])[honest])), "RH_aloft not finite where it is defined"


def test_Wsat_aloft_matches_hand_integral():
    """Wsat_aloft = sum_k qsat(T_atm,p_mid)*dm over sigma<0.5 -- the saturation column [mm]."""
    m = _core()
    d = m.diagnostics()
    T = np.asarray(d["T_atm"])                      # (nx, nz) TOP->SURFACE, K
    p_mid = np.asarray(m.p_mid)                      # (nz,)
    dm = np.asarray(m.dm)                            # (nz,)
    sigma = np.asarray(m.sigma)                      # (nz,)
    mask = sigma < ALOFT_SIGMA_THRESH
    qsat = th.qsat(T, p_mid[None, :])
    want = np.sum(qsat[:, mask] * dm[None, mask], axis=1)
    assert np.allclose(d["Wsat_aloft_mm"], want, rtol=1e-6, atol=1e-9)


def test_RH_aloft_consistent_and_bounded():
    """RH_aloft == aloft_moisture_mm / Wsat_aloft, and is a physical relative humidity (>=0, not wildly
    super-saturated at the column scale)."""
    m = _core()
    d = m.diagnostics()
    rh = np.asarray(d["aloft_moisture_mm"]) / np.maximum(np.asarray(d["Wsat_aloft_mm"]), 1e-12)
    assert np.allclose(d["RH_aloft"], rh, rtol=1e-6, atol=1e-9)
    assert np.all(d["RH_aloft"] >= 0.0)
    assert np.all(d["RH_aloft"] <= 1.5)             # column-mean RH; allow a little overshoot, not nonsense


def test_RH_aloft_masked_where_saturation_column_is_degenerate():
    """PHY-15. RH_aloft is a REPORTED relative humidity, so it must not fabricate a number where its
    denominator has collapsed. Past the cold trap the canonical world's aloft column reaches 50-200 K
    and Wsat_aloft falls to ~1e-49 mm while advection leaves a round-off vapour residue behind; the
    unmasked ratio read up to 2.0e4 (~2e6 %) there, and exactly 0.0 where the residue underflowed too.

    Driven here directly on ``_aloft_thermo`` (a warm test core cannot reach 50 K on its own): freeze the
    aloft layers of one column, leave a vapour residue in it, and require NaN -- while every other column
    keeps a BITWISE-identical value, and the CONSUMED Wsat_aloft is untouched by the masking.
    """
    m = _core()
    q = np.clip(np.asarray(m.q), 0.0, th.Q_MAX)
    base = m._aloft_thermo(q)

    mask = np.asarray(m.sigma) < ALOFT_SIGMA_THRESH
    cold = 3                                            # the doomed column
    m.T_atm[cold, mask] = 60.0                          # K -- the far-field cold trap, esat ~ 1e-38 Pa
    q = q.copy()
    q[cold, mask] = 1e-12                               # a vapour residue, far above that saturation
    got = m._aloft_thermo(q)

    W = np.asarray(got["Wsat_aloft"]); RH = np.asarray(got["RH_aloft"])
    assert W[cold] < RH_ALOFT_WSAT_MIN_MM, "the forced column did not actually go degenerate"
    assert np.isnan(RH[cold]), "a degenerate saturation column must report NaN, not a fabricated ratio"

    others = np.ones(m.cfg.nx, bool); others[cold] = False
    assert np.all(np.isfinite(RH[others])), "the mask must not touch honest columns"
    assert np.array_equal(RH[others], np.asarray(base["RH_aloft"])[others]), "honest columns moved"
    # the CONSUMED field is unaffected by the reporting mask (Stage C reads Wsat_aloft, never RH_aloft)
    assert np.all(np.isfinite(W)), "Wsat_aloft must stay finite -- the mask is on the ratio only"
    assert np.array_equal(W[others], np.asarray(base["Wsat_aloft"])[others])


def test_q_aloft_is_mass_weighted_mean():
    m = _core()
    d = m.diagnostics()
    sigma = np.asarray(m.sigma); dm = np.asarray(m.dm)
    mask = sigma < ALOFT_SIGMA_THRESH
    want = np.asarray(d["aloft_moisture_mm"]) / np.sum(dm[mask])     # mm / (kg/m^2 of aloft air) = kg/kg
    assert np.allclose(d["q_aloft"], want, rtol=1e-6, atol=1e-12)
    assert np.all(d["q_aloft"] >= 0.0)


# ----------------------------------------------------------------- clouds: water path + optical depth (item 3)
def test_cloud_water_path_and_optical_depth_present_nonneg():
    m = _core()
    d = m.diagnostics()
    for k in ("cloud_water_path_mm", "cloud_tau_lw_col", "cloud_tau_sw_col"):
        assert k in d, f"missing cloud column field {k!r}"
        assert np.asarray(d[k]).shape == (m.cfg.nx,)
        assert np.all(np.asarray(d[k]) >= 0.0), f"{k} must be non-negative"


def test_cloud_columns_zero_when_clouds_off():
    m = _core(clouds=False)
    d = m.diagnostics()
    for k in ("cloud_water_path_mm", "cloud_tau_lw_col", "cloud_tau_sw_col",
              "cloud_sw_albedo", "cloud_cover"):
        assert np.allclose(d[k], 0.0), f"{k} must be 0 with clouds off"


def test_return_lwp_is_additive_byte_identical():
    """cloud_optical_properties(return_lwp=True) must not change the existing returns."""
    rng = np.random.default_rng(0)
    nz = 12
    f = np.clip(rng.random((4, nz)), 0, 1)
    T = np.linspace(230.0, 300.0, nz)[None, :] * np.ones((4, 1))
    p_mid = np.linspace(2e4, 9e4, nz)[None, :] * np.ones((4, 1))
    dm = np.full((4, nz), 5e3)
    base = rad.cloud_optical_properties(f, T, p_mid, dm, return_tau_sw=True)
    ext = rad.cloud_optical_properties(f, T, p_mid, dm, return_tau_sw=True, return_lwp=True)
    assert len(ext) == len(base) + 1
    for a, b in zip(base, ext):
        assert np.allclose(a, b)
    lwp = ext[-1]
    assert lwp.shape == f.shape and np.all(lwp >= 0.0)


# ----------------------------------------------------------------- clear-sky surface SW (item 4)
def test_clear_sky_surface_sw_ge_cloudy():
    """Surface SW under clear sky must be >= under cloud (clouds reflect/absorb SW)."""
    m = _core()
    d = m.diagnostics()
    assert np.all(d["surf_down_sw_clear"] >= d["surf_down_sw"] - 1e-6)
    assert np.all(d["surf_down_sw"] >= 0.0)


def test_clearsky_transmission_and_diffuse_fraction_bounded():
    m = _core()
    d = m.diagnostics()
    assert np.all((d["clearsky_transmission"] >= 0.0) & (d["clearsky_transmission"] <= 1.0 + 1e-9))
    assert np.all((d["diffuse_frac"] >= 0.0) & (d["diffuse_frac"] <= 1.0 + 1e-9))


def test_radiate_clouds_override_is_byte_identical_default():
    """_radiate(clouds=None) must equal _radiate() (the cfg.clouds path); clouds=False gives the clear call."""
    m = _core()
    q = np.clip(m.q, 0.0, 0.5)
    a = m._radiate(q)
    b = m._radiate(q, clouds=None)
    assert np.allclose(a["OLR"], b["OLR"]) and np.allclose(a["surf_down_sw"], b["surf_down_sw"])
    clear = m._radiate(q, clouds=False)
    assert "cloud_sw_albedo" not in clear or np.allclose(clear.get("cloud_sw_albedo", 0.0), 0.0)


# ----------------------------------------------------------------- the Stage-C handoff contract (item 1 exposure)
def test_stage_c_inputs_exposes_aloft_callables():
    m = _core()
    sc = m.stage_c_inputs()
    th_mid = float(np.median(m.diagnostics()["theta_deg"]))
    for k in ("T_aloft_of_theta", "q_aloft_of_theta", "Wsat_aloft_of_theta", "RH_aloft_of_theta"):
        assert k in sc, f"stage_c_inputs missing {k!r}"
        v = float(sc[k](th_mid))
        assert np.isfinite(v)
    # existing contract fields still present (regression)
    for k in ("sst_of_theta", "wetness_of_theta", "aloft_supply_of_theta", "ascent_gate",
              "subsidence_gate", "diagnostics"):
        assert k in sc


# ----------------------------------------------------------------- regression: existing keys preserved
def test_existing_diagnostics_keys_preserved():
    m = _core()
    d = m.diagnostics()
    for k in ("theta_deg", "T_surf_C", "T_atm", "q", "q_c", "cond_path", "aloft_moisture_mm",
              "cloud_sw_albedo", "cloud_cover", "cloud_fraction", "D_surf", "OLR", "toa_net",
              "precip_mm_day", "u_surf", "S", "sigma"):
        assert k in d, f"regression: lost existing diagnostics key {k!r}"
