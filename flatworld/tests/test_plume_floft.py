"""Tests for flatworld.plume_floft (the entraining-plume f_loft estimate, OPEN_ISSUES Brief 2).

These use SYNTHETIC environments (a moist adiabat with a prescribed sub-saturation profile) so they run
in milliseconds without a Stage A spin-up; the Stage-A-driven table is exercised by the __main__ report
and the cascade wiring, not the unit gate."""
import numpy as np
import pytest
from flatworld import plume_floft as pf
from flatworld import thermo as th
from flatworld import constants as C


def _moist_adiabat_env(T_sfc=305.0, rh=0.7, p_top=8e3, n=60):
    """A moist-adiabat temperature profile with a uniform sub-saturation (RH<1) -- a buoyant environment
    for a saturated updraft (the saturated core is moister than the RH<1 surroundings)."""
    p = np.linspace(C.P_SURF, p_top, n)
    T_env = th.temperature_on_moist_adiabat(T_sfc, C.P_SURF, p)
    q_env = rh * th.qsat(T_env, p)
    return p, T_env, q_env


def test_symbolic_numeric_crosscheck():
    # the L*q / virtual-temperature / mixing algebra agrees with sympy to ~machine epsilon
    out = pf.symbolic_crosscheck()
    assert max(out.values()) < 1e-10


def test_eps_zero_reproduces_moist_adiabat():
    # a saturated parcel with NO entrainment must follow thermo.temperature_on_moist_adiabat
    T0 = 300.0
    p = np.linspace(C.P_SURF, 1.0e4, 60)
    T_ma = th.temperature_on_moist_adiabat(T0, C.P_SURF, p)
    q_env = th.qsat(T_ma, p)
    r = pf.integrate_plume(p, T_ma, q_env, T0, eps=0.0, PE=0.5, start_saturated=True, n_sub=800)
    T_at = np.interp(p[::-1], r.p[::-1], r.T_p[::-1])[::-1]
    assert np.max(np.abs(T_at - T_ma)) < 1.0          # < 1 K over the whole ascent


def test_cape_positive_and_decreases_with_entrainment():
    p, T_env, q_env = _moist_adiabat_env()
    capes = [pf.integrate_plume(p, T_env, q_env, 305.0, eps, 0.5).cape
             for eps in (0.0, 0.1e-3, 0.4e-3)]
    assert capes[0] > 0                                # a saturated updraft in sub-saturated air is buoyant
    assert capes[0] > capes[1] > capes[2]              # entrainment of drier air erodes CAPE


def test_detrainment_height_decreases_with_entrainment():
    p, T_env, q_env = _moist_adiabat_env()
    zd = [pf.integrate_plume(p, T_env, q_env, 305.0, eps, 0.5).z_det_km
          for eps in (0.05e-3, 0.2e-3, 0.4e-3)]
    assert zd[0] > zd[1] > zd[2]                        # more entrainment -> lower convective top


def test_floft_in_unit_interval_and_decreases_with_PE():
    p, T_env, q_env = _moist_adiabat_env()
    fl = [pf.integrate_plume(p, T_env, q_env, 305.0, 0.1e-3, PE).f_loft for PE in (0.3, 0.5, 0.7, 0.8)]
    for f in fl:
        assert 0.0 < f < 1.0
    # higher precipitation efficiency -> more local rain -> LESS lofted (f_loft = 1 - PE + corrections)
    assert fl[0] > fl[1] > fl[2] > fl[3]


def test_floft_below_one_minus_PE_floor():
    # f_loft must be AT LEAST (1-PE): everything not rained locally is carried/detrained aloft
    p, T_env, q_env = _moist_adiabat_env()
    for PE in (0.3, 0.6):
        r = pf.integrate_plume(p, T_env, q_env, 305.0, 0.05e-3, PE)
        assert r.f_loft >= (1.0 - PE) - 1e-6


def test_frozen_constants_are_in_swept_ranges():
    # the frozen (eps, PE) must lie inside the reported sweep brackets (not an out-of-family value)
    assert min(pf.EPS_SWEEP) <= pf.EPS_FROZEN <= max(pf.EPS_SWEEP)
    assert min(pf.PE_SWEEP) <= pf.PE_FROZEN <= max(pf.PE_SWEEP)


def test_floft_of_theta_from_table_clips_and_maps():
    # the theta->SST->f_loft callable clips to the band and holds the nearest value beyond the table ends
    sst_C = np.array([30.0, 35.0, 40.0, 45.0])
    f_loft = np.array([0.99, 0.70, 0.72, 0.98])             # includes degenerate (~1) endpoints to clip
    sst_of_theta = lambda th_: np.interp(th_, [35.0, 50.0], [45.0, 25.0])   # warm equatorward
    fn = pf.floft_of_theta_from_table(sst_C, f_loft, sst_of_theta, clip=(0.05, 0.95))
    vals = fn(np.array([35.0, 42.0, 50.0]))
    assert np.all((vals >= 0.05) & (vals <= 0.95))          # clipped
    # beyond-table SST (25 C at theta 50) holds the nearest (30 C) table value, then clips
    assert fn(50.0) == pytest.approx(min(f_loft[0], 0.95))


def test_cape_of_theta_maps_sst_to_cape_unclipped():
    # the theta->SST->CAPE callable holds plume CAPE [J/kg] (NOT clipped to the f_loft 0-1 band) and
    # warms with SST (more convective vigor over the hot inner sea)
    sst_C = np.array([20.0, 30.0, 40.0])
    cape = np.array([50.0, 800.0, 2500.0])                 # J/kg, increasing with SST
    sst_of_theta = lambda th_: np.interp(th_, [35.0, 55.0], [40.0, 20.0])   # warm equatorward
    fn = pf.cape_of_theta_from_table(sst_C, cape, sst_of_theta)
    assert fn(35.0) == pytest.approx(2500.0)               # hot equatorward row -> high CAPE, unclipped
    assert fn(55.0) == pytest.approx(50.0)                 # cool poleward row -> low CAPE
    assert fn(35.0) > fn(45.0) > fn(55.0)                  # monotone with the SST backbone


def test_cape_map_from_theta_broadcasts_per_row_to_map_shape():
    # the per-row CAPE(theta) broadcasts down the rows of the (nx, ne) map -> dC['cape'] shape
    nx, ne = 5, 7
    theta = np.linspace(35.0, 55.0, nx)
    cape_of_theta = lambda th: np.asarray(th, float) * 10.0       # distinct per row
    cmap = pf.cape_map_from_theta(cape_of_theta, theta, (nx, ne))
    assert cmap.shape == (nx, ne)
    assert np.allclose(cmap, (theta * 10.0)[:, None])             # constant along-track, varies cross-track
    assert cmap.flags.writeable                                   # .copy() -> not a read-only broadcast view


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))


class _MockCore:
    """Minimal AD-core stand-in carrying the pressure-grid fields ad_core_environments reads, so the
    extractor + envs-override path is unit-testable without a fused-core spin-up."""
    def __init__(self, ssts_C, nz=14):
        p, T_env, q_env = _moist_adiabat_env()                    # a physical sounding for integrate_plume
        self.p_mid = np.linspace(C.P_SURF, 8e3, nz)               # any order; integrate_plume sorts by p
        self.cfg = type("cfg", (), {"nx": len(ssts_C)})()
        self.theta_deg = np.linspace(25.0, 60.0, len(ssts_C))
        self.T_surf = np.asarray(ssts_C, float) + C.T_FREEZE
        self.T_atm = np.interp(self.p_mid, p, T_env)[None, :] * np.ones((len(ssts_C), 1))
        self.q = np.interp(self.p_mid, p, q_env)[None, :] * np.ones((len(ssts_C), 1))


def test_ad_core_environments_selects_wet_band_and_feeds_floft_table():
    """The f_loft leak fix: ad_core_environments extracts the wet-ocean (28-55 C) columns from the fused
    core, and floft_sst_table(envs=...) samples the plume from THOSE instead of Stage A's columns. Pins:
    (1) only wet-band columns are kept; (2) the table builds with no Stage A object (stage_a=None);
    (3) f_loft stays in [0,1] sorted by SST."""
    core = _MockCore([10.0, 40.0, 50.0, 90.0, 130.0])             # 10/90/130 outside [28,55]; 40/50 inside
    envs = pf.ad_core_environments(core)
    assert all(pf.SAMPLE_SST_LO <= e["sst_C"] <= pf.SAMPLE_SST_HI for e in envs)
    assert len(envs) == 2                                         # only the 40 C, 50 C columns
    s, f, zd, cape = pf.floft_sst_table(None, pf.EPS_FROZEN, pf.PE_FROZEN, envs=envs)
    assert s.size == 2 and np.all(np.isfinite(f))
    assert np.all((f >= 0.0) & (f <= 1.0)) and np.all(np.diff(s) >= 0.0)


def test_ad_core_environments_reads_a_real_ad_core_attribute_contract():
    """Lock the attribute contract against a REAL StageADCore -- the hand-rolled _MockCore above can drift
    from it. ad_core_environments must read the core's ACTUAL fields, and diagnostics() must expose D_surf
    (the land-T backbone's downwelling-LW source). This catches a future rename of self.q / p_mid / T_surf /
    T_atm / theta_deg that the mock would silently miss. A fresh (un-run) minimal core suffices -- these
    fields are all set in __init__; cheap to build."""
    from flatworld.stage_ad_core import StageADCore, StageADConfig
    core = StageADCore(StageADConfig(nx=24, nz=16))
    d = core.diagnostics()
    assert "D_surf" in d and np.asarray(d["D_surf"]).shape == (core.cfg.nx,)   # land-T backbone source
    # force two columns into the wet-ocean band so the extraction runs deterministically (independent of
    # the fresh core's incidental SST range / the ambient L); T_atm/q stay the real core's pressure-grid soundings
    core.T_surf = core.T_surf.copy(); core.T_surf[:2] = np.array([35.0, 50.0]) + C.T_FREEZE
    envs = pf.ad_core_environments(core)
    assert len(envs) >= 2
    assert set(envs[0].keys()) == {"i", "sst_K", "theta_deg", "p", "T", "q", "sst_C"}
    assert all(pf.SAMPLE_SST_LO <= e["sst_C"] <= pf.SAMPLE_SST_HI for e in envs)
    assert envs[0]["p"].shape == (core.cfg.nz,) and envs[0]["T"].shape == (core.cfg.nz,)   # real pressure grid
