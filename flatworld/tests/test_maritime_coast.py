"""Acceptance tests for MARITIME MODERATION of the Stage-C temperature map (2026-07-18).

The legacy map applied the continentality ramp only BETWEEN two land backbones and left the
land<->sea shoreline as a raw ``where(sea, SST, T_land)`` step -- a +20 K zero-width cliff on the
hot-land lens_diurnal world, which also sharpened the coastal thermal low via Phi(T). The fix blends
the coastal land day-mean toward the row's SST (and the diurnal range toward the sea's) by
m = exp(-d_coast/maritime_L_km), at sea level (before the altitude lapse), with the scale anchored
to the measured ~610 km sea-breeze intrusion. maritime_L_km=0/None is the byte-identical legacy
escape. These tests pin the exact blend algebra, the OFF escape, the default, and the behavioral
point: the shoreline step shrinks by ~the blend factor while the deep interior is untouched.
"""
import numpy as np
import pytest
from flatworld.topography import Topography
from flatworld.stage_c_map import StageCMap, StageCConfig


def _topo():
    return Topography.procedural_continent(seed=3, nx=60, ne=60, Le_km=6000, width_km=3500, center_theta=48)


def _legacy_land_mean(m):
    """The pre-maritime land day-mean reconstructed exactly from the map's own backbones."""
    f = 1.0 - np.exp(-m.d_coast / m.cfg.L_cont_km)
    return ((1.0 - f) * m.Tland_C[:, None] + f * m.Tland_int_C[:, None]
            - m.cfg.lapse_land * m.h)


def test_default_on():
    # 720 km: the AS-1 recalibration on the canonical lens_diurnal world (Jake, 2026-07-26 -- the
    # grid-stable band intrusion; MARITIME_L_CALIBRATION_RESULTS.md). Supersedes the b23/L=8e7-era
    # 610 this pin was written against (a stale-constant failure found by the GEO-22 gate run).
    assert StageCConfig().maritime_L_km == 720.0
    # the diurnal RANGE moderates over a SEPARATE, shorter length than the day-MEAN (coastal-range fix),
    # and the sea carries a small non-zero 2 m air swing (not the ~0.1 K SST number).
    assert StageCConfig().maritime_range_L_km == 150.0
    assert StageCConfig().diurnal_range_sea_K == 1.0


@pytest.mark.parametrize("off", [0.0, None])
def test_off_is_byte_identical_legacy(off):
    """maritime_L_km 0/None reproduces the legacy sharp-shoreline map bit-for-bit (the escape)."""
    m = StageCMap(_topo(), StageCConfig(maritime_L_km=off))
    expect = _legacy_land_mean(m)
    assert np.array_equal(m.T_map_C[m.land], expect[m.land])


def test_on_matches_exact_blend_algebra():
    """ON: every land cell's day-mean is exactly m*SST + (1-m)*legacy_sea_level - lapse."""
    m = StageCMap(_topo(), StageCConfig())
    f = 1.0 - np.exp(-m.d_coast / m.cfg.L_cont_km)
    sea_level = (1.0 - f) * m.Tland_C[:, None] + f * m.Tland_int_C[:, None]
    w = np.exp(-m.d_coast / m.cfg.maritime_L_km)
    expect = w * m.SST_C[:, None] + (1.0 - w) * sea_level - m.cfg.lapse_land * m.h
    assert np.array_equal(m.T_map_C[m.land], expect[m.land])


def test_sea_cells_unchanged():
    """Moderation acts on LAND only: sea T/min/max are identical ON vs OFF."""
    on = StageCMap(_topo(), StageCConfig())
    off = StageCMap(_topo(), StageCConfig(maritime_L_km=0.0))
    for a, b in [(on.T_map_C, off.T_map_C), (on.T_min_C, off.T_min_C), (on.T_max_C, off.T_max_C)]:
        assert np.array_equal(a[on.sea], b[off.sea])


def test_deep_interior_untouched():
    """Beyond ~5 e-folding lengths the moderation is negligible (< 0.1 K on a 30 K contrast)."""
    on = StageCMap(_topo(), StageCConfig())
    off = StageCMap(_topo(), StageCConfig(maritime_L_km=0.0))
    deep = on.land & (on.d_coast > 5.0 * on.cfg.maritime_L_km)
    if deep.any():
        assert np.abs(on.T_map_C[deep] - off.T_map_C[deep]).max() < 0.1


def test_shoreline_step_shrinks_and_range_damps():
    """Behavioral lock on a forced hot-land world: the 1-cell coastal land-sea step shrinks to
    ~the first-cell blend residual, and the coastal diurnal range is damped toward the sea's."""
    from scipy import ndimage as ndi
    topo = _topo()
    hot = dict(Tland_C_of_theta=lambda th: 20.0 + 20.0 + 0.0 * np.asarray(th, float),
               sst_C_of_theta=lambda th: 20.0 + 0.0 * np.asarray(th, float))   # flat +20 K contrast
    on = StageCMap(topo, StageCConfig(**hot))
    # maritime_range_L_km=0 too: the range moderation is no longer nested inside the day-mean block
    # (it is genuinely decoupled), so 'all moderation off' must say so explicitly.
    off = StageCMap(topo, StageCConfig(maritime_L_km=0.0, maritime_range_L_km=0.0, **hot))
    coast_land = on.land & ndi.binary_dilation(on.sea)
    step_on = on.T_map_C[coast_land].mean() - 20.0
    step_off = off.T_map_C[coast_land].mean() - 20.0
    # the OFF step is most of the +20 K contrast (continentality + the fixture's coastal-terrain
    # lapse eat a few K); ON collapses it -- the exact algebra is pinned by the tests above, here
    # we lock the headline behavior only
    assert step_off > 8.0
    w = np.exp(-on.d_coast[coast_land] / on.cfg.maritime_L_km).mean()
    assert w > 0.5                       # first land cells sit well inside the intrusion length
    assert step_on < 0.5 * step_off
    # diurnal range at the coast is damped toward the sea's over the DECOUPLED maritime_range_L_km
    # (shorter than the day-mean length), per-cell exactly. diurnal_range_sea_K is the sea's 2 m AIR
    # swing, so the SURFACE range is moderated toward its surface-equivalent sea_K/g_air -- which the
    # g_air damping then turns back into exactly sea_K of air swing at the shoreline:
    #   rng_on = w*(diurnal_range_sea_K/g_air) + (1-w)*rng_off,  w = exp(-d/maritime_range_L_km)
    w_cells = np.exp(-on.d_coast[coast_land] / on.cfg.maritime_range_L_km)
    f_cells = 1.0 - np.exp(-on.d_coast[coast_land] / on.cfg.L_cont_km)
    g_cells = (1.0 - f_cells) * on.cfg.bl_damp_g_coast + f_cells * on.cfg.bl_damp_g_interior
    rng_on = on.dT_diurnal_C[coast_land]
    rng_off = off.dT_diurnal_C[coast_land]
    assert rng_on.mean() < rng_off.mean()
    expect = w_cells * (on.cfg.diurnal_range_sea_K / g_cells) + (1.0 - w_cells) * rng_off
    assert np.allclose(rng_on, expect, atol=1e-9)


def test_air_range_is_continuous_across_the_shoreline():
    """PR #389 review finding 4: diurnal_range_sea_K is the sea's 2 m AIR swing, but the land side
    moderates a SURFACE range that g_air later damps. Moderating toward sea_K directly landed the fully
    flushed first land cell on g_air*sea_K ~ 0.76 K of air swing against the sea's 1.0 K -- so "the
    swing land is damped toward" was not the swing the sea actually has. Targeting sea_K/g_air in
    surface space removes that step.

    Checked in the limit that isolates it: with the range moderation reaching everywhere (a huge
    maritime_range_L_km), every land cell must carry the sea's air swing. The tolerance covers the
    residual 1 - exp(-d/L) ~ 3e-6 at L = 1e9 km, not the effect under test -- the old law missed by
    1 - g_air ~ 0.24 K, five orders of magnitude larger."""
    topo = _topo()
    cfg = StageCConfig(maritime_range_L_km=1e9)          # m_rng -> 1 everywhere: fully flushed
    m = StageCMap(topo, cfg)
    assert np.allclose(m.dT_air_C[m.land], cfg.diurnal_range_sea_K, atol=1e-4), \
        "fully moderated land air range must equal the sea's air swing, not g_air times it"
    assert np.allclose(m.dT_air_C[m.sea], cfg.diurnal_range_sea_K, atol=1e-12)
    # and the shoreline itself: the first land cells sit within a hair of the sea's swing
    from scipy import ndimage as ndi
    coast = m.land & ndi.binary_dilation(m.sea)
    assert np.abs(m.dT_air_C[coast] - cfg.diurnal_range_sea_K).max() < 1e-4


def test_migration_shape_fit_never_returns_a_negative_range():
    """PR #389 review round 3: the archive migration's per-row fit of the intrinsic range solves for the
    row's two BACKBONE DIURNAL RANGES, which are physically non-negative -- but an unconstrained
    least-squares solve does not know that, and on the seed-90 window it returned A < 0 on 6 land cells
    (-1.42 K, -0.31 K). Nothing downstream clamps A, so a negative there is one geometry away from a
    committed archive with T_high < T_low; it only stayed positive because at d = 40 km the sea term
    carries 77% of the weight. The non-negative solve makes A >= 0 hold BY CONSTRUCTION.

    The fixture is a row whose A RISES with distance inland -- the real world's direction, and the
    mechanism behind the two real rows: the fit band [200, 1500] km sits where the interior basis
    function dominates, so a steep enough rise is best matched by a large interior coefficient and a
    NEGATIVE coastal one, which then extrapolates below zero back at the shoreline. The assertion is
    pinned to be non-vacuous: this same fixture drives the unconstrained solve to about -12.7 K."""
    rep = pytest.importorskip("repropagate_coastal_diurnal")
    d = np.tile(np.linspace(40.0, 3000.0, 80), (3, 1))
    land = np.ones_like(d, bool)
    A_raw = 5.0 + 0.012 * d                                  # rises inland, as the real intrinsic range does

    # control: the unconstrained solve this replaced DOES go negative on this fixture, so the
    # assertions below are testing the constraint rather than passing for free.
    c1, c2 = rep._shape_basis(d)
    band = (d[0] >= 200.0) & (d[0] <= 1500.0)
    M = np.stack([c1[0][band], c2[0][band]], axis=1)
    coef_ols = np.linalg.lstsq(M, A_raw[0][band], rcond=None)[0]
    assert (c1[0] * coef_ols[0] + c2[0] * coef_ols[1]).min() < -1.0, "fixture no longer discriminates"

    A_fit = rep._fit_shape_A(A_raw, d, land, (200.0, 1500.0))
    assert np.all(A_fit >= 0.0), f"fitted intrinsic range went negative (min {A_fit.min():.3f} K)"
    # and the resulting migrated range is still a valid (non-negative) diurnal range everywhere
    dT_new, _ = rep.new_air_range(np.zeros_like(d), d, land, A=A_fit)
    assert np.all(dT_new >= 0.0), "migrated diurnal range went negative -> T_high < T_low"


def test_coastal_highlands_stay_lapse_cooled():
    """The blend acts at sea level: a coastal mountain is still lapse-cooled by its full altitude
    (T = blended sea-level value - lapse*h exactly -- checked implicitly by the algebra test, and
    here behaviorally: the ON-vs-OFF difference is INDEPENDENT of altitude)."""
    m_on = StageCMap(_topo(), StageCConfig())
    m_off = StageCMap(_topo(), StageCConfig(maritime_L_km=0.0))
    d = (m_on.T_map_C - m_off.T_map_C)[m_on.land]
    h = m_on.h[m_on.land]
    hi = h > np.percentile(h, 80)
    # difference is set by d_coast alone; regressing it on altitude must find ~nothing beyond the
    # incidental d_coast/altitude correlation -- pin the direct check: recompute expected diff
    w = np.exp(-m_on.d_coast / m_on.cfg.maritime_L_km)[m_on.land]
    f = 1.0 - np.exp(-m_on.d_coast / m_on.cfg.L_cont_km)
    sea_level = ((1.0 - f) * m_on.Tland_C[:, None] + f * m_on.Tland_int_C[:, None])[m_on.land]
    sst2d = np.broadcast_to(m_on.SST_C[:, None], m_on.land.shape)[m_on.land]
    expect = w * (sst2d - sea_level)
    assert np.allclose(d, expect, atol=1e-9)
    assert hi.any()   # the fixture does have highlands to exercise the lapse path


def test_range_length_decoupled_from_mean_length():
    """The diurnal RANGE follows maritime_range_L_km; the day-MEAN follows maritime_L_km. Changing ONLY
    maritime_range_L_km leaves the day-mean bit-identical but changes the coastal swing -- a shorter
    range length un-moderates the coast sooner, so its diurnal range is LARGER (the coastal-range fix)."""
    topo = _topo()
    short = StageCMap(topo, StageCConfig(maritime_range_L_km=150.0))
    long_ = StageCMap(topo, StageCConfig(maritime_range_L_km=610.0))
    assert np.array_equal(short.T_map_C, long_.T_map_C)          # day-mean independent of the range length
    coast = short.land & (short.d_coast < 2.0 * 150.0)
    assert coast.any()
    assert short.dT_air_C[coast].mean() > long_.dT_air_C[coast].mean()


def test_range_escape_reuses_mean_length_byte_identical():
    """maritime_range_L_km=None reuses maritime_L_km for the range; with diurnal_range_sea_K=0 that is the
    byte-identical legacy 610-coupled range (dT = (1-exp(-d/maritime_L_km))*un-moderated range on land, 0
    on sea)."""
    topo = _topo()
    off = StageCMap(topo, StageCConfig(maritime_L_km=0.0, maritime_range_L_km=0.0,
                                       diurnal_range_sea_K=0.0,
                                       diurnal_range_sea_surface_K=0.0))   # un-moderated base range
    # sea_surface too: the legacy field had ZERO sea swing in BOTH the skin and the air, and those are
    # now separate config fields, so reproducing it byte-for-bit means zeroing both.
    esc = StageCMap(topo, StageCConfig(maritime_range_L_km=None, diurnal_range_sea_K=0.0,
                                       diurnal_range_sea_surface_K=0.0))
    w = np.exp(-esc.d_coast / esc.cfg.maritime_L_km)                                  # legacy SINGLE length
    expect = np.where(esc.sea, 0.0, (1.0 - w) * off.dT_diurnal_C)
    assert np.allclose(esc.dT_diurnal_C, expect, atol=1e-9)
