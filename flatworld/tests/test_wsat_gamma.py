"""Acceptance tests for the aloft-capacity land modulation ``wsat_aloft_gamma`` (2026-07-18).

The aloft saturation capacity was a pure zonal profile, so every capacity-set transition (the
cold-edge stratiform shutoff) drew a razor-straight line in e (Jake's observation). Physically,
standing land-forced aloft warmth should meander it around the continents. The term:

    Wsat_map = Wsat_row * exp(CC * gamma * (T_map - SST_row)),   CC = 0.07 /K

with the row's SST as the reference (the all-sea AD backbone's world) so sea cells are EXACTLY
unperturbed, warm land columns hold more aloft, and lapse-cooled highlands hold less. gamma is a
best-guess 0.25 (Earth lower-free-troposphere ~0.3 shaded down for f=0 WTG homogenization), swept
0.1-0.5, magnitude charged to the aloft-thermal-structure named wall. 0/None -> byte-identical
per-row legacy capacity.
"""
import numpy as np
import pytest
from flatworld.topography import Topography
from flatworld.stage_c_map import StageCMap, StageCConfig, WSAT_ALOFT_CC_PER_K


def _topo():
    return Topography.procedural_continent(seed=3, nx=60, ne=60, Le_km=6000, width_km=3500, center_theta=48)


def test_default_gamma():
    assert StageCConfig().wsat_aloft_gamma == 0.25


@pytest.mark.parametrize("off", [0.0, None])
def test_gamma_off_is_per_row_legacy(off):
    m = StageCMap(_topo(), StageCConfig(wsat_aloft_gamma=off))
    assert m.Wsat_aloft_map.shape == (m.nx, 1)
    assert np.array_equal(m.Wsat_aloft_map[:, 0], m.Wsat_aloft)


def test_gamma_on_matches_exact_algebra():
    m = StageCMap(_topo(), StageCConfig())
    dT = m.T_map_C - m.SST_C[:, None]
    expect = m.Wsat_aloft[:, None] * np.exp(WSAT_ALOFT_CC_PER_K * m.cfg.wsat_aloft_gamma * dT)
    assert np.array_equal(m.Wsat_aloft_map, expect)


def test_sea_cells_exactly_unperturbed():
    """Over sea T_map == SST_row exactly (the maritime map holds sea at the buffered SST), so the
    factor is exp(0) == 1 and the sea capacity is BIT-identical to the per-row legacy."""
    m = StageCMap(_topo(), StageCConfig())
    W2 = np.broadcast_to(m.Wsat_aloft_map, m.land.shape)
    legacy = np.broadcast_to(m.Wsat_aloft[:, None], m.land.shape)
    assert np.array_equal(W2[m.sea], legacy[m.sea])


def test_warm_land_holds_more_cold_land_less():
    """Directional lock on forced-contrast worlds (the standalone default land runs ~at SST, so the
    contrast is forced via the backbone callables, one map per sign): columns over warm land
    anomalies hold MORE aloft than the row value; cold land columns hold LESS."""
    topo = _topo()
    sst = lambda t: 20.0 + 0.0 * np.asarray(t, float)
    for offs, cmp in [(+20.0, np.greater), (-20.0, np.less)]:
        tl = lambda t, o=offs: 20.0 + o + 0.0 * np.asarray(t, float)
        m = StageCMap(topo, StageCConfig(sst_C_of_theta=sst, Tland_C_of_theta=tl,
                                         Tland_interior_C_of_theta=tl))
        W2 = np.broadcast_to(m.Wsat_aloft_map, m.land.shape)
        row = np.broadcast_to(m.Wsat_aloft[:, None], m.land.shape)
        dT = m.T_map_C - m.SST_C[:, None]
        sel = m.land & (np.abs(dT) > 2.0) & (np.sign(dT) == np.sign(offs))
        assert sel.any()
        assert np.all(cmp(W2[sel], row[sel]))


def test_gamma_run_finite_and_closed():
    """Stepping smoke with the 2-D capacity: finite non-negative P, wind cap respected, cumulative
    water closure in the standalone partially-relaxed band (a loose 0.3 floor -- looser than the
    salt-CCN lock's 0.4; both are order-of-magnitude closure smokes, not tight budget gates)."""
    m = StageCMap(_topo(), StageCConfig(dt=900.0, wa_rh_crit=0.80))
    for _ in range(120):
        m.step()
    d = m.diagnostics()
    assert np.isfinite(d["P_mm_day"]).all() and (d["P_mm_day"] >= 0).all()
    assert np.abs(d["u"]).max() <= 40.1 and np.abs(d["v"]).max() <= 40.1
    assert 0.3 < d["water_closure_cum"] < 1.5
