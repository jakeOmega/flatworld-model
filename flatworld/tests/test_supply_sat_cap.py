"""The aloft-supply SATURATION CAP (`wa_supply_sat_cap`, default ON at 1.0).

The supply relaxation is a boundary-condition device (moisture arriving via the unresolved
poleward-aloft limb). Uncapped, a target above a cold column's aloft saturation capacity turns it
into a moisture pump: the lens_diurnal wet-belt artifact (2026-07-18) force-fed 4-9 mm/day of
spurious rainout over theta 47-60 (sea AND land) where the legacy 8 mm F-floor stood against a
collapsed Wsat_aloft (~0-2 mm). The cap clips the TARGET at cap*Wsat_aloft: the relaxation may
fill a column TO saturation, never through it. Where Wsat >= F the clip never binds (validated
warm-flank behavior unchanged); genuine cold-ward advective rainout is untouched (it lives in
advWa, not the target). `inf` is the byte-identical legacy escape.
"""
import numpy as np

from flatworld.stage_c_map import StageCMap, StageCConfig
from flatworld.topography import Topography


def _small_map(**cfg_kw):
    t = Topography.procedural_continent(seed=11, nx=40, ne=32, Le_km=6000)
    return StageCMap(t, StageCConfig(**cfg_kw))


def _set_wsat(m, wsat_row):
    """Hand-set the per-row aloft capacity AND refresh the derived 2-D map (`Wsat_aloft_map`, built
    in __init__) in its gamma-off per-row form -- these tests pin the CAP channel's per-row algebra
    in isolation; the gamma land modulation has its own exact tests (test_wsat_gamma)."""
    m.Wsat_aloft = np.asarray(wsat_row, float)
    m.Wsat_aloft_map = m.Wsat_aloft[:, None]


def test_cap_binds_where_target_exceeds_capacity():
    m = _small_map()
    m.F = np.full(m.nx, 8.0)                                   # the legacy floor as a uniform target
    _set_wsat(m, np.linspace(20.0, 0.1, m.nx))                 # warm -> collapsed-cold capacity
    F2 = m._supply_target()
    assert np.all(F2 <= m.Wsat_aloft[:, None] + 1e-12)         # never above capacity
    expect = np.minimum(8.0, m.Wsat_aloft)[:, None]
    assert np.allclose(F2, np.broadcast_to(expect, F2.shape))  # exactly min(F, Wsat)


def test_cap_is_inert_where_capacity_is_ample():
    m = _small_map()
    _set_wsat(m, np.full(m.nx, 100.0))                         # capacity far above any target
    capped = m._supply_target().copy()
    m.cfg.wa_supply_sat_cap = np.inf
    uncapped = m._supply_target()
    assert np.array_equal(capped, uncapped)                    # the clip never binds -> identical


def test_inf_cap_is_the_legacy_law_byte_identical():
    m = _small_map(wa_supply_sat_cap=np.inf)
    F2 = m._supply_target()
    assert np.array_equal(F2, np.broadcast_to(m.F[:, None], (m.nx, m.ne)))  # pre-fix formula verbatim


def test_default_is_on_at_the_physical_value():
    assert StageCConfig().wa_supply_sat_cap == 1.0


def test_cold_strip_forced_rainout_collapses_under_cap():
    """Behavioral lock on the artifact. The pump acts LOCALLY at every cold row (legacy holds
    ~F/tau_supply everywhere); genuine advective shedding acts only near the warm->cold entry and
    decays inward. So the discriminating region is the DEEP cold interior: legacy stays pumped
    (~9 mm/day here), capped decays to the advective residual (~1.5). Warm rows unchanged."""
    P = {}
    for name, cap in (("capped", 1.0), ("legacy", np.inf)):
        m = _small_map(wa_supply_sat_cap=cap)
        m.F = np.full(m.nx, 8.0)
        _set_wsat(m, np.where(np.arange(m.nx) >= m.nx // 2, 0.2, 60.0))    # poleward half: collapsed
        for _ in range(300):
            m.step()
        P[name] = m.P.copy()
        nx = m.nx
    deep = np.arange(nx) >= (3 * nx) // 4                      # deep cold interior (past the entry shedding)
    p_deep_legacy = float(P["legacy"][deep].mean())
    p_deep_capped = float(P["capped"][deep].mean())
    assert p_deep_legacy > 8.0                                 # the pump reproduces (test has teeth)
    assert p_deep_capped < 2.5                                 # capped: only the advective residual
    assert p_deep_capped < p_deep_legacy / 4.0                 # and the channel collapses
    # warm-side invariance away from the strip boundary (the 2 rows nearest the transition feel the
    # capped strip's changed upstream W_a; cell-wise equality there is not the claim)
    warm_interior = np.arange(nx) < nx // 2 - 2
    assert np.allclose(P["capped"][warm_interior].mean(axis=1),
                       P["legacy"][warm_interior].mean(axis=1), rtol=0.05, atol=0.1)
