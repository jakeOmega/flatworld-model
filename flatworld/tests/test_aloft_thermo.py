"""Workstream C: Stage C consumes the AD core's RESOLVED aloft thermodynamics behind a default-OFF flag.

Stage C's aloft saturation moisture path Wsat_aloft [mm] was derived from a FAKE FIXED-LAPSE aloft
temperature (T_aloft = SST - lapse_aloft*H_aloft). The AD core resolves the aloft profile and exposes it
via cascade.stage_c_inputs(); the `aloft_thermo_from_core` flag (env COSMO_ALOFT_THERMO_FROM_CORE) lets
Stage C take Wsat_aloft from that profile instead. This file gates:
  * BYTE-IDENTICAL: with the flag OFF (default), supplying the callables changes nothing.
  * CONSUMPTION: with the flag ON, a clearly-different resolved profile actually moves Wsat_aloft (and
    the downstream precip), proving Stage C reads the core's profile.
"""
import numpy as np

from flatworld.stage_c_map import StageCMap, StageCConfig
from flatworld.topography import Topography


def _topo():
    return Topography.procedural_continent(seed=3, nx=40, ne=40, Le_km=6000)


def _fast():
    return dict(dt=900.0, check_every=300, tol=5e-4, patience=3, max_seconds=30)


# a resolved Wsat_aloft that is CLEARLY different from any fixed-lapse value (a flat, small column)
def _resolved_wsat(th):
    return np.full_like(np.asarray(th, float), 2.0)   # [mm]


def test_flag_off_supplying_callables_is_byte_identical():
    """OFF path (default): the resolved-profile callables are present but IGNORED -> Wsat_aloft verbatim."""
    t = _topo()
    base = StageCMap(t, StageCConfig())                                  # no callables at all
    withcb = StageCMap(t, StageCConfig(Wsat_aloft_of_theta=_resolved_wsat,
                                       T_aloft_of_theta=lambda th: np.full_like(np.asarray(th, float), -40.0),
                                       RH_aloft_of_theta=lambda th: np.full_like(np.asarray(th, float), 0.5)))
    assert withcb.cfg.aloft_thermo_from_core is False                   # default OFF (env unset)
    assert np.array_equal(base.Wsat_aloft, withcb.Wsat_aloft)           # exact, not just close


def test_flag_off_full_run_is_byte_identical():
    """End-to-end: the deliverable precip map is byte-identical with the callables supplied, flag OFF."""
    t = _topo()
    base = StageCMap(t, StageCConfig(**_fast()))
    withcb = StageCMap(t, StageCConfig(Wsat_aloft_of_theta=_resolved_wsat, **_fast()))
    base.run(verbose=False); withcb.run(verbose=False)
    assert np.array_equal(base.diagnostics()["P_mm_day"], withcb.diagnostics()["P_mm_day"])


def test_flag_on_consumes_resolved_wsat():
    """ON + Wsat_aloft_of_theta -> Wsat_aloft IS the resolved column, differing from the fixed-lapse value."""
    t = _topo()
    base = StageCMap(t, StageCConfig())                                  # fixed-lapse stand-in
    on = StageCMap(t, StageCConfig(aloft_thermo_from_core=True, Wsat_aloft_of_theta=_resolved_wsat))
    assert not np.allclose(on.Wsat_aloft, base.Wsat_aloft)              # the profile actually moved it
    assert np.allclose(on.Wsat_aloft, _resolved_wsat(t.theta_deg))      # and it IS the resolved column


def test_flag_on_falls_back_to_T_aloft():
    """ON with only T_aloft_of_theta (no Wsat callable) -> Wsat_aloft derived from the resolved aloft T."""
    from flatworld.stage_c_map import _wsat_column
    from flatworld import constants as C
    t = _topo()
    T_aloft_C = lambda th: np.full_like(np.asarray(th, float), -30.0)   # a cold resolved aloft layer [C]
    on = StageCMap(t, StageCConfig(aloft_thermo_from_core=True, T_aloft_of_theta=T_aloft_C))
    expect = _wsat_column(np.maximum(T_aloft_C(t.theta_deg) + C.T_FREEZE, 180.0))
    assert np.allclose(on.Wsat_aloft, expect)


def test_flag_on_moves_precip():
    """ON: a clearly-drier resolved aloft column reduces the aloft saturation -> a different precip map."""
    t = _topo()
    base = StageCMap(t, StageCConfig(**_fast()))
    on = StageCMap(t, StageCConfig(aloft_thermo_from_core=True, Wsat_aloft_of_theta=_resolved_wsat, **_fast()))
    base.run(verbose=False); on.run(verbose=False)
    assert not np.array_equal(base.diagnostics()["P_mm_day"], on.diagnostics()["P_mm_day"])
