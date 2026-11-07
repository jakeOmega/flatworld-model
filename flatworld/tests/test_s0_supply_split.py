"""S0 supply-split instrumentation (ROADMAP_BC_moisture_circulation_feedback.md) -- the default-OFF
limb-knockout hook in StageCMap must be byte-identical when off, must touch ONLY the aloft layer W_a
(the surface layer W_s and its drain sink are untouched), and must monotonically reduce the aloft supply
and the precip when a limb is removed."""
import numpy as np
from flatworld.topography import Topography
from flatworld.stage_c_map import StageCMap, StageCConfig


def _maps(n_make):
    """n_make fresh StageCMaps on the same topology/cfg (deterministic, identical initial state)."""
    topo = Topography.procedural_continent(seed=3, nx=40, ne=40, Le_km=6000)
    return [StageCMap(topo, StageCConfig()) for _ in range(n_make)]


def _step(m, n=40):
    for _ in range(n):
        m.step()


def test_s0_knockout_flags_default_off_and_are_byte_identical():
    base, ref = _maps(2)
    assert base._s0_kill_F is False and base._s0_kill_U is False     # defaults OFF
    ref._s0_kill_F = False; ref._s0_kill_U = False                   # explicit OFF must match the default
    _step(base); _step(ref)
    assert np.array_equal(base.W_a, ref.W_a)
    assert np.array_equal(base.W_s, ref.W_s)
    assert np.array_equal(base.P, ref.P)


def test_s0_knockouts_touch_only_the_aloft_layer():
    # After ONE step (precip EMA still 0 -> no land-recycling feedback into W_s yet), each knockout changes
    # ONLY the aloft layer W_a; the surface layer W_s is bit-identical to baseline. (Over many steps the
    # changed precip perturbs W_s over land via recycling and advects across coasts, so the exact invariant
    # is a single-step property.)
    base, killF, killU = _maps(3)
    killF._s0_kill_F = True
    killU._s0_kill_U = True
    for m in (base, killF, killU):
        _step(m, n=1)
    assert np.array_equal(base.W_s, killF.W_s)
    assert np.array_equal(base.W_s, killU.W_s)
    assert killF.W_a.sum() < base.W_a.sum()                         # F delivery removed from W_a
    assert killU.W_a.sum() < base.W_a.sum()                         # U detrainment removed from W_a


def test_s0_kill_F_reduces_aloft_and_precip():
    base, killF = _maps(2)
    killF._s0_kill_F = True
    _step(base); _step(killF)
    assert np.all(killF.W_a <= base.W_a + 1e-12)                    # F is a non-negative source: removing it can only lower W_a
    assert killF.W_a.sum() < base.W_a.sum()
    assert killF.P.sum() < base.P.sum()                            # ... and it rains out -> less precip


def test_s0_kill_U_reduces_aloft():
    base, killU = _maps(2)
    killU._s0_kill_U = True
    _step(base); _step(killU)
    assert np.all(killU.W_a <= base.W_a + 1e-12)
    assert killU.W_a.sum() < base.W_a.sum()                        # U is a real, active aloft source


def test_s0_kill_both_is_not_more_than_either_alone():
    base, killF, killU, killB = _maps(4)
    killF._s0_kill_F = True
    killU._s0_kill_U = True
    killB._s0_kill_F = True; killB._s0_kill_U = True
    for m in (base, killF, killU, killB):
        _step(m)
    # removing both sources cannot leave MORE aloft moisture than removing just one
    assert killB.W_a.sum() <= killF.W_a.sum() + 1e-9
    assert killB.W_a.sum() <= killU.W_a.sum() + 1e-9
