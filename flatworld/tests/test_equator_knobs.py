"""Regression tests for the equatorial-chimney DIAGNOSTIC knobs (default-OFF):
``equator_norain_cells``, ``equator_norain_keep_ceiling``, ``equator_ceiling_lwc_scale``.

These lock the CROSS-TRACK axis of the knobs. The original `equator_ceiling_lwc_scale` bug indexed
``qc_ceiling[..., :N]`` — which (the field is ``(nx, nz)``) scaled the first N *vertical levels*, not the
first N *cross-track cells* (the equator is axis 0). It was silent — the cap-sweep returned byte-identical
output across scales. The guards here: scaling the cap must change the EQUATOR cells (axis 0) and leave a
poleward column untouched; and the default (``equator_norain_cells=0``) must be byte-identical regardless
of the scale.
"""
import numpy as np
from flatworld import thermo as th
from flatworld.stage_ad_core import StageADCore, StageADConfig


def _moist_core():
    """A small prognostic-condensate core, hot + SATURATED + condensate-laden so cloud cover ~ 1 and the
    retained-water CEILING is the active precip sink at every column (so scaling it has a measurable effect)."""
    m = StageADCore(StageADConfig(nx=20, nz=16, prognostic_condensate=True, qc_retained_ceiling=True))
    m.T_atm = np.full((m.cfg.nx, m.cfg.nz), 290.0)
    m.q_c = np.full((m.cfg.nx, m.cfg.nz), 5.0e-3)               # condensate well above the ~0.3 g/m^3 ceiling
    return m


def _q_saturated(m):
    return th.qsat(m.T_atm, m.p_mid[None, :]) * 1.2            # supersaturated -> full cover + condensation


def _precip(m, q, ncells, scale, keep_ceiling=True):
    """One prognostic-microphysics step on a COPY of the state -> surface precip [kg/m^2/s] per cross-track
    cell (axis 0). Restores the mutated state so the caller can re-run with other knobs."""
    qc0, Ta0 = m.q_c.copy(), m.T_atm.copy()
    m.cfg.equator_norain_cells = int(ncells)
    m.cfg.equator_norain_keep_ceiling = bool(keep_ceiling)
    m.cfg.equator_ceiling_lwc_scale = float(scale)
    m._prognostic_microphysics(q.copy(), dt=60.0)
    P = np.asarray(m._precip).copy()
    m.q_c, m.T_atm = qc0, Ta0
    return P


def test_ceiling_scale_acts_on_cross_track_equator_cells_only():
    """Scaling the equatorial cap (keep-ceiling) must change the precip in the first N CROSS-TRACK cells and
    leave the poleward cells byte-identical -- the axis the bug got wrong (it scaled the vertical instead)."""
    m = _moist_core()
    q = _q_saturated(m)
    N = 4
    P1 = _precip(m, q, ncells=N, scale=1.0)
    P100 = _precip(m, q, ncells=N, scale=100.0)
    assert (P1[:N] > 0).all(), "fixture sanity: the ceiling must be raining at the equator cells"
    # equator cells (axis 0, [:N]) must change; a higher cap retains more q_c -> LESS surface precip there
    assert not np.allclose(P1[:N], P100[:N]), "scaling the cap did not change the equator cells (axis bug?)"
    assert (P100[:N] < P1[:N]).all(), "a higher equatorial cap should reduce local precip"
    # poleward cells (>= N) must be untouched
    assert np.allclose(P1[N:], P100[N:]), "the equator knob leaked into the poleward (off-axis) cells"


def test_default_off_is_byte_identical():
    """equator_norain_cells=0 (knob off) is byte-identical regardless of the cap scale (the scale-line is gated
    on ncells>0 and on scale!=1.0)."""
    m = _moist_core()
    q = _q_saturated(m)
    base = _precip(m, q, ncells=0, scale=1.0)
    assert np.array_equal(base, _precip(m, q, ncells=0, scale=100.0))   # scale ignored when the knob is off
    # and with the knob ON, scale=1.0 is the no-op vs scale=1.0 (the != 1.0 guard never fires)
    on = _precip(m, q, ncells=4, scale=1.0)
    assert np.array_equal(on, _precip(m, q, ncells=4, scale=1.0))
