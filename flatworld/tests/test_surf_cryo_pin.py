"""Cryo wetness pin (``surf_water_cryo_pin``) — a rider on the #302/#303 surface-water budget, default
OFF.

Below-freezing ground is ice/frost-covered, not dry salt flat: over any long horizon the cold trap
accumulates an ice cover (deposition beats sublimation below freezing) that a short simulation window
can never grow from its own rain, so the cover is PRESCRIBED, keyed on the physical freeze threshold
``surf_water_cryo_tfreeze`` (default 273.15 K) rather than a chosen coastline x. Each thermo substep,
wherever ``T_surf < surf_water_cryo_tfreeze``, ``W_surf`` is FLOORED (a one-way top-up, NEVER a
removal) at ``W_cryo = surf_water_cryo_w if > 0 else surf_water_wmin`` [kg/m^2] — the minimum mass at
which ``wetness = clip(W/wmin, 0, 1)`` reads exactly 1 (an ice COVER, not an ocean). The ``surf_salt``
rider gives the top-up water ocean salinity (``surf_salt_salinity0``): ``S_dis += topup*salinity0``.

Top-ups are ACCOUNTED into ``cum_w_cryo`` / ``cum_s_cryo`` (the snap/edge ``_accum_pending`` idiom) so
the machine-precision closures still gate: ``surf_residual = surf_store - (cum_p_land - cum_evap -
cum_w_snap + cum_w_edge + cum_w_cryo)`` and ``salt_residual = salt_store - cum_s_edge - cum_s_cryo``.
The block runs AFTER the snap-to-zero floor (the top-up must not be snapped) and BEFORE the
``surf_water_edge_source`` Dirichlet pin (the wall cell's full moving layer wins over the floor).

Battery style: whole ``_thermo_substep``s on constructed inert states (the test_surf_salt /
test_surf_water_budget pattern), closures asserted near machine precision (gated RELATIVE to the cum
terms — 1e-9 relative is used throughout since these are the newest accounting terms; they are exact-0
in the simple single-substep cases below).
"""
import warnings

import numpy as np
import pytest

from flatworld import constants as C
from flatworld.stage_ad_core import StageADConfig, StageADCore

DT = 60.0
COLD = 250.0                     # skin cold enough that qsat ~ 0 -> evaporation negligible/zero
WARM = 330.0


def _core(**kw):
    kw.setdefault("nx", 8)
    kw.setdefault("nz", 12)
    kw.setdefault("theta_max_deg", 70.0)
    kw.setdefault("prognostic_condensate", True)
    kw.setdefault("precip_reevap", True)
    kw.setdefault("surf_water_budget", True)
    return StageADCore(StageADConfig(**kw))


def _inert_state(m, T_skin, precip=0.0, T_air=320.0, q0=1e-4, W=None, S_dis=None, S_dep=None):
    """Isothermal, deeply sub-saturated column with a prescribed skin, lagged surface flux, and
    hand-set surface water / salt columns (verbatim from test_surf_salt.py / test_surf_water_budget.py)."""
    m.T_atm[...] = T_air
    m.q[...] = q0
    m.q_c[...] = 0.0
    m.T_surf[...] = np.asarray(T_skin, float)
    m._precip = np.asarray(np.broadcast_to(np.asarray(precip, float), (m.cfg.nx,)), float).copy()
    if W is not None:
        m.W_surf[...] = np.asarray(np.broadcast_to(np.asarray(W, float), (m.cfg.nx,)), float)
    if S_dis is not None:
        m.S_dis[...] = np.asarray(np.broadcast_to(np.asarray(S_dis, float), (m.cfg.nx,)), float)
    if S_dep is not None:
        m.S_dep[...] = np.asarray(np.broadcast_to(np.asarray(S_dep, float), (m.cfg.nx,)), float)


# ============================================================================ flags & OFF-gates
def test_flag_defaults_and_env_plumbing(monkeypatch):
    cfg = StageADConfig()
    assert cfg.surf_water_cryo_pin is False
    assert cfg.surf_water_cryo_tfreeze == pytest.approx(273.15)
    assert cfg.surf_water_cryo_w == pytest.approx(0.0)
    monkeypatch.setenv("COSMO_SURF_WATER_CRYO_PIN", "1")
    monkeypatch.setenv("COSMO_SURF_WATER_CRYO_TFREEZE", "260.0")
    monkeypatch.setenv("COSMO_SURF_WATER_CRYO_W", "5.0")
    cfg = StageADConfig()
    assert cfg.surf_water_cryo_pin is True
    assert cfg.surf_water_cryo_tfreeze == pytest.approx(260.0)
    assert cfg.surf_water_cryo_w == pytest.approx(5.0)


def test_cryo_without_budget_warns_and_stays_inert():
    """surf_water_cryo_pin without surf_water_budget: WARN (not raise -- canonical_env can carry a
    rider that only acts when the water budget itself is ON) and stay structurally inert."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = _core(surf_water_budget=False, surf_water_cryo_pin=True)
        assert any("surf_water_cryo_pin" in str(x.message) for x in w)
        assert any("surf_water_budget" in str(x.message) for x in w)
    assert m.W_surf is None
    _inert_state(m, WARM)
    m._thermo_substep(DT)                                     # no crash, nothing allocated
    assert m.W_surf is None


# ============================================================================ the floor itself
def test_cryo_floor_tops_up_cold_cells_only():
    """Half-cold, half-warm world, zero water everywhere: one substep floors ONLY the cold cells to
    wmin (wetness reads exactly 1 there), leaves the warm cells bone dry, and the accounted top-up
    alone closes the surface gate."""
    m = _core(surf_water_cryo_pin=True)
    T = np.array([COLD] * 4 + [WARM] * 4)
    cold = T < m.cfg.surf_water_cryo_tfreeze
    _inert_state(m, T, W=0.0)
    m.reset_water_budget()
    m._thermo_substep(DT)
    W = np.asarray(m.W_surf)
    wet = np.asarray(m._wetness())
    assert np.all(W[cold] == pytest.approx(m.cfg.surf_water_wmin, rel=1e-9))
    assert np.all(wet[cold] == pytest.approx(1.0, abs=1e-12))
    assert np.all(W[~cold] == 0.0)                            # no top-up, no fabricated water
    wc = m.water_conservation()
    assert wc["cum_w_cryo"] > 0.0
    scale = max(abs(wc["W_total"]), abs(wc["cum_w_cryo"]), 1.0)
    assert abs(wc["surf_residual"]) <= 1e-9 * scale


def test_cryo_floor_is_one_way_never_removes():
    """The floor only ADDS water: a cold cell already above wmin is left untouched (never pulled DOWN
    toward the floor), and a cold cell below wmin is topped to EXACTLY wmin. Evaporation is eliminated
    (rather than just tolerated) by seeding q above qsat(250 K) -- the test_surf_salt cold-state trick
    -- so the top-up is the ONLY thing that can move W this substep (drift=0 kills transport too)."""
    m = _core(surf_water_cryo_pin=True, surf_water_drift=0.0)
    _inert_state(m, COLD, W=0.0, q0=2e-3)                     # q > qsat(250 K) ~ 6e-4 -> zero E deficit
    i_high, i_low = 2, 5
    m.W_surf[i_high] = 50.0                                   # already well above the wmin=10 floor
    m.W_surf[i_low] = 3.0                                     # below the floor
    m.reset_water_budget()
    m._thermo_substep(DT)
    W = np.asarray(m.W_surf)
    assert W[i_high] == pytest.approx(50.0, rel=1e-12)        # untouched -- never pulled toward the floor
    assert W[i_low] == pytest.approx(m.cfg.surf_water_wmin, rel=1e-12)   # topped exactly to wmin


def test_retreat_leaves_no_ratchet():
    """A cold cell pins to the ~10 mm film; once it warms past tfreeze the film is left to evaporate on
    its own (no further top-up, no re-triggering), draining over several substeps -- the doc'd 'no
    melt-pulse ratchet' behavior."""
    m = _core(surf_water_cryo_pin=True, surf_water_drift=0.0)
    _inert_state(m, COLD, W=0.0)
    m.reset_water_budget()
    m._thermo_substep(DT)
    W_pinned = np.asarray(m.W_surf).copy()
    assert np.all(W_pinned == pytest.approx(m.cfg.surf_water_wmin, rel=1e-9))
    cum_before = m.water_conservation()["cum_w_cryo"]
    m.T_surf[...] = WARM                                      # the freeze line retreats
    for _ in range(10):
        m._thermo_substep(DT)
    wc = m.water_conservation()
    assert wc["cum_w_cryo"] == pytest.approx(cum_before, rel=1e-12)   # no further top-up once warm
    assert np.all(np.asarray(m.W_surf) < W_pinned)            # the film evaporates away


# ============================================================================ the surf_salt rider
def test_salt_rider_ocean_salinity_on_first_topup_then_concentrates():
    """The topped-up water carries S_dis/W == salinity0 EXACTLY on the very first top-up (a clean cell
    receiving only cryo-sourced water); over subsequent substeps evaporation (which moves no salt)
    concentrates the standing film, so S_dis/W drifts slightly ABOVE salinity0 -- that is correct
    physics, not a bug, so it is not asserted to stay pinned at 0.035. The salt gate still closes,
    relative to the accounted cum_s_cryo."""
    m = _core(surf_water_cryo_pin=True, surf_salt=True)
    _inert_state(m, COLD, W=0.0)
    m.S_dis[...] = 0.0
    m.S_dep[...] = 0.0
    m.reset_water_budget()
    m._thermo_substep(DT)
    W1 = np.asarray(m.W_surf)
    S1 = np.asarray(m.S_dis)
    assert np.all(W1 > 0.0)
    assert S1 / W1 == pytest.approx(m.cfg.surf_salt_salinity0, rel=1e-12)
    for _ in range(9):
        m._thermo_substep(DT)
    W2 = np.asarray(m.W_surf)
    S2 = np.asarray(m.S_dis)
    assert np.all(S2 / W2 >= m.cfg.surf_salt_salinity0 * (1.0 - 1e-9))   # never DILUTES below salinity0
    wc = m.water_conservation()
    s_scale = max(abs(wc["S_total"]), abs(wc["cum_s_cryo"]), 1.0)
    assert abs(wc["salt_residual"]) <= 1e-9 * s_scale
    w_scale = max(abs(wc["W_total"]), abs(wc["cum_w_cryo"]), 1.0)
    assert abs(wc["surf_residual"]) <= 1e-9 * w_scale


# ============================================================================ the knobs
def test_cryo_tfreeze_knob_moves_the_threshold():
    """A colder threshold (240 K) leaves a 250 K skin UNPINNED -- the physical 0 C default is a live,
    sweepable knob, not hard-coded."""
    m = _core(surf_water_cryo_pin=True, surf_water_cryo_tfreeze=240.0)
    _inert_state(m, COLD, W=0.0)
    m.reset_water_budget()
    m._thermo_substep(DT)
    assert np.all(np.asarray(m.W_surf) == 0.0)
    assert m.water_conservation()["cum_w_cryo"] == 0.0


def test_cryo_w_knob_sets_an_explicit_floor():
    """surf_water_cryo_w > 0 overrides the '= wmin' default: the floor lands at the explicit value and
    wetness still reads exactly 1 (clip saturates above wmin regardless of the floor's magnitude)."""
    m = _core(surf_water_cryo_pin=True, surf_water_cryo_w=25.0)
    _inert_state(m, COLD, W=0.0)
    m.reset_water_budget()
    m._thermo_substep(DT)
    W = np.asarray(m.W_surf)
    assert np.all(W == pytest.approx(25.0, rel=1e-12))
    assert np.all(np.asarray(m._wetness()) == pytest.approx(1.0, abs=1e-12))


def test_cryo_w_below_snap_floor_raises():
    """An explicit positive cover thinner than the denormal snap floor (W_SNAP_FLOOR = 1e-30) would be
    snapped to 0 and re-floored every substep -- accounted, so no leak, but pure denormal churn with
    wetness ~ 0. The config validation makes that nonsense config loud (review note on the cryo-pin
    PR). <= 0 (the '= wmin' sentinel) and any physical depth stay accepted."""
    with pytest.raises(ValueError, match="snap floor"):
        _core(surf_water_cryo_pin=True, surf_water_cryo_w=1e-35)
    _core(surf_water_cryo_pin=True, surf_water_cryo_w=0.0)       # sentinel: fine
    _core(surf_water_cryo_pin=True, surf_water_cryo_w=25.0)      # physical depth: fine


# ============================================================================ checkpoint formats
def test_checkpoint_round_trip_cryo_keys():
    """cum_w_cryo / cum_s_cryo round-trip through state()/load_state() and the closure gates stay
    continuous across the resume (more substeps after the reload still close relative to the reloaded
    cums)."""
    m = _core(surf_water_cryo_pin=True, surf_salt=True)
    _inert_state(m, COLD, W=0.0)
    m.S_dis[...] = 0.0
    m.S_dep[...] = 0.0
    m.reset_water_budget()
    for _ in range(3):
        m._thermo_substep(DT)
    st = m.state()
    assert "cum_w_cryo" in st and "cum_s_cryo" in st
    m2 = _core(surf_water_cryo_pin=True, surf_salt=True)
    m2.load_state(st)
    assert m2._cum_w_cryo == m._cum_w_cryo
    assert m2._cum_s_cryo == m._cum_s_cryo
    for _ in range(3):
        m2._thermo_substep(DT)
    wc = m2.water_conservation()
    w_scale = max(abs(wc["W_total"]), abs(wc["cum_w_cryo"]), 1.0)
    s_scale = max(abs(wc["S_total"]), abs(wc["cum_s_cryo"]), 1.0)
    assert abs(wc["surf_residual"]) <= 1e-9 * w_scale
    assert abs(wc["salt_residual"]) <= 1e-9 * s_scale


def test_cryo_off_no_keys_and_zero_cum():
    """Flag OFF: no cum_w_cryo/cum_s_cryo checkpoint keys, and water_conservation reports the key as
    identically 0.0 (the byte-identical OFF path)."""
    m = _core()                                               # budget ON (helper default), cryo OFF
    _inert_state(m, COLD, W=0.0)
    m._thermo_substep(DT)
    st = m.state()
    assert "cum_w_cryo" not in st and "cum_s_cryo" not in st
    assert m.water_conservation()["cum_w_cryo"] == 0.0


def test_checkpoint_missing_cryo_keys_loads_as_zero():
    """A checkpoint written before the cryo pin existed (no cum_w_cryo/cum_s_cryo keys) loads into a
    cryo-ON core without error, starting the cums at 0.0 (the back-compat convention every other
    cum_* field already follows)."""
    m = _core(surf_water_cryo_pin=True)
    _inert_state(m, COLD, W=0.0)
    m.reset_water_budget()
    for _ in range(3):
        m._thermo_substep(DT)
    st = m.state()
    st.pop("cum_w_cryo", None)
    m2 = _core(surf_water_cryo_pin=True)
    m2.load_state(st)                                         # no error despite the missing key
    assert m2._cum_w_cryo == 0.0


# ============================================================================ cryo + edge source together
def test_cryo_and_edge_source_together():
    """Both riders ON, all cells cold: the Dirichlet edge pin wins at the wall cell (a full moving layer
    at ocean salinity, NOT the cryo floor -- cryo runs first but the pin overwrites it), every cold
    INTERIOR cell floors at the cryo pin, and BOTH residual gates close with the two accounted top-ups
    (cum_boundary_influx + cum_w_cryo; cum_s_edge + cum_s_cryo)."""
    m = _core(surf_water_cryo_pin=True, surf_water_edge_source=True, surf_salt=True)
    _inert_state(m, COLD, W=0.0)
    m.S_dis[...] = 0.0
    m.S_dep[...] = 0.0
    m.reset_water_budget()
    m._thermo_substep(DT)
    W = np.asarray(m.W_surf)
    W_pin = C.RHO_LIQ * m.cfg.surf_water_h_layer
    assert W[-1] == pytest.approx(W_pin, rel=1e-12)           # the edge pin wins at the wall cell
    assert np.all(W[:-1] == pytest.approx(m.cfg.surf_water_wmin, rel=1e-9))   # interior cells at the floor
    wc = m.water_conservation()
    w_scale = max(abs(wc["W_total"]), abs(wc["cum_boundary_influx"]), abs(wc["cum_w_cryo"]), 1.0)
    s_scale = max(abs(wc["S_total"]), abs(wc["cum_s_edge"]), abs(wc["cum_s_cryo"]), 1.0)
    assert abs(wc["surf_residual"]) <= 1e-9 * w_scale
    assert abs(wc["salt_residual"]) <= 1e-9 * s_scale
