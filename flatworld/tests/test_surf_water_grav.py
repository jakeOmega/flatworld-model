"""Gravity leveling flow (``surf_water_grav_flow``) — default OFF, a rider on the #302 surface-water
budget that adds the height-gradient transport the wind creep is blind to, and RETIRES the binary
halite dam when ON (crust becomes topography in the surface height eta = W/RHO_LIQ + S_dep/RHO_HALITE).

The term (stage_ad_core step 3g): donor = higher-eta side; u_g = min(sqrt(g*head), u_cfl); moving
depth h_eff = min(donor free depth, head/2) where free = max(W - conn_frac*wmin, 0) is the
PERCOLATION cutoff (sub-threshold water is disconnected pools, laterally immobile); donor-cell
fluxes on the creep's telescoping coefficients, S_dis riding at donor salinity.

OFF-path byte-identity is pinned two ways: the untouched test_surf_salt / test_surf_water_budget
batteries exercise the flag-OFF code (the dam block is verbatim there), and the flat-eta test here
asserts the ON path is bit-identical to OFF when the term has nothing to do (zero flux is exact).

Battery style: constructed inert states + direct ``surface_water_salt_step`` calls for the
transport-only cases (evap/precip zeroed -> conservation is exact), whole ``_thermo_substep``s for
the dam-retirement A/B (the test_surf_salt pattern)."""
import numpy as np
import pytest

from flatworld import constants as C
from flatworld.stage_ad_core import StageADConfig, StageADCore, surface_water_salt_step

DT = 60.0
COLD = 250.0                     # skin cold enough that qsat ~ 0 -> evaporation negligible/zero
DAY = 86400.0                    # direct-step dt for the leveling cases (u_cfl stays non-binding)


def _core(**kw):
    kw.setdefault("nx", 8)
    kw.setdefault("nz", 12)
    kw.setdefault("theta_max_deg", 5.0)          # small box -> small dx -> leveling converges in
    kw.setdefault("prognostic_condensate", True)  # tens of DAY-steps instead of ~1e6 (dW/step ~ 1/dx)
    kw.setdefault("precip_reevap", True)
    kw.setdefault("surf_water_budget", True)
    kw.setdefault("surf_salt", True)
    kw.setdefault("surf_water_grav_flow", True)
    kw.setdefault("surf_water_drift", 0.0)       # kill the wind creep: gravity in isolation
    return StageADCore(StageADConfig(**kw))


def _transport_state(m, W, S_dis=0.0, S_dep=0.0):
    """Hand-set surface fields with ALL local sources/sinks zeroed, so direct
    ``surface_water_salt_step`` calls are pure transport (+ saturation adjustment):
    conservation of sum(W*dx) and total salt is then exact, not evap-polluted."""
    m.T_surf[...] = COLD
    m._precip[...] = 0.0
    m._evap[...] = 0.0
    m._surf_flash[...] = 0.0
    m.W_surf[...] = np.asarray(np.broadcast_to(np.asarray(W, float), (m.cfg.nx,)), float)
    m.S_dis[...] = np.asarray(np.broadcast_to(np.asarray(S_dis, float), (m.cfg.nx,)), float)
    m.S_dep[...] = np.asarray(np.broadcast_to(np.asarray(S_dep, float), (m.cfg.nx,)), float)


def _totals(m):
    W = np.asarray(m.W_surf)
    S = np.asarray(m.S_dis) + np.asarray(m.S_dep)
    return float(np.sum(W * m._dx_w)), float(np.sum(S * m._dx_w))


def _steps(m, n, dt=DAY):
    for _ in range(n):
        surface_water_salt_step(m, m.cfg, dt, np)


# ============================================================================ flags & OFF-gates
def test_flag_defaults_and_env_plumbing(monkeypatch):
    cfg = StageADConfig()
    assert cfg.surf_water_grav_flow is False
    assert cfg.surf_water_conn_frac == pytest.approx(0.5)
    monkeypatch.setenv("COSMO_SURF_WATER_GRAV_FLOW", "1")
    monkeypatch.setenv("COSMO_SURF_WATER_CONN_FRAC", "0.25")
    cfg = StageADConfig()
    assert cfg.surf_water_grav_flow is True
    assert cfg.surf_water_conn_frac == pytest.approx(0.25)


def test_flat_eta_on_is_bit_identical_to_off():
    """Uniform W, uniform salt, no crust, creep off -> eta is flat AND STAYS flat -> the gravity
    flux is EXACTLY zero every step, and the ON run must be bit-identical to the OFF run (adding a
    signed 0.0 preserves bits; the dam block it skips is inert with no crust). This is the zero-flux
    limiting case AND the cheap ON/OFF gate. (With creep on, the walls un-flatten eta after one
    substep and ON legitimately diverges -- that interaction is the other tests' job.)"""
    ms = [_core(surf_water_grav_flow=g) for g in (False, True)]
    for m in ms:
        _transport_state(m, W=20.0, S_dis=0.7, S_dep=0.0)   # 35 g/kg, above the conn threshold
        _steps(m, 5, dt=DT)
    for a in ("W_surf", "S_dis", "S_dep"):
        assert np.array_equal(np.asarray(getattr(ms[0], a)), np.asarray(getattr(ms[1], a))), a


# ============================================================================ leveling & conservation
def test_step_profile_levels_and_conserves_exactly():
    """A wet block beside a shallow block levels to the conserved uniform depth (the analytic
    equilibrium: eta uniform, here (200+10)/2 = 105 with equal dx), conserving sum(W*dx) and total
    salt to round-off at every step, monotonically (no sloshing past the mean). The ptp-monotone
    assert holds for locally-monotone profiles like this one -- the scheme's no-overshoot guarantee
    is PER-FACE, and a local eta extremum flanked by much larger heads can transiently reorder
    (bounded, conservative, same equilibrium; see the 3g comment)."""
    m = _core()
    W0 = np.where(np.arange(m.cfg.nx) < m.cfg.nx // 2, 200.0, 10.0)
    _transport_state(m, W=W0, S_dis=np.where(W0 > 100, 7.0, 0.0))
    w_tot0, s_tot0 = _totals(m)
    ptp_prev = np.inf
    for _ in range(3000):                                   # the tail is slow: transfer ~ head^1.5
        surface_water_salt_step(m, m.cfg, DAY, np)
        w_tot, s_tot = _totals(m)
        assert w_tot == pytest.approx(w_tot0, rel=1e-13)
        assert s_tot == pytest.approx(s_tot0, rel=1e-13)
        ptp = float(np.ptp(np.asarray(m.W_surf)))
        assert ptp <= ptp_prev + 1e-9                      # monotone leveling, no rebound
        ptp_prev = ptp
    W = np.asarray(m.W_surf)
    dxw = np.broadcast_to(np.asarray(m._dx_w, float), W0.shape)   # _dx_w is a SCALAR on uniform grids
    target = float(np.sum(W0 * dxw) / np.sum(dxw))
    assert np.ptp(W) < 3.0                                  # leveled (from an initial ptp of 190)
    assert W.mean() == pytest.approx(target, rel=1e-6)


def test_percolation_cutoff_immobile_below_and_parks_at_threshold():
    """(a) A donor BELOW W_conn = conn_frac*wmin moves nothing (bit-identical state: disconnected
    pools); (b) a donor above it drains only its connected excess and PARKS AT the threshold, never
    below (the free-depth form, no binary flip-flop)."""
    m = _core()
    w_conn = m.cfg.surf_water_conn_frac * m.cfg.surf_water_wmin      # 0.5 * 10 = 5 kg/m^2
    # (a) sub-threshold donor: exact immobility
    W0 = np.zeros(m.cfg.nx); W0[0] = 0.8 * w_conn
    _transport_state(m, W=W0)
    before = np.asarray(m.W_surf).copy()
    _steps(m, 50)
    assert np.array_equal(np.asarray(m.W_surf), before)
    # (b) donor above threshold: drains toward the receiver, parks at w_conn
    W0 = np.zeros(m.cfg.nx); W0[0] = w_conn + 3.0
    _transport_state(m, W=W0)
    _steps(m, 4000)
    W = np.asarray(m.W_surf)
    assert W[0] >= w_conn - 1e-9                            # never below the threshold
    assert W[0] == pytest.approx(w_conn, abs=0.2)           # parked at it
    assert W[1] == pytest.approx(3.0, abs=0.2)              # the excess arrived next door


# ============================================================================ crust as topography
def test_crust_sill_blocks_as_topography_and_passes_when_removed():
    """A halite sill TALLER than the upstream pool blocks by eta alone (the donor side of both sill
    interfaces holds no free water, and the pool's eta is below the sill crest) -- the dam behavior
    re-emerging from topography; remove the crust and the same pool floods through."""
    for sill_m, expect_flow in ((0.5, False), (0.0, True)):
        m = _core()
        W0 = np.zeros(m.cfg.nx); W0[0] = 50.0               # 5 cm pool
        S_dep0 = np.zeros(m.cfg.nx); S_dep0[1] = C.RHO_HALITE * sill_m
        _transport_state(m, W=W0, S_dep=S_dep0)
        w_tot0, s_tot0 = _totals(m)
        _steps(m, 200)
        w_tot, s_tot = _totals(m)
        assert w_tot == pytest.approx(w_tot0, rel=1e-13)
        assert s_tot == pytest.approx(s_tot0, rel=1e-13)
        downstream = float(np.sum(np.asarray(m.W_surf)[2:]))
        if expect_flow:
            assert downstream > 1.0                         # flooded past cell 1
        else:
            assert downstream == 0.0                        # sealed behind the sill
            assert np.asarray(m.W_surf)[0] == pytest.approx(50.0)


def test_salt_rides_at_donor_salinity_and_never_goes_negative():
    """Leveling out of a salty pool exports salt at the pool's salinity: the receiver's salinity
    approaches (never exceeds) the donor's, total salt is conserved, and no field goes negative."""
    m = _core()
    W0 = np.zeros(m.cfg.nx); W0[0] = 100.0
    _transport_state(m, W=W0, S_dis=np.where(W0 > 0, 3.5, 0.0))     # 35 g/kg pool
    _, s_tot0 = _totals(m)
    _steps(m, 800)
    W = np.asarray(m.W_surf); Sd = np.asarray(m.S_dis); Sp = np.asarray(m.S_dep)
    assert (W >= 0).all() and (Sd >= 0).all() and (Sp >= 0).all()
    assert _totals(m)[1] == pytest.approx(s_tot0, rel=1e-13)
    live = W > 1.0
    sal = Sd[live] / W[live]
    assert (sal <= 0.035 * (1 + 1e-9)).all()                # never saltier than the source


# ============================================================================ the dam is retired
def test_dam_retired_when_grav_flow_on():
    """The exact test_surf_salt blocking scenario (10 m creep flow vs a 15 m receiver-side crust,
    block_factor 1): with grav_flow OFF the dam blocks (legacy path verbatim); with grav_flow ON the
    dam block is skipped and the creep delivers -- crust now gates flow only through eta (and a 15 m
    crust cell holds no free water, so it cannot push back)."""
    def downstream_gain(grav_on):
        m = _core(nx=8, theta_max_deg=70.0, surf_water_grav_flow=grav_on,
                  surf_water_drift=0.015)                   # wind creep ON (the blocked flow)
        m.T_surf[...] = COLD
        m._precip[...] = 0.0
        m._evap[...] = 0.0
        m._surf_flash[...] = 0.0
        mid = m.cfg.nx // 2
        m.W_surf[...] = 0.0
        m.W_surf[mid + 1:] = 1.0e4                          # 10 m flow
        m.S_dis[...] = 0.0
        m.S_dep[...] = 0.0
        m.S_dep[mid] = C.RHO_HALITE * 15.0                  # 15 m crust on the receiver
        w0 = float(np.asarray(m.W_surf)[mid])
        surface_water_salt_step(m, m.cfg, DT, np)
        return float(np.asarray(m.W_surf)[mid]) - w0
    assert downstream_gain(False) == 0.0                    # legacy dam blocks
    assert downstream_gain(True) > 0.0                      # retired: the creep delivers
