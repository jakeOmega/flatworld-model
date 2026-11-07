"""
Tests for the diurnal sea-breeze -> Stage C precip coupling (issue #131).

Stage C precip is built on a STATIC day-mean wind, so the strong AFTERNOON sea-breeze convergence
(a nonlinear rain trigger) is averaged away. This module couples the existing diurnal breeze
(`bl_diurnal`, #113) into the moisture/precip via `stage_c_diurnal`, either as the full sub-daily
periodic solve (`solve_diurnal_precip`) or the cheap peak-convergence kinematic overlay
(`afternoon_precip_overlay`).

Load-bearing invariants pinned here:
  * the `_step_with_wind` seam is a PURE REFACTOR -- `step()` is byte-for-byte identical (the deliverable
    day-mean map is untouched);
  * the periodic precip solve converges and closes its water budget over the last cycle;
  * the diurnal rain RHYTHM (amplitude) tracks the afternoon convergence (the rain is breeze-driven);
  * the kinematic overlay's afternoon snapshot rains more than the day-mean snapshot (it OVER-reads the
    net effect -- the faithful periodic net total-rain change is small / not robustly positive, see #131);
  * the CFL sub-cycle stays finite under a strong breeze;
  * the kinematic overlay leaves the moisture state untouched (a true diagnostic).
"""
import numpy as np
import pytest

from flatworld.topography import Topography
from flatworld.stage_c_map import StageCMap, StageCConfig
from flatworld import stage_c_diurnal as scd
from flatworld import geometry as g


# ---- shared fixtures: a small converged coast map + a synthetic diurnal breeze solve ----
def _coast_map(seed=5, nx=30, ne=36):
    """A small Stage C map run to a quick quasi-steady state (the day-mean warm start)."""
    t = Topography.procedural_continent(seed=seed, nx=nx, ne=ne, Le_km=5000)
    m = StageCMap(t, StageCConfig(dt=900.0, check_every=300, tol=5e-4, patience=3, max_seconds=30))
    m.run(verbose=False)
    return m


def _synthetic_sold(m, amp=15.0, K=8):
    """A controlled ``bl_diurnal.solve_diurnal_wind``-shaped result. As in the real breeze solve, the
    per-phase wind is the converged day-mean (balanced) wind PLUS a zero-day-mean diurnal breeze anomaly
    -- here a CONVERGENCE BAND at the centre of the along-track (e) axis: v_anom = a_k*(-tanh((j-ne/2)/s))
    so dv/de < 0 (convergence) when a_k>0 (afternoon) and divergence when a_k<0 (night). a_k = amp*cos(...)
    has zero day-mean, so the day-mean of the phases == the balanced wind (== the overlay's day-mean
    reference), and the afternoon phase adds convergence on top -> the rectification the issue describes."""
    nx, ne = m.nx, m.ne
    j = np.arange(ne)
    prof = -np.tanh((j - ne / 2.0) / 3.0)                      # convergent band at centre when scaled >0
    ub, vb = m.u.copy(), m.v.copy()                            # the converged balanced (day-mean) wind
    T_day = g.CLOCK.T_day
    phase_t = np.linspace(0.0, T_day, K, endpoint=False)
    shape = np.cos(2.0 * np.pi * np.arange(K) / K)             # +1 at noon (k=0), -1 at midnight (k=K/2)
    u_phase = np.zeros((K, nx, ne))
    v_phase = np.zeros((K, nx, ne))
    for k in range(K):
        u_phase[k] = ub
        v_phase[k] = vb + amp * shape[k] * prof[None, :]       # background + zero-day-mean breeze anomaly
    speed_phase = np.hypot(u_phase, v_phase)
    night_mask = shape < 0.0                                   # day = around noon, night = around midnight
    return dict(phase_t=phase_t, u_phase=u_phase, v_phase=v_phase,
                speed_phase=speed_phase, night_mask=night_mask)


def _dcfg(**kw):
    base = dict(cfl_safety=0.8, max_days=40, period_tol=5e-3, period_patience=2, max_seconds=120.0)
    base.update(kw)
    return scd.StageCDiurnalConfig(**base)


# ---- 1. the _step_with_wind seam is a pure refactor: step() is byte-for-byte identical ----
def test_step_with_wind_refactor_byte_identical():
    # Two identically-constructed maps (construction is deterministic given topo + config). Advancing one
    # via step() and the other via _step_with_wind(*_balanced_wind()) at the default dt=None must produce
    # bit-identical W_s / W_a / P -- proving the extraction + dt-default path did not perturb the deliverable.
    def fresh():
        t = Topography.procedural_continent(seed=3, nx=40, ne=40, Le_km=5000)
        return StageCMap(t, StageCConfig())
    a, b = fresh(), fresh()
    for _ in range(50):
        a.step()
        b._step_with_wind(*b._balanced_wind())   # dt defaults to cfg.dt -> identical arithmetic
    assert np.array_equal(a.W_s, b.W_s)
    assert np.array_equal(a.W_a, b.W_a)
    assert np.array_equal(a.P, b.P)
    # the seam returns the pre-EMA instantaneous precip [mm/day] and convergence [1/s]
    P_now, conv = a.step()
    assert P_now.shape == a.P.shape and conv.shape == a.P.shape
    assert np.all(np.isfinite(P_now)) and np.all(np.isfinite(conv))


# ---- 2. the periodic precip solve reaches a periodic state within the day guard ----
def test_diurnal_precip_reaches_periodic_state():
    m = _coast_map()
    sold = _synthetic_sold(m)
    sol = scd.solve_diurnal_precip(m, sold, _dcfg(), verbose=False)
    assert sol["converged"] is True
    assert sol["days"] <= sol["days_guard"]
    assert sol["P_phase"].shape == (sold["phase_t"].size, m.nx, m.ne)
    assert np.all(np.isfinite(sol["P_phase"]))


# ---- 3. the periodic solve closes its water budget over the LAST cycle (spin-up reset, not dominated) ----
def test_diurnal_precip_water_budget_closes():
    m = _coast_map()
    sold = _synthetic_sold(m)
    sol = scd.solve_diurnal_precip(m, sold, _dcfg(), verbose=False)
    assert 0.7 <= sol["water_closure_last_cycle"] <= 1.3


# ---- 4. overlay AFTERNOON enhancement: the afternoon snapshot rains more than the day-mean snapshot ----
def test_overlay_afternoon_exceeds_daymean_snapshot():
    # The KINEMATIC overlay (moisture held fixed) shows the afternoon convergence band raining more than the
    # static day-mean snapshot, localized at the front. NB this is the overlay's afternoon enhancement, which
    # OVER-READS the net effect (the standing moisture is not depleted) -- the faithful periodic solve shows
    # the NET day-mean total-rain change is small / not robustly positive (see the quant script + #131 docs).
    m = _coast_map()
    sold = _synthetic_sold(m)
    ov = scd.afternoon_precip_overlay(m, sold)
    assert ov["P_afternoon"].sum() >= ov["P_daymean"].sum()
    # the overlay localizes EXTRA rain where the afternoon flow converges (the front), not uniformly
    assert ov["P_diff"].max() > 0.0


# ---- 5. HEADLINE: the diurnal rain RHYTHM is driven by the afternoon convergence ----
def test_diurnal_rain_rhythm_tracks_afternoon_convergence():
    # The issue's claim is rectification + redistribution: afternoon convergent showers with a strong
    # diurnal rhythm. The mechanism-true, robust signature (NOT the self-consistently-high day-mean
    # correlation, which the equilibrated day-mean precip always has) is that the diurnal precip SWING is
    # concentrated WHERE the breeze drives the convergence swing, and afternoon precip is
    # convergence-correlated. The swing (P_amplitude) is matched against the convergence SWING
    # (conv_amplitude), NOT the instantaneous conv_afternoon snapshot: the precip swing is driven by the
    # time-VARYING (breeze) convergence, whereas the snapshot is dominated by the static background-wind
    # convergence -- correlating a swing against a background-polluted snapshot conflates the two (and is
    # day-mean-wind-dependent: it was incidentally +0.35 under the pre-#151 orography-double-counting wind
    # and flips to -0.32 once g_terrain is corrected to 0). The static background cancels in max-min, so the
    # swing-vs-swing correlation isolates the breeze mechanism and stays POSITIVE regardless of g_terrain
    # (+0.13 at the old 0.6, ~+0.6 at the corrected default across seeds) -- it never sign-flips like the
    # snapshot metric. (conv_amplitude is g_terrain-independent by construction; P_amplitude is not, so the
    # correlation magnitude still moves with the wind -- hence the floor is set off the default-g_terrain ~0.6.)
    m = _coast_map()
    sold = _synthetic_sold(m)
    sol = scd.solve_diurnal_precip(m, sold, _dcfg(), verbose=False)
    r_amp = scd.precip_convergence_corr(sol["P_amplitude"], sol["conv_amplitude"])
    r_aft = scd.precip_convergence_corr(sol["P_afternoon"], sol["conv_afternoon"])
    assert r_amp > 0.3                                       # the rain's diurnal swing follows the breeze convergence swing (obs ~+0.59)
    assert r_aft > 0.2                                       # afternoon precip is substantially convergence-driven
    assert sol["P_amplitude"].max() > 0.0                    # there IS a diurnal rain rhythm


# ---- 6. the CFL sub-cycle stays finite under a strong (inflated) breeze ----
def test_cfl_subcycle_stable_under_strong_breeze():
    m = _coast_map()
    sold = _synthetic_sold(m, amp=60.0)                     # ~60 m/s: would be unstable at the day-mean dt
    sol = scd.solve_diurnal_precip(m, sold, _dcfg(max_days=12), verbose=False)
    assert np.all(np.isfinite(sol["P_phase"]))
    assert np.all(np.isfinite(m.W_s)) and np.all(np.isfinite(m.W_a))
    assert sol["n_sub_max"] > 1                             # it actually sub-cycled (dt < a full phase step)


# ---- 7. the kinematic overlay is a NO-OP on the moisture state (a pure diagnostic) ----
def test_afternoon_overlay_is_kinematic_noop_on_moisture():
    m = _coast_map()
    sold = _synthetic_sold(m)
    Ws0, Wa0, P0, it0 = m.W_s.copy(), m.W_a.copy(), m.P.copy(), m.iter
    _ = scd.afternoon_precip_overlay(m, sold)
    assert np.array_equal(m.W_s, Ws0)
    assert np.array_equal(m.W_a, Wa0)
    assert np.array_equal(m.P, P0)
    assert m.iter == it0


# ---- 8. the cascade fails FAST on stage_c_diurnal_precip without bl_diurnal_wind (premise guard) ----
def test_cascade_requires_bl_diurnal_for_stage_c_diurnal():
    from flatworld import cascade
    with pytest.raises(ValueError, match="bl_diurnal_wind"):
        cascade.run_cascade("test", core="ad", lantern_day=True, bl_diurnal_wind=False,
                            stage_c_diurnal_precip=True, verbose=False)


# ============================ cascade integration (slow) ============================
# ---- 9. the cascade leaves the deliverable byte-identical when OFF + attaches diagnostics when ON ----
@pytest.mark.slow
def test_cascade_stage_c_diurnal_off_byte_identical_and_on_attaches():
    from flatworld import cascade, config
    base = cascade.run_cascade("test", core="ad", lantern_day=True, verbose=False)
    assert "P_phase" not in base["dC"]                            # default OFF: no diurnal-precip fields
    dcfg = config.bl_diurnal("test", max_days=15, max_seconds=120)
    scfg = config.stage_c_diurnal("test", max_days=15, max_seconds=120)
    on = cascade.run_cascade("test", core="ad", lantern_day=True, verbose=False,
                             bl_diurnal_wind=True, bl_diurnal_cfg=dcfg,
                             stage_c_diurnal_precip=True, stage_c_diurnal_cfg=scfg)
    dC = on["dC"]
    # the deliverable maps are UNTOUCHED -- the diurnal precip solve is additional diagnostics only
    assert np.array_equal(base["dC"]["P_mm_day"], dC["P_mm_day"])
    assert np.array_equal(base["dC"]["T_map_C"], dC["T_map_C"])
    assert np.array_equal(base["dC"]["u"], dC["u"])
    # diagnostics attached
    assert "P_phase" in dC and "stage_c_diurnal" in dC
    assert dC["P_phase"].shape[0] == dcfg.n_phase
    assert "total_rain_change_frac" in dC["stage_c_diurnal"]
