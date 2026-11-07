"""Tests for the DIURNAL SUN coupling (COSMO_DIURNAL_SUN; DIURNAL_COUPLING_SCOPING.md chunks 4.1/4.2).

Default-OFF gates: the flag OFF must leave every stepping input the IDENTICAL day-mean object (the
byte-identity merge gate is additionally proven at the shell level -- a 40-step coupled-state sha256 at
the base commit vs the branch, recorded in flatworld/DIURNAL_RESULTS.md). ON gates: the one-day table's
phase-mean must equal the static day-mean forcing exactly (the renormalization contract), a probe
column's cycle must match ``geometry.diurnal_flux_cycle``, the in-loop radiation must actually receive
the phase rows, and the physical diurnal-skin heat capacity must engage. All on the DEFAULT (test)
world -- no canonical env."""
import numpy as np
import pytest

from flatworld import constants as C
from flatworld import geometry as g
from flatworld import radiation as rad
from flatworld.stage_ad_core import StageADCore, StageADConfig

NPH = 48                                     # coarse test table (production 1440); divides the day evenly


def _core(diurnal=True, **kw):
    kw.setdefault("nx", 24)
    kw.setdefault("nz", 12)
    kw.setdefault("theta_max_deg", 70.0)
    return StageADCore(StageADConfig(diurnal_sun=diurnal, diurnal_nphase=NPH, **kw))


# ------------------------------------------------------------------ table construction (4.1)
def test_table_phase_mean_equals_daymean_exactly():
    """The renormalization contract: mean over phase rows == self.S (and S_diffuse) to rounding, so the
    diurnal run's time-mean SW forcing is the day-mean run's forcing BY CONSTRUCTION."""
    m = _core()
    assert m._S_table.shape == (NPH, m.cfg.nx)
    np.testing.assert_allclose(m._S_table.mean(axis=0), m.S, rtol=1e-12)
    assert m._S_diff_table is not None                      # sw_diffuse defaults ON
    np.testing.assert_allclose(m._S_diff_table.mean(axis=0), m.S_diffuse, rtol=1e-12)


def test_raw_table_truncation_small():
    """The UN-renormalized table's phase-mean must already sit within quadrature truncation of the
    analytic day-mean (finite n_suns + midpoint sampling; geometry's own self-consistency check runs at
    1e-3) -- i.e. the renormalization absorbs a sub-percent correction, not a wrong shape."""
    m = _core()
    raw = g.diurnal_flux_table(np.asarray(m.x, float), n_phase=NPH)
    rel = np.abs(raw.mean(axis=0) - m.S) / np.maximum(m.S, 1e-12)
    assert float(rel.max()) < 5e-3


def test_probe_column_matches_diurnal_flux_cycle():
    """A probe column's table cycle matches ``diurnal_flux_cycle`` (the established instantaneous
    machinery) interpolated to the row midpoints, up to the per-column renormalization scale (~1)."""
    m = _core()
    j = m.cfg.nx // 2
    x_j = float(np.asarray(m.x)[j])
    t_c, S_c = g.diurnal_flux_cycle(x_j, n_steps=4 * NPH)
    t_mid = (np.arange(NPH) + 0.5) * (g.CLOCK.T_day / NPH)
    S_ref = np.interp(t_mid, t_c, S_c) * m.cfg.i0_scale
    col = m._S_table[:, j]
    scale = float(m.S[j] / (S_ref.mean() + 1e-30))          # the renormalization the core applied
    day = col > 1e-3 * col.max()                            # compare in daylight (night is a hard zero)
    np.testing.assert_allclose(col[day], (S_ref * scale)[day], rtol=2e-2)


def test_night_is_dark_with_collimator():
    """With the default-ON collimator the geometric night is a hard zero in the instantaneous table --
    the rectification driver (the pre-flight's 290-C-swing physics) is present, not smoothed away."""
    m = _core()
    j = m.cfg.nx // 2
    col = m._S_table[:, j]
    assert float(col.min()) == 0.0                          # geometric night: a hard zero
    assert float(col.max()) > 5.0 * float(col.mean())       # noon well above the day-mean (the swing driver)


# ------------------------------------------------------------------ OFF path (the default-OFF gate)
def test_off_radiate_receives_identical_daymean_objects(monkeypatch):
    """Flag OFF: the in-loop radiation receives self.S / self.S_diffuse THEMSELVES (identity, not just
    equality) whatever sw_inst says -- the code path is the verbatim shipped one."""
    m = _core(diurnal=False)
    seen = {}
    orig = rad.column_radiation

    def spy(p_edge, T, q, T_s, S_toa, alb, **kw):
        seen["S"] = S_toa
        seen["S_diffuse"] = kw.get("S_diffuse")
        return orig(p_edge, T, q, T_s, S_toa, alb, **kw)

    monkeypatch.setattr(rad, "column_radiation", spy)
    m.step()
    assert seen["S"] is m.S
    assert seen["S_diffuse"] is m.S_diffuse
    assert m._S_table is None and m._S_now is None          # no tables built when OFF


def test_off_heat_capacity_is_spinup_blend():
    m = _core(diurnal=False)
    Cs = np.asarray(m._surf_heat_capacity())
    C_ocean = C.slab_heat_capacity(m.cfg.ocean_depth_spinup, C.RHO_LIQ, C.CP_LIQ)
    C_rock = C.slab_heat_capacity(1.0, 2600.0, 800.0)
    wet = np.asarray(m._wetness())
    np.testing.assert_allclose(Cs, C_ocean * wet + C_rock * (1 - wet), rtol=1e-12)


# ------------------------------------------------------------------ ON path (4.1 stepping + 4.2 capacity)
def test_on_inloop_radiation_receives_phase_rows(monkeypatch):
    """Flag ON: the in-loop radiation receives the held row buffer (the row indexed by start-of-step
    sim_time), while a DIAGNOSTIC radiation call still receives the day-mean self.S."""
    m = _core()
    m.run_dynamics = False                                  # fixed dt = cfg.dt -> deterministic phase walk
    m.cfg.dt = g.CLOCK.T_day / NPH                          # one table row per step
    seen = []
    orig = rad.column_radiation

    def spy(p_edge, T, q, T_s, S_toa, alb, **kw):
        seen.append(S_toa)
        return orig(p_edge, T, q, T_s, S_toa, alb, **kw)

    monkeypatch.setattr(rad, "column_radiation", spy)
    for k in range(4):
        m.step()
        np.testing.assert_array_equal(np.asarray(seen[-1]), m._S_table[k])
        assert seen[-1] is m._S_now
    seen.clear()
    m.diagnostics()                                         # diagnostics stay DAY-MEAN (sw_inst=False)
    assert any(s is m.S for s in seen)
    assert not any(s is m._S_now for s in seen)


def test_on_time_mean_forcing_over_day_equals_daymean():
    """Walking every row once (one whole day at one row per step) delivers a time-mean SW input equal to
    the static day-mean -- the unit-level statement of gate (i) in the scoping doc."""
    m = _core()
    m.run_dynamics = False
    m.cfg.dt = g.CLOCK.T_day / NPH
    acc = np.zeros(m.cfg.nx)
    for _ in range(NPH):
        m.step()
        acc += np.asarray(m._S_now)
    np.testing.assert_allclose(acc / NPH, m.S, rtol=1e-12)


def test_on_heat_capacity_is_physical_skin():
    m = _core()
    Cs = np.asarray(m._surf_heat_capacity())
    wet = np.asarray(m._wetness())
    np.testing.assert_allclose(Cs, C.C_OCEAN_MIXED * wet + C.C_BARE_ROCK * (1 - wet), rtol=1e-12)


def test_phase_reseed_on_load_state_and_driver_hook():
    """load_state re-pins the held row to the RESTORED sim_time; set_diurnal_phase_index (the captured-
    replay driver hook) pins an explicit row, wrapping mod n_phase."""
    m = _core()
    st = {k: np.asarray(getattr(m, k)).copy() for k in ("T_surf", "T_atm", "q", "q_c", "omega", "psi", "b")}
    st["sim_time"] = 3.0 * g.CLOCK.T_day + 2.5 * (g.CLOCK.T_day / NPH)   # 3 whole days + 2.5 rows -> row 2
    m.load_state(st)
    np.testing.assert_array_equal(np.asarray(m._S_now), m._S_table[2])
    m.set_diurnal_phase_index(NPH + 5)                      # wraps
    np.testing.assert_array_equal(np.asarray(m._S_now), m._S_table[5])


# ------------------------------------------------------------------ construction contracts
def test_capture_construction_has_no_row_per_step_contract():
    """PR #301 review finding 1: the phase rides a device TIME clock (accumulated by the realized dt),
    so there is deliberately NO n_phase*dt == T_day requirement -- a capture_ready core with any
    (n_phase, dt_dyn) pairing constructs fine, and the adaptive dt is exact by construction."""
    m = StageADCore(StageADConfig(nx=8, nz=8, capture_ready=True, diurnal_sun=True,
                                  diurnal_nphase=1440, dt_dyn=300.0))   # mismatched pairing: fine now
    assert m._S_table.shape == (1440, 8)
    assert m._diurnal_time_dev is None                      # the device clock exists only on-device


def test_dt_rad_hold_warns():
    with pytest.warns(UserWarning, match="dt_rad"):
        _core(dt_rad=3600.0)


def test_adaptive_dt_half_row_repetition(monkeypatch):
    """Finding-1 semantics pin: with dt = HALF a table row (the adaptive-dt stand-in), each row must be
    served exactly twice -- the phase follows TRUE time (floor(sim_time/row_dur)), never the step count."""
    m = _core()
    m.run_dynamics = False
    m.cfg.dt = (g.CLOCK.T_day / NPH) / 2.0                  # half a row per step
    seen = []
    orig = rad.column_radiation

    def spy(p_edge, T, q, T_s, S_toa, alb, **kw):
        seen.append(np.array(S_toa))                        # value copy (the buffer is reused in place)
        return orig(p_edge, T, q, T_s, S_toa, alb, **kw)

    monkeypatch.setattr(rad, "column_radiation", spy)
    for k in range(12):
        m.step()
        np.testing.assert_array_equal(seen[-1], m._S_table[(k // 2) % NPH])


# ------------------------------------------------------------------ optics worlds (finding 4: the renorm
# is load-bearing under the canonical shade / the lens; the module globals are read at call time and the
# day-mean caches key on them, so monkeypatching them retrofits the world without an import-scope env)
def test_shade_world_phase_mean(monkeypatch):
    """Canonical graded-shade world (m33_runs/shade_D4_bright.npz): the renormalized table's phase-mean
    must still equal the day-mean forcing exactly, with sane (order-1) renormalization gains."""
    import os
    shade_path = os.path.join(os.path.dirname(__file__), "..", "..", "m33_runs", "shade_D4_bright.npz")
    sd = np.load(shade_path)
    monkeypatch.setattr(g, "_SHADE", (np.asarray(sd["alpha_deg"], float), np.asarray(sd["T_eff"], float)))
    m = _core()
    np.testing.assert_allclose(m._S_table.mean(axis=0), m.S, rtol=1e-12)
    raw = g.diurnal_flux_table(np.asarray(m.x, float), n_phase=NPH)
    scale = m.S / np.maximum(raw.mean(axis=0), 1e-30)
    assert float(scale.max()) < 2.0 and float(scale.min()) > 0.5   # shade-only: renorm stays a truncation fix


def test_lens_world_phase_mean(monkeypatch):
    """COSMO_LENS_RESHAPE world (the phase-uniform-gain approximation): phase-mean == day-mean must hold
    exactly, the equatorial pool must show a genuine renormalization GAIN (the lens moved day-mean flux
    the 1-D instantaneous shape cannot know about), and no column may be starved to a zero cycle."""
    monkeypatch.setattr(g, "_LENS_RESHAPE", True)
    m = _core(nx=96)                                        # resolve the 2-deg pool (nx=24 starts at 3.3 deg
    #                                                         and the reshape passes a pool-less grid through)
    np.testing.assert_allclose(m._S_table.mean(axis=0), m.S, rtol=1e-12)
    raw = g.diurnal_flux_table(np.asarray(m.x, float), n_phase=NPH)
    scale = m.S / np.maximum(raw.mean(axis=0), 1e-30)
    assert float(scale[0]) > 1.5                            # equator pool: lens gain > 1 (canonical ~x8.8)
    assert not np.any((m.S > 1e-6) & (m._S_table.max(axis=0) <= 0.0))   # no forced-but-dark columns


# ------------------------------------------------------------------ slow wetness keying (finding 2)
def test_wetness_keys_on_slow_T():
    """Diurnal ON: the classifier ramp reads the SLOW EMA temperature, not tonight's skin -- a hot flat
    (T_slow above SST_DRY) stays DRY through a cold night (T_surf below SST_WET_HI)."""
    m = _core()
    m._T_surf_slow[...] = C.T_FREEZE + 100.0                # day-scale classification: hot dry flat
    m.T_surf[...] = C.T_FREEZE + 20.0                       # tonight's skin: cold
    assert float(np.max(m._wetness())) == 0.0               # stays dry (no nightly re-wetting)
    off = _core(diurnal=False)
    off.T_surf[...] = C.T_FREEZE + 20.0
    assert float(np.min(off._wetness())) == 1.0             # OFF keeps the instantaneous keying


def test_wetness_construction_fallback_keys_on_slow_T():
    """The "construction / first radiation, no state yet" fallback (``precip is None``) must ALSO key
    on the slow EMA ``T_key``, not tonight's instantaneous skin -- a bad merge-conflict resolution (PR
    #301 into main) regressed exactly this branch back to ``self.T_surf`` while the sibling branch two
    lines above it correctly kept ``T_key``. Requires ``budget_wet_fraction`` so the function reaches
    the ``precip is None`` guard instead of returning at the ramp-only branch above it."""
    m = _core(budget_wet_fraction=True)
    m._T_surf_slow[...] = C.T_FREEZE + 100.0                # day-scale classification: hot dry flat
    m.T_surf[...] = C.T_FREEZE + 20.0                       # tonight's skin: cold
    m._precip = None                                        # force the "no state yet" fallback branch
    assert float(np.max(m._wetness())) == 0.0               # must stay dry (keyed on T_key, not T_surf)


def test_wetness_freeze_pins_seed_classification():
    m = _core(diurnal_wetness_freeze=True)
    frozen = np.array(m._wetness_frozen)
    m.T_surf[...] = C.T_FREEZE + 20.0                       # move the skin far from the seed state
    np.testing.assert_array_equal(np.asarray(m._wetness()), frozen)


def test_slow_T_ema_relaxes_with_dt_over_tau():
    """The end-of-step EMA update: T_slow' == T_slow + (dt/tau)(T_surf_end - T_slow), realized-dt weighted."""
    m = _core()
    m.run_dynamics = False
    m.cfg.dt = 3600.0
    T_slow0 = np.array(m._T_surf_slow)
    m.step()
    expect = T_slow0 + (3600.0 / m.cfg.diurnal_wetness_tau) * (np.asarray(m.T_surf) - T_slow0)
    np.testing.assert_allclose(np.asarray(m._T_surf_slow), expect, rtol=1e-12)


def test_heat_capacity_keys_on_slow_wetness():
    """4.2 blend inherits the slow keying: a hot-classified flat keeps C_BARE_ROCK through a cold night
    (the review's ~130x night heat-capacity inflation is gone)."""
    m = _core()
    m._T_surf_slow[...] = C.T_FREEZE + 100.0
    m.T_surf[...] = C.T_FREEZE + 20.0
    np.testing.assert_allclose(np.asarray(m._surf_heat_capacity()), C.C_BARE_ROCK, rtol=1e-12)


# ------------------------------------------------------------------ checkpoint clock continuity (audit F4)
def test_state_carries_diurnal_clock_and_load_repins():
    """state() persists the TRUE diurnal time (the device clock when the capture path holds one --
    host sim_time is frozen under replay -- else sim_time), and load_state re-pins the phase from it:
    a resumed run keeps sun-forcing CONTINUITY instead of snapping to a driver's nominal step phase."""
    m = _core()
    m.sim_time = 3.25 * g.CLOCK.T_day
    st = m.state()
    assert st["diurnal_clock"] == pytest.approx(m.sim_time)
    # a (host-simulated) device clock takes precedence over sim_time (the captured-path truth)
    t_dev = (7 + 0.5) * m._diurnal_dt_phase
    m._diurnal_time_dev = np.array([t_dev])
    st2 = m.state()
    assert st2["diurnal_clock"] == pytest.approx(t_dev)
    # load: the phase pin follows the SAVED clock, not the checkpoint's (stale-under-capture) sim_time
    m2 = _core()
    m2._diurnal_time_dev = np.zeros(1)                     # simulate the captured-path clock buffer
    m2.load_state(st2)
    np.testing.assert_array_equal(np.asarray(m2._S_now), m2._S_table[7])
    assert float(m2._diurnal_time_dev[0]) == pytest.approx(t_dev)


def test_load_state_without_clock_falls_back_to_sim_time():
    """A pre-clock (old-format) checkpoint re-pins from the restored sim_time -- the prior behavior."""
    m = _core()
    st = m.state()
    st.pop("diurnal_clock")
    st["sim_time"] = 5.5 * m._diurnal_dt_phase
    m2 = _core()
    m2.load_state(st)
    np.testing.assert_array_equal(np.asarray(m2._S_now), m2._S_table[5])
    assert "diurnal_clock" not in st


def test_off_state_has_no_diurnal_clock_key():
    """Flag OFF: the checkpoint format is unchanged (no diurnal keys) -- the byte-identity contract."""
    m = _core(diurnal=False)
    assert "diurnal_clock" not in m.state()
