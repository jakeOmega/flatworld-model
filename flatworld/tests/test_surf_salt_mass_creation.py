"""Regression: ``surface_water_salt_step`` must CONSERVE total salt even when the P-E source drives
a near-dry cell's ``W_surf`` negative.

The bug (run11/run12 crust runaway to ~4e24 kg/m^2; branch claude/fix-surf-salt-mass-creation):
an offline eq_cycle held ``_surf_flash`` from climatology that exceeded the held ``_precip``, so
``_p_land = max(P,0) - surf_flash < 0`` pushed a near-dry cell's W below zero (the in-core
``p_land >= 0`` guarantee does not hold when the forcing is frozen). Then

  * salinity ``S_dis / max(W, W_SNAP_FLOOR)`` with W<0 clamps the denominator to 1e-30 -> ~1e22,
  * the creep/gravity riders advect salt at that poisoned salinity -> a huge conserving +/- dipole,
  * the salt snap floor (``S_dis < S_SNAP_FLOOR``, TRUE for large negatives) deletes the negative
    half -> net mass CREATION, which crusts into runaway ``S_dep``.

With no salt source active (edge-source + cryo pins OFF), a correct step conserves total salt to
round-off. This test drives the exact trigger and asserts conservation; it FAILS (salt created by
~1e12x) on the pre-fix code."""
import numpy as np
import pytest

from flatworld.stage_ad_core import (StageADConfig, StageADCore, surface_water_salt_step,
                                     repair_runaway_crust, SURF_SALT_SDEP_PHYS_MAX,
                                     SURF_SALT_SDEP_SATURATION)

DT = 86400.0


def _core(**kw):
    kw.setdefault("nx", 6)
    kw.setdefault("nz", 12)
    kw.setdefault("theta_max_deg", 5.0)
    kw.setdefault("prognostic_condensate", True)
    kw.setdefault("precip_reevap", True)
    kw.setdefault("surf_water_budget", True)
    kw.setdefault("surf_salt", True)
    kw.setdefault("surf_water_grav_flow", True)
    kw.setdefault("surf_water_edge_source", False)   # no boundary salt source -> conservation exact
    kw.setdefault("surf_water_cryo_pin", False)      # no cryo salt source
    kw.setdefault("surf_water_drift", 0.015)         # wind creep ON (the advecting rider)
    return StageADCore(StageADConfig(**kw))


def _total_salt(m):
    S = np.asarray(m.S_dis) + np.asarray(m.S_dep)
    return float(np.sum(S * m._dx_w))


def test_salt_conserved_when_p_minus_e_drives_cell_negative():
    m = _core()
    m.T_surf[...] = 250.0        # cold: qsat ~ 0, evaporation negligible
    m._precip[...] = 0.0
    m._evap[...] = 0.0
    m._surf_flash[...] = 0.0
    # near-dry salt-flat cells (W ~ 1e-7, above W_SNAP_FLOOR) carrying dissolved salt + seeded crust,
    # flanked by a little water so the creep has a live flux.
    W = np.array([1.0e-2, 1.0e-7, 1.0e-7, 1.0e-7, 1.0e-2, 1.0e-2])
    m.W_surf[...] = W
    m.S_dis[...] = 0.35 * W                       # dissolved salt at saturation salinity
    m.S_dep[...] = 5000.0                         # seeded halite crust
    # p_land = max(precip,0) - surf_flash = -surf_flash < 0 on the dry cells; *dt >> W -> W goes < 0.
    m._surf_flash[1:4] = 1.0e-9

    s0 = _total_salt(m)
    surface_water_salt_step(m, m.cfg, DT, np)
    s1 = _total_salt(m)

    # no salt source is active -> total salt is conserved to round-off. The bug creates ~1e12x.
    assert s1 == pytest.approx(s0, rel=1e-9), (
        f"total salt not conserved: {s0:.6e} -> {s1:.6e} (x{s1/max(s0,1e-300):.3e}); "
        f"max S_dep={float(np.max(m.S_dep)):.3e}")


# ============================================================ load-time guard: repair blown-up crust

def test_repair_resets_runaway_crust_to_saturation_and_accounts_removed_salt():
    # a state carrying a physically-impossible halite crust (the run11/run12 4e24 kg/m^2 runaway)
    dx_w = np.full(5, 2000.0)
    S_dep = np.array([5.0e3, 5.0e3, 4.28e24, 5.0e3, 5.0e3])   # cell 2 blown up
    repaired, removed, n = repair_runaway_crust(
        S_dep, dx_w, SURF_SALT_SDEP_PHYS_MAX, SURF_SALT_SDEP_SATURATION, np)
    assert n == 1
    # runaway cell RESET to the saturation scale (not the detection ceiling) -- a physical restart
    assert repaired[2] == pytest.approx(SURF_SALT_SDEP_SATURATION)
    assert float(np.max(repaired)) <= SURF_SALT_SDEP_SATURATION
    # removed salt is the excess above the saturation reset, area-weighted [kg/m]
    assert removed == pytest.approx((4.28e24 - SURF_SALT_SDEP_SATURATION) * 2000.0, rel=1e-12)
    # untouched (non-runaway) cells are byte-identical
    assert np.array_equal(repaired[[0, 1, 3, 4]], S_dep[[0, 1, 3, 4]])


def test_repair_is_noop_on_physical_crust():
    dx_w = np.full(4, 2000.0)
    # all physical: <= detection ceiling. A near-saturation 1.97e6 (> the 2e6-ish reset scale, but far
    # below the ceiling) must be LEFT ALONE -- the guard only resets cells above the detection ceiling.
    S_dep = np.array([0.0, 5.0e3, 1.97e6, 5.0e7])
    repaired, removed, n = repair_runaway_crust(
        S_dep, dx_w, SURF_SALT_SDEP_PHYS_MAX, SURF_SALT_SDEP_SATURATION, np)
    assert n == 0
    assert removed == 0.0
    assert np.array_equal(repaired, S_dep)                   # byte-identical, no reset


def test_detection_ceiling_and_saturation_scale_are_well_separated():
    # detection ceiling far above real crust (~2e6) and far below the runaway (>=1e17)
    assert 1.0e7 < SURF_SALT_SDEP_PHYS_MAX < 1.0e15
    # repair target is the saturation scale, well below the detection ceiling
    assert 1.0e5 < SURF_SALT_SDEP_SATURATION < 1.0e7
    assert SURF_SALT_SDEP_SATURATION < SURF_SALT_SDEP_PHYS_MAX


def test_load_state_self_heals_blown_up_checkpoint(capsys):
    # a resume from an ALREADY-corrupted checkpoint (run11/run12) must reset the runaway crust on load
    m = _core()
    m.T_surf[...] = 300.0
    st = m.state()                                   # a valid full state dict
    sdep = np.array(st["S_dep"], dtype=float)
    sdep[3] = 4.28e24                                # inject the run11/run12 runaway
    st["S_dep"] = sdep

    m2 = _core()
    m2.load_state(st)

    assert float(np.max(m2.S_dep)) <= SURF_SALT_SDEP_SATURATION
    assert float(np.asarray(m2.S_dep)[3]) == pytest.approx(SURF_SALT_SDEP_SATURATION)  # saturation reset
    assert "runaway-crust cell" in capsys.readouterr().out       # logged, not silent


def test_load_state_repair_leaves_salt_budget_consistent():
    # Finding 1: after a repair, the salt_residual gate must still close. The gate is
    # (S_total - S_total_reset) - cum_s_edge - cum_s_cryo + cum_s_snap; the guard re-pins the baseline
    # to the repaired total and zeroes the (corrupt-trajectory) salt cum-trackers -> residual ~ 0.
    m = _core(surf_water_edge_source=True, surf_water_cryo_pin=True)
    m.T_surf[...] = 300.0
    st = m.state()
    sdep = np.array(st["S_dep"], dtype=float)
    sdep[3] = 4.28e24
    st["S_dep"] = sdep
    # a real corrupt checkpoint carries nonzero salt cum-trackers + a stale (pre-repair) baseline
    st["cum_s_edge"], st["cum_s_cryo"], st["cum_s_snap"] = 1.234e5, 5.6e3, -7.8e4
    st["S_total_reset"] = float(np.sum((np.asarray(st["S_dis"]) + sdep) * m._dx_w))

    m2 = _core(surf_water_edge_source=True, surf_water_cryo_pin=True)
    m2.load_state(st)

    S_total = float(np.sum((np.asarray(m2.S_dis) + np.asarray(m2.S_dep)) * m2._dx_w))
    salt_residual = (S_total - m2._S_total_reset) - m2._cum_s_edge - m2._cum_s_cryo + m2._cum_s_snap
    assert salt_residual == pytest.approx(0.0, abs=1e-6 * max(abs(S_total), 1.0))
    assert m2._cum_s_edge == 0.0 and m2._cum_s_cryo == 0.0 and m2._cum_s_snap == 0.0


def test_load_state_clean_checkpoint_untouched(capsys):
    m = _core()
    m.T_surf[...] = 300.0
    m.S_dep[...] = 1.5e3                              # physical crust
    st = m.state()
    sdep_before = np.array(st["S_dep"], dtype=float)
    cum_before = (st.get("cum_s_edge", 0.0), st.get("cum_s_cryo", 0.0), st.get("cum_s_snap", 0.0))

    m2 = _core()
    m2.load_state(st)

    assert np.array_equal(np.asarray(m2.S_dep), sdep_before)   # byte-identical, no reset
    # a clean load preserves the loaded salt cum-trackers (the fresh-window zeroing is repair-only)
    assert (m2._cum_s_edge, m2._cum_s_cryo, m2._cum_s_snap) == cum_before
    assert "GUARD: repaired" not in capsys.readouterr().out
