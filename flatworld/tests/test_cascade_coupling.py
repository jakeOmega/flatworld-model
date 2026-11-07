"""Issue-A Part 3 test: the A<->B fixed-point loop wires B's moisture+transport into A correctly.

Exercises the coupling MECHANICS (the individual units are TDD'd in test_stage_b / test_stage_a_injection):
A is switched to injected transport, B's wall-zeroed flux is installed each outer iteration, the single
overturning cell survives the coupling, and the ribbon-SST drift is tracked across iterations. Tiny
configs + a max_seconds guard keep it fast; it asserts structure that holds regardless of full spin-up.
"""
import numpy as np
import pytest
from flatworld.cascade import (couple_stage_a_b, _ascent_weight, cell_ascent_gate,
                               _subsidence_weight, subsidence_gate, land_surface_diurnal)


# ---- land diurnal-cycle backbone (Issue #21): the cascade's surface.py-reused diurnal solve ----
# CHARACTERIZATION TESTS -- read before "fixing" a failure here. Both diurnal tests below pin an EMERGENT
# physical result (the heat-capacity swing law and the seasonless warmer-interior finding) that falls out of
# the surface energy balance, NOT an invariant. If a code change breaks one: STOP and sanity-check the change
# -- but do NOT assume the change must therefore be wrong, and NEVER tune the model, or loosen/retarget a
# threshold, just to make it pass. Real physics (a new mechanism, a fixed bug, a different forcing or constant)
# can LEGITIMATELY move the sign/magnitude; if so, re-derive the expected value and UPDATE the test to the new
# physics with a note on why it moved. The thresholds are already deliberately loose (these magnitudes, off a
# tiny partially-spun-up A, are SOFT) -- they guard the SIGN/order-of-magnitude so a genuine shift is VISIBLE,
# not so a number gets frozen and the model bent back to it.
@pytest.fixture(scope="module")
def land_diurnal_endpoints():
    """A tiny, partially-spun-up Stage A column run through land_surface_diurnal at BOTH continentality
    endpoints -- shared by the diurnal-backbone tests so the 600-step spin-up + surface solves run once.
    Returns (theta_deg, (Tmean,Tmin,Tmax)_coast, (Tmean,Tmin,Tmax)_interior). A partially-spun-up tiny A
    suffices: the helper just needs a plausible downwelling-LW / air-temperature column; the diurnal
    contrasts are set by surface heat capacity, not A's exact convergence."""
    from flatworld import config as cfgmod, constants as C
    from flatworld.stage_a_column import StageA
    A = StageA(cfgmod.stage_a("test", nx=20, nz=12, max_seconds=20, tol=3e-3, check_every=100))
    for _ in range(600):
        A.step()
    th = np.linspace(32, 56, 13)
    coast = land_surface_diurnal(A, th, wetness=0.40, C_surf=C.C_MOIST_LAND)
    interior = land_surface_diurnal(A, th, wetness=0.10, C_surf=C.C_ARID_LAND)
    return th, coast, interior


def test_land_surface_diurnal_atm_override_drives_the_land_balance():
    """The AD-core land-T fix: passing ``atm=(D_surf_of_theta, Tair_of_theta)`` drives the land surface
    balance off THAT atmosphere instead of a Stage A column (stage_a may be None). This is how the fused-core
    cascade keeps land on the resolved-circulation gradient rather than Stage A's greenhouse-collapse cliff.
    Pins: (1) the override works with no Stage A; (2) a warmer atmosphere yields warmer land everywhere
    (the override genuinely sets the balance); (3) a smooth atmosphere yields a smooth land profile -- no
    cliff injected by the plumbing (the bug was Stage A's collapse cliff crushing the band to ~1 deg)."""
    from flatworld import constants as C
    th = np.linspace(40, 54, 8)
    warm = (lambda t: 380.0, lambda t: 300.0)        # (D_surf [W/m^2], T_air [K]) -- warm, bright atmosphere
    cold = (lambda t: 250.0, lambda t: 270.0)        # 30 K colder air, much less downwelling LW
    Tw = land_surface_diurnal(None, th, wetness=0.10, C_surf=C.C_ARID_LAND, atm=warm)[0]
    Tc = land_surface_diurnal(None, th, wetness=0.10, C_surf=C.C_ARID_LAND, atm=cold)[0]
    assert np.all(np.isfinite(Tw)) and np.all(np.isfinite(Tc))   # (1) runs with stage_a=None
    assert np.all(Tw > Tc)                                       # (2) warmer atmosphere -> warmer land
    assert np.max(np.abs(np.diff(Tw))) < 25.0                    # (3) smooth atm -> smooth land (no cliff)


def test_land_surface_diurnal_arid_swings_more_than_moist(land_diurnal_endpoints):
    # CHARACTERIZATION (see the section note above -- careful, don't tune-to-pass; a real physics change may
    # legitimately move this).
    # land_surface_diurnal runs Stage A's column through surface.solve_periodic_surface at two continentality
    # endpoints. The arid interior (C_ARID_LAND, low thermal inertia) must swing MORE than the damp coast
    # (C_MOIST_LAND) at every latitude, with T_min <= T_mean <= T_max -- the heat-capacity law the Stage C
    # diurnal field is built on.
    _, (mc, nc, xc), (mi, ni, xi) = land_diurnal_endpoints
    assert np.all(nc <= mc + 1e-9) and np.all(mc <= xc + 1e-9)        # ordering, coast
    assert np.all(ni <= mi + 1e-9) and np.all(mi <= xi + 1e-9)        # ordering, interior
    assert np.all((xi - ni) >= (xc - nc))                            # arid swings >= moist EVERYWHERE
    assert (xi - ni).mean() > (xc - nc).mean() + 5.0                 # and materially more on average


def test_cascade_interior_daymean_warmer_than_coast_in_ribbon(land_diurnal_endpoints):
    # CHARACTERIZATION (see the section note above -- careful, don't tune-to-pass; a real physics change may
    # legitimately move this).
    # Characterization of the Issue #21 finding (guards the CASCADE deliverable, which the standalone
    # no-regression test cannot see): on a SEASONLESS world the surface-EB-solved arid INTERIOR runs WARMER
    # in the day-MEAN than the damp coast across the habitable ribbon -- the OPPOSITE sign of the legacy
    # `continentality_K` interior *cooling* the cascade now supersedes. The standalone test
    # (test_diurnal_daymean_is_legacy_continentality_model_standalone) only pins the standalone DEFAULT
    # (coast - continentality_K); this pins the solved cascade values land_surface_diurnal actually feeds
    # Stage C as Tland_interior_C_of_theta. Magnitudes here are SOFT (a tiny, partially-spun A) -- the test
    # guards the SIGN and an order-of-magnitude, deliberately loose so it tracks the physics, not a number.
    from flatworld.stage_c_map import StageCConfig
    th, (mc, nc, xc), (mi, ni, xi) = land_diurnal_endpoints
    rib = (th >= 45) & (th <= 51)                                     # the habitable ribbon band (~46-50 deg)
    assert rib.sum() >= 3                                             # several ribbon points, not a single one
    # SIGN (the finding): interior day-mean is WARMER than coast across the WHOLE ribbon, never cooler.
    assert np.all(mi[rib] >= mc[rib] - 1e-6)
    assert (mi[rib] - mc[rib]).mean() > 3.0                           # and materially warmer (measured ~+6 K)
    # ... by the OPPOSITE sign of the legacy continentality_K COOLING: the solved interior sits well ABOVE the
    # legacy `coast - continentality_K`, so carrying the legacy cooling would have the interior too cold.
    cont_K = StageCConfig().continentality_K
    assert (mi[rib] - (mc[rib] - cont_K)).mean() > cont_K            # solved interior > legacy-cooled by > cont_K
    # the continental signal is carried as the SWING, not a mean cooling: interior swing >> coast swing here.
    assert ((xi - ni) - (xc - nc))[rib].mean() > 15.0                # measured ~+29 K (interior ~42 vs coast ~13)


def test_ascent_weight_gates_loft_to_rising_branches():
    # a realistic single-cell w(theta) (fine grid like Stage B's nx=120): big equatorial updraft (theta<15),
    # subsidence, a secondary band ascent (theta~37-44), subsidence again. The gate must be ~1 in the
    # ascending band branch and 0 in subsidence, with the equatorial spike NOT setting the in-domain scale
    # (normalized poleward of 20 -- on the real grid w is already negative by theta~18).
    th = np.linspace(0, 60, 121)
    w_mid = np.where(th < 15, 1.9 * np.cos(np.radians(6 * th)),               # equatorial updraft
                     np.where((th > 36) & (th < 45), 0.18, -0.1))            # band ascent, else subsidence
    a = _ascent_weight(th, w_mid, theta_norm_min=20.0)
    assert np.all((a >= 0) & (a <= 1))
    assert a[np.argmin(np.abs(th - 40))] > 0.8            # rising band branch -> lofting on
    assert a[np.argmin(np.abs(th - 30))] == 0.0           # inner-sea subsidence -> no lofting
    assert a[np.argmin(np.abs(th - 50))] == 0.0           # ribbon subsidence -> no lofting


def test_cell_ascent_gate_callable_zero_in_subsidence():
    th = np.linspace(0, 60, 40)
    nz = 11
    w = np.zeros((nz, th.size))
    w[nz // 2] = np.where((th > 36) & (th < 45), 0.2, -0.1)   # ascend only in the band
    w[nz // 2][th < 12] = 1.5                                  # equatorial updraft (out of Stage-C domain)
    gate = cell_ascent_gate(dict(theta_deg=th, w=w))
    assert float(gate(40.0)) > 0.5                            # band branch lofts
    assert float(gate(30.0)) == 0.0 and float(gate(50.0)) == 0.0   # subsidence does not


# ---- subsidence gate on the convective DRAIN (Refinement 1: drain sibling of the loft gate) ----
def test_subsidence_weight_keys_on_sign_of_w_above_noise_floor():
    # the drain gate is a pure allowed/suppressed gate on w's sign ABOVE a relative noise floor
    # (1% of max|w|, the 2026-06-11 fix): weak-but-GENUINE band ascent (>=1% of the cell scale) gets
    # full weight, dead-zone wiggle (<1%) does NOT (the converged AD-core's standing zones sit at
    # ~0.5-0.7% and previously opened the drain over the rainless warm-edge sea -- the seed-137 bug).
    th = np.linspace(0, 60, 121)
    w_mid = np.where(th < 15, 1.9 * np.cos(np.radians(6 * th)),               # equatorial updraft (w>0)
                     np.where((th > 36) & (th < 45), 0.02, -0.5))            # weak band ascent, strong subsidence
    w_mid = np.where((th >= 50) & (th < 56), 0.012, w_mid)                   # dead-zone wiggle ~0.6% of max
    s = _subsidence_weight(th, w_mid)
    assert set(np.unique(s)) <= {0.0, 1.0}                    # still a pure gate (no magnitude weighting)
    assert s[np.argmin(np.abs(th - 40))] == 1.0              # genuine weak ascent (1.05% of max) -> allowed
    assert s[np.argmin(np.abs(th - 30))] == 0.0              # inner-sea subsidence -> drain suppressed
    assert s[np.argmin(np.abs(th - 52))] == 0.0              # NOISE-scale positive wiggle -> suppressed


def test_subsidence_weight_one_percent_floor_is_the_exact_boundary():
    # pin the 1%-of-max|w| floor EXACTLY. It applies on BOTH the AD core AND the legacy Stage-B path,
    # so genuine-but-weak ascent JUST below 1% of max|w| is now gated off there too (the 2026-06-11
    # fix) -- intended. Plateaus (not spikes) so the 3-point smoothing preserves the interior probe.
    th = np.linspace(0.0, 60.0, 121)
    w = np.full_like(th, -0.5)                                   # subsidence background
    w[(th >= 8.0) & (th <= 14.0)] = 1.0                          # wide max plateau -> max|w|=1, floor=0.01
    w[(th >= 20.0) & (th <= 26.0)] = 0.0102                      # 1.02% of max -> allowed
    w[(th >= 40.0) & (th <= 46.0)] = 0.0098                      # 0.98% of max -> gated off
    s = _subsidence_weight(th, w)
    assert set(np.unique(s)) <= {0.0, 1.0}                       # still a pure gate
    assert s[np.argmin(np.abs(th - 23.0))] == 1.0               # just above the floor -> ascent allowed
    assert s[np.argmin(np.abs(th - 43.0))] == 0.0               # just below the floor -> gated (B path too)


def test_subsidence_gate_callable_one_in_ascent_zero_in_subsidence():
    th = np.linspace(0, 60, 40)
    nz = 11
    w = np.zeros((nz, th.size))
    w[nz // 2] = np.where((th > 36) & (th < 45), 0.2, -0.1)   # ascend only in the band
    w[nz // 2][th < 12] = 1.5                                  # equatorial updraft (out of Stage-C domain)
    gate = subsidence_gate(dict(theta_deg=th, w=w))
    assert float(gate(40.0)) == 1.0                          # band branch: drain allowed
    assert float(gate(30.0)) == 0.0 and float(gate(50.0)) == 0.0   # subsidence: drain suppressed


def test_lantern_day_preserves_day_mean_but_changes_diurnal_swing():
    """End-to-end through cascade.land_surface_diurnal (the hook run_cascade(lantern_day=True) drives):
    the lantern painted day must preserve every latitude's day-MEAN flux exactly (the zonal climate
    backbone is untouched) while CHANGING the diurnal swing (the 16-h band day vs the natural passage).
    A tiny partially-spun Stage A suffices -- the day-mean preservation is exact, the swing is set by
    the surface solve."""
    from flatworld import config as cfgmod, constants as C, geometry as g, lantern
    from flatworld.stage_a_column import StageA
    A = StageA(cfgmod.stage_a("test", nx=20, nz=12, max_seconds=20, tol=3e-3, check_every=100))
    for _ in range(600):
        A.step()
    th = np.linspace(34.0, 54.0, 9)
    S_of = lambda thd: g.day_mean_flux(g.x_of_theta(np.radians(np.asarray(thd, float))))  # natural day-mean
    day_cycle = lambda x: lantern.day_flux_cycle(x, S_of)
    # the LOCKED-premise invariant: the lantern day-mean flux EQUALS the natural day-mean at every theta
    for thd in th:
        t, F = lantern.day_flux_cycle(float(g.x_of_theta(np.radians(thd))), S_of)
        assert abs(float(np.trapezoid(F, t)) / (t[-1] - t[0]) - float(S_of(thd))) < 1.0
    nat = land_surface_diurnal(A, th, wetness=0.10, C_surf=C.C_ARID_LAND)
    lan = land_surface_diurnal(A, th, wetness=0.10, C_surf=C.C_ARID_LAND, day_cycle=day_cycle)
    nmean, nmin, nmax = nat
    lmean, lmin, lmax = lan
    # backbone: the surface DAY-MEAN is ~unchanged (residual = the Jensen sigma T^4 nonlinearity only)
    assert np.max(np.abs(lmean - nmean)) < 4.0
    # diurnal swing CHANGES (the painted 16-h band day is not the natural multi-sun passage)
    assert not np.allclose(lmax - lmin, nmax - nmin, atol=1.0)


def test_couple_a_b_injects_b_flux_and_preserves_single_cell():
    res = couple_stage_a_b(
        "test", verbose=False, max_outer=2, outer_tol=1e-9,           # tiny tol -> run both outer iters
        a_overrides=dict(nx=20, nz=14, max_seconds=40, tol=2e-3, check_every=100),
        b_overrides=dict(nx=50, nz=19, max_seconds=25, tol=2e-3, check_every=200))
    A = res["A"]
    # A is now driven by B's injected flux divergence, not its own diffusion
    assert A.cfg.transport_mode == "injected"
    assert A._injected_F is not None
    assert A._injected_F[0] == 0.0 and A._injected_F[-1] == 0.0       # wall-zeroed (energy closure)
    assert np.max(np.abs(A._injected_F)) > 0.0                        # a real transport was installed
    # the single overturning cell survives the moisture+transport coupling
    assert res["dB"]["surf_sign_changes"] == 0
    # the fiducial anchor was set from A's diffusive baseline (positive W/m)
    assert res["fiducial"] > 0.0
    # the outer loop recorded a ribbon-SST drift per iteration
    assert len(res["drift_hist"]) == 2
