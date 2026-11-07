"""Tests for flatworld.surface (diurnally-resolved surface energy balance).

The physics the task insists on: surface heat capacity sets the diurnal swing, and because sigma T^4
is nonlinear the time-mean temperature falls below the radiative-mean as the swing grows (Jensen),
which is why the band can be misplaced if the diurnal cycle is ignored.
"""
import numpy as np
import pytest
from flatworld import geometry as g, constants as C, surface as S


def _sw_cycle(theta_deg=50.0, n=288, trans=0.85, albedo=0.1):
    x = float(g.x_of_theta(np.radians(theta_deg)))
    t, S_inst = g.diurnal_flux_cycle(x, n_steps=n, n_suns=8)
    return t, trans * (1 - albedo) * S_inst


def test_heat_capacity_controls_swing():
    t, SW = _sw_cycle()
    swings = []
    for Cs in (C.C_BARE_ROCK, C.C_MOIST_LAND, C.C_OCEAN_MIXED):
        r = S.solve_periodic_surface(t, SW, 250.0, 280.0, 4e-3, C.P_SURF, 5.0,
                                     C_surf=Cs, wetness=0.0, max_days=60)
        swings.append(r["Ts_range"])
    # bigger heat capacity -> smaller diurnal swing, strictly
    assert swings[0] > swings[1] > swings[2]
    assert swings[2] < 1.0          # ocean barely stirs
    assert swings[0] > 30.0         # bare rock lurches


def test_jensen_gap_grows_as_capacity_drops():
    # time-mean Ts <= (<sigT4>/sigma)^{1/4}, with the gap larger for the low-capacity surface
    t, SW = _sw_cycle()
    def gap(Cs):
        r = S.solve_periodic_surface(t, SW, 250.0, 280.0, 4e-3, C.P_SURF, 5.0,
                                     C_surf=Cs, wetness=0.0, max_days=120)
        return (r["sigT4_mean"] / C.SIGMA_SB) ** 0.25 - r["Ts_mean"]
    gap_rock = gap(C.C_BARE_ROCK)
    gap_land = gap(C.C_MOIST_LAND)
    assert gap_rock > gap_land >= -1e-6     # gap is non-negative (Jensen) and larger for low C
    assert gap_rock > 1.0                    # a real, band-relevant correction (a few K)


def test_periodic_energy_balance_invariant():
    # once converged, <sigT4> = <SW_abs> + LW_down - <SH> - <LE> (pinned by the mean forcing)
    t, SW = _sw_cycle()
    LW_down = 300.0
    r = S.solve_periodic_surface(t, SW, LW_down, 285.0, 6e-3, C.P_SURF, 5.0,
                                 C_surf=C.C_MOIST_LAND, wetness=0.5, max_days=200, tol=1e-5)
    lhs = r["sigT4_mean"]
    rhs = r["SW_abs_mean"] + LW_down - r["SH_mean"] - r["LE_mean"]
    assert lhs == pytest.approx(rhs, rel=1e-3)         # closes exactly when the cycle is periodic


def test_evaporation_increases_with_wetness_and_warmth():
    t, SW = _sw_cycle()
    dry = S.solve_periodic_surface(t, SW, 300.0, 285.0, 6e-3, C.P_SURF, 5.0,
                                   C_surf=C.C_MOIST_LAND, wetness=0.2, max_days=120)
    wet = S.solve_periodic_surface(t, SW, 300.0, 285.0, 6e-3, C.P_SURF, 5.0,
                                   C_surf=C.C_MOIST_LAND, wetness=1.0, max_days=120)
    assert wet["evap_mm_day"] > dry["evap_mm_day"] > 0


def test_radiative_mean_reference():
    assert S.radiative_mean_temperature(0.0, C.SIGMA_SB * 300.0**4) == pytest.approx(300.0, rel=1e-9)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
