"""Tests for flatworld.thermo (moist thermodynamics)."""
import numpy as np
import pytest
from flatworld import thermo as th
from flatworld import constants as C


def test_esat_reference_points():
    assert th.esat(C.T_FREEZE) == pytest.approx(611.2, rel=1e-6)        # 0 C
    assert float(th.esat(293.15)) == pytest.approx(2339.0, rel=0.03)    # ~20 C
    assert float(th.esat(303.15)) == pytest.approx(4246.0, rel=0.03)    # ~30 C


def test_esat_monotone_increasing():
    T = np.linspace(230, 360, 200)
    assert np.all(np.diff(th.esat(T)) > 0)


def test_qsat_behaviour():
    # increases with T, decreases with p, positive
    assert th.qsat(300, 1e5) > th.qsat(280, 1e5) > 0
    assert th.qsat(300, 5e4) > th.qsat(300, 1e5)


def test_T_boil_inverts_esat_clamp():
    """``T_boil(p)`` is the exact inverse of the model's own warm-branch esat at the qsat-clamp
    threshold e = 0.999 p (#294): esat(T_boil(p)) == 0.999 p, and T >= T_boil(p) reproduces the
    boiling mask ``esat(T) >= 0.999 p`` on both sides of the threshold."""
    p = np.geomspace(5e3, 1.1e5, 40)                       # the model pressure range (p_top ~ 5 kPa .. surface)
    Tb = np.asarray(th.T_boil(p))
    assert np.allclose(np.asarray(th.esat(Tb)), 0.999 * p, rtol=1e-12)
    assert np.all(np.asarray(th.esat(Tb + 1e-6)) > 0.999 * p)   # just above T_boil -> boiling
    assert np.all(np.asarray(th.esat(Tb - 1e-6)) < 0.999 * p)   # just below -> not boiling
    # ~100 C at 1 atm -- Bolton-consistent (the model's esat), not the steam table
    assert float(th.T_boil(np.asarray(float(C.P_SURF)))) - C.T_FREEZE == pytest.approx(100.0, abs=3.0)


def test_dqsat_dT_matches_finite_difference():
    T, p = 295.0, 9e4
    fd = (th.qsat(T + 0.01, p) - th.qsat(T - 0.01, p)) / 0.02
    assert th.dqsat_dT(T, p) == pytest.approx(fd, rel=1e-3)


def test_moist_lapse_below_dry_and_limits():
    dry = th.dry_adiabat_lapse_rate()
    assert th.moist_adiabat_lapse_rate(300, 1e5) < dry          # moist < dry (latent release)
    # cold/dry air: moist adiabat -> dry adiabat
    assert th.moist_adiabat_lapse_rate(220, 3e4) == pytest.approx(dry, rel=0.05)


def test_moist_adiabat_decreasing_with_height():
    p = np.array([1e5, 8e4, 5e4, 2e4])
    T = th.temperature_on_moist_adiabat(300.0, 1e5, p)
    assert np.all(np.diff(T) < 0)                                # cools upward
    assert T[0] == pytest.approx(300.0, abs=0.5)                 # surface value preserved


def test_dry_column_follows_dry_adiabat():
    # a very dry/cold surface parcel must follow the dry adiabat in T(p): T/T0 = (p/p0)^kappa
    p = np.array([1e5, 7e4, 4e4])
    T = th.temperature_on_moist_adiabat(240.0, 1e5, p)
    T_dry = 240.0 * (p / 1e5) ** C.KAPPA_POISSON
    assert np.allclose(T, T_dry, rtol=0.02)


def test_virtual_temperature_exceeds_T_and_matches_linear_approx():
    # moist air is less dense -> T_v > T; condensate loading lowers it
    T, q_v = 300.0, 0.02
    assert th.virtual_temperature(T, q_v) > T
    approx = T * (1.0 + 0.608 * q_v)
    assert th.virtual_temperature(T, q_v) == pytest.approx(approx, rel=2e-3)
    # condensate loading reduces the virtual temperature
    assert th.virtual_temperature(T, q_v, q_c=0.005) < th.virtual_temperature(T, q_v, q_c=0.0)


def test_density_temperature_is_the_virtual_temperature_minus_the_loading():
    # PHY-19 step 2: T_rho = T_v(T, q_v) - T*q_cond. Zero condensate is EXACTLY the virtual temperature
    # (an IEEE identity, not an approximation).
    T, q_v, q_cond = 300.0, 0.02, 1.5e-3
    assert th.density_temperature(T, q_v, 0.0) == th.virtual_temperature(T, q_v)
    assert th.density_temperature(T, q_v, q_cond) == th.virtual_temperature(T, q_v) - T * q_cond
    # the loading is a REDUCTION of exactly T*q_cond (~0.45 K here)
    d = th.virtual_temperature(T, q_v) - th.density_temperature(T, q_v, q_cond)
    assert d == pytest.approx(T * q_cond, rel=1e-12)
    # array-generic, elementwise
    Ta = np.array([280.0, 300.0]); qa = np.array([0.005, 0.02]); qc = np.array([0.0, 2e-3])
    assert np.allclose(th.density_temperature(Ta, qa, qc), th.virtual_temperature(Ta, qa) - Ta * qc)


def test_density_temperature_loading_term_is_exact_for_specific_humidities():
    """The LOADING is exact, not a linearization -- and the whole residual against the gas law's closed
    form is the INHERITED vapor term, with no condensate part. For specific humidities (mass fractions
    of TOTAL mass, which is what this model's q is -- qsat returns eps*e/(p-(1-eps)e) = rho_v/rho),
    only the gas phases exert pressure, so with q_d = 1 - q_v - q_cond,
        p/(rho R_d T) = q_d + q_v/eps = 1 + q_v(1/eps - 1) - q_cond    =>  T_rho = T(1 + 0.608 q_v - q_cond)
    with no expansion anywhere. The ratio form T(1+r/eps)/(1+r) is the MIXING-ratio expression.
    (The vapor residual below is ``virtual_temperature``'s DEFAULT, ratio-form branch -- ROADMAP PHY-20.
    It is default ON and canonical, so it is asserted here as the legacy behaviour; its fix lives behind
    ``exact=True`` / ``cfg.virtual_temp_exact``, whose twin assert at the end of this test shows the
    residual go to exactly zero.)"""
    rng = np.random.default_rng(0)
    T = rng.uniform(250.0, 330.0, 400)
    q_v = rng.uniform(1e-4, 0.10, 400)
    q_cond = rng.uniform(1e-4, 5e-3, 400)
    c = 1.0 / C.EPS_MW - 1.0                                  # 0.608..., the exact coefficient
    # 1. switching the loading on subtracts EXACTLY T*q_cond (to round-off), at every q_v
    got = th.density_temperature(T, q_v, 0.0) - th.density_temperature(T, q_v, q_cond)
    assert np.allclose(got, T * q_cond, rtol=1e-11, atol=1e-11)
    # 2. the residual vs the gas-law closed form is EXACTLY the vapor ratio-form residual: the loading
    #    contributes none of it (a linearized loading would leave an O(q_v*q_cond*T) condensate residue)
    exact = T * (1.0 + c * q_v - q_cond)
    vapor_residual = th.virtual_temperature(T, q_v) - T * (1.0 + c * q_v)
    assert np.allclose(th.density_temperature(T, q_v, q_cond) - exact, vapor_residual,
                       rtol=1e-9, atol=1e-11)
    # 3. ... and that residual is independent of q_cond -- the sharpest statement of "the loading is exact"
    r1 = th.density_temperature(T, q_v, q_cond) - T * (1.0 + c * q_v - q_cond)
    r2 = th.density_temperature(T, q_v, 3.0 * q_cond) - T * (1.0 + c * q_v - 3.0 * q_cond)
    assert np.allclose(r1, r2, rtol=1e-9, atol=1e-11)
    # size of the default-path vapor residual, for the record: sub-0.1 K at q_v=0.02, ~1.7 K at q_v=0.10
    assert abs(float(th.virtual_temperature(300.0, 0.02) - 300.0 * (1 + c * 0.02))) < 0.1
    assert 1.0 < abs(float(th.virtual_temperature(300.0, 0.10) - 300.0 * (1 + c * 0.10))) < 2.5
    # PHY-20 twin: with the EXACT specific-humidity branch selected the residual is exactly zero, at
    # every q_v AND every q_cond -- the whole density temperature is then the gas-law closed form.
    # (to round-off only: the code composes T(1+c q_v) - T q_cond, the reference T(1+c q_v - q_cond) --
    #  algebraically the same line, associated differently in IEEE arithmetic.)
    assert np.allclose(th.density_temperature(T, q_v, q_cond, exact=True), exact, rtol=1e-15, atol=1e-13)
    # (the scalar twin: the default q_c = 0.0 makes `x - 0.0 == x` exact here, so this one IS bitwise
    #  -- but it is asserted with a tolerance anyway, since nothing downstream relies on the last ulp
    #  and a future re-association of the expression should not turn into a red test. PR #839 review.)
    assert float(th.virtual_temperature(300.0, 0.10, exact=True)) == pytest.approx(
        300.0 * (1 + c * 0.10), rel=1e-15)


def test_vapor_pressure_inverts_qsat():
    T, p = 298.0, 9e4
    q = th.qsat(T, p)
    e = th.vapor_pressure_from_q(q, p)
    assert float(e) == pytest.approx(float(th.esat(T)), rel=1e-9)


def test_lcl_parcel_just_saturates():
    # a sub-saturated surface parcel lifted to its LCL should be ~exactly saturated there
    T_surf, p_surf = 300.0, 1.0e5
    q_surf = 0.7 * th.qsat(T_surf, p_surf)              # RH 70%
    p_lcl, T_lcl = th.lifting_condensation_level(T_surf, q_surf, p_surf)
    assert p_lcl < p_surf and T_lcl < T_surf            # LCL is above and cooler
    assert float(th.qsat(T_lcl, p_lcl)) == pytest.approx(q_surf, rel=0.03)


def test_lcl_at_surface_for_saturated_parcel():
    T_surf, p_surf = 295.0, 1.0e5
    q_surf = th.qsat(T_surf, p_surf)                    # already saturated -> LCL at the surface
    p_lcl, T_lcl = th.lifting_condensation_level(T_surf, q_surf, p_surf)
    assert float(p_lcl) == pytest.approx(p_surf, rel=0.02)


def test_saturation_mse_and_mse_consistency():
    T, p, z = 300.0, 9e4, 1000.0
    # saturation_mse == mse evaluated at q = qsat
    assert float(th.saturation_mse(T, p, z)) == pytest.approx(float(th.mse(T, p, z, th.qsat(T, p))))
    # MSE rises if either T or q rises (the L*q coefficient sign)
    assert th.mse(T, p, z, 0.02) > th.mse(T, p, z, 0.01)


def test_evaporation_unit_conversion():
    # LE that evaporates 1 mm/day:  1 mm/day = 1 kg/m^2/day = L_vap/86400 W/m^2
    LE = C.L_VAP / 86400.0
    assert th.evaporation_rate_mm_day(LE) == pytest.approx(1.0, rel=1e-9)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
