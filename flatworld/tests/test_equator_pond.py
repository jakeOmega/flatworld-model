"""Gates for the equatorial pond/albedo surface-equilibrium model (``_equator_pond``, roadmap A).

The model is a stand-alone surface-energy-balance + water-budget characterization, so the gates
check the physics that the verdict rests on: the dry-flat anchor round-trips; the open-water tile is
the evaporative thermostat (a cool, unique root) whose temperature is INDEPENDENT of the air
temperature (the cancellation identity); the uniform surface is bistable (a cool root for fixed open
water, runaway-to-hot for the T-keyed wetness ramp); and the wet-fraction closure / regime
classification behave as advertised.
"""
import numpy as np
import pytest

from flatworld import _equator_pond as ep
from flatworld import constants as C


def _forcing(**kw):
    """A deterministic, L-independent forcing: fixed equatorial SW, LW anchored to a 138 C dry flat."""
    base = dict(S_sfc=1711.5, T_air=80.0 + C.T_FREEZE, q_air=0.07, wind=7.0,
                albedo_water=0.08, albedo_dry=0.30, T_dry_target=138.0 + C.T_FREEZE)
    base.update(kw)
    f = ep.EquatorForcing(LW_down=0.0, **base)
    f.LW_down = ep.anchor_lw_down(f.S_sfc, f.T_dry_target, f.T_air, f.wind, f.albedo_dry, Ch=f.Ch)
    return f


def test_dry_anchor_roundtrips():
    """The LW anchor puts the dry salt-flat tile (no evaporation) exactly at the target temperature."""
    f = _forcing(T_dry_target=138.0 + C.T_FREEZE)
    assert abs(ep.salt_flat_temperature(f) - f.T_dry_target) < 0.05


def test_open_water_is_a_cool_thermostat():
    """A freely-evaporating water tile self-limits FAR below the dry flat, with a physical evaporation."""
    f = _forcing()
    Tw, Ew = ep.open_water_equilibrium(f)
    Td = ep.salt_flat_temperature(f)
    assert 50.0 < Tw - C.T_FREEZE < 75.0           # the evaporative thermostat, ~60-65 C
    assert Td - Tw > 60.0                            # far cooler than the 138 C salt flat
    assert 40.0 < Ew < 130.0                         # potential evaporation, ~70 mm/day


def test_Tw_independent_of_air_temperature():
    """KEY identity: with the dry-flat anchor, the two SH terms net to Ck(T_d - T_w) and T_air cancels,
    so the wet tile's temperature and evaporation are independent of the (unknown) air temperature."""
    a = ep.open_water_equilibrium(_forcing(T_air=55.0 + C.T_FREEZE))
    b = ep.open_water_equilibrium(_forcing(T_air=120.0 + C.T_FREEZE))
    assert abs(a[0] - b[0]) < 1e-6                    # T_w identical
    assert abs(a[1] - b[1]) < 1e-6                    # E_w identical


def test_evaporation_increases_with_wind():
    """More wind -> more evaporation (and a cooler wet tile)."""
    Ews = [ep.open_water_equilibrium(_forcing(wind=U))[1] for U in (5.0, 7.0, 10.0, 14.0)]
    assert all(b > a for a, b in zip(Ews, Ews[1:]))


def test_wet_fraction_clip_and_monotone():
    f = _forcing()
    _, Ew = ep.open_water_equilibrium(f)
    assert ep.wet_fraction(0.5 * Ew, Ew) == pytest.approx(0.5)
    assert ep.wet_fraction(2.0 * Ew, Ew) == 1.0       # clipped at full coverage
    assert ep.wet_fraction(0.0, Ew) == 0.0
    fr = [ep.wet_fraction(F, Ew) for F in np.linspace(0, 2 * Ew, 20)]
    assert all(b >= a for a, b in zip(fr, fr[1:]))    # monotone non-decreasing


def test_classify_regime_boundaries():
    Ew = 70.0
    assert ep.classify_regime(1.5 * Ew, Ew) == "open_water"
    assert ep.classify_regime(0.5 * Ew, Ew) == "pond_field"
    assert ep.classify_regime(0.005 * Ew, Ew) == "salt_flat"


def test_surface_state_is_closure_dependent():
    """The crux (precise framing): the two states come from two CLOSURES, not two roots of one fixed
    map. EACH closure has a UNIQUE root -- a fixed open-water surface a COOL one (sign change in
    55-65 C); the T-keyed wetness ramp stays net-heating across the dry band (no cool root) and crosses
    only at the HOT ~138 C salt flat. So the water budget (which closure), not bistability of one map,
    picks the state."""
    f = _forcing()
    Tc = np.linspace(45.0, 160.0, 1151)
    r_ow, r_ramp = ep.seb_residual_curve(f, Tc)
    ow_roots = Tc[np.where(np.diff(np.sign(r_ow)) != 0)[0]]
    ramp_roots = Tc[np.where(np.diff(np.sign(r_ramp)) != 0)[0]]
    assert ow_roots.size == 1 and 55.0 < ow_roots[0] < 65.0     # open water: ONE cool root
    assert ramp_roots.size == 1 and ramp_roots[0] > 130.0       # ramp: ONE hot root (runaway, no cool root)


def test_fraction_from_area_mean_inverts():
    f = ep.fraction_from_area_mean_T(82.5, 62.4, 137.8)
    assert f == pytest.approx((137.8 - 82.5) / (137.8 - 62.4), abs=1e-6)
    assert ep.fraction_from_area_mean_T(62.4, 62.4, 137.8) == 1.0   # mean == wet tile  -> all wet (f=1)
    assert ep.fraction_from_area_mean_T(200.0, 62.4, 137.8) == 0.0  # hotter than dry tile -> clipped to 0


def test_radiative_and_linear_means_agree_on_mostly_wet():
    """The canonical-core fraction is 'mostly wet' under either mean (linear-T ~0.73, radiative T^4
    ~0.79); the radiative mean is the higher (sigma T^4 weights the hot dry tile more)."""
    f_lin = ep.fraction_from_area_mean_T(82.5, 62.4, 137.8)
    f_rad = ep.fraction_from_radiative_mean_T(82.5, 62.4, 137.8)
    assert f_rad > f_lin > 0.6                        # both mostly-wet; radiative reads wetter
    assert f_rad == pytest.approx(0.79, abs=0.02)


def test_zone_state_is_internally_consistent():
    f = _forcing()
    st = ep.zone_state(35.0, f)
    assert st["T_wet_C"] < st["T_area_mean_C"] < st["T_dry_C"]      # mean between the tiles
    assert f.albedo_water <= st["albedo_mean"] <= f.albedo_dry      # albedo between the tiles
    assert st["regime"] == ep.classify_regime(35.0, st["E_wet_mm_day"])
    assert 0.0 <= st["wet_fraction"] <= 1.0


def test_regime_diagram_spans_all_three_regimes():
    f = _forcing()
    _, Ew = ep.open_water_equilibrium(f)
    F = np.array([0.001 * Ew, 0.5 * Ew, 5.0 * Ew])
    frac, Tmean, alb, (Ew2, Tw, Td) = ep.regime_diagram(f, F)
    assert frac[0] < 0.02 and frac[1] == pytest.approx(0.5) and frac[2] == 1.0
    assert Tmean[0] > Tmean[1] > Tmean[2]              # wetter -> cooler zone mean
    assert Td > Tw and Ew2 == pytest.approx(Ew)


def test_canonical_supply_warns_off_canonical_L():
    """int P dx is L-calibrated to the fixture's L=8e7 world; running off-L must warn (the silent
    de-calibration trap)."""
    orig = C.L_SUN
    try:
        C.L_SUN = 2.0e7
        with pytest.warns(UserWarning, match="L-calibrated"):
            int_P_dx, T_dry, T_core = ep.canonical_supply()
        assert int_P_dx > 0 and T_dry > 100.0          # still returns the (off-L) values
    finally:
        C.L_SUN = orig
