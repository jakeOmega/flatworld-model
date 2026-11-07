"""Unit tests for ``climate_anchors.anchors_from_profiles`` -- deriving the geology band anchors
(ribbon/ice/salt/band) from an AD-core zonal climate, so a new canonical world is a one-command re-pin.

The derivation is exercised on SYNTHETIC profiles (no checkpoint) covering the cases that broke the
naive rules: a warm-shifted ribbon (liquid runoff peaks equatorward of it) and a far-field snow bump
(a lens-donated poleward re-warming that must NOT be read as the calving divide)."""
import numpy as np
import pytest

from flatworld import climate_anchors as ca


def _cooling_profile(theta, t_equator=120.0, t_slope=-3.0):
    """A monotone warm->cold T_surf(theta): hot inner flat cooling through 30 C and 0 C to deep cold."""
    return t_equator + t_slope * theta


def test_ribbon_ice_salt_from_a_monotone_profile():
    th = np.linspace(0.0, 80.0, 400)
    T = _cooling_profile(th)                                  # 120 C at eq, crosses 30 C @30, 0 C @40, 100C @6.7
    a = ca.anchors_from_profiles(th, T)
    assert a["ribbon"][0] == pytest.approx(30.0, abs=0.5)     # 30 C down-crossing
    assert a["ribbon"][1] == pytest.approx(40.0, abs=0.5)     # 0 C down-crossing
    assert a["ice"] == a["ribbon"][1]                         # ice = the cold edge
    assert a["salt"] == pytest.approx(6.67, abs=0.5)          # 100 C (salt-flat) down-crossing
    ca.assert_anchors_monotonic(a)                            # salt < r0 < r1 <= ice < band


def test_band_is_the_near_margin_snow_peak_not_a_far_field_bump():
    # a lens-donated far-field re-warming makes a SECOND snow peak far poleward; band must pick the
    # NEAR-margin dome (just poleward of ice), not the far bump.
    th = np.linspace(0.0, 80.0, 400)
    T = _cooling_profile(th)                                  # ice line ~theta40
    w_snow = np.zeros_like(th)
    w_snow[(th > 40) & (th < 45)] = 0.8                       # the near-margin accumulation dome
    w_snow[(th > 65) & (th < 72)] = 1.5                       # a bigger FAR-field bump (lens) -- a decoy
    a = ca.anchors_from_profiles(th, T, w_snow=w_snow)
    assert 40.0 < a["band"] < 52.0, "band must be the near-margin dome, not the far-field lens bump"


def test_warm_shifted_world_is_monotonic_and_salt_bounds_the_peak_guard():
    # a warm-shifted ribbon (like the day-160 world): 30 C @43, 0 C @47 -- narrow + equatorward. The salt
    # sill lands well equatorward, and the anchors stay monotone.
    th = np.linspace(0.0, 80.0, 800)
    T = 250.0 - 4.8 * th                                      # 30 C @~45.8, 0 C @~52 ... tune to warm-shift
    T = np.where(th < 20, 250.0 - 2.0 * th, T)               # hotter inner flat
    a = ca.anchors_from_profiles(th, T)
    ca.assert_anchors_monotonic(a)
    assert a["salt"] < a["ribbon"][0]                         # the salt sill is equatorward of the ribbon


def test_wetness_sets_the_salt_sill_when_given():
    th = np.linspace(0.0, 80.0, 400)
    T = _cooling_profile(th)
    wet = (th >= 15.0).astype(float)                          # open water only poleward of theta15
    a = ca.anchors_from_profiles(th, T, wetness=wet)
    assert a["salt"] == pytest.approx(15.0, abs=0.3)          # innermost open-water column


def test_no_ribbon_raises():
    th = np.linspace(0.0, 80.0, 100)
    T = np.full_like(th, 200.0)                               # never cools below 30 C -> no ribbon
    with pytest.raises(ValueError, match="warm-edge"):
        ca.anchors_from_profiles(th, T)


def test_monotonic_guard_rejects_a_bad_anchor_set():
    with pytest.raises(ValueError, match="salt < ribbon"):
        ca.assert_anchors_monotonic(dict(ribbon=(40.0, 35.0), band=50.0, salt=25.0, ice=45.0))  # r0>r1


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
