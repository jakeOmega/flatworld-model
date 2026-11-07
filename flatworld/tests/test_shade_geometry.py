"""Gates for the graded lantern shade in geometry (COSMO_SHADE_FILE): default OFF must be
byte-identical (the _SHADE=None path is the pre-shade code), an active shade must multiply the
flux law by T_eff(alpha) exactly (replacing the binary rim), and the clock must never see it."""
import numpy as np

import flatworld.geometry as g
from flatworld import constants as C


def _with_shade(alpha, T_eff):
    return (np.asarray(alpha, float), np.asarray(T_eff, float))


def test_shade_multiplies_flux_and_replaces_rim():
    saved = g._SHADE
    try:
        r_over = C.L_SUN * 1.0                                  # overhead: alpha=0, rim fully open
        base = g._flux_one_sun(r_over)
        g._SHADE = _with_shade([0.0, 89.0], [0.5, 0.5])         # flat 50% shade
        assert np.isclose(g._flux_one_sun(r_over), 0.5 * base)
        # angle dependence: a shade dark above 45 deg kills the low sun even where the old rim was open
        g._SHADE = _with_shade([0.0, 44.0, 46.0, 89.0], [1.0, 1.0, 0.0, 0.0])
        r_60 = C.L_SUN / np.cos(np.radians(60.0))
        assert g._flux_one_sun(r_60) == 0.0
        r_30 = C.L_SUN / np.cos(np.radians(30.0))
        g._SHADE = None
        f30_rim = g._flux_one_sun(r_30)
        g._SHADE = _with_shade([0.0, 44.0, 46.0, 89.0], [1.0, 1.0, 0.0, 0.0])
        assert np.isclose(g._flux_one_sun(r_30), f30_rim / g._visible_fraction(r_30))
        # T_eff > 1 encodes brighter suns behind darker shades (the I0_mult fold-in)
        g._SHADE = _with_shade([0.0, 89.0], [3.0, 3.0])
        assert np.isclose(g._flux_one_sun(r_over), 3.0 * base)
    finally:
        g._SHADE = saved


def test_clock_and_off_path_untouched():
    saved = g._SHADE
    try:
        g._SHADE = _with_shade([0.0, 89.0], [0.0, 0.0])         # a fully dark shade
        # the clock's reference flux passes gate=False -> never shaded (spacing D is a setting value)
        r = C.L_SUN * 1.3
        assert g._flux_one_sun(r, tau0=C.TAU0_HAZE, gate=False) > 0.0
        clock = g.SunsClock()
        assert np.isclose(clock.spacing, g.CLOCK.spacing)       # identical clock under any shade
        g._SHADE = None
        assert g._flux_one_sun(r) > 0.0                         # off path alive and well
    finally:
        g._SHADE = saved
