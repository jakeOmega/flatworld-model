"""Tests for the optics-layer equatorial-focus lens reshape (flatworld.geometry, COSMO_LENS_RESHAPE).

Pins the two properties that live only in scratch/PR-body verification before this: the (b) reshape is
POWER-CONSERVING on the passed grid, and a partial/scalar request inside the donor range that is
returned UN-reshaped emits the one-time silent-forcing guard warning (the import-timing-L_SUN trap
class). The default-OFF byte-identity is already covered by test_geometry.py.
"""
import warnings

import numpy as np

from flatworld import geometry as g


def _profile(n=1500, theta_max_deg=70.0):
    """A synthetic full cross-track grid (0..theta_max) + an arbitrary strictly-positive day-mean S."""
    x_max = g.C.L_SUN * np.tan(np.radians(theta_max_deg))
    xs = np.linspace(0.0, x_max, n)
    S = 1000.0 / (1.0 + (xs / g.C.L_SUN) ** 2)          # cos^2-like, positive, monotone
    return xs, S


def test_reshape_conserves_power_on_the_grid():
    """Finding 3(iii): total power (trapz of S over x) is invariant under the reshape, by construction."""
    xs, S = _profile()
    out = g._apply_lens_reshape(xs, S)
    p0 = np.trapezoid(S, xs)
    p1 = np.trapezoid(out, xs)
    assert abs(p1 / p0 - 1.0) < 1e-12
    # and the redistribution actually moved power to the equator (non-vacuous)
    assert out[0] > 2.0 * S[0]


def test_reshape_is_identity_far_poleward_of_donor():
    """A sub-range entirely poleward of theta_donor is returned bit-identical (and must NOT warn)."""
    td = g._LENS_THETA_DONOR
    x_lo = g.C.L_SUN * np.tan(np.radians(td + 5.0))
    x_hi = g.C.L_SUN * np.tan(np.radians(td + 25.0))
    xs = np.linspace(x_lo, x_hi, 64)
    S = 1000.0 / (1.0 + (xs / g.C.L_SUN) ** 2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = g._apply_lens_reshape(xs, S)
    assert np.array_equal(out, S)
    assert not any("UN-reshaped" in str(m.message) for m in w)


def test_partial_grid_inside_donor_warns_once():
    """Finding 2 guard: a scalar/short request inside the donor range is passed through UN-reshaped and
    must fire the loud one-time warning (a diagnostic sampling the day-mean flux pointwise during a
    reshape run would otherwise silently get the un-lensed forcing)."""
    g._LENS_PASSTHROUGH_WARNED = False                  # reset the module-level one-shot latch
    x_eq = g.C.L_SUN * np.tan(np.radians(0.5))          # deep inside the pool/donor region
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = g._apply_lens_reshape(np.array([x_eq]), np.array([500.0]))   # size < 8 -> pass through
        again = g._apply_lens_reshape(np.array([x_eq]), np.array([500.0])) # second call: no new warn
    assert np.array_equal(out, np.array([500.0]))       # genuinely un-reshaped (the trap)
    assert np.array_equal(again, np.array([500.0]))
    msgs = [str(m.message) for m in w if "UN-reshaped" in str(m.message)]
    assert len(msgs) == 1                               # one-time only
