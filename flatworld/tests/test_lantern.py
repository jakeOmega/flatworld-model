"""Gates for the lantern painted-day forcing family (``flatworld.lantern``, locked premise
2026-06-11; shape updated to the analytic plateau 2026-06-26): exact day-mean preservation at every
latitude (the climate-invariance guarantee), the analytic plateau inside the band (fading smoothly
to a dark night), the scaled NATURAL day outside, the C1 morph in between, and the
``land_surface_diurnal(day_cycle=...)`` hook contract. Pure-numpy
fast tests (no Stage A)."""
import numpy as np
import pytest

from flatworld import geometry as g
from flatworld import lantern

S_TEST = lambda th: 700.0 - 6.0 * np.asarray(th, float)        # noqa: E731  synthetic budget


def _cycle(th_deg, n=288):
    x = float(g.x_of_theta(np.radians(th_deg)))
    return lantern.day_flux_cycle(x, S_TEST, n_steps=n)


@pytest.mark.parametrize("th", [30.0, 43.0, 47.5, 52.0, 60.0])
def test_day_mean_equals_budget_everywhere(th):
    t, F = _cycle(th)
    assert np.trapezoid(F, t) / t[-1] == pytest.approx(S_TEST(th), rel=1e-9)
    assert np.all(F >= 0.0)


def test_band_day_is_the_plateau():
    t, F = _cycle(47.5)
    tau_h = np.minimum(t, t[-1] - t) / 3600.0
    noon = S_TEST(47.5) / lantern.DUTY
    assert F[0] == pytest.approx(noon, rel=5e-3)               # noon peak (numerical renorm tolerance)
    # the analytic plateau fades smoothly (erfc edges) into a dark night — no hard cutoff, but
    # negligible well past the lit span and ~dark at midnight
    assert F[tau_h > 9.5].max() < 1e-3 * noon                 # faded out beyond the ~16-h lit span
    assert F[len(t) // 2] < 1e-6 * noon                       # midnight ~dark (erfc tail, not exactly 0)


def test_outside_band_is_the_scaled_natural_day():
    th = lantern.BAND_THETA[0] - lantern.TRANS_HOT - 3.0       # well equatorward of the band -> pure natural
    t, F = _cycle(th)
    x = float(g.x_of_theta(np.radians(th)))
    _, G = g.diurnal_flux_cycle(x, n_steps=288, n_suns=2)      # natural shape (unshaded env here)
    assert lantern.band_weight(th) == 0.0
    # 0.995 not ~1: lantern's natural component is _one_sun_total (direct + forward-scattered haze)
    # while diurnal_flux_cycle is the direct beam -- the shapes differ by the diffuse fraction
    assert np.corrcoef(F, G)[0, 1] > 0.995


def test_morph_zone_blends_both_components():
    th = lantern.BAND_THETA[0] - lantern.TRANS_HOT / 2.0       # mid warm-side morph (anchor-relative)
    assert 0.0 < lantern.band_weight(th) < 1.0
    t, F = _cycle(th)
    tau_h = np.minimum(t, t[-1] - t) / 3600.0
    late = F[(tau_h > 9.0)]
    assert late.any() and late.max() > 0.0                     # the natural tail shines past the 16-h window


def test_band_weight_structure():
    # anchor-relative: 1 inside the band, 0 well outside it, partial in the morph zones (so this can't
    # go stale when BAND_THETA tracks the ribbon -- the 2026-06-28 regression).
    b0, b1 = lantern.BAND_THETA
    inside = 0.5 * (b0 + b1)
    below = b0 - lantern.TRANS_HOT - 2.0
    above = b1 + lantern.TRANS_COLD + 2.0
    w = lantern.band_weight(np.array([below, inside, above]))
    assert w[0] == 0.0 and w[2] == 0.0                         # fully outside -> scaled natural day
    assert w[1] == 1.0                                         # mid-band -> pure plateau
    assert 0.0 < lantern.band_weight(b0 - lantern.TRANS_HOT / 2.0) < 1.0    # morph blends


def test_day_cycle_hook_contract():
    # the cascade hook calls day_cycle(x) -> (t, F) with solve_periodic_surface's conventions
    t, F = _cycle(47.5, n=144)
    assert t[0] == 0.0 and t[-1] == pytest.approx(g.CLOCK.T_day)
    assert len(t) == 145 and np.allclose(np.diff(t), t[1] - t[0])
    assert np.all(np.isfinite(F))


# ---- realized-lens-day path (2026-07-18): default when a genuine 2-D optic is loaded; the analytic
# plateau above is the byte-identical escape. The lens-specific kappa/night-length numbers are verified
# under the canonical env in realized_day_verify.py -- here we lock the SELECTION + the invariants that
# must hold in BOTH shapes, in the bare (lens-free) test world.

def test_use_realized_selection(monkeypatch):
    # explicit override always wins
    assert lantern._use_realized(True) is True
    assert lantern._use_realized(False) is False
    # auto: plateau when no genuine 2-D optic is loaded (the bare test world)
    monkeypatch.setattr(g, "_SHADE2D", None)
    monkeypatch.delenv("COSMO_LANTERN_PLATEAU", raising=False)
    assert lantern._use_realized(None) is False
    # auto: realized when a 2-D optic IS loaded (the canonical lens world)
    monkeypatch.setattr(g, "_SHADE2D", ("sentinel",))
    assert lantern._use_realized(None) is True
    # explicit escape forces the plateau even with a lens loaded
    monkeypatch.setenv("COSMO_LANTERN_PLATEAU", "1")
    assert lantern._use_realized(None) is False


def test_realized_preserves_day_mean_and_nonneg():
    # the climate-invariance guarantee must hold in the realized shape too (renormalized to S)
    x = float(g.x_of_theta(np.radians(47.5)))
    t, F = lantern.day_flux_cycle(x, S_TEST, n_steps=288, realized=True)
    assert np.trapezoid(F, t) / t[-1] == pytest.approx(S_TEST(47.5), rel=1e-9)
    assert np.all(F >= 0.0)
    assert t[0] == 0.0 and t[-1] == pytest.approx(g.CLOCK.T_day) and len(t) == 289


def test_realized_shape_differs_from_plateau():
    # SAME day-mean, DIFFERENT diurnal structure -> the two branches are genuinely distinct forcings
    x = float(g.x_of_theta(np.radians(47.5)))
    _, Fp = lantern.day_flux_cycle(x, S_TEST, n_steps=288, realized=False)  # plateau escape
    _, Fr = lantern.day_flux_cycle(x, S_TEST, n_steps=288, realized=True)   # realized
    assert not np.allclose(Fp, Fr)


def test_plateau_escape_is_byte_identical(monkeypatch):
    # in the bare env the DEFAULT is the plateau, so realized=False and COSMO_LANTERN_PLATEAU must
    # reproduce it exactly (the byte-identical regression escape)
    monkeypatch.setattr(g, "_SHADE2D", None)
    monkeypatch.delenv("COSMO_LANTERN_PLATEAU", raising=False)
    x = float(g.x_of_theta(np.radians(47.5)))
    t0, F0 = lantern.day_flux_cycle(x, S_TEST, n_steps=288)                 # auto -> plateau
    t1, F1 = lantern.day_flux_cycle(x, S_TEST, n_steps=288, realized=False)
    assert np.array_equal(F0, F1) and np.array_equal(t0, t1)
    monkeypatch.setenv("COSMO_LANTERN_PLATEAU", "1")
    _, F2 = lantern.day_flux_cycle(x, S_TEST, n_steps=288)                  # env forces the escape
    assert np.array_equal(F0, F2)
