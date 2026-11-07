"""Tests for the fixed geometry/forcing givens (flatworld.geometry, flatworld.constants).

These pin the INPUTS: the metric, the flux law, the suns' clock, and the absolute day-mean flux,
all against the numbers stated in ../../cosmology.md. If any of these break, an input drifted.
"""
import numpy as np
import pytest
from flatworld import geometry as g
from flatworld import constants as C


def _collimator_off(monkeypatch):
    """Force the NO-collimator reference state (full haze, fiducial I0) for the doc-reference numerics.
    The collimator is now ON by default (the accurate worldbuilding geometry), but cosmology.md still
    states the no-collimator fluxes (the Track-C doc refresh is deferred -- ALLON_CAMPAIGN.md), so the
    'matches_doc' numerics are pinned in the collimator-off limit until the doc is refreshed."""
    monkeypatch.setattr(C, "COLLIMATOR", False)
    monkeypatch.setattr(C, "I0_OVERHEAD", C.I0_OVERHEAD / C.COLLIMATOR_I0_REPIN)  # undo the repin -> fiducial


def test_metric_matches_doc_table():
    # S4 distance table at L=2L (suns raised L->2L, 2026-06-07): 45->20000, 50->23800, 70->55000 km
    # (exactly 2x the old L=1e7 values -- x = L*tan(theta), so the metric scales linearly with L_SUN).
    assert g.x_of_theta(np.radians(45)) / 1e3 == pytest.approx(20000, rel=1e-6)
    assert g.x_of_theta(np.radians(50)) / 1e3 == pytest.approx(23800, rel=2e-3)
    assert g.x_of_theta(np.radians(70)) / 1e3 == pytest.approx(55000, rel=2e-3)


def test_metric_inverse_roundtrip():
    x = np.linspace(0, 5e7, 50)
    assert np.allclose(g.x_of_theta(g.theta_of_x(x)), x, rtol=1e-12, atol=1e-6)


def test_flux_law_identity():
    # cos^3 theta must equal L^3/(L^2+x^2)^{3/2} exactly (the two derivations of the same law)
    th = np.radians(np.linspace(0, 80, 100))
    assert np.allclose(np.cos(th) ** 3, g.flux_shape_of_x(g.x_of_theta(th)), atol=1e-12)


def test_suns_clock_from_floor():
    # The clock is fixed by the midnight floor, not free. At L=2L: D~410,000 km, v~4.75 km/s, day~24 h
    # (D and v both double with L_SUN; the day is held at 24 h, so the floor-fixing requirement is met
    # by the doubled spacing/speed). The flux law -- hence the floor ratio -- is L-invariant.
    assert g.CLOCK.spacing_km == pytest.approx(410000, rel=0.02)
    assert g.CLOCK.speed_km_s == pytest.approx(4.75, rel=0.05)
    assert g.CLOCK.T_day / 3600 == pytest.approx(24.0, rel=1e-6)


def test_clock_is_brightness_independent():
    # I0 cancels in the floor ratio: changing the brightness must not change the clock.
    import importlib
    orig = C.I0_OVERHEAD
    try:
        C.I0_OVERHEAD = orig * 3.0
        importlib.reload(g)
        assert g.CLOCK.spacing_km == pytest.approx(410214, rel=1e-6)   # L=2L: exactly 2x the old 205107
    finally:
        C.I0_OVERHEAD = orig
        importlib.reload(g)


def test_absolute_flux_matches_doc(monkeypatch):
    # S6: noon@theta50 ~ 3300 W/m^2 ; equator noon > 20x Earth clear-noon (~1000 W/m^2). Pinned in the
    # collimator-off limit (the doc's current basis; see _collimator_off).
    _collimator_off(monkeypatch)
    x50 = g.x_of_theta(np.radians(50))
    assert g._flux_one_sun(g.slant_distance(x50, 0.0)) == pytest.approx(3300, rel=0.05)
    assert g._flux_one_sun(g.slant_distance(0.0, 0.0)) / 1000.0 > 20.0


def test_day_mean_flux_monotone_decreasing():
    x = g.x_of_theta(np.radians(np.array([0, 20, 30, 45, 50, 60, 70, 78], float)))
    S = g.day_mean_flux(x)
    assert np.all(np.diff(S) < 0)              # strictly colder away from the sun-line
    assert np.all(S > 0)


def test_day_mean_equals_diurnal_cycle_mean():
    # Self-consistency: the analytic unfolded day-mean must equal the time-mean of the diurnal cycle.
    for th in (30.0, 50.0, 65.0):
        x = float(g.x_of_theta(np.radians(th)))
        t, S = g.diurnal_flux_cycle(x, n_steps=6000, n_suns=10)
        cycle_mean = np.trapezoid(S, t) / g.CLOCK.T_day
        assert cycle_mean == pytest.approx(g.day_mean_flux_scalar(x), rel=2e-3)


def test_day_mean_flux_quadrature_resolution_independent():
    # The sinh-stretched quadrature must be converged: doubling nodes must not move S.
    x = float(g.x_of_theta(np.radians(50)))
    s_coarse = g.day_mean_flux_scalar(x, n=3000)
    s_fine = g.day_mean_flux_scalar(x, n=12000)
    assert s_fine == pytest.approx(s_coarse, rel=1e-4)


def test_day_mean_flux_vectorized_equals_scalar_reference():
    # The vectorized+cached day_mean_flux MUST reproduce the reference loop of day_mean_flux_scalar
    # BIT-FOR-BIT (it is the most-used function -- the C7 vectorization must not move any number).
    x = g.x_of_theta(np.radians(np.array([0, 10, 20, 30, 45, 50, 55, 60, 70, 78, 85], float)))
    ref = np.array([g.day_mean_flux_scalar(float(xi)) for xi in x])   # the old per-x loop, verbatim
    vec = g.day_mean_flux(x)
    assert np.array_equal(vec, ref)                                   # exact, not just close
    # scalar-collapse contract preserved: a size-1 input returns a python float == the scalar path
    s1 = g.day_mean_flux(np.array([float(x[4])]))
    assert isinstance(s1, float) and s1 == ref[4]


def test_day_mean_flux_cache_hits_and_returns_fresh_copy():
    # Re-requesting the SAME grid must hit the memo (the whole point: stages/sweep members share a grid)
    # yet each call must hand back a FRESH, writeable array so a caller can mutate it without corrupting
    # the shared cache entry.
    g._day_mean_flux_cached.cache_clear()
    x = g.x_of_theta(np.radians(np.array([20, 40, 55, 70], float)))
    a = g.day_mean_flux(x)
    info_after_first = g._day_mean_flux_cached.cache_info()
    assert info_after_first.misses == 1 and info_after_first.hits == 0
    b = g.day_mean_flux(x)
    info_after_second = g._day_mean_flux_cached.cache_info()
    assert info_after_second.hits == 1                                # second identical call HIT the cache
    assert np.array_equal(a, b)
    assert a.flags.writeable and a is not b                           # fresh writeable copies, not aliases
    a[:] = -1.0                                                       # mutate the returned array...
    c = g.day_mean_flux(x)
    assert np.array_equal(c, b)                                       # ...cache entry is untouched


def test_day_mean_flux_cache_not_stale_under_constant_change():
    # The cache key carries the module-env fingerprint, so a changed constant is a MISS, never a stale
    # hit. I0 is a linear multiplier on the direct flux -> doubling it must double S (not return the
    # cached value), and restoring it must return the original bit-for-bit.
    x = g.x_of_theta(np.radians(np.array([30, 45, 50], float)))
    s0 = g.day_mean_flux(x)
    orig = C.I0_OVERHEAD
    try:
        C.I0_OVERHEAD = orig * 2.0
        s1 = g.day_mean_flux(x)
    finally:
        C.I0_OVERHEAD = orig
    assert np.allclose(s1, 2.0 * s0)
    assert np.array_equal(g.day_mean_flux(x), s0)                     # restored constant -> original value


def test_day_mean_diffuse_flux_vectorized_equals_scalar_reference():
    # The vectorized+cached day_mean_diffuse_flux MUST reproduce the reference loop of
    # day_mean_diffuse_flux_scalar BIT-FOR-BIT (the C7 diffuse vectorization must not move any number).
    x = g.x_of_theta(np.radians(np.array([0, 10, 20, 30, 45, 50, 55, 60, 70, 78, 85], float)))
    ref = np.array([g.day_mean_diffuse_flux_scalar(float(xi)) for xi in x])   # the old per-x loop, verbatim
    vec = g.day_mean_diffuse_flux(x)
    assert np.array_equal(vec, ref)                                  # exact, not just close
    # scalar-collapse contract preserved: a size-1 input returns a python float == the scalar path
    s1 = g.day_mean_diffuse_flux(np.array([float(x[4])]))
    assert isinstance(s1, float) and s1 == ref[4]
    # explicit non-default scattering params are also vectorized bit-for-bit against the scalar loop
    ref_p = np.array([g.day_mean_diffuse_flux_scalar(float(xi), omega0=0.6, g_haze=0.3) for xi in x])
    assert np.array_equal(g.day_mean_diffuse_flux(x, omega0=0.6, g_haze=0.3), ref_p)


def test_day_mean_diffuse_flux_vectorized_matches_scalar_under_graded_shade(monkeypatch):
    # The default-world tests above exercise the binary _visible_fraction collimator branch; PRODUCTION
    # runs the GRADED-SHADE branch (COSMO_SHADE_FILE loaded), where _diffuse_one_sun/_flux_one_sun route
    # through _shade_factor on a 2-D (nx, n) slant-distance array. Pin that branch bit-for-bit too by
    # monkeypatching a synthetic _SHADE table (the env-key carries its bytes, so this is a clean cache miss).
    monkeypatch.setattr(C, "COLLIMATOR", True)
    alpha_deg = np.array([0.0, 30.0, 60.0, 85.0])
    t_eff = np.array([1.2, 1.0, 0.5, 0.0])                            # brighter overhead, dimming to the rim
    monkeypatch.setattr(g, "_SHADE", (alpha_deg, t_eff))
    x = g.x_of_theta(np.radians(np.array([0, 20, 40, 55, 70], float)))
    ref = np.array([g.day_mean_diffuse_flux_scalar(float(xi)) for xi in x])   # scalar loop, same shade table
    assert np.array_equal(g.day_mean_diffuse_flux(x), ref)           # graded-shade broadcast is exact too
    assert np.all(ref >= 0)


def test_day_mean_diffuse_flux_cache_hits_and_returns_fresh_copy():
    # Re-requesting the SAME grid must hit the memo, yet each call must hand back a FRESH, writeable
    # array so a caller can mutate it without corrupting the shared cache entry.
    g._day_mean_diffuse_flux_cached.cache_clear()
    x = g.x_of_theta(np.radians(np.array([20, 40, 55, 70], float)))
    a = g.day_mean_diffuse_flux(x)
    info_after_first = g._day_mean_diffuse_flux_cached.cache_info()
    assert info_after_first.misses == 1 and info_after_first.hits == 0
    b = g.day_mean_diffuse_flux(x)
    assert g._day_mean_diffuse_flux_cached.cache_info().hits == 1     # second identical call HIT the cache
    assert np.array_equal(a, b)
    assert a.flags.writeable and a is not b                           # fresh writeable copies, not aliases
    a[:] = -1.0                                                       # mutate the returned array...
    c = g.day_mean_diffuse_flux(x)
    assert np.array_equal(c, b)                                       # ...cache entry is untouched


def test_day_mean_diffuse_flux_cache_not_stale_under_constant_change():
    # The cache key carries the module-env fingerprint AND the resolved omega0/g_haze, so a changed
    # constant is a MISS, never a stale hit. omega0 is a linear multiplier on the diffuse flux -> halving
    # C.OMEGA0_HAZE must halve S (not return the cached value), and restoring it recovers the original.
    x = g.x_of_theta(np.radians(np.array([40, 55, 60], float)))
    s0 = g.day_mean_diffuse_flux(x)
    orig = C.OMEGA0_HAZE
    try:
        C.OMEGA0_HAZE = 0.5 * orig
        s1 = g.day_mean_diffuse_flux(x)
    finally:
        C.OMEGA0_HAZE = orig
    assert np.allclose(s1, 0.5 * s0)
    assert np.array_equal(g.day_mean_diffuse_flux(x), s0)            # restored constant -> original value


def test_airless_theta50_recovers_278K(monkeypatch):
    # Sanity on the fiducial amplitude: a black, airless, no-transport surface at theta=50 sits at
    # ~278 K under the day-mean flux -- i.e. I0 reproduces the *forcing* behind the legacy 5 C value,
    # while leaving the *predicted* temperature (with albedo/greenhouse/transport) free. Collimator-off
    # limit (the fiducial-I0 reference; the collimator brightens the ribbon, a separate finding).
    _collimator_off(monkeypatch)
    S50 = g.day_mean_flux_scalar(float(g.x_of_theta(np.radians(50))))
    T = (S50 / C.SIGMA_SB) ** 0.25
    assert T == pytest.approx(278.0, abs=2.0)


# ======================================================================================
# Stage 5 (CLEARSKY_SW_PLAN): the haze forward-scatter diffuse-down day-mean flux.
# ======================================================================================
def test_haze_diffuse_flux_positive_and_rises_relative_to_direct_poleward():
    # The recovered forward-scattered haze flux is positive everywhere, and its RATIO to the (pure-loss)
    # direct beam GROWS poleward -- higher airmass -> thicker slant haze -> more of the dropped light is
    # forward-scattered to the ground (a hazy low sun is diffuse-dominated).
    ths = np.array([20.0, 35.0, 50.0, 60.0])
    x = g.x_of_theta(np.radians(ths))
    direct = g.day_mean_flux(x)
    diffuse = g.day_mean_diffuse_flux(x)
    assert np.all(diffuse > 0)
    ratio = diffuse / direct
    assert np.all(np.diff(ratio) > 0)                          # diffuse/direct rises with latitude


def test_haze_per_sun_energy_budget_telescopes():
    # The single-sun haze budget must close EXACTLY: direct (through-beam) + forward-diffuse + backscatter +
    # haze-absorption == the geometric reference flux F_ref = I0*(L/r)^3 (no extra extinction created or lost).
    omega0, gh = C.OMEGA0_HAZE, C.G_HAZE
    b_fwd = 0.5 * (1.0 + gh)
    for th in (10.0, 40.0, 65.0):
        r = float(g.slant_distance(g.x_of_theta(np.radians(th)), 0.0))
        F_ref = C.I0_OVERHEAD * (C.L_SUN / r) ** 3
        F_dir = float(g._flux_one_sun(r))
        F_dif = float(g._diffuse_one_sun(r))
        extinguished = F_ref - F_dir
        F_back = omega0 * (1.0 - b_fwd) * extinguished         # backscattered to space (lost)
        F_abs = (1.0 - omega0) * extinguished                 # absorbed in the haze (lost)
        assert F_dir + F_dif + F_back + F_abs == pytest.approx(F_ref, rel=1e-12)
        assert F_dir < F_dir + F_dif < F_ref                  # recovery, with a residual net loss


def test_haze_diffuse_recovers_most_of_the_slant_loss_day_mean():
    # The day-mean direct+diffuse sits between the pure-loss direct beam and the no-haze day-mean (computed
    # with the SAME clock -- only the slant haze removed from the beam). A scattering-dominated haze (omega0
    # high, forward-peaked) recovers MOST of the dropped light, so the NET loss is small.
    x = float(g.x_of_theta(np.radians(50.0)))
    direct = g.day_mean_flux_scalar(x)
    diffuse = g.day_mean_diffuse_flux_scalar(x)
    A = np.sqrt(C.L_SUN**2 + x**2)
    u = np.linspace(0.0, 20.0, 6000); e = A * np.sinh(u); de_du = A * np.cosh(u)
    r = g.slant_distance(x, e)
    F_ref = C.I0_OVERHEAD * (C.L_SUN / r) ** 3                 # geometric (no-haze) per-sun flux
    nohaze = 2.0 * np.trapezoid(F_ref * de_du, u) / g.CLOCK.spacing    # same clock, slant haze removed
    assert direct < direct + diffuse < nohaze                 # recovery, but a residual net loss
    assert (direct + diffuse) > 0.7 * nohaze                  # scattering-dominated haze: small NET loss


def test_haze_split_leaves_direct_flux_and_clock_untouched():
    # Stage 5 changes ONLY the surface-flux (diffuse) role: the scattering params move the diffuse flux but
    # NEVER the direct beam or the midnight clock (separate code paths -- the load-bearing requirement).
    x = g.x_of_theta(np.radians(np.array([0.0, 30.0, 50.0, 70.0])))
    direct0 = g.day_mean_flux(x).copy()
    spacing0, speed0 = g.CLOCK.spacing_km, g.CLOCK.speed_km_s
    d_a = g.day_mean_diffuse_flux(x, omega0=0.9, g_haze=0.7)
    d_b = g.day_mean_diffuse_flux(x, omega0=0.5, g_haze=0.2)
    assert not np.allclose(d_a, d_b)                           # the scattering params DO move the diffuse
    assert np.array_equal(g.day_mean_flux(x), direct0)        # ... but NOT the direct day-mean flux
    assert g.CLOCK.spacing_km == spacing0 and g.CLOCK.speed_km_s == speed0   # ... nor the clock


# ======================================================================================
# The suns' COLLIMATORS (constants.COLLIMATOR): dark co-moving vertical tubes that gate each sun, so
# night is a geometric zero. NOW ON BY DEFAULT (the accurate worldbuilding geometry; COSMO_COLLIMATOR=0
# for the byte-identical no-gate limit). geometry reads the flag at call time.
# ======================================================================================
def test_collimator_on_is_the_default():
    assert C.COLLIMATOR is True                                   # committed default = the collimator world
    # the I0 re-pin holds the equator day-mean at its no-collimator value even with the thin haze
    assert g.day_mean_flux_scalar(0.0) == pytest.approx(1570, rel=0.02)


def test_collimator_off_limit_is_byte_identical(monkeypatch):
    # The OFF path is still reachable (COSMO_COLLIMATOR=0): no gate, full TAU0_HAZE=1.0, fiducial I0 ->
    # the pre-collimator equator day-mean, and the night is haze-darkened (NOT a geometric zero).
    _collimator_off(monkeypatch)
    assert g.day_mean_flux_scalar(0.0) == pytest.approx(1570, rel=0.02)
    _, S = g.diurnal_flux_cycle(float(g.x_of_theta(np.radians(50.0))), n_steps=20000, n_suns=8)
    assert S.min() > 0.0                                          # haze floor, not the collimator's hard zero


def test_collimator_visible_fraction_is_a_clean_step(monkeypatch):
    # The finite-disk gate: 1 nearly overhead, 0.5 at the cutoff zenith, 0 well beyond, monotone between
    # (the partial-eclipse terminator across the sun's disk).
    monkeypatch.setattr(C, "COLLIMATOR", True)
    r_cut = C.L_SUN / np.cos(C.COLLIMATOR_ZCUT)                   # zenith z == z_cut
    assert float(g._visible_fraction(C.L_SUN * 1.0001)) == pytest.approx(1.0, abs=1e-6)
    assert float(g._visible_fraction(r_cut)) == pytest.approx(0.5, abs=1e-3)
    assert float(g._visible_fraction(r_cut * 5)) == pytest.approx(0.0, abs=1e-9)
    rr = np.linspace(C.L_SUN * 1.001, r_cut * 3, 400)
    assert np.all(np.diff(g._visible_fraction(rr)) <= 1e-12)      # monotone non-increasing


def test_collimator_keeps_the_spacing(monkeypatch):
    # D is KEPT: the clock uses the TAU0_HAZE=1.0 reference (ungated), so rebuilding it with the collimator
    # on gives the same spacing as off (H/R sets the NIGHT LENGTH, not the spacing).
    monkeypatch.setattr(C, "COLLIMATOR", True)
    assert g.SunsClock().spacing_km == pytest.approx(g.CLOCK.spacing_km, rel=1e-9)


def test_collimator_hard_cutoff_latitude(monkeypatch):
    # Beyond theta = z_cut the noon sun never clears the tube -> permanent night -> day-mean flux 0;
    # just inside, the flux is still positive (the frozen polar cap that cools the already-cold high lats).
    monkeypatch.setattr(C, "COLLIMATOR", True)
    zc = np.degrees(C.COLLIMATOR_ZCUT)
    assert float(g.day_mean_flux(g.x_of_theta(np.radians(zc - 3)))) > 0
    assert float(g.day_mean_flux(g.x_of_theta(np.radians(zc + 1)))) == pytest.approx(0.0, abs=1e-3)


def test_collimator_night_is_a_geometric_zero(monkeypatch):
    # The ribbon's diurnal cycle hits exactly 0 (pitch black) for part of the day, and the day is shorter
    # than 24 h -- the leak-free axisymmetric blackout (no haze needed for darkness).
    monkeypatch.setattr(C, "COLLIMATOR", True)
    x50 = float(g.x_of_theta(np.radians(50.0)))
    t, S = g.diurnal_flux_cycle(x50, n_steps=20000, n_suns=8)
    assert S.min() == pytest.approx(0.0, abs=1e-9)               # pitch black at midnight
    daylen = (S > 1e-6 * S.max()).mean() * 24.0
    assert 6.0 < daylen < 18.0                                   # a real night exists (H/R=0.3 -> ~15.4 h day)


def test_collimator_i0_repin_holds_the_equator_and_brightens_the_ribbon(monkeypatch):
    # The I0 re-pin holds the EQUATOR day-mean fixed under the thin haze; the ribbon (where the haze had
    # dimmed the low sun more) BRIGHTENS -- the documented gradient-flattening finding.
    x50 = float(g.x_of_theta(np.radians(50.0)))
    # baseline = the no-collimator world (full haze, fiducial I0)
    _collimator_off(monkeypatch)
    S_eq_off, S_rib_off = g.day_mean_flux_scalar(0.0), g.day_mean_flux_scalar(x50)
    fiducial_I0 = C.I0_OVERHEAD                                   # already de-repinned by _collimator_off
    # the collimator world WITH the I0 re-pin applied
    monkeypatch.setattr(C, "COLLIMATOR", True)
    monkeypatch.setattr(C, "I0_OVERHEAD", fiducial_I0 * C.COLLIMATOR_I0_REPIN)
    assert g.day_mean_flux_scalar(0.0) == pytest.approx(S_eq_off, rel=0.03)   # equator held
    assert g.day_mean_flux_scalar(x50) > 1.2 * S_rib_off                       # ribbon brighter (finding)


def test_haze_diffuse_reads_omega0_at_call_time(monkeypatch):
    # Regression: _diffuse_one_sun / day_mean_diffuse_flux resolve OMEGA0_HAZE / G_HAZE at CALL time, so a
    # monkeypatched constant is honoured (binding the default at def-time would silently ignore the patch).
    # The diffuse is linear in omega0, so halving the constant halves the flux; zeroing it kills it entirely.
    omega0_0 = C.OMEGA0_HAZE
    r = C.L_SUN * 3.0                                          # a low sun: nonzero slant haze -> nonzero diffuse
    base = float(g._diffuse_one_sun(r))
    assert base > 0.0
    monkeypatch.setattr(C, "OMEGA0_HAZE", 0.5 * omega0_0)
    assert float(g._diffuse_one_sun(r)) == pytest.approx(0.5 * base, rel=1e-12)
    x = float(g.x_of_theta(np.radians(55.0)))
    monkeypatch.setattr(C, "OMEGA0_HAZE", 0.0)                 # no single-scattering albedo -> no diffuse at all
    assert float(g.day_mean_diffuse_flux(x)) == pytest.approx(0.0, abs=1e-12)


# ======================================================================================
# The 2-D lantern lens T(alpha, psi) carried INSTANTANEOUSLY in the diurnal-sun tables (2026-07-15).
# The genuine redistributing optic (COSMO_SHADE2D) now flows through diurnal_flux_table / diurnal_flux_cycle
# per sun (via _direct_flux_xe / _diffuse_flux_xe), so a diurnal AD-core run sees the lens's real cycle SHAPE
# -- while default-OFF (_SHADE2D is None) stays BYTE-IDENTICAL to the pre-2026-07-15 per-sun train.
# ======================================================================================
def test_diurnal_flux_table_byte_identical_without_2d_lens():
    # With no 2-D lens loaded, the refactored diurnal_flux_table MUST reproduce the old per-sun train sum
    # (a bare loop of _flux_one_sun / _diffuse_one_sun over the 2*n_suns+1 nearest suns) BIT-FOR-BIT.
    x = g.x_of_theta(np.radians(np.array([0.0, 30.0, 50.0, 65.0])))
    nph, nsuns = 240, 6
    clock = g.CLOCK
    t = (np.arange(nph) + 0.5) * (clock.T_day / nph)
    e_self = clock.speed * t
    for diffuse in (False, True):
        one = g._diffuse_one_sun if diffuse else g._flux_one_sun
        ref = np.zeros((nph, x.size))
        for k in range(-nsuns, nsuns + 1):                       # the pre-refactor formula, verbatim
            r = g.slant_distance(x[None, :], (e_self - k * clock.spacing)[:, None])
            ref += one(r)
        got = g.diurnal_flux_table(x, n_phase=nph, n_suns=nsuns, diffuse=diffuse)
        assert np.array_equal(got, ref)                          # exact, not just close


def test_diurnal_2d_lens_psi_independent_reduces_to_1d_shade(monkeypatch):
    # A psi-INDEPENDENT 2-D lens T(alpha, psi) = T_eff(alpha) is physically the SAME optic as the radial
    # 1-D shade T_eff(alpha). The instantaneous diurnal table must therefore be identical whether the shade
    # is applied as the 1-D _SHADE (via _flux_one_sun's gate) or lifted to a psi-flat _SHADE2D (via the 2-D
    # instantaneous path) -- the mechanism-level statement of "the 2-D lens reduces to the 1-D shade".
    monkeypatch.setattr(C, "COLLIMATOR", True)
    alpha_deg = np.array([0.0, 30.0, 60.0, 85.0])
    t_eff = np.array([1.2, 1.0, 0.5, 0.05])                      # brighter overhead, dimming to the rim
    x = g.x_of_theta(np.radians(np.array([0.0, 30.0, 50.0, 65.0])))
    nph = 288
    # (a) the radial 1-D shade
    monkeypatch.setattr(g, "_SHADE", (alpha_deg, t_eff))
    monkeypatch.setattr(g, "_SHADE2D", None)
    tbl_1d = g.diurnal_flux_table(x, n_phase=nph)
    tbl_1d_diff = g.diurnal_flux_table(x, n_phase=nph, diffuse=True)
    # (b) the SAME shade as a psi-independent 2-D lens (every psi column equal)
    psi_deg = np.array([0.0, 45.0, 90.0])
    Z = np.repeat(t_eff[:, None], psi_deg.size, axis=1)          # (n_alpha, n_psi), psi-independent
    monkeypatch.setattr(g, "_SHADE2D", (alpha_deg, psi_deg, Z))
    tbl_2d = g.diurnal_flux_table(x, n_phase=nph)
    tbl_2d_diff = g.diurnal_flux_table(x, n_phase=nph, diffuse=True)
    assert np.allclose(tbl_2d, tbl_1d, rtol=1e-9, atol=1e-9)     # instantaneously identical (direct)
    assert np.allclose(tbl_2d_diff, tbl_1d_diff, rtol=1e-9, atol=1e-9)   # ... and diffuse


def test_factorized_2d_lens_recovers_target_daymean_and_reshapes_the_day(monkeypatch):
    # The factorization T(alpha,psi) = S_target(x) * psi_hat(e) / F_bare(r) with a NORMALIZED day shape
    # psi_hat [(2/D) int psi_hat de = 1] delivers day-mean S_target(x) for ANY psi_hat -- the construction
    # _lens_2d_canonical_design.py uses to REDUCE TO the canonical S(theta) while freely choosing the day
    # SHAPE. Verify it end-to-end through the real geometry path: (1) day_mean_flux+diffuse under the lens
    # recovers S_target, (2) the diurnal table carries the lens (phase-mean == day-mean, gain ~1), and (3)
    # the lens genuinely reshapes the day (the table differs from the un-lensed one at the same day-mean).
    from flatworld._shade_design import _one_sun_total
    monkeypatch.setattr(C, "COLLIMATOR", True)
    monkeypatch.setattr(g, "_SHADE", (np.array([0.0, 90.0]), np.array([1.0, 1.0])))  # uniform T=1 (ungated, smooth)
    monkeypatch.setattr(g, "_SHADE2D", None)
    L = C.L_SUN; clock = g.CLOCK; D = clock.spacing; v = clock.speed
    th = np.linspace(1e-3, 84.9, 800); xg = L * np.tan(np.radians(th))
    S_target = np.asarray(g.day_mean_flux(xg)) + np.asarray(g.day_mean_diffuse_flux(xg))  # the OFF-path total
    S_of_x = lambda xq: np.interp(np.abs(xq), xg, S_target)
    # a normalized Gaussian day shape psi_hat(e): (2/D) int_0^inf exp(-(e/w)^2) de == 1
    w = v * 4.0 * 3600.0                                         # ~4-h Gaussian half-width (a real day, not a plateau)
    norm = (2.0 / D) * (w * np.sqrt(np.pi) / 2.0)
    psi_hat = lambda e: np.exp(-(np.asarray(e, float) / w) ** 2) / norm
    alpha = np.linspace(0.05, 88.5, 220); psi = np.linspace(0.0, 90.0, 140)
    A, P = np.meshgrid(np.radians(alpha), np.radians(psi), indexing="ij")
    x_g = L * np.tan(A) * np.sin(P); e_g = L * np.tan(A) * np.cos(P); r_g = L / np.cos(A)
    T2d = np.clip(S_of_x(x_g) * psi_hat(e_g) / np.maximum(_one_sun_total(r_g), 1e-30), 0.0, None)
    monkeypatch.setattr(g, "_SHADE2D", (alpha, psi, T2d))
    # (1) day-mean total under the lens reduces to the target
    th_chk = np.array([2.0, 15.0, 30.0, 45.0, 55.0]); x_chk = L * np.tan(np.radians(th_chk))
    S_lens = np.asarray(g.day_mean_flux(x_chk)) + np.asarray(g.day_mean_diffuse_flux(x_chk))
    assert np.allclose(S_lens, S_of_x(x_chk), rtol=0.03)         # reduces to S_target (quadrature/interp)
    # (2) the diurnal DIRECT table's phase-mean matches the day-mean direct flux (AD-core renorm gain ~1)
    tbl = g.diurnal_flux_table(x_chk, n_phase=288, n_suns=8)
    assert np.allclose(tbl.mean(axis=0), np.atleast_1d(g.day_mean_flux(x_chk)), rtol=0.05)
    # (3) the lens genuinely reshapes the diurnal cycle vs the un-lensed table
    monkeypatch.setattr(g, "_SHADE2D", None)
    tbl_off = g.diurnal_flux_table(x_chk, n_phase=288, n_suns=8)
    assert not np.allclose(tbl, tbl_off, rtol=0.1)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
