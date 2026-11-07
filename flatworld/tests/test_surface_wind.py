"""surface_wind: the 10-m reduction's invariants (form-level; the values are documented model-soft)."""
import numpy as np
import pytest

from flatworld import surface_wind as sw
from flatworld import classify


def test_u10_factor_bounds_and_monotonicity():
    # factor strictly in (0,1) for physical z0, and rougher surface -> stronger reduction
    z0 = np.array([sw.Z0_SEA, sw.Z0_ICE, 0.01, 0.05, 0.8, 1.5])
    f = sw.u10_factor(z0)
    assert np.all(f > 0.0) and np.all(f < 1.0)
    assert np.all(np.diff(f) < 0.0)


def test_u10_factor_rejects_z0_at_reference_height():
    with pytest.raises(ValueError):
        sw.u10_factor(np.array([12.0]))


def test_u10_factor_shallow_bl_keeps_match_above_reference():
    # tiny h_bl: z_sl floors at 3*z_ref, factor stays sane (<1) rather than blowing past 1
    f = sw.u10_factor(np.array([0.05]), h_bl=50.0)
    assert 0.0 < float(f[0]) < 1.0


def test_two_meter_wind_below_ten_meter():
    z0 = np.array([3e-3, 0.03, 0.8])
    f2 = sw.u10_factor(z0, z_ref=2.0)
    f10 = sw.u10_factor(z0, z_ref=10.0)
    assert np.all(f2 < f10)
    # open-ground 2-m/10-m ratio ~0.7-0.8; tall vegetation much lower
    ratio = f2 / f10
    assert 0.7 < ratio[0] < 0.85 and ratio[2] < 0.5


def test_orographic_z0_lettau_and_cap():
    # moderate hills: 200 m relief over 10 km -> z0_eff = 0.5*200^2/1e4 = 2 m
    assert sw.orographic_z0(200.0, 1e4) == pytest.approx(2.0)
    # rugged mountains hit the literature cap, not Lettau's extrapolation
    assert sw.orographic_z0(800.0, 8e3) == pytest.approx(5.0)
    # smooth plains: effectively no orographic addition
    assert sw.orographic_z0(20.0, 2e4) < 0.02


def test_roughness_map_sea_land_and_koppen_classes():
    land = np.array([[False, True, True, True]])
    kop = np.array([[0, classify.IDX["BWh"], classify.IDX["Af"], classify.IDX["EF"]]])
    z0 = sw.roughness_map(land, koppen=kop)
    assert z0[0, 0] == sw.Z0_SEA
    assert z0[0, 1] == sw.KOPPEN_Z0["BWh"]
    assert z0[0, 2] == sw.KOPPEN_Z0["Af"]
    assert z0[0, 3] == sw.Z0_ICE
    # no koppen -> default land roughness
    z0d = sw.roughness_map(land)
    assert np.all(z0d[0, 1:] == sw.Z0_LAND_DEFAULT)


def test_night_factor_sea_unchanged_land_graded():
    land = np.array([[False, True, True]])
    dT = np.array([[12.0, 0.0, 12.0]])
    f = sw.night_factor(land, dT)
    assert f[0, 0] == 1.0                      # sea: no decoupling regardless of dT
    assert f[0, 1] == 1.0                      # calm-swing land: still coupled
    assert f[0, 2] == pytest.approx(0.25)      # strong-swing land: full decoupling floor


def test_bulk_flux_wind_level_to_screen_reduction():
    """The GEO-8 level->screen bridge: its three legs pinned against hand-computed values.

    (a) SEA cells get the pure daytime log-layer value (no nocturnal decoupling, no swing).
    (b) SATURATED-SWING LAND gets day * (w + (1-w)*night_floor) -- the day/night arithmetic mean of
        the day wind and the fully-decoupled night wind.
    (c) The z1 <= z_ref short-circuit makes the log-layer factor exactly 1, so the reduction is
        carried entirely by the decoupling/averaging (the actual canonical case: z1 = 20 m > 10 m,
        but the guard must be exact when a caller passes a level at/below screen height).
    """
    land = np.array([[False, True]])
    z0 = np.array([[sw.Z0_SEA, 0.01]])
    U = np.array([[10.0, 10.0]])
    dT = np.array([[0.0, 99.0]])               # sea: no swing; land: saturates the decoupling ramp
    nf, w = 0.25, 0.5

    f10 = sw.u10_factor(z0)                     # the daytime log-layer factor, per cell
    out = sw.bulk_flux_wind(U, land, z0, dT, night_floor=nf, day_fraction=w)

    # (a) sea: night_factor is 1 over sea, so day == night == day wind; the average is the day wind.
    assert out[0, 0] == pytest.approx(10.0 * f10[0, 0])
    # (b) saturated-swing land: night = day * night_floor, mean = day*(w + (1-w)*nf).
    day_land = 10.0 * f10[0, 1]
    assert out[0, 1] == pytest.approx(day_land * (w + (1.0 - w) * nf))

    # (c) z1 <= z_ref short-circuits the log layer to factor 1: the sea cell then returns U exactly,
    #     and the land cell returns only the decoupling/averaging reduction of U.
    out_sc = sw.bulk_flux_wind(U, land, z0, dT, z1=10.0, night_floor=nf, day_fraction=w)
    assert out_sc[0, 0] == pytest.approx(10.0)
    assert out_sc[0, 1] == pytest.approx(10.0 * (w + (1.0 - w) * nf))


def test_shelter_factor_flat_terrain_is_unsheltered():
    nx, ne = 20, 30
    h = np.zeros((nx, ne))
    u = np.full((nx, ne), -20.0); v = np.zeros((nx, ne))
    x = np.arange(nx) * 50e3; e = np.arange(ne) * 50e3
    s = sw.shelter_factor(h, u, v, x, e)
    assert np.allclose(s, 1.0)


def test_shelter_factor_ridge_shelters_its_lee_only():
    # wind blows toward -x (equatorward, u<0): UPWIND of a cell is +x (poleward). A ridge at
    # ix=12 tall enough to BLOCK the flow (M = N*H/U = 0.011*3000/5 = 6.6 > M_BLOCK) must shelter
    # the ground in its lee out to the WAKE footprint (150 km = 3 cells here) and not the cells
    # poleward (ix>12), whose upwind ray runs away from the ridge.  Farther equatorward still, the
    # wake has run out: under the retired 300 km gate the whole ray counted as "in the lee", which is
    # the defect DOC-34 closed.  The wake deliberately reaches farther than the windstorm does --
    # WAKE_FOOTPRINT_KM against LEE_FOOTPRINT_KM (PR #731 review, finding 2).
    nx, ne = 24, 8
    h = np.zeros((nx, ne)); h[12, :] = 3000.0
    u = np.full((nx, ne), -5.0); v = np.zeros((nx, ne))
    x = np.arange(nx) * 50e3; e = np.arange(ne) * 50e3
    s = sw.shelter_factor(h, u, v, x, e, L_km=300.0)
    assert np.all(s[11, :] < 0.7)                       # the foot of the lee slope
    assert np.all(s[9, :] < 0.7)                        # 150 km downwind: still inside the wake
    assert np.allclose(s[:8, :], 1.0)                   # beyond the wake footprint
    assert np.allclose(s[14:20, :], 1.0)                # windward
    assert np.all(s > sw.SHELTER_FLOOR - 1e-12) and np.all(s <= 1.0 + 1e-12)
    # ...and the same ridge under a wind fast enough to ride over it (M ~ 1.1) shelters NOTHING:
    # that is the windstorm case, the other branch's business
    s_fast = sw.shelter_factor(h, np.full((nx, ne), -30.0), v, x, e, L_km=300.0)
    assert np.allclose(s_fast, 1.0)


def test_attach_shelter_masks_and_summary():
    nx, ne = 24, 8
    h = np.zeros((nx, ne)); h[12, :] = 3000.0
    land = np.ones((nx, ne), bool)
    dC = dict(u=np.full((nx, ne), -10.0), v=np.zeros((nx, ne)), land=land,   # M = 3.3: blocked
              dT_air_C=np.full((nx, ne), 4.0), theta_deg=np.linspace(44, 58, nx),
              T_map_C=np.full((nx, ne), 15.0))
    sw.attach(dC)
    topo = dict(height=h, x=np.arange(nx) * 50e3, e=np.arange(ne) * 50e3)
    calm = 0.85 * float(dC["u10_day_ms"][20, 0])     # between the sheltered and the open 10-m wind
    summ = sw.attach_shelter(dC, topo, calm_ms=calm)
    assert dC["shelter"].shape == (nx, ne)
    assert np.all(dC["u10_day_sheltered_ms"] <= dC["u10_day_ms"] + 1e-12)
    # the cell at the foot of the lee falls under the calm threshold; the open plain does not
    assert dC["calm_habitable"][11, 0] and not dC["calm_habitable"][20, 0]
    assert 0.0 < summ["calm_of_habitable"] < 1.0


# ---- issue #114: terrain wind AMPLIFICATION (downslope windstorms / gap jets / foehn) ----

def test_mountain_nondim_height_scaling():
    # M = N H / U: taller ridge & stronger stratification raise M; faster wind lowers it
    assert sw.mountain_nondim_height(3000.0, 30.0, N=0.011) == pytest.approx(0.011 * 3000 / 30)
    assert sw.mountain_nondim_height(6000.0, 30.0) > sw.mountain_nondim_height(3000.0, 30.0)
    assert sw.mountain_nondim_height(3000.0, 60.0) < sw.mountain_nondim_height(3000.0, 30.0)


def test_amplification_from_M_peaks_in_critical_band():
    # factor = 1 (no amplification) for free-flow (M->0) and blocked (M->inf); peaks near M_crit
    a_lo = sw.amplification_from_M(0.02)
    a_crit = sw.amplification_from_M(1.0)
    a_hi = sw.amplification_from_M(50.0)
    assert a_crit == pytest.approx(2.5)            # default ceiling at the critical mountain height
    assert a_lo < 1.05 and a_hi < 1.05
    assert a_crit > a_lo and a_crit > a_hi
    # bounded by [1, amp_max] everywhere
    M = np.linspace(0.01, 100, 500)
    amp = sw.amplification_from_M(M)
    assert np.all(amp <= 2.5 + 1e-9) and np.all(amp >= 1.0 - 1e-9)


def test_orographic_amplification_capped():
    # sub-grid relief at the critical height approaches the ceiling, not beyond
    a = sw.orographic_amplification(3000.0, 30.0, N=0.011, amp_cap=2.5)
    assert a == pytest.approx(2.5, abs=0.2)
    # a custom lower ceiling caps it
    assert sw.orographic_amplification(3000.0, 30.0, amp_cap=1.8) <= 1.8 + 1e-9


def test_terrain_amplification_flags_lee_of_ridge_only():
    # cross-wind ridge at the critical height (M = 0.011*3000/30 = 1.1); wind toward -x; the LEE
    # (equatorward, ix<ridge) is the windstorm side -- but only within the footprint
    nx, ne = 24, 8
    h = np.zeros((nx, ne)); h[12, :] = 3000.0
    u = np.full((nx, ne), -30.0); v = np.zeros((nx, ne))
    x = np.arange(nx) * 50e3; e = np.arange(ne) * 50e3
    amp, prone = sw.terrain_amplification(h, u, v, x, e, N=0.011)
    assert np.all(amp >= 1.0 - 1e-9)
    assert prone[11, 0] and amp[11, 0] > 1.3            # the foot of the lee slope
    assert not prone[10, 0] and amp[10, 0] == pytest.approx(1.0)  # 100 km out: beyond the footprint
    assert not prone[18, 0] and amp[18, 0] == pytest.approx(1.0)  # windward: ray runs away
    # a wider footprint reaches the plain cell too -- the gate IS the footprint
    amp_w, prone_w = sw.terrain_amplification(h, u, v, x, e, N=0.011, footprint_km=300.0)
    assert prone_w[10, 0] and prone_w[8, 0]
    # flat terrain is never amplification-prone
    flat_amp, flat_prone = sw.terrain_amplification(np.zeros((nx, ne)), u, v, x, e)
    assert np.allclose(flat_amp, 1.0) and not flat_prone.any()


def test_terrain_amplification_consumes_111_froude():
    # given a per-cell hydraulic Froude (#111), transcritical (Fr~1) lee cells flag prone,
    # subcritical (Fr<<1) lees do not -- the criterion that #111 makes free
    nx, ne = 24, 8
    h = np.zeros((nx, ne)); h[12, :] = 3000.0
    u = np.full((nx, ne), -30.0); v = np.zeros((nx, ne))
    x = np.arange(nx) * 50e3; e = np.arange(ne) * 50e3
    _, prone_sub = sw.terrain_amplification(h, u, v, x, e, froude=np.full((nx, ne), 0.1))
    _, prone_crit = sw.terrain_amplification(h, u, v, x, e, froude=np.full((nx, ne), 1.0))
    assert prone_crit[11, 0] and not prone_sub[11, 0]
    # ...and the froude path feeds the SHELTER branch too (one regime, both branches): Fr = 0.1 is
    # M = 10, blocked, so the same lee cell is sheltered there and not at Fr = 1
    s_sub = sw.shelter_factor(h, u, v, x, e, froude=np.full((nx, ne), 0.1))
    s_crit = sw.shelter_factor(h, u, v, x, e, froude=np.full((nx, ne), 1.0))
    assert s_sub[11, 0] < 0.7 and s_crit[11, 0] == pytest.approx(1.0)


def test_attach_amplification_excludes_windstorm_cells_from_calm_habitable():
    nx, ne = 24, 8
    h = np.zeros((nx, ne)); h[12, :] = 3000.0
    land = np.ones((nx, ne), bool)
    dC = dict(u=np.full((nx, ne), -30.0), v=np.zeros((nx, ne)), land=land,
              dT_air_C=np.full((nx, ne), 4.0), theta_deg=np.linspace(44, 58, nx),
              T_map_C=np.full((nx, ne), 15.0))
    sw.attach(dC)
    topo = dict(height=h, x=np.arange(nx) * 50e3, e=np.arange(ne) * 50e3)
    sw.attach_shelter(dC, topo, calm_ms=50.0)          # high threshold: shelter alone keeps lee 'calm'
    assert dC["calm_habitable"][11, 0]                 # before amplification the lee counts as calm
    summ = sw.attach_amplification(dC, topo)
    assert dC["amplification"].shape == (nx, ne)
    assert dC["amplification_prone"][11, 0]
    assert not dC["calm_habitable"][11, 0]             # windstorm alley carved out of calm-habitable
    assert 0.0 < summ["amplification_prone_land_frac"] <= 1.0
    # with shelter attached the unified factor rides along, and it is exactly the product
    assert np.allclose(dC["terrain_factor"], dC["shelter"] * dC["amplification"])
    assert np.allclose(dC["u10_day_terrain_ms"], dC["u10_day_ms"] * dC["terrain_factor"])


# ---- ROADMAP DOC-34: the two branches are ONE regime law on ONE M ----

def test_shelter_from_M_is_the_blocked_branch():
    # exactly 1 up to the onset (so it adds nothing at the windstorm peak), then Sheppard's blocked
    # fraction takes the wind down toward the wake floor, monotonically
    M = np.array([0.01, 0.5, 1.0, sw.M_BLOCK, 1.5 * sw.M_BLOCK, 4.0, 10.0, 1e4])
    s = sw.shelter_from_M(M)
    assert np.allclose(s[:4], 1.0)
    assert s[4] == pytest.approx(1.0 - (1.0 - sw.SHELTER_FLOOR) * (1.0 - 1.0 / 1.5))
    assert np.all(np.diff(s[3:]) <= 1e-12)
    assert s[-1] == pytest.approx(sw.SHELTER_FLOOR, abs=1e-3)
    assert np.all(s > sw.SHELTER_FLOOR - 1e-12) and np.all(s <= 1.0)


def test_terrain_factor_from_M_is_one_hump_then_one_trough():
    """The composed law must not double-count: the windstorm bump and the blocked branch are
    alternatives sequenced in M, so g(M) is 1 at M << 1, rises to exactly amp_max at M_crit (no
    shelter there at all), falls back through 1 once, and settles toward the wake floor."""
    assert sw.terrain_factor_from_M(1.0) == pytest.approx(sw.AMP_MAX)          # no double count
    assert sw.terrain_factor_from_M(1e-3) == pytest.approx(1.0, abs=1e-6)
    assert sw.terrain_factor_from_M(1e4) == pytest.approx(sw.SHELTER_FLOOR, abs=1e-3)
    M = np.geomspace(1e-3, 1e4, 4000)
    g = sw.terrain_factor_from_M(M)
    k = int(np.argmax(g))
    assert np.all(np.diff(g[:k]) >= -1e-9) and np.all(np.diff(g[k:]) <= 1e-9), "not single-humped"
    # one crossing of 1 on the way down, and it sits past the blocked onset (the bump runs out first)
    below = M[k:][g[k:] < 1.0]
    assert below.min() > sw.M_BLOCK
    assert np.all(g[M >= 4.0] < 1.0) and np.all(g[(M > 0.5) & (M < 2.0)] > 1.0)


def test_the_branches_are_disjoint_by_construction_not_by_luck():
    """The defect DOC-34 closed -- stated as what it actually is.

    PR #731's review (finding 3) is right that "no cell is both sheltered below 0.7 and amplified
    above 1.3" is NOT an archive measurement: it is a fact about the constants.  `shelter < 0.7`
    needs M >~ 3.2, and there `amplification_from_M <= 1.283 < 1.3` for ANY footprint weight (a
    smaller w only pushes both factors toward 1).  So the honest test is the analytic one -- assert
    the separation over the whole regime axis, where it CAN fail if a constant moves -- and then
    confirm the field code inherits it.  Under the retired law the same pair of thresholds fired
    together on 96.9 % of the sheltered land, which is why the property is worth pinning at all."""
    M = np.geomspace(1e-3, 1e4, 20000)
    s_of_M = sw.shelter_from_M(M)
    a_of_M = sw.amplification_from_M(M)
    assert not ((s_of_M < 0.7) & (a_of_M > 1.3)).any(), "the two branches now overlap in M"
    m_shelter = M[s_of_M < 0.7].min()
    assert m_shelter > 3.0 and a_of_M[M >= m_shelter].max() < 1.3

    rng = np.random.default_rng(3)
    nx, ne = 60, 12
    h = np.zeros((nx, ne))
    for ix in range(4, nx, 6):                                  # a ridge every 300 km, heights 300-6000 m
        h[ix, :] = rng.uniform(300.0, 6000.0)
    u = -rng.uniform(3.0, 40.0, (nx, ne)); v = np.zeros((nx, ne))
    x = np.arange(nx) * 50e3; e = np.arange(ne) * 50e3
    tf = sw.terrain_factors(h, u, v, x, e)
    s, a = tf["shelter"], tf["amplification"]
    assert (s < 0.7).any() and (a > 1.3).any(), "fixture: both regimes must occur"
    assert not ((s < 0.7) & (a > 1.3)).any()
    assert np.allclose(tf["factor"], s * a)
    # both footprint weights are true fractions: 0 where nothing rises upwind, 1 at a ridge foot,
    # and the wake's is never smaller than the windstorm's (it reaches farther)
    for key in ("w_lee", "w_wake"):
        assert np.all((tf[key] >= 0.0) & (tf[key] <= 1.0))
        assert np.allclose(tf[key][tf["relief"] == 0.0], 0.0)
        assert np.allclose(tf[key][3, :], 1.0)                  # the foot of the ridge at ix=4
    assert np.all(tf["w_wake"] >= tf["w_lee"] - 1e-12)


def test_lee_footprint_is_a_fraction_of_the_obstacle():
    """A long gentle lee slope is reached in proportion to how much of it lies within the footprint;
    a plain a footprint beyond the foot sees nothing.  Fine grid so the footprint spans cells."""
    nx, ne = 120, 4
    x = np.arange(nx) * 10e3; e = np.arange(ne) * 10e3           # 10 km cells, footprint = 5 cells
    h = np.zeros((nx, ne))
    h[60:80, :] = np.linspace(0.0, 2000.0, 20)[:, None]           # a 200 km lee slope rising to +x
    h[80:, :] = 2000.0
    u = np.full((nx, ne), -10.0); v = np.zeros((nx, ne))          # wind toward -x: lee is ix < 80
    M, w_lee, w_wake, relief = sw.lee_regime(h, u, v, x, e)
    assert relief[59, 0] == pytest.approx(2000.0)                 # the foot sees the whole obstacle...
    # ...but only the part within the footprint is in reach: ~50 of the slope's 200 km, read at the
    # far ray's own sample distances (the shared quadrature puts its last sub-50 km sample at ~37 km,
    # so the fraction lands a little under the 50/200 an exact cutoff would give)
    assert 0.12 < w_lee[59, 0] < 0.30
    assert w_lee[70, 0] > w_lee[59, 0]                            # halfway up: more of it near
    assert w_lee[50, 0] < 0.05                                    # the plain: out of the windstorm's reach
    # the wake reaches three times farther, so the same ground sees more of the obstacle
    assert w_wake[59, 0] > w_lee[59, 0] and w_wake[50, 0] > w_lee[50, 0]
    # ONE quadrature, so w is a fraction by construction and never needs clipping (review finding 5)
    assert np.all(w_lee <= 1.0 + 1e-12) and np.all(w_wake <= 1.0 + 1e-12)
    # and the regime is the obstacle's, not the near slope's: M reads the 2000 m crest from every
    # cell whose 300 km ray reaches it (the crest sits at ix=80, so ix >= 50)
    assert np.allclose(M[52:60, 0], 0.011 * 2000.0 / 10.0)


def test_sub_footprint_grid_reads_the_adjacent_cell():
    # on a grid coarser than the footprint the near ray is floored at one cell, so a ridge in the
    # next cell upwind is fully in reach (w = 1) and the cell beyond it is not (w = 0)
    nx, ne = 12, 4
    h = np.zeros((nx, ne)); h[6, :] = 1500.0
    u = np.full((nx, ne), -10.0); v = np.zeros((nx, ne))
    x = np.arange(nx) * 120e3; e = np.arange(ne) * 120e3          # 120 km cells > 50 km footprint
    _, w_lee, _, _ = sw.lee_regime(h, u, v, x, e)
    assert w_lee[5, 0] == pytest.approx(1.0) and w_lee[4, 0] < 1e-9


# ---- ROADMAP DOC-45: the THIRD branch -- the two-sided escape barrier (enclosure) ----

def _basin_and_lee(n_ridge=1500.0, cell_m=10e3):
    """One fixture carrying both cases the branch has to tell apart, on a 10 km grid with the wind
    blowing toward -x (so 'downwind' is decreasing ix).

      ix 20-24  a ridge          -> ix 15-19 is its LEE SLOPE, open beyond: enclosure must be 1
      ix 40-44  a ridge, and     -> ix 36-38 is a BASIN FLOOR walled both ways: enclosure must bite
      ix 33-34  a second ridge      (both walls inside the 50 km = 5-cell radius)
    """
    nx, ne = 60, 6
    h = np.zeros((nx, ne))
    h[20:25, :] = n_ridge
    h[40:45, :] = n_ridge
    h[33:35, :] = n_ridge
    u = np.full((nx, ne), -10.0); v = np.zeros((nx, ne))
    x = np.arange(nx) * cell_m; e = np.arange(ne) * cell_m
    return h, u, v, x, e


def test_stagnant_fraction_is_sheppards_blocked_fraction():
    """The escape barrier's only nonlinearity: exactly 0 up to the onset, then 1 - onset/M_enc,
    monotone to 1.  Sheppard's own criterion -- a surface parcel of speed U rises only U/N, so the
    layer below D*(1 - 1/M_enc) can never surmount a barrier of height D."""
    M = np.array([0.0, 0.5, sw.M_ENC_ONSET, 2.0 * sw.M_ENC_ONSET, 10.0 * sw.M_ENC_ONSET, 1e6])
    b = sw.stagnant_fraction(M)
    assert np.allclose(b[:3], 0.0)
    assert b[3] == pytest.approx(0.5) and b[4] == pytest.approx(0.9)
    assert b[-1] == pytest.approx(1.0, abs=1e-5)
    assert np.all(np.diff(b) >= -1e-12) and np.all((b >= 0.0) & (b <= 1.0))


def test_the_enclosure_branch_is_one_on_a_lee_slope_and_bites_on_a_basin_floor():
    """ROADMAP DOC-45, the whole point of the branch: the upwind-ray regime cannot tell a cell at
    the foot of a ridge with OPEN ground beyond it (a foehn lee -- correctly accelerated) from one
    at the foot of a ridge with a SECOND ridge beyond it (a basin floor -- stagnant, the opposite).
    The two-sided escape barrier can, and it is exactly 1 in the first case by construction."""
    h, u, v, x, e = _basin_and_lee()
    tf = sw.terrain_factors(h, u, v, x, e)
    enc, D = tf["enclosure"], tf["D_enc"]

    lee = slice(14, 20)            # the open lee of the ix 20-24 ridge: nothing rises downwind
    assert np.allclose(D[lee, :], 0.0), "a lee slope has no downwind barrier"
    assert np.allclose(enc[lee, :], 1.0), "the branch must not touch a resolved lee slope"
    assert (tf["amplification"][lee, :] > 1.3).any(), "fixture: the lee must carry a windstorm gain"

    floor = slice(36, 39)          # walled by the ix 40-44 ridge upwind and the ix 33-34 one downwind
    assert (D[floor, :] > 0.5 * 1500.0).all(), "the basin floor's barrier is its lower wall"
    assert (enc[floor, :] < 1.0).all(), "the basin floor must be suppressed"
    # the suppression is an OVERRIDE, not a scaling: the composed factor is pulled toward the pool's
    # own residual rather than toward a fraction of the ray law's answer.  It is DELIBERATELY
    # conservative -- Sheppard's criterion is literally a STEP (a floor below the dividing
    # streamline is stagnant, full stop) and the stagnant-DEPTH fraction used as a blend weight
    # under-suppresses near onset, so a 1,500 m basin under a 10 m/s wind reads x1.35 where the ray
    # law alone said x2.10.  Right sign, softened magnitude; the bracket is swept in the instrument.
    g_ray = tf["shelter"] * tf["amplification"]
    assert (tf["factor"][floor, :] < 0.7 * g_ray[floor, :]).all(), "no real suppression"
    assert tf["factor"].min() >= sw.ENCLOSURE_FLOOR - 1e-12, "the blend cannot undershoot the floor"


def test_a_valley_across_the_wind_is_enclosed_and_one_along_it_is_not():
    """The branch reads the escape barrier on the WIND AXIS, because ventilation is done by the
    wind: the same valley is a stagnant trough when it lies across the flow and an open corridor
    when it lies along it.  One geometry, two winds -- the discriminating control."""
    nx, ne = 40, 40
    h = np.zeros((nx, ne))
    h[:, :17] = 1500.0; h[:, 23:] = 1500.0        # walls at both ends of the e axis: a valley in e
    x = np.arange(nx) * 10e3; e = np.arange(ne) * 10e3
    mid = (slice(None), slice(19, 21))            # the floor, both walls inside the 50 km radius

    across = sw.terrain_factors(h, np.zeros((nx, ne)), np.full((nx, ne), 10.0), x, e)
    along = sw.terrain_factors(h, np.full((nx, ne), 10.0), np.zeros((nx, ne)), x, e)
    assert (across["D_enc"][mid] > 750.0).all() and (across["enclosure"][mid] < 1.0).all()
    assert np.allclose(along["D_enc"][mid], 0.0) and np.allclose(along["enclosure"][mid], 1.0)


def test_the_enclosure_depth_does_not_depend_on_the_ray_it_is_cut_from():
    """The box test, in code.  A distance-like metric read out of a finite window can saturate
    against it (root AGENTS.md), so the branch's radius is built not to: ``ray_distances`` windows
    ON the cutoffs, which makes the samples inside the first window the same whether the ray stops
    at the radius or runs on to 300 km.  So the barrier is the SAME array at every ray length -- and
    doubling the radius changes the answer only by finding farther walls, never by re-sampling."""
    h, u, v, x, e = _basin_and_lee()
    D = sw.enclosure_depth(h, u, v, x, e, radius_km=sw.ENCLOSURE_RADIUS_KM)
    for L_km in (sw.ENCLOSURE_RADIUS_KM, 150.0, 300.0, 600.0):
        _, (up,) = sw._upwind_relief_nested(h, u, v, x, e, L_km, (sw.ENCLOSURE_RADIUS_KM,))
        _, (down,) = sw._upwind_relief_nested(h, -u, -v, x, e, L_km, (sw.ENCLOSURE_RADIUS_KM,))
        assert np.array_equal(np.minimum(up, down), D), f"the barrier moved at a {L_km} km ray"
    # What is NOT invariant, and must not be claimed: changing the RADIUS re-samples the window
    # (geomspace(d0, R, n_per) moves with R), so a 100 km reading is not a superset of a 50 km one
    # and the two are separate READINGS, not a nested family.  That is why the instrument's radius
    # sweep shares one edge set, the way its footprint sweep already does.
    wide = sw.enclosure_depth(h, u, v, x, e, radius_km=2.0 * sw.ENCLOSURE_RADIUS_KM)
    assert wide.max() >= D.max() - 1e-9 and not np.array_equal(wide, D)


def test_the_enclosure_branch_off_is_byte_identical_to_the_two_branch_law():
    """The default-OFF gate: ``enclosure_radius_km = 0`` restores the DOC-34 law exactly, and the
    other two branches are untouched whether it runs or not -- the branch reads its own short
    scans and never adds a cutoff to the long ray's quadrature, which would move every sample."""
    h, u, v, x, e = _basin_and_lee()
    on = sw.terrain_factors(h, u, v, x, e)
    off = sw.terrain_factors(h, u, v, x, e, enclosure_radius_km=0.0)
    for key in ("shelter", "amplification", "M", "w_lee", "w_wake", "relief"):
        assert np.array_equal(on[key], off[key]), f"{key} moved when the enclosure branch ran"
    assert np.array_equal(off["enclosure"], np.ones_like(h))
    assert np.array_equal(off["factor"], off["shelter"] * off["amplification"])
    assert np.allclose(on["factor"], on["shelter"] * on["amplification"] * on["enclosure"])
    assert (on["factor"] != off["factor"]).any(), "fixture: the branch must bite somewhere"


def test_the_composed_factor_is_bounded_by_the_two_regimes_over_the_whole_axis():
    """The blend cannot double-count and cannot run away: over the full (M, M_enc) plane the
    composed factor always lies between the ray law's own answer and the pool's residual, so it
    never falls below ENCLOSURE_FLOOR and never exceeds the windstorm ceiling."""
    M = np.geomspace(1e-3, 1e3, 400)[:, None]
    M_enc = np.concatenate([[0.0], np.geomspace(1e-3, 1e3, 399)])[None, :]
    g_ray = sw.terrain_factor_from_M(M) * np.ones_like(M_enc)
    b = sw.stagnant_fraction(M_enc) * np.ones_like(M)
    g = g_ray + b * (sw.ENCLOSURE_FLOOR - g_ray)
    assert np.all(g >= np.minimum(g_ray, sw.ENCLOSURE_FLOOR) - 1e-12)
    assert np.all(g <= np.maximum(g_ray, sw.ENCLOSURE_FLOOR) + 1e-12)
    assert np.all(g >= sw.ENCLOSURE_FLOOR - 1e-12) and np.all(g <= sw.AMP_MAX + 1e-12)
    assert np.allclose(g[:, 0], g_ray[:, 0]), "M_enc = 0 must leave the ray law alone"


def test_attach_amplification_composes_the_enclosure_and_can_key_calm_on_it():
    """The cascade entry point: the composed ``terrain_factor`` is the product of all THREE
    branches, and ``calm_on_terrain`` re-asks the calm question of the composed wind.  Default ON
    (Jake's call on ROADMAP DOC-45's second half); ``calm_on_terrain=False`` keeps the
    shelter-then-carve mask for comparison, and the re-ask inherits ``attach_shelter``'s threshold.

    The fixture is a DEEP walled basin -- 4,000 m ridges 50 km apart -- because that is the case
    the two maskings disagree on: a floor walled that high is blocked (shelter 0.56) and enclosed
    (enclosure 0.48) but NOT a windstorm alley (M = 4.4 puts the bump at 1.10, under amp_prone), so
    the carve leaves it alone and the only question is which wind the calm test is asked of.  On the
    sheltered wind alone an 18 m/s day wind reads 10.2 m/s and the floor is not calm; on the
    composed wind it reads 5.3 and it is -- which is Jake's "sheltered valleys are the habitable
    microgeography" call arriving in the mask that is supposed to express it."""
    nx, ne = 56, 6
    h = np.zeros((nx, ne)); h[20:25, :] = 4000.0; h[30:35, :] = 4000.0
    u = np.full((nx, ne), -10.0); v = np.zeros((nx, ne))
    x = np.arange(nx) * 10e3; e = np.arange(ne) * 10e3
    floor = slice(26, 30)

    def fresh():
        return dict(u=u, v=v, land=np.ones((nx, ne), bool),
                    T_map_C=np.full((nx, ne), 15.0), u10_day_ms=np.full((nx, ne), 18.0))
    topo = dict(x=x, e=e, height=h)

    base = fresh(); sw.attach_shelter(base, topo); sw.attach_amplification(base, topo)
    assert np.allclose(base["terrain_factor"],
                       base["shelter"] * base["amplification"] * base["enclosure"])
    assert (base["enclosure"][floor, :] < 1.0).all(), "fixture: the branch must bite on the floor"
    assert (base["amplification"][floor, :] < 1.3).all(), "fixture: the floor must not be carved"

    # the DEFAULT mask is the composed wind's, carved by the windstorm flag
    want = (sw._calm_habitable_mask(base["land"], base["T_map_C"], base["u10_day_terrain_ms"])
            & ~base["amplification_prone"])
    assert np.array_equal(base["calm_habitable"], want)
    assert base["calm_habitable"][floor, :].all(), "the enclosed floor must read calm on the real wind"

    # the pre-DOC-45 mask is still reachable: the sheltered wind's, carved -- and the floor is not
    # calm on it, which is the whole disagreement
    legacy = fresh(); sw.attach_shelter(legacy, topo)
    sw.attach_amplification(legacy, topo, calm_on_terrain=False)
    want_legacy = (sw._calm_habitable_mask(legacy["land"], legacy["T_map_C"],
                                           legacy["u10_day_sheltered_ms"])
                   & ~legacy["amplification_prone"])
    assert np.array_equal(legacy["calm_habitable"], want_legacy)
    assert not legacy["calm_habitable"][floor, :].any(), "fixture: the floor is not calm on shelter alone"
    assert int(base["calm_habitable"].sum()) > int(legacy["calm_habitable"].sum())

    # the re-ask inherits the threshold `attach_shelter` was given: at 20 m/s the open plain's
    # 18 m/s composed wind is calm too, which it is not at the 10 m/s default
    wide = fresh(); sw.attach_shelter(wide, topo, calm_ms=20.0); sw.attach_amplification(wide, topo)
    plain = slice(0, 10)
    assert wide["calm_habitable"][plain, :].all() and not base["calm_habitable"][plain, :].any()
    assert float(wide["calm_ms"]) == 20.0

    # ...and the enclosure branch can be switched off, which restores the two-branch factor
    # exactly; the calm question is then asked of THAT composed wind
    off = fresh(); sw.attach_shelter(off, topo, enclosure_radius_km=0.0)
    sw.attach_amplification(off, topo, enclosure_radius_km=0.0)
    assert np.array_equal(off["terrain_factor"], off["shelter"] * off["amplification"])
    want_off = (sw._calm_habitable_mask(off["land"], off["T_map_C"], off["u10_day_terrain_ms"])
                & ~off["amplification_prone"])
    assert np.array_equal(off["calm_habitable"], want_off)
    assert not off["calm_habitable"][floor, :].any(), "without the enclosure branch the floor is not calm"

    # THE DIRECTION THE FLIP RUNS ON THE MAP.  The basin floor above is the case that ADDS calm
    # ground, and it is rare; the common case is a lee slope below `amp_prone`, which goes the
    # other way -- calm on the sheltered wind, not calm on the composed one, and not carved.
    # Measured on the shipped map (2026-09-14, the map's own ladder): the composed wind is above the
    # sheltered one on 52.8 % of map land (strictly) and below it on 0.17 %, so a 10 m/s calm
    # proxy (windstorm carve applied, no temperature window) drops from 56.7 % to 48.6 % of map land -- 8.0 %
    # removed, none added.  Here: a 255 m ridge at 10 m/s puts M ~ 0.28 and the bump near 1.20
    # within its 50 km footprint, so a 9 m/s wind is calm on the shelter branch and 10.8 m/s on
    # the composed one.
    h2 = np.zeros((nx, ne)); h2[20:22, :] = 255.0
    topo2 = dict(x=x, e=e, height=h2)
    lee = slice(16, 20)                       # downwind of the ridge (upwind is +x), 10-40 km out

    def fresh2():
        return dict(u=u, v=v, land=np.ones((nx, ne), bool),
                    T_map_C=np.full((nx, ne), 15.0), u10_day_ms=np.full((nx, ne), 9.0))
    default2 = fresh2(); sw.attach_shelter(default2, topo2); sw.attach_amplification(default2, topo2)
    legacy2 = fresh2(); sw.attach_shelter(legacy2, topo2)
    sw.attach_amplification(legacy2, topo2, calm_on_terrain=False)
    a2 = default2["amplification"][lee, :]
    assert (a2 > 1.05).all() and (a2 < 1.3).all(), (
        f"fixture: the lee must be gained but not windstorm-prone, got {a2.min():.2f}-{a2.max():.2f}")
    assert not default2["amplification_prone"][lee, :].any()
    assert np.allclose(default2["enclosure"][lee, :], 1.0), "fixture: a lee slope is not enclosed"
    assert legacy2["calm_habitable"][lee, :].all(), "the lee is calm on the sheltered wind alone"
    assert not default2["calm_habitable"][lee, :].any(), (
        "a lee windier than the threshold on the composed wind must not read calm")
    assert int(default2["calm_habitable"].sum()) < int(legacy2["calm_habitable"].sum())


# ---- issue #114: GUST factors ----

def test_turbulence_intensity_and_gust_factor_increase_with_roughness():
    z0 = np.array([sw.Z0_SEA, 0.05, 1.5])
    I = sw.turbulence_intensity(z0)
    G = sw.gust_factor(z0)
    assert np.all(np.diff(I) > 0)                       # rougher -> more turbulent
    assert np.all(G > 1.0) and np.all(np.diff(G) > 0)
    # open-terrain gust factor in the standard ~1.4-1.8 band
    assert 1.4 < float(sw.gust_factor(np.array([0.05]))[0]) < 1.8


def test_convective_gust_speed_scales_with_cape():
    # cold-pool outflow ~ k*sqrt(2 CAPE); zero/negative CAPE -> no bump
    assert sw.convective_gust_speed(0.0) == 0.0
    assert sw.convective_gust_speed(-50.0) == 0.0
    assert sw.convective_gust_speed(2000.0) > sw.convective_gust_speed(500.0)
    assert sw.convective_gust_speed(2000.0) == pytest.approx(0.3 * np.sqrt(2 * 2000.0))


def test_attach_gust_exceeds_mean_and_adds_convective_outflow():
    nx, ne = 6, 5
    dC = dict(u=np.full((nx, ne), -20.0), v=np.zeros((nx, ne)),
              land=np.ones((nx, ne), bool), dT_air_C=np.full((nx, ne), 4.0),
              theta_deg=np.linspace(44, 58, nx))
    sw.attach(dC)
    sw.attach_gust(dC)
    land = dC["land"]
    assert dC["u10_day_gust_ms"].shape == (nx, ne)
    assert np.all(dC["u10_day_gust_ms"] >= dC["u10_day_ms"] - 1e-12)          # gust >= mean
    assert dC["u10_day_gust_ms"][land].mean() > dC["u10_day_ms"][land].mean()
    base = dC["u10_day_gust_ms"].copy()
    sw.attach_gust(dC, cape=np.full((nx, ne), 2000.0))                        # convective outflow lifts it
    assert np.all(dC["u10_day_gust_ms"] >= base - 1e-12)
    assert np.any(dC["u10_day_gust_ms"] > base + 1e-9)
    # #115 M3.4 eddy gust: default 0 is byte-identical; a nonzero term adds in quadrature
    sw.attach_gust(dC)                                                        # no eddy gust
    g0 = dC["u10_day_gust_ms"].copy()
    sw.attach_gust(dC, eddy_gust_ms=sw.EDDY_GUST_RMS_INNERSEA)
    assert np.allclose(dC["u10_day_gust_ms"], np.hypot(g0, sw.EDDY_GUST_RMS_INNERSEA))


def test_attach_shapes_ordering_and_summary():
    rng = np.random.default_rng(0)
    nx, ne = 12, 9
    land = rng.random((nx, ne)) > 0.5
    dC = dict(u=np.full((nx, ne), -30.0), v=np.zeros((nx, ne)),
              land=land, dT_air_C=np.where(land, 10.0, 0.5),
              theta_deg=np.linspace(44, 58, nx))
    s = sw.attach(dC, koppen=None)
    for k in ("z0", "u10_factor", "u10_day_ms", "u10_night_ms"):
        assert dC[k].shape == (nx, ne)
    spd = np.hypot(dC["u"], dC["v"])
    # the habitability ladder is ordered: lowest level >= 10-m day >= 10-m night
    assert np.all(dC["u10_day_ms"] <= spd + 1e-12)
    assert np.all(dC["u10_night_ms"] <= dC["u10_day_ms"] + 1e-12)
    # night reduction bites on land only
    sea = ~land
    assert np.allclose(dC["u10_night_ms"][sea], dC["u10_day_ms"][sea])
    assert np.all(dC["u10_night_ms"][land] < dC["u10_day_ms"][land])
    assert s["land_u10_day_mean_ms"] < s["land_lev_mean_ms"]


def test_the_near_field_is_a_subsample_of_the_far_field():
    """PR #731 review, finding 5: the footprint weight must come from the SAME quadrature as the
    relief it divides.  Two independent log-spaced scans are not nested -- a 300 km ray puts 4 of its
    8 samples inside 50 km, a 50 km ray puts all 8 -- so the short scan could EXCEED the long one,
    which on the shipped map happened on 16 % of land: `w` clipped to 1 (so those cells took the full
    effect while the "fraction of the obstacle within reach" reading was false of them) and the
    regime's own H was biased LOW by up to 391 m, toward the windstorm branch and away from blocking.
    Nested maxima make it impossible rather than rare."""
    rng = np.random.default_rng(17)
    nx, ne = 40, 40
    h = rng.uniform(0.0, 3000.0, (nx, ne))                 # rough enough that a short ray can win
    u = -rng.uniform(3.0, 30.0, (nx, ne)); v = rng.uniform(-10.0, 10.0, (nx, ne))
    x = np.arange(nx) * 12e3; e = np.arange(ne) * 12e3
    relief, (near_lee, near_wake) = sw._upwind_relief_nested(
        h, u, v, x, e, 300.0, (sw.LEE_FOOTPRINT_KM, sw.WAKE_FOOTPRINT_KM))
    assert np.all(near_lee <= relief + 1e-9) and np.all(near_wake <= relief + 1e-9)
    assert np.all(near_lee <= near_wake + 1e-9)            # nested cutoffs, nested maxima
    assert (near_lee < relief).any(), "fixture: the footprint must bite somewhere"


def test_a_fixed_km_footprint_is_sampled_the_same_way_at_every_resolution():
    """PR #731 RE-review, finding 1: the first single-quadrature fix cured the nesting violation and
    introduced a resolution one.  One ``geomspace(d0, L, n)`` has sample positions that scale with
    the CELL, so a footprint fixed in km is resolved differently at every grid -- on a 300 km ray of
    8 samples the outermost sample inside 50 km was 48.8 / 50.0 / 48.1 / **32.4** km at 48.8 / 24.4 /
    12.2 / 6.1 km cells, so the finest grid missed the outer 35 % of the windstorm footprint
    entirely.  It showed up as the served lee-gain fraction FALLING at the finest rung (8.11 % ->
    6.14 %) where the form it replaced rose monotonically: the finer the grid, the worse the
    footprint was resolved, which is the wrong sign for a convergence claim.

    ``ray_distances`` windows the ray on the cutoffs instead, so each footprint keeps a FIXED sample
    count at every resolution and each cutoff is sampled exactly."""
    cuts = (sw.LEE_FOOTPRINT_KM * 1e3, sw.WAKE_FOOTPRINT_KM * 1e3)
    counts = []
    for d0_km in (48.8, 24.4, 12.2, 6.1, 3.0):
        d = sw.ray_distances(d0_km * 1e3, 300e3, cuts)
        assert np.all(np.diff(d) > 0), "distances must be sorted and unique"
        for c in cuts:                                     # every cutoff is itself a sample point
            assert np.isclose(d, c).any(), f"{c/1e3:.0f} km is not sampled at {d0_km} km cells"
        counts.append((int((d <= cuts[0] + 1).sum()), int((d <= cuts[1] + 1).sum())))
    assert len(set(counts)) == 1, f"the footprints are resolved differently by grid: {counts}"
    # the outermost sample inside each footprint IS the cutoff, at every resolution -- the property
    # whose absence produced the 6.14 % rung
    for d0_km in (48.8, 12.2, 6.1):
        d = sw.ray_distances(d0_km * 1e3, 300e3, cuts)
        assert np.isclose(d[d <= cuts[0] + 1].max(), cuts[0])
    # a grid coarser than both footprints degrades to the adjacent cell rather than failing
    coarse = sw.ray_distances(200e3, 300e3, cuts)
    assert coarse[0] == 200e3 and coarse[-1] == 300e3


def test_the_shared_scan_never_overrides_attach_amplifications_own_parameters():
    """PR #731 RE-review, finding 2.  ``attach_shelter`` stashes its ``terrain_factors`` result so
    ``attach_amplification`` need not re-scan -- but the first version read that stash
    UNCONDITIONALLY, so every one of amplification's own N / amp_max / M_crit / ln_width /
    footprint_km / U_floor was silently discarded whenever shelter had run first.  Production was
    unaffected (the cascade calls both with defaults) and it would have bitten exactly the workflow
    ROADMAP DOC-46 queues: an N sweep through the attach path, returning the shipped N with no error.
    The stash now carries the parameter key it was computed under, and is POPPED rather than left in
    ``dC`` -- callers ``np.savez(dC)``, and it is scratch."""
    nx, ne = 24, 8
    h = np.zeros((nx, ne)); h[12, :] = 3000.0
    topo = dict(height=h, x=np.arange(nx) * 50e3, e=np.arange(ne) * 50e3)

    def fresh():
        dC = dict(u=np.full((nx, ne), -30.0), v=np.zeros((nx, ne)), land=np.ones((nx, ne), bool),
                  dT_air_C=np.full((nx, ne), 4.0), theta_deg=np.linspace(44, 58, nx),
                  T_map_C=np.full((nx, ne), 15.0))
        sw.attach(dC)
        return dC

    for kw in ({"N": 0.022}, {"amp_max": 5.0}, {"footprint_km": 300.0}):
        alone = fresh()
        sw.attach_amplification(alone, topo, **kw)
        after = fresh()
        sw.attach_shelter(after, topo, calm_ms=50.0)
        sw.attach_amplification(after, topo, **kw)
        assert np.allclose(alone["amplification"], after["amplification"]), (
            f"attach_shelter's stash overrode attach_amplification's own {list(kw)[0]}")
        assert "_terrain_factors" not in after, "the scratch stash was left in dC for np.savez"

    # ...and on the DEFAULT path the scan really is shared, which is why the stash exists
    calls = []
    orig = sw.terrain_factors
    try:
        sw.terrain_factors = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
        dC = fresh()
        sw.attach_shelter(dC, topo, calm_ms=50.0)
        sw.attach_amplification(dC, topo)
        assert len(calls) == 1, f"the default path re-scanned ({len(calls)} calls)"
    finally:
        sw.terrain_factors = orig


def test_the_rays_resolution_knob_reaches_the_quadrature_on_every_entry_point():
    """PR #731 re-review, item 4.  ``n_per`` is the one ray-resolution knob the four terrain entry
    points expose, and it has to REACH ``ray_distances`` rather than sit in the signature.  The
    argument it replaced (``n_steps``, a total sample count inherited from the pre-windowing ray) did
    exactly the latter for the whole first draft of this law: it was accepted, defaulted, threaded
    from one function to the next -- and dropped on the floor at ``_upwind_relief_nested``, which
    takes ``n_per``.  A caller who thought they were refining the ray was refining nothing, and the
    sweep ROADMAP DOC-46 queues would have read as an insensitivity.  So: refining the quadrature
    must MOVE the answer, on each of the four."""
    nx, ne = 40, 12
    x = np.arange(nx) * 12e3
    e = np.arange(ne) * 12e3
    X, E = np.meshgrid(x, e, indexing="ij")
    # two ridges at different upwind distances, so the sample positions decide what the ray finds:
    # a tall one at 3 km relief (M ~ 3.3 at 10 m/s -> the blocked branch) and a lower one nearer.
    h = (3000.0 * np.exp(-((X - 120e3) ** 2) / (2 * (18e3) ** 2))
         + 900.0 * np.exp(-((X - 300e3) ** 2) / (2 * (10e3) ** 2)) + 0.0 * E)
    u = np.full_like(h, -10.0)          # wind blowing toward -x, so the ray looks toward +x
    v = np.zeros_like(h)

    coarse = sw.terrain_factors(h, u, v, x, e, n_per=2)
    fine = sw.terrain_factors(h, u, v, x, e, n_per=12)
    for k in ("relief", "M", "w_lee", "w_wake", "amplification"):
        assert not np.allclose(coarse[k], fine[k]), f"terrain_factors: n_per never reached {k}"
    assert not np.allclose(sw.lee_regime(h, u, v, x, e, n_per=2)[0],
                           sw.lee_regime(h, u, v, x, e, n_per=12)[0]), "lee_regime: n_per is dead"
    assert not np.allclose(sw.shelter_factor(h, u, v, x, e, n_per=2),
                           sw.shelter_factor(h, u, v, x, e, n_per=12)), "shelter_factor: n_per is dead"
    assert not np.allclose(sw.terrain_amplification(h, u, v, x, e, n_per=2)[0],
                           sw.terrain_amplification(h, u, v, x, e, n_per=12)[0]), \
        "terrain_amplification: n_per is dead"
    # and it is a PER-WINDOW count, not a total: every footprint edge stays a sample at every n_per,
    # which is the property that makes a fixed-km footprint resolution-stable in the first place.
    for n in (1, 2, 4, 12):
        d = sw.ray_distances(6.1e3, 300e3, (50e3, 150e3), n_per=n)
        assert 50e3 in d and 150e3 in d and d[-1] == 300e3
        assert np.sum(d <= 50e3) == max(n, 2)      # the window's own samples (d0 and the edge at n=1)
