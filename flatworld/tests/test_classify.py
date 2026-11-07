"""Tests for the seasonless Koppen / habitability classification (downstream post-processor)."""
import numpy as np
import pytest
from flatworld import classify


def test_seasonless_koppen_basic_types():
    land = np.array([True, True, True, True, True, True])
    T = np.array([35.0, 14.0, 25.0, 25.0, 5.0, -20.0])      # hot-dry, cool, warm-wet, warm-dry, tundra, ice
    P = np.array([50.0, 1500.0, 3000.0, 100.0, 600.0, 100.0])
    code = classify.koppen_class(T, P, land)
    lab = [classify.LABELS[c - 1] for c in code]
    assert lab[0] == "BWh"        # hot desert
    assert lab[1] == "Cfb"        # temperate oceanic (10-18 C, wet)
    assert lab[2] == "Af"         # tropical rainforest (hot, very wet)
    assert lab[3] == "BWh"        # hot but dry -> desert
    assert lab[4] == "ET"         # tundra (0-10 C)
    assert lab[5] == "EF"         # ice cap (<0)


def test_forbidden_families_never_emitted():
    # whatever the (T,P), only the seasonless-allowed types appear (no D, no Cs/Cw, no Cfa)
    rng = np.random.default_rng(0)
    T = rng.uniform(-40, 50, 5000); P = rng.uniform(0, 6000, 5000)
    code = classify.koppen_class(T, P, np.ones(5000, bool))
    present = set(classify.LABELS[c - 1] for c in np.unique(code) if c > 0)
    allowed = set(classify.LABELS)
    assert present <= allowed
    for forbidden in ("Dfb", "Cfa", "Csa", "Cwa", "Dwc"):
        assert forbidden not in present


def test_sea_is_zero():
    land = np.array([True, False])
    code = classify.koppen_class(np.array([15.0, 15.0]), np.array([1000.0, 1000.0]), land)
    assert code[1] == 0 and code[0] != 0


def test_habitability_classes():
    land = np.array([True, True, True, True, False])
    T = np.array([20.0, 20.0, 20.0, -10.0, 20.0])
    P = np.array([100.0, 350.0, 1000.0, 100.0, 1000.0])
    code = classify.habitability_class(T, P, land)
    assert code[0] == classify.HAB_IDX["desert"]
    assert code[1] == classify.HAB_IDX["steppe"]
    assert code[2] == classify.HAB_IDX["humid"]
    assert code[3] == classify.HAB_IDX["frozen"]
    assert code[4] == classify.HAB_IDX["ocean"]


def test_class_fractions_sum():
    land = np.ones((10, 10), bool)
    T = np.full((10, 10), 14.0); P = np.full((10, 10), 1000.0)   # all Cfb
    fr = classify.class_fractions(classify.koppen_class(T, P, land), classify.LABELS)
    assert fr.get("Cfb", 0) == pytest.approx(1.0)


# ---- frost-aware classification (Issue #21 follow-up): E family keyed on the AIR extremes ----
def test_koppen_fallback_unchanged():
    # existing-caller regression: with the extremes omitted, the output is BYTE-identical to the mean scheme.
    rng = np.random.default_rng(1)
    T = rng.uniform(-40, 50, 4000); P = rng.uniform(0, 6000, 4000); land = np.ones(4000, bool)
    base = classify.koppen_class(T, P, land)
    assert isinstance(base, np.ndarray)                          # fallback returns ONLY the code array
    # passing only one of the two extremes is still the fallback (both required)
    assert np.array_equal(classify.koppen_class(T, P, land, T_min_C=T), base)


def test_koppen_EF_never_thaws():
    # the substantive physical change: EF = the WARMEST instant is still sub-freezing (never thaws). A cell
    # whose air max reaches >=0 (but mean<0) thaws by day -> ET, not EF (the old mean-keyed scheme called both EF).
    land = np.array([True, True]); P = np.array([100.0, 100.0])
    T = np.array([-5.0, -5.0])                                   # both have a sub-zero day-mean (old scheme: EF)
    Tmin = np.array([-12.0, -12.0]); Tmax = np.array([-2.0, 3.0])  # first never thaws; second thaws by day
    code, _ = classify.koppen_class(T, P, land, Tmin, Tmax)
    lab = [classify.LABELS[c - 1] for c in code]
    assert lab[0] == "EF" and lab[1] == "ET"


def test_koppen_ET_freezes_nightly_thaws_by_day():
    land = np.array([True]); P = np.array([800.0])
    T = np.array([4.0])                                          # cool mean, below the 10 C cap
    code, frosts = classify.koppen_class(T, P, land, np.array([-3.0]), np.array([12.0]))
    assert classify.LABELS[code[0] - 1] == "ET" and bool(frosts[0]) is False  # ET is not a "warm class" overlay


def test_frost_does_not_demote_temperate_mean():
    # the mean-cap guard: a warm-MEAN cell (12 C) that frosts at night stays its day-mean class (Cfb) with the
    # frosts overlay set -- it is NOT demoted to tundra.
    land = np.array([True]); P = np.array([900.0]); T = np.array([12.0])
    code, frosts = classify.koppen_class(T, P, land, np.array([-2.0]), np.array([22.0]))
    assert classify.LABELS[code[0] - 1] == "Cfb" and bool(frosts[0]) is True


def test_residual_mild_folds_into_cfb():
    # subtlety (b): a cool-mean (0-10 C) but FROST-FREE cell is no longer ET -> folds into Cfb (oceanic).
    land = np.array([True]); P = np.array([900.0]); T = np.array([5.0])
    code, frosts = classify.koppen_class(T, P, land, np.array([1.0]), np.array([9.0]))
    assert classify.LABELS[code[0] - 1] == "Cfb" and bool(frosts[0]) is False


def test_habitability_frost_prone():
    # livable day-mean but a freezing air night -> frost-prone; a frost-free livable cell stays humid; a
    # sub-zero-mean cell is frozen.
    land = np.array([True, True, True, False])
    T = np.array([8.0, 8.0, -5.0, 8.0]); P = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    Tmin = np.array([-3.0, 2.0, -12.0, -3.0])
    code = classify.habitability_class(T, P, land, T_min_C=Tmin)
    assert code[0] == classify.HAB_IDX["frost-prone"]
    assert code[1] == classify.HAB_IDX["humid"]
    assert code[2] == classify.HAB_IDX["frozen"]
    assert code[3] == classify.HAB_IDX["ocean"]
    # fallback (no air min) emits no frost-prone -> byte-identical to the mean-only scheme
    base = classify.habitability_class(T, P, land)
    assert classify.HAB_IDX["frost-prone"] not in set(np.unique(base))


def test_extended_class_each_type_and_priority():
    """Extended ('Köppen-plus') classifier: one engineered cell per class hits its code, and the priority
    order resolves overlaps (lethal extremes win; the frost/shaded types are arid-gated so a dry cold/dim
    cell stays a desert)."""
    c = classify
    # (label, T, P_yr, Tmin, Tmax, wet_bulb, surf_sw)
    cells = [
        ("SF",  120.0,   10.0,  80.0, 160.0, 20.0, 300.0),   # hot & rainless -> salt flat (beats BWh)
        ("XC",  -90.0,    0.0, -95.0, -85.0, -90.0, 50.0),   # < -78 -> lethal cold
        ("WW",   35.0, 2000.0,  30.0,  40.0, 36.0, 200.0),   # wet-bulb > 35
        ("Wd",   33.0, 2000.0,  28.0,  38.0, 32.0, 200.0),   # wet-bulb 31-35
        ("EF",  -10.0,  300.0, -15.0,  -2.0, -10.0, 100.0),  # never thaws (non-arid)
        ("ET",    3.0,  400.0,  -3.0,   8.0,  2.0, 100.0),   # frost night, thaws by day
        ("Sh-t", 14.0, 1500.0,   2.0,  22.0, 15.0,  10.0),   # warm, dim, frost-free
        ("Sh-c",  5.0, 1000.0,   1.0,  12.0,  6.0,  10.0),   # cool, dim, frost-free
        ("BWh",  30.0,   50.0,  20.0,  45.0, 18.0, 300.0),   # hot arid (dry -> low wet-bulb)
        ("BSh",  25.0,  500.0,  18.0,  35.0, 20.0, 300.0),   # hot semi-arid
        ("Af",   25.0, 2500.0,  22.0,  30.0, 28.0, 300.0),   # hot wet
        ("Am",   25.0,  700.0,  22.0,  30.0, 28.0, 300.0),   # hot, monsoon band
        ("Cfb",  14.0, 1500.0,   5.0,  22.0, 18.0, 300.0),   # temperate, bright, frost-free
        # The cool arid pairs split on the NIGHTLY LOW, not on a season (2026-07-27). Same T and P,
        # opposite T_min, opposite class -- these four rows ARE the split's pin.
        ("BWk",   5.0,   20.0,  -4.0,  14.0,  4.0, 200.0),   # cold desert  — freezes at night
        ("BSk",   5.0,  180.0,  -4.0,  14.0,  4.0, 200.0),   # cold steppe  — freezes at night
        ("BWt",   5.0,   20.0,   1.0,  12.0,  4.0, 200.0),   # temperate desert — frost-free
        ("BSt",   5.0,  180.0,   1.0,  12.0,  4.0, 200.0),   # temperate steppe — frost-free
    ]
    T = np.array([r[1] for r in cells]); P = np.array([r[2] for r in cells])
    Tmin = np.array([r[3] for r in cells]); Tmax = np.array([r[4] for r in cells])
    wb = np.array([r[5] for r in cells]); sw = np.array([r[6] for r in cells])
    land = np.ones(len(cells), bool)
    code = c.extended_class(T, P, land, Tmin, Tmax, wb, sw)
    got = [c.EXT_LABELS[k - 1] if k > 0 else "sea" for k in code]
    assert got == [r[0] for r in cells], dict(zip([r[0] for r in cells], got))
    # priority: a DRY, cold, DIM cell is a desert (the shaded/frost types are arid-gated), not Sh-c/ET
    # -- and which desert is decided by the night: freezing -> BWk, frost-free -> BWt.
    for tmin, want in ((-4.0, "BWk"), (0.0, "BWt")):
        dcd = c.extended_class(np.array([5.0]), np.array([20.0]), np.array([True]),
                               np.array([tmin]), np.array([12.0]), np.array([4.0]), np.array([5.0]))
        assert c.EXT_LABELS[dcd[0] - 1] == want
    # sea stays 0
    assert c.extended_class(T, P, np.zeros(len(cells), bool), Tmin, Tmax, wb, sw).tolist() == [0] * len(cells)
    # CSF is the eighteenth class and a MASK class: no predicate can reach it, so the battery above
    # cannot produce it and it is exercised here, under law 3 with the basin mask. THREE cells carry
    # the whole rule -- the hot rainless pan (row 0) stays SF inside the mask, the cool DRY floor (the
    # BWt row, 20 mm/yr against an R_th of 240) becomes CSF, and the mild WET floor (Cfb, 1500 mm/yr)
    # is not an evaporite pan at all and falls through unchanged (PR #454) -- and nothing outside the
    # mask moves.
    # CSF is the LAST class `extended_class` itself can paint; A5's three sit after it in the table
    # but are assigned by the serving classifier, never here (see `_MASK_CLASSES`).
    assert c.EXT_IDX["CSF"] == 18 and c.EXT_LABELS[17] == "CSF"
    assert set(c.EXT_LABELS[18:]) == {"FZF", "BPN", "WSP"}
    labels = [r[0] for r in cells]
    i_sf, i_dry, i_wet = 0, labels.index("BWt"), labels.index("Cfb")
    bm = np.zeros(len(cells), bool); bm[[i_sf, i_dry, i_wet]] = True
    c3 = c.extended_class(T, P, land, Tmin, Tmax, wb, sw, wet_bulb_high_C=wb + 1.0, heat_law=3,
                          basin_mask=bm)
    assert c.EXT_LABELS[c3[i_sf] - 1] == "SF", "a genuinely hot pan inside the mask must stay SF"
    assert c.EXT_LABELS[c3[i_dry] - 1] == "CSF"
    assert c.EXT_LABELS[c3[i_wet] - 1] == "Cfb", "a WET closed floor is a lake basin, not a salt pan"
    c2 = c.extended_class(T, P, land, Tmin, Tmax, wb, sw, wet_bulb_high_C=wb + 1.0, heat_law=2)
    assert np.array_equal(c3[~bm], c2[~bm]), "law 3 moved a cell the basin mask never touched"
    assert c3[i_wet] == c2[i_wet], "a fall-through cell must be exactly what law 2 said"


def test_heat_law2_dryness_adjusted():
    """GEO-20: law 2 splits the tiers by EXPOSURE STATISTIC so the labels keep their meanings
    (Jake, 2026-08-02: Kuwait must read difficult, not impossible — move thresholds, not labels).
    Wd = the daily-high wet-bulb within 4 K of its dryness-adjusted critical (dangerous hours,
    shelterable); WW = the day-MEAN exceeds the critical at its own depression (no respite), or the
    high clears the 35 °C theoretical limit outright. Pinned cells:
      * a saturated ribbon-edge cell (day-mean wb 31.4 > its critical 30.8) is merely Wd under
        law 1 but WW under law 2 — around-the-clock heat above the measured critical wet-bulb;
      * a 54 °C hot-DRY cell (wb 18 → law 1 says ordinary desert) is **Wd** under law 2: its high
        wet-bulb 22.9 clears the dryness-lowered danger line 17.2, but the mean sits far below its
        critical — Kuwait/Death-Valley-class heat: brutally dangerous hours, livable by sheltering.
        (The 22.9 is clipped-Stull scale — rh_high ~ 3.8% is under the fit's 5% floor; psychrometric
        ~22.0; the Wd verdict survives on either scale. See wet_bulb_high's validity paragraph.)
      * the absolute-peak backstop: a cell whose mean is safe but whose high wet-bulb passes 35 °C
        is WW regardless — brief exposure past the theoretical limit kills without needing a
        sustained mean (the original 'a daily high of 40 kills you even if the mean is 34' case;
        wbh fed directly, since no conserved-vapor day reaches it from a benign mean)."""
    from flatworld import classify as c, thermo
    land = np.array([True, True, True])
    T = np.array([32.0, 42.0, 30.0]); Th = np.array([36.0, 54.0, 40.0]); RH = np.array([0.95, 0.07, 0.60])
    Tmin = np.array([27.0, 28.0, 24.0]); P = np.array([1500.0, 30.0, 900.0]); sw = np.array([200.0, 300.0, 200.0])
    wb = thermo.wet_bulb(T, RH)
    wbh = np.asarray(thermo.wet_bulb_high(T, Th, RH), float).copy()
    wbh[2] = 36.0                                   # the backstop cell's spike, fed directly
    law1 = c.extended_class(T, P, land, Tmin, Th, wb, sw)
    law2 = c.extended_class(T, P, land, Tmin, Th, wb, sw, wet_bulb_high_C=wbh, heat_law=2)
    assert [c.EXT_LABELS[k - 1] for k in law1] == ["Wd", "BWh", "Af"]
    assert [c.EXT_LABELS[k - 1] for k in law2] == ["WW", "Wd", "WW"]
    # law 1 is the default and ignores the high-wb argument entirely (byte-identical guard)
    law1_with_arg = c.extended_class(T, P, land, Tmin, Th, wb, sw, wet_bulb_high_C=wbh)
    assert np.array_equal(law1, law1_with_arg)
    # law 2 without its required input is an error, not a silent fallback
    with pytest.raises(ValueError, match="wet_bulb_high_C"):
        c.extended_class(T, P, land, Tmin, Th, wb, sw, heat_law=2)
    # threshold anchors: 31 at saturation, 20 at the D=35 dry limit, floored beyond
    ww, wd = c.heat_thresholds_C(np.array([40.0, 55.0, 90.0]), np.array([40.0, 20.0, 10.0]))
    assert np.allclose(ww, [31.0, 20.0, 20.0]) and np.allclose(wd, ww - 4.0)


def test_heat_law3_adds_the_basin_class_and_nothing_else():
    """GEO-21: law 3 IS law 2 plus the CSF mask class, and that 'plus' is the whole difference.

    Pinned, on a randomised battery so it is not a hand-picked-cell argument:
      * off the mask, law 3 and law 2 agree cell for cell -- no threshold moved;
      * on the mask, a LAND cell is CSF when it fails SF's own gate AND is losing water: a hot pan
        stays SF (the point of the class: 'salt flat' goes back to meaning lethally hot), and a floor
        under a wet climate is a lake basin, so it keeps exactly the class law 2 gave it;
      * a masked SEA cell is still sea -- the override cannot outrank the land mask, which is what
        forces the fine-grid display paths to apply the split themselves (classify's docstring);
      * law 3 without the mask raises rather than silently degrading to law 2."""
    from flatworld import classify as c
    rng = np.random.default_rng(11)
    n = 8000                      # the mask x SF-gate x water-gate split is a ~2% corner; n must feed it
    T = rng.uniform(-100.0, 85.0, n)
    P = rng.uniform(0.0, 3000.0, n)
    land = rng.random(n) < 0.75
    Tmin = T - rng.uniform(0.0, 30.0, n); Tmax = T + rng.uniform(0.0, 30.0, n)
    wb = T - rng.uniform(0.0, 12.0, n); wbh = Tmax - rng.uniform(0.0, 30.0, n)
    sw = rng.uniform(0.0, 400.0, n); Tsm = Tmax + rng.uniform(0.0, 220.0, n)
    bm = rng.random(n) < 0.2
    kw = dict(T_surf_max_C=Tsm, wet_bulb_high_C=wbh)
    law2 = c.extended_class(T, P, land, Tmin, Tmax, wb, sw, heat_law=2, **kw)
    law3 = c.extended_class(T, P, land, Tmin, Tmax, wb, sw, heat_law=3, basin_mask=bm, **kw)
    assert np.array_equal(law2[~bm], law3[~bm]), "law 3 is not law 2 off the basin mask"
    sf_gate = c.salt_flat_gate(Tsm, P)
    water = c.csf_water_gate(T, P)
    moved = bm & land & ~sf_gate & water
    stayed = bm & land & ~sf_gate & ~water
    assert moved.sum() > 100, "the fixture must actually exercise the override"
    assert stayed.sum() > 100, "the fixture must also exercise the WET fall-through"
    assert (law3[moved] == c.EXT_IDX["CSF"]).all()
    assert np.array_equal(law3[stayed], law2[stayed]), \
        "a masked floor that is not losing water must keep its ordinary law-2 class"
    assert (law3[bm & land & sf_gate] == c.EXT_IDX["SF"]).all(), "a hot pan must not become CSF"
    assert (law3[bm & ~land] == 0).all(), "the override must not paint sea"
    assert not (law2 == c.EXT_IDX["CSF"]).any(), "CSF is unreachable below law 3"
    with pytest.raises(ValueError, match="basin_mask"):
        c.extended_class(T, P, land, Tmin, Tmax, wb, sw, heat_law=3, **kw)
    with pytest.raises(ValueError, match="wet_bulb_high_C"):
        c.extended_class(T, P, land, Tmin, Tmax, wb, sw, heat_law=3, basin_mask=bm, T_surf_max_C=Tsm)
    with pytest.raises(ValueError, match="1, 2 or 3"):
        c.extended_class(T, P, land, Tmin, Tmax, wb, sw, heat_law=4, **kw)
    # the default law still ignores the mask entirely -- the byte-identity guard the flip rests on
    assert np.array_equal(c.extended_class(T, P, land, Tmin, Tmax, wb, sw, T_surf_max_C=Tsm),
                          c.extended_class(T, P, land, Tmin, Tmax, wb, sw, basin_mask=bm, **kw))


def test_csf_water_gate_four_quadrants():
    """PR #454: a masked floor is an evaporite pan only where it is LOSING water, and the gate that
    decides that has to be right in all four corners of the (warm/cold) x (wet/dry) plane.

    The gate is `P < max(R_th(T), 50)`, and the MAX is the whole design:
      * warm-WET  -> falls through to the ordinary tree; a closed floor under rain is a lake basin,
        and painting it evaporite is the failure the gate exists to prevent;
      * warm-DRY  -> CSF on Koppen aridity, the standard "evaporation beats rainfall" line;
      * cold-SNOWY (P >= 50) -> falls through, and lands on EF -- perennial ice, correctly, because
        there IS a supply to build it from;
      * cold-BONE-DRY -> CSF, NOT EF. R_th has clipped to zero below -7 C, so a bare `P < R_th`
        would call nothing arid there and destroy the cold half of the class; sublimation empties a
        frozen pan whether or not it ever thaws (Don Juan Pond), so the absolute 50 mm/yr floor is
        what decides it. This corner is why there is deliberately no thaw gate."""
    from flatworld import classify as c
    # R_th: warm 30 C -> 740 mm/yr; cold -30 C -> clipped to 0, so the 50 mm/yr floor decides
    T = np.array([30.0, 30.0, -30.0, -30.0])
    P = np.array([2000.0, 100.0, 400.0, 5.0])
    assert c.aridity_threshold_mm_yr(T).tolist() == [740.0, 740.0, 0.0, 0.0]
    assert c.csf_water_gate(T, P).tolist() == [False, True, False, True]
    # ...and the same four cells through the classifier, on the mask, at law 3.  The cold pair never
    # thaws (T_max = -22), so a fall-through there lands on EF -- which is the point of the fourth
    # corner: only the WATER gate keeps the bone-dry one out of the ice cap.
    land = np.ones(4, bool); bm = np.ones(4, bool)
    Tmin = np.array([18.0, 18.0, -38.0, -38.0]); Tmax = np.array([38.0, 38.0, -22.0, -22.0])
    wb = np.array([12.0, 12.0, -35.0, -35.0]); wbh = np.array([14.0, 14.0, -33.0, -33.0])
    sw = np.full(4, 200.0); Tsm = np.array([45.0, 45.0, -20.0, -20.0])   # well under SF's 70 C gate
    kw = dict(T_surf_max_C=Tsm, wet_bulb_high_C=wbh)
    code = c.extended_class(T, P, land, Tmin, Tmax, wb, sw, heat_law=3, basin_mask=bm, **kw)
    got = [c.EXT_LABELS[k - 1] for k in code]
    assert got == ["Af", "CSF", "EF", "CSF"], got
    # the two fall-through cells are exactly what law 2 would have said off the mask -- the gate
    # returns them to the ordinary tree rather than inventing a class for them
    law2 = c.extended_class(T, P, land, Tmin, Tmax, wb, sw, heat_law=2, **kw)
    fell = np.array([True, False, True, False])
    assert np.array_equal(code[fell], law2[fell])
    # the rainless cut is SHARED with salt_flat_gate, not a second copy of 50.0
    assert c.SF_P_MAX_MM_YR == 50.0
    assert c.csf_water_gate(np.array([-30.0, -30.0]),
                            np.array([49.9, 50.0])).tolist() == [True, False]


def test_csf_falls_through_to_the_tree_not_to_sea():
    """The fall-through only produces a real class if the caller hands in `land | basin`; with the
    bare land mask a below-datum floor is zeroed by `code[~land] = 0` and reads as OCEAN -- a
    transparent hole on the display paths. Both halves pinned, because the second is what every
    fine-grid caller now does."""
    from flatworld import classify as c
    T = np.array([25.0]); P = np.array([2500.0])                # warm and wet -> gate says NOT a pan
    Tmin = np.array([15.0]); Tmax = np.array([33.0]); wb = np.array([18.0]); sw = np.array([250.0])
    kw = dict(T_surf_max_C=np.array([40.0]), wet_bulb_high_C=np.array([20.0]), heat_law=3,
              basin_mask=np.array([True]))
    below_datum = np.array([False])                             # the floor is not `elev > 0`
    assert c.extended_class(T, P, below_datum, Tmin, Tmax, wb, sw, **kw)[0] == 0
    widened = below_datum | kw["basin_mask"]
    assert c.EXT_LABELS[c.extended_class(T, P, widened, Tmin, Tmax, wb, sw, **kw)[0] - 1] == "Af"


def test_mask_class_table_and_conds_stay_in_step():
    """`np.select` DROPS trailing choices when the two lists disagree, so a class appended to `_E`
    without a matching cond would silently mis-code the tail of the table instead of failing.  The
    guard is the length identity `len(conds) == len(_E) - len(_MASK_CLASSES)`; this pins both halves
    of it, and that the mask classes are LAST (which is what keeps stored codes 1-17 stable)."""
    from flatworld import classify as c
    # `CSF` first, then A5's three surface classes -- which this module DEFINES but never assigns:
    # the split turns on a brine level and a deflation capacity that live on the serving side, so
    # `extended_class` still paints `CSF` alone and `map_webapp.mapdata._koppen_classify` partitions
    # it (EXTENDED_CLIMATE_CLASSES.md, the FZF/BPN/WSP section).
    assert c._MASK_CLASSES == ("CSF", "FZF", "BPN", "WSP")
    assert c.EXT_LABELS[-len(c._MASK_CLASSES):] == list(c._MASK_CLASSES)
    assert len(c._E) - len(c._MASK_CLASSES) == c._N_PREDICATE_CLASSES
    assert [c.EXT_IDX[lab] for lab in ("SF", "XC", "BSt", "CSF")] == [1, 2, 17, 18], \
        "appending a mask class renumbered an existing one -- every stored koppen map is now wrong"
    # the gate SF and the CSF override share (a hand-written second copy is what would drift)
    assert c.salt_flat_gate(np.array([71.0, 69.0, 71.0]), np.array([49.0, 49.0, 51.0])).tolist() \
        == [True, False, False]


def test_wet_bulb_high_conserves_vapor():
    """thermo.wet_bulb_high holds the day-mean VAPOR fixed while T rises to the daily high: at
    saturation midday RH falls below 1, so the high wet-bulb sits well under the fixed-RH read; and
    it can never exceed the fixed-RH read in the warm regime (that read silently adds vapor)."""
    from flatworld import thermo
    wbh = float(thermo.wet_bulb_high(30.0, 40.0, 1.0))
    fixed = float(thermo.wet_bulb(40.0, 1.0))
    assert wbh < fixed - 2.0                      # ~conserved e: RH(40°C) ≈ esat(30)/esat(40) ≈ 0.58
    assert wbh > float(thermo.wet_bulb(30.0, 1.0)) - 1.0    # but the warm hour is no cooler than the mean


def test_wet_bulb_basic():
    """Stull wet-bulb: ~dry-bulb at saturation, lower when drier, higher at warmer T for the same RH."""
    from flatworld import thermo
    assert abs(float(thermo.wet_bulb(30.0, 1.0)) - 30.0) < 1.0
    assert float(thermo.wet_bulb(30.0, 0.3)) < float(thermo.wet_bulb(30.0, 0.8)) < 30.0
    assert float(thermo.wet_bulb(40.0, 0.6)) > float(thermo.wet_bulb(30.0, 0.6))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
