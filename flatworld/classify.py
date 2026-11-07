"""
flatworld.classify -- seasonless Koppen-Geiger classification of the model's (T, P) maps.

A downstream post-processor (not a climate method): it turns the model's predicted time-mean
temperature and precipitation into climate classes. Koppen keys on coldest/warmest-MONTH means and
precipitation SEASONALITY; this world has NO seasons (fixed sun-track), so at any place the coldest
month = warmest month = annual mean and precipitation is aseasonal. Forcing Koppen's thresholds
through that degeneracy collapses ~30 types to a handful and FORBIDS whole families: no continental
(D) types (need a cold winter AND warm summer at one place), no dry-season subtypes (Cs/Cw/Aw...), no
hot-summer Cfa. The validated seasonless logic is preserved here (see the retired koppen.py); only the
INPUT changes -- it now consumes this model's predicted (T, P), not the prior anchored profile.
"""
from __future__ import annotations
import numpy as np

# Koppen types that CAN occur on the seasonless world, with standard-ish colours.
_K = [("Af", "#0000FE"), ("Am", "#0077FF"), ("BWh", "#FF0000"), ("BSh", "#F5A300"),
      ("BWk", "#FF9695"), ("BSk", "#FFDB63"), ("Cfb", "#32CD32"), ("ET", "#B2B2B2"), ("EF", "#5A5A5A")]
LABELS = [k for k, _ in _K]
COLORS = [c for _, c in _K]
IDX = {k: i + 1 for i, (k, _) in enumerate(_K)}                 # 1-based; 0 = sea
DESC = {"Af": "tropical rainforest", "Am": "tropical monsoon", "BWh": "hot desert", "BSh": "hot steppe",
        "BWk": "cold desert", "BSk": "cold steppe", "Cfb": "temperate oceanic", "ET": "tundra",
        "EF": "ice cap"}


def aridity_threshold_mm_yr(T_C):
    """Koppen's aridity threshold R_th = max(20*T + 140, 0) [mm/yr] -- THE one copy.

    Aseasonal, so Koppen's rainfall-regime choice collapses to the '+140' case (no dry season to key
    the +0/+280 variants on). A cell is arid when P < R_th, desert when P < R_th/2. The ``max(.., 0)``
    matters and is not cosmetic: below T = -7 C the linear form goes NEGATIVE, and Koppen's aridity
    line is then clipped to zero -- i.e. NOTHING is arid there, however dry it actually is (the
    formula is a proxy for potential evapotranspiration, and Koppen never meant it to run past
    freezing). That clip is exactly why ``csf_water_gate`` cannot use this threshold alone.

    DTYPE-PRESERVING on purpose: the hard classifier works in float64 and the soft twin
    (``map_webapp.mapdata.biome_blend_weights``) in float32, and both must be able to call this
    without a cast changing their arithmetic."""
    return np.maximum(20.0 * np.asarray(T_C) + 140.0, 0.0)


def koppen_class(T_C, P_yr, land, T_min_C=None, T_max_C=None):
    """Seasonless Koppen codes (0 = sea, 1.. = LABELS order). T_C [deg C], P_yr [mm/yr], land mask.

    Aridity threshold (aseasonal -> Koppen's '+140' case): R_th = 20*T + 140 mm/yr.
    Arid if P < R_th (BW if P < R_th/2 else BS; suffix h if T>=18 else k). Otherwise: A (T>=18; Af if
    P>=720 else Am). The warm/arid (A/B) and the temperate Cfb floor stay keyed on the day-MEAN.

    Frost-aware E family (Issue #21 follow-up). Frost is an EXTREME (does the surface ever REACH
    freezing?), so the mean is a poor proxy. When the air diurnal extremes ``T_min_C``/``T_max_C`` are
    BOTH supplied, the cold (E) family is re-keyed on them instead of the mean:
      * ``EF`` (ice cap) = the warmest instant of every (identical) day is still sub-freezing
        (``T_max_C < 0``) -> the surface NEVER thaws -- sharper and more physical than the old mean < 0;
      * ``ET`` (tundra) = freezes every night, thaws by day (``T_min_C < 0 <= T_max_C``), applied only
        BELOW the 10 C day-mean Cfb floor so a warm-mean climate with a big swing is never demoted;
      * a cool-mean but FROST-FREE cell (the residual mild class) folds into ``Cfb`` (oceanic).
    A second return value ``frosts`` (a boolean mask) flags any warmer (A/B/Cfb) land cell whose air
    night still freezes -- a non-destructive overlay so that signal is not lost where the mean is too
    warm to be E. When the extremes are omitted (both ``None``) the result is byte-identical to the
    prior mean-based scheme (Cfb 10<=T<18, ET 0<=T<10, EF T<0) and ONLY ``code`` is returned, so
    existing callers are untouched. No D, no dry-season, no Cfa -- forbidden on the seasonless world.
    """
    T = np.asarray(T_C, float); P = np.asarray(P_yr, float)
    Rth = aridity_threshold_mm_yr(T)
    code = np.zeros(T.shape, np.int16)
    arid = P < Rth
    BW = arid & (P < 0.5 * Rth); BS = arid & ~BW
    hot = T >= 18.0
    code[BW & hot] = IDX["BWh"]; code[BW & ~hot] = IDX["BWk"]
    code[BS & hot] = IDX["BSh"]; code[BS & ~hot] = IDX["BSk"]
    nona = ~arid
    A = nona & hot
    code[A & (P >= 720)] = IDX["Af"]; code[A & (P < 720)] = IDX["Am"]
    temperate = nona & ~hot                                   # the non-arid T<18 band: Cfb/ET/EF territory
    land_b = np.asarray(land, bool)
    if T_min_C is None or T_max_C is None:
        # mean-keyed fallback -- byte-identical to the prior scheme (existing-caller regression guard).
        code[temperate & (T >= 10.0)] = IDX["Cfb"]
        code[temperate & (T >= 0.0) & (T < 10.0)] = IDX["ET"]
        code[temperate & (T < 0.0)] = IDX["EF"]
        code[~land_b] = 0
        return code
    # extreme-keyed E family (the more-physical change), capped below the 10 C day-mean Cfb floor.
    Tmin = np.asarray(T_min_C, float); Tmax = np.asarray(T_max_C, float)
    code[temperate & (T >= 10.0)] = IDX["Cfb"]               # cap: warm-mean stays Cfb (frosts overlay below)
    cold = temperate & (T < 10.0)                            # below the cap: decide E on the air extremes
    EF = cold & (Tmax < 0.0)                                 # warmest instant still sub-freezing -> never thaws
    ET = cold & ~EF & (Tmin < 0.0)                           # freezes every night, thaws by day
    mild = cold & ~EF & ~ET                                  # cool but frost-free -> fold into Cfb (oceanic)
    code[EF] = IDX["EF"]; code[ET] = IDX["ET"]; code[mild] = IDX["Cfb"]
    code[~land_b] = 0
    # frosts overlay: any warmer (A/B/Cfb) land cell whose air night still freezes (non-destructive).
    warm_classes = np.isin(code, [IDX["Af"], IDX["Am"], IDX["BWh"], IDX["BWk"],
                                  IDX["BSh"], IDX["BSk"], IDX["Cfb"]])
    frosts = land_b & warm_classes & (Tmin < 0.0)
    return code, frosts


# habitability classes (the project's own scheme): is land livable? frost-prone sits between humid and
# frozen (Issue #21 follow-up): livable day-mean but a freezing air night.
HAB = [("ocean", "#cfe2f0"), ("desert", "#d9a441"), ("steppe", "#e8d18a"),
       ("humid", "#3aa83a"), ("frost-prone", "#86c5b4"), ("frozen", "#dfe9f2")]
HAB_IDX = {k: i for i, (k, _) in enumerate(HAB)}


def habitability_class(T_C, P_yr, land, T_min_C=None):
    """A coarse habitability scheme: ocean / hot desert / semi-arid steppe / temperate-humid /
    frost-prone / frozen. The only TEMPERATURE gates are frozen (day-mean<0) and frost-prone (livable
    day-mean but a freezing air night); the remaining livable, frost-free land is split by PRECIP alone:
    desert P<250 mm/yr; steppe 250-500; humid >500. (So a cool-but-frost-free arid cell still classifies
    desert -- there is no separate cold-desert class here, unlike Koppen's BWk; issue #88 item 4.)

    ``frost-prone`` (Issue #21 follow-up) is land with a LIVABLE day-mean (>=0) but a freezing AIR
    night (``T_min_C < 0``). Seasonlessness makes it load-bearing: with a fixed sun-track every night is
    identical, so this is not Earth's seasonal frost -- it means the ground freezes EVERY night, forever,
    with no frost-free season, a far harsher and more distinctive statement than a temperate-latitude
    analogue. Emitted only when ``T_min_C`` is supplied; when omitted the scheme is byte-identical to the
    prior mean-only one (no frost-prone cell)."""
    T = np.asarray(T_C, float); P = np.asarray(P_yr, float)
    land = np.asarray(land, bool)
    code = np.full(T.shape, HAB_IDX["ocean"], np.int16)
    frozen = land & (T < 0.0)
    if T_min_C is not None:
        frost_prone = land & ~frozen & (np.asarray(T_min_C, float) < 0.0)
    else:
        frost_prone = np.zeros(T.shape, bool)
    avail = land & ~frozen & ~frost_prone                    # livable, frost-free land split by precip
    desert = avail & (P < 250.0)
    steppe = avail & (P >= 250.0) & (P < 500.0)
    humid = avail & (P >= 500.0)
    code[frozen] = HAB_IDX["frozen"]; code[desert] = HAB_IDX["desert"]
    code[steppe] = HAB_IDX["steppe"]; code[humid] = HAB_IDX["humid"]
    code[frost_prone] = HAB_IDX["frost-prone"]
    return code


# ------------------------------------------------------------------ extended "Köppen-plus" classification
# Surviving seasonless-Köppen types PLUS the world-specific lethal/limiting types, in ONE priority-ordered
# mutually-exclusive scheme (spec + prose: cosmology_models/EXTENDED_CLIMATE_CLASSES.md). 0 = sea, 1.. =
# THE COOL ARID CLASSES SPLIT ON NIGHTLY FROST, NOT ON KOPPEN'S `k` (2026-07-27, Jake).
# Koppen's `k` (kalt) marks a desert with a FROZEN WINTER -- a season that stops growth for months --
# and this world has no seasons, so the letter imports a fact that is never true here.  Worse, the
# single T < 18 C class it defines is BIMODAL on this world: measured on the canonical map, 69% of
# its cells freeze every night (day-mean 1.5 C, night -7.2, day +10.1) and 31% never freeze at all
# (day-mean 13.4 C, night +3.8, day +22.6).  A 12 C gap in mean temperature inside one class is far
# coarser than the threshold that drew it, and the two halves are agriculturally opposite: one needs
# frost-hardy crops or protection every single night, the other is frost-free, year-round farmable
# ground that was being painted as "cold desert".
#
# So the cool arid classes split on `T_min < 0` -- the freezing point of water, a physical threshold
# on a real continuum, not a convention.  This is the SAME gate the scheme already uses for ET
# ("freezes every night, thaws by day"); it was simply never applied on the arid side, which is the
# inconsistency this removes.  Nothing else moves: `hot` (T >= 18 C) still separates BWh/BSh, and no
# hot arid cell freezes at night anyway (measured 0.00%), so the split cannot interact with it.
#   BWk / BSk  frost EVERY night, thaw EVERY day -- cold, but diurnally, never seasonally
#   BWt / BSt  cool and FROST-FREE all year
# The EARTH-Koppen table `_K` above keeps its own `BWk`/`BSk`: that scheme IS Koppen, and is only
# ever used for comparison against Earth.
# EXT_LABELS order = priority order (first match wins -> determinate). The five new types key on principled
# physical gates (day-mean surface T, annual P, warm-phase wet-bulb, day-mean surface sunlight). Night FROST
# is a cross-cutting MODIFIER rather than a base class, and is applied as the non-destructive overlay above.
# There is no ULTRAVIOLET overlay and nothing needs one: the suns emit nothing between 200 and 320 nm, so the
# only ultraviolet reaching the ground is the mild 320-400 nm band at roughly half an Earthly tropic noon's
# biologically-weighted dose (derive_spectrum_albedos.py section [6]). Uniform across the map and below the
# Earth comparator everywhere, it draws no boundary, which is exactly what a base class or an overlay is for.
# SF's wording, A5 Q6 (2026-08-23).  The old "salt flat — lethally hot, rainless evaporite pan" made
# two claims the gate does not support and one the salt model contradicts.  Measured on the shipped
# w12 window: the class covers 10 % of its land, NONE of it below datum and none in a basin, and the
# map's own DOC-20 salt verdict calls 38 of 40 sampled SF cells a wind-capped VENEER with a standing
# residue of 0.0 kg/m^2/yr -- ground the wind sweeps, not a standing evaporite bed.  The gate itself
# says only two things (surface peak past 70 °C, under 50 mm/yr), so the description now says exactly
# those two and what becomes of the salt.  The LABEL is untouched: `SF` is the stored identity and
# every palette, barren set and gate keys on that string.
_E = [("SF", "#5C0000",
       "baked flat — rainless, and the ground passes 70 °C in the sun; salt blows across it "
       "rather than standing on it"),
      ("XC", "#0B0B3B", "lethal cold — survivable only with serious protection; the approach to the cold trap"),
      # Law-agnostic wording (PR #453 review finding 1): the criterion values belong to the heat law
      # (see extended_class heat_law / EXTENDED_CLIMATE_CLASSES.md), not the legend — under law 1
      # these are the fixed 35/31 day-mean cuts, under law 2 the mean-vs-peak dryness-adjusted split.
      # The MEANINGS are the stable part (Jake): Wd = difficult, WW = impossible.
      ("WW", "#6A1B9A", "lethal heat — wet-bulb past the survivable limit around the clock (or an outright lethal peak); uninhabitable"),
      ("Wd", "#BA68C8", "dangerous heat — the hottest hours are dangerous even in shade; livable by sheltering through the peak"),
      # The description states the CLIMATE and nothing else: whether ice actually stands here is the
      # ice-cap layer's question, and the two derivations legitimately disagree in both directions
      # (EXTENDED_CLIMATE_CLASSES.md row 5; map_webapp/README.md; ICE_MARGIN_RESULTS.md section 1).
      ("EF", "#5A5A5A", "perennial frost — the daily high never clears 0 °C"),
      ("ET", "#B2B2B2", "tundra — freezes every night, thaws by day"),
      ("Sh-t", "#5B8A9A", "shaded temperate — mild but light-limited (perpetual overcast)"),
      ("Sh-c", "#3E5C6B", "shaded cool — cool, wet, dimmest livable ground"),
      ("BWh", "#FF0000", "hot desert"), ("BSh", "#F5A300", "hot steppe"),
      ("Af", "#0000FE", "tropical rainforest"), ("Am", "#0077FF", "tropical monsoon"),
      ("Cfb", "#32CD32", "temperate oceanic — the bright, farmable ribbon"),
      # The cool arid pairs say what the ground DOES (frost nightly vs never), not what Koppen's `k`
      # would have implied; the seasonless-world reasoning behind the split is the comment block above
      # and EXTENDED_CLIMATE_CLASSES.md, not the legend.
      # NO THAW CLAIM, and it must not come back: `EF` is gated `nona`, so ARID never-thaw ground has
      # nowhere to go but BWk/BSk, whose predicates read only `Tmin < 0` (unlike ET, which carries
      # `& (Tmax >= 0)`). Measured: 0.53 M km^2 of such ground under drawn ice alone -- the same
      # invariant mapdata.py's koppen note states as "NOT-EF DOES NOT IMPLY THAWS" (PR #554 review
      # finding 1). Each row now claims exactly what its gate implies: frost every night.
      ("BWk", "#FF9695", "cold desert — freezes every night"),
      ("BSk", "#FFDB63", "cold steppe — freezes every night"),
      ("BWt", "#E8C4B8", "temperate desert — cool and frost-free all year"),
      ("BSt", "#CFE0A0", "temperate steppe — cool and frost-free all year"),
      # MASK CLASS -- no predicate reaches it; the caller supplies the mask and law 3 paints it after
      # the select (GEO-21). Appended LAST so codes 1-17 keep their numbers on every class map ever
      # stored.
      # The NAME is deliberately temperature-neutral (2026-08-04). The class is GEOGRAPHIC -- a
      # drained below-datum floor losing water -- and the earlier "cool evaporite basin" wording
      # asserted something the definition does not test: 14 of the shipped window's 1,035 CSF cells
      # run above 30 °C day-mean (max 41.2 °C). The CODE is untouched: `CSF` is the stored identity.
      # "at any temperature" is how the DISPLAY says temperature-neutral (the wording PR #549 settled
      # on for SF_DESC_PRE_LAW3): it states the class's reach without a claim the gate never makes.
      ("CSF", "#9C9682",
       "salt basin — a drained below-datum floor losing more water than it receives, "
       "at any temperature"),
      # --- A5's three surface classes: what a `CSF` floor actually IS -----------------------------
      # MASK CLASSES TOO, and painted by a caller `extended_class` never sees: the split turns on a
      # basin's own brine level and on a deflation capacity, neither of which this module has or
      # should have (they come out of the serving side's river job).  `extended_class` therefore
      # still paints only `CSF`, exactly as before -- these three exist here so the VOCABULARY,
      # the colors and the barren/palette tables live in one place, the way every other class does.
      # `map_webapp.mapdata._koppen_classify` is the one caller that assigns them, and it assigns
      # them as a PARTITION of `CSF`: every cell that was `CSF` becomes exactly one of these three
      # and no cell outside `CSF` becomes any of them, which is why the split needed no new law.
      # Investigation, criteria and per-basin verdicts: map_webapp/SALT_BASINS_INVESTIGATION.md.
      # Colors: all three sit in `CSF`'s grey-tan family so a reader who knew the old swatch still
      # reads them as the same kind of ground -- FZF toward the cold blue-grey, BPN darker and damper
      # than the old anchor, WSP the bright dry crust the old anchor was too dark for.
      ("FZF", "#8E9AA6",
       "frozen basin floor — a drained below-datum floor that stays frozen: nothing liquid to "
       "move its salt, which banks in the ice and leaves again at the melting margin"),
      ("BPN", "#8C8878",
       "brine pan — a drained below-datum floor that never dries out, so its salt stands as a "
       "damp crust over brine instead of blowing away on the wind"),
      ("WSP", "#BFB9A6",
       "wind-scoured playa — a drained below-datum floor that dries for part of each day, where "
       "the wind carries off more salt than the sky delivers and bare clay and silt are left")]
EXT_LABELS = [a for a, _, _ in _E]
EXT_COLORS = [b for _, b, _ in _E]
EXT_DESC = {a: c for a, _, c in _E}
EXT_IDX = {a: i + 1 for i, (a, _, _) in enumerate(_E)}          # 1-based; 0 = sea

# Classes NO predicate in `extended_class` can select: a caller hands in the mask and the class is
# painted after the `np.select`. They sit at the END of `_E` so adding one never renumbers a stored
# class map, and `conds` must stay exactly `len(_E) - len(_MASK_CLASSES)` long -- `np.select` silently
# DROPS trailing choices when the two lists disagree, which mis-codes the tail of the table instead of
# failing, so the length is asserted at import and again at the call.
_MASK_CLASSES = ("CSF", "FZF", "BPN", "WSP")
_N_PREDICATE_CLASSES = 17                                       # rows of `conds` in extended_class
assert len(_E) - len(_MASK_CLASSES) == _N_PREDICATE_CLASSES, \
    f"extended-class table has {len(_E) - len(_MASK_CLASSES)} predicate classes, conds has {_N_PREDICATE_CLASSES}"
assert EXT_LABELS[-len(_MASK_CLASSES):] == list(_MASK_CLASSES), "mask classes must be LAST in _E"

SF_T_SURF_MAX_C = 70.0                                          # SF's diurnal surface-peak gate [°C]
SF_P_MAX_MM_YR = 50.0                                           # ...and its rainless gate [mm/yr]


def salt_flat_gate(T_surf_max_C, P_yr):
    """SF's own condition: the diurnal SURFACE peak clears 70 °C AND the cell is rainless (< 50 mm/yr).

    Factored out of ``conds[0]`` because law 3's CSF override has to ask the same question from
    OUTSIDE the classifier -- a drained basin floor sits below the datum, so it is not in ``land`` and
    ``extended_class`` can never paint it (see the ``basin_mask`` note there) -- and a second
    hand-written copy of the threshold pair is exactly what drifts."""
    return ((np.asarray(T_surf_max_C, float) > SF_T_SURF_MAX_C)
            & (np.asarray(P_yr, float) < SF_P_MAX_MM_YR))


def csf_water_gate(T_C, P_yr):
    """CSF's WATER-BALANCE condition: does this floor actually lose more water than it receives?

        P < max(R_th(T), SF_P_MAX_MM_YR)   with R_th = ``aridity_threshold_mm_yr``

    A drained below-datum floor is a closed sink -- it has no outlet, so whatever arrives leaves as
    vapour or stays. That makes it an EVAPORITE pan only where the arriving water is small enough to
    be removed; a closed floor under a genuinely wet climate is a LAKE, not a salt pan, and calling it
    one is the failure this gate exists to prevent. (The class layer states the climate of the floor;
    what standing water ends up on it is the display side's ``display_drainage`` question, which the
    classifier deliberately does not consult -- see EXTENDED_CLIMATE_CLASSES.md.)

    The predicate is deliberately a MAXIMUM of two thresholds, because neither alone is right over the
    whole temperature range this world spans:

    * **Koppen aridity where the formula means something (warm).** ``R_th`` is Koppen's own
      evaporative-demand proxy, so ``P < R_th`` is the standard "evaporation beats rainfall" line and
      is what decides a warm floor.
    * **An absolute hyper-arid floor where it does not (cold).** ``R_th`` is clipped to 0 below
      -7 C, so a bare ``P < R_th`` would call NOTHING arid there and destroy the cold half of the
      class outright (measured on the shipped window: 781 of 967 CSF cells). But water still leaves a
      cold closed basin -- by SUBLIMATION if never by melt -- so an inflow under
      ``SF_P_MAX_MM_YR`` = 50 mm/yr loses to vapour at ANY temperature. That is the same "rainless"
      cut ``salt_flat_gate`` uses, deliberately the SAME named constant: the two gates ask one
      question ("is this ground rainless?") and must move together.

    Note there is deliberately **no thaw gate**. A never-thawing bone-dry floor is a brine pan, not an
    ice cap -- Antarctica's Don Juan Pond is the real case -- because no melt is not the same as no
    vapour loss; letting such a cell fall through to ``EF`` would claim perennial ice on ground that
    has no supply to build it from."""
    return np.asarray(P_yr, float) < np.maximum(aridity_threshold_mm_yr(np.asarray(T_C, float)),
                                                SF_P_MAX_MM_YR)


def heat_thresholds_C(T_max_C, wet_bulb_high_C):
    """Dryness-adjusted humid-heat thresholds [°C] for the WW/Wd classes (GEO-20b) -> ``(ww, wd)``.

    The fixed 35/31 °C cut is the theoretical humid limit and is far too permissive in DRY heat: the
    Penn State Noll Laboratory work (Vecellio et al.) puts the critical wet-bulb near ~31 °C in
    warm-humid air and WELL below 35 in hot-dry air -- at ~55 °C dry-bulb / ~0% RH (marginally
    survivable) the wet-bulb is only ~20 °C. So the lethal threshold is a function of dryness, indexed
    by the daily-high wet-bulb DEPRESSION ``D = T_max - T_wb_high`` (0 at saturation -- a real physical
    continuum, so every boundary case grades determinately, per the project's unambiguity test):

        ww = clip(31 - (11/35) * D, 20, 31)      # 31 °C at saturation -> 20 °C at the D~35 K dry limit
        wd = ww - 4                              # "dangerous" a fixed 4 K below lethal (the old 35-31 gap)

    Anchors: D=0 -> 31 (the measured humid critical wet-bulb, NOT the 35 °C theoretical limit);
    D=35 K (~55 °C dry / ~0% RH) -> 20. Linear between, clipped -- the two anchor points are the
    defensible data, a shape beyond linear is not.

    Scale note (PR #453 review): the anchors are stated on the TRUE-wet-bulb scale, but the field
    compared against them (``thermo.wet_bulb_high``) rides Stull's RH >= 5% clip, which floors the
    reported wet-bulb above ~51 °C daily-high dry-bulb (~23.4 at the 55-dry anchor vs the true
    ~20.1). Near the dry limit the comparison is therefore ~2-3 K generous. Not live today -- every
    clipped land cell on the shipped archive classifies SF first, re-measured after GEO-21's reshuffle
    (PR #454) and still true -- but that shield is EMPIRICAL, not structural, so the mismatch is
    documented here rather than silently absorbed (EXTENDED_CLIMATE_CLASSES.md, the CSF section)."""
    D = np.maximum(np.asarray(T_max_C, float) - np.asarray(wet_bulb_high_C, float), 0.0)
    ww = np.clip(31.0 - (11.0 / 35.0) * D, 20.0, 31.0)
    return ww, ww - 4.0


def extended_class(T_C, P_yr, land, T_min_C, T_max_C, wet_bulb_C, surf_sw_wm2, T_surf_max_C=None,
                   wet_floor_mm=0.0, wet_bulb_high_C=None, heat_law=1, *, basin_mask=None):
    """Extended seasonless climate class per cell (0 = sea, 1.. = EXT_LABELS, priority order).

    Inputs: day-mean surface ``T_C`` [°C], annual ``P_yr`` [mm/yr], ``land`` mask, diurnal AIR extremes
    ``T_min_C``/``T_max_C`` [°C], the ``wet_bulb_C`` [°C] and day-mean surface sunlight ``surf_sw_wm2``
    [W/m²]. ``T_surf_max_C`` is the diurnal SURFACE-peak T [°C] for the salt-flat gate (the ground reaches
    70–300 °C even where the day-mean air is ~45; defaults to ``T_C`` if omitted). ``wet_floor_mm`` is an
    OPTIONAL precipitation floor for the humid-heat classes; it DEFAULTS TO 0 (no floor) because wet-bulb is
    the principled lethality metric whether a cell is lethal by high humidity OR by an extreme dry-bulb (a
    deadly wet-bulb is reachable at modest RH if the air is hot enough), so a low-LOCAL-precip but lethal-
    wet-bulb cell is humid heat, not desert (Jake, 2026-06-26). The rainless DRY extreme is removed first by
    the SF salt-flat gate (surface-peak T > 70 °C); genuinely dry deserts have low wet-bulb and stay BW*/BS*.
    wet_bulb/surf_sw may be 1-D (broadcast) or 2-D. The
    lethal/limiting types take priority; the frost-defined ice/tundra and the light-limited "shaded" types are
    gated to NON-arid land (Köppen B-before-E precedence). See EXTENDED_CLIMATE_CLASSES.md.

    ``heat_law`` selects the WW/Wd criterion (GEO-20). 1 (default, byte-identical): the shipped rows --
    DAY-MEAN wet-bulb against the fixed 35/31 °C cut. 2: the mean-vs-peak split against the
    dryness-adjusted ``heat_thresholds_C`` -- **Wd** when the daily-HIGH wet-bulb ``wet_bulb_high_C``
    (required then) comes within 4 K of its critical (dangerous hours, shelterable), **WW** when the
    day-MEAN wet-bulb exceeds the critical at its own depression, or the high passes 35 °C outright
    (uninhabitable). The authoritative statement of law 2, with the Earth calibration that set the
    split, is EXTENDED_CLIMATE_CLASSES.md.
    3 (GEO-21) is a CLASSIFICATION-law version, not a third heat law: law 2's thresholds UNCHANGED
    plus the ``CSF`` mask class. ``basin_mask`` (required then) marks the drained below-datum basin
    floors, and a masked LAND cell becomes ``CSF`` when it is not itself a genuine hot pan
    (``salt_flat_gate``) AND it is losing water (``csf_water_gate``) -- so "salt flat" goes back to
    meaning lethally hot, a sub-lethal salt basin gets its own class, and a closed floor under a wet
    climate (a lake basin, not a pan) falls THROUGH to the ordinary tree instead of being called
    evaporite. The env/stamp names stay COSMO_HEAT_CLASS_LAW / ``koppen_law``: the law
    selects a whole classification, and heat is only the part of it that moved first.
    ``basin_mask`` is IGNORED below law 3 rather than rejected -- the stamp decides which law runs, so
    a caller holding the mask passes it unconditionally instead of branching on the law it is about to
    hand over. Note the override can only reach LAND: ``code[~land] = 0`` is the last word here, so a
    caller whose basin floors sit OUTSIDE its land mask (every fine-grid display path -- a floor below
    the datum is not ``elev > 0``) must apply the same ``salt_flat_gate``/``csf_water_gate`` split
    itself afterwards. Such a caller should also hand in ``land | basin_mask`` as ``land``, so a floor
    that FALLS THROUGH the water gate is classified by the ordinary tree instead of being zeroed back
    to sea (a transparent hole on every display path); below law 3 that substitution changes nothing,
    because the caller's own override then paints every masked cell regardless.
    Callers flip via COSMO_HEAT_CLASS_LAW at archive-build time; consumers of a STORED class map must
    use the law stamped in the archive (``koppen_law``), never the env -- stored code != stored law is
    exactly the invariant test_stored_koppen_matches_recompute exists to hold."""
    T = np.asarray(T_C, float); P = np.asarray(P_yr, float); land_b = np.asarray(land, bool)
    Tmin = np.asarray(T_min_C, float); Tmax = np.asarray(T_max_C, float)
    Tsm = T if T_surf_max_C is None else np.broadcast_to(np.asarray(T_surf_max_C, float), T.shape)
    wb = np.broadcast_to(np.asarray(wet_bulb_C, float), T.shape)
    sw = np.broadcast_to(np.asarray(surf_sw_wm2, float), T.shape)
    wet = P >= wet_floor_mm                                     # optional precip floor (default 0 = off)
    if heat_law == 1:
        ww_hot, wd_hot = wb > 35.0, wb > 31.0
    elif heat_law in (2, 3):
        if wet_bulb_high_C is None:
            raise ValueError(f"extended_class: heat_law={heat_law} needs wet_bulb_high_C "
                             f"(the daily-HIGH wet-bulb)")
        if heat_law == 3 and basin_mask is None:
            raise ValueError("extended_class: heat_law=3 needs basin_mask (the drained below-datum "
                             "basin floors); pass an all-False mask for a world that has none")
        # The two tiers read two different EXPOSURE STATISTICS, so the labels keep their meanings
        # (Jake, 2026-08-02, on the PR #453 review's Earth check: Kuwait City must not read
        # "unsurvivable" -- inhabited deserts shelter through the peak; move the thresholds, not the
        # labels). Wd = the daily-HIGH wet-bulb reaches within 4 K of its dryness-adjusted critical:
        # the hottest hours are dangerous, life is possible by sheltering through them -- Kuwait,
        # Jacobabad, Death Valley all land here on their hot-month climatology. WW = the day-MEAN
        # wet-bulb exceeds the critical at ITS OWN depression (heat uncompensable around the clock,
        # no cool hours to shed the load -- nowhere on present Earth) OR the daily high clears the
        # 35 °C theoretical limit outright (brief exposure kills regardless of respite -- the
        # original "a daily high of 40 kills you even if the mean is 34" case). Wet-bulb is already
        # the in-shade metric, so "shelter" buys survival only BELOW critical, which is what makes
        # the mean-vs-peak split the determinate reading of difficult-vs-impossible.
        wbh = np.broadcast_to(np.asarray(wet_bulb_high_C, float), T.shape)
        crit_mean, _ = heat_thresholds_C(T, wb)                 # critical at the day-mean depression
        _, wd_thr = heat_thresholds_C(Tmax, wbh)                # wd_thr = the daily-high critical - 4
        ww_hot = (wb > crit_mean) | (wbh > 35.0)
        wd_hot = wbh > wd_thr
        # WW => Wd holds EMPIRICALLY on the shipped archive (0 violating cells) but is NOT structural:
        # the RH clip can inflate wbh, shrink D_high, and push the high's critical above the mean's.
        # The conds order (WW before Wd) is what arbitrates -- don't build later logic on "WW is a
        # strictly worse Wd" without re-checking (PR #453 re-review).
    else:
        raise ValueError(f"extended_class: unknown heat_law {heat_law!r} (1, 2 or 3)")
    Rth = aridity_threshold_mm_yr(T)
    arid = P < Rth; BW = arid & (P < 0.5 * Rth); hot = T >= 18.0
    nona = ~arid; frostfree = Tmin >= 0.0
    conds = [
        salt_flat_gate(Tsm, P),                                 # SF  salt flat (surface-peak T; rainless)
        T < -78.0,                                              # XC  lethal cold
        wet & ww_hot,                                           # WW  lethal heat (wet-bulb; floor off by default)
        wet & wd_hot,                                           # Wd  dangerous heat
        nona & (Tmax < 0.0),                                    # EF  ice cap (wet cold; arid cold -> BWk/BWt)
        nona & (Tmin < 0.0) & (Tmax >= 0.0) & (T < 10.0),       # ET  tundra
        nona & (T >= 10.0) & (sw < 20.0) & frostfree,           # Sh-t shaded temperate (dim, frost-free)
        nona & (T >= 0.0) & (T < 10.0) & (sw < 20.0) & frostfree,  # Sh-c shaded cool
        arid & BW & hot,                                        # BWh
        arid & ~BW & hot,                                       # BSh
        nona & hot & (P >= 720.0),                              # Af
        nona & hot & (P < 720.0),                              # Am
        nona & ~hot,                                            # Cfb (remaining temperate: frost-free, bright)
        arid & BW & ~hot & ~frostfree,                          # BWk  cold desert (frost nightly)
        arid & ~BW & ~hot & ~frostfree,                         # BSk  cold steppe (frost nightly)
        arid & BW & ~hot & frostfree,                           # BWt  temperate desert (frost-free)
        arid & ~BW & ~hot & frostfree,                          # BSt  temperate steppe (frost-free)
    ]
    assert len(conds) == len(_E) - len(_MASK_CLASSES), \
        f"conds has {len(conds)} rows for {len(_E) - len(_MASK_CLASSES)} predicate classes"
    code = np.select(conds, list(range(1, len(conds) + 1)), default=0).astype(np.int16)
    if heat_law == 3:                                           # the CSF mask class (GEO-21)
        # conds[0] IS the SF gate, on the arrays it was broadcast against -- reuse it rather than
        # re-deriving, so a hot pan inside the mask cannot end up split between the two rules.
        # The WATER gate is the second half of the rule: a masked floor is an evaporite pan only where
        # it actually loses more water than it receives, so a WET closed floor (a lake basin) falls
        # THROUGH to the ordinary tree and keeps whatever climate class it earns. Measured on the
        # shipped seed-90 window the gate keeps all 967 masked cells, but it is not vacuous there --
        # ~850 are decided by the 50 mm/yr rainless floor (cold, where R_th has clipped to 0) and
        # ~117 by R_th itself, the smallest margin ~6 mm/yr.
        bm = np.broadcast_to(np.asarray(basin_mask, bool), code.shape)
        code[bm & land_b & ~np.broadcast_to(conds[0], code.shape)
             & csf_water_gate(T, P)] = EXT_IDX["CSF"]
    code[~land_b] = 0
    return code


def class_fractions(code, labels, land=None):
    """Return {label: fraction of LAND cells} for a class map (sea = code 0 excluded for Koppen/extended).

    Pass ``land`` (the true continent mask) for the EXTENDED scheme: ``classify_npz`` reclassifies hot
    OCEAN cells to the SF code in place, so ``code != 0`` would wrongly count those sea salt flats as land.
    With ``land`` given, the denominator and numerators are restricted to genuine land (sea salt flats are
    a sea feature -- drawn on the map, not counted in the land class-share)."""
    one_based = labels is LABELS or labels is EXT_LABELS
    if land is not None:
        land_cells = np.asarray(land, bool)
    else:
        land_cells = code != 0 if one_based else code != HAB_IDX["ocean"]
    n = max(int(np.sum(land_cells)), 1)
    out = {}
    for i, lab in enumerate(labels):
        idx = (i + 1) if one_based else i
        frac = float(np.sum((code == idx) & land_cells)) / n
        if frac > 0:
            out[lab] = frac
    return out
