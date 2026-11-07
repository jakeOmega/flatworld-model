"""
flatworld.surface_wind -- the honest near-surface (10 m) wind diagnostic, reduced from the model's
lowest dynamics level.

Why this exists (the wind-habitability question, 2026-06-12): every wind number the model quotes --
the AD-core's ``u_surf``, Stage C's balanced ``|V|`` (whose background IS that u_surf) -- lives at
the LOWEST MODEL LEVEL, z1 = dz above the ground, under the bulk-drag closure. On the CANONICAL
BL-refined stretched grid that level is z1 = 20 m (``dz_min = 20 m`` -- set CONFIG-side by the
production driver, NOT by ``canonical_env.sh``, so reading the env alone misleads: it implies the
uniform grid and a ~140 m level; the old nz=18 fleet sat at ~1.2 km). Whatever its height, it is a
boundary-layer DAY-MEAN wind, not what anyone standing on the ground feels; habitability lives at
screen level. This module reduces the lowest-level wind to 10-m day/night estimates with explicit,
documented assumptions. It is a DIAGNOSTIC POST-PROCESS: it changes no prognostic field and no
climate. The level->screen bridge for a surface SOLVE (not habitability) is ``bulk_flux_wind`` below
(ROADMAP GEO-8); read ITS docstring for the convention/target caveat.

Method (deliberately simple; the reduction's FORM is standard surface-layer theory, its INPUTS are
model-soft -- quote band ranges, not pointwise precision):

1. WELL-MIXED DAYTIME BL. u(z1) is taken as the mixed-layer wind U_m: a convectively-mixed BL is
   near-uniform in speed above its surface layer, so with z1 at/below the assumed BL depth (Stage C's
   H_bl = 1 km) no further shear correction is applied between z1 and the surface-layer top. [The
   older mixed-layer reading assumed a ~1.2 km z1 sitting AT the BL depth; on the canonical z1 = 20 m
   grid pass ``z1`` to ``u10_factor`` to match from the actual level instead -- see its docstring.
   Assumption, model-soft: where the real BL is shallower than the match height the 10-m estimate is
   an over-estimate -- conservative for habitability.]

2. NEUTRAL LOG SURFACE LAYER. The lowest ~10% of the BL follows u(z) = (u*/k) ln(z/z0). Matching
   U_m at the surface-layer top z_sl = max(sl_frac*h_bl, 3*z_ref):

       u10 = U_m * ln(z_ref/z0) / ln(z_sl/z0)

   with z0 from a Koppen-informed roughness map (sea / ice / desert / steppe / vegetated). This is
   the daytime ("coupled") wind: gusts reach it, lulls sit below it.

3. NOCTURNAL DECOUPLING (land only). With surface radiative cooling the surface layer stratifies
   and decouples from the BL flow (the BL wind rides on as a detached jet aloft); 10-m winds
   collapse toward a fraction of their daytime value. We grade the collapse by the cell's own
   screen-air diurnal range dT_air (the model's measure of how strongly that surface cools):
   factor = 1 - (1 - night_floor) * clip(dT_air / decouple_dT, 0, 1). Over sea (negligible swing):
   no decoupling. [Empirical form, model-soft; the SIGN (calm nights over strongly-cooling land) is
   firm, independent of the exact day/night split.]

Confidence tiers for anything quoted from here: reduction-factor FORM [firm], factor VALUES
[model-soft via z0/h_bl classes], input u_lev [model-soft: diagnostic balance + bulk closure].

The ladder has two further [model-soft] diagnostic layers (issue #114), both post-process, no climate
change:

 * THE TERRAIN FACTORS -- shelter and lee amplification as the two branches of ONE regime law
   (``terrain_factors``; ROADMAP DOC-34). Stratified flow meeting a ridge either rides over it
   (nondimensional mountain height M = N*H/U << 1: no effect), breaks over it (M ~ 1: a downslope
   WINDSTORM, 2-3x in the lee -- gap jets, foehn) or is blocked by it (M >> 1: a stagnant SHELTERED
   wake). ``shelter_from_M`` and ``amplification_from_M`` are those two branches, sequenced in M so
   they never both act strongly on one cell, and the effect is applied only within the ridge's
   FOOTPRINT (a few tens of km beyond the lee slope). ``shelter_factor`` / ``attach_shelter`` and
   ``terrain_amplification`` / ``attach_amplification`` serve the branches separately (their product
   is the unified factor); the regime consumes #111's per-cell hydraulic Froude when present, else
   the topo+stratification estimate. Amplification-prone cells are carved OUT of ``calm_habitable``.
   Sub-grid twin of ``orographic_z0`` is ``orographic_amplification`` (capped).

 * GUST factors (``gust_factor`` / ``attach_gust``). Everything else here is a time-MEAN wind;
   habitability also keys on gusts. G = 1 + g_peak*I_u (turbulence intensity I_u from z0), plus a
   convective cold-pool outflow ~sqrt(2*CAPE) where CAPE is supplied. Quote winds as mean AND gust.

The amplification/gust constants are LITERATURE-ANCHORED, not derived -- documented at each function.
"""
from __future__ import annotations
import numpy as np

# roughness lengths z0 [m] -- standard surface-layer values by cover class
Z0_SEA = 2.0e-4        # open water (low-wind Charnock-ish; the cell's seas are not fetch-limited)
Z0_ICE = 1.0e-3        # smooth snow/ice sheet
Z0_LAND_DEFAULT = 0.05 # mixed open terrain (used when no Koppen map is supplied)

# Koppen class -> z0 [m]: what the CLASS says grows/lies there sets the roughness.
KOPPEN_Z0 = {
    "Af": 1.5,    # tropical rainforest: tall closed canopy
    "Am": 1.0,    # monsoon forest
    "Cfb": 0.8,   # temperate oceanic woodland/mosaic
    "BSh": 0.01,  # hot steppe: sparse low cover
    "BSk": 0.01,  # cold steppe (frost every night)
    "BSt": 0.02,  # temperate steppe: frost-free, so a taller/denser sward than BSk
    "BWh": 3e-3,  # hot desert: bare/gravel
    "BWk": 3e-3,  # cold desert (frost every night)
    "BWt": 3e-3,  # temperate desert: bare/gravel too -- frost-free, but still desert
    "ET": 0.03,   # tundra: low shrub/sedge
    "EF": Z0_ICE, # ice cap
}


def roughness_map(land, koppen=None, labels=None):
    """z0(x,e) [m] from the land mask and (optionally) the seasonless-Koppen code map.

    ``koppen`` is the classify.koppen_class code array (0 = sea, 1.. = labels order); ``labels``
    defaults to classify.LABELS. Without a Koppen map all land gets Z0_LAND_DEFAULT."""
    land = np.asarray(land, bool)
    z0 = np.where(land, Z0_LAND_DEFAULT, Z0_SEA).astype(float)
    if koppen is not None:
        if labels is None:
            from . import classify
            labels = classify.LABELS
        code = np.asarray(koppen, int)
        for i, lab in enumerate(labels):
            if lab in KOPPEN_Z0:
                z0[land & (code == i + 1)] = KOPPEN_Z0[lab]
    return z0


def orographic_z0(relief_m, wavelength_m, z0_cap=5.0):
    """EFFECTIVE roughness length [m] for SUB-GRID terrain (form drag), Lettau's relation for
    ridge-and-valley relief: z0_eff ~ 0.5 * H^2 / lambda (H the ridge-to-valley relief, lambda the
    ridge spacing), capped at z0_cap (the literature's mountainous-terrain ceiling; Lettau
    over-extrapolates for steep relief). [model-soft]

    Use to estimate what relief BELOW the model's grid does to the AREA-MEAN 10-m wind: pass the
    result as z0 to u10_factor. It does NOT resolve where in the sub-grid terrain the wind goes --
    valley floors sit further below this mean (Winstral-type sheltering, lee separation: another
    ~30-60 % at real ridge scales), crests above it."""
    z0 = 0.5 * np.asarray(relief_m, float) ** 2 / np.asarray(wavelength_m, float)
    return np.minimum(z0, float(z0_cap))


def u10_factor(z0, h_bl=1000.0, z_ref=10.0, sl_frac=0.10, z1=None):
    """Neutral log-layer reduction factor u(z_ref)/U_m for mixed-layer wind U_m (see module doc).

    z_sl = max(sl_frac*h_bl, 3*z_ref) keeps the match height above z_ref on shallow BLs. Factor is
    in (0, 1) for any z0 < z_ref.

    ``z1`` (lowest MODEL-LEVEL height [m], optional -- BL_REFINEMENT_SPEC.md #F): with the BL-refined
    stretched grid the model's lowest level can sit at/near z_ref, so the input wind already IS the
    z_ref wind and the mixed-layer log extrapolation no longer applies. When ``z1`` is given the match
    height is ``z1`` itself (reduce from the ACTUAL lowest level, not the assumed surface-layer top), so
    the factor -> 1 as z1 -> z_ref; when ``z1 <= z_ref`` the factor is exactly 1 (the lowest level is at
    or below screen height -- no extrapolation, short-circuited BEFORE the z0<z_ref guard). ``z1=None``
    (default) keeps the mixed-layer reduction -- byte-identical to the pre-refinement diagnostic."""
    z0 = np.asarray(z0, float)
    if z1 is not None and float(z1) <= float(z_ref):
        return np.ones_like(z0)                                 # lowest level <= z_ref: it IS the z_ref wind
    z_sl = max(sl_frac * float(h_bl), 3.0 * float(z_ref)) if z1 is None else float(z1)
    if np.any(z0 >= z_ref):
        raise ValueError(f"u10_factor: z0 must be < z_ref={z_ref} m (got max {z0.max():.3g}).")
    return np.log(z_ref / z0) / np.log(z_sl / z0)


def night_factor(land, dT_air_C, night_floor=0.25, decouple_dT=8.0):
    """Nocturnal-decoupling multiplier on the 10-m DAY wind (1 over sea; toward night_floor over
    strongly-cooling land). dT_air_C is the screen-air diurnal range map (dC['dT_air_C']).

    NB (issue #113 finding): this models the SURFACE-LAYER nocturnal decoupling -- the stable boundary
    layer detaching the 10-m wind from the BL flow aloft, an ~60% reduction BELOW the model's lowest
    level (whatever its height -- z1 = 20 m on the canonical BL-refined grid, ~1.2 km on the old nz=18
    fleet). It is NOT the diurnal sea/land breeze. #113's dynamical breeze solve
    (flatworld.bl_diurnal) resolves the breeze and found it a SEPARATE, smaller (~few-%), orientation-keyed
    effect that barely modulates the lowest-level SPEED on the windy band -- so it does NOT supersede this
    factor (an earlier plan to replace night_factor with a 'measured day/night ratio' rested on conflating
    the two mechanisms). This parameterized surface-layer decoupling stays the best estimate of the
    nocturnal 10-m calm, which the lowest-level dynamical solve cannot see."""
    land = np.asarray(land, bool)
    ramp = np.clip(np.asarray(dT_air_C, float) / float(decouple_dT), 0.0, 1.0)
    f = 1.0 - (1.0 - float(night_floor)) * ramp
    return np.where(land, f, 1.0)


def bulk_flux_wind(level_wind, land, z0, dT_air_C, h_bl=1000.0, z1=None, z_ref=10.0,
                   day_fraction=0.5, night_floor=0.25, decouple_dT=8.0):
    """ONE effective SCREEN-LEVEL wind speed [m/s] for a constant-wind bulk-flux surface solve, reduced
    from a lowest-model-level / boundary-layer wind under a NAMED, SIMPLE convention -- NOT the uniquely
    correct value (see the caveat at the end; there is no single "right" one).

    WHY THIS EXISTS (ROADMAP GEO-8). ``flatworld.surface.turbulent_fluxes`` closes the sensible heat
    (SH) and latent heat (LE) fluxes with bulk transfer coefficients ``Ch = Ce = 1.2e-3``, which are
    the standard values for a **10 m** reference wind. Every wind the model itself carries -- the AD
    core's ``u_surf``, Stage C's balanced ``|V|`` (solved over an H_bl = 1000 m layer) -- is a
    lowest-model-level / boundary-layer DAY-MEAN wind (module doc above). Feeding one of those straight
    into a 10-m bulk formula roughly doubles SH and LE. This function is a documented bridge, so a
    caller that wants "the model's wind" in a surface solve gets a screen-level equivalent instead of
    the raw level value.

    WHICH TERM CARRIES THE REDUCTION -- do not mis-attribute it to height. On the canonical BL-refined
    grid the lowest level sits at only **z1 = 20 m** (``dz_min = 20 m``; NB this is set CONFIG-side by
    ``run_driver`` -> ``_sl_dt0_gpu.build_donor_core_dev``, *not* by ``canonical_env.sh``, so reading
    the env alone gives the uniform grid and a wrong ~140 m), and the AD core's own surface drag applies
    ``tau = rho*C_D*|u1|*u1`` with the SAME 10-m ``C_D = 1.2e-3`` at that level -- i.e. the dynamics
    already treat ``u1`` as a ~10-m wind. The log-layer step is therefore worth 8-22 %, not a factor 2.
    What carries the rest is (2) + (3) below: the nocturnal decoupling and the day/night average.

    The reduction is the module's own ladder, in order: (1) the neutral log surface layer,
    ``u10_factor`` (pass ``z1`` to match from the ACTUAL lowest model level rather than the assumed
    surface-layer top); (2) nocturnal decoupling over land, ``night_factor``, keyed on the screen-air
    diurnal range; (3) a day/night time average, because a constant-wind diurnal solve integrates
    ONE wind through both halves of the day.

    ⚠️ TWO LAYERS OF SOFTNESS, and the SECOND is a CONVENTION, not just a soft parameter:
    - The parameters ``night_floor`` / ``decouple_dT`` / ``day_fraction`` are model-soft [values]; the
      reduction FORM is standard surface-layer theory [firm]. ``day_fraction`` (daylit share of the
      24 h day; realized 12.6/24 = 0.525 vs the 0.5 default -> ~3 %) is not load-bearing.
    - **The AVERAGING FUNCTIONAL itself is a chosen convention, and the RIGHT single wind is
      TARGET-DEPENDENT.** This returns the ``day_fraction``-weighted arithmetic mean of the day and
      night winds. But for a flux LINEAR in ``u`` the flux-conserving single wind is the time-mean of
      ``u`` (~this arithmetic mean), whereas the diurnal EXTREMES key on different winds: ``Ts_max`` is
      set by the DAY wind, ``Ts_min`` by the NIGHT wind. A solve pinned to reproduce the diurnal RANGE
      (the ice-margin use) is therefore not served by any single mean. Moreover stable nocturnal
      stratification collapses ``Ch`` below its neutral value, so the effective night wind is LOWER
      than the decoupled speed this uses -- i.e. the real spread is WIDER and TWO-SIDED, not just the
      one-sided ``night_floor`` band. **So treat the output as ONE convention's estimate, quote a band,
      and do not read "X sits at the low edge of the bracket" as a pin** -- the bracket here spans this
      convention's soft parameters, not the space of admissible conventions. Returns an array shaped
      like ``level_wind``."""
    f10 = u10_factor(z0, h_bl=h_bl, z_ref=z_ref, z1=z1)
    day = np.asarray(level_wind, float) * f10
    night = day * night_factor(land, dT_air_C, night_floor=night_floor, decouple_dT=decouple_dT)
    w = float(day_fraction)
    return w * day + (1.0 - w) * night


def attach(dC, koppen=None, h_bl=1000.0, labels=None, z_ref=10.0, z1=None):
    """Compute the near-surface day/night wind maps from a Stage C diagnostics dict and ATTACH them
    to it (keys: z0, u10_factor, u10_day_ms, u10_night_ms). Returns a small summary dict (band/
    all-land means) for reports. ``h_bl`` should match the Stage C config's H_bl. ``z_ref`` is the
    estimate height [m]: 10 (default, the meteorological standard) or 2 (head height -- NOTE the
    log law is invalid below a tall canopy; over forest classes the 2-m number means a clearing).

    ``z1`` (lowest model-level height [m], optional): pass it on a BL-refined run so the reduction
    matches from the ACTUAL lowest level (-> ~1 when z1 ~ z_ref) instead of log-extrapolating from the
    mixed layer; default None keeps the mixed-layer reduction (byte-identical). See ``u10_factor``."""
    u = np.asarray(dC["u"], float); v = np.asarray(dC["v"], float)
    land = np.asarray(dC["land"], bool)
    spd = np.hypot(u, v)
    z0 = roughness_map(land, koppen=koppen, labels=labels)
    f10 = u10_factor(z0, h_bl=h_bl, z_ref=z_ref, z1=z1)
    u10_day = spd * f10
    dT_air = np.asarray(dC.get("dT_air_C", np.zeros_like(spd)), float)
    u10_night = u10_day * night_factor(land, dT_air)
    dC["z0"] = z0; dC["u10_factor"] = f10
    dC["u10_day_ms"] = u10_day; dC["u10_night_ms"] = u10_night
    out = dict(
        lev_mean_ms=float(spd.mean()),
        u10_day_mean_ms=float(u10_day.mean()),
        u10_night_mean_ms=float(u10_night.mean()),
    )
    if land.any():
        out.update(
            land_lev_mean_ms=float(spd[land].mean()),
            land_u10_day_mean_ms=float(u10_day[land].mean()),
            land_u10_night_mean_ms=float(u10_night[land].mean()),
        )
    return out


def _upwind_ray_max(height, u, v, x_m, e_m, L_km, n_steps, mode):
    """Shared Winstral-style upwind-ray scan behind ``lee_regime`` (both its rays): for each cell,
    sample the terrain along the LOCAL UPWIND ray out to L_km (log-spaced distances resolve the near
    ridge AND the far range without N^2 cost) and accumulate the max over distance of either the
    horizon SLOPE ``(h_up-h)/d`` (mode='slope'; kept for probes, no longer read by the law -- ROADMAP
    DOC-34 retired the slope-keyed shelter) or the upwind RELIEF ``h_up-h`` (mode='relief').
    Cross-track (x) is clamped at the walls, along-track (e) is periodic. Returns the (nx, ne)
    accumulator, NOT floored at 0 (callers floor). Keeping the two scans on ONE code path locks them
    together (the box-saturation caveat for distance-like metrics applies to both identically)."""
    from scipy.ndimage import map_coordinates
    h = np.asarray(height, float)
    nx, ne = h.shape
    dx = float(np.mean(np.diff(np.asarray(x_m, float))))
    de = float(np.mean(np.diff(np.asarray(e_m, float))))
    sp = np.hypot(u, v)
    with np.errstate(invalid="ignore", divide="ignore"):
        ux = np.where(sp > 1e-9, -np.asarray(u, float) / sp, 0.0)   # upwind unit vector
        ue = np.where(sp > 1e-9, -np.asarray(v, float) / sp, 0.0)
    ix, ie = np.meshgrid(np.arange(nx, dtype=float), np.arange(ne, dtype=float), indexing="ij")
    acc = np.zeros_like(h)
    dists = np.geomspace(max(dx, de), L_km * 1e3, n_steps)
    for d in dists:
        ci = np.clip(ix + ux * (d / dx), 0, nx - 1)                  # cross-track: clamped at walls
        ce = np.mod(ie + ue * (d / de), ne)                          # along-track: periodic
        h_up = map_coordinates(h, [ci, ce], order=1, mode="nearest")
        acc = np.maximum(acc, (h_up - h) / d if mode == "slope" else (h_up - h))
    return acc


# ======================================================================================
# issue #114 / ROADMAP DOC-34: the TERRAIN factors -- shelter and lee amplification as the two
# branches of ONE regime law.
#
# Stratified flow meeting a ridge does one of three things, and the nondimensional mountain height
# M = N*H/U (N the Brunt-Vaisala frequency, H the upwind ridge relief, U the incident wind -- the
# inverse of the mountain Froude number Fr = U/(N*H)) says which:
#   - M << 1 (fast flow / low ridge): the flow rides over linearly; the lee is neither sheltered
#                       nor accelerated -> factor ~ 1.
#   - M ~ O(1)        : wave overturning / hydraulic-jump regime -> a downslope WINDSTORM in the lee
#                       (factor up to the literature ceiling ~2-3x).
#   - M >> 1 (blocked): the low-level flow stagnates upstream and goes around; the lee is a
#                       stagnant WAKE -> the lee is SHELTERED (factor < 1).
# These are alternative fates of one lee, not independent effects, which is why they are computed
# from ONE M and composed as one factor.  The earlier form keyed shelter on Winstral's upwind
# horizon SLOPE with a grid-tied halving constant (sx_half = 0.008 at >100 km cells) while the
# amplification keyed on RELIEF: measured on the shipped map (map_webapp/_felt_wind_ab.py) the
# slope form was scale-free at no resolution, and the two terms fired on the SAME cells (96.9 % of
# the land the shelter painted below 0.7 also carried amplification above 1.3), so their product read
# as an acceleration on the most sheltered ground.  Keyed on M, the branches are SEQUENCED in M and
# cannot both act strongly on one cell.
#
# THE FOOTPRINTS, AND WHY THERE ARE TWO OF THEM.  A lee effect does not reach forever downwind, so
# each branch is applied with the weight w = H_near / H: the fraction of the obstacle's relief that
# stands within that branch's reach upwind of the cell.  A cell at the foot of a steep ridge sees the
# whole ridge inside the footprint (w = 1); a cell on a plain beyond it sees only the tail of the
# slope (w -> 0); a long gentle slope is reached in proportion to how much of it lies within reach.
# The two branches get DIFFERENT reaches, because they are different objects: a wave-breaking
# WINDSTORM is pinned to the lee slope that drives it (strongest within ~10-20 km of the base, gone
# within a few tens of km -> LEE_FOOTPRINT_KM = 50), while a blocked WAKE is a momentum deficit that
# persists until turbulence refills it and on Earth runs many obstacle widths -- hundreds of km
# behind blocked islands, and this world has no rotation to close it (WAKE_FOOTPRINT_KM = 150; see
# the constant for the full argument).  An earlier revision of this block asserted the two were "of
# the same order", which is wrong on Earth's own evidence and wrong in the direction that flattered
# the result (PR #731 review, finding 2).
# The REGIME (M) is read off the full L_km ray for both, because the obstacle's HEIGHT is what sets
# it and the long ray is what resolves the obstacle (measured grid-invariant across a 49->6 km
# sweep); only the effects' REACH is short, and all three quantities come from ONE quadrature
# (`_upwind_relief_nested`) so each w is a true fraction. This replaces the old gate
# `relief > 0 over 300 km`, under which a cell on a plain 250 km downwind of a ridge was "in its lee"
# and 20 % of the map's land carried a >= 1.3x multiplier on a day-mean wind.
#
# THE THIRD BRANCH: ENCLOSURE, AND WHY THE RAY LAW NEEDS ONE.  M above reads only the TALLEST
# obstacle UPWIND, so it says nothing about where the air goes next -- and a cell at the foot of a
# ridge with open ground beyond it (a lee slope: foehn, correctly accelerated) is indistinguishable
# from a cell at the foot of a ridge with a second ridge just beyond it (a basin floor: stagnant,
# the opposite).  The regime law gave both the windstorm gain.  The missing quantity is the ESCAPE
# barrier: the height the air at this cell must climb to LEAVE.  It is the same dividing-streamline
# criterion, in Sheppard's (1956) original and literal sense -- a surface parcel of speed U carries
# 0.5*U^2 and must spend 0.5*N^2*D^2 to rise D against stratification N, so it surmounts a barrier
# of height D only if M_enc = N*D/U < 1, and above that the layer below D*(1 - 1/M_enc) is stagnant.
# D is measured TWO-SIDED on the wind axis -- `enclosure_depth` = min(upwind rim, downwind rim)
# within ENCLOSURE_RADIUS_KM -- which is what makes the branch tell the two cases apart: on a lee
# slope the ground falls away downwind, the downwind rim is 0, D is 0 and the branch is EXACTLY 1,
# while a basin floor (or a valley lying across the wind) is walled both ways and D is the lower of
# its two walls.  The result is blended, not multiplied: g = g_ray + b*(ENCLOSURE_FLOOR - g_ray)
# with b = clip(1 - M_ENC_ONSET/M_enc) Sheppard's stagnant fraction, so at b = 1 the ray law's
# answer is OVERRIDDEN by the pool's residual rather than scaled by it (a decoupled floor's wind is
# set by its own drainage circulation, not by the ambient wind times an upwind ridge), and the
# result is bounded between the two regimes' own values -- no double count is possible, and the
# composed factor cannot fall below ENCLOSURE_FLOOR.  It is served as a factor all the same
# (`enclosure` = g/g_ray, exactly 1 where the branch does nothing), so the three branches still
# multiply to `factor`.
#
# WHAT THE TWO-SIDED WIND-AXIS D IS, AND WHAT IT IS NOT.  It is an UPPER bound on enclosure: it asks
# only about the flow direction, so it counts a basin that is open to the SIDE as enclosed, and on
# an f = 0 world lateral escape is easy.  The matching LOWER bound is the all-azimuth barrier (the
# lowest pass in any direction, i.e. a radius-limited topographic pit), which on this archive is
# ~70x rarer (`_felt_wind_ab.py` report 7 quotes both).  The wind-axis form is the one implemented
# because ventilation is done BY THE WIND, and because it is the form that is exactly 1 on a lee
# slope by construction.  A pure DOWNWIND-rim barrier would additionally claim upstream blocking
# (the stagnation zone on a ridge's windward side, also real Sheppard physics) -- that is a
# different term and is deliberately not folded in here.
#
# All constants LITERATURE-ANCHORED, not derived ([model-soft], like the roughness classes): N a
# representative tropospheric ~0.011 s^-1; the windstorm ceiling amp_max ~2.5 and the critical band;
# M_BLOCK ~2, where the blocked branch takes hold (Sheppard's dividing streamline puts a blocked
# layer below H(1 - 1/M) for M > 1; flow splitting and upstream stagnation set in at M ~ 1.5-2 in the
# numerical regime diagrams); the wake floor ~0.2 (a lee wake keeps a weak recirculating flow, not
# a dead calm); the two footprints above.  Confidence: the EXISTENCE and SIGN of the two regimes and
# their ordering in M [firm]; the per-cell VALUES and which valleys are alleys [model-soft,
# terrain-resolution-limited].
#
# ⚠️ THE RESULT IS LINEAR IN N, AND N IS AN EARTH ANCHOR.  M = N*H/U scales with N while the branch
# thresholds do not, so N sets how much of a world is in each regime. Swept on the shipped map
# (12 km) with everything else held: land at M > 2 runs 0.18 / 2.46 / 5.44 / 8.27 % at N = 0.007 /
# 0.011 / 0.016 / 0.022 s^-1. So "blocking is rare on this world" is a statement about N = 0.011, not
# about the world. Worse, the constant is labelled FREE-TROPOSPHERIC while the physics wants the
# stability of the low-level layer of depth ~H that actually meets the ridge, and this band has no
# reason to carry one value of it (the felt-wind ladder's own night-decoupling ramp asserts a
# strongly stable near-surface layer over the cold-edge land where people live; the hot inner sea at
# ~4,000 J/kg of CAPE is nearly the opposite). Deriving N(theta) from the model is ROADMAP DOC-46,
# and it is BLOCKED on ROADMAP PHY-17 (2026-09-14): the canonical AD core's own state is statically
# UNSTABLE through its lowest 1-2 km equatorward of ~50 deg (its convective adjustment cannot remove
# that within the production sweep cap), so a bulk N read off it is zero over every ridge depth up to 3 km in the map band, and the
# checkpoint holds one noon state and no night profile. Its reference-profile N is 0.0103 s^-1, which is
# the anchor to within 7 %; what is unknown is the LOW-LEVEL departure from it
# (`cosmology_models/phy17_bl_stratification_probe.py`). Until DOC-46 lands, quote the sweep with any
# number taken off these layers.
# ======================================================================================

N_TROP = 0.011           # representative free-tropospheric Brunt-Vaisala frequency [1/s] (model-soft)
AMP_MAX = 2.5            # downslope-windstorm amplification ceiling (lit.: lee winds 2-3x the mean)
M_BLOCK = 2.0            # nondimensional height at which the BLOCKED (sheltered-wake) branch takes hold
SHELTER_FLOOR = 0.2      # the wake's residual wind fraction as M -> inf (lit.: ~0.1-0.3 of free stream)
RAY_SAMPLES_PER_WINDOW = 4  # log-spaced samples per footprint window on the upwind ray.  Windowed
                            # rather than one geomspace over the whole ray so a fixed-km footprint is
                            # resolved identically at every grid -- see `ray_distances`.
LEE_FOOTPRINT_KM = 50.0   # reach of a downslope WINDSTORM beyond the ridge foot [km]: strongest
                          # within ~10-20 km of the base and gone within a few tens of km.
WAKE_FOOTPRINT_KM = 150.0 # reach of a BLOCKED WAKE beyond the ridge foot [km] -- deliberately LONGER
                          # than the windstorm's, and the asymmetry is physics, not bookkeeping.  A
                          # wave-breaking windstorm is pinned to the lee slope that drives it; a
                          # blocked wake is a momentum deficit that persists until turbulence
                          # refills it, and Earth's own blocked-island wakes run many obstacle
                          # widths -- hundreds to thousands of km behind Hawaii, Madeira, the
                          # Canaries -- *with* rotation available to close them.  This world has
                          # f = 0 exactly (cosmology_models/CLAUDE.md, "Standing physics/geometry
                          # facts"): there is no Rossby radius to bound the
                          # upstream blocking laterally or to shut the wake downstream, so if
                          # anything the wake should reach FARTHER here, and 150 km (3x the
                          # windstorm's, half the ray) is the conservative end of that argument
                          # rather than an estimate of it.  [model-soft; the layer's most
                          # consequential single constant -- see the sweep in _felt_wind_ab.py]
ENCLOSURE_RADIUS_KM = 50.0  # how far out the ESCAPE barrier is looked for [km] -- the third branch's
                          # one length scale.  Set equal to LEE_FOOTPRINT_KM on purpose: the branch
                          # exists to override the WINDSTORM gain on ground that is enclosed rather
                          # than merely downwind of a ridge, and the windstorm gain reaches 50 km, so
                          # the barrier is read over exactly the neighbourhood the gain is claimed
                          # over.  A barrier 200 km downwind does not trap the air HERE -- the flow
                          # crosses 200 km of open ground first and stagnates there instead.
                          # [model-soft; swept 25/50/100/150 km in _felt_wind_ab.py report 7]
M_ENC_ONSET = 1.0         # nondimensional escape height N*D/U at which the floor starts to decouple.
                          # Sheppard's (1956) OWN onset, not the raised M_BLOCK = 2.0: a surface
                          # parcel carries 0.5*U^2 and must spend 0.5*N^2*D^2 to climb a barrier of
                          # height D under stratification N, so it escapes iff N*D/U < 1.  M_BLOCK
                          # was raised above Sheppard's value only to keep the blocked branch clear
                          # of the windstorm bump, and that reason does not apply here: this branch
                          # is geometrically disjoint from the bump (D is identically 0 on a lee
                          # slope), so the literature onset can be used as it stands.
ENCLOSURE_FLOOR = 0.2     # the decoupled pool's residual wind fraction (lit.: a stagnant cold-air
                          # pool keeps a weak drainage/thermal circulation, ~0.1-0.3 of the flow
                          # aloft -- the same band SHELTER_FLOOR is anchored in, and deliberately
                          # the same value: a decoupled pool is if anything CALMER than a lee wake,
                          # so 0.2 is the conservative, less-suppressing end of the range.
CALM_MS = 10.0            # the 10-m day wind below which ground counts as calm enough to settle --
                          # `calm_habitable`'s threshold, in one place because two entry points read
                          # it (see `_calm_habitable_mask`).
CALM_T_LO_C, CALM_T_HI_C = 0.0, 30.0   # the livability window `calm_habitable` also requires


def mountain_nondim_height(relief_m, U, N=N_TROP, U_floor=2.0):
    """Nondimensional mountain height M = N*H/U (the inverse mountain Froude number) -- the single
    parameter that sets the lee-flow regime. ``relief_m`` the upwind ridge relief H [m], ``U`` the
    incident wind speed [m/s] (floored at ``U_floor`` so calm cells don't blow M up), ``N`` the
    stratification [1/s]. M~O(1) is the downslope-windstorm band; M > M_BLOCK the blocked band."""
    U = np.maximum(np.asarray(U, float), float(U_floor))
    return float(N) * np.asarray(relief_m, float) / U


def amplification_from_M(M, amp_max=AMP_MAX, M_crit=1.0, ln_width=0.9):
    """Lee-wind amplification factor (>= 1) as a function of the nondimensional mountain height M --
    the WINDSTORM branch of the regime law.

    A log-Gaussian bump centred on M_crit: factor = 1 + (amp_max-1)*exp(-(ln(M/M_crit)/ln_width)^2).
    -> amp_max at the critical height, decaying to 1 for both free-flow (M->0) and blocked (M->inf)
    flow. Symmetric in log M (the two non-amplifying limits are the small/large ends). The form is a
    smooth stand-in for the windstorm-onset band, not a derived response curve [model-soft]."""
    M = np.asarray(M, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(np.maximum(M, 1e-12) / float(M_crit))
    bump = np.exp(-(lr / float(ln_width)) ** 2)
    return 1.0 + (float(amp_max) - 1.0) * bump


def shelter_from_M(M, M_block=M_BLOCK, floor=SHELTER_FLOOR):
    """Lee-wake shelter factor s(M) in (floor, 1] -- the BLOCKED branch of the regime law.

    Sheppard's dividing-streamline criterion: for M above the onset the layer below H*(1 - M_block/M)
    of the ridge is blocked and the lee beneath it is a stagnant wake, so the 10-m wind is reduced in
    proportion to that blocked fraction, down to the wake's residual ``floor``:

        s = 1 - (1 - floor) * clip(1 - M_block/M, 0, 1)

    Exactly 1 for M <= M_block, so the branch contributes NOTHING at the windstorm peak (M_crit = 1 <
    M_block) -- so the composed law cannot double-count.  That property is BY CONSTRUCTION rather
    than derived: it follows from M_block sitting far enough above M_crit for the bump to have run
    out, which is a choice of the pair (M_block, ln_width), not a fact about the flow.  At
    M_block = 1.5 the composed factor at M = 2.5 would go x1.29 -> x1.02.  The onset is the one new
    constant DOC-34 added (Sheppard's own onset is M = 1; the regime diagrams put flow splitting at
    M ~ 1.5-2, and the windstorm band must be allowed to run out first) [model-soft]."""
    M = np.asarray(M, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        blocked = np.clip(1.0 - float(M_block) / np.maximum(M, 1e-12), 0.0, 1.0)
    return 1.0 - (1.0 - float(floor)) * blocked


def stagnant_fraction(M_enc, onset=M_ENC_ONSET):
    """Sheppard's stagnant fraction b in [0, 1] for the ESCAPE barrier -- the ENCLOSURE branch's
    only nonlinearity.

    ``M_enc`` = N*D/U is the nondimensional height of the barrier the air at this cell must climb to
    LEAVE (``enclosure_depth``), on the same N and U as the ray law's M so the two live on one axis.
    A surface parcel carries 0.5*U^2 and spends 0.5*N^2*z^2 rising z against stratification N, so it
    reaches only z = U/N above itself: for a barrier of height D the layer below D*(1 - 1/M_enc) can
    never surmount it and is a stagnant pool.  That fraction is b:

        b = clip(1 - onset/M_enc, 0, 1)          -- 0 for M_enc <= onset, -> 1 as M_enc -> inf

    Identical in form to ``shelter_from_M``'s blocked fraction, which is the same criterion applied
    to the obstacle a cell sits BEHIND rather than the one it sits INSIDE.  The floor of the pool is
    always inside the stagnant layer once M_enc > 1, so the sharp criterion is a STEP at M_enc = 1
    and this ramp is a deliberate softening of it: it under-suppresses near onset, which is the
    conservative direction, and it is the right shape given that D is read off a gridded surface
    [form firm (Sheppard), the softening model-soft]."""
    M_enc = np.asarray(M_enc, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.clip(1.0 - float(onset) / np.maximum(M_enc, 1e-12), 0.0, 1.0)


def enclosure_depth(height, u, v, x_m, e_m, radius_km=ENCLOSURE_RADIUS_KM,
                    n_per=RAY_SAMPLES_PER_WINDOW):
    """The TWO-SIDED escape barrier D [m] on the wind axis -> (nx, ne), floored at 0.

    ``D = min(upwind rim, downwind rim)`` within ``radius_km``, each rim the maximum relief above
    the cell along that ray (the same windowed quadrature the ray law runs, ``_upwind_relief_nested``,
    with the wind reversed for the downwind half).  So:

      - open ground, and a RESOLVED LEE SLOPE: the ground falls away downwind, the downwind rim is
        0, and D is EXACTLY 0 -- the branch is exactly 1 there, by construction, not by tuning;
      - an enclosed basin floor, or a valley lying ACROSS the wind: walled both ways, and D is the
        LOWER of the two walls (the easier of the two ways out);
      - a valley lying ALONG the wind: open both ways, D = 0, correctly ventilated.

    ``radius_km`` is the branch's one length scale (see ``ENCLOSURE_RADIUS_KM``).  The result does
    NOT depend on the ray length it is cut from: ``ray_distances`` windows on the cutoffs, so the
    samples inside the first window are the same whether the ray stops at ``radius_km`` or runs on
    to 300 km -- which is what makes this distance-like metric box-independent by construction
    (pinned by ``test_the_enclosure_depth_does_not_depend_on_the_ray_it_is_cut_from``).  Returns
    zeros for ``radius_km <= 0``, which is the branch's OFF path.

    ``height`` (nx, ne) [m] on the Stage C grid; ``u``/``v`` give the axis; ``x_m``/``e_m`` the grid
    coordinate vectors [m] (x clamped at the walls, e periodic, as everywhere on this ray)."""
    h = np.asarray(height, float)
    if float(radius_km) <= 0.0:
        return np.zeros_like(h)
    u = np.asarray(u, float); v = np.asarray(v, float)
    up, _ = _upwind_relief_nested(h, u, v, x_m, e_m, radius_km, (radius_km,), n_per=n_per)
    down, _ = _upwind_relief_nested(h, -u, -v, x_m, e_m, radius_km, (radius_km,), n_per=n_per)
    return np.minimum(up, down)


def terrain_factor_from_M(M, amp_max=AMP_MAX, M_crit=1.0, ln_width=0.9, M_block=M_BLOCK,
                          floor=SHELTER_FLOOR):
    """The unified regime law g(M) = amplification_from_M(M) * shelter_from_M(M): the factor a ridge
    of nondimensional height M imposes on the 10-m wind in its lee (before the footprint weight).
    -> 1 at M << 1; the windstorm bump (amp_max at M_crit); through 1 again near M ~ 3 where the
    decaying bump meets the blocked branch; down toward ``floor`` as M -> inf.  One hump, then one
    trough -- the two branches are alternatives sequenced in M, never a product of two strong terms
    on one M (``shelter_from_M`` is exactly 1 up to M_block)."""
    return (amplification_from_M(M, amp_max=amp_max, M_crit=M_crit, ln_width=ln_width)
            * shelter_from_M(M, M_block=M_block, floor=floor))


def orographic_amplification(relief_m, U, N=N_TROP, amp_cap=AMP_MAX):
    """SUB-GRID lee-windstorm amplification factor (parallels ``orographic_z0``): the capped
    amplification that relief BELOW the model grid imposes on the lee 10-m wind. ``relief_m`` the
    sub-grid ridge-to-valley relief [m], ``U`` the incident wind [m/s], ``amp_cap`` the documented
    ceiling. Use alongside ``orographic_z0`` for the sub-grid wind correction -- z0 lowers the AREA
    mean, this raises the LEE-valley extreme."""
    M = mountain_nondim_height(relief_m, U, N=N)
    return amplification_from_M(M, amp_max=amp_cap)


H_EPS_M = 1e-6           # relief below this is no obstacle at all: w is 0 rather than 0/0


def lee_regime(height, u, v, x_m, e_m, N=N_TROP, L_km=300.0, footprint_km=LEE_FOOTPRINT_KM,
               wake_footprint_km=WAKE_FOOTPRINT_KM, n_per=RAY_SAMPLES_PER_WINDOW, U_floor=2.0,
               froude=None):
    """The ONE regime computation both terrain branches read: -> (M, w_lee, w_wake, relief).

    ``relief`` H [m] is the maximum upwind relief along the LOCAL UPWIND ray out to ``L_km`` (the
    height of the tallest obstacle the cell sits in the lee of; floored at 0 -- terrain below you is
    not an obstacle).  ``M`` = N*H/U with U the local wind speed |V| (floored at ``U_floor``), or
    1/``froude`` when a per-cell hydraulic Froude is supplied (issue #111's ``dC['froude']`` -- a
    DISTINCT Froude number used as a co-critical regime flag, see ``terrain_amplification``).

    ``w_lee`` and ``w_wake`` in [0, 1] are the two FOOTPRINT weights: the fraction of that obstacle's
    relief standing within ``footprint_km`` (the windstorm's reach) and within ``wake_footprint_km``
    (the blocked wake's, which is longer -- see the constants). Both come from the SAME quadrature as
    ``relief`` (``_upwind_relief_nested``), so each is a true fraction, ``w <= 1`` by construction
    rather than by clipping. Both are 0 where there is no obstacle.

    ``n_per`` is that quadrature's resolution: log-spaced samples PER WINDOW (``ray_distances``),
    not a total along the ray. Raising it refines every footprint equally instead of redistributing
    samples between them -- which is the whole reason the ray is windowed, and why the total-count
    knob this argument replaced could not be the one exposed.

    ``U`` is the local wind SPEED |V|, not the ridge-normal component, while ``M_crit = 1`` is a
    cross-ridge criterion. The ray geometry recovers most of the orientation dependence for free (a
    wind-ALIGNED valley finds no upwind relief and gets M = 0), so what is left is the cosine on an
    obliquely-crossed ridge, which at display grade is inside the constants' own softness.
    ``U_floor`` bounds the other end: a becalmed cell would otherwise take a large M and so the
    STRONGEST shelter, a diagnostic that makes calm ground calmer. At this world's ~10 m/s band winds
    the floor is far from binding; it matters only if this is ever run on a near-calm field.

    ``height`` (nx, ne) [m] on the Stage C grid; ``u``/``v`` the wind components (direction, and
    the fallback regime speed); ``x_m``/``e_m`` the grid coordinate vectors [m] (x clamped at the
    walls, e periodic)."""
    relief, (near_lee, near_wake) = _upwind_relief_nested(
        height, u, v, x_m, e_m, L_km, (footprint_km, wake_footprint_km), n_per=n_per)
    obst = relief > H_EPS_M
    den = np.where(obst, relief, 1.0)
    w_lee = np.where(obst, near_lee / den, 0.0)
    w_wake = np.where(obst, near_wake / den, 0.0)
    if froude is not None:
        M = 1.0 / np.maximum(np.asarray(froude, float), 1e-6)
    else:
        M = mountain_nondim_height(relief, np.hypot(u, v), N=N, U_floor=U_floor)
    return M, w_lee, w_wake, relief


def terrain_factors(height, u, v, x_m, e_m, N=N_TROP, L_km=300.0, footprint_km=LEE_FOOTPRINT_KM,
                    wake_footprint_km=WAKE_FOOTPRINT_KM, n_per=RAY_SAMPLES_PER_WINDOW,
                    amp_max=AMP_MAX, M_crit=1.0, ln_width=0.9, M_block=M_BLOCK,
                    floor=SHELTER_FLOOR, U_floor=2.0, froude=None,
                    enclosure_radius_km=ENCLOSURE_RADIUS_KM, M_enc_onset=M_ENC_ONSET,
                    enclosure_floor=ENCLOSURE_FLOOR):
    """All THREE terrain branches -> dict(shelter, amplification, enclosure, factor, M, w_lee,
    w_wake, relief, D_enc, M_enc), each (nx, ne) on the Stage C grid ([model-soft] texture
    diagnostics).  The first two come from one ``lee_regime`` call on the UPWIND ray; the third
    from the two-sided ESCAPE barrier (``enclosure_depth``) about the cell.

        shelter       = 1 - w_wake * (1 - shelter_from_M(M))         in (floor, 1]
        amplification = 1 + w_lee  * (amplification_from_M(M) - 1)   in [1, amp_max]
        g_ray         = shelter * amplification                       the upwind-ray law
        factor        = g_ray + b * (enclosure_floor - g_ray)         blended toward the pool
        enclosure     = factor / g_ray                                in (0, 1], exactly 1 at b = 0

    with ``b = stagnant_fraction(M_enc)`` and ``M_enc = N*D_enc/U`` on the same N and U the
    topo-estimated M uses.  That shared axis holds only when ``froude`` is None: on the #111 path
    M is 1/Fr, a different definition, and M_enc stays N*D/U (the hydraulic Froude says nothing
    about the barrier downwind of a cell).  The separation of the enclosure branch from the
    windstorm bump does not rest on the axis either way -- it rests on GEOMETRY: D is identically
    0 on a lee slope, whatever M is.

    ``factor`` IS the product of the three served branches, so what a reader composes from the
    layers is exactly the law -- and the product is legitimate because the branches are alternatives
    rather than independent effects: the first two are sequenced in M (``shelter`` < 1 only above
    M_block, where the windstorm bump has largely run out), and the third OVERRIDES both, blending
    toward the pool's own residual rather than scaling their answer, so ``factor`` always lies
    between ``g_ray`` and ``enclosure_floor`` and can never fall below the latter.  Multiply
    ``factor`` into the 10-m DAY wind for the terrain-textured felt wind.

    ``enclosure_radius_km = 0`` switches the third branch OFF and returns the two-branch law
    byte-identically (``enclosure`` all ones, ``factor`` exactly ``shelter * amplification``); at
    the default radius the OTHER two branches are untouched either way, because the enclosure runs
    its own short scans and never adds a cutoff to the long ray's quadrature.

    ⚠️ WHAT ``factor`` IS NOT: a frequency-weighted wind. The windstorm branch is the same KIND of
    object as ``convective_gust_speed``'s CAPE -- a regime that fires when conditions are met -- and
    that function's docstring refuses to turn a potential into a mean-day number without a frequency
    weight. The defence for composing this one into a DAY-MEAN wind is specific to this world, and it
    is a real consequence of the setting rather than a convenience: there are no seasons and no
    synoptic variability here (f = 0, one steady overturning cell, no travelling weather systems), so
    the incident wind on a ridge is very nearly the same wind every day. A lee that is in the
    windstorm regime is in it CONTINUOUSLY, which is exactly what makes a mean-day multiplier
    defensible here where it would not be on Earth -- and why convection, which is intermittent on
    any world, is still refused. The residual intermittency is the diurnal one the ladder already
    carries (the breeze phase and the nocturnal decoupling), not a synoptic one."""
    M, w_lee, w_wake, relief = lee_regime(
        height, u, v, x_m, e_m, N=N, L_km=L_km, footprint_km=footprint_km,
        wake_footprint_km=wake_footprint_km, n_per=n_per, U_floor=U_floor, froude=froude)
    s = 1.0 - w_wake * (1.0 - shelter_from_M(M, M_block=M_block, floor=floor))
    a = 1.0 + w_lee * (amplification_from_M(M, amp_max=amp_max, M_crit=M_crit, ln_width=ln_width) - 1.0)
    g_ray = s * a
    # the ENCLOSURE branch: the two-sided escape barrier, on the same N and U as M.  `froude` is
    # NOT consumed here -- the #111 hydraulic Froude is a co-critical flag for the lee-wave regime
    # over a ridge, and says nothing about the barrier downwind of this cell.
    D_enc = enclosure_depth(height, u, v, x_m, e_m, radius_km=enclosure_radius_km, n_per=n_per)
    M_enc = mountain_nondim_height(D_enc, np.hypot(u, v), N=N, U_floor=U_floor)
    b = stagnant_fraction(M_enc, onset=M_enc_onset)
    g = g_ray + b * (float(enclosure_floor) - g_ray)
    enc = np.where(g_ray > 0.0, g / np.where(g_ray > 0.0, g_ray, 1.0), 1.0)
    return dict(shelter=s, amplification=a, enclosure=enc, factor=g, M=M, w_lee=w_lee,
                w_wake=w_wake, relief=relief, D_enc=D_enc, M_enc=M_enc)


def ray_distances(d0_m, L_m, cuts_m, n_per=RAY_SAMPLES_PER_WINDOW):
    """Sample distances for the upwind ray: every cutoff is itself a sample point, and each WINDOW
    between consecutive cutoffs gets ``n_per`` log-spaced samples -> a sorted 1-D array [m].

    This is what makes a fixed-distance footprint resolution-STABLE.  A single
    ``geomspace(d0, L, n)`` has sample positions that scale with the CELL, so a fixed-km footprint is
    resolved differently at every grid: on a 300 km ray of 8 samples the outermost sample inside
    50 km is 48.8 / 50.0 / 48.1 / **32.4** km at 48.8 / 24.4 / 12.2 / 6.1 km cells -- so the finest
    grid never samples the outer 35 % of the windstorm footprint at all, under-reads ``near_lee``,
    and the served lee-gain fraction FALLS at the finest rung (8.11 % -> 6.14 %) where the form it
    replaced rose monotonically.  The finer the grid, the worse a fixed-distance window was resolved,
    which is the wrong sign for a convergence claim (PR #731 re-review, finding 1).  Windowing fixes
    the sample COUNT per window instead, so every footprint is resolved the same way at every
    resolution, while nesting -- and with it ``near <= relief`` by construction -- is preserved.

    The edges are carried explicitly rather than left to ``geomspace`` to supply, because at
    ``n_per`` = 1 it returns only the LEFT endpoint -- which would silently drop ``L_m`` itself and
    shorten the ray.  At the production ``n_per`` >= 2 they are already there and the union is
    unchanged; this only makes the promise above true at every ``n_per``."""
    d0 = float(d0_m)
    edges = ([d0] + [float(c) for c in sorted(set(cuts_m)) if d0 < float(c) < float(L_m)]
             + [float(L_m)])
    if edges[-1] <= edges[0]:
        return np.array([d0])
    return np.unique(np.concatenate([np.asarray(edges, float)] +
                                    [np.geomspace(a, b, int(n_per))
                                     for a, b in zip(edges[:-1], edges[1:])]))


def _upwind_relief_nested(height, u, v, x_m, e_m, L_km, cutoffs_km,
                          n_per=RAY_SAMPLES_PER_WINDOW):
    """The upwind RELIEF at the full ray length AND at each of ``cutoffs_km``, from ONE scan and ONE
    windowed quadrature -> (relief, [near_k, ...]).

    Why one scan rather than two calls to ``_upwind_ray_max``: the log-spaced distances depend on the
    ray LENGTH, so a separate short scan is not a sub-sample of the long one -- at a 12 km cell the
    300 km ray puts 4 of its 8 samples inside 50 km where a 50 km ray puts all 8.  Two quadratures
    made ``near > relief`` on 16 % of the shipped map's land, which broke the "fraction of the
    obstacle within reach" reading of ``w`` (it clipped to 1) and biased the regime's own H LOW,
    toward the windstorm branch and away from blocking (PR #731 review, finding 5).  Nested maxima
    over ONE sample set make ``near <= relief`` true by construction.

    Why those samples are WINDOWED (``ray_distances``) rather than one geomspace: see there -- a
    fixed sample count per window is what keeps a fixed-km footprint resolved the same way at every
    grid, which the first version of this fix did not.

    Cutoffs are floored at one grid cell, so on a grid coarser than a footprint the near field is
    the adjacent upwind cell -- the coarsest honest answer."""
    from scipy.ndimage import map_coordinates
    h = np.asarray(height, float)
    nx, ne = h.shape
    dx = float(np.mean(np.diff(np.asarray(x_m, float))))
    de = float(np.mean(np.diff(np.asarray(e_m, float))))
    sp = np.hypot(u, v)
    with np.errstate(invalid="ignore", divide="ignore"):
        ux = np.where(sp > 1e-9, -np.asarray(u, float) / sp, 0.0)
        ue = np.where(sp > 1e-9, -np.asarray(v, float) / sp, 0.0)
    ix, ie = np.meshgrid(np.arange(nx, dtype=float), np.arange(ne, dtype=float), indexing="ij")
    d0 = max(dx, de)
    relief = np.zeros_like(h)
    cuts = [max(float(c) * 1e3, d0) for c in cutoffs_km]
    near = [np.zeros_like(h) for _ in cuts]
    for d in ray_distances(d0, L_km * 1e3, cuts, n_per=n_per):
        ci = np.clip(ix + ux * (d / dx), 0, nx - 1)
        ce = np.mod(ie + ue * (d / de), ne)
        rel = map_coordinates(h, [ci, ce], order=1, mode="nearest") - h
        relief = np.maximum(relief, rel)
        for k, c in enumerate(cuts):
            if d <= c + 1e-9:
                near[k] = np.maximum(near[k], rel)
    relief = np.maximum(relief, 0.0)
    return relief, [np.maximum(n, 0.0) for n in near]


def shelter_factor(height, u, v, x_m, e_m, L_km=300.0, n_per=RAY_SAMPLES_PER_WINDOW, N=N_TROP,
                   wake_footprint_km=WAKE_FOOTPRINT_KM, M_block=M_BLOCK, floor=SHELTER_FLOOR,
                   froude=None):
    """Topographic wind-shelter factor s(x,e) in (floor, 1] -- Jake's 'sheltered valleys = the
    habitable microgeography' call made quantitative ([model-soft] texture diagnostic): the BLOCKED
    branch of the regime law, footprint-weighted (see ``terrain_factors``).

    A cross-wind ridge tall enough to block the flow (M = N*H/U > M_block) shelters the ground in
    its lee out to ``wake_footprint_km``; a valley ALIGNED with the wind correctly gets s ~ 1 (no
    upwind relief); a ridge the flow rides over (M ~ 1) is the windstorm case and shelters nothing
    -- see ``terrain_amplification`` for that branch.  Winstral's horizon-slope form and its
    grid-tied ``sx_half`` are retired (ROADMAP DOC-34).

    NB on THIS world the blocked regime is uncommon and the resolved shelter correspondingly thin
    (see the module block): at ~10 m/s band winds M > M_block needs ~1.8 km of upwind relief, so the
    valley-scale shelter Jake's "sheltered valleys" asks about is mostly SUB-GRID here, and this
    layer is the resolved, regional part of it.  A valley lying ACROSS the wind is reached by the
    ENCLOSURE branch instead (``enclosure_depth``), which needs only ~900 m of two-sided barrier at
    these winds rather than ~1.8 km of upwind ridge, and which is what tells that valley from a lee
    slope.  ``orographic_z0`` remains the third companion and is a different object again -- a DRAG
    (a uniform reduction across a cell), not a sheltered lee."""
    return terrain_factors(height, u, v, x_m, e_m, N=N, L_km=L_km,
                           wake_footprint_km=wake_footprint_km, n_per=n_per, M_block=M_block,
                           floor=floor, froude=froude,
                           enclosure_radius_km=0.0)["shelter"]   # branch not read: skip its scans


def _terrain_key(N, L_km, footprint_km, wake_footprint_km, amp_max, M_crit, ln_width, U_floor,
                 froude, enclosure_radius_km=ENCLOSURE_RADIUS_KM, M_enc_onset=M_ENC_ONSET,
                 enclosure_floor=ENCLOSURE_FLOOR):
    """The identity of a ``terrain_factors`` result: every argument that changes a number, plus
    whether a per-cell Froude was supplied.  ``attach_amplification`` reuses ``attach_shelter``'s
    scan only when its own arguments give this same key.  The enclosure branch's three constants
    are in it for the reason the other eight are: unkeyed, the reuse would hand back a result
    computed under someone else's constants (PR #731 re-review, finding 2)."""
    return (float(N), float(L_km), float(footprint_km), float(wake_footprint_km), float(amp_max),
            float(M_crit), float(ln_width), float(U_floor), froude is not None,
            float(enclosure_radius_km), float(M_enc_onset), float(enclosure_floor))


def _calm_habitable_mask(land, T_map_C, u10_day_ms, calm_ms=CALM_MS):
    """The ``calm_habitable`` criterion in ONE place: land, a day-mean T in the livability window,
    and a 10-m day wind below ``calm_ms``.  Two entry points evaluate it on two different winds --
    ``attach_shelter`` on the sheltered wind (the default) and ``attach_amplification`` on the
    composed one under ``calm_on_terrain`` -- and the thresholds must not drift between them."""
    return (np.asarray(land, bool)
            & (np.asarray(T_map_C, float) >= CALM_T_LO_C)
            & (np.asarray(T_map_C, float) <= CALM_T_HI_C)
            & (np.asarray(u10_day_ms, float) < float(calm_ms)))


def attach_shelter(dC, topo, L_km=300.0, calm_ms=CALM_MS, wake_footprint_km=WAKE_FOOTPRINT_KM,
                   enclosure_radius_km=ENCLOSURE_RADIUS_KM):
    """Compute the shelter factor on a cascade's own topography + wind and ATTACH it (keys:
    shelter, u10_day_sheltered_ms, calm_habitable). Requires attach() to have run (needs
    u10_day_ms). ``topo`` is the cascade topo namespace/dict with x, e, height. Consumes
    ``dC['froude']`` (the #111 per-cell hydraulic Froude) when present, as ``attach_amplification``
    does, so the two branches read one regime. Returns summary.

    calm_habitable = land, day-mean T in the 0-30 C livability window, AND sheltered 10-m day
    wind below calm_ms -- the 'where would people actually settle' mask.

    NB the mask attached HERE keys on ``u10_day_sheltered_ms`` (the BLOCKED branch alone), because
    the composed wind does not exist yet at this point in the chain; ``attach_amplification`` then
    RE-ASKS the calm question of the composed ``u10_day_terrain_ms`` (its default since DOC-45,
    Jake's call) and boolean-CARVES the windstorm-prone cells out.  The carve is deliberate and
    stays: "is this ground calm" and "is this ground a windstorm alley" are different questions,
    and a habitability mask wants the second answered categorically (a place that is periodically
    a windstorm alley is not a calm place to settle, whatever its mean) rather than averaged into
    the first.  The WIND the first question is asked of is the composed one because keying it on
    the shelter branch alone meant the mask could not see the windstorm gain at all below
    ``amp_prone`` (1.29 judged on an unamplified wind, 1.31 removed outright -- a cliff at the
    threshold) nor the enclosure branch at all.  ``calm_ms`` is recorded on ``dC`` so the re-ask
    uses the same threshold (ROADMAP DOC-45)."""
    height = topo["height"] if isinstance(topo, dict) else topo.height
    x_m = topo["x"] if isinstance(topo, dict) else topo.x
    e_m = topo["e"] if isinstance(topo, dict) else topo.e
    fr = dC.get("froude")
    # The enclosure scans run here too, although this function reads only `shelter` (unlike
    # `shelter_factor` / `terrain_amplification`, which pass 0): the stash below is what
    # `attach_amplification` reuses, and it must carry the branch under the key it records.
    tf = terrain_factors(height, dC["u"], dC["v"], x_m, e_m, L_km=L_km,
                         wake_footprint_km=wake_footprint_km, froude=fr,
                         enclosure_radius_km=enclosure_radius_km)
    # Both branches come from one scan, and `attach_amplification` reuses it -- but ONLY if it was
    # computed under the same law parameters, which is what the key records.  Unkeyed, the reuse
    # silently discarded that function's own N/amp_max/... whenever this one ran first, which would
    # break exactly the N sweep ROADMAP DOC-46 asks for (PR #731 re-review, finding 2).
    dC["_terrain_factors"] = (_terrain_key(N_TROP, L_km, LEE_FOOTPRINT_KM, wake_footprint_km,
                                           AMP_MAX, 1.0, 0.9, 2.0, fr,
                                           enclosure_radius_km, M_ENC_ONSET, ENCLOSURE_FLOOR), tf)
    s = tf["shelter"]
    u10s = np.asarray(dC["u10_day_ms"], float) * s
    land = np.asarray(dC["land"], bool)
    T = np.asarray(dC["T_map_C"], float)
    calm_hab = _calm_habitable_mask(land, T, u10s, calm_ms)
    dC["shelter"] = s; dC["u10_day_sheltered_ms"] = u10s; dC["calm_habitable"] = calm_hab
    dC["calm_ms"] = np.float64(calm_ms)   # so `attach_amplification`'s re-ask uses this threshold
    nl = max(int(land.sum()), 1)
    hab = land & (T >= CALM_T_LO_C) & (T <= CALM_T_HI_C)
    return dict(land_shelter_mean=float(s[land].mean()) if land.any() else 1.0,
                habitable_land_frac=float(hab.sum()) / nl,
                calm_habitable_land_frac=float(calm_hab.sum()) / nl,
                calm_of_habitable=float(calm_hab.sum()) / max(int(hab.sum()), 1))


def terrain_amplification(height, u, v, x_m, e_m, N=N_TROP, L_km=300.0,
                          n_per=RAY_SAMPLES_PER_WINDOW, amp_max=AMP_MAX, M_crit=1.0,
                          ln_width=0.9, U_floor=2.0, amp_prone=1.3, froude=None,
                          footprint_km=LEE_FOOTPRINT_KM):
    # (the wake footprint is the shelter branch's; this one is pinned to the windstorm's)
    """Lee-windstorm amplification map (amp, prone) on the Stage C grid -- the WINDSTORM branch of
    the regime law, footprint-weighted (see ``terrain_factors``; [model-soft] texture diagnostic).

    ``amp`` = 1 + w_lee*(amplification_from_M(M) - 1) with M = N*H/U off the tallest obstacle on the
    L_km upwind ray and w_lee the fraction of it standing within ``footprint_km``: a cell at the foot of
    a critical-height ridge (M ~ 1) gets the full factor, a plain a few tens of km beyond it nothing.
    ``prone`` = amp > amp_prone flags windstorm/gap-jet/foehn cells.  The gate is the footprint now,
    not ``relief > 0`` over the whole 300 km ray.

    ``froude`` (optional): a per-cell hydraulic Froude number Fr=|V|/c -- issue #111's solver stores
    this (``dC['froude']``, the Froude of the thermal-low circulation). When supplied it REPLACES the
    N*H/U regime estimate via M=1/Fr. NB these are DISTINCT Froude numbers -- the hydraulic Fr of the
    flow vs the mountain Fr=U/(N*H) over the ridge -- not literally one quantity; they are used here as
    CO-CRITICAL regime flags (both signal windstorm-proneness near unity), and the footprint weight
    still ties the result to real terrain. A correlated proxy off #111, not a derived equivalence.

    Note on ``U`` in the fallback regime estimate: it is the local lowest-model-level wind speed
    |V| (the BL-depth wind), not the 10-m wind the factor multiplies. That is deliberate -- the CROSS-MOUNTAIN
    flow speed sets the lee-wave regime, while the windstorm accelerates the SURFACE wind beneath it;
    and on the #111-froude path U drops out entirely (Fr already encodes |V|/c).

    ``height`` (nx, ne) [m]; ``u``/``v`` used for DIRECTION and (fallback) the regime speed |V|;
    ``x_m``/``e_m`` the grid coordinate vectors [m] (x clamped at the walls, e periodic)."""
    amp = terrain_factors(height, u, v, x_m, e_m, N=N, L_km=L_km, footprint_km=footprint_km,
                          n_per=n_per, amp_max=amp_max, M_crit=M_crit, ln_width=ln_width,
                          U_floor=U_floor, froude=froude,
                          enclosure_radius_km=0.0)["amplification"]  # branch not read: skip its scans

    prone = amp > float(amp_prone)
    return amp, prone


def attach_amplification(dC, topo, N=N_TROP, L_km=300.0, amp_prone=1.3, amp_max=AMP_MAX,
                         footprint_km=LEE_FOOTPRINT_KM,
                         enclosure_radius_km=ENCLOSURE_RADIUS_KM, calm_on_terrain=True,
                         calm_ms=None):
    """Compute the lee-windstorm amplification and the enclosure branch on a cascade's own
    topography + wind and ATTACH them (keys: amplification, amplification_prone,
    u10_day_amplified_ms, enclosure). Consumes ``dC['froude']`` (the #111 per-cell hydraulic Froude)
    when present for the WINDSTORM regime, else falls back to the N*H/U topo estimate.

    CRUCIALLY: if ``calm_habitable`` is already attached (from ``attach_shelter``), this REMOVES the
    amplification-prone cells from it -- a windstorm alley is not a calm place to settle even if the
    shelter index alone called it calm.  And when ``shelter`` is attached it also attaches the
    COMPOSED terrain factor (keys: terrain_factor = shelter * amplification * enclosure,
    u10_day_terrain_ms) -- the composition the regime law licenses (ROADMAP DOC-34), with the
    enclosure branch that tells a basin floor from a lee slope (DOC-45).

    ``calm_on_terrain`` (default True -- Jake's call, 2026-09-14, on the DOC-45 recommendation)
    re-asks the calm question of the COMPOSED wind instead of the shelter branch alone, keeping the
    categorical windstorm carve.  See ``attach_shelter``'s note for why: the carve is the right
    shape, but asking the wind question of a wind the windstorm gain is absent from puts a cliff at
    ``amp_prone``, and that wind cannot see the enclosure branch at all.  Requires ``shelter`` and
    ``u10_day_ms`` (it needs the composed wind to exist); silently falls back to the sheltered-wind
    mask otherwise.  ``calm_on_terrain=False`` keeps the pre-DOC-45 shelter-then-carve mask, for a
    comparison.  ``calm_ms`` defaults to the threshold ``attach_shelter`` recorded on ``dC``
    (``CALM_MS`` if none), so the two entry points cannot disagree about what "calm" means.

    ``enclosure_radius_km = 0`` switches the enclosure branch off, restoring the two-branch factor
    byte-identically.  Returns a summary dict."""
    height = topo["height"] if isinstance(topo, dict) else topo.height
    x_m = topo["x"] if isinstance(topo, dict) else topo.x
    e_m = topo["e"] if isinstance(topo, dict) else topo.e
    fr = dC.get("froude")
    want = _terrain_key(N, L_km, footprint_km, WAKE_FOOTPRINT_KM, amp_max, 1.0, 0.9, 2.0, fr,
                        enclosure_radius_km, M_ENC_ONSET, ENCLOSURE_FLOOR)
    stash = dC.pop("_terrain_factors", None)   # POPPED: it is scratch, and callers np.savez(dC)
    if stash is not None and stash[0] == want:
        tf = stash[1]
    else:
        # `terrain_factors` rather than `terrain_amplification`: this path needs the ENCLOSURE
        # branch too, and that function is the windstorm branch on its own.
        tf = terrain_factors(height, dC["u"], dC["v"], x_m, e_m, N=N, L_km=L_km,
                             footprint_km=footprint_km, amp_max=amp_max, froude=fr,
                             enclosure_radius_km=enclosure_radius_km)
    amp = tf["amplification"]
    enc = tf["enclosure"]
    prone = amp > float(amp_prone)
    dC["amplification"] = amp
    dC["amplification_prone"] = prone
    dC["enclosure"] = enc
    land = np.asarray(dC["land"], bool)
    if "u10_day_ms" in dC:
        dC["u10_day_amplified_ms"] = np.asarray(dC["u10_day_ms"], float) * amp
        if "shelter" in dC:
            dC["terrain_factor"] = np.asarray(dC["shelter"], float) * amp * enc
            dC["u10_day_terrain_ms"] = np.asarray(dC["u10_day_ms"], float) * dC["terrain_factor"]
    if "calm_habitable" in dC:
        if calm_on_terrain and "u10_day_terrain_ms" in dC:
            if calm_ms is None:
                calm_ms = float(dC["calm_ms"]) if "calm_ms" in dC else CALM_MS
            dC["calm_habitable"] = _calm_habitable_mask(
                land, dC["T_map_C"], dC["u10_day_terrain_ms"], calm_ms)
        dC["calm_habitable"] = np.asarray(dC["calm_habitable"], bool) & ~prone
    nl = max(int(land.sum()), 1)
    out = dict(
        land_amplification_mean=float(amp[land].mean()) if land.any() else 1.0,
        land_enclosure_mean=float(enc[land].mean()) if land.any() else 1.0,
        enclosure_land_frac=float(int(((enc < 1.0) & land).sum())) / nl,
        amplification_prone_land_frac=float(int((prone & land).sum())) / nl,
        froude_source=("#111 per-cell" if fr is not None else "topo+stratification fallback"),
    )
    if "calm_habitable" in dC:
        out["calm_habitable_land_frac"] = float(int(np.asarray(dC["calm_habitable"]).sum())) / nl
    return out


# ======================================================================================
# issue #114: GUST factors -- everything else here is a time-MEAN wind, but habitability also keys on
# GUSTS. Standard surface-layer turbulence intensity: I_u = sigma_u/U ~ a_u*k/ln(z/z0) (a_u=sigma_u/u*
# ~ 2.5, von Karman k=0.4), so the gust factor G = U_gust/U_mean = 1 + g_peak*I_u with the peak factor
# g_peak ~ 2-3. Rougher cover -> stronger gusts. A separate convective cold-pool outflow adds a
# density-current speed ~ sqrt(2*CAPE) where convection is deep. CONSTANTS LITERATURE-ANCHORED, NOT
# DERIVED (a_u, k, g_peak, the outflow coefficient): quote the gust as a band, not a precise value.
# ======================================================================================

def turbulence_intensity(z0, z_ref=10.0, a_u=2.5, k=0.4):
    """Streamwise turbulence intensity I_u = sigma_u/U ~ a_u*k/ln(z_ref/z0) for a neutral surface
    layer (a_u = sigma_u/u* ~ 2.5, von Karman k = 0.4). Rougher z0 -> higher I_u [model-soft]."""
    z0 = np.asarray(z0, float)
    if np.any(z0 >= z_ref):
        raise ValueError(f"turbulence_intensity: z0 must be < z_ref={z_ref} m (got max {z0.max():.3g}).")
    return float(a_u) * float(k) / np.log(z_ref / z0)


def gust_factor(z0, z_ref=10.0, g_peak=3.0, a_u=2.5, k=0.4):
    """Gust factor G = U_gust/U_mean = 1 + g_peak*I_u (peak factor g_peak ~ 2-3; the value 3 is the
    common ~3 s gust over a ~10 min mean). Multiplies a mean wind to a gust [model-soft]."""
    return 1.0 + float(g_peak) * turbulence_intensity(z0, z_ref=z_ref, a_u=a_u, k=k)


def convective_gust_speed(cape, k_outflow=0.3):
    """Convective cold-pool OUTFLOW speed [m/s] ~ k_outflow*sqrt(2*CAPE) -- the density-current speed
    of a downdraft-driven gust front (CAPE [J/kg] the convective available potential energy). Zero
    for non-buoyant (CAPE <= 0) columns. k_outflow ~ 0.3 is the standard efficiency [model-soft].

    CAVEAT on the model's CAPE: the cascade outputs ``dC['cape']`` = the ENTRAINING-PLUME CAPE from
    plume_floft (the saturated updraft's positive-buoyancy integral vs a near-moist-adiabatic
    environment), evaluated per ROW off the SST backbone (``cape_map_from_theta``) and broadcast
    along-track -- so it carries no along-track texture at all.  On the canonical ``lens_diurnal``
    world it is LARGE: ~4,000 J/kg across the habitable band (p5 3,994, p95 5,116 on the shipped map
    archive), so this formula returns ~27 m/s there, a bump that would swamp the roughness gust
    (measured: the map's land gust median 13.6 -> 41.9 m/s against a 9.8 m/s mean).  [An earlier
    edition of this docstring called the field "SMALL (tens of J/kg)" and the bump "~few m/s";
    that does not describe the canonical edition -- ROADMAP DOC-34.]  More importantly CAPE is a
    POTENTIAL: a cold pool needs convection to actually fire, so a gust built from CAPE alone needs
    a convective FREQUENCY weight this module does not have.  Treat the term as an upper-envelope
    "if a storm fires here" gust, not a mean-day gust -- which is why the map webapp's felt-wind gust
    leaves it out (``map_webapp/derived.py::fw_convective_gust``). [model-soft, frequency-unweighted]"""
    cape = np.asarray(cape, float)
    return float(k_outflow) * np.sqrt(2.0 * np.clip(cape, 0.0, None))


# issue #115 (Track M, M3.4): the organised-convection EDDY GUST. The convection-permitting (e,z) slice
# forced by the real b23 inner sea gives a transient surface-wind RMS of ~11 m/s (propagating moist
# filaments / cold-pool fronts over the hot ocean) -- a THIRD gust contribution, distinct from the
# boundary-layer turbulence factor and the local single-cell cold-pool outflow above. It is an
# independent fluctuation, so it adds in QUADRATURE. [model-soft: a 2-D slice, amplitude-soft.]
EDDY_GUST_RMS_INNERSEA = 11.0          # [m/s] M3.4 RMS organised-convection eddy gust over the inner sea


def attach_gust(dC, z_ref=10.0, g_peak=3.0, cape=None, eddy_gust_ms=0.0):
    """Compute the 10-m DAY gust wind and ATTACH it (keys: u10_gust_factor, u10_day_gust_ms). Requires
    ``attach`` to have run (needs z0 and u10_day_ms). u10_day_gust = G(z0)*u10_day, plus a convective
    cold-pool outflow speed where ``cape`` (per-cell [J/kg], optional) is large, plus -- in quadrature --
    an ``eddy_gust_ms`` organised-convection transient (the #115 M3.4 inner-sea term ~``EDDY_GUST_RMS_INNERSEA``;
    pass it per-cell or scalar over the inner sea, default 0 = byte-identical). The night wind is calmer
    (stable, decoupled) so the gust panel keys on the DAY (coupled, gusty) wind. Returns a summary dict."""
    z0 = np.asarray(dC["z0"], float)
    G = gust_factor(z0, z_ref=z_ref, g_peak=g_peak)
    gust = np.asarray(dC["u10_day_ms"], float) * G
    if cape is not None:
        gust = gust + convective_gust_speed(cape)
    if np.any(eddy_gust_ms):                                  # #115 M3.4: independent transient -> quadrature
        gust = np.hypot(gust, eddy_gust_ms)
    dC["u10_gust_factor"] = G
    dC["u10_day_gust_ms"] = gust
    land = np.asarray(dC["land"], bool)
    out = dict(gust_factor_mean=float(G.mean()), u10_day_gust_mean_ms=float(gust.mean()))
    if land.any():
        out["land_u10_day_gust_mean_ms"] = float(gust[land].mean())
    return out


def wind_report(cache, out=None, koppen=None, h_bl=1000.0, dpi=200):
    """Render the wind habitability report from a persisted cascade .npz: lowest-level |V| with the
    cap-clip outline, the 10-m day/night maps, the terrain-sheltered and lee-amplified day winds
    (issue #114), the 10-m day gust map, and the zonal profiles. Panels appear as their inputs are
    available (shelter/amplification need topo; the gust panel always)."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import geometry as g

    z = np.load(cache, allow_pickle=True)
    dC = z["dC"].item()
    kop = z["koppen"] if (koppen is None and "koppen" in z) else koppen
    topo = z["topo"].item() if "topo" in z else None
    summary = attach(dC, koppen=kop, h_bl=h_bl)
    if topo is not None and topo.get("height") is not None:
        summary.update(attach_shelter(dC, topo))
        summary.update(attach_amplification(dC, topo))   # #114: lee-windstorm exposure (carves calm_habitable)
    summary.update(attach_gust(dC, cape=dC.get("cape")))  # #114: 10-m day gust (convective bump if CAPE present)

    spd = np.hypot(np.asarray(dC["u"], float), np.asarray(dC["v"], float))
    cap = float(dC.get("wind_cap", 40.0))
    capped = spd >= 0.999 * cap
    e_km = np.asarray(topo["e"], float) / 1e3 if topo is not None else np.arange(spd.shape[1])
    x_km = np.asarray(topo["x"], float) / 1e3 if topo is not None else np.arange(spd.shape[0])
    land = np.asarray(dC["land"], bool)
    th = np.asarray(dC["theta_deg"], float)

    # build the panel list dynamically; each entry = (field, title, overlay, vmax). The base wind-
    # ladder panels share spd.max() so the calm habitability structure stays legible/comparable; the
    # LEE-AMPLIFIED/GUST panels (which run up to ~2.5x) get their OWN scale so a few windstorm cells
    # don't wash out every map (review #1).
    base_vmax = float(spd.max())
    panels = [(spd, f"lowest model level |V| [m/s]\n(diagnostic-soft; red dash = at cap {cap:.0f})", "capped", base_vmax),
              (dC["u10_day_ms"], "10-m wind, DAY (coupled BL) [m/s]\n(log-layer reduction; z0 from cover)", None, base_vmax),
              (dC["u10_night_ms"], "10-m wind, NIGHT (decoupled land) [m/s]\n(graded by the cell's own diurnal range)", None, base_vmax)]
    if "shelter" in dC:
        # the calm overlay is drawn over the wind the mask is COMPUTED on: the composed wind since
        # DOC-45's default (`calm_on_terrain`), the sheltered wind alone when only `attach_shelter`
        # has run -- a caption that names a threshold must sit on the field the threshold was
        # applied to (PR #814 review, round 2)
        # the threshold the mask was actually built with (`attach_shelter` records it), not the
        # module default -- the same shape as the defect this caption fixed, one level down
        calm_thr = float(dC["calm_ms"]) if "calm_ms" in dC else CALM_MS
        calm_cap = (f"red = calm habitable: land, {CALM_T_LO_C:.0f}–{CALM_T_HI_C:.0f} °C, "
                    f"<{calm_thr:.0f} m/s on this wind")
        if "u10_day_terrain_ms" in dC:
            panels.append((dC["u10_day_terrain_ms"],
                           f"10-m day wind, TERRAIN-COMPOSED [m/s]\n(shelter x lee gain x enclosure; {calm_cap})",
                           "calm", base_vmax))
        else:
            panels.append((dC["u10_day_sheltered_ms"],
                           f"10-m day wind, TERRAIN-SHELTERED [m/s]\n({calm_cap})", "calm", base_vmax))
    if "u10_day_amplified_ms" in dC:
        panels.append((dC["u10_day_amplified_ms"],
                       "10-m day wind, LEE-AMPLIFIED [m/s]\n(downslope/gap/foehn; magenta = amplification-prone; own scale)",
                       "prone", float(np.nanmax(dC["u10_day_amplified_ms"]))))
    if "u10_day_gust_ms" in dC:
        panels.append((dC["u10_day_gust_ms"],
                       "10-m day GUST [m/s]\n(G=1+g·σ_u/U, lit. constants; +cold-pool where CAPE; own scale)",
                       None, float(np.nanmax(dC["u10_day_gust_ms"]))))
    n_pan = len(panels) + 1
    fig, axs = plt.subplots(1, n_pan, figsize=(5.8 * n_pan, 5.2))
    for ax, (fld, title, overlay, vmax) in zip(axs[:-1], panels):
        im = ax.pcolormesh(e_km, x_km, fld, shading="nearest", cmap="viridis", vmin=0.0, vmax=vmax)
        ax.contour(e_km, x_km, land.astype(float), levels=[0.5], colors="k", linewidths=0.5)
        if overlay == "capped" and capped.any():
            ax.contour(e_km, x_km, capped.astype(float), levels=[0.5], colors="r",
                       linewidths=0.7, linestyles="--")
        if overlay == "calm" and "calm_habitable" in dC and dC["calm_habitable"].any():
            ax.contourf(e_km, x_km, dC["calm_habitable"].astype(float), levels=[0.5, 1.5],
                        colors=["r"], alpha=0.45)
        if overlay == "prone" and "amplification_prone" in dC and dC["amplification_prone"].any():
            ax.contourf(e_km, x_km, dC["amplification_prone"].astype(float), levels=[0.5, 1.5],
                        colors=["m"], alpha=0.45)
        plt.colorbar(im, ax=ax, fraction=0.046)
        cand = [t for t in range(0, 81, 2) if th.min() - 1e-6 <= t <= th.max() + 1e-6]
        ax.set_yticks([g.x_of_theta(np.radians(t)) / 1e3 for t in cand])
        ax.set_yticklabels([f"{t}°" for t in cand], fontsize=7)
        ax.set_title(title, fontsize=9); ax.set_xlabel("along-track e (km)")
    ax = axs[-1]
    ax.plot(th, spd.mean(axis=1), "k-", lw=1.5, label="lowest level (row mean)")
    if "ubg" in dC:
        ax.plot(th, np.abs(np.asarray(dC["ubg"], float)), "k:", lw=1, label="|u_bg| backbone")
    ax.plot(th, np.asarray(dC["u10_day_ms"]).mean(axis=1), "C0-", lw=1.5, label="10-m day")
    ax.plot(th, np.asarray(dC["u10_night_ms"]).mean(axis=1), "C1-", lw=1.5, label="10-m night")
    if "u10_day_sheltered_ms" in dC:
        ax.plot(th, np.asarray(dC["u10_day_sheltered_ms"]).mean(axis=1), "C2-", lw=1.5,
                label="10-m day, sheltered")
    if "u10_day_amplified_ms" in dC:
        ax.plot(th, np.asarray(dC["u10_day_amplified_ms"]).mean(axis=1), "C3-", lw=1.5,
                label="10-m day, lee-amplified")
    if "u10_day_gust_ms" in dC:
        ax.plot(th, np.asarray(dC["u10_day_gust_ms"]).mean(axis=1), "C4--", lw=1.5,
                label="10-m day, gust")
    ax.set_xlabel("θ (deg)"); ax.set_ylabel("wind speed [m/s]")
    ax.set_title("zonal-mean wind by estimate height\n(the habitability ladder)", fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"near-surface wind report — {os.path.basename(cache)}  "
                 f"(land means: lev {summary.get('land_lev_mean_ms', float('nan')):.1f} → "
                 f"day {summary.get('land_u10_day_mean_ms', float('nan')):.1f} → "
                 f"night {summary.get('land_u10_night_mean_ms', float('nan')):.1f} m/s)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if out is None:
        out = f"wind_report_{os.path.splitext(os.path.basename(cache))[0]}.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"saved {out}")
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="10-m wind habitability report from a persisted cascade")
    p.add_argument("cache", help="cascade .npz (e.g. climate_seed67.npz or cascade_ckpt_<tag>_<profile>.npz)")
    p.add_argument("--out", default=None)
    p.add_argument("--h-bl", type=float, default=1000.0, help="boundary-layer depth [m] (Stage C H_bl)")
    p.add_argument("--dpi", type=int, default=200)
    a = p.parse_args()
    s = wind_report(a.cache, out=a.out, h_bl=a.h_bl, dpi=a.dpi)
    for k, v in s.items():
        print(f"  {k:30s} {v:6.2f}" if isinstance(v, (int, float)) else f"  {k:30s} {v}")
