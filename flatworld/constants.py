"""
flatworld.constants -- the setting's fixed physical givens, in SI, with provenance.

Everything here is an INPUT (a given of the setting or of real physics), verified against
``../../cosmology.md`` where the doc states it. Nothing here is a climate *output* -- no
temperatures, winds, rain rates, or ocean structure. Those are derived by the model.

The single free *amplitude* of the forcing -- the suns' brightness ``I0`` -- is documented at
``I0_OVERHEAD`` below: it is an input (a setting parameter), not a temperature anchor.

References are to sections of the cosmology doc (``../../cosmology.md``):
  S2  shape/gravity/underside ; S3 suns ; S4 sunlight/temperature ; S5 winds/rain ; S6 the day.
"""
from __future__ import annotations
import os
import numpy as np


# Defaults of every COSMO_* knob this module resolves, recorded as it resolves them. Consumed by
# ``ckpt_world.env_diff`` to tell "the checkpoint did not carry this key, and the live value is the same
# default it therefore used" from "the live value differs, so the world differs". Without that distinction
# a knob merely EXPORTED at its own default reads as a world change: PR #440 pinned COSMO_ALBEDO_* /
# COSMO_F_VIS_SW in canonical_env.sh at exactly the values below, and every checkpoint predating it then
# failed the world guard -- which blocked the whole band-window rebuild chain on a difference of nothing.
ENV_DEFAULTS: "dict[str, str]" = {}


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean setting from the environment (parallels stage_ad_core._env_flag)."""
    ENV_DEFAULTS.setdefault(name, "1" if default else "0")
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() not in ("", "0", "false", "no", "off")


def _env_float(name: str, default: float) -> float:
    """Read a float setting from the environment (read at import, like every flatworld given --
    set COSMO_* in the SHELL before importing flatworld; OPERATIONS.md import-timing trap).

    An EMPTY value counts as unset. ``export COSMO_ALBEDO_ICE=`` is how a shell script blanks a knob,
    and ``float("")`` raises -- which would kill the run at IMPORT, before any traceback could name
    the knob. (``_env_flag`` above deliberately differs: it reads an empty string as OFF, since for a
    boolean that is the reading an operator blanking a flag intends.) A non-empty but unparseable
    value still raises, on purpose: a typo'd number must not be silently swallowed into the default."""
    ENV_DEFAULTS.setdefault(name, repr(float(default)))
    v = os.environ.get(name)
    return default if v is None or not v.strip() else float(v)

# --------------------------------------------------------------------------------------
# Universal physical constants (real physics; the world "runs on ordinary physics", S1)
# --------------------------------------------------------------------------------------
SIGMA_SB = 5.670374419e-8      # Stefan-Boltzmann [W m^-2 K^-4]
T_FREEZE = 273.15              # 0 deg C in K
KM = 1.0e3                     # metres per km

# --------------------------------------------------------------------------------------
# Geometry & gravity (S2). Uniform field, no falloff with altitude.
# --------------------------------------------------------------------------------------
G_ACCEL = 9.8                  # surface gravity [m s^-2]; uniform "down", does NOT weaken with height
# Consequence honored elsewhere: no escape to space => atmospheric inventory is conserved.

# --------------------------------------------------------------------------------------
# The suns (S3). Height L is the one explicitly tunable length (Appendix H). ⚠️ The default below (20,000 km)
# is a TEST/nominal value; the CANONICAL world is 56,520 km (COSMO_L_SUN=5.652e7, canonical_env.sh) — see below.
# --------------------------------------------------------------------------------------
L_SUN = float(os.environ.get("COSMO_L_SUN", 2.0e7))   # suns' height above the plane [m]; 2.0e7 = 20,000 km.
                               # RAISED L->2L (2026-06-07 worldbuilding decision): doubling the suns' height
                               # weakens the cross-track temperature gradient -> weaker poleward transport ->
                               # the (transport-heated) habitable ribbon moves EQUATORWARD toward a higher sun,
                               # while full brightness keeps a scorching equatorial salt flat. Day stays 24 h
                               # (imposed); spacing/speed scale with L; flux S(theta)=I0 cos^3 theta is
                               # L-invariant (so the ribbon's sun-elevation CEILING is set by I0, not L -- the
                               # AD-core finding, flatworld/RAISE_SUNS_RESULTS.md). Was 1.0e7 (10,000 km).
                               # ⚠️ 2.0e7 IS A TEST/NOMINAL DEFAULT, *NOT* THE CANONICAL WORLD. The canonical
                               # worldbuilding world is L = 5.652e7 (56,520 km), selected by COSMO_L_SUN=5.652e7 in
                               # canonical_env.sh (the lens_diurnal edition, canonical since 2026-07-18). The 8.0e7
                               # (80,000 km) this comment used to name was a REAL earlier world: L=8e7 was CHOSEN by Jake on
                               # 2026-06-15 (archive/COSMO_REFRESH_ROADMAP.md) and spanned a whole lineage of editions --
                               # sh_b23L8 -> bl_dz20_i100 (2026-06-18) -> nsub30 (2026-06-26) -> bm_tau2h (2026-07-02) -- until
                               # lens_diurnal superseded it on 2026-07-18. That 2026-06-15 is the DECISION's date, not any
                               # edition's; do not conflate them (CANONICAL_WORLD.md, "Editions (history -- do not conflate)").
                               # Do NOT read this 2.0e7 as "the setting value" --
                               # it is what tests / quick runs use. Set COSMO_L_SUN in the shell BEFORE importing
                               # flatworld (import-timing forcing trap -- OPERATIONS.md; full world -> CANONICAL_WORLD.md).

# Haze / atmospheric extinction for the *visible/solar* slant path that fixes the day's clock and the
# low-sun dimming (S6, Appendix C). tau0 is the extra slant optical depth at the noon reference minus 1
# airmass; "hazy, humid-Earth sky" choice. This is the SOLAR-band extinction used for the geometric
# day-clock; the climate radiation scheme (radiation.py) carries its own band optical depths.
# This 1.0 is BOTH the collimator-off flux haze AND the clock-spacing reference (see COLLIMATOR below: with
# the collimator on, the FLUX uses the thin TAU0_HAZE_THIN while the clock still uses this 1.0 reference).
TAU0_HAZE = 1.0

# Haze SCATTERING split (CLEARSKY_SW_PLAN Stage 5). TAU0_HAZE above is the slant EXTINCTION (absorption +
# scattering combined). For the day-clock and the DIRECT beam it stays pure extinction (unchanged). But for
# the SURFACE-FLUX role, treating all of it as pure loss is wrong: a real Rayleigh/aerosol haze SCATTERS most
# of the extinguished light, and the FORWARD-scattered part still reaches the ground as DIFFUSE light (a hazy
# sky is bright, not dark). Two physical givens partition the extinction -- both ESTABLISHED atmospheric
# values, NOT tuned to any climate output (cross-cutting rule #3):
#   * OMEGA0_HAZE -- single-scattering albedo (scattered / extinguished). Rayleigh is pure scattering (omega=1);
#     humid-sky tropospheric aerosol is ~0.85-0.98 (scattering-dominated; dust lower). 0.9 is a conservative
#     scattering-dominated value for the "humid-Earth sky" the haze represents.
#   * G_HAZE -- scattering asymmetry g = <cos(scatter angle)>. Rayleigh is symmetric (g=0); accumulation-mode
#     aerosol is forward-peaked (Mie, g~0.6-0.75). The two-stream FORWARD (downward) hemispheric fraction is
#     (1+g)/2 (backscatter fraction (1-g)/2), so g=0.7 -> ~0.85 of the scattered light goes forward to ground.
# Used ONLY by the diffuse-down stream (geometry.day_mean_diffuse_flux + radiation.shortwave), default OFF
# behind the sw_diffuse flag; the midnight-clock role of TAU0_HAZE (SunsClock) is left untouched.
OMEGA0_HAZE = 0.9             # haze single-scattering albedo (scattering-dominated humid sky; Rayleigh=1)
G_HAZE = 0.7                  # haze scattering asymmetry (forward-peaked aerosol; Rayleigh=0)

# Midnight darkness floor that *fixes* the suns' clock (S6/Appendix C): at midnight the two nearest
# suns together deliver this fraction of noon. A moonlit ~1e-6 floor through tau0=1 air gives the ~24 h
# day, spacing ~205,000 km, speed ~2.37 km/s. These follow from the floor; they are NOT independent.
MIDNIGHT_FLOOR_FRAC = 1.0e-6

# Reference latitude for the day-clock geometry (the noon zenith angle used in the brightness law). The
# doc fixes the clock from a temperate-band observer at theta0 = 50 deg (Appendix C).
THETA0_CLOCK_DEG = 50.0

# --------------------------------------------------------------------------------------
# The suns' COLLIMATORS (S3/S6) -- OPT-IN worldbuilding geometry behind COSMO_COLLIMATOR (default OFF;
# the no-collimator clock+flux stay BYTE-IDENTICAL when off, so the test gate is unchanged).
# --------------------------------------------------------------------------------------
# Each sun rides inside a co-moving, axis-vertical dark CYLINDER (open top & bottom). A ground observer
# sees the sun only while the line of sight threads the tube -- i.e. while the sun's zenith angle z < z_cut,
# with tan(z_cut)=R/h set by the tube's height/radius ratio H/R = 2/tan(z_cut). Because cos z = L/r the
# gate is simply r < r_cut = L/cos(z_cut); the visible disk FRACTION (geometry._visible_fraction) softens
# the edge over the sun's finite angular width. Consequences (derived -- geometry + cosmology.md S6):
#   * an axisymmetric, leak-free blackout -- NIGHT IS A GEOMETRIC ZERO (no haze needed for darkness);
#   * a sharp day/night with a brief finite-disk partial-eclipse SUNSET (~4 min at the ribbon for H/R=0.3);
#   * a HARD cutoff LATITUDE: the noon zenith angle equals theta, so beyond theta=z_cut the sun never
#     clears the tube even at noon => permanent night, freezing (cooling) the already-cold high latitudes.
# With darkness now geometric, the haze is made THIN & bright (TAU0_HAZE_THIN): it brightens the day by
# forward-scatter (Stage-5 diffuse) without darkening the night. The spacing D is KEPT at its established
# value (the SunsClock derives it from the TAU0_HAZE=1.0 reference floor regardless), so H/R alone sets the
# night length; the MIDNIGHT_FLOOR_FRAC clock-pin is retired as the *darkness* mechanism. (See
# eclipsing_cylinder.py for the H/R study; the FULL climate re-equilibration with the gate on is STAGED --
# OPEN_ISSUES "collimator".)
COLLIMATOR = _env_flag("COSMO_COLLIMATOR", True)    # DEFAULT ON (the worldbuilding geometry of the accurate
                                                    # model): dark-tube geometric night + thin bright haze +
                                                    # the I0 re-pin. Set COSMO_COLLIMATOR=0 for the no-gate,
                                                    # full TAU0_HAZE=1.0, fiducial-I0 limit (BYTE-IDENTICAL to
                                                    # the pre-collimator clock+flux).
COLLIMATOR_HR = 0.30          # tube height/radius ratio (worldbuilding choice). z_cut=atan(2/HR)~81.5 deg ->
                              # ~8.6 h pitch-black night / ~15.4 h day at the theta0=50 ribbon, permanent-night
                              # cutoff latitude ~81.5 deg. COSMO_COLLIMATOR_HR env overrides for sweeps.
COLLIMATOR_HR = float(os.environ.get("COSMO_COLLIMATOR_HR", COLLIMATOR_HR))
COLLIMATOR_ZCUT = float(np.arctan(2.0 / COLLIMATOR_HR))   # cutoff zenith angle [rad] (sun blocked beyond this)
R_SUN_BODY = 9.0e4            # sun's physical radius [m] (~90 km): sets the sun's angular size and hence the
                              # finite-disk partial-eclipse SUNSET duration (~4 min at the ribbon for H/R=0.3).
TAU0_HAZE_THIN = 0.30         # the collimator world's THIN bright haze for the FLUX (vs the TAU0_HAZE=1.0
                              # clock reference): darkness is geometric, so the haze only brightens the day.
# I0 RE-PIN. Thinning the haze (1.0->0.3) brightens the day-mean; to HOLD the equator's day-mean FLUX (hence
# its temperature -- the re-pin is a pure SW-geometry ratio, INDEPENDENT of the LW/greenhouse scheme) the suns
# are dimmed by S_eq(full haze, no gate)/S_eq(thin haze, gated) at i0=1. RE-DERIVED at the current L=2e7
# geometry (2026-06-09, ALLON_CAMPAIGN.md): S_eq_off=1570.19, S_eq_on=1570.95 at the prior 0.841 -> the
# precise value is 0.841 * 1570.19/1570.95 = 0.8406 (the prior 0.841 already held the equator to 0.05%). NOTE
# this holds the EQUATOR; the haze dimmed low sun more, so thinning it brightens the ribbon ~1.55x (S_rib50
# 337->522) and flattens the latitude gradient -- a FINDING (not neutral) the production campaign measures.
COLLIMATOR_I0_REPIN = float(os.environ.get("COSMO_COLLIMATOR_I0REPIN", 0.8406))

# --- THE FREE FORCING AMPLITUDE (an INPUT, not a temperature target) ---------------------------------
# I0 = the flux a single sun delivers to a surface directly beneath it at distance L (theta=0, noon),
# i.e. the amplitude in I(theta) = I0 * cos^3(theta). The shape (cos^3) and L are given; this amplitude
# is the suns' luminosity, the one free brightness parameter. The fiducial value below is the one
# consistent with the doc's stated *fluxes* -- ~3,300 W/m^2 at theta=50 noon and ">20x Earth" at the
# equator (S6) -- which is a FLUX statement, independent of the (retired) "5 C at theta=50" temperature
# anchor. Surface temperature is predicted downstream from this; see DESIGN.md section 1.
#   check: I0 * cos^3(50) * exp(-tau0*(sec50 - 1)) ~= 3300 W/m^2  (geometry.verify_against_doc)
I0_OVERHEAD = 21650.0 * float(os.environ.get("COSMO_I0_SCALE", 1.0)) \
                      * (COLLIMATOR_I0_REPIN if COLLIMATOR else 1.0)   # [W m^-2] one sun overhead at L.
                               # FIDUCIAL 21650 (i0_scale=1.0). COSMO_I0_SCALE env dims/brightens the suns in
                               # parallel runs WITHOUT editing this file (e.g. 0.6 = 60% brightness) -- the
                               # clock is brightness-INDEPENDENT so spacing/speed are unaffected; only the
                               # flux S(theta)=I0 cos^3 theta scales. Used to explore the sun-angle<->equator-T
                               # tradeoff (RAISE_SUNS_RESULTS.md). Committed default is full brightness (1.0).

# --------------------------------------------------------------------------------------
# Dry-air thermodynamics (Earth-like atmosphere, S2/S5). Real-physics constants.
# --------------------------------------------------------------------------------------
P_SURF = 1.013e5               # reference surface pressure [Pa] (~1 atm; S8 cold-trap inventory)
R_DRY = 287.0                  # specific gas constant, dry air [J kg^-1 K^-1]
CP_AIR = 1004.0                # specific heat of air at constant pressure [J kg^-1 K^-1]
KAPPA_POISSON = R_DRY / CP_AIR # ~0.286, for potential temperature / dry adiabat
RHO_AIR0 = 1.2                 # reference near-surface air density [kg m^-3]

# --------------------------------------------------------------------------------------
# Water thermodynamics (real physics; the world is wet, S5/S9).
# --------------------------------------------------------------------------------------
L_VAP = 2.5e6                  # latent heat of vaporization [J kg^-1]
L_FUS = 3.34e5                 # latent heat of fusion [J kg^-1]
L_SUB = L_VAP + L_FUS          # latent heat of sublimation [J kg^-1] (~2.83e6; deposition/sublimation
                               # below freezing -- the ice-phase latent heat the prognostic condensate
                               # microphysics uses, CLOUD_PROGNOSTIC_ROADMAP Chunk 2)
R_VAP = 461.5                  # specific gas constant, water vapor [J kg^-1 K^-1]
EPS_MW = R_DRY / R_VAP         # ~0.622, molar-mass ratio (for specific humidity from vapor pressure)
ES0 = 611.2                    # saturation vapor pressure at 0 C [Pa] (Clausius-Clapeyron reference)
CP_LIQ = 4187.0               # specific heat of liquid water [J kg^-1 K^-1]
RHO_LIQ = 1000.0              # density of liquid water [kg m^-3]
SALINITY_REF = 0.035          # reference ocean salinity (mass fraction); marine, S9
RHO_HALITE = 2170.0           # density of solid halite (NaCl) [kg m^-3] -- the deposited salt
                              # crust of the surf_salt budget (dam height = S_dep / RHO_HALITE)

# --------------------------------------------------------------------------------------
# Surface heat capacities, as effective slab heat capacities C [J m^-2 K^-1] (used by the diurnally
# resolved surface energy balance, Stage A/C). These set the diurnal swing: thin (rock) lurches, thick
# (ocean mixed layer) barely stirs -- the maritime buffering that sets the lived climate (S6).
# C = rho * c_p * depth.  Derived, not anchored.
# --------------------------------------------------------------------------------------
def slab_heat_capacity(depth_m: float, rho: float, c: float) -> float:
    """Effective areal heat capacity [J m^-2 K^-1] of a slab of given depth, density, specific heat."""
    return rho * c * depth_m

# Representative surfaces (depths are the diurnal skin/active layer, not the full reservoir):
C_BARE_ROCK = slab_heat_capacity(0.10, 2600.0, 800.0)    # ~10 cm of dry rock/sand (thin -> big swing)
C_ARID_LAND = slab_heat_capacity(0.25, 1600.0, 900.0)    # ~25 cm dry soil/regolith active layer: drier ->
                                                          # shallower, lower-inertia layer than damp soil, so a
                                                          # LARGER diurnal swing (the "dry interior" continentality
                                                          # endpoint, Stage C diurnal cycle). Between bare rock and
                                                          # moist land: 3.7x below C_MOIST_LAND, 1.7x above C_BARE_ROCK.
C_MOIST_LAND = slab_heat_capacity(0.50, 1800.0, 1500.0)  # damp soil, deeper active layer
C_OCEAN_MIXED = slab_heat_capacity(50.0, RHO_LIQ, CP_LIQ) # 50 m ocean mixed layer (huge -> tiny swing)

# --------------------------------------------------------------------------------------
# Default radiative tunables (documented inputs; swept in config.py). NOT outputs.
# The four surface albedos are BROADBAND means under the solar-analog 50/50 visible/near-IR
# split (radiation.F_VIS_SW default). They are SPECTRUM-DEPENDENT: each moves by the ratio the
# derivation script ``cosmology_models/derive_spectrum_albedos.py`` computes from published
# spectral reflectance. Under the ADOPTED suns' spectrum (SPECTRUM_PLAN.md rev 5: solar-like,
# 200-320 nm notched out, delivered flux held) all four move by less than 0.005 -- under half
# the quantum these two-decimal constants are quoted at -- so the values below ARE the adopted
# design, not stale pre-spectrum defaults. Signs are the ones physics demands: the notched band
# is bright for snow (~0.98) and dark for soil (~0.05), so removing it darkens ice (0.60 ->
# 0.5965) and brightens land (0.25 -> 0.2536), each by ~1 % of itself. Env-overridable
# (COSMO_ALBEDO_*, set in the shell BEFORE import -- spectrum_env.sh re-exports exactly these
# four, i.e. it is a no-op); unset env is byte-identical either way.
# --------------------------------------------------------------------------------------
ALBEDO_OCEAN = _env_float("COSMO_ALBEDO_OCEAN", 0.08)    # broadband surface albedo of open water
ALBEDO_LAND = _env_float("COSMO_ALBEDO_LAND", 0.25)      # broadband surface albedo of land (rock/soil mean)
ALBEDO_ICE = _env_float("COSMO_ALBEDO_ICE", 0.60)        # broadband surface albedo of ice/snow
ALBEDO_DRYLAND = _env_float("COSMO_ALBEDO_DRYLAND", 0.30)  # salt flat / bare rock in the hot zone
                               # (brighter than ocean). Consolidates the three hand-duplicated 0.30
                               # literals (stage_a_column / stage_ad_config / stage_ad_column), the
                               # same drift hazard the SST_WET_HI note below records (values
                               # unchanged, byte-identical).
CO2_PPM = 400.0               # CO2 mixing ratio [ppmv] (greenhouse input; Earth-like default)
# Surface evaporation-efficiency (wetness) ramp endpoints [deg C]: open ocean (wet=1) below SST_WET_HI,
# dry salt flat (wet=0) above SST_DRY -- the liquid-water / salt-flat evaporation limit (the
# Komabayashi-Ingersoll latent-runaway cap). The source of truth for the four cascade wetness sites
# (stage_ad_core.wetness_of_Tsurf, stage_a_column._wetness, stage_b_overturning._wetness_of_theta,
# stage_c_map._wetness_col) AND salt_sill.SST_WET_HI (the load-bearing salt-sill / equator-basin geography),
# formerly hand-duplicated 55/70 literals -- a drift hazard, now consolidated (values unchanged, byte-identical).
# A few pure DIAGNOSTIC scripts still inline the ramp (flagged in LEGACY_SWEEP_FINDINGS.md, left as-is).
SST_WET_HI = 55.0
SST_DRY = 70.0

__all__ = [n for n in dir() if not n.startswith("_")]
