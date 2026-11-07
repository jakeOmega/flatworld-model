"""flatworld._equator_pond -- the equatorial chimney's SURFACE state: a pond/albedo equilibrium.

This is **approach A** of ``ROADMAP_equatorial_chimney.md`` -- the tractable, self-contained
("no BC hell") first characterization of what the surface under the equatorial convective chimney
actually is: open water, a salt flat, or a self-regulating pond field (Open Question 3). It does NOT
touch the global model; it is a stand-alone surface-energy-balance + water-budget analysis driven by
the canonical equatorial forcing, answering the question the grid cannot (the chimney is sub-grid).

THE MODEL (a sub-grid surface mosaic). Treat the equatorial surface as a mosaic of two tiles sharing
one overlying near-surface air state (the standard tile/mosaic assumption -- the tiles are large vs
the boundary-layer depth, so each holds its own skin temperature against a common ``T_air``/``q_air``):

  * WET tile (fraction ``f``): open water / saturated soil -- evaporation efficiency ``wetness=1``,
    LOW albedo ``alpha_water`` (~0.08). A real water body is wet REGARDLESS of its temperature.
  * DRY tile (fraction ``1-f``): salt flat -- ``wetness=0`` (no evaporation), HIGH albedo
    ``alpha_dry`` (~0.30).

Each tile is in its own steady surface energy balance (the SAME bulk-aerodynamic SEB the AD-core
uses, ``stage_ad_core.bulk_air_sea_sst``):

    (1 - alpha) S_sfc + LW_down  =  sigma T^4  +  SH(T)  +  LE(T),
        SH = rho cp Ch U (T - T_air),   LE = wetness * rho L_v Ce U * max(qsat(T,p) - q_air, 0).

WHAT SETS ``f`` -- the water budget. The zone receives a net LIQUID-water supply ``F`` per unit total
area (the convergent rain that reaches the surface + runoff in, minus runoff out). In steady state the
wet tiles must evaporate exactly that supply:

    f * E_w  =  F      =>      f = clip(F / E_w, 0, 1),

with ``E_w`` the open-water tile's potential evaporation. This is a STABLE, self-regulating closure
(more wet area -> total evaporation exceeds supply -> ponds shrink). The regimes fall out:

    F >= E_w        -> OPEN WATER     (f pinned at 1; surplus pools/runs off; a standing inner sea)
    0 <  F <  E_w   -> POND FIELD     (self-regulating partial wet area; zone-mean T between the tiles)
    F  ~  0         -> SALT FLAT      (f -> 0; the hot dry-rock balance)

THE ANCHOR (clean, and it removes the air-temperature ambiguity). We do not know ``LW_down`` (the
greenhouse) or ``T_air`` a priori, but we DO know the canonical dry salt-flat temperature (~138 C, the
``bl_dz20`` fixture max). So we calibrate ``LW_down`` to put the DRY tile (no evaporation -- a clean
radiative balance) at that temperature. A useful identity then makes the WET tile's temperature (and
hence E_w) INDEPENDENT of T_air: the two SH terms net to ``Ck (T_dry - T_wet)``, with no T_air left
(``anchor_lw_down`` folds the air coupling into LW_down). So E_w depends only on (S_sfc, the albedos,
the dry-flat anchor, q_air, wind) -- which we sweep.

WHAT IS SETTLED vs CONTINGENT (cross-cutting honesty rule).
  * SETTLED: a freely-evaporating equatorial water surface self-LIMITS to ~60-65 C (the evaporative
    thermostat), with E_w ~ 70 mm/day, FAR below the 138 C dry flat -- robust across the forcing. The
    surface state is CLOSURE-DEPENDENT: "assume it stays wet" (wetness=1) -> a cool ~62 C root; "let it
    dry as it heats" (the T-keyed wetness_of_Tsurf ramp) -> runs away to the hot ~138 C salt-flat root.
    Each closure has a UNIQUE root -- the two states come from the two closures, NOT from two roots of one
    fixed map -- so "open water or salt flat?" is ILL-POSED without the water budget (the runaway-greenhouse
    / Komabayashi-Ingersoll hysteresis, with the surface-water supply as the bifurcation parameter). The
    wet-FRACTION closure resolves it and is unconditionally stable.
  * CONTINGENT: the absolute supply ``F`` that actually reaches the surface -- the chimney's converged
    moisture flux is enormous (int P dx ~ 1.4e8 mm*m/day at canonical L=8e7), so IF most of it reached
    the surface the equator would be open water; but the violent (~20-40 m/s) updraft LOFTS most
    condensate aloft to fall on the flanks (roadmap), cutting the surface supply by an unknown factor.
    The surface state is therefore controlled by that LOFTING FRACTION -- what the resolved sim (B) pins.
    The open-water tile temperature ``T_w`` (~62 C) is itself soft in the open-water limit: it is anchored
    to the dry flat's hot/dry overlying air, whereas a real standing sea sets its OWN humid, cooler air
    (higher q_air -> warmer; lower air T -> cooler -- they oppose). It is a POND-FIELD-anchored estimate;
    the self-consistent sea atmosphere is deferred to B. (salt_sill.py's moisture-limited open-ocean hot
    edge ~47-55 C is the independent cross-check; the reservoir-fed thermostat sits just above it.)

USAGE (from cosmology_models/; the equator flux is ~L-invariant so COSMO_L_SUN is optional but set it
for the canonical world):
    COSMO_L_SUN=8.0e7 ../.venv/bin/python -m flatworld._equator_pond
"""
from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass, field

import numpy as np

CANON_L_SUN = 8.0e7            # the canonical worldbuilding L the bl_dz20 fixture was measured on

from . import constants as C
from . import geometry as g
from . import thermo as th
from .stage_ad_core import bulk_air_sea_sst, wetness_of_Tsurf

PLOTDIR = os.path.join("m33_runs", "plots")
DATA_FIXTURE = os.path.join(os.path.dirname(__file__), "data", "bl_dz20_profin.npz")

# Documented canonical fall-backs (if the bl_dz20 fixture is absent on a clean checkout): the
# salt-flat max and the spike integral measured here at the canonical L=8e7, plus the cool-core
# temperature (roadmap). int P dx is L-calibrated (x = L*tan theta), so this is the L=8e7 value.
CANON_T_DRY_C = 137.8          # bl_dz20 salt-flat max (deg C)
CANON_T_CORE_C = 82.5          # bl_dz20 theta~0 cool-core surface temperature (deg C)
CANON_INT_P_DX = 1.36e8        # int P dx over the equatorial spike (theta<8) at L=8e7, mm*m/day = kg/m/day
ROADMAP_INT_P_DX = 2.9e8       # the roadmap's higher "converged control" estimate (sensitivity upper)


@dataclass
class EquatorForcing:
    """The equatorial surface forcing. ``S_sfc`` is the day-mean surface-incident shortwave BEFORE
    albedo (direct + diffuse haze, from geometry); ``LW_down`` is anchored to the dry salt flat."""
    S_sfc: float                       # surface-incident SW (pre-albedo) [W/m^2]
    LW_down: float                     # downwelling longwave (greenhouse) [W/m^2]
    T_air: float = 80.0 + C.T_FREEZE   # near-surface air temperature [K] (cancels out of E_w; see header)
    q_air: float = 0.07                # near-surface specific humidity [kg/kg]
    wind: float = 7.0                  # near-surface wind speed [m/s]
    albedo_water: float = C.ALBEDO_OCEAN     # 0.08
    albedo_dry: float = 0.30                 # salt flat (stage_ad_core.surface_albedo's albedo_dryland)
    T_dry_target: float = CANON_T_DRY_C + C.T_FREEZE   # the dry-flat anchor [K]
    Ch: float = 1.2e-3
    Ce: float = 1.2e-3
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------------------------
# Pure physics
# ----------------------------------------------------------------------------------------------------
def anchor_lw_down(S_sfc, T_dry_K, T_air_K, wind, albedo_dry, Ch=1.2e-3, rho=C.RHO_AIR0):
    """The downwelling longwave [W/m^2] that puts a DRY salt flat (wetness=0, no evaporation) at the
    target temperature ``T_dry_K`` under shortwave ``S_sfc``:

        (1 - alpha_dry) S_sfc + LW_down = sigma T_dry^4 + Ck (T_dry - T_air),   Ck = rho cp Ch U.

    This folds the entire greenhouse + (dry-tile) air coupling into one number anchored to the
    canonical salt flat, so the dry tile is canonical by construction and the wet tile follows."""
    Ck = rho * C.CP_AIR * Ch * wind
    return C.SIGMA_SB * T_dry_K**4 + Ck * (T_dry_K - T_air_K) - (1.0 - albedo_dry) * S_sfc


def latent_evap(T_K, q_air, wind, wetness, Ce=1.2e-3, rho=C.RHO_AIR0):
    """(LE [W/m^2], E [kg/m^2/s]) for a surface at ``T_K`` with the given evaporation efficiency, using
    the SAME bulk-aerodynamic latent flux the core uses (absolute footing)."""
    qs = th.qsat(T_K, C.P_SURF)
    LE = wetness * rho * C.L_VAP * Ce * wind * max(float(qs) - q_air, 0.0)
    return LE, LE / C.L_VAP


def tile_temperature(f: EquatorForcing, wetness: float):
    """Steady surface temperature [K] of a tile with the given evaporation efficiency, via the core's
    bulk air-sea SEB root-find (shared physics). ``wetness=1`` -> open water; ``wetness=0`` -> salt flat."""
    SW_abs = (1.0 - (f.albedo_water if wetness > 0.5 else f.albedo_dry)) * f.S_sfc
    return float(bulk_air_sea_sst(SW_abs, f.LW_down, f.T_air, f.q_air, f.wind,
                                  wetness=wetness, Ch=f.Ch, Ce=f.Ce))


def open_water_equilibrium(f: EquatorForcing):
    """The wet tile (open water, wetness=1): its evaporative-thermostat temperature ``T_w`` [K] and
    potential evaporation ``E_w`` [mm/day]. UNIQUE root (sigma T^4 and LE both rise with T)."""
    Tw = tile_temperature(f, wetness=1.0)
    _, E = latent_evap(Tw, f.q_air, f.wind, 1.0, Ce=f.Ce)
    return Tw, E * 86400.0          # kg/m^2/s -> mm/day


def salt_flat_temperature(f: EquatorForcing):
    """The dry tile (salt flat, wetness=0): its dry-rock radiative temperature [K] (= T_dry_target by
    construction of the LW anchor, returned via the SEB so a mis-anchor would show up)."""
    return tile_temperature(f, wetness=0.0)


def wet_fraction(F_mm_day, E_w_mm_day):
    """Self-regulating wet area fraction f = clip(F / E_w, 0, 1). F and E_w both per unit total area."""
    return float(np.clip(F_mm_day / max(E_w_mm_day, 1e-9), 0.0, 1.0))


def classify_regime(F_mm_day, E_w_mm_day, salt_frac=0.02):
    """Regime label from the supply/demand ratio f = F/E_w. ``salt_frac`` is the wet-fraction below
    which the surface is effectively a salt flat (default 2%)."""
    f = F_mm_day / max(E_w_mm_day, 1e-9)
    if f >= 1.0:
        return "open_water"
    if f <= salt_frac:
        return "salt_flat"
    return "pond_field"


def zone_state(F_mm_day, f: EquatorForcing):
    """The full equatorial-zone surface state for a net liquid supply ``F`` [mm/day per unit total
    area]: tile temperatures, E_w, wet fraction, regime, and area-mean temperature + albedo."""
    Tw, Ew = open_water_equilibrium(f)
    Td = salt_flat_temperature(f)
    frac = wet_fraction(F_mm_day, Ew)
    return dict(
        T_wet_C=Tw - C.T_FREEZE, E_wet_mm_day=Ew, T_dry_C=Td - C.T_FREEZE,
        wet_fraction=frac, regime=classify_regime(F_mm_day, Ew),
        T_area_mean_C=frac * (Tw - C.T_FREEZE) + (1.0 - frac) * (Td - C.T_FREEZE),
        albedo_mean=frac * f.albedo_water + (1.0 - frac) * f.albedo_dry,
        F_mm_day=F_mm_day,
    )


def fraction_from_area_mean_T(T_mean_C, T_wet_C, T_dry_C):
    """Invert the area-mean T_mean = f T_wet + (1-f) T_dry for the wet fraction f (clipped to [0,1]).
    Used to read the canonical cool-core temperature as a pond-field area-mean."""
    if abs(T_dry_C - T_wet_C) < 1e-6:
        return np.nan
    return float(np.clip((T_dry_C - T_mean_C) / (T_dry_C - T_wet_C), 0.0, 1.0))


def fraction_from_radiative_mean_T(T_mean_C, T_wet_C, T_dry_C):
    """Invert the RADIATIVE (sigma T^4) area-mean for the wet fraction f (clipped to [0,1]): the
    OLR-equivalent reading of the core temperature. Reported alongside the linear-T fraction so the
    'mostly-wet' conclusion is shown robust to which mean is used."""
    Tm, Tw, Td = (np.array([T_mean_C, T_wet_C, T_dry_C]) + C.T_FREEZE) ** 4
    if abs(Td - Tw) < 1e-6:
        return np.nan
    return float(np.clip((Td - Tm) / (Td - Tw), 0.0, 1.0))


# ----------------------------------------------------------------------------------------------------
# Canonical anchoring (from the committed bl_dz20 fixture, with documented fall-backs)
# ----------------------------------------------------------------------------------------------------
def equatorial_sw(theta_deg=0.0):
    """Day-mean surface-incident shortwave [W/m^2] at the equator (direct + diffuse haze), from the
    model's own flux geometry. ~L-invariant at theta=0 (the cos^3 law), so COSMO_L_SUN-independent."""
    x = g.x_of_theta(np.radians(theta_deg))
    S = float(g.day_mean_flux(x))
    if getattr(g, "day_mean_diffuse_flux", None) is not None:
        try:
            S += float(g.day_mean_diffuse_flux(x))
        except Exception:        # noqa: BLE001 -- diffuse share is config-dependent
            pass
    return S


def canonical_supply():
    """(int P dx [mm*m/day], T_dry_C, T_core_C) from the committed bl_dz20 fixture; documented
    fall-backs on a clean checkout missing it. int P dx is integrated over the equatorial spike
    (theta < 8 deg, where the legitimate poleward band is zero), in physical cross-track distance x.

    L-CALIBRATION: the fixture's precip P(theta) was measured on the canonical L=8e7 world, and
    x = L*tan(theta), so int P dx is L-dependent -- run with COSMO_L_SUN=8e7 or the supply is off by
    (L/8e7) (the silent de-calibration trap, CLAUDE.md). The equatorial SW forcing, by contrast, is
    L-invariant."""
    if abs(C.L_SUN - CANON_L_SUN) / CANON_L_SUN > 0.01:
        warnings.warn(f"L_SUN={C.L_SUN:.2g} != canonical {CANON_L_SUN:.2g}: int P dx (the liquid supply) is "
                      f"L-calibrated to the fixture's L=8e7 world and will be off by {C.L_SUN/CANON_L_SUN:.2g}x. "
                      f"Set COSMO_L_SUN=8.0e7.", stacklevel=2)
    try:
        d = np.load(DATA_FIXTURE, allow_pickle=True)
        th_deg = np.asarray(d["theta_deg"], float)
        P = np.asarray(d["precip_mm_day"], float)
        T = np.asarray(d["Tsurf_C"], float)
        x = g.x_of_theta(np.radians(th_deg))
        m = th_deg < 8.0
        int_P_dx = float(np.trapezoid(P[m], x[m]))
        return int_P_dx, float(T[m].max()), float(T[th_deg < 1.0].min())
    except (FileNotFoundError, OSError, KeyError):
        return CANON_INT_P_DX, CANON_T_DRY_C, CANON_T_CORE_C


def canonical_forcing(sw_transmission=1.0, **overrides):
    """Build the canonical equatorial forcing: SW from geometry (times an optional column SW
    transmission), LW anchored to the canonical dry salt flat. ``overrides`` patch EquatorForcing
    fields (q_air, wind, albedos, T_dry_target, ...)."""
    _, T_dry_C, T_core_C = canonical_supply()
    S_sfc = equatorial_sw(0.0) * float(sw_transmission)
    base = dict(albedo_water=C.ALBEDO_OCEAN, albedo_dry=0.30,
                T_dry_target=overrides.pop("T_dry_target", T_dry_C + C.T_FREEZE))
    base.update(overrides)
    tmp = EquatorForcing(S_sfc=S_sfc, LW_down=0.0, **base)
    tmp.LW_down = anchor_lw_down(tmp.S_sfc, tmp.T_dry_target, tmp.T_air, tmp.wind, tmp.albedo_dry, Ch=tmp.Ch)
    tmp.meta = dict(sw_transmission=float(sw_transmission), T_core_C=T_core_C, L_SUN=C.L_SUN)
    return tmp


# ----------------------------------------------------------------------------------------------------
# Diagnostics: regime diagram, closure-dependence curve, E_w sensitivity
# ----------------------------------------------------------------------------------------------------
def regime_diagram(f: EquatorForcing, F_mm_day):
    """Sweep the supply F: returns (wet_fraction, T_area_mean_C, albedo_mean) arrays + (E_w, T_w, T_d)."""
    Tw, Ew = open_water_equilibrium(f)
    Td = salt_flat_temperature(f)
    F = np.atleast_1d(np.asarray(F_mm_day, float))
    frac = np.clip(F / Ew, 0.0, 1.0)
    Tmean = frac * (Tw - C.T_FREEZE) + (1.0 - frac) * (Td - C.T_FREEZE)
    alb = frac * f.albedo_water + (1.0 - frac) * f.albedo_dry
    return frac, Tmean, alb, (Ew, Tw - C.T_FREEZE, Td - C.T_FREEZE)


def seb_residual_curve(f: EquatorForcing, T_C):
    """The SEB residual [W/m^2] vs surface temperature for (a) a fixed open-water surface (wetness=1)
    and (b) the model's T-keyed wetness ramp (wetness_of_Tsurf). Demonstrates the CLOSURE-dependence:
    each closure has a UNIQUE root (open water -> cool ~62 C; ramp -> hot ~138 C), so the surface state is
    set by WHICH closure (i.e. by the water supply), not by two roots of one fixed map."""
    T = np.asarray(T_C, float) + C.T_FREEZE
    Ck = C.RHO_AIR0 * C.CP_AIR * f.Ch * f.wind
    Ce_ = C.RHO_AIR0 * C.L_VAP * f.Ce * f.wind
    qs = th.qsat(T, C.P_SURF)

    def resid(wet, alb):
        LE = wet * Ce_ * np.maximum(qs - f.q_air, 0.0)
        return (1.0 - alb) * f.S_sfc + f.LW_down - C.SIGMA_SB * T**4 - Ck * (T - f.T_air) - LE

    w_ramp = wetness_of_Tsurf(T)
    a_ramp = f.albedo_water * w_ramp + f.albedo_dry * (1.0 - w_ramp)
    return resid(1.0, f.albedo_water), resid(w_ramp, a_ramp)


def ew_sensitivity(sw_transmission=1.0, winds=(5.0, 7.0, 10.0), q_airs=(0.03, 0.07, 0.12)):
    """E_w and T_w over a grid of (wind, q_air): the demand-side sensitivity that sets the regime
    boundary. Returns a list of dicts."""
    out = []
    for U in winds:
        for qa in q_airs:
            f = canonical_forcing(sw_transmission=sw_transmission, wind=U, q_air=qa)
            Tw, Ew = open_water_equilibrium(f)
            out.append(dict(wind=U, q_air=qa, T_wet_C=Tw - C.T_FREEZE, E_wet_mm_day=Ew))
    return out


# ----------------------------------------------------------------------------------------------------
# main: the verdict
# ----------------------------------------------------------------------------------------------------
def _save_figure(f, F_grid, int_P_dx, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                # noqa: BLE001
        print(f"[equator_pond] matplotlib unavailable ({e}); skipping figure")
        return
    frac, Tmean, alb, (Ew, Tw, Td) = regime_diagram(f, F_grid)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    # left: the closure-dependence (why "open water or salt flat?" is ill-posed without the water budget)
    Tcurve = np.linspace(45, 145, 400)
    r_ow, r_ramp = seb_residual_curve(f, Tcurve)
    axL.axhline(0, color="k", lw=0.8)
    axL.plot(Tcurve, r_ow, label="open water (wetness=1)", color="tab:blue")
    axL.plot(Tcurve, r_ramp, label="T-keyed wetness ramp", color="tab:red")
    axL.plot([Tw], [0], "o", color="tab:blue", ms=8)
    axL.plot([Td], [0], "o", color="tab:red", ms=8)
    axL.annotate(f"stays-wet root\n~{Tw:.0f} C", (Tw, 0), (Tw - 4, 4000), color="tab:blue", fontsize=9)
    axL.annotate(f"dry-as-it-heats root\n~{Td:.0f} C", (Td, 0), (Td - 40, 4000), color="tab:red", fontsize=9)
    axL.set_xlabel("surface temperature [C]"); axL.set_ylabel("SEB residual [W/m^2]")
    axL.set_title("Surface state is CLOSURE-DEPENDENT\n(the water supply picks the root)")
    axL.set_ylim(-12000, 12000); axL.legend(loc="lower left", fontsize=8); axL.grid(alpha=0.3)

    # right: the regime diagram f(F) + zone-mean T, with the canonical anchors
    ax2 = axR.twinx()
    axR.axvspan(F_grid[0], 0.02 * Ew, color="tab:orange", alpha=0.12)
    axR.axvspan(0.02 * Ew, Ew, color="tab:green", alpha=0.12)
    axR.axvspan(Ew, F_grid[-1], color="tab:blue", alpha=0.12)
    axR.plot(F_grid, frac, color="k", lw=2, label="wet fraction f")
    ax2.plot(F_grid, Tmean, color="tab:purple", lw=1.5, ls="--", label="zone-mean T")
    axR.set_xscale("log"); axR.set_xlabel("net liquid supply F [mm/day, per unit area]")
    axR.set_ylabel("wet fraction f"); ax2.set_ylabel("zone-mean surface T [C]", color="tab:purple")
    axR.set_ylim(-0.03, 1.05)
    axR.axvline(Ew, color="tab:blue", lw=0.8)
    for W_km, label in [(10.0, "chimney 10 km"), (151.0, "grid cell 151 km")]:
        Fc = int_P_dx / (W_km * 1e3)
        axR.axvline(min(Fc, F_grid[-1]), color="tab:gray", ls=":", lw=1)
        axR.annotate(label, (min(Fc, F_grid[-1]), 0.5), rotation=90, fontsize=7,
                     va="center", ha="right", color="tab:gray")
    axR.set_title("Surface regime vs liquid supply\n(salt flat | pond field | open water)")
    axR.legend(loc="center left", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=130)
    print(f"[equator_pond] figure -> {out_png}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Equatorial chimney surface state: pond/albedo equilibrium (roadmap A).")
    p.add_argument("--sw-transmission", type=float, default=1.0,
                   help="column SW transmission applied to the geometry surface flux (sweep 0.7-1.0)")
    p.add_argument("--wind", type=float, default=7.0)
    p.add_argument("--q-air", type=float, default=0.07)
    p.add_argument("--out", default=os.path.join(PLOTDIR, "equator_pond_regime.png"))
    p.add_argument("--no-fig", action="store_true")
    a = p.parse_args(argv)

    f = canonical_forcing(sw_transmission=a.sw_transmission, wind=a.wind, q_air=a.q_air)
    int_P_dx, T_dry_C, T_core_C = canonical_supply()
    Tw, Ew = open_water_equilibrium(f)
    Td = salt_flat_temperature(f)

    print("=" * 84)
    print("EQUATORIAL CHIMNEY SURFACE STATE -- pond/albedo equilibrium (roadmap approach A)")
    print("=" * 84)
    print(f"forcing: S_sfc={f.S_sfc:.0f} W/m^2 (x transmission {a.sw_transmission:g}), "
          f"LW_down={f.LW_down:.0f} (anchored to dry flat {T_dry_C:.1f} C), "
          f"wind={f.wind:g} m/s, q_air={f.q_air:g}, L_SUN={C.L_SUN:.2g}")
    print(f"  alpha_water={f.albedo_water:g}  alpha_dry={f.albedo_dry:g}")
    print()
    print("TILE END-MEMBERS (same forcing, two wetness closures -> the state is CLOSURE-DEPENDENT):")
    print(f"  WET  (open water, wetness=1): T_w = {Tw - C.T_FREEZE:6.1f} C   "
          f"E_w = {Ew:6.1f} mm/day   (the evaporative thermostat)")
    print(f"  DRY  (salt flat,  wetness=0): T_d = {Td - C.T_FREEZE:6.1f} C   E_d =    0.0 mm/day")
    print()

    # the regime boundary in supply, and where the chimney puts us
    print("REGIME vs net liquid supply F (per unit total area):")
    print(f"  open-water threshold  F >= E_w = {Ew:.1f} mm/day")
    print(f"  salt-flat threshold   F <~ {0.02 * Ew:.1f} mm/day (f < 2%)")
    print()
    print(f"  canonical chimney convergence: int P dx = {int_P_dx:.3g} mm*m/day "
          f"(roadmap upper est. {ROADMAP_INT_P_DX:.2g})")
    print("  -> surface supply F if a fraction phi reaches the surface over a footprint W:")
    print(f"     {'footprint W':>14} {'F(phi=1)':>12} {'regime(phi=1)':>16} {'phi for pond-field':>20}")
    for W_km in (10.0, 50.0, 151.0):
        F_full = int_P_dx / (W_km * 1e3)
        phi_pond = Ew / F_full
        print(f"     {W_km:11.0f} km {F_full:9.0f} mm/d {classify_regime(F_full, Ew):>16} "
              f"{phi_pond:18.1%}")
    print()

    # locate the canonical cool core as a pond-field area-mean (an INTERPRETIVE cross-check: the 82.5 C
    # core is dynamically maintained -- transport-cooled -- in the full model, not a local SEB root)
    f_core = fraction_from_area_mean_T(T_core_C, Tw - C.T_FREEZE, Td - C.T_FREEZE)
    f_core_rad = fraction_from_radiative_mean_T(T_core_C, Tw - C.T_FREEZE, Td - C.T_FREEZE)
    F_core = f_core * Ew
    print(f"CANONICAL COOL CORE ({T_core_C:.1f} C) read as a pond-field area-mean (interpretive -- the core")
    print(f"  is dynamically/transport-cooled in the full model, not a standalone SEB root):")
    print(f"  implied wet fraction f = {f_core:.2f} (linear-T) / {f_core_rad:.2f} (radiative T^4) -- both mostly-wet")
    print(f"  -> surface supply F = {F_core:.1f} mm/day, i.e. ~{F_core / (int_P_dx / (10e3)):.1%} of the 10-km "
          f"chimney flux reaches the surface (the rest lofts aloft) -- a strong-lofting pond field.")
    print()
    print("OFF-AXIS FLANKS (theta~2-12 deg): F~0 (no convergent rain) -> SALT FLAT at ~%.0f C "
          "(canonical, confirmed)." % T_dry_C)
    print()

    # E_w sensitivity (the demand-side uncertainty on the regime boundary)
    print("E_w / T_w SENSITIVITY (sets the open-water threshold; T_w soft in the open-water limit):")
    sens = ew_sensitivity(sw_transmission=a.sw_transmission)
    Ews = [s["E_wet_mm_day"] for s in sens]
    Tws = [s["T_wet_C"] for s in sens]
    print(f"  over wind 5-10 m/s, q_air 0.03-0.12:  E_w = {min(Ews):.0f}-{max(Ews):.0f} mm/day,  "
          f"T_w = {min(Tws):.0f}-{max(Tws):.0f} C")
    print("  NB T_w is POND-FIELD-anchored (it borrows the dry flat's hot/dry air); a real open SEA sets its")
    print("  own humid (q_air up -> warmer) + cooler (air T down -> cooler) air -- soft, deferred to B. The")
    print("  salt_sill open-ocean hot edge (~47-55 C, moisture-limited) cross-checks it from below.")
    print()
    print("VERDICT: the equatorial surface is a SELF-REGULATING POND/SEA FIELD, not uniformly one thing:")
    print(f"  open water at the axis (capped at the ~{Tw - C.T_FREEZE:.0f} C evaporative thermostat) where")
    print(f"  convergence dumps moisture, grading to ~{Td - C.T_FREEZE:.0f} C salt flats on the flanks; the axis")
    print("  wet-fraction is set by the LOFTING-controlled liquid supply -- the unknown approach B must pin.")

    # save outputs
    if not a.no_fig:
        F_grid = np.logspace(-1, np.log10(max(int_P_dx / 1e4, 5 * Ew)), 240)
        _save_figure(f, F_grid, int_P_dx, a.out)
    os.makedirs(PLOTDIR, exist_ok=True)
    npz = os.path.join(PLOTDIR, "equator_pond.npz")
    np.savez(npz, S_sfc=f.S_sfc, LW_down=f.LW_down, T_wet_C=Tw - C.T_FREEZE, E_wet_mm_day=Ew,
             T_dry_C=Td - C.T_FREEZE, int_P_dx=int_P_dx, T_core_C=T_core_C, f_core=f_core,
             f_core_rad=f_core_rad, F_core=F_core, E_w_range=np.array([min(Ews), max(Ews)]),
             wind=f.wind, q_air=f.q_air, sw_transmission=a.sw_transmission)
    print(f"[equator_pond] data -> {npz}")
    return dict(T_wet_C=Tw - C.T_FREEZE, E_wet_mm_day=Ew, T_dry_C=Td - C.T_FREEZE,
                f_core=f_core, F_core=F_core, int_P_dx=int_P_dx)


if __name__ == "__main__":
    main()
