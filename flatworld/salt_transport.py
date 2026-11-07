"""
flatworld.salt_transport -- the wind-keyed salt-aerosol SOURCE/TRANSPORT/SETTLING scheme (GitHub #237,
the source/transport checkbox). Turns the salt-CCN droplet-number field ``N_d(theta)`` from a PRESCRIBED
scalar (the #243 AD-core coupling) into an EMERGENT field built from the resolved near-surface wind.

WHAT IT MODELS (the physical chain the issue specifies), 1-D and steady across-track:

  1. EMISSION off the equatorial salt flats, keyed on the resolved near-surface (10 m) wind. A bare
     evaporite pan is erodible where the friction velocity u* exceeds the halite saltation threshold
     u*t (Shao & Lu 2000, cohesion included). The horizontal saltation flux is the Owen (1964) u*^3 law
     Q = (rho_a/g) u*^3 (1 - (u*t/u*)^2); the vertical (suspension) dust flux is F = alpha * Q
     (sandblasting efficiency alpha). Emission is ZERO wherever u* <= u*t or the cell is not a flat.
  2. SIZE BINNING of the suspended dust into a lognormal emission mass distribution -- so a grain's fate
     follows its size. The number emitted per bin is f_i*F/m_i (m_i the grain mass ~ d_i^3), which weights
     NUMBER heavily toward the fine/accumulation mode (the CCN-relevant tail).
  3. CELL TRANSPORT. The single overturning cell sweeps the surface dust equatorward to the inner-sea
     updraft, lofts it (loft efficiency eta_i = w_loft/(w_loft+v_i): fine grains loft, coarse fall back),
     and returns it poleward ALOFT in a layer (u_aloft, H_aloft). Along that poleward return each bin
     settles out with gravitational deposition, giving the steady advection-settling profile
     C_i(x) = C_i0 * exp(-(x - x_loft)/reach_i), reach_i = u_aloft*H_aloft/v_settle(d_i). COARSE
     (short reach) re-deposits near the line (the sun-line dune province); the FINE accumulation mode
     (reach ~ the band distance and beyond) survives to the rain band as CCN.
  4. N_d(theta) = N_ref + (1/1e6) * sum_i C_i(x)  [cm^-3], the emergent cloud-droplet-number field the
     #243 AD-core limbs (Twomey r_eff; warm-rain autoconversion (N_d/N_ref)^beta; decoupled haze) consume.

CONFIDENCE. The DIRECTION and REGIME are firm (the wind is supercritical; coarse stays, fine reaches the
band; the band crosses from clean-maritime toward continental loading). Every ABSOLUTE MAGNITUDE is
NAMED-WALL soft -- the wall is aerosol microphysics + crust-state erodibility (alpha, the emission PSD,
the loft/settling layer geometry are each 1-2 orders uncertain). These soft knobs are SWEPT and reported,
NEVER tuned to a target N_d (cf. the repo's free-parameter discipline). This module is a DIAGNOSTIC,
ONE-WAY producer of N_d: the salt term is second-order on the circulation (the issue's scope note), so the
field is built from the resolved wind and fed forward; it does not feed back on the dynamics.

L-ROBUSTNESS (the project's standing box/L discipline). Band loading is NOT exponentially L-sensitive even
though the transport path ``x_band = L*tanθ`` grows with L while the per-bin ``reach`` is L-independent: the
flats SOURCE width also scales with L, so a longer transport (more settling) is offset by a wider/stronger
source. The two largely cancel -- band ``N_d`` stays continental at both L=2e7 (~21k) and the canonical
L=8e7 (~29k for a fixed synthetic wind) -- so the metric does not saturate against the L choice.

RELATION TO ``../aeolian_salt.py``: that is the stdlib-only back-of-envelope that established the regime
(u*/u*t ~ 1.4-2.0, the three size fates, the ~few-% trap leak). This module is the gridded, production
version on the flatworld theta-grid that produces the actual ``N_d(theta)`` field, reusing the same
standard aeolian/aerosol relations (cross-checked against the BOE in ``tests/test_salt_transport.py``).

Pure numpy, device-free, cheap (a couple of (nx, n_bin) array ops): it runs as a post-process on a
resolved wind, so it adds no cost to the hot step path. Default OFF in the core (``salt_ccn_source`` stays
``'prescribed'``); this module only builds a field when explicitly invoked.

CLI:  python -m flatworld.salt_transport <cascade.npz>   # build N_d(theta) from a persisted run's u_surf
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from . import constants as C
from . import geometry as g
from . import surface_wind as sw

# ----------------------------------------------------------------- physical givens (halite + air)
RHO_HALITE = 2160.0        # kg/m^3 -- halite grain density
MU_AIR = 1.8e-5            # Pa s  -- air dynamic viscosity (weakly T-dependent; constant for this scheme)
KARMAN = 0.4               # von Karman constant


@dataclass
class SaltTransportConfig:
    """Soft, SWEPT knobs of the salt source/transport scheme (NONE tuned to a target N_d). Every value is a
    literature-standard mid estimate; the named wall is aerosol microphysics + crust-state erodibility."""
    # --- emission (the erodible flats + the saltation/dust closure) ---
    flats_theta_hi: float = 25.0     # [deg] poleward edge of the bare salt flats (the salt sill); the
    #                                  erodible source runs from the sun-line to here. Equatorward of the
    #                                  open sea's hot edge the surface is a bare evaporite pan.
    z0_flats: float = 1.0e-3         # [m] aerodynamic roughness of the (rough-crust) salt pan
    d_salt_opt: float = 125.0e-6     # [m] optimal (most-mobile) saltating grain -> sets the bed threshold u*t
    alpha_sandblast: float = 3.0e-5  # [1/m] vertical(dust)/horizontal(saltation) flux ratio F/Q (Owens-Lake
    #                                  /Bodele class ~1e-5..1e-4; the dominant magnitude knob -- SWEPT)
    T_flat_C: float = 125.0          # [degC] day-mean salt-flat surface/air temperature (sets flats air density)
    shao_A_N: float = 0.0123         # Shao & Lu (2000) threshold coefficient
    shao_gamma: float = 3.0e-4       # [N/m] Shao & Lu interparticle-cohesion coefficient (the playa floor)
    # --- emitted suspension size distribution (lognormal by MASS) ---
    n_bin: int = 12                  # number of log-spaced size bins
    d_min: float = 5.0e-8            # [m] finest bin (0.05 um -- accumulation mode)
    d_max: float = 3.0e-5            # [m] coarsest suspended bin (30 um)
    d_mmd: float = 3.0e-6            # [m] mass median diameter of the emitted suspension (~3 um)
    sigma_g: float = 2.0             # geometric standard deviation of the emission lognormal
    # --- cell loft + poleward-aloft return (the transport geometry) ---
    loft_theta_deg: float = 0.0      # [deg] latitude of the inner-sea updraft where surface dust is lofted
    w_loft: float = 0.1              # [m/s] updraft speed competing against settling at the loft point
    u_aloft: float = 5.0             # [m/s] poleward-aloft return-branch wind speed
    H_aloft: float = 4000.0          # [m] depth of the poleward-aloft transport layer
    v_dep_min: float = 5.0e-4        # [m/s] dry-deposition velocity FLOOR -- the turbulent+Brownian removal
    #                                  the accumulation mode feels in ADDITION to gravitational settling (the
    #                                  "Greenfield gap" minimum ~0.1-1 mm/s; also a proxy for wet scavenging
    #                                  along the path). Without it pure-Stokes settling gives the sub-um mode a
    #                                  near-infinite reach and the band would sit at the NEAR-SOURCE loading;
    #                                  this floor is what bounds the accumulation-mode lifetime (it sets the
    #                                  BOE's lumped "surviving fraction"). NAMED-WALL soft (aerosol microphysics).
    # --- CCN reference ---
    n_d_ref: float = 100.0           # [cm^-3] clean-maritime background droplet number (the #243 zero-point)
    # --- the input-wind reduction (lowest model level -> 10 m), used by the core wiring ---
    h_bl: float = 1000.0             # [m] boundary-layer depth for the log-layer reduction
    z1: float | None = None          # [m] lowest model-level height (for the BL-refined reduction; see surface_wind)

    def bin_diameters(self):
        """Geometric bin-center diameters [m] and the lognormal-by-mass emission fraction f_i (sum 1)."""
        edges = np.geomspace(self.d_min, self.d_max, self.n_bin + 1)
        d = np.sqrt(edges[:-1] * edges[1:])                              # geometric centers
        lnsig = np.log(self.sigma_g)
        # lognormal mass-density in ln d; integrate over each bin via the trapezoid of the edges
        def cdf(x):
            from math import erf
            return np.array([0.5 * (1.0 + erf(np.log(xi / self.d_mmd) / (np.sqrt(2.0) * lnsig)))
                             for xi in np.atleast_1d(x)])
        f = cdf(edges[1:]) - cdf(edges[:-1])
        f = f / f.sum()
        return d, f


# ----------------------------------------------------------------- 1. emission physics
def air_density(T_C, p_surf=None):
    """Near-surface air density [kg/m^3] from the ideal gas law at the (hot) flat-surface temperature."""
    p = C.P_SURF if p_surf is None else p_surf
    return p / (C.R_DRY * (np.asarray(T_C, float) + C.T_FREEZE))


def friction_velocity(u10, z0, z_ref=10.0):
    """Neutral log-law friction velocity u* [m/s] from a 10-m wind. CONSERVATIVE over the super-heated
    flats (an unstable BL raises the true u* above this neutral estimate)."""
    return np.asarray(u10, float) * KARMAN / np.log(z_ref / z0)


def threshold_ustar(d, rho_a, A_N=0.0123, gamma=3.0e-4, rho_p=RHO_HALITE):
    """Shao & Lu (2000) fluid-threshold friction velocity u*t [m/s] with interparticle cohesion (the gamma
    term dominates for fine grains and gives the playa its finite, non-zero threshold). d, rho_a scalar/array."""
    d = np.asarray(d, float)
    return np.sqrt(A_N * (rho_p * C.G_ACCEL * d / rho_a + gamma / (rho_a * d)))


def saltation_flux(ustar, ustar_t, rho_a):
    """Owen (1964) horizontal saltation mass flux Q [kg/m/s] (per unit width). Zero below threshold.
    Fully vectorized in ``ustar`` (and broadcasting ``ustar_t``/``rho_a``)."""
    ustar = np.asarray(ustar, float)
    ustar_t = np.asarray(ustar_t, float)
    rho_a = np.asarray(rho_a, float)
    over = ustar > ustar_t
    safe_u = np.where(over, ustar, 1.0)                              # avoid 0/0 in the masked-out cells
    q = (rho_a / C.G_ACCEL) * ustar ** 3 * (1.0 - (ustar_t / safe_u) ** 2)
    return np.where(over, q, 0.0)


def settling_velocity(d, rho_p=RHO_HALITE, mu=MU_AIR):
    """Stokes terminal velocity [m/s] (valid Re<<1, the FINE suspension; coarse grains settle even faster
    than Stokes, so using Stokes only OVER-estimates their reach -> 'coarse stays near the source' is the
    conservative conclusion). Buoyancy dropped (air ~2000x lighter than halite). d in m."""
    d = np.asarray(d, float)
    return rho_p * C.G_ACCEL * d ** 2 / (18.0 * mu)


# ----------------------------------------------------------------- 2-4. the gridded source/transport solve
def build_nd_field(theta_deg, x_m, u10, cfg=None, T_air_C=None, flats_mask=None):
    """Build the emergent salt-CCN droplet-number field ``N_d(theta)`` [cm^-3] from a resolved near-surface
    (10 m) wind. Returns ``(N_d, diag)``: ``N_d`` an (nx,) array, ``diag`` a dict of intermediates
    (per-bin reach, deposition split, lofted number flux, the band/peak loadings) for reporting/tests.

    Parameters
    ----------
    theta_deg : (nx,) cross-track latitude [deg]
    x_m       : (nx,) cross-track distance [m] (= L*tan(theta); the grid the cell settles along)
    u10       : (nx,) resolved 10-m wind speed [m/s] over each column (the EMISSION driver)
    cfg       : SaltTransportConfig (soft knobs); defaults if None
    T_air_C   : optional (nx,) flat-surface air temperature [degC]; default uniform cfg.T_flat_C
    flats_mask: optional (nx,) bool ERODIBLE-SOURCE mask that REPLACES the static geometric
                ``theta <= flats_theta_hi`` pan (None -> the static mask, byte-identical). This is the
                DYNAMIC-source hook (``stage_ad_core.refresh_salt_ccn``): the caller passes the live
                dry-surface mask (wetness below threshold), so the source shrinks/grows with the state.
                NB the collection-to-loft transport geometry is unchanged -- all masked emission is
                collected to the equatorial loft point -- so a mask far from the flats contributes
                loading but not a new transport pathway (fine while the dry surface is the equatorial
                flat, which the wetness physics enforces; documented caveat).

    The field is N_ref everywhere when the wind is everywhere sub-threshold (a clean no-op), so the master
    coupling stays a no-op until the flats actually erode."""
    cfg = cfg or SaltTransportConfig()
    theta_deg = np.asarray(theta_deg, float)
    x = np.asarray(x_m, float)
    u10 = np.asarray(u10, float)
    nx = x.size

    # --- 1. emission over the erodible flats ---
    flats = (theta_deg <= cfg.flats_theta_hi) if flats_mask is None \
        else np.asarray(flats_mask, bool)                            # bare evaporite pan mask (static | live)
    T_C = np.full(nx, cfg.T_flat_C) if T_air_C is None else np.asarray(T_air_C, float)
    rho_a = air_density(T_C)                                          # (nx,) flats air density
    ustar = friction_velocity(u10, cfg.z0_flats)                     # (nx,)
    ustar_t = threshold_ustar(cfg.d_salt_opt, rho_a, A_N=cfg.shao_A_N, gamma=cfg.shao_gamma)  # (nx,) bed threshold
    Q = saltation_flux(ustar, ustar_t, rho_a)                        # (nx,) horizontal saltation flux
    Q = np.where(flats, Q, 0.0)                                      # erode only the flats
    F = cfg.alpha_sandblast * Q                                       # (nx,) vertical (suspended) dust mass flux

    # --- 2. size bins: number emission per bin, per column ---
    d, f_mass = cfg.bin_diameters()                                  # (n_bin,), (n_bin,)
    m_grain = RHO_HALITE * (np.pi / 6.0) * d ** 3                     # (n_bin,) grain mass [kg]
    # number flux density per (column, bin) [#/m^2/s] = f_i * F(x) / m_i
    E_N = (F[:, None] * f_mass[None, :]) / m_grain[None, :]          # (nx, n_bin)

    # --- 3. cell transport: collect emission to the loft point, loft, advect poleward-aloft, settle ---
    dx = np.gradient(x)                                              # cell widths [m] (handles non-uniform x)
    v_set = settling_velocity(d)                                     # (n_bin,) gravitational Stokes fall speed
    v_dep = v_set + cfg.v_dep_min                                    # (n_bin,) TOTAL deposition velocity (gravity +
    #                                                                 turbulent/Brownian dry-dep floor) -> the sink
    #                                                                 that bounds the accumulation-mode lifetime
    eta_loft = cfg.w_loft / (cfg.w_loft + v_set)                      # (n_bin,) loft efficiency (gravity vs updraft)
    # total lofted NUMBER flux per bin [#/m/s] = eta_i * integral over the flats of E_N,i dx
    Phi = eta_loft * np.sum(E_N * dx[:, None], axis=0)               # (n_bin,)
    C0 = Phi / (cfg.u_aloft * cfg.H_aloft)                            # (n_bin,) loft-point concentration [#/m^3]
    reach_dep = cfg.u_aloft * cfg.H_aloft / v_dep                    # (n_bin,) FIELD e-folding (gravity+dry-dep) [m]
    reach_grav = cfg.u_aloft * cfg.H_aloft / v_set                   # (n_bin,) SIZE-SELECTIVE gravitational reach [m]
    x_loft = g.x_of_theta(np.radians(cfg.loft_theta_deg))
    s = np.clip(x[:, None] - x_loft, 0.0, None) / reach_dep[None, :]  # poleward-only; upstream of loft -> 0
    Cx = C0[None, :] * np.exp(-s)                                    # (nx, n_bin) aloft concentration [#/m^3]
    Cx = np.where((x[:, None] >= x_loft), Cx, 0.0)

    # --- 4. assemble N_d(theta) [cm^-3] ---
    C_total = np.sum(Cx, axis=1)                                     # (nx,) all-bin number conc [#/m^3]
    N_d = cfg.n_d_ref + C_total / 1.0e6                              # #/m^3 -> #/cm^3, on the clean background

    diag = _diagnostics(cfg, theta_deg, x, d, f_mass, v_set, eta_loft, reach_grav, Phi, C0, Cx, N_d,
                        ustar, ustar_t, Q, F)
    return N_d, diag


def _diagnostics(cfg, theta_deg, x, d, f_mass, v_set, eta_loft, reach_grav, Phi, C0, Cx, N_d,
                 ustar, ustar_t, Q, F):
    """Assemble the per-bin / fate diagnostics used by reports and tests (no physics, just bookkeeping).

    TWO DISTINCT DELIVERY STORIES, do not conflate them:
     * ``fate`` / ``coarse_to_flats_frac`` / ``fine_to_band_frac`` are the SIZE-SELECTIVE split, on the
       GRAVITATIONAL reach (which sizes go where): coarse re-deposits on the flats (the sun-line dunes), an
       intermediate size leaks over the cool sea, the fine mode would gravitationally reach the band. This is
       the *shape* of the size-fate, independent of the lifetime floor.
     * ``delivered_band_frac`` is the actual fraction of lofted NUMBER that survives to the band in the FIELD
       (on ``reach_dep`` = gravity + the size-independent ``v_dep_min`` dry-deposition floor). It is much
       smaller than ``fine_to_band_frac`` because the floor caps even the fine mode's lifetime (the BOE's
       lumped "surviving fraction") -- this, not the gravity-only split, is what sets ``N_d_band``."""
    x_band = g.x_of_theta(np.radians(_band_theta(theta_deg)))
    x_sill = g.x_of_theta(np.radians(cfg.flats_theta_hi))
    fate = np.where(reach_grav >= x_band, "band",
                    np.where(reach_grav >= x_sill, "leak", "flats"))
    iband = int(np.argmin(np.abs(theta_deg - _band_theta(theta_deg))))
    # number-weighted survival to the band on the FIELD's reach_dep: since C0_i ∝ Phi_i (the lofted number
    # flux), the per-bin band/loft concentration ratio summed over bins IS the number-weighted delivered
    # fraction = sum_i Phi_i*exp(-(x_band-x_loft)/reach_dep_i) / sum_i Phi_i.
    C0_sum = float(np.sum(C0))
    delivered_band_frac = float(np.sum(Cx[iband, :]) / C0_sum) if C0_sum > 0.0 else 0.0
    return dict(
        d_um=d * 1e6, f_mass=f_mass, v_settle_mm_s=v_set * 1e3, eta_loft=eta_loft,
        reach_km=reach_grav / 1e3, Phi_num=Phi, C0=C0, fate=fate,
        ustar=ustar, ustar_t=float(np.atleast_1d(ustar_t).flat[0]) if np.ndim(ustar_t) else float(ustar_t),
        ustar_over_t=ustar / ustar_t, Q=Q, F=F,
        N_d=N_d, N_d_band=float(N_d[iband]), N_d_peak=float(N_d.max()),
        band_theta=_band_theta(theta_deg), x_band_km=x_band / 1e3,
        coarse_to_flats_frac=float(f_mass[reach_grav < x_sill].sum()),    # gravity-only, size-selective
        fine_to_band_frac=float(f_mass[reach_grav >= x_band].sum()),      # gravity-only, size-selective
        delivered_band_frac=delivered_band_frac,                          # FIELD survival (gravity + dry-dep floor)
    )


def _band_theta(theta_deg, default=46.0):
    """The canonical liquid-rain band latitude [deg] used for the band-loading READOUT only (nsub30 ~theta46;
    clipped into the supplied grid range). READOUT-ONLY: the returned ``N_d`` field is full-resolution, so the
    #243 consumer is unaffected by this latitude -- it only sets which θ the ``N_d_band`` summary number is
    read at. The canonical rain-band latitude is itself under remeasure (root `ROADMAP.md` DOC-2); when it
    settles, update this default (or read at the measured band)."""
    lo, hi = float(np.min(theta_deg)), float(np.max(theta_deg))
    return float(np.clip(default, lo, hi))


# ----------------------------------------------------------------- core-facing builder (lowest level -> 10 m)
def nd_field_from_u_surf(theta_deg, x_m, u_surf, cfg=None, T_air_C=None, flats_mask=None):
    """Core-facing wrapper: reduce the model's LOWEST-LEVEL cross-track wind ``u_surf`` [m/s] to a 10-m wind
    (neutral log-layer reduction over the flats roughness) and build N_d. ``u_surf`` may be signed (the
    cross-track branch) -- emission keys on SPEED, so the magnitude is used. Returns ``(N_d, diag)``.
    ``flats_mask`` (optional) is passed through to ``build_nd_field`` (the dynamic-source hook)."""
    cfg = cfg or SaltTransportConfig()
    f10 = sw.u10_factor(np.asarray(cfg.z0_flats, float), h_bl=cfg.h_bl, z1=cfg.z1)
    u10 = np.abs(np.asarray(u_surf, float)) * float(f10)
    return build_nd_field(theta_deg, x_m, u10, cfg=cfg, T_air_C=T_air_C, flats_mask=flats_mask)


# ----------------------------------------------------------------- CLI: build N_d from a persisted cascade .npz
def _report_from_npz(path, cfg=None):
    cfg = cfg or SaltTransportConfig()
    z = np.load(path, allow_pickle=True)
    if "dC" in z:                                                   # a Stage-C cascade cache
        dC = z["dC"].item()
        theta_deg = np.asarray(dC["theta_deg"], float)
        x_m = g.x_of_theta(np.radians(theta_deg))
        u_surf = np.hypot(np.asarray(dC["u"], float), np.asarray(dC.get("v", 0.0), float)).mean(axis=1)
    else:                                                          # an AD-core diagnostics dump
        theta_deg = np.asarray(z["theta_deg"], float)
        x_m = np.asarray(z["x_km"], float) * 1e3 if "x_km" in z else g.x_of_theta(np.radians(theta_deg))
        u_surf = np.abs(np.asarray(z["u_surf"], float))
    N_d, diag = nd_field_from_u_surf(theta_deg, x_m, u_surf, cfg=cfg)
    print("=" * 88)
    print(f"salt-transport N_d(theta) from {path}")
    print("=" * 88)
    print(f"  flats edge theta<= {cfg.flats_theta_hi:.0f} deg, alpha={cfg.alpha_sandblast:.0e}/m, "
          f"u_aloft={cfg.u_aloft:.0f} m/s, H_aloft={cfg.H_aloft:.0f} m, N_ref={cfg.n_d_ref:.0f} cm^-3")
    print(f"  u*/u*t over the flats: max {np.nanmax(diag['ustar_over_t']):.2f}  "
          f"(>1 => erosive); band theta~{diag['band_theta']:.0f} (x~{diag['x_band_km']:.0f} km)")
    print(f"  {'d[um]':>7} {'f_mass':>8} {'v[mm/s]':>9} {'eta':>6} {'reach[km]':>10}  fate")
    for i in range(len(diag["d_um"])):
        print(f"  {diag['d_um'][i]:7.2f} {diag['f_mass'][i]:8.3f} {diag['v_settle_mm_s'][i]:9.3f} "
              f"{diag['eta_loft'][i]:6.3f} {diag['reach_km'][i]:10.0f}  {diag['fate'][i]}")
    print(f"  size-selective (gravity) split: coarse->flats(dunes) {diag['coarse_to_flats_frac']:.2f}, "
          f"fine->band {diag['fine_to_band_frac']:.2f}")
    print(f"  FIELD delivered-to-band number fraction (gravity + dry-dep floor): "
          f"{diag['delivered_band_frac']:.3f}  <- this, not the gravity split, sets N_d_band")
    print(f"  N_d at the band ~ {diag['N_d_band']:8.1f} cm^-3   (peak over the source ~ {diag['N_d_peak']:.1f})")
    print(f"  -> band loading vs clean maritime ({cfg.n_d_ref:.0f}): "
          f"{diag['N_d_band'] / cfg.n_d_ref:.1f}x  "
          f"({'continental-polluted (>=1000)' if diag['N_d_band'] >= 1000 else 'maritime-to-continental'})")
    print("  (direction firm; magnitude named-wall-soft = aerosol microphysics + crust erodibility; NOT tuned)")
    print("=" * 88)
    return N_d, diag


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="build the emergent salt-CCN N_d(theta) field from a persisted run")
    p.add_argument("npz", help="cascade .npz (Stage-C cache with dC, or an AD-core diagnostics dump)")
    p.add_argument("--alpha", type=float, default=None, help="sandblast efficiency alpha [1/m] (sweep knob)")
    p.add_argument("--flats-theta-hi", type=float, default=None, help="poleward edge of the salt flats [deg]")
    a = p.parse_args()
    cfg = SaltTransportConfig()
    if a.alpha is not None:
        cfg.alpha_sandblast = a.alpha
    if a.flats_theta_hi is not None:
        cfg.flats_theta_hi = a.flats_theta_hi
    _report_from_npz(a.npz, cfg=cfg)
