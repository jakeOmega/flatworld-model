"""
flatworld.bl_forcing -- the boundary-layer geopotential forcing Phi(T, terrain) for the (x, e) maps.

Factored out of ``stage_c_map`` so the #111 momentum solve (and the diurnal #113 / Walker #112 siblings)
share ONE forcing build. Phi is the thermal-low + terrain field the boundary-layer wind flows down:

    Phi = -beta_T * (T_v - Tref(row)) + g_terrain * G * h          [m^2/s^2]

  * warm surface -> LOW Phi (thermal low; air converges in -> ascent -> rain);
  * high terrain -> HIGH Phi (air rides over high ground);
  * Tref is the per-row (zonal-mean) reference, so Phi is the ALONG-TRACK anomaly that breaks the
    e-symmetry (a uniform cross-track warming does not drive a wind);
  * smoothed over the frictional/advective adjustment length ``phi_smooth_km`` -- the boundary layer
    integrates heating over that fetch and must not feel grid-scale coastline jumps that would spike
    the ascent w ~ del^2(Phi). The smoothing BC ("nearest" across the x-walls, "wrap" along e) MUST
    match the solver operators (flatworld.operators: clamped in x, np.roll-periodic in e), or the
    periodic derivative differences a raw land/sea jump at the e-seam (a spurious precip stripe, #82).

VIRTUAL temperature (issue #80): the buoyancy that drives the wind is set by the virtual temperature
T_v = T(1 + q/eps)/(1 + q), not dry T -- moist sea air (q ~ 0.1 at a 55 C surface) is ~5-10 K of
EQUIVALENT extra warmth on top of the 30 K land/sea T contrast, non-negligible. ``geopotential`` takes
T_v directly; ``virtual_T_map`` builds it from the dry T map + the surface-moisture column W_s. When
the moisture column is zero (or omitted), T_v == T and the build is byte-identical to the dry form.
"""
from __future__ import annotations
import numpy as np
import scipy.ndimage as ndi
from . import constants as C
from . import thermo as th

# W_s is a column water path [mm == kg/m^2]; the matching near-surface specific humidity is
# q = W_s / (RHO_AIR0 * H_moist), the inverse of stage_c_map._wsat_column (rho_air * H_moist * qsat).
H_MOIST = 2500.0


def specific_humidity_from_column(W_s, H_moist=H_MOIST):
    """Near-surface specific humidity q [kg/kg] from a column-water path W_s [mm], inverting
    _wsat_column = RHO_AIR0 * H_moist * qsat. Clipped to [0, 0.1]: real near-surface q tops out ~0.04
    at a ~55 C wet surface, so 0.1 is a generous sanity ceiling that still flags a unit error (a much
    larger clip would silently swallow one)."""
    q = np.asarray(W_s, float) / (C.RHO_AIR0 * H_moist)
    return np.clip(q, 0.0, 0.1)


def virtual_T_map(T_map_C, W_s=None, H_moist=H_MOIST):
    """Virtual temperature map in CELSIUS (same units as T_map_C) from the dry day-mean T map and the
    surface-moisture column W_s [mm]. T_v - T is the vapor-buoyancy bonus (always >= 0). With W_s None
    or all-zero, returns T_map_C unchanged (dry-limit byte-identical)."""
    T_map_C = np.asarray(T_map_C, float)
    if W_s is None or not np.any(W_s):
        return T_map_C            # dry limit: return T unchanged (avoid the +-T_FREEZE round-trip error)
    q = specific_humidity_from_column(W_s, H_moist)
    Tv_K = th.virtual_temperature(T_map_C + C.T_FREEZE, q)
    return Tv_K - C.T_FREEZE


def geopotential(Tv_map_C, height, dx, de, beta_T=30.0, g_terrain=0.6, phi_smooth_km=400.0):
    """Boundary-layer geopotential anomaly Phi [m^2/s^2] from a (VIRTUAL) temperature map [C] and the
    terrain height [m]. Pass ``Tv_map_C`` already in virtual temperature (see ``virtual_T_map``); pass
    dry T for the legacy dry build. Tref is the per-row zonal mean. Smoothed over ``phi_smooth_km`` with
    the wall-clamped / e-periodic BC the operators use.

    This reproduces ``StageCMap._geopotential`` exactly when fed dry T and the same (beta_T, g_terrain,
    phi_smooth_km)."""
    Tv_map_C = np.asarray(Tv_map_C, float)
    Tref = np.mean(Tv_map_C, axis=1, keepdims=True)        # zonal-mean reference per row
    Phi = -beta_T * (Tv_map_C - Tref) + g_terrain * C.G_ACCEL * np.asarray(height, float)
    sig = (max(phi_smooth_km / (dx / 1e3), 1.0), max(phi_smooth_km / (de / 1e3), 1.0))
    return ndi.gaussian_filter(Phi, sigma=sig, mode=("nearest", "wrap"))


def make_phi_forcing(Tv_map_C, height, dx, de, beta_T=30.0, g_terrain=0.6, phi_smooth_km=400.0):
    """Return a callable ``phi_of(t)`` (static by default) giving Phi at model time t. The momentum
    solver calls ``phi_of(t)`` each step, so swapping in a time-varying T(t) build (issue #113 diurnal
    sea/land breezes) becomes a forcing swap, not a solver rewrite. The default ignores t and returns
    the static Phi built once here."""
    Phi = geopotential(Tv_map_C, height, dx, de, beta_T, g_terrain, phi_smooth_km)

    def phi_of(t=0.0):
        return Phi

    return phi_of
