"""
flatworld.geometry -- the flat-plane geometry and the radiative forcing (GIVENS).

This module encodes the setting's fixed inputs and nothing downstream:
  * the cross-track metric  x = L*tan(theta)  (strongly nonlinear in theta),
  * the day-mean flux law   I(theta) = I0 * cos^3(theta) = I0 * L^3/(L^2+x^2)^{3/2},
  * the train-of-suns clock  (v, D, T_day) fixed by the midnight-darkness floor, and
  * the absolute instantaneous and day-mean surface flux, S(theta) [W/m^2], the radiative
    driver used by every stage.

All of it is checked against the cosmology doc by ``verify_against_doc()``.

Coordinates
-----------
``theta``  : angle from the sun-line [rad unless _deg]. theta=0 is the equator (sun-line).
``x``      : cross-track horizontal distance from the sun-line [m] = L*tan(theta). The climate
             gradient ("latitude") direction. GRID IN x (metres), never uniform in theta.
``e``      : along-track horizontal distance [m] (east-west; the suns move this way). Mean climate
             is independent of e; topology breaks the symmetry (handled in stage_c_map).
``z``      : height above the plane [m].
"""
from __future__ import annotations
import functools
import os
import warnings

import numpy as np
from . import constants as C

# ======================================================================================
# GRADED LANTERN SHADE (Jake's premise extension, 2026-06-11): each sun's opaque collimator
# cylinder generalized to a wall with transmission T_eff(alpha), alpha = angle from the vertical
# (zenith-angle-only by the cylinder's rotational symmetry). Loaded at import from
# COSMO_SHADE_FILE (npz: alpha_deg, T_eff; T_eff = I0_mult * transmission, so values > 1 encode
# brighter suns behind darker shades). Default OFF -> byte-identical to the binary collimator.
# Designs come from flatworld._shade_design. The CLOCK never sees the shade (spacing D is the
# established setting value; SunsClock passes gate=False).
# NOTE: T_eff(alpha) is the day-MEAN DELIVERY (azimuthally symmetric) — it does NOT redistribute flux
# cross-track between latitudes. To reshape the lantern cross-track (move day-mean S between
# latitudes) use the physical 2-D lens T(alpha, psi): lantern_wall_figure.py / plot_lantern_optic.py.
# ======================================================================================
_SHADE = None
_shade_path = os.environ.get("COSMO_SHADE_FILE")
if _shade_path:
    try:
        _sd = np.load(_shade_path)
        _SHADE = (np.asarray(_sd["alpha_deg"], float), np.asarray(_sd["T_eff"], float))
    except Exception as _e:  # noqa: BLE001 -- re-raise with the env contract spelled out
        raise RuntimeError(
            f"COSMO_SHADE_FILE={_shade_path!r} could not be loaded (cwd {os.getcwd()!r}): "
            f"{type(_e).__name__}: {_e}. Expected an npz with 'alpha_deg'/'T_eff' arrays "
            f"(flatworld._shade_design.save_shades); relative paths resolve against the cwd -- run "
            f"from cosmology_models/. Unset COSMO_SHADE_FILE for the binary collimator.") from _e

# GRADED 2-D LANTERN LENS T(alpha, psi) (the genuine redistributing optic; COSMO_SHADE2D_FILE, default
# OFF). Unlike the radial T_eff(alpha) shade above (day-mean DELIVERY only), a 2-D transmission/gain
# T(alpha, psi) can move day-mean flux CROSS-TRACK between latitudes: (alpha, psi) <-> ground offset
# (x = L*tan(alpha)*sin(psi) cross-track, e = L*tan(alpha)*cos(psi) along-track). Loaded from an npz
# (alpha_deg, psi_deg, T_eff[alpha, psi]); T may exceed 1 (a lens CONCENTRATES, an absorber cannot).
# The lens has an EXACT INSTANTANEOUS per-sun form: a sun whose foot is at along-track offset e from the
# observer's meridian is seen at (alpha=arccos(L/r), psi=atan2(|x|,|e|)) with r=sqrt(L^2+x^2+e^2), so
# ``_shade_factor_2d(x, e)`` gives its transmission at any moment of the day. It is therefore applied BOTH
# in the day-mean integrals (day_mean_flux) AND, since 2026-07-15, instantaneously in the diurnal-sun
# tables (diurnal_flux_table / diurnal_flux_cycle, via _direct_flux_xe / _diffuse_flux_xe) -- so a diurnal
# AD-core run sees the lens's real cycle SHAPE, not just its day-mean magnitude. (Only the shapeless
# COSMO_LENS_RESHAPE (b) day-mean redistribution below has no instantaneous form.) Default OFF -> _SHADE2D
# is None -> _direct_flux_xe/_diffuse_flux_xe collapse to the 1-D path -> BYTE-IDENTICAL everywhere.
# Designs come from nz_resolution_robustness/_lens_2d_shade_design.py (the equatorial-focus reshape) and
# _lens_2d_canonical_design.py (day-mean reduces to the canonical 1-D shade); see LENS2D_DIURNAL_RESULTS.md.
_SHADE2D = None
_shade2d_path = os.environ.get("COSMO_SHADE2D_FILE")
if _shade2d_path:
    try:
        _sd2 = np.load(_shade2d_path)
        _SHADE2D = (np.asarray(_sd2["alpha_deg"], float), np.asarray(_sd2["psi_deg"], float),
                    np.asarray(_sd2["T_eff"], float))            # (na,), (np_,), (na, np_)
    except Exception as _e2:  # noqa: BLE001
        raise RuntimeError(
            f"COSMO_SHADE2D_FILE={_shade2d_path!r} could not be loaded (cwd {os.getcwd()!r}): "
            f"{type(_e2).__name__}: {_e2}. Expected an npz with 'alpha_deg'/'psi_deg'/'T_eff' "
            f"(T_eff shape (n_alpha, n_psi); _lens_2d_shade_design.py). Unset it for the 1-D shade.") from _e2
    # A 2-D lens is CALIBRATED against the thin-haze bare flux (its designer divides by _one_sun_total,
    # which uses TAU0_HAZE_THIN), but at runtime _flux_one_sun(gate=False) reads _active_tau0() -- thin
    # only while COLLIMATOR is ON. With the collimator OFF the runtime multiplies the thin-designed T by
    # the FULL-haze bare flux, so the day-mean silently stops reducing to its design target. Warn once.
    if not C.COLLIMATOR:
        warnings.warn(
            f"COSMO_SHADE2D_FILE={_shade2d_path!r} is loaded but COSMO_COLLIMATOR is OFF: the lens is "
            f"calibrated against the thin-haze bare flux (TAU0_HAZE_THIN), while _active_tau0() is the "
            f"FULL haze with the collimator off -- the day-mean will NOT reduce to the lens's design "
            f"target. Keep the collimator ON (the default) when using a 2-D lens.", RuntimeWarning,
            stacklevel=2)


def _shade_factor(r):
    """T_eff at slant distance r [m] (alpha from cos(alpha) = L/r); edge values held outside."""
    a = np.degrees(np.arccos(np.clip(C.L_SUN / np.asarray(r, float), -1.0, 1.0)))
    return np.interp(a, _SHADE[0], _SHADE[1])


def _shade_factor_2d(x, e):
    """2-D lens transmission/gain T(alpha, psi) at ground offset (x cross-track, e along-track) [m].
    alpha = arccos(L/r) with r = sqrt(L^2 + x^2 + e^2); psi = atan2(|x|, |e|) (0 = along-track pure, 90 =
    cross-track pure). Bilinear on the (alpha_deg, psi_deg) table, edge-held outside. Vectorized."""
    xa = np.abs(np.asarray(x, float)); ea = np.abs(np.asarray(e, float))
    r = np.sqrt(C.L_SUN**2 + xa**2 + ea**2)
    al = np.degrees(np.arccos(np.clip(C.L_SUN / r, -1.0, 1.0)))
    ps = np.degrees(np.arctan2(xa, ea))
    ax, px, Z = _SHADE2D
    ia = np.clip(np.searchsorted(ax, al) - 1, 0, len(ax) - 2)
    ip = np.clip(np.searchsorted(px, ps) - 1, 0, len(px) - 2)
    fa = np.clip((al - ax[ia]) / (ax[ia + 1] - ax[ia]), 0.0, 1.0)
    fp = np.clip((ps - px[ip]) / (px[ip + 1] - px[ip]), 0.0, 1.0)
    return ((1 - fa) * (1 - fp) * Z[ia, ip] + fa * (1 - fp) * Z[ia + 1, ip]
            + (1 - fa) * fp * Z[ia, ip + 1] + fa * fp * Z[ia + 1, ip + 1])


def _direct_flux_xe(x, e):
    """Per-sun DIRECT flux at ground offset (x, e): the 2-D lens (bare projected+haze x T(alpha,psi))
    when active, else the unchanged 1-D path _flux_one_sun(r) -> BYTE-IDENTICAL when _SHADE2D is None."""
    if _SHADE2D is None:
        return _flux_one_sun(slant_distance(x, e))
    r = slant_distance(x, e)
    return _flux_one_sun(r, gate=False) * _shade_factor_2d(x, e)


def _diffuse_flux_xe(x, e, omega0, g_haze):
    """Per-sun DIFFUSE flux at (x, e): 2-D-lens-shaped when active, else the unchanged 1-D path."""
    r = slant_distance(x, e)
    if _SHADE2D is None:
        return _diffuse_one_sun(r, omega0, g_haze)
    return _diffuse_one_sun(r, omega0, g_haze, gate=False) * _shade_factor_2d(x, e)


# ======================================================================================
# The cross-track metric:  x = L * tan(theta)
# ======================================================================================
def x_of_theta(theta_rad):
    """Cross-track distance x [m] from the sun-line, given the angle theta [rad]."""
    return C.L_SUN * np.tan(theta_rad)


def theta_of_x(x):
    """Angle theta [rad] from the sun-line, given cross-track distance x [m]."""
    return np.arctan(np.asarray(x, float) / C.L_SUN)


def cos_theta_of_x(x):
    """cos(theta) = L / sqrt(L^2 + x^2). Avoids a tan/arctan round-trip; exact."""
    x = np.asarray(x, float)
    return C.L_SUN / np.sqrt(C.L_SUN**2 + x**2)


def flux_shape_of_x(x):
    """The day-mean flux SHAPE cos^3(theta) = L^3/(L^2+x^2)^{3/2} as a function of x [m]."""
    return cos_theta_of_x(x) ** 3


# ======================================================================================
# Instantaneous surface flux from the train of suns
# ======================================================================================
def _active_tau0():
    """Slant-haze optical depth for the FLUX: the thin bright haze when the collimator is on, else the
    full reference TAU0_HAZE. (The clock always uses the TAU0_HAZE reference -- see SunsClock.)"""
    return C.TAU0_HAZE_THIN if C.COLLIMATOR else C.TAU0_HAZE


def _visible_fraction(r):
    """Fraction (0..1) of a sun's disk that threads its collimator tube at slant distance r [m].

    Each sun rides in a co-moving vertical dark cylinder (constants.COLLIMATOR): the observer sees it only
    while the line of sight clears the rim, i.e. while the zenith angle z (cos z = L/r) is below the cutoff
    z_cut = COLLIMATOR_ZCUT. The sun's FINITE angular radius rho = R_SUN_BODY/r softens the cut into a
    circular-segment area fraction (the partial-eclipse sunset): a smooth step from 1 (z < z_cut - rho,
    fully threading) to 0 (z > z_cut + rho, fully blocked). u = (z_cut - z)/rho is the signed disk
    penetration. Vectorized in r."""
    r = np.asarray(r, float)
    z_c = np.arccos(np.clip(C.L_SUN / r, -1.0, 1.0))
    rho = C.R_SUN_BODY / r
    u = np.clip((C.COLLIMATOR_ZCUT - z_c) / rho, -1.0, 1.0)
    return 0.5 + (u * np.sqrt(1.0 - u * u) + np.arcsin(u)) / np.pi


def _flux_one_sun(r, tau0=None, gate=None):
    """Flux on a horizontal surface [W/m^2] from one sun at slant distance r [m].

    F = I0 * (L/r)^3 * exp(-tau0*(r/L - 1)):
      (L/r)^2 inverse-square (relative to the overhead-at-L reference where F=I0),
      (L/r)    projection onto the horizontal (cos zenith = L/r),
      exp(...) slant-path haze extinction (airmass = sec zenith = r/L; zero extra at the vertical).
    At the sub-solar point (r=L) this is exactly I0 (the gate is 1 overhead).

    ``tau0``/``gate`` default to the module-active values: the thin haze + collimator gate when
    COSMO_COLLIMATOR is on, else the full haze and no gate -> BYTE-IDENTICAL to the pre-collimator law.
    With the gate on, F is multiplied by the visible disk fraction (0 beyond the cutoff zenith), so night
    is a geometric zero. The clock passes ``tau0=TAU0_HAZE, gate=False`` so the spacing D is unchanged.
    """
    tau0 = _active_tau0() if tau0 is None else tau0
    gate = C.COLLIMATOR if gate is None else gate
    Lr = C.L_SUN / r
    F = C.I0_OVERHEAD * Lr**3 * np.exp(-tau0 * (r / C.L_SUN - 1.0))
    if gate and _SHADE is not None:
        F = F * _shade_factor(r)          # the graded shade REPLACES the binary cylinder rim
    elif gate:
        F = F * _visible_fraction(r)
    return F


def _diffuse_one_sun(r, omega0=None, g_haze=None, gate=None):
    """Forward-scattered DIFFUSE haze flux on a horizontal surface [W/m^2] from one sun at slant distance r.

    CLEARSKY_SW_PLAN Stage 5. ``_flux_one_sun`` keeps only the DIRECT beam through the slant haze; the haze
    removes ``(1 - exp(-tau_slant))`` of the geometric reference flux ``F_ref = I0*(L/r)^3`` (the overhead-haze
    is folded into I0, so ``tau_slant = TAU0_HAZE*(r/L - 1)`` is the EXTRA slant optical depth beyond the
    vertical). Treating that as pure loss is wrong: a fraction ``omega0`` (the single-scattering albedo) is
    SCATTERED, of which the FORWARD hemisphere ``b_fwd = (1+g)/2`` (the two-stream forward-scatter fraction)
    continues downward and reaches the ground as DIFFUSE light. So the recovered diffuse flux is

        F_diff = F_ref * omega0 * b_fwd * (1 - exp(-tau_slant)).

    SINGLE-SCATTERING approximation (one scatter, the forward part escapes the haze downward without further
    interaction) -- clean, energy-bounded, and conservative for the moderate slant tau here. At overhead
    (r=L, tau_slant=0) it is exactly 0 (no extra extinction -> no diffuse), and it grows toward
    ``F_ref*omega0*b_fwd`` at low sun where the slant haze is opaque. Pairs with ``_flux_one_sun`` so the
    direct + diffuse + backscatter + haze-absorption budget telescopes to F_ref (energy-conserving)."""
    omega0 = C.OMEGA0_HAZE if omega0 is None else omega0     # resolve at CALL time -- def-time defaults
    g_haze = C.G_HAZE if g_haze is None else g_haze          # would miss a monkeypatched constant
    Lr = C.L_SUN / r
    F_ref = C.I0_OVERHEAD * Lr ** 3                          # geometric reference flux (vertical haze in I0)
    tau_slant = _active_tau0() * (r / C.L_SUN - 1.0)         # EXTRA slant haze beyond the vertical reference
    b_fwd = 0.5 * (1.0 + g_haze)                             # two-stream forward (downward) scattered fraction
    F_diff = F_ref * omega0 * b_fwd * (1.0 - np.exp(-tau_slant))
    gate = C.COLLIMATOR if gate is None else gate            # gate=False -> bare diffuse (2-D lens applies its own T)
    if gate and _SHADE is not None:
        F_diff = F_diff * _shade_factor(r)                   # shaded beam -> proportionally less to scatter
    elif gate:
        F_diff = F_diff * _visible_fraction(r)               # blocked sun -> no beam to scatter -> no diffuse
    return F_diff


def slant_distance(x, e_offset):
    """Slant distance r [m] from a ground observer at cross-track x to a sun whose foot on the track
    is at along-track offset ``e_offset`` from the observer's meridian. r^2 = L^2 + x^2 + e_offset^2."""
    return np.sqrt(C.L_SUN**2 + np.asarray(x, float) ** 2 + np.asarray(e_offset, float) ** 2)


# ======================================================================================
# The suns' clock: solve spacing D (hence speed v) from the midnight-darkness floor.
# The 24 h day is the imposed choice (Appendix H / C); D follows from the floor, v = D / T_day.
# Brightness I0 cancels in the floor ratio, so the clock is brightness-independent (as it must be).
# ======================================================================================
class SunsClock:
    """The train's kinematics, fixed by requiring midnight to bottom out at MIDNIGHT_FLOOR_FRAC."""

    def __init__(self, T_day_s: float = 86400.0,
                 theta0_deg: float = C.THETA0_CLOCK_DEG,
                 floor_frac: float = C.MIDNIGHT_FLOOR_FRAC):
        self.T_day = float(T_day_s)
        x0 = x_of_theta(np.radians(theta0_deg))           # observer's cross-track distance at the clock ref
        r_noon = slant_distance(x0, 0.0)                   # nearest sun overhead at noon

        def floor_ratio(half_spacing):
            # midnight: the two nearest suns sit at +/- half_spacing along-track. The clock uses the
            # TAU0_HAZE=1.0 REFERENCE flux, ungated -- so the established spacing D is KEPT whether or not
            # the collimator is on (with the collimator the floor is no longer the *darkness* mechanism;
            # the geometric cutoff is -- but D stays the established setting value). I0 cancels in the ratio.
            r_mid = slant_distance(x0, half_spacing)
            return (2.0 * _flux_one_sun(r_mid, tau0=C.TAU0_HAZE, gate=False)
                    / _flux_one_sun(r_noon, tau0=C.TAU0_HAZE, gate=False))

        # bisection for the half-spacing s_mid that makes the midnight ratio == floor_frac
        lo, hi = 1.0, 5.0e10
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if floor_ratio(mid) > floor_frac else (lo, mid)
        self.half_spacing = 0.5 * (lo + hi)                # D/2 [m]
        self.spacing = 2.0 * self.half_spacing             # D [m] (sun-to-sun gap = one day)
        self.speed = self.spacing / self.T_day             # v [m/s]

    @property
    def speed_km_s(self):
        return self.speed / 1e3

    @property
    def spacing_km(self):
        return self.spacing / 1e3


# A module-level default clock (24 h day, hazy sky) -- the setting's clock.
CLOCK = SunsClock()


# ======================================================================================
# Day-mean surface flux S(theta) [W/m^2] -- THE radiative driver.
#
# Because the train is periodic (spacing D in space, T_day in time), the time-average over one day of
# the summed train equals one sun's flux integrated over all along-track space, divided by the spacing:
#       S(x) = (1/D) * \int_{-inf}^{+inf} F_one( r=sqrt(L^2+x^2+e^2) ) de.
# Exact (no train-truncation), and a clean 1-D integral per latitude.
# ======================================================================================
def day_mean_flux_scalar(x: float, clock: SunsClock | None = None, n: int = 6000) -> float:
    """Day-mean incident surface flux S [W/m^2] at one cross-track distance x [m] (no albedo).

    Integrates one sun over all along-track e and divides by the spacing. The integrand is sharply
    peaked at e=0 (width ~ sqrt(L^2+x^2)) with a 1/e^3 tail, so we use a sinh-stretched grid that puts
    most nodes near the peak and still reaches far into the tail -- robust where adaptive quad on
    [0, inf) under-samples the peak.

    NOTE: this scalar path does NOT apply the (b) equatorial-focus reshape (COSMO_LENS_RESHAPE) --
    that redistribution is only well-defined over the whole cross-track grid (see day_mean_flux). A
    reshape run therefore emits a one-time warning here so a pointwise sampler is not silently fed the
    un-lensed forcing.
    """
    clock = CLOCK if clock is None else clock              # CALL-time default (a reassigned module
    #   CLOCK is honoured -- def-time binding would silently pin the import-time clock; same rule
    #   as the omega0/g_haze constants below)
    if _LENS_RESHAPE and _SHADE2D is None:
        _lens_passthrough_warn(float(np.degrees(np.arctan(abs(float(x)) / C.L_SUN))))
    A = np.sqrt(C.L_SUN**2 + float(x) ** 2)                # peak width scale
    u = np.linspace(0.0, 20.0, n)                          # sinh(20) ~ 2.4e8 -> e reaches ~2e8*A (huge)
    e = A * np.sinh(u)
    de_du = A * np.cosh(u)
    integrand = _direct_flux_xe(x, e) * de_du
    val = np.trapezoid(integrand, u)
    return 2.0 * val / clock.spacing                       # factor 2 for the symmetric +/- e halves


def _day_mean_flux_vectorized(x, spacing: float, n: int):
    """Array-valued core of ``day_mean_flux``: the SAME sinh-stretched quadrature as
    ``day_mean_flux_scalar``, broadcast over the whole cross-track grid at once (no Python loop).

    Element i is ``day_mean_flux_scalar(x[i], clock, n)`` for a clock of the given ``spacing`` --
    identical arithmetic in an identical order (peak scale A, sinh/cosh nodes, slant distance, the
    per-sun flux, then a trapezoid over u), just done with a leading x-axis. Verified bit-for-bit
    against the scalar reference (``tests/test_geometry.py``)."""
    x = np.ascontiguousarray(np.atleast_1d(np.asarray(x, float)))
    A = np.sqrt(C.L_SUN**2 + x**2)                          # (nx,) peak width scale, per latitude
    u = np.linspace(0.0, 20.0, n)                          # (n,) sinh-stretched quadrature nodes
    e = A[:, None] * np.sinh(u)[None, :]                    # (nx, n) along-track distance
    de_du = A[:, None] * np.cosh(u)[None, :]               # (nx, n) jacobian du -> de
    integrand = _direct_flux_xe(x[:, None], e) * de_du
    val = np.trapezoid(integrand, u, axis=1)              # (nx,) integral over one +/- e half
    return 2.0 * val / spacing                             # factor 2 for the symmetric +/- e halves


def _flux_env_key():
    """Every module-level input BESIDES (x, spacing, n) that the day-mean flux integrals read,
    packed hashable so the lru_caches below never return a value stale w.r.t. a monkeypatched constant
    (L, I0, the collimator gate, the active haze tau0, the collimator geometry) or a swapped graded-
    shade table. Read LIVE at call time -- a change to any of these is a cache MISS, so the value is
    recomputed with the current constants. Shared by BOTH the direct beam (``day_mean_flux``) and the
    forward-scattered diffuse haze (``day_mean_diffuse_flux``): both integrands read exactly this set of
    module-level inputs (the diffuse path's extra inputs -- omega0, g_haze -- are passed as explicit
    cache params after their None->constant default is resolved at call time, so they are NOT here)."""
    shade = None if _SHADE is None else (_SHADE[0].tobytes(), _SHADE[1].tobytes())
    shade2d = None if _SHADE2D is None else (_SHADE2D[0].tobytes(), _SHADE2D[1].tobytes(),
                                             _SHADE2D[2].tobytes())
    return (C.L_SUN, C.I0_OVERHEAD, bool(C.COLLIMATOR), _active_tau0(),
            C.COLLIMATOR_ZCUT, C.R_SUN_BODY, shade, shade2d)


@functools.lru_cache(maxsize=256)
def _day_mean_flux_cached(x_bytes: bytes, shape, n: int, spacing: float, env_key):
    """Memoized ``_day_mean_flux_vectorized`` keyed on the raw x-grid bytes + the scalar params
    (n, spacing) + the module-env fingerprint ``env_key``. ``env_key`` is unused in the body -- the
    compute reads C.* live -- but its presence in the key is what makes any constant change a miss.
    Returns a READ-ONLY array; the public wrapper hands out a fresh copy so no caller can corrupt the
    shared cache entry in place."""
    x = np.frombuffer(x_bytes, dtype=np.float64).reshape(shape)
    out = _day_mean_flux_vectorized(x, spacing, n)
    out.setflags(write=False)
    return out


# ======================================================================================
# LENS RESHAPE (optics-layer equatorial-focus redistribution; COSMO_LENS_RESHAPE, default OFF).
# The physical lantern is a 2-D REDISTRIBUTING lens T(alpha, psi) that CAN move day-mean flux
# CROSS-TRACK between latitudes -- unlike the radial T_eff(alpha) shade above, which is day-mean
# DELIVERY only (see the _shade_design banner). Here we apply the day-mean such a lens produces as a
# fixed multiplicative gain g(theta) on day_mean_flux's output: dim the donor band
# [theta_pool, theta_donor] by (1 - phi) with smooth C1 edges, pile the removed power onto a narrow
# cos^2 equatorial peak [0, theta_pool], power-conserving on the canonical grid. Default OFF -> g == 1
# -> BYTE-IDENTICAL. Realizability (rim floor / finite-sun blur / power) verified in
# nz_resolution_robustness/_lens_reshape_design.py. This is the pragmatic (b) form; the production
# form is a genuine 2-D T(alpha, psi) shade file integrated by day_mean_flux (follow-up).
# ======================================================================================
_LENS_RESHAPE = os.environ.get("COSMO_LENS_RESHAPE", "0").strip().lower() in ("1", "true", "on", "yes")
_LENS_PHI = float(os.environ.get("COSMO_LENS_PHI", "0.3"))
_LENS_THETA_POOL = float(os.environ.get("COSMO_LENS_THETA_POOL", "2.0"))
_LENS_THETA_DONOR = float(os.environ.get("COSMO_LENS_THETA_DONOR", "30.0"))
_LENS_EDGE_DEG = float(os.environ.get("COSMO_LENS_EDGE_DEG", "1.5"))

_LENS_PASSTHROUGH_WARNED = False


def _lens_passthrough_warn(theta_min: float) -> None:
    """One-time LOUD warning that a reshape run returned UN-lensed flux for a partial/scalar request.

    The (b) reshape is only meaningful over the whole cross-track grid, so a scalar
    (``day_mean_flux_scalar``) or sub-range call is passed through untouched -- but if that request
    reaches into the modified region [0, theta_donor] it silently gets the UN-reshaped forcing (up to
    ~9x at the equator). Harmless for the full production grid; this catches the trap where a
    diagnostic samples the day-mean flux pointwise while COSMO_LENS_RESHAPE is live (the same
    silent-forcing class as the import-timing L_SUN trap). Only fires inside the donor range -- outside
    it the identity is exact and silence is correct."""
    global _LENS_PASSTHROUGH_WARNED
    if _LENS_PASSTHROUGH_WARNED or theta_min > _LENS_THETA_DONOR:
        return
    _LENS_PASSTHROUGH_WARNED = True
    warnings.warn(
        f"COSMO_LENS_RESHAPE is ON but a partial/scalar day-mean flux request "
        f"(theta_min={theta_min:.2f} deg, inside donor {_LENS_THETA_DONOR:.1f} deg) was returned "
        f"UN-reshaped -- the (b) reshape needs the full cross-track grid. A diagnostic sampling the "
        f"day-mean flux pointwise while a reshape run is live gets the un-lensed forcing.",
        RuntimeWarning, stacklevel=3)


def _apply_lens_reshape(x_arr, S):
    """Equatorial-focus redistribution of a day-mean profile S on grid x_arr (optics-layer lens;
    COSMO_LENS_RESHAPE). Dim the donor band [theta_pool, theta_donor] by (1 - phi) with smooth C1
    edges, pile the removed power onto a cos^2 equatorial peak [0, theta_pool]. POWER-CONSERVING ON
    THE PASSED GRID BY CONSTRUCTION (added == removed, same quadrature) -- so the full production grid
    conserves exactly. A partial grid (single column / sub-range that does not span the equator through
    past the donor) is returned UNCHANGED: the redistribution is only meaningful over the whole
    cross-track domain, and outside the donor/pool the reshape is the identity anyway."""
    S = np.asarray(S, float)
    if S.size < 8:
        xa0 = np.abs(np.asarray(x_arr, float))
        _lens_passthrough_warn(float(np.degrees(np.arctan(xa0.min() / C.L_SUN))) if xa0.size else 0.0)
        return S
    xa = np.abs(np.asarray(x_arr, float))
    theta = np.degrees(np.arctan(xa / C.L_SUN))
    phi, tp, td, e = _LENS_PHI, _LENS_THETA_POOL, _LENS_THETA_DONOR, _LENS_EDGE_DEG
    if theta.min() > tp or theta.max() < td:                     # partial grid -> pass through untouched
        _lens_passthrough_warn(float(theta.min()))
        return S
    order = np.argsort(xa); xs = xa[order]; th = theta[order]; Ss = S[order]

    def smoothstep(a, lo, hi):
        u = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
        return u * u * (3.0 - 2.0 * u)

    donor_w = smoothstep(th, tp, tp + e) * (1.0 - smoothstep(th, td - e, td))   # ~1 in donor, 0 outside
    S_dim = Ss * (1.0 - phi * donor_w)                           # dim the donor (smooth C1 edges)
    dP = np.trapezoid(Ss - S_dim, xs)                            # power removed [W per m along-track]
    x_tp = C.L_SUN * np.tan(np.radians(tp))
    shape = np.where(th < tp, np.cos(0.5 * np.pi * xs / x_tp) ** 2, 0.0)   # cos^2 equatorial peak
    norm = np.trapezoid(shape, xs)
    S_sorted = S_dim + (dP * shape / norm if norm > 0 else 0.0)  # pile removed power onto the pool
    out = np.empty_like(S)
    out[order] = S_sorted                                        # unsort back to the caller's order
    return out


def day_mean_flux(x, clock: SunsClock | None = None, n: int = 6000):
    """Vectorized, cached day-mean incident surface flux S [W/m^2] over an array of cross-track x [m].

    Vectorized (one broadcast quadrature over the whole grid, replacing the former nx-long Python loop
    of scalar quadratures) and memoized on the (x-grid, spacing, n) + module-env fingerprint, because
    the SAME theta/x grid is re-requested by every stage instance and every sweep member. Numerically
    identical to a loop of ``day_mean_flux_scalar`` (bit-for-bit; see tests). Returns a FRESH writeable
    array (the cache holds a read-only copy), so a caller may mutate the result in place as before."""
    clock = CLOCK if clock is None else clock              # CALL-time default (see day_mean_flux_scalar)
    x_arr = np.ascontiguousarray(np.atleast_1d(np.asarray(x, float)))
    out = _day_mean_flux_cached(x_arr.tobytes(), x_arr.shape, int(n),
                                float(clock.spacing), _flux_env_key())
    if _LENS_RESHAPE and _SHADE2D is None:                       # (b) post-hoc reshape; (a) 2-D shade supersedes it
        out = _apply_lens_reshape(x_arr, out)
    return out.copy() if out.size > 1 else float(out[0])


def day_mean_diffuse_flux_scalar(x: float, clock: SunsClock | None = None, n: int = 6000,
                                 omega0=None, g_haze=None) -> float:
    """Day-mean FORWARD-SCATTERED diffuse haze flux [W/m^2] at cross-track x [m] (CLEARSKY_SW_PLAN Stage 5).

    The diffuse analogue of ``day_mean_flux_scalar``: it unfolds the train the SAME way (one sun integrated
    over all along-track e, divided by the spacing) but uses ``_diffuse_one_sun`` (the haze forward-scatter)
    instead of ``_flux_one_sun`` (the direct beam). This is the extra surface SW the slant haze forward-
    scatters back to the ground -- recovered light the pure-loss direct-beam ``day_mean_flux`` drops. Adds to
    (does not replace) the direct day-mean flux; the clock and the direct flux are untouched."""
    clock = CLOCK if clock is None else clock              # CALL-time default (see day_mean_flux_scalar)
    A = np.sqrt(C.L_SUN**2 + float(x) ** 2)                # same sinh-stretched peak-resolving grid as the direct
    u = np.linspace(0.0, 20.0, n)
    e = A * np.sinh(u)
    de_du = A * np.cosh(u)
    integrand = _diffuse_flux_xe(x, e, omega0, g_haze) * de_du
    val = np.trapezoid(integrand, u)
    return 2.0 * val / clock.spacing                       # factor 2 for the symmetric +/- e halves


def _day_mean_diffuse_flux_vectorized(x, spacing: float, n: int, omega0: float, g_haze: float):
    """Array-valued core of ``day_mean_diffuse_flux``: the SAME sinh-stretched quadrature as
    ``day_mean_diffuse_flux_scalar``, broadcast over the whole cross-track grid at once (no Python loop).

    The diffuse twin of ``_day_mean_flux_vectorized`` -- identical peak-resolving grid and arithmetic
    order, but with ``_diffuse_one_sun`` (the haze forward-scatter) in place of ``_flux_one_sun`` (the
    direct beam). ``omega0``/``g_haze`` arrive ALREADY RESOLVED (the None->constant default is applied by
    the public wrapper), so the arithmetic matches a loop of ``day_mean_diffuse_flux_scalar`` bit-for-bit
    (``tests/test_geometry.py``)."""
    x = np.ascontiguousarray(np.atleast_1d(np.asarray(x, float)))
    A = np.sqrt(C.L_SUN**2 + x**2)                          # (nx,) peak width scale, per latitude
    u = np.linspace(0.0, 20.0, n)                          # (n,) sinh-stretched quadrature nodes
    e = A[:, None] * np.sinh(u)[None, :]                    # (nx, n) along-track distance
    de_du = A[:, None] * np.cosh(u)[None, :]               # (nx, n) jacobian du -> de
    integrand = _diffuse_flux_xe(x[:, None], e, omega0, g_haze) * de_du
    val = np.trapezoid(integrand, u, axis=1)              # (nx,) integral over one +/- e half
    return 2.0 * val / spacing                             # factor 2 for the symmetric +/- e halves


@functools.lru_cache(maxsize=256)
def _day_mean_diffuse_flux_cached(x_bytes: bytes, shape, n: int, spacing: float,
                                  omega0: float, g_haze: float, env_key):
    """Memoized ``_day_mean_diffuse_flux_vectorized`` keyed on the raw x-grid bytes + the scalar params
    (n, spacing, and the RESOLVED omega0/g_haze) + the module-env fingerprint ``env_key`` (shared with the
    direct path via ``_flux_env_key``). ``env_key`` is unused in the body -- the compute reads C.* live --
    but its presence in the key is what makes any constant change a miss. omega0/g_haze are explicit key
    params (not folded into env_key) because a caller may pass them per-call; the wrapper resolves
    None->C.OMEGA0_HAZE/C.G_HAZE at CALL time so a monkeypatched constant is still a miss. Returns a
    READ-ONLY array; the public wrapper hands out a fresh copy so no caller can corrupt the cache in place."""
    x = np.frombuffer(x_bytes, dtype=np.float64).reshape(shape)
    out = _day_mean_diffuse_flux_vectorized(x, spacing, n, omega0, g_haze)
    out.setflags(write=False)
    return out


def day_mean_diffuse_flux(x, clock: SunsClock | None = None, n: int = 6000, omega0=None, g_haze=None):
    """Vectorized, cached day-mean forward-scattered diffuse haze flux [W/m^2] over an array of x [m].

    The diffuse twin of ``day_mean_flux``: one broadcast quadrature over the whole grid (replacing the
    former nx-long Python loop of scalar quadratures) and memoized on the (x-grid, spacing, n, resolved
    omega0/g_haze) + module-env fingerprint, because the SAME theta/x grid is re-requested by every stage
    instance and every sweep member. Numerically identical to a loop of ``day_mean_diffuse_flux_scalar``
    (bit-for-bit; see tests). ``omega0``/``g_haze`` default to the live ``C.OMEGA0_HAZE``/``C.G_HAZE``
    resolved HERE at call time (so a monkeypatched constant is honoured -- a cache MISS, not a stale hit).
    Returns a FRESH writeable array (the cache holds a read-only copy), so a caller may mutate in place."""
    clock = CLOCK if clock is None else clock                # CALL-time default (see day_mean_flux_scalar)
    omega0 = C.OMEGA0_HAZE if omega0 is None else omega0     # resolve at CALL time (matches _diffuse_one_sun)
    g_haze = C.G_HAZE if g_haze is None else g_haze          # so a monkeypatched constant keys a cache miss
    x_arr = np.ascontiguousarray(np.atleast_1d(np.asarray(x, float)))
    out = _day_mean_diffuse_flux_cached(x_arr.tobytes(), x_arr.shape, int(n), float(clock.spacing),
                                        float(omega0), float(g_haze), _flux_env_key())
    if _LENS_RESHAPE and _SHADE2D is None:                       # (b) post-hoc reshape; (a) 2-D shade supersedes it
        out = _apply_lens_reshape(x_arr, out)
    return out.copy() if out.size > 1 else float(out[0])


def diurnal_flux_cycle(x: float, n_steps: int = 1440, clock: SunsClock | None = None,
                       n_suns: int = 6):
    """Instantaneous surface flux over one day at cross-track x [m]: returns (t [s], S(t) [W/m^2]).

    Sums the ``n_suns`` nearest suns of the train. Used by the diurnally resolved surface energy
    balance (Stage A / C). The time-mean of this equals ``day_mean_flux(x)`` to truncation in n_suns.
    """
    clock = CLOCK if clock is None else clock              # CALL-time default (see day_mean_flux_scalar)
    t = np.linspace(0.0, clock.T_day, n_steps + 1)         # t=0 at a sun's transit of this meridian (noon)
    e_self = clock.speed * t                               # along-track position of the transiting sun
    S = np.zeros_like(t)
    for k in range(-n_suns, n_suns + 1):
        # lens-aware per-sun direct flux at this sun's along-track offset (x cross-track, e along-track):
        # applies the 2-D lens T(alpha, psi) instantaneously when active; collapses to _flux_one_sun(r)
        # -> BYTE-IDENTICAL when _SHADE2D is None.
        S += _direct_flux_xe(x, e_self - k * clock.spacing)
    return t, S


def diurnal_flux_table(x, n_phase: int = 1440, clock: SunsClock | None = None, n_suns: int = 6,
                       diffuse: bool = False):
    """One day of instantaneous surface flux on a (phase, x) grid: returns (n_phase, nx) [W/m^2].

    Row ``j`` is the summed train flux at the phase MIDPOINT ``t_j = (j + 0.5) * T_day / n_phase``
    (t=0 a sun's transit of the observer's meridian = noon, as in ``diurnal_flux_cycle``), summing the
    ``2*n_suns+1`` nearest suns; the per-sun collimator gate / graded shade / active haze come along via
    ``_flux_one_sun``. Because the model's 2-D ``(x, z)`` slice sits at ONE along-track meridian and the
    train moves along-track, every column shares the same phase -- the whole slice lights and darkens
    together, so one row is the whole domain's simultaneous forcing. With ``diffuse=True`` the rows are
    the forward-scattered haze flux (``_diffuse_one_sun``) instead -- the diffuse twin.

    THE 2-D LENS (COSMO_SHADE2D) IS CARRIED INSTANTANEOUSLY (2026-07-15): each sun contributes through
    ``_direct_flux_xe`` / ``_diffuse_flux_xe`` at its along-track offset ``e``, which evaluate the lens
    ``T(alpha=arccos(L/r), psi=atan2(|x|,|e|))`` at that instant -- so with a 2-D lens loaded the table
    already carries the lens's real cycle SHAPE (and its phase-mean equals ``day_mean_flux`` to
    truncation), NOT just its day-mean magnitude. When ``_SHADE2D is None`` these collapse to
    ``_flux_one_sun(r)`` / ``_diffuse_one_sun(r)`` -> BYTE-IDENTICAL to the pre-2026-07-15 per-sun train.

    NOTE (consumers -- the AD core's ``COSMO_DIURNAL_SUN`` path): the row time-mean equals the analytic
    ``day_mean_flux`` only to truncation (finite ``n_suns``, midpoint phase sampling), and the table still
    does NOT carry the SHAPELESS optics-layer day-mean redistribution (COSMO_LENS_RESHAPE (b), which has
    no instantaneous 1-D form here -- only the genuine 2-D lens (a) above does). Consumers therefore still
    RENORMALIZE each column so the table's phase-mean matches their day-mean forcing exactly (see
    ``stage_ad_core.StageADCore._build_diurnal_tables``); with the 2-D lens that renormalization is a
    small finite-``n_suns`` truncation correction (gain ~1), and for COSMO_LENS_RESHAPE it remains the
    phase-uniform per-column gain approximation -- the diurnal SHAPE is this per-sun train, the day-mean
    MAGNITUDE stays the model's S(x)."""
    clock = CLOCK if clock is None else clock                  # CALL-time default (see day_mean_flux_scalar)
    x = np.ascontiguousarray(np.atleast_1d(np.asarray(x, float)))
    t = (np.arange(n_phase) + 0.5) * (clock.T_day / n_phase)   # (n_phase,) phase midpoints [s]
    e_self = clock.speed * t                                   # transiting sun's along-track position
    S = np.zeros((n_phase, x.size))
    for k in range(-n_suns, n_suns + 1):
        e_off = (e_self - k * clock.spacing)[:, None]          # (n_phase, 1) this sun's along-track offset
        S += (_diffuse_flux_xe(x[None, :], e_off, None, None) if diffuse
              else _direct_flux_xe(x[None, :], e_off))
    return S


# ======================================================================================
# Verification against the cosmology doc (run as a test / via __main__).
# ======================================================================================
def verify_against_doc(verbose: bool = True):
    """Assert that the givens reproduce the doc's stated numbers. Returns a dict of checks."""
    out = {}

    # 1) cross-track metric (S4 table) -- L=2L values (suns raised L->2L 2026-06-07; x scales with L_SUN)
    for th_deg, x_doc_km in [(45, 20000), (50, 23800), (70, 55000)]:
        x_km = x_of_theta(np.radians(th_deg)) / 1e3
        out[f"x({th_deg})"] = x_km
        assert abs(x_km - x_doc_km) / x_doc_km < 0.01, (th_deg, x_km, x_doc_km)

    # 2) flux-law identity  cos^3 theta == L^3/(L^2+x^2)^{3/2}
    th = np.radians(np.array([0, 17, 30, 45, 50, 63, 70], float))
    ident = np.max(np.abs(np.cos(th) ** 3 - flux_shape_of_x(x_of_theta(th))))
    out["flux_law_identity_maxabs"] = ident
    assert ident < 1e-12, ident

    # 3) suns' clock from the midnight floor (S6 / Appendix C) at L=2L: D ~410,000 km, v ~4.75 km/s
    #    (both double with L_SUN; day held at 24 h; the flux-law floor ratio is L-invariant)
    out["spacing_km"] = CLOCK.spacing_km
    out["speed_km_s"] = CLOCK.speed_km_s
    assert abs(CLOCK.spacing_km - 410000) / 410000 < 0.02, CLOCK.spacing_km
    assert abs(CLOCK.speed_km_s - 4.75) / 4.75 < 0.05, CLOCK.speed_km_s

    # 4) absolute flux amplitude matches the doc's stated fluxes (S6): ~3300 W/m^2 at theta=50 noon,
    #    and the equator noon is ">20x Earth".
    #    ⚠️ BASIS: these fluxes are INCIDENT (top-of-atmosphere) -- this module applies only geometry,
    #    the lens, and the slant haze; ALBEDO_ATM / K_WV_SW_NIR / cloud are applied DOWNSTREAM in
    #    radiation.py, which receives this very quantity as its ``S_toa``. The ">20x Earth" pin below
    #    therefore divides an incident flux by Earth's SURFACE clear noon (1000 W/m^2) -- a basis
    #    mismatch inherited from an older doc edition. It is KEPT AS-IS because it is a legacy pin on
    #    the nominal L=2e7 world (asserted only with the collimator OFF) and moving it would silently
    #    re-baseline the gate; the like-for-like TOA ratio is reported alongside it instead. The DOC's
    #    "times Earth" figures were re-based to the 1361 W/m^2 solar constant on 2026-07-26
    #    (flatworld/VALIDATION.md, "Finding (flux basis)"). Do not copy the /1000 form into new code.
    x50 = x_of_theta(np.radians(50.0))
    F_noon_50 = _flux_one_sun(slant_distance(x50, 0.0))
    F_noon_eq = _flux_one_sun(slant_distance(0.0, 0.0))
    out["F_noon_theta50"] = F_noon_50
    out["F_noon_equator"] = F_noon_eq
    out["equator_noon_over_earth"] = F_noon_eq / 1000.0        # LEGACY basis-mismatched pin (see above)
    out["equator_noon_over_earth_toa"] = F_noon_eq / 1361.0    # like-for-like: incident vs solar constant
    if not C.COLLIMATOR:                                    # the doc fluxes are the canonical (no-collimator) world;
        assert abs(F_noon_50 - 3300) / 3300 < 0.05, F_noon_50   # the thin-haze + I0 re-pin shift these (a finding,
        assert F_noon_eq / 1000.0 > 20.0, F_noon_eq             # not a regression) -- checked separately for the gate.

    # 5) day-mean flux S(theta) falls off correctly (monotone decreasing away from the equator) and
    #    the diurnal-cycle time-mean reproduces the analytic day-mean (self-consistency).
    xs = x_of_theta(np.radians(np.array([0, 30, 45, 50, 60, 70], float)))
    S = day_mean_flux(xs)
    out["S_daymean_Wm2"] = dict(zip([0, 30, 45, 50, 60, 70], np.round(S, 2)))
    assert np.all(np.diff(S) < 0), S                       # strictly decreasing with latitude
    t, Sc = diurnal_flux_cycle(float(x50), n_steps=4000, n_suns=8)
    S_mean_cycle = np.trapezoid(Sc, t) / CLOCK.T_day
    out["daymean_vs_cycle_rel"] = abs(S_mean_cycle - day_mean_flux_scalar(float(x50))) / day_mean_flux_scalar(float(x50))
    assert out["daymean_vs_cycle_rel"] < 1e-3, out["daymean_vs_cycle_rel"]

    # 6) haze forward-scatter diffuse flux (Stage 5): a POSITIVE recovered fraction of the direct day-mean,
    #    growing toward the dimmer (higher-airmass) latitudes. It only ADDS surface SW; the direct flux above
    #    and the clock are untouched (separate code paths).
    S_diff = day_mean_diffuse_flux(xs)
    out["S_diffuse_Wm2"] = dict(zip([0, 30, 45, 50, 60, 70], np.round(S_diff, 2)))
    out["diffuse_frac_theta50"] = float(S_diff[3] / S[3])
    assert np.all(S_diff > 0), S_diff
    assert (S_diff[-1] / S[-1]) > (S_diff[0] / S[0])        # diffuse/direct ratio higher at the dim pole

    if verbose:
        print("== flatworld.geometry: verification against cosmology.md ==")
        print(f"  metric x(theta):  45->{out['x(45)']:.0f}  50->{out['x(50)']:.0f}  70->{out['x(70)']:.0f} km")
        print(f"  flux-law identity cos^3 == L^3/(L^2+x^2)^3/2 : max|diff| = {ident:.1e}")
        print(f"  suns' clock: spacing D = {CLOCK.spacing_km:,.0f} km, speed v = {CLOCK.speed_km_s:.3f} km/s,"
              f" day = {CLOCK.T_day/3600:.2f} h")
        print(f"  absolute flux: noon@theta50 = {F_noon_50:.0f} W/m^2 (doc ~3300),"
              f" equator noon = {F_noon_eq:.0f} W/m^2 = {F_noon_eq/1000:.0f}x Earth (doc >20x)")
        print(f"  day-mean S(theta) [W/m^2]: " +
              "  ".join(f"{d}d={s:.0f}" for d, s in zip([0,30,45,50,60,70], S)))
        print(f"  diurnal-cycle mean reproduces analytic day-mean to {out['daymean_vs_cycle_rel']:.1e} (rel)")
        print("  ALL CHECKS PASSED")
    return out


if __name__ == "__main__":
    verify_against_doc()
