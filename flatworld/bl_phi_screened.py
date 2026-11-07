"""
flatworld.bl_phi_screened -- a SCREENED-POISSON boundary-layer geopotential Phi for the (x, e) maps.

WHY (roadmap C1, ``ROADMAP_BC_moisture_circulation_feedback.md``). The shipped Stage-C geopotential
(``bl_forcing.geopotential``) builds Phi as a thermal-low field that is then **Gaussian-smoothed** over the
frictional fetch ``phi_smooth_km ~ 400 km``. That is fine for a *static* thermal-only Phi, but the
moisture<->circulation feedback (Track C) needs to add a **latent-heating** source on the SAME footing as
the thermal one -- and a Gaussian-smoothed thermal term and a latent point/line source are NOT on the same
amplitude/roll-off footing even at matched length (the Gaussian closure simply cannot reproduce the
screened-Poisson Green's function). So both forcings must pass through ONE elliptic inverse:

    (lap - lam^2) Phi = src_thermal + src_latent          [the screened Poisson / modified Helmholtz op]

This module is that inverse plus the two source builders. It is a **new** solver, NOT an extension of
``bl_forcing`` -- the Gaussian and the screened-Poisson smoothers have genuinely different kernels (a
Gaussian vs the modified Bessel ``K_0``), and that difference is the point.

THE OPERATOR.  ``(lap - lam^2)`` is the modified-Helmholtz / screened-Poisson (a.k.a. Yukawa) operator.
Its 2-D free-space Green's function -- the response to a unit point source ``(lap - lam^2)G = -delta`` --
is ``G(r) = (1/2pi) K_0(lam r)`` with ``K_0`` the order-0 modified Bessel function of the second kind. So
the inverse acts as a low-pass smoother whose kernel is ``K_0`` and whose length scale is the **screening
length** ``lam^-1``. Two consequences we lean on:
  * **No null space.** Unlike pure Poisson, the screening term ``-lam^2`` makes the operator negative-
    definite (every eigenvalue ``<= -lam^2 < 0``), so it inverts cleanly even under all-periodic BCs -- no
    zero-mode/gauge handling. The spectral solve is a single divide.
  * **``K_0`` is LOG-SINGULAR at a point source** (``K_0(z) ~ -ln(z)`` as ``z->0``). A sharp rain cell or a
    coastline jump in the source therefore drives a grid-scale ``Phi`` cusp (and a spurious ``lap(Phi)``
    ascent spike that SHARPENS with resolution). The fix -- the elliptic restatement of the issue-#82
    coastline field-smoothing -- is to **smooth both sources to a finite physical size before the
    inversion** (``smooth_source`` / the ``source_smooth_km`` argument). The screening-length match does
    NOT cover this: it is a *shape* (de-singularization) requirement, not an amplitude one.

THE SCREENING LENGTH (the 400 km consistency check).  The damped f=0 boundary-layer balance ``V = -grad
Phi / r_fric`` with a Newtonian relaxation of Phi toward its thermal-equilibrium value gives ``lap Phi -
lam^2 Phi = -lam^2 Phi_eq`` with ``lam^-1 = c / r_fric`` -- ``c`` the boundary-layer density-current
(reduced-gravity gravity-wave) speed, NOT a Rossby radius (f=0, ``L_d -> inf``). We pin ``c`` from PHYSICS
-- the density-current speed of a thermal-low geopotential anomaly, ``c = sqrt(2 beta_T dT_char)`` (the
same ``c^2 = 2*dPhi`` Bernoulli speed ``bl_momentum`` uses) with ``dT_char`` the characteristic land/sea
thermal contrast (~30 K) -- and confirm it lands at ``lam^-1 ~ 400 km``, i.e. the same fetch the Gaussian
build smooths over. Crucially ``c`` is pinned from the thermal contrast, NOT inherited from ``wind_cap``
(a numerical validity edge that happens to be ~40 m/s for the same reason). That the two agree is the
*consistency check*, not the definition. **This pin is order-of-magnitude (good to ~sqrt(2)), not 6%**
(PR #162): ``lam^-1 = c/r_fric`` takes the thermal-relaxation rate equal to the momentum-drag rate (two
distinct epsilons both set to ``r_fric``), and ``c = sqrt(2 beta_T dT_char)`` carries a density-current
sqrt(2) (front vs. linear gravity-wave speed). The 424-vs-400 closeness is within that band -- and the
*shipped* Gaussian length is literally ``wind_cap/r_fric = 400 km``, numerically near-degenerate with this
pin; the point is the SCALE agrees, derived independently of the cap.

THE TWO SOURCES (on one footing).  Both are written as ``src = -lam^2 * Phi_eq`` so the inverse reproduces
``Phi_eq`` at large scale (low-k limit ``Phi_hat -> Phi_eq_hat``) -- a Lorentzian-smoothed thermal/latent
equilibrium field, the screened-Poisson analog of the Gaussian build:
  * **thermal** ``Phi_eq_thermal = -beta_T (T_v - Tref) + g_terrain * g * h`` (== ``bl_forcing``'s
    pre-smoothing field FOR A MATCHED ``g_terrain``; note both default to 0 here, matching the live
    ``StageCConfig.g_terrain=0``, whereas ``bl_forcing.geopotential``'s signature default is still 0.6);
  * **latent**  ``Phi_eq_latent  = -gms_geopot * (P - P_baseline)`` -- a precip ANOMALY lowers Phi like a
    warm anomaly. ``gms_geopot`` [m^2/s^2 per mm/day] is the latent-heating->geopotential efficiency
    (i.e. ``L * GMS``); it is **C2's to derive/freeze/lock** from the AD-core gross moist stability, NOT a
    knob to fit here. The **anomaly** ``P - P_baseline`` (not full ``P``) is used because the baseline
    latent heating is already carried by ``T_map_C`` -- else the gain is biased high (roadmap C/C2).

SCOPE (C1).  This module is the SOLVER + the two source builders + the gates (``K_0`` recovery, the 400 km
screening-length consistency, bounded source response). It is **not yet wired into** ``stage_c_map.step``
-- closing the moisture->P->src_latent->Phi->wind loop is C3, and the GMS coefficient is C2. The default
build is **thermal-only** (``gms_geopot = 0`` -> latent inert), a drop-in alternative to
``bl_forcing.geopotential`` that leaves the shipped Stage-C path byte-identical.

CAVEAT (variable c).  The constant-``lam^2`` solve here is **spectral** (DCT in the walled cross-track x,
FFT in the periodic along-track e) -- a single O(N log N) transform-divide-transform, and it admits the
analytic ``K_0`` gate. If ``c`` (hence ``lam``) were made to vary in space (e.g. with the local stability),
the operator loses its constant-coefficient spectral diagonalization: it would need a sparse direct/iterative
solve and the analytic ``K_0`` gate no longer applies. Flagged, not pursued -- the linear build is
constant-``lam^2``.

Abbreviations: BL = boundary layer; ``lam`` = screening rate [1/m], ``lam^-1`` the screening LENGTH;
``r_fric`` = the BL linear drag rate [1/s]; GMS = gross moist stability; ``K_0`` = order-0 modified Bessel
function of the second kind; DCT/FFT = discrete cosine / fast Fourier transform; ``Tref`` = the per-row
(zonal-mean) reference; ``Phi_eq`` = a forcing's thermal-equilibrium geopotential.
"""
from __future__ import annotations
import numpy as np
import scipy.ndimage as ndi
from scipy.fft import dct, idct
from . import constants as C
from . import bl_forcing as blf

# --- the physical pin of the screening length lam^-1 = c/r_fric ----------------------------------------
# c is the BL density-current (reduced-gravity gravity-wave) speed of a thermal-low geopotential anomaly,
# c = sqrt(2 beta_T dT_char) -- the SAME c^2 = 2*dPhi Bernoulli speed bl_momentum uses, with dPhi ~ beta_T
# * dT_char. dT_char is the characteristic land/sea thermal contrast that drives the BL flow; the docs put
# it at ~30-60 K, and the conservative low end (30 K) gives c ~ 42 m/s and lam^-1 ~ 424 km ~ 400 km. This
# is pinned from the THERMAL CONTRAST, independently of wind_cap = 40 m/s (a numerical cap that lands near
# the same value for the same density-current reason). DERIVED, never fitted to a band magnitude.
DENSITY_CURRENT_DT_CHAR = 30.0     # characteristic thermal-low (land/sea) contrast [K]
R_FRIC_DEFAULT = 1.0e-4            # BL linear drag rate [1/s] (matches StageCConfig.r_fric)
BETA_T_DEFAULT = 30.0             # thermal-low strength [m^2/s^2/K] (matches StageCConfig.beta_T)

# Default source de-singularization length, as a FRACTION of the screening length lam^-1 (PR #162 review).
# The screened inversion ALREADY supplies the physical ~lam^-1 (~400 km) smoothing -- that is what the K_0
# kernel IS -- so the source pre-smooth must only CAP the K_0 log-cusp at a sharp rain cell / coastline jump,
# NOT re-apply the full fetch. Two constraints fix it: (i) it must be SMALL vs lam^-1 or it double-counts the
# inversion's own smoothing and attenuates the thermal Phi 2-7x at land/sea scales (the review's measured
# finding); (ii) it must be a FIXED PHYSICAL length, NOT grid-scale -- a grid-scale (few-cell) smooth is
# resolution-DEPENDENT, so the cusp re-sharpens with resolution (the issue-#82 sharpening the smoothing exists
# to prevent; verified spread ~1.0 over 64->256). lam^-1/4 ~ 106 km satisfies both: thermal Phi tracks the
# Gaussian-400 build to ~0.76-0.93 of amplitude (an honest drop-in) AND the point-source peak is
# resolution-independent (spread ~0.05 over 64->256). NOT a free fit -- a de-singularization scale.
SOURCE_DESING_FRAC = 0.25         # default source_smooth_km = SOURCE_DESING_FRAC * lam^-1 (~106 km)


def density_current_speed(beta_T: float = BETA_T_DEFAULT,
                          dT_char: float = DENSITY_CURRENT_DT_CHAR) -> float:
    """BL density-current speed ``c = sqrt(2 * beta_T * dT_char)`` [m/s] -- the reduced-gravity
    gravity-wave / Bernoulli speed of a thermal-low geopotential anomaly of amplitude ``beta_T*dT_char``
    (the same ``c^2 = 2*dPhi`` form ``bl_momentum`` uses). Pinned from the thermal contrast, NOT ``wind_cap``."""
    return float(np.sqrt(2.0 * beta_T * dT_char))


def screening_length_km(c: float | None = None, r_fric: float = R_FRIC_DEFAULT,
                        beta_T: float = BETA_T_DEFAULT, dT_char: float = DENSITY_CURRENT_DT_CHAR) -> float:
    """Screening LENGTH ``lam^-1 = c / r_fric`` in km. ``c`` defaults to the physically-pinned
    ``density_current_speed(beta_T, dT_char)``. With the defaults this is ~424 km -- the free consistency
    check that the screened-Poisson fetch matches the Gaussian build's ``phi_smooth_km ~ 400 km``."""
    c = density_current_speed(beta_T, dT_char) if c is None else c
    return c / r_fric / 1.0e3


def screening_rate(c: float | None = None, r_fric: float = R_FRIC_DEFAULT,
                  beta_T: float = BETA_T_DEFAULT, dT_char: float = DENSITY_CURRENT_DT_CHAR) -> float:
    """Screening RATE ``lam = 1 / (screening length)`` [1/m] (the ``lam`` in ``lap - lam^2``)."""
    return 1.0 / (screening_length_km(c, r_fric, beta_T, dT_char) * 1.0e3)


# --- the spectral screened-Poisson inverse -------------------------------------------------------------
def _axis_eigs(n: int, d: float, bc: str) -> np.ndarray:
    """Eigenvalues of the 1-D second-difference operator along an axis of ``n`` points, spacing ``d``,
    diagonalized by the matching transform:
      * ``periodic`` -> FFT, mode k:  -(2/d^2)(1 - cos(2 pi k / n))   ( = -(4/d^2) sin^2(pi k / n) );
      * ``neumann``  -> DCT-II, mode j: -(2/d^2)(1 - cos(pi j / n))   (zero-gradient / 'nearest' walls).
    All ``<= 0``; with the ``-lam^2`` screening the full 2-D symbol is strictly negative (invertible)."""
    if bc == "periodic":
        k = np.fft.fftfreq(n)                      # cycles per sample in [0, .5) U [-.5, 0)
        return -(2.0 / d ** 2) * (1.0 - np.cos(2.0 * np.pi * k))
    if bc == "neumann":
        j = np.arange(n)
        return -(2.0 / d ** 2) * (1.0 - np.cos(np.pi * j / n))
    raise ValueError(f"bc must be 'periodic' or 'neumann', got {bc!r}")


def solve_screened_poisson(src, dx, de, lam, bc_x: str = "neumann"):
    """Solve ``(lap - lam^2) Phi = src`` on the (x, e) grid by a constant-coefficient spectral inverse.

    Conventions match the moisture operators (``flatworld.operators``): along-track ``e`` is **periodic**
    (FFT); cross-track ``x`` has **walls** -- ``bc_x='neumann'`` (DEFAULT, zero-gradient / the 'nearest'
    Gaussian-smoothing wall) via a DCT-II, or ``bc_x='periodic'`` (FFT, used only for the analytic ``K_0``
    gate on a large domain). The eigenvalue denominator ``ax + ae - lam^2`` is strictly negative, so this
    is a single divide -- no null space, no iterations.

    src : (nx, ne) real source array (the SUM of the smoothed thermal + latent sources).
    lam : screening rate [1/m] (``screening_rate``).
    Returns Phi [same units as src * m^2] -- for src in [1/s^2], Phi in [m^2/s^2]."""
    src = np.asarray(src, float)
    nx, ne = src.shape
    ae = _axis_eigs(ne, de, "periodic")
    ax = _axis_eigs(nx, dx, bc_x)
    denom = ax[:, None] + ae[None, :] - lam ** 2          # strictly < 0 everywhere
    if bc_x == "neumann":
        S = np.fft.fft(dct(src, type=2, norm="ortho", axis=0), axis=1)
        Phi = idct(np.fft.ifft(S / denom, axis=1).real, type=2, norm="ortho", axis=0)
    else:  # periodic x
        Phi = np.fft.ifft2(np.fft.fft2(src) / denom).real
    return Phi


def screened_laplacian(F, dx, de, lam, bc_x: str = "neumann"):
    """Apply the SAME discrete operator ``(lap - lam^2)`` the spectral solver inverts: a clean
    second-difference with ``e`` periodic and ``x`` either zero-gradient ('neumann', mirror ``F[-1]=F[0]``)
    or periodic. Used (a) as the round-off self-consistency check ``screened_laplacian(solve(src)) == src``
    and (b) as the ``lap(Phi)`` ascent diagnostic. NOTE this differs from ``operators.lap`` at the x-walls
    (which zeroes the wall second-derivative for the small eddy-diffusion term); this operator's wall
    stencil is the one the DCT-II inverts, so the two are exact inverses."""
    F = np.asarray(F, float)
    d2e = (np.roll(F, -1, axis=1) - 2.0 * F + np.roll(F, 1, axis=1)) / de ** 2
    d2x = np.zeros_like(F)
    d2x[1:-1, :] = (F[2:, :] - 2.0 * F[1:-1, :] + F[:-2, :]) / dx ** 2
    if bc_x == "neumann":
        d2x[0, :] = (F[1, :] - F[0, :]) / dx ** 2          # mirror F[-1]=F[0]  (zero-gradient wall)
        d2x[-1, :] = (F[-2, :] - F[-1, :]) / dx ** 2
    elif bc_x == "periodic":
        d2x[0, :] = (F[1, :] - 2.0 * F[0, :] + F[-1, :]) / dx ** 2
        d2x[-1, :] = (F[0, :] - 2.0 * F[-1, :] + F[-2, :]) / dx ** 2
    else:
        raise ValueError(f"bc_x must be 'periodic' or 'neumann', got {bc_x!r}")
    return d2x + d2e - lam ** 2 * F


# --- source builders + de-singularizing smoothing ------------------------------------------------------
def smooth_source(src, dx, de, smooth_km):
    """Gaussian-smooth a forcing source to a finite physical size ``smooth_km`` BEFORE the inversion --
    the de-singularization that keeps a sharp rain cell / coastline jump from driving a grid-scale ``K_0``
    cusp (the elliptic restatement of the issue-#82 coastline fix). BCs match the operators / the inverse:
    'nearest' (clamped) across the cross-track x-walls, 'wrap' (periodic) along e. ``smooth_km <= 0`` is a
    no-op pass-through."""
    if smooth_km is None or smooth_km <= 0.0:
        return np.asarray(src, float)
    sig = (max(smooth_km / (dx / 1e3), 1e-6), max(smooth_km / (de / 1e3), 1e-6))
    return ndi.gaussian_filter(np.asarray(src, float), sigma=sig, mode=("nearest", "wrap"))


def thermal_equilibrium_phi(Tv_map_C, height, beta_T=BETA_T_DEFAULT, g_terrain=0.0):
    """The thermal-low equilibrium geopotential ``Phi_eq = -beta_T (T_v - Tref) + g_terrain * g * h``
    [m^2/s^2] -- the pre-smoothing field inside ``bl_forcing.geopotential`` FOR A MATCHED ``g_terrain``
    (warm surface -> low Phi; high terrain -> high Phi; ``Tref`` the per-row zonal mean). NB the default
    ``g_terrain=0`` matches the live ``StageCConfig.g_terrain=0``, but ``bl_forcing.geopotential``'s
    signature still defaults to 0.6, so the two coincide only when passed the same ``g_terrain``. Pass a
    VIRTUAL temperature map (``bl_forcing.virtual_T_map``) for the moist-buoyancy build, or dry T for the
    legacy form."""
    Tv = np.asarray(Tv_map_C, float)
    Tref = np.mean(Tv, axis=1, keepdims=True)
    return -beta_T * (Tv - Tref) + g_terrain * C.G_ACCEL * np.asarray(height, float)


def latent_equilibrium_phi(P, P_baseline=None, gms_geopot=0.0):
    """The latent-heating equilibrium geopotential ``Phi_eq = -gms_geopot * (P - P_baseline)`` [m^2/s^2]:
    a precip ANOMALY lowers Phi like a warm anomaly (latent heating -> thermal low -> more ascent, a
    positive feedback). ``P``/``P_baseline`` in [mm/day]; ``gms_geopot`` [m^2/s^2 per mm/day] is the
    latent-heating->geopotential efficiency (= ``L * GMS``), now DERIVED + frozen by C2 as
    ``moist_stability.GMS_GEOPOT_FROZEN`` (~4.7); default here stays 0 (latent inert) until C3 wires that
    value -- still default-OFF until J1. The ANOMALY (not full P) is used so the baseline latent heating
    already in ``T_map_C`` is not double-counted (roadmap C/C2). ``P_baseline=None`` -> zero anomaly.

    GUARD (PR #162 review): a non-zero ``gms_geopot`` with ``P`` supplied but NO ``P_baseline`` is almost
    certainly a C3 wiring slip -- it would silently return a zero (thermal-only) Phi, the feedback lost with
    no error. So that combination RAISES; pass ``P_baseline`` explicitly (e.g. the AD-core precip baked into
    ``T_map_C``, or ``0.0`` for a full-P drive) to opt in."""
    P = np.asarray(P, float)
    if P_baseline is None:
        if gms_geopot != 0.0:
            raise ValueError("latent forcing requested (gms_geopot != 0) but P_baseline is None -- the "
                             "anomaly zero-point must be set explicitly (the C3 double-count guard); pass "
                             "P_baseline (the T_map_C-consistent precip, or 0.0 to drive on full P).")
        return np.zeros_like(P)
    return -gms_geopot * (P - np.asarray(P_baseline, float))


def geopotential_screened(Tv_map_C, height, dx, de, *, P=None, P_baseline=None, gms_geopot=0.0,
                          beta_T=BETA_T_DEFAULT, g_terrain=0.0, r_fric=R_FRIC_DEFAULT,
                          dT_char=DENSITY_CURRENT_DT_CHAR, c=None, lam=None,
                          source_smooth_km=None, bc_x="neumann", return_parts=False):
    """Build the boundary-layer geopotential ``Phi`` [m^2/s^2] through the ONE screened-Poisson inverse,
    summing the thermal and latent forcings on a single footing (roadmap C1). Drop-in alternative to
    ``bl_forcing.geopotential``; **thermal-only by default** (``gms_geopot=0`` -> latent inert), so the
    shipped Stage-C path is untouched.

    Pipeline:  Phi_eq = thermal + latent  ->  src = -lam^2 Phi_eq  ->  smooth(src)  ->  (lap - lam^2)^-1.
    The ``-lam^2 Phi_eq`` form makes the inverse reproduce ``Phi_eq`` at large scale (a Lorentzian-smoothed
    thermal/latent field, the screened analog of the Gaussian build). The inversion's K_0 kernel ALREADY
    supplies the physical ~lam^-1 smoothing, so the source pre-smooth only CAPS the K_0 log-cusp at sharp
    rain cells / coastline jumps -- at the small default ``SOURCE_DESING_FRAC*lam^-1`` the thermal Phi
    tracks the Gaussian-400 build's amplitude to ~0.76-0.93 (an honest quantitative drop-in, PR #162),
    distinct from the genuine K_0-vs-Gaussian kernel shape difference.

      lam            : screening rate [1/m]; default ``screening_rate(c, r_fric, beta_T, dT_char)``
                       (lam^-1 ~ 424 km). Pass ``c`` to override the density-current speed, or ``lam`` direct.
      source_smooth_km: de-singularization size [km]; default ``SOURCE_DESING_FRAC * lam^-1 ~ 106 km`` -- a
                       FIXED PHYSICAL length SMALL vs lam^-1. Do NOT set it to ~lam^-1 (double-counts the
                       inversion's own smoothing -> thermal Phi attenuated 2-7x at land/sea scales) NOR to a
                       grid-scale (few-cell) value (resolution-dependent -> the cusp re-sharpens with the
                       grid, the issue-#82 failure). See ``SOURCE_DESING_FRAC``.
      return_parts   : also return a dict with Phi_thermal, Phi_latent, lam, source_smooth_km, src.

    Pass ``Tv_map_C`` already in virtual temperature for the moist build (``bl_forcing.virtual_T_map``)."""
    if lam is None:
        lam = screening_rate(c=c, r_fric=r_fric, beta_T=beta_T, dT_char=dT_char)
    if source_smooth_km is None:
        source_smooth_km = SOURCE_DESING_FRAC / lam / 1.0e3      # small de-singularization scale (~lam^-1/4)
    phi_eq_thermal = thermal_equilibrium_phi(Tv_map_C, height, beta_T=beta_T, g_terrain=g_terrain)
    phi_eq_latent = latent_equilibrium_phi(P, P_baseline, gms_geopot) if P is not None \
        else np.zeros_like(phi_eq_thermal)
    src = -lam ** 2 * (phi_eq_thermal + phi_eq_latent)
    src = smooth_source(src, dx, de, source_smooth_km)
    Phi = solve_screened_poisson(src, dx, de, lam, bc_x=bc_x)
    if not return_parts:
        return Phi
    # the thermal/latent split: invert each smoothed source alone (the operator is linear, so they sum to Phi)
    src_t = smooth_source(-lam ** 2 * phi_eq_thermal, dx, de, source_smooth_km)
    src_l = smooth_source(-lam ** 2 * phi_eq_latent, dx, de, source_smooth_km)
    parts = dict(Phi_thermal=solve_screened_poisson(src_t, dx, de, lam, bc_x=bc_x),
                 Phi_latent=solve_screened_poisson(src_l, dx, de, lam, bc_x=bc_x),
                 lam=lam, screening_length_km=1.0 / lam / 1.0e3, source_smooth_km=source_smooth_km, src=src)
    return Phi, parts


# --- the analytic K_0 Green's function (the C1 gate reference) ------------------------------------------
def greens_function_k0(r, lam):
    """The analytic 2-D screened-Poisson Green's function for a UNIT POINT SOURCE: the response to
    ``(lap - lam^2) Phi = +delta`` is ``Phi(r) = -(1/2pi) K_0(lam r)`` (since ``(lap-lam^2)[(1/2pi)K_0(lam
    r)] = -delta``). ``solve_screened_poisson`` fed a unit point source (one cell = ``1/(dx de)``) must
    recover this at radii >> grid spacing and << domain (where discreteness and the box BC are negligible)."""
    from scipy.special import k0
    r = np.asarray(r, float)
    return -(1.0 / (2.0 * np.pi)) * k0(lam * np.maximum(r, 1e-30))


def _demo():
    """Print the three C1 gates' numbers (cheap; ``python -m flatworld.bl_phi_screened``)."""
    from scipy.special import k0
    lam = screening_rate()
    Lkm = screening_length_km()
    print("== C1 screened-Poisson Phi gates ==")
    print(f"density-current speed c = sqrt(2*beta_T*dT_char) = {density_current_speed():.2f} m/s "
          f"(beta_T={BETA_T_DEFAULT}, dT_char={DENSITY_CURRENT_DT_CHAR} K)")
    print(f"screening length lam^-1 = c/r_fric = {Lkm:.0f} km  (target ~400 km; from PHYSICS, not wind_cap)")

    # K_0 recovery on a large periodic domain
    N, Ldom = 256, 24e5
    dxx = Ldom / N
    src = np.zeros((N, N)); src[N // 2, N // 2] = 1.0 / (dxx * dxx)
    Phi = solve_screened_poisson(src, dxx, dxx, lam, bc_x="periodic")
    ii, jj = np.indices((N, N))
    r = np.hypot((ii - N // 2) * dxx, (jj - N // 2) * dxx)
    m = (r > 4 * dxx) & (r < 1.2 / lam)
    rel = np.abs(-Phi[m] - (1 / (2 * np.pi)) * k0(lam * r[m])) / np.abs((1 / (2 * np.pi)) * k0(lam * r[m]))
    print(f"K_0 Green's-function recovery: median rel err {np.median(rel) * 100:.1f}% over r in [4dx, 1.2 lam^-1]")

    # bounded source response: peak |Phi| vs resolution, source-smoothed
    print("bounded source (point src, smoothed at lam^-1): peak|Phi| vs resolution")
    for n in (64, 128, 256):
        d = 8e6 / n
        s = np.zeros((n, n)); s[n // 2, n // 2] = 1.0 / (d * d)
        ph = solve_screened_poisson(smooth_source(s, d, d, Lkm), d, d, lam, bc_x="neumann")
        print(f"    res={d/1e3:5.0f} km   peak|Phi|={np.abs(ph).max():.4e}")


if __name__ == "__main__":
    _demo()
