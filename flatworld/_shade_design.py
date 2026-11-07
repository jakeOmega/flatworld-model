"""Graded-collimator ("lantern shade") design tool — Jake's T(α) idea (2026-06-11).

⚠️  SCOPE — DON'T BE MISLED (2026-07-06).  This module is the shared optics KERNEL
    (`_one_sun_total`, `kernel`, `rce_curve`) the LIVE climate path (`geometry` → `cascade`) and the
    doc-figure tools both build on — it is foundational, NOT archivable.  The T(α) it designs is
    AZIMUTHALLY SYMMETRIC: it is the day-MEAN DELIVERY only, and it can only CARVE (0 ≤ T ≤ 1 →
    "brighten the band" = "darken everything else + raise I₀").  It CANNOT redistribute day-mean flux
    cross-track between latitudes (e.g. focus θ<30 onto the equator) without also mangling the
    along-track day (noon spikes at every latitude).  To RESHAPE the lantern cross-track you need the
    physical 2-D REDISTRIBUTING lens T(α, ψ) — the projector that separates cross-track (which
    latitude) from along-track (time of day): see `lantern_wall_figure.py` / `plot_lantern_optic.py`
    (the premise) and `lantern_finite_design.py` (rim/blur realizability).  The radial table here is
    the delivery mechanism, not the physical optic — don't try to focus cross-track with it.

PREMISE OBJECT. Each sun's opaque cylinder becomes a wall with graded transmission T(α), α = angle
from the vertical (a surface of revolution ⇒ zenith-angle-only dependence, exactly the symmetry
Jake specified: 45° elevation at distance r gives the same flux at any latitude/time of day). The
current collimator is the binary special case T = step(α < z_cut). Everything downstream stays
DERIVED physics: the day-mean S(θ) is the same train integral with T(α) in the integrand.

THE LINEAR STRUCTURE. Transmission multiplies flux, so the day-mean is exactly linear in T:
    S(θ) = I0_mult · Σ_j K(θ, α_j) · T_j ,   0 ≤ T_j ≤ 1,
with K built from the model's own optics (1/r³ geometry+projection, thin slant haze, the
forward-scatter diffuse recovery — geometry.py's integrand with an α-bin indicator inserted).
Designing a target S(θ) is bounded linear least squares (scipy lsq_linear) + a smoothness penalty;
the required overall sun brightening I0_mult is read off the solution (T can only carve, never
amplify — "brighten the band" = "darken everything else + raise I₀").

THE FORECAST. Per-column RCE T_surf(S) is anchored EMPIRICALLY to lx1000's steady state (the
all-on model's own RCE-limit world: its T_surf(θ) against the standard day-mean S(θ), which is
L-invariant in θ) — so the band/equator placement forecast inherits the real greenhouse/cloud
behavior, not a hand formula. The cell-drive index ∫(S_abs − OLR(T_band))⁺dx (the heat the strip
would export if circulation flattened it to band temperature) is reported RELATIVE to the current
single-track world — the hurricane forecast. Peak NOON flux at the band is checked against the
human-habitability ceiling (~1.4 kW/m² Earth TOA; ~1 kW/m² Earth surface).

USAGE (from cosmology_models/):
  ../.venv/bin/python -m flatworld._shade_design            # the 3 standing candidates -> PNGs + table
"""
from __future__ import annotations
import os

import numpy as np

from . import constants as C
from . import convergence as conv

SIGMA = 5.670374419e-8
PLOTDIR = os.path.join("m33_runs", "plots")
ALPHA_EDGES = np.arange(0.0, 89.5, 1.0)            # 1-degree shade bins (the design resolution)
THETA_GRID = np.linspace(0.5, 84.5, 141)           # design grid in cross-track theta [deg]


# ----------------------------------------------------------------------------------- the kernel
def _one_sun_total(r, tau0=None):
    """Direct + forward-scattered diffuse flux from one UNSHADED sun at slant distance r — the
    model's own optics (geometry._flux_one_sun + _diffuse_one_sun, gate removed: the shade is the
    new gate and is applied by the caller)."""
    tau0 = C.TAU0_HAZE_THIN if tau0 is None else tau0
    Lr = C.L_SUN / np.asarray(r, float)
    F_ref = C.I0_OVERHEAD * Lr**3
    tau_slant = tau0 * (1.0 / Lr - 1.0)
    direct = F_ref * np.exp(-tau_slant)
    diffuse = F_ref * C.OMEGA0_HAZE * 0.5 * (1.0 + C.G_HAZE) * (1.0 - np.exp(-tau_slant))
    return direct + diffuse


def kernel(theta_deg=THETA_GRID, alpha_edges=ALPHA_EDGES, n=4001):
    """K[i, j] = day-mean flux at theta_i per unit transmission in alpha-bin j [W/m^2].

    Same train unfolding as geometry.day_mean_flux_scalar (one sun over all along-track e, divided
    by the spacing; sinh-stretched grid), with the alpha-bin indicator in the integrand."""
    from .geometry import CLOCK, x_of_theta
    a_lo, a_hi = alpha_edges[:-1], alpha_edges[1:]
    K = np.zeros((len(theta_deg), len(a_lo)))
    u = np.linspace(0.0, 20.0, n)
    for i, th in enumerate(theta_deg):
        x = x_of_theta(np.radians(th))
        A = np.sqrt(C.L_SUN**2 + x**2)
        e = A * np.sinh(u); de_du = A * np.cosh(u)
        r = np.sqrt(C.L_SUN**2 + x**2 + e**2)
        alpha = np.degrees(np.arccos(np.clip(C.L_SUN / r, -1.0, 1.0)))
        base = _one_sun_total(r) * de_du
        for j in range(len(a_lo)):
            sel = (alpha >= a_lo[j]) & (alpha < a_hi[j])
            if sel.any():
                K[i, j] = 2.0 * np.trapezoid(np.where(sel, base, 0.0), u) / CLOCK.spacing
    return K


# ------------------------------------------------------- the RCE anchor (the model's own columns)
ANCHOR_PATH = os.path.join("m33_runs", "rce_anchor.npz")


def build_anchor(n_steps=4500, out=ANCHOR_PATH):
    """The honest S -> T_RCE anchor: the core's own column physics (radiation, clouds, convection,
    the wetness ramp) run THERMO-ONLY over a controlled S grid (each column gets its own fixed S;
    columns are independent, so one run sweeps the whole curve). Replaces the original lx1000
    anchor, which proved INVALID: dry-surface columns trap seeded vapor (no evaporation in, no
    rainout out) and the gray radiation scheme extrapolates pathologically far above its
    ~200-320 K calibration -- lx1000's 491 °C equator was scheme breakdown, not physics. This
    sweep starts from the core's own default (near-dry) init, so columns carry only the vapor
    their own hydrology makes; T tops out ~200 °C (mild extrapolation, flagged) -- the 125 °C
    strip target sits comfortably inside."""
    from . import _m33_m34_run as _m33
    core = _m33._make_core(125, 18, theta_max_deg=86.0)
    S_grid = np.geomspace(2.0, 3000.0, core.cfg.nx)
    core.S = S_grid.copy()
    core.S_sw_in = S_grid.copy()
    if getattr(core, "S_diffuse", None) is not None:
        core.S_diffuse = np.zeros_like(S_grid)                 # S carries the TOTAL design flux
    # WARM-BRANCH INIT (the first cool-init sweep capped every wet column at ~12-15 °C: the
    # single-column shrouded/cloud-lock attractor — the #91 bistability isolated in pure column
    # physics). The world is the WARM basin (Jake's call), so the anchor approaches from a hot,
    # near-dry start: a gray dry-RCE T estimate per column, the reference profile shifted to match.
    T0 = np.clip((S_grid * 0.80 * 1.5 / SIGMA) ** 0.25, 160.0, 480.0)
    core.T_surf = T0.copy()
    prof = np.asarray(core.T_atm[0], float)
    core.T_atm = (T0[:, None] + (prof - prof[-1])[None, :]).clip(140.0, 500.0)
    core.run_dynamics = False
    for _ in range(n_steps):
        core.step()
    st1 = np.asarray(core.state()["T_surf"], float)
    for _ in range(300):                                       # drift check window
        core.step()
    Ts_K = np.asarray(core.state()["T_surf"], float)
    drift = float(np.max(np.abs(Ts_K - st1)))
    Ts = Ts_K - 273.15
    np.savez(out, S=S_grid, T_surf_C=Ts, drift_K_300=drift)
    print(f"[shade_design] anchor {out}: T({S_grid[0]:.0f} W/m²)={Ts[0]:.1f} °C … "
          f"T({S_grid[-1]:.0f})={Ts[-1]:.1f} °C; drift over last 300 steps {drift:.2f} K")
    return S_grid, Ts


def rce_curve():
    """Monotone S -> T_surf [C] interpolant from the thermo-only column anchor (build_anchor);
    analytic gray fallback (clean checkout / tests). Returns (T_of_S, S_of_T, S_lo)."""
    try:
        d = np.load(ANCHOR_PATH)
        S, Ts = np.asarray(d["S"], float), np.asarray(d["T_surf_C"], float)
        Ts = np.maximum.accumulate(Ts)                          # monotone (the wet/dry ramp can jump)
        lo = max(float(S[0]), 1e-2)
    except Exception:                                           # noqa: BLE001 -- fallback: gray RCE
        S = np.geomspace(1.0, 1.2e4, 200)
        Ts = (S * 0.85 * 1.6 / SIGMA) ** 0.25 - 273.15
        lo = 1.0
    def T_of_S(s):
        return np.interp(np.asarray(s, float), S, Ts)
    def S_of_T(t):
        return np.interp(np.asarray(t, float), Ts, S)
    return T_of_S, S_of_T, lo


# ----------------------------------------------------------------------------------- the design
def design(band=(40.0, 50.0), band_T=(5.0, 25.0), strip_half=3.0, strip_T=125.0,
           smooth=0.05, theta=THETA_GRID, K=None, anchors=None, noon_cap=1100.0):
    """Solve for the shade T(α). Two target modes: WINDOWS (furnace strip at strip_T + band window
    at band_T, everything else lightly weighted) or, when ``anchors`` is given, a FULL MONOTONE
    PROFILE: anchors = [(theta_deg, T_C or None-for-dark), ...] interpolated in S-space (weights:
    10 inside the strip and the ribbon's 30->0 fall, 2 elsewhere). Returns a result dict."""
    from scipy.optimize import lsq_linear
    K = kernel(theta) if K is None else K
    T_of_S, S_of_T, _ = rce_curve()
    if anchors is not None:
        athe = np.array([a[0] for a in anchors], float)
        # anchor values: plain number = target T_C (mapped through the RCE curve); ("S", value) =
        # DIRECT flux target [W/m^2] (the post-anchor design language: the column thermostat shelf
        # at S~400-900 -> ~6-15 C makes S-space the robust way to place the ribbon); None = dark.
        def _aval(v):
            if v is None:
                return 0.0
            if isinstance(v, tuple) and v[0] == "S":
                return float(v[1])
            return float(S_of_T(v))
        aS = np.array([_aval(a[1]) for a in anchors], float)
        target = np.interp(theta, athe, aS)
        w = np.full_like(theta, 2.0)
        w[theta <= strip_half] = 10.0
        w[(theta >= band[0]) & (theta <= band[1])] = 10.0
    else:
        target = np.zeros_like(theta)
        w = np.full_like(theta, 0.3)                           # moat & elsewhere: light "don't care"
        strip = theta <= strip_half
        target[strip] = S_of_T(strip_T); w[strip] = 10.0
        in_band = (theta >= band[0]) & (theta <= band[1])
        tb = np.linspace(band_T[1], band_T[0], in_band.sum())  # warm inner edge -> cool outer edge
        target[in_band] = S_of_T(tb); w[in_band] = 10.0
        fringe = theta > band[1] + 8.0
        target[fringe] = 0.0; w[fringe] = 2.0
    nb = K.shape[1]
    D2 = np.diff(np.eye(nb), 2, axis=0)
    A = np.vstack([w[:, None] * K, smooth * np.linalg.norm(K) / nb * D2])
    b = np.concatenate([w * target, np.zeros(D2.shape[0])])
    M = 60.0                                                   # allow up to 60x sun brightening
    # HABITABILITY NOON CAP: instantaneous noon flux at theta is linear in the single bin T(alpha=theta)
    # (noon_j = Traw_j * F_one(L/cos a_j)), so a peak-flux ceiling outside the furnace is an EXACT
    # per-variable upper bound -- no one-sided penalty tricks needed. The strip keeps its scorch.
    centers_b = 0.5 * (ALPHA_EDGES[:-1] + ALPHA_EDGES[1:])
    F_noon = _one_sun_total(C.L_SUN / np.cos(np.radians(np.minimum(centers_b, 88.0))))
    # Cap from the ribbon's inner edge outward: the moat is uninhabitable hot desert by design
    # (supra-Earth noon there is thematic, not a problem); only where people LIVE does peak flux bind.
    alpha_hab = band[0] - 2.0
    cap = np.inf if noon_cap is None else noon_cap              # [W/m^2]; None = uncapped (sharpest edge)
    ub = np.where(centers_b > alpha_hab, np.minimum(M, cap / np.maximum(F_noon, 1e-9)), M)
    sol = lsq_linear(A, b, bounds=(0.0, ub), tol=1e-10, max_iter=500)
    Traw = sol.x
    I0_mult = max(float(Traw.max()), 1e-9)
    shade = Traw / I0_mult
    S_ach = K @ Traw
    Tfc = T_of_S(np.maximum(S_ach, 1e-2))
    # metrics
    from .geometry import x_of_theta
    x = x_of_theta(np.radians(theta))
    alb = 0.15
    def drive(S_prof, T_band_ref):
        OLRb = SIGMA * (T_band_ref + 273.15) ** 4 * 0.65       # crude effective emission for the index
        ex = np.maximum(S_prof * (1 - alb) - OLRb, 0.0)
        return float(np.trapezoid(ex, x))
    from .geometry import day_mean_flux
    S_now = day_mean_flux(x)
    T_mid = 0.5 * (band_T[0] + band_T[1])
    a_cent = np.degrees(np.arctan(x / C.L_SUN))
    centers = 0.5 * (ALPHA_EDGES[:-1] + ALPHA_EDGES[1:])
    noon = I0_mult * np.interp(a_cent, centers, shade) * _one_sun_total(C.L_SUN / np.cos(np.radians(a_cent)))
    i30 = np.where((Tfc[:-1] > 30) & (Tfc[1:] <= 30))[0]
    i0c = np.where((Tfc[:-1] > 0) & (Tfc[1:] <= 0))[0]
    bandfc = (float(theta[i30[-1]]) if len(i30) else np.nan, float(theta[i0c[-1]]) if len(i0c) else np.nan)
    bw_km = (np.tan(np.radians(bandfc[1])) - np.tan(np.radians(bandfc[0]))) * C.L_SUN / 1e3 \
        if np.isfinite(bandfc).all() else np.nan
    return dict(theta=theta, target=target, S=S_ach, shade=shade, alpha=centers, I0_mult=I0_mult,
                T_forecast=Tfc, band_fc=bandfc, band_w_km=bw_km, T_eq=float(Tfc[0]),
                drive_rel=drive(S_ach, T_mid) / max(drive(S_now, T_mid), 1e-9),
                noon_band=float(np.interp(np.mean(band), theta, noon)),
                noon_eq=float(noon[0]), residual=float(np.sqrt(np.mean((S_ach - target)[w >= 10] ** 2))))


CANDIDATES = dict(
    A_doc_classic=dict(band=(45.0, 53.0), band_T=(2.0, 28.0), strip_half=3.0, strip_T=125.0),
    B_inner_band=dict(band=(33.0, 42.0), band_T=(2.0, 28.0), strip_half=3.0, strip_T=125.0),
    C_wide_temperate=dict(band=(38.0, 52.0), band_T=(0.0, 30.0), strip_half=5.0, strip_T=115.0),
    # D: the ORIGINAL doc geography as a full monotone profile — boiling strip, hot-desert decline,
    # a NARROW ribbon falling 30->0 across theta 45-50, then dark to the cryo fringe.
    D_monotone_ribbon=dict(band=(45.0, 50.0), strip_half=3.0,
                           anchors=[(0.0, 125.0), (3.0, 125.0), (15.0, 85.0), (30.0, 60.0),
                                    (42.0, 38.0), (45.0, 30.0), (50.0, 0.0), (56.0, -60.0),
                                    (70.0, None), (84.5, None)]),
    # D2: the SHELF design (post-anchor language, direct S-space): strip S~1800 (robustly on the
    # dry-hot branch -> ~125-150 C salt furnace), monotone decline through the bistable moat
    # (geologically dry per the evaporite/sill argument), ribbon = the cloud-thermostat shelf
    # window S 550-800 at theta 45-50 (the shelf pins T at ~6-15 C with HUGE tolerance), the cold
    # edge = the intrinsic shelf break at the S~350 crossing, dark beyond.
    D2_shelf=dict(band=(45.0, 50.0), strip_half=3.0,
                  anchors=[(0.0, ("S", 1800.0)), (3.0, ("S", 1800.0)), (15.0, ("S", 1350.0)),
                           (30.0, ("S", 1050.0)), (42.0, ("S", 850.0)), (45.0, ("S", 700.0)),
                           (50.0, ("S", 350.0)), (56.0, ("S", 60.0)), (70.0, None), (84.5, None)]),
    # D3: the WIDE-STRIP variant (overnight 2026-06-11: the +-3 deg strip is ~1 grid cell at any L
    # with the standard nx/theta grids -- edge-mixing erodes it; the L8 pair's equator dipped to
    # ~75 C. +-6 deg = ~2 cells at nx250 and resists the erosion; rest of the profile unchanged.)
    D3_widestrip=dict(band=(45.0, 50.0), strip_half=6.0,
                      anchors=[(0.0, ("S", 1800.0)), (6.0, ("S", 1800.0)), (16.0, ("S", 1350.0)),
                               (30.0, ("S", 1050.0)), (42.0, ("S", 850.0)), (45.0, ("S", 700.0)),
                               (50.0, ("S", 350.0)), (56.0, ("S", 60.0)), (70.0, None), (84.5, None)]),
)


def render(name, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 10))
    axes[0].plot(res["theta"], res["target"], "k:", label="target S(θ)")
    axes[0].plot(res["theta"], res["S"], "g-", lw=2, label="achievable S(θ)")
    axes[0].set_ylabel("day-mean S [W/m²]"); axes[0].set_yscale("symlog", linthresh=10)
    axes[1].plot(res["alpha"], res["shade"], "b-", lw=2)
    axes[1].set_ylabel("shade T(α) [0–1]"); axes[1].set_xlabel("α from vertical [deg]")
    axes[1].set_title(f"sun brightening required: I0 × {res['I0_mult']:.1f}", fontsize=9)
    axes[2].plot(res["theta"], res["T_forecast"], "r-", lw=2)
    axes[2].axhspan(0, 30, color="green", alpha=0.10)
    axes[2].set_ylabel("RCE-forecast T_surf [°C]"); axes[2].set_xlabel("theta [deg]")
    axes[2].set_title(f"eq {res['T_eq']:.0f} °C; band fc θ{res['band_fc'][0]:.1f}–{res['band_fc'][1]:.1f} "
                      f"(~{res['band_w_km']:,.0f} km); drive ×{res['drive_rel']:.2f} of current; "
                      f"noon band {res['noon_band']:.0f} W/m²", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best") if ax is axes[0] else None
    fig.suptitle(f"shade design {name}")
    fig.tight_layout()
    out = os.path.join(PLOTDIR, f"shade_design_{name}.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


def forecast_shade(T_eff, K=None, theta=THETA_GRID):
    """RCE forecast for an ARBITRARY T_eff(α) profile (e.g. a smoothed design): achieved S, T(θ),
    band crossings, ribbon noon. The linear structure makes this exact — no re-solve needed."""
    K = kernel(theta) if K is None else K
    T_of_S, _, _ = rce_curve()
    S = K @ np.asarray(T_eff, float)
    Tfc = T_of_S(np.maximum(S, 1e-2))
    i30 = np.where((Tfc[:-1] > 30) & (Tfc[1:] <= 30))[0]
    i0c = np.where((Tfc[:-1] > 0) & (Tfc[1:] <= 0))[0]
    fc = (float(theta[i30[-1]]) if len(i30) else np.nan, float(theta[i0c[-1]]) if len(i0c) else np.nan)
    bw = (np.tan(np.radians(fc[1])) - np.tan(np.radians(fc[0]))) * C.L_SUN / 1e3 \
        if np.isfinite(fc).all() else np.nan
    centers = 0.5 * (ALPHA_EDGES[:-1] + ALPHA_EDGES[1:])
    F_noon = _one_sun_total(C.L_SUN / np.cos(np.radians(np.minimum(centers, 88.0))))
    mid = 0.5 * (fc[0] + fc[1]) if np.isfinite(fc).all() else 47.0
    noon = float(np.interp(mid, centers, np.asarray(T_eff) * F_noon))
    return dict(S=S, T_forecast=Tfc, band_fc=fc, band_w_km=bw, T_eq=float(Tfc[0]), noon_band=noon)


def save_shades(caps=(None, 1700.0, 1400.0, 1100.0), smooth_sigmas=(3.0, 6.0), outdir="m33_runs"):
    """Write the candidate-D shade files (npz: alpha_deg, T_eff = I0_mult × transmission) for the
    geometry.COSMO_SHADE_FILE hook — one per noon cap on the sharpness-vs-noon dial, PLUS
    Gaussian-smoothed variants of the sharp (uncapped) design (Jake's shape-sensitivity ask:
    a physically-graded shade with only roughly the right shape)."""
    from scipy.ndimage import gaussian_filter1d
    K = kernel()
    out = []
    sharp_T = None
    for cap in caps:
        res = design(K=K, noon_cap=cap, **CANDIDATES["D2_shelf"])
        T_eff = res["shade"] * res["I0_mult"]
        if cap is None:
            sharp_T = T_eff
        name = f"shade_D_cap{'none' if cap is None else int(cap)}"
        path = os.path.join(outdir, name + ".npz")
        np.savez(path, alpha_deg=res["alpha"], T_eff=T_eff,
                 noon_cap=np.nan if cap is None else cap,
                 band_fc=np.array(res["band_fc"]), I0_mult=res["I0_mult"])
        print(f"[shade_design] {path}: band fc θ{res['band_fc'][0]:.1f}–{res['band_fc'][1]:.1f} "
              f"({res['band_w_km']:,.0f} km), noon {res['noon_band']:.0f} W/m², I0× {res['I0_mult']:.2f}")
        out.append(path)
    for sig in smooth_sigmas:                                  # bins are 1 deg -> sigma in degrees
        T_s = gaussian_filter1d(sharp_T, sig, mode="nearest")
        fc = forecast_shade(T_s, K=K)
        path = os.path.join(outdir, f"shade_D_smooth{int(sig)}.npz")
        centers = 0.5 * (ALPHA_EDGES[:-1] + ALPHA_EDGES[1:])
        np.savez(path, alpha_deg=centers, T_eff=T_s, noon_cap=np.nan,
                 band_fc=np.array(fc["band_fc"]), smooth_sigma_deg=sig)
        print(f"[shade_design] {path}: σ={sig:.0f}° smoothing -> band fc "
              f"θ{fc['band_fc'][0]:.1f}–{fc['band_fc'][1]:.1f} ({fc['band_w_km']:,.0f} km), "
              f"eq {fc['T_eq']:.0f} °C, ribbon noon {fc['noon_band']:.0f} W/m²")
        out.append(path)
    return out


SH_TAGS = ["sh_d0", "sh_d17", "sh_d14", "sh_d11", "sh_d0s3", "sh_d0s6", "sh_d0L8", "sh_s6L8", "sh_d0L8h", "sh_w6L8", "sh_d0L4", "sh_b23L8", "sh_w1L8", "sh_w2L8"]


# The bespoke "bright strip" designs (D4/D5) that have NO save_shades factory: D4_bright is the
# CANONICAL b23 world (sh_b23L8); D5a/D5b are the band-width iteration (sh_w1L8/sh_w2L8). The
# committed m33_runs/shade_D4_bright.npz / _D5a_cliff.npz / _D5b_mid.npz are the AUTHORITATIVE
# files the converged checkpoints were run under (whitelisted in .gitignore). These recipes
# reproduce them only NUMERICALLY-CLOSE (the exact lsq design is solver/anchor-sensitive and not
# bit-recoverable) -- see ``save_bright_shades``. D4 = the D3 wide strip (+-6 deg) BRIGHTENED to
# S~2,190 (lifts the wind-cooled equatorial minimum clear of the 70 C wet-shroud trapdoor); D5a/D5b
# steepen the ribbon's thermostat-shelf traverse (1,050->350 over ~4-6 deg) on that same bright base.
BRIGHT_CANDIDATES = dict(
    D4_bright=dict(band=(45.0, 50.0), strip_half=6.0,
                   anchors=[(0.0, ("S", 2190.0)), (6.0, ("S", 2190.0)), (16.0, ("S", 1350.0)),
                            (30.0, ("S", 1050.0)), (42.0, ("S", 850.0)), (45.0, ("S", 700.0)),
                            (50.0, ("S", 350.0)), (56.0, ("S", 60.0)), (70.0, None), (84.5, None)]),
    D5a_cliff=dict(band=(45.0, 50.0), strip_half=6.0,
                   anchors=[(0.0, ("S", 2190.0)), (6.0, ("S", 2190.0)), (16.0, ("S", 1350.0)),
                            (30.0, ("S", 1050.0)), (44.0, ("S", 1000.0)), (46.0, ("S", 700.0)),
                            (48.0, ("S", 500.0)), (50.0, ("S", 350.0)), (56.0, ("S", 60.0)),
                            (70.0, None), (84.5, None)]),
    D5b_mid=dict(band=(45.0, 50.0), strip_half=6.0,
                 anchors=[(0.0, ("S", 2190.0)), (6.0, ("S", 2190.0)), (16.0, ("S", 1350.0)),
                          (30.0, ("S", 1050.0)), (44.0, ("S", 1000.0)), (47.0, ("S", 650.0)),
                          (50.0, ("S", 350.0)), (56.0, ("S", 60.0)), (70.0, None), (84.5, None)]),
)
# filename stem (in m33_runs/) of the authoritative committed fixture per recipe
BRIGHT_FILES = dict(D4_bright="shade_D4_bright", D5a_cliff="shade_D5a_cliff", D5b_mid="shade_D5b_mid")


def save_bright_shades(outdir="m33_runs", suffix="_regen", verbose=True):
    """Regenerate the bright-strip shade designs (``BRIGHT_CANDIDATES``) and report each one's
    deviation from the AUTHORITATIVE committed fixture (``m33_runs/shade_D*.npz``).

    PROVENANCE / re-roll tool, NOT a replacement for the committed files. By default it writes to
    ``shade_D4_bright_regen.npz`` etc. (``suffix='_regen'``) so it never clobbers the authoritative
    fixture the converged sh_* checkpoints were run under; pass ``suffix=''`` to write the canonical
    name (only for a deliberate new design point, never to "refresh" a fixture an existing checkpoint
    depends on -- the regenerated T_eff differs by ~0.3 from the committed file, which would silently
    perturb that world). The committed fixture stays the source of truth; this records the design
    intent and lets a new design point start from the same recipe. Returns a list of
    (name, path, max|dT_eff vs committed|, I0_mult) tuples."""
    K = kernel()
    out = []
    for name, kw in BRIGHT_CANDIDATES.items():
        res = design(K=K, noon_cap=None, **kw)
        T_eff = res["shade"] * res["I0_mult"]
        path = os.path.join(outdir, BRIGHT_FILES[name] + suffix + ".npz")
        np.savez(path, alpha_deg=res["alpha"], T_eff=T_eff, I0_mult=res["I0_mult"])
        committed = os.path.join(outdir, BRIGHT_FILES[name] + ".npz")
        dev = np.nan
        if os.path.exists(committed):
            ref = np.load(committed)
            if np.asarray(ref["T_eff"]).shape == T_eff.shape:
                dev = float(np.max(np.abs(T_eff - np.asarray(ref["T_eff"]))))
        out.append((name, path, dev, float(res["I0_mult"])))
        if verbose:
            print(f"[shade_design] {path}: I0× {res['I0_mult']:.3f}, band fc "
                  f"θ{res['band_fc'][0]:.1f}–{res['band_fc'][1]:.1f}; max|ΔT_eff vs committed| "
                  f"{dev:.3f} (committed fixture is authoritative)")
    return out


def flatten_day(shade_path, band=(43.0, 54.0), s_tol_rel=0.02, s_tol_abs=2.0, out=None):
    """FLAT-DAY reparameterization (Jake, 2026-06-12: "as flat a day as possible in the band").

    The core runs on day-mean S(θ) only, so every T(α) with the same S(θ) is climate-identical;
    the leftover freedom is the DAY SHAPE. The instantaneous flux any observer sees from a sun at
    zenith angle α is T(α)·G(α) with G(α) = the one-sun optics — a pure function of α — so the
    experienced day is the trajectory of T·G along the sun's passage. Minimizing the band's PEAK
    flux subject to S(θ) unchanged is a LINEAR minimax (scipy linprog):

        min t   s.t.   T_j·G_j ≤ t  (α_j in the band's daylight range),
                       |K·T − S₀| ≤ tol,   0 ≤ T_j ≤ M.

    The optimum is the physically flattest possible day: T(α) ∝ 1/G(α) over the lit window (the
    shade brightens toward low sun to cancel the slant dimming), clipped by the day-mean budget.
    Saves a climate-identical shade file (rel-ΔS printed) + returns the day-shape data."""
    from scipy.optimize import linprog
    d = np.load(shade_path)
    T0 = np.asarray(d["T_eff"], float)
    centers = 0.5 * (ALPHA_EDGES[:-1] + ALPHA_EDGES[1:])
    K = kernel()
    S0 = K @ T0
    G = _one_sun_total(C.L_SUN / np.cos(np.radians(np.minimum(centers, 88.0))))
    nb = len(T0)
    lit = (centers >= band[0] - 1.0) & (centers <= 80.0)        # the band's daylight alpha range
    c = np.zeros(nb + 1); c[-1] = 1.0                            # vars [T..., t]; min t
    A_ub, b_ub = [], []
    for j in np.where(lit)[0]:                                   # T_j G_j - t <= 0
        row = np.zeros(nb + 1); row[j] = G[j]; row[-1] = -1.0
        A_ub.append(row); b_ub.append(0.0)
    tol = s_tol_rel * S0 + s_tol_abs
    A_ub = np.vstack(A_ub + [np.concatenate([K, np.zeros((len(S0), 1))], axis=1),
                             np.concatenate([-K, np.zeros((len(S0), 1))], axis=1)])
    b_ub = np.concatenate([np.asarray(b_ub), S0 + tol, -(S0 - tol)])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0.0, 60.0)] * nb + [(0.0, None)],
                  method="highs")
    assert res.success, res.message
    T1 = res.x[:nb]
    S1 = K @ T1
    rel = float(np.max(np.abs(S1 - S0) / np.maximum(S0, 1.0)))
    out = out or shade_path.replace(".npz", "_flatday.npz")
    np.savez(out, alpha_deg=centers, T_eff=T1, parent=os.path.basename(shade_path),
             band=np.array(band), peak_band_flux=float(res.x[-1]))
    print(f"[flatten_day] {out}: band peak {float(res.x[-1]):.0f} W/m² "
          f"(was {float(np.max(T0[lit] * G[lit])):.0f}); max rel ΔS(θ) = {rel:.4f}")
    return dict(T0=T0, T1=T1, centers=centers, G=G, S0=S0, S1=S1, peak=float(res.x[-1]), out=out)


def seed_runs(tags=None, smoke_steps=10, n_anchor=1500):
    """Warm-branch seeds for the shade fleet: under each tag's OWN shade env, the core's S(θ) IS
    the design flux, so T_surf = T_RCE_warm(S) puts every column on the warm branch of its own
    forcing (a cool fresh start would land the wet columns in the single-column shrouded
    attractor — the bistability the first anchor sweep exposed). Quiescent dynamics, clock 0.
    Env-bearing tags re-exec through a subprocess (constants/geometry are import-time)."""
    import subprocess as sp
    import sys as _s
    from . import _allon_fleet as _af
    from . import _warm_restart as _wr
    from ._box_scout import _needs_subprocess, _refuse_if_busy
    done = []
    for tag in (tags or SH_TAGS):
        spec = _wr._spec_of(tag)
        if _needs_subprocess(spec):
            env = dict(os.environ); env.update(spec["env"]); env.pop("PYTHONPATH", None)
            r = sp.run([_s.executable, "-m", "flatworld._shade_design", "seed", tag],
                       env=env, capture_output=True, text=True, timeout=3600)
            print(r.stdout.strip())
            if r.returncode != 0:
                raise RuntimeError(f"{tag}: seed subprocess failed: {r.stderr[-400:]}")
            done.append(tag)
            continue
        _refuse_if_busy(tag)
        core = _wr._tag_core(spec)
        T_of_S, _, _ = rce_curve()
        T0 = np.clip(T_of_S(np.maximum(core.S, 1e-2)) + 273.15, 100.0, 480.0)
        core.T_surf = T0.copy()
        prof = np.asarray(core.T_atm[0], float)
        core.T_atm = (T0[:, None] + (prof - prof[-1])[None, :]).clip(90.0, 500.0)
        # ANCHORED SPIN-UP (the first warm-seeded wave collapsed in <53 d: quiescent dynamics +
        # the full ~300 °C design contrast = a transient super-cell that chills every column below
        # the thermostat shelf, and the single-column bistability locks them cold). Fix with the
        # core's own prescribed-SST mode (M2b.3): run the dynamics against the FIXED design
        # surface until the cell equilibrates, then release the surface and install. No physics
        # change at measurement time -- an initialization procedure.
        core.couple_surface = False
        core.run_dynamics = True
        for k in range(n_anchor):
            core.step()
            if k % 300 == 0:
                assert core._state_is_finite(), f"{tag}: anchored spin went non-finite at {k}"
        core.couple_surface = True
        st = core.state()
        core.load_state(st)
        for _ in range(smoke_steps):
            core.step()
            assert core._state_is_finite(), f"{tag}: smoke step went non-finite"
        conv.save_checkpoint(_af._paths(spec)["ckpt"], st, {"iter": 0, "sim_time": 0.0})
        Ts = T0 - 273.15
        import flatworld.stage_ad_core as _sac
        psi_mx = float(np.abs(np.asarray(_sac.to_host(core.psi))).max())
        print(f"[shade_design] {tag}: anchored-spin seed installed (T eq {Ts[0]:.0f} °C, min "
              f"{Ts.min():.0f}, max {Ts.max():.0f}; {n_anchor} fixed-SST steps, max|psi| {psi_mx:.2e})")
        done.append(tag)
    return done


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "shades":
        save_shades()
        raise SystemExit(0)
    if len(_sys.argv) > 1 and _sys.argv[1] == "brights":   # regen the bespoke D4/D5 designs (provenance)
        save_bright_shades(suffix="" if "--overwrite" in _sys.argv else "_regen")
        raise SystemExit(0)
    if len(_sys.argv) > 1 and _sys.argv[1] == "seed":
        seed_runs(_sys.argv[2:] or None)
        raise SystemExit(0)
    K = kernel()
    print(f"{'candidate':18s} {'I0×':>6s} {'eqT':>6s} {'band fc':>12s} {'width km':>9s} "
          f"{'drive×':>7s} {'noonW':>6s} {'resid':>6s}")
    for name, kw in CANDIDATES.items():
        res = design(K=K, **kw)
        out = render(name, res)
        print(f"{name:18s} {res['I0_mult']:6.1f} {res['T_eq']:6.0f} "
              f"θ{res['band_fc'][0]:5.1f}–{res['band_fc'][1]:5.1f} {res['band_w_km']:9,.0f} "
              f"{res['drive_rel']:7.4f} {res['noon_band']:6.0f} {res['residual']:6.1f}  -> {out}")
