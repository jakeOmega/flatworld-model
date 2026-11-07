"""
flatworld.bl_diurnal -- the DIURNAL sea/land-breeze surface wind (issue #113).

WHY. The lantern (painted) day -- a 16-h half-cosine of light + 8-h night -- makes the land/sea thermal
contrast BREATHE ~10-20 K over the day. The boundary layer responds in hours: afternoon wind maxima,
calm mornings, and coastal sea-breeze (onshore by day) / land-breeze (offshore by night) REVERSALS,
whose density-current scale sqrt(2*beta_T*dT) for the diurnal dT is ~10-20 m/s -- comparable to the mean
wind itself. Until now that physics lived only as the PARAMETERIZED nocturnal-decoupling discount in
surface_wind.night_factor. This module makes it DYNAMICAL by time-stepping the #111 nonlinear BL momentum
solve (flatworld.bl_momentum) through the diurnal cycle instead of solving for a steady state.

HOW (the #111 host model makes this nearly a forcing swap). bl_momentum.solve_surface_wind already
accepts a callable phi_of(t) and steps with an RK2 corrector that evaluates phi_of(t+dt); the only new
machinery is (a) a TIME-VARYING forcing built from the diurnally-resolved surface temperature, and (b) a
PERIODIC-integration driver that runs the layer over repeated days and stops when the sampled wind CYCLE
stops drifting day-to-day (a periodicity test, not a steady-state test). Everything else -- the tendency,
the CFL dt, the mean/anomaly split, the diagnostics -- is reused from bl_momentum unchanged.

THE FORCING (the load-bearing construction). Phi(t) = Phi_clim + Phi'(t), where
  * Phi_clim is the #111 CLIMATOLOGICAL geopotential -- geopotential(virtual_T(T_map), ..., phi_smooth_km)
    -- the SAME field #111's steady solve flows down (long, 400-km frictional adjustment fetch). It is the
    day-mean.
  * Phi'(t) is the ZERO-day-mean diurnal ANOMALY: geopotential(virtual_T(T(t)), ..., phi_smooth_km_diurnal)
    with its day-mean subtracted, smoothed over the SHORTER breeze-adjustment fetch (U*tau ~ 100-150 km,
    NOT the 400-km climatological length -- the breeze is the short-scale response and the climatological
    smoothing would wash the coastline contrast that drives it). The diurnal smoothing length is DERIVED
    (adjustment fetch / grid scale) and SWEPT, never fitted; the breeze magnitude is reported as a band.
Because Phi'(t) has zero day-mean by construction, the day-mean of Phi(t) is EXACTLY Phi_clim -- so the
day-mean wind matches #111's steady solve structurally (acceptance criterion 3), and the mean climate is
untouched.

THE TEMPERATURE CYCLE. T(x,e,t) is reconstructed (cheap, mean-exact) rather than re-solving the surface
slab per (x,e) cell: one shared NORMALIZED diurnal shape s(t) (mean 0) is built from a single
surface.solve_periodic_surface call driven by the lantern flux -- this carries the physical afternoon-lag
PHASE (skin T peaks ~1-2 h after noon, not at noon). Each land cell scales s(t) to its own already-computed
diurnal RANGE (T_max_C - T_min_C) about its day-mean T_map_C, so the cycle's day-mean is T_map_C EXACTLY
and its range matches T_max-T_min EXACTLY. Sea cells are held flat (the breeze is a land-driven anomaly).
[Option-A limitation: a single shared shape carries one representative skew/phase, not a per-cell
coast-vs-interior phase difference -- disclosed; the SIGN/SHAPE is firm, the magnitude is soft.]

TIERS. Firm: the SIGN/SHAPE -- afternoon wind max, calm mornings, onshore-day/offshore-night coastal
reversal. Soft (magnitude): the m/s, the night/day ratios, the sea-breeze penetration km -- they inherit
the diurnal smoothing-length sweep, the diagnostic u_surf background, and #111's drag-closure band. The
model's ~100-km grid cannot resolve a real (tens-of-km) sea-breeze front, so all magnitudes are
smoothed-scale.
"""
from __future__ import annotations
import warnings
import numpy as np
from dataclasses import dataclass, field

from . import constants as C
from . import geometry as g
from . import surface as surf
from . import lantern
from . import operators as ops
from . import convergence as conv
from . import bl_forcing as blf
from . import bl_momentum as blm


@dataclass
class BLDiurnalConfig:
    """Knobs for the diurnal sea/land-breeze solve. ``mom`` is the underlying #111 momentum config (drag
    closure, beta_T, climatological phi_smooth_km, scheme, eddy viscosity); the diurnal-only knobs are the
    sampling resolution, the SHORT breeze-anomaly smoothing length, and the periodicity guards."""
    mom: blm.BLMomentumConfig = field(default_factory=blm.BLMomentumConfig)
    n_phase: int = 8                   # samples per day (phase 0 = noon)
    phi_smooth_km_diurnal: float = 120.0   # breeze-anomaly smoothing length [km] (U*tau adjustment fetch;
                                           #   SHORTER than the 400-km climatological mom.phi_smooth_km).
                                           #   DERIVED + SWEPT, not fitted -- report the breeze as a band.
    # --- periodicity convergence / guards (the diurnal cycle, not steady state) ---
    max_days: int = 60                 # outer-loop day cap (guard)
    period_tol: float = 2.0e-3         # day-to-day cycle drift tolerance (normalized)
    period_patience: int = 2
    max_seconds: float = 600.0
    checkpoint_path: str | None = None
    checkpoint_every_days: int = 10
    # --- the shared diurnal-shape solve (representative arid land -> a clear afternoon-lagged shape) ---
    shape_C_surf: float = C.C_ARID_LAND
    shape_wetness: float = 0.10
    shape_LW_down: float = 350.0       # representative downwelling LW [W/m^2] (sets the discarded mean only)
    shape_T_air_K: float = 295.0
    shape_theta_deg: float = 47.5      # band-representative latitude for the shared day-shape


def diurnal_shape(S_of_theta, dcfg: BLDiurnalConfig, clock=None, x_shape=None, n_steps=144):
    """One shared, mean-zero normalized diurnal temperature shape s(t_phase) [dimensionless] carrying the
    physical afternoon-lag PHASE. Built from a single surface.solve_periodic_surface call driven by the
    lantern day at a band-representative x; ONLY the normalized shape (not the absolute level) is used, so
    the representative LW_down / T_air just set the discarded mean. Returns (t_phase [s], s) with
    mean(s)=0; the per-cell amplitude is applied at reconstruction."""
    clock = g.CLOCK if clock is None else clock
    x_shape = g.x_of_theta(np.radians(dcfg.shape_theta_deg)) if x_shape is None else x_shape
    t, F = lantern.day_flux_cycle(float(x_shape), S_of_theta, n_steps=n_steps, clock=clock)
    SW_abs = 0.9 * (1.0 - C.ALBEDO_LAND) * F
    r = surf.solve_periodic_surface(t, SW_abs, dcfg.shape_LW_down, dcfg.shape_T_air_K, 3e-3,
                                    C.P_SURF, wind=5.0, C_surf=dcfg.shape_C_surf,
                                    wetness=dcfg.shape_wetness, max_days=60, tol=2e-3)
    t_phase = np.arange(dcfg.n_phase) * clock.T_day / dcfg.n_phase     # phase 0 = noon
    s_raw = np.interp(t_phase, r["t"], r["Ts_cycle"])
    s = s_raw - s_raw.mean()                                           # mean-zero shape (level discarded)
    return t_phase, s


def build_diurnal_forcing(dC, dx, de, dcfg: BLDiurnalConfig, S_of_theta, clock=None, x_shape=None):
    """Build the time-varying boundary-layer forcing Phi(t) = Phi_clim + zero-mean diurnal anomaly for the
    diurnal momentum solve. Returns a dict:
        phase_t   : (K,) sample times over one day [s], phase 0 = noon.
        T_cycle_C : (K, nx, ne) reconstructed surface-temperature cycle [C] (day-mean == T_map_C exactly).
        phi       : (K, nx, ne) the forcing slices [m^2/s^2] (day-mean == phi_clim exactly).
        phi_clim  : (nx, ne) the #111 climatological geopotential (the day-mean / steady forcing).
        phi_of    : callable phi_of(t) -> (nx, ne), circular-lerp between the K slices in t mod T_day.
        shape_s   : (K,) the shared mean-zero diurnal shape.
    """
    clock = g.CLOCK if clock is None else clock
    mom = dcfg.mom
    T_map = np.asarray(dC["T_map_C"], float)
    land = np.asarray(dC["land"], bool)
    W_s = dC.get("W_s") if mom.virtual_temp else None
    height = np.asarray(dC["height"], float)

    phase_t, s = diurnal_shape(S_of_theta, dcfg, clock=clock, x_shape=x_shape)
    s_range = float(s.max() - s.min())
    amp = (np.asarray(dC["T_max_C"], float) - np.asarray(dC["T_min_C"], float)) / max(s_range, 1e-12)

    # reconstruct the per-cell cycle: land scales the shared shape to its own range about its day-mean;
    # sea is held flat. mean_k(s)=0 -> day-mean == T_map exactly; range == T_max-T_min exactly.
    T_cycle = np.empty((dcfg.n_phase,) + T_map.shape, float)
    for k in range(dcfg.n_phase):
        T_cycle[k] = np.where(land, T_map + amp * s[k], T_map)

    # the #111 CLIMATOLOGICAL Phi (the day-mean / steady forcing, long 400-km fetch)
    Tv_clim = blf.virtual_T_map(T_map, W_s)
    phi_clim = blf.geopotential(Tv_clim, height, dx, de, beta_T=mom.phi_beta_T,
                                g_terrain=mom.phi_g_terrain, phi_smooth_km=mom.phi_smooth_km)

    # the diurnal ANOMALY at the SHORT breeze-adjustment fetch, day-mean removed -> zero-mean anomaly
    anom = np.empty_like(T_cycle)
    for k in range(dcfg.n_phase):
        Tv_k = blf.virtual_T_map(T_cycle[k], W_s)
        anom[k] = blf.geopotential(Tv_k, height, dx, de, beta_T=mom.phi_beta_T,
                                   g_terrain=mom.phi_g_terrain, phi_smooth_km=dcfg.phi_smooth_km_diurnal)
    anom -= anom.mean(axis=0, keepdims=True)                            # zero day-mean by construction
    phi = phi_clim[None, :, :] + anom

    T_day = clock.T_day
    dt_phase = T_day / dcfg.n_phase

    def phi_of(t=0.0):
        pos = (float(np.mod(t, T_day))) / dt_phase
        k0 = int(np.floor(pos)) % dcfg.n_phase
        k1 = (k0 + 1) % dcfg.n_phase
        w = pos - np.floor(pos)
        return (1.0 - w) * phi[k0] + w * phi[k1]

    return dict(phase_t=phase_t, T_cycle_C=T_cycle, phi=phi, phi_clim=phi_clim,
                phi_of=phi_of, shape_s=s)


def _integrate_to(u, v, t_model, t_target, phi_of, u_bg, v_bg, ax_k, ay_k, dx, de, mom):
    """RK2/Euler-step the momentum field from t_model up to t_target, re-evaluating the time-varying
    forcing each step and capping dt so it lands exactly on t_target (the next phase sample). Returns
    (u, v, t_model). Reuses bl_momentum's tendency + CFL dt unchanged."""
    eps = 1e-6
    while t_model < t_target - eps:
        dt = min(blm._cfl_dt(u, v, dx, de, mom), t_target - t_model)
        Phi1 = np.asarray(phi_of(t_model), float)          # evaluate the lerp ONCE per step (it was
        gPx = ops.ddx(Phi1, dx)                            #   built twice, once per gradient axis)
        gPe = ops.dde(Phi1, de)
        dudt, dvdt = blm._tendency(u, v, gPx, gPe, u_bg, v_bg, ax_k, ay_k, dx, de, mom)
        if mom.scheme == "euler":
            u = u + dt * dudt
            v = v + dt * dvdt
        else:  # RK2 / Heun -- the corrector sees the forcing at t+dt (this is what resolves the cycle)
            Phi2 = np.asarray(phi_of(t_model + dt), float)
            gPx2, gPe2 = ops.ddx(Phi2, dx), ops.dde(Phi2, de)
            u1, v1 = u + dt * dudt, v + dt * dvdt
            d2u, d2v = blm._tendency(u1, v1, gPx2, gPe2, u_bg, v_bg, ax_k, ay_k, dx, de, mom)
            u = u + 0.5 * dt * (dudt + d2u)
            v = v + 0.5 * dt * (dvdt + d2v)
        t_model += dt
        if not (np.isfinite(u).all() and np.isfinite(v).all()):
            raise FloatingPointError("bl_diurnal diverged (CFL/forcing blow-up)")
    return u, v, t_model


def solve_diurnal_wind(phi_of, ubg, dx, de, dcfg: BLDiurnalConfig, phase_t,
                       T_map_C=None, height=None, u0=None, v0=None, clock=None, verbose=False):
    """Time-step the #111 BL momentum layer through the diurnal cycle to a PERIODIC state.

    Unlike bl_momentum.solve_surface_wind (which stops on steady-state drift -> 0), this integrates over
    repeated days and stops when the sampled wind CYCLE stops drifting day-to-day (a periodicity test).
    Warm-starts from the steady solve on the day-mean forcing so the periodic attractor is reached in a
    few days. Returns the per-phase wind fields + the day-mean / afternoon-peak / night-min maps and the
    steady reference (for the consistency gate)."""
    clock = g.CLOCK if clock is None else clock
    mom = dcfg.mom
    phase_t = np.asarray(phase_t, float)
    K = phase_t.size
    T_day = clock.T_day
    Phi0 = np.asarray(phi_of(0.0), float)
    nx, ne = Phi0.shape
    u_bg = np.broadcast_to(np.asarray(ubg, float).reshape(-1, 1) if np.ndim(ubg) == 1 else ubg,
                           (nx, ne)).astype(float)
    v_bg = np.zeros((nx, ne))

    # warm-start: steady #111 solve on the day-mean forcing (== phi_clim by construction). Doubles as the
    # consistency-gate reference (acceptance 3: day-mean wind ~ the steady solve).
    phi_daymean = np.mean([np.asarray(phi_of(t), float) for t in phase_t], axis=0)
    steady = blm.solve_surface_wind(phi_daymean, ubg, dx, de, mom,
                                    T_map_C=T_map_C, height=height, verbose=False)
    u = steady["u"].copy() if u0 is None else np.asarray(u0, float).copy()
    v = steady["v"].copy() if v0 is None else np.asarray(v0, float).copy()

    ax_k, ay_k = (blm.katabatic_accel(T_map_C if T_map_C is not None else Phi0, height, dx, de, mom)
                  if mom.katabatic else (0.0, 0.0))

    mon = conv.ConvergenceMonitor(tol=dcfg.period_tol, patience=dcfg.period_patience,
                                  max_iter=dcfg.max_days, max_seconds=dcfg.max_seconds,
                                  name="diurnal cycle drift", absolute=True)
    t_model = 0.0
    prev_cycle = None
    speed_phase = np.zeros((K, nx, ne))
    u_phase = np.zeros((K, nx, ne)); v_phase = np.zeros((K, nx, ne))
    day = 0
    for day in range(1, dcfg.max_days + 1):
        cycle = np.zeros((K, nx, ne))
        for k in range(K):
            t_target = (day - 1) * T_day + phase_t[k]
            u, v, t_model = _integrate_to(u, v, t_model, t_target, phi_of, u_bg, v_bg,
                                          ax_k, ay_k, dx, de, mom)
            cycle[k] = np.hypot(u, v)
            u_phase[k] = u; v_phase[k] = v
        # finish the day so the next day starts cleanly at the phase-0 boundary
        u, v, t_model = _integrate_to(u, v, t_model, day * T_day, phi_of, u_bg, v_bg,
                                      ax_k, ay_k, dx, de, mom)
        denom = max(float(cycle.mean()), 1e-9)
        drift = float(np.max(np.abs(cycle - prev_cycle))) / denom if prev_cycle is not None else np.inf
        prev_cycle = cycle.copy()
        speed_phase = cycle
        if verbose:
            print(f"[bl_diurnal] day={day:>3} <|V|>={denom:6.1f} "
                  f"max={float(cycle.max()):6.1f} cycle-drift={drift:.2e}")
        stop = mon.update(day, drift if np.isfinite(drift) else 1.0)
        if dcfg.checkpoint_path and day % dcfg.checkpoint_every_days == 0:
            conv.save_checkpoint(dcfg.checkpoint_path, dict(u=u, v=v, day=day), {"day": day})
        if stop:
            break

    if not mon.converged:
        warnings.warn(
            f"bl_diurnal periodic solve did NOT reach a periodic cycle ({mon.stop_reason}); the reported "
            f"fields are the LAST day and may still be drifting. A max_seconds stop is wall-clock-dependent, "
            f"so the result is then NON-REPRODUCIBLE -- raise max_days/max_seconds or loosen period_tol.",
            RuntimeWarning, stacklevel=2)

    daylen_h = lantern.DAYLEN_H
    circ_h = np.minimum(phase_t, T_day - phase_t) / 3600.0
    night = circ_h > daylen_h / 2.0
    day_mask = ~night
    day_mean = speed_phase.mean(axis=0)
    day_mean_u = u_phase.mean(axis=0); day_mean_v = v_phase.mean(axis=0)
    # afternoon-peak / night-min SPEED maps (reported standalone). NB do NOT divide these to form a
    # day/night ratio: max-over-day / min-over-night is biased < 1 by construction (order statistics of an
    # oscillation, not a physical contrast -- see MODELING_LESSONS.md #113). The honest day/night signal is
    # the like-for-like orientation dipole at the fixed breeze phases (k_on/k_off) in diurnal_diagnostics.
    afternoon_peak = speed_phase[day_mask].max(axis=0) if day_mask.any() else speed_phase.max(axis=0)
    night_min = speed_phase[night].min(axis=0) if night.any() else speed_phase.min(axis=0)
    # diurnal rectification: the day-mean VELOCITY differs from the steady solve of the day-mean forcing
    # because nonlinear advection of the breeze oscillation drives a mean flow. The FORCING day-mean is
    # unchanged (== phi_clim, structural), so this is a genuine finding, not a mean-climate drift; it is
    # small where the background wind dominates the breeze (the real band) and grows in the weak-mean limit.
    svec = np.concatenate([steady["u"].ravel(), steady["v"].ravel()])
    dvec = np.concatenate([day_mean_u.ravel(), day_mean_v.ravel()])
    rect = float(np.linalg.norm(dvec - svec) / max(np.linalg.norm(svec), 1e-9))
    return dict(phase_t=phase_t, u_phase=u_phase, v_phase=v_phase, speed_phase=speed_phase,
                day_mean_speed=day_mean, day_mean_u=day_mean_u, day_mean_v=day_mean_v,
                afternoon_peak_speed=afternoon_peak, night_min_speed=night_min,
                steady_speed=steady["speed"], steady_u=steady["u"], steady_v=steady["v"],
                rectification_frac=rect, night_mask=night, converged=mon.converged, days=day,
                days_guard=dcfg.max_days, stop_reason=mon.stop_reason)


# ============================ diagnostics: breeze + per-Koppen night/day ratios ============================

def onshore_normal(land):
    """Unit onshore normal (sea -> land) on the (x,e) grid from the land mask alone. Built as the gradient
    of a signed distance-to-coast field (>0 inland, <0 offshore), so it points up the land gradient = into
    the land. Returns (n_x, n_e). NB the distance transform is NOT e-periodic -- near the e-seam the
    direction is approximate; the firm coastal-cell signs (used for the reversal signature) are unaffected."""
    from scipy import ndimage as ndi
    land = np.asarray(land, bool)
    d_land = ndi.distance_transform_edt(land)
    d_sea = ndi.distance_transform_edt(~land)
    s = np.where(land, d_land, -d_sea).astype(float)          # signed distance: + inland, - offshore
    gx = ops.ddx(s, 1.0); ge = ops.dde(s, 1.0)
    mag = np.hypot(gx, ge)
    with np.errstate(invalid="ignore", divide="ignore"):
        n_x = np.where(mag > 1e-9, gx / mag, 0.0)
        n_e = np.where(mag > 1e-9, ge / mag, 0.0)
    return n_x, n_e


def local_solar_hour(t_s, T_day):
    """Local solar hour from the model time (t=0 == noon, the lantern/geometry convention): noon=12,
    midnight=0/24. Used to time the sea-breeze (afternoon) / land-breeze (pre-dawn) peaks."""
    return np.mod(12.0 + np.asarray(t_s, float) / 3600.0, 24.0)


def diurnal_diagnostics(sol, dC, koppen=None, dx=None, de=None, clock=None, labels=None,
                        coast_band_cells=2, align_thresh=0.5):
    """Worldbuilding-facing diagnostics from a diurnal solve -- the sea/land-breeze signature. What the
    dynamical solve resolves is the BREEZE: an afternoon onshore / pre-dawn offshore reversal whose
    amplitude tracks the local diurnal temperature range dT via the density-current scale sqrt(2*beta*dT).

    Three honest readings (the per-Koppen day/night RATIO that an earlier draft reported is NOT here: it
    averages a real ORIENTATION DIPOLE to ~1 and conflates the breeze with the unresolved nocturnal
    surface-layer decoupling that surface_wind.night_factor still owns -- see the #113 finding):
      * breeze REVERSAL fraction + amplitude (peak-to-peak cross-shore anomaly), and its by-Koppen split;
      * the ORIENTATION DIPOLE: the day/night SPEED ratio is <1 on coasts whose onshore is into the steady
        wind (windier days) and >1 on lee coasts (windier nights) -- keyed on coast orientation vs the wind,
        NOT on Koppen class; the spatial mean ~1 because the two cancel;
      * the breeze PEAK TIMES (sea-breeze afternoon, land-breeze pre-dawn) and the intrusion distance.
    """
    clock = g.CLOCK if clock is None else clock
    land = np.asarray(dC["land"], bool)
    speed_phase = sol["speed_phase"]; u_phase = sol["u_phase"]; v_phase = sol["v_phase"]
    phase_t = np.asarray(sol["phase_t"], float)
    K = phase_t.size

    from scipy import ndimage as ndi
    d_land = ndi.distance_transform_edt(land)
    coast_band = land & (d_land <= coast_band_cells) & (d_land >= 1)
    cb = coast_band if coast_band.any() else land

    n_x, n_e = onshore_normal(land)
    u_cs = u_phase * n_x[None] + v_phase * n_e[None]          # >0 onshore (sea breeze), <0 offshore
    cs_anom = u_cs - u_cs.mean(axis=0, keepdims=True)         # the BREEZE = cross-shore anomaly (steady removed)

    # sea-breeze / land-breeze peak PHASES from the coastal-mean breeze (not from speed extrema, which would
    # bias a ratio < 1 by sampling). k_on = peak onshore (sea breeze), k_off = peak offshore (land breeze).
    coastal_breeze = cs_anom[:, cb].mean(axis=1)
    k_on = int(np.argmax(coastal_breeze)); k_off = int(np.argmin(coastal_breeze))
    afternoon_cs = u_cs[k_on]; night_cs = u_cs[k_off]
    reversal = afternoon_cs - night_cs                        # >0 = onshore-by-day (the breeze signature)
    breeze_reversal_frac = float(np.mean(reversal[cb] > 0.0)) if cb.any() else float("nan")

    # breeze AMPLITUDE: peak-to-peak cross-shore anomaly per cell (the density-current response)
    breeze_amp = cs_anom.max(axis=0) - cs_anom.min(axis=0)
    amp_median = float(np.median(breeze_amp[cb])) if cb.any() else float("nan")
    amp_p90 = float(np.percentile(breeze_amp[cb], 90)) if cb.any() else float("nan")

    # per-cell breeze peak HOURS (local solar): sea-breeze (max anomaly) and land-breeze (min)
    hour = local_solar_hour(phase_t, clock.T_day)
    sea_breeze_hour = float(np.median(hour[np.argmax(cs_anom, axis=0)][cb])) if cb.any() else float("nan")
    land_breeze_hour = float(np.median(hour[np.argmin(cs_anom, axis=0)][cb])) if cb.any() else float("nan")

    # the ORIENTATION DIPOLE: signed projection of the onshore normal onto the steady wind. cos>0 = sea
    # UPWIND (sea breeze reinforces the wind by day -> windier days, ratio<1); cos<0 = sea DOWNWIND
    # (windier nights, ratio>1). The spatial mean cancels -- this is why a single per-class ratio is ~1.
    su = np.asarray(sol["steady_u"], float); sv = np.asarray(sol["steady_v"], float)
    smag = np.hypot(su, sv)
    scos = (su * n_x + sv * n_e) / np.maximum(smag, 1e-9)
    ratio = speed_phase[k_off] / np.maximum(speed_phase[k_on], 1e-9)   # land-breeze / sea-breeze SPEED
    facing = cb & (scos > align_thresh); lee = cb & (scos < -align_thresh)
    ratio_windfacing = float(ratio[facing].mean()) if facing.any() else float("nan")
    ratio_lee = float(ratio[lee].mean()) if lee.any() else float("nan")
    ratio_spatial_mean = float(ratio[cb].mean()) if cb.any() else float("nan")

    # by-Koppen breeze AMPLITUDE (meaningful per-class number -- the hot/arid classes get the big breezes)
    amp_by_koppen = {}
    if koppen is not None:
        if labels is None:
            from . import classify
            labels = classify.LABELS
        code = np.asarray(koppen, int)
        for i, lab in enumerate(labels):
            m = cb & (code == i + 1)
            if m.any():
                amp_by_koppen[lab] = float(np.median(breeze_amp[m]))

    # sea-breeze intrusion distance: inland decay of the breeze REVERSAL to 1/e (box-tested lower bound).
    # Pass the PHYSICAL spacings (not the isotropic cell-count d_land): the intrusion runs along the coast
    # normal, so a coast facing along x must scale by dx and one along e by de -- they differ on the grid.
    pen_km, saturated = _penetration_km(reversal, land, (dx or 0.0) / 1e3, (de or 0.0) / 1e3)

    return dict(sea_breeze_idx=k_on, land_breeze_idx=k_off,
                afternoon_cross_shore=afternoon_cs, night_cross_shore=night_cs,
                cross_shore_reversal=reversal, cross_shore_anomaly=cs_anom, cross_shore_phase=u_cs,
                coast_band=coast_band, onshore_normal_x=n_x, onshore_normal_e=n_e,
                breeze_reversal_frac=breeze_reversal_frac, breeze_amplitude=breeze_amp,
                breeze_amp_median_ms=amp_median, breeze_amp_p90_ms=amp_p90,
                breeze_amp_by_koppen=amp_by_koppen,
                sea_breeze_peak_hour=sea_breeze_hour, land_breeze_peak_hour=land_breeze_hour,
                daynight_ratio_windfacing=ratio_windfacing, daynight_ratio_lee=ratio_lee,
                daynight_ratio_spatial_mean=ratio_spatial_mean,
                sea_breeze_penetration_km=pen_km, penetration_box_saturated=saturated)


def _penetration_km(afternoon_cs, land, dx_km, de_km):
    """Sea-breeze intrusion distance [km]: how far inland the afternoon onshore flow persists before it
    decays to 1/e of its coastal value. Marches outward by ISOTROPIC cell ring (robust -- integer rings
    are never empty, unlike physical-width bins on an anisotropic grid), then converts the crossing ring to
    PHYSICAL km via the distance transform SAMPLED by the cell spacings (dx_km, de_km): so a coast whose
    inland normal runs along x is scaled by dx and one along e by de -- they differ on the real grid, and
    multiplying the bare cell count by a single spacing (the pre-#113-review code) mis-scaled either.
    Box-tested -- returns (km, saturated) where saturated=True flags a penetration that reaches the inland
    edge of the domain (a LOWER BOUND, not a resolved scale)."""
    from scipy import ndimage as ndi
    land = np.asarray(land, bool)
    dx_km = float(dx_km or 0.0); de_km = float(de_km or 0.0)
    if dx_km <= 0.0 or de_km <= 0.0 or not land.any():
        return 0.0, False
    dl = np.round(ndi.distance_transform_edt(land)).astype(int)         # isotropic cell rings (the march)
    d_phys = ndi.distance_transform_edt(land, sampling=(dx_km, de_km))  # PHYSICAL km (axis-aware)
    coastal = afternoon_cs[land & (dl == 1)]
    if coastal.size == 0 or coastal.mean() <= 0:
        return 0.0, False
    thr = coastal.mean() / np.e
    maxd = int(dl[land].max())
    pen_cells = 0
    for d in range(1, maxd + 1):
        m = land & (dl == d)
        if not m.any():
            continue
        if afternoon_cs[m].mean() < thr:
            break
        pen_cells = d
    if pen_cells == 0:
        return 0.0, False
    pen_km = float(d_phys[land & (dl == pen_cells)].mean())             # km of the 1/e-crossing ring
    return pen_km, bool(pen_cells >= maxd)
