"""
flatworld.stage_c_diurnal -- couple the diurnal sea/land breeze (#113, ``bl_diurnal``) into Stage C's
precipitation (issue #131).

Stage C precip rides a STATIC day-mean wind (``StageCMap`` builds the boundary-layer geopotential
``Phi`` once from a static temperature map; moisture is advected by that time-mean balanced wind). Rain
is a NONLINEAR THRESHOLD process, so the strong AFTERNOON sea-breeze convergence peak -- which can cross
saturation where the daily MEAN convergence does not -- gets averaged away, and the map UNDER-COUNTS the
convergent rain along coastal and inland sea-breeze fronts. The diurnal breeze itself is already solved
to a periodic state by ``bl_diurnal`` (#113), but it only feeds the reported wind, never the precip.

This module re-steps the converged two-layer moisture (``W_s``, ``W_a``) on the breeze winds, so the
afternoon convergent rain is RESOLVED rather than smeared. Two entry points, sharing the byte-identical
``StageCMap._step_with_wind`` seam (so the precip physics is identical to the deliverable; only the wind
and the sub-cycled timestep change):

  * ``solve_diurnal_precip`` -- PRIMARY: the full sub-daily periodic solve. Warm-starts from the
    converged day-mean moisture, steps it through repeated lantern-days on the time-varying breeze with
    its own advective CFL sub-cycle, and stops on a periodicity test -> the resolved per-phase precip
    cycle ``P_phase`` + a rectification-corrected day-mean ``P_daymean_rect``.
  * ``afternoon_precip_overlay`` -- FALLBACK: a one-step KINEMATIC overlay. Holds the converged day-mean
    moisture fixed, evaluates the precip once at the afternoon-peak breeze wind, and restores the state
    (a pure diagnostic). ~One step; shows WHERE the afternoon convergence would rain given the standing
    moisture, but over-reads the rate (moisture is not depleted/recharged).

Both are robustness-tiered the same way the absolute Stage C precip already is: the rectification SIGN
(more total convergent rain) and LOCATION (coastal/inland sea-breeze fronts) are the firm claim; the
mm/day is diagnostic-wind-soft.
"""
from __future__ import annotations
import warnings
import numpy as np
from dataclasses import dataclass
from . import geometry as g
from . import convergence as conv


@dataclass
class StageCDiurnalConfig:
    """Config for the diurnal-precip solve (the time-stepped Stage-C sibling of ``bl_diurnal``). The
    phase count K is INHERITED from the breeze solve (``sold``), not set here. ``test`` is a quick
    periodic solve; ``production`` tightens the periodicity tol, lifts the day guard, and checkpoints
    (see ``config.stage_c_diurnal``)."""
    cfl_safety: float = 0.8          # Courant safety factor for the moisture advective sub-step
    max_days: int = 40               # outer-loop day cap (guard)
    # day-to-day precip-cycle drift tolerance. Floored ~3e-3 by the land-ET recycling feedback (E_land =
    # recycle*self.P, and self.P is an EMA that carries a slow intra-/inter-day memory), so 5e-3 is the
    # practical periodic criterion; production can tighten with more days but will asymptote near the floor.
    period_tol: float = 5e-3
    period_patience: int = 2
    max_seconds: float = 300.0
    clip_wind: float = 80.0          # divergence guard [m/s] on pathological spikes -- NOT a physical cap
    checkpoint_path: str | None = None


def _wind_interp(phase_t, u_phase, v_phase, T_day):
    """Circular linear interpolation of the per-phase breeze wind in ``t mod T_day`` (the same uniform-
    phase lerp ``bl_diurnal.build_diurnal_forcing.phi_of`` uses). Returns ``wind_of(t) -> (u, v)``.

    Assumes UNIFORMLY-spaced phases (positions by ``K`` alone, like the #113 forcing) -- guarded loudly so
    a future non-uniform breeze solve fails here instead of silently mis-positioning the wind."""
    phase_t = np.asarray(phase_t, float)
    K = int(phase_t.size)
    dt_phase = T_day / K
    expected = np.arange(K) * dt_phase
    if not np.allclose(np.mod(phase_t, T_day), expected, atol=1e-6 * T_day):
        raise ValueError("stage_c_diurnal._wind_interp expects UNIFORMLY-spaced phases (phase 0 = noon, "
                         f"step T_day/K); got non-uniform phase_t={phase_t.tolist()}. The lerp positions by "
                         "K alone, so non-uniform phases would silently mis-position the breeze wind.")

    def wind_of(t=0.0):
        pos = float(np.mod(t, T_day)) / dt_phase
        k0 = int(np.floor(pos)) % K
        k1 = (k0 + 1) % K
        w = pos - np.floor(pos)
        return (1.0 - w) * u_phase[k0] + w * u_phase[k1], (1.0 - w) * v_phase[k0] + w * v_phase[k1]

    return wind_of


def _afternoon_phase(sold):
    """The afternoon sea-breeze peak phase index: the DAY phase (not night) maximizing the domain-mean
    wind speed. ``sold['afternoon_peak_speed']`` is a scalar SPEED field (per-cell max over the day) and
    is NOT a usable single phase -- the vector wind at ``k_aft`` is what the precip step needs."""
    spd = np.asarray(sold["speed_phase"], float)
    night = np.asarray(sold["night_mask"], bool)
    K = spd.shape[0]
    cand = np.where(~night)[0]
    if cand.size == 0:
        cand = np.arange(K)
    return int(cand[int(np.argmax([float(spd[k].mean()) for k in cand]))])


def _advance_to(Cmap, wind_of, t_model, t_target, dcfg):
    """Sub-cycle the moisture from ``t_model`` to ``t_target`` on the time-varying breeze, each sub-step
    at its own CFL ``dt`` (advective Courant + the explicit-diffusion stability limit), landing exactly
    on ``t_target``. Returns ``(t_model, P_now, conv, n_sub)`` -- the LAST sub-step's pre-EMA precip
    [mm/day] and convergence [1/s], and the sub-step count."""
    eps = 1e-6
    c = Cmap.cfg
    dx, de = Cmap.dx, Cmap.de
    # explicit 2-D Laplacian stability for the kappa_q moisture diffusion (dt <= h^2 / (4 kappa))
    dt_diff = 0.25 * min(dx, de) ** 2 / max(c.kappa_q, 1e-9)
    P_now = conv_now = None
    n_sub = 0
    while t_model < t_target - eps:
        u, v = wind_of(t_model)
        # divergence guard (NOT a physical cap): clip pathological wind spikes so a single bad cell can't
        # detonate the explicit advection; the breeze itself is left uncapped (the convergence is the signal).
        umax = float(np.max(np.abs(u)))
        vmax = float(np.max(np.abs(v)))
        if umax > dcfg.clip_wind or vmax > dcfg.clip_wind:
            u = np.clip(u, -dcfg.clip_wind, dcfg.clip_wind)
            v = np.clip(v, -dcfg.clip_wind, dcfg.clip_wind)
            umax = min(umax, dcfg.clip_wind)
            vmax = min(vmax, dcfg.clip_wind)
        dt_adv = dcfg.cfl_safety * min(dx / max(umax, 1e-9), de / max(vmax, 1e-9))
        # never step LARGER than the day-mean's proven-stable cfg.dt (the moisture source/sink relaxations
        # were tuned for it); sub-cycle SMALLER only when the stronger breeze's advective CFL demands it.
        dt = min(dt_adv, dt_diff, c.dt, t_target - t_model)
        P_now, conv_now = Cmap._step_with_wind(u, v, dt=dt)
        t_model += dt
        n_sub += 1
        if not (np.isfinite(Cmap.W_s).all() and np.isfinite(Cmap.W_a).all()):
            raise FloatingPointError("stage_c_diurnal moisture diverged (CFL/forcing blow-up)")
    return t_model, P_now, conv_now, n_sub


def solve_diurnal_precip(Cmap, sold, dcfg: StageCDiurnalConfig, clock=None, verbose=False):
    """PRIMARY: re-step the converged two-layer moisture through the diurnal breeze cycle to a PERIODIC
    state, resolving the afternoon convergent rain the day-mean averages away.

    ``Cmap`` is the CONVERGED ``StageCMap`` (its ``W_s``/``W_a`` are the day-mean warm start); ``sold`` is
    the ``bl_diurnal.solve_diurnal_wind`` result (per-phase winds ``u_phase``/``v_phase``, ``phase_t``,
    ``night_mask``, ``speed_phase``). The phase count K is inherited from ``sold``. Mutates ``Cmap``'s
    moisture to the periodic cycle (the cascade has already captured the day-mean deliverable ``dC`` by
    this point). Returns the resolved precip cycle + the rectification diagnostics."""
    clock = g.CLOCK if clock is None else clock
    T_day = clock.T_day
    phase_t = np.asarray(sold["phase_t"], float)
    u_phase = np.asarray(sold["u_phase"], float)
    v_phase = np.asarray(sold["v_phase"], float)
    K = phase_t.size
    wind_of = _wind_interp(phase_t, u_phase, v_phase, T_day)

    # warm-start is the already-converged day-mean moisture in Cmap. Reset the water-budget accumulators
    # so closure is measured over the LAST converged cycle, not dominated by the day-mean spin-up totals.
    Cmap._cum_E = 0.0
    Cmap._cum_P = 0.0

    mon = conv.ConvergenceMonitor(tol=dcfg.period_tol, patience=dcfg.period_patience,
                                  max_iter=dcfg.max_days, max_seconds=dcfg.max_seconds,
                                  name="diurnal precip cycle drift", absolute=True)
    P_phase = np.zeros((K, Cmap.nx, Cmap.ne))
    conv_phase = np.zeros((K, Cmap.nx, Cmap.ne))
    prev_cycle = None
    n_sub_max = 0
    last_closure = float("nan")
    day = 0
    for day in range(1, dcfg.max_days + 1):
        cumE0, cumP0 = Cmap._cum_E, Cmap._cum_P            # closure over THIS day only
        cycle = np.zeros((K, Cmap.nx, Cmap.ne))
        cyc_conv = np.zeros((K, Cmap.nx, Cmap.ne))
        t_model = (day - 1) * T_day
        # sample each phase at the END of a full dt_phase advance (no zero-length phase-0 interval): the
        # advances to phase_t[1..K-1] fill those phases; completing the day to day*T_day (== phase 0 / noon
        # of the next day) fills phase 0. Each interval is one dt_phase, so the wind is resolved throughout.
        for k in range(1, K):
            t_target = (day - 1) * T_day + phase_t[k]
            t_model, P_now, cv, ns = _advance_to(Cmap, wind_of, t_model, t_target, dcfg)
            cycle[k] = P_now
            cyc_conv[k] = cv
            n_sub_max = max(n_sub_max, ns)
        t_model, P_now, cv, ns = _advance_to(Cmap, wind_of, t_model, day * T_day, dcfg)
        cycle[0] = P_now                                   # phase 0 (noon) sample completes the day
        cyc_conv[0] = cv
        n_sub_max = max(n_sub_max, ns)
        denom = max(float(cycle.mean()), 1e-9)
        drift = float(np.max(np.abs(cycle - prev_cycle))) / denom if prev_cycle is not None else np.inf
        prev_cycle = cycle.copy()
        P_phase = cycle
        conv_phase = cyc_conv
        last_closure = (Cmap._cum_P - cumP0) / max(Cmap._cum_E - cumE0, 1e-12)
        if verbose:
            print(f"[stage_c_diurnal] day={day:>3} <P>={denom:6.2f} mm/day max={float(cycle.max()):7.2f} "
                  f"cycle-drift={drift:.2e} closure={last_closure:.3f} n_sub<= {n_sub_max}")
        stop = mon.update(day, drift if np.isfinite(drift) else 1.0)
        if dcfg.checkpoint_path and stop:
            conv.save_checkpoint(dcfg.checkpoint_path,
                                 dict(W_s=Cmap.W_s, W_a=Cmap.W_a, day=day), {"day": day})
        if stop:
            break

    if not mon.converged:
        warnings.warn(
            f"stage_c_diurnal periodic precip solve did NOT reach a periodic cycle ({mon.stop_reason}); "
            f"the reported P_phase is the LAST day and may still be drifting. A max_seconds stop is "
            f"wall-clock-dependent (NON-REPRODUCIBLE) -- raise max_days/max_seconds or loosen period_tol.",
            RuntimeWarning, stacklevel=2)

    night = np.asarray(sold["night_mask"], bool)
    k_aft = _afternoon_phase(sold)
    P_daymean_rect = P_phase.mean(axis=0)
    conv_daymean = conv_phase.mean(axis=0)
    P_amplitude = P_phase.max(axis=0) - P_phase.min(axis=0)
    # the diurnal SWING of convergence -- the mechanism-true partner of P_amplitude. The STATIC background
    # (balanced-wind) convergence is identical in every phase, so it cancels in max-min; what survives is the
    # BREEZE-driven convergence swing alone (exactly, in the linear/unclipped regime -- the clip_wind guard in
    # _advance_to never fires for physical breezes). That makes it the apples-to-apples field for the precip
    # swing (also a max-min), and -- unlike the instantaneous conv_afternoon snapshot, which the background
    # convergence dominates -- it is independent of the day-mean wind (e.g. the g_terrain orography steering).
    conv_amplitude = conv_phase.max(axis=0) - conv_phase.min(axis=0)
    return dict(phase_t=phase_t, P_phase=P_phase, conv_phase=conv_phase,
                P_daymean_rect=P_daymean_rect, conv_daymean=conv_daymean,
                k_aft=k_aft, P_afternoon=P_phase[k_aft], conv_afternoon=conv_phase[k_aft],
                P_amplitude=P_amplitude, conv_amplitude=conv_amplitude, night_mask=night,
                water_closure_last_cycle=float(last_closure),
                converged=bool(mon.converged), days=int(day), days_guard=int(dcfg.max_days),
                stop_reason=mon.stop_reason, n_sub_max=int(n_sub_max))


def _snapshot(Cmap):
    return (Cmap.W_s.copy(), Cmap.W_a.copy(), Cmap.P.copy(), Cmap.u.copy(), Cmap.v.copy(),
            Cmap._cum_E, Cmap._cum_P, Cmap.vapor_export, Cmap.iter)


def _restore(Cmap, snap):
    (Cmap.W_s, Cmap.W_a, Cmap.P, Cmap.u, Cmap.v,
     Cmap._cum_E, Cmap._cum_P, Cmap.vapor_export, Cmap.iter) = (
        snap[0], snap[1], snap[2], snap[3], snap[4], snap[5], snap[6], snap[7], snap[8])


def afternoon_precip_overlay(Cmap, sold, clock=None):
    """FALLBACK: a one-step KINEMATIC overlay. Holds the converged day-mean moisture FIXED and evaluates
    the precip once at the afternoon-peak breeze wind (and once at the static balanced wind for a matched
    day-mean reference), restoring ``Cmap``'s full state after each so it is a pure diagnostic (no
    moisture stepping). Returns the afternoon vs day-mean precip + convergence and their difference.
    Tier: sign/location firm; the rate over-reads (the standing moisture is not depleted/recharged)."""
    k_aft = _afternoon_phase(sold)
    u_aft = np.asarray(sold["u_phase"][k_aft], float)
    v_aft = np.asarray(sold["v_phase"][k_aft], float)

    snap = _snapshot(Cmap)
    try:
        P_aft, conv_aft = Cmap._step_with_wind(u_aft, v_aft)
    finally:
        _restore(Cmap, snap)

    snap = _snapshot(Cmap)
    try:
        P_dm, conv_dm = Cmap._step_with_wind(*Cmap._balanced_wind())   # matched-methodology day-mean
    finally:
        _restore(Cmap, snap)

    return dict(k_aft=k_aft, P_afternoon=P_aft, conv_afternoon=conv_aft,
                P_daymean=P_dm, conv_daymean=conv_dm, P_diff=P_aft - P_dm)


def precip_convergence_corr(P, convergence, mask=None):
    """Pearson correlation of precip with convergence (-div V) over ``mask`` (default: cells with any
    precip or convergence variability). The issue's headline metric (day-mean r ~ 0.59 on seed 1000)."""
    P = np.asarray(P, float).ravel()
    cvg = np.asarray(convergence, float).ravel()
    if mask is not None:
        m = np.asarray(mask, bool).ravel()
        P, cvg = P[m], cvg[m]
    if P.size < 2 or np.std(P) < 1e-12 or np.std(cvg) < 1e-12:
        return float("nan")
    return float(np.corrcoef(P, cvg)[0, 1])
