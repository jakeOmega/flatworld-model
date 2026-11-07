"""
flatworld.bl_momentum -- the NONLINEAR boundary-layer momentum solve for the (x, e) surface wind.

WHY (issue #111). Stage C's reported wind is a LINEAR friction balance V = u_bg - grad(Phi)/r_fric:
its magnitude scales as 1/r_fric (a chosen constant, never a prediction) and it violates the
frictionless Bernoulli ceiling of its own Phi field in ~26% of cells (more kinetic energy than the
potential drop the parcel descended -- internally inconsistent). This module replaces the REPORTED
wind with a prognostic momentum solve that is energetically SELF-LIMITING by construction.
  NOTE (W0, 2026-06-30, DIAGNOSTIC_WIND_W0_RESULTS.md): the ~26% Bernoulli-excess / ~80% cap-binding are
  L=2e7-era. On the canonical nsub30 world (L=8e7) BOTH re-measure to ~0% -- the diagnostic wind is gentle
  (|V| max ~9.5 m/s, far below the 40 m/s cap) and Bernoulli-consistent. The diagnostic-wind softness
  there is the DRAG-LAW MAGNITUDE (linear -grad(Phi)/r_fric vs this self-limiting momentum balance gives a
  ~1.25x speed ratio -> ~+36% band-mean precip when the moisture rides it), NOT the cap.

THE MODEL (fixed-Phi single-layer momentum; issue scope 1, the cheapest of the candidate scopes).
Time-step to steady state:

    dV/dt + (V.grad)V = -grad(Phi) - drag(V) + K_b lap(V) + a_kata

  * (V.grad)V        nonlinear momentum advection -- this is what bounds the speed: in steady state
                     B = 1/2|V|^2 + Phi is non-increasing along streamlines (drag/diffusion only
                     remove energy, -grad(Phi) is conservative), so |V|^2 <= 2*(Phi_ceiling - Phi).
                     The linear balance's Bernoulli violation is eliminated STRUCTURALLY, not capped.
  * -grad(Phi)       the THERMAL-LOW forcing (flatworld.bl_forcing, on VIRTUAL temperature). The reported
                     wind-Phi is thermal-only by default (g_terrain=0): treating terrain as a PE potential
                     would let air convert a whole multi-km mountain into a ~190 m/s free-fall, which is
                     NOT the climatological surface wind -- terrain wind effects live in the opt-in
                     katabatic term (cold drainage, below) and issue #114 (downslope windstorms/gap jets),
                     and terrain roughness is in surface_wind.py. Phi enters as a callable phi_of(t)
                     (static by default) so the diurnal sea/land breeze (#113) is a forcing swap.
  * the mean/anomaly split: the AD core ALREADY resolved the zonal-mean u_surf(theta) with its own drag,
                     so this solver relaxes the zonal mean <V>_e toward V_bg=(u_bg,0) at rate r_relax (the
                     structural u_surf reproduction -- NOT a fit) and applies surface drag only to the
                     ANOMALY V'=V-<V>_e (re-draging the mean would double-count the AD core's drag).
  * drag(V')         surface friction on the anomaly. DEFAULT 'quadratic': -C_D|V|V'/H_bl (physical surface
                     stress, self-limiting). 'linear': -r_mom*V' (the sensitivity arm: AD core 6.1e-6, the
                     sh_b23dx4 x4 probe 2.44e-5). The drag closure is SWEPT -> the honest magnitude band.
  * K_b lap(V)       small eddy viscosity (regularizes the upwind advection at supercritical fronts).
  * a_kata           OPTIONAL katabatic/slope-flow term off cold high terrain (built by
                     katabatic_accel; default OFF / byte-identical baseline). Issue scope 2.

Reuses the shared wall/periodic operators (flatworld.operators): walls in cross-track x (equator x[0],
pole x[-1], one-sided), periodic in along-track e. Advection is the consistent upwind ADVECTIVE
derivative -(V.grad)V (flatworld.operators.upwind_adv) -- the NON-conservative material derivative, which
is the correct form for a DIVERGENT boundary-layer flow (the conservative flux form -div(Vu) would inject
a spurious u*div(V)), single-discretization upwinded so it is stable.

DIAGNOSTICS (issue asks for more than u,v -- the siblings #112/#114 read these): per-cell Froude number
Fr=|V|/c with c^2 = 2*max(Phi_row_max - Phi, c_floor^2) (Fr<=1 <=> within the Phi Bernoulli ceiling),
boundary-layer depth proxy c^2/g_reduced, the along-track wind divergence, and the Bernoulli-excess
fraction (the #111 acceptance metric).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from . import operators as ops
from . import constants as C
from . import convergence as conv
from . import bl_forcing as blf


@dataclass
class BLMomentumConfig:
    # --- the mean/anomaly split (issue #111 design) ---
    # The AD core ALREADY resolved the zonal-mean surface wind u_surf(theta) WITH its own surface drag, so
    # this (x,e) solver owns only the ALONG-TRACK ANOMALY. The zonal mean <V>_e is relaxed toward V_bg=
    # (u_bg,0) at rate r_relax (this is what reproduces u_surf -- a STRUCTURAL match, not a fit), and the
    # surface drag below damps the ANOMALY V' = V - <V>_e (re-draging the mean would double-count the AD
    # core's drag). Set r_relax=0 to fall back to whole-field drag relaxing to V_bg.
    r_relax: float = 1.0e-4           # zonal-mean relaxation rate toward u_bg [1/s]
    # --- surface drag on the anomaly (the SWEPT uncertainty -> the honest magnitude error bar) ---
    # DEFAULT 'quadratic': the physically honest surface stress -C_D|V|V'/H_bl (rate scales with total
    # wind speed) -- self-limiting, no arbitrary linear rate. 'linear' (-r_mom*V') is the sensitivity arm
    # (incl. the AD core 6.1e-6 and the sh_b23dx4 x4 probe 2.44e-5).
    drag_form: str = "quadratic"       # 'quadratic' (-C_D|V|V'/H_bl) | 'linear' (-r_mom*V')
    r_mom: float = 6.1e-6              # linear anomaly-drag rate [1/s]; default = the AD core bulk-drag rate
                                       #   (C_D*U/dz, nz=18). Weak -> the anomaly reaches its thermal Bernoulli
                                       #   ceiling sqrt(2*beta_T*dT) ~ tens of m/s. SWEPT, not fitted.
    C_D: float = 1.2e-3               # bulk drag coefficient (quadratic form)
    H_bl: float = 1000.0              # boundary-layer depth [m] (quadratic drag scale)
    # --- wind-Phi build (DECOUPLED from Stage C's moisture-Phi, which keeps its own beta_T/g_terrain/
    #     phi_smooth_km and dry T for the moisture-advection numerics). Terrain OFF by default (see above). ---
    virtual_temp: bool = True         # fold vapor buoyancy into the wind-Phi (issue #80); False -> dry T
    phi_beta_T: float = 30.0          # thermal-low strength [m^2/s^2/K]
    phi_g_terrain: float = 0.0        # terrain PE factor in the wind-Phi (0 = thermal-only reported wind)
    phi_smooth_km: float = 400.0      # wind-Phi smoothing length [km] (frictional adjustment fetch)
    # --- forcing-derived diagnostics ---
    g_reduced: float = 0.1            # reduced gravity [m/s^2] for the layer-depth proxy bl_depth=c^2/g'
    c_floor: float = 2.0              # floor wave speed [m/s] (avoids div-by-0 in Froude on Phi-flat rows)
    # --- regularization / time stepping ---
    K_b: float = 5.0e3                # eddy viscosity [m^2/s] (upwind regularizer)
    cfl: float = 0.35                 # CFL safety factor on the adaptive dt
    scheme: str = "rk2"              # 'rk2' (Heun) | 'euler'
    # --- two-layer momentum wind (W1, DIAGNOSTIC_WIND_ROADMAP; opt-in, default-OFF byte-identical) ---
    # bl_momentum solves the SURFACE BL field; Stage C's moisture is two-layer and W_a rides the
    # POLEWARD-ALOFT return. W1 scope (a) -- the MINIMAL fork chosen by the W0 evidence (the swap moved the
    # rate, not the peak placement, so the convergence pattern is preserved) -- carries the aloft wind as
    # the existing kinematic DEPARTURE off the momentum surface wind (aloft_departure_wind, the single
    # source of truth for Stage C's inline algebra), and INSTRUMENTS its Bernoulli/divergence budget so the
    # (a)-vs-(b) decision rests on evidence. two_layer=False -> solve_two_layer_wind returns only the
    # surface solve (byte-identical to build_wind_field). u_aloft_frac / aloft_departure_sign MIRROR the
    # StageCConfig defaults so the module reproduces what Stage C actually advects.
    two_layer: bool = False
    u_aloft_frac: float = 1.0          # aloft poleward wind = u_aloft_frac*|ubg| (opposite the surface)
    aloft_departure_sign: float = 1.0  # sign of the dynamical departure the aloft layer carries
    aloft_phi_gamma: float = 1.0       # DIAGNOSTIC aloft-Phi reference Phi_aloft = -gamma*Phi_surf (the
                                       # leading-order two-layer baroclinic flip: the warm column is a
                                       # surface LOW and an aloft HIGH). Only the aloft Bernoulli diagnostic
                                       # reads it; nothing prognostic depends on gamma, and the (a)-vs-(b)
                                       # call is a surface-vs-aloft RATIO so it is robust to the gamma scale.
    # --- katabatic (opt-in, issue scope 2) ---
    katabatic: bool = False
    c_kata: float = 1.0               # katabatic efficiency (O(1))
    # --- convergence / guards ---
    tol: float = 1.0e-4               # steady-state tol on normalized max|dV|/<|V|>/step
    patience: int = 4
    check_every: int = 50
    max_iter: int = 400_000
    max_seconds: float = 600.0
    checkpoint_path: str | None = None
    checkpoint_every: int = 20_000


def momentum_smoothing_km(cfg: BLMomentumConfig, V_char: float | None = None):
    """The self-limiting momentum field's OWN frictional/advective adjustment length [km] -- the W2
    re-derivation that REPLACES the diagnostic balance's cap-saturated `wind_cap/r_fric = 400 km`
    (stage_c_map.frictional_smoothing_km). That 400 km is a *cap-saturated-wind fixed point*: it assumes the
    operative wind is pinned at the 40 m/s cap. The momentum wind has NO cap, so the operative length must be
    re-derived from the momentum solve's OWN drag closure (DIAGNOSTIC_WIND_ROADMAP W2 / risk item):

      * quadratic drag (DEFAULT): the advective adjustment fetch is where nonlinear advection V*grad(V) ~
        V^2/L balances quadratic surface drag C_D*|V|*V'/H_bl ~ C_D*V^2/H_bl, i.e. L ~ H_bl/C_D -- which is
        INDEPENDENT of V_char. The cap-saturation ambiguity ("which V_char?") therefore DISSOLVES
        structurally, exactly the roadmap's "no cap -> the fixed point dissolves". Default (H_bl=1000 m,
        C_D=1.2e-3) -> 833 km (~2x the diagnostic 400 km: the gentler, un-capped wind integrates over a
        longer fetch).
      * linear drag: the frictional relaxation rate IS r_mom (speed-independent), so the fetch is the
        measured momentum characteristic speed over that rate, L = V_char/r_mom. Needs V_char (the solved
        momentum field's mean speed); falls back to wind_cap if unset (the conservative cap-length).

    Returned in km (matches frictional_smoothing_km / phi_smooth_km units)."""
    if cfg.drag_form == "quadratic":
        return cfg.H_bl / cfg.C_D / 1.0e3
    V = float(V_char) if V_char is not None else 40.0     # linear arm: measured V_char, else the cap-length
    return V / cfg.r_mom / 1.0e3


def katabatic_accel(T_map_C, height, dx, de, cfg: BLMomentumConfig):
    """Prandtl slope-flow acceleration (ax, ay) [m/s^2] off COLD high terrain: cold dense air drains
    DOWN-slope. a = c_kata * g' * (dT_cold/T_ref) * (-grad h), with dT_cold = max(Tref_row - T, 0) >= 0
    (cold anomaly only -> never an up-slope katabatic). Returns (0,0) fields when disabled, so the
    baseline is byte-identical. Stability on extreme cold/steep terrain rests on the CFL-adaptive dt +
    the NaN guard (no explicit acceleration cap); add one if a real ice-edge/highland run blows up."""
    if not cfg.katabatic:
        z = np.zeros_like(np.asarray(height, float))
        return z, z.copy()
    T = np.asarray(T_map_C, float)
    Tref = np.mean(T, axis=1, keepdims=True)
    dT_cold = np.maximum(Tref - T, 0.0)                  # [K], >=0 where colder than the row mean
    T_ref_K = np.mean(T) + C.T_FREEZE
    coeff = cfg.c_kata * cfg.g_reduced * (dT_cold / T_ref_K)
    ax = coeff * (-ops.ddx(height, dx))
    ay = coeff * (-ops.dde(height, de))
    return ax, ay


def _advect(u, v, dx, de):
    """Material-derivative advection -(V.grad)u and -(V.grad)v via the consistent upwind ADVECTIVE
    derivative (flatworld.operators.upwind_adv) -- the correct form for a divergent BL flow, stable."""
    return ops.upwind_adv(u, u, v, dx, de), ops.upwind_adv(v, u, v, dx, de)


def _tendency(u, v, gPhi_x, gPhi_e, u_bg, v_bg, ax_k, ay_k, dx, de, cfg: BLMomentumConfig):
    adv_u, adv_v = _advect(u, v, dx, de)
    # mean/anomaly split: relax the zonal mean toward V_bg (=u_surf), drag the anomaly.
    um = np.mean(u, axis=1, keepdims=True); vm = np.mean(v, axis=1, keepdims=True)
    ubgm = np.mean(u_bg, axis=1, keepdims=True)
    relax_u = -cfg.r_relax * (um - ubgm)
    relax_v = -cfg.r_relax * (vm - 0.0)
    up, vp = u - um, v - vm                              # along-track anomaly
    if cfg.drag_form == "quadratic":
        # surface-stress rate scales with the TOTAL wind speed |V| (physical), decelerating the anomaly.
        rq = cfg.C_D * np.sqrt(u * u + v * v) / cfg.H_bl
        drag_u, drag_v = -rq * up, -rq * vp
    else:  # linear drag on the anomaly
        drag_u, drag_v = -cfg.r_mom * up, -cfg.r_mom * vp
    dudt = adv_u - gPhi_x + relax_u + drag_u + cfg.K_b * ops.lap(u, dx, de) + ax_k
    dvdt = adv_v - gPhi_e + relax_v + drag_v + cfg.K_b * ops.lap(v, dx, de) + ay_k
    return dudt, dvdt


def _cfl_dt(u, v, dx, de, cfg: BLMomentumConfig):
    umax = max(float(np.max(np.abs(u))), 1.0)
    vmax = max(float(np.max(np.abs(v))), 1.0)
    dt_adv = min(dx / umax, de / vmax)
    dt_diff = 0.5 / (cfg.K_b * (1.0 / dx**2 + 1.0 / de**2) + 1e-30)
    return cfg.cfl * min(dt_adv, dt_diff)


def solve_surface_wind(phi, ubg, dx, de, cfg: BLMomentumConfig | None = None,
                       T_map_C=None, height=None, u0=None, v0=None, verbose=False):
    """Solve the steady boundary-layer momentum field on the (x, e) grid.

    phi      : (nx, ne) geopotential array OR a callable phi_of(t) -> (nx, ne) (flatworld.bl_forcing).
    ubg      : (nx,) or (nx, ne) background cross-track wind [m/s] (the AD core's u_surf, broadcast row-wise).
    dx, de   : grid spacings [m].
    T_map_C, height : needed only for the optional katabatic term (cfg.katabatic).
    Returns a dict: u, v, speed, froude, wind_div, bl_depth, wave_c2, bernoulli_excess_frac, iters,
    converged, stop_reason, dt.
    """
    cfg = cfg or BLMomentumConfig()
    static = not callable(phi)
    phi_of = (lambda t=0.0: phi) if static else phi
    Phi0 = np.asarray(phi_of(0.0), float)
    nx, ne = Phi0.shape
    u_bg = np.broadcast_to(np.asarray(ubg, float).reshape(-1, 1) if np.ndim(ubg) == 1 else ubg,
                           (nx, ne)).astype(float)
    v_bg = np.zeros((nx, ne))
    u = u_bg.copy() if u0 is None else np.asarray(u0, float).copy()
    v = np.zeros((nx, ne)) if v0 is None else np.asarray(v0, float).copy()
    t_model = 0.0
    it = 0
    # resume from a checkpoint if one exists (the guarded run resumes, not restarts)
    if cfg.checkpoint_path and u0 is None and v0 is None:
        st, _ = conv.load_checkpoint(cfg.checkpoint_path)
        if st is not None:
            u, v = np.asarray(st["u"], float), np.asarray(st["v"], float)
            it = int(st.get("iter", 0)); t_model = float(st.get("t_model", 0.0))
            if verbose: print(f"[bl_mom] resumed at iter {it}")
    ax_k, ay_k = katabatic_accel(T_map_C if T_map_C is not None else Phi0, height, dx, de, cfg) \
        if cfg.katabatic else (0.0, 0.0)
    # static Phi: gradients are constant -> hoist out of the step loop (the common case, ~2x faster).
    gPx0, gPe0 = (ops.ddx(Phi0, dx), ops.dde(Phi0, de)) if static else (None, None)

    mon = conv.ConvergenceMonitor(tol=cfg.tol, patience=cfg.patience, max_iter=cfg.max_iter,
                                  max_seconds=cfg.max_seconds, name="max|dV|/<|V|>/step", absolute=True)
    V_prev = np.hypot(u, v).copy()
    dt = _cfl_dt(u, v, dx, de, cfg)
    while True:
        for _ in range(cfg.check_every):
            if static:
                gPx, gPe = gPx0, gPe0
            else:
                gPx, gPe = ops.ddx(np.asarray(phi_of(t_model), float), dx), \
                           ops.dde(np.asarray(phi_of(t_model), float), de)
            dudt, dvdt = _tendency(u, v, gPx, gPe, u_bg, v_bg, ax_k, ay_k, dx, de, cfg)
            dt = _cfl_dt(u, v, dx, de, cfg)
            if cfg.scheme == "euler":
                u = u + dt * dudt
                v = v + dt * dvdt
            else:  # RK2 / Heun: the corrector sees the forcing at t+dt (matters only for time-varying Phi)
                if static:
                    gPx2, gPe2 = gPx0, gPe0
                else:
                    Phi2 = np.asarray(phi_of(t_model + dt), float)
                    gPx2, gPe2 = ops.ddx(Phi2, dx), ops.dde(Phi2, de)
                u1, v1 = u + dt * dudt, v + dt * dvdt
                d2u, d2v = _tendency(u1, v1, gPx2, gPe2, u_bg, v_bg, ax_k, ay_k, dx, de, cfg)
                u = u + 0.5 * dt * (dudt + d2u)
                v = v + 0.5 * dt * (dvdt + d2v)
            it += 1
            t_model += dt                                  # accumulated model time (NOT it*dt -- dt adapts)
            if not (np.isfinite(u).all() and np.isfinite(v).all()):
                raise FloatingPointError(f"bl_momentum diverged at iter {it} (CFL/forcing blow-up)")
        spd = np.hypot(u, v)
        denom = max(float(np.mean(spd)), 1e-9)
        drift = float(np.max(np.abs(spd - V_prev))) / denom / cfg.check_every
        V_prev = spd.copy()
        stop = mon.update(it, drift)
        if verbose and (it % (cfg.check_every * 20) == 0 or stop):
            print(f"[bl_mom] it={it:>7} <|V|>={denom:6.1f} max|V|={float(spd.max()):6.1f} "
                  f"drift={drift:.2e} dt={dt:.0f}")
        if cfg.checkpoint_path and it % cfg.checkpoint_every == 0:
            conv.save_checkpoint(cfg.checkpoint_path, dict(u=u, v=v, iter=it, t_model=t_model), {"iter": it})
        if stop:
            break

    Phi = np.asarray(phi_of(t_model), float)
    return _diagnostics(u, v, Phi, dx, de, cfg, mon, dt)


def bernoulli_excess_frac(u, v, phi):
    """Fraction of cells whose Phi-DRIVEN kinetic energy exceeds the frictionless Bernoulli ceiling of
    the driving Phi field -- the linear balance's failure mode (#111). The ceiling is the GLOBAL drop
    Phi_max - Phi_local (cross-track u lets a parcel descend from anywhere in the connected domain), and
    the Phi-driven part is the ANOMALY V'=V-<V>_e (the imposed zonal mean is externally-set momentum, not
    descended from Phi), so the test is 1/2|V'|^2 <= (Phi_max - Phi_local). The SAME measurement applied to
    the linear balance and the momentum field gives the apples-to-apples comparison -- see _bl_wind_sweep.
    NOTE: this differs from the OPEN_ISSUES-history '26%' (total speed vs PER-ROW ceiling); it is the
    consistent metric the solver is designed against."""
    u = np.asarray(u, float); v = np.asarray(v, float); phi = np.asarray(phi, float)
    up = u - u.mean(axis=1, keepdims=True); vp = v - v.mean(axis=1, keepdims=True)
    dPhi = np.maximum(float(phi.max()) - phi, 0.0)
    return float(np.mean(0.5 * (up * up + vp * vp) > dPhi + 1e-6))


def bernoulli_excess_frac_perrow(u, v, phi):
    """The HISTORICAL per-row Bernoulli-excess metric (the OPEN_ISSUES '26%' figure): the ceiling is the
    PER-ROW drop max_e(phi_row) - phi_local (a parcel descends only within its own cross-track row),
    NOT the global max_e bernoulli_excess_frac uses. Strictly STRICTER than the global metric (a smaller
    ceiling), so it reports a LARGER excess for the same wind. Kept so the '26% (per-row, L=2e7) -> 0%
    (nsub30)' re-measurement can be quoted on a like-for-like metric, separating the WORLD change (gentler
    L=8e7 wind) from the METRIC change (per-row -> global ceiling). Anomaly V'=V-<V>_e as in the global
    version."""
    u = np.asarray(u, float); v = np.asarray(v, float); phi = np.asarray(phi, float)
    up = u - u.mean(axis=1, keepdims=True); vp = v - v.mean(axis=1, keepdims=True)
    dPhi = np.maximum(phi.max(axis=1, keepdims=True) - phi, 0.0)
    return float(np.mean(0.5 * (up * up + vp * vp) > dPhi + 1e-6))


def _diagnostics(u, v, Phi, dx, de, cfg, mon, dt):
    spd = np.hypot(u, v)
    # LOCAL Froude: speed vs the along-track density-current wave speed c^2 = 2*(row-max Phi - Phi)
    # (e is periodic, so a parcel accesses its whole cross-track row). Fr>1 flags supercritical/drainage
    # flow -- a meaningful regime diagnostic (read by #114), not an error. Floored to avoid div-by-0.
    dPhi_row = np.maximum(np.max(Phi, axis=1, keepdims=True) - Phi, 0.0)
    wave_c2 = np.maximum(2.0 * dPhi_row, cfg.c_floor**2)
    froude = spd / np.sqrt(wave_c2)
    bl_depth = wave_c2 / cfg.g_reduced
    wind_div = ops.div(u, v, dx, de)
    return dict(u=u, v=v, speed=spd, froude=froude, wind_div=wind_div, bl_depth=bl_depth,
                wave_c2=wave_c2, dPhi_row=dPhi_row, bernoulli_excess_frac=bernoulli_excess_frac(u, v, Phi),
                iters=mon.history[-1][0] if mon.history else 0, converged=mon.converged,
                stop_reason=mon.stop_reason, dt=float(dt))


def build_wind_field(dC, dx, de, cfg: BLMomentumConfig | None = None, verbose=False):
    """Post-process entry point: from a converged Stage C diagnostics dict ``dC`` (needs T_map_C, W_s,
    height, ubg), build the virtual-temperature wind-Phi and solve the nonlinear BL momentum field.
    Returns the solver result dict plus the built Phi. This is what the cascade calls to switch the
    REPORTED wind from the linear balance to the momentum solve."""
    cfg = cfg or BLMomentumConfig()
    T_map_C = np.asarray(dC["T_map_C"], float)
    W_s = dC.get("W_s") if cfg.virtual_temp else None
    Tv = blf.virtual_T_map(T_map_C, W_s)
    Phi = blf.geopotential(Tv, dC["height"], dx, de, beta_T=cfg.phi_beta_T,
                           g_terrain=cfg.phi_g_terrain, phi_smooth_km=cfg.phi_smooth_km)
    sol = solve_surface_wind(Phi, dC["ubg"], dx, de, cfg, T_map_C=T_map_C, height=dC["height"],
                             verbose=verbose)
    sol["phi"] = Phi
    return sol


# ============================ W1: the two-layer momentum wind ============================
# bl_momentum above solves the SURFACE boundary-layer field; the Stage-C moisture is TWO-LAYER and the
# aloft column W_a rides the poleward-ALOFT return. W1 (DIAGNOSTIC_WIND_ROADMAP) makes the moisture ride a
# momentum-consistent wind on BOTH layers. The W0 evidence (the swap moved the band RATE, not the peak
# PLACEMENT -> the convergence pattern is preserved) selected the MINIMAL scope (a): keep the existing
# kinematic aloft DEPARTURE, but off the momentum surface wind, and INSTRUMENT its consistency to decide
# whether (b) -- a genuine second momentum solve for the return branch -- is warranted.


def aloft_departure_wind(u, v, ubg, u_aloft_frac=1.0, aloft_departure_sign=1.0):
    """The poleward-aloft return wind as the kinematic DEPARTURE off a surface wind (u, v) -- the SINGLE
    source of truth for this algebra, called by ``stage_c_map._step_with_wind`` (no longer a duplicate;
    equivalence to the prior inline expression is locked by test_aloft_departure_wind_matches_stage_c_inline):

        u_aloft = -u_aloft_frac*ubg + aloft_departure_sign*(u - ubg)
        v_aloft =                     aloft_departure_sign*v

    i.e. the poleward MEAN (opposite the surface, magnitude u_aloft_frac*|ubg|) plus the signed dynamical
    departure of the surface wind from its background. ``ubg`` is (nx,) or (nx, ne) and is broadcast
    row-wise. With (a) this rides the MOMENTUM surface wind; nothing here is new physics -- it is the
    existing two-layer cartoon, now driven by a momentum-consistent surface field."""
    u = np.asarray(u, float); v = np.asarray(v, float)
    nx, ne = u.shape
    ubg2d = np.broadcast_to(np.asarray(ubg, float).reshape(-1, 1) if np.ndim(ubg) == 1 else ubg, (nx, ne))
    u_aloft = -u_aloft_frac * ubg2d + aloft_departure_sign * (u - ubg2d)
    v_aloft = aloft_departure_sign * v
    return u_aloft, v_aloft


def two_layer_diagnostics(u, v, u_aloft, v_aloft, phi_surf, dx, de, cfg: BLMomentumConfig):
    """Instrument the aloft branch so the W1 (a)-vs-(b) decision is evidence-based. Returns, for BOTH
    layers, the speed stats, the wind divergence (the convergence that places rain), and the consistent
    ``bernoulli_excess_frac``. The aloft layer has no independent prognostic Phi in the two-layer cartoon,
    so its Bernoulli ceiling uses the leading-order baroclinic reference Phi_aloft = -gamma*Phi_surf (the
    warm column is a surface low and an aloft high; gamma = cfg.aloft_phi_gamma). The decision is the
    SURFACE-vs-ALOFT comparison: scope (a) is sufficient when the aloft Bernoulli-excess is NOT materially
    worse than the surface's (the algebraic aloft wind inherits the surface field's self-limitation);
    scope (b) is warranted only if the aloft branch is markedly Bernoulli-violating (its KE exceeds the
    upper-level pressure drop -- a real second momentum solve would then be needed to bound it)."""
    phi_aloft = -float(cfg.aloft_phi_gamma) * np.asarray(phi_surf, float)
    spd_s = np.hypot(np.asarray(u, float), np.asarray(v, float))
    spd_a = np.hypot(np.asarray(u_aloft, float), np.asarray(v_aloft, float))
    be_s = bernoulli_excess_frac(u, v, phi_surf)
    be_a = bernoulli_excess_frac(u_aloft, v_aloft, phi_aloft)
    # decision rule: aloft "self-limiting enough" if its excess is within an absolute 5% floor OR within
    # 1.5x the surface's excess (a ratio is meaningless when the surface excess is ~0, the nsub30 case).
    aloft_ok = (be_a <= 0.05) or (be_a <= 1.5 * be_s + 0.02)
    return dict(
        surf_speed_mean=float(spd_s.mean()), surf_speed_max=float(spd_s.max()),
        aloft_speed_mean=float(spd_a.mean()), aloft_speed_max=float(spd_a.max()),
        surf_div=ops.div(u, v, dx, de), aloft_div=ops.div(u_aloft, v_aloft, dx, de),
        surf_bernoulli_excess_frac=be_s, aloft_bernoulli_excess_frac=be_a,
        aloft_ok=bool(aloft_ok),
        scope_decision=("(a) sufficient -- aloft inherits the surface self-limitation" if aloft_ok
                        else "(b) warranted -- aloft branch is Bernoulli-violating, needs a return solve"))


def solve_two_layer_wind(dC, dx, de, cfg: BLMomentumConfig | None = None, verbose=False):
    """W1 entry point: the TWO-LAYER momentum wind. Solves the surface momentum field (build_wind_field),
    and -- when ``cfg.two_layer`` is set -- adds the poleward-aloft return as the kinematic departure off
    that surface wind (scope (a)) plus the aloft Bernoulli/divergence instrumentation and the (a)-vs-(b)
    verdict. ``cfg.two_layer=False`` (DEFAULT) returns ONLY the surface solve -- byte-identical to
    build_wind_field (the gate's "reduces to the surface solve when the aloft term is off"). The returned
    dict adds (when two_layer): u_aloft, v_aloft, and a ``two_layer`` sub-dict of diagnostics."""
    cfg = cfg or BLMomentumConfig()
    sol = build_wind_field(dC, dx, de, cfg, verbose=verbose)
    if not cfg.two_layer:
        return sol                                              # byte-identical surface-only solve
    u_aloft, v_aloft = aloft_departure_wind(sol["u"], sol["v"], dC["ubg"],
                                            cfg.u_aloft_frac, cfg.aloft_departure_sign)
    diag = two_layer_diagnostics(sol["u"], sol["v"], u_aloft, v_aloft, sol["phi"], dx, de, cfg)
    sol["u_aloft"], sol["v_aloft"] = u_aloft, v_aloft
    sol["two_layer"] = diag
    if verbose:
        print(f"  two-layer aloft: <|V|>={diag['aloft_speed_mean']:.1f} max={diag['aloft_speed_max']:.1f} m/s "
              f"| Bernoulli-excess surf {100*diag['surf_bernoulli_excess_frac']:.1f}% "
              f"aloft {100*diag['aloft_bernoulli_excess_frac']:.1f}% -> {diag['scope_decision']}")
    return sol
