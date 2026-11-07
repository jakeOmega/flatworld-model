"""
flatworld.cascade -- run the full A -> B -> C pipeline and wire each stage into the next.

Stage A (cheap RCE-EBM backbone) predicts the absolute SST(theta) and a land-temperature backbone;
Stage B (authoritative zonal-mean overturning) takes A's SST as its buoyancy forcing and derives the
circulation + the poleward-aloft rain band F(theta); Stage C (the dynamical 2-D map) consumes A's
SST/land-T and B's band over a topology to produce the deliverable T(x,e) and P(x,e) maps, which are
then classified. See DESIGN.md for the authority/coupling rationale (B is authoritative for the
zonal mean; A is the initializer/sweep engine).
"""
from __future__ import annotations
import numpy as np
from collections.abc import Mapping
from scipy.interpolate import interp1d
from . import constants as C
from . import geometry as g
from . import surface as surf
from . import config as cfgmod
from .stage_a_column import StageA
from .stage_b_overturning import StageBAtmosphere, dimensional_poleward_flux
from .climate_map import StageCMap
from .topography import Topography, TopoGrid
from .topo_contract import TopographyContract
from . import classify


def _interp(theta_deg, values):
    # Linear in theta, CLAMPED outside the data range (fill_value=(first, last)) -- never extrapolated;
    # callers rely on the clamp at the theta-domain edges. (The former `fill="extrapolate"` kwarg was
    # dead -- interp1d always clamped here regardless -- so it is removed, issue #88 item 4.)
    f = interp1d(theta_deg, values, bounds_error=False, fill_value=(values[0], values[-1]))
    return lambda th: f(np.asarray(th, float))


def _interp_smooth(theta_deg, values):
    # Shape-preserving C1 interpolant (PCHIP: monotone between nodes, no overshoot), clamped outside
    # the data range like _interp. Use where a piecewise-LINEAR interp's derivative kinks at coarse
    # backbone nodes print as visible artifacts in a dense downstream field -- the Stage C wind rows:
    # over Phi-flat open sea the plotted speed IS ubg(theta), so the ~48-node backbone's tent kinks
    # rendered as horizontal banding in the wind panel (the seed67 stripes).
    from scipy.interpolate import PchipInterpolator
    th = np.asarray(theta_deg, float); va = np.asarray(values, float)
    f = PchipInterpolator(th, va, extrapolate=False)
    return lambda x: f(np.clip(np.asarray(x, float), th[0], th[-1]))


# --- single source of truth for the poleward-aloft band target F(theta) [mm] (issue #88 item 4) ---
# Both backbones (this A->B->C cascade AND the fused AD-core) and the standalone moisture diagnostics map
# Stage B's NORMALIZED [0,1] precip-band SHAPE to a moisture-column target by the same affine law: an ~8 mm
# floor rising to ~50 mm at the band peak. It lived as a duplicated `8.0 + 42.0*shape` magic literal in
# both backbones (a drift hazard); hoisted here so a change moves in lockstep.
F_FLOOR_MM = 8.0          # aloft-supply column floor [mm] (band shape = 0)
F_SPAN_MM = 42.0          # span added at the band peak [mm] (band shape = 1 -> floor+span = ~50 mm)


def aloft_supply_target(band_shape):
    """Map a NORMALIZED [0,1] precip-band shape to the aloft moisture-supply column F [mm]."""
    return F_FLOOR_MM + F_SPAN_MM * np.asarray(band_shape, float)


# Chimney guard for the ABSOLUTE aloft-column supply: aloft latitudes equatorward of this theta carry the
# equatorial convective chimney (the 1/dx precip singularity at theta~0) and the hot inner sea. The
# Stage-C ribbon domain starts ~here; equatorward of it the absolute F is clamped so the chimney/inner-sea
# column cannot leak into the ribbon supply.
ALOFT_CHIMNEY_THETA = 28.0   # deg


def _legacy_supply(precip_mm_day):
    """The legacy NORMALIZE-TO-SHAPE law: clip precip, divide by its own max -> [0,1] shape, re-impose the
    affine 8-50 mm scale. Byte-identical to the historic inline `8 + 42*(precip/precip.max())`."""
    band = np.clip(np.asarray(precip_mm_day, float), 0.0, None)
    if band.max() > 0:
        band = band / band.max()
    return aloft_supply_target(band)


def build_aloft_supply_target(theta_deg, precip_mm_day, aloft_moisture_mm=None, absolute=False,
                              chimney_theta=ALOFT_CHIMNEY_THETA):
    """The poleward-aloft moisture-supply band target F(theta) [mm] handed to Stage C. Two laws:

    * ``absolute=False`` (DEFAULT, byte-identical): the legacy NORMALIZE-TO-SHAPE law (``_legacy_supply``).
      Carried verbatim from the nondimensional Stage B; on the AD core it DISCARDS the resolved absolute
      precip magnitude, and the equatorial chimney's 1/dx spike (the global precip.max(), ~362 mm/day on
      the canonical world vs a ~20 mm/day band peak) dominates the /max() and crushes the ribbon supply
      toward the 8 mm floor (~67 % of ribbon cells pinned at the floor; a real ~22 mm band cell -> ~8.8 mm).

    * ``absolute=True`` (the AD-core fix, OPT-IN): use the RESOLVED ALOFT MOISTURE COLUMN
      ``aloft_moisture_mm`` [mm] directly (StageADCore._aloft_moisture_column) -- the core's own absolute
      prediction of the moisture aloft to carry poleward. No normalization, no re-imposed scale; the
      chimney can't dominate (a water-substance COLUMN is qsat-bounded, no 1/dx blow-up). Belt-and-braces:
      the equatorward (theta < ``chimney_theta``) values are clamped to the chimney_theta value so the
      inner-sea/chimney column never leaks into the Stage-C ribbon domain. Requires ``aloft_moisture_mm``
      (only the AD core supplies it; nondimensional Stage B has no absolute column) -> falls back to the
      legacy law (with a warning) if missing.

    SOFTNESS (named wall): the absolute aloft column's MAGNITUDE is closure-/diagnostic-soft until a
    CONVERGED AD-core column state pins it (GPU/M3) -- the test-profile bounded run's vapour field is not
    equilibrated. The band SHAPE/location and the removal of the chimney-normalization artifact are firm;
    the absolute mm is staged for the converged run (B5d). NB this is a unit-REGIME change: the aloft
    vapour column above sigma<0.5 is plausibly ~1-5 mm (a few % of the column), possibly BELOW the legacy
    8 mm floor, so flipping ON may LOWER ribbon F, not raise it -- the sign is unknown until the converged
    run, and B5d must validate the ribbon-P RESPONSE (and may need to retune tau_supply), not just that F
    moved. See FTHETA_SUPPLY_RESULTS.md."""
    theta_deg = np.asarray(theta_deg, float)
    if not absolute:
        return _legacy_supply(precip_mm_day)
    if aloft_moisture_mm is None:
        import warnings
        warnings.warn("build_aloft_supply_target(absolute=True) requires aloft_moisture_mm (AD core only); "
                      "falling back to the legacy normalize-to-shape law.", RuntimeWarning)
        return _legacy_supply(precip_mm_day)
    F = np.clip(np.asarray(aloft_moisture_mm, float), 0.0, None)
    edge = theta_deg >= chimney_theta
    if edge.any():                                  # clamp the equatorward chimney/inner-sea region
        F_edge = float(F[edge][np.argmin(theta_deg[edge])])
        F = np.where(edge, F, np.minimum(F, F_edge))
    return F


# The bulk-flux wind the LAND diurnal surface solve runs at [m/s]. It is a SCREEN-LEVEL (10 m) wind,
# NOT one of the model's own wind fields, and that distinction is the content of ROADMAP GEO-8:
#   * `surface.turbulent_fluxes` closes SH/LE with Ch = Ce = 1.2e-3, the standard transfer coefficients
#     for a 10 m REFERENCE wind (coefficient and height are a matched pair);
#   * every wind the model carries -- the AD core's `u_surf`, Stage C's balanced |V| (whose background
#     IS that u_surf, solved over an H_bl = 1000 m layer) -- is a lowest-level / boundary-layer wind,
#     and a DAY-MEAN. On the canonical world those median ~10.4-10.5 m/s.
# So the two are not rival values of one quantity, and the ~10.5 must be REDUCED before it can enter a
# 10-m bulk formula. `surface_wind.bulk_flux_wind` is that reduction (log surface layer -> nocturnal
# decoupling over land -> the day/night time average a CONSTANT-wind solve forces).
# MEASURED on the canonical seed-90 archive (`verify_diurnal_wind.py`): the archived field reduces to a
# daily-effective screen wind of 4.6-6.6 m/s over the thaw fringe (midpoint ~5.6), so 5.0 is a defensible
# effective value rather than a stray literal -- toward the low end, though NOT pinned: that 4.6-6.6 is
# one averaging convention's soft-parameter span, and the true uncertainty is wider and two-sided (the
# day/night mean is itself a choice, target-dependent; stable nocturnal Ch collapse pushes the night wind
# lower -- see surface_wind.bulk_flux_wind). A re-take at the honest per-cell wind is an open residual
# (GEO-8; it needs the canonical checkpoint, so it rides with GEO-6's re-roll). Sized on the archive: the
# skin diurnal range moves -0.6 K (mixed-layer match) to -2.7 K (z1 match) of ~36 K, i.e. T_high/T_low by
# ~0.1-0.7 K.
# NOTE the height argument alone does NOT justify 5.0. The canonical lowest level sits at z1 = 20 m (the
# BL-refined stretch, dz_min = 20 m -- set CONFIG-side by run_driver -> build_donor_core_dev, NOT by
# canonical_env.sh), and the core's own bulk drag uses the SAME 10-m C_D = 1.2e-3 there, so the
# log-layer step is worth only 8-22 %. The reduction to ~5 is carried by the parameterized nocturnal
# decoupling -- the named wall on this whole number.
# Driving the solve with the RAW ~10.5 m/s level wind instead roughly doubles SH/LE and is what makes
# the archived skin extremes unreproducible (0 of 8 fringe representatives close; `verify_day_shape.py
# --wind archive`) -- it would shrink the skin range by ~12 K. Change this only with that measurement
# re-run: it moves every shipped T_high/T_low.
WIND_BULK_SCREEN_MS = 5.0


def land_surface_diurnal(stage_a: StageA, theta_deg, wetness=0.40, C_surf=C.C_MOIST_LAND,
                         albedo=C.ALBEDO_LAND, wind=WIND_BULK_SCREEN_MS, day_cycle=None, atm=None):
    """Diurnally-resolved LAND surface temperature vs latitude, from an atmospheric column (downwelling
    LW + near-surface air-T) run through flatworld.surface.solve_periodic_surface -- the SAME diurnal
    machinery Stage A applies to its land diagnostic. Returns (Tmean_C, Tmin_C, Tmax_C) arrays [deg C]: the
    day-mean PLUS the diurnal extremes (Issue #21). The mean carries the heat-capacity / sigma T^4 (Jensen)
    correction the ocean backbone does not need; the min/max carry the day/night swing. ``wetness`` /
    ``C_surf`` select the continentality endpoint -- damp coast (C_MOIST_LAND, wetter) vs arid interior
    (C_ARID_LAND, drier, lower thermal inertia -> a far larger swing).

    The driving atmosphere is ``stage_a`` by default (its downwelling LW + lowest-layer air-T). Pass
    ``atm=(D_surf_of_theta, Tair_of_theta)`` -- two ``theta_deg -> value`` callables (LW [W/m^2], air-T
    [K]) -- to drive the land balance from a DIFFERENT backbone. The AD-core cascade passes the fused
    core's atmosphere here so the land surface rides the resolved-circulation gradient, NOT the
    transport-poor Stage A column (whose greenhouse-collapse cliff otherwise crushes the land band to ~1
    deg of theta; the ocean SST already comes from the core). ``stage_a`` may be None when ``atm`` is given.

    ``wind`` is a SCREEN-LEVEL (10 m) bulk-flux wind, defaulting to ``WIND_BULK_SCREEN_MS`` -- read that
    constant's note before passing one of the model's OWN wind fields here: those are boundary-layer
    winds and must go through ``surface_wind.bulk_flux_wind`` first (ROADMAP GEO-8)."""
    if atm is not None:
        D_surf, Tair = atm
    else:
        d = stage_a.diagnostics()
        D_surf = _interp(d["theta_deg"], d["D_surf"])
        Tair = _interp(d["theta_deg"], stage_a.T_atm[:, -1])         # lowest-layer air temperature [K]
    mean, tmin, tmax = [], [], []
    for th in theta_deg:
        x = float(g.x_of_theta(np.radians(th)))
        # day_cycle: optional (x) -> (t, S_inst) override -- the LANTERN painted day (flatworld.lantern,
        # locked premise 2026-06-11); None = the natural (shaded-geometry) passage, byte-identical legacy
        t, S_inst = g.diurnal_flux_cycle(x, n_steps=144, n_suns=6) if day_cycle is None else day_cycle(x)
        SW_abs = 0.9 * (1 - albedo) * S_inst                        # land albedo, clear-ish sky
        r = surf.solve_periodic_surface(t, SW_abs, float(D_surf(th)), float(Tair(th)), 3e-3,
                                        C.P_SURF, wind=wind, C_surf=C_surf, wetness=wetness,
                                        max_days=60, tol=2e-3)
        mean.append(r["Ts_mean"] - C.T_FREEZE)
        tmin.append(r["Ts_min"] - C.T_FREEZE)
        tmax.append(r["Ts_max"] - C.T_FREEZE)
    return np.array(mean), np.array(tmin), np.array(tmax)


def land_temperature_backbone(stage_a: StageA, theta_deg):
    """Day-mean LAND surface temperature [C] vs latitude (the damp-coast continentality endpoint) -- a thin
    wrapper over land_surface_diurnal kept for callers that need only the mean."""
    return land_surface_diurnal(stage_a, theta_deg)[0]


def _ribbon_mean_sst(dA):
    th = dA["theta_deg"]
    return float(np.mean(dA["T_surf_C"][(th >= 45) & (th <= 51)]))


def couple_stage_a_b(profile="test", verbose=True, max_outer=None, outer_tol=None,
                     a_overrides=None, b_overrides=None, a_warmstart=None, relax=1.0):
    """The Issue-A A<->B transport-injection loop -- **STRUCTURALLY INCOMPATIBLE; NOT the deliverable.**

    Kept in-tree as the documented record of the attempt (OPEN_ISSUES.md Issue A, resolved
    won't-fix-within-scope). It injects B's diagnosed flux divergence -dF_B/dx as A's transport, but B's
    closed-cell transport vanishes at its poleward wall (F_B(theta_max)=0), so it carries ZERO energy
    across A's cold edge and cannot feed A's radiating sponge -- A's energy balance breaks irreducibly
    (closure rel ~1.0 vs the diffusive ~0.34 at production res; the sponge export collapses ~3.1e8->7e6
    W/m) and the ribbon freezes. The unrelaxed loop also diverges
    (Picard oscillation); ``relax``<1 tames the oscillation but NOT the energy-closure break (it is
    structural, not an iteration artifact). Closing the A/B transport-latitude gap needs a true dynamical
    dimensionalization of B (a deeper rebuild than scoped), not this injection. The deliverable cascade
    uses diffusive A (run_cascade couple=False).

    Mechanism (for the record): Stage A's heat transport is replaced by Stage B's diagnosed
    flux-divergence -dF_B/dx, iterated to a fixed point because B is driven by A's SST and vice versa:

        A (diffusive baseline -> the fiducial transport magnitude + the first SST)
        loop: A's SST/wetness -> B's circulation+moisture -> dimensional F_B -> inject -dF_B/dx into A
              -> re-solve A (warm-started) -> until the ribbon-mean SST stops moving.

    B's transport SHAPE (Rossby=inf) is fixed; its MAGNITUDE is anchored to A's diffusive-baseline peak
    times cfg.transport_anchor (the swept free parameter). A's energy closure is preserved every iterate
    (the injected divergence integrates to zero). Returns the converged stages, diagnostics, the fiducial
    anchor, and the per-iteration drift / F_B residual histories (the fixed-point 'coupling closed' check).
    """
    max_outer = (3 if profile != "production" else 5) if max_outer is None else max_outer
    outer_tol = (0.5 if profile != "production" else 0.2) if outer_tol is None else outer_tol
    # the outer loop manages its own warm-starting (B via load_state, A by re-running the same instance),
    # so DISABLE the per-stage checkpoint_path -- otherwise each B.run()/A.run() would resume from a stale
    # checkpoint and clobber the warm-start. Crash-resumability is traded for a correct fixed point.
    a_kw = {"checkpoint_path": None, **dict(a_overrides or {})}
    b_kw = {"checkpoint_path": None, **dict(b_overrides or {})}

    # ---- Stage A: diffusive baseline (the current model) -> SST + the fiducial transport magnitude ----
    A = StageA(cfgmod.stage_a(profile, **a_kw))
    if a_warmstart is not None:
        try:
            A.load_state(a_warmstart)                             # warm-start from a converged diffusive A
            if verbose: print(f"[A<->B] warm-started diffusive baseline from a saved state")
        except Exception as e:
            if verbose: print(f"[A<->B] warm-start failed ({e}); cold start")
    if verbose: print("=== Stage A (diffusive baseline) ===")
    A.run(verbose=verbose)
    dA_diff = A.diagnostics()
    Fdiff = A.recompute_diffusive_flux()
    fiducial = float(np.max(Fdiff[Fdiff > 0])) if np.any(Fdiff > 0) else float(np.max(np.abs(Fdiff)))
    A.cfg.transport_mode = "injected"                              # switch A onto B's transport

    # ---- A<->B fixed point ----
    drift_hist, fb_resid_hist = [], []
    B = dB = None
    B_state = None
    rib_prev = _ribbon_mean_sst(dA_diff)
    for outer in range(max_outer):
        dA = A.diagnostics()
        sst_of_theta = _interp(dA["theta_deg"], dA["T_surf_C"])
        wetness_of_theta = _interp(dA["theta_deg"], A._wetness())
        B = StageBAtmosphere(cfgmod.stage_b(profile, Tsurf_C_of_theta=sst_of_theta,
                                            wetness_of_theta=wetness_of_theta, **b_kw))
        if B_state is not None and B_state["b"].shape == B.b.shape:
            B.load_state(B_state)                                  # warm-start the dynamics across iters
        if verbose: print(f"=== Stage B (outer {outer+1}/{max_outer}, driven by A's SST) ===")
        B.run(verbose=verbose)
        dB = B.diagnostics(); B_state = B.state()
        _, F_B = dimensional_poleward_flux(B, anchor=fiducial * A.cfg.transport_anchor)
        x_face = np.arange(A.cfg.nx + 1) * A.dx
        theta_face = np.degrees(g.theta_of_x(x_face))
        F_on_A = np.interp(theta_face, dB["theta_deg"], F_B)       # onto A's interfaces (clamps to 0 poleward)
        F_prev = None if A._injected_F is None else A._injected_F.copy()
        if F_prev is not None and relax < 1.0:                     # under-relax the Picard iteration: the
            F_on_A = (1.0 - relax) * F_prev + relax * F_on_A       # unrelaxed map oscillates/diverges
        A.set_injected_flux(F_on_A)
        if F_prev is not None:
            denom = max(np.max(np.abs(A._injected_F)), 1e-30)
            fb_resid_hist.append(float(np.max(np.abs(A._injected_F - F_prev)) / denom))
        if verbose: print(f"=== Stage A (outer {outer+1}/{max_outer}, injected -dF_B/dx) ===")
        A.run(verbose=verbose)
        rib_new = _ribbon_mean_sst(A.diagnostics())
        drift = abs(rib_new - rib_prev); drift_hist.append(drift); rib_prev = rib_new
        if verbose:
            fbr = fb_resid_hist[-1] if fb_resid_hist else float("nan")
            print(f"[A<->B] outer {outer+1}: ribbon SST drift {drift:.3f} C, F_B resid {fbr:.3e}")
        if drift < outer_tol and (not fb_resid_hist or fb_resid_hist[-1] < 0.05):
            if verbose: print(f"[A<->B] converged after {outer+1} outer iterations")
            break

    dA = A.diagnostics()
    sst_of_theta = _interp(dA["theta_deg"], dA["T_surf_C"])
    return dict(A=A, B=B, dA=dA, dB=dB, sst_of_theta=sst_of_theta, fiducial=fiducial,
                dA_diffusive=dA_diff, drift_hist=drift_hist, fb_resid_hist=fb_resid_hist)


def lem_cascade_topo(seed=0, nx_fine=600, ne_fine=800, nx_c=300, ne_c=400, theta_lo=28.0, theta_hi=62.0,
                     Le_km=12000.0, engine="richdem", mask_reconcile=True, return_fine=False,
                     allow_unconverged=False, verbose=False, **erode_kw):
    """Build the climate-grid topology from the **fine LEM** (geology roadmap G6, the end-to-end integration):
    ``erosion.lem_contract`` (skeleton -> erode + rivers + seafloor + mask reconciliation, at the fine
    worldbuilding resolution) -> ``topo_contract.downsample`` (orography-preserving coarse-grain to the
    cascade grid, with the post-downsample land-fraction re-pin) -> the coarse ``TopographyContract`` Stage C
    consumes (via ``to_topogrid``).

    Returns the COARSE contract (cascade input); with ``return_fine=True`` returns ``(fine, coarse)`` so the
    caller keeps the fine worldbuilding map (for ``viz_topo`` / the gallery). The fine LEM build is the
    expensive step; the downsample + re-pin are cheap. ``erode_kw`` (maturity_steps, k_stream, ...) pass to
    the LEM. This is the one place geology feeds the climate: everything else binds to the contract."""
    from . import erosion                                         # lazy: keep landlab off the cascade import path
    from . import topo_contract as tcm
    fine = erosion.lem_contract(seed=seed, nx=nx_fine, ne=ne_fine, theta_lo=theta_lo, theta_hi=theta_hi,
                                Le_km=Le_km, engine=engine, mask_reconcile=mask_reconcile,
                                allow_unconverged=allow_unconverged, verbose=verbose, **erode_kw)
    # ``nx_c``/``ne_c`` are the DELIVERED climate grid (the same sense as cfgmod.topo_production's nx/ne). The
    # fine map is PADDED in e (Stage C's periodic seam isolation), so coarse-grain to the full padded coarse
    # count that yields ne_c delivered columns; downsample carries the delivered/pad split proportionally, and
    # the cascade crops to it after Stage C (_deliver_crop). x is not padded, so nx_c maps directly.
    ne_fine_full = fine.mask.shape[1]
    ne_c_full = max(ne_c + 1, int(round(ne_c * ne_fine_full / max(int(ne_fine), 1))))
    coarse = tcm.downsample(fine, nx_c, ne_c_full, target_land_fraction=float(fine.mask.mean()))
    coarse.meta.update(lem_fine_shape=tuple(fine.mask.shape), lem_engine=engine,
                       lem_mask_reconcile=bool(mask_reconcile))
    return (fine, coarse) if return_fine else coarse


def _resolve_topo(topo, profile, topo_seed, topo_kwargs, topo_kind="procedural"):
    """Pick Stage C's topology -- the one place the cascade decides between generating a topology and
    consuming an externally supplied one.

      * topo is None (the DEFAULT, the historical behavior): procedurally GENERATE a random continent
        from topo_seed/topo_kwargs (production res from cfgmod.topo_production, else the test grid).
        topo_kind selects the generator: "procedural" (the default fBm continent), "tectonic" (the
        geology+climate-consistent plate/belt/island generator, flatworld.tectonics via
        Topography.tectonic_continent), or "lem" (geology roadmap G6: the fine landscape-evolution model
        built by lem_cascade_topo and downsampled to the cascade grid -- erosion, rivers, seafloor, rifts).
      * topo is supplied: INPUT it as-is -- a TopographyContract (the source-agnostic geology interface,
        projected to a TopoGrid via .to_topogrid()), a TopoGrid, a path to a saved .npz (Topography.load),
        or a Mapping of x/e/land/height arrays (Topography.from_arrays) -- so a topology from another model
        (e.g. a geophysics run or the LEM) drives the climate directly. It is validated against Stage C's
        uniform-grid assumptions. topo_kwargs is procedural-only and cannot be combined with an explicit
        topology (which already fixes the grid)."""
    if topo is None:
        if topo_kind == "lem":                                   # geology roadmap G6: fine LEM -> downsample
            if profile == "production":
                tk = {**cfgmod.topo_production_lem(topo_seed), **(topo_kwargs or {})}
            else:                                                # a small, fast fine grid for the test profile
                # the test smoke grid is deliberately coarse (sub-converged) -> pass the G7.c override so the
                # resolution guard does not reject this intentional fast-test combo (production stays guarded).
                tk = {**dict(seed=topo_seed, nx_fine=80, ne_fine=110, nx_c=40, ne_c=55,
                             theta_lo=28.0, theta_hi=62.0, Le_km=12000.0, engine="richdem", mask_reconcile=True,
                             allow_unconverged=True),
                      **(topo_kwargs or {})}
            return lem_cascade_topo(**tk).to_topogrid()
        if topo_kind == "tectonic":
            if profile == "production":
                tk = {**cfgmod.topo_production_tectonic(topo_seed), **(topo_kwargs or {})}
            else:
                tk = {**dict(seed=topo_seed, nx=90, ne=120, theta_lo=28.0, theta_hi=62.0, Le_km=12000.0),
                      **(topo_kwargs or {})}
            return Topography.tectonic_continent(**tk)
        if profile == "production":
            tk = {**cfgmod.topo_production(topo_seed), **(topo_kwargs or {})}
        else:
            tk = {**dict(seed=topo_seed, nx=90, ne=90, Le_km=6000.0, width_km=3500.0, center_theta=48.0),
                  **(topo_kwargs or {})}
        return Topography.procedural_continent(**tk)
    if topo_kwargs:
        raise ValueError("topo_kwargs applies only to the procedural default; do not pass it alongside "
                         "an explicit topo (the supplied topology already fixes the grid)")
    if isinstance(topo, TopographyContract):                 # the source-agnostic geology contract (G1)
        return topo.to_topogrid()
    if isinstance(topo, TopoGrid):
        return topo.validate()
    if isinstance(topo, Mapping):
        return Topography.from_arrays(**topo)
    if isinstance(topo, (str, bytes)) or hasattr(topo, "__fspath__"):
        return Topography.load(topo)
    raise TypeError(f"topo must be a TopoGrid, a path to a saved topology (.npz), or a Mapping of "
                    f"x/e/land/height arrays; got {type(topo).__name__}")


def _deliver_crop(topo, dC):
    """Crop Stage C's 2-D maps to the topology's DELIVERED central region (the central Le_km of a PADDED
    tectonic topology), so the returned T/P maps and the cropped topo are the deliverable -- a segment of a
    non-periodic infinite plane. Stage C ran on the FULL padded grid (its periodic-e seam in the pad), so the
    GLOBAL water closure (area-integrated over the closed padded domain) is kept as-is; the land/sea precip
    means are recomputed on the crop. For an unpadded topo (no central_e_slice, e.g. the procedural default)
    this is a no-op. Returns (topo_out, dC_out)."""
    sl = topo.meta.get("central_e_slice")
    if not sl or (sl[0] == 0 and sl[1] >= topo.e.size):
        return topo, dC
    from . import tectonics
    lo, hi = sl
    topo_out = tectonics.central_crop(topo)
    out = dict(dC)
    # crop EVERY 2-D (x,e) map so the delivered dC is shape-consistent (the classifiers consume the day-mean
    # AND the diurnal/air extremes together -- a partial crop mismatches their shapes). Includes the Issue #21
    # surface extremes and the screen-level air extremes alongside the day-mean/precip/wind/moisture maps.
    for k in ("T_map_C", "P_mm_day", "u", "v", "W_a", "W_s", "land", "height",
              "T_min_C", "T_max_C", "dT_diurnal_C", "T_air_min_C", "T_air_max_C", "dT_air_C"):
        if k in out and getattr(out[k], "ndim", 0) == 2:
            out[k] = out[k][:, lo:hi]
    P, land = out.get("P_mm_day"), out.get("land")               # crop-local precip means; KEEP global closure
    if P is not None and land is not None:
        out["P_land_mean"] = float(P[land].mean()) if land.any() else 0.0
        out["P_sea_mean"] = float(P[~land].mean()) if (~land).any() else 0.0
    return topo_out, out


def _ascent_weight(theta_deg, w_mid, theta_norm_min=20.0):
    """Pure [0,1] cell-ascent weight from a mid-height vertical-velocity profile (split out for testing).
    1 where the cell RISES (positive w, normalized to its max poleward of theta_norm_min so the huge
    equatorial updraft -- outside the Stage-C domain -- does not set the scale), 0 where it SUBSIDES.
    Lightly smoothed. Uses B's ascent SHAPE (authoritative), not its magnitude (B is nondimensional)."""
    th = np.asarray(theta_deg, float); pos = np.clip(np.asarray(w_mid, float), 0.0, None)
    if pos.size >= 3:
        pos = np.convolve(pos, np.ones(3) / 3.0, mode="same")
    ref = pos[th >= theta_norm_min]
    m = ref.max() if (ref.size and ref.max() > 0) else (pos.max() if pos.max() > 0 else 1.0)
    return np.clip(pos / m, 0.0, 1.0)


def cell_ascent_gate(dB, theta_norm_min=20.0):
    """A callable theta_deg -> [0,1] weight on LOCAL convective lofting, from Stage B's overturning:
    1 in the cell's ascending branches, 0 where it subsides. This is the equatorward-vs-poleward
    partition the single cell sets (PR #7): the cell rises at the equatorial updraft (theta~0, outside
    the Stage-C domain) and in a SECONDARY ascent at the rain band (theta~37-44), and SUBSIDES over the
    inner sea (theta~28-37) and the habitable ribbon (theta>~45). So local surface moisture is lofted to
    the poleward-aloft layer only where the cell ascends; over the subsiding wet ocean it is NOT lofted
    -- it rains locally / rides equatorward to loft at the equatorial updraft (the F(theta) pathway that
    sets the circulation band at theta~36). Used to GATE the plume's loft efficiency in run_cascade."""
    w = np.asarray(dB["w"], float)
    w_mid = w[w.shape[0] // 2]
    a = _ascent_weight(dB["theta_deg"], w_mid, theta_norm_min)
    return _interp(np.asarray(dB["theta_deg"], float), a)


def _subsidence_weight(theta_deg, w_mid, rel_floor=0.01):
    """Pure [0,1] convection-allowed weight from a mid-height vertical-velocity profile (split out for
    testing): 1 where the cell GENUINELY ASCENDS, 0 where it subsides OR is neutral/dead. Lightly
    smoothed in theta (a 3-point mean) so the gate is not grid-noisy right at the sign change.

    "Genuinely" = w >= rel_floor * max|w| (2026-06-11, the seed-137 map bug): the original SIGN-ONLY
    rule (w >= 0) was built for Stage B's clean nondimensional cell, but the CONVERGED AD-core's
    dead/standing zones wiggle at ~0.5-0.7% of the cell scale around zero, and the sign rule promoted
    +1e-4 m/s noise to "ascent" -- opening the convective drain over the warm-edge sea that the zonal
    world keeps RAINLESS (40 mm/day of spurious map rain). Physically: organized deep convection needs
    genuine large-scale ascent; neutral/dead air does not qualify. The floor is RELATIVE to the
    profile's own scale, so Stage B's genuine ascent zones (O(1) relative) are untouched and only the
    marginal sign-crossing cells shift. Magnitude is otherwise NOT weighted (unlike _ascent_weight) --
    above the floor this is still a pure allowed/suppressed gate."""
    th = np.asarray(theta_deg, float); w = np.asarray(w_mid, float)
    if w.size >= 3:
        w = np.convolve(w, np.ones(3) / 3.0, mode="same")
    floor = rel_floor * float(np.abs(w).max()) if w.size else 0.0
    return (w >= max(floor, 1e-300)).astype(float)


def subsidence_gate(dB):
    """A callable theta_deg -> [0,1] gate on the convective DRAIN, from Stage B's overturning (Refinement
    1): 1 where the single cell ASCENDS (deep convection allowed), 0 where it SUBSIDES (deep convection
    capped from above -- why Earth's subsidence zones are dry despite warm oceans). The cell rises at the
    equatorial updraft (theta~0, outside the Stage-C domain) and in a SECONDARY band ascent (theta~37-44),
    and SUBSIDES over the inner sea (theta~28-37) and the habitable ribbon (theta>~45). So over the warm
    subsiding ribbon sea the convective drain is suppressed -- the evaporated moisture advects equatorward
    into the ascending branch instead of raining spuriously in place. Wired into StageCConfig.
    subsidence_of_theta. Sibling of cell_ascent_gate (which gates the loft FRACTION, Brief 2); this gates
    WHETHER the warm subsiding column drains at all -- the drain-gating step Brief 2 deferred."""
    w = np.asarray(dB["w"], float)
    w_mid = w[w.shape[0] // 2]
    s = _subsidence_weight(dB["theta_deg"], w_mid)
    return _interp(np.asarray(dB["theta_deg"], float), s)


def _run_ad_core(profile, ad_overrides, rce_steps, dyn_steps, verbose=True):
    """Build and run the fused AD core (``stage_ad_core.StageADCore``) for the ``core='ad'`` cascade.

    At ``production`` the core's full convergent ``run()`` is used (checkpointed, guarded -- the expensive
    M3-staged run). At ``test`` (and any non-production profile) a BOUNDED budget is used: a fast no-flow
    RCE spin-up (dynamics off, large thermodynamic dt) then a coupled dynamical phase (dynamics + resolved
    heat + moisture). The bounded run exercises the Stage-C interface contract end-to-end but is **NOT
    converged** -- the coupled core is CFL-stiff (dt ~ seconds), so a converged run is GPU/M3-budget work
    (see ``M2B6_RESULTS.md``). ``ad_overrides`` patches the config."""
    from .stage_ad_core import StageADCore, StageADConfig
    ov = dict(ad_overrides or {})
    if profile == "production":
        # scope the steady-state drift off the slow frigid cold tail by default (the turbulent cold edge
        # converges slowest and would otherwise stall the run against the guard) -- mirrors Stage A's
        # conv_theta_max; overridable. The interesting region (rain band ~36, ribbon ~49-52) is well inside.
        cfg = StageADConfig(checkpoint_path=ov.pop("checkpoint_path", "stageAD_prod.npz"),
                            conv_theta_max=ov.pop("conv_theta_max", 55.0), **ov)
        ad = StageADCore(cfg)
        if verbose: print("=== Stage AD core (production; full convergent run -- M3-staged, expensive) ===")
        ad.run(verbose=verbose)
        return ad
    cfg = StageADConfig(nx=ov.pop("nx", 24), nz=ov.pop("nz", 16),
                        theta_max_deg=ov.pop("theta_max_deg", 70.0), **ov)
    ad = StageADCore(cfg)
    if verbose:
        print(f"=== Stage AD core (test; bounded {rce_steps} RCE + {dyn_steps} dyn steps -- NOT converged) ===")
    return _bounded_ad_run(ad, rce_steps, dyn_steps)


def _bounded_ad_run(ad, rce_steps, dyn_steps):
    """The test-profile budget: RCE spin-up (dynamics off, big thermo dt) then coupled steps."""
    ad.run_dynamics = False                                       # fast no-flow RCE spin-up (large dt)
    for _ in range(rce_steps):
        ad.step()
    ad.run_dynamics = True                                        # coupled dynamical phase (heat + moisture)
    for _ in range(dyn_steps):
        ad.step()
    return ad


def _load_ad_core(ckpt_spec, warm_days=0.25, verbose=True):
    """The CONVERGED-CHECKPOINT backbone (the geology-arc adapter, 2026-06-11): build the AD core for
    a converged fleet run and LOAD its state, instead of spinning a core up in-cascade -- so Stage C
    (and everything downstream: band F, gates, classify) runs off the canonical world (e.g.
    ``sh_b23L8``) at zero spin-up cost. Returns ``(core, sim_day)``.

    ``ckpt_spec``: a fleet TAG (looked up in ``_allon_fleet.FLEET``) or a spec-like dict with keys
    ``ckpt`` (path), ``nx``, ``nz``, ``theta_max``, ``env`` (dict), ``run_kwargs`` (with
    ``overrides``) -- the dict form is for tests / off-registry checkpoints.

    ENV CONTRACT: env-bearing worlds (``COSMO_L_SUN`` / ``COSMO_SHADE_FILE`` / ``COSMO_I0_SCALE`` ...)
    bake into ``flatworld.constants``/``geometry`` AT IMPORT, so the spec's env must already be
    applied to THIS process. Verified loudly here -- a mismatch raises instead of silently building
    the wrong world. Use ``python -m flatworld.cascade ckpt <tag>`` to re-exec with the env applied.

    WARM-UP: per-step rate fields (``_precip``) are not checkpoint state, so the loaded core is
    stepped ~``warm_days`` sim-days (negligible vs the converged hold) and the precip handed to the
    Stage-C gates is the warm-up TIME-MEAN -- averaging single-step convective flicker out of the
    band shape the gates and F(theta) consume."""
    import os
    from . import _allon_fleet as _af
    from . import _warm_restart as _wr
    from . import convergence as conv
    from . import _ckpt_fingerprint as _ckfp

    if isinstance(ckpt_spec, str):
        spec = _wr._spec_of(ckpt_spec)
        ckpt = _af._paths(spec)["ckpt"]
    else:
        spec = dict(ckpt_spec)
        ckpt = spec["ckpt"]
    for k, v in (spec.get("env") or {}).items():
        if os.environ.get(k) != v:
            raise RuntimeError(
                f"checkpoint backbone {spec.get('tag', ckpt)}: spec env {k}={v!r} is not applied to this "
                f"process (got {os.environ.get(k)!r}). Env bakes in at import -- launch via "
                f"`python -m flatworld.cascade ckpt <tag>` (re-execs with the env) or export it first.")
    st, meta = conv.load_checkpoint(ckpt)
    if st is None:
        raise FileNotFoundError(f"checkpoint backbone: no checkpoint at {ckpt}")
    core = _wr._tag_core(spec)
    core.load_state(st)
    # Embedded-fingerprint guard (additive to the declared-env raise above): the checkpoint now carries
    # the world/grid it was produced in, so a wrong-world load fails LOUDLY even when the spec declared
    # no env. STRICT here -- silently building the wrong world is the failure mode this adapter guards.
    # (An old checkpoint with no fingerprint loads through unaffected -- validate_resume never raises on
    # an absent fingerprint, so existing converged checkpoints stay usable.)
    _ckfp.validate_resume(meta, core.cfg, mode="strict", context=spec.get("tag", ckpt))
    sim_day = float((meta or {}).get("sim_time", 0.0)) / 86400.0
    if verbose:
        print(f"=== Stage AD core (LOADED converged checkpoint {spec.get('tag', ckpt)}, "
              f"day {sim_day:.0f}; diagnostic warm-up {warm_days} d) ===")
    target, t, acc, n = warm_days * 86400.0, 0.0, None, 0
    while t < target:
        core.step()
        dt = core._last_dt_dyn if core.run_dynamics and core._last_dt_dyn else core._last_dt_th
        t += float(dt or core.cfg.dt)
        p = getattr(core, "_precip", None)
        if p is not None:
            p = np.asarray(p, float)
            f = getattr(core, "_surf_flash", None)
            if f is not None:
                # SURFACE FLASH-BOIL (PR #298): the gates/F consume the surface LIQUID supply, so
                # average the POST-flash flux (rain flashed off a super-boiling skin re-entered the
                # BL as vapour and never wetted anything). Flag OFF: _surf_flash stays 0 and this is
                # bit-identical to averaging _precip alone.
                p = np.maximum(p - np.asarray(f, float), 0.0)
            acc = p.copy() if acc is None else acc + p
            n += 1
    if n:
        core._precip = acc / n                                    # gates/F read the time-mean band shape
        f = getattr(core, "_surf_flash", None)
        if f is not None:
            # The time-mean above is already POST-flash, but any later _wetness() call (flag ON)
            # subtracts the core's INSTANTANEOUS _surf_flash from whatever sits in _precip -- a
            # category mismatch (time-mean minus last-substep flash) that would double-subtract.
            # Zero the flash buffer so _wetness sees exactly the post-flash time-mean supply.
            # Flag OFF: the buffer is already 0 (no-op, byte-identical).
            f[...] = 0.0
    return core, sim_day


def run_cascade(profile="test", topo_seed=0, topo_kwargs=None, verbose=True, couple=False,
                a_warmstart=None, topo=None, floft_anchored=True, topo_kind="procedural",
                core="ad", ad_overrides=None, ad_rce_steps=1500, ad_dyn_steps=8000,
                ad_checkpoint=None, ad_core=None, ad_warm_days=0.25, stage_c_overrides=None,
                lantern_day=False,
                bl_momentum_wind=False, bl_momentum_cfg=None,
                bl_diurnal_wind=False, bl_diurnal_cfg=None,
                stage_c_diurnal_precip=False, stage_c_diurnal_cfg=None,
                aloft_supply_absolute=None):
    """Run A -> B -> C and return a result dict with the wired backbone and the final maps + classes.

    THE DELIVERABLE is ``couple=False`` (default): diffusive Stage A (its own energy-closing transport)
    -> Stage B with the absolute-footing moisture (Issue-A Part 1). ``couple=True`` runs the
    Issue-A transport injection (couple_stage_a_b) -- which is **STRUCTURALLY INCOMPATIBLE** and is NOT
    the deliverable: B's closed-cell transport vanishes at its poleward wall, so it cannot export the
    equatorial heat surplus across A's radiating sponge, and injecting it as A's sole closure breaks A's
    energy balance (closure rel ~1.0 vs the diffusive ~0.34 at production res) and freezes the ribbon. Kept in-tree only as the
    documented record of the attempt -- see OPEN_ISSUES.md Issue A (resolved: won't-fix-within-scope).

    ``core`` selects the zonal-mean backbone (M2b.6 integration). **DEFAULT ``"ad"``** (2026-06-09): the
    fused all-on AD-core IS the maximally-accurate model, so it is the default backbone. The cheap
    ``cascade test`` regression gate is pinned to ``"diffusive"`` in ``__main__`` (and the topo/diagnostic
    callers pass ``core="diffusive"`` explicitly) so the diffusive path stays exercised and green.
      * ``"diffusive"`` -- the legacy A -> B cascade above (the cheap gate / sweep engine).
      * ``"ad"`` (DEFAULT) -- the FUSED dimensional core (``stage_ad_core.StageADCore``) supplies SST + circulation +
        band through the SAME Stage-C interface contract (``StageADCore.stage_c_inputs``); Stage A still
        runs as the cheap column / land-T engine + the plume f_loft anchor (the roadmap's "Stage A survives
        as the 1-D engine; the core runs once per scenario"). At the ``test`` profile the core is run on a
        BOUNDED budget (``ad_rce_steps`` no-flow RCE spin-up, then ``ad_dyn_steps`` coupled dynamical
        steps) -- it exercises the interface end-to-end but is NOT converged (the converged run is the
        expensive M3-staged production, see M2B6_RESULTS.md); at ``production`` it calls the core's full
        ``run()``. ``ad_overrides`` patches the core config. The downstream (band F, land-T, f_loft,
        subsidence gate, Stage C, classify) is byte-identical across both cores -- which is exactly how
        M2b.6 verifies the Stage-C decoupling is at the interface contract, not at frozen tunings.

    Topology: by default Stage C runs over a procedurally GENERATED random continent (topo_seed /
    topo_kwargs). Pass ``topo`` to INPUT an externally supplied topology instead -- a TopoGrid, a path to
    a saved .npz, or a dict of x/e/land/height arrays -- e.g. to compute the climate a separate
    geophysics model's topology produces. See _resolve_topo for the accepted forms.

    ``floft_anchored`` (default True): build Stage C's convective<->stratiform split as f_loft_eff(theta) =
    cell_ascent_gate(B) * plume_efficiency(A) (OPEN_ISSUES Brief 2) -- B's overturning ascent gates WHERE
    local lofting happens (only the cell's ascending band branch; ~0 over the subsiding wet ocean), the
    frozen+locked entraining plume (flatworld.plume_floft) sets the loft-vs-rain efficiency there. Replaces
    the free scalar f_loft=0.5. Set False to revert to the scalar.

    ``lantern_day`` (core="ad" only): drive Stage C's resolved land diurnal cycle with the LOCKED
    lantern painted day (``flatworld.lantern``: 16-h half-cosine in the band, scaled natural day
    outside, day-means anchored to the core's own S(theta) so the climate is untouched) instead of
    the natural sun-passage cycle. The lantern world's diurnal ranges differ from the natural day's
    even though every day-MEAN is identical. Default False (byte-identical legacy).

    ``ad_checkpoint`` (core="ad" only): a fleet TAG (e.g. ``"sh_b23L8"``) or spec-like dict -- LOAD a
    CONVERGED fleet checkpoint as the backbone instead of spinning a core up in-cascade (the geology-arc
    adapter: Stage C / classify run off the canonical world at zero spin-up cost). The spec's env
    (COSMO_L_SUN / COSMO_SHADE_FILE ...) must already be applied to the process -- ``_load_ad_core``
    verifies and raises otherwise; ``python -m flatworld.cascade ckpt <tag>`` re-execs with the env
    applied. ``ad_warm_days`` is the short diagnostic warm-up that repopulates per-step rate fields
    (precip -> the gates) as a time-mean.

    ``ad_core`` (core="ad" only): a PRE-BUILT, already-state-loaded ``StageADCore`` to use AS the
    backbone, bypassing both ``_load_ad_core`` and ``_run_ad_core``. Use it to build/load a converged
    core ONCE and run Stage C over many topologies with it (e.g. the nsub30 gallery sweep, via
    ``nsub30_backbone.build_loaded_nsub30_core``) -- ``stage_c_inputs()`` is read-only, so one core can
    drive N calls. Takes precedence over ``ad_checkpoint``."""
    # fail fast on the #113 flag combo before any A->B->C work: the diurnal breeze is premise-dependent.
    if bl_diurnal_wind and not lantern_day:
        raise ValueError("bl_diurnal_wind=True requires lantern_day=True -- the diurnal sea/land-breeze "
                         "magnitude is tied to the painted-day premise; the natural-day shape would mislead.")
    # the #131 diurnal-precip coupling consumes the breeze solve's per-phase winds, so it requires it.
    if stage_c_diurnal_precip and not bl_diurnal_wind:
        raise ValueError("stage_c_diurnal_precip=True requires bl_diurnal_wind=True -- the diurnal precip "
                         "coupling re-steps the moisture on the breeze solve's per-phase winds (u_phase/"
                         "v_phase), which only the #113 diurnal wind solve produces.")
    coupling = None
    ad = None
    if couple:
        cpl = couple_stage_a_b(profile, verbose=verbose, a_warmstart=a_warmstart)
        A, B, dA, dB = cpl["A"], cpl["B"], cpl["dA"], cpl["dB"]
        sst_C = dA["T_surf_C"]; sst_of_theta = cpl["sst_of_theta"]
        wetness_of_theta = _interp(dA["theta_deg"], A._wetness())
        coupling = dict(fiducial=cpl["fiducial"], drift_hist=cpl["drift_hist"],
                        fb_resid_hist=cpl["fb_resid_hist"], dA_diffusive=cpl["dA_diffusive"])
    elif core == "ad":
        # ---- the FUSED dimensional core (M2b): the AD core supplies SST + circulation + band + the LAND
        #      atmosphere (downwelling LW + near-surface air-T) + the f_loft plume environments, all through
        #      the Stage-C interface contract. The legacy diffusive Stage A is NOT run in this path: it was
        #      vestigial -- land_surface_diurnal (atm=land_atm) and floft_sst_table (envs=floft_envs) both
        #      OVERRIDE it with the core, and its only other use was a theta-RANGE for th_grid (clamped to
        #      [28,62], identical from the core's theta grid). Dropping the per-call A.run() saves ~2.6 min/
        #      seed and is byte-identical (Jake, 2026-06-25). A stays None; the downstream A-args are the
        #      override-shadowed fallbacks. (The diffusive path below still runs Stage A as its backbone.)
        A = None
        if ad_core is not None:
            # a PRE-BUILT, already-loaded core supplied by the caller (e.g. nsub30_backbone): reuse it
            # across many topologies without re-building/re-loading per call. stage_c_inputs() is
            # read-only (no stepping), so passing the same core object to N run_cascade calls is safe.
            ad = ad_core
            if verbose: print("=== Stage AD core (PRE-BUILT/loaded core supplied by caller) ===")
        elif ad_checkpoint is not None:
            # the converged-world path (geology arc): load a fleet checkpoint, no in-cascade spin-up
            ad, _ad_day = _load_ad_core(ad_checkpoint, warm_days=ad_warm_days, verbose=verbose)
        else:
            ad = _run_ad_core(profile, ad_overrides, ad_rce_steps, ad_dyn_steps, verbose=verbose)
        sci = ad.stage_c_inputs()
        dad = sci["diagnostics"]
        dA = dad                                                  # theta-range source for th_grid/th_probe
                                                                  # (replaces the dropped Stage A diagnostics)
        # locks the "dropping Stage A is byte-identical" claim: th_grid clamps to [28,62], identical to the
        # dropped Stage A's range iff the core's theta grid spans beyond that band (it covers ~0-80).
        assert float(dad["theta_deg"].min()) < 28.0 and float(dad["theta_deg"].max()) > 62.0, (
            "AD-core theta grid must span [28,62] for the Stage-A drop to keep th_grid byte-identical")
        sst_C = dad["T_surf_C"]
        sst_of_theta = sci["sst_of_theta"]
        wetness_of_theta = sci["wetness_of_theta"]
        # the AD core's diagnostics carry the Stage-C gates' input contract (theta_deg / w / precip),
        # so cell_ascent_gate / subsidence_gate / the band F all consume it identically to Stage B's dB
        dB = dict(theta_deg=dad["theta_deg"], w=dad["w"], precip=dad["precip_mm_day"],
                  surf_sign_changes=dad["surf_sign_changes"], psi=ad.psi.copy(),
                  rising_theta=dad["rising_theta"],   # carried for the figures (make_figures panel 2)
                  aloft_moisture_mm=dad.get("aloft_moisture_mm"))   # absolute aloft column for the F-supply fix
        B = None                                                  # the AD core SUBSUMES Stage B
    else:
        # ---- legacy one-way: Stage A diffusive backbone -> Stage B driven by A's SST ----
        A = StageA(cfgmod.stage_a(profile))
        if verbose: print("=== Stage A (backbone) ===")
        A.run(verbose=verbose)
        dA = A.diagnostics()
        sst_C = dA["T_surf_C"]
        sst_of_theta = _interp(dA["theta_deg"], sst_C)
        B = StageBAtmosphere(cfgmod.stage_b(profile, Tsurf_C_of_theta=sst_of_theta))
        if verbose: print("=== Stage B (overturning, driven by A's SST) ===")
        B.run(verbose=verbose)
        dB = B.diagnostics()
        wetness_of_theta = _interp(dA["theta_deg"], A._wetness())

    # the poleward-aloft band F(theta) [mm]. DEFAULT (legacy, byte-identical): the normalize-to-shape law
    # (8 mm floor -> ~50 mm at the band peak). OPT-IN absolute (AD core only): the resolved aloft moisture
    # COLUMN, so the chimney spike can no longer dominate a /max() and crush the ribbon (build_aloft_supply_
    # target; env COSMO_ALOFT_SUPPLY_ABSOLUTE / kwarg aloft_supply_absolute; default OFF). Stage B path has
    # no absolute column -> always legacy.
    from .stage_ad_core import _env_flag
    absolute_supply = (aloft_supply_absolute if aloft_supply_absolute is not None
                       else _env_flag("COSMO_ALOFT_SUPPLY_ABSOLUTE", False))
    F_target = build_aloft_supply_target(dB["theta_deg"], dB["precip"],
                                         aloft_moisture_mm=dB.get("aloft_moisture_mm"),
                                         absolute=absolute_supply)
    F_of_theta = _interp(dB["theta_deg"], F_target)

    # The wetness ramp (ocean=1, salt flat=0) wired into Stage C so its sea-surface evaporation matches the
    # backbone's (no open-ocean evaporation over the theta<40 salt-flat hot zone) -- the same
    # absolute-footing wetness mask. From Stage A (diffusive/coupled paths) or the AD core (core="ad"); set
    # in the backbone branch above. Removes the hot-zone precip blow-up (OPEN_ISSUES c2).

    # ---- land-temperature backbone + RESOLVED DIURNAL CYCLE (Issue #21) ----
    # Two continentality endpoints run through Stage A's diurnally-resolved land-surface balance
    # (flatworld.surface): the damp COAST (C_MOIST_LAND, wetter) and the arid INTERIOR (C_ARID_LAND, drier,
    # lower thermal inertia). Each gives a day-mean AND the day/night extremes. Stage C blends them
    # coast<->interior by continentality and carries T_min/T_max per cell (the 10-60 K land swing the prior
    # day-mean-minus-fixed-offset model omitted). NOTE the solved interior day-mean runs slightly WARMER than
    # the coast (a seasonless world has no cold winter -- the continental signal is the SWING), superseding the
    # legacy continentality_K cooling.
    th_grid = np.linspace(max(dA["theta_deg"].min(), 28), min(dA["theta_deg"].max(), 62), 40)
    day_cycle = None
    S_of = None                                                   # the core's day-mean flux S(theta) (#113 reuse)
    if lantern_day:
        if ad is None:
            raise ValueError("lantern_day=True requires core='ad' -- the core's day-mean S(theta) "
                             "anchors the painted family")
        from . import lantern
        S_of = _interp(dad["theta_deg"], dad["S"])
        day_cycle = lambda x: lantern.day_flux_cycle(x, S_of)     # noqa: E731
        if verbose:
            print("  diurnal forcing: LANTERN painted day (16-h band cosine; day-means = the core's S)")
    # Drive the LAND surface balance off the AD core's atmosphere (its downwelling LW + near-surface
    # air-T) when running the fused core, so land rides the resolved-circulation gradient instead of the
    # transport-poor Stage A column. Stage A's greenhouse-collapse cliff otherwise crushes the land
    # habitable band (and its Koppen classes) to ~1 deg of theta while the ocean SST -- already from the
    # core -- keeps the true ~9 deg band. Legacy/diffusive paths (ad is None) keep the Stage A atmosphere.
    # NOTE both continentality endpoints below (damp coast, arid interior) share this ONE column atmosphere:
    # the coast<->interior contrast is the surface wetness + heat capacity, NOT a distinct air mass (Stage A
    # drove both off a single column too -- a standing simplification, not introduced here).
    land_atm = None
    if ad is not None:
        land_atm = (_interp(dad["theta_deg"], dad["D_surf"]),
                    _interp(dad["theta_deg"], dad["T_air_surf_C"] + C.T_FREEZE))
    Tc_mean, Tc_min, Tc_max = land_surface_diurnal(A, th_grid, wetness=0.40, C_surf=C.C_MOIST_LAND,
                                                   day_cycle=day_cycle, atm=land_atm)
    Ti_mean, Ti_min, Ti_max = land_surface_diurnal(A, th_grid, wetness=0.10, C_surf=C.C_ARID_LAND,
                                                   day_cycle=day_cycle, atm=land_atm)
    Tland = Tc_mean                                                  # coastal mean = the legacy backbone
    tland_of_theta = _interp(th_grid, Tc_mean)
    tland_interior_of_theta = _interp(th_grid, Ti_mean)
    dT_coast = np.maximum(Tc_max - Tc_min, 0.0)
    dT_interior = np.maximum(Ti_max - Ti_min, 0.0)
    dT_coast_of_theta = _interp(th_grid, dT_coast)
    dT_interior_of_theta = _interp(th_grid, dT_interior)
    # per-endpoint diurnal skew = mean fraction of the range sitting ABOVE the day-mean (noon-heavy for arid)
    skew_coast = float(np.clip(np.mean((Tc_max - Tc_mean) / np.maximum(dT_coast, 1e-6)), 0.0, 1.0))
    skew_interior = float(np.clip(np.mean((Ti_max - Ti_mean) / np.maximum(dT_interior, 1e-6)), 0.0, 1.0))
    if verbose:
        rib = (th_grid >= 45) & (th_grid <= 51)
        print(f"  land diurnal cycle (Issue #21): coast dT~{dT_coast[rib].mean():.0f} K, "
              f"interior dT~{dT_interior[rib].mean():.0f} K at the ribbon (skew {skew_coast:.2f}/{skew_interior:.2f}); "
              f"interior mean {Ti_mean[rib].mean()-Tc_mean[rib].mean():+.1f} K vs coast")

    # physically-anchored convective<->stratiform split f_loft(theta) for Stage C's LOCAL convective
    # lofting (OPEN_ISSUES Brief 2 / c3). It is the PRODUCT of two physical inputs:
    #   (1) Stage B's cell-ascent GATE a_cell(theta) -- WHERE local lofting happens: the cell rises in the
    #       band branch (theta~37-44) and subsides over the inner sea / ribbon, so local moisture is lofted
    #       to the poleward layer only in the ascending branch; over the subsiding wet ocean it is NOT
    #       lofted (it rains locally / rides equatorward to loft at the equatorial updraft, the F(theta)
    #       pathway -- PR #7's circulation-set band at theta~36). This is the equatorward-vs-poleward
    #       partition set by the cell, not by a local-column proxy.
    #   (2) the entraining-plume EFFICIENCY f_loft_plume(theta) ~0.7 -- of the moisture lofted in the
    #       ascending branch, the fraction detrained aloft vs rained in place (frozen, LOCKED eps, PE;
    #       built from A's own wet-ocean columns, no extra Stage A run).
    # f_loft_eff(theta) = a_cell(theta) * f_loft_plume(theta). Replaces the free scalar f_loft=0.5. None
    # falls back to the scalar (floft_anchored=False, or too few wet columns to anchor).
    floft_of_theta = None
    cape_of_theta = None                                     # #114: per-cell CAPE source for gusts (set below)
    if floft_anchored:
        from . import plume_floft as pf
        try:
            # ONE entraining-plume table feeds BOTH the f_loft split and the CAPE map (review #3 -- the
            # table already returns both columns; no second plume integration). In the fused-core path the
            # plume ENVIRONMENT (the T/q sounding the updraft entrains) is read from the AD core's resolved
            # columns, not Stage A's transport-poor ones -- so the convective/stratiform split rides the same
            # circulation gradient as the rest of the deliverable (the land-T fix's sibling).
            floft_envs = pf.ad_core_environments(ad) if ad is not None else None
            s_tab, f_tab, _, cape_tab = pf.floft_sst_table(A, pf.EPS_FROZEN, pf.PE_FROZEN, envs=floft_envs)
            floft_plume = pf.floft_of_theta_from_table(s_tab, f_tab, sst_of_theta)
            ascent = cell_ascent_gate(dB)
            floft_of_theta = lambda th: np.clip(np.asarray(ascent(th), float)
                                                * np.asarray(floft_plume(th), float), 0.0, 1.0)
            if verbose:
                th_probe = np.linspace(max(dA["theta_deg"].min(), 28), 52, 13)
                fv = np.asarray(floft_of_theta(th_probe), float)
                pk = th_probe[int(np.argmax(fv))]
                print(f"  f_loft(theta) = cell-ascent gate x plume efficiency (eps={pf.EPS_FROZEN*1e3:.2f}"
                      f"/km, PE={pf.PE_FROZEN}): peak {fv.max():.2f} @theta{pk:.0f}, "
                      f"~0 over subsidence (was uniform scalar 0.5)")
            # CAPE is DIAGNOSTIC-only: its own guard so a CAPE failure can NEVER discard the f_loft
            # CLIMATE input above (review #1 -- otherwise it would silently revert to scalar f_loft=0.5).
            try:
                cape_of_theta = pf.cape_of_theta_from_table(s_tab, cape_tab, sst_of_theta)
            except Exception as ce:
                if verbose: print(f"  CAPE map skipped ({ce}); gust falls back to turbulence-only")
                cape_of_theta = None
        except Exception as e:
            if verbose: print(f"  f_loft anchoring skipped ({e}); using scalar f_loft=0.5")
            floft_of_theta = None
            cape_of_theta = None

    # convective-drain subsidence gate (Refinement 1): suppress the convective drain where Stage B's single
    # cell SUBSIDES (the warm subsiding ribbon sea), so it stops raining its un-lofted moisture in place.
    # Keyed on the SIGN of B's w(theta) -- the drain sibling of the loft-fraction cell_ascent_gate above.
    subsidence_of_theta = subsidence_gate(dB)

    # ---- Stage C: the dynamical 2-D map over a topology, consuming A (SST, land-T) + B (band) ----
    # topo is either GENERATED (procedural default) or the INPUT topology supplied by the caller.
    topo = _resolve_topo(topo, profile, topo_seed, topo_kwargs, topo_kind=topo_kind)
    # the AD core RESOLVES the zonal surface wind, so the ad path wires the REAL surface return into
    # Stage C's background wind (Jake's seed-137 review: the legacy flat -4 m/s _default_ubg is a
    # pre-core placeholder -- b23 blows ~-34 m/s across the band, and the sea panel showed the
    # placeholder). Other backbones keep the default (B is nondimensional, A has no wind).
    ubg_kw = {}
    if ad is not None and "u_surf" in dad:
        # PCHIP, not linear: ubg(theta) renders bare over Phi-flat sea rows, so linear-node kinks
        # band the wind panel (see _interp_smooth). Values move only between backbone nodes (C1 vs
        # C0 of the same node set); the cap/clip behavior downstream is unchanged.
        ubg_kw["ubg_of_theta"] = _interp_smooth(dad["theta_deg"], dad["u_surf"])
    # salt-aerosol CCN warm-rain coupling (#237, Stage-C side): when the AD core resolved a salt N_d field,
    # wire it (and the core's reference / susceptibility) into Stage C so the deliverable map P(x,e) carries
    # the salt band-magnitude effect SELF-CONSISTENTLY with the column's autoconv limb (#243). Empty (-> Stage
    # C salt_ccn stays OFF, byte-identical) on every other backbone and whenever the core's salt_ccn is OFF
    # (N_d None). stage_c_overrides still wins (applied last), so a sweep can override beta / N_d.
    salt_ccn_kw = {}
    if ad is not None and sci.get("salt_ccn_nd_of_theta") is not None:
        salt_ccn_kw = dict(salt_ccn=True, salt_ccn_nd_of_theta=sci["salt_ccn_nd_of_theta"],
                           salt_ccn_nd_ref=ad.cfg.salt_ccn_nd_ref,
                           salt_ccn_beta=ad.cfg.salt_ccn_autoconv_beta)
        # drop any salt key the caller set explicitly so stage_c_overrides wins (no duplicate-kwarg collision)
        salt_ccn_kw = {k: v for k, v in salt_ccn_kw.items() if k not in (stage_c_overrides or {})}
    # resolved aloft thermodynamics (Workstream C): the AD core exposes the RESOLVED aloft profile
    # (T_aloft, q_aloft, Wsat_aloft, RH_aloft); wire it into Stage C so it can retire the fake fixed-lapse
    # aloft temperature and take Wsat_aloft [mm] from the real column. HARMLESS to pass unconditionally --
    # Stage C ignores these callables unless cfg.aloft_thermo_from_core (env COSMO_ALOFT_THERMO_FROM_CORE) is
    # ON, so the default map is byte-identical. Empty on every non-AD backbone (no resolved profile). The flag
    # itself is NOT forced here: it rides the StageCConfig env default (or an explicit stage_c_overrides), so
    # the shipped default stays the fixed-lapse path.
    aloft_thermo_kw = {}
    if ad is not None:
        aloft_thermo_kw = dict(Wsat_aloft_of_theta=sci.get("Wsat_aloft_of_theta"),
                               T_aloft_of_theta=sci.get("T_aloft_of_theta"),
                               RH_aloft_of_theta=sci.get("RH_aloft_of_theta"))
        # drop any key the caller set explicitly so stage_c_overrides wins (no duplicate-kwarg collision)
        aloft_thermo_kw = {k: v for k, v in aloft_thermo_kw.items() if k not in (stage_c_overrides or {})}
    # stage_c_overrides: tune Stage C's StageCConfig WITHOUT switching the whole `profile` (e.g. enable
    # the resolved_supply B-closure on the test Stage A, when the BACKBONE is the canonical world -- the
    # production profile's Stage A is a different, here-unstable config). Applied last so it wins.
    Ccfg = cfgmod.stage_c(profile, **ubg_kw, **salt_ccn_kw, **aloft_thermo_kw,
                          sst_C_of_theta=sst_of_theta, Tland_C_of_theta=tland_of_theta,
                          Tland_interior_C_of_theta=tland_interior_of_theta,
                          dT_diurnal_coast_of_theta=dT_coast_of_theta,
                          dT_diurnal_interior_of_theta=dT_interior_of_theta,
                          diurnal_skew_coast=skew_coast, diurnal_skew_interior=skew_interior,
                          aloft_supply_of_theta=F_of_theta, wetness_of_theta=wetness_of_theta,
                          f_loft_of_theta=floft_of_theta, subsidence_of_theta=subsidence_of_theta,
                          **(stage_c_overrides or {}))
    Cmap = StageCMap(topo, Ccfg)
    if verbose: print("=== Stage C (dynamical map, driven by A+B backbone) ===")
    Cmap.run(verbose=verbose)
    dC = Cmap.diagnostics()
    # the stop record travels WITH the product (PR #852: this call used to discard it, so no product said
    # whether its climate converged). A plain dict; save_result_npz pickles dC, so it persists as-is and
    # every existing reader ignores the extra key. Deliverable routes gate on it
    # (stagec_convergence.require_converged / check_product).
    dC["stage_c_run"] = dict(Cmap.last_run)
    # ---- issue #114: output a per-cell CAPE map (entraining-plume CAPE via the SST backbone) so the
    # gust convective-outflow bump (surface_wind.attach_gust) has a source. DIAGNOSTIC only -- it feeds
    # no prognostic and no climate field; absent when f_loft anchoring is off (gust falls back to the
    # turbulence-only factor). CAPE(theta) is per-row (the air mass's convective vigor at that latitude,
    # peaking over the hot inner sea), broadcast across the along-track axis. ----
    if cape_of_theta is not None:
        from . import plume_floft as pf
        dC["cape"] = pf.cape_map_from_theta(cape_of_theta, dC["theta_deg"],
                                            np.asarray(dC["T_map_C"]).shape)
        if verbose:
            print(f"  CAPE map output (dC['cape']): peak {float(np.nanmax(dC['cape'])):.0f} J/kg "
                  f"(entraining-plume CAPE via SST backbone; gust convective-outflow source)")
    # ---- issue #111: replace the REPORTED wind with the nonlinear BL momentum solve ----
    # Stage C's internal moisture advection keeps using its capped linear-balance wind (wind_cap/phi_smooth_km
    # untouched -- the moisture CFL needs them); only the REPORTED (u, v) -- what surface_wind.py and the
    # habitability ladder read -- switches to the energetically self-limiting momentum field. Default OFF
    # (byte-identical). The balance wind is preserved as u_bal/v_bal for provenance.
    if bl_momentum_wind:
        from . import bl_momentum as blm
        blcfg = bl_momentum_cfg or blm.BLMomentumConfig()
        if verbose: print("=== reported wind: nonlinear BL momentum solve (#111) ===")
        sol = blm.build_wind_field(dC, Cmap.dx, Cmap.de, blcfg, verbose=verbose)
        dC["u_bal"], dC["v_bal"] = dC["u"], dC["v"]                 # provenance: the linear-balance wind
        dC["u"], dC["v"] = sol["u"], sol["v"]                        # reported field switches
        for k in ("froude", "wind_div", "bl_depth", "wave_c2"):      # extra per-cell diagnostics (#111)
            dC[k] = sol[k]
        dC["bl_momentum"] = dict(converged=sol["converged"], iters=sol["iters"],
                                 bernoulli_excess_frac=sol["bernoulli_excess_frac"],
                                 drag_form=blcfg.drag_form, stop_reason=sol["stop_reason"])
        if verbose:
            spd = sol["speed"]
            print(f"  reported wind: <|V|>={spd.mean():.1f} p90={np.percentile(spd,90):.1f} "
                  f"max={spd.max():.1f} m/s  Bernoulli-excess={100*sol['bernoulli_excess_frac']:.2f}%")
    # deliver the central crop of a padded tectonic topology (segment of an infinite plane); the global
    # water closure stays on the full padded periodic domain. No-op for the unpadded procedural default.
    topo, dC = _deliver_crop(topo, dC)

    # ---- classify the deliverable maps ----
    P_yr = dC["P_mm_day"] * 365.0
    # frost-aware classification (Issue #21 follow-up): pass the screen-level AIR extremes so the E family
    # (EF never thaws / ET freezes nightly) and the frost-prone habitability class key on the air min/max.
    koppen, koppen_frosts = classify.koppen_class(dC["T_map_C"], P_yr, dC["land"],
                                                  dC["T_air_min_C"], dC["T_air_max_C"])
    hab = classify.habitability_class(dC["T_map_C"], P_yr, dC["land"], dC["T_air_min_C"])

    # ---- issue #113: the DIURNAL sea/land-breeze solve (afternoon wind max, calm mornings, coastal
    # reversals). Time-steps the #111 BL momentum layer through the lantern day to a periodic state; it is
    # ADDITIONAL diagnostics -- the reported day-mean wind dC["u"]/["v"] is left untouched (the day-mean is
    # unchanged, acceptance 3). Default OFF (byte-identical). The breeze is premise-dependent, so it REQUIRES
    # the lantern painted day. Runs after classify so the per-Koppen day/night ratios key on the final map. ----
    if bl_diurnal_wind:
        from . import bl_diurnal as bldm
        dcfg = bl_diurnal_cfg or bldm.BLDiurnalConfig()
        if verbose: print("=== reported wind: DIURNAL sea/land-breeze solve (#113) ===")
        fdiu = bldm.build_diurnal_forcing(dC, Cmap.dx, Cmap.de, dcfg, S_of, clock=g.CLOCK)
        sold = bldm.solve_diurnal_wind(fdiu["phi_of"], dC["ubg"], Cmap.dx, Cmap.de, dcfg, fdiu["phase_t"],
                                       T_map_C=dC["T_map_C"], height=dC["height"], verbose=verbose)
        diag = bldm.diurnal_diagnostics(sold, dC, koppen=koppen, dx=Cmap.dx, de=Cmap.de, clock=g.CLOCK)
        dC["diurnal_phase_t"] = sold["phase_t"]
        dC["diurnal_speed_phase"] = sold["speed_phase"]
        dC["diurnal_afternoon_peak_ms"] = sold["afternoon_peak_speed"]
        dC["diurnal_night_min_ms"] = sold["night_min_speed"]
        dC["diurnal_day_mean_ms"] = sold["day_mean_speed"]
        dC["diurnal_day_mean_u"] = sold["day_mean_u"]; dC["diurnal_day_mean_v"] = sold["day_mean_v"]
        dC["diurnal_cross_shore_phase"] = diag["cross_shore_phase"]
        dC["diurnal_breeze_amplitude_ms"] = diag["breeze_amplitude"]
        dC["diurnal_breeze_amp_by_koppen"] = diag["breeze_amp_by_koppen"]
        dC["bl_diurnal"] = dict(converged=sold["converged"], days=sold["days"],
                                stop_reason=sold["stop_reason"], rectification_frac=sold["rectification_frac"],
                                breeze_reversal_frac=diag["breeze_reversal_frac"],
                                breeze_amp_median_ms=diag["breeze_amp_median_ms"],
                                breeze_amp_p90_ms=diag["breeze_amp_p90_ms"],
                                sea_breeze_peak_hour=diag["sea_breeze_peak_hour"],
                                land_breeze_peak_hour=diag["land_breeze_peak_hour"],
                                daynight_ratio_windfacing=diag["daynight_ratio_windfacing"],
                                daynight_ratio_lee=diag["daynight_ratio_lee"],
                                sea_breeze_penetration_km=diag["sea_breeze_penetration_km"],
                                penetration_box_saturated=diag["penetration_box_saturated"],
                                breeze_amp_by_koppen=diag["breeze_amp_by_koppen"])
        if verbose:
            print(f"  diurnal: converged={sold['converged']} in {sold['days']}d; day-mean vs #111 steady "
                  f"rectification={100*sold['rectification_frac']:.1f}%; sea-breeze reverses at "
                  f"{100*diag['breeze_reversal_frac']:.0f}% of coasts (amp median {diag['breeze_amp_median_ms']:.1f}, "
                  f"p90 {diag['breeze_amp_p90_ms']:.1f} m/s), peaks ~{diag['sea_breeze_peak_hour']:.0f}h, "
                  f"intrusion ~{diag['sea_breeze_penetration_km']:.0f} km"
                  f"{' (box-saturated, lower bound)' if diag['penetration_box_saturated'] else ''}")
            print(f"  orientation dipole (day/night SPEED): {diag['daynight_ratio_windfacing']:.2f} on "
                  f"wind-facing coasts vs {diag['daynight_ratio_lee']:.2f} on lee coasts "
                  f"(spatial mean ~{diag['daynight_ratio_spatial_mean']:.2f}; night_factor unchanged -- the "
                  f"~60% nocturnal decoupling is a separate surface-layer effect this solve does not resolve)")

    # ---- issue #131: couple the DIURNAL breeze into Stage C's PRECIP. Stage C's deliverable precip rides
    # the STATIC day-mean wind, so the strong afternoon sea-breeze convergence (a nonlinear rain trigger) is
    # averaged away. Re-step the converged two-layer moisture on the breeze solve's per-phase winds (``sold``,
    # from the bl_diurnal_wind block above, which this REQUIRES) to a periodic state -> the resolved diurnal
    # precip cycle + a rectification-corrected day-mean. Additional diagnostics only: the deliverable
    # dC["P_mm_day"] was captured before this and is NOT overwritten. Default OFF (byte-identical). ----
    if stage_c_diurnal_precip:
        from . import stage_c_diurnal as scdm
        scfg = stage_c_diurnal_cfg or scdm.StageCDiurnalConfig()
        if verbose: print("=== precip: DIURNAL sea/land-breeze coupling (#131) ===")
        # the coupling re-steps the full Cmap moisture on the breeze winds; for a PADDED topology
        # _deliver_crop cropped dC (so sold's per-phase winds are cropped) while Cmap stays full -> a shape
        # mismatch. Supported on the unpadded/procedural path only; fail loudly otherwise (not garbage).
        if sold["u_phase"].shape[1:] != Cmap.W_s.shape:
            raise NotImplementedError(
                "stage_c_diurnal_precip: the breeze winds (built on the possibly-cropped dC, shape "
                f"{sold['u_phase'].shape[1:]}) do not match the Stage C grid (Cmap {Cmap.W_s.shape}) -- the "
                "diurnal-precip coupling currently supports the UNPADDED/procedural topology only (the "
                "deliver-crop padded path is not wired). Use topo_kind='procedural' or an unpadded topo.")
        scsol = scdm.solve_diurnal_precip(Cmap, sold, scfg, clock=g.CLOCK, verbose=verbose)
        dC["P_phase"] = scsol["P_phase"]                              # (K, nx, ne) resolved diurnal precip cycle
        dC["P_daymean_rect"] = scsol["P_daymean_rect"]               # rectification-corrected day-mean precip
        dC["P_afternoon"] = scsol["P_afternoon"]                      # precip at the afternoon sea-breeze phase
        dC["P_diurnal_amplitude"] = scsol["P_amplitude"]            # max-min over the day (the rain rhythm)
        dC["diurnal_precip_phase_t"] = scsol["phase_t"]
        # the rectification metrics (sign + correlations) -- the firm claim is sign/location, mm/day is soft
        r_aft = scdm.precip_convergence_corr(scsol["P_afternoon"], scsol["conv_afternoon"])
        r_dm = scdm.precip_convergence_corr(scsol["P_daymean_rect"], scsol["conv_daymean"])
        # the rain SWING vs the convergence SWING (both max-min): the static background convergence cancels,
        # isolating the breeze-driven rhythm. Matched field-types (swing vs swing) -- NOT P_amplitude vs the
        # instantaneous conv_afternoon, which conflates a swing with a background-dominated snapshot.
        r_amp = scdm.precip_convergence_corr(scsol["P_amplitude"], scsol["conv_amplitude"])
        # net total-rain change: rectified cycle-mean vs the static deliverable. NB P_daymean_rect is the
        # mean of the pre-EMA instantaneous precip over the cycle, dC["P_mm_day"] is the EMA-smoothed static
        # steady map -- at a converged steady state the EMA == the steady instantaneous, so this is
        # effectively instantaneous-cycle-mean vs instantaneous-static (like-for-like); tiered SOFT regardless.
        tot_rect = float(scsol["P_daymean_rect"].sum())
        tot_static = float(dC["P_mm_day"].sum())
        dC["stage_c_diurnal"] = dict(converged=scsol["converged"], days=scsol["days"],
                                     stop_reason=scsol["stop_reason"], n_sub_max=scsol["n_sub_max"],
                                     water_closure_last_cycle=scsol["water_closure_last_cycle"],
                                     k_aft=scsol["k_aft"], corr_afternoon=r_aft, corr_daymean=r_dm,
                                     corr_amplitude=r_amp,
                                     total_rain_change_frac=(tot_rect - tot_static) / max(tot_static, 1e-9))
        if verbose:
            print(f"  diurnal precip: converged={scsol['converged']} in {scsol['days']}d (n_sub<= "
                  f"{scsol['n_sub_max']}); closure={scsol['water_closure_last_cycle']:.3f}; "
                  f"total convergent rain change {100*dC['stage_c_diurnal']['total_rain_change_frac']:+.1f}% "
                  f"(rectified day-mean vs static); afternoon precip<->convergence r={r_aft:.2f} "
                  f"(rain-rhythm<->convergence r={r_amp:.2f})")

    if verbose:
        # use the sst_of_theta callable (grid-independent) so the ribbon mean is well-posed for any backbone
        print("\n=== cascade result ===")
        print(f"  backbone ({core}) ribbon SST: "
              f"{np.mean(np.asarray(sst_of_theta(np.linspace(45, 51, 7)), float)):.1f} C")
        print(f"  single cell: surf-sign-changes={dB['surf_sign_changes']}")
        print(f"  map water closure: {dC['water_closure_cum']:.3f}")
        print(f"  Koppen fractions: {classify.class_fractions(koppen, classify.LABELS)}")
        print(f"  habitability:     {classify.class_fractions(hab, classify.HAB)}")
    return dict(A=A, B=B, ad=ad, C=Cmap, dA=dA, dB=dB, dC=dC, topo=topo,
                koppen=koppen, koppen_frosts=koppen_frosts, habitability=hab, P_yr=P_yr, coupling=coupling,
                backbone=dict(sst_C=sst_C, F_target=F_target, Tland=Tland, theta_grid=th_grid))


def save_result_npz(res, path):
    """Persist a run_cascade() result to an npz readable by make_figures' cache loader.

    Mirrors make_figures' cache format exactly -- keys dA/dB/dC/koppen/topo, with the diagnostics
    dicts stored as object arrays (recover with ``z["dA"].item()``) and the TopoGrid stored as a
    ``vars()`` dict (recover with ``SimpleNamespace(**z["topo"].item())``). Keeps a full production
    A->B->C result plottable without re-running the cascade. Stage checkpoints (stage*_prod.npz) are
    separate and untouched."""
    np.savez(path, dA=res["dA"], dB=res["dB"], dC=res["dC"],
             koppen=res["koppen"], topo=vars(res["topo"]))


if __name__ == "__main__":
    import sys
    # usage: python -m flatworld.cascade [profile] [topo_seed] [topo_path] [core]
    #        python -m flatworld.cascade ckpt <tag> [profile] [topo_path]
    #   topo_path (optional): an .npz topology (Topography.save format) to INPUT instead of generating a
    #   random continent -- e.g. a topology produced by a separate geophysics model.
    #   core (optional): "diffusive" | "ad". DEFAULT POLICY: the ``test`` profile uses the CHEAP DIFFUSIVE
    #   backbone (the fast regression gate -- ``cascade test N`` stays green/quick), and ``production`` uses
    #   the all-on AD-core (the maximally-accurate default). Pass an explicit 4th arg to override.
    #   ckpt mode: run the cascade off a CONVERGED fleet checkpoint (the geology-arc adapter). Env-bearing
    #   tags bake constants at import, so this mode RE-EXECS itself with the spec's env applied
    #   (`ckpt-inproc` is the post-exec entry; not for direct use).
    if len(sys.argv) > 1 and sys.argv[1] in ("ckpt", "ckpt-inproc"):
        import os
        import subprocess
        tag = sys.argv[2]
        profile = sys.argv[3] if len(sys.argv) > 3 else "test"
        topo = sys.argv[4] if len(sys.argv) > 4 else None
        if sys.argv[1] == "ckpt":
            from . import _warm_restart as _wr
            env = dict(os.environ)
            env.update(_wr._spec_of(tag)["env"] or {})
            env.pop("PYTHONPATH", None)
            cmd = [sys.executable, "-m", "flatworld.cascade", "ckpt-inproc", tag, profile] \
                + ([topo] if topo else [])
            raise SystemExit(subprocess.run(cmd, env=env).returncode)
        res = run_cascade(profile=profile, topo=topo, core="ad", ad_checkpoint=tag,
                          lantern_day=True)    # the ckpt worlds are lantern worlds (locked premise)
        out = f"cascade_ckpt_{tag}_{profile}.npz"
        save_result_npz(res, out)
        print(f"saved checkpoint-backbone cascade result to {out}")
        raise SystemExit(0)
    profile = sys.argv[1] if len(sys.argv) > 1 else "test"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    topo = sys.argv[3] if len(sys.argv) > 3 else None
    core = sys.argv[4] if len(sys.argv) > 4 else ("diffusive" if profile == "test" else "ad")
    res = run_cascade(profile=profile, topo_seed=seed, topo=topo, core=core)
    out = f"cascade_{profile}.npz"                      # e.g. cascade_production.npz / cascade_test.npz
    save_result_npz(res, out)
    print(f"saved full cascade result to {out}")
