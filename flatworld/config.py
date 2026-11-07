"""
flatworld.config -- named run configurations and sensitivity sweeps.

Two profiles: "test" (low resolution / reduced domain -- the validation configuration used by the
test suite and the convergence/box-size demos, runs in seconds to a few minutes) and "production"
(full resolution, the launch configuration documented in README.md, designed to fit the ~10 h budget;
see the cost extrapolation there). Sensitivity-sweep helpers vary the documented forcing inputs
(I0, albedo, CO2) so the predicted temperature is reported with its dependence, never as a single
hostage to the free amplitude.
"""
from __future__ import annotations
from dataclasses import replace
from .stage_a_column import StageAConfig
from .stage_b_overturning import StageBConfig
from .climate_map import StageCConfig
from .bl_momentum import BLMomentumConfig
from .bl_diurnal import BLDiurnalConfig
from .stage_c_diurnal import StageCDiurnalConfig


# -------------------------------------------------------------------- named profiles
def stage_a(profile="test", **overrides):
    if profile == "test":
        cfg = StageAConfig(nx=48, nz=24, dt=7200.0, ocean_depth_spinup=3.0,
                           check_every=250, patience=4, tol=1.5e-4, max_seconds=600)
    elif profile == "production":
        # full resolution; ocean_depth_spinup kept small (equilibrium is C-independent) to converge in
        # time. The convergence metric is restricted to the habitable band + hot zone (conv_theta_max=58)
        # so the slow deep-cold radiative tail (tiny T^4, creeps for a long time) does not gate the stop;
        # tol=1e-4 K/step is then comfortably reached by the band (OPEN_ISSUES Q8a). Checkpointing lets a
        # guarded run resume.
        #
        # nx=512, dt=7200 s (2 h): with implicit_diffusion (default, OPEN_ISSUES Issue D resolved) the
        # MSE-diffusion CFL is gone, so the stable dt no longer collapses as ~nx^-3 -- it is stable to
        # ~21 h at nx=512 (the nx-independent local-physics ceiling). dt=7200 is a conservative,
        # accuracy-preserving choice (a regression vs the explicit scheme matches the converged ribbon to
        # <0.1 C). nx=512 (dx~54 km) resolves the 0-30 C habitable band to 24 cells ON THE WORLD THAT
        # CERTIFICATE WAS SET ON -- whose band is 2.9 deg wide (theta 49.5-52.4). The transferable quantity
        # is the CELL COUNT, not "nx=512": on the canonical lens_diurnal forcing the Stage-A band is ~0.5 deg
        # wide, so nx=512 buys FOUR cells there and the band edges do not converge until nx=3072 (31 cells).
        # Re-count cells before quoting a converged edge on any new world (PR #437; OPEN_ISSUES "Status of
        # the absolute results"). What nx=512 settles, on the certifying world, is
        # the band's WIDTH and STRUCTURE (the cross-track slope is resolution-converged, edges <0.06 deg
        # nx=256->512); the band's absolute PLACEMENT still inherits the absolute-T uncertainty (+-10 C
        # kappa, +-2.3 C theta_max bounce) through that slope -- so do NOT emit a bare "theta49.5-52.4"
        # without the placement error bar (~+-2 deg at the warm edge); see VALIDATION Finding 1. The
        # conv_theta_max=58 cutoff is band-insensitive (theta<=55/58/62 give identical edges,
        # flatworld.band_cutoff_test), a pure time-saver. The sponge cold boundary is on by default
        # (sponge_relax_frac=0.2).
        cfg = StageAConfig(nx=512, nz=40, dt=7200.0, ocean_depth_spinup=10.0,
                           check_every=500, patience=5, tol=1e-4, conv_theta_max=58.0,
                           max_seconds=9 * 3600, checkpoint_path="stageA_prod.npz",
                           checkpoint_every=20000)
    else:
        raise ValueError(profile)
    return replace(cfg, **overrides)


def stage_b(profile="test", **overrides):
    if profile == "test":
        cfg = StageBConfig(nx=120, nz=31, dt=3.0e-3, check_every=500, patience=4, tol=2e-5,
                           max_seconds=600)
    elif profile == "production":
        cfg = StageBConfig(nx=400, nz=81, dt=1.5e-3, check_every=1000, patience=5, tol=5e-6,
                           max_seconds=9 * 3600, checkpoint_path="stageB_prod.npz",
                           checkpoint_every=50000)
    else:
        raise ValueError(profile)
    return replace(cfg, **overrides)


# Track B / J1 -- the frozen all-sea reference export E_ref for the TOTAL-REDUCTION resolved supply, the
# PRODUCTION default (the moisture-source teleconnection: real continents dry the band by starving its
# upstream evaporative source). Calibrated as the mean export of the ALL-SEA canonical world; J1's geology
# ensemble validated B as first-order (~38 % +- 17 % band-drying, first-order on 100 % of worlds;
# `J1_PRODUCTION_RESULTS.md`). RESOLUTION-STABLE (~1.3 % over nx 72->120), so one scalar suffices. Jake chose
# the all-sea MEAN (allows mild local enrichment where a longitude exports above the mean), 2026-06-19.
#
# TO RE-WIRE A NEW SIMULATED WORLD: when the canonical world changes (new AD-core checkpoint / new backbone),
# re-run the checkpoint-backbone calibrator (`cosmology_models/calibrate_eref_lens_diurnal.py` for the
# current canonical; generalize it for the next world -- all-sea export of the LOADED checkpoint, not a
# fresh spin-up; `python -m flatworld.calibrate_supply_eref` is the non-checkpoint fallback) and paste the
# printed value here. It is frozen-and-locked like GMS_GEOPOT_FROZEN / the plume f_loft -- a calibrated
# constant, not a fitted knob.
SUPPLY_EREF_FROZEN = 1.008e-3   # [mm/s], all-sea mean export, canonical lens_diurnal (run17 ckpt_final,
                                # nx150, gallery domain theta 19.5-60.5, 2026-07-18 promotion; was 1.519e-4
                                # for bm_tau2h day1620 and 2.540e-4 for the retired bl_dz20_i100 world --
                                # the ~6.6x jump tracks the I0=2.5 lensed world's stronger evaporation)
SUPPLY_DIFFUSION_KM = 1500.0    # along-track eddy-diffusion length sigma [km] of the off-map supply pathway
                                # (climate_map.supply_diffusion_km). The PHYSICAL, box-independent replacement
                                # for the box-referenced supply_zonal_mixing: sigma = sqrt(2*K*tau), K_eddy ~
                                # 1e4-1e5 m^2/s (f=0 plane -> NO baroclinic eddies; e is the cell's homogeneous
                                # direction), tau ~ 2e7 s (off-map equatorward->loft->poleward round trip) ->
                                # sigma ~ 0.6-2 Mm, canonical central ~1,500 km (Jake, 2026-06-28). The evaporation
                                # shadow survives this (edges blur ~few thousand km); 6,000 km (Earth-like K~1e6,
                                # likely too high here) is the upper-bound stress test. supply_zonal_mixing stays 0.


def stage_c(profile="test", **overrides):
    if profile == "test":
        cfg = StageCConfig(dt=900.0, check_every=300, patience=4, tol=5e-4, max_seconds=600)
    elif profile == "production":
        # B (supply closure) is ON in production (J1 default-flip, 2026-06-19): total-reduction resolved
        # supply, normalized by the frozen all-sea E_ref. NB the "test" profile keeps B OFF -- E_ref is
        # CANONICAL-WORLD-specific (the all-sea export of the AD-core backbone), and only production runs the
        # canonical world; a standalone/test world has no valid E_ref, so resolved_supply there would
        # mis-normalize. C (prognostic_phi) stays OFF (J1: secondary). wa_rh_crit=0.80 (matching the AD
        # core's CLOUD_RH_CRIT) turns ON the sub-grid/critical-RH stratiform onset so RH_aloft is not pinned
        # to 1 and the map's clouds keep their 2-D longitude structure (OPEN_ISSUES 2026-06-27).
        cfg = StageCConfig(dt=600.0, check_every=500, patience=5, tol=1e-4, max_seconds=9 * 3600,
                           checkpoint_path="stageC_prod.npz", checkpoint_every=20000,
                           resolved_supply=True, supply_zonal_mixing=0.0,
                           supply_ref_export=SUPPLY_EREF_FROZEN, wa_rh_crit=0.80,
                           supply_diffusion_km=SUPPLY_DIFFUSION_KM)
    else:
        raise ValueError(profile)
    # phi_smooth_km is a DERIVED field (= wind_cap/r_fric, resolved in StageCConfig.__post_init__). cfg above
    # already resolved it to 400 km, and dataclasses.replace re-passes that resolved value -- so a wind_cap/
    # r_fric sweep through this entry point would otherwise keep the STALE 400 km (the de-sync issue #88 item 4
    # set out to fix, but via the production sweep path; PR #95 review). Reset it to None so __post_init__
    # re-derives from the swept values, unless the caller set phi_smooth_km explicitly.
    if ("wind_cap" in overrides or "r_fric" in overrides) and "phi_smooth_km" not in overrides:
        overrides["phi_smooth_km"] = None
    return replace(cfg, **overrides)


def bl_momentum(profile="test", **overrides):
    """Config for the #111 nonlinear BL surface-wind solve (the REPORTED wind post-process). 'test' is a
    fast solve; 'production' tightens tol / lifts the guard / checkpoints for the converged read."""
    if profile == "test":
        cfg = BLMomentumConfig(tol=1e-4, patience=4, check_every=50, max_seconds=120)
    elif profile == "production":
        cfg = BLMomentumConfig(tol=2e-5, patience=5, check_every=100, max_seconds=3 * 3600,
                               checkpoint_path="bl_wind_prod.npz", checkpoint_every=20000)
    else:
        raise ValueError(profile)
    return replace(cfg, **overrides)


def bl_diurnal(profile="test", **overrides):
    """Config for the #113 diurnal sea/land-breeze solve (the time-stepped sibling of bl_momentum). Reuses
    the same #111 drag/beta_T/scheme per profile; 'test' is a quick periodic solve, 'production' tightens
    the periodicity tol, lifts the day guard, and checkpoints."""
    mom = bl_momentum(profile)
    if profile == "test":
        cfg = BLDiurnalConfig(mom=mom, n_phase=8, max_days=40, period_tol=2e-3, max_seconds=300)
    elif profile == "production":
        cfg = BLDiurnalConfig(mom=replace(mom, max_seconds=3 * 3600), n_phase=16, max_days=120,
                              period_tol=5e-4, max_seconds=3 * 3600,
                              checkpoint_path="bl_diurnal_prod.npz", checkpoint_every_days=10)
    else:
        raise ValueError(profile)
    return replace(cfg, **overrides)


def stage_c_diurnal(profile="test", **overrides):
    """Config for the #131 diurnal-precip solve (couple the sea/land breeze into Stage C's moisture/precip;
    the time-stepped sibling of stage_c). The phase count K is inherited from the breeze solve. 'test' is a
    quick periodic solve; 'production' tightens the periodicity tol, lifts the day guard, and checkpoints."""
    if profile == "test":
        cfg = StageCDiurnalConfig(max_days=40, period_tol=5e-3, period_patience=2, max_seconds=300)
    elif profile == "production":
        cfg = StageCDiurnalConfig(max_days=120, period_tol=2e-3, period_patience=3, max_seconds=3 * 3600,
                                  checkpoint_path="stage_c_diurnal_prod.npz")
    else:
        raise ValueError(profile)
    return replace(cfg, **overrides)


def topo_production(seed=0):
    """Full-resolution topology grid spec for a production Stage-C run (km / counts)."""
    # NOTE (b23 anchors re-pin, 2026-06-11): theta 28-62 is the OLD world's interesting zone. Under the
    # b23 climatology it EXCLUDES the salt flats (theta<23.2) and CLIPS the snow band at 62 (the snow
    # extends to ~70, so ice-sheet meltwater catchments are undercounted by the 62-70 tail). The
    # production domain + nx await the final-world topography decision (Jake) -- L8 audit, geology arc.
    return dict(seed=seed, nx=300, ne=400, theta_lo=28.0, theta_hi=62.0, Le_km=12000.0)


def topo_production_tectonic(seed=0):
    """Full-resolution TECTONIC topology spec (same grid as topo_production, so it is a drop-in for a
    production Stage-C run; the generator is flatworld.tectonics.generate via Topography.tectonic_continent)."""
    return dict(seed=seed, nx=300, ne=400, theta_lo=28.0, theta_hi=62.0, Le_km=12000.0)


def topo_production_lem(seed=0, nx_fine=1500):
    """Full-resolution **LEM** topology spec (geology roadmap G6/G7): run the landscape-evolution model FINE
    (the worldbuilding deliverable, ``nx_fine``x``ne_fine``) and DOWNSAMPLE to the cascade's climate grid
    (``nx_c``x``ne_c`` = the same 300x400 as ``topo_production``, so the climate grid is unchanged and the
    regression vs. the current production map is apples-to-apples). ``engine='space'``
    (erosion+deposition+rivers+deltas), ``mask_reconcile=True`` (the G6 mask policy). Consumed by
    ``cascade.run_cascade(topo_kind='lem')`` / ``cascade.lem_cascade_topo``. The fine LEM build is the expensive
    step (~minutes with the G7.d route-every-N optimization); Stage A/B/C then run at the usual production cost
    on the 300x400 downsample.

    **The grid is SOLVED for square cells at the live L (issue #83), not hard-coded.** The cross-track axis is
    sized in DEGREES (``x = L*tan theta`` over theta 28-62) and the along-track axis in KM (``Le_km``), so a
    fixed ``nx_fine == ne_fine`` is only square at one L. The L->2L raise (``constants.L_SUN = 2e7``) doubled the
    cross-track span to ``2e7*(tan62-tan28) = 26,980 km``, so the old hard-coded 800x800 silently became 33.7 km
    cross-track x 15.0 km along-track -- geo-mean 22.5 km (OUTSIDE the <=20 km converged window) and 2.25:1
    anisotropic (tripping ``ASPECT_WARN_RATIO``). We now fix ``nx_fine`` and SOLVE ``ne_fine`` via
    ``erosion.square_resolution`` so the physical cells are square at whatever L is pinned: at L=2e7 this is
    **1500 x 667 -> 17.99 km cross-track x 17.99 km along-track (geo-mean 18.0 km, aspect 1.00, converged)**.
    A future L change re-solves automatically (cells stay square; the geo-mean tracks the span, guarded by
    ``erosion.resolution_tier``). ``test_config.test_topo_production_lem_grid_is_square_and_converged`` asserts
    the aspect/resolution guards are silent on the shipped config, so the next L change trips loudly.

    **Cost.** ~1.0e6 fine cells (was 6.4e5 at 800x800) -> ~1.0 GB peak RSS (``erosion.estimate_peak_rss_gb``,
    richdem ~0.6 KB/cell post-#216) and ~1.6x the old erode runtime -- both comfortable. The 300x400 climate downsample is unchanged
    (1500->300 = 5x, 667->400 = 1.67x; ``topo_contract.downsample`` block-means at any ratio).

    **G7.c grid convergence.** The box-shrink convergence study (`G7C_RESULTS.md`) found the LEM texture is
    grid-CONVERGED for geo-mean cells <~18-20 km, and ``erosion.resolution_tier`` enforces that window; the
    solved 18.0 km production cell sits just inside it.

    **G7.d runtime (Jake, 2026-06-08).** The fine build was depression-router-bound (~95% of per-step cost);
    G7.d's route-every-N (``ROUTE_EVERY_PRODUCTION=5``, flowing in via ``lem_cascade_topo`` -> ``lem_contract``)
    cut it ~5-6x, which is what makes the ~1.0e6-cell fine build affordable. **Climate-orography handoff (the G6
    protocol):** re-gridding the fine map and downsampling to the UNCHANGED 300x400 climate grid is an
    **orography-only refresh** -- the land/sea mask is ~99% stable and only the orography shifts mildly (the
    downsample to the coarse climate grid averages out most of the finer detail), so the staged production
    cascade picks up the new orography automatically and the bigger ``cosmology.md`` climate sync stays the
    tracked follow-up.

    **G7.f coarse-to-fine ON by default (Jake, 2026-06-08).** ``coarse_to_fine=True`` here (configurable -- set
    ``False`` for the single-stage build): the production fine LEM is built coarse-first then refined, a further
    ~1.5-1.9x speedup that reproduces a full run on the ensemble-median anchors (``G7F_RESULTS.md``). It makes
    the production landscape a (validated) c2f realization; the c2f-vs-full climate-orography handoff is measured
    mask-stable (an orography-only refresh)."""
    from .erosion import square_resolution                # lazy: keep landlab off the config import path
    nx_fine, ne_fine = square_resolution(int(nx_fine), None, 28.0, 62.0, 12000.0)
    return dict(seed=seed, nx_fine=nx_fine, ne_fine=ne_fine, nx_c=300, ne_c=400, theta_lo=28.0, theta_hi=62.0,
                Le_km=12000.0, engine="space", mask_reconcile=True, coarse_to_fine=True)


# -------------------------------------------------------------------- sensitivity sweeps
def stage_a_sensitivity_grid():
    """The documented forcing/greenhouse sensitivity sweep for the absolute-temperature prediction.
    Returns a list of (label, dict-of-overrides) applied to a stage_a('test') config.

    NOTE (2026-06-04): the suns'-brightness amplitude **I0 is now PINNED at the fiducial 1.0** — a
    worldbuilding choice (M2A0_RESULTS.md; it was the largest, and a *choosable*, ribbon-T slice ~18 C).
    Its three rows are kept here only as a documented **sensitivity** (how the world would differ at
    another brightness), NOT as a carried error-budget uncertainty. The operative radiative uncertainties
    are the physical families below — albedo / CO2 / greenhouse-tau (small, ~2.5 C RSS, evaporatively
    buffered) — plus transport kappa (~10 C) and the omitted clouds (#19), per M2A0_RESULTS.md."""
    grid = []
    for i0 in (0.85, 1.0, 1.15):                      # PINNED at 1.0; swept only for sensitivity (not an uncertainty)
        grid.append((f"I0x{i0}", dict(i0_scale=i0)))
    for alb in (0.06, 0.08, 0.12):                    # ocean albedo
        grid.append((f"albedo{alb}", dict(albedo_ocean=alb)))
    for co2 in (200.0, 400.0, 800.0):                 # CO2 (log greenhouse forcing)
        grid.append((f"co2_{int(co2)}", dict(co2_ppm=co2)))
    for tau in (0.30, 0.40, 0.50):                    # well-mixed LW optical depth (greenhouse strength)
        grid.append((f"tau_wm{tau}", dict(tau_lw_wellmixed=tau)))
    return grid


def cloud_sensitivity_grid():
    """The cloud sensitivity configuration for issue #19, applied as ``StageADConfig`` overrides on the
    fused AD-core (clouds live in the core, not Stage A). Returns a list of (label, dict-of-overrides).

    NOTE: the cloud OPTICAL DEPTH is **computed from physics** (the condensed-water path + droplet size;
    ``radiation.cloud_water_path``), NOT swept. The DEFAULT (``cloud_base``) is the physical best estimate.
    What is varied here is the residual, genuinely-physical microphysical uncertainty -- the in-cloud water
    amount (``cloud_water_scale``: bundles the effective radius / sub-adiabaticity / rainout-ceiling spread)
    and the sub-grid humidity variance (``cloud_rh_crit``) -- to REPORT the sensitivity, NOT to fit it. (This
    is the irreducible cloud-microphysics uncertainty, the dominant one in real climate too.) Run side by
    side -- or via the COSMO_CLOUD* env knobs across processes."""
    grid = [("clear", dict(clouds=False)),
            ("cloud_base", dict(clouds=True))]
    for ws in (0.5, 1.0, 2.0):                         # in-cloud water amount (microphysics: r_eff / sub-adiab / rainout)
        grid.append((f"cloud_water{ws}", dict(clouds=True, cloud_water_scale=ws)))
    for rhc in (0.70, 0.80, 0.90):                     # sub-grid humidity variance (cloud onset RH)
        grid.append((f"cloud_rhc{rhc}", dict(clouds=True, cloud_rh_crit=rhc)))
    for cb in (0.0, 0.3, 0.6):                         # optional resolved-ascent convective-cloud boost
        grid.append((f"cloud_conv{cb}", dict(clouds=True, cloud_conv_boost=cb)))
    return grid
