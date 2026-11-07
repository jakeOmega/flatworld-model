"""
flatworld.stage_ad_core -- the fused dimensional radiative-convective-DYNAMICAL core (M2b).

This is the multi-session "unified core" of the M2/M3 roadmap (``archive/UNIFIED_CORE_ROADMAP.md``
and the session split in ``archive/M2B_SUBTASKS.md``): a single dimensional 2-D ``(x, z)`` **anelastic**
radiative-convective-dynamical solver that subsumes Stage A (the cross-track RCE-EBM backbone) AND
Stage B (the f=0 overturning) into ONE domain, so the transport is RESOLVED rather than parameterized
by a free ``kappa_atm``. It exists to (a) dissolve Issue A by building one fused domain whose only
energy export is OLR (no flux injection between a closed cell and an open sponge), and (b) set up the
M3 pinning test (does the ribbon T(theta) curve go insensitive to the turbulence closure?).

THIS FILE WAS BUILT IN STAGES (M2b.1 .. M2b.6 -- ALL COMPLETE, 2026-06-04; the per-stage record is
``archive/M2B_SUBTASKS.md`` + ``archive/M2B{2..6}_RESULTS.md``): the geometry/grids/anelastic reference
state + run-loop harness (M2b.1), the anelastic omega-psi dynamics + buoyancy torque + Smagorinsky
closure with the Nu~Ra^beta benchmark (M2b.2), radiation-in-the-loop + moist convective adjustment +
prognostic moisture with the independent-RCE anchor (M2b.3), the prognostic surface energy balance +
boundary-layer / surface-flux physics with the bulk air-sea SST and radiative-subsidence (`w_sub`)
cross-checks (M2b.4), the tier-1 conservation gates (energy with NO export term, anelastic mass,
cumulative water) + the resolved VERTICAL moisture flux + the f=0 wind-driven ocean cubic (M2b.5), and
the steady run + the resolved HEAT transport + the Stage-C interface contract + the lineage regression
(M2b.6). Since 2026-06-09 it is the DEFAULT cascade backbone (``cascade.run_cascade(core="ad")``), with
the later physics ON by default (radiatively-active clouds, LW-window OLR cap, two-band + diffuse
shortwave, RH-detrain convection, dissipative heating, fluxlim heat advection, vapor buoyancy, the
collimator geometry) -- each knob still toggles OFF to its byte-identical limit.
The `step()` operator-splits the dynamics (M2b.2) and the column+surface thermodynamics (M2b.3/4),
each gated by a switch (``run_dynamics`` / ``couple_thermo`` / ``couple_surface``) the tests flip to
isolate limbs. The resolved flow advects both MOISTURE (in x and z) AND HEAT, so the inter-column
transport is resolved with no free ``kappa_atm`` and Issue A is structurally dissolved (see WHY
below); whether T(theta) goes closure-insensitive (the pin) is the still-open M3 study.

================================  ARCHITECTURE DECISIONS (#0 - #5)  ================================
These are settled HERE, in M2b.1, and bind every later subtask (from ``archive/UNIFIED_CORE_ROADMAP.md``
and ``archive/M2B_SUBTASKS.md`` "Architecture", made concrete against the existing modules).

  #0  ANELASTIC, not Boussinesq.  The cross-domain temperature span (~180 K, hot inner sea to cold
      edge) strains constant density, so the reference state is a variable density ``rho_ref(z)`` (a
      hydrostatically-balanced, horizontally-uniform base state; see ``_reference_profile``). The
      streamfunction enforces ANELASTIC mass continuity div(rho_ref * u) = 0, i.e.
          rho_ref * u = +d psi / dz ,    rho_ref * w = -d psi / dx ,
      so Stage B's constant-density Poisson pattern (``stage_b_overturning._build_poisson`` -> splu)
      generalizes to the rho_ref-weighted elliptic operator (built in M2b.2).

  #1  DUAL VERTICAL COORDINATE.  Two co-existing vertical grids, reconciled every step:
        * the PRESSURE grid (column thermodynamics) -- ``p_edge`` (nz+1, TOP->SURFACE), ``p_mid``,
          ``dm``, ``sigma``, ``RH(sigma)`` -- carries ``T_atm`` and runs radiation + convective
          adjustment (reuse ``radiation.column_radiation``; the convective adjustment is this
          module's own enthalpy-conserving ``moist_convective_adjust``).
        * the HEIGHT grid (dynamics) -- a grid UNIFORM IN z (so Stage B's uniform-Cartesian stencils
          are valid) -- carries the dynamics (omega, psi).
      They are reconciled by ``_hydrostatic_z`` (lifted from Stage A): the per-column physical heights
      of the pressure layers, from the live temperature (``_pressure_to_height``/``_height_to_pressure``
      do the field interpolation between them). The cross-track grid is UNIFORM IN PHYSICAL DISTANCE x
      (x = (arange(nx)+0.5)*x_max/nx, x_max = g.x_of_theta(theta_max)), NEVER uniform in theta
      (theta = arctan(x/L) is strongly nonlinear; the f=0 flat-plane metric).
      INDEX-ORDER CONVENTION (a real gotcha -- settled here): the pressure-grid vertical axis is the
      LAST axis, ordered TOP(0)->SURFACE(nz-1) [Stage A]; the height-grid vertical axis is the FIRST
      axis, ordered SURFACE(0)->TOP(nz-1) [Stage B]. So T_atm is (nx, nz); omega/psi are (nz, nx).
      M2b.3 REFINEMENT -- prognostic moisture q is on the PRESSURE grid (nx, nz), co-located with T_atm,
      NOT on the height grid as originally sketched. Its master grid is where its physics is exact and
      energy-conserving: evaporation/condensation/radiation/qsat all use (T_atm, p_mid) directly, and the
      latent heat of condensation is released to T on the same grid (no cross-grid remap). It is mapped to
      the height grid ONLY for advection by the resolved flow, as a one-way INCREMENT (``_advect_moisture``)
      -- so q never round-trips and accrues no spurious round-trip diffusion. (Advection-master-on-height
      would force that round-trip every step and make the latent-heat remap non-conservative.)

  #2  REAL BUOYANCY from ABSOLUTE temperature.  b = g * alpha * (T - T_ref(z)) with alpha = 1/T_ref(z)
      (ideal gas), i.e. b = g (T - T_ref)/T_ref; the stratification N^2 = (g/T_ref) dtheta_ref/dz is
      DIAGNOSED from the state, never prescribed. Consequently ``kappa_atm`` and the nondimensional
      ``Ra`` of Stages A/B both DROP OUT -- the transport is resolved, and Ra becomes a *diagnostic*
      (contrast Stage B's prescribed Ra=60, kappa_atm=2e6). (Buoyancy torque enters in M2b.2.)

  #3  RADIATION IN THE LOOP.  ``radiation.column_radiation`` is called on the LIVE column state every
      step (M2b.3) -- NOT a Newtonian relaxation toward Stage A's columns, which would smuggle the
      radiative time scale tau_rad back in as a verdict-affecting free knob.

  #4  CLOSURE: state-dependent SMAGORINSKY.  nu_t = (C_s * Delta)^2 |S| (Delta the grid scale, |S| the
      strain-rate magnitude) makes the eddy viscosity/diffusivity STATE-dependent, with a small
      molecular-like floor (``nu_b``, ``kappa_b``). This replaces Stage B's constant nondim nu/kappa_b
      and is the knob M3 perturbs (C_s, kappa_b +-2x) while refining resolution. (Active in M2b.2.)

  #5  ENERGY BOUNDARY CONDITIONS (the Issue-A dissolver).  The poleward wall has psi = 0 => u = 0 =>
      the advective energy export across it is EXACTLY zero; ALL export to space is OLR. The equator
      x=0 is a zero-flux symmetry edge, psi = 0. There is NO sponge export term (Stage A's bounded
      relaxation clip is kept, if at all, ONLY as a numerical spin-up limiter, never as an export).
      The cold-trap / cold-edge temperature is an OUTPUT, never imposed (imposing it would re-pin the
      export and manufacture a spurious closure).

WHY THE FUSED DOMAIN DISSOLVES ISSUE A: Issue A could not close because Stage B's *closed* cell exports
zero across its wall while Stage A's *open* radiating sponge needs a delivered surplus to radiate, so
injecting B's flux divergence into A starved the sponge and froze the ribbon. With ONE domain whose
only export is OLR and whose walls are psi=0, there is no injection and no closed-vs-open mismatch --
the surplus radiates through the columns' own OLR. It DOES close to a physical steady state (the M2b.6
steady run; ``archive/M2B6_RESULTS.md``) -- but the dissolution is consistency, not correctness (the
no-export energy closure is structurally forced); the closure-insensitivity pin is the M3 study.

VALIDATION STANCE (external first; see ``archive/UNIFIED_CORE_ROADMAP.md`` "Validation"): gate against EXTERNAL
anchors, not the doc's own A/B/C lineage (which is wrong where it matters). Tier 1 conservation laws
(energy/mass/water, pure theorems); tier 2 analytic limits (single-cell from Rossby->inf + continuity,
the f=0 ocean cubic, an independently-derived RCE temperature); tier 3 in-solver benchmarks (Nu~Ra^beta
vs the literature exponent, a standard RCE); tier 4 cross-method agreement (subsidence rate, bulk SST).
The no-dynamics (psi==0) == Stage-A(kappa_atm=0) check is run ONLY as a regression test against a
suspect baseline -- necessary (catches code divergence), not sufficient (a shared bug passes green).

MODULE SPLIT (#289 Part 2, 2026-07-07, move-only): the module-level helpers this file grew are now
``stage_ad_kernels.py`` (grid/stencils/Poisson/Smagorinsky/tridiag/interp), ``stage_ad_column.py``
(wetness/albedo, convective adjustment, microphysics kernels, single-column RCE, bulk SST) and
``stage_ad_config.py`` (``StageADConfig`` + the ``_env_*`` knob helpers). THIS module keeps
``StageADCore`` and the Rayleigh-Benard harness, and RE-EXPORTS every moved name (public and
underscore) so every existing ``stage_ad_core.X`` import and monkeypatch target keeps working; the
three lazily-REBOUND numba kernel caches are forwarded live via module ``__getattr__`` (see the
block after the imports). New code should import moved names from their new homes.
"""
from __future__ import annotations
import os
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.fft import dst, idst, set_workers
from scipy.optimize import brentq
from dataclasses import dataclass, field
from . import constants as C
from . import geometry as g
from . import thermo as th
from . import radiation as rad
from . import bl_cloud as blc                                 # Chunk-4 BL/stratocumulus closure
from . import convergence as conv
from . import _ckpt_fingerprint as _ckfp                       # checkpoint config/world fingerprint (resume guard)
from . import conv_min_scale as _cms                           # PHY-19 (b): the min-horizontal-scale operator
from .device import get_xp, to_host, to_device, array_module, is_device   # M3.1: numpy/CuPy dispatch (optional GPU)
from .stage_b_overturning import wind_driven_ocean_profile   # M2b.5: the f=0 ocean cubic (analytic anchor)


# ======================================================================================
# #289 PART-2 MODULE SPLIT -- the moved helpers, re-imported here both FOR THE CORE'S OWN USE
# (the class calls them as bare names, so they must live in this module's namespace -- which is
# also exactly what keeps ``monkeypatch.setattr(stage_ad_core, "X", ...)`` working for names the
# core consumes) and AS THE BACK-COMPAT RE-EXPORT SURFACE for the 20+ external importers of
# ``flatworld.stage_ad_core``. EVERY moved top-level name appears here (public and underscore);
# ``noqa: F401`` -- several are re-exports only, never used by the core itself.
# ======================================================================================
from . import stage_ad_kernels as _sad_kernels
from . import stage_ad_column as _sad_column
from .stage_ad_kernels import (  # noqa: F401
    _stretched_z, _stretched_x_edges, stretched_x_cell_count,
    _ddx, _ddz, _upwind, _IMPLICIT_ADVECTION_SCHEMES, _STRANG_SCHEMES,
    _implicit_advection_scheme, _upwind_vertical, _div_grad, _div_grad_z_arm, _div_grad_aniso,
    build_anelastic_poisson, SeparablePoisson, StretchedSeparablePoisson, velocities_from_psi,
    strain_rate_magnitude,
    _mixing_dz, smagorinsky_viscosity, smagorinsky_viscosity_aniso, vorticity_tendency,
    _vertical_eddy_term, _solve_tridiag_columns, hydrostatic_height,
    _interp_columns, _interp_columns_pair, _build_interp_numba, _interp_columns_parallel,
)
from .stage_ad_column import (  # noqa: F401
    wetness_of_Tsurf, ALOFT_SIGMA_THRESH, surface_albedo,
    moist_convective_adjust, _build_conv_numba_kernel, moist_convective_adjust_parallel,
    moist_convective_segment_mix,
    _build_micro_numba_kernel, _MICRO_SCRATCH, _micro_fuse_salt_cond_params,
    prognostic_microphysics_reevap_parallel, single_column_rce, bulk_air_sea_sst,
    radiative_subsidence_rate,
)
from .stage_ad_config import (  # noqa: F401
    _env_flag, _env_float, _env_int, _env_str, StageADConfig,
)

# The three numba-kernel caches are REBOUND (via ``global``) inside their new home modules on
# first ON-path use, so a ``from``-import here would freeze the import-time ``None`` snapshot and
# break every ``stage_ad_core._MICRO_NUMBA_KERNEL is not None`` liveness check (the cond-tau and
# micro-parity tests read them). Reads are forwarded dynamically instead: module ``__getattr__``
# is consulted exactly when normal attribute lookup fails -- i.e. for these three names, which are
# deliberately NOT from-imported above -- and returns the CURRENT binding in the defining module.
# (``_MICRO_SCRATCH`` is mutated in place, never rebound, so its from-import above stays live.
# The residual core never reads these three as bare names, so nothing here needs them at
# module-global scope.)
_LIVE_CACHE_HOMES = {"_INTERP_NUMBA": _sad_kernels,
                     "_CONV_NUMBA_KERNEL": _sad_column,
                     "_MICRO_NUMBA_KERNEL": _sad_column}


def __getattr__(name):
    home = _LIVE_CACHE_HOMES.get(name)
    if home is not None:
        return getattr(home, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# #302 surf_water_budget: the W snap-to-zero floor [kg/m^2]. For W << W_min the wetness ramp makes
# E proportional to W, so a drying column decays EXPONENTIALLY and never reaches 0 -- the creep chain
# then carries a denormal tail onto far dry cells (min W ~ 2.7e-318 in the GPU capture probe):
# harmless in fp64 arithmetic, but a latent CPU/GPU denormal-flush divergence for byte-parity gates.
# W below this floor (dimensionally absurd: 1e-30 kg/m^2 ~ a million-th of a proton per m^2) snaps to
# exactly 0.0, with the removed mass counted into ``cum_w_snap`` so the machine-precision surface
# closure stays exact. A numerical-hygiene constant, not a physics knob.
W_SNAP_FLOOR = 1.0e-30
# The SALT twin (run6 empirical finding, 2026-07-12): the "S needs no snap floor" argument held only
# for the direct dry-out path (W=0 donor moves no salt; dry-cell salt is a normal-range crust). It
# missed the CREEP TAIL: a cell whose W rides just above the floor receives salt at the donor's
# salinity as a product of two tiny numbers, so S_dis decays into denormals (observed 1e-143..1e-209),
# and the saturation adjustment then converts them into DENORMAL CRUST (S_dep 1e-222..1e-311) that is
# immobile and never re-dissolves at W=0 -- the same latent CPU/GPU denormal-flush parity hazard
# W_SNAP_FLOOR exists to kill, plus log-axis garbage on the dashboard. Snapped salt is counted into
# ``cum_s_snap`` so the machine-precision salt_residual closure stays exact.
S_SNAP_FLOOR = W_SNAP_FLOOR

# The saturation column [mm] below which the REPORTED aloft relative humidity ``RH_aloft`` is masked to
# NaN (``_aloft_thermo``). A reporting-hygiene constant, not a physics knob: nothing in the model reads
# ``RH_aloft`` -- ``Wsat_aloft`` and the aloft vapour column, which ARE read, are untouched by it.
# WHY: in the far cold field the aloft column reaches 50-200 K and its saturation capacity collapses
# through 1e-13 mm to ~1e-49 mm, while the vapour left there by advection is a round-off residue. The
# ratio of two such numbers is not a relative humidity: measured on the canonical run17 state it reads
# up to 2.0e4 (~2e6 %) where the residue survives and exactly 0.0 where the residue underflows too --
# both fabricated. NaN says "no meaningful ratio here"; the reader takes the numerator and denominator,
# which ``diagnostics()`` exports separately as ``aloft_moisture_mm`` / ``Wsat_aloft_mm``.
# THE VALUE IS A TASTE CALL, stated: 1e-6 mm is one nanometre of liquid water over the WHOLE column --
# monolayer scale, orders below any meteorologically meaningful amount, and six decades above the
# ``1e-12`` divide-by-zero guard in the ratio itself. It is not delicate: measured on run17 every floor
# from 1e-12 to 1e-3 mm masks the SAME 59 of 975 columns, because ``Wsat_aloft`` jumps from 1.3e-3 mm at
# the last honest column (theta 85.58) to 7.5e-14 mm at the first degenerate one (theta 85.87) -- the
# same wide-empty-gap argument the S_dep detection ceiling above rests on.
RH_ALOFT_WSAT_MIN_MM = 1.0e-6

# Deposited-halite crust S_dep [kg/m^2] guard constants (used ONLY as a runaway guard, never in the
# physics). The DETECTION ceiling and the REPAIR target are deliberately DECOUPLED:
#   * SURF_SALT_SDEP_SATURATION -- the emergent evaporite-crust saturation (~1 km of halite; the model
#     saturates at ~1.97e6, GRAV_FLOW_RESULTS, and ~3.08e6 in the fixed offline eq_cycle). A runaway
#     cell is RESET to this physical equilibrium, so a healed dead salt-flat cell restarts at the crust
#     it would actually carry -- not at an arbitrary bounded-but-unphysical ceiling.
#   * SURF_SALT_SDEP_PHYS_MAX -- the runaway DETECTION ceiling (~46 km of halite): geologically
#     impossible (real evaporite beds are <~1 km / ~2e6), ~50x the saturation, and ~1e9x below the
#     numerical runaway (>=1e17) the surf-salt mass-creation bug produced. Kept high on purpose so the
#     guard NEVER false-flags a legitimately thick crust; there is a wide empty gap (~1e7..~1e14)
#     between any real crust and the runaways, so the exact detection value is not delicate.
# A checkpoint or step whose S_dep exceeds the ceiling is corrupt; ``repair_runaway_crust`` resets it
# to the saturation scale and accounts the removed spurious salt.
SURF_SALT_SDEP_SATURATION = 2.0e6
SURF_SALT_SDEP_PHYS_MAX = 1.0e8


def repair_runaway_crust(S_dep, dx_w, detect_max, repair_to, xp):
    """Reset physically-impossible deposited-halite crust (a numerical runaway, ``S_dep > detect_max``)
    to ``repair_to`` (the evaporite-saturation scale) -- NOT to the detection ceiling, so a healed dead
    salt-flat cell restarts at its physical crust equilibrium.

    Returns ``(S_dep_repaired, removed_kg_per_m, n_clamped)``: the repaired field is ``repair_to`` on
    runaway cells and ``S_dep`` elsewhere; ``removed_kg_per_m = sum((S_dep - repair_to) * runaway * dx_w)``
    is the area-integrated spurious salt removed [kg/m] (to re-baseline / account the salt budget); and
    ``n_clamped`` is the number of cells reset. A no-op (byte-identical, removed 0, n 0) whenever all
    ``S_dep <= detect_max`` -- safe to run unconditionally on every checkpoint load. Pure; works on host
    or device arrays via ``xp``. Does NOT mutate ``S_dep``."""
    runaway = S_dep > detect_max
    repaired = xp.where(runaway, repair_to, S_dep)      # exact reset (no S_dep-repair_to ULP loss at 1e24)
    removed = float(xp.sum((S_dep - repaired) * dx_w))  # (S_dep - repair_to) on runaway cells, 0 elsewhere
    n = int(xp.sum(runaway))
    return repaired, removed, n


def surface_water_salt_step(st, cfg, dt, xp):
    """Prognostic surface-water/salt substep (the #302 budget), extracted VERBATIM from
    ``StageADCore._thermo_substep_impl`` so the core substep and the offline equilibrium
    integrator (``flatworld.eq_cycle``) share ONE implementation. Mutates ``st`` in place
    (W_surf/S_dis/S_dep, the working buffers, and the cum-trackers) exactly as the inline
    block did; in-core ``st is self`` so the behaviour is byte-identical.

    ``st`` must expose the attributes the block reads/writes (an ``StageADCore`` does; an offline
    shim must supply them):
      state (mutated): ``W_surf``, ``S_dis``, ``S_dep``  (fp64 ``(nx,)``)
      per-substep inputs (read): ``T_surf``, ``_precip``, ``_evap``, ``_surf_flash``, ``_u_surf``
        (or ``None``), ``_W_ucreep_fallback``, ``run_dynamics``
      working buffers (in-place scratch): ``_p_land``, ``_W_flux``, ``_S_flux``
      grid: ``dx``, ``_dx_cell`` (or ``None``), ``_dx_w``, ``_dx_min``, ``_dx_last``
      accounting (mutated): the six ``_cum_{w,s}_{snap,cryo,edge}`` and, on the capture branch
        (``cfg.capture_ready``), their ``_pending`` twins + the ``_accum_pending`` helper.
    ``cfg`` supplies the ``surf_*`` knobs; ``xp`` is the array module (numpy on host).
    """
    self = st   # keep the extracted body verbatim (attribute access + in-place mutation)
    c = cfg
    # --- #302 PROGNOSTIC SURFACE-WATER BUDGET update:  dW = dt*(P_land - E) + dt*div(creep).
    #     Runs HERE so it consumes (a) the LAGGED ``self._precip`` (the previous substep's final
    #     surface flux, post de-spike/nest -- exactly the lag the wetness/flash already run on)
    #     and (b) THIS substep's ``_surf_flash`` (computed from that same lagged flux in the SEB
    #     block above) and capped ``self._evap``. All ops in-place on persistent fp64 buffers
    #     (capture-safe; static flag branch). W stays >= 0 structurally: E <= W/dt (the LE cap),
    #     P_land >= 0 (flash = min(max(P,0), cap) so max(P,0) - flash >= 0), and the transport is
    #     donor-cell upwind under a CFL clip (a convex update). ---
    # 1) landed LIQUID: lagged precip minus the flashed portion (it never wetted anything -- the
    #    same flux split the legacy _wetness supply used; _surf_flash == 0 when that flag is OFF).
    self._p_land[...] = xp.maximum(self._precip, 0.0) - self._surf_flash
    # 2) local source/sink (fp64 accumulator: under thermo_fp32 the increment is fp32-granular,
    #    the running W total never re-rounds -- the SEB-freeze discipline).
    self.W_surf += (self._p_land - self._evap) * dt
    # 2a) NEGATIVE-W GUARD (before transport). In-core W stays >= 0 structurally (E <= W/dt and
    #    p_land = max(P,0) - min(flash, cap) >= 0). But a FROZEN/inconsistent forcing -- notably the
    #    offline eq_cycle, which holds _surf_flash and _precip from separate climatology EMAs so
    #    p_land = max(P,0) - surf_flash can be < 0 -- can drive a near-dry cell's W below zero. A
    #    negative W then POISONS the salt riders below: the salinity S_dis/max(W, W_SNAP_FLOOR) floors
    #    its denominator to 1e-30 (-> ~1e22) while Wcap = min(W, .) goes negative, so the creep/gravity
    #    salt flux (Wcap * salinity) explodes into a huge +/- S_dis dipole whose negative half the salt
    #    snap floor (step 4a, S_dis < FLOOR is TRUE for large negatives) then deletes -- NET salt
    #    mass-creation that crusts into a runaway S_dep (the run11/run12 ~4e24 kg/m^2 blowup). Clamp
    #    W < 0 to exactly 0 HERE, before any transport reads it, accounting the restored mass into
    #    cum_w_snap with the step-4 machinery. Condition is W < 0 (NOT < W_SNAP_FLOOR): positive
    #    denormals are left for the step-4 floor, so this is byte-identical whenever W >= 0 (in-core:
    #    the clamp is a no-op and cum_w_snap += 0).
    _wneg = xp.where(self.W_surf < 0.0, self.W_surf, 0.0)
    xp.subtract(self.W_surf, _wneg, out=self.W_surf)
    _wneg_sum = xp.sum(_wneg * self._dx_w)                    # [kg/m] signed (negative -> restored mass)
    if c.capture_ready:
        self._cum_w_snap_pending = self._accum_pending(self._cum_w_snap_pending, _wneg_sum, xp)
    else:
        self._cum_w_snap += float(_wneg_sum)
    # 3) LATERAL CREEP TRANSPORT (the salt_sill inflow physics, extracted): surface water is
    #    dragged at the wind-drift speed u_creep = drift * u_surf (signed resolved surface wind;
    #    fixed equatorward -drift*cfg.wind when no dynamics), moving at most the wind-driven
    #    layer rho*H_layer per column: F = u_e * min(W_donor, rho*H). At a deep reservoir edge
    #    this IS the salt-sill inflow capacity Q_in = u_current*H_layer (~15 m^2/s per unit
    #    along-track width); over a thin interior film it is the depth-limited creep u*W, so a
    #    wet front advances at a FINITE, derived rate (<= u_creep) -- the #302 physics. Donor-cell
    #    upwind + the CFL clip keep the update positivity-preserving; closed walls (F=0), so the
    #    domain integral telescopes to zero and the global budget closes with NO boundary influx
    #    (the ACCOUNTED exception: the surf_water_edge_source pin, step 5 below).
    #    surf_salt rider: DISSOLVED salt S_dis advects on the same donor-cell fluxes at the
    #    donor's salinity, and the deposited-halite DAM gates u_e -- the flag blocks below.
    if self.run_dynamics and self._u_surf is not None:
        _u_c = c.surf_water_drift * self._u_surf           # (nx,) signed creep [m/s] (fp32-safe: upcasts)
    else:
        _u_c = self._W_ucreep_fallback                     # fixed equatorward creep (persistent buffer)
    _u_e = 0.5 * (_u_c[:-1] + _u_c[1:])                    # interface speeds (nx-1,)
    if c.surf_water_edge_source:
        # OPEN-BOUNDARY creep speed at the pinned ocean cell: the psi=0 wall BC makes the wind
        # in the LAST column structurally zero, so the centered average above halves (and, as
        # the wall column's wind IS exactly 0, effectively throttles) the pin's only export --
        # the box artifact the edge source exists to soften, applied to the wind as well as the
        # mass. One-sided ghost (du/dx = 0 across the open face): the last interface moves at
        # the interior neighbor's creep speed. Inert on the no-dynamics fallback (uniform _u_c).
        _u_e[-1:] = _u_c[-2:-1]
    _u_cfl = 0.25 * self._dx_min / dt                      # CFL clip (never binds in practice:
    #   u~0.1 m/s). Tightest width on the stretched grid (conservative for every donor cell);
    #   == self.dx on the uniform grid (same Python float, byte-identical).
    _u_e = xp.clip(_u_e, -_u_cfl, _u_cfl)
    _Wcap = xp.minimum(self.W_surf, C.RHO_LIQ * c.surf_water_h_layer)   # transportable depth [kg/m^2]
    if c.surf_salt and not c.surf_water_grav_flow:
        # HALITE DAM -- RETIRED when surf_water_grav_flow is ON (step 3g below): crust enters the
        # gravity term's surface height, so ponding behind / overtopping of a crust sill emerges from
        # the eta gradient itself, and wind-creep-over-sill vs gravity-return is a resolved physical
        # competition instead of a binary block. This block is the LEGACY (grav-flow OFF) path,
        # kept verbatim for the byte-identical gate.
        # HALITE DAM (overtopping/weir criterion -- FLOW-DEPENDENT blocking): an interface is
        # blocked, both directions, where the deposited-crust HEIGHT h = S_dep/RHO_HALITE [m] at
        # EITHER neighbor exceeds block_factor * the moving-layer depth _Wcap/RHO_LIQ [m] of the
        # DONOR side (upwind convention, same as the flux below). A deep, vigorous flow overtops
        # (and the saturation adjustment then dissolves the crust it crosses); a shallow creeping
        # film is stopped; a trace crust never dams (h ~ 0). No epsilon and no free mass
        # threshold: RHO_HALITE is a physical constant, block_factor a physically-anchored ratio.
        _h_crust = self.S_dep * (1.0 / C.RHO_HALITE)                     # crust height [m] (nx,)
        _h_dam = xp.maximum(_h_crust[:-1], _h_crust[1:])                 # tallest adjacent crust (nx-1,)
        _h_flow = (xp.where(_u_e > 0.0, _Wcap[:-1], _Wcap[1:])
                   * (c.surf_salt_block_factor / C.RHO_LIQ))             # donor moving depth [m]
        _u_e = xp.where(_h_dam > _h_flow, xp.zeros((), dtype=_u_e.dtype), _u_e)   # dtype-matched zero
    self._W_flux[...] = xp.where(_u_e > 0.0, _u_e * _Wcap[:-1], _u_e * _Wcap[1:])
    if self._dx_cell is None:
        _coef_l = _coef_r = dt / self.dx                   # uniform: one scalar (verbatim math)
    else:
        _coef_l = dt / self._dx_cell[1:]                   # gain side: each cell's OWN width
        _coef_r = dt / self._dx_cell[:-1]                  # loss side (conserves sum(W*dx_cell))
    if c.surf_salt:
        # DISSOLVED-salt advection [kg/m/s]: the same donor-cell water flux at the donor's
        # salinity S_dis/W [kg salt / kg water]. eps = W_SNAP_FLOOR is exact for every live cell
        # (sub-floor W was snapped to exactly 0 last substep, and a W=0 donor has _W_flux == 0,
        # so the eps never becomes transport). Positivity: per interface |dS| <= 0.25 * Wcap_donor
        # * salinity_donor <= 0.25 * S_dis_donor (the CFL clip), so a donor keeps >= half its
        # dissolved salt per substep even losing through both faces. S fluxes are computed from
        # the PRE-update W (the same W the water flux used) and applied before W advances.
        _sal_l = self.S_dis[:-1] / xp.maximum(self.W_surf[:-1], W_SNAP_FLOOR)
        _sal_r = self.S_dis[1:] / xp.maximum(self.W_surf[1:], W_SNAP_FLOOR)
        self._S_flux[...] = xp.where(_u_e > 0.0, self._W_flux * _sal_l, self._W_flux * _sal_r)
        self.S_dis[1:] += self._S_flux * _coef_l           # gain across the left interface
        self.S_dis[:-1] -= self._S_flux * _coef_r          # loss across the right interface
    self.W_surf[1:] += self._W_flux * _coef_l              # gain across the left interface
    self.W_surf[:-1] -= self._W_flux * _coef_r             # loss across the right interface
    # 3g) GRAVITY LEVELING FLOW (surf_water_grav_flow): a second interface flux, driven by the
    #    surface-height gradient the wind creep is blind to (without it NOTHING knows water is deeper
    #    in one cell than its neighbor: deep pools stand beside starved cells, water piles without
    #    bound at wind stagnation points, the reservoir edge never spreads -- the run10/run12
    #    findings). Height eta = W/RHO_LIQ + S_dep/RHO_HALITE [m]: CRUST IS TOPOGRAPHY, so ponding
    #    behind (and overtopping of) a crust sill emerges from the gradient -- this is what retires
    #    the binary halite dam above. PERCOLATION CUTOFF: only water above W_conn = conn_frac*wmin
    #    is laterally connected (below it the cell is disconnected sub-grid pools; wet_frac < 1 is a
    #    WETTED-AREA fraction, not a depth) -- deliberately NO thin-film hydraulics; the free-depth
    #    form also parks a draining cell AT W_conn instead of flip-flopping across a binary gate.
    #    FLOW LAW (crude by design): donor = higher-eta side; speed u_g = min(sqrt(g*head), u_cfl)
    #    (Froude~1 gravity-current/dam-break scaling, no Manning friction -- the RATE is a named-wall
    #    soft magnitude, the leveled equilibrium eta is law-independent); depth h_eff = min(donor
    #    free depth, head/2). u_g*dt <= 0.25*dx and h_eff <= head/2 make each face's transfer shave
    #    at most head/8 off its own head -- a PER-FACE no-overshoot guarantee (locally-monotone
    #    profiles level without crossing; a local extremum flanked by much larger heads can
    #    transiently reorder via its neighbors' faces, bounded + conservative, and still levels to
    #    the same equilibrium). Donor-cell fluxes on the
    #    same telescoping _coef_l/_coef_r as the creep => conserves sum(W*dx) and total salt to
    #    round-off (internal transport, no new cum-tracker); S_dis rides at the donor's salinity
    #    (the creep rider idiom, eps exact for live cells for the same snap-floor reason).
    #    Elementwise temporaries on fixed shapes (the saturation-adjustment idiom), capture-safe.
    if c.surf_water_grav_flow:
        _eta = self.W_surf * (1.0 / C.RHO_LIQ)                  # water height [m] = [kg/m^2]/[kg/m^3]
        if c.surf_salt:
            _eta = _eta + self.S_dep * (1.0 / C.RHO_HALITE)     # + immobile crust topography [m]
        _dh = _eta[:-1] - _eta[1:]                              # signed head [m]; > 0 -> flow in +x
        _head = xp.abs(_dh)                                     # [m]
        _u_g = xp.minimum(xp.sqrt(C.G_ACCEL * _head), _u_cfl)   # [m/s] = sqrt([m/s^2][m]); CFL-clipped
        _W_conn = c.surf_water_conn_frac * c.surf_water_wmin    # percolation threshold [kg/m^2]
        _W_free = xp.maximum(self.W_surf - _W_conn, 0.0)        # connected (mobile) water [kg/m^2]
        _h_free = xp.where(_dh > 0.0, _W_free[:-1], _W_free[1:]) * (1.0 / C.RHO_LIQ)  # donor depth [m]
        _h_eff = xp.minimum(_h_free, 0.5 * _head)               # monotone-leveling depth [m]
        _F_g = xp.sign(_dh) * (_u_g * _h_eff * C.RHO_LIQ)       # water flux [kg/m/s] = [m/s][m][kg/m^3]
        if c.surf_salt:
            _sal_gl = self.S_dis[:-1] / xp.maximum(self.W_surf[:-1], W_SNAP_FLOOR)
            _sal_gr = self.S_dis[1:] / xp.maximum(self.W_surf[1:], W_SNAP_FLOOR)
            _S_g = xp.where(_F_g > 0.0, _F_g * _sal_gl, _F_g * _sal_gr)   # donor-salinity [kg/m/s]
            self.S_dis[1:] += _S_g * _coef_l
            self.S_dis[:-1] -= _S_g * _coef_r
        self.W_surf[1:] += _F_g * _coef_l
        self.W_surf[:-1] -= _F_g * _coef_r
    # 3e) DISSOLVED-SALT LATERAL LOCK EXCHANGE (salt_lock_exchange): the baroclinic BRINE RETURN -- a
    #    third dissolved-salt rider, but one that moves SALT with ZERO NET WATER. Where two CONNECTED
    #    water columns differ in salinity the denser (saltier) water flows one way underneath while the
    #    fresher returns over the top: an equal-volume two-layer gravity exchange, net water flux ~0,
    #    net salt flux nonzero. This is the salt-return path the creep + gravity riders LACK (both carry
    #    salt only WITH a net water flux), so without it an evaporating margin over-concentrates brine
    #    and over-crusts halite with no way to relax the salinity contrast against its neighbours
    #    (OPEN_ISSUES "geologically slow salt return"; salt_sill.lock_exchange_flux is the analytic half).
    #    ONE salinity computation feeds BOTH the buoyancy driver g' and the flux magnitude: the volumetric
    #    concentration C == RHO_LIQ*sal identically (dilute linear EOS), so g' is driven by |dsal| and the
    #    flux magnitude is written RHO_LIQ*|dsal| -- never a second, independently-floored C expression
    #    (which could flip sign(dC) != sign(dsal) in a near-dry cell into an anti-diffusive up-gradient
    #    flux). Donor-cell-free, gradient-driven; antisymmetric on the SAME _coef_l/_coef_r telescoping as
    #    the creep => conserves total salt to round-off (internal transport, no cum-tracker). The u_cfl clip
    #    alone bounds each face's donor removal at <= 0.125*S_dis_donor (H_int < W_donor/RHO_LIQ and
    #    |dC| <= C_donor), so S_dis stays > 0; the explicit donor cap below is defense-in-depth and scales
    #    the SINGLE F on both sides (never the two faces independently -- that would break antisymmetry).
    #    H_int uses CONNECTED water only (the surf_water_conn_frac percolation cutoff): a disconnected
    #    film/pool on either side hosts no counter-flow -> H_int = 0, no exchange (physical scoping).
    #    Equilibrium (F -> 0) is equal salinity = equal density (the volume-weighted mean). Elementwise
    #    local temporaries on fixed shapes (the gravity-block idiom), static flag branch => capture-safe.
    #    Named walls: the linear EOS (drho ~ 0.75*rho*ds; equal-salinity == equal-density) and the crude
    #    Froude-scaled RATE (surf_salt_lock_ce the single free efficiency). NOTE: this term ignores crust --
    #    on the LEGACY grav-flow-OFF path the halite dam blocks the creep across a tall-crust interface but
    #    NOT this exchange (consequence-free in canon: production runs grav_flow ON, dam retired; OPEN_ISSUES).
    if c.surf_salt and c.salt_lock_exchange:
        _W_conn_x = c.surf_water_conn_frac * c.surf_water_wmin              # percolation threshold [kg/m^2]
        _sal_lx = self.S_dis[:-1] / xp.maximum(self.W_surf[:-1], W_SNAP_FLOOR)   # donor-free salinity [kg/kg]
        _sal_rx = self.S_dis[1:] / xp.maximum(self.W_surf[1:], W_SNAP_FLOOR)
        _dsal = _sal_lx - _sal_rx                                          # signed salinity contrast [kg/kg]
        _gp = C.G_ACCEL * (0.75 * xp.abs(_dsal))                           # reduced gravity g' [m/s^2]
        _h_conn = xp.maximum(self.W_surf - _W_conn_x, 0.0) * (1.0 / C.RHO_LIQ)   # connected depth [m]
        _H_int = xp.minimum(_h_conn[:-1], _h_conn[1:])                     # shared exchange depth [m]
        _u_ex = xp.minimum(c.surf_salt_lock_ce * xp.sqrt(_gp * _H_int), _u_cfl)  # exchange speed [m/s], CFL-clipped
        # line flux F [kg/m/s] = u_e [m/s] * (H_int/2) [m] * |dC| [kg/m^3]; |dC| == RHO_LIQ*|dsal|.
        _F_x = xp.sign(_dsal) * (_u_ex * (0.5 * _H_int) * (C.RHO_LIQ * xp.abs(_dsal)))
        # donor cap (defense-in-depth): limit the high-salinity donor's per-face loss to 0.25*S_dis_donor;
        # scale the SINGLE F (donor side picked by sign(F)) so +F*_coef_l / -F*_coef_r stay a matched pair.
        _S_don = xp.where(_F_x > 0.0, self.S_dis[:-1], self.S_dis[1:])     # donor dissolved salt [kg/m^2]
        _coef_don = xp.where(_F_x > 0.0, _coef_r, _coef_l)                 # donor loss coefficient [s/m]
        _F_x = xp.sign(_F_x) * xp.minimum(xp.abs(_F_x), 0.25 * _S_don / _coef_don)
        self.S_dis[1:] += _F_x * _coef_l                                   # low-salinity side gains
        self.S_dis[:-1] -= _F_x * _coef_r                                  # high-salinity side loses (antisymmetric)
    # 3b) SATURATION ADJUSTMENT (surf_salt): instantaneous precipitate/dissolve exchange between
    #    dissolved salt and the halite crust. Dissolved capacity = surf_salt_saturation * W
    #    [kg/m^2]; the excess of S_dis precipitates into S_dep, any deficit re-dissolves crust up
    #    to min(deficit, S_dep). So evaporation (which moves NO salt -- E debits W alone, step 2)
    #    concentrates brine to saturation and then crusts it over, and rain / arriving ocean
    #    water re-dissolves the crust; both directions emerge, no extra knobs. Conserves
    #    S_dis + S_dep to round-off (a single transfer, subtracted from one field and added to
    #    the other). At W = 0 the capacity is 0 -> ALL salt sits as crust. (The old "S needs no
    #    snap floor" claim here was WRONG for the creep tail -- see the S_SNAP_FLOOR module note
    #    and step 4a below.) Elementwise, in-place, capture-safe. KINETICS are a named
    #    wall: instantaneous equilibrium (a dissolution rate would add a free timescale).
    if c.surf_salt:
        _cap = c.surf_salt_saturation * self.W_surf                       # dissolved capacity [kg/m^2]
        _xfer = (xp.maximum(self.S_dis - _cap, 0.0)                       # precipitation (>0) ...
                 - xp.minimum(xp.maximum(_cap - self.S_dis, 0.0), self.S_dep))  # ... or dissolution (<0)
        self.S_dis -= _xfer
        self.S_dep += _xfer
    # 4) SNAP-TO-ZERO FLOOR (module constant W_SNAP_FLOOR; review + GPU-probe denormal finding):
    #    E ~ W below W_min => a drying column decays exponentially and never reaches 0, and the
    #    creep chain then smears a DENORMAL tail (W ~ 1e-318) across dry cells -- a latent CPU/GPU
    #    denormal-flush divergence for byte-parity gates. Snap sub-floor W to exactly 0.0 and count
    #    the removed mass into cum_w_snap [kg/m] so the machine-precision closure stays exact
    #    (surf_supply subtracts it). In-place, capture-safe; W >= 0 so ``snapped`` >= 0 and
    #    W - snapped lands snapped cells at exactly 0.0.
    _snapped = xp.where(self.W_surf < W_SNAP_FLOOR, self.W_surf, 0.0)
    xp.subtract(self.W_surf, _snapped, out=self.W_surf)
    _snap_sum = xp.sum(_snapped * self._dx_w)                 # [kg/m] -- already an area total (no dt)
    if c.capture_ready:
        self._cum_w_snap_pending = self._accum_pending(self._cum_w_snap_pending, _snap_sum, xp)
    else:
        self._cum_w_snap += float(_snap_sum)
    # 4a) SALT SNAP FLOOR (S_SNAP_FLOOR; the W idiom, verbatim, on both salt fields): the creep
    #    chain smears DENORMAL S_dis tails across near-dry cells (donor-salinity x tiny flux),
    #    and the saturation adjustment then converts them into immobile denormal CRUST -- the
    #    same CPU/GPU denormal-flush parity hazard the W floor kills (run6: S_dis 1e-143..1e-209,
    #    S_dep 1e-222..1e-311 at nominal day 380). Snap both to exactly 0.0; count the removed
    #    salt into cum_s_snap so the machine-precision salt_residual closure stays exact.
    if c.surf_salt:
        _s_snapped = xp.where(self.S_dis < S_SNAP_FLOOR, self.S_dis, 0.0)
        xp.subtract(self.S_dis, _s_snapped, out=self.S_dis)
        _sd_snapped = xp.where(self.S_dep < S_SNAP_FLOOR, self.S_dep, 0.0)
        xp.subtract(self.S_dep, _sd_snapped, out=self.S_dep)
        _s_snap_sum = xp.sum((_s_snapped + _sd_snapped) * self._dx_w)   # [kg/m] area total
        if c.capture_ready:
            self._cum_s_snap_pending = self._accum_pending(self._cum_s_snap_pending, _s_snap_sum, xp)
        else:
            self._cum_s_snap += float(_s_snap_sum)
    # 4b) CRYO WETNESS PIN (surf_water_cryo_pin): below-freezing ground is ice/frost-covered,
    #    not dry salt flat -- the cold trap's ice cover, which a ~100-day window can never grow
    #    from its own rain, PRESCRIBED and keyed on the physical freeze threshold rather than a
    #    chosen coastline x. FLOOR W (a one-way top-up, never a removal) at W_cryo wherever
    #    T_surf < T_freeze: wetness = clip(W/wmin) then reads 1 there and the surface
    #    evaporates (sublimates) at its q_sat(T_surf)-limited rate -- tiny below 0 C, which is
    #    the physics, not a throttle. The surf_salt rider gives the topped-up water ocean
    #    salinity (salinity0), kept under the dissolved capacity 0.35*W by construction
    #    (0.035*dW <= 0.035*W_cryo << 0.35*W_cryo), so no spurious crusting. The top-up is
    #    ACCOUNTED into cum_w_cryo / cum_s_cryo (the snap/edge _accum_pending idiom) so the
    #    machine-precision surf_residual / salt_residual gates close. Runs AFTER the snap floor
    #    (the top-up must not be snapped) and BEFORE the edge pin (the wall cell's Dirichlet
    #    ocean wins over the floor). A retreating freeze line leaves only ~W_cryo (~10 mm)
    #    behind, drying in days -- no melt-pulse ratchet; the deep reservoir stays the edge
    #    pin. Elementwise, in-place on persistent fp64 state, capture-safe (0-d device sums
    #    through _accum_pending; fixed shapes, static flag branch).
    if c.surf_water_cryo_pin:
        _w_cryo = (c.surf_water_cryo_w if c.surf_water_cryo_w > 0.0
                   else c.surf_water_wmin)                 # [kg/m^2] (<=0 -> "= wmin", the doc'd default)
        _topup = xp.where(self.T_surf < c.surf_water_cryo_tfreeze,
                          xp.maximum(_w_cryo - self.W_surf, 0.0),
                          0.0)                             # [kg/m^2] >= 0 (fp64: W_surf is fp64 always)
        self.W_surf += _topup
        _cryo_sum = (xp.sum(_topup) * self.dx if self._dx_cell is None
                     else xp.sum(_topup * self._dx_cell))  # [kg/m] -- area total (no dt), like the snap
        if c.capture_ready:
            self._cum_w_cryo_pending = self._accum_pending(self._cum_w_cryo_pending, _cryo_sum, xp)
        else:
            self._cum_w_cryo += float(_cryo_sum)
        if c.surf_salt:
            _s_topup = _topup * c.surf_salt_salinity0      # ocean-salinity salt riding the top-up [kg/m^2]
            self.S_dis += _s_topup
            _s_cryo_sum = (xp.sum(_s_topup) * self.dx if self._dx_cell is None
                           else xp.sum(_s_topup * self._dx_cell))   # [kg/m]
            if c.capture_ready:
                self._cum_s_cryo_pending = self._accum_pending(self._cum_s_cryo_pending, _s_cryo_sum, xp)
            else:
                self._cum_s_cryo += float(_s_cryo_sum)
    # 5) OFF-GRID OCEAN SUPPLY (surf_water_edge_source): Dirichlet pin of the LAST cell (index
    #    nx-1 = the theta_max poleward edge, the upstream end of the equatorward creep) to a
    #    full moving layer at ocean salinity. The model window sits on an INFINITE plane, so
    #    the poleward ocean continues off-grid; the #302 closed wall is the box artifact this
    #    flag softens. The pin is a direct mass reset (like the snap floor: x dx, no dt), so
    #    its SIGNED per-substep delta is accumulated into cum_w_edge / cum_s_edge and the
    #    machine-precision surf_residual / salt_residual gates still close with the boundary
    #    open. Runs AFTER the snap floor so the pinned state is the true end-of-substep
    #    boundary condition. Capture-safe: 0-d device deltas through _accum_pending (never a
    #    host float() on the capture branch); the pins are scalar-fill kernels on fixed buffers.
    if c.surf_water_edge_source:
        _w_pin = C.RHO_LIQ * c.surf_water_h_layer          # pinned W [kg/m^2] = the full moving layer
        _dw_edge = (_w_pin - self.W_surf[-1]) * self._dx_last   # signed injected water [kg/m] (0-d)
        if c.capture_ready:
            self._cum_w_edge_pending = self._accum_pending(self._cum_w_edge_pending, _dw_edge, xp)
        else:
            self._cum_w_edge += float(_dw_edge)
        self.W_surf[-1] = _w_pin
        if c.surf_salt:
            _s_pin = _w_pin * c.surf_salt_salinity0        # pinned DISSOLVED salt [kg/m^2]
            _ds_edge = (_s_pin - (self.S_dis[-1] + self.S_dep[-1])) * self._dx_last   # total-salt delta
            if c.capture_ready:
                self._cum_s_edge_pending = self._accum_pending(self._cum_s_edge_pending, _ds_edge, xp)
            else:
                self._cum_s_edge += float(_ds_edge)
            self.S_dis[-1] = _s_pin
            self.S_dep[-1] = 0.0                           # the open ocean face carries no crust


class StageADCore:
    """The fused anelastic radiative-convective-dynamical core: the grids + anelastic reference state,
    radiation-in-the-loop column thermodynamics + the prognostic surface energy balance, and the
    resolved omega-psi dynamics, in one coupled step() (CPU numpy, or device-resident CuPy on
    device='gpu'). See the module docstring for the architecture decisions #0-#5."""

    def __init__(self, cfg: StageADConfig | None = None):
        self.cfg = cfg or StageADConfig()
        c = self.cfg
        if c.theta_max_deg >= 89.9:
            raise ValueError(f"theta_max_deg={c.theta_max_deg} >= 89.9: x=L*tan(theta) diverges near "
                             "90 deg; use <= ~85 (the cold zone of interest is theta<~80).")
        if c.nx < 4 or c.nz < 4:
            raise ValueError(f"nx={c.nx}, nz={c.nz}: need >= 4 points per axis for the stencils/BCs.")
        if c.lapse_ref <= 0 or c.T_ref_top >= c.T_ref_surf:
            raise ValueError("reference state needs lapse_ref>0 and T_ref_top<T_ref_surf.")
        if c.heat_advection not in ("upwind", "fluxlim"):
            raise ValueError(f"heat_advection={c.heat_advection!r}: expected 'upwind' or 'fluxlim'.")
        # PHY-22 B': exactly two DERIVED forms, no free coefficient. Refuse anything else at
        # construction rather than silently falling through to one of them mid-run (the form is
        # re-read every ``_update_buoyancy`` call, so a typo would otherwise be a silent OFF).
        # Validated unconditionally -- also with the switch OFF -- so a mis-set env key is caught at
        # build time on the OFF path too, before a production arm burns hours on the wrong form.
        if c.buoyancy_pressure_form not in ("linear", "theta"):
            raise ValueError(f"buoyancy_pressure_form={c.buoyancy_pressure_form!r}: expected 'linear' "
                             "(alpha = R_d dT_ref/dz, the core's own pressure convention) or 'theta' "
                             "(the full Lipps-Hemler leg). See ROADMAP PHY-22 / "
                             "flatworld/PHY22_MAPPING_CONVENTION_SCOPING.md section 9.1.")
        if c.tracer_vert_implicit and c.capture_ready and c.device != "gpu":
            raise ValueError("tracer_vert_implicit + capture_ready requires device='gpu': the captured "
                             "step must keep the per-column tracer solve on device (the Thomas RawKernel); "
                             "a host solve cannot be recorded into the graph.")
        if c.dissipation_form not in ("operator", "strain"):
            raise ValueError(f"dissipation_form={c.dissipation_form!r}: expected 'operator' or 'strain'.")
        if c.qc_autoconv_scheme not in ("kessler", "sundqvist"):
            raise ValueError(f"qc_autoconv_scheme={c.qc_autoconv_scheme!r}: expected 'kessler' or 'sundqvist'.")
        if c.surface_drag_rate < 0.0:
            raise ValueError(f"surface_drag_rate={c.surface_drag_rate}: must be >= 0 (0 = off).")
        if c.dt_rad < 0.0:
            raise ValueError(f"dt_rad={c.dt_rad}: must be >= 0 (0 = recompute radiation every step).")
        if c.budget_endorheic and not c.budget_wet_fraction:
            raise ValueError("budget_endorheic requires budget_wet_fraction=True: the endorheic pool closure "
                             "lives inside the budget-wet-fraction branch of _wetness, so ON alone is a silent "
                             "no-op (temperature-proxy wetness mislabeled as the endorheic closure).")
        if c.surf_water_budget:
            if c.surf_water_wmin <= 0.0 or c.surf_water_reservoir <= 0.0 or c.surf_water_h_layer <= 0.0:
                raise ValueError("surf_water_budget needs surf_water_wmin/_reservoir/_h_layer > 0 "
                                 f"(got {c.surf_water_wmin}, {c.surf_water_reservoir}, {c.surf_water_h_layer}).")
            if c.budget_wet_fraction or c.budget_endorheic:
                import warnings
                warnings.warn("surf_water_budget=True SUBSUMES budget_wet_fraction/budget_endorheic (one "
                              "water accounting -- #302): wetness comes from the prognostic W alone; the "
                              "budget_* flags are ignored while surf_water_budget is ON.", stacklevel=2)
            if c.surf_salt and (c.surf_salt_saturation <= 0.0 or c.surf_salt_block_factor <= 0.0
                                or c.surf_salt_salinity0 < 0.0 or c.surf_salt_dry_init < 0.0):
                raise ValueError("surf_salt needs surf_salt_saturation/_block_factor > 0 and "
                                 "_salinity0/_dry_init >= 0 "
                                 f"(got {c.surf_salt_saturation}, {c.surf_salt_block_factor}, "
                                 f"{c.surf_salt_salinity0}, {c.surf_salt_dry_init}).")
            if c.salt_lock_exchange and c.surf_salt_lock_ce <= 0.0:
                raise ValueError("salt_lock_exchange needs surf_salt_lock_ce > 0 "
                                 f"(got {c.surf_salt_lock_ce}).")
            if c.salt_lock_exchange and not c.surf_salt:
                import warnings
                warnings.warn("salt_lock_exchange requires surf_salt (it moves the dissolved-salt field "
                              "S_dis, which exists only when surf_salt is ON) -- the lock-exchange rider "
                              "is inert while surf_salt is OFF.", stacklevel=2)
            if c.surf_water_cryo_pin and c.surf_water_cryo_tfreeze <= 0.0:
                raise ValueError("surf_water_cryo_pin needs surf_water_cryo_tfreeze > 0 K "
                                 f"(got {c.surf_water_cryo_tfreeze}).")
            if c.surf_water_cryo_pin and 0.0 < c.surf_water_cryo_w < W_SNAP_FLOOR:
                # An explicit positive cover thinner than the denormal snap floor would be snapped to 0
                # and re-floored EVERY substep (accounted, so no leak -- but pure denormal churn with
                # wetness ~ 0: a nonsense config that should be loud, review note on the cryo-pin PR).
                raise ValueError("surf_water_cryo_pin: explicit surf_water_cryo_w below the snap floor "
                                 f"W_SNAP_FLOOR={W_SNAP_FLOOR:g} (got {c.surf_water_cryo_w:g}) -- use "
                                 "<= 0 for '= surf_water_wmin', or a physical depth.")
        elif c.surf_salt or c.surf_water_edge_source or c.surf_water_cryo_pin:
            # WARN-AND-IGNORE, deliberately not an error: canonical_env.sh carries
            # COSMO_SURF_WATER_EDGE_SOURCE=1 as a rider that only acts when the water budget itself is
            # ON, so a canonical-env run with the budget OFF must not crash. Every salt/edge/cryo code
            # path below gates on surf_water_budget AND its own flag, so all are structurally inert here.
            import warnings
            warnings.warn("surf_salt/surf_water_edge_source/surf_water_cryo_pin require "
                          "surf_water_budget=True and are IGNORED without it (no salt state, closed "
                          "walls, no cryo floor).", stacklevel=2)
        if c.grid_type not in ("uniform", "stretched"):
            raise ValueError(f"grid_type={c.grid_type!r}: expected 'uniform' or 'stretched'.")
        if c.precision not in ("fp64", "fp32"):
            raise ValueError(f"precision={c.precision!r}: expected 'fp64' or 'fp32'.")
        # Single source of truth for the prognostic-state / dynamics-arithmetic dtype. fp64 -> np.float64
        # (byte-identical: every later .astype(self._dtype) is then a value-preserving no-op); fp32 ->
        # np.float32. The Poisson + the implicit vertical-diffusion solves cast UP to fp64 regardless.
        self._dtype = np.float32 if c.precision == "fp32" else np.float64
        if c.thermo_grid_type not in ("uniform", "stretched"):
            raise ValueError(f"thermo_grid_type={c.thermo_grid_type!r}: expected 'uniform' or 'stretched'.")
        if c.x_grid_type not in ("uniform", "stretched"):
            raise ValueError(f"x_grid_type={c.x_grid_type!r}: expected 'uniform' or 'stretched'.")
        if c.x_grid_type == "stretched":
            if c.chimney_nest or c.chimney_param or c.equator_norain_cells > 0:
                raise ValueError("x_grid_type='stretched' REPLACES the equatorial-chimney treatments "
                                 "(chimney_nest / chimney_param / equator_norain_cells): the refined axis "
                                 "resolves the updraft the nest/param/suppression stood in for -- don't stack.")
            if c.device == "gpu" and c.poisson_method != "dst":
                raise ValueError(
                    "x_grid_type='stretched' with device='gpu' requires poisson_method='dst' (the device "
                    "eigh+GEMM solver): the splu host-round-trip config would mix device thermo fields "
                    "with the host x-centers operator array (a mixed-input crash). Use dst, or device='cpu'.")
            if c.salt_ccn_nd_file:
                raise ValueError(
                    "x_grid_type='stretched' with COSMO_SALT_CCN_ND_FILE: the N_d file is an (nx,) profile "
                    "sampled on a UNIFORM grid -- it must be rebuilt/resampled for the stretched grid "
                    "(nx is derived from the builder). Rebuild the file on the stretched centers first.")
        if c.chimney_param and c.equator_norain_cells > 0:
            raise ValueError("chimney_param and equator_norain_cells are mutually exclusive (don't suppress "
                             "the equatorial precip source AND redistribute it -- pick one).")
        if c.chimney_nest and (c.chimney_param or c.equator_norain_cells > 0):
            raise ValueError("chimney_nest is mutually exclusive with chimney_param / equator_norain_cells "
                             "(the two-way nest already owns the inner-zone convective response -- don't also "
                             "de-spike or suppress it).")
        if c.prognostic_rain:
            # PROGNOSTIC RAIN q_r (stretched-x stage 7; config note at ``prognostic_rain``).
            if not c.prognostic_condensate:
                raise ValueError("prognostic_rain requires prognostic_condensate (q_r sources from the "
                                 "autoconversion of q_c; the single-pass core has no autoconversion).")
            if c.chimney_param or c.chimney_nest:
                raise ValueError("prognostic_rain is mutually exclusive with chimney_param / chimney_nest "
                                 "(both rewrite self._precip assuming the instant same-substep rainout the "
                                 "prognostic reservoir replaces -- stacking them double-books the rain).")
            if c.capture_ready and not c.thermo_fp32:
                # Capture is supported through the PRODUCTION capture path only (the thermo-fp32
                # swap-reconcile wrapper, which reconciles q_r into its persistent fp64 buffer every
                # substep -- mid-substep rebinds are then harmless by construction). The fp64-capture
                # variant would need the full in-place store discipline threaded through the q_r
                # reroute/sediment/re-evap chain; run_driver never uses fp64+capture, so it stays guarded.
                raise ValueError("prognostic_rain + capture_ready requires thermo_fp32=True (the "
                                 "production capture path); fp64-capture is not wired for q_r.")
            if c.rain_v_term <= 0.0 or c.rain_reevap_tau <= 0.0:
                raise ValueError(f"prognostic_rain: rain_v_term={c.rain_v_term} and "
                                 f"rain_reevap_tau={c.rain_reevap_tau} must both be > 0.")
        if c.salt_ccn:
            if not c.prognostic_condensate:
                raise ValueError("salt_ccn requires prognostic_condensate (the autoconversion + in-cloud-LWP "
                                 "machinery the CCN limbs modulate; the single-pass core has neither).")
            if c.chimney_param or c.equator_norain_cells > 0:
                raise ValueError("salt_ccn is mutually exclusive with chimney_param / equator_norain_cells "
                                 "(the equatorial-precip diagnostics rewrite the autoconversion demand the salt "
                                 "limb scales -- combining them muddies the CCN sensitivity; pick one).")
            if c.salt_ccn_source not in ("prescribed", "transport"):
                raise ValueError(f"salt_ccn_source={c.salt_ccn_source!r}: expected 'prescribed' or 'transport'.")
            if c.salt_ccn_haze and c.salt_ccn_haze_tau_cap <= 0.0:
                raise ValueError(f"salt_ccn_haze_tau_cap={c.salt_ccn_haze_tau_cap}: must be > 0 (a positive "
                                 "dust-storm optical-depth ceiling; use inf to recover the bare linear form).")
            if c.salt_ccn_dynamic:
                # DYNAMIC salt-CCN production (the cadenced refresh_salt_ccn thermostat; config note).
                # Validated BEFORE the transport-mode footgun warning below so the dynamic+transport
                # combination raises cleanly instead of warning first.
                if c.salt_ccn_source != "prescribed":
                    raise ValueError("salt_ccn_dynamic replaces the in-step 'transport' source scheme (the "
                                     "capture-hostile per-step refresh); use salt_ccn_source='prescribed' "
                                     "(optionally with an ND_FILE t=0 profile) and let the driver-cadenced "
                                     "refresh_salt_ccn() evolve N_d.")
                if c.salt_ccn_refresh_days <= 0.0 or c.salt_ccn_tau_days <= 0.0:
                    raise ValueError(f"salt_ccn_dynamic: refresh_days={c.salt_ccn_refresh_days} and "
                                     f"tau_days={c.salt_ccn_tau_days} must both be > 0.")
                if not (0.0 < c.salt_ccn_dry_thresh < 1.0):
                    raise ValueError(f"salt_ccn_dry_thresh={c.salt_ccn_dry_thresh}: must be in (0, 1) "
                                     "(a wetness threshold).")
                if c.salt_ccn_scav_pmm <= 0.0 or c.salt_ccn_emis_scale <= 0.0:
                    raise ValueError(f"salt_ccn_dynamic: scav_pmm={c.salt_ccn_scav_pmm} and "
                                     f"emis_scale={c.salt_ccn_emis_scale} must both be > 0.")
            if c.salt_ccn_source == "transport" and c.salt_ccn_transport_refresh_every <= 0:
                import warnings
                warnings.warn(                                        # footgun guard (#237 review): the field
                    "salt_ccn_source='transport' with salt_ccn_transport_refresh_every=0: N_d stays at the "
                    "CLEAN reference (N_ref) until you rebuild it from a resolved wind. Call "
                    "_build_salt_ccn_field() after spin-up, or set salt_ccn_transport_refresh_every>0, or the "
                    "transport coupling is a silent no-op.", stacklevel=2)
        elif c.salt_ccn_dynamic:
            raise ValueError("salt_ccn_dynamic requires salt_ccn (the master flag builds the N_d field the "
                             "dynamic refresh evolves).")
        if c.cond_tau_salt:
            if not c.salt_ccn:
                raise ValueError("cond_tau_salt requires salt_ccn (the #237 N_d CCN field is what sets the "
                                 "spatially-varying tau_phase(x); without it there is no salt field to couple to).")
            if c.cond_tau <= 0.0:
                raise ValueError("cond_tau_salt requires cond_tau > 0 (cond_tau is the clean-air reference "
                                 "tau that the (N_ref/N_d)^gamma factor scales; tau=0 has no rate to modulate).")
        if c.cond_pdf_tail:
            if c.prognostic_condensate:
                raise ValueError("cond_pdf_tail applies to the SINGLE-PASS core only (prognostic_condensate="
                                 "False): the prognostic path's source already relaxes q_c toward the PDF-tail "
                                 "first moment when qc_pdf_cloud=True (the canonical footing) -- there is "
                                 "nothing for this flag to fix there.")
            if not c.qc_pdf_cloud:
                raise ValueError("cond_pdf_tail requires qc_pdf_cloud (it unifies the condensation target "
                                 "with the PDF cover the radiation computes; with qc_pdf_cloud=False the "
                                 "cover is the RH-diagnostic and the unification target does not exist).")
        if c.cloud_sw_twostream and c.cloud_sw_overlap != "column":
            import warnings
            warnings.warn(                                    # footgun guard (PR #328 review finding 1)
                f"cloud_sw_twostream=True with cloud_sw_overlap={c.cloud_sw_overlap!r}: the two-stream cloud "
                "SW solves the reflectance on the COLUMN-TOTAL footing, so cloud_sw_overlap is INERT under it "
                "(the 'random' overlap is not applied, even at ssa=1). The ssa=1 byte-identical guarantee "
                "therefore holds only for overlap='column' (the default/canonical). Set cloud_sw_overlap="
                "'column' to silence this.", stacklevel=2)
        if c.thermo_dt_min > 0.0 and c.capture_ready:
            raise ValueError("thermo_dt_min > 0 is incompatible with capture_ready=True: the sub-cycling "
                             "loop's variable iteration count breaks CUDA graph capture. For GPU production "
                             "use the finest global dt with capture instead.")
        if c.adv_subcycle_n > 1 and c.capture_ready:
            raise ValueError("adv_subcycle_n > 1 is incompatible with capture_ready=True: the per-column "
                             "sub-cycling hits float(xp.max(...)) which is a D2H sync illegal under CUDA "
                             "graph capture.")
        if c.diurnal_sun:
            # DIURNAL SUN (chunk 4.1) construction-time sanity checks (config doc: stage_ad_config.py).
            # NOTE (PR #301 review finding 1): there is deliberately NO row-per-step (n_phase*dt == T_day)
            # requirement -- the phase is carried by an in-place-accumulated device TIME scalar (the
            # diurnal clock), exact under the CFL-adaptive dt on both the eager and captured paths.
            if c.diurnal_nphase < 8:
                raise ValueError(f"diurnal_nphase={c.diurnal_nphase}: needs >= 8 phase rows to resolve a "
                                 "diurnal cycle (production default 1440 = 60 s phase resolution).")
            if c.diurnal_wetness_tau <= 0.0:
                raise ValueError(f"diurnal_wetness_tau={c.diurnal_wetness_tau}: must be > 0 s (the slow "
                                 "surface-T/E_w EMA timescale; >= 1 day recommended -- review finding 2).")
            if c.dt_rad > 0.0:
                import warnings
                warnings.warn("diurnal_sun with dt_rad > 0: the held radiation ALSO holds the diurnal "
                              "phase across the hold window, aliasing the cycle. Use dt_rad=0 (recompute "
                              "every step -- the canonical cadence) for diurnal runs.", stacklevel=2)
            if c.salt_ccn and c.salt_ccn_dynamic and c.salt_ccn_wind_tau_days <= 0.0:
                raise ValueError(f"salt_ccn_wind_tau_days={c.salt_ccn_wind_tau_days}: must be > 0 days "
                                 "(the |u_surf|^3 emission-key EMA under the diurnal sun -- #308 item 2).")
        # (dz_min validity, 0 < dz_min < z_top/(nz-1), is enforced by ``_stretched_z`` once z_top is known.)

        # --- cross-track grid: IN PHYSICAL DISTANCE x [m], never theta (#1). Default UNIFORM (a single
        #     scalar spacing -- the byte-identical path); with x_grid_type="stretched" the capped-geometric
        #     ``_stretched_x_edges`` grid, fine at the equatorial axis (the chimney-nest replacement
        #     program). The x-side mirrors the vertical-stretch plumbing exactly:
        #       ``self._dx_op``   -- what the OPERATORS take: the scalar dx (uniform, verbatim stencils) or
        #                            the x CENTERS array (stretched; kernels derive h^-/h^+/dxc from it);
        #       ``self._dx_cell`` -- TRUE cell widths from the edge tiling (None when uniform); sums to
        #                            x_max exactly -- the budget weight and flux-form/W-transport divisor
        #                            (NOT the operators' centers-derived dxc; both are second-order, but
        #                            budgets must tile the domain exactly);
        #       ``self._dx_w``    -- the integral weight: the scalar dx (uniform -- byte-identical sums) or
        #                            ``_dx_cell`` (stretched);
        #       ``self._dx_min``  -- tightest width -> the CFL;
        #       ``self.dx``       -- the scalar spacing on the uniform grid; ``None`` when stretched, so
        #                            any un-migrated ``self.dx`` consumer fails LOUDLY on the stretched
        #                            path instead of silently using a wrong scalar (uniform is untouched).
        x_max = g.x_of_theta(np.radians(c.theta_max_deg))
        # The REQUESTED far-field column count, kept because the stretched builder OVERWRITES c.nx
        # with the derived one below -- and the auto plateau spacing (x_dx_max = 0 -> inner_end/nx)
        # reads the REQUESTED value, so anything that must rebuild this exact grid later (the
        # cross-grid warm start, flatworld.regrid_x) needs it and cannot recover it from cfg.
        self._nx_requested = int(c.nx)
        self._x_stretched = (c.x_grid_type == "stretched")
        if self._x_stretched:
            # box-extension tail (parameterized coarsen-start -- see stage_ad_config). The deg form
            # routes through the SAME x_of_theta expression as the box end, so a pure-extension arm's
            # coarsen-start is float-identical to the baseline box end (bitwise inner grid).
            tail_from_x = None
            if c.x_tail_from_deg > 0.0 or c.x_tail_from > 0.0:
                if c.x_tail_from_deg > 0.0 and c.x_tail_from > 0.0:
                    raise ValueError("set x_tail_from_deg OR x_tail_from, not both (they name the same "
                                     "coarsen-start; the deg form is the pinning-safe one).")
                tail_from_x = (g.x_of_theta(np.radians(c.x_tail_from_deg)) if c.x_tail_from_deg > 0.0
                               else c.x_tail_from)
            # 0 -> auto dx_max anchors at the INNER domain end: tail_from when tailed (so the auto
            # plateau equals what the same nx request gives on an untailed box ending there -- the
            # pure-extension arm's bitwise-control property), x_max otherwise (the rule unchanged).
            inner_end = tail_from_x if tail_from_x is not None else x_max
            dx_max = c.x_dx_max if c.x_dx_max > 0 else inner_end / c.nx
            tail_kw = {}
            if tail_from_x is not None:
                tail_kw = dict(tail_from=tail_from_x, tail_dx_max=c.x_tail_dx_max,
                               tail_ratio=(c.x_tail_ratio if c.x_tail_ratio > 0.0 else c.x_ratio))
            edges = _stretched_x_edges(x_max, c.x_dx_min, dx_max, ratio=c.x_ratio, n_fine=c.x_n_fine,
                                       **tail_kw)
            if c.nx != len(edges) - 1:
                tail_note = (f", tail from {tail_from_x:g} m cap {c.x_tail_dx_max:g} m" if tail_kw else "")
                print(f"[stage_ad] x_grid_type='stretched': nx DERIVED from the grid builder: "
                      f"{c.nx} -> {len(edges) - 1} (dx {c.x_dx_min:g}..{dx_max:g} m, ratio {c.x_ratio:g}, "
                      f"n_fine {c.x_n_fine}{tail_note}; size arrays via stretched_x_cell_count)", flush=True)
                c.nx = len(edges) - 1
            self.x = 0.5 * (edges[:-1] + edges[1:])                # cell CENTERS [m] (nx,)
            self._dx_cell = np.diff(edges)                         # TRUE widths [m]; sum == x_max exactly
            self._dx_min = float(self._dx_cell.min())              # == c.x_dx_min by construction
            self._dx_op = self.x                                   # operators derive spacings from CENTERS
            self._dx_w = self._dx_cell                             # integral weight (nx,)
            self._dx_div = self._dx_cell[:, None]                  # (nx, 1) flux-form cell-volume divisor
            self._dx_last = float(self._dx_cell[-1])               # poleward edge cell width (the plateau)
            self._inv_dxw_row = 1.0 / self._dx_cell[None, :]       # (1, nx) -> the per-cell CFL caps
            self.dx = None                                         # loud-fail for un-migrated consumers
        else:
            if c.x_tail_from > 0.0 or c.x_tail_from_deg > 0.0:
                raise ValueError("x_tail_from/_deg > 0 (the outward-coarsening box-extension tail) requires "
                                 "x_grid_type='stretched' -- the uniform grid has no tail to extend.")
            self.x = (np.arange(c.nx) + 0.5) * (x_max / c.nx)      # cell centers [m]
            self.dx = float(x_max / c.nx)                          # PYTHON float (not np.float64): a numpy
            #   scalar coefficient upcasts an fp32 field (numpy weak-promotion applies only to PYTHON scalars),
            #   which would silently revert the fp32 dynamics stencils to fp64; a Python float preserves the
            #   field dtype. Identical value/result on the fp64 path (byte-identical). Same reasoning for the
            #   uniform vertical spacings below.
            self._dx_cell = None
            self._dx_min = self.dx
            self._dx_op = self.dx                                  # scalar -> uniform verbatim stencils
            self._dx_w = self.dx                                   # scalar weight -> byte-identical sums
            self._dx_div = self.dx                                 # scalar divisor -> byte-identical
            self._dx_last = self.dx
            self._inv_dxw_row = None
        # --- scale-aware convective-adjustment gate (stretched-x gray zone; default OFF) ---
        #     ``_conv_gate_col`` is None (OFF -- the byte-identical path) or an (nx, 1) per-column factor
        #     w_conv(dx_local) in [0, 1] multiplying the Betts-Miller relaxation fraction: 0 where the grid
        #     RESOLVES convection (fine axis cells), 1 where it is fully parameterized, smoothstep in
        #     ln(dx) between. Collapses to None when every cell is at w_conv == 1 (e.g. gating flags set on
        #     a coarse uniform grid), so "ON at coarse-uniform" is EXACTLY OFF, not merely close.
        self._conv_gate_col = None
        self._conv_gate_heat = None                                # who-lifts accumulator [J/m^2] (nx,)
        if (c.conv_dx_off > 0.0) != (c.conv_dx_full > 0.0):
            raise ValueError("conv_dx_off and conv_dx_full must be set together (both > 0) or both 0 (off).")
        if c.conv_dx_full > 0.0:
            if not (0.0 < c.conv_dx_off < c.conv_dx_full):
                raise ValueError(f"need 0 < conv_dx_off < conv_dx_full (got {c.conv_dx_off:g}, "
                                 f"{c.conv_dx_full:g}).")
            widths = self._dx_cell if self._dx_cell is not None else np.full(c.nx, self.dx)
            t = np.log(widths / c.conv_dx_off) / np.log(c.conv_dx_full / c.conv_dx_off)
            t = np.clip(t, 0.0, 1.0)
            w_conv = t * t * (3.0 - 2.0 * t)                       # smoothstep in ln(dx)
            if np.any(w_conv < 1.0):                               # any gated column at all?
                if c.capture_ready or c.device == "gpu":
                    # Raised only when the gate actually ENGAGES (review, PR #320): a gate config that
                    # collapses to OFF on a coarse grid stays runnable under capture / on device.
                    raise NotImplementedError("the convective-adjustment gate is CPU-path only for now "
                                              "(the gated blend allocates and the who-lifts accumulator "
                                              "is host numpy; device-safe buffers are a follow-up).")
                self._conv_gate_col = self._cast(w_conv[:, None].copy())   # (nx, 1), dtype-matched
                self._conv_gate_heat = np.zeros(c.nx)              # who-lifts: accumulated |dT| enthalpy
                #   (diagnostic-only; NOT checkpoint-persisted -- within-run x-shape, resets on resume)
        # --- PHY-19 remedy (b): the MINIMUM HORIZONTAL SCALE filter on the convective TENDENCY
        #     (``conv_min_scale_m``, env COSMO_CONV_MIN_SCALE_M, metres; 0.0 = OFF = byte-identical).
        #     Built ONCE here from the cell centres and true widths, so the per-substep cost is one
        #     dense (n_act x n_act) @ (n_act x nz) product per field and nothing is allocated in the
        #     step. ``_minsc_n`` is the count of columns narrower than W (a contiguous prefix on the
        #     stretched-x family: the finest cells are at the x=0 axis), ``_minsc_A_host`` the fp64
        #     operator, and ``_minsc_W`` the width it was built for -- re-checked at the call site so a
        #     warm-start route that sets cfg.conv_min_scale_m on a BUILT core still takes effect
        #     (rebuilt on the eager warmup step, never inside a captured graph).
        self._minsc_n = 0
        self._minsc_A_host = None
        self._minsc_W = 0.0
        self._minsc_A = {}          # dtype -> the operator on self.xp at that dtype (built lazily)
        self._minsc_gemm = {}       # dtype -> the capture-safe tiled-GEMM RawKernel (device path only)
        self._minsc_buf = {}        # dtype -> (dT, dq, fT, fq) fixed (n_act, nz) work buffers
        #   (the operator is built below, once ``self.dm`` exists -- the builder asserts the layer mass
        #    is horizontally uniform, which is what lets the mass weighting cancel. The gray-zone-gate
        #    refusal lives in the builder too, so BOTH entry paths hit it: __init__ and the
        #    documented warm-start route that sets cfg.conv_min_scale_m on a BUILT core.)
        self.theta = g.theta_of_x(self.x)
        self.theta_deg = np.degrees(self.theta)
        self._build_salt_ccn_field()                              # salt-aerosol CCN field N_d (#237; None when OFF)
        # DYNAMIC salt-CCN init tracking: with an ND_FILE the file IS the t=0 profile (the first cadenced
        # refresh RELAXES from it); without one the first refresh_salt_ccn() JUMPS N_d to the target built
        # from the seed state (there is no meaningful profile to relax from). load_state flips this True
        # when a checkpoint carries a live "N_d" (the resume path).
        self._salt_dyn_initialized = bool(c.salt_ccn and c.salt_ccn_dynamic and c.salt_ccn_nd_file)
        self.S = g.day_mean_flux(self.x) * c.i0_scale              # DERIVED day-mean flux [W/m^2]
        # Stage 5 haze forward-scatter (default OFF -> S_diffuse=None, S_sw_in=S, byte-identical). When on,
        # S_diffuse is the extra column-top diffuse-down SW fed to column_radiation (propagated to the surface
        # on its own column path); S_sw_in = S + S_diffuse is the TOTAL down-SW the absorbed-SW / energy-budget
        # diagnostics count as incoming.
        self.S_diffuse = g.day_mean_diffuse_flux(self.x) * c.i0_scale if c.sw_diffuse else None
        self.S_sw_in = self.S + self.S_diffuse if c.sw_diffuse else self.S
        # DIURNAL SUN tables (chunk 4.1; default OFF -> all None, byte-identical). self.S / S_diffuse /
        # S_sw_in above stay the DAY-MEAN for every non-stepping consumer (init guess, spatial shape,
        # diagnostics, the energy gate); only the in-loop radiation reads the _S_now buffers (_radiate's
        # sw_inst). Built HOST here; _place_thermo_on_device moves them (Phase D pattern).
        self._S_table = self._S_diff_table = None      # (n_phase, nx) one-day instantaneous-sun tables
        self._S_now = self._S_diff_now = None          # (nx,) held row buffers (fixed address -> capture-safe)
        self._diurnal_time_dev = None                  # (1,) float64 device diurnal clock [s] (capture path);
        #     accumulated IN PLACE by the realized dt each step -> the phase is exact under adaptive dt
        #     (PR #301 review finding 1; a one-row-per-step counter assumed the dt cap binds, which the
        #     canonical diurnal state contradicts). Eager/CPU reads host sim_time instead.
        self._diurnal_kernel_dev = {}                  # device gather RawKernel(s), keyed by C-dtype (lazy)
        if c.diurnal_sun:
            self._build_diurnal_tables()

        # --- PRESSURE grid (column thermodynamics; #1): interfaces TOP->SURFACE, layers between ---
        if c.thermo_grid_type == "stretched":
            # Co-refine the PRESSURE grid near P_SURF (BL_REFINEMENT_SPEC.md #F, P2): the same sinh stretch
            # in the surface-distance coordinate d = P_SURF - p (fine at d=0 = the surface), reversed to the
            # TOP->SURFACE convention. Finer near-surface pressure layers give a converged 2 m air
            # temperature / latent flux. Default "uniform" -> the linspace below (byte-identical).
            d = _stretched_z(C.P_SURF - c.p_top, c.nz + 1, c.thermo_dp_min)   # d[0]=0 (surface), fine near 0
            self.p_edge = (C.P_SURF - d)[::-1].copy()             # TOP(p_top) -> SURFACE(P_SURF), fine at surface
            self.p_edge[0] = c.p_top; self.p_edge[-1] = C.P_SURF  # pin endpoints exactly (kill round-off)
        else:
            self.p_edge = np.linspace(c.p_top, C.P_SURF, c.nz + 1)
        self.p_mid = 0.5 * (self.p_edge[:-1] + self.p_edge[1:])
        self.dm = np.diff(self.p_edge) / C.G_ACCEL                 # layer mass [kg/m^2]
        self.sigma = self.p_mid / C.P_SURF
        self.RH = c.rh_top + (c.rh_surf - c.rh_top) * self.sigma   # RH(sigma): moist below, dry aloft
        self.C_layer = self.dm * C.CP_AIR                          # per-layer heat capacity [J/m^2/K]
        if c.conv_min_scale_m and float(c.conv_min_scale_m) > 0.0:
            self._build_min_scale_operator()   # PHY-19 (b); here because it asserts on self.dm above
        # issue #99: per-interface log-pressure steps for the vectorized ``_hydrostatic_z``, precomputed
        # with the SAME scalar np.log calls its original per-interface loop made (array-log vs scalar-log
        # can differ in the last ulp on SIMD builds; the scalar precompute keeps the heights bit-identical).
        # INVARIANT (PR #102 review): this assumes ``p_edge`` is immutable after __init__ (set once above,
        # never written again in the package) -- any future p-grid mutation must recompute _dlnp or the
        # heights silently desync.
        self._dlnp = np.array([float(np.log(self.p_edge[k + 1]) - np.log(self.p_edge[k]))
                               for k in range(c.nz)])

        # --- HEIGHT grid (dynamics; #1): SURFACE(0)->TOP(nz-1). The top is the reference height of p_top,
        #     so the two grids span the same column in the reference state. Default UNIFORM in z; with
        #     grid_type="stretched" (BL_REFINEMENT_SPEC.md) a sinh-stretch fine near the surface. The
        #     vertical operators take ``self._dz_op``: a SCALAR float (uniform -> the current verbatim
        #     stencils) OR the ``z`` LEVELS array (stretched -> the non-uniform 3-point/finite-volume
        #     branches, derived from the levels). ``self.dz`` stays the (scalar) lowest spacing for state/
        #     diagnostics; ``self._dz_min`` (tightest interval) sets the CFL; ``self._dz_cell`` (per-level
        #     cell thickness, summing to z_top) weights the vertical integrals. ---
        self.z_top = self._z_of_ref_pressure(c.p_top)
        self._stretched = (c.grid_type == "stretched")
        if self._stretched:
            self.z = _stretched_z(self.z_top, c.nz, c.dz_min)      # fine-near-surface levels [m]
            dz_iv = np.diff(self.z)                                # (nz-1,) interval spacings
            self.dz = float(dz_iv[0])                              # lowest spacing [m] (diagnostics/state)
            self._dz_min = float(dz_iv.min())                      # tightest interval -> CFL
            dzc = np.empty(c.nz)                                   # per-level cell thickness (integration weight)
            dzc[1:-1] = 0.5 * (self.z[2:] - self.z[:-2])
            dzc[0] = 0.5 * (self.z[1] - self.z[0])
            dzc[-1] = 0.5 * (self.z[-1] - self.z[-2])
            self._dz_cell = dzc                                    # sum(dzc) == z_top (exact telescoping)
            self._dz_op = self.z                                   # operators derive non-uniform spacings
        else:
            self.z = np.linspace(0.0, self.z_top, c.nz)            # uniform height levels [m]
            self.dz = float(self.z[1] - self.z[0])                 # PYTHON float (see self.dx note: a numpy
            self._dz_min = self.dz                                 #   scalar would upcast the fp32 stencils)
            self._dz_cell = None                                   # uniform -> scalar-dz integral form
            self._dz_op = self.dz                                  # scalar -> uniform verbatim stencils

        # --- ANELASTIC reference state rho_ref(z), T_ref(z), p_ref(z) (#0) ---
        self.p_ref, self.T_ref = self._reference_profile(self.z)
        self.rho_ref = self.p_ref / (C.R_DRY * self.T_ref)         # ideal gas; hydrostatic-consistent
        self.N2_ref = self._brunt_vaisala_sq(self.T_ref, self.z)   # diagnosed background stratification
        # PHY-22 B' (``buoyancy_pressure_term``): the two fixed (nz,) reference rows the pressure term
        # reads. Built ALWAYS -- they are two tiny host arrays and building them unconditionally keeps
        # the OFF path free of any build-time branch -- but read only when the flag is ON.
        #   alpha(z) = R_d * dT_ref/dz [m/s^2]: the ANALYTIC derivative of the reference profile, which
        #   is the constant -R_d*lapse_ref in the lapse branch and EXACTLY 0 in the isothermal cap. It is
        #   taken analytically rather than by np.gradient because the profile has a kink at z_iso that a
        #   centred difference would smear over one cell; the equivalent numerical identity
        #   alpha == -g - R_d*T_ref*dln(rho_ref)/dz is pinned as a test instead of used here.
        _z_iso_ref = (c.T_ref_surf - c.T_ref_top) / c.lapse_ref            # cap onset height [m]
        self._buoy_alpha = np.where(self.z <= _z_iso_ref,
                                    -C.R_DRY * c.lapse_ref, 0.0)           # (nz,) [m/s^2]
        self._inv_p_ref = 1.0 / self.p_ref                                 # (nz,) [1/Pa]
        #   the layer pressures as a SINGLE ROW (1, nz), TOP->SURFACE like T_atm: p_mid is x-independent
        #   (the mass grid is the same in every column), so the pressure map rides the shared bracket as
        #   a broadcast row and costs one cheap flat gather, not an (nx, nz) materialisation.
        self._p_mid_row = self.p_mid[None, :].copy()                       # (1, nz) [Pa]
        #   the level-index column the clamp gather clips, precomputed rather than built per call: an
        #   ``xp.arange`` inside the captured region is an allocation the graph does not need.
        self._buoy_kcol = np.arange(c.nz)[:, None]                         # (nz, 1) level index

        # --- anelastic dynamics operators (#0, #2; M2b.2): the 1/rho_ref factor, the vortex-stretching
        #     coefficient d(ln rho_ref)/dz, and the LU-factorized streamfunction Poisson. ---
        self._beta = 1.0 / self.rho_ref                            # 1/rho_ref on the height grid (nz,)
        self.dlnrho_dz = np.gradient(np.log(self.rho_ref), self.z)  # anelastic stretching coeff (nz,) <0
        # Phase E: cache the lowest-cell thickness as a HOST float NOW (self.z is still host here), so the
        # per-substep surface-drag curl (``_surface_drag_curl``) and ``bulk_drag_rate`` read a python scalar
        # instead of ``float(self.z[1]-self.z[0])`` -- which becomes a device->host sync once self.z is placed
        # on device (illegal during CUDA-graph capture). Value-identical to the inline expression.
        self._z1 = float(self.z[1] - self.z[0])                    # lowest interior cell thickness [m]
        # These two STATIC (nz,) profiles are initialised HOST here; on the GPU path Phase E
        # (_place_dynamics_on_device) places them on device once at init, so the dynamics substep makes
        # no per-substep H2D copies (history: archive/M31_RESULTS.md §3c). On CPU they stay numpy
        # (no-op placement).
        if c.poisson_method == "dst":
            # Separable DST x tridiagonal fast inverter (drop-in for splu; closed-wall global core only).
            # GUARD: the DST path assumes psi=0 Dirichlet on ALL FOUR walls -- separability breaks under the
            # chimney's open_outer_edge Neumann BC. StageADCore is always the closed-wall core (the chimney
            # is a SEPARATE class with its own open-edge build), so this is structurally safe; the assert
            # makes a future attempt to route an open-edge operator through DST fail loudly, not silently.
            assert not getattr(c, "open_outer_edge", False), \
                "poisson_method='dst' is for the closed-wall core only (open_outer_edge breaks separability)"
            if self._x_stretched:
                # The DST-I sine basis diagonalizes only the CONSTANT-coefficient (uniform-dx) x-operator;
                # the stretched grid takes the eigendecomposition + GEMM generalization (the arm-G
                # machinery): one init-time symmetric-tridiagonal eigh, two dense GEMMs per solve, the
                # SAME batched z-Thomas. Drop-in splu semantics; residual-parity-gated against splu.
                self._lu_dyn = StretchedSeparablePoisson(self.rho_ref, c.nx, c.nz, self._dx_op,
                                                         self._dz_op)
            else:
                self._lu_dyn = SeparablePoisson(self.rho_ref, c.nx, c.nz, self._dx_op, self._dz_op,
                                                workers=c.poisson_workers)
        elif c.poisson_method == "splu":
            self._lu_dyn = build_anelastic_poisson(self.rho_ref, c.nx, c.nz, self._dx_op, self._dz_op)
        else:
            raise ValueError(f"poisson_method must be 'splu' or 'dst', got {c.poisson_method!r}")
        self._last_dt_dyn = 0.0                                    # last adaptive dynamics dt (diagnostic)
        self._diss_Dcol = None                                     # M3.4b Task 2: per-substep operator KE-removal
        #     stash (consistent pre-update omega), set in _dynamics_substep; None -> the dissipative-heating
        #     tendency recomputes (standalone/test calls). See _dissipative_heating_tendency / _dynamics_substep.

        # --- state arrays ---
        #   thermodynamics on the PRESSURE grid (nx, nz), A-convention (last axis TOP->SURFACE):
        T0 = (np.maximum(self.S, c.s_init_floor) / C.SIGMA_SB) ** 0.25   # local radiative-eq surface guess
        self.T_surf = self._cast(np.clip(T0, c.t_init_floor, 360.0))     # surface / ocean mixed-layer T [K] (nx,)
        z_col = -8000.0 * np.log(self.p_mid / C.P_SURF)            # scale-height z guess for the column
        self.T_atm = self._cast(np.maximum(self.T_surf[:, None] - 6.5e-3 * z_col[None, :], c.t_atm_init_floor))  # (nx, nz)
        # DIURNAL SUN wetness keying state (chunk 4.2 / PR #301 review finding 2; default OFF -> all None,
        # byte-identical). Under diurnal_sun the surface T swings hundreds of K per cycle, but the wetness
        # ramp (wetness_of_Tsurf) is a TIME-MEAN climate classifier -- evaluated on instantaneous T it
        # re-wets the dry salt flats every night (evaporation from a waterless surface + a ~130x night
        # heat-capacity inflation that pins the dawn floor just below SST_DRY, clipping the cold half of
        # the cycle the 4.4 comparison exists to measure). So ON, the ramp / endorheic-liquid / capacity
        # keying reads a SLOW surface temperature: _T_surf_slow, an EMA of T_surf with tau =
        # diurnal_wetness_tau (>= 1 day), updated in-place at end-of-step with the realized dt; the
        # budget leg's wet-tile demand gets the same smoothing (_E_w_slow -- instantaneous E_w -> 0 at
        # night would drive budget_f = F/E_w -> 1, re-adding wetness). diurnal_wetness_freeze instead
        # FREEZES the whole wetness field at the seed state's classification (_wetness_frozen; the clean
        # 4.4 control). #303's prognostic surface-water budget dissolves all of this automatically -- the
        # EMA keying is the legacy-wetness path's interim fix. Built fp64 host; placed on device later.
        self._T_surf_slow = np.asarray(self.T_surf, dtype=np.float64).copy() if c.diurnal_sun else None
        self._E_w_slow = None                          # lazy fp64 EMA of the wet-tile demand (budget leg)
        self._wetness_frozen = (np.asarray(wetness_of_Tsurf(np.asarray(self.T_surf, float)))
                                if (c.diurnal_sun and c.diurnal_wetness_freeze) else None)
        #   prognostic specific humidity q on the PRESSURE grid (nx, nz), co-located with T_atm (M2b.3):
        #   its master grid is the pressure grid, where evaporation/condensation/radiation/qsat are EXACT
        #   and energy-conserving (latent heat released on the same grid as T). It is reconciled to the
        #   height grid ONLY for advection by the resolved flow (one-way increment map; see
        #   _advect_moisture). [Architecture refinement over the M2b.1 "q on the height grid" placement:
        #   advection-master-on-height forces a per-step round-trip that spuriously diffuses q and makes
        #   the latent-heat remap non-conservative -- pressure-master avoids both. See archive/M2B3_RESULTS.md.]
        #   PROGNOSTIC SURFACE WATER W (nx,) [kg/m^2 = mm] (#302; surf_water_budget; None when OFF so every
        #   flag-OFF path is untouched). Built BEFORE the q warm start below because _wetness() (flag ON)
        #   reads it. fp64 ALWAYS -- see _init_surf_water.
        self.W_surf = None
        #   PROGNOSTIC SURFACE SALT (surf_salt; None when OFF): DISSOLVED S_dis + DEPOSITED halite crust
        #   S_dep, both (nx,) [kg/m^2] fp64 ALWAYS (they ride the same budget arithmetic as W -- the
        #   fp32 bit-freeze argument applies identically). Built with W in _init_surf_water.
        self.S_dis = None
        self.S_dep = None
        if c.surf_water_budget:
            self._init_surf_water("construction")
        wet0 = self._wetness()
        self.q = self._cast(np.minimum(
            wet0[:, None] * self.RH[None, :] * th.qsat(np.asarray(self.T_atm, float), self.p_mid[None, :]),
            0.5))                                                  # fixed-RH warm start; evolves prognostically
        #   prognostic CLOUD CONDENSATE q_c (nx, nz), co-located with q/T_atm on the pressure grid
        #   (CLOUD_PROGNOSTIC_ROADMAP Chunk 1). Initialised 0; with prognostic_condensate OFF it is never
        #   sourced/advected (stays 0 -> byte-identical). The retained reservoir condensation feeds and
        #   autoconversion + the resolved flow move.
        self.q_c = np.zeros((c.nx, c.nz), dtype=self._dtype)
        #   prognostic FALLING RAIN q_r (nx, nz), co-located with q_c on the pressure grid (stretched-x
        #   stage 7; config note at ``prognostic_rain``). None when the flag is OFF (no allocation, no
        #   code path touches it -> byte-identical); zeros when ON. Sourced by the autoconversion+ceiling
        #   sink (rerouted in _prognostic_microphysics), advected like q_c (_advect_rain), sedimented at
        #   -rain_v_term (_sediment_rain -> self._precip), re-evaporated below cloud (_rain_reevap).
        #   Checkpointed and device-placed exactly as q_c is.
        self.q_r = np.zeros((c.nx, c.nz), dtype=self._dtype) if c.prognostic_rain else None
        #   Chunk-4 BL/stratocumulus deck cover (nx, nz): the overcast cover the closure stamps on its deck
        #   cells, read by the radiation as a cover FLOOR so the deck radiates as a solid low cloud. Derived
        #   per-step by _bl_stratocumulus, but it is READ by the next step's radiation BEFORE that step
        #   recomputes it -- so it is carry-over state across a save/resume boundary and IS checkpointed
        #   (key "bl_deck_cover", present only when the closure is live; see state()/load_state). Init 0
        #   (no floor) for a cold core or a legacy checkpoint that predates the key.
        #   fp64 ALWAYS (the W_surf discipline): the live field is xp.where(mask, float, 0.0) -> float64
        #   whatever the working dtype, so the round-trip must not silently re-type it.
        self._bl_deck_cover = np.zeros((c.nx, c.nz))
        #   dynamics on the HEIGHT grid (nz, nx), B-convention (first axis SURFACE->TOP). The DYNAMICS
        #   state (omega, psi, b) is placed on the configured device: self.xp is numpy ("cpu") or CuPy
        #   ("gpu"). On the GPU path (Phase C) BOTH dynamics linear solves run device-resident (the DST
        #   Poisson + the implicit tridiag below); on CPU the splu/DST Poisson stays host numpy.
        #   The THERMO state (T_atm, q, T_surf, q_c) is built host above and moved onto the device by
        #   _place_thermo_on_device (Phase D, end of __init__), so on device="gpu" the coupled column
        #   physics runs device-resident too.
        self.xp = array_module(c.device)
        self._on_device = (c.device == "gpu")
        # GPU dynamics solves ON DEVICE (Phase C): when device="gpu" AND poisson_method="dst", the two host
        # linear solves in the dynamics substep -- the anelastic Poisson (_dynamics_substep +
        # _operator_ke_removal_col) and the implicit-vertical-diffusion tridiag -- run device-resident,
        # removing the per-substep host<->device round-trips. There is NO device analogue of the splu
        # factorization, so a GPU run with poisson_method="splu" KEEPS the M3.1 host round-trip
        # (omega->host->splu->psi->device) -- a valid (uncaptureable) config, retained unchanged. Phase C
        # therefore gates on ``_dyn_on_device`` (= on_device AND dst), NOT bare ``_on_device``. The Poisson +
        # tridiag are built fp64 UNCONDITIONALLY (mirroring the host seam's RHS-upcast: the ill-conditioned
        # solves must stay fp64 even on an fp32 working core).
        self._dyn_on_device = self._on_device and c.poisson_method == "dst"
        self._lu_dyn_dev = None        # device DST Poisson (GPUSeparablePoisson), or None unless _dyn_on_device
        self._tridiag_kernel_dev = None  # device Thomas RawKernel for the implicit tridiag, or None otherwise
        self._convadj_kernel_dev = {}  # device convadjust RawKernel(s), keyed by C-dtype; built lazily (Phase E)
        self._lw_kernel_dev = {}       # device LW two-stream RawKernel(s), keyed by C-dtype; built lazily (Phase E #2)
        self._micro_kernel_dev = {}    # device microphysics RawKernel(s), keyed by C-dtype; built lazily (Phase E #3)
        self._micro_aux = {}           # PERSISTENT working-dtype (dm, p_mid) for the micro kernel, keyed by dtype
        self._convadj_aux = {}         # PERSISTENT static convadjust aux (p_c/dm_c/RHt_c), keyed by device
        #                                dtype; written ONCE in warmup (PR #198 review item 1 -- capture
        #                                hardening: stop the static aux relying on CuPy pool address determinism)
        self._segmix_kernel_dev = {}   # device segment-mixer RawKernel(s), keyed by C-dtype (PHY-17 step 4)
        self._segmix_aux = {}          # ...and its persistent static aux, the _convadj_aux pattern verbatim
        if self._dyn_on_device:
            from ._gpu_kernels import GPUSeparablePoisson, GPUStretchedPoisson, build_tridiag_cols_kernel
            import cupy as _cp_dev
            if self._x_stretched:
                # The stretched twin: eigh+GEMM (host init via the CPU class -> factors moved to device
                # once); the uniform DST path below is untouched (byte-identical).
                self._lu_dyn_dev = GPUStretchedPoisson(self.rho_ref, c.nx, c.nz, self.x, self._dz_op,
                                                       _cp_dev.float64)
            else:
                self._lu_dyn_dev = GPUSeparablePoisson(self.rho_ref, c.nx, c.nz, self._dx_op, self._dz_op,
                                                       _cp_dev.float64)
            if c.vertical_implicit_diffusion or c.prognostic_rain or c.tracer_vert_implicit:
                # The tridiag is also fp64 (the implicit solve is done fp64 on host; mirror it on device).
                # prognostic_rain: the q_r sedimentation is the same per-column solve (capture-safe path).
                # tracer_vert_implicit: the vertical tracer solve dispatches to the same kernel.
                self._tridiag_kernel_dev = build_tridiag_cols_kernel(_cp_dev.float64)
        self.omega = self.xp.asarray(np.zeros((c.nz, c.nx), dtype=self._dtype))   # vorticity [1/s] (M2b.2)
        self.psi = self.xp.asarray(np.zeros((c.nz, c.nx), dtype=self._dtype))     # anelastic streamfunction
        self.b = self.xp.asarray(np.zeros((c.nz, c.nx), dtype=self._dtype))      # buoyancy g(T_v-T_ref)/T_ref [m/s^2];
        #     held (M2b.2 fixed-forcing tests) or from the evolving column T and q (M2b.3 coupling,
        #     _update_buoyancy; T_v = T with virtual_buoyancy OFF)
        #   ocean mixed layer: carried by T_surf with the mixed-layer heat capacity (M2b.4 surface
        #   balance, _surf_heat_capacity); M2b.5 couples the f=0 wind-driven ocean cubic to the surface winds.
        self.ml_depth = c.ocean_depth_spinup                       # [m]
        self._spinup_Cwet = {}           # PHY-17 spin-up sea depth: per-(array module, dtype) wet capacity
        self._spinup_Cwet_params = None  #   cache, rebuilt only when (depth, cutoff) change
        self._spinup_params()            # refuse a bad depth / cutoff at CONSTRUCTION
        # --- coupling switches: which tendencies step() applies. Defaults = the full core; the tests flip
        #   them to isolate limbs (couple_thermo=False -> held b, the pure-dynamics M2b.2 checks;
        #   run_dynamics=False -> the psi==0 no-flow column limit; couple_surface=False -> the M2b.3
        #   prescribed fixed-SST RCE limit). ---
        self.couple_thermo = True      # radiation + surface fluxes + convection + moisture update T_atm/q & b
        self.couple_surface = True     # M2b.4: the prognostic surface ENERGY BALANCE (T_surf an OUTPUT).
        #     When False, T_surf is a PRESCRIBED boundary condition (the M2b.3 fixed-SST RCE limit); the
        #     RCE anchor tests set it False. Gated inside couple_thermo (the surface step needs the
        #     radiation/flux call), so couple_thermo=False also freezes the surface.
        self.run_dynamics = True       # the anelastic omega-psi substep + resolved-flow q advection
        self.couple_heat = True        # M2b.6: the resolved flow advects column POTENTIAL TEMPERATURE
        #     between columns/levels (the inter-column HEAT transport that closes Issue A in the fused
        #     domain). Gated inside run_dynamics+couple_thermo; set False for the M2b.3-5 limit where the
        #     flow carries only MOISTURE and each column sits at its own local RCE (T(theta) == lineage).
        self._last_rad = None          # cache of the last column_radiation output (diagnostics + #86 sub-cycle)
        self._E_w_cache = None         # budget_wet_fraction: held E_w (wet-tile SEB) buffer, refreshed once/step
        self._rad_age = 0.0            # model-seconds since the held radiation was last refreshed (#86)
        self._rad_T_ref = None         # T_atm snapshot at the last radiation refresh (#86 large-move guard)
        self._rad_Tsurf_ref = None     # T_surf snapshot at the last refresh (#86: surf_net holds sigma*T_surf^4)
        self._u_surf = None            # the resolved surface cross-track wind u[1,:] (#85 item 1), kept on the
        #     SAME array module as the dynamics (Phase D: device on GPU, no host stash -- the dynamics produces
        #     u on device and the thermo reads it on device, killing the per-substep (nx,) host pull).
        #     Deliberately NOT persisted in state()/load_state(): step() runs the dynamics substep (which re-
        #     stashes it) BEFORE the thermo substep that reads it, and the None-guard falls back to cfg.wind.
        self._salt_u3_slow = None      # |u_surf|^3 EMA [m^3/s^3] -- the DE-ALIASED dynamic-salt emission wind
        #     key under diurnal_sun (#308 item 2; tau = salt_ccn_wind_tau_days). Lazily seeded on the first
        #     step (eager, pre-capture) and advanced IN PLACE at end-of-step (capture-recordable, the
        #     _T_surf_slow pattern); persisted in state()/load_state() so a resumed run's first refreshes
        #     are not phase-biased. None (and never touched) unless diurnal_sun AND salt_ccn_dynamic.
        # --- EQ-1 climatology EMAs (clim_accum): per-column true-time means of the offline surface
        #     eq-cycle's forcing fields, so eq_cycle.equilibrate_on_resume reads a matured climatology
        #     from the checkpoint instead of RE-INTEGRATING the core ~n_true_days eager (pre-capture).
        #     DIAGNOSTIC ONLY -- they never feed physics (ON vs OFF is byte-identical in every prognostic
        #     field). Lazily seeded on the first step and advanced in place at end-of-step (the
        #     _salt_u3_slow / _T_surf_slow idiom); persisted in state()/load_state(). None until seeded.
        self._clim_tsurf_slow = None   # EMA of T_surf [K]
        self._clim_qbl_slow = None     # EMA of q[:, -1] (BL specific humidity) [kg/kg]
        self._clim_wind_slow = None    # EMA of the bulk-aerodynamic exchange wind [m/s]
        self._clim_evap_slow = None    # EMA of E = _evap [kg/m^2/s]
        self._clim_precip_slow = None  # EMA of P = _precip [kg/m^2/s]
        self._clim_flash_slow = None   # EMA of _surf_flash [kg/m^2/s]
        self._clim_usurf_slow = None   # EMA of _u_surf [m/s] (None on the no-dynamics fallback)
        self._clim_seed_days = None    # true_sim_days() at first seed -- the EMA maturity clock (persisted)
        self._precip = np.zeros(c.nx)  # column condensation rate [kg/m^2/s] (M2b.3)
        self._evap = np.zeros(c.nx)    # column surface evaporation rate E = LE/L_vap [kg/m^2/s] (M2b.5)
        self._surf_flash = np.zeros(c.nx)  # surface flash-boil flux [kg/m^2/s] (surf_flash_boil; stays 0 OFF)
        # --- surf_water_budget (#302) persistent working buffers (allocated only when ON; all (nx,)-class
        #     fp64, written IN PLACE each substep -> capture-safe fixed addresses, the _surf_flash idiom) ---
        if c.surf_water_budget:
            self._p_land = np.zeros(c.nx)            # landed liquid this substep [kg/m^2/s] (lagged P - flash)
            self._W_flux = np.zeros(max(c.nx - 1, 1))  # interface creep flux [kg/m/s], + toward larger x
            self._W_ucreep_fallback = np.full(c.nx, -c.surf_water_drift * c.wind)   # no-dynamics creep [m/s]
            #   (fixed EQUATORWARD drift at the cfg exchange wind -- salt_sill's u_current; the resolved
            #    signed u_surf replaces it whenever the dynamics are live)
            if c.surf_salt:
                self._S_flux = np.zeros(max(c.nx - 1, 1))  # interface DISSOLVED-salt flux [kg/m/s] (the
                #   _W_flux idiom: persistent fp64, written in place each substep -- capture-safe address)
        self._T_boil_surf = float(th.T_boil(C.P_SURF))   # skin boiling point [K] (~372; scalar, cached so the
        #     captured surface block never recomputes/syncs it -- the same T_boil the #294 air masks use)
        self._last_dt_th = 0.0         # last thermo sub-step dt (diagnostic)
        # --- M2b.5 cumulative water-budget trackers (tier-1 water gate; OPEN_ISSUES c5: trust the
        #     CUMULATIVE, not the instantaneous). Area-weighted time integrals of the surface evaporation
        #     E and the column condensation P over the whole domain [kg/m of along-track, accumulated over
        #     time]; their ratio -> 1 at steady state (sources balance sinks). ``reset_water_budget``
        #     zeros them so a measurement window can start AFTER spin-up (the transient stores water). ---
        self._cum_evap = 0.0
        self._cum_precip = 0.0
        self._cum_surf_flash = 0.0     # cumulative surface flash-boil [same units]; 0 unless surf_flash_boil
        self._cum_p_land = 0.0         # cumulative LANDED liquid (P - flash) [same units]; 0 unless
        #     surf_water_budget -- the surface-water budget's own supply integral, sampled on the SAME
        #     lagged flux the W update consumed (so the surf_residual closure is exact, not off-by-a-substep)
        self._cum_w_snap = 0.0         # cumulative W snap-floor removals [kg/m] (surf_water_budget: the
        #     denormal-tail floor -- W < W_SNAP_FLOOR snapped to exactly 0; counted here so the surface
        #     closure stays exact to round-off)
        self._cum_w_edge = 0.0         # cumulative edge-pin WATER injection [kg/m] (surf_water_edge_source:
        #     the Dirichlet ocean pin's per-step (source_W - W[-1])*dx, signed -- so the surf_residual gate
        #     closes with the open boundary; 0 unless the flag is ON)
        self._cum_s_edge = 0.0         # cumulative edge-pin SALT injection [kg/m] (surf_salt + edge source:
        #     the pin's per-step total-salt delta at the pinned cell, signed; 0 unless both flags ON)
        self._cum_w_cryo = 0.0         # cumulative cryo-floor WATER top-up [kg/m] (surf_water_cryo_pin:
        #     the prescribed ice cover's per-substep sum(max(W_cryo - W, 0))*dx, >= 0 by construction;
        #     0 unless the flag is ON)
        self._cum_s_cryo = 0.0         # cumulative cryo-floor SALT top-up [kg/m] (surf_salt + cryo pin:
        #     the top-up water at ocean salinity, salinity0 * the water term; 0 unless both flags ON)
        self._cum_s_snap = 0.0         # cumulative SALT snap-floor removals [kg/m] (surf_salt: the
        #     denormal creep-tail floor -- S_dis/S_dep < S_SNAP_FLOOR snapped to exactly 0; counted here
        #     so the salt closure stays exact to round-off. See the S_SNAP_FLOOR module note.)
        # W-budget closure baseline: total W at the last reset_water_budget [kg/m of along-track]. The
        # machine-precision surface gate is (sum W dx - this) == cum_p_land - cum_evap (transport telescopes
        # to zero across the closed walls). 0.0 when the flag is OFF (never consulted).
        self._W_total_reset = (float(np.sum(self.W_surf * self._dx_w)) if c.surf_water_budget else 0.0)
        # #294-A ATMOSPHERIC closure baseline, the twin of _W_total_reset: total column water
        # (q + q_c + q_r) at the last reset [kg/m]. The atmospheric gate is
        #   (atm_total - this) == cum_evap + cum_surf_flash - cum_precip - cum_qcap + cum_qfloor
        # to round-off (the flux-form transport telescopes across the psi=0 walls; there is no sponge
        # export -- see the module note). Set here rather than lazily so a run that never calls
        # reset_water_budget still has an exact baseline.
        self._atm_total_reset = self._atm_total()
        # Salt-budget closure baseline: total salt (S_dis + S_dep) at the last reset [kg/m]. The salt gate
        # is (sum (S_dis+S_dep) dx - this) == cum_s_edge (transport telescopes; the saturation adjustment
        # conserves the sum exactly; evaporation moves no salt). 0.0 when surf_salt is OFF.
        self._S_total_reset = (float(np.sum((self.S_dis + self.S_dep) * self._dx_w))
                               if (c.surf_water_budget and c.surf_salt) else 0.0)
        # capture-ready (Phase B(v)): per-step budget area-sums are accumulated here OFF the host-sync path
        # and folded into _cum_* at the checkpoint cadence (``_flush_water_budget``). 0.0 when capture_ready
        # is OFF (never touched). Kept as scalars on CPU; the GPU port makes these 0-d device buffers.
        # IDIOM NOTE (#302 review finding 2): these pendings are accumulated IN PLACE into persistent 0-d
        # buffers (``_accum_pending``, promoted from the 0.0 float on the first eager pass), NOT by the old
        # rebind (``pending = pending + inc``). The rebind bakes only the LAST step's term under CUDA-graph
        # replay (the Python rebind never re-executes -- the realized-climate driver's linchpin 2), so the
        # budget integrals silently stopped accumulating across replays and the #302 surface closure
        # (``surf_residual``) would false-alarm under the captured production driver: W (a real in-place
        # state buffer) advances per replay while cum_p_land/cum_evap did not. The in-place add IS re-executed
        # by every replay, so the integrals now genuinely accumulate under capture. Eagerly value-identical
        # (same terms, same order; float() at flush unchanged).
        self._cum_evap_pending = 0.0
        self._cum_precip_pending = 0.0
        self._cum_surf_flash_pending = 0.0
        self._cum_p_land_pending = 0.0   # surf_water_budget: the landed-liquid area-sum, same deferral idiom
        self._cum_w_snap_pending = 0.0   # surf_water_budget: the snap-floor removals (see W_SNAP_FLOOR)
        self._cum_w_edge_pending = 0.0   # surf_water_edge_source: the Dirichlet-pin water delta, same idiom
        self._cum_s_edge_pending = 0.0   # surf_salt + edge source: the pin's salt delta, same idiom
        self._cum_w_cryo_pending = 0.0   # surf_water_cryo_pin: the cryo-floor water top-up, same idiom
        self._cum_s_cryo_pending = 0.0   # surf_salt + cryo pin: the cryo-floor salt top-up, same idiom
        self._cum_s_snap_pending = 0.0   # surf_salt: the salt snap-floor removals (see S_SNAP_FLOOR)
        # Positivity-clamp instrumentation (implicit advection only): accumulated count and mass of
        # clamped negative q/q_c per step, device-side under capture. Read/reset between replay batches.
        self._clamp_count = 0.0
        self._clamp_mass = 0.0
        # #294-A q-ceiling/floor accounting (``_clip_q``): the water the physical [0, th.Q_MAX] clip
        # destroys / invents [kg/m]. CUMULATIVE budget terms (checkpointed + reset by reset_water_budget,
        # exactly like _cum_evap), NOT the read+reset _clamp_* instrumentation -- they must close the
        # column budget over the same window the E/P cums measure.
        self._cum_qcap = 0.0
        self._cum_qfloor = 0.0
        # ...and the water the IMPLICIT-ADVECTION positivity clamp injects (_clamp_tracer_positive).
        # Same ledger, different call site; identically 0.0 on the explicit path (no clamp there).
        self._cum_qadv = 0.0
        self._cum_qcap_pending = 0.0
        self._cum_qfloor_pending = 0.0
        self._cum_qadv_pending = 0.0
        # capture-ready (Phase E): the TRUE-time accumulation deferred on-device (dt is a 0-d device scalar on
        # the capture path; ``float(dt)`` in step() would sync). Accumulated IN PLACE via ``_accum_pending``
        # (the shared #302 budget idiom) so it advances per captured replay, and folded into ``sim_time`` at
        # the flush cadence (``_flush_sim_time``). 0.0 when capture_ready is OFF (never touched); ``_accum_pending``
        # promotes this float to a fixed 0-d device buffer on the first eager pass.
        self._sim_time_pending = 0.0
        self._iter_pending = 0.0           # the step-COUNT twin of _sim_time_pending (+1.0 per captured
        #     step, same in-place idiom): ``self.iter += 1`` is host code, so under CUDA-graph replay it
        #     froze at warmup+1 (run6: iter=11 at nominal day 380) -- the #327 sim_time fix, extended to
        #     the counter. Folded by ``_flush_sim_time``.
        self.iter = 0
        self.sim_time = 0.0                # cumulative TRUE physical time [s] = Sigma(adaptive dt); the
        #     dynamics dt is CFL-adaptive (_cfl_dt) and runs WELL below dt_dyn, so iter*dt_dyn overcounts
        #     physical time 3-12x on production grids -- track the real elapsed time here (persisted in state()).

        # --- PHASE D: place the COLUMN-THERMO state AND the fixed grid arrays it touches ON DEVICE ---
        # The coupled column physics (radiation/surface/thermo/microphysics/convadjust/advection) is now
        # array-module-generic, so on device="gpu" we make the whole coupled step() device-resident. The
        # central design move (per the port audit): pre-place EVERY fixed array the thermo hot path meets a
        # state array with -- p_edge/p_mid/dm/sigma/C_layer/S/S_diffuse/S_sw_in/T_ref/z/_dlnp/RH/_dz_cell --
        # so a host (nz,)/(nx,) fixed array never collides with a device state array (the mixed-input trap:
        # numpy x CuPy raises). Gated on _on_device so the CPU path is BYTE-IDENTICAL (the arrays stay numpy,
        # every reference is the verbatim numpy op). Done AFTER the state arrays are built above so we move the
        # already-seeded values. The host-only CONSUMERS (state(), diagnostics(), the conservation gates,
        # _aloft_moisture_column) bring fields back with to_host(), so the device placement does not break them.
        if self._on_device:
            self._place_thermo_on_device()
        if self._dyn_on_device:
            self._place_dynamics_on_device()   # Phase E: H2D-free dynamics substep (capture prerequisite)

    def _build_salt_ccn_field(self):
        """(Re)build the salt-aerosol CCN droplet-number field ``self.N_d`` [cm^-3] from cfg (#237). ``None``
        when ``salt_ccn`` is OFF (byte-identical). Two sources (``salt_ccn_source``):

         * 'prescribed' (DEFAULT, the #243 AD-core sweep): a PRESCRIBED (nx,) field built in-core: uniform at
           ``salt_ccn_nd`` everywhere, or (``salt_ccn_band_only``) the loaded value within [theta_lo, theta_hi]
           and the reference value elsewhere. ``salt_ccn_nd`` None -> N_ref (the master-ON no-op). OPT-IN
           override ``salt_ccn_nd_file`` (env ``COSMO_SALT_CCN_ND_FILE``, default unset): load an arbitrary
           (nx,) PROFILE from a .npy -- the capture-safe carrier for the emergent ``salt_transport`` field
           built offline from a converged wind (the in-step 'transport' refresh is capture-hostile). Runtime
           load (in ``_build_salt_ccn_field``, not module scope), so no import-timing forcing tax.
         * 'transport' (the #237 source/transport scheme): EMERGENT -- built by ``salt_transport`` from the
           resolved lowest-level wind ``self._u_surf`` (reduced to 10 m), so the flats' wind-keyed salt
           emission, size-binned settling, and poleward-aloft transport set N_d at the band. Before the wind
           is resolved (``_u_surf`` None, e.g. at construction) the field is CLEAN (N_d == N_ref, an exact
           no-op), so a cold start is a no-op until refreshed.

        Only N_d/N_ref enters the limbs. Idempotent and re-callable AFTER construction (the ``_salt_ccn_sweep``
        driver mutates cfg then re-invokes this; the transport refresh re-invokes it from the live wind):
        builds a host array and places it on the compute device when the core is already on-device. In
        ``__init__`` the core is not yet on-device, so it returns a host array that the later
        ``_place_thermo_on_device`` moves -- byte-identical to the old inline construction."""
        c = self.cfg
        self._micro_warm = {}            # eagerly drop the fused warm-factor cache (belt-and-suspenders: the
        #                                  builder in _micro_fuse_salt_cond_params also catches the N_d swap by identity)
        if not c.salt_ccn:
            self.N_d = None
            return
        if c.salt_ccn_source == "transport":
            N_d = self._salt_ccn_transport_field()
        elif c.salt_ccn_nd_file:
            # OPT-IN prescribed PROFILE from a .npy (nx,) file -- the capture-safe carrier for the emergent
            # salt_transport field built offline from a converged wind (the in-step 'transport' refresh is
            # capture-hostile). Fixed at construction, so the captured graph sees a constant field.
            N_d = np.load(c.salt_ccn_nd_file).astype(self._dtype)
            if N_d.shape != (c.nx,):
                raise ValueError(f"COSMO_SALT_CCN_ND_FILE {c.salt_ccn_nd_file!r}: shape {N_d.shape} != (nx={c.nx},)")
            # Value guard: a corrupt/negative/NaN N_d propagates silently into (N_d/N_ref)**beta
            # (beta<0 -> NaN/inf warm factors) and the Twomey r_eff -- catch it at construction, loudly.
            if not np.all(np.isfinite(N_d)) or float(N_d.min()) <= 0.0:
                raise ValueError(
                    f"COSMO_SALT_CCN_ND_FILE {c.salt_ccn_nd_file!r}: N_d must be finite and > 0 "
                    f"(min={float(np.nanmin(N_d)):.3g}, has_nonfinite={not bool(np.all(np.isfinite(N_d)))})")
        else:
            nd = c.salt_ccn_nd if c.salt_ccn_nd is not None else c.salt_ccn_nd_ref
            N_d = np.full(c.nx, float(nd), dtype=self._dtype)
            if c.salt_ccn_band_only:
                th_deg = np.asarray(to_host(self.theta_deg))
                outband = (th_deg < c.salt_ccn_band_theta_lo) | (th_deg > c.salt_ccn_band_theta_hi)
                N_d[outband] = float(c.salt_ccn_nd_ref)
        self.N_d = to_device(N_d, self.xp) if getattr(self, "_on_device", False) else N_d

    def _salt_ccn_transport_field(self):
        """Build the EMERGENT transport N_d host array [cm^-3] (#237). Returns the clean reference field when
        the wind is not yet resolved (``_u_surf`` None), so a cold start / dynamics-off core is an exact no-op."""
        from . import salt_transport as stx
        c = self.cfg
        u_surf = getattr(self, "_u_surf", None)
        if u_surf is None:
            return np.full(c.nx, float(c.salt_ccn_nd_ref), dtype=self._dtype)
        st_cfg = stx.SaltTransportConfig(
            flats_theta_hi=c.salt_ccn_transport_flats_theta_hi,
            alpha_sandblast=c.salt_ccn_transport_alpha,
            v_dep_min=c.salt_ccn_transport_vdepmin,
            n_d_ref=c.salt_ccn_nd_ref,
            z1=getattr(self, "_z1", None),
        )
        th = np.asarray(to_host(self.theta_deg), float)
        xm = np.asarray(to_host(self.x), float)
        u = np.asarray(to_host(u_surf), float)
        N_d, _ = stx.nd_field_from_u_surf(th, xm, u, cfg=st_cfg)
        return N_d.astype(self._dtype)

    def refresh_salt_ccn(self, dt_days=None, log=True):
        """Cadenced DYNAMIC salt-CCN refresh (``salt_ccn_dynamic``; the salt-cloud thermostat -- full design
        note on the config flag). Recomputes the emergent N_d TARGET from the LIVE state (dry-surface mask x
        wind-keyed emission -> phase-2 transport -> precip scavenging), relaxes the live ``N_d`` toward it
        with the ``salt_ccn_tau_days`` aerosol memory, and rewrites the held ``N_d`` buffer AND every derived
        consumer buffer IN PLACE (addresses stable -> captured CUDA graphs replay the refreshed values).

        CALL DISCIPLINE (the capture-hostility lesson, MODELING_LESSONS/#237): this does HOST work (device
        pulls, numpy salt_transport) -- the DRIVER calls it at a replay-batch boundary (stream synchronized,
        outside any captured region) every ``salt_ccn_refresh_days``; it must NEVER run inside ``step()`` or
        during graph capture. The old per-step 'transport' refresh died on exactly this: (a) host ops inside
        the captured step, (b) ``self.iter``-gated -- Python doesn't re-execute under replay, so the gate
        never fired, and (c) ``_build_salt_ccn_field`` REASSIGNS ``N_d``, orphaning the address the graph
        recorded. (a) is solved by the call site, (b) by the driver's sim-day clock, (c) by in-place writes.

        ``dt_days``: sim-days since the previous refresh (the relaxation interval); None -> the configured
        cadence ``salt_ccn_refresh_days``. The FIRST call on an un-initialized field (no ND_FILE, no
        checkpointed N_d) jumps straight to the target (init-from-seed-state). Returns a small diagnostics
        dict (n_dry, target/live stats) for the driver log and tests."""
        c = self.cfg
        if not (c.salt_ccn and c.salt_ccn_dynamic):
            raise RuntimeError("refresh_salt_ccn requires salt_ccn + salt_ccn_dynamic (set COSMO_SALT_CCN=1 "
                               "COSMO_SALT_CCN_DYNAMIC=1 in the shell before construction).")
        self._salt_refresh_count = getattr(self, "_salt_refresh_count", 0) + 1   # #308 item 4 (the
        #   silent-static warn in step() keys on this staying 0)
        from . import salt_transport as stx
        # --- 1. dry-source mask: VALUE-LEVEL wetness (works under BOTH wetness models -- the legacy
        #     T-ramp/budget branches and the #302 W-based wetness; _wetness() is the single authority) ---
        wet = np.asarray(to_host(self._wetness()), dtype=np.float64)
        dry = wet < c.salt_ccn_dry_thresh
        # --- 2. the live lowest-level wind. ``_u_surf`` is stashed by the dynamics substep; before any step
        #     (e.g. the driver's pre-capture init refresh, right after load_state) reconstruct it from the
        #     loaded psi exactly as the driver diagnostics do (u[1,:] is the _u_surf convention). ---
        u_surf = getattr(self, "_u_surf", None)
        if u_surf is None:
            u_all, _w = velocities_from_psi(self.psi, self._beta, self._dx_op, self._dz_op)
            u_surf = u_all[1, :]
        elif c.capture_ready and self.run_dynamics:
            # #308 item 3 -- INPUT-LIVENESS assert. ``_u_surf`` is this refresh's one NON-PERSISTENT-
            # buffer input: a view stashed by the capture-time dynamics substep, whose BASE buffer the
            # replayed graph rewrites in place (the view keeps the base alive, so the stash tracks the
            # replays). If a refactor ever breaks that liveness (fresh per-replay allocations, a pool
            # behavior change), the refresh would silently key the emission on a FROZEN wind -- so
            # cross-check against a fresh eager recompute from the held ``psi`` (same deterministic
            # kernels, same input -> must match). Cadenced host-side work: one extra velocities solve
            # per refresh (~0.5 sim-day), never inside step()/capture.
            u_all, _w = velocities_from_psi(self.psi, self._beta, self._dx_op, self._dz_op)
            xp_u = get_xp(u_all)
            u_ref = u_all[1, :]
            # NON-FINITE FIRST (don't misattribute a blow-up to capture staleness). A NaN/Inf wind here
            # means the DYNAMICS went non-finite UPSTREAM (classically the explicit vertical tracer/omega
            # limb outside the CFL -- DT=60 on a stretched/resolved axis needs COSMO_TRACER_VERT_IMPLICIT=1,
            # STRETCHX_CAMPAIGN.md P2b). ``allclose(nan, .)`` is False, so WITHOUT this check the staleness
            # branch below fires and sends a debugger chasing the capture that is not broken -- report the
            # blow-up honestly instead. (The downstream non-finite-N_d guard would also catch it, but only
            # after this misleading message.)
            finite = (bool(xp_u.all(xp_u.isfinite(u_ref)))
                      and bool(xp_u.all(xp_u.isfinite(u_surf))))
            if not finite:
                raise FloatingPointError(
                    "refresh_salt_ccn: the surface wind is NON-FINITE -- the dynamics blew up UPSTREAM "
                    "(a numerical blow-up, NOT a capture-liveness failure). Common cause: DT too large "
                    "for the explicit vertical tracer/omega limb on a stretched/resolved axis -- set "
                    "COSMO_TRACER_VERT_IMPLICIT=1 (STRETCHX_CAMPAIGN.md P2b) or lower COSMO_DT.")
            if not bool(xp_u.allclose(u_ref, u_surf, rtol=1e-6, atol=1e-9)):
                raise RuntimeError(
                    "refresh_salt_ccn: stashed _u_surf does not match the wind recomputed from the "
                    "held psi -- the capture-time stash has gone STALE under replay (#308 item 3); "
                    "the emission would key on a frozen wind. Rebuild the capture / fix the liveness.")
        u = np.asarray(to_host(u_surf), dtype=np.float64)
        u3s = getattr(self, "_salt_u3_slow", None)
        if c.diurnal_sun and u3s is not None:
            # #308 item 2 -- under the diurnal sun the instantaneous sample is PHASE-LOCKED at the
            # refresh cadence (0.5 day against a 1-day cycle) and O(1)-biased through the Owen u*^3
            # convexity; key on the cube root of the |u|^3 EMA instead (its time-mean preserves the
            # emission-relevant cubic moment; the transport takes |u|, so no sign is lost). The EMA
            # is None until the first step under BOTH flags -- the driver's pre-capture init refresh
            # then reads the instantaneous wind once (the documented init-from-seed jump).
            u = np.cbrt(np.maximum(np.asarray(to_host(u3s), dtype=np.float64), 0.0))
        # --- 3. emission + transport: the phase-2 salt_transport pipeline (Shao&Lu threshold + Owen u*^3
        #     saltation + sandblasting alpha + loft/advection-settling reach), with the LIVE mask replacing
        #     the static pan and emis_scale (the phase-2-anchored normalization) scaling the emission. ---
        st_cfg = stx.SaltTransportConfig(
            flats_theta_hi=c.salt_ccn_transport_flats_theta_hi,
            alpha_sandblast=c.salt_ccn_transport_alpha * c.salt_ccn_emis_scale,
            v_dep_min=c.salt_ccn_transport_vdepmin,
            n_d_ref=c.salt_ccn_nd_ref,
            z1=getattr(self, "_z1", None),
        )
        th_deg = np.asarray(to_host(self.theta_deg), float)
        xm = np.asarray(to_host(self.x), float)
        N_raw, tdiag = stx.nd_field_from_u_surf(th_deg, xm, u, cfg=st_cfg, flats_mask=dry)
        # --- 4. sink: precipitation scavenging. Rain removes aerosol at rate P/P_scav per day (one e-fold
        #     per scav_pmm mm of rain), in steady competition with the tau-day source/lifetime:
        #     excess_ss = excess_raw / (1 + tau*P/P_scav). Cadence-independent (a steady balance, not a
        #     per-refresh decrement). ``_precip`` is the lagged surface-reaching flux [kg/m^2/s]. ---
        P_mmday = np.maximum(np.asarray(to_host(self._precip), dtype=np.float64), 0.0) * 86400.0
        scav = 1.0 / (1.0 + c.salt_ccn_tau_days * P_mmday / c.salt_ccn_scav_pmm)
        N_tgt = c.salt_ccn_nd_ref + (np.asarray(N_raw, np.float64) - c.salt_ccn_nd_ref) * scav
        # --- 5. relax the live field toward the target (exact geometric relaxation; fp64 on host) ---
        cur = np.asarray(to_host(self.N_d), dtype=np.float64)
        if not getattr(self, "_salt_dyn_initialized", False):
            new = N_tgt.copy()                                   # init-from-seed-state (no profile to relax from)
            self._salt_dyn_initialized = True
        else:
            dd = float(dt_days) if dt_days is not None else float(c.salt_ccn_refresh_days)
            # #308 item 1: source lifetime and scavenging act in PARALLEL, so the approach to the
            # scavenged equilibrium runs at (1/tau + P/P_scav) per column -- the scheme's own ODE
            # rate (the bare 1/tau lagged the sink 3-6x under band rain; the equilibrium N_tgt
            # above is unchanged).
            rate = 1.0 / c.salt_ccn_tau_days + P_mmday / c.salt_ccn_scav_pmm     # [1/day]
            new = N_tgt + (cur - N_tgt) * np.exp(-max(dd, 0.0) * rate)
        new = np.maximum(new, 1e-3)                              # the ND_FILE loader's N_d > 0 invariant
        if not np.all(np.isfinite(new)):
            raise FloatingPointError("refresh_salt_ccn: non-finite N_d (check wind/precip state).")
        # --- 6. IN-PLACE writes: the held N_d buffer + every derived consumer buffer. NB coerce through
        #     to_device -- cupy's `dev[...] = numpy` routes to fill() and throws (the W_surf lesson). ---
        self.N_d[...] = to_device(new.astype(self.N_d.dtype, copy=False), get_xp(self.N_d))
        self._refresh_salt_ccn_consumers()
        out = dict(n_dry=int(dry.sum()), wet_min=float(wet.min()),
                   u_surf_max=float(np.abs(u).max()),   # max |wind key| at z1 (~1.2 km) -- the 10-m
                   #   reduction happens inside salt_transport (was mislabeled ``u10_max``)
                   tgt_max=float(N_tgt.max()), tgt_med=float(np.median(N_tgt)),
                   live_max=float(new.max()), live_med=float(np.median(new)),
                   N_tgt=N_tgt, N_live=new, transport_diag=tdiag)
        if log:
            print(f"[salt_ccn_dynamic] refresh: n_dry={out['n_dry']}/{c.nx} "
                  f"target max/med={out['tgt_max']:.0f}/{out['tgt_med']:.0f} "
                  f"live max/med={out['live_max']:.0f}/{out['live_med']:.0f} cm^-3 "
                  f"(dt={float(dt_days) if dt_days is not None else c.salt_ccn_refresh_days:.2f}d "
                  f"tau={c.salt_ccn_tau_days:.1f}d)", flush=True)
        return out

    def _refresh_salt_ccn_consumers(self):
        """Recompute every DERIVED N_d consumer buffer IN PLACE after an in-place N_d refresh. The live
        consumers and why each needs (or doesn't need) work here:

          * Twomey cloud-albedo limb (``_radiate``: ``cloud_n_d = self.N_d[:, None]``) -- a VIEW of the held
            buffer; eager calls re-read it and captured graphs recorded its device address -> nothing to do.
          * salt-haze SW limb (``_radiate``: ``sw_tau_haze`` from ``self.N_d``) -- computed live per call /
            recorded as device ops on the held buffer -> nothing to do.
          * per-column ``cond_tau_salt`` tau(x) -- computed live from ``self.N_d`` inside the generic micro
            path every call -> nothing to do.
          * the fused micro kernels' warm-rain autoconv factor ``(N_d/N_ref)^beta`` -- CACHED on the core
            (``_micro_warm``, address-stable by design; ``_micro_fuse_salt_cond_params``) and STALENESS-
            TRACKED BY N_d IDENTITY, which an in-place refresh deliberately does NOT change -> this is the
            one buffer that MUST be rewritten here, in place, per cached dtype."""
        c = self.cfg
        cache = getattr(self, "_micro_warm", None)
        if not cache:
            return
        for key, (warm, nd_src) in list(cache.items()):
            _dtype_str, apply_warm = key
            if not apply_warm:
                continue                                        # the all-ones placeholder is never read
            xp_w = get_xp(warm)
            nd = th._F(self.N_d, xp_w)                          # the exact _micro_fuse_salt_cond_params recipe
            warm[...] = xp_w.ascontiguousarray((nd / c.salt_ccn_nd_ref) ** c.salt_ccn_autoconv_beta,
                                               dtype=warm.dtype)
            cache[key] = (warm, self.N_d)                       # re-pin the tracked identity (same object)

    def _place_thermo_on_device(self):
        """Move the column-thermo prognostic state and the fixed grid arrays it meets onto ``self.xp``
        (CuPy) so the coupled ``step()`` runs device-resident (Phase D). Called once from ``__init__`` only
        when ``device='gpu'``. The fixed arrays are placed so that ``self.dm[None, :]`` etc. produce a DEVICE
        array wherever they multiply a device state field (no per-site ``xp.asarray`` and no mixed-input
        crash). Idempotent-safe: ``to_device`` is a no-op on an already-device array."""
        xp = self.xp
        # prognostic / derived state the coupled step writes (kept at the working dtype, fp64 default)
        self.T_surf = to_device(self.T_surf, xp)
        self.T_atm = to_device(self.T_atm, xp)
        self.q = to_device(self.q, xp)
        self.q_c = to_device(self.q_c, xp)
        self._bl_deck_cover = to_device(self._bl_deck_cover, xp)
        self._precip = to_device(self._precip, xp)
        self._evap = to_device(self._evap, xp)
        self._surf_flash = to_device(self._surf_flash, xp)
        for _cn in ("_clim_tsurf_slow", "_clim_qbl_slow", "_clim_wind_slow", "_clim_evap_slow",
                    "_clim_precip_slow", "_clim_flash_slow", "_clim_usurf_slow"):
            _cv = getattr(self, _cn, None)   # EQ-1 climatology EMAs: None until seeded / on a legacy ckpt
            if _cv is not None:
                setattr(self, _cn, to_device(_cv, xp))
        if self.W_surf is not None:                      # #302 surf_water_budget state + working buffers
            self.W_surf = to_device(self.W_surf, xp)     #   (fp64 on device too -- never the working dtype)
            self._p_land = to_device(self._p_land, xp)
            self._W_flux = to_device(self._W_flux, xp)
            self._W_ucreep_fallback = to_device(self._W_ucreep_fallback, xp)
            if self.S_dis is not None:                   # surf_salt state + its flux buffer (fp64 likewise)
                self.S_dis = to_device(self.S_dis, xp)
                self.S_dep = to_device(self.S_dep, xp)
                self._S_flux = to_device(self._S_flux, xp)
        if self.q_r is not None:                         # prognostic rain (stage 7): placed exactly as q_c
            self.q_r = to_device(self.q_r, xp)
        # fixed grid arrays the thermo hot path multiplies a device state with (fp64 -- they set/preserve the
        # absolute scale; the working-dtype cast lives on the state stores). Placing them on device kills the
        # mixed-input trap at the radiation boundary (column_radiation infers xp from p_edge / q), in
        # _advect_heat (p_mid x device theta), and across the microphysics (p_mid/dm x device q).
        # PHY-22 B' (``buoyancy_pressure_term``): the four fixed rows ``_update_buoyancy`` reads when the
        # term is ON must be placed too. ``_p_mid_row`` in particular is a COPY of the HOST ``p_mid``
        # taken at build time, so it would NOT follow ``p_mid`` onto the device and every step would
        # re-upload it inside ``_interp_columns_pair``'s ``asarray`` -- eager-legal but CAPTURE-FATAL
        # (the same H2D trap ``_place_dynamics_on_device`` documents). They are placed unconditionally,
        # like every other fixed row here, so the OFF path is unaffected either way.
        for name in ("p_edge", "p_mid", "dm", "sigma", "C_layer", "S", "T_ref", "z", "_dlnp", "RH",
                     "_p_mid_row", "_inv_p_ref", "_buoy_alpha", "_buoy_kcol"):
            setattr(self, name, to_device(getattr(self, name), xp))
        if self.S_diffuse is not None:
            self.S_diffuse = to_device(self.S_diffuse, xp)
            self.S_sw_in = to_device(self.S_sw_in, xp)   # S + S_diffuse: a distinct array, place it too
        else:
            self.S_sw_in = self.S                        # alias to the now-device S (sw_diffuse OFF)
        if self.cfg.diurnal_sun:
            # DIURNAL SUN (chunk 4.1/4.2): the one-day tables + the held row buffers (fixed device
            # addresses -- the captured graph gathers into _S_now in place), the (1,) float64 device
            # diurnal CLOCK the capture path accumulates the realized dt into (in-place add; a 0-d rebind
            # would not accumulate under replay), and the slow-wetness keying state (finding 2).
            self._S_table = to_device(self._S_table, xp)
            self._S_now = to_device(self._S_now, xp)
            if self._S_diff_table is not None:
                self._S_diff_table = to_device(self._S_diff_table, xp)
                self._S_diff_now = to_device(self._S_diff_now, xp)
            if self._T_surf_slow is not None:
                self._T_surf_slow = to_device(self._T_surf_slow, xp)
            if self._wetness_frozen is not None:
                self._wetness_frozen = to_device(self._wetness_frozen, xp)
            if self._E_w_slow is not None:
                self._E_w_slow = to_device(self._E_w_slow, xp)
            if self._diurnal_time_dev is None:               # keep the clock's address across a re-place
                self._diurnal_time_dev = xp.zeros(1, dtype=xp.float64)   # (load_state re-invokes this method)
        if self._dz_cell is not None:
            self._dz_cell = to_device(self._dz_cell, xp)
        if self._dx_cell is not None:                          # stretched-x: budget weight + flux divisor
            self._dx_cell = to_device(self._dx_cell, xp)       #   (pond fill, W-transport, vapor mass, sums)
            self._dx_w = self._dx_cell                         # re-alias the views onto the device array
            self._dx_div = self._dx_cell[:, None]
        if self.N_d is not None:                         # salt-CCN field (#237); None when salt_ccn OFF
            self.N_d = to_device(self.N_d, xp)

    def _place_dynamics_on_device(self):
        """Move the fixed DYNAMICS reference arrays onto ``self.xp`` so the resolved dynamics substep is
        H2D-FREE (Phase E / CUDA-graph capture prerequisite). Called once from ``__init__`` only when
        ``_dyn_on_device`` (device='gpu' AND poisson_method='dst'). Without this, the per-substep
        ``velocities_from_psi(psi, self._beta, ...)`` and ``vorticity_tendency(..., self.dlnrho_dz, ...)``
        re-upload the host (nz,) reference profiles to device EVERY step -- a host->device transfer that is
        eager-legal (the eager parity ran fine) but ILLEGAL inside a CUDA-graph capture (it aborts the
        capture). Placing them once makes the substep read them device-resident, exactly the Phase-D thermo
        placement applied to the dynamics limb. Idempotent (to_device no-ops an already-device array).

        ``_dz_op`` is the vertical-operator spacing: on a UNIFORM grid it is the SCALAR ``self.dz`` (no
        device array -> the stencils take a python float, no H2D), so it is left as-is; on a STRETCHED grid it
        was set to ``self.z`` BEFORE the thermo placement moved ``self.z`` to device, so re-alias it to the
        now-device ``self.z`` (the operators then index a device spacing array, not the stale host one)."""
        xp = self.xp
        self._beta = to_device(self._beta, xp)                 # 1/rho_ref (nz,) -> velocities_from_psi
        self.dlnrho_dz = to_device(self.dlnrho_dz, xp)         # anelastic stretching coeff (nz,) -> vorticity_tendency
        self.rho_ref_dev = to_device(self.rho_ref, xp)         # (nz,) for the device mass-conservation / strain limbs
        if getattr(self, "_salt_u3_slow", None) is not None:   # the restored wind-key EMA rides the
            self._salt_u3_slow = to_device(self._salt_u3_slow, xp)   # dynamics device (u_surf's module)
        if np.ndim(self._dz_op) != 0:                          # stretched grid: _dz_op was self.z (now on device)
            self._dz_op = self.z                               # re-alias to the device-placed self.z
        # _dz_cell already placed by _place_thermo_on_device (the _vint integral weight).
        if self._x_stretched:                                  # stretched-x: the operators' centers array and
            self._dx_op = to_device(self._dx_op, xp)           #   the per-cell CFL row live on device too
            self._inv_dxw_row = to_device(self._inv_dxw_row, xp)
            #   (self.x stays HOST -- init-time consumers and diagnostics read it; _dx_op is the device
            #   twin the per-substep operator calls take, exactly the _dz_op/self.z split.)

    # ------------------------------------------------------------------ precision helpers (fp32 spike)
    def _obuf(self, current):
        """Capture-ready store target (Phase A): return ``current`` (the pre-allocated prognostic buffer, to
        write into IN PLACE for a fixed graph address) when ``capture_ready``, else ``None`` (the verbatim
        shipped rebind). One choke-point so every funnelled store site reads ``out=self._obuf(self.X)``."""
        return current if self.cfg.capture_ready else None

    def _cast(self, arr, out=None):
        """Cast a prognostic-state array to the working dtype ``self._dtype`` (np.float64 -> a no-op that
        preserves byte-identity; np.float32 -> single precision). Device-agnostic: it uses the array's own
        module (``get_xp``) so a CuPy field stays on the GPU. The single seam every prognostic store goes
        through after thermo/buoyancy/advection arithmetic (which upcasts to fp64 via the fp64-internal
        external modules / fp64 constants) so the STATE stays at the configured precision.

        CAPTURE-READY ``out=`` (Phase A): when ``out`` is a pre-allocated buffer, write the cast result
        IN PLACE into it (``out[...] = arr`` after the dtype cast) and return ``out`` -- so the prognostic
        store keeps a FIXED device address across the step (a captured graph records addresses; a rebind
        allocates a new one). The VALUE is identical to the rebinding return path (same astype / same copy),
        so OFF-vs-ON is bit-identical. ``out is None`` is the verbatim shipped behaviour."""
        if out is None:
            if self._dtype is np.float64:
                return arr                                        # byte-identical fast path (no cast/copy)
            return get_xp(arr).asarray(arr, dtype=self._dtype)
        # in-place: cast (a no-op view under fp64) then copy the SAME bits into the fixed buffer.
        out[...] = arr if self._dtype is np.float64 else get_xp(arr).asarray(arr, dtype=self._dtype)
        return out

    def _accumulate(self, field, increment, out=None):
        """Apply a prognostic tendency increment to ``field`` honouring the precision config:

          * fp64, or fp32 with ``fp32_accumulate_fp64`` OFF: plain add, then cast to the working dtype
            (the fp32-add limb -- exposes the small-increment round-off stall the sub-flag fixes).
          * fp32 with ``fp32_accumulate_fp64`` ON: accumulate in fp64 then round once to fp32
            -- (field.astype(f64) + increment.astype(f64)).astype(f32) -- the standard stall guard.

        ``increment`` is the FULL dt*tendency (already includes dt). On the fp64 path this is just
        ``field + increment`` (byte-identical).

        CAPTURE-READY ``out=`` (Phase A): when ``out`` is a pre-allocated buffer, write the result IN PLACE
        into it and return ``out`` (fixed address for graph capture). The arithmetic is performed exactly as
        the rebinding path (same op order, same dtype) then stored into the buffer, so OFF-vs-ON is bit-
        identical. Elementwise add is alias-safe (``out`` may be ``field`` itself). ``out is None`` is the
        verbatim shipped behaviour."""
        if self._dtype is np.float64:
            if out is None:
                return field + increment
            xp = get_xp(field)
            return xp.add(field, increment, out=out)              # in-place, bit-identical to field+increment
        if self.cfg.fp32_accumulate_fp64:
            xp = get_xp(field)
            res = xp.asarray((xp.asarray(field, dtype=np.float64)
                              + xp.asarray(increment, dtype=np.float64)), dtype=self._dtype)
            if out is None:
                return res
            out[...] = res
            return out
        return self._cast(field + increment, out=out)

    # ------------------------------------------------------------------ reference state (#0)
    def _reference_profile(self, z):
        """Anelastic reference state on height grid ``z`` [m]: a hydrostatically-balanced,
        horizontally-uniform base profile. Constant-lapse troposphere T = T_surf - lapse*z with the
        EXACT hydrostatic pressure p = P_SURF (T/T_surf)^(g/(R lapse)); an isothermal cap T = T_top with
        p = p(z_iso) exp(-g (z-z_iso)/(R T_top)) above the height z_iso where T hits the floor. Returns
        (p_ref [Pa], T_ref [K]); rho_ref = p_ref/(R T_ref) then satisfies dp_ref/dz = -rho_ref g exactly
        (the analytic forms are the hydrostatic solutions), so the state is hydrostatic-consistent.

        Note: at the default p_top/lapse/T_ref_top the isothermal cap sits ABOVE the height-grid top
        (z_iso ~= 23 km > z_top ~= 20 km), so the height grid stays purely in the lapse branch -- the cap
        branch is exercised only by the fine-grid inverter ``_z_of_ref_pressure`` and is there for
        robustness if those inputs change."""
        c = self.cfg
        z = np.asarray(z, float)
        z_iso = (c.T_ref_surf - c.T_ref_top) / c.lapse_ref         # height where T reaches the floor [m]
        T = np.maximum(c.T_ref_surf - c.lapse_ref * z, c.T_ref_top)
        expo = C.G_ACCEL / (C.R_DRY * c.lapse_ref)
        T_lapse = c.T_ref_surf - c.lapse_ref * np.minimum(z, z_iso)  # clamp to the lapse region for p_lapse
        p_lapse = C.P_SURF * (T_lapse / c.T_ref_surf) ** expo
        p_iso_base = C.P_SURF * (c.T_ref_top / c.T_ref_surf) ** expo
        p_iso = p_iso_base * np.exp(-C.G_ACCEL * (z - z_iso) / (C.R_DRY * c.T_ref_top))
        p = np.where(z <= z_iso, p_lapse, p_iso)
        return p, T

    def _z_of_ref_pressure(self, p):
        """Height z [m] at which the reference profile reaches pressure ``p`` [Pa] (inverts the
        analytic reference profile on a fine grid). Used to set the height-grid top from p_top."""
        zf = np.linspace(0.0, 100.0e3, 200001)
        pf, _ = self._reference_profile(zf)
        return float(np.interp(np.log(float(p)), np.log(pf)[::-1], zf[::-1]))

    def _brunt_vaisala_sq(self, T_ref, z):
        """Diagnosed background stratification N^2(z) = (g/theta_ref) dtheta_ref/dz [1/s^2] of the
        reference profile (potential temperature theta_ref = T_ref (P_SURF/p_ref)^kappa). Diagnostic
        (#2: N^2 is diagnosed, not prescribed); positive => stable reference."""
        theta_ref = T_ref * (C.P_SURF / self.p_ref) ** C.KAPPA_POISSON
        dthdz = np.gradient(theta_ref, z)
        return C.G_ACCEL / theta_ref * dthdz

    # ------------------------------------------------------------------ vertical-grid reconciliation (#1)
    def _hydrostatic_z(self, T_atm=None):
        """Layer-center heights z [m] of the PRESSURE grid, per column, from the hydrostatic
        (hypsometric) relation applied to the live temperature. Shape (nx, nz), ordered TOP->SURFACE to
        match ``T_atm`` (so z[:, nz-1] ~ 0 at the surface, z[:, 0] ~ z_top aloft). Lifted from Stage A;
        this is the bridge between the pressure grid (thermo) and the height grid (dynamics).

        Vectorized over the interfaces (issue #99), BIT-IDENTICAL to the original per-interface Python
        loop: the hypsometric increments are formed in one shot against the precomputed log-pressure
        steps ``self._dlnp`` (same per-element IEEE ops in the same grouping; ``_dlnp`` was computed
        with the loop's own scalar ``np.log`` calls, see ``__init__``) and accumulated surface-up by
        ``np.cumsum`` -- the same sequential left-to-right accumulation the loop performed (cumsum is
        ufunc ``accumulate``, defined sequential, never pairwise).

        DENSITY TEMPERATURE IN THE THICKNESS (ROADMAP PHY-21, ``cfg.hydrostatic_virtual_t``, default OFF):
        the hypsometric relation ``dz = (R_d T / g) dln p`` is exact only on the DENSITY (virtual)
        temperature -- ``p = rho R_d T_rho`` is what the ideal gas law says for moist air -- so on the
        default path every layer of a moist column is UNDER-THICK by exactly the factor
        ``T/T_rho = 1/(1 + 0.608 q - q_cond)``, and its thermo state lands too LOW on the dynamics height
        grid. That is an OMISSION, not one of the anelastic approximations: the anelastic approximation
        lives in the dry REFERENCE state (``rho_ref``/``T_ref``/``p_ref``), while this function is a
        diagnostic of the LIVE column's actual geometry (see the config comment for the full argument and
        the measured sizes -- the 500 hPa level is 55-240 m too low on the canonical state).
        ON, the SAME integral runs on ``T_rho``, built from the live ``self.q`` (clipped to
        ``[0, Q_MAX]``, exactly as ``_update_buoyancy`` clips it) through the SAME ``thermo`` entry
        points the buoyancy uses, honouring ``cfg.virtual_temp_exact`` (which formula) and
        ``cfg.condensate_loading`` (whether the suspended cloud+rain mass is carried) -- so the bridge and
        the buoyancy can never disagree about what the density temperature is. OFF, ``Tsrc`` IS
        ``self.T_atm`` and every operation below is untouched, so the path is BYTE-IDENTICAL. The branch
        is at the PYTHON level (a CUDA graph records the branch that ran at capture and a replay never
        re-enters this Python -- so on a capture_ready run set the flag BEFORE the first step); it is
        re-read every call, so a post-build set takes effect on the next EAGER step.

        MOISTURE SOURCE: ``self.q``/``self.q_c``/``self.q_r``, never a passed-in argument. Every in-step
        caller passes ``self.T_atm`` (the two ``self._z`` refreshes in ``_thermo_substep_impl`` bracket
        the column thermo, and q is consistent with T at both), so the pairing is the live column's. An
        offline caller that passes a HYPOTHETICAL ``T_atm`` (the profilers, the PHY-17 linearity probe)
        gets the LIVE q against it. Be precise about what that breaks (PR #841 review F7): with q held
        fixed inside a call, ``T_rho = T (1 + 0.608 q - q_cond)`` is still EXACTLY LINEAR in the
        ``T_atm`` argument, and so is this function -- the flag does NOT make the heights nonlinear in
        T. What it breaks is the ``z(E[T]) == E[z(T)]`` identity that
        ``test_phy17_impact_analysis.py:330-345`` rests on, because a day-MEAN call pairs a day-mean T
        with an INSTANTANEOUS q. Freezing q across the averaged calls restores the identity.

        THIS IS NOT THE ONLY DEPARTURE IN THE BRIDGE, and it is the smaller one (ROADMAP PHY-22):
        ``_update_buoyancy`` references b to the DRY REFERENCE profile ``T_ref(z)``, i.e. to
        ``p_ref(z)``, while these heights place each level where the LIVE column puts it -- so the two
        are different pressure coordinates whether or not this flag is ON. On the canonical state that
        prior live-vs-reference offset is ~7.4x this term's contrast. This flag is the correct
        COMPLETION of the live map, not the closing of a self-contained omission."""
        c = self.cfg
        T = self.T_atm if T_atm is None else T_atm
        Tsrc = T
        if c.hydrostatic_virtual_t:                                # PHY-21: thickness on the density temperature
            q_clip = get_xp(self.q).clip(self.q, 0.0, th.Q_MAX)    # (nx, nz) vapour mass fraction [kg/kg]
            if c.condensate_loading:
                xpc = get_xp(self.q_c)
                q_cond = xpc.maximum(self.q_c, 0.0)                # cloud, clipped at 0        [kg/kg]
                if self.q_r is not None:
                    q_cond = q_cond + xpc.maximum(self.q_r, 0.0)   # ... plus falling rain      [kg/kg]
                Tsrc = th.density_temperature(T, q_clip, q_cond, exact=c.virtual_temp_exact)   # [K]
            else:
                Tsrc = th.virtual_temperature(T, q_clip, exact=c.virtual_temp_exact)           # [K]
        xp = get_xp(T)                                             # device-generic (Phase D); numpy bit-identical
        # dtype matches the original np.empty (fp64 default) so the CPU path is BYTE-IDENTICAL (the fp64
        # intermediate the heights were always built in -- the working-dtype cast is downstream at the store).
        Tedge = xp.empty((c.nx, c.nz + 1))
        Tedge[:, 1:-1] = 0.5 * (Tsrc[:, :-1] + Tsrc[:, 1:])
        Tedge[:, 0] = Tsrc[:, 0]; Tedge[:, -1] = Tsrc[:, -1]
        dz = (C.R_DRY * Tedge[:, 1:] / C.G_ACCEL) * self._dlnp[None, :]   # interface k -> k+1 thickness
        z_edge = xp.empty((c.nx, c.nz + 1))
        z_edge[:, -1] = 0.0                                        # surface interface = nz (z=0) upward
        z_edge[:, :-1] = xp.cumsum(dz[:, ::-1], axis=1)[:, ::-1]
        return 0.5 * (z_edge[:, :-1] + z_edge[:, 1:])

    def _pressure_to_height(self, field_p, z_p):
        """Reconcile a PRESSURE-grid field (nx, nz; TOP->SURFACE) onto the HEIGHT grid
        (nz, nx; SURFACE->TOP), per column, by linear interpolation in physical height. ``z_p`` (nx, nz)
        is the layer-center heights from ``_hydrostatic_z`` (TOP->SURFACE, decreasing). One-way read
        (decision #1's dual-grid bridge); used to build the buoyancy/qsat inputs on the dynamics grid.
        Off-domain queries clamp to the column's end values (np.interp default). Vectorized over the
        columns (issue #99): one batched ``_interp_columns`` call, BIT-IDENTICAL to the per-column
        ``np.interp`` loop it replaced (the ::-1 views feed the same SURFACE->TOP increasing-z samples)."""
        # DEVICE (Phase D): the numba advect_parallel kernel is host-only -> use the xp-generic
        # ``_interp_columns`` on device (the numba path is a CPU fuse). numpy default is byte-identical.
        interp = (_interp_columns_parallel if (self.cfg.advect_parallel and not self._on_device)
                  else _interp_columns)
        # Phase E: on the capture path route to the GENERIC comparison-count branch (no bincount/no D2H
        # check) -- CuPy's bincount syncs internally, capture-fatal. Only _interp_columns takes the flag (the
        # numba parallel path is host-only, never on the device/capture path). OFF -> no kwarg (byte-identical).
        if self.cfg.capture_ready and self._dyn_on_device:
            return _interp_columns(self.z[None, :], z_p[:, ::-1], field_p[:, ::-1], force_generic=True).T
        return interp(self.z[None, :], z_p[:, ::-1], field_p[:, ::-1]).T

    def _height_to_pressure(self, field_h, z_p):
        """Inverse reconciliation: a HEIGHT-grid field (nz, nx; SURFACE->TOP) onto the PRESSURE grid
        (nx, nz; TOP->SURFACE), per column, by linear interpolation in height. ``z_p`` as above. Used to
        map the resolved-flow moisture ADVECTIVE INCREMENT back onto q's pressure-grid master (one-way,
        increment-only -- so q is never round-tripped and accrues no spurious round-trip diffusion).
        Vectorized over the columns (issue #99): one batched ``_interp_columns`` call against the SHARED
        height grid, BIT-IDENTICAL to the per-column ``np.interp`` loop it replaced."""
        # DEVICE (Phase D): gate the host-only numba kernel off on device; route field_h through its own xp
        # (it may be a device array -- the dynamics u/w) so ``.T`` stays on-device. numpy default byte-identical.
        interp = (_interp_columns_parallel if (self.cfg.advect_parallel and not self._on_device)
                  else _interp_columns)
        return interp(z_p, self.z[None, :], get_xp(field_h).asarray(field_h).T)

    def _pressure_to_height_pair(self, field_p1, field_p2, z_p, field_p3=None, field_p4=None):
        """Two ``_pressure_to_height`` maps that share the column heights ``z_p`` (the ``_update_buoyancy`` T+q
        pair): one bracket build, two gathers (``_interp_columns_pair``). BIT-IDENTICAL to two separate
        ``_pressure_to_height`` calls -- the bracket is a pure function of (self.z, z_p), shared by both.
        With ``advect_parallel`` ON each field is mapped by the prange kernel (bit-identical; the per-column
        bracket recompute is cheap inside numba, so the two parallel gathers beat the single numpy bracket).

        ``field_p3`` (PHY-19 condensate loading) adds a THIRD field to the SAME bracket and returns a
        3-tuple -- so turning the loading on costs one extra gather, not a second bracket build (the
        heavy part). ``field_p3=None`` is the shipped two-field call, byte-identical.

        ``field_p4`` (PHY-22 B', the buoyancy pressure term) adds a FOURTH on the same bracket, and is
        where the layer pressures ``self._p_mid_row`` ride. It is a SINGLE ROW (1, nz), which the
        batched kernel gathers with the cheap flat ``take``; the numba column-parallel kernel has no
        single-row-value shape, so on that path it falls back to the numpy ``_interp_columns``, which is
        bit-identical to it by construction (the numba kernels are gated bitwise against it). Passing
        ``field_p4`` without ``field_p3`` raises rather than silently dropping the field."""
        x = self.z[None, :]; xp = z_p[:, ::-1]
        if self.cfg.advect_parallel and not self._on_device:      # numba is host-only -> device uses the xp pair
            out = (_interp_columns_parallel(x, xp, field_p1[:, ::-1]).T,
                   _interp_columns_parallel(x, xp, field_p2[:, ::-1]).T)
            if field_p3 is None:
                if field_p4 is not None:
                    raise ValueError("_pressure_to_height_pair: field_p4 requires field_p3.")
                return out
            out = out + (_interp_columns_parallel(x, xp, field_p3[:, ::-1]).T,)
            if field_p4 is None:
                return out
            return out + (_interp_columns_parallel(x, xp, field_p4[:, ::-1]).T,)
        # Phase E: on the capture path route to the GENERIC comparison-count branch (no bincount, no D2H
        # check) -- CuPy's bincount syncs internally, capture-fatal. OFF -> default False (byte-identical).
        gen = self.cfg.capture_ready and self._dyn_on_device
        if field_p3 is None:
            if field_p4 is not None:
                raise ValueError("_pressure_to_height_pair: field_p4 requires field_p3.")
            o1, o2 = _interp_columns_pair(x, xp, field_p1[:, ::-1], field_p2[:, ::-1], force_generic=gen)
            return o1.T, o2.T
        if field_p4 is None:
            o1, o2, o3 = _interp_columns_pair(x, xp, field_p1[:, ::-1], field_p2[:, ::-1],
                                              field_p3[:, ::-1], force_generic=gen)
            return o1.T, o2.T, o3.T
        o1, o2, o3, o4 = _interp_columns_pair(x, xp, field_p1[:, ::-1], field_p2[:, ::-1],
                                              field_p3[:, ::-1], field_p4[:, ::-1], force_generic=gen)
        return o1.T, o2.T, o3.T, o4.T

    # ------------------------------------------------------------------ surface inputs (#shared with Stage A)
    def _exchange_wind(self):
        """The bulk-aerodynamic exchange WIND for the surface fluxes: the RESOLVED surface cross-track
        wind |u_surf| with a gustiness floor when ``resolved_flux_wind`` + the dynamics are live (the
        WISHE-like feedback, #85 item 1), else the fixed scalar ``cfg.wind``. Factored out so the
        surface-flux step AND the budget wet-fraction's potential evaporation ``E_pot`` see the SAME
        wind (the budget closure compares supply to the evaporation the step will actually apply)."""
        c = self.cfg
        if c.resolved_flux_wind and self.run_dynamics and self._u_surf is not None:
            xp = get_xp(self._u_surf)                              # device-generic (Phase D): u_surf stays on device
            return xp.sqrt(self._u_surf**2 + c.gustiness**2)      # (nx,) resolved exchange wind
        return c.wind                                              # scalar (byte-identical off-limit)

    def _budget_E_w(self, q, wind):
        """The per-column WET-TILE potential evaporation ``E_w`` [kg/m^2/s] for the budget wet-fraction
        closure: the evaporation of an open-water tile sitting at its OWN evaporative-thermostat
        temperature ``T_w`` -- the wetness=1 surface-energy-balance root (the cool ~62 C branch), found
        with the core's own ``bulk_air_sea_sst`` on the column's cached radiative drivers (absorbed SW
        ``surf_abs_sw`` + downwelling LW ``D_surf``) and near-surface air. ``E_w`` is the per-unit-wet-area
        demand the supply ``F`` must meet -- the native per-column form of approach A's
        ``_equator_pond.open_water_equilibrium``.

        WHY the wet-tile T_w, not the cell's current T_surf: at a hot salt-flat surface (~130 C) ``qsat``
        is super-boiling, so a potential-evaporation evaluated at ``T_surf`` is ~10x the wet-tile value and
        the closure would be far too sluggish to ever ignite the wet state (the chicken-and-egg trap --
        PR review). The wet tile self-limits to the thermostat regardless of the dry-flat forcing, so its
        demand is small (E_w ~ 70 mm/day, approach A) and any real convergent supply clips f toward 1.
        (Minor approximation: ``surf_abs_sw`` carries the column's CURRENT albedo, not the wet tile's lower
        one -- a second-order SW difference vs the dominant temperature effect, conservative toward wet.)"""
        rad = self._last_rad
        xp = get_xp(self.T_atm)                                    # device-generic (Phase D)
        T_w = bulk_air_sea_sst(rad["surf_abs_sw"], rad["D_surf"],
                               self.T_atm[:, -1], q[:, -1], wind, wetness=1.0, n_bisect=30)  # 30: sub-uK over
        #   the 150-900 K bracket (750/2^30 ~ 7e-7 K); was 50 (~7e-13 K, needless). Halves the bisection cost.
        return C.RHO_AIR0 * 1.2e-3 * wind * xp.maximum(th.qsat(T_w, C.P_SURF) - q[:, -1], 0.0)  # kg/m^2/s

    def _init_surf_water(self, source):
        """Initialize the prognostic surface water ``W_surf`` (nx,) [kg/m^2] from the CURRENT state's
        LEGACY wetness diagnosis (#302 back-compat: 'the seas that exist get their water'): columns the
        legacy T-ramp ``wetness_of_Tsurf(T_surf)`` calls wet (>= 0.5, i.e. T_surf <= ~62.5 C, the ramp
        midpoint) get the deep reservoir ``surf_water_reservoir``; dry columns get 0. (At init/load time
        the legacy ``_wetness``'s budget_f term is identically zero -- ``_precip`` is a lagged working
        buffer, zeroed at construction and not checkpointed -- so the ramp IS the full legacy diagnosis.)
        Works seeded from EITHER attractor: hot-dry (only the poleward ocean below the ramp midpoint ->
        only it gets water) or cool-wet (the inner sea is cool -> it gets water too). Called at
        construction and by ``load_state`` when the checkpoint has no "W_surf" key; the init is stamped
        in the log. W is fp64 ALWAYS (per-substep increments ~1e-3 kg/m^2 against the ~1.5e5 kg/m^2
        reservoir are far below the fp32 ULP there -- the SEB-freeze class of bug; W is an (nx,) surface
        field so the fp64 carry is cheap, exactly the ``_seb_incr_buf`` reasoning). Off the hot path
        (init/load only), so the host pull for the stamp is fine."""
        c = self.cfg
        xp = get_xp(self.T_surf)
        ramp = wetness_of_Tsurf(self.T_surf)
        W = xp.where(xp.asarray(ramp, xp.float64) >= 0.5, c.surf_water_reservoir, 0.0)
        if getattr(self, "W_surf", None) is not None and self.W_surf.shape == W.shape:
            # in-place into the held buffer (capture-safe address). NB: coerce to the BUFFER's own array
            # module -- during load_state T_surf (hence W) is transiently HOST-side while W_surf is already
            # a device buffer, and cupy's `[...] = numpy_array` routes to fill() and throws (probe-caught).
            self.W_surf[...] = to_device(W, get_xp(self.W_surf))
        else:
            self.W_surf = xp.asarray(W, xp.float64)
        if c.surf_salt:
            # surf_salt: the initial WATER carries ocean-baseline DISSOLVED salt (S_dis = salinity0*W;
            # dry cells -> 0, they hold no water to dissolve into). The DRY-INITIALIZED cells (W = 0)
            # additionally start with a large DEPOSITED halite crust `surf_salt_dry_init` -- they are
            # evaporite pans, not clean rock, and would otherwise stay saltless forever (halite only
            # precipitates from evaporating brine, which a dry-start cell never has; see the knob's
            # docstring + regions.dry_mask's "WHY A UNION" note). Ocean cells (W > 0) stay crust-free.
            # When water later reaches a seeded flat the saturation adjustment re-dissolves the crust
            # (deficit branch), so no extra code is needed there. Same in-place/coerce idiom as W above.
            S0 = W * c.surf_salt_salinity0                             # [kg/m^2] dissolved; kg/kg-WATER conv.
            Z0 = xp.where(W > 0.0, 0.0, c.surf_salt_dry_init)          # [kg/m^2] deposited crust: dry flats seeded
            if getattr(self, "S_dis", None) is not None and self.S_dis.shape == S0.shape:
                self.S_dis[...] = to_device(S0, get_xp(self.S_dis))
                self.S_dep[...] = to_device(Z0, get_xp(self.S_dep))
            else:
                self.S_dis = xp.asarray(S0, xp.float64)
                self.S_dep = xp.asarray(Z0, xp.float64)
        n_wet = int(to_host(xp.sum(self.W_surf > 0.0)))
        n_dry = c.nx - n_wet
        print(f"[surf_water_budget] W_surf initialized from the legacy wetness diagnosis ({source}): "
              f"{n_wet}/{c.nx} columns wet at W={c.surf_water_reservoir:.3g} kg/m^2, the rest dry (W=0)"
              + (f"; S_dis = {c.surf_salt_salinity0:g}*W (ocean baseline), S_dep = "
                 + (f"{c.surf_salt_dry_init:.3g} kg/m^2 seeded on the {n_dry} dry cells (0 on ocean)"
                    if c.surf_salt_dry_init > 0.0 else "0") if c.surf_salt else ""),
              flush=True)

    def _wetness(self):
        """Per-column evaporation efficiency in [0,1]. DEFAULT: the temperature-keyed ocean<->salt-flat
        ramp ``wetness_of_Tsurf(T_surf)``.

        With ``cfg.surf_water_budget`` (#302, checked FIRST -- it SUBSUMES the budget branches below):
        wetness = clip(W_surf / surf_water_wmin, 0, 1), a pure function of the PROGNOSTIC surface water.
        The T-ramp is NO LONGER a wetness source (its physical content emerges from the budget: hot => E
        huge => W drains => dry), and budget_f / the endorheic pool are not consulted (their supply-limited
        roles are what the W equation does natively). Cast through ``th._F`` so the fp32 thermo gate sees a
        gate-dtype wetness (no residual fp64 sub-path), while W itself stays the fp64 accumulator.

        With ``cfg.budget_wet_fraction`` (B5): ``f = max(ramp, budget_f)`` where ``budget_f =
        clip(F / E_w, 0, 1)`` is the SUPPLY-limited wet fraction (``F`` = the surface-reaching liquid supply
        ``self._precip``; ``E_w`` = the WET-TILE evaporative-thermostat demand, ``_budget_E_w`` -- the
        wetness=1 SEB root, NOT the evaporation at the cell's hot T_surf). The ``max(ramp, .)`` is
        load-bearing: it keeps **cool reservoir/ocean columns FULLY WET** (ramp=1 -> a full-potential
        evaporative SOURCE, evaporation able to exceed local precip), and only ADDS wetness where the ramp
        WRONGLY dries a *supplied* hot convergence surface (ramp=0, budget_f>0 -> the equatorial sea wets
        when fed). A bare ``budget_f`` would throttle the open ocean's source role to E<=precip (PR #177
        review S1) -- the ocean is reservoir-limited, not supply-limited. So: SUPPLY sets the state where
        the surface is supply-limited (salt flats / the convergence core), the ramp keeps the reservoir
        ocean a source. A convergence core the T-ramp wrongly calls a 138 C salt flat (it has no stable
        cool-wet root -- EQUATOR_POND_RESULTS.md / CHIMNEY_B4_IGNITION_RESULTS.md) becomes a standing
        cool-WET sea when fed -- the surface step's LE = f*E_pot(T_surf) stays RESPONSIVE, so a fully-wet
        (f=1) surface relaxes self-consistently to its evaporative-thermostat T_w.

        EXACT AT THE CLIPS, an approximation in between (PR #177 review S1, B5b/B5d). f=0 (ramp-identical)
        and f=1 (T_surf->T_w, full evaporation) are right; for 0<f<1 the single bucket evaporates the wet
        fraction at the cell's hot T_surf rather than the wet tiles' cooler T_w, so it OVER-evaporates
        (E>F) -- the pond-field water budget f*E_w=F closes only at the endpoints. Closing it in the
        interior needs the two-TEMPERATURE tile mosaic (carry T_w and T_d), deferred to B5b/B5d. Lagged one
        step on ``_precip`` -- well-posed.

        WARM-START GUARD: ``_wetness`` is called during construction (to seed q_init) and in the FIRST
        radiation call BEFORE ``q``/``_precip``/``_last_rad`` exist; there it falls back to the ramp.
        (Under ``surf_water_budget`` no guard is needed: W_surf is built before the first call.)

        DIURNAL SUN (PR #301 review finding 2): the ramp is a TIME-MEAN climate classifier -- keyed on
        instantaneous T under a resolved diurnal cycle it re-wets the dry flats every night (waterless
        evaporation + a ~130x night heat-capacity inflation pinning the dawn floor near SST_DRY). So with
        ``diurnal_sun`` every temperature-keyed leg (the ramp, the endorheic liquid gate) reads the SLOW
        EMA ``_T_surf_slow`` (tau = diurnal_wetness_tau), and the budget leg's demand reads the matching
        ``_E_w_slow`` (instantaneous E_w -> 0 at night would drive budget_f -> 1). With
        ``diurnal_wetness_freeze`` the whole field is PINNED at the seed classification (the clean 4.4
        control). OFF -> every read is the verbatim instantaneous object (byte-identical). #302's
        prognostic surface-water budget (below, checked FIRST) supersedes all of this when it's ON."""
        c = self.cfg
        if c.surf_water_budget:
            # #302: wetness from the prognostic water column ALONE (subsumes ramp + budget_f + endorheic,
            # and the diurnal_sun T_key keying below -- W_surf is itself a slow, integrated quantity).
            if self.W_surf is None:
                raise RuntimeError("surf_water_budget flipped ON after construction: the W state/buffers "
                                   "are built in __init__ -- set COSMO_SURF_WATER_BUDGET (or the config "
                                   "field) BEFORE constructing the core, never post-hoc.")
            xp = get_xp(self.W_surf)
            return th._F(xp.clip(self.W_surf / c.surf_water_wmin, 0.0, 1.0), xp)
        if c.diurnal_sun and c.diurnal_wetness_freeze and getattr(self, "_wetness_frozen", None) is not None:
            wf = self._wetness_frozen
            return wf if wf.dtype == self.T_surf.dtype else wf.astype(self.T_surf.dtype)
        T_key = self.T_surf
        if c.diurnal_sun and getattr(self, "_T_surf_slow", None) is not None:
            T_key = self._T_surf_slow
            if T_key.dtype != self.T_surf.dtype:
                T_key = T_key.astype(self.T_surf.dtype)    # match the fp32 substep's working dtype
        if not c.budget_wet_fraction:
            return wetness_of_Tsurf(T_key)
        precip = getattr(self, "_precip", None)
        q = getattr(self, "q", None)
        rad = getattr(self, "_last_rad", None)
        if precip is None or q is None or rad is None:             # construction / first radiation: no state yet
            return wetness_of_Tsurf(T_key)   # PR #301 merge fix: this fallback must key on T_key too (was
            #   reverted to self.T_surf by a bad conflict resolution -- see the docstring above / test_diurnal_sun.py)
        if c.surf_flash_boil:
            # SURFACE FLASH-BOIL: the flashed portion of the landed flux re-entered the BL as vapour and
            # never wetted anything -- only the post-flash LIQUID supplies the wet fraction / endorheic
            # pool. Lagged one substep exactly like ``precip`` itself (``_surf_flash`` is written in the
            # SEB block; the two lags differ by one substep during transients -- identical at steady
            # state -- hence the max(., 0) guard). OFF: ``_surf_flash`` stays 0 and this branch is skipped.
            precip = get_xp(precip).maximum(precip - self._surf_flash, 0.0)
        # E_w (the wet-tile evaporative-thermostat demand) is set by the RADIATIVE drivers, which refresh only
        # on the FIRST thermo substep (``_skip_rad_in_thermo_sub`` gates the other n_sub-1). So compute the SEB
        # bisection ONLY when radiation refreshes and CACHE it (in-place -> capture-safe) for the remaining
        # substeps -- E_w is slowly-varying, so reusing it removes ~n_sub redundant bisections/step (the
        # per-substep wetness still updates via the evolving ``precip``). Profiled: the uncached bisection was
        # called ~n_sub+ x/step (~60), the dominant budget_wet_fraction cost; caching cuts it to ~1. The cache
        # refreshes every step at the first substep, so it is never stale across steps.
        if getattr(self, "_skip_rad_in_thermo_sub", False) and getattr(self, "_E_w_cache", None) is not None:
            E_w = self._E_w_cache                                   # reuse (radiation/E_w unchanged this step)
        else:
            E_w = self._budget_E_w(q, self._exchange_wind())       # first substep / albedo: (re)compute
            if getattr(self, "_E_w_cache", None) is None or self._E_w_cache.shape != E_w.shape:
                self._E_w_cache = E_w                               # first alloc (eager / pre-capture)
            else:
                self._E_w_cache[...] = E_w                          # in-place write -> held buffer (capture-safe)
                E_w = self._E_w_cache
        if c.diurnal_sun and getattr(self, "_E_w_slow", None) is not None:
            # finding 2 (budget leg): use the DAY-SCALE demand, not tonight's -- see the docstring above.
            E_w = (self._E_w_slow if self._E_w_slow.dtype == E_w.dtype
                   else self._E_w_slow.astype(E_w.dtype))
        xp = get_xp(precip)                                         # device-generic (Phase D)
        if c.budget_endorheic:
            # ENDORHEIC inner-sea closure: size the wet pool to the WATER balance instead of clamping f per
            # cell. The basin fills contiguously from the axis out to x_sea, the outermost cell where the
            # cumulative convergent supply still covers the cumulative wet-tile evaporation. self.x is UNIFORM
            # in x, so the cross-track integral is cumsum(.)*self.dx (a scalar). ``frac`` is the per-cell fill
            # fraction (1 deep in the pool, grading linearly to 0 at the balance crossing so conservation is
            # exact, not one-cell-quantized); ``contig`` (cumprod of "cell at least partially supplied") keeps
            # the pool contiguous from the axis, so an isolated far supply (e.g. the ribbon band) cannot
            # spuriously wet the salt flat. Net: the pool evaporates int_0^x_sea E_w dx == the deposited precip,
            # returning ALL of it to the atmosphere (E=P, no leak) -- vs budget_f, which drops the excess.
            if self._dx_cell is None:
                cumF = xp.cumsum(precip) * self.dx
                edx = xp.maximum(E_w, 1e-12) * self.dx
            else:                                                  # stretched x: per-cell widths inside
                cumF = xp.cumsum(precip * self._dx_cell)
                edx = xp.maximum(E_w, 1e-12) * self._dx_cell
            cumE = xp.cumsum(edx)
            frac = xp.clip((cumF - cumE) / edx + 1.0, 0.0, 1.0)                  # graded fill (exact-conservative)
            # contiguous from the axis AND liquid (no standing sea below freezing -- beyond the freezing line the
            # water is ice, a different sink). If the supply exceeds the whole liquid zone's evaporation, the pool
            # caps at the freezing line and the residual is NOT balanced here (it freezes / must export aloft) --
            # a genuine water-budget signal, not silently dropped like the per-cell branch.
            liquid = (T_key > 273.15).astype(precip.dtype)
            contig = xp.cumprod((cumF >= cumE - edx).astype(precip.dtype) * liquid)
            f_endo = frac * contig
            return xp.maximum(wetness_of_Tsurf(T_key), f_endo)
        budget_f = xp.clip(precip / xp.maximum(E_w, 1e-12), 0.0, 1.0)
        return xp.maximum(wetness_of_Tsurf(T_key), budget_f)  # ramp keeps the reservoir ocean a source

    def _albedo(self):
        """Per-column surface albedo (ocean/salt-flat/ice); see ``surface_albedo``."""
        c = self.cfg
        return surface_albedo(self.T_surf, self._wetness(), c.albedo_ocean, c.albedo_dryland, c.albedo_ice)

    def _surf_heat_capacity(self):
        """Per-column effective surface heat capacity [J/m^2/K] (M2b.4; Stage A's scheme): an ocean mixed
        layer where wet, low-mass rock where dry (so the salt-flat hot zone equilibrates fast instead of
        blocking convergence). The steady surface ENERGY BALANCE is heat-capacity-independent; this only
        sets the spin-up rate (the diurnal swing is a Stage-C concern -- the core runs on day-mean flux).

        DIURNAL SUN (chunk 4.2): with ``diurnal_sun`` the sun IS resolved in time, so the heat capacity now
        SETS the diurnal swing (the whole point) and the fast spin-up slab above is the wrong physics --
        the physical diurnal skins (``constants.py``) replace it: ``C_BARE_ROCK`` (~10 cm rock, the big-swing
        dry salt flat) where dry, ``C_OCEAN_MIXED`` (50 m mixed layer, the barely-stirring sea) where wet,
        blended by the same wetness. OFF is the verbatim spin-up blend (byte-identical).

        SPIN-UP SEA DEPTH (``spinup_ocean_depth_m`` / ``spinup_ocean_min_theta_deg``, env
        COSMO_SPINUP_OCEAN_DEPTH_M / COSMO_SPINUP_OCEAN_MIN_THETA_DEG; ROADMAP PHY-17): under the diurnal
        sun, a positive depth replaces the 50 m ``C_OCEAN_MIXED`` WET term with a slab of that depth, in the
        wet columns POLEWARD of the static cutoff only -- a spin-up rate device (config note). Read at CALL
        time, so a post-build assignment (``ckpt_world``'s physics restore) is honoured; on the GPU it is
        baked at capture. 0.0 takes the untouched line (byte-identical)."""
        c = self.cfg
        wet = self._wetness()
        C_wet = self._spinup_wet_capacity(wet)     # None when OFF; refuses a bad depth / cutoff / no sun
        if c.diurnal_sun:
            if C_wet is not None:
                # per-column [J m^-2 K^-1] = [kg m^-3][J kg^-1 K^-1][m]; masked columns hold C_OCEAN_MIXED
                # exactly, so their product is bit-identical to the OFF line; the dry term is untouched
                return C_wet * wet + C.C_BARE_ROCK * (1.0 - wet)
            return C.C_OCEAN_MIXED * wet + C.C_BARE_ROCK * (1.0 - wet)
        C_ocean = C.slab_heat_capacity(c.ocean_depth_spinup, C.RHO_LIQ, C.CP_LIQ)
        C_rock = C.slab_heat_capacity(1.0, 2600.0, 800.0)                  # ~1 m rock (fast)
        return C_ocean * wet + C_rock * (1.0 - wet)

    def _spinup_params(self):
        """``(depth_m, min_theta_deg)`` when the PHY-17 spin-up sea depth is ON, else None -- the ONE place
        the knob is validated (called from ``__init__`` so a bad launch fails at construction, and on every
        ``_surf_heat_capacity`` call so a post-build assignment is validated too).

        ``cfg.spinup_ocean_depth_m`` [m]: 0.0 (the default) = OFF. A positive depth d must lie in (0, 50]:
        a slab deeper than the physical layer only slows the spin-up, and a silent clip would hide a typo;
        d = 50 is the physical layer itself (``slab_heat_capacity(50, RHO_LIQ, CP_LIQ)`` IS C_OCEAN_MIXED).
        A nonzero depth WITHOUT the diurnal sun is refused (the diurnal-OFF slab is ``ocean_depth_spinup``).
        ``cfg.spinup_ocean_min_theta_deg`` [deg] must lie in [0, 90). NaN fails every comparison, so it is
        refused too."""
        c = self.cfg
        d = float(getattr(c, "spinup_ocean_depth_m", 0.0) or 0.0)
        if d == 0.0:
            return None
        if not c.diurnal_sun:
            raise ValueError(
                f"spinup_ocean_depth_m = {d!r} m needs diurnal_sun: without the diurnal sun the wet slab is "
                f"ocean_depth_spinup ({c.ocean_depth_spinup!r} m) and this knob would be silently ignored. "
                f"Unset COSMO_SPINUP_OCEAN_DEPTH_M, or set ocean_depth_spinup instead.")
        if not (0.0 < d <= 50.0):
            raise ValueError(f"spinup_ocean_depth_m = {d!r} m is outside (0, 50]: it is a SPIN-UP depth "
                             f"for the 50 m physical mixed layer (C_OCEAN_MIXED), not a free parameter")
        cut = float(getattr(c, "spinup_ocean_min_theta_deg", 25.0))
        if not (0.0 <= cut < 90.0):
            raise ValueError(f"spinup_ocean_min_theta_deg = {cut!r} deg is outside [0, 90): it is the static "
                             f"cross-track cutoff equatorward of which the physical 50 m layer is kept")
        return d, cut

    def _spinup_wet_capacity(self, wet):
        """The per-column WET heat capacity [J m^-2 K^-1] under the spin-up sea depth, on ``wet``'s array
        module and in ``wet``'s dtype, or None when the knob is OFF.

        Column j gets ``slab_heat_capacity(d, RHO_LIQ, CP_LIQ)`` where ``theta_deg[j] >= cutoff`` and the
        physical ``C_OCEAN_MIXED`` elsewhere. Both values are exact in float32 (209,350,000 and d * 4,187,000
        at the protocol's d = 5), so the masked columns' product with ``wet`` is bit-identical to the OFF
        line's scalar product in fp64 AND fp32. CAPTURE-SAFE: the array is built (one host->device copy) on
        the first call for each (array module, dtype) -- an eager step, before any CUDA-graph capture -- and
        reused on every later call, including inside a captured step; the cache is cleared only when the
        (depth, cutoff) pair changes, which a running captured graph never does."""
        params = self._spinup_params()
        if params is None:
            return None
        if params != self._spinup_Cwet_params:
            self._spinup_Cwet = {}
            self._spinup_Cwet_params = params
        xp = th._xp_of(wet)
        key = (xp.__name__, np.dtype(wet.dtype).str)
        arr = self._spinup_Cwet.get(key)
        if arr is None:
            d, cut = params
            host = np.where(np.asarray(to_host(self.theta_deg), dtype=np.float64) >= cut,
                            C.slab_heat_capacity(d, C.RHO_LIQ, C.CP_LIQ), C.C_OCEAN_MIXED)
            arr = xp.asarray(host.astype(wet.dtype))
            self._spinup_Cwet[key] = arr
        return arr

    # ------------------------------------------------------------------ diurnal sun (COSMO_DIURNAL_SUN, 4.1)
    def _build_diurnal_tables(self):
        """Precompute the one-day instantaneous-sun tables for the DIURNAL forcing (chunk 4.1,
        DIURNAL_COUPLING_SCOPING.md). Rows come from ``geometry.diurnal_flux_table`` (the per-sun train sum
        at phase midpoints -- collimator gate / graded shade / haze AND the full 2-D COSMO_SHADE2D lens
        T(alpha,psi) included per sun, evaluated at each sun's instantaneous along-track offset), then each
        COLUMN is RENORMALIZED so its phase-mean equals the day-mean forcing self.S EXACTLY. That absorbs
        (a) the finite-``n_suns``/midpoint truncation and (b) the SHAPELESS optics-layer day-mean
        redistribution COSMO_LENS_RESHAPE, which has no instantaneous 1-D form -- so the (b) reshape alone
        is treated as a phase-uniform per-column gain (a documented approximation: it also reshapes the
        cycle's SHAPE, not modeled). So mean_phase(_S_table) == self.S bit-for-bit up to rounding, and the
        diurnal run's time-mean SW forcing is the day-mean run's forcing by construction. The diffuse twin
        is built the same way against self.S_diffuse when sw_diffuse. Called from __init__ on the HOST
        arrays (before device placement).

        2-D LENS (2026-07-15): the genuine redistributing lens T(alpha,psi) now flows through the table
        INSTANTANEOUSLY (``geometry.diurnal_flux_table`` sums per sun via ``_direct_flux_xe`` /
        ``_diffuse_flux_xe``, which evaluate T at that sun's (alpha, psi)), so the diurnal SHAPE the AD
        core reads IS the lens's real cycle -- the renormalization below is then only the small
        finite-``n_suns`` truncation correction (gain ~1), not the former phase-uniform-gain approximation.
        That phase-uniform approximation survives ONLY for the shapeless COSMO_LENS_RESHAPE (b) form."""
        c = self.cfg

        def _normalize(raw, target, label):
            mean = raw.mean(axis=0)                           # (nx,) phase-mean of the raw shape
            tgt = np.asarray(target, float)
            # RELATIVE starvation guard (review finding 4; was an absolute mean > 1e-12): a column whose
            # raw instantaneous shape carries < 1e-3 of its day-mean target has no meaningful cycle to
            # scale (e.g. a collimator-gated column an optics-layer lens nevertheless feeds) -- amplifying
            # it would be unbounded. Such columns get a ZERO cycle, LOUDLY (their phase-mean != day-mean;
            # acceptable only because it should never fire on a physical config -- measured zero columns
            # on the default, canonical-shade, and lens worlds).
            ok = mean > 1e-3 * np.maximum(tgt, 0.0)
            ok &= mean > 0.0
            starved = (tgt > 1e-6) & ~ok
            if np.any(starved):
                import warnings
                warnings.warn(f"diurnal {label} table: {int(starved.sum())} column(s) carry day-mean "
                              "forcing but an instantaneous train shape < 1e-3 of it (deep shade + lens?) "
                              "-- they get a ZERO diurnal cycle, breaking phase-mean == day-mean THERE.",
                              stacklevel=3)
            scale = np.where(ok, tgt / np.where(ok, mean, 1.0), 0.0)
            smax = float(scale.max()) if scale.size else 0.0
            if smax > 25.0:
                # cap-WARN, not clamp: clamping would break the phase-mean == day-mean invariant. With
                # the 2-D lens now carried instantaneously the renorm gain is ~1 (truncation only); a
                # large gain here therefore flags the phase-uniform COSMO_LENS_RESHAPE (b) case (the
                # equatorial-focus pool measured ~x8.8) being stretched -- audit the optics config.
                import warnings
                warnings.warn(f"diurnal {label} table: max renormalization gain x{smax:.1f} (> 25) -- "
                              "the phase-uniform-lens approximation is stretched; audit the optics.",
                              stacklevel=3)
            return raw * scale[None, :]

        x_host = np.asarray(self.x, float)                    # __init__-time: still host
        raw = g.diurnal_flux_table(x_host, n_phase=c.diurnal_nphase) * c.i0_scale
        self._S_table = _normalize(raw, self.S, "direct")
        self._S_now = self._S_table[0].copy()                 # (nx,) held row (persistent buffer)
        if self.S_diffuse is not None:
            raw_d = g.diurnal_flux_table(x_host, n_phase=c.diurnal_nphase, diffuse=True) * c.i0_scale
            self._S_diff_table = _normalize(raw_d, self.S_diffuse, "diffuse")
            self._S_diff_now = self._S_diff_table[0].copy()
        self._diurnal_dt_phase = g.CLOCK.T_day / c.diurnal_nphase   # row duration [s]

    def _diurnal_index(self):
        """Current table row from the START-OF-STEP true time: floor(sim_time / row_dur) mod n_phase --
        the SAME formula the device gather kernel computes from the device clock, so the eager and
        captured paths quantize the phase identically. Under the CFL-adaptive dt, rows repeat/skip
        exactly as true time dictates (finding 1: the phase never runs fast/slow of the physics)."""
        return int(self.sim_time / self._diurnal_dt_phase) % self.cfg.diurnal_nphase

    def _sync_diurnal_now(self, idx=None, t_abs=None):
        """(Re)load the held instantaneous-sun buffers from table row ``idx`` (default: from sim_time),
        and re-pin the device diurnal CLOCK when it exists (to sim_time, to the ABSOLUTE time ``t_abs``
        when given -- the checkpoint-clock resume path -- or to the row-start time when an explicit
        ``idx`` is given). EAGER-only writes (host, or host->device at eager time) -- load_state /
        drivers / the non-capture step path; never inside a captured graph."""
        if not self.cfg.diurnal_sun:
            return
        if t_abs is not None:
            t_clock = float(t_abs)
            idx = int(t_clock / self._diurnal_dt_phase) % self.cfg.diurnal_nphase
        elif idx is None:
            idx = self._diurnal_index()
            t_clock = float(self.sim_time)
        else:
            idx = int(idx) % self.cfg.diurnal_nphase
            t_clock = idx * self._diurnal_dt_phase       # the row's START time (kernel floor -> same row)
        self._S_now[...] = self._S_table[idx]
        if self._S_diff_now is not None:
            self._S_diff_now[...] = self._S_diff_table[idx]
        if self._diurnal_time_dev is not None:
            self._diurnal_time_dev[...] = t_clock

    def set_diurnal_phase_index(self, idx):
        """PUBLIC driver hook: pin the diurnal phase to table row ``idx`` (wraps mod n_phase). Captured-
        replay drivers MUST call this after seeding/resume -- the host ``sim_time`` does not advance under
        replay (the _sl_dt0_gpu linchpin 2), so the phase is carried by the device CLOCK this re-pins
        (the clock then self-advances by the realized dt inside the captured step, exact under adaptive
        dt). A driver that tracks time by replay count at a bound dt cap passes
        ``resume_step % diurnal_nphase``; a driver with true seconds passes ``t / (T_day/n_phase)``."""
        self._sync_diurnal_now(idx)

    def _get_diurnal_kernel(self, dtype):
        """The device diurnal-gather RawKernel for ``dtype`` (capture path), built once per C-precision and
        cached on the core. First-call NVRTC compile happens on the eager warmup step, never inside capture
        (the cuFFT-plan-in-warmup rule -- same pattern as the convadjust/LW/micro kernels)."""
        import numpy as _np
        key = "double" if _np.dtype(dtype) == _np.float64 else "float"
        k = self._diurnal_kernel_dev.get(key)
        if k is None:
            from ._gpu_kernels import build_diurnal_gather_kernel
            k = build_diurnal_gather_kernel(dtype)
            self._diurnal_kernel_dev[key] = k
        return k

    def _advance_diurnal_sun(self):
        """Refresh the held instantaneous-sun row buffers for THIS step (start-of-step phase -- the
        tendency reads the forcing at the step's start, the convergence-is-correctness convention).

        Two legs, both EXACT under the CFL-adaptive dt (PR #301 review finding 1 -- the phase is a
        function of TRUE elapsed time, never of the step count, because the realized dt runs BELOW the
        dt cap on the canonical diurnal state and a row-per-step counter would run the sun ~2x fast):
          * CAPTURE path (capture_ready AND on-device): a device gather kernel computes
            ``row = floor(clock / row_dur) % n_phase`` from the (1,) float64 device diurnal CLOCK and
            copies that row of each table into the FIXED _S_now buffers. The clock is accumulated by the
            REALIZED dt at end-of-step (see step()) with an in-place ``xp.add`` (NOT a rebind -- a rebind
            would not accumulate under replay, the _sl_dt0_gpu linchpin-2 trap), so replays advance the
            phase by exactly the physics' own elapsed time. No host sync anywhere.
          * EAGER path (CPU, or GPU without capture): the SAME floor formula on the host ``sim_time``
            (rows repeat/skip as true time dictates), row copied in place."""
        if self.cfg.capture_ready and self._on_device:
            k = self._get_diurnal_kernel(self._S_table.dtype)
            nx = np.int64(self._S_table.shape[1])
            nph = np.int64(self.cfg.diurnal_nphase)
            row_dur = np.float64(self._diurnal_dt_phase)
            nblk = (int(nx) + 255) // 256
            k((nblk,), (256,), (self._S_table, self._diurnal_time_dev, row_dur, nph, nx, self._S_now))
            if self._S_diff_now is not None:
                k((nblk,), (256,), (self._S_diff_table, self._diurnal_time_dev, row_dur, nph, nx,
                                    self._S_diff_now))
        else:
            self._sync_diurnal_now()

    def _radiate(self, q, conv=None, clouds=None, sw_inst=False):
        """``rad.column_radiation`` on the live column state with this core's radiative + CLOUD (#19)
        inputs. ONE wrapper so every radiation call site (the thermo step, the diagnostics, the
        energy/subsidence gates) shares the SAME greenhouse + cloud configuration -- no drift. Clouds
        default ON (``cfg.clouds``); pass ``clouds=False`` for the pre-cloud clear-sky call byte-for-byte
        (used by the clear-sky surface-SW diagnostic). ``clouds=None`` -> ``cfg.clouds`` (byte-identical
        to the historic no-arg call). ``conv`` is the optional per-layer resolved-ascent boost signal
        (#19, only used when ``cloud_conv_boost`` > 0).

        ``sw_inst`` (DIURNAL SUN, chunk 4.1): True ONLY from the in-loop thermo substep -- with
        ``diurnal_sun`` it reads the held INSTANTANEOUS-sun buffers (_S_now/_S_diff_now) instead of the
        day-mean self.S/S_diffuse. Every other caller (diagnostics, the energy gate, CRE, subsidence)
        leaves it False, so their outputs stay DAY-MEAN-equivalent -- deliberate: a snapshot diagnostic
        radiating at one phase would alias the cycle. OFF (or sw_inst=False) passes the identical self.S
        objects -> byte-identical."""
        c = self.cfg
        clouds = c.clouds if clouds is None else clouds
        if sw_inst and c.diurnal_sun:
            S_in, S_diff_in = self._S_now, self._S_diff_now
        else:
            S_in, S_diff_in = self.S, self.S_diffuse
        # PROGNOSTIC condensate (Chunk 1): feed the RETAINED reservoir q_c (+ the cover floor) so the cloud
        # optics read the prognostic in-cloud water path, not the adiabatic reconstruction. OFF -> no kwargs
        # (cloud_qc defaults None) so the call is byte-identical to the diagnostic-cloud path.
        qc_kw = (dict(cloud_qc=self.q_c, cloud_f_min=c.cloud_f_min, cloud_qc_thresh=c.qc_cover_thresh,
                      cloud_pdf=c.qc_pdf_cloud, cloud_qc_decouple_cap=c.qc_radiative_cap_decouple)
                 if c.prognostic_condensate else {})
        # Chunk 4: the BL/stratocumulus deck presents an OVERCAST cover floor to the optics (so the deck
        # radiates as a solid low cloud, consistent with the source's emissivity) -- 0 elsewhere / when off.
        if c.prognostic_condensate and c.bl_stratocumulus:
            qc_kw["cloud_cover_floor"] = self._bl_deck_cover
        # Salt-aerosol CCN Twomey cloud-albedo limb (#237; default OFF -> not passed, byte-identical). When on,
        # feed the prescribed droplet-number field N_d [cm^-3] as (nx,1) so the cloud optics shrink the LIQUID
        # r_eff (-> brighter cloud) per column. Only the ratio N_d/N_ref enters; ON at N_d==N_ref is a no-op.
        if c.salt_ccn and c.salt_ccn_twomey:
            qc_kw["cloud_n_d"] = self.N_d[:, None]
            qc_kw["cloud_n_d_ref"] = c.salt_ccn_nd_ref
        # FUSED DEVICE LW kernel (Phase E follow-up #2): on the device path replace the launch-bound LW
        # two-stream layer recurrence (the ~2 bands x 2 sweeps x nz tiny launches every step) with the fused
        # one-thread-per-column kernel. A dtype GETTER so column_radiation matches the kernel to dtau's dtype
        # (fp32 captured path / fp64). None on CPU -> the verbatim numpy/numba path (byte-identical). The
        # first-call NVRTC compile happens in the eager warmup, never inside capture.
        lw_dev = (self._get_lw_kernel
                  if (self._on_device and getattr(self, "_lw_fuse_enabled", True)) else None)
        # Salt-aerosol HAZE SW-dimming limb (#237; DECOUPLED from the Twomey/autoconv precip limbs and from the
        # geometry-baked S_diffuse -- gated solely on salt_ccn_haze, default OFF -> sw_tau_haze stays None,
        # byte-identical). tau_haze = min(haze_tau_ref * max(N_d/N_ref - 1, 0), haze_tau_cap): zero at/below
        # the reference loading, rising with salt aerosol, then CAPPED at the dust-storm ceiling so the
        # band-anchored linear form stays physical at the source loading (#237 AREA 1 roadmap; the bare linear
        # form gives tau ~ tens at the near-source N_d, which coagulation + export rule out). Runtime
        # N_d-dependent (per column); haze_tau_cap=inf recovers the bare unbounded linear form.
        sw_tau_haze = None
        if c.salt_ccn and c.salt_ccn_haze:
            xp_h = get_xp(self.N_d)
            sw_tau_haze = xp_h.minimum(
                c.salt_ccn_haze_tau_ref * xp_h.maximum(self.N_d / c.salt_ccn_nd_ref - 1.0, 0.0),
                c.salt_ccn_haze_tau_cap)
        return rad.column_radiation(
            self.p_edge[None, :], self.T_atm, q, self.T_surf, S_in, self._albedo(),
            co2_ppm=c.co2_ppm, k_wv=c.k_wv_lw, tau_wellmixed=c.tau_lw_wellmixed,
            clouds=clouds, cloud_rh_crit=c.cloud_rh_crit, cloud_water_scale=c.cloud_water_scale,
            cloud_conv=conv, cloud_conv_boost=c.cloud_conv_boost, cloud_sw_overlap=c.cloud_sw_overlap,
            cloud_sw_ssa_liq=c.cloud_sw_ssa_liq, cloud_sw_ssa_ice=c.cloud_sw_ssa_ice,
            cloud_sw_twostream=c.cloud_sw_twostream,
            lw_window=c.lw_window, f_win=c.f_lw_window, k_cont=c.k_cont_self,
            lw_conserving_split=c.lw_conserving_split,
            sw_bands=c.sw_bands, sw_diffuse=c.sw_diffuse, S_diffuse=S_diff_in,
            lw_parallel=c.radiation_parallel, lw_device_kernel=lw_dev, sw_tau_haze=sw_tau_haze, **qc_kw)

    def _ascent_signal(self):
        """A per-layer 0..1 resolved-ASCENT signal on the pressure grid for the optional convective-cloud
        boost (#19, ``cloud_conv_boost`` > 0): the resolved vertical velocity ``w`` mapped to q's layers
        and normalized so the strongest domain ascent is 1 and subsidence/zero is 0. The sub-grid
        convective cloud that forms in active updrafts even where grid-mean RH is below saturation -- the
        resolved-circulation cloud SOURCE the unified core enables. Only meaningful with the dynamics
        live; returns the convective signal the cloud-fraction diagnostic reads.

        CAVEAT (review #69 item 4): this is a RELATIVE measure -- normalized by the domain-max ascent, NOT
        an absolute w threshold -- so the boost scales with the strongest updraft present, not a fixed
        vigour. Only bites when ``cloud_conv_boost`` is enabled; whoever runs that sweep should keep the
        relative normalization in mind (an absolute-w gate would be the alternative)."""
        z_p = getattr(self, "_z", None)
        if z_p is None:
            z_p = self._hydrostatic_z(self.T_atm)
        _, w = self.velocities()                                  # height grid (nz, nx)
        # Phase D: keep w on the dynamics array module (device on GPU) -- _height_to_pressure routes it through
        # its own get_xp, so w_p stays device-resident (no to_host pull). numpy default byte-identical.
        xp = get_xp(self.T_atm)
        w_p = self._height_to_pressure(w, z_p)                    # (nx, nz) on q's pressure layers
        wmax = float(xp.max(w_p))                                 # scalar gate (off any captured region; #1)
        return xp.clip(w_p / wmax, 0.0, 1.0) if wmax > 0.0 else xp.zeros_like(w_p)

    # ------------------------------------------------------------------ column thermodynamics (#2,#3; M2b.3)
    def _get_convadj_kernel(self, dtype):
        """The fused column-parallel convective-adjustment RawKernel for ``dtype`` (Phase E profile
        follow-up), built once per C-precision and cached on the core. Returns None if CuPy is unavailable
        (so the caller falls back to the Python loop). The first-call NVRTC compile happens here -- on the
        eager warmup step path, NEVER inside CUDA-graph capture (mirrors the cuFFT-plan-in-warmup rule)."""
        import numpy as _np
        key = "double" if _np.dtype(dtype) == _np.float64 else "float"
        k = self._convadj_kernel_dev.get(key)
        if k is None:
            from ._gpu_kernels import build_convadj_cols_kernel
            k = build_convadj_cols_kernel(dtype)
            self._convadj_kernel_dev[key] = k
        return k

    def _get_segmix_kernel(self, dtype):
        """The fused one-thread-per-column SEGMENT-MIXER RawKernel for ``dtype`` (ROADMAP PHY-17 step 4:
        the converged convective adjustment), built once per C-precision and cached on the core. The
        first-call NVRTC compile happens here -- on the eager warmup step path, NEVER inside CUDA-graph
        capture (the same rule ``_get_convadj_kernel`` follows)."""
        import numpy as _np
        key = "double" if _np.dtype(dtype) == _np.float64 else "float"
        k = self._segmix_kernel_dev.get(key)
        if k is None:
            from ._gpu_kernels import build_segmix_cols_kernel
            k = build_segmix_cols_kernel(dtype)
            self._segmix_kernel_dev[key] = k
        return k

    def _get_lw_kernel(self, dtype):
        """The fused one-thread-per-column LW two-stream RawKernel for ``dtype`` (radiation-limb fuse,
        Phase E follow-up #2), built once per C-precision and cached on the core. A dtype GETTER passed to
        ``column_radiation`` so the two-stream dispatch picks the kernel matched to ``dtau``'s dtype (the
        fp32 captured radiation path / fp64). First-call NVRTC compile happens on the eager warmup step,
        never inside capture (the cuFFT-plan-in-warmup rule)."""
        import numpy as _np
        key = "double" if _np.dtype(dtype) == _np.float64 else "float"
        k = self._lw_kernel_dev.get(key)
        if k is None:
            from ._gpu_kernels import build_two_stream_lw_kernel
            k = build_two_stream_lw_kernel(dtype)
            self._lw_kernel_dev[key] = k
        return k

    def _get_micro_kernel(self, dtype):
        """The fused canonical microphysics+reevap RawKernel for ``dtype`` (radiation/thermo-limb fuse, Phase E
        follow-up #3), built once per C-precision and cached on the core. First-call NVRTC compile happens on
        the eager warmup step, never inside capture (the cuFFT-plan-in-warmup rule)."""
        import numpy as _np
        key = "double" if _np.dtype(dtype) == _np.float64 else "float"
        k = self._micro_kernel_dev.get(key)
        if k is None:
            from ._gpu_kernels import build_micro_cols_kernel
            k = build_micro_cols_kernel(dtype)
            self._micro_kernel_dev[key] = k
        return k

    def _micro_device(self, q, dt):
        """FUSED DEVICE microphysics (Phase E follow-up #3): the captured step's largest limb (~7 ms / 39%).
        Runs the canonical ``_prognostic_microphysics`` + ``_below_cloud_reevap`` (precip_reevap ON) as ONE
        column-parallel RawKernel (``micro_cols``), updating ``self.T_atm``/``self.q_c`` IN PLACE and the local
        ``q`` in place, and setting ``self._precip`` to the surface flux -- the SAME contract as the numpy
        ``_prognostic_microphysics`` (returns the updated ``q``). Gated to the canonical config by the caller
        (the same predicate the numba CPU-fuse uses), so this only ever runs the path ``_micro_numba`` mirrors.
        Round-off-equivalent to the numpy reference (ship gate: the climate-match)."""
        from ._gpu_kernels import micro_cols_device
        c = self.cfg
        dtype = self.T_atm.dtype
        kernel = self._get_micro_kernel(dtype)
        # persistent working-dtype (dm, p_mid): the kernel needs every float arg in ONE dtype (fp32 captured
        # path), but the grid arrays may be fp64. Cast ONCE per dtype (static; capture-safe -- built in warmup).
        key = "double" if np.dtype(dtype) == np.float64 else "float"
        aux = self._micro_aux.get(key)
        if aux is None:
            aux = (self.xp.ascontiguousarray(self.dm.astype(dtype)),
                   self.xp.ascontiguousarray(self.p_mid.astype(dtype)))
            self._micro_aux[key] = aux
        dm_w, p_mid_w = aux
        q = self.xp.ascontiguousarray(q)                        # no-op if already contiguous (it is)
        # #280 finite-tau cond (scalar; 0 -> kernel skips) + #237 salt warm-autoconv factor. warm is CACHED
        # (address-stable for capture; the pow is done here at warmup, never inside the captured step). When
        # both inactive (canonical world): cond_tau=0 + apply_warm=0 -> the kernel runs the identical shipped
        # arithmetic (branch skipped / warm_factor never read) -> byte-identical.
        cond_tau_k, apply_warm_k, warm_k = _micro_fuse_salt_cond_params(self, self.xp, dtype)
        precip = micro_cols_device(kernel, self.T_atm, q, self.q_c, dm_w, p_mid_w, dt,
                                   c.cloud_rh_crit, c.qc_autoconv_tau, c.qc_autoconv_crit,
                                   c.qc_autoconv_tau_ice, c.qc_autoconv_crit_ice, c.cloud_f_min,
                                   c.precip_reevap_rate,
                                   cond_tau=cond_tau_k, apply_warm=apply_warm_k, warm_factor=warm_k,
                                   virga_flash_boil=int(c.virga_flash_boil),    # #294 boiling-layer physics
                                   boil_suppress=int(c.cond_boil_suppress))     # (0/0 = shipped arithmetic)
        if c.capture_ready:
            self._precip[...] = precip                          # in-place into the fixed (nx,) buffer
        else:
            self._precip = precip
        return q

    def _thermo_substep(self, dt):
        """One column-thermodynamics sub-step on the PRESSURE grid (decision #3, radiation IN THE LOOP):
        radiative heating + surface sensible heat + prognostic-moisture evaporation/condensation +
        moist convective adjustment + (M2b.4) the prognostic SURFACE ENERGY BALANCE. Updates
        ``self.T_atm`` (nx, nz), ``self.q`` (nx, nz), and -- when ``couple_surface`` -- ``self.T_surf``
        (nx). With ``couple_surface=False`` the surface temperature is a PRESCRIBED boundary condition
        (the M2b.3 fixed-SST RCE limit; the RCE-anchor tests run there).

        Energy-consistent by construction: the latent heat of condensation is released to the air on the
        SAME pressure grid as the vapor is removed (no cross-grid remap); the convective adjustment
        conserves column enthalpy; and the surface LOSES exactly the SH + LE it deposits into the air
        (so the surface-atmosphere exchange is closed). Mirrors ``single_column_rce``'s per-column physics
        (the independent RCE anchor), so the psi==0 / fixed-SST limit of the core reproduces it.

        THERMO-fp32 GATE (``cfg.thermo_fp32``): the shared thermo leaf-math (radiation.py / thermo.py) is
        flipped to float32 for the duration of THIS substep only (``th.set_thermo_dtype``; ``rad._F`` follows
        the same module dtype), so the column arithmetic genuinely runs single precision. The persistent
        state arrays stay float64 (the dynamics never see fp32), and every fp32 increment that adds to an
        fp64 field upcasts the sum back to fp64 -> a PERSISTENT-fp64 ACCUMULATOR. Restored in ``finally`` so
        a raise can't leave the module fp32. Default OFF -> ``set_thermo_dtype`` never called -> byte-identical.

        IMPLEMENTATION (the genuine-fp32 isolation, not leaf-only): leaf dtype-following alone is NOT enough --
        with the persistent state fp64 the FIRST in-core op (``dt*out["H"]/self.C_layer`` = fp32/fp64 -> fp64)
        and the convective-adjustment pair math (which READS ``self.T_atm``) would silently run fp64. So under
        the flag we temporarily SWAP the ``self.*`` arrays the thermo READS to float32 (state T_atm/q/T_surf/q_c
        + the fixed C_layer/dm), run the body (which therefore computes in fp32 throughout -- and ``_radiate``/
        ``_surf_heat_capacity`` read these swapped ``self.*`` automatically), then RECONCILE the state by the
        INCREMENT form -- ``new_fp64 = old_fp64 + (work_fp32 - old_fp32).astype(f64)`` -- so the fp64 absolute
        is preserved and only the per-step CHANGE carries fp32 granularity (a true persistent-fp64 accumulator;
        a naive ``work.astype(f64)`` would re-round the ~300 K absolute every step = a spurious ~3e-5 K/step
        rounding masquerading as the radiation bias under test). Diagnostics (cum_evap/precip, _precip, _evap)
        stay as the impl writes them (off the physics path). The fixed arrays are restored exactly."""
        if not self.cfg.thermo_fp32:
            return self._thermo_substep_impl(dt)
        if self.cfg.capture_ready and self._dyn_on_device:
            return self._thermo_substep_fp32_capture(dt)
        f32 = np.float32
        xp = get_xp(self.T_atm)                             # device-generic (Phase D): the swap/reconcile stays on F's module
        # snapshot the fp64 persistent state we will reconcile
        T0, q0, Ts0, qc0 = self.T_atm, self.q, self.T_surf, self.q_c
        qr0 = self.q_r                                      # None unless prognostic_rain (stage 7)
        C_layer0, dm0 = self.C_layer, self.dm
        # SEB freeze fix (default ON): carry the surface increment in fp64 (see seb_legacy_fp32). The
        # persistent (nx,) fp64 buffer is allocated lazily (eager path -> no capture, a rebind is fine).
        seb_fix = not self.cfg.seb_legacy_fp32
        if seb_fix and (getattr(self, "_seb_incr_buf", None) is None
                        or self._seb_incr_buf.shape != Ts0.shape):
            self._seb_incr_buf = xp.empty(Ts0.shape, dtype=xp.float64)
        prev = th.set_thermo_dtype(f32)
        try:
            # swap reads to fp32 (state -> fp32 working copies; fixed arrays -> fp32 views)
            self.T_atm = T0.astype(f32); self.q = q0.astype(f32)
            self.T_surf = Ts0.astype(f32); self.q_c = qc0.astype(f32)
            if qr0 is not None:
                self.q_r = qr0.astype(f32)                  # q_r swaps/reconciles exactly as q_c does
            self.C_layer = C_layer0.astype(f32); self.dm = dm0.astype(f32)
            self._seb_fix_active = seb_fix
            self._thermo_substep_impl(dt)
            # reconcile by the INCREMENT form: fp64 absolute + fp32-granular change (xp -> device-resident)
            def _accum(old64, new32, old_for_incr):
                return old64 + (xp.asarray(new32, f32) - old_for_incr.astype(f32)).astype(xp.float64)
            self.T_atm = _accum(T0, self.T_atm, T0)
            self.q = _accum(q0, self.q, q0)
            if seb_fix:
                # apply the carried fp64 SEB increment to the fp64 absolute (no fp32 round-trip/difference).
                # Guard on couple_surface: when OFF the impl never wrote _seb_incr_buf (the SEB block is
                # gated by couple_surface), so T_surf must stay PRESCRIBED (the M2b.3 fixed-SST limit).
                self.T_surf = (Ts0 + self._seb_incr_buf) if self.couple_surface else Ts0
            else:
                self.T_surf = _accum(Ts0, self.T_surf, Ts0)
            self.q_c = _accum(qc0, self.q_c, qc0)
            if qr0 is not None:
                self.q_r = _accum(qr0, self.q_r, qr0)
        finally:
            th.set_thermo_dtype(prev)
            self._seb_fix_active = False
            self.C_layer, self.dm = C_layer0, dm0          # restore the fixed arrays exactly
        return None

    def _thermo_substep_fp32_capture(self, dt):
        """CAPTURE-READY fp32 swap-reconcile (Phase E): the IDENTITY-preserving variant of the fp32 path above,
        for CUDA-graph capture. The shipped path REBINDS ``self.T_atm`` etc. TWICE per step (the ``astype(f32)``
        working copy AND the ``_accum`` reconcile to a fresh fp64 array) -- each a fresh allocation, so the
        captured graph's baked buffer ADDRESSES would change every step (capture-breaking). Here:

          * the fp32 working copies are PERSISTENT scratch (allocated once, written IN PLACE each step), so the
            impl writes its ``_obuf`` stores into a FIXED fp32 address;
          * the reconcile writes IN PLACE into the ORIGINAL fp64 buffer (``T0``) and re-binds ``self.T_atm`` to
            that SAME object, so the step ENDS with the prognostic state at its fixed (graph-baked) fp64 address.

        ALIAS-SAFE: during the body the impl writes the fp32 SCRATCH, so ``T0`` is NEVER mutated -> at reconcile
        ``T0`` still holds the pre-body fp64 absolute, the correct increment baseline. The arithmetic is the SAME
        astype/subtract/add as the shipped ``_accum`` (only the destination differs), so eager ON==OFF is
        bit-identical (the harness ON-vs-OFF check is the proof). The fixed C_layer/dm fp32 views are PRECOMPUTED
        once (``_C_layer_f32``/``_dm_f32``) -- not re-cast per step (a fresh alloc + a finally rebind)."""
        f32 = np.float32
        xp = get_xp(self.T_atm)
        T0, q0, Ts0, qc0 = self.T_atm, self.q, self.T_surf, self.q_c
        C_layer0, dm0 = self.C_layer, self.dm
        # persistent fp32 scratch (allocate once; warmup precedes capture so they exist before record)
        qr0 = self.q_r                                     # None unless prognostic_rain (stage 7)
        sc = getattr(self, "_fp32_scratch", None)
        if sc is None or sc["T"].shape != T0.shape:
            sc = {"T": xp.empty_like(T0, dtype=f32), "q": xp.empty_like(q0, dtype=f32),
                  "Ts": xp.empty_like(Ts0, dtype=f32), "qc": xp.empty_like(qc0, dtype=f32)}
            if qr0 is not None:
                sc["qr"] = xp.empty_like(qr0, dtype=f32)   # q_r swaps/reconciles exactly as q_c does
            self._fp32_scratch = sc
            self._C_layer_f32 = C_layer0.astype(f32)       # fixed arrays: fp32 once (address-stable)
            self._dm_f32 = dm0.astype(f32)
            # SEB freeze fix: persistent fp64 (nx,) increment buffer (allocated in warmup, before capture)
            self._seb_incr_buf = xp.empty_like(Ts0, dtype=xp.float64)
        sc["T"][...] = T0; sc["q"][...] = q0; sc["Ts"][...] = Ts0; sc["qc"][...] = qc0
        if qr0 is not None:
            sc["qr"][...] = qr0
        seb_fix = not self.cfg.seb_legacy_fp32             # default ON (see seb_legacy_fp32)
        prev = th.set_thermo_dtype(f32)
        try:
            self.T_atm = sc["T"]; self.q = sc["q"]; self.T_surf = sc["Ts"]; self.q_c = sc["qc"]
            if qr0 is not None:
                self.q_r = sc["qr"]
            self.C_layer = self._C_layer_f32; self.dm = self._dm_f32
            self._seb_fix_active = seb_fix
            self._thermo_substep_impl(dt)               # writes its _obuf stores into the PERSISTENT fp32 scratch
            # reconcile IN PLACE into the original fp64 buffers (T0 untouched by the body -> correct baseline):
            #   new_fp64 = old_fp64 + (work_fp32 - old_fp32).astype(f64), written into old_fp64 itself.
            def _accum_inplace(old64, work32):
                xp.add(old64, (work32 - old64.astype(f32)).astype(xp.float64), out=old64)
                return old64
            self.T_atm = _accum_inplace(T0, self.T_atm)   # self.X re-bound to the SAME persistent fp64 object
            self.q = _accum_inplace(q0, self.q)
            if seb_fix:
                # FREEZE FIX: the SEB increment was carried out of the fp32 substep into _seb_incr_buf;
                # add it to the fp64 T_surf absolute IN PLACE (the difference-reconcile would have
                # re-quantized the sub-ULP ocean increment to 0). Ts0 is the graph-baked fp64 buffer.
                # Guard on couple_surface (a construction-time bool -> capture-safe fixed topology): when
                # OFF the impl never wrote _seb_incr_buf, so T_surf stays PRESCRIBED (M2b.3 fixed-SST).
                if self.couple_surface:
                    xp.add(Ts0, self._seb_incr_buf, out=Ts0)
                self.T_surf = Ts0
            else:
                self.T_surf = _accum_inplace(Ts0, self.T_surf)
            self.q_c = _accum_inplace(qc0, self.q_c)
            if qr0 is not None:
                self.q_r = _accum_inplace(qr0, self.q_r)
        finally:
            th.set_thermo_dtype(prev)
            self._seb_fix_active = False
            self.C_layer, self.dm = C_layer0, dm0          # restore the persistent fp64 fixed arrays
        return None

    def _match_module(self, arr, xp):
        """``arr`` on ``xp``'s module. A no-op (the SAME object, one isinstance check) when it already
        is, so both shipped paths -- host field with host weights, device field with device weights --
        are byte-identical and allocation-free, including under graph capture.

        It exists for the MIXED moment inside ``load_state``: the prognostic fields are assigned as
        HOST arrays there and only placed on the device at the end of the method, while the fixed grid
        weights (``dm``, ``_dx_cell``) are already device-resident from construction. Anything that
        integrates a field between those two points therefore meets a host field and a device weight,
        and CuPy raises rather than promoting."""
        if get_xp(arr) is xp:
            return arr
        return np.asarray(to_host(arr)) if xp is np else xp.asarray(arr)

    def _column_mass(self, f, xp=None):
        """Area+mass integral of a (nx,nz) mixing ratio -> [kg per metre of along-track]. THE one weighting
        for every water-budget mass (``water_conservation``'s vapor/cond/rain, ``_clip_q``'s clipped mass,
        ``_atm_total``'s baseline) so the closure residual is exact rather than off by a weighting.
        Branches on the grid exactly as the shipped budget did: uniform -> sum(f*dm)*dx (``_dx_w`` is a
        scalar there and cannot be indexed); stretched -> sum(f*dm*dx_cell). Returns a DEVICE scalar (not
        float()) -- the caller decides whether to sync, so the hot path stays capture-legal.

        The weights follow ``f``'s module (``_match_module``). Without that, ``load_state`` on a GPU core
        raised ``TypeError: Unsupported type <class 'numpy.ndarray'>`` for any checkpoint that does not
        carry ``atm_total_reset`` -- the branch that re-pins the baseline calls ``_atm_total`` while the
        just-loaded ``q`` is still host. ``--resume`` always carries that key, so the defect was latent
        until the cross-grid warm start began deliberately DROPPING it to open a clean closure window."""
        xp = xp or get_xp(f)
        dm = self._match_module(self.dm, xp)
        if self._dx_cell is None:
            return xp.sum(f * dm[None, :]) * self.dx
        return xp.sum(f * dm[None, :] * self._match_module(self._dx_cell, xp)[:, None])

    def _atm_total(self):
        """Total column water q + q_c + q_r [kg/m] -- the ATMOSPHERIC twin of ``sum(W_surf*dx)``, and the
        quantity ``_atm_total_reset`` baselines so ``water_conservation``'s ``atm_residual`` is exact.
        Summed term-by-term in the SAME order as the vapor/cond/rain masses it must agree with."""
        xp = get_xp(self.q)
        tot = (float(self._column_mass(xp.clip(self.q, 0.0, th.Q_MAX), xp))
               + float(self._column_mass(self.q_c, xp)))
        if self.q_r is not None:
            tot += float(self._column_mass(self.q_r, xp))
        return tot

    def _clamp_tracer_positive(self, field):
        """The implicit-advection positivity clamp, WITH the water it injects accounted (#294-A, PR #360
        review). ``max(field, 0)`` is a one-sided clip: where advection round-off drives a tracer
        negative, flooring it CREATES water -- the same KIND of event as ``_clip_q``'s floor, just at a
        different call site. It must therefore enter the SAME ledger, or ``atm_residual`` silently
        absorbs it and reports a leak that is really an uncounted source. It did: on the canonical
        ``vertical_implicit`` path the residual read +9.51 kg/m (rel 7.5e-9) -- exactly the injected
        mass -- and would have failed analyze_run's `< 1e-9` gate for a reason unrelated to any leak.

        Distinct from the pre-existing ``_clamp_count``/``_clamp_mass`` instrumentation, which is
        read+reset per replay batch and sums UNWEIGHTED mixing ratios (Sigma min(q,0)) -- ~1e9x smaller
        than the mass integral, and on a different window from the budget. Kept as-is (its consumers
        expect those semantics); this is a separate CUMULATIVE, reset-aligned, mass-weighted term.

        Inert on the explicit path (monotone advection never clamps), so `_cum_qadv` stays 0.0 there and
        the ledger is byte-identical to the pre-review shape."""
        xp = get_xp(field)
        neg = xp.minimum(field, 0)
        self._clamp_count = self._clamp_count + xp.sum(neg < 0)     # legacy instrumentation (unweighted)
        self._clamp_mass = self._clamp_mass + xp.sum(neg)
        inc = self._column_mass(xp.maximum(-field, 0.0), xp)        # the INJECTED mass [kg/m], >= 0
        if self.cfg.capture_ready:
            self._cum_qadv_pending = self._accum_pending(self._cum_qadv_pending, inc, xp)
        else:
            self._cum_qadv += float(inc)
        xp.maximum(field, 0, out=field)

    def _clip_q(self, q):
        """Clip vapour to the PHYSICAL bounds [0, th.Q_MAX] and ACCOUNT what the clip moves (#294-A).

        ``th.Q_MAX`` is the q implied by ``th.qsat``'s own ``e <= E_FRAC_MAX*p`` clamp -- the pure-steam
        ceiling, and a pure constant (p cancels). It is imported, never re-typed: a state cap BELOW the
        saturation cap is the bug this replaces (the Earth-era 0.5 bound at qsat(92.6 C)=0.694 on the lens
        world, so condensation could never fire in that cell and this clip became its ONLY vapour sink).

        BOTH bounds are accounted, as SEPARATE signed counters so they cannot cancel:
          ``_cum_qcap``   [kg/m] -- water the ceiling DESTROYS (the #294-A leak; ~0 once the ceiling is
                          physical, since condensation relaxes q to qsat <= Q_MAX below boiling; it can
                          still bite in a genuinely boiling layer, where cond_boil_suppress deliberately
                          lets steam pass -- and now it is VISIBLE instead of silent).
          ``_cum_qfloor`` [kg/m] -- water the 0-floor CREATES (round-off; the advection limbs have their
                          own ``_clamp_mass``, which this does not double-count -- different call sites).
        Both enter ``water_conservation`` so the column budget CLOSES:
            d(storage)/dt == E + flash - P - qcap + qfloor.
        Without them the budget missed ~15% of all evaporation and the gauge read the leak as spin-up.

        Capture-safe: the same ``_accum_pending`` in-place idiom as the other budget integrals (device-side
        under graph replay, host-synced only at the flush cadence). Costs two (nx,nz) reductions per thermo
        substep -- deliberate: an unaccounted water sink is not worth the saving."""
        xp = get_xp(q)
        over = xp.maximum(q - th.Q_MAX, 0.0)                       # >= 0: mass the ceiling deletes [kg/kg]
        under = xp.maximum(-q, 0.0)                                # >= 0: mass the floor invents [kg/kg]
        inc_cap, inc_flr = self._column_mass(over, xp), self._column_mass(under, xp)   # [kg/m]
        if self.cfg.capture_ready:
            self._cum_qcap_pending = self._accum_pending(self._cum_qcap_pending, inc_cap, xp)
            self._cum_qfloor_pending = self._accum_pending(self._cum_qfloor_pending, inc_flr, xp)
        else:
            self._cum_qcap += float(inc_cap)
            self._cum_qfloor += float(inc_flr)
        return xp.clip(q, 0.0, th.Q_MAX)

    # ---------------------------------------------------------------- PHY-19 (b): min-scale filter
    def _build_min_scale_operator(self):
        """Build the minimum-horizontal-scale operator for ``cfg.conv_min_scale_m`` and log what it
        actually covers. Called once from ``__init__`` when the flag is on, and again at the call site
        if the width changes on a built core (the warm-start route).

        The operator itself lives in ``flatworld/conv_min_scale.py`` -- pure numpy, no core, no config
        -- so its conservation and reflection properties are testable without a world. What this method
        adds is the core's own grid, the dtype/device copies, and the fixed work buffers."""
        if self.cfg.conv_dx_full > 0.0 or self._conv_gate_col is not None:
            # The gray-zone gate makes the Betts-Miller blend fraction an (nx, 1) PER-COLUMN array, and
            # PHY19_AXIS_CONVECTION.md section 5(b) step 3 records that the telescoping argument for
            # filtering a tendency before a per-column blend has NOT been checked. Refuse rather than
            # run an unverified conservation claim. Checked HERE, not only in __init__, so the
            # warm-start route (set cfg.conv_min_scale_m on a built core) cannot walk around it.
            raise NotImplementedError(
                "conv_min_scale_m and the gray-zone convective gate (conv_dx_off/conv_dx_full) cannot "
                "both be on: under the gate the blend fraction is per-column and the filter's "
                "column-conservation argument has not been checked for that case "
                "(PHY19_AXIS_CONVECTION.md 5(b) step 3). Turn one off.")
        W = float(self.cfg.conv_min_scale_m)
        # to_host on BOTH grid arrays: the operator is built on the host in fp64, and this method is
        # reachable AFTER device placement via the warm-start route (the call site rebuilds when the
        # width changes), where self.x and self._dx_cell are CuPy -- np.asarray on a CuPy array raises.
        widths = (np.asarray(to_host(self._dx_cell), float) if self._dx_cell is not None
                  else np.full(self.cfg.nx, float(self.dx)))     # uniform grid: one width everywhere
        n, A = _cms.build_min_scale_operator(np.asarray(to_host(self.x), float), widths, W)
        self._minsc_n, self._minsc_A_host, self._minsc_W = n, A, W
        self._minsc_A = {}                                       # drop any device/dtype copies of the old W
        self._minsc_gemm = {}
        self._minsc_buf = {}
        # dm is the LAYER MASS and is horizontally uniform in this core (_divfree_mass_flux carries a
        # fixed, horizontally-uniform layer mass), so the mass weighting dm_.k in
        # filt(d)_ik = (A @ (dm_.k d_.k))_i / dm_ik divides straight back out and the filter is A @ d.
        # The weighting is kept in the DERIVATION (conv_min_scale's docstring) so the operator is
        # correct the day the layer mass stops being uniform; this assert is what will fire on that day.
        assert np.ndim(self.dm) == 1, (
            "conv_min_scale assumes a horizontally-uniform layer mass dm (nz,), so the mass weighting "
            f"cancels; dm is now {np.shape(self.dm)} and the filter must carry the weights explicitly "
            "(see flatworld/conv_min_scale.py, 'THE OPERATOR')")
        print("[PHY-19] " + _cms.describe(A, widths, W, n_act=n, nx=self.cfg.nx), flush=True)

    def _minsc_ops(self, xp, dtype, nz):
        """``(A, gemm, dT, dq, fT, fq)`` for the active dtype -- the operator on this array module, the
        capture-safe device GEMM kernel (``None`` on the host path), and the four fixed ``(n_act, nz)``
        work buffers, built once per dtype and cached.

        Lazily allocated to match the ACTIVE dtype exactly as ``_conv_Tpre``/``_conv_qpre`` are: the
        captured fp32 path swaps the thermo state to fp32 inside ``_thermo_substep_fp32_capture``, and
        the eager warmup step runs before any graph capture, so both entries exist by the time a graph
        is recorded. Nothing here allocates on a warm call.

        WHY A RAWKERNEL AND NOT ``cupy.matmul`` ON DEVICE. CuPy's cuBLAS binding REFUSES calls during
        CUDA-graph capture ("calling cuBLAS API during stream capture is currently unsupported"), which
        is exactly why this package already carries a capture-safe tiled GEMM
        (``_gpu_kernels.build_gemm_kernel`` / ``gemm_device``, written for the stretched-Poisson
        transforms). The filter reuses it unchanged -- no new kernel source. On the host path numpy's
        ``matmul`` has no such restriction, so the two paths differ in dispatch only."""
        key = np.dtype(dtype)
        ent = self._minsc_A.get(key)
        if ent is None:
            ent = xp.ascontiguousarray(xp.asarray(self._minsc_A_host, dtype=key))
            self._minsc_A[key] = ent
        gemm = None
        if self._on_device:
            gemm = self._minsc_gemm.get(key)
            if gemm is None:
                from ._gpu_kernels import build_gemm_kernel
                gemm = build_gemm_kernel(key)          # NVRTC compile: warmup path, never in capture
                self._minsc_gemm[key] = gemm
        bufs = self._minsc_buf.get(key)
        if bufs is None or bufs[0].shape != (self._minsc_n, nz):
            bufs = tuple(xp.empty((self._minsc_n, nz), dtype=key) for _ in range(4))
            self._minsc_buf[key] = bufs
        return (ent, gemm) + bufs

    def _apply_min_scale_filter(self, q):
        """Filter the convective adjustment's tendency horizontally, in place, on the active block.

        ``dT = T_atm - _conv_Tpre`` and ``dq = q - _conv_qpre`` are the change the adjustment just made
        -- whichever branch ran (capped Gauss-Seidel sweep, fused device kernel, or segment mixer), and
        whether or not it converged. Both are replaced by ``A @ d`` on the columns narrower than ``W``
        and left untouched everywhere else, then written back as ``X_pre + filt(dX)``. The Betts-Miller
        blend that follows sees the filtered target and runs unchanged.

        WHAT THIS CONSERVES, AND WHAT IT DOES NOT. Column enthalpy ``sum c_p dx dm dT`` and column water
        ``sum dx dm dq`` are conserved to round-off per level (the operator's columns are normalized to
        ``sum_i dx_i A_ij = dx_j``; ``conv_min_scale`` asserts it at build time). PER-COLUMN
        conservation is deliberately broken -- moving adjustment work between columns is the whole
        point, and the ``_bm`` blend's per-column-exactness comment must be read as applying to the
        blend, not to this.

        SUPERSATURATION. Filtering a TENDENCY is not bounded the way filtering a STATE would be: where a
        neighbour dried hard and this column is nearly dry, ``sum_j A_ij dq_j`` can carry a column below
        zero or above saturation. That is handled by what already follows -- the condensation /
        prognostic-microphysics step reads the FILTERED state, and the next substep's ``_clip_q``
        accounts anything the 0-floor invents into ``_cum_qfloor`` (which enters ``water_conservation``,
        so it is visible in the budget rather than silent). There is deliberately NO second fix here.

        CAPTURE SAFETY. Fixed buffers, no allocation, no host sync, no data-dependent control flow, and
        ``xp``-generic (numpy or CuPy). The only new per-substep work is the two subtractions that form
        the tendency on the active block, the two matrix products, and the two adds back. The device
        product goes through the package's capture-safe tiled-GEMM RawKernel rather than
        ``cupy.matmul``, because cuBLAS cannot be called during stream capture (see ``_minsc_ops``).
        ``X[:n]`` is a LEADING-axis slice of a C-contiguous array, so it is itself C-contiguous, which
        is what ``gemm_device``'s contiguity assert requires."""
        xp = get_xp(self.T_atm)
        n = self._minsc_n
        A, gemm, dT, dq, fT, fq = self._minsc_ops(xp, self.T_atm.dtype, self.T_atm.shape[1])
        xp.subtract(self.T_atm[:n], self._conv_Tpre[:n], out=dT)
        xp.subtract(q[:n], self._conv_qpre[:n], out=dq)
        if gemm is None:
            xp.matmul(A, dT, out=fT)             # (n, n) @ (n, nz) -- one product per field per substep
            xp.matmul(A, dq, out=fq)
        else:
            from ._gpu_kernels import gemm_device
            gemm_device(gemm, A, dT, fT)
            gemm_device(gemm, A, dq, fq)
        xp.add(self._conv_Tpre[:n], fT, out=self.T_atm[:n])
        xp.add(self._conv_qpre[:n], fq, out=q[:n])

    def _thermo_substep_impl(self, dt):
        c = self.cfg
        xp = get_xp(self.T_atm)                                    # device-generic (Phase D); numpy on CPU
        self._z = z_p = self._hydrostatic_z(self.T_atm)            # (nx, nz) cache for convective adjust
        # #294-A: the ceiling is the PHYSICAL one (th.Q_MAX, the q at qsat's own e<=0.999p clamp), NOT a
        # re-typed literal. It was 0.5 -- an Earth-era guard (Earth q ~ 0.04) that never bound until the
        # lens world put a 92 C SEA at theta~22, where qsat = 0.694 > 0.5. A cap BELOW qsat means
        # condensation can never fire in that cell, so this clip becomes its ONLY vapour sink and
        # silently deletes ~15% of all evaporation -- and it was invisible to water_conservation because
        # the diagnostics applied the same clip. The clipped mass is now accounted (_cum_qcap).
        q = self._clip_q(self.q)
        # --- radiation in the loop (#3) on the live column state, with CLOUDS (#19) if enabled, SUB-CYCLED
        #     at the dt_rad cadence (#86). The radiative field (H/surf_net/OLR) evolves on hours-days, so it
        #     is RECOMPUTED only when stale (age >= dt_rad) or the state moved a lot (max|dT_atm| > 1 K), and
        #     HELD constant otherwise -- while the fast physics below stays on the substep clock. dt_rad <= 0
        #     (the default) recomputes EVERY step -> byte-identical. The convective-cloud boost reads the
        #     resolved ascent (only when cloud_conv_boost>0 and the dynamics are live); it is recomputed only
        #     on a refresh (it is a radiation input). NOT a Newtonian relaxation (#3): the held field is the
        #     model's OWN radiation, just refreshed at a physical cadence. ---
        #     The large-move guard watches BOTH T_atm AND T_surf: the held out["surf_net"] embeds
        #     sigma*T_surf^4 (radiation.py: surf_net_lw = sigma*T_s^4 - D_surf), so a fast-moving surface
        #     (a shallow/quick mixed layer) must also force an early refresh, else the semi-implicit surface
        #     step pairs a stale sigma*T_s^4 in R_net with a live 4*sigma*T_s^3 in dRdT (small + self-
        #     correcting each refresh, but the guard makes the hold robust regardless of surface speed).
        # capture-ready (Phase B(iii)): ALWAYS refresh -> drop the data-dependent staleness branch (the
        # held-vs-recompute decision is a capture-blocking control-flow + a device->host max() sync). This is
        # BYTE-IDENTICAL on the default path (dt_rad <= 0 ALREADY recomputes every step); with dt_rad > 0 it
        # is the deliberate GPU "always do the work" trade (refresh where the held cadence would have skipped),
        # which is NOT bit-identical -- but the held field is the model's OWN radiation, so always-refresh is
        # the MORE accurate limit, not a different physics. Default-OFF leaves the shipped staleness test.
        refresh_rad = (not getattr(self, '_skip_rad_in_thermo_sub', False)) and (
            c.capture_ready or self._last_rad is None or c.dt_rad <= 0.0 or self._rad_age >= c.dt_rad
            or (self._rad_T_ref is not None
                and (float(xp.max(xp.abs(self.T_atm - self._rad_T_ref))) > 1.0
                     or float(xp.max(xp.abs(self.T_surf - self._rad_Tsurf_ref))) > 1.0)))
        if refresh_rad:
            conv = (self._ascent_signal() if (c.clouds and c.cloud_conv_boost > 0.0 and self.run_dynamics)
                    else None)
            out = self._radiate(q, conv=conv, sw_inst=True)   # diurnal ON: THIS step's instantaneous sun
            self._last_rad = out
            self._rad_age = 0.0
            self._rad_T_ref = self.T_atm.copy() if c.dt_rad > 0.0 else None
            self._rad_Tsurf_ref = self.T_surf.copy() if c.dt_rad > 0.0 else None
        else:
            out = self._last_rad
            self._rad_age += dt
        dTa = dt * out["H"] / self.C_layer                        # (nx, nz) radiative heating [K]
        # --- surface turbulent fluxes (bulk aerodynamic): SENSIBLE -> lowest layer; LATENT -> vapor.
        #     The exchange WIND is the fixed cfg.wind by default; with resolved_flux_wind (#85 item 1) and
        #     the dynamics live it is the RESOLVED surface cross-track wind |u_surf| with a gustiness floor
        #     (sqrt(u_surf^2 + gustiness^2)), so evaporation follows where the cell's surface branch blows
        #     (the WISHE-like feedback). Then Ck/Ce become (nx,) per-column; SH/LE/dRdT below broadcast. ---
        wind = self._exchange_wind()                              # resolved |u_surf|+gust, or scalar cfg.wind
        # The 1.2e-3 bulk transfer coefficient (Ch=Ce) is the STANDARD ~10 m exchange value. On the uniform
        # grid it is applied at the lowest level z1 ~ 1.2 km (a documented mismatch -- the coefficient is a
        # 10 m value used a kilometre up); on the BL-refined grid z1 ~ 10-30 m, so the SAME coefficient is
        # now applied at the height it is calibrated for (no value change -- the fix is the height, not the
        # number; BL_REFINEMENT_SPEC.md #F).
        Ck = C.RHO_AIR0 * C.CP_AIR * 1.2e-3 * wind
        Ce = C.RHO_AIR0 * C.L_VAP * 1.2e-3 * wind
        wetness = self._wetness()
        qs_surf = th.qsat(self.T_surf, C.P_SURF)
        SH = Ck * (self.T_surf - self.T_atm[:, -1])               # W/m^2 into the lowest layer
        # LE = wetness * E_pot(T_surf): the evaporation stays RESPONSIVE to T_surf so a fully-wet surface
        # (f=1) self-consistently relaxes to its evaporative-thermostat T_w (the wetness=1 SEB root) -- the
        # cool-wet equator. NOTE (PR #177 review S1, B5b/B5d): in the PARTIAL regime 0<f<1 this evaporates
        # the wet fraction at the cell's hot T_surf rather than the wet tiles' cooler T_w, so it
        # OVER-evaporates (E>F) -- the single-bucket T_surf cannot represent the two-temperature pond-field
        # mosaic. Closing f*E_w=F in the interior needs that mosaic (carry T_w and T_d), deferred to B5b/B5d.
        LE = wetness * Ce * xp.maximum(qs_surf - q[:, -1], 0.0)   # W/m^2 (absolute footing; wetness-gated)
        _le_capped = None
        if c.surf_water_budget:
            # --- #302 HARD WATER LIMIT: a column cannot evaporate more water than it holds this substep
            #     (E <= W/dt). The smooth wetness f = W/W_min already keeps E*dt << W at production
            #     substeps (E*dt/W = f*E_pot*dt/W <= E_pot*dt/W_min ~ 1e-2 at dt=60 s); the cap is the
            #     GUARANTEE at any dt (test configs run dt up to 7200 s, where the ramp alone would
            #     overdraw). The SEB below uses this SAME limited LE in R_net (and drops the latent
            #     dqsat/dT stiffness where the cap binds -- the cap is water-set, T-independent), and the
            #     q source + self._evap read it too, so energy-water consistency is structural. th._F:
            #     the cap follows the fp32 thermo gate (W itself stays the fp64 accumulator); dt may be
            #     a 0-d device scalar under capture (python-float/device -> device, no host sync).
            _le_lim = th._F(self.W_surf * (C.L_VAP / dt), xp)
            _le_capped = LE >= _le_lim
            LE = xp.minimum(LE, _le_lim)
        if c.capture_ready:
            self._evap[...] = LE / C.L_VAP                        # in-place into the fixed (nx,) buffer
        else:
            self._evap = LE / C.L_VAP                             # surface evaporation E [kg/m^2/s] (M2b.5)
        dTa[:, -1] += dt * SH / self.C_layer[-1]
        # PLAIN add (NOT _accumulate -- that adds an fp32 cast the original didn't): OFF is the verbatim
        # fp64-intermediate rebind; ON writes the same sum IN PLACE (xp.add out=, bit-identical value).
        if c.capture_ready:
            xp = get_xp(self.T_atm); xp.add(self.T_atm, dTa, out=self.T_atm)
        else:
            self.T_atm = self.T_atm + dTa
        # --- M2b.4: the prognostic SURFACE ENERGY BALANCE (T_surf an OUTPUT, never imposed; #5). The
        #     surface gains its absorbed SW + downwelling LW (both in out["surf_net"]) and LOSES sigma
        #     T^4 + the SAME SH + LE it just deposited into the air -- so the column is energy-consistent
        #     (the surface's loss == the atmosphere's gain). SEMI-IMPLICIT in T_surf (sigma T^4, SH, LE
        #     all stiff at high SST via Clausius-Clapeyron), exactly Stage A's unconditionally-stable
        #     evaporative-thermostat step. The mixed-layer heat capacity sets only the spin-up rate.
        #     Gated by couple_surface: when False, T_surf is the M2b.3 prescribed fixed-SST boundary. ---
        if self.couple_surface:
            R_net = out["surf_net"] - SH - LE                     # surface tendency numerator [W/m^2]
            if c.surf_water_budget:
                # #302: where the water cap binds, LE is water-set (T-independent) -> zero latent
                # dqsat/dT stiffness in the semi-implicit denominator; uncapped columns keep the
                # verbatim term. (Flag OFF takes the untouched line below -- byte-identical.)
                dRdT = (4 * C.SIGMA_SB * self.T_surf**3 + Ck
                        + xp.where((LE > 0) & (~_le_capped),
                                   wetness * Ce * th.dqsat_dT(self.T_surf, C.P_SURF), 0.0))
            else:
                dRdT = (4 * C.SIGMA_SB * self.T_surf**3 + Ck
                        + xp.where(LE > 0, wetness * Ce * th.dqsat_dT(self.T_surf, C.P_SURF), 0.0))
            Csurf = self._surf_heat_capacity()
            seb_incr = dt * R_net / (Csurf + dt * dRdT)           # the SEB T_surf increment [K]
            if c.surf_flash_boil:
                # --- SURFACE FLASH-BOIL (the surface-side completion of #294; OPEN_ISSUES (B), config
                #     note at ``surf_flash_boil``): rain landing on a skin above ITS boiling point
                #     flash-evaporates on contact, up to the surface ENERGY cap -- the flux whose latent
                #     debit cools the (post-SEB) skin exactly to T_boil(P_SURF):
                #       surf_cap = (T_eff - T_boil) * C_surf / (L_VAP * dt)
                #       units: K * (J m^-2 K^-1) / ((J kg^-1) * s) = kg m^-2 s^-1  (the #294 boil_cap
                #       with the layer heat content cp*dm replaced by the skin's C_surf).
                #     min(): flux beyond the cap lands as LIQUID normally (the skin is no longer
                #     superheated once at T_boil) -- the #296 continuity-at-the-mask-edge lesson at the
                #     surface: cap -> 0 as T_eff -> T_boil+, so the closure is continuous and no state
                #     pins. The debit is EXPLICIT (-L_VAP*flash*dt/Csurf, NOT through the semi-implicit
                #     denominator) so the energy bookkeeping is exact: Csurf*dT_flash == -L_VAP*flash*dt,
                #     and at the cap the skin lands AT T_boil to round-off. Folding it into ``seb_incr``
                #     BEFORE the branches below routes it through the SAME carried fp64 path as the rest
                #     of the SEB (the thermo-fp32 freeze fix), never a bypass. ``self._precip`` is the
                #     previous substep's surface-reaching flux -- the same one-substep lag ``_wetness``
                #     already runs on ("lagged, well-posed"); the matching vapour source is added beside
                #     the evaporation below. In-place write into the persistent (nx,) buffer + a static
                #     config branch -> capture-safe; OFF skips everything (byte-identical).
                T_eff = self.T_surf + seb_incr                    # post-SEB skin this substep [K]
                surf_cap = xp.maximum(T_eff - self._T_boil_surf, 0.0) * Csurf / (C.L_VAP * dt)
                flash = xp.minimum(xp.maximum(self._precip, 0.0), surf_cap)   # [kg/m^2/s]
                self._surf_flash[...] = flash                     # AFTER _surf_heat_capacity: Csurf/wetness
                #     this substep read the LAGGED flash, consistent with the wetness/albedo calls above
                seb_incr = seb_incr - flash * (C.L_VAP * dt / Csurf)   # exact latent debit [K]
            if getattr(self, "_seb_fix_active", False):
                # FREEZE FIX: under thermo_fp32, self.T_surf here is the fp32 SCRATCH; adding the
                # ~1.7e-6 K ocean increment into it swamps under the ~3e-5 K fp32 ULP at ~278 K (then
                # the wrapper's difference-reconcile re-quantizes it to 0 -> bit-frozen). Instead CARRY
                # seb_incr (correct + representable as a standalone fp32 value) into the persistent fp64
                # buffer; the fp32 wrapper applies it to the fp64 T_surf absolute. (nx,) only -> cheap.
                self._seb_incr_buf[...] = seb_incr
            elif c.capture_ready:                                 # plain add in-place (see the T_atm note above)
                xp = get_xp(self.T_surf); xp.add(self.T_surf, seb_incr, out=self.T_surf)
            else:
                self.T_surf = self.T_surf + seb_incr
        # --- THERMO-fp32 GATE intermediate capture (guarded; byte-identical when ``_fp32_dbg`` unset).
        #     Snapshots the RAW in-substep products (radiation H/OLR, the surface fluxes SH/LE, the SEB
        #     increment) -- the genuine-fp32 completeness probe needs these BEFORE the increment reconciliation
        #     flattens the state-field signal. These are real-magnitude fluxes, so fp32 shows ~1e-4-1e-6 and a
        #     residual fp64 sub-path would show ~1e-13 in its own product (the false-PASS guard).
        if getattr(self, "_fp32_dbg", None) is not None:
            # DIAGNOSTIC snapshots only (the genuine-fp32 completeness probe) -> bring to host (to_host is a
            # no-op on numpy). Off the physics path; fp32 granularity is read from these host copies.
            self._fp32_dbg.update(SH=np.asarray(to_host(SH), np.float64).copy(),
                                  LE=np.asarray(to_host(LE), np.float64).copy(),
                                  H=np.asarray(to_host(out["H"]), np.float64).copy(),
                                  OLR=np.asarray(to_host(out["OLR"]), np.float64).copy(),
                                  seb_incr=np.asarray(to_host(seb_incr), np.float64).copy()
                                  if self.couple_surface else np.zeros(1),
                                  Tatm_preconv=np.asarray(to_host(self.T_atm), np.float64).copy())
        # evaporation moistens the lowest layer (E = LE/L_vap, kg/m^2/s, into the layer mass dm[-1])
        q = q.copy()
        q[:, -1] += (LE / C.L_VAP) * dt / self.dm[-1]
        if c.surf_flash_boil and self.couple_surface:
            # SURFACE FLASH-BOIL vapour source: the flux flashed off the super-boiling skin this substep
            # (the SEB block above) re-enters the boundary layer as VAPOUR -- the exact water the skin's
            # latent debit paid for (water+energy conjugate; see the SEB note). Gated on couple_surface
            # like the flash itself (a prescribed skin never writes _surf_flash; the buffer stays 0).
            q[:, -1] += self._surf_flash * dt / self.dm[-1]
        if c.surf_water_budget:
            surface_water_salt_step(self, c, dt, xp)
        # TWO-WAY NEST (chimney_nest): snapshot the PRE-convective column (post radiation/SEB/evaporation) so
        # the resolved chimney response can REPLACE the inner-zone convective result after the block. Persistent
        # buffers (fixed addresses -> capture-safe), realloc on dtype/shape change (the _conv_Tpre pattern).
        if c.chimney_nest and getattr(self, "_nest_active", False):
            if (getattr(self, "_nest_Tpre", None) is None or self._nest_Tpre.shape != self.T_atm.shape
                    or self._nest_Tpre.dtype != self.T_atm.dtype):
                self._nest_Tpre = xp.empty_like(self.T_atm); self._nest_qpre = xp.empty_like(q)
                self._nest_scratch = xp.empty_like(self.T_atm); self._nest_scratch2 = xp.empty_like(self.T_atm)
            self._nest_Tpre[...] = self.T_atm
            self._nest_qpre[...] = q
        # Betts-Miller finite-tau: snapshot the PRE-adjustment column so we can relax fractionally toward
        # the (instantaneously) adjusted target instead of snapping to it (conv_tau=0 -> instantaneous).
        # The scale-aware gate (stretched-x gray zone) rides the SAME snapshot/blend: it multiplies the
        # relaxation fraction per column, and with conv_tau == 0 it blends toward the adjusted state by
        # w_conv alone -- either way a per-column-scalar linear blend of two enthalpy/water-conserving
        # states, so column conservation is exact (the _bm argument below, unchanged).
        _gate = self._conv_gate_col is not None
        _bm = c.conv_tau > 0.0
        # PHY-19 (b): the min-scale filter needs the SAME pre-adjustment snapshot, so it forces the
        # snapshot on even when neither the finite-tau blend nor the gray-zone gate would take it
        # (section 5(b) step 1). The width is re-read here so a warm-start route that sets
        # cfg.conv_min_scale_m on a BUILT core takes effect; the rebuild happens on the eager warmup
        # step and never inside a captured graph (capture replays a recorded graph, it does not re-enter
        # this Python). A PYTHON-level branch, never a zero-width operator: the OFF graph must not
        # contain the filter at all.
        _minsc = bool(c.conv_min_scale_m and float(c.conv_min_scale_m) > 0.0)
        if _minsc and float(c.conv_min_scale_m) != self._minsc_W:
            self._build_min_scale_operator()
        _minsc = _minsc and self._minsc_n > 0    # a grid coarser than W everywhere is an exact no-op
        if _bm or _gate or _minsc:
            # persistent snapshot buffers (fixed addresses -> CUDA-graph-capture safe; in-place copy of the
            # live pre-adjustment state). Allocated lazily to match the active dtype (fp32 under thermo_fp32).
            if (getattr(self, "_conv_Tpre", None) is None or self._conv_Tpre.shape != self.T_atm.shape
                    or self._conv_Tpre.dtype != self.T_atm.dtype):
                self._conv_Tpre = xp.empty_like(self.T_atm)
                self._conv_qpre = xp.empty_like(q)
            self._conv_Tpre[...] = self.T_atm
            self._conv_qpre[...] = q
        # --- moist convective adjustment: mixes T (to the moist adiabat) AND lofts q (the sub-grid
        #     VERTICAL moisture transport), both conservatively (enthalpy + water); ``convective_closure``
        #     selects the q redistribution (rh_detrain is the production config default / q_homogenize the
        #     compared gray variant, CLEARSKY_SW_PLAN Stage 3 -- the function default arg is q_homogenize). ---
        # OPERATOR SPLIT with the prognostic scheme (CLOUD_PROGNOSTIC_ROADMAP Chunk 5): this convective
        # vapour redistribution runs FIRST and conserves column water EXACTLY -- it only MOVES vapour
        # between layers (rh_detrain can leave a layer supersaturated where the detrained water exceeds the
        # fixed-RH holding capacity), it NEVER condenses or precipitates. The prognostic condensation
        # (_prognostic_microphysics, below) then reads the post-detrainment ``q`` as the vapour the
        # condensation source sees, so the two partition cleanly: the convective closure owns VAPOUR
        # TRANSPORT, the prognostic scheme owns CONDENSATE -- there is no path where rh_detrain and the
        # prognostic condensation both claim the same supersaturation (the joint d/dt INTEGRAL(q+q_c)=E-P
        # budget closes, gated by test_cloud_prognostic.py's Chunk-5 tests).
        # CPU-fuse: the column-PARALLEL numba kernel (default OFF) when it can reproduce the numpy path to
        # round-off -- i.e. NOT freeze_gamma (kernel recomputes Gamma live), q present, and fp64. Otherwise
        # the numpy path (BYTE-IDENTICAL default). The numpy function is left untouched (OFF bit-identity is
        # structural). q is a real ndarray here (set just above), and T_atm/_z/p_mid are fp64 C-contiguous.
        # PHY-17 step 4: the CONVERGED adjustment. When ``conv_segment_mixer`` is ON the capped sweep is
        # replaced by the one-pass segment mixer -- the SAME operator (same pairwise predicate, same moist
        # adiabat, same enthalpy/water theorems, same 1e-9 K stopping rule) solved to its fixed point
        # instead of relaxed four sweeps toward it, at a cost bounded by conv_segment_max_passes rather
        # than by the depth-squared sweep count convergence needs. ``conv_max_sweeps`` is not read here.
        # This branch is FIRST so the mixer, when ON, wins over the two fast paths below; the only textual
        # change to the three shipped branches is the ``if`` -> ``elif`` keyword on the next one, and the
        # whole block is unreachable at the default (conv_segment_mixer=False) -> OFF is byte-identical.
        # FALLBACK (mirroring the convadjust_parallel guard): conv_freeze_gamma ON (the mixer always
        # evaluates Gamma live, at the converged temperatures) or q is None (the device kernel always
        # writes q) drop through to the shipped sweep. On device the fused RawKernel runs it one thread
        # per column (built once per C-precision on the eager warmup, never inside capture).
        # TRUNCATION DIAGNOSTIC: on the host path the routine raises a RuntimeWarning when a column hits
        # the pass bound with its active set still growing; on the device it writes each column's passes
        # used (negated when truncated) into ``self._segmix_aux[dtype]["status"]``, a persistent buffer
        # allocated in warmup -- read it at a BATCH boundary (never per step: that would sync the capture).
        if c.conv_segment_mixer and not c.conv_freeze_gamma and q is not None:
            # ``_on_device``, NOT the ``_dyn_on_device`` the fused-convadjust branch below uses: the thermo
            # arrays are CuPy whenever _on_device (_place_thermo_on_device), while _dyn_on_device also
            # requires poisson_method == "dst". Gating on the latter would hand the numpy-only host body a
            # device array (it raises) on a device run with a non-DST Poisson solver.
            skernel = self._get_segmix_kernel(self.T_atm.dtype) if self._on_device else None
            moist_convective_segment_mix(self.T_atm, self.dm, self._z, self.p_mid, q=q,
                                         closure=c.convective_closure, rh_surf=c.rh_surf, rh_top=c.rh_top,
                                         max_passes=c.conv_segment_max_passes,
                                         gamma_iters=c.conv_segment_gamma_iters,
                                         fixed_topology=bool(c.capture_ready),
                                         device_kernel=skernel, device_aux=self._segmix_aux)
        elif (c.convadjust_parallel and not c.conv_freeze_gamma and q is not None
                and c.precision == "fp64" and not self._on_device):   # numba is host-only -> device uses numpy-generic path
            moist_convective_adjust_parallel(self.T_atm, self.dm, self._z, self.p_mid, c.conv_max_sweeps,
                                             q=q, closure=c.convective_closure,
                                             rh_surf=c.rh_surf, rh_top=c.rh_top)
        elif c.capture_ready:
            # FUSED DEVICE KERNEL (Phase E profile follow-up): on the device capture path replace the
            # ~5700-launch Gauss-Seidel Python loop (84% of the captured step) with ONE column-parallel
            # RawKernel. Gated on _dyn_on_device + no freeze_gamma (the kernel computes Gamma live, matching
            # the default Gauss-Seidel) -- mirroring the numba-fallback conditions. The host/non-device
            # capture path (rare) keeps the fixed-topology Python loop (byte-identical to OFF). The kernel is
            # built once for T_atm's dtype (fp32 on the captured fp32 path) and cached on the core; first-call
            # NVRTC compile happens during the eager warmup (the step path), never inside capture.
            dkernel = None
            if self._dyn_on_device and not c.conv_freeze_gamma:
                dkernel = self._get_convadj_kernel(self.T_atm.dtype)
            moist_convective_adjust(self.T_atm, self.dm, self._z, self.p_mid, c.conv_max_sweeps, q=q,
                                    closure=c.convective_closure, rh_surf=c.rh_surf, rh_top=c.rh_top,
                                    freeze_gamma=c.conv_freeze_gamma, fixed_topology=True,
                                    device_kernel=dkernel,        # GPU capture: ONE fused kernel (or drop the exits)
                                    device_aux=self._convadj_aux)  # persistent static aux (p_c/dm_c/RHt_c); item 1
        else:
            moist_convective_adjust(self.T_atm, self.dm, self._z, self.p_mid, c.conv_max_sweeps, q=q,
                                    closure=c.convective_closure, rh_surf=c.rh_surf, rh_top=c.rh_top,
                                    freeze_gamma=c.conv_freeze_gamma)   # OFF: verbatim shipped call (byte-identical)
        if _minsc:
            # PHY-19 (b): refuse to let the adjustment deliver its heating and drying to a column
            # narrower than conv_min_scale_m. Runs on WHICHEVER branch above ran, before the blend.
            self._apply_min_scale_filter(q)
        if _bm or _gate:
            # relax fractionally toward the adjusted target: X <- X_pre + min(dt/tau,1)*(X_adj - X_pre).
            # Linear blend of two enthalpy/water-conserving states -> conserves both exactly (the SAME scalar
            # frac is applied UNIFORMLY to T and q across all layers, so the column integrals telescope -- a
            # vertically/species-varying tau would break that; a per-COLUMN scalar -- the gate -- telescopes
            # identically). O(dt) tendency -> timestep-convergent;
            # tau->0 recovers the instantaneous scheme. The ARRAY ops are in-place, NO array temp allocations
            # (capture-safe: a temp alloc inside the captured step can alias a live buffer on replay):
            # X_adj -= X_pre; X_adj *= frac; X_adj += X_pre  ==  X_pre + frac*(X_adj - X_pre). (frac itself is
            # a 0-d device scalar -- the same graph-clean flavour the dynamics already mint inside capture;
            # conv_tau is code-default-OFF, but the CANONICAL bm_tau2h world runs conv_tau=7200, so the
            # production capture path DOES replay this block -- hence the in-place discipline. The GATE
            # branch allocates an (nx,1) product and is guarded OFF under capture_ready at init.)
            if _bm:
                frac = xp.minimum(dt / c.conv_tau, 1.0)      # device-scalar-safe (dt may be a device scalar)
                if _gate:
                    frac = frac * self._conv_gate_col        # (nx, 1) per-column gated fraction (CPU path)
            else:
                frac = self._conv_gate_col                   # pure gate at conv_tau == 0 (instantaneous)
            if _gate:
                # who-lifts diagnostic: per-column |enthalpy| the ADJUSTMENT wanted to move (pre-blend),
                # accumulated so "who does the lifting vs x" is measurable across the gray zone.
                self._conv_gate_heat += np.abs(np.asarray(self.T_atm) - np.asarray(self._conv_Tpre)
                                               ).dot(self.C_layer)
            self.T_atm -= self._conv_Tpre; self.T_atm *= frac; self.T_atm += self._conv_Tpre
            q -= self._conv_qpre; q *= frac; q += self._conv_qpre
        # condensation of supersaturation + latent heat to the air (on-grid, energy-consistent). The
        # condensate's FATE differs by mode (CLOUD_PROGNOSTIC_ROADMAP):
        #   OFF (default, byte-identical): all supersaturation rains immediately (single-pass).
        #   ON: the real-microphysics step (Chunk 2) -- dqsat/dT-corrected saturation adjustment SOURCES the
        #       retained reservoir q_c, suspended q_c RE-EVAPORATES in sub-saturated air, and a rate
        #       autoconversion (+ a retained-water ceiling) RAINS it, all phase-aware (liquid/ice) and under a
        #       coupled-sink positivity limiter. OFF sub-flags reduce it to the Chunk-1 skeleton.
        if getattr(self, "_fp32_dbg", None) is not None:
            # convadjust ΔT = post-convadjust T_atm minus the pre-convadjust snapshot captured above
            self._fp32_dbg["convadj_dT"] = (np.asarray(to_host(self.T_atm), np.float64)
                                            - self._fp32_dbg["Tatm_preconv"])
        if c.prognostic_condensate:
            # The CANONICAL prognostic-cloud combination (the only config the fused micro+reevap kernels
            # mirror): pdf source + dqsat denom + ice phase + reevaporation + retained decoupled ceiling +
            # precip_reevap + kessler autoconv, no equator suppression / chimney. Both the numba CPU-fuse and
            # the GPU device kernel cover EXACTLY this; any other config falls to the numpy-generic method.
            _micro_canonical = (c.qc_pdf_cloud and c.qc_dqsat_adjust and c.qc_ice_phase and c.qc_reevaporation
                                and c.qc_retained_ceiling and c.qc_radiative_cap_decouple and c.precip_reevap
                                and c.equator_norain_cells == 0 and c.qc_autoconv_scheme == "kessler"
                                and not c.chimney_param and not c.cond_tau_salt and not c.prognostic_rain)
                                #   prognostic_rain (stage 7) reroutes the autoconversion sink into q_r --
                                #   NOT mirrored by the fused kernels; it must take the generic path below.
                                #   #280 finite-tau condensation (constant cond_tau), #237 salt N_d warm-rain
                                #   autoconv, and the #294 boiling-layer physics (virga_flash_boil energy-cap
                                #   virga + cond_boil_suppress, two int flags) are all THREADED into the fused
                                #   numba/device kernels (cond_tau scalar + warm_factor (nx,) via
                                #   _micro_fuse_salt_cond_params); all reduce to the shipped arithmetic when
                                #   inactive (cond_tau=0 / warm never read / flags=0 -> the boil branches are
                                #   skipped, byte-identical). Only the per-column cond_tau_salt
                                #   (spatially-varying tau) stays generic-routed.
            if (c.thermo_other_parallel and c.precision == "fp64" and _micro_canonical and not self._on_device):
                # CPU-fuse #4: the column-PARALLEL numba fused microphysics+reevap (default OFF, host-only).
                q = prognostic_microphysics_reevap_parallel(self, q, dt)
            elif (self._on_device and c.capture_ready and _micro_canonical
                  and getattr(self, "_micro_fuse_enabled", True)):
                # FUSED DEVICE KERNEL (Phase E follow-up #3): the captured step's largest limb (~7 ms / 39%) as
                # ONE column-parallel RawKernel (mirrors the numba fuse / _micro_numba). Gated to the device
                # capture path + canonical config; the kernel is built once in warmup, never inside capture.
                # (``_micro_fuse_enabled`` is a perf-A/B toggle, default ON; never set False in production.)
                q = self._micro_device(q, dt)
            else:
                q = self._prognostic_microphysics(q, dt)   # numpy-generic (BYTE-IDENTICAL default)
            if c.prognostic_rain:
                # PROGNOSTIC RAIN (stage 7): the microphysics just SOURCED q_r (the rerouted
                # autoconversion+ceiling sink); now sediment it at -rain_v_term (implicit; the surface-
                # face flux -> self._precip) and re-evaporate it in sub-saturated air (latent cooling on
                # the real prognostic T_atm -- the cold-pool mechanism). Runs every thermo substep, so
                # the rain reservoir evolves on the same clock as its source.
                q = self._prognostic_rain_step(q, dt)
        else:
            if c.cond_pdf_tail:
                # sub-grid PDF-tail condensation target (ROADMAP GPU-7, default OFF -> the grid-mean line
                # below, byte-identical): condense the FIRST moment of the supersaturated tail of the SAME
                # total-water PDF whose ZEROTH moment is the radiation's cloud cover
                # (rad.cloud_pdf_partition; q_t = q + q_c, q_c == 0 on this path) -- cover and condensate
                # become ONE closure. qc_eq = E[(q_t - qsat)^+]: the box drizzles its tail excess when
                # partly cloudy (fires wherever f > 0, i.e. down toward RH -> rh_crit as the tail clears)
                # and reduces BIT-IDENTICALLY to the grid-mean adjustment when overcast (Q1 >= 1 =>
                # qc_eq = q_t - qsat). The cond_tau frac below still sets the RATE (target from the PDF,
                # rate from tau). Single-pass fate is unchanged: all condensed water rains immediately.
                _, cond = rad.cloud_pdf_partition(q + self.q_c, self.T_atm, self.p_mid[None, :],
                                                  c.cloud_rh_crit)
            else:
                cond = xp.maximum(q - th.qsat(self.T_atm, self.p_mid[None, :]), 0.0)
            if c.cond_boil_suppress:
                # #294: no vapour-liquid equilibrium above boiling -- suppress the single-pass condensation
                # in boiling layers (esat >= 0.999 p, where the qsat clamp fabricates a finite "saturation");
                # the supersaturation there is STEAM and advects on (the hot zone is a vapour pass-through).
                cond = xp.where(th.esat(self.T_atm) >= 0.999 * self.p_mid[None, :], 0.0, cond)
            if c.cond_tau > 0.0:
                # finite-tau saturation adjustment (#280): condense only frac = min(dt/tau, 1) [dimensionless]
                # of the supersaturation this sub-step (tau->0 -> frac=1 -> the shipped single pass; the
                # residual supersaturation carries to the next sub-step). ONE frac scales cond everywhere it
                # is used below (q sink, latent heat, precip), so water/energy consistency is untouched.
                # In-place on the fresh temp; frac is a 0-d device scalar (graph-clean, the conv_tau flavour).
                cond *= xp.minimum(dt / c.cond_tau, 1.0)
            q = q - cond
            if c.capture_ready:                                  # plain add in-place (see the T_atm note above)
                xp.add(self.T_atm, C.L_VAP * cond / C.CP_AIR, out=self.T_atm)
                self._precip[...] = xp.sum(cond * self.dm[None, :], axis=1) / dt   # in-place (nx,) buffer
            else:
                self.T_atm = self.T_atm + C.L_VAP * cond / C.CP_AIR
                self._precip = xp.sum(cond * self.dm[None, :], axis=1) / dt  # single-pass: all supersaturation rains
        if getattr(self, "_fp32_dbg", None) is not None:
            self._fp32_dbg["precip"] = np.asarray(to_host(self._precip), np.float64).copy()
        self._apply_chimney_param()   # B5c: de-spike the equatorial 1/dx precip (default OFF -> byte-identical)
        # the local ``q`` is a fresh array (clip/copy + microphysics return). OFF: the verbatim rebind
        # (``self.q = q``; the final _cast below applies the dtype). ON: copy q's bits into the fixed q
        # buffer in place -- the SAME bits the rebind would have stored, then the final _cast (no-op fp64 /
        # idempotent fp32) keeps it consistent. (Kept distinct from the final _cast so OFF is verbatim.)
        if c.capture_ready:
            self.q[...] = q
        else:
            self.q = q
        self._apply_chimney_nest(dt)   # two-way nest: replace inner-zone convection with the resolved chimney
        # --- M2b.5: accumulate the area-weighted water-budget integrals (the surface evaporation source E
        #     and the column condensation sink P), uniform-dx cross-track weight, over this dt. The
        #     cumulative ratio P/E -> 1 at steady state (tier-1 water closure; CLOSED_ISSUES c5). ---
        # CAPTURE-READY (Phase B(v)): the budget integrals are DIAGNOSTICS (not state the step needs), and
        # the per-step ``float(np.sum(...))`` is a device->host sync on the hot path. Accumulate the
        # per-step area-sum into a device-resident (nx,)->scalar pending buffer and pull to the host scalar
        # only at the checkpoint cadence (``flush_budget``). VALUE is preserved -- the same sum*dt terms,
        # summed in the same order -- so the cumulative totals match at any flush. OFF = the verbatim
        # per-step host accumulation (byte-identical).
        if c.capture_ready:
            # IN-PLACE pending accumulation (#302 review finding 2 -- see ``_accum_pending``): the recorded
            # in-place add re-executes per captured replay, so these integrals genuinely accumulate under
            # the captured production driver (the old rebind baked only the last replayed step's term).
            # Eagerly value-identical (same terms, same order).
            self._cum_evap_pending = self._accum_pending(
                self._cum_evap_pending, xp.sum(self._evap * self._dx_w) * dt, xp)
            self._cum_precip_pending = self._accum_pending(
                self._cum_precip_pending, xp.sum(self._precip * self._dx_w) * dt, xp)
            if c.surf_flash_boil:   # surface flash-boil: the flashed (vapour-returned) area-sum, same idiom
                self._cum_surf_flash_pending = self._accum_pending(
                    self._cum_surf_flash_pending, xp.sum(self._surf_flash * self._dx_w) * dt, xp)
            if c.surf_water_budget:   # #302: the LANDED-liquid area-sum -- sampled from the stashed _p_land
                #   (the lagged flux the W update consumed; the live self._precip was just overwritten by
                #   the microphysics), so the surface closure is exact, not off by a substep.
                self._cum_p_land_pending = self._accum_pending(
                    self._cum_p_land_pending, xp.sum(self._p_land * self._dx_w) * dt, xp)
        else:
            self._cum_evap += float(xp.sum(self._evap * self._dx_w)) * dt
            self._cum_precip += float(xp.sum(self._precip * self._dx_w)) * dt
            if c.surf_flash_boil:   # surface flash-boil: the flashed (vapour-returned) area-sum, same idiom
                self._cum_surf_flash += float(xp.sum(self._surf_flash * self._dx_w)) * dt
            if c.surf_water_budget:   # #302 landed-liquid area-sum (see the capture branch note)
                self._cum_p_land += float(xp.sum(self._p_land * self._dx_w)) * dt
        # refresh the layer-center heights from the POST-thermo T_atm so the resolved-flow advection
        # (which interpolates the winds at z_p) uses heights consistent with the field it transports --
        # not the pre-thermo z_p cached at the top of this step.
        self._z = self._hydrostatic_z(self.T_atm)
        # fp32 spike: the column thermo above runs fp64-internally (radiation.py/surface.py/thermo.py +
        # fp64 constants upcast the arithmetic; those shared modules are NOT made dtype-generic), so the
        # prognostic STATE is reduced back to the working dtype here (no-op under fp64 -> byte-identical).
        # capture-ready: cast IN PLACE into the fixed buffers (a fixed address per graph capture).
        self.T_atm = self._cast(self.T_atm, out=self._obuf(self.T_atm))
        self.q = self._cast(self.q, out=self._obuf(self.q))
        self.T_surf = self._cast(self.T_surf, out=self._obuf(self.T_surf))
        self.q_c = self._cast(self.q_c, out=self._obuf(self.q_c))
        self._last_dt_th = dt

    # ------------------------------------------------------------------ prognostic cloud microphysics (Chunk 2)
    def _cloud_cover(self, q):
        """The cloud cover the prognostic scheme reads -- the SAME cover the radiation uses
        (``_radiate``/``column_radiation``), so the retained-water ceiling (in-cloud LWC = rho*q_c/f vs the
        radiative LWC bound) sits on ONE footing with the optics. Two footings, selected by ``qc_pdf_cloud``:
          * Chunk 3 (default): the sub-grid total-water PDF cover (``rad.cloud_pdf_partition`` on q + q_c).
          * Chunk 1/2: the RH-diagnostic cover (``rad.cloud_fraction``) FLOORED to ``cloud_f_min`` wherever
            ``q_c > qc_cover_thresh`` (clear air + condensate is unphysical)."""
        c = self.cfg
        if c.qc_pdf_cloud:
            f, _ = rad.cloud_pdf_partition(q + self.q_c, self.T_atm, self.p_mid[None, :], c.cloud_rh_crit)
            return f
        xp = get_xp(q)                                            # device-generic (Phase D)
        f = rad.cloud_fraction(q, self.T_atm, self.p_mid[None, :], c.cloud_rh_crit)
        return xp.where(self.q_c > c.qc_cover_thresh, xp.maximum(f, c.cloud_f_min), f)

    def _autoconversion_demand(self, qc, ice, dt):
        """Per-step autoconversion removal demand [kg/kg] (cloud water -> precip), phase-split when
        ``qc_ice_phase``. ``qc_autoconv_scheme`` selects the closure (best estimates, REPORTED as a
        sensitivity, never fitted -- roadmap rule 4):
          "kessler"  : threshold+timescale  P = max(q_c - crit, 0)/tau
          "sundqvist": smooth onset         P = q_c/tau * (1 - exp(-(q_c/crit)^2))   (the compared variant).
        The ICE branch (below T_FREEZE) uses ``qc_autoconv_tau_ice``/``qc_autoconv_crit_ice`` -- ice
        aggregation->snow, which precipitates more readily (lower threshold)."""
        c = self.cfg
        xp = get_xp(qc)                                           # device-generic (Phase D)
        if c.qc_ice_phase:
            tau = xp.where(ice, c.qc_autoconv_tau_ice, c.qc_autoconv_tau)
            crit = xp.where(ice, c.qc_autoconv_crit_ice, c.qc_autoconv_crit)
        else:
            tau, crit = c.qc_autoconv_tau, c.qc_autoconv_crit
        if c.qc_autoconv_scheme == "sundqvist":
            p_auto = qc / tau * (1.0 - xp.exp(-(qc / xp.maximum(crit, 1e-30)) ** 2))   # smooth onset [kg/kg/s]
        else:
            p_auto = xp.maximum(qc - crit, 0.0) / tau                                  # Kessler threshold [kg/kg/s]
        # Salt-aerosol CCN warm-rain efficiency limb (#237; default OFF -> no scaling, byte-identical). Scale
        # the WARM (liquid) autoconversion DEMAND by (N_d/N_ref)^beta: beta<0 SUPPRESSION (sub-um CCN ->
        # smaller, more numerous drops -> delayed coalescence) | beta>0 INVIGORATION (giant salt CCN seed large
        # drops). LIQUID only -- salt is a warm-rain CCN; the ice (snow-aggregation) branch is untouched. At
        # N_d == N_ref the factor is exactly 1.0 (the clean zero-point). Applied to the DEMAND here, BEFORE the
        # caller adds the retained-water ceiling force-rain -- so the ceiling-masking of this limb is diagnosable
        # (q_c/LWP can move while surface precip stays ceiling-clamped; see _salt_ccn_sweep.py).
        if c.salt_ccn and c.salt_ccn_autoconv:
            nd = th._F(self.N_d, xp)[:, None]                                            # (nx,1), device/dtype-safe
            warm_factor = (nd / c.salt_ccn_nd_ref) ** c.salt_ccn_autoconv_beta
            p_auto = p_auto * xp.where(ice, 1.0, warm_factor)
        return p_auto * dt

    def _prognostic_microphysics(self, q, dt):
        """The Chunk-2 (CLOUD_PROGNOSTIC_ROADMAP) real-microphysics saturation/precip step on the live column
        state, replacing the Chunk-1 skeleton's crude condense->rain. Updates ``self.T_atm``, ``self.q_c`` and
        ``self._precip``; returns the updated vapour ``q``. ENERGY- and WATER-consistent: condensation/
        re-evaporation move water between q and q_c at the SAME level with the matching latent heat sign, and
        only autoconversion+ceiling remove water from the column (the precip P). The Chunk-2 sub-flags reduce
        this to the skeleton (bare adjustment, no re-evap, warm-phase, no ceiling); in the instant-rain limit
        (tau->0, crit=0) the skeleton is byte-identical to the OFF single-pass core.

        Steps:
          1. PHASE: a BINARY ice/liquid split at T_FREEZE (qc_ice_phase) -- below it the latent heat is L_SUB
             (deposition/sublimation), else L_VAP; the same hard split the optics use, not a graded mixed phase.
          2. dqsat/dT-CORRECTED saturation adjustment (qc_dqsat_adjust): C = (q-qsat)/(1+(L/cp) dqsat/dT). The
             bare q-qsat over-condenses (latent warming raises qsat); one Newton step leaves the air at
             saturation TO FIRST ORDER (a large supersaturation lands slightly SUB-saturated, never residual-
             super -> it self-corrects the next step). The SAME denom corrects re-evaporation (cooling lowers qsat).
          3. CONDENSATION sources q_c; RE-EVAPORATION (qc_reevaporation) of suspended q_c sinks it back to
             vapour with latent COOLING. SOURCE footing (qc_pdf_cloud, Chunk 3 default): the saturated-tail
             supersaturation of the TOTAL-WATER PDF (rad.cloud_pdf_partition's qc_eq) -- the source fires from
             the tail at grid-mean RH < 1, sharing ONE PDF with the cover; OFF -> the grid-mean excess (Chunk 2).
          4. AUTOCONVERSION (rate) + the RETAINED-WATER CEILING (qc_retained_ceiling: in-cloud LWC=rho*q_c/f
             above the radiative LWC bound rains immediately) remove q_c to precip. That bound is the Chunk-3
             physical-plausibility LWC when qc_radiative_cap_decouple is on (one number with the decoupled
             optics), else the Chunk-1 adiabatic-rainout cap (CLOUD_LWC_MAX/IWC_MAX).
          5. COUPLED-SINK POSITIVITY LIMITER (review #4): the total per-step removal can never exceed the
             available q_c. Sequential cap (re-evaporation toward saturation first, then precip takes the
             remainder) -- exact positivity (q_c >= 0 to the bit) and a clean skeleton reduction."""
        c = self.cfg
        cp = C.CP_AIR
        xp = get_xp(self.T_atm)                                  # device-generic (Phase D); numpy on CPU
        qs = th.qsat(self.T_atm, self.p_mid[None, :])
        # #294 (cond_boil_suppress): the boiling/supercritical mask -- layers with NO vapour-liquid
        # equilibrium, where the qsat clamp would otherwise let cooling "condense" spurious q_c. None when
        # OFF (default) -> every use below is skipped, byte-identical.
        boil = (th.esat(self.T_atm) >= 0.999 * self.p_mid[None, :]) if c.cond_boil_suppress else None
        if c.qc_ice_phase:
            ice = self.T_atm < C.T_FREEZE
            L_eff = xp.where(ice, C.L_SUB, C.L_VAP)              # deposition/sublimation vs condensation/evap
        else:
            ice = xp.zeros(self.T_atm.shape, dtype=bool)
            L_eff = C.L_VAP
        # (2) saturation-adjustment Newton denominator (1 with the bare skeleton adjustment)
        if c.qc_dqsat_adjust:
            denom = 1.0 + (L_eff / cp) * th.dqsat_dT(self.T_atm, self.p_mid[None, :])
        else:
            denom = 1.0
        # (3) condensation source + re-evaporation sink. Two footings (qc_pdf_cloud):
        #   PDF (Chunk 3, default): the source is the saturated-tail supersaturation of the TOTAL-WATER PDF
        #     (rad.cloud_pdf_partition -> qc_eq), so the adjustment drives q_c toward qc_eq -- condensation
        #     (net > 0) FIRES from the tail even at grid-mean RH < 1, and re-evaporation (net < 0) drains the
        #     suspended deck toward the equilibrium partition. ONE PDF for cover (the ceiling/optics) AND source.
        #   Grid-mean (Chunk 2): the source is the grid-mean excess max(q - qsat, 0); re-evaporation the deficit.
        if c.qc_pdf_cloud:
            _, qc_eq = rad.cloud_pdf_partition(q + self.q_c, self.T_atm, self.p_mid[None, :], c.cloud_rh_crit)
            # NB (PR #144 review #1): ``denom`` is the OVERCAST-limit Newton correction (exact at Q1>=1, where
            # net=(q-qsat)/denom = the Chunk-2 grid-mean line), applied as an APPROXIMATE damping for partial
            # cloud (-1<Q1<1, where qc_eq depends nonlinearly on qsat via both qsat and Delta=(1-rh_crit)*qsat).
            # It is not the exact partial-cloud Newton step -- but denom>1 always, so it only DAMPS the
            # relaxation (never over-corrects/oscillates), and the fixed point is correct because qc_eq is
            # recomputed each step (this is why re-evaporation is per-step geometric, not single-pass).
            if boil is not None:
                # #294: no equilibrium condensate above boiling -- qc_eq -> 0 in boiling layers, so the
                # signed adjustment below both KILLS the condensation source (net <= 0) and DRAINS any
                # suspended/advected q_c back to vapour (evap_demand = q_c/denom; denom == 1 there exactly,
                # since dqsat_dT == 0 in the clamped branch). The fused kernels mirror this op-for-op.
                qc_eq = xp.where(boil, 0.0, qc_eq)
            net = (qc_eq - self.q_c) / denom                    # signed adjustment toward the PDF equilibrium
            cond = xp.maximum(net, 0.0)                         # condensation source -> q_c (from the PDF tail)
            evap_demand = xp.maximum(-net, 0.0) if c.qc_reevaporation else xp.zeros_like(self.q_c)
        else:
            excess = q - qs                                      # +supersaturated / -sub-saturated
            cond = xp.maximum(excess, 0.0) / denom              # condensation source -> q_c (grid-mean)
            if c.qc_reevaporation:
                evap_demand = xp.maximum(-excess, 0.0) / denom  # re-evaporation toward saturation (sub-sat cells)
            else:
                evap_demand = xp.zeros_like(self.q_c)
            if boil is not None:
                # #294 grid-mean footing: same suppression -- no condensation in boiling layers, and the
                # re-evaporation demand is the FULL suspended q_c (not the deficit toward the clamp qsat,
                # which is meaningless above boiling). denom == 1 on the mask (clamped dqsat_dT == 0).
                cond = xp.where(boil, 0.0, cond)
                if c.qc_reevaporation:
                    evap_demand = xp.where(boil, self.q_c / denom, evap_demand)
        if c.cond_tau > 0.0:
            # finite-tau saturation adjustment (#280): relax only frac = min(dt/tau, 1) [dimensionless] of
            # the way toward the saturation equilibrium (PDF qc_eq or grid-mean qsat) this sub-step, so the
            # snap becomes the finite-rate dq_c/dt -> (qc_eq - q_c)/(denom*tau) and the per-dynamics-step
            # total is n_sub-convergent. cond and evap_demand are the two signs of ONE adjustment ``net``,
            # so the SAME frac scales both (scaling the signed net); the q<->q_c exchange and its latent
            # heat stay exactly conjugate, and the positivity limiter below acts on the scaled demands
            # unchanged. tau->0 -> frac=1 -> the shipped instantaneous adjustment. The fixed point is
            # unchanged (the equilibrium is recomputed each step; the relaxation is geometric).
            if c.cond_tau_salt:
                # salt/CCN-coupled tau(x) (ROADMAP GPU-7): tau_phase is CCN-set, and the CCN is the #237
                # sea-salt N_d(x) -- so tau(x) = cond_tau * (N_ref/N_d)^gamma [s], short over the salt-
                # loaded inner sea, long in clean air (gamma: see the config note; 2/3 = fixed-LWC,
                # Twomey-consistent). frac becomes (nx,1) -- conservation-safe because the exchange is a
                # per-CELL q<->q_c conjugate pair (the SAME per-column frac scales cond and evap_demand,
                # so each cell's exchange and latent heat stay exactly conjugate; only conv_tau's
                # column-integral blend needs a scalar frac). At N_d == N_ref the factor is exactly 1.0
                # (bit-identical to the constant-tau scheme -- the #237 clean zero-point).
                # the floor makes the N_d > 0 invariant explicit rather than load-bearing-by-construction
                # (every current source builds a strictly-positive field, so the maximum is a bitwise
                # no-op; it guards a future near-zero ultra-clean transport field -- PR #283 review nit).
                nd = xp.maximum(th._F(self.N_d, xp), 1e-30)                           # (nx,) [cm^-3]
                _ctau = c.cond_tau * (c.salt_ccn_nd_ref / nd) ** c.cond_tau_salt_exp  # (nx,) [s]
                _cfrac = xp.minimum(dt / _ctau[:, None], 1.0)   # (nx,1), broadcasts over z
            else:
                _cfrac = xp.minimum(dt / c.cond_tau, 1.0)       # 0-d device scalar (graph-clean)
            cond = cond * _cfrac
            evap_demand = evap_demand * _cfrac
        qc_avail = self.q_c + cond                              # condensate available to the sinks this step
        # (4) precip sinks: rate autoconversion + the in-cloud retained-water ceiling (one number with the optics)
        precip_demand = self._autoconversion_demand(qc_avail, ice, dt)
        if c.equator_norain_cells > 0 and c.equator_norain_keep_ceiling:
            # CLEAN suppression test: cut only the AUTOCONVERSION rain at the equator; the retained-water
            # ceiling below still force-rains the excess, so q_c stays physically bounded (no ~1 m pile-up).
            precip_demand[:c.equator_norain_cells] = 0.0
        if c.qc_retained_ceiling:
            f = self._cloud_cover(q)
            rho_air = self.p_mid[None, :] / (C.R_DRY * self.T_atm)        # [kg/m^3], same as the optics
            # the retained ceiling shares the radiative LWC bound (ONE number): the Chunk-3 decoupled physical-
            # plausibility LWC when qc_radiative_cap_decouple is on, else the Chunk-1 adiabatic-rainout cap.
            if c.qc_radiative_cap_decouple:
                lwc_ceiling = xp.where(ice, rad.CLOUD_IWC_PLAUSIBLE, rad.CLOUD_LWC_PLAUSIBLE)
            else:
                lwc_ceiling = xp.where(ice, rad.CLOUD_IWC_MAX, rad.CLOUD_LWC_MAX)
            qc_ceiling = lwc_ceiling * f / xp.maximum(rho_air, 1e-9)      # grid-mean retained ceiling [kg/kg]
            if c.equator_norain_cells > 0 and c.equator_ceiling_lwc_scale != 1.0:
                # DIAGNOSTIC: scale the rainout CAP in the suppressed equatorial cells only. The cap (CLOUD_LWC_MAX
                # ~0.3 g/m^3) is EARTH-observed cloud microphysics, and on the ~90 C equator the adiabatic in-cloud
                # LWC is ~170 g/m^3, so the cap binds ~560x -- letting the column hold almost none. Sweeping the cap
                # up toward the adiabatic value (no-rainout limit) measures how much the band redistribution depends
                # on that Earth assumption (vs the column's structural inability to loft). See _equator_ceiling_sweep.py.
                qc_ceiling[:c.equator_norain_cells] *= c.equator_ceiling_lwc_scale   # cross-track (nx) is axis 0 (NOT [...,:N] = the vertical)
            # only CAP where there is real cloud cover (f >= f_min). Where the cover collapses below the floor
            # there is no in-cloud water to bound (qc_ceiling -> 0), and the ceiling must NOT degenerate into
            # "rain everything" -- such trace condensate is left to re-evaporation/advection (PR #143 review #3).
            ceil_excess = xp.where(f >= c.cloud_f_min, xp.maximum(qc_avail - qc_ceiling, 0.0), 0.0)
            precip_demand = precip_demand + ceil_excess
        if c.equator_norain_cells > 0 and not c.equator_norain_keep_ceiling:
            # DIAGNOSTIC suppression test (no-ceiling): no rain-out at all in the innermost N cells -> the condensate
            # is retained (q_c, water-conserving) and the ascending equatorial branch advects it poleward. Lets q_c
            # pile unbounded (the ~1 m load); use equator_norain_keep_ceiling for the physically-bounded clean test.
            precip_demand[:c.equator_norain_cells] = 0.0
        if boil is not None:
            # #294: NO precip forms in a boiling layer (liquid cannot persist there to fall) -- kills both
            # the autoconversion and the retained-ceiling force-rain contributions; the condensate instead
            # drains through the re-evaporation demand above. Applied after every precip_demand contribution.
            precip_demand = xp.where(boil, 0.0, precip_demand)
        # (5) coupled-sink positivity limiter: re-evaporation first (toward saturation), precip takes the rest,
        #     each capped so the TOTAL removal never exceeds qc_avail (q_c >= 0 exactly).
        evap = xp.minimum(evap_demand, qc_avail)
        precip_qc = xp.minimum(precip_demand, qc_avail - evap)
        # capture-ready: store these INTERMEDIATE q_c / T_atm updates in place so the buffer ADDRESS stays
        # fixed through to the final _cast in _thermo_substep_impl (a rebind here would make that _cast write
        # into a fresh array). OFF = the verbatim rebind. q_c is a fresh sum -> copy-in; T_atm is a plain add.
        if c.capture_ready:
            self.q_c[...] = qc_avail - evap - precip_qc        # >= 0 to the bit (copy into the fixed buffer)
            q = q - cond + evap                                # local q (a temporary) -- rebind is fine
            xp = get_xp(self.T_atm); xp.add(self.T_atm, L_eff * (cond - evap) / cp, out=self.T_atm)
        else:
            self.q_c = qc_avail - evap - precip_qc             # >= 0 to the bit
            q = q - cond + evap                                # condensation removes / re-evaporation returns
            self.T_atm = self.T_atm + L_eff * (cond - evap) / cp   # latent heating (cond) / cooling (evap)
        # The precip P. By default (precip_reevap OFF, byte-identical) it is the produced-aloft column
        # autoconversion total -- booked to the surface at the production latitude. ON, the FALLING rain is
        # re-evaporated below cloud (deficit-limited) and ``_precip`` becomes the SURFACE-reaching flux.
        if c.prognostic_rain:
            # PROGNOSTIC RAIN (stage 7, config note at ``prognostic_rain``): the autoconversion+ceiling
            # sink SOURCES the falling-rain reservoir q_r instead of the instant same-substep column
            # rainout -- BOTH phases join the one reservoir (the liquid autoconversion AND the ice
            # branch; ONE rain_v_term, the named draft wall). The within-step fall closures
            # (_below_cloud_reevap / the direct _precip booking below) are SUPERSEDED on this path:
            # sedimentation books the surface flux and _rain_reevap is the prognostic replacement of the
            # below-cloud re-evaporation (both in _prognostic_rain_step, called right after this method
            # by _thermo_substep_impl -- self._precip is set THERE this substep). precip_qc is a per-step
            # mixing-ratio increment [kg/kg], the same units/grid as q_r. Funnelled through _obuf like
            # every prognostic q_r store: under capture self.q_r is the fp32 wrapper's persistent scratch
            # and the store must keep that fixed address (the RHS temp materializes before the in-place
            # copy, so the aliased write is safe; eager path = the verbatim rebind, value-identical).
            self.q_r = self._cast(self.q_r + precip_qc, out=self._obuf(self.q_r))
        elif c.precip_reevap:
            # pass the PRODUCTION ice mask (step (1) above: pre-heating phase, already gated on qc_ice_phase) so the
            # re-evap source phasing is the EXACT conjugate of the production's latent release -- closes the
            # qc_ice_phase footgun (#148 review #2) and the post- vs pre-heating source-phase slip (#3).
            reevap_precip = self._below_cloud_reevap(q, precip_qc, dt, src_ice=ice)  # surface flux; updates q, T_atm
            if c.capture_ready:
                self._precip[...] = reevap_precip
            else:
                self._precip = reevap_precip
        elif c.capture_ready:
            self._precip[...] = xp.sum(precip_qc * self.dm[None, :], axis=1) / dt   # in-place (nx,) buffer
        else:
            self._precip = xp.sum(precip_qc * self.dm[None, :], axis=1) / dt   # autoconversion+ceiling precip [kg/m^2/s]
        return q

    def conv_gate_diagnostics(self):
        """The scale-aware convective-gate instrumentation (None when the gate is OFF): per-column cell
        width [m], the gate ``w_conv`` in [0, 1], and the accumulated pre-blend |enthalpy| the adjustment
        moved [J/m^2] -- "who does the lifting vs x" across the gray zone (compare against the resolved
        |w| profile to see the parameterized->resolved handover)."""
        if self._conv_gate_col is None:
            return None
        # NB device landmine (review, PR #321): np.asarray on a CuPy array RAISES -- fine today (the
        # gate is guarded OFF on device='gpu'), but a future device conv-gate port must to_host here.
        widths = self._dx_cell if self._dx_cell is not None else np.full(self.cfg.nx, self.dx)
        return {"width_m": np.asarray(widths, float).copy(),
                "w_conv": np.asarray(self._conv_gate_col[:, 0], dtype=float).copy(),
                "conv_heat_Jm2": self._conv_gate_heat.copy()}

    def _apply_chimney_param(self):
        """B5c (cfg.chimney_param, default OFF -> no-op / byte-identical): replace the divergent equatorial
        1/dx precip SPIKE in ``self._precip`` with ``equator_chimney_param``'s resolution-INDEPENDENT
        grid-mean ``P_param`` -- the finite (1-phi)*M_tot redistributed over a fixed footprint, with the
        ribbon band (|x| >= x_detect) passed through unchanged. So the surface hydrology, the cascade aloft-
        supply target, and ``budget_wet_fraction`` then see a finite, dx-convergent equatorial supply instead
        of the grid artifact (P_peak ~ 1/dx).

        SURFACE-PRECIP swap only (the B5c design / the B5a pattern): the condensation latent heat is REAL (it
        happened; kept in T_atm) and the dynamics are untouched -- only the SURFACE-rain fate is
        re-parameterized. The phi-exported fraction (~phi=2% of the zone's M_tot, calibrated
        CHIMNEY_PHI_CALIBRATION_RESULTS.md) was already condensed out of ``q`` aloft, so dropping it from the
        surface book leaves only a negligible ~phi equatorial-aloft q-budget gap; the H_param latent
        redistribution is deferred. Applied each thermo substep right after ``self._precip`` is set (both the
        prognostic and single-pass paths) and before the water-budget accumulation, so every consumer sees
        the de-spiked profile. Units: ``_precip`` [kg/m^2/s] <-> P_param [mm/day] via x86400."""
        c = self.cfg
        if not c.chimney_param:
            return
        from . import equator_chimney_param as ecp           # local import (no module-load-order coupling)
        if getattr(self, "_chimney_cache", None) is None:    # STATIC footprint -- built once, reused each substep
            self._chimney_cache = ecp.precompute_footprint(self.theta_deg)
            # HOIST the "is the zone resolved?" test to CACHE-BUILD time (capture-ready, Phase B(ii)): the
            # footprint is static, so mask.any() is a per-run CONSTANT. Caching it removes the per-substep
            # device->host .any() sync from the hot path. Byte-identical: the same constant decides the
            # early-return either way.
            self._chimney_resolved = bool(self._chimney_cache[0].any())
        mask, w, shp_z = self._chimney_cache
        resolved = self._chimney_resolved if c.capture_ready else bool(mask.any())
        if not resolved:                                     # zone unresolved at this nx -> nothing to de-spike
            return
        phi = c.chimney_param_phi if c.chimney_param_phi is not None else ecp.LOFT_FRACTION_PHI
        # DEVICE (Phase D): ``equator_chimney_param`` is host numpy (its cached footprint mask/weights are host)
        # and the de-spike is a tiny (nx,) operation done once per substep, so run it on a HOST copy of the
        # precip profile and write the masked cells back to the (possibly device) buffer. numpy in -> verbatim
        # (byte-identical); on GPU this is a bounded (nx,) host excursion (chimney is default-OFF, off the
        # production-stack path). The ``mask`` (host bool) indexes a host _precip on device via to_host/asarray.
        P_in_host = np.asarray(to_host(self._precip)) * 86400.0
        P_param_mm = ecp.apply_cached(P_in_host, mask, w, shp_z, phi)
        if get_xp(self._precip) is np:
            self._precip[mask] = P_param_mm[mask] / 86400.0  # CPU: verbatim in-place write (byte-identical)
        else:
            P_out = P_in_host / 86400.0
            P_out[mask] = P_param_mm[mask] / 86400.0
            self._precip[...] = to_device(P_out, self.xp)    # write back the de-spiked (nx,) profile

    def set_chimney_nest_response(self, resp):
        """Install the resolved chimney response (``chimney_couple.chimney_response`` output dict with keys
        ``w`` (nx,), ``H`` (nx, nz) [K/s], ``Mcond`` (nx,) [kg/m^2/s], ``qexp`` (nx, nz) [1/s], ``P`` (nx,)
        [kg/m^2/s]) as the held nest parameterization the thermo substep consumes (``_apply_chimney_nest``).
        Allocates the device buffers + scratch on first call (working dtype, on the state's device); thereafter
        writes IN PLACE (fixed addresses -> CUDA-graph-capture safe; the captured step reads the updated bits on
        replay). Call between graph replays from the host-excursion loop, every N core-days, and -- crucially --
        BEFORE ``capture_step`` on the GPU path (else the captured graph records the nest as inactive and later
        activation is a no-op). Requires ``cfg.chimney_nest`` (else it's inert)."""
        xp = self.xp
        w = np.asarray(resp["w"], np.float64)
        H = np.asarray(resp["H"], np.float64)
        Mc = np.asarray(resp["Mcond"], np.float64)
        qexp = np.asarray(resp["qexp"], np.float64)
        P = np.asarray(resp["P"], np.float64)
        nx, nz = self.cfg.nx, self.cfg.nz
        if (w.shape != (nx,) or H.shape != (nx, nz) or Mc.shape != (nx,)
                or qexp.shape != (nx, nz) or P.shape != (nx,)):
            raise ValueError(f"nest response shapes w{w.shape}/H{H.shape}/Mcond{Mc.shape}/qexp{qexp.shape}/"
                             f"P{P.shape} != (nx={nx}, nz={nz})")
        dt = self._dtype
        if getattr(self, "_nest_w", None) is None:
            self._nest_w = to_device(w.reshape(nx, 1).astype(dt), xp)        # (nx,1) held weight
            self._nest_w1d = to_device(w.astype(dt), xp)                     # (nx,)
            self._nest_H = to_device(H.astype(dt), xp)                       # (nx,nz) heating [K/s]
            self._nest_Mcond = to_device(Mc.reshape(nx, 1).astype(dt), xp)   # (nx,1) condensation [kg/m^2/s]
            self._nest_qexp = to_device(qexp.astype(dt), xp)                 # (nx,nz) aloft export [1/s]
            self._nest_P = to_device(P.astype(dt), xp)                       # (nx,) surface precip [kg/m^2/s]
            self._nest_qcol = xp.empty((nx, 1), dtype=dt)                    # scratch: column vapour (nx,1)
            self._nest_f = xp.empty((nx, 1), dtype=dt)                       # scratch: sink fraction (nx,1)
            self._nest_r = xp.empty((nx, 1), dtype=dt)                       # scratch: cap-conservation ratio (nx,1)
            self._nest_r1d = xp.empty((nx,), dtype=dt)                       # scratch: r as (nx,) for precip
            self._nest_scratchP = xp.empty((nx,), dtype=dt)                  # scratch: precip (nx,)
        else:
            self._nest_w[...] = to_device(w.reshape(nx, 1).astype(dt), xp)
            self._nest_w1d[...] = to_device(w.astype(dt), xp)
            self._nest_H[...] = to_device(H.astype(dt), xp)
            self._nest_Mcond[...] = to_device(Mc.reshape(nx, 1).astype(dt), xp)
            self._nest_qexp[...] = to_device(qexp.astype(dt), xp)
            self._nest_P[...] = to_device(P.astype(dt), xp)
        self._nest_active = True

    def _apply_chimney_nest(self, dt):
        """Two-way nest (``cfg.chimney_nest``, default OFF -> no-op / byte-identical): REPLACE the inner-zone
        convective result with the resolved chimney response (installed by ``set_chimney_nest_response``). Over
        the inner-zone weight ``w``, with ``(T_pre, q_pre)`` snapshotted before the convective block:
            T_atm <- T_pre + H*dt                         (the resolved convective HEATING profile)
            q     <- q_pre*(1 - f) + qexp*dt              (vapour SINK as a fraction f of the available column
                                                           vapour, never negative; + the aloft export source)
            precip <- nest_P                              (the resolved surface supply)
        where f = min(Mcond*dt / INT q_pre dm, 1) is computed HERE from the live column vapour (so condensation
        can never over-drain a layer -- the heating peaks aloft where q is small). Energy and water close by
        COLUMN integral: cp*INT H dm == L_VAP*(Mcond - Mexport) == L_VAP*P, and the column vapour change ==
        -(Mcond - Mexport) == -P. Capture-safe: in-place elementwise + out=/keepdims reductions on persistent
        buffers (NO boolean scatter, NO temp allocation); byte-identical when inactive / w==0."""
        c = self.cfg
        if not c.chimney_nest or not getattr(self, "_nest_active", False):
            return
        xp = self.xp
        w = self._nest_w                              # (nx, 1) held weight
        s = self._nest_scratch                        # (nx, nz) scratch A (allocated at the snapshot)
        s2 = self._nest_scratch2                      # (nx, nz) scratch B
        # --- column vapour INT q_pre dm (nx,1), the sink fraction f, and the CAP-CONSERVATION ratio r ---
        # When the resolved condensation Mcond*dt EXCEEDS the core column's available vapour (a dry inner
        # column vs the chimney's moist-inflow-fed condensation), only qcol can actually condense: the sink
        # caps at f=1. To CONSERVE water AND energy the deposited precip, the aloft export qexp, and the latent
        # heating H must ALL scale by the same achieved fraction r = min(1, qcol/(Mcond*dt)) -- else the nest
        # deposits phantom surface rain (r<1 case: precip>vapour removed -> water created) and over-heats /
        # over-exports. r == 1 when uncapped -> BYTE-IDENTICAL to the pre-fix path. (Root fix -- give the nest
        # the convergent moisture SOURCE so the column can supply Mcond -- is the follow-up; this guard makes
        # the coupling water/energy-closed unconditionally in the meantime.)
        s[...] = self._nest_qpre; s *= self.dm[None, :]          # q_pre * dm (reuse scratch A)
        xp.sum(s, axis=1, keepdims=True, out=self._nest_qcol)    # column vapour (nx,1)
        self._nest_qcol += 1e-30
        self._nest_f[...] = self._nest_Mcond; self._nest_f *= dt; self._nest_f += 1e-30   # Mcond*dt (eps-guarded)
        self._nest_r[...] = self._nest_qcol; self._nest_r /= self._nest_f   # r = qcol/(Mcond*dt); Mcond=0 -> r=inf->1
        xp.minimum(self._nest_r, 1.0, out=self._nest_r)                # r = min(1, qcol/(Mcond*dt))  (no scaling if Mcond=0)
        self._nest_f /= self._nest_qcol                               # f = Mcond*dt/qcol
        xp.minimum(self._nest_f, 1.0, out=self._nest_f)               # f = min(., 1)
        self._nest_f *= -1.0; self._nest_f += 1.0                     # (1 - f)
        self._nest_r1d[...] = self._nest_r[:, 0]                      # r as (nx,) for the precip update
        # --- T_atm += w * ((T_pre + r*H*dt) - T_atm)   (heating scaled by the achieved condensation fraction) ---
        s[...] = self._nest_H; s *= dt; s *= self._nest_r      # r*H*dt (r (nx,1) broadcasts over (nx,nz))
        s2[...] = self._nest_Tpre; s2 += s; s2 -= self.T_atm; s2 *= w
        self.T_atm += s2
        # --- q += w * ((q_pre*(1-f) + r*qexp*dt) - q) ---
        s[...] = self._nest_qexp; s *= dt; s *= self._nest_r   # r*qexp*dt
        s2[...] = self._nest_qpre; s2 *= self._nest_f; s2 += s  # q_pre*(1-f) + r*qexp*dt = target_q
        s2 -= self.q; s2 *= w
        self.q += s2
        # --- surface precip += w * (r*nest_P - precip) ---
        sp = self._nest_scratchP                      # (nx,)
        sp[...] = self._nest_P; sp *= self._nest_r1d; sp -= self._precip; sp *= self._nest_w1d
        self._precip += sp

    def _below_cloud_reevap(self, q, precip_qc, dt, src_ice=None):
        """Below-cloud precipitation re-evaporation (``precip_reevap``, master env COSMO_PRECIP_REEVAP):
        integrate the downward precip flux through the column and re-evaporate (rain) / sublimate (snow,
        T < T_FREEZE) a portion in each SUB-SATURATED layer below where it formed, returning the
        SURFACE-reaching flux [kg/m^2/s] and feeding the evaporated vapour back into ``q`` (with the matching
        latent cooling on ``self.T_atm``, IN PLACE on both). Without it, ``_precip`` is the column
        autoconversion total -- rain credited to the surface at the PRODUCTION latitude even where an elevated
        cloud base sits over dry sub-cloud air, so the ribbon's surface hydrology is biased WET and moisture is
        under-recycled equatorward (CLOSED_ISSUES "Sub-cloud VIRGA", PR #148; ``_virga_diag`` is the post-hoc
        estimate this makes prognostic).

        The falling precip descends through the column WITHIN the step (no suspended-rain reservoir -- the
        re-evaporated VAPOUR is what persists, carried away by ``step()``'s ``_advect_moisture``; an explicit
        prognostic sedimentation variable is the named deeper refinement). SNOW and RAIN are tracked as two
        downward mass fluxes Fs, Fr [kg/m^2/s] (phased by the layer each parcel FORMS in: snow below T_FREEZE,
        rain above), so the latent bookkeeping stays the EXACT CONJUGATE of the production across the freezing
        level. Per layer, in order: (1) the autoconversion source joins Fs/Fr; (2) MELT -- snow entering a
        warm layer (T>=T_FREEZE) melts to rain, absorbing L_FUS (cooling the layer); (3) RE-EVAPORATION of a
        portion, the flux- and deficit-limited total

            E_k = min( Fs_k + Fr_k ,  K_e * max(qsat_k - q_k, 0) * dm_k ,  max(qsat_k - q_k, 0)/denom_k * dm_k/dt )

        split between snow (sublimation, L_SUB) and rain (evaporation, L_VAP) by their falling fractions. The
        MIDDLE term is the physical rate: the evaporation is PROPORTIONAL to the ABSOLUTE saturation deficit
        max(qsat_k - q_k, 0) [kg/kg] at the rate K_e = ``precip_reevap_rate`` (absolute deficit, NOT (1-RH)).
        Three properties follow: (i) it SELF-LIMITS near saturation (deficit -> 0 => no evaporation => rain
        lands); (ii) a COLD column has a tiny qsat hence a tiny ABSOLUTE deficit => its falling SNOW lands even
        at low RH (the ``_virga_diag`` DEFICIT_REF correctness condition the (1-RH) form gets wrong); (iii) the
        per-layer fractional loss E_k/F ~ K_e(qsat-q)dm_k/F falls with F => heavy rain reaches the ground while
        drizzle virgas. NB this last is a FLUX-LIMIT effect (E<=F), NOT a rain-content-dependent rate: the bulk
        rate carries no ventilation / (rho*q_r)^0.525 term, so it is a Kessler/Sundqvist-class absolute-deficit
        DRIVER, not the full drop-microphysical rate (a defensible bulk simplification, reported as such). The
        FIRST term is the flux limit (never evaporate more than is falling); the THIRD is the
        anti-oversaturation safety cap (q never exceeds qsat in a step), with the dqsat/dT ``denom`` matching
        ``_prognostic_microphysics`` step 2. In BOILING layers (esat >= 0.999 p) the ``virga_flash_boil``
        flag FLOORS the ceiling at the flash-boil ENERGY cap (full-flux evaporation down to the layer's
        boiling point), keeping the deficit-driven rate as the other arm of the max() so ON >= OFF
        everywhere and a rain-cooled layer exits the boiling mask instead of pinning at T_boil (see the
        inline note and the config entry -- #292/#294 + the PR #296 review fix).
        GRID-CONVERGENT: rate-limited E_k ∝ dm_k, so the column loss is
        K_e∫(qsat-q)dm, a mass integral (the re-evap COOLING is intensive, also nz-independent).
        CAVEAT (#148 review #1): the MELT term is the exception -- ALL snow flux melts in the single first
        warm layer, so its cooling tendency ∝ 1/dm_k SHARPENS at the freezing level as nz rises (the
        column-INTEGRATED L_FUS is conserved; only its vertical CONCENTRATION is grid-dependent). Active in
        snow (cold/poleward) columns where the freezing level sits inside the precipitating column; at the
        production nz=48 it is a fraction of a K (not a stability risk). A physical fix -- spreading the melt
        over a finite bright-band depth (~a few hundred m) -- is a deferred refinement, NOT dropping the melt
        term (that reintroduces the L_FUS leak this scheme closes).

        CONSERVATION (the load-bearing gate). Water and energy move only WITHIN the column: the round trip
        q -> q_c -> rain -> q nets to zero source/sink (the surface flux Fs+Fr is the only water that leaves as
        P), so ``water_conservation`` still closes d/dt INTEGRAL(q+q_c)dm = E - P with the new SURFACE P. The
        latent removal is the EXACT conjugate of the production warming, INCLUDING across the freezing level:
        snow formed cold released L_SUB aloft, and on the way down loses exactly L_SUB -- either by subliming
        cold (L_SUB) or by melting (L_FUS) then evaporating warm (L_VAP), L_FUS+L_VAP=L_SUB. (The single-flux
        scheme dropped that L_FUS, a one-signed spurious heating where snow virgas into warm sub-cloud -- PR
        #148 review #1.) No top-of-atmosphere term -> ``energy_conservation`` is unaffected. Adds NO prognostic
        state (a per-step diagnostic of q / T_atm / precip_qc), so checkpoints are unchanged."""
        c = self.cfg
        cp = C.CP_AIR
        xp = get_xp(self.T_atm)                                 # device-generic (Phase D); numpy on CPU
        dm = self.dm                                            # (nz,) layer mass [kg/m^2]
        p = self.p_mid[None, :]                                 # (1, nz)
        T = self.T_atm                                          # (nx, nz) post-condensation column temperature
        qs = th.qsat(T, p)                                      # (nx, nz)
        warm = T >= C.T_FREEZE                                  # (nx, nz) layer at/above freezing
        L_layer = xp.where(warm, C.L_VAP, C.L_SUB)             # the dqsat-cap phase (L_VAP warm / L_SUB ice layers)
        if c.qc_dqsat_adjust:
            denom = 1.0 + (L_layer / cp) * th.dqsat_dT(T, p)    # Newton anti-overshoot (matches the cond step)
        else:
            denom = xp.ones_like(T)
        deficit = xp.maximum(qs - q, 0.0)                      # (nx, nz) ABSOLUTE saturation deficit [kg/kg]
        rate = c.precip_reevap_rate * deficit * dm[None, :]    # (nx, nz) deficit-proportional rate [kg/m^2/s]
        sat_cap = deficit / denom * dm[None, :] / dt           # (nx, nz) anti-oversaturation cap [kg/m^2/s]
        evap_cap = xp.minimum(rate, sat_cap)                   # (nx, nz) deficit-limited ceiling (both phases share it)
        if c.virga_flash_boil:
            # A layer above boiling (esat >= 0.999 p: qsat-clamped / supercritical) cannot hold falling liquid
            # AT ALL -- the constraint is "liquid can't persist", NOT "room up to saturation" (#294: the old
            # sat_cap bound was ~0 in a steam-saturated boiling layer, deficit ~ 0, so heavy rain passed
            # through to a >100 C surface). FLOOR the cap at the flash-boil ENERGY cap: the flash evaporates
            # the full flux until its cooling lands the layer at its own boiling point T_boil(p)
            # (esat(T_boil) = 0.999 p, thermo.T_boil) -- boil_cap = (T - T_boil) cp dm / (L_VAP dt) is the
            # flux whose latent cooling does exactly that. [K * (J/kg/K) * (kg/m^2) / ((J/kg) * s) =
            # kg/m^2/s.] The flash itself is therefore bounded in BOTH the added vapour (dq <=
            # cp (T-T_boil)/L_VAP per step) and its cooling -- the q/energy blow-up guard. It is a
            # maximum() FLOOR, not a replacement (PR #296 review): the ordinary deficit-driven rate keeps
            # acting AT and near T_boil (rain wet-bulb-cools a boiling layer BELOW boiling and out of the
            # mask, exactly the flag-OFF closure), so ON evaporation >= OFF everywhere, the closure is
            # continuous at the mask edge, and a rain-cooled layer can never PIN at T_boil as a permanent
            # pass-through (the replace form converged to ~99% surface pass-through of a theta0-scale flux
            # in an iterated moist-stack A/B -- the exact #294 failure). In the motivating steam-saturated
            # state (deficit ~ 0 -> rate ~ 0) the floor reduces to the pure energy cap. q may exceed the
            # qsat clamp afterwards (physical: steam above boiling); pair with cond_boil_suppress so the
            # excess is exported as vapour, not re-condensed. L_VAP is the right latent heat: a boiling
            # layer is warm, so any falling snow has already melted (the melt term above the E line) and
            # the flashed flux is all rain. Default OFF above.
            boil = th.esat(T) >= 0.999 * p                     # (nx, nz) boiling / supercritical layers
            boil_cap = xp.maximum(T - th.T_boil(p), 0.0) * cp * dm[None, :] / (C.L_VAP * dt)
            evap_cap = xp.where(boil, xp.maximum(boil_cap, evap_cap), evap_cap)
        # phase the production source by the layer it FORMS in (snow below freezing, rain above), and track the
        # two fluxes separately + a MELT term so the re-evaporation latent heat is the exact CONJUGATE of the
        # production across the freezing level (the L_FUS the single-flux scheme dropped; PR #148 review #1).
        src = precip_qc * dm[None, :] / dt                     # (nx, nz) per-layer precip SOURCE flux [kg/m^2/s]
        # Phase the source by the PRODUCTION ice mask (pre-heating, qc_ice_phase-aware) -- NOT the kernel's
        # post-heating ``warm`` -- so snow flux corresponds 1:1 to where production released L_SUB. With
        # qc_ice_phase OFF src_ice is all-False => all rain (production released L_VAP everywhere), so the
        # melt/L_SUB path is inert and the conjugacy holds (no spurious one-signed L_FUS). (#148 review #2/#3.)
        # ``_prognostic_microphysics`` passes its own ``ice``; a direct caller (tests) may omit it -> fall back
        # to the same definition off the current column (qc_ice_phase-gated), reproducing the prior behaviour.
        if src_ice is None:
            src_ice = (T < C.T_FREEZE) if c.qc_ice_phase else xp.zeros(T.shape, dtype=bool)
        src_snow = xp.where(src_ice, src, 0.0)                 # snow forms where production made ice (released L_SUB)
        src_rain = src - src_snow                              # rain otherwise (released L_VAP)
        Fs = xp.zeros(q.shape[0]); Fr = xp.zeros(q.shape[0])   # (nx,) downward SNOW / RAIN mass flux [kg/m^2/s]
        dq = xp.zeros_like(q)                                  # re-evaporated vapour per layer [kg/kg]
        dT = xp.zeros_like(q)                                  # latent (melt + re-evap) cooling per layer [K]
        for k in range(q.shape[1]):                            # top (0) -> surface (nz-1): precip falls downward
            Fs = Fs + src_snow[:, k]
            Fr = Fr + src_rain[:, k]
            melt = xp.where(warm[:, k], Fs, 0.0)              # snow melts to rain in a warm layer (absorbs L_FUS)
            Fs = Fs - melt; Fr = Fr + melt
            dT[:, k] = dT[:, k] - C.L_FUS * (melt * dt / dm[k]) / cp
            Ftot = Fs + Fr
            E = xp.minimum(Ftot, evap_cap[:, k])              # total re-evaporation, flux- AND deficit-limited
            f = E / xp.maximum(Ftot, 1e-30)                   # split by the falling snow/rain fractions
            Es = f * Fs; Er = f * Fr
            Fs = Fs - Es; Fr = Fr - Er
            dq[:, k] = (Es + Er) * dt / dm[k]                 # vapour returned to layer k
            dT[:, k] = dT[:, k] - (C.L_SUB * Es + C.L_VAP * Er) * dt / (dm[k] * cp)   # phase-conjugate cooling
        q += dq                                                # re-evaporated vapour rejoins q (advected by step())
        if self.cfg.capture_ready:                             # plain add in-place (fixed buffer address)
            xp = get_xp(self.T_atm); xp.add(self.T_atm, dT, out=self.T_atm)
        else:
            self.T_atm = self.T_atm + dT                       # melt + phase-conjugate re-evaporation cooling
        return Fs + Fr                                         # surface-reaching precip (snow + rain) [kg/m^2/s]

    # ------------------------------------------------------------------ PROGNOSTIC RAIN q_r (stage 7)
    def _sediment_rain(self, dt):
        """Vertical SEDIMENTATION of the prognostic rain reservoir ``self.q_r`` at the terminal fall speed
        ``rain_v_term``, per-column implicit backward-Euler on the PRESSURE grid (last axis TOP -> SURFACE,
        so "down" is INCREASING k) -- the core-conventions port of chimney.py's ``_vsediment_implicit``
        (which is (nz, nx) height-grid, "down" = decreasing j). Implicit because v_term*dt/dz >> 1 at the
        production spacings (7 m/s * 60 s = 420 m >> the ~20 m BL-refined dz -- explicit is impossible);
        backward-Euler upwind is an M-matrix -> UNCONDITIONALLY STABLE + MONOTONE (q_r >= 0 preserved,
        no clipping). Updates ``self.q_r`` in place (rebind) and returns the SURFACE-face flux
        (nx,) [kg/m^2/s] -- the rain reaching the ground, which the caller books as ``self._precip``.

        DISCRETE FORM (per column; q_k = q_r[:, k], unknowns at n+1). Rain falls only DOWNWARD, so the
        donor cell of every face flux is the layer ABOVE it and the system is lower-BIDIAGONAL (a tridiag
        with zero superdiagonal -- solved by the existing ``_solve_tridiag_columns`` Thomas sweep):

            F_k = v_term * rho_k * q_k^{n+1}            downward flux through layer k's BOTTOM face
                                                        [m/s * kg/m^3 * kg/kg = kg/m^2/s]
            dm_k (q_k^{n+1} - q_k^n) = dt (F_{k-1} - F_k)      (F_{-1} = 0: nothing falls in from space)
        =>  (dm_k + dt a_k) q_k^{n+1} - dt a_{k-1} q_{k-1}^{n+1} = dm_k q_k^n,   a_k = v_term * rho_k

        with rho_k = p_mid/(R_DRY * T_atm) the live layer density [kg/m^3] (the same expression the
        retained-ceiling optics use). CONSERVATION BY CONSTRUCTION: summing the update over k telescopes
        the interior fluxes, so  d(INTEGRAL q_r dm) = -dt * F_{nz-1}  EXACTLY (the only mass leaving is
        the surface flux, whatever rho is) -- the booked surface rain equals the column loss to round-off.
        The mass-coordinate flux rho*v*q is the standard omega_sed = rho*g*v_term [Pa/s] sedimentation.

        DEVICE (eager GPU): ``_solve_tridiag_columns`` is host numpy, so the bands take a bounded
        (nz, nx) host round-trip (eager-only -- capture_ready is validated OFF with prognostic_rain;
        the device Thomas RawKernel wiring is the named GPU follow-up)."""
        c = self.cfg
        xp = get_xp(self.q_r)
        # a = v_term * rho on the pressure grid [kg/m^2/s per unit mixing ratio]; live density (fp32-safe:
        # th._F casts the fp64 p_mid under the thermo-fp32 gate, mirroring the LE-cap idiom).
        rho = th._F(self.p_mid[None, :], xp) / (C.R_DRY * self.T_atm)     # (nx, nz) [kg/m^3]
        a = c.rain_v_term * rho                                           # (nx, nz)
        dm = th._F(self.dm[None, :], xp)                                  # (1, nz) layer mass [kg/m^2]
        # bands for _solve_tridiag_columns: (nz, nx) with sub[k] multiplying x[k-1] (the layer ABOVE on the
        # pressure grid -- orientation: k increases DOWNWARD, so the rain couples each layer to the one above).
        diag = (dm + dt * a).T                                             # (nz, nx)
        sub = xp.zeros_like(diag)
        sub[1:, :] = -dt * a.T[:-1, :]                                     # -dt*a_{k-1} (sub[0] unused)
        sup = xp.zeros_like(diag)                                          # no UPWARD coupling: rain never rises
        rhs = (dm * self.q_r).T                                            # (nz, nx) mass form dm_k q_k^n
        if xp is np:
            qr_new = _solve_tridiag_columns(sub, diag, sup, rhs).T         # (nx, nz)
        elif self._tridiag_kernel_dev is not None:
            # DEVICE Thomas RawKernel (fp64, capture-safe -- built at init, warmup precedes capture):
            # bands cast UP to fp64 (the implicit-solve discipline; under the thermo-fp32 gate the state
            # views are fp32), the solve stays on device -- no host round-trip inside the captured step.
            from ._gpu_kernels import solve_tridiag_columns_device
            f64 = xp.float64
            qr_new = solve_tridiag_columns_device(self._tridiag_kernel_dev, sub.astype(f64),
                                                  diag.astype(f64), sup.astype(f64),
                                                  rhs.astype(f64)).astype(diag.dtype).T
        else:                                                              # eager gpu+splu: bounded host excursion
            qr_new = to_device(_solve_tridiag_columns(to_host(sub), to_host(diag), to_host(sup),
                                                      to_host(rhs)), xp).T
        qr_new = xp.maximum(qr_new, 0.0)          # M-matrix keeps >= 0; guard the solver's round-off only
        self.q_r = self._cast(qr_new, out=self._obuf(self.q_r))            # working dtype; fixed addr under capture
        return a[:, -1] * self.q_r[:, -1]         # surface flux F_{nz-1} = v_term*rho_surf*q_r_surf [kg/m^2/s]

    def _rain_reevap(self, q, dt):
        """Below-cloud RE-EVAPORATION of the falling rain ``self.q_r`` in SUB-SATURATED air, with the
        matching latent COOLING on the real prognostic ``self.T_atm`` -- the cold-pool mechanism, and the
        prognostic replacement of ``_below_cloud_reevap``'s within-step fall (chimney.py's
        ``_reevap_condensate`` ported to the core's (nx, nz) pressure grid, drawing from q_r only --
        suspended q_c keeps its own re-evaporation inside ``_prognostic_microphysics``). Mutates ``q``
        IN PLACE (the caller's working vapour array, a fresh temporary) and returns it.

        Closure (per layer, dt-consistent):  evap = min(deficit, q_r) * (1 - exp(-dt/rain_reevap_tau)),
        deficit = max(qsat - q, 0) / denom with the dqsat/dT Newton denominator denom = 1 + (L/cp) dqsat_dT
        (matches ``_prognostic_microphysics`` step 2 -- the added vapour's latent cooling lowers qsat, so
        the bare deficit would overshoot; the Newton step lands AT saturation to first order and never
        over-saturates). Deficit-limiting IS the below-cloud gate: inside cloud the air is ~saturated ->
        deficit ~ 0 -> no re-evaporation (no explicit cloud mask needed). The latent heat is phased by the
        LOCAL layer (L_SUB below T_FREEZE -- sublimation; L_VAP above -- evaporation) when ``qc_ice_phase``,
        the same per-layer phasing ``_below_cloud_reevap`` applies to its falling fluxes. NAMED WALL: q_r
        is a single PHASE-LESS reservoir (one v_term), so the melt L_FUS bookkeeping across the freezing
        level (``_below_cloud_reevap``'s exact conjugacy) is dropped -- ice-formed rain that re-evaporates
        in a warm layer mis-books its L_FUS share (bounded by L_FUS/L_SUB ~ 12% of the ice fraction's
        latent exchange).

        BOILING layers (#294/#299, both flags default OFF -> the plain closure): ``cond_boil_suppress``
        makes the demand the FULL available q_r there (no vapour-liquid equilibrium -- liquid cannot
        persist, it drains at the tau rate); ``virga_flash_boil`` FLOORS the per-step evaporation at the
        flash-boil ENERGY cap dq = (T - T_boil) cp / L_VAP [kg/kg] (the amount whose latent cooling lands
        the layer exactly AT its own boiling point) -- a max() FLOOR over the deficit closure, never a
        replacement (ON >= OFF everywhere; continuity at the mask edge -- the PR #296 review lesson),
        availability-limited. The exact port of chimney.py's ``_reevap_condensate`` legs.

        WATER/ENERGY: vapour gained == q_r lost per cell (exact conjugate); cp*dT == -L_eff*evap per cell,
        so the column integrals close to round-off (the test gate)."""
        c = self.cfg
        cp = C.CP_AIR
        xp = get_xp(self.T_atm)
        T = self.T_atm
        p = self.p_mid[None, :]
        qs = th.qsat(T, p)
        if c.qc_ice_phase:
            L_eff = xp.where(T < C.T_FREEZE, C.L_SUB, C.L_VAP)   # local-layer phase (sublimation vs evap)
        else:
            L_eff = C.L_VAP
        if c.qc_dqsat_adjust:
            denom = 1.0 + (L_eff / cp) * th.dqsat_dT(T, p)       # Newton anti-overshoot (matches cond step)
        else:
            denom = 1.0
        deficit = xp.maximum(qs - q, 0.0) / denom                # vapour addable before re-saturating [kg/kg]
        avail = self.q_r
        rate = 1.0 - xp.exp(-dt / c.rain_reevap_tau)             # per-step relaxation fraction (dt-consistent)
        if c.cond_boil_suppress or c.virga_flash_boil:
            boil = th.esat(T) >= 0.999 * p                       # boiling/supercritical layers (#294 mask)
        if c.cond_boil_suppress:
            # above boiling the demand is the full falling rain, not the deficit toward the clamped qsat
            # (denom == 1 on the mask -- the clamped dqsat_dT == 0 -- so no Newton damping is lost).
            deficit = xp.where(boil, avail, deficit)
        evap = xp.minimum(deficit, avail) * rate
        if c.virga_flash_boil:
            # the energy-cap floor: min(cap, avail) keeps availability; max(., evap) keeps ON >= OFF.
            boil_dq = xp.maximum(T - th.T_boil(p), 0.0) * (cp / C.L_VAP)   # [kg/kg]
            evap = xp.where(boil, xp.maximum(xp.minimum(boil_dq, avail), evap), evap)
        self.q_r = self._cast(self.q_r - evap, out=self._obuf(self.q_r))   # falling rain loses the evaporated share
        q += evap                                                # ...which rejoins the vapour IN PLACE
        self.T_atm = self.T_atm - L_eff * evap / cp              # exact-conjugate latent COOLING (cold pools)
        return q

    def _prognostic_rain_step(self, q, dt):
        """One thermo-substep advance of the prognostic rain reservoir (``prognostic_rain``; called by
        ``_thermo_substep_impl`` right after the microphysics SOURCED q_r): (1) SEDIMENT q_r at
        -rain_v_term (``_sediment_rain``, implicit) and book the surface-face flux as ``self._precip`` --
        exactly where the instant path used to put the autoconversion total, so every downstream consumer
        (the cumulative water budget, the wetness/W_surf supply, the surface flash-boil) sees the
        SURFACE-REACHING flux unchanged in kind; (2) RE-EVAPORATE q_r in sub-saturated air
        (``_rain_reevap``), returning the vapour to ``q`` with the latent cooling on ``T_atm``. Order
        mirrors chimney.py's ``_fall_microphysics`` (autoconvert -> transport/sediment -> re-evaporate ->
        book the surface rain). The RESOLVED advection of q_r is NOT here -- it rides the shared
        divergence-free mass flux beside q_c once per dynamics step (``_advect_rain`` in ``step()``),
        the same cadence/scheme split q_c itself uses (thermo substeps own the column physics; the
        dynamics step owns the transport)."""
        surf_flux = self._sediment_rain(dt)
        if self.cfg.capture_ready:
            self._precip[...] = surf_flux              # in-place into the fixed (nx,) buffer (graph address)
        else:
            self._precip = surf_flux                   # eager: the verbatim rebind
        return self._rain_reevap(q, dt)

    # ------------------------------------------------------------------ BL / stratocumulus closure (Chunk 4)
    def _bl_stratocumulus(self, w, dt):
        """The Chunk-4 (CLOUD_PROGNOSTIC_ROADMAP) boundary-layer / stratocumulus closure: maintain
        radiatively-driven low cloud under a subsidence inversion -- the regime grid-mean RH + bulk
        condensation (Chunks 1-3) under-resolve, the band's own subsiding-branch low deck (CLOUDS_RESULTS.md
        §4a, σ≈0.7-0.9). Updates ``self.q_c``, ``self.q``, ``self.T_atm`` and ``self._bl_deck_cover`` (the
        radiation cover floor); needs the resolved vertical velocity ``w`` (height grid, nz, nx) -- inert
        without dynamics. The leaf physics is ``flatworld.bl_cloud`` (deck detection + the cloud-top-cooling
        source / entrainment sink); here we apply the rates with the matching latent heat under the SAME
        coupled-sink positivity discipline as ``_prognostic_microphysics``.

        SINGLE-RESERVOIR + ENERGY-CONSISTENT (roadmap rule 3): the source CONDENSES vapour (latent warming)
        and the entrainment sink EVAPORATES condensate back to vapour (latent cooling) at the same level, so
        water only moves between q and q_c (total water q+q_c conserved within the column; NO new precip
        sink), and the cloud-top radiative COOLING that scales the source is applied by the radiation itself
        (via q_c -> dtau_lw_cloud) -- the closure adds no radiative tendency, so the cooling is not
        double-counted. Bounded by construction (the source saturates as the deck emissivity -> 1; the sink
        is linear in q_c; a hard source cap backs it up) -> it cannot run away (the magnitude is closure-set,
        direction-firm / magnitude-soft, the entrainment efficiency reported as a sensitivity)."""
        c = self.cfg
        xp = get_xp(self.T_atm)                                            # device-generic (Phase D)
        z_p = getattr(self, "_z", None)
        if z_p is None:
            z_p = self._hydrostatic_z(self.T_atm)
        # Phase D: keep w on the dynamics array module -> _height_to_pressure routes it through get_xp, so
        # w_p stays device-resident (no to_host pull). numpy default byte-identical.
        w_p = self._height_to_pressure(w, z_p)                            # (nx, nz) resolved w on q's layers
        mask = blc.deck_mask(self.T_atm, w_p, self.p_mid[None, :], z_p,
                             sigma_lo=c.bl_sc_sigma_lo, sigma_hi=c.bl_sc_sigma_hi,
                             stability_min=c.bl_sc_stability_min, p_surf=C.P_SURF)
        S, D = blc.stratocumulus_rates(self.T_atm, self.q_c, mask, self.p_mid[None, :], self.dm[None, :],
                                       deck_cover=c.bl_sc_deck_cover, ctop_cool_eff=c.bl_sc_ctop_cool_eff,
                                       entrain_eff=c.bl_sc_entrain_eff, tau_entrain=c.bl_sc_tau_entrain,
                                       source_max=c.bl_sc_source_max, f_win=c.f_lw_window)
        L_eff = xp.where(self.T_atm < C.T_FREEZE, C.L_SUB, C.L_VAP)
        q = self._clip_q(self.q)                                           # #294-A physical ceiling + accounting
        cond = xp.minimum(S * dt, q)                                       # can't condense more vapour than present
        evap = xp.minimum(D * dt, self.q_c + cond)                        # entrainment drying, capped at available q_c
        self.q_c = self._cast(self.q_c + cond - evap, out=self._obuf(self.q_c))      # >= 0 (evap <= q_c + cond)
        self.q = self._cast(q - cond + evap, out=self._obuf(self.q))                 # total water q + q_c conserved
        self.T_atm = self._cast(self.T_atm + L_eff * (cond - evap) / C.CP_AIR,
                                out=self._obuf(self.T_atm))                          # latent warm(cond)/cool(evap)
        self._bl_deck_cover = xp.where(mask, c.bl_sc_deck_cover, 0.0)     # overcast cover floor for the radiation

    def _update_buoyancy(self):
        """Set the dynamics buoyancy ``self.b`` (nz, nx) from the live column state (#2): map ``T_atm``
        (and, with ``virtual_buoyancy``, the prognostic ``q``) to the height grid and form
        b = g (T_v - T_ref(z)) / T_ref(z). Only the HORIZONTAL gradient of b torques the flow
        (free-slip walls + ``-d b/dx``), so the horizontally-uniform part is inert; the absolute form
        keeps the buoyancy physical and lets ``N^2`` stay diagnosed.

        VAPOR BUOYANCY (issue #80, ``cfg.virtual_buoyancy``, default ON): moist air is less dense than
        dry air at the same (T, p), so the density temperature is the virtual temperature
        (``thermo.virtual_temperature``) -- with q reaching ~0.10 over the wet warm edge this is up to
        ~+20 K of buoyancy the T-only form misses, and it is moist-column-selective (zero over the dry
        salt flat), i.e. a real horizontal buoyancy-gradient term, not an inert offset. The reference
        state is DRY (q_ref = 0 -> T_v_ref = T_ref), so no change to the reference profile is needed,
        and the off-limit (``virtual_buoyancy=False``) is byte-identical to the pre-#80 T-only form.

        CONDENSATE LOADING (PHY-19 step 2, ``cfg.condensate_loading``, default OFF): the SUSPENDED
        condensed water the column carries has weight, so the density temperature is
        T_rho = T_v - T*q_cond with q_cond = max(q_c,0) + max(q_r,0) the cloud+rain mass fraction
        (``thermo.density_temperature``) -- EXACT for specific humidities, not a linearization. No free
        parameter. Expected size at the PHY-19 sun-line axis
        ~1-2 K against ~30 K of vapor-buoyancy contrast -- a correctness term, NOT the fix for the axis
        rain column. The branch is at the PYTHON level on purpose: the step is captured into a CUDA
        graph, and a zero multiplier would change the OFF graph. q_cond is summed on the PRESSURE grid
        and rides the SAME bracket build as T and q (one extra gather, not a second bracket). A no-op
        when the prognostic fields are absent (q_c stays 0 with ``prognostic_condensate`` OFF, q_r is
        None with ``prognostic_rain`` OFF).

        EXACT SPECIFIC-HUMIDITY FORM (ROADMAP PHY-20, ``cfg.virtual_temp_exact``, default OFF): the vapor
        term above is T(1 + q/eps)/(1 + q), the EXACT density temperature for MIXING RATIOS, evaluated on
        this model's SPECIFIC humidities -- the right function at the wrong variable, which always
        UNDER-states the vapor buoyancy (1.82 K at T = 330 K, q = 0.10, ~9% of the vapor increment; 13.60 K
        at q = 0.2964). ON selects T(1 + 0.608 q - q_cond), exact for specific humidities. The flag is
        passed straight through to ``thermo`` -- one array-generic implementation, so CPU and GPU cannot
        diverge -- and is RE-READ every call, so the documented warm-start route (set
        ``cfg.virtual_temp_exact`` on a BUILT core) takes effect on the next EAGER step with no rebuild --
        but it must be set BEFORE the first step on a capture_ready run, since a CUDA graph records the
        branch that ran at capture and a replay never re-enters this Python. It is
        ORTHOGONAL to both flags above: at q = 0 the two forms are exactly T, so with ``virtual_buoyancy``
        OFF it is inert, and the loading is -T*q_cond either way. Turning it ON MOVES CANONICAL NUMBERS.

        THE MISSING PRESSURE TERM (ROADMAP PHY-22 option B', ``cfg.buoyancy_pressure_term``, default
        OFF). The dynamics eliminate pressure by writing the anelastic pressure term as the PERFECT
        GRADIENT -grad(p'/rho_ref), whose curl is exactly zero. What that leaves in the vorticity
        equation is NOT g*tau but
            b_eff = -g rho'/rho_ref - (p'/rho_ref) dln(rho_ref)/dz  ~  g*tau + R_d (dT_ref/dz) * eps,
        with eps = p'/p_ref -- so the shipped form omits exactly one term, and that term is proportional
        to the REFERENCE LAPSE RATE (R_d dT_ref/dz = -1.866 m/s^2), not to g*kappa (+2.803). It is the
        largest available correction on the canonical state, 0.151x of the resolved torque against
        PHY-21's 0.073x, PHY-19 loading's 0.035x and PHY-20's 0.002x -- larger than the three together --
        because it is first order in how far the live columns' HEIGHTS depart from the base state
        (kilometres here), not second order in the temperature departure.
        ``p_hydro`` IS NOT FREE: ``_hydrostatic_z`` gives the heights of layer CENTRES, so the pressure at
        the DYNAMICS height levels needs a second interpolation, which rides the SAME bracket
        ``_pressure_to_height_pair`` already builds (the layer pressures ``_p_mid_row`` are x-independent,
        so they go in as a single broadcast row). It is therefore the SAME hypsometric integration that
        placed T and q: with ``hydrostatic_virtual_t`` ON both come from one integral on the same T_rho,
        and the bridge cannot disagree with itself about where the column is.
        TWO FORMS, selected by ``cfg.buoyancy_pressure_form`` and validated at construction:
          "linear" -- db = alpha(z)*eps, alpha = R_d dT_ref/dz (the analytic ``self._buoy_alpha``: the
            constant -R_d*lapse_ref in the lapse branch, exactly 0 in the isothermal cap). The
            derivation's own answer, and the coefficient that maximises the shape agreement with the
            independent solenoidal target.
          "theta"  -- db = g*(T_h/T_ref)*((1+eps)^(-kappa) - 1), which added to g*tau is EXACTLY
            g*(theta_h - theta_ref)/theta_ref: the FULL Lipps-Hemler leg, carrying the whole nonlinear
            factor and the T_h/T_ref prefactor, not merely its first-order truncation alpha = -g*kappa.
        No free numeric coefficient is exposed: the amplitude is not settled to better than ~2x, so the
        two DERIVED endpoints ship and the impact arm measures the spread (scoping note section 9.1).
        CLAMPED CELLS -- eps IS FROZEN AT THE COLUMN-END VALUE. Where a dynamics height lies above the
        live column's topmost layer centre or below its lowest, the interpolated pressure is an
        interpolator end-value while p_ref(z) keeps falling, so a raw eps there would SWING (that swing is
        the mean-0.888 m/s^2 step the scoping note measured). eps is therefore gathered from the last
        ON-COLUMN level of each column at both ends -- the same clamping convention the bridge already
        applies to T, q and q_cond on the way up and to u and w on the way back, it adds no parameter,
        and it makes b exactly C0 across the clamp boundary. Argued and fixed BEFORE measurement; the
        three candidates are measured for sensitivity in ``phy22_clamped_cell_rules.py`` (scoping note
        section 9.2 for the argument, section 10 for the numbers).
        THE TERM IS A MOMENTUM SOURCE ONLY: b has exactly one physics consumer, ``vorticity_tendency``,
        so the tracer and heat budgets are untouched by construction.
        DEFAULT OFF and BYTE-IDENTICAL OFF: the branch is at the PYTHON level (a zero multiplier would
        change the captured CUDA graph), the OFF path takes the verbatim shipped calls, and both keys are
        re-read every call so the post-build warm-start route works on an EAGER core -- but they must be
        set BEFORE the first step on a capture_ready run. TURNING IT ON MOVES CANONICAL NUMBERS: it
        reduces the equator-to-pole buoyancy contrast aloft, which should weaken the overturning and
        could move the band (a first-order prediction from the sign of the contrast, not a run)."""
        z_p = getattr(self, "_z", None)
        if z_p is None:
            z_p = self._hydrostatic_z(self.T_atm)
        load = self.cfg.condensate_loading
        exact = self.cfg.virtual_temp_exact        # PHY-20: exact specific-humidity density temperature
        pterm = self.cfg.buoyancy_pressure_term    # PHY-22 B': the missing pressure term
        p_row = self._p_mid_row if pterm else None   # (1, nz) layer pressures [Pa], or OFF -> no 4th field
        p_h = None                                   # (nz, nx) live hydrostatic pressure at the dyn levels [Pa]
        if load:
            xpc = get_xp(self.q_c)
            q_cond_p = xpc.maximum(self.q_c, 0.0)                 # (nx, nz) cloud, clipped at 0
            if self.q_r is not None:
                q_cond_p = q_cond_p + xpc.maximum(self.q_r, 0.0)  # ... plus falling rain
        if self.cfg.virtual_buoyancy:
            # T and q share the column heights z_p -> map BOTH with a single bracket build (bit-identical to
            # two _pressure_to_height calls; halves the heavy per-row searchsorted+bincount+cumsum).
            q_clip = get_xp(self.q).clip(self.q, 0.0, th.Q_MAX)
            if load:
                if pterm:                                         # + p on the SAME bracket (4 gathers)
                    T_h, q_h, qc_h, p_h = self._pressure_to_height_pair(self.T_atm, q_clip, z_p,
                                                                        q_cond_p, p_row)
                else:
                    T_h, q_h, qc_h = self._pressure_to_height_pair(self.T_atm, q_clip, z_p, q_cond_p)
                T_h = th.density_temperature(T_h, q_h, qc_h, exact=exact)   # T_v - T*q_cond   [K]
            else:
                if pterm:                                         # p takes the free THIRD slot
                    T_h, q_h, p_h = self._pressure_to_height_pair(self.T_atm, q_clip, z_p, p_row)
                else:
                    T_h, q_h = self._pressure_to_height_pair(self.T_atm, q_clip, z_p)
                T_h = th.virtual_temperature(T_h, q_h, exact=exact)  # density temperature T_v >= T   [K]
        elif load:
            # vapor buoyancy OFF, loading ON: the flags are ORTHOGONAL -> T - T*q_cond (loading alone).
            # ``exact`` is passed for symmetry and is INERT here: both forms give exactly T at q_v = 0.
            if pterm:
                T_h, qc_h, p_h = self._pressure_to_height_pair(self.T_atm, q_cond_p, z_p, p_row)
            else:
                T_h, qc_h = self._pressure_to_height_pair(self.T_atm, q_cond_p, z_p)
            T_h = th.density_temperature(T_h, 0.0, qc_h, exact=exact)  # == T - T*q_cond   [K]
        elif pterm:
            T_h, p_h = self._pressure_to_height_pair(self.T_atm, p_row, z_p)   # T and p, one bracket
        else:
            T_h = self._pressure_to_height(self.T_atm, z_p)       # (nz, nx)
        # b feeds the fp32 dynamics stencils, but T_ref (host fp64) upcasts the expression -> cast back so
        # the buoyancy STATE stays at the working dtype (else the dynamics half silently reverts to fp64).
        # capture-ready: cast in place into the fixed b buffer (on CPU b is numpy; the GPU buoyancy device-
        # placement is Phase C/D -- on CPU the expr and the buffer share the numpy module, so out= is valid).
        b_expr = C.G_ACCEL * (T_h - self.T_ref[:, None]) / self.T_ref[:, None]
        if pterm:                                                 # PHY-22 B': + the pressure term
            b_expr = b_expr + self._buoyancy_pressure_term(T_h, p_h, z_p)
        self.b = self._cast(b_expr, out=self._obuf(self.b))

    def _buoyancy_pressure_term(self, T_h, p_h, z_p):
        """PHY-22 B': the buoyancy term the core's own pressure convention calls for and the shipped
        ``b = g*tau`` omits. Returns (nz, nx) [m/s^2], to be ADDED to the shipped expression.

        ``T_h`` (nz, nx) [K] is the density temperature already mapped onto the dynamics height grid --
        the SAME array the shipped leg uses, so every existing buoyancy flag composes without a second
        definition. ``p_h`` (nz, nx) [Pa] is the live column's hydrostatic pressure at those heights,
        from the shared bracket. ``z_p`` (nx, nz) [m] are the column layer-centre heights, TOP->SURFACE,
        needed only to locate the ends of each live column for the clamp rule.

        Array-generic and CUDA-capture safe: every operation below is an elementwise ufunc, a reduction
        or a gather, with NO Python branch on data (the only Python ``if`` is on the FORM string, which
        is read before capture) and no device->host sync.

        The clamp rule -- freeze eps at the column-end value -- and the argument for choosing it over
        hydrostatic extrapolation or a taper are in ``PHY22_MAPPING_CONVENTION_SCOPING.md`` section 9.2."""
        xpm = get_xp(T_h)
        nzh = self.z.shape[0]
        # eps = p'/p_ref, the fractional hydrostatic pressure departure of the live column [dimensionless].
        # ``_inv_p_ref`` is the fixed reference row 1/p_ref(z); the (nz,1) host operands sit in exactly the
        # position ``self.T_ref[:, None]`` occupies in the shipped expression.
        eps = p_h * self._inv_p_ref[:, None] - 1.0                       # (nz, nx) [dimensionless]
        # --- CLAMPED CELLS: freeze eps at the last ON-COLUMN dynamics level, at BOTH ends.
        #     k_top[i] = highest dynamics level at or below column i's topmost layer centre;
        #     k_bot[i] = lowest  dynamics level at or above column i's lowest  layer centre.
        #     Both are comparison counts (exact integers, no interpolation), clipped into range so a
        #     degenerate column cannot index out of bounds. The gather index clip(k, k_bot, k_top) then
        #     freezes eps outside [k_bot, k_top] and is the identity inside it, so the ON-COLUMN values
        #     are untouched and b is exactly C0 across the boundary.
        zg = self.z[None, :]                                             # (1, nz_h) [m], SURFACE->TOP
        k_top = (zg <= z_p[:, :1]).sum(axis=1) - 1                       # (nx,) index
        k_bot = (zg < z_p[:, -1:]).sum(axis=1)                           # (nx,) index
        k_top = xpm.minimum(xpm.maximum(k_top, 0), nzh - 1)
        k_bot = xpm.minimum(xpm.maximum(k_bot, 0), nzh - 1)
        kk = self._buoy_kcol                                             # (nz_h, 1) level index (fixed)
        idx = xpm.minimum(xpm.maximum(kk, k_bot[None, :]), k_top[None, :])   # (nz_h, nx) gather index
        eps = xpm.take_along_axis(eps, idx, axis=0)                      # (nz, nx) frozen off-column
        if self.cfg.buoyancy_pressure_form == "theta":
            # The FULL Lipps-Hemler leg: g*tau + this == g*(theta_h - theta_ref)/theta_ref exactly, with
            # theta_h = T_h*(P_SURF/p_hydro)^kappa. (1+eps) > 0 always, since p_hydro and p_ref are both
            # strictly positive. First order this is -g*kappa*eps, i.e. 1.50x the "linear" coefficient.
            # ``expm1(-kappa*log1p(eps))`` rather than ``(1+eps)**-kappa - 1``: the two are the same
            # number, but the difference form cancels catastrophically at small eps (which is most of
            # the habitable band), and this term must reduce CLEANLY to -g*kappa*eps there or the
            # small-amplitude convergence check below cannot separate it from round-off.
            fac = xpm.expm1(-C.KAPPA_POISSON * xpm.log1p(eps))           # (p_ref/p_hydro)^kappa - 1
            return C.G_ACCEL * (T_h / self.T_ref[:, None]) * fac         # [m/s^2]
        # "linear": alpha(z)*eps with alpha = R_d dT_ref/dz -- the buoyancy that survives the curl under
        # this core's perfect-gradient pressure convention, to first order in (tau, eps).
        return self._buoy_alpha[:, None] * eps                           # [m/s^2]

    # ------------------------------------------------------------------ MASS-CONSISTENT tracer transport
    # (the M3-prerequisite fix, CLOSED_ISSUES "tracer-advection mass-consistency"): a SINGLE discretely
    # divergence-free mass flux on the pressure grid drives BOTH the heat (theta) and moisture (q) limbs,
    # so a UNIFORM tracer stays uniform (uniform-invariance) AND the column integral INTEGRAL_k(tracer*dm)
    # is conserved to round-off. The shared helpers below build that flux and apply the flux-form upwind.
    def _divfree_mass_flux(self, u_p):
        """Build a discretely DIVERGENCE-FREE pressure-grid mass flux from the resolved cross-track wind
        ``u_p`` (nx, nz, on the pressure grid). Returns the face mass-flow rates per unit along-track
        length [kg/(m*s)]:

            ``Fx`` (nx+1, nz)   horizontal flow through the x-faces (face f between cells f-1 and f;
                                f=0 equator wall, f=nx pole wall -> both ZERO);
            ``Fz`` (nx, nz+1)   UPWARD flow through the vertical interfaces (c=0 model top, c=nz surface
                                -> both ZERO).

        Why a derived flux and not the raw (u_p, w_p): the anelastic dynamics keeps ``div(rho_ref u)=0`` on
        the HEIGHT grid, but the per-layer point velocities interpolated onto the PRESSURE grid are NOT
        discretely divergence-free, so a flux-form scheme using them spuriously redistributes a uniform
        tracer (the M3-prerequisite bug). Instead we use the FIXED, horizontally-uniform layer mass ``dm``
        (= dp/g, the reference air mass per layer -- the ``rho_ref`` of the issue's "div-free rho_ref mass
        flux") and DERIVE the vertical flow from horizontal mass continuity, integrated from the surface:

          1. horizontal face flux  ``Fx[f,k] = u_face[f,k] * dm[k]``  (u_face = x-interface average of u_p);
          2. RIGID-LID projection: subtract each face's mass-weighted column mean of u_face so the
             column-integrated horizontal transport is zero -- which forces ``Fz`` to vanish at BOTH the
             surface and the model top (no net mass through either lid);
          3. vertical flux from continuity  ``Fz[i,k] = Fz[i,k+1] - (Fx[i+1,k]-Fx[i,k])``, with Fz[surface]=0.

        Then for EVERY interior cell ``(Fx[i+1,k]-Fx[i,k]) + (Fz[i,k]-Fz[i,k+1]) = 0`` exactly (telescoping
        cumulative sum), so the flux is discretely divergence-free with zero boundary flux -- the two
        properties the flux-form upwind needs for uniform-invariance + exact conservation.

        DEVICE (M3.1): the array module follows ``u_p`` (``get_xp``); the fixed reference layer mass
        ``self.dm`` (a tiny (nz,) host array) is moved onto the same device. numpy in -> numpy out,
        bit-identical to pre-M3.1."""
        xp = get_xp(u_p)
        nx, nz = u_p.shape
        dm = xp.asarray(self.dm)[None, :]                        # (1, nz) fixed reference layer mass
        u_face = xp.zeros((nx + 1, nz))
        u_face[1:nx] = 0.5 * (u_p[:-1] + u_p[1:])               # x-interface winds; walls (0, nx) stay 0
        # rigid-lid projection: remove the barotropic (mass-weighted column-mean) part per face, so the
        # column-integrated horizontal mass transport is zero -> Fz=0 at the surface AND the top.
        u_face = u_face - (u_face * dm).sum(axis=1, keepdims=True) / dm.sum()
        Fx = u_face * dm                                        # (nx+1, nz) horizontal mass flow
        dH = Fx[1:] - Fx[:-1]                                   # (nx, nz) per-cell horizontal divergence
        Fz = xp.zeros((nx, nz + 1))
        # Fz[:,k] = -sum_{j>=k} dH[:,j]  (integrate continuity up from the surface Fz[:,nz]=0); the
        # barotropic removal makes the top Fz[:,0] = -sum_all dH = 0 as well.
        Fz[:, :nz] = -xp.cumsum(dH[:, ::-1], axis=1)[:, ::-1]
        return Fx, Fz

    def _flux_form_upwind(self, tracer, Fx, Fz):
        """First-order-upwind FLUX-FORM tendency d(tracer)/dt [tracer/s] under the divergence-free mass
        flux (``Fx``, ``Fz`` from ``_divfree_mass_flux``). Donor-cell upwinding at each face; the per-cell
        tendency is ``-(net mass-weighted tracer outflow) / (dm*dx)``. Because the mass flux is discretely
        divergence-free with zero wall/lid flux, a UNIFORM tracer gives zero tendency (uniform-invariance)
        and ``INTEGRAL_ik tracer*dm*dx`` is conserved to round-off (interior face fluxes telescope)."""
        xp = get_xp(tracer)
        nx, nz = tracer.shape
        # horizontal: upwind donor at the interior x-faces (f=1..nx-1); Fx>0 is +x (poleward) -> left donor
        tval_x = xp.where(Fx[1:nx] > 0, tracer[:-1], tracer[1:])
        Fxt = xp.zeros((nx + 1, nz)); Fxt[1:nx] = Fx[1:nx] * tval_x   # wall fluxes stay 0
        # vertical: upwind donor at the interior interfaces (c=1..nz-1); Fz>0 is UP -> donor is below (c)
        tval_z = xp.where(Fz[:, 1:nz] > 0, tracer[:, 1:nz], tracer[:, :nz - 1])
        Fzt = xp.zeros((nx, nz + 1)); Fzt[:, 1:nz] = Fz[:, 1:nz] * tval_z   # top/surface fluxes stay 0
        net_out = (Fxt[1:] - Fxt[:-1]) + (Fzt[:, :-1] - Fzt[:, 1:])         # (nx, nz) net outflow
        return -net_out / (xp.asarray(self.dm)[None, :] * self._dx_div)

    def _flux_form_fluxlim(self, tracer, Fx, Fz):
        """A higher-order (van Leer flux-limited / MUSCL) FLUX-FORM tendency under the SAME divergence-free
        mass flux as ``_flux_form_upwind`` -- the lower-implicit-diffusion option (M3.4b Task 1). At each
        interior face the donor-cell value is reconstructed to second order with the van Leer limiter
        ``phi(r) = (r+|r|)/(1+|r|)``, ``r`` the ratio of the upwind-side to downwind-side gradient:

            tracer_face = tracer[donor] + 0.5 * phi(r) * (tracer[acceptor] - tracer[donor])

        phi=0 recovers first-order upwind (at extrema r<=0, and at the near-wall faces where the far-upwind
        cell is out of range -> upwind fallback); phi=1 is Lax-Wendroff (centered). Because a SINGLE value
        is used for each face's flux, the interior fluxes telescope and the wall/lid fluxes are zero -- so it
        is BYTE-FOR-BYTE as conservative as the upwind sibling (``INTEGRAL tracer*dm*dx`` invariant to
        round-off) and uniform-invariant (a uniform tracer has zero gradients -> phi correction 0 -> upwind
        on a uniform field -> zero tendency). It is TVD/monotone under the dynamics' CFL-limited ``dt`` (van
        Leer sits in the Sweby TVD region). The point of the lower diffusion: it shrinks the SPURIOUS
        ``int(cp*T)`` change that upwind theta_pot diffusion injects, isolating the REAL APE->KE sink in the
        energy-closure LIMIT test (``heat_limb_enthalpy_audit`` / ``archive/M33_M34_RESULTS.md``)."""
        xp = get_xp(tracer)
        nx, nz = tracer.shape
        tiny = 1e-30

        def _phi(num, denom):                                   # van Leer correction 0.5*phi(num/denom)*denom
            r = num / xp.where(xp.abs(denom) > tiny, denom, tiny)
            return 0.5 * ((r + xp.abs(r)) / (1.0 + xp.abs(r))) * denom   # ->0 as denom->0 (uniform/extrema)

        # --- horizontal x-faces f=1..nx-1 (between cell f-1 and f); Fx>0 => left donor -----------------
        TL, TR = tracer[:-1], tracer[1:]                        # (nx-1, nz): cells f-1 (left), f (right)
        Fxf = Fx[1:nx]
        TLL = xp.concatenate([TL[:1], tracer[:-2]], axis=0)     # cell f-2 (far-upwind for Fx>0); f=1 -> upwind
        TRR = xp.concatenate([tracer[2:], TR[-1:]], axis=0)     # cell f+1 (far-upwind for Fx<0); f=nx-1 -> upwind
        face_pos = TL + _phi(TL - TLL, TR - TL)                 # Fx>0: donor=TL, acceptor=TR, farup=TLL
        face_neg = TR + _phi(TR - TRR, TL - TR)                 # Fx<0: donor=TR, acceptor=TL, farup=TRR
        Fxt = xp.zeros((nx + 1, nz))
        Fxt[1:nx] = Fxf * xp.where(Fxf > 0, face_pos, face_neg)

        # --- vertical z-faces c=1..nz-1 (between cell c-1 ABOVE and c BELOW); Fz>0 is UP => below donor --
        TA, TB = tracer[:, :-1], tracer[:, 1:]                  # (nx, nz-1): cells c-1 (above), c (below)
        Fzf = Fz[:, 1:nz]
        TBB = xp.concatenate([tracer[:, 2:], TB[:, -1:]], axis=1)   # cell c+1 (further below; Fz>0 far-upwind)
        TAA = xp.concatenate([TA[:, :1], tracer[:, :-2]], axis=1)   # cell c-2 (further above; Fz<0 far-upwind)
        face_up = TB + _phi(TB - TBB, TA - TB)                  # Fz>0: donor=TB(below), acceptor=TA(above)
        face_dn = TA + _phi(TA - TAA, TB - TA)                  # Fz<0: donor=TA(above), acceptor=TB(below)
        Fzt = xp.zeros((nx, nz + 1))
        Fzt[:, 1:nz] = Fzf * xp.where(Fzf > 0, face_up, face_dn)

        net_out = (Fxt[1:] - Fxt[:-1]) + (Fzt[:, :-1] - Fzt[:, 1:])
        return -net_out / (xp.asarray(self.dm)[None, :] * self._dx_div)

    def _tracer_vert_implicit_solve(self, tracer, Fz, dt, clip_negative=True):
        """Backward-Euler donor-cell solve of the VERTICAL flux-form tracer limb: return ``tracer_new``
        solving, per column i (pressure grid, k increases DOWNWARD; interface c between cell c-1 ABOVE
        and cell c BELOW; ``Fz`` (nx, nz+1) the UPWARD face mass flow [kg/(m*s)], zero at the top c=0 and
        surface c=nz -- the ``_divfree_mass_flux`` contract),

            dm_k dx_i (q_k^{n+1} - q_k^n) = -dt (Fzt_k - Fzt_{k+1}),   Fzt_c = Fz_c * q_donor(c)^{n+1}

        with the SAME donor convention as ``_flux_form_upwind`` (Fz_c > 0 is UP -> donor is cell c, below
        the face) but the face value taken at t^{n+1}. Splitting Fz_c = Fz_c^+ - Fz_c^- gives the
        tridiagonal bands (divide through by dx_i so they stay O(dm), the ``_sediment_rain`` idiom;
        r = dt/dx_i [s/m]):

            diag_k = dm_k + r (Fz_k^+ + Fz_{k+1}^-)         [kg/m^2]   (outflow through both faces)
            sub_k  = -r Fz_k^-                                          (inflow from the layer ABOVE)
            sup_k  = -r Fz_{k+1}^+                                      (inflow from the layer BELOW)

        Positive diagonal, non-positive off-diagonals, and every COLUMN of the matrix sums to dm_k (the
        off-diagonals are exactly the donor terms) -> an M-matrix: UNCONDITIONALLY STABLE at any
        Courant with q >= 0 preserved (inverse-positivity), and summing the update over k telescopes
        the interior fluxes so INTEGRAL_k tracer*dm is preserved EXACTLY up to the (zero) top/surface
        fluxes -- the same conservation statement as the explicit limb. PRECISION (#324 review): the
        M-matrix gives POSITIVITY unconditionally, but NOT a max principle for this solve in
        isolation -- a cell with net vertical mass CONVERGENCE (row sum dm_k + r(Fz_k - Fz_{k+1}) <
        dm_k; Fz alone is not divergence-free per cell) can concentrate the tracer above the initial
        max, which is physically correct at fixed anelastic dm. NO-NEW-EXTREMA is a property of the
        FULL split with the shared divergence-free flux (the compensating horizontal divergence
        removes the concentration) -- a caller reusing this solve standalone gets positivity and
        conservation, not monotonicity. The un-CFL'd vertical tracer Courant exposure
        ``_tracer_courant_guard`` measures is what this solve absorbs (chimney.py's
        NEST_CHIM_IMPLICIT split ported to the core's shared mass flux).

        ``clip_negative`` guards the Thomas sweep's round-off only (the M-matrix keeps the exact solution
        >= 0) -- moisture-class tracers clip, theta does not. The band build is pure elementwise ops from
        ``Fz`` (no data-dependent per-cell branching), so on device it records into the captured graph;
        the solve dispatches to the fp64 device Thomas RawKernel exactly as ``_sediment_rain`` does
        (host numpy / device kernel / bounded host excursion, in that order)."""
        xp = get_xp(tracer)
        nz = tracer.shape[1]
        dm = th._F(self.dm[None, :], xp)                         # (1, nz) fixed reference layer mass [kg/m^2]
        r = dt / self._dx_div                                    # [s/m]; scalar (uniform) or (nx, 1) true widths
        rFp = r * xp.maximum(Fz, 0.0)                            # (nx, nz+1) upward   face mass / dx [kg/m^2]
        rFm = r * xp.maximum(-Fz, 0.0)                           # (nx, nz+1) downward face mass / dx [kg/m^2]
        diag = dm + rFp[:, :nz] + rFm[:, 1:]                     # out: up through the top + down through the bottom
        sub = -rFm[:, :nz]                                       # in from ABOVE (downward flux, face k); sub[:,0]=0
        sup = -rFp[:, 1:]                                        # in from BELOW (upward flux, face k+1); sup[:,-1]=0
        rhs = dm * tracer                                        # mass form dm_k q_k^n [kg/m^2 * tracer]
        # bands for _solve_tridiag_columns: (nz, nx), sub[k] multiplying x[k-1] (the layer ABOVE).
        if xp is np:
            out = _solve_tridiag_columns(sub.T, diag.T, sup.T, rhs.T).T
        elif self._tridiag_kernel_dev is not None:
            # DEVICE Thomas RawKernel (fp64, capture-safe -- built at init, warmup precedes capture):
            # bands cast UP to fp64 (the implicit-solve discipline), the solve stays on device -- no
            # host round-trip inside the captured step (the _sediment_rain idiom).
            from ._gpu_kernels import solve_tridiag_columns_device
            f64 = xp.float64
            out = solve_tridiag_columns_device(self._tridiag_kernel_dev, sub.T.astype(f64),
                                               diag.T.astype(f64), sup.T.astype(f64),
                                               rhs.T.astype(f64)).astype(diag.dtype).T
        else:                                                    # eager gpu without the kernel: bounded host excursion
            out = to_device(_solve_tridiag_columns(to_host(sub.T), to_host(diag.T), to_host(sup.T),
                                                   to_host(rhs.T)), xp).T
        if clip_negative:
            out = xp.maximum(out, 0.0)     # M-matrix keeps >= 0; guard the solver's round-off only
        return out

    def _fz_zero(self, Fz):
        """Persistent all-zero vertical-flux twin of ``Fz`` for the horizontal-only explicit sub-update
        (allocated once, never written; a per-call ``zeros_like`` would be 4 allocations/step and a
        fresh pool address each recording)."""
        buf = getattr(self, "_fz_zero_buf", None)
        if buf is None or buf.shape != Fz.shape or get_xp(buf) is not get_xp(Fz):
            self._fz_zero_buf = buf = get_xp(Fz).zeros_like(Fz)
        return buf

    def _advect_tracer_split_implicit(self, field, kappa, Fx, Fz, dt):
        """The ``tracer_vert_implicit`` ON-path shared by the moisture-class tracers (q / q_c / q_r):
        Lie split, HORIZONTAL-FIRST -- the explicit horizontal flux-form upwind + cross-track diffusion
        advance ``field`` over dt (they ride the dx/|u| cap the CFL controller enforces), then the
        backward-Euler vertical donor-cell solve absorbs the un-CFL'd vertical limb on the SAME ``Fz``.
        Horizontal-first keeps the vertical solve last so its positivity bounds the returned state
        below (no explicit sub-update can re-introduce a negative after it); NO-NEW-EXTREMA holds for
        this full split because ``Fz`` and ``Fx`` share the divergence-free construction (the solve's
        docstring has the precise statement -- it is NOT a property of the vertical solve alone).
        Returns the RAW solve output; the caller funnels it through
        ``_cast(..., out=self._obuf(...))`` (fixed address under capture)."""
        f_star = field + dt * (self._flux_form_upwind(field, Fx, self._fz_zero(Fz))
                               + self._cross_diffuse(field, kappa))
        return self._tracer_vert_implicit_solve(f_star, Fz, dt)

    def _cross_diffuse(self, field, kappa):
        """Conservative FLUX-FORM cross-track (x) eddy-diffusion tendency [field/s]: interface fluxes
        telescope and the wall flux is zero, so ``INTEGRAL_x field`` (per layer) is invariant -- it only
        REDISTRIBUTES, never sources. Shared by the heat and moisture limbs so the two siblings match."""
        xp = get_xp(field)
        if self._dx_cell is None:
            face = kappa * (field[1:, :] - field[:-1, :]) / self.dx     # flux at the i+1/2 faces, (nx-1, nz)
            out = xp.zeros_like(field)
            out[:-1, :] += face / self.dx                               # cell i gains its i+1/2 face flux
            out[1:, :] -= face / self.dx                                # cell i loses its i-1/2 face flux
        else:
            hx = (self._dx_op[1:] - self._dx_op[:-1])[:, None]          # CENTER spacing at the faces (nx-1, 1)
            #   (_dx_op == self.x on CPU; the device twin after placement -- keeps the op device-resident)
            face = kappa * (field[1:, :] - field[:-1, :]) / hx          # face-gradient flux
            out = xp.zeros_like(field)
            out[:-1, :] += face / self._dx_cell[:-1, None]              # divergence over each cell's TRUE width
            out[1:, :] -= face / self._dx_cell[1:, None]
        return out

    def _resolved_mass_flux(self, u, flux):
        """Return the divergence-free mass flux ``(Fx, Fz)`` for the resolved-flow tracer advection: use the
        precomputed ``flux`` if ``step()`` passed one (the shared per-substep build), else map ``u`` to the
        pressure grid and build it here (byte-identical to the old inline path -- a direct caller / test that
        omits ``flux`` gets exactly the previous result). The three advected tracers (q, theta, q_c) use the
        SAME ``u`` and ``z_p`` within a substep, so a single build is bit-identical to three (asserted in the
        advect sub-profiler) -- the dedup that removes 2 of the 3 redundant ``_divfree_mass_flux`` calls."""
        if flux is not None:
            return flux
        z_p = getattr(self, "_z", None)
        if z_p is None:
            z_p = self._hydrostatic_z(self.T_atm)
        u_p = self._height_to_pressure(u, z_p)                   # (nx, nz) cross-track wind at the tracer layers
        return self._divfree_mass_flux(u_p)

    def _advect_moisture(self, u, dt, flux=None):
        """Advect the prognostic moisture by the RESOLVED flow in BOTH x and z (the M2b.3 deliverable "q
        advected by the resolved flow"). The cross-track wind ``u`` (height grid) is mapped to q's
        pressure-grid layers (``u_p``); a SINGLE discretely divergence-free mass flux (``_divfree_mass_flux``,
        the vertical limb derived from horizontal continuity) then drives a flux-form upwind step
        (``_flux_form_upwind``) plus a small conservative cross-track eddy diffusion (``_cross_diffuse``):

          * CROSS-TRACK: the surface branch carries moisture equatorward, the return branch poleward, with
            zero flux at the equator/pole walls.
          * VERTICAL: the ascending branch LOFTS vapor and the subsiding branch brings dry air down -- the
            resolved-flow complement to the convective adjustment's sub-grid loft. The flux through the
            surface and model top is ZERO, so the surface evaporation source and the condensation sink
            (handled in ``_thermo_substep``) are the ONLY vertical water sources/sinks.

        Done directly on q's pressure-grid master. The divergence-free mass flux makes this BOTH
        uniform-invariant (a uniform q stays uniform) AND exactly water-conserving (``INTEGRAL_k q*dm``
        invariant, interior fluxes telescope) -- the M3-prerequisite fix shared with ``_advect_heat``. On the
        EXPLICIT path (``advection_scheme == "explicit"``) no post-update positivity clamp is applied: the
        monotone first-order upwind under the dynamics' CFL-limited ``dt`` keeps q >= 0 without the
        one-directional water source a blanket ``max(.,0)`` clamp would introduce in the dry subsiding
        columns.

        COURANT DEPENDENCE (review caveat, _sl_FINAL_VERDICT.md -- which predates the clamp below, PR #203):
        "monotone under the CFL-limited dt" holds only while the per-cell tracer Courant <= 1. With
        ``advection_scheme != "explicit"`` the implicit-ω relief lets ``dt`` rise above that bound, so a
        finer dz or a stronger chimney can undershoot -- so on that IMPLICIT path a per-step positivity clamp
        (below) counts the excursions and the injected water (``_clamp_count``/``_clamp_mass``, read+reset
        via ``flush_clamp_stats``) and resets q to >= 0. ``step()`` also runs ``_tracer_courant_guard`` on
        the ON-path (skipped under ``capture_ready``) to warn/raise on Courant > 1 -- see it.

        ``flux`` (optional ``(Fx, Fz)``): the precomputed divergence-free mass flux. ``step()`` builds it ONCE
        per substep and shares it across the moisture/heat/condensate limbs (the three use the SAME ``u`` and
        ``z_p`` -> bit-identical flux); a direct caller (tests) that omits it rebuilds it here, byte-identical."""
        c = self.cfg
        Fx, Fz = self._resolved_mass_flux(u, flux)
        if c.tracer_vert_implicit:
            # split limb (config note at ``tracer_vert_implicit``): explicit horizontal + implicit vertical,
            # unconditionally monotone in z -- the clamp below then has only round-off to count.
            self.q = self._cast(self._advect_tracer_split_implicit(self.q, c.kappa_q, Fx, Fz, dt),
                                out=self._obuf(self.q))
        else:
            self.q = self._accumulate(self.q, dt * (self._flux_form_upwind(self.q, Fx, Fz)
                                                    + self._cross_diffuse(self.q, c.kappa_q)),
                                      out=self._obuf(self.q))   # capture-ready: in-place (tendency built first)
        if c.advection_scheme != "explicit":
            self._clamp_tracer_positive(self.q)   # accounted: the injection enters _cum_qadv

    def _advect_condensate(self, u, dt, flux=None):
        """Advect the prognostic cloud condensate ``q_c`` by the RESOLVED flow (CLOUD_PROGNOSTIC_ROADMAP
        Chunk 1), riding the SAME discretely divergence-free mass flux (``_divfree_mass_flux``) that drives
        ``q`` and ``theta`` -- so a UNIFORM q_c stays uniform and ``INTEGRAL q_c*dm`` is conserved to
        round-off for free, and the deck rides the flow / survives between steps.

        Uses q_c's OWN, SMALLER eddy diffusivity ``kappa_qc`` -- NOT the vapour ``kappa_q``: q_c is spatially
        intermittent (zero in clear air, sharp cloud edges), so the larger vapour regulariser would
        over-smear and dissolve the thin stratus deck the scheme exists to capture. First-order upwind
        (monotone under the dynamics' CFL-limited ``dt``) keeps q_c >= 0 on the EXPLICIT path with no clamp --
        mirroring ``_advect_moisture``, the closest positive-tracer sibling. ``flux`` as in
        ``_advect_moisture``. SAME Courant caveat: monotonicity needs per-cell Courant <= 1, which the
        implicit-ω relief can exceed -> on that IMPLICIT path the same per-step positivity clamp as
        ``_advect_moisture`` runs below (shared ``_clamp_count``/``_clamp_mass`` counters, read+reset via
        ``flush_clamp_stats``), and ``_tracer_courant_guard`` (in ``step()``, ON-path only) warns/raises on
        Courant > 1 (_sl_FINAL_VERDICT.md predates the clamp, PR #203)."""
        c = self.cfg
        Fx, Fz = self._resolved_mass_flux(u, flux)
        if c.tracer_vert_implicit:
            self.q_c = self._cast(self._advect_tracer_split_implicit(self.q_c, c.kappa_qc, Fx, Fz, dt),
                                  out=self._obuf(self.q_c))
        else:
            self.q_c = self._accumulate(self.q_c, dt * (self._flux_form_upwind(self.q_c, Fx, Fz)
                                                        + self._cross_diffuse(self.q_c, c.kappa_qc)),
                                        out=self._obuf(self.q_c))   # capture-ready: in-place (tendency built first)
        if c.advection_scheme != "explicit":
            self._clamp_tracer_positive(self.q_c)   # accounted: the injection enters _cum_qadv

    def _advect_rain(self, u, dt, flux=None):
        """Advect the prognostic FALLING RAIN ``q_r`` by the RESOLVED flow (stretched-x stage 7) -- the
        exact sibling of ``_advect_condensate``, riding the SAME shared divergence-free mass flux
        (``flux``): one more flux-form call per dynamics step, so at resolved updraft speeds
        w ~ 25-36 m/s >> rain_v_term the rain LOFTS with the flow instead of the instant column rainout
        (the structural fix the port exists for). Same first-order upwind scheme q_c uses, same small
        ``kappa_qc`` regulariser (rain is as spatially intermittent as cloud water), same
        uniform-invariance + INTEGRAL q_r*dm conservation to round-off, same implicit-path positivity
        clamp (shared ``_clamp_count``/``_clamp_mass``) and ``_tracer_courant_guard`` coverage (the
        guard measures the shared flux, not per-tracer). The SEDIMENTATION at -rain_v_term is NOT here --
        it is the per-thermo-substep implicit ``_sediment_rain`` (the chimney split (u, w) vs
        (u, w - v_term) becomes resolved-advection + separate implicit fall in the core)."""
        c = self.cfg
        Fx, Fz = self._resolved_mass_flux(u, flux)
        if c.tracer_vert_implicit:
            self.q_r = self._cast(self._advect_tracer_split_implicit(self.q_r, c.kappa_qc, Fx, Fz, dt),
                                  out=self._obuf(self.q_r))
        else:
            self.q_r = self._accumulate(self.q_r, dt * (self._flux_form_upwind(self.q_r, Fx, Fz)
                                                        + self._cross_diffuse(self.q_r, c.kappa_qc)),
                                        out=self._obuf(self.q_r))
        if c.advection_scheme != "explicit":
            self._clamp_tracer_positive(self.q_r)   # accounted: the injection enters _cum_qadv

    def _advect_heat(self, u, dt, flux=None):
        """Advect the column POTENTIAL TEMPERATURE by the RESOLVED flow in BOTH x and z -- the M2b.6
        inter-column HEAT transport that closes the fused-domain energy exchange (Issue A). Potential
        temperature ``theta_pot = T (P_SURF/p)^kappa`` is the adiabatic invariant the flow rearranges; T
        is recovered ``T = theta_pot (p/P_SURF)^kappa`` after the advection, so the resolved ASCENT cools
        and the resolved SUBSIDENCE WARMS adiabatically (the subtropical subsidence-warming the
        milestone-1 PoC lacked -- the mechanism that lets the cell move T(theta) off the per-column RCE
        lineage), while radiation + convective adjustment + condensation (in ``_thermo_substep``) keep
        acting on T on the pressure grid.

        WHY THIS DISSOLVES ISSUE A (and why "energy closes" is then a *consistency* check, not a verdict):
        the SAME discretely divergence-free mass flux as ``_advect_moisture`` drives a flux-form step (van
        Leer ``fluxlim`` by default, first-order ``upwind`` optional -- ``cfg.heat_advection``) whose interior
        fluxes telescope (ZERO flux through the surface, model top, and equator/pole walls),
        plus a conservative cross-track eddy diffusion, so the resolved transport adds NO net energy to the
        domain -- it only REDISTRIBUTES theta_pot. With psi=0 walls (#5) the lateral export is exactly zero,
        so the global TOA balance INTEGRAL(absorbed SW - OLR)dx is structurally unchanged by transport; the
        substance of the Issue-A test is therefore NOT "does energy close" (it must, by construction -- the
        forced-observable trap the roadmap warns of) but "does the heat-transporting cell reach a PHYSICAL
        steady state (no frozen/runaway tail, |psi|->0 poleward) and what T(theta) does it settle on."

        MASS-CONSISTENT (the M3-prerequisite fix, OPEN_ISSUES "tracer-advection mass-consistency"; shared
        with ``_advect_moisture``): the transport uses the FIXED reference layer mass ``dm`` and a vertical
        mass flux DERIVED from horizontal continuity (``_divfree_mass_flux``), so it is BOTH
        uniform-invariant (a UNIFORM theta stays uniform -- the bug this fix removes) AND mass-conserving
        (``INTEGRAL theta_pot dm`` invariant to round-off). Either scheme is TVD-stable under the
        dynamics' CFL-limited ``dt``. ``flux`` as in ``_advect_moisture``."""
        c = self.cfg
        Fx, Fz = self._resolved_mass_flux(u, flux)
        exner = (self.p_mid[None, :] / C.P_SURF) ** C.KAPPA_POISSON   # (1, nz) Exner function p^kappa
        theta = self.T_atm / exner                               # (nx, nz) potential temperature [K]
        advect = self._flux_form_fluxlim if c.heat_advection == "fluxlim" else self._flux_form_upwind
        # HEAT-ADVECTION SUB-CYCLING (diagnostic; _sl_FINAL_VERDICT.md): the first-order upwind heat
        # transport runs at the DYNAMICS dt, so its numerical diffusion (w*dz/2)(1-Courant) drops as the
        # implicit-omega dt raises the chimney Courant. This was a CANDIDATE carrier of the dt=60 climate
        # shift -- but sub-cycling it to n=3 (dt_sub=20 = reference) moved the gap by ZERO (FALSIFIED;
        # _sl_FINAL_VERDICT.md). The real carrier is the column THERMO cadence, not heat advection. The flag
        # is kept as the diagnostic that proves heat-advection diffusion is NOT the carrier (and the tracer-
        # Courant exposure it touches is a separate robustness risk -- see _tracer_courant_guard).
        # ``heat_advect_subcycle_n>1`` advances theta in n sub-steps at dt/n with the SAME frozen mass flux
        # (Fx, Fz). n=1 (default) = the verbatim single step (byte-id).
        n_h = int(getattr(c, "heat_advect_subcycle_n", 1))
        if c.tracer_vert_implicit:
            # split limb (config note at ``tracer_vert_implicit``): the HORIZONTAL flux limb stays
            # explicit at the configured order (fluxlim/upwind -- a zeroed Fz drops the vertical fluxes),
            # then the backward-Euler vertical donor-cell solve advances theta on the SAME Fz. The
            # vertical limb is FIRST-ORDER donor-cell regardless of ``heat_advection`` -- the price of the
            # unconditionally monotone M-matrix (the same operator choice as the omega
            # ``_implicit_vertical_advection``). Applied per sub-cycle so n_h keeps its dt_h semantics.
            Fz0 = self._fz_zero(Fz)
            dt_h = dt / max(n_h, 1)
            for _ in range(max(n_h, 1)):
                theta = theta + dt_h * (advect(theta, Fx, Fz0) + self._cross_diffuse(theta, c.kappa_T))
                theta = self._tracer_vert_implicit_solve(theta, Fz, dt_h, clip_negative=False)
        elif n_h <= 1:
            theta = theta + dt * (advect(theta, Fx, Fz) + self._cross_diffuse(theta, c.kappa_T))
        else:
            dt_h = dt / n_h
            for _ in range(n_h):
                theta = theta + dt_h * (advect(theta, Fx, Fz) + self._cross_diffuse(theta, c.kappa_T))
        # exner/theta are fp64 (Exner from p_mid); recover T and cast back to the working dtype (T_atm store).
        self.T_atm = self._cast(theta * exner, out=self._obuf(self.T_atm))   # recover T (in-place capture-ready)

    def _tracer_courant_guard(self, u, dt):
        """ROBUSTNESS GATE for the implicit-ω advection scheme (review caveat, _sl_FINAL_VERDICT.md): when
        the implicit-ω limb relieves the ``dz_min/|w|`` CFL (``advection_scheme != "explicit"``), the
        per-step ``dt`` rises above the vertical-advection CFL the EXPLICIT tracer transport (``_advect_heat``
        / ``_advect_moisture`` / ``_advect_condensate``) still rides. Those flux-form transports are monotone
        (so q, q_c >= 0 with nothing for the tracers' positivity clamp to remove) ONLY while each cell's
        total outgoing-mass Courant ``dt * (out-face mass flux) / (dm*dx) <= 1``. On the converged donor it
        stays < 1 because
        ``_cfl_dt`` paired the global ``dz_min`` (near-surface) with the global ``|w|max`` (chimney aloft) --
        a worst case that does not physically coincide -- but that is STATE- and RESOLUTION-dependent: a
        finer dz or a stronger/lower chimney could push a cell past Courant 1 -> the moisture/condensate
        limbs' per-step positivity clamp then injects unphysical water (counted: ``_clamp_count``/
        ``_clamp_mass``, read via ``flush_clamp_stats``) instead of q/q_c silently feeding negative values
        into condensation/latent heat.

        So when the scheme is ON, measure the REALIZED per-cell Courant on the SAME divergence-free
        pressure-grid mass flux ``Fz``/``Fx`` the tracers actually advect with (NOT the height-grid dynamics
        ``w`` -- the tracers ride the pressure-grid ``Fz``), and warn ONCE if it exceeds 1 (or raise, if
        ``tracer_courant_strict``). This is purely diagnostic: it does not alter the prognostic state.
        Under ``tracer_vert_implicit`` the vertical limb is the backward-Euler solve (no Courant bound),
        so the warn threshold applies to the HORIZONTAL part only (the vertical Courant is still
        computed for the message). Cost is one extra ``_divfree_mass_flux`` per step on the ON-path only;
        OFF (the default explicit path) NEVER calls this method -> byte-identical, zero new work.

        NB the heat-advection sub-cycle (`heat_advect_subcycle_n`) NOT moving the dt=60 climate gap
        (_sl_FINAL_VERDICT.md) proves this tracer-Courant exposure is a ROBUSTNESS/correctness risk, NOT the
        climate carrier (which is the column thermo cadence) -- the two are separate."""
        c = self.cfg
        xp = get_xp(u)
        z_p = getattr(self, "_z", None)
        if z_p is None:
            z_p = self._hydrostatic_z(self.T_atm)
        u_p = self._height_to_pressure(u, z_p)
        Fx, Fz = self._divfree_mass_flux(u_p)                    # (nx+1,nz), (nx,nz+1) pressure-grid mass flow
        dm_dx = xp.asarray(self.dm)[None, :] * self._dx_div      # cell mass per along-track length
        #   ((1, nz) on the uniform grid; (nx, nz) on the stretched grid -- per-cell true widths)
        # per-cell OUTGOING mass flux summed over the four faces (donor-cell upwind leaves through the faces
        # whose flow points OUT of the cell): horizontal out = max(Fx_right,0)+max(-Fx_left,0); vertical
        # out = max(Fz_top,0)+max(-Fz_bottom,0). Fz>0 is UP (top face = index k, bottom = k+1).
        out_x = xp.maximum(Fx[1:], 0.0) + xp.maximum(-Fx[:-1], 0.0)        # (nx, nz)
        out_z = xp.maximum(Fz[:, :-1], 0.0) + xp.maximum(-Fz[:, 1:], 0.0)  # (nx, nz)
        if c.tracer_vert_implicit:
            # the vertical limb is the backward-Euler donor-cell solve (unconditionally monotone at any
            # Courant), so only the still-explicit HORIZONTAL limb carries a monotonicity bound.
            courant = float(xp.max(out_x * dt / dm_dx))
        else:
            courant = float(xp.max((out_x + out_z) * dt / dm_dx))
        # the vertical-only Courant the reviewers named (|w|*dt/dz analog on the pressure grid), reported too
        cz = float(xp.max(out_z * dt / dm_dx))
        self._last_tracer_courant = courant
        if courant > 1.0:
            msg = (f"tracer vertical/total Courant {courant:.2f} > 1 at dt={dt:.1f}s under "
                   f"advection_scheme='{c.advection_scheme}' (vertical part {cz:.2f}): the explicit "
                   f"flux-form tracer transport (q, q_c, theta) is no longer guaranteed monotone -- the "
                   f"moisture/condensate limbs' per-step positivity clamp injects unphysical water (counted "
                   f"in _clamp_count/_clamp_mass; read via flush_clamp_stats), and theta has no clamp. The "
                   f"implicit-ω CFL relief raised dt above the tracer-advection CFL. Lower dt_dyn, refine "
                   f"nz with care, or sub-cycle the tracers. See _sl_FINAL_VERDICT.md.")
            if getattr(c, "tracer_courant_strict", False):
                raise RuntimeError(msg)
            if not getattr(self, "_warned_tracer_courant", False):
                import warnings
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                self._warned_tracer_courant = True

    def _operator_ke_removal_col(self, u, w, nu_t, nu_v=None, T_diff=None):
        """The per-column resolved-KE the applied momentum-removal OPERATORS actually remove [W/m^2]
        (M3.4b Task 2 -- the budget-closing dissipation): the eddy diffusion ``div(nu_t grad omega)``
        PLUS, when on, the surface drag's Rayleigh sink ``-surface_drag_rate*omega`` at j=1 AND the honest
        bulk friction's curl-of-stress (``surface_drag_bulk``) -- each folded into the same linear inversion,
        so D_col is the TOTAL removed KE and the dissipative-heating deposit returns it all (no drag leak).

        ``T_diff`` (optional): the ACTUAL applied diffusion-vorticity tendency, supplied by the
        implicit-vertical-diffusion path (where the vertical limb is the implicit increment
        (omega_new-omega_star)/dt, NOT the explicit ``div(nu_v grad omega)``). When None (the default /
        explicit path) it is recomputed from the explicit operator below -- byte-identical. The
        combined vorticity tendency is inverted through the SAME anelastic Poisson the dynamics uses
        (``_lu_dyn``) to its streamfunction increment, giving the velocity tendency (du, dw); the resolved-KE
        change density is ``-rho_ref (u*du + w*dw)`` [W/m^3] (positive where KE is removed), column-integrated.

        WHY THIS, NOT the strain-form ``int rho_ref nu_t |S|^2`` (the Task-2 finding): in this vorticity-
        streamfunction core the eddy diffusion acts on VORTICITY (``div(nu_t grad omega)``), which for a
        spatially-varying nu_t is NOT the curl of the variable-viscosity momentum stress ``div(2 nu_t S)``.
        So the KE it removes differs from the strain-form Smagorinsky production by the grad(nu_t) commutator
        -- measured ~2.6x SMALLER (resolution-robust) on the converged Task-1 states. The operator form
        equals the actual APE->KE->dissipation leak ``|enth|`` (ratio ~1.1-1.2 -> 1 with resolution), so it
        is what closes ``rel_imbalance``; the strain form over-returns ~2.6x and would overshoot it negative.

        ANISOTROPIC CLOSURE (issue #81): when ``nu_v`` is given the dynamics applied the ANISOTROPIC eddy
        operator ``_div_grad_aniso(omega, nu_t=nu_h, nu_v)``, so the KE-removal inversion MUST use the same
        operator (PITFALL: using the isotropic ``_div_grad`` here would return the wrong removed-KE and the
        energy-closing deposit would not balance). ``nu_v=None`` keeps the isotropic ``_div_grad`` (byte-id)."""
        if T_diff is None:
            T_diff = (_div_grad(self.omega, nu_t, self._dx_op, self._dz_op) if nu_v is None
                      else _div_grad_aniso(self.omega, nu_t, nu_v, self._dx_op, self._dz_op))  # eddy-diffusion tend.
            # Surface drag (#85 item 2 follow-up, CLOSED): the Rayleigh sink -rate*omega(j=1) AND the honest
            # bulk-friction curl-of-stress (surface_drag_bulk) the substep applies are each one more
            # vorticity-space momentum-removal term, and the Poisson inversion below is LINEAR -- fold them
            # into the SAME solve so D_col is the TOTAL (eddy + drag) removed KE and the dissipative-heating
            # deposit returns it all (no drag energy leak on the operator path). Both OFF (the defaults) leave
            # T_diff untouched (byte-identical).
            if self.cfg.surface_drag_rate > 0.0:
                T_diff[1, :] = T_diff[1, :] - self.cfg.surface_drag_rate * self.omega[1, :]
            if self.cfg.surface_drag_bulk:
                T_diff = T_diff + self._surface_drag_curl(u)       # honest bulk drag: same tendency as the substep
        # else: the implicit-vertical-diffusion path passed the ACTUAL applied diffusion tendency.
        if self._dyn_on_device:
            # Device-resident KE removal (Phase C): solve T_diff through the SAME device DST Poisson, recover
            # the velocity tendency + the KE-removal density ON DEVICE, integrate with the (device-aware)
            # _vint, and pull ONLY the small (nx,) D_col to host (the host coupled thermo consumes it -- a
            # documented (nx,) seam, NOT the (nz,nx) round-trip the host path took). T_diff may arrive host
            # (the implicit path builds it from omega_new on host) -> place it on device for the fp64 solve.
            xp = self.xp
            T_diff_dev = to_device(T_diff, xp).astype(xp.float64)            # fp64 RHS for the ill-cond. solve
            dpsi = self._lu_dyn_dev.solve(T_diff_dev).astype(self._dtype)    # (nz,nx) device, back to dtype
            du, dw = velocities_from_psi(dpsi, self._beta, self._dx_op, self._dz_op)   # device velocity tendency
            # Phase E: read the DEVICE-placed rho_ref (``rho_ref_dev``) -- ``xp.asarray(self.rho_ref, ...)``
            # would re-upload the host (nz,) profile to device EVERY substep (H2D, illegal during capture).
            rho_dev = xp.asarray(self.rho_ref_dev, du.dtype)[:, None]        # already device -> no-op cast, no H2D
            ke_dens = -rho_dev * (u * du + w * dw)                           # [W/m^3] device
            D_col = self._vint(ke_dens)                                      # (nx,) on device
            # Phase D: the COUPLED thermo is now device-resident, so KEEP D_col on device (the dissipative-
            # heating deposit multiplies it against the device eps_p/dm -- a host pull would re-trigger the
            # mixed-input trap). The Phase-C DYNAMICS LIMB (couple_thermo=False) has no thermo consumer, so
            # pull to host there (the M3.1 (nx,) seam) -- byte-identical to the shipped Phase-C path.
            return D_col if self.couple_thermo else to_host(D_col)
        # CPU: the host-only LU solve; bring (u, w, rho_ref) to host too (to_host is a no-op on numpy =>
        # byte-identical). The deposited column heat is a host quantity (coupled thermo is host-only).
        dpsi = self._lu_dyn.solve(to_host(T_diff).ravel()).reshape(self.cfg.nz, self.cfg.nx)
        du, dw = velocities_from_psi(dpsi, self._beta, self._dx_op, self._dz_op)     # the velocity tendency it induces
        u_h, w_h = to_host(u), to_host(w)
        ke_dens = -to_host(self.rho_ref)[:, None] * (u_h * du + w_h * dw)    # [W/m^3]
        return self._vint(ke_dens)                                          # (nx,) column KE removal [W/m^2]

    def _dissipative_heating_tendency(self, u, w):
        """M3.4b Task 2 -- the FRICTIONAL-DISSIPATION-HEATING tendency [K/s] on the PRESSURE grid (nx, nz):
        return the kinetic energy the eddy viscosity removes from the resolved flow back to the air as heat.
        Closes the REAL APE->KE->dissipation leak Task 1 proved (``_advect_heat`` cools int(cp*T) -- the
        cell's APE->KE conversion -- the eddy viscosity ``vorticity_tendency`` removes that KE, and NOTHING
        re-heated T_atm, so int(SW_abs-OLR) plateaued at ~|enth|).

        The column-integrated heat to return ``D_col`` [W/m^2] depends on ``cfg.dissipation_form``:
          - ``"operator"`` (default, budget-closing): the KE the vorticity-diffusion operator actually removes
            (``_operator_ke_removal_col``) -- equals the leak |enth|, so rel_imbalance -> 0.
          - ``"strain"`` (the spec's literal term, diagnostic): the strain-form Smagorinsky production
            ``int rho_ref nu_t |S|^2 dz`` -- the Task-2 finding shows this OVER-estimates the removed KE ~2.6x.
        nu_t (FULL: (C_s*Delta)^2|S| + floor) and |S| are recomputed from the SAME (u, w) the dynamics used.

        GRID + EXACT ENERGY RETURN: dissipation lives on the height grid (nz, nx), T_atm on the pressure grid
        (nx, nz). The column heat ``D_col`` is deposited with the (sign-definite, >=0) strain-dissipation
        vertical SHAPE remapped height->pressure (placement matters: it sets where OLR -- hence T(theta) --
        responds), RENORMALIZED per column by ``beta`` so ``sum_k cp*dm_k*dT_k == D_col`` EXACTLY (to
        round-off) regardless of the np.interp error -- the energy-return identity the build is gated on.
        Using the (positive) strain shape -- even for the operator magnitude -- keeps the deposition robust
        (the shape's column integral never vanishes where there is flow) and the per-layer sign clean. Host
        path (coupled thermo is host-only; ``_height_to_pressure`` is the batched np.interp-equivalent
        reconciliation, issue #99)."""
        c = self.cfg
        z_p = getattr(self, "_z", None)
        if z_p is None:
            z_p = self._hydrostatic_z(self.T_atm)
        # ANISOTROPIC CLOSURE (#81): match the eddy viscosity the dynamics actually applied -- nu_h on the
        # strain shape (the deposition SHAPE only, less critical) and (nu_h, nu_v) into the operator KE
        # removal (the MAGNITUDE, load-bearing: the operator-form D_col must invert the SAME aniso operator).
        if c.closure_anisotropic:
            nu_t, nu_v = smagorinsky_viscosity_aniso(u, w, self._dx_op, self._dz_op, c.C_s, c.nu_b)
        else:
            nu_t = smagorinsky_viscosity(u, w, self._dx_op, self._dz_op, c.C_s, c.nu_b)   # FULL nu_t (Smagorinsky + floor)
            nu_v = None
        if c.nu_h_floor:
            nu_t = nu_t + c.nu_h_floor                  # grid-independent horizontal floor (match the dynamics' nu_h)
        eps_spec = nu_t * strain_rate_magnitude(u, w, self._dx_op, self._dz_op) ** 2  # (nz, nx) specific strain diss [W/kg], >=0
        if c.dissipation_form == "operator":
            # Prefer the value the dynamics substep stashed from the PRE-update vorticity (consistent with the
            # (u, w) that produced this substep's diffusion); fall back to recomputing from self.omega for a
            # direct/standalone call (the unit tests, the harness audit) where no substep ran. This makes the
            # LIVE deposit equal the KE the substep actually removed (omega_old), not a one-tendency-stale
            # omega_new -- see ``_dynamics_substep``.
            D_col = self._diss_Dcol
            if D_col is None:
                D_col = self._operator_ke_removal_col(u, w, nu_t, nu_v)      # (nx,) operator KE removal [W/m^2]
        else:                                                                # "strain" (validated at construction)
            # rho_ref is a host dynamics reference profile; eps_spec is device on GPU -> place rho_ref onto
            # eps_spec's module to avoid the mixed-input trap (no-op on CPU; byte-identical).
            rho_e = get_xp(eps_spec).asarray(self.rho_ref)
            D_col = self._vint(rho_e[:, None] * eps_spec)                     # (nx,) strain-form diss [W/m^2]
        eps_p = self._height_to_pressure(eps_spec, z_p)                       # (nx, nz) >=0 vertical SHAPE, pressure grid
        xpm = get_xp(eps_p)                                                   # device-generic (Phase D)
        # D_col rides eps_p's module: on the dst device path it is already device; on the GPU+splu coupled
        # path _operator_ke_removal_col returns it HOST (the host splu branch) while eps_p is device, so place
        # it onto eps_p's module to avoid the mixed-input trap (no-op when both already match).
        D_col = to_device(D_col, xpm)
        denom = xpm.sum(self.dm[None, :] * eps_p, axis=1)                     # (nx,) sum_k dm_k eps_p [W/m^2], >0 with flow
        beta = xpm.where(denom > 1e-30, D_col / (denom + 1e-30), 0.0)         # per-column energy-exact rescale to D_col
        return eps_p * beta[:, None] / C.CP_AIR                               # (nx, nz) [K/s]: sum cp dm dT == D_col

    # ------------------------------------------------------------------ resolved dynamics (#0,#2,#4; M2b.2)
    def velocities(self):
        """Current anelastic velocities (u, w) on the height grid from the streamfunction psi."""
        return velocities_from_psi(self.psi, self._beta, self._dx_op, self._dz_op)

    def _vint(self, F):
        """Vertical column integral int F dz over the height grid -> (nx,). UNIFORM grid:
        ``sum(F, axis=0) * dz`` (the rectangle rule, byte-identical). STRETCHED grid: per-level
        cell-thickness weights ``sum_j F_j * dzc_j`` (dzc sums exactly to z_top), so the column energy
        integrals (KE removal, strain dissipation) stay correct under the non-uniform spacing.

        DEVICE (Phase C): array-module-agnostic via ``get_xp(F)`` so a device-resident ``ke_dens`` (the
        operator KE-removal path) is integrated ON DEVICE and only the small (nx,) result crosses to host.
        On a numpy ``F`` this is the verbatim ``np.sum`` (byte-identical); ``self._dz_cell`` (host) is moved
        onto F's module + dtype only on the device branch."""
        xp = get_xp(F)
        if self._dz_cell is None:
            return xp.sum(F, axis=0) * self.dz
        dz_cell = self._dz_cell if xp is np else xp.asarray(self._dz_cell, F.dtype)
        return xp.sum(F * dz_cell[:, None], axis=0)

    def bulk_drag_rate(self, C_D=1.2e-3):
        """The bulk-aerodynamic linear (Rayleigh) vorticity-drag rate r = C_D * U / z1 [1/s], from the
        model's own givens: the standard surface bulk transfer coefficient ``C_D`` (= Ch = Ce = 1.2e-3,
        ``surface.py``), the bulk-flux wind ``cfg.wind``, and the LIVE lowest-level thickness
        z1 = z[1] - z[0]. This is the momentum sink the free-slip bottom lacks; pass the result as
        ``cfg.surface_drag_rate`` (BL_REFINEMENT_SPEC.md #F). z1 is read from the live grid, so it tracks
        the refinement automatically: on the uniform fleet grid z1 ~ 1.2 km (nz=18) -> r ~ 6e-6 1/s
        ~ 1/(1.9 d); on a BL-refined grid z1 ~ 10-30 m raises r proportionally -- the ~10 m exchange level
        wants the standard 10 m C_D. A GIVEN, not a tuned knob."""
        z1 = self._z1                                                # Phase E: cached host float (z may be on device)
        return C_D * self.cfg.wind / z1

    def _surface_drag_curl(self, u):
        """Vorticity tendency [1/s^2] from HONEST bulk surface friction: the curl of the surface stress
        tau = rho*C_D*|u1|*u1. The lowest-cell horizontal-momentum sink is du/dt = -(C_D*|u1|/z1)*u1 (the
        stress divergence over the lowest cell, quadratic in the LOCAL wind, standard C_D); the vorticity
        equation sees its vertical derivative d/dz(du/dt). This is a GENUINE momentum/KE sink (unlike the
        Rayleigh omega[1] damping, which barely touches the buoyancy-driven jet) -- physical friction, not a
        knob. Applied only at the lowest interior level (j=1); zero elsewhere. The KE it removes is returned
        as heat (this same tendency is folded into ``_operator_ke_removal_col``), so the budget stays closed."""
        xp = get_xp(u)
        z1 = self._z1                                                # Phase E: cached host float (no D2H sync)
        Du = xp.zeros_like(u)
        u1 = u[1, :]
        Du[1, :] = -(self.cfg.surface_drag_cd * xp.abs(u1) / z1) * u1     # lowest-cell momentum drag [m/s^2]
        return _ddz(Du, self._dz_op)                                     # d/dz of the drag force -> domega/dt

    def _implicit_vertical_diffusion(self, omega_star, dt, coef_z):
        """Backward-Euler solve of the VERTICAL eddy-diffusion limb: return ``omega_new`` solving
        ``(I - dt L_z) omega_new = omega_star`` per column, with ``L_z`` the SAME finite-volume vertical
        operator ``_vertical_eddy_term`` applies (face-averaged ``coef_z``, per-layer face/cell spacings)
        and Dirichlet ``omega=0`` at the free-slip walls (j=0, nz-1). Removing this (usually binding)
        diffusive limb from the explicit dt is the ~min(dz) CFL relief of BL_REFINEMENT_SPEC.md #C/#d.
        ``coef_z`` is the vertical eddy viscosity (nu_v on the anisotropic path, nu_t on the isotropic
        path), scalar or (nz, nx).

        DEVICE (Phase C): on device="gpu" the bands are assembled with the device array module and the
        per-column Thomas solve runs in the device RawKernel (``_tridiag_kernel_dev``, fp64), returning a
        device ``omega_new`` -- no host round-trip. On CPU (and GPU+splu, which lacks the device solver) it
        is the verbatim host numpy solve (the bands are built fp64, the Thomas sweep is
        ``_solve_tridiag_columns``; byte-identical)."""
        if self._dyn_on_device:
            xp = self.xp
            omega_star = to_device(omega_star, xp).astype(xp.float64)        # fp64, on device
            cz = (to_device(coef_z, xp).astype(xp.float64)) * xp.ones_like(omega_star)
            czp = 0.5 * (cz[1:-1] + cz[2:])
            czm = 0.5 * (cz[:-2] + cz[1:-1])
            if self._stretched:
                z = xp.asarray(self.z, xp.float64)
                hp = (z[2:] - z[1:-1])[:, None]
                hm = (z[1:-1] - z[:-2])[:, None]
                dzc = (0.5 * (z[2:] - z[:-2]))[:, None]
            else:
                hp = hm = dzc = self.dz
            A_p = xp.zeros_like(omega_star); A_m = xp.zeros_like(omega_star)
            A_p[1:-1] = czp / (hp * dzc)
            A_m[1:-1] = czm / (hm * dzc)
            sub = -dt * A_m
            sup = -dt * A_p
            diag = 1.0 + dt * (A_p + A_m)
            diag[0] = 1.0; sup[0] = 0.0
            diag[-1] = 1.0; sub[-1] = 0.0
            rhs = omega_star.copy(); rhs[0] = 0.0; rhs[-1] = 0.0
            from ._gpu_kernels import solve_tridiag_columns_device
            return solve_tridiag_columns_device(self._tridiag_kernel_dev, sub, diag, sup, rhs)
        # CPU host numpy (byte-identical to the shipped path).
        omega_star = np.asarray(to_host(omega_star), float)
        cz = np.asarray(to_host(coef_z), float) * np.ones_like(omega_star)
        czp = 0.5 * (cz[1:-1] + cz[2:])                          # (nz-2, nx) j+1/2 face coefficient
        czm = 0.5 * (cz[:-2] + cz[1:-1])                         # j-1/2
        if self._stretched:
            z = self.z
            hp = (z[2:] - z[1:-1])[:, None]                      # (nz-2, 1)
            hm = (z[1:-1] - z[:-2])[:, None]
            dzc = (0.5 * (z[2:] - z[:-2]))[:, None]
        else:
            hp = hm = dzc = self.dz
        A_p = np.zeros_like(omega_star); A_m = np.zeros_like(omega_star)
        A_p[1:-1] = czp / (hp * dzc)                             # L_z F[j] = A_p(F[j+1]-F[j]) - A_m(F[j]-F[j-1])
        A_m[1:-1] = czm / (hm * dzc)
        sub = -dt * A_m                                          # coef of omega_new[j-1]
        sup = -dt * A_p                                          # coef of omega_new[j+1]
        diag = 1.0 + dt * (A_p + A_m)
        diag[0] = 1.0; sup[0] = 0.0                              # Dirichlet omega=0 at the walls (free-slip)
        diag[-1] = 1.0; sub[-1] = 0.0
        rhs = omega_star.copy(); rhs[0] = 0.0; rhs[-1] = 0.0
        return _solve_tridiag_columns(sub, diag, sup, rhs)

    def _implicit_vertical_advection(self, omega_star, dt, w):
        """Backward-Euler solve of the VERTICAL omega-advection limb: return ``omega_new`` solving
        ``(I + dt A_adv) omega_new = omega_star`` per column, with ``A_adv`` the FIRST-ORDER UPWIND
        operator ``w d/dz`` -- the SAME discrete operator ``_upwind_vertical`` applies (backward where
        w>0, forward where w<0, the same per-layer interval spacings) and Dirichlet ``omega=0`` at the
        free-slip walls (j=0, nz-1). Removing this (chimney-binding) advective limb from the explicit dt is
        the ~dz_min/|w| CFL relief of ADVECTION_DT_ROADMAP.md. The discretization is an M-matrix
        (positive diagonal, non-positive off-diagonals, diagonally dominant) -> UNCONDITIONALLY STABLE and
        MONOTONE (no spurious negatives / overshoot, so w need NOT be clipped -- clipping would break the
        div-free continuity ``_divfree_mass_flux`` relies on). ``w`` is the resolved vertical velocity
        (nz, nx).

        Upwind A_adv F[j] = w[j]/h_m (F[j]-F[j-1])  where w>0  (h_m = z[j]-z[j-1]);
                          = w[j]/h_p (F[j+1]-F[j])  where w<0  (h_p = z[j+1]-z[j]).
        So (I + dt A_adv): diag[j] = 1 + dt|w[j]|/h, sub[j] = -dt w[j]/h_m (w>0), sup[j] = +dt w[j]/h_p (w<0).

        DEVICE (Phase D): on device="gpu" the bands are assembled with the device array module and the
        per-column Thomas solve runs in the device RawKernel (``_tridiag_kernel_dev``, fp64), returning a
        device ``omega_new`` -- no host round-trip. On CPU it is the verbatim host numpy solve
        (byte-identical)."""
        if self._dyn_on_device:
            xp = self.xp
            omega_star = to_device(omega_star, xp).astype(xp.float64)
            w = to_device(w, xp).astype(xp.float64)
            nz = omega_star.shape[0]
            if self._stretched:
                z = xp.asarray(self.z, xp.float64)
                hm = xp.empty(nz, dtype=xp.float64); hp = xp.empty(nz, dtype=xp.float64)
                hm[1:] = z[1:] - z[:-1]; hm[0] = hm[1]
                hp[:-1] = z[1:] - z[:-1]; hp[-1] = hp[-2]
                hm = hm[:, None]; hp = hp[:, None]
            else:
                hm = hp = self.dz
            wpos = w > 0.0
            A_m = xp.where(wpos, w / hm, 0.0)
            A_p = xp.where(wpos, 0.0, -w / hp)
            sub = -dt * A_m
            sup = -dt * A_p
            diag = 1.0 + dt * (A_m + A_p)
            diag[0] = 1.0; sup[0] = 0.0
            diag[-1] = 1.0; sub[-1] = 0.0
            rhs = omega_star.copy(); rhs[0] = 0.0; rhs[-1] = 0.0
            from ._gpu_kernels import solve_tridiag_columns_device
            return solve_tridiag_columns_device(self._tridiag_kernel_dev, sub, diag, sup, rhs)
        # CPU host numpy (byte-identical).
        omega_star = np.asarray(to_host(omega_star), float)
        w = np.asarray(to_host(w), float)
        nz = omega_star.shape[0]
        if self._stretched:
            z = self.z
            hm = np.empty(nz); hp = np.empty(nz)
            hm[1:] = z[1:] - z[:-1]; hm[0] = hm[1]
            hp[:-1] = z[1:] - z[:-1]; hp[-1] = hp[-2]
            hm = hm[:, None]; hp = hp[:, None]
        else:
            hm = hp = self.dz
        wpos = w > 0.0
        A_m = np.where(wpos, w / hm, 0.0)              # coupling to j-1, active where w>0 (backward)
        A_p = np.where(wpos, 0.0, -w / hp)             # coupling to j+1, active where w<0 (forward); -w>=0
        sub = -dt * A_m                                # coef of omega_new[j-1]
        sup = -dt * A_p                                # coef of omega_new[j+1]
        diag = 1.0 + dt * (A_m + A_p)                  # A_m + A_p = |w|/h >= 0 -> diagonally dominant
        diag[0] = 1.0; sup[0] = 0.0                    # Dirichlet omega=0 at the walls (free-slip)
        diag[-1] = 1.0; sub[-1] = 0.0
        rhs = omega_star.copy(); rhs[0] = 0.0; rhs[-1] = 0.0
        return _solve_tridiag_columns(sub, diag, sup, rhs)

    def _advect_vertical_implicit(self, omega_star, dt, w):
        """Apply the backward-Euler vertical-advection limb over ``dt``, optionally SUB-CYCLING (option 2)
        the stiff chimney columns. With ``adv_subcycle_n>1`` each column whose advective Courant number
        ``|w|*dt/dz`` exceeds ``adv_subcycle_courant`` is advanced by n back-Euler sub-solves at dt/n (full
        dt elsewhere). The sub-cycle reduces the per-limb O(dt) back-Euler truncation where ``w/dz`` is
        extreme, at the cost of n cheap tridiag solves on the flagged columns only. ``adv_subcycle_n==1``
        (default) is the verbatim single solve -> byte-identical. The whole-array solve is vectorized; the
        flagged-column sub-cycle overwrites only those columns, so unflagged columns keep the single-dt
        result (each back-Euler solve is unconditionally stable + monotone, so the blend is well-posed)."""
        c = self.cfg
        omega_new = self._implicit_vertical_advection(omega_star, dt, w)     # single full-dt solve (all cols)
        n_sub = int(getattr(c, "adv_subcycle_n", 1))
        if n_sub <= 1:
            return omega_new                                                # byte-identical to the single solve
        xp = self.xp if self._dyn_on_device else np
        w_h = xp.asarray(w) if self._dyn_on_device else np.asarray(to_host(w), float)
        dz_face = self._dz_min if not self._stretched else float(np.min(np.diff(self.z)))
        courant_col = float(xp.max(xp.abs(w_h))) * dt / dz_face             # scalar max advective Courant
        if courant_col <= float(getattr(c, "adv_subcycle_courant", 1.0)):
            return omega_new
        courant = xp.max(xp.abs(w_h), axis=0) * dt / dz_face                # per-column max advective Courant
        stiff = courant > float(getattr(c, "adv_subcycle_courant", 1.0))     # (nx,) the chimney columns
        if not bool(xp.any(stiff)):
            return omega_new
        om = (to_device(omega_star, xp) if self._dyn_on_device
              else np.asarray(to_host(omega_star), float)).copy()
        dt_sub = dt / n_sub
        for _ in range(n_sub):
            om = self._implicit_vertical_advection(om, dt_sub, w_h)
        omega_new = omega_new.copy()
        omega_new[:, stiff] = om[:, stiff]                                   # sub-cycled result in stiff cols only
        return omega_new

    def _cfl_dt(self, u, w, nu_t, nu_v=None):
        """Adaptive explicit-Euler time step: the smallest of the configured ``dt_dyn`` cap and the
        advective (dx/|u|, dz/|w|) and diffusive (0.5/(nu_t (1/dx^2+1/dz^2))) CFL limits, times the
        ``cfl`` safety factor. Keeps the resolved-dynamics substep stable across the warm-edge buoyancy
        forcing and the full Rayleigh-number sweep without hand-tuning dt per case.

        ANISOTROPIC CLOSURE (issue #81): with ``nu_v`` given the diffusive limit uses the DIRECTIONAL split
        ``0.5/(nu_h/dx^2 + nu_v/dz^2)`` (per-point max), instead of the isotropic ``0.5/(nu_t(1/dx^2+1/dz^2))``
        -- this is where the ~50-300x dt gain comes from, because nu_v/dz^2 replaces the inflated nu_t/dz^2.
        ``nu_v=None`` keeps the isotropic limit (byte-identical).

        DEVICE (M3.1): the three CFL reductions are pulled to host scalars (``float(xp.max(...))``) -- a
        per-substep device->host sync. ``max`` is reduction-order-independent, so dt carries no EXTRA
        divergence beyond whatever the u/w/nu_t inputs already have: those match CPU<->GPU to fp64 round-off
        (FMA-contraction differences upstream), so dt is round-off-identical, not strictly bit-identical. A
        python float dt keeps the downstream tendency update device-clean."""
        c = self.cfg
        xp = get_xp(u)
        eps = 1e-30
        # The vertical CFL uses ``self._dz_min`` (the TIGHTEST interval) -- == self.dz on a uniform grid
        # (byte-identical), but the fine near-surface spacing on a stretched grid (BL_REFINEMENT_SPEC.md #C).
        # With ``vertical_implicit_diffusion`` the vertical DIFFUSIVE limb is solved implicitly, so its
        # ~1/dz_min^2 term is DROPPED from the diffusive limit (the relief); the vertical ADVECTIVE limit
        # dz_min/|w| below remains (implicit diffusion does not stabilize advection) -- BL_REFINEMENT_SPEC.md #d.
        # The OFF branch is the verbatim pre-P2 expression (byte-identical; _dz_min == self.dz on a uniform grid).
        # CAPTURE-READY device path (Phase E): the four ``float(xp.max(...))`` below are device->host syncs,
        # ILLEGAL during CUDA-graph capture (a synchronize on a capturing stream raises -- it aborts the
        # capture, it is not recorded). On ``capture_ready and _dyn_on_device`` keep dt a 0-d DEVICE scalar:
        # the ``xp.max`` reductions stay kernels in the captured chain and dt threads through the tendency /
        # thermo / advection limbs by ordinary broadcast (device_array * device_scalar). The Python builtin
        # ``min`` would itself sync (it compares the scalars host-side), so the cap/CFL minima are taken with
        # ``xp.minimum`` over device scalars. fp64 result is the SAME value as the host-float path (the
        # reductions are order-independent), so GPU-fp64 parity is unchanged -- only the dtype/locus of ``dt``
        # differs. OFF (and CPU/splu) is the verbatim shipped expression -> byte-identical (the 10-config SHA).
        device_dt = self.cfg.capture_ready and self._dyn_on_device
        if device_dt and self._x_stretched:
            # PER-CELL caps, device 0-d scalars (the stretched twin of the host branch below): every
            # reduction is an xp.max kernel; eps folds in as a python-float RATE operand (eps/_dx_min) --
            # no H2D, capture-clean, exactly the uniform device branch's discipline.
            inv_dxw = self._inv_dxw_row                        # device (1, nx) after placement
            inv_dz2 = 1.0 / self._dz_min**2
            if c.vertical_implicit_diffusion:
                diff_rate = xp.max(nu_t * inv_dxw**2)
            elif nu_v is None:
                diff_rate = xp.max(nu_t * (inv_dxw**2 + inv_dz2))
            else:
                diff_rate = xp.max(nu_t * inv_dxw**2 + nu_v * inv_dz2)
            dt = c.cfl / (xp.max(xp.abs(u) * inv_dxw) + eps / self._dx_min)
            if not _implicit_advection_scheme(c.advection_scheme):
                dt = xp.minimum(dt, c.cfl * self._dz_min / (xp.max(xp.abs(w)) + eps))
            dt = xp.minimum(dt, c.cfl * 0.5 / (diff_rate + eps))
            dt = xp.minimum(dt, c.dt_dyn)
            return dt
        if device_dt:
            inv_dx2 = 1.0 / self.dx**2
            inv_dz2 = 1.0 / self._dz_min**2
            if c.vertical_implicit_diffusion:
                diff_rate = (xp.max(nu_t) * inv_dx2 if nu_v is None
                             else xp.max(nu_t * inv_dx2))
            elif nu_v is None:
                diff_rate = xp.max(nu_t) * (inv_dx2 + inv_dz2)
            else:
                diff_rate = xp.max(nu_t * inv_dx2 + nu_v * inv_dz2)
            # Seed the chain with a DEVICE 0-d term (an ``xp.max`` reduction), then fold every Python-float
            # bound in as a SCALAR OPERAND to ``xp.minimum`` -- NOT via ``xp.asarray(float)``, which would be a
            # host->device transfer (ILLEGAL during capture). CuPy passes a python scalar as a kernel arg, no
            # H2D. ``xp.minimum`` is exact selection -> value-identical to the host ``min(...)`` (eager ON==OFF).
            dt = c.cfl * self.dx / (xp.max(xp.abs(u)) + eps)         # device 0-d scalar
            if not _implicit_advection_scheme(c.advection_scheme):
                dt = xp.minimum(dt, c.cfl * self._dz_min / (xp.max(xp.abs(w)) + eps))
            dt = xp.minimum(dt, c.cfl * 0.5 / (diff_rate + eps))
            dt = xp.minimum(dt, c.dt_dyn)                            # python-float cap as scalar operand (no H2D)
            return dt                                                 # 0-d device scalar (graph-clean)
        if self._x_stretched:
            # PER-CELL horizontal caps (stretched x, CPU-only path): the advective cap is the max over
            # cells of |u|/dx_local (the local TRUE cell width -- ``_inv_dxw_row``), and the diffusive
            # x-limb likewise uses the local width, so the fine axis cells bind the dt they actually
            # constrain instead of a single global scalar. ``eps`` folds in as a rate (eps/_dx_min) so the
            # degenerate all-zero-velocity case stays bounded exactly as on the uniform path.
            inv_dxw = self._inv_dxw_row                            # (1, nx) true-width reciprocals
            if c.vertical_implicit_diffusion:
                diff_rate = float(xp.max(nu_t * inv_dxw**2))
            elif nu_v is None:
                diff_rate = float(xp.max(nu_t * (inv_dxw**2 + 1.0 / self._dz_min**2)))
            else:
                diff_rate = float(xp.max(nu_t * inv_dxw**2 + nu_v / self._dz_min**2))
            caps = [c.dt_dyn,
                    c.cfl / (float(xp.max(xp.abs(u) * inv_dxw)) + eps / self._dx_min),
                    c.cfl * 0.5 / (diff_rate + eps)]
            if not _implicit_advection_scheme(c.advection_scheme):
                caps.append(c.cfl * self._dz_min / (float(xp.abs(w).max()) + eps))
            return min(caps)
        if c.vertical_implicit_diffusion:
            diff_rate = (float(xp.max(nu_t)) * (1.0 / self.dx**2) if nu_v is None
                         else float(xp.max(nu_t / self.dx**2)))
        elif nu_v is None:
            diff_rate = float(xp.max(nu_t)) * (1.0 / self.dx**2 + 1.0 / self._dz_min**2)
        else:
            diff_rate = float(xp.max(nu_t / self.dx**2 + nu_v / self._dz_min**2))   # per-point directional max
        # The vertical omega-ADVECTION CFL term ``dz_min/|w|`` is the ONLY binding cap on the converged
        # chimney world (ADVECTION_DT_ROADMAP.md). With ``advection_scheme="vertical_implicit"`` that limb
        # is solved by a backward-Euler upwind tridiagonal (``_implicit_vertical_advection``, unconditionally
        # stable), so it is DROPPED here (the relief). The HORIZONTAL-advection cap (dx/|u|) and the diffusion
        # cap stay, so dt lands at min(dt_dyn, dx/|u|, diffusion) ~ 60 s. OFF -> the verbatim expression
        # (byte-identical). NB only the VERTICAL term is relieved; horizontal omega advection stays explicit.
        caps = [c.dt_dyn,
                c.cfl * self.dx / (float(xp.abs(u).max()) + eps),
                c.cfl * 0.5 / (diff_rate + eps)]
        if not _implicit_advection_scheme(c.advection_scheme):
            caps.append(c.cfl * self._dz_min / (float(xp.abs(w).max()) + eps))
        return min(caps)

    def _explicit_vorticity_increment(self, omega, u, w, nu_t, nu_v, coef_z):
        """The per-unit-time EXPLICIT vorticity tendency the implicit substep front-loads: the full
        ``vorticity_tendency`` (buoyancy torque + advection + stretching + horizontal eddy + the vertical
        eddy term) MINUS the vertical eddy (moved implicit) PLUS the vertical-advection add-back (moved
        implicit, when the advection limb is on), all evaluated at the GIVEN ``omega`` with frozen
        (u, w, nu_t, nu_v, b). Returns a tendency [1/s²] (NOT yet multiplied by dt). Surface drag is NOT
        folded here -- the caller applies it once (drag is a linear sink that need not be symmetrized; it is
        already unconditionally stable). Factored out so the E½-D-E½ Strang variant can re-evaluate E on the
        post-implicit-block state (symmetrizing the explicit forcing, which the plain Strang leaves
        front-loaded -- the dominant O(dt) splitting term per archive/_sl_strang_RESULTS.md)."""
        tend = vorticity_tendency(omega, self.b, u, w, self.dlnrho_dz, nu_t, self._dx_op, self._dz_op, nu_t_z=nu_v)
        tend = tend - _vertical_eddy_term(omega, coef_z, self._dz_op)        # remove the implicit vertical eddy
        if _implicit_advection_scheme(self.cfg.advection_scheme):
            tend = tend + _upwind_vertical(omega, w, self._dz_op)           # add back the implicit vertical adv
        return tend

    def _substep_update_implicit(self, u, w, nu_t, nu_v):
        """P2 (BL_REFINEMENT_SPEC.md #d): the omega update with the VERTICAL eddy-diffusion limb solved
        IMPLICITLY (backward Euler), so the ~min(dz) diffusive CFL no longer binds the explicit dt. The
        explicit tendency carries everything EXCEPT the vertical eddy (advection, buoyancy torque, vortex
        stretching, the HORIZONTAL eddy, surface drag); the vertical eddy is then inverted by
        ``_implicit_vertical_diffusion``. The dissipative-heating ENERGY closure stays exact: the operator
        KE removal is recomputed from the ACTUAL applied diffusion tendency (horizontal eddy + the implicit
        vertical increment as a rate + drag), passed to ``_operator_ke_removal_col``."""
        c = self.cfg
        dt = self._cfl_dt(u, w, nu_t, nu_v)
        coef_z = nu_v if nu_v is not None else nu_t                # vertical eddy viscosity for L_z
        tend_full = vorticity_tendency(self.omega, self.b, u, w, self.dlnrho_dz, nu_t, self._dx_op,
                                       self._dz_op, nu_t_z=nu_v)
        eddy_v = _vertical_eddy_term(self.omega, coef_z, self._dz_op)   # the vertical limb to move implicit
        # IMPLICIT VERTICAL ADVECTION (ADVECTION_DT_ROADMAP.md): if ON, ALSO remove the vertical omega-
        # advection limb from the explicit tendency (tend_full carries -_upwind = -(u dw/dx + w dw/dz), so
        # adding back +_upwind_vertical cancels its w-term) and invert it by a SECOND backward-Euler
        # tridiagonal solve below. OFF -> adv_v==0, the omega_adv solve is skipped (omega_adv is omega_star)
        # -> byte-identical to the diffusion-only path.
        implicit_adv = _implicit_advection_scheme(c.advection_scheme)
        strang = (c.advection_scheme in _STRANG_SCHEMES)         # A½-D-A½ implicit-limb symmetry
        strang_e = (c.advection_scheme == "vertical_implicit_strang_e")   # ALSO symmetrize explicit forcing E
        adv_v = _upwind_vertical(self.omega, w, self._dz_op) if implicit_adv else 0.0
        # The explicit forcing E (per dt). For E-symmetrized Strang we apply only the FIRST half here and the
        # SECOND half (E re-evaluated on the post-implicit state) after the implicit block; otherwise the full E.
        e_frac = 0.5 if strang_e else 1.0
        omega_star = self.omega + (e_frac * dt) * (tend_full - eddy_v + adv_v)   # explicit: all but the vertical eddy (& adv if ON)
        if c.surface_drag_rate > 0.0:                            # drag: linear sink, applied once (not symmetrized)
            omega_star[1, :] = omega_star[1, :] - dt * c.surface_drag_rate * self.omega[1, :]
        if c.surface_drag_bulk:                                   # honest bulk surface friction (curl of tau)
            omega_star = omega_star + dt * self._surface_drag_curl(u)
        # IMPLICIT-ADVECTION COMPOSITION of the two backward-Euler limbs:
        #   "vertical_implicit"          = LIE-TROTTER: advection(dt) THEN diffusion(dt). O(dt) splitting error
        #                                  (the "+5.5K chimney shift" once blamed on it was FALSIFIED -- the
        #                                  dt=60 carrier is the column thermo cadence, fixed by thermo_n_sub,
        #                                  PR #203; _sl_FINAL_VERDICT.md supersedes the archive diagnosis).
        #   "vertical_implicit_strang"   = STRANG A½-D-A½ of the two IMPLICIT limbs. O(dt²) inter-limb split,
        #                                  but FOUND not to fix the shift (commutator negligible; (ii) dominates).
        #   "vertical_implicit_strang_e" = E½-[A½-D-A½]-E½: ALSO symmetrize the explicit forcing E (the
        #                                  dominant O(dt) split term, error (ii)) by re-evaluating E on the
        #                                  post-implicit state for the second half-step (frozen u,w,b).
        # Either way the ADVECTION limb is conservative transport (NOT booked as dissipative heat); the energy
        # closure books ONLY the single full-dt DIFFUSION increment (omega_postD - omega_preD)/dt, regardless
        # of composition. ``omega_preD`` is the state the diffusion solve sees; ``omega_postD`` its output.
        # OFF -> implicit_adv False -> omega_preD IS omega_star, no advection solve -> byte-identical.
        if not implicit_adv:
            omega_preD = omega_star
            omega_postD = self._implicit_vertical_diffusion(omega_preD, dt, coef_z)
            omega_block = omega_postD
        elif strang:
            omega_a1 = self._advect_vertical_implicit(omega_star, 0.5 * dt, w)   # advection HALF step
            omega_preD = omega_a1
            omega_postD = self._implicit_vertical_diffusion(omega_preD, dt, coef_z)   # diffusion FULL step
            omega_block = self._advect_vertical_implicit(omega_postD, 0.5 * dt, w)     # advection HALF step
        else:                                                    # Lie-Trotter (vertical_implicit)
            omega_preD = self._advect_vertical_implicit(omega_star, dt, w)
            omega_postD = self._implicit_vertical_diffusion(omega_preD, dt, coef_z)
            omega_block = omega_postD
        # E-SYMMETRY (strang_e): the explicit forcing's SECOND half-step, E re-evaluated on the post-implicit
        # ``omega_block`` (frozen u,w,b -- no second Poisson solve, an O(dt²)-faithful frozen-velocity Strang).
        # The advection add-back / vertical-eddy removal are recomputed consistently in the helper. omega_new
        # is the fully-updated vorticity; the energy closure still books only the (omega_postD-omega_preD)
        # DIFFUSION increment, so the second explicit half is NOT mis-booked as heat.
        if strang_e:
            e2 = self._explicit_vorticity_increment(omega_block, u, w, nu_t, nu_v, coef_z)
            omega_new = omega_block + (0.5 * dt) * e2
        else:
            omega_new = omega_block
        # energy closure: the ACTUAL applied DIFFUSION tendency = horizontal eddy (explicit) + the implicit
        # vertical DIFFUSION increment as a rate + drag -> the operator KE removal the dissipative heating
        # returns. The advection increment(s) (conservative transport) are NOT booked as dissipative heat: the
        # diffusion increment is (omega_postD - omega_preD)/dt (== (omega_new - omega_star)/dt when OFF, since
        # omega_preD IS omega_star and omega_postD IS omega_new -> byte-identical). NB for Strang ``omega_new``
        # carries the SECOND half-advection past omega_postD; the closure must use omega_postD/omega_preD (the
        # diffusion solve's actual in/out), NOT omega_new, or the second advection leaks into the heat book.
        if c.dissipative_heating and self.couple_heat and c.dissipation_form == "operator":
            eddy_full = (_div_grad(self.omega, nu_t, self._dx_op, self._dz_op) if nu_v is None
                         else _div_grad_aniso(self.omega, nu_t, nu_v, self._dx_op, self._dz_op))
            # Energy book uses the DIFFUSION increment (omega_postD - omega_preD)/dt, NOT (omega_new - omega_star):
            # with the implicit-ω advection add-back, omega_new carries the moved/second advection limb past
            # omega_postD, which is conservative transport and must NOT leak into the heat book (see the comment
            # block above -- "the closure must use omega_postD/omega_preD ... NOT omega_new").
            if self._dyn_on_device:
                # Device path (Phase C): build T_diff ON DEVICE (omega_postD is device); _operator_ke_removal_col
                # solves it through the device Poisson and pulls only the (nx,) D_col. No host round-trip.
                # (GPU+splu keeps the state HOST -> falls to the host branch, mixing nothing.)
                T_diff = (eddy_full - eddy_v + (omega_postD - omega_preD) / dt)
                if c.surface_drag_rate > 0.0:
                    T_diff[1, :] = T_diff[1, :] - c.surface_drag_rate * self.omega[1, :]
                if c.surface_drag_bulk:
                    T_diff = T_diff + self._surface_drag_curl(u)
            else:
                omega_preD_h = np.asarray(to_host(omega_preD), float)
                omega_postD_h = np.asarray(to_host(omega_postD), float)
                T_diff = (np.asarray(to_host(eddy_full), float) - np.asarray(to_host(eddy_v), float)
                          + (omega_postD_h - omega_preD_h) / dt)
                if c.surface_drag_rate > 0.0:
                    T_diff[1, :] = T_diff[1, :] - c.surface_drag_rate * np.asarray(to_host(self.omega), float)[1, :]
                if c.surface_drag_bulk:                          # honest bulk drag: fold its curl-of-stress in too
                    T_diff = T_diff + np.asarray(to_host(self._surface_drag_curl(u)), float)
            self._diss_Dcol = self._operator_ke_removal_col(u, w, nu_t, nu_v, T_diff=T_diff)
        else:
            self._diss_Dcol = None
        # _implicit_vertical_diffusion is a backward-Euler tridiagonal solve done fp64 (the second linear
        # solve in the dynamics path -- on host on CPU, on the device RawKernel on GPU/Phase C); cast the
        # result back to the working dtype. ``to_device`` is a no-op when omega_new is already device-resident.
        if c.capture_ready:                                      # in-place into the fixed omega buffer
            self.omega[...] = self._cast(omega_new)
        else:
            self.omega = to_device(self._cast(omega_new), self.xp)
        self.omega[0, :] = 0.0; self.omega[-1, :] = 0.0          # free-slip top/bottom (#5)
        self.omega[:, 0] = 0.0; self.omega[:, -1] = 0.0          # zero across the equator/pole walls
        self._last_dt_dyn = dt
        return u, w, dt

    def _dynamics_substep(self):
        """One explicit anelastic omega-psi substep against the CURRENT buoyancy field ``self.b``:
        invert psi from omega (Poisson, #0), recover (u, w), form the Smagorinsky eddy viscosity (#4;
        ANISOTROPIC nu_h/nu_v if ``closure_anisotropic``, #81), CFL-limit dt, advance omega by the buoyancy
        torque + advection + stretching + eddy diffusion (``vorticity_tendency``) + an optional surface
        Rayleigh drag (#85), then apply the free-slip wall BC omega=0 (#5; psi=0 is enforced by the
        Poisson). Stashes the resolved surface wind (for the #85 resolved-flux closure). With
        ``vertical_implicit_diffusion`` (P2) the update from here delegates to ``_substep_update_implicit``
        (backward-Euler vertical eddy limb; the Strang advection schemes live there too). Returns (u, w, dt).

        DEVICE (Phase C): on device="gpu" the anelastic Poisson runs ON DEVICE (the separable DST x
        tridiagonal ``_lu_dyn_dev``), so omega is NOT copied to host -- the M3.1 per-substep round-trip is
        removed. On CPU the variable-coefficient splu/DST Poisson stays host numpy (omega copied host<->device
        is a no-op there). Everything else (velocities, Smagorinsky, the vorticity tendency, the wall BCs)
        stays on the configured array module."""
        c = self.cfg
        # POISSON STAYS fp64 (the ill-conditioned solve; forward error ~ kappa(op)*eps grows with nx/nz):
        # cast the RHS UP to fp64 (an fp32 RHS would solve in fp32 and amplify the kappa(op)*eps error), then
        # cast psi back to the working dtype. On an fp64 core the upcast is a no-op (byte-identical).
        if self._dyn_on_device:
            # Device-resident solve: feed the fp64 device omega into the device DST solver, no host trip.
            # ``solve`` reads ``b[1:-1,1:-1].copy()`` and writes a fresh psi, so it never aliases self.omega.
            omega_f64 = self.omega.astype(self.xp.float64)        # fp64 RHS (no-op copy on an fp64 core)
            psi_dev = self._lu_dyn_dev.solve(omega_f64)           # (nz, nx) device, fp64
            if c.capture_ready:
                self.psi[...] = self._cast(psi_dev)               # in-place, cast back to the working dtype
            else:
                self.psi = self._cast(psi_dev)                    # rebind, cast back to the working dtype
        else:
            # CPU: the host splu/DST solve (``to_host`` is np.asarray; ``ravel().astype(f64)`` is the upcast).
            omega_host = to_host(self.omega)
            psi_host = self._lu_dyn.solve(omega_host.ravel().astype(np.float64)).reshape(c.nz, c.nx)
            if c.capture_ready:                                   # in-place into the fixed psi buffer (the
                self.psi[...] = self._cast(psi_host)              #   host->device copy targets a stable address)
            else:
                self.psi = to_device(self._cast(psi_host), self.xp)  # host -> device, back to the working dtype
        u, w = velocities_from_psi(self.psi, self._beta, self._dx_op, self._dz_op)
        # ANISOTROPIC closure (#81): nu_v=None on the isotropic path (byte-identical), else the directional
        # (nu_h, nu_v) split feeds the eddy operator + the diffusive CFL + the operator KE-removal alike.
        if c.closure_anisotropic:
            nu_t, nu_v = smagorinsky_viscosity_aniso(u, w, self._dx_op, self._dz_op, c.C_s, c.nu_b)
        else:
            nu_t, nu_v = smagorinsky_viscosity(u, w, self._dx_op, self._dz_op, c.C_s, c.nu_b), None
        if c.nu_h_floor:
            nu_t = nu_t + c.nu_h_floor                  # grid-independent horizontal eddy-viscosity floor (#equator width)
        # stash the resolved surface cross-track wind u[1,:] (j=1, the diagnostics() convention) for the
        # resolved-wind surface fluxes (#85 item 1). Phase D: keep it on the dynamics array module (device on
        # GPU) -- the thermo step now reads it device-resident, so the M3.1 per-substep host pull is gone.
        self._u_surf = u[1, :]
        # M3.4b Task 2: stash the operator KE removal NOW, while self.omega is the PRE-update vorticity that is
        # consistent with (u, w) (== L(self.psi)) and equals the diffusion div(nu_t grad omega) this substep
        # applies below -- so the dissipative-heating deposit returns exactly the KE removed (not a one-tendency-
        # stale omega). Only when the term will actually be applied (couple_heat + operator form). Else None
        # (the tendency falls back to recomputing for standalone/test calls).
        if c.vertical_implicit_diffusion:
            return self._substep_update_implicit(u, w, nu_t, nu_v)   # P2: vertical eddy limb backward-Euler
        self._diss_Dcol = (self._operator_ke_removal_col(u, w, nu_t, nu_v)
                           if (c.dissipative_heating and self.couple_heat
                               and c.dissipation_form == "operator") else None)
        dt = self._cfl_dt(u, w, nu_t, nu_v)
        # IMPLICIT VERTICAL ADVECTION (ADVECTION_DT_ROADMAP.md) on the EXPLICIT-diffusion path: if ON, remove
        # the vertical omega-advection limb from the explicit tendency (add back +_upwind_vertical to cancel
        # the -w dw/dz inside vorticity_tendency's -_upwind), accumulate the explicit increment, then invert
        # the moved limb by a backward-Euler upwind tridiagonal solve. OFF -> adv_v==0, no solve (byte-identical).
        # NB the Strang composition (vertical_implicit_strang) is defined ONLY on the implicit-vertical-
        # diffusion path (_substep_update_implicit), where it can interleave the two backward-Euler limbs.
        # On THIS explicit-diffusion path there is no implicit-diffusion limb to interleave, so reject it
        # rather than silently relieving the CFL (via _cfl_dt) while running plain Lie-Trotter advection.
        if c.advection_scheme in _STRANG_SCHEMES:
            raise ValueError(f"advection_scheme='{c.advection_scheme}' requires "
                             "vertical_implicit_diffusion=True (the Strang split interleaves the two "
                             "implicit limbs; there is none on the explicit-diffusion path).")
        implicit_adv = (c.advection_scheme == "vertical_implicit")
        adv_v = _upwind_vertical(self.omega, w, self._dz_op) if implicit_adv else 0.0
        domega = dt * (vorticity_tendency(
            self.omega, self.b, u, w, self.dlnrho_dz, nu_t, self._dx_op, self._dz_op, nu_t_z=nu_v) + adv_v)
        # surface drag (#85 item 2): a linear Rayleigh vorticity sink at the lowest interior level (j=1),
        # the momentum sink the free-slip bottom lacks. 0 = off (byte-identical). Stable for any dt (the
        # damping factor dt*rate << 1 at the CFL dt). BOTH drags' removed KE IS returned as heat on the
        # default dissipative-heating operator path (each folded into _operator_ke_removal_col's inversion --
        # the _diss_Dcol stash above already includes them), so the energy budget stays closed with drag ON.
        if c.surface_drag_rate > 0.0:
            domega[1, :] = domega[1, :] - dt * c.surface_drag_rate * self.omega[1, :]
        if c.surface_drag_bulk:                                  # honest bulk surface friction (curl of tau)
            domega = domega + dt * self._surface_drag_curl(u)
        self.omega = self._accumulate(self.omega, domega, out=self._obuf(self.omega))  # in-place capture-ready
        if implicit_adv:                                         # invert the moved vertical-advection limb
            self.omega = to_device(self._cast(
                self._implicit_vertical_advection(self.omega, dt, w)), self.xp)
        self.omega[0, :] = 0.0; self.omega[-1, :] = 0.0          # free-slip top/bottom (#5)
        self.omega[:, 0] = 0.0; self.omega[:, -1] = 0.0          # zero across the equator/pole walls
        self._last_dt_dyn = dt
        return u, w, dt

    def seed_warm_edge_buoyancy(self, dT_edge=40.0, h_d=4000.0):
        """Set ``self.b`` to a FIXED warm-edge buoyancy field for the M2b.2 dynamics validation -- a
        stand-in for the column buoyancy. A horizontal temperature contrast ``dT_edge`` [K] (warm at the
        equator x small, cool poleward) decaying with height over ``h_d`` [m]:  b = g (dT_edge *
        shape(x)) / T_ref(0) * exp(-z/h_d), with shape(x) the normalized day-mean-flux profile (1 at the
        equator -> 0 at the pole). Its poleward-decreasing horizontal gradient torques exactly one
        overturning cell with an equatorward surface return (Rossby->inf + continuity). HOLDS the
        buoyancy fixed (sets ``couple_thermo=False``) so the prognostic column thermo (M2b.3) does not
        overwrite it -- this is the pure-dynamics validation entry point. Returns the field."""
        xp = get_xp(self.S)                                                       # device-generic (Phase D: S/z placed on device)
        shape = (self.S - self.S.min()) / (self.S.max() - self.S.min() + 1e-30)   # (nx,) 1@eq -> 0@pole
        decay = xp.exp(-self.z / h_d)                                             # (nz,)
        b = (C.G_ACCEL * dT_edge / float(self.T_ref[0])) * decay[:, None] * shape[None, :]
        self.b = self.xp.asarray(self._cast(b))   # working dtype, then onto the configured device (M3.1)
        self.couple_thermo = False              # hold b: validate the resolved dynamics against a fixed forcing
        return self.b

    def _device_thermo_precision_guard(self):
        """PHASE D device-guard (load-bearing): on the GPU coupled path, make a forgotten/mismatched thermo-
        precision configuration fail LOUDLY, never silently fall back to fp64. A forgotten fp32 gate on GPU
        means the column transcendentals (qsat/esat/dqsat_dT/radiation) run fp64 -- and GA102 (the RTX 3080)
        throttles fp64 transcendentals 1:64, so a silent fp64 thermo would eat the ~5.7x thermo win the whole
        port exists to capture. The asymmetry matters (and is the design intent):

          * fp32 INTENDED (``cfg.thermo_fp32``) but the gate cannot engage -> ASSERT (fatal). The gate is
            ``th.set_thermo_dtype``; verify it is the wired mechanism and that engaging it actually flips the
            module working dtype to float32 (catches a thermo build where ``set_thermo_dtype`` is a no-op /
            the wrong module is imported -> the fp32 path would silently run fp64).
          * fp64 on device (the default; the fp64 parity test legitimately runs here) -> WARN ONCE (non-fatal)
            that fp64 thermo on the GPU forgoes the fp32 transcendental speedup -- so a user who MEANT to run
            fp32 sees the message, but the fp64 machine-precision parity run is not killed.

        Also guards the dynamics seam: a GPU coupled run with ``poisson_method != 'dst'`` keeps the dynamics'
        host round-trip (not device-resident), which silently defeats the coupled-on-device goal -> warn once."""
        if self.cfg.thermo_fp32:
            # Verify the gate is real: flipping th.set_thermo_dtype(fp32) must actually set the module dtype to
            # float32 (then restore). If it doesn't, the fp32 thermo path would silently run fp64 -- fatal.
            prev = th.set_thermo_dtype(np.float32)
            engaged = (th._FLOAT is np.float32)
            th.set_thermo_dtype(prev)
            assert engaged, (
                "device='gpu' + thermo_fp32=True but th.set_thermo_dtype(np.float32) did NOT engage fp32 "
                "(th._FLOAT is not float32) -- the fp32 thermo gate is broken, so the column transcendentals "
                "would SILENTLY run fp64 (the 64x GA102 throttle the port exists to avoid). Fix the thermo "
                "dtype gate before running the GPU fp32 path.")
        elif not getattr(self, "_warned_gpu_fp64_thermo", False):
            import warnings
            warnings.warn(
                "device='gpu' coupled step running with FP64 thermo arithmetic (thermo_fp32=False). GA102 "
                "throttles fp64 transcendentals 1:64, so the column physics forgoes the ~5.7x fp32 thermo "
                "speedup. Set StageADConfig(thermo_fp32=True) to engage the gate-driven fp32 arithmetic "
                "(prognostic state stays fp64 via the increment-reconcile accumulator).", RuntimeWarning,
                stacklevel=3)
            self._warned_gpu_fp64_thermo = True
        if self.run_dynamics and not self._dyn_on_device and not getattr(self, "_warned_gpu_splu_coupled", False):
            import warnings
            warnings.warn(
                "device='gpu' coupled step with poisson_method != 'dst': the dynamics linear solves keep the "
                "M3.1 host round-trip (omega->host->splu->psi->device), so the coupled step is NOT fully "
                "device-resident. Set poisson_method='dst' for the Phase-C device dynamics (and Phase-E "
                "capture-readiness).", RuntimeWarning, stacklevel=3)
            self._warned_gpu_splu_coupled = True

    # ------------------------------------------------------------------ one step (M2b.2 dynamics + M2b.3 thermo)
    def step(self):
        """Advance one time step, operator-split into the resolved dynamics and the column thermodynamics
        (each gated by a switch so the tests can isolate the limbs; see ``__init__``):

          1. DYNAMICS (M2b.2, if ``run_dynamics``): one CFL-adaptive anelastic omega-psi substep against
             the current buoyancy ``self.b`` (``_dynamics_substep``), returning (u, w, dt_dyn).
          2. THERMO (M2b.3/4, if ``couple_thermo``): the column radiation + surface energy balance +
             prognostic moisture + convective adjustment (``_thermo_substep``); then, if the flow is live,
             advect q by (u, w) (``_advect_moisture``) -- plus, under ``prognostic_condensate``, the cloud
             condensate q_c (``_advect_condensate``) and the BL stratocumulus deck (``bl_stratocumulus``)
             -- and, if ``couple_heat``, advect the column POTENTIAL TEMPERATURE by (u, w)
             (``_advect_heat``, the inter-column HEAT transport that closes Issue A; M2b.6) plus the
             ``dissipative_heating`` KE-return; then refresh the buoyancy from the POST-advection T
             (``_update_buoyancy``) for the next dynamics substep.

        TIME STEP: with the dynamics live the whole step rides the CFL-limited ``dt_dyn`` (one consistent
        clock for advection and the slow column physics). The column thermo may SUB-CYCLE within the step
        at dt_dyn/n (``thermo_n_sub`` fixed n -- 30 on the canonical nsub30 world -- or the
        ``thermo_dt_min`` auto-split), with radiation computed on the FIRST sub-step and reused by the
        rest.
        With the dynamics OFF (the psi==0 RCE limit) there is no CFL, so the column relaxes on the large
        thermodynamic ``dt`` -- fast convergence to local RCE, the M2b.3 anchor case.

        SCOPE: with ``couple_heat`` (the M2b.6 default) the resolved flow transports BOTH moisture and heat
        between columns, so the cell can move T(theta) OFF the per-column-RCE diffusive lineage (the
        Issue-A test in one fused domain). Set ``couple_heat=False`` to recover the M2b.3-5 limit where the
        flow carries only moisture and each column sits at its own local RCE (T(theta) == lineage). T_surf
        is a prognostic OUTPUT when ``couple_surface`` (M2b.4), a prescribed boundary otherwise.

        DEVICE (Phase D): with device="gpu" the WHOLE coupled step runs device-resident -- the dynamics
        solves on-device (Phase C, gated on poisson_method='dst') AND the column physics (radiation, surface
        energy balance, prognostic microphysics, convective adjustment, the grid reconciliation, the
        production cloud stack) are array-module-generic, so a coupled GPU run no longer raises. The device
        path requires ``poisson_method='dst'`` for the dynamics to be device-resident (a GPU+splu coupled run
        would keep the dynamics' host round-trip; a loud guard catches it). fp32 thermo arithmetic is engaged
        by the ``thermo_fp32`` gate (NOT by input dtype); a FORGOTTEN gate on GPU would silently run fp64
        thermo = the 64x GA102 transcendental throttle the whole port exists to avoid, so the device-guard
        below makes a mismatched/forgotten gate fail LOUDLY (fp32-intended-but-not-engaged -> assert) or warn
        once (fp64-on-device -> the throttle warning), never a silent fp64 fallthrough."""
        if self._on_device and self.couple_thermo:
            self._device_thermo_precision_guard()
        if (self.cfg.equator_norain_cells > 0 and not self.cfg.prognostic_condensate
                and not getattr(self, "_warned_norain_advect", False)):
            # self-protecting guard: equator_norain_cells retains the would-be rain as q_c, but q_c only
            # transports poleward via _advect_condensate, which is gated on prognostic_condensate. Without it the
            # retained water just sits/re-evaporates locally and the "forced into the poleward flow" mechanism
            # silently does not apply. (The suppress drivers set COSMO_PROGNOSTIC_CONDENSATE=1, so this is a guard.)
            import warnings
            warnings.warn("equator_norain_cells>0 but prognostic_condensate is OFF: retained q_c will not advect "
                          "poleward (no _advect_condensate) -- the suppression-redistribution mechanism is inert.",
                          RuntimeWarning, stacklevel=2)
            self._warned_norain_advect = True
        if self.cfg.diurnal_sun:
            self._advance_diurnal_sun()      # hold THIS step's instantaneous sun (start-of-step phase; 4.1)
        u = w = None
        if self.run_dynamics:
            u, w, dt = self._dynamics_substep()
        else:
            dt = self.cfg.dt
        if self.couple_thermo:
            n_thermo_sub = self.cfg.thermo_n_sub
            if n_thermo_sub <= 1 and self.cfg.thermo_dt_min > 0.0:
                dt_f = float(dt) if not isinstance(dt, (int, float)) else dt
                if dt_f > self.cfg.thermo_dt_min:
                    n_thermo_sub = int(np.ceil(dt_f / self.cfg.thermo_dt_min))
            if n_thermo_sub > 1:
                dt_sub = dt / n_thermo_sub
                self._thermo_substep(dt_sub)
                self._skip_rad_in_thermo_sub = True
                try:
                    for _ in range(n_thermo_sub - 1):
                        self._thermo_substep(dt_sub)
                finally:
                    self._skip_rad_in_thermo_sub = False
            else:
                self._thermo_substep(dt)
            if self.run_dynamics:
                # ROBUSTNESS GATE (review caveat, _sl_FINAL_VERDICT.md): only when the implicit-ω scheme has
                # relieved the vertical CFL -- the explicit default NEVER enters here (byte-identical, no new
                # work). Warns/raises if the realized tracer Courant exceeds 1 (beyond it the advection
                # limbs' per-step q/q_c positivity clamp injects mass, counted in flush_clamp_stats).
                if (_implicit_advection_scheme(self.cfg.advection_scheme)
                        and not self.cfg.capture_ready):
                    self._tracer_courant_guard(u, dt)
                # Build the divergence-free mass flux ONCE per substep and share it across the moisture/heat/
                # condensate limbs -- they all advect with the SAME u and z_p, so the three _divfree_mass_flux
                # builds were bit-identical (asserted in the advect sub-profiler). Removes 2 of 3 redundant
                # interp+flux builds; bit-identical (the shared flux is exactly what each limb recomputed).
                flux = self._resolved_mass_flux(u, None)
                self._advect_moisture(u, dt, flux=flux)   # resolved-flow moisture transport (x AND z; M2b.3/5)
                if self.cfg.prognostic_condensate:
                    self._advect_condensate(u, dt, flux=flux)   # resolved-flow CONDENSATE transport (Chunk 1; OFF: skipped)
                    if self.cfg.prognostic_rain:
                        self._advect_rain(u, dt, flux=flux)   # resolved-flow RAIN transport (stage 7; OFF: skipped)
                    if self.cfg.bl_stratocumulus:
                        self._bl_stratocumulus(w, dt)   # BL/stratocumulus deck under the subsidence inversion (Chunk 4)
                if self.couple_heat:
                    self._advect_heat(u, dt, flux=flux)   # resolved-flow HEAT transport (x AND z; M2b.6, Issue A)
                    if self.cfg.dissipative_heating:  # M3.4b Task 2: return the eddy-removed KE as heat. GATED
                        # on couple_heat -- it is the CONJUGATE return of _advect_heat's APE->KE cooling, so
                        # without that cooling sink it would be a spurious energy source.
                        self.T_atm = self._accumulate(self.T_atm,
                                                      dt * self._dissipative_heating_tendency(u, w),
                                                      out=self._obuf(self.T_atm))   # in-place capture-ready
                self._update_buoyancy()           # refresh b from the POST-advection T for the next substep
        # DIURNAL SUN end-of-step advances (findings 1 + 2 of the PR #301 review): ALL in-place,
        # device-generic ops, so the captured graph replays them and the phase/keying state advances by
        # the REALIZED dt (dt may be a 0-d device scalar here -- never float()ed).
        #   * the device diurnal CLOCK += dt -- the next step's gather reads start-of-step true time;
        #   * the slow-wetness EMAs (T_surf and, on budget runs, the wet-tile demand E_w) relax toward
        #     the current state with weight dt/tau (linearized; dt << tau) -- the TIME-MEAN keying that
        #     stops the classifier ramp re-wetting the flats every night. Skipped under the freeze
        #     variant (the frozen field never moves).
        if self.cfg.diurnal_sun:
            xpd = get_xp(self.T_surf)
            if self._diurnal_time_dev is not None:
                xpd.add(self._diurnal_time_dev, dt, out=self._diurnal_time_dev)
            if not self.cfg.diurnal_wetness_freeze:
                a = dt / self.cfg.diurnal_wetness_tau
                xpd.add(self._T_surf_slow, a * (self.T_surf - self._T_surf_slow), out=self._T_surf_slow)
                if self.cfg.budget_wet_fraction and getattr(self, "_E_w_cache", None) is not None:
                    if self._E_w_slow is None or self._E_w_slow.shape != self._E_w_cache.shape:
                        self._E_w_slow = xpd.asarray(self._E_w_cache, dtype=xpd.float64)   # seed (eager, pre-capture)
                    else:
                        xpd.add(self._E_w_slow, a * (self._E_w_cache - self._E_w_slow), out=self._E_w_slow)
            if self.cfg.salt_ccn and self.cfg.salt_ccn_dynamic and u is not None:
                # #308 item 2: the DE-ALIASED dynamic-salt emission wind key -- an EMA of |u_surf|^3
                # (the Owen-cubed moment the emission actually integrates; EMA-ing u and cubing later
                # would undercorrect through the convexity). Same in-place/capture semantics as the
                # wetness EMAs above; the refresh reads cbrt of this instead of tonight's snapshot.
                # The knob's "days" are DIURNAL CYCLES (its whole point is spanning the cycle), so
                # the conversion rides the clock's T_day, not a seconds literal (PR #312 review nit).
                xpu = get_xp(u)
                u3 = xpu.abs(u[1, :]) ** 3
                if self._salt_u3_slow is None or self._salt_u3_slow.shape != u3.shape:
                    self._salt_u3_slow = xpu.asarray(u3, dtype=xpu.float64)   # seed (eager, pre-capture)
                else:
                    a3 = dt / (self.cfg.salt_ccn_wind_tau_days * g.CLOCK.T_day)
                    xpu.add(self._salt_u3_slow, a3 * (u3 - self._salt_u3_slow), out=self._salt_u3_slow)
        # EQ-1 climatology EMAs (clim_accum): per-column true-time means of the offline eq-cycle's forcing
        # fields, so equilibrate_on_resume reads a matured climatology from the checkpoint instead of
        # re-integrating the core in eager mode (~hours/resume; ROADMAP GPU-14). DIAGNOSTIC ONLY -- these
        # buffers never feed physics, so ON vs OFF is byte-identical in every prognostic field. Same
        # in-place, capture-legal EMA idiom as the _salt_u3_slow / _T_surf_slow blocks above (dt may be a
        # 0-d device scalar -- never float()ed). Independent of diurnal_sun; tau in DIURNAL CYCLES (->
        # CLOCK.T_day), matching the hook's n_true_days window. The wind field may be the scalar cfg.wind
        # (no resolved wind); the seed broadcasts it to (nx,) and the update broadcasts in place.
        if self.cfg.clim_accum:
            xpc = get_xp(self.T_surf)
            if self._clim_tsurf_slow is None:            # first (eager, pre-capture) seed -> start the maturity
                self._clim_seed_days = float(self.true_sim_days())   # clock (host float; never under capture)
            a_c = dt / (self.cfg.clim_ema_tau_days * g.CLOCK.T_day)
            _clim_fields = [("_clim_tsurf_slow", self.T_surf), ("_clim_qbl_slow", self.q[:, -1]),
                            ("_clim_wind_slow", self._exchange_wind()), ("_clim_evap_slow", self._evap),
                            ("_clim_precip_slow", self._precip), ("_clim_flash_slow", self._surf_flash)]
            if self._u_surf is not None:
                _clim_fields.append(("_clim_usurf_slow", self._u_surf))
            for _cn, _cv in _clim_fields:
                _buf = getattr(self, _cn)
                if _buf is None or getattr(_buf, "shape", ()) != (self.cfg.nx,):
                    _seed = xpc.asarray(_cv, dtype=xpc.float64)   # seed (eager, pre-capture)
                    if _seed.shape != (self.cfg.nx,):             # scalar wind -> (nx,)
                        _seed = xpc.broadcast_to(_seed, (self.cfg.nx,))
                    setattr(self, _cn, _seed.copy())              # OWN a writable buffer (never alias the live field)
                else:
                    xpc.add(_buf, a_c * (_cv - _buf), out=_buf)
        # CAPTURE-READY (Phase E): on the device path ``dt`` is a 0-d DEVICE scalar (see _cfl_dt), so
        # ``float(dt)`` here would be a device->host sync -- ILLEGAL during CUDA-graph capture (it aborts the
        # capture). Accumulate the true-time on-device via ``_accum_pending`` -- the SAME in-place idiom the
        # water-budget cums use (#302 review finding 2): an in-place add re-executes on the fixed buffer every
        # replay, so the running sum genuinely advances per replay (the old ``pending = pending + dt`` rebind
        # baked only the last replayed step, freezing sim_time under the captured driver -- ``_sim_time_pending``
        # was simply MISSED in that #302 generalization). Folded into ``sim_time`` at the flush cadence
        # (``_flush_sim_time``). OFF (and CPU) is the verbatim host accumulation -> byte-identical.
        if self.cfg.capture_ready and self._dyn_on_device and self.run_dynamics:
            _xps = get_xp(self.T_surf)
            self._sim_time_pending = self._accum_pending(self._sim_time_pending, dt, _xps)
            # the step COUNTER rides the same pending idiom: ``self.iter += 1`` below is host code and
            # never re-executes under captured replay, so iter froze at warmup+1 (run6: iter=11 at
            # nominal day 380) while sim_time -- fixed by #327 -- advanced. +1.0 is a python-scalar
            # kernel operand (no H2D); folded with sim_time in _flush_sim_time.
            self._iter_pending = self._accum_pending(self._iter_pending, 1.0, _xps)
        else:
            self.sim_time += float(dt)             # accumulate TRUE physical time (the adaptive dt actually taken)
            self.iter += 1
        # #237 emergent salt-CCN: periodically rebuild N_d from the now-resolved wind (default refresh_every=0
        # -> never runs -> byte-identical; skipped under graph capture, where you rebuild between segments).
        c = self.cfg
        if (c.salt_ccn and c.salt_ccn_source == "transport" and c.salt_ccn_transport_refresh_every
                and not c.capture_ready and self._u_surf is not None
                and self.iter % c.salt_ccn_transport_refresh_every == 0):
            self._build_salt_ccn_field()
        # #308 item 4 -- the silent-static footgun: a driver that runs salt_ccn_dynamic ON but never
        # calls refresh_salt_ccn (no cadence wired) LOOKS like "thermostat ON" while running a frozen
        # salt field. Warn once past ~2 sim-days with zero refreshes ever. Eager-path only by nature
        # (captured replays never re-enter this Python) -- exactly the drivers at risk: the harnesses
        # that capture are the ones that already wire the cadence.
        if (c.salt_ccn and c.salt_ccn_dynamic and not getattr(self, "_warned_salt_static", False)
                and getattr(self, "_salt_refresh_count", 0) == 0 and self.sim_time > 2.0 * g.CLOCK.T_day):
            import warnings
            warnings.warn("salt_ccn_dynamic is ON but refresh_salt_ccn has NEVER run (> 2 sim-days): "
                          "the run is salt-STATIC while looking dynamic -- wire the driver's batch-"
                          "boundary refresh cadence (#308 item 4).", RuntimeWarning, stacklevel=2)
            self._warned_salt_static = True

    # ------------------------------------------------------------------ diagnostics & checkpoint
    def _aloft_moisture_column(self, q, sigma_thresh=ALOFT_SIGMA_THRESH):
        """Column-integrated water vapour in the ALOFT layers (sigma < ``sigma_thresh``) [kg/m^2 = mm of
        water]. This is the RESOLVED, ABSOLUTE-magnitude poleward-aloft moisture the single overturning
        cell carries to the ribbon and dew-points on its return branch -- the principled OPT-IN
        replacement (env COSMO_ALOFT_SUPPLY_ABSOLUTE; the legacy law stays the DEFAULT) for the legacy
        normalize-to-shape supply band F(theta), which discarded the core's absolute precip magnitude
        and re-imposed an arbitrary 8-50 mm scale (cascade.build_aloft_supply_target).

        ``q`` is the (nx, nz) vapour field (TOP->SURFACE on the pressure grid, as ``diagnostics`` clips
        it); ``self.dm`` (nz,) the layer masses; ``self.sigma`` (nz,) = p/P_SURF. Vapour only (the
        airborne moisture available to rain), not the cloud condensate q_c. Unlike the precip RATE
        [mm/day], this water-substance COLUMN [mm] is bounded by qsat and does NOT carry the equatorial
        chimney's 1/dx grid-scale singularity, so it can feed F(theta) without a /max() normalization that
        the chimney would dominate."""
        xp = get_xp(q)                                                # device-generic (Phase D)
        mask = self.sigma < sigma_thresh
        return xp.sum(xp.asarray(q)[:, mask] * self.dm[None, mask], axis=1)   # (nx,) [kg/m^2 = mm]

    def _aloft_thermo(self, q, sigma_thresh=ALOFT_SIGMA_THRESH, wsat_min_mm=RH_ALOFT_WSAT_MIN_MM):
        """RESOLVED aloft-layer (sigma < ``sigma_thresh``) thermodynamics, the principled replacement for
        Stage C's fake fixed-lapse aloft (``T_aloft = SST - 6.5K/km*4km``). Returns device-generic (nx,)
        arrays: mass-weighted mean temperature ``T_aloft`` [K] and specific humidity ``q_aloft`` [kg/kg]
        over the aloft layers; the SATURATION water column ``Wsat_aloft`` = sum qsat(T,p)*dm [kg/m^2 = mm]
        (the column saturation capacity Stage C currently derives from the fake lapse); and the
        column-integrated relative humidity ``RH_aloft`` = aloft_moisture / Wsat_aloft. ``Wsat_aloft`` and
        the aloft moisture share the SAME mask/integration, so RH_aloft is exactly their ratio (Phase-C
        wishlist item 1 -- EXPOSURE only; Stage C does the consumption).

        ``RH_aloft`` is MASKED TO NaN wherever ``Wsat_aloft < wsat_min_mm`` (ROADMAP PHY-15). Past the
        cold trap the aloft column reaches 50-200 K, ``Wsat_aloft`` collapses to ~1e-49 mm, and the
        vapour advection leaves behind is a round-off residue (~1e-8 mm = 1e-10 of a band column): their
        ratio is not a humidity. Unmasked, the canonical run17 state has 59 such columns (of 975), and
        every one of them reports a number the guard floor invented: 20,039.8 (~2e6 %) and 137 on the two
        where the residue was largest -- the first exactly ``residue / 1e-12`` -- an innocuous-looking
        **0.056** on a third whose true ratio is 4.5e34, and exactly **0.0** on the remaining 56 where the
        residue underflowed to zero as well. The quiet ones are the dangerous half: they read as ordinary
        (5 % / bone-dry) humidities. NaN says instead that
        there is no ratio to report; the raw pieces stay available, since ``diagnostics()`` exports
        ``aloft_moisture_mm`` and ``Wsat_aloft_mm`` separately. NOTE the floor only CAPPED the blow-up,
        it did not create it: the unfloored ratio at that column is 2.7e5, so raising or removing the
        1e-12 guard is not a fix. Nothing in the model consumes RH_aloft (``Wsat_aloft``, which IS
        consumed, is returned unchanged), so this moves no result -- see RH_ALOFT_WSAT_MIN_MM for the
        floor's value, its rationale, and the measurement showing the choice is not delicate."""
        xp = get_xp(q)
        mask = self.sigma < sigma_thresh
        dm_a = self.dm[None, mask]                                    # (1, n_aloft) layer masses
        m_a = float(xp.sum(self.dm[mask]))                            # total aloft air mass [kg/m^2]
        qm = xp.asarray(q)[:, mask]
        Tm = self.T_atm[:, mask]
        T_aloft = xp.sum(Tm * dm_a, axis=1) / m_a                     # mass-weighted mean T [K]
        q_aloft = xp.sum(qm * dm_a, axis=1) / m_a                     # mass-weighted mean q [kg/kg]
        aloft_moist = xp.sum(qm * dm_a, axis=1)                       # aloft vapour column [kg/m^2 = mm]
        qsat = th.qsat(self.T_atm[:, mask], self.p_mid[None, mask])   # (nx, n_aloft) saturation q
        Wsat_aloft = xp.sum(qsat * dm_a, axis=1)                      # saturation column [kg/m^2 = mm]
        RH_aloft = xp.where(Wsat_aloft < wsat_min_mm,                 # degenerate denominator -> no ratio
                            xp.asarray(xp.nan, dtype=Wsat_aloft.dtype),
                            aloft_moist / xp.maximum(Wsat_aloft, 1e-12))
        return dict(T_aloft=T_aloft, q_aloft=q_aloft, Wsat_aloft=Wsat_aloft, RH_aloft=RH_aloft)

    def _cloud_columns(self, out, q):
        """Column cloud water path + LW/SW optical depths (Phase-C wishlist item 3), recomputed OFF the hot
        loop from the SAME cloud fraction ``out['cloud_fraction']`` the radiation used, so the column optics
        are consistent with the in-loop radiation (no drift) without adding per-step work to ``_radiate``.
        Returns device-generic (nx,) arrays; all zero when clouds are off (``out`` carries no cloud keys)."""
        c = self.cfg
        xp = get_xp(q)
        f = out.get("cloud_fraction")
        if f is None or not c.clouds:
            z = xp.zeros(c.nx)
            return dict(cloud_water_path=z, cloud_tau_lw_col=z, cloud_tau_sw_col=z)
        f = xp.asarray(f)
        # mirror _radiate's cap / prognostic kwargs so the recomputed optics match the in-loop ones exactly
        cap_kw = (dict(lwc_max=rad.CLOUD_LWC_PLAUSIBLE, iwc_max=rad.CLOUD_IWC_PLAUSIBLE)
                  if (c.prognostic_condensate and c.qc_radiative_cap_decouple) else {})
        qc_kw = dict(qc=self.q_c) if c.prognostic_condensate else {}
        if c.salt_ccn and c.salt_ccn_twomey:                          # mirror _radiate's Twomey thread (#237)
            qc_kw["n_d"] = self.N_d[:, None]
            qc_kw["n_d_ref"] = c.salt_ccn_nd_ref
        dtau_lw_cloud, _R, tau_scat, _dsw, lwp = rad.cloud_optical_properties(
            f, self.T_atm, self.p_mid[None, :], self.dm[None, :], water_scale=c.cloud_water_scale,
            return_tau_sw=True, return_lwp=True, ssa_liq=c.cloud_sw_ssa_liq, ssa_ice=c.cloud_sw_ssa_ice,
            **qc_kw, **cap_kw)
        return dict(
            cloud_water_path=xp.sum(f * lwp, axis=1),               # grid-mean in-cloud water column [kg/m^2 = mm]
            cloud_tau_lw_col=xp.sum(dtau_lw_cloud, axis=1),         # grid-mean column cloud LW optical depth
            cloud_tau_sw_col=xp.sum(f * tau_scat, axis=1),         # column cloud SW scattering optical depth (=tau_tot)
        )

    def diagnostics(self):
        """Grid/state summary, the resolved-circulation diagnostics (M2b.2), and the column-thermodynamic
        diagnostics (M2b.3). The tier-1 conservation gates (energy/mass/water closure) are the separate
        ``energy_conservation`` / ``mass_conservation`` / ``water_conservation`` methods (M2b.5).

        Circulation (height grid, B-convention): ``u``/``w`` velocities [m/s]; ``u_surf`` the lowest
        interior level's cross-track wind (j=1); ``surf_sign_changes`` the equatorward/poleward reversals
        of u_surf (0 == ONE cell, the Rossby->inf theorem-invariant); ``rising_theta`` the latitude of
        peak mid-height ascent; ``psi_single_signed`` whether the interior streamfunction is one-signed.

        Thermo (pressure grid): a FRESH ``column_radiation`` call -> ``OLR``/``toa_net``; the surface-air
        temperature ``T_air_surf_C`` (lowest layer); ``precip_mm_day`` the SURFACE precipitation rate
        (== the column condensation rate only in the diagnostic-cloud limit; under ``prognostic_condensate``
        it is the autoconversion/ceiling rain, less below-cloud re-evaporation when that is enabled);
        ``vapor_path`` the column-integrated water vapor [kg/m^2]; ``T_atm``/``q`` the full fields."""
        c = self.cfg
        # DEVICE (Phase D): diagnostics is array-module-generic and returns HOST arrays. ``_h`` brings a
        # (possibly device) array to host AND copies (np.array) -- preserving the original ``.copy()``
        # semantics on CPU (a consumer may mutate a returned field) while giving a fresh host array on GPU.
        # Runs at checkpoint cadence, off the hot step path.
        _h = lambda a: np.array(to_host(a))
        u, w = self.velocities()
        xpd = get_xp(u)
        u_surf = u[1, :]
        s = xpd.sign(u_surf[3:-3])
        surf_sign_changes = int(xpd.sum(s[1:] * s[:-1] < 0))
        wmid = w[self.cfg.nz // 2, :]
        psi_int = self.psi[1:-1, 1:-1]
        pmax = float(xpd.abs(self.psi).max())
        psi_single_signed = bool(xpd.all(psi_int >= -1e-6 * (pmax + 1e-30))
                                 or xpd.all(psi_int <= 1e-6 * (pmax + 1e-30)))
        xp = get_xp(self.q)
        q = xp.clip(self.q, 0.0, th.Q_MAX)
        out = self._radiate(q)
        absorbed_sw = self.S_sw_in - out["planetary_reflected"]   # incoming counts the Stage-5 diffuse haze
        vapor_path = xp.sum(q * self.dm[None, :], axis=1)
        theta_deg = to_host(self.theta_deg)
        # cloud fields (#19): zero when clouds are off (out carries no cloud keys then), so the dict shape
        # is stable whether or not clouds are enabled.
        cloud_cover = out.get("cloud_cover", np.zeros(c.nx))
        cloud_sw_albedo = out.get("cloud_sw_albedo", np.zeros(c.nx))
        cloud_fraction = out.get("cloud_fraction", np.zeros((c.nx, c.nz)))
        # --- Phase-C diagnostics wishlist (additive, read-only; off the hot loop) ---
        aloft = self._aloft_thermo(q)                       # resolved aloft T/q/Wsat/RH (item 1)
        cl = self._cloud_columns(out, q)                    # cloud water path + column optical depth (item 3)
        out_clear = self._radiate(q, clouds=False)          # clear-sky surface SW (item 4)
        surf_down_sw = out["surf_down_sw"]                  # surface SW reaching ground under THIS sky [W/m^2]
        surf_down_sw_clear = out_clear["surf_down_sw"]      # clear-sky surface SW [W/m^2]
        S_sw_in = self.S_sw_in                             # TOA down-SW (direct + Stage-5 diffuse) [W/m^2]
        clearsky_transmission = surf_down_sw_clear / xp.maximum(S_sw_in, 1e-12)
        diffuse_frac = (self.S_diffuse / xp.maximum(S_sw_in, 1e-12)
                        if self.S_diffuse is not None else xp.zeros(c.nx))
        return dict(
            theta_deg=theta_deg, x_km=to_host(self.x) / 1e3, S=to_host(self.S),
            cloud_cover=to_host(cloud_cover), cloud_sw_albedo=to_host(cloud_sw_albedo),
            cloud_fraction=to_host(cloud_fraction),
            z=_h(self.z), z_top=self.z_top, dz=self.dz,
            p_ref=self.p_ref.copy(), T_ref=_h(self.T_ref), rho_ref=self.rho_ref.copy(),
            N2_ref=self.N2_ref.copy(),
            T_surf_C=to_host(self.T_surf - C.T_FREEZE), T_atm=_h(self.T_atm), q=to_host(q),
            q_c=_h(self.q_c),                                  # prognostic cloud condensate [kg/kg] (Chunk 1)
            cond_path=to_host(xp.sum(self.q_c * self.dm[None, :], axis=1)),   # column condensate path [kg/m^2]
            # prognostic rain (stage 7): zeros when the flag is OFF -- stable dict shape, the cloud-field
            # convention. rain_path is the in-flight column rain water [kg/m^2].
            q_r=(_h(self.q_r) if self.q_r is not None else np.zeros((c.nx, c.nz))),
            rain_path=(to_host(xp.sum(self.q_r * self.dm[None, :], axis=1)) if self.q_r is not None
                       else np.zeros(c.nx)),
            T_air_surf_C=to_host(self.T_atm[:, -1] - C.T_FREEZE), vapor_path=to_host(vapor_path),
            aloft_moisture_mm=to_host(self._aloft_moisture_column(q)),   # absolute aloft-vapour column [mm]
            # resolved aloft layer (sigma<0.5) -- replaces Stage C's fake fixed-lapse aloft (Phase-C item 1):
            T_aloft_C=to_host(aloft["T_aloft"] - C.T_FREEZE), q_aloft=to_host(aloft["q_aloft"]),
            Wsat_aloft_mm=to_host(aloft["Wsat_aloft"]), RH_aloft=to_host(aloft["RH_aloft"]),
            # cloud water path + column optical depth (Phase-C item 3):
            cloud_water_path_mm=to_host(cl["cloud_water_path"]),
            cloud_tau_lw_col=to_host(cl["cloud_tau_lw_col"]), cloud_tau_sw_col=to_host(cl["cloud_tau_sw_col"]),
            # clear-sky surface SW + diffuse fraction (Phase-C item 4):
            surf_down_sw=to_host(surf_down_sw), surf_down_sw_clear=to_host(surf_down_sw_clear),
            clearsky_transmission=to_host(clearsky_transmission), diffuse_frac=to_host(diffuse_frac),
            sigma=_h(self.sigma),                              # p/P_SURF (nz,) -- re-integrate aloft col at another ceiling
            D_surf=to_host(out["D_surf"]),                      # downwelling LW at surface [W/m^2] (land-T backbone)
            OLR=to_host(out["OLR"]), toa_net=to_host(absorbed_sw - out["OLR"]), albedo=to_host(self._albedo()),
            precip_mm_day=to_host(self._precip) * 86400.0,      # kg/m^2/s -> mm/day
            surf_flash_mm_day=to_host(self._surf_flash) * 86400.0,   # surface flash-boil flux (0 when the
            #   flag is OFF); surface-reaching LIQUID = precip_mm_day - surf_flash_mm_day
            wetness=to_host(self._wetness()),                   # evaporation efficiency [0,1] (whatever the
            #   active closure: T-ramp / budget / #302 prognostic W)
            surf_water_mm=(to_host(self.W_surf) if self.W_surf is not None
                           else np.zeros(c.nx)),                # #302 prognostic surface water [kg/m^2 = mm]
            u=to_host(u), w=to_host(w), u_surf=to_host(u_surf), surf_sign_changes=surf_sign_changes,
            rising_theta=float(theta_deg[int(xpd.argmax(wmid))]),
            psi_single_signed=psi_single_signed,
            max_abs_psi=pmax, last_dt_dyn=self._last_dt_dyn, last_dt_th=self._last_dt_th, iter=self.iter,
        )

    def subsidence_diagnostics(self, sigma_lo=0.30, sigma_hi=0.75):
        """M2b.4 cross-check (tier 4): the **radiative-subsidence rate** per column, for comparison
        against M2a's external anchor `w_sub = -0.25 cm/s` (`archive/M2A_RESULTS.md`). Over the
        free-tropospheric
        band sigma in [sigma_lo, sigma_hi] (above the surface boundary layer, below the stratosphere) it
        reads, from a FRESH radiation call on the live state, the mean radiative heating `Q_rad` [K/day]
        (LW cooling dominates -> negative) and the static stability `d theta_pot/dz` [K/km], then forms
        `w_sub = Q_rad / (d theta/dz)` (``radiative_subsidence_rate``) -- the velocity implied by the
        column's OWN radiation + stability, no free U.

        This validates the **strength** of the subsidence the core's column physics imply (sign +
        order), NOT its existence (a transporting cell's ribbon subsidence is structurally forced -- the
        milestone-1 / M2a 'forced-observable' trap). The resolved dynamical `w` matching this magnitude
        is the M2b.5/6 closure (Issue A), not asserted here. Returns per-column arrays."""
        c = self.cfg
        xp = get_xp(self.q)                                                # device-generic (Phase D)
        q = xp.clip(self.q, 0.0, th.Q_MAX)
        out = self._radiate(q)
        Q_rad_K_day = out["H"] / self.C_layer[None, :] * 86400.0           # (nx, nz) [K/day]
        z = self._hydrostatic_z(self.T_atm)                               # (nx, nz) TOP->SURFACE [m]
        theta = self.T_atm * (C.P_SURF / self.p_mid[None, :]) ** C.KAPPA_POISSON
        band = (self.sigma >= sigma_lo) & (self.sigma <= sigma_hi)         # free-trop layers
        # stability d theta/dz over the band: theta and z are gradient'd along the SAME axis, so the
        # per-index spacing (including its TOP->SURFACE sign) cancels in the ratio -- no explicit flip needed.
        dtheta_dz = xp.gradient(theta, axis=1) / xp.gradient(z, axis=1)    # (nx, nz) [K/m]
        stability_K_km = to_host(xp.mean(dtheta_dz[:, band], axis=1) * 1000.0)   # [K/km] (host: tier-4 anchor)
        Q_rad_band = to_host(xp.mean(Q_rad_K_day[:, band], axis=1))         # [K/day]
        w_sub_cm_s = radiative_subsidence_rate(Q_rad_band, stability_K_km)  # host-only anchor helper
        return dict(Q_rad_K_day=Q_rad_band, stability_K_km=stability_K_km,
                    w_sub_cm_s=w_sub_cm_s, theta_deg=to_host(self.theta_deg))

    def cloud_radiative_effect(self):
        """The cloud radiative effect (CRE) per column -- the standard cloud-vs-clear TOA flux difference on
        the SAME atmospheric/surface state (issue #19; the headline diagnostic for 'how clouds move the hot
        zone'). Radiation is computed twice on the live state -- once WITH clouds (the config's cloud
        inputs; optical depth physics-derived), once clear-sky -- and differenced, per column:

            cre_sw  = absorbed_SW(cloud) - absorbed_SW(clear)   [W/m^2]  (<=0: cloud-albedo COOLING)
            cre_lw  = OLR(clear) - OLR(cloud)                   [W/m^2]  (>=0: cloud-greenhouse WARMING)
            cre_net = cre_sw + cre_lw                           [W/m^2]  (net TOA cloud forcing; <0 cools)

        The cloudy call uses ``clouds=True`` REGARDLESS of ``cfg.clouds``, so the implied cloud forcing of
        any state (even a clear-sky-converged one) is measurable. A strongly NEGATIVE ``cre_net`` over the
        hot equatorward columns and ~0 over the subsiding ribbon is the signature that clouds cool the hot
        zone preferentially -> the habitable band can extend equatorward, toward a higher sun. (TOA forcing
        is firmer than the implied surface-T shift, which also needs the local feedback / transport.)"""
        c = self.cfg
        q = get_xp(self.q).clip(self.q, 0.0, th.Q_MAX)                 # device-generic (Phase D); CRE is a diagnostic
        # mirror the live thermo step: feed the resolved-ascent convective boost when it is active, so the
        # CRE diagnostic sees the SAME cloud field the in-loop radiation does (else it understates the forcing
        # for a conv-boost study; pure RH when boost=0 / no flow -- review #69 item 1).
        conv = (self._ascent_signal() if (c.cloud_conv_boost > 0.0 and self.run_dynamics) else None)
        # mirror _radiate's FULL radiative config in BOTH calls (PR #101 review): the conserving band
        # split changes the CLEAR background-band opacity and the diffuse SW beam is albedo-scaled too,
        # so omitting either would difference a config the in-loop radiation never ran.
        # mirror _radiate's prognostic-condensate kwargs when ON (off by default -> empty -> byte-identical):
        # the CRE diagnostic then reads the SAME prognostic in-cloud water + PDF cover the in-loop radiation does.
        qc_kw = (dict(cloud_qc=self.q_c, cloud_f_min=c.cloud_f_min, cloud_qc_thresh=c.qc_cover_thresh,
                      cloud_pdf=c.qc_pdf_cloud, cloud_qc_decouple_cap=c.qc_radiative_cap_decouple)
                 if c.prognostic_condensate else {})
        if c.prognostic_condensate and c.bl_stratocumulus:    # mirror the Chunk-4 deck cover floor (see _radiate)
            qc_kw["cloud_cover_floor"] = self._bl_deck_cover
        cloudy = rad.column_radiation(
            self.p_edge[None, :], self.T_atm, q, self.T_surf, self.S, self._albedo(),
            co2_ppm=c.co2_ppm, k_wv=c.k_wv_lw, tau_wellmixed=c.tau_lw_wellmixed, clouds=True,
            cloud_rh_crit=c.cloud_rh_crit, cloud_water_scale=c.cloud_water_scale,
            cloud_conv=conv, cloud_conv_boost=c.cloud_conv_boost, cloud_sw_overlap=c.cloud_sw_overlap,
            cloud_sw_ssa_liq=c.cloud_sw_ssa_liq, cloud_sw_ssa_ice=c.cloud_sw_ssa_ice,
            cloud_sw_twostream=c.cloud_sw_twostream,
            lw_window=c.lw_window, f_win=c.f_lw_window, k_cont=c.k_cont_self,
            lw_conserving_split=c.lw_conserving_split,
            sw_bands=c.sw_bands, sw_diffuse=c.sw_diffuse, S_diffuse=self.S_diffuse, **qc_kw)
        clear = rad.column_radiation(
            self.p_edge[None, :], self.T_atm, q, self.T_surf, self.S, self._albedo(),
            co2_ppm=c.co2_ppm, k_wv=c.k_wv_lw, tau_wellmixed=c.tau_lw_wellmixed, clouds=False,
            lw_window=c.lw_window, f_win=c.f_lw_window, k_cont=c.k_cont_self,
            lw_conserving_split=c.lw_conserving_split,
            sw_bands=c.sw_bands, sw_diffuse=c.sw_diffuse, S_diffuse=self.S_diffuse)
        cre_sw = (self.S - cloudy["planetary_reflected"]) - (self.S - clear["planetary_reflected"])
        cre_lw = clear["OLR"] - cloudy["OLR"]
        # SURFACE cloud forcing: the change in radiative input the SURFACE receives (absorbed SW + down LW).
        # Cloud SW reflection cuts surf_abs_sw (<0); cloud LW raises D_surf (>0). The surface T responds to
        # THIS (not the TOA net), buffered by the surface restoring feedback (Planck + sensible + the steep
        # evaporative thermostat) -- so the implied SST shift uses the surface forcing, not cre_net.
        cre_surf_sw = cloudy["surf_abs_sw"] - clear["surf_abs_sw"]
        cre_surf_lw = cloudy["D_surf"] - clear["D_surf"]
        return dict(theta_deg=to_host(self.theta_deg), cre_sw=to_host(cre_sw), cre_lw=to_host(cre_lw),
                    cre_net=to_host(cre_sw + cre_lw),
                    cre_surf_sw=to_host(cre_surf_sw), cre_surf_lw=to_host(cre_surf_lw),
                    cre_surf=to_host(cre_surf_sw + cre_surf_lw),
                    cloud_cover=to_host(cloudy["cloud_cover"]), cloud_sw_albedo=to_host(cloudy["cloud_sw_albedo"]),
                    OLR_clear=to_host(clear["OLR"]), OLR_cloud=to_host(cloudy["OLR"]))

    # ------------------------------------------------------------------ Stage-C interface contract (M2b.6)
    def poleward_quiescence(self, frac=0.80):
        """M2b.6 acceptance: verify the circulation goes QUIESCENT toward the poleward wall (``|psi|->0``),
        i.e. the cell does not run into the cold edge / the cold tail is not driving a spurious return cell.
        Returns the ratio of max|psi| in the poleward ``1-frac`` fraction of the domain (the cold tail) to
        the global max|psi|; ->0 means the overturning is confined equatorward of the truncation, so the
        ``theta_max`` boundary is not contaminating the answer (scope the steady-state claim to
        ``conv_theta_max``)."""
        xpp = get_xp(self.psi)                                     # device-generic (Phase D)
        pmax = float(xpp.abs(self.psi).max())
        i0 = int(frac * self.cfg.nx)
        pole = float(xpp.abs(self.psi[:, i0:]).max()) if i0 < self.cfg.nx else 0.0
        return dict(global_max_psi=pmax, poleward_max_psi=pole, ratio=pole / (pmax + 1e-30),
                    poleward_theta_deg=float(to_host(self.theta_deg)[i0]) if i0 < self.cfg.nx else float("nan"))

    def stage_c_inputs(self):
        """M2b.6 deliverable: emit the Stage-C INTERFACE CONTRACT fields from the core state, so Stage C can
        consume the fused core in place of the A->B backbone. Returns a dict of ``theta_deg -> value``
        callables matching EXACTLY what ``cascade.run_cascade`` builds from Stage A + Stage B:

          ``sst_of_theta``          [C]    the core's surface temperature -- the ocean SST backbone
          ``wetness_of_theta``      [0,1]  the core's evaporation-efficiency ramp (ocean<->salt flat)
          ``aloft_supply_of_theta`` [mm]   the poleward-aloft moisture-supply band F(theta) via the SAME
                                           ``cascade.build_aloft_supply_target`` the live cascade uses:
                                           the legacy normalize-to-shape law by default, or the resolved
                                           absolute aloft moisture column under COSMO_ALOFT_SUPPLY_ABSOLUTE
          ``ascent_gate``           [0,1]  ``cascade.cell_ascent_gate`` fed the core's resolved w (gates
                                           the loft FRACTION -- where the cell rises)
          ``subsidence_gate``       [0,1]  ``cascade.subsidence_gate`` fed the core's resolved w (gates the
                                           convective DRAIN -- the sign of w)

        The gates REUSE ``cascade.cell_ascent_gate`` / ``subsidence_gate`` verbatim (the core's
        ``diagnostics()`` dict carries the ``w`` / ``theta_deg`` / ``precip`` they read), so the Stage-C
        contract is byte-for-byte the SAME object whether driven by A+B or by the fused core -- which is how
        M2b.6 VERIFIES the decoupling lives at the interface contract (what each field means physically),
        not at frozen numerical tunings. (cascade is imported lazily to avoid the import cycle with
        ``run_cascade``'s ``core='ad'`` branch.)"""
        from . import cascade
        d = self.diagnostics()
        th = d["theta_deg"]
        # NB: the live `run_cascade(core="ad")` path recomputes F(theta) from the synthesized dB itself
        # (it consumes only this dict's `diagnostics`/`sst`/`wetness`), so this `aloft_supply_of_theta`
        # field is consumed only by tests + standalone callers -- but it tracks the SAME shared builder so
        # the two cannot drift. DEFAULT (legacy normalize-to-shape) is byte-identical; the opt-in absolute
        # aloft-column law fires under env COSMO_ALOFT_SUPPLY_ABSOLUTE (build_aloft_supply_target).
        F_target = cascade.build_aloft_supply_target(
            th, d["precip_mm_day"], aloft_moisture_mm=d.get("aloft_moisture_mm"),
            absolute=_env_flag("COSMO_ALOFT_SUPPLY_ABSOLUTE", False))
        dgate = dict(theta_deg=th, w=d["w"], precip=d["precip_mm_day"])   # the cascade gates' input contract
        return dict(sst_of_theta=cascade._interp(th, d["T_surf_C"]),
                    wetness_of_theta=cascade._interp(th, self._wetness()),
                    aloft_supply_of_theta=cascade._interp(th, F_target),
                    ascent_gate=cascade.cell_ascent_gate(dgate),
                    subsidence_gate=cascade.subsidence_gate(dgate),
                    # RESOLVED aloft layer, exposed for Phase C to retire its fake fixed-lapse aloft
                    # (T_aloft = SST - 6.5K/km*4km) and recover Wsat_aloft from the real profile (item 1).
                    # EXPOSURE only -- Stage C does the consumption; nothing here changes the cascade.
                    T_aloft_of_theta=cascade._interp(th, d["T_aloft_C"]),
                    q_aloft_of_theta=cascade._interp(th, d["q_aloft"]),
                    Wsat_aloft_of_theta=cascade._interp(th, d["Wsat_aloft_mm"]),
                    RH_aloft_of_theta=cascade._interp(th, d["RH_aloft"]),
                    # salt-aerosol CCN droplet-number field N_d(theta) [cm^-3] (#237): the emergent/prescribed
                    # column field, exposed so the cascade can wire Stage C's warm-rain salt coupling
                    # SELF-CONSISTENTLY with the core's autoconv limb. None when salt_ccn is OFF -> Stage C's
                    # salt coupling stays OFF (byte-identical).
                    salt_ccn_nd_of_theta=(cascade._interp(th, to_host(self.N_d)) if self.N_d is not None else None),
                    diagnostics=d, poleward_quiescence=self.poleward_quiescence())

    # ------------------------------------------------------------------ conservation gates (#1; M2b.5)
    def energy_conservation(self):
        """TIER-1 hard gate (archive/M2B_SUBTASKS.md M2b.5): GLOBAL energy closure with **NO export
        term**. The only energy export to space is OLR (the psi=0 walls make the advective lateral
        export exactly zero, #5 -- this is the Issue-A dissolver), so the domain-integrated
        top-of-atmosphere imbalance
        is purely (absorbed SW - OLR):

            net_imbalance = INTEGRAL_x (absorbed_SW(x) - OLR(x)) dx        [W per m of along-track]

        with the uniform-in-distance cross-track weight ``dx`` (the flat slab's area element is uniform in
        x). At a converged steady state this -> 0 to a small residual relative to the gross absorbed SW.
        There is NO lateral/sponge export added or subtracted anywhere -- the *absence* of such a term is
        the property under test, NOT a number to tune. (Per-column ``toa_net`` is also returned.)

        M2b.6 NOTE (the diagnostic M3 must watch): with the resolved flow now carrying HEAT
        (``_advect_heat``), the columns NO LONGER balance individually -- the cell moves energy between
        them -- so it is the GLOBAL ``net_imbalance`` that must -> 0, not ``toa_net`` per column. The heat
        transport is now mass-consistent and fully flux-form in BOTH limbs (``_divfree_mass_flux``), so it
        conserves ``INTEGRAL theta_pot dm`` to round-off and exports zero across the psi=0 walls -- there is
        no longer a cross-track *advective*-form energy leak. ONE caveat remains to CONFIRM at convergence,
        not a structural guarantee: the flux-form vertical limb conserves the isentropic mass integral
        ``INTEGRAL theta_pot dm`` but not exactly ``INTEGRAL cp*T dm`` (the Exner factor varies between
        layers, so adiabatic vertical rearrangement of theta_pot is the standard static-energy treatment
        that SHOULD net to zero at a closed steady state). So ``rel_imbalance`` is the DOMINANT
        non-conservative term to watch, to be confirmed -> 0 at convergence; if M3 sees it fail to shrink,
        the isentropic-vs-enthalpy bookkeeping is the suspect to audit before trusting the pinned
        T(theta)."""
        c = self.cfg
        xp = get_xp(self.q)                                           # device-generic (Phase D); the gate holds on GPU
        q = xp.clip(self.q, 0.0, th.Q_MAX)
        out = self._radiate(q)
        absorbed_sw = self.S_sw_in - out["planetary_reflected"]       # (nx,) [W/m^2] (incoming + Stage-5 diffuse)
        toa_net = absorbed_sw - out["OLR"]                            # (nx,) per-column TOA imbalance
        net_imbalance = float(xp.sum(toa_net * self._dx_w))             # W/m along-track (scalar pulled to host)
        gross = float(xp.sum(absorbed_sw * self._dx_w))
        return dict(toa_net=to_host(toa_net), absorbed_sw=to_host(absorbed_sw), OLR=to_host(out["OLR"]),
                    net_imbalance=net_imbalance, gross_absorbed=gross,
                    rel_imbalance=net_imbalance / (gross + 1e-30))

    def mass_conservation(self):
        """TIER-1 hard gate (archive/M2B_SUBTASKS.md M2b.5): ANELASTIC mass continuity div(rho_ref u) = 0. The
        mass streamfunction enforces it BY CONSTRUCTION -- rho_ref u = d psi/dz, rho_ref w = -d psi/dx, so
        div(rho_ref u) = d^2 psi/dxdz - d^2 psi/dzdx, which the centered interior stencils cancel to
        round-off. Returns the residual field div(rho_ref u) and its DEEP-interior max (excluding the
        two-cell border where the one-sided edge stencils live), normalized by a momentum-gradient scale
        rho_ref|u|/dx so the tolerance is dimensionless. A clean theorem-invariant: it certifies the
        Poisson solve + velocity recovery are mutually consistent, independent of any physics."""
        u, w = self.velocities()
        xpm = get_xp(u)                                              # device-generic (Phase D)
        rho = xpm.asarray(self.rho_ref)[:, None]
        div = _ddx(rho * u, self._dx_op) + _ddz(rho * w, self._dz_op)    # (nz, nx) anelastic divergence
        deep = div[2:-2, 2:-2]
        if self._dx_cell is None:
            scale = float(xpm.max(xpm.abs(rho * u))) / self.dx + 1e-30    # momentum-gradient magnitude
        else:
            # PER-CELL gradient scale |rho u|_i / dx_local_i (review, PR #318): a single /dx_min scale is
            # the LARGEST possible normalizer, which makes rel_div_deep read ~dx_max/dx_min too SMALL in
            # the coarse plateau -- under-reporting is the wrong failure mode for a consistency
            # diagnostic. The per-cell max keeps the ratio honest everywhere on the stretched grid.
            scale = float(xpm.max(xpm.abs(rho * u) * xpm.asarray(self._inv_dxw_row))) + 1e-30
            #   (after device placement _inv_dxw_row IS a device array -> asarray is a no-op there)
        return dict(div=to_host(div), max_abs_div_deep=float(xpm.max(xpm.abs(deep))) if deep.size else 0.0,
                    rel_div_deep=(float(xpm.max(xpm.abs(deep))) / scale) if deep.size else 0.0)

    def water_conservation(self):
        """TIER-1 hard gate (archive/M2B_SUBTASKS.md M2b.5; CLOSED_ISSUES c5: trust the CUMULATIVE, not the
        instantaneous): the column water budget closes, surface evaporation source == condensation sink.
        Returns the CUMULATIVE area-weighted integrals (since the last ``reset_water_budget``) of the
        evaporation E and precipitation P, and their closure ratio P/E -> 1 at steady state, plus the
        instantaneous (noisier) integrals and the current total vapor mass. Measure over a window AFTER
        spin-up (the transient stores/releases column vapor, so cumulative E != P until the storage term
        is a negligible fraction); ``reset_water_budget`` starts the window.

        #294-A -- WHICH KEY IS THE GATE. ``closure_cum`` (P/(E+flash) -> 1) is a CLOSED-domain, STEADY-STATE
        identity: it is a readiness gauge, NOT a conservation gate, and it CANNOT see a leak (a silent sink
        depresses it exactly like un-finished spin-up does -- that ambiguity hid the q<=0.5 leak for 400+
        run15 days at ~0.25). The conservation gate is **atm_residual -> 0**, which holds transient or not:
            (vapor+cond+rain - atm_total_reset)
                == cum_evap + cum_surf_flash - cum_precip - cum_qcap + cum_qfloor + cum_qadv
        EVERY non-advective source/sink of column water must appear on the right, or the residual absorbs
        it and reports a leak: ``cum_qcap``/``cum_qfloor`` (the thermo clip) AND ``cum_qadv`` (the implicit
        advection positivity clamp -- 0.0 on the explicit path; omitting it read +7.5e-9 relative on the
        CANONICAL implicit path, PR #360 review).

        Two caveats on reading it:
        - RELATIVE, not absolute (as surf_residual): it is a difference of large totals, so gate it against
          max(atm_total, cum terms). Machine-precision closure is ~1e-16 relative on all advection paths.
        - A small TRANSIENT residual is expected under ``cond_boil_suppress`` and is NOT a leak: evaporation
          can push q past Q_MAX *after* the top-of-substep clip, so a gauge sampled in between sees ``store``
          with that excess clipped out (vapor_mass clips) while cum_evap already counted it and cum_qcap has
          not yet. It is caught at the next clip -- expect it on a boiling world; do not read it as a sink."""
        self._flush_water_budget()                                  # fold any capture-ready pending sums in
        xp = get_xp(self.q)                                          # device-generic (Phase D); scalars pulled to host
        evap_inst = float(xp.sum(self._evap * self._dx_w))            # kg/m/s along-track (area-weighted)
        precip_inst = float(xp.sum(self._precip * self._dx_w))
        # surface flash-boil (surf_flash_boil): flashed rain re-enters the atmosphere as vapour, so it is a
        # SECOND surface->atmosphere source next to E -- at steady state cum_precip == cum_evap +
        # cum_surf_flash. Including it in the closure denominators keeps the P/E->1 gate meaningful with the
        # flag ON; OFF both terms are exactly 0.0 (value-identical closure).
        flash_inst = float(xp.sum(self._surf_flash * self._dx_w))
        vapor_mass = float(self._column_mass(xp.clip(self.q, 0.0, th.Q_MAX), xp))
        # PROGNOSTIC RAIN (stage 7): the in-flight rain reservoir is COLUMN WATER STORAGE the closure's
        # storage term must count -- with the flag ON the state stores water in q, q_c AND q_r, so a
        # consumer reconstructing d(storage)/dt = E - P needs INTEGRAL (q + q_c + q_r) dm dx, not
        # vapor_mass alone (which never counted q_c either -- the pre-existing convention; both reservoir
        # masses are therefore reported as their OWN keys, the OFF-path tokens untouched). 0.0 when OFF.
        rain_mass = float(self._column_mass(self.q_r, xp)) if self.q_r is not None else 0.0
        cond_mass = float(self._column_mass(self.q_c, xp))
        # --- #302 surf_water_budget: the SURFACE water budget closes on its own integrals --
        #       (sum W dx - W_total_reset)  ==  cum_p_land - cum_evap        (machine precision)
        #     because the update is exactly dt*(P_land - E + div(F)) and the closed-wall transport
        #     divergence telescopes to ~0 in the area sum. cum_boundary_influx is structurally 0.0
        #     with closed walls (the reservoir 'unbounded source' is the deep W init, not a boundary
        #     term); with surf_water_edge_source ON it reports the ACCOUNTED Dirichlet-pin injection
        #     cum_w_edge, which surf_supply includes so the residual still closes with the open
        #     boundary. All keys 0.0/absent-cost when OFF.
        #     PRECISION NOTE: surf_residual is the difference of area totals ~1e13 kg/m against a deep
        #     reservoir background, so it is exact only to the fp64 cancellation floor ~1e-16 * W_total --
        #     gate it RELATIVE to max(W_total, cum terms), not absolutely. Keys are identically 0.0 OFF.
        #     surf_salt adds the SALT closure: (sum (S_dis+S_dep) dx - S_total_reset) == cum_s_edge
        #     (transport telescopes, the saturation adjustment conserves the sum, E moves no salt);
        #     same relative-precision gating.
        if self.W_surf is not None:
            W_total = float(xp.sum(self.W_surf * self._dx_w))
            surf_store = W_total - self._W_total_reset
            # supply = landed liquid - evaporation - snap-floor removals + the accounted edge injection
            #          + the accounted cryo-floor top-up (surf_water_cryo_pin; 0.0 with the flag OFF)
            surf_supply = (self._cum_p_land - self._cum_evap - self._cum_w_snap + self._cum_w_edge
                           + self._cum_w_cryo)
        else:
            W_total = surf_store = surf_supply = 0.0
        if self.S_dis is not None:
            S_total = float(xp.sum((self.S_dis + self.S_dep) * self._dx_w))
            salt_store = S_total - self._S_total_reset
        else:
            S_total = salt_store = 0.0
        return dict(cum_evap=self._cum_evap, cum_precip=self._cum_precip,
                    cum_surf_flash=self._cum_surf_flash,
                    closure_cum=self._cum_precip / (self._cum_evap + self._cum_surf_flash + 1e-30),
                    evap_inst=evap_inst, precip_inst=precip_inst, surf_flash_inst=flash_inst,
                    closure_inst=precip_inst / (evap_inst + flash_inst + 1e-30), vapor_mass=vapor_mass,
                    # #294-A: the q-clip terms [kg/m]. cum_qcap is water the PHYSICAL ceiling destroyed
                    # (was a silent ~15%-of-E leak while the ceiling was the Earth-era 0.5); cum_qfloor is
                    # water the 0-floor invented. atm_supply is the NET column source they complete:
                    #   d(vapor+cond+rain)/dt == atm_supply, exactly -- gate on atm_residual (a leak shows
                    # ONLY there; closure_cum -> 1 is a CLOSED-domain identity and cannot see a sink).
                    cum_qcap=self._cum_qcap, cum_qfloor=self._cum_qfloor,
                    cum_qadv=self._cum_qadv,
                    atm_supply=(self._cum_evap + self._cum_surf_flash - self._cum_precip
                                - self._cum_qcap + self._cum_qfloor + self._cum_qadv),
                    # NaN (not 0.0) when the baseline is unknowable -- a mid-window resume from a pre-#294-A
                    # checkpoint. NaN fails a `< tol` gate loudly instead of passing a fabricated zero;
                    # reset_water_budget starts a clean window. See load_state.
                    atm_store_change=((vapor_mass + cond_mass + rain_mass) - self._atm_total_reset
                                      if self._atm_total_reset is not None else float("nan")),
                    atm_residual=((vapor_mass + cond_mass + rain_mass) - self._atm_total_reset
                                  - (self._cum_evap + self._cum_surf_flash - self._cum_precip
                                     - self._cum_qcap + self._cum_qfloor + self._cum_qadv)
                                  if self._atm_total_reset is not None else float("nan")),
                    # stage-7 storage reservoirs (see the note above): rain_mass is 0.0 when
                    # prognostic_rain is OFF; cond_mass reports the (always-present) q_c reservoir.
                    rain_mass=rain_mass, cond_mass=cond_mass,
                    # #302 surface-water keys (identically 0.0 with the flag OFF):
                    W_total=W_total, cum_p_land=self._cum_p_land, cum_w_snap=self._cum_w_snap,
                    cum_boundary_influx=self._cum_w_edge,
                    cum_w_cryo=self._cum_w_cryo,
                    surf_store_change=surf_store,
                    surf_residual=surf_store - surf_supply,
                    surf_closure=(surf_store / surf_supply if surf_supply != 0.0 else 1.0),
                    # surf_salt keys (identically 0.0 with the flag OFF). salt_RESIDUAL is the
                    # gate; salt_closure is DISPLAY-ONLY -- it reads 1.0 whenever the accounted
                    # injections are 0 (closed domain), where a genuine leak shows only in
                    # salt_residual. Do not gate on salt_closure.
                    S_total=S_total, cum_s_edge=self._cum_s_edge,
                    cum_s_cryo=self._cum_s_cryo, cum_s_snap=self._cum_s_snap,
                    salt_store_change=salt_store,
                    # supply = edge injection + cryo top-up - snap-floor removals (the salt twin of
                    # surf_supply's -cum_w_snap term)
                    salt_residual=salt_store - self._cum_s_edge - self._cum_s_cryo + self._cum_s_snap,
                    salt_closure=(salt_store / (self._cum_s_edge + self._cum_s_cryo - self._cum_s_snap)
                                  if (self._cum_s_edge + self._cum_s_cryo - self._cum_s_snap) != 0.0
                                  else 1.0))

    def region_water(self, mask=None):
        """Column-water storage [kg/m] and surface supply rate [kg/m/s] integrated over a COLUMN MASK --
        the sub-domain twin of ``water_conservation``'s global integrals. ``mask`` is an (nx,) bool (None
        -> the warm region ``T_surf > 0 C``, the only part the eq-cycle's salt dynamics act on).

        WHY A SUB-REGION NEEDS A DIFFERENT GATE (the correction to a natural but wrong instinct):
        ``P/(E+flash) -> 1`` is a CLOSED-domain identity. A mask is OPEN -- the warm region is the
        evaporative SOURCE feeding the poleward upper branch, a net vapour exporter -- so its ratio
        converges to ``1 - export/(E+flash)``, a constant BELOW 1, never to 1. Gating a mask on 1 is
        unreachable by construction. The region-valid statement of the same intent ("the climatology the
        eq-cycle hooks is quasi-steady") is instead

            |d(store)/dt| / supply_rate  ->  0

        which holds transient or not and needs no closure assumption. The caller differences ``store``
        across gauge samples for the trend; ``supply_rate`` normalizes it.

        ``supply_rate`` prefers the persisted per-column climatology EMAs (``clim_accum``) over the
        instantaneous ``_evap``/``_surf_flash``, which are diurnally noisy -- a gauge read once per
        gazetteer must not alias the day cycle. Returns None when the mask is empty."""
        xp = get_xp(self.q)
        Ts = to_host(self.T_surf)
        m = (Ts > C.T_FREEZE) if mask is None else np.asarray(mask, bool)
        if not m.any():
            return None
        dxw = self._dx_w if self._dx_cell is not None else np.full(self.cfg.nx, self.dx)
        dxw = np.asarray(to_host(dxw), np.float64)
        # storage: sum_z (q + q_c + q_r) dm, per column, then area-weight the masked columns
        col = np.asarray(to_host(xp.sum(xp.clip(self.q, 0.0, th.Q_MAX) * self.dm[None, :], axis=1)), np.float64)
        col = col + np.asarray(to_host(xp.sum(self.q_c * self.dm[None, :], axis=1)), np.float64)
        if self.q_r is not None:
            col = col + np.asarray(to_host(xp.sum(self.q_r * self.dm[None, :], axis=1)), np.float64)
        E = self._clim_evap_slow if self._clim_evap_slow is not None else self._evap
        F = self._clim_flash_slow if self._clim_flash_slow is not None else self._surf_flash
        E = np.asarray(to_host(E), np.float64)
        F = np.asarray(to_host(F), np.float64) if F is not None else np.zeros_like(E)
        return dict(store=float(np.sum(col[m] * dxw[m])),
                    supply_rate=float(np.sum((E[m] + F[m]) * dxw[m])),
                    n_cols=int(m.sum()), extent=float(np.sum(dxw[m])),
                    smoothed=bool(self._clim_evap_slow is not None))

    @staticmethod
    def _accum_pending(pending, inc, xp):
        """IN-PLACE pending accumulation (#302 review finding 2, generalized to all budget pendings): add
        ``inc`` into a persistent 0-d fp64 buffer and return it. The 0.0 float is PROMOTED to the buffer on
        the first (eager warmup) pass -- warmup always precedes capture -- after which the same fixed buffer
        is written in place forever. Under CUDA-graph replay the recorded in-place add re-executes every
        replay, so the budget integrals genuinely accumulate across replays; the previous rebind idiom
        (``pending = pending + inc``) baked only the LAST replayed step's term (the Python rebind never
        re-executes -- the driver linchpin), which made the #302 ``surf_residual`` closure false-alarm under
        the captured production driver (W advances per replay; the cums did not). Eagerly value-identical:
        the same per-step terms are added in the same order."""
        if isinstance(pending, float):
            pending = xp.asarray(pending, xp.float64)          # promote once, eagerly (fixed address)
        xp.add(pending, inc, out=pending)
        return pending

    def _zero_pendings(self):
        """Zero every budget pending, IN PLACE where it has been promoted to a buffer (a rebind to 0.0
        would orphan the address a captured graph keeps accumulating into -- the replay would then add
        into a buffer nobody reads). Floats (never-promoted / capture_ready OFF) just reset to 0.0."""
        for nm in ("_cum_evap_pending", "_cum_precip_pending", "_cum_surf_flash_pending",
                   "_cum_p_land_pending", "_cum_w_snap_pending", "_cum_w_edge_pending",
                   "_cum_s_edge_pending", "_cum_w_cryo_pending", "_cum_s_cryo_pending",
                   "_cum_s_snap_pending", "_cum_qcap_pending", "_cum_qfloor_pending",
                   "_cum_qadv_pending"):
            p = getattr(self, nm)
            if isinstance(p, float):
                setattr(self, nm, 0.0)
            else:
                p[...] = 0.0

    def _flush_water_budget(self):
        """Capture-ready (Phase B(v)): fold the per-step PENDING budget area-sums (accumulated off the host-
        sync hot path) into the host cumulative totals, then zero the pending. A no-op when capture_ready is
        OFF (the pending stay 0.0 and _cum_* are accumulated per-step as shipped). Called by every consumer of
        _cum_* (water_conservation, state, reset_water_budget) so a checkpoint/diagnostic always sees the full
        total. Value-preserving: the per-step sum*dt terms are the SAME and folded in their accumulation order.
        ``float(...)`` matches the OFF path's per-step float() (np.sum -> python float, value-identical).
        The pendings are in-place 0-d buffers once promoted (``_accum_pending``) -- they accumulate across
        captured replays -- and are zeroed IN PLACE here (``_zero_pendings``) so the graph-baked addresses
        survive the flush. bool()/float() on a device 0-d sync; this runs at checkpoint cadence, off the
        hot path."""
        if (self._cum_evap_pending or self._cum_precip_pending or self._cum_surf_flash_pending
                or self._cum_p_land_pending or self._cum_w_snap_pending
                or self._cum_w_edge_pending or self._cum_s_edge_pending
                or self._cum_w_cryo_pending or self._cum_s_cryo_pending
                or self._cum_s_snap_pending
                or self._cum_qcap_pending or self._cum_qfloor_pending
                or self._cum_qadv_pending):
            self._cum_qcap += float(self._cum_qcap_pending)          # #294-A q-ceiling/floor accounting
            self._cum_qfloor += float(self._cum_qfloor_pending)
            self._cum_qadv += float(self._cum_qadv_pending)          # advection-clamp injection (PR #360)
            self._cum_evap += float(self._cum_evap_pending)
            self._cum_precip += float(self._cum_precip_pending)
            self._cum_surf_flash += float(self._cum_surf_flash_pending)
            self._cum_p_land += float(self._cum_p_land_pending)
            self._cum_w_snap += float(self._cum_w_snap_pending)
            self._cum_w_edge += float(self._cum_w_edge_pending)
            self._cum_s_edge += float(self._cum_s_edge_pending)
            self._cum_w_cryo += float(self._cum_w_cryo_pending)
            self._cum_s_cryo += float(self._cum_s_cryo_pending)
            self._cum_s_snap += float(self._cum_s_snap_pending)
            self._zero_pendings()

    def _flush_sim_time(self):
        """Capture-ready (Phase E): fold the on-device PENDING true-time accumulation (``_sim_time_pending``,
        accumulated on-device because ``dt`` is a 0-d device scalar on the capture path) into the host
        ``sim_time``, then zero the pending IN PLACE. A no-op when capture_ready is OFF (the pending stays the
        python 0.0 sentinel and ``sim_time`` is accumulated per-step as shipped). Called by ``state`` (the
        checkpoint) so a persisted/resumed run sees the full elapsed time; live readers use the
        NON-destructive ``true_sim_seconds`` instead. Value-preserving: the SAME per-step dt terms, summed in
        their accumulation order.

        The pending is zeroed IN PLACE (a 0-d buffer once promoted by ``_accum_pending``) -- NOT reassigned to
        ``0.0`` -- because the captured graph writes to this exact buffer address every replay; reassigning
        would ORPHAN it and freeze sim_time after the first flush (the batch-2+ freeze this method used to
        cause, exactly the ``_zero_pendings`` hazard). ``float(...)`` on a 0-d device buffer syncs; this runs
        at the flush cadence, off the hot path."""
        p = self._sim_time_pending
        self.sim_time += float(p)                  # float() of a 0-d buffer syncs; of the 0.0 sentinel is 0.0
        if isinstance(p, float):
            self._sim_time_pending = 0.0
        else:
            p[...] = 0.0                            # zero IN PLACE (keep the graph-baked address; see docstring)
        # the step-count twin (same freeze, same fix -- self.iter += 1 is host code, dead under replay):
        # rounding is exact (a sum of 1.0s is integer-representable in fp64 far past any run length).
        q = self._iter_pending
        self.iter += int(round(float(q)))
        if isinstance(q, float):
            self._iter_pending = 0.0
        else:
            q[...] = 0.0

    def true_sim_seconds(self):
        """Cumulative TRUE physical time [s] = Sigma(adaptive dt), correct on ALL paths and NON-DESTRUCTIVE:
        the folded host ``sim_time`` plus any device-pending delta not yet folded. Use this (not
        ``iter * dt_dyn``, which overcounts 3-12x when the CFL-adaptive dt runs below the cap) wherever a
        physics cadence or diagnostic needs elapsed physical time. Reading it does not flush."""
        return float(self.sim_time) + float(self._sim_time_pending)   # float() of a 0-d buffer syncs

    def true_sim_days(self):
        """``true_sim_seconds`` in physical days (Sigma dt / 86400)."""
        return self.true_sim_seconds() / 86400.0

    def flush_clamp_stats(self):
        """Read and reset the positivity-clamp instrumentation (implicit advection only).
        Returns (count, mass) where count is the number of cells clamped and mass is the
        total negative mixing ratio removed [kg/kg summed over cells]. Both zero when no
        clamp fired. Call between replay batches (the float() D2H is capture-illegal).

        NOT A WATER BUDGET, and easy to misread as one (PR #360 review): ``mass`` is an UNWEIGHTED
        Sigma min(q,0) [kg/kg], ~1e9x smaller than the physical ``INTEGRAL dm dx`` -- e.g. -7.6e-9 here
        where the injected water is 9.5 kg/m -- and it lives on a read+reset window unaligned with the
        budget's. For the conserved mass the clamp injects, use ``water_conservation``'s **cum_qadv**
        (cumulative, mass-weighted, reset-aligned); it is what makes ``atm_residual`` close."""
        count = float(self._clamp_count) if self._clamp_count else 0.0
        mass = float(self._clamp_mass) if self._clamp_mass else 0.0
        self._clamp_count = 0.0
        self._clamp_mass = 0.0
        return count, mass

    def reset_water_budget(self):
        """Zero the cumulative water-budget integrals (start a fresh measurement window after spin-up).
        Under surf_water_budget the W-closure baseline ``_W_total_reset`` is re-pinned to the CURRENT
        total, so the surface gate (surf_store_change == cum_p_land - cum_evap) measures the same window.
        The ATMOSPHERIC baseline ``_atm_total_reset`` is re-pinned the same way and unconditionally (#294-A),
        so ``atm_residual`` also measures this window -- without the re-pin it would carry the pre-reset
        storage as a permanent false leak."""
        self._cum_evap = 0.0
        self._cum_precip = 0.0
        self._cum_surf_flash = 0.0
        self._cum_qcap = 0.0            # #294-A: the q-ceiling/floor terms share the E/P measurement window
        self._cum_qfloor = 0.0
        self._cum_qadv = 0.0
        self._cum_p_land = 0.0
        self._cum_w_snap = 0.0
        self._cum_w_edge = 0.0
        self._cum_s_edge = 0.0
        self._cum_w_cryo = 0.0
        self._cum_s_cryo = 0.0
        self._cum_s_snap = 0.0
        self._zero_pendings()          # IN PLACE where promoted (a rebind would orphan graph-baked buffers)
        self._atm_total_reset = self._atm_total()   # #294-A: re-pin the atmospheric closure baseline too
        if self.W_surf is not None:
            xp = get_xp(self.W_surf)
            self._W_total_reset = float(xp.sum(self.W_surf * self._dx_w))   # off hot path: host pull is fine
        if self.S_dis is not None:
            xp = get_xp(self.S_dis)
            self._S_total_reset = float(xp.sum((self.S_dis + self.S_dep) * self._dx_w))

    # ------------------------------------------------------------------ f=0 wind-driven ocean (#5; M2b.5)
    def ocean_momentum_profile(self, nz_ocean=400, H=200.0):
        """TIER-2 analytic anchor (archive/M2B_SUBTASKS.md M2b.5): the f=0 wind-driven OCEAN momentum profile,
        coupled to the core's surface CROSS-TRACK wind. With no Coriolis there is no Ekman layer; the
        steady homogeneous-layer cubic ``wind_driven_ocean_profile`` (shared verbatim with Stage B, the
        canonical analytic solution) reverses at z = -H/3 with the lower two-thirds running AGAINST the
        wind (peak return -1/3 of the surface speed). On a non-rotating world this fixes the ocean
        circulation SHAPE, not the absolute speed (exactly as Stage B fixes the atmospheric shape) -- so
        the per-column profile is the analytic cubic SCALED by each column's surface wind ``u_surf``.

        The ocean here is MOMENTUM-ONLY: it does NOT impose a surface temperature. The cold-trap / cold-edge
        SST stays an OUTPUT of the coupled energy balance (#5: imposing it would re-pin the export and
        manufacture a spurious closure). Returns the analytic profile (z, u_norm, reversal depth/index) and
        the per-column scaled velocity ``u_ocean`` (nx, nz_ocean)."""
        u, _ = self.velocities()
        u_surf = u[1, :]                                             # surface cross-track wind per column
        z, u_norm, rev, ridx = wind_driven_ocean_profile(nz_ocean, H)
        u_ocean = u_surf[:, None] * u_norm[None, :]                 # (nx, nz_ocean), shape-preserving scale
        return dict(z=z, u_norm=u_norm, reversal_depth=rev, reversal_index=ridx,
                    u_ocean=u_ocean, u_surf_wind=u_surf)

    def _state_is_finite(self):
        """True iff the prognostic state is all-finite. The run loop checks this BEFORE checkpointing or
        continuing, so a CFL/closure blow-up (the failure mode of a long, expensive coupled run) aborts
        cheaply and **does not overwrite the last good checkpoint with NaN/Inf** -- the M3 cost guard
        Jake asked for (a wasted multi-hour GPU run is bad; a poisoned checkpoint that also kills the
        resume is worse)."""
        def _finite(a):                                          # per-field xp: psi/omega/b may be device
            xp = get_xp(a)
            return bool(xp.all(xp.isfinite(a)))
        return (_finite(self.T_atm) and _finite(self.T_surf) and _finite(self.psi)
                and _finite(self.omega) and _finite(self.q) and _finite(self.q_c)
                and (self.q_r is None or _finite(self.q_r)))

    def climatology_state(self):
        """Host per-column time-mean forcing from the persisted EQ-1 climatology EMAs (``clim_accum``),
        as a dict keyed by the ``eq_cycle.Climatology`` field names it feeds -- ``T_surf, q_bl, wind,
        E, P, surf_flash, u_surf`` (all ``(nx,)`` fp64; ``u_surf`` may be None on a no-dynamics core).
        Returns None -- signalling the eq-cycle to re-integrate the climatology the slow way -- when
        accumulation is OFF, the EMAs are unseeded (a cold core / legacy checkpoint), OR they have not yet
        MATURED: they must have accumulated at least ``clim_maturity_tau_mult * clim_ema_tau_days``
        e-folding times of true time since seeding, else the mean is still weighted toward its
        instantaneous, diurnal-phase-biased seed and using it would silently inject a phase-biased forcing
        (the maturity gate; ``clim_maturity_tau_mult=0`` disables it)."""
        if not self.cfg.clim_accum or self._clim_tsurf_slow is None or self._clim_seed_days is None:
            return None
        tau_days = self.cfg.clim_ema_tau_days * (g.CLOCK.T_day / 86400.0)   # diurnal cycles -> physical days
        age_days = float(self.true_sim_days()) - self._clim_seed_days       # eager (resume-time) call -> no D2H
        if age_days < self.cfg.clim_maturity_tau_mult * tau_days:
            return None
        _h = lambda a: None if a is None else np.asarray(to_host(a), dtype=np.float64)
        return dict(T_surf=_h(self._clim_tsurf_slow), q_bl=_h(self._clim_qbl_slow),
                    wind=_h(self._clim_wind_slow), E=_h(self._clim_evap_slow),
                    P=_h(self._clim_precip_slow), surf_flash=_h(self._clim_flash_slow),
                    u_surf=_h(self._clim_usurf_slow))

    def state(self):
        self._flush_water_budget()                                  # fold any capture-ready pending sums in
        self._flush_sim_time()                                      # fold the device-deferred true-time in (Phase E)
        # checkpoints are host arrays: bring the (possibly device-resident) dynamics state to host. For
        # device="cpu", to_host is a no-op alias (np.asarray of an ndarray), so this is unchanged on CPU.
        st = dict(T_surf=to_host(self.T_surf), T_atm=to_host(self.T_atm), omega=to_host(self.omega),
                  psi=to_host(self.psi), b=to_host(self.b), q=to_host(self.q), q_c=to_host(self.q_c),
                  iter=self.iter, cum_evap=self._cum_evap, cum_precip=self._cum_precip,
                  cum_surf_flash=self._cum_surf_flash,   # 0.0 unless surf_flash_boil (extra key, back-compat)
                  # #294-A: the q-clip terms + the atmospheric closure baseline, so atm_residual survives a
                  # checkpoint/resume round-trip (the surf twin of W_total_reset). Unconditional extra keys;
                  # load_state defaults them, so an OLD checkpoint still loads (its atm baseline is then
                  # re-pinned to the resumed state = a fresh window, which is the honest reading -- the
                  # pre-fix run never recorded what its clip destroyed).
                  cum_qcap=self._cum_qcap, cum_qfloor=self._cum_qfloor, cum_qadv=self._cum_qadv,
                  atm_total_reset=self._atm_total_reset,
                  sim_time=self.sim_time)
        if self.q_r is not None:
            # PROGNOSTIC RAIN (stage 7) state -- key present ONLY when the flag is ON (the W_surf/N_d
            # convention), so a flag-OFF checkpoint is byte-identical to the pre-flag format.
            st["q_r"] = to_host(self.q_r)
        if self.cfg.prognostic_condensate and self.cfg.bl_stratocumulus:
            # Chunk-4 BL/stratocumulus deck COVER: derived within a step, but the radiation reads it as a
            # cover FLOOR at the TOP of the next step -- before _bl_stratocumulus re-derives it -- so an
            # un-checkpointed resume radiates the first post-resume step with a 0 floor and the trajectory
            # diverges bitwise from the uninterrupted run (a reproducibility defect, not a bias: it
            # re-derives within one step). Key present ONLY when the closure is live (the same gate the
            # radiation uses), so a flag-OFF checkpoint keeps the pre-key byte format exactly.
            st["bl_deck_cover"] = to_host(self._bl_deck_cover)
        if self.W_surf is not None:
            # #302 surf_water_budget prognostic state (keys present ONLY when the flag is ON, so a
            # flag-OFF checkpoint is byte-identical to the pre-flag format). W_total_reset carries the
            # surface-closure baseline so the surf_residual gate survives a checkpoint/resume round-trip.
            st["W_surf"] = to_host(self.W_surf)
            st["cum_p_land"] = self._cum_p_land
            st["cum_w_snap"] = self._cum_w_snap
            st["W_total_reset"] = self._W_total_reset
            if self.cfg.surf_water_edge_source:
                st["cum_w_edge"] = self._cum_w_edge      # the pin's accounted injection (gate continuity)
            if self.cfg.surf_water_cryo_pin:
                st["cum_w_cryo"] = self._cum_w_cryo      # the cryo floor's accounted top-up (same convention)
        if self.S_dis is not None:
            # surf_salt prognostic state (keys only when ON -- the W_surf/N_d convention): dissolved +
            # crust fields and the salt-closure carry-overs, so the salt_residual gate survives a resume.
            st["S_dis"] = to_host(self.S_dis)
            st["S_dep"] = to_host(self.S_dep)
            st["S_total_reset"] = self._S_total_reset
            st["cum_s_edge"] = self._cum_s_edge
            st["cum_s_snap"] = self._cum_s_snap          # the salt snap floor's removals (gate continuity)
            if self.cfg.surf_water_cryo_pin:
                st["cum_s_cryo"] = self._cum_s_cryo      # (key only when the cryo pin is ON -- back-compat)
        if self.N_d is not None and self.cfg.salt_ccn_dynamic:
            # DYNAMIC salt-CCN: the live N_d is prognostic-at-cadence state (the relaxation memory).
            # Key present ONLY when the dynamic flag is ON (a static-salt N_d is config-derived, not
            # state), so flag-OFF checkpoints stay byte-identical to the pre-flag format.
            st["N_d"] = to_host(self.N_d)
            if self.cfg.diurnal_sun and getattr(self, "_salt_u3_slow", None) is not None:
                # the de-aliased emission wind key (#308 item 2): persisting the EMA keeps a resumed
                # run's first refreshes unbiased (re-seeding from the resume phase's instantaneous
                # wind would re-introduce exactly the phase-locked bias for ~one EMA tau).
                st["salt_u3_slow"] = to_host(self._salt_u3_slow)
        if self.cfg.clim_accum and self._clim_tsurf_slow is not None:
            # EQ-1 climatology EMAs: persist the matured forcing means so equilibrate_on_resume reads
            # them from the checkpoint instead of re-integrating the core. Keys present ONLY when
            # clim_accum is ON AND the EMAs are seeded, so a flag-OFF / cold checkpoint keeps the
            # pre-flag byte format. u_surf may be None (no-dynamics core) -> that key is simply absent.
            for _cn, _ck in (("_clim_tsurf_slow", "clim_tsurf_slow"), ("_clim_qbl_slow", "clim_qbl_slow"),
                             ("_clim_wind_slow", "clim_wind_slow"), ("_clim_evap_slow", "clim_evap_slow"),
                             ("_clim_precip_slow", "clim_precip_slow"), ("_clim_flash_slow", "clim_flash_slow"),
                             ("_clim_usurf_slow", "clim_usurf_slow")):
                _cv = getattr(self, _cn)
                if _cv is not None:
                    st[_ck] = to_host(_cv)
            if self._clim_seed_days is not None:
                st["clim_seed_days"] = float(self._clim_seed_days)   # the EMA maturity clock (age since seed)
        if self.cfg.diurnal_sun:
            # the TRUE diurnal time [s] -- the phase carrier: the device clock when the capture path
            # holds one (host sim_time is documented-frozen under replay -- _flush_sim_time NOTE),
            # else the host sim_time. load_state re-pins the phase from this, so a resumed run keeps
            # sun-forcing CONTINUITY instead of snapping to a driver's nominal step phase. Key present
            # only when the flag is ON (the OFF checkpoint format is unchanged).
            st["diurnal_clock"] = (float(to_host(self._diurnal_time_dev)[0])
                                   if self._diurnal_time_dev is not None else float(self.sim_time))
        return st

    def fingerprint(self):
        """JSON fingerprint of this core's config + the import-baked world forcing, for the checkpoint
        meta (``meta["ad_fingerprint"]``). Never raises -- a metadata failure returns ``""``."""
        return _ckfp.build_fingerprint(self.cfg)

    def load_state(self, st):
        # thermo state -> host numpy; dynamics state -> the configured device (self.xp). On CPU self.xp is
        # numpy, so this is identical to the pre-M3.1 np.asarray loads.
        # Cast every loaded prognostic field to the working dtype (self._cast): a no-op under fp64 (so a
        # legacy checkpoint loads byte-identically), and the documented cross-precision warm-start path
        # otherwise -- an fp64 checkpoint can seed an fp32 run (downcast) and vice-versa (upcast).
        # np.ASCONTIGUOUSARRAY (not asarray): an npz round-trip PRESERVES fortran_order, and a transposed/
        # interpolated seed (np.apply_along_axis over axis 0 returns an F-contiguous view) then loads
        # F-ordered -- which _cast/astype KEEP (order='K'), so the layout reaches the device buffers and the
        # fused RawKernels' flat i*nz+k indexing reads scrambled elements (the 2026-07-02 nx1500 "NaN" was
        # exactly this). Normalizing here makes every downstream consumer layout-safe; no-op on the driver's
        # own C-ordered checkpoints.
        self.T_surf = self._cast(np.ascontiguousarray(st["T_surf"]))
        self.T_atm = self._cast(np.ascontiguousarray(st["T_atm"]))
        self.omega = self.xp.asarray(self._cast(np.ascontiguousarray(st.get("omega", to_host(self.omega)))))
        self.psi = self.xp.asarray(self._cast(np.ascontiguousarray(st.get("psi", to_host(self.psi)))))
        self.b = self.xp.asarray(self._cast(np.ascontiguousarray(st.get("b", to_host(self.b)))))
        self.q = self._cast(np.ascontiguousarray(st.get("q", self.q)))
        # q_c (Chunk 1): an OLD checkpoint (pre-prognostic-condensate) has no "q_c" key -> load to ZERO, so
        # resuming a legacy run starts with an empty condensate reservoir (the documented back-compat path).
        qc = st.get("q_c")
        self.q_c = self._cast(np.ascontiguousarray(qc)) if qc is not None else np.zeros(
            (self.cfg.nx, self.cfg.nz), dtype=self._dtype)
        # q_r (stage 7): mirrors q_c's back-compat -- a checkpoint WITHOUT "q_r" (any pre-flag / flag-OFF
        # save) loads to ZERO when the flag is ON; a q_r-carrying checkpoint into a flag-OFF core ignores
        # it (q_r stays None -- the W_surf/N_d convention).
        if self.cfg.prognostic_rain:
            qr = st.get("q_r")
            self.q_r = self._cast(np.ascontiguousarray(qr)) if qr is not None else np.zeros(
                (self.cfg.nx, self.cfg.nz), dtype=self._dtype)
        # BL/stratocumulus deck COVER (the radiation's cover floor): restore it so the first post-resume
        # step radiates the deck the saved run had -> the resumed trajectory is BIT-IDENTICAL. Loaded at
        # the CHECKPOINT's own dtype (never self._cast): the live field is whatever
        # ``xp.where(mask, float, 0.0)`` yields -- fp64 on both backends today, independent of the working
        # precision -- so re-typing it here could itself break the parity this restores.
        # _place_thermo_on_device below moves it to the device on a GPU core, exactly as it does for q_c.
        #   * key ABSENT (any legacy / pre-key / closure-OFF checkpoint) -> zeros, i.e. the pre-fix
        #     behaviour: no floor on the first step, re-derived by that step's _bl_stratocumulus.
        #   * key present but WRONG-SHAPE -> zeros + a loud line, never a crash: this field is
        #     re-derived within one step, so a fresh start on it is cheap and self-healing. (Reachable
        #     from the UNIFORM regrid path, _warm_restart._regrid, which interpolates the fields it
        #     knows and passes the rest through. The stretched-x path, flatworld.regrid_x, classifies
        #     this field explicitly and interpolates it, so a seed from there arrives correctly shaped.)
        #   * key carried into a closure-OFF core -> ignored (the W_surf/N_d convention); the radiation
        #     does not read the floor there.
        if self.cfg.prognostic_condensate and self.cfg.bl_stratocumulus:
            bdc = st.get("bl_deck_cover")
            if bdc is not None and np.shape(bdc) != (self.cfg.nx, self.cfg.nz):
                print(f"[bl_stratocumulus] checkpoint bl_deck_cover shape {np.shape(bdc)} != core "
                      f"({self.cfg.nx}, {self.cfg.nz}) -- starting the deck cover at 0 (it re-derives "
                      f"in one step; the first post-resume radiation sees no cover floor).", flush=True)
                bdc = None
            self._bl_deck_cover = (np.ascontiguousarray(np.asarray(bdc))
                                   if bdc is not None else np.zeros((self.cfg.nx, self.cfg.nz)))
        self.iter = int(st.get("iter", 0))
        self._cum_evap = float(st.get("cum_evap", 0.0))
        self._cum_precip = float(st.get("cum_precip", 0.0))
        self._cum_surf_flash = float(st.get("cum_surf_flash", 0.0))   # 0.0 for a pre-flag checkpoint
        self._cum_p_land = float(st.get("cum_p_land", 0.0))           # 0.0 for a pre-#302 checkpoint
        self._cum_w_snap = float(st.get("cum_w_snap", 0.0))
        # #294-A. The clip terms default to 0.0 on a pre-fix checkpoint (it never recorded them).
        self._cum_qcap = float(st.get("cum_qcap", 0.0))
        self._cum_qfloor = float(st.get("cum_qfloor", 0.0))
        self._cum_qadv = float(st.get("cum_qadv", 0.0))
        self._cum_qcap_pending = 0.0
        self._cum_qfloor_pending = 0.0
        self._cum_qadv_pending = 0.0
        # The atmospheric baseline needs care on a PRE-FIX checkpoint (no ``atm_total_reset`` key):
        #   - cums all zero  -> the window starts HERE, so baseline == the just-loaded storage. Exact.
        #   - cums non-zero  -> a MID-window pre-fix resume. The storage at that window's start was never
        #     recorded and cannot be reconstructed, so the residual is UNKNOWABLE -> None -> reported as
        #     NaN. Do NOT re-pin to the current storage: that pairs a zero store-change with the carried
        #     cums and manufactures a ~1e-2 relative FALSE LEAK -- the gate would cry wolf on exactly the
        #     run15 resume this fix exists to serve. Zeroing the cums instead is not an option either:
        #     _W_total_reset IS carried (a #302 key), so the SURFACE closure would break in its place.
        #     A ``reset_water_budget`` (or the next eq-cycle injection) starts a clean, gateable window.
        atr = st.get("atm_total_reset")
        if atr is not None:
            self._atm_total_reset = float(atr)
        elif self._cum_evap == 0.0 and self._cum_precip == 0.0 and self._cum_surf_flash == 0.0:
            self._atm_total_reset = self._atm_total()
        else:
            self._atm_total_reset = None
        self._cum_evap_pending = 0.0       # capture-ready budget deferral: a resume starts with no pending,
        self._cum_precip_pending = 0.0     #   so a stale pre-load pending can't later flush into loaded totals
        self._cum_surf_flash_pending = 0.0 #   (rebinds to 0.0 floats are safe HERE: capture must always
        self._cum_p_land_pending = 0.0     #    happen AFTER load_state -- documented below -- so no graph
        self._cum_w_snap_pending = 0.0     #    holds the old promoted buffers)
        self._cum_w_edge_pending = 0.0     #   (edge-source / salt / cryo pendings: same reasoning)
        self._cum_s_edge_pending = 0.0
        self._cum_w_cryo_pending = 0.0
        self._cum_s_cryo_pending = 0.0
        self._cum_s_snap_pending = 0.0
        self._sim_time_pending = 0.0       #   (likewise the Phase-E device-deferred true-time pending)
        self._iter_pending = 0.0           #   (and its step-count twin)
        # --- #302 surf_water_budget state: load W if the checkpoint carries it; otherwise (ANY pre-flag
        #     checkpoint -- either attractor) run the documented init path: W from the LEGACY wetness
        #     diagnosis of the JUST-LOADED T_surf (stamped in the log by _init_surf_water). The closure
        #     baseline W_total_reset re-pins to the post-load total when absent, so the surf_residual gate
        #     measures the post-load window (consistent with the loaded cum_* which then start together).
        #     A checkpoint that carries W into a flag-OFF core ignores it (W_surf stays None; documented).
        if self.cfg.surf_water_budget:
            wk = st.get("W_surf")
            if wk is not None:
                W = np.asarray(np.ascontiguousarray(wk), np.float64)   # fp64 ALWAYS (never the working dtype)
                self.W_surf = to_device(W, self.xp) if self._on_device else W
                if self.cfg.surf_salt:
                    # surf_salt load ladder (fp64 ALWAYS, the W discipline): (a) a surf_salt checkpoint
                    # carries S_dis/S_dep; (b) an INTERIM single-field "S_surf" checkpoint (the manual
                    # runs 1-11 format) loads as all-dissolved (S_dep=0; the first saturation adjustment
                    # re-partitions it); (c) any older checkpoint falls back to the ocean-baseline init
                    # S_dis = salinity0*W, S_dep = 0. A salt-carrying checkpoint into a surf_salt-OFF
                    # core is ignored (S_* stay None -- the W_surf/N_d convention).
                    sk, zk = st.get("S_dis"), st.get("S_dep")
                    if sk is None:
                        sk = st.get("S_surf")                          # (b) the interim single-field format
                    if sk is not None:
                        S = np.asarray(np.ascontiguousarray(sk), np.float64)
                    else:
                        S = W * self.cfg.surf_salt_salinity0           # (c) legacy fallback
                    Z = (np.asarray(np.ascontiguousarray(zk), np.float64) if zk is not None
                         else np.zeros_like(S))
                    self.S_dis = to_device(S, self.xp) if self._on_device else S
                    self.S_dep = to_device(Z, self.xp) if self._on_device else Z
                    # GUARD (self-healing resume): repair a physically-impossible halite crust carried
                    #   by the checkpoint -- the surf-salt mass-creation runaway (run11/run12 S_dep ->
                    #   ~4e24 kg/m^2, which makes the eq_cycle u_grav ~1.4e11 m/s and destroys the salt
                    #   budget). Cells above the detection ceiling SURF_SALT_SDEP_PHYS_MAX are RESET to
                    #   the saturation scale SURF_SALT_SDEP_SATURATION (a physical crust restart, not the
                    #   ceiling) and LOGGED loudly (no silent repair). No-op / byte-identical on a clean
                    #   checkpoint. Runs BEFORE the S_total_reset baseline below so the reset measures the
                    #   repaired state (and the salt cum-trackers are zeroed there for a consistent
                    #   fresh salt-budget window). The upstream mass-creation is fixed (the W<0 clamp,
                    #   step 2a); this heals checkpoints written before that fix.
                    if SURF_SALT_SDEP_PHYS_MAX > 0.0:
                        _xpz = get_xp(self.S_dep)
                        _old_max = float(_xpz.max(self.S_dep))
                        _rep, _removed, _nrep = repair_runaway_crust(
                            self.S_dep, self._dx_w, SURF_SALT_SDEP_PHYS_MAX,
                            SURF_SALT_SDEP_SATURATION, _xpz)
                        self._crust_repaired = _nrep > 0
                        if self._crust_repaired:
                            self.S_dep = _rep
                            print(f"[surf_salt] GUARD: reset {_nrep} runaway-crust cell(s) "
                                  f"(S_dep max {_old_max:.3e} -> saturation scale "
                                  f"{SURF_SALT_SDEP_SATURATION:.1e} kg/m^2; detect > "
                                  f"{SURF_SALT_SDEP_PHYS_MAX:.1e}); removed {_removed:.3e} kg/m of "
                                  f"spurious salt; salt budget re-baselined (surf-salt mass-creation "
                                  f"guard).", flush=True)
                # Stamp the LOADED water so the constructor's earlier "(construction)" legacy-diagnosis
                # line can't be misread as the run's actual state (it is always printed first and is
                # always REPLACED here when the checkpoint carries W -- a silent replace confused a
                # real launch, 2026-07-09). Off the hot path (load only).
                _n_wet = int(np.sum(W > 0.0))
                _s_src = ("" if not self.cfg.surf_salt else
                          "; S_dis/S_dep from checkpoint" if st.get("S_dis") is not None else
                          "; S_dis from interim S_surf (all-dissolved)" if st.get("S_surf") is not None
                          else f"; S_dis = {self.cfg.surf_salt_salinity0:g}*W (legacy fallback)")
                print(f"[surf_water_budget] W_surf LOADED from checkpoint ({_n_wet}/{self.cfg.nx} "
                      f"columns wet -- replaces the construction-time legacy diagnosis){_s_src}",
                      flush=True)
            else:
                self._init_surf_water("load_state: no W_surf in checkpoint -> legacy-wetness diagnosis")
            wt = st.get("W_total_reset")
            xpw = get_xp(self.W_surf)
            self._W_total_reset = (float(wt) if wt is not None
                                   else float(xpw.sum(self.W_surf * self._dx_w)))
            self._cum_w_edge = float(st.get("cum_w_edge", 0.0))        # 0.0 for any pre-edge-source ckpt
            self._cum_w_cryo = float(st.get("cum_w_cryo", 0.0))        # 0.0 for any pre-cryo-pin ckpt
            if self.cfg.surf_salt:
                stt = st.get("S_total_reset")
                xps = get_xp(self.S_dis)
                _repaired = getattr(self, "_crust_repaired", False)
                # recompute the baseline from the CURRENT total when the checkpoint carries none OR the
                # crust guard just repaired a runaway (the stored reset includes the removed spurious
                # salt, so keeping it would break the salt_residual gate).
                self._S_total_reset = (float(stt)
                                       if (stt is not None and not _repaired)
                                       else float(xps.sum((self.S_dis + self.S_dep) * self._dx_w)))
                if _repaired:
                    # FRESH salt-budget window after a repair. The salt_residual gate is
                    # (S_total − S_total_reset) − cum_s_edge − cum_s_cryo + cum_s_snap; we just re-pinned
                    # S_total_reset to the repaired total (so the first term is 0), and the LOADED
                    # cum_s_* describe the CORRUPT pre-repair trajectory, so keeping them would leave the
                    # gate at −(cum_s_edge+cum_s_cryo−cum_s_snap) ≠ 0. Zero them: the salt accounting
                    # restarts consistently (0 == 0) from the repaired state. Water trackers are
                    # untouched (the repair only moved salt).
                    self._cum_s_edge = self._cum_s_cryo = self._cum_s_snap = 0.0
                else:
                    self._cum_s_edge = float(st.get("cum_s_edge", 0.0))
                    self._cum_s_cryo = float(st.get("cum_s_cryo", 0.0))   # 0.0 for any pre-cryo-pin ckpt
                    self._cum_s_snap = float(st.get("cum_s_snap", 0.0))   # 0.0 for any pre-salt-snap ckpt
        # --- DYNAMIC salt-CCN state: restore the live N_d when the checkpoint carries it (present only
        #     when the flag was ON at save). Reassignment is safe here -- load_state is documented
        #     pre-capture (see the Phase-E note below) and _place_thermo_on_device re-places the field;
        #     the identity-tracked warm-factor cache catches the swap (cleared anyway, belt-and-braces).
        #     A checkpoint WITHOUT "N_d" (any pre-flag / static-salt checkpoint) keeps the constructed
        #     field and the first refresh_salt_ccn() initializes from the loaded state; a checkpoint that
        #     carries N_d into a dynamic-OFF core ignores it (mirrors the W_surf convention).
        if self.cfg.salt_ccn and self.cfg.salt_ccn_dynamic:
            ndk = st.get("N_d")
            if ndk is not None:
                N = np.ascontiguousarray(np.asarray(ndk)).astype(self._dtype)
                if N.shape != (self.cfg.nx,) or not np.all(np.isfinite(N)) or float(N.min()) <= 0.0:
                    raise ValueError(f"checkpoint N_d: must be (nx={self.cfg.nx},), finite and > 0 "
                                     f"(shape={N.shape}).")
                self.N_d = N
                self._micro_warm = {}
                self._salt_dyn_initialized = True
            u3k = st.get("salt_u3_slow")
            # the de-aliased emission wind-key EMA (#308 item 2): restore when present; an OLD (pre-
            # EMA) checkpoint leaves it None -- the first post-resume refresh then reads the
            # instantaneous wind once and the next step re-seeds the EMA (documented transient).
            self._salt_u3_slow = (np.ascontiguousarray(np.asarray(u3k, dtype=np.float64))
                                  if u3k is not None else None)
        # EQ-1 climatology EMAs: restore the matured forcing means when present. A legacy / cold /
        # flag-OFF checkpoint carries no keys -> leave None; step() re-seeds lazily and
        # equilibrate_on_resume falls back to re-integration until they mature. Host arrays here;
        # _place_thermo_on_device (below) moves them to the device on a GPU core.
        if self.cfg.clim_accum:
            for _cn, _ck in (("_clim_tsurf_slow", "clim_tsurf_slow"), ("_clim_qbl_slow", "clim_qbl_slow"),
                             ("_clim_wind_slow", "clim_wind_slow"), ("_clim_evap_slow", "clim_evap_slow"),
                             ("_clim_precip_slow", "clim_precip_slow"), ("_clim_flash_slow", "clim_flash_slow"),
                             ("_clim_usurf_slow", "clim_usurf_slow")):
                _cvk = st.get(_ck)
                setattr(self, _cn, (np.ascontiguousarray(np.asarray(_cvk, dtype=np.float64))
                                    if _cvk is not None else None))
            _sd = st.get("clim_seed_days")            # maturity clock: restore, else None -> re-seeds on next step
            self._clim_seed_days = float(_sd) if _sd is not None else None
        self.sim_time = float(st.get("sim_time", 0.0))   # 0.0 for a legacy checkpoint (run() back-fills an estimate)
        # PHASE D: the thermo state was loaded as HOST (np.asarray) -> re-place it on device so the coupled
        # step stays device-resident after a resume (the fixed grid arrays are already device -> to_device no-op).
        # Idempotent on CPU (no-op). NOTE for Phase E: a captured graph records buffer ADDRESSES, so do NOT
        # capture across load_state -- it rebinds T_atm/q/T_surf/q_c to fresh device buffers (new addresses).
        # DIURNAL SUN (4.1/4.2): re-seed the slow-wetness keying state from the LOADED surface (the seed
        # state's classification -- finding 2), host-built here so the placement below moves it; the E_w
        # EMA restarts lazily (first step re-seeds it from the fresh cache).
        if self.cfg.diurnal_sun:
            T_host = np.asarray(to_host(self.T_surf), dtype=np.float64)
            self._T_surf_slow = T_host.copy()
            self._wetness_frozen = (np.asarray(wetness_of_Tsurf(T_host))
                                    if self.cfg.diurnal_wetness_freeze else None)
            self._E_w_slow = None
        if self._on_device:
            self._place_thermo_on_device()
        if self._dyn_on_device:
            self._place_dynamics_on_device()   # Phase E: H2D-free dynamics substep (capture prerequisite)
        # DIURNAL SUN (4.1): re-pin the held row buffers + the device diurnal CLOCK to the RESTORED
        # true diurnal time -- the checkpoint's ``diurnal_clock`` when present (sun-forcing CONTINUITY
        # across a resume; on the capture path sim_time is frozen under replay, so the saved clock is
        # the only true record), else the restored sim_time (pre-clock checkpoints; exact on CPU).
        # Captured-replay drivers re-pinning from nominal step bookkeeping (set_diurnal_phase_index)
        # should do so only when no saved clock exists. No-op when OFF.
        dclk = st.get("diurnal_clock")
        self._sync_diurnal_now(t_abs=(float(dclk) if dclk is not None else None))

    # ------------------------------------------------------------------ run loop (mirrors StageA.run)
    def run(self, verbose=True):
        """Drive the core to a steady state with the shared convergence/guard/checkpoint harness.

        The steady-state diagnostic is the per-step max of THREE drifts, each below ``tol`` for
        ``patience`` checks: the CIRCULATION relative drift max|dpsi|/max|psi| (the M2b.2 dynamics), the
        ATMOSPHERIC-temperature drift max|dT_atm| (the M2b.3 column physics -- the field that relaxes in
        the psi==0 RCE limit, where T_surf is held), and the surface-temperature drift max|dT_surf| (live
        once the surface balance lands in M2b.4); all over theta <= conv_theta_max. With no buoyancy
        forcing AND no thermo the flow is quiescent (psi == 0 -> circulation drift defined 0) and the
        harness still converges at zero drift -- the skeleton/dynamics self-test.

        ROBUSTNESS (M3 cost guards): (1) a NON-FINITE state (CFL/closure blow-up) aborts the run
        immediately, **without** checkpointing -- so a blown-up coupled run neither burns the rest of the
        guard budget nor poisons the last good checkpoint (it raises ``FloatingPointError`` so a sweep
        driver records the failed config and moves on). (2) Checkpoints are only ever written from a finite
        state. CAVEAT for the turbulent M3 regime: the steady-state metric here is a POINTWISE drift, which
        a deeply-turbulent core (M2a Ra~3.7e21) may never drive below ``tol`` because eddy internal
        variability keeps ``max|dpsi|`` finite -- a long turbulent run should converge a TIME-AVERAGED
        T(theta) and report the internal-variability noise floor (archive/UNIFIED_CORE_ROADMAP.md M3),
        not wait on this pointwise drift; the ``max_sim_days`` TRUE-time budget and the
        ``max_iter``/``max_seconds`` guards bound the wait either way."""
        c = self.cfg
        seed_legacy_iter = 0               # >0 -> back-fill sim_time for a pre-feature checkpoint (no sim_time)
        if c.checkpoint_path:
            st, meta = conv.load_checkpoint(c.checkpoint_path)
            if st is not None:
                self.load_state(st)
                _ckfp.validate_resume(meta, c, mode=c.resume_validate, context="StageAD.run")
                if self.iter > 0 and self.sim_time == 0.0:
                    seed_legacy_iter = self.iter   # legacy resume: estimate past true-time from the first batch's dt
                if verbose:
                    print(f"[StageAD] resumed from checkpoint at iter {self.iter}")
        conv_mask = (np.ones(c.nx, bool) if c.conv_theta_max is None
                     else self.theta_deg <= c.conv_theta_max)
        mon = conv.ConvergenceMonitor(tol=c.tol, patience=c.patience, max_iter=c.max_iter,
                                      max_seconds=c.max_seconds, scale=1.0,
                                      name="max(d|psi|/|psi|, |dTatm|, |dTsurf|)/step", absolute=True)
        T_prev = self.T_surf.copy()
        Ta_prev = self.T_atm.copy()
        psi_prev = self.psi.copy()
        xpp = get_xp(self.psi)              # numpy or CuPy for the circulation-drift reductions (psi is device on GPU)
        xpt = get_xp(self.T_surf)           # thermo state module (device on GPU) for the T-drift reductions
        conv_mask_t = xpt.asarray(conv_mask)  # conv_mask onto the thermo module (Phase D: T_surf/T_atm device on GPU)
        while True:
            for _ in range(c.check_every):
                self.step()
            if seed_legacy_iter:
                # One-time back-fill: a pre-feature checkpoint resumes with sim_time=0 but iter>0. Estimate the
                # un-recorded past true-time as iter_at_resume * (mean dt observed this first batch). Approximate
                # (early dt was nearer the cap), but it lands the legacy run on the true-time axis well enough.
                mean_dt = self.sim_time / max(c.check_every, 1)
                self.sim_time += seed_legacy_iter * mean_dt
                seed_legacy_iter = 0
            if not self._state_is_finite():
                # blow-up: do NOT checkpoint (preserve the last good state on disk) and abort loudly so an
                # M3 sweep records this config as failed rather than silently burning to the guard limit.
                msg = (f"NON-FINITE state at iter {self.iter} (CFL/closure blow-up); aborting WITHOUT "
                       f"checkpoint to preserve the last good resume state.")
                if verbose:
                    print(f"[StageAD] {msg}")
                raise FloatingPointError(msg)
            drift_T = float(xpt.max(xpt.abs((self.T_surf - T_prev)[conv_mask_t]))) / c.check_every
            drift_Ta = float(xpt.max(xpt.abs((self.T_atm - Ta_prev)[conv_mask_t]))) / c.check_every
            denom = max(float(xpp.abs(self.psi).max()), 1e-30)
            drift_psi = float(xpp.max(xpp.abs(self.psi - psi_prev))) / denom / c.check_every
            drift = max(drift_T, drift_Ta, drift_psi)
            T_prev = self.T_surf.copy()
            Ta_prev = self.T_atm.copy()
            psi_prev = self.psi.copy()
            stop = mon.update(self.iter, drift)
            if c.max_sim_days and self.sim_time >= c.max_sim_days * 86400.0:
                stop = True                        # TRUE-time budget reached (the honest fixed-length stop)
                if not mon.stop_reason:
                    mon.stop_reason = (f"reached max_sim_days={c.max_sim_days:g} "
                                       f"(true {self.sim_time / 86400.0:.1f} d, iter {self.iter})")
            if verbose and (self.iter % (c.check_every * 10) == 0 or stop):
                print(f"[StageAD] it={self.iter:>8} <Tsurf>={float(xpt.mean(self.T_surf)) - C.T_FREEZE:7.2f}C "
                      f"max|psi|={float(xpp.abs(self.psi).max()):.2e} dpsi={drift_psi:.2e} "
                      f"dTa={drift_Ta:.2e} dT={drift_T:.2e}K/step")
            # checkpoint once per checkpoint_every WINDOW (and only from a finite state): iter advances by
            # check_every each loop, so an ``== 0`` test silently never fires unless check_every divides
            # checkpoint_every -- ``< check_every`` fires on the first check crossing the window boundary,
            # robust to any (check_every, checkpoint_every) pairing. Critical for the long M3 run: a mistuned
            # cadence must not leave an expensive guarded run with NO periodic resume point.
            if c.checkpoint_path and self.iter % c.checkpoint_every < c.check_every and self._state_is_finite():
                conv.save_checkpoint(c.checkpoint_path, self.state(),
                                     {"iter": self.iter, "sim_time": self.sim_time,
                                      "ad_fingerprint": self.fingerprint()})
            if stop:
                break
        if c.checkpoint_path and self._state_is_finite():
            conv.save_checkpoint(c.checkpoint_path, self.state(),
                                 {"iter": self.iter, "sim_time": self.sim_time,
                                  "ad_fingerprint": self.fingerprint()})
        if verbose:
            print(f"[StageAD] STOP: {mon.stop_reason}  ({mon.elapsed:.1f}s, {self.iter} iters)")
        return mon.summary()


# ======================================================================================
# TIER-3 SOLVER BENCHMARK (M2b.2): Rayleigh-Benard Nu ~ Ra^beta against the LITERATURE exponent.
#
# This validates the *dynamical solver* independent of the doc (archive/M2B_SUBTASKS.md / UNIFIED_CORE_ROADMAP
# "Validation" tier 3): run the SAME kernel functions the climate core uses
# (build_anelastic_poisson / velocities_from_psi / vorticity_tendency / smagorinsky_viscosity /
# _upwind / _div_grad) in a classic 2-D Rayleigh-Benard cell and recover the known scaling exponent.
# If beta does NOT land near the literature exponent for the chosen wall BC (~0.28-0.29 no-slip; the
# stress-free "ultimate" ~0.417 -- see ``rayleigh_benard_run``) the solver has a formulation bug -- a
# stop-and-report point (archive/M2B_SUBTASKS.md tier 3; measured record: archive/M2B2_RESULTS.md).
# ======================================================================================
def rayleigh_benard_run(Ra, Pr=1.0, nx=160, nz=80, aspect=2.0, t_end=120.0, t_avg_start=60.0,
                        cfl=0.4, seed=0, slip="free", max_steps=800000, return_fields=False):
    """Run the SHARED anelastic dynamical kernel in a Boussinesq Rayleigh-Benard box; return the
    Nusselt number Nu.

    Nondimensional (depth H = buoyancy jump dB = free-fall time = 1):  nu = sqrt(Pr/Ra),
    kappa = sqrt(1/(Pr*Ra)), so Ra = dB*H^3/(nu*kappa) and Pr = nu/kappa. Boussinesq is the rho_ref==1
    LIMIT of the anelastic core: the streamfunction Poisson becomes the standard Laplacian and the
    vortex-stretching term vanishes, so this exercises EXACTLY the kernel the climate core uses. DNS:
    C_s = 0, so the Smagorinsky eddy viscosity collapses to the molecular nu (no closure -- the scaling
    must come from the resolved dynamics).

    Box: hot plate b=1 at the bottom (j=0), cold b=0 at the top (j=nz-1); insulating sides (db/dx=0);
    psi=0 closed box. ``slip`` selects the wall vorticity BC -- the SCALING EXPONENT DEPENDS ON IT, and
    the solver must recover the right LITERATURE value for EACH:
      * ``slip="free"`` (stress-free, omega=0 on the walls) -- MATCHES THE CORE (#5). 2-D stress-free RB
        reaches the rigorous "ultimate" upper bound Nu ~ Ra^(5/12) ~ Ra^0.417 (Whitehead & Doering,
        PRL 2011) -- a STEEPER exponent than no-slip.
      * ``slip="no"`` (no-slip, Thom's wall vorticity omega_wall = 2*psi_adjacent/dn^2 from psi=0 and
        d psi/dn=0) -- the CLASSICAL 2-D RB benchmark, Nu ~ Ra^(~2/7) ~ Ra^0.28-0.29 (the band the
        roadmap quotes). Recovering BOTH exponents from the same kernel is the conclusive solver check.
    The buoyancy is PROGNOSTIC (advection-diffusion -- the thermal field whose rising/sinking plumes
    carry the heat), advanced with the same _upwind/_div_grad. Nu is the exact volume-+-time-averaged
    steady-state relation Nu = 1 + <w b>_V / kappa, time-averaged over the window t >= ``t_avg_start``.

    INTEGRATE TO A FIXED PHYSICAL TIME (free-fall units), NOT a fixed step count: the buoyancy
    instability grows from the seed over O(10) free-fall times, and the CFL-limited dt SHRINKS with
    resolution -- so a fixed step count would give finer grids LESS physical time and leave them
    spuriously conductive (Nu~1), masquerading as a resolution effect. ``t_end`` must exceed the
    spin-up; the average window [t_avg_start, t_end] must span many turnovers for a stable turbulent
    mean. ``max_steps`` guards a stalled run. The exponent is resolution-sensitive (the thermal
    boundary layer thins as ~1/(2 Nu), so the high-Ra end under-resolves and STEEPENS beta) -- choose
    nz to keep >~3 cells in the BL at the largest Ra (``_m2b2_rb_benchmark.py`` runs the resolution
    check + the production sweep). Returns a dict (Ra, Pr, grid, slip, Nu, Nu_std, nu, kappa, n_steps,
    t[, fields])."""
    nu = float(np.sqrt(Pr / Ra)); kappa = float(np.sqrt(1.0 / (Pr * Ra)))
    H = 1.0
    z = np.linspace(0.0, H, nz); dz = float(z[1] - z[0])
    x = np.linspace(0.0, aspect, nx); dx = float(x[1] - x[0])
    rho_ref = np.ones(nz)                                       # Boussinesq: constant reference density
    beta_inv = np.ones(nz)                                      # 1/rho_ref
    dlnrho = np.zeros(nz)                                       # no stretching in the Boussinesq limit
    lu = build_anelastic_poisson(rho_ref, nx, nz, dx, dz)       # == the standard Laplacian here
    b = (1.0 - z[:, None] / H) * np.ones((nz, nx))             # linear conduction profile ...
    rng = np.random.default_rng(seed)
    b[1:-1, 1:-1] += 5.0e-2 * rng.standard_normal((nz - 2, nx - 2))   # ... + a seed perturbation
    omega = np.zeros((nz, nx))
    inv_h2 = 1.0 / dx**2 + 1.0 / dz**2

    def apply_wall_vorticity(om, ps):
        if slip == "free":                                      # stress-free: omega = 0
            om[0, :] = 0.0; om[-1, :] = 0.0; om[:, 0] = 0.0; om[:, -1] = 0.0
        elif slip == "no":                                      # Thom: omega = 2 psi_adjacent / dn^2
            om[0, :] = 2.0 * ps[1, :] / dz**2; om[-1, :] = 2.0 * ps[-2, :] / dz**2
            om[:, 0] = 2.0 * ps[:, 1] / dx**2; om[:, -1] = 2.0 * ps[:, -2] / dx**2
        else:
            raise ValueError(f"slip must be 'free' or 'no', got {slip!r}")
        return om

    t, nstep = 0.0, 0
    nus = []
    while t < t_end and nstep < max_steps:
        # psi=0 on all walls ALWAYS (no normal flow). The Poisson uses identity wall rows
        # (psi_wall = RHS_wall), so the wall RHS must be 0 -- the Thom no-slip wall VORTICITY lives in
        # ``omega`` (for the diffusion stencil), NOT in the streamfunction BC.
        om_poisson = omega.copy()
        om_poisson[0, :] = 0.0; om_poisson[-1, :] = 0.0; om_poisson[:, 0] = 0.0; om_poisson[:, -1] = 0.0
        psi = lu.solve(om_poisson.ravel()).reshape(nz, nx)
        u, w = velocities_from_psi(psi, beta_inv, dx, dz)
        nu_t = smagorinsky_viscosity(u, w, dx, dz, 0.0, nu)     # C_s=0 -> nu_t == molecular nu (DNS)
        dt = min(cfl * dx / (np.abs(u).max() + 1e-30),
                 cfl * dz / (np.abs(w).max() + 1e-30),
                 cfl * 0.5 / (nu * inv_h2), cfl * 0.5 / (kappa * inv_h2))
        omega = omega + dt * vorticity_tendency(omega, b, u, w, dlnrho, nu_t, dx, dz)
        b = b + dt * (-_upwind(b, u, w, dx, dz) + _div_grad(b, kappa, dx, dz))
        omega = apply_wall_vorticity(omega, psi)               # free-slip or no-slip (Thom) wall BC
        b[0, :] = 1.0; b[-1, :] = 0.0                           # hot/cold plates (fixed b)
        b[:, 0] = b[:, 1]; b[:, -1] = b[:, -2]                  # insulating sides (db/dx = 0)
        t += dt; nstep += 1
        if t >= t_avg_start:
            nus.append(1.0 + float(np.mean(w * b)) / kappa)     # Nu = 1 + <w b>_V / kappa (exact rel.)
    if not nus:                                                 # t_avg_start never reached (short run)
        nus = [1.0 + float(np.mean(w * b)) / kappa]
    out = dict(Ra=float(Ra), Pr=Pr, nx=nx, nz=nz, aspect=aspect, slip=slip, n_steps=nstep, t=t,
               Nu=float(np.mean(nus)), Nu_std=float(np.std(nus)), nu=nu, kappa=kappa)
    if return_fields:
        out.update(psi=psi, u=u, w=w, b=b, omega=omega, x=x, z=z)
    return out


def nu_ra_scaling(Ra_list, fit_from=0, **kw):
    """Sweep ``rayleigh_benard_run`` over ``Ra_list`` and least-squares fit log(Nu) = beta*log(Ra) + c.
    ``fit_from`` drops the leading (weakly-supercritical) points from the fit if needed. Returns a dict
    with Ra, Nu, Nu_std (arrays), ``beta`` and its standard error ``beta_err``, the ``prefactor``, and
    the raw ``runs``."""
    runs = [rayleigh_benard_run(Ra, **kw) for Ra in Ra_list]
    Ra = np.array([r["Ra"] for r in runs]); Nu = np.array([r["Nu"] for r in runs])
    xl = np.log(Ra[fit_from:]); yl = np.log(Nu[fit_from:])
    A = np.vstack([xl, np.ones_like(xl)]).T
    coef, *_ = np.linalg.lstsq(A, yl, rcond=None)
    beta, lnc = float(coef[0]), float(coef[1])
    resid = yl - A @ coef
    dof = max(xl.size - 2, 1)
    sxx = float(np.sum((xl - xl.mean())**2))
    beta_err = float(np.sqrt(float(np.sum(resid**2)) / dof / sxx)) if sxx > 0 else float("nan")
    return dict(Ra=Ra, Nu=Nu, Nu_std=np.array([r["Nu_std"] for r in runs]),
                beta=beta, beta_err=beta_err, prefactor=float(np.exp(lnc)), runs=runs)
