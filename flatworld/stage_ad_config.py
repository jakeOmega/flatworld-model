"""
flatworld.stage_ad_config -- ``StageADConfig``, the AD core's config dataclass (#289 Part 2 split).

The fused core's configuration dataclass plus the ``_env_*`` env-knob helpers its default
factories read, MOVED VERBATIM out of ``stage_ad_core.py`` (issue #289 Part 2 -- a move-only
split; zero token changes inside any def or class body). This file is pure declaration + knob
documentation: every field, its default, its ``COSMO_*`` env override, and the results doc that
set it. Construction-time validation lives in ``StageADCore.__init__`` (``stage_ad_core.py``),
not here.

Back-compat: ``stage_ad_core`` re-exports ``StageADConfig`` and the ``_env_*`` helpers, so
existing ``from flatworld.stage_ad_core import StageADConfig`` references keep working; new code
should import from this module.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from . import constants as C
from . import radiation as rad


# ======================================================================================
# Env-knob helpers (issue #19 clouds + parallel sweeps): the cloud on/off switch and the swept cloud
# scales default from COSMO_CLOUD* environment variables, so cloud-vs-clear and parameter-sweep runs can
# run side by side -- each in its own process on its own core -- WITHOUT editing code (mirrors the
# COSMO_L_SUN / COSMO_I0_SCALE raise-suns pattern). Read via dataclass default_factory so EACH
# StageADConfig() picks up the CURRENT environment; unset -> the documented per-field defaults (clouds
# default ON -- the all-on accurate model, ALLON_CAMPAIGN.md; env COSMO_CLOUDS=0 for the clear-sky limit).
# ======================================================================================
def _env_flag(name, default):
    """Boolean env knob: '1'/'true'/'on'/'yes' -> True, '0'/'false'/'off'/'no' -> False, unset -> default.

    Registers as '1'/'0' (matching ``constants._env_flag``, and inside the closed boolean vocabulary
    ``ckpt_world._env_value_equal`` compares through) -- see ``_env_float`` for why every reader here must.
    The world-defining ``COSMO_CLOUDS`` is read through this one, so a pin-at-default export of it would
    otherwise have re-run the #440 chain-block against every checkpoint predating the export."""
    _register(name, "1" if default else "0")
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "on", "yes")


def _register(name, value):
    """Record a knob's CODE DEFAULT in the shared ``constants.ENV_DEFAULTS``.

    So ``ckpt_world.env_diff`` can tell "the checkpoint omitted this key, the live env restates its
    default" from a genuine world change -- the #440 chain-block, which pinned knobs in
    ``canonical_env.sh`` at exactly their existing defaults and thereby invalidated every checkpoint
    predating the export. EVERY reader in this module must call it: a module-local reader that skips
    registration is one pin-at-default away from repeating that block, and review found exactly that gap
    here twice (only ``_env_float`` registered, while the world-defining ``COSMO_CLOUDS`` is a flag).

    Defensive by design -- ``ckpt_world`` is used by tools that deliberately avoid importing flatworld, so
    a failure to register must degrade to the stricter comparison rather than break the reader."""
    try:
        from .constants import ENV_DEFAULTS
        ENV_DEFAULTS.setdefault(name, value)
    except Exception:
        pass


def _env_float(name, default):
    """Float env knob; unset or unparseable -> default."""
    if default is not None:                    # a None default is "no code default" -- nothing to record,
        _register(name, repr(float(default)))  # and env_diff's absent-key rule then correctly reports a diff
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_int(name, default):
    """Integer env knob; unset or unparseable -> default."""
    if default is not None:
        _register(name, str(int(default)))
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_str(name, default):
    """String env knob; unset -> default."""
    if default is not None:
        _register(name, str(default))
    v = os.environ.get(name)
    return v.strip() if v is not None else default


@dataclass
class StageADConfig:
    # --- grids -------------------------------------------------------------------------------------
    nx: int = 80                       # cross-track columns (uniform in PHYSICAL DISTANCE x, never theta)
    nz: int = 30                       # vertical points: pressure-grid LAYERS == height-grid LEVELS
    theta_max_deg: float = 70.0        # poleward truncation (x = L*tan(theta) diverges near 90 deg)
    p_top: float = 5.0e3               # model top [Pa]; fixes the height-grid top via the reference state
    device: str = "cpu"                # M3.1: "cpu" (numpy) or "gpu" (CuPy). The GPU path is OPTIONAL
                                       #   (requirements-gpu.txt) and, in M3.1, runs the PURE-DYNAMICS limb
                                       #   (the dynamics omega-psi substep; splu Poisson stays on host). The
                                       #   coupled column-physics device path (Phase D) is now live: a
                                       #   coupled run with device="gpu" requires poisson_method="dst".
    # --- WORKING PRECISION (fp32 production spike). Default "fp64" = BYTE-IDENTICAL to the historic core:
    #     fp64 is the DEVELOPMENT/CI default for bug-detection SENSITIVITY (the ~kappa*eps consistency floor
    #     that catches a discretization leak -- an fp32 ~1e-3 floor would bury it), NOT for conservation
    #     (which is structural and closes to round-off at ANY precision). "fp32" stores the prognostic state
    #     (T_atm/T_surf/q/q_c/omega/psi/b) and runs the dynamics stencils/advection in single precision, but
    #     the ill-conditioned Poisson solve STAYS fp64 (its forward error ~ kappa(op)*eps grows with
    #     resolution): the psi-RHS is cast up to fp64 for the splu solve and psi cast back to fp32 at the
    #     existing host/device seam. The thermo column physics (radiation.py/surface.py/thermo.py/microphysics)
    #     is fp64-internal (shared by Stage A/B/C + the cascade -- NOT made dtype-generic): it upcasts on entry
    #     and returns fp64, and the fp32 path casts the result back to fp32 on store. So the fp32 speedup is
    #     bandwidth-bound and comes mainly from the DYNAMICS half (see FP32_SPIKE_NOTES.md). env COSMO_PRECISION.
    precision: str = field(default_factory=lambda: _env_str("COSMO_PRECISION", "fp64"))   # "fp64" | "fp32"
    # --- fp32 round-off-stall fix (only meaningful under precision="fp32"; default OFF). When ON, the
    #     prognostic tendency update is accumulated in fp64 -- field = (field.astype(f64) + dt*tend).astype(f32)
    #     -- the standard guard against the small-increment round-off stall over the ~1e7 steps of a converged
    #     run (when dt*tendency falls below the fp32 ULP of the field, a pure-fp32 add silently drops it). OFF
    #     -> the plain fp32 add (the spike A/Bs the two). Inert under precision="fp64".
    fp32_accumulate_fp64: bool = field(default_factory=lambda: _env_flag("COSMO_FP32_ACCUM_FP64", False))
    # --- THERMO-fp32 LONG-RUN-BIAS GATE (COSMO_THERMO_FP32; default OFF = byte-identical). ORTHOGONAL to
    #     ``precision`` above (which is the DYNAMICS/state fp32 spike). This flag runs ONLY the column-thermo
    #     ARITHMETIC -- radiation.py (two_stream LW/SW), the moist convective adjustment, the in-loop SEB
    #     update, condensation, qsat/dqsat -- in float32 (via ``thermo.set_thermo_dtype`` toggled around
    #     ``_thermo_substep``), while the PERSISTENT prognostic state (T_atm/q/T_surf/...) and ALL the dynamics
    #     stay float64. The fp32 thermo produces an fp32 INCREMENT (dTa / dq) which adds to the fp64 state ->
    #     numpy upcasts the sum back to fp64, so the state is a PERSISTENT-fp64 ACCUMULATOR by construction (the
    #     standard round-off-stall mitigation -- the increments round once to fp32, the running total never does).
    #     This isolates the genuinely-new question -- does the fp32 thermo arithmetic introduce a PERSISTENT
    #     radiation/temperature BIAS over a long coupled run (the GPU-port go/no-go) -- which the dynamics-fp32
    #     gate (0.008 C, PR #192) did NOT answer. CPU-fp32 here != GPU-fp32 (reduction order / FMA /
    #     transcendental impls differ); a PASS still needs on-device confirmation. env COSMO_THERMO_FP32.
    thermo_fp32: bool = field(default_factory=lambda: _env_flag("COSMO_THERMO_FP32", False))
    # --- SEB-increment fp32 FREEZE FIX (default ON = the fixed behaviour). DIAGNOSTIC-ONLY flag: set
    #     COSMO_SEB_LEGACY_FP32=1 to RESTORE the old (buggy) behaviour, kept solely for the regression-test
    #     contrast and the OFF-vs-ON speed A/B -- NOT a supported physics knob. Only meaningful under the
    #     thermo_fp32 path (inert under pure fp64, which never round-trips the state through fp32). THE BUG:
    #     under thermo_fp32 the per-substep SEB increment (~1.7e-6 K over a 5 m ocean mixed layer at ~10 W/m2)
    #     was applied INSIDE the fp32 scratch copy of T_surf, where it is swamped under the ~3e-5 K fp32 ULP at
    #     ~278 K; the increment-reconcile then recovers it as a DIFFERENCE of two fp32 values near 278 K
    #     ((work32 - old64.astype(f32))), which re-quantizes any sub-ULP change to EXACTLY 0 -> ocean T_surf
    #     bit-frozen (dry/rock columns, ~1 m heat capacity, clear the ULP and evolve). THE FIX: carry the
    #     computed seb_incr OUT of the fp32 substep and add it to the fp64 T_surf absolute directly (the (nx,)
    #     surface field only -- the expensive (nx,nz) dynamics/thermo stay fp32). seb_incr ~1.7e-6 is exactly
    #     representable in fp32 standalone; only ADDING it to ~278 lost it. See MODELING_LESSONS.md.
    seb_legacy_fp32: bool = field(default_factory=lambda: _env_flag("COSMO_SEB_LEGACY_FP32", False))
    # --- BL-refined stretched vertical grid (BL_REFINEMENT_SPEC.md): default OFF (uniform, byte-identical).
    #     "stretched" replaces the uniform height grid with a sinh-stretch (``_stretched_z``) whose lowest
    #     spacing is ``dz_min`` [m], so a ~10-30 m lowest level resolves the BL jet and the 10 m wind is
    #     read from a model level rather than log-extrapolated. The non-uniform vertical operators
    #     (``_ddz``/``_div_grad``/the anelastic Poisson/Smagorinsky/CFL) branch on the array ``dz``; the
    #     scalar (uniform) path is the current code verbatim. ``vertical_implicit_diffusion`` (P2) moves the
    #     vertical eddy-diffusion limb implicit to relieve the ~min(dz) diffusive CFL; OFF = explicit,
    #     byte-identical. ``thermo_grid_type`` (P2) co-refines the PRESSURE grid near P_SURF for a converged
    #     2 m temperature; OFF = uniform p_edge, byte-identical. env COSMO_GRID_TYPE / COSMO_DZ_MIN /
    #     COSMO_VERT_IMPLICIT_DIFF / COSMO_THERMO_GRID.
    grid_type: str = field(default_factory=lambda: _env_str("COSMO_GRID_TYPE", "uniform"))   # uniform|stretched
    dz_min: float = field(default_factory=lambda: _env_float("COSMO_DZ_MIN", 20.0))          # lowest spacing [m]
    vertical_implicit_diffusion: bool = field(
        default_factory=lambda: _env_flag("COSMO_VERT_IMPLICIT_DIFF", False))
    # --- STRETCHED CROSS-TRACK GRID (the chimney-nest REPLACEMENT program, 2026-07-10): default
    #     "uniform" = the single-scalar-dx grid, BYTE-IDENTICAL. "stretched" selects the capped-geometric
    #     ``_stretched_x_edges`` x grid -- ``x_n_fine`` cells of exactly ``x_dx_min`` at the equatorial
    #     axis, a ~``x_ratio``-per-cell geometric transition, and an OUTER-PINNED plateau of ``x_dx_max``
    #     cells (``x_dx_max = 0`` -> auto: today's uniform spacing x_max/nx, so the far field matches the
    #     uniform grid) -- resolving the equatorial updraft inside the core instead of via the (retired)
    #     chimney nest. When stretched, ``nx`` is DERIVED from the builder (the constructor overrides
    #     ``c.nx`` with a loud log; size arrays ahead of time via ``stretched_x_cell_count``).
    #     Mutually exclusive with chimney_nest / chimney_param / equator_norain_cells (it replaces them).
    #     env COSMO_X_GRID_TYPE / COSMO_X_DX_MIN / COSMO_X_DX_MAX / COSMO_X_RATIO / COSMO_X_N_FINE.
    x_grid_type: str = field(default_factory=lambda: _env_str("COSMO_X_GRID_TYPE", "uniform"))  # uniform|stretched
    x_dx_min: float = field(default_factory=lambda: _env_float("COSMO_X_DX_MIN", 2.0e3))     # axis cell width [m]
    x_dx_max: float = field(default_factory=lambda: _env_float("COSMO_X_DX_MAX", 0.0))       # 0 -> auto: x_max/nx
    x_ratio: float = field(default_factory=lambda: _env_float("COSMO_X_RATIO", 1.05))        # transition growth/cell
    x_n_fine: int = field(default_factory=lambda: _env_int("COSMO_X_N_FINE", 24))            # cells at x_dx_min
    # --- OUTWARD-COARSENING TAIL / BOX EXTENSION (2026-07-10): default 0 = NO TAIL = the three-block
    #     grid above, bitwise. The box-wall finding: run1's MATURE cell fills the theta_max=80 box
    #     (|psi| ~10% of max at x=264 Mm, T_surf -51 C and still falling at the wall; day-mean S ~ 0
    #     beyond theta~70, so the outer state is transport-controlled and the wall truncates it
    #     mid-decline). ``x_tail_from`` [m] starts an outward-coarsening tail: the ordinary stretched
    #     grid is built on [0, x_tail_from] BITWISE, then widths grow geometrically (``x_tail_ratio``;
    #     0 -> x_ratio per the auto convention) from x_dx_max up to the ``x_tail_dx_max`` cap and
    #     plateau to the (extended) domain end. Nothing dynamical lives out there (S ~ 0, q ~ 0), so
    #     coarse cells suffice -- where the coarsening may START is a finding-dependent choice, hence
    #     the parameterization (Jake, 2026-07-10). A theta_max=86 box costs ~+21% columns with
    #     tail_from = the old theta80 x_max, or FEWER columns than the theta80 grid with
    #     tail_from ~ theta70 (the forcing-dead outer half of the plateau replaced by ~coarse cells).
    #     Requires x_grid_type="stretched". PINNING DISCIPLINE: prefer ``x_tail_from_deg`` -- it routes
    #     through the SAME g.x_of_theta expression the box end uses, so tail_from_deg = the baseline's
    #     theta_max gives a coarsen-start (and, via the auto rule below, a plateau spacing) that is
    #     FLOAT-IDENTICAL to the baseline grid's -- the pure-extension arm's inner grid is then bitwise
    #     the baseline grid with zero hand-copied floats (a decimal COSMO_X_TAIL_FROM would break the
    #     bitwise control). The x_dx_max 0 -> auto rule anchors at the INNER domain end (tail_from when
    #     tailed, x_max otherwise -- the untailed rule unchanged): auto = inner_end / nx(requested).
    #     NB a coarsen-start INSIDE the old box (tail_from_deg < the old theta_max) therefore
    #     auto-derives a FINER plateau than production -- such arms set x_dx_max explicitly (full-repr
    #     float; see STRETCHX_CAMPAIGN.md box ladder).
    #     env COSMO_X_TAIL_FROM_DEG / COSMO_X_TAIL_FROM / COSMO_X_TAIL_DX_MAX / COSMO_X_TAIL_RATIO.
    x_tail_from_deg: float = field(default_factory=lambda: _env_float("COSMO_X_TAIL_FROM_DEG", 0.0))  # 0 = unset
    x_tail_from: float = field(default_factory=lambda: _env_float("COSMO_X_TAIL_FROM", 0.0))       # 0 = no tail
    x_tail_dx_max: float = field(default_factory=lambda: _env_float("COSMO_X_TAIL_DX_MAX", 3.0e6)) # tail cap [m]
    x_tail_ratio: float = field(default_factory=lambda: _env_float("COSMO_X_TAIL_RATIO", 0.0))     # 0 -> x_ratio
    # --- SCALE-AWARE CONVECTIVE-ADJUSTMENT GATING (the stretched-x gray zone): default 0/0 = OFF =
    #     BYTE-IDENTICAL. The moist convective adjustment parameterizes SUB-GRID convection; on a
    #     stretched grid whose fine axis cells RESOLVE the updraft, applying it there double-counts (and
    #     kills) the resolved convection the refinement exists to capture. The per-column gate
    #     ``w_conv(dx_local)`` multiplies the Betts-Miller relaxation fraction: 0 for cells narrower than
    #     ``conv_dx_off`` (convection-permitting -- the resolved dynamics owns it), 1 for cells wider than
    #     ``conv_dx_full`` (fully parameterized -- the current behavior), a smoothstep in ln(dx) between
    #     (the gray zone). The blend is a per-column SCALAR, so the closure's per-column enthalpy+water
    #     conservation is preserved exactly. The ramp endpoints CANNOT be derived -- they are defended by
    #     the insensitivity sweep (x2 each way; the program's go/no-go gate), and the who-lifts diagnostic
    #     (``conv_gate_diagnostics``) instruments the handover directly. Cells all wider than
    #     ``conv_dx_full`` collapse the gate to exactly OFF -- NB that collapse is CONDITIONAL on
    #     ``conv_dx_full`` sitting below the target grid's coarse width (~430-605 km in production;
    #     the intended gray zone ~10-150 km is far below it, but a carelessly large ``conv_dx_full``
    #     WOULD gate a uniform grid). Semantics: the gate scales the relaxation RATE (w/tau), not the
    #     cumulative parameterized work -- gray-zone columns still relax toward convective neutrality,
    #     only slower; w_conv = 0 alone fully exempts a column.
    #     env COSMO_CONV_DX_OFF / COSMO_CONV_DX_FULL [m].
    conv_dx_off: float = field(default_factory=lambda: _env_float("COSMO_CONV_DX_OFF", 0.0))
    conv_dx_full: float = field(default_factory=lambda: _env_float("COSMO_CONV_DX_FULL", 0.0))
    # --- IMPLICIT VERTICAL OMEGA-ADVECTION (ADVECTION_DT_ROADMAP.md): default "explicit" = BYTE-IDENTICAL.
    #     The explicit dt is CFL-limited by the VERTICAL omega-advection term ``dt = cfl*dz_min/|w|`` -- the
    #     equatorial chimney column (|w|~1 m/s, a 1/dx grid-scale singularity); the ONLY binding term on the
    #     converged donor (6.1 s vs the 60 s dt_dyn cap). "vertical_implicit" moves that single limb to
    #     backward-Euler UPWIND -- a 1-D tridiagonal solve per column (``_implicit_vertical_advection``,
    #     reusing ``_solve_tridiag_columns``), the direct analog of ``vertical_implicit_diffusion``. Upwind
    #     backward-Euler is monotone (M-matrix: positive diagonal, non-positive off-diagonals) and
    #     UNCONDITIONALLY STABLE, so omega advection no longer caps dt; ``_cfl_dt`` then drops the
    #     ``dz_min/|w|`` term (the HORIZONTAL omega-advection and diffusion caps are KEPT, so dt lands at
    #     min(dt_dyn, horiz-adv, diffusion) ~ 60 s -> ~3x fewer steps (~2.7x realized: the intensified
    #     chimney pulls the CFL just under the 60 s cap); the converged donor's explicit dt is a steady 20 s,
    #     NOT the roadmap's 6.1 s turbulent-burst snapshot, so the step-count ceiling is ~3x not ~10x --
    #     measured in archive/_sl_dt_convergence_RESULTS.md). NB this ~3x is the STEP-COUNT relief only; a
    #     BARE dt=60 raise shifts the climate (thermo-cadence carrier; _sl_FINAL_VERDICT.md), fixed by
    #     thermo_n_sub sub-cycling (PR #203 -- see the warning below). COMPOSES with
    #     ``vertical_implicit_diffusion`` (two sequential tridiagonal solves: advection then diffusion), the
    #     production BL-refined path. The advection increment is TRANSPORT, not dissipation -- so it is NOT
    #     booked as dissipative heat (the heating closure reads only the diffusion solve's increment).
    #     ("Transport" here means the advective-form upwind w*dω/dz, which is NOT strictly integral-conserving
    #     when dw/dz != 0 -- the point is only that it carries vorticity, it does not REMOVE KE.) env COSMO_ADVECTION.
    # ⚠️ TEST/nominal default is "explicit"; the CANONICAL nsub30 world runs "vertical_implicit" (PR #203 —
    #    viable because thermo_n_sub sub-cycling fixes the dt-cadence carrier that made naive dt=60 fail in #197;
    #    NOT the #197 "settled-negative" verdict, which predates the sub-cycling fix). NOT set by canonical_env.sh;
    #    the production driver (nsub30_backbone.py / m33_runs/_nsub30_production.py) sets it. See CANONICAL_WORLD.md.
    advection_scheme: str = field(default_factory=lambda: _env_str("COSMO_ADVECTION", "explicit"))   # explicit|vertical_implicit|vertical_implicit_strang[_e]
    # vertical-ω-advection chimney sub-cycling (option 2): when > 1 AND advection_scheme is an implicit
    # variant, the implicit vertical-advection limb is sub-stepped n times at dt/n in the columns where the
    # advective Courant number w*dt/dz exceeds ``adv_subcycle_courant`` (the stiff chimney), full dt elsewhere.
    # This attacks the O(dt) advection-limb truncation where w/dz is extreme, cheaply (a tridiag solve is cheap;
    # only flagged columns sub-cycle). 1 = OFF (byte-identical to the single-solve limb). Composes with Strang.
    adv_subcycle_n: int = field(default_factory=lambda: int(_env_float("COSMO_ADV_SUBCYCLE_N", 1.0)))
    adv_subcycle_courant: float = field(default_factory=lambda: _env_float("COSMO_ADV_SUBCYCLE_COURANT", 1.0))
    # HEAT-advection sub-cycling: advance _advect_heat's theta update in n sub-steps at dt/n (frozen mass
    # flux), decoupling the heat-transport numerical-diffusion Courant from the (relieved) dynamics dt. Kept
    # as the DIAGNOSTIC that proved heat-advection diffusion is NOT the dt=60 climate carrier: sub-cycling
    # moved the gap by ZERO; the carrier is the column-thermo cadence (_sl_FINAL_VERDICT.md), fixed by
    # thermo_n_sub sub-cycling (PR #203). 1 = OFF (byte-identical single step). Use n≈dt_dyn/dt_ref to restore
    # the reference Courant (e.g. n=3 at dt60 -> dt_sub=20 = the explicit reference's heat-advection step).
    heat_advect_subcycle_n: int = field(default_factory=lambda: int(_env_float("COSMO_HEAT_SUBCYCLE_N", 1.0)))
    # TRACER-COURANT GATE (review caveat, _sl_FINAL_VERDICT.md): when advection_scheme != "explicit" the
    # implicit-ω CFL relief lets dt rise above the vertical CFL the EXPLICIT flux-form upwind tracer
    # transport needs to stay monotone (Courant <= 1). Beyond it the advection limbs clamp q/q_c back to
    # >= 0 each step (excursion mass counted; flush_clamp_stats -- theta has no clamp), so ``step()``
    # measures the realized per-cell tracer Courant on the pressure-grid mass flux and warns once if > 1.
    # ``tracer_courant_strict`` RAISES instead of warning (for a production ON-run that must not silently
    # inject clamp mass). Both are NO-OPs on the default explicit path (the guard is never called).
    tracer_courant_strict: bool = field(default_factory=lambda: _env_flag("COSMO_TRACER_COURANT_STRICT", False))
    # IMPLICIT VERTICAL TRACER ADVECTION (COSMO_TRACER_VERT_IMPLICIT, default OFF -> byte-identical):
    # the structural fix for the tracer-Courant exposure above. The explicit flux-form tracer transport's
    # VERTICAL limb rides OUTSIDE the CFL controller (the implicit-ω relief and the stretched-x resolved
    # equatorial updraft push the vertical tracer Courant past 1 -> the positivity clamp injects mass and
    # the run can go non-finite). ON: the VERTICAL limb of ALL FOUR advected tracers (q, q_c, q_r when
    # prognostic_rain, theta) moves to a per-column backward-Euler donor-cell solve on the SAME
    # divergence-free pressure-grid mass flux ``Fz`` and the SAME ``dm``/dx weights, in conservative flux
    # form (column INTEGRAL tracer*dm preserved to round-off; zero top/surface flux as on the explicit
    # path). The system is an M-matrix -> UNCONDITIONALLY STABLE at any Courant with POSITIVITY
    # preserved (no clamp mass); no-new-extrema holds for the FULL split under the shared
    # divergence-free flux, not for the vertical solve in isolation (#324 review -- the solve's
    # docstring has the precise statement). The HORIZONTAL limb stays explicit (it rides the dx/|u| cap the CFL
    # controller DOES enforce) -- the chimney nest's NEST_CHIM_IMPLICIT split (PR #257) ported to the
    # core's tracers. Device/capture-ready: the band build is pure elementwise ops and the solve
    # dispatches to the fp64 device Thomas RawKernel (the _sediment_rain idiom); capture_ready without
    # device='gpu' raises (a host solve cannot be recorded into the graph).
    tracer_vert_implicit: bool = field(default_factory=lambda: _env_flag("COSMO_TRACER_VERT_IMPLICIT", False))
    thermo_grid_type: str = field(default_factory=lambda: _env_str("COSMO_THERMO_GRID", "uniform"))
    thermo_dp_min: float = field(default_factory=lambda: _env_float("COSMO_THERMO_DP_MIN", 500.0))  # [Pa]
    # --- streamfunction-Poisson solver (speedup Phase 1): "splu" (default) builds the verbatim sparse-LU
    #     ``build_anelastic_poisson`` (BYTE-IDENTICAL to current behaviour). "dst" builds the separable
    #     DST x tridiagonal ``SeparablePoisson`` -- a machine-precision drop-in (max-rel-err < 1e-9 vs splu)
    #     that is far cheaper at production nx. Valid ONLY for the closed-wall global core (asserted at build;
    #     the chimney's open_outer_edge BC breaks separability). env COSMO_POISSON_METHOD.
    poisson_method: str = field(default_factory=lambda: _env_str("COSMO_POISSON_METHOD", "splu"))
    # --- DST-Poisson FFT thread count (speedup #3): the separable DST solve (poisson_method="dst")
    #     calls scipy.fft.dst/idst, which are SINGLE-THREADED unless given workers>1. This wraps the solve
    #     in scipy.fft.set_workers(n). DEFAULT 1 (=current behaviour, unset => structurally the bare call).
    #     workers> 1 only splits the batched 1-D transforms across the wavenumber axis -> BIT-IDENTICAL result
    #     (each 1-D DST is computed identically); it is a pure wall-clock lever (~5-6x on the FFT at prod nx).
    #     A negative value (-1) means "all available cores" (scipy's os.cpu_count() convention). env
    #     COSMO_POISSON_WORKERS. Inert when poisson_method != "dst".
    poisson_workers: int = field(default_factory=lambda: _env_int("COSMO_POISSON_WORKERS", 1))
    # --- anelastic reference state (#0): a hydrostatic, horizontally-uniform base profile ----------
    T_ref_surf: float = 300.0          # reference-state surface temperature [K] (representative mid-domain)
    lapse_ref: float = 6.5e-3          # reference lapse rate [K/m] (troposphere)
    T_ref_top: float = 150.0           # isothermal-cap floor [K] (stops T_ref running to absurd values aloft)
    # --- INITIAL-CONDITION floors on the crude radiative-eq guess T0 = (max(S, s_init_floor)/sigma)^0.25,
    #     clipped to [t_init_floor, 360] K (air column to t_atm_init_floor). Far out S -> 0, and the far-field
    #     EQUILIBRIUM -> ~0 K too (no sun + transport -> 0 at large x, by energy conservation; the greenhouse
    #     "floor" is transport-sustained and vanishes far enough out). The old hard-coded 120/130 clip started
    #     the cold cap WARM-biased AT the clip (the flat -154 C far-field plateau was ~the 120 K clip,
    #     unconverged -- ~2 yr e-folding). DEFAULT is now a COLD start (50 K, 2026-07-14). BYTE-DIFFERENT from
    #     the pre-2026-07-14 code for FRESH spin-ups only; checkpoint resumes (canonical/run11) are unaffected
    #     (they load T, they don't re-init) -- set 120/130/1.0 to reproduce the old warm start. The floor can
    #     go this low ONLY because thermo.esat now pole-guards the Magnus formula (unguarded it NaN'd below
    #     ~30 K); that guard also lets the far field COOL below 30 K toward its true ~0 K limit at RUNTIME.
    #     Validated: fresh spin-up STABLE at 40/50/80 K init (and 10 K with the esat guard).
    #     env COSMO_T_INIT_FLOOR / COSMO_T_ATM_INIT_FLOOR / COSMO_S_INIT_FLOOR.
    t_init_floor: float = field(default_factory=lambda: _env_float("COSMO_T_INIT_FLOOR", 50.0))       # [K] surface IC clip lower bound (cold start)
    t_atm_init_floor: float = field(default_factory=lambda: _env_float("COSMO_T_ATM_INIT_FLOOR", 50.0))   # [K] air-column IC floor
    s_init_floor: float = field(default_factory=lambda: _env_float("COSMO_S_INIT_FLOOR", 1.0e-3))     # [W/m^2] S floor in T0 (small -> the T-clip is the effective floor)
    # --- water vapor: fixed-RH profile RH(sigma), sigma = p/p_surf (thermo grid) --------------------
    rh_surf: float = 0.8
    rh_top: float = 0.2
    # --- radiation tunables (documented inputs; used from M2b.3, #3) --------------------------------
    co2_ppm: float = 400.0
    albedo_ocean: float = C.ALBEDO_OCEAN
    albedo_ice: float = C.ALBEDO_ICE
    albedo_dryland: float = C.ALBEDO_DRYLAND
    k_wv_lw: float = rad.K_WV_LW
    tau_lw_wellmixed: float = rad.TAU_LW_WELLMIXED
    i0_scale: float = 1.0              # multiply the fiducial suns' brightness (PINNED at 1.0; sweepable)
    # --- radiatively-active clouds (#19): default ON (the all-on accurate model; env COSMO_CLOUDS=0 to
    #     disable -> the pre-cloud clear-sky core byte-for-byte). The cloud OPTICAL DEPTH is COMPUTED FROM
    #     PHYSICS (the condensed-water path + droplet size; ``radiation.cloud_water_path``), NOT swept.
    #     ``cloud_water_scale`` (1.0 = the physical best estimate) and ``cloud_rh_crit`` are the residual
    #     PHYSICAL sensitivities (microphysics water amount; sub-grid humidity variance) reported, not fitted
    #     -- env COSMO_CLOUD_WATERSCALE/_RHCRIT.
    clouds: bool = field(default_factory=lambda: _env_flag("COSMO_CLOUDS", True))
    cloud_rh_crit: float = field(default_factory=lambda: _env_float("COSMO_CLOUD_RHCRIT", rad.CLOUD_RH_CRIT))
    cloud_water_scale: float = field(default_factory=lambda: _env_float("COSMO_CLOUD_WATERSCALE", 1.0))
    cloud_conv_boost: float = field(default_factory=lambda:
                                    _env_float("COSMO_CLOUD_CONVBOOST", rad.CLOUD_CONV_BOOST))
    #     SW layer-overlap for the COLUMN cloud albedo (``radiation.cloud_sw_albedo``): "column" (the
    #     DEFAULT since 2026-06-25 -- a single two-stream on the cloud-fraction-weighted column-total SW
    #     optical depth, the SAME linear-grid-mean footing the LW limb already uses) or "random" (the legacy
    #     reflection PROBABILITY product 1-prod(1-f*R_in)). "random" INFLATES a contiguous saturated column's
    #     albedo toward ~0.99 by stacking layers; "column" removes that inconsistency (the cloud-albedo
    #     overlap audit, CLOUDS_RESULTS.md §4b) -- it is the canonical nsub30_swfix_col world's scheme, so it
    #     is now the default (was "random"; Jake 2026-06-25, after a script forgot the flag and got ~0.999).
    #     Env COSMO_CLOUD_SW_OVERLAP=random restores the legacy scheme. The cosmology DOC now carries the
    #     column-overlap numbers (~0.92-0.95 albedo; NSUB30_DOC_NUMBERS.md, climate-refresh PR #206).
    cloud_sw_overlap: str = field(default_factory=lambda: _env_str("COSMO_CLOUD_SW_OVERLAP", "column"))
    #     Cloud SW single-scattering albedo (SSA): 1.0 (DEFAULT, byte-identical) = REFLECT-ONLY clouds, the
    #     surface light is then an UPPER BOUND (real clouds absorb near-IR). Set < 1 (env COSMO_CLOUD_SW_SSA
    #     for both phases, or _LIQ/_ICE) to turn on cloud SW ABSORPTION -> a real surface-light estimate. This
    #     is a MODEL change (lowers TOA albedo -> more absorbed SW -> warming); validate via a converged run.
    #     Under the two-band SW (sw_bands, the default/canonical), this SSA is the NEAR-IR-band value: the
    #     absorption is confined to the near-IR (real cloud SW absorption band); the VISIBLE band stays
    #     conservative (SSA~1), so the surface visible floor is preserved (see two_stream docstring below).
    cloud_sw_ssa_liq: float = field(default_factory=lambda:
                                    _env_float("COSMO_CLOUD_SW_SSA_LIQ", _env_float("COSMO_CLOUD_SW_SSA", rad.CLOUD_SSA_LIQ)))
    cloud_sw_ssa_ice: float = field(default_factory=lambda:
                                    _env_float("COSMO_CLOUD_SW_SSA_ICE", _env_float("COSMO_CLOUD_SW_SSA", rad.CLOUD_SSA_ICE)))
    #     Cloud SW TWO-STREAM (COSMO_CLOUD_SW_TWOSTREAM, default OFF -> reflect-only, byte-identical). When ON
    #     (with ssa<1) the cloud SW reflectance/absorptance/transmittance are solved JOINTLY (non-conservative
    #     two-stream, radiation.two_stream_nc), so A_abs*S_incident heats the deck from the FULL incident beam --
    #     the thermodynamic cloud-burn-off driver the reflect-only ssa knob under-drives (it heats only the
    #     post-reflection remnant, ~25-50x too weak on a bright deck). MODEL change (deck heating -> less cloud
    #     -> lower albedo -> warming); validate via a CONVERGED run. Absorption is NEAR-IR-CONFINED under the
    #     two-band SW (visible band conservative, so the surface visible floor survives -- the fix for the
    #     grey-spectral limb that absorbed the visible too and drove surface light ~10^2-10^3x too dark) and
    #     strongly ssa-sensitive -> SWEEP ssa (soft magnitude), do not pin. ssa=1 -> A_abs=0 -> byte-identical
    #     even with the flag ON. (Gray-SW mode -- sw_bands off, non-production -- keeps the old grey whole-beam
    #     absorption, since there is no visible band to protect.)
    #     NB the two-stream solves the reflectance on the COLUMN-TOTAL footing, so ``cloud_sw_overlap`` is
    #     INERT under it (the "random" overlap is not applied); the ssa=1 byte-identical guarantee therefore
    #     holds only for overlap="column" (the default/canonical) -- StageADCore warns on the random combo.
    cloud_sw_twostream: bool = field(default_factory=lambda: _env_flag("COSMO_CLOUD_SW_TWOSTREAM", False))
    # --- PROGNOSTIC cloud condensate (CLOUD_PROGNOSTIC_ROADMAP Chunk 1): a RETAINED, advected condensate
    #     reservoir q_c, replacing the single-pass "supersaturation rains immediately" rain-out. Default OFF
    #     (env COSMO_PROGNOSTIC_CONDENSATE=1) -> the core is BYTE-IDENTICAL to the single-pass scheme. ON:
    #     supersaturation SOURCES q_c (not precip), q_c is ADVECTED by the resolved flow on its OWN (smaller)
    #     diffusivity ``kappa_qc``, SUNK by a crude single-rate autoconversion, RADIATED via the #19 optics
    #     (in-cloud LWP = rho*q_c/f*dz with the cover floor below), and CHECKPOINTED -- so a radiatively-cooled
    #     deck SURVIVES the step. The crude sink + RH-diagnostic cover (floored) are the skeleton; the real
    #     microphysics (Chunk 2) and the consistent sub-grid PDF cover (Chunk 3) are the sub-flag blocks below.
    prognostic_condensate: bool = field(default_factory=lambda: _env_flag("COSMO_PROGNOSTIC_CONDENSATE", False))   # ⚠️ TEST default False (diagnostic clouds); CANONICAL = True (canonical_env.sh).
    kappa_qc: float = 5.0              # q_c sub-grid eddy-diffusivity [m^2/s] -- SMALLER than kappa_q=50: q_c is
                                       # spatially intermittent (zero in clear air, sharp cloud edges), so the
                                       # vapour regulariser would over-smear/dissolve the thin stratus deck the
                                       # scheme exists to capture; the deck width/survival is a box-test-sensitive
                                       # outcome with a reported kappa_qc sensitivity (roadmap rule 5).
    qc_autoconv_tau: float = 1.0e3     # crude autoconversion timescale tau [s] (Kessler-class; the simplest
                                       # closed sink -- the real condensation<->precip split is Chunk 2).
    qc_autoconv_crit: float = field(default_factory=lambda: _env_float("COSMO_QC_AUTOCONV_CRIT", 1.0e-3))   # ⚠️ TEST default 1e-3; CANONICAL = 5e-4 (canonical_env.sh).
    #   ^ cloud-water threshold q_c,crit [kg/kg]: only condensate ABOVE this rains (P_auto =
    #     max(q_c - q_c,crit, 0)/tau), so a thin sub-threshold deck persists. Env-overridable (default kept
    #     at the historical 1.0e-3 -> byte-identical). The cloud-ON sensitivity sweep (_autoconv_sweep.py)
    #     found this knob SETS the ribbon deck thickness / cloud albedo (LWP 295->1844 over 0.1->1.0 g/kg)
    #     while the equilibrated ribbon RAIN is supply-limited (~9 mm/d, ~flat); the principled CANONICAL is
    #     the classic Kessler ~0.5e-3 (set via COSMO_QC_AUTOCONV_CRIT for the cloud-ON canonical fleet).
    cloud_f_min: float = 0.05          # cover FLOOR f >= f_min wherever q_c > qc_cover_thresh -- the cover<->
                                       # condensate consistency rule the in-cloud LWP = rho*q_c/f*dz conversion
                                       # rests on (clear air + condensate is unphysical; Chunk 3 -> the PDF cover).
    qc_cover_thresh: float = 1.0e-7    # q_c,thresh [kg/kg]: condensate above which the cover floor bites.
    # --- CHUNK 2 (CLOUD_PROGNOSTIC_ROADMAP) real microphysics SUB-FLAGS. All only act when
    #     ``prognostic_condensate`` is ON, so production (master OFF) is UNTOUCHED/byte-identical regardless of
    #     these. They default to the ACCURATE Chunk-2 physics (so opting into the prognostic scheme gets the
    #     real microphysics by default); turning them OFF reduces the ON path to the Chunk-1 SKELETON, which is
    #     the A/B baseline the precip-efficiency shift is measured against (and, in the instant-rain limit,
    #     byte-identical to the OFF single-pass core).
    qc_dqsat_adjust: bool = field(default_factory=lambda: _env_flag("COSMO_QC_DQSAT_ADJUST", True))
    #   ^ dqsat/dT-corrected saturation adjustment C = (q-qsat)/(1 + (L/cp) dqsat/dT) instead of the bare
    #     q-qsat (which over-condenses: the latent warming raises qsat, leaving the air sub-saturated against
    #     the NEW T). One Newton step leaves the air AT saturation; matters for the small-supersaturation thin
    #     deck. OFF -> the bare skeleton adjustment.
    qc_reevaporation: bool = field(default_factory=lambda: _env_flag("COSMO_QC_REEVAP", True))
    #   ^ re-evaporate SUSPENDED q_c back to vapour in sub-saturated air (RH<1), latent COOLING returned at the
    #     same level -- the energy-consistent inverse of condensation; removes the haze artifact's WATER half
    #     (advected q_c persisting in dry subsiding columns). A falling-rain flux (downward energy transport)
    #     is a NAMED later refinement; this is suspended-q_c re-evaporation only. OFF -> q_c never re-evaporates.
    qc_ice_phase: bool = field(default_factory=lambda: _env_flag("COSMO_QC_ICE_PHASE", True))
    #   ^ ice phase below T_FREEZE (the band's rain peaks at the cold edge): deposition/sublimation uses L_SUB
    #     (not L_VAP) and an ICE autoconversion->snow branch (qc_autoconv_tau_ice/_crit_ice) runs in cold
    #     layers. This is a BINARY ice/liquid split at T_FREEZE (pure ice below 0 C, pure liquid above) -- the
    #     SAME hard split radiation.py's optics and thermo.esat already use, NOT a graded mixed phase; a linear
    #     L/phase ramp (e.g. 0 -> -20 C) is a named future refinement (PR #143 review #1). OFF -> warm-phase-
    #     only (L_VAP everywhere, the liquid autoconversion only) = NOT the band-edge rain (the explicit descope
    #     the roadmap requires naming).
    qc_retained_ceiling: bool = field(default_factory=lambda: _env_flag("COSMO_QC_RETAINED_CEILING", True))
    #   ^ tie the RETAINED-water ceiling to the radiative CLOUD_LWC_MAX/IWC_MAX (radiation.py) so the prognostic
    #     and radiative water ceilings are ONE number: in-cloud LWC = rho*q_c/f (the SAME floored cover the
    #     radiation reads) above the phase ceiling is force-rained this step (rainout, not the gradual
    #     autoconversion). OFF -> q_c is bounded only by autoconversion.
    qc_autoconv_scheme: str = field(default_factory=lambda: _env_str("COSMO_QC_AUTOCONV", "kessler"))
    #   ^ autoconversion (cloud water -> precip) closure: "kessler" (DEFAULT; threshold+timescale
    #     max(q_c-crit,0)/tau) | "sundqvist" (the compared variant q_c/tau*(1-exp(-(q_c/crit)^2))). Best
    #     estimates, REPORTED as a sensitivity, never fitted (roadmap rule 4); the off variant always runs.
    qc_autoconv_tau_ice: float = 1.0e3   # ICE autoconversion (aggregation->snow) timescale tau [s] (Kessler-class,
                                         # qc_ice_phase only); ice aggregates on a similar timescale to warm rain.
    qc_autoconv_crit_ice: float = 1.0e-4 # ICE cloud-water threshold q_c,crit_ice [kg/kg]: ice precipitates more
                                         # readily than liquid (lower threshold than the 1e-3 liquid crit).
    equator_norain_cells: int = 0        # DIAGNOSTIC (default 0 = OFF, byte-identical): suppress the surface
                                         # rain-out in the innermost N cross-track cells (the equator). The
                                         # would-be precip is RETAINED as condensate q_c (water-conserving), so the
                                         # ascending equatorial branch advects it poleward -- the test of whether the
                                         # spurious 1-cell equatorial precip delta (a theta=0 boundary artifact) is
                                         # moisture that SHOULD reach the habitable ribbon. See _equator_suppress_test.py.
    equator_ceiling_lwc_scale: float = 1.0  # DIAGNOSTIC (default 1.0 = off): multiply the rainout cap in the
                                         # suppressed equatorial cells (keep-ceiling mode) to test how much the band
                                         # redistribution depends on the EARTH-observed cap (CLOUD_LWC_MAX ~0.3 g/m^3,
                                         # which binds ~560x at the ~90 C equator vs the adiabatic ~170 g/m^3).
    equator_norain_keep_ceiling: bool = False  # when suppressing: keep the retained-water CEILING force-rain
                                         # (suppress only the autoconversion-rate rain). Bounds q_c to physical
                                         # cloud-water amounts (the no-ceiling default lets q_c pile to ~1 m) -> the
                                         # CLEAN redistribution test: does the band response survive at physical q_c,
                                         # or was it driven by the unphysical equatorial reservoir?
    budget_wet_fraction: bool = field(default_factory=lambda: _env_flag("COSMO_BUDGET_WET_FRACTION", False))
                                         # B5 (default OFF = byte-identical): augment the TEMPERATURE-keyed wetness ramp
                                         # with a SUPPLY-based wet fraction -- f = max(ramp, clip(F/E_w, 0, 1)) per column,
                                         # F = the surface-reaching liquid SUPPLY (last step's self._precip), E_w = the
                                         # WET-TILE evaporative-thermostat demand (the wetness=1 SEB root ~62 C/~70 mm/day,
                                         # _budget_E_w, NOT the demand at the cell's hot T_surf). The max() keeps the cool
                                         # ocean a full evaporative SOURCE; the budget limb wets a SUPPLIED hot salt flat
                                         # the ramp wrongly dries. Exact at the clips; the partial regime over-evaporates
                                         # (single-bucket -- two-tile mosaic is B5b/B5d; PR #177 S1). Carries the equatorial-
                                         # chimney finding (the monostable-WET equator: a convergence core is a standing
                                         # sea, not a dry salt flat) into the core. The T-ramp has NO stable cool-wet root
                                         # at a convergence core (runs away to the 138 C salt flat -- Komabayashi-Ingersoll;
                                         # EQUATOR_POND_RESULTS.md / CHIMNEY_B4_IGNITION_RESULTS.md); SUPPLY, not
                                         # temperature, sets the state. Mirrors _equator_pond.wet_fraction (approach A).
    budget_endorheic: bool = field(default_factory=lambda: _env_flag("COSMO_BUDGET_ENDORHEIC", False))
                                         # ENDORHEIC inner-sea closure (default OFF; requires budget_wet_fraction).
                                         # The per-cell budget_f above caps f=1 and DROPS the excess precip where the
                                         # convergent supply exceeds the local wet-tile demand -- a surface-water LEAK
                                         # (that dropped water should re-evaporate and, via the aloft return, reach the
                                         # band). The inner convergence basin has NO liquid outflow, so it must AREA-
                                         # balance: it fills [0, x_sea] where the cumulative supply covers the cumulative
                                         # wet-tile evaporation (int_0^x_sea E_w dx = int_0^x_sea P dx), axis-outward. So
                                         # the pool's evaporation exactly RETURNS the deposited precip to the atmosphere
                                         # (E=P over the pool -> no leak), and the pool EXTENT emerges from the balance
                                         # instead of the wetness being clamped at the rain footprint. See _wetness.
    chimney_param: bool = field(default_factory=lambda: _env_flag("COSMO_CHIMNEY_PARAM", False))
                                         # B5c (default OFF = byte-identical): replace the divergent equatorial 1/dx
                                         # precip SPIKE in self._precip with equator_chimney_param's resolution-
                                         # INDEPENDENT grid-mean P_param (the finite (1-phi)*M_tot redistributed over a
                                         # fixed footprint; the ribbon band |x|>=x_detect passes through unchanged), so
                                         # the surface hydrology / aloft-supply target / budget_wet_fraction see a
                                         # finite, dx-convergent equatorial supply, not the grid artifact. SURFACE-
                                         # PRECIP swap only: the condensation latent heat is REAL (kept); the
                                         # phi-export q-budget gap (~phi=2% of the tiny zone) + the H_param latent
                                         # redistribution are DEFERRED (CHIMNEY_PHI_CALIBRATION_RESULTS.md). Mutually
                                         # exclusive with equator_norain_cells (don't suppress the source AND
                                         # redistribute it). phi = chimney_param_phi or LOFT_FRACTION_PHI (~0.02).
    chimney_param_phi: float | None = None   # phi override for chimney_param (None -> equator_chimney_param.LOFT_FRACTION_PHI)
    chimney_nest: bool = field(default_factory=lambda: _env_flag("COSMO_CHIMNEY_NEST", False))
                                         # TWO-WAY NEST (default OFF = byte-identical): consume the RESOLVED
                                         # chimney response (flatworld.chimney_couple.chimney_response, set via
                                         # set_chimney_nest_response from the host-excursion loop every N
                                         # core-days). Each thermo substep, REPLACE the inner-zone (|x|<x_detect)
                                         # convective result with the resolved chimney's (held-weight blend w):
                                         # T <- T_pre + H*dt ; q <- q_pre*(1-f) + qexp*dt with f =
                                         # min(Mcond*dt/INT q_pre dm, 1) (fractional vapour sink, never <0) ;
                                         # surface precip <- nest_P -- so the resolved deep-convective heating
                                         # profile (the detrainment-temperature mechanism, finding 1.5) and
                                         # surface supply replace the coarse one, conservatively (per column
                                         # cp*INT H dm == L_VAP*P and net vapour change == -P, by construction).
                                         # The aloft re-evaporation (qexp) is advected to the band by the
                                         # resolved circulation (the emergent moisture export). Capture-safe
                                         # (in-place elementwise on held buffers). Mutually exclusive with
                                         # chimney_param / equator_norain_cells (don't also de-spike/suppress).
    # --- CHUNK 3 (CLOUD_PROGNOSTIC_ROADMAP) sub-flags. Like the Chunk-2 ones, ONLY act when
    #     ``prognostic_condensate`` is ON (production/master-OFF is byte-identical regardless), and default to
    #     the ACCURATE Chunk-3 physics. Turning them OFF reduces the ON path to the Chunk-2 grid-mean scheme.
    qc_pdf_cloud: bool = field(default_factory=lambda: _env_flag("COSMO_QC_PDF_CLOUD", True))
    #   ^ put cloud COVER and the condensation SOURCE on ONE sub-grid total-water PDF (rad.cloud_pdf_partition):
    #     the cover is the saturated-tail probability and the source the saturated-tail supersaturation integral
    #     of the SAME distribution, so cover<->water are consistent on one footing (the haze fix's cover half).
    #     This RETIRES the skeleton's RH-diagnostic cover + grid-mean condensation source (a source from the PDF
    #     tail fires at grid-mean RH < 1). OFF -> the Chunk-2 grid-mean saturation adjustment + floored RH cover.
    qc_radiative_cap_decouple: bool = field(default_factory=lambda: _env_flag("COSMO_QC_CAP_DECOUPLE", True))
    #   ^ DECOUPLE the radiative (and retained-water) ceiling from the adiabatic-RAINOUT cap: bound in-cloud
    #     water at the PHYSICAL-plausibility LWC (rad.CLOUD_LWC_PLAUSIBLE/IWC_PLAUSIBLE, ~a few g/m^3) instead of
    #     CLOUD_LWC_MAX/IWC_MAX, so the cloud radiative LWP TRACKS the retained reservoir q_c (rainout is the
    #     autoconversion sink's job; re-clipping at the rainout cap double-counts it). Needs the PDF cover to make
    #     the in-cloud rho*q_c/f conversion physical. OFF -> the Chunk-1 rainout-cap optics (cap+cover-set strength).
    # --- CHUNK 4 (CLOUD_PROGNOSTIC_ROADMAP) boundary-layer / stratocumulus closure. Like the Chunk-2/3
    #     sub-flags, ONLY acts when ``prognostic_condensate`` is ON (production/master-OFF byte-identical
    #     regardless), defaults ON-when-master-ON, and needs the resolved flow (the subsidence signal w) live --
    #     it is inert without dynamics. Sources/sinks the SAME prognostic q_c (no second condensate), so the
    #     budget stays single-reservoir. The radiatively-driven low-cloud regime grid-mean RH under-resolves
    #     (``flatworld/bl_cloud.py``); direction-firm / magnitude-soft (the deck depth/LWP is a parameter choice).
    bl_stratocumulus: bool = field(default_factory=lambda: _env_flag("COSMO_BL_STRATOCUMULUS", True))
    #   ^ maintain radiatively-driven low cloud under a subsidence inversion: a cloud-top-radiative-cooling
    #     condensation SOURCE + an entrainment SINK on q_c, in subsiding inversion-capped columns within the
    #     low-cloud sigma band. OFF -> the Chunk-1/3 bulk scheme alone (no stratocumulus deck in the dry
    #     subsiding branch). All bl_sc_* knobs below act only when this AND the master are ON.
    bl_sc_sigma_lo: float = 0.70       # low-cloud deck sigma band (sigma = p/P_SURF): CLOUDS_RESULTS.md §4a's
    bl_sc_sigma_hi: float = 0.90       #   low deck sits sigma~0.7-0.9. The source/sink act only in this band.
    bl_sc_stability_min: float = 2.0   # min static stability dtheta/dz [K/km] at the deck top for an inversion
                                       #   cap (a neutral/unstable column fails -> inert without an inversion).
    bl_sc_entrain_eff: float = field(default_factory=lambda: _env_float("COSMO_BL_SC_ENTRAIN", 1.0))
    #   ^ entrainment efficiency A_ent (the REPORTED sensitivity, roadmap rule 4): scales the entrainment-drying
    #     sink D = A_ent * q_c / tau_entrain. Higher -> a thinner steady deck. Best estimate 1.0, NEVER fitted.
    bl_sc_ctop_cool_eff: float = 1.0   # fraction of the cloud-top LW cooling flux that maintains condensate
                                       #   (eta_rad); 1.0 = the full window cooling f_win*sigma*T^4 (magnitude-soft).
    bl_sc_tau_entrain: float = 3600.0  # entrainment-drying relaxation timescale [s] (~BL turnover; ~1 hr).
    bl_sc_deck_cover: float = 1.0      # overcast deck cover f_deck for the in-cloud rho*q_c/f conversion AND the
                                       #   radiation cover floor (stratocumulus is overcast) -- so the optics see
                                       #   the deck as a solid low cloud, consistent with the source's emissivity.
    bl_sc_source_max: float = 1.0e-6   # hard cap on the maintenance source [kg/kg/s] -- the backstop runaway
                                       #   guard (the source already saturates as the deck emissivity -> 1).
    # --- BELOW-CLOUD PRECIPITATION RE-EVAPORATION (the falling-precip "virga" fix; OPEN_ISSUES 2026-06-16
    #     "Sub-cloud VIRGA"). Master flag (env COSMO_PRECIP_REEVAP), default OFF -> BYTE-IDENTICAL (the column
    #     autoconversion total is booked straight to the surface, as before). ON acts ONLY on the
    #     ``prognostic_condensate`` path (where ``_precip`` is the produced-aloft autoconversion total -- the
    #     single-pass core never calls _prognostic_microphysics, so it is unaffected): integrate the downward
    #     precip flux through the column and re-evaporate / sublimate a portion in each SUB-SATURATED layer
    #     below cloud, feeding the vapour back to ``q`` (then carried by step()'s _advect_moisture) so
    #     ``_precip`` becomes the SURFACE-reaching flux. Rate-limited by the ABSOLUTE saturation deficit
    #     max(qsat-q,0) -- NOT RH: a cold column has a tiny qsat -> tiny deficit -> falling SNOW lands even at
    #     low RH (the _virga_diag DEFICIT_REF condition). The global E=P budget is unchanged: this RELOCATES
    #     surface precip (re-evaporated vapour advects away), it is not a water/energy sink. See
    #     ``_below_cloud_reevap``. Adds no prognostic state -> checkpoints unchanged.
    precip_reevap: bool = field(default_factory=lambda: _env_flag("COSMO_PRECIP_REEVAP", False))   # ⚠️ TEST default False; CANONICAL = True (canonical_env.sh).
    precip_reevap_rate: float = field(default_factory=lambda: _env_float("COSMO_PRECIP_REEVAP_RATE", 1.5e-5))
    #   ^ below-cloud rain-evaporation RATE coefficient K_e [1/s]: the per-layer flux loss is
    #     E_k = min( F, K_e * max(qsat_k - q_k, 0) * dm_k ), i.e. the evaporation rate is PROPORTIONAL to the
    #     ABSOLUTE saturation deficit (qsat - q) [kg/kg] -- a Kessler/Sundqvist-class absolute-deficit DRIVER
    #     (NOT the (1-RH) form), in BULK (no ventilation / (rho*q_r)^0.525 rain-content term: the "heavy
    #     survives, drizzle virgas" behaviour comes from the flux LIMIT E<=F, not a content-dependent rate).
    #     Three consequences fall out: it SELF-LIMITS near saturation (deficit -> 0 => no evaporation, rain
    #     lands); a COLD column has a tiny qsat hence a tiny absolute deficit => its falling SNOW lands even at
    #     low RH (the _virga_diag DEFICIT_REF correctness condition); and the fractional loss per layer
    #     E_k/F ~ K_e(qsat-q)dm_k/F FALLS with F, so heavy rain reaches the ground while drizzle virgas.
    #     GRID-CONVERGENT (rate-limited E_k ∝ dm_k => the column loss is K_e∫(qsat-q)dm, a mass integral).
    #     K_e ~ 1.5e-5/s gives a few-km full-virga depth for moderate rain in dry air (scaling linearly with
    #     rain rate); the REPORTED sensitivity (roadmap rule 4/5), a best estimate NEVER fitted to a target.
    # --- PROGNOSTIC RAIN q_r (stretched-x program stage 7, STRETCHED_X_PLAN.md §5; the chimney.py
    #     q_r port -- CHIMNEY_B4_RESULTS.md is the reference physics). Master flag
    #     (env COSMO_PROGNOSTIC_RAIN), default OFF -> BYTE-IDENTICAL (self.q_r is None, no code path
    #     touched). Requires ``prognostic_condensate`` (q_r sources from the autoconversion of q_c).
    #     ON: the autoconversion+ceiling sink of ``_prognostic_microphysics`` SOURCES a prognostic
    #     falling-rain reservoir q_r (nx, nz) instead of the instant same-substep column rainout --
    #     at resolved updraft speeds w ~ 25-36 m/s >> the ~7 m/s rain terminal velocity the instant
    #     fall is structurally wrong (condensate must LOFT with the flow). q_r then (a) rides the SAME
    #     divergence-free resolved mass flux as q_c (``_advect_rain``, one more flux-form call per
    #     dynamics step); (b) SEDIMENTS at -rain_v_term via a per-column implicit backward-Euler solve
    #     each thermo substep (``_sediment_rain``; v_term*dt/dz >> 1 makes explicit impossible), whose
    #     surface-face flux is the rain reaching ground -> ``self._precip``; (c) RE-EVAPORATES in
    #     sub-saturated air (``_rain_reevap``: deficit-limited with the dqsat/dT Newton denominator,
    #     relaxing over rain_reevap_tau -- the chimney's ``_reevap_condensate`` closure), with the
    #     latent COOLING on the real prognostic T_atm -- the COLD-POOL mechanism the resolved axis
    #     needs. Checkpointed/device-placed exactly like q_c ("q_r" key only when ON). Supersedes the
    #     within-step ``precip_reevap`` fall on the ON path (q_r's own re-evap is its prognostic
    #     replacement; ``_below_cloud_reevap``'s docstring names exactly this refinement). NAMED WALLS
    #     (draft): single-moment Kessler-class bulk rain; ONE v_term for liquid AND ice (no snow fall
    #     speed, no melt L_FUS bookkeeping -- ice-formed precipitate that re-evaporates in a warm layer
    #     drops the L_FUS conjugacy term _below_cloud_reevap carries); CPU + eager-GPU only
    #     (capture_ready raises NotImplementedError -- the conv-gate precedent).
    prognostic_rain: bool = field(default_factory=lambda: _env_flag("COSMO_PROGNOSTIC_RAIN", False))
    rain_v_term: float = field(default_factory=lambda: _env_float("COSMO_RAIN_VTERM", 7.0))
    #   ^ rain terminal fall speed [m/s] (~5-10; matches chimney.py's v_term=7.0 -- the lofting-
    #     competition knob: resolved w > v_term keeps rain aloft, w < v_term lets it sediment out).
    rain_reevap_tau: float = field(default_factory=lambda: _env_float("COSMO_RAIN_REEVAP_TAU", 1.0e3))
    #   ^ q_r below-cloud re-evaporation relaxation timescale [s] (matches chimney.py's reevap_tau=1e3;
    #     per-substep fraction 1 - exp(-dt/tau), dt-consistent). Deficit-limited, so it self-limits at
    #     saturation regardless of tau; the REPORTED sensitivity, a best estimate never fitted.
    # FLASH-BOIL VIRGA (COSMO_VIRGA_FLASH_BOIL, default OFF -> byte-identical): a sub-cloud layer above its
    # boiling point (esat >= 0.999 p, the qsat-clamp/supercritical condition) cannot hold falling liquid AT
    # ALL -- liquid water cannot exist above its boiling point regardless of vapour loading, so the constraint
    # is "liquid can't persist", NOT "there is room up to saturation" (#292; bound corrected by #294: the
    # original sat_cap deficit bound was ~0 in a steam-saturated boiling layer, so heavy rain passed through
    # to a >100 C surface -- the exact failure the flag was written to prevent). ON, the reevap ceiling in a
    # boiling layer becomes max(boil_cap, the ordinary deficit-limited cap), with the flash-boil ENERGY cap
    # boil_cap = (T - T_boil) cp dm / (L_VAP dt) -- the flux whose evaporative cooling lands the layer
    # exactly at its own boiling point T_boil(p) (thermo.T_boil, esat(T_boil)=0.999p). A maximum() FLOOR,
    # not a replacement (PR #296 review): the deficit-driven rate keeps acting at/near T_boil (rain
    # wet-bulb-cools the layer BELOW boiling and out of the mask, as with the flag OFF), so ON evaporation
    # >= OFF everywhere and a rain-cooled layer can never PIN at T_boil into a permanent pass-through; in
    # the motivating steam-saturated state (deficit ~ 0 -> rate ~ 0) the floor reduces to the pure energy
    # cap. The FLASH itself is q/energy-bounded (per step dq <= cp(T-T_boil)/L_VAP; its cooling stops at
    # T_boil -- the rate-limited arm and phase-1 q_c drain may cool a little further, exactly as OFF does).
    # q MAY exceed the qsat clamp there (physical: the layer is steam); pair with cond_boil_suppress below,
    # else the next step's saturation adjustment re-condenses the excess (a flash<->condense loop).
    # THREADED into the fused numba/device micro kernels (#295 threaded the old bound and removed the ~4x
    # generic-path routing; #294 corrected the branch body in place -- an int flag the kernels branch on,
    # 0 -> byte-identical shipped arithmetic), so the canonical fused CPU/GPU paths stay fused with this ON.
    virga_flash_boil: bool = field(default_factory=lambda: _env_flag("COSMO_VIRGA_FLASH_BOIL", False))
    # BOILING-LAYER CONDENSATION SUPPRESSION (COSMO_COND_BOIL_SUPPRESS, default OFF -> byte-identical; #294's
    # preferred UPSTREAM fix): above boiling there is NO vapour-liquid equilibrium at that pressure -- water
    # is entirely vapour -- but the qsat clamp (e <= 0.999 p) hands the microphysics a finite qsat, so any
    # cooling of a moist boiling layer "condenses" spurious q_c -> precip over the hot core (the spurious
    # equatorial rain source under an intense focus). ON: in boiling layers (esat >= 0.999 p) the
    # condensation source is ZERO (PDF qc_eq -> 0 / grid-mean & single-pass cond -> 0), NO precip forms
    # there (precip_demand -> 0), and suspended/advected q_c drains back to vapour through the normal
    # re-evaporation demand -- so advected moisture crosses the hot zone as VAPOUR (the physically-correct
    # pass-through: the hot inner zone is a moisture conduit, not a rain regime). THREADED into the fused
    # numba/device micro kernels like virga_flash_boil (0 -> byte-identical).
    cond_boil_suppress: bool = field(default_factory=lambda: _env_flag("COSMO_COND_BOIL_SUPPRESS", False))
    # SURFACE FLASH-BOIL (COSMO_SURF_FLASH_BOIL, default OFF -> byte-identical; the SURFACE-side completion
    # of #294 -- OPEN_ISSUES spin-off (B), cross-referenced from #297): rain formed in SUB-boiling air falls past both
    # #294 flags (they key on boiling AIR layers) yet can land on a skin far above ITS boiling point (the
    # equatorial salt flat runs 400-650 C) -- liquid cannot wet a super-boiling surface, so letting it enter
    # the wetness/endorheic supply as landed water is unphysical. ON: at the surface-exchange step, where
    # T_surf > T_boil(P_SURF) (thermo.T_boil, the model's own esat inverse -- the #294 mask at the surface
    # pressure), the incoming surface-reaching flux flash-evaporates ON CONTACT up to the surface ENERGY cap
    #     surf_cap = (T_surf - T_boil(P_SURF)) * C_surf / (L_VAP * dt)      [kg/m^2/s]
    # (C_surf = the SEB's own per-column surface heat capacity, _surf_heat_capacity): the flux whose latent
    # debit cools the skin exactly to its boiling point. The flashed mass becomes VAPOUR in the lowest layer
    # (a q source, like evaporation) and its latent heat L_VAP*flash is debited from the SURFACE energy
    # budget (through the same seb_incr the fp32 freeze-fix carries), so by construction the flash cools the
    # skin toward -- never below -- T_boil (exact on the fp64 path; under thermo_fp32 the fp32-scratch cap
    # vs fp64-absolute debit can land a cap-limited skin ~1 fp32 ULP, ~6e-5 K, below -- no consequence).
    # Flux beyond the cap lands as liquid NORMALLY (wets, supplies the
    # endorheic budget): once the skin is cooled to boiling it is no longer superheated -- the #296 review's
    # max()/continuity lesson applied at the surface (no pinned dead state; closure continuous at the mask
    # edge as the cap -> 0). The flashed portion is EXCLUDED from the _wetness supply (it never wetted
    # anything); ``_precip`` keeps its meaning (precip flux LEAVING the atmosphere, pre-flash) and the
    # flashed flux is diagnosed separately (``_surf_flash`` / cum_surf_flash in water_conservation).
    # An energy-budget contact closure, NOT droplet physics (no Leidenfrost microphysics); inert unless
    # couple_surface (a prescribed skin has no energy budget to debit). Lives entirely in the xp surface
    # block (capture-safe in-place buffer + static branch) -- the fused micro kernels are untouched.
    surf_flash_boil: bool = field(default_factory=lambda: _env_flag("COSMO_SURF_FLASH_BOIL", False))
    # --- PROGNOSTIC SURFACE-WATER BUDGET (COSMO_SURF_WATER_BUDGET, default OFF -> byte-identical; GitHub
    #     #302). The T-ramp wetness (wetness_of_Tsurf: wet below 55 C, dry above 70 C, keyed on T_surf
    #     ALONE) wets ANY cooled surface with zero supply -- `_wetness = max(ramp, budget_f)` means cooling
    #     alone fabricates a sea on an interior salt flat (the #302 mechanism gating the full cool-wet
    #     collapse). ON: wet/dry becomes a CONSERVATION statement -- a prognostic per-column surface water
    #     column W(x) [kg/m^2 = mm] evolves as
    #         dW/dt = P_landing - E_actual + d/dx(lateral creep transport)
    #     with P_landing = the surface-reaching LIQUID (lagged `_precip` minus the #298 `_surf_flash`
    #     flashed portion -- flux definitions shared with the flash exclusion), E_actual = the SEB's own
    #     evaporation HARD-LIMITED by the available water (E <= W/dt per substep; the SEB's latent heat
    #     uses the SAME limited E -- energy-water consistent), and the transport the salt_sill.py
    #     wind-drift creep (see surf_water_drift/surf_water_h_layer below). Wetness (evaporation
    #     efficiency) = clip(W / surf_water_wmin, 0, 1): the T-ramp is NO LONGER a wetness source -- its
    #     physical content (hot surfaces can't hold water) EMERGES from the budget (hot => E huge => W
    #     drains => dry; a salt flat is simply the W=0 geography). SUBSUMES budget_wet_fraction /
    #     budget_endorheic (one water accounting: supply-limited wetness and the endorheic pool are what
    #     the budget does natively); those flags are ignored when this is ON. W is checkpointed prognostic
    #     state ("W_surf"); a checkpoint without it (any pre-flag checkpoint, either attractor) initializes
    #     W from the LEGACY wetness diagnosis of the loaded state -- columns with wetness_of_Tsurf(T_surf)
    #     >= 0.5 get the deep reservoir surf_water_reservoir, the rest 0 ("the seas that exist get their
    #     water"; stamped in the log). W is carried fp64 ALWAYS (an (nx,) surface field; per-substep
    #     increments ~1e-3 kg/m^2 against a ~1.5e5 kg/m^2 reservoir would bit-freeze in fp32 -- the SEB
    #     freeze-fix class of bug, pinned by test). Capture-safe (persistent in-place (nx,) buffers,
    #     static branches). Brine/halite chemistry is OPT-IN: ``surf_salt`` (below) adds the dissolved/
    #     deposited salt fields and the halite dam that terminates a slow creep flood; with it OFF the
    #     flats are just W=0 ground (no salt sill -- the pre-surf_salt documented caveat).
    surf_water_budget: bool = field(default_factory=lambda: _env_flag("COSMO_SURF_WATER_BUDGET", False))
    surf_water_wmin: float = field(default_factory=lambda: _env_float("COSMO_SURF_WATER_WMIN", 10.0))
    #   ^ wetness saturation depth W_min [kg/m^2 = mm]: wetness = clip(W/W_min, 0, 1). ~10 mm is the
    #     ponding/film depth at which a surface evaporates as open water (of order the diurnal evaporation
    #     depth E_w*1day ~ a few mm and standing-film scales); a smooth, principled ramp -- NOT a tuning
    #     knob. Sensitivity (documented in WATER_BUDGET_RESULTS.md): W_min only shapes the partial-wetness
    #     sliver at a front/drying edge (cells hold either ~0 or >> W_min at steady state); 5-20 mm moves
    #     the front's wetness onset by < one cell-fill time, not the budget.
    surf_water_reservoir: float = field(default_factory=lambda: _env_float("COSMO_SURF_WATER_RESERVOIR", 1.5e5))
    #   ^ init W [kg/m^2] for columns the LEGACY wetness calls wet (>= 0.5) when no checkpointed W exists:
    #     1.5e5 kg/m^2 = 150 m of water = RHO_LIQ * surf_water_h_layer -- the model ocean is exactly its
    #     wind-driven transport layer (the abyss below does not participate in the creep), so the reservoir
    #     supplies the transport at full salt-sill capacity from step one. Deep enough that net E-P
    #     (~mm/day) cannot drain it on any practical run (~1e4+ days).
    surf_water_h_layer: float = field(default_factory=lambda: _env_float("COSMO_SURF_WATER_HLAYER", 150.0))
    #   ^ transportable-depth cap H_layer [m]: the creep flux moves at most the wind-driven surface layer,
    #     F = u_creep * min(W, RHO_LIQ*H_layer). EXTRACTED from salt_sill.solve_sill's H_layer=150 m (the
    #     doc's f=0 wind-driven layer depth, ~100-300 m: no rotation => no Ekman skin, stress mixes down to
    #     the stratification limit). At a deep reservoir edge this reproduces the salt-sill inflow capacity
    #     Q_in = u_current*H_layer (~15 m^2/s per unit along-track width) exactly; over a thin interior
    #     film it reduces to F = u*W (depth-limited creep). Sensitivity: doc range 100-300 m scales the
    #     reservoir-edge capacity linearly; the interior thin-film advance rate is H-independent.
    surf_water_drift: float = field(default_factory=lambda: _env_float("COSMO_SURF_WATER_DRIFT", 0.015))
    #   ^ surface-drift fraction: creep speed u_creep = drift * (signed resolved surface wind u_surf; the
    #     fixed equatorward -drift*cfg.wind when no dynamics). EXTRACTED from salt_sill's u_current (the
    #     Earth "~1.5-3 % of wind" surface-drift heuristic; 0.015 * the 6 m/s default cfg.wind = 0.09 m/s
    #     ~ salt_sill's 0.10 m/s, giving a fallback reservoir-edge capacity 0.09*150 = 13.5 m^2/s ~
    #     salt_sill's Q_in = 15). f=0 removes the Coriolis brake, so the real drift is if anything FASTER --
    #     0.015 is the conservative lower end (an advance-rate LOWER bound). Sensitivity: front advance
    #     speed and reservoir-edge capacity both scale linearly with it (0.01-0.03 doc-plausible).
    surf_water_grav_flow: bool = field(default_factory=lambda: _env_flag("COSMO_SURF_WATER_GRAV_FLOW", False))
    #   ^ GRAVITY LEVELING FLOW (default OFF -> byte-identical): a second, gravity-driven interface flux
    #     alongside the wind creep. Without it the budget's ONLY lateral transport is wind drift -- nothing
    #     knows water is deeper in one cell than its neighbor, so deep pools stand beside starved cells,
    #     water piles without bound at wind stagnation points, and the poleward reservoir edge never
    #     spreads (the run10/run12 findings). The term: surface height eta = W/RHO_LIQ + S_dep/RHO_HALITE
    #     [m] (CRUST IS TOPOGRAPHY -- ponding behind crust sills emerges from the gradient, so the term
    #     RETIRES the binary halite dam: the dam block is skipped when this flag is ON, kept verbatim on
    #     the OFF path); donor = the higher-eta side; speed u_g = min(sqrt(G_ACCEL*head), u_cfl) (crude
    #     gravity-current / dam-break scaling -- deliberately NOT thin-film hydraulics, see conn_frac);
    #     depth h_eff = min(donor free depth, head/2) (with the CFL clip this bounds each face's
    #     transfer at head/8 of its OWN head -- a PER-FACE no-overshoot guarantee, so locally-monotone
    #     profiles level without sloshing at any dt; a local extremum flanked by much larger heads can
    #     transiently reorder via its neighbors' faces, but the update stays conservative and bounded
    #     and levels to the same equilibrium). Named walls: the
    #     flow LAW is crude (Froude~1 front, no Manning friction -- rate-soft, equilibrium-eta exact);
    #     offline (eq_cycle) the rate is additionally sub-cycle-capped (see eq_cycle n_sub).
    surf_water_conn_frac: float = field(default_factory=lambda: _env_float("COSMO_SURF_WATER_CONN_FRAC", 0.5))
    #   ^ percolation cutoff for the gravity flow (fraction of surf_water_wmin): only water ABOVE
    #     W_conn = conn_frac * wmin is connected enough to flow laterally; below it the cell is
    #     disconnected pools on a flat (the wet_frac < 1 sub-grid picture) and gravity moves nothing --
    #     a draining cell parks AT W_conn rather than flip-flopping across a binary gate. 0.5 sits in
    #     the 2-D site-percolation-threshold range (~0.5-0.59); FREE within ~0.3-1.0 -- swept, never
    #     tuned to an outcome (FREE_PARAMETERS.md). Wind creep is NOT cut by this (drift drags any film).
    # --- PROGNOSTIC SURFACE SALT + HALITE DAM (surf_salt; requires surf_water_budget, else warned +
    #     inert). Two (nx,) fp64 fields ride the water budget: DISSOLVED salt ``S_dis`` [kg/m^2]
    #     advects with the creep water at the donor cell's salinity; DEPOSITED halite ``S_dep``
    #     [kg/m^2] is immobile crust. An instantaneous, parameter-free saturation adjustment
    #     exchanges them each substep: dissolved capacity = surf_salt_saturation * W, the excess
    #     precipitates, any deficit re-dissolves crust -- so evaporating brine crusts over, and rain
    #     or arriving ocean water re-dissolves the crust. Salinity convention THROUGHOUT: kg salt per
    #     kg WATER (W is the water mass alone; salt is carried separately, not part of W).
    #     THE DAM: an interface is blocked (u_e = 0, both directions) where the crust HEIGHT
    #     S_dep/RHO_HALITE at either neighbor exceeds surf_salt_block_factor * the DONOR cell's
    #     moving-layer depth min(W, RHO_LIQ*H_layer)/RHO_LIQ -- an overtopping/weir criterion, so the
    #     blocking is FLOW-DEPENDENT: a deep, vigorous flow overtops (and then dissolves) a thin dam,
    #     a shallow creeping film is stopped by it, and a trace crust never dams (height ~ 0). This
    #     models the doc's salt sill that terminates the creep flood (closes the surf_water_budget
    #     "sill NOT modeled" caveat above). OFF -> byte-identical (no salt state, no dam, no new
    #     checkpoint keys). Named walls (documented, not modeled): dissolution KINETICS (the
    #     adjustment is instantaneous equilibrium; a rate limit would add a free timescale),
    #     MECHANICAL dam erosion by the overtopping flow, and BRINE NON-IDEALITY -- saturated brine
    #     evaporates ~25% slower than the fresh-water E the SEB computes (water activity ~0.75,
    #     boiling elevated ~+7 C) and is ~1200 kg/m^3 dense while the weir depth divides by
    #     RHO_LIQ=1000 (~20% depth bias, inside the block_factor 0.5-2 sweep). All slow the
    #     concentration->crust endgame; none change its direction.
    surf_salt: bool = field(default_factory=lambda: _env_flag("COSMO_SURF_SALT", False))
    surf_salt_saturation: float = field(default_factory=lambda: _env_float("COSMO_SURF_SALT_SATURATION", 0.35))
    #   ^ halite saturation [kg salt / kg WATER] for the precipitate/dissolve exchange. Physically
    #     anchored, not a tuning knob: NaCl saturation 0.264 kg/kg-SOLUTION = 0.264/0.736 = 0.359
    #     kg/kg-water; 0.35 == salt_sill.py's S_HALITE = 350 g/kg and the cosmology doc's "~10x ocean".
    surf_salt_block_factor: float = field(default_factory=lambda: _env_float("COSMO_SURF_SALT_BLOCK_FACTOR", 1.0))
    #   ^ overtopping ratio in the dam criterion (dimensionless): blocked where crust height >
    #     factor * donor moving-layer depth. 1.0 = the physical weir default (a crust taller than the
    #     water moving toward it stops that flow); FREE within ~0.5-2 (FREE_PARAMETERS.md) -- swept,
    #     never tuned to an outcome.
    surf_salt_salinity0: float = field(default_factory=lambda: _env_float("COSMO_SURF_SALT_SALINITY0", 0.035))
    #   ^ initial / legacy-checkpoint-fallback / edge-source salinity [kg/kg-water], == constants.
    #     SALINITY_REF (Earth-ocean 35 g/kg-solution = 35/965 = 0.0363 kg/kg-water; keeping the
    #     familiar 0.035 figure is a ~3.6% relative convention delta -- far below any sensitivity
    #     here). One number, one home.
    surf_salt_dry_init: float = field(default_factory=lambda: _env_float("COSMO_SURF_SALT_DRY_INIT", 5000.0))
    #   ^ SEED SALT ON DRY-INITIALIZED CELLS [kg/m^2 of deposited halite crust S_dep]. The init
    #     (_init_surf_water) starts every HOT column DRY (W = 0); with the salt seeded proportional to
    #     water (S_dis = salinity0*W, S_dep = 0) those columns hold NO salt, and since halite only
    #     precipitates from evaporating BRINE they never acquire any -- so the model's "dry salt flat"
    #     carries zero salt in its own prognostic field. This knob makes a dry-initialized cell start
    #     as an evaporite pan: S_dep = surf_salt_dry_init where W = 0 at init (ocean cells stay
    #     crust-free). "Salt presence covers the dry assignment," at the source, not in a diagnostic
    #     patch (regions.dry_mask). Physically anchored, NOT tuned: 5000 ~ the salt left when one
    #     reservoir-depth of ocean evaporates (surf_water_reservoir*surf_salt_salinity0 = 1.5e5*0.035
    #     ~ 5250) ~ eq_cycle's "~135 m net evaporation builds the flat" (~4725) ~ 2.3 m of halite --
    #     LARGE as salt, SMALL as topography vs the 150 m reservoir (crust height 5000/RHO_HALITE ~
    #     2.3 m). MAGNITUDE IS SOFT/SWEEPABLE (an initial condition; the mature-flat mass is
    #     order-of-magnitude physical, not fitted). CEILING for anyone raising it: the crust is
    #     topography (eta += S_dep/RHO_HALITE) and dams an interface where its height exceeds
    #     block_factor*donor moving depth -- so keep it well below RHO_HALITE*block_factor*h_layer
    #     (~3.25e5 kg/m^2) or the seed will wall the ocean off the flats. 0.0 -> byte-identical to the
    #     prior surf_salt init (dry cells crust-free); the surf_salt=False path is unaffected either way.
    # --- DISSOLVED-SALT LATERAL LOCK EXCHANGE (salt_lock_exchange; the baroclinic BRINE RETURN. Requires
    #     surf_salt, else inert -- it moves S_dis, which only exists when surf_salt is ON). Default OFF ->
    #     byte-identical. A THIRD dissolved-salt rider, alongside the creep + gravity riders, but one that
    #     moves SALT with ZERO NET WATER: where two CONNECTED water columns differ in salinity the denser
    #     (saltier) water flows one way underneath while the fresher returns over the top -- an equal-volume
    #     two-layer exchange, so net water transport ~0 but net salt transport is nonzero. This is the salt
    #     return path the creep+gravity riders LACK (they only carry salt with a net water flux): without it
    #     an evaporating coastal margin concentrates brine and crusts halite with no way to relax the
    #     salinity contrast against its neighbours. Interface flux (donor-cell-free, gradient-driven):
    #       Delta_s = |sal_l - sal_r|            (salinity mass fraction, sal = S_dis/W)
    #       g' = G_ACCEL * 0.75 * Delta_s        [m/s^2]  reduced gravity (NaCl: Delta_rho ~ 0.75*rho*Delta_s)
    #       H_int = min over the interface of the CONNECTED water depth max(W-W_conn,0)/RHO_LIQ  [m]
    #       u_e = min(surf_salt_lock_ce * sqrt(g'*H_int), u_cfl)   [m/s]
    #       F = sign(dsal) * u_e * (H_int/2) * RHO_LIQ*|dsal|      [kg/m/s], applied antisymmetrically on
    #                                                              the creep telescoping -> EXACT salt conservation.
    #     H_int uses only CONNECTED water (the surf_water_conn_frac percolation cutoff): a disconnected
    #     film/pool on either side hosts no counter-flow -> H_int=0, no exchange (physical scoping). The
    #     u_cfl clip alone keeps each face's removal <= 0.125*S_dis_donor (positivity). Equilibrium (F->0)
    #     is equal salinity = equal density (the volume-weighted mean). NAMED WALLS: the linear equation of
    #     state (Delta_rho = 0.75*rho*Delta_s, and equal-salinity==equal-density); the crude Froude-scaled
    #     exchange RATE (surf_salt_lock_ce is the single free efficiency); untouched siblings OUT OF SCOPE
    #     (Raoult/activity evaporation suppression over brine; brine density in the gravity-flow eta).
    salt_lock_exchange: bool = field(default_factory=lambda: _env_flag("COSMO_SALT_EXCHANGE", False))
    surf_salt_lock_ce: float = field(default_factory=lambda: _env_float("COSMO_SALT_C_E", 0.25))
    #   ^ the lock-exchange efficiency c_e (dimensionless): the STRATIFICATION-ARREST factor on the
    #     gravity-current exchange velocity u_e = c_e*sqrt(g'*H_int). 0.25 is a physical mid-estimate --
    #     BELOW the ~0.45-0.5 energy-conserving lock-exchange front Froude (Benjamin 1968; the value the
    #     stage_walker_rce gravity-current gate recovers) because continuous stratification and interfacial
    #     mixing arrest a real exchange below the idealized dam-break front. FREE within ~0.1-0.5 (swept,
    #     never tuned to an outcome; FREE_PARAMETERS.md). Only acts when salt_lock_exchange is ON.
    # --- OFF-GRID OCEAN SUPPLY (surf_water_edge_source; requires surf_water_budget, else warned +
    #     inert). Dirichlet pin of the LAST cross-track cell (index nx-1 = the theta_max poleward
    #     edge, the upstream end of the fallback equatorward creep) to a full moving layer
    #     W = RHO_LIQ*surf_water_h_layer at surf_salt_salinity0 salinity: the model window sits on an
    #     INFINITE plane, so the poleward ocean continues off-grid and the closed wall is the box
    #     artifact -- this softens that named wall. The injected/removed mass is accounted
    #     (cum_w_edge / cum_s_edge) so the machine-precision surf_residual gate still closes with the
    #     pin active. OFF -> byte-identical (the #302 closed walls). Canonical: canonical_env.sh sets
    #     COSMO_SURF_WATER_EDGE_SOURCE=1 (a rider -- it only acts when the water budget itself is ON).
    surf_water_edge_source: bool = field(default_factory=lambda: _env_flag("COSMO_SURF_WATER_EDGE_SOURCE", False))
    # --- CRYO WETNESS PIN (surf_water_cryo_pin; requires surf_water_budget, else warned + inert).
    #     Below-freezing ground is ice/frost-covered, not dry salt flat: over any long horizon the cold
    #     trap accumulates an ice cover (deposition beats sublimation below freezing), which a ~100-day
    #     window can never grow from its own rain -- so the cover is PRESCRIBED, keyed on the physical
    #     freeze threshold, not on a chosen coastline x. Each substep, wherever T_surf <
    #     surf_water_cryo_tfreeze, W is FLOORED (one-way top-up, never a removal) at surf_water_cryo_w,
    #     so wetness = clip(W/wmin, 0, 1) reads exactly 1 there and the surface evaporates (sublimates)
    #     at its q_sat(T_surf)-limited rate -- tiny below 0 C, which is the physics, not a throttle.
    #     The surf_salt rider gives the injected water ocean salinity (salinity0), so later melt/creep
    #     carries salt-consistent water. Top-ups are ACCOUNTED (cum_w_cryo / cum_s_cryo) so the
    #     machine-precision surf_residual / salt_residual gates still close. A retreating freeze line
    #     leaves only ~10 mm behind (dries in days -- no melt-pulse ratchet); the deep reservoir stays
    #     the edge pin. OFF -> byte-identical (no new checkpoint keys). Watch item: pinned-wet cold
    #     ground can shift surface albedo -- an ice-albedo feedback on the freeze line (OPEN_ISSUES.md).
    surf_water_cryo_pin: bool = field(default_factory=lambda: _env_flag("COSMO_SURF_WATER_CRYO_PIN", False))
    surf_water_cryo_tfreeze: float = field(default_factory=lambda: _env_float("COSMO_SURF_WATER_CRYO_TFREEZE", 273.15))
    #   ^ freeze threshold [K]: the physical 0 C, not a tuning knob.
    surf_water_cryo_w: float = field(default_factory=lambda: _env_float("COSMO_SURF_WATER_CRYO_W", 0.0))
    #   ^ floored W [kg/m^2]; <= 0 (default) means "= surf_water_wmin" -- the minimum mass that reads
    #     wetness = 1 (an ice COVER, not an ocean). Explicit values stay independently sweepable, but
    #     must be >= the denormal snap floor W_SNAP_FLOOR (1e-30) -- validated at construction.
    # --- SALT-AEROSOL CCN (GitHub #237, AD-core side). The equatorial salt flats are a steady aeolian
    #     salt/dust SOURCE (the cell's ~12 m/s near-surface wind is supercritical; ../aeolian_salt.py), and
    #     salt is an excellent cloud-condensation nucleus (CCN). A PRESCRIBED cloud-droplet-number field
    #     ``N_d`` (clean-maritime ~100 ... salt-loaded continental >=1000 cm^-3, SWEPT) couples into the
    #     warm-rain/stratiform band through two precip-physics limbs + one decoupled radiative limb. The
    #     wind-keyed emergent SOURCE/TRANSPORT scheme is salt_ccn_source="transport" below (default
    #     "prescribed"). Master OFF (env COSMO_SALT_CCN) -> BYTE-IDENTICAL (no N_d field; both limbs inert).
    #     REQUIRES prognostic_condensate (the autoconversion + in-cloud-LWP machinery); mutually exclusive
    #     with the equatorial-precip diagnostics chimney_param / equator_norain_cells. Only the RATIO
    #     N_d/N_ref enters either limb, so the absolute units are free -- documented as cm^-3; N_d == N_ref
    #     is the clean zero-point (ratio 1 -> exact no-op). Report direction-firm / magnitude-SOFT (named
    #     wall = aerosol microphysics + crust-state erodibility); NEVER tuned to a target.
    salt_ccn: bool = field(default_factory=lambda: _env_flag("COSMO_SALT_CCN", False))
    salt_ccn_nd_ref: float = field(default_factory=lambda: _env_float("COSMO_SALT_CCN_ND_REF", 100.0))
    #   ^ reference (clean-maritime) droplet number N_ref [cm^-3]; the today-anchor. Only N_d/N_ref matters.
    salt_ccn_nd: float | None = field(default_factory=lambda: _env_float("COSMO_SALT_CCN_ND", None))
    #   ^ prescribed loaded droplet number N_d [cm^-3]; None -> defaults to N_ref (so master ON at default is
    #     a no-op). The SWEEP sets this away from N_ref (clean ~100 ... loaded ~1000-10000).
    salt_ccn_nd_file: str = field(default_factory=lambda: _env_str("COSMO_SALT_CCN_ND_FILE", ""))
    #   ^ OPT-IN (prescribed source only): path to a .npy (nx,) N_d field [cm^-3]. When set it supersedes the
    #     scalar/band-mask above with an arbitrary prescribed PROFILE -- the capture-safe way to feed the
    #     EMERGENT salt_transport field (built offline from a converged wind) without the capture-hostile
    #     in-step 'transport' refresh. Unset ("") -> the exact prior scalar/band-mask behaviour (byte-identical).
    salt_ccn_twomey: bool = field(default_factory=lambda: _env_flag("COSMO_SALT_CCN_TWOMEY", True))
    #   ^ Twomey cloud-albedo limb (acts only when master ON): more CCN -> smaller LIQUID droplet effective
    #     radius r_eff = CLOUD_REFF_LIQ*(N_ref/N_d)^(1/3) -> higher cloud SW optical depth -> brighter cloud.
    #     LIQUID only (salt is a warm-rain CCN; the band is warm-rain); the ice r_eff is untouched. Note the
    #     model's cloud LW limb is mass-based (no r_eff), so this moves cloud SW albedo, not LW greenhouse.
    salt_ccn_autoconv: bool = field(default_factory=lambda: _env_flag("COSMO_SALT_CCN_AUTOCONV", True))
    #   ^ warm-rain autoconversion-efficiency limb (acts only when master ON): scale the autoconversion DEMAND
    #     by (N_d/N_ref)^beta. SIGN-AMBIGUOUS -- run as TWO bracketing experiments: beta<0 SUPPRESSION (sub-um
    #     CCN -> smaller, more numerous drops -> delayed coalescence) and beta>0 INVIGORATION (giant salt CCN
    #     seed large drops). Routes the micro step to the cupy-generic path (excluded from _micro_canonical).
    salt_ccn_autoconv_beta: float = field(default_factory=lambda: _env_float("COSMO_SALT_CCN_BETA", -0.8))
    #   ^ exponent on (N_d/N_ref) for the autoconv rate; |beta| ~ the warm-rain susceptibility So in 0.3-1.2
    #     (a swept BRACKET, not KK's single -1.79). Default -0.8 (mid-range suppression); set >0 to invigorate.
    salt_ccn_band_only: bool = False
    #   ^ uniform N_d (default) vs BAND-MASKED: load N_d only within [theta_lo, theta_hi] (else N_ref), to
    #     isolate band-local microphysics from upstream convective-source loading (a cheap second sweep run).
    salt_ccn_band_theta_lo: float = 40.0   # band-mask cross-track theta window [deg] (band_only mode only).
    salt_ccn_band_theta_hi: float = 60.0
    salt_ccn_haze: bool = field(default_factory=lambda: _env_flag("COSMO_SALT_CCN_HAZE", False))
    #   ^ DECOUPLED salt-haze SW-dimming limb (default OFF; acts only when master ON): an N_d-keyed
    #     salt-aerosol SW optical depth dims the surface, reusing the constants.OMEGA0_HAZE single-scattering
    #     split (scattered fraction re-emerges as diffuse-down, absorbed fraction lost). STRUCTURALLY SEPARATE
    #     from the precip limbs above so a haze issue never gates the precip deliverable (Jake, #237). The
    #     dimming optical depth at N_d is tau_salt = salt_ccn_haze_tau_ref * max(N_d/N_ref - 1, 0).
    salt_ccn_haze_tau_ref: float = field(default_factory=lambda: _env_float("COSMO_SALT_CCN_HAZE_TAU", 0.02))
    #   ^ salt-haze SW optical depth PER UNIT of (N_d/N_ref - 1) [-]; a best-estimate aerosol burden, REPORTED
    #     as a sensitivity, never fitted. The LAW is tau = tau_ref * (N_d/N_ref - 1), so tau_ref=0.02 gives a
    #     thin haze at low loading (tau~0.02 at 2x N_ref) GROWING LINEARLY to tau~0.18 at 10x and ~0.20 at the
    #     continental >=1000 cm^-3 (10x maritime) -- a thin-to-moderate haze, not a fixed-per-decade value.
    salt_ccn_haze_tau_cap: float = field(default_factory=lambda: _env_float("COSMO_SALT_CCN_HAZE_TAU_CAP", 3.0))
    #   ^ PHYSICAL CEILING on the salt-haze SW optical depth: tau_salt = min(tau_ref*(N_d/N_ref-1), tau_cap)
    #     (#237 AREA 1, roadmap item 1). The band-anchored LINEAR tau form above over-extrapolates at the
    #     SOURCE loading -- at the transport scheme's near-source N_d (~1e5 cm^-3) the bare linear form gives
    #     tau ~ tens, which is UNPHYSICAL: real accumulation-mode number (and hence AOD) is capped at
    #     dust-storm class by coagulation (timescale ~hours at 1e11 m^-3) and advective export, neither of
    #     which the steady well-mixed N_d carries. The cap holds the haze to a strong-dust-storm AOD so the
    #     limb stays physical over the source. DEFAULT 3.0 = a representative strong dust storm; SWEEP it
    #     across the dust-storm range O(1-10) -- it is NAMED-WALL soft (coagulation + export microphysics),
    #     reported as a sensitivity, NEVER tuned to a target. tau_cap=inf recovers the bare unbounded linear
    #     form (the pre-cap behaviour). The existing band-loading sweeps (tau < ~2.5) are below the cap, so
    #     this is byte-identical there; it only bites at the source-class loading the cap exists to bound.
    # --- #237 source/transport scheme: make N_d EMERGENT from the resolved wind (vs the prescribed field above) ---
    salt_ccn_source: str = field(default_factory=lambda: _env_str("COSMO_SALT_CCN_SOURCE", "prescribed"))
    #   ^ 'prescribed' (DEFAULT): N_d is the scalar/band-masked prescribed field above (the #243 AD-core sweep).
    #     'transport': N_d is built by ``salt_transport`` (the wind-keyed saltation/dust emission off the flats,
    #     size-binned, cell-lofted, settled along the poleward-aloft return -> the accumulation mode reaches the
    #     band as CCN). The resolved wind is unknown at construction, so transport mode starts CLEAN (N_d == N_ref,
    #     an exact no-op) and is refreshed from the live ``_u_surf`` either every ``salt_ccn_transport_refresh_every``
    #     steps or by an explicit ``_build_salt_ccn_field()`` re-call after spin-up. Needs a resolved wind
    #     (``run_dynamics``); ``salt_ccn_nd`` / ``salt_ccn_band_only`` are ignored (the field is emergent).
    salt_ccn_transport_alpha: float = field(default_factory=lambda: _env_float("COSMO_SALT_CCN_ALPHA", 3.0e-5))
    #   ^ sandblasting efficiency alpha = F_dust/Q_saltation [1/m]; the dominant emission-magnitude knob (SWEPT,
    #     Owens-Lake/Bodele class ~1e-5..1e-4). Named-wall soft (crust-state erodibility).
    salt_ccn_transport_flats_theta_hi: float = field(
        default_factory=lambda: _env_float("COSMO_SALT_CCN_FLATS_HI", 25.0))
    #   ^ poleward edge of the erodible salt flats [deg] (the salt sill; equatorward is the bare evaporite pan).
    salt_ccn_transport_vdepmin: float = field(default_factory=lambda: _env_float("COSMO_SALT_CCN_VDEPMIN", 5.0e-4))
    #   ^ dry-deposition velocity floor [m/s] bounding the accumulation-mode lifetime (SWEPT; aerosol microphysics).
    salt_ccn_transport_refresh_every: int = field(
        default_factory=lambda: int(_env_float("COSMO_SALT_CCN_REFRESH", 0.0)))
    #   ^ rebuild the emergent transport N_d from the live wind every N steps (0 = OFF: build once + explicit
    #     re-call only). One-way + second-order, so a periodic refresh as the wind converges suffices. The refresh
    #     is skipped under GPU graph capture (rebuild between captured segments instead).
    # --- DYNAMIC salt-CCN production (COSMO_SALT_CCN_DYNAMIC, default OFF -> byte-identical): make N_d
    #     PROGNOSTIC-AT-CADENCE instead of the frozen prescribed/file profile -- the salt-cloud negative
    #     feedback (thermostat) the static field structurally lacks. Every collapse experiment to date ran
    #     with N_d frozen at hot-dry values through the whole collapse; physically, when the world cools/
    #     wets and the flats shrink, the salt source collapses toward the marine background -> clouds dim
    #     (Twomey) and rain out faster (the (N_d/N_ref)^beta autoconv suppression relaxes) -> sunlight
    #     returns -> the surface re-dries. The refresh pipeline (``StageADCore.refresh_salt_ccn``, called by
    #     the DRIVER at a replay-batch boundary every ``salt_ccn_refresh_days`` -- NEVER inside step()/a
    #     captured graph; that is exactly why the old per-step 'transport' refresh was abandoned as
    #     capture-hostile):
    #       1. SOURCE MASK: dry cells, wetness < salt_ccn_dry_thresh (value-level ``_wetness()``, so it
    #          works under BOTH wetness models -- the legacy T-ramp/budget and the #302 W-based wetness).
    #       2. EMISSION: the phase-2 ``salt_transport`` physics (Shao&Lu threshold u*t + the Owen u*^3
    #          saltation law + sandblasting alpha) keyed on the LIVE |u_surf|, restricted to the mask;
    #          scaled by salt_ccn_emis_scale (the phase-2 normalization anchor, see below).
    #       3. TRANSPORT: salt_transport's own loft + poleward-aloft advection-settling reach (the
    #          phase-2-consistent choice; the offline `_phase2_build_nd.py` is this same pipeline).
    #       4. SINKS: precipitation scavenging -- one e-fold of aerosol removed per salt_ccn_scav_pmm mm
    #          of rain (rate P/P_scav per day, in steady competition with the tau-day lifetime:
    #          excess_ss = excess_raw / (1 + tau_days*P_mmday/P_scav)) -- plus relaxation of the LIVE N_d
    #          toward the target with timescale salt_ccn_tau_days (the aerosol-lifetime memory; when the
    #          mask empties the whole field decays to the marine background N_ref).
    #       5. IN-PLACE WRITE: the held N_d device buffer and every DERIVED consumer buffer (the fused
    #          micro kernels' cached warm_factor) are rewritten IN PLACE (addresses stable), so captured
    #          CUDA graphs replay the refreshed values -- the capture-safe design the per-step refresh
    #          lacked (it also reassigned N_d, silently orphaning the captured buffer).
    #     INIT: ON without an ND_FILE -> the first refresh_salt_ccn() call JUMPS N_d to the target built
    #     from the seed state; ON with an ND_FILE -> the file is the t=0 profile and dynamics take over
    #     (relaxation from it). The live N_d is CHECKPOINTED state ("N_d" in state()/STATE_KEYS, key
    #     present only when this flag is ON -- flag-OFF checkpoints stay byte-identical).
    #     Requires salt_ccn; mutually exclusive with salt_ccn_source='transport' (this scheme replaces it).
    #     Confidence: direction firm; magnitudes NAMED-WALL soft (aerosol microphysics + crust
    #     erodibility; the emission threshold/exponent + scavenging efficiency are documented
    #     sensitivities, never tuned to a target).
    salt_ccn_dynamic: bool = field(default_factory=lambda: _env_flag("COSMO_SALT_CCN_DYNAMIC", False))
    salt_ccn_refresh_days: float = field(default_factory=lambda: _env_float("COSMO_SALT_CCN_REFRESH_DAYS", 0.5))
    #   ^ refresh cadence [sim-days]. The driver fires the refresh at its replay-batch boundary, so the
    #     realized cadence granularity is the driver's diag batch (0.25 d at DIAG_STEPS=360/dt=60 --
    #     0.5 d divides cleanly). Well below the ~2 d aerosol memory, so cadence != physics.
    salt_ccn_tau_days: float = field(default_factory=lambda: _env_float("COSMO_SALT_CCN_TAU_DAYS", 2.0))
    #   ^ aerosol-lifetime memory tau [days]: the live N_d relaxes toward the refresh target as
    #     exp(-dt/tau) (accumulation-mode residence ~days against deposition/export; named-wall soft).
    salt_ccn_dry_thresh: float = field(default_factory=lambda: _env_float("COSMO_SALT_CCN_DRY_THRESH", 0.3))
    #   ^ dry-source threshold: a cell is an erodible source when _wetness() < this (0.3 = well onto the
    #     dry side of both wetness models' ramps; the mask is sharp at steady state -- cells sit at ~0 or
    #     ~1 -- so 0.2-0.5 moves only a drying-front sliver).
    salt_ccn_scav_pmm: float = field(default_factory=lambda: _env_float("COSMO_SALT_CCN_SCAV_PMM", 10.0))
    #   ^ wet-scavenging e-folding rain amount [mm]: rain removes aerosol at rate P_mmday/this per day
    #     (one e-fold of the local column per ~10 mm of rain -- accumulation-mode below/in-cloud
    #     scavenging class; named-wall soft, swept not tuned).
    salt_ccn_wind_tau_days: float = field(default_factory=lambda: _env_float("COSMO_SALT_CCN_WIND_TAU_DAYS", 1.0))
    #   ^ e-folding memory [days] of the |u_surf|^3 EMA that keys the DYNAMIC emission under
    #     COSMO_DIURNAL_SUN (#308 item 2): the refresh's 0.5-day cadence phase-locks an instantaneous
    #     wind sample against the diurnal cycle, and the Owen u*^3 convexity makes that O(1)-biased --
    #     so with the diurnal sun ON the refresh keys on cbrt(EMA(|u|^3)) instead (the EMA advances
    #     in-place inside step(), capture-recordable, exactly the _T_surf_slow pattern). 1 day spans
    #     one full cycle (>= the aliasing period) while staying ~2x faster than the aerosol memory.
    #     The "days" here are DIURNAL CYCLES: the conversion uses geometry.CLOCK.T_day (not 86400),
    #     so the one-cycle reading survives a clock change. Inert unless BOTH diurnal_sun and
    #     salt_ccn_dynamic are ON (byte-identical otherwise).
    salt_ccn_emis_scale: float = field(default_factory=lambda: _env_float("COSMO_SALT_CCN_EMIS_SCALE", 0.0585))
    #   ^ emission normalization (multiplies alpha_sandblast in the dynamic refresh only). ANCHORED, not
    #     free: nz_resolution_robustness/_salt_dynamic_anchor.py calibrates it so the refresh applied to
    #     the day-1000 hot-dry lens seed reproduces the production static profile
    #     (_phase2_Nd_emergent.npy, the field every static-N_d arm ran with) -- pattern corr > 0.9, peak
    #     within ~2x. DEFAULT 0.0585 = that calibration (excess ratio (12411-100)/(210633-100); the
    #     hot-dry state's u^3 emission is ~17x the cool-wet clean-base wind that built the reference, so
    #     anchoring to the production forcing rescales it back -- provenance/rerun:
    #     SALT_DYNAMIC_RESULTS.md). Within the alpha_sandblast named-wall (1-2 order) uncertainty.
    # --- Climatology accumulation (EQ-1 resume-hook speedup): maintain per-column true-time EMAs of the
    #     forcing fields the offline surface eq-cycle needs (T_surf, q_bl, exchange wind, E, P, surf_flash,
    #     u_surf), so ``eq_cycle.equilibrate_on_resume`` reads a matured climatology from the checkpoint
    #     instead of RE-INTEGRATING the core ~n_true_days in eager (pre-capture) mode (~hours/resume; see
    #     eq_cycle.py perf note + ROADMAP GPU-14). Diagnostic-only: the EMAs never feed physics, so ON vs
    #     OFF is byte-identical in every prognostic field (the checkpoint just gains keys when ON). Default
    #     ON -- the cost is a handful of on-device ops/step (no D2H sync, capture-legal, the _salt_u3_slow
    #     idiom) and the whole point is the mean being ready at any resume.
    clim_accum: bool = field(default_factory=lambda: _env_flag("COSMO_CLIM_ACCUM", True))
    clim_ema_tau_days: float = field(default_factory=lambda: _env_float("COSMO_CLIM_EMA_TAU_DAYS", 1.0))
    #   ^ e-folding memory [world-days; "days" convert through geometry.CLOCK.T_day] of the climatology
    #     EMAs. Calibrated for the PRODUCTION hook usage, which runs DIURNAL_SUN OFF -- there is no diurnal
    #     cycle to average out, so the EMA only needs to smooth CONVECTIVE transients (sub-daily); tau=1 day
    #     does that AND converges within the true-time a run realistically banks (true time accrues ~10-17x
    #     slower than nominal step-days on the lens world, so a multi-nominal-day run banks only ~1-3 TRUE
    #     days -- a longer tau would stay seed-dominated and never mature). ONLY IF running the hook with
    #     DIURNAL_SUN ON does the boxcar-vs-EMA phase-leakage matter (a first-order EMA attenuates the
    #     diurnal cycle by 1/sqrt(1+(2*pi*tau/T_day)^2) instead of nulling integer harmonics exactly: ~16%
    #     leak at tau=1 cycle, ~3% at tau=5); raise tau to >=5 there (the maturity gate scales with it).
    clim_maturity_tau_mult: float = field(default_factory=lambda: _env_float("COSMO_CLIM_MATURITY_TAU_MULT", 2.0))
    #   ^ maturity gate: climatology_state() returns the stored climatology ONLY once the EMAs have
    #     accumulated at least this many e-folding times (mult * tau) of TRUE time since seeding; below it
    #     the mean is still seed-dominated (at the t=0 initial condition), so the hook falls back to
    #     re-integration instead of silently injecting an unconverged forcing. 2*tau leaves the seed at
    #     ~e^-2 ~ 14% weight; with the tau=1 default the fast path engages after ~2 TRUE days of banked
    #     sim time. 0 disables the gate (any seeded state qualifies -- test use).
    # --- LW window band + humidity continuum (CLEARSKY_SW_PLAN Stage 2): default ON (env COSMO_LW_WINDOW=0
    #     to disable -> the gray byte-identical core). ``f_lw_window`` is a spectroscopic given (the window's
    #     Planck fraction) and ``k_cont_self`` is anchored to the observed tropical window opacity
    #     (radiation.py) -- the resulting hot-end OLR cap is a PREDICTED outcome, not a fitted target.
    lw_window: bool = field(default_factory=lambda: _env_flag("COSMO_LW_WINDOW", True))
    f_lw_window: float = field(default_factory=lambda: _env_float("COSMO_F_LW_WINDOW", rad.F_LW_WINDOW))
    # --- conserving band split (issue #84 item 3): the background band carries the out-of-window
    #     CLEAR-SKY opacity /(1-f_win), restoring the Earth-calibrated gray Planck-mean the window split
    #     otherwise dilutes (the "gray-baseline drift"). Default OFF = the accepted moved-baseline
    #     (byte-identical); the fleet's rad_consband variant A/Bs it (env COSMO_LW_CONSERVING_SPLIT=1).
    lw_conserving_split: bool = field(default_factory=lambda: _env_flag("COSMO_LW_CONSERVING_SPLIT", False))
    k_cont_self: float = field(default_factory=lambda: _env_float("COSMO_K_CONT_SELF", rad.K_CONT_SELF))
    # --- convective moisture closure (CLEARSKY_SW_PLAN Stage 3): HOW a convecting layer pair redistributes
    #     its (conserved) water. "q_homogenize" (byte-identical) homogenizes specific humidity ->
    #     the constant-q aloft slug (CLEARSKY_SW_FINDINGS root cause #1); "rh_detrain" redistributes toward
    #     the Manabe fixed-RH target (RH FALLS with height, no slug -- using rh_surf/rh_top above). Default
    #     "rh_detrain" (the all-on accurate model; env COSMO_CONV_CLOSURE=q_homogenize for the gray byte-
    #     identical slug variant). The fixed-RH profile is a modelling choice; q_homogenize is kept as the
    #     comparison variant, never silently dropped.
    convective_closure: str = field(default_factory=lambda: _env_str("COSMO_CONV_CLOSURE", "rh_detrain"))
    # --- SW diffuse stream + haze forward-scatter (CLEARSKY_SW_PLAN Stage 5): default ON (env
    #     COSMO_SW_DIFFUSE=0 to disable -> byte-identical). ON adds the haze forward-scattered diffuse-down
    #     surface SW (geometry.day_mean_diffuse_flux), recovering the slant-haze light the direct day-mean
    #     flux drops as pure loss. The OMEGA0_HAZE/G_HAZE split are physical givens (constants.py); the
    #     midnight-clock role of TAU0_HAZE (SunsClock) is untouched.
    sw_diffuse: bool = field(default_factory=lambda: _env_flag("COSMO_SW_DIFFUSE", True))
    # --- DIURNAL SUN COUPLING (DIURNAL_COUPLING_SCOPING.md chunks 4.1/4.2; default OFF -> byte-identical).
    #     ON, the IN-LOOP radiation reads the INSTANTANEOUS train-of-suns flux S(x, phase) (phase =
    #     true elapsed time mod T_day) instead of the static day-mean self.S -- from a precomputed one-day
    #     (diurnal_nphase, nx) table (geometry.diurnal_flux_table, per-column renormalized so its
    #     phase-mean == the day-mean forcing EXACTLY, lens/shade/truncation absorbed), plus the diffuse
    #     twin when sw_diffuse. Every OTHER self.S consumer (diagnostics, the energy gate, CRE, the
    #     spatial-shape use, init) keeps the DAY-MEAN (see stage_ad_core._radiate's sw_inst).
    #     ALSO gates the surface heat capacity (chunk 4.2): ON uses the PHYSICAL diurnal skin
    #     (C_BARE_ROCK dry / C_OCEAN_MIXED wet -- the swing-setting capacity); OFF keeps the fast
    #     spin-up slab (1 m rock / ocean_depth_spinup m ocean), which only sets the spin-up rate.
    #     PHASE CARRIER (PR #301 review finding 1): the phase is a function of TRUE ELAPSED TIME, never
    #     of the step count -- eager/CPU reads host sim_time; the captured path reads a (1,) float64
    #     DEVICE diurnal clock accumulated in place by the realized dt inside the graph (exact under the
    #     CFL-adaptive dt; row = floor(t/row_dur) mod n_phase computed in the gather kernel). Only static
    #     sanity (n_phase >= 8, tau > 0) is checked at construction; there is no row-per-step contract.
    #     WETNESS KEYING (finding 2): ON, the wetness ramp / endorheic-liquid gate / heat-capacity blend
    #     key on a SLOW surface temperature (EMA, tau = diurnal_wetness_tau; instantaneous-T keying
    #     re-wets the dry flats every night), and the budget leg's wet-tile demand E_w is smoothed with
    #     the same tau. diurnal_wetness_freeze=1 instead PINS the wetness field at the seed state's
    #     classification (the cheap 4.4 control). Superseded by #303's prognostic surface water when it
    #     lands (this is the legacy-wetness path's fix).
    diurnal_sun: bool = field(default_factory=lambda: _env_flag("COSMO_DIURNAL_SUN", False))
    diurnal_nphase: int = field(default_factory=lambda: _env_int("COSMO_DIURNAL_NPHASE", 1440))
    diurnal_wetness_tau: float = field(default_factory=lambda: _env_float("COSMO_DIURNAL_WETNESS_TAU", 86400.0))
    diurnal_wetness_freeze: bool = field(default_factory=lambda: _env_flag("COSMO_DIURNAL_WETNESS_FREEZE", False))
    # --- two-band shortwave (CLEARSKY_SW_PLAN Stage 1): default ON (env COSMO_SW_BANDS=0 to disable ->
    #     the gray one-beam SW, byte-identical). ON splits the solar beam into a transparent VISIBLE fraction
    #     (rad.F_VIS_SW, reaches the surface regardless of water-vapour path -> the surface PAR / transmission
    #     FLOOR) and an absorbing NEAR-IR fraction carrying rad.K_WV_SW_NIR; both are spectroscopic givens.
    sw_bands: bool = field(default_factory=lambda: _env_flag("COSMO_SW_BANDS", True))
    # --- vapor buoyancy (issue #80): the dynamics buoyancy uses the VIRTUAL temperature
    #     T_v = T(1 + q/eps)/(1 + q) (``thermo.virtual_temperature``) instead of T alone, so moist
    #     columns are buoyant relative to dry ones at the same T. NOT an inert offset: the correction
    #     is ~0 over the dry salt flat / cold tail but up to ~+20 K equivalent over the wet warm edge
    #     (q_sat ~ 0.10 at SST 55 C), i.e. a missing HORIZONTAL buoyancy gradient of the same order as
    #     the temperature contrasts that drive the cell, exactly where ascent (-> moisture loft, rain
    #     band) is set. Default ON (the all-on accurate model; env COSMO_VIRTUAL_BUOYANCY=0 for the
    #     T-only byte-identical limit). It is the VAPOR half of the density temperature; the condensate
    #     half is ``condensate_loading`` directly below.
    virtual_buoyancy: bool = field(default_factory=lambda: _env_flag("COSMO_VIRTUAL_BUOYANCY", True))
    # --- CONDENSATE LOADING (PHY-19 step 2): the weight of the SUSPENDED condensed water the column
    #     carries -- prognostic cloud q_c plus prognostic rain q_r -- reduces the buoyancy, so the density
    #     temperature is T_rho = T_v(T, q) - T*q_cond (``thermo.density_temperature``). For SPECIFIC
    #     humidities -- which is what this model carries -- T_rho = T(1 + 0.608 q - q_cond) is EXACT,
    #     straight from the gas law; the loading is not a linearization. ON, ``_update_buoyancy`` maps
    #     q_cond = max(q_c,0) + max(q_r,0) to the height grid alongside T and q and subtracts the
    #     loading. There is NO free parameter: q_cond is a prognostic mass fraction, coefficient 1.
    #     (The VAPOR half is a separate matter with its own switch, ``virtual_temp_exact`` below:
    #     ``virtual_temperature`` applies the mixing-ratio ratio form to a specific humidity. Default
    #     ON, canonical, opposite in sign -- ROADMAP PHY-20.)
    #     WHY: the model carries prognostic condensate and rain (COSMO_PROGNOSTIC_CONDENSATE=1,
    #     COSMO_PROGNOSTIC_RAIN=1 in the canonical world) but the buoyancy has always been vapor-only, so
    #     a column full of cloud and rain is as buoyant as a dry one at the same T_v -- a plain omission.
    #     SIZE, stated up front so it is not mistaken for a cure: at the PHY-19 sun-line axis the expected
    #     loading is ~1-2 K against ~30 K of vapor-buoyancy contrast. This is a CORRECTNESS term, and step
    #     2 of PHY-19 -- it is NOT the fix for the 2,081 mm/day axis rain column (flatworld/OPEN_ISSUES.md
    #     PHY-19, PHY17_D4_MIXER_RESULTS.md 4.4-4.5).
    #     DEFAULT OFF (env COSMO_CONDENSATE_LOADING=1): the OFF limit is byte-identical, gated at the
    #     PYTHON level (an ``if``, never a zero multiplier -- the step is captured into a CUDA graph and a
    #     multiplier would change the OFF graph). ORTHOGONAL to ``virtual_buoyancy``: with vapor buoyancy
    #     OFF and loading ON the buoyancy is T - T*q_cond (the loading alone). A NO-OP when the prognostic
    #     condensate/rain fields are absent (q_c stays 0, q_r is None), so it is safe to leave ON.
    #     TURN-ON ROUTES (corrected by the PR #839 review, finding 2 -- the old note said "not the
    #     environment", which is true only of the analysis loader and is the wrong advice for an arm):
    #     for a PRODUCTION arm use ``COSMO_CONDENSATE_LOADING=1`` in the launching shell --
    #     ``run_driver`` builds the config at :1024 and captures at :1134, so the key bakes in before
    #     capture, and it is recorded in ``world.cosmo_env`` as well as ``config``, so rebuilds
    #     reproduce it. Only a ``build_core_with_world`` ANALYSIS rebuild needs the post-build
    #     ``core.cfg.condensate_loading = True``, because that loader's ``assert_env_matches``
    #     (correctly) refuses a live COSMO_* key the checkpoint's world never recorded. See
    #     ``virtual_temp_exact`` below for the full statement of the trade.
    condensate_loading: bool = field(default_factory=lambda: _env_flag("COSMO_CONDENSATE_LOADING", False))
    # --- EXACT SPECIFIC-HUMIDITY DENSITY TEMPERATURE (ROADMAP PHY-20): ``thermo.virtual_temperature``
    #     computes T_v = T(1 + q/eps)/(1 + q + q_c), which is the EXACT density temperature for MIXING
    #     RATIOS -- but this model's humidities are SPECIFIC (``thermo.qsat`` returns
    #     eps*e/(p-(1-eps)e) = rho_v/rho). For specific humidities the exact form is
    #     T_rho = T(1 + (1/eps - 1) q - q_c) = T(1 + 0.608 q - q_c), straight from the gas law with no
    #     expansion. The two are the SAME function of the same parcel reached through different
    #     variables (r = q/(1-q_v-q_c) maps one onto the other identically), so the default path is not
    #     a dilute approximation -- it is the right function at the wrong variable, and it always
    #     UNDER-states the vapour buoyancy.
    #     SIZE (the gap has the closed form T*(1/eps-1)*q^2/(1+q) at q_c = 0, always positive): 1.82 K
    #     at T = 330 K, q = 0.10 -- about 9 % of the whole vapour increment -- and 13.60 K at q = 0.2964.
    #     WHY IT MATTERS: ``virtual_buoyancy`` is default ON and the AD core's dynamics torque on the
    #     virtual temperature, so the deficit sits in the term that drives ascent over the warm sea and
    #     the band's warm edge (q ~ 0.10-0.15), hence in the moisture loft and the rain band.
    #     ON, ``_update_buoyancy`` selects the exact form in BOTH the vapour-only and the
    #     condensate-loading branches (``thermo.virtual_temperature``/``density_temperature`` take the
    #     same ``exact=`` keyword, and both are array-generic, so CPU and GPU share one implementation).
    #     There is NO free parameter: the coefficient is 1/eps - 1, a constant of the gas law.
    #     DEFAULT OFF (env COSMO_VIRTUAL_TEMP_EXACT=1): OFF is byte-identical -- the default branch runs
    #     the identical expression -- and the branch is at the PYTHON level, never a blended multiplier
    #     (the step is captured into a CUDA graph and a multiplier would change the OFF graph).
    #     ORTHOGONAL to ``virtual_buoyancy`` and to ``condensate_loading``: both forms give exactly T at
    #     q = 0, so with vapour buoyancy OFF the flag is inert, and the loading term is ``-T*q_cond``
    #     under either form.
    #     TURNING IT ON MOVES CANONICAL NUMBERS -- it is a world change and needs its own impact arm.
    #     TWO TURN-ON ROUTES, and they are for different jobs (PR #839 review, finding 2):
    #       (a) A PRODUCTION ARM (``run_driver.py``) -- USE THE ENVIRONMENT: export
    #           ``COSMO_VIRTUAL_TEMP_EXACT=1`` in the launching shell. ``run_driver`` builds the core at
    #           :1024 (a fresh ``StageADConfig``, so the key is read AT CONSTRUCTION) and captures at
    #           :1134, 110 lines later, so the ON branch is the one recorded -- no ordering hazard. It
    #           does NOT call ``ckpt_world.assert_env_matches`` (it resumes via ``load_run_checkpoint``
    #           -> ``core.load_state``), so an unrecorded key does not trip any guard there. The key then
    #           lands in BOTH ``world.cosmo_env`` and ``config``, so every later rebuild reproduces it
    #           from the replayed env. This is the route the PHY-19 arms take.
    #       (b) A ``build_core_with_world`` ANALYSIS REBUILD -- use the post-build assignment:
    #           ``core.cfg.virtual_temp_exact = True``. That loader DOES call ``assert_env_matches``,
    #           which correctly refuses a live COSMO_* key the checkpoint's world never recorded, so the
    #           env route is unavailable there. ``_update_buoyancy`` re-reads the flag every call, so the
    #           assignment takes effect on the next EAGER step with no rebuild -- but SET IT BEFORE THE
    #           FIRST STEP on a capture_ready/GPU core: a CUDA graph records the Python branch that ran
    #           at capture and a replay does not re-enter this Python (the same ordering rule
    #           ``run_driver.apply_conv_max_sweeps`` documents).
    #     THE TRADE: route (b) is carried by no env key, so it is recorded only in the fingerprint's
    #     ``config``. ``ckpt_world._restore_physics_config`` now restores it on rebuild (before that it
    #     was silently dropped -- PR #839 finding 1), but route (a) remains the reproducible one for
    #     anything that will be checkpointed and re-analysed.
    virtual_temp_exact: bool = field(default_factory=lambda: _env_flag("COSMO_VIRTUAL_TEMP_EXACT", False))
    # --- VIRTUAL TEMPERATURE IN THE HYPSOMETRIC BRIDGE (ROADMAP PHY-21): ``_hydrostatic_z`` integrates
    #     dz = (R_d * T_edge / g) * dln p on the DRY temperature ``T_atm``. The hypsometric relation is
    #     exact only on the DENSITY (virtual) temperature -- p = rho R_d T_rho is what "ideal gas" means
    #     for moist air -- so every layer of a moist column comes out UNDER-THICK by exactly the factor
    #     T/T_rho = 1/(1 + 0.608 q - q_cond). This is NOT one of the anelastic approximations: the
    #     approximation lives in the REFERENCE state (rho_ref/T_ref/p_ref, dry by documented design,
    #     ``_update_buoyancy``), whereas ``_hydrostatic_z`` is a DIAGNOSTIC of the live column's actual
    #     geometry and is the ONE bridge between the thermo (pressure) grid and the dynamics (height)
    #     grid -- ``_pressure_to_height`` interpolates against the PHYSICAL height grid through it. A
    #     too-thin moist column therefore lands its T and q too LOW on the dynamics grid, samples the
    #     resolved u/w from the wrong altitude coming back, AND under-states the pair thickness the
    #     convective adjustment's lapse-rate test uses (``stage_ad_column`` dz_pair; the third in-step
    #     consumer) -- moist-column-selective, so a horizontal gradient error, not an inert offset. The
    #     same package already integrates this relation on T_v where someone worked it through
    #     (``plume_floft._env_height_of_p``).
    #     BUT "the approximation lives in the reference state, this is just live geometry" is only HALF
    #     the story, and the smaller half (PR #841 review F3; ROADMAP PHY-22). b = g(T_rho - T_ref(z))/
    #     T_ref(z) is the anelastic buoyancy only if the parcel at height z sits at the REFERENCE
    #     pressure p_ref(z); mapping mass levels through the LIVE column -- dry OR moist -- asserts
    #     p(z) != p_ref(z), and on the canonical state that prior departure carries a band-to-band
    #     delta-b contrast 7.4x this term's (with 5.256 % of column mass already above the dynamics lid,
    #     against 0.000 % on the reference map). So this flag is the correct COMPLETION of a choice the
    #     code already made, NOT the closing of a self-contained omission. PHY-22 files the convention
    #     itself; it is deliberately not touched here.
    #     SIZE, MEASURED on the canonical run17 checkpoint (``phy21_hydrostatic_virtual_t_sizing.py``):
    #     the 500 hPa level is 55 m (habitable band) to 240 m (axis) too low and the column top up to
    #     852 m (4.2 % of the 20.1 km domain depth); the whole-column band-mean |delta b| is
    #     9.5e-3 (habitable band) to 6.9e-2 m/s^2 (salt flat), and the BAND-TO-BAND CONTRAST of the
    #     band-mean delta b -- the part that torques the flow -- is 3.9e-2 m/s^2 against PHY-20's
    #     2.1e-2 on the same state, i.e. ~1.9x PHY-20 (both legs measured with virtual_temp_exact ON,
    #     i.e. PHY-21 GIVEN PHY-20).
    #     THE SIGN IS STATE-DEPENDENT. What is one-signed is the HEIGHT correction (dz > 0, monotone in
    #     q -- algebra). The BUOYANCY correction is delta_b ~ -(g/T_ref)(dT_rho/dz) dz, which follows the
    #     LAPSE RATE: positive under a normal lapse, NEGATIVE under an inversion. This world has a deep
    #     one (habitable-band T rises 281 -> 312 K from 676 to 551 hPa), so stretching the column pulls
    #     COLDER air to a fixed height: the band-mean delta b is positive in the lowest 2 km in all six
    #     bands but NEGATIVE in the 2-8 km slab in four of six, where PHY-20 is positive everywhere.
    #     The two terms therefore OPPOSE each other through the mid-troposphere on the canonical state,
    #     and the net effect on moist ascent is not a priori one-signed -- which is a reason to run them
    #     in ONE impact arm rather than assuming they add.
    #     ON, ``_hydrostatic_z`` integrates the SAME expression on the density temperature, which it
    #     builds from the LIVE ``self.q`` (clipped to [0, Q_MAX], as ``_update_buoyancy`` does) through
    #     the SAME ``thermo`` entry points ``_update_buoyancy`` uses, honouring ``virtual_temp_exact``
    #     (formula) and ``condensate_loading`` (whether the suspended cloud+rain mass is carried) -- so
    #     the bridge and the buoyancy can never disagree about what "density temperature" means, and the
    #     three flags stay separable. There is NO free parameter.
    #     (Condensate in the thickness is gated on ``condensate_loading`` deliberately, to keep PHY-19
    #     step 2's arm separable; it is worth ~2 m of the 695 m column-top error on the salt flat and
    #     ~0.01 m at the surface, measured -- the term is vapour, in practice.)
    #     DEFAULT OFF (env COSMO_HYDROSTATIC_VIRTUAL_T=1): OFF is byte-identical -- the source array is
    #     literally ``self.T_atm`` and every op below it is untouched -- and the branch is at the PYTHON
    #     level, never a blended multiplier (the step is captured into a CUDA graph and a multiplier
    #     would change the OFF graph). Array-generic, so CPU and GPU cannot diverge.
    #     TURNING IT ON MOVES CANONICAL NUMBERS -- it is a world change and needs its own impact arm.
    #     TWO TURN-ON ROUTES, and they are for different jobs -- the same split ``virtual_temp_exact``
    #     documents above (PR #839 review, finding 2):
    #       (a) A PRODUCTION ARM (``run_driver.py``) -- USE THE ENVIRONMENT: export
    #           ``COSMO_HYDROSTATIC_VIRTUAL_T=1`` in the launching shell. ``run_driver`` builds the core
    #           at :1024 (a fresh ``StageADConfig``, so the key is read AT CONSTRUCTION) and captures at
    #           :1134, so the ON branch is the one recorded -- no ordering hazard -- and it does NOT call
    #           ``ckpt_world.assert_env_matches``, so an unrecorded key trips no guard there. The key then
    #           lands in BOTH ``world.cosmo_env`` and ``config``, so every later rebuild reproduces it.
    #       (b) A ``build_core_with_world`` ANALYSIS REBUILD -- use the post-build assignment:
    #           ``core.cfg.hydrostatic_virtual_t = True``. That loader DOES call ``assert_env_matches``,
    #           which correctly refuses a live COSMO_* key the checkpoint's world never recorded, so the
    #           env route is unavailable there. ``_hydrostatic_z`` re-reads the flag every call, so the
    #           assignment takes effect on the next EAGER step with no rebuild -- but SET IT BEFORE THE
    #           FIRST STEP on a capture_ready/GPU core, for the CUDA-graph reason above.
    #     THE TRADE is the same one: route (b) is carried by no env key and is recorded only in the
    #     fingerprint's ``config``, from which ``ckpt_world._restore_physics_config`` (generic since
    #     PR #839 finding 1, so this key needs no list edit) restores it on rebuild; route (a) remains
    #     the reproducible one for anything that will be checkpointed and re-analysed.
    hydrostatic_virtual_t: bool = field(
        default_factory=lambda: _env_flag("COSMO_HYDROSTATIC_VIRTUAL_T", False))
    # --- THE MISSING BUOYANCY PRESSURE TERM (ROADMAP PHY-22, option B' of the scoping note
    #     ``flatworld/PHY22_MAPPING_CONVENTION_SCOPING.md``; the build decisions are its section 9):
    #     the dynamics eliminate pressure by writing the anelastic pressure term as the PERFECT GRADIENT
    #     -grad(p'/rho_ref), whose curl is exactly zero (``_dynamics_substep``'s own docstring says so).
    #     Splitting (1/rho_ref) grad p' = grad(p'/rho_ref) + (p'/rho_ref)(dln rho_ref/dz) zhat leaves, in
    #     the vorticity equation,
    #         b_eff = -g rho'/rho_ref - (p'/rho_ref) dln(rho_ref)/dz            (exact)
    #               ~  g tau  +  R_d (dT_ref/dz) eps                           (linearized)
    #     with tau = T'/T_ref and eps = p'/p_ref.  THE SHIPPED b = g tau OMITS THE SECOND TERM, and that
    #     term is proportional to the REFERENCE LAPSE RATE, R_d dT_ref/dz = -1.8655 m/s^2, NOT to
    #     g*kappa = +2.8014 m/s^2 (with this project's g = 9.8 and c_p = 1004) -- so "just switch to
    #     potential temperature" supplies 1.5017x the
    #     correction the core's own convention calls for, because the Lipps-Hemler pressure term is
    #     itself not a perfect gradient when dtheta_ref/dz != 0.  Derived by hand, confirmed with sympy,
    #     and re-derived independently from the Lipps-Hemler side; the two routes agree exactly
    #     (scoping note sections 1.3-1.4).
    #     IT IS THE LARGEST AVAILABLE CORRECTION.  As RMS torque -db/dx against the irreducible base
    #     torque on the canonical run17 state: this term 0.151x, PHY-21 0.073x, PHY-19 condensate
    #     loading 0.035x, PHY-20 0.002x -- larger than the three moisture terms together (0.110x).  It is
    #     first order in dz/H (how far the live columns' HEIGHTS depart from the base state, kilometres
    #     here), not second order in the temperature departure, which is why it outranks the O(q) terms.
    #     The ordering holds in 11 of 13 states sampled and BREAKS AT TWO OF EIGHT DIURNAL PHASES, where
    #     PHY-21 overtakes it 2.3-2.5x -- a quarter of the sampled cycle, so "largest" is a statement
    #     about a day mean, not about every instant (scoping note section 2.8).
    #     ONE DERIVED FORM AND ONE SENSITIVITY LEG -- NOT two co-equal derivations (PR #843 review F1).
    #     ``linear`` IS THE UNIQUE first-order completion for this reference state under ANY
    #     perfect-gradient elimination: re-derived from the core's own split AND from the Lipps-Hemler
    #     split, both give g*tau + R_d(dT_ref/dz)*eps.  ``theta`` would be right if the pressure term were
    #     -c_p theta_ref grad(pi') AND the base state were isentropic; it is neither, so switching the
    #     buoyancy to theta while rho_ref/p_ref stay non-isentropic leaves a FIRST-ORDER RESIDUE
    #         theta - linear = -(g*kappa + R_d dT_ref/dz)*eps = -0.9359*eps m/s^2,
    #     i.e. 0.502x the correction IN THE SAME DIRECTION -- theta OVERSHOOTS rather than bracketing an
    #     honest uncertainty.  And 0.502x is the floor: the measured amplitude ratio on run17 is 1.514
    #     domain-wide but 1.65-1.71 in every named band, so inside the HABITABLE BAND theta over-supplies
    #     by ~0.71x the correction.  So: ``linear`` is the PRIMARY form and the derived term; ``theta`` is
    #     a ONE-SIDED UPPER SENSITIVITY, kept because against an independent target it has the lowest RMS
    #     error of anything tried (though ``linear`` has the higher CORRELATION, 0.9992 vs 0.9975, and
    #     shape is what a torque cares about) and because the linearization cannot exclude it -- the
    #     genuinely unresolved quantity is the dropped O(tau*eps) nonlinearity, which NEITHER form
    #     represents.  Jake's 2026-09-19 ruling: both run, in one arm; the note's section 5 request for an
    #     exposed numeric alpha is replaced by the stricter house rule (sweep-not-tune admits no free
    #     numeric knob), so the two forms ship as named options and nothing in between is offered.
    #     MOIST KAPPA: ``theta`` uses the DRY kappa on T_rho -- Emanuel's density potential temperature
    #     theta_rho, a recognised quantity, so the leg is well defined.  kappa_m/kappa_d =
    #     (1 + 0.608 q)/(1 + 0.87 q) = 0.938 at q = 0.3, i.e. the moist correction is <= 6.2 % of the
    #     theta CORRECTION -- an order below the 1.5x form spread, and it does not touch ``linear``.
    #     THREE CAVEATS RIDE WITH IT AND MUST NOT BE DROPPED.  (i) The p' a bridge can supply is the
    #     HYDROSTATIC part; the non-hydrostatic part lives in the elliptic solve and is unavailable in an
    #     omega-psi core, in this variant or any other.  (ii) It makes the system internally CONSISTENT in
    #     its buoyancy; it does not make it VALID -- the anelastic expansion parameter |theta'|/theta_ref
    #     measures 0.274 in the mean here, and the dropped O(tau*eps) term is 1.22 m/s^2 against |b| = 3.10.
    #     (iii) Without touching the elliptic solve this leaves a DIFFERENTLY inconsistent system: b_eff
    #     contains p', the very thing the curl was supposed to have eliminated, so the result is
    #     "anelastic plus a diagnostic source term", a system with no energy principle.  The note still
    #     recommends it (an inconsistent approximation is worse than a consistent one, and the term moves
    #     the torque TOWARD the exact solenoidal source) and files (iii) as the argument for E1.
    #     DEFAULT OFF (env COSMO_BUOYANCY_PRESSURE_TERM=1): OFF is byte-identical -- the branch is at the
    #     PYTHON level, never a zero multiplier (the step is captured into a CUDA graph and a multiplier
    #     would change the OFF graph), and no array the OFF path touches is altered.  Array-generic, so
    #     CPU and GPU cannot diverge; b has ONE physics consumer (``vorticity_tendency``) and no GPU
    #     kernel recomputes it.  TURNING IT ON MOVES CANONICAL NUMBERS: it removes a band-to-band delta-b
    #     contrast of 4.27e-1 m/s^2 that is strongly negative in the hot half and ~0 poleward, i.e. it
    #     REDUCES the equator-to-pole buoyancy contrast aloft, which should WEAKEN the overturning and
    #     could move the band.  That is a FIRST-ORDER PREDICTION from the sign of the contrast, not a run.
    #     TWO TURN-ON ROUTES, the same split ``virtual_temp_exact`` and ``hydrostatic_virtual_t`` document
    #     above: (a) a PRODUCTION ARM via ``run_driver.py`` uses the ENVIRONMENT (export
    #     COSMO_BUOYANCY_PRESSURE_TERM=1, and COSMO_BUOYANCY_PRESSURE_FORM=theta for the other form) --
    #     run_driver builds a fresh StageADConfig before it captures, so the ON branch is the one
    #     recorded and the keys land in both ``world.cosmo_env`` and ``config``; (b) a
    #     ``build_core_with_world`` ANALYSIS REBUILD uses the post-build assignment
    #     ``core.cfg.buoyancy_pressure_term = True`` (that loader's world guard correctly refuses an
    #     unrecorded live COSMO_* key).  ``_update_buoyancy`` re-reads both keys every call, so (b) takes
    #     effect on the next EAGER step with no rebuild -- but SET IT BEFORE THE FIRST STEP on a
    #     capture_ready/GPU core, since a CUDA graph records the Python branch that ran at capture.
    #     ``ckpt_world._restore_physics_config`` restores both keys on rebuild (they are ordinary physics
    #     keys, covered by its inverted-set rule).
    buoyancy_pressure_term: bool = field(
        default_factory=lambda: _env_flag("COSMO_BUOYANCY_PRESSURE_TERM", False))
    # --- WHICH FORM the term above takes (env COSMO_BUOYANCY_PRESSURE_FORM; inert while
    #     ``buoyancy_pressure_term`` is OFF).  EXACTLY TWO values are accepted, both derived, neither a
    #     tunable number; anything else is refused at construction.
    #       "linear" (the DEFAULT and the PRIMARY form) -- db = alpha(z) * eps with the analytic
    #           alpha(z) = R_d * dT_ref/dz: the constant -R_d*lapse_ref = -1.8655 m/s^2 in the lapse
    #           branch and exactly 0 in the isothermal cap (dT_ref/dz = 0 there).  This is the
    #           derivation's own and UNIQUE answer -- the buoyancy that survives the curl under the
    #           pressure convention the code has ALREADY CHOSEN, reached identically from the
    #           Lipps-Hemler side -- and the coefficient that maximises the shape agreement with the
    #           independent solenoidal target.  (At the default reference state
    #           z_iso ~ 23.1 km sits ABOVE the height-grid lid at ~20.1 km, so the cap branch is never
    #           exercised on the canonical grid; it is implemented because the reference state is
    #           configurable and a zero there is the correct value, not a guard.)
    #       "theta" (the SENSITIVITY leg, not a second derivation -- see above) -- db =
    #           g * (T_rho_h/T_ref) * ((1 + eps)^(-kappa) - 1), which added to the shipped
    #           g*tau gives EXACTLY g*(theta_h - theta_ref)/theta_ref with
    #           theta_h = T_rho_h*(P_SURF/p_hydro)^kappa: the FULL Lipps-Hemler buoyancy, carrying the
    #           whole nonlinear factor and the T_rho_h/T_ref prefactor, not merely its first-order
    #           truncation alpha = -g*kappa.  Scored against the independent target the full theta leg
    #           had the LOWEST error of any candidate, which is why it ships at all rather than being
    #           ruled out by the 1.50x argument -- but it is an UPPER bound that knowingly over-supplies
    #           by ~0.5x domain-wide and ~0.71x in the habitable band, not a co-equal answer.
    #     One discretisation difference from the scoping note's L4 leg is declared in note section 9.1
    #     and measured in section 10.2: L4 forms theta on the PRESSURE grid (where p = p_mid exactly) and
    #     interpolates theta, whereas this forms theta at the height level from separately interpolated
    #     T_rho and p.  Same continuous expression, different discretisation.  The shipped choice
    #     composes exactly with "linear" (both read the identical eps), needs no second bracket build,
    #     and leaves the meaning of virtual_buoyancy/condensate_loading/virtual_temp_exact untouched.
    buoyancy_pressure_form: str = field(
        default_factory=lambda: _env_str("COSMO_BUOYANCY_PRESSURE_FORM", "linear"))   # linear|theta
    # --- closure (#4 Smagorinsky) + the M3 resolution/closure knobs --------------------------------
    C_s: float = 0.17                  # Smagorinsky constant: nu_t = (C_s * Delta)^2 |S|  (M3 perturbs)
    kappa_b: float = 1.0               # sub-grid buoyancy-diffusivity FLOOR [m^2/s] (M3 perturbs +-2x)
    nu_b: float = 1.0                  # sub-grid viscosity FLOOR [m^2/s]            (M3 perturbs +-2x)
    nu_h_floor: float = 0.0            # GRID-INDEPENDENT eddy-viscosity floor [m^2/s] added to nu_t (default 0 =
                                       # OFF, byte-identical). The Smagorinsky nu_h = (C_s*dx)^2|S| scales as dx^2
                                       # -> vanishes with the grid, so the equatorial convergence front collapses
                                       # to the grid scale (the 1-cell precip/ascent singularity). HYPOTHESIS TESTED
                                       # AND FALSIFIED (`_nu_h_sweep.py`): a fixed physical floor does NOT give a
                                       # frictional width ~nu_h/U -- at 1e8 m^2/s, nu/U ~ 10,000 km yet the spike
                                       # stays ONE cell. So the width is NOT friction-set (it is the ~10 km depth
                                       # scale, sub-grid). NB the floor is added to nu_t, which is HORIZONTAL-only
                                       # on the anisotropic path (closure_anisotropic, what the sweep used: nu_v is
                                       # separate) but ISOTROPIC on the default path (nu_t_z=None feeds nu_t to both
                                       # arms of _div_grad) -- the broadening null holds either way.
    # --- ANISOTROPIC closure (issue #81): split the Smagorinsky eddy viscosity into nu_h ~ (C_s dx)^2|S|
    #     and nu_v ~ (C_s dz)^2|S| instead of the isotropic Delta^2 = dx*dz. Default OFF (env
    #     COSMO_ANISO_CLOSURE=1 to enable) -- UNLIKE the other all-on options this is a STRUCTURAL closure
    #     choice that must be A/B'd against the fleet baseline before any default flip (it changes the
    #     circulation strength, hence transported heat, hence T(theta) -- the M3.5 question). On the
    #     ~1000:1 fleet grid it both de-inflates the vertical mixing ~10^3 (physics) AND lifts the
    #     diffusive-CFL dt ceiling ~10^2 (performance). OFF is byte-identical to the isotropic core.
    closure_anisotropic: bool = field(default_factory=lambda: _env_flag("COSMO_ANISO_CLOSURE", False))
    # --- surface drag (issue #85 item 2): a linear (Rayleigh) vorticity damping -surface_drag_rate*omega
    #     applied at the lowest interior level (j=1), the cheap momentum sink the free-slip bottom lacks.
    #     Default 0.0 = OFF (byte-identical). Pairs with closure_anisotropic (#81): once the inflated
    #     vertical viscosity drops, this is what keeps u_surf physical as the momentum sink. A representative
    #     value is ~1/(2 days) ~ 5.8e-6 1/s; the bulk-aerodynamic derivation from the model's own givens
    #     (C_D ~ Ch = 1.2e-3, U = cfg.wind = 6 m/s, the nz=18 lowest-level dz ~ 1176 m) gives
    #     r = C_D*U/dz ~ 6.1e-6 1/s ~ 1/(1.9 days) -- a given, not a tuned knob. The conjugate
    #     dissipative-heating return of the drag's removed KE is IMPLEMENTED (folded into the operator
    #     KE-removal inversion), so drag ON is energy-closed on the default dissipative_heating="operator"
    #     path; only the non-default strain form / heating-off limits would leak it (diagnostic-only).
    surface_drag_rate: float = field(default_factory=lambda: _env_float("COSMO_SURFACE_DRAG_RATE", 0.0))
    # --- HONEST bulk surface friction (default OFF, byte-identical). Unlike surface_drag_rate (a Rayleigh
    #     sink on omega[1], which barely reduces the buoyancy-driven surface jet -- damping vorticity at one
    #     level is not a momentum sink), this applies the curl of the bulk-aerodynamic surface STRESS
    #     tau = rho*C_D*|u1|*u1 as a genuine lowest-cell momentum drag: du/dt = -(C_D|u1|/dz1) u1, fed to the
    #     vorticity tendency as d/dz of that force. Quadratic in the LOCAL wind, with the STANDARD C_D
    #     (= the heat/moisture transfer coeff, surface.py); it is the physical surface friction, NOT a knob
    #     tuned to reduce wind. ENERGY-CLOSED like surface_drag_rate: the KE it removes is folded into
    #     _operator_ke_removal_col's inversion and returned to T_atm as frictional heat (no budget leak).
    #     C_D from ``surface_drag_cd``. env COSMO_SURFACE_DRAG_BULK / COSMO_SURFACE_DRAG_CD.
    #     DEFAULT ON (2026-06-16, Jake): the honest surface friction is part of the all-on accurate model
    #     (like clouds / lw_window / rh_detrain above); set COSMO_SURFACE_DRAG_BULK=0 for the frictionless
    #     byte-identical limit. (The legacy linear ``surface_drag_rate`` Rayleigh sink stays default 0.)
    surface_drag_bulk: bool = field(default_factory=lambda: _env_flag("COSMO_SURFACE_DRAG_BULK", True))
    surface_drag_cd: float = field(default_factory=lambda: _env_float("COSMO_SURFACE_DRAG_CD", 1.2e-3))
    kappa_q: float = 50.0              # sub-grid moisture eddy-diffusivity [m^2/s] for the resolved-flow
                                       # q advection (M2b.3); small, numerical-regularization scale
    kappa_T: float = 50.0              # sub-grid HEAT eddy-diffusivity [m^2/s] for the resolved-flow
                                       # potential-temperature advection (M2b.6); conservative flux-form
                                       # cross-track, small numerical-regularization scale (sibling of kappa_q)
    heat_advection: str = "fluxlim"    # theta_pot advection scheme: "fluxlim" (DEFAULT, van Leer, the
                                       # all-on accurate model) | "upwind" (1st-order, byte-identical to
                                       # M2b.6). BOTH are conservative flux form (int theta_pot dm and
                                       # mass_conservation untouched). fluxlim is ~33x LESS numerically
                                       # diffusive (1-D bump: upwind loses 23.4% of peak, fluxlim 0.7%), so
                                       # it preserves the cross-track T slope that sets the ribbon band
                                       # placement; it is equally TVD-stable at the same CFL dt. (M3.4b Task 1
                                       # also used the swap as the energy-closure discriminator: |enth| did
                                       # NOT collapse under fluxlim -> the residual sink is REAL, not upwind
                                       # numerical diffusion.)
    dissipative_heating: bool = True   # M3.4b Task 2: return the eddy-viscosity-removed KE to T_atm as heat,
                                       # CLOSING the real APE->KE->dissipation leak Task 1 proved
                                       # (rel_imbalance plateaus at ~|enth|). DEFAULT ON (the energy-closed
                                       # accurate model) -> returns the missing frictional-heating term so
                                       # int(SW_abs-OLR) -> 0 at steady state; set False for the Task-1 core
                                       # byte-identical (leaky) limit. The deposit MAGNITUDE follows
                                       # ``dissipation_form`` below.
    dissipation_form: str = "operator"  # M3.4b Task 2 FINDING: the heat to return is the KE the VORTICITY-form
                                       # operator div(nu_t grad omega) actually removes ("operator", the
                                       # budget-closing default) -- NOT the strain-form Smagorinsky production
                                       # int(rho_ref*nu_t*|S|^2) ("strain", the spec's literal term), which
                                       # OVER-estimates it ~2.6x (resolution-robust) because div(nu_t grad
                                       # omega) is not the curl of the variable-viscosity stress. "strain" is
                                       # kept as the diagnostic that revealed the gap.
    # --- surface / ocean (used from M2b.4) ---------------------------------------------------------
    ocean_depth_spinup: float = 5.0    # mixed-layer depth for spin-up [m] (equilibrium independent of it)
    # --- SPIN-UP sea depth UNDER THE DIURNAL SUN (ROADMAP PHY-17, the frozen-physics canonical run;
    #     env COSMO_SPINUP_OCEAN_DEPTH_M, in METRES, default 0.0 = OFF and BYTE-IDENTICAL). A spin-up
    #     DEVICE, not physics: with ``diurnal_sun`` the wet-column heat capacity is the physical 50 m
    #     mixed layer ``constants.C_OCEAN_MIXED``, and its relaxation time in the habitable band is
    #     MONTHS (58-105 true days band-mean, 105-204 at the cold end, flatworld/PHY17_LONG_RUN_SIZING.md),
    #     so a plain run to equilibrium costs weeks of GPU. ON (> 0), the wet capacity is instead a slab
    #     of this depth, ``slab_heat_capacity(depth, RHO_LIQ, CP_LIQ)`` -- blended by the SAME wetness,
    #     with the dry ``C_BARE_ROCK`` term UNCHANGED -- so the sea relaxes ~50/depth times faster.
    #     WHEN IT IS ONLY A RATE: the slab carries no ocean heat transport and no deep water (T_surf is
    #     the whole ocean state), so its STEADY surface energy balance does not contain C; C sets how fast
    #     T_surf gets there and how big the diurnal swing is. The swing is the route by which C can move
    #     the mean, and it does so wherever a THRESHOLD sits inside the (10x larger) thin-slab swing:
    #     (i) the 2 K sea-ice albedo ramp at the cold edge (rectifies the day-mean albedo; ~5 W m^-2 in
    #     one column at worst, PR #849 review) and (ii) BOILING, which is not a rectification but an
    #     IRREVERSIBLE water-store change: ``surf_cap`` (the flash-boil cap) is proportional to C_surf and
    #     the landed liquid is P - flash, so at a wet pool near T_boil a thin slab changes W_surf itself.
    #     Under the converged segment mixer the axis (theta < 0.2 deg) becomes exactly such a pool (39-45
    #     wet columns at 81-94 C) -- hence the static mask below, which keeps the physical 50 m there.
    #     The protocol (flatworld/PHY17_FROZEN_RUN_PROTOCOL.md) also ends every spin-up with a stretch at
    #     the physical 50 m and accepts the run on THAT stretch's drift.
    #     WHAT IT SCALES: every wet column POLEWARD OF ``spinup_ocean_min_theta_deg``, INCLUDING frozen wet
    #     columns (the model has no separate sea-ice heat capacity or latent heat of fusion -- ice is
    #     albedo only, ``surface_albedo``). Nothing else: not the dry rock term, not the masked wet columns
    #     (bit-identical to OFF), not ``ocean_depth_spinup`` (the diurnal-OFF slab), not the atmosphere.
    #     RESUME-SWITCHABLE BY DESIGN: this is a PHYSICS-bucket field, NOT in
    #     ``_ckpt_fingerprint.WORLD_DEFINING_CONFIG`` (it changes no grid), so a stage-1 checkpoint made
    #     ON resumes with it OFF (``run_driver --resume`` builds from the SHELL env, then loads the
    #     state). Every checkpoint's fingerprint records the value (``config``) and the env key
    #     (``world.cosmo_env``), and ``run_manifest.json`` records it per launch. The core reads it at
    #     CALL time in ``_surf_heat_capacity`` (so ``ckpt_world._restore_physics_config`` reaches an
    #     analysis rebuild), which on the GPU means it is baked into the CUDA graph at capture: set it
    #     BEFORE launch, never on a running core.
    #     REFUSED: a nonzero value without ``diurnal_sun`` (the diurnal-OFF slab is ``ocean_depth_spinup``;
    #     silently ignoring the knob there would be worse), and any value outside (0, 50] m (deeper than
    #     the physical layer is not a spin-up accelerator). 50.0 exactly reproduces C_OCEAN_MIXED bit for
    #     bit (same product), which the tests pin.
    spinup_ocean_depth_m: float = field(default_factory=lambda: _env_float("COSMO_SPINUP_OCEAN_DEPTH_M", 0.0))
    # --- ...and the STATIC cross-track mask it applies behind (env COSMO_SPINUP_OCEAN_MIN_THETA_DEG, in
    #     DEGREES of theta, default 25.0; read only while spinup_ocean_depth_m > 0). The thin slab applies
    #     only where theta >= this cutoff; every wet column equatorward keeps the physical C_OCEAN_MIXED, bit
    #     for bit. WHY (PR #849 review finding 1): equatorward of it the thin slab brings BOILING into the
    #     swing (the mixer's axis pool, 4.6-18 K below T_boil; the hot-sea shore at 19.0-19.2 deg, whose
    #     projected 5 m daily max is 15.5 K below it) and buys nothing, because those columns already relax
    #     in days at 50 m (tau_local 1.3-3.5 true days; the coupled SST response to the mixer decays with
    #     e-folding 6-15 true days at 19.5-25 deg -- D4 5.6-6.2, D2 14.3-14.7). THE VALUE: 25.0 deg, fixed
    #     2026-09-28 before any run (protocol section 3.1, measured by ``phy17_spinup_mask_cutoff.py``):
    #     every wet column poleward of it keeps a >= 27 K margin to boiling at its projected 5 m daily max
    #     in every existing arm, 5.8 deg (~6,700 km) poleward of the last near-boiling column, and every
    #     column it withholds reaches 90 % of its response at 50 m within the run's minimum length; the
    #     admissible range is 21-30 deg. STATIC by design -- never keyed on a prognostic (a water-depth
    #     mask would flicker at a pool edge, switching C by 10x). Capture-safe: the per-column capacity
    #     is built on an eager step and cached per (array module, dtype); a value outside [0, 90) is refused.
    spinup_ocean_min_theta_deg: float = field(
        default_factory=lambda: _env_float("COSMO_SPINUP_OCEAN_MIN_THETA_DEG", 25.0))
    wind: float = 6.0                 # near-surface wind for bulk surface fluxes [m/s] (the FIXED bulk-flux
                                       # wind; the resolved-wind closure below replaces it when enabled)
    # --- resolved-wind surface fluxes (issue #85 item 1): use the RESOLVED surface cross-track wind
    #     |u(z~surface)| (with a convective-gustiness floor) in the bulk SH/LE fluxes instead of the fixed
    #     cfg.wind, so evaporation follows where the cell's surface branch actually blows (the WISHE-like
    #     wind-evaporation feedback). Default OFF (env COSMO_RESOLVED_FLUX_WIND=1) -- A/B before any default
    #     flip. ``gustiness`` is the calm-column convective floor wind so quiescent columns still evaporate;
    #     the exchange wind is sqrt(u_surf^2 + gustiness^2). OFF -> the scalar cfg.wind, byte-identical.
    resolved_flux_wind: bool = field(default_factory=lambda: _env_flag("COSMO_RESOLVED_FLUX_WIND", False))
    gustiness: float = 3.0             # convective-gustiness floor [m/s] for the resolved-wind surface fluxes
    conv_max_sweeps: int = 8           # max convective-adjustment sweeps per step (M2b.3)
    # --- Betts-Miller-style FINITE convective adjustment timescale (env COSMO_CONV_TAU, default 0 =
    #     instantaneous = byte-identical). The shipped scheme adjusts the column to the moist adiabat +
    #     rh_detrain q target INSTANTANEOUSLY every thermo sub-step; sub-cycling that nonlinear, target-
    #     seeking operator against the dt-scaled forcing makes the climate TIMESTEP-DEPENDENT (it pumps
    #     more moisture aloft the more often it's called -- CHIMNEY_HIRES_RESULTS.md). The physical fix:
    #     give convection a finite timescale tau and apply a FRACTIONAL relaxation toward the adjusted
    #     target, frac = min(dt/tau, 1) -- a proper O(dt) tendency that is timestep-CONVERGENT and reduces
    #     to the instantaneous scheme as tau->0. The linear blend of the (enthalpy/water-conserving) pre-
    #     and post-adjustment states conserves both exactly. Physical tau ~ 1-3 h (~7200 s).
    conv_tau: float = field(default_factory=lambda: _env_float("COSMO_CONV_TAU", 0.0))   # ⚠️ TEST default 0.0 (instantaneous); CANONICAL = 7200 (finite Betts-Miller, the "bm_tau2h" edition; canonical_env.sh).
    # --- FINITE condensation saturation-adjustment timescale (issue #280; env COSMO_COND_TAU, default 0 =
    #     instantaneous = byte-identical) -- the conv_tau treatment applied to the OTHER instantaneous
    #     operator the CHIMNEY_HIRES root-cause left behind: the saturation adjustment condenses the FULL
    #     supersaturation (single-pass path) / relaxes q_c the FULL way to the PDF equilibrium
    #     (_prognostic_microphysics) every thermo sub-step, so sub-cycling it against the dt-scaled forcing
    #     leaves the inner-zone climate n_sub-DEPENDENT (~52 C at n_sub 30<->60 even at conv_tau=7200; the
    #     ribbon is conv_tau-converged to ~1.1 C). The fix: scale the saturation-adjustment EXCHANGE
    #     (condensation source AND its re-evaporation conjugate -- one signed adjustment) by
    #     frac = min(dt_sub/tau, 1) [dimensionless], turning the snap into a finite-rate relaxation
    #     dq_c/dt -> (qc_eq - q_c)/(denom*tau) whose per-dynamics-step total is n_sub-convergent
    #     (= dt_dyn/tau to leading order). tau->0 => frac=1 recovers the shipped instantaneous scheme.
    #     Water/energy stay exactly consistent: cond and evap carry the SAME frac, so the q<->q_c exchange
    #     and its latent heat remain conjugate; the positivity limiter runs on the scaled demands unchanged.
    #     NOT in the fused numba/device microphysics kernels (the #237 salt-CCN pattern): cond_tau>0 routes
    #     to the array-generic _prognostic_microphysics (still GPU-capable, just un-fused); threading it
    #     into the fused kernels is a logged follow-up if a cond_tau>0 canonical flip is ever adjudicated.
    cond_tau: float = field(default_factory=lambda: _env_float("COSMO_COND_TAU", 0.0))   # default 0.0 (instantaneous, byte-identical) = the canonical world; a tau>0 flip needs the A/B + sweep (ROADMAP GPU-7).
    # --- salt/CCN-coupled cond_tau (ROADMAP GPU-7; env COSMO_COND_TAU_SALT, default OFF = the constant-tau
    #     scheme above, byte-identical). The phase-relaxation time the finite tau stands in for is CCN-set
    #     (thermo.phase_relaxation_tau: tau_phase = 1/(4 pi D N r_bar) -- ~3 s at the clean-maritime
    #     100 cm^-3, sub-second in salt-loaded continental air), and this world's CCN is dominated by
    #     sea-salt aerosol off the evaporating inner sea (#237, the N_d field). So the defensible tau is
    #     SPATIALLY VARYING: short over the high-salt inner sea, long in clean poleward air. ON, the
    #     relaxation timescale becomes the per-column
    #         tau(x) = cond_tau * (N_ref / N_d(x))^gamma        [s]
    #     with ``cond_tau`` reinterpreted as the CLEAN-AIR (N_d == N_ref) reference tau and gamma =
    #     ``cond_tau_salt_exp``. gamma's default 2/3 is the fixed-liquid-water scaling of the tau_phase law
    #     (r_bar ~ (LWC/N)^(1/3) at fixed LWC => N*r_bar ~ N^(2/3)), CONSISTENT with the Twomey limb's
    #     r_eff = CLOUD_REFF_LIQ*(N_ref/N_d)^(1/3) closure; gamma = 1 is the fixed-r_bar alternative (both
    #     swept, never tuned -- the #237 pattern). Only the ratio N_ref/N_d enters (units-free); at
    #     N_d == N_ref the factor is EXACTLY 1.0, so a clean field is bit-identical to constant tau (the
    #     #237 zero-point). CONSERVATION: safe under a spatially-varying frac -- the cond_tau exchange is a
    #     per-CELL q<->q_c conjugate pair (one frac scales both signs at the same cell), unlike conv_tau's
    #     column-integral blend, which needs ONE scalar frac to telescope (see the _bm note in the substep).
    #     Requires salt_ccn (the N_d field) + cond_tau > 0 (the reference). N_d is a COLUMN (nx,) field, so
    #     tau(x) varies cross-track only -- "clean ALOFT air" needs a z-resolved aerosol field the model
    #     does not carry (out of scope, noted in COND_TAU_RESULTS.md). cond_tau > 0 already routes the
    #     microphysics out of the fused kernels, so the salt coupling inherits that route-to-generic.
    cond_tau_salt: bool = field(default_factory=lambda: _env_flag("COSMO_COND_TAU_SALT", False))
    cond_tau_salt_exp: float = field(default_factory=lambda: _env_float("COSMO_COND_TAU_SALT_EXP", 2.0 / 3.0))
    #   ^ gamma in tau(x) = cond_tau*(N_ref/N_d)^gamma: 2/3 = fixed-LWC diffusional scaling (default,
    #     Twomey-consistent); 1 = fixed-r_bar. A physically-derived best estimate, REPORTED as a
    #     sensitivity, never fitted (roadmap rule 4).
    # --- sub-grid PDF-tail condensation target for the SINGLE-PASS core (ROADMAP GPU-7; env
    #     COSMO_COND_PDF_TAIL, default OFF = the shipped grid-mean adjustment, byte-identical). The
    #     single-pass path (prognostic_condensate OFF) condenses the grid-MEAN excess max(q - qsat, 0)
    #     while its radiation cover is the ZEROTH moment of the sub-grid total-water PDF
    #     (rad.cloud_pdf_partition) -- inconsistent closures (cover from the PDF, condensate from the
    #     mean). ON, the condensation target becomes the FIRST moment of the supersaturated tail of the
    #     SAME PDF (qc_eq = E[(q_t - qsat)^+]): cover and condensate are ONE closure -- the box drizzles
    #     its tail excess when partly cloudy (f > 0 <=> cond > 0) and reduces BIT-IDENTICALLY to the
    #     grid-mean adjustment when overcast (Q1 >= 1 => qc_eq = q_t - qsat). ``cond_tau`` still sets the
    #     rate (the PDF tail sets the TARGET, tau the RATE -- sub-grid-consistent AND dt-convergent
    #     together). NB the PROGNOSTIC path needs no such flag: with qc_pdf_cloud=True (canonical) its
    #     source already relaxes q_c toward the same tail first moment -- this flag closes the one path
    #     where the roadmap's cover-vs-condensate inconsistency actually remains. Requires
    #     qc_pdf_cloud (the cover footing it unifies with); invalid with prognostic_condensate ON.
    cond_pdf_tail: bool = field(default_factory=lambda: _env_flag("COSMO_COND_PDF_TAIL", False))
    # --- frozen-Gamma convective sweeps (issue #99): hold each pair's moist lapse rate at its (live,
    #     Gauss-Seidel) sweep-1 value through the later sweeps of ONE adjustment call, instead of
    #     re-deriving it on temperatures the sweep itself only nudged -- the #86 dt_rad hold-a-slow-field
    #     pattern, applied to the leaf calls that the #99 profile measured as the fleet step's wall
    #     (~half of coupled step cost; the cap is hit with every pair active, so the cost is
    #     conv_max_sweeps * (nz-1) leaf calls per step). Enthalpy/water conservation are
    #     Gamma-independent (exact either way); the flag changes only which adiabat sweeps >= 2 relax
    #     toward, by O(dGamma * sweep dT). Default OFF = byte-identical; flip needs the fleet A/B
    #     (the #81/#85/#86 precedent). Measured at fleet resolution: coupled step 8.5 -> 7.3 ms (~1.16x;
    #     ~1.24x against the pre-#99 baseline), and the ON trajectory converges to the OFF state
    #     (max|dT_atm| 2.4 K in the early coupled transient -> 0.07 K after 150 coupled steps).
    conv_freeze_gamma: bool = field(default_factory=lambda: _env_flag("COSMO_CONV_FREEZE_GAMMA", False))
    # --- ONE-PASS SEGMENT MIXER: the CONVERGED convective adjustment (ROADMAP PHY-17 step 4; env
    #     COSMO_CONV_SEGMENT_MIXER, default OFF = the capped Gauss-Seidel sweep, BYTE-IDENTICAL). The
    #     shipped adjustment relaxes a column toward the moist adiabat PAIRWISE, so information travels
    #     one layer pair per sweep and the sweeps needed grow with the SQUARE of the unstable layer's
    #     depth in levels. On the canonical 144-level boundary-layer-refined grid the production cap
    #     conv_max_sweeps = 4 is nowhere near the fixed point: one call on the run17 checkpoint's map-band
    #     columns leaves 35.8 / 29.7 / 18.1 / 3.6 / 0.0 K of a 43.1 K potential-temperature deficit after
    #     4 / 16 / 64 / 256 / 1,024 sweeps, and over the salt flats even 8,192 sweeps have not converged
    #     (flatworld/OPEN_ISSUES.md, the PHY-17 entry). ON, the adjustment instead finds each contiguous
    #     statically unstable SEGMENT -- by the sweep's OWN pairwise predicate -- and mixes the whole
    #     segment to the moist adiabat in one operation, with a merge-and-rescan loop that stops on the
    #     sweep's OWN rule (no pair unstable by more than 1e-9 K). Same operator, same fixed point,
    #     bounded cost: stage_ad_column.moist_convective_segment_mix, and PHY17_SEGMENT_MIXER_RESULTS.md
    #     for the equivalence numbers and the cost. ``conv_max_sweeps`` is NOT read on this path.
    #     FALLS BACK to the sweep whenever the mixer cannot be the same operator: conv_freeze_gamma ON
    #     (the mixer always evaluates Gamma live, at the converged temperatures) or q is None (the fused
    #     device kernel always writes q, mirroring convadjust_parallel's guard).
    conv_segment_mixer: bool = field(default_factory=lambda: _env_flag("COSMO_CONV_SEGMENT_MIXER", False))
    conv_segment_max_passes: int = field(default_factory=lambda: _env_int("COSMO_CONV_SEGMENT_MAX_PASSES", 48))
    #   ^ the merge-and-rescan bound. A mixed block grows at most one layer per side per pass once the
    #     first pass has absorbed every initially-unstable pair, so the requirement is set by how far a
    #     block eats into the marginally-stable air above it. MEASURED on run17's own 975 columns: every
    #     column converges by 40 passes and all but 5 by 32, the stragglers being the deep salt-flat
    #     columns near latitude 2 deg. 48 is that measurement plus headroom, NOT a tuned number; raise it
    #     (and re-measure) for a state with deeper unstable layers. Exhausting the bound WITH THE ACTIVE
    #     SET STILL GROWING is a truncated adjustment, and it is reported, not silent: the numpy path
    #     raises a RuntimeWarning and both paths record it per column in ``passes_out`` (negated).
    conv_segment_gamma_iters: int = field(default_factory=lambda: _env_int("COSMO_CONV_SEGMENT_GAMMA_ITERS", 3))
    #   ^ iterations of the moist-adiabat fixed point inside ONE block mix (the profile shape depends on
    #     Gamma, which depends on the temperature the mix produces). MEASURED contraction on the canonical
    #     probe columns: 2e-3 to 5e-2 per iterate -- the |dGamma/dT| x depth estimate of ~1/25 is an upper
    #     bound the iteration beats, because iterate 0 already lands close. Enthalpy and water are
    #     conserved EXACTLY at every iterate, converged or not, so this knob buys profile accuracy only,
    #     never conservation.
    #     CHOSEN BY MEASUREMENT, on TWO criteria, over all 975 canonical columns (the table is in
    #     PHY17_SEGMENT_MIXER_RESULTS.md section 4; this is a numerics default picked by sweeping, NOT a
    #     value tuned to hit a target):
    #       (a) it must reach the SAME converged partition as a tightly-converged reference
    #           (gamma_iters=40, max_passes=400), and
    #       (b) every column must end below the 1e-9 K instability tolerance, none truncated.
    #     2 is the CHEAPEST value meeting both; 3 is the value shipped, and the gap between them is the
    #     margin, priced:
    #                       partition   max|dT| vs the tight reference   cost vs the NUMPY 4-sweep cap (min/median)
    #         gamma_iters=1  DIFFERENT   2.0e-2 K                        --          <- criterion (a) FAILS
    #         gamma_iters=2  same        2.1e-7 K                        1.16x/1.17x
    #         gamma_iters=3  same        4.9e-8 K                        1.36x/1.39x <- shipped
    #         gamma_iters=6  same        7.6e-9 K                        2.05x/2.09x
    #     (round-robin interleaved on a quiet box; the residual instability is below 1e-9 K at every one
    #     of them, since that is the loop's own exit condition.) NOTE THE CURRENCY: those ratios are
    #     numpy-versus-NUMPY, and the numpy sweep is dispatch-bound -- they are the right comparison for
    #     THIS knob (it scales the same work in both paths) and the wrong one for the mixer's production
    #     cost, which is ~45x the FUSED sweep on the GPU (PHY17_SEGMENT_MIXER_RESULTS.md section 3).
    #     2 sits ONE STEP above a measured failure -- gamma_iters=1 partitions the column differently --
    #     and 3 buys ~4x the distance from that floor. PRICED IN THE DEVICE'S OWN CURRENCY
    #     (phy17_mixer_cost_model.py, warp-issued transcendentals on run17's 975 columns): the third
    #     iterate costs 1.32x the mixer at gamma_iters=2 (48.7x -> 64.1x the fused sweep), and 1.19x with
    #     the shipped kernel loop nest (10.2x -> 12.2x). That is a fifth of the mixer, not the "under 1%
    #     of the step" an earlier reading of the numpy ratio suggested -- so it IS a real price, and it is
    #     paid deliberately: the arm this ships into EVOLVES the state away from run17, the stress case is
    #     the deep salt-flat columns, and a wrong PARTITION is not a small error.
    #     Re-measure on a new state (phy17_segment_mixer_equivalence.py, report 3b) rather than inherit.
    #     A PASS-COUNT asymptote is NOT a criterion here: the numpy path's exit is whole-array, so the
    #     passes actually paid are set by the LAST column to converge (39-40 at every setting from 2 to
    #     12), and this knob multiplies the work inside each of them.
    # --- MINIMUM HORIZONTAL SCALE for the convective tendency (ROADMAP PHY-19 remedy (b); env
    #     COSMO_CONV_MIN_SCALE_M, in METRES, default 0.0 = OFF and BYTE-IDENTICAL). ON, the change the
    #     convective adjustment makes to a column -- its dT and dq across the adjustment call, whichever
    #     branch ran (capped sweep, fused device kernel, or segment mixer) -- is passed through a
    #     conservative, mass-weighted HORIZONTAL filter of width ``conv_min_scale_m`` before the
    #     Betts-Miller blend. It does not change HOW MUCH convection happens; it changes HOW NARROW the
    #     heating and drying are allowed to be.
    #     WHY: on the production stretched-x grid the sun-line axis cells are 2 km wide and the adjustment
    #     is column-by-column, so the scheme can deliver a whole convecting column's heating and drying
    #     into one 2 km cell -- and does: a 12-16x peak/strip rain ratio at a ONE-CELL half-max width,
    #     against the converged cap-64 arm's 1.5-3 at 26-33 cells (flatworld/PHY19_AXIS_CONVECTION.md
    #     section 2). A 2-D slab cannot represent the cold-pool outflow or the along-line breakup of a
    #     line updraft, and both put a FLOOR on the delivery scale.
    #     THE VALUE: W = 30,000 m, range 10,000-60,000, FIXED on 2026-09-17 HST before any run read it
    #     (section 4.1 -- two independent arguments, cold-pool spreading and updraft-depth breakup, both
    #     land at tens of kilometres). It is a PARAMETER OF THE TERM, not a tuning knob: sweep it, do not
    #     fit it. The code default is 0.0 (OFF) so nothing inherits the value by accident.
    #     PROPERTIES (flatworld/conv_min_scale.py, which has the full derivation): column enthalpy
    #     sum c_p dx dm dT and water sum dx dm dq are conserved to ROUND-OFF per level (the operator's
    #     columns are normalized LAST, and the build asserts it); PER-COLUMN conservation is deliberately
    #     broken (moving adjustment work between columns is the point); the operator is NOT
    #     uniform-invariant (row sums are not 1 on a stretched grid -- the bound is measured on the built
    #     operator and logged); the wall at x = 0 REFLECTS (it is a psi = 0 symmetry wall); and every
    #     column whose cell is at least W wide is EXACTLY the identity, so at W = 30 km the term touches
    #     79 of 975 production columns (~620 km, inside theta < 0.7 deg) and CANNOT REACH the habitable
    #     band at theta 43.5-50.5, where cells are hundreds of km wide.
    #     WHAT FOLLOWS IT: the condensation step reads the FILTERED state, so supersaturation the filter
    #     creates or removes is handled by the microphysics exactly as any other supersaturation. That is
    #     deliberate -- there is no second fix here.
    #     DEFAULT OFF (0.0) is a PYTHON-level branch, never a zero-width or zero-weight operator: the step
    #     is captured into a CUDA graph and a multiplier would change the OFF graph.
    #     REFUSED IN COMBINATION with the gray-zone gate (conv_dx_off/conv_dx_full): under that gate the
    #     blend's ``frac`` is an (nx, 1) per-column array and the telescoping argument for filtering a
    #     tendency before a per-column blend has NOT been checked (section 5(b) step 3).
    #     TURN-ON ROUTES, as for conv_segment_mixer / condensate_loading / virtual_temp_exact (corrected
    #     by the PR #839 review, finding 2 -- the old note said "not the environment", which is true only
    #     of the analysis loader): for a PRODUCTION arm use ``COSMO_CONV_MIN_SCALE_M=30000`` in the
    #     launching shell (``run_driver`` builds the config at :1024, captures at :1134, and the key is
    #     recorded in ``world.cosmo_env`` as well as ``config``). Only a ``build_core_with_world``
    #     ANALYSIS rebuild needs the post-build ``core.cfg.conv_min_scale_m = 30_000.0``, because that
    #     loader's ``assert_env_matches`` (correctly) refuses a COSMO_* key the checkpoint never
    #     recorded. The core builds its operator in __init__ AND re-checks the width at the call site,
    #     so a post-build assignment takes effect (it rebuilds once, on the eager warmup step, never
    #     inside a captured graph).
    conv_min_scale_m: float = field(default_factory=lambda: _env_float("COSMO_CONV_MIN_SCALE_M", 0.0))
    # --- column-PARALLEL convective adjustment (CPU-fuse, issue #99 / branch claude/cpu-fuse-multithread).
    #     The numpy moist_convective_adjust is vectorized across columns but its k-loop is Gauss-Seidel
    #     (sequential in nz); the nx COLUMNS are independent, so a numba @njit(parallel=True) prange OVER
    #     COLUMNS threads it AND fuses the per-pair qsat/lapse-rate leaves. Default OFF = the numpy path,
    #     BYTE-IDENTICAL (the function above is untouched). ON = the numba kernel
    #     (moist_convective_adjust_parallel), matching numpy to round-off PER STEP (~1e-13 K; it reassociates
    #     the leaf transcendentals -> a different-but-valid FP path, NOT bit-identical). The production
    #     guarantee is CLIMATOLOGICAL (the time-mean P, T(theta), band), not trajectory tracking: in the f=0
    #     core a per-step round-off perturbation is deterministic-chaos-amplified over a long run, so two
    #     trajectories decorrelate -- immaterial to the science (the +-10 C budget swamps it), but the OFF and
    #     ON runs are NOT expected to stay close step-for-step indefinitely. (The discontinuous predicates --
    #     ``if f >= cloud_f_min``, the convective ``unstable`` test -- mean a cell within ~1e-13 of a threshold
    #     can flip branches, so "round-off" is a generic per-run bound, not a hard per-cell one; this is
    #     inherited from the numpy scheme, not introduced by the kernel.) Wired at the _thermo_substep call
    #     site; FALLS BACK to numpy whenever the kernel cannot reproduce the
    #     numpy path to round-off: conv_freeze_gamma ON, q is None, or precision != "fp64". Measured
    #     speedup (full cores) in archive/_cpufuse_prod_RESULTS.md. env COSMO_CONVADJUST_PARALLEL.
    convadjust_parallel: bool = field(default_factory=lambda: _env_flag("COSMO_CONVADJUST_PARALLEL", False))
    # CPU-fuse #2 (radiation): column-PARALLEL (numba prange) LW two-stream recurrence. Default OFF = the
    #     numpy path, BYTE-IDENTICAL (radiation.two_stream_lw / _windowed untouched). ON = the numba kernel
    #     (radiation.two_stream_lw_parallel), matching numpy to round-off (~1e-13; the leaf exp/T^4
    #     reassociate -> a different-but-valid FP path, NOT bit-identical). Wired at the _radiate refresh
    #     call (column_radiation(..., lw_parallel=True)); FALLS BACK to numpy whenever the kernel cannot
    #     reproduce the numpy path: non-2-D inputs or non-fp64 (radiation._lw_parallel_ok). Covers the
    #     production clouds/window/SW-band config for free (those only feed the dtau/band_frac into the
    #     two-stream). Measured speedup in archive/_cpufuse_radiation_RESULTS.md. env COSMO_RADIATION_PARALLEL.
    radiation_parallel: bool = field(default_factory=lambda: _env_flag("COSMO_RADIATION_PARALLEL", False))
    # CPU-fuse #3 (advect+buoy): column-PARALLEL (numba prange) batched interp for the resolved-flow grid
    #     reconciliation (_height_to_pressure / _pressure_to_height), the hot part (~66%) of the advect+buoy
    #     limb. Default OFF = the numpy `_interp_columns` (untouched, BYTE-IDENTICAL). ON = the prange-over-
    #     columns numba kernels (`_interp_columns_parallel`), BIT-IDENTICAL in fp64 (pure +,-,*,/ + integer
    #     bracket -> no FP reassociation). FALLS BACK to numpy for any non-production shape / non-fp64. env
    #     COSMO_ADVECT_PARALLEL.
    advect_parallel: bool = field(default_factory=lambda: _env_flag("COSMO_ADVECT_PARALLEL", False))
    # CPU-fuse #4 (thermo_other): column-PARALLEL (numba prange) MICROPHYSICS + below-cloud REEVAP -- the new
    #     dominant un-accelerated limb once convadjust/radiation/advect were threaded (~90% of thermo_other:
    #     _prognostic_microphysics + _below_cloud_reevap, FUSED into one prange-over-columns kernel). Default OFF
    #     = the numpy methods (untouched, BYTE-IDENTICAL). ON reproduces the numpy reference to round-off
    #     (~1e-13, esat's exp reassociates). Covers ONLY the CANONICAL prognostic-cloud config (qc_pdf_cloud,
    #     qc_dqsat_adjust, qc_ice_phase, qc_retained_ceiling, qc_radiative_cap_decouple, qc_reevaporation,
    #     kessler autoconv, precip_reevap, equator_norain_cells==0, fp64); FALLS BACK to numpy otherwise. env
    #     COSMO_THERMO_OTHER_PARALLEL.
    thermo_other_parallel: bool = field(default_factory=lambda: _env_flag("COSMO_THERMO_OTHER_PARALLEL", False))
    # --- GPU CUDA-graph CAPTURE-READINESS (phases A+B of the fp32-GPU port; CPU-only, default OFF). When
    #     ON, the per-step launch sequence is made (1) IN-PLACE into pre-allocated buffers -- the ~9 hot
    #     prognostic-state rebinds (self.X = self.X + ... / self._accumulate / self._cast) become
    #     arr[...] = / out= writes into buffers allocated once in __init__ -- and (2) FIXED-TOPOLOGY: the
    #     data-dependent early-exits (convadjust unstable/changed skips, the chimney mask .any() return) are
    #     dropped so the kernel chain is constant every step. This is the prerequisite for CUDA-graph capture
    #     (a captured graph records fixed device addresses + a constant op sequence; a rebind allocates a new
    #     address and a skip changes the sequence). Default OFF -> the shipped rebinding code path, BYTE-
    #     IDENTICAL (proven by SHA, _gpu_capture_sha.py). ON is BIT-IDENTICAL on the state arrays (the IEEE
    #     ops are unchanged -- only the write destination moves) and round-off-or-bit-identical on the budget
    #     diagnostics (cum_evap/precip; the only legitimate reorder). On CPU this changes nothing physical --
    #     it is purely the memory/control-flow rewrite the device port needs. env COSMO_AD_CAPTURE_READY.
    capture_ready: bool = field(default_factory=lambda: _env_flag("COSMO_AD_CAPTURE_READY", False))
    # --- time stepping -----------------------------------------------------------------------------
    dt: float = 7200.0                 # thermo time step [s] (2 h); used by the column physics (M2b.3+)
    dt_dyn: float = 60.0               # MAX explicit dynamics time step [s]; the substep is further
                                       # CFL-limited (advection + diffusion) each step (see _cfl_dt)
    cfl: float = 0.3                   # CFL safety factor for the adaptive dynamics dt
    max_sim_days: float = 0.0          # TRUE-time run budget [physical days]: run() stops once sim_time
                                       # (= Sigma adaptive dt) crosses it. 0 = off (use max_iter/convergence).
                                       # This is the honest fixed-length criterion -- iter*dt_dyn overcounts.
    # --- radiation sub-cycling (issue #86): recompute the full column radiation only every ``dt_rad``
    #     MODEL-SECONDS, holding the radiative heating H / surf_net / OLR fields constant between refreshes
    #     while the fast physics (surface fluxes, convection, condensation, advection) stay on the substep
    #     clock. The radiative field evolves on hours-days, so refreshing at a physical cadence (e.g. 1800 =
    #     30 model-min) is ~10^3x fewer column_radiation calls at the CFL-stiff coupled dt. Default 0.0 =
    #     recompute EVERY step (byte-identical). NOT a Newtonian relaxation (decision #3): the held field is
    #     the model's OWN radiation, just refreshed less often -- and refreshed EARLY if the state moves a
    #     lot (max|dT_atm| > 1 K since the last refresh). The diagnostics/conservation gates make FRESH
    #     calls (they bypass the cache), so the closure tests are unaffected.
    dt_rad: float = field(default_factory=lambda: _env_float("COSMO_DT_RAD", 0.0))
    # --- thermo sub-cycling: when thermo_dt_min > 0 and the dynamics dt exceeds it, the thermo substep
    #     is called ceil(dt/thermo_dt_min) times at dt/n_sub each, ALL columns. PR #197 validated that
    #     all-column sub-cycling closes the dt-sensitivity gap (noise floor); localized sub-cycling
    #     (only hot columns) fails because advection imports errors. INCOMPATIBLE with capture_ready
    #     (the variable-length loop breaks CUDA graph capture); for GPU production use the finest global
    #     dt with capture instead. Default 0.0 = off = byte-identical (one thermo call per step).
    thermo_dt_min: float = 0.0
    # --- FIXED thermo sub-cycling (capture-compatible): sub-cycle the thermo substep a FIXED number of
    #     times per dynamics step. Unlike thermo_dt_min (which computes n_sub from the CFL-adaptive dt and
    #     breaks capture), this is a CONSTANT loop count -> the captured graph replays it identically.
    #     Set thermo_n_sub = ceil(dt_dyn / dt_thermo_target), e.g. 30 for dt_dyn=60 / dt_thermo=2.
    #     Default 1 = off = byte-identical (one thermo call per step, the shipped path).
    thermo_n_sub: int = field(default_factory=lambda: int(_env_float("COSMO_THERMO_N_SUB", 1)))   # ⚠️ TEST default 1; CANONICAL nsub30 = 30 (dt_dyn=60 / dt_thermo=2). Set in the production driver, NOT canonical_env.sh. (CANONICAL_WORLD.md)
    # --- convergence / guards (MIRRORS StageAConfig) -----------------------------------------------
    tol: float = 1.0e-5                # steady-state stop tolerance on the drift metric
    conv_theta_max: float | None = None  # measure the drift only over theta <= this (mask the slow cold tail)
    check_every: int = 200
    patience: int = 4
    max_iter: int = 4_000_000
    max_seconds: float = 9.0 * 3600
    checkpoint_path: str | None = None
    checkpoint_every: int = 20000
    # --- resume config/world guard (_ckpt_fingerprint): every checkpoint stores a fingerprint of this
    #     config + the import-baked world forcing (L_SUN / I0 / collimator / shade); on resume it is
    #     compared against the current process. "warn" (default) prints any mismatch and PROCEEDS (so a
    #     warm-start under a different config/grid stays legal); "strict" RAISES on a world/grid mismatch
    #     (the silently-wrong-world case, e.g. the sh_b23L8 collapse); "off" skips. An OLD checkpoint
    #     with no fingerprint loads cleanly in every mode (never raises). env COSMO_RESUME_VALIDATE.
    resume_validate: str = field(default_factory=lambda: _env_str("COSMO_RESUME_VALIDATE", "warn"))
