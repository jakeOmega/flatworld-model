"""flatworld._ckpt_fingerprint -- store the AD-core CONFIG + WORLD forcing in a checkpoint, and
validate it on resume.

THE BUG THIS GUARDS (``cosmology_models/MODELING_LESSONS.md`` -- ``sh_b23L8`` resume "collapse"): an
AD-core ``.npz`` checkpoint stores only the prognostic STATE, not the config or the import-baked world
forcing. Several world-defining knobs do not live in ``StageADConfig`` at all -- they bake into the
modules AT IMPORT: ``COSMO_L_SUN`` -> ``constants.L_SUN``, ``COSMO_I0_SCALE`` ->
``constants.I0_OVERHEAD``, ``COSMO_COLLIMATOR*`` -> ``constants.COLLIMATOR*``, ``COSMO_SHADE_FILE`` ->
``geometry._SHADE``. Because they bake at import, a process that imports ``flatworld`` WITHOUT the env
set is silently "born in the wrong world", and a checkpoint's arrays are then interpreted under the
wrong forcing -- a silently-wrong result that looks exactly like a physics finding.

THE FIX. Stamp a versioned JSON fingerprint of ``{config, world}`` into the checkpoint meta
(``meta["ad_fingerprint"]``) at save time, and compare it against the current process on resume. The
JSON string rides the existing ``convergence.save_checkpoint`` meta channel unchanged (meta values are
``np.asarray``'d on save and 0-d arrays ``.item()``'d back on load, so a ``str`` round-trips).

RESUME IS STILL FREE TO DIFFER. Warm-starting deliberately resumes a donor under a DIFFERENT config (a
different grid, different physics -- see ``_warm_restart.seed``). So validation is graded:
  * ``"warn"``  (default) -- print a classified diff of any mismatch and PROCEED.
  * ``"strict"``          -- RAISE only when a WORLD-DEFINING field differs (grid or world forcing: the
                            array-meaning-changing subset); warn on other physics toggles.
  * ``"off"``             -- skip.

BACKWARD COMPATIBLE (the user's explicit requirement): an OLD checkpoint with no fingerprint loads
cleanly in EVERY mode (a one-line note, never a raise) -- only checkpoints written after this change
carry the fingerprint and gain the mismatch check.
"""
from __future__ import annotations
import dataclasses
import json
import math
import os

from . import constants as C
from . import geometry as g

SCHEMA = 1

# --- VOLATILE: runtime / IO / arithmetic-path knobs whose difference is NOT a physics mismatch, so a
#     warm-start with a longer guard, a finer checkpoint cadence, a different precision/device, etc. is
#     silent. The list is INVERTED: every config field NOT named here is treated as physics, so a NEW
#     physics knob is covered automatically (the safe default -- when in doubt a field warns, it does
#     not silently pass). precision/device are the load_state-documented cross-precision / CPU<->GPU
#     warm-start path. Names that are not (currently) config fields are harmless -- the set is only
#     consulted for keys that actually appear in ``dataclasses.asdict(cfg)``.
VOLATILE_FIELDS = frozenset({
    "checkpoint_path", "checkpoint_every", "check_every", "tol", "patience",
    "max_iter", "max_seconds", "max_sim_days", "conv_theta_max",
    "device", "precision", "poisson_method", "poisson_workers", "resume_validate",
    "thermo_fp32", "seb_legacy_fp32", "fp32_accumulate_fp64", "capture_ready",
})

# --- WORLD-DEFINING (config side): the grid the saved arrays live on -- HORIZONTAL (nx, theta_max_deg)
#     AND VERTICAL. The vertical set includes nz/p_top AND the grid-stretch knobs that move the levels at
#     FIXED nz: grid_type/dz_min set the z-levels, thermo_grid_type/thermo_dp_min the pressure levels. A
#     mismatch in ANY of these means the saved arrays live on different levels and cannot be
#     reinterpreted -> strict mode raises. (The world-FORCING half lives in world_snapshot().)
#
#     THE THREE ANELASTIC REFERENCE CONSTANTS BELONG HERE TOO (ROADMAP PHY-23, added with PHY-22 B').
#     ``StageADCore.__init__`` sets ``self.z_top = self._z_of_ref_pressure(c.p_top)`` and builds
#     ``self.z`` from it, so ``T_ref_surf`` / ``lapse_ref`` / ``T_ref_top`` move EVERY z-level at fixed
#     ``nz`` -- exactly as ``grid_type``/``dz_min`` do. They were classified as "physics" by the
#     inverted-set rule, which produced a SILENT HALF-RESTORE rather than a refusal: measured on a
#     6.5 -> 5.0 K/km mismatch, ``_restore_physics_config`` sets ``cfg.lapse_ref`` back to the
#     checkpoint's value while ``self.z_top`` (21,379 m vs the run's 20,124 m), ``self.z``, ``p_ref``,
#     ``T_ref``, ``rho_ref``, ``_beta`` and the Poisson LU/DST factorisation all stay built from the
#     OTHER one -- so every downstream diagnostic reads a ``cfg`` that does not describe the arrays.
#     PHY-22 B' makes this worse and is why it is fixed here rather than filed: ``_buoy_alpha`` is
#     ``R_d * dT_ref/dz``, so ``lapse_ref`` now sets a physics COEFFICIENT and not only a grid.
WORLD_DEFINING_CONFIG = frozenset({
    "nx", "nz", "theta_max_deg", "p_top",
    "grid_type", "dz_min", "thermo_grid_type", "thermo_dp_min",
    "T_ref_surf", "lapse_ref", "T_ref_top",
})

_L_SUN_ABS_TOL = 1.0     # [m] human-scale tolerance for the world-forcing L_SUN compare (cf _b23_regen)
_FLOAT_RTOL = 1e-9       # physics floats round-trip through JSON text; compare with a tiny rel. tol
_MISSING = object()      # sentinel: a key absent on one side (schema drift) is skipped, not flagged


def _json_default(o):
    """Coerce any stray non-JSON value (e.g. a numpy scalar) so a save never raises on serialization."""
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:  # noqa: BLE001
            pass
    return str(o)


def world_snapshot():
    """The world-FORCING that bakes into ``constants``/``geometry`` at import -- the half that is NOT in
    ``StageADConfig``. Read from the live modules so it reflects the world this process actually built.

    ``cosmo_env`` records ALL ``COSMO_*`` env vars purely as a human-readable diagnostic; it is a nested
    dict and is NOT used for the pass/fail compare (the resolved scalars above already capture the
    world, and comparing raw env strings would false-positive on config-feeding vars like
    ``COSMO_PRECISION``)."""
    return {
        "L_SUN": float(C.L_SUN),
        "I0_OVERHEAD": float(C.I0_OVERHEAD),
        "COLLIMATOR": bool(C.COLLIMATOR),
        "COLLIMATOR_HR": float(C.COLLIMATOR_HR),
        "COLLIMATOR_I0_REPIN": float(C.COLLIMATOR_I0_REPIN),
        "shade_path": (str(g._shade_path) if getattr(g, "_shade_path", None) else None),
        "shade_loaded": getattr(g, "_SHADE", None) is not None,
        # the 2-D lens / reshape optics are import-baked into geometry exactly like the 1-D shade,
        # so a stale-import mismatch on them must be visible to the diff backstop too (review #352
        # finding 5 -- the run1x lens worlds are selected by precisely these knobs). Back-compat:
        # diff_fingerprint compares only keys present on BOTH sides, so fingerprints written before
        # these keys existed are unaffected.
        "shade2d_path": (str(g._shade2d_path) if getattr(g, "_shade2d_path", None) else None),
        "shade2d_loaded": getattr(g, "_SHADE2D", None) is not None,
        "lens_reshape": bool(getattr(g, "_LENS_RESHAPE", False)),
        # The DELIVERED SPECTRUM (PR #437): COSMO_F_VIS_SW / COSMO_ALBEDO_ATM bake into radiation at
        # import exactly like L_SUN bakes into constants, and they change what the saved arrays MEAN
        # (a resume under the wrong spectrum is the sh_b23L8 failure with a different knob -- review
        # #437 finding 2). ALBEDO_LAND rides with them: three of the four surface albedos ARE
        # StageADConfig fields (albedo_ocean/ice/dryland) so the config half of the diff catches
        # those, but ALBEDO_LAND is not -- it is read straight off constants at bl_diurnal.py,
        # stage_c_map.py, salt_snow_darkening.py and as a cascade default arg, so nothing else in the
        # fingerprint would see it move (review 2026-07-31, finding F6). Lazy import: radiation pulls
        # in the thermo stack, and this module is imported by the checkpoint path, so keep the
        # top-level graph thin.
        # Back-compat: diff_fingerprint compares only keys present on BOTH sides, so pre-#437
        # checkpoints are unaffected (same contract as the shade2d/lens keys above).
        **_spectrum_snapshot(),
        "cosmo_env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("COSMO_")},
    }


def _spectrum_snapshot():
    """The import-baked spectrum givens, as resolved scalars. Never raises (see build_fingerprint).

    ALBEDO_LAND is here rather than in StageADConfig because it has no config field to ride on --
    see the comment at the call site."""
    try:
        from . import constants as const
        from . import radiation as rad
        return {"F_VIS_SW": float(rad.F_VIS_SW), "ALBEDO_ATM": float(rad.ALBEDO_ATM),
                "ALBEDO_LAND": float(const.ALBEDO_LAND)}
    except Exception:  # noqa: BLE001 -- metadata is never worth crashing a run for
        return {}


def build_fingerprint(cfg):
    """``StageADConfig`` + the world forcing as a versioned JSON string for ``meta["ad_fingerprint"]``.

    NEVER raises: a multi-hour run must not die on a metadata write, so a serialization failure returns
    ``""`` (which ``parse_fingerprint`` treats as "no fingerprint")."""
    try:
        payload = {"schema": SCHEMA, "config": dataclasses.asdict(cfg), "world": world_snapshot()}
        return json.dumps(payload, allow_nan=True, sort_keys=True, default=_json_default)
    except Exception:  # noqa: BLE001 -- metadata is never worth crashing a run for
        return ""


def parse_fingerprint(meta):
    """Pull the fingerprint dict out of a checkpoint's ``meta``; ``None`` for an old/empty/unparseable
    checkpoint (the back-compat path -- callers must treat ``None`` as "no config info available")."""
    if not meta:
        return None
    raw = meta.get("ad_fingerprint")
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    raw = str(raw)
    if not raw:
        return None
    try:
        fp = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return fp if isinstance(fp, dict) else None


def _equal(a, b, *, abs_tol=0.0):
    """Value equality: exact for int/str/bool/None; NaN==NaN; rel-tol (and optional abs-tol) for floats
    (which round-trip through JSON text). bool is compared exactly (not as a float)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, float) or isinstance(b, float):
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            return a == b
        if math.isnan(fa) and math.isnan(fb):
            return True
        if math.isinf(fa) or math.isinf(fb):
            return fa == fb
        if abs_tol > 0.0 and abs(fa - fb) <= abs_tol:
            return True
        return abs(fa - fb) <= _FLOAT_RTOL * max(abs(fa), abs(fb), 1.0)
    return a == b


def diff_fingerprint(saved_fp, cfg):
    """Classify the mismatch between a saved fingerprint and the current ``cfg`` + world.

    Returns ``{"world_defining": [...], "physics": [...], "volatile_ignored": [...]}`` where each entry
    is ``(field, saved_value, current_value)``. Only keys present on BOTH sides are compared, so adding
    or removing a config field across versions never spuriously flags (schema drift is benign)."""
    out = {"world_defining": [], "physics": [], "volatile_ignored": []}

    cur_cfg = dataclasses.asdict(cfg)
    saved_cfg = saved_fp.get("config") or {}
    for k in sorted(set(cur_cfg) & set(saved_cfg)):
        if _equal(saved_cfg[k], cur_cfg[k]):
            continue
        entry = (k, saved_cfg[k], cur_cfg[k])
        if k in VOLATILE_FIELDS:
            out["volatile_ignored"].append(entry)
        elif k in WORLD_DEFINING_CONFIG:
            out["world_defining"].append(entry)
        else:
            out["physics"].append(entry)

    cur_world = world_snapshot()
    saved_world = saved_fp.get("world") or {}
    for k in sorted(set(cur_world) & set(saved_world)):
        if isinstance(cur_world[k], dict) or isinstance(saved_world[k], dict):
            continue                                    # cosmo_env: diagnostic only, not pass/fail
        abs_tol = _L_SUN_ABS_TOL if k == "L_SUN" else 0.0
        if _equal(saved_world[k], cur_world[k], abs_tol=abs_tol):
            continue
        out["world_defining"].append(("world." + k, saved_world[k], cur_world[k]))

    return out


def _format_diff(diff):
    def _block(title, entries):
        if not entries:
            return []
        rows = [f"    {k}: checkpoint={old!r} -> current={new!r}" for k, old, new in entries]
        return [f"  {title}:"] + rows
    lines = []
    lines += _block("WORLD/GRID (changes the meaning of the saved arrays)", diff["world_defining"])
    lines += _block("physics toggles", diff["physics"])
    lines += _block("ignored (runtime/IO knobs)", diff["volatile_ignored"])
    return "\n".join(lines)


def validate_resume(meta, cfg, mode="warn", context=""):
    """Compare a checkpoint's stored fingerprint against the current ``cfg`` + world on resume.

    ``mode``: ``"off"`` -> no-op; ``"warn"`` (default) -> print any mismatch and proceed (warm-start is
    allowed); ``"strict"`` -> RAISE ``RuntimeError`` when a WORLD-DEFINING field differs (grid or world
    forcing), warn on other physics-toggle differences.

    A MISSING fingerprint is handled FIRST and in EVERY mode (including ``"strict"``): an old/bare-meta
    checkpoint prints one note and returns -- it NEVER raises. So strict can only raise on a fingerprint
    that is present AND world-mismatched, never on an absent one (old checkpoints stay fully usable).
    Two more never-raise degradations match ``build_fingerprint``'s "metadata never crashes a run" stance:
    an UNRECOGNIZED schema prints a note and still attempts the (possibly partial) compare, and a compare
    that itself throws (e.g. a future array-valued field) prints a note and proceeds.
    Returns the diff dict (or ``{}`` when there is nothing to report)."""
    tag = f"[{context}] " if context else ""
    if mode == "off":
        return {}
    saved = parse_fingerprint(meta)
    if saved is None:
        print(f"{tag}checkpoint predates config fingerprinting (no stored config/world); proceeding "
              f"without a resume config check.")
        return {}
    saved_schema = saved.get("schema")
    if saved_schema != SCHEMA:
        # Forward-compat: an UNRECOGNIZED layout is surfaced, not silently trusted. The field-intersection
        # compare below still runs (a v2 that only ADDED fields stays comparable), but a truly incompatible
        # layout could otherwise look "clean", so flag it. NEVER raise -- schema drift must stay
        # back-compatible, exactly like a missing fingerprint.
        print(f"{tag}NOTE: checkpoint fingerprint schema {saved_schema!r} differs from the current schema "
              f"{SCHEMA!r}; the config/world compare may be partial.")
    try:
        diff = diff_fingerprint(saved, cfg)
    except Exception as e:  # noqa: BLE001 -- a compare failure must never crash a resume (cf build_fingerprint)
        print(f"{tag}NOTE: could not compare the checkpoint fingerprint ({e!r}); proceeding without a "
              f"resume config check.")
        return {}
    if not diff["world_defining"] and not diff["physics"]:
        return diff
    body = _format_diff(diff)
    if mode == "strict" and diff["world_defining"]:
        raise RuntimeError(
            f"{tag}checkpoint world/config mismatch on resume (strict mode): the saved state was "
            f"produced in a DIFFERENT world/grid, so its arrays cannot be reinterpreted here. Re-launch "
            f"with the checkpoint's forcing (the COSMO_* env must be applied BEFORE importing flatworld), "
            f"or resume with resume_validate='warn' if this is an intentional warm-start.\n{body}")
    print(f"{tag}WARNING: resuming under a config that DIFFERS from the checkpoint (warm-start is "
          f"allowed -- verify this is intended):\n{body}")
    return diff
