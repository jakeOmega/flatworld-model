"""flatworld.stagec_convergence -- one convergence standard for the climate-map stage, a recorded stop in
every product, and the gate that keeps an unconverged solve out of a deliverable.

WHY THIS EXISTS (PR #852, ``flatworld/STAGEC_CONVERGENCE_SERVED_MAP.md``). The served seed-30 world's coarse
climate stopped at iteration 300 on the ``test`` profile's 600-second WALL-CLOCK guard, far from converged
(window land rain 0.397 against 1.686 mm/day once converged), and nothing recorded it: ``StageCMap.run``
returned the stop summary and ``cascade.run_cascade`` threw it away. The 10 km nests stopped on their own
rule, but that rule is PER STEP, so at their 127 s step it admits ~7x the physical drift the parent's 900 s
step does. This module fixes the three together:

1. **The rule, in model time** (``conv_rule="model_time"`` on ``StageCConfig``). The stop metric is the
   same field the per-step rule reads -- the largest relative change of the aloft moisture column,
   ``max|dW_a| / <|W_a|>`` over a check interval -- but divided by the interval's MODEL SECONDS rather
   than its step count, and the checks are spaced in model seconds rather than steps. Every constant is
   the PRODUCTION profile's existing intent (``config.stage_c("production")``: ``tol=1e-4`` per
   ``dt=600`` s step, checks every 500 steps, patience 5) re-expressed per unit model time, so at
   ``dt=600`` the new rule makes the same stop decision the production profile always did, and at any
   other ``dt`` it holds the solve to the same physical standard:

   * ``MODEL_TIME_TOL_PER_S = 1e-4 / 600 s = 1.667e-7 1/s`` -- a relative aloft-moisture drift of at most
     1.44 % per model-day, at the worst cell;
   * ``MODEL_TIME_CHECK_S = 500 x 600 s = 3.0e5 s`` (3.47 model-days) between checks;
   * ``MODEL_TIME_PATIENCE = 5`` consecutive passing checks, i.e. a 17.4 model-day window.

   Not tuned to anything: the numbers are fixed here from the production profile before any solve reads
   them (``STAGEC_CONVERGENCE_FIX.md`` section 1 records the order).

2. **The guard, in model time.** A deliverable solve has NO wall-clock guard (a wall clock makes the stop
   depend on the build machine -- the served parent's stop at 300 was one box's speed). Its backstop is
   ``MODEL_TIME_MAX_S = 180 model-days``: three times the ~58 model-days the served parent took from a
   cold start to converge under this rule's own window (crossing at ~3,900 x 900 s = 41 days, plus the
   17.4-day patience window). Hitting it is a NON-converged stop, and is refused like any other.

3. **The record and the gate.** ``stop_record`` is what ``StageCMap.run`` returns and ``run_cascade``
   now keeps as ``dC["stage_c_run"]``: converged, stop kind (converged / iteration_cap / wall_clock /
   other), iterations, model time, final drift, the water closure, and the rule and tolerance that made
   the call. ``require_converged`` is the PRODUCER gate (a route that is about to write a deliverable
   refuses an unconverged stop unless its operator passed ``--allow-unconverged``, and the product
   records that it was forced). ``check_product`` is the CONSUMER gate. In ``mode="build"`` (anything that
   writes a deliverable) it passes ONLY a product whose record says it converged on a standard at least as
   strict as this one -- compared as an effective rate per model second, not a rule name
   (``meets_standard``) -- and refuses an unconverged, off-standard or UNRECORDED product (every product
   built before this fix, nests included) unless the consumer's ``--allow-unconverged-climate``, which it
   records. In ``mode="read"`` (a serve path over existing archives) it only warns. ``check_window_records``
   applies the same to a consumer of a band-window cache (the window's parent AND its nest).

THE WATER-CLOSURE SCREEN (``closure_verdict``) -- for the AUDIT of existing archives, and the text of a
refusal; it no longer lets any product through (the record is the proof). Stage C's instantaneous closure is
``(P + vapour export) / (E + S_band)`` with ``P`` the step's 1 %-per-step EMA of the rain, so it reaches ~1
only once the solve is steady. On the served route it reads **0.9204 / 0.9206 / 0.9199** for the three
archived products truncated at iteration 300, and **1.0016** once converged (PR #852 section 2).
``CLOSURE_TOL = 0.01`` sits between the converged floor (1.6e-3) and the truncation signature (8.0e-2) --
their geometric mean is 1.1e-2 -- so every measured product classifies the same way for any threshold in
~[0.005, 0.03]. **It is a SCREEN, not a proof.** On the same solve the closure already reads 0.9935 at
iteration 600 and 0.9993 at 1,200, both far from converged (window rain 1.05 and 1.45 against 1.69), so a
product truncated later than the 600-s guard's first check would PASS it. The stop record is the proof; the
screen exists for products that predate it. It is meaningless for a NEST (an open domain whose budget
closes only jointly with its parent -- ``climate_nest`` docstring), so nests are exempt.
"""
from __future__ import annotations

import json
import math

import numpy as np

# ---- the rule (the production profile's intent, per unit model time) --------------------------------
PRODUCTION_TOL_PER_STEP = 1e-4          # config.stage_c("production").tol
PRODUCTION_DT_S = 600.0                 # config.stage_c("production").dt
PRODUCTION_CHECK_EVERY = 500            # config.stage_c("production").check_every
PRODUCTION_PATIENCE = 5                 # config.stage_c("production").patience

MODEL_TIME_TOL_PER_S = PRODUCTION_TOL_PER_STEP / PRODUCTION_DT_S       # 1.667e-7 1/s = 1.44 %/model-day
MODEL_TIME_CHECK_S = PRODUCTION_CHECK_EVERY * PRODUCTION_DT_S           # 3.0e5 s = 3.47 model-days
MODEL_TIME_PATIENCE = PRODUCTION_PATIENCE                               # 5 checks = 17.4 model-days
MODEL_TIME_MAX_S = 180.0 * 86400.0      # the machine-independent backstop (see module docstring, point 2)

CLOSURE_TOL = 0.01                      # |water_closure_inst - 1| screen threshold (full-domain products)

RECORD_SCHEMA = "stage_c_run/1"
STOP_KINDS = ("converged", "iteration_cap", "wall_clock", "other")


class StageCNotConverged(RuntimeError):
    """A deliverable route refused a climate-map solve (or product) that did not converge."""


def deliverable_overrides():
    """``StageCConfig`` overrides that put a solve on the deliverable standard: the model-time rule, the
    model-time backstop, and NO wall-clock guard. Physics-free -- merge with a route's own physics
    overrides (e.g. ``geo_climate_testcases._stage_c_overrides()``)."""
    return dict(conv_rule="model_time", conv_tol_per_s=MODEL_TIME_TOL_PER_S,
                conv_check_s=MODEL_TIME_CHECK_S, patience=MODEL_TIME_PATIENCE,
                conv_max_model_s=MODEL_TIME_MAX_S, max_seconds=None)


def check_steps(rule, dt, check_every, conv_check_s):
    """Steps per convergence check: the config's ``check_every`` for the per-step rule, and the model-time
    interval ``conv_check_s`` rounded to whole steps (at least one) for the model-time rule."""
    if rule == "model_time":
        return max(1, int(round(float(conv_check_s) / float(dt))))
    return int(check_every)


def drift_metric(dWa_max, Wa_mean_abs, n_steps, dt, rule):
    """The stop metric for one check interval of ``n_steps`` steps of ``dt`` seconds.

    ``per_step``: ``max|dW_a| / <|W_a|> / n_steps`` -- the historical form, written EXACTLY as
    ``StageCMap.run`` always computed it (so the default path is bit-for-bit unchanged).
    ``model_time``: the same relative change divided by the interval's model seconds ``n_steps * dt``,
    in 1/s. For a smoothly relaxing field the per-step value scales with ``dt``; this one does not."""
    denom = max(float(Wa_mean_abs), 1e-9)
    if rule == "model_time":
        return float(dWa_max) / denom / (int(n_steps) * float(dt))
    return float(dWa_max) / denom / int(n_steps)


def stop_kind(converged, iterations, max_iter, elapsed_s, max_seconds):
    """Classify a stop in the order ``ConvergenceMonitor.update`` tests it: converged first, then the
    iteration guard, then the wall clock."""
    if converged:
        return "converged"
    if max_iter is not None and iterations >= max_iter:
        return "iteration_cap"
    if max_seconds is not None and elapsed_s >= max_seconds:
        return "wall_clock"
    return "other"


def stop_record(*, mon_summary, rule, tol, tol_units, check_steps_, patience, dt, iterations,
                max_iter, max_seconds, final_drift, water_closure_inst, water_closure_cum, nest,
                max_model_s=None):
    """The stop summary every product carries (JSON-safe scalars only)."""
    conv = bool(mon_summary.get("converged"))
    elapsed = float(mon_summary.get("elapsed_s") or 0.0)
    kind = stop_kind(conv, int(iterations), max_iter, elapsed, max_seconds)

    def _f(v):
        v = float(v)
        return v if math.isfinite(v) else None

    return dict(schema=RECORD_SCHEMA, converged=conv, stop_kind=kind,
                stop_reason=str(mon_summary.get("stop_reason")),
                rule=str(rule), tol=float(tol), tol_units=str(tol_units),
                check_steps=int(check_steps_), check_model_s=float(check_steps_) * float(dt),
                patience=int(patience), dt_s=float(dt), iterations=int(iterations),
                model_time_s=float(iterations) * float(dt),
                model_time_days=float(iterations) * float(dt) / 86400.0,
                max_iter=None if max_iter is None else int(max_iter),
                max_model_days=None if max_model_s is None else float(max_model_s) / 86400.0,
                max_seconds=None if max_seconds is None else float(max_seconds),
                final_drift=_f(final_drift), elapsed_s=elapsed,
                water_closure_inst=_f(water_closure_inst), water_closure_cum=_f(water_closure_cum),
                nest=bool(nest))


def _describe(rec):
    return (f"{rec.get('stop_kind')} after {rec.get('iterations')} iterations "
            f"({rec.get('model_time_days', float('nan')):.1f} model-days, rule {rec.get('rule')}, "
            f"final drift {rec.get('final_drift')} {rec.get('tol_units')} vs tol {rec.get('tol')}): "
            f"{rec.get('stop_reason')}")


def require_converged(rec, *, allow_unconverged=False, what="climate-map solve", log=print):
    """PRODUCER gate: return ``rec`` if the solve converged; otherwise raise ``StageCNotConverged`` -- or, when
    the operator passed ``allow_unconverged``, warn loudly and mark the record ``allowed_unconverged``
    (so the product says it was forced, mirroring ``erosion.check_resolution``)."""
    if rec is None:
        raise StageCNotConverged(f"{what}: no stop record (the solve did not report how it stopped)")
    if isinstance(rec, dict) and rec.get("converged"):
        ok, why = meets_standard(rec)
        if ok:
            return rec
        msg = f"{what}: converged, but NOT on the deliverable standard -- {why}"
    else:
        msg = f"{what}: did NOT converge -- {_describe(rec)}"
    if not allow_unconverged:
        raise StageCNotConverged(msg + ". A deliverable must converge; pass --allow-unconverged only for a "
                                 "deliberate smoke/wiring build (the product then records that it was forced).")
    log(f"WARNING: {msg} -- FORCED by --allow-unconverged; the product records allowed_unconverged=True")
    rec = dict(rec)
    rec["allowed_unconverged"] = True
    return rec


def closure_verdict(water_closure_inst, tol=CLOSURE_TOL):
    """The legacy screen: 'ok', 'suspect' (|closure - 1| > tol) or 'unknown' (absent / non-finite)."""
    try:
        v = float(water_closure_inst)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(v):
        return "unknown"
    return "suspect" if abs(v - 1.0) > tol else "ok"


def effective_standard(rec):
    """``(tol_per_s, check_model_s, patience)`` -- a record's convergence standard in model-time units,
    whichever rule produced it: a per-step tolerance becomes ``tol / dt_s``. ``None`` entries when the record
    does not say."""
    try:
        tol, dt = float(rec.get("tol")), float(rec.get("dt_s"))
    except (TypeError, ValueError):
        return None, None, None
    units = rec.get("tol_units")
    tol_s = tol if units == "1/s" else (tol / dt if units == "1/step" else None)
    chk = rec.get("check_model_s")
    try:
        chk = float(chk)
    except (TypeError, ValueError):
        chk = None
    pat = rec.get("patience")
    return tol_s, chk, (int(pat) if pat is not None else None)


def meets_standard(rec):
    """``(ok, reason)``: is a CONVERGED record's standard at least as strict as the deliverable one? Compared
    as an effective rate, not a rule name (PR #853 review finding 1), so a per-step record strict enough in
    model time passes -- e.g. the production profile itself at dt=600 s -- and a looser one of either rule
    does not: the tolerance per model second no larger than ``MODEL_TIME_TOL_PER_S``, the check interval no
    shorter than ``MODEL_TIME_CHECK_S`` (less half a step, the rounding ``check_steps`` does), and at least
    ``MODEL_TIME_PATIENCE`` consecutive passes."""
    tol_s, chk, pat = effective_standard(rec)
    dt = rec.get("dt_s") or 0.0
    if tol_s is None or chk is None or pat is None:
        return False, "the record does not state its tolerance, units, check interval and patience"
    if tol_s > MODEL_TIME_TOL_PER_S * (1.0 + 1e-9):
        return False, (f"its tolerance {tol_s:.3e} 1/s (rule {rec.get('rule')}) is looser than the "
                       f"deliverable {MODEL_TIME_TOL_PER_S:.3e} 1/s")
    if chk < MODEL_TIME_CHECK_S - 0.5 * float(dt) - 1e-6:
        return False, f"its check interval {chk:.0f} s is shorter than the deliverable {MODEL_TIME_CHECK_S:.0f} s"
    if pat < MODEL_TIME_PATIENCE:
        return False, f"its patience {pat} is below the deliverable {MODEL_TIME_PATIENCE}"
    return True, "on the deliverable standard"


def classify_record(dC, *, kind):
    """``(verdict, detail)`` for a product's own stop record, ``verdict`` one of ``converged`` (converged ON
    the deliverable standard), ``unconverged``, ``off_standard`` (converged, but to a looser standard) or
    ``unrecorded`` (no record at all -- every product built before this fix)."""
    rec = dC.get("stage_c_run") if isinstance(dC, dict) else None
    if not isinstance(rec, dict):
        if kind == "nest":
            return "unrecorded", "a nest product with NO stop record (a nest's water closure cannot screen it)"
        ci = dC.get("water_closure_inst") if isinstance(dC, dict) else None
        v = closure_verdict(ci)
        screen = {"suspect": f"and its water closure {float(ci):.4f} carries the truncated-solve signature "
                             f"(the served iteration-300 parents read 0.920)" if v == "suspect" else "",
                  "ok": f"its water closure {float(ci):.4f} passes the screen, which cannot prove "
                        f"convergence (0.9935 passes at iteration 600)" if v == "ok" else "",
                  "unknown": "and no water closure to screen"}[v]
        return "unrecorded", f"a product with NO stop record; {screen}"
    if not rec.get("converged"):
        return "unconverged", f"its solve did NOT converge -- {_describe(rec)}"
    ok, why = meets_standard(rec)
    if not ok:
        return "off_standard", f"it converged, but not on the deliverable standard: {why}"
    return "converged", _describe(rec)


def check_product(dC, *, kind, allow_unconverged=False, mode="build", what="climate product", log=print):
    """The CONSUMER gate for a climate product. ``kind``: ``"full_domain"`` (a coarse parent) or ``"nest"`` (a
    refined sidecar). Returns a JSON-safe dict for the consumer to embed in its own output; its ``status`` is
    ``converged``, or ``forced_<verdict>`` when a non-converged product was consumed under the override, or
    (``mode="read"`` only) the bare verdict.

    ``mode="build"`` -- anything that WRITES a deliverable (the window build and its pre-flight, the compact
    archive, the committed panel renders): only a product whose record says it converged ON the deliverable
    standard passes. An unconverged, off-standard or UNRECORDED product -- every product built before this
    fix, parents and nests alike -- is refused unless ``allow_unconverged`` (the consumer's
    ``--allow-unconverged-climate``), which is recorded as ``forced_*`` in the output.

    ``mode="read"`` -- a SERVE path reading an existing archive (the map webapp): never refuses; warns once
    and returns the verdict, so the live map keeps serving the editions that exist."""
    if kind not in ("full_domain", "nest"):
        raise ValueError(f"kind must be 'full_domain' or 'nest', got {kind!r}")
    if mode not in ("build", "read"):
        raise ValueError(f"mode must be 'build' or 'read', got {mode!r}")
    verdict, detail = classify_record(dC, kind=kind)
    rec = dC.get("stage_c_run") if isinstance(dC, dict) else None
    out = dict(kind=kind, verdict=verdict, record=rec if isinstance(rec, dict) else None)
    if isinstance(dC, dict) and dC.get("water_closure_inst") is not None and not isinstance(rec, dict):
        out["water_closure_inst"] = float(dC["water_closure_inst"])
    if verdict == "converged":
        out["status"] = "converged"
        return out
    msg = f"{what}: {detail}"
    if mode == "read":
        log(f"WARNING: {msg} (served as-is)")
        out["status"] = verdict
        return out
    if not allow_unconverged:
        raise StageCNotConverged(msg + ". A deliverable is built only on a product that converged on the "
                                 "deliverable standard: re-solve it through geo_climate_testcases / "
                                 "refine_climate_window, or pass --allow-unconverged-climate for a deliberate "
                                 "smoke/wiring build (the output then records the forcing).")
    log(f"WARNING: {msg} -- consumed under --allow-unconverged-climate")
    out["status"] = f"forced_{verdict}"
    return out


def check_window_records(zw, *, allow_unconverged=False, what="window", log=print):
    """BUILD gate for a consumer of a band-window CACHE (the compact archive): the window must say it was built
    on a converged parent AND, if it consumed a refined sidecar (its ``climate_refined`` knob is non-empty), on a
    converged nest (PR #853 review finding 3). ``zw`` is the loaded cache (an npz or a dict). A window that
    predates the record is unrecorded, and refused like any other unrecorded product. Returns
    ``dict(parent=..., nest=...)`` statuses for the consumer to record."""
    files = set(zw.files) if hasattr(zw, "files") else set(zw)

    def _status(key):
        if key not in files:
            return None
        txt = str(zw[key])
        if not txt:
            return ""
        try:
            return json.loads(txt).get("status")
        except (ValueError, AttributeError):
            return "unreadable"

    nest_used = "climate_refined" in files and bool(str(zw["climate_refined"]))
    got = dict(parent=_status("climate_stop"), nest=_status("climate_refined_stop") if nest_used else "unused")
    problems = []
    if got["parent"] != "converged":
        problems.append(f"its parent climate is {got['parent'] or 'unrecorded (the cache predates the record)'}")
    if nest_used and got["nest"] != "converged":
        problems.append(f"its refined sidecar is {got['nest'] or 'unrecorded (the cache predates the record)'}")
    if not problems:
        return got
    msg = f"{what}: " + "; ".join(problems)
    if not allow_unconverged:
        raise StageCNotConverged(msg + ". Rebuild the window on converged climate, or pass "
                                 "--allow-unconverged-climate for a deliberate smoke/wiring build.")
    log(f"WARNING: {msg} -- consumed under --allow-unconverged-climate")
    return dict(got, forced=True)


def to_json(obj):
    """Compact JSON for embedding in an npz as a 0-d unicode array (allow_pickle=False safe)."""
    return json.dumps(obj, sort_keys=True, default=_json_default)


def _json_default(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.bool_):
        return bool(v)
    return str(v)
