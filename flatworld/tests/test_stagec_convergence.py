"""The Stage-C convergence fix (PR #852 section 4; ``flatworld/STAGEC_CONVERGENCE_FIX.md``): one standard in
MODEL time, a recorded stop in every product, and deliverable routes that refuse a truncated solve.

The served seed-30 world's coarse climate stopped at iteration 300 on a 600-second wall clock and nothing
recorded it; its 10 km nests stopped on a PER-STEP rule ~7x looser at their shorter step. These tests pin
the three halves of the fix on tiny grids (seconds), plus the routes' wiring with the solve stubbed."""
import importlib
import json
import os
import sys

import numpy as np
import pytest

from flatworld import convergence as conv
from flatworld import stagec_convergence as scv
from flatworld.stage_c_map import StageCConfig, StageCMap
from flatworld.topography import Topography

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # cosmology_models/


@pytest.fixture(scope="module")
def topo():
    return Topography.procedural_continent(seed=4, nx=40, ne=40, Le_km=6000)


def _solve(topo, **kw):
    m = StageCMap(topo, StageCConfig(**kw))
    return m, m.run(verbose=False)


# ---------------------------------------------------------------------------------------------------
# 1. the rule
# ---------------------------------------------------------------------------------------------------

def test_the_rule_is_the_production_profiles_intent_per_unit_model_time():
    """No free number: every constant is config.stage_c('production')'s, re-expressed per model second."""
    from flatworld import config as cfgmod
    prod = cfgmod.stage_c("production")
    assert scv.MODEL_TIME_TOL_PER_S == pytest.approx(prod.tol / prod.dt)          # 1e-4 / 600 s
    assert scv.MODEL_TIME_CHECK_S == pytest.approx(prod.check_every * prod.dt)    # 500 x 600 s
    assert scv.MODEL_TIME_PATIENCE == prod.patience == 5
    ov = scv.deliverable_overrides()
    assert ov["conv_rule"] == "model_time" and ov["max_seconds"] is None          # no wall clock
    assert ov["conv_max_model_s"] == pytest.approx(180 * 86400.0)
    assert StageCConfig().conv_rule == "per_step"                                  # the default is unchanged
    with pytest.raises(ValueError, match="conv_rule"):
        StageCConfig(conv_rule="per_second")


def test_model_time_rule_stops_at_one_model_time_whatever_the_step(topo):
    """THE fix. At a fixed per-step tolerance, halving dt halves each step's change, so the per-step rule
    stops EARLIER in model time -- the nests' ~7x-looser standard (PR #852). The model-time rule stops at
    the same model time at every dt. Tolerance and check spacing are matched at dt=900 so the two rules
    agree there (the next test)."""
    tol, every, dt_ref = 5e-4, 300, 900.0
    per, mod = {}, {}
    for dt in (900.0, 225.0):
        _, s = _solve(topo, dt=dt, check_every=every, tol=tol, patience=4, max_seconds=None, max_iter=200_000)
        per[dt] = s["model_time_s"]
        _, s = _solve(topo, dt=dt, **{**scv.deliverable_overrides(), "conv_tol_per_s": tol / dt_ref,
                                      "conv_check_s": every * dt_ref, "patience": 4})
        assert s["converged"] and s["rule"] == "model_time" and s["tol_units"] == "1/s"
        mod[dt] = s["model_time_s"]
    assert per[225.0] < 0.6 * per[900.0]                              # the per-step rule's defect, measured
    assert mod[225.0] == pytest.approx(mod[900.0], abs=every * dt_ref)   # within one check interval
    assert mod[225.0] > 1.5 * per[225.0]


def test_model_time_rule_reproduces_the_per_step_decision_at_the_reference_dt(topo):
    """With the tolerance and the check spacing matched at dt, the two rules make the SAME stop decision on the
    SAME trajectory -- so at the production dt the new rule is the production profile's own standard."""
    dt, tol, every = 900.0, 5e-4, 300
    a, sa = _solve(topo, dt=dt, check_every=every, tol=tol, patience=3, max_seconds=None)
    b, sb = _solve(topo, dt=dt, conv_rule="model_time", conv_tol_per_s=tol / dt, conv_check_s=every * dt,
                   patience=3, max_seconds=None)
    assert sa["iterations"] == sb["iterations"] and sa["check_steps"] == sb["check_steps"] == every
    assert np.array_equal(a.W_a, b.W_a) and np.array_equal(a.P, b.P)       # same trajectory, same stop
    assert sb["final_drift"] == pytest.approx(sa["final_drift"] / dt, rel=1e-12)


def _historical_run(m):
    """The per-step loop exactly as StageCMap.run read before the fix (the bit-identity reference)."""
    c = m.cfg
    mon = conv.ConvergenceMonitor(tol=c.tol, patience=c.patience, max_iter=c.max_iter,
                                  max_seconds=c.max_seconds, name="max|dWa|/<Wa>/step", absolute=True)
    Wa_prev = m.W_a.copy()
    while True:
        for _ in range(c.check_every):
            m.step()
        denom = max(np.mean(np.abs(m.W_a)), 1e-9)
        drift = float(np.max(np.abs(m.W_a - Wa_prev))) / denom / c.check_every
        Wa_prev = m.W_a.copy()
        if mon.update(m.iter, drift):
            break
    return mon.summary()


def test_the_default_rule_is_bit_identical_to_the_historical_loop(topo):
    """The OFF path: every non-deliverable caller (unit tests, sweeps, study scripts) keeps the per-step rule,
    and its trajectory and stop are bit-for-bit what the pre-fix loop produced."""
    # dt=450, NOT the 900 s the model-time pins use as their reference: any dt-dependent leak into the
    # per-step metric must show here (a dt=900 run is blind to a factor dt/900 -- mutation M8)
    kw = dict(dt=450.0, check_every=200, tol=5e-4, patience=3, max_seconds=None)
    old = StageCMap(topo, StageCConfig(**kw)); so = _historical_run(old)
    new = StageCMap(topo, StageCConfig(**kw)); sn = new.run(verbose=False)
    assert sn["iterations"] == so["iterations"] and sn["converged"] == so["converged"]
    for k in ("W_a", "W_s", "P", "T_map_C"):
        assert np.array_equal(getattr(new, k), getattr(old, k)), k
    assert {k: sn[k] for k in so if k != "elapsed_s"} == {k: so[k] for k in so if k != "elapsed_s"}


# ---------------------------------------------------------------------------------------------------
# 2. the stop record
# ---------------------------------------------------------------------------------------------------

def test_run_returns_and_keeps_a_complete_stop_record(topo):
    m, s = _solve(topo, dt=900.0, **scv.deliverable_overrides())
    rec = m.last_run
    assert {k: s[k] for k in rec} == rec                         # run() returns it merged into the summary
    assert rec["schema"] == scv.RECORD_SCHEMA and rec["converged"] and rec["stop_kind"] == "converged"
    assert rec["rule"] == "model_time" and rec["tol"] == pytest.approx(scv.MODEL_TIME_TOL_PER_S)
    assert rec["check_steps"] == round(scv.MODEL_TIME_CHECK_S / 900.0)
    assert rec["model_time_s"] == pytest.approx(rec["iterations"] * 900.0)
    assert rec["final_drift"] < rec["tol"] and rec["max_seconds"] is None
    assert rec["max_iter"] == int(np.ceil(scv.MODEL_TIME_MAX_S / 900.0))
    assert 0.99 < rec["water_closure_inst"] < 1.01 and rec["nest"] is False
    json.loads(scv.to_json(rec))                                  # JSON-safe for the npz/prov embeddings


@pytest.mark.parametrize("kw,kind", [
    (dict(max_iter=600, max_seconds=None), "iteration_cap"),
    (dict(max_seconds=0.0), "wall_clock"),
    (dict(conv_rule="model_time", conv_max_model_s=5 * 86400.0, max_seconds=None), "iteration_cap"),
])
def test_every_stop_is_classified(topo, kw, kind):
    base = dict(dt=900.0, check_every=300, tol=1e-12, patience=4)       # a tolerance nothing meets
    base.update(kw)
    m, s = _solve(topo, **base)
    assert not s["converged"] and s["stop_kind"] == kind
    if "conv_max_model_s" in kw:                                          # the model-time backstop IS max_iter
        assert s["max_iter"] == int(np.ceil(5 * 86400.0 / 900.0)) and s["iterations"] >= s["max_iter"]


# ---------------------------------------------------------------------------------------------------
# 3. the gates
# ---------------------------------------------------------------------------------------------------

def _rec(converged, kind="converged", **kw):
    """A record ON the deliverable standard by default (model-time rule at dt=900 s); override fields to
    make it looser."""
    r = dict(schema=scv.RECORD_SCHEMA, converged=converged, stop_kind=kind, stop_reason="x", rule="model_time",
             tol=scv.MODEL_TIME_TOL_PER_S, tol_units="1/s", dt_s=900.0, check_steps=333,
             check_model_s=333 * 900.0, patience=scv.MODEL_TIME_PATIENCE, iterations=5661,
             model_time_days=59.0, final_drift=1e-9)
    r.update(kw)
    return r


# the production profile's own per-step standard, and the test profile's (3.3x looser in model time)
PROD_PER_STEP = dict(rule="per_step", tol=1e-4, tol_units="1/step", dt_s=600.0, check_steps=500,
                     check_model_s=500 * 600.0, patience=5)
TEST_PER_STEP = dict(rule="per_step", tol=5e-4, tol_units="1/step", dt_s=900.0, check_steps=300,
                     check_model_s=300 * 900.0, patience=4)


def test_the_producer_gate_refuses_every_unconverged_stop():
    assert scv.require_converged(_rec(True)) == _rec(True)
    for kind in ("wall_clock", "iteration_cap", "other"):
        with pytest.raises(scv.StageCNotConverged, match=kind):
            scv.require_converged(_rec(False, kind))
    with pytest.raises(scv.StageCNotConverged, match="no stop record"):
        scv.require_converged(None)
    with pytest.raises(scv.StageCNotConverged, match="NOT on the deliverable standard"):
        scv.require_converged(_rec(True, **TEST_PER_STEP))           # converged, but to a looser standard
    logs = []
    forced = scv.require_converged(_rec(False, "wall_clock"), allow_unconverged=True, log=logs.append)
    assert forced["allowed_unconverged"] is True and "FORCED" in logs[0]


def test_the_standard_is_compared_as_an_effective_rate_not_a_rule_name():
    """PR #853 review finding 1: ``converged: True`` is only as good as the standard it converged to. The
    production profile's per-step standard at dt=600 s IS the deliverable one; the test profile's is 3.3x
    looser; a STRICTER per-step record passes; a short check interval or low patience does not."""
    assert scv.meets_standard(_rec(True))[0]
    assert scv.meets_standard(_rec(True, **PROD_PER_STEP))[0]
    ok, why = scv.meets_standard(_rec(True, **TEST_PER_STEP))
    assert not ok and "looser" in why
    strict = dict(PROD_PER_STEP, tol=1e-4, dt_s=900.0, check_steps=334, check_model_s=334 * 900.0)
    assert scv.meets_standard(_rec(True, **strict))[0]                 # 1.1e-7 1/s: stricter, per-step
    assert not scv.meets_standard(_rec(True, check_steps=100, check_model_s=90000.0))[0]
    assert not scv.meets_standard(_rec(True, patience=4))[0]
    assert not scv.meets_standard(_rec(True, tol_units=None))[0]
    assert not scv.meets_standard({"converged": True})[0]


def test_build_consumers_refuse_everything_but_a_record_on_the_standard():
    """PR #853 review findings 1-2: in BUILD mode only a record that converged ON the deliverable standard
    passes -- an off-standard record, an unconverged one and an UNRECORDED product (parent or nest, whatever
    its closure) are refused unless the override, which the output records as forced_*."""
    quiet = dict(log=lambda m: None)
    assert scv.check_product({"stage_c_run": _rec(True)}, kind="full_domain", **quiet)["status"] == "converged"
    assert scv.check_product({"stage_c_run": _rec(True, **PROD_PER_STEP)}, kind="nest",
                             **quiet)["status"] == "converged"
    cases = {"off_standard": {"stage_c_run": _rec(True, **TEST_PER_STEP)},
             "unconverged": {"stage_c_run": _rec(False, "wall_clock")},
             "unrecorded": {"water_closure_inst": 1.0016}}               # closure OK -- still refused
    for verdict, dC in cases.items():
        for kind in ("full_domain", "nest"):
            with pytest.raises(scv.StageCNotConverged):
                scv.check_product(dC, kind=kind, **quiet)
            g = scv.check_product(dC, kind=kind, allow_unconverged=True, **quiet)
            assert g["status"] == f"forced_{verdict}" and g["verdict"] == verdict
    with pytest.raises(scv.StageCNotConverged, match="0.9204"):          # the refusal names the signature
        scv.check_product({"water_closure_inst": 0.9204}, kind="full_domain", **quiet)
    with pytest.raises(scv.StageCNotConverged, match="cannot prove"):
        scv.check_product({"water_closure_inst": 0.9935}, kind="full_domain", **quiet)
    with pytest.raises(scv.StageCNotConverged):                          # a legacy nest, whatever it says
        scv.check_product({"water_closure_inst": 1.0, "nest_rim_w": 1}, kind="nest", **quiet)
    with pytest.raises(ValueError):
        scv.check_product({}, kind="window", **quiet)


def test_serve_readers_warn_and_never_refuse():
    """The webapp's read path keeps serving the editions that exist: ``mode="read"`` returns the verdict."""
    logs = []
    for dC, verdict in (({}, "unrecorded"), ({"stage_c_run": _rec(False, "wall_clock")}, "unconverged"),
                        ({"stage_c_run": _rec(True, **TEST_PER_STEP)}, "off_standard")):
        g = scv.check_product(dC, kind="full_domain", mode="read", log=logs.append)
        assert g["status"] == verdict
    assert len(logs) == 3 and all("served as-is" in m for m in logs)
    assert scv.check_product({"stage_c_run": _rec(True)}, kind="nest", mode="read")["status"] == "converged"


def _window(parent="converged", nest=None, refined_path=""):
    """A band-window cache's Stage-C keys as ``build_band_window`` writes them; ``None`` omits a key."""
    w = {"climate_refined": refined_path}
    if parent is not None:
        w["climate_stop"] = scv.to_json(dict(status=parent))
    if nest is not None:
        w["climate_refined_stop"] = scv.to_json(dict(status=nest)) if nest else ""
    return w


def test_window_consumers_check_the_parent_and_the_nest():
    """PR #853 review finding 3: a consumer of a window cache (the compact archive, a render) needs the
    window's parent AND, when it consumed one, its nest to have converged -- a window forced under its own
    override, or one that predates the record, is refused here too."""
    quiet = dict(log=lambda m: None)
    assert scv.check_window_records(_window(nest=""), **quiet) == dict(parent="converged", nest="unused")
    assert scv.check_window_records(_window(nest="converged", refined_path="n.npz"),
                                    **quiet)["nest"] == "converged"
    for w in (_window(parent="forced_unconverged", nest=""),                           # forced parent
              _window(parent=None),                                                    # pre-record window
              _window(nest="forced_unrecorded", refined_path="n.npz"),                 # forced nest
              _window(nest=None, refined_path="n.npz")):                               # nest, no record
        with pytest.raises(scv.StageCNotConverged):
            scv.check_window_records(w, **quiet)
        assert scv.check_window_records(w, allow_unconverged=True, **quiet)["forced"] is True


def test_the_closure_screen_threshold_and_its_documented_blind_spot():
    """0.9204/0.9206/0.9199 (the three truncated editions) are caught and 1.0016 (converged) is not -- and so
    is 0.9935, the SAME solve at iteration 600 with window rain at 62 % of converged. The screen cannot
    replace the record; this pin keeps the note's warning honest."""
    for v in (0.9204, 0.9206, 0.9199):
        assert scv.closure_verdict(v) == "suspect"
    assert scv.closure_verdict(1.0016) == "ok"
    assert scv.closure_verdict(0.9935) == "ok"                     # the blind spot, stated
    assert scv.closure_verdict(float("nan")) == scv.closure_verdict(None) == "unknown"


# ---------------------------------------------------------------------------------------------------
# 4. the routes' wiring (the solve stubbed -- the routes need the canonical backbone to run for real)
# ---------------------------------------------------------------------------------------------------

@pytest.fixture
def isolated_import():
    """Import a route module with the COSMO_* env restored afterwards (the refine_writer pattern)."""
    saved = os.environ.copy()
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    try:
        yield importlib.import_module
    finally:
        for k in set(os.environ) - set(saved):
            del os.environ[k]
        os.environ.update(saved)


def test_geo_climate_testcases_solves_on_the_deliverable_standard_and_gates(isolated_import):
    gct = isolated_import("geo_climate_testcases")
    ov = gct._deliverable_stage_c()
    assert ov["resolved_supply"] is True                              # the gallery physics is kept...
    for k, v in scv.deliverable_overrides().items():                  # ...on the deliverable standard
        assert ov[k] == v
    assert gct._deliverable_stage_c(max_model_days=5)["conv_max_model_s"] == 5 * 86400.0
    res = {"dC": {"stage_c_run": _rec(False, "wall_clock")}}
    with pytest.raises(scv.StageCNotConverged, match="seed 30"):
        gct._gate_stage_c(res, 30, allow_unconverged=False)
    assert gct._gate_stage_c(res, 30, allow_unconverged=True)["allowed_unconverged"] is True
    assert res["dC"]["stage_c_run"]["allowed_unconverged"] is True    # the product carries the forcing


def test_refine_nests_share_the_parents_standard_and_gate_per_tile(isolated_import, monkeypatch):
    rcw = isolated_import("refine_climate_window")
    args = rcw._parse_args([])
    assert args.max_hours is None and not args.allow_unconverged       # no wall clock by default
    ov = rcw._tile_overrides("NB", 127.0, args, "ck", {"base": 1})
    assert ov["conv_rule"] == "model_time" and ov["max_seconds"] is None and ov["dt"] == 127.0
    assert ov["nest_bc"] == "NB" and ov["base"] == 1
    assert rcw._tile_overrides("NB", 127.0, rcw._parse_args(["--max-hours", "2"]), "ck", {})["max_seconds"] == 7200
    smoke = rcw._tile_overrides("NB", 127.0, rcw._parse_args(["--smoke"]), "ck", {})
    assert "conv_rule" not in smoke and smoke["max_seconds"] == 600.0      # the wiring pass keeps its budget

    from flatworld import cascade
    monkeypatch.setattr(cascade, "run_cascade",
                        lambda **kw: {"dC": {"stage_c_run": _rec(False, "iteration_cap")}})
    monkeypatch.setattr(rcw, "_log_supply_mode", lambda *a, **k: None)
    monkeypatch.setattr(rcw, "_log_wall_health", lambda *a, **k: None)
    with pytest.raises(scv.StageCNotConverged, match="iteration_cap"):
        rcw._solve_tile(None, None, "NB", 127.0, args, "ck", {})
    res = rcw._solve_tile(None, None, "NB", 127.0, rcw._parse_args(["--smoke"]), "ck", {})
    assert res["dC"]["stage_c_run"]["allowed_unconverged"] is True       # --smoke implies the override
    combined = rcw._combine_tile_runs([dict(_rec(True), tile=0), dict(_rec(False, "iteration_cap"), tile=1)])
    assert combined["converged"] is False and combined["stop_kind"] == "iteration_cap"
    assert [t["tile"] for t in combined["tiles"]] == [0, 1]


def test_the_consumers_expose_the_override_and_record_what_they_read(isolated_import):
    bw = isolated_import("build_band_window")
    assert bw.parse_args([]).allow_unconverged_climate is False
    assert bw.parse_args(["--allow-unconverged-climate"]).allow_unconverged_climate is True
    assert {"climate_stop", "climate_refined_stop"} <= set(bw._CACHE_KEYS)
    bwc = isolated_import("band_window_compact")
    good = {"stage_c_run": _rec(True)}
    rec = bwc.climate_stop_records(_window(nest="converged", refined_path="n.npz"), good, "w", "c")
    assert json.loads(str(rec["climate_stop"]))["status"] == "converged"
    assert json.loads(str(rec["climate_refined_stop"]))["status"] == "converged"
    assert json.loads(str(rec["window_climate_gate"])) == dict(parent="converged", nest="converged")
    with pytest.raises(scv.StageCNotConverged):                        # an unconverged parent climate
        bwc.climate_stop_records(_window(nest=""), {"stage_c_run": _rec(False, "wall_clock")}, "w", "c")
    with pytest.raises(scv.StageCNotConverged):                        # a window over a forced nest
        bwc.climate_stop_records(_window(nest="forced_unconverged", refined_path="n.npz"), good, "w", "c")
    forced = bwc.climate_stop_records(_window(parent=None), {}, "w", "c", allow_unconverged=True)
    assert json.loads(str(forced["climate_stop"]))["status"] == "forced_unrecorded"
    assert json.loads(str(forced["window_climate_gate"]))["forced"] is True
    # ...and build() takes its gates AND its archive keys from that one function, before the heavy work
    src = open(bwc.__file__, encoding="utf-8").read()
    body = src[src.index("def build(args):"):]
    assert body.index("climate_stop_records(") < body.index("_quantize_elev(")
    assert "**stop_records," in body


# ---------------------------------------------------------------------------------------------------
# 5. the audit checker
# ---------------------------------------------------------------------------------------------------

def test_the_audit_flags_truncated_and_unconverged_products(tmp_path, isolated_import):
    audit = isolated_import("stagec_closure_audit")
    paths = {}
    for name, dC in [("served", {"water_closure_inst": 0.9204, "P_land_mean": 0.09}),
                     ("converged_legacy", {"water_closure_inst": 1.0016}),
                     ("recorded_bad", {"water_closure_inst": 1.0, "stage_c_run": _rec(False, "wall_clock")}),
                     ("recorded_ok", {"water_closure_inst": 1.0, "stage_c_run": _rec(True)}),
                     ("recorded_loose", {"water_closure_inst": 1.0, "stage_c_run": _rec(True, **TEST_PER_STEP)}),
                     ("nest", {"water_closure_inst": 0.5, "nest_rim_w": np.zeros(2)})]:
        p = tmp_path / f"{name}.npz"
        np.savez(p, dC=dC)
        paths[name] = str(p)
    rows = {os.path.basename(r["path"])[:-4]: r for r in
            (audit.audit_one(p, hash_files=False) for p in paths.values())}
    assert rows["served"]["flagged"] and rows["recorded_bad"]["flagged"]
    assert not rows["converged_legacy"]["flagged"] and not rows["recorded_ok"]["flagged"]
    assert rows["recorded_loose"]["flagged"] and rows["recorded_loose"]["verdict"].startswith("OFF-STANDARD")
    assert not rows["nest"]["flagged"] and rows["nest"]["kind"] == "nest"
    assert audit.main([paths["served"], "--no-hash"]) == 2
    assert audit.main([paths["served"], "--no-hash", "--report-only"]) == 0
    assert audit.main([paths["recorded_ok"], "--no-hash"]) == 0


def test_the_refine_prov_records_the_standard_the_tiles_solved_to(isolated_import):
    """PR #853 review S2 and the stale-tol finding: the nest's ``.prov.json`` carries its stop record and the
    parent-climate gate, and a real nest's prov does not print the legacy per-step ``--tol`` beside them."""
    rcw = isolated_import("refine_climate_window")

    class T:
        dx, de = 10165.0, 10164.0

    for argv, rule, tol in (([], "model_time", None), (["--smoke"], "per_step", 5e-4)):
        prov = rcw.prov_record(rcw._parse_args(argv), T(), 127.0, 0.98, _rec(True), {"status": "converged"},
                               minutes=1.0)
        assert prov["stage_c_run"] == _rec(True) and prov["parent_climate_gate"]["status"] == "converged"
        assert prov["conv_rule"] == rule and prov["tol"] == tol
        json.dumps(prov)
    src = open(rcw.__file__, encoding="utf-8").read()
    assert "prov = prov_record(a, topo, dt, agree, stage_c_rec, parent_gate," in src
    main = src[src.index("def main(argv=None):"):]
    assert main.index("parent_gate = scv.check_product(dC_coarse") < main.index("build_nest_bc")


def test_as1_arms_solve_on_the_deliverable_standard(isolated_import):
    """PR #853 review finding 5, made EXPLICIT: every AS-1 arm solves on the deliverable standard at its own
    maritime_L_km (the header prices it at ~7.5 CPU-h and flags the truncated historical verdicts)."""
    as1 = isolated_import("_as1_maritime_sweep")
    gct = isolated_import("geo_climate_testcases")
    assert as1.ARM_CONVERGENCE == "deliverable"
    ov = as1.arm_stage_c_overrides(gct, 123.0)
    assert ov["maritime_L_km"] == 123.0 and ov["resolved_supply"] is True
    for k, v in scv.deliverable_overrides().items():
        assert ov[k] == v
    assert "maritime_L_km" not in gct._stage_c_overrides()            # the patch is restored
    assert "allow_unconverged=False" in open(as1.__file__, encoding="utf-8").read()


def test_the_refine_driver_refuses_an_unconverged_parent_before_the_backbone(tmp_path, isolated_import):
    """C7, behaviourally: ``refine_climate_window.main`` on a tiny synthetic window refuses a parent with no
    stop record (the served nests' iteration-300 parents have none) BEFORE it builds the canonical backbone;
    under --allow-unconverged it gets past the gate (and stops later, on the toy inputs, for other reasons)."""
    rcw = isolated_import("refine_climate_window")
    win = tmp_path / "band_window_data.npz"
    np.savez(win, x_f=np.linspace(6.0e7, 6.1e7, 8), e_f=np.linspace(6.0e7, 6.1e7, 9), elev=np.ones((8, 9)))
    clim = tmp_path / "parent.npz"
    np.savez(clim, dC=np.array({"water_closure_inst": 0.9204, "land": np.zeros((2, 2), bool)}, dtype=object))
    argv = ["--window", str(win), "--climate", str(clim), "--out", str(tmp_path / "o.npz"),
            "--ckpt", str(tmp_path / "no_such_ckpt")]
    with pytest.raises(scv.StageCNotConverged, match="parent climate"):
        rcw.main(argv)
    with pytest.raises(Exception) as ei:                             # past the gate, fails on the toy later
        rcw.main(argv + ["--allow-unconverged"])
    assert not isinstance(ei.value, scv.StageCNotConverged)


def test_a_geoclim_cache_hit_counts_only_a_product_on_the_standard(tmp_path, isolated_import):
    """G1 (PR #853 re-review): without --force, an existing climate product is 'done' only if its own record
    says it converged on the deliverable standard -- an unrecorded or off-standard cached product fails the
    seed (unless --allow-unconverged)."""
    gct = isolated_import("geo_climate_testcases")
    paths = {}
    for name, dC in (("ok", {"stage_c_run": _rec(True)}), ("unrecorded", {"water_closure_inst": 0.9204}),
                     ("loose", {"stage_c_run": _rec(True, **TEST_PER_STEP)})):
        p = tmp_path / f"climate_seed30_{name}.npz"
        np.savez(p, dC=np.array(dC, dtype=object))
        paths[name] = str(p)
    assert gct.cached_product_ok(paths["ok"], 30) is True
    assert gct.cached_product_ok(paths["unrecorded"], 30) is False
    assert gct.cached_product_ok(paths["loose"], 30) is False
    assert gct.cached_product_ok(paths["unrecorded"], 30, allow_unconverged=True) is True
    src = open(gct.__file__, encoding="utf-8").read()
    main = src[src.index("def main():"):]
    assert "if cached_product_ok(npz_path, s, a.allow_unconverged):" in main
