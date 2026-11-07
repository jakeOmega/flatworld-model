"""#294-A: the vapour cap is the PHYSICAL ceiling (``th.Q_MAX``), and what it clips is ACCOUNTED.

Background (the bug these tests lock down). The AD core clipped its prognostic vapour to ``q <= 0.5``
every thermo substep, writing the clipped state back and never counting what it removed -- and the
budget diagnostics applied the SAME clip, so ``water_conservation`` could not see the loss. The 0.5
was an Earth-era guard (Earth q ~ 0.04, so it never bound) introduced with the original prognostic
moisture (bae5d56, M2b.3). It stopped being slack when the lens world put a 92 C SEA at theta~22:
``qsat(92.6 C) = 0.694 > 0.5``, so the cap bound BELOW saturation -- condensation could never fire in
that cell, the clip became its only vapour sink, and it silently deleted ~15% of ALL evaporation.
Measured on run15 (ckpt day 238->408): ``d(storage)`` fell 13-19% short of ``d(E + flash - P)`` on
every checkpoint pair while ``max(q)`` sat pinned at 0.499911.

The fix is one number with one home: ``th.Q_MAX``, derived from the same ``E_FRAC_MAX`` clamp
``th.qsat`` already applies, so a state cap can never again sit below the saturation cap.
"""
import numpy as np
import pytest
from flatworld import thermo as th
from flatworld.stage_ad_core import StageADCore, StageADConfig


def _core(**kw):
    kw.setdefault("nx", 8)
    kw.setdefault("nz", 6)
    kw.setdefault("max_iter", 1)
    kw.setdefault("checkpoint_every", 0)
    kw.setdefault("tol", 0.0)
    return StageADCore(StageADConfig(**kw))


def _cell_weight(m, i):
    return float(m.dm[-1]) * (m.dx if m._dx_cell is None else float(m._dx_cell[i]))


# ------------------------------------------------------------------ the ceiling is the physical one
def test_q_max_is_the_qsat_clamp_and_pressure_independent():
    """Q_MAX must BE the q at qsat's own clamp -- not a re-typed literal, and not a profile: with
    e = f*p the pressure cancels, so the ceiling is a constant at every level."""
    for p in (1.0e3, 5.0e4, 1.01325e5, 3.0e5):
        # at/above boiling qsat saturates at exactly the clamp value, whatever the pressure
        q_hot = float(th.qsat(np.array([900.0]), np.array([p]))[0])
        assert q_hot == pytest.approx(th.Q_MAX, rel=1e-12), f"clamp not p-independent at p={p}"


def test_qsat_never_exceeds_q_max():
    """The invariant the whole fix rests on: a state cap at Q_MAX can never bind below saturation,
    so condensation is always free to fire before the clip does."""
    T = np.linspace(150.0, 900.0, 5001)
    for p in (5.0e3, 1.01325e5):
        assert float(th.qsat(T, np.full_like(T, p)).max()) <= th.Q_MAX


def test_q_max_is_steam_not_earth():
    """Regression on the actual bug: the old 0.5 bound below saturation on a boiling shoreline."""
    qs_shoreline = float(th.qsat(np.array([92.6 + 273.15]), np.array([1.01325e5]))[0])
    assert qs_shoreline > 0.5, "the run15 shoreline qsat must exceed the retired 0.5 cap"
    assert qs_shoreline < th.Q_MAX, "...and must sit BELOW the physical ceiling, so the clip cannot bite"


# ------------------------------------------------------------------ the clip is accounted, exactly
def test_ceiling_clip_mass_is_counted_exactly():
    m = _core()
    m.reset_water_budget()
    excess = 0.25
    m.q[3, -1] = th.Q_MAX + excess
    expect = excess * _cell_weight(m, 3)
    q = m._clip_q(m.q)
    assert m._cum_qcap == expect, "the deleted mass must be counted to the bit"
    assert q[3, -1] == th.Q_MAX
    assert m._cum_qfloor == 0.0, "a ceiling event must not touch the floor counter"


def test_floor_clip_mass_is_counted_separately():
    """Signed counters must not cancel: a ceiling deletion and a floor invention are different bugs."""
    m = _core()
    m.reset_water_budget()
    m.q[5, -1] = -0.1
    expect = 0.1 * _cell_weight(m, 5)
    m._clip_q(m.q)
    assert m._cum_qfloor == expect
    assert m._cum_qcap == 0.0


def test_clip_is_inert_when_within_bounds():
    m = _core()
    m.reset_water_budget()
    before = m.q.copy()
    q = m._clip_q(m.q)
    assert np.array_equal(q, before)
    assert m._cum_qcap == 0.0 and m._cum_qfloor == 0.0


# ------------------------------------------------------------------ the budget closes (THE gate)
def test_atmospheric_budget_closes_to_round_off():
    """THE conservation gate (#294-A). closure_cum (P/(E+flash) -> 1) is a closed-domain STEADY-STATE
    identity -- it cannot distinguish a leak from unfinished spin-up, which is exactly how the q<=0.5
    sink hid for 400+ run15 days. atm_residual holds transient or not, so it is what we gate on.
    Relative, not absolute: it is a difference of large totals (the fp64 cancellation floor)."""
    m = _core(nx=24, nz=18, max_iter=60)
    m.reset_water_budget()
    for _ in range(60):
        m.step()
    wc = m.water_conservation()
    store = wc["vapor_mass"] + wc["cond_mass"] + wc["rain_mass"]
    scale = max(abs(store), abs(wc["cum_evap"]), abs(wc["cum_precip"]), 1e-30)
    assert abs(wc["atm_residual"]) / scale < 1e-12, (
        f"atmospheric water budget does not close: residual {wc['atm_residual']:.6e} "
        f"(store_change {wc['atm_store_change']:.6e} vs supply {wc['atm_supply']:.6e})")


@pytest.mark.parametrize("scheme,extra", [
    ("explicit", {}),
    ("vertical_implicit", {}),
    ("vertical_implicit_strang_e", {"vertical_implicit_diffusion": True}),
])
def test_atmospheric_budget_closes_on_every_advection_path(scheme, extra):
    """PR #360 review finding. The budget must close on the IMPLICIT paths too -- and they are the ones
    that matter: `canonical_env.sh`'s world and run15 both run implicit advection (`strang_e`), while
    every other closure test here runs the explicit default.

    The bug this locks down: `advection_scheme != "explicit"` floors each tracer with `max(field, 0)`,
    which CREATES water wherever advection round-off went negative. That is the same kind of event as
    `_clip_q`'s floor but at a different call site, and it was missing from the ledger -- so the residual
    silently absorbed it: +9.51 kg/m (rel 7.5e-9) on `vertical_implicit`, enough to fail analyze_run's
    `< 1e-9` gate for a reason unrelated to any real leak. Note the tell: on the explicit path that same
    9.51 shows up as `cum_qfloor` (the thermo clip catches the negatives); on the implicit path the
    advection clamp floors them FIRST, so `cum_qfloor` reads 0 and the mass lands nowhere.

    (`_clamp_mass` could not have caught it either: it sums UNWEIGHTED mixing ratios, ~1e9x smaller than
    the mass integral, on a read+reset window unaligned with the budget's.)"""
    m = _core(nx=24, nz=18, max_iter=60, i0_scale=2.0, prognostic_condensate=True,
              advection_scheme=scheme, **extra)
    m.reset_water_budget()
    for _ in range(60):
        m.step()
    wc = m.water_conservation()
    store = wc["vapor_mass"] + wc["cond_mass"] + wc["rain_mass"]
    scale = max(abs(store), abs(wc["cum_evap"]), abs(wc["cum_precip"]), 1e-30)
    assert abs(wc["atm_residual"]) / scale < 1e-12, (
        f"budget does not close on advection_scheme={scheme!r}: residual {wc['atm_residual']:.6e} "
        f"(cum_qfloor {wc['cum_qfloor']:.6g}, cum_qadv {wc['cum_qadv']:.6g})")
    # the clamp term must be ZERO on the explicit path (monotone advection never clamps) and carry the
    # injection on the implicit ones -- i.e. the term is real, not a fudge that happens to absorb error
    if scheme == "explicit":
        assert wc["cum_qadv"] == 0.0
    else:
        assert wc["cum_qadv"] > 0.0


def test_atm_residual_exposes_an_unaccounted_sink():
    """The gate must FAIL when water vanishes -- otherwise it is decoration. Delete vapour behind the
    accounting's back and confirm the residual reports exactly what was taken (this is the signature
    the real leak had; the pre-fix budget was blind to it because the diagnostics clipped too)."""
    m = _core(nx=24, nz=18, max_iter=20)
    m.reset_water_budget()
    for _ in range(20):
        m.step()
    stolen = float(m._column_mass(0.5 * m.q))
    m.q *= 0.5                                   # a silent sink, uncounted by construction
    wc = m.water_conservation()
    assert wc["atm_residual"] == pytest.approx(-stolen, rel=1e-9), (
        "atm_residual must equal the unaccounted loss")


def test_reset_repins_the_atmospheric_baseline():
    """Without the re-pin the pre-reset inventory reads as a permanent false leak."""
    m = _core(nx=16, nz=12, max_iter=10)
    for _ in range(10):
        m.step()
    m.reset_water_budget()
    wc = m.water_conservation()
    assert wc["atm_store_change"] == pytest.approx(0.0, abs=1e-6)
    assert wc["atm_residual"] == pytest.approx(0.0, abs=1e-6)


def test_budget_survives_a_checkpoint_round_trip():
    """atm_residual must be resume-proof (the surf twin: W_total_reset). A run that checkpoints mid-window
    and resumes must not manufacture a leak out of the round trip."""
    m = _core(nx=16, nz=12, max_iter=20)
    m.reset_water_budget()
    for _ in range(20):
        m.step()
    st = m.state()
    before = m.water_conservation()
    m2 = _core(nx=16, nz=12, max_iter=20)
    m2.load_state({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()})
    after = m2.water_conservation()
    for k in ("cum_qcap", "cum_qfloor", "atm_store_change", "atm_supply", "atm_residual"):
        assert after[k] == pytest.approx(before[k], rel=1e-12, abs=1e-6), f"{k} did not survive the round trip"


def test_pre_fix_checkpoint_at_window_start_gets_an_exact_baseline():
    """A pre-#294-A checkpoint carries no atm_total_reset. When its cums are zero the window starts at
    the resume, so the baseline IS the loaded storage -- exact, and immediately gateable."""
    m = _core(nx=16, nz=12, max_iter=10)
    for _ in range(10):
        m.step()
    m.reset_water_budget()                       # cums zeroed -> an unambiguous window start
    st = m.state()
    for k in ("atm_total_reset", "cum_qcap", "cum_qfloor"):
        st.pop(k, None)                          # emulate the pre-#294-A checkpoint format
    m2 = _core(nx=16, nz=12, max_iter=10)
    m2.load_state(st)
    wc = m2.water_conservation()
    assert wc["atm_store_change"] == pytest.approx(0.0, abs=1e-6)
    assert wc["atm_residual"] == pytest.approx(0.0, abs=1e-6)
    assert wc["cum_qcap"] == 0.0 and wc["cum_qfloor"] == 0.0


def test_pre_fix_midwindow_resume_reports_nan_not_a_false_leak():
    """THE regression for a bug this fix nearly shipped. A pre-#294-A checkpoint resumed MID-window
    carries large cums but no baseline. Re-pinning the baseline to the current storage pairs a zero
    store-change with those cums and manufactures a ~1e-2 relative FALSE LEAK -- which would trip the
    conservation gate on exactly the run15 resume this work exists to serve. The window is genuinely
    unreconstructable (the pre-fix run never recorded its start storage), so the honest answer is NaN:
    it fails a `< tol` gate loudly rather than passing a fabricated zero."""
    m = _core(nx=16, nz=12, max_iter=10)
    m.reset_water_budget()
    for _ in range(10):
        m.step()
    st = m.state()
    assert st["cum_evap"] > 0.0, "fixture must be mid-window for this test to mean anything"
    for k in ("atm_total_reset", "cum_qcap", "cum_qfloor"):
        st.pop(k, None)
    m2 = _core(nx=16, nz=12, max_iter=10)
    m2.load_state(st)
    wc = m2.water_conservation()
    assert np.isnan(wc["atm_residual"]), "an unknowable baseline must report NaN, not a false leak"
    assert np.isnan(wc["atm_store_change"])
    # ...and a reset makes it gateable again
    m2.reset_water_budget()
    assert m2.water_conservation()["atm_residual"] == pytest.approx(0.0, abs=1e-6)
