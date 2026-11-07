"""Surface flash-boil (``surf_flash_boil`` / COSMO_SURF_FLASH_BOIL, default OFF).

The surface-side completion of the #294 boiling-layer physics (OPEN_ISSUES spin-off (B)): rain
formed in SUB-boiling air falls past both #294 flags (they key on boiling AIR layers) yet can land
on a skin far above ITS boiling point (the equatorial salt flat runs 400-650 C). ON: at the
surface-exchange step, where T_surf > T_boil(P_SURF), the landed flux flash-evaporates on contact
up to the surface ENERGY cap

    surf_cap = (T_eff - T_boil(P_SURF)) * C_surf / (L_VAP * dt)     [kg/m^2/s]

(T_eff = the post-SEB skin; C_surf = the SEB's own surface heat capacity): the flashed mass becomes
vapour in the lowest layer, its latent heat is debited from the surface energy budget (through the
same carried ``seb_incr`` as the rest of the SEB), and flux beyond the cap lands as liquid normally
-- the #296 max()/continuity lesson at the surface (the cap -> 0 continuously at the mask edge; a
capped flash lands the skin exactly AT T_boil, never below -- an fp64-leg invariant, which is what
these tests run; under thermo_fp32 a cap-limited skin can land ~1 fp32 ULP below, no consequence).

The operator lives inside the surface block of ``_thermo_substep_impl``, so the tests drive whole
thermo substeps on CONSTRUCTED states built to be inert everywhere else (isothermal = convectively
stable, deeply sub-saturated = no condensation): a one-substep ON-vs-OFF diff then isolates the
operator's writes exactly (T_surf, lowest-layer q, ``_surf_flash``) and the closures are asserted
to machine precision.
"""
import numpy as np
import pytest

from flatworld import constants as C
from flatworld import thermo as th
from flatworld.stage_ad_core import StageADConfig, StageADCore

DT = 60.0


def _core(**kw):
    """A minimal prognostic core for direct ``_thermo_substep`` calls (no dynamics stepping)."""
    kw.setdefault("nx", 6)
    kw.setdefault("nz", 12)
    kw.setdefault("theta_max_deg", 70.0)
    kw.setdefault("prognostic_condensate", True)
    kw.setdefault("precip_reevap", True)
    return StageADCore(StageADConfig(**kw))


def _inert_hot_skin_state(m, T_skin, precip, T_air=320.0, q0=1e-4):
    """Isothermal (convectively stable), deeply sub-saturated column under a prescribed skin, with a
    preset lagged surface-reaching flux ``self._precip`` -- the only mobile physics in a substep is
    then radiation/surface exchange (identical ON vs OFF) plus the flash operator itself."""
    m.T_atm[...] = T_air
    m.q[...] = q0
    m.q_c[...] = 0.0
    m.T_surf[...] = np.asarray(T_skin, float)
    m._precip = np.asarray(np.broadcast_to(np.asarray(precip, float), (m.cfg.nx,)), float).copy()


def test_flag_default_off_and_env_plumbing(monkeypatch):
    assert StageADConfig().surf_flash_boil is False
    monkeypatch.setenv("COSMO_SURF_FLASH_BOIL", "1")
    assert StageADConfig().surf_flash_boil is True


# ------------------------------------------------------- the operator, isolated: exact closures
def test_flash_water_energy_closure_machine_precision():
    """One substep, super-boiling skins (620 K), flux-limited regime (precip << cap). The ON-vs-OFF
    diff is EXACTLY the operator: the skin's energy debit Csurf*dTs == -L_VAP*flash*dt, the
    lowest-layer vapour gain dm*dq == +flash*dt (water/energy conjugate, machine precision), the
    flash equals min(precip, cap), and NOTHING else moves."""
    P = 200.0 / 86400.0                                    # ~the live theta0-scale landing flux
    out = {}
    for flag in (False, True):
        m = _core(surf_flash_boil=flag)
        _inert_hot_skin_state(m, 620.0, P)
        Csurf = np.asarray(m._surf_heat_capacity()).copy()  # what the SEB will use (T_surf unchanged pre-SEB)
        m._thermo_substep(DT)
        out[flag] = (m.T_surf.copy(), m.q.copy(), m.T_atm.copy(), m.q_c.copy(),
                     np.asarray(m._surf_flash).copy(), Csurf)
    Ts_off, q_off = out[False][0], out[False][1]
    Ts_on, q_on, T_atm_on, qc_on, flash, Csurf = out[True]
    # the expected flash from the OFF run's own post-SEB skin (identical arithmetic to the operator)
    T_eff = Ts_off                                          # OFF final skin == T_surf + seb_incr, bit-for-bit
    cap = np.maximum(T_eff - th.T_boil(C.P_SURF), 0.0) * Csurf / (C.L_VAP * DT)
    m2 = _core()
    P_arr = np.full(m2.cfg.nx, P)
    assert np.all(cap > 0)                                  # premise: every skin still super-boiling post-SEB
    assert np.all(P_arr < cap)                              # premise: flux-limited regime
    assert flash == pytest.approx(P_arr, rel=1e-12)         # the FULL flux flashes
    # energy: the skin paid exactly the latent heat of the flashed mass
    assert Csurf * (Ts_off - Ts_on) == pytest.approx(C.L_VAP * flash * DT, rel=1e-12)
    # water: the flashed mass arrived as lowest-layer vapour
    assert (q_on - q_off)[:, -1] * np.asarray(m2.dm)[-1] == pytest.approx(flash * DT, rel=1e-12)
    # ...and nothing else moved: the operator touches ONLY T_surf and q[:, -1]
    assert np.array_equal(q_on[:, :-1], q_off[:, :-1])
    assert np.array_equal(T_atm_on, out[False][2])
    assert np.array_equal(qc_on, out[False][3])


def test_cap_saturation_lands_skin_exactly_at_T_boil():
    """Flux >= cap: the flash saturates the ENERGY cap -- the skin lands exactly AT its boiling
    point (never below), ``_surf_flash`` == cap, and the un-flashable remainder survives as landed
    liquid (excluded from the flash diagnostic)."""
    m_off = _core()
    _inert_hot_skin_state(m_off, 400.0, 0.0)
    Csurf = np.asarray(m_off._surf_heat_capacity()).copy()
    m_off._thermo_substep(DT)
    Tb = float(th.T_boil(C.P_SURF))
    cap = np.maximum(np.asarray(m_off.T_surf) - Tb, 0.0) * Csurf / (C.L_VAP * DT)   # post-SEB cap
    assert np.all(cap > 0)
    P = 2.0 * cap                                           # twice what the skin can flash
    m = _core(surf_flash_boil=True)
    _inert_hot_skin_state(m, 400.0, P)
    m._thermo_substep(DT)
    flash = np.asarray(m._surf_flash)
    assert flash == pytest.approx(cap, rel=1e-12)           # the flash saturates the energy cap...
    assert np.allclose(np.asarray(m.T_surf), Tb, atol=1e-9)  # ...landing the skin exactly AT T_boil
    assert np.all(np.asarray(m.T_surf) >= Tb - 1e-9)        # and NEVER below it
    # the remainder P - cap is the surviving landed liquid: _precip keeps its (pre-flash) meaning,
    # so the surface-reaching LIQUID diagnostic is _precip - _surf_flash = P - cap by construction.


def test_flash_continuous_at_the_mask_edge():
    """Skins straddling T_boil under the same rain: the flash is min(precip, cap) with
    cap -> 0 continuously as the (post-SEB) skin approaches boiling from above -- no jump at the
    mask edge (sub-boiling skins flash nothing; barely-super-boiling skins flash only the sliver
    their superheat can pay for), so the closure cannot pin a dead state (the #296 lesson)."""
    Tb = float(th.T_boil(C.P_SURF))
    skins = np.array([Tb - 5.0, Tb - 0.5, Tb + 0.5, Tb + 1.5, Tb + 3.0, Tb + 8.0])
    P = 100.0 / 86400.0
    m_off = _core()
    _inert_hot_skin_state(m_off, skins, P)
    Csurf = np.asarray(m_off._surf_heat_capacity()).copy()
    m_off._thermo_substep(DT)
    T_eff = np.asarray(m_off.T_surf)                        # the post-SEB skin the operator sees
    m = _core(surf_flash_boil=True)
    _inert_hot_skin_state(m, skins, P)
    m._thermo_substep(DT)
    flash = np.asarray(m._surf_flash)
    cap = np.maximum(T_eff - Tb, 0.0) * Csurf / (C.L_VAP * DT)
    assert flash == pytest.approx(np.minimum(P, cap), abs=1e-18)   # exactly min(precip, cap)...
    assert np.all(flash[T_eff <= Tb] == 0.0)                # ...zero on/below the edge
    assert np.all(flash <= cap + 1e-18)                     # ...and bounded by the vanishing cap above it
    # the FLASH never pulls a skin below T_boil (skins the ordinary SEB leaves sub-boiling stay
    # exactly where the OFF closure puts them -- flash = 0 there, T_final = T_eff)
    assert np.all(np.asarray(m.T_surf) >= np.minimum(T_eff, Tb) - 1e-9)
    assert np.array_equal(np.asarray(m.T_surf)[T_eff <= Tb], T_eff[T_eff <= Tb])


def test_no_precip_and_sub_boiling_are_byte_identical():
    """Two locality gates, one substep each: (a) super-boiling skins but NO landed flux -> ON is
    byte-identical to OFF (min(0, cap) = 0, debit exactly zero); (b) rain onto hot-but-SUB-boiling
    skins (the 60-95 C air-over-hot-ground regime this flag must NOT touch) -> ON byte-identical."""
    for T_skin, P in ((620.0, 0.0), (360.0, 200.0 / 86400.0)):
        out = {}
        for flag in (False, True):
            m = _core(surf_flash_boil=flag)
            _inert_hot_skin_state(m, T_skin, P)
            m._thermo_substep(DT)
            out[flag] = (m.T_surf.copy(), m.q.copy(), m.T_atm.copy(), m.q_c.copy(),
                         np.asarray(m._precip).copy())
        for a, b in zip(out[False], out[True]):
            assert np.array_equal(a, b)


def test_full_coupled_steps_off_vs_on_sub_boiling_world():
    """Byte-identity at the full ``step()`` level on a world with NO super-boiling skin (dimmed sun,
    budget wetness + endorheic branches ON so the _wetness supply path is genuinely crossed): the
    flag ON changes NOTHING anywhere below the mask. (The base-vs-branch OFF gate is the npz
    comparison in SURF_FLASH_BOIL_RESULTS.md; this pins ON == OFF below the mask forever.)"""
    out = {}
    for flag in (False, True):
        m = StageADCore(StageADConfig(nx=24, nz=12, theta_max_deg=75.0, i0_scale=0.85,
                                      prognostic_condensate=True, precip_reevap=True,
                                      budget_wet_fraction=True, budget_endorheic=True,
                                      surf_flash_boil=flag))
        for _ in range(6):
            m.step()
        assert float(np.max(np.asarray(m.T_surf))) < th.T_boil(C.P_SURF) - 1.0   # premise: nothing boils
        out[flag] = (m.T_surf.copy(), m.q.copy(), m.T_atm.copy(), m.q_c.copy(),
                     np.asarray(m._precip).copy(), np.asarray(m._wetness()).copy(),
                     m.water_conservation()["closure_cum"])
    for a, b in zip(out[False], out[True]):
        assert np.array_equal(np.asarray(a), np.asarray(b))


# ------------------------------------------------------------------- bookkeeping and accounting
def test_flashed_mass_excluded_from_wetness_supply():
    """The budget-wetness supply must see only the POST-flash liquid: a flag-ON core with
    (_precip=P, _surf_flash=F) computes the same wetness as a flag-OFF core supplied P-F directly
    -- on both the per-cell budget_f branch and the endorheic-pool branch."""
    for endo in (False, True):
        cores = {}
        for flag in (False, True):
            m = _core(nx=8, surf_flash_boil=flag, budget_wet_fraction=True, budget_endorheic=endo)
            _inert_hot_skin_state(m, 330.0, 0.0)
            m._thermo_substep(DT)                           # identical spin (no boiling): builds _last_rad
            cores[flag] = m
        P = np.linspace(0.0, 8e-3, 8)
        F = 0.6 * P
        cores[True]._precip = P.copy()
        cores[True]._surf_flash[...] = F
        cores[False]._precip = (P - F).copy()               # hand the OFF core the post-flash supply directly
        w_on = np.asarray(cores[True]._wetness())
        w_off = np.asarray(cores[False]._wetness())
        assert np.allclose(w_on, w_off, rtol=0, atol=0), f"endorheic={endo}"


def test_cum_surf_flash_accounting_and_closure_denominator():
    """``cum_surf_flash`` accumulates exactly sum(flash * dx * dt) over substeps; the
    water_conservation closure counts the flash as a second surface->atmosphere source (at steady
    state cum_precip == cum_evap + cum_surf_flash), and the state()/load_state() round-trip
    persists it (a pre-flag checkpoint loads to 0.0)."""
    m = _core(surf_flash_boil=True)
    _inert_hot_skin_state(m, 620.0, 150.0 / 86400.0)
    expect = 0.0
    for _ in range(3):
        m._precip = np.full(m.cfg.nx, 150.0 / 86400.0)      # re-arm the lagged flux each substep
        m._thermo_substep(DT)
        expect += float(np.sum(np.asarray(m._surf_flash) * m.dx)) * DT
    wc = m.water_conservation()
    assert wc["cum_surf_flash"] == pytest.approx(expect, rel=1e-12)
    assert wc["cum_surf_flash"] > 0.0
    assert wc["closure_cum"] == pytest.approx(
        wc["cum_precip"] / (wc["cum_evap"] + wc["cum_surf_flash"] + 1e-30), rel=1e-12)
    # checkpoint round-trip + pre-flag back-compat
    st = m.state()
    m2 = _core(surf_flash_boil=True)
    m2.load_state(st)
    assert m2._cum_surf_flash == m._cum_surf_flash
    st.pop("cum_surf_flash")
    m2.load_state(st)
    assert m2._cum_surf_flash == 0.0


# ------------------------------------------------- the #297 coupling: FIXED by #294-A (was pinned here)
def test_297_steam_lowest_layer_flashed_vapour_survives_past_the_old_clip():
    """#297 REGRESSION (the caveat this test used to PIN is now FIXED -- #294-A).

    History: ``_thermo_substep_impl`` clipped q <= 0.5 every substep while a steam layer's clamp qsat
    is ~0.998, so flashed vapour that lifted the lowest layer past 0.5 was silently destroyed one
    substep later. This test asserted that destruction (q landing at exactly 0.5) as a bounded, known
    caveat. It was not a small one: on run15 that clip deleted ~15% of ALL evaporation, invisibly,
    because the budget diagnostics applied the same clip.

    The cap is now the PHYSICAL ceiling ``th.Q_MAX`` (~0.998, the q at qsat's own clamp), so vapour
    between the old 0.5 and saturation SURVIVES -- which is the whole point: below Q_MAX the physics
    (condensation, or boil-suppressed pass-through) decides the fate, never a guard constant."""
    P = 0.03                                                # heavy flux -> dq ~ 2e-3 in the lowest layer
    m = _core(surf_flash_boil=True)
    _inert_hot_skin_state(m, 650.0, P, T_air=500.0, q0=0.4990)   # steam column: q just under the OLD clip
    dm_low = float(np.asarray(m.dm)[-1])
    m.reset_water_budget()
    m._thermo_substep(DT)
    flash_mass = float(np.asarray(m._surf_flash)[0]) * DT   # [kg/m^2] added to the lowest layer
    assert flash_mass == pytest.approx(P * DT, rel=1e-12)   # cap is huge at 650 K: full flux flashed
    q_low = float(m.q[0, -1])
    assert q_low > 0.5                                      # the flash pushed the layer past the OLD clip
    assert q_low < th.Q_MAX                                 # ...but nowhere near the physical ceiling
    # THE FIX: with no new rain armed, another substep must NOT erase the excess -- the vapour is still
    # there (the old code left exactly 0.5 here), and nothing was clipped away.
    m._precip[...] = 0.0
    m._thermo_substep(DT)
    assert float(m.q[0, -1]) == pytest.approx(q_low, rel=1e-9), "flashed steam must survive below Q_MAX"
    assert m._cum_qcap == 0.0, "nothing may be clipped while the column is below the physical ceiling"
    # headroom control: the same flash into a non-steam layer survives in full (unchanged by the fix)
    m2 = _core(surf_flash_boil=True)
    _inert_hot_skin_state(m2, 650.0, P, T_air=500.0, q0=0.40)
    m2._thermo_substep(DT)
    assert float(m2.q[0, -1]) == pytest.approx(0.40 + P * DT / dm_low, rel=1e-9)


def test_297_clip_at_the_physical_ceiling_is_accounted_not_silent():
    """The ceiling can STILL bite -- but only in a genuinely boiling column, and only visibly.

    What makes it live is ``cond_boil_suppress`` (#294, ON in run15/canonical): it deliberately removes
    the condensation sink above boiling so steam advects through, which also removes the only thing that
    would otherwise hold q at qsat. Then evaporation/flash can drive q past the physical ceiling and the
    clip is the last bound. With boil-suppress OFF the ceiling is unreachable by construction --
    condensation pins q at the clamped qsat, which IS Q_MAX (the control below). That is exactly
    OPEN_ISSUES' "pre-existing, but the #294 steam-as-vapour pass-through makes it live".

    Clipping there is defensible; SILENCE was the bug. The removed mass now lands in ``cum_qcap`` and
    in ``atm_residual``'s ledger instead of vanishing."""
    P = 0.03
    m = _core(surf_flash_boil=True, cond_boil_suppress=True)
    _inert_hot_skin_state(m, 650.0, P, T_air=500.0, q0=th.Q_MAX - 1e-5)   # steam, pinned at the ceiling
    m.reset_water_budget()
    for _ in range(3):
        m._thermo_substep(DT)
    assert m._cum_qcap > 0.0, "a clip at the physical ceiling must be counted, never silent"
    assert float(m.q[0, -1]) <= th.Q_MAX + 1e-12, "and it must still bound the state"
    # control: with boil-suppress OFF, condensation holds q at the clamped qsat == Q_MAX, so the ceiling
    # is never reached and nothing is clipped -- the clip is a backstop, not a routine sink.
    m2 = _core(surf_flash_boil=True, cond_boil_suppress=False)
    _inert_hot_skin_state(m2, 650.0, P, T_air=500.0, q0=th.Q_MAX - 1e-5)
    m2.reset_water_budget()
    for _ in range(3):
        m2._thermo_substep(DT)
    assert m2._cum_qcap == 0.0


# ------------------------------------------------------------------ capture-ready CPU leg
def test_capture_ready_cpu_leg_bitwise_and_pending_budget():
    """``capture_ready=True`` on CPU (review test-gap): exercises the capture-branch in-place stores
    the default path rebinds -- the SEB apply (``xp.add(..., out=T_surf)``) and the DEFERRED
    pending-budget accumulation (``_cum_surf_flash_pending``, folded only at the flush) -- with the
    flag ON. The physics must match the default path bitwise (the in-place forms store the same
    bits), the flash budget must sit in the PENDING scalar before the flush and land in
    ``cum_surf_flash`` after it, equal to the default leg's per-step accumulation."""
    P = 150.0 / 86400.0
    out = {}
    for cap in (False, True):
        m = _core(surf_flash_boil=True, capture_ready=cap)
        _inert_hot_skin_state(m, 620.0, P)
        for _ in range(3):
            m._precip = np.full(m.cfg.nx, P)                # re-arm the lagged flux each substep
            m._thermo_substep(DT)
        out[cap] = m
    m_def, m_cap = out[False], out[True]
    # the capture leg genuinely deferred the budget: pending holds it all until the flush
    assert float(m_cap._cum_surf_flash) == 0.0
    assert float(m_cap._cum_surf_flash_pending) > 0.0
    wc_def, wc_cap = m_def.water_conservation(), m_cap.water_conservation()   # wc flushes the pending
    assert float(m_cap._cum_surf_flash_pending) == 0.0     # folded at the flush
    assert wc_def["cum_surf_flash"] > 0.0                  # non-vacuous: the flash fired
    assert wc_cap["cum_surf_flash"] == wc_def["cum_surf_flash"]   # same terms, same order -> same bits
    for a, b in ((m_def.T_surf, m_cap.T_surf), (m_def.q, m_cap.q), (m_def.T_atm, m_cap.T_atm),
                 (np.asarray(m_def._surf_flash), np.asarray(m_cap._surf_flash))):
        assert np.array_equal(np.asarray(a), np.asarray(b))


# --------------------------------------------------------------------------- fp32 carried path
def test_flash_debit_rides_the_fp32_seb_carry():
    """Under ``thermo_fp32`` the SEB increment is carried in fp64 (the freeze fix); the flash debit
    is folded into that SAME increment, never a bypass -- so the ON-vs-OFF skin difference matches
    the latent debit at fp32 tolerance and the vapour arrives in the reconciled q."""
    P = 200.0 / 86400.0
    out = {}
    for flag in (False, True):
        m = _core(surf_flash_boil=flag, thermo_fp32=True)
        _inert_hot_skin_state(m, 620.0, P)
        Csurf = np.asarray(m._surf_heat_capacity()).copy()
        m._thermo_substep(DT)
        out[flag] = (np.asarray(m.T_surf, float).copy(), np.asarray(m.q, float).copy(), Csurf)
    Ts_off, q_off, Csurf = out[False]
    Ts_on, q_on, _ = out[True]
    flash = np.asarray(_core()._precip * 0 + P)             # full flux flashes (cap huge at 620 K)
    assert Csurf * (Ts_off - Ts_on) == pytest.approx(C.L_VAP * flash * DT, rel=1e-5)   # fp32-granular
    assert (q_on - q_off)[:, -1] * float(np.asarray(_core().dm)[-1]) == pytest.approx(
        flash * DT, rel=1e-5)
