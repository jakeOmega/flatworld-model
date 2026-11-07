"""Chunk-1 acceptance tests for the PROGNOSTIC cloud condensate (CLOUD_PROGNOSTIC_ROADMAP).

The walking skeleton: a minimal, complete, conserving end-to-end prognostic cloud loop -- a retained
condensate reservoir ``q_c`` that is advected (conserving), sourced (supersaturation), sunk (a crude
autoconversion rate), radiated (reusing the #19 optics with an in-cloud LWP = rho*q_c/f*dz and a cover
floor), and checkpointed. Default OFF -> byte-identical to the single-pass core. These tests are fast
(low resolution, no slow spin-up; the analytic single cell drives the advection checks).
"""
import numpy as np
import pytest
from flatworld import constants as C
from flatworld import thermo as th
from flatworld import radiation as rad
from flatworld import bl_cloud as blc
from flatworld.stage_ad_core import StageADCore, StageADConfig


def _analytic_single_cell(m, amp=2.0e4):
    """Impose an analytic single-overturning-cell streamfunction (zero on all four walls), freeze the
    layer heights, and return the resolved cross-track wind ``u`` (height grid). A fast, deterministic
    divergence-free flow for the mass-consistent transport tests (mirrors test_stage_ad_core)."""
    m._z = m._hydrostatic_z(m.T_atm)
    X = m.x[None, :] / m.x.max()
    Z = m.z[:, None] / m.z.max()
    m.psi = amp * np.sin(np.pi * X) * np.sin(np.pi * Z)
    u, w = m.velocities()
    return u, w


# ------------------------------------------------------------------ default-OFF regression guard
def _skeleton_kw():
    """The Chunk-2/3 sub-flags reduced to the Chunk-1 SKELETON (bare adjustment, no re-evaporation, warm-phase,
    no retained ceiling, Kessler, grid-mean source, rainout cap) -- the A/B baseline the real microphysics is
    measured against, and (with instant rain) the byte-identical reduction to the OFF single-pass core."""
    return dict(qc_dqsat_adjust=False, qc_reevaporation=False, qc_ice_phase=False,
                qc_retained_ceiling=False, qc_autoconv_scheme="kessler",
                qc_pdf_cloud=False, qc_radiative_cap_decouple=False)


def _chunk2_kw():
    """The Chunk-3 sub-flags turned OFF -> the ON path is the Chunk-2 grid-mean saturation adjustment with the
    floored RH cover and the adiabatic-rainout cap. The pre-Chunk-3 footing the Chunk-2 characterization tests
    (grid-mean dqsat landing, fast re-evaporation to the deficit, the steady deck, the rainout-cap ceiling)
    are written against -- their assertions stay valid on THIS footing; the PDF default is exercised by the
    Chunk-3 tests below."""
    return dict(qc_pdf_cloud=False, qc_radiative_cap_decouple=False)


def test_off_is_byte_identical_via_instant_rain_limit():
    """The load-bearing regression guard. With the Chunk-2 sub-flags reduced to the SKELETON, the prognostic
    machinery REDUCES EXACTLY to the OFF single-pass scheme in the instant-rain limit (tau -> 0, crit = 0):
    every bit of supersaturation sourced into q_c rains the same step, so q_c returns to 0 and the column
    thermo (precip, q, T) is BIT-IDENTICAL to the OFF single-pass core. (The Chunk-2 real microphysics -- the
    dqsat/dT correction etc. -- deliberately DEPARTS from this skeleton; that departure is characterized by
    the A/B test below, not here.) Clear-sky so the radiation matches too."""
    base = dict(nx=16, nz=16, theta_max_deg=55.0, clouds=False)
    off = StageADCore(StageADConfig(prognostic_condensate=False, **base))
    on = StageADCore(StageADConfig(prognostic_condensate=True, qc_autoconv_tau=1e-9,
                                   qc_autoconv_crit=0.0, **_skeleton_kw(), **base))
    for m in (off, on):
        m.run_dynamics = False                                   # pure column physics
    for _ in range(30):
        off.step(); on.step()
    assert np.array_equal(off.T_atm, on.T_atm)                   # bit-identical T
    assert np.array_equal(off.q, on.q)                           # bit-identical vapour
    assert np.array_equal(off.T_surf, on.T_surf)                 # bit-identical surface
    assert np.array_equal(off._precip, on._precip)               # bit-identical precip
    assert np.all(on.q_c == 0.0)                                 # all condensate rained instantly


def test_off_master_byte_identical_regardless_of_chunk2_subflags():
    """OFF (the production default, ``prognostic_condensate=False``) is byte-identical regardless of the
    Chunk-2 sub-flags -- the master knob gates the WHOLE scheme, so the sub-flags are inert when off and
    production never moves until the Chunk-7 opt-in."""
    base = dict(nx=16, nz=16, theta_max_deg=55.0)
    a = StageADCore(StageADConfig(prognostic_condensate=False, **base))
    b = StageADCore(StageADConfig(prognostic_condensate=False, qc_dqsat_adjust=True, qc_reevaporation=True,
                                  qc_ice_phase=True, qc_retained_ceiling=True,
                                  qc_autoconv_scheme="sundqvist",
                                  qc_pdf_cloud=True, qc_radiative_cap_decouple=True, **base))
    for _ in range(20):
        a.step(); b.step()
    assert np.array_equal(a.T_atm, b.T_atm)                      # sub-flags inert when the master is OFF
    assert np.array_equal(a.q, b.q)
    assert np.all(a.q_c == 0.0) and np.all(b.q_c == 0.0)


def test_off_leaves_qc_zero_and_flag_engages():
    """With prognostic_condensate OFF the new field never activates (q_c stays exactly 0 through a full
    coupled step); turning it ON changes the trajectory (the flag is live, not a silent no-op)."""
    base = dict(nx=16, nz=16, theta_max_deg=55.0)
    off = StageADCore(StageADConfig(prognostic_condensate=False, **base))
    on = StageADCore(StageADConfig(prognostic_condensate=True, **base))
    for _ in range(20):
        off.step(); on.step()
    assert np.all(off.q_c == 0.0)                                # OFF never sources/advects q_c
    assert np.any(on.q_c > 0.0)                                  # ON builds a condensate reservoir
    assert not np.array_equal(off.q, on.q)                       # the flag changes the moisture budget


# ------------------------------------------------------------------ conserving advection
def test_condensate_advection_conserves_mass_and_stays_positive():
    """Riding the SAME divergence-free mass flux as q/theta, a condensate blob is transported with
    INTEGRAL q_c*dm conserved to round-off and q_c >= 0 throughout (monotone upwind, no clamp)."""
    m = StageADCore(StageADConfig(nx=24, nz=18, theta_max_deg=55.0, kappa_qc=0.0,
                                  prognostic_condensate=True))
    u, _ = _analytic_single_cell(m)
    m._z = m._hydrostatic_z(m.T_atm)
    m.q_c[:] = 0.0
    i0 = m.cfg.nx // 2
    m.q_c[i0 - 1:i0 + 2, m.cfg.nz // 2] = 1.0e-3                 # a mid-domain condensate blob
    w0 = float(np.sum(m.q_c * m.dm[None, :]))
    for _ in range(50):
        m._advect_condensate(u, 600.0)
    w1 = float(np.sum(m.q_c * m.dm[None, :]))
    assert abs(w1 - w0) / w0 < 1e-12                             # condensate mass conserved to round-off
    assert np.all(m.q_c >= 0.0)                                  # monotone upwind keeps q_c >= 0


def test_condensate_advection_is_uniform_invariant():
    """A UNIFORM q_c stays uniform under the resolved-flow transport (the divergence-free flux + the
    conservative cross-diffusion both vanish on a flat field) -- the uniform-invariance the mass-consistent
    scheme guarantees, with q_c's own (default, nonzero) diffusivity."""
    m = StageADCore(StageADConfig(nx=24, nz=18, theta_max_deg=55.0, prognostic_condensate=True))
    u, _ = _analytic_single_cell(m)
    m.q_c[:] = 3.0e-4
    before = m.q_c.copy()
    m._advect_condensate(u, 600.0)
    assert np.allclose(m.q_c, before, rtol=1e-12)               # uniform q_c untouched


# ------------------------------------------------------------------ restart / checkpoint
def test_restart_roundtrips_condensate_old_checkpoint_loads_zero_and_nan_trips_guard():
    """q_c round-trips through state()/load_state(); an OLD checkpoint (no "q_c" key) loads to ZERO (the
    documented back-compat path); a NaN in q_c trips the finite guard."""
    cfg = dict(nx=16, nz=14, theta_max_deg=55.0, prognostic_condensate=True)
    m = StageADCore(StageADConfig(**cfg))
    m.q_c[:] = np.linspace(0.0, 1.0e-3, m.q_c.size).reshape(m.q_c.shape)
    st = m.state()
    assert "q_c" in st                                          # condensate is checkpointed
    m2 = StageADCore(StageADConfig(**cfg))
    m2.load_state(st)
    assert np.array_equal(m2.q_c, m.q_c)                        # round-trip carries q_c
    old = {k: v for k, v in st.items() if k != "q_c"}          # a pre-prognostic checkpoint
    m3 = StageADCore(StageADConfig(**cfg))
    m3.q_c[:] = 7.0                                             # poison; must be overwritten to 0
    m3.load_state(old)
    assert np.all(m3.q_c == 0.0)                                # old checkpoint -> empty reservoir
    m.q_c[0, 0] = np.nan
    assert not m._state_is_finite()                            # NaN in q_c aborts checkpointing


# ------------------------------------------------------------------ the re-derived reservoir water budget
def test_water_budget_reservoir_form_closes():
    """Total water q + q_c is the conserved tracer: d/dt INTEGRAL (q + q_c) dm = E - P (P now the
    autoconversion precip, not instantaneous supersaturation). Measured over a window on a no-flow column
    with a uniform moderate SST (bounded q, no clip artifact)."""
    m = StageADCore(StageADConfig(nx=20, nz=18, theta_max_deg=55.0, prognostic_condensate=True,
                                  qc_autoconv_crit=5.0e-4))
    m.run_dynamics = False
    m.couple_surface = False                                    # fixed SST -> bounded q (no clip source/sink)
    m.T_surf[:] = 292.0
    for _ in range(20):                                         # spin out the initial transient
        m.step()
    m.reset_water_budget()
    tw0 = float(np.sum((m.q + m.q_c) * m.dm[None, :]) * m.dx)   # area-weighted total water [kg/m]
    for _ in range(40):
        m.step()
    tw1 = float(np.sum((m.q + m.q_c) * m.dm[None, :]) * m.dx)
    wb = m.water_conservation()
    lhs = tw1 - tw0                                             # storage change
    rhs = wb["cum_evap"] - wb["cum_precip"]                     # E - P
    assert abs(lhs - rhs) / (abs(rhs) + 1e-12) < 1e-6           # reservoir budget closes


# ------------------------------------------------------------------ the headline new capability
def test_prognostic_deck_persists_where_single_pass_would_rain():
    """The capability the whole scheme exists for: a radiatively-cooled, no-ascent column RETAINS q_c
    across many steps (a persistent deck), where the single-pass core rains it all away. With a high
    autoconversion threshold the condensate never reaches the rain-out cut, so it accumulates and persists;
    the OFF core instead rains every step (its cumulative precip exceeds the ON core's ~0)."""
    base = dict(nx=12, nz=18, theta_max_deg=55.0, clouds=False)
    on = StageADCore(StageADConfig(prognostic_condensate=True, qc_autoconv_crit=1.0e-1, **base))
    off = StageADCore(StageADConfig(prognostic_condensate=False, **base))
    for m in (on, off):
        m.run_dynamics = False
        m.couple_surface = False
        m.T_surf[:] = 295.0
        m.q = th.qsat(m.T_atm, m.p_mid[None, :]).copy()        # start saturated -> cooling condenses
        m.reset_water_budget()
    for _ in range(60):
        on.step(); off.step()
    cond_path_on = float(np.sum(on.q_c * on.dm[None, :]))
    assert cond_path_on > 0.0                                   # ON retains a persistent condensate deck
    assert np.all(on.q_c >= 0.0)
    assert (off.water_conservation()["cum_precip"]              # OFF rained away what ON holds as cloud
            > on.water_conservation()["cum_precip"])


# ------------------------------------------------------------------ cover<->condensate consistency floor
def test_cover_floor_keeps_in_cloud_water_finite():
    """The cover<->condensate consistency floor (load-bearing): a DRY column (RH cover -> 0) carrying
    condensate q_c > 0 gets a minimal cover f >= f_min, so the in-cloud LWP = rho*q_c/f conversion stays
    FINITE (no /f blow-up) and every layer with q_c > thresh has f >= f_min."""
    m = StageADCore(StageADConfig(nx=8, nz=16, theta_max_deg=55.0, prognostic_condensate=True, clouds=True,
                                  **_chunk2_kw()))               # the Chunk-1 RH-cover floor (PDF cover is Chunk 3)
    m.q[:] = 1.0e-6                                             # very dry -> RH-diagnostic cover ~ 0
    m.q_c[:] = 1.0e-3                                           # but condensate present everywhere
    out = m._radiate(np.clip(m.q, 0.0, 0.5))
    assert np.all(np.isfinite(out["dtau_lw_cloud"]))           # in-cloud LW opacity finite (floor bit)
    assert np.all(np.isfinite(out["cloud_sw_albedo"]))         # SW albedo finite
    f = out["cloud_fraction"]
    floored = m.q_c > m.cfg.qc_cover_thresh
    assert np.all(f[floored] >= m.cfg.cloud_f_min - 1e-12)      # the floor bites where condensate is present


# ------------------------------------------------------------------ radiative-effect signs with q_c water
def test_prognostic_water_path_tolerates_subadiabatic_forward():
    """PR #142 review #4: ``cloud_optical_properties(qc=...)`` forwards ``**wp_kw`` into
    ``prognostic_cloud_water_path``; a caller passing ``subadiabatic=`` (valid for the adiabatic path) must
    NOT raise a TypeError on the prognostic path -- it is accepted and IGNORED (the reservoir is the actual
    condensate, no sub-adiabaticity applies), so the result is identical with or without it."""
    f = np.array([0.5, 0.5]); T = np.array([285.0, 250.0]); p = np.array([8.0e4, 4.0e4])
    dm = np.array([1.0e3, 1.0e3]); qc = np.array([2.0e-4, 1.0e-4])
    base = rad.cloud_optical_properties(f, T, p, dm, qc=qc)
    forwarded = rad.cloud_optical_properties(f, T, p, dm, qc=qc, subadiabatic=0.6)   # must not raise
    for a, b in zip(base, forwarded):
        assert np.array_equal(a, b)                            # subadiabatic ignored -> identical


def test_chunk1_radiative_cap_saturates_at_rainout_cap_when_not_decoupled():
    """CHARACTERIZATION (pre-Chunk-3 footing): with the DEFAULT caps (the adiabatic-rainout ceiling
    CLOUD_LWC_MAX, ``qc_radiative_cap_decouple`` OFF) the prognostic in-cloud water SATURATES -- once grid-mean
    q_c exceeds the cap onset the LWP stops tracking q_c (radiative strength is cap+cover-set). Chunk 3 flips
    this on the DECOUPLED footing (the test below); this pins the un-decoupled limit stays as it was."""
    f = np.array([0.05])                                       # floored cover
    T = np.array([285.0]); p = np.array([8.0e4]); dm = np.array([1.0e3])
    lwp_small, _ = rad.prognostic_cloud_water_path(np.array([1.0e-5]), f, T, p, dm)   # below the cap onset
    lwp_big, _ = rad.prognostic_cloud_water_path(np.array([1.0e-3]), f, T, p, dm)     # far above (q_c ~ crit)
    lwp_huge, _ = rad.prognostic_cloud_water_path(np.array([5.0e-3]), f, T, p, dm)    # 5x crit
    assert lwp_small[0] < lwp_big[0]                           # tracks q_c below the cap onset
    assert np.isclose(lwp_big[0], lwp_huge[0])                 # SATURATED above it (rainout cap, un-decoupled)


def test_chunk3_radiative_cap_decoupled_tracks_qc():
    """CHUNK 3 (the FLIP of the Chunk-1 cap-saturation limitation, PR #142 review #1): with the radiative cap
    DECOUPLED from the rainout ceiling (the physical-plausibility LWC, passed by the caller), the in-cloud LWP
    now TRACKS q_c across the range the rainout cap SATURATED -- the cloud radiative strength is a genuine
    function of the retained reservoir, bounded only by the (much higher) physical-plausibility ceiling. Same
    inputs as the un-decoupled test above (where lwp(1e-3) == lwp(5e-3), saturated)."""
    f = np.array([0.05]); T = np.array([285.0]); p = np.array([8.0e4]); dm = np.array([1.0e3])
    cap = dict(lwc_max=rad.CLOUD_LWC_PLAUSIBLE, iwc_max=rad.CLOUD_IWC_PLAUSIBLE)
    lwp_a, _ = rad.prognostic_cloud_water_path(np.array([2.0e-5]), f, T, p, dm, **cap)   # above the OLD onset
    lwp_b, _ = rad.prognostic_cloud_water_path(np.array([6.0e-5]), f, T, p, dm, **cap)   # the OLD cap saturated here
    lwp_c, _ = rad.prognostic_cloud_water_path(np.array([1.2e-4]), f, T, p, dm, **cap)
    assert lwp_a[0] < lwp_b[0] < lwp_c[0]                       # LWP now TRACKS q_c (no rainout-cap saturation)
    # and it remains BOUNDED at the physical-plausibility ceiling far above (stability preserved, not unbounded)
    lwp_hi, _ = rad.prognostic_cloud_water_path(np.array([1.0e-2]), f, T, p, dm, **cap)
    lwp_hi2, _ = rad.prognostic_cloud_water_path(np.array([5.0e-2]), f, T, p, dm, **cap)
    assert np.isclose(lwp_hi[0], lwp_hi2[0])                    # bounded at the plausibility cap (not the rainout cap)
    assert lwp_hi[0] > lwp_c[0]                                 # but that bound is far above the old rainout onset


def test_prognostic_cloud_radiative_signs():
    """With the cloud water sourced from the prognostic q_c (not the adiabatic reconstruction), the cloud
    radiative effect keeps the #19 signs: SW cooling (cloud raises planetary reflection) and LW warming
    (cloud lowers OLR)."""
    m = StageADCore(StageADConfig(nx=8, nz=16, theta_max_deg=55.0, prognostic_condensate=True, clouds=True))
    m.q[:] = th.qsat(m.T_atm, m.p_mid[None, :])                # saturated -> the PDF cover (Chunk-3 default) > 0
    m.q_c[:] = 1.0e-3
    q = np.clip(m.q, 0.0, 0.5)
    cloudy = m._radiate(q)                                      # clouds ON + prognostic q_c (PDF cover + decoupled cap)
    m.cfg.clouds = False
    clear = m._radiate(q)                                       # clear-sky on the same state
    assert np.all(cloudy["planetary_reflected"] >= clear["planetary_reflected"] - 1e-9)   # SW cools
    assert np.all(cloudy["OLR"] <= clear["OLR"] + 1e-9)                                    # LW warms


# ============================================================ CHUNK 2 -- real microphysics (A/B'd vs skeleton)
def test_dqsat_adjustment_leaves_air_at_saturation():
    """The dqsat/dT-corrected saturation adjustment (Chunk 2) leaves a supersaturated layer AT saturation
    after ONE step (no residual super/sub-saturation), where the BARE q-qsat adjustment OVER-condenses (the
    latent warming raises qsat, so the bare step leaves the air SUB-saturated against the new T). Tested on
    one isolated _prognostic_microphysics call (autoconversion off via a huge crit, re-evap inert on a
    supersaturated cell)."""
    m = StageADCore(StageADConfig(nx=8, nz=16, theta_max_deg=55.0, prognostic_condensate=True,
                                  qc_autoconv_crit=1.0e3, qc_retained_ceiling=False, **_chunk2_kw()))
    m.q_c[:] = 0.0
    m.q[:] = 1.02 * th.qsat(m.T_atm, m.p_mid[None, :])         # 2% supersaturated (a realistic per-substep excess)
    # corrected (Chunk-2 default): one Newton step -> at saturation
    q_corr = m._prognostic_microphysics(m.q.copy(), 600.0)
    rh_corr = q_corr / th.qsat(m.T_atm, m.p_mid[None, :])
    assert np.all(np.abs(rh_corr - 1.0) < 1.0e-3)              # corrected adjustment lands AT saturation
    # bare skeleton adjustment on the same start: over-condenses -> ends SUB-saturated (RH < 1)
    m2 = StageADCore(StageADConfig(nx=8, nz=16, theta_max_deg=55.0, prognostic_condensate=True,
                                   qc_autoconv_crit=1.0e3, qc_retained_ceiling=False, qc_dqsat_adjust=False,
                                   **_chunk2_kw()))
    m2.q_c[:] = 0.0
    m2.q[:] = 1.3 * th.qsat(m2.T_atm, m2.p_mid[None, :])
    q_bare = m2._prognostic_microphysics(m2.q.copy(), 600.0)
    rh_bare = q_bare / th.qsat(m2.T_atm, m2.p_mid[None, :])
    assert np.all(rh_bare < 1.0)                                # bare over-condenses (sub-saturated after)
    assert np.all(rh_bare <= rh_corr + 1e-12)                  # corrected is closer to (and at) saturation


def test_reevaporation_drives_qc_to_zero_in_subsaturated_layer():
    """Suspended-q_c re-evaporation (Chunk 2): in a humid SUB-saturated layer (RH<1) the retained condensate
    re-evaporates back to vapour (latent cooling), driving q_c -> 0 -- the haze artifact's WATER half removed
    physically. With re-evaporation OFF the same q_c PERSISTS (no sink in sub-saturated air below the
    autoconversion threshold). Isolated on direct ``_prognostic_microphysics`` calls (no surface evaporation /
    convection refilling), with the deck SMALL relative to the sub-saturation deficit so it can fully drain."""
    base = dict(nx=8, nz=16, theta_max_deg=55.0, prognostic_condensate=True, qc_autoconv_crit=1.0,
                qc_retained_ceiling=False, **_chunk2_kw())     # grid-mean re-evap to the deficit (PDF reevap below)
    on = StageADCore(StageADConfig(qc_reevaporation=True, **base))
    off = StageADCore(StageADConfig(qc_reevaporation=False, **base))
    for m in (on, off):
        m.T_atm[:] = 285.0                                     # uniform warm column (deficit >> the deck everywhere)
        m.q[:] = 0.7 * th.qsat(m.T_atm, m.p_mid[None, :])     # humid but sub-saturated
        m.q_c[:] = 1.0e-4                                      # a thin suspended deck (below the autoconv crit)
        for _ in range(5):
            m.q[:] = m._prognostic_microphysics(m.q.copy(), 600.0)
    assert float(np.max(on.q_c)) < 1.0e-9                       # re-evaporation drains the suspended deck
    assert float(np.max(off.q_c)) > 5.0e-5                      # without it the deck persists in sub-saturated air


def test_coupled_sink_positivity_limiter():
    """The coupled-sink positivity limiter (review #4): autoconversion + re-evaporation + the retained ceiling
    cannot remove MORE than the available q_c in one step (q_c >= 0 exactly), tested where the combined demand
    far exceeds q_c (tau << dt so dt/tau >> 1, AND a deep sub-saturation so re-evaporation alone would also
    overshoot)."""
    m = StageADCore(StageADConfig(nx=8, nz=16, theta_max_deg=55.0, prognostic_condensate=True,
                                  qc_autoconv_tau=1.0e-3, qc_autoconv_crit=0.0,   # dt/tau ~ 6e5 >> 1
                                  qc_autoconv_tau_ice=1.0e-3, qc_autoconv_crit_ice=0.0))
    m.q[:] = 0.2 * th.qsat(m.T_atm, m.p_mid[None, :])          # deeply sub-saturated -> large re-evap demand too
    m.q_c[:] = 1.0e-3
    q = m._prognostic_microphysics(m.q.copy(), 600.0)
    assert np.all(m.q_c >= 0.0)                                 # positivity holds across the coupled update
    assert np.all(np.isfinite(q)) and np.all(np.isfinite(m.T_atm))


def test_cold_edge_column_condenses_via_ice_path():
    """Mixed/ice phase (Chunk 2, IN SCOPE because the band's rain peaks at the cold edge): a column below
    T_FREEZE condenses via the ICE path -- deposition uses the sublimation latent heat L_SUB (not L_VAP), and
    the ice autoconversion branch rains it. Verified by the latent-heat increment in a cold cell matching
    L_SUB, and by ice condensate forming/raining below freezing."""
    m = StageADCore(StageADConfig(nx=6, nz=16, theta_max_deg=55.0, prognostic_condensate=True,
                                  qc_autoconv_crit_ice=1.0e3, qc_retained_ceiling=False))  # hold ice q_c to inspect
    m.T_atm[:] = 250.0                                          # entire column below freezing -> ice phase
    m.q_c[:] = 0.0
    qs = th.qsat(m.T_atm, m.p_mid[None, :])
    m.q[:] = 1.2 * qs                                           # supersaturated -> deposition
    T0 = m.T_atm.copy()
    q = m._prognostic_microphysics(m.q.copy(), 600.0)
    cond = (m.q - q)                                            # what deposited to ice this step (re-evap inert here)
    assert np.all(m.q_c > 0.0)                                  # ice condensate formed below freezing
    dT_pred = C.L_SUB * cond / C.CP_AIR                         # ice path uses L_SUB, not L_VAP
    assert np.allclose(m.T_atm - T0, dT_pred, rtol=1e-6)        # deposition latent heat is L_SUB
    assert not np.allclose(m.T_atm - T0, C.L_VAP * cond / C.CP_AIR, rtol=1e-3)   # and NOT L_VAP


def test_steady_qc_is_finite_and_budget_closes():
    """Chunk-2 steady state: q_c balances source vs sinks to a FINITE deck (not 0, not unbounded), and the
    re-derived reservoir budget d/dt INTEGRAL(q+q_c)dm = E - P closes with P/E -> 1 as dq_c/dt -> 0."""
    m = StageADCore(StageADConfig(nx=16, nz=18, theta_max_deg=55.0, prognostic_condensate=True, **_chunk2_kw()))
    m.run_dynamics = False
    m.couple_surface = False
    m.T_surf[:] = 295.0
    for _ in range(500):                                        # spin to a steady reservoir (and column RCE)
        m.step()
    qc_mid = float(np.mean(m.q_c))
    assert qc_mid > 0.0 and np.isfinite(qc_mid)                 # a finite, non-zero steady deck (not 0, not unbounded)
    qcp0 = float(np.sum(m.q_c * m.dm[None, :]) * m.dx)
    m.reset_water_budget()
    tw0 = float(np.sum((m.q + m.q_c) * m.dm[None, :]) * m.dx)
    for _ in range(150):
        m.step()
    tw1 = float(np.sum((m.q + m.q_c) * m.dm[None, :]) * m.dx)
    qcp1 = float(np.sum(m.q_c * m.dm[None, :]) * m.dx)
    wb = m.water_conservation()
    lhs = tw1 - tw0
    rhs = wb["cum_evap"] - wb["cum_precip"]
    assert abs(lhs - rhs) / (abs(rhs) + 1e-12) < 1e-6          # the reservoir budget closes (exact conservation)
    assert abs(qcp1 - qcp0) / max(wb["cum_precip"], 1e-30) < 1e-3   # dq_c/dt -> 0 (the reservoir is steady)
    pe = wb["cum_precip"] / max(wb["cum_evap"], 1e-30)
    assert abs(pe - 1.0) < 0.1                                  # P/E -> 1 once the column is steady


def test_sundqvist_autoconversion_variant_runs():
    """The compared autoconversion variant (Sundqvist) RUNS (never silently dropped) and gives a different
    precip-efficiency than Kessler -- a reported sensitivity, not a fit."""
    base = dict(nx=16, nz=18, theta_max_deg=55.0, prognostic_condensate=True, clouds=False)
    kess = StageADCore(StageADConfig(qc_autoconv_scheme="kessler", **base))
    sund = StageADCore(StageADConfig(qc_autoconv_scheme="sundqvist", **base))
    for m in (kess, sund):
        m.run_dynamics = False
        m.couple_surface = False
        m.T_surf[:] = 293.0
        m.reset_water_budget()
    for _ in range(60):
        kess.step(); sund.step()
    assert np.all(np.isfinite(sund.q_c)) and float(np.max(sund.q_c)) > 0.0   # the variant runs and builds cloud
    # the two closures partition source->cloud->precip differently (a sensitivity, both physical)
    assert kess.water_conservation()["cum_precip"] != sund.water_conservation()["cum_precip"]


def test_autoconv_scheme_rejects_unknown():
    """PR #143 review #4: ``qc_autoconv_scheme`` is validated at construction -- an unknown string (typo,
    trailing space, wrong case) RAISES instead of silently falling back to Kessler (mirrors the
    heat_advection scheme guard), so a misconfigured env-driven sweep fails loudly."""
    import pytest as _pytest
    StageADCore(StageADConfig(nx=8, nz=14, theta_max_deg=55.0, qc_autoconv_scheme="kessler"))   # valid
    StageADCore(StageADConfig(nx=8, nz=14, theta_max_deg=55.0, qc_autoconv_scheme="sundqvist"))  # valid
    for bad in ("Kessler", "kessler ", "kessIer", "linear"):
        with _pytest.raises(ValueError):
            StageADCore(StageADConfig(nx=8, nz=14, theta_max_deg=55.0, qc_autoconv_scheme=bad))


def test_retained_ceiling_does_not_rain_all_when_cover_collapses():
    """PR #143 review #3: where the cover collapses below the floor (trace condensate q_c <= qc_cover_thresh in
    sub-saturated air, so f -> 0), the retained-water ceiling must NOT degenerate into "rain everything" --
    such trace q_c is left to re-evaporation/advection, not force-rained by a qc_ceiling -> 0. (A real deck,
    q_c > thresh, still gets the floored cover and IS capped.) Re-evaporation off so the ceiling is the only
    sub-saturated sink under test."""
    base = dict(nx=6, nz=16, theta_max_deg=55.0, prognostic_condensate=True, qc_reevaporation=False,
                qc_autoconv_crit=1.0, **_chunk2_kw())              # autoconv inert (huge crit); rainout-cap ceiling
    m = StageADCore(StageADConfig(**base))
    m.T_atm[:] = 285.0
    m.q[:] = 0.5 * th.qsat(m.T_atm, m.p_mid[None, :])              # sub-saturated -> raw RH cover ~ 0
    m.q_c[:] = 5.0e-8                                               # trace, below qc_cover_thresh (1e-7)
    m._prognostic_microphysics(m.q.copy(), 600.0)
    assert float(np.max(m.q_c)) > 0.0                              # trace condensate is NOT fully rained away
    # a real deck (above the cover threshold) is still capped by the ceiling (finite, not blown up)
    m2 = StageADCore(StageADConfig(**base))
    m2.T_atm[:] = 285.0
    m2.q[:] = 0.5 * th.qsat(m2.T_atm, m2.p_mid[None, :])
    m2.q_c[:] = 2.0e-4
    m2._prognostic_microphysics(m2.q.copy(), 600.0)
    assert 0.0 < float(np.max(m2.q_c)) < 2.0e-4                    # capped to the retained ceiling, still positive


def test_ab_precip_efficiency_shift_vs_skeleton():
    """The headline Chunk-2 A/B: the real microphysics changes the effective PRECIPITATION EFFICIENCY versus
    the Chunk-1 skeleton (a FINDING, not a tuning). With re-evaporation returning suspended condensate to
    vapour and the dqsat correction trimming over-condensation, the full scheme rains LESS per unit
    evaporation than the bare skeleton -- documented here as a measured shift, not asserted to a target."""
    base = dict(nx=20, nz=18, theta_max_deg=55.0, prognostic_condensate=True, clouds=False)
    skel = StageADCore(StageADConfig(**_skeleton_kw(), **base))
    full = StageADCore(StageADConfig(**base))                   # Chunk-2 defaults (all real microphysics)
    for m in (skel, full):
        m.run_dynamics = False
        m.couple_surface = False
        m.T_surf[:] = 295.0
        for _ in range(40):
            m.step()
        m.reset_water_budget()
        for _ in range(60):
            m.step()
    pe_skel = skel.water_conservation()["cum_precip"] / max(skel.water_conservation()["cum_evap"], 1e-30)
    pe_full = full.water_conservation()["cum_precip"] / max(full.water_conservation()["cum_evap"], 1e-30)
    assert np.isfinite(pe_skel) and np.isfinite(pe_full)
    assert pe_skel != pe_full                                   # the microphysics shifts precip efficiency (finding)


# ============================================================ CHUNK 3 -- one PDF for cover AND source; cap decouple
def test_pdf_partition_cover_zero_iff_condensate():
    """The haze fix on ONE footing (CLOUD_PROGNOSTIC_ROADMAP Chunk 3): cover and the condensation source come
    from the SAME total-water PDF, so f > 0 <=> qc_eq > 0 EXACTLY -- no cloud fraction without condensed water
    (the haze artifact's cover half), and no condensate where there is no cover. Swept across the saturated
    tail with a uniform column."""
    nz = 12
    T = np.full(nz, 285.0); p = np.linspace(3.0e4, 9.0e4, nz)
    qs = th.qsat(T, p)
    # total water from well below cloud onset (qs - Delta, Delta=(1-rh_crit)*qs) to well above (overcast)
    for frac in (0.5, 0.75, 0.8, 0.85, 0.9, 1.0, 1.1, 1.3):
        f, qc_eq = rad.cloud_pdf_partition(frac * qs, T, p, rh_crit=0.8)
        assert np.all((f > 0.0) == (qc_eq > 0.0))               # cover present IFF condensate present (one PDF)
        assert np.all((f >= 0.0) & (f <= 1.0))                  # a probability
        assert np.all(qc_eq >= 0.0)
    # the overcast limit reduces to the grid-mean adjustment: qc_eq -> q_t - qsat, f -> 1
    f_hi, qc_hi = rad.cloud_pdf_partition(1.5 * qs, T, p, rh_crit=0.8)
    assert np.allclose(f_hi, 1.0)
    assert np.allclose(qc_hi, 1.5 * qs - qs)                    # grid-mean excess in the overcast limit


def test_pdf_cover_monotone_in_total_water():
    """Cover is MONOTONE in the grid-mean total water (a well-posed PDF diagnostic): more water -> at least as
    much cloud, never less."""
    nz = 10
    T = np.full(nz, 280.0); p = np.linspace(3.0e4, 9.0e4, nz)
    qs = th.qsat(T, p)
    prev = None
    for frac in np.linspace(0.6, 1.4, 17):
        f, _ = rad.cloud_pdf_partition(frac * qs, T, p, rh_crit=0.8)
        if prev is not None:
            assert np.all(f >= prev - 1e-12)                    # non-decreasing in q_t
        prev = f


def test_pdf_source_fires_at_grid_mean_subsaturation():
    """The load-bearing Chunk-3 point (review #2): the condensation source fires from the PDF's saturated TAIL
    even at grid-mean RH < 1 -- so a humid (RH=0.9 > rh_crit) but grid-mean SUB-saturated column with no
    condensate SOURCES q_c under the PDF, where the Chunk-2 GRID-MEAN source (which keys on max(q-qsat,0)) makes
    NONE. The two footings are A/B'd on one isolated microphysics call."""
    base = dict(nx=8, nz=16, theta_max_deg=55.0, prognostic_condensate=True, qc_autoconv_crit=1.0,  # no rain
                qc_retained_ceiling=False)
    pdf = StageADCore(StageADConfig(qc_pdf_cloud=True, **base))
    grid = StageADCore(StageADConfig(qc_pdf_cloud=False, **base))
    for m in (pdf, grid):
        m.T_atm[:] = 285.0
        m.q[:] = 0.9 * th.qsat(m.T_atm, m.p_mid[None, :])      # RH 0.9: sub-saturated in the mean, > rh_crit
        m.q_c[:] = 0.0
        m._prognostic_microphysics(m.q.copy(), 600.0)
    assert float(np.max(pdf.q_c)) > 0.0                         # the PDF tail condenses at grid-mean RH < 1
    assert float(np.max(grid.q_c)) == 0.0                       # the grid-mean source makes none (RH < 1)


def test_pdf_reevaporation_drains_suspended_deck():
    """Chunk-3 re-evaporation: under the PDF, a suspended deck in a humid SUB-saturated column (below cloud
    onset, qc_eq = 0) re-evaporates toward the PDF equilibrium (here zero), draining q_c -> ~0 over the
    saturation-adjustment relaxation; with re-evaporation OFF the deck persists. (The relaxation is per-step
    geometric -- the Newton-corrected adjustment -- vs the Chunk-2 grid-mean full-deficit drain, so it needs
    more steps; the haze-water removal is the same physics.)"""
    base = dict(nx=8, nz=16, theta_max_deg=55.0, prognostic_condensate=True, qc_autoconv_crit=1.0,
                qc_retained_ceiling=False)                      # autoconv inert; re-evap the only sub-sat sink
    on = StageADCore(StageADConfig(qc_reevaporation=True, **base))
    off = StageADCore(StageADConfig(qc_reevaporation=False, **base))
    for m in (on, off):
        m.T_atm[:] = 285.0
        m.q[:] = 0.6 * th.qsat(m.T_atm, m.p_mid[None, :])      # below cloud onset (rh_crit 0.8) -> qc_eq = 0
        m.q_c[:] = 1.0e-4
        for _ in range(250):                                   # geometric relaxation; slowest at the high-qsat top
            m.q[:] = m._prognostic_microphysics(m.q.copy(), 600.0)
    assert float(np.max(on.q_c)) < 1.0e-8                       # PDF re-evaporation drains the suspended deck
    assert float(np.max(off.q_c)) > 5.0e-5                      # without it the deck persists in sub-saturated air


def test_chunk3_radiative_strength_tracks_qc_in_core():
    """Chunk-3 cap decoupling, at the CORE level: with the radiative cap decoupled (default), the cloud LW
    opacity the in-loop radiation sees (sum of dtau_lw_cloud = f*(kabs/D)*LWP, LINEAR in the in-cloud water)
    TRACKS q_c -- a wetter reservoir is more opaque. With the cap NOT decoupled (the Chunk-1 rainout ceiling)
    the optics SATURATE, so the opacity is far less sensitive to q_c. The headline q_c-driven feedback that
    Chunk 1 only got in sign. (The column SW albedo pins near 1 for a fully-cloudy column -- a poor probe -- so
    the LINEAR LW opacity is used.)"""
    base = dict(nx=8, nz=16, theta_max_deg=55.0, prognostic_condensate=True, clouds=True)

    def lw_opacity(qc_val, decouple):
        m = StageADCore(StageADConfig(qc_radiative_cap_decouple=decouple, **base))
        m.q[:] = th.qsat(m.T_atm, m.p_mid[None, :])            # saturated -> PDF cover well above 0
        m.q_c[:] = qc_val
        return float(np.sum(m._radiate(np.clip(m.q, 0.0, 0.5))["dtau_lw_cloud"]))

    a_lo, a_hi = lw_opacity(3.0e-4, True), lw_opacity(1.5e-3, True)     # decoupled: LWP tracks q_c
    b_lo, b_hi = lw_opacity(3.0e-4, False), lw_opacity(1.5e-3, False)   # rainout cap: saturates
    assert a_hi > 1.2 * a_lo                                    # the decoupled optics track the reservoir (>20%)
    assert (a_hi - a_lo) > 3.0 * (b_hi - b_lo)                  # the un-decoupled (rainout) optics are ~saturated/flat


def test_chunk3_pdf_flag_changes_trajectory_and_off_master_inert():
    """The Chunk-3 PDF flag is LIVE (it changes the moisture trajectory when the master is ON) but INERT when
    the master is OFF (production untouched). One coupled run each."""
    base = dict(nx=12, nz=16, theta_max_deg=55.0)
    pdf = StageADCore(StageADConfig(prognostic_condensate=True, qc_pdf_cloud=True, **base))
    grid = StageADCore(StageADConfig(prognostic_condensate=True, qc_pdf_cloud=False, **base))
    for _ in range(20):
        pdf.step(); grid.step()
    assert not np.array_equal(pdf.q_c, grid.q_c)                # the PDF source/cover changes q_c (live flag)
    # master OFF: the Chunk-3 flags are inert (byte-identical to the single-pass core)
    off_a = StageADCore(StageADConfig(prognostic_condensate=False, qc_pdf_cloud=True,
                                      qc_radiative_cap_decouple=True, **base))
    off_b = StageADCore(StageADConfig(prognostic_condensate=False, qc_pdf_cloud=False,
                                      qc_radiative_cap_decouple=False, **base))
    for _ in range(20):
        off_a.step(); off_b.step()
    assert np.array_equal(off_a.T_atm, off_b.T_atm) and np.array_equal(off_a.q, off_b.q)


# ============================================================ CHUNK 4 -- BL / stratocumulus closure
def _subsiding_inversion_core(nx=8, nz=18, w0=-0.02, **kw):
    """A core set up for the BL/stratocumulus closure WITHOUT the resolved dynamics: the default-initialised
    T_atm is stably stratified in potential temperature (a ~6.5 K/km lapse -> dtheta/dz ~ +3 K/km, above the
    2 K/km inversion gate), so a uniform SUBSIDENCE (w0 < 0, height grid) makes every column a subsidence-
    inversion column. ``run_dynamics`` off so ``step`` won't run the flow; the BL closure is driven directly
    by ``_bl_stratocumulus(w, dt)``. Returns (core, w) with w the (nz, nx) height-grid vertical velocity."""
    m = StageADCore(StageADConfig(nx=nx, nz=nz, theta_max_deg=55.0, prognostic_condensate=True, **kw))
    m.run_dynamics = False
    m.couple_surface = False
    m._z = m._hydrostatic_z(m.T_atm)
    m.q[:] = 0.5 * th.qsat(m.T_atm, m.p_mid[None, :])      # sub-saturated (RH 0.5 < rh_crit) -> the BULK makes no cloud
    m.q_c[:] = 0.0
    w = np.full((m.cfg.nz, m.cfg.nx), w0)                   # uniform subsidence (height grid, nz, nx)
    return m, w


def _sigma(m):
    return m.p_mid / C.P_SURF


def test_bl_deck_forms_and_holds_where_bulk_is_clear():
    """The headline Chunk-4 capability: a subsidence-inversion column FORMS and HOLDS a stratocumulus deck
    (persistent q_c in the sigma~0.7-0.9 band) where the bulk Chunk-1/3 scheme leaves it clear. The bulk
    microphysics on the SAME sub-saturated column makes ~no condensate (qc_eq=0 at grid-mean RH<rh_crit); the
    BL closure builds a deck confined to the low-cloud band and it persists across many steps."""
    # the bulk path makes no cloud in this sub-saturated subsiding column
    bulk, _ = _subsiding_inversion_core()
    bulk._prognostic_microphysics(bulk.q.copy(), 600.0)
    assert float(np.max(bulk.q_c)) < 1.0e-12                # the bulk scheme leaves the subsiding column clear
    # the BL closure forms and holds a deck in the low-cloud band
    m, w = _subsiding_inversion_core()
    for _ in range(80):
        m._bl_stratocumulus(w, 600.0)
    band = (_sigma(m) >= m.cfg.bl_sc_sigma_lo) & (_sigma(m) <= m.cfg.bl_sc_sigma_hi)
    assert float(np.max(m.q_c[:, band])) > 1.0e-5          # a persistent deck formed in the band
    assert np.all(m.q_c[:, ~band] == 0.0)                  # and ONLY in the band (no cloud outside it)
    assert np.all(m.q_c >= 0.0)
    # hold check: a few more steps barely move it (steady, persistent -- not a transient)
    qc_a = m.q_c.copy()
    for _ in range(20):
        m._bl_stratocumulus(w, 600.0)
    assert np.max(np.abs(m.q_c - qc_a)) < 0.05 * float(np.max(qc_a))   # the deck HOLDS (steady)


def test_bl_deck_inert_without_subsidence_inversion():
    """The closure is inert where there is no subsidence inversion: an ASCENDING column (w > 0) gets NO deck
    (the subsidence gate fails), and a STATICALLY NEUTRAL column (no inversion cap) gets none either."""
    up, w_up = _subsiding_inversion_core(w0=+0.02)         # ascending -> subsidence gate fails
    for _ in range(40):
        up._bl_stratocumulus(w_up, 600.0)
    assert float(np.max(up.q_c)) == 0.0                    # no deck without subsidence
    # neutral stratification (theta uniform with height) -> the inversion-cap gate fails even when subsiding
    neutral, w = _subsiding_inversion_core()
    theta0 = blc.potential_temperature(neutral.T_atm[:, -1:], neutral.p_mid[None, -1:])
    neutral.T_atm[:] = theta0 * (neutral.p_mid[None, :] / C.P_SURF) ** C.KAPPA_POISSON   # uniform theta
    neutral._z = neutral._hydrostatic_z(neutral.T_atm)
    neutral.q[:] = 0.5 * th.qsat(neutral.T_atm, neutral.p_mid[None, :])
    for _ in range(40):
        neutral._bl_stratocumulus(w, 600.0)
    assert float(np.max(neutral.q_c)) == 0.0              # no deck without a stable inversion cap


def test_bl_deck_does_not_run_away_or_limit_cycle():
    """Paired with 'forms and holds': the deck is BOUNDED (no runaway) and CONVERGES (no limit cycle). The
    source saturates as the deck emissivity -> 1 while the entrainment sink is linear in q_c, so q_c relaxes
    monotonically to a finite steady value -- tested even with a strong cloud-top cooling (ctop_cool_eff high)."""
    m, w = _subsiding_inversion_core(bl_sc_ctop_cool_eff=3.0)
    hist = []
    for _ in range(400):
        m._bl_stratocumulus(w, 600.0)
        hist.append(float(np.max(m.q_c)))
    assert np.all(np.isfinite(m.q_c))
    assert float(np.max(m.q_c)) < rad.CLOUD_LWC_PLAUSIBLE  # bounded well below a runaway (and the radiative cap)
    # converged (no limit cycle): the last stretch is steady, and the increments don't oscillate in sign
    tail = np.array(hist[-50:])
    assert (tail.max() - tail.min()) < 1.0e-3 * tail.mean()        # steady tail (converged, not cycling)
    diffs = np.diff(np.array(hist[5:]))
    assert np.sum(diffs < -1e-12) <= 2                             # essentially monotone approach (no oscillation)


def test_bl_source_emissivity_closes_with_radiation():
    """'Closes with the radiation, no double-count': the closure's cloud-top emissivity uses the SAME
    kabs-based law as the optics (eps = 1 - exp(-kabs*LWP_in)), and the closure applies NO radiative
    tendency itself -- it only moves water between q and q_c (total water conserved) with the matching latent
    heat. The cloud-top COOLING is applied by the radiation, via the deck's q_c feeding dtau_lw_cloud."""
    m, w = _subsiding_inversion_core()
    dm = m.dm[None, :]
    qc = np.full_like(m.T_atm, 3.0e-4)                              # a deck thick enough that the eps floor is inactive
    eps = blc.deck_emissivity(qc, m.T_atm, m.p_mid[None, :], dm, deck_cover=m.cfg.bl_sc_deck_cover)
    lwp_in = qc * dm / m.cfg.bl_sc_deck_cover
    assert np.allclose(eps, 1.0 - np.exp(-rad.CLOUD_KABS_LW * lwp_in))   # SAME kabs law as cloud_optical_properties
    # the closure is single-reservoir + energy-consistent: total water conserved across one BL step
    tw0 = float(np.sum((m.q + m.q_c) * dm))
    m._bl_stratocumulus(w, 600.0)
    tw1 = float(np.sum((m.q + m.q_c) * dm))
    assert abs(tw1 - tw0) / (tw0 + 1e-30) < 1e-12                   # q + q_c conserved (no spurious source/sink)


def test_bl_deck_cover_floor_makes_radiation_see_the_deck():
    """The deck must RADIATE as a solid low cloud: the closure stamps an overcast cover floor on its deck
    cells, so the in-loop radiation gives the deck a real cloud LW opacity even though the grid-mean RH cover
    there is ~0 (sub-saturated subsidence). Without the floor (bl off) the same q_c reads near-zero cover ->
    near-zero opacity."""
    m, w = _subsiding_inversion_core(clouds=True)
    for _ in range(40):
        m._bl_stratocumulus(w, 600.0)
    band = (_sigma(m) >= m.cfg.bl_sc_sigma_lo) & (_sigma(m) <= m.cfg.bl_sc_sigma_hi)
    out_on = m._radiate(np.clip(m.q, 0.0, 0.5))                    # bl on: deck cover floor active
    assert float(np.max(out_on["dtau_lw_cloud"][:, band])) > 0.0   # the deck has real LW opacity (radiates)
    # same q_c, but with the cover floor OFF: the sub-saturated band reads ~0 cover -> ~0 cloud opacity
    m.cfg.bl_stratocumulus = False
    out_off = m._radiate(np.clip(m.q, 0.0, 0.5))
    assert (float(np.max(out_off["dtau_lw_cloud"][:, band]))
            < 0.1 * float(np.max(out_on["dtau_lw_cloud"][:, band])))   # no floor -> the deck barely radiates


def _bl_roundtrip_kw():
    """A small COUPLED core (real dynamics + radiation) that forms a BL deck within a few steps -- the
    round-trip fixture: the deck cover must be stamped by ``step()`` itself, not by a hand-driven call."""
    return dict(nx=16, nz=12, prognostic_condensate=True, bl_stratocumulus=True, clouds=True,
                conv_tau=7200.0)


def test_bl_deck_cover_checkpoint_roundtrip_resume_is_bit_identical():
    """The deck cover is CARRY-OVER state across a save/resume boundary, so it is checkpointed. It is
    derived per step by the closure, but the next step's radiation reads it as a cover FLOOR *before* that
    step re-derives it -- so an un-checkpointed resume radiates its first step with no floor and the whole
    trajectory diverges bitwise from the uninterrupted run (a reproducibility defect, not a bias; the twin
    of the PR #322 q_r round-trip, which had to disable this flag to dodge it).

    Asserted: the key round-trips; the resumed trajectory is BIT-identical; a LEGACY checkpoint (no key)
    loads zeros without crashing (and -- the sensitivity guard -- then reproduces the divergence this test
    exists to catch); a WRONG-SHAPE key (a regridded seed) falls back to zeros and still steps; the key is
    ABSENT when the closure is off (the flag-OFF checkpoint byte format is unchanged); and a cover-carrying
    checkpoint into a closure-OFF core is ignored (the W_surf/N_d convention)."""
    kw = _bl_roundtrip_kw()
    m = StageADCore(StageADConfig(**kw))
    for _ in range(8):
        m.step()
    st = m.state()
    assert "bl_deck_cover" in st
    assert float(np.max(st["bl_deck_cover"])) > 0.0            # a real deck is under test (not a null round-trip)
    m2 = StageADCore(StageADConfig(**kw))
    m2.load_state(st)
    assert np.array_equal(m2._bl_deck_cover, m._bl_deck_cover)
    for _ in range(3):
        m.step(); m2.step()
    for f in ("T_atm", "T_surf", "q", "q_c", "_bl_deck_cover", "_precip"):
        assert np.array_equal(getattr(m, f), getattr(m2, f)), f   # resumed trajectory BIT-identical
    # LEGACY checkpoint (pre-key / any older save): loads zeros, no crash -- and the pre-fix divergence
    # is exactly what this test guards (so the assertions above are sensitive, not vacuous).
    legacy = {k: v for k, v in st.items() if k != "bl_deck_cover"}
    m3 = StageADCore(StageADConfig(**kw))
    m3.load_state(legacy)
    assert np.all(m3._bl_deck_cover == 0.0)
    for _ in range(3):
        m3.step()
    assert not np.array_equal(m3.T_atm, m.T_atm)               # the defect, reproduced from a legacy save
    # WRONG-SHAPE key (a regridded seed passes unknown keys through at the DONOR shape): zeros, not a crash
    bad = dict(st)
    bad["bl_deck_cover"] = np.ones((kw["nx"] // 2, kw["nz"] // 2))
    m4 = StageADCore(StageADConfig(**kw))
    m4.load_state(bad)
    assert np.all(m4._bl_deck_cover == 0.0)
    m4.step()                                                  # and the core still runs (self-healing)
    # closure OFF -> no key at all (the flag-OFF checkpoint format is byte-unchanged), and a
    # cover-carrying checkpoint into an OFF core is ignored.
    off = StageADCore(StageADConfig(**{**kw, "bl_stratocumulus": False}))
    off.step()
    assert "bl_deck_cover" not in off.state()
    off2 = StageADCore(StageADConfig(**{**kw, "bl_stratocumulus": False}))
    off2.load_state(st)
    assert np.all(off2._bl_deck_cover == 0.0)


def test_bl_entrainment_efficiency_sensitivity():
    """The entrainment efficiency is a REPORTED sensitivity (roadmap rule 4), not a fit: a higher
    ``bl_sc_entrain_eff`` dries the deck faster -> a THINNER steady deck. Two runs, a measured shift."""
    thin, w1 = _subsiding_inversion_core(bl_sc_entrain_eff=2.0)
    thick, w2 = _subsiding_inversion_core(bl_sc_entrain_eff=0.5)
    for _ in range(150):
        thin._bl_stratocumulus(w1, 600.0)
        thick._bl_stratocumulus(w2, 600.0)
    qc_thin = float(np.max(thin.q_c))
    qc_thick = float(np.max(thick.q_c))
    assert qc_thin > 0.0 and qc_thick > 0.0
    assert qc_thick > 1.5 * qc_thin                               # less entrainment -> a thicker deck (sensitivity)


def test_bl_deck_keys_on_sigma_resolution_robust():
    """Box-test (rule 5): the deck detection keys on the PHYSICAL coordinate sigma = p/P_SURF, not a grid
    index, so the deck forms in the SAME sigma band [0.7, 0.9] at two different vertical resolutions -- it
    does not move with the grid."""
    for nz in (18, 30):
        m, w = _subsiding_inversion_core(nz=nz)
        for _ in range(60):
            m._bl_stratocumulus(w, 600.0)
        deck = np.where(np.max(m.q_c, axis=0) > 1.0e-6)[0]       # levels that grew a deck
        assert deck.size > 0                                      # a deck formed at this resolution
        sig = _sigma(m)[deck]
        assert np.all((sig >= m.cfg.bl_sc_sigma_lo - 1e-9) & (sig <= m.cfg.bl_sc_sigma_hi + 1e-9))   # in the band


def test_bl_integrated_source_is_resolution_robust():
    """Box-test on MAGNITUDE (PR #145 review #1): cloud-top radiative cooling is a SINGLE surface flux per
    column, so the column-INTEGRATED deck condensate (which sets the cloud radiative forcing, as
    dtau_lw_cloud sums over the band) must NOT scale with the number of cells in the σ-band. The source is
    one cloud-top flux distributed across the deck by mass, so the integrated deck water path converges
    across vertical resolutions (a per-cell flux would roughly DOUBLE it from nz=18 -> 45)."""
    def integrated_deck_lwp(nz):
        m, w = _subsiding_inversion_core(nx=6, nz=nz)
        for _ in range(200):                                     # spin to a steady deck
            m._bl_stratocumulus(w, 600.0)
        band = (_sigma(m) >= m.cfg.bl_sc_sigma_lo) & (_sigma(m) <= m.cfg.bl_sc_sigma_hi)
        return float(np.sum(m.q_c[0, band] * m.dm[band] / m.cfg.bl_sc_deck_cover))   # in-cloud LWP [kg/m^2]
    lwp18, lwp45 = integrated_deck_lwp(18), integrated_deck_lwp(45)
    assert lwp18 > 0.0 and lwp45 > 0.0
    assert abs(lwp45 - lwp18) / lwp18 < 0.10                     # integrated deck water converges (not n_cells-scaled)


def test_bl_source_latent_heating_is_phase_exact_across_freezing():
    """PR #145 review follow-up: the source distributes condensation so the per-cell latent heat (applied in
    _bl_stratocumulus with each cell's own phase L_eff) integrates EXACTLY to the cloud-top cooling flux
    F_cool, even on a deck that STRADDLES the freezing line -- because the rate denominator is the deck's
    L-weighted mass Σ_deck(L_eff·dm), not L_top·M_deck. Checked on a mixed-phase deck against an independent
    F_cool reconstruction."""
    nx, nz = 1, 12
    T = np.full((nx, nz), 265.0); T[:, -3:] = 280.0            # deck straddles freezing (ice aloft, liquid below)
    p = np.linspace(3.0e4, 1.0e5, nz)[None, :]
    dm = np.full((1, nz), 500.0)
    mask = np.broadcast_to((p / C.P_SURF >= 0.70) & (p / C.P_SURF <= 0.90), (nx, nz)).copy()
    qc = np.where(mask, 3.0e-4, 0.0)
    S, _ = blc.stratocumulus_rates(T, qc, mask, p, dm, deck_cover=1.0, ctop_cool_eff=1.0, entrain_eff=1.0,
                                   tau_entrain=3600.0, source_max=1.0e-6, f_win=0.30)
    L_eff = np.where(T < C.T_FREEZE, C.L_SUB, C.L_VAP)
    integrated_latent = float(np.sum(L_eff * S * dm))          # = F_cool if phase-exact
    LWP = float(np.sum(np.where(mask, qc * dm, 0.0)))
    eps = max(1.0 - np.exp(-rad.CLOUD_KABS_LW * LWP), blc.BL_SC_EPS_FLOOR)
    T_top = T[0, int(np.argmax(mask[0]))]
    F_cool = eps * 0.30 * C.SIGMA_SB * T_top ** 4
    assert abs(L_eff[mask][0] - L_eff[mask][-1]) > 0.0         # the deck really is mixed-phase
    assert abs(integrated_latent - F_cool) / F_cool < 1e-12    # integrated latent heating == one cloud-top flux


def test_bl_deck_persists_under_coupled_bulk_drain():
    """Coupled persistence (PR #145 review #2): in ``step`` the bulk microphysics (``_prognostic_microphysics``,
    whose re-evaporation drains q_c toward the PDF equilibrium ≈0 in the sub-saturated deck region) runs
    BEFORE ``_bl_stratocumulus`` on the same q_c. Exercised together in the step order, the BL closure still
    MAINTAINS a deck against the bulk drain (it does not get fully re-evaporated away)."""
    m, w = _subsiding_inversion_core()
    for _ in range(120):
        m.q = m._prognostic_microphysics(m.q.copy(), 600.0)     # bulk: re-evaporation drains the sub-saturated deck
        m._bl_stratocumulus(w, 600.0)                            # BL: cloud-top-cooling rebuild
    band = (_sigma(m) >= m.cfg.bl_sc_sigma_lo) & (_sigma(m) <= m.cfg.bl_sc_sigma_hi)
    assert float(np.max(m.q_c[:, band])) > 1.0e-5               # a deck persists despite the competing bulk sink
    assert np.all(m.q_c >= 0.0)


def test_bl_flag_live_and_master_off_inert():
    """The Chunk-4 flag is LIVE (it changes the trajectory when the master is ON and the flow is live) but
    INERT when the master is OFF (production untouched -- byte-identical)."""
    base = dict(nx=12, nz=16, theta_max_deg=55.0)
    on = StageADCore(StageADConfig(prognostic_condensate=True, bl_stratocumulus=True, **base))
    off = StageADCore(StageADConfig(prognostic_condensate=True, bl_stratocumulus=False, **base))
    for _ in range(25):
        on.step(); off.step()
    assert not np.array_equal(on.q_c, off.q_c)                   # the BL closure changes q_c (live flag)
    # master OFF: the Chunk-4 flag + knobs are inert (byte-identical to the single-pass core)
    a = StageADCore(StageADConfig(prognostic_condensate=False, bl_stratocumulus=True,
                                  bl_sc_entrain_eff=2.0, **base))
    b = StageADCore(StageADConfig(prognostic_condensate=False, bl_stratocumulus=False, **base))
    for _ in range(20):
        a.step(); b.step()
    assert np.array_equal(a.T_atm, b.T_atm) and np.array_equal(a.q, b.q)
    assert np.all(a.q_c == 0.0) and np.all(b.q_c == 0.0)


# ============================================================ CHUNK 5 -- rh_detrain operator-split reconciliation
@pytest.mark.parametrize("closure", ["rh_detrain", "q_homogenize"])
def test_chunk5_operator_split_joint_conservation(closure):
    """The Chunk-5 deliverable (CLOUD_PROGNOSTIC_ROADMAP): the convective moisture closure (which redistributes
    VAPOUR q) and the prognostic condensation (which owns CONDENSATE q_c) compose cleanly -- the JOINT total
    water q + q_c is conserved, d/dt INTEGRAL (q + q_c) dm = E - P, with the convective closure live ON the same
    column. Tested for BOTH closures: rh_detrain (the production default) and the q_homogenize compared variant.
    A warm fixed SST drives the column convectively unstable (so the adjustment actually fires) -- the operator
    split is genuinely exercised, not a dormant code path.

    SCOPE (PR #146 review note 2): this is COLUMN physics (run_dynamics=False) -- the convection<->condensation
    split IS column-only, so this is the right scope for Chunk 5. The resolved-flow q/q_c advection and the
    Chunk-4 BL-stratocumulus q<->q_c path (both gated on run_dynamics) are conserving and locked elsewhere
    (test_stage_ad_core.py's interior-advection + storage-closure gates; test_bl_source_emissivity_closes_with_
    radiation here), so the joint budget is closed end-to-end across the suite, not only in this column limit."""
    m = StageADCore(StageADConfig(nx=20, nz=18, theta_max_deg=55.0, prognostic_condensate=True,
                                  convective_closure=closure, qc_autoconv_crit=5.0e-4))
    m.run_dynamics = False
    m.couple_surface = False                                    # fixed SST -> bounded q (no clip source/sink)
    m.T_surf[:] = 292.0
    for _ in range(20):                                         # spin out the initial transient
        m.step()
    m.reset_water_budget()
    tw0 = float(np.sum((m.q + m.q_c) * m.dm[None, :]) * m.dx)   # area-weighted total water [kg/m]
    for _ in range(40):
        m.step()
    tw1 = float(np.sum((m.q + m.q_c) * m.dm[None, :]) * m.dx)
    wb = m.water_conservation()
    lhs = tw1 - tw0                                             # storage change
    rhs = wb["cum_evap"] - wb["cum_precip"]                     # E - P
    assert abs(lhs - rhs) / (abs(rhs) + 1e-12) < 1e-6          # joint reservoir budget closes under the closure


def test_chunk5_convective_detrainment_partitions_no_double_count():
    """The load-bearing Chunk-5 point: the convective vapour closure and the prognostic condensation PARTITION
    the supersaturation rather than double-count it. On a warm, near-saturated, convectively-unstable column the
    rh_detrain step (1) conserves column water EXACTLY (it MOVES vapour between layers, it does not condense or
    rain -- so it can never 'claim' supersaturation as precip) while (2) leaving a layer SUPERSATURATED (the
    detrained water exceeds the fixed-RH holding capacity); the prognostic microphysics is then the SOLE sink
    that claims that supersaturation into q_c. No path where both claim the same water. (Footing, PR #146 review
    note 3: this runs on the PRODUCTION default qc_pdf_cloud=True, so the microphysics source is the saturated
    TAIL of the total-water PDF -- the grid-mean supersaturation checked here is a SUBSET of that tail, so the
    handoff holds a fortiori under the default footing as well as the grid-mean one.)"""
    from flatworld.stage_ad_core import moist_convective_adjust
    m = StageADCore(StageADConfig(nx=4, nz=18, theta_max_deg=55.0, prognostic_condensate=True,
                                  qc_autoconv_crit=1.0, qc_retained_ceiling=False))   # autoconv inert -> q_c is the source readout
    m.run_dynamics = False
    # a warm, saturated, convectively-UNSTABLE column: a super-adiabatic kick in the lower troposphere so the
    # adjustment fires, with q at saturation so the detrained water overruns the rh_detrain (RH<=0.8) target.
    m.T_atm[:] = 290.0 - 6.0 * (np.arange(m.cfg.nz)[::-1] / m.cfg.nz)[None, :]   # ~cool aloft, warm low
    m.T_atm[:, -4:] += 8.0                                      # super-adiabatic kick near the surface (unstable)
    m.q[:] = th.qsat(m.T_atm, m.p_mid[None, :])                # saturated -> large pair water W
    m.q_c[:] = 0.0
    m._z = m._hydrostatic_z(m.T_atm)
    q_conv = m.q.copy()
    w0 = float(np.sum(q_conv * m.dm[None, :]))                  # column vapour BEFORE the convective closure
    moist_convective_adjust(m.T_atm, m.dm, m._z, m.p_mid, m.cfg.conv_max_sweeps, q=q_conv,
                            closure="rh_detrain", rh_surf=m.cfg.rh_surf, rh_top=m.cfg.rh_top)
    w1 = float(np.sum(q_conv * m.dm[None, :]))                  # ... and AFTER it
    assert abs(w1 - w0) / w0 < 1e-10                            # (1) convection removed NO water (matches the canonical rh_detrain gate rtol)
    super_after = q_conv - th.qsat(m.T_atm, m.p_mid[None, :])
    assert float(np.max(super_after)) > 0.0                    # (2) it LEFT a supersaturated layer for the sink
    # (3) the prognostic microphysics is the sole claimant: it sources q_c from that handed-off supersaturation
    m.q = m._prognostic_microphysics(q_conv.copy(), 600.0)
    assert float(np.max(m.q_c)) > 0.0                          # microphysics claims the supersaturation into q_c
    # and the claim conserves total water across the microphysics step minus the (here inert) precip
    tw_pre = w1                                                 # vapour pre-microphysics (q_c was 0)
    tw_post = float(np.sum((m.q + m.q_c) * m.dm[None, :]))
    precip_removed = float(np.sum(m._precip)) * 600.0          # column precip flux * dt (~0: autoconv crit=1.0 inert)
    assert precip_removed < 1e-6 * tw_pre                       # no autoconversion fired -> q_c is a pure source
    assert abs(tw_post - tw_pre) / tw_pre < 1e-9               # condensation is a q->q_c move (no leak)


@pytest.mark.parametrize("closure", ["rh_detrain", "q_homogenize"])
def test_chunk5_off_byte_identical_under_both_closures(closure):
    """OFF byte-identical for the operator split (roadmap Chunk-5 test): under BOTH convective closures, the
    prognostic machinery in the instant-rain SKELETON limit (tau->0, crit=0) reduces EXACTLY to the OFF
    single-pass core -- the convective closure runs identically in both, and the only difference (where the
    supersaturation is condensed) collapses to the same single-pass result. Production never moves regardless of
    the closure until the Chunk-7 opt-in."""
    base = dict(nx=16, nz=16, theta_max_deg=55.0, clouds=False, convective_closure=closure)
    off = StageADCore(StageADConfig(prognostic_condensate=False, **base))
    on = StageADCore(StageADConfig(prognostic_condensate=True, qc_autoconv_tau=1e-9,
                                   qc_autoconv_crit=0.0, **_skeleton_kw(), **base))
    for m in (off, on):
        m.run_dynamics = False
    for _ in range(30):
        off.step(); on.step()
    assert np.array_equal(off.T_atm, on.T_atm)                   # bit-identical T regardless of closure
    assert np.array_equal(off.q, on.q)                           # bit-identical vapour
    assert np.array_equal(off._precip, on._precip)               # bit-identical precip
    assert np.all(on.q_c == 0.0)                                 # all condensate rained instantly


def test_chunk5_closures_feed_distinct_qc_trajectories():
    """The operator split is genuinely closure-COUPLED, not closure-blind: rh_detrain (RH falls with height,
    no aloft slug) and q_homogenize (the constant-q slug) hand the prognostic condensation DIFFERENT vapour
    profiles, so the resulting q_c reservoir differs. (Both conserve -- guarded above; this only shows the
    convective closure actually reaches the prognostic source.)"""
    base = dict(nx=12, nz=16, theta_max_deg=55.0, prognostic_condensate=True)
    rd = StageADCore(StageADConfig(convective_closure="rh_detrain", **base))
    qh = StageADCore(StageADConfig(convective_closure="q_homogenize", **base))
    for _ in range(25):
        rd.step(); qh.step()
    assert np.any(rd.q_c > 0.0) and np.any(qh.q_c > 0.0)         # both build a reservoir
    assert not np.allclose(rd.q_c, qh.q_c)                       # the closure reaches the condensation source


# ================================================================== BELOW-CLOUD PRECIP RE-EVAPORATION (virga)
# The falling-precip "virga" fix (OPEN_ISSUES 2026-06-16 "Sub-cloud VIRGA"): ``_below_cloud_reevap`` integrates
# the downward precip flux and re-evaporates a deficit-limited portion below cloud, so ``_precip`` becomes the
# SURFACE-reaching flux (not the produced-aloft total). Default OFF -> byte-identical; the kernel conserves
# water + energy exactly and is limited by the ABSOLUTE deficit (cold snow lands; near-saturated rain lands).
def _reevap_column(Tval, RH, k_prod, nz=18, rate=1.5e-5, prod_qc=2.0e-4, dt=600.0, prognostic=True):
    """A throwaway column: uniform T, uniform RH, rain produced (autoconversion) in layer ``k_prod`` (the
    cloud base) only. Calls ``_below_cloud_reevap`` directly and returns (core, produced_flux, surface_flux,
    q_before, q_after) so a test can read off the virga fraction and check conservation in isolation."""
    m = StageADCore(StageADConfig(nx=4, nz=nz, theta_max_deg=55.0, prognostic_condensate=prognostic,
                                  precip_reevap=True, precip_reevap_rate=rate))
    m.T_atm[:] = Tval
    qs = th.qsat(m.T_atm, m.p_mid[None, :])
    q = (RH * qs).copy()
    precip_qc = np.zeros((4, nz)); precip_qc[:, k_prod] = prod_qc        # rain made at the cloud base
    produced = float(np.sum(precip_qc * m.dm[None, :], axis=1).mean()) / dt
    q_before = q.copy()
    F = m._below_cloud_reevap(q, precip_qc, dt)
    return m, produced, float(F.mean()), q_before, q


def test_precip_reevap_off_byte_identical_via_instant_rain_limit():
    """OFF byte-identical (the load-bearing regression guard, mirrors the Chunk-1 instant-rain test): with
    ``precip_reevap=False`` AND the Chunk-2 sub-flags reduced to the skeleton in the instant-rain limit, the
    prognostic core is BIT-IDENTICAL to the OFF single-pass core -- the new flag does not perturb the OFF path."""
    base = dict(nx=16, nz=16, theta_max_deg=55.0, clouds=False)
    off = StageADCore(StageADConfig(prognostic_condensate=False, precip_reevap=False, **base))
    on = StageADCore(StageADConfig(prognostic_condensate=True, qc_autoconv_tau=1e-9, qc_autoconv_crit=0.0,
                                   precip_reevap=False, **_skeleton_kw(), **base))
    for m in (off, on):
        m.run_dynamics = False
    for _ in range(30):
        off.step(); on.step()
    assert np.array_equal(off.T_atm, on.T_atm)                   # bit-identical T (precip_reevap OFF inert)
    assert np.array_equal(off.q, on.q)                           # bit-identical vapour
    assert np.array_equal(off._precip, on._precip)               # bit-identical precip


def test_precip_reevap_inert_on_single_pass_path_regardless_of_flag():
    """The single-pass core (``prognostic_condensate=False``) never calls ``_prognostic_microphysics``, so
    ``precip_reevap`` (True or False) is INERT there -- the clouds-OFF production path is byte-identical
    regardless of the new master flag."""
    base = dict(nx=16, nz=16, theta_max_deg=55.0, prognostic_condensate=False)
    a = StageADCore(StageADConfig(precip_reevap=False, **base))
    b = StageADCore(StageADConfig(precip_reevap=True, **base))
    for _ in range(25):
        a.step(); b.step()
    assert np.array_equal(a.q, b.q) and np.array_equal(a.T_atm, b.T_atm)    # inert on the single-pass path
    assert np.array_equal(a._precip, b._precip)


def test_precip_reevap_flag_is_live_and_reduces_surface_precip():
    """The flag is live (not a silent no-op): with an elevated cloud base over dry sub-cloud air, turning
    re-evaporation ON reduces the SURFACE precip below the produced-aloft total (the virga the OFF core books
    straight to the surface), and moves the difference into the sub-cloud vapour."""
    base = dict(nx=8, nz=18, theta_max_deg=55.0, prognostic_condensate=True)
    off = StageADCore(StageADConfig(precip_reevap=False, **base))
    on = StageADCore(StageADConfig(precip_reevap=True, **base))
    for m in (off, on):                                          # an elevated cloud base over sub-saturated air
        m.T_atm[:] = 290.0
        m.q[:] = 0.4 * th.qsat(m.T_atm, m.p_mid[None, :])
        m.q_c[:] = 0.0
        m.q_c[:, 8] = 2.0e-3                                     # condensate at an elevated base -> rains
    q_off = off._prognostic_microphysics(off.q.copy(), 600.0)
    q_on = on._prognostic_microphysics(on.q.copy(), 600.0)
    assert np.all(on._precip <= off._precip + 1e-30)            # ON books less to the surface (some virgas)
    assert np.any(on._precip < off._precip * 0.99)              # materially less where the sub-cloud is dry
    assert np.sum(q_on * on.dm[None, :]) > np.sum(q_off * off.dm[None, :])   # the virga'd water is now vapour


def test_precip_reevap_conserves_water_and_energy_in_the_kernel():
    """The kernel moves water from the falling flux into ``q`` and the matching latent heat out of ``T_atm``,
    both to round-off: the vapour ADDED equals (produced - surface) over the step, and the cooling increment
    equals -L_eff * dq / cp exactly (the conjugate of the production warming)."""
    m, produced, surface, q0, q1 = _reevap_column(300.0, 0.35, 6)
    T0 = m.T_atm.copy()
    # recompute the cooling the kernel should have applied from the vapour change it produced
    # (T_atm was already mutated in-place by _reevap_column's call; reconstruct from q0/q1)
    m2, produced2, surface2, q0b, q1b = _reevap_column(300.0, 0.35, 6)
    dW = float(np.sum((q1b - q0b) * m2.dm[None, :], axis=1).mean())     # vapour added [kg/m^2]
    assert abs(dW - (produced2 - surface2) * 600.0) / ((produced2 - surface2) * 600.0) < 1e-10   # water closes
    # energy: the applied cooling matches -L_eff*dq/cp layer by layer (read T before/after via a clean rebuild)
    m3 = StageADCore(StageADConfig(nx=4, nz=18, theta_max_deg=55.0, prognostic_condensate=True,
                                   precip_reevap=True))
    m3.T_atm[:] = 300.0
    q = (0.35 * th.qsat(m3.T_atm, m3.p_mid[None, :])).copy()
    precip_qc = np.zeros((4, 18)); precip_qc[:, 6] = 2.0e-4
    Tpre = m3.T_atm.copy(); qpre = q.copy()
    m3._below_cloud_reevap(q, precip_qc, 600.0)
    L_eff = np.where(Tpre < C.T_FREEZE, C.L_SUB, C.L_VAP)
    assert np.allclose(m3.T_atm - Tpre, -L_eff * (q - qpre) / C.CP_AIR, rtol=1e-12, atol=1e-12)   # exact conjugate


def test_precip_reevap_mixed_phase_energy_is_conjugate_across_freezing():
    """Energy conjugacy ACROSS the freezing level (PR #148 review #1): SNOW formed cold (production released
    L_SUB) that virgas in WARM sub-cloud must lose exactly L_SUB on the way down -- by melting (L_FUS, cooling
    the layer) then evaporating as rain (L_VAP), L_FUS + L_VAP = L_SUB. Without the melt term the single-flux
    scheme removed only L_VAP, leaving a one-signed L_FUS spurious HEATING right in the ribbon configuration.
    Set up that configuration (cold cloud aloft -> warm dry sub-cloud, snow produced cold) and verify the
    round-trip latent removal equals L_SUB * (re-evaporated snow mass) to round-off, and water still closes."""
    nz = 30
    m = StageADCore(StageADConfig(nx=4, nz=nz, theta_max_deg=55.0, prognostic_condensate=True,
                                  precip_reevap=True))
    Tprof = np.linspace(-25.0, 25.0, nz) + C.T_FREEZE          # cold aloft -> warm surface; freezing level mid-column
    m.T_atm[:] = Tprof[None, :]
    qs = th.qsat(m.T_atm, m.p_mid[None, :])
    RH = np.where(Tprof < C.T_FREEZE, 0.92, 0.30)              # cold humid (snow survives) -> warm dry (it virgas)
    q = (RH[None, :] * qs).copy()
    precip_qc = np.zeros((4, nz)); precip_qc[:, 3] = 3.0e-4    # snow produced near the top (T<0 -> L_SUB aloft)
    T0 = m.T_atm.copy(); q0 = q.copy()
    F = m._below_cloud_reevap(q, precip_qc, 600.0)
    reevap_mass = float(np.sum((q - q0) * m.dm[None, :], axis=1).mean())          # kg/m^2 re-evaporated
    cooling = float(np.sum(-(m.T_atm - T0) * C.CP_AIR * m.dm[None, :], axis=1).mean())   # J/m^2 latent removal
    produced = float(np.sum(precip_qc * m.dm[None, :], axis=1).mean())
    surface = float(F.mean()) * 600.0
    assert reevap_mass > 0.3 * produced                       # the configuration really does virga (test has teeth)
    # the snow released L_SUB aloft; the conjugate removal for the re-evaporated mass is L_SUB (no L_FUS leak)
    assert abs(C.L_SUB * reevap_mass - cooling) / cooling < 1e-6     # conjugate across the freezing level
    assert abs((produced - surface) - reevap_mass) / reevap_mass < 1e-9   # water closes (melt is internal)


def test_precip_reevap_partial_virga_cross_phase_energy_closes():
    """PARTIAL cross-phase virga (PR #148 review #1 coverage note): the mixed-phase conjugacy test above runs
    in the FULL-virga limit (surface->0), where L_SUB*reevap_mass == cooling happens to hold. This exercises
    the case where SOME melted rain reaches the ground (surface > 0), via a NON-CIRCULAR energy invariant
    independent of the kernel's internal melt/sublimation/evaporation split: treating the latent potential
    U = snow*L_SUB + rain*L_VAP (vapour = 0), closed-system conservation gives cooling = U_in - U_surface. With
    all production SNOW (cold layer) and an all-warm surface (every surviving flake has melted -> surface is
    pure RAIN):  cooling == L_SUB*snow_in - L_VAP*surface. Heavy snow over a warm sub-cloud lands ~half."""
    nz = 30
    m = StageADCore(StageADConfig(nx=4, nz=nz, theta_max_deg=55.0, prognostic_condensate=True,
                                  precip_reevap=True))
    Tprof = np.where(np.arange(nz) < 6, -15.0, 12.0) + C.T_FREEZE    # cold top (snow) -> all-WARM sub-cloud
    m.T_atm[:] = Tprof[None, :]
    qs = th.qsat(m.T_atm, m.p_mid[None, :])
    RH = np.where(Tprof < C.T_FREEZE, 0.95, 0.55)                   # cold humid (snow survives) -> warm dry
    q = (RH[None, :] * qs).copy()
    precip_qc = np.zeros((4, nz)); precip_qc[:, 2] = 3.0e-3         # HEAVY snow -> ~half reaches the ground
    snow_in = float(np.sum(precip_qc * m.dm[None, :], axis=1).mean())   # all production is snow (cold layer)
    T0 = m.T_atm.copy()
    F = m._below_cloud_reevap(q, precip_qc, 600.0)
    surface = float(F.mean()) * 600.0
    cooling = float(np.sum(-(m.T_atm - T0) * C.CP_AIR * m.dm[None, :], axis=1).mean())
    assert 0.1 < surface / snow_in < 0.9                          # genuinely PARTIAL virga (surface terms live)
    assert abs(cooling - (C.L_SUB * snow_in - C.L_VAP * surface)) / cooling < 1e-6   # non-circular conjugacy


def test_precip_reevap_respects_qc_ice_phase_off():
    """PR #148 review #2 (footgun fix): with ``qc_ice_phase=False`` the PRODUCTION releases L_VAP everywhere
    (no ice), so below-cloud re-evaporation must remove L_VAP too -- it must NOT call sub-freezing production
    "snow" and sublimate it at L_SUB, which would re-introduce the one-signed L_FUS=L_SUB-L_VAP over-cooling
    the mixed-phase path exists to avoid. In an ALL-COLD column (where the un-gated snow path WOULD sublimate
    at L_SUB), verify the latent removal is the L_VAP conjugate of the all-rain production (not L_SUB), and
    water closes. (Exercises the qc_ice_phase-gated source-phase fallback in _below_cloud_reevap.)"""
    nz = 24
    m = StageADCore(StageADConfig(nx=4, nz=nz, theta_max_deg=55.0, prognostic_condensate=True,
                                  precip_reevap=True, qc_ice_phase=False))
    m.T_atm[:] = -10.0 + C.T_FREEZE                               # entire column below freezing
    q = (0.40 * th.qsat(m.T_atm, m.p_mid[None, :])).copy()        # sub-saturated -> the falling precip virgas
    precip_qc = np.zeros((4, nz)); precip_qc[:, 4] = 3.0e-4       # produced in a cold layer (but NO ice: phase off)
    T0 = m.T_atm.copy(); q0 = q.copy()
    F = m._below_cloud_reevap(q, precip_qc, 600.0)
    reevap_mass = float(np.sum((q - q0) * m.dm[None, :], axis=1).mean())
    cooling = float(np.sum(-(m.T_atm - T0) * C.CP_AIR * m.dm[None, :], axis=1).mean())
    produced = float(np.sum(precip_qc * m.dm[None, :], axis=1).mean())
    surface = float(F.mean()) * 600.0
    assert reevap_mass > 0.3 * produced                          # it really virgas (test has teeth)
    assert abs(C.L_VAP * reevap_mass - cooling) / cooling < 1e-6  # L_VAP conjugate (NOT L_SUB) -> qc_ice_phase honored
    assert abs((produced - surface) - reevap_mass) / reevap_mass < 1e-9   # water closes


def test_precip_reevap_cold_low_rh_lands_its_snow_deficit_not_rh():
    """THE correctness condition (``_virga_diag`` DEFICIT_REF; the spec's load-bearing requirement): virga is
    limited by the ABSOLUTE deficit (qsat - q), NOT RH. At the SAME low RH and the SAME geometry, a COLD
    column (tiny qsat -> tiny deficit) lands almost all its falling SNOW, while a WARM column (large deficit)
    virgas almost all its rain -- the opposite of what an RH-only (1-RH) survival metric would give (which,
    being RH-only, would virga BOTH at the same low RH)."""
    _, prod_cold, surf_cold, _, _ = _reevap_column(235.0, 0.40, 15)     # cold, low RH, tiny absolute deficit
    _, prod_warm, surf_warm, _, _ = _reevap_column(305.0, 0.40, 15)     # warm, SAME low RH, large deficit
    virga_cold = 1.0 - surf_cold / prod_cold
    virga_warm = 1.0 - surf_warm / prod_warm
    assert virga_cold < 0.10                                    # cold: falling snow LANDS (deficit-limited)
    assert virga_warm > 0.80                                    # warm: rain virgas (same RH, large deficit)


def test_precip_reevap_saturated_subcloud_rain_lands():
    """Self-limiting near saturation: rain falling through a NEAR-SATURATED sub-cloud (RH -> 1, deficit -> 0)
    mostly LANDS -- the deficit-proportional rate vanishes as the air saturates, so a saturated column does
    NOT spuriously virga its rain (contrast the dry-sub-cloud case, which virgas heavily)."""
    _, prod_wet, surf_wet, _, _ = _reevap_column(290.0, 0.98, 9)        # near-saturated sub-cloud
    _, prod_dry, surf_dry, _, _ = _reevap_column(290.0, 0.40, 9)        # dry sub-cloud, same T/geometry
    assert surf_wet / prod_wet > 0.80                          # near-saturated: rain reaches the ground
    assert surf_dry / prod_dry < 0.20                          # dry: most of it virgas


def test_precip_reevap_heavy_rain_survives_drizzle_virgas():
    """Flux dependence (physical, falls out of the flux limit): the per-layer fractional loss ~ K_e(qsat-q)dm/F
    decreases with the rain flux F, so HEAVY rain reaches the ground while light DRIZZLE virgas -- through the
    SAME dry sub-cloud."""
    _, prod_lo, surf_lo, _, _ = _reevap_column(290.0, 0.50, 9, prod_qc=2.0e-5)   # drizzle
    _, prod_hi, surf_hi, _, _ = _reevap_column(290.0, 0.50, 9, prod_qc=4.0e-2)   # downpour
    assert surf_lo / prod_lo < 0.10                            # drizzle virgas
    assert surf_hi / prod_hi > 0.50                            # the downpour mostly survives


def test_precip_reevap_water_budget_closes_in_a_coupled_run():
    """Water closure with re-evaporation ON (the load-bearing conservation gate): on a no-flow column,
    d/dt INTEGRAL(q + q_c) dm = E - P closes to round-off with the new SURFACE precip P (no silent source/
    sink), and once steady P/E -> 1 -- the re-evaporated water re-enters q and is booked out of P, not lost.
    (Re-evaporation slows the approach to a steady reservoir, since the sub-cloud must moisten before rain
    lands, so the spin is longer than the no-re-evap steady test.)"""
    m = StageADCore(StageADConfig(nx=16, nz=18, theta_max_deg=55.0, prognostic_condensate=True,
                                  precip_reevap=True, **_chunk2_kw()))
    m.run_dynamics = False
    m.couple_surface = False
    m.T_surf[:] = 295.0
    for _ in range(1800):                                       # spin to a steady reservoir (re-evap recycles -> slow)
        m.step()
    m.reset_water_budget()
    tw0 = float(np.sum((m.q + m.q_c) * m.dm[None, :]) * m.dx)
    for _ in range(150):
        m.step()
    tw1 = float(np.sum((m.q + m.q_c) * m.dm[None, :]) * m.dx)
    wb = m.water_conservation()
    lhs = tw1 - tw0
    rhs = wb["cum_evap"] - wb["cum_precip"]
    assert abs(lhs - rhs) / (abs(rhs) + 1e-12) < 1e-6          # the reservoir budget closes with surface P (EXACT)
    assert abs(wb["closure_cum"] - 1.0) < 0.1                  # P/E -> 1 once steady


def test_precip_reevap_adds_no_energy_leak():
    """Energy: re-evaporation is INTERNAL (latent cooling with no top-of-atmosphere term -- the exact conjugate
    of the production warming, pinned by the kernel test), so it introduces NO energy leak. The coupled core's
    global TOA imbalance (absorbed SW - OLR) is the same with re-evaporation ON as OFF, to a tiny tolerance, at
    matched (still-converging) step counts -- the imbalance is the unconverged coupled core's own (the M3
    convergence study), not a re-evaporation source/sink. (Convergence to TOA -> 0 is the expensive M3 run.)"""
    base = dict(nx=20, nz=18, theta_max_deg=70.0, prognostic_condensate=True)
    off = StageADCore(StageADConfig(precip_reevap=False, **base)); off.run_dynamics = True; off.couple_heat = True
    on = StageADCore(StageADConfig(precip_reevap=True, **base)); on.run_dynamics = True; on.couple_heat = True
    for _ in range(200):
        off.step(); on.step()
    assert on._state_is_finite()                               # the coupled re-evap run does not blow up
    eo = off.energy_conservation()["rel_imbalance"]
    en = on.energy_conservation()["rel_imbalance"]
    assert abs(en - eo) < 5.0e-3                               # re-evap adds no TOA leak (tracks the OFF core)


def test_precip_reevap_column_integral_is_grid_convergent():
    """The rate-limited form E_k = K_e (qsat-q) dm_k makes the column-integrated re-evaporation a MASS INTEGRAL
    K_e INTEGRAL(qsat-q)dm -- so the surface-reaching fraction is GRID-CONVERGENT in nz (it does not drift as
    the vertical grid refines). Same physical column (uniform T/RH, cloud base at the same SIGMA fraction),
    doubled nz, the virga fraction barely moves."""
    def virga(nz):
        m, prod, surf, _, _ = _reevap_column(290.0, 0.45, k_prod=nz // 2, nz=nz)   # cloud base at mid-column
        return 1.0 - surf / prod
    v18, v36 = virga(18), virga(36)
    assert abs(v18 - v36) < 0.08                               # grid-convergent (not a per-layer geometric loss)


def test_precip_reevap_resume_roundtrip():
    """Checkpoint/resume with re-evaporation ON: the scheme adds NO prognostic state, so state()/load_state()
    round-trips and a run continued from a reloaded checkpoint is BIT-IDENTICAL to the uninterrupted run (the
    surface-precip flux is a per-step diagnostic, re-derived from q/T/precip_qc each step -- nothing extra to
    checkpoint). Pure column physics (run_dynamics=False) so the comparison isolates the re-evap state question
    from the dynamics' uncheckpointed layer-height cache ``_z`` (a pre-existing core property, not re-evap)."""
    cfg = dict(nx=16, nz=16, theta_max_deg=55.0, prognostic_condensate=True, precip_reevap=True)
    a = StageADCore(StageADConfig(**cfg)); a.run_dynamics = False
    for _ in range(20):
        a.step()
    st = a.state()
    b = StageADCore(StageADConfig(**cfg)); b.run_dynamics = False
    b.load_state(st)
    for _ in range(15):                                         # continue both from the same checkpoint
        a.step(); b.step()
    assert np.array_equal(a.q, b.q) and np.array_equal(a.T_atm, b.T_atm)      # bit-identical continuation
    assert np.array_equal(a._precip, b._precip)                # surface precip re-derived identically
