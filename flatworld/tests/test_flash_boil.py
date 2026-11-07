"""#294 boiling-layer physics: flash-boil virga + boiling condensation-suppression.

Two default-OFF flags, both threaded into the fused numba/device micro kernels so the canonical
fused CPU/GPU fast path stays fused when they are ON:

  * ``virga_flash_boil`` (b, corrected by #294): a boiling layer (esat >= 0.999 p) cannot hold
    falling liquid AT ALL, so the below-cloud reevap evaporates the FULL falling flux there --
    bounded only by the flash-boil ENERGY cap boil_cap = (T - T_boil) cp dm/(L_VAP dt) (evaporative
    cooling stops at the layer's own boiling point), NOT by the saturation deficit (the pre-#294
    sat_cap bound, which is ~0 in a steam-saturated layer and let heavy rain through to a >100 C
    surface).
  * ``cond_boil_suppress`` (a, the upstream fix): above boiling there is no vapour-liquid
    equilibrium, so NO condensation fires and NO precip forms in boiling layers (PDF qc_eq -> 0 /
    grid-mean cond -> 0; precip_demand -> 0), and suspended q_c drains back to vapour -- advected
    moisture crosses the hot zone as vapour (the pass-through), instead of the qsat clamp
    fabricating spurious condensate.

Every test drives the physics DIRECTLY (``_below_cloud_reevap`` / ``_prognostic_microphysics`` on a
constructed column state) so the boiling regime is pinned exactly; the fused-kernel parity test then
locks the numba (and, when CuPy is present, the device) mirrors to the generic path at round-off.
"""
import dataclasses

import numpy as np
import pytest
from flatworld import device as _dev

from flatworld import constants as C
from flatworld import thermo as th
from flatworld.stage_ad_core import StageADConfig, StageADCore


def _core(**kw):
    """A minimal canonical-prognostic core for direct microphysics calls (no stepping)."""
    kw.setdefault("nx", 4)
    kw.setdefault("nz", 12)
    kw.setdefault("theta_max_deg", 70.0)
    kw.setdefault("prognostic_condensate", True)
    kw.setdefault("precip_reevap", True)
    return StageADCore(StageADConfig(**kw))


def _steam_column_state(m, nboil_from=None, T_hot=450.0):
    """(T, q) with a cool condensing top and a SUPER-BOILING, steam-saturated (q = qsat, deficit = 0)
    stack from layer ``nboil_from`` down -- the exact state where the pre-#294 sat_cap bound failed."""
    nz = m.cfg.nz
    k0 = nz // 3 if nboil_from is None else nboil_from
    T = np.full((m.cfg.nx, nz), 280.0)
    T[:, k0:] = T_hot
    qs = np.asarray(th.qsat(T, m.p_mid[None, :]))
    q = qs.copy()                                          # SATURATED everywhere: deficit = 0 in every layer
    #   (cool top too, so the deficit-limited bulk reevap is inert there and the OFF/ON contrast below
    #    isolates the BOILING stack -- the only place the flash-boil bound differs)
    assert np.all(np.asarray(th.esat(T[:, k0:])) >= 0.999 * m.p_mid[None, k0:])   # genuinely boiling
    assert np.all(np.asarray(th.esat(T[:, :k0])) < 0.999 * m.p_mid[None, :k0])    # top genuinely not
    return T, q, k0


def test_flags_default_off():
    cfg = StageADConfig()
    assert cfg.virga_flash_boil is False
    assert cfg.cond_boil_suppress is False


# ---------------------------------------------------------------- (b) flash-boil virga, full flash
def test_flash_boil_steam_column_flashes_full_flux():
    """The #294 failure state: rain falls into a steam-saturated (deficit = 0) boiling stack.
    Pre-fix (flash OFF here, since OFF reduces to the deficit-limited bulk rate): the deficit-zero
    stack evaporates NOTHING and the full flux lands on the hot surface. Fixed (flash ON): the flux
    flash-evaporates COMPLETELY in the first boiling layer (the energy cap is huge at 450 K), the
    surface stays dry, water and energy close to machine precision, and no layer cools below its
    own boiling point."""
    F = 235.0 / 86400.0                                    # the #294 theta0 flux [kg/m^2/s]
    dt = 60.0
    out = {}
    for flash in (False, True):
        m = _core(virga_flash_boil=flash)
        T, q0, k0 = _steam_column_state(m)
        m.T_atm = T.copy()
        precip_qc = np.zeros_like(q0)
        precip_qc[0, 2] = F * dt / m.dm[2]                 # rain forms in the cool top (sub-boiling)
        q = q0.copy()
        P = m._below_cloud_reevap(q, precip_qc, dt, src_ice=np.zeros(q0.shape, bool))
        out[flash] = (np.asarray(P).copy(), q, m.T_atm.copy())
    P_off, _, _ = out[False]
    P_on, q_on, T_on = out[True]
    assert P_off[0] == pytest.approx(F, rel=1e-12)         # OFF: deficit=0 -> rain punches through
    assert P_on[0] == 0.0                                  # ON: complete flash, dry surface
    m = _core(virga_flash_boil=True)
    T, q0, k0 = _steam_column_state(m)
    # water closure: column vapour gain + surface flux == the injected source
    dW = np.sum((q_on[0] - q0[0]) * m.dm) + P_on[0] * dt
    assert dW == pytest.approx(F * dt, rel=1e-12)
    # energy conjugacy: cp * column dT == -L_VAP * column dq (warm column: no ice, no melt)
    dE = C.CP_AIR * np.sum((T_on[0] - T[0]) * m.dm) + C.L_VAP * np.sum((q_on[0] - q0[0]) * m.dm)
    assert abs(dE) < 1e-6 * C.L_VAP * F * dt
    # the energy-cap floor: no boiling layer cooled below its own T_boil
    Tb = np.asarray(th.T_boil(m.p_mid))
    assert np.all(T_on[0, k0:] >= Tb[k0:] - 1e-9)
    # untouched columns byte-identical (the flash acted only where flux fell)
    assert np.array_equal(q_on[1:], q0[1:])
    assert np.array_equal(T_on[1:], T[1:])


def test_flash_boil_energy_cap_partial_flash():
    """A barely-superheated boiling stack (T = T_boil + eps), SATURATED so the deficit-rate arm of the
    max() floor is zero and the pure energy-cap semantics are isolated: each layer can only
    flash-evaporate boil_cap = eps cp dm/(L_VAP dt) before landing at its own boiling point, so a
    heavy flux is only PARTIALLY flashed -- the surviving surface flux is exactly F - sum(boil_cap),
    and every flashed layer lands exactly at T_boil (the flash's blow-up guard: bounded dq, bounded
    cooling). (On a MOIST stack the rate arm takes over below/at T_boil -- see the floor test.)"""
    dt = 60.0
    eps = 0.05                                             # [K] superheat above the boiling point
    m = _core(virga_flash_boil=True)
    nz = m.cfg.nz
    k0 = nz // 3
    Tb = np.asarray(th.T_boil(m.p_mid))
    T = np.full((m.cfg.nx, nz), 280.0)
    T[:, k0:] = Tb[None, k0:] + eps                        # just above boiling -> tiny energy cap
    qs = np.asarray(th.qsat(T, m.p_mid[None, :]))
    q0 = qs.copy()                                         # saturated everywhere: only the ENERGY cap evaporates
    caps = eps * C.CP_AIR * m.dm[k0:] / (C.L_VAP * dt)     # per-layer boil_cap [kg/m^2/s]
    F = 2.0 * float(caps.sum())                            # flux twice what the stack can flash
    m.T_atm = T.copy()
    precip_qc = np.zeros_like(q0)
    precip_qc[0, 2] = F * dt / m.dm[2]
    q = q0.copy()
    P = m._below_cloud_reevap(q, precip_qc, dt, src_ice=np.zeros(q0.shape, bool))
    assert float(P[0]) == pytest.approx(F - float(caps.sum()), rel=1e-10)   # exactly the un-flashable rest
    assert np.allclose(m.T_atm[0, k0:], Tb[k0:], atol=1e-9)                 # every layer landed AT T_boil
    assert np.all(q[0, k0:] - q0[0, k0:] <= eps * C.CP_AIR / C.L_VAP + 1e-15)   # dq bounded by the cap


def test_flash_boil_floor_keeps_rate_limited_arm_moist_stack():
    """The PR #296 review fix: the energy cap is a max() FLOOR over the ordinary deficit-limited cap,
    not a replacement. Two consequences, both pinned here on the REACHABLE moist state (q = 0.02,
    deficit large -- not the steam-saturated state):

    (1) single step, barely-superheated boiling stack (boil_cap ~ 0 << the K_e rate): flash ON is
        BIT-IDENTICAL to OFF -- the rate arm wins the max(), handing back the flag-OFF closure (the
        replace form would have zeroed the evaporation and passed the flux through);
    (2) iterated (the review's pinning A/B, no reheating): a hot boiling stack absorbs a theta0-scale
        flux indefinitely with ON exactly as with OFF -- rain-cooled layers cross T_boil and EXIT the
        boiling mask into ordinary sub-boiling virga instead of pinning at T_boil as a permanent
        pass-through (the replace form converged to ~99% per-step surface pass-through here)."""
    dt = 60.0
    F = 235.0 / 86400.0

    # --- (1) single step at eps superheat: rate >> boil_cap -> max() hands back the OFF closure ---
    out = {}
    for flash in (False, True):
        m = _core(virga_flash_boil=flash)
        nz = m.cfg.nz
        k0 = nz // 3
        Tb = np.asarray(th.T_boil(m.p_mid))
        T = np.full((m.cfg.nx, nz), 280.0)
        T[:, k0:] = Tb[None, k0:] + 0.05                   # barely superheated: boil_cap tiny
        q0 = np.full(T.shape, 0.02)                        # moist but far from the ~0.998 clamp qsat
        m.T_atm = T.copy()
        pqc = np.zeros_like(q0)
        pqc[0, 2] = F * dt / m.dm[2]
        q = q0.copy()
        P = m._below_cloud_reevap(q, pqc, dt, src_ice=np.zeros(q0.shape, bool))
        out[flash] = (np.asarray(P).copy(), q, m.T_atm.copy())
    for a, b in zip(out[False], out[True]):
        assert np.array_equal(a, b)                        # rate-dominated regime: ON == OFF to the bit

    # --- (2) iterated pinning A/B: ON absorbs like OFF; layers exit the boiling mask ---
    res = {}
    for flash in (False, True):
        m = _core(virga_flash_boil=flash)
        nz = m.cfg.nz
        k0 = nz // 3
        T = np.full((m.cfg.nx, nz), 280.0)
        T[:, k0:] = 450.0                                  # hot boiling stack; flash active initially
        q = np.full(T.shape, 0.02)
        m.T_atm = T.copy()
        pqc = np.zeros_like(q)
        pqc[0, 2] = F * dt / m.dm[2]
        cum = 0.0
        for _ in range(400):                               # ~7 h of steady theta0-scale rain, no reheat
            P = m._below_cloud_reevap(q, pqc.copy(), dt, src_ice=np.zeros(T.shape, bool))
            cum += float(P[0])
        Tb = np.asarray(th.T_boil(m.p_mid))
        res[flash] = (cum / (400 * F), float((m.T_atm[0, k0:] - Tb[k0:]).min()))
    assert res[True][0] < 1e-9                             # no pass-through builds up (matches OFF)
    assert res[True][0] <= res[False][0] + 1e-12           # ON evaporation >= OFF (the floor guarantee)
    assert res[True][1] < 0.0                              # rain-cooled layers EXITED the boiling mask


# ------------------------------------------------- (a) boiling condensation-suppression (upstream)
def test_cond_boil_suppress_kills_condensation_and_drains_qc():
    """A steam-heavy boiling column (q ~ 0.95 qsat: inside the PDF tail, so the clamp-qsat
    microphysics would condense). OFF: spurious condensate GROWS in the boiling layers (the #294
    upstream artifact). ON: no condensation, no precip forms in boiling layers, and the suspended
    q_c drains back to vapour, closing water and energy to machine precision."""
    dt = 60.0
    out = {}
    for suppress in (False, True):
        m = _core(cond_boil_suppress=suppress)
        nz = m.cfg.nz
        T = np.full((m.cfg.nx, nz), 420.0)                 # the WHOLE column super-boiling
        qs = np.asarray(th.qsat(T, m.p_mid[None, :]))
        q0 = 0.95 * qs                                     # near-steam: in the PDF tail
        qc0 = np.full(T.shape, 5e-4)                       # suspended condensate that must drain
        m.T_atm = T.copy()
        m.q_c = qc0.copy()
        q = m._prognostic_microphysics(q0.copy(), dt)
        out[suppress] = (m.q_c.copy(), np.asarray(m._precip).copy(), q, m.T_atm.copy())
    qc_off, _, _, _ = out[False]
    qc_on, P_on, q_on, T_on = out[True]
    m = _core(cond_boil_suppress=True)
    T = np.full((m.cfg.nx, m.cfg.nz), 420.0)
    qs = np.asarray(th.qsat(T, m.p_mid[None, :]))
    q0 = 0.95 * qs
    qc0 = np.full(T.shape, 5e-4)
    # OFF: the clamp-qsat spurious condensation NET-GROWS the column condensate (upper layers pile it;
    # the retained ceiling force-rains some lower layers -- the spurious equatorial precip channel)
    assert np.sum(qc_off[0] * m.dm) > np.sum(qc0[0] * m.dm)
    assert np.all(qc_on == 0.0)                            # ON: suspended q_c fully drained (instantaneous tau)
    assert np.all(P_on == 0.0)                             # ON: no precip forms in boiling layers
    assert np.allclose(q_on - q0, qc0, rtol=1e-12)         # ...it went back to vapour
    # water: d(q + q_c) + P dt == 0; energy: cp dT == L_VAP d(q_c) (evaporative cooling of the drain)
    dW = np.sum((q_on[0] - q0[0] + qc_on[0] - qc0[0]) * m.dm) + P_on[0] * dt
    assert abs(dW) < 1e-12 * np.sum(q0[0] * m.dm)
    dE = C.CP_AIR * np.sum((T_on[0] - T[0]) * m.dm) + C.L_VAP * np.sum((q_on[0] - q0[0]) * m.dm)
    assert abs(dE) < 1e-6 * C.L_VAP * np.sum(qc0[0] * m.dm)


def test_boiling_flags_are_noops_on_a_subboiling_state():
    """Locality (the 'off/non-boiling columns byte-identical' #294 gate): on a state with NO boiling
    layer, both flags ON are BYTE-IDENTICAL to OFF through the full generic microphysics+reevap --
    the masks are empty, so the shipped arithmetic is untouched."""
    rng = np.random.default_rng(11)
    dt = 60.0
    m_ref = _core()
    nz = m_ref.cfg.nz
    T = (300.0 - np.linspace(0, 90, nz))[None, :] + rng.standard_normal((m_ref.cfg.nx, nz)) * 4.0
    assert np.all(np.asarray(th.esat(T)) < 0.999 * m_ref.p_mid[None, :])   # nothing boils
    qs = np.asarray(th.qsat(T, m_ref.p_mid[None, :]))
    q0 = qs * (0.5 + 0.8 * rng.random(T.shape))            # sub- and super-saturated cells
    qc0 = np.abs(rng.standard_normal(T.shape)) * 2e-4
    out = {}
    for flags in (False, True):
        m = _core(virga_flash_boil=flags, cond_boil_suppress=flags)
        m.T_atm = T.copy()
        m.q_c = qc0.copy()
        q = m._prognostic_microphysics(q0.copy(), dt)
        out[flags] = (q, m.T_atm.copy(), m.q_c.copy(), np.asarray(m._precip).copy())
    for a, b in zip(out[False], out[True]):
        assert np.array_equal(a, b)                        # byte-identical, not just close


# ------------------------------------------------------------- fused-kernel parity (the fast path)
def test_micro_fuse_flags_on_matches_generic_roundoff():
    """The fused numba kernel (and the CuPy device kernel when present) reproduces the generic
    ``_prognostic_microphysics`` to round-off with BOTH #294 flags ON, on a mixed state that
    includes steam-saturated super-boiling columns -- the gate that lets ``_micro_canonical`` keep
    the fused CPU/GPU fast path with the boiling physics enabled. NON-VACUOUS: the flags visibly
    move the state vs the flags-off baseline (a kernel ignoring the new int args would fail)."""
    pytest.importorskip("numba")
    import flatworld.stage_ad_core as adc
    import flatworld._gpu_kernels as adc_gpu
    from flatworld.stage_ad_core import prognostic_microphysics_reevap_parallel

    rng = np.random.default_rng(7)
    nx, nz = 24, 16
    cfg = StageADConfig(nx=nx, nz=nz, theta_max_deg=80.0, prognostic_condensate=True,
                        precip_reevap=True, virga_flash_boil=True, cond_boil_suppress=True)
    m = StageADCore(cfg)
    p = m.p_mid
    T = (300.0 - np.linspace(0, 90, nz))[None, :] + rng.standard_normal((nx, nz)) * 4.0
    T[:6, nz // 3:] += 250.0                               # 6 super-boiling steam columns
    qs = np.asarray(th.qsat(T, p[None, :]))
    q0 = qs * (0.5 + 0.8 * rng.random((nx, nz)))
    q0[:6, nz // 3:] = qs[:6, nz // 3:] * (0.98 + 0.04 * rng.random((6, nz - nz // 3)))
    qc0 = np.abs(rng.standard_normal((nx, nz))) * 2e-4
    dt = 2.0                                               # a production-scale thermo substep dt [s]

    ref = StageADCore(cfg)                                 # generic numpy reference
    ref.T_atm = T.copy(); ref.q_c = qc0.copy()
    q_np = ref._prognostic_microphysics(q0.copy(), dt)
    T_np, qc_np, P_np = ref.T_atm.copy(), ref.q_c.copy(), np.asarray(ref._precip).copy()

    base = StageADCore(dataclasses.replace(cfg, virga_flash_boil=False, cond_boil_suppress=False))
    base.T_atm = T.copy(); base.q_c = qc0.copy()
    base._prognostic_microphysics(q0.copy(), dt)
    assert np.max(np.abs(T_np - base.T_atm)) > 1e-3        # NON-VACUOUS: the flags move the state

    m.T_atm = T.copy(); m.q_c = qc0.copy()
    q_nb = prognostic_microphysics_reevap_parallel(m, q0.copy(), dt)
    assert adc._MICRO_NUMBA_KERNEL is not None             # the @njit kernel genuinely ran
    assert np.max(np.abs(T_np - m.T_atm)) < 1e-9
    assert np.max(np.abs(q_np - q_nb)) < 1e-12
    assert np.max(np.abs(qc_np - m.q_c)) < 1e-12
    assert np.max(np.abs(P_np - m._precip)) < 1e-12

    # gpu_available(), not _HAVE_CUPY (which is only "the wheel is installed"): the numba CPU half
    # above runs without a card and must keep running, so only the DEVICE half is gated.
    if _dev.gpu_available():                               # the actual GPU perf path, fp64 parity dtype
        import cupy as cp
        from flatworld._gpu_kernels import build_micro_cols_kernel, micro_cols_device
        kernel = build_micro_cols_kernel(cp.float64)
        Td = cp.ascontiguousarray(cp.asarray(T, cp.float64))
        qd = cp.ascontiguousarray(cp.asarray(q0, cp.float64))
        qcd = cp.ascontiguousarray(cp.asarray(qc0, cp.float64))
        dmd = cp.ascontiguousarray(cp.asarray(m.dm, cp.float64))
        pmd = cp.ascontiguousarray(cp.asarray(m.p_mid, cp.float64))
        P_dev = micro_cols_device(kernel, Td, qd, qcd, dmd, pmd, float(dt),
                                  cfg.cloud_rh_crit, cfg.qc_autoconv_tau, cfg.qc_autoconv_crit,
                                  cfg.qc_autoconv_tau_ice, cfg.qc_autoconv_crit_ice, cfg.cloud_f_min,
                                  cfg.precip_reevap_rate, virga_flash_boil=1, boil_suppress=1)
        assert np.max(np.abs(T_np - cp.asnumpy(Td))) < 1e-9
        assert np.max(np.abs(q_np - cp.asnumpy(qd))) < 1e-11
        assert np.max(np.abs(qc_np - cp.asnumpy(qcd))) < 1e-11
        assert np.max(np.abs(P_np - cp.asnumpy(P_dev))) < 1e-11
