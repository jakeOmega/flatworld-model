"""Acceptance tests for the salt-aerosol CCN coupling (GitHub #237, AD-core side).

A prescribed cloud-droplet-number field ``N_d`` couples into (1) the Twomey cloud-albedo limb
(``radiation.cloud_optical_properties`` effective radius) and (2) the warm-rain autoconversion
efficiency, plus an optional decoupled salt-haze SW-dimming limb. Default OFF -> byte-identical to the
pre-feature core. The clean zero-point is ``N_d == salt_ccn_nd_ref`` (only the ratio N_d/N_ref enters
both limbs), so the master ON at the reference is a no-op. These tests are fast (low resolution,
pure-column physics where possible).
"""
import numpy as np
import pytest
from flatworld import constants as C
from flatworld import device as dev
from flatworld import radiation as rad
from flatworld.stage_ad_core import StageADCore, StageADConfig

# salt_ccn requires the prognostic-condensate micro path (the autoconversion + LWP machinery).
PROG = dict(prognostic_condensate=True)

gpu = pytest.mark.skipif(not dev.gpu_available(),
                         reason="no CuPy/CUDA device (optional GPU backend: requirements-gpu.txt)")


def _relerr(a, b):
    a = np.asarray(dev.to_host(a), dtype=np.float64)
    b = np.asarray(dev.to_host(b), dtype=np.float64)
    den = np.maximum(np.abs(a), np.abs(b))
    den = np.where(den < 1e-30, 1.0, den)
    return float(np.max(np.abs(a - b) / den))


def _base(**kw):
    return dict(nx=16, nz=16, theta_max_deg=55.0, **kw)


# --------------------------------------------------------------- config defaults + construction guards
def test_salt_ccn_defaults_off_and_field_absent():
    """Master OFF by default; the sub-limbs default ON-when-master-ON (mirroring the prognostic_condensate
    sub-flag convention); the decoupled haze limb defaults OFF. No N_d field is built when the master is OFF."""
    cfg = StageADConfig(**_base())
    assert cfg.salt_ccn is False
    assert cfg.salt_ccn_twomey is True
    assert cfg.salt_ccn_autoconv is True
    assert cfg.salt_ccn_haze is False
    core = StageADCore(cfg)
    assert core.N_d is None


def test_salt_ccn_requires_prognostic_condensate():
    with pytest.raises(ValueError):
        StageADCore(StageADConfig(salt_ccn=True, prognostic_condensate=False, **_base()))


def test_salt_ccn_mutually_exclusive_with_equatorial_precip_diagnostics():
    with pytest.raises(ValueError):
        StageADCore(StageADConfig(salt_ccn=True, chimney_param=True, **PROG, **_base()))
    with pytest.raises(ValueError):
        StageADCore(StageADConfig(salt_ccn=True, equator_norain_cells=2, **PROG, **_base()))


# --------------------------------------------------------------- N_d field construction
def test_salt_ccn_builds_uniform_nd_field():
    cfg = StageADConfig(salt_ccn=True, salt_ccn_nd=500.0, salt_ccn_nd_ref=100.0, **PROG, **_base())
    core = StageADCore(cfg)
    assert core.N_d is not None
    nd = np.asarray(core.N_d)
    assert nd.shape == (cfg.nx,)
    assert np.allclose(nd, 500.0)


def test_salt_ccn_band_only_loads_only_the_band():
    """Band-masked mode loads N_d only within [theta_lo, theta_hi]; elsewhere it stays at the reference
    (clean) value -- this isolates band-local microphysics from upstream convective-source loading."""
    cfg = StageADConfig(salt_ccn=True, salt_ccn_band_only=True, salt_ccn_nd=500.0, salt_ccn_nd_ref=100.0,
                        salt_ccn_band_theta_lo=40.0, salt_ccn_band_theta_hi=55.0, **PROG, **_base())
    core = StageADCore(cfg)
    nd = np.asarray(core.N_d)
    th = np.asarray(core.theta_deg)
    inband = (th >= 40.0) & (th <= 55.0)
    assert np.allclose(nd[inband], 500.0)
    assert np.allclose(nd[~inband], 100.0)


# --------------------------------------------------------------- byte-identity at the reference (no-op)
def test_salt_ccn_on_at_reference_byte_identical_cpu():
    """On the CPU-default numpy-generic micro path, the master ON at N_d == N_ref is BYTE-IDENTICAL to the
    master OFF: the Twomey ratio (N_ref/N_d)^(1/3) and the autoconv ratio (N_d/N_ref)^beta are both exactly
    1.0 at the reference, so the salt code adds no perturbation. This is the load-bearing 'clean zero-point'
    guard -- it must hold as the Twomey + autoconv limbs are wired in."""
    base = _base(clouds=True, **PROG)
    off = StageADCore(StageADConfig(salt_ccn=False, **base))
    on = StageADCore(StageADConfig(salt_ccn=True, salt_ccn_nd=100.0, salt_ccn_nd_ref=100.0,
                                   salt_ccn_autoconv_beta=-0.8, **base))
    for m in (off, on):
        m.run_dynamics = False
    for _ in range(20):
        off.step(); on.step()
    assert np.array_equal(off.T_atm, on.T_atm)
    assert np.array_equal(off.q, on.q)
    assert np.array_equal(off.T_surf, on.T_surf)
    assert np.array_equal(off._precip, on._precip)


# --------------------------------------------------------------- Twomey cloud-albedo limb (radiation.py)
def _overcast_deck(nz=24, nx=2, T_surf=300.0):
    """A saturated overcast multi-column for the cloud-optics unit tests (mirrors test_clouds._column)."""
    from flatworld import thermo as th
    p_edge = np.linspace(5.0e3, C.P_SURF, nz + 1)
    p_mid = 0.5 * (p_edge[:-1] + p_edge[1:])
    z = -8000.0 * np.log(p_mid / C.P_SURF)
    T1 = np.maximum(T_surf - 6.5e-3 * z, 150.0)
    T = np.tile(T1, (nx, 1))
    f = np.ones((nx, nz))
    dm = rad.layer_mass(p_edge)
    return f, T, p_mid, dm


def test_twomey_at_reference_is_noop():
    """n_d == n_d_ref makes the Twomey ratio (N_ref/N_d)^(1/3) exactly 1.0 -> r_eff unchanged -> the cloud
    optics are BYTE-IDENTICAL to the no-n_d (constant r_eff) path. The clean zero-point for the radiation limb."""
    f, T, p_mid, dm = _overcast_deck()
    n_ref = 100.0
    n_d = np.full((f.shape[0], 1), n_ref)
    base = rad.cloud_optical_properties(f, T, p_mid, dm)
    ref = rad.cloud_optical_properties(f, T, p_mid, dm, n_d=n_d, n_d_ref=n_ref)
    for a, b in zip(base, ref):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_twomey_loaded_brightens_liquid_cloud():
    """More CCN (higher N_d) -> smaller LIQUID r_eff -> larger SW optical depth -> larger in-cloud reflectance
    R_in. Direction-firm (the Twomey effect). Evaluated on the warm (liquid) low deck."""
    f, T, p_mid, dm = _overcast_deck(T_surf=300.0)
    n_ref = 100.0
    clean = rad.cloud_optical_properties(f, T, p_mid, dm, n_d=np.full((f.shape[0], 1), n_ref), n_d_ref=n_ref)
    loaded = rad.cloud_optical_properties(f, T, p_mid, dm, n_d=np.full((f.shape[0], 1), 10 * n_ref), n_d_ref=n_ref)
    R_clean, R_loaded = np.asarray(clean[1]), np.asarray(loaded[1])
    liquid = np.asarray(T) >= C.T_FREEZE                              # liquid layers only (warm deck)
    cloudy = liquid & (R_clean > 1e-6)
    assert cloudy.any()
    assert np.all(R_loaded[cloudy] > R_clean[cloudy])                # brighter where there is liquid cloud


def test_twomey_leaves_ice_cloud_untouched():
    """Salt is a WARM-rain CCN: the ice r_eff (CLOUD_REFF_ICE) is untouched, so a fully-glaciated (cold) deck
    is byte-identical regardless of N_d."""
    f, T, p_mid, dm = _overcast_deck(T_surf=200.0)                    # cold -> all ice
    assert np.all(np.asarray(T) < C.T_FREEZE)
    n_ref = 100.0
    clean = rad.cloud_optical_properties(f, T, p_mid, dm, n_d=np.full((f.shape[0], 1), n_ref), n_d_ref=n_ref)
    loaded = rad.cloud_optical_properties(f, T, p_mid, dm, n_d=np.full((f.shape[0], 1), 10 * n_ref), n_d_ref=n_ref)
    assert np.array_equal(np.asarray(clean[1]), np.asarray(loaded[1]))   # ice cloud unchanged by CCN


def test_twomey_core_loaded_raises_cloud_albedo():
    """End-to-end through the core: a salt-LOADED core (high N_d) has a HIGHER column cloud SW albedo than the
    clean core (N_d == N_ref), with the autoconv limb OFF so this isolates the Twomey radiative thread."""
    base = _base(clouds=True, **PROG)
    common = dict(salt_ccn=True, salt_ccn_autoconv=False, salt_ccn_nd_ref=100.0, **base)
    clean = StageADCore(StageADConfig(salt_ccn_nd=100.0, **common))
    loaded = StageADCore(StageADConfig(salt_ccn_nd=2000.0, **common))
    for m in (clean, loaded):
        m.run_dynamics = False
    for _ in range(30):
        clean.step(); loaded.step()
    a_clean = np.asarray(clean._radiate(clean.q)["cloud_sw_albedo"])
    a_loaded = np.asarray(loaded._radiate(loaded.q)["cloud_sw_albedo"])
    assert a_loaded.mean() > a_clean.mean()                              # brighter clouds under salt loading


# --------------------------------------------------------------- warm-rain autoconversion limb (sign-ambiguous)
def _qc_ice(nx=16, nz=16, qc_val=2.0e-3, ice=False):
    qc = np.full((nx, nz), qc_val)
    mask = np.full((nx, nz), ice, dtype=bool)
    return qc, mask


def _autoconv_core(**kw):
    return StageADCore(StageADConfig(salt_ccn=True, salt_ccn_nd_ref=100.0, **PROG, **_base(), **kw))


def test_autoconv_at_reference_is_noop():
    """salt-ON autoconversion at N_d == N_ref scales the demand by (N_ref/N_ref)^beta == 1.0 -> the demand is
    byte-identical to salt-OFF (the clean zero-point for the warm-rain limb)."""
    off = StageADCore(StageADConfig(salt_ccn=False, **PROG, **_base()))
    on = _autoconv_core(salt_ccn_nd=100.0, salt_ccn_autoconv_beta=-0.8)
    qc, ice = _qc_ice()
    d_off = np.asarray(off._autoconversion_demand(qc, ice, 60.0))
    d_on = np.asarray(on._autoconversion_demand(qc, ice, 60.0))
    assert np.array_equal(d_off, d_on)


def test_autoconv_suppression_lowers_warm_rain():
    """beta<0 SUPPRESSION: more CCN -> smaller, more numerous drops -> delayed coalescence -> LESS warm rain.
    The loaded demand is the clean demand times (N_d/N_ref)^beta."""
    clean = _autoconv_core(salt_ccn_nd=100.0, salt_ccn_autoconv_beta=-0.8)
    loaded = _autoconv_core(salt_ccn_nd=1000.0, salt_ccn_autoconv_beta=-0.8)
    qc, ice = _qc_ice()
    d_clean = np.asarray(clean._autoconversion_demand(qc, ice, 60.0))
    d_loaded = np.asarray(loaded._autoconversion_demand(qc, ice, 60.0))
    assert np.all(d_loaded < d_clean)
    assert np.allclose(d_loaded, d_clean * (10.0 ** -0.8))               # the (N_d/N_ref)^beta factor


def test_autoconv_invigoration_raises_warm_rain():
    """beta>0 INVIGORATION: giant salt CCN seed large drops -> MORE warm rain."""
    clean = _autoconv_core(salt_ccn_nd=100.0, salt_ccn_autoconv_beta=0.5)
    loaded = _autoconv_core(salt_ccn_nd=1000.0, salt_ccn_autoconv_beta=0.5)
    qc, ice = _qc_ice()
    d_clean = np.asarray(clean._autoconversion_demand(qc, ice, 60.0))
    d_loaded = np.asarray(loaded._autoconversion_demand(qc, ice, 60.0))
    assert np.all(d_loaded > d_clean)


def test_autoconv_leaves_ice_phase_untouched():
    """Salt is a WARM-rain CCN: the ICE autoconversion (snow aggregation) is untouched by N_d."""
    clean = _autoconv_core(salt_ccn_nd=100.0, salt_ccn_autoconv_beta=-0.8)
    loaded = _autoconv_core(salt_ccn_nd=5000.0, salt_ccn_autoconv_beta=-0.8)
    qc, ice = _qc_ice(qc_val=2.0e-3, ice=True)                           # fully glaciated
    d_clean = np.asarray(clean._autoconversion_demand(qc, ice, 60.0))
    d_loaded = np.asarray(loaded._autoconversion_demand(qc, ice, 60.0))
    assert np.array_equal(d_clean, d_loaded)                             # ice snow rate unchanged by CCN


def test_autoconv_salt_routes_to_generic_path_under_parallel():
    """salt_ccn_autoconv excludes the config from the _micro_canonical predicate, so the salt scaling is
    applied even when thermo_other_parallel would otherwise pick the (salt-UNAWARE) numba fuse. The parallel
    result must therefore MATCH the serial numpy-generic result (both apply the salt scaling) -- if the
    exclusion were missing, the numba fuse would silently ignore N_d and the two would diverge."""
    canon = dict(prognostic_condensate=True, clouds=True, precip_reevap=True, salt_ccn=True,
                 salt_ccn_nd=1000.0, salt_ccn_nd_ref=100.0, salt_ccn_autoconv_beta=-0.8, **_base())
    serial = StageADCore(StageADConfig(thermo_other_parallel=False, **canon))
    parallel = StageADCore(StageADConfig(thermo_other_parallel=True, **canon))
    for m in (serial, parallel):
        m.run_dynamics = False
    for _ in range(15):
        serial.step(); parallel.step()
    assert np.allclose(np.asarray(serial._precip), np.asarray(parallel._precip), rtol=1e-9, atol=0)
    assert np.allclose(serial.q_c, parallel.q_c, rtol=1e-9, atol=0)


def test_autoconv_microphysics_conserves_water_under_loading():
    """The salt limb only rescales how much condensate rains -- it must NOT create or destroy water. The local
    invariant of one microphysics step (no evaporation/advection involved): the column total water (vapour +
    condensate) decreases by EXACTLY the precip it produced. Holds for any beta/N_d (salt moves q_c->precip,
    a conservative transfer)."""
    core = _autoconv_core(clouds=True, salt_ccn_nd=2000.0, salt_ccn_autoconv_beta=-0.8)
    core.run_dynamics = False
    core.q_c = np.full_like(core.q_c, 1.5e-3)                            # seed retained condensate
    q = np.asarray(core.q).copy()
    dm = np.asarray(core.dm)
    w0 = q @ dm + np.asarray(core.q_c) @ dm
    q2 = core._prognostic_microphysics(q.copy(), 60.0)                  # updates self.q_c, self._precip
    w1 = np.asarray(q2) @ dm + np.asarray(core.q_c) @ dm
    precip_mass = np.asarray(core._precip) * 60.0                        # surface flux * dt -> column mass [kg/m^2]
    assert np.all(np.asarray(core.q_c) >= 0.0)                          # positivity limiter intact
    assert np.allclose(w1 + precip_mass, w0, rtol=1e-10, atol=1e-12)   # exact water closure


# --------------------------------------------------------------- decoupled salt-haze SW-dimming limb
def _sw_inputs(nx=2, N=10, S=400.0):
    dtau = np.full((nx, N), 0.05)
    return np.full(nx, S), dtau, np.full(nx, 0.15)


def test_haze_off_is_byte_identical():
    """tau_haze=None (default) -> the SW path is byte-identical to the pre-feature shortwave."""
    S, dtau, a = _sw_inputs()
    base = rad.shortwave(S, dtau, a)
    none = rad.shortwave(S, dtau, a, tau_haze=None)
    for k in base:
        assert np.array_equal(np.asarray(base[k]), np.asarray(none[k]))


def test_haze_dims_surface_and_conserves_energy():
    """A salt haze (tau_haze>0) DIMS the surface SW and CONSERVES energy against the ORIGINAL incoming flux
    (the forward-scattered fraction re-emerges as diffuse-down, backscatter -> space, absorption -> aloft)."""
    S, dtau, a = _sw_inputs()
    base = rad.shortwave(S, dtau, a)
    haze = rad.shortwave(S, dtau, a, tau_haze=np.full(2, 0.3))
    assert np.all(np.asarray(haze["surf_down"]) < np.asarray(base["surf_down"]))     # net dimming
    total = (np.asarray(haze["surf_abs"]) + np.asarray(haze["H"]).sum(axis=-1)
             + np.asarray(haze["planetary_reflected"]))
    assert np.allclose(total, np.asarray(S), rtol=1e-12, atol=1e-9)                  # energy closes vs S_toa


def test_haze_monotone_in_loading():
    S, dtau, a = _sw_inputs()
    light = rad.shortwave(S, dtau, a, tau_haze=np.full(2, 0.1))
    heavy = rad.shortwave(S, dtau, a, tau_haze=np.full(2, 0.6))
    assert np.all(np.asarray(heavy["surf_down"]) < np.asarray(light["surf_down"]))   # more haze -> darker


def test_haze_default_off_and_independently_toggleable():
    """salt_ccn_haze defaults OFF and is DECOUPLED -- it can run with BOTH precip limbs off (Jake #237: a haze
    problem must not gate the precip deliverable)."""
    assert StageADConfig(salt_ccn=True, salt_ccn_nd=2000.0, **PROG, **_base()).salt_ccn_haze is False
    core = StageADCore(StageADConfig(salt_ccn=True, salt_ccn_haze=True, salt_ccn_twomey=False,
                                     salt_ccn_autoconv=False, salt_ccn_nd=2000.0, **PROG, **_base()))
    assert core.N_d is not None


def test_haze_core_loaded_dims_surface_sw():
    """End-to-end: a salt-LOADED core with ONLY the haze limb on has lower surface SW than the clean core."""
    base = _base(clouds=True, **PROG)
    common = dict(salt_ccn=True, salt_ccn_haze=True, salt_ccn_twomey=False, salt_ccn_autoconv=False,
                  salt_ccn_nd_ref=100.0, salt_ccn_haze_tau_ref=0.05, **base)
    clean = StageADCore(StageADConfig(salt_ccn_nd=100.0, **common))
    loaded = StageADCore(StageADConfig(salt_ccn_nd=5000.0, **common))
    for m in (clean, loaded):
        m.run_dynamics = False
    for _ in range(20):
        clean.step(); loaded.step()
    sw_clean = np.asarray(clean._radiate(clean.q)["surf_down_sw"]).mean()
    sw_loaded = np.asarray(loaded._radiate(loaded.q)["surf_down_sw"]).mean()
    assert sw_loaded < sw_clean                                          # salt haze dims the surface


# ------------------------------------------------------ haze tau ceiling (#237 AREA 1 roadmap item 1)
def test_haze_tau_cap_default_and_validation():
    """The salt-haze optical-depth ceiling defaults to a dust-storm O(several) value and must be > 0
    (inf recovers the bare unbounded linear form)."""
    assert StageADConfig(**_base()).salt_ccn_haze_tau_cap == 3.0
    with pytest.raises(ValueError):                              # validated in StageADCore.__init__
        StageADCore(StageADConfig(salt_ccn=True, salt_ccn_haze=True, salt_ccn_haze_tau_cap=0.0,
                                  salt_ccn_nd=2000.0, **PROG, **_base()))


def _haze_core(nd, tau_cap, tau_ref=0.02):
    base = _base(clouds=True, **PROG)
    return StageADCore(StageADConfig(
        salt_ccn=True, salt_ccn_haze=True, salt_ccn_twomey=False, salt_ccn_autoconv=False,
        salt_ccn_nd_ref=100.0, salt_ccn_haze_tau_ref=tau_ref, salt_ccn_nd=nd,
        salt_ccn_haze_tau_cap=tau_cap, **base))


def test_haze_tau_cap_bounds_source_dimming():
    """At SOURCE-class loading (N_d ~ 1e5 cm^-3) the bare linear tau ~ tens; the cap holds it to a
    dust-storm AOD, so the CAPPED core dims the surface SW LESS than the uncapped (tau=inf) one."""
    capped = _haze_core(1.0e5, tau_cap=3.0)
    uncapped = _haze_core(1.0e5, tau_cap=np.inf)
    sw_capped = np.asarray(capped._radiate(capped.q)["surf_down_sw"]).mean()
    sw_uncapped = np.asarray(uncapped._radiate(uncapped.q)["surf_down_sw"]).mean()
    assert sw_capped > sw_uncapped          # the cap REDUCES tau at source -> less dimming -> brighter surface


def test_haze_tau_cap_byte_identical_below_cap():
    """At BAND-class loading the linear tau (~2.4 here) is BELOW the dust-storm cap, so capping is a no-op:
    the capped and uncapped cores give byte-identical surface SW (the cap only bites at the source)."""
    nd_band = 5000.0                        # tau = 0.02*(50-1) = 2.45 < 3.0 cap
    capped = _haze_core(nd_band, tau_cap=3.0)
    uncapped = _haze_core(nd_band, tau_cap=np.inf)
    sw_capped = np.asarray(capped._radiate(capped.q)["surf_down_sw"])
    sw_uncapped = np.asarray(uncapped._radiate(uncapped.q)["surf_down_sw"])
    assert np.array_equal(sw_capped, sw_uncapped)


# --------------------------------------------------------------- GPU parity (CuPy); skips on a CPU-only box
@gpu
def test_twomey_gpu_matches_numpy():
    """The Twomey r_eff(N_d) thread is cupy-generic -> fp64 GPU matches numpy to machine precision."""
    import cupy as cp
    f, T, p_mid, dm = _overcast_deck(nz=20, nx=8, T_surf=300.0)
    n_d = np.full((f.shape[0], 1), 1500.0)
    args = (f, T, p_mid, dm)
    ref = rad.cloud_optical_properties(*args, n_d=n_d, n_d_ref=100.0, return_tau_sw=True)
    out = rad.cloud_optical_properties(*(cp.asarray(a) for a in args),
                                       n_d=cp.asarray(n_d), n_d_ref=100.0, return_tau_sw=True)
    assert dev.is_device(out[1]), "cloud_optical_properties silently fell back to host"
    for r, o in zip(ref, out):
        assert _relerr(r, o) < 1e-11


@gpu
def test_haze_gpu_matches_numpy():
    """The salt-haze SW dimming in shortwave is cupy-generic -> fp64 GPU matches numpy to machine precision."""
    import cupy as cp
    S, dtau, a = _sw_inputs(nx=8, N=20)
    tau_haze = np.full(8, 0.4)
    ref = rad.shortwave(S, dtau, a, tau_haze=tau_haze)
    out = rad.shortwave(cp.asarray(S), cp.asarray(dtau), cp.asarray(a), tau_haze=cp.asarray(tau_haze))
    assert dev.is_device(out["surf_down"]), "shortwave silently fell back to host"
    for k in ref:
        assert _relerr(ref[k], out[k]) < 1e-11, f"shortwave[{k}] GPU-vs-numpy mismatch"


@gpu
def test_salt_core_gpu_matches_cpu():
    """END-TO-END parity: a salt-loaded core (all three limbs ON) stepped on device='gpu' matches the CPU
    numpy-generic path to fp64 round-off. This is the load-bearing GPU correctness check -- the salt
    autoconv routes to the cupy-generic _prognostic_microphysics (un-fused, on-device) and must agree."""
    common = dict(salt_ccn=True, salt_ccn_nd=2000.0, salt_ccn_nd_ref=100.0, salt_ccn_autoconv_beta=-0.8,
                  salt_ccn_haze=True, clouds=True, **PROG, **_base())
    cpu = StageADCore(StageADConfig(device="cpu", **common))
    gpu_core = StageADCore(StageADConfig(device="gpu", **common))
    for m in (cpu, gpu_core):
        m.run_dynamics = False
    for _ in range(15):
        cpu.step(); gpu_core.step()
    for name in ("T_atm", "q", "q_c", "_precip", "T_surf"):
        assert _relerr(getattr(cpu, name), getattr(gpu_core, name)) < 1e-9, f"{name}: GPU-vs-CPU mismatch"
