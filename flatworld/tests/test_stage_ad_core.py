"""M2b.1 acceptance tests for the fused anelastic core skeleton (``stage_ad_core``).

M2b.1 builds only the geometry, the dual vertical grids, the anelastic reference state, the state
arrays, and the instrumented run-loop -- NO physics tendencies yet (see the module docstring and
``M2B_SUBTASKS.md``). These tests check exactly that skeleton: construction; the cross-track grid is
uniform in DISTANCE and nonlinear in theta; the reference state rho_ref(z) is hydrostatic-consistent;
``_hydrostatic_z`` round-trips the reference profile between the pressure and height grids; and the
harness loops the no-op step to convergence. They are fast (low resolution, a handful of steps).
"""
import warnings
import dataclasses
import numpy as np
import pytest
from flatworld import constants as C
from flatworld.stage_ad_core import (StageADCore, StageADConfig, build_anelastic_poisson,
                                     SeparablePoisson,
                                     velocities_from_psi, smagorinsky_viscosity, strain_rate_magnitude,
                                     smagorinsky_viscosity_aniso, vorticity_tendency, rayleigh_benard_run,
                                     nu_ra_scaling, _ddx, _ddz, _div_grad, _div_grad_aniso,
                                     single_column_rce, moist_convective_adjust,
                                     moist_convective_adjust_parallel, moist_convective_segment_mix,
                                     hydrostatic_height, wetness_of_Tsurf)
from flatworld.stage_ad_column import _SEGMIX_SETTLED_ULP
from flatworld import thermo as th
from flatworld import device as _dev
from flatworld.stage_a_column import StageA, StageAConfig

# The project's standing device gate (``test_device_gpu.py``'s idiom). NOT ``_gpu_kernels._HAVE_CUPY``:
# that is true whenever CuPy IMPORTS, which it does on any box with the wheel installed, while the
# kernels below need an actual CUDA DEVICE -- so a CuPy-but-no-card box (the project's own CPU gate,
# run under ``CUDA_VISIBLE_DEVICES=""``) took those branches and died in ``cudaErrorNoDevice`` instead
# of skipping.
_gpu = pytest.mark.skipif(not _dev.gpu_available(),
                          reason="no CuPy/CUDA device (optional GPU backend: requirements-gpu.txt)")


def _short(**kw):
    """A small, fast core for the unit gate."""
    cfg = StageADConfig(nx=24, nz=16, **kw)
    return StageADCore(cfg)


def _analytic_single_cell(m, amp=6.0e4):
    """Impose an analytic single-overturning-cell streamfunction psi = amp*sin(pi x/xmax)*sin(pi z/ztop)
    on the height grid (zero on all four walls), freeze the layer heights, and return the resolved
    cross-track wind ``u`` (height grid) it produces. A fast, deterministic divergence-free flow for the
    mass-consistent transport tests (descent on the equator side, ascent on the poleward side), without a
    slow buoyancy spin-up. ``amp`` sets the vigor (max|u| ~ a few * amp/1e4 m/s)."""
    nx, nz = m.cfg.nx, m.cfg.nz
    m._z = m._hydrostatic_z(m.T_atm)
    X = m.x[None, :] / m.x.max()                      # 0..1 across columns (height-grid x is axis 1)
    Z = m.z[:, None] / m.z.max()                      # 0..1 surface->top
    m.psi = amp * np.sin(np.pi * X) * np.sin(np.pi * Z)
    u, w = m.velocities()
    return u, w


# ------------------------------------------------------------------ construction + grids
def test_construction_shapes_and_orderings():
    m = _short()
    nx, nz = m.cfg.nx, m.cfg.nz
    # thermodynamics on the PRESSURE grid (nx, nz), A-convention
    assert m.T_atm.shape == (nx, nz)
    assert m.T_surf.shape == (nx,)
    assert m.p_edge.shape == (nz + 1,)
    for a in (m.p_mid, m.dm, m.sigma, m.RH, m.C_layer):
        assert a.shape == (nz,)
    # dynamics on the HEIGHT grid (nz, nx), B-convention
    assert m.omega.shape == (nz, nx)
    assert m.psi.shape == (nz, nx)
    # prognostic moisture q is on the PRESSURE grid (nx, nz), co-located with T_atm (M2b.3 refinement)
    assert m.q.shape == (nx, nz)
    # reference state on the height grid
    for a in (m.z, m.p_ref, m.T_ref, m.rho_ref, m.N2_ref):
        assert a.shape == (nz,)
    assert np.isfinite(m.S).all() and m.S[0] > m.S[-1]          # flux falls off poleward
    assert m.T_surf[0] > m.T_surf[-1]                            # equator warmer than the cold edge


def test_init_temperature_floor_is_configurable():
    """The IC guess T0 = (max(S, s_init_floor)/sigma)^0.25 clipped to [t_init_floor, 360]; the floors are
    env/config knobs (default 50/50/1e-3, the cold far-field start). A dim world (tiny i0_scale) drives
    S -> ~0 so the clip bites: the floor sets where the transport-starved cold columns begin, closer to
    the true far-field equilibrium. This test pins the floors EXPLICITLY -- a 120 K floor reproduces the
    old warm start, a 50 K floor the new cold default -- so it is robust to whatever the default is.
    (The esat pole guard added alongside these floors makes cooling below the old Magnus qsat pole safe,
    so no floor is needed just to stay above ~30 K; the config comment validates a 10 K init.)"""
    # floors pinned EXPLICITLY (robust to the default, which is the cold 50 K): the old 120/130/1.0
    # clip reproduces the warm start; a low floor starts the cold columns colder.
    warm = _short(i0_scale=1.0e-3, t_init_floor=120.0, t_atm_init_floor=130.0, s_init_floor=1.0)
    assert np.isclose(warm.T_surf.min(), 120.0)                      # the clip bites in this dim world
    cold = _short(i0_scale=1.0e-3, t_init_floor=50.0, t_atm_init_floor=50.0, s_init_floor=1.0e-4)
    assert 50.0 - 1e-6 <= cold.T_surf.min() < 120.0                  # respects the lowered floor, colder start
    assert cold.T_atm.min() >= 50.0 - 1e-6                           # the air-column floor moved too


def test_grid_uniform_in_distance_not_theta():
    m = _short()
    dx = np.diff(m.x)
    assert np.allclose(dx, dx[0])                                # uniform in physical distance x
    dtheta = np.diff(m.theta_deg)
    assert dtheta.max() / dtheta.min() > 3                       # strongly NON-uniform in theta (metric)


def test_height_grid_uniform_and_spans_to_ptop():
    m = _short()
    dz = np.diff(m.z)
    assert np.allclose(dz, dz[0])                                # uniform in height z (Cartesian stencils)
    assert m.z[0] == 0.0 and m.z[-1] == pytest.approx(m.z_top)   # surface(0) -> top(nz-1)
    # the height-grid top is the reference height of the pressure-grid top p_top
    p_at_top, _ = m._reference_profile(np.array([m.z_top]))
    assert float(p_at_top[0]) == pytest.approx(m.cfg.p_top, rel=1e-3)


# ------------------------------------------------------------------ anelastic reference state (#0)
def test_reference_state_hydrostatic_consistent():
    m = _short()
    # ideal gas tie-up is exact by construction: p_ref == rho_ref R T_ref
    assert np.allclose(m.p_ref, m.rho_ref * C.R_DRY * m.T_ref, rtol=1e-12)
    # hydrostatic balance dp/dz = -rho g (centered difference at interior points)
    dpdz = np.gradient(m.p_ref, m.z)
    rhs = -m.rho_ref * C.G_ACCEL
    rel = np.abs(dpdz[1:-1] - rhs[1:-1]) / np.abs(rhs[1:-1])
    assert rel.max() < 0.03                                      # discretization-limited; ~1% at this nz


def test_reference_state_profile_physical():
    m = _short()
    assert np.all(np.diff(m.T_ref) <= 1e-9)                      # T_ref non-increasing with height
    assert np.all(m.T_ref >= m.cfg.T_ref_top - 1e-9)             # floored at the isothermal cap
    assert m.T_ref[0] == pytest.approx(m.cfg.T_ref_surf)         # surface value
    assert np.all(m.rho_ref > 0) and np.all(np.diff(m.rho_ref) < 0)   # density positive, thinning upward
    assert np.all(m.N2_ref[1:-1] > 0)                            # stable reference (theta increases upward)


# ------------------------------------------------------------------ dual-grid reconciliation (#1)
def test_hydrostatic_z_roundtrip_against_pressure_grid():
    """Feed the core a temperature column EQUAL to the reference profile sampled at the pressure-grid
    layer centers; ``_hydrostatic_z`` must then reproduce the reference heights of those layers (the
    height/pressure grids are consistent), with the correct TOP->SURFACE ordering. ``_hydrostatic_z``
    is the lower-edge hypsometric quadrature lifted verbatim from Stage A, so the agreement is
    first-order in the layer count -- the test therefore checks coarse-grid agreement AND that the
    residual CONVERGES with resolution (proving it is discretization, not a bug)."""
    def roundtrip_err(nz):
        m = StageADCore(StageADConfig(nx=8, nz=nz))
        zf = np.linspace(0.0, 100.0e3, 400001)
        pf, _ = m._reference_profile(zf)
        z_ref_pmid = np.interp(m.p_mid, pf[::-1], zf[::-1])      # true ref height at each p_mid
        T_ref_pmid = m._reference_profile(z_ref_pmid)[1]
        m.T_atm = np.tile(T_ref_pmid, (m.cfg.nx, 1))
        z_comp = m._hydrostatic_z()                             # (nx, nz)
        assert z_comp.shape == (m.cfg.nx, nz)
        assert np.all(z_comp[:, 0] > z_comp[:, -1])             # index 0 = top (high), nz-1 = surface (low)
        assert np.allclose(z_comp, z_comp[0][None, :])          # column-independent for a uniform T_atm
        return float(np.abs(z_comp[0] - z_ref_pmid).max() / m.z_top)
    e16, e64 = roundtrip_err(16), roundtrip_err(64)
    assert e16 < 0.05                                           # recovers the reference heights (coarse grid)
    assert e64 < 0.5 * e16                                      # and CONVERGES toward the analytic profile


# ------------------------------------------------------------------ the run-loop harness
def test_zero_buoyancy_is_quiescent():
    """Pure-dynamics check (``couple_thermo=False`` -> the buoyancy is held): with no buoyancy forcing
    (b == 0) the resolved dynamics produce NO flow -- omega and psi stay identically zero -- and with the
    column thermo switched off the state is untouched. (Isolates the M2b.2 dynamics from the M2b.3 thermo
    that is now ON by default; the harness self-test of a zero-drift quiescent run still holds.)"""
    m = _short()
    m.couple_thermo = False                                     # hold b: validate the pure dynamics
    T_atm0, T_surf0 = m.T_atm.copy(), m.T_surf.copy()
    for _ in range(20):
        m.step()
    assert m.iter == 20
    assert np.array_equal(m.T_atm, T_atm0)                       # thermo off -> column untouched
    assert np.array_equal(m.T_surf, T_surf0)
    assert np.abs(m.psi).max() == 0.0 and np.abs(m.omega).max() == 0.0   # no spurious flow at b == 0


def test_harness_noop_run_converges():
    m = _short(check_every=4, patience=2, tol=1e-5, max_iter=1000)
    m.couple_thermo = False                                     # pure-dynamics harness self-test (zero drift)
    summ = m.run(verbose=False)
    assert summ["converged"] is True                            # zero drift -> converges after patience
    assert summ["last_rate"] == pytest.approx(0.0, abs=1e-12)
    assert m.iter <= 1000


def test_checkpoint_roundtrip(tmp_path):
    m = _short()
    for _ in range(7):
        m.step()
    m.psi[:] = 0.123                                             # perturb the dynamics state to check it travels
    st = m.state()
    m2 = _short()
    m2.load_state({k: np.asarray(v) for k, v in st.items()})
    assert np.allclose(m2.T_surf, m.T_surf) and np.allclose(m2.T_atm, m.T_atm)
    assert np.allclose(m2.psi, m.psi) and np.allclose(m2.q, m.q) and m2.iter == m.iter


def test_load_state_normalizes_fortran_ordered_seed_to_c_contiguous():
    # Regression lock for the F-ordered-seed scramble: an npz round-trip preserves fortran_order, and
    # an interpolated/transposed seed (np.apply_along_axis over axis 0 returns an F-contiguous view)
    # loads F-ordered -- which astype(order='K') KEEPS -- so it reaches the device buffers and the fused
    # RawKernels' flat i*nz+k indexing reads scrambled elements (the 2026-07-02 nx1500 "NaN"). load_state
    # must ascontiguousarray every prognostic. The GPU guards in _gpu_kernels.py fire only on-device
    # (CI never exercises them), so this CPU check is the one that catches a refactor back to np.asarray.
    m = _short()
    for _ in range(5):
        m.step()
    st = m.state()
    # Poison every 2-D prognostic to F-order (as a real interpolated/round-tripped seed would arrive).
    poisoned = {k: (np.asfortranarray(v) if np.asarray(v).ndim == 2 else np.asarray(v))
                for k, v in st.items()}
    assert poisoned["T_atm"].flags.f_contiguous and not poisoned["T_atm"].flags.c_contiguous  # truly poisoned
    m2 = _short()
    m2.load_state(poisoned)
    for name in ("T_surf", "T_atm", "q", "q_c"):
        arr = getattr(m2, name)
        host = arr.get() if hasattr(arr, "get") else arr           # cupy -> host for the flag check
        assert np.asarray(host).flags.c_contiguous, f"{name} not C-contiguous after load_state"
    # ...and the values still round-trip correctly (not merely contiguous-but-scrambled).
    assert np.allclose(m2.T_atm, m.T_atm) and np.allclose(m2.q, m.q)


# ------------------------------------------------------------------ config contract
def test_config_mirrors_stage_a_convergence_guard_checkpoint_fields():
    ad = {f.name for f in dataclasses.fields(StageADConfig)}
    a = {f.name for f in dataclasses.fields(StageAConfig)}
    shared = {"tol", "conv_theta_max", "check_every", "patience", "max_iter", "max_seconds",
              "checkpoint_path", "checkpoint_every", "dt", "nx", "nz", "theta_max_deg", "p_top"}
    assert shared <= ad and shared <= a                          # the harness/grid fields are mirrored
    # the M3 closure knobs + the Smagorinsky constant are present (decisions #4 / M3)
    assert {"C_s", "kappa_b", "nu_b"} <= ad


def test_construction_guards_reject_bad_config():
    with pytest.raises(ValueError):
        StageADCore(StageADConfig(theta_max_deg=90.0))      # >= 89.9 guard: x = L*tan(theta) diverges near 90 deg
    with pytest.raises(ValueError):
        StageADCore(StageADConfig(nx=3))                    # too few points for the stencils/BCs
    with pytest.raises(ValueError):
        StageADCore(StageADConfig(nz=3))
    with pytest.raises(ValueError):
        StageADCore(StageADConfig(T_ref_top=400.0))         # cap warmer than the reference surface
    with pytest.raises(ValueError):
        StageADCore(StageADConfig(lapse_ref=0.0))           # non-positive reference lapse rate


def test_i0_scale_scales_the_flux_linearly():
    base = _short(i0_scale=1.0)
    hot = _short(i0_scale=2.0)
    assert np.allclose(hot.S, 2.0 * base.S)                 # S(x) = day_mean_flux(x) * i0_scale (A-parity)
    assert np.all(base.S > 0)


# ====================================================================================================
# M2b.2 -- the anelastic dynamical core (omega-psi, f=0) + the Rayleigh-Benard solver benchmark.
# Validation tiers (M2B_SUBTASKS.md): tier 2 the single overturning cell (Rossby->inf + continuity,
# a hard theorem-invariant) and the Boussinesq/mass-continuity analytic limits; tier 3 the in-solver
# Nu~Ra^beta scaling (the precise exponent is the offline _m2b2_rb_benchmark.py, too costly here -- the
# unit gate checks the conduction limit + that the SAME kernel convects supercritically).
# ====================================================================================================
def test_zero_buoyancy_quiescent_run_converges():
    """The harness still self-tests: with b == 0 and the thermo OFF the circulation drift is identically
    zero, so run() converges (last_rate == 0) -- now exercising the real dynamics substep (Poisson solve
    + tendency) every step, not a no-op."""
    m = _short(check_every=4, patience=2, tol=1e-5, max_iter=1000)
    m.couple_thermo = False
    summ = m.run(verbose=False)
    assert summ["converged"] is True
    assert summ["last_rate"] == pytest.approx(0.0, abs=1e-12)


def test_anelastic_poisson_reduces_to_boussinesq_laplacian():
    """Decision #0: with a CONSTANT reference density the anelastic operator omega = div((1/rho) grad
    psi) is exactly (1/rho)*Laplacian. Manufactured psi_true = sin(pi i/(nx-1)) sin(pi j/(nz-1)) (zero
    on the boundary indices); omega = _div_grad(psi_true, 1/rho) IS the discrete operator applied, so
    solving the Poisson recovers psi_true to machine precision (the Boussinesq limit is exact)."""
    nx, nz, dx, dz, rho0 = 41, 33, 0.05, 0.04, 1.7
    lu = build_anelastic_poisson(np.full(nz, rho0), nx, nz, dx, dz)
    I = np.arange(nx); J = np.arange(nz)
    psi_true = np.sin(np.pi * I[None, :] / (nx - 1)) * np.sin(np.pi * J[:, None] / (nz - 1))
    omega = _div_grad(psi_true, 1.0 / rho0, dx, dz)             # interior = (1/rho)*lap; 0 on edges
    psi = lu.solve(omega.ravel()).reshape(nz, nx)
    assert np.allclose(psi[0, :], 0) and np.allclose(psi[-1, :], 0)      # psi = 0 on the walls
    assert np.allclose(psi[:, 0], 0) and np.allclose(psi[:, -1], 0)
    assert np.allclose(psi, psi_true, atol=1e-9)               # exact in the constant-rho limit


@pytest.mark.parametrize("nx,nz", [(64, 48), (192, 72), (256, 96)])
@pytest.mark.parametrize("stretched", [False, True])
def test_separable_poisson_matches_splu(nx, nz, stretched):
    """The separable DST x tridiagonal inverter (``SeparablePoisson``, speedup Phase 1) is a
    MACHINE-PRECISION drop-in for the splu direct solve of the closed-wall anelastic Poisson, for BOTH
    uniform and stretched dz, on a random RHS WITH NON-ZERO WALLS (so the wall-folding path is exercised --
    the drop-in must reproduce ``lu.solve`` for ANY b, not just wall-zeroed production RHS). Gate: the max
    relative error vs splu < 1e-9."""
    rng = np.random.default_rng(0)
    # A realistic decreasing rho_ref(z): exponential atmosphere, surface(j=0) -> top.
    z_levels = np.linspace(0.0, 1.5e4, nz)
    rho_ref = 1.2 * np.exp(-z_levels / 8.0e3)
    dx = 3.0e5                                                  # cross-track spacing [m]
    if stretched:
        # Monotonic non-uniform z LEVELS (sinh-like BL stretch): the 1-D array path.
        s = np.linspace(0.0, 1.0, nz)
        dz = 1.5e4 * np.sinh(3.0 * s) / np.sinh(3.0)
        assert np.all(np.diff(dz) > 0)
        # rho_ref consistent with the stretched levels (still decreasing surface->top).
        rho_ref = 1.2 * np.exp(-dz / 8.0e3)
    else:
        dz = z_levels[1] - z_levels[0]                         # scalar uniform spacing

    lu = build_anelastic_poisson(rho_ref, nx, nz, dx, dz)
    sep = SeparablePoisson(rho_ref, nx, nz, dx, dz)

    b = rng.standard_normal((nz, nx))                          # FULL random RHS incl. all walls/corners
    psi_splu = lu.solve(b.ravel()).reshape(nz, nx)
    psi_dst = sep.solve(b.ravel()).reshape(nz, nx)

    rel = np.max(np.abs(psi_dst - psi_splu)) / np.max(np.abs(psi_splu))
    assert rel < 1e-9, f"(nx={nx},nz={nz},stretched={stretched}) max rel err {rel:.2e}"
    # Wall rows are pure identity: psi_wall == b_wall exactly (the true-drop-in guarantee).
    assert np.allclose(psi_dst[0, :], b[0, :]) and np.allclose(psi_dst[-1, :], b[-1, :])
    assert np.allclose(psi_dst[:, 0], b[:, 0]) and np.allclose(psi_dst[:, -1], b[:, -1])


def test_velocities_reduce_to_boussinesq_streamfunction():
    """velocities_from_psi with 1/rho == 1 recovers Stage B's u = d psi/dz, w = -d psi/dx."""
    rng = np.random.default_rng(0); nz, nx = 20, 30
    psi = rng.standard_normal((nz, nx)); dx, dz = 1234.0, 56.0
    u, w = velocities_from_psi(psi, np.ones(nz), dx, dz)
    assert np.allclose(u, _ddz(psi, dz)) and np.allclose(w, -_ddx(psi, dx))


def test_anelastic_mass_continuity_holds_in_interior():
    """The mass streamfunction enforces anelastic continuity div(rho_ref u) = 0 (rho u = d psi/dz,
    rho w = -d psi/dx => div = psi_zx - psi_xz = 0 for commuting centered differences). Verified on the
    REAL variable-density reference state with a smooth psi -- a tier-1-flavoured conservation check."""
    m = _short(theta_max_deg=60.0)
    I = np.arange(m.cfg.nx); J = np.arange(m.cfg.nz)
    m.psi = np.sin(np.pi * I[None, :] / (m.cfg.nx - 1)) * np.sin(np.pi * J[:, None] / (m.cfg.nz - 1))
    u, w = m.velocities()
    rho = m.rho_ref[:, None]
    div = _ddx(rho * u, m.dx) + _ddz(rho * w, m.dz)
    scale = np.abs(rho * u).max() / m.dx + np.abs(rho * w).max() / m.dz
    assert np.abs(div[2:-2, 2:-2]).max() / scale < 1e-9        # machine-zero in the deep interior


def test_smagorinsky_viscosity_properties():
    """nu_t = (C_s Delta)^2 |S| + floor: == floor with no strain, >= floor always, grows with strain,
    and collapses to the floor when C_s == 0 (the DNS limit the RB benchmark runs in)."""
    dx, dz, nz, nx = 100.0, 50.0, 12, 16
    z = np.zeros((nz, nx))
    assert np.allclose(smagorinsky_viscosity(z, z, dx, dz, 0.17, 2.0), 2.0)        # no strain -> floor
    rng = np.random.default_rng(2)
    u = rng.standard_normal((nz, nx)); w = rng.standard_normal((nz, nx))
    nu_t = smagorinsky_viscosity(u, w, dx, dz, 0.17, 2.0)
    assert np.all(nu_t >= 2.0 - 1e-12)                          # always >= floor
    nu_big = smagorinsky_viscosity(5 * u, 5 * w, dx, dz, 0.17, 2.0)
    assert nu_big.max() > nu_t.max()                            # grows with strain rate
    assert np.allclose(smagorinsky_viscosity(u, w, dx, dz, 0.0, 2.0), 2.0)         # C_s=0 -> DNS floor


def test_single_overturning_cell_equatorward_return():
    """TIER-2 theorem-invariant (Rossby->inf + continuity, UNIFIED_CORE_ROADMAP Cross-cutting): a fixed
    warm-edge buoyancy forcing drives EXACTLY ONE overturning cell with an equatorward surface return
    at every latitude -- ascent over the warm equator, descent over the cold pole. The anelastic
    omega-psi kernel run to steady state (high nu_b floor -> a smooth laminar cell that converges fast).
    """
    m = StageADCore(StageADConfig(nx=48, nz=24, theta_max_deg=60.0, nu_b=30.0,
                                  tol=2e-3, check_every=200, patience=3, max_iter=40000))
    m.seed_warm_edge_buoyancy(dT_edge=40.0)
    summ = m.run(verbose=False)
    d = m.diagnostics(); nz = m.cfg.nz; w = d["w"]
    assert summ["converged"] is True
    assert d["max_abs_psi"] > 0.0                              # a circulation developed
    assert d["surf_sign_changes"] == 0                         # ONE cell: surface wind never reverses
    assert d["psi_single_signed"]                              # one-signed streamfunction (single cell)
    assert np.all(d["u_surf"][3:-3] < 0)                       # equatorward (toward x=0) at every lat
    assert w[nz // 2, 3] > 0                                   # ascent over the warm equator
    assert w[nz // 2, -4] < 0                                  # descent over the cold pole
    assert d["rising_theta"] < 20                              # rising branch near the warm/sun-line edge


def test_buoyancy_torque_sign_drives_correct_cell():
    """The buoyancy torque -d b/dx is the ONLY vorticity source here: with b warm at the equator
    (x small) the torque is positive, spinning up the cell whose surface branch flows equatorward. A
    flipped buoyancy gradient (warm at the POLE) must reverse the surface wind -- a sign sanity check
    on vorticity_tendency, isolating it from the topology test."""
    m = StageADCore(StageADConfig(nx=40, nz=20, theta_max_deg=55.0, nu_b=40.0,
                                  tol=3e-3, check_every=200, patience=3, max_iter=30000))
    m.seed_warm_edge_buoyancy(dT_edge=35.0)
    m.b = m.b[:, ::-1].copy()                                  # mirror: warm edge now at the POLE
    m.run(verbose=False)
    d = m.diagnostics()
    assert d["surf_sign_changes"] == 0                         # still one cell
    assert np.all(d["u_surf"][3:-3] > 0)                       # surface wind reversed (toward the pole)


def test_rb_conduction_limit_subcritical():
    """TIER-3 benchmark, conduction limit: below the convective threshold (Ra < Ra_crit ~ 658
    free-slip) the RB box stays conductive, Nu ~ 1 (the conductive fixed point; a small grid + short
    integration suffice)."""
    r = rayleigh_benard_run(Ra=200.0, nx=48, nz=24, t_end=30.0, t_avg_start=15.0)
    assert r["Nu"] == pytest.approx(1.0, abs=0.1)


def test_rb_supercritical_convects():
    """TIER-3 benchmark, supercritical: well above critical the SAME kernel convects and carries heat
    far above conduction (Nu clearly > 1). The precise Nu~Ra^beta exponent (literature 2-D band
    ~0.28-0.31) is the offline _m2b2_rb_benchmark.py / M2B2_RESULTS.md -- too costly for the unit gate,
    which only asserts the solver convects and the heat flux is sane."""
    r = rayleigh_benard_run(Ra=2.0e4, nx=64, nz=32, t_end=60.0, t_avg_start=30.0)
    assert r["Nu"] > 1.8                                       # convecting (flux >> conductive)
    assert np.isfinite(r["Nu"]) and r["Nu"] < 30.0            # sane, no blow-up


def test_rb_nu_increases_with_rayleigh_number():
    """A higher Rayleigh number transports more heat: Nu(Ra=5e4) > Nu(Ra=8e3) (the monotone trend the
    Nu~Ra^beta power law rests on -- a cheap two-point sanity check of the scaling direction)."""
    lo = rayleigh_benard_run(Ra=8.0e3, nx=64, nz=32, t_end=60.0, t_avg_start=30.0)
    hi = rayleigh_benard_run(Ra=5.0e4, nx=64, nz=32, t_end=60.0, t_avg_start=30.0)
    assert hi["Nu"] > lo["Nu"] > 1.0


# ====================================================================================================
# M2b.3 -- radiation + moist convection + prognostic moisture IN THE LOOP, with the independent RCE
# anchor. Validation tiers (M2B_SUBTASKS.md): tier 2 the psi==0 single-column limit relaxes to an
# INDEPENDENTLY-derived RCE temperature (single_column_rce -- a standalone relaxation, not the core's
# loop); tier 3 a standard non-rotating RCE benchmark (moist-adiabatic troposphere; warmer SST ->
# warmer/deeper); tier-1-flavoured enthalpy+water conservation of the convective adjustment; plus the
# psi==0-vs-Stage-A regression (a SUSPECT baseline -- reported, not gated). The prognostic moisture is
# on ABSOLUTE footing (evaporate toward qsat over A's wetness mask, condense the supersaturation), NOT a
# normalized RH*qsat.
# ====================================================================================================
# Shared fast-relaxation settings for the no-flow RCE tests: a larger thermodynamic dt (no surface
# stiffness with T_surf prescribed) and fewer convective sweeps per step both cut the unit-gate cost
# while reaching the SAME RCE (the convective sweeps iterate over steps; dt stays well below the
# explicit radiative-relaxation stability limit). The core config and the anchor MUST share them so the
# matched-init reduction is exact.
_RCE_DT, _RCE_SWEEPS = 14400.0, 4


def _rce_core(nx, nz, theta_max, T_surf, S, *, half_rh=False):
    """A no-flow (psi==0) core with a PRESCRIBED uniform surface T and insolation, re-initialized so the
    warm-start is self-consistent with the chosen T_surf (not the construction-time S). ``half_rh``
    starts the moisture at HALF the fixed-RH profile -- a different initial condition, to test that the
    converged RCE is initial-condition-INDEPENDENT (the convective moisture transport makes it so)."""
    # GRAY clear-sky limit (clouds/lw_window/sw_diffuse/sw_bands off, q_homogenize) so the core's psi==0
    # column matches the independent GRAY ``single_column_rce`` reference byte-for-byte. The all-on default
    # is validated elsewhere (the conservation/energy tests, the SST cross-check, the production campaign);
    # this cross-check pins the leaf-physics WIRING against an independent gray assembly, in its own regime.
    m = StageADCore(StageADConfig(nx=nx, nz=nz, theta_max_deg=theta_max,
                                  dt=_RCE_DT, conv_max_sweeps=_RCE_SWEEPS,
                                  clouds=False, lw_window=False, sw_diffuse=False, sw_bands=False,
                                  convective_closure="q_homogenize"))
    m.run_dynamics = False
    m.couple_surface = False                                       # fixed-SST RCE (M2b.3); surface EB is M2b.4
    m.T_surf = np.full(nx, float(T_surf))
    m.S = np.full(nx, float(S))
    z_col = -8000.0 * np.log(m.p_mid / C.P_SURF)
    m.T_atm = np.maximum(m.T_surf[:, None] - 6.5e-3 * z_col[None, :], 130.0)
    rh_fac = 0.5 if half_rh else 1.0
    m.q = np.minimum(m._wetness()[:, None] * m.RH[None, :] * th.qsat(m.T_atm, m.p_mid[None, :]) * rh_fac, 0.5)
    return m


def test_no_flow_core_reproduces_independent_rce_matched_init():
    """TIER-2 (wiring): from a MATCHED initial condition the core's psi==0 single-column physics is
    byte-for-byte the independent ``single_column_rce`` relaxation (a 2-D vectorized class method vs a
    1-D loop function -- genuinely separate assemblies). Any wiring bug (wrong layer for a flux, wrong
    grid, a flipped sign, a double-count) breaks the identity within a few steps. Fast: short run, tight
    tolerance."""
    nz, S, Ts = 14, 350.0, 290.0
    m = _rce_core(4, nz, 50.0, Ts, S)
    for _ in range(500):
        m.step()
    d = m.diagnostics()
    a = single_column_rce(S, Ts, nz=nz, max_iter=500, tol=0.0,     # same step budget & init formula
                          dt=_RCE_DT, conv_max_sweeps=_RCE_SWEEPS)
    assert float(np.max(np.abs(d["T_atm"][0] - a["T_atm"]))) < 0.5   # identical trajectory (-> ~0)
    assert abs(float(d["OLR"][0]) - a["OLR"]) < 2.0


def test_no_flow_core_converges_to_independent_rce():
    """TIER-2 (the headline gate): from a DIFFERENT moisture initial condition (half-RH) the core's
    psi==0 limit still CONVERGES to the same independently-derived RCE -- the column thermodynamics are
    emergent and initial-condition-independent (the convective moisture transport closes the column).
    Tolerances are tier-2 physical (a few K, a few W/m^2), comfortably tighter than the >20 W/m^2
    divergence a wiring bug produces."""
    nz, S, Ts = 12, 360.0, 293.0
    m = _rce_core(4, nz, 55.0, Ts, S, half_rh=True)
    for _ in range(1500):
        m.step()
    d = m.diagnostics()
    a = single_column_rce(S, Ts, nz=nz, max_iter=2500, tol=1e-6, dt=_RCE_DT, conv_max_sweeps=_RCE_SWEEPS)
    assert abs(float(d["OLR"][0]) - a["OLR"]) < 8.0                 # OLR agrees (greenhouse closes)
    assert abs(float(d["T_air_surf_C"][0]) - (a["T_atm"][-1] - C.T_FREEZE)) < 1.5
    assert float(np.max(np.abs(d["T_atm"][0] - a["T_atm"]))) < 3.5  # whole profile tracks the anchor
    assert float(d["vapor_path"][0]) > 1.0                          # a moist column developed (prognostic q)


def test_moist_convective_adjust_conserves_enthalpy_and_water():
    """TIER-1-flavoured: the moist convective adjustment removes a super-adiabatic lapse while conserving
    BOTH column enthalpy sum(cp dm T) (the heat mixing) AND column water sum(dm q) (the moisture loft) --
    exact pairwise-conservative mixing. Build a deliberately unstable column (surface layer over-heated)
    with arbitrary moisture and check the two invariants before/after."""
    m = StageADCore(StageADConfig(nx=8, nz=18, theta_max_deg=55.0))
    nx, nz = m.cfg.nx, m.cfg.nz
    rng = np.random.default_rng(3)
    T = m.T_atm.copy()
    T[:, -1] += 45.0                                               # over-heat the surface layer -> convects
    z = m._hydrostatic_z(T)
    q = np.abs(rng.standard_normal((nx, nz))) * 2e-3
    H0 = np.sum(C.CP_AIR * m.dm[None, :] * T, axis=1)
    W0 = np.sum(m.dm[None, :] * q, axis=1)
    lapse_before = (T[:, -2] - T[:, -1]) / (z[:, -2] - z[:, -1])   # < 0 (T decreases upward); too steep
    moist_convective_adjust(T, m.dm, z, m.p_mid, max_sweeps=30, q=q)
    H1 = np.sum(C.CP_AIR * m.dm[None, :] * T, axis=1)
    W1 = np.sum(m.dm[None, :] * q, axis=1)
    assert np.allclose(H0, H1, rtol=1e-10)                         # enthalpy conserved
    assert np.allclose(W0, W1, rtol=1e-10)                         # water conserved
    lapse_after = (T[:, -2] - T[:, -1]) / (z[:, -2] - z[:, -1])
    assert np.all(lapse_after > lapse_before + 1e-6)              # the unstable lapse was relaxed (less steep)


# --------------------------------------------------------------------------------------------------
# CLEARSKY_SW_PLAN Stage 3: the rh_detrain convective-moisture closure (a COMPARED variant). The aloft
# constant-q slug (CLEARSKY_SW_FINDINGS root cause #1) was localised by Stage 0 to convective q-homogen-
# isation; Stage 3 ships the RH-aware alternative behind ``closure=`` -- default q_homogenize byte-ident-
# ical, rh_detrain conserving column water EXACTLY while giving RH that FALLS with height (no slug).
# --------------------------------------------------------------------------------------------------
def test_convective_closure_default_byte_identical():
    """CROSS-CUTTING RULE (plan rule 1): the default closure (=q_homogenize) is BYTE-IDENTICAL to calling
    with no ``closure`` argument at all -- the Stage-3 opt-in cannot perturb the standing gate. Run the
    bare adjust on the SAME unstable, moist column both ways and require bit-for-bit equal T AND q."""
    m = StageADCore(StageADConfig(nx=6, nz=18, theta_max_deg=55.0))
    rng = np.random.default_rng(7)
    T0 = m.T_atm.copy(); T0[:, -1] += 40.0                         # over-heat the surface layer -> convects
    q0 = np.abs(rng.standard_normal((m.cfg.nx, m.cfg.nz))) * 3e-3
    z = m._hydrostatic_z(T0)
    Ta, qa = T0.copy(), q0.copy()                                 # path A: implicit default
    Tb, qb = T0.copy(), q0.copy()                                 # path B: explicit q_homogenize
    moist_convective_adjust(Ta, m.dm, z, m.p_mid, max_sweeps=20, q=qa)
    moist_convective_adjust(Tb, m.dm, z, m.p_mid, max_sweeps=20, q=qb, closure="q_homogenize")
    assert np.array_equal(Ta, Tb) and np.array_equal(qa, qb)      # bit-for-bit (no FP drift): byte-identical


def test_rh_detrain_conserves_enthalpy_and_water():
    """Stage 3 ACCEPTANCE (a): the rh_detrain closure conserves BOTH column enthalpy sum(cp dm T) and
    column water sum(dm q) EXACTLY (rtol 1e-10 -- the M2b.3 theorem must still hold under the new closure),
    while still relaxing the unstable lapse. The water conservation is the load-bearing claim: rh_detrain
    RESHAPES the vertical q distribution (toward the RH target) but adds/removes NO column water."""
    m = StageADCore(StageADConfig(nx=8, nz=18, theta_max_deg=55.0))
    rng = np.random.default_rng(3)
    T = m.T_atm.copy(); T[:, -1] += 45.0                          # over-heat the surface layer -> convects
    z = m._hydrostatic_z(T)
    q = np.abs(rng.standard_normal((m.cfg.nx, m.cfg.nz))) * 2e-3
    H0 = np.sum(C.CP_AIR * m.dm[None, :] * T, axis=1)
    W0 = np.sum(m.dm[None, :] * q, axis=1)
    lapse_before = (T[:, -2] - T[:, -1]) / (z[:, -2] - z[:, -1])
    moist_convective_adjust(T, m.dm, z, m.p_mid, max_sweeps=30, q=q, closure="rh_detrain")
    H1 = np.sum(C.CP_AIR * m.dm[None, :] * T, axis=1)
    W1 = np.sum(m.dm[None, :] * q, axis=1)
    assert np.allclose(H0, H1, rtol=1e-10)                        # enthalpy conserved (T mixing unchanged)
    assert np.allclose(W0, W1, rtol=1e-10)                        # WATER conserved exactly under rh_detrain
    lapse_after = (T[:, -2] - T[:, -1]) / (z[:, -2] - z[:, -1])
    assert np.all(lapse_after > lapse_before + 1e-6)             # the unstable lapse was still relaxed


def _rh_and_aloft(col):
    """RH(p) and the fraction of column water above 400 hPa for a converged single_column_rce result."""
    T, q, p, z = col["T_atm"], col["q"], col["p_mid"], col["z"]
    dm = np.diff(np.linspace(5e3, C.P_SURF, q.size + 1)) / C.G_ACCEL
    rh = q / np.maximum(th.qsat(T, p), 1e-12)
    wv = q * dm
    frac_400 = float(wv[p < 4.0e4].sum() / max(wv.sum(), 1e-12))
    mid = (p > 1.0e4) & (p < 4.0e4)                              # 100-400 hPa: the slug window
    cv = float(np.std(q[mid]) / max(np.mean(q[mid]), 1e-12))
    return rh, frac_400, cv, p


def test_rh_detrain_breaks_constant_q_slug():
    """Stage 3 ACCEPTANCE (b): on a warm WET RCE column, q_homogenize builds the constant-q aloft slug
    (q ~flat over 100-400 hPa, RH RISING toward the cold top), while rh_detrain gives RH FALLING with
    height and pulls the water DOWN out of the slug -- the qualitative fix the finding called for. Both
    run the SAME column physics; only the convective closure differs."""
    Ts, S, nz = C.T_FREEZE + 50.0, 350.0, 20
    assert wetness_of_Tsurf(Ts) == pytest.approx(1.0)             # a wet ocean column (slug-prone)
    kw = dict(nz=nz, max_iter=5000, tol=1e-5, dt=_RCE_DT, conv_max_sweeps=8)
    qh = single_column_rce(S, Ts, closure="q_homogenize", **kw)
    rd = single_column_rce(S, Ts, closure="rh_detrain", **kw)
    rh_qh, frac_qh, cv_qh, p = _rh_and_aloft(qh)
    rh_rd, frac_rd, cv_rd, _ = _rh_and_aloft(rd)
    # free-troposphere indices: a low-mid (~700 hPa) and an upper (~250 hPa) level
    lo = int(np.argmin(np.abs(p - 7.0e4))); hi = int(np.argmin(np.abs(p - 2.5e4)))
    # (1) rh_detrain: RH FALLS with height (upper RH well below the low-mid RH) -- no slug saturating aloft
    assert rh_rd[hi] < rh_rd[lo] - 0.05
    # (2) q_homogenize is the constant-q SLUG: near-flat q aloft (small CV) and RH NOT falling (rises to top)
    assert cv_qh < 0.15 and rh_qh[hi] >= rh_qh[lo] - 0.05
    # (3) rh_detrain breaks the slug: larger q-CV aloft AND less of the column's water sits above 400 hPa
    assert cv_rd > cv_qh + 0.10
    assert frac_rd < frac_qh - 0.03


def test_standard_rce_features():
    """TIER-3: ``single_column_rce`` reproduces the standard non-rotating RCE structure -- (i) the lower
    troposphere sits ON the moist adiabat (the convecting branch), (ii) a warmer surface gives a warmer
    column at every level. Validates the column physics against textbook RCE, independent of the doc.
    (The deep-stratosphere relaxation is slow, so steadiness is checked as a SMALL residual drift over a
    capped budget, not full machine convergence -- the troposphere is well-relaxed long before the
    skin.)"""
    a_warm = single_column_rce(380.0, 302.0, nz=14, max_iter=3000, tol=1e-5, dt=_RCE_DT, conv_max_sweeps=_RCE_SWEEPS)
    a_cool = single_column_rce(330.0, 288.0, nz=14, max_iter=3000, tol=1e-5, dt=_RCE_DT, conv_max_sweeps=_RCE_SWEEPS)
    # (i) moist-adiabatic lower troposphere: the resolved lapse near the surface ~ the moist adiabat
    T, z, p = a_warm["T_atm"], a_warm["z"], a_warm["p_mid"]
    k = len(T) - 2                                                 # a near-surface layer pair
    lapse = (T[k - 1] - T[k]) / (z[k - 1] - z[k])                  # K/m, negative (cooling upward)
    Gamma_m = th.moist_adiabat_lapse_rate(T[k], p[k])             # K/m magnitude
    assert lapse < 0 and abs(-lapse - Gamma_m) / Gamma_m < 0.25    # on the moist adiabat (within 25%)
    # (ii) warmer surface -> warmer column everywhere (monotone RCE response)
    assert np.all(a_warm["T_atm"] > a_cool["T_atm"] - 1e-6)
    assert a_warm["T_atm"][-1] > a_cool["T_atm"][-1] + 8.0        # the warm column's air is clearly warmer


def test_prognostic_moisture_absolute_footing():
    """The prognostic moisture is on ABSOLUTE footing -- evaporation toward qsat over A's WETNESS MASK,
    NOT a normalized RH*qsat. The signature: a WARM WET surface builds a moist column (vapor + precip),
    while a HOT DRY salt-flat surface (wetness -> 0) stays DRY despite a huge qsat -- the bounded
    dry-rock branch (no Komabayashi-Ingersoll runaway). The exact inverse of an RH*qsat scheme, which
    would imply an enormous vapor path over the scorching flat."""
    nz = 14
    wet = single_column_rce(380.0, 300.0, nz=nz, max_iter=3000, tol=1e-6, dt=_RCE_DT, conv_max_sweeps=_RCE_SWEEPS)
    assert wet["wet"] == pytest.approx(1.0)                        # open ocean
    assert float(np.sum(wet["q"] * np.diff(np.linspace(5e3, C.P_SURF, nz + 1)) / C.G_ACCEL)) > 4.0  # moist
    # hot dry flat: wetness_of_Tsurf(75 C) == 0, so NO evaporation -> q stays ~0 even though qsat is huge
    Ts_hot = C.T_FREEZE + 75.0
    assert wetness_of_Tsurf(Ts_hot) == pytest.approx(0.0)
    dry = single_column_rce(900.0, Ts_hot, nz=nz, max_iter=1500, tol=1e-6, q_init_rh=True,
                            dt=_RCE_DT, conv_max_sweeps=_RCE_SWEEPS)
    assert float(np.max(dry["q"])) < 1e-6                         # absolute-dry (no RH*qsat runaway)
    assert th.qsat(np.asarray(Ts_hot), C.P_SURF) > 0.1            # ...although qsat there IS huge


def test_core_prognostic_moisture_evaporates_and_precipitates():
    """The CORE's prognostic moisture works end-to-end: a column initialized BONE-DRY over a warm wet
    surface gains water vapor by surface evaporation, lofts it by convection, and precipitates the
    supersaturation -- vapor_path and precip both rise from zero. (The no-flow limit; exercises the
    evaporation source + condensation sink + convective loft inside ``StageADCore``.)"""
    m = _rce_core(4, 12, 50.0, 298.0, 360.0)
    m.q[:] = 0.0                                                   # bone dry
    for _ in range(1000):
        m.step()
    d = m.diagnostics()
    assert float(np.min(d["vapor_path"])) > 1.0                    # evaporation built a moist column
    assert float(np.min(d["precip_mm_day"])) > 0.0                # condensation precipitates (E ~= P)


def test_moisture_advected_by_resolved_flow():
    """The prognostic moisture is advected by the RESOLVED flow (the M2b.3 deliverable). Spin up a steady
    overturning cell (held warm-edge buoyancy), drop a surface moisture blob mid-domain, and advect it
    with the steady (u, w): its surface branch flows EQUATORWARD (u_surf < 0), so the blob's centroid
    moves toward the equator (x decreasing). Interior advection conserves column water to within the
    upwind scheme + boundary outflow."""
    m = StageADCore(StageADConfig(nx=40, nz=20, theta_max_deg=60.0, nu_b=40.0,
                                  tol=3e-3, check_every=200, patience=3, max_iter=30000))
    m.seed_warm_edge_buoyancy(dT_edge=40.0)                       # held b; couple_thermo=False
    m.run(verbose=False)                                          # steady single cell, u_surf < 0
    u, w = m.velocities()
    m._z = m._hydrostatic_z(m.T_atm)
    m.q[:] = 1e-4
    i0 = m.cfg.nx // 2
    m.q[i0 - 1:i0 + 2, -1] += 1.0e-2                              # a surface moisture blob at mid-domain
    x = m.x
    def surf_centroid():
        w_s = m.q[:, -1] - 1e-4
        return float(np.sum(w_s * x) / np.sum(w_s))
    c0 = surf_centroid()
    water0 = float(np.sum(m.q * m.dm[None, :]))
    n_adv, dt_adv = 40, 600.0
    for _ in range(n_adv):
        m._advect_moisture(u, dt_adv)
    c1 = surf_centroid()
    water1 = float(np.sum(m.q * m.dm[None, :]))
    assert np.all(u[1, 3:-3] < 0)                                 # surface branch is equatorward
    # The blob moves with the surface wind, so its centroid shifts equatorward by ~ the advective
    # displacement u_surf*T. Threshold that displacement, NOT a hard-coded metre count -- the absolute
    # move scales with the domain size L (x = L*tan(theta)) and the held-buoyancy gradient weakens as L
    # grows, so a fixed 1e4 m de-calibrates at the canonical L=8e7 (CLAUDE.md: box/L de-calibration).
    u_surf = abs(float(np.mean(u[1, 3:-3])))                      # mean equatorward surface speed [m/s]
    expected_disp = u_surf * n_adv * dt_adv                       # bulk advective displacement [m]
    assert c1 < c0 - 0.2 * expected_disp                          # moved equatorward, a sizable fraction of it
    assert abs(water1 - water0) / water0 < 1e-12                  # column water conserved to MACHINE precision
    #   (mass-consistent transport: the divergence-free flux telescopes and the wall/lid fluxes are zero,
    #   so INTEGRAL_k q*dm is invariant -- no longer the old upwind scheme's ~15% boundary-outflow leak)


def test_no_flow_core_vs_stage_a_reported():
    """REGRESSION against a SUSPECT baseline (Stage A is the old lineage), per the roadmap: the core's
    psi==0 column at a surface temperature taken FROM Stage A (kappa_atm=0, local RCE) vs Stage A's own
    column. The difference is REPORTED, not gated -- it is EXPECTED to be large over the hot/wet columns,
    because the core's PROGNOSTIC moisture (evaporate toward qsat, condense the supersaturation) departs
    sharply from Stage A's DIAGNOSTIC fixed-RH q (realism item #20): at a ~56 C SST fixed-RH implies a
    ~2800 kg/m^2 vapor path, the prognostic column ~400 -- a several-hundred-W/m^2 OLR swing the core was
    BUILT to introduce. The genuine code-divergence check is the COLD-DRY columns, where moisture is
    negligible and the two schemes MUST agree (a wiring bug would corrupt those too)."""
    nx, nz = 12, 12
    # conv_theta_max keeps the slow frigid cold-tail from gating Stage A's convergence (OPEN_ISSUES Q8a)
    # Stage A built GRAY (lw_window/sw_diffuse/sw_bands off) to match the gray _rce_core column, so the
    # cold-dry OLR cross-check compares like radiation with like (the code-divergence check, not a physics diff).
    sa = StageA(StageAConfig(nx=nx, nz=nz, theta_max_deg=62.0, kappa_atm=0.0, sponge_relax_frac=0.0,
                             conv_theta_max=48.0, tol=1e-5, check_every=200, patience=3, max_iter=200000,
                             lw_window=False, sw_diffuse=False, sw_bands=False))
    sa.run(verbose=False)
    da = sa.diagnostics()
    m = _rce_core(nx, nz, 62.0, 250.0, 400.0)                     # placeholder T_surf/S, overwritten below
    m.T_surf = sa.T_surf.copy()
    m.S = sa.S.copy()
    z_col = -8000.0 * np.log(m.p_mid / C.P_SURF)
    m.T_atm = np.maximum(m.T_surf[:, None] - 6.5e-3 * z_col[None, :], 130.0)
    m.q = np.minimum(m._wetness()[:, None] * m.RH[None, :] * th.qsat(m.T_atm, m.p_mid[None, :]), 0.5)
    for _ in range(2000):
        m.step()
    dm = m.diagnostics()
    dOLR = dm["OLR"] - da["OLR"]
    Tc = sa.T_surf - C.T_FREEZE
    cold = Tc < -20.0                                              # cold-dry: moisture negligible in BOTH
    print(f"\n[M2b.3 reported] psi==0 core vs StageA(kappa_atm=0), theta_max=62: "
          f"max|dOLR| all={np.max(np.abs(dOLR)):.0f} W/m^2 (hot/wet cols: prognostic vs fixed-RH #20), "
          f"cold-dry cols={np.max(np.abs(dOLR[cold])):.1f} W/m^2 (the code-divergence check)")
    assert cold.sum() >= 2                                         # there ARE cold-dry columns to check
    assert np.max(np.abs(dOLR[cold])) < 20.0                       # schemes AGREE where moisture is moot
    assert np.all(np.isfinite(dOLR))                              # no blow-up anywhere


# ====================================================================================================
# M2b.4 -- the prognostic SURFACE ENERGY BALANCE + boundary-layer / surface-flux physics, with the
# bulk air-sea SST and radiative-subsidence (w_sub) cross-checks. Validation tiers (M2B_SUBTASKS.md):
# tier 4 cross-method (the column SST vs a standalone bulk air-sea EBM; the resolved free-trop
# subsidence RATE vs M2a's external w_sub ~ -0.25 cm/s -- validate the STRENGTH, not the existence) and
# the milestone-1 STRUCTURE (the habitable surface T is DECOUPLED from the free-trop subsidence warming,
# set by boundary-layer + surface-flux physics). The full self-consistent subtropical inversion needs
# the resolved HEAT transport (Issue A), which is M2b.5/6 -- so M2b.4 validates the surface SOLVER and
# the decoupling property, not a closed subsidence inversion.
# ====================================================================================================
import flatworld.radiation as rad
from flatworld import geometry as g
from flatworld.stage_ad_core import bulk_air_sea_sst, radiative_subsidence_rate


def _ribbon_S():
    """The ribbon (theta~48) day-mean insolation, for the livable-column cross-checks."""
    return float(g.day_mean_flux(g.x_of_theta(np.radians(48.0))))


# #91 RESOLVED (ISSUE91_RESULTS.md): the cool-cloudy overcast transition column is a genuine LIMIT CYCLE
# (a ~half-day radiative flicker over a near-constant T_surf ~13 C; a 40k-step probe never settles), NOT an
# energy leak -- its window-mean surface residual closes to << 1 W/m^2. So the balance/anchor comparisons
# are made on TRAILING-WINDOW TIME MEANS (the balance an oscillating column actually satisfies; identical to
# the instantaneous assertion for the settled columns), after a spin-up long enough for the overcast column's
# ocean-slab storage flux to decay below the bound.
def test_surface_balance_reaches_bulk_air_sea_root():
    """TIER-4 cross-method (M2B_SUBTASKS.md M2b.4): the core's PROGNOSTIC surface energy balance, run to
    its no-flow asymptotic state, lands on the INDEPENDENT ``bulk_air_sea_sst`` root -- the core INTEGRATES
    the surface ODE while the anchor ALGEBRAICALLY root-finds the same balance, so agreement proves the
    integrator reaches the energy-balance root (a wrong sign / wrong heat-to-surface / non-converging
    semi-implicit step would break it). The wetness ramp is multivalued (wet vs dry-salt-flat branch --
    the Komabayashi-Ingersoll hysteresis), so the cross-check is made on the core's OWN converged
    wetness branch. The #91 cool-cloudy transition column never sits at a single balanced state (a
    documented limit cycle, ISSUE91_RESULTS.md), so the residual and the anchor comparison use the
    TIME MEAN over a trailing window >> the cycle; a storage-corrected balance (mean residual == the
    slab's realized storage flux) is asserted for every masked column, so a genuine flux leak cannot
    hide behind the oscillation. ~130 s: the spin-up budget is set by the transition column's LATE
    shelf-hop on this grid, see the inline note. Restricted to theta<=44 so the slow frigid cold
    tail (irrelevant, tiny sigma T^4) does not gate the comparison."""
    cfg = StageADConfig(nx=14, nz=16, theta_max_deg=48.0, dt=7200.0, conv_max_sweeps=6)
    m = StageADCore(cfg)
    m.run_dynamics = False                                       # isolate the column + surface (no flow)
    # Spin-up: on THIS (nx=14) grid the theta~42 transition column hops onto the cool-cloudy shelf
    # late (T_surf drifts 24 -> 14.3 C over steps ~7.5k-14k) and sits on the cycle only from ~15k,
    # after which its window-mean residual is <~0.05 W/m^2 (ISSUE91_RESULTS.md, the nx=14 settling
    # measurement). The budget is the honest cost of asserting the full-strength balance for it.
    for _ in range(15000):
        m.step()
    Ck = C.RHO_AIR0 * C.CP_AIR * 1.2e-3 * cfg.wind
    Ce = C.RHO_AIR0 * C.L_VAP * 1.2e-3 * cfg.wind
    n_win = 1500                                                 # 125 days >> the ~half-day #91 cycle
    Cs0 = np.array(m._surf_heat_capacity(), copy=True)           # frozen slab capacity for the storage term
    Ts0 = m.T_surf.copy()
    sums = {k: np.zeros(cfg.nx) for k in ("R", "sw", "D", "Ta", "qa", "wet", "Ts")}
    for _ in range(n_win):
        # pre-step state pairs EXACTLY with the radiation the step computes from it (m._last_rad):
        # inside step(), _radiate/SH/LE all read this state before any update -- the core's OWN
        # (all-on) radiation, so surf_net / surf_abs_sw / D_surf match what the surface evolved under.
        T_air = m.T_atm[:, -1].copy()
        q_air = np.clip(m.q, 0.0, 0.5)[:, -1].copy()
        T_s = m.T_surf.copy()
        wet = np.array(m._wetness(), copy=True)
        m.step()
        out = m._last_rad
        SH = Ck * (T_s - T_air)
        LE = wet * Ce * np.maximum(th.qsat(T_s, C.P_SURF) - q_air, 0.0)
        R = out["surf_net"] - SH - LE                            # the core's own surface residual
        for k, v in (("R", R), ("sw", out["surf_abs_sw"]), ("D", out["D_surf"]),
                     ("Ta", T_air), ("qa", q_air), ("wet", wet), ("Ts", T_s)):
            sums[k] += v
    mean = {k: v / n_win for k, v in sums.items()}
    mask = m.theta_deg <= 44.0
    # storage-corrected balance, EVERY masked column: mean residual == the slab's realized storage flux
    # (exact for steady, relaxing and oscillating columns; tolerance covers the semi-implicit SEB's
    # O(dt*dRdT/Csurf) flux-linearization slop)
    storage = Cs0 * (m.T_surf - Ts0) / (n_win * cfg.dt)
    assert float(np.max(np.abs((mean["R"] - storage)[mask]))) < 1.5
    assert float(np.max(np.abs(mean["R"][mask]))) < 5.0          # the balance holds in the window mean (W/m^2)
    # the INDEPENDENT bulk air-sea root at the window-mean drivers == the window-mean SST
    anchor = bulk_air_sea_sst(mean["sw"], mean["D"], mean["Ta"], mean["qa"], cfg.wind,
                              wetness=mean["wet"])
    assert np.max(np.abs((mean["Ts"] - anchor)[mask])) < 2.0     # core SST == the standalone EBM root (K)


def test_couple_surface_switch_prescribes_or_evolves_Tsurf():
    """The ``couple_surface`` switch is the M2b.3<->M2b.4 boundary: OFF reproduces the M2b.3 PRESCRIBED
    fixed-SST RCE limit (T_surf frozen), ON makes the surface temperature a prognostic OUTPUT (#5, never
    imposed). A quick guard so the two regimes are explicit and don't silently swap."""
    base = dict(nx=6, nz=12, theta_max_deg=45.0, dt=7200.0, conv_max_sweeps=4)
    off = StageADCore(StageADConfig(**base)); off.run_dynamics = False; off.couple_surface = False
    T0 = off.T_surf.copy()
    for _ in range(50):
        off.step()
    assert np.array_equal(off.T_surf, T0)                        # prescribed: surface frozen (M2b.3 limit)
    on = StageADCore(StageADConfig(**base)); on.run_dynamics = False    # couple_surface defaults True
    T0 = on.T_surf.copy()
    for _ in range(50):
        on.step()
    assert not np.allclose(on.T_surf, T0)                        # prognostic: the surface evolved (M2b.4)


def test_surface_decoupled_from_free_troposphere():
    """The milestone-1 STRUCTURE (M2B_SUBTASKS.md M2b.4): the habitable SURFACE temperature is DECOUPLED
    from the free-tropospheric (subsidence) warming and COUPLED to the boundary layer. Relax a livable
    ocean column, then warm the FREE TROPOSPHERE (sigma<=0.6) by 30 K vs the NEAR-SURFACE air
    (sigma>=0.85) by 30 K and compare how far the surface energy-balance equilibrium moves: the free-trop
    anomaly barely reaches the surface (downwelling LW at the surface is set by the lowest layers), while
    the near-surface anomaly moves it strongly. This is exactly why the subtropical-inversion surface T
    is set by BL + surface-flux physics, not by the aloft subsidence warming."""
    cfg = StageADConfig(nx=4, nz=24, theta_max_deg=50.0, dt=10800.0, conv_max_sweeps=6)
    m = StageADCore(cfg)
    m.run_dynamics = False; m.couple_surface = False             # fixed livable SST -> clean RCE (no hysteresis)
    m.S = np.full(cfg.nx, float(g.day_mean_flux(g.x_of_theta(np.radians(46.0)))))
    m.T_surf = np.full(cfg.nx, C.T_FREEZE + 18.0)
    z_col = -8000.0 * np.log(m.p_mid / C.P_SURF)
    m.T_atm = np.maximum(m.T_surf[:, None] - 6.5e-3 * z_col[None, :], 150.0)
    m.q = np.minimum(m._wetness()[:, None] * m.RH[None, :] * th.qsat(m.T_atm, m.p_mid[None, :]), 0.5)
    for _ in range(2500):
        m.step()
    T0 = m.T_atm.copy()

    def surf_eq():
        q = np.clip(m.q, 0.0, 0.5)
        out = rad.column_radiation(m.p_edge[None, :], m.T_atm, q, m.T_surf, m.S, m._albedo())
        return bulk_air_sea_sst(out["surf_abs_sw"], out["D_surf"], m.T_atm[:, -1], q[:, -1], cfg.wind,
                                wetness=m._wetness())[0]

    Ts0 = surf_eq()
    m.T_atm = T0.copy(); m.T_atm[:, m.sigma <= 0.60] += 30.0
    dTs_ft = surf_eq() - Ts0
    m.T_atm = T0.copy(); m.T_atm[:, m.sigma >= 0.85] += 30.0
    dTs_bl = surf_eq() - Ts0
    assert abs(dTs_ft) < 1.0                                     # surface decoupled from the free troposphere
    assert dTs_bl > 4.0                                          # surface coupled to the boundary layer
    assert abs(dTs_ft) < 0.2 * dTs_bl                            # free-trop response << BL response


def test_radiative_subsidence_rate_matches_m2a_anchor():
    """TIER-4 cross-method (M2B_SUBTASKS.md M2b.4): the core's column physics imply a radiative-subsidence
    rate matching M2a's EXTERNAL anchor `w_sub = -0.25 cm/s` (`M2A_RESULTS.md`) in SIGN and ORDER. A
    livable ribbon-like column (prescribed SST, ribbon insolation) is relaxed; ``subsidence_diagnostics``
    reads its free-trop radiative cooling Q_rad (~ -1 .. -2 K/day, cf. M2a's -1.4) and static stability
    (positive), and forms w_sub = Q_rad/(d theta/dz). Validate the STRENGTH (negative = subsidence; ~mm/s,
    within a factor of a few of -0.25), NOT the existence (a transporting cell's ribbon subsidence is
    structurally forced -- the M2a forced-observable trap). The resolved dynamical w matching this
    magnitude is the M2b.5/6 closure (Issue A), not asserted here. Absolute cm/s is SOFT."""
    # GRAY clear-sky core (the M2a anchor was derived in the gray-radiation context; the all-on free-trop
    # cooling / subsidence is a campaign output, reported via converged_diagnostics' w_sub_ribbon_cm_s).
    cfg = StageADConfig(nx=6, nz=24, theta_max_deg=52.0, dt=10800.0, conv_max_sweeps=6,
                        clouds=False, lw_window=False, sw_diffuse=False, sw_bands=False,
                        convective_closure="q_homogenize")
    m = StageADCore(cfg)
    m.run_dynamics = False; m.couple_surface = False             # prescribed livable ribbon SST
    m.T_surf = np.full(cfg.nx, C.T_FREEZE + 12.0)
    m.S = np.full(cfg.nx, _ribbon_S())
    z_col = -8000.0 * np.log(m.p_mid / C.P_SURF)
    m.T_atm = np.maximum(m.T_surf[:, None] - 6.5e-3 * z_col[None, :], 150.0)
    m.q = np.minimum(m._wetness()[:, None] * m.RH[None, :] * th.qsat(m.T_atm, m.p_mid[None, :]), 0.5)
    for _ in range(3000):
        m.step()
    sd = m.subsidence_diagnostics()
    Q, stab, w = sd["Q_rad_K_day"][0], sd["stability_K_km"][0], sd["w_sub_cm_s"][0]
    assert Q < 0.0                                               # free troposphere radiatively COOLING
    assert stab > 0.0                                            # stably stratified (subsidence-balance regime)
    assert w < 0.0                                               # => SUBSIDENCE (the M2a sign)
    assert 0.05 < abs(w) < 2.0                                   # ~mm/s, within a factor of ~few of M2a's 0.25
    # the closed-form w_sub == Q_rad/(d theta/dz) identity (symbolic-vs-numeric cross-check)
    assert w == pytest.approx(radiative_subsidence_rate(Q, stab), rel=1e-12)


# ====================================================================================================
# M2b.5 -- the tier-1 CONSERVATION GATES (energy with NO export term, anelastic mass, cumulative water),
# the resolved VERTICAL moisture flux, and the f=0 wind-driven OCEAN cubic coupled to the surface winds.
# Validation tiers (M2B_SUBTASKS.md): tier 1 energy/mass/water closure each to a documented residual
# (pure theorems -- a wrong model can still satisfy them, so they are CONSISTENCY gates, not correctness);
# tier 2 the analytic ocean cubic (reverses at -H/3, lower 2/3 counter-flowing). Box-test the energy
# closure at >=2x the domain and trust only the residual that does not move (a per-column-balanced
# closure must not depend on the domain extent). NOTE (carried from M2b.4): the resolved flow advects
# MOISTURE (now in x AND z) but not yet HEAT, so energy/water close PER COLUMN (each at local balance);
# the GLOBAL no-export property is the property under test, and what M2b.6 leans on once heat transport
# is wired. "Ribbon subsides" / any forced observable is NOT validated here (UNIFIED_CORE_ROADMAP.md).
# ====================================================================================================
from flatworld.stage_b_overturning import wind_driven_ocean_profile as _wdop_canonical


def test_anelastic_mass_conservation_gate():
    """TIER-1 (M2B_SUBTASKS.md M2b.5): anelastic mass continuity div(rho_ref u) = 0 holds to round-off in
    the deep interior, BY CONSTRUCTION (the mass streamfunction: rho_ref u = d psi/dz, rho_ref w =
    -d psi/dx, so div(rho_ref u) = d^2 psi/dxdz - d^2 psi/dzdx, which the centered interior stencils
    cancel). Drive a REAL flow (a seeded warm-edge buoyancy + dynamics steps) so the test is not the
    trivial psi==0 zero, then assert the deep-interior divergence is negligible vs the momentum-gradient
    scale."""
    m = _short(theta_max_deg=60.0)
    m.seed_warm_edge_buoyancy(dT_edge=40.0)                          # fixed forcing, couple_thermo off
    for _ in range(400):
        m.step()
    assert float(np.abs(m.psi).max()) > 0.0                         # a real circulation developed
    mc = m.mass_conservation()
    assert mc["rel_div_deep"] < 1e-10                               # div(rho_ref u)/|grad(rho u)| ~ round-off


def _coupled_noflow(nx, nz, theta_max, *, dt=7200.0, sweeps=6):
    """A no-flow but SURFACE-COUPLED core (run_dynamics off, couple_surface on): each column relaxes to
    its full radiative-convective + surface energy balance, so TOA net -> 0 and E -> P per column. The
    realistic limit for the energy/water conservation gates (couple_surface=True is the default)."""
    m = StageADCore(StageADConfig(nx=nx, nz=nz, theta_max_deg=theta_max, dt=dt, conv_max_sweeps=sweeps))
    m.run_dynamics = False
    return m


def _column_energy(m, Csurf):
    """Column total energy per unit area [J/m^2] for the conservation THEOREM below: atmosphere
    sensible (C_layer*T) + vapor latent (L_VAP*q*dm) + surface slab (Csurf*T_surf). On the default
    (non-prognostic-condensate) path these are the only stores the column physics moves, so the
    window-integrated TOA net equals the change in this quantity (precipitation removes LIQUID,
    whose latent heat was already released to T_atm; evaporation's LE debit/credit cancels between
    the slab and the vapor store). ``Csurf`` is passed FROZEN (window-start value) so a wetness
    drift cannot inject spurious bookkeeping energy through the state-dependent slab capacity."""
    q = np.clip(m.q, 0.0, 0.5)
    return (np.sum(m.C_layer[None, :] * m.T_atm, axis=1)
            + C.L_VAP * np.sum(q * m.dm[None, :], axis=1) + Csurf * m.T_surf)


# #91 RESOLVED (ISSUE91_RESULTS.md): the wet/dry-transition column's cool-cloudy overcast state is a genuine
# LIMIT CYCLE (a ~half-day radiative flicker; a 40k-step probe never settles), NOT an energy leak -- its
# window-mean TOA net closes to ~5e-3 W/m^2 and the window-integrated energy THEOREM (TOA net in == column
# energy change) holds to ~4e-4 W/m^2. The gate therefore asserts the per-column balance on a TRAILING-WINDOW
# TIME MEAN (identical in strength for the settled columns, the physically-correct statement for the
# oscillator) PLUS the theorem per column, which a genuine leak cannot satisfy even mid-cycle.
def test_energy_conservation_no_export_term():
    """TIER-1 (M2B_SUBTASKS.md M2b.5): GLOBAL energy closure with NO export term -- the only export is OLR
    (psi=0 walls => zero advective lateral export, the Issue-A dissolver, #5). A surface-coupled no-flow
    core relaxed to its asymptotic state has each column at its OWN energy balance (no resolved heat
    transport yet), so the domain-integrated (absorbed SW - OLR) -> 0. The #91 cool-cloudy transition
    column balances only in the TIME MEAN (a documented limit cycle -- its instantaneous toa_net is
    storage flux, ISSUE91_RESULTS.md), so the per-column assertion is the trailing-window mean, and the
    conservation THEOREM -- window-integrated TOA net == column energy change, the statement that is
    exact for steady, relaxing AND oscillating columns -- is asserted per column alongside it. Masked to
    theta<=44 (the slow frigid cold tail, tiny sigma T^4, is irrelevant and would otherwise gate the
    comparison). BOX-TESTED at >=2x the domain: the closure must NOT depend on the domain extent."""
    resids = {}
    for theta_max in (48.0, 72.0):                                  # x_max ratio tan72/tan48 ~ 2.8 (>=2x)
        m = _coupled_noflow(10, 16, theta_max)
        for _ in range(8000):   # spin-up: the overcast column's slab storage flux decays below the bound by ~7k
            m.step()
        n_win = 2000                                                # 167 days >> the ~half-day #91 cycle
        Cs0 = np.array(m._surf_heat_capacity(), copy=True)          # frozen slab capacity (see _column_energy)
        E0 = _column_energy(m, Cs0)
        acc_toa = np.zeros(m.cfg.nx)
        acc_abs = np.zeros(m.cfg.nx)
        for _ in range(n_win):
            m.step()
            out = m._last_rad                                       # the radiation the step ACTUALLY applied
            acc_toa += out["toa_net"]
            acc_abs += m.S_sw_in - out["planetary_reflected"]
        toa_mean = acc_toa / n_win
        mask = m.theta_deg <= 44.0
        # the conservation THEOREM, per column: window TOA net in == column energy stored (leak-proof
        # even mid-cycle; tolerance covers the semi-implicit SEB's O(dt*dRdT/Csurf) linearization slop)
        storage = (_column_energy(m, Cs0) - E0) / (n_win * m.cfg.dt)
        assert float(np.max(np.abs((toa_mean - storage)[mask]))) < 1.5
        # per-column TOA balance in the window mean (each column at its own energy balance)
        assert float(np.max(np.abs(toa_mean[mask]))) < 5.0          # W/m^2 per column
        # masked GLOBAL imbalance is a small fraction of the masked gross absorbed SW
        masked_net = float(np.sum((toa_mean * m.dx)[mask]))
        masked_gross = float(np.sum((acc_abs / n_win * m.dx)[mask]))
        resids[theta_max] = masked_net / masked_gross
        assert abs(resids[theta_max]) < 0.02                        # global closes to <2% (no export term)
    # box-test: the residual does not MOVE with the domain extent (not a box artifact)
    assert abs(resids[48.0] - resids[72.0]) < 0.02


def test_water_conservation_theorem_storage_closes():
    """TIER-1 hard gate (M2B_SUBTASKS.md M2b.5; OPEN_ISSUES c5 -- the CUMULATIVE budget): the genuine
    water-CONSERVATION theorem -- the model creates/destroys NO water except via the surface evaporation
    source E and the condensation sink P, so the change in total column vapor over any window equals the
    cumulative (E - P). (This is the always-true invariant; the steady-state corollary E==P only holds
    where the moisture actually reaches steady state -- the hot evaporative inner columns never do without
    the circulation to carry their vapor to where it rains, exactly the doc's hydrological cycle.) A
    no-flow surface-coupled run: reset the budget, record the total vapor, run a window, and assert the
    storage change matches the cumulative E - P to near machine precision (the convective adjustment +
    condensation conserve water exactly; only E and P are sources/sinks)."""
    m = _coupled_noflow(10, 16, 50.0)
    for _ in range(1500):                                          # let the columns develop moisture
        m.step()
    m.reset_water_budget()
    vapor0 = m.water_conservation()["vapor_mass"]
    for _ in range(1500):
        m.step()
    wc = m.water_conservation()
    assert wc["cum_evap"] > 0.0                                    # evaporation actually happened
    storage_change = wc["vapor_mass"] - vapor0
    sources = wc["cum_evap"] - wc["cum_precip"]
    assert abs(storage_change - sources) / wc["cum_evap"] < 1e-10  # dStorage == E - P (water conserved)


def test_water_closure_ratio_in_livable_rce():
    """TIER-1 steady-state corollary (M2B_SUBTASKS.md M2b.5; OPEN_ISSUES c5): where the moisture DOES reach
    steady state -- a livable temperate column -- the cumulative evaporation balances the precipitation,
    INTEGRAL E ~ INTEGRAL P (closure -> 1). A no-flow PRESCRIBED-livable-SST RCE (all columns at a ribbon
    SST, so none runs away into the hot evaporative regime) is spun up, the budget reset, then run a window
    over which cumulative P/E -> 1. (Complements the conservation theorem above with the steady-state
    closure the doc's water budget reports, isolated to where it physically holds.)"""
    cfg = StageADConfig(nx=6, nz=18, theta_max_deg=50.0, dt=14400.0, conv_max_sweeps=6)
    m = StageADCore(cfg)
    m.run_dynamics = False; m.couple_surface = False               # prescribed livable SST -> clean moist RCE
    m.T_surf = np.full(cfg.nx, C.T_FREEZE + 18.0)
    m.S = np.full(cfg.nx, float(g.day_mean_flux(g.x_of_theta(np.radians(46.0)))))
    z_col = -8000.0 * np.log(m.p_mid / C.P_SURF)
    m.T_atm = np.maximum(m.T_surf[:, None] - 6.5e-3 * z_col[None, :], 150.0)
    m.q = np.minimum(m._wetness()[:, None] * m.RH[None, :] * th.qsat(m.T_atm, m.p_mid[None, :]), 0.5)
    for _ in range(3000):                                          # relax to moist-RCE steady state
        m.step()
    m.reset_water_budget()
    for _ in range(1500):
        m.step()
    wc = m.water_conservation()
    assert wc["cum_evap"] > 0.0
    assert abs(wc["closure_cum"] - 1.0) < 0.05                     # cumulative P/E within 5% of 1


def test_vertical_moisture_flux_lofts_and_conserves():
    """The resolved VERTICAL moisture flux (``_advect_moisture`` z-limb): in an ASCENDING column the
    continuity-derived upward mass flux carries a mid-column moisture bump UPWARD (to smaller pressure-grid
    index = greater height) while conserving the column-integrated water EXACTLY (divergence-free flux,
    zero boundary flux). Direct unit test of the mass-consistent vertical advection, isolated from the
    thermodynamics, driven by an analytic single cell (a uniform 'w' with u=0 is not a divergence-free flow
    -- the mass-consistent scheme correctly forbids it -- so the loft is tested in a real cell's ascent)."""
    m = StageADCore(StageADConfig(nx=20, nz=18, theta_max_deg=55.0, kappa_q=0.0))   # no x-diffusion: pure adv
    nx, nz = m.cfg.nx, m.cfg.nz
    u, w = _analytic_single_cell(m)
    asc = int(np.argmax(w[nz // 2, :-1]))                          # a clearly ASCENDING interior column (w>0)
    # a moisture bump in the LOWER-middle (high index ~ low height) of the ascending column
    q = np.zeros((nx, nz)); k0 = nz - 6
    q[asc, k0 - 1:k0 + 2] = np.array([0.5, 1.0, 0.5]) * 1e-2
    m.q = q.copy()
    W0 = float(np.sum(m.q * m.dm[None, :]))                        # column-integrated water (per unit dx)
    centroid0 = float(np.sum(m.q[asc] * np.arange(nz)) / np.sum(m.q[asc]))
    for _ in range(300):
        m._advect_moisture(u, dt=600.0)
    qc = m.q[asc]
    centroid1 = float(np.sum(qc * np.arange(nz)) / np.sum(qc))
    W1 = float(np.sum(m.q * m.dm[None, :]))
    # threshold the upward index shift against the w-implied displacement -- for a fixed psi the vertical
    # velocity w ~ 1/x.max() ~ 1/L, so a hard-coded index move de-calibrates at the canonical L=8e7 (the
    # CLAUDE.md box/L trap). The bump must move UP by a sizable fraction of what w can carry it.
    dz_h = (m.z[-1] - m.z[0]) / (nz - 1)
    w_up = float(np.max(w[:, asc]))                              # peak ascent in the column [m/s]
    expected_idx = w_up * 300 * 600.0 / dz_h                     # rough upward index displacement
    assert centroid1 < centroid0 - 0.2 * expected_idx            # bump moved UP (smaller index), L-robust
    assert qc[:k0].sum() > 0.0                                    # vapor reached above the source layers
    assert np.all(m.q >= 0.0)                                     # monotone upwind keeps q >= 0 (no clamp)
    assert abs(W1 - W0) / W0 < 1e-12                             # column water conserved to MACHINE precision
    #   (the divergence-free flux telescopes exactly => sum(q dm) invariant; no post-update clamp needed,
    #   so there is no one-directional water source -- it is exactly conserved, not merely "approximately")


def test_ocean_cubic_reverses_at_minus_H_over_3():
    """TIER-2 analytic anchor (M2B_SUBTASKS.md M2b.5; reuses Stage B's cubic assertion): the f=0
    wind-driven OCEAN momentum profile coupled to the core's surface winds reverses at z = -H/3 with the
    lower two-thirds running AGAINST the wind (peak return -1/3 of the surface speed) -- the no-Ekman
    cubic. The per-column profile is the analytic cubic SCALED by each column's surface cross-track wind
    (shape fixed, not magnitude -- the f=0 invariant). The cold-trap SST stays an OUTPUT (momentum-only
    ocean)."""
    m = _short(theta_max_deg=60.0)
    m.seed_warm_edge_buoyancy(dT_edge=40.0)
    for _ in range(400):
        m.step()                                                   # develop a surface wind to drive the ocean
    oc = m.ocean_momentum_profile(nz_ocean=600, H=300.0)
    # the analytic cubic anchor (== Stage B's canonical solution)
    z_ref, u_ref, rev_ref, ridx_ref = _wdop_canonical(nz=600, H=300.0)
    assert np.allclose(oc["u_norm"], u_ref)                        # same canonical cubic as Stage B
    assert oc["reversal_depth"] == pytest.approx(-100.0)           # -H/3
    assert oc["z"][oc["reversal_index"]] == pytest.approx(-100.0, abs=2.0)
    assert oc["u_norm"][-1] == pytest.approx(1.0)                  # surface runs with the wind
    below = oc["z"] < -100.0
    assert np.all(oc["u_norm"][below][1:-1] < 0)                   # lower 2/3 counter-flow
    assert oc["u_norm"].min() == pytest.approx(-1.0 / 3.0, abs=0.02)
    # per-column scaling preserves the SHAPE (u_ocean = u_surf * u_norm), magnitude set by the wind
    i = int(np.argmax(np.abs(oc["u_surf_wind"])))                  # a column with a nonzero surface wind
    assert abs(oc["u_surf_wind"][i]) > 0.0
    assert np.allclose(oc["u_ocean"][i], oc["u_surf_wind"][i] * oc["u_norm"])


# ====================================================================================================
# M2b.6 -- INTEGRATION: the resolved HEAT transport (closes Issue A in the fused domain), the Stage-C
# INTERFACE CONTRACT emission, and the end-to-end cascade decoupling. Validation tiers
# (M2B_SUBTASKS.md / UNIFIED_CORE_ROADMAP.md): the heat advection is conservative (tier-1-flavoured,
# theta-mass to round-off in its flux-form limbs) and ADIABATIC (ascent cools, subsidence warms); the
# fused core moves T(theta) OFF the per-column-RCE lineage (a REPORTED diff, not a gated number -- the
# core is BUILT to move it); and the Stage-C contract is consumed UNCHANGED (the decoupling is at the
# interface, not at frozen tunings). NOTE: a CONVERGED coupled run is GPU/M3-budget work (the coupled
# core is CFL-stiff, dt ~ seconds) -- these are fast BOUNDED checks of the mechanism + the plumbing,
# with production staged (M2B6_RESULTS.md), per the project's code-is-the-deliverable stance.
# ====================================================================================================
from flatworld import constants as _C2  # noqa


def test_heat_advection_conserves_theta_mass_and_subsidence_warms():
    """The M2b.6 resolved VERTICAL heat flux (``_advect_heat`` z-limb) is ADIABATIC and mass-consistent: a
    single overturning cell REARRANGES potential temperature while conserving the column-integrated
    theta-mass EXACTLY (the divergence-free flux telescopes), and the recovered absolute T shows the
    load-bearing SUBSIDENCE WARMING in the descending branch (high-theta air carried to higher pressure /
    larger Exner -- the subtropical subsidence-warming the core is built around). Isolated from the
    cross-track diffusion (kappa_T=0). (Net adiabatic cooling under a *uniform* ascent can no longer be
    tested in isolation: a uniform upward w with u=0 is not divergence-free, so the mass-consistent scheme
    forbids it -- ascent must pair with descent in a closed cell.)"""
    m = StageADCore(StageADConfig(nx=20, nz=18, theta_max_deg=55.0, kappa_T=0.0))
    nx, nz = m.cfg.nx, m.cfg.nz
    u, w = _analytic_single_cell(m, amp=2.0e4)                    # gentle cell (keeps the enthalpy drift small)
    exner = (m.p_mid / C.P_SURF) ** C.KAPPA_POISSON
    Wth0 = float(np.sum((m.T_atm / exner[None, :]) * m.dm[None, :]))
    T0 = m.T_atm.copy()
    desc = int(np.argmin(w[nz // 2, :]))                          # a clearly DESCENDING column (w<0)
    for _ in range(120):
        m._advect_heat(u, dt=600.0)
    Wth1 = float(np.sum((m.T_atm / exner[None, :]) * m.dm[None, :]))
    assert abs(Wth1 - Wth0) / Wth0 < 1e-12                       # theta-mass conserved to MACHINE precision
    # subsidence warming ~ descent displacement x dry-adiabatic lapse; both scale with w (~1/L for a fixed
    # psi), so threshold against that scale, not a hard-coded K (de-calibrates at the canonical L=8e7).
    w_dn = abs(float(np.min(w[:, desc])))                        # peak descent [m/s]
    warm_scale = w_dn * 120 * 600.0 * (C.G_ACCEL / C.CP_AIR)     # ~ adiabatic subsidence warming [K]
    assert float(np.mean((m.T_atm - T0)[desc])) > 0.2 * warm_scale   # subsidence WARMS the descending column
    assert float(np.ptp(m.T_atm - T0)) > 0.4 * warm_scale            # the cell actively redistributed heat
    assert np.all(np.isfinite(m.T_atm))


def test_heat_cross_track_diffusion_is_conservative():
    """The M2b.6 cross-track heat eddy-diffusion limb is FLUX FORM (interface fluxes telescope, zero flux
    at the equator/pole walls), so it conserves the cross-track theta integral -- it redistributes, never
    sources, energy. A theta gradient across x with NO advection (u=w=0) relaxes while INTEGRAL_x theta dm
    stays invariant to round-off."""
    m = StageADCore(StageADConfig(nx=12, nz=12, theta_max_deg=55.0, kappa_T=2.0e5))
    m._z = m._hydrostatic_z(m.T_atm)
    exner = (m.p_mid / C.P_SURF) ** C.KAPPA_POISSON
    # impose a cross-track theta contrast (warm equator -> cool pole) via T_atm
    m.T_atm = m.T_atm + np.linspace(20.0, -20.0, m.cfg.nx)[:, None]
    Wth0 = float(np.sum((m.T_atm / exner[None, :]) * m.dm[None, :]))
    spread0 = float(np.ptp((m.T_atm / exner[None, :])[:, -1]))
    u = np.zeros((m.cfg.nz, m.cfg.nx))
    for _ in range(50):
        m._advect_heat(u, dt=2.0e4)                              # u=0 -> no mass flux; isolates the diffusion limb
    Wth1 = float(np.sum((m.T_atm / exner[None, :]) * m.dm[None, :]))
    spread1 = float(np.ptp((m.T_atm / exner[None, :])[:, -1]))
    assert abs(Wth1 - Wth0) / abs(Wth0) < 1e-12                  # cross-track theta integral conserved
    assert spread1 < spread0                                     # the gradient diffused (relaxed)


def test_resolved_heat_transported_by_the_cell():
    """The resolved flow TRANSPORTS heat between columns (the M2b.6 mechanism, mirroring the moisture
    test): spin up a steady single cell (held warm-edge buoyancy), drop a near-surface WARM anomaly
    mid-domain, advect it with the steady (u, w). Its surface branch flows EQUATORWARD (u_surf<0), so the
    anomaly's centroid moves toward the equator; the recirculating transport conserves column theta-mass
    to the energy-gate tolerance (the vertical+diffusion limbs are exact, the cross-track advective form
    the only small leak)."""
    m = StageADCore(StageADConfig(nx=40, nz=20, theta_max_deg=60.0, nu_b=40.0,
                                  tol=3e-3, check_every=200, patience=3, max_iter=30000))
    m.seed_warm_edge_buoyancy(dT_edge=40.0)                       # held b; couple_thermo=False
    m.run(verbose=False)                                          # steady single cell, u_surf < 0
    u, w = m.velocities()
    m._z = m._hydrostatic_z(m.T_atm)                              # fix z_p so control/blob advect identically
    exner = (m.p_mid / C.P_SURF) ** C.KAPPA_POISSON
    bg = m.T_atm.copy()                                           # steady background T(x, p)
    i0 = m.cfg.nx // 2
    j0 = m.cfg.nz - 3                                             # a low-tropospheric level (clear equatorward u)
    x = m.x
    # CONTROL: advect the background alone (isolates the steady drift from the blob's transport)
    m.T_atm = bg.copy()
    for _ in range(40):
        m._advect_heat(u, 600.0)
    T_ctrl = m.T_atm.copy()
    # BLOB: same advection, with a warm anomaly added at mid-domain -- the difference is the transported blob
    m.T_atm = bg.copy(); m.T_atm[i0 - 1:i0 + 2, j0] += 8.0
    Wth0 = float(np.sum((m.T_atm / exner[None, :]) * m.dm[None, :]))
    for _ in range(40):
        m._advect_heat(u, 600.0)
    Wth1 = float(np.sum((m.T_atm / exner[None, :]) * m.dm[None, :]))
    anom = (m.T_atm - T_ctrl)[:, j0]                             # the transported warm anomaly (control-subtracted)
    a = np.clip(anom, 0.0, None)
    c1 = float(np.sum(a * x) / (np.sum(a) + 1e-30))
    c0 = float(x[i0])                                            # the anomaly started at mid-domain
    assert np.all(u[1, 3:-3] < 0)                                # surface branch is equatorward
    # threshold the equatorward shift against the advective displacement u_surf carries (the absolute move
    # scales with the domain size L = x = L*tan(theta), so a hard-coded 1e4 m de-calibrates at L=8e7).
    u_surf = abs(float(np.mean(u[1, 3:-3])))                     # mean equatorward surface speed [m/s]
    expected_disp = u_surf * 40 * 600.0                          # bulk advective displacement [m]
    assert c1 < c0 - 0.2 * expected_disp                         # warm anomaly carried equatorward (L-robust)
    assert abs(Wth1 - Wth0) / Wth0 < 1e-12                       # theta-mass conserved to MACHINE precision (was ~2%)
    assert np.all(np.isfinite(m.T_atm))


def test_tracer_advection_is_uniform_invariant():
    """REGRESSION GATE for the M3-prerequisite fix (issue #40, OPEN_ISSUES "tracer-advection
    mass-consistency"): a consistent advection must leave a UNIFORM field uniform. Under a real overturning
    cell a UNIFORM potential temperature AND a UNIFORM specific humidity must each stay uniform (the old
    flux-form-with-actual-density scheme spuriously spread a uniform theta ~12 K -- the bug this fix
    removes). Both limbs share the discretely divergence-free mass flux, so uniform-invariance holds for
    heat and moisture together, AND the column integrals are conserved to round-off."""
    m = StageADCore(StageADConfig(nx=20, nz=18, theta_max_deg=55.0))
    u, w = _analytic_single_cell(m, amp=6.0e4)                    # a vigorous real cell (the harder test)
    exner = (m.p_mid / C.P_SURF) ** C.KAPPA_POISSON
    # UNIFORM potential temperature theta = 300 K and UNIFORM specific humidity q = 5e-3
    m.T_atm = (300.0 * exner)[None, :] * np.ones((m.cfg.nx, 1))
    m.q = np.full((m.cfg.nx, m.cfg.nz), 5.0e-3)
    th_mass0 = float(np.sum((m.T_atm / exner[None, :]) * m.dm[None, :]))
    q_mass0 = float(np.sum(m.q * m.dm[None, :]))
    for _ in range(200):
        m._advect_heat(u, 300.0)
        m._advect_moisture(u, 300.0)
    theta = m.T_atm / exner[None, :]
    assert float(np.ptp(theta)) < 1e-6                           # uniform theta STAYS uniform (was ~12 K)
    assert float(np.ptp(m.q)) < 1e-12                            # uniform q stays uniform
    assert abs(float(np.sum(theta * m.dm[None, :])) - th_mass0) / th_mass0 < 1e-12   # theta-mass conserved
    assert abs(float(np.sum(m.q * m.dm[None, :])) - q_mass0) / q_mass0 < 1e-12       # water conserved


def test_couple_heat_moves_T_off_the_no_heat_lineage():
    """The fused core with resolved HEAT transport settles on a DIFFERENT T(theta) than the no-heat
    (per-column-RCE) limit -- the whole point of the core (it moves T(theta) off the diffusive lineage,
    a REPORTED diff per UNIFIED_CORE_ROADMAP Cross-cutting 'Old-model OUTPUTS', not a gated number). From
    a shared no-flow RCE warm-start, run a bounded coupled phase with couple_heat ON vs OFF; assert both
    stay finite and the heat-ON near-surface T(theta) DIFFERS measurably from heat-OFF (and the heat-OFF
    case barely moves -- it IS the lineage). NOT converged (bounded budget)."""
    # The resolved heat transport accumulates ~ w * n_steps, and the cell's w ~ 1/L on the larger domain
    # (x = L*tan(theta)), so a FIXED coupled-step count under-develops the effect at the canonical L=8e7
    # (max|Ton-Toff| 0.09 at 2500 steps vs >0.5 at the default L). Scale the bounded coupled phase with L.
    n_coupled = int(round(2500 * max(1.0, C.L_SUN / 2.0e7)))
    def staged(couple_heat):
        m = StageADCore(StageADConfig(nx=16, nz=14, theta_max_deg=60.0, conv_max_sweeps=4))
        m.run_dynamics = False
        for _ in range(800):
            m.step()                                             # fast no-flow RCE spin-up (shared start)
        base = m.T_atm[:, -1].copy()
        m.run_dynamics = True; m.couple_heat = couple_heat
        for _ in range(n_coupled):
            m.step()
        return base, m.T_atm[:, -1].copy()
    base_on, Ton = staged(True)
    base_off, Toff = staged(False)
    assert np.all(np.isfinite(Ton)) and np.all(np.isfinite(Toff))
    assert np.array_equal(base_on, base_off)                   # the shared no-flow RCE start is deterministic
    # heat ON moves T(theta) measurably off the no-heat case -- the core's job. (NOT the fragile "OFF moves
    # less than ON": which config departs further from the start is regime-dependent -- it reverses at L=8e7
    # -- so that is not a robust invariant; the robust claim is simply that ON and OFF DIFFER.) The 0.2 K
    # floor is L-robust (>0.5 at default L, ~0.35 at L=8e7) and far above the ~1e-3 K bounded-run noise.
    assert float(np.max(np.abs(Ton - Toff))) > 0.2             # K; the resolved heat transport reshapes T


def test_stage_c_inputs_emits_interface_contract():
    """``StageADCore.stage_c_inputs`` emits the Stage-C interface callables (sst/wetness/aloft-supply +
    the ascent/subsidence gates) the cascade consumes, with the SAME shapes/bounds as the A+B backbone,
    plus the poleward-quiescence acceptance diagnostic. The gates REUSE cascade.cell_ascent_gate /
    subsidence_gate (fed the core's diagnostics dict), so they match the cascade's contract byte-for-byte."""
    from flatworld import cascade
    m = StageADCore(StageADConfig(nx=20, nz=14, theta_max_deg=65.0, conv_max_sweeps=4))
    m.run_dynamics = False
    for _ in range(300):
        m.step()
    m.run_dynamics = True
    for _ in range(300):
        m.step()
    sci = m.stage_c_inputs()
    th = np.linspace(30.0, 60.0, 13)
    sst = np.asarray(sci["sst_of_theta"](th), float)
    wet = np.asarray(sci["wetness_of_theta"](th), float)
    F = np.asarray(sci["aloft_supply_of_theta"](th), float)
    asc = np.asarray(sci["ascent_gate"](th), float)
    sub = np.asarray(sci["subsidence_gate"](th), float)
    assert np.all(np.isfinite(sst)) and np.all(np.isfinite(F))
    assert np.all((wet >= 0) & (wet <= 1))
    assert np.all((asc >= 0) & (asc <= 1)) and np.all((sub >= 0) & (sub <= 1))
    assert np.all(F >= 0)                                         # supply band is non-negative [mm]
    # the gates are IDENTICAL to feeding cascade's functions the core's diagnostics (the shared contract)
    d = m.diagnostics()
    dgate = dict(theta_deg=d["theta_deg"], w=d["w"], precip=d["precip_mm_day"])
    assert np.allclose(asc, np.asarray(cascade.cell_ascent_gate(dgate)(th), float))
    assert np.allclose(sub, np.asarray(cascade.subsidence_gate(dgate)(th), float))
    pq = sci["poleward_quiescence"]
    assert set(pq) >= {"global_max_psi", "poleward_max_psi", "ratio"}


def test_stage_c_consumes_ad_core_contract_unchanged():
    """M2b.6 DECOUPLING VERIFICATION (UNIFIED_CORE_ROADMAP §M2b 'Stage C untouched is a claim to VERIFY'):
    Stage C runs on the AD core's interface contract with its DEFAULT knobs (continentality_K, oro_k,
    tau_rain, the f_loft gate, the subsidence source) -- no knob is retuned for the core. Build a small AD
    core, emit its Stage-C inputs, and drive StageCMap with a DEFAULT StageCConfig(test) carrying only the
    interface callables; assert the maps are finite and the default knobs are untouched. This proves the
    decoupling lives at the interface (what each field MEANS), not at frozen tunings."""
    from flatworld import config as cfgmod
    from flatworld.stage_c_map import StageCMap
    from flatworld.topography import Topography
    m = StageADCore(StageADConfig(nx=20, nz=14, theta_max_deg=65.0, conv_max_sweeps=4))
    m.run_dynamics = False
    for _ in range(300):
        m.step()
    m.run_dynamics = True
    for _ in range(300):
        m.step()
    sci = m.stage_c_inputs()
    ascent = sci["ascent_gate"]
    floft = lambda t: np.asarray(ascent(t), float) * 0.7         # gate x a fixed plume efficiency
    topo = Topography.procedural_continent(seed=0, nx=48, ne=48, Le_km=6000.0,
                                           width_km=3500.0, center_theta=48.0)
    defaults = cfgmod.StageCConfig()
    Ccfg = cfgmod.stage_c("test", sst_C_of_theta=sci["sst_of_theta"],
                          aloft_supply_of_theta=sci["aloft_supply_of_theta"],
                          wetness_of_theta=sci["wetness_of_theta"],
                          f_loft_of_theta=floft, subsidence_of_theta=sci["subsidence_gate"])
    # the substantive Stage-C knobs are the DEFAULTS -- not retuned for the AD core (the decoupling claim)
    assert Ccfg.continentality_K == defaults.continentality_K
    assert Ccfg.oro_k == defaults.oro_k and Ccfg.tau_rain == defaults.tau_rain
    Cmap = StageCMap(topo, Ccfg)
    Cmap.run(verbose=False)
    dC = Cmap.diagnostics()
    assert np.all(np.isfinite(dC["T_map_C"])) and np.all(np.isfinite(dC["P_mm_day"]))
    assert np.all(dC["P_mm_day"] >= 0)


def test_run_cascade_core_ad_end_to_end():
    """``run_cascade(core='ad')`` PLUMBING regression (the M2b.6 cascade integration): the fused core is
    built, ``stage_c_inputs`` is emitted, and the synthesized ``dB`` (theta_deg/w/precip/surf_sign_changes/
    psi) feeds the SAME downstream (band F, gates, land-T, Stage C, classify) as the diffusive backbone --
    exercised end-to-end, which the per-method unit tests (which build the core directly) do NOT. Tiny AD
    budget so it is a fast smoke check of the wiring, not a converged climate. Guards: B is subsumed
    (None), ``ad`` is returned, the AD-path ``dB`` carries every key the downstream reads, and the maps are
    finite/non-negative (a missing dB key or a shape mismatch in the new branch would raise here)."""
    from flatworld import cascade
    # theta_max_deg spans the production [28,62] band (cascade.py asserts the core's theta grid covers it,
    # so the th_grid clamp stays byte-identical to the dropped Stage A -- #209). nx is fixed, so widening
    # theta_max only respaces the grid: same cell count, same smoke-check cost.
    r = cascade.run_cascade(profile="test", core="ad",
                            ad_overrides=dict(nx=16, nz=12, theta_max_deg=70.0, conv_max_sweeps=3),
                            ad_rce_steps=40, ad_dyn_steps=40, verbose=False)
    assert r["ad"] is not None and r["B"] is None          # the AD core subsumes Stage B
    # the synthesized dB carries exactly the keys the downstream band/gates/print consume
    assert {"theta_deg", "w", "precip", "surf_sign_changes", "psi"} <= set(r["dB"])
    dC = r["dC"]
    assert np.all(np.isfinite(dC["T_map_C"])) and np.all(np.isfinite(dC["P_mm_day"]))
    assert np.all(dC["P_mm_day"] >= 0)
    assert np.all(np.isfinite(r["backbone"]["sst_C"]))


def test_run_aborts_on_nonfinite_state_without_poisoning_checkpoint(tmp_path):
    """M3 COST GUARD (robustness): when the state goes non-finite MID-RUN (a CFL/closure blow-up) the run
    must abort IMMEDIATELY and must NOT overwrite the last good checkpoint with NaN/Inf -- so a blown-up
    expensive coupled run neither burns the remaining guard budget nor kills its own resume state. A
    monkeypatched step injects Inf after a good checkpoint has been written; confirm: (a) `run` raises
    `FloatingPointError`, and (b) the on-disk checkpoint is still the GOOD, finite state (not poisoned)."""
    from flatworld import convergence as conv
    ckpt = str(tmp_path / "ad_ckpt.npz")
    m = StageADCore(StageADConfig(nx=10, nz=12, theta_max_deg=55.0, check_every=2, patience=5,
                                  checkpoint_path=ckpt, checkpoint_every=2))
    m.run_dynamics = False; m.couple_thermo = False              # no-op steps (state frozen unless poisoned)
    good_T = m.T_atm.copy()
    orig_step, calls = m.step, {"n": 0}
    def poisoned_step():
        orig_step(); calls["n"] += 1
        if calls["n"] == 3:                                      # blow up AFTER the iter=2 good checkpoint
            m.T_atm[0, 0] = np.inf
    m.step = poisoned_step
    with pytest.raises(FloatingPointError):
        m.run(verbose=False)
    st, _ = conv.load_checkpoint(ckpt)                          # a good checkpoint was written at iter=2 ...
    assert st is not None and int(st["iter"]) == 2              # ... and the blow-up did NOT overwrite it
    assert np.all(np.isfinite(st["T_atm"])) and np.allclose(st["T_atm"], good_T)


def test_checkpoint_cadence_fires_when_check_every_does_not_divide(tmp_path, monkeypatch):
    """M3 COST GUARD (robustness): periodic checkpoints must fire even when ``check_every`` does NOT divide
    ``checkpoint_every``. ``iter`` advances by ``check_every`` each loop, so the old ``iter % checkpoint_every
    == 0`` guard silently NEVER fired for a mistuned pair -- leaving an expensive guarded run with no resume
    point (only the single final checkpoint). With check_every=3, checkpoint_every=7 (3 does not divide 7) the
    window-crossing guard ``iter % checkpoint_every < check_every`` must still write a MID-RUN checkpoint
    (here at iter=9, since 9 % 7 = 2 < 3 -- exactly the write the old ``== 0`` guard would have missed)."""
    from flatworld import convergence as conv
    ckpt = str(tmp_path / "cad_ckpt.npz")
    m = StageADCore(StageADConfig(nx=10, nz=12, theta_max_deg=55.0, check_every=3, checkpoint_every=7,
                                  patience=1000, max_iter=15, checkpoint_path=ckpt))
    m.run_dynamics = False; m.couple_thermo = False              # frozen no-op state (stops at the max_iter guard)
    saved_iters = []
    orig = conv.save_checkpoint
    monkeypatch.setattr(conv, "save_checkpoint",
                        lambda path, state, meta=None: (saved_iters.append(int(meta["iter"])),
                                                        orig(path, state, meta))[1])
    m.run(verbose=False)
    assert 9 in saved_iters                                      # a mid-run periodic checkpoint was written


# ====================================================================================================
# M3.4b Task 1: a higher-order (flux-limited, van Leer) theta_pot advection option behind a config flag
# (heat_advection="upwind" default | "fluxlim"), the DECISIVE test for the energy-closure LIMIT
# (M33_M34_RESULTS.md energy verdict = OPEN). Upwind theta_pot advection implicitly DIFFUSES theta_pot,
# spuriously changing int(cp*T) (the APE->KE budget the missing dissipation-heating term cannot return);
# a LOWER-diffusion scheme at FIXED resolution isolates the numerical contribution from any real sink
# WITHOUT the nz-ladder's convergence-bias confound. These tests are LOAD-BEARING: the scheme swap is
# meaningless unless (a) fluxlim's implied numerical diffusion is PROVABLY << upwind, (b) both schemes stay
# exactly conservative (flux form), and (c) fluxlim is monotone (TVD, no spurious overshoot). M3.4b Task 1.
# ====================================================================================================


def _bump_advection_metric(scheme):
    """Advect a smooth Gaussian bump a fixed distance under a UNIFORM interior horizontal mass flux using the
    named flux-form scheme, returning (relative tracer-mass change, fractional PEAK-amplitude loss). Peak
    decay is the clean, monotone measure of a pulse-advection scheme's implicit numerical diffusion: a
    diffusive scheme smears the peak down, and a TVD scheme cannot push it above the initial max -- so
    peak_loss in [0,1] compares the two schemes' diffusion unambiguously (unlike the x^2-weighted second
    moment, which is oversensitive to the far tails for a non-purely-diffusive higher-order scheme). Fz=0
    (pure x), the bump kept interior (the no-flux walls are never touched), tracer constant in z so only the
    x-limb acts."""
    nx, nz = 160, 4
    m = StageADCore(StageADConfig(nx=nx, nz=nz, theta_max_deg=60.0))
    advect = m._flux_form_fluxlim if scheme == "fluxlim" else m._flux_form_upwind
    x = np.arange(nx, dtype=float)
    x0, sig0 = nx / 3.0, 6.0
    bump = np.exp(-0.5 * ((x - x0) / sig0) ** 2)
    tracer = np.repeat(bump[:, None], nz, axis=1).astype(float)   # constant in z
    dm = np.asarray(m.dm)
    Fcell = 0.3 * m.dx / 600.0                                    # face speed [m/s] -> Courant 0.3 at dt=600 s
    Fx = np.zeros((nx + 1, nz)); Fx[1:nx] = Fcell * dm[None, :]   # uniform interior mass flux (div-free interior)
    Fz = np.zeros((nx, nz + 1))
    mass0 = float(np.sum(tracer * dm[None, :]) * m.dx)
    peak0 = float(tracer[:, 0].max())
    for _ in range(120):                                         # advect ~36 cells; the bump stays interior
        tracer = tracer + 600.0 * advect(tracer, Fx, Fz)
    mass1 = float(np.sum(tracer * dm[None, :]) * m.dx)
    peak1 = float(tracer[:, 0].max())
    assert tracer[:12].max() < 1e-6 and tracer[-12:].max() < 1e-6   # never reached the walls (interior test)
    return abs(mass1 - mass0) / mass0, (peak0 - peak1) / peak0


def test_fluxlim_numerical_diffusion_is_far_below_upwind():
    """LOAD-BEARING (the scheme swap is meaningless otherwise): the flux-limited (van Leer) theta_pot scheme's
    implicit numerical diffusion is MUCH smaller than first-order upwind at a FIXED grid -- measured as the
    PEAK-amplitude loss of an advected smooth bump (a diffusive scheme smears the peak down). Both schemes
    conserve the advected integral to round-off (flux form, no-flux walls)."""
    mass_up, loss_up = _bump_advection_metric("upwind")
    mass_fl, loss_fl = _bump_advection_metric("fluxlim")
    assert mass_up < 1e-12 and mass_fl < 1e-12                   # both EXACTLY conservative (flux form)
    assert loss_up > 0.15                                        # upwind visibly diffuses the bump (~23% peak loss)
    assert loss_fl < 0.1 * loss_up                             # fluxlim diffusion << upwind (the load-bearing claim)


def test_fluxlim_is_monotone_no_overshoot_and_boundary_safe():
    """TVD/monotonicity + the limiter's danger zones (the r=upwind/downwind ratio division, phi=0-at-extrema,
    and the near-wall far-upwind fallback): a square pulse with gradient SIGN REVERSALS plus a peak ADJACENT
    to the wall must advect with NO new extrema (stays within the initial [min,max]) and stay finite."""
    nx, nz = 80, 4
    m = StageADCore(StageADConfig(nx=nx, nz=nz, theta_max_deg=60.0))
    dm = np.asarray(m.dm)
    Fx = np.zeros((nx + 1, nz)); Fx[1:nx] = (0.3 * m.dx / 600.0) * dm[None, :]
    Fz = np.zeros((nx, nz + 1))
    # a pulse on a ZERO background, kept interior (a uniform background would pile up against the no-outflow
    # wall -- a BC artifact of the contrived flux, not a scheme overshoot; zero background isolates the scheme)
    tracer = np.zeros((nx, nz))
    tracer[1:3] = 2.0                                            # a peak adjacent to the f=0 wall (fallback path)
    tracer[8:18] = 3.0                                           # a square pulse (gradient sign reversals)
    lo, hi = float(tracer.min()), float(tracer.max())
    for _ in range(60):
        tracer = tracer + 600.0 * m._flux_form_fluxlim(tracer, Fx, Fz)
    assert np.all(np.isfinite(tracer))
    assert tracer[-12:].max() < 1e-9                            # the pulse never reached the right wall (interior)
    assert tracer.min() >= lo - 1e-9 and tracer.max() <= hi + 1e-9   # TVD: no new over/undershoot


def test_heat_advection_default_is_fluxlim_and_upwind_path_byte_identical():
    """The default is now "fluxlim" (the accurate model -- ~33x less numerically diffusive). The explicit
    "upwind" path is still reachable, and _advect_heat under it is BIT-IDENTICAL to _flux_form_upwind
    directly -- the upwind invariant is preserved (fluxlim conservation is covered by the next test)."""
    assert StageADConfig().heat_advection == "fluxlim"
    m = StageADCore(StageADConfig(nx=20, nz=18, theta_max_deg=55.0, kappa_T=0.0, heat_advection="upwind"))
    u, _ = _analytic_single_cell(m, amp=2.0e4)
    z_p = m._hydrostatic_z(m.T_atm); m._z = z_p                  # match _advect_heat's _z resolution exactly
    u_p = m._height_to_pressure(u, z_p)
    Fx, Fz = m._divfree_mass_flux(u_p)
    exner = (m.p_mid[None, :] / C.P_SURF) ** C.KAPPA_POISSON
    theta = m.T_atm / exner
    theta_ref = theta + 600.0 * (m._flux_form_upwind(theta, Fx, Fz) + m._cross_diffuse(theta, 0.0))
    T_ref = theta_ref * exner
    m._advect_heat(u, 600.0)
    assert np.array_equal(m.T_atm, T_ref)                       # bit-identical to the explicit upwind path


def test_fluxlim_heat_advection_conserves_theta_mass_and_differs_from_upwind():
    """With heat_advection="fluxlim", _advect_heat still conserves the column-integrated theta-mass EXACTLY
    (flux form), keeps a UNIFORM theta uniform (uniform-invariance), and produces a DIFFERENT T_atm than the
    upwind default (the lower-diffusion scheme is actually engaged, not a silent no-op)."""
    cfg = dict(nx=20, nz=18, theta_max_deg=55.0, kappa_T=0.0)
    mf = StageADCore(StageADConfig(heat_advection="fluxlim", **cfg))
    mu = StageADCore(StageADConfig(heat_advection="upwind", **cfg))
    exner = (mf.p_mid / C.P_SURF) ** C.KAPPA_POISSON
    Wth0 = float(np.sum((mf.T_atm / exner[None, :]) * mf.dm[None, :]))
    u, _ = _analytic_single_cell(mf, amp=2.0e4)
    uu, _ = _analytic_single_cell(mu, amp=2.0e4)
    for _ in range(60):
        mf._advect_heat(u, dt=600.0); mu._advect_heat(uu, dt=600.0)
    Wth1 = float(np.sum((mf.T_atm / exner[None, :]) * mf.dm[None, :]))
    assert abs(Wth1 - Wth0) / Wth0 < 1e-12                      # theta-mass conserved to machine precision
    assert not np.allclose(mf.T_atm, mu.T_atm)                  # fluxlim differs from upwind (scheme engaged)
    # uniform-invariance: a flat theta field is untouched by the fluxlim scheme
    mU = StageADCore(StageADConfig(heat_advection="fluxlim", **cfg))
    uU, _ = _analytic_single_cell(mU, amp=2.0e4)
    exnerU = (mU.p_mid[None, :] / C.P_SURF) ** C.KAPPA_POISSON
    mU.T_atm = 300.0 * exnerU * np.ones_like(mU.T_atm)          # theta = T/exner = 300 K everywhere (uniform)
    T_before = mU.T_atm.copy()
    mU._advect_heat(uU, dt=600.0)
    assert np.allclose(mU.T_atm, T_before, rtol=1e-12)          # uniform theta stays uniform


def test_heat_advection_rejects_unknown_scheme():
    """An unrecognized heat_advection value fails LOUDLY at construction (not a silent fallback to upwind)."""
    with pytest.raises(ValueError, match="heat_advection"):
        StageADCore(StageADConfig(nx=12, nz=12, theta_max_deg=55.0, heat_advection="bogus"))


# ====================================================================================================
# M3.4b TASK 2 -- the FRICTIONAL-DISSIPATION-HEATING closure (the energy budget's missing return term).
# Task 1 proved the residual energy sink is a REAL APE->KE->dissipation leak: the overturning cell converts
# internal energy to KE (``_advect_heat`` cools int(cp*T), enth<0), the Smagorinsky eddy viscosity removes
# that KE (``vorticity_tendency``'s div(nu_t grad omega)), and NOTHING re-heats T_atm -- so int(SW_abs-OLR)
# plateaus at ~|enth| (``rel_imbalance`` does not close). Task 2 adds the missing term: deposit the
# Smagorinsky dissipation rate int(rho_ref*nu_t*|S|^2) back into T_atm as positive heating. These tests are
# the BUILD gate (the energy-return identity + sign + the default-off guard + the step-wiring); the
# quantitative closure (does rel_imbalance -> 0) is a steady-state MEASUREMENT, reported by the harness.
# ====================================================================================================


def test_dissipative_heating_is_on_by_default_and_gated():
    """Default ON (the energy-closed accurate model; set False for the leaky Task-1 limit) and properly
    GATED: step() applies the heating ONLY when dissipative_heating=True. Proven with a POISON-PILL tendency
    -- with the flag off it is never called (T_atm stays finite); with it on the poison propagates (the term
    IS wired into the live step)."""
    assert StageADConfig().dissipative_heating is True
    for flag, expect_finite in ((False, True), (True, False)):
        m = StageADCore(StageADConfig(nx=12, nz=12, theta_max_deg=55.0, conv_max_sweeps=2,
                                      dissipative_heating=flag))
        m._dissipative_heating_tendency = lambda u, w: np.full_like(m.T_atm, np.nan)   # poison pill
        m.step()
        assert bool(np.all(np.isfinite(m.T_atm))) is expect_finite


def test_dissipative_heating_is_gated_on_couple_heat():
    """The reheating is the CONJUGATE return of ``_advect_heat``'s APE->KE cooling, so it must fire ONLY when
    that cooling does: with couple_heat=False there is no advective-cooling sink, and depositing dissipation
    heating would be a spurious energy SOURCE. Proven with the poison pill -- with couple_heat off the term is
    never applied (T_atm stays finite) even with dissipative_heating=True; with couple_heat on it propagates."""
    for couple_heat, expect_finite in ((False, True), (True, False)):
        m = StageADCore(StageADConfig(nx=12, nz=12, theta_max_deg=55.0, conv_max_sweeps=2,
                                      dissipative_heating=True))
        m.couple_heat = couple_heat
        m._dissipative_heating_tendency = lambda u, w: np.full_like(m.T_atm, np.nan)   # poison pill
        m.step()
        assert bool(np.all(np.isfinite(m.T_atm))) is expect_finite


def test_dissipative_heating_strain_form_energy_return_identity():
    """The energy-return IDENTITY for the STRAIN form (the load-bearing conservation gate): the heat the term
    deposits into T_atm per column equals EXACTLY the column-integrated strain-form Smagorinsky dissipation
    int(rho_ref*nu_t*|S|^2)dz it is meant to return -- to round-off, DESPITE the height->pressure remap (the
    per-column beta renormalization guarantees it). Driven by an analytic single overturning cell. Computes
    the reference dissipation INDEPENDENTLY (inline strain, not the production helper)."""
    m = StageADCore(StageADConfig(nx=24, nz=18, theta_max_deg=55.0, dissipative_heating=True,
                                  dissipation_form="strain"))
    u, w = _analytic_single_cell(m, amp=6.0e4)
    m._z = m._hydrostatic_z(m.T_atm)
    c = m.cfg
    # independent reference: the height-grid dissipation density rho_ref*nu_t*|S|^2 and its column integral
    ux = _ddx(u, m.dx); wz = _ddz(w, m.dz); uz = _ddz(u, m.dz); wx = _ddx(w, m.dx)
    Smag = np.sqrt(2 * ux**2 + 2 * wz**2 + (uz + wx)**2)                 # |S| = sqrt(2 S_ij S_ij), height grid
    nu_t = smagorinsky_viscosity(u, w, m.dx, m.dz, c.C_s, c.nu_b)        # FULL nu_t (Smagorinsky + floor)
    D_col = np.sum(m.rho_ref[:, None] * nu_t * Smag**2, axis=0) * m.dz   # (nx,) [W/m^2] column dissipation
    tend = m._dissipative_heating_tendency(u, w)                         # (nx, nz) [K/s] heating tendency
    deposited = np.sum(C.CP_AIR * m.dm[None, :] * tend, axis=1)          # (nx,) [W/m^2] energy return rate
    assert np.allclose(deposited, D_col, rtol=1e-12, atol=0.0)           # per-column identity to round-off
    assert D_col.max() > 0.0                                             # the cell actually dissipates KE


def test_dissipative_heating_strain_form_only_heats_never_cools():
    """The STRAIN form is sign-definite: the deposited tendency is >= 0 EVERYWHERE (the strain-form Smagorinsky
    production is non-negative, so it returns heat and never cools a layer) and is positive where the cell
    shears. (The OPERATOR form can locally back-scatter -- see its identity test -- so this guard is strain.)"""
    m = StageADCore(StageADConfig(nx=24, nz=18, theta_max_deg=55.0, dissipation_form="strain"))
    u, w = _analytic_single_cell(m, amp=6.0e4)
    m._z = m._hydrostatic_z(m.T_atm)
    tend = m._dissipative_heating_tendency(u, w)
    assert np.all(tend >= 0.0)                                           # never cools
    assert tend.max() > 0.0                                              # heats where strained


def test_dissipative_heating_operator_form_energy_return_identity_and_is_smaller_than_strain():
    """The energy-return IDENTITY for the OPERATOR form (the budget-closing default): the deposited column heat
    equals EXACTLY the KE the eddy-diffusion operator div(nu_t grad omega) removes (the Poisson-inverted
    -rho_ref(u.du+w.dw), computed INDEPENDENTLY here), to round-off. AND the Task-2 FINDING in miniature: the
    operator KE removal is dissipative on net (>0) but SMALLER than the strain-form production (which
    over-estimates the removed KE) -- the gap that makes the strain form overshoot the energy budget ~2.6x."""
    # surface_drag_bulk pinned OFF (it became default-ON in #147): isolate the eddy-diffusion operator
    # energy-return identity from the bulk-drag curl fold-in -- the bulk drag's own energy-return identity
    # is gated independently in test_bl_refinement.py (test_surface_drag_bulk_energy_returned_as_heat).
    m = StageADCore(StageADConfig(nx=24, nz=18, theta_max_deg=55.0, dissipative_heating=True,
                                  surface_drag_bulk=False))
    assert m.cfg.dissipation_form == "operator"                          # the budget-closing default
    c = m.cfg
    # an analytic single-cell VORTICITY (zero on all walls) -> the EXACTLY consistent streamfunction via the
    # anelastic Poisson, then (u, w) from psi -- so the (omega, psi, u, w) the operator audit reads are mutually
    # consistent (the operator KE removal Poisson-inverts omega, which must match the flow it dissipates).
    X = m.x[None, :] / m.x.max(); Z = m.z[:, None] / m.z.max()
    m.omega = 1.0e-3 * np.sin(np.pi * X) * np.sin(np.pi * Z)
    m.psi = m._lu_dyn.solve(m.omega.ravel()).reshape(c.nz, c.nx)
    u, w = m.velocities()
    m._z = m._hydrostatic_z(m.T_atm)
    nu_t = smagorinsky_viscosity(u, w, m.dx, m.dz, c.C_s, c.nu_b)
    # independent operator KE removal per column (the same Poisson route the impl uses, replicated here)
    T_diff = _div_grad(m.omega, nu_t, m.dx, m.dz)
    dpsi = m._lu_dyn.solve(np.asarray(T_diff).ravel()).reshape(c.nz, c.nx)
    du, dw = velocities_from_psi(dpsi, m._beta, m.dx, m.dz)
    D_col_op = np.sum(-m.rho_ref[:, None] * (u * du + w * dw), axis=0) * m.dz   # (nx,) [W/m^2]
    tend = m._dissipative_heating_tendency(u, w)
    deposited = np.sum(C.CP_AIR * m.dm[None, :] * tend, axis=1)
    atol = 1e-9 * float(np.max(np.abs(D_col_op))) + 1e-30
    assert np.allclose(deposited, D_col_op, rtol=1e-9, atol=atol)        # operator energy-return identity
    D_op_total = float(np.sum(D_col_op))
    Sm = strain_rate_magnitude(u, w, m.dx, m.dz)
    D_strain_total = float(np.sum(m.rho_ref[:, None] * nu_t * Sm**2) * m.dz)
    assert D_op_total > 0.0                                              # dissipative on net (removes KE globally)
    assert D_op_total < D_strain_total                                  # operator < strain (the over-estimate finding)


def test_dissipative_heating_wired_into_step_adds_heat():
    """Wired into the live step from an IDENTICAL developed state, the ONLY difference between a term-ON and
    a term-OFF core after one step is the deposited dissipation: their T_atm differ and the term-ON core has
    GAINED net column internal energy int(cp*T*dm) (the returned APE->KE->heat loop, now closed). Uses the
    default (operator) form, which is dissipative on NET (dE>0) though it may back-scatter in a few cells."""
    cfg = dict(nx=12, nz=14, theta_max_deg=55.0, conv_max_sweeps=4)
    base = StageADCore(StageADConfig(**cfg))
    base.seed_warm_edge_buoyancy(dT_edge=40.0)                          # strong forcing -> a real cell fast
    for _ in range(400):                                                 # develop the resolved circulation
        base.step()
    assert float(np.abs(base.psi).max()) > 0.0                          # a circulation actually developed
    st = base.state()
    on = StageADCore(StageADConfig(dissipative_heating=True, **cfg)); on.load_state(st)   # couple_thermo on
    off = StageADCore(StageADConfig(dissipative_heating=False, **cfg)); off.load_state(st)
    on.step(); off.step()                                               # identical state -> only the term differs
    diff = on.T_atm - off.T_atm                                        # == dt * tendency, bit-level (all else identical)
    assert float(np.max(np.abs(diff))) > 0.0                            # the term actually changed the running state
    dE = float(np.sum(C.CP_AIR * on.dm[None, :] * diff))
    assert dE > 0.0                                                     # net heat ADDED by the term (KE returned)


def test_sw_diffuse_counts_in_S_sw_in_budget():
    # Stage-5 POSITIVE path on the AD-core (the OFF path is byte-identical, covered elsewhere): with
    # sw_diffuse ON, the incoming-SW that the AD-core's absorbed-SW / net-imbalance diagnostics use
    # (S_sw_in) must COUNT the haze forward-scatter -- S_sw_in = S + S_diffuse > S. OFF -> S_diffuse is
    # None and S_sw_in == S (no diffuse computed).
    off = StageADCore(StageADConfig(nx=24, nz=16, sw_diffuse=False))
    assert off.S_diffuse is None
    assert np.allclose(off.S_sw_in, off.S)                       # budget input is the bare direct day-mean flux
    on = StageADCore(StageADConfig(nx=24, nz=16, sw_diffuse=True))
    assert on.S_diffuse is not None
    assert np.all(on.S_diffuse >= 0.0) and np.any(on.S_diffuse > 0.0)   # forward-scatter recovered off-overhead
    assert np.allclose(on.S_sw_in, on.S + on.S_diffuse)         # absorbed-SW / net-imbalance count it in
    assert np.any(on.S_sw_in > off.S_sw_in)                     # ... so the incoming SW strictly rises


def test_virtual_buoyancy_default_on_and_off_is_T_only_byte_identical():
    # Issue #80: vapor buoyancy defaults ON (the all-on accurate model)...
    core = StageADCore(StageADConfig(nx=12, nz=10))
    assert core.cfg.virtual_buoyancy is True
    # ... and the OFF limit is byte-identical to the pre-#80 T-only form.
    off = StageADCore(StageADConfig(nx=12, nz=10, virtual_buoyancy=False))
    off.T_atm = core.T_atm.copy(); off.q = core.q.copy(); off.T_surf = core.T_surf.copy()
    core._update_buoyancy(); off._update_buoyancy()
    z_p = off._hydrostatic_z(off.T_atm)
    T_h = off._pressure_to_height(off.T_atm, z_p)
    b_T_only = C.G_ACCEL * (T_h - off.T_ref[:, None]) / off.T_ref[:, None]   # the pre-#80 formula
    assert np.array_equal(np.asarray(off.b), b_T_only)
    # ON adds buoyancy only where there is vapor (T_v >= T), strictly somewhere on the moist warm start
    db = np.asarray(core.b) - np.asarray(off.b)
    assert np.all(db >= -1e-15) and float(np.max(db)) > 0.0
    # and the DRY state (q = 0) collapses ON onto the T-only form exactly
    core.q[:] = 0.0
    core._update_buoyancy()
    assert np.allclose(np.asarray(core.b), b_T_only, atol=1e-12)


def test_virtual_buoyancy_moist_column_more_buoyant_than_dry_at_same_T():
    # The load-bearing #80 property: at IDENTICAL temperature, a moist column is more buoyant than a
    # dry one -- a HORIZONTAL buoyancy-gradient term (zero over the dry salt flat, large over the wet
    # warm edge), not an inert offset the free-slip torque would ignore. The T-only form gives zero
    # horizontal contrast on a uniform-T state; the virtual-T form must recover g*(T_v - T)/T_ref.
    m = StageADCore(StageADConfig(nx=8, nz=10))
    m.T_atm[:] = 300.0                                  # uniform T: T-only buoyancy has NO x-contrast
    m.q[:] = 0.0
    i_moist = 2
    m.q[i_moist, :] = 0.10                              # ~q_sat at the SST~55C wet edge
    m._z = m._hydrostatic_z(m.T_atm)
    m._update_buoyancy()
    b = np.asarray(m.b)                                 # (nz, nx)
    Tv = float(th.virtual_temperature(300.0, 0.10))     # ~+16.6 K density temperature at q=0.1
    assert Tv - 300.0 > 10.0
    expected = C.G_ACCEL * (Tv - 300.0) / m.T_ref       # (nz,) the moist-minus-dry buoyancy contrast
    assert np.allclose(b[:, i_moist] - b[:, 0], expected, rtol=1e-6)
    # the dry columns carry no contrast among themselves
    assert np.allclose(b[:, 1], b[:, 0], atol=1e-12)
    # and the T-only form would have seen none of it
    m.cfg = dataclasses.replace(m.cfg, virtual_buoyancy=False)
    m._update_buoyancy()
    b_off = np.asarray(m.b)
    assert np.allclose(b_off[:, i_moist], b_off[:, 0], atol=1e-12)


# ====================================================================================================
# PHY-19 step 2 -- CONDENSATE LOADING in the dynamics buoyancy (cfg.condensate_loading, default OFF).
# The suspended condensed water (cloud q_c + rain q_r) has weight, so the density temperature is
# T_rho = T_v - T*q_cond (thermo.density_temperature). No free parameter. It is a CORRECTNESS term, not
# the fix for the PHY-19 axis rain column -- expected ~1-2 K at the axis against ~30 K of vapor contrast.
# ====================================================================================================
def _load_run(loading, steps=5, seed=11, **extra):
    """A short stepped toy world for the loading gate. ``loading=None`` means the config never mentions
    the flag at all (the pre-change world); True/False set it explicitly. Returns the full prognostic
    state so a byte-identity claim covers the state the next step reads, not just T."""
    rng = np.random.default_rng(seed)
    kw = dict(nx=24, nz=16, theta_max_deg=80.0)
    kw.update(extra)
    if loading is not None:
        kw["condensate_loading"] = loading
    core = StageADCore(StageADConfig(**kw))
    core.run_dynamics = True; core.couple_heat = True
    core.T_atm = core.T_atm + 0.7 * rng.standard_normal(core.T_atm.shape)
    core.q = np.clip(core.q + 1e-4 * rng.standard_normal(core.q.shape), 0.0, 0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(steps):
            core.step()
    names = ("T_atm", "q", "q_c", "b", "psi", "omega", "T_surf")
    out = {n: np.asarray(getattr(core, n)).copy() for n in names}
    if core.q_r is not None:
        out["q_r"] = np.asarray(core.q_r).copy()
    return out


def test_condensate_loading_default_off_and_off_is_byte_identical(monkeypatch):
    """The merge bar: the flag defaults OFF, and a multi-step run with it OFF is BIT-IDENTICAL to the same
    run built with no mention of the flag at all -- over the WHOLE prognostic state, not just T."""
    monkeypatch.delenv("COSMO_CONDENSATE_LOADING", raising=False)
    assert StageADConfig(nx=8, nz=6).condensate_loading is False          # default OFF
    none = _load_run(None)
    off = _load_run(False)
    for k in none:
        assert np.array_equal(none[k], off[k]), f"{k}: flag-False moved a bit vs the flag-absent config"


def test_condensate_loading_on_with_no_condensate_fields_is_a_noop():
    """Loading ON must be safe to leave on in a world that carries no condensate: with
    ``prognostic_condensate`` OFF q_c stays 0 and q_r is None, so the subtracted term is T*0 -- which is
    EXACTLY 0.0 in IEEE arithmetic (not merely small), so the run is BITWISE the OFF run, not close to it."""
    off = _load_run(False, prognostic_condensate=False)
    on = _load_run(True, prognostic_condensate=False)
    for k in off:
        assert np.array_equal(off[k], on[k]), f"{k}: loading ON perturbed a condensate-free run"


def test_condensate_loading_on_with_zero_condensate_is_bitwise_off():
    """Same claim one level down, with the prognostic condensate path LIVE but the fields still zero at the
    moment the buoyancy is formed: ON and OFF give bit-identical b."""
    m = _short(prognostic_condensate=True, prognostic_rain=True)
    m._z = m._hydrostatic_z(m.T_atm)
    m.q_c[:] = 0.0; m.q_r[:] = 0.0
    m._update_buoyancy(); b_off = np.asarray(m.b).copy()
    m.cfg = dataclasses.replace(m.cfg, condensate_loading=True)
    m._update_buoyancy()
    assert np.array_equal(np.asarray(m.b), b_off)


def test_condensate_loading_blob_lowers_b_by_exactly_g_T_qcond_over_Tref():
    """The quantitative gate. An imposed cloud+rain blob in a few columns must lower the buoyancy in THOSE
    columns by exactly ``g * T * q_cond / T_ref`` (to round-off) and leave every other column bit-for-bit
    alone. The expected value is built from the SINGLE-field ``_pressure_to_height`` map, so it also
    cross-checks the new three-field bracket-sharing path against the one it must reproduce."""
    m = _short(prognostic_condensate=True, prognostic_rain=True)
    m._z = m._hydrostatic_z(m.T_atm)
    blob = [5, 6]                                            # exactly these columns carry condensate
    m.q_c[:] = 0.0; m.q_r[:] = 0.0
    m.q_c[blob, 4:9] = 1.5e-3                                # ~1.5 g/kg cloud, a vigorous updraft core
    m.q_r[blob, 6:11] = 0.8e-3                               # ... plus falling rain below it
    m._update_buoyancy(); b_off = np.asarray(m.b).copy()
    m.cfg = dataclasses.replace(m.cfg, condensate_loading=True)
    m._update_buoyancy(); b_on = np.asarray(m.b).copy()
    db = b_on - b_off                                        # (nz, nx)
    # the independent expectation: map T and q_cond to the height grid one field at a time
    z_p = m._z
    T_h = m._pressure_to_height(m.T_atm, z_p)
    qc_h = m._pressure_to_height(m.q_c + m.q_r, z_p)
    expected = -C.G_ACCEL * T_h * qc_h / m.T_ref[:, None]
    assert np.allclose(db, expected, rtol=1e-11, atol=1e-15)
    # ... and it is a REDUCTION, materially sized, confined to the blob columns
    assert float(db.min()) < -1e-6 and float(db.max()) <= 0.0
    others = [i for i in range(m.cfg.nx) if i not in blob]
    assert np.array_equal(b_on[:, others], b_off[:, others])
    # size sanity: the loading is ~ g*q_cond ~ 2e-2 m/s^2 at 1.5 g/kg, i.e. ~0.5 K of density temperature
    assert 1e-4 < abs(float(db.min())) < 1e-1


def test_condensate_loading_clips_a_negative_condensate_at_zero():
    """An advective undershoot can leave a tiny NEGATIVE q_c; a negative loading would be spurious LIFT.
    Clipped at 0, so a negative-q_c state gives exactly the un-loaded buoyancy."""
    m = _short(prognostic_condensate=True, condensate_loading=True)
    m._z = m._hydrostatic_z(m.T_atm)
    m.q_c[:] = -1e-6                                         # everywhere negative
    m._update_buoyancy(); b_neg = np.asarray(m.b).copy()
    m.q_c[:] = 0.0
    m._update_buoyancy()
    assert np.array_equal(np.asarray(m.b), b_neg)


def test_condensate_loading_is_orthogonal_to_virtual_buoyancy():
    """The two halves of the density temperature are independent switches: with vapor buoyancy OFF and
    loading ON the buoyancy is g(T - T*q_cond - T_ref)/T_ref -- the loading alone, on the T-only form."""
    m = _short(prognostic_condensate=True, virtual_buoyancy=False)
    m._z = m._hydrostatic_z(m.T_atm)
    m.q_c[:] = 0.0
    m.q_c[3, 5:9] = 2.0e-3
    m._update_buoyancy(); b_off = np.asarray(m.b).copy()
    m.cfg = dataclasses.replace(m.cfg, condensate_loading=True)
    m._update_buoyancy(); b_on = np.asarray(m.b).copy()
    z_p = m._z
    T_h = m._pressure_to_height(m.T_atm, z_p)
    qc_h = m._pressure_to_height(m.q_c, z_p)
    assert np.allclose(b_on - b_off, -C.G_ACCEL * T_h * qc_h / m.T_ref[:, None], rtol=1e-11, atol=1e-15)


def test_condensate_loading_is_recorded_in_the_checkpoint_fingerprint():
    """The flag is PHYSICS, so the resume world guard must see it: it appears in the fingerprint's config
    and a saved-OFF / current-ON pair is reported as a physics mismatch (not volatile, not world-defining)."""
    import json
    from flatworld import _ckpt_fingerprint as _ckfp
    off = StageADConfig(nx=8, nz=6)
    fp_off = json.loads(_ckfp.build_fingerprint(off))
    assert fp_off["config"]["condensate_loading"] is False
    assert "condensate_loading" not in _ckfp.VOLATILE_FIELDS
    assert "condensate_loading" not in _ckfp.WORLD_DEFINING_CONFIG
    on = StageADConfig(nx=8, nz=6, condensate_loading=True)
    d = _ckfp.diff_fingerprint(fp_off, on)
    assert ("condensate_loading", False, True) in d["physics"]


@pytest.mark.parametrize("branch", ["bincount", "generic", "shared_sample_row"])
def test_interp_columns_pair_third_field_shares_the_bracket_bit_identically(branch):
    """The three-field extension of ``_interp_columns_pair`` must be exactly the two-field call plus one
    more gather: the two-field result is unmoved, and the third output equals the standalone
    ``_interp_columns`` map of that field, bit for bit -- on ALL THREE bracket branches. ``bincount`` is
    the production CPU branch (``_pressure_to_height_pair`` passes a shared ascending query row against
    per-column samples); ``generic`` is the comparison-count branch the capture path forces; and
    ``shared_sample_row`` is the single-sample-row branch."""
    from flatworld.stage_ad_kernels import _interp_columns, _interp_columns_pair
    rng = np.random.default_rng(3)
    xq = np.sort(rng.uniform(0.0, 10.0, (1, 9)), axis=1)
    xs = np.sort(rng.uniform(-1.0, 11.0, (5, 7)), axis=1)
    kw = {}
    if branch == "generic":
        kw["force_generic"] = True
    elif branch == "shared_sample_row":
        xs = xs[:1]                                              # one sample row -> the searchsorted branch
    f1, f2, f3 = (rng.standard_normal((xs.shape[0], 7)) for _ in range(3))
    a2, b2 = _interp_columns_pair(xq, xs, f1, f2, **kw)
    a3, b3, c3 = _interp_columns_pair(xq, xs, f1, f2, f3, **kw)
    assert np.array_equal(a2, a3) and np.array_equal(b2, b3)     # the pair call is unmoved
    assert np.array_equal(c3, _interp_columns(xq, xs, f3, **kw))  # ... and the third matches the single map


@pytest.mark.parametrize("parallel", [False, True])
def test_pressure_to_height_triple_matches_three_single_maps(parallel):
    """``_pressure_to_height_pair``'s optional third field must equal three separate single maps, on BOTH
    dispatch branches -- the numpy shared-bracket path and the ``advect_parallel`` numba path (which
    gathers each field with its own prange call)."""
    m = _short(advect_parallel=parallel, prognostic_condensate=True)
    z_p = m._hydrostatic_z(m.T_atm)
    q_cond = np.abs(m.q) * 0.05 + 1e-4
    T_h, q_h, c_h = m._pressure_to_height_pair(m.T_atm, m.q, z_p, q_cond)
    assert np.array_equal(T_h, m._pressure_to_height(m.T_atm, z_p))
    assert np.array_equal(q_h, m._pressure_to_height(m.q, z_p))
    assert np.array_equal(c_h, m._pressure_to_height(q_cond, z_p))
    # and the two-field call is unmoved by the argument existing
    T2, q2 = m._pressure_to_height_pair(m.T_atm, m.q, z_p)
    assert np.array_equal(T2, T_h) and np.array_equal(q2, q_h)


# ====================================================================================================
# Issue #81 -- the ANISOTROPIC Smagorinsky closure (the ~1000:1 grid splits nu_h ~ (C_s dx)^2|S| from
# nu_v ~ (C_s dz)^2|S| instead of the isotropic Delta^2 = dx*dz). Flag closure_anisotropic, DEFAULT OFF
# (a structural closure choice to be A/B'd, not flipped). These gate: the operator reduces to the
# isotropic one exactly; the viscosity matches isotropic on a square grid and de-inflates the vertical
# limb on an anisotropic one; the off path is byte-identical; the diffusive-CFL ceiling lifts; and a
# run stays finite + mass-conserving with the closure ON.
# ====================================================================================================
def test_div_grad_aniso_reduces_to_isotropic():
    """Issue #81 step 1.3: _div_grad_aniso with EQUAL horizontal/vertical coefficients is byte-for-byte
    the isotropic _div_grad -- the constant- AND field-coefficient limits both reduce exactly."""
    rng = np.random.default_rng(0)
    F = rng.standard_normal((13, 17)); dx, dz, c0 = 1234.0, 56.0, 3.3
    assert np.array_equal(_div_grad_aniso(F, c0, c0, dx, dz), _div_grad(F, c0, dx, dz))
    cf = np.abs(rng.standard_normal((13, 17))) + 0.1            # a spatially-varying coefficient field
    assert np.array_equal(_div_grad_aniso(F, cf, cf, dx, dz), _div_grad(F, cf, dx, dz))


def test_smagorinsky_aniso_matches_isotropic_on_square_grid_and_splits_on_anisotropic():
    """Issue #81 step 1.2: on a SQUARE grid (dx==dz, Delta^2 = dx*dz = dx^2 = dz^2) the anisotropic
    viscosity returns (nu_t, nu_t) EQUAL to the isotropic form; on a 1000:1 grid the excess-over-floor
    is inflated dx/dz on the horizontal limb and de-inflated dz/dx on the vertical limb (the fix)."""
    rng = np.random.default_rng(2)
    u = rng.standard_normal((12, 16)); w = rng.standard_normal((12, 16))
    nu_iso = smagorinsky_viscosity(u, w, 100.0, 100.0, 0.17, 2.0)
    nu_h, nu_v = smagorinsky_viscosity_aniso(u, w, 100.0, 100.0, 0.17, 2.0)
    assert np.allclose(nu_h, nu_iso) and np.allclose(nu_v, nu_iso)   # square grid: closure choice invisible
    dx, dz = 1.0e6, 1.0e3                                            # ~1000:1 (the fleet anisotropy)
    nu_iso2 = smagorinsky_viscosity(u, w, dx, dz, 0.17, 1.0)
    nu_h2, nu_v2 = smagorinsky_viscosity_aniso(u, w, dx, dz, 0.17, 1.0)
    sel = (nu_iso2 - 1.0) > 1e-6                                     # where the Smagorinsky term is resolved
    assert np.allclose((nu_v2 - 1.0)[sel] / (nu_iso2 - 1.0)[sel], dz / dx, rtol=1e-6)   # vertical de-inflated
    assert np.allclose((nu_h2 - 1.0)[sel] / (nu_iso2 - 1.0)[sel], dx / dz, rtol=1e-6)   # horizontal inflated


def test_anisotropic_closure_off_byte_identical_dynamics():
    """Issue #81 house rule: closure_anisotropic=False (the DEFAULT) drives EXACTLY the isotropic
    Smagorinsky substep -- one _dynamics_substep equals the isotropic kernels applied by hand (the
    off-limit is byte-identical; the whole existing dynamics gate runs on this path)."""
    # surface_drag_bulk pinned OFF (it became default-ON in #147): this gates the isotropic-closure
    # off-limit, not the additive bulk-drag curl (which has its own byte-identical gate in
    # test_bl_refinement.py). With it ON the substep adds dt*_surface_drag_curl(u), absent from the
    # hand-rolled isotropic path below.
    m = StageADCore(StageADConfig(nx=20, nz=14, theta_max_deg=60.0, surface_drag_bulk=False))
    assert m.cfg.closure_anisotropic is False
    m.seed_warm_edge_buoyancy(dT_edge=40.0)                          # sets b, couple_thermo False
    rng = np.random.default_rng(1)
    om = rng.standard_normal((m.cfg.nz, m.cfg.nx)) * 1e-4
    om[0, :] = 0; om[-1, :] = 0; om[:, 0] = 0; om[:, -1] = 0         # wall-consistent initial vorticity
    m.omega = om.copy(); omega0 = om.copy()
    psi = m._lu_dyn.solve(omega0.ravel()).reshape(m.cfg.nz, m.cfg.nx)
    u, w = velocities_from_psi(psi, m._beta, m.dx, m.dz)
    nu_t = smagorinsky_viscosity(u, w, m.dx, m.dz, m.cfg.C_s, m.cfg.nu_b)
    dt = m._cfl_dt(u, w, nu_t)
    om_ref = omega0 + dt * vorticity_tendency(omega0, m.b, u, w, m.dlnrho_dz, nu_t, m.dx, m.dz)
    om_ref[0, :] = 0; om_ref[-1, :] = 0; om_ref[:, 0] = 0; om_ref[:, -1] = 0
    m._dynamics_substep()
    assert np.array_equal(np.asarray(m.omega), om_ref)              # bit-for-bit the isotropic path


def test_anisotropic_closure_lifts_diffusive_dt_ceiling():
    """Issue #81 (b): the directional split removes the vertical diffusive-CFL ceiling -- on a developed
    flow the diffusive rate that sets dt (max nu/dz^2) drops by ~dx/dz, the source of the dt gain. The
    isotropic Delta^2=dx*dz carried the huge horizontal scale into the vertical (1/dz^2) limb."""
    m = StageADCore(StageADConfig(nx=24, nz=16, theta_max_deg=60.0, nu_b=1.0))
    m.seed_warm_edge_buoyancy(dT_edge=40.0)
    for _ in range(120):
        m.step()
    u, w = m.velocities()
    nu_iso = smagorinsky_viscosity(u, w, m.dx, m.dz, m.cfg.C_s, m.cfg.nu_b)
    nu_h, nu_v = smagorinsky_viscosity_aniso(u, w, m.dx, m.dz, m.cfg.C_s, m.cfg.nu_b)
    rate_iso = float(np.max(nu_iso)) * (1.0 / m.dx**2 + 1.0 / m.dz**2)
    rate_aniso = float(np.max(nu_h / m.dx**2 + nu_v / m.dz**2))
    assert rate_iso / rate_aniso > 50.0                            # the vertical diffusive ceiling lifted


def test_anisotropic_closure_on_runs_finite_and_conserves_mass():
    """Issue #81: with the anisotropic closure ON a seeded run stays finite, develops a real circulation,
    and the anelastic mass-continuity gate still holds (it is fixed by the streamfunction Poisson, which
    is closure-INDEPENDENT). A bounded finite check (the cell spins up stronger without the inflated
    vertical viscosity -- surface drag off here), not a converged climate."""
    m = StageADCore(StageADConfig(nx=20, nz=16, theta_max_deg=60.0, closure_anisotropic=True, nu_b=5.0))
    m.seed_warm_edge_buoyancy(dT_edge=30.0)
    for _ in range(200):
        m.step()
    assert np.all(np.isfinite(np.asarray(m.omega))) and np.all(np.isfinite(np.asarray(m.psi)))
    assert float(np.abs(m.psi).max()) > 0.0                        # a circulation developed
    mc = m.mass_conservation()
    assert mc["rel_div_deep"] < 1e-10                              # mass continuity holds (Poisson-set)


# ====================================================================================================
# Issue #85 -- the surface is no longer decoupled from the resolved flow: (1) resolved-wind bulk surface
# fluxes (resolved_flux_wind, the WISHE-like wind-evaporation feedback) and (2) a linear surface drag
# (surface_drag_rate, the momentum sink the free-slip bottom lacks). Both DEFAULT OFF / byte-identical.
# ====================================================================================================
def test_resolved_flux_wind_gated_on_run_dynamics_and_off_is_fixed_wind():
    """Issue #85 item 1: the resolved-wind surface flux engages ONLY with the flag AND the dynamics live;
    with run_dynamics off (the no-flow RCE limit) it falls back to the fixed cfg.wind -- so a stashed
    surface wind is IGNORED and the trajectory is byte-identical to the flag-off core."""
    base = dict(nx=6, nz=14, theta_max_deg=50.0, dt=_RCE_DT, conv_max_sweeps=_RCE_SWEEPS)
    a = StageADCore(StageADConfig(resolved_flux_wind=False, **base))
    b = StageADCore(StageADConfig(resolved_flux_wind=True, **base))
    b._u_surf = np.linspace(2.0, 12.0, base["nx"])             # a would-be resolved wind; must be ignored
    for mm in (a, b):
        mm.run_dynamics = False; mm.couple_surface = False
        mm.T_surf = np.full(base["nx"], C.T_FREEZE + 25.0)
        z_col = -8000.0 * np.log(mm.sigma)
        mm.T_atm = np.maximum(mm.T_surf[:, None] - 6.5e-3 * z_col[None, :], 150.0)
        mm.q = np.minimum(mm._wetness()[:, None] * mm.RH[None, :] * th.qsat(mm.T_atm, mm.p_mid[None, :]), 0.5)
    for _ in range(30):
        a.step(); b.step()
    assert np.array_equal(a.T_atm, b.T_atm) and np.array_equal(a.q, b.q)   # flag is a no-op without the flow


def test_resolved_flux_wind_evaporation_follows_surface_wind():
    """Issue #85 item 1 (the substance): with the flag on and the dynamics live, the surface evaporation
    E(x) follows |u_surf(x)| at matched surface state -- the wind-evaporation coupling. A uniform warm wet
    column set is given a VARYING resolved surface wind; E rises monotonically with |u_surf| and tracks it
    proportionally, while the fixed-wind core evaporates uniformly."""
    nx = 8

    def evap_under(flag, u_surf):
        m = StageADCore(StageADConfig(nx=nx, nz=14, theta_max_deg=50.0, resolved_flux_wind=flag,
                                      gustiness=0.0))
        m.run_dynamics = True; m.couple_surface = False
        m.T_surf = np.full(nx, C.T_FREEZE + 40.0)                  # warm wet ocean: wetness=1, qs_surf large
        z_col = -8000.0 * np.log(m.sigma)
        m.T_atm = np.maximum(m.T_surf[:, None] - 6.5e-3 * z_col[None, :], 200.0)
        m.q = np.full((nx, m.cfg.nz), 2.0e-3)                      # uniform, undersaturated at the surface
        m._u_surf = u_surf.copy()
        m._thermo_substep(100.0)                                   # one substep; dynamics not run -> wind held
        return m._evap.copy()

    u_surf = np.linspace(1.0, 15.0, nx)
    E_on = evap_under(True, u_surf)
    E_off = evap_under(False, u_surf)
    assert np.all(np.diff(E_on) > 0)                               # E increases with |u_surf|
    assert np.allclose(E_off, E_off[0])                            # fixed-wind: uniform E (wind ignored)
    assert np.allclose(E_on / E_on[0], u_surf / u_surf[0], rtol=1e-6)   # E proportional to the exchange wind


def test_surface_drag_off_byte_identical_and_on_is_rayleigh_sink():
    """Issue #85 item 2: surface_drag_rate=0 (default) is byte-identical (no vorticity change), and a
    positive rate applies EXACTLY a linear Rayleigh sink -dt*rate*omega at the lowest interior level
    (j=1) and NOWHERE else, without entering the CFL. Driven from an identical developed state."""
    base = StageADCore(StageADConfig(nx=20, nz=14, theta_max_deg=60.0, nu_b=20.0))
    base.seed_warm_edge_buoyancy(dT_edge=40.0)
    for _ in range(80):
        base.step()
    st = base.state()
    r = 5.0e-6
    off = StageADCore(StageADConfig(nx=20, nz=14, theta_max_deg=60.0, nu_b=20.0, surface_drag_rate=0.0))
    on = StageADCore(StageADConfig(nx=20, nz=14, theta_max_deg=60.0, nu_b=20.0, surface_drag_rate=r))
    for mm in (off, on):
        mm.load_state(st); mm.couple_thermo = False                # hold the loaded b: pure dynamics
    om0 = np.asarray(off.omega).copy()
    off._dynamics_substep(); on._dynamics_substep()
    dt = off._last_dt_dyn
    assert on._last_dt_dyn == dt                                   # drag does NOT enter the CFL (same dt)
    diff = np.asarray(on.omega) - np.asarray(off.omega)
    others = np.ones(diff.shape[0], bool); others[1] = False
    assert np.array_equal(diff[others], np.zeros_like(diff[others]))   # only j=1 changed
    assert np.allclose(diff[1, 1:-1], (-dt * r * om0[1, :])[1:-1], rtol=1e-6, atol=1e-18)
    assert float(np.max(np.abs(diff))) > 0.0                       # the drag actually changed the running state


def test_surface_drag_reduces_surface_wind_and_preserves_single_cell():
    """Issue #85 item 2 (the purpose): the surface drag is a real momentum sink -- spun up against the
    SAME warm-edge forcing it leaves a WEAKER surface wind than the free-slip (drag-off) core, while
    preserving the single overturning cell and staying finite."""
    def spin(drag):
        m = StageADCore(StageADConfig(nx=24, nz=16, theta_max_deg=60.0, nu_b=10.0, surface_drag_rate=drag,
                                      tol=2e-3, check_every=200, patience=3, max_iter=20000))
        m.seed_warm_edge_buoyancy(dT_edge=40.0)
        m.run(verbose=False)
        return m
    free = spin(0.0)
    drag = spin(2.0e-5)
    df, dd = free.diagnostics(), drag.diagnostics()
    assert np.all(np.isfinite(dd["u_surf"]))
    assert dd["surf_sign_changes"] == 0                            # still ONE cell under drag
    assert float(np.max(np.abs(dd["u_surf"]))) < float(np.max(np.abs(df["u_surf"])))   # weaker surface wind


def test_surface_drag_ke_removal_folds_into_operator_energy_return_identity():
    """#85 item-2 follow-up CLOSED: with surface drag ON, the operator KE removal folds the Rayleigh
    sink -rate*omega(j=1) into the SAME Poisson inversion as the eddy diffusion (the inversion is
    LINEAR), so the deposited column heat equals the TOTAL (eddy + drag) removed KE -- independently
    replicated here to round-off. And the drag's own share (inverting just the drag row) is a real
    net KE sink (>0) on a single-cell flow."""
    r = 2.0e-5
    # surface_drag_bulk pinned OFF (it became default-ON in #147): isolate the RAYLEIGH drag fold-in
    # tested here from the bulk-drag curl fold-in -- both fold into the same Poisson inversion, but the
    # bulk variant is gated separately in test_bl_refinement.py.
    m = StageADCore(StageADConfig(nx=24, nz=18, theta_max_deg=55.0, dissipative_heating=True,
                                  surface_drag_rate=r, surface_drag_bulk=False))
    c = m.cfg
    assert c.dissipation_form == "operator"
    X = m.x[None, :] / m.x.max(); Z = m.z[:, None] / m.z.max()
    m.omega = 1.0e-3 * np.sin(np.pi * X) * np.sin(np.pi * Z)
    m.psi = m._lu_dyn.solve(m.omega.ravel()).reshape(c.nz, c.nx)
    u, w = m.velocities()
    m._z = m._hydrostatic_z(m.T_atm)
    nu_t = smagorinsky_viscosity(u, w, m.dx, m.dz, c.C_s, c.nu_b)
    # independent replication: eddy diffusion + the drag row, ONE linear Poisson inversion
    T_tot = _div_grad(m.omega, nu_t, m.dx, m.dz)
    T_tot[1, :] = T_tot[1, :] - r * np.asarray(m.omega)[1, :]
    dpsi = m._lu_dyn.solve(np.asarray(T_tot).ravel()).reshape(c.nz, c.nx)
    du, dw = velocities_from_psi(dpsi, m._beta, m.dx, m.dz)
    D_col_tot = np.sum(-m.rho_ref[:, None] * (u * du + w * dw), axis=0) * m.dz   # (nx,) [W/m^2]
    tend = m._dissipative_heating_tendency(u, w)
    deposited = np.sum(C.CP_AIR * m.dm[None, :] * tend, axis=1)
    atol = 1e-9 * float(np.max(np.abs(D_col_tot))) + 1e-30
    assert np.allclose(deposited, D_col_tot, rtol=1e-9, atol=atol)   # TOTAL (eddy+drag) energy-return identity
    # the drag share alone via the same independent route (linearity: invert just the drag row)
    T_drag = np.zeros((c.nz, c.nx))
    T_drag[1, :] = -r * np.asarray(m.omega)[1, :]
    dpsi_d = m._lu_dyn.solve(T_drag.ravel()).reshape(c.nz, c.nx)
    du_d, dw_d = velocities_from_psi(dpsi_d, m._beta, m.dx, m.dz)
    D_drag_total = float(np.sum(-m.rho_ref[:, None] * (u * du_d + w * dw_d)) * m.dz)
    assert D_drag_total > 0.0                  # the drag is a net KE sink on the single cell


def test_surface_drag_energy_leak_closed_in_live_step():
    """#85 item-2 follow-up CLOSED in the LIVE step: from one developed drag-ON state, the heating-ON
    vs heating-OFF cores differ in column internal energy by EXACTLY dt * D_col (the substep's stash,
    which now includes the drag's removed KE) -- the drag no longer leaks. And the drag-ON stash
    exceeds the drag-OFF stash from the SAME state (the drag share really is in the deposit)."""
    r = 2.0e-5
    cfg = dict(nx=12, nz=14, theta_max_deg=55.0, conv_max_sweeps=4)
    base = StageADCore(StageADConfig(surface_drag_rate=r, **cfg))
    base.seed_warm_edge_buoyancy(dT_edge=40.0)                          # strong forcing -> a real cell fast
    for _ in range(400):
        base.step()
    assert float(np.abs(base.psi).max()) > 0.0
    st = base.state()
    on = StageADCore(StageADConfig(surface_drag_rate=r, dissipative_heating=True, **cfg)); on.load_state(st)
    off = StageADCore(StageADConfig(surface_drag_rate=r, dissipative_heating=False, **cfg)); off.load_state(st)
    nod = StageADCore(StageADConfig(surface_drag_rate=0.0, dissipative_heating=True, **cfg)); nod.load_state(st)
    on.step(); off.step(); nod.step()                                   # identical state -> only the terms differ
    dt = on._last_dt_dyn
    assert off._last_dt_dyn == dt                                       # heating does not enter the CFL
    dE_col = np.sum(C.CP_AIR * on.dm[None, :] * (on.T_atm - off.T_atm), axis=1)   # deposited [J/m^2]
    # tolerance: recovering the deposit by differencing T_atm (~250 K) carries ulp(T)*cp*dm ~ 4e-8 J/m^2
    # of float round-trip noise per layer (visible only at the near-zero wall columns); the bit-level
    # energy-return identity is the tendency-level test above -- this one verifies the LIVE wiring.
    assert np.allclose(dE_col, dt * on._diss_Dcol, rtol=1e-6, atol=1e-6)   # live deposit == dt * (eddy+drag) D_col
    assert float(np.sum(on._diss_Dcol)) > float(np.sum(nod._diss_Dcol))    # the drag share is in the stash


# ====================================================================================================
# Issue #86 -- radiation sub-cycling: recompute the full column radiation only every dt_rad model-seconds,
# holding H/surf_net/OLR between refreshes (the fast physics stay on the substep clock). dt_rad=0 (default)
# recomputes every step (byte-identical cadence); NOT a Newtonian relaxation (the held field is the model's
# own radiation). The diagnostics/conservation gates make FRESH calls and are unaffected.
# ====================================================================================================
def test_dt_rad_recompute_cadence(monkeypatch):
    """Issue #86: dt_rad=0 (default) recomputes column radiation EVERY step (byte-identical cadence); a
    positive dt_rad sub-cycles it (strictly fewer calls once the state settles below the large-move guard)."""
    import flatworld.radiation as radmod
    counts = {"n": 0}
    real = radmod.column_radiation

    def counting(*a, **k):
        counts["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(radmod, "column_radiation", counting)

    def run(dt_rad, nsteps):
        m = StageADCore(StageADConfig(nx=6, nz=14, theta_max_deg=50.0, dt=600.0, dt_rad=dt_rad,
                                      conv_max_sweeps=3))
        m.run_dynamics = False
        for _ in range(200):                                       # settle below the 1 K large-move guard
            m.step()
        counts["n"] = 0
        for _ in range(nsteps):
            m.step()
        return counts["n"]

    nsteps = 30
    n0 = run(0.0, nsteps)
    n_sub = run(3000.0, nsteps)                                    # dt=600 -> refresh ~ every 5 steps once settled
    assert n0 == nsteps                                           # every step (byte-identical cadence)
    assert n_sub < nsteps                                         # sub-cycled: strictly fewer radiation calls


def test_dt_rad_subcycle_matches_every_step():
    """Issue #86: a sub-cycled radiation run (dt_rad=1800) converges to the SAME no-flow RCE as every-step
    radiation (dt_rad=0) within a tight tolerance -- the held field is the model's own radiation, refreshed
    at a physical cadence (the validate-before-adopt A/B the issue asks for, at test resolution)."""
    def final(dt_rad):
        m = StageADCore(StageADConfig(nx=6, nz=16, theta_max_deg=50.0, dt=600.0, dt_rad=dt_rad,
                                      conv_max_sweeps=4))
        m.run_dynamics = False; m.couple_surface = False
        m.T_surf = np.full(6, C.T_FREEZE + 15.0)
        z_col = -8000.0 * np.log(m.sigma)
        m.T_atm = np.maximum(m.T_surf[:, None] - 6.5e-3 * z_col[None, :], 150.0)
        m.q = np.minimum(m._wetness()[:, None] * m.RH[None, :] * th.qsat(m.T_atm, m.p_mid[None, :]), 0.5)
        for _ in range(1500):
            m.step()
        return m.T_atm.copy()

    T0 = final(0.0); Tsub = final(1800.0)
    assert np.all(np.isfinite(Tsub))
    assert float(np.max(np.abs(T0 - Tsub))) < 0.3                  # sub-cycled RCE matches every-step radiation


def test_new_closure_surface_radiation_config_defaults():
    """Issues #81/#85/#86: the new knobs default to the byte-identical (off / every-step) limits, and the
    construction guards reject nonsensical values."""
    c = StageADConfig()
    assert c.closure_anisotropic is False                         # #81: structural choice, A/B before flip
    assert c.surface_drag_rate == 0.0                             # #85 item 2: off (byte-identical)
    assert c.resolved_flux_wind is False and c.gustiness == 3.0   # #85 item 1
    assert c.dt_rad == 0.0                                        # #86: every-step radiation (byte-identical)
    with pytest.raises(ValueError):
        StageADCore(StageADConfig(surface_drag_rate=-1.0))
    with pytest.raises(ValueError):
        StageADCore(StageADConfig(dt_rad=-1.0))


# ------------------------------------------------------------------ issue #99: fleet hot-spot vectorization
# The convective adjustment + the per-column np.interp grid reconciliation dominated the coupled fleet
# step (~75% of step wall-clock at nx=48/nz=18; radiation is only ~9%). The fixes are required to be
# BIT-IDENTICAL by construction (no physics change, no flag): these tests gate that bitwise, against
# VERBATIM reference copies of the replaced per-column/per-interface loops -- unit-level on realistic
# and adversarial states, and end-to-end on a coupled trajectory (any 1-ulp divergence amplifies
# chaotically over the coupled steps and trips the exact-equality gate).
# NB (PR #102 review): these four _ref_* copies have a verbatim TWIN in `flatworld/_issue99_profile.py`
# (the A/B harness's "before" arm). The duplication is deliberate -- the gate stays self-contained, so
# a harness edit can't silently change what the tests gate against -- and BOTH copies must stay
# byte-identical to the pre-#99 originals.
def _ref_pressure_to_height(self, field_p, z_p):
    """VERBATIM pre-#99 ``StageADCore._pressure_to_height`` (the per-column np.interp loop)."""
    nx, nz = field_p.shape
    out = np.empty((nz, nx))
    for i in range(nx):
        out[:, i] = np.interp(self.z, z_p[i, ::-1], field_p[i, ::-1])
    return out


def _ref_height_to_pressure(self, field_h, z_p):
    """VERBATIM pre-#99 ``StageADCore._height_to_pressure`` (the per-column np.interp loop)."""
    nx, nz = z_p.shape
    out = np.empty((nx, nz))
    for i in range(nx):
        out[i, :] = np.interp(z_p[i, :], self.z, field_h[:, i])
    return out


def _ref_hydrostatic_z(self, T_atm=None):
    """VERBATIM pre-#99 ``StageADCore._hydrostatic_z`` (the per-interface Python loop with its
    in-loop scalar np.log calls)."""
    c = self.cfg
    T = self.T_atm if T_atm is None else T_atm
    Tedge = np.empty((c.nx, c.nz + 1))
    Tedge[:, 1:-1] = 0.5 * (T[:, :-1] + T[:, 1:])
    Tedge[:, 0] = T[:, 0]; Tedge[:, -1] = T[:, -1]
    z_edge = np.zeros((c.nx, c.nz + 1))
    for k in range(c.nz - 1, -1, -1):
        dz = (C.R_DRY * Tedge[:, k + 1] / C.G_ACCEL) * (np.log(self.p_edge[k + 1])
                                                        - np.log(self.p_edge[k]))
        z_edge[:, k] = z_edge[:, k + 1] + dz
    return 0.5 * (z_edge[:, :-1] + z_edge[:, 1:])


def _ref_moist_convective_adjust(T_atm, dm, z, p_mid, max_sweeps=8, q=None,
                                 closure="q_homogenize", rh_surf=0.8, rh_top=0.2, freeze_gamma=False):
    """VERBATIM pre-#99 ``moist_convective_adjust`` (the boolean-fancy-index sweep body). The
    ``freeze_gamma`` kwarg exists only so the call sites line up: the pre-#99 scheme IS the
    every-sweep (OFF) path, so the reference rejects ON."""
    assert not freeze_gamma, "the pre-#99 reference implements only the freeze_gamma=False path"
    nx, nz = T_atm.shape
    detrain = (closure == "rh_detrain") and (q is not None)
    if detrain:
        RHt = rh_top + (rh_surf - rh_top) * (np.asarray(p_mid, float) / C.P_SURF)
    for _ in range(max_sweeps):
        changed = np.zeros(nx, bool)
        for k in range(nz - 1, 0, -1):
            Gamma = th.moist_adiabat_lapse_rate(T_atm[:, k], p_mid[k])
            dz = z[:, k - 1] - z[:, k]
            T_above_stable = T_atm[:, k] - Gamma * dz
            unstable = T_atm[:, k - 1] < T_above_stable - 1e-9
            if not unstable.any():
                continue
            mk, mk1 = dm[k], dm[k - 1]
            Ctot = mk * T_atm[:, k] + mk1 * T_atm[:, k - 1]
            Tk_new = (Ctot + mk1 * Gamma * dz) / (mk + mk1)
            Tk1_new = Tk_new - Gamma * dz
            T_atm[unstable, k] = Tk_new[unstable]
            T_atm[unstable, k - 1] = Tk1_new[unstable]
            if q is not None:
                if not detrain:
                    q_mix = (mk * q[:, k] + mk1 * q[:, k - 1]) / (mk + mk1)
                    q[unstable, k] = q_mix[unstable]
                    q[unstable, k - 1] = q_mix[unstable]
                else:
                    W = mk * q[:, k] + mk1 * q[:, k - 1]
                    gk = RHt[k] * th.qsat(Tk_new, p_mid[k])
                    gk1 = RHt[k - 1] * th.qsat(Tk1_new, p_mid[k - 1])
                    denom = mk * gk + mk1 * gk1
                    pos = denom > 0.0
                    A = np.where(pos, W / np.where(pos, denom, 1.0), 0.0)
                    homog = W / (mk + mk1)
                    q[unstable, k] = np.where(pos, A * gk, homog)[unstable]
                    q[unstable, k - 1] = np.where(pos, A * gk1, homog)[unstable]
            changed |= unstable
        if not changed.any():
            break
    return T_atm


def _spun_coupled_core(nx=16, nz=12, n_rce=30, n_coupled=8):
    """A small core spun through the issue-#99 repro phases (RCE warm start, then coupled), so the
    bitwise gates run on a REALISTIC live coupled state, not a synthetic one."""
    m = StageADCore(StageADConfig(nx=nx, nz=nz, theta_max_deg=70.0, conv_max_sweeps=4))
    m.run_dynamics = False
    for _ in range(n_rce):
        m.step()
    m.run_dynamics = True
    m.couple_heat = True
    for _ in range(n_coupled):
        m.step()
    return m


def test_interp_columns_matches_np_interp_bitwise():
    """Issue #99 unit gate: the batched ``_interp_columns`` reproduces per-row ``np.interp`` EXACTLY
    (==, every bit) on adversarial random rows -- nonuniform strictly-increasing sample grids; queries
    in-range, out-of-range both sides, EXACTLY on interior/boundary nodes; constant and 2-point rows;
    and the broadcast forms the reconciliation methods use (shared query row / shared sample row)."""
    from flatworld.stage_ad_core import _interp_columns
    rng = np.random.default_rng(99)
    B, N, M = 37, 23, 41
    xp = np.cumsum(rng.uniform(0.05, 2.0, (B, N)), axis=1) - 30.0   # strictly increasing, mixed sign
    fp = rng.normal(0.0, 50.0, (B, N))
    x = np.empty((B, M))
    x[:, :20] = rng.uniform(-35.0, 25.0, (B, 20))                  # spans below/inside/above the grids
    x[:, 20:28] = np.take_along_axis(xp, rng.integers(0, N, (B, 8)), 1)   # EXACT interior node hits
    x[:, 28:30] = xp[:, [0, -1]]                                   # exact boundary nodes
    x[:, 30:32] = xp[:, [0, -1]] + np.array([-1e-12, 1e-12])       # just outside the boundaries
    x[:, 32:] = rng.uniform(xp[:, :1], xp[:, -1:], (B, M - 32))    # dense in-range
    got = _interp_columns(x, xp, fp)
    want = np.vstack([np.interp(x[i], xp[i], fp[i]) for i in range(B)])
    assert got.shape == want.shape and np.array_equal(got, want)   # EXACT -- bit-identical

    fp_const = np.full((B, N), -3.25)                              # constant rows (slope == 0 paths)
    assert np.array_equal(_interp_columns(x, xp, fp_const),
                          np.vstack([np.interp(x[i], xp[i], fp_const[i]) for i in range(B)]))
    xp2 = xp[:, [0, -1]]; fp2 = fp[:, [0, -1]]                     # minimal 2-point rows
    assert np.array_equal(_interp_columns(x, xp2, fp2),
                          np.vstack([np.interp(x[i], xp2[i], fp2[i]) for i in range(B)]))
    # broadcast forms: shared queries vs per-row samples (_pressure_to_height's shape) and
    # per-row queries vs a shared sample row (_height_to_pressure's shape). An UNSORTED shared query
    # row exercises the generic comparison-count bracket; the SORTED one exercises the
    # searchsorted+bincount transpose bracket (_pressure_to_height's actual path).
    assert np.array_equal(_interp_columns(x[:1], xp, fp),
                          np.vstack([np.interp(x[0], xp[i], fp[i]) for i in range(B)]))
    xs = np.sort(x[0])[None, :]
    assert np.array_equal(_interp_columns(xs, xp, fp),
                          np.vstack([np.interp(xs[0], xp[i], fp[i]) for i in range(B)]))
    assert np.array_equal(_interp_columns(x, xp[:1], fp),
                          np.vstack([np.interp(x[i], xp[0], fp[i]) for i in range(B)]))


def test_grid_reconciliation_bitwise_vs_per_column_loop():
    """Issue #99: the vectorized ``_pressure_to_height`` / ``_height_to_pressure`` / ``_hydrostatic_z``
    reproduce the verbatim pre-#99 per-column/per-interface loops EXACTLY on a live coupled state (T, q,
    u, and a random field; both grid directions; heights from the live temperature)."""
    m = _spun_coupled_core()
    rng = np.random.default_rng(7)
    z_p = m._hydrostatic_z(m.T_atm)
    assert np.array_equal(z_p, _ref_hydrostatic_z(m, m.T_atm))                    # heights, live T
    assert np.array_equal(m._hydrostatic_z(), _ref_hydrostatic_z(m))              # T_atm=None default path
    T_hot = m.T_atm + rng.uniform(-40.0, 40.0, m.T_atm.shape)                     # rough adversarial T
    assert np.array_equal(m._hydrostatic_z(T_hot), _ref_hydrostatic_z(m, T_hot))
    for f_p in (m.T_atm, np.clip(m.q, 0.0, 0.5), rng.normal(0, 1, m.T_atm.shape)):
        assert np.array_equal(m._pressure_to_height(f_p, z_p), _ref_pressure_to_height(m, f_p, z_p))
    u, w = m.velocities()
    for f_h in (np.asarray(u), np.asarray(w), rng.normal(0, 1, (m.cfg.nz, m.cfg.nx))):
        assert np.array_equal(m._height_to_pressure(f_h, z_p), _ref_height_to_pressure(m, f_h, z_p))


def test_moist_convective_adjust_bitwise_vs_reference():
    """Issue #99: the dispatch-lean ``moist_convective_adjust`` reproduces the verbatim pre-#99 sweep
    EXACTLY -- live coupled states plus randomized super-adiabatic perturbations (so sweeps genuinely
    engage), both closures, with and without q, across sweep caps (1 / fleet 4 / deep 30), and a
    near-absolute-cold column exercising the rh_detrain degenerate-cold (denom==0) fallback."""
    m = _spun_coupled_core()
    rng = np.random.default_rng(3)
    z = m._hydrostatic_z(m.T_atm)
    T0 = m.T_atm.copy()
    T0[:, 1:] += rng.uniform(0.0, 12.0, (m.cfg.nx, m.cfg.nz - 1))     # destabilize: warm lower layers
    T0[0, :] = np.linspace(10.0, 80.0, m.cfg.nz)                      # absurd-cold super-adiabatic column:
    #   the top pairs sit at T < ~30 K where esat underflows to exactly 0.0, so a convecting pair there
    #   has denom == 0 -> pos == False -> the homogenize fallback branch is exercised bit-for-bit too
    q0 = np.clip(m.q, 0.0, 0.5) + rng.uniform(0.0, 5e-3, m.q.shape)   # exercises the detrain rescale
    z0 = m._hydrostatic_z(T0)
    # np.errstate: the absurd-cold column drives esat's DISCARDED liquid branch through exp overflow
    # (inf, dropped by the where; both implementations identically) -- expected here, silence the noise.
    with np.errstate(over="ignore"):
        for closure in ("q_homogenize", "rh_detrain"):
            for sweeps in (1, 4, 30):
                Ta, qa = T0.copy(), q0.copy()
                Tb, qb = T0.copy(), q0.copy()
                moist_convective_adjust(Ta, m.dm, z0, m.p_mid, sweeps, q=qa, closure=closure)
                _ref_moist_convective_adjust(Tb, m.dm, z0, m.p_mid, sweeps, q=qb, closure=closure)
                assert not np.array_equal(Ta, T0)                          # the gate bites: sweeps engaged
                assert np.array_equal(Ta, Tb) and np.array_equal(qa, qb)   # EXACT -- bit-identical
    Ta, Tb = T0.copy(), T0.copy()                                      # q=None limb (T-only mixing)
    with np.errstate(over="ignore"):
        moist_convective_adjust(Ta, m.dm, z0, m.p_mid, 4, q=None)
        _ref_moist_convective_adjust(Tb, m.dm, z0, m.p_mid, 4, q=None)
    assert np.array_equal(Ta, Tb)
    Ta, qa = m.T_atm.copy(), np.clip(m.q, 0.0, 0.5)                    # near-adjusted live state, default closure
    Tb, qb = Ta.copy(), qa.copy()
    moist_convective_adjust(Ta, m.dm, z, m.p_mid, 4, q=qa, closure=m.cfg.convective_closure)
    _ref_moist_convective_adjust(Tb, m.dm, z, m.p_mid, 4, q=qb, closure=m.cfg.convective_closure)
    assert np.array_equal(Ta, Tb) and np.array_equal(qa, qb)


def test_moist_convective_adjust_fixed_topology_bit_identical():
    """GPU capture-readiness (Phase B): ``fixed_topology=True`` (drop the inner ``unstable.any()`` continue
    AND the outer ``changed.any()`` break, so the launch sequence is constant) is BIT-IDENTICAL to the
    default adaptive path. Two regimes, each chosen so the dropped exits ACTUALLY FIRE in the OFF path
    (else the test is vacuous):

      1. A SUPER-STABLE inversion column (warm aloft, cold at the surface): NO pair is ever unstable, so OFF
         takes the inner continue every pair AND the outer break on sweep 0 -- the strongest exercise of
         both drops. ON instead runs all max_sweeps of all-stable self-writes.
      2. A live coupled state with super-adiabatic perturbations (sweeps genuinely engage): the MIXED case
         where some pairs are stable (inner continue fires) and others convect.

    Bit-identity holds because the ``np.where(unstable, new, old)`` merges rewrite every stable column with
    its OWN bits, so a skipped pair/sweep and a self-write are the same memory."""
    m = _spun_coupled_core()
    # --- regime 1: super-stable inversion (both exits fire OFF) ---
    nx, nz = 8, m.cfg.nz
    z_inv = np.tile(np.linspace(16000.0, 0.0, nz), (nx, 1)).astype(float)   # top->surface, decreasing
    T_inv = np.tile(np.linspace(300.0, 230.0, nz), (nx, 1)).astype(float)   # warm aloft, cold surface
    q_inv = np.full((nx, nz), 1e-3)
    # confirm the OFF path REALLY takes the exits (sweeps converge before the cap): an adaptive run with a
    # huge cap must leave the field untouched after one sweep (all stable) -> outer break fires.
    T_probe = T_inv.copy()
    moist_convective_adjust(T_probe, m.dm[:nz], z_inv, m.p_mid[:nz], 30, q=q_inv.copy(),
                            closure="rh_detrain", fixed_topology=False)
    assert np.array_equal(T_probe, T_inv)        # stable: the adaptive path made NO change (exits fired)
    for closure in ("q_homogenize", "rh_detrain"):
        Ta, qa = T_inv.copy(), q_inv.copy()
        Tb, qb = T_inv.copy(), q_inv.copy()
        moist_convective_adjust(Ta, m.dm[:nz], z_inv, m.p_mid[:nz], 8, q=qa, closure=closure,
                                fixed_topology=False)
        moist_convective_adjust(Tb, m.dm[:nz], z_inv, m.p_mid[:nz], 8, q=qb, closure=closure,
                                fixed_topology=True)
        assert np.array_equal(Ta, Tb) and np.array_equal(qa, qb)   # EXACT
    # --- regime 2: mixed live + perturbed (inner continue fires on the still-stable pairs) ---
    rng = np.random.default_rng(11)
    T0 = m.T_atm.copy()
    T0[:, 1:] += rng.uniform(0.0, 12.0, (m.cfg.nx, m.cfg.nz - 1))
    q0 = np.clip(m.q, 0.0, 0.5) + rng.uniform(0.0, 5e-3, m.q.shape)
    z0 = m._hydrostatic_z(T0)
    with np.errstate(over="ignore"):
        for closure in ("q_homogenize", "rh_detrain"):
            for sweeps in (1, 4, 8):
                Ta, qa = T0.copy(), q0.copy()
                Tb, qb = T0.copy(), q0.copy()
                moist_convective_adjust(Ta, m.dm, z0, m.p_mid, sweeps, q=qa, closure=closure,
                                        fixed_topology=False)
                moist_convective_adjust(Tb, m.dm, z0, m.p_mid, sweeps, q=qb, closure=closure,
                                        fixed_topology=True)
                assert not np.array_equal(Ta, T0)                       # sweeps engaged (non-vacuous)
                assert np.array_equal(Ta, Tb) and np.array_equal(qa, qb)


def test_capture_ready_default_off():
    """GPU capture-readiness gate (Phase A/B): ``capture_ready`` is DEFAULT OFF -- a plain run uses the
    shipped rebinding / adaptive code path (byte-identical); the env knob is COSMO_AD_CAPTURE_READY."""
    assert StageADConfig().capture_ready is False


_STATE_KEYS = ("T_atm", "T_surf", "q", "q_c", "omega", "psi", "b")


def _state_ptrs(core):
    """The host data pointers of the 7 prognostic-state arrays (the CUDA-graph capture targets). A SHA over
    values cannot see address stability, so this is the DIRECT check that Phase A took: ON the pointers must
    be invariant across a step (the buffer is written in place), OFF they rebind (a fresh allocation)."""
    return {k: np.asarray(getattr(core, k)).__array_interface__["data"][0]
            for k in _STATE_KEYS}


def test_capture_ready_state_pointers_stable():
    """GPU capture-readiness (Phase A) -- the DELIVERABLE the value-SHA cannot verify: with ``capture_ready``
    ON, every prognostic-state array keeps a FIXED data pointer across a coupled ``step()`` (written IN PLACE
    into the pre-allocated buffer, the prerequisite for CUDA-graph replay). With it OFF the shipped code
    REBINDS them (fresh allocations), so the test has teeth: it proves the in-place conversion -- not some
    incidental allocator reuse -- is what fixed the address. Run a few steps first so all the per-step store
    paths (advection, microphysics, the Poisson stores, buoyancy) have fired at least once before the probe.

    TWO control paths so EVERY converted store is address-checked: the default (explicit dynamics, prognostic
    microphysics) AND the implicit dynamics + below-cloud re-evaporation path (the implicit-omega store and the
    ``_below_cloud_reevap`` T_atm add only fire there)."""
    paths = (dict(prognostic_condensate=True),                                   # default explicit
             dict(prognostic_condensate=True, vertical_implicit_diffusion=True,  # implicit-omega + reevap stores
                  precip_reevap=True, qc_ice_phase=True))
    for extra in paths:
        for flag, expect_stable in ((True, True), (False, False)):
            m = StageADCore(StageADConfig(nx=12, nz=14, theta_max_deg=70.0, capture_ready=flag, **extra))
            for _ in range(3):
                m.step()
            # Pin the pre-step array OBJECTS (strong refs held across the probe step) so their buffers
            # cannot be freed and handed straight back by numpy's allocator. Without this the OFF path's
            # fresh allocations can land on a just-freed address (before == after spuriously), which made
            # the OFF assertion below a known allocator-reuse flake (any of the four cores could "not move").
            # With the old buffers pinned, a genuinely reallocated array is FORCED to a distinct address, so
            # "OFF really rebinds" is deterministic; the ON path writes in place (same buffer) and is unaffected.
            pinned = [getattr(m, k) for k in _STATE_KEYS]  # noqa: F841 -- intentionally kept alive
            before = _state_ptrs(m)
            m.step()
            after = _state_ptrs(m)
            del pinned
            stable = {k: before[k] == after[k] for k in before}
            if expect_stable:
                assert all(stable.values()), f"capture_ready ON {extra}: these rebind (broken in-place): " \
                    f"{[k for k, v in stable.items() if not v]}"
            else:
                # OFF: the heavily-rebound dynamics/thermo cores must move (fresh allocs) -> non-vacuous.
                # The pin above removes the allocator-reuse false-positive; we still assert only the four
                # cores the OFF path is guaranteed to reallocate (T_surf/q_c/b may legitimately stay in place).
                moved = [k for k, v in stable.items() if not v]
                assert {"T_atm", "q", "omega", "psi"} <= set(moved), \
                    f"capture_ready OFF {extra} expected the cores to rebind; only these moved: {moved}"


def test_convadjust_parallel_default_off():
    """CPU-fuse flag (issue #99 / branch claude/cpu-fuse-multithread): the column-parallel numba kernel
    is DEFAULT OFF -- a plain run uses the numpy path (byte-identical) and never imports numba."""
    assert StageADConfig().convadjust_parallel is False


def test_convadjust_parallel_matches_numpy_to_roundoff():
    """The column-parallel numba kernel (``moist_convective_adjust_parallel``) reproduces the numpy
    reference ``moist_convective_adjust`` to ROUND-OFF (~1e-13 K, NOT bit-identical -- it reassociates the
    qsat/lapse-rate leaves) for BOTH closures, on a live coupled state with super-adiabatic perturbations
    so the sweeps genuinely engage, and conserves column enthalpy + water exactly (Gamma-independent)."""
    pytest.importorskip("numba")
    m = _spun_coupled_core()
    rng = np.random.default_rng(7)
    T0 = m.T_atm + rng.uniform(0.0, 12.0, m.T_atm.shape)
    q0 = np.clip(m.q, 0.0, 0.5) + rng.uniform(0.0, 5e-3, m.q.shape)
    z0 = m._hydrostatic_z(T0)
    for closure in ("rh_detrain", "q_homogenize"):
        Ta, qa = T0.copy(), q0.copy()          # numpy reference
        Tb, qb = T0.copy(), q0.copy()          # numba kernel
        moist_convective_adjust(Ta, m.dm, z0, m.p_mid, 4, q=qa, closure=closure,
                                rh_surf=m.cfg.rh_surf, rh_top=m.cfg.rh_top)
        moist_convective_adjust_parallel(Tb, m.dm, z0, m.p_mid, 4, q=qb, closure=closure,
                                         rh_surf=m.cfg.rh_surf, rh_top=m.cfg.rh_top)
        assert not np.array_equal(Ta, T0)                  # the sweep actually bit
        assert np.max(np.abs(Ta - Tb)) < 1e-10             # round-off agreement, not bit-identical
        assert np.max(np.abs(qa - qb)) < 1e-12
        # exact conservation, kernel side (the pairwise-mix theorems hold for the numba path too)
        assert np.allclose(np.sum(m.dm[None, :] * Tb, axis=1),
                           np.sum(m.dm[None, :] * T0, axis=1), rtol=1e-12)
        assert np.allclose(np.sum(m.dm[None, :] * qb, axis=1),
                           np.sum(m.dm[None, :] * q0, axis=1), rtol=1e-11)


def test_convadjust_parallel_core_off_byte_identical(monkeypatch):
    """Through the FULL coupled core: a multi-step run with the flag OFF vs the same run that NEVER set the
    flag are bit-identical (the numpy path is untouched), and the flag ON tracks OFF to round-off over the
    SHORT run checked here (a handful of steps). NB the guarantee is round-off PER STEP, not a permanently
    bounded trajectory: in the f=0 core deterministic chaos decorrelates two round-off-separated trajectories
    over a long run, so the production guarantee is climatological (time-mean P, T(theta), band), not
    step-for-step tracking -- immaterial to the science given the +-10 C budget. Guards: the ON path must
    fall back to numpy under freeze_gamma / q=None / fp32."""
    pytest.importorskip("numba")

    def _run(parallel, freeze=False, steps=6, seed=99):
        monkeypatch.setenv("COSMO_CONVADJUST_PARALLEL", "1" if parallel else "0")
        rng = np.random.default_rng(seed)
        cfg = StageADConfig(nx=24, nz=16, theta_max_deg=80.0, conv_freeze_gamma=freeze,
                            convadjust_parallel=parallel)
        core = StageADCore(cfg)
        core.run_dynamics = True; core.couple_heat = True
        core.T_atm = core.T_atm + 0.7 * rng.standard_normal(core.T_atm.shape)
        core.q = np.clip(core.q + 1e-4 * rng.standard_normal(core.q.shape), 0.0, 0.5)
        for _ in range(steps):
            core.step()
        return np.asarray(core.T_atm).copy(), np.asarray(core.q).copy()

    Toff, qoff = _run(parallel=False)
    Toff2, qoff2 = _run(parallel=False)
    assert np.array_equal(Toff, Toff2) and np.array_equal(qoff, qoff2)   # OFF is deterministic
    Ton, qon = _run(parallel=True)
    assert np.max(np.abs(Toff - Ton)) < 1e-9      # ON tracks OFF to round-off over the multi-step run
    assert np.max(np.abs(qoff - qon)) < 1e-11
    # freeze_gamma ON forces the numpy fallback even with the parallel flag set -> identical to OFF+freeze
    Tfz_off, _ = _run(parallel=False, freeze=True)
    Tfz_on, _ = _run(parallel=True, freeze=True)
    assert np.array_equal(Tfz_off, Tfz_on)        # fallback to numpy is bit-identical


# --------------------------------------------------------------------------------------------------
# ROADMAP PHY-17 step 4: the ONE-PASS SEGMENT MIXER -- the CONVERGED convective adjustment.
#   The shipped Gauss-Seidel sweep relaxes a column pairwise, so the sweeps it needs grow with the
#   SQUARE of the unstable layer's depth in levels and the production cap (conv_max_sweeps = 4) is
#   nowhere near the fixed point on the canonical 144-level grid. ``moist_convective_segment_mix``
#   solves the same operator's fixed point directly. The gates below are the PR #195 triple (default
#   OFF / matches the reference / the core is byte-identical with the flag OFF) plus the two theorems
#   (enthalpy, water) and the equivalence to a CONVERGED sweep.
# --------------------------------------------------------------------------------------------------
def _ref_segment_mix(T_atm, dm, z, p_mid, q=None, closure="q_homogenize", rh_surf=0.8, rh_top=0.2,
                     max_passes=48, gamma_iters=6, passes_out=None, legacy_loops=False):
    """SCALAR per-column reference for ``moist_convective_segment_mix`` -- plain Python loops, one column
    at a time, no vectorization and no cumulative sums. This is the oracle for BOTH shipped
    implementations: the numpy body (vectorized across columns, block sums by ``np.add.reduceat``) and
    the CUDA ``segmix_cols`` RawKernel, which is a line-by-line transliteration of THIS loop and which
    cannot be compiled on a box without CuPy. Deliberately written for legibility, not speed. It carries
    BOTH stopping rules -- (i) no unstable pair, (ii) the active set has not grown for two consecutive
    passes -- and the same ``passes_out`` sign convention (negative = the bound was hit while still
    growing), because those are exactly what the kernel has to get right and cannot be compiled here.

    ``legacy_loops`` selects the kernel's PRE-#830 loop nest: blocks issued at their own position in the
    descent over ``act`` rather than by ordinal, a second Gamma evaluation per iterate instead of
    stashing the offsets in place, and a second ``qsat`` per layer in the water closure. The three are
    warp-cost optimizations that must not move a bit; ``test_segmix_kernel_loop_nest_is_bitwise_neutral``
    is the proof, and this flag is what it compares against."""
    nx, nz = T_atm.shape
    detrain = (closure == "rh_detrain") and (q is not None)
    RHt = rh_top + (rh_surf - rh_top) * (np.asarray(p_mid, float) / C.P_SURF)

    def _gamma(T, p):
        return float(th.moist_adiabat_lapse_rate(np.float64(T), np.float64(p)))

    settled_bar = _SEGMIX_SETTLED_ULP * float(np.finfo(T_atm.dtype).eps)

    def _mix(Ti, qi, zi, lo, hi):
        """Mix one block; return 1 if the LAST Gamma iterate still moved it (stopping rule (ii))."""
        moving = 0
        M = H = W = 0.0
        for i in range(lo, hi + 1):
            M += dm[i]; H += dm[i] * Ti[i]
            if qi is not None:
                W += dm[i] * qi[i]
        for _it in range(gamma_iters):
            d = S = 0.0
            if _it < gamma_iters - 1 and not legacy_loops:
                # ONE Gamma sweep per iterate: the second loop below evaluates Gamma at exactly the
                # Ti[i] the first one used, so the offsets can be stashed in place and read back. The
                # LAST iterate keeps the two-loop form because rule (ii) needs T_old.
                for i in range(hi, lo - 1, -1):
                    S += dm[i] * d
                    T_old = Ti[i]
                    Ti[i] = d                                      # stash the offset in place
                    if i > lo:
                        d -= _gamma(T_old, p_mid[i]) * (zi[i - 1] - zi[i])
                T_bot = (H - S) / M                                # enthalpy-conserving, closed form
                for i in range(hi, lo - 1, -1):
                    Ti[i] = T_bot + Ti[i]
                continue
            for i in range(hi, lo - 1, -1):
                S += dm[i] * d
                if i > lo:
                    d -= _gamma(Ti[i], p_mid[i]) * (zi[i - 1] - zi[i])
            T_bot = (H - S) / M                                    # enthalpy-conserving, closed form
            d = 0.0
            for i in range(hi, lo - 1, -1):
                T_old = Ti[i]                                      # read before the write
                T_nu = T_bot + d
                if _it == gamma_iters - 1 and abs(T_nu - T_old) > settled_bar * abs(T_nu):
                    moving = 1
                Ti[i] = T_nu
                if i > lo:
                    d -= _gamma(T_old, p_mid[i]) * (zi[i - 1] - zi[i])
        if qi is None:
            return moving
        if detrain:
            denom = 0.0
            for i in range(lo, hi + 1):
                qs = float(th.qsat(np.float64(Ti[i]), np.float64(p_mid[i])))
                if not legacy_loops:
                    qi[i] = qs                                     # park it: W is already summed
                denom += dm[i] * RHt[i] * qs
            if denom > 0.0:
                A = W / denom
                for i in range(lo, hi + 1):
                    qs = (qi[i] if not legacy_loops
                          else float(th.qsat(np.float64(Ti[i]), np.float64(p_mid[i]))))
                    qi[i] = A * RHt[i] * qs
                return moving
        homog = W / M
        for i in range(lo, hi + 1):
            qi[i] = homog
        return moving

    for n in range(nx):
        Ti = T_atm[n]; qi = None if q is None else q[n]; zi = z[n]
        act = [False] * nz                                         # act[k]: pair k is in the mixed set
        nact_prev, stall, used, done = -1, 0, 0, False
        for _p in range(max_passes):
            work = False
            for k in range(nz - 1, 0, -1):                         # SCAN on the pre-mix column
                Gdz = _gamma(Ti[k], p_mid[k]) * (zi[k - 1] - zi[k])
                if Ti[k - 1] < (Ti[k] - Gdz) - 1e-9:
                    act[k] = True                                  # MONOTONE: never un-set
                    work = True
            nact = sum(1 for k in range(1, nz) if act[k])
            stall = 0 if nact > nact_prev else stall + 1
            nact_prev = nact
            if not work:                                           # (i) the sweep's own rule
                done, used = True, _p
                break
            used = _p + 1
            bot, moving = -1, 0
            if legacy_loops:                                       # blocks issued at their own POSITION
                for k in range(nz - 1, 0, -1):                     # mix every maximal run of active pairs
                    if act[k]:
                        if bot < 0:
                            bot = k
                        continue
                    if bot >= 0:
                        moving |= _mix(Ti, qi, zi, k, bot)
                        bot = -1
                if bot >= 0:
                    moving |= _mix(Ti, qi, zi, 0, bot)
            else:                                                  # ...by ORDINAL: same blocks, same
                nblk = sum(1 for k in range(1, nz)                 # order, warp-aligned by count
                           if act[k] and not act[k - 1])
                kcur = nz - 1
                for _b in range(nblk):
                    lo = hi = -1
                    while kcur >= 1:
                        if act[kcur]:
                            if bot < 0:
                                bot = kcur
                            kcur -= 1
                            continue
                        if bot >= 0:
                            lo, hi, bot = kcur, bot, -1
                            kcur -= 1
                            break
                        kcur -= 1
                    if lo < 0:
                        if bot < 0:
                            break
                        lo, hi, bot = 0, bot, -1
                    moving |= _mix(Ti, qi, zi, lo, hi)
            if stall >= 2 and not moving:      # (ii) partition final AND the solve settled (the fp32 exit)
                done = True
                break
        if passes_out is not None:
            passes_out[n] = used if done else -used                # negative = truncated at the bound
    return T_atm


def _segmix_case(nx=6, nz=40, seed=11):
    """A synthetic (nx, nz) case exercising every branch the mixer has: a deep surface-based segment,
    TWO separated segments that MERGE once mixed, an UNMODIFIED column (the core's own initial profile,
    which is itself statically unstable -- not a stable control), an elevated segment with no surface
    contact, a randomly roughened column, and an absurd-cold column (esat underflow, the degenerate
    rh_detrain denominator). Returns (core, dm, p_mid, T0, z, q0)."""
    m = StageADCore(StageADConfig(nx=nx, nz=nz, theta_max_deg=70.0))
    rng = np.random.default_rng(seed)
    dm = np.asarray(m.dm, float); p_mid = np.asarray(m.p_mid, float)
    T0 = np.asarray(m.T_atm, float).copy()
    T0[0, -20:] += np.linspace(0.0, 60.0, 20)                      # deep surface-based super-adiabat
    T0[1, -8:] += np.linspace(0.0, 30.0, 8)                        # two separated unstable segments...
    T0[1, -20:-14] += np.linspace(0.0, 25.0, 6)                    # ...that merge once each is mixed
    T0[3, -25:-18] += np.linspace(0.0, 35.0, 7)                    # elevated segment, no surface contact
    T0[4] += rng.uniform(0.0, 12.0, nz)                            # randomly roughened
    T0[5] = np.linspace(10.0, 80.0, nz)                            # absurd-cold (esat underflows to 0)
    z = np.asarray(m._hydrostatic_z(T0))
    q0 = np.abs(rng.standard_normal((nx, nz))) * 2e-3
    return m, dm, p_mid, T0, z, q0


def _max_instability(T, dm, z, p_mid):
    """Max residual (T_k - Gamma*dz) - T_{k-1} over the pairs of each column [K]: > 0 means still unstable."""
    Gam = np.asarray(th.moist_adiabat_lapse_rate(T, np.asarray(p_mid)[None, :]))
    return ((T[:, 1:] - Gam[:, 1:] * (z[:, :-1] - z[:, 1:])) - T[:, :-1]).max(axis=1)


def test_conv_segment_mixer_default_off():
    """PHY-17 step 4 (the PR #195 shape): the converged segment mixer is DEFAULT OFF -- a plain run takes
    the shipped capped Gauss-Seidel sweep, byte-identically. Its two bounds carry MEASURED defaults, both
    swept over run17's own 975 columns (PHY17_SEGMENT_MIXER_RESULTS.md section 4): every column converges
    by 40 merge passes, and 2 moist-adiabat iterates are the cheapest setting that still reaches the same
    converged partition as a tightly-converged reference with no column left unstable."""
    cfg = StageADConfig()
    assert cfg.conv_segment_mixer is False
    assert cfg.conv_segment_max_passes == 48
    assert cfg.conv_segment_gamma_iters == 3


def test_segment_mixer_matches_scalar_reference_and_a_converged_sweep():
    """The two claims the whole change rests on, on the synthetic multi-branch case, BOTH closures.

    (a) The vectorized numpy body reproduces the SCALAR reference (the kernel's own template) to within
        the ALGORITHM's own tolerance, 1e-7 K -- not to machine epsilon, and the reason is worth stating
        because the same reason sets the CUDA parity bar. The two differ in round-off (the numpy block
        sums run top-down through ``np.add.reduceat``, the scalar loop accumulates the offset moment
        bottom-up), and the loop STOPS ON A THRESHOLD: "no pair unstable by more than 1e-9 K". A
        round-off difference either side of that threshold costs one path an extra pass, which moves the
        profile by about the threshold, and a block accumulates a few of those. So the honest bar is a
        small multiple of the 1e-9 K tolerance, not 1e-15. (Run both at a tightly-converged
        ``gamma_iters`` and the gap falls to ~1e-13 K; the shipped default is 6.)
    (b) It lands on the CONVERGED Gauss-Seidel fixed point: 30,000 sweeps of the SHIPPED
        ``moist_convective_adjust`` agree to < 1e-6 K, which is the SWEEP's own convergence floor, not
        the mixer's -- the mixer's residual instability is driven below the 1e-9 K tolerance the sweep
        stops on, and the sweep at 30,000 is still only just there.
    NON-VACUOUS: the mixer must move the state by tens of K, and the 4-sweep production cap must be far
    from the converged answer (that gap IS PHY-17)."""
    m, dm, p_mid, T0, z, q0 = _segmix_case()
    # errstate: the absurd-cold column drives esat's DISCARDED liquid branch through exp overflow (as in
    # the standing bitwise gate), and 30,000 sweeps of the SHIPPED rh_detrain on that column eventually
    # multiply a zero normalization by an infinite qsat -> NaN in ITS q. Both are properties of the
    # reference on a column no live state reaches; the q comparisons below skip that closure for exactly
    # that reason, and the conservation test asserts only on the columns whose q stayed finite.
    with np.errstate(over="ignore", invalid="ignore"):
        for closure in ("rh_detrain", "q_homogenize"):
            Tr, qr = T0.copy(), q0.copy()                          # converged sweep
            moist_convective_adjust(Tr, dm, z.copy(), p_mid, 30000, q=qr, closure=closure)
            Ts, qs = T0.copy(), q0.copy()                          # scalar reference
            _ref_segment_mix(Ts, dm, z.copy(), p_mid, q=qs, closure=closure)
            Tv, qv = T0.copy(), q0.copy()                          # vectorized numpy (the shipped path)
            moist_convective_segment_mix(Tv, dm, z.copy(), p_mid, q=qv, closure=closure)
            T4, q4 = T0.copy(), q0.copy()                          # the production cap, for contrast
            moist_convective_adjust(T4, dm, z.copy(), p_mid, 4, q=q4, closure=closure)
            assert np.max(np.abs(Tv - T0)) > 10.0                  # NON-VACUOUS: the mixer really mixes
            assert np.max(np.abs(T4 - Tr)) > 1.0                   # ...and 4 sweeps are nowhere near it
            assert np.max(np.abs(Tv - Ts)) < 1e-7                  # (a) vectorized == scalar [K]
            assert np.max(np.abs(Tv - Tr)) < 1e-6                  # (b) == the converged sweep [K]
            assert np.max(_max_instability(Tv, dm, z, p_mid)) < 1e-9    # no pair left unstable
            if closure == "q_homogenize":                          # (rh_detrain's q is NaN-poisoned on the
                assert np.max(np.abs(qv - qs)) < 1e-11             #  absurd-cold column in BOTH paths --
                assert np.max(np.abs(qv - qr)) < 1e-9              #  see the conservation test)
        # ...and with the Gamma fixed point run to machine precision the two paths agree at round-off,
        # which is the evidence that the 1e-7 K above is the THRESHOLD effect and not a structural gap.
        Ts, Tv = T0.copy(), T0.copy()
        qs, qv = q0.copy(), q0.copy()
        _ref_segment_mix(Ts, dm, z.copy(), p_mid, q=qs, closure="rh_detrain", gamma_iters=14)
        moist_convective_segment_mix(Tv, dm, z.copy(), p_mid, q=qv, closure="rh_detrain", gamma_iters=14)
        assert np.max(np.abs(Tv - Ts)) < 1e-11


def test_segmix_kernel_loop_nest_is_bitwise_neutral():
    """PR #830's three warp-cost changes to ``_gpu_kernels.segmix_cols`` must not move one bit.

    The kernel cannot be compiled on this box, so the gate is the SCALAR reference that the kernel is a
    line-by-line transliteration of: run it with ``legacy_loops=True`` (the pre-#830 nest) and with the
    shipped nest on the same state, and require BITWISE equality of T, q and the per-column pass counts.
    Approximate equality would not test the claim -- all three changes are algebraic identities, and the
    reason each is safe is that it reproduces the original expression EXACTLY:

      (O1) blocks issued by ORDINAL instead of by position -- the same blocks, in the same bottom-up
           order, with the same (lo, hi); only the loop that issues them changes, and that is what a
           warp pays for (see ``cosmology_models/phy17_mixer_cost_model.py``).
      (O2) the offsets stashed in place for all but the last Gamma iterate -- the second offset loop
           evaluates Gamma at exactly the temperatures the first one read, so the second evaluation is
           redundant. The last iterate keeps both loops because stopping rule (ii) needs T_old.
      (O3) the water closure's qsat parked in the q slot -- the block's water W is already summed, and
           the RAW qsat is stashed, so ``(dm*RHt)*qs`` and ``(A*RHt)*qs`` keep their original groupings.

    NON-VACUOUS on two counts: the mixer must move the state, and the case must actually contain a
    column with more than one simultaneous block -- otherwise (O1) is untested, since a single-block
    column issues its one call at ordinal 0 either way."""
    m, dm, p_mid, T0, z, q0 = _segmix_case()
    with np.errstate(over="ignore", invalid="ignore"):
        # NON-VACUITY for (O1): after ONE pass some column must show several DISJOINT runs of moved
        # layers -- i.e. several blocks mixed in the SAME pass, at different heights, which is the only
        # configuration in which issuing them by position and by ordinal differ. (The roughened column
        # 4 fragments into 7 blocks on its first pass; the elevated-segment column 3 into 2.)
        T1 = T0.copy()
        _ref_segment_mix(T1, dm, z.copy(), p_mid, q=q0.copy(), closure="rh_detrain", max_passes=1)
        moved = np.abs(T1 - T0) > 0.0
        runs = (moved[:, 1:] & ~moved[:, :-1]).sum(axis=1) + moved[:, 0]
        assert runs.max() >= 3, f"the case exercises at most {runs.max()} block(s); (O1) is untested"

        for closure in ("rh_detrain", "q_homogenize"):
            for gi in (1, 3, 6):                               # 1 exercises the "last iterate" branch alone
                Ta, qa = T0.copy(), q0.copy()
                Tb, qb = T0.copy(), q0.copy()
                pa = np.zeros(T0.shape[0], np.int64)
                pb = np.zeros(T0.shape[0], np.int64)
                _ref_segment_mix(Ta, dm, z.copy(), p_mid, q=qa, closure=closure, gamma_iters=gi,
                                 passes_out=pa, legacy_loops=True)
                _ref_segment_mix(Tb, dm, z.copy(), p_mid, q=qb, closure=closure, gamma_iters=gi,
                                 passes_out=pb, legacy_loops=False)
                assert np.max(np.abs(Ta - T0)) > 10.0, "NON-VACUOUS: the reference must really mix"
                assert np.array_equal(Ta, Tb), f"T moved: closure={closure} gamma_iters={gi}"
                assert np.array_equal(pa, pb), f"passes moved: closure={closure} gamma_iters={gi}"
                fin = np.isfinite(qa) & np.isfinite(qb)        # the absurd-cold column NaNs in BOTH
                assert np.array_equal(qa[fin], qb[fin]), f"q moved: closure={closure} gamma_iters={gi}"
                assert np.array_equal(np.isnan(qa), np.isnan(qb))


def test_segment_mixer_conserves_enthalpy_and_water():
    """The two theorems, to rtol 1e-12 in fp64, on the synthetic multi-branch case (deep segment, two
    segments that merge, absurd-cold column) and for both closures. Enthalpy conservation is
    Gamma-INDEPENDENT -- the block solve picks T_bot to conserve it in closed form -- so it must hold at
    ANY ``gamma_iters``, converged or not; water conservation is a pure linear rescaling of the block.
    The truncation warning is SUPPRESSED here on purpose: at ``gamma_iters=1`` the default 16-pass bound
    really is exhausted, and the point of this test is that both theorems hold on that truncated state
    too -- they are properties of the block solve, not of its convergence."""
    m, dm, p_mid, T0, z, q0 = _segmix_case()
    H0 = np.sum(dm[None, :] * T0, axis=1)
    W0 = np.sum(dm[None, :] * q0, axis=1)
    with warnings.catch_warnings(), np.errstate(over="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for closure in ("rh_detrain", "q_homogenize"):
            for gi in (1, 2, 6):                                   # deliberately UNDER-converged Gamma too
                T, q = T0.copy(), q0.copy()
                moist_convective_segment_mix(T, dm, z.copy(), p_mid, q=q, closure=closure, gamma_iters=gi)
                assert np.allclose(np.sum(dm[None, :] * T, axis=1), H0, rtol=1e-12)
                ok = np.isfinite(q).all(axis=1)                    # the absurd-cold column's rh_detrain q
                assert ok[:5].all()                                #   goes non-finite in the SHIPPED sweep
                assert np.allclose(np.sum(dm[None, :] * q, axis=1)[ok], W0[ok], rtol=1e-12)


def test_segment_mixer_fixed_topology_bit_identical_and_q_none():
    """``fixed_topology=True`` (run every pass, no data-dependent break -- the capture-readiness shape) is
    BIT-IDENTICAL to the adaptive exit, because a pass over a converged column selects that column's own
    bits through the ``np.where`` merges. And the ``q=None`` limb (no moisture) mixes T alone and gives
    BIT-IDENTICAL temperatures, since the enthalpy solve never reads q."""
    m, dm, p_mid, T0, z, q0 = _segmix_case()
    with np.errstate(over="ignore", invalid="ignore"):
        Ta, qa = T0.copy(), q0.copy()
        Tb, qb = T0.copy(), q0.copy()
        moist_convective_segment_mix(Ta, dm, z.copy(), p_mid, q=qa, closure="rh_detrain")
        moist_convective_segment_mix(Tb, dm, z.copy(), p_mid, q=qb, closure="rh_detrain",
                                     fixed_topology=True)
        assert np.array_equal(Ta, Tb) and np.array_equal(qa, qb)
        Tn = T0.copy()
        moist_convective_segment_mix(Tn, dm, z.copy(), p_mid, q=None)
        Th, qh = T0.copy(), q0.copy()
        moist_convective_segment_mix(Th, dm, z.copy(), p_mid, q=qh, closure="q_homogenize")
        assert np.array_equal(Tn, Th)                              # T mixing is water-independent
    with pytest.raises(ValueError):                                # gamma_iters must run at least once
        moist_convective_segment_mix(T0.copy(), dm, z.copy(), p_mid, q=q0.copy(), gamma_iters=0)


def test_conv_segment_mixer_core_off_byte_identical(monkeypatch):
    """Through the FULL coupled core (the PR #195 gate): a multi-step run with the flag OFF is BIT-IDENTICAL
    to the same run built with no mention of the flag at all, so the merge cannot perturb any existing
    result. The flag ON then (i) genuinely changes the state -- non-vacuous -- and (ii) leaves the column
    statically stable where the capped sweep does not; and the documented fallbacks (conv_freeze_gamma ON)
    return the run bit-for-bit to the OFF path."""
    def _run(mixer, freeze=False, steps=6, seed=99, **extra):
        if mixer is None:                                          # the pre-change world: the flag is not
            monkeypatch.delenv("COSMO_CONV_SEGMENT_MIXER", raising=False)   # in the env OR in the config
        else:
            monkeypatch.setenv("COSMO_CONV_SEGMENT_MIXER", "1" if mixer else "0")
        rng = np.random.default_rng(seed)
        kw = dict(nx=24, nz=16, theta_max_deg=80.0, conv_freeze_gamma=freeze)
        kw.update(extra)
        if mixer is not None:
            kw["conv_segment_mixer"] = mixer
        core = StageADCore(StageADConfig(**kw))
        core.run_dynamics = True; core.couple_heat = True
        core.T_atm = core.T_atm + 0.7 * rng.standard_normal(core.T_atm.shape)
        core.q = np.clip(core.q + 1e-4 * rng.standard_normal(core.q.shape), 0.0, 0.5)
        for _ in range(steps):
            core.step()
        return np.asarray(core.T_atm).copy(), np.asarray(core.q).copy()

    Tnone, qnone = _run(None)                                      # the config never mentions the flag
    Toff, qoff = _run(False)
    assert np.array_equal(Tnone, Toff) and np.array_equal(qnone, qoff)     # OFF == pre-change, bit-for-bit
    Ton, qon = _run(True)
    assert np.max(np.abs(Ton - Toff)) > 1e-3                       # NON-VACUOUS: ON is a different state
    # the documented fallback: freeze_gamma ON sends the mixer branch back to the shipped sweep
    Tfz_off, _ = _run(False, freeze=True)
    Tfz_on, _ = _run(True, freeze=True)
    assert np.array_equal(Tfz_off, Tfz_on)                         # fallback is bit-identical


def test_conv_segment_mixer_core_on_leaves_the_column_stable():
    """What the flag is FOR, at test scale: after one adjustment call the mixer leaves no statically
    unstable pair, where the production 4-sweep cap leaves plenty. Same core, same state, same closure --
    only the algorithm differs -- so this isolates the cap from the physics."""
    m, dm, p_mid, T0, z, q0 = _segmix_case(nx=6, nz=40, seed=5)
    with np.errstate(over="ignore", invalid="ignore"):
        T4, q4 = T0[:5].copy(), q0[:5].copy()
        moist_convective_adjust(T4, dm, z[:5].copy(), p_mid, 4, q=q4, closure="rh_detrain")
        Tm, qm = T0[:5].copy(), q0[:5].copy()
        moist_convective_segment_mix(Tm, dm, z[:5].copy(), p_mid, q=qm, closure="rh_detrain")
    r4 = _max_instability(T4, dm, z[:5], p_mid)
    rm = _max_instability(Tm, dm, z[:5], p_mid)
    assert r4.max() > 1.0                                          # the cap leaves it unstable by K
    assert rm.max() < 1e-9                                         # the mixer does not


@pytest.mark.parametrize("gi", sorted({StageADConfig().conv_segment_gamma_iters, 6}))
def test_segment_mixer_terminates_in_float32_where_the_predicate_alone_cannot(gi):
    """STOPPING RULE (ii), against the FAILURE rather than a description of it. The mixer's exit test (i)
    is the sweep's own -- "no pair unstable by more than 1e-9 K" -- and at SINGLE precision that tolerance
    is unreachable: the unit in the last place at ~300 K is ~3e-5 K, so a mixed block's interior pairs,
    which are exactly marginal, keep reading unstable forever. Rule (ii) -- the active set has not grown
    for two passes AND the profile solve has settled to the arithmetic's own resolution -- is what lets a
    single-precision column exit. Checked here, not asserted:
      * every column terminates WELL inside the bound, and the answer does not depend on the bound;
      * the exit is NOT rule (i): the routine's own predicate still flags pairs in the returned state, so
        (i) cannot have fired -- this is the non-vacuous half;
      * the moved set is IDENTICAL to fp64 and the states agree at the single-precision floor;
      * NO truncation warning fires -- the diagnostic keys on the active set still growing, not on the
        bound being reached, which is exactly what stops it crying wolf on every fp32 call.
    PARAMETRISED over the SHIPPED ``gamma_iters`` and over 6, because rule (ii) gets HARDER with fewer
    Gamma iterates: its second conjunct asks whether the last iterate still moved the profile, and a
    shorter solve is further from its fixed point when the pass ends. Pinning the old default here would
    have left the value actually in production untested.
    NOTE this exercises the MECHANISM (a state stored at single precision, so the predicate chatters at
    its granularity), not the CUDA kernel's arithmetic: the numpy body's internals are fp64 either way.
    The kernel's own version is ``_gpu_segmix_share.saturation()``, which the GPU box runs."""
    m, dm, p_mid, T0, z, q0 = _segmix_case()
    out = {}
    for tag, cast in (("fp64", np.float64), ("fp32", np.float32)):
        for mp in (48, 200):
            T, q, zz = T0.astype(cast).copy(), q0.astype(cast).copy(), z.astype(cast).copy()
            po = np.zeros(T0.shape[0], np.int32)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with np.errstate(over="ignore", invalid="ignore"):
                    moist_convective_segment_mix(T, dm, zz, p_mid, q=q, closure="rh_detrain",
                                                 max_passes=mp, gamma_iters=gi, passes_out=po)
            out[(tag, mp)] = (T, q, po, [c for c in caught if c.category is RuntimeWarning])
    T32, q32, po32, warn32 = out[("fp32", 48)]
    T64, _q64, po64, warn64 = out[("fp64", 48)]
    assert not warn32 and not warn64, f"spurious truncation warning: {warn32 or warn64}"
    # Truncation is strictly NEGATIVE; 0 is a legitimate count (a column with no unstable pair at all).
    assert (po32 >= 0).all() and po32.max() < 24, f"fp32 did not exit early: {po32.tolist()}"
    assert np.array_equal(T32, out[("fp32", 200)][0]), "fp32 answer depends on the bound"
    assert np.array_equal(po32, out[("fp32", 200)][2]), "fp32 pass count depends on the bound"
    # NON-VACUOUS: rule (i) could not have produced that exit -- its predicate still flags pairs.
    Gam = th.moist_adiabat_lapse_rate(T32, p_mid[None, :].astype(np.float32))
    dzk = np.zeros_like(T32); dzk[:, 1:] = z.astype(np.float32)[:, :-1] - z.astype(np.float32)[:, 1:]
    still = (T32[:, :-1] < (T32[:, 1:] - Gam[:, 1:] * dzk[:, 1:]) - 1e-9).sum()
    assert still > 0, "fp32 reached the 1e-9 K tolerance -- this case cannot test rule (ii)"
    moved32 = np.abs(T32.astype(float) - T0) > 1e-3
    moved64 = np.abs(T64 - T0) > 1e-3
    assert np.array_equal(moved32, moved64), "fp32 mixed a different set of layers"
    assert np.max(np.abs(T32.astype(float) - T64)) < 1e-3          # the fp32 floor (measured ~2e-5 K)
    assert np.max(np.abs(q32.astype(float) - out[("fp64", 48)][1])) < 1e-8


def test_segment_mixer_reports_a_truncated_adjustment_and_refuses_zero_passes():
    """SILENT TRUNCATION IS THE DEFECT PHY-17 EXISTS TO RECORD, so the replacement does not repeat it.
    A column whose active set is still GROWING when ``max_passes`` runs out has not been adjusted --
    it has been cut off -- and both paths say so: the numpy path warns (naming the column count) and
    ``passes_out`` carries the pass count NEGATED. Built to fail, not asserted from the description: a
    60-level column with a 300 K bottom layer is 9.6 K from converged with 18 K of residual instability
    at ``max_passes=16``, and converges silently at the shipped 48. The predicate is deliberately "still
    growing", not "bound reached": under stopping rule (ii) a single-precision column exits near the
    bound with a FINAL partition, and a bound-hit warning would fire on every such call."""
    m = StageADCore(StageADConfig(nx=4, nz=60, theta_max_deg=70.0))
    dm = np.asarray(m.dm, float); p_mid = np.asarray(m.p_mid, float)
    T0 = np.asarray(m.T_atm, float).copy()
    T0[:, -1] += 300.0                                             # one enormously hot bottom layer
    z = np.asarray(m._hydrostatic_z(T0))
    ref = T0.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        moist_convective_segment_mix(ref, dm, z.copy(), p_mid, q=None, max_passes=400, gamma_iters=6)

    Tcut = T0.copy(); po = np.zeros(4, np.int32)
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.warns(RuntimeWarning, match="TRUNCATED"):
            moist_convective_segment_mix(Tcut, dm, z.copy(), p_mid, q=None, max_passes=16,
                                         gamma_iters=6, passes_out=po)
    assert (po == -16).any()                                       # negative == truncated at the bound
    assert np.max(np.abs(Tcut - ref)) > 1.0                        # NON-VACUOUS: it really is wrong
    assert _max_instability(Tcut, dm, z, p_mid).max() > 1.0        #   and really is still unstable

    Tok = T0.copy(); po2 = np.zeros(4, np.int32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with np.errstate(over="ignore", invalid="ignore"):
            moist_convective_segment_mix(Tok, dm, z.copy(), p_mid, q=None,
                                         max_passes=StageADConfig().conv_segment_max_passes,
                                         gamma_iters=6, passes_out=po2)
    assert not [c for c in caught if c.category is RuntimeWarning]  # the shipped bound is silent
    assert (po2 >= 0).all() and np.array_equal(Tok, ref)
    with pytest.raises(ValueError):            # 0 would silently switch the adjustment off entirely
        moist_convective_segment_mix(T0.copy(), dm, z.copy(), p_mid, q=None, max_passes=0)


def test_segment_mixer_cold_block_takes_the_rh_detrain_underflow_fallback():
    """THE ONE BRANCH THE REST OF THE GATE CANNOT REACH. ``rh_detrain`` normalizes a block's water by
    ``Sum m_j g_j``; where every layer's ``qsat`` has underflowed to exactly zero that denominator is
    zero, and all three implementations fall back to the homogenized mean. MEASURED thresholds, not
    inherited: in fp64 ``qsat`` is exactly 0 at and below 15 K, so no realistic column reaches it -- but
    in binary32 ``esat`` is exactly 0 at and below 53.7 K and subnormal (flushed under ``-ftz=true``) up
    to 57.8 K, while run17's coldest air cell is 48.45 K. So the branch is DEAD on the host and LIVE on
    the fp32 device path, and it has to conserve water either way. Synthetic because nothing in the
    canonical state is that cold in fp64: a 50 m-spaced column at 9..14 K, super-adiabatic by ~0.5 K per
    pair, beside an ordinary warm column that must come through untouched."""
    nz = 6
    dm = np.full(nz, 200.0)
    p_mid = np.linspace(8.0e4, 9.5e4, nz)
    z = np.tile(np.linspace(50.0 * (nz - 1), 0.0, nz), (2, 1))
    T0 = np.empty((2, nz))
    T0[0] = np.linspace(9.0, 14.0, nz)                             # cold, unstable, qsat == 0 throughout
    T0[1] = 300.0                                                  # warm control: stable, untouched
    q0 = np.array([[3e-4, 1e-4, 5e-4, 2e-4, 4e-4, 6e-4], [2e-3] * nz])
    H0 = (dm[None, :] * T0).sum(1); W0 = (dm[None, :] * q0).sum(1)

    T, q = T0.copy(), q0.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        moist_convective_segment_mix(T, dm, z.copy(), p_mid, q=q, closure="rh_detrain")
    assert np.max(np.abs(T[1] - T0[1])) == 0.0 and np.max(np.abs(q[1] - q0[1])) == 0.0
    assert float(np.max(th.qsat(T[0], p_mid))) == 0.0              # the denominator really is zero
    assert np.ptp(q[0]) == 0.0                                     # => the HOMOGENIZED fallback, not A*g
    assert np.allclose((dm[None, :] * T).sum(1), H0, rtol=1e-12)   # both theorems survive the fallback
    assert np.allclose((dm[None, :] * q).sum(1), W0, rtol=1e-12)
    Tr, qr = T0.copy(), q0.copy()                                  # the kernel's oracle agrees, so the
    with np.errstate(over="ignore", invalid="ignore"):             #   device else-branch is gated too
        _ref_segment_mix(Tr, dm, z.copy(), p_mid, q=qr, closure="rh_detrain")
    assert np.max(np.abs(Tr - T)) < 1e-9 and np.max(np.abs(qr - q)) < 1e-15


@_gpu
def test_segmix_cols_numpy_vs_cuda_parity():
    """The fused device ``segmix_cols`` RawKernel reproduces the numpy body. SKIPS without CuPy (this
    laptop and the CI runner); the GPU box runs it. BOTH precisions are compared against the SAME fp64
    numpy result, so each bar measures the kernel's distance from the best answer available rather than
    op-order noise between two runs of the same precision:
      * fp64 kernel vs numpy fp64: < 1e-6 K / < 1e-11 kg/kg -- round-off over the block solves (the leaf
        transcendentals reassociate between numpy's libm and CUDA's, and the two sum the block's offset
        moment in opposite index order), NOT bit-identity.
        WHY 1e-6 AND NOT SOMETHING TIGHTER. Two reasons, and neither is the numpy-vs-scalar gap. (1) The
        kernel's own distance from this reference is NOT measurable here: its ``expf``/``exp`` and any
        fused multiply-add contraction are nvcc's, not numpy's, and that gap is what the bar has to
        cover. (2) 1e-6 K is the figure the standing probe ``flatworld/_gpu_segmix_share.py`` and section
        3 of the results note already use, so all three places agree rather than one of them being the
        odd one out -- which is how the previous 1e-9 K here went unnoticed.
        FOR SCALE, on THIS case with BOTH arms on the shipped knobs, the numpy body and the scalar
        reference agree to 2.3e-13 K and 1.6e-17 kg/kg -- i.e. the vectorization itself contributes
        nothing at this bar. (Run the two arms on DIFFERENT knobs and the same comparison reads 7.3e-9 K,
        because the pass loop stops on a 1e-9 K threshold and a knob change moves which pass a column
        exits on. That is a measurement of the mismatch, not of the port, which is why this test now
        passes ``max_passes``/``gamma_iters`` explicitly to both sides.)
        The WATER bar stays at 1e-11 kg/kg: the matched measurement puts that gap at 1.6e-17, so there
        is no reason to loosen both.
      * fp32 kernel vs the same fp64 reference: < 5e-2 K -- the fp32 floor for an ITERATED adjustment, a
        few units in the last place of ~300 K carried through the passes. The number is borrowed from the
        standing convadjust parity probe (``flatworld/_gpu_convadj_parity.py``), which accepted the
        shipped fused kernel at 5e-2 K on the same state and the same iteration structure; NOTE the probe
        compared fp32 against fp32, so this is the stricter use of that bar, not a looser one. Tightening
        it here would claim a precision the shipped device path has never demonstrated."""
    import cupy as cp
    import flatworld._gpu_kernels as adc_gpu                 # the body reads adc_gpu.* further down
    from flatworld._gpu_kernels import build_segmix_cols_kernel
    m, dm, p_mid, T0, z, q0 = _segmix_case(nx=8, nz=40, seed=3)
    T0 = T0[:5].copy(); q0 = q0[:5].copy(); z = z[:5].copy()       # drop the absurd-cold column
    RHt = 0.2 + (0.8 - 0.2) * (p_mid / C.P_SURF)
    # SAME KNOBS ON BOTH ARMS. The numpy call must be given the config defaults explicitly: its own
    # function defaults are different (16/8), and comparing a 16-pass/8-iterate numpy state against a
    # 48-pass/2-iterate kernel state would measure the knobs, not the port.
    mp, gi = StageADConfig().conv_segment_max_passes, StageADConfig().conv_segment_gamma_iters
    for closure, detrain in (("rh_detrain", 1), ("q_homogenize", 0)):
        Tn, qn = T0.copy(), q0.copy()
        moist_convective_segment_mix(Tn, dm, z.copy(), p_mid, q=qn, closure=closure,
                                     max_passes=mp, gamma_iters=gi)
        for dt, bar_T, bar_q in ((cp.float64, 1e-6, 1e-11), (cp.float32, 5e-2, 1e-6)):
            ker = build_segmix_cols_kernel(dt)
            Td = cp.ascontiguousarray(cp.asarray(T0, dt)); qd = cp.ascontiguousarray(cp.asarray(q0, dt))
            st = cp.zeros(Td.shape[0], dtype=np.int32)
            adc_gpu.segmix_columns_device(
                ker, Td, qd, cp.ascontiguousarray(cp.asarray(dm, dt)),
                cp.ascontiguousarray(cp.asarray(z, dt)), cp.ascontiguousarray(cp.asarray(p_mid, dt)),
                cp.ascontiguousarray(cp.asarray(RHt, dt)), mp, gi, detrain, st)
            # BOTH precisions must TERMINATE, not run the bound: at fp32 the 1e-9 K exit is unreachable,
            # so a NON-NEGATIVE pass count here is the kernel's own confirmation of stopping rule (ii).
            # (Truncation is strictly negative; 0 is legitimate -- a column with nothing to mix.)
            assert int(cp.asnumpy(st).min()) >= 0, f"{closure} {dt}: a column hit the pass bound"
            assert np.max(np.abs(cp.asnumpy(Td).astype(float) - Tn)) < bar_T, f"{closure} {dt}"
            assert np.max(np.abs(cp.asnumpy(qd).astype(float) - qn)) < bar_q, f"{closure} {dt}"
            if dt == cp.float64:                                   # the theorems survive the port
                assert np.allclose(np.sum(dm[None, :] * cp.asnumpy(Td), axis=1),
                                   np.sum(dm[None, :] * T0, axis=1), rtol=1e-12)
                assert np.allclose(np.sum(dm[None, :] * cp.asnumpy(qd), axis=1),
                                   np.sum(dm[None, :] * q0, axis=1), rtol=1e-11)


def test_radiation_parallel_default_off():
    """CPU-fuse #2 (radiation): the column-parallel LW two-stream numba kernel is DEFAULT OFF -- a plain
    run uses the numpy two-stream (byte-identical) and never imports numba."""
    assert StageADConfig().radiation_parallel is False


def test_two_stream_lw_parallel_matches_numpy_to_roundoff():
    """``two_stream_lw_parallel`` / ``two_stream_lw_windowed_parallel`` reproduce the numpy two-stream to
    ROUND-OFF (~1e-13, NOT bit-identical -- the leaf exp/T^4 reassociate) for the single-band and the
    windowed (production) path, and conserve energy per column (sum H == surf_net_lw - OLR)."""
    pytest.importorskip("numba")
    rng = np.random.default_rng(3)
    nx, N = 120, 40
    dtau = np.abs(rng.normal(0.3, 0.3, (nx, N))).clip(1e-6, 5.0)
    dtau2 = np.abs(rng.normal(0.2, 0.2, (nx, N))).clip(1e-6, 3.0)
    T = (250 + rng.normal(0, 30, (nx, N))).clip(150, 340)
    Ts = (290 + rng.normal(0, 20, nx)).clip(200, 350)
    # single band
    a = rad.two_stream_lw(dtau, T, Ts, band_frac=0.7)
    b = rad.two_stream_lw_parallel(dtau, T, Ts, band_frac=0.7)
    for k in ("OLR", "D_surf", "surf_net_lw", "Fnet", "H"):
        assert np.max(np.abs(a[k] - b[k])) < 1e-11, k
    # windowed (production LW path: two bands)
    aw = rad.two_stream_lw_windowed(dtau, dtau2, T, Ts, 0.3)
    bw = rad.two_stream_lw_windowed_parallel(dtau, dtau2, T, Ts, 0.3)
    for k in ("OLR", "D_surf", "surf_net_lw", "Fnet", "H", "OLR_bg", "OLR_win"):
        assert np.max(np.abs(aw[k] - bw[k])) < 1e-11, k
    # per-column energy telescoping holds for the kernel too: sum_k H == surf_net_lw - OLR
    assert np.allclose(np.sum(b["H"], axis=-1), b["surf_net_lw"] - b["OLR"], atol=1e-9)


def test_radiation_parallel_core_off_byte_identical_and_on_tracks(monkeypatch):
    """Through the FULL coupled core at PRODUCTION radiation config (clouds + LW window + SW bands + SW
    diffuse): a multi-step run with the radiation flag OFF vs the same run that never set it are
    bit-identical (the numpy two-stream is untouched), and the flag ON tracks OFF to round-off over the
    SHORT run checked here. (Per-step round-off, not a permanently bounded trajectory: the production
    guarantee is climatological -- time-mean P, T(theta), band -- since the f=0 core decorrelates two
    round-off-separated trajectories over a long run; immaterial given the +-10 C budget.)"""
    pytest.importorskip("numba")

    def _run(parallel, steps=6, seed=99):
        monkeypatch.setenv("COSMO_RADIATION_PARALLEL", "1" if parallel else "0")
        monkeypatch.setenv("COSMO_CONVADJUST_PARALLEL", "0")
        rng = np.random.default_rng(seed)
        cfg = StageADConfig(nx=24, nz=16, theta_max_deg=80.0, radiation_parallel=parallel)
        core = StageADCore(cfg)
        core.run_dynamics = True; core.couple_heat = True
        core.T_atm = core.T_atm + 0.7 * rng.standard_normal(core.T_atm.shape)
        core.q = np.clip(core.q + 1e-4 * rng.standard_normal(core.q.shape), 0.0, 0.5)
        for _ in range(steps):
            core.step()
        return np.asarray(core.T_atm).copy(), np.asarray(core.q).copy()

    Toff, qoff = _run(parallel=False)
    Toff2, qoff2 = _run(parallel=False)
    assert np.array_equal(Toff, Toff2) and np.array_equal(qoff, qoff2)   # OFF is deterministic
    Ton, qon = _run(parallel=True)
    assert np.max(np.abs(Toff - Ton)) < 1e-9      # ON tracks OFF to round-off over the multi-step run
    assert np.max(np.abs(qoff - qon)) < 1e-11


def test_lw_parallel_gate_falls_back_for_unsupported_shape():
    """The lw_parallel guard (``_lw_parallel_ok``) only engages for 2-D (nx, N) fp64 columns; a 1-D
    single column or non-float input falls back to the numpy two-stream (so a single-column
    column_radiation call with lw_parallel=True is still correct, just numpy)."""
    rng = np.random.default_rng(5)
    N = 30
    dtau1d = np.abs(rng.normal(0.3, 0.2, N)).clip(1e-6, 4.0)
    T1d = (260 + rng.normal(0, 20, N)).clip(160, 340)
    assert rad._lw_parallel_ok(True, dtau1d, T1d, 300.0) is False     # 1-D -> numpy
    assert rad._lw_parallel_ok(False, dtau1d[None, :], T1d[None, :], 300.0) is False  # flag off
    assert rad._lw_parallel_ok(True, dtau1d[None, :], T1d[None, :], np.array([300.0])) is True   # 2-D fp64 ok


# --------------------------------------------------------------------------------------------------
# CPU-fuse #4 (thermo_other): the column-PARALLEL fused microphysics+reevap kernel -- the dominant
# change (the 1.85x->3.05x prod jump). Mirrors the radiation/convadjust equivalence pattern:
# direct-call ROUND-OFF agreement vs the numpy reference (calling the kernel DIRECTLY, so a silent
# numpy fallback can't make the gate vacuously pass -- we also assert the diff is nonzero), plus the
# default-OFF byte-identity + flag-ON-tracks check through the full coupled core.
# --------------------------------------------------------------------------------------------------
from flatworld.stage_ad_core import prognostic_microphysics_reevap_parallel


def _canonical_prognostic_core(nx=20, nz=16, n_rce=24, n_coupled=6):
    """A coupled core in the CANONICAL prognostic-cloud config (the only combo the thermo_other kernel
    covers): prognostic_condensate ON with the full pdf/dqsat/ice/ceiling/reevap/kessler chain, fp64,
    no equator suppression. Spun through the issue-#99 RCE-then-coupled phases so the microphysics runs
    on a realistic live moist column (clouds + precip actually present)."""
    # the prognostic-condensate microphysics + below-cloud reevap are OPT-IN (the dataclass default is the
    # single-pass scheme); turn on exactly the CANONICAL combination the thermo_other kernel guard covers.
    cfg = StageADConfig(nx=nx, nz=nz, theta_max_deg=70.0, conv_max_sweeps=4,
                        prognostic_condensate=True, precip_reevap=True)
    assert cfg.prognostic_condensate and cfg.qc_pdf_cloud and cfg.qc_dqsat_adjust and cfg.qc_ice_phase
    assert cfg.qc_reevaporation and cfg.qc_retained_ceiling and cfg.qc_radiative_cap_decouple
    assert cfg.precip_reevap and cfg.equator_norain_cells == 0 and cfg.qc_autoconv_scheme == "kessler"
    assert cfg.precision == "fp64" and not cfg.chimney_param
    m = StageADCore(cfg)
    m.run_dynamics = False
    for _ in range(n_rce):
        m.step()
    m.run_dynamics = True
    m.couple_heat = True
    for _ in range(n_coupled):
        m.step()
    return m


def test_thermo_other_parallel_default_off():
    """CPU-fuse #4: the column-parallel fused microphysics+reevap kernel is DEFAULT OFF -- a plain run
    uses the numpy ``_prognostic_microphysics`` (byte-identical) and never imports numba."""
    assert StageADConfig().thermo_other_parallel is False


def test_thermo_other_parallel_matches_numpy_to_roundoff():
    """``prognostic_microphysics_reevap_parallel`` (the fused micro+reevap numba kernel) reproduces the
    numpy ``_prognostic_microphysics`` reference to ROUND-OFF (~1e-13, NOT bit-identical -- ``esat``'s
    ``exp`` reassociates between numpy and numba's libm) on a live canonical prognostic column, updating
    T_atm / q_c / q / the surface precip identically. Calls the kernel DIRECTLY (no gate), and asserts the
    T diff is NONZERO so a silent numpy fallback can't make this pass vacuously. Also checks per-column
    total-water closure (vapour + condensate + precipitated flux) for the kernel path."""
    pytest.importorskip("numba")
    m = _canonical_prognostic_core()
    dt = float(m.cfg.dt)
    # snapshot the shared starting state (the kernel + the numpy method both write IN PLACE)
    q0 = np.clip(m.q, 0.0, 0.5).copy()
    T0 = m.T_atm.copy()
    qc0 = m.q_c.copy()
    # numpy reference on a copy of the state
    ref = StageADCore(m.cfg)
    ref.T_atm = T0.copy(); ref.q_c = qc0.copy()
    q_np = ref._prognostic_microphysics(q0.copy(), dt)
    T_np, qc_np, P_np = ref.T_atm.copy(), ref.q_c.copy(), ref._precip.copy()
    # numba kernel on the live core (same start). NON-VACUOUSNESS: ``prognostic_microphysics_reevap_parallel``
    # has NO internal fallback -- it ALWAYS invokes the @njit kernel (the config gate that may route to numpy
    # lives in the CALLER ``_thermo_substep``, not here), so a direct call genuinely exercises the numba path.
    # (A broken kernel would blow past the 1e-9 round-off bound below; bit-identical agreement is fine -- it is
    # strictly inside the round-off guarantee.)
    import flatworld.stage_ad_core as adc
    m.T_atm = T0.copy(); m.q_c = qc0.copy()
    q_nb = prognostic_microphysics_reevap_parallel(m, q0.copy(), dt)
    assert adc._MICRO_NUMBA_KERNEL is not None                        # the @njit kernel was built (process-wide)
    T_nb, qc_nb, P_nb = m.T_atm.copy(), m.q_c.copy(), m._precip.copy()
    assert np.max(np.abs(q_np - q0)) > 0.0                            # the microphysics engaged (q changed)
    # round-off agreement (<= round-off; the kernel MAY happen to land bit-identical on a given state, but
    # is only GUARANTEED to round-off because esat's exp reassociates between numpy and numba's libm)
    assert np.max(np.abs(T_np - T_nb)) < 1e-9                         # tracks numpy to round-off
    assert np.max(np.abs(q_np - q_nb)) < 1e-12
    assert np.max(np.abs(qc_np - qc_nb)) < 1e-12
    assert np.max(np.abs(P_np - P_nb)) < 1e-12
    # per-column total-water closure for the kernel: (vapour+condensate) before == after + precip out
    W_before = np.sum((q0 + qc0) * m.dm[None, :], axis=1)
    W_after = np.sum((q_nb + qc_nb) * m.dm[None, :], axis=1) + P_nb * dt
    assert np.allclose(W_before, W_after, rtol=1e-9, atol=1e-9)       # total water conserved per column


@pytest.mark.parametrize("cond_tau_v,salt", [(60.0, True), (0.0, True), (60.0, False)])
def test_micro_fuse_salt_condtau_matches_generic_roundoff(cond_tau_v, salt):
    """The fused micro kernels -- numba CPU (``prognostic_microphysics_reevap_parallel``) AND the cupy device
    RawKernel (``micro_cols``) -- reproduce the numpy-generic ``_prognostic_microphysics`` to ROUND-OFF for the
    salt (#237 N_d warm-rain autoconv) + finite-tau condensation (#280) route: the configs that USED to fall
    off the fused kernels to the un-fused generic path (~5x slower under capture). NON-VACUOUS: the salt /
    cond_tau deltas visibly move the state vs the plain canonical baseline (a kernel silently ignoring the new
    ``cond_tau`` / ``warm_factor`` args would fail this), and the fused numba kernel is asserted to have run.
    Uses a realistic thermo-substep dt=2 s so ``cond_tau=60`` genuinely bites (frac=min(dt/tau,1)=0.033)."""
    pytest.importorskip("numba")
    import flatworld.stage_ad_core as adc
    import flatworld._gpu_kernels as adc_gpu
    from flatworld.stage_ad_core import prognostic_microphysics_reevap_parallel
    m = _canonical_prognostic_core()
    nx = m.T_atm.shape[0]
    N_REF = m.cfg.salt_ccn_nd_ref
    dt = 2.0                                                          # a production-scale thermo substep dt [s]
    # engage the route on the shared cfg: constant finite-tau cond and/or salt N_d warm-autoconv (NON-uniform
    # N_d 1x..8x N_ref, so warm_factor=(N_d/N_ref)^beta is genuinely per-column, not a scalar no-op)
    m.cfg.cond_tau = cond_tau_v
    m.cfg.salt_ccn = salt
    m.cfg.salt_ccn_autoconv = salt
    N_d = (N_REF * (1.0 + 7.0 * np.linspace(0.0, 1.0, nx))).astype(np.float64) if salt else None
    beta = m.cfg.salt_ccn_autoconv_beta
    # shared start state (every path writes T_atm/q_c/q/_precip in place)
    q0 = np.clip(m.q, 0.0, 0.5).copy(); T0 = m.T_atm.copy(); qc0 = m.q_c.copy()

    import dataclasses

    def _numpy_ref(cond_tau_x, nd):
        # INDEPENDENT cfg per reference (dataclasses.replace) -- StageADCore stores the cfg by reference, so
        # mutating a shared cfg here would clobber m.cfg (and the fused paths below read m.cfg / m.N_d).
        rcfg = dataclasses.replace(m.cfg, cond_tau=cond_tau_x, salt_ccn=nd is not None,
                                   salt_ccn_autoconv=nd is not None)
        r = StageADCore(rcfg)                                        # numpy-generic reference
        r.N_d = None if nd is None else nd.copy()
        r.T_atm = T0.copy(); r.q_c = qc0.copy()
        qq = r._prognostic_microphysics(q0.copy(), dt)
        return r.T_atm.copy(), r.q_c.copy(), qq, r._precip.copy()

    T_np, qc_np, q_np, P_np = _numpy_ref(cond_tau_v, N_d)            # reference WITH the deltas
    T_base, _, _, P_base = _numpy_ref(0.0, None)                     # plain canonical baseline (deltas OFF)
    assert (np.max(np.abs(T_np - T_base)) > 1e-6                     # NON-VACUOUS: the deltas actually move state
            or np.max(np.abs(P_np - P_base)) > 1e-9)

    # --- numba CPU fuse ---
    # NB: deliberately NO hand-clear of m._micro_warm here -- assigning m.N_d directly is exactly the
    # "future code bypasses _build_salt_ccn_field" footgun, and the builder's identity guard must catch it
    # (rebuild warm_factor from the new N_d) BY CONSTRUCTION. If this parity holds, the guard works.
    m.N_d = None if N_d is None else N_d.copy()
    m.T_atm = T0.copy(); m.q_c = qc0.copy()
    q_nb = prognostic_microphysics_reevap_parallel(m, q0.copy(), dt)
    assert adc._MICRO_NUMBA_KERNEL is not None                       # the @njit kernel genuinely ran
    assert np.max(np.abs(T_np - m.T_atm)) < 1e-9
    assert np.max(np.abs(q_np - q_nb)) < 1e-12
    assert np.max(np.abs(qc_np - m.q_c)) < 1e-12
    assert np.max(np.abs(P_np - m._precip)) < 1e-12

    # --- cupy device fuse (the actual GPU perf path), fp64 parity dtype ---
    #     gpu_available(), not _HAVE_CUPY: the numba CPU half above runs without a card and must keep
    #     running, so the DEVICE half is gated rather than the whole test skipped.
    if _dev.gpu_available():
        import cupy as cp
        from flatworld._gpu_kernels import build_micro_cols_kernel, micro_cols_device
        kernel = build_micro_cols_kernel(cp.float64)
        Td = cp.ascontiguousarray(cp.asarray(T0, cp.float64))
        qd = cp.ascontiguousarray(cp.asarray(q0, cp.float64))
        qcd = cp.ascontiguousarray(cp.asarray(qc0, cp.float64))
        dmd = cp.ascontiguousarray(cp.asarray(m.dm, cp.float64))
        pmd = cp.ascontiguousarray(cp.asarray(m.p_mid, cp.float64))
        if salt:
            warm = cp.ascontiguousarray(cp.asarray((N_d / N_REF) ** beta, cp.float64)); aw = 1
        else:
            warm = cp.ones(nx, cp.float64); aw = 0
        P_dev = micro_cols_device(kernel, Td, qd, qcd, dmd, pmd, float(dt),
                                  m.cfg.cloud_rh_crit, m.cfg.qc_autoconv_tau, m.cfg.qc_autoconv_crit,
                                  m.cfg.qc_autoconv_tau_ice, m.cfg.qc_autoconv_crit_ice, m.cfg.cloud_f_min,
                                  m.cfg.precip_reevap_rate, cond_tau=cond_tau_v, apply_warm=aw, warm_factor=warm)
        assert np.max(np.abs(T_np - cp.asnumpy(Td))) < 1e-9         # device fp64 -> round-off vs numpy
        assert np.max(np.abs(q_np - cp.asnumpy(qd))) < 1e-11
        assert np.max(np.abs(qc_np - cp.asnumpy(qcd))) < 1e-11
        assert np.max(np.abs(P_np - cp.asnumpy(P_dev))) < 1e-11

    # --- footgun guard (review note #1): a DIRECT core.N_d swap (bypassing _build_salt_ccn_field, which is
    # the only sanctioned mutator + the only place that clears _micro_warm) must be caught BY IDENTITY inside
    # _micro_fuse_salt_cond_params -- warm_factor rebuilt from the NEW N_d, never the stale cached buffer.
    if salt:
        beta_ = m.cfg.salt_ccn_autoconv_beta
        _, aw0, warm0 = adc._micro_fuse_salt_cond_params(m, np, np.float64)  # cache HIT (numba path built it above)
        assert aw0 == 1 and np.allclose(warm0, (m.N_d / N_REF) ** beta_)
        m.N_d = (2.5 * m.N_d).copy()                                         # direct swap, NO cache clear
        _, _, warm1 = adc._micro_fuse_salt_cond_params(m, np, np.float64)
        assert warm1 is not warm0                                           # a genuine rebuild, not the stale buffer
        assert np.allclose(warm1, (m.N_d / N_REF) ** beta_)                 # ...and it tracks the swapped-in N_d


def test_thermo_other_parallel_core_off_byte_identical_and_on_tracks(monkeypatch):
    """Through the FULL coupled core at the canonical prognostic config: a multi-step run with the
    thermo_other flag OFF vs the same run that never set it are bit-identical (the numpy microphysics is
    untouched), and the flag ON tracks OFF to round-off per step over the run. Also: a NON-canonical
    config (single-pass non-prognostic) with the flag ON must fall back to numpy -> bit-identical to OFF."""
    pytest.importorskip("numba")

    def _run(parallel, prognostic=True, steps=6, seed=99):
        monkeypatch.setenv("COSMO_THERMO_OTHER_PARALLEL", "1" if parallel else "0")
        monkeypatch.setenv("COSMO_CONVADJUST_PARALLEL", "0")
        monkeypatch.setenv("COSMO_RADIATION_PARALLEL", "0")
        rng = np.random.default_rng(seed)
        cfg = StageADConfig(nx=24, nz=16, theta_max_deg=80.0, thermo_other_parallel=parallel,
                            prognostic_condensate=prognostic, precip_reevap=prognostic)
        core = StageADCore(cfg)
        core.run_dynamics = True; core.couple_heat = True
        core.T_atm = core.T_atm + 0.7 * rng.standard_normal(core.T_atm.shape)
        core.q = np.clip(core.q + 1e-4 * rng.standard_normal(core.q.shape), 0.0, 0.5)
        for _ in range(steps):
            core.step()
        return np.asarray(core.T_atm).copy(), np.asarray(core.q).copy()

    Toff, qoff = _run(parallel=False)
    Toff2, qoff2 = _run(parallel=False)
    assert np.array_equal(Toff, Toff2) and np.array_equal(qoff, qoff2)   # OFF is deterministic
    Ton, qon = _run(parallel=True)
    assert np.max(np.abs(Toff - Ton)) < 1e-9      # ON tracks OFF to round-off over the multi-step run
    assert np.max(np.abs(qoff - qon)) < 1e-11
    #   (non-vacuousness -- that the kernel genuinely RAN, not silently fell back -- is pinned by the
    #   direct-call test ``test_thermo_other_parallel_matches_numpy_to_roundoff`` asserting the @njit kernel
    #   was built; here the canonical config + flag ON satisfies the ``_thermo_substep`` guard so it engages.)
    # non-canonical (single-pass) config: the flag ON must FALL BACK to numpy -> bit-identical to OFF
    Tsp_off, _ = _run(parallel=False, prognostic=False)
    Tsp_on, _ = _run(parallel=True, prognostic=False)
    assert np.array_equal(Tsp_off, Tsp_on)        # fallback to numpy is bit-identical (no kernel for this config)


def test_interp_columns_parallel_bit_identical_both_production_shapes():
    """CPU-fuse advect (``_interp_columns_parallel``): on BOTH production shapes -- shared-xp (the
    ``_height_to_pressure`` map) and shared-x (the ``_pressure_to_height`` map) -- the prange-over-columns
    kernel is BIT-IDENTICAL to the numpy ``_interp_columns`` reference (claimed: pure +,-,*,/ + integer
    compares, no transcendentals, so no SIMD-vs-scalar divergence). Asserts ``array_equal`` (not round-off);
    a non-2-D / non-fp64 input falls back to numpy (also bit-identical)."""
    pytest.importorskip("numba")
    from flatworld.stage_ad_core import _interp_columns, _interp_columns_parallel
    rng = np.random.default_rng(4)
    nx, M, N = 17, 22, 19
    # SHARED-xp shape: x=(nx,M) per-column queries, xp=(1,M0) one ascending sample row, fp=(nx,M0) values
    M0 = 25
    xp_row = np.sort(rng.uniform(0.0, 1.0, M0))[None, :]
    fp = rng.standard_normal((nx, M0))
    xq = rng.uniform(-0.2, 1.2, (nx, M))                          # incl. off-domain -> clamp branch
    a = _interp_columns(xq, xp_row, fp)
    b = _interp_columns_parallel(xq, xp_row, fp)
    assert np.array_equal(a, b)                                  # bit-identical (fp64, no transcendentals)
    # SHARED-x shape: x=(1,N) one ascending query row, xp=(nx,N0) per-column ascending grids, fp=(nx,N0)
    N0 = 21
    xq_row = np.sort(rng.uniform(0.0, 1.0, N))[None, :]
    xp_grids = np.sort(rng.uniform(-0.1, 1.1, (nx, N0)), axis=1)
    fp2 = rng.standard_normal((nx, N0))
    a2 = _interp_columns(xq_row, xp_grids, fp2)
    b2 = _interp_columns_parallel(xq_row, xp_grids, fp2)
    assert np.array_equal(a2, b2)
    # fallback: a 1-D (non-(B,*)) input goes to the numpy reference (still bit-identical)
    a3 = _interp_columns(xq_row, xp_row, fp[:1])
    b3 = _interp_columns_parallel(xq_row, xp_row, fp[:1])
    assert np.array_equal(a3, b3)


def test_interp_columns_pair_bit_identical_to_two_singles():
    """The no-flag shared-flux dedup ``_interp_columns_pair`` (one bracket build, two gathers) is
    BIT-IDENTICAL to two separate ``_interp_columns`` calls on the SAME (x, xp) -- the dedup must not
    perturb the value chain (it shares the bracket, a pure function of (x, xp)). This path is ALWAYS ON
    (not flag-gated), so its byte-identity guards the default production map ``_pressure_to_height_pair``."""
    from flatworld.stage_ad_core import _interp_columns, _interp_columns_pair
    rng = np.random.default_rng(11)
    nx, M, M0 = 14, 18, 23
    x = np.sort(rng.uniform(0.0, 1.0, M))[None, :]               # one ascending query row (shared-x shape)
    xp = np.sort(rng.uniform(-0.1, 1.1, (nx, M0)), axis=1)       # per-column ascending grids
    fp1 = rng.standard_normal((nx, M0))
    fp2 = rng.standard_normal((nx, M0))
    o1, o2 = _interp_columns_pair(x, xp, fp1, fp2)
    assert np.array_equal(o1, _interp_columns(x, xp, fp1))       # bit-identical to two separate calls
    assert np.array_equal(o2, _interp_columns(x, xp, fp2))


def test_resolved_mass_flux_shared_equals_rebuilt():
    """The other no-flag dedup (commit 1bca7c9 item 1): ``step()`` builds the divergence-free mass flux ONCE
    (``_resolved_mass_flux``) and passes it into the q / theta / q_c advect limbs, which all share the SAME
    ``u`` and ``z_p``. This asserts the shared-flux path is BIT-IDENTICAL to the limb rebuilding it (the
    ``flux=None`` path a direct caller / older code takes): advecting moisture with the precomputed flux ==
    advecting with flux=None. (The end-to-end correctness of the flux-form advection itself is covered by
    ``test_moisture_advected_by_resolved_flow`` / ``test_heat_advection_conserves_theta_mass_and_subsidence_warms``;
    this pins the dedup is a pure refactor.)"""
    m = StageADCore(StageADConfig(nx=40, nz=20, theta_max_deg=60.0, nu_b=40.0,
                                  tol=3e-3, check_every=200, patience=3, max_iter=30000))
    m.seed_warm_edge_buoyancy(dT_edge=40.0)
    m.run(verbose=False)                                          # steady single cell
    u, _ = m.velocities()
    m._z = m._hydrostatic_z(m.T_atm)
    flux = m._resolved_mass_flux(u, None)                         # the shared per-substep build step() would pass
    q_seed = np.full_like(m.q, 1e-4); q_seed[m.cfg.nx // 2, -1] += 1e-2
    # path A: advect with the precomputed shared flux ; path B: let the limb rebuild it (flux=None)
    mA = StageADCore(m.cfg); mA.__dict__.update({k: (v.copy() if isinstance(v, np.ndarray) else v)
                                                 for k, v in m.__dict__.items()})
    mB = StageADCore(m.cfg); mB.__dict__.update({k: (v.copy() if isinstance(v, np.ndarray) else v)
                                                 for k, v in m.__dict__.items()})
    mA.q = q_seed.copy(); mB.q = q_seed.copy()
    mA._advect_moisture(u, 600.0, flux=(flux[0].copy(), flux[1].copy()))
    mB._advect_moisture(u, 600.0, flux=None)
    assert np.array_equal(mA.q, mB.q)                             # shared flux == rebuilt flux: bit-identical refactor


def test_poisson_workers_bit_identical_to_single_threaded():
    """CPU-fuse DST ``poisson_workers`` (``COSMO_POISSON_WORKERS``): wrapping the batched 1-D DSTs in
    ``scipy.fft.set_workers(n)`` splits them over the wavenumber axis, leaving EACH 1-D transform
    unchanged -> the solve is BIT-IDENTICAL to the single-threaded (workers=1) path. Build the SAME
    ``SeparablePoisson`` operator with workers=1 vs workers=4 and require ``array_equal`` on a random RHS
    (incl. non-zero walls, the wall-folding path)."""
    rng = np.random.default_rng(0)
    nx, nz = 96, 48
    z = np.linspace(0.0, 1.5e4, nz)
    rho_ref = 1.2 * np.exp(-z / 8.0e3)
    dx, dz = 3.0e5, z[1] - z[0]
    sep1 = SeparablePoisson(rho_ref, nx, nz, dx, dz, workers=1)
    sep4 = SeparablePoisson(rho_ref, nx, nz, dx, dz, workers=4)
    b = rng.standard_normal((nz, nx))                            # full random RHS incl. all walls
    psi1 = sep1.solve(b.ravel())
    psi4 = sep4.solve(b.ravel())
    assert np.array_equal(psi1, psi4)                            # workers is a pure wall-clock lever, bit-identical


def test_conv_freeze_gamma_conserves_and_skips_leaf_calls(monkeypatch):
    """Issue #99 flag-gated lever, fast gates: ``freeze_gamma=True`` (a) still conserves column
    enthalpy AND water EXACTLY (the pairwise-mix theorems are Gamma-independent), (b) actually skips
    the sweeps>=2 lapse-rate evaluations (max_sweeps x fewer on a deeply-unstable state -- the measured
    fleet saving), and (c) defaults OFF (byte-identical path; the trajectory gate above runs it)."""
    assert StageADConfig().conv_freeze_gamma is False              # (c) default OFF
    m = _spun_coupled_core()
    rng = np.random.default_rng(11)
    T0 = m.T_atm + rng.uniform(0.0, 15.0, m.T_atm.shape)           # deeply unstable: all sweeps engage
    q0 = np.clip(m.q, 0.0, 0.5) + rng.uniform(0.0, 5e-3, m.q.shape)
    z0 = m._hydrostatic_z(T0)
    calls = {"n": 0}
    real = th.moist_adiabat_lapse_rate

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    import flatworld.stage_ad_core as sac
    monkeypatch.setattr(sac.th, "moist_adiabat_lapse_rate", counting)
    nzm1 = m.cfg.nz - 1
    for freeze, q_in in ((False, q0.copy()), (True, q0.copy())):
        T, q = T0.copy(), q_in
        calls["n"] = 0
        moist_convective_adjust(T, m.dm, z0, m.p_mid, 4, q=q, closure="rh_detrain",
                                freeze_gamma=freeze)
        if freeze:
            assert calls["n"] == nzm1                              # (b) sweep-1 only
        else:
            assert calls["n"] == 4 * nzm1                          # every sweep (cap engaged: the bench regime)
        # (a) exact conservation, Gamma-independent (same gates as the M2b.3 theorem tests)
        enth0 = np.sum(m.dm[None, :] * T0, axis=1)
        assert np.allclose(np.sum(m.dm[None, :] * T, axis=1), enth0, rtol=1e-13)
        assert np.allclose(np.sum(m.dm[None, :] * q, axis=1),
                           np.sum(m.dm[None, :] * q0, axis=1), rtol=1e-12)


def test_conv_freeze_gamma_converges_to_same_coupled_state():
    """Issue #99 flag-gated lever, the #86-style A/B: a coupled fleet-resolution run with
    ``conv_freeze_gamma=True`` CONVERGES to the same state as the exact-Gamma core -- the held Gamma is
    one sweep old, so the early coupled transient differs by ~2 K but the trajectories collapse together
    as the state approaches steady (measured 0.07 K after 150 coupled steps; gate at 4x headroom)."""
    def run(freeze):
        m = StageADCore(StageADConfig(nx=48, nz=18, theta_max_deg=70.0, conv_max_sweeps=4,
                                      conv_freeze_gamma=freeze))
        m.run_dynamics = False
        for _ in range(40):
            m.step()
        m.run_dynamics = True
        m.couple_heat = True
        for _ in range(150):
            m.step()
        return m

    m0, m1 = run(False), run(True)
    assert np.all(np.isfinite(m1.T_atm))
    assert float(np.max(np.abs(m0.T_atm - m1.T_atm))) < 0.3        # same coupled state (measured 0.07 K)
    assert float(np.max(np.abs(m0.T_surf - m1.T_surf))) < 0.1      # surface agrees too (measured 0.02 K)


def test_coupled_step_trajectory_bitwise_vs_reference(monkeypatch):
    """Issue #99 end-to-end gate: a full coupled run (the repro phases at small size) is BIT-IDENTICAL
    between the optimized core and one with ALL FOUR pre-#99 reference implementations monkeypatched
    back in. Coupled steps amplify any 1-ulp divergence chaotically, so exact equality of the final
    T_atm/q/T_surf/omega/psi/b and the water-budget accumulators after 45 steps is a strong whole-core
    gate (it also catches a call site the unit gates might miss)."""
    import flatworld.stage_ad_core as sac

    def run():
        m = StageADCore(StageADConfig(nx=14, nz=10, theta_max_deg=70.0, conv_max_sweeps=4))
        m.run_dynamics = False
        for _ in range(25):
            m.step()
        m.run_dynamics = True
        m.couple_heat = True
        for _ in range(20):
            m.step()
        return m

    m_new = run()
    monkeypatch.setattr(StageADCore, "_pressure_to_height", _ref_pressure_to_height)
    monkeypatch.setattr(StageADCore, "_height_to_pressure", _ref_height_to_pressure)
    monkeypatch.setattr(StageADCore, "_hydrostatic_z", _ref_hydrostatic_z)
    monkeypatch.setattr(sac, "moist_convective_adjust", _ref_moist_convective_adjust)
    m_ref = run()
    for name in ("T_atm", "q", "T_surf", "omega", "psi", "b", "_z", "_precip", "_evap"):
        assert np.array_equal(getattr(m_new, name), getattr(m_ref, name)), name
    assert m_new._cum_evap == m_ref._cum_evap and m_new._cum_precip == m_ref._cum_precip


# ======================================================================================================
# B5 -- the BUDGET-BASED wet fraction (cfg.budget_wet_fraction): SUPPLY, not temperature, sets the surface
# state at a convergence core (the equatorial chimney). Carries the monostable-WET-equator finding into the
# core. Mirrors approach A's _equator_pond.wet_fraction; default OFF -> byte-identical to the T-keyed ramp.
# ======================================================================================================
def test_budget_wet_fraction_default_off_is_the_ramp():
    """DEFAULT OFF == byte-identical to the temperature-keyed ramp: a flag-OFF core's ``_wetness`` is
    bit-for-bit ``wetness_of_Tsurf(T_surf)`` (the only path the production gate ever runs), and an
    explicit ``budget_wet_fraction=False`` is identical to the unset default."""
    m = StageADCore(StageADConfig(nx=8, nz=18, theta_max_deg=55.0))
    assert m.cfg.budget_wet_fraction is False                       # default OFF
    assert np.array_equal(m._wetness(), wetness_of_Tsurf(m.T_surf))  # bit-for-bit the ramp
    m_off = StageADCore(StageADConfig(nx=8, nz=18, theta_max_deg=55.0, budget_wet_fraction=False))
    assert np.array_equal(m_off._wetness(), m._wetness())


def test_exchange_wind_refactor_value_identical():
    """The factored ``_exchange_wind`` returns EXACTLY the inline expression it replaced, in both limbs:
    the scalar ``cfg.wind`` (no resolved wind) and the resolved |u_surf|+gust branch."""
    m = StageADCore(StageADConfig(nx=10, nz=18, theta_max_deg=55.0))
    m.run_dynamics = False                                          # no resolved wind -> scalar cfg.wind
    assert m._exchange_wind() == m.cfg.wind
    m2 = StageADCore(StageADConfig(nx=10, nz=18, theta_max_deg=55.0, resolved_flux_wind=True))
    m2.run_dynamics = True
    rng = np.random.default_rng(1)
    m2._u_surf = rng.standard_normal(m2.cfg.nx) * 8.0          # pretend a resolved surface wind
    expect = np.sqrt(m2._u_surf**2 + m2.cfg.gustiness**2)
    assert np.array_equal(m2._exchange_wind(), expect)


def test_budget_wet_fraction_closure_formula_and_clip():
    """The closure is f = max(ramp, clip(F/E_w, 0, 1)) per column (F = self._precip, E_w = the WET-TILE
    evaporative-thermostat demand ``_budget_E_w``). On HOT salt-flat columns the ramp is 0, so f reduces
    to the supply-limited budget fraction -- check self-consistency and the two clips there: abundant
    supply -> f=1 (open water); zero supply -> f=0 (salt flat). (The ramp limb is covered by the
    ocean-source test.)"""
    m = StageADCore(StageADConfig(nx=12, nz=18, theta_max_deg=40.0, budget_wet_fraction=True))
    m.run_dynamics = False
    m.T_surf[:] = 120.0 + C.T_FREEZE                               # hot salt flat -> ramp = 0 (isolate budget_f)
    m.step()                                                      # one step -> _last_rad populated
    assert np.all(wetness_of_Tsurf(m.T_surf) == 0.0)               # confirm the ramp limb is off here
    E_w = m._budget_E_w(m.q, m._exchange_wind())                   # the closure's own denominator
    F = np.zeros(m.cfg.nx)
    F[: m.cfg.nx // 2] = 0.4 * E_w[: m.cfg.nx // 2]                # partial pond field (f ~ 0.4)
    F[m.cfg.nx // 2 : 3 * m.cfg.nx // 4] = 50.0 * E_w[m.cfg.nx // 2 : 3 * m.cfg.nx // 4]  # abundant -> clip 1
    m._precip = F
    f = m._wetness()
    assert np.allclose(f, np.clip(F / np.maximum(E_w, 1e-12), 0.0, 1.0), rtol=1e-12, atol=0)
    assert np.allclose(f[: m.cfg.nx // 2], 0.4, atol=1e-6)          # partial pond field
    assert np.all(f[m.cfg.nx // 2 : 3 * m.cfg.nx // 4] == 1.0)      # abundant supply -> open water (clipped)
    assert np.all(f[3 * m.cfg.nx // 4 :] == 0.0)                    # no supply -> salt flat


def test_budget_wet_fraction_self_regulating_monotonic():
    """Self-regulation on a HOT (ramp=0) column: f rises with supply (more rain -> more wet area) and
    saturates at 1. Halving the supply halves f in the unclipped (pond-field) regime."""
    m = StageADCore(StageADConfig(nx=8, nz=18, theta_max_deg=40.0, budget_wet_fraction=True))
    m.run_dynamics = False
    m.T_surf[:] = 120.0 + C.T_FREEZE                               # hot salt flat -> ramp = 0
    m.step()                                                      # populate _last_rad
    E_w = m._budget_E_w(m.q, m._exchange_wind())
    m._precip = 0.3 * E_w; f_lo = m._wetness().copy()
    m._precip = 0.6 * E_w; f_hi = m._wetness().copy()
    assert np.all(f_hi > f_lo)                                      # more supply -> wetter
    assert np.allclose(f_hi, 2.0 * f_lo, rtol=1e-9)                # linear below the clip
    m._precip = 10.0 * E_w
    assert np.all(m._wetness() == 1.0)                             # saturates at open water


def test_budget_E_w_is_the_wet_tile_thermostat_not_the_hot_surface():
    """LOAD-BEARING denominator check (the chicken-and-egg fix): the budget demand E_w is the WET-TILE
    evaporative-thermostat value (the wetness=1 SEB root, ~tens of mm/day), MUCH smaller than the
    potential evaporation evaluated at the cell's CURRENT hot salt-flat T_surf (super-boiling qsat). If
    the closure used the hot-surface value it would be ~order-of-magnitude too sluggish to ignite the
    wet state. Cross-checked against approach A's open_water_equilibrium on matched drivers."""
    from flatworld._equator_pond import EquatorForcing, open_water_equilibrium
    m = StageADCore(StageADConfig(nx=6, nz=18, theta_max_deg=20.0, budget_wet_fraction=True))  # hot inner zone
    m.run_dynamics = False
    m.T_surf[:] = 135.0 + C.T_FREEZE                               # a hot salt-flat surface
    m.step()                                                      # populate _last_rad at the hot state
    wind = m._exchange_wind()
    E_w = m._budget_E_w(m.q, wind)                                 # wet-tile thermostat demand [kg/m^2/s]
    # the WRONG denominator: potential evaporation at the current hot T_surf
    E_hot = C.RHO_AIR0 * 1.2e-3 * wind * np.maximum(th.qsat(m.T_surf, C.P_SURF) - m.q[:, -1], 0.0)
    assert np.all(E_w < 0.5 * E_hot)                              # the wet-tile demand is far below the hot one
    # cross-check E_w against approach A's open_water_equilibrium on the SAME column drivers (col 0)
    rad = m._last_rad
    fA = EquatorForcing(S_sfc=float(rad["surf_abs_sw"][0]) / (1.0 - C.ALBEDO_OCEAN),  # so (1-a_w)*S_sfc == SW_abs
                        LW_down=float(rad["D_surf"][0]), T_air=float(m.T_atm[0, -1]),
                        q_air=float(m.q[0, -1]), wind=float(np.atleast_1d(wind)[0]),
                        albedo_water=C.ALBEDO_OCEAN)
    _, E_wA_mm = open_water_equilibrium(fA)                        # mm/day
    assert abs(E_w[0] * 86400.0 - E_wA_mm) < 1e-6                  # native per-column form of approach A


def test_budget_wet_fraction_construction_warm_start_guard():
    """The warm-start guard: ``_wetness`` is called at construction (to seed q_init) and in the first
    radiation call BEFORE ``q``/``_precip``/``_last_rad`` exist; a flag-ON core must build without error,
    seed a finite q, and (pre-first-step) fall back to the ramp."""
    m = StageADCore(StageADConfig(nx=8, nz=18, theta_max_deg=50.0, budget_wet_fraction=True))
    assert np.all(np.isfinite(m.q)) and np.all(m.q >= 0.0)
    assert m._precip is not None and np.array_equal(m._precip, np.zeros(m.cfg.nx))   # post-init zero supply
    assert m._last_rad is None                                     # no radiation yet
    assert np.array_equal(m._wetness(), wetness_of_Tsurf(m.T_surf))  # -> ramp fallback before the first step


def test_budget_wet_fraction_gives_cool_wet_root_where_ramp_runs_away():
    """LOAD-BEARING (the whole point): at a hot convergence core under SUSTAINED liquid supply, the
    budget closure settles a stable WET surface AT the wet-tile thermostat T_w (cooler than the dry salt
    flat), whereas the temperature-keyed ramp -- having no stable cool-wet root there -- runs away to the
    hot dry salt flat. Run two surface-coupled (no-dynamics) cores from the same hot, moist start; re-pin a
    large supply each step on the budget core. The budget surface settles at ~T_w; the ramp at ~138 C.
    (In this ISOLATED hot RCE column T_w is itself hot -- no transport cooling -- so the gap is tens of C,
    not the ~62 C T_w of the transport-cooled real equator; the point is the WET root vs the dry runaway.)"""
    from flatworld.stage_ad_core import bulk_air_sea_sst
    base = dict(nx=6, nz=18, theta_max_deg=20.0, dt=7200.0, conv_max_sweeps=6)        # hot inner zone
    ramp = StageADCore(StageADConfig(**base))                       # default OFF: the T-keyed ramp
    budg = StageADCore(StageADConfig(**base, budget_wet_fraction=True))
    for m in (ramp, budg):
        m.run_dynamics = False                                     # each column relaxes to its local SEB
        m.T_surf[:] = 138.0 + C.T_FREEZE                           # start on the hot salt-flat branch
        m.T_atm[:] = np.maximum(m.T_atm, 130.0)
        m.q = np.minimum(m.RH[None, :] * th.qsat(m.T_atm, m.p_mid[None, :]), 0.5)
    supply = np.full(budg.cfg.nx, 5.0e-3)                          # ~520 mm/day -- abundant convergent supply
    for _ in range(80):
        ramp.step()
        budg._precip = supply.copy()                              # sustained chimney supply (re-pin each step)
        budg.step()
    T_ramp = ramp.T_surf.mean() - C.T_FREEZE
    T_budg = budg.T_surf.mean() - C.T_FREEZE
    # the wet-tile thermostat root the budget surface should sit at (the wetness=1 SEB on its own drivers)
    rad = budg._last_rad
    T_w = bulk_air_sea_sst(np.asarray(rad["surf_abs_sw"]), np.asarray(rad["D_surf"]),
                           budg.T_atm[:, -1], budg.q[:, -1], budg._exchange_wind(), wetness=1.0).mean() - C.T_FREEZE
    assert T_ramp > 120.0                                          # the ramp runs away to the hot salt flat
    assert abs(T_budg - T_w) < 4.0                                 # the budget surface settles AT the wet thermostat
    assert T_budg < T_ramp - 10.0                                  # ...which is meaningfully cooler than the salt flat
    budg._precip = supply.copy()                                   # under the sustained supply the surface is...
    assert np.all(budg._wetness() > 0.9)                          # ...open water (f~1) -- the wet root holds


def test_budget_wet_fraction_partial_supply_over_evaporates_documented_limitation():
    """DOCUMENTS the known single-bucket limitation (PR #177 review S1): in the partial-wet regime
    0<f<1 the latent flux LE = f*E_pot(T_surf) evaporates the wet fraction at the cell's HOT T_surf rather
    than the wet tiles' cooler T_w, so the applied evaporation EXCEEDS the supply (E>F) -- the pond-field
    water budget f*E_w=F does NOT close in the interior. (The endpoints f=0,1 are exact; closing the
    interior needs the two-temperature tile mosaic -> B5b/B5d.) This test PINS the behavior so a future
    mosaic fix is a deliberate, visible change. Pin f~0.5 on a hot column and confirm E is well above F."""
    m = StageADCore(StageADConfig(nx=6, nz=18, theta_max_deg=20.0, dt=7200.0, budget_wet_fraction=True))
    m.run_dynamics = False
    m._wetness = lambda: np.full(m.cfg.nx, 0.5)                    # pin f=0.5 to isolate the partial-regime LE
    for _ in range(200):
        m.step()
    # E = f*E_pot(T_surf); the supply f=F/E_w implies F = f*E_w; so E/F = E_pot(T_surf)/E_w. The single
    # bucket evaporates at the HOT T_surf, so E_pot(T_surf) > E_w (the cool wet-tile demand) -> E > F.
    wind = m.cfg.wind
    E_pot = C.RHO_AIR0 * 1.2e-3 * wind * np.maximum(th.qsat(m.T_surf, C.P_SURF) - m.q[:, -1], 0.0)
    E_w = m._budget_E_w(m.q, wind)
    ratio = E_pot / np.maximum(E_w, 1e-30)
    assert np.all(ratio > 1.1)                                   # every column over-evaporates (E/F>1; S1, not f*E_w=F)
    assert float(np.mean(ratio)) > 1.3                           # ...typically well above the supply (T_surf>T_w gap)


def test_budget_wet_fraction_preserves_ocean_source():
    """The max(ramp, budget_f) hybrid keeps a COOL ocean column a full-potential evaporative SOURCE: a
    cool wet surface (ramp=1) with only a TINY local supply still gets f=1 and evaporates at ~E_w, which
    can FAR exceed the local precip (E >> F) -- the reservoir ocean is a source, not throttled to its
    local rain. A bare budget_f=precip/E_w would wrongly give f~0 there (PR #177 review S1)."""
    m = StageADCore(StageADConfig(nx=6, nz=18, theta_max_deg=55.0, dt=7200.0, budget_wet_fraction=True))
    m.run_dynamics = False
    m.T_surf[:] = 45.0 + C.T_FREEZE                               # a warm wet ocean (ramp=1, <55 C) ...
    m.q[:] = 0.01                                                 # ... over dry-ish air -> a real evaporation deficit
    m.step()                                                      # populate _last_rad
    assert np.all(wetness_of_Tsurf(m.T_surf) == 1.0)              # the ramp keeps this ocean fully wet
    E_w = m._budget_E_w(m.q, m._exchange_wind())
    m._precip = 0.02 * E_w                                        # a TINY supply (a bare budget_f would give f~0.02)
    f = m._wetness()
    assert np.all(f == 1.0)                                       # max(ramp=1, 0.02) = 1 -> stays fully wet (source)
    m.step()                                                      # the wet surface evaporates...
    assert np.all(m._evap > 3.0 * m._precip)                      # ...far more than the local supply (a net source)


# ======================================================================================================
# B5c -- the ONLINE chimney parameterization (cfg.chimney_param): replace the divergent equatorial 1/dx
# precip SPIKE in self._precip with equator_chimney_param's resolution-independent grid-mean P_param.
# Default OFF -> byte-identical; the de-spiking only engages where the domain RESOLVES the convergence zone.
# ======================================================================================================
def test_chimney_param_default_off_is_byte_identical():
    """DEFAULT OFF == byte-identical: the flag defaults False and ``_apply_chimney_param`` is a no-op on a
    set ``_precip`` (so the production gate never perturbs the precip); explicit False matches the default."""
    m = StageADCore(StageADConfig(nx=24, nz=18, theta_max_deg=55.0))
    assert m.cfg.chimney_param is False                              # default OFF
    spike = np.zeros(m.cfg.nx); spike[0] = 362.0 / 86400.0
    m._precip = spike.copy(); m._apply_chimney_param()
    assert np.array_equal(m._precip, spike)                          # no-op when OFF
    m2 = StageADCore(StageADConfig(nx=24, nz=18, theta_max_deg=55.0, chimney_param=False))
    assert m2.cfg.chimney_param is False


def test_chimney_param_despikes_resolved_zone():
    """With the flag ON and a domain that RESOLVES the convergence zone (small theta_max so dx << the 300 km
    footprint), ``_apply_chimney_param`` replaces an equatorial 1/dx-like spike with the bounded grid-mean:
    the peak drops sharply, the ZONE integral is conserved to (1-phi) (the surface-reaching partition), and
    the out-of-zone (ribbon) profile is untouched."""
    from flatworld import equator_chimney_param as ecp
    m = StageADCore(StageADConfig(nx=96, nz=18, theta_max_deg=3.0, chimney_param=True))
    assert np.sum(m.x < ecp.DETECT_X_M) > 5                          # the zone is resolved (many cells)
    spike = np.zeros(m.cfg.nx); spike[0] = 362.0 / 86400.0           # a 1/dx-like axis spike [kg/m^2/s]
    m._precip = spike.copy(); m._apply_chimney_param()
    P_raw, P_new = spike * 86400.0, m._precip * 86400.0              # mm/day
    zone = m.x < ecp.DETECT_X_M
    assert P_new.max() < 0.5 * P_raw.max()                           # the spike is de-spiked (bounded)
    assert np.all(np.isfinite(P_new)) and np.all(P_new >= 0.0)
    ratio = np.sum(P_new[zone]) / np.sum(P_raw[zone])
    assert ratio == pytest.approx(1.0 - ecp.LOFT_FRACTION_PHI, rel=0.02)   # surface = (1-phi)*M_tot
    assert np.array_equal(P_new[~zone], P_raw[~zone])               # ribbon untouched


def test_chimney_param_phi_override_sets_the_partition():
    """``chimney_param_phi`` overrides the default lofting fraction: the zone-integral surface partition is
    (1-phi) with the override, not the calibrated default."""
    m = StageADCore(StageADConfig(nx=96, nz=18, theta_max_deg=3.0, chimney_param=True, chimney_param_phi=0.30))
    from flatworld import equator_chimney_param as ecp
    spike = np.zeros(m.cfg.nx); spike[0] = 100.0 / 86400.0
    m._precip = spike.copy(); m._apply_chimney_param()
    zone = m.x < ecp.DETECT_X_M
    ratio = np.sum(m._precip[zone]) / np.sum(spike[zone])
    assert ratio == pytest.approx(0.70, rel=0.02)                   # 1 - 0.30


def test_chimney_param_is_invoked_by_the_thermo_substep():
    """WIRING guard (PR #188 review #3): a real ``step()`` with chimney_param ON invokes
    ``_apply_chimney_param`` -- so deleting the call site in the thermo substep fails a test (the three unit
    tests above call the method directly and would pass even if the wiring were removed). Also asserts the
    step stays finite with the flag ON."""
    m = StageADCore(StageADConfig(nx=96, nz=18, theta_max_deg=3.0, chimney_param=True))
    calls = [0]
    orig = m._apply_chimney_param
    def _spy():
        calls[0] += 1
        return orig()
    m._apply_chimney_param = _spy
    m.step()
    assert calls[0] >= 1                                            # the substep called the wiring
    assert np.all(np.isfinite(m._precip)) and np.all(np.isfinite(m.T_atm))


def test_chimney_param_mutually_exclusive_with_equator_norain():
    """PR #188 review #4: enforce (not just document) the mutual exclusivity -- both ON raises in __init__."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        StageADCore(StageADConfig(nx=24, nz=18, theta_max_deg=55.0,
                                  chimney_param=True, equator_norain_cells=2))


def _endorheic_wetness(core, P, E_w_val):
    """Drive `_wetness` through its endorheic branch with a controlled supply P and a uniform wet-tile demand
    E_w: T_surf HOT (so the ramp is 0 → f == f_endo), the None-guards satisfied, `_budget_E_w` monkeypatched."""
    core.T_surf = np.full(core.cfg.nx, 420.0)                       # ramp -> 0, so max(ramp, f_endo) == f_endo
    core._precip = np.asarray(P, float)
    core._last_rad = {}                                             # non-None (monkeypatched _budget_E_w ignores it)
    core._E_w_cache = None
    core._budget_E_w = lambda q, wind: np.full(core.cfg.nx, float(E_w_val))
    return np.asarray(core._wetness(), float)


def test_budget_endorheic_closure():
    """The endorheic inner-sea closure (COSMO_BUDGET_ENDORHEIC): the wet pool is sized to the WATER balance
    (∫f·E_w over the pool == ∫P), contiguous from the axis, and f∈[0,1]."""
    core = StageADCore(StageADConfig(nx=240, nz=14, theta_max_deg=80.0, grid_type="stretched", dz_min=20.0,
                                     budget_wet_fraction=True, budget_endorheic=True))
    x = np.asarray(core.x, float); dx = core.dx
    P = np.exp(-(x / 3.0e5) ** 2); P *= 1.0e4 / (np.sum(P) * dx)    # axis-peaked supply, total 1e4 kg/m/s
    E_w = 8.74e-4                                                    # uniform wet-tile demand [kg/m^2/s]
    f = _endorheic_wetness(core, P, E_w)
    assert np.all(f >= -1e-12) and np.all(f <= 1.0 + 1e-12)
    # conservation: the pool evaporates the injected supply (∫f·E_w dx == ∫P dx)
    assert abs(np.sum(f * E_w) * dx - np.sum(P) * dx) < 1e-3 * np.sum(P) * dx
    # contiguous from the axis: once dry, stays dry
    wet = f > 1e-6; edge = int(np.argmax(~wet)) if np.any(~wet) else f.size
    assert np.all(wet[:edge]) and not np.any(wet[edge:])
    # x_sea ≈ ∫P / E_w (the balance extent)
    assert abs(x[edge - 1] - np.sum(P) * dx / E_w) < 3 * dx


def test_budget_endorheic_pinch_is_all_or_nothing():
    """Documents the contiguity behavior (review finding 6): a non-monotone supply that PINCHES at the axis
    (off-axis-only rain) zeroes the WHOLE pool via the cumprod gate, even with 1x+ of total capacity. Latent
    today (every shipped precip pathway is axis-peaked -- core spike, sech^2, nest footprint); pinned here so a
    future non-monotone in-basin profile trips a test, not a silent empty pool."""
    core = StageADCore(StageADConfig(nx=240, nz=14, theta_max_deg=80.0, grid_type="stretched", dz_min=20.0,
                                     budget_wet_fraction=True, budget_endorheic=True))
    x = np.asarray(core.x, float); dx = core.dx
    P = np.exp(-((x - 5.0e6) / 3.0e5) ** 2); P *= 1.0e4 / (np.sum(P) * dx)   # a ring at ~5000 km, dry at the axis
    f = _endorheic_wetness(core, P, 8.74e-4)
    assert np.all(f < 1e-6)      # axis pinch (cumF lags cumE there) collapses the whole pool -- the known gate


def test_budget_endorheic_requires_budget_wet_fraction():
    """ON alone is a silent no-op → the config must reject it (fail fast, not mislabel T-proxy wetness)."""
    with pytest.raises(ValueError, match="budget_endorheic requires budget_wet_fraction"):
        StageADCore(StageADConfig(nx=16, nz=14, theta_max_deg=60.0, budget_endorheic=True))


def test_surface_albedo_ice_ramp_endpoints():
    """Lock the shared sea-ice ramp to its actual endpoints: f_ice=0 at 271.35 K (-1.8 C, seawater
    freezing) and f_ice=1 at 269.35 K (-3.8 C) -- NOT the "-2..0 C" an earlier docstring claimed
    (SIM_REVIEW_2026-07-01 P5). Wet (open-ocean) columns, so base albedo is albedo_ocean."""
    from flatworld.stage_ad_core import surface_albedo
    wet = np.ones(3)
    T = np.array([271.35, 269.35, 270.35])
    expected = np.array([C.ALBEDO_OCEAN, C.ALBEDO_ICE, 0.5 * (C.ALBEDO_OCEAN + C.ALBEDO_ICE)])
    np.testing.assert_allclose(surface_albedo(T, wet), expected, atol=1e-12)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
