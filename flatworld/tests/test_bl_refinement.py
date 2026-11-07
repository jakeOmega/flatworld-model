"""Boundary-layer vertical-refinement (stretched height grid) tests.

These cover the optional BL-refined stretched vertical grid added to the AD core (default OFF /
byte-identical -- see ``flatworld/BL_REFINEMENT_SPEC.md``). They split into:

  * REGRESSION GATE (this file's headline test): a small UNIFORM-grid run reproduces a committed fixture
    (captured on the pre-change tree) to round-off -- the guarantee that adding the stretched-grid branches
    changed nothing on the default (uniform) path. Compared with a tolerance, NOT bit-equality: the fixture
    stores absolute float64 values that depend on the numpy/scipy/BLAS stack, so cross-version bit equality
    is not promised (the in-env operator tests below are the true bit-level guard). Capture by running with
    ``BL_CAPTURE_REGRESSION=1`` once on a known-good tree (writes the fixture and SKIPs); thereafter compares.
  * OPERATOR tests: the non-uniform _ddz / _div_grad / anelastic-Poisson discretizations converge on a
    stretched grid AND reduce byte-for-byte to the uniform expressions when the grid is uniform.
  * GENERATOR tests: the stretched-grid generator is monotone, hits the endpoints exactly, and pins
    ``dz_min``.

Kept in a SEPARATE file from ``test_stage_ad_core.py`` so the BL feature's tests don't collide with the
live grid-convergence fleet edits to that file.
"""
import os
import numpy as np
import pytest

from flatworld.stage_ad_core import (StageADCore, StageADConfig, _stretched_z, _ddz, _ddx, _div_grad,
                                     _div_grad_aniso, build_anelastic_poisson, _mixing_dz,
                                     smagorinsky_viscosity, smagorinsky_viscosity_aniso,
                                     _vertical_eddy_term, _solve_tridiag_columns)
from flatworld import surface_wind as sw

_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "bl_regression_uniform.npz")

# The small, fixed UNIFORM core whose evolved state is the byte-identical regression reference. Fixed
# nx/nz and step count so the fixture is deterministic and cheap. Defaults are otherwise the all-on
# accurate model (clouds/LW-window/etc.), so this exercises the full coupled step on the default path.
_REGRESS_NX, _REGRESS_NZ, _REGRESS_STEPS = 48, 24, 12


# ------------------------------------------------------------------ stretched-grid generator (P0.1)
@pytest.mark.parametrize("z_top,nz,dz_min", [(2.0e4, 36, 20.0), (2.0e4, 30, 50.0), (1.5e4, 54, 10.0)])
def test_stretched_z_monotone_endpoints_and_dz_min(z_top, nz, dz_min):
    z = _stretched_z(z_top, nz, dz_min)
    assert z.shape == (nz,)
    assert z[0] == 0.0                                          # surface exactly at 0
    assert z[-1] == pytest.approx(z_top, rel=1e-12)             # spans exactly to z_top
    dz = np.diff(z)
    assert np.all(dz > 0)                                       # strictly monotone increasing
    assert dz[0] == pytest.approx(dz_min, rel=1e-6)             # lowest spacing pinned to dz_min
    assert np.all(np.diff(dz) > 0)                              # spacing coarsens monotonically with height
    # the refinement is real: lowest spacing is far below the uniform spacing it replaces
    assert dz[0] < 0.5 * (z_top / (nz - 1))


def test_stretched_z_reduces_toward_uniform_as_dz_min_approaches_uniform():
    """When dz_min approaches the uniform spacing z_top/(nz-1) the grid approaches uniform."""
    z_top, nz = 2.0e4, 21
    uniform_dz = z_top / (nz - 1)
    z = _stretched_z(z_top, nz, dz_min=0.999 * uniform_dz)
    dz = np.diff(z)
    assert np.allclose(dz, uniform_dz, rtol=2e-3)               # nearly uniform


def test_stretched_z_rejects_dz_min_above_uniform():
    z_top, nz = 2.0e4, 30
    uniform_dz = z_top / (nz - 1)
    with pytest.raises(ValueError):
        _stretched_z(z_top, nz, dz_min=1.5 * uniform_dz)        # no stretching possible (dz_min too large)


# ------------------------------------------------------------------ non-uniform vertical operators
def _field_on_z(z, fn):
    """Build an (nz, nx) field from a 1-D vertical profile fn(z), constant across the nx columns."""
    nx = 5
    return np.repeat(fn(z)[:, None], nx, axis=1)


def test_ddz_nonuniform_second_order_accurate():
    """On a stretched grid, the unequal-interval 3-point _ddz recovers d/dz of an analytic field, and
    converges at ~2nd order as the grid refines."""
    z_top = 1.0e4
    k = 2 * np.pi / z_top
    f = lambda z: np.sin(k * z)
    df = lambda z: k * np.cos(k * z)
    errs = []
    for nz in (40, 80):
        z = _stretched_z(z_top, nz, dz_min=z_top / (nz - 1) * 0.25)
        F = _field_on_z(z, f)
        d = _ddz(F, z)[:, 0]
        interior = slice(1, -1)
        errs.append(np.max(np.abs(d[interior] - df(z)[interior])) / k)
    assert errs[0] < 0.05                                       # accurate on the coarse stretched grid
    assert errs[0] / errs[1] > 3.3                              # ~2nd-order convergence (halving -> ~1/4)


def test_ddz_uniform_array_branch_matches_scalar():
    """On a UNIFORM grid the array (levels) branch agrees with the scalar branch to round-off (the scalar
    branch itself -- the default path -- is byte-identical by construction)."""
    z = np.linspace(0.0, 1.0e4, 32)
    dz = z[1] - z[0]
    F = _field_on_z(z, lambda z: np.cos(3e-4 * z) + 0.1 * z / z.max())
    assert np.allclose(_ddz(F, z), _ddz(F, dz), rtol=1e-10, atol=1e-12)


def test_div_grad_uniform_array_branch_matches_scalar():
    z = np.linspace(0.0, 1.0e4, 28)
    dz, dx = z[1] - z[0], 5.0e4
    rng = np.random.default_rng(0)
    F = rng.standard_normal((28, 9))
    coef = 1.0 + 0.3 * rng.random((28, 9))
    assert np.allclose(_div_grad(F, coef, dx, z), _div_grad(F, coef, dx, dz), rtol=1e-10, atol=1e-12)


def test_div_grad_aniso_reduces_to_isotropic_on_stretched_grid():
    """With coef_x == coef_z the anisotropic FV vertical arm equals the isotropic one on a stretched grid
    (the existing test_stage_ad_core check is for a uniform grid; this extends it to non-uniform)."""
    z = _stretched_z(1.2e4, 30, dz_min=80.0)
    dx = 4.0e4
    rng = np.random.default_rng(1)
    F = rng.standard_normal((30, 11))
    coef = 0.5 + rng.random((30, 11))
    assert np.allclose(_div_grad_aniso(F, coef, coef, dx, z), _div_grad(F, coef, dx, z), rtol=1e-12,
                       atol=1e-14)


def test_div_grad_laplacian_second_order_on_stretched_grid():
    """div(grad F) with unit coef recovers the analytic Laplacian of a vertical profile on a stretched
    grid and converges as the grid refines (interior, away from the one-sided walls)."""
    z_top = 1.0e4
    k = np.pi / z_top
    f = lambda z: np.sin(k * z)
    lap = lambda z: -(k ** 2) * np.sin(k * z)
    dx = 1.0e9                                                  # huge dx -> x-arm ~ 0 (F constant in x anyway)
    errs = []
    for nz in (40, 80):
        z = _stretched_z(z_top, nz, dz_min=z_top / (nz - 1) * 0.3)
        F = _field_on_z(z, f)
        d = _div_grad(F, 1.0, dx, z)[:, 2]                      # interior column (nx=5 -> cols 1..3 interior)
        interior = slice(2, -2)                                 # skip rows touching the one-sided edges
        scale = (k ** 2)
        errs.append(np.max(np.abs(d[interior] - lap(z)[interior])) / scale)
    assert errs[0] < 0.05
    assert errs[0] / errs[1] > 3.0


# ------------------------------------------------------------------ anelastic Poisson on a stretched grid
def _poisson_invert_recovers(rho_ref, nx, nz, dx, z):
    """Build psi_true (zero on all four walls), form omega = div((1/rho) grad psi) with the SAME _div_grad
    operator the Poisson assembles, solve, and return (psi_recovered, psi_true)."""
    X = (np.arange(nx) / (nx - 1))[None, :]
    Z = (z / z[-1])[:, None]
    psi_true = 50.0 * np.sin(np.pi * X) * np.sin(np.pi * Z)     # 0 on i in {0,nx-1}, j in {0,nz-1}
    beta = (1.0 / np.asarray(rho_ref, float))[:, None]          # (nz,1) -> broadcasts over x in _div_grad
    omega = _div_grad(psi_true, beta, dx, z)                    # interior operator; 0 on the edges
    lu = build_anelastic_poisson(rho_ref, nx, nz, dx, z)
    psi = lu.solve(omega.ravel()).reshape(nz, nx)
    return psi, psi_true


def test_anelastic_poisson_inverts_div_grad_on_stretched_grid_constant_rho():
    """Constant rho: the stretched-grid Poisson inverts _div_grad to machine precision (the cleanest
    self-consistency check of the per-layer face-spacing stencil)."""
    nx, nz, dx = 12, 30, 4.0e4
    z = _stretched_z(1.2e4, nz, dz_min=60.0)
    rho_ref = np.full(nz, 1.0)
    psi, psi_true = _poisson_invert_recovers(rho_ref, nx, nz, dx, z)
    assert np.allclose(psi, psi_true, rtol=1e-9, atol=1e-9)


def test_anelastic_poisson_inverts_div_grad_on_stretched_grid_variable_rho():
    """Variable rho_ref(z) (a realistic exponential atmosphere): the stretched-grid Poisson still inverts
    _div_grad to machine precision, AND the rows adjacent to the Dirichlet walls (j=1, j=nz-2 -- where
    h^-/h^+ are most unequal) are recovered just as well (the spot an FV sign/spacing slip would hide)."""
    nx, nz, dx = 12, 36, 5.0e4
    z = _stretched_z(1.5e4, nz, dz_min=20.0)
    rho_ref = 1.2 * np.exp(-z / 8.0e3)                         # decays with height
    psi, psi_true = _poisson_invert_recovers(rho_ref, nx, nz, dx, z)
    assert np.allclose(psi, psi_true, rtol=1e-8, atol=1e-8)
    for jwall in (1, nz - 2):                                   # the near-wall rows, most unequal spacing
        assert np.allclose(psi[jwall], psi_true[jwall], rtol=1e-8, atol=1e-8)


def test_anelastic_poisson_uniform_array_branch_matches_scalar():
    """On a uniform grid the array-branch assembly inverts the same field as the scalar branch (the scalar
    branch -- the default path -- is byte-identical by construction)."""
    nx, nz, dx = 10, 24, 6.0e4
    z = np.linspace(0.0, 1.0e4, nz)
    dz = z[1] - z[0]
    rho_ref = 1.0 * np.exp(-z / 9.0e3)
    p_arr, pt = _poisson_invert_recovers(rho_ref, nx, nz, dx, z)
    # scalar-branch reference inversion of the SAME omega-from-scalar-operator
    X = (np.arange(nx) / (nx - 1))[None, :]; Z = (z / z[-1])[:, None]
    psi_true = 50.0 * np.sin(np.pi * X) * np.sin(np.pi * Z)
    omega_s = _div_grad(psi_true, (1.0 / rho_ref)[:, None], dx, dz)
    lu_s = build_anelastic_poisson(rho_ref, nx, nz, dx, dz)
    p_scalar = lu_s.solve(omega_s.ravel()).reshape(nz, nx)
    assert np.allclose(p_arr, p_scalar, rtol=1e-9, atol=1e-9)


# ------------------------------------------------------------------ Smagorinsky local-dz mixing length
def test_mixing_dz_scalar_passthrough_and_stretched_cells():
    assert _mixing_dz(123.4) == 123.4                          # scalar -> unchanged (uniform, byte-identical)
    z = _stretched_z(1.2e4, 30, dz_min=20.0)
    ldz = _mixing_dz(z)
    assert ldz.shape == (30, 1)
    assert ldz[0, 0] == pytest.approx(z[1] - z[0])             # finest at the surface ~ dz_min
    assert np.all(np.diff(ldz[:, 0]) > 0)                      # coarsens with height (local cell size)
    assert ldz[0, 0] < 0.1 * ldz[-1, 0]                        # BL cells far smaller than the top


def test_smagorinsky_uniform_array_matches_scalar():
    z = np.linspace(0.0, 1.0e4, 24); dz, dx = z[1] - z[0], 4.0e4
    rng = np.random.default_rng(3)
    u = rng.standard_normal((24, 9)); w = rng.standard_normal((24, 9))
    assert np.allclose(smagorinsky_viscosity(u, w, dx, z, 0.17, 1.0),
                       smagorinsky_viscosity(u, w, dx, dz, 0.17, 1.0), rtol=1e-10, atol=1e-12)
    hz, vz = smagorinsky_viscosity_aniso(u, w, dx, z, 0.17, 1.0)
    hs, vs = smagorinsky_viscosity_aniso(u, w, dx, dz, 0.17, 1.0)
    assert np.allclose(hz, hs, rtol=1e-10, atol=1e-12) and np.allclose(vz, vs, rtol=1e-10, atol=1e-12)


def test_smagorinsky_stretched_uses_small_delta_in_bl():
    """On a stretched grid the eddy-viscosity excess over the floor scales with the LOCAL cell, so the
    fine BL layers get a far smaller nu than the coarse top layers under the same strain (the property
    that stops the refined layers over-smoothing the jet)."""
    z = _stretched_z(1.2e4, 30, dz_min=20.0); dx = 4.0e4
    u = np.ones((30, 9)); w = np.zeros((30, 9))
    # impose a uniform vertical shear du/dz in PHYSICAL terms by setting u linear in z (constant |S|)
    u = (z / z[-1])[:, None] * np.ones((1, 9))
    nu = smagorinsky_viscosity(u, w, dx, z, 0.17, 0.0)         # floor 0 -> pure Smagorinsky part
    # excess over floor in the lowest interior layer << in the top interior layer (smaller Delta)
    assert nu[1, 4] < nu[-2, 4]


# ------------------------------------------------------------------ stretched-core construction + smoke
def test_stretched_core_constructs_and_grids_are_consistent():
    cfg = StageADConfig(nx=24, nz=30, grid_type="stretched", dz_min=20.0, theta_max_deg=60.0)
    m = StageADCore(cfg)
    assert m._stretched is True
    assert m.z.shape == (30,) and m.z[0] == 0.0 and m.z[-1] == pytest.approx(m.z_top)
    assert m.dz == pytest.approx(20.0, rel=1e-6)               # lowest spacing pinned
    assert m._dz_min <= m.dz + 1e-9                            # tightest interval (==dz here, surface-finest)
    assert np.isclose(float(np.sum(m._dz_cell)), m.z_top)      # integration weights sum to z_top
    assert m._lu_dyn is not None                               # the non-uniform Poisson factorized cleanly


def test_stretched_core_pure_dynamics_smoke_stays_finite():
    """End-to-end exercise of the non-uniform dynamics path (Poisson invert -> velocities -> Smagorinsky ->
    CFL -> vorticity tendency): seed a fixed warm-edge buoyancy and run a handful of pure-dynamics steps;
    the state must stay finite and spin up a circulation."""
    cfg = StageADConfig(nx=24, nz=30, grid_type="stretched", dz_min=50.0, theta_max_deg=60.0)
    m = StageADCore(cfg)
    m.seed_warm_edge_buoyancy(dT_edge=8.0)                     # sets couple_thermo=False (pure dynamics)
    for _ in range(12):
        m.step()
    assert m._state_is_finite()
    assert float(np.max(np.abs(m.psi))) > 0.0                  # a circulation developed


def test_stretched_core_coupled_smoke_stays_finite():
    """A few FULL coupled steps (default all-on physics) on a stretched grid stay finite -- exercises the
    column integrals (_vint), thermo, and the resolved transport on the non-uniform grid."""
    cfg = StageADConfig(nx=24, nz=30, grid_type="stretched", dz_min=80.0, theta_max_deg=60.0)
    m = StageADCore(cfg)
    for _ in range(5):
        m.step()
    assert m._state_is_finite()


# ================================================================== P1: surface-flux + 10 m wind
def test_u10_factor_default_unchanged_byte_identical():
    """z1=None reproduces the pre-refinement mixed-layer reduction exactly."""
    z0 = np.array([2e-4, 0.05, 0.5])
    got = sw.u10_factor(z0, h_bl=1000.0, z_ref=10.0)
    z_sl = max(0.10 * 1000.0, 3.0 * 10.0)
    expect = np.log(10.0 / z0) / np.log(z_sl / z0)
    assert np.array_equal(got, expect)                         # bit-for-bit (the default path is untouched)


def test_u10_factor_z1_short_circuits_to_identity_at_screen_level():
    z0 = np.array([2e-4, 0.05, 1.5])                           # incl. forest z0=1.5 (< z_ref, but > nothing)
    assert np.allclose(sw.u10_factor(z0, z_ref=10.0, z1=10.0), 1.0)   # z1 == z_ref -> identity
    assert np.allclose(sw.u10_factor(z0, z_ref=10.0, z1=6.0), 1.0)    # z1 < z_ref -> identity (no extrapolation)
    # short-circuit returns BEFORE the z0<z_ref guard: a forest z0=1.5 with z1<=z_ref must not raise
    sw.u10_factor(np.array([1.5]), z_ref=10.0, z1=8.0)         # (would be fine anyway, but exercises the order)


def test_u10_factor_z1_match_height_tends_to_identity():
    """With z1 given the reduction matches from z1 (not the ~100 m mixed layer), so it is milder and
    -> 1 as z1 -> z_ref."""
    z0 = 2e-4
    f_mixed = sw.u10_factor(z0, z_ref=10.0)                    # from ~100 m
    f_z1_50 = sw.u10_factor(z0, z_ref=10.0, z1=50.0)           # from 50 m
    f_z1_15 = sw.u10_factor(z0, z_ref=10.0, z1=15.0)           # from 15 m
    assert f_mixed < f_z1_50 < f_z1_15 < 1.0                   # closer match height -> larger factor
    assert sw.u10_factor(z0, z_ref=10.0, z1=10.5) > 0.97       # near identity just above z_ref


def test_manufactured_low_level_jet_resolved_by_refinement():
    """Spec validation #3: impose an analytic near-surface jet u(z)=U0*exp(-z/h_j) (wind largest near the
    ground, the resolved return-branch jet). The BL-refined grid's lowest level sits at ~z_ref, so the 10 m
    estimate recovers the true 10 m wind to <few %; the uniform grid's lowest level (~570 m) badly
    under-samples the jet and the log-extrapolated estimate is off by a large margin."""
    U0, h_j, z_top, nz = 40.0, 300.0, 2.0e4, 36
    u_true = lambda z: U0 * np.exp(-z / h_j)
    true_10m = u_true(10.0)
    z0 = 2e-4                                                  # open sea

    z_str = _stretched_z(z_top, nz, dz_min=10.0)              # lowest level AT 10 m
    z1_str = float(z_str[1])
    est_str = float(sw.u10_factor(z0, z_ref=10.0, z1=z1_str)) * u_true(z1_str)

    z_uni = np.linspace(0.0, z_top, nz)                       # lowest level ~ 570 m
    z1_uni = float(z_uni[1])
    est_uni = float(sw.u10_factor(z0, z_ref=10.0)) * u_true(z1_uni)   # mixed-layer log-extrapolation

    assert abs(est_str - true_10m) / true_10m < 0.05          # refined: recovers the 10 m wind to <5 %
    assert abs(est_uni - true_10m) / true_10m > 0.4           # uniform: off by a large margin (jet unresolved)
    assert abs(est_str - true_10m) < abs(est_uni - true_10m)  # refinement strictly closer


def test_bulk_drag_rate_scales_inverse_with_lowest_level():
    """bulk_drag_rate = C_D*U/z1 from the live lowest-level thickness: ~6e-6 1/s on a coarse uniform grid,
    far larger (proportionally) on the BL-refined grid where z1 ~ 10-30 m."""
    m_uni = StageADCore(StageADConfig(nx=24, nz=18, theta_max_deg=70.0))
    r_uni = m_uni.bulk_drag_rate()
    z1_uni = float(m_uni.z[1] - m_uni.z[0])
    assert r_uni == pytest.approx(1.2e-3 * m_uni.cfg.wind / z1_uni)
    assert 1e-6 < r_uni < 2e-5                                 # ~1/(1-2 days) on the ~1.2 km lowest level

    m_str = StageADCore(StageADConfig(nx=24, nz=30, grid_type="stretched", dz_min=20.0, theta_max_deg=70.0))
    r_str = m_str.bulk_drag_rate()
    assert r_str > 10.0 * r_uni                                # z1 ~ 20 m << 1.2 km -> much stronger drag


# ================================================================== P2: implicit vertical diffusion
def test_tridiag_solver_matches_dense():
    """The batched Thomas solver reproduces a dense per-column solve."""
    nz, nx = 8, 4
    rng = np.random.default_rng(7)
    sub = rng.random((nz, nx)); sup = rng.random((nz, nx))
    diag = 4.0 + rng.random((nz, nx)); rhs = rng.standard_normal((nz, nx))
    x = _solve_tridiag_columns(sub, diag, sup, rhs)
    for i in range(nx):
        M = np.diag(diag[:, i]) + np.diag(sub[1:, i], -1) + np.diag(sup[:-1, i], 1)
        assert np.allclose(M @ x[:, i], rhs[:, i])


@pytest.mark.parametrize("grid_type,dz_min", [("uniform", 20.0), ("stretched", 20.0)])
def test_implicit_solver_inverts_Lz(grid_type, dz_min):
    """(I - dt L_z) omega_new = omega_star solved by _implicit_vertical_diffusion must invert the SAME
    vertical operator _vertical_eddy_term applies: take a known field x (0 on walls), form
    rhs = x - dt*L_z(x), solve, and recover x. Verifies on a variable coef_z (nz, nx)."""
    kw = dict(grid_type=grid_type, dz_min=dz_min) if grid_type == "stretched" else {}
    m = StageADCore(StageADConfig(nx=10, nz=24, theta_max_deg=60.0, **kw))
    nz, nx = m.cfg.nz, m.cfg.nx
    rng = np.random.default_rng(2)
    x = rng.standard_normal((nz, nx)); x[0] = 0.0; x[-1] = 0.0   # Dirichlet walls
    coef_z = 50.0 + 30.0 * rng.random((nz, nx))                  # variable vertical eddy viscosity
    dt = 0.5
    Lz_x = _vertical_eddy_term(x, coef_z, m._dz_op)             # interior-x only (x-edges left 0)
    rhs = x - dt * Lz_x
    x_rec = m._implicit_vertical_diffusion(rhs, dt, coef_z)
    # compare on the interior x columns where the two operators coincide (the x-walls are zeroed anyway)
    assert np.allclose(x_rec[:, 1:-1], x[:, 1:-1], rtol=1e-9, atol=1e-9)


def test_implicit_diffusion_relieves_vertical_cfl():
    """With vertical_implicit_diffusion the diffusive CFL drops its vertical term, so the limiting dt is
    >= the explicit one for the same flow (and strictly larger when the vertical diffusive term bound)."""
    # high eddy-viscosity floor + fine dz_min so the VERTICAL diffusive limit (0.5/(nu/dz_min^2)) binds
    # the explicit dt (well below the dt_dyn cap); implicit diffusion then drops it.
    common = dict(nx=24, nz=30, grid_type="stretched", dz_min=10.0, theta_max_deg=60.0, nu_b=200.0)
    m_off = StageADCore(StageADConfig(**common))
    m_on = StageADCore(StageADConfig(vertical_implicit_diffusion=True, **common))
    m_off.seed_warm_edge_buoyancy(dT_edge=8.0)
    u, w = m_off.velocities()
    nu_t = smagorinsky_viscosity(u, w, m_off.dx, m_off._dz_op, m_off.cfg.C_s, m_off.cfg.nu_b)
    dt_off = m_off._cfl_dt(u, w, nu_t, None)
    dt_on = m_on._cfl_dt(u, w, nu_t, None)
    assert dt_on >= dt_off
    assert dt_on > 1.5 * dt_off                                  # the fine-dz vertical diffusive term WAS binding


def test_implicit_off_is_byte_identical_explicit_path():
    """vertical_implicit_diffusion=False takes the explicit substep verbatim: a stretched run with the
    flag OFF reproduces a stretched run constructed without ever setting it (same default path)."""
    cfg_a = StageADConfig(nx=24, nz=30, grid_type="stretched", dz_min=40.0, theta_max_deg=60.0)
    cfg_b = StageADConfig(nx=24, nz=30, grid_type="stretched", dz_min=40.0, theta_max_deg=60.0,
                          vertical_implicit_diffusion=False)
    ma, mb = StageADCore(cfg_a), StageADCore(cfg_b)
    ma.seed_warm_edge_buoyancy(dT_edge=6.0); mb.seed_warm_edge_buoyancy(dT_edge=6.0)
    for _ in range(8):
        ma.step(); mb.step()
    assert np.array_equal(ma.psi, mb.psi) and np.array_equal(ma.omega, mb.omega)


def test_implicit_coupled_smoke_stays_finite():
    """A few FULL coupled steps with implicit vertical diffusion ON stay finite (exercises the implicit
    solve + the dissipative-heating energy-closure path on a stretched grid)."""
    m = StageADCore(StageADConfig(nx=24, nz=30, grid_type="stretched", dz_min=40.0, theta_max_deg=60.0,
                                  vertical_implicit_diffusion=True))
    for _ in range(5):
        m.step()
    assert m._state_is_finite()
    assert float(np.max(np.abs(m.psi))) > 0.0


# ------------------------------------------------------------------ P2: thermo (pressure) co-refinement
def test_thermo_grid_uniform_default_is_linspace():
    """Default thermo_grid_type='uniform' keeps the linspace pressure grid (byte-identical)."""
    from flatworld import constants as C
    m = StageADCore(StageADConfig(nx=12, nz=30, theta_max_deg=60.0))
    assert np.array_equal(m.p_edge, np.linspace(m.cfg.p_top, C.P_SURF, m.cfg.nz + 1))


def test_thermo_grid_stretched_fine_near_surface_and_consistent():
    """thermo_grid_type='stretched' clusters the pressure layers at P_SURF (lowest layer ~ thermo_dp_min,
    far below the uniform spacing) and all derived fields recompute from p_edge."""
    from flatworld import constants as C
    m = StageADCore(StageADConfig(nx=12, nz=30, theta_max_deg=60.0,
                                  thermo_grid_type="stretched", thermo_dp_min=500.0))
    assert m.p_edge[0] == pytest.approx(m.cfg.p_top) and m.p_edge[-1] == pytest.approx(C.P_SURF)
    assert np.all(np.diff(m.p_edge) > 0)                       # monotone top -> surface
    dp = np.diff(m.p_edge)
    uniform_dp = (C.P_SURF - m.cfg.p_top) / m.cfg.nz
    assert dp[-1] == pytest.approx(500.0, rel=1e-3)            # finest layer at the surface ~ thermo_dp_min
    assert dp[-1] < 0.3 * uniform_dp                           # genuinely refined vs uniform
    # derived fields are consistent with the new p_edge
    assert np.allclose(m.p_mid, 0.5 * (m.p_edge[:-1] + m.p_edge[1:]))
    assert np.allclose(m.dm, np.diff(m.p_edge) / C.G_ACCEL)
    assert np.allclose(m.sigma, m.p_mid / C.P_SURF)
    assert m.C_layer.shape == (m.cfg.nz,)


def test_thermo_grid_stretched_coupled_smoke_stays_finite():
    """A few full coupled steps with a co-refined pressure grid stay finite."""
    m = StageADCore(StageADConfig(nx=24, nz=30, theta_max_deg=60.0,
                                  thermo_grid_type="stretched", thermo_dp_min=800.0))
    for _ in range(5):
        m.step()
    assert m._state_is_finite()


def test_full_bl_refinement_both_grids_smoke():
    """Both refinements together (stretched height + stretched pressure + implicit vertical diffusion) --
    the configuration a BL-convergence run would use -- constructs and steps finite."""
    m = StageADCore(StageADConfig(nx=24, nz=30, theta_max_deg=60.0, grid_type="stretched", dz_min=20.0,
                                  vertical_implicit_diffusion=True, thermo_grid_type="stretched",
                                  thermo_dp_min=800.0))
    for _ in range(5):
        m.step()
    assert m._state_is_finite()


# ------------------------------------------------------------------ honest bulk surface friction
def test_surface_drag_bulk_default_on():
    # DEFAULT ON (2026-06-16): honest surface friction is part of the all-on accurate model (like
    # clouds / lw_window). COSMO_SURFACE_DRAG_BULK=0 gives the frictionless byte-identical limit.
    assert StageADConfig().surface_drag_bulk is True
    import os
    os.environ["COSMO_SURFACE_DRAG_BULK"] = "0"
    try:
        assert StageADConfig().surface_drag_bulk is False       # env override -> off (byte-identical)
    finally:
        del os.environ["COSMO_SURFACE_DRAG_BULK"]


def test_surface_drag_curl_is_localized_momentum_sink():
    """_surface_drag_curl is the curl of tau = rho*C_D*|u1|*u1: the lowest-cell momentum tendency
    Du[1] = -(C_D|u1|/z1) u1 opposes the local wind (a drag, quadratic), zero above the surface cell."""
    m = StageADCore(StageADConfig(nx=12, nz=30, theta_max_deg=60.0, grid_type="stretched", dz_min=20.0))
    u = 10.0 * np.ones((m.cfg.nz, m.cfg.nx))                    # uniform +10 m/s test wind
    curl = np.asarray(m._surface_drag_curl(u))
    z1 = m.z[1] - m.z[0]
    Du1 = -(m.cfg.surface_drag_cd * abs(10.0) / z1) * 10.0      # expected lowest-cell momentum sink
    assert Du1 < 0                                             # opposes the +10 m/s wind (a drag)
    # the curl (d/dz of a sink localized at j=1) vanishes in the upper interior
    assert np.allclose(curl[5:-1], 0.0)
    # and flips sign with the wind (quadratic |u|u): -u gives +Du1
    curl_neg = np.asarray(m._surface_drag_curl(-u))
    assert np.allclose(curl_neg, -curl)


def test_surface_drag_bulk_removes_surface_kinetic_energy():
    """The honest bulk drag is a genuine momentum/KE sink (the property the inert omega[1] Rayleigh sink
    lacks): the velocity tendency it induces (curl-of-stress, inverted through the anelastic Poisson) does
    NEGATIVE work on the resolved flow. Fast + deterministic -- impose a strong analytic overturning cell."""
    from flatworld.stage_ad_core import velocities_from_psi
    m = StageADCore(StageADConfig(nx=16, nz=30, theta_max_deg=60.0, grid_type="stretched", dz_min=20.0))
    X = (m.x / m.x.max())[None, :]; Z = (m.z / m.z.max())[:, None]
    m.psi = 6.0e6 * np.sin(np.pi * X) * np.sin(np.pi * Z)       # strong cell -> surface wind ~tens of m/s
    u, w = m.velocities()
    assert np.abs(np.asarray(u)[1, :]).max() > 5.0             # a real wind for the quadratic drag to act on
    curl = np.asarray(m._surface_drag_curl(u))                 # d(omega)/dt from the surface stress
    dpsi = m._lu_dyn.solve(curl.ravel()).reshape(m.cfg.nz, m.cfg.nx)
    du, dw = velocities_from_psi(dpsi, m._beta, m.dx, m._dz_op)
    work = float((np.asarray(m.rho_ref)[:, None] * (np.asarray(u) * np.asarray(du)
                                                    + np.asarray(w) * np.asarray(dw))).sum())
    assert work < 0                                            # drag REMOVES kinetic energy


def test_surface_drag_bulk_energy_returned_as_heat():
    """ENERGY CLOSURE (parity with surface_drag_rate): the KE the honest bulk drag removes is folded into the
    operator KE-removal D_col, so the dissipative-heating deposit returns it (no budget leak). By linearity of
    the anelastic Poisson inversion, turning surface_drag_bulk ON raises D_col by EXACTLY the drag's own removed
    KE -- the curl-of-stress inverted ALONE -- on top of the eddy removal: no leak, no double count."""
    from flatworld.stage_ad_core import velocities_from_psi
    base = dict(nx=16, nz=30, theta_max_deg=60.0, grid_type="stretched", dz_min=20.0,
                dissipative_heating=True, dissipation_form="operator")
    m_off = StageADCore(StageADConfig(surface_drag_bulk=False, **base))
    m_on = StageADCore(StageADConfig(surface_drag_bulk=True, **base))
    for m in (m_off, m_on):                                     # identical strong cell + a nonzero eddy baseline
        X = (m.x / m.x.max())[None, :]; Z = (m.z / m.z.max())[:, None]
        m.psi = 6.0e6 * np.sin(np.pi * X) * np.sin(np.pi * Z)   # -> a real surface wind for the quadratic drag
        m.omega = 1e-3 * np.sin(np.pi * X) * np.sin(np.pi * Z)  # nonzero so the eddy removal D_off != 0
    u, w = m_on.velocities()                                    # same psi -> same (u, w) in both cores
    nu_t = 10.0
    D_off = np.asarray(m_off._operator_ke_removal_col(u, w, nu_t))   # eddy only (bulk OFF)
    D_on = np.asarray(m_on._operator_ke_removal_col(u, w, nu_t))     # eddy + bulk-drag fold
    # the drag curl inverted ALONE -> its own per-column removed-KE [W/m^2] (the independent reference)
    curl = np.asarray(m_on._surface_drag_curl(u))
    dpsi = m_on._lu_dyn.solve(curl.ravel()).reshape(m_on.cfg.nz, m_on.cfg.nx)
    du, dw = velocities_from_psi(dpsi, m_on._beta, m_on.dx, m_on._dz_op)
    drag_removal = np.asarray(m_on._vint(-np.asarray(m_on.rho_ref)[:, None] * (u * du + w * dw)))
    assert np.allclose(D_on - D_off, drag_removal, rtol=1e-6, atol=1e-9)   # the fold == the drag's own KE
    assert (D_on - D_off).sum() > 0                            # net: KE removed -> positive heat returned


def _regress_core():
    # surface_drag_bulk pinned OFF: it became default-ON 2026-06-16 (the all-on accurate model), but this
    # fixture predates that and the guard's job is cross-BLAS reproducibility of the radiation/Poisson
    # operators, NOT the drag term (covered by the three drag tests above). Pinning off keeps the committed
    # frictionless fixture valid; the drag-on default is exercised by the production runs + those tests.
    #
    # FIXTURE PROVENANCE: recaptured 2026-07-17 for PR #340 (cold 50 K far-field IC + esat pole guard),
    # which INTENTIONALLY flipped the fresh-spin-up init-floor defaults (t_init_floor 120->50,
    # t_atm_init_floor 130->50, s_init_floor 1.0->1e-3) -- "byte-different for FRESH spin-ups only" per that
    # commit. This test IS a fresh spin-up, so #340 legitimately invalidated the prior fixture (the PR just
    # missed recapturing it). Verified the only change is the IC default: pinning the OLD floors
    # (120/130/1.0) at HEAD reproduces the pre-#340 fixture BIT-EXACTLY, so the radiation/Poisson operators
    # are untouched. The init floors are left at the (current) defaults so this keeps guarding the true
    # default fresh-spin-up path; recapture on any future INTENTIONAL default-path change.
    cfg = StageADConfig(nx=_REGRESS_NX, nz=_REGRESS_NZ, theta_max_deg=70.0, surface_drag_bulk=False)
    return StageADCore(cfg)


def _evolve_state(m, nsteps):
    for _ in range(nsteps):
        m.step()
    return m.state()


def test_uniform_grid_byte_identical_regression():
    """The default (uniform) path did not drift: a fixed small run reproduces the committed pre-change
    fixture to round-off. Capture with BL_CAPTURE_REGRESSION=1 on a known-good tree.

    TOLERANCE, not bit-equality (PR #135 review): the fixture stores ABSOLUTE float64 values from the
    coupled physics (radiation, the scipy splu Poisson solve), which depend on the numpy/scipy/BLAS stack.
    Comparing a committed .npz with ``array_equal`` therefore tests cross-version bit reproducibility --
    which those libraries do not promise (~1e-13 abs / ~1e-15 rel drift across BLAS builds; psi's near-zero
    points drift up to ~3e-11 ABS) -- NOT the "uniform path unchanged" invariant. The true bit-level guard
    is the in-environment array-branch-vs-scalar-branch operator tests above (``_ddz``/``_div_grad``/Poisson/
    Smagorinsky), plus the main-vs-branch same-interpreter check (maxabsdiff == 0, run at review time). So
    this gate uses ``allclose`` with a tolerance comfortably below any genuine regression (which would be
    orders of magnitude larger) yet above the cross-BLAS rounding floor -- incl. atol covering psi's
    near-zero ~3e-11 (the review's suggested atol=1e-12 would itself trip there)."""
    m = _regress_core()
    st = _evolve_state(m, _REGRESS_STEPS)
    keys = ("T_surf", "T_atm", "omega", "psi", "b", "q")

    if os.environ.get("BL_CAPTURE_REGRESSION"):
        os.makedirs(os.path.dirname(_FIXTURE), exist_ok=True)
        np.savez_compressed(_FIXTURE, **{k: np.asarray(st[k]) for k in keys})
        pytest.skip(f"captured regression fixture -> {_FIXTURE}")

    if not os.path.exists(_FIXTURE):
        pytest.skip(f"no regression fixture at {_FIXTURE}; capture with BL_CAPTURE_REGRESSION=1")

    ref = np.load(_FIXTURE)
    for k in keys:
        got = np.asarray(st[k])
        assert got.shape == ref[k].shape, f"{k}: shape {got.shape} != {ref[k].shape}"
        # rtol catches any real drift on the O(100) T/psi fields; atol covers cross-BLAS rounding on the
        # near-zero points (psi early ~ 0, observed ~3e-11 abs). A genuine regression is >>1e-9 rel.
        assert np.allclose(got, ref[k], rtol=1e-9, atol=1e-9), \
            f"{k}: uniform-grid run drifted from the regression fixture beyond cross-BLAS round-off"
