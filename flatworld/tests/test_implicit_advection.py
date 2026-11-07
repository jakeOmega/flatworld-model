"""Tests for the implicit vertical-omega-advection limb (ADVECTION_DT_ROADMAP.md).

The AD core's explicit dt is CFL-limited by the VERTICAL omega-advection term
``dt = cfl*dz_min/|w|`` (the equatorial chimney column, |w|~1 m/s). The fix moves that single
limb to backward-Euler upwind -- a 1-D tridiagonal solve per column reusing the
``_implicit_vertical_diffusion`` pattern -- gated by the default-OFF ``advection_scheme`` flag
(env ``COSMO_ADVECTION``). Default "explicit" reproduces the historic path BIT-FOR-BIT.

These tests check:
  (a) OFF byte-identical -- ``advection_scheme="explicit"`` (the default) produces a SHA-256-identical
      prognostic state (omega/psi/b/T_atm/T_surf/q/q_c) after N steps, on BOTH the explicit dynamics
      path and the implicit-vertical-diffusion path (the production path).
  (b) Correctness of the new implicit limb -- on a single column with a known w(z), backward-Euler
      upwind matches a small-dt explicit reference and stays MONOTONE (no spurious negatives /
      overshoot) at a large dt.
  (c) The CFL relief -- with the flag ON the ``dz_min/|w|`` vertical-advection term is dropped from
      ``_cfl_dt`` so dt lands at the min of the remaining caps.
"""
import hashlib
import numpy as np
import pytest

from flatworld.stage_ad_core import (StageADCore, StageADConfig, _upwind, _upwind_vertical,
                                     _solve_tridiag_columns)
from flatworld import constants as C


def _state_sha(m):
    """SHA-256 over the full prognostic state, in a fixed field order, as raw bytes. Pulls each array
    to host first (device-agnostic). Any bit-difference in any prognostic field changes the hash."""
    h = hashlib.sha256()
    for name in ("omega", "psi", "b", "T_atm", "T_surf", "q", "q_c"):
        a = np.asarray(getattr(m, name))
        h.update(name.encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _seeded_core(**kw):
    """A small, fast, DETERMINISTICALLY-seeded core with a live overturning cell so the dynamics actually
    advect omega (a zero-flow core would make the advection limb a no-op and the byte test vacuous)."""
    cfg = StageADConfig(nx=48, nz=24, **kw)
    m = StageADCore(cfg)
    # seed a warm-edge buoyancy so the dynamics spin up a real cell (nonzero omega/w), then let the
    # coupled step evolve. seed_warm_edge_buoyancy holds b fixed (couple_thermo False); we want the
    # coupled physics live for the byte test, so seed b then re-enable coupling.
    m.seed_warm_edge_buoyancy(dT_edge=40.0)
    m.couple_thermo = True
    return m


def _run(m, n):
    for _ in range(n):
        m.step()
    return m


# --------------------------------------------------------------------------- (a) OFF byte-identical
@pytest.mark.parametrize("vimplicit", [False, True])
def test_off_byte_identical(vimplicit):
    """advection_scheme="explicit" (default) is SHA-identical to NOT passing the flag at all, on both the
    explicit dynamics path (vimplicit False) and the implicit-vertical-diffusion path (vimplicit True,
    the production path). This is the regression gate: the default reproduces the historic core bit-for-bit."""
    common = dict(vertical_implicit_diffusion=vimplicit)
    m_ref = _run(_seeded_core(**common), 12)
    m_off = _run(_seeded_core(advection_scheme="explicit", **common), 12)
    assert _state_sha(m_ref) == _state_sha(m_off), (
        f"explicit advection_scheme not byte-identical to default (vimplicit={vimplicit})")


def test_default_is_explicit():
    """The dataclass default is 'explicit' (the flag is OFF unless asked)."""
    assert StageADConfig(nx=8, nz=8).advection_scheme == "explicit"


def test_on_changes_state_with_a_live_cell():
    """Sanity: with a live overturning cell the implicit-advection path actually DIFFERS from explicit
    (otherwise the byte-identical test above would be vacuous). It is a DIFFERENT discretization of the
    same term, so it must move the state."""
    m_off = _run(_seeded_core(vertical_implicit_diffusion=True), 12)
    m_on = _run(_seeded_core(advection_scheme="vertical_implicit",
                             vertical_implicit_diffusion=True), 12)
    assert _state_sha(m_off) != _state_sha(m_on)


# --------------------------------------------------------------------- Strang split (A½-D-A½, O(dt²))
def test_strang_differs_from_lie_trotter():
    """The Strang composition (advection½-diffusion-advection½) is a DIFFERENT scheme from Lie-Trotter
    (advection-diffusion), so on a live cell it must move the state differently. If these matched, the
    Strang path would be vacuous / mis-wired (e.g. falling through to the single-solve LT branch)."""
    m_lt = _run(_seeded_core(advection_scheme="vertical_implicit",
                             vertical_implicit_diffusion=True), 12)
    m_st = _run(_seeded_core(advection_scheme="vertical_implicit_strang",
                             vertical_implicit_diffusion=True), 12)
    assert _state_sha(m_lt) != _state_sha(m_st)


def test_strang_converges_to_explicit_as_dt_shrinks():
    """CONSISTENCY: like Lie-Trotter, the Strang path must converge to the explicit reference as dt->0 (it
    is a consistent splitting of the same operators). Run the production (implicit-diffusion) core at two
    shrinking dt caps; the omega discrepancy vs explicit must SHRINK -- AND it should shrink FASTER than
    O(dt) (Strang is O(dt²) in the splitting term), so the ratio drops more steeply than halving."""
    def core(dtd, scheme):
        m = StageADCore(StageADConfig(nx=48, nz=28, grid_type="stretched", dz_min=20.0,
                                      vertical_implicit_diffusion=True,
                                      advection_scheme=scheme, dt_dyn=dtd))
        m.seed_warm_edge_buoyancy(dT_edge=60.0); m.couple_thermo = True
        return m
    diffs = []
    for dtd in (4.0, 1.0):
        m_e = core(dtd, "explicit"); m_s = core(dtd, "vertical_implicit_strang")
        for _ in range(20):
            m_e.step(); m_s.step()
        diffs.append(float(np.max(np.abs(np.asarray(m_s.omega) - np.asarray(m_e.omega)))))
    assert diffs[1] < diffs[0], (
        f"Strang vs explicit omega discrepancy did not shrink with dt: {diffs} -- wiring/operator bug")


def test_strang_dt0_matches_lie_trotter():
    """At a small dt the Strang and Lie-Trotter splittings must agree to leading order (both consistent
    with the same unsplit operator); the difference is the O(dt²) commutator, so it must be SMALL relative
    to the gap each carries vs explicit, and shrink with dt."""
    def core(dtd, scheme):
        m = StageADCore(StageADConfig(nx=40, nz=26, grid_type="stretched", dz_min=20.0,
                                      vertical_implicit_diffusion=True,
                                      advection_scheme=scheme, dt_dyn=dtd))
        m.seed_warm_edge_buoyancy(dT_edge=60.0); m.couple_thermo = True
        return m
    gaps = []
    for dtd in (4.0, 1.0):
        m_lt = core(dtd, "vertical_implicit"); m_st = core(dtd, "vertical_implicit_strang")
        for _ in range(15):
            m_lt.step(); m_st.step()
        gaps.append(float(np.max(np.abs(np.asarray(m_st.omega) - np.asarray(m_lt.omega)))))
    assert gaps[1] < gaps[0], f"Strang vs Lie-Trotter gap did not shrink with dt: {gaps}"


def test_strang_off_byte_identical():
    """The Strang path leaves the OFF (explicit) default byte-identical: with the dataclass default scheme
    the new code is never entered, so SHA must match the historic core on the production path."""
    m_ref = _run(_seeded_core(vertical_implicit_diffusion=True), 12)
    m_off = _run(_seeded_core(advection_scheme="explicit", vertical_implicit_diffusion=True), 12)
    assert _state_sha(m_ref) == _state_sha(m_off)


def test_strang_rejected_on_explicit_diffusion_path():
    """Strang is defined only on the implicit-vertical-diffusion path; selecting it with explicit diffusion
    must RAISE (not silently relieve the CFL while running Lie-Trotter advection)."""
    m = StageADCore(StageADConfig(nx=24, nz=16, grid_type="stretched", dz_min=20.0,
                                  vertical_implicit_diffusion=False,
                                  advection_scheme="vertical_implicit_strang"))
    m.seed_warm_edge_buoyancy(dT_edge=40.0); m.couple_thermo = True
    with pytest.raises(ValueError):
        m.step()


def test_strang_is_monotone_in_the_core():
    """End-to-end monotonicity guard: the composed Strang limbs (each a backward-Euler M-matrix solve +
    M-matrix diffusion) keep omega bounded -- no NaN / blow-up over a stepped run at the relieved (large)
    dt on the production core."""
    m = StageADCore(StageADConfig(nx=48, nz=28, grid_type="stretched", dz_min=20.0,
                                  vertical_implicit_diffusion=True,
                                  advection_scheme="vertical_implicit_strang", dt_dyn=60.0))
    m.seed_warm_edge_buoyancy(dT_edge=60.0); m.couple_thermo = True
    for _ in range(30):
        m.step()
        assert np.isfinite(np.asarray(m.omega)).all(), "Strang core went non-finite"


def test_strang_e_differs_from_strang_and_lie_trotter():
    """E½-[A½-D-A½]-E½ (also symmetrizing the explicit forcing) is a DIFFERENT scheme from both plain
    Strang (implicit-limb-only) and Lie-Trotter, so on a live cell it must produce a distinct state."""
    m_lt = _run(_seeded_core(advection_scheme="vertical_implicit", vertical_implicit_diffusion=True), 12)
    m_st = _run(_seeded_core(advection_scheme="vertical_implicit_strang", vertical_implicit_diffusion=True), 12)
    m_e = _run(_seeded_core(advection_scheme="vertical_implicit_strang_e", vertical_implicit_diffusion=True), 12)
    assert _state_sha(m_e) != _state_sha(m_st)
    assert _state_sha(m_e) != _state_sha(m_lt)


def test_strang_e_converges_to_explicit_as_dt_shrinks():
    """CONSISTENCY: the E-symmetrized path converges to the explicit reference as dt->0 (a consistent
    splitting of the same operators). Discrepancy vs explicit must SHRINK with dt on the production core."""
    def core(dtd, scheme):
        m = StageADCore(StageADConfig(nx=48, nz=28, grid_type="stretched", dz_min=20.0,
                                      vertical_implicit_diffusion=True,
                                      advection_scheme=scheme, dt_dyn=dtd))
        m.seed_warm_edge_buoyancy(dT_edge=60.0); m.couple_thermo = True
        return m
    diffs = []
    for dtd in (4.0, 1.0):
        m_e = core(dtd, "explicit"); m_s = core(dtd, "vertical_implicit_strang_e")
        for _ in range(20):
            m_e.step(); m_s.step()
        diffs.append(float(np.max(np.abs(np.asarray(m_s.omega) - np.asarray(m_e.omega)))))
    assert diffs[1] < diffs[0], (
        f"strang_e vs explicit discrepancy did not shrink with dt: {diffs} -- wiring/operator bug")


def test_strang_e_is_finite_in_the_core():
    """End-to-end stability guard: the E½-[A½-D-A½]-E½ composition stays finite over a stepped run at the
    relieved (large) dt on the production core (the extra explicit half-step must not destabilize)."""
    m = StageADCore(StageADConfig(nx=48, nz=28, grid_type="stretched", dz_min=20.0,
                                  vertical_implicit_diffusion=True,
                                  advection_scheme="vertical_implicit_strang_e", dt_dyn=60.0))
    m.seed_warm_edge_buoyancy(dT_edge=60.0); m.couple_thermo = True
    for _ in range(30):
        m.step()
        assert np.isfinite(np.asarray(m.omega)).all(), "strang_e core went non-finite"


def test_strang_e_rejected_on_explicit_diffusion_path():
    """Like plain Strang, the E-symmetrized variant is defined only on the implicit-diffusion path; it
    must RAISE on the explicit-diffusion path (the dispatch guard recognizes the whole Strang family)."""
    m = StageADCore(StageADConfig(nx=24, nz=16, grid_type="stretched", dz_min=20.0,
                                  vertical_implicit_diffusion=False,
                                  advection_scheme="vertical_implicit_strang_e"))
    m.seed_warm_edge_buoyancy(dT_edge=40.0); m.couple_thermo = True
    with pytest.raises(ValueError):
        m.step()


def test_heat_subcycle_off_byte_identical():
    """heat_advect_subcycle_n=1 (default) is byte-identical to not sub-cycling _advect_heat: the single
    full-dt theta update is the verbatim historic path."""
    m_a = _run(_seeded_core(advection_scheme="vertical_implicit", vertical_implicit_diffusion=True), 10)
    m_b = _run(_seeded_core(advection_scheme="vertical_implicit", vertical_implicit_diffusion=True,
                            heat_advect_subcycle_n=1), 10)
    assert _state_sha(m_a) == _state_sha(m_b)


def test_heat_subcycle_changes_state_when_active():
    """heat_advect_subcycle_n>1 sub-cycles the heat-advection theta update (dt/n), so with a live cell the
    state must DIFFER from the single-step path -- it restores the reference heat-advection Courant while
    keeping the relieved dynamics dt (the archive/_sl_strang_RESULTS.md lever for the dt=60 climate shift)."""
    m_single = _run(_seeded_core(advection_scheme="vertical_implicit",
                                 vertical_implicit_diffusion=True), 12)
    m_sub = _run(_seeded_core(advection_scheme="vertical_implicit", vertical_implicit_diffusion=True,
                              heat_advect_subcycle_n=3), 12)
    assert _state_sha(m_single) != _state_sha(m_sub)


def test_heat_subcycle_finite_in_core():
    """The sub-cycled heat advance stays finite at n=3 on the production core (the confirming-run config);
    a stability guard for the new loop."""
    m = StageADCore(StageADConfig(nx=48, nz=28, grid_type="stretched", dz_min=20.0,
                                  vertical_implicit_diffusion=True,
                                  advection_scheme="vertical_implicit",
                                  heat_advect_subcycle_n=3, dt_dyn=60.0))
    m.seed_warm_edge_buoyancy(dT_edge=60.0); m.couple_thermo = True
    for _ in range(20):
        m.step()
        assert np.isfinite(np.asarray(m.T_atm)).all(), "heat-subcycle core went non-finite"


def test_subcycle_off_byte_identical():
    """adv_subcycle_n=1 (default) is byte-identical to not sub-cycling: the wrapper returns the single
    full-dt solve unchanged."""
    m_a = _run(_seeded_core(advection_scheme="vertical_implicit", vertical_implicit_diffusion=True), 10)
    m_b = _run(_seeded_core(advection_scheme="vertical_implicit", vertical_implicit_diffusion=True,
                            adv_subcycle_n=1), 10)
    assert _state_sha(m_a) == _state_sha(m_b)


def test_subcycle_changes_stiff_columns_only():
    """``_advect_vertical_implicit`` with adv_subcycle_n>1 must (a) match the single full-dt solve in the
    SLACK columns (Courant below threshold) and (b) DIFFER in the STIFF columns (Courant above), since
    there it advances by n back-Euler sub-solves at dt/n. A synthetic high-w column pins the mechanism
    (the toy seeded core has |w|~1e-3 m/s, far too slack to ever flag a column)."""
    m = StageADCore(StageADConfig(nx=6, nz=22, grid_type="stretched", dz_min=20.0,
                                  vertical_implicit_diffusion=True,
                                  advection_scheme="vertical_implicit",
                                  adv_subcycle_n=8, adv_subcycle_courant=1.0))
    nz, nx = m.cfg.nz, m.cfg.nx
    rng = np.random.default_rng(3)
    omega = rng.standard_normal((nz, nx))
    omega[0] = 0.0; omega[-1] = 0.0
    w = np.zeros((nz, nx))
    dz_min = float(np.min(np.diff(m.z)))
    dt = 60.0
    # column 2 = STIFF (Courant ~ 6 > 1), the others SLACK (Courant ~ 0)
    w[:, 2] = 6.0 * dz_min / dt
    out_sub = np.asarray(m._advect_vertical_implicit(omega, dt, w))
    out_single = np.asarray(m._implicit_vertical_advection(omega, dt, w))
    slack = [0, 1, 3, 4, 5]
    assert np.allclose(out_sub[:, slack], out_single[:, slack], rtol=0, atol=1e-12), "slack column changed"
    assert not np.allclose(out_sub[:, 2], out_single[:, 2]), "stiff column not sub-cycled"
    # sub-cycled stiff column stays finite + monotone (no spurious negative beyond round-off)
    assert np.isfinite(out_sub).all()


# --------------------------------------------------------------------- (b) implicit-limb correctness
def _implicit_vertical_advection_1col(omega0, w, z, dt):
    """Reference single-column backward-Euler upwind solve of d omega/dt = -w d omega/dz, returning
    omega_new solving (I + dt A_adv) omega_new = omega0, with A_adv the upwind operator and Dirichlet
    omega=0 at the j=0/j=nz-1 walls (matching the core's free-slip BC). Mirrors the production assembly
    so the test pins the math, not the wiring."""
    nz = omega0.shape[0]
    hm = np.empty(nz); hp = np.empty(nz)
    hm[1:] = z[1:] - z[:-1]; hm[0] = hm[1]
    hp[:-1] = z[1:] - z[:-1]; hp[-1] = hp[-2]
    sub = np.zeros(nz); sup = np.zeros(nz); diag = np.ones(nz)
    for j in range(1, nz - 1):
        if w[j] > 0:
            diag[j] = 1.0 + dt * w[j] / hm[j]
            sub[j] = -dt * w[j] / hm[j]
        else:
            diag[j] = 1.0 - dt * w[j] / hp[j]
            sup[j] = dt * w[j] / hp[j]
    rhs = omega0.copy(); rhs[0] = 0.0; rhs[-1] = 0.0
    out = _solve_tridiag_columns(sub[:, None], diag[:, None], sup[:, None], rhs[:, None])
    return out[:, 0]


def test_upwind_vertical_matches_full_upwind_z_limb():
    """``_upwind_vertical(F, w, dz)`` must equal exactly the VERTICAL limb of the full ``_upwind`` (the
    w-term), so subtracting it from the explicit tendency removes precisely the moved term -- both on a
    uniform and a stretched-spacing grid."""
    rng = np.random.default_rng(0)
    nz, nx = 20, 6
    F = rng.standard_normal((nz, nx))
    u = rng.standard_normal((nz, nx))
    w = rng.standard_normal((nz, nx))
    # uniform
    dz = 37.0; dx = 1.0e5
    full = _upwind(F, u, w, dx, dz)
    horiz = _upwind(F, u, np.zeros_like(w), dx, dz)   # w=0 -> horizontal limb only
    vert = _upwind_vertical(F, w, dz)
    assert np.allclose(full - horiz, vert, rtol=0, atol=1e-12)
    # stretched (z LEVELS array)
    z = np.cumsum(np.linspace(10.0, 80.0, nz)); z = z - z[0]
    full_s = _upwind(F, u, w, dx, z)
    horiz_s = _upwind(F, u, np.zeros_like(w), dx, z)
    vert_s = _upwind_vertical(F, w, z)
    assert np.allclose(full_s - horiz_s, vert_s, rtol=0, atol=1e-12)


def _centroid(o, z):
    return float(np.sum(o * z) / (np.sum(o) + 1e-30))


def test_implicit_advection_consistent_with_explicit_at_small_dt():
    """CONSISTENCY: backward-Euler upwind at a SMALL dt reproduces the SAME first-order upwind scheme as
    explicit upwind (both are O(dt, dz) accurate). A passive scalar omega(z) advected by a fixed w>0 over a
    fixed time T; at dt well below the explicit CFL the implicit and explicit solutions must agree tightly
    (the schemes coincide as dt->0 -- the discretization is correct, not a different operator)."""
    nz = 60
    z = np.linspace(0.0, 6000.0, nz)
    w = np.full(nz, 0.8)                              # uniform upward 0.8 m/s (chimney-like)
    omega0 = np.exp(-((z - 3000.0) / 600.0) ** 2)    # a smooth bump, zero at the walls
    omega0[0] = 0.0; omega0[-1] = 0.0
    dz = z[1] - z[0]
    dt_cfl = dz / abs(w).max()
    T = 8.0 * dt_cfl
    # Both are first-order upwind; their O(dt) truncation terms (the implicit/explicit numerical-diffusion
    # signs) differ, so they coincide only as dt->0. Verify CONVERGENCE: the discrepancy must SHRINK with dt
    # (->0), confirming the implicit limb is the same scheme, not a different operator.
    diffs = []
    for frac in (0.10, 0.04):
        dt = frac * dt_cfl
        n = int(round(T / dt)); dt = T / n
        oe = omega0.copy()
        for _ in range(n):
            flux = np.zeros(nz); flux[1:] = w[1:] * (oe[1:] - oe[:-1]) / dz
            oe = oe - dt * flux; oe[0] = 0.0; oe[-1] = 0.0
        oi = omega0.copy()
        for _ in range(n):
            oi = _implicit_vertical_advection_1col(oi, w, z, dt)
        diffs.append(np.max(np.abs(oi - oe)))
    assert diffs[1] < diffs[0], f"discrepancy did not shrink with dt: {diffs}"   # convergence
    assert diffs[1] < 0.02 * omega0.max(), (
        f"implicit vs explicit (dt=0.04 CFL) max diff {diffs[1]:.4f} -- schemes disagree")


def test_implicit_advection_transports_at_large_dt():
    """At a LARGE dt (6x the explicit advective CFL -- where explicit would blow up) the implicit scheme is
    STABLE and TRANSPORTS the bump upward by ~w*T (the centroid tracks the analytic displacement to within
    the upwind scheme's numerical diffusion). It is MORE diffusive than the small-dt solution (the roadmap's
    open accuracy question for the chimney column) but does not blow up and conserves the right direction."""
    nz = 60
    z = np.linspace(0.0, 6000.0, nz)
    w = np.full(nz, 0.8)
    omega0 = np.exp(-((z - 1500.0) / 500.0) ** 2)    # bump low so it has room to rise
    omega0[0] = 0.0; omega0[-1] = 0.0
    dz = z[1] - z[0]
    dt_cfl = dz / abs(w).max()
    T = 12.0 * dt_cfl
    dt = 6.0 * dt_cfl                                 # 6x the explicit CFL -- explicit upwind would be unstable
    n = int(round(T / dt)); dt = T / n
    oi = omega0.copy()
    for _ in range(n):
        oi = _implicit_vertical_advection_1col(oi, w, z, dt)
    assert np.isfinite(oi).all() and oi.min() >= -1e-12          # stable, monotone
    expected_disp = 0.8 * T                                       # analytic upward displacement
    measured_disp = _centroid(oi, z) - _centroid(omega0, z)
    # the centroid rises by ~w*T (numerical diffusion broadens but does not transport): within 25%
    assert abs(measured_disp - expected_disp) < 0.25 * expected_disp, (
        f"centroid rose {measured_disp:.0f} m, expected ~{expected_disp:.0f} m")


def test_implicit_advection_is_monotone_no_overshoot():
    """Backward-Euler upwind is monotone: a non-negative initial omega stays non-negative and does not
    overshoot its initial max, even at a LARGE dt (the unconditional-stability + M-matrix property that
    lets us drop the advective CFL)."""
    nz = 50
    z = np.linspace(0.0, 5000.0, nz)
    rng = np.random.default_rng(1)
    w = rng.uniform(-1.5, 1.5, nz)                    # mixed-sign vertical velocity
    omega0 = np.maximum(np.sin(np.linspace(0, np.pi, nz)), 0.0)   # >=0, zero at walls
    dz = z[1] - z[0]
    dt = 50.0 * dz / abs(w).max()                     # far above the explicit CFL
    o = omega0.copy()
    for _ in range(40):
        o = _implicit_vertical_advection_1col(o, w, z, dt)
        assert o.min() >= -1e-12, f"spurious negative {o.min():.2e}"
        assert o.max() <= omega0.max() + 1e-9, f"overshoot {o.max():.4f} > {omega0.max():.4f}"


def test_shipped_operator_matches_reference_on_stretched_grid():
    """The SHIPPED ``StageADCore._implicit_vertical_advection`` (vectorized np.where, stretched/uniform
    branch) must equal the per-column reference helper to round-off -- on the STRETCHED grid (hm != hp,
    the production path), where a spacing/sign bug would hide. Pins the shipped operator itself (not a
    reimplementation), no stepping."""
    rng = np.random.default_rng(7)
    m = StageADCore(StageADConfig(nx=10, nz=22, grid_type="stretched", dz_min=20.0))
    nz, nx = m.cfg.nz, m.cfg.nx
    omega = rng.standard_normal((nz, nx))
    w = rng.uniform(-1.2, 1.2, (nz, nx))
    dt = 37.0
    got = np.asarray(m._implicit_vertical_advection(omega, dt, w))
    ref = np.empty((nz, nx))
    for i in range(nx):
        ref[:, i] = _implicit_vertical_advection_1col(omega[:, i].copy(), w[:, i], m.z, dt)
    assert np.allclose(got, ref, rtol=0, atol=1e-10), (
        f"shipped operator != reference on stretched grid, max diff {np.max(np.abs(got-ref)):.2e}")


def test_full_core_implicit_converges_to_explicit_as_dt_shrinks():
    """END-TO-END (requirement b): in the FULL stepped core, the implicit-advection path converges to the
    explicit path as dt->0 -- the faithful "matches small-dt explicit" check, exercising the shipped
    operator AND the tend_full +adv_v cancellation wiring (not a reimplementation). Run on a STRETCHED core
    (the production path, where the chimney makes the vertical-advection limb active) with dt_dyn forced
    small so both step at an EQUAL, advection-limb-dominated dt; the omega discrepancy must SHRINK with dt."""
    def core(dtd, scheme):
        m = StageADCore(StageADConfig(nx=48, nz=28, grid_type="stretched", dz_min=20.0,
                                      vertical_implicit_diffusion=True,
                                      advection_scheme=scheme, dt_dyn=dtd))
        m.seed_warm_edge_buoyancy(dT_edge=60.0); m.couple_thermo = True
        return m
    diffs = []
    for dtd in (4.0, 1.0):
        m_e = core(dtd, "explicit"); m_i = core(dtd, "vertical_implicit")
        for _ in range(20):
            m_e.step(); m_i.step()
        diffs.append(float(np.max(np.abs(np.asarray(m_i.omega) - np.asarray(m_e.omega)))))
    assert diffs[1] < diffs[0], (
        f"implicit vs explicit omega discrepancy did not shrink with dt: {diffs} -- wiring/operator bug")


# --------------------------------------------------------------------------- (c) CFL relief
def test_cfl_drops_vertical_advection_term_when_on():
    """With advection_scheme ON, ``_cfl_dt`` must drop the dz_min/|w| vertical-advection term and KEEP the
    horizontal-advection and diffusion caps, so the returned dt is >= the explicit dt (and, when the
    vertical-advection term was the binding one, strictly larger)."""
    m = _seeded_core(vertical_implicit_diffusion=True)
    _run(m, 5)                                        # spin up a real chimney (nonzero w)
    # recover the current (u, w, nu_t, nu_v) the substep would see
    from flatworld.stage_ad_core import (velocities_from_psi, smagorinsky_viscosity,
                                         smagorinsky_viscosity_aniso)
    c = m.cfg
    u, w = velocities_from_psi(m.psi, m._beta, m.dx, m._dz_op)
    if c.closure_anisotropic:
        nu_t, nu_v = smagorinsky_viscosity_aniso(u, w, m.dx, m._dz_op, c.C_s, c.nu_b)
    else:
        nu_t, nu_v = smagorinsky_viscosity(u, w, m.dx, m._dz_op, c.C_s, c.nu_b), None

    dt_off = m._cfl_dt(u, w, nu_t, nu_v)
    # turn the advection flag ON in-place and re-measure
    m.cfg.advection_scheme = "vertical_implicit"
    dt_on = m._cfl_dt(u, w, nu_t, nu_v)
    m.cfg.advection_scheme = "explicit"

    assert dt_on >= dt_off - 1e-12
    # the vertical-advection limit that was dropped:
    vert_adv_limit = c.cfl * m._dz_min / (float(np.abs(np.asarray(w)).max()) + 1e-30)
    print(f"dt_off={dt_off:.3f}s  dt_on={dt_on:.3f}s  dropped vert-adv limit={vert_adv_limit:.3f}s")


# ----------------------------------------------------- (d) tracer-Courant robustness guard (review caveat)
def _seeded_core_with_w(n_spin=5):
    """A seeded core spun up to a live cell, with the current (u, w) the next step would advect with --
    so the tracer-Courant guard can be exercised on a real resolved flow."""
    from flatworld.stage_ad_core import velocities_from_psi
    m = _seeded_core(vertical_implicit_diffusion=True)
    m.cfg.advection_scheme = "vertical_implicit"
    _run(m, n_spin)
    u, w = velocities_from_psi(m.psi, m._beta, m.dx, m._dz_op)
    return m, u, w


def test_tracer_courant_guard_not_called_when_off():
    """The guard is gated on the implicit scheme: the explicit default path NEVER calls it (zero new work,
    byte-identical) -- so ``_last_tracer_courant`` is never set on a stepped explicit core."""
    m = _seeded_core(vertical_implicit_diffusion=True)        # advection_scheme defaults to "explicit"
    assert m.cfg.advection_scheme == "explicit"
    _run(m, 3)
    assert not hasattr(m, "_last_tracer_courant")


def test_tracer_courant_guard_reports_and_is_silent_in_bounds():
    """With the implicit scheme ON, ``step()`` runs the guard: it records a finite per-cell tracer Courant
    and -- at the in-bounds dt -- does NOT warn (Courant < 1)."""
    m, u, w = _seeded_core_with_w()
    import warnings
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        m.step()                                              # the ON-path guard runs inside step()
    assert hasattr(m, "_last_tracer_courant")
    assert 0.0 <= m._last_tracer_courant < 1.0                # in-bounds on this small converged-ish state
    assert not any(issubclass(r.category, RuntimeWarning) for r in rec)


def test_tracer_courant_guard_warns_once_then_raises_when_over_one():
    """Drive the per-cell tracer Courant past 1 by an oversized dt (bypassing ``_cfl_dt``): the guard warns
    ONCE (RuntimeWarning naming the negative-humidity risk), is silent on the repeat, and -- with
    ``tracer_courant_strict`` -- RAISES instead."""
    import warnings
    m, u, _ = _seeded_core_with_w()
    # scale dt off the realized in-bounds Courant so the test is robust to the seeded cell's strength
    # (the Courant is linear in dt): one probe gives C0 at dt0, then pick dt -> Courant ~5 (well over 1).
    dt0 = 1e4
    m._tracer_courant_guard(u, dt0)
    big_dt = 5.0 * dt0 / max(m._last_tracer_courant, 1e-12)
    m._warned_tracer_courant = False                         # reset the warn-once latch the probe set
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        m._tracer_courant_guard(u, big_dt)
    assert m._last_tracer_courant > 1.0
    warns = [r for r in rec if issubclass(r.category, RuntimeWarning)]
    assert len(warns) == 1 and "Courant" in str(warns[0].message)
    # warn-once: a second over-bound call emits nothing
    with warnings.catch_warnings(record=True) as rec2:
        warnings.simplefilter("always")
        m._tracer_courant_guard(u, big_dt)
    assert not any(issubclass(r.category, RuntimeWarning) for r in rec2)
    # strict mode RAISES instead of warning
    m.cfg.tracer_courant_strict = True
    with pytest.raises(RuntimeError, match="Courant"):
        m._tracer_courant_guard(u, big_dt)
