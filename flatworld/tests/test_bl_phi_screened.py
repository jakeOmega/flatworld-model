"""Roadmap C1: the screened-Poisson boundary-layer geopotential solver + its three gates.

Covers ``flatworld.bl_phi_screened`` -- a NEW elliptic ``(lap - lam^2)`` Phi inverse built so the thermal
and latent forcings ride ONE operator (the prognostic-Phi feedback, Track C). The deliverable gates, per
``ROADMAP_BC_moisture_circulation_feedback.md`` C1:

  * SOLVER correctness: spectral inverse recovers the discrete operator to round-off (both BCs);
    invertible with no null space (the screening term); periodic-e translation invariance.
  * K_0 GATE: a unit point source recovers the analytic 2-D Green's function -(1/2pi) K_0(lam r).
  * 400 km GATE: the screening length lam^-1 = c/r_fric lands at ~400 km with c pinned from PHYSICS
    (the density-current speed), NOT inherited from wind_cap.
  * BOUNDED-SOURCE GATE: source-smoothing makes the point-source peak |Phi| resolution-INDEPENDENT
    (the K_0 log-singularity is de-singularized); without it the peak sharpens with resolution.

Plus the source builders (thermal == bl_forcing's pre-smoothing field; latent default-inert; the latent
sign), the one-footing linearity (parts sum to the whole), and a soft pattern-consistency with the
Gaussian build. The default build is thermal-only -> the shipped Stage-C path is untouched (C1 is the
solver + gates, not yet wired into the loop -- that is C3).
"""
import numpy as np
import pytest
from scipy.special import k0

from flatworld import bl_phi_screened as bps
from flatworld import bl_forcing as blf
from flatworld import constants as C


# ============================================================ solver correctness
@pytest.mark.parametrize("bc_x", ["neumann", "periodic"])
def test_solver_inverts_discrete_operator_to_roundoff(bc_x):
    """The spectral solve inverts the SAME discrete (lap - lam^2) operator ``screened_laplacian`` applies:
    op(solve(src)) == src to round-off. This is the real guard on the DCT/FFT eigenvalue normalization."""
    rng = np.random.default_rng(0)
    nx, ne = 40, 32
    dx, de = 5.0e3, 6.0e3
    lam = bps.screening_rate()
    src = rng.standard_normal((nx, ne))
    Phi = bps.solve_screened_poisson(src, dx, de, lam, bc_x=bc_x)
    res = bps.screened_laplacian(Phi, dx, de, lam, bc_x=bc_x)
    assert np.max(np.abs(res - src)) / src.std() < 1e-9


def test_solver_has_no_null_space_constant_source():
    """The -lam^2 screening removes Poisson's null space: a CONSTANT source inverts to the uniform
    Phi = -src/lam^2 (since lap of a constant is 0). Pure Poisson would be singular here."""
    nx, ne, dx, de = 24, 20, 7.0e3, 8.0e3
    lam = bps.screening_rate()
    src = np.full((nx, ne), 3.0)
    Phi = bps.solve_screened_poisson(src, dx, de, lam, bc_x="neumann")
    assert np.allclose(Phi, -3.0 / lam ** 2, rtol=1e-9)


def test_periodic_e_translation_invariance():
    """e is periodic: rolling the source along e rolls the solution by the same shift, to round-off
    (the operator is translation-invariant in e)."""
    rng = np.random.default_rng(2)
    nx, ne, dx, de = 28, 24, 6.0e3, 6.0e3
    lam = bps.screening_rate()
    src = rng.standard_normal((nx, ne))
    Phi = bps.solve_screened_poisson(src, dx, de, lam, bc_x="neumann")
    PhiR = bps.solve_screened_poisson(np.roll(src, 5, axis=1), dx, de, lam, bc_x="neumann")
    assert np.allclose(PhiR, np.roll(Phi, 5, axis=1), atol=1e-12)


def test_neumann_wall_is_zero_gradient():
    """The 'neumann' x-BC is zero-gradient at the walls (the 'nearest'-clamped smoothing wall): a source
    symmetric about the cross-track centre yields a Phi whose normal derivative vanishes at x[0]/x[-1]."""
    nx, ne, dx, de = 41, 16, 5.0e3, 5.0e3
    lam = bps.screening_rate()
    # a smooth, e-uniform bump centred in x -> the reflective wall gives ~zero one-sided gradient there
    xi = np.arange(nx)
    src = np.exp(-((xi - nx // 2) ** 2) / (2 * 4.0 ** 2))[:, None] * np.ones((1, ne))
    Phi = bps.solve_screened_poisson(src, dx, de, lam, bc_x="neumann")
    assert abs(Phi[1, 0] - Phi[0, 0]) / dx < 1e-3 * abs(Phi[:, 0]).max() / dx
    assert abs(Phi[-1, 0] - Phi[-2, 0]) / dx < 1e-3 * abs(Phi[:, 0]).max() / dx


# ============================================================ the K_0 analytic gate
def test_k0_greens_function_recovery():
    """GATE: a unit point source recovers the analytic Green's function Phi = -(1/2pi) K_0(lam r) at
    radii >> grid and << domain (where discreteness and the box BC are negligible). Large periodic square."""
    lam = bps.screening_rate()
    N, Ldom = 256, 24.0e5                      # domain ~ 6 lam^-1 across; lam^-1 ~ 4.2e5 m
    d = Ldom / N
    src = np.zeros((N, N)); src[N // 2, N // 2] = 1.0 / (d * d)   # unit point source (integral = 1)
    Phi = bps.solve_screened_poisson(src, d, d, lam, bc_x="periodic")
    ii, jj = np.indices((N, N))
    r = np.hypot((ii - N // 2) * d, (jj - N // 2) * d)
    m = (r > 4 * d) & (r < 1.2 / lam)
    ana = (1.0 / (2.0 * np.pi)) * k0(lam * r[m])
    rel = np.abs(-Phi[m] - ana) / np.abs(ana)
    assert np.median(rel) < 0.05               # ~1.8% in practice
    assert np.percentile(rel, 90) < 0.10
    # pointwise at r ~ lam^-1 (the e-folding scale, where the kernel is most informative)
    k = np.argmin(np.abs(r - 1.0 / lam))
    assert abs(-Phi.flat[k] - (1.0 / (2.0 * np.pi)) * k0(1.0)) / ((1.0 / (2.0 * np.pi)) * k0(1.0)) < 0.05


def test_greens_function_helper_matches_scipy_k0():
    """``greens_function_k0`` is the analytic reference -(1/2pi) K_0(lam r) used by the gate above."""
    lam = bps.screening_rate()
    r = np.array([1e5, 2e5, 4e5, 8e5])
    assert np.allclose(bps.greens_function_k0(r, lam), -(1.0 / (2.0 * np.pi)) * k0(lam * r))
    assert (bps.greens_function_k0(r, lam) < 0).all()      # the (lap-lam^2)^-1 of a +point source is < 0


# ============================================================ the 400 km screening-length gate
def test_screening_length_is_about_400km_from_physics():
    """GATE: lam^-1 = c/r_fric lands at ~400 km with c = sqrt(2 beta_T dT_char) pinned from the thermal
    contrast -- the free consistency check that the screened fetch matches the Gaussian phi_smooth_km."""
    c = bps.density_current_speed()
    assert c == pytest.approx(np.sqrt(2.0 * 30.0 * 30.0))        # 42.43 m/s, from beta_T & dT_char alone
    L = bps.screening_length_km()
    assert 380.0 < L < 460.0                                     # ~424 km ~ the 400 km frictional length
    assert L == pytest.approx(c / bps.R_FRIC_DEFAULT / 1e3)
    # consistency with the shipped Gaussian smoothing length (within ~20%)
    assert abs(L - 400.0) / 400.0 < 0.20


def test_c_pinned_from_physics_not_wind_cap():
    """``c`` comes from the density-current physics (beta_T, dT_char), NOT by inheriting wind_cap = 40 m/s
    (a numerical validity edge). The two land near each other for the same density-current reason -- that
    is the consistency, not the definition -- so c != wind_cap exactly."""
    from flatworld.stage_c_map import StageCConfig
    wind_cap = StageCConfig().wind_cap                          # 40 m/s, the numerical cap
    assert bps.density_current_speed() != wind_cap              # 42.43 != 40: independent pin
    assert abs(bps.density_current_speed() - wind_cap) / wind_cap < 0.15   # but close (same physics)


def test_screening_length_robust_across_thermal_contrast():
    """Across the documented land/sea contrast band dT_char ~ 25-35 K the screening length stays ~400 km
    (~387-458 km) -- the ~400 km consistency is not a knife-edge of one dT_char choice."""
    for dT in (25.0, 30.0, 35.0):
        L = bps.screening_length_km(dT_char=dT)
        assert abs(L - 400.0) / 400.0 < 0.20


# ============================================================ the bounded-source-response gate
def _point_source_peak(n, lam, smooth_km, Ldom=8.0e6):
    d = Ldom / n
    s = np.zeros((n, n)); s[n // 2, n // 2] = 1.0 / (d * d)
    s = bps.smooth_source(s, d, d, smooth_km)
    return float(np.abs(bps.solve_screened_poisson(s, d, d, lam, bc_x="neumann")).max())


def test_bounded_source_response_is_resolution_independent_when_smoothed():
    """GATE: a FIXED PHYSICAL source-smooth makes the point-source peak |Phi| resolution-INDEPENDENT (the
    K_0 log-cusp is de-singularized). Holds at both the full screening length AND the smaller default
    de-singularization scale; WITHOUT smoothing it sharpens with resolution (the spurious grid-scale
    lap(Phi) spike this smoothing prevents)."""
    lam = bps.screening_rate()
    ns = (64, 128, 256)
    full = [_point_source_peak(n, lam, bps.screening_length_km()) for n in ns]          # lam^-1
    default = [_point_source_peak(n, lam, bps.SOURCE_DESING_FRAC * bps.screening_length_km()) for n in ns]
    raw = [_point_source_peak(n, lam, 0.0) for n in ns]
    assert (max(full) - min(full)) / np.mean(full) < 0.02                # lam^-1: very flat
    assert (max(default) - min(default)) / np.mean(default) < 0.10       # default ~lam^-1/4: still flat
    # raw: grows monotonically and materially with resolution (the log-singularity)
    assert raw[2] > raw[1] > raw[0] and raw[2] / raw[0] > 1.3
    assert raw[2] > 2.0 * default[2]           # the un-smoothed cusp (still growing) tops the bounded value


def test_grid_scale_smoothing_is_NOT_resolution_independent():
    """WHY the default is a fixed PHYSICAL length, not a grid-scale (few-cell) one (PR #162): a smooth
    tied to the grid (n_cells * dx) shrinks with resolution, so the K_0 cusp re-sharpens -- the issue-#82
    failure the smoothing exists to prevent. Demonstrates the rejected alternative actually fails."""
    lam = bps.screening_rate()
    def peak_gridscale(n, ncells=3, Ldom=8.0e6):
        d = Ldom / n
        s = np.zeros((n, n)); s[n // 2, n // 2] = 1.0 / (d * d)
        s = bps.smooth_source(s, d, d, ncells * d / 1e3)               # smooth = 3 cells (grid-tied)
        return float(np.abs(bps.solve_screened_poisson(s, d, d, lam, bc_x="neumann")).max())
    pk = [peak_gridscale(n) for n in (64, 128, 256)]
    assert pk[2] > pk[1] > pk[0]                                       # grows with resolution (cusp returns)
    assert (max(pk) - min(pk)) / np.mean(pk) > 0.3                     # materially resolution-dependent


def test_default_source_smooth_is_honest_amplitude_dropin():
    """REGRESSION GUARD (PR #162): at its DEFAULT source-smooth the screened thermal Phi is a quantitative
    drop-in for the Gaussian-400 build (amplitude within ~30 %), NOT attenuated. The old default
    (source_smooth = lam^-1) double-counted the inversion's own smoothing and attenuated the thermal Phi
    2-7x -- guard against regressing to it. Measured on a domain a few lam^-1 wide (the land/sea regime)."""
    import scipy.ndimage as ndi
    nx, ne, dx, de = 40, 240, 8.0e3, 8.0e3              # along-track ~1900 km ~ 4.5 lam^-1
    rng = np.random.default_rng(12)
    T = 10.0 * ndi.gaussian_filter(rng.standard_normal((nx, ne)), 3.0, mode=("nearest", "wrap"))
    T[:, ne // 2:] += 8.0; h = np.zeros((nx, ne))
    Phi_g = blf.geopotential(T, h, dx, de, beta_T=30.0, g_terrain=0.0, phi_smooth_km=400.0)
    amp = lambda P: float(np.std(P - P.mean(axis=1, keepdims=True)))
    a_g = amp(Phi_g)
    a_default = amp(bps.geopotential_screened(T, h, dx, de))                         # the new default
    a_oversmooth = amp(bps.geopotential_screened(T, h, dx, de,
                                                 source_smooth_km=bps.screening_length_km()))  # the old default
    assert 0.6 < a_default / a_g < 1.4         # honest quantitative drop-in (kernel diff only)
    assert a_oversmooth / a_g < 0.5            # the lam^-1 over-smooth is materially weaker (the bug)
    assert a_default > 1.8 * a_oversmooth      # the fix restores the amplitude the old default lost


# ============================================================ source builders
def test_thermal_source_matches_bl_forcing_presmoothing_field():
    """``thermal_equilibrium_phi`` is exactly the field bl_forcing.geopotential smooths: -beta_T(T-Tref)
    + g_terrain*g*h with Tref the per-row zonal mean."""
    rng = np.random.default_rng(5)
    nx, ne = 20, 16
    T = 20.0 * rng.standard_normal((nx, ne))
    h = 1000.0 * rng.random((nx, ne))
    got = bps.thermal_equilibrium_phi(T, h, beta_T=30.0, g_terrain=0.6)
    Tref = T.mean(axis=1, keepdims=True)
    expect = -30.0 * (T - Tref) + 0.6 * C.G_ACCEL * h
    assert np.allclose(got, expect)


def test_latent_source_default_inert_and_sign():
    """``latent_equilibrium_phi`` is inert when gms_geopot=0 (zero), and a POSITIVE precip anomaly lowers
    Phi_eq (latent heating -> thermal low), the feedback's sign."""
    P = np.array([[1.0, 5.0], [3.0, 2.0]])
    base = np.full_like(P, 2.0)
    assert np.array_equal(bps.latent_equilibrium_phi(P, base, gms_geopot=0.0), np.zeros_like(P))
    assert np.array_equal(bps.latent_equilibrium_phi(P, None, gms_geopot=0.0), np.zeros_like(P))   # inert
    phi_eq = bps.latent_equilibrium_phi(P, base, gms_geopot=10.0)
    assert (phi_eq[P > base] < 0).all() and (phi_eq[P < base] > 0).all()   # rain anomaly -> low Phi


def test_latent_without_baseline_raises_when_active():
    """The C3 footgun guard (PR #162): a non-zero gms_geopot with P but NO P_baseline raises rather than
    silently returning a thermal-only Phi (the feedback lost by omission). The inert gms_geopot=0 path and
    an explicit P_baseline=0 both stay fine."""
    P = np.array([[1.0, 5.0], [3.0, 2.0]])
    with pytest.raises(ValueError, match="P_baseline"):
        bps.latent_equilibrium_phi(P, None, gms_geopot=10.0)
    with pytest.raises(ValueError, match="P_baseline"):
        bps.geopotential_screened(np.zeros((8, 8)), np.zeros((8, 8)), 8e3, 8e3, P=np.ones((8, 8)),
                                  gms_geopot=10.0)                    # P_baseline omitted -> raises
    # opt in explicitly: P_baseline=0 drives on full P, no raise
    assert np.array_equal(bps.latent_equilibrium_phi(P, 0.0, gms_geopot=10.0), -10.0 * P)


def test_geopotential_screened_thermal_only_by_default():
    """The assembled build is thermal-only by default (gms_geopot=0): passing P with no baseline/gain
    leaves Phi identical to the no-P build -- the latent path is built but inert (Stage C untouched)."""
    rng = np.random.default_rng(6)
    nx, ne, dx, de = 24, 20, 8.0e3, 9.0e3
    T = 10.0 * rng.standard_normal((nx, ne)); h = 500.0 * rng.random((nx, ne))
    P = 20.0 * rng.random((nx, ne))
    Phi_noP = bps.geopotential_screened(T, h, dx, de)
    Phi_P = bps.geopotential_screened(T, h, dx, de, P=P)            # no P_baseline/gms -> still inert
    assert np.array_equal(Phi_noP, Phi_P)


def test_one_footing_parts_sum_to_total():
    """Thermal and latent ride the SAME inverse: Phi_thermal + Phi_latent == Phi (operator linearity),
    to round-off -- the 'one footing' the roadmap requires."""
    rng = np.random.default_rng(7)
    nx, ne, dx, de = 28, 24, 8.0e3, 8.0e3
    T = 12.0 * rng.standard_normal((nx, ne)); h = 800.0 * rng.random((nx, ne))
    P = 30.0 * rng.random((nx, ne)); base = np.full((nx, ne), 10.0)
    Phi, parts = bps.geopotential_screened(T, h, dx, de, P=P, P_baseline=base, gms_geopot=8.0,
                                           return_parts=True)
    assert np.allclose(parts["Phi_thermal"] + parts["Phi_latent"], Phi, atol=1e-10)
    assert parts["screening_length_km"] == pytest.approx(bps.screening_length_km())


def test_latent_feedback_lowers_phi_under_a_rain_blob():
    """A localized POSITIVE precip anomaly drives a Phi LOW at the blob (convergence -> ascent -> more
    rain: the positive feedback's sign), localized and decaying away. Domain >> the smoothing length so
    the blob stays resolved (a domain narrower than ~lam^-1 would smear it to a near-uniform offset)."""
    nx, ne, dx, de = 160, 160, 16.0e3, 16.0e3                       # ~2560 km box >> ~424 km smoothing
    T = np.zeros((nx, ne)); h = np.zeros((nx, ne))
    base = np.zeros((nx, ne)); P = base.copy()
    ci, cj = nx // 2, ne // 2
    P[ci, cj] = 50.0                                                # a single wet cell (mm/day)
    _, parts = bps.geopotential_screened(T, h, dx, de, P=P, P_baseline=base, gms_geopot=10.0,
                                         return_parts=True)
    pl = parts["Phi_latent"]
    assert pl.min() < 0 and abs(pl.min()) > 1e-6                    # a real low formed
    # the low is at the blob (argmin within a smoothing length of the centre)
    imin, jmin = np.unravel_index(np.argmin(pl), pl.shape)
    assert abs(imin - ci) * dx < 1.0 / parts["lam"] and abs(jmin - cj) * de < 1.0 / parts["lam"]
    # localized: |Phi_latent| at a corner (~4 lam^-1 from the blob) is a small fraction of the central low
    assert abs(pl[0, 0]) < 0.15 * abs(pl[ci, cj])


def test_screened_thermal_pattern_consistent_with_gaussian_build():
    """SOFT consistency: the screened thermal Phi correlates strongly with the shipped Gaussian-400 build
    over the same field (same large-scale thermal low; the kernels differ -- Lorentzian vs Gaussian -- so
    this is a pattern check, not an amplitude match: that difference is exactly why a NEW solver is needed)."""
    rng = np.random.default_rng(8)
    nx, ne, dx, de = 60, 80, 8.0e3, 9.0e3
    import scipy.ndimage as ndi
    T = 10.0 * ndi.gaussian_filter(rng.standard_normal((nx, ne)), 3.0, mode=("nearest", "wrap"))
    T[:, ne // 2:] += 8.0                                           # a land/sea jump
    h = np.zeros((nx, ne))
    Phi_g = blf.geopotential(T, h, dx, de, beta_T=30.0, g_terrain=0.0, phi_smooth_km=400.0)
    Phi_s = bps.geopotential_screened(T, h, dx, de, source_smooth_km=0.0)   # pure inversion smoothing
    a = Phi_s - Phi_s.mean(axis=1, keepdims=True)
    b = Phi_g - Phi_g.mean(axis=1, keepdims=True)
    corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    assert corr > 0.9


def test_screened_solver_wired_only_in_gated_prognostic_path():
    """C1 shipped the solver standalone; Track C / C3 WIRES it into Stage C -- but only inside the gated
    prognostic-Phi hook (`_update_phi_prognostic`), via a LAZY import. So importing stage_c_map and running
    the shipped STATIC-Phi step never imports or calls bl_phi_screened (the byte-identical default path).
    (Supersedes the C1-era 'stage_c_map never mentions bl_phi_screened' check, which C3 intentionally
    changes.)"""
    import importlib, sys, inspect
    # lazy-import BEHAVIOUR (robust to comments): freshly importing stage_c_map must NOT drag in the
    # screened solver -- the static-Phi default path never references it.
    sys.modules.pop("flatworld.bl_phi_screened", None)
    sys.modules.pop("flatworld.stage_c_map", None)
    sc = importlib.import_module("flatworld.stage_c_map")
    assert "flatworld.bl_phi_screened" not in sys.modules      # not imported at module load -> lazy / gated
    assert "bl_phi_screened" in inspect.getsource(sc.StageCMap._update_phi_prognostic)  # only in the C3 hook
