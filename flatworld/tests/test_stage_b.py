"""Stage B tests: the f=0 overturning shape + ocean cubic, checked INDEPENDENTLY against the doc's
"robust" claims (single cell, equatorward surface wind, poleward-aloft moisture, no-Ekman cubic)."""
import numpy as np
import pytest
from flatworld import constants as C
from flatworld import thermo as th
from flatworld.stage_b_overturning import (StageBAtmosphere, StageBConfig, wind_driven_ocean_profile)


# ---- (2) the ocean wind-driven profile (analytic) ----
def test_ocean_cubic_reverses_at_minus_H_over_3():
    z, u, rev, ridx = wind_driven_ocean_profile(nz=600, H=300.0)
    assert rev == pytest.approx(-100.0)                       # -H/3
    assert z[ridx] == pytest.approx(-100.0, abs=2.0)          # numerical zero-crossing at -H/3
    assert u[-1] == pytest.approx(1.0)                        # surface runs with the wind
    # lower two-thirds run AGAINST the wind (negative), peaking near -1/3 at s=-2/3
    below = z < -100.0
    assert np.all(u[below][1:-1] < 0)                         # opposite sign below the reversal
    assert u.min() == pytest.approx(-1.0 / 3.0, abs=0.02)     # peak return = -1/3 surface speed


def test_ocean_profile_has_no_thin_ekman_layer():
    # the cubic is smooth over the whole depth (no thin near-surface boundary layer): the reversal
    # sits at H/3, not within a few percent of the surface as a rotating Ekman spiral would.
    z, u, rev, ridx = wind_driven_ocean_profile(nz=400, H=200.0)
    assert abs(z[ridx]) > 0.2 * 200.0                         # reversal deep (~H/3), not an Ekman skin


# ---- (1) the atmospheric overturning shape ----
@pytest.fixture(scope="module")
def relaxed_atmos():
    cfg = StageBConfig(nx=100, nz=31, moist=True, max_seconds=60)
    m = StageBAtmosphere(cfg)
    for _ in range(5000):
        m.step()
    return m


def test_single_overturning_cell(relaxed_atmos):
    d = relaxed_atmos.diagnostics()
    assert d["surf_sign_changes"] == 0                        # ONE cell: surface wind never reverses


def test_rising_at_equator_sinking_at_pole(relaxed_atmos):
    d = relaxed_atmos.diagnostics()
    w = d["w"]; nz = relaxed_atmos.cfg.nz
    assert w[nz // 2, 3] > 0                                   # ascent over the hot equator
    assert w[nz // 2, -4] < 0                                  # descent over the cold pole
    assert d["rising_theta"] < 15                              # rising branch near the sun-line


def test_equatorward_surface_wind_everywhere(relaxed_atmos):
    d = relaxed_atmos.diagnostics()
    us = d["u_surf"][3:-3]
    assert np.all(us < 0)                                      # u<0 = toward the equator, at every lat


def test_moisture_transport_is_poleward_aloft(relaxed_atmos):
    # the headline two-layer premise: moist flow poleward ALOFT, dry return equatorward at the SURFACE
    d = relaxed_atmos.diagnostics()
    assert d["flux_upper"] > 0                                 # poleward moisture flux aloft
    assert d["flux_lower"] < 0                                 # equatorward (dry) flux at the surface


def test_whole_ocean_is_wet_rain_at_and_equatorward_of_ribbon(relaxed_atmos):
    d = relaxed_atmos.diagnostics()
    th = d["theta_deg"]; p = d["precip"]
    # rain present over the inner sea AND the ribbon edge; peak NOT poleward of the ribbon
    assert p[(th >= 40) & (th <= 51)].max() > 0
    assert th[np.argmax(p)] < 52                               # rain peaks at/equatorward of the ribbon


def test_stageb_moisture_evaporation_law():
    """Issue-A moisture rebuild: B evaporates from A's WETNESS MASK (no evap over the salt flat),
    toward ABSOLUTE qsat at B's dimensional column temperature (NOT a normalized tracer/qsat).

    After one step from rest (q=0, psi=0 -> no advection, no condensation), the surface moisture must
    equal the evaporation law q[0] = wetness(SST) * qsat_absolute[0] * evap_eff, with wetness = A's
    55->70 C ramp. This pins BOTH facets: (a) the source is the wet ocean (wetness=0 over the salt flat
    SST>=70 C), and (b) the saturation is absolute (a realistic kg/kg, ~0.05 at 40 C -- NOT 1.0 from
    a /qsat.max normalization)."""
    # SST spanning salt flat (90 C, wetness 0) -> wet ocean (40 C, wetness 1) -> cold edge (-20 C)
    sst_of_theta = lambda th_deg: np.interp(th_deg, [0.0, 60.0], [90.0, -20.0])
    cfg = StageBConfig(nx=60, nz=21, moist=True, Tsurf_C_of_theta=sst_of_theta)
    m = StageBAtmosphere(cfg)
    m.step()                                                  # one step from rest
    SST_C = sst_of_theta(m.theta_deg)
    wetness = np.clip((70.0 - SST_C) / (70.0 - 55.0), 0.0, 1.0)
    qsat_surf_abs = th.qsat(np.maximum(SST_C + C.T_FREEZE, 150.0), C.P_SURF)   # absolute kg/kg at zeta=0
    expected_q0 = wetness * qsat_surf_abs * cfg.evap_eff
    assert np.allclose(m.q[0, :], expected_q0, rtol=1e-3, atol=1e-6)
    # absolute saturation: a realistic kg/kg field max (NOT normalized to exactly 1.0); wet ocean ~0.05
    assert m.qsat[0, np.argmin(np.abs(SST_C - 40.0))] == pytest.approx(qsat_surf_abs[np.argmin(np.abs(SST_C - 40.0))], rel=1e-6)
    assert np.all(m.q[0, wetness == 0.0] == 0.0)              # zero evaporation over the salt flat


def test_convergence_and_checkpoint_roundtrip():
    cfg = StageBConfig(nx=80, nz=25, moist=False, max_seconds=30)
    m = StageBAtmosphere(cfg)
    for _ in range(200):
        m.step()
    st = m.state()
    m2 = StageBAtmosphere(cfg)
    m2.load_state({k: np.asarray(v) for k, v in st.items()})
    assert np.allclose(m2.b, m.b) and np.allclose(m2.omega, m.omega)


# ---- INF-8: the separable DST streamfunction solver (drop-in for the splu back-substitution) ----
@pytest.mark.parametrize("nx,nz", [(100, 31), (200, 41), (61, 25)])
def test_stageb_dst_poisson_matches_splu(nx, nz):
    """The default "dst" solver (``SeparablePoisson``, ported from the AD core) is a MACHINE-PRECISION
    drop-in for the "splu" LU back-substitution on Stage B's constant-coefficient streamfunction Poisson --
    verified against THIS stage's own splu build on stored omega fields (the roadmap's ~1e-10 gate, run
    BEFORE trusting the swap). Random RHS incl. non-zero interior; walls are omega=0 (production BC) so the
    interior inversion is what is exercised."""
    rng = np.random.default_rng(nx * nz)
    m_dst = StageBAtmosphere(StageBConfig(nx=nx, nz=nz, poisson_method="dst"))
    m_splu = StageBAtmosphere(StageBConfig(nx=nx, nz=nz, poisson_method="splu"))
    assert type(m_dst._lu).__name__ == "SeparablePoisson"
    for _ in range(5):
        omega = rng.standard_normal((nz, nx))
        omega[0, :] = omega[-1, :] = omega[:, 0] = omega[:, -1] = 0.0     # omega=0 walls (Stage B BC)
        psi_dst = m_dst._lu.solve(omega.ravel())
        psi_splu = m_splu._lu.solve(omega.ravel())
        rel = np.max(np.abs(psi_dst - psi_splu)) / (np.max(np.abs(psi_splu)) + 1e-300)
        assert rel < 1e-9, f"dst vs splu psi disagree at {rel:.2e} (nx={nx}, nz={nz})"


def test_stageb_dst_integration_matches_splu():
    """A solver swap is not byte-identical, so gate the INTEGRATED result: stepping the SAME model with
    "dst" vs "splu" for many steps must leave the physics verdicts identical and the final streamfunction /
    circulation metrics equal within rtol (the C4 concern that a 3M-step explicit integration could amplify
    the ~1e-13 per-solve roundoff -- it does not; the steady attractor is robust)."""
    n = 3000
    common = dict(nx=80, nz=25, moist=True)
    m_splu = StageBAtmosphere(StageBConfig(poisson_method="splu", **common))
    m_dst = StageBAtmosphere(StageBConfig(poisson_method="dst", **common))
    for _ in range(n):
        m_splu.step()
    for _ in range(n):
        m_dst.step()
    d_s, d_d = m_splu.diagnostics(), m_dst.diagnostics()
    assert d_d["surf_sign_changes"] == d_s["surf_sign_changes"]          # identical cell count (verdict)
    assert d_d["rising_theta"] == pytest.approx(d_s["rising_theta"])      # identical rising branch
    denom = np.max(np.abs(d_s["psi"])) + 1e-300
    assert np.max(np.abs(d_d["psi"] - d_s["psi"])) / denom < 1e-8         # final psi equal within rtol
    assert d_d["flux_upper"] == pytest.approx(d_s["flux_upper"], rel=1e-6)
    assert d_d["flux_lower"] == pytest.approx(d_s["flux_lower"], rel=1e-6)


def test_stageb_invalid_poisson_method_raises():
    with pytest.raises(ValueError, match="poisson_method"):
        StageBAtmosphere(StageBConfig(nx=40, nz=21, poisson_method="banana"))


def test_stageb_fused_upwind_is_bit_identical():
    """The opt-in numba-fused upwind tendency (``fuse_upwind=True``) is BIT-IDENTICAL to the numpy ``_upwind``
    -- same divided-difference and branch order, pure fp64, so ``array_equal`` holds on a random field INCL.
    the wall edges (where the backward/forward difference is zeroed). It is a pure wall-clock lever, no
    numerics change."""
    rng = np.random.default_rng(7)
    m_np = StageBAtmosphere(StageBConfig(nx=64, nz=21, fuse_upwind=False))
    m_nb = StageBAtmosphere(StageBConfig(nx=64, nz=21, fuse_upwind=True))
    assert m_np._upwind_kernel is None and m_nb._upwind_kernel is not None
    for _ in range(4):
        F = rng.standard_normal((21, 64))
        u = rng.standard_normal((21, 64))
        w = rng.standard_normal((21, 64))
        assert np.array_equal(m_np._upwind(F, u, w), m_nb._upwind(F, u, w))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
