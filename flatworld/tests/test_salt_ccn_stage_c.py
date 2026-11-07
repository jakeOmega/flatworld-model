"""Acceptance tests for the salt-aerosol CCN coupling on the Stage-C deliverable map (GitHub #237,
Stage-C side; the AD-core autoconv limb #243's sibling).

When ``salt_ccn`` is ON, Stage C scales its WARM-rain conversion -- the convective precipitation
efficiency PE=(1-f_loft) AND the stratiform rainout demand -- by ``(N_d/N_ref)^beta`` per cross-track
row, so the deliverable ``P(x,e)`` map carries the salt CCN band-magnitude effect. Default OFF and
master-ON-at-the-reference (N_d == N_ref) are BYTE-IDENTICAL to the pre-feature map. Direction-firm /
magnitude named-wall-soft (aerosol microphysics + crust-state erodibility); the field is SWEPT, never
tuned to a target. These tests are fast (low-resolution standalone maps + one tiny AD core for the
interface-contract exposure).
"""
import numpy as np
import pytest
from flatworld.topography import Topography
from flatworld.stage_c_map import StageCMap, StageCConfig


def _topo():
    return Topography.procedural_continent(seed=3, nx=60, ne=60, Le_km=6000, width_km=3500, center_theta=48)


def _run(topo, cfg, nsteps=120):
    m = StageCMap(topo, cfg)
    for _ in range(nsteps):
        m.step()
    return m


# --------------------------------------------------------------- config defaults + field construction
def test_salt_ccn_defaults_off_and_field_absent():
    """Master OFF by default; a built map carries no warm-rain factor / N_d field (the precip code keeps
    its verbatim lines -> byte-identical)."""
    cfg = StageCConfig()
    assert cfg.salt_ccn is False
    assert cfg.salt_ccn_nd is None
    assert cfg.salt_ccn_nd_ref == 100.0
    assert cfg.salt_ccn_beta == -0.8
    m = StageCMap(_topo(), cfg)
    assert m.salt_warm_col is None
    assert m.N_d is None


def test_salt_ccn_builds_uniform_factor_from_scalar_nd():
    m = StageCMap(_topo(), StageCConfig(salt_ccn=True, salt_ccn_nd=1000.0, salt_ccn_nd_ref=100.0,
                                        salt_ccn_beta=-0.8))
    assert m.N_d is not None and m.N_d.shape == (m.nx,)
    assert np.allclose(m.N_d, 1000.0)
    # warm factor = (N_d/N_ref)^beta = 10^-0.8, uniform, broadcast over e
    assert m.salt_warm_col.shape == (m.nx, 1)
    assert np.allclose(m.salt_warm_col, 10.0 ** -0.8)


def test_salt_ccn_nd_none_defaults_to_reference_unit_factor():
    """salt_ccn_nd None -> N_d == N_ref everywhere -> factor exactly 1.0 (the clean zero-point)."""
    m = StageCMap(_topo(), StageCConfig(salt_ccn=True, salt_ccn_nd=None, salt_ccn_nd_ref=100.0))
    assert np.allclose(m.N_d, 100.0)
    assert np.array_equal(m.salt_warm_col, np.ones((m.nx, 1)))


def test_salt_ccn_callable_nd_field_is_per_row():
    """A callable N_d(theta) (the cascade-wired AD-core field) sets a per-cross-track-row factor, mirroring
    f_loft_of_theta."""
    topo = _topo()
    # a monotone N_d(theta): clean equatorward, loaded poleward
    nd_of_theta = lambda th: 100.0 + 20.0 * np.asarray(th, float)
    m = StageCMap(topo, StageCConfig(salt_ccn=True, salt_ccn_nd_of_theta=nd_of_theta,
                                     salt_ccn_nd_ref=100.0, salt_ccn_beta=-0.8))
    expect = (nd_of_theta(m.theta_deg) / 100.0) ** -0.8
    assert np.allclose(m.salt_warm_col[:, 0], expect)
    # the callable OVERRIDES the scalar salt_ccn_nd (mirrors f_loft_of_theta over the scalar)
    m2 = StageCMap(topo, StageCConfig(salt_ccn=True, salt_ccn_nd_of_theta=nd_of_theta,
                                      salt_ccn_nd=99999.0, salt_ccn_nd_ref=100.0))
    assert np.allclose(m2.N_d, nd_of_theta(m2.theta_deg))


# --------------------------------------------------------------- byte-identity (the OFF / zero-point limit)
@pytest.mark.parametrize("rh_crit", [1.0, 0.80])
def test_master_on_at_reference_is_byte_identical(rh_crit):
    """Master ON with N_d == N_ref (factor 1.0) reproduces the OFF map BIT-for-BIT -- on both the legacy
    (wa_rh_crit=1.0) and the production sub-grid (wa_rh_crit<1) stratiform closures."""
    topo = _topo()
    base = dict(dt=900.0, wa_rh_crit=rh_crit)
    off = _run(topo, StageCConfig(**base))
    ref = _run(topo, StageCConfig(salt_ccn=True, salt_ccn_nd=100.0, salt_ccn_nd_ref=100.0, **base))
    assert np.array_equal(off.P, ref.P)
    assert np.array_equal(off.W_a, ref.W_a)
    assert np.array_equal(off.W_s, ref.W_s)


@pytest.mark.parametrize("rh_crit", [1.0, 0.80])
def test_master_on_at_reference_byte_identical_with_nonconstant_f_loft(rh_crit):
    """The zero-point is BIT-for-BIT even with a NON-constant f_loft(theta) callable -- the production wiring.
    The convective re-split is written as the original split plus the salt delta (U = U0 + (P_conv0 - P_conv)),
    so at factor 1 P_conv == P_conv0 exactly and U == U0, avoiding the 1-(1-f_loft) != f_loft float drift a
    plain U=(1-pe)*drain would carry (~1e-14). Guards PR #248 review finding 1."""
    topo = _topo()
    floft = lambda th: 0.30 + 0.20 * np.clip((np.asarray(th, float) - 40.0) / 20.0, 0.0, 1.0)   # 0.30..0.50
    base = dict(dt=900.0, wa_rh_crit=rh_crit, f_loft_of_theta=floft)
    off = _run(topo, StageCConfig(**base))
    ref = _run(topo, StageCConfig(salt_ccn=True, salt_ccn_nd=100.0, salt_ccn_nd_ref=100.0, **base))
    assert np.array_equal(off.P, ref.P)
    assert np.array_equal(off.W_a, ref.W_a)
    assert np.array_equal(off.W_s, ref.W_s)


# --------------------------------------------------------------- the warm-rain effect (sign-bracketed)
@pytest.mark.parametrize("rh_crit", [1.0, 0.80])
def test_suppression_lowers_invigoration_raises_total_precip(rh_crit):
    """SIGN-AMBIGUOUS, bracketed: at a fixed loaded N_d, beta<0 SUPPRESSION reduces the map's total rain and
    beta>0 INVIGORATION raises it, relative to the reference -- so P(suppress) < P(ref) < P(invigorate). The
    absolute R is magnitude-soft (named wall = aerosol microphysics); only the ORDERING is asserted."""
    topo = _topo()
    base = dict(dt=900.0, wa_rh_crit=rh_crit)
    ref = _run(topo, StageCConfig(salt_ccn=True, salt_ccn_nd=100.0, salt_ccn_nd_ref=100.0, **base))
    supp = _run(topo, StageCConfig(salt_ccn=True, salt_ccn_nd=1000.0, salt_ccn_nd_ref=100.0,
                                   salt_ccn_beta=-0.8, **base))
    invig = _run(topo, StageCConfig(salt_ccn=True, salt_ccn_nd=1000.0, salt_ccn_nd_ref=100.0,
                                    salt_ccn_beta=0.5, **base))
    Pref, Psupp, Pinvig = ref.P.sum(), supp.P.sum(), invig.P.sum()
    assert Psupp < Pref < Pinvig
    # band LOCATION (the cross-track argmax of the zonal-mean precip) is NOT moved by the magnitude term
    zm = lambda m: m.P.mean(axis=1)
    assert np.argmax(zm(supp)) == np.argmax(zm(ref)) == np.argmax(zm(invig))


def test_salt_on_keeps_a_closed_water_budget_and_stays_finite():
    """The salt coupling re-routes warm-rain conversion but must not break conservation: the cumulative
    water budget still closes, the map is finite, P>=0, and the wind cap is respected."""
    topo = _topo()
    base = dict(dt=900.0, wa_rh_crit=0.80)
    for beta, nd in [(-0.8, 1000.0), (0.5, 1000.0), (-1.2, 5000.0)]:
        m = _run(topo, StageCConfig(salt_ccn=True, salt_ccn_nd=nd, salt_ccn_nd_ref=100.0,
                                    salt_ccn_beta=beta, **base), nsteps=150)
        d = m.diagnostics()
        assert np.isfinite(d["P_mm_day"]).all() and (d["P_mm_day"] >= 0).all()
        assert np.abs(d["u"]).max() <= 40.1 and np.abs(d["v"]).max() <= 40.1
        # closure preserved (loose: standalone, PARTIALLY RELAXED -- storage still filling, so the
        # cum ratio sits below 1). Floor 0.5 -> 0.4 with the supply saturation cap (2026-07-18):
        # capping the cold-column delivery shrinks the S_band source term, which lowers the
        # partially-relaxed ratio a hair (measured 0.493); the budget identity is unchanged and the
        # converged closure still -> ~1 (test_supply_sat_cap locks the cap's behavior directly).
        assert 0.4 < d["water_closure_cum"] < 1.5


def test_convective_pe_clip_guards_strong_invigoration():
    """The convective precipitation efficiency PE=(1-f_loft)*factor is clipped to [0,1], so a strong
    invigoration factor over a low-f_loft row cannot push PE>1 (and the aloft complement 1-PE>=0). The run
    stays finite and physical."""
    topo = _topo()
    # f_loft=0.1 -> PE=0.9; factor 10^(0.8*... ) >> 1 would give PE>1 without the clip
    m = _run(topo, StageCConfig(salt_ccn=True, salt_ccn_nd=100000.0, salt_ccn_nd_ref=100.0,
                                salt_ccn_beta=0.8, f_loft=0.1, dt=900.0), nsteps=80)
    d = m.diagnostics()
    assert np.isfinite(d["P_mm_day"]).all() and (d["P_mm_day"] >= 0).all()


# --------------------------------------------------------------- the AD-core -> Stage-C interface contract
def test_stage_c_inputs_exposes_nd_when_salt_on_else_none():
    """The AD core's ``stage_c_inputs`` exposes ``salt_ccn_nd_of_theta`` (a theta->N_d callable) when the
    core's salt_ccn is ON, and None when OFF -- the field the cascade wires into Stage C."""
    from flatworld.stage_ad_core import StageADCore, StageADConfig

    def _core(**kw):
        cfg = StageADConfig(nx=16, nz=16, theta_max_deg=55.0, **kw)
        m = StageADCore(cfg)
        m.run_dynamics = False
        for _ in range(3):
            m.step()
        return m

    off = _core()
    assert off.stage_c_inputs()["salt_ccn_nd_of_theta"] is None
    on = _core(salt_ccn=True, prognostic_condensate=True, salt_ccn_nd=1000.0, salt_ccn_nd_ref=100.0)
    fn = on.stage_c_inputs()["salt_ccn_nd_of_theta"]
    assert fn is not None
    th_mid = float(np.median(on.diagnostics()["theta_deg"]))
    assert np.isfinite(float(fn(th_mid))) and float(fn(th_mid)) > 0.0
