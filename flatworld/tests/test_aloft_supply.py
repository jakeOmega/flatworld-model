"""WS-A: the absolute aloft-moisture-column supply F(theta) fix.

The legacy law normalized the AD core's ABSOLUTE precip(theta) to a [0,1] shape and re-imposed an
arbitrary 8-50 mm scale -- so the equatorial chimney's 1/dx precip spike (the global precip.max())
dominated the /max() and crushed the ribbon's aloft supply toward the 8 mm floor. The fix
(``cascade.build_aloft_supply_target(absolute=True)`` fed by ``StageADCore._aloft_moisture_column``) uses
the resolved absolute aloft moisture column directly. Default OFF -> byte-identical.
"""
import numpy as np
import pytest

from flatworld import cascade
from flatworld.stage_ad_core import StageADCore, StageADConfig, ALOFT_SIGMA_THRESH


def _legacy_inline(precip):
    band = np.clip(np.asarray(precip, float), 0.0, None)
    if band.max() > 0:
        band = band / band.max()
    return 8.0 + 42.0 * band


def test_default_law_is_byte_identical_to_legacy():
    th = np.linspace(0, 70, 24)
    rng = np.random.default_rng(3)
    precip = np.abs(rng.normal(size=24))
    precip[1] = 362.0                                   # an equatorial chimney-like spike
    F = cascade.build_aloft_supply_target(th, precip, aloft_moisture_mm=None, absolute=False)
    assert np.array_equal(F, _legacy_inline(precip))    # exact, not just close


def test_absolute_uses_the_column_not_the_normalized_shape():
    th = np.linspace(0, 70, 24)
    precip = np.abs(np.random.default_rng(4).normal(size=24)); precip[1] = 362.0
    aloft = np.linspace(2.0, 25.0, 24)
    F = cascade.build_aloft_supply_target(th, precip, aloft_moisture_mm=aloft, absolute=True)
    # over the Stage-C domain (theta >= chimney_theta) F IS the absolute column (no 8 mm floor / no /max)
    inside = th >= cascade.ALOFT_CHIMNEY_THETA
    assert np.allclose(F[inside], aloft[inside])
    assert F.max() <= 25.0 + 1e-9                        # bounded by the column, not the 50 mm band cap


def test_absolute_clamps_the_equatorward_chimney():
    th = np.linspace(0, 70, 24)
    aloft = np.full(24, 10.0)
    aloft[th < cascade.ALOFT_CHIMNEY_THETA] = 999.0      # a chimney/inner-sea column blow-up equatorward
    F = cascade.build_aloft_supply_target(th, np.ones(24), aloft_moisture_mm=aloft, absolute=True)
    # equatorward region is clamped to the chimney-edge value (10), never the 999 blow-up
    assert F[th < cascade.ALOFT_CHIMNEY_THETA].max() <= 10.0 + 1e-9
    assert np.allclose(F[th >= cascade.ALOFT_CHIMNEY_THETA], 10.0)


def test_absolute_without_column_falls_back_and_warns():
    th = np.linspace(0, 70, 24)
    precip = np.abs(np.random.default_rng(5).normal(size=24))
    with pytest.warns(RuntimeWarning):
        F = cascade.build_aloft_supply_target(th, precip, aloft_moisture_mm=None, absolute=True)
    assert np.array_equal(F, _legacy_inline(precip))


def test_aloft_moisture_column_integral():
    # unbound method on a light stub: only self.sigma / self.dm are used
    class _Stub:
        pass
    s = _Stub()
    s.sigma = np.array([0.1, 0.3, 0.6, 0.9])             # top->surface; aloft = sigma<0.5 -> layers 0,1
    s.dm = np.array([2.0, 2.0, 2.0, 2.0])
    q = np.array([[1.0, 2.0, 3.0, 4.0],                  # (nx=2, nz=4)
                  [0.0, 0.0, 5.0, 5.0]])
    out = StageADCore._aloft_moisture_column(s, q, sigma_thresh=0.5)
    assert np.allclose(out, [1 * 2 + 2 * 2, 0.0])        # only the aloft (sigma<0.5) layers contribute


def _small_core():
    m = StageADCore(StageADConfig(nx=8, nz=14, theta_max_deg=60.0, conv_max_sweeps=4))
    m.run_dynamics = False
    for _ in range(150):
        m.step()
    m.run_dynamics = True
    for _ in range(150):
        m.step()
    return m


def test_diagnostics_carry_absolute_aloft_column():
    m = _small_core()
    d = m.diagnostics()
    assert "aloft_moisture_mm" in d and "sigma" in d
    aloft = np.asarray(d["aloft_moisture_mm"], float)
    assert aloft.shape == d["theta_deg"].shape
    assert np.all(np.isfinite(aloft)) and np.all(aloft >= 0)
    # matches a direct re-integration of q over sigma<thresh
    direct = m._aloft_moisture_column(d["q"], ALOFT_SIGMA_THRESH)
    assert np.allclose(aloft, direct)


def test_stage_c_inputs_default_off_is_legacy_flag_on_is_column(monkeypatch):
    m = _small_core()
    th = np.linspace(30.0, 58.0, 11)
    d = m.diagnostics()

    monkeypatch.delenv("COSMO_ALOFT_SUPPLY_ABSOLUTE", raising=False)
    F_off = np.asarray(m.stage_c_inputs()["aloft_supply_of_theta"](th), float)
    F_legacy = np.asarray(cascade._interp(d["theta_deg"], _legacy_inline(d["precip_mm_day"]))(th), float)
    assert np.allclose(F_off, F_legacy)                 # default OFF == legacy normalize-to-shape

    monkeypatch.setenv("COSMO_ALOFT_SUPPLY_ABSOLUTE", "1")
    F_on = np.asarray(m.stage_c_inputs()["aloft_supply_of_theta"](th), float)
    F_col = np.asarray(cascade.build_aloft_supply_target(
        d["theta_deg"], d["precip_mm_day"], aloft_moisture_mm=d["aloft_moisture_mm"], absolute=True), float)
    F_col_i = np.asarray(cascade._interp(d["theta_deg"], F_col)(th), float)
    assert np.allclose(F_on, F_col_i)                   # flag ON == the absolute aloft column
    assert not np.allclose(F_on, F_off)                 # and it actually differs from legacy
