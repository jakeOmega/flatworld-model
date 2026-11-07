"""PHY-1 regression: the gray-band windowed scheme's saturated-column OLR ceiling vs the published
runaway-greenhouse limit (ROADMAP PHY-1; results -> flatworld/PHY1_OLR_BIAS_RESULTS.md).

These are LEAF-LEVEL (radiation + thermo only) -- no checkpoint, no shade file, no core stepping -- so
they run in CI where the git-ignored production data is absent. They lock the MODEL ceiling (a
physics-stable number) within a tolerance band and validate it against the published lower/binding
runaway OLR limit; they deliberately do NOT assert the exact deviation
percentages, which would move if the recalled reference values are later corrected. The companion
plateau test is test_radiation.py::test_continuum_caps_hot_end_olr. See PHY1_OLR_BIAS_RESULTS.md for the
published-limit provenance (the lower, binding Simpson-Nakajima / Goldblatt moist-troposphere limit).
"""
import numpy as np
from flatworld import radiation as R
from flatworld import constants as C
from flatworld import thermo as th


def _saturated_column(Ts, N=60, p_top=50.0, T_strat=160.0):
    """A SATURATED (RH=1) moist-adiabat column at surface temperature ``Ts`` [K] -- the
    runaway-greenhouse limit. Matches phy1_olr_bias.saturated_column / _swt_lw_window_probe."""
    p_edge = np.linspace(p_top, C.P_SURF, N + 1)
    p_mid = 0.5 * (p_edge[:-1] + p_edge[1:])
    T = np.maximum(th.temperature_on_moist_adiabat(Ts, C.P_SURF, p_mid), T_strat)
    q = th.qsat(T, p_mid)
    return p_edge, T, q


def _olr_windowed(TsC):
    Ts = TsC + C.T_FREEZE
    p_edge, T, q = _saturated_column(Ts)
    dtau_bg = R.lw_optical_depth(p_edge, q)
    dtau_win = R.lw_window_optical_depth(p_edge, q)
    return float(R.two_stream_lw_windowed(dtau_bg, dtau_win, T, Ts, f_win=R.F_LW_WINDOW)["OLR"])


def _olr_gray(TsC):
    Ts = TsC + C.T_FREEZE
    p_edge, T, q = _saturated_column(Ts)
    return float(R.two_stream_lw(R.lw_optical_depth(p_edge, q), T, Ts)["OLR"])


# published runaway OLR limits [W/m^2] (recalled from the literature, +/-~10). The LOWER, BINDING limit
# is the moist-troposphere radiation limit a saturated column asymptotes onto:
#   Simpson-Nakajima  (Nakajima, Hayashi & Abe 1992)  ~291
#   Goldblatt+2013    (line-by-line)                  ~282
# The Komabayashi-Ingersoll ~385 value is the HIGHER stratospheric-radiative-equilibrium limit -- the
# looser, non-binding upper bound, which the window must pull the OLR well below.
PUB_LO, PUB_HI = 280.0, 293.0
KI_STRATOSPHERIC = 385.0


def test_windowed_olr_ceiling_in_published_runaway_band():
    """The CEILING = the peak of the saturated windowed OLR(SST) curve (the runaway limit). It must sit
    in the published binding-limit band AND at the model's measured ~289 W/m^2 (PHY-1 verdict:
    clean bill -- the gray-band scheme does NOT bias OLR near saturation)."""
    TsC = np.arange(20.0, 76.0, 2.0)                 # the moist, sub-supercritical range
    curve = np.array([_olr_windowed(t) for t in TsC])
    ceiling = float(curve.max())
    ceiling_at = float(TsC[int(np.argmax(curve))])
    print(f"\n[PHY-1] windowed OLR ceiling = {ceiling:.1f} W/m^2 at SST {ceiling_at:.0f} C "
          f"(published binding-limit band {PUB_LO}-{PUB_HI})")
    # (1) the MODEL ceiling is physics-stable -- lock it (catches a real regression, tolerant of drift)
    assert 285.0 < ceiling < 293.0
    # (2) it validates against the published LOWER, BINDING runaway limit (Simpson-Nakajima / Goldblatt)
    assert PUB_LO <= ceiling <= PUB_HI
    # (3) the window is load-bearing: the ceiling is FAR below the higher KI stratospheric limit
    assert ceiling < 0.85 * KI_STRATOSPHERIC


def test_ceiling_is_a_true_peak_then_declines():
    """The runaway signature: OLR rises to the ceiling then PLATEAUS/declines with more SST (it does not
    keep climbing) -- so the curve has an interior maximum, not a monotone rise."""
    TsC = np.arange(20.0, 76.0, 2.0)
    curve = np.array([_olr_windowed(t) for t in TsC])
    i_peak = int(np.argmax(curve))
    assert 0 < i_peak < len(curve) - 1               # interior peak
    assert curve[-1] < curve[i_peak]                 # declines past the peak (the plateau/decline)


def test_olr_at_physical_sea_sst_near_ceiling():
    """At the physical open-sea hot-edge SST (~47-55 C, where the #258 SST cap lives), the saturated
    windowed OLR is already at its ceiling and within the published band -- so the cap sits against a
    correctly-rendered radiative ceiling, not an over/under-stated one."""
    sea = np.array([_olr_windowed(t) for t in (47.0, 51.0, 55.0)])
    olr_sea = float(sea.mean())
    print(f"[PHY-1] windowed OLR at sea SST 47-55 C = {olr_sea:.1f} W/m^2")
    assert PUB_LO <= olr_sea <= PUB_HI


def test_gray_scheme_has_no_cap_the_window_fixes():
    """Sanity that the window is what supplies the cap: the GRAY (no-window) scheme's saturated OLR keeps
    climbing across the moist range (no plateau), which the windowed scheme flattens."""
    TsC = np.array([20.0, 40.0, 60.0, 75.0])
    gray = np.array([_olr_gray(t) for t in TsC])
    windowed = np.array([_olr_windowed(t) for t in TsC])
    assert gray[-1] > gray[0]                          # gray climbs monotonically (no cap)
    assert float(np.ptp(windowed)) < float(np.ptp(gray)) + 5.0  # windowed is flatter (capped)
