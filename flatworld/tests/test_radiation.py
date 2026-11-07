"""Tests for flatworld.radiation (gray-band two-stream column scheme).

The key correctness check is the analytic gray-atmosphere limit: the band model, relaxed to radiative
equilibrium, must reproduce (a) OLR = sigma T_eff^4, (b) a height-constant net LW flux, (c) the
analytic T(tau) profile, and (d) the skin temperature T_eff/2^{1/4}. Plus exact energy conservation.
"""
import numpy as np
import pytest
from flatworld import radiation as R
from flatworld import constants as C
from flatworld import thermo as th


def _column(N=60, p_top=200.0):
    return np.linspace(p_top, C.P_SURF, N + 1)


def _saturated_column(Ts, N=50, p_top=50.0, T_strat=160.0):
    """A SATURATED (RH=1) moist-adiabat column at surface temperature ``Ts`` [K] -- the runaway-greenhouse
    limit used to test the hot-end OLR cap. Returns (p_edge, T_layers, q)."""
    p_edge = np.linspace(p_top, C.P_SURF, N + 1)
    p_mid = 0.5 * (p_edge[:-1] + p_edge[1:])
    T = np.maximum(th.temperature_on_moist_adiabat(Ts, C.P_SURF, p_mid), T_strat)
    q = th.qsat(T, p_mid)                                   # saturated everywhere (RH=1)
    return p_edge, T, q


def test_lw_energy_identity_exact():
    # telescoping: sum of layer LW heating == surf_net_lw - OLR, to machine precision, ANY profile
    N = 40
    p_edge = _column(N)
    dtau = np.full(N, 0.1)
    T = np.linspace(300, 220, N)
    lw = R.two_stream_lw(dtau, T, 305.0)
    assert np.sum(lw["H"]) == pytest.approx(float(lw["surf_net_lw"] - lw["OLR"]), abs=1e-9)


def test_lw_transparent_limit():
    # zero optical depth: surface radiates straight to space, no downwelling, no atmospheric heating
    N = 30
    dtau = np.zeros(N)
    T = np.full(N, 250.0)
    lw = R.two_stream_lw(dtau, T, 300.0)
    assert lw["OLR"] == pytest.approx(C.SIGMA_SB * 300.0**4, rel=1e-9)
    assert float(lw["D_surf"]) == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(lw["H"], 0.0, atol=1e-9)


def test_gray_radiative_equilibrium_matches_analytic():
    N = 60
    p_edge = _column(N)
    T_eff, tau_surf = 255.0, 4.0
    T, Ts, tauc = R.radiative_equilibrium_gray(p_edge, T_eff, tau_surf, n_iter=60000)
    # (a) OLR = sigma T_eff^4
    dtau = tau_surf * np.diff(p_edge) / (C.P_SURF - p_edge[0])
    lw = R.two_stream_lw(dtau, T, Ts)
    assert float(lw["OLR"]) == pytest.approx(C.SIGMA_SB * T_eff**4, rel=2e-3)
    # (b) net LW flux constant with height (radiative equilibrium)
    assert np.ptp(lw["Fnet"]) < 0.05 * float(lw["OLR"])
    # (c) air-temperature profile matches the analytic gray solution
    T_analytic = R.gray_air_temperature(tauc, T_eff)
    assert np.max(np.abs(T - T_analytic)) < 1.0
    # (d) surface in equilibrium
    assert float(C.SIGMA_SB * T_eff**4 - lw["surf_net_lw"]) == pytest.approx(0.0, abs=0.5)


def test_skin_temperature():
    # the top of the radiative-equilibrium column approaches T_eff/2^{1/4}
    N = 120
    p_edge = _column(N, p_top=20.0)
    T_eff = 255.0
    T, Ts, tauc = R.radiative_equilibrium_gray(p_edge, T_eff, 3.0, n_iter=60000)
    T_skin = R.gray_air_temperature(0.0, T_eff)
    assert T[0] == pytest.approx(T_skin, abs=2.0)               # top layer near the skin temperature


def test_greenhouse_monotonic_in_optical_depth():
    p_edge = _column(60)
    Ts = [R.radiative_equilibrium_gray(p_edge, 255.0, tt, n_iter=40000)[1]
          for tt in (0.2, 1.0, 3.0, 6.0)]
    assert all(np.diff(Ts) > 0)                                 # more greenhouse -> warmer surface


def test_greenhouse_monotonic_in_water_vapor():
    # via the actual optical-depth parameterization: more q -> larger LW optical depth -> warmer
    p_edge = _column(40)
    q_dry = np.full(40, 1e-3)
    q_moist = np.full(40, 1.5e-2)
    tau_dry = R.lw_optical_depth(p_edge, q_dry).sum()
    tau_moist = R.lw_optical_depth(p_edge, q_moist).sum()
    assert tau_moist > tau_dry


def test_co2_logarithmic_forcing():
    # doubling CO2 increases the well-mixed optical depth by ~CO2_LOG_SENS*ln 2
    p_edge = _column(20)
    q = np.zeros(20)
    t1 = R.lw_optical_depth(p_edge, q, co2_ppm=R.CO2_REF_PPM).sum()
    t2 = R.lw_optical_depth(p_edge, q, co2_ppm=2 * R.CO2_REF_PPM).sum()
    assert (t2 - t1) == pytest.approx(R.TAU_LW_WELLMIXED * R.CO2_LOG_SENS * np.log(2), rel=1e-3)


def test_shortwave_energy_conservation():
    # absorbed at surface + absorbed in column + reflected to space == incoming TOA
    N = 30
    p_edge = _column(N)
    q = np.full(N, 5e-3)
    dtau_sw = R.sw_optical_depth(p_edge, q)
    sw = R.shortwave(400.0, dtau_sw, alpha_surf=0.2)
    total = float(sw["surf_abs"]) + float(np.sum(sw["H"])) + float(sw["planetary_reflected"])
    assert total == pytest.approx(400.0, rel=1e-9)


def test_vectorized_columns_match_loop():
    # the (..., N) batching must equal looping column-by-column
    N = 25
    p_edge = _column(N)
    rng = np.random.default_rng(0)
    dtau = rng.uniform(0.01, 0.2, size=(5, N))
    T = rng.uniform(220, 300, size=(5, N))
    Ts = rng.uniform(250, 320, size=5)
    batch = R.two_stream_lw(dtau, T, Ts)
    for i in range(5):
        one = R.two_stream_lw(dtau[i], T[i], Ts[i])
        assert float(batch["OLR"][i]) == pytest.approx(float(one["OLR"]), rel=1e-12)


def test_column_radiation_toa_and_surface_consistent():
    # at an arbitrary state, toa_net and surf_net are finite and the budget pieces are consistent
    N = 40
    p_edge = _column(N)
    q = np.full(N, 4e-3)
    T = np.linspace(290, 215, N)
    out = R.column_radiation(p_edge, T, q, T_surf=295.0, S_toa=350.0, alpha_surf=0.1)
    # TOA net = absorbed SW - OLR ; both should be physical (OLR ~ a few hundred W/m^2)
    assert 100 < float(out["OLR"]) < 500
    assert np.isfinite(out["toa_net"]) and np.isfinite(out["surf_net"])


# ======================================================================================
# Two-band shortwave (CLEARSKY_SW_PLAN Stage 1): transparent visible window + absorbing near-IR.
# ======================================================================================
def _wvp_column(WVP, nz=40, p_top=200.0):
    """A column whose total water-vapour path == ``WVP`` kg/m^2 (uniform specific humidity), plus its
    pressure edges -- the controlled input for the SW-transmission acceptance checks."""
    p_edge = np.linspace(p_top, C.P_SURF, nz + 1)
    dm = np.diff(p_edge) / C.G_ACCEL
    q = np.full(nz, WVP / dm.sum())
    return p_edge, q


def _clear_sky_transmission(WVP, sw_bands):
    """surf_down / S for an absorber-only column (alpha_surf=0) -- the clear-sky transmission factor."""
    p_edge, q = _wvp_column(WVP)
    k = R.K_WV_SW_NIR if sw_bands else R.K_WV_SW
    dtau = R.sw_optical_depth(p_edge, q, k)
    f_vis = R.F_VIS_SW if sw_bands else None
    sw = R.shortwave(1000.0, dtau, alpha_surf=0.0, f_vis=f_vis)
    return float(sw["surf_down"]) / 1000.0, sw


def test_sw_bands_earth_pw_transmission_is_predicted():
    # (a) Earth-PW (~30 kg/m^2) clear-sky transmission lands in 0.70-0.75 -- a PREDICTION to validate, set
    # by the spectroscopic f_vis + the single Earth-anchored near-IR coefficient (NOT fit to this test).
    t, _ = _clear_sky_transmission(30.0, sw_bands=True)
    assert 0.70 <= t <= 0.75


def test_sw_bands_high_pw_floors():
    # (b) at high (advected) water-vapour path the gray scheme -> 0 with no floor; the two-band scheme
    # FLOORS at ~40-50% because the transparent visible window passes regardless of WVP.
    for WVP in (150.0, 250.0, 400.0):
        tg, _ = _clear_sky_transmission(WVP, sw_bands=False)
        tb, _ = _clear_sky_transmission(WVP, sw_bands=True)
        assert tg < tb                                  # two-band always lets more through than gray here
        assert 0.40 <= tb <= 0.50                       # and it floors in the 40-50% band
    # the floor is f_vis*(1-alpha_atm) in the WVP->inf limit
    t_inf, _ = _clear_sky_transmission(5000.0, sw_bands=True)
    assert t_inf == pytest.approx(R.F_VIS_SW * (1.0 - R.ALBEDO_ATM), abs=2e-3)


def test_sw_bands_energy_conserved_per_call():
    # (c) reflected + atm-absorbed + surface-absorbed == S, exactly, for the two-band path at any WVP
    for WVP in (10.0, 100.0, 400.0):
        p_edge, q = _wvp_column(WVP)
        dtau = R.sw_optical_depth(p_edge, q, R.K_WV_SW_NIR)
        sw = R.shortwave(800.0, dtau, alpha_surf=0.25, f_vis=R.F_VIS_SW)
        total = float(sw["surf_abs"]) + float(np.sum(sw["H"])) + float(sw["planetary_reflected"])
        assert total == pytest.approx(800.0, rel=1e-12)


def test_sw_bands_off_is_byte_identical_to_gray():
    # (d) sw_bands=False reproduces the pre-Stage-1 gray scheme bit-for-bit, in shortwave AND through
    # column_radiation (the default must not move the 213-test gate).
    rng = np.random.default_rng(3)
    N = 35
    p_edge = np.linspace(150.0, C.P_SURF, N + 1)
    q = rng.uniform(1e-4, 2e-2, size=N)
    dtau = R.sw_optical_depth(p_edge, q)
    sw_new = R.shortwave(420.0, dtau, alpha_surf=0.2, f_vis=None)
    # reproduce the exact pre-refactor gray formula independently
    t = np.exp(-dtau); one = np.ones(1)
    Pdown = np.concatenate([one, np.cumprod(t)]); down = (420.0 * 0.9) * Pdown
    surf_down = down[-1]
    assert float(sw_new["surf_down"]) == surf_down
    # and column_radiation default (sw_bands=False) == explicit gray
    T = np.linspace(295, 215, N)
    g = R.column_radiation(p_edge, T, q, 298.0, 360.0, 0.12)
    g2 = R.column_radiation(p_edge, T, q, 298.0, 360.0, 0.12, sw_bands=False)
    assert float(g["surf_abs_sw"]) == float(g2["surf_abs_sw"])
    assert np.array_equal(g["H"], g2["H"])


def test_sw_bands_heating_deposited_where_the_water_is():
    # (e) review #72.3: moving WV absorption into the near-IR sub-band changes WHERE SW is deposited. Verify
    # against a controlled near-IR reference: with moisture ALOFT the SW heating peaks aloft; with moisture
    # LOW it peaks low; the visible band deposits NOTHING in the column (transparent). Catches a scheme that
    # gets the surface flux right but misplaces the column heating (which feeds back on lapse/convection).
    N = 40
    p_edge = np.linspace(150.0, C.P_SURF, N + 1)
    for where, sl, expect_top in (("aloft", slice(0, 10), True), ("low", slice(30, 40), False)):
        q = np.zeros(N); q[sl] = 0.02
        dtau = R.sw_optical_depth(p_edge, q, R.K_WV_SW_NIR)
        sw = R.shortwave(1000.0, dtau, alpha_surf=0.0, f_vis=R.F_VIS_SW)
        H = sw["H"]
        peak = int(np.argmax(H))
        frac_aloft = H[:10].sum() / H.sum()
        if expect_top:
            assert peak < 10 and frac_aloft > 0.9        # heating tracks the aloft moisture
        else:
            assert peak >= 30 and frac_aloft < 0.1       # heating tracks the low moisture
    # the visible band alone deposits no column heating: a dry near-IR column heats nothing either
    q0 = np.zeros(N)
    sw0 = R.shortwave(1000.0, R.sw_optical_depth(p_edge, q0, R.K_WV_SW_NIR), alpha_surf=0.0, f_vis=R.F_VIS_SW)
    assert np.allclose(sw0["H"], 0.0, atol=1e-12)


# ======================================================================================
# Stage 2 (CLEARSKY_SW_PLAN): LW window band + its humidity continuum. The acceptance criteria are
# (a) the continuum CAPS the hot-end OLR (a plateau, not just a fin); (b) an OPEN window removes the
# surface cloak but OLR still CLIMBS (2a alone insufficient); (c) reduces to gray at f_win->0;
# (d) energy conserved; (e) lw_window=False is byte-identical to the pre-Stage-2 gray scheme.
# ======================================================================================
def test_band_frac_one_is_byte_identical():
    # the new band_frac kwarg must leave the default single-band call EXACTLY unchanged
    N = 30
    dtau = np.full(N, 0.15)
    T = np.linspace(300, 220, N)
    a = R.two_stream_lw(dtau, T, 305.0)
    b = R.two_stream_lw(dtau, T, 305.0, band_frac=1.0)
    for k in ("OLR", "D_surf", "surf_net_lw"):
        assert float(a[k]) == float(b[k])               # bit-for-bit
    assert np.array_equal(a["H"], b["H"])


def test_lw_window_off_byte_identical():
    # column_radiation with lw_window=False must be byte-identical to the default (pre-Stage-2) call
    N = 40
    p_edge = _column(N)
    q = np.full(N, 5e-3)
    T = np.linspace(295, 215, N)
    base = R.column_radiation(p_edge, T, q, T_surf=300.0, S_toa=350.0, alpha_surf=0.1)
    off = R.column_radiation(p_edge, T, q, T_surf=300.0, S_toa=350.0, alpha_surf=0.1, lw_window=False)
    for k in ("OLR", "surf_net", "toa_net", "D_surf"):
        assert float(base[k]) == float(off[k])
    assert "OLR_win" not in off                          # window diagnostics only present when on


def test_lw_conserving_split_default_off_byte_identical_and_on_strengthens_greenhouse():
    """Issue #84 item 3: lw_conserving_split=False (default) is byte-identical to the accepted
    moved-baseline windowed call; ON gives the background band the out-of-window clear-sky opacity
    /(1-f_win) -> a stronger greenhouse (lower OLR) exactly where the window is open (dry column)."""
    N = 40
    p_edge = _column(N)
    q = np.full(N, 1e-4)                                 # DRY column: the window is OPEN -> max drift
    T = np.linspace(295, 215, N)
    kw = dict(T_surf=300.0, S_toa=350.0, alpha_surf=0.1, lw_window=True)
    base = R.column_radiation(p_edge, T, q, **kw)
    off = R.column_radiation(p_edge, T, q, lw_conserving_split=False, **kw)
    on = R.column_radiation(p_edge, T, q, lw_conserving_split=True, **kw)
    for k in ("OLR", "surf_net", "toa_net", "D_surf"):
        assert float(base[k]) == float(off[k])           # default == explicit OFF, bit-for-bit
    assert float(on["OLR"]) < float(off["OLR"])          # conserving -> stronger greenhouse (dry column)


def test_lw_conserving_split_restores_gray_drylimit_thin_planck_mean():
    """The POINT of the conserving split (issue #84 item 3): in the optically-THIN dry limit (open
    window, dtau_win=0) the background band at dtau/(1-f_win) restores the Earth-calibrated GRAY
    Planck-mean absorption the plain split dilutes by ~f_win. The plain split's OLR deficit vs gray
    must shrink by >=10x under the conserving split."""
    N = 40
    dtau_bg = np.full(N, 0.0005)                         # thin: column tau ~ 0.02 (first-order regime;
                                                         # the deficit ratio ~ tau, measured 0.03 here
                                                         # and ~0.11 by column tau ~ 0.08)
    dtau_win = np.zeros(N)                               # dry: open window
    T = np.linspace(290, 220, N)
    Ts = 300.0
    f = R.F_LW_WINDOW
    gray = R.two_stream_lw(dtau_bg, T, Ts)
    plain = R.two_stream_lw_windowed(dtau_bg, dtau_win, T, Ts, f_win=f)
    cons = R.two_stream_lw_windowed(dtau_bg / (1.0 - f), dtau_win, T, Ts, f_win=f)
    err_plain = abs(float(plain["OLR"]) - float(gray["OLR"]))
    err_cons = abs(float(cons["OLR"]) - float(gray["OLR"]))
    assert err_plain > 0.0
    assert err_cons < 0.1 * err_plain                    # the thin-limit Planck-mean is restored


def test_lw_window_reduces_to_gray_at_fwin_zero():
    # acceptance (c): f_win -> 0 reduces the windowed two-stream to the single gray band, EXACTLY
    N = 35
    dtau_bg = np.full(N, 0.4)
    dtau_win = np.full(N, 2.0)                            # arbitrary; weighted by f_win=0 -> must drop out
    T = np.linspace(300, 210, N)
    gray = R.two_stream_lw(dtau_bg, T, 305.0)
    win0 = R.two_stream_lw_windowed(dtau_bg, dtau_win, T, 305.0, f_win=0.0)
    assert float(win0["OLR"]) == pytest.approx(float(gray["OLR"]), rel=1e-12)
    assert float(win0["surf_net_lw"]) == pytest.approx(float(gray["surf_net_lw"]), rel=1e-12)
    assert np.allclose(win0["H"], gray["H"], atol=1e-12)


def test_lw_window_energy_conserved():
    # acceptance (d): the windowed LW telescopes exactly, sum(layer H) == surf_net_lw - OLR
    N = 45
    p_edge, T, q = _saturated_column(330.0, N=N)
    dtau_bg = R.lw_optical_depth(p_edge, q)
    dtau_win = R.lw_window_optical_depth(p_edge, q)
    lw = R.two_stream_lw_windowed(dtau_bg, dtau_win, T, 330.0, f_win=R.F_LW_WINDOW)
    assert np.sum(lw["H"]) == pytest.approx(float(lw["surf_net_lw"] - lw["OLR"]), abs=1e-7)
    # and the column-level energy bookkeeping is finite/consistent
    out = R.column_radiation(p_edge, T, q, T_surf=330.0, S_toa=400.0, alpha_surf=0.1, lw_window=True)
    assert np.isfinite(out["toa_net"]) and np.isfinite(out["surf_net"])
    assert float(out["OLR"]) == pytest.approx(float(out["OLR_bg"] + out["OLR_win"]), rel=1e-12)


def test_lw_window_optical_depth_grows_quadratically_with_humidity():
    # the self-continuum is ~quadratic in vapor: ~0 when dry (window OPEN), large when moist (window CLOSES)
    N = 30
    p_edge = _column(N, p_top=100.0)
    tau_dry = R.lw_window_optical_depth(p_edge, np.full(N, 1e-4)).sum()
    tau_mid = R.lw_window_optical_depth(p_edge, np.full(N, 1e-2)).sum()
    tau_wet = R.lw_window_optical_depth(p_edge, np.full(N, 2e-2)).sum()
    assert tau_dry < 1e-3                                 # essentially transparent when dry
    assert tau_wet > 1.0                                 # optically thick when moist
    # doubling q roughly QUADRUPLES tau (self-broadening e ~ q): ratio well above the linear 2x
    assert (tau_wet / tau_mid) > 3.0


def test_open_window_removes_cloak_but_olr_still_climbs():
    # acceptance (b): an ALWAYS-OPEN window (continuum off) lets surface emission escape (OLR > gray, the
    # cloak removed) but OLR keeps CLIMBING with Ts -- 2a alone does NOT cap.
    def olr_gray(Ts):
        p_edge, T, q = _saturated_column(Ts)
        return float(R.two_stream_lw(R.lw_optical_depth(p_edge, q), T, Ts)["OLR"])

    def olr_open_window(Ts):
        p_edge, T, q = _saturated_column(Ts)
        dtau_bg = R.lw_optical_depth(p_edge, q)
        dtau_win = np.zeros_like(dtau_bg)                 # 2a: window held OPEN (no continuum)
        return float(R.two_stream_lw_windowed(dtau_bg, dtau_win, T, Ts, f_win=R.F_LW_WINDOW)["OLR"])

    Ts_cool, Ts_hot = 288.0, 360.0
    # cloak removed: at a moderate Ts the open window emits MORE than the fully-cloaked gray scheme
    assert olr_open_window(Ts_cool) > olr_gray(Ts_cool) + 20.0
    # still climbs: the f_win*sigma*Ts^4 leak makes OLR rise strongly with Ts (no plateau)
    assert olr_open_window(Ts_hot) > olr_open_window(Ts_cool) + 40.0


def test_continuum_caps_hot_end_olr():
    # acceptance (a): with the continuum ON, OLR(Ts) for a SATURATED column PLATEAUS (a cap, not a fin).
    # The cap value is a PREDICTED outcome to validate (NOT tuned), so we assert a plateau firmly and the
    # value in a broad physical band, and REPORT the number.
    Ts_list = np.array([288.0, 303.0, 318.0, 333.0])     # 15..60 C: the moist, sub-supercritical range

    def olr_windowed(Ts):
        p_edge, T, q = _saturated_column(Ts)
        dtau_bg = R.lw_optical_depth(p_edge, q)
        dtau_win = R.lw_window_optical_depth(p_edge, q)
        return float(R.two_stream_lw_windowed(dtau_bg, dtau_win, T, Ts, f_win=R.F_LW_WINDOW)["OLR"])

    def olr_open(Ts):
        p_edge, T, q = _saturated_column(Ts)
        dtau_bg = R.lw_optical_depth(p_edge, q)
        return float(R.two_stream_lw_windowed(dtau_bg, np.zeros_like(dtau_bg), T, Ts,
                                              f_win=R.F_LW_WINDOW)["OLR"])

    capped = np.array([olr_windowed(t) for t in Ts_list])
    fin = np.array([olr_open(t) for t in Ts_list])
    span_capped = float(capped.max() - capped.min())
    span_fin = float(fin.max() - fin.min())
    print(f"\n[Stage 2 cap] windowed OLR over 15..60C = {np.round(capped, 1)} W/m^2 "
          f"(span {span_capped:.1f}); open-window fin span {span_fin:.1f}")
    # the continuum FLATTENS the curve: the capped span is small AND far below the open-window fin
    assert span_capped < 30.0                             # a plateau (not a fin)
    assert span_capped < 0.4 * span_fin                  # and dramatically flatter than the open window
    # the predicted cap lands in the physical Komabayashi-Ingersoll / Simpson-Nakajima neighborhood
    cap = float(capped.mean())
    assert 250.0 < cap < 340.0                            # validate (not tune); ~280-310 is the literature band


# ======================================================================================
# Stage 5 (CLEARSKY_SW_PLAN): SW diffuse-down stream + haze forward-scatter. Acceptance:
# (a) energy conserved counting the diffuse input; (b) the diffuse stream RAISES surface SW;
# (c) S_diffuse=None / sw_diffuse=False is byte-identical; (d) the TOA budget counts the diffuse in.
# ======================================================================================
def test_sw_diffuse_off_byte_identical():
    # (c) S_diffuse=None reproduces the pre-Stage-5 shortwave bit-for-bit (gray AND two-band), and
    # column_radiation sw_diffuse=False ignores S_diffuse entirely (byte-identical to the default).
    rng = np.random.default_rng(5)
    N = 30
    p_edge = np.linspace(150.0, C.P_SURF, N + 1)
    q = rng.uniform(1e-4, 2e-2, size=N)
    dtau = R.sw_optical_depth(p_edge, q)
    for f_vis in (None, R.F_VIS_SW):
        a = R.shortwave(500.0, dtau, alpha_surf=0.2, f_vis=f_vis)
        b = R.shortwave(500.0, dtau, alpha_surf=0.2, f_vis=f_vis, S_diffuse=None)
        for k in ("surf_abs", "surf_down", "planetary_reflected"):
            assert np.array_equal(np.asarray(a[k]), np.asarray(b[k]))
        assert np.array_equal(a["H"], b["H"])
    T = np.linspace(295, 215, N)
    g1 = R.column_radiation(p_edge, T, q, 298.0, 360.0, 0.12)
    g2 = R.column_radiation(p_edge, T, q, 298.0, 360.0, 0.12, sw_diffuse=False, S_diffuse=120.0)
    for k in ("surf_abs_sw", "toa_net", "planetary_reflected", "surf_net"):
        assert float(g1[k]) == float(g2[k])             # off -> S_diffuse ignored, byte-identical


def test_sw_diffuse_energy_conserved():
    # (a) surf_abs + sum(H) + planetary_reflected == S_toa + S_diffuse, exactly, gray AND two-band, any WVP
    for WVP in (10.0, 100.0, 400.0):
        p_edge, q = _wvp_column(WVP)
        for f_vis in (None, R.F_VIS_SW):
            k = R.K_WV_SW if f_vis is None else R.K_WV_SW_NIR
            dtau = R.sw_optical_depth(p_edge, q, k)
            sw = R.shortwave(800.0, dtau, alpha_surf=0.25, f_vis=f_vis, S_diffuse=300.0)
            total = float(sw["surf_abs"]) + float(np.sum(sw["H"])) + float(sw["planetary_reflected"])
            assert total == pytest.approx(800.0 + 300.0, rel=1e-12)


def test_sw_diffuse_raises_surface_sw():
    # (b) the diffuse-down stream adds surface SW: surf_down and surf_abs RISE vs the direct-only call, at
    # every WVP (the recovered haze light reaches the surface). The visible part passes regardless of WVP.
    for WVP in (30.0, 150.0, 400.0):
        p_edge, q = _wvp_column(WVP)
        dtau = R.sw_optical_depth(p_edge, q, R.K_WV_SW_NIR)
        base = R.shortwave(700.0, dtau, alpha_surf=0.1, f_vis=R.F_VIS_SW)
        diff = R.shortwave(700.0, dtau, alpha_surf=0.1, f_vis=R.F_VIS_SW, S_diffuse=250.0)
        assert float(diff["surf_down"]) > float(base["surf_down"])
        assert float(diff["surf_abs"]) > float(base["surf_abs"])


def test_sw_diffuse_column_radiation_counts_diffuse_in_toa_budget():
    # through column_radiation: sw_diffuse=True raises the surface SW AND counts the diffuse input in the TOA
    # SW budget (absorbed SW = S_toa + S_diffuse - reflected), so toa_net rises by the absorbed diffuse and is
    # not left dangling. The rise is BOUNDED by the diffuse input (energy can't be created).
    N = 40
    p_edge = _column(N)
    q = np.full(N, 8e-3)
    T = np.linspace(295, 215, N)
    base = R.column_radiation(p_edge, T, q, 298.0, 360.0, 0.1, sw_bands=True)
    diff = R.column_radiation(p_edge, T, q, 298.0, 360.0, 0.1, sw_bands=True,
                              sw_diffuse=True, S_diffuse=200.0)
    assert float(diff["surf_abs_sw"]) > float(base["surf_abs_sw"])     # more surface SW absorbed
    assert float(diff["surf_down_sw"]) > float(base["surf_down_sw"])
    assert float(diff["surf_net"]) > float(base["surf_net"])           # surface net radiation rises
    d_toa = float(diff["toa_net"]) - float(base["toa_net"])
    assert 0.0 < d_toa < 200.0                                         # counted in, bounded by the input


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
