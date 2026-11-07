"""Tests for radiatively-active clouds (GitHub issue #19).

Two layers: (1) the reusable radiative-effect MACHINERY in ``radiation.py`` -- cloud fraction from RH,
cloud LW optical depth (greenhouse, warming) + a column SW albedo (cooling), energy conservation, the
overlap assumption, and the HIGH-vs-LOW asymmetry (the crux: low warm cloud net-cools, high cold cloud
net-warms); and (2) the CORE INTEGRATION in ``stage_ad_core.py`` -- clouds default OFF (byte-identical
clear sky), the env knobs, the cloud diagnostics, the cloud radiative effect, and the energy gate with
clouds on.

The cloud parameters are documented INPUTS swept (not fitted); these tests assert SIGNS, DIRECTIONS, and
CONSERVATION (firm), not absolute magnitudes (soft -- #19 is the 'unbounded' uncertainty).
"""
import os
import numpy as np
import pytest
from flatworld import constants as C
from flatworld import thermo as th
from flatworld import radiation as rad
from flatworld.radiation import (cloud_fraction, cloud_optical_properties, cloud_sw_albedo,
                                 cloud_water_path, column_radiation, lw_optical_depth, sw_optical_depth,
                                 two_stream_lw, two_stream_lw_windowed, lw_window_optical_depth,
                                 shortwave, layer_mass, two_stream_nc)
from flatworld.stage_ad_core import StageADCore, StageADConfig


# ----------------------------------------------------------------------------------------------------
# A representative single column (warm surface, cooling aloft) for the machinery tests.
# ----------------------------------------------------------------------------------------------------
def _column(nz=24, T_surf=300.0, S=350.0, rh=0.8):
    p_edge = np.linspace(5.0e3, C.P_SURF, nz + 1)
    p_mid = 0.5 * (p_edge[:-1] + p_edge[1:])
    sigma = p_mid / C.P_SURF
    z = -8000.0 * np.log(sigma)
    T = np.maximum(T_surf - 6.5e-3 * z, 150.0)
    q = rh * th.qsat(T, p_mid)
    return p_edge, p_mid, T, q, float(T_surf), float(S)


# ====================================================================================================
# Cloud-fraction diagnostic
# ====================================================================================================
def test_cloud_fraction_is_an_rh_continuum():
    """Cloud fraction rises from 0 below rh_crit to 1 at saturation, monotone in RH (the Sundqvist
    diagnostic on the principled RH continuum) -- 0 in dry air, 1 at/over saturation, and graded between."""
    _, p_mid, T, _, _, _ = _column()
    qs = th.qsat(T, p_mid)
    f_dry = cloud_fraction(0.3 * qs, T, p_mid, rh_crit=0.7)      # RH 0.3 < crit
    f_mid = cloud_fraction(0.85 * qs, T, p_mid, rh_crit=0.7)     # RH 0.85, between crit and 1
    f_sat = cloud_fraction(1.0 * qs, T, p_mid, rh_crit=0.7)      # RH 1.0
    assert np.all(f_dry == 0.0)                                  # below crit -> clear
    assert np.all((f_mid > 0.0) & (f_mid < 1.0))                # graded
    assert np.allclose(f_sat, 1.0)                              # saturated -> overcast
    assert np.all(f_sat >= f_mid) and np.all(f_mid >= f_dry)     # monotone in RH


def test_cloud_fraction_convective_boost_increases_cloud():
    """The optional resolved-ascent convective boost nudges cloud fraction toward overcast where ``conv``
    is positive (sub-grid convective cloud the grid-mean RH misses) and is inert where conv=0 or
    conv_boost=0 (the default)."""
    _, p_mid, T, _, _, _ = _column()
    qs = th.qsat(T, p_mid)
    q = 0.75 * qs                                               # RH 0.75, partial cloud
    base = cloud_fraction(q, T, p_mid, rh_crit=0.7)
    conv = np.ones_like(q)                                      # everywhere ascending
    boosted = cloud_fraction(q, T, p_mid, rh_crit=0.7, conv=conv, conv_boost=0.5)
    assert np.all(boosted >= base - 1e-12) and np.any(boosted > base)   # boost adds cloud
    inert = cloud_fraction(q, T, p_mid, rh_crit=0.7, conv=conv, conv_boost=0.0)
    assert np.allclose(inert, base)                            # conv_boost=0 -> pure RH (default behaviour)


# ====================================================================================================
# Cloud optical properties + the SW overlap albedo
# ====================================================================================================
def test_cloud_sw_albedo_bounds_and_overlap():
    """The column SW cloud albedo is in [0,1], is 0 with no cloud, equals R_in for a single overcast deck,
    and is monotone non-decreasing in cloud fraction (random-overlap reflection probability)."""
    nz = 10
    f0 = np.zeros(nz)
    assert cloud_sw_albedo(f0, np.full(nz, 0.5)) == pytest.approx(0.0)
    f1 = np.zeros(nz); f1[3] = 1.0                              # one overcast deck, R_in=0.5
    assert cloud_sw_albedo(f1, np.full(nz, 0.5)) == pytest.approx(0.5)
    a_lo = cloud_sw_albedo(np.full(nz, 0.2), np.full(nz, 0.5))
    a_hi = cloud_sw_albedo(np.full(nz, 0.6), np.full(nz, 0.5))
    assert 0.0 <= a_lo <= a_hi <= 1.0                          # bounded + monotone in cloud fraction


def test_cloud_sw_albedo_column_overlap_below_random():
    """``overlap='column'`` (a single two-stream on the column-TOTAL scattering optical depth, the LW limb's
    footing) is LOWER than ``overlap='random'`` (the per-layer reflection-probability product) for a stacked
    saturated column -- the random product's inflation (CLOUDS_RESULTS.md §4b). They AGREE for a single
    overcast deck (no stacking), and the tau_sw-supplied result matches the R_in back-out."""
    nz = 20
    g = rad.CLOUD_G_ASYM
    k = 2.0 / (1.0 - g)
    # a contiguous saturated deck: 20 layers, each f=1, modest per-layer reflectance R_in~0.3
    R_in = np.full(nz, 0.30)
    f = np.ones(nz)
    A_rand = float(cloud_sw_albedo(f, R_in, overlap="random"))
    A_col = float(cloud_sw_albedo(f, R_in, overlap="column"))
    assert A_rand > 0.99                                        # the product over 20 layers saturates ~1
    assert A_col < A_rand                                       # column-total two-stream is far lower
    # the merged-deck answer = one two-stream on the summed tau:  tau = R_in*k/(1-R_in) per layer
    tau = R_in * k / (1.0 - R_in)
    tau_tot = np.sum(f * tau)
    assert A_col == pytest.approx(tau_tot / (tau_tot + k))
    # passing tau_sw explicitly matches backing it out of R_in
    assert cloud_sw_albedo(f, R_in, overlap="column", tau_sw=tau) == pytest.approx(A_col)
    # single overcast deck: no stacking, the two modes agree (both = R_in)
    f1 = np.zeros(nz); f1[5] = 1.0
    assert (cloud_sw_albedo(f1, R_in, overlap="column")
            == pytest.approx(cloud_sw_albedo(f1, R_in, overlap="random"), abs=1e-9))


def test_cloud_albedo_earth_marine_stratocumulus_anchor():
    """EXTERNAL anchor: the optics chain (LWP, r_eff -> tau -> R_in -> column albedo) must reproduce an
    OBSERVED cloud albedo for an OBSERVED cloud, independent of the flatworld climate. Marine stratocumulus,
    LWP ~100 g/m^2, r_eff ~10 um -> tau ~15 -> observed albedo ~0.5-0.7. A thicker deck (LWP 300) is brighter
    but still physical (< 0.9). This pins the model's tau formula + asymmetry g (0.85) against observation."""
    r_eff = 10.0e-6
    def deck_albedo(lwp):
        tau = 1.5 * lwp / (C.RHO_LIQ * r_eff)                   # the documented geometric-optics extinction
        R_in = tau / (tau + 2.0 / (1.0 - rad.CLOUD_G_ASYM))     # the model's conservative two-stream
        return float(cloud_sw_albedo(np.array([1.0]), np.array([R_in])))   # single overcast deck
    A_sc = deck_albedo(0.10)                                    # 100 g/m^2 marine stratocumulus
    A_thick = deck_albedo(0.30)                                 # 300 g/m^2 thick deck
    assert 0.45 < A_sc < 0.72, f"marine-sc albedo {A_sc:.3f} outside observed ~0.5-0.7"
    assert A_sc < A_thick < 0.92, f"thick-deck albedo {A_thick:.3f} unphysical"


def test_cloud_sw_absorption_lowers_albedo_and_absorbs_in_column():
    """Cloud SW absorption (ssa < 1) moves part of the cloud extinction from SCATTERING to ABSORPTION: vs
    reflect-only (ssa=1) it LOWERS the cloud reflection albedo and ADDS in-column SW absorption, and ssa=1 is
    BYTE-IDENTICAL to the historic reflect-only scheme. (The SURFACE-light direction is deliberately NOT
    asserted: it is regime-dependent -- a lower albedo lets more light in [brighter], in-column absorption
    removes some [darker]; for a thin deck the first dominates, for a thick ribbon deck the second does. That
    net is a converged-run question, which is why cloud SW absorption is tiered as a model change.) Run
    through ``column_radiation`` (the integration point)."""
    p_edge, p_mid, T, q, T_surf, S = _column(rh=1.0)            # saturated -> a real cloud deck
    common = dict(clouds=True, sw_bands=True)
    val = lambda out, key: float(np.ravel(out[key])[0])         # (1,)-column output -> scalar
    out_reflect = column_radiation(p_edge[None, :], T[None, :], q[None, :], T_surf, S, 0.1, **common)
    # ssa=1 explicitly must reproduce the default (reflect-only) exactly
    out_ssa1 = column_radiation(p_edge[None, :], T[None, :], q[None, :], T_surf, S, 0.1,
                                cloud_sw_ssa_liq=1.0, cloud_sw_ssa_ice=1.0, **common)
    assert val(out_ssa1, "cloud_sw_albedo") == pytest.approx(val(out_reflect, "cloud_sw_albedo"), abs=1e-12)
    assert val(out_ssa1, "surf_down_sw") == pytest.approx(val(out_reflect, "surf_down_sw"), abs=1e-9)
    # ssa<1: less reflection (lower albedo) + more SW absorbed in-column (the robust signatures)
    out_abs = column_radiation(p_edge[None, :], T[None, :], q[None, :], T_surf, S, 0.1,
                               cloud_sw_ssa_liq=0.90, cloud_sw_ssa_ice=0.90, **common)
    assert val(out_abs, "cloud_sw_albedo") < val(out_reflect, "cloud_sw_albedo")   # less reflected
    # column SW absorption = S - reflected - surf_abs must EXCEED the reflect-only column absorption
    col_abs_on = S - val(out_abs, "planetary_reflected") - val(out_abs, "surf_abs_sw")
    col_abs_off = S - val(out_reflect, "planetary_reflected") - val(out_reflect, "surf_abs_sw")
    assert col_abs_on > col_abs_off                            # the cloud now absorbs SW in-column


# ====================================================================================================
# NON-CONSERVATIVE cloud SW two-stream (two_stream_nc + COSMO_CLOUD_SW_TWOSTREAM): the burn-off driver.
# ====================================================================================================
def test_two_stream_nc_limits_and_energy():
    """The non-conservative two-stream (R, A_abs, T) must: (a) reduce to the reflect-only column albedo
    ``tau_scat/(tau_scat+2/(1-g))`` with A_abs=0 in the conservative (tau_abs=0) limit -- BYTE-IDENTICAL,
    the closed form lifted inline (not a hand estimate); (b) give A_abs ~ tau_abs for a thin cloud (single
    absorption); (c) saturate to R<1, 0<A_abs<1 for a thick deck; and (d) close energy R+A_abs+T=1 exactly
    over a grid of (tau, ssa)."""
    g = rad.CLOUD_G_ASYM
    # (a) conservative limit: exact match to cloud_sw_albedo's column formula, A_abs == 0
    tau_s = np.array([0.5, 5.0, 50.0, 200.0])
    R, A, T = two_stream_nc(tau_s, np.zeros_like(tau_s), g)
    R_cons = tau_s / (tau_s + 2.0 / (1.0 - g))                  # the model's conservative two-stream (closed form)
    assert np.array_equal(R, R_cons)                           # byte-identical reflectance
    assert np.array_equal(T, 1.0 - R_cons)
    assert np.all(A == 0.0)
    # (b) thin cloud: absorptance approaches the absorbing optical depth
    ta = np.array([1e-3, 1e-2])
    _, A_thin, _ = two_stream_nc(ta.copy(), ta.copy(), g)      # tau_scat = tau_abs, both small
    assert A_thin == pytest.approx(ta, rel=0.02)
    # (c) thick deck at ssa<1: bounded, reflection capped below unity, absorptance positive
    for ssa in (0.99, 0.95):
        Rk, Ak, Tk = two_stream_nc(np.array([ssa * 50.0]), np.array([(1.0 - ssa) * 50.0]), g)
        assert 0.0 < Rk[0] < 1.0 and 0.0 < Ak[0] < 1.0 and Tk[0] >= 0.0
    # (d) energy closes exactly over a grid
    ts = np.array([0.1, 1.0, 10.0, 100.0])
    for ssa in (1.0, 0.999, 0.99, 0.9, 0.5):
        Rg, Ag, Tg = two_stream_nc(ssa * ts, (1.0 - ssa) * ts, g)
        assert np.max(np.abs(Rg + Ag + Tg - 1.0)) < 1e-12


def test_two_stream_nc_absorptance_monotone_in_ssa():
    """Lowering the single-scattering albedo (more absorbing cloud) must MONOTONICALLY raise the absorptance
    and lower the reflectance at fixed extinction -- the physical direction (the burn-off knob)."""
    g = rad.CLOUD_G_ASYM
    tau_ext = 40.0
    ssas = np.array([1.0, 0.995, 0.99, 0.97, 0.95, 0.90])
    sc = lambda x: float(np.ravel(x)[0])
    R = np.array([sc(two_stream_nc(np.array([s * tau_ext]), np.array([(1 - s) * tau_ext]), g)[0]) for s in ssas])
    A = np.array([sc(two_stream_nc(np.array([s * tau_ext]), np.array([(1 - s) * tau_ext]), g)[1]) for s in ssas])
    assert np.all(np.diff(A) > 0)                              # A_abs rises as ssa falls
    assert np.all(np.diff(R) < 0)                              # R falls as ssa falls
    assert A[0] == 0.0                                         # reflect-only at ssa=1


def test_cloud_sw_twostream_byte_identical_at_ssa1():
    """REGRESSION GUARD (the second byte-identical guarantee): turning the two-stream ON with ssa=1 (A_abs=0)
    is bit-identical to the reflect-only (flag OFF) column, so the flag is a true no-op until ssa<1."""
    p_edge, _, T, q, Ts, S = _column(rh=1.0)
    common = dict(clouds=True, sw_bands=True, cloud_sw_overlap="column")
    off = column_radiation(p_edge[None, :], T[None, :], q[None, :], Ts, S, 0.1, **common)             # flag OFF
    on1 = column_radiation(p_edge[None, :], T[None, :], q[None, :], Ts, S, 0.1,
                           cloud_sw_twostream=True, cloud_sw_ssa_liq=1.0, cloud_sw_ssa_ice=1.0, **common)
    for k in ("H", "surf_net", "toa_net", "surf_abs_sw", "planetary_reflected", "cloud_sw_albedo", "OLR"):
        assert np.array_equal(np.asarray(off[k]), np.asarray(on1[k])), f"{k} not byte-identical"


def test_cloud_sw_twostream_heats_the_deck_and_reflects_less():
    """With the two-stream ON and ssa<1, the cloud ABSORBS SW in the near-IR band: vs reflect-only it reports a
    positive column ``cloud_sw_absorptance``, deposits MORE atmospheric SW heating (the deck warms -- the
    burn-off driver), and reflects LESS to space (the near-IR reflects below the conservative albedo). Direction
    only (magnitude is ssa-soft). NB: the surface is deliberately NOT asserted to dim -- cloud SW absorption is
    near-IR-CONFINED (the visible band stays conservative), so the surface floor is PRESERVED, not collapsed;
    that is the separate ``test_cloud_sw_twostream_preserves_visible_floor``. The earlier grey-spectral limb ate
    the visible floor and drove the surface ~10^2-10^3x too dark -- the bug this fix removes."""
    p_edge, _, T, q, Ts, S = _column(rh=1.0)
    common = dict(clouds=True, sw_bands=True, cloud_sw_overlap="column")
    val = lambda out, key: float(np.ravel(out[key])[0])
    reflect = column_radiation(p_edge[None, :], T[None, :], q[None, :], Ts, S, 0.1,
                               cloud_sw_twostream=True, cloud_sw_ssa_liq=1.0, cloud_sw_ssa_ice=1.0, **common)
    absb = column_radiation(p_edge[None, :], T[None, :], q[None, :], Ts, S, 0.1,
                            cloud_sw_twostream=True, cloud_sw_ssa_liq=0.95, cloud_sw_ssa_ice=0.95, **common)
    assert "cloud_sw_absorptance" in absb and val(absb, "cloud_sw_absorptance") > 0.0
    assert val(absb, "cloud_sw_albedo") < val(reflect, "cloud_sw_albedo")            # less reflected (near-IR)
    assert val(absb, "planetary_reflected") < val(reflect, "planetary_reflected")    # less to space
    col_abs_on = S - val(absb, "planetary_reflected") - val(absb, "surf_abs_sw")     # column SW absorption
    col_abs_off = S - val(reflect, "planetary_reflected") - val(reflect, "surf_abs_sw")
    assert col_abs_on > col_abs_off                                                  # the deck now absorbs SW


def test_cloud_sw_twostream_preserves_visible_floor():
    """THE FIX (near-IR-confined cloud SW absorption). A THICK saturated deck under the non-conservative
    two-stream (ssa<1). Real clouds are ~non-absorbing in the visible (they reflect it); cloud SW absorption is
    near-IR. The grey-spectral limb instead absorbed the water-transparent VISIBLE band too, collapsing surface
    light ~10^2-10^3x (surf_down toward ~0 -- the reported bug). With the band split, the visible band is
    conservative, so surface light stays within a factor of ~2 of the reflect-only deck -- a physical
    dim-overcast level, NOT ~0. (Pre-fix this ratio was ~0.005; the assert below fails on the grey limb.)"""
    p_edge, _, T, q, Ts, S = _column(rh=1.0)
    common = dict(clouds=True, sw_bands=True, cloud_sw_twostream=True, cloud_sw_overlap="column",
                  cloud_water_scale=0.5)
    val = lambda out, key: float(np.ravel(out[key])[0])
    reflect = column_radiation(p_edge[None, :], T[None, :], q[None, :], Ts, S, 0.1,
                               cloud_sw_ssa_liq=1.0, cloud_sw_ssa_ice=1.0, **common)
    absb = column_radiation(p_edge[None, :], T[None, :], q[None, :], Ts, S, 0.1,
                            cloud_sw_ssa_liq=0.95, cloud_sw_ssa_ice=0.95, **common)
    assert val(reflect, "surf_down_sw") > 5.0                                        # reflect-only floor is healthy
    # the visible floor survives the absorbing deck: surf light NOT collapsed by orders of magnitude.
    # Pre-fix (grey pre-split carve) this ratio is ~0.0009; the near-IR-confined fix keeps it ~0.8.
    assert val(absb, "surf_down_sw") > 0.3 * val(reflect, "surf_down_sw")


def test_shortwave_cloud_absorption_conserves_energy():
    """``shortwave`` with the cloud-absorption limb (cloud_abs/cloud_w) still closes the SW budget exactly for
    BOTH the direct and the diffuse streams: surf_abs + sum(H) + planetary_reflected == S_toa (+ S_diffuse)."""
    nz = 12
    S_toa = np.array([1500.0])
    dtau_sw = np.linspace(0.0, 0.05, nz)[None, :]
    alpha_atm_eff = np.array([0.10 + 0.90 * 0.6])              # Rayleigh + a bright cloud reflection R
    cloud_abs = np.array([0.25])                               # fraction of the entering beam the cloud absorbs
    w = np.zeros((1, nz)); w[0, 4:8] = 0.25                    # deposit in the mid-column "deck", sum_k = 1
    for f_vis in (None, 0.5):                                  # gray + two-band
        for S_diffuse in (None, np.array([120.0])):           # direct-only + with the diffuse stream
            sw = shortwave(S_toa, dtau_sw, np.array([0.08]), alpha_atm_eff,
                           f_vis=f_vis, S_diffuse=S_diffuse, cloud_abs=cloud_abs, cloud_w=w)
            total_in = float(S_toa[0]) + (float(S_diffuse[0]) if S_diffuse is not None else 0.0)
            budget = float(np.ravel(sw["surf_abs"])[0] + np.sum(sw["H"]) + np.ravel(sw["planetary_reflected"])[0])
            assert budget == pytest.approx(total_in, rel=1e-9)
    # the None default is byte-identical to omitting the cloud limb
    base = shortwave(S_toa, dtau_sw, np.array([0.08]), alpha_atm_eff, f_vis=0.5)
    none = shortwave(S_toa, dtau_sw, np.array([0.08]), alpha_atm_eff, f_vis=0.5, cloud_abs=None, cloud_w=None)
    assert np.array_equal(base["H"], none["H"]) and np.array_equal(base["surf_abs"], none["surf_abs"])


def test_shortwave_two_band_cloud_confines_absorption_to_near_ir():
    """``shortwave`` two-band cloud optics (cloud_R_vis/cloud_R_nir set): the VISIBLE band is conservative
    (reflects R_vis, NO cloud absorption) and the NEAR-IR band carries the cloud absorption. So even a cloud
    that absorbs nearly ALL the near-IR it admits CANNOT drive the surface to ~0 -- the visible floor
    ``S*f_vis*(1-alpha_atm)*(1-R_vis)`` reaches the ground intact (visible is water-transparent, t==1). This is
    the near-IR confinement that replaces the grey pre-split carve. Energy closes per band, with/without the
    diffuse stream. cloud_R_vis=None -> the grey path (byte-identical, covered elsewhere)."""
    nz = 12
    S = np.array([1200.0])
    dtau = np.linspace(0.0, 0.03, nz)[None, :]
    alpha_atm = 0.10                                   # CLEAR-sky Rayleigh (cloud reflection passed per band)
    fvis = 0.5
    Rvis = np.array([0.85])                            # thick deck: conservative visible reflects strongly
    Rnir = np.array([0.55])                            # near-IR reflects less (non-conservative)
    caf = np.array([0.95])                             # near-IR: the cloud absorbs almost all it admits
    w = np.zeros((1, nz)); w[0, 4:8] = 0.25            # deposit in the mid-column deck, sum_k = 1
    vis_floor = float(S[0]) * fvis * (1.0 - alpha_atm) * (1.0 - float(Rvis[0]))   # analytic surface visible floor
    for S_diffuse in (None, np.array([200.0])):
        sw = shortwave(S, dtau, np.array([0.10]), alpha_atm, f_vis=fvis,
                       cloud_R_vis=Rvis, cloud_R_nir=Rnir, cloud_abs=caf, cloud_w=w, S_diffuse=S_diffuse)
        surf_down = float(np.ravel(sw["surf_down"])[0])
        floor = vis_floor if S_diffuse is None else vis_floor * (1.0 + float(S_diffuse[0]) / float(S[0]))
        assert surf_down >= 0.999 * floor              # the visible floor survives the absorbing deck (NOT ~0)
        total_in = float(S[0]) + (float(S_diffuse[0]) if S_diffuse is not None else 0.0)
        budget = float(np.ravel(sw["surf_abs"])[0] + np.sum(sw["H"]) + np.ravel(sw["planetary_reflected"])[0])
        assert budget == pytest.approx(total_in, rel=1e-9)   # energy closes per band, diffuse counted in


def test_cloud_optics_thicker_for_low_warm_than_high_cold():
    """The cloud optical depth is COMPUTED from the physical condensed-water path -- so a single LOW warm
    deck (large qsat drop -> more condensate, liquid, small droplets) is optically THICKER (more SW
    reflectance AND more LW opacity) than a single HIGH cold deck (little water, ice, large crystals). The
    asymmetry now emerges from physics, not a height switch."""
    p_edge, p_mid, T, _, _, _ = _column()
    dm = layer_mass(p_edge)
    k_top, k_bot = 2, len(p_mid) - 3                           # a high (cold) vs a low (warm) single deck
    def deck(k):
        f = np.zeros_like(p_mid); f[k] = 1.0
        lwp, ice = cloud_water_path(f, T, p_mid, dm)
        dtau_lw, R_in = cloud_optical_properties(f, T, p_mid, dm)
        return float(lwp[k]), float(R_in[k]), float(dtau_lw[k])
    lwp_hi, R_hi, lw_hi = deck(k_top)
    lwp_lo, R_lo, lw_lo = deck(k_bot)
    assert lwp_lo > lwp_hi                                     # low warm cloud holds more condensed water
    assert R_lo > R_hi                                        # ... so it is more SW-reflective
    assert lw_lo > lw_hi                                      # ... and more LW-opaque (both from the water path)


# ====================================================================================================
# Radiative effect: LW warms, SW cools, energy conserved
# ====================================================================================================
def test_clouds_off_is_byte_identical_to_clear_sky():
    """REGRESSION GUARD: ``column_radiation`` with clouds off (the default) is bit-identical to the
    pre-cloud clear-sky call, and carries NO cloud keys -- so Stage A and the clouds-disabled core are
    unchanged byte-for-byte."""
    p_edge, _, T, q, Ts, S = _column()
    a = column_radiation(p_edge, T, q, Ts, S, 0.08, co2_ppm=400.0)                 # default clouds=False
    b = column_radiation(p_edge, T, q, Ts, S, 0.08, co2_ppm=400.0, clouds=False)   # explicit
    for k in ("OLR", "toa_net", "surf_net", "planetary_reflected", "D_surf"):
        assert np.array_equal(a[k], b[k])
    assert "cloud_cover" not in a and "cloud_fraction" not in a


def test_cloud_lw_reduces_olr_warming():
    """The cloud LW opacity is a GREENHOUSE term: adding cloud reduces the OLR (and raises the downwelling
    LW at the surface) -- the warming limb. The OLR responds ONLY to the LW (the cloud SW albedo does not
    enter the OLR), so it isolates the greenhouse without any knob."""
    p_edge, p_mid, T, q, Ts, S = _column()
    clear = column_radiation(p_edge, T, q, Ts, S, 0.08, clouds=False)
    cloudy = column_radiation(p_edge, T, q, Ts, S, 0.08, clouds=True, cloud_rh_crit=0.5)
    assert cloudy["OLR"] < clear["OLR"]                         # cloud greenhouse cuts the OLR (warming)
    assert cloudy["D_surf"] > clear["D_surf"]                   # ... and adds downwelling LW at the surface


def test_cloud_sw_increases_reflection_cooling():
    """The cloud SW albedo is a COOLING term: clouds reflect more SW to space (planetary_reflected up,
    surface-absorbed SW down). The planetary reflection responds ONLY to the SW (the cloud LW does not
    enter it), so it isolates the albedo without any knob."""
    p_edge, _, T, q, Ts, S = _column()
    clear = column_radiation(p_edge, T, q, Ts, S, 0.08, clouds=False)
    cloudy = column_radiation(p_edge, T, q, Ts, S, 0.08, clouds=True, cloud_rh_crit=0.5)
    assert cloudy["planetary_reflected"] > clear["planetary_reflected"]   # more SW reflected (cooling)
    assert cloudy["surf_abs_sw"] < clear["surf_abs_sw"]                    # less SW reaches the surface


def test_cloud_radiation_conserves_shortwave_energy():
    """SW energy closes WITH clouds: incoming S == planetary_reflected + atmospheric SW absorption +
    surface SW absorption (the cloud reflection just raises the planetary albedo, the reflected SW never
    enters). The cloud LW is energy-conserving by the two-stream construction (tested in test_radiation)."""
    p_edge, _, T, q, Ts, S = _column()
    out = column_radiation(p_edge, T, q, Ts, S, 0.08, clouds=True, cloud_rh_crit=0.5)
    dtau_sw = sw_optical_depth(p_edge, q)
    sw = shortwave(S, dtau_sw, 0.08, alpha_atm=0.10 + 0.90 * out["cloud_sw_albedo"])
    atm_abs = float(np.sum(sw["H"]))                            # SW absorbed in the column
    budget = out["planetary_reflected"] + atm_abs + out["surf_abs_sw"]
    assert float(budget) == pytest.approx(S, rel=1e-9)         # SW energy conserved with clouds


def test_high_cloud_warms_relative_to_low_cloud_cools_the_crux():
    """THE CRUX (issue #19): the net cloud forcing rises monotonically with cloud HEIGHT -- a LOW (warm)
    cloud is net-COOLING (SW albedo of a water-rich, small-droplet deck dominates) while a HIGH (cold) cloud
    is net-WARMER (a water-poor, large-crystal ice deck reflects little SW, and its cold top cuts the OLR --
    the greenhouse limb). All from the PHYSICAL water path (no height switch). Sweep an equal-water deck up
    the column and confirm the net forcing increases with height, with the low deck cooling. The water is
    scaled up so even the thin high deck is radiatively active (isolating the height effect from the
    water-amount effect)."""
    p_edge, p_mid, T, q, Ts, S = _column(T_surf=305.0)
    dm = layer_mass(p_edge)
    dtau_lw_clear = lw_optical_depth(p_edge, q)
    dtau_sw = sw_optical_depth(p_edge, q)
    olr_clear = float(two_stream_lw(dtau_lw_clear, T, Ts)["OLR"])
    refl_clear = float(shortwave(S, dtau_sw, 0.08, 0.10)["planetary_reflected"])

    def net_cre(k_cloud):
        f = np.zeros_like(p_mid); f[k_cloud] = 1.0             # one overcast deck at level k
        dtau_lw_cl, R_in = cloud_optical_properties(f, T, p_mid, dm, water_scale=3.0)   # boost so high deck is active
        olr = float(two_stream_lw(dtau_lw_clear + dtau_lw_cl, T, Ts)["OLR"])
        A = float(cloud_sw_albedo(f, R_in))
        refl = float(shortwave(S, dtau_sw, 0.08, 0.10 + 0.90 * A)["planetary_reflected"])
        return (refl_clear - refl) + (olr_clear - olr), (refl_clear - refl), (olr_clear - olr)

    net_high, sw_high, lw_high = net_cre(2)                    # a high (cold) deck
    net_low, sw_low, lw_low = net_cre(len(p_mid) - 3)         # a low (warm) deck
    assert lw_high > lw_low                                    # high cloud: stronger LW greenhouse (colder top)
    assert sw_low < sw_high                                    # low cloud: stronger SW reflection (cooling)
    assert net_high > net_low                                  # net forcing rises with height (the asymmetry)
    assert net_low < 0.0                                       # the low warm cloud net-COOLS


# ====================================================================================================
# Cloud-window coupling (GitHub issue #84): clouds are ~black across the 8-13 um window, so the cloud LW
# optical depth must enter the WINDOW band too, not just the background band. These guard the fix's two
# physical regimes (open window -> cloud closes it / shut window -> no double-count) + the energy and
# byte-identical invariants. The differential ("with the fix" vs "without") is taken at the
# ``two_stream_lw_windowed`` level -- exactly the line the fix changed -- since column_radiation has no toggle.
# ====================================================================================================
def _window_cloud_olr_drop(rh, T_surf, water_scale=3.0):
    """OLR DROP (>0) and the change in surface downwelling LW from folding an overcast LOW cloud deck's LW
    opacity into the WINDOW band (the issue-#84 fix), at relative humidity ``rh``. Mirrors column_radiation:
    the background band always carries the cloud; only the window band differs (no-cloud vs cloud)."""
    p_edge, p_mid, T, q, Ts, _ = _column(T_surf=T_surf, rh=rh)
    dm = layer_mass(p_edge)
    dtau_bg = lw_optical_depth(p_edge, q)
    dtau_win = lw_window_optical_depth(p_edge, q)
    f = np.zeros_like(p_mid); f[len(p_mid) - 3] = 1.0          # one overcast LOW deck (near surface)
    dtau_cloud, _ = cloud_optical_properties(f, T, p_mid, dm, water_scale=water_scale)
    bg = dtau_bg + dtau_cloud                                 # background band: cloud folded in either way
    without = two_stream_lw_windowed(bg, dtau_win, T, Ts, f_win=rad.F_LW_WINDOW)               # cloud NOT in window
    with_ = two_stream_lw_windowed(bg, dtau_win + dtau_cloud, T, Ts, f_win=rad.F_LW_WINDOW)    # cloud in window (fix)
    return (float(without["OLR"] - with_["OLR"]),             # OLR drop (>0 = greenhouse warming)
            float(with_["D_surf"] - without["D_surf"]),       # surface downwelling LW change (>0)
            float(dtau_win.sum()))


def test_cloud_closes_the_open_window_dry_column():
    """Issue #84: over a DRY column the LW window is OPEN (continuum ~0), so folding an overcast low cloud
    deck's opacity into the window band CLOSES it -- OLR drops (the low-cloud greenhouse the old
    background-band-only treatment threw away) and the surface downwelling LW rises. This is the band where
    low-cloud LW forcing is strongest (the only band the surface otherwise radiates to space through)."""
    dOLR, dD_surf, win_tau = _window_cloud_olr_drop(rh=0.10, T_surf=300.0)
    assert win_tau < 0.1                                       # the window really is open (dry continuum ~0)
    assert dOLR > 2.0                                          # the cloud closes it -> OLR drops (warming)
    assert dD_surf > 5.0                                       # ... and the surface gets more downwelling LW


def test_cloud_window_coupling_vanishes_when_window_already_shut():
    """Issue #84 anti-double-count guard: where humidity has ALREADY closed the window (a saturated hot
    column, dtau_win >> 1), folding the cloud opacity into the window band moves OLR only marginally -- a
    cloud cannot shut an already-shut window twice. The coupling is regime-gated: strong where the window is
    open (dry/cool), negligible where it is closed (hot/moist)."""
    dOLR_dry, _, _ = _window_cloud_olr_drop(rh=0.10, T_surf=300.0)       # open window
    dOLR_wet, _, win_tau = _window_cloud_olr_drop(rh=1.0, T_surf=320.0)  # continuum-shut window
    assert win_tau > 5.0                                      # the window is shut by the continuum alone
    assert abs(dOLR_wet) < 0.5                                # the fix barely moves OLR here (no double-count)
    assert dOLR_dry > 8.0 * abs(dOLR_wet)                     # strongest where the window is open, not shut


def test_cloud_window_energy_telescopes():
    """Issue #84: with the cloud opacity in BOTH bands the windowed two-stream still telescopes exactly
    (sum of layer LW heating == surf_net_lw - OLR) -- the fix adds opacity, it does not break the
    energy-conserving band construction. And via column_radiation the per-band OLRs still sum to the total."""
    p_edge, p_mid, T, q, Ts, S = _column(T_surf=305.0, rh=0.7)
    dm = layer_mass(p_edge)
    f = cloud_fraction(q, T, p_mid, rh_crit=0.5)
    dtau_cloud, _ = cloud_optical_properties(f, T, p_mid, dm)
    bg = lw_optical_depth(p_edge, q) + dtau_cloud
    win = lw_window_optical_depth(p_edge, q) + dtau_cloud     # cloud in the window band too (the fix)
    lw = two_stream_lw_windowed(bg, win, T, Ts, f_win=rad.F_LW_WINDOW)
    assert np.sum(lw["H"]) == pytest.approx(float(lw["surf_net_lw"] - lw["OLR"]), abs=1e-7)
    out = column_radiation(p_edge, T, q, Ts, S, 0.08, clouds=True, cloud_rh_crit=0.5, lw_window=True)
    assert float(out["OLR"]) == pytest.approx(float(out["OLR_bg"] + out["OLR_win"]), rel=1e-12)
    assert np.isfinite(out["toa_net"]) and np.isfinite(out["surf_net"])


def test_cloud_window_term_present_iff_both_on():
    """Issue #84 regression guard: the cloud LW opacity enters the reported window optical depth ONLY when
    clouds AND the window are both on. With the window on but clouds OFF the window carries the bare
    self-continuum (no cloud term) -- so the clouds-off and window-off limbs stay byte-identical, the fix
    touches only the both-on path."""
    p_edge, _, T, q, Ts, S = _column(T_surf=300.0, rh=0.6)
    win_clear = lw_window_optical_depth(p_edge, q)
    off = column_radiation(p_edge, T, q, Ts, S, 0.08, clouds=False, lw_window=True)
    assert np.allclose(off["dtau_win"], win_clear)            # clouds off -> bare continuum in the window
    on = column_radiation(p_edge, T, q, Ts, S, 0.08, clouds=True, cloud_rh_crit=0.5, lw_window=True)
    assert np.allclose(on["dtau_win"], win_clear + on["dtau_lw_cloud"])   # clouds on -> + cloud opacity (the fix)
    assert np.any(on["dtau_win"] > off["dtau_win"] + 1e-12)   # the cloud actually added window opacity


# ====================================================================================================
# Core integration
# ====================================================================================================
def test_core_clouds_on_by_default():
    """Clouds default ON in the core config (the all-on accurate model; env COSMO_CLOUDS=0 to disable), and
    the cloud config fields exist."""
    cfg = StageADConfig()
    assert cfg.clouds is True
    assert {"clouds", "cloud_rh_crit", "cloud_water_scale", "cloud_conv_boost"} <= set(
        f.name for f in __import__("dataclasses").fields(StageADConfig))


def test_core_clouds_off_gives_zero_cover():
    """The clouds-OFF limit (explicit clouds=False) reports zero cloud cover -- the pre-cloud clear-sky
    core byte-for-byte (the OFF path is still reachable now that ON is the default)."""
    m = StageADCore(StageADConfig(nx=8, nz=14, theta_max_deg=55.0, clouds=False))
    m.run_dynamics = False
    for _ in range(50):
        m.step()
    d = m.diagnostics()
    assert np.all(d["cloud_cover"] == 0.0)                     # no cloud when off


def test_env_knobs_configure_clouds(monkeypatch):
    """The COSMO_CLOUD* env knobs set the cloud config (parallel cloud-vs-clear / sweep runs without code
    edits, the raise-suns COSMO_* pattern); unset -> documented defaults."""
    monkeypatch.setenv("COSMO_CLOUDS", "1")
    monkeypatch.setenv("COSMO_CLOUD_WATERSCALE", "1.5")
    monkeypatch.setenv("COSMO_CLOUD_RHCRIT", "0.85")
    cfg = StageADConfig()
    assert cfg.clouds is True
    assert cfg.cloud_water_scale == 1.5 and cfg.cloud_rh_crit == 0.85
    monkeypatch.setenv("COSMO_CLOUDS", "0")
    assert StageADConfig().clouds is False                     # explicit 0 -> off
    monkeypatch.delenv("COSMO_CLOUDS")
    assert StageADConfig().clouds is True                      # unset -> default ON (the accurate model)


def test_core_cloud_radiative_effect_signs():
    """The core's ``cloud_radiative_effect`` returns the standard cloud forcing with the right signs where
    cloud forms: SW cooling (cre_sw <= 0), LW warming (cre_lw >= 0), and a nonzero net somewhere. Measured
    on a no-flow surface-coupled core (the moist ribbon/poleward columns form cloud here; the hot-zone
    cloud needs the resolved moisture convergence -- that is the study, not this unit gate)."""
    m = StageADCore(StageADConfig(nx=16, nz=20, theta_max_deg=58.0, clouds=True, conv_max_sweeps=4))
    m.run_dynamics = False
    for _ in range(400):
        m.step()
    cre = m.cloud_radiative_effect()
    assert np.all(cre["cre_sw"] <= 1e-6)                       # SW cloud effect cools (<=0)
    assert np.all(cre["cre_lw"] >= -1e-6)                      # LW cloud effect warms (>=0)
    assert np.any(cre["cloud_cover"] > 0.1)                    # cloud actually formed somewhere
    assert np.any(np.abs(cre["cre_net"]) > 1.0)               # a nonzero net forcing


def test_core_cloud_radiative_effect_mirrors_radiate_config():
    """``cloud_radiative_effect`` must difference the SAME radiative config the in-loop ``_radiate`` runs
    (PR #101 review): on a conserving-band-split core its clear-sky leg must equal a direct
    ``column_radiation`` call WITH ``lw_conserving_split=True`` (the split rescales the CLEAR background
    band, so the plain-split OLR genuinely differs), and its cloudy leg must reproduce ``_radiate``'s OLR
    byte-for-byte (same clouds + window + bands + diffuse SW)."""
    from flatworld import radiation as rad
    m = StageADCore(StageADConfig(nx=12, nz=16, theta_max_deg=58.0, clouds=True,
                                  lw_conserving_split=True, conv_max_sweeps=4))
    m.run_dynamics = False
    for _ in range(300):
        m.step()
    c = m.cfg
    q = np.clip(m.q, 0.0, 0.5)
    cre = m.cloud_radiative_effect()
    assert np.allclose(cre["OLR_cloud"], m._radiate(q)["OLR"])          # cloudy leg == the in-loop config
    clear = rad.column_radiation(
        m.p_edge[None, :], m.T_atm, q, m.T_surf, m.S, m._albedo(),
        co2_ppm=c.co2_ppm, k_wv=c.k_wv_lw, tau_wellmixed=c.tau_lw_wellmixed, clouds=False,
        lw_window=c.lw_window, f_win=c.f_lw_window, k_cont=c.k_cont_self,
        lw_conserving_split=True, sw_bands=c.sw_bands, sw_diffuse=c.sw_diffuse, S_diffuse=m.S_diffuse)
    plain = rad.column_radiation(
        m.p_edge[None, :], m.T_atm, q, m.T_surf, m.S, m._albedo(),
        co2_ppm=c.co2_ppm, k_wv=c.k_wv_lw, tau_wellmixed=c.tau_lw_wellmixed, clouds=False,
        lw_window=c.lw_window, f_win=c.f_lw_window, k_cont=c.k_cont_self,
        lw_conserving_split=False, sw_bands=c.sw_bands, sw_diffuse=c.sw_diffuse, S_diffuse=m.S_diffuse)
    assert np.allclose(cre["OLR_clear"], clear["OLR"])                  # clear leg carries the split
    assert not np.allclose(clear["OLR"], plain["OLR"])                  # ...and the split genuinely matters


def test_core_water_conservation_theorem_with_clouds():
    """Clouds do NOT break conservation -- they are RADIATIVE, so the exact water-conservation theorem
    (dStorage == cumulative E - P) still holds to machine precision with clouds on (clouds touch neither
    the evaporation source nor the condensation sink). This is the robust 'conservation intact' gate;
    SW-energy conservation with clouds is checked per-call in ``test_cloud_radiation_conserves_shortwave_energy``.

    (Note: the per-column TOA STEADY-STATE corollary toa_net->0 is NOT asserted with clouds -- the cloud
    SW/greenhouse feedback makes the hot wet/dry TRANSITION columns bistable (a cool-cloudy branch vs the
    hot dry-rock branch), so they do not settle to a single TOA-balanced state in a bounded budget. That
    bistability is a reported FINDING of the cloud study, not a conservation failure.)"""
    m = StageADCore(StageADConfig(nx=10, nz=16, theta_max_deg=50.0, clouds=True, conv_max_sweeps=6))
    m.run_dynamics = False
    for _ in range(1500):                                       # develop moisture (and cloud)
        m.step()
    m.reset_water_budget()
    vapor0 = m.water_conservation()["vapor_mass"]
    for _ in range(1500):
        m.step()
    wc = m.water_conservation()
    assert wc["cum_evap"] > 0.0                                # evaporation actually happened
    storage_change = wc["vapor_mass"] - vapor0
    sources = wc["cum_evap"] - wc["cum_precip"]
    assert abs(storage_change - sources) / wc["cum_evap"] < 1e-10   # dStorage == E - P, clouds intact


def test_core_cloud_on_changes_temperature_vs_clear():
    """Clouds CHANGE the equilibrium temperature (they are radiatively active, not inert): a no-flow
    surface-coupled cloud-on core settles on a different surface temperature than the clear-sky core from
    the SAME start, over the columns where cloud forms. (Direction/magnitude of the ribbon shift is the
    coupled study; here we only assert the radiative coupling is live.)"""
    def relaxed(clouds):
        m = StageADCore(StageADConfig(nx=12, nz=18, theta_max_deg=55.0, clouds=clouds, conv_max_sweeps=6,
                                      dt=10800.0))
        m.run_dynamics = False
        for _ in range(3000):
            m.step()
        return m.diagnostics()
    d_clear = relaxed(False)
    d_cloud = relaxed(True)
    cloudy = d_cloud["cloud_cover"] > 0.2                      # columns where cloud actually forms
    assert np.any(cloudy)
    dT = (d_cloud["T_surf_C"] - d_clear["T_surf_C"])[cloudy]
    assert np.max(np.abs(dT)) > 0.5                            # clouds shifted the equilibrium T measurably


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
