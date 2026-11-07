"""
flatworld.radiation -- a gray-band two-stream column radiation scheme (the greenhouse).

This is the physics that lets the model PREDICT absolute surface temperature (from flux + albedo +
greenhouse) instead of anchoring it. It is a standard semi-gray, two-stream, layer-emissivity band
model -- the kind used in simple climate models -- chosen over RRTMG because (a) no Fortran toolchain
is available here and (b) the flat-plane geometry needs a custom column anyway. It is transparent,
energy-conserving by construction, and validated against the analytic gray-atmosphere limit
(``radiative_equilibrium_gray`` + tests).

Longwave: layer emissivities e_k = 1 - exp(-D*dtau_k) (D = 1.66 diffusivity); each layer and the
surface emit sigma T^4; up/down diffuse fluxes are accumulated recursively. The LW optical depth comes
from the water-vapor path (Clausius-Clapeyron humidity -> greenhouse feedback) plus a well-mixed
component (CO2 + continuum). Shortwave: a down-beam with small atmospheric absorption, surface albedo,
and one reflected up-beam.

All routines accept arrays shaped (..., N) so a whole chain of columns is done at once (last axis =
the N atmospheric layers, ordered TOP -> BOTTOM). Fluxes are at the N+1 interfaces (TOP -> SURFACE).
"""
from __future__ import annotations
import os
import numpy as np
from . import constants as C
from . import thermo as th        # qsat for the cloud-fraction RH diagnostic (#19); no import cycle
from .device import get_xp, is_device   # numpy/CuPy array-module dispatch (M3 GPU backend, Phase D)


def _F(x):
    """Coerce to the SHARED thermo working dtype (``thermo._FLOAT``: float64 default = byte-identical;
    float32 under the COSMO_THERMO_FP32 gate), on the array module of ``x`` (numpy OR CuPy).

    GPU PORT (Phase D): ``get_xp(x).asarray`` infers the module from the input array, so a CuPy array in
    -> CuPy out (the line-399 host wall ``np.asarray(dtau, float)`` lived HERE -- killing it here kills it
    at every call site at once). The DTYPE is still driven by the global ``th._FLOAT`` gate, NOT the input's
    own dtype: a numpy fp64 default is a value-preserving no-op (so the CPU path is BYTE-IDENTICAL), and the
    fp32 work is reached via ``set_thermo_dtype(np.float32)`` exactly as in the CPU fp32 gate -- so GPU fp32
    is the SAME gated path, not a silent dtype-follow that would bypass the gate. Reading ``th._FLOAT`` live
    keeps radiation and thermo in lockstep."""
    return get_xp(x).asarray(x, th._FLOAT)


D_DIFF = 1.66                     # LW diffusivity factor (hemispheric mean of 1/cos)

# Default optical-depth tunables (documented INPUTS; swept in config.py). Calibrated so an Earth-like
# moist column gives a ~30-40 K greenhouse (surface warmer than the no-atmosphere effective temp),
# comparable to Earth's. These set greenhouse STRENGTH; surface temperature is the output.
K_WV_LW = 0.10                    # LW mass absorption coeff of water vapor [m^2/kg] (gray-equivalent)
TAU_LW_WELLMIXED = 0.40           # well-mixed (CO2+continuum) LW optical depth, surface, at CO2_ref
CO2_REF_PPM = 400.0               # reference CO2 for the log forcing
CO2_LOG_SENS = 0.30               # d(tau_wellmixed)/d ln(CO2/CO2_ref): logarithmic CO2 forcing
K_WV_SW = 0.008                   # SW (near-IR) mass absorption coeff of water vapor [m^2/kg]
ALBEDO_ATM = C._env_float("COSMO_ALBEDO_ATM", 0.10)
                                  # clear-sky atmospheric/Rayleigh SW reflection (planetary, ex-surface).
                                  # SPECTRUM-DEPENDENT (Rayleigh ~ lambda^-4 is almost purely visible/UV),
                                  # and the suns' spectrum IS specified (SPECTRUM_PLAN.md rev 5): solar-like
                                  # with 200-320 nm notched out. Re-deriving 0.10 under that notch gives
                                  # 0.0941 -- the ozone-aware Rayleigh reweighting, bracket 0.077-0.102 --
                                  # so the ADOPTED value is still 0.10 and this default IS the design, not a
                                  # placeholder for one. (It moves the NET surface shortwave +0.107 % in
                                  # concert with F_VIS_SW below, which is the gate; derive_spectrum_albedos.py
                                  # section [7].) The rejected full retune's <=0.05 is NOT the world -- see
                                  # spectrum_env.sh's "WHAT USED TO BE HERE". Env-overridable, read at import
                                  # (set in the SHELL first); default = the canonical value, so unset env is
                                  # byte-identical.
D_SW_DIFF = 1.66                  # diffuse-SW path enhancement (hemispheric-mean sec; CLEARSKY_SW_PLAN Stage 5):
                                  # the haze forward-scattered DIFFUSE down-stream travels a longer slant path
                                  # than the direct beam, so its column near-IR absorption uses D_SW_DIFF*dtau_sw.

# ---- Two-band shortwave (CLEARSKY_SW_PLAN Stage 1): default OFF -> the gray path is BYTE-IDENTICAL ----
# The gray ``exp(-K_WV_SW * WVP)`` has NO spectral window, so clear-sky surface SW -> 0 with no floor as the
# (advected) water-vapour path grows -- fine at Earth PW (~71%), wrong in the extrapolation to high WVP
# (CLEARSKY_SW_FINDINGS root-cause #2). The real solar spectrum has a window: ~half the energy is VISIBLE,
# where water vapour is ~transparent, so real clear-sky surface SW FLOORS at ~40-50% even at huge WVP. The
# two-band scheme (``sw_bands=True``) splits the beam into a transparent visible band + an absorbing near-IR
# band, recovering that floor (and the surface PAR / photosynthesis number directly from the visible band).
#
# FREE-QUANTITY DISCIPLINE (plan review #72.2 -- say explicitly which quantity moves):
#   * F_VIS_SW is the SPECTROSCOPIC GIVEN -- the visible/near-IR energy split of the solar beam is fixed
#     external physics, NOT tuned. (~half the TOA solar irradiance is at lambda < ~0.7 um.)
#   * K_WV_SW_NIR -- the near-IR band's EFFECTIVE water-vapour absorption coefficient -- is the SINGLE FREE
#     quantity, anchored to the EXTERNAL Earth-PW benchmark (clear-sky transmission ~0.71 at WVP~30 kg/m^2,
#     i.e. matching the existing gray scheme's Earth-PW value, itself ~Earth's observed 0.75-0.80). It is the
#     ONE coefficient that moves; F_VIS_SW does not. At Earth PW the resulting transmission is then a
#     PREDICTION TO VALIDATE (tests), not a number any OTHER quantity is fudged to hit.
#
# TWO CONSCIOUS SIMPLIFICATIONS (flagged for the record; both deferred, neither changes the gray default):
#   * RAYLEIGH IS APPLIED GRAY across both bands -- ``alpha_atm`` (the 10% Rayleigh reflection) is charged to
#     the visible and near-IR beams equally, whereas real Rayleigh ~ lambda^-4 is almost entirely a VISIBLE
#     effect. The TOTAL planetary reflection is preserved, but it under-reflects the visible / over-reflects
#     the near-IR, so the reported visible PAR (S*(1-alpha_atm)*f_vis) is a SOFT CEILING, not a hard one
#     (real Rayleigh scatters more blue out of the direct beam, partly forward-scattered back down). The
#     spectral Rayleigh split + diffuse-down stream is the Stage-5 (haze/diffuse) item.
#   * F_VIS_SW = 0.5 ASSUMES SUN-LIKE SUNS -- the visible/near-IR split is the ~solar (~5800 K) value, but
#     the setting's cosmology pins the suns only by FLUX + GEOMETRY (I0*cos^3 theta, day-mean balance) and
#     never states a photospheric temperature / spectral class. So f_vis (like the Rayleigh 0.10, the Earth-PW
#     transmission anchor, and the PAR) is an inherited SOLAR-ANALOG assumption, not fixed by the world; if the
#     suns were ever specified notably cooler/hotter, both f_vis and the visible PAR move with it.
#     UPDATE (SPECTRUM_PLAN.md rev 5): the suns' spectrum IS now specified, and it is SOLAR-LIKE -- an
#     engineered, non-thermal beam that keeps the Sun's own band shape except for a clean notch from 200
#     to 320 nm (the biologically damaging ultraviolet), with the total delivered flux held. Because the
#     notch is renormalized away rather than subtracted, the water-transparent (lambda < 0.7 um) share
#     re-derives to 0.4897 against this committed 0.50 and the PAR fraction (400-700 nm) to 0.4078 against
#     the solar 0.40 -- both inside the slop the knob is quoted at, so 0.50 IS the adopted design, not an
#     un-updated placeholder. (p != F_VIS_SW: they differ by the ultraviolet, and the 4.57 umol/J PAR
#     conversion applies to p, NEVER to F_VIS_SW.) spectrum_env.sh is consequently a no-op that re-exports
#     0.50; the REJECTED full retune's 0.22 is NOT the world. Env-overridable below; the default is the
#     canonical value, so unset env is byte-identical.
#     K_WV_SW_NIR does NOT move with it: the near-IR keeps a solar shape and only carries more energy, so
#     absorption per unit vapour is unchanged (re-anchoring would double-count).
F_VIS_SW = C._env_float("COSMO_F_VIS_SW", 0.50)
                                  # visible fraction of the solar beam (water-transparent; see UPDATE above)
K_WV_SW_NIR = 0.0183              # near-IR effective WV absorption coeff [m^2/kg]: the SINGLE free quantity,
                                  # anchored so f_vis + (1-f_vis)*exp(-K*30) reproduces the ~0.71 Earth-PW
                                  # transmission (0.90*[0.5 + 0.5*exp(-0.0183*30)] = 0.710).

# ---- Longwave window band + its humidity continuum (CLEARSKY_SW_PLAN Stage 2): default OFF -----------
# The gray LW above totally CLOAKS the surface (no spectral window), and -- because the model's qsat clamps
# the vapor path at the supercritical limit -- the hot-end OLR has NO Komabayashi-Ingersoll plateau (it
# keeps climbing, so the equator T is uninterpretable; CLEARSKY_SW_FINDINGS.md). Stage 2 adds a two-band
# LW: a background band (fraction 1-f_win, the gray ``dtau_lw``) + an 8-13 um WINDOW band (fraction f_win)
# whose opacity is the water-vapor SELF-CONTINUUM. The window is OPEN in the dry limit (removes the surface
# cloak -- "2a") and CLOSES as humidity rises (the continuum is ~quadratic in vapor -- "2b"), which pins the
# window's emission level and CAPS the hot-end OLR. Default OFF (``lw_window=False``) -> the gray path is
# BYTE-IDENTICAL (mirrors how #19 clouds shipped). FREE-QUANTITY DISCIPLINE (plan review #72): F_LW_WINDOW
# is a SPECTROSCOPIC GIVEN (the Planck fraction in the window) and K_CONT_SELF is anchored to an EXTERNAL
# benchmark (the observed tropical window opacity), so the resulting OLR cap is a PREDICTED outcome to
# validate, NOT a target the continuum strength is tuned to hit.
F_LW_WINDOW = 0.30                # spectral fraction of thermal LW in the 8-13 um "atmospheric window".
                                  # SPECTROSCOPIC GIVEN: the Planck fraction between 8 and 13 um at terrestrial
                                  # emission temperatures (~255-288 K) is ~0.27-0.31 (Planck integral); 0.30 is
                                  # that fraction, NOT fitted to any climate output.
K_CONT_SELF = 1.5e-2              # water-vapor SELF-continuum mass-absorption coeff [m^2/kg] at E_REF_CONT.
                                  # The window's only absorber (so the window is transparent when dry). ANCHORED
                                  # to the observed "dirty window": a reference tropical column (PW ~ 50 kg/m^2,
                                  # surface e ~ 30 hPa) has window optical depth ~ 1 (the window is semi-
                                  # transparent on Earth) -> K_CONT_SELF ~ 1.5e-2. An EXTERNAL Earth benchmark
                                  # (cf. Stage 1's Earth-PW anchor), NOT the model's desired OLR cap.
E_REF_CONT = 1.0e3               # reference H2O partial pressure [Pa] for the self-broadening (e/E_REF_CONT).
                                  # Folds with K_CONT_SELF into one anchored coefficient; ~lower-troposphere e.

# ---- Radiatively-active clouds (GitHub issue #19): the cloud optical depth is COMPUTED FROM PHYSICS ----
# The cloud OPTICAL DEPTH is NOT a free tunable: it is set by the cloud's condensed-water path (which the
# model's OWN thermodynamics fix -- a rising saturated parcel condenses water as its qsat falls with height)
# and the droplet/crystal size. So the radiative strength is DERIVED (``cloud_water_path`` -> the optical
# depths), not swept. What remains are genuine MICROPHYSICAL constants -- the droplet effective radius, the
# entrainment sub-adiabaticity, the autoconversion (rainout) cloud-water ceiling, the LW mass-absorption
# coefficient -- set to observational BEST ESTIMATES (NOT fitted to any climate target). They carry the
# real, IRREDUCIBLE cloud uncertainty (cloud microphysics is THE dominant uncertainty in real climate too);
# ``cloud_water_scale`` (default 1.0 = the best estimate) perturbs them together only to REPORT that residual
# sensitivity. Clouds default OFF everywhere -> every clear-sky call is BYTE-IDENTICAL to the pre-cloud scheme.
CLOUD_RH_CRIT = 0.80           # critical RH for cloud onset: the sub-grid humidity variance a single-value
                               # column lacks (standard 0.75-0.85); the one genuinely sub-grid closure left.
CLOUD_SUBADIABATIC = 0.6       # in-cloud water as a fraction of the adiabatic value (entrainment dilutes it);
                               # observed ~0.5-0.8.
CLOUD_LWC_MAX = 0.30e-3        # ceiling on in-cloud LIQUID water content [kg/m^3] -- the rainout/autoconversion
                               # limit (condensate above this precipitates, so it is not optically active);
                               # observed cloud-mean ~0.2-0.5 g/m^3.
CLOUD_IWC_MAX = 0.05e-3        # ceiling on in-cloud ICE water content [kg/m^3]; observed ~0.01-0.1 g/m^3.
CLOUD_REFF_LIQ = 12.0e-6       # liquid droplet effective radius [m] (marine ~10-15 um).
CLOUD_REFF_ICE = 30.0e-6       # ice crystal effective radius [m] (~20-60 um; larger -> optically thinner/g).
CLOUD_KABS_LW = 0.10e3         # cloud LW FLUX mass-absorption coeff [m^2/kg] (~0.1 m^2/g; Stephens 1978) --
                               # the in-two-stream dtau uses this / the diffusivity D, so the cloud layer
                               # emissivity is the physical 1 - exp(-kabs * LWP).
CLOUD_G_ASYM = 0.85            # SW scattering asymmetry (forward-peaked water/ice); sets the in-cloud
                               # two-stream reflectance R = tau_sw / (tau_sw + 2/(1-g)).
# SW single-scattering albedo (SSA) -- the fraction of intercepted SW that is SCATTERED (vs ABSORBED).
# Default 1.0 == REFLECT-ONLY: clouds scatter all they intercept, absorb none (the historic scheme; the
# surface light it gives is then an UPPER BOUND, since real clouds absorb some near-IR -- the crux flagged
# in extract_nsub30_doc.py). Set < 1 (env COSMO_CLOUD_SW_SSA) to turn on cloud SW ABSORPTION: the
# extinction splits into a scattering part (ssa*tau, drives the reflectance) and an absorbing part
# ((1-ssa)*tau, added to the column SW optical depth so the cloudy layers absorb+heat). Broadband cloud
# SSA is high (visible ~1, near-IR ~0.95-0.98) -> a few % column absorption; best-estimate, NOT fitted.
CLOUD_SSA_LIQ = 1.0            # liquid-cloud broadband SW single-scattering albedo (~0.99 obs; 1.0 = OFF).
CLOUD_SSA_ICE = 1.0            # ice-cloud   broadband SW single-scattering albedo (~0.95 obs; 1.0 = OFF).
CLOUD_CONV_BOOST = 0.0         # OPTIONAL extra cloud FRACTION in resolved-ascent layers (sub-grid convective
                               # cloud); default 0 -> cloud fraction is pure RH (still circulation-following).
# ---- Chunk-3 (CLOUD_PROGNOSTIC_ROADMAP) decoupled radiative cap: a PHYSICAL-PLAUSIBILITY in-cloud water
#      content bound, NOT the rainout/autoconversion ceiling. Once the sub-grid PDF cover makes the in-cloud
#      ``rho*qc/f`` conversion physical, the prognostic LWP is let TRACK the retained reservoir ``qc`` up to a
#      bound set by what REAL clouds hold (a few g/m^3), rather than re-clipping at the rainout cap
#      (CLOUD_LWC_MAX/IWC_MAX) -- rainout is handled explicitly by the Chunk-2 autoconversion sink, so the
#      rainout clip would DOUBLE-COUNT it. These keep the optics bounded (stability) without pinning the
#      cloud radiative strength to the cap; they are ~10-20x the rainout caps so LWP tracks qc over the whole
#      relevant range. Observational best estimates (deep-cloud peak LWC), NOT fitted -- the residual cloud
#      microphysics uncertainty (cf. CLOUD_LWC_MAX) is unchanged.
CLOUD_LWC_PLAUSIBLE = 3.0e-3   # physical in-cloud LIQUID water content bound [kg/m^3] (~3 g/m^3; deep cumulus
                               # peak ~1-3 g/m^3) -- the decoupled radiative ceiling (NOT the 0.3 g/m^3 rainout cap).
CLOUD_IWC_PLAUSIBLE = 1.0e-3   # physical in-cloud ICE water content bound [kg/m^3] (~1 g/m^3; ice clouds are
                               # water-poorer than liquid) -- the decoupled radiative ICE ceiling.
CLOUD_PDF_WIDTH_FLOOR = 1.0e-9 # floor [kg/kg] on the sub-grid total-water PDF half-width (= (1-rh_crit)*qsat),
                               # so Q1 = (q_t - qsat)/width stays finite where qsat -> 0 (cold/high layers).


# ======================================================================================
# Cloud fraction + cloud optical properties (issue #19). The REUSABLE, core-independent radiative-effect
# machinery: cloud fraction -> added LW optical depth (greenhouse, warming) + a column SW albedo (cooling).
# Pure functions of the column state, shared by any stage that calls ``column_radiation``.
# ======================================================================================
def cloud_fraction(q, T_layers, p_mid, rh_crit=CLOUD_RH_CRIT, conv=None, conv_boost=CLOUD_CONV_BOOST):
    """Diagnosed fractional cloud cover per layer in [0,1], from the column's OWN moisture state, shape
    matching ``q`` (..., N).

    A relative-humidity diagnostic (Sundqvist 1989): cloud rises linearly from 0 at ``rh_crit`` to 1 at
    saturation,  f = clip((RH - rh_crit)/(1 - rh_crit), 0, 1),  RH = q / qsat(T, p).  RH is a determinate
    physical continuum (every boundary/scaling case has a graded answer -- the project's principled-quantity
    test), so the cloud field is well-posed. Because the resolved circulation SETS the moisture field, this
    places deep cloud in the moist ascending branch and clears the dry subsiding branch with NO prescribed
    latitude profile -- the physically faithful source the unified core enables.

    ``conv`` (optional, same shape) is a 0..1 resolved-ascent / convective-activity signal; where it is
    positive the cloud fraction is nudged toward overcast by ``conv_boost`` -- the sub-grid convective cloud
    that forms in active updrafts even where the grid-mean RH is below saturation. Off by default
    (conv_boost=0): cloud fraction is then pure RH (still circulation-following, via the moisture field)."""
    xp = get_xp(q)
    RH = _F(q) / xp.maximum(th.qsat(T_layers, p_mid), 1e-12)
    f = xp.clip((RH - rh_crit) / (1.0 - rh_crit), 0.0, 1.0)
    if conv is not None and conv_boost > 0.0:
        f = f + (1.0 - f) * conv_boost * xp.clip(_F(conv), 0.0, 1.0)
    return xp.clip(f, 0.0, 1.0)


def cloud_pdf_partition(q_t, T_layers, p_mid, rh_crit=CLOUD_RH_CRIT, width_floor=CLOUD_PDF_WIDTH_FLOOR):
    """Sub-grid TOTAL-WATER PDF cloud partition (CLOUD_PROGNOSTIC_ROADMAP Chunk 3) -- ONE distribution that
    yields cloud fraction AND the condensation source on a single footing, retiring the Chunk-1/2 split of
    an RH-diagnostic cover vs a grid-mean condensation source (the haze artifact). Shape matching ``q_t``
    (..., N). Returns ``(f, qc_eq)``: the cloud fraction ``f`` in [0,1] and the equilibrium GRID-MEAN
    condensate ``qc_eq`` [kg/kg].

    Closure: a UNIFORM ("top-hat") sub-grid distribution of total water ``q_t = q + q_c`` about its grid-mean,
    half-width ``Delta = (1 - rh_crit) * qsat`` -- the ``cloud_rh_crit`` role GENERALISED from a cover-onset
    threshold to the PDF variance (so cover onset and condensate onset COINCIDE, the haze fix). With the
    normalised saturation departure ``Q1 = (q_t - qsat) / Delta``:

      * cloud fraction      f = P(q_t > qsat) = clip((Q1 + 1)/2, 0, 1)
      * equilibrium condensate (the saturated-tail integral of the supersaturation, E[(q_t - qsat)^+]):
            qc_eq = 0                         (Q1 <= -1, entire cell sub-saturated)
                  = (Delta/4) * (1 + Q1)^2    (-1 < Q1 < 1, partial cloud)
                  = q_t - qsat                (Q1 >= 1, entire cell saturated)

    Both ``f`` and ``qc_eq`` come from the SAME ``Q1``, so ``f > 0`` <=> ``qc_eq > 0`` exactly (no cover
    without water, no grid-mean-only source): the condensation source the microphysics reads is the
    PDF-integrated supersaturation ``qc_eq``, not the grid-mean excess ``max(q_t - qsat, 0)``, so it fires
    from the saturated tail even at grid-mean RH < 1. In the overcast limit (Q1 >= 1) ``qc_eq -> q_t - qsat``
    and ``f -> 1``, i.e. it REDUCES to the grid-mean saturation adjustment -- the consistent continuation."""
    xp = get_xp(q_t)
    qs = th.qsat(T_layers, p_mid)
    delta = xp.maximum((1.0 - rh_crit) * qs, width_floor)
    Q1 = (_F(q_t) - qs) / delta
    f = xp.clip(0.5 * (Q1 + 1.0), 0.0, 1.0)
    qc_eq = xp.where(Q1 >= 1.0, _F(q_t) - qs,
                     xp.where(Q1 <= -1.0, 0.0, 0.25 * delta * (1.0 + Q1) ** 2))
    return f, xp.maximum(qc_eq, 0.0)


def cloud_water_path(f, T_layers, p_mid, dm, subadiabatic=CLOUD_SUBADIABATIC, lwc_max=CLOUD_LWC_MAX,
                     iwc_max=CLOUD_IWC_MAX, water_scale=1.0):
    """Per-layer in-cloud condensed-water PATH [kg/m^2] from the model's OWN thermodynamics -- the physical
    basis of the cloud optical depth (so the optical depth is computed, not a free knob). Shape (..., N).

    A rising saturated parcel condenses water as its saturation humidity falls with height; the in-cloud
    condensate at a level is the ADIABATIC water condensed from cloud base to there, diluted by entrainment
    (``subadiabatic``) and CAPPED at the rainout/autoconversion ceiling (``lwc_max``/``iwc_max`` -- condensate
    above that precipitates, so it is not optically active). Within a contiguous cloudy band the condensed
    mixing ratio at layer k is ``dq = qsat(band foot) - qsat(k)`` (the band foot = the qsat just BELOW the
    cloud's lowest layer, i.e. the lifting-condensation level); the in-cloud water content is
    ``LWC = subadiabatic * rho_air * dq`` capped at the phase ceiling, and the layer water path is
    ``LWP = LWC * dz = LWC * dm / rho_air``. Liquid vs ice splits at the freezing point. ``water_scale``
    (default 1.0 = the physical best estimate) multiplies the result, only to REPORT the residual
    microphysical sensitivity. Returns (LWP (..., N) [kg/m^2], ice mask (..., N))."""
    xp = get_xp(T_layers)
    T = _F(T_layers)
    qs = th.qsat(T, p_mid)                                        # (..., N) TOP->SURFACE
    cloudy = _F(f) > 1e-3
    nz = qs.shape[-1]
    qbase = qs.copy()                                            # surface layer: no LCL below -> base = own qsat
    for k in range(nz - 2, -1, -1):                              # walk SURFACE-1 -> TOP; the layer below k is k+1
        inherit = cloudy[..., k] & cloudy[..., k + 1]            # same contiguous band as the layer below
        qbase[..., k] = xp.where(inherit, qbase[..., k + 1], qs[..., k + 1])   # else band foot = qsat just BELOW (LCL)
    dq = xp.maximum(qbase - qs, 0.0) * cloudy                    # condensed mixing ratio from band base [kg/kg]
    rho_air = _F(p_mid) / (C.R_DRY * T)           # [kg/m^3]
    ice = T < C.T_FREEZE
    lwc = subadiabatic * rho_air * dq                            # in-cloud water content [kg/m^3] (adiabatic, diluted)
    lwc = xp.minimum(lwc, xp.where(ice, iwc_max, lwc_max))       # rainout/autoconversion ceiling
    lwp = water_scale * lwc * _F(dm) / xp.maximum(rho_air, 1e-9)   # LWC * dz [kg/m^2]
    return lwp, ice


def prognostic_cloud_water_path(qc, f, T_layers, p_mid, dm, lwc_max=CLOUD_LWC_MAX, iwc_max=CLOUD_IWC_MAX,
                                water_scale=1.0, subadiabatic=None):
    """In-cloud condensed-water PATH [kg/m^2] from the PROGNOSTIC grid-mean condensate ``qc`` [kg/kg], the
    Chunk-1 (CLOUD_PROGNOSTIC_ROADMAP) replacement for the adiabatic ``cloud_water_path`` reconstruction.
    Shape (..., N). Returns (LWP_in (..., N) [kg/m^2], ice mask (..., N)).

    ``qc`` is a GRID-MEAN mixing ratio, so the in-cloud content divides by the cloud fraction ``f``:
    ``LWC_in = rho_air * qc / f`` (the ``/f`` the haze fix rests on -- with ``f`` already FLOORED by the
    caller's cover<->condensate rule so ``f -> 0`` with ``qc > 0`` cannot blow up).
    ``LWP_in = LWC_in * dz = LWC_in * dm / rho_air``.

    CAP -- ``lwc_max``/``iwc_max`` bound the in-cloud LWC for stability. Two footings (the CALLER chooses, by
    passing the cap values):
      * CHUNK-1 default (``CLOUD_LWC_MAX``/``IWC_MAX``, the adiabatic-rainout ceiling): the same ceiling the
        #19 diagnostic ``cloud_water_path`` uses, so the prognostic and diagnostic RADIATIVE optics sit on one
        footing -- but once grid-mean ``qc`` exceeds the cap onset (~1.5e-5 kg/kg at the f_min floor) the optics
        SATURATE and stop tracking ``qc`` (radiative strength is cap+cover-set, sign-firm, magnitude-flat).
      * CHUNK-3 DECOUPLED (``CLOUD_LWC_PLAUSIBLE``/``IWC_PLAUSIBLE``, a PHYSICAL-plausibility bound ~a few
        g/m^3, NOT the rainout cap; PR #142 review #1): the radiative LWP TRACKS ``qc`` up to what real clouds
        hold, so the cloud radiative strength is a genuine function of the retained reservoir. Rainout is
        handled explicitly by the Chunk-2 autoconversion sink, so re-clipping at the rainout ceiling would
        DOUBLE-COUNT it. This needs the Chunk-3 PDF cover to make the ``/f`` conversion physical (the f_min
        floor would otherwise inflate in-cloud LWC) -- see ``cloud_pdf_partition`` and the roadmap.

    ``subadiabatic`` is accepted only for signature-compatibility with ``cloud_water_path`` (so a caller
    forwarding it through ``cloud_optical_properties(**wp_kw)`` does not hit a TypeError, PR #142 review #4)
    but is INAPPLICABLE: the reservoir IS the actual condensate, not an entrainment-diluted adiabatic
    reconstruction, so there is no sub-adiabaticity to apply."""
    del subadiabatic                                                # inapplicable on the prognostic path (see above)
    xp = get_xp(T_layers)
    T = _F(T_layers)
    rho_air = _F(p_mid) / (C.R_DRY * T)               # [kg/m^3]
    ice = T < C.T_FREEZE
    lwc_in = rho_air * _F(qc) / xp.maximum(_F(f), 1e-30)   # in-cloud LWC [kg/m^3]
    lwc_in = xp.minimum(lwc_in, xp.where(ice, iwc_max, lwc_max))     # rainout/autoconversion ceiling
    lwp = water_scale * lwc_in * _F(dm) / xp.maximum(rho_air, 1e-9)        # LWC * dz [kg/m^2]
    return lwp, ice


def cloud_optical_properties(f, T_layers, p_mid, dm, g_asym=CLOUD_G_ASYM, water_scale=1.0, qc=None,
                             return_tau_sw=False, return_lwp=False,
                             ssa_liq=CLOUD_SSA_LIQ, ssa_ice=CLOUD_SSA_ICE,
                             n_d=None, n_d_ref=None, **wp_kw):
    """Per-layer cloud optics from the cloud-water path -- the optical depth is COMPUTED, not a free knob.
    Shape (..., N). Returns (dtau_lw_cloud, R_in), or (dtau_lw_cloud, R_in, tau_scat, dtau_sw_abs) when
    ``return_tau_sw`` (the per-layer SW SCATTERING optical depth for the overlap albedo, and the grid-mean
    cloud SW ABSORPTION optical depth to add to the column SW beam). ``return_lwp`` (additive, default
    OFF -> existing returns byte-identical) APPENDS the per-layer in-cloud water path ``lwp`` [kg/m^2] as
    the LAST tuple element -- a read-only diagnostic of the water path the optics were built from (Phase-C
    cloud water path / optical-depth wishlist), with no effect on the optics themselves.

    The water path is the ADIABATIC reconstruction (``cloud_water_path``) by default; when ``qc`` (the
    prognostic grid-mean condensate [kg/kg]) is given it is instead the PROGNOSTIC in-cloud path
    (``prognostic_cloud_water_path``, CLOUD_PROGNOSTIC_ROADMAP Chunk 1) -- the only change is swapping the
    reconstruction for the retained reservoir; the SW/LW machinery below is unchanged.

    SW (albedo, COOLING): the EXTINCTION optical depth from geometric optics,
        tau_ext = 1.5 * LWP / (rho_w * r_eff),
    with ``r_eff`` the PHASE effective radius (liquid ~12 um / ice ~30 um). The single-scattering albedo
    ``ssa`` (1.0 = reflect-only; < 1 turns on near-IR absorption) splits it into a SCATTERING part
    ``tau_scat = ssa*tau_ext`` (the reflectance ``R_in = tau_scat/(tau_scat + 2/(1-g))``, a conservative-
    scattering two-stream on the scattered light) and an ABSORBING part ``tau_abs = (1-ssa)*tau_ext`` returned
    grid-mean (``dtau_sw_abs = f*tau_abs``) for ``column_radiation`` to add to the beam's SW optical depth, so
    the cloudy layers absorb + heat. ssa=1 -> tau_abs=0, tau_scat=tau_ext: BYTE-IDENTICAL to the reflect-only
    scheme. (Splitting extinction by ssa and reusing the conservative two-stream on the scattered part is the
    standard first approximation -- not the full non-conservative two-stream -- adequate at this resolution.)

    LW (greenhouse, WARMING): a GRID-MEAN cloud LW optical depth ``dtau_lw_cloud = f * (kabs/D) * LWP`` added
    to the clear-sky dtau and run through the SAME two-stream (the /D makes the layer emissivity the physical
    ``1 - exp(-kabs * LWP)``); the cloud-top TEMPERATURE then sets the warming automatically.

    The HIGH-vs-LOW asymmetry now EMERGES from the physics, no height switch: a warm low cloud is water-rich
    (large qsat drop) and small-droplet -> optically THICK/reflective (net-COOLING); a cold high cloud is
    water-poor and large-crystal (ice) -> SW-THIN (net-WARMING via its cold LW emission)."""
    xp = get_xp(f)
    f = _F(f)
    if qc is not None:
        lwp, ice = prognostic_cloud_water_path(qc, f, T_layers, p_mid, dm, water_scale=water_scale, **wp_kw)
    else:
        lwp, ice = cloud_water_path(f, T_layers, p_mid, dm, water_scale=water_scale, **wp_kw)
    # Salt-aerosol CCN Twomey limb (#237; default n_d=None -> reff_liq is the constant CLOUD_REFF_LIQ, so
    # r_eff is byte-identical to the pre-feature path). More CCN -> more, smaller LIQUID droplets at fixed
    # water -> smaller r_eff -> larger tau -> brighter cloud. RATIO form anchored at n_d_ref (n_d == n_d_ref
    # -> factor 1.0 -> no-op), so only N_d/N_ref enters. LIQUID only -- salt is a warm-rain CCN; the ice
    # r_eff is untouched (the band is warm-rain). The model's cloud LW limb (dtau_lw_cloud) is mass-based and
    # carries no r_eff, so this moves the SW albedo, not the LW greenhouse.
    reff_liq = CLOUD_REFF_LIQ
    if n_d is not None:
        reff_liq = CLOUD_REFF_LIQ * (n_d_ref / _F(n_d)) ** (1.0 / 3.0)   # n_d_ref a python float / device array
    r_eff = xp.where(ice, CLOUD_REFF_ICE, reff_liq)
    tau_ext = 1.5 * lwp / (C.RHO_LIQ * r_eff)
    ssa = xp.where(ice, ssa_ice, ssa_liq)                       # 1.0 -> reflect-only (byte-identical)
    tau_scat = ssa * tau_ext                                    # scattered fraction drives the reflectance
    R_in = tau_scat / (tau_scat + 2.0 / (1.0 - g_asym))
    dtau_lw_cloud = f * (CLOUD_KABS_LW / D_DIFF) * lwp           # grid-mean; /D -> two-stream emissivity 1-exp(-kabs LWP)
    extra = (lwp,) if return_lwp else ()                        # additive read-only diagnostic (Phase-C)
    if return_tau_sw:
        dtau_sw_abs = f * (1.0 - ssa) * tau_ext                 # grid-mean cloud SW absorption optical depth
        return (dtau_lw_cloud, R_in, tau_scat, dtau_sw_abs) + extra
    return (dtau_lw_cloud, R_in) + extra


def cloud_sw_albedo(f, R_in, overlap="random", tau_sw=None, g_asym=CLOUD_G_ASYM):
    """Column shortwave cloud albedo (..., ) from the per-layer cloud fraction ``f`` and in-cloud
    reflectance ``R_in``. Bounded [0,1]; reduces to 0 with no cloud. Two layer-overlap modes:

    * ``overlap="random"`` (DEFAULT, byte-identical to the historic scheme): the probability the beam is
      reflected by at least one cloud deck,  A = 1 - prod_k (1 - f_k R_in_k).  Reduces to R_in for a single
      overcast deck. (Maximum overlap -- A = max_k f_k R_in_k -- is the opposite extreme.)
    * ``overlap="column"``: a SINGLE two-stream on the cloud-fraction-weighted COLUMN-TOTAL SW optical depth,
      tau_tot = sum_k f_k tau_sw,k,  A = tau_tot/(tau_tot + 2/(1-g)).  This is the physically faithful
      reflectance of a CONTIGUOUS saturated column and puts the SW limb on the SAME linear-grid-mean footing
      the cloud LW already uses. ``tau_sw`` (per-layer SW extinction, from ``cloud_optical_properties(...,
      return_tau_sw=True)``) is required; if omitted it is backed out of R_in (tau = R_in*k/(1-R_in),
      k=2/(1-g)).

    Why the mode matters (cloud-albedo overlap audit, CLOUDS_RESULTS.md §4b): "random" multiplies independent
    per-layer reflection probabilities, so stacking ~20 saturated layers inflates a per-deck ~0.85-0.9 up
    toward ~0.99 -- an OVERLAP artifact, not real opacity. "column" removes that inconsistency (a contiguous
    saturated ribbon column drops from ~0.99 to ~0.95, ~4x more surface light). The column albedo folds into
    the planetary SW reflection (``column_radiation``), conserving energy: the reflected SW never enters.

    DELIBERATE SIMPLIFICATION (review #69, flagged so a future reader knows it is a choice, not an oversight):
    the SW reflection is applied at TOA, ABOVE the column's (small near-IR water-vapour) SW absorption -- so
    for a LOW deck sitting under most of the vapour this slightly overstates the reflection efficiency; it is
    energy-conserving and adequate at this model's altitude resolution. (The historic SW-vs-LW overlap
    inconsistency is what ``overlap="column"`` resolves.)"""
    xp = get_xp(f)
    f = _F(f)
    R_in = _F(R_in)
    if overlap == "column":
        k = 2.0 / (1.0 - g_asym)
        if tau_sw is None:                                  # back out per-layer tau from R_in = tau/(tau+k)
            tau_sw = R_in * k / xp.maximum(1.0 - R_in, 1e-12)
        tau_tot = xp.sum(f * _F(tau_sw), axis=-1)           # cloud-fraction-weighted column-total SW tau
        return tau_tot / (tau_tot + k)
    refl = xp.clip(f * R_in, 0.0, 1.0)
    return 1.0 - xp.prod(1.0 - refl, axis=-1)


def two_stream_nc(tau_scat, tau_abs, g_asym=CLOUD_G_ASYM):
    """NON-CONSERVATIVE two-stream (R, A_abs, T) of a cloud column from its SW scattering optical depth
    ``tau_scat`` and ABSORBING optical depth ``tau_abs`` (both column-total scalars, shape (...,)), with
    asymmetry ``g_asym``. Returns column reflectance ``R``, absorptance ``A_abs`` and transmittance ``T``
    with ``R + A_abs + T = 1`` by construction (energy closes automatically), so a cloud that reflects R
    also ABSORBS A_abs (-> deposited as in-cloud heating; the burn-off feedback) and transmits T -- unlike
    ``cloud_sw_albedo``, which is reflect-only (A_abs implicitly 0).

    Method: the delta-Eddington-scaled two-stream with the SAME hemispheric backscatter b = (1-g)/2 the
    reflect-only ``cloud_sw_albedo`` uses. With b=(1-g)/2, delta-scaling preserves the conservative (tau_abs
    -> 0) reflectance EXACTLY: ``R -> tau_scat/(tau_scat + 2/(1-g))`` = the ``overlap="column"`` cloud
    albedo. So the ``tau_abs = 0`` limit is byte-identical to the reflect-only scheme (taken via the exact
    branch below); absorption appears only for ``tau_abs > 0`` (single-scattering albedo omega < 1).

    Closure choice (NOT Meador-Weaver / Eddington): gamma1 = 1 - omega(1-b), gamma2 = omega*b with
    b=(1-g)/2 is effectively a mean-diffusivity mu_bar = 1 two-stream -- DICTATED by consistency with the
    existing reflect-only limb (it is what makes the conservative limit match ``cloud_sw_albedo`` exactly).
    Relative to the standard Eddington/hemispheric-mean gammas the diffuse path is shorter, so at fixed omega
    this UNDER-absorbs somewhat -- a bias that folds into the (already soft, swept) ssa knob, not a separate
    uncertainty.

    Units: tau_scat, tau_abs, R, A_abs, T are dimensionless [-]. This solver is per-band: ``column_radiation``
    calls it for a NON-CONSERVATIVE NEAR-IR band (tau_abs>0) and a CONSERVATIVE VISIBLE band (tau_abs=0), so the
    absorption is NEAR-IR-CONFINED, not grey (real cloud SW absorption is near-IR; PR #332). The magnitude is
    strongly ``omega``-sensitive (a soft cloud-microphysics knob -> SWEEP omega, do not pin); the DIRECTION
    (absorption heats the deck) is firm."""
    xp = get_xp(tau_scat)
    tau_scat = _F(tau_scat)
    tau_abs = _F(tau_abs)
    tau = tau_scat + tau_abs                                     # column-total extinction optical depth
    omega = tau_scat / xp.maximum(tau, 1e-30)                    # single-scattering albedo (0 where no cloud)
    # delta-Eddington scaling: truncate the g^2 forward-scatter spike (standard; keeps the thick-deck
    # reflectance physical for the forward-peaked water/ice phase function).
    fsc = g_asym * g_asym                                        # forward-peak fraction
    one_m = xp.maximum(1.0 - omega * fsc, 1e-30)
    tau_p = one_m * tau                                          # scaled optical depth
    omega_p = (1.0 - fsc) * omega / one_m                        # scaled single-scattering albedo
    g_p = g_asym / (1.0 + g_asym)                                # scaled asymmetry
    b = 0.5 * (1.0 - g_p)                                        # hemispheric backscatter fraction [-]
    gamma1 = 1.0 - omega_p * (1.0 - b)                           # two-stream loss coefficient [-]
    gamma2 = omega_p * b                                         # two-stream backscatter coefficient [-]
    k = xp.sqrt(xp.maximum(gamma1 * gamma1 - gamma2 * gamma2, 1e-30))
    r_inf = gamma2 / (gamma1 + k)                                # semi-infinite-layer reflectance [-]
    e = xp.exp(-k * tau_p)
    e2 = e * e
    denom = xp.maximum(1.0 - r_inf * r_inf * e2, 1e-30)
    R = r_inf * (1.0 - e2) / denom                              # layer reflectance (diffuse) [-]
    T = (1.0 - r_inf * r_inf) * e / denom                       # layer diffuse transmittance [-]
    # EXACT conservative branch (tau_abs == 0 <-> omega == 1): reproduce cloud_sw_albedo's column formula
    # bit-for-bit (the general k->0 limit is a 0/0 numerically), so the reflect-only limit is byte-identical.
    R_cons = tau_scat / (tau_scat + 2.0 / (1.0 - g_asym))
    absorbing = tau_abs > 0.0
    R = xp.where(absorbing, R, R_cons)
    T = xp.where(absorbing, T, 1.0 - R_cons)
    A_abs = xp.clip(1.0 - R - T, 0.0, 1.0)                      # absorptance = the energy-closing residual
    return R, A_abs, T


# ======================================================================================
# Optical depths from the column state
# ======================================================================================
def layer_mass(p_edge):
    """Dry-ish air mass per area in each layer [kg/m^2], from interface pressures p_edge [Pa],
    ordered TOP->BOTTOM (increasing p). Shape (..., N+1) -> (..., N)."""
    xp = get_xp(p_edge)
    p_edge = _F(p_edge)
    return xp.diff(p_edge, axis=-1) / C.G_ACCEL


def lw_optical_depth(p_edge, q, co2_ppm=CO2_REF_PPM, k_wv=K_WV_LW,
                     tau_wellmixed=TAU_LW_WELLMIXED):
    """LW optical depth of each layer [-], shape (..., N).

    dtau = k_wv * (vapor path) + tau_wellmixed * (layer dry mass / column dry mass) * co2_factor.
    The water-vapor term gives the greenhouse FEEDBACK (more q -> more dtau). The well-mixed term
    (CO2 + continuum) is distributed by mass and scaled logarithmically with CO2.
    """
    xp = get_xp(q)
    dm = layer_mass(p_edge)                                  # kg/m^2 per layer
    vapor_path = _F(q) * dm                   # kg/m^2 of water vapor per layer
    co2_factor = 1.0 + CO2_LOG_SENS * np.log(max(float(co2_ppm), 1.0) / CO2_REF_PPM)  # host scalar
    col_mass = xp.sum(dm, axis=-1, keepdims=True)
    dtau = k_wv * vapor_path + tau_wellmixed * co2_factor * dm / xp.maximum(col_mass, 1e-9)
    return xp.maximum(dtau, 0.0)


def sw_optical_depth(p_edge, q, k_wv_sw=K_WV_SW):
    """SW (solar-band) absorption optical depth of each layer [-], shape (..., N). Mostly water-vapor
    near-IR; small."""
    xp = get_xp(q)
    dm = layer_mass(p_edge)
    return xp.maximum(k_wv_sw * _F(q) * dm, 0.0)


def lw_window_optical_depth(p_edge, q, k_cont=K_CONT_SELF, e_ref=E_REF_CONT):
    """Window-band LW optical depth per layer [-], shape (..., N) -- the water-vapor SELF-continuum (Stage 2).

    ``dtau_win = k_cont * (vapor path) * (e / e_ref)``, with ``e`` the vapor PARTIAL pressure (Pa). The
    self-broadening factor ``e/e_ref`` makes the window optical depth ~QUADRATIC in humidity, so the window
    is essentially transparent in the dry limit (``dtau_win -> 0`` -- the surface radiates ~directly to space)
    and goes OPTICALLY THICK in the moist limit (``dtau_win >> 1`` -- the window CLOSES), which pins the
    window's emission level and caps the hot-end OLR. This is the load-bearing physics of Stage 2b: an
    always-open window (``dtau_win = 0``) removes the surface cloak but does NOT cap OLR; the continuum's
    CLOSING is what produces the Komabayashi-Ingersoll / Simpson-Nakajima plateau."""
    xp = get_xp(p_edge)
    p_edge = _F(p_edge)
    dm = layer_mass(p_edge)
    p_mid = 0.5 * (p_edge[..., :-1] + p_edge[..., 1:])
    q = _F(q)
    e = th.vapor_pressure_from_q(q, p_mid)                   # vapor partial pressure [Pa]
    return xp.maximum(k_cont * (q * dm) * (e / e_ref), 0.0)


# ======================================================================================
# Two-stream longwave (layer-emissivity band model). Energy-conserving by construction.
# ======================================================================================
def two_stream_lw(dtau, T_layers, T_surf, eps_surf=1.0, D=D_DIFF, band_frac=1.0):
    """Diffuse-gray LW radiative transfer through a column.

    Parameters
    ----------
    dtau     : (..., N) LW optical depth per layer (TOP->BOTTOM)
    T_layers : (..., N) layer temperatures [K]
    T_surf   : (...) surface temperature [K]
    eps_surf : surface LW emissivity (1 = blackbody)
    band_frac: spectral fraction of the thermal LW carried by THIS band (Stage 2). The Planck emission of
               every layer AND the surface is scaled by it, so the band's fluxes/heating scale by band_frac
               and the multi-band totals are band sums. Default 1.0 -> the full gray band (multiply by 1.0 is
               exact, so the single-band call is BYTE-IDENTICAL to the pre-Stage-2 scheme).

    Returns dict with:
      OLR        (...)        outgoing LW at TOA [W/m^2]
      D_surf     (...)        downwelling LW at the surface [W/m^2]
      Fnet       (..., N+1)   net UPWARD LW flux at each interface (TOP->SURFACE) [W/m^2]
      H          (..., N)     LW heating of each layer [W/m^2] (positive = heating)
      surf_net_lw(...)        net LW the surface LOSES = sigma Ts^4 - D_surf [W/m^2]
    """
    xp = get_xp(dtau)
    dtau = _F(dtau)
    T_layers = _F(T_layers)
    *batch, N = dtau.shape
    t = xp.exp(-D * dtau)                                    # layer transmissivity
    e = 1.0 - t                                              # layer emissivity
    B = band_frac * C.SIGMA_SB * T_layers**4                 # layer Planck emission (each face), band-weighted
    Bs = band_frac * C.SIGMA_SB * _F(T_surf)**4

    eB = e * B
    # Explicit interface recurrence -- numerically STABLE for any optical depth (a cumulative-product
    # form would divide by exp(-D*tau), which underflows to 0 for the optically thick moist/hot columns
    # this world has). The loop is over layers only (vectorized over the column batch), ~nz iterations.
    Dn = xp.zeros((*batch, N + 1), eB.dtype)                # downwelling, Dn[...,0]=0 at TOA
    for k in range(N):
        Dn[..., k + 1] = Dn[..., k] * t[..., k] + eB[..., k]
    D_surf = Dn[..., N]
    U_surf = eps_surf * Bs + (1.0 - eps_surf) * D_surf
    Up = xp.zeros((*batch, N + 1), eB.dtype)                # upwelling, from the surface up
    Up[..., N] = U_surf
    for k in range(N - 1, -1, -1):
        Up[..., k] = Up[..., k + 1] * t[..., k] + eB[..., k]

    Fnet = Up - Dn                                          # net upward at each interface
    H = Fnet[..., 1:] - Fnet[..., :-1]                      # layer heating = F_bottom - F_top (upward+)
    return dict(OLR=Up[..., 0], D_surf=D_surf, Fnet=Fnet, H=H,
                surf_net_lw=Bs - D_surf)


def two_stream_lw_windowed(dtau_bg, dtau_win, T_layers, T_surf, f_win=F_LW_WINDOW, eps_surf=1.0, D=D_DIFF):
    """Two-band gray LW (Stage 2): a BACKGROUND band (spectral fraction ``1-f_win``, optical depth
    ``dtau_bg`` = the gray ``dtau_lw``) PLUS a WINDOW band (fraction ``f_win``, optical depth ``dtau_win`` =
    the self-continuum from ``lw_window_optical_depth``). Each band is an independent gray two-stream weighted
    by its spectral fraction (``two_stream_lw(..., band_frac=...)``); the column totals are the band sums.

    Energy-conserving by construction: each band telescopes (sum of layer H == surf_net_lw - OLR), so the
    band SUMS do too. Reduces to the single gray band at ``f_win -> 0`` (the window band carries zero Planck
    weight). The returned dict matches ``two_stream_lw`` (OLR, D_surf, Fnet, H, surf_net_lw) and additionally
    reports the per-band OLR (``OLR_bg``, ``OLR_win``) so the window's contribution can be read off.

    The physics this buys (CLEARSKY_SW_PLAN Stage 2): with ``dtau_win`` GROWING with humidity (the continuum)
    the window CLOSES in the moist limit, pinning its emission level and CAPPING the hot-end OLR (the
    Komabayashi-Ingersoll plateau, 2b). An ALWAYS-OPEN window (``dtau_win = 0``) only removes the surface
    cloak -- OLR still climbs as ``f_win*sigma*Ts^4`` -- so 2a alone is necessary but NOT sufficient."""
    bg = two_stream_lw(dtau_bg, T_layers, T_surf, eps_surf, D, band_frac=1.0 - f_win)
    win = two_stream_lw(dtau_win, T_layers, T_surf, eps_surf, D, band_frac=f_win)
    return dict(OLR=bg["OLR"] + win["OLR"], D_surf=bg["D_surf"] + win["D_surf"],
                Fnet=bg["Fnet"] + win["Fnet"], H=bg["H"] + win["H"],
                surf_net_lw=bg["surf_net_lw"] + win["surf_net_lw"],
                OLR_bg=bg["OLR"], OLR_win=win["OLR"])


# ======================================================================================
# COLUMN-PARALLEL (numba prange) LW two-stream recurrence -- the CPU-fuse radiation speedup.
#
#   The diffuse-gray LW transfer (``two_stream_lw``) is a per-column sequential nz-sweep (the
#   interface recurrence: a top->surface down pass and a surface->top up pass). Columns are
#   INDEPENDENT, so the sweep parallelizes over the nx columns exactly like the convective-adjustment
#   kernel in stage_ad_core.py. The numpy reference pays ~2*N small per-layer array dispatches in its
#   two Python ``for k`` loops; the numba kernel fuses the whole per-column transfer (transmissivity
#   ``t = exp(-D*dtau)``, Planck ``B = band_frac*sigma*T^4``, both recurrences, Fnet/H) into one cache-
#   local sweep with no temporaries, then prange over columns.
#
#   This is a COMPARED-VARIANT speedup gated behind a default-OFF flag (COSMO_RADIATION_PARALLEL): the
#   numpy ``two_stream_lw`` / ``two_stream_lw_windowed`` stay UNTOUCHED (OFF byte-identity is structural).
#   The ON path is wired at the core's radiation refresh call (``_thermo_substep`` -> ``_radiate`` ->
#   ``column_radiation(..., lw_parallel=True)``), and FALLS BACK to numpy whenever the kernel cannot
#   reproduce the numpy path bit-for-round-off: non-2-D inputs (the kernel needs (nx, N) columns), or
#   non-fp64 (an fp32 specialization would diverge past round-off). It covers the PRODUCTION radiation
#   config for free -- clouds / LW window / SW bands / SW diffuse only change the ``dtau`` and
#   ``band_frac`` FED INTO the two-stream, so the recurrence kernel reproduces zero cloud/band logic.
#
#   numba is imported LAZILY (only when the flag is ON); a plain/default run never imports it. The kernel
#   is built once on first use and cached (numba @njit cache=True), so the JIT cost is paid once per
#   process. Matches the numpy reference to round-off (~1e-13), NOT bit-for-bit -- the leaf exp/T^4
#   reassociate. See archive/_cpufuse_radiation_RESULTS.md and tests/test_stage_ad_core.py.
# ======================================================================================
_LW_NUMBA_KERNEL = None   # built lazily on first ON call


def _build_lw_numba_kernel():
    """Compile (once) the prange-over-columns single-band LW two-stream kernel. Constants are pulled to
    module floats so the njit kernel closes over plain numbers (no attribute access in the hot loop)."""
    from numba import njit, prange   # lazy: only imported when COSMO_RADIATION_PARALLEL is ON

    SIGMA_SB = float(C.SIGMA_SB)

    @njit(parallel=True, cache=True)
    def _two_stream_lw_numba(dtau, T_layers, T_surf, eps_surf, D, band_frac,
                             OLR, D_surf_out, surf_net_lw, Fnet, H):
        nx, N = dtau.shape
        for i in prange(nx):                                   # COLUMNS ARE INDEPENDENT -> parallel
            Bs = band_frac * SIGMA_SB * T_surf[i] ** 4
            # down pass: Dn[k+1] = Dn[k]*t[k] + eB[k], Dn[0]=0 at TOA (top->surface, sequential in k)
            Dn_k = 0.0                                          # downwelling at the current TOP interface
            for k in range(N):
                tk = np.exp(-D * dtau[i, k])                    # layer transmissivity
                ek = 1.0 - tk                                   # layer emissivity
                Bk = band_frac * SIGMA_SB * T_layers[i, k] ** 4 # band-weighted Planck emission
                eBk = ek * Bk
                Dn_kp1 = Dn_k * tk + eBk                        # downwelling at the BOTTOM interface of layer k
                Fnet[i, k] = -Dn_k                              # net upward at interface k (Up filled below)
                Dn_k = Dn_kp1
            D_surf_i = Dn_k                                     # = Dn[..., N]
            Fnet[i, N] = -D_surf_i                              # net upward at the surface interface (Up below)
            D_surf_out[i] = D_surf_i
            surf_net_lw[i] = Bs - D_surf_i
            # up pass: Up[N]=U_surf, Up[k] = Up[k+1]*t[k] + eB[k] (surface->top, sequential in k)
            Up_kp1 = eps_surf * Bs + (1.0 - eps_surf) * D_surf_i   # U_surf
            Fnet[i, N] += Up_kp1                                # net upward at surface = Up_surf - Dn_surf
            for k in range(N - 1, -1, -1):
                tk = np.exp(-D * dtau[i, k])
                ek = 1.0 - tk
                Bk = band_frac * SIGMA_SB * T_layers[i, k] ** 4
                eBk = ek * Bk
                Up_k = Up_kp1 * tk + eBk                        # upwelling at the TOP interface of layer k
                Fnet[i, k] += Up_k                              # net upward at interface k = Up_k - Dn_k
                Up_kp1 = Up_k
            OLR[i] = Up_kp1                                     # upwelling at TOA = Up[..., 0]
            # H[k] = Fnet[k+1] - Fnet[k] (layer heating, upward+)
            for k in range(N):
                H[i, k] = Fnet[i, k + 1] - Fnet[i, k]

    return _two_stream_lw_numba


def two_stream_lw_parallel(dtau, T_layers, T_surf, eps_surf=1.0, D=D_DIFF, band_frac=1.0):
    """Column-parallel (numba prange) drop-in for ``two_stream_lw``'s recurrence. REQUIRES 2-D (nx, N)
    fp64 inputs (the caller guards this and falls back to numpy otherwise). Reproduces the numpy
    reference to round-off (~1e-13), not bit-for-bit (the leaf exp/T^4 reassociate). Returns the same
    dict (OLR, D_surf, Fnet, H, surf_net_lw)."""
    global _LW_NUMBA_KERNEL
    if _LW_NUMBA_KERNEL is None:
        _LW_NUMBA_KERNEL = _build_lw_numba_kernel()
    dtau = np.ascontiguousarray(dtau, dtype=np.float64)
    T_layers = np.ascontiguousarray(T_layers, dtype=np.float64)
    T_surf = np.ascontiguousarray(np.broadcast_to(np.asarray(T_surf, float), dtau.shape[:1]),
                                  dtype=np.float64)
    nx, N = dtau.shape
    OLR = np.empty(nx, dtype=np.float64)
    D_surf = np.empty(nx, dtype=np.float64)
    surf_net_lw = np.empty(nx, dtype=np.float64)
    Fnet = np.empty((nx, N + 1), dtype=np.float64)
    H = np.empty((nx, N), dtype=np.float64)
    _LW_NUMBA_KERNEL(dtau, T_layers, T_surf, float(eps_surf), float(D), float(band_frac),
                     OLR, D_surf, surf_net_lw, Fnet, H)
    return dict(OLR=OLR, D_surf=D_surf, Fnet=Fnet, H=H, surf_net_lw=surf_net_lw)


def two_stream_lw_windowed_parallel(dtau_bg, dtau_win, T_layers, T_surf, f_win=F_LW_WINDOW,
                                    eps_surf=1.0, D=D_DIFF):
    """Column-parallel drop-in for ``two_stream_lw_windowed``: two band calls to the parallel kernel,
    same dict (incl. per-band OLR_bg/OLR_win). REQUIRES 2-D fp64 inputs (caller-guarded)."""
    bg = two_stream_lw_parallel(dtau_bg, T_layers, T_surf, eps_surf, D, band_frac=1.0 - f_win)
    win = two_stream_lw_parallel(dtau_win, T_layers, T_surf, eps_surf, D, band_frac=f_win)
    return dict(OLR=bg["OLR"] + win["OLR"], D_surf=bg["D_surf"] + win["D_surf"],
                Fnet=bg["Fnet"] + win["Fnet"], H=bg["H"] + win["H"],
                surf_net_lw=bg["surf_net_lw"] + win["surf_net_lw"],
                OLR_bg=bg["OLR"], OLR_win=win["OLR"])


def _sw_one_beam(S_in, t, alpha_surf, one):
    """Internal: a single down+reflected-up SW beam of TOA-incident flux ``S_in`` (already past the TOA
    Rayleigh reflection) through a column with per-layer transmissivity ``t`` (..., N). Returns
    (surf_down, surf_abs, escaped_up, H) -- the surface down/absorbed flux, the up-beam escaping to space,
    and the per-layer absorption heating. Shared by the gray and two-band paths so they conserve energy
    identically (the two-band visible beam just passes t==1)."""
    xp = get_xp(t)
    Pdown = xp.concatenate([one, xp.cumprod(t, axis=-1)], axis=-1)        # (...,N+1)
    down = S_in[..., None] * Pdown
    H = down[..., :-1] - down[..., 1:]                       # absorbed from the down-beam, per layer
    surf_down = down[..., -1]
    surf_abs = (1.0 - alpha_surf) * surf_down
    up0 = alpha_surf * surf_down                             # reflected at the surface
    Qd = xp.concatenate([xp.cumprod(t[..., ::-1], axis=-1)[..., ::-1], one], axis=-1)  # (...,N+1)
    up = up0[..., None] * Qd                                # up[k] = up0 * prod_{i>=k} t[i]
    H = H + (up[..., 1:] - up[..., :-1])                    # absorbed from the up-beam, per layer
    return surf_down, surf_abs, up[..., 0], H


def _sw_down_stream(S_in, f_vis, t_nir, alpha_surf, one):
    """One down+reflected SW stream of (post-Rayleigh) incident flux ``S_in`` through a column with near-IR
    transmissivity ``t_nir`` (..., N). GRAY (``f_vis=None``): one absorbing beam at ``t_nir``. TWO-BAND
    (``f_vis`` set, Stage 1): a transparent VISIBLE fraction (t==1) + an absorbing NEAR-IR fraction at
    ``t_nir``. Returns (surf_down, surf_abs, escaped_up, H). Shared by the direct beam and the Stage-5
    diffuse-down stream (which differ only in ``S_in`` and the path-enhanced ``t_nir``), so both conserve
    energy identically: S_in == surf_abs + sum(H) + escaped_up."""
    if f_vis is None:                                      # gray: one absorbing beam (byte-identical path)
        return _sw_one_beam(S_in, t_nir, alpha_surf, one)
    f_vis = float(f_vis)
    t_vis = get_xp(t_nir).ones_like(t_nir)                 # visible: water-transparent, no layer absorption
    sd_v, sa_v, esc_v, H_v = _sw_one_beam(S_in * f_vis, t_vis, alpha_surf, one)
    sd_n, sa_n, esc_n, H_n = _sw_one_beam(S_in * (1.0 - f_vis), t_nir, alpha_surf, one)
    return sd_v + sd_n, sa_v + sa_n, esc_v + esc_n, H_v + H_n   # H_v == 0 (visible is transparent)


def shortwave(S_toa, dtau_sw, alpha_surf, alpha_atm=ALBEDO_ATM, *, f_vis=None, S_diffuse=None,
              tau_haze=None, omega0_haze=None, g_haze=None, cloud_abs=None, cloud_w=None,
              cloud_R_vis=None, cloud_R_nir=None):
    """Down/up SW beam with small atmospheric absorption and surface reflection.

    Returns dict:
      surf_abs   (...)      SW absorbed at the surface [W/m^2]
      H          (..., N)   SW heating of each layer [W/m^2]
      planetary_reflected   (...) SW reflected back to space [W/m^2]
      surf_down  (...)      SW reaching the surface (down) [W/m^2]

    GRAY (``f_vis=None``, default): one absorbing beam with per-layer optical depth ``dtau_sw`` applied to
    the WHOLE beam (the pre-Stage-1 scheme; BYTE-IDENTICAL).

    TWO-BAND (``f_vis`` set, CLEARSKY_SW_PLAN Stage 1): the solar beam splits into a VISIBLE fraction
    ``f_vis`` (water-transparent -> Rayleigh/``alpha_atm`` only, t==1, reaches the surface regardless of
    WVP -> the surface PAR / transmission FLOOR ~ f_vis*(1-alpha_atm)) and a NEAR-IR fraction ``1-f_vis``
    that carries the water-vapour absorption -- here ``dtau_sw`` is the NEAR-IR optical depth (built with
    ``K_WV_SW_NIR``, see ``column_radiation``). Only the near-IR beam heats the layers, so the SW heating
    is deposited where the near-IR is absorbed (ALOFT, in the moist upper troposphere) -- the deposition
    the gray scheme spread over the whole beam. Energy is conserved per band, hence per call.

    DIFFUSE-DOWN (``S_diffuse`` set, CLEARSKY_SW_PLAN Stage 5; default ``None`` -> BYTE-IDENTICAL): an extra
    downward DIFFUSE stream of column-top flux ``S_diffuse`` -- the haze forward-scatter the geometry recovers
    (``geometry.day_mean_diffuse_flux``) from the slant haze that the direct day-mean flux ``S_toa`` drops as
    pure loss. It is processed exactly like the direct beam (same ``f_vis`` split, same surface albedo) but on
    a path-enhanced near-IR transmissivity ``exp(-D_SW_DIFF*dtau_sw)`` (diffuse light travels a longer slant
    path). It passes the column Rayleigh (``1-alpha_atm``) on entry just like the direct beam, then absorbs /
    reflects through the column to the surface -- so it RAISES surf_down/surf_abs (and the visible PAR). Energy
    conserved with the diffuse input counted in: ``surf_abs + sum(H) + planetary_reflected == S_toa + S_diffuse``.

    CLOUD SW ABSORPTION -- two regimes, both default -> BYTE-IDENTICAL:
    * TWO-BAND (``cloud_R_vis``/``cloud_R_nir`` set; the production path, PR #332). The cloud REFLECTION is
      per band -- VISIBLE reflects ``cloud_R_vis`` (conservative, no absorption -> the water-transparent floor
      survives), NEAR-IR reflects ``cloud_R_nir`` and ABSORBS ``cloud_abs`` of the near-IR it admits. Here
      ``alpha_atm`` is the CLEAR-sky Rayleigh (cloud reflection handled per band, not folded in). This confines
      cloud SW absorption to the near-IR, fixing the grey limb that ate the visible band (surf light ~10^2-10^3x
      too dark). Requires ``f_vis``.
    * GREY (``cloud_R_vis=None``; the legacy/gray-SW path). Cloud REFLECTION enters via ``alpha_atm`` (caller
      passes alpha_atm_eff); ``cloud_abs`` is the fraction of the WHOLE beam ENTERING the column (post-Rayleigh,
      post-cloud-reflection) that the cloud absorbs, applied grey ahead of the ``f_vis`` split.
    In both, ``cloud_w`` (..., N) is the per-layer deposition weights (sum_k = 1 over cloud layers, but ALL-ZERO
    where there is no absorbing cloud -- ssa=1 -> cloud_abs=0 -> deposited flux exactly 0, so do not assert
    sum==1); the absorbed flux is carved from BOTH the direct and diffuse down-beams and deposited as layer
    heating ``H`` (R + A_abs + T close). Energy closes per band: ``surf_abs + sum(H) + planetary_reflected ==
    S_toa (+ S_diffuse)``."""
    xp = get_xp(dtau_sw)
    dtau_sw = _F(dtau_sw)
    *batch, N = dtau_sw.shape
    S_toa = xp.broadcast_to(_F(S_toa), tuple(batch)).astype(th._FLOAT)
    alpha_surf = xp.broadcast_to(_F(alpha_surf), tuple(batch)).astype(th._FLOAT)
    one = xp.ones((*batch, 1), th._FLOAT)
    # --- SALT-AEROSOL HAZE dimming (#237, decoupled; default tau_haze=None -> byte-identical) ---------------
    # A salt-aerosol haze of (per-column) slant optical depth ``tau_haze`` intercepts the direct beam ABOVE the
    # column. Reusing the haze single-scattering split (constants.OMEGA0_HAZE / G_HAZE -- the SAME physics the
    # geometry's forward-scatter uses): the intercepted flux is partly SCATTERED (omega0) and partly ABSORBED
    # (1-omega0). Of the scattered light the forward fraction b_fwd=(1+g)/2 continues downward and is folded
    # into the diffuse-down stream (a partial recovery -> NET dimming), the backward fraction goes to space,
    # and the absorbed fraction heats the haze aloft (added to the top layer). Energy conserves against the
    # ORIGINAL S_toa (+ S_diffuse): the forward-scatter is carved FROM the direct beam, not added input.
    haze_bsca = haze_ab = None
    if tau_haze is not None:
        omega0 = C.OMEGA0_HAZE if omega0_haze is None else float(omega0_haze)
        g_h = C.G_HAZE if g_haze is None else float(g_haze)
        b_fwd = 0.5 * (1.0 + g_h)
        tau_h = xp.broadcast_to(_F(tau_haze), tuple(batch)).astype(th._FLOAT)
        intercept = S_toa * (1.0 - xp.exp(-tau_h))          # direct-beam flux intercepted by the salt haze
        scattered = omega0 * intercept
        fwd_diffuse = b_fwd * scattered                     # forward-scattered -> diffuse-down (partial recovery)
        haze_bsca = (1.0 - b_fwd) * scattered               # back-scattered -> space
        haze_ab = (1.0 - omega0) * intercept                # absorbed in the haze (deposited aloft)
        S_toa = S_toa - intercept                           # the direct beam is dimmed by the salt haze
        S_diffuse = fwd_diffuse if S_diffuse is None else (_F(S_diffuse) + fwd_diffuse)
    t_nir = xp.exp(-dtau_sw)                                # near-IR (gray: whole-beam) transmissivity
    cloud_H = None

    if cloud_R_vis is not None:
        # === TWO-BAND CLOUD OPTICS (near-IR-confined cloud SW absorption) ================================
        # The VISIBLE band is CONSERVATIVE -- the cloud reflects ``cloud_R_vis`` and absorbs NOTHING, and the
        # visible is water-transparent (t==1), so the surface visible floor SURVIVES a strongly-absorbing deck.
        # The NEAR-IR band is NON-CONSERVATIVE -- the cloud reflects ``cloud_R_nir`` and ABSORBS ``cloud_abs``
        # of the near-IR it admits (-> in-cloud heating). Clear-sky Rayleigh ``alpha_atm`` is grey (both bands).
        # This REPLACES the grey pre-split carve, which ate the water-transparent visible band and drove surface
        # light ~10^2-10^3x too dark. Requires the two-band SW (``f_vis`` set). Energy closes per band:
        # surf_abs + sum(H) + planetary_reflected == S_toa (+ S_diffuse).
        if f_vis is None:
            raise ValueError("two-band cloud optics (cloud_R_vis) requires the two-band SW (f_vis) to be set")
        fvis = float(f_vis)
        R_vis = _F(cloud_R_vis)
        R_nir = _F(cloud_R_nir)
        caf = _F(cloud_abs)[..., None] if cloud_abs is not None else None   # near-IR absorbed frac of entering beam
        cw = _F(cloud_w) if cloud_w is not None else None                   # per-layer deposition weights, sum_k=1
        t_vis = xp.ones_like(t_nir)                                         # visible: water-transparent

        def _two_band(S_src, t_nir_band):
            """One SW stream (direct or diffuse) through the two-band cloud. Returns
            (surf_down, surf_abs, escaped_up, H_col, reflected_to_space, cloud_absorption_H)."""
            Sray = S_src * (1.0 - alpha_atm)                # past the clear-sky (grey) Rayleigh reflection
            Svis0 = Sray * fvis                             # visible share of the beam
            Snir0 = Sray * (1.0 - fvis)                     # near-IR share
            refl = Svis0 * R_vis + Snir0 * R_nir            # cloud reflection to space (per band)
            Svis = Svis0 - Svis0 * R_vis                    # visible entering the column (no cloud absorption)
            Snir = Snir0 - Snir0 * R_nir                    # near-IR entering the column
            cH = None
            if caf is not None:
                abs_nir = caf * Snir[..., None]             # near-IR cloud absorption (visible band: none)
                cH = abs_nir * cw
                Snir = Snir - abs_nir[..., 0]               # only the transmitted remnant continues down
            sd_v, sa_v, esc_v, H_v = _sw_one_beam(Svis, t_vis, alpha_surf, one)
            sd_n, sa_n, esc_n, H_n = _sw_one_beam(Snir, t_nir_band, alpha_surf, one)
            return sd_v + sd_n, sa_v + sa_n, esc_v + esc_n, H_v + H_n, refl, cH

        surf_down, surf_abs, escaped, H, refl_dir, cloud_H = _two_band(S_toa, t_nir)
        planetary_reflected = S_toa * alpha_atm + refl_dir + escaped
        if S_diffuse is not None:                          # Stage-5 diffuse-down, same per-band cloud treatment
            S_diff = xp.broadcast_to(_F(S_diffuse), tuple(batch)).astype(th._FLOAT)
            t_nir_d = xp.exp(-D_SW_DIFF * dtau_sw)          # longer diffuse path -> stronger near-IR absorption
            sd_d, sa_d, esc_d, H_d, refl_dif, cH_d = _two_band(S_diff, t_nir_d)
            surf_down = surf_down + sd_d
            surf_abs = surf_abs + sa_d
            H = H + H_d
            planetary_reflected = planetary_reflected + S_diff * alpha_atm + refl_dif + esc_d
            if cH_d is not None:
                cloud_H = cH_d if cloud_H is None else cloud_H + cH_d
    else:
        # === GREY SW (existing scheme; cloud_R_vis=None -> BYTE-IDENTICAL) ================================
        S_in_total = S_toa * (1.0 - alpha_atm)              # past the TOA Rayleigh reflection
        # --- CLOUD in-beam SW ABSORPTION (grey two-stream limb; cloud_abs=None -> byte-identical) ---
        # Carve the cloud-absorbed fraction ``caf`` from the (grey) down-beam BEFORE the f_vis split and hold it
        # as cloud-layer heating; the beam that continues is the transmitted remnant. Grey (whole beam), matching
        # the grey cloud reflection folded into alpha_atm. (The two-band branch above confines this to near-IR.)
        caf = cw = None
        if cloud_abs is not None:
            caf = _F(cloud_abs)[..., None]                 # (...,1) absorbed fraction of the ENTERING beam
            cw = _F(cloud_w)                               # (...,N) per-layer deposition weights, sum_k = 1
            abs_dir = caf * S_in_total[..., None]          # W/m^2 absorbed from the direct beam
            cloud_H = abs_dir * cw                         # deposited in the cloud layers
            S_in_total = S_in_total - abs_dir[..., 0]      # only the transmitted remnant continues down

        surf_down, surf_abs, escaped, H = _sw_down_stream(S_in_total, f_vis, t_nir, alpha_surf, one)
        planetary_reflected = S_toa * alpha_atm + escaped

        if S_diffuse is not None:                          # --- Stage 5: haze forward-scattered diffuse-down ---
            S_diff = xp.broadcast_to(_F(S_diffuse), tuple(batch)).astype(th._FLOAT)
            S_diff_in = S_diff * (1.0 - alpha_atm)         # the diffuse stream also passes the column Rayleigh
            if cloud_abs is not None:                      # cloud absorbs the diffuse down-beam too
                abs_dif = caf * S_diff_in[..., None]
                cloud_H = cloud_H + abs_dif * cw
                S_diff_in = S_diff_in - abs_dif[..., 0]
            t_nir_d = xp.exp(-D_SW_DIFF * dtau_sw)         # longer diffuse path -> stronger near-IR absorption
            sd_d, sa_d, esc_d, H_d = _sw_down_stream(S_diff_in, f_vis, t_nir_d, alpha_surf, one)
            surf_down = surf_down + sd_d
            surf_abs = surf_abs + sa_d
            H = H + H_d
            planetary_reflected = planetary_reflected + S_diff * alpha_atm + esc_d

    if cloud_H is not None:                               # cloud SW absorption -> in-cloud heating (closes R+A+T=1)
        H = H + cloud_H

    if haze_bsca is not None:                              # salt-haze backscatter -> space; absorption -> aloft
        planetary_reflected = planetary_reflected + haze_bsca
        # deposit the haze absorption in the top layer (index 0 = TOP) so the column energy budget closes
        H = xp.concatenate([(H[..., :1] + haze_ab[..., None]), H[..., 1:]], axis=-1)

    return dict(surf_abs=surf_abs, H=H, planetary_reflected=planetary_reflected, surf_down=surf_down)


def _lw_device_ok(lw_device_kernel, dtau, T_layers, T_surf):
    """Gate: the fused DEVICE LW kernel runs only for the 2-D (nx, N) device-array column-chain shape (the
    production GPU radiation call). ``lw_device_kernel`` is a GETTER ``dtype -> RawKernel`` (or None) so the
    caller builds a kernel matched to ``dtau``'s dtype (fp32 captured path / fp64) -- avoiding a byte-reinterpret
    if a fp32 kernel met an fp64 dtau. A None getter, a host array, a non-2-D shape, or a mismatched layer axis
    falls back to the numpy-generic sequential xp path (itself device-correct, just launch-bound)."""
    if lw_device_kernel is None:
        return False
    if not (is_device(dtau) and is_device(T_layers)):
        return False
    if dtau.ndim != 2 or T_layers.ndim != 2 or dtau.shape != T_layers.shape:
        return False
    if dtau.dtype.kind != "f" or T_layers.dtype.kind != "f" or dtau.dtype != T_layers.dtype:
        return False
    return True


def _lw_parallel_ok(lw_parallel, dtau, T_layers, T_surf):
    """Gate: the column-parallel LW kernel can reproduce the numpy two-stream only for the 2-D (nx, N)
    fp64 column-chain shape (the production core's radiation call). Any other shape (1-D single column,
    higher-rank batch) or non-float dtype falls back to numpy. dtau and T_layers must be 2-D with a
    matching layer axis; T_surf must broadcast to (nx,)."""
    if not lw_parallel:
        return False
    if is_device(dtau) or is_device(T_layers) or is_device(T_surf):
        return False                                  # numba kernel is host-fp64 only; CuPy -> sequential xp path
    dtau = np.asarray(dtau)
    T_layers = np.asarray(T_layers)
    if dtau.ndim != 2 or T_layers.ndim != 2 or dtau.shape != T_layers.shape:
        return False
    if dtau.dtype.kind != "f" or T_layers.dtype.kind != "f":
        return False
    Ts = np.asarray(T_surf)
    if Ts.dtype.kind not in "fi":
        return False
    try:                                              # T_surf must broadcast to the (nx,) column axis
        np.broadcast_to(Ts, dtau.shape[:1])
    except ValueError:
        return False
    return True


_CLOUD_RAD_FREEZE_CACHE: dict = {}


def _cloud_rad_freeze(xp, f_like):
    """The COSMO_CLOUD_RAD_FREEZE cloud-locking profile, matched to ``f_like``'s module/dtype/shape.

    Returns ``(f, qc)`` arrays for the radiation to use in place of the live cloud field, or ``None``
    when the env is unset OR the stored profile's shape differs from ``f_like`` (a different grid,
    e.g. the nested chimney -- deliberately left live). Loaded once per (module, dtype, shape) and
    cached, so the hot loop -- and CUDA graph capture -- only ever reads a constant device buffer.
    """
    path = os.environ.get("COSMO_CLOUD_RAD_FREEZE")
    if not path:
        return None
    key = (path, xp.__name__, str(f_like.dtype), tuple(f_like.shape))
    hit = _CLOUD_RAD_FREEZE_CACHE.get(key)
    if hit is None:
        d = np.load(path)
        if tuple(d["f"].shape) != tuple(f_like.shape):
            hit = (None, None)
        else:
            hit = (xp.asarray(d["f"], dtype=f_like.dtype), xp.asarray(d["qc"], dtype=f_like.dtype))
        _CLOUD_RAD_FREEZE_CACHE[key] = hit
    return None if hit[0] is None else hit


def column_radiation(p_edge, T_layers, q, T_surf, S_toa, alpha_surf,
                     co2_ppm=CO2_REF_PPM, alpha_atm=ALBEDO_ATM,
                     k_wv=K_WV_LW, tau_wellmixed=TAU_LW_WELLMIXED, k_wv_sw=K_WV_SW,
                     eps_surf=1.0, *, clouds=False, cloud_rh_crit=CLOUD_RH_CRIT, cloud_water_scale=1.0,
                     cloud_conv=None, cloud_conv_boost=CLOUD_CONV_BOOST, cloud_sw_overlap="random",
                     cloud_sw_ssa_liq=CLOUD_SSA_LIQ, cloud_sw_ssa_ice=CLOUD_SSA_ICE,
                     cloud_sw_twostream=False,
                     cloud_qc=None, cloud_f_min=0.0, cloud_qc_thresh=0.0,
                     cloud_pdf=False, cloud_qc_decouple_cap=False, cloud_cover_floor=None,
                     cloud_n_d=None, cloud_n_d_ref=None,
                     sw_bands=False, f_vis=F_VIS_SW, k_wv_sw_nir=K_WV_SW_NIR,
                     lw_window=False, f_win=F_LW_WINDOW, k_cont=K_CONT_SELF,
                     lw_conserving_split=False,
                     sw_diffuse=False, S_diffuse=None, lw_parallel=False, lw_device_kernel=None,
                     sw_tau_haze=None):
    """Full SW+LW radiative heating of a column (or chain of columns).

    Returns dict with per-layer net radiative heating ``H`` [W/m^2], surface net radiative flux
    ``surf_net`` [W/m^2] (absorbed - emitted + downwelling), top-of-atmosphere net ``toa_net``
    [W/m^2] (absorbed SW - OLR; the global-balance diagnostic), and OLR.

    CLOUDS (issue #19, default OFF -> the clear-sky path is BYTE-IDENTICAL). With ``clouds=True`` a
    cloud-fraction field is diagnosed from the column's RH (``cloud_fraction``) and turned into per-layer
    cloud optics (``cloud_optical_properties``): the cloud LW optical depth is ADDED to the clear-sky
    dtau_lw (the greenhouse WARMING, run through the same two-stream so it depends on cloud-top T) -- and,
    when the LW window is also on, to the WINDOW band too (clouds are ~black across 8-13 um; GitHub issue
    #84), and a
    column SW cloud albedo (``cloud_sw_albedo``, the COOLING) is folded into the planetary reflection by
    raising the atmospheric albedo to alpha_atm + (1-alpha_atm)*A_cloud -- which conserves energy (the
    reflected SW never enters the column). The cloud diagnostics (per-layer ``cloud_fraction``, the column
    ``cloud_cover`` under random overlap, and ``cloud_sw_albedo``) are added to the returned dict ONLY when
    clouds are on. ``cloud_conv`` is the optional resolved-ascent boost signal (see ``cloud_fraction``).

    CLOUD SW ABSORPTION (``cloud_sw_twostream=True``, default OFF -> reflect-only, BYTE-IDENTICAL). The
    reflect-only cloud SW (above) makes the cloud a perfect reflector; the DEFAULT ``cloud_sw_ssa_*``=1 knob's
    absorption (folded into ``dtau_sw``) acts only on the post-reflection remnant beam, so a bright deck barely
    heats. With ``cloud_sw_twostream`` the cloud's reflectance R, ABSORPTANCE A_abs and transmittance T are
    solved JOINTLY (``two_stream_nc`` on the column-total optics), R+A_abs+T=1; R sets the TOA reflection and
    ``A_abs*S_incident`` is deposited as in-cloud HEATING (in ``shortwave``) -- the thermodynamic burn-off
    driver. ssa=1 -> A_abs=0, R = the column albedo -> byte-identical. The column ``cloud_sw_absorptance`` is
    added to the returned diagnostics. Under the two-band SW (``sw_bands``, the default) with ssa<1 the
    absorption is NEAR-IR-CONFINED (PR #332): the VISIBLE band is conservative (reflect-only, so the surface
    visible floor survives) and only the NEAR-IR band absorbs -- ``two_stream_nc`` is called per band and the
    per-band reflectances are passed to ``shortwave`` as ``cloud_R_vis``/``cloud_R_nir``; the ssa knobs are the
    near-IR SSA. (Gray-SW mode -- ``sw_bands=False``, non-production -- keeps the old grey whole-beam
    absorption.) Strongly ssa-sensitive (SWEEP ssa; magnitude soft). NB the two-stream reflectance is on the
    column-total footing, so ``cloud_sw_overlap`` is INERT under it (the ssa=1 byte-identical limit holds for
    overlap="column"; StageADCore warns on the random combo).

    PROGNOSTIC CONDENSATE (CLOUD_PROGNOSTIC_ROADMAP Chunk 1, default OFF -> ``cloud_qc=None`` is the
    BYTE-IDENTICAL adiabatic-diagnostic path). When ``cloud_qc`` (the grid-mean condensate [kg/kg]) is
    supplied the in-cloud water path comes from that RETAINED reservoir instead of the adiabatic
    reconstruction (``prognostic_cloud_water_path``); the RH cover is first FLOORED to ``cloud_f_min``
    wherever ``cloud_qc > cloud_qc_thresh`` (the cover<->condensate consistency rule, so the in-cloud
    ``rho*qc/f`` conversion stays finite as the RH cover -> 0).

    SUB-GRID PDF COVER (CLOUD_PROGNOSTIC_ROADMAP Chunk 3, ``cloud_pdf=True``; needs ``cloud_qc``). The
    RH-diagnostic cover + ``cloud_f_min`` floor are REPLACED by the total-water PDF cover
    (``cloud_pdf_partition`` on ``q + cloud_qc``), so cover and the condensation source share ONE footing
    (no cover without water). DECOUPLED RADIATIVE CAP (``cloud_qc_decouple_cap=True``): the prognostic
    in-cloud water is bounded by the PHYSICAL-plausibility LWC (``CLOUD_LWC_PLAUSIBLE``/``IWC_PLAUSIBLE``),
    NOT the adiabatic-rainout ceiling, so the cloud LWP TRACKS ``cloud_qc`` (rainout is the Chunk-2
    autoconversion sink's job). Both default OFF -> the Chunk-1 RH-cover/rainout-cap path is unchanged.

    BL/STRATOCUMULUS COVER FLOOR (CLOUD_PROGNOSTIC_ROADMAP Chunk 4, ``cloud_cover_floor=None`` default ->
    BYTE-IDENTICAL). A per-cell cover floor (same shape as ``f``): the cover is raised to
    ``max(f, cloud_cover_floor)`` after the RH/PDF cover is computed, so the subsidence-inversion
    stratocumulus deck (which the closure stamps with an overcast floor) radiates as a solid low cloud even
    where the grid-mean RH cover is ~0. Off-deck cells carry floor 0 -> ``f`` unchanged.

    TWO-BAND SW (CLEARSKY_SW_PLAN Stage 1; ``sw_bands=False`` default -> the gray SW path is
    BYTE-IDENTICAL). With ``sw_bands=True`` the solar beam is split into a transparent visible fraction
    ``f_vis`` and an absorbing near-IR fraction carrying ``k_wv_sw_nir`` (see ``shortwave``), so the
    clear-sky surface SW FLOORS at ~f_vis*(1-alpha_atm) instead of -> 0 at high water-vapour path, and only
    the near-IR fraction heats the layers. (When clouds are on, the REFLECT-ONLY cloud SW albedo is applied
    grey across both bands; but under the cloud SW two-stream ABSORPTION limb with ssa<1 the SW IS band-split --
    a conservative visible + an absorbing near-IR reflectance, PR #332 -- so cloud SW absorption is confined to
    the near-IR. The cloud LW is separately split into the window band, issue #84.)

    LW WINDOW (CLEARSKY_SW_PLAN Stage 2, default OFF -> the gray path is BYTE-IDENTICAL). With
    ``lw_window=True`` the LW is split into a background band (the gray ``dtau_lw``) + an 8-13 um window
    band whose opacity is the water-vapor self-continuum (``lw_window_optical_depth``) PLUS the cloud LW
    opacity when clouds are also on (issue #84; clouds are ~black across the window), via
    ``two_stream_lw_windowed``. The window OPENS in the dry limit (removes the gray scheme's total surface
    cloak) and CLOSES as humidity rises (the continuum), which caps the hot-end OLR. ``OLR_bg``/``OLR_win``
    (the per-band OLR) and ``dtau_win`` are added to the returned dict when the window is on.

    CONSERVING BAND SPLIT (issue #84 item 3, default OFF -> the accepted moved-baseline is BYTE-IDENTICAL).
    The Earth-calibrated clear-sky gray opacity (``k_wv``/``tau_wellmixed``) is a FULL-SPECTRUM Planck-mean;
    handing the window band (Planck weight ``f_win``) only the continuum while the background band keeps the
    unscaled gray opacity makes the dry-limit total greenhouse WEAKER than the calibration intended (the
    "gray-baseline drift"). With ``lw_conserving_split=True`` the background band carries the higher
    out-of-window CLEAR-SKY opacity ``dtau_clear/(1-f_win)``, so the thin-limit Planck-mean absorption
    reduces to the Earth-calibrated gray total. The CLOUD opacity is NOT rescaled (it is computed from
    cloud physics per band -- clouds are ~gray across the thermal spectrum -- not Earth-anchored).

    SW DIFFUSE (CLEARSKY_SW_PLAN Stage 5, default OFF -> the gray path is BYTE-IDENTICAL). With
    ``sw_diffuse=True`` and a per-column ``S_diffuse`` (the haze forward-scatter day-mean flux from
    ``geometry.day_mean_diffuse_flux``), a diffuse-down SW stream is added in ``shortwave`` (see there): it
    recovers the slant-haze light the direct flux ``S_toa`` drops as pure loss, RAISING the surface SW. The
    diffuse input is counted in the TOA SW budget (``toa_net`` uses absorbed SW = ``S_toa + S_diffuse -
    planetary_reflected``), so callers' energy closure stays consistent. Off (default) -> ``S_diffuse`` is
    ignored and the budget is the pre-Stage-5 ``S_toa``-only one (byte-identical).
    """
    xp = get_xp(q)                                        # numpy/CuPy dispatch (Phase D); q is always present
    dtau_lw = lw_optical_depth(p_edge, q, co2_ppm, k_wv, tau_wellmixed)
    dtau_lw_clear = dtau_lw                                # clear-sky gray opacity (pre-cloud), for the
                                                           # conserving band split below (issue #84 item 3)
    # SW optical depth: gray uses k_wv_sw on the whole beam; the two-band path (Stage 1) passes the NEAR-IR
    # optical depth (k_wv_sw_nir) -- only the near-IR fraction (1-f_vis) carries it (see ``shortwave``).
    dtau_sw = sw_optical_depth(p_edge, q, k_wv_sw_nir if sw_bands else k_wv_sw)
    sw_f_vis = f_vis if sw_bands else None
    alpha_atm_eff = alpha_atm
    cloud_sw_abs_frac = None                              # cloud SW two-stream absorptance limb (below); None -> off
    cloud_sw_abs_w = None
    cloud_R_vis = cloud_R_nir = None                      # two-band cloud optics (near-IR-confined abs); None -> grey
    cloud_diag = None
    if clouds:
        p_edge_a = _F(p_edge)
        p_mid = 0.5 * (p_edge_a[..., :-1] + p_edge_a[..., 1:])
        dm = layer_mass(p_edge_a)
        if cloud_qc is not None and cloud_pdf:
            # SUB-GRID PDF COVER (CLOUD_PROGNOSTIC_ROADMAP Chunk 3): cover from the total-water PDF
            # (cloud_pdf_partition on q + qc), the SAME distribution whose saturated-tail integral is the
            # microphysics condensation source -- so cover and source share ONE footing (no cover without
            # water, the haze fix; this REPLACES the Chunk-1 RH cover + f_min floor). The continuous PDF cover
            # also smooths the floor's step at cloud_qc_thresh (PR #142 review #3).
            f, _ = cloud_pdf_partition(_F(q) + _F(cloud_qc),
                                       T_layers, p_mid, cloud_rh_crit)
            if cloud_conv is not None and cloud_conv_boost > 0.0:   # same convective-ascent nudge as the RH path
                f = xp.clip(f + (1.0 - f) * cloud_conv_boost * xp.clip(_F(cloud_conv), 0.0, 1.0),
                            0.0, 1.0)
        else:
            f = cloud_fraction(q, T_layers, p_mid, cloud_rh_crit, cloud_conv, cloud_conv_boost)
            if cloud_qc is not None:
                # PROGNOSTIC condensate Chunk-1 COVER<->CONDENSATE consistency floor (load-bearing): clear air
                # carrying condensate is unphysical -- a layer with qc > thresh gets a minimal cover f >= f_min,
                # so the in-cloud LWP = rho*qc/f*dz conversion cannot blow up as the RH-diagnostic cover -> 0.
                # NB (PR #142 review #3): a STEP in cover at cloud_qc_thresh; Chunk 3's PDF cover (above) smooths it.
                f = xp.where(_F(cloud_qc) > cloud_qc_thresh, xp.maximum(f, cloud_f_min), f)
        if cloud_cover_floor is not None:
            # CHUNK-4 BL/stratocumulus deck cover floor (CLOUD_PROGNOSTIC_ROADMAP): the closure stamps an
            # OVERCAST cover on its subsidence-inversion deck cells, so the deck radiates as a solid low cloud
            # (consistent with the source's emissivity, which assumes the same overcast in-cloud rho*qc/f).
            # Off-deck cells carry floor 0 -> f unchanged. Default None -> byte-identical.
            f = xp.maximum(f, _F(cloud_cover_floor))
        # CLOUD-RADIATIVE LOCKING (mechanism probe, default OFF -> byte-identical). With
        # COSMO_CLOUD_RAD_FREEZE=<npz: f,qc at this grid> the RADIATION sees a fixed cloud field (cover
        # AND condensate) in place of the live one, while microphysics/dynamics run free -- the standard
        # cloud-locking test for cloud-radiation-feedback-driven variability. Only the cloud terms are
        # locked; the water-vapor radiative feedback stays live. Grids whose shape doesn't match the
        # stored profile (e.g. the nested chimney) are left live.
        _frz = _cloud_rad_freeze(xp, f)
        if _frz is not None:
            f, cloud_qc = _frz
        if cloud_qc is not None:
            # DECOUPLED RADIATIVE CAP (Chunk 3): bound the in-cloud water at the physical-plausibility LWC
            # (a few g/m^3) so the LWP TRACKS qc, NOT at the rainout cap (which would double-count the Chunk-2
            # autoconversion sink). OFF -> the Chunk-1 rainout cap (CLOUD_LWC_MAX), via the function default.
            cap_kw = (dict(lwc_max=CLOUD_LWC_PLAUSIBLE, iwc_max=CLOUD_IWC_PLAUSIBLE)
                      if cloud_qc_decouple_cap else {})
            dtau_lw_cloud, R_in, tau_sw_cloud, dtau_sw_cloud = cloud_optical_properties(
                f, T_layers, p_mid, dm, water_scale=cloud_water_scale, qc=cloud_qc, return_tau_sw=True,
                ssa_liq=cloud_sw_ssa_liq, ssa_ice=cloud_sw_ssa_ice,
                n_d=cloud_n_d, n_d_ref=cloud_n_d_ref, **cap_kw)
        else:
            dtau_lw_cloud, R_in, tau_sw_cloud, dtau_sw_cloud = cloud_optical_properties(
                f, T_layers, p_mid, dm, water_scale=cloud_water_scale, return_tau_sw=True,
                ssa_liq=cloud_sw_ssa_liq, ssa_ice=cloud_sw_ssa_ice,
                n_d=cloud_n_d, n_d_ref=cloud_n_d_ref)
        dtau_lw = dtau_lw + dtau_lw_cloud                  # cloud greenhouse: add LW opacity per layer
        cloud_sw_abs_col = None
        if cloud_sw_twostream:
            # NON-CONSERVATIVE two-stream cloud SW: the cloud REFLECTS R, ABSORBS A_abs (-> deck heating,
            # the burn-off feedback) and TRANSMITS T, R+A_abs+T=1. On the column-total footing (like
            # overlap="column"): tau_sw_cloud = per-layer SW SCATTERING opt depth, dtau_sw_cloud = per-layer
            # (grid-mean, f-weighted) ABSORPTION opt depth. The absorption is deposited in ``shortwave`` (from
            # the incident beam), NOT folded into dtau_sw (that remnant-beam path under-heats a bright deck ~25-50x).
            tau_scat_tot = xp.sum(f * _F(tau_sw_cloud), axis=-1)      # f-weighted column SW scattering opt depth
            tau_abs_tot = xp.sum(_F(dtau_sw_cloud), axis=-1)          # f-weighted column SW absorbing opt depth
            cloud_sw_abs_w = _F(dtau_sw_cloud) / xp.maximum(          # per-layer deposition weights (sum_k = 1)
                xp.sum(_F(dtau_sw_cloud), axis=-1, keepdims=True), 1e-30)
            if sw_bands and (cloud_sw_ssa_liq < 1.0 or cloud_sw_ssa_ice < 1.0):
                # TWO-BAND cloud optics (near-IR-confined cloud SW absorption). Real liquid water is
                # ~non-absorbing in the VISIBLE (ssa~1-1e-7) -> the visible band is CONSERVATIVE (reflectance
                # R_vis = two_stream_nc(tau_scat, 0), A_vis=0); cloud SW absorption is confined to the NEAR-IR
                # (R_nir + A_nir from the ssa<1 two-stream). This FIXES the grey-spectral limb, which absorbed
                # the water-transparent visible band too and drove surface light ~10^2-10^3x too dark
                # (CLOUDS_RESULTS.md). The per-band reflection + near-IR absorption are applied in ``shortwave``
                # (cloud_R_vis/cloud_R_nir); alpha_atm_eff stays the CLEAR-sky Rayleigh. ssa=1 skips this branch
                # (-> grey sub-branch below, BYTE-IDENTICAL to the reflect-only column).
                R_nir, A_nir_col, _T = two_stream_nc(tau_scat_tot, tau_abs_tot, CLOUD_G_ASYM)
                # Visible SSA~1 -> the visible SCATTERING depth is the FULL f-weighted extinction, NOT the
                # near-IR-ssa-scaled tau_scat_tot. tau_scat_tot + tau_abs_tot = sum f*tau_ext (the ssa split of
                # one extinction), so this makes R_vis exactly ssa-INDEPENDENT -- the visible does not know about
                # near-IR absorption (no near-IR-ssa leak into the visible reflectance). (PR #332 review F1.)
                R_vis, _Av, _Tv = two_stream_nc(tau_scat_tot + tau_abs_tot,
                                                xp.zeros_like(tau_abs_tot), CLOUD_G_ASYM)  # conservative, full ext
                cloud_sw_abs_frac = A_nir_col / xp.maximum(1.0 - R_nir, 1e-12)   # near-IR absorbed frac of entering beam
                cloud_R_vis, cloud_R_nir = R_vis, R_nir
                A_cloud = f_vis * R_vis + (1.0 - f_vis) * R_nir          # beam-weighted effective column albedo (diag)
                cloud_sw_abs_col = (1.0 - f_vis) * A_nir_col             # beam-weighted column absorptance (diag)
                alpha_atm_eff = alpha_atm                                # cloud reflection handled PER BAND in shortwave
            else:
                # GREY whole-beam two-stream: ssa=1 (no absorption -> BYTE-IDENTICAL) OR gray-SW (no visible band).
                A_cloud, cloud_sw_abs_col, _T_cloud = two_stream_nc(tau_scat_tot, tau_abs_tot, CLOUD_G_ASYM)
                cloud_sw_abs_frac = cloud_sw_abs_col / xp.maximum(1.0 - A_cloud, 1e-12)  # frac of the ENTERING beam
                alpha_atm_eff = alpha_atm + (1.0 - alpha_atm) * A_cloud
        else:
            dtau_sw = dtau_sw + dtau_sw_cloud              # legacy: cloud SW absorption on the remnant beam (0 at ssa=1)
            A_cloud = cloud_sw_albedo(f, R_in, overlap=cloud_sw_overlap, tau_sw=tau_sw_cloud)  # column SW cloud albedo
            alpha_atm_eff = alpha_atm + (1.0 - alpha_atm) * A_cloud   # fold reflection into the planetary reflection
        cloud_cover = 1.0 - xp.prod(1.0 - f, axis=-1)      # column cloud cover (random overlap of f)
        cloud_diag = dict(cloud_fraction=f, cloud_cover=cloud_cover, cloud_sw_albedo=A_cloud,
                          dtau_lw_cloud=dtau_lw_cloud)
        if cloud_sw_abs_col is not None:
            cloud_diag["cloud_sw_absorptance"] = cloud_sw_abs_col    # column SW absorptance (two-stream limb)
    if lw_window:
        # Stage 2: split LW into a background band (the gray dtau_lw, with any cloud opacity folded in) +
        # an 8-13 um window band carrying the clear-sky self-continuum.
        dtau_win = lw_window_optical_depth(p_edge, q, k_cont)
        if clouds:
            # CLOUD-WINDOW COUPLING (GitHub issue #84): water/ice clouds are nearly black across the 8-13 um
            # window, so the SAME cloud LW optical depth belongs in BOTH bands -- the cloud absorbs the
            # surface's window emission just as it does the background. Folding it into the background band
            # alone (the all-OFF-defaults era's deferral) left the window cloud-TRANSPARENT, understating the
            # low-cloud greenhouse by ~f_win exactly where the window is OPEN (the cool/dry ribbon & poleward
            # columns); where humidity has already closed the window (hot/moist) it adds ~nothing. Cloud LW
            # absorption is fairly flat over 8-13 um, so reusing the same kabs (-> dtau_lw_cloud) is the
            # standard first take, not a window-specific coefficient.
            dtau_win = dtau_win + dtau_lw_cloud
        # Conserving band split (issue #84 item 3): rescale only the CLEAR-SKY share of the background
        # band by 1/(1-f_win) so the two bands reduce to the Earth-calibrated gray Planck-mean in the
        # thin/dry limit; cloud opacity enters both bands unscaled. OFF (default) -> the accepted
        # moved-baseline background band (the unscaled gray total), byte-identical.
        if lw_conserving_split:
            dtau_bg = dtau_lw_clear / (1.0 - f_win) + (dtau_lw - dtau_lw_clear)
        else:
            dtau_bg = dtau_lw
        # NB the gate validates only ``dtau_bg``, but the windowed parallel path runs the kernel on BOTH
        # ``dtau_bg`` and ``dtau_win``. This relies on the production invariant that the two bands share the
        # (nx, N) fp64 shape: ``dtau_win`` is ``lw_window_optical_depth(p_edge, q, ...)`` (+ the same-shape
        # ``dtau_lw_cloud``), built from the SAME column grid as ``dtau_bg`` (the gray ``dtau_lw``). So a
        # gate pass on ``dtau_bg`` implies ``dtau_win`` is kernel-eligible too. ``two_stream_lw_windowed_parallel``
        # itself dispatches each band through the same numba kernel; were the shapes ever to diverge the kernel
        # would error rather than silently mis-compute, so this is a guarded assumption, not a silent hazard.
        if _lw_device_ok(lw_device_kernel, dtau_bg, T_layers, T_surf):
            from ._gpu_kernels import two_stream_lw_windowed_device   # fused device kernel (GPU path)
            ker = lw_device_kernel(dtau_bg.dtype)                     # getter -> dtype-matched RawKernel
            lw = two_stream_lw_windowed_device(ker, dtau_bg, dtau_win, T_layers, T_surf,
                                               f_win, eps_surf, D_DIFF)
        elif _lw_parallel_ok(lw_parallel, dtau_bg, T_layers, T_surf):
            lw = two_stream_lw_windowed_parallel(dtau_bg, dtau_win, T_layers, T_surf, f_win, eps_surf)
        else:
            lw = two_stream_lw_windowed(dtau_bg, dtau_win, T_layers, T_surf, f_win, eps_surf)
    else:
        if _lw_device_ok(lw_device_kernel, dtau_lw, T_layers, T_surf):
            from ._gpu_kernels import two_stream_lw_cols_device
            ker = lw_device_kernel(dtau_lw.dtype)
            lw = two_stream_lw_cols_device(ker, dtau_lw, T_layers, T_surf, eps_surf, D_DIFF, 1.0)
        elif _lw_parallel_ok(lw_parallel, dtau_lw, T_layers, T_surf):
            lw = two_stream_lw_parallel(dtau_lw, T_layers, T_surf, eps_surf)
        else:
            lw = two_stream_lw(dtau_lw, T_layers, T_surf, eps_surf)
    sw_S_diffuse = S_diffuse if sw_diffuse else None         # Stage 5: haze forward-scatter (default OFF)
    sw = shortwave(S_toa, dtau_sw, alpha_surf, alpha_atm_eff, f_vis=sw_f_vis, S_diffuse=sw_S_diffuse,
                   tau_haze=sw_tau_haze, cloud_abs=cloud_sw_abs_frac, cloud_w=cloud_sw_abs_w,
                   cloud_R_vis=cloud_R_vis, cloud_R_nir=cloud_R_nir)
    H = lw["H"] + sw["H"]
    surf_net = sw["surf_abs"] - lw["surf_net_lw"]           # SW abs + LW down - LW up
    # absorbed SW = (column-top down-SW) - reflected; the down-SW includes the Stage-5 diffuse haze
    # forward-scatter when it is on (else just S_toa, so toa_net is byte-identical to the pre-Stage-5 form).
    S_sw_in = _F(S_toa) + (_F(S_diffuse) if sw_S_diffuse is not None else 0.0)
    toa_net = (S_sw_in - sw["planetary_reflected"]) - lw["OLR"]
    out = dict(H=H, surf_net=surf_net, toa_net=toa_net, OLR=lw["OLR"],
               D_surf=lw["D_surf"], surf_abs_sw=sw["surf_abs"], surf_down_sw=sw["surf_down"],
               planetary_reflected=sw["planetary_reflected"],
               dtau_lw=dtau_lw, dtau_sw=dtau_sw)
    if cloud_diag is not None:
        out.update(cloud_diag)
    if lw_window:
        out.update(dtau_win=dtau_win, OLR_bg=lw["OLR_bg"], OLR_win=lw["OLR_win"])
    return out


# ======================================================================================
# Analytic gray-atmosphere radiative equilibrium -- for TESTING the band model.
# ======================================================================================
def gray_air_temperature(tau, T_eff, D=D_DIFF):
    """Analytic gray radiative-equilibrium AIR temperature at LW optical depth ``tau`` (from TOA).

    In radiative equilibrium the net LW flux is constant (= OLR = sigma T_eff^4), which gives the
    classic two-stream result  sigma T(tau)^4 = (1/2) sigma T_eff^4 (1 + D*tau).  Hence the skin
    temperature (tau=0) is T_eff / 2^{1/4}. (The surface sits warmer, with a surface-air jump.)
    """
    return (0.5 * T_eff**4 * (1.0 + D * tau)) ** 0.25


def radiative_equilibrium_gray(p_edge, T_eff, tau_surf, n_iter=40000, D=D_DIFF):
    """Relax a column to gray RADIATIVE equilibrium with the band model, for validation against the
    analytic gray solution. No atmospheric SW absorption; all SW (= sigma T_eff^4) absorbed at the
    surface; q=0 so the LW optical depth is purely the well-mixed gray total ``tau_surf`` distributed
    by mass. Returns (T_layers, T_surf, tau_at_layer_centers).

    HOST-ONLY (numpy): a single-column scalar relaxation loop (``float(T_eff)``, ``dm.min()``, host
    early-stop on ``np.max(abs(dT))``). It is a TEST/validation helper for the analytic gray limit, not
    on any GPU production path, so it is left numpy-only by design (see archive/_gpu_rad_port_RESULTS.md).
    """
    p_edge = _F(p_edge)
    N = p_edge.size - 1
    dm = layer_mass(p_edge)
    dtau = tau_surf * dm / dm.sum()                         # gray: optical depth distributed by mass
    tau_center = np.cumsum(dtau) - 0.5 * dtau               # optical depth at each layer center
    SW_surf = C.SIGMA_SB * T_eff**4                         # all SW absorbed at the surface
    T = np.full(N, T_eff, float)
    Ts = float(T_eff)
    Clay = dm * C.CP_AIR                                    # layer heat capacities [J/m^2/K]
    Cs = C.C_OCEAN_MIXED
    dt = 0.3 * Clay.min() / (4 * C.SIGMA_SB * T_eff**3)     # stable relaxation step
    for it in range(n_iter):
        lw = two_stream_lw(dtau, T, Ts)
        dT = dt * lw["H"] / Clay
        T = T + dT
        Ts = Ts + dt * (SW_surf - lw["surf_net_lw"]) / Cs
        if it % 1000 == 0 and np.max(np.abs(dT)) < 1e-6:       # early stop once relaxed
            break
    return T, Ts, tau_center
