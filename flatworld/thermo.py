"""
flatworld.thermo -- moist thermodynamics (real physics, shared by all stages).

Saturation vapor pressure, saturation specific humidity, the moist-adiabatic lapse rate, and bulk
surface-flux helpers. Nothing here is setting-specific; it is ordinary atmospheric thermodynamics,
used so the model's humidity, greenhouse, and convection are physical rather than tuned.
"""
from __future__ import annotations
import numpy as np
from . import constants as C
from .device import get_xp


# --- thermo working dtype (COSMO_THERMO_FP32 long-run-bias gate) -------------------------------------
# The shared thermo leaf-math coerces its inputs to a working float dtype via ``_F`` (was a bare
# ``_F(x)`` everywhere = unconditional float64). ``_FLOAT`` defaults to float64, so every
# ``_F(x)`` is a value-preserving no-op and the module stays BYTE-IDENTICAL. The StageAD core's
# ``COSMO_THERMO_FP32`` knob flips it to float32 around the column-thermo substep ONLY (dynamics + the
# persistent prognostic state stay float64), so the thermo ARITHMETIC genuinely runs single precision --
# the point of the gate. The bare Python-float constants in ``constants`` are dtype-"weak" (they preserve
# float32 in mixed ops), so once the inputs are float32 the whole expression evaluates in float32.
#
# GPU port: ``_F`` and the elementwise leaf-math are array-module-generic. ``_F(x)`` places ``x`` on the
# SAME module its input lives on (``get_xp(x)``: numpy in -> numpy out, CuPy in -> CuPy out) and the
# functions open with ``xp = get_xp(<input>)`` then use ``xp.<op>`` -- so a CuPy array runs on the GPU
# with NO host round-trip, while a numpy caller is BYTE-IDENTICAL to the pre-port code. The fp32 gate is
# ORTHOGONAL to the device: ``_F`` still hard-coerces to the GLOBAL ``_FLOAT`` (so ``set_thermo_dtype``
# keeps downcasting float64->float32 exactly as #196 intends -- NOT input-dtype-following), and the
# "fp32 in -> fp32 out" GPU precision is achieved the same way the CPU gate gets it: via
# ``set_thermo_dtype(np.float32)``.
_FLOAT = np.float64


def set_thermo_dtype(dt):
    """Set the thermo working dtype (np.float64 default = byte-identical, np.float32 = the fp32 gate).
    Returns the PREVIOUS dtype so the caller can restore it."""
    global _FLOAT
    prev = _FLOAT
    _FLOAT = np.dtype(dt).type
    return prev


def _xp_of(*args):
    """Pick the array module for a multi-arg function: CuPy iff ANY argument is a CuPy ndarray, else
    numpy. (Single ``get_xp`` on the leading array usually suffices since T/Ts lead every signature,
    but this is robust to a mixed call -- a CuPy array + a Python/numpy scalar -- which must run on the
    device, not host-transfer.)"""
    for a in args:
        if get_xp(a) is not np:
            return get_xp(a)
    return np


def _F(x, xp=None):
    """Coerce to the thermo working dtype on a chosen array module (``_FLOAT``, GLOBAL working dtype, so
    the fp32 gate's downcast still bites regardless of device).

    ``xp=None`` -> use x's OWN module (``get_xp(x)``): the single-array entry points (``esat`` etc.) want
    this. Multi-arg functions pass an EXPLICIT ``xp`` (the function-level module, CuPy iff any arg is on
    the device) so a Python/numpy scalar arg is placed on the SAME device as the array args -- without it
    a numpy 0-d scalar would collide with a CuPy array in the arithmetic. numpy in (xp=numpy) is
    byte-identical to the pre-port ``np.asarray(x, float)``."""
    return (xp or get_xp(x)).asarray(x, _FLOAT)


def esat(T):
    """Saturation vapor pressure [Pa] over liquid water (T>=0 C) / ice (T<0 C).

    Bolton (1980) over liquid; Murphy-Koop-style ice branch below freezing. T in Kelvin.
    """
    xp = get_xp(T)
    T = _F(T)
    Tc = T - C.T_FREEZE
    # Guard the Magnus/Bolton POLES: each denominator hits zero at Tc = -243.5 C (liquid, T=29.65 K) /
    # -265.5 C (ice, T=7.65 K); just BELOW the pole the exponent flips sign and overflows to +inf ->
    # esat=inf -> qsat=NaN (a hard ~30 K wall that blows up any cell trying to cool colder). Clamp each
    # branch's Tc just above its own pole so esat stays finite (underflows to ~0, the physically correct
    # tiny value) for arbitrarily cold T. BYTE-IDENTICAL for the SELECTED branch at all T >= ~8 K: below
    # freezing e_ice is chosen and e_ice is unchanged for T >= 7.65 K (its pole); the e_liq clamp only
    # ever touches values the ``where`` discards for cold T. Lets the far field cool toward its true ~0 K
    # radiative limit (no sun, transport -> 0 at large x) instead of NaN-ing at the pole.
    Tc_liq = xp.maximum(Tc, -243.0)                            # denom >= 0.5 -> exp underflows, no overflow
    Tc_ice = xp.maximum(Tc, -265.0)                            # denom >= 0.5 (ice branch, pole -265.5)
    e_liq = 611.2 * xp.exp(17.67 * Tc_liq / (Tc_liq + 243.5))
    e_ice = 611.2 * xp.exp(21.87 * Tc_ice / (Tc_ice + 265.5))
    return xp.where(Tc >= 0.0, e_liq, e_ice)


# The vapor-pressure clamp fraction used by ``qsat`` (e <= E_FRAC_MAX * p), and THE physical ceiling on
# specific humidity it implies. ONE number, ONE home: any state cap on q must come from here, never from
# a re-typed literal -- a state cap BELOW Q_MAX silently deletes water in steam columns (#294 spin-off A:
# the q<=0.5 Earth-era guard bound at qsat(92.6 C)=0.694 on the lens world and leaked ~15% of all
# evaporation, invisible to water_conservation because the diagnostics clipped too).
E_FRAC_MAX = 0.999
# q at the clamp. Note p CANCELS -- this is a pure constant, not a profile: with e = f*p,
#   q = eps*f*p / (p - (1-eps)*f*p) = eps*f / (1 - (1-eps)*f).
Q_MAX = C.EPS_MW * E_FRAC_MAX / (1.0 - (1.0 - C.EPS_MW) * E_FRAC_MAX)   # ~0.998393 kg/kg (~pure steam)


def qsat(T, p):
    """Saturation specific humidity [kg/kg] at temperature T [K], pressure p [Pa].

    The vapor pressure is clamped to e <= E_FRAC_MAX * p: at temperatures so high that esat exceeds
    the total pressure the air is supercritical (the surface water flash-boils) and the
    specific-humidity formula would diverge -- physically q cannot exceed ~1, so we cap there, at
    ``Q_MAX``. Below boiling this clamp never bites.

    NOTE (was wrong before #294-A): the old docstring claimed this "only guards the lethal hot zone
    where there is no liquid ocean anyway". It does NOT -- a boiling *shoreline* (liquid water under
    super-boiling air, e.g. run15's 92 C sea at theta~22) sits exactly here, evaporating hard, and is
    the regime the clamp most affects."""
    xp = _xp_of(T, p)
    T = _F(T, xp); p = _F(p, xp)
    e = xp.minimum(esat(T), E_FRAC_MAX * p)
    return C.EPS_MW * e / (p - (1.0 - C.EPS_MW) * e)


def T_boil(p):
    """Boiling temperature [K] at pressure ``p`` [Pa] *in the model's own esat*: the T where
    ``esat(T) = 0.999 p`` — the ``qsat`` clamp threshold — so ``T >= T_boil(p)`` is exactly the
    boiling/supercritical mask ``esat(T) >= 0.999 p`` the microphysics uses (#292/#294). Closed-form
    inverse of the warm-branch Bolton formula: with ``y = ln(0.999 p / 611.2)``,
    ``Tc = 243.5 y / (17.67 - y)``. Warm (liquid) branch only — valid for p > ~612 Pa (where
    T_boil > freezing), which covers every model pressure (p_top ~ 5 kPa); ~99.9 kPa gives ~99 °C
    (Bolton's small offset from the steam-table 100 °C is *correct* here — consistency with the
    model's ``esat`` is the requirement, not the steam table)."""
    xp = get_xp(p)
    p = _F(p)
    y = xp.log(0.999 * p / 611.2)
    return C.T_FREEZE + 243.5 * y / (17.67 - y)


def wet_bulb(T_C, RH):
    """Stull (2011) empirical wet-bulb temperature [deg C] from dry-bulb ``T_C`` [deg C] and relative
    humidity ``RH`` (fraction, 0-1). Accurate to ~0.3 deg C over RH 5-99 %, T -20..50 deg C at ~sea level
    (J. Appl. Meteor. Climatol. 50, 2267) -- RH is clipped into [5, 99] % to stay in range. Used to flag the
    humid-heat lethality classes (the warm, near-saturated ribbon edge); the dry hot salt flat has low RH
    and so a low wet-bulb, and is caught by its own (higher-priority) class anyway."""
    xp = get_xp(T_C)
    T = _F(T_C, xp)
    rh = xp.clip(_F(RH, xp) * 100.0, 5.0, 99.0)
    return (T * xp.arctan(0.151977 * xp.sqrt(rh + 8.313659))
            + xp.arctan(T + rh) - xp.arctan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * xp.arctan(0.023101 * rh) - 4.686035)


def wet_bulb_high(T_avg_C, T_high_C, RH):
    """Daily-HIGH wet-bulb temperature [deg C] at CONSERVED VAPOR (GEO-20a) -- the field the humid-heat
    classes read under heat_law=2, and the ONE source for it (mapdata carries a committed snapshot).

    The day's water content is set by the day-mean state; the warm hour heats that SAME air. So the
    high-hour RH is e_mean/esat(T_high) with e_mean = RH * esat(T_avg) -- NOT the day-mean RH.
    Evaluating Stull at (T_high, RH_mean) instead holds RH fixed while T rises, which silently ADDS
    vapor with temperature and overstates the high wet-bulb by several K exactly where the WW/Wd calls
    are made. (Conserved-vapor is itself a mild overstatement where the warm hour entrains drier air
    aloft, so this is the conservative-but-honest bound, not a lower one.)

    Validity: Stull's fit is stated to T = 50 deg C; above that this is an extrapolation, tolerated
    because by there the verdict is decided by the ~20 deg C dry-limit threshold floor and the class's
    own softness exceeds the fit error. RH clipping into Stull's [5, 99]% happens inside wet_bulb --
    and the LOW clip is the binding one (PR #453 review): wherever rh_high < 5% (16% of land cells on
    the shipped archive) the returned value is the clip floor wb(T_high, 5%) ~ 0.506*T_high - 4.39,
    not the physical wet-bulb -- so above ~51.4 deg C daily-high dry-bulb this field carries NO
    humidity discrimination, and at the law's 55-deg-dry anchor it reads ~23.4 where the true
    psychrometric wet-bulb is ~20.1. Today every such land cell classifies SF first (measured: a full
    psychrometric swap moves 0.46% of land), so nothing bites. GEO-21 reshuffled SF and the shield was
    RE-MEASURED on the rebuilt law-3 archive (PR #454): still 0 clip-live cells outside SF, and 0 of
    the 967 CSF cells clip-live, on the ~40 km climate grid and the 3 km display grid alike. The
    shield is EMPIRICAL, not structural -- SF's gate can fail on rainfall alone, leaving a genuinely
    hot cell (882 such cells on the shipped window), so re-measure when the geography or the canonical
    world moves; see EXTENDED_CLIMATE_CLASSES.md. The heat_thresholds_C anchors are
    stated on the TRUE-wet-bulb scale; on this clipped scale they are ~2-3 K generous near the dry
    limit."""
    xp = get_xp(T_avg_C)
    e_mean = _F(RH, xp) * esat(_F(T_avg_C, xp) + C.T_FREEZE)
    rh_high = e_mean / esat(_F(T_high_C, xp) + C.T_FREEZE)
    return wet_bulb(T_high_C, rh_high)


def desat_dT(T):
    """d(esat)/dT [Pa/K], differentiating the Bolton/ice esat formula exactly (so it is consistent
    with ``esat`` rather than an idealized constant-L Clausius-Clapeyron)."""
    xp = get_xp(T)
    T = _F(T)
    Tc = T - C.T_FREEZE
    e = esat(T)
    d_liq = e * 17.67 * 243.5 / (Tc + 243.5) ** 2
    d_ice = e * 21.87 * 265.5 / (Tc + 265.5) ** 2
    return xp.where(Tc >= 0.0, d_liq, d_ice)


def dqsat_dT(T, p):
    """d(qsat)/dT [kg/kg/K]. Uses the exact derivative of ``esat`` for self-consistency:
    qsat = eps e/(p-(1-eps)e)  =>  dqsat/dT = eps p (de/dT)/(p-(1-eps)e)^2."""
    xp = _xp_of(T, p)
    T = _F(T, xp)
    p = _F(p, xp)
    e = esat(T)
    clamped = e >= 0.999 * p            # supercritical: qsat capped, derivative ~0
    e = xp.minimum(e, 0.999 * p)
    denom = p - (1.0 - C.EPS_MW) * e
    return xp.where(clamped, 0.0, C.EPS_MW * p * desat_dT(T) / denom**2)


def moist_adiabat_lapse_rate(T, p):
    """Saturated (moist) adiabatic lapse rate dT/dz [K/m] (negative of, i.e. magnitude of cooling).

    Standard formula:  Gamma_m = g/cp * (1 + L qs/(Rd T)) / (1 + L^2 qs eps/(cp Rd T^2)).
    Reduces to the dry adiabat g/cp when qs->0.
    """
    T = _F(T, _xp_of(T, p))
    qs = qsat(T, p)
    num = 1.0 + C.L_VAP * qs / (C.R_DRY * T)
    den = 1.0 + C.L_VAP**2 * qs * C.EPS_MW / (C.CP_AIR * C.R_DRY * T**2)
    return (C.G_ACCEL / C.CP_AIR) * num / den


def dry_adiabat_lapse_rate():
    """Dry adiabatic lapse rate g/cp [K/m] ~ 9.8/1004 ~ 0.00976 K/m."""
    return C.G_ACCEL / C.CP_AIR


def temperature_on_moist_adiabat(T_surf, p_surf, p_levels):
    """Integrate a moist adiabat upward from (T_surf, p_surf) to the given pressure levels [Pa].

    Returns T [K] at each p in p_levels. Uses dT/dp = (R T / (cp p)) * moist correction, integrated
    with small steps. p_levels need not be sorted finely; we sub-step in log-pressure for accuracy.

    HOST-ONLY (validation/diagnostic): a sequential Python-scalar log-pressure integration (``T`` and
    ``p`` are Python floats inside the loop), NOT a GPU kernel and NOT on the AD-core hot path -- so it
    is deliberately left numpy-resident (byte-identical for a numpy ``p_levels``; pass a host array).
    """
    p_levels = _F(p_levels)
    order = np.argsort(-p_levels)                      # integrate from high p (surface) upward
    p_sorted = p_levels[order]
    T_out = np.empty_like(p_sorted)
    T = float(T_surf)
    p = float(p_surf)
    for i, p_tgt in enumerate(p_sorted):
        # sub-step in log-pressure from current p down to p_tgt
        n_sub = max(1, int(np.ceil(abs(np.log(p / max(p_tgt, 1.0))) / 0.02)))
        lnp = np.linspace(np.log(p), np.log(max(p_tgt, 1.0)), n_sub + 1)
        for j in range(n_sub):
            qs = qsat(T, p)
            num = 1.0 + C.L_VAP * qs / (C.R_DRY * T)
            den = 1.0 + C.L_VAP**2 * qs * C.EPS_MW / (C.CP_AIR * C.R_DRY * T**2)
            dlnT_dlnp = (C.R_DRY / C.CP_AIR) * num / den   # = d ln T / d ln p along the moist adiabat
            dlnp = lnp[j + 1] - lnp[j]
            T = T * np.exp(dlnT_dlnp * dlnp)
            p = np.exp(lnp[j + 1])
        T_out[i] = T
    # unsort
    result = np.empty_like(p_levels)
    result[order] = T_out
    return result


def saturation_mse(T, p, z):
    """Saturation moist static energy [J/kg] = cp T + g z + L qsat(T,p). For CAPE / convective tests."""
    xp = _xp_of(T, p, z)
    return C.CP_AIR * _F(T, xp) + C.G_ACCEL * _F(z, xp) + C.L_VAP * qsat(T, p)


def mse(T, p, z, q):
    """Moist static energy [J/kg] = cp T + g z + L q, for an arbitrary (not necessarily saturated)
    specific humidity q. (``saturation_mse`` is this with q = qsat.) L_VAP is taken constant, consistent
    with the rest of the module -- the T-dependence of the latent heat is carried by qsat, not by L."""
    xp = _xp_of(T, z, q)
    return (C.CP_AIR * _F(T, xp) + C.G_ACCEL * _F(z, xp)
            + C.L_VAP * _F(q, xp))


def virtual_temperature(T, q_v, q_c=0.0, *, exact=False):
    """Virtual (density) temperature [K]: the temperature dry air would need to match the moist parcel's
    density at the same pressure. q_v is the vapor and q_c the condensed-water mass fraction [kg/kg].

    TWO FORMS OF THE SAME FUNCTION, selected by ``exact`` (ROADMAP **PHY-20**):

      ``exact=False`` (DEFAULT, the legacy/canonical path)
          ``T_v = T (1 + q_v/eps) / (1 + q_v + q_c)``                                            [K]
          This is the EXACT density temperature when its arguments are MIXING RATIOS
          ``r_i = rho_i/rho_d``. This model's humidities are SPECIFIC (``qsat`` returns
          ``eps e/(p - (1-eps) e)`` = rho_v/rho; the mixing ratio would be ``eps e/(p - e)``), so on
          the default path the right function is evaluated at the wrong variable, which always
          UNDER-states the vapor buoyancy. Kept as the default because it is what every run to date
          -- the canonical world included -- integrated; switching it is a world change.

      ``exact=True`` (ROADMAP PHY-20, default OFF)
          ``T_rho = T (1 + (1/eps - 1) q_v - q_c) = T (1 + 0.608 q_v - q_c)``                    [K]
          The EXACT density temperature for SPECIFIC humidities, straight from the gas law with no
          expansion: only the gas phases exert pressure, so with rho the TOTAL density and
          ``q_d + q_v + q_c = 1``,  ``p/(rho R_d T) = q_d + q_v/eps = 1 + (1/eps - 1) q_v - q_c``.
          It is NOT a dilute approximation of the branch above -- substitute ``r_i = q_i/(1-q_v-q_c)``
          into the ratio form and it collapses onto this line identically.

    SIZE OF THE DIFFERENCE (the two forms agree to O(q_v); they part at O(q_v^2)). At q_c = 0 the gap
    has the closed form

        T_exact - T_ratio = T (1/eps - 1) q_v^2 / (1 + q_v)     [K]  -- strictly positive

    i.e. 1.82 K at (T = 330 K, q_v = 0.10), about 9 % of the whole vapor increment, and 13.60 K at
    q_v = 0.2964. The default path is therefore always the LESS buoyant one.

    The branch is at the PYTHON level (an ``if``, never a blended multiplier): the AD core's step is
    captured into a CUDA graph and a multiplier would change the OFF graph. Array-generic (numpy or
    cupy; a device array stays on device)."""
    xp = _xp_of(T, q_v, q_c)
    T = _F(T, xp); q_v = _F(q_v, xp); q_c = _F(q_c, xp)
    if exact:
        # units: [K] * (dimensionless + [kg/kg]*dimensionless - [kg/kg]) = [K]
        return T * (1.0 + (1.0 / C.EPS_MW - 1.0) * q_v - q_c)
    return T * (1.0 + q_v / C.EPS_MW) / (1.0 + q_v + q_c)


def density_temperature(T, q_v, q_cond, *, exact=False):
    """Density temperature [K] WITH CONDENSATE LOADING: the virtual temperature reduced by the weight of
    the suspended condensed water (cloud droplets + falling rain) the parcel carries,

        T_rho = virtual_temperature(T, q_v, exact=exact) - T * q_cond      [K]

    q_v is the vapor specific humidity and ``q_cond`` the CONDENSED-water mass fraction (cloud q_c +
    rain q_r) [kg/kg]; both are SPECIFIC humidities (fractions of TOTAL mass), and ``q_cond`` is
    expected already clipped at 0.

    THE LOADING TERM IS EXACT, not a linearization. For specific humidities the density temperature
    follows from the gas law in one line, with no expansion anywhere: only the gas phases exert
    pressure, so with rho the TOTAL density (condensate included) and q_d + q_v + q_cond = 1,

        p / (rho R_d T) = q_d + q_v/eps = 1 + q_v (1/eps - 1) - q_cond
        => T_rho = T (1 + 0.608 q_v - q_cond)                      [EXACT in q_v, q_cond]

    so the condensed water enters as precisely ``- T q_cond``. (The ratio form
    ``T (1 + r/eps)/(1 + r)`` is the exact expression for MIXING ratios r_i = rho_i/rho_d; substitute
    r = q/(1-q) and it collapses onto the line above. It is not the specific-humidity formula.)
    This model's q is a specific humidity by its own code: ``qsat`` returns ``eps e/(p - (1-eps) e)``,
    which is rho_v/rho -- the mixing ratio would be ``eps e/(p - e)``.

    THE VAPOR PART IS THE ONE WITH A SWITCH ON IT (``exact``, ROADMAP PHY-20). This function always
    composes the exact loading ``- T q_cond`` onto whatever vapor term ``virtual_temperature``
    returns, so switching the LOADING on always changes the buoyancy by precisely
    ``-g T q_cond / T_ref`` whichever vapor form is selected, and the two corrections stay separable
    and separately reviewable. There is no free parameter in the loading: q_cond is a prognostic mass
    fraction and the coefficient is 1.

      ``exact=False`` (default): a HYBRID -- the ratio-form vapor term (which under-states T_v by
        ~0.07 K at q_v = 0.02 and ~1.8 K at q_v = 0.10, see ``virtual_temperature``) plus the exact
        loading. This is the canonical-world path, kept default so the merge is byte-identical.
      ``exact=True``: the fully exact specific-humidity density temperature
        ``T (1 + 0.608 q_v - q_cond)``, i.e. ``virtual_temperature(..., exact=True)`` with the same
        ``- T q_cond``.

    Array-generic (numpy or cupy; a device array stays on device), like ``virtual_temperature``."""
    xp = _xp_of(T, q_v, q_cond)
    T = _F(T, xp)
    return virtual_temperature(T, q_v, exact=exact) - T * _F(q_cond, xp)


def vapor_pressure_from_q(q, p):
    """Invert qsat's q = eps e/(p-(1-eps)e) for the vapor pressure e [Pa] given specific humidity q and
    pressure p [Pa]:  e = q p / (eps + q(1-eps))."""
    xp = _xp_of(q, p)
    q = _F(q, xp); p = _F(p, xp)
    return q * p / (C.EPS_MW + q * (1.0 - C.EPS_MW))


def phase_relaxation_tau(n_d_cm3, r_bar=10.0e-6, d_vap=2.5e-5):
    """Diffusional phase-relaxation (supersaturation e-folding) timescale of a droplet population,

        tau_phase = 1 / (4 pi D N r_bar)   [s]

    (standard warm-cloud microphysics, e.g. Korolev & Mazin 2003's quasi-steady supersaturation
    relaxation): a supersaturation decays by vapor diffusion onto the existing droplet surface, so the
    rate is set by the droplet NUMBER concentration N and mean radius r_bar. Units audit:
    d_vap [m^2/s] * N [1/m^3] * r_bar [m] = [1/s] -> tau [s]. ``n_d_cm3`` is the droplet/CCN number in
    the model's N_d convention [cm^-3] (x1e6 -> m^-3). Defaults: r_bar = 10 um (a mature warm-cloud
    droplet), d_vap = 2.5e-5 m^2/s (water vapor in air near 290 K, 1 atm). Magnitudes: ~3 s at the
    clean-maritime 100 cm^-3, tens of seconds at ~10 cm^-3 (very clean air), sub-second in salt-loaded
    continental air (>=1000 cm^-3) -- i.e. COMPARABLE to the canonical 1 s thermo substep, which is why
    tau=0 (instantaneous) is the sub-physical fast extreme there, not a mild idealization (#280 /
    ROADMAP GPU-7). This is the physical ANCHOR for choosing/sweeping ``cond_tau`` and the absolute
    scale behind the salt-coupled ``cond_tau_salt`` ratio law (which itself consumes only N_ref/N_d);
    array-generic (a device N_d array stays on device)."""
    xp = get_xp(n_d_cm3)
    n_m3 = _F(n_d_cm3, xp) * 1.0e6                      # [cm^-3] -> [m^-3]
    return 1.0 / (4.0 * np.pi * d_vap * n_m3 * r_bar)   # [s]


def lifting_condensation_level(T_surf, q_surf, p_surf=C.P_SURF):
    """Lifting condensation level of a surface parcel (T_surf [K], specific humidity q_surf, p_surf [Pa]).

    Returns (p_lcl [Pa], T_lcl [K]). Uses Bolton (1980, eq. 22) for the LCL temperature
        T_lcl = 2840 / (3.5 ln T - ln e_hPa - 4.805) + 55   [K]   (e the vapor pressure)
    then the dry-adiabat (Poisson) pressure  p_lcl = p_surf (T_lcl/T_surf)^(cp/R).  Below the LCL the
    parcel rises dry-adiabatically conserving q_surf; at the LCL it just saturates (qsat(T_lcl,p_lcl) ~=
    q_surf), which is what the entraining-plume integrator uses to switch from the dry to the moist branch.
    """
    xp = _xp_of(T_surf, q_surf, p_surf)
    T_surf = _F(T_surf, xp)
    e_hPa = xp.maximum(vapor_pressure_from_q(q_surf, p_surf) / 100.0, 1e-6)   # vapor pressure [hPa]
    T_lcl = 2840.0 / (3.5 * xp.log(T_surf) - xp.log(e_hPa) - 4.805) + 55.0
    T_lcl = xp.minimum(T_lcl, T_surf)                                          # never above the parcel T
    p_lcl = p_surf * (T_lcl / T_surf) ** (C.CP_AIR / C.R_DRY)
    return p_lcl, T_lcl


# --------------------------------------------------------------------------------------
# Bulk surface fluxes (sensible & latent), aerodynamic formulae.
# --------------------------------------------------------------------------------------
def sensible_heat_flux(T_surf, T_air, wind, rho=C.RHO_AIR0, Ch=1.2e-3):
    """Bulk sensible heat flux [W/m^2] from surface to air (positive = surface heats air)."""
    xp = _xp_of(T_surf, T_air, wind)
    return rho * C.CP_AIR * Ch * wind * (_F(T_surf, xp) - _F(T_air, xp))


def latent_heat_flux(T_surf, q_air, p_surf, wind, rho=C.RHO_AIR0, Ce=1.2e-3, wetness=1.0):
    """Bulk latent heat flux (evaporation) [W/m^2]. ``wetness`` in [0,1] scales land dryness
    (1 = open water / saturated surface, <1 = limited soil moisture)."""
    xp = _xp_of(T_surf, q_air, p_surf, wind)
    qs_surf = qsat(T_surf, p_surf)
    return wetness * rho * C.L_VAP * Ce * wind * xp.maximum(qs_surf - _F(q_air, xp), 0.0)


def evaporation_rate_mm_day(LE):
    """Convert a latent heat flux LE [W/m^2] to an evaporation rate [mm/day]."""
    return LE / C.L_VAP * 86400.0          # kg/m^2/s -> mm/day (1 kg/m^2 = 1 mm)
