"""
flatworld.surface -- the diurnally-resolved surface energy balance.

The task is explicit: even though the deliverable is time-averaged, the surface energy balance must
respect the diurnal cycle and the surface heat capacity, because sigma T^4 is nonlinear and the
diurnal swing is large over low-heat-capacity surfaces -- so the time-mean temperature is NOT the
radiative-mean temperature, and getting this wrong misplaces the habitable band.

This module integrates a single surface slab,
        C dTs/dt = SW_abs(t) + LW_down - eps*sigma*Ts^4 - SH(Ts) - LE(Ts),
to a periodic (diurnal) steady state and returns BOTH the time-mean temperature <Ts> and the
energetically-meaningful <sigma Ts^4>, which differ when the swing is large. Used by Stage A (ocean
backbone + a land diagnostic) and per-cell by Stage C.

Key invariant (validated): over a periodic cycle <C dTs/dt> = 0, so
        <sigma Ts^4> = <SW_abs> + LW_down - <SH> - <LE>
is pinned by the MEAN forcing, independent of C -- but <Ts> falls below (<sigma Ts^4>/sigma)^{1/4} by
Jensen's inequality as the swing grows. That gap is the diurnal correction the band placement needs.
"""
from __future__ import annotations
import numpy as np
from . import constants as C
from . import thermo as th
from .thermo import _xp_of


def turbulent_fluxes(Ts, T_air, q_air, p_surf, wind, wetness=1.0,
                     Ch=1.2e-3, Ce=1.2e-3, rho=C.RHO_AIR0):
    """Instantaneous sensible (SH) and latent (LE) heat fluxes [W/m^2] from surface to air.

    Array-module-generic (numpy or CuPy): infers the module from its inputs (CuPy iff any array arg is
    on the device), so a CuPy array runs on the GPU and a numpy caller is byte-identical. ``th.qsat`` is
    likewise device-aware."""
    xp = _xp_of(Ts, T_air, q_air, p_surf, wind)
    SH = rho * C.CP_AIR * Ch * wind * (Ts - T_air)
    qs = th.qsat(Ts, p_surf)
    LE = wetness * rho * C.L_VAP * Ce * wind * xp.maximum(qs - q_air, 0.0)
    return SH, LE


def solve_periodic_surface(t, SW_abs_cycle, LW_down, T_air, q_air, p_surf, wind,
                           C_surf, wetness=1.0, emiss=1.0, Ts_init=None,
                           max_days=400, tol=1e-3, Ch=1.2e-3, Ce=1.2e-3):
    """Integrate the surface slab to a periodic diurnal steady state.

    Parameters
    ----------
    t            : (n+1,) time over ONE day [s], t[0]=0, t[-1]=T_day (uniform).
    SW_abs_cycle : (n+1,) absorbed shortwave at the surface over the day [W/m^2].
    LW_down      : downwelling longwave at the surface [W/m^2] (held constant over the day; the
                   atmosphere is slow relative to a day).
    T_air, q_air : near-surface air temperature [K] / specific humidity [kg/kg] (held over the day).
    wind         : near-surface wind speed [m/s] for the bulk fluxes.
    C_surf       : surface slab heat capacity [J/m^2/K] (ocean mixed layer huge, rock small).
    wetness      : evaporation efficiency in [0,1] (1=open water, <1=dry land).
    emiss        : surface LW emissivity.

    Returns a dict of the converged cycle and its time-means (Ts_mean, sigT4_mean, SH_mean, LE_mean,
    Ts_min, Ts_max, Ts_range, evap_mm_day, n_days_to_converge).

    HOST-ONLY (standalone Stage A backbone / Stage C per-cell, NOT the AD-core hot path): a sequential
    Python-scalar diurnal Euler integration (``Ts`` is a Python float, with ``float(...)`` extracts of
    the per-day mean and the cycle min/max), so it is deliberately numpy-resident -- not a GPU kernel.
    The in-loop SEB the GPU path actually needs is the elementwise update inside ``stage_ad_core``.
    """
    t = np.asarray(t, float)
    n = t.size - 1
    dt = t[1] - t[0]
    SW = np.asarray(SW_abs_cycle, float)
    # initialize near the steady mean so high-C (ocean) surfaces start essentially converged
    if Ts_init is None:
        Ts_init = ((np.mean(SW) + LW_down) / (emiss * C.SIGMA_SB)) ** 0.25
    Ts = float(Ts_init)
    prev_mean = None
    for day in range(max_days):
        acc_T = acc_T4 = acc_SH = acc_LE = 0.0
        for i in range(n):
            SH, LE = turbulent_fluxes(Ts, T_air, q_air, p_surf, wind, wetness, Ch, Ce)
            net = SW[i] + LW_down - emiss * C.SIGMA_SB * Ts**4 - SH - LE
            Ts = Ts + dt * net / C_surf
            acc_T += Ts; acc_T4 += C.SIGMA_SB * Ts**4; acc_SH += SH; acc_LE += LE
        mean_T = acc_T / n
        if prev_mean is not None and abs(mean_T - prev_mean) < tol:
            break
        prev_mean = mean_T
    # one final recorded day for the cycle + means
    Ts_cycle = np.empty(n + 1); Ts_cycle[0] = Ts
    acc_T = acc_T4 = acc_SH = acc_LE = 0.0
    for i in range(n):
        SH, LE = turbulent_fluxes(Ts, T_air, q_air, p_surf, wind, wetness, Ch, Ce)
        net = SW[i] + LW_down - emiss * C.SIGMA_SB * Ts**4 - SH - LE
        Ts = Ts + dt * net / C_surf
        Ts_cycle[i + 1] = Ts
        acc_T += Ts; acc_T4 += C.SIGMA_SB * Ts**4; acc_SH += SH; acc_LE += LE
    LE_mean = acc_LE / n
    # the n-point left-sum mean of SW the Euler loop actually used (NOT SW.mean() over n+1 points,
    # which would double-count the t=0 == t=T_day noon endpoint); the discrete energy identity
    # <sigT4> = <SW_abs> + LW_down - <SH> - <LE> closes exactly against this when the cycle is periodic.
    SW_abs_mean = float(np.mean(SW[:-1]))
    return dict(t=t, Ts_cycle=Ts_cycle, Ts_mean=acc_T / n, sigT4_mean=acc_T4 / n,
                SH_mean=acc_SH / n, LE_mean=LE_mean, SW_abs_mean=SW_abs_mean,
                Ts_min=float(Ts_cycle.min()), Ts_max=float(Ts_cycle.max()),
                Ts_range=float(Ts_cycle.max() - Ts_cycle.min()),
                evap_mm_day=th.evaporation_rate_mm_day(LE_mean),
                n_days_to_converge=day + 1)


def radiative_mean_temperature(SW_abs_mean, LW_down, emiss=1.0):
    """The temperature a surface would hold if it had infinite heat capacity (no swing):
    (<SW_abs> + LW_down)/(eps sigma) ^ 1/4. The diurnal time-mean <Ts> sits at or below this."""
    return ((SW_abs_mean + LW_down) / (emiss * C.SIGMA_SB)) ** 0.25
