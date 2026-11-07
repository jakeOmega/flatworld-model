"""
flatworld.plume_floft -- an offline entraining-plume / single-column estimate of f_loft (OPEN_ISSUES
Brief 2, sharpening c3). The cheaper-than-CRM way to replace Stage C's free convective<->stratiform
split knob ``f_loft`` with a PHYSICALLY-ANCHORED, bracketed range.

What f_loft means (the exact definition wired into Stage C).
    f_loft = fraction of the convective moisture flux that DETRAINS ALOFT (becomes the traveling
             stratiform supply U) rather than PRECIPITATING in the convective column (P_conv)
           = 1 - (convective precipitation efficiency).
Stage C drains the surface layer at rate W_s/tau_conv; a fraction f_loft of that drainage is detrained
to the aloft layer (to ride poleward) and (1-f_loft) rains in place. This module estimates that split
from a rising entraining plume in a Stage-A environment.

The model (a standard bulk entraining plume + a precipitation-efficiency microphysics closure).
  * Environment: a single Stage-A column (its OWN RCE atmosphere; run kappa_atm=0 for the local limit),
    read as T_env(p), q_env(p), z(p). Stage A has already convectively adjusted the column, so the
    environment sits on (near) a moist adiabat, which carries ~no surface-parcel CAPE -- so the updraft is
    started SATURATED at the surface (at the near-surface AIR temperature, not the skin SST -- see
    integrate_plume), and its buoyancy is the moisture excess of a saturated core in the SUB-saturated
    (RH~0.2-0.8) free troposphere.
  * Parcel: a saturated updraft lifted in pressure on the (entraining) moist adiabat, condensing the
    vapour the cooling sheds. Each condensation increment dcond: a fraction PE rains out LOCALLY
    (-> P_conv), the rest (1-PE) is carried as condensate q_c.
  * Entrainment: over each height step the parcel mixes in a fraction eps*dz of environmental air
    (drier, eventually cooler), conserving liquid-water static energy and total water, with a
    saturation adjustment. Entraining dry air evaporates condensate and erodes buoyancy -- this is what
    sets the convective TOP (the level of neutral buoyancy, the detrainment level z_det).
  * At z_det the plume detrains its surviving vapour + condensate aloft. With mass flux M(z) growing as
    exp(int eps dz):
        f_loft = D_F / (D_F + R_F),  D_F = M(z_det)*(q_v+q_c)(z_det),  R_F = int M*PE*dcond  (rain flux)
    -- so in the no-entrainment limit f_loft -> 1-PE, and entrainment/finite convective top are the
    corrections. The model RELOCATES the assumption to (eps, PE): bounded, with the detrainment LEVEL
    tied to buoyancy (and used to adjudicate the unreconciled convective-top heights across stages).

Honesty / lock discipline (OPEN_ISSUES Brief 2; the lam_latent lesson in MODELING_LESSONS.md).
  (eps, PE) are picked on the PLUME'S OWN physics -- the eps->0 moist-adiabat validation, the
  detrainment-height match to a physical convective top, and literature/physical PE bounds -- NOT by
  tuning the downstream cascade rain to 0.5 or to the doc's 25-50. ``EPS_FROZEN``/``PE_FROZEN`` below are
  FROZEN and LOCKED: f_loft must not become a covert band-magnitude knob. The full (eps, PE) sweep is
  reported as the bracketed range; the frozen pair is the single curve wired into the cascade.

Run (from cosmology_models/):
    ../.venv/bin/python -m flatworld.plume_floft            # table + sweep + figure (test-res Stage A)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from . import constants as C
from . import thermo as th
from .stage_a_column import StageA, StageAConfig


# ======================================================================================
# FROZEN, LOCKED (eps, PE) -- set on the plume's own physics, NOT fitted to the cascade rain.
# See pick_frozen_eps() (detrainment-height match) and the module docstring. Do NOT retune these to
# move the ribbon-sea/inner-sea rain: that is exactly the covert-knob failure the lock prevents.
# ======================================================================================
EPS_FROZEN = 0.05e-3        # fractional entrainment rate [1/m] = 0.05 /km. PICKED on the detrainment-
                            # height match (pick_frozen_eps / __main__): in this near-neutral, convectively
                            # adjusted environment the buoyancy-defined convective top DROPS steeply with
                            # eps (mean z_det ~5.4 km at 0.05/km, ~2 km at 0.10, collapsing to cloud base
                            # by 0.2/km), so a PHYSICAL deep-convective top (the weakly-entraining cores
                            # that actually reach the upper troposphere) requires the LOW end -- 0.05/km
                            # lands the top at ~5 km, between Stage C's 4 km aloft level and Stage B's
                            # hard-coded 12 km (and adjudicates the aloft level toward the lower end). NOT
                            # fitted to the cascade rain. LOCKED.
PE_FROZEN = 0.55            # convective precipitation efficiency [-] (mid of the literature 0.3-0.8
                            # bracket; deep maritime convection is reasonably efficient). LOCKED.
EPS_SWEEP = (0.05e-3, 0.10e-3, 0.20e-3, 0.40e-3, 0.80e-3)   # 0.05-0.8 /km
PE_SWEEP = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
WET_SST_LO, WET_SST_HI = 40.0, 55.0     # the wetness=1 wet-ocean band [deg C] the brief anchors over
# columns SAMPLED for the f_loft(SST) curve: the CONVECTING wet ocean (SST > Stage C's conv_sst_C ~30,
# wetness=1). The brief's headline 40-55 band is a thin strip near the salt-flat edge in this model (the
# cross-track SST gradient is steep there), so we sample the broader convecting ocean and report the
# 40-55 anchor within it.
SAMPLE_SST_LO, SAMPLE_SST_HI = 28.0, 55.0


@dataclass
class PlumeResult:
    """One entraining-plume ascent in a fixed environment."""
    p: np.ndarray; z: np.ndarray                      # parcel pressure [Pa] / height above surface [m]
    T_p: np.ndarray; q_v: np.ndarray; q_c: np.ndarray  # parcel T [K], vapour, condensate [kg/kg]
    T_env: np.ndarray; q_env: np.ndarray              # environment at the parcel's pressure
    B: np.ndarray; M: np.ndarray; cumR: np.ndarray    # buoyancy [m/s^2], mass flux [-], cum rain flux
    lcl_idx: int; det_idx: int                        # updraft-base (=0) and detrainment-level indices
    cape: float; cin: float                           # [J/kg]
    eps: float; PE: float; sst_C: float
    f_loft: float                                     # the headline split
    z_det_km: float                                   # detrainment height above surface [km]


def _sat_adjust(s_l, q_t, p):
    """Saturation adjustment at pressure p [Pa]: given liquid-water static energy s_l = cp*T - L*q_c
    [J/kg] (geopotential dropped -- it cancels in same-level mixing) and total water q_t = q_v+q_c,
    return (T, q_v, q_c). Try the unsaturated branch (q_c=0 -> T=s_l/cp); if that is supersaturated,
    Newton-solve cp*T = s_l + L*(q_t - qsat(T,p))."""
    T_unsat = s_l / C.CP_AIR
    if q_t <= th.qsat(T_unsat, p):
        return T_unsat, q_t, 0.0
    T = T_unsat
    for _ in range(40):
        qs = th.qsat(T, p); dqs = th.dqsat_dT(T, p)
        f = C.CP_AIR * T - s_l - C.L_VAP * (q_t - qs)
        fp = C.CP_AIR + C.L_VAP * dqs
        step = f / fp
        T -= step
        if abs(step) < 1e-8:
            break
    qs = th.qsat(T, p)
    q_c = max(q_t - qs, 0.0)
    return T, q_t - q_c, q_c


def _env_height_of_p(pe, Te, qe):
    """Geometric height z(p) of the environment [m], hydrostatic from the surface up. Returns an
    interpolation lambda. (The parcel sits at the environment's pressure, so the geometric height of a
    pressure level is the ENVIRONMENT's hydrostatic property -- this is what sets dz for the entrainment
    fraction eps*dz and the detrainment-height report, NOT the warm parcel's own thickness.)"""
    o = np.argsort(-pe)                                           # descending pressure (surface first)
    pd = pe[o]; Td = Te[o]; qd = qe[o]
    Tv = th.virtual_temperature(Td, qd, 0.0)
    z = np.zeros_like(pd)
    for k in range(1, pd.size):
        Tvbar = 0.5 * (Tv[k] + Tv[k - 1])
        z[k] = z[k - 1] + (C.R_DRY * Tvbar / C.G_ACCEL) * (np.log(pd[k - 1]) - np.log(pd[k]))
    # extrapolate to the true surface pressure if the lowest level is above it
    zsurf = z[0] - (C.R_DRY * Tv[0] / C.G_ACCEL) * (np.log(C.P_SURF) - np.log(pd[0]))
    pp = np.concatenate([[C.P_SURF], pd]); zz = np.concatenate([[zsurf], z])
    o2 = np.argsort(pp)
    return lambda p: np.interp(p, pp[o2], zz[o2])


def integrate_plume(p_env, T_env_p, q_env_p, sst_K, eps, PE, *, start_saturated=False,
                    n_sub=600, p_top=None):
    """Integrate a saturated entraining updraft upward in pressure through a fixed environment.

    f_loft mode (the default): the updraft starts SATURATED at the SURFACE, at the environment's
    near-surface AIR temperature, and rises on the (entraining) moist adiabat. This base is chosen
    because Stage A has convectively adjusted the column to (near) the moist adiabat, so a column-T
    surface parcel carries ~no CAPE; the updraft's buoyancy comes from being SATURATED in a SUB-saturated
    free troposphere (env RH 0.2-0.8), and entrainment of that drier air is what erodes the buoyancy and
    sets the convective top (see the parcel-init comment below for why the surface AIR temperature, not
    the skin SST, is the right base). ``sst_K`` only labels the result by SST.

    Validation mode (``start_saturated``): start the saturated parcel at the SURFACE at ``sst_K``, for the
    eps->0 reproduction of ``thermo.temperature_on_moist_adiabat``.

    Parameters: ``p_env, T_env_p, q_env_p`` environment [Pa],[K],[kg/kg] (any order); ``eps`` fractional
    entrainment [1/m]; ``PE`` convective precipitation efficiency (local-rain fraction of each
    condensation increment).
    """
    order = np.argsort(p_env)
    pe = np.asarray(p_env, float)[order]; Te = np.asarray(T_env_p, float)[order]; qe = np.asarray(q_env_p, float)[order]
    p_top = float(pe.min()) if p_top is None else p_top
    Tenv = lambda p: np.interp(p, pe, Te)
    qenv = lambda p: np.interp(p, pe, qe)
    zenv = _env_height_of_p(pe, Te, qe)

    if start_saturated:                                           # validation: saturated, from sst_K
        p_base = C.P_SURF; T_p = float(sst_K); q_v = th.qsat(T_p, p_base)
    else:
        # f_loft: a SATURATED surface updraft at the environment's near-surface AIR temperature. Two
        # reasons this is the right base (both learned the hard way): (1) Stage A has convectively
        # adjusted the column to ~the moist adiabat, so it carries no surface-parcel CAPE -- the updraft's
        # buoyancy is the moisture excess of a SATURATED core over the SUB-saturated (RH~0.2-0.8) free
        # troposphere; (2) over the warm salt-flat-edge strip the skin SST is actually a touch COOLER than
        # the overlying air (the ocean is evaporatively capped while the air is heated laterally from the
        # hot salt flat), so a skin-SST parcel is spuriously negatively buoyant -- using the surface AIR
        # temperature, saturated, isolates the moisture-driven buoyancy that actually drives the updraft.
        p_base = C.P_SURF; T_p = float(Tenv(C.P_SURF)); q_v = th.qsat(T_p, p_base)
    q_c = 0.0; M = 1.0; cumR = 0.0
    lnp = np.linspace(np.log(p_base), np.log(p_top), n_sub + 1)
    p_grid = np.exp(lnp)
    nrec = n_sub + 1
    P = np.empty(nrec); Z = np.empty(nrec); TP = np.empty(nrec); QV = np.empty(nrec); QC = np.empty(nrec)
    TE = np.empty(nrec); QE = np.empty(nrec); BB = np.empty(nrec); MM = np.empty(nrec); RR = np.empty(nrec)

    def _record(i, p):
        P[i] = p; Z[i] = float(zenv(p)); TP[i] = T_p; QV[i] = q_v; QC[i] = q_c
        Te_i = float(Tenv(p)); qe_i = float(qenv(p)); TE[i] = Te_i; QE[i] = qe_i; MM[i] = M; RR[i] = cumR
        Tv_p = th.virtual_temperature(T_p, q_v, q_c); Tv_e = th.virtual_temperature(Te_i, qe_i, 0.0)
        BB[i] = C.G_ACCEL * (Tv_p - Tv_e) / Tv_e

    _record(0, p_grid[0])
    for i in range(n_sub):
        p = p_grid[i]; p_next = p_grid[i + 1]; dlnp = lnp[i] - lnp[i + 1]   # > 0 going up
        dz = max(float(zenv(p_next) - zenv(p)), 0.0)              # env geometric thickness of the layer
        # 1. adiabatic lift in pressure
        if q_v >= 0.9999 * th.qsat(T_p, p):                       # saturated -> moist adiabat (pressure form)
            qs = th.qsat(T_p, p)
            num = 1.0 + C.L_VAP * qs / (C.R_DRY * T_p)
            den = 1.0 + C.L_VAP**2 * qs * C.EPS_MW / (C.CP_AIR * C.R_DRY * T_p**2)
            dlnT_dlnp = (C.R_DRY / C.CP_AIR) * num / den
            T_next = T_p * np.exp(-dlnT_dlnp * dlnp)
        else:                                                     # unsaturated -> dry adiabat
            T_next = T_p * (p_next / p) ** C.KAPPA_POISSON
        # 2. condensation of the vapour the cooling sheds; rain PE locally, carry (1-PE)
        qsat_next = th.qsat(T_next, p_next)
        if q_v > qsat_next:
            dcond = q_v - qsat_next
            q_v = qsat_next
            cumR += M * PE * dcond
            q_c += (1.0 - PE) * dcond
        T_p = T_next
        # 3. entrainment of environmental air over dz (mix s_l & q_t; saturation-adjust)
        if eps > 0.0:
            delta = eps * dz
            Te_n = float(Tenv(p_next)); qe_n = float(qenv(p_next))
            q_t = (q_v + q_c + delta * qe_n) / (1.0 + delta)
            s_l = (C.CP_AIR * T_p - C.L_VAP * q_c + delta * C.CP_AIR * Te_n) / (1.0 + delta)
            T_p, q_v, q_c = _sat_adjust(s_l, q_t, p_next)
            M *= (1.0 + delta)
        _record(i + 1, p_next)

    # ---- post-process: detrainment level (top of the buoyant updraft), CAPE/CIN, f_loft ----
    lcl_idx = 0                                                   # saturated updraft starts at the surface
    buoyant = BB > 0.0
    det_idx = int(np.max(np.where(buoyant)[0])) if buoyant.any() else 0
    dZ = np.gradient(Z)
    cape = float(np.sum(np.clip(BB, 0.0, None) * dZ))
    cin = 0.0                                                     # sub-cloud CIN not modelled (cloud-base start)
    D_F = MM[det_idx] * (QV[det_idx] + QC[det_idx])
    R_F = RR[det_idx]
    f_loft = float(D_F / (D_F + R_F)) if (D_F + R_F) > 0 else float("nan")
    return PlumeResult(p=P, z=Z, T_p=TP, q_v=QV, q_c=QC, T_env=TE, q_env=QE, B=BB, M=MM, cumR=RR,
                       lcl_idx=lcl_idx, det_idx=det_idx, cape=cape, cin=cin, eps=eps, PE=PE,
                       sst_C=float(sst_K - C.T_FREEZE), f_loft=f_loft, z_det_km=float(Z[det_idx] / 1e3))


# ======================================================================================
# Environments from Stage A (the local-RCE columns), and the f_loft(SST) tables
# ======================================================================================
def stage_a_environments(stage_a: StageA, sst_lo=SAMPLE_SST_LO, sst_hi=SAMPLE_SST_HI):
    """Per-column environments (p_mid, T_atm, q, SST) from a converged Stage A, for the convecting
    wet-ocean columns (SST in [sst_lo, sst_hi]). The plume's environment is the column's OWN profile.

    NOTE (a finding -- see the docstring of ``environment_stage_a``): the brief proposed kappa_atm=0 for
    the strictly-local limit, but in THIS model the wet-ocean columns run away to an unphysical hot,
    supersaturated state without horizontal transport (the Komabayashi-Ingersoll moist runaway the
    wetness ramp only caps above 55 C). So the environment must be the TRANSPORT-ON column -- which is
    also the realistic sounding the convection actually sees."""
    q = stage_a._q()                                              # (nx, nz) absolute humidity
    p_mid = stage_a.p_mid                                         # (nz,)
    sst_C = stage_a.T_surf - C.T_FREEZE
    envs = []
    for i in range(stage_a.cfg.nx):
        if sst_lo <= sst_C[i] <= sst_hi:
            envs.append(dict(i=i, sst_K=float(stage_a.T_surf[i]), theta_deg=float(stage_a.theta_deg[i]),
                             p=p_mid.copy(), T=stage_a.T_atm[i].copy(), q=q[i].copy(), sst_C=float(sst_C[i])))
    return envs


def ad_core_environments(core, sst_lo=SAMPLE_SST_LO, sst_hi=SAMPLE_SST_HI):
    """Per-column plume environments (p_mid, T_atm, q, SST) from a converged AD core -- the fused-core
    analog of ``stage_a_environments``. The AD core carries the SAME pressure-grid fields (``T_atm`` (nx,nz),
    specific humidity ``q`` (nx,nz), ``p_mid`` (nz), ``T_surf`` (nx)) with the SAME TOP->SURFACE ordering, and
    ``integrate_plume`` sorts its environment by pressure internally, so the columns drop in unchanged. Using
    the core means the convective/stratiform split's plume environment rides the RESOLVED-circulation sounding
    instead of the transport-poor Stage A column (whose greenhouse-collapse cliff otherwise leaks into the
    rain split -- the same defect fixed for the land-T backbone)."""
    q = np.clip(core.q, 0.0, th.Q_MAX)
    p_mid = core.p_mid
    sst_C = core.T_surf - C.T_FREEZE
    envs = []
    for i in range(core.cfg.nx):
        if sst_lo <= sst_C[i] <= sst_hi:
            envs.append(dict(i=i, sst_K=float(core.T_surf[i]), theta_deg=float(core.theta_deg[i]),
                             p=p_mid.copy(), T=core.T_atm[i].copy(), q=q[i].copy(), sst_C=float(sst_C[i])))
    return envs


def environment_stage_a(profile="test", nx=200):
    """A converged Stage A to read plume environments from. Uses the model's DEFAULT (transport-on)
    diffusivity, NOT kappa_atm=0: the strictly-local kappa=0 limit the brief suggested makes the
    wet-ocean columns blow up into a moist runaway here (q -> ~0.4, a 77 C mid-column) because those
    columns are stabilized by horizontal transport -- so the transport-on column is the physical (and
    only usable) environment. ``nx`` is bumped above the test default so the steep wet-ocean SST band is
    sampled by several columns (the SST gradient is sharp at the salt-flat edge)."""
    from . import config as cfgmod
    cfg = cfgmod.stage_a(profile, nx=nx)
    A = StageA(cfg); A.run(verbose=False)
    return A


def floft_sst_table(stage_a: StageA, eps, PE, *, envs=None, **kw):
    """f_loft for every wet-ocean column at one (eps, PE). Returns (sst_C[], f_loft[], z_det_km[], cape[])
    sorted by SST. ``envs`` (optional) supplies pre-extracted environments -- e.g. from
    ``ad_core_environments(core)`` so the plume is sampled from the fused AD core; default reads
    ``stage_a_environments(stage_a)`` (the legacy Stage A column path, byte-identical)."""
    if envs is None:
        envs = stage_a_environments(stage_a)
    sst, fl, zd, cape = [], [], [], []
    for e in envs:
        r = integrate_plume(e["p"], e["T"], e["q"], e["sst_K"], eps, PE, **kw)
        sst.append(e["sst_C"]); fl.append(r.f_loft); zd.append(r.z_det_km); cape.append(r.cape)
    o = np.argsort(sst)
    return (np.array(sst)[o], np.array(fl)[o], np.array(zd)[o], np.array(cape)[o])


def floft_band(stage_a: StageA, eps_sweep=EPS_SWEEP, pe_sweep=PE_SWEEP, **kw):
    """The f_loft(SST) BAND over the full (eps, PE) sweep, plus the frozen-pair curve. Returns a dict
    with a common SST grid and the min/max/frozen f_loft on it."""
    s0, _, _, _ = floft_sst_table(stage_a, EPS_FROZEN, PE_FROZEN, **kw)
    sst_grid = np.linspace(float(np.min(s0)), float(np.max(s0)), 16)
    curves = []
    for eps in eps_sweep:
        for PE in pe_sweep:
            s, f, _, _ = floft_sst_table(stage_a, eps, PE, **kw)
            good = np.isfinite(f)
            if good.sum() >= 2:
                curves.append(np.interp(sst_grid, s[good], f[good]))
    curves = np.array(curves)
    s, ff, _, _ = floft_sst_table(stage_a, EPS_FROZEN, PE_FROZEN, **kw)
    good = np.isfinite(ff)
    frozen = np.interp(sst_grid, s[good], ff[good]) if good.sum() >= 2 else np.full_like(sst_grid, np.nan)
    return dict(sst_grid=sst_grid, f_min=curves.min(axis=0), f_max=curves.max(axis=0),
                f_mean=curves.mean(axis=0), frozen=frozen)


def floft_of_theta_from_table(sst_C, f_loft, sst_of_theta, clip=(0.05, 0.95)):
    """Assemble the ``f_loft_of_theta`` callable from a tabulated f_loft(SST): maps theta -> SST (the
    cascade's SST backbone) -> f_loft, holding the nearest wet-band value beyond the table ends (the
    split is only meaningful where there is a wet convective source) and clipping to ``clip`` for
    numerical safety. Split out from make_floft_of_theta so it is unit-testable without a Stage A run."""
    s = np.asarray(sst_C, float); f = np.asarray(f_loft, float)
    good = np.isfinite(f)
    s = s[good]; f = f[good]
    if s.size < 2:
        fill = float(np.nanmean(f)) if good.any() else 0.5
        return lambda th: np.clip(np.full_like(np.asarray(th, float), fill), *clip)
    o = np.argsort(s); s = s[o]; f = f[o]
    f_of_sst = lambda sstc: np.interp(np.asarray(sstc, float), s, f)                 # flat beyond the ends

    def f_loft_of_theta(theta_deg):
        return np.clip(f_of_sst(np.asarray(sst_of_theta(theta_deg), float)), clip[0], clip[1])
    return f_loft_of_theta


def make_floft_of_theta(stage_a: StageA, sst_of_theta, eps=EPS_FROZEN, PE=PE_FROZEN,
                        clip=(0.05, 0.95), **kw):
    """Build the ``f_loft_of_theta`` callable Stage C consumes from a converged Stage A: the frozen-(eps,
    PE) plume table over the convecting wet-ocean columns, assembled by floft_of_theta_from_table."""
    s, f, _, _ = floft_sst_table(stage_a, eps, PE, **kw)
    return floft_of_theta_from_table(s, f, sst_of_theta, clip=clip)


def cape_of_theta_from_table(sst_C, cape, sst_of_theta):
    """Assemble a ``cape_of_theta`` callable from a tabulated entraining-plume CAPE(SST) [J/kg] -- the
    convective-vigor sibling of ``floft_of_theta_from_table``, but UNCLIPPED (CAPE is J/kg, not a 0-1
    fraction; only floored at 0). theta -> SST (the cascade's SST backbone) -> CAPE, holding the nearest
    wet-band value beyond the table ends."""
    return floft_of_theta_from_table(sst_C, cape, sst_of_theta, clip=(0.0, np.inf))


def make_cape_of_theta(stage_a: StageA, sst_of_theta, eps=EPS_FROZEN, PE=PE_FROZEN, **kw):
    """Build a ``cape_of_theta`` callable (entraining-plume CAPE [J/kg] vs theta via the SST backbone)
    from a converged Stage A -- the convective-vigor sibling of ``make_floft_of_theta``. The cascade
    evaluates it on the Stage C grid to output a per-cell CAPE map (``dC['cape']``), the source for the
    convective cold-pool gust bump in ``surface_wind.attach_gust``. Standalone convenience: it
    re-integrates the wet-ocean plume table (no extra Stage A run, but a second integration vs
    make_floft_of_theta). The cascade itself shares ONE ``floft_sst_table`` call across both the
    f_loft and CAPE assemblers (see run_cascade) and does not call this."""
    s, _, _, cape = floft_sst_table(stage_a, eps, PE, **kw)
    return cape_of_theta_from_table(s, cape, sst_of_theta)


def cape_map_from_theta(cape_of_theta, theta_deg, map_shape):
    """Per-cell CAPE map [J/kg] for ``dC['cape']``: evaluate a ``cape_of_theta`` callable on the
    per-row theta and broadcast across the along-track axis to ``map_shape`` (the (nx, ne) Stage C
    map). A per-row theta vector against a 2-D map broadcasts down the rows; anything else (already
    2-D, or a non-2-D map) is returned as-is. Split out of run_cascade so the broadcast is unit-tested."""
    cape_row = np.asarray(cape_of_theta(np.asarray(theta_deg, float)), float)
    if cape_row.ndim == 1 and len(map_shape) == 2:
        cape_row = np.broadcast_to(cape_row[:, None], map_shape).copy()
    return cape_row


# ======================================================================================
# Symbolic vs numeric cross-check (the L*q coefficient-bug class; OPERATIONS.md item 5)
# ======================================================================================
def symbolic_crosscheck(rtol=1e-10):
    """Cross-check the buoyancy / MSE / entrainment-mixing algebra symbolically (sympy) against the
    numeric helpers, on random inputs -- the exact ``L*q``-coefficient-bug guard OPERATIONS.md flags.
    Returns a dict of max relative errors (all ~ machine epsilon). Raises AssertionError on mismatch."""
    import sympy as sp
    T, qv, qc, z, p, e_pa = sp.symbols("T q_v q_c z p e", positive=True)
    eps = sp.Rational(C.EPS_MW).limit_denominator(10**9)
    cp, g, Lv, Rd = (sp.Float(C.CP_AIR), sp.Float(C.G_ACCEL), sp.Float(C.L_VAP), sp.Float(C.R_DRY))

    # 1. virtual temperature  T_v = T (1 + q_v/eps)/(1 + q_v + q_c)
    Tv_sym = sp.lambdify((T, qv, qc), T * (1 + qv / sp.Float(C.EPS_MW)) / (1 + qv + qc), "numpy")
    # 1b. density temperature with CONDENSATE LOADING (PHY-19): T_rho = T_v(T, q_v) - T q_cond, the
    #     shipped composition. The loading term is EXACT for specific humidities (see below).
    Trho_sym = sp.lambdify((T, qv, qc),
                           T * (1 + qv / sp.Float(C.EPS_MW)) / (1 + qv) - T * qc, "numpy")
    # 1c. the EXACT specific-humidity density temperature (PHY-20, cfg.virtual_temp_exact):
    #     T_rho = T (1 + (1/eps - 1) q_v - q_c), from the gas law with no expansion.
    Tv_exact_sym = sp.lambdify((T, qv, qc),
                               T * (1 + (1 / sp.Float(C.EPS_MW) - 1) * qv - qc), "numpy")
    # 2. MSE  h = cp T + g z + L q   (the L coefficient)
    mse_sym = sp.lambdify((T, z, qv), cp * T + g * z + Lv * qv, "numpy")
    # 3. vapour pressure inverse  e = q p/(eps + q(1-eps))
    e_sym = sp.lambdify((qv, p), qv * p / (sp.Float(C.EPS_MW) + qv * (1 - sp.Float(C.EPS_MW))), "numpy")
    # 4. unsaturated entrainment-mixing temperature update (q_c=0): T' = (cp T + delta cp T_env)/((1+delta) cp)
    Te, delta = sp.symbols("T_env delta", positive=True)
    Tmix_sym = sp.lambdify((T, Te, delta), (cp * T + delta * cp * Te) / ((1 + delta) * cp), "numpy")

    rng = np.random.default_rng(0)
    Ts = rng.uniform(250, 320, 200); qvs = rng.uniform(1e-4, 3e-2, 200); qcs = rng.uniform(0, 5e-3, 200)
    zs = rng.uniform(0, 1.5e4, 200); ps = rng.uniform(2e4, 1e5, 200); Tes = rng.uniform(240, 310, 200)
    ds = rng.uniform(1e-4, 5e-2, 200)

    def relerr(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-30)))

    out = {}
    out["virtual_temperature"] = relerr(th.virtual_temperature(Ts, qvs, qcs), Tv_sym(Ts, qvs, qcs))
    out["density_temperature"] = relerr(th.density_temperature(Ts, qvs, qcs), Trho_sym(Ts, qvs, qcs))
    out["virtual_temperature_exact"] = relerr(th.virtual_temperature(Ts, qvs, qcs, exact=True),
                                              Tv_exact_sym(Ts, qvs, qcs))
    out["density_temperature_exact"] = relerr(th.density_temperature(Ts, qvs, qcs, exact=True),
                                              Tv_exact_sym(Ts, qvs, qcs))
    out["mse"] = relerr(th.mse(Ts, ps, zs, qvs), mse_sym(Ts, zs, qvs))
    out["vapor_pressure_from_q"] = relerr(th.vapor_pressure_from_q(qvs, ps), e_sym(qvs, ps))
    # numeric unsaturated mixing (mirror _sat_adjust's unsaturated branch) vs symbolic
    s_l = (cp_n := C.CP_AIR) * Ts + ds * cp_n * Tes   # = (1+delta)*cp*T'
    Tmix_num = (s_l) / ((1 + ds) * cp_n)
    out["entrain_mix_T"] = relerr(Tmix_num, Tmix_sym(Ts, Tes, ds))
    for k, v in out.items():
        assert v < rtol, f"symbolic-numeric mismatch in {k}: relerr={v:.2e}"
    return out


# ======================================================================================
# Report + figure (__main__)
# ======================================================================================
def pick_frozen_eps(stage_a, pe=0.5):
    """Report the entrainment-rate <-> detrainment-height relationship the frozen eps is picked on:
    for each eps in EPS_SWEEP, the mean detrainment height over the wet-ocean columns. ``EPS_FROZEN`` is
    the one whose mean convective top lands at ~5 km (its comment justifies ~5.4 km) -- a physical
    deep-convective top that ADJUDICATES Stage B's hard-coded 12 km vs Stage C's 4 km aloft level toward
    the lower end. (The former ``target_top_km=(12,15)`` parameter was dead AND inconsistent with that
    ~5 km pick, so it is removed; issue #88 item 4.)"""
    rows = []
    for eps in EPS_SWEEP:
        _, _, zd, _ = floft_sst_table(stage_a, eps, pe)
        rows.append((eps, float(np.nanmean(zd))))
    return rows


def main(profile="test"):
    import os
    nx = 128 if profile == "test" else 256
    print(f"=== Stage A (transport-on; nx={nx}) for plume environments ===")
    A = environment_stage_a(profile, nx=nx)
    envs = stage_a_environments(A)
    print(f"convecting wet-ocean columns (SST {SAMPLE_SST_LO:.0f}-{SAMPLE_SST_HI:.0f} C): {len(envs)} "
          f"(theta {min(e['theta_deg'] for e in envs):.1f}-{max(e['theta_deg'] for e in envs):.1f}, "
          f"SST {min(e['sst_C'] for e in envs):.1f}-{max(e['sst_C'] for e in envs):.1f})")

    print("\n=== symbolic vs numeric cross-check (L*q-coefficient guard) ===")
    for k, v in symbolic_crosscheck().items():
        print(f"  {k:24s} max relerr {v:.2e}")

    print("\n=== eps <-> detrainment-height (pick the frozen eps; adjudicates 4 km vs 12 km) ===")
    print(f"  {'eps [1/km]':>11} {'mean z_det [km]':>15}")
    for eps, zd in pick_frozen_eps(A):
        mark = "  <- FROZEN" if abs(eps - EPS_FROZEN) < 1e-9 else ""
        print(f"  {eps*1e3:>11.2f} {zd:>15.1f}{mark}")

    print("\n=== f_loft band over the (eps, PE) sweep, with the frozen pair ===")
    band = floft_band(A)
    print(f"  {'SST[C]':>7} {'f_min':>7} {'f_max':>7} {'f_mean':>7} {'frozen':>7}")
    for j in range(0, len(band["sst_grid"]), 2):
        print(f"  {band['sst_grid'][j]:>7.1f} {band['f_min'][j]:>7.2f} {band['f_max'][j]:>7.2f} "
              f"{band['f_mean'][j]:>7.2f} {band['frozen'][j]:>7.2f}")
    print(f"\n  FROZEN (eps={EPS_FROZEN*1e3:.2f}/km, PE={PE_FROZEN}): "
          f"f_loft over the wet band = {np.nanmin(band['frozen']):.2f}-{np.nanmax(band['frozen']):.2f} "
          f"(default knob was 0.5)")

    # figure: f_loft(SST) band + frozen curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax.fill_between(band["sst_grid"], band["f_min"], band["f_max"], alpha=0.25, color="C0",
                    label="(eps, PE) sweep band")
    ax.plot(band["sst_grid"], band["f_mean"], "C0--", lw=1, label="sweep mean")
    ax.plot(band["sst_grid"], band["frozen"], "C3-", lw=2,
            label=f"frozen eps={EPS_FROZEN*1e3:.2f}/km, PE={PE_FROZEN}")
    ax.axhline(0.5, color="0.5", ls=":", lw=1, label="old free default 0.5")
    ax.set_xlabel("wet-ocean SST [C]"); ax.set_ylabel("f_loft  (= 1 - conv. precip. efficiency)")
    ax.set_title("Entraining-plume f_loft(SST)"); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    # detrainment height vs eps
    for eps in EPS_SWEEP:
        s, _, zd, _ = floft_sst_table(A, eps, 0.5)
        ax2.plot(s, zd, marker=".", label=f"eps={eps*1e3:.2f}/km")
    ax2.axhspan(4, 6, color="0.85", label="frozen conv. top ~5 km (EPS_FROZEN)")
    ax2.axhline(4.0, color="C1", ls=":", label="Stage C aloft 4 km")
    ax2.set_xlabel("wet-ocean SST [C]"); ax2.set_ylabel("detrainment height [km]")
    ax2.set_title("Convective top vs entrainment"); ax2.legend(fontsize=7)
    fig.tight_layout()
    out = "plume_floft.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved figure -> {os.path.abspath(out)}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
