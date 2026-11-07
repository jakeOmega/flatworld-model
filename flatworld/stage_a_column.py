"""
flatworld.stage_a_column -- Stage A: the cross-track radiative-convective-diffusive backbone.

A chain of single-column radiative-convective-equilibrium (RCE) atmospheres at cross-track distances
x (uniform in PHYSICAL DISTANCE, never in theta), coupled by a diffusive heat transport that stands in
for the mean overturning. Each column carries an atmosphere (nz pressure layers, fixed-RH water vapor
for the greenhouse feedback) over an ocean mixed layer (the aquaplanet backbone). Driven by the
DERIVED day-mean flux S(x) (geometry.day_mean_flux), real radiation (radiation.column_radiation), and
a moist convective adjustment. It PREDICTS the absolute surface temperature T(x) from flux + albedo +
greenhouse + transport -- no 5 C anchor.

Role in the cascade (see DESIGN.md): the cheap backbone. It initializes Stage B and runs the
I0/albedo/CO2 sensitivity sweeps; its transport diffusivity is meant to be CALIBRATED to B's explicit
transport (B is authoritative for the zonal mean). Over the ocean the diurnal swing is negligible, so
Stage A uses day-mean forcing; the diurnal-cycle / heat-capacity effect that matters over land is
carried by flatworld.surface and applied in Stage C.

Boundaries (Q5): equator x=0 is a symmetry edge (zero flux); the cold edge x_max is a CLOSED wall
(no flux through it) backed by a RADIATING SPONGE -- over the poleward-most band the columns are
relaxed toward their LOCAL radiative equilibrium (toa_net -> 0), so the poleward heat transport is
radiated to space there (physically, by the unmodeled colder tail) rather than reflected back. This
replaces the old grid-dependent F[-1]=F[-2] outflow, which let a box-position-dependent amount of heat
leave and contaminated the ribbon SST by +-3 C (it did NOT cleanly box-converge -- OPEN_ISSUES Issue
B). Energy closure counts the sponge's net heat removal as the export to the unmodeled tail:
int(absorbed SW - OLR) dx == -int(sponge power) dx. Box-size independence is tested by moving x_max
(theta_max): the sponge always sits at the cold edge, so the ribbon decouples from where the wall is.
"""
from __future__ import annotations
import numpy as np
import scipy.linalg as sla
from dataclasses import dataclass, field
from . import constants as C
from . import geometry as g
from . import thermo as th
from . import radiation as rad
from . import convergence as conv


@dataclass
class StageAConfig:
    nx: int = 80                       # cross-track columns (uniform in km)
    theta_max_deg: float = 70.0        # poleward truncation (deep cold, short of cryogenic N2 band)
    nz: int = 30                       # atmospheric layers
    p_top: float = 5.0e3               # model top [Pa]
    # water vapor: fixed relative humidity profile RH(sigma), sigma = p/p_surf
    rh_surf: float = 0.8
    rh_top: float = 0.2
    # radiation tunables (documented inputs; sweepable)
    co2_ppm: float = 400.0
    albedo_ocean: float = C.ALBEDO_OCEAN
    albedo_ice: float = C.ALBEDO_ICE
    albedo_dryland: float = C.ALBEDO_DRYLAND  # salt flat / bare rock in the hot zone (brighter than ocean)
    k_wv_lw: float = rad.K_WV_LW
    tau_lw_wellmixed: float = rad.TAU_LW_WELLMIXED
    i0_scale: float = 1.0              # multiply the fiducial suns' brightness (sensitivity sweep)
    # LW window band + humidity continuum (CLEARSKY_SW_PLAN Stage 2): default ON (the accurate model;
    # set False for the gray byte-identical limit). The hot-end OLR cap is a PREDICTED outcome.
    lw_window: bool = True
    f_lw_window: float = rad.F_LW_WINDOW
    k_cont_self: float = rad.K_CONT_SELF
    lw_conserving_split: bool = False  # issue #84 item 3: background band carries the out-of-window
                                       # CLEAR-SKY opacity /(1-f_win), restoring the Earth-calibrated gray
                                       # Planck-mean in the dry/thin limit; OFF = the accepted moved-baseline
                                       # (byte-identical). Fleet A/B: the rad_consband variant.
    # SW diffuse stream + haze forward-scatter (CLEARSKY_SW_PLAN Stage 5): default ON (set False for the
    # byte-identical limit). ON adds the haze forward-scattered diffuse-down surface SW
    # (geometry.day_mean_diffuse_flux), recovering the slant-haze light the direct flux drops as pure loss;
    # the midnight-clock role of TAU0_HAZE is untouched (separate code path), so only the surface-flux role changes.
    sw_diffuse: bool = True
    # Two-band shortwave (CLEARSKY_SW_PLAN Stage 1): default ON (set False for the gray one-beam limit). ON
    # splits the solar beam into a transparent visible fraction (reaches the surface regardless of WVP -> the
    # surface PAR/transmission FLOOR) + an absorbing near-IR fraction; both are spectroscopic givens.
    sw_bands: bool = True
    # transport (calibrated to Stage B; kappa=0 gives the local RCE limit)
    kappa_atm: float = 2.0e6           # atmospheric MSE diffusivity [m^2/s]
    kappa_ocn: float = 1.0e4           # ocean heat diffusivity [m^2/s] (much weaker, mixed layer)
    transport_mode: str = "diffusive"  # "diffusive" (THE DELIVERABLE; the IMEX/explicit kappa_atm
                                       # closure) or "injected": replace it with Stage B's diagnosed flux
                                       # divergence -dF_B/dx. "injected" is STRUCTURALLY INCOMPATIBLE
                                       # (breaks A's energy closure -- B's closed cell can't export across
                                       # A's sponge) and is kept only as the documented Issue-A attempt;
                                       # see OPEN_ISSUES.md Issue A (won't-fix-within-scope).
    transport_anchor: float = 1.0      # dimensionless multiplier on the injected-flux magnitude (only
                                       # used in the retired "injected" mode).
    implicit_diffusion: bool = True    # IMEX: solve the (stiff, latent-driven) MSE diffusion IMPLICITLY
                                       # (backward Euler, unconditionally stable) instead of the legacy
                                       # explicit scheme whose dt collapsed as ~nx^-3 at the salt-flat
                                       # moisture transition (OPEN_ISSUES Issue D). Same FV operator and
                                       # M*cp conversion => identical steady state; only stability/cost
                                       # differ. Set False for the legacy explicit scheme (regression).
    # radiating cold-boundary sponge (replaces the old outflow boundary; OPEN_ISSUES Issue B)
    sponge_width_deg: float = 8.0      # poleward band (theta_max-width .. theta_max) over which the
                                       # sponge ramps in; must sit well poleward of the ribbon
    sponge_relax_frac: float = 0.2     # per-step fraction of the way to LOCAL radiative equilibrium at
                                       # the sponge core (0 disables -> reverts to a closed wall)
    # surface / time stepping
    ocean_depth_spinup: float = 5.0    # accelerated mixed-layer depth for spin-up [m] (equil. independent of it)
    wind: float = 6.0                  # near-surface wind for bulk fluxes [m/s]
    conv_max_sweeps: int = 8           # max convective-adjustment sweeps per step (early-exits)
    dt: float = 7200.0                 # time step [s] (2 h). With implicit_diffusion the MSE-diffusion
                                       # CFL is GONE (backward Euler), so dt is limited only by the
                                       # nx-INDEPENDENT explicit local physics (~the radiative/diurnal
                                       # time): measured stable to ~21-23 h at nx up to 600. (Legacy
                                       # explicit dt collapsed as ~nx^-3: ~400 s at nx=200.)
    # convergence / guards
    tol: float = 1.0e-5                # |d<Tsurf>/dt| stop tolerance [K per step-check]
    conv_theta_max: float | None = None  # if set, the convergence drift is measured only over theta <=
                                       # this (the habitable band + hot zone), so the slow deep-cold
                                       # radiative tail does not gate the stop (OPEN_ISSUES Q8a/README).
    check_every: int = 200
    patience: int = 4
    max_iter: int = 4_000_000
    max_seconds: float = 9.0 * 3600
    checkpoint_path: str | None = None
    checkpoint_every: int = 20000


class StageA:
    def __init__(self, cfg: StageAConfig | None = None):
        self.cfg = cfg or StageAConfig()
        c = self.cfg
        if c.theta_max_deg >= 89.0:
            raise ValueError(f"theta_max_deg={c.theta_max_deg} >= 89: x=L*tan(theta) diverges near 90 deg; "
                             "use <= ~85 (the cold zone of interest is theta<~80).")
        # --- cross-track grid: UNIFORM IN PHYSICAL DISTANCE x [m], not theta ---
        x_max = g.x_of_theta(np.radians(c.theta_max_deg))
        self.x = (np.arange(c.nx) + 0.5) * (x_max / c.nx)          # cell centers [m]
        self.dx = x_max / c.nx
        self.theta = g.theta_of_x(self.x)
        self.theta_deg = np.degrees(self.theta)
        self.S = g.day_mean_flux(self.x) * c.i0_scale              # DERIVED day-mean flux [W/m^2]
        # Stage 5 haze forward-scatter (default OFF -> S_diffuse=None, S_sw_in=S, byte-identical). When on,
        # S_diffuse is the extra column-top diffuse-down SW (passed to column_radiation, which propagates it
        # to the surface on its own column path); S_sw_in = S + S_diffuse is the TOTAL down-SW the absorbed-SW
        # / energy-closure diagnostics count as incoming.
        self.S_diffuse = g.day_mean_diffuse_flux(self.x) * c.i0_scale if c.sw_diffuse else None
        self.S_sw_in = self.S + self.S_diffuse if c.sw_diffuse else self.S
        # --- vertical pressure grid (interfaces TOP->SURFACE), layers between ---
        self.p_edge = np.linspace(c.p_top, C.P_SURF, c.nz + 1)
        self.p_mid = 0.5 * (self.p_edge[:-1] + self.p_edge[1:])
        self.dm = np.diff(self.p_edge) / C.G_ACCEL                 # layer mass [kg/m^2]
        self.sigma = self.p_mid / C.P_SURF
        self.RH = c.rh_top + (c.rh_surf - c.rh_top) * self.sigma   # RH(sigma): moist below, dry aloft
        # surface heat capacity is per-column (ocean vs salt-flat rock), via _surf_heat_capacity()
        self.C_layer = self.dm * C.CP_AIR                          # per-layer heat capacity [J/m^2/K]
        # --- initial state: a mild guess (no anchor); columns relax from here ---
        T0 = (np.maximum(self.S, 1.0) / C.SIGMA_SB) ** 0.25        # local radiative-eq surface guess
        self.T_surf = np.clip(T0, 120.0, 360.0)
        z_init = -8000.0 * np.log(self.p_mid / C.P_SURF)          # scale-height z guess (no T_atm yet)
        self.T_atm = np.maximum(self.T_surf[:, None] - 6.5e-3 * z_init[None, :], 130.0)
        self._injected_F = None                                   # Stage B's poleward flux [W/m] at the
                                                                  # nx+1 interfaces (set per outer iter)
        self.iter = 0

    # ------------------------------------------------------------------ helpers
    def _hydrostatic_z(self, T_atm=None):
        """Layer-center heights z [m] from the hydrostatic relation, per column. Shape (nx, nz)."""
        T = self.T_atm if T_atm is None else T_atm
        # dz between interfaces from hypsometric eq; integrate from surface up
        Tedge = np.empty((self.cfg.nx, self.cfg.nz + 1))
        Tedge[:, 1:-1] = 0.5 * (T[:, :-1] + T[:, 1:])
        Tedge[:, 0] = T[:, 0]; Tedge[:, -1] = T[:, -1]
        dlnp = np.diff(np.log(self.p_edge))                        # negative going up? p increases to surf
        # p_edge increases TOP->SURFACE, so from surface (last) upward ln p decreases
        z_edge = np.zeros((self.cfg.nx, self.cfg.nz + 1))
        for k in range(self.cfg.nz - 1, -1, -1):                  # surface interface = nz (z=0) upward
            dz = (C.R_DRY * Tedge[:, k + 1] / C.G_ACCEL) * (np.log(self.p_edge[k + 1]) - np.log(self.p_edge[k]))
            z_edge[:, k] = z_edge[:, k + 1] + dz
        return 0.5 * (z_edge[:, :-1] + z_edge[:, 1:])

    def _albedo(self):
        """Surface albedo: ocean, brightening to ice at the cold edge (sea-ice ramp over T_surf
        -1.8..-3.8 C: f_ice=0 at 271.35 K = seawater freezing, f_ice=1 at 269.35 K) and to
        bright dry salt-flat/rock in the hot zone (where wetness->0)."""
        c = self.cfg
        wet = self._wetness()
        base = c.albedo_ocean * wet + c.albedo_dryland * (1.0 - wet)        # ocean <-> salt flat
        f_ice = np.clip((271.35 - self.T_surf) / 2.0, 0.0, 1.0)
        return base + (c.albedo_ice - base) * f_ice

    def _surf_heat_capacity(self):
        """Per-column effective surface heat capacity [J/m^2/K]: ocean mixed layer where wet, low-mass
        rock where dry (so the salt-flat hot zone equilibrates fast rather than blocking convergence).
        The equilibrium is C-independent; this only sets the spin-up rate."""
        c = self.cfg
        wet = self._wetness()
        C_ocean = C.slab_heat_capacity(c.ocean_depth_spinup, C.RHO_LIQ, C.CP_LIQ)
        C_rock = C.slab_heat_capacity(1.0, 2600.0, 800.0)                  # ~1 m rock (fast)
        return C_ocean * wet + C_rock * (1.0 - wet)

    def _q(self):
        # Absolute humidity is set by WATER AVAILABILITY, not by RH*qsat: over the waterless salt flats
        # the air is dry in absolute terms (near-zero q) however much the scorching air COULD hold --
        # the doc's dry equatorial rising branch. (Using RH*qsat fails here because qsat ~ 1 at >100 C,
        # so even tiny RH would imply a huge vapor path and an unphysical runaway greenhouse.) Ocean
        # columns carry the moist RH profile; salt flats carry a near-zero background.
        wet = self._wetness()[:, None]
        q_moist = self.RH[None, :] * th.qsat(self.T_atm, self.p_mid[None, :])
        q_dry = 1.0e-3 * self.sigma[None, :]               # ~0 absolute humidity over the dry flats
        return np.minimum(wet * q_moist + (1.0 - wet) * q_dry, 0.5)

    def _sponge_mask(self):
        """Poleward radiating-sponge weight in [0,1]: 0 equatorward of (theta_max - sponge_width),
        ramping linearly to 1 at the cold edge. Over this band the columns are relaxed toward their
        LOCAL radiative equilibrium so the poleward heat transport is radiated to space there rather
        than reflected by a wall -- the well-posed replacement for the old grid-dependent outflow
        (OPEN_ISSUES Issue B)."""
        c = self.cfg
        if c.sponge_relax_frac <= 0 or c.sponge_width_deg <= 0:
            return np.zeros(c.nx)
        th0 = c.theta_max_deg - c.sponge_width_deg
        return np.clip((self.theta_deg - th0) / c.sponge_width_deg, 0.0, 1.0)

    def _sponge_dTs(self, out, Csurf):
        """Radiating cold-boundary sponge (OPEN_ISSUES Issue B): over the poleward band, relax T_surf
        toward LOCAL radiative equilibrium (toa_net -> 0) so the poleward-transported heat is radiated
        to space there (by the unmodeled colder tail) instead of reflected by a wall. Explicit,
        BOUNDED relaxation: the target offset T_re - T_surf ~ toa_net/(dOLR/dT ~ 4 sigma T^3) is clipped
        (the linearization overshoots wildly for cold columns far from equilibrium during spin-up,
        which previously NaN'd at fine grids); the per-step move is sponge_relax_frac of the clipped
        offset (< 1 -> stable), and the clip is inactive near steady state, so it does not change the
        converged result. Returns the surface temperature increment [K]; stores the applied power for
        the energy-closure diagnostic."""
        c = self.cfg
        sp = self._sponge_mask()
        if not sp.any():
            self._sponge_power = np.zeros(c.nx)
            return np.zeros(c.nx)
        toa_net = (self.S_sw_in - out["planetary_reflected"]) - out["OLR"]  # W/m^2 (>0 heats the column)
        dTre = np.clip(toa_net / np.maximum(4 * C.SIGMA_SB * self.T_surf**3, 1.0), -100.0, 100.0)
        dTs_sp = sp * c.sponge_relax_frac * dTre                           # bounded relaxation step [K]
        self._sponge_power = Csurf * dTs_sp / c.dt                         # W/m^2 applied (>0 warms)
        return dTs_sp

    def _wetness(self):
        """Evaporation efficiency in [0,1]. Open ocean (1) where liquid water is stable; ramps to 0
        (dry salt flat) as SST exceeds the liquid-water limit -- the hot zone where the sea has boiled
        away to evaporite (cosmology S9). This is what caps the equatorial latent-heat runaway: a dry
        surface there reaches a bounded dry-rock radiative balance (the doc's ~135 C equator) instead
        of pumping unbounded latent heat into an atmosphere that cannot radiate it (Komabayashi-
        Ingersoll). Ramp centered between SST_WET_HI and SST_DRY (deg C)."""
        SST_WET_HI, SST_DRY = C.SST_WET_HI, C.SST_DRY   # shared ramp endpoints (constants.py)
        Tc = self.T_surf - C.T_FREEZE
        return np.clip((SST_DRY - Tc) / (SST_DRY - SST_WET_HI), 0.0, 1.0)

    # ------------------------------------------------------------------ one step
    def step(self):
        c = self.cfg
        self._z = self._hydrostatic_z()                                    # cache z once per step
        q = self._q()
        out = rad.column_radiation(self.p_edge[None, :], self.T_atm, q, self.T_surf, self.S,
                                   self._albedo(), co2_ppm=c.co2_ppm, k_wv=c.k_wv_lw,
                                   tau_wellmixed=c.tau_lw_wellmixed,
                                   lw_window=c.lw_window, f_win=c.f_lw_window, k_cont=c.k_cont_self,
                                   lw_conserving_split=c.lw_conserving_split,
                                   sw_bands=c.sw_bands, sw_diffuse=c.sw_diffuse, S_diffuse=self.S_diffuse)
        # surface turbulent fluxes (to the lowest layer) and their stiff derivatives wrt T_surf
        Ck = C.RHO_AIR0 * C.CP_AIR * 1.2e-3 * c.wind
        Ce = C.RHO_AIR0 * C.L_VAP * 1.2e-3 * c.wind
        SH = Ck * (self.T_surf - self.T_atm[:, -1])
        qs_surf = th.qsat(self.T_surf, C.P_SURF)
        wetness = self._wetness()                                          # ocean->salt-flat ramp
        wet = (qs_surf > q[:, -1]) & (wetness > 0)
        LE = wetness * Ce * np.where(wet, qs_surf - q[:, -1], 0.0)
        # --- surface energy balance: SEMI-IMPLICIT in T_surf (sigma T^4, SH, LE all stiff at high
        #     SST via Clausius-Clapeyron) -> unconditionally stable, finds the evaporative thermostat.
        R_net = out["surf_net"] - SH - LE                                  # tendency numerator (W/m^2)
        dRdT = (4 * C.SIGMA_SB * self.T_surf**3
                + Ck + np.where(wet, wetness * Ce * th.dqsat_dT(self.T_surf, C.P_SURF), 0.0))
        Csurf = self._surf_heat_capacity()
        dTs = c.dt * R_net / (Csurf + c.dt * dRdT)
        dTs += c.dt * self._diffuse(self.T_surf, c.kappa_ocn)              # ocean transport (explicit)
        dTs += self._sponge_dTs(out, Csurf)                                # radiating cold-boundary sponge
        # --- atmosphere: radiative heating + (SH+LE) deposited by mass; MSE transport added below ---
        dTa = out["H"] / self.C_layer
        turb = (SH + LE)[:, None] * (self.dm[None, :] / self.dm.sum())     # mass-weighted deposition
        dTa += turb / self.C_layer
        # MSE horizontal transport increment (IMEX implicit or legacy explicit), from start-of-step
        # state to match the legacy operator timing; applied as a uniform per-column T_atm shift.
        mse_delta = self._mse_transport_increment()
        # integrate
        self.T_surf = self.T_surf + dTs
        self.T_atm = self.T_atm + c.dt * dTa + mse_delta[:, None]
        # --- moist convective adjustment (enthalpy-conserving) ---
        self._convective_adjust()
        self.iter += 1
        return out

    def _diffuse(self, field, kappa):
        """Finite-volume diffusion tendency d/dt[field] for a 1-D (nx) field [units of field per s].
        Equator x=0: zero flux (symmetry). Cold edge: CLOSED wall (no flux through it) -- the poleward
        heat transport is removed by the radiating sponge in the interior poleward band, not by a
        grid-dependent outflow (OPEN_ISSUES Issue B)."""
        F = np.zeros(self.cfg.nx + 1)                              # fluxes at interfaces 0..nx
        F[1:-1] = -kappa * (field[1:] - field[:-1]) / self.dx      # interior interfaces
        F[0] = 0.0                                                 # equator symmetry: no flux
        F[-1] = 0.0                                                # cold edge: closed wall (sponge radiates)
        tend = (F[:-1] - F[1:]) / self.dx                          # divergence: into each cell
        return tend

    def set_injected_flux(self, F_interfaces):
        """Install Stage B's diagnosed poleward energy flux [W/m] at A's nx+1 interfaces (Issue-A
        coupling). Forces both walls to zero (equator symmetry + closed cold wall) so the injected
        -dF_B/dx adds zero net energy to the domain -- the energy-closure requirement. Called by the
        cascade each A<->B outer iteration."""
        F = np.asarray(F_interfaces, float).copy()
        if F.shape != (self.cfg.nx + 1,):
            raise ValueError(f"injected flux must have {self.cfg.nx + 1} interfaces, got {F.shape}")
        F[0] = 0.0
        F[-1] = 0.0
        self._injected_F = F

    def _injected_transport_increment(self):
        """Per-column uniform T_atm increment [K] from Stage B's injected flux divergence -dF_B/dx.

        Reuses the EXACT conservative divergence (F[:-1]-F[1:])/dx and the dt*heating/(M*cp) conversion
        of the diffusive scheme, so the sign convention and cell-by-cell conservation match. Applied
        EXPLICITLY: the injected forcing is a fixed source with no spatial coupling of the unknown (no
        diffusion operator), so there is no CFL stiffness and no IMEX solve. Before B exists (no flux
        installed) it returns zero -> A relaxes to local RCE."""
        c = self.cfg
        if self._injected_F is None:
            self._last_mse_flux = np.zeros(c.nx + 1)
            return np.zeros(c.nx)
        F = self._injected_F
        self._last_mse_flux = F                                    # diagnostic parity with the diffusive path
        heating_col = (F[:-1] - F[1:]) / self.dx                   # [W/m^2] per column (same divergence)
        M = self.dm.sum()
        return c.dt * heating_col / (M * C.CP_AIR)                 # explicit per-column uniform T increment

    def recompute_diffusive_flux(self):
        """A's down-gradient diffusive MSE flux F = -kappa_atm*dH_col/dx at the nx+1 interfaces [W/m],
        from the CURRENT state, REGARDLESS of transport_mode. This is the non-circular F_A for the
        Issue-A corr(F_A, F_B) metric: in injected mode A's transport IS B's flux, so comparing that to
        F_B would be circular -- the meaningful check is whether A's own thermodynamics (its column-MSE
        gradient) now IMPLIES a B-like transport. Read-only; does not touch state or _last_mse_flux."""
        z = self._hydrostatic_z()
        q = self._q()
        mse = C.CP_AIR * self.T_atm + C.G_ACCEL * z + C.L_VAP * q
        H_col = np.sum(self.dm[None, :] * mse, axis=1)
        F = np.zeros(self.cfg.nx + 1)
        F[1:-1] = -self.cfg.kappa_atm * (H_col[1:] - H_col[:-1]) / self.dx
        return F

    def _mse_transport_increment(self):
        """Per-column uniform T_atm increment [K] from one step of horizontal MSE transport.

        The transport is a diffusion of the column moist static energy H_col [J/m^2] with kappa_atm,
        discretized as the same finite-volume operator as before (zero flux at the equator symmetry edge
        and at the closed cold wall; the radiating sponge removes the poleward heat). The explicit
        forward-Euler increment is dt*heating/(M*cp) applied UNIFORMLY to every layer (mass-weighted
        heating / mass-weighted heat capacity == uniform), exactly the legacy scheme.

        With ``implicit_diffusion`` (default) the same transport is instead time-stepped by BACKWARD
        EULER (OPEN_ISSUES Issue D). The model transport equation is M*cp dT_c/dt = kappa_atm d2H/dx2
        (the MSE flux divergence deposited as sensible heat), and H(T_c) is nonlinear in T through the
        latent term L*q -- so the effective T-diffusivity is kappa_atm*(c_eff/cp), where c_eff = dH/dT_c
        is the MOIST heat capacity (cp + the geopotential R + L dq/dT). At the salt-flat moisture
        transition c_eff/cp reaches ~10x (the documented ~12x CFL deficit), so an explicit step needs a
        ~nx^-3-collapsing dt. Backward Euler linearizes H(T^{n+1}) = H^n + M c_eff dT_c and solves the
        tridiagonal system (I - (dt*kappa_atm/cp) d2/dx2 . c_eff) dT_c = dT_expl -- UNCONDITIONALLY
        STABLE for any dt, with the moist amplification carried INSIDE the implicit operator (the bug in
        a naive (I - dt*kappa*Lap) solve is that it omits c_eff/cp, leaving the stiff moist mode
        explicit). The FV spatial operator and the dT_expl forcing are identical to the explicit scheme,
        and the c_eff correction vanishes at steady state (dT_c->0), so the converged state is IDENTICAL
        to the explicit scheme -- only the transient path and the stability/cost differ. Also stores the
        MSE flux for the transport diagnostic."""
        c = self.cfg
        if c.transport_mode == "injected":
            return self._injected_transport_increment()           # Issue-A: B's -dF_B/dx replaces diffusion
        z = getattr(self, "_z", None)
        if z is None:
            z = self._hydrostatic_z()
        q = self._q()
        mse = C.CP_AIR * self.T_atm + C.G_ACCEL * z + C.L_VAP * q  # [J/kg] per layer
        H_col = np.sum(self.dm[None, :] * mse, axis=1)             # column MSE [J/m^2]
        F = np.zeros(c.nx + 1)
        F[1:-1] = -c.kappa_atm * (H_col[1:] - H_col[:-1]) / self.dx
        F[0] = 0.0
        F[-1] = 0.0                                               # closed wall (sponge radiates the excess)
        self._last_mse_flux = F                                    # save for the transport diagnostic
        heating_col = (F[:-1] - F[1:]) / self.dx                  # [W/m^2] per column (explicit divergence)
        M = self.dm.sum()
        dT_expl = c.dt * heating_col / (M * C.CP_AIR)             # forward-Euler increment [K] (per column)
        if not c.implicit_diffusion:
            return dT_expl
        # --- backward Euler: build the c_eff-weighted tridiagonal operator (rebuilt each step; c_eff
        #     tracks the state). c_eff = dH_col/dT_c for a uniform column warming: cp + R (geopotential)
        #     + the latent feedback L*dq/dT (fixed-RH Clausius-Clapeyron, only where wet & uncapped). ---
        qmoist = self.RH[None, :] * th.qsat(self.T_atm, self.p_mid[None, :])
        wet = self._wetness()[:, None]
        active = (wet > 0) & (np.minimum(wet * qmoist, 0.5) < 0.5)   # latent feedback live (not dry/capped)
        dqdT = np.where(active, wet * self.RH[None, :] * th.dqsat_dT(self.T_atm, self.p_mid[None, :]), 0.0)
        c_eff = C.CP_AIR + C.R_DRY + C.L_VAP * np.sum(self.dm[None, :] * dqdT, axis=1) / M   # [J/kg/K], (nx,)
        beta = (c.dt * c.kappa_atm / C.CP_AIR) / self.dx**2        # implicit diffusion number / c_eff
        ab = np.zeros((3, c.nx))
        ab[0, 1:] = -beta * c_eff[1:]                             # super-diagonal:  -beta c_{i+1}
        ab[2, :-1] = -beta * c_eff[:-1]                           # sub-diagonal:    -beta c_{i-1}
        ab[1, :] = 1.0 + 2.0 * beta * c_eff                       # diagonal (interior): 1 + 2 beta c_i
        ab[1, 0] = 1.0 + beta * c_eff[0]                          # Neumann ends (closed wall): one-sided
        ab[1, -1] = 1.0 + beta * c_eff[-1]
        return sla.solve_banded((1, 1), ab, dT_expl)

    def _convective_adjust(self, max_sweeps=None):
        """Moist convective adjustment: remove super-(moist-)adiabatic layers by enthalpy-conserving
        pairwise mixing (Manabe-style), vectorized over columns. Conserves sum(cp dm T) per column.
        Capped at ``conv_max_sweeps`` per call (early-exits when nothing is unstable); the profile
        relaxes over many steps, so a per-step cap costs accuracy only transiently in spin-up."""
        nz = self.cfg.nz
        max_sweeps = max_sweeps or self.cfg.conv_max_sweeps
        z = getattr(self, "_z", None)
        if z is None:
            z = self._hydrostatic_z()
        for _ in range(max_sweeps):
            changed = np.zeros(self.cfg.nx, bool)
            for k in range(nz - 1, 0, -1):                        # pair (k below/higher-p, k-1 above)
                Gamma = th.moist_adiabat_lapse_rate(self.T_atm[:, k], self.p_mid[k])   # K/m
                dz = z[:, k - 1] - z[:, k]                        # >0 (k-1 is higher)
                T_above_stable = self.T_atm[:, k] - Gamma * dz
                unstable = self.T_atm[:, k - 1] < T_above_stable - 1e-9
                if not unstable.any():
                    continue
                mk, mk1 = self.dm[k], self.dm[k - 1]
                Ctot = mk * self.T_atm[:, k] + mk1 * self.T_atm[:, k - 1]
                Tk_new = (Ctot + mk1 * Gamma * dz) / (mk + mk1)
                Tk1_new = Tk_new - Gamma * dz
                self.T_atm[unstable, k] = Tk_new[unstable]
                self.T_atm[unstable, k - 1] = Tk1_new[unstable]
                changed |= unstable
            if not changed.any():
                break

    # ------------------------------------------------------------------ diagnostics
    def diagnostics(self):
        c = self.cfg
        q = self._q()
        out = rad.column_radiation(self.p_edge[None, :], self.T_atm, q, self.T_surf, self.S,
                                   self._albedo(), co2_ppm=c.co2_ppm, k_wv=c.k_wv_lw,
                                   tau_wellmixed=c.tau_lw_wellmixed,
                                   lw_window=c.lw_window, f_win=c.f_lw_window, k_cont=c.k_cont_self,
                                   lw_conserving_split=c.lw_conserving_split,
                                   sw_bands=c.sw_bands, sw_diffuse=c.sw_diffuse, S_diffuse=self.S_diffuse)
        self._mse_transport_increment()                          # refresh _last_mse_flux
        # energy closure on the CLOSED domain (both walls zero-flux): the radiating sponge removes the
        # poleward-transported heat, so the net TOA imbalance integrated over the domain must equal the
        # sponge's net heat removal (the export to the unmodeled colder tail):
        #   int(absorbed SW - OLR) dx == -int(sponge power) dx.
        absorbed_sw = self.S_sw_in - out["planetary_reflected"]  # incoming SW counts the Stage-5 diffuse haze
        toa_net = absorbed_sw - out["OLR"]
        toa_integral = np.trapezoid(toa_net, self.x)             # W/m along-track
        # sponge power consistent with this fresh radiation call (see _sponge_dTs):
        sp = self._sponge_mask()
        dTre = np.clip(toa_net / np.maximum(4 * C.SIGMA_SB * self.T_surf**3, 1.0), -100.0, 100.0)
        sponge_power = self._surf_heat_capacity() * (sp * c.sponge_relax_frac * dTre) / c.dt  # W/m^2 (>0 warms)
        lateral_export = -float(np.trapezoid(sponge_power, self.x))   # W/m radiated by the sponge (>0)
        return dict(
            theta_deg=self.theta_deg, x_km=self.x / 1e3, S=self.S,
            T_surf_C=self.T_surf - C.T_FREEZE, OLR=out["OLR"], D_surf=out["D_surf"],
            albedo=self._albedo(), toa_net=toa_net,
            toa_integral=toa_integral, lateral_export=lateral_export,
            energy_closure_resid=toa_integral - lateral_export,
            mse_flux=self._last_mse_flux, T_atm=self.T_atm.copy(),
            habitable_mask=(self.T_surf - C.T_FREEZE > 0) & (self.T_surf - C.T_FREEZE < 30),
        )

    def state(self):
        return dict(T_surf=self.T_surf, T_atm=self.T_atm, iter=self.iter)

    def load_state(self, st):
        self.T_surf = np.asarray(st["T_surf"]); self.T_atm = np.asarray(st["T_atm"])
        self.iter = int(st.get("iter", 0))

    # ------------------------------------------------------------------ run loop
    def run(self, verbose=True):
        c = self.cfg
        if c.checkpoint_path:
            st, _ = conv.load_checkpoint(c.checkpoint_path)
            if st is not None:
                self.load_state(st)
                if verbose:
                    print(f"[StageA] resumed from checkpoint at iter {self.iter}")
        # steady-state diagnostic: max |dT_surf| per step over the check interval (K/step), the proper
        # rate-of-change-of-state metric (robust to compensating drifts a global mean would hide).
        # With conv_theta_max set, the metric covers only theta <= conv_theta_max, so the slow deep-cold
        # radiative tail (tiny sigma T^4, creeps for a long time) does not gate the stop (OPEN_ISSUES Q8a).
        conv_mask = (np.ones(c.nx, bool) if c.conv_theta_max is None
                     else self.theta_deg <= c.conv_theta_max)
        mon = conv.ConvergenceMonitor(tol=c.tol, patience=c.patience, max_iter=c.max_iter,
                                      max_seconds=c.max_seconds, scale=1.0, name="max|dTsurf|/step",
                                      absolute=True)
        T_prev = self.T_surf.copy()
        while True:
            for _ in range(c.check_every):
                self.step()
            drift = float(np.max(np.abs((self.T_surf - T_prev)[conv_mask]))) / c.check_every
            T_prev = self.T_surf.copy()
            stop = mon.update(self.iter, drift)
            if verbose and (self.iter % (c.check_every * 10) == 0 or stop):
                d = self.diagnostics()
                rib = (self.theta_deg >= 45) & (self.theta_deg <= 51)
                print(f"[StageA] it={self.iter:>8} <Tsurf>={np.mean(self.T_surf)-273.15:7.2f}C "
                      f"ribbonSST={np.mean(d['T_surf_C'][rib]):6.2f}C drift={drift:.2e}K/step "
                      f"clos.resid={d['energy_closure_resid']:.2e}W/m")
            # checkpoint once per checkpoint_every WINDOW: iter advances by check_every each loop, so an
            # ``== 0`` test silently never fires unless check_every divides checkpoint_every -- ``< check_every``
            # fires on the first check that crosses the window boundary, robust to any (check_every,
            # checkpoint_every) pairing (so a guarded long run is never left without a periodic resume point).
            if c.checkpoint_path and self.iter % c.checkpoint_every < c.check_every:
                conv.save_checkpoint(c.checkpoint_path, self.state(), {"iter": self.iter})
            if stop:
                break
        if c.checkpoint_path:
            conv.save_checkpoint(c.checkpoint_path, self.state(), {"iter": self.iter})
        if verbose:
            print(f"[StageA] STOP: {mon.stop_reason}  ({mon.elapsed:.1f}s, {self.iter} iters)")
        return mon.summary()
