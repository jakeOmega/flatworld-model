"""
flatworld.stage_b_overturning -- Stage B: the 2-D (x, z) non-rotating overturning + ocean structure.

AUTHORITATIVE for the zonal-mean circulation/ocean structure (Stage A only parameterizes transport).
It DERIVES the circulation shape independently rather than assuming the prior work's. Two pieces, both
f = 0 (no Coriolis):

  (1) StageBAtmosphere -- a streamfunction-vorticity Boussinesq solver in the (cross-track, height)
      plane. It is NON-DIMENSIONAL with a turbulent effective Rayleigh number and a compressed aspect
      ratio: on a non-rotating world the circulation SHAPE (number of cells, the direction of the
      surface wind, the height of the moisture-bearing return) is set by the Rossby number (= infinity),
      not by absolute scales, so a scaled solver fixes the shape and direction -- not the wind speeds.
      Driven by the buoyancy gradient implied by the Stage-A surface temperature backbone (NOT a
      cos^3 Newtonian target), with optional interactive moisture. It tests the doc's "robust" claims
      INDEPENDENTLY: (a) a SINGLE overturning cell with an equatorward surface wind at every latitude,
      and (b) the moisture transport is POLEWARD ALOFT (rain falls at/poleward of the ribbon).

  (2) wind_driven_ocean_profile -- the f=0 wind-driven momentum profile: with no Coriolis there is no
      Ekman layer; the homogeneous cubic reverses at z = -H/3 and the lower 2/3 runs opposite the
      surface (the three-layer stack, upwelling in the middle). Verified against the analytic cubic.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from dataclasses import dataclass
from . import constants as C
from . import geometry as g
from . import thermo as th
from . import convergence as conv


# ======================================================================================
# (2) The wind-driven ocean momentum profile with f = 0 (analytic).
# ======================================================================================
def wind_driven_ocean_profile(nz=400, H=200.0):
    """Steady wind-driven velocity profile u(z) in a homogeneous layer of depth H, NO rotation.

    With no Coriolis the wind-driven flow is not pinned into a thin Ekman layer; the steady momentum
    balance (constant interior pressure gradient set by zero NET transport, surface stress at z=0,
    no-slip at z=-H) has the cubic solution
        u(s) ∝ (3s + 1)(s + 1),   s = z/H in [-1, 0],
    with roots at s = -1 (no-slip bottom) and s = -1/3 (the interior reversal). The upper third runs
    with the wind; the lower two thirds run against it, peaking at -1/3 of the surface speed at
    s = -2/3. Returns (z, u_normalized, reversal_depth, reversal_index)."""
    s = np.linspace(-1.0, 0.0, nz)
    u = (3.0 * s + 1.0) * (s + 1.0)
    u = u / u[-1]                                   # unit surface speed (s=0)
    z = s * H
    # the INTERIOR reversal (exclude the s=-1 bottom root): first sign change above the bottom
    interior = np.where(np.diff(np.sign(u[1:])) != 0)[0]
    rev_idx = int(interior[0]) + 1 if interior.size else nz // 3
    return z, u, -H / 3.0, rev_idx


# ======================================================================================
# (1) The 2-D atmospheric overturning (non-dimensional streamfunction-vorticity Boussinesq, f = 0).
# ======================================================================================
@dataclass
class StageBConfig:
    nx: int = 200                      # cross-track points (compressed-aspect box)
    nz: int = 41                       # vertical points
    theta_max_deg: float = 60.0        # poleward truncation
    aspect: float = 12.0               # nondim box width (compressed; fixes shape, not aspect)
    Ra: float = 60.0                   # effective (turbulent) Rayleigh number -- supercritical
    nu: float = 6.0e-3                 # nondim eddy viscosity
    kappa_b: float = 6.0e-3            # nondim buoyancy diffusivity
    tau_relax: float = 2.0             # nondim radiative-relaxation time
    hd: float = 0.4                    # nondim surface-forcing decay height
    N2: float = 0.15                   # nondim background stratification
    moist: bool = True
    kappa_q: float = 4.0e-3
    evap_eff: float = 0.8
    lam_latent: float = 4.0            # latent-heating coupling (nondim; ~ (L/cp) qsat / dT_dry)
    dt: float = 3.0e-3
    tol: float = 2e-5                  # steady-state tol on normalized max|d psi|/step
    check_every: int = 500
    patience: int = 4
    max_iter: int = 3_000_000
    max_seconds: float = 9.0 * 3600
    poisson_method: str = "dst"        # streamfunction inverter (INF-8): "dst" (default) = the separable
                                       # DST(x) x tridiag(z) SeparablePoisson ported from the AD core -- a
                                       # machine-precision drop-in for the LU solve at a fraction of the cost;
                                       # "splu" = the pre-INF-8 sparse-LU back-substitution, kept behind the
                                       # flag as the A/B reference (a solver swap is not byte-identical).
    fuse_upwind: bool = False          # opt-in: numba-fuse the upwind advection tendency (a pure wall-clock
                                       # lever, BIT-IDENTICAL to the numpy path; needs numba). Default OFF =
                                       # the numpy path, so the default run never requires numba.
    checkpoint_path: str | None = None
    checkpoint_every: int = 50000
    Tsurf_C_of_theta: object = None    # callable theta_deg -> day-mean SST/T (deg C), from Stage A
    wetness_of_theta: object = None    # callable theta_deg -> evaporation efficiency in [0,1], from
                                       # Stage A's wetness ramp (no evaporation over the salt-flat hot
                                       # zone). None -> recompute A's 55->70 C ramp from Tsurf_C.


_STAGEB_UPWIND_NUMBA = None        # njit upwind-tendency kernel, JIT-compiled once on first fuse_upwind use


def _build_stageb_upwind_numba():
    """Compile (once) the fused first-order upwind advection tendency ``u . grad F`` for Stage B (INF-8).

    A single pass over the grid computes ``where(u>0, u*Fxm, u*Fxp) + where(w>0, w*Fzm, w*Fzp)`` with NO
    temporaries -- versus the numpy ``_upwind``'s six full-grid arrays per call. It is BIT-IDENTICAL to the
    numpy path (verified in ``test_stage_b``): pure fp64 ``+,-,*,/`` in the SAME divided-difference and
    branch order, no transcendentals and no cross-cell accumulation, and ``@njit(cache=True)`` (no
    ``fastmath``/``parallel``) keeps strict IEEE ordering. Wall zeros match the numpy edges exactly (the
    backward/forward difference that would reach outside the grid is set to 0, as ``np.zeros_like`` leaves
    it). Default OFF; built only when ``cfg.fuse_upwind`` is set, so the default run never imports numba."""
    from numba import njit          # lazy: only imported when fuse_upwind is ON

    @njit(cache=True)
    def _upwind_kernel(F, u, w, dxi, dze):
        nz, nx = F.shape
        out = np.empty_like(F)
        for j in range(nz):
            for i in range(nx):
                uu = u[j, i]
                if uu > 0.0:                                    # backward x-difference (Fxm; 0 at the i=0 wall)
                    fx = (F[j, i] - F[j, i - 1]) / dxi if i >= 1 else 0.0
                else:                                           # forward x-difference (Fxp; 0 at the i=nx-1 wall)
                    fx = (F[j, i + 1] - F[j, i]) / dxi if i <= nx - 2 else 0.0
                ww = w[j, i]
                if ww > 0.0:                                    # backward z-difference (Fzm; 0 at the j=0 wall)
                    fz = (F[j, i] - F[j - 1, i]) / dze if j >= 1 else 0.0
                else:                                           # forward z-difference (Fzp; 0 at the j=nz-1 wall)
                    fz = (F[j + 1, i] - F[j, i]) / dze if j <= nz - 2 else 0.0
                out[j, i] = uu * fx + ww * fz
        return out

    return _upwind_kernel


class StageBAtmosphere:
    def __init__(self, cfg: StageBConfig | None = None):
        self.cfg = cfg or StageBConfig()
        c = self.cfg
        # nondim grid: xi in [0, aspect], zeta in [0, 1]
        self.xi = np.linspace(0.0, c.aspect, c.nx)
        self.zeta = np.linspace(0.0, 1.0, c.nz)
        self.dxi = self.xi[1] - self.xi[0]
        self.dze = self.zeta[1] - self.zeta[0]
        # map xi -> physical cross-track distance -> theta (uniform in DISTANCE)
        x_max = g.x_of_theta(np.radians(c.theta_max_deg))
        self.x_phys = (self.xi / c.aspect) * x_max
        self.theta_deg = np.degrees(g.theta_of_x(self.x_phys))
        # surface temperature backbone (from Stage A) -> normalized surface buoyancy
        self.Tsurf_C = self._Tsurf_C(self.theta_deg)
        Ts = self.Tsurf_C - self.Tsurf_C.mean()
        bs = Ts / (Ts.max() - Ts.min() + 1e-9)                       # normalized, zero-mean (nx,)
        Z = np.tile(self.zeta, (c.nx, 1)).T                          # (nz, nx)
        self.b_eq = bs[None, :] * np.exp(-Z / c.hd) + c.N2 * Z        # target buoyancy (nz, nx)
        self.b = self.b_eq.copy()
        self.omega = np.zeros((c.nz, c.nx))
        self.psi = np.zeros((c.nz, c.nx))
        self.q = np.zeros((c.nz, c.nx))
        self.precip = np.zeros(c.nx)
        # ABSOLUTE saturation field [kg/kg] from the PHYSICAL column temperature (surface T minus
        # lapse*height). Issue-A rebuild: NOT normalized -- the carried moisture must saturate where the
        # real aloft capacity drops, not where a normalized tracer crosses a normalized threshold.
        Tcol_K = (self.Tsurf_C[None, :] + C.T_FREEZE) - 6.5e-3 * (Z * 12000.0)   # ~12 km deep
        p_col = C.P_SURF * np.exp(-(Z * 12000.0) / 8000.0)
        self.qsat = th.qsat(np.maximum(Tcol_K, 150.0), p_col)        # absolute kg/kg
        # evaporative source = Stage A's WETNESS ramp (no evaporation over the salt-flat hot zone), not a
        # binary is_sea -- the real wet ocean is the only moisture source (Issue-A rebuild).
        self.wetness = self._wetness_of_theta(self.theta_deg)        # (nx,) in [0,1]
        self._build_poisson()
        # opt-in numba-fused upwind tendency (INF-8): built once per process, bit-identical to the numpy path.
        self._upwind_kernel = None
        if self.cfg.fuse_upwind:
            global _STAGEB_UPWIND_NUMBA
            if _STAGEB_UPWIND_NUMBA is None:
                _STAGEB_UPWIND_NUMBA = _build_stageb_upwind_numba()
            self._upwind_kernel = _STAGEB_UPWIND_NUMBA
        self.iter = 0

    def _Tsurf_C(self, theta_deg):
        if self.cfg.Tsurf_C_of_theta is not None:
            return np.asarray(self.cfg.Tsurf_C_of_theta(theta_deg), float)
        return np.interp(theta_deg, [0, 30, 45, 50, 55, 60],
                         [125, 95, 45, 25, -10, -55])                # standalone fallback only

    def _wetness_of_theta(self, theta_deg):
        """Evaporation efficiency in [0,1] from Stage A's wetness ramp (ocean=1, salt flat=0). Uses the
        passed-in callable (the cascade wires A's own _wetness); else recomputes A's 55->70 C SST ramp
        from this stage's SST backbone so the standalone mask is still A-consistent."""
        if self.cfg.wetness_of_theta is not None:
            return np.clip(np.asarray(self.cfg.wetness_of_theta(theta_deg), float), 0.0, 1.0)
        SST_WET_HI, SST_DRY = C.SST_WET_HI, C.SST_DRY                # shared ramp endpoints (constants.py)
        return np.clip((SST_DRY - self.Tsurf_C) / (SST_DRY - SST_WET_HI), 0.0, 1.0)

    def _build_poisson(self):
        """Factorize the constant-coefficient (Boussinesq) streamfunction Poisson ``M psi = omega`` with
        ``psi = 0`` Dirichlet on all four walls. Both solvers expose the SAME ``.solve(b_flat) -> psi_flat``
        interface (a flat ``nz*nx`` RHS in, a flat psi out), selected by ``cfg.poisson_method`` (INF-8):

          "dst"  (default) -- the separable DST(x) x tridiag(z) inverter ``SeparablePoisson``, ported from the
                 AD core (``stage_ad_core``). Stage B's operator IS the anelastic Poisson at ``beta = 1/rho_ref
                 = 1`` (constant density) on a uniform ``(dxi, dze)`` grid, so ``SeparablePoisson`` built with
                 ``rho_ref = 1`` reproduces THIS exact discrete operator and inverts it at a small fraction of
                 the LU back-substitution cost -- machine-precision-identical to the "splu" solve for any RHS
                 (verified in ``test_stage_b`` against this stage's own splu build).
          "splu" -- the pre-INF-8 sparse-LU factorization + per-step back-substitution, retained UNCHANGED
                 behind the flag as the A/B reference (a solver swap is not byte-identical, so both are kept).
        """
        nz, nx = self.cfg.nz, self.cfg.nx
        if self.cfg.poisson_method == "dst":
            # Lazy import breaks the stage_ad_core <-> stage_b_overturning module cycle (the core imports
            # wind_driven_ocean_profile from here AT MODULE LOAD, so a top-level import back would be
            # circular). By StageBAtmosphere.__init__ time both modules are fully imported, so this resolves
            # cleanly and costs only a dict lookup.
            from .stage_ad_core import SeparablePoisson
            self._lu = SeparablePoisson(np.ones(nz), nx, nz, self.dxi, self.dze)
        elif self.cfg.poisson_method == "splu":
            idx = lambda j, i: j * nx + i
            ix2, iz2 = 1.0 / self.dxi**2, 1.0 / self.dze**2
            r, col, val = [], [], []
            for j in range(nz):
                for i in range(nx):
                    k = idx(j, i)
                    if j in (0, nz - 1) or i in (0, nx - 1):
                        r.append(k); col.append(k); val.append(1.0)
                    else:
                        r += [k] * 5
                        col += [k, idx(j, i + 1), idx(j, i - 1), idx(j + 1, i), idx(j - 1, i)]
                        val += [-2 * ix2 - 2 * iz2, ix2, ix2, iz2, iz2]
            self._lu = splu(sp.csc_matrix((val, (r, col)), shape=(nz * nx, nz * nx)))
        else:
            raise ValueError(f"poisson_method must be 'dst' or 'splu', got {self.cfg.poisson_method!r}")

    def _ddx(self, F):
        o = np.zeros_like(F)
        o[:, 1:-1] = (F[:, 2:] - F[:, :-2]) / (2 * self.dxi)
        o[:, 0] = (F[:, 1] - F[:, 0]) / self.dxi
        o[:, -1] = (F[:, -1] - F[:, -2]) / self.dxi
        return o

    def _ddz(self, F):
        o = np.zeros_like(F)
        o[1:-1, :] = (F[2:, :] - F[:-2, :]) / (2 * self.dze)
        o[0, :] = (F[1, :] - F[0, :]) / self.dze
        o[-1, :] = (F[-1, :] - F[-2, :]) / self.dze
        return o

    def _lap(self, F):
        o = np.zeros_like(F)
        o[1:-1, 1:-1] = ((F[1:-1, 2:] - 2 * F[1:-1, 1:-1] + F[1:-1, :-2]) / self.dxi**2
                         + (F[2:, 1:-1] - 2 * F[1:-1, 1:-1] + F[:-2, 1:-1]) / self.dze**2)
        return o

    def _upwind(self, F, u, w):
        if self._upwind_kernel is not None:                          # INF-8 fused path (bit-identical, opt-in)
            return self._upwind_kernel(F, u, w, self.dxi, self.dze)
        Fxm = np.zeros_like(F); Fxp = np.zeros_like(F); Fzm = np.zeros_like(F); Fzp = np.zeros_like(F)
        Fxm[:, 1:] = (F[:, 1:] - F[:, :-1]) / self.dxi
        Fxp[:, :-1] = (F[:, 1:] - F[:, :-1]) / self.dxi
        Fzm[1:, :] = (F[1:, :] - F[:-1, :]) / self.dze
        Fzp[:-1, :] = (F[1:, :] - F[:-1, :]) / self.dze
        return np.where(u > 0, u * Fxm, u * Fxp) + np.where(w > 0, w * Fzm, w * Fzp)

    def velocities(self):
        return self._ddz(self.psi), -self._ddx(self.psi)

    def step(self):
        c = self.cfg
        self.psi = self._lu.solve((self.omega).ravel()).reshape(c.nz, c.nx)
        u, w = self.velocities()
        cond = self._moisture_step(u, w) if c.moist else 0.0
        self.omega = self.omega + c.dt * (-self._upwind(self.omega, u, w)
                                          - c.Ra * self._ddx(self.b) + c.nu * self._lap(self.omega))
        self.b = (self.b + c.dt * (-self._upwind(self.b, u, w) + (self.b_eq - self.b) / c.tau_relax
                                   + c.kappa_b * self._lap(self.b)) + c.lam_latent * cond)
        # BCs: omega=0 on walls (free overturning); b zero-gradient on walls
        self.omega[0, :] = 0; self.omega[-1, :] = 0; self.omega[:, 0] = 0; self.omega[:, -1] = 0
        self.b[0, :] = self.b[1, :]; self.b[-1, :] = self.b[-2, :]
        self.b[:, 0] = self.b[:, 1]; self.b[:, -1] = self.b[:, -2]
        self.iter += 1
        return u, w

    def _moisture_step(self, u, w):
        c = self.cfg
        # conservative-ish advection + diffusion
        self.q = self.q + c.dt * (-self._upwind(self.q, u, w) + c.kappa_q * self._lap(self.q))
        # Evaporate into the lowest level from the WET OCEAN (A's wetness), toward ABSOLUTE qsat. NOTE this
        # is a PER-STEP saturation relaxation with NO dt: the bottom row jumps wetness*evap_eff of the way to
        # qsat each step while advection only removes O(dt), so for any wetness*evap_eff >~ 0.05 the surface
        # floor effectively pins at qsat REGARDLESS of the ramp value -- the wetness/evap_eff ramp is thus
        # near-BINARY (on/off), and the source is formally dt-dependent. Left as-is on purpose: B is consumed
        # SHAPE-only (its precip is normalized to F(theta) in cascade.py), so the pinned-saturated source sets
        # the circulation-shaped band and the absolute rate is discarded. The takeaway for anyone sweeping
        # evap_eff / B-wetness: they barely move the result (issue #88 item 3). The lam_latent term in step()
        # sits outside the dt bracket for the same reason (cond is already a per-step condensation amount).
        evap = self.wetness * np.maximum(self.qsat[0, :] - self.q[0, :], 0.0) * c.evap_eff
        self.q[0, :] = self.q[0, :] + evap
        # condense supersaturation -> precip (the only sink)
        cond = np.maximum(self.q - self.qsat, 0.0)
        self.q = self.q - cond
        self.precip = 0.98 * self.precip + 0.02 * (cond.sum(axis=0) * self.dze / c.dt)
        return cond

    def diagnostics(self):
        u, w = self.velocities()
        u_surf = u[1, :]
        s = np.sign(u_surf[3:-3])
        sign_changes = int(np.sum(s[1:] * s[:-1] < 0))
        # moisture flux by height: <u q> integrated over xi; poleward = +xi
        uq = u * self.q
        flux_by_z = np.array([np.trapezoid(uq[j, :], self.xi) for j in range(self.cfg.nz)])
        upper_half = self.zeta > 0.5
        lower_half = ~upper_half
        flux_upper = float(np.mean(flux_by_z[upper_half]))
        flux_lower = float(np.mean(flux_by_z[lower_half]))
        # nondim poleward HEAT transport shape int(u*b)dz per column (the Issue-A transport profile;
        # its peak latitude is what disagreed with A's diffusive closure -- dimensionalized in the
        # cascade and injected into A as -dF_B/dx). Distinct from the moisture flux flux_by_z (int u*q).
        heat_flux_nd = np.sum(u * self.b, axis=0) * self.dze
        # rising branch location (max w at mid-height)
        wmid = w[self.cfg.nz // 2, :]
        i_rise = int(np.argmax(wmid))
        return dict(theta_deg=self.theta_deg, u=u, w=w, psi=self.psi.copy(), b=self.b.copy(),
                    q=self.q.copy(), u_surf=u_surf, surf_sign_changes=sign_changes,
                    max_psi=float(np.abs(self.psi).max()), precip=self.precip.copy(),
                    flux_by_z=flux_by_z, flux_upper=flux_upper, flux_lower=flux_lower,
                    heat_flux_nd=heat_flux_nd, wetness=self.wetness.copy(),
                    rising_theta=self.theta_deg[i_rise], x_phys=self.x_phys)

    def state(self):
        return dict(b=self.b, omega=self.omega, q=self.q, iter=self.iter)

    def load_state(self, st):
        self.b = np.asarray(st["b"]); self.omega = np.asarray(st["omega"])
        self.q = np.asarray(st.get("q", self.q)); self.iter = int(st.get("iter", 0))

    def run(self, verbose=True):
        c = self.cfg
        if c.checkpoint_path:
            st, _ = conv.load_checkpoint(c.checkpoint_path)
            if st is not None:
                self.load_state(st)
                if verbose: print(f"[StageB] resumed at iter {self.iter}")
        mon = conv.ConvergenceMonitor(tol=c.tol, patience=c.patience, max_iter=c.max_iter,
                                      max_seconds=c.max_seconds, name="max|dpsi|/|psi|/step",
                                      absolute=True)
        psi_prev = self.psi.copy()
        while True:
            for _ in range(c.check_every):
                self.step()
            # INF-8: reuse the psi the last step() already solved (drop the redundant per-check re-solve).
            # step() solves psi = M^{-1} omega at its START, so after check_every steps self.psi is the
            # streamfunction one step stale in omega -- immaterial to a steady-state drift tolerance (the
            # drift is sampled every check_every steps, and both ends of the difference use this convention).
            denom = max(np.abs(self.psi).max(), 1e-12)
            drift = float(np.max(np.abs(self.psi - psi_prev))) / denom / c.check_every
            psi_prev = self.psi.copy()
            stop = mon.update(self.iter, drift)
            if verbose and (self.iter % (c.check_every * 10) == 0 or stop):
                d = self.diagnostics()
                print(f"[StageB] it={self.iter:>8} max|psi|={d['max_psi']:.3e} "
                      f"surf-sign-changes={d['surf_sign_changes']} rising@th={d['rising_theta']:.0f} "
                      f"flux(up/lo)={d['flux_upper']:+.2e}/{d['flux_lower']:+.2e} drift={drift:.2e}")
            # checkpoint once per checkpoint_every WINDOW: iter advances by check_every each loop, so an
            # ``== 0`` test silently never fires unless check_every divides checkpoint_every -- ``< check_every``
            # fires on the first check that crosses the window boundary, robust to any (check_every,
            # checkpoint_every) pairing (matches Stage A / the AD core; a guarded long run always gets a
            # periodic resume point).
            if c.checkpoint_path and self.iter % c.checkpoint_every < c.check_every:
                conv.save_checkpoint(c.checkpoint_path, self.state(), {"iter": self.iter})
            if stop:
                break
        if verbose:
            print(f"[StageB] STOP: {mon.stop_reason} ({mon.elapsed:.1f}s)")
        return mon.summary()


def dimensional_poleward_flux(B, anchor):
    """Map Stage B's nondimensional overturning heat-transport shape to a DIMENSIONAL poleward column
    energy flux F_B(theta) [W/m], on B's own theta grid (Issue-A coupling).

    B (Rossby=inf) fixes the transport SHAPE -- WHERE the poleward flux peaks in latitude -- not its
    magnitude, so the single ``anchor`` [W/m] carries ALL the dimensional scales (a velocity scale, the
    buoyancy->temperature conversion, rho*cp*depth) and is the swept free parameter (the new
    kappa-equivalent). F_B = anchor * (int(u*b)dz normalized to unit peak). Injecting -dF_B/dx into A
    thus moves A's transport DIVERGENCE onto B's latitudinal shape (the +0.20 fix), with the magnitude
    a documented free knob -- NOT a scalar on A's own shape (which would reproduce the +0.20).

    F_B VANISHES at both walls (u = dpsi/dz = 0 where psi=0 along the equator-symmetry and cold-wall
    boundaries), the requirement for the injected divergence to add zero net energy to Stage A. Asserted
    then set exactly, as a guard against future boundary-condition changes.

    Returns (theta_deg, F_B[W/m]) on B's grid; the cascade interpolates onto A's interface latitudes."""
    u, _ = B.velocities()
    g_nd = np.sum(u * B.b, axis=0) * B.dze                     # nondim int(u*b)dz, (nx,)
    shape = g_nd / (np.max(np.abs(g_nd)) + 1e-300)             # unit-peak transport shape
    F_B = anchor * shape
    tol = 1e-3 * np.max(np.abs(F_B))
    assert abs(F_B[0]) <= tol and abs(F_B[-1]) <= tol, (
        f"F_B must vanish at both walls for energy closure; got {F_B[0]:.3e}, {F_B[-1]:.3e}")
    F_B[0] = 0.0
    F_B[-1] = 0.0
    return B.theta_deg.copy(), F_B
