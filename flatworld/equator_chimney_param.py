"""flatworld.equator_chimney_param -- the approach-C sub-grid convective-chimney PARAMETERIZATION
(Chunk B5c, the gate-(a) / resolution-independence half).

THE ARTIFACT (ROADMAP_equatorial_chimney.md / CHIMNEY_SIM_DESIGN.md s4). The global AD core cannot resolve
the equatorial convergence: the ascent collapses to exactly ONE grid cell wide at every resolution, so the
equatorial precip peak DIVERGES as ``P_peak ~ 1/dx`` (a grid artifact -- e.g. the canonical bl_dz20_i100
fixture spikes to 362 mm/day in the innermost cell, then 0). The INTEGRAL under that spike,
``M_tot = int P_spike dx``, is finite and resolution-independent -- it is the physical total moisture
convergence at the chimney. A faithful global model should return that finite, resolution-independent
grid-mean to the grid, NOT the 1/dx spike.

WHAT THIS MODULE DOES (the parameterization, offline). Given a raw cross-track precip profile ``P_raw(x)``
that contains the spike, it
  1. DETECTS the convergence zone (the innermost cells, ``|x| < x_detect``);
  2. INTEGRATES the finite, resolution-independent total ``M_tot = int_zone P_raw dx`` [mm/day.m];
  3. PARTITIONS it by the lofting fraction ``phi`` (the SWEPT knob): a surface-reaching ``(1-phi)*M_tot``
     and a lofted/exported ``phi*M_tot`` (detrained aloft -> surface-vapor export to the equatorial /
     ribbon pathway, NOT precipitated in place);
  4. REDISTRIBUTES the surface-reaching part over a FIXED PHYSICAL footprint ``x_width`` (a normalized
     ``sech^2``/Gaussian shape, ``int g dx = 1``), giving a grid-mean rain ``P_param(x)`` whose peak scales
     ``~ 1/x_width`` (FIXED) -- NOT ``1/dx``. That is the divergence killer;
  5. RETURNS the grid-mean latent-heating density ``H(x)`` from the condensation (``L_VAP*rho_w*M_tot``
     spread over the footprint) -- also resolution-independent.

GATE (a) -- RESOLUTION-INDEPENDENCE (the validation this session delivers). ``resolution_independence_demo``
builds the same physical spike at several ``dx``, parameterizes each, and shows the raw peak diverges
(``~1/dx``) while ``M_tot`` and the parameterized peak/profile stay PUT. That is gate (a) of the roadmap's
revised C gate (PR #166): the grid-mean equatorial rain/heating is resolution-independent.

TWO WALLS (why this is the gate-(a) HALF, not the full calibrated C; Jake 2026-06-19).
  * NECESSITY (Q2) is UNMEASURABLE. Whether the divergent spike actually contaminates the CONVERGED ribbon
    climate is not measurable in feasible time (the equator->ribbon transit is ~tens of true-days, the band
    metric is noisy, a 1-D column can't do loft-to-flanks -- PR #166 retired that gate). The spike is also
    already gated OUT of the geology climatology (``_build_bl_precip_clim.py``). So C's necessity stays
    unestablished; this module is the resolution-independent FIX, not a proof it is needed.
  * phi-CALIBRATION (gate b) is BLOCKED. The true lofting fraction ``phi`` (local-rain vs export) needs the
    B4 chimney re-run with a NON-REFLECTING lateral boundary -- the current closed box (``Psi_max=0`` =>
    ``psi=0`` on every wall) forces export=0 and box-clamps the annulus radius (PR #173). So ``phi`` is the
    SWEPT knob here, never pinned; calibrating it is a B4-BC-upgrade chunk. Estimates to BRACKET, not fit:
    B4's ~40 % lofted-condensate shoulder; EQUATOR_POND's phi~0.4 % open-water axis; A's ask "pin phi".

THE phi HOOK to B5a / B5b. The surface-reaching ``(1-phi)*M_tot`` IS the sustained supply the rest of the
chimney chain consumes: ``F_net_surface`` feeds B5b's basin EXTENT (``equator_basin.solve_basin`` F_net) and
the grid-mean ``P_param`` feeds B5a's ``StageADConfig.budget_wet_fraction`` surface supply ``F``. So ``phi``
is the ONE swept knob shared across approach-C's A (surface), B (dynamics), C (global) -- ``phi_sweep`` maps
it for the downstream consumers.

SCOPE. OFFLINE + tested + a gate-(a) demo, mirroring ``equator_basin.py`` / ``salt_sill.py``. The ONLINE
AD-core wiring (replace the divergent ``self._precip`` spike with ``P_param`` each step, behind a default-OFF
``chimney_param`` flag) is STAGED, not done here (it is a global-run change -> B5d); ``stage_ad_core.py`` is
left byte-identical. The committed fixture ``data/bl_dz20_profin.npz`` (the canonical bl_dz20_i100 spike)
is the real-world input.

CANONICAL WORLD: run with ``COSMO_L_SUN=8e7`` in the SHELL (never module-top -- the import-timing forcing
trap; CLAUDE.md). The resolution-independence and the footprint ``x_width`` are PHYSICAL (metres), so the
gate-(a) result is L-INVARIANT in structure; the tests assert that structure, not latitudes.

CLI:  COSMO_L_SUN=8.0e7 ../.venv/bin/python -m flatworld.equator_chimney_param   (gate-(a) demo + fixture + phi sweep)
"""
from __future__ import annotations
import os
import numpy as np
from dataclasses import dataclass, field

from . import constants as C, geometry as g

_trapz = getattr(np, "trapezoid", None) or np.trapz   # np.trapz removed in numpy 2.0

# ---- soft anchors (documented, swept-to-report -- NOT fitted; cosmology_models/CLAUDE.md "Working stance") ----
LOFT_FRACTION_PHI = 0.02    # the lofting fraction phi: condensate exported aloft OUT OF THE LOCAL CONVERGENCE
#                             ZONE vs reaching the SURFACE within it. CALIBRATED by B5c gate-(b)
#                             (CHIMNEY_PHI_CALIBRATION_RESULTS.md) from the B4-1 open-edge chimney's CONSERVED
#                             condensate export, EQUILIBRATED and EDGE-AWARE (PR #187 review):
#                             (1) phi rides a SPIN-UP TRANSIENT (~0.21) for ~4 h then PLATEAUS -- measure
#                             after 6 h spin-up, NOT in the transient (the earlier 0.21 was the transient);
#                             (2) phi falls STEEPLY with measurement radius (equilibrated: 0.071/0.039/0.016
#                             at X=300/600/1200 km, ~x0.6 per radius-doubling, STILL DESCENDING at the cut) --
#                             most condensate that leaves 300 km rains again farther out. phi is measured at
#                             ~the M_tot detect radius (DETECT_X_M=1500 km) ~= 0.02, NOT the 0.07 at the 300 km
#                             footprint. NB this is "leaves the LOCAL zone aloft": the far ribbon is at
#                             x=L*tan(45-54deg) ~ 80,000-100,000 km (~65x the cut), where the extrapolated phi
#                             ~ 0 -- essentially NONE survives aloft to the far ribbon (it joins the large-scale
#                             pathway; the ribbon is fed by the broad circulation, not ballistic chimney loft). So
#                             F_surface=(1-phi)*M_tot ~= 0.98*M_tot: the equatorial chimney is STRONGLY
#                             LOCAL-RAIN-dominated. Soft walls: edge-placement/measurement-radius (DOMINANT,
#                             an order of magnitude), slow plateau drift + mild dx (plateau ~0.05-0.08 at the
#                             300 km footprint across nx=96/192), convective closure (footprint 0.07-0.13).
#                             Equilibrated pulsation noise is small (~0.01, plateau-window scatter). phi stays
#                             SWEPT downstream (phi_sweep); B5b's regime is phi-INSENSITIVE (endorheic for all
#                             phi), so this firms the F_surface MAGNITUDE (~=M_tot) for B5a/B5d, not the geography.
CHIMNEY_X_WIDTH_M = 3.0e5   # the chimney rain FOOTPRINT half-width [m] over which the finite total is spread
#                             (the same convergence footprint equator_basin uses; B4 keeps P>E out to ~300 km).
#                             FIXED in metres -- this is WHY the grid-mean is resolution-independent (peak ~1/W,
#                             not 1/dx). A soft knob (sweep it); the resolution-INDEPENDENCE holds for any fixed W.
DETECT_X_M = 1.5e6          # the convergence-detection radius [m]: cells with |x| < x_detect are the chimney
#                             zone whose spike is integrated + replaced. Wide enough to capture the 1/dx spike
#                             (one cell at any feasible dx) but inside the ribbon (x(theta=45)=L at canonical L).
#                             Soft; the integral M_tot is insensitive to it as long as it brackets the spike.


@dataclass
class ChimneyParamResult:
    # grids: axis (theta~0, hot) -> poleward; theta INCREASING, x increasing.
    theta_deg: np.ndarray          # cross-track latitude [deg]
    x_m: np.ndarray                # cross-track distance x = L*tan(theta) [m]
    P_raw_mm_day: np.ndarray       # the input precip (with the 1/dx convergence spike) [mm/day]
    P_param_mm_day: np.ndarray     # the parameterized grid-mean SURFACE rain (finite, resolution-indep.) [mm/day]
    H_param_W_m2: np.ndarray       # the grid-mean latent-HEATING density from condensation [W/m^2]
    # the resolution-independent ledger
    M_tot_mm_day_m: float          # int P_spike dx over the convergence zone [mm/day.m] -- resolution-INVARIANT
    M_tot_m2s: float               # ... as a water volume flux per along-track width [m^2/s]
    F_net_surface_m2s: float       # (1-phi)*M_tot -- the surface-reaching SUPPLY fed to B5a/B5b [m^2/s/width]
    F_export_m2s: float            # phi*M_tot -- lofted/exported aloft to the equatorial/ribbon pathway [m^2/s/width]
    # diagnostics
    phi: float                     # the lofting fraction used (SWEPT)
    x_width_m: float               # the footprint half-width [m]
    x_detect_m: float              # the convergence-detection radius [m]
    peak_raw_mm_day: float         # max of P_raw in the zone (DIVERGES ~1/dx -- the artifact)
    peak_param_mm_day: float       # max of P_param (BOUNDED ~1/x_width -- the fix)
    n_zone: int                    # number of grid cells flagged as the convergence zone
    meta: dict


# ---- the footprint shape ---------------------------------------------------------------------------------
def convergence_shape(x, *, x_width=CHIMNEY_X_WIDTH_M, x0=0.0, shape="sech2"):
    """A normalized cross-track footprint ``g(x)`` for the chimney rain, ``int g dx = 1`` (axis-peaked,
    ~0 on the flanks). ``x_width`` is the half-width [m] (FIXED in metres -> the redistributed peak scales
    ~1/x_width, the resolution-independence mechanism). ``shape`` in {"sech2", "gauss"}."""
    x = np.asarray(x, dtype=float)
    u = (x - x0) / x_width
    if shape == "gauss":
        raw = np.exp(-0.5 * u * u)
    elif shape == "sech2":
        # sech^2 with a clipped argument to avoid cosh overflow on the far flanks (-> 0 there anyway).
        raw = 1.0 / np.cosh(np.clip(u, -40.0, 40.0)) ** 2
    else:
        raise ValueError(f"unknown shape {shape!r} (use 'sech2' or 'gauss')")
    area = _trapz(raw, x)
    if not np.isfinite(area) or area <= 0:
        raise ValueError("convergence_shape: degenerate normalization (check x grid / x_width)")
    return raw / area


# ---- detect + integrate the convergence ------------------------------------------------------------------
_MM_DAY_TO_M_S = 1.0e-3 / 86400.0                   # 1 mm/day = 1.157e-8 m/s


def cell_widths(x, *, x_inner=0.0):
    """Finite-VOLUME cell widths ``Δx_i`` [m] for a CELL-CENTERED cross-track grid ``x`` (the convention the
    global core uses: ``x[0] = Δθ/2`` off the axis, never the axis node itself). Cell faces sit at the
    midpoints between centers; the INNERMOST face is clamped to ``x_inner`` (= 0, the equatorial symmetry
    axis), so the axis cell that holds the convergence spike is counted in FULL. This is why a node-based
    trapezoid is WRONG here -- it drops the half-cell ``[0, x[0]]`` (the chimney core) and half-weights the
    spike cell, undercounting ``M_tot`` ~1.8x (PR #179 review). The FV sum ``Σ P_i·Δx_i`` is the
    conservation-consistent integral the model's water budget actually uses."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 1:
        return np.array([2.0 * (x[0] - x_inner)])
    faces = np.empty(n + 1)
    faces[1:-1] = 0.5 * (x[:-1] + x[1:])
    faces[0] = x_inner                                  # the symmetry axis (x=0): count the spike cell fully
    faces[-1] = x[-1] + 0.5 * (x[-1] - x[-2])           # outer half-cell by extrapolation
    return np.diff(faces)


def detect_convergence(x, P, *, x_detect=DETECT_X_M):
    """Boolean mask of the equatorial convergence zone: cells with ``|x| < x_detect`` (the 1/dx spike sits in
    the innermost cell at any feasible dx). Returns the mask. Deliberately geometric, NOT a P-threshold, so
    it is resolution- and amplitude-independent (the same physical zone at every dx)."""
    x = np.asarray(x, dtype=float)
    return np.abs(x) < float(x_detect)


def integrate_convergence(x, P, mask):
    """The resolution-INVARIANT total moisture convergence ``M_tot = Σ_zone P_i·Δx_i`` [mm/day.m] and as a
    volume flux per along-track width [m^2/s], by the FINITE-VOLUME cell-width sum (``cell_widths``) -- the
    conservation-consistent quadrature for the cell-centered grid (NOT a node-based trapezoid, which would
    undercount the boundary spike cell ~1.8x). The sum is over the flagged zone only; it is finite and
    dx-independent even as the raw PEAK diverges ~1/dx."""
    x = np.asarray(x, dtype=float)
    P = np.asarray(P, dtype=float)
    if mask.sum() < 1:
        return 0.0, 0.0
    w = cell_widths(x)
    M_mm_day_m = float(np.sum(np.where(mask, P, 0.0) * w))
    M_m2s = M_mm_day_m * _MM_DAY_TO_M_S            # mm/day.m -> m/s.m = m^2/s
    return M_mm_day_m, M_m2s


# ---- the parameterization --------------------------------------------------------------------------------
def parameterize(theta_deg, P_raw, *, phi=LOFT_FRACTION_PHI, x_width=CHIMNEY_X_WIDTH_M,
                 x_detect=DETECT_X_M, shape="sech2"):
    """Replace the 1/dx convergence spike in ``P_raw(theta)`` with the finite, resolution-independent
    grid-mean chimney rain + latent heating. Returns a ``ChimneyParamResult``.

    The non-convergence precip (the ribbon band, |x| >= x_detect) is passed through UNCHANGED -- the
    parameterization touches ONLY the chimney zone.
    """
    theta_deg = np.asarray(theta_deg, dtype=float)
    P_raw = np.asarray(P_raw, dtype=float)
    if theta_deg.shape != P_raw.shape:
        raise ValueError("theta_deg and P_raw must share shape")
    phi = float(np.clip(phi, 0.0, 1.0))
    x = g.x_of_theta(np.radians(theta_deg))

    mask = detect_convergence(x, P_raw, x_detect=x_detect)
    M_mm_day_m, M_m2s = integrate_convergence(x, P_raw, mask)

    # surface-reaching vs lofted partition (the swept phi)
    F_surface = (1.0 - phi) * M_m2s
    F_export = phi * M_m2s
    M_surface_mm_day_m = (1.0 - phi) * M_mm_day_m

    # redistribute the surface part over the FIXED footprint -> finite grid-mean rain [mm/day]. Normalize the
    # shape over the RETAINED (masked) cells via the FV cell-width sum, so Σ(P_chimney·Δx) = M_surface exactly
    # (conservation on exactly the cells the spike is replaced in -- the sech^2 tail past x_detect is folded in).
    shp = convergence_shape(x, x_width=x_width, x0=0.0, shape=shape)   # axis-peaked analytic footprint [1/m]
    w = cell_widths(x)
    shp_z = np.where(mask, shp, 0.0)
    znorm = np.sum(shp_z * w)
    shp_z = shp_z / znorm if znorm > 0 else shp_z                     # Σ(shp_z·Δx) = 1 over the mask
    P_chimney = M_surface_mm_day_m * shp_z                            # mm/day.m * 1/m = mm/day

    # splice: ribbon band untouched, chimney zone replaced by the finite grid-mean
    P_param = np.where(mask, P_chimney, P_raw)

    # latent heating from the FULL converged moisture (all condenses, phi-independent); same zone-normalized
    # footprint so Σ(H·Δx) = L_VAP*rho*M_tot exactly over the chimney column
    H_param = (C.L_VAP * C.RHO_LIQ * M_m2s) * shp_z                   # W/m^2 grid-mean density (0 outside mask)

    peak_raw = float(np.max(P_raw[mask])) if mask.any() else 0.0
    peak_param = float(np.max(P_param[mask])) if mask.any() else 0.0

    return ChimneyParamResult(
        theta_deg=theta_deg, x_m=x, P_raw_mm_day=P_raw, P_param_mm_day=P_param, H_param_W_m2=H_param,
        M_tot_mm_day_m=M_mm_day_m, M_tot_m2s=M_m2s, F_net_surface_m2s=F_surface, F_export_m2s=F_export,
        phi=phi, x_width_m=float(x_width), x_detect_m=float(x_detect),
        peak_raw_mm_day=peak_raw, peak_param_mm_day=peak_param, n_zone=int(mask.sum()),
        meta=dict(shape=shape, L_SUN=C.L_SUN, mm_day_to_m_s=_MM_DAY_TO_M_S),
    )


# ---- online (per-substep) fast path: precompute the static footprint once, redistribute each step --------
def precompute_footprint(theta_deg, *, x_width=CHIMNEY_X_WIDTH_M, x_detect=DETECT_X_M, shape="sech2"):
    """Precompute the STATIC footprint structure for the online (per-substep) chimney parameterization (the
    AD core's ``chimney_param`` path): the convergence ``mask``, the FV cell widths ``w``, and the
    zone-normalized footprint ``shp_z`` (Σ shp_z·Δx = 1 over the mask). All depend only on the FIXED θ grid,
    so this is computed ONCE per run and reused every substep by ``apply_cached`` -- avoiding the per-step
    cosh/cell-width/normalization rebuild (PR #188 review #6; matters at the high-nx GPU run). Returns
    ``(mask, w, shp_z)``."""
    theta_deg = np.asarray(theta_deg, dtype=float)
    x = g.x_of_theta(np.radians(theta_deg))
    mask = detect_convergence(x, None, x_detect=x_detect)
    w = cell_widths(x)
    shp = convergence_shape(x, x_width=x_width, x0=0.0, shape=shape)
    shp_z = np.where(mask, shp, 0.0)
    znorm = float(np.sum(shp_z * w))
    if znorm > 0:
        shp_z = shp_z / znorm
    return mask, w, shp_z


def apply_cached(P_raw_mm_day, mask, w, shp_z, phi):
    """The per-substep fast path: return ``P_param`` [mm/day] = ``P_raw`` with ONLY the masked (chimney-zone)
    cells replaced by the ``(1-phi)*M_tot`` grid-mean over the cached footprint ``shp_z`` (``M_tot`` = the FV
    cell-width sum over the zone). Out-of-zone cells are a verbatim copy (bit-exact -- PR #188 review #5).
    Equivalent to ``parameterize(...).P_param_mm_day`` but without rebuilding the static structure. The
    caller writes back only ``P_param[mask]`` to keep the ribbon byte-identical."""
    P = np.asarray(P_raw_mm_day, dtype=float)
    phi = float(np.clip(phi, 0.0, 1.0))
    M_mm_day_m = float(np.sum(np.where(mask, P, 0.0) * w))
    out = P.copy()
    out[mask] = ((1.0 - phi) * M_mm_day_m) * shp_z[mask]
    return out


# ---- gate (a): resolution-independence demo --------------------------------------------------------------
def _synthetic_spike(nx, *, M_target_mm_day_m=4.0e7, theta_max_deg=60.0, ribbon_mm_day=4.0,
                     ribbon_theta_deg=50.0, ribbon_width_deg=3.0):
    """A synthetic raw profile on ``nx`` cells: a ONE-CELL convergence spike at the axis whose amplitude is
    set so the FV integral ``Σ P_i·Δx_i = M_target`` EXACTLY at every nx (the physical total is fixed; the
    peak then scales ~1/Δx -- the artifact), plus a fixed broad ribbon band. The peak DIVERGES as nx grows;
    the integral does NOT. This is the controlled input for gate (a). The amplitude uses the SAME FV cell
    width ``cell_widths`` that ``integrate_convergence`` sums, so the injection and the measurement agree
    exactly (no quadrature mismatch)."""
    theta = np.linspace(0.0, theta_max_deg, nx)
    x = g.x_of_theta(np.radians(theta))
    P = np.zeros(nx)
    # one-cell spike at the axis: amplitude * (FV cell width of cell 0) = M_target
    P[0] = M_target_mm_day_m / cell_widths(x)[0]
    # a broad, resolution-independent ribbon band (Gaussian in theta) -- the part that should pass through
    P += ribbon_mm_day * np.exp(-0.5 * ((theta - ribbon_theta_deg) / ribbon_width_deg) ** 2)
    return theta, P


def resolution_independence_demo(*, nx_list=(240, 480, 960, 1920), phi=LOFT_FRACTION_PHI,
                                 x_width=CHIMNEY_X_WIDTH_M, M_target_mm_day_m=4.0e7, shape="sech2"):
    """GATE (a). Build the SAME physical spike at several ``nx`` (so the raw peak ~1/dx), parameterize each,
    and return a table: ``nx, dx_axis_km, peak_raw, M_tot, peak_param, F_surface``. The PASS criterion is
    that ``M_tot`` (the exactly resolution-INVARIANT physical total) and the grid-mean ``peak_param`` are
    nx-INVARIANT while ``peak_raw`` grows ~1/dx.

    FOOTPRINT-RESOLUTION FLOOR: the grid-mean peak is invariant only once the grid RESOLVES the footprint
    (``dx <~ x_width``); on a coarser grid a fixed-metre footprint cannot be represented. This is a MILD,
    physical requirement -- far weaker than resolving the chimney ascent itself (``dx < 10 km``); the default
    ``nx_list`` keeps ``dx_axis <~ x_width`` (the regime the real global runs, nx250+, operate in)."""
    rows = []
    for nx in nx_list:
        theta, P = _synthetic_spike(nx, M_target_mm_day_m=M_target_mm_day_m)
        x = g.x_of_theta(np.radians(theta))
        r = parameterize(theta, P, phi=phi, x_width=x_width, shape=shape)
        rows.append(dict(nx=nx, dx_axis_km=(x[1] - x[0]) / 1e3, peak_raw=r.peak_raw_mm_day,
                         M_tot=r.M_tot_mm_day_m, peak_param=r.peak_param_mm_day,
                         F_surface_m2s=r.F_net_surface_m2s, n_zone=r.n_zone))
    return rows


def resolution_independence_verdict(rows, *, rtol=0.02):
    """Reduce a demo table to the gate-(a) verdict. PASS = (1) ``M_tot`` is nx-invariant to ``rtol`` -- the
    exactly resolution-independent physical total; (2) the grid-mean ``peak_param`` is nx-invariant to
    ``rtol`` (over the footprint-resolved rows); and (3) ``peak_raw`` DIVERGES (max/min >> 1) -- the artifact
    the parameterization removes. Returns the spreads + a boolean ``passed``."""
    M = np.array([r["M_tot"] for r in rows])
    pp = np.array([r["peak_param"] for r in rows])
    pr = np.array([r["peak_raw"] for r in rows])
    M_spread = float((M.max() - M.min()) / max(abs(M.mean()), 1e-30))
    pp_spread = float((pp.max() - pp.min()) / max(abs(pp.mean()), 1e-30))
    raw_ratio = float(pr.max() / max(pr.min(), 1e-30))    # >> 1 means the raw peak diverged
    passed = (M_spread < rtol) and (pp_spread < rtol) and (raw_ratio > 2.0)
    return dict(M_spread=M_spread, peak_param_spread=pp_spread, raw_peak_ratio=raw_ratio, passed=passed)


# ---- the phi hook for B5a / B5b --------------------------------------------------------------------------
def phi_sweep(theta_deg, P_raw, *, phi_grid=(0.0, 0.1, 0.25, 0.4, 0.5, 0.75, 0.9, 0.99),
              x_width=CHIMNEY_X_WIDTH_M, x_detect=DETECT_X_M, shape="sech2"):
    """Map the surface-reaching supply across the swept ``phi`` -- the hand-off to B5b's basin EXTENT
    (``F_net_surface``) and B5a's surface wetting (``peak_param``). Returns a list of rows."""
    rows = []
    for phi in phi_grid:
        r = parameterize(theta_deg, P_raw, phi=phi, x_width=x_width, x_detect=x_detect, shape=shape)
        rows.append(dict(phi=phi, F_surface_m2s=r.F_net_surface_m2s, F_export_m2s=r.F_export_m2s,
                         peak_param_mm_day=r.peak_param_mm_day))
    return rows


# ---- real-world input (the committed fixture) ------------------------------------------------------------
def from_fixture(path="data/bl_dz20_profin.npz", *, phi=LOFT_FRACTION_PHI, x_width=CHIMNEY_X_WIDTH_M,
                 x_detect=DETECT_X_M, shape="sech2"):
    """Parameterize the REAL canonical-world spike (the committed bl_dz20_i100 fixture: theta_deg,
    precip_mm_day). Returns the ``ChimneyParamResult``. The fixture lives under ``flatworld/data/`` -- resolve
    relative to this module so the CLI works from any cwd."""
    here = os.path.dirname(os.path.abspath(__file__))
    full = path if os.path.isabs(path) else os.path.join(here, path)
    d = np.load(full)
    return parameterize(d["theta_deg"], d["precip_mm_day"], phi=phi, x_width=x_width,
                        x_detect=x_detect, shape=shape)


# ---- report ----------------------------------------------------------------------------------------------
def _report():
    L_km = C.L_SUN / 1e3
    print(f"\n=== equator_chimney_param (B5c, gate-(a) resolution-independence) | L_SUN = {L_km:,.0f} km ===")
    if abs(C.L_SUN - 8.0e7) > 1e-3:
        print("  [warn] COSMO_L_SUN != 8e7 -- not the canonical world; set it in the SHELL for canonical numbers.")

    print("\n-- GATE (a): resolution-independence of the grid-mean chimney rain --")
    rows = resolution_independence_demo()
    print(f"  {'nx':>5} {'dx_axis[km]':>11} {'peak_raw':>11} {'M_tot[mm/d.m]':>15} {'peak_param':>11} {'n_zone':>7}")
    for r in rows:
        print(f"  {r['nx']:>5d} {r['dx_axis_km']:>11.1f} {r['peak_raw']:>11.1f} {r['M_tot']:>15.3e}"
              f" {r['peak_param']:>11.3f} {r['n_zone']:>7d}")
    v = resolution_independence_verdict(rows)
    print(f"  verdict: M_tot spread {v['M_spread']:.2e}, peak_param spread {v['peak_param_spread']:.2e}, "
          f"raw-peak ratio {v['raw_peak_ratio']:.1f}x  -> GATE (a) {'PASS' if v['passed'] else 'FAIL'}")
    print("  READ: the raw peak DIVERGES ~1/dx; M_tot + the parameterized peak stay PUT -> divergence killed.")

    print("\n-- the real canonical spike (bl_dz20_i100 fixture), phi swept --")
    r0 = from_fixture(phi=LOFT_FRACTION_PHI)
    print(f"  raw spike peak {r0.peak_raw_mm_day:.1f} mm/day in {r0.n_zone} cell(s); "
          f"M_tot {r0.M_tot_mm_day_m:.3e} mm/day.m = {r0.M_tot_m2s:.3e} m^2/s")
    print(f"  -> parameterized grid-mean peak {r0.peak_param_mm_day:.2f} mm/day (phi={r0.phi:g}), "
          f"footprint half-width {r0.x_width_m/1e3:.0f} km")
    print(f"  {'phi':>5} {'F_surface[m2/s]':>16} {'F_export[m2/s]':>15} {'peak_param[mm/d]':>17}")
    for s in phi_sweep(r0.theta_deg, r0.P_raw_mm_day):
        print(f"  {s['phi']:>5.2f} {s['F_surface_m2s']:>16.3e} {s['F_export_m2s']:>15.3e}"
              f" {s['peak_param_mm_day']:>17.3f}")
    print("  READ: (1-phi)*M_tot = F_surface feeds B5b's basin EXTENT / B5a's surface wetting; phi SWEPT")
    print("        (gate-b calibration blocked on the B4-BC upgrade).")
    print()


if __name__ == "__main__":
    _report()
