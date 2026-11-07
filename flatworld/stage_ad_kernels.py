"""
flatworld.stage_ad_kernels -- the AD core's module-level numerical kernels (#289 Part 2 split).

The shared dynamical/numerical kernels of the fused anelastic core, MOVED VERBATIM out of
``stage_ad_core.py`` (issue #289 Part 2 -- a move-only split; zero token changes inside any def or
class body): the BL-refined stretched vertical grid, the finite-difference stencils (derivative,
upwind, divergence-of-gradient forms), the anelastic streamfunction Poisson builder and the
``SeparablePoisson`` DST fast solver, velocity recovery from psi, the Smagorinsky closures, the
vorticity tendency, the tridiagonal column solver, hydrostatic layer heights, and the
``_interp_columns`` family with its lazy numba builders (the ``_INTERP_NUMBA`` cache lives HERE).

These are MODULE-LEVEL pure functions (grid spacing passed in, no ``self``) for two reasons:
``StageADCore`` and the Rayleigh-Benard benchmark (``stage_ad_core.rayleigh_benard_run``) call the
SAME code, so the benchmark validates the very solver the core uses (the tier-3 anchor); and they
are trivially unit-testable. Unless a docstring says otherwise, arrays are on the (nz, nx) HEIGHT
grid, B-convention: FIRST axis vertical, SURFACE(0) -> TOP(nz-1).

Back-compat: ``stage_ad_core`` re-exports every name here (public and underscore), so existing
``from flatworld.stage_ad_core import X`` / ``stage_ad_core.X`` references keep working; new code
should import from this module.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.fft import dst, idst, set_workers
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq
from . import constants as C
from .device import get_xp, is_device


# ======================================================================================
# BL-REFINED STRETCHED VERTICAL GRID (BL_REFINEMENT_SPEC.md): the OPTIONAL fine-near-surface height grid
# that replaces the uniform ``linspace(0, z_top, nz)`` when ``grid_type="stretched"`` -- so a ~10-30 m
# lowest level resolves the boundary-layer jet instead of log-extrapolating the 10 m wind from ~400 m+.
# Default OFF (uniform), so this is never called on the byte-identical path.
# ======================================================================================
def _stretched_z(z_top, nz, dz_min):
    """A monotone sinh-stretched height grid ``z[0]=0 .. z[nz-1]=z_top``, fine near the surface, whose
    lowest spacing ``z[1]-z[0]`` equals ``dz_min`` exactly.

        z(u) = z_top * sinh(s*u) / sinh(s) ,   u = linspace(0, 1, nz)

    The single stretch parameter ``s`` is solved (Brentq) so ``z[1]-z[0] = dz_min``: as ``s -> 0`` the
    grid is uniform (``z[1]-z[0] -> z_top/(nz-1)``) and as ``s`` grows the lowest spacing shrinks
    monotonically, so a unique ``s>0`` exists iff ``0 < dz_min < z_top/(nz-1)`` -- the construction guard.
    This is the SAME sinh family as the along-track (``geometry.py``) and the day-mean-flux quadrature
    grids, but with a solve for ``s`` (they use a fixed stretch), so it is not a drop-in reuse of either."""
    z_top = float(z_top); dz_min = float(dz_min); nz = int(nz)
    uniform_dz = z_top / (nz - 1)
    if not (0.0 < dz_min < uniform_dz):
        raise ValueError(f"dz_min={dz_min:g} must satisfy 0 < dz_min < z_top/(nz-1)={uniform_dz:g} "
                         f"for a stretched grid (dz_min >= uniform spacing leaves nothing to refine)")
    u = np.linspace(0.0, 1.0, nz)
    u1 = u[1]                                                  # = 1/(nz-1)

    def lowest_dz(s):                                          # z[1]-z[0] as a function of the stretch s
        return z_top * np.sinh(s * u1) / np.sinh(s)

    # lowest_dz is monotone DECREASING in s, from z_top*u1 (=uniform_dz, at s->0) toward 0 (large s).
    # Bracket: s_lo small (lowest_dz ~ uniform_dz > dz_min); grow s_hi until lowest_dz < dz_min.
    s_lo, s_hi = 1e-6, 1.0
    while lowest_dz(s_hi) > dz_min:
        s_hi *= 2.0
        if s_hi > 1e6:                                         # unreachable for a valid dz_min; guard anyway
            raise ValueError(f"could not bracket the stretch for dz_min={dz_min:g}")
    s = brentq(lambda s: lowest_dz(s) - dz_min, s_lo, s_hi, xtol=1e-12, rtol=1e-14)
    z = z_top * np.sinh(s * u) / np.sinh(s)
    z[0] = 0.0; z[-1] = z_top                                  # pin the endpoints exactly (kill round-off)
    return z


# ======================================================================================
# STRETCHED CROSS-TRACK (x) GRID -- the chimney-nest REPLACEMENT program (2026-07-10): resolve the
# equatorial updraft INSIDE the core (dx ~ km at the axis) instead of nesting a separate chimney model,
# so the axis columns get the core's real radiation/SEB/surface physics and the nest's ill-posed
# coupling quantities (phi, M_cond, the sech^2 footprint, Psi_max, the held surface air) cease to
# exist. Default OFF (``x_grid_type="uniform"``) -> never called on the byte-identical path.
# NOT the sinh family of ``_stretched_z``: a single-knob sinh cannot hold BOTH a ~km axis spacing AND
# the current far-field spacing on a 3e8 m domain (solving it for dx_min at the axis coarsens the far
# field several-fold). The capped-geometric family below holds both ends by construction.
# ======================================================================================
def _stretched_x_edges(x_max, dx_min, dx_max, ratio=1.05, n_fine=24,
                       tail_from=None, tail_dx_max=None, tail_ratio=None):
    """Capped-geometric cross-track cell EDGES [m], fine at the equatorial axis:
    ``edges[0] = 0.0 .. edges[-1] = x_max`` (both pinned exactly), widths ``np.diff(edges)``.

    Three blocks, axis outward:
      1. FINE:       ``n_fine`` cells of width exactly ``dx_min`` against the axis (x=0);
      2. TRANSITION: widths growing geometrically at ~``ratio`` per cell, strictly within
                     (``dx_min``, ``dx_max``];
      3. PLATEAU:    cells of width ``dx_max`` out to ``x_max``.

    THE PLATEAU IS PINNED TO THE OUTER EDGE: plateau edges sit at ``x_max - k*dx_max`` EXACTLY (built
    from ``x_max`` inward), so every refinement rung (any ``dx_min``/``ratio``/``n_fine`` at fixed
    ``dx_max``/``x_max``) shares its far-field cell edges bit-for-bit with every other rung -- the
    dx_min convergence ladder's far fields are then identical by construction, not by tolerance.

    THE STITCH (the one solved parameter, mirroring ``_stretched_z``'s Brentq-for-exactness): a
    geometric transition does not generally end on a plateau edge, so the realized growth ratio ``r``
    is re-solved near the requested ``ratio`` -- adding transition cells if needed to keep every width
    <= ``dx_max`` -- such that the transition spans EXACTLY from the fine block's outer edge to the
    plateau edge ``x_max - n_plateau*dx_max``. |r - ratio| is a few percent of (ratio - 1) at
    production scales (the span adjustment is < one ``dx_max`` over a ~100-cell transition).

    The growth RATE has C^0 corners at the two block boundaries (rate steps 1 -> r -> 1). At
    ratio <= ~1.1 the step is a few percent per cell; an explicitly C^1 (smoothstep-in-ln-width)
    profile was evaluated and deferred -- it costs ~50% more transition cells, and the validation
    campaign's mild-refinement control rung tests for transition-zone artifacts empirically.

    OUTWARD-COARSENING TAIL (the box-extension program, 2026-07-10): with ``tail_from`` set, the
    three-block grid above is built on [0, tail_from] EXACTLY as if ``tail_from`` were the domain
    end (bitwise -- a tailed grid's inner region IS the untailed grid), and the domain then
    continues to ``x_max`` with widths growing geometrically from ``dx_max`` at ``tail_ratio``
    (default ``ratio``) per cell up to the ``tail_dx_max`` cap, then a coarse plateau of exact
    ``tail_dx_max`` cells. The tail is pinned INWARD (edges laid outward from ``tail_from``), so
    grids differing only in ``x_max`` -- the box-test ladder -- share every tail edge bitwise
    except the outermost; the misfit is absorbed by the OUTERMOST cell (width in
    [tail_dx_max, 2*tail_dx_max)), the deadest cell in the domain. Motivation: day-mean forcing is
    ~zero beyond theta ~70 deg yet run1's MATURE cell fills the theta_max=80 box (the outer wall
    BINDS -- the far state is smooth transport-controlled decay, truncated mid-decline), so the
    box must extend; the tail buys a 2-2.5x larger box for ~+21% columns from tail_from = the old
    x_max, or for FEWER columns than the theta80 grid when ``tail_from`` replaces the
    forcing-dead outer half of the old plateau (STRETCHX_CAMPAIGN.md box ladder)."""
    x_max = float(x_max); dx_min = float(dx_min); dx_max = float(dx_max)
    ratio = float(ratio); n_fine = int(n_fine)
    if tail_from is not None:
        # ---- the outward-coarsening tail: inner grid on [0, tail_from] verbatim, then coarsen ----
        tail_from = float(tail_from)
        r_t = float(tail_ratio) if tail_ratio is not None else ratio
        if tail_dx_max is None:
            raise ValueError("tail_from is set but tail_dx_max is not: the tail's coarse cap is a "
                             "physics choice and must be given explicitly")
        tail_dx_max = float(tail_dx_max)
        if not (0.0 < tail_from < x_max):
            raise ValueError(f"tail_from={tail_from:g} m must lie strictly inside (0, x_max={x_max:g} m)")
        if not (tail_dx_max > dx_max):
            raise ValueError(f"tail_dx_max={tail_dx_max:g} m must exceed dx_max={dx_max:g} m "
                             "(the tail COARSENS outward)")
        if not (1.0 < r_t):
            raise ValueError(f"tail_ratio={r_t:g}: the tail growth ratio must exceed 1")
        inner = _stretched_x_edges(tail_from, dx_min, dx_max, ratio=ratio, n_fine=n_fine)
        # Tail transition: widths dx_max * r_t**k, k=1..m, strictly inside (dx_max, tail_dx_max) --
        # the same floor-and-decrement convention as the inner transition. No stitch solve here: the
        # tail plateau starts wherever the transition ends (inner-pinned), the OUTER edge absorbs.
        m = int(np.floor(np.log(tail_dx_max / dx_max) / np.log(r_t)))
        if m > 0 and dx_max * r_t**m >= tail_dx_max:           # floor landed exactly on the cap
            m -= 1
        widths_tt = dx_max * r_t ** np.arange(1, m + 1)
        trans = tail_from + np.cumsum(widths_tt)               # laid OUTWARD from tail_from (inner-pinned)
        E_t = float(trans[-1]) if m > 0 else tail_from         # where the coarse plateau starts
        P = x_max - E_t                                        # the coarse-plateau span
        if P < tail_dx_max:
            raise ValueError(f"tail domain too short: the coarsening transition ends at {E_t:g} m, "
                             f"leaving {P:g} m < one tail_dx_max cell ({tail_dx_max:g} m) before "
                             f"x_max={x_max:g} m -- raise x_max (theta_max), lower tail_dx_max, or "
                             "raise tail_ratio")
        n_tp = int(np.floor(P / tail_dx_max))                  # >= 1 by the check above
        edges = np.concatenate([
            inner,                                             # [0, tail_from], bitwise the untailed grid
            trans,                                             # coarsening transition, inner-pinned
            E_t + tail_dx_max * np.arange(1, n_tp),            # plateau: exact multiples from E_t
            [x_max],                                           # outermost cell absorbs: [tail_dx_max, 2*tail_dx_max)
        ])
        w_tail = np.diff(edges[len(inner) - 1:])
        atol = 8.0 * np.spacing(x_max)
        if not (np.all(w_tail > 0) and np.all(w_tail >= dx_max - atol)
                and np.all(w_tail < 2.0 * tail_dx_max + atol)):
            raise AssertionError("stretched-x tail produced widths outside [dx_max, 2*tail_dx_max) -- bug")
        return edges
    if not (0.0 < dx_min < dx_max):
        raise ValueError(f"need 0 < dx_min < dx_max (got dx_min={dx_min:g}, dx_max={dx_max:g})")
    if not (1.0 < ratio):
        raise ValueError(f"ratio={ratio:g}: the transition growth ratio must exceed 1")
    if n_fine < 1:
        raise ValueError(f"n_fine={n_fine}: need at least one fine cell at the axis")

    L_fine = n_fine * dx_min
    # Nominal transition at the REQUESTED ratio: widths dx_min*ratio**k, k=1..n_t, all < dx_max.
    n_t = int(np.floor(np.log(dx_max / dx_min) / np.log(ratio)))
    if dx_min * ratio**n_t >= dx_max:                          # floor landed exactly on dx_max
        n_t -= 1

    def span(r, n):                                            # transition length: dx_min * sum_{k=1..n} r^k
        return dx_min * r * (r**n - 1.0) / (r - 1.0)

    T_nom = span(ratio, n_t) if n_t > 0 else 0.0
    if L_fine + T_nom + dx_max > x_max:
        raise ValueError(f"domain too small: fine ({L_fine:g} m) + transition ({T_nom:g} m) + one plateau "
                         f"cell ({dx_max:g} m) exceed x_max={x_max:g} m -- shrink n_fine/dx_min or raise ratio")

    # Stitch solve: with n transition cells the realized ratio r must satisfy BOTH width <= dx_max
    # (r <= r_cap(n) = (dx_max/dx_min)**(1/n)) and span(r, n) = T_req. A root exists iff
    # n*dx_min < T_req <= span(r_cap(n), n); search n upward from the nominal n_t. Once n*dx_min
    # reaches T_req no larger n can work either (the lower bound only grows) -- then T_req itself is
    # infeasible under the caps and the transition takes one more plateau cell (T_req += dx_max).
    # Production (ratio 1.05, dx_max/dx_min ~ 200) resolves on the first branch, n = n_t.
    def _solve_stitch(T_req):
        n = max(n_t, 1)
        # Feasibility is tested at the ACTUAL lower bracket span(1+1e-12, n) -- not the n*dx_min proxy --
        # so whenever the loop proceeds, f(a) < 0 <= f(b) and the brentq bracket is valid by construction
        # (review finding 3, PR #316: the proxy left a razor-thin band where brentq would raise instead of
        # cleanly borrowing a plateau cell).
        while span(1.0 + 1e-12, n) < T_req:
            r_cap_n = (dx_max / dx_min) ** (1.0 / n)
            if span(r_cap_n, n) >= T_req:
                return n, brentq(lambda rr: span(rr, n) - T_req, 1.0 + 1e-12, r_cap_n,
                                 xtol=1e-14, rtol=1e-15)
            n += 1
        return None                                            # infeasible at this T_req

    n_plateau = int(np.floor((x_max - L_fine - T_nom) / dx_max))
    while True:
        E = x_max - n_plateau * dx_max                         # the plateau edge the transition lands on
        stitch = _solve_stitch(E - L_fine)
        if stitch is not None:
            n_cells, r = stitch
            break
        n_plateau -= 1                                         # widen the transition by one plateau cell
        if n_plateau < 1:
            raise ValueError("could not stitch the transition to the plateau (domain too small for the "
                             f"requested dx_min={dx_min:g}/dx_max={dx_max:g}/ratio={ratio:g}/n_fine={n_fine})")

    widths_t = dx_min * r ** np.arange(1, n_cells + 1)
    edges = np.concatenate([
        dx_min * np.arange(n_fine + 1),                        # fine block: 0 .. L_fine, exact multiples
        L_fine + np.cumsum(widths_t),                          # transition (last entry ~= E, re-pinned below)
        x_max - dx_max * np.arange(n_plateau - 1, -1, -1),     # plateau: x_max - k*dx_max, EXACT, outer-pinned
    ])
    edges[n_fine + n_cells] = E                                # pin the stitch edge exactly (kill round-off)
    edges[0] = 0.0
    w = np.diff(edges)
    # Width-bound self-check with an ULP-SCALED ABSOLUTE slack (review finding 1, PR #316): the plateau
    # edges are built as x_max - k*dx_max, so consecutive diffs carry up to ~1 ULP(x_max) of round-off --
    # a RELATIVE dx_max tolerance false-trips on perfectly valid grids once x_max/dx_max >~ 4500 (exactly
    # the fine-far-field regime the convergence ladder explores). Sub-micron at production x_max.
    atol = 8.0 * np.spacing(x_max)
    if not (np.all(w > 0) and np.all(w >= dx_min - atol) and np.all(w <= dx_max + atol)):
        raise AssertionError("stretched-x construction produced widths outside [dx_min, dx_max] -- bug")
    return edges


def stretched_x_cell_count(x_max, dx_min, dx_max, ratio=1.05, n_fine=24,
                           tail_from=None, tail_dx_max=None, tail_ratio=None):
    """The ``nx`` a stretched-x grid realizes (``len(edges) - 1``), for drivers that must size arrays
    before constructing the core (checkpoint shapes, sweep bookkeeping). Just builds the edges."""
    return len(_stretched_x_edges(x_max, dx_min, dx_max, ratio=ratio, n_fine=n_fine,
                                  tail_from=tail_from, tail_dx_max=tail_dx_max,
                                  tail_ratio=tail_ratio)) - 1


# ======================================================================================
# THE SHARED DYNAMICAL KERNEL (M2b.2): finite-difference stencils, the anelastic streamfunction
# Poisson, the velocity recovery, the Smagorinsky eddy viscosity, and the vorticity tendency.
#
# These are MODULE-LEVEL pure functions (grid spacing passed in) for two reasons: (1) ``StageADCore``
# and the Rayleigh-Benard benchmark (``rayleigh_benard_run``) call the SAME code, so the benchmark
# validates the very solver the core uses (the tier-3 anchor); (2) they are trivially unit-testable.
# The stencils are Stage B's (``stage_b_overturning._ddx/_ddz/_lap/_upwind``) made dimensional (spacing
# is an argument, not ``self.dxi/dze``). All operate on the (nz, nx) HEIGHT grid, B-convention: the
# FIRST axis is vertical, SURFACE(0) -> TOP(nz-1); the second axis is cross-track x.
# ======================================================================================
def _ddx(F, dx):
    """Centered d/dx (one-sided at the i-edges). (nz, nx).

    ``dx`` is EITHER a SCALAR uniform spacing -- the current verbatim centered/one-sided stencils (the
    byte-identical default path) -- OR the 1-D array of x CENTERS (nx,) for a stretched grid, in which case
    the interior uses the second-order UNEQUAL-INTERVAL 3-point formula and the edges the local one-sided
    differences. With equal intervals the array branch reduces (to round-off) to the scalar branch.

    DEVICE (M3.1): the array module is inferred from ``F`` (``get_xp``), so a numpy field returns numpy
    (bit-identical to pre-M3.1) and a CuPy field stays on the GPU. Same pattern in every kernel below."""
    xp = get_xp(F)
    o = xp.zeros_like(F)
    if np.ndim(dx) == 0:
        o[:, 1:-1] = (F[:, 2:] - F[:, :-2]) / (2 * dx)
        o[:, 0] = (F[:, 1] - F[:, 0]) / dx
        o[:, -1] = (F[:, -1] - F[:, -2]) / dx
        return o
    x = xp.asarray(dx)                                          # x CENTERS (nx,)
    hm = (x[1:-1] - x[:-2])[None, :]                            # h^- = x[i]-x[i-1]  (1, nx-2)
    hp = (x[2:] - x[1:-1])[None, :]                             # h^+ = x[i+1]-x[i]  (1, nx-2)
    o[:, 1:-1] = (-hp / (hm * (hm + hp)) * F[:, :-2]
                  + (hp - hm) / (hm * hp) * F[:, 1:-1]
                  + hm / (hp * (hm + hp)) * F[:, 2:])
    o[:, 0] = (F[:, 1] - F[:, 0]) / (x[1] - x[0])
    o[:, -1] = (F[:, -1] - F[:, -2]) / (x[-1] - x[-2])
    return o


def _ddz(F, dz):
    """Centered d/dz (one-sided at the j-edges). First axis is vertical.

    ``dz`` is EITHER a SCALAR uniform spacing -- the current verbatim centered/one-sided stencils (the
    byte-identical default path) -- OR the 1-D array of z LEVELS (nz,) for a stretched grid, in which case
    the interior uses the second-order UNEQUAL-INTERVAL 3-point formula and the edges the local one-sided
    differences. With equal intervals the array branch reduces (to round-off) to the scalar branch."""
    xp = get_xp(F)
    o = xp.zeros_like(F)
    if np.ndim(dz) == 0:
        o[1:-1, :] = (F[2:, :] - F[:-2, :]) / (2 * dz)
        o[0, :] = (F[1, :] - F[0, :]) / dz
        o[-1, :] = (F[-1, :] - F[-2, :]) / dz
        return o
    z = xp.asarray(dz)                                          # z LEVELS (nz,)
    hm = (z[1:-1] - z[:-2])[:, None]                            # h^- = z[j]-z[j-1]  (nz-2, 1)
    hp = (z[2:] - z[1:-1])[:, None]                             # h^+ = z[j+1]-z[j]  (nz-2, 1)
    o[1:-1, :] = (-hp / (hm * (hm + hp)) * F[:-2, :]
                  + (hp - hm) / (hm * hp) * F[1:-1, :]
                  + hm / (hp * (hm + hp)) * F[2:, :])
    o[0, :] = (F[1, :] - F[0, :]) / (z[1] - z[0])
    o[-1, :] = (F[-1, :] - F[-2, :]) / (z[-1] - z[-2])
    return o


def _upwind(F, u, w, dx, dz):
    """First-order upwind advection u.grad(F) = u dF/dx + w dF/dz (Stage B's scheme, dimensional).
    Backward difference where the velocity is positive, forward where negative.

    ``dx`` scalar -> uniform cross-track spacing (verbatim); ``dx`` the x CENTERS array -> the local
    interval x[i]-x[i-1] (backward) / x[i+1]-x[i] (forward) for the stretched grid. ``dz`` scalar ->
    uniform vertical spacing (verbatim); ``dz`` the z LEVELS array -> the local interval
    z[j]-z[j-1] (backward) / z[j+1]-z[j] (forward) for the stretched grid."""
    xp = get_xp(F)
    Fxm = xp.zeros_like(F); Fxp = xp.zeros_like(F); Fzm = xp.zeros_like(F); Fzp = xp.zeros_like(F)
    if np.ndim(dx) == 0:
        Fxm[:, 1:] = (F[:, 1:] - F[:, :-1]) / dx
        Fxp[:, :-1] = (F[:, 1:] - F[:, :-1]) / dx
    else:
        dxi = (xp.asarray(dx)[1:] - xp.asarray(dx)[:-1])[None, :]   # interval spacings (1, nx-1)
        Fxm[:, 1:] = (F[:, 1:] - F[:, :-1]) / dxi
        Fxp[:, :-1] = (F[:, 1:] - F[:, :-1]) / dxi
    if np.ndim(dz) == 0:
        Fzm[1:, :] = (F[1:, :] - F[:-1, :]) / dz
        Fzp[:-1, :] = (F[1:, :] - F[:-1, :]) / dz
    else:
        dzi = (xp.asarray(dz)[1:] - xp.asarray(dz)[:-1])[:, None]   # interval spacings (nz-1, 1)
        Fzm[1:, :] = (F[1:, :] - F[:-1, :]) / dzi
        Fzp[:-1, :] = (F[1:, :] - F[:-1, :]) / dzi
    return xp.where(u > 0, u * Fxm, u * Fxp) + xp.where(w > 0, w * Fzm, w * Fzp)


# The advection_scheme values that move the VERTICAL ω-advection limb to a backward-Euler implicit solve
# (and so relieve the dz_min/|w| CFL cap). "vertical_implicit" = Lie-Trotter (advection THEN diffusion) --
# the CANONICAL production scheme (nsub30: dt_dyn=60 + thermo_n_sub=30). The dt=60 climate shift once
# blamed on its O(dt) splitting was carried by the column-thermo CADENCE, fixed by thermo_n_sub
# sub-cycling (PR #203; _sl_FINAL_VERDICT.md / ADVECTION_DT_ROADMAP.md). All three schemes reduce to the
# explicit historic path BYTE-IDENTICALLY when not selected. Centralized so the three CFL/substep
# dispatch sites stay in sync.
_IMPLICIT_ADVECTION_SCHEMES = frozenset({"vertical_implicit", "vertical_implicit_strang",
                                         "vertical_implicit_strang_e"})
# The Strang-family variants (require the implicit-vertical-diffusion / production path):
#   "vertical_implicit_strang"   = A½-D-A½ (symmetrize the two IMPLICIT limbs only). Found to NOT fix the
#                                  chimney shift (the inter-limb commutator is negligible; archive/_sl_strang_RESULTS.md).
#   "vertical_implicit_strang_e" = E½-[A½-D-A½]-E½ (ALSO symmetrize the front-loaded EXPLICIT forcing E;
#                                  likewise found NOT to move the shift -- archive/_sl_strang_RESULTS.md).
#                                  Re-evaluates E on the post-implicit state.
_STRANG_SCHEMES = frozenset({"vertical_implicit_strang", "vertical_implicit_strang_e"})


def _implicit_advection_scheme(scheme):
    """True if ``scheme`` moves the vertical ω-advection limb to an implicit (CFL-relieving) solve."""
    return scheme in _IMPLICIT_ADVECTION_SCHEMES


def _upwind_vertical(F, w, dz):
    """The VERTICAL limb of ``_upwind`` alone: w dF/dz, first-order upwind (backward where w>0, forward
    where w<0). Exactly equal to ``_upwind(F, u, w, dx, dz) - _upwind(F, u, 0, dx, dz)`` for any (u, dx)
    -- the w-term only (test ``test_upwind_vertical_matches_full_upwind_z_limb``). Used to (a) subtract the
    vertical-advection limb from the explicit tendency and (b) define the operator the backward-Euler
    ``_implicit_vertical_advection`` solve inverts, so the two are the SAME discrete operator. ``dz`` scalar
    -> uniform interval (verbatim); ``dz`` the z LEVELS array -> the local interval z[j]-z[j-1] (backward) /
    z[j+1]-z[j] (forward) for the stretched grid (the same spacings ``_upwind`` uses)."""
    xp = get_xp(F)
    Fzm = xp.zeros_like(F); Fzp = xp.zeros_like(F)
    if np.ndim(dz) == 0:
        Fzm[1:, :] = (F[1:, :] - F[:-1, :]) / dz
        Fzp[:-1, :] = (F[1:, :] - F[:-1, :]) / dz
    else:
        dzi = (xp.asarray(dz)[1:] - xp.asarray(dz)[:-1])[:, None]   # interval spacings (nz-1, 1)
        Fzm[1:, :] = (F[1:, :] - F[:-1, :]) / dzi
        Fzp[:-1, :] = (F[1:, :] - F[:-1, :]) / dzi
    return xp.where(w > 0, w * Fzm, w * Fzp)


def _div_grad(F, coef, dx, dz):
    """Conservative variable-coefficient Laplacian div(coef * grad F), compact 5-point, with ``coef``
    arithmetic-averaged to the cell faces. ``coef`` is a scalar or an (nz, nx) field. For a CONSTANT
    coef this reduces EXACTLY to coef * (compact Laplacian) -- i.e. Stage B's ``_lap`` times coef -- so
    the constant-eddy-viscosity (DNS / Rayleigh-Benard) limit is the standard scheme. Interior only;
    the edges are left zero (the BCs are applied by the caller)."""
    xp = get_xp(F)
    coef = coef * xp.ones_like(F)
    o = xp.zeros_like(F)
    cxp = 0.5 * (coef[1:-1, 1:-1] + coef[1:-1, 2:])      # x-face coefficient at i+1/2
    cxm = 0.5 * (coef[1:-1, 1:-1] + coef[1:-1, :-2])      # at i-1/2
    czp = 0.5 * (coef[1:-1, 1:-1] + coef[2:, 1:-1])       # z-face coefficient at j+1/2
    czm = 0.5 * (coef[1:-1, 1:-1] + coef[:-2, 1:-1])      # at j-1/2
    o[1:-1, 1:-1] = _div_grad_x_arm(F, cxp, cxm, dx) + _div_grad_z_arm(F, czp, czm, dz)
    return o


def _div_grad_x_arm(F, cxp, cxm, dx):
    """The interior HORIZONTAL arm of the conservative variable-coefficient Laplacian, shared by
    ``_div_grad`` and ``_div_grad_aniso`` (their x-limbs are identical). ``dx`` SCALAR -> the uniform
    ``/dx**2`` form (verbatim, byte-identical); ``dx`` the x CENTERS array -> the finite-volume per-cell
    form ``(cxp*(F_{i+1}-F_i)/h^+ - cxm*(F_i-F_{i-1})/h^-)/dxc`` with face spacings h^+/h^- and the cell
    thickness dxc = 1/2 (x_{i+1}-x_{i-1}). For equal intervals the FV form reduces (to round-off) to
    ``/dx**2``."""
    xp = get_xp(F)
    if np.ndim(dx) == 0:
        return (cxp * (F[1:-1, 2:] - F[1:-1, 1:-1]) - cxm * (F[1:-1, 1:-1] - F[1:-1, :-2])) / dx**2
    x = xp.asarray(dx)
    hp = (x[2:] - x[1:-1])[None, :]                      # h^+ = x[i+1]-x[i]      (1, nx-2)
    hm = (x[1:-1] - x[:-2])[None, :]                     # h^- = x[i]-x[i-1]      (1, nx-2)
    dxc = (0.5 * (x[2:] - x[:-2]))[None, :]              # cell thickness         (1, nx-2)
    return (cxp * (F[1:-1, 2:] - F[1:-1, 1:-1]) / hp
            - cxm * (F[1:-1, 1:-1] - F[1:-1, :-2]) / hm) / dxc


def _div_grad_z_arm(F, czp, czm, dz):
    """The interior VERTICAL arm of the conservative variable-coefficient Laplacian, shared by
    ``_div_grad`` and ``_div_grad_aniso`` (their z-limbs are identical). ``dz`` SCALAR -> the uniform
    ``/dz**2`` form (verbatim, byte-identical); ``dz`` the z LEVELS array -> the finite-volume per-layer
    form ``(czp*(F_{j+1}-F_j)/h^+ - czm*(F_j-F_{j-1})/h^-)/dzc`` with face spacings h^+/h^- and the cell
    thickness dzc = 1/2 (z_{j+1}-z_{j-1}). For equal intervals the FV form reduces (to round-off) to
    ``/dz**2``."""
    xp = get_xp(F)
    if np.ndim(dz) == 0:
        return (czp * (F[2:, 1:-1] - F[1:-1, 1:-1]) - czm * (F[1:-1, 1:-1] - F[:-2, 1:-1])) / dz**2
    z = xp.asarray(dz)
    hp = (z[2:] - z[1:-1])[:, None]                      # h^+ = z[j+1]-z[j]      (nz-2, 1)
    hm = (z[1:-1] - z[:-2])[:, None]                     # h^- = z[j]-z[j-1]      (nz-2, 1)
    dzc = (0.5 * (z[2:] - z[:-2]))[:, None]              # cell thickness         (nz-2, 1)
    return (czp * (F[2:, 1:-1] - F[1:-1, 1:-1]) / hp
            - czm * (F[1:-1, 1:-1] - F[:-2, 1:-1]) / hm) / dzc


def _div_grad_aniso(F, coef_x, coef_z, dx, dz):
    """ANISOTROPIC variant of ``_div_grad`` (issue #81): the conservative variable-coefficient Laplacian
    div(coef * grad F) with a SEPARATE coefficient on the horizontal (``coef_x``) and vertical (``coef_z``)
    limbs -- the load-bearing piece of the anisotropic Smagorinsky closure. ``coef_x``/``coef_z`` are
    scalars or (nz, nx) fields, each arithmetic-averaged to the relevant cell faces. For ``coef_x ==
    coef_z == coef`` this reduces EXACTLY (byte-for-byte) to ``_div_grad(F, coef, dx, dz)``, so the
    isotropic limit is recovered (test ``test_div_grad_aniso_reduces_to_isotropic``). Interior only; the
    edges are left zero (the BCs are applied by the caller).

    WHY (issue #81): on the ~1000:1 anisotropic fleet grid (dx ~ 1e6 m, dz ~ 1e3 m) the isotropic
    Smagorinsky Delta^2 = dx*dz inflates the VERTICAL eddy term (the 1/dz^2 stencil) by ~dx/dz ~ 10^3.
    Splitting nu_h (~(C_s dx)^2|S|) on the x-faces from nu_v (~(C_s dz)^2|S|) on the z-faces makes the
    vertical mixing Delta_z-based -- more physical AND it lifts the diffusive CFL ceiling ~10^3."""
    xp = get_xp(F)
    cx = coef_x * xp.ones_like(F)
    cz = coef_z * xp.ones_like(F)
    o = xp.zeros_like(F)
    cxp = 0.5 * (cx[1:-1, 1:-1] + cx[1:-1, 2:])          # x-face coefficient at i+1/2
    cxm = 0.5 * (cx[1:-1, 1:-1] + cx[1:-1, :-2])          # at i-1/2
    czp = 0.5 * (cz[1:-1, 1:-1] + cz[2:, 1:-1])           # z-face coefficient at j+1/2
    czm = 0.5 * (cz[1:-1, 1:-1] + cz[:-2, 1:-1])          # at j-1/2
    o[1:-1, 1:-1] = _div_grad_x_arm(F, cxp, cxm, dx) + _div_grad_z_arm(F, czp, czm, dz)
    return o


def build_anelastic_poisson(rho_ref, nx, nz, dx, dz, open_outer_edge=False):
    """LU factorization of the ANELASTIC streamfunction-vorticity inversion (decision #0):

        omega = div( (1/rho_ref) grad psi )    [self-adjoint, variable-coefficient elliptic]

    with psi = 0 Dirichlet on ALL FOUR walls (i in {0, nx-1}, j in {0, nz-1}).

    OPEN OUTER EDGE (``open_outer_edge=True``, the chimney B4-1 open/radiation lateral BC): the
    outer-edge INTERIOR rows (i=nx-1, j in 1..nz-2) become a Neumann / open-outflow condition
    d psi/dx = 0 (``psi[j,nx-1] - psi[j,nx-2] = 0``, RHS 0) instead of psi=0, while the axis (i=0),
    bottom (j=0) and top (j=nz-1) -- INCLUDING the two outer-edge CORNERS -- stay psi=0 Dirichlet. This
    lets the INTERIOR set the full inflow/outflow profile u(z)=beta*dpsi/dz at the edge (the chimney's
    self-selected outflow exits; w~0 there) while the corner Dirichlet pins the net cross-edge mass flux
    ``int dpsi/dz dz = psi(top)-psi(surf) = 0`` automatically (NO Lagrange row needed; the lid/ground
    streamline condition does it). Dirichlet on the remaining 3 walls keeps the operator non-singular
    (cf. ``build_anelastic_poisson_periodic_x``). Default ``False`` -> the verbatim all-walls-Dirichlet
    assembly below (byte-identical). ``rho_ref`` is the (nz,)
    reference density on the height grid (first axis, SURFACE->TOP). The mass streamfunction enforces
    anelastic continuity div(rho_ref u) = 0 via rho_ref u = +d psi/dz, rho_ref w = -d psi/dx; then
    omega = du/dz - dw/dx = div((1/rho_ref) grad psi). Finite-volume stencil with 1/rho_ref averaged to
    the z-faces (beta = 1/rho_ref):
        center : -2 beta_j/dx^2 - (beta_{j+1/2} + beta_{j-1/2})/dz^2
        i +-1  :  beta_j/dx^2
        j +1   :  beta_{j+1/2}/dz^2 ;   j -1 : beta_{j-1/2}/dz^2
    For a CONSTANT rho_ref this is exactly Stage B's constant-coefficient Laplacian, so the Boussinesq
    limit is recovered (verified in the tests). Returns a ``scipy.sparse.linalg.splu`` object; solving
    ``lu.solve(omega.ravel()).reshape(nz, nx)`` returns psi with psi=0 on the walls.

    STRETCHED GRID (BL_REFINEMENT_SPEC.md / the stretched-x program): ``dz`` is EITHER a scalar uniform
    spacing (the verbatim byte-identical assembly above) OR the 1-D array of z LEVELS (nz,), in which case
    the vertical face/cell spacings are PER LAYER -- the j+1 coefficient is ``beta_{j+1/2}/(h^+_j dzc_j)`` and
    the j-1 coefficient ``beta_{j-1/2}/(h^-_j dzc_j)`` with h^+_j=z[j+1]-z[j], h^-_j=z[j]-z[j-1],
    dzc_j=1/2 (z[j+1]-z[j-1]), and the vertical part of the centre is minus their sum. This is EXACTLY the
    operator ``_div_grad`` applies (same face averages, same per-layer spacings), so ``lu.solve`` inverts
    ``_div_grad`` on the stretched grid to machine precision. For equal intervals it reduces to the uniform
    stencil. Likewise ``dx`` is EITHER a scalar uniform spacing OR the 1-D array of x CENTERS (nx,), in which
    case the horizontal face/cell spacings are PER CELL -- the i+1 coefficient is ``beta_j/(h^+_i dxc_i)`` and
    the i-1 coefficient ``beta_j/(h^-_i dxc_i)`` with h^+_i=x[i+1]-x[i], h^-_i=x[i]-x[i-1],
    dxc_i=1/2 (x[i+1]-x[i-1]), and the horizontal part of the centre is minus their sum -- EXACTLY the
    operator ``_div_grad_x_arm`` applies (same derivation, transposed to the x axis), so ``lu.solve`` inverts
    ``_div_grad`` on the stretched-x grid to machine precision too. ``dx``/``dz`` stretch INDEPENDENTLY --
    either, both, or neither may be arrays."""
    beta = 1.0 / np.asarray(rho_ref, float)                  # 1/rho_ref at the levels, (nz,)
    beta_zface = 0.5 * (beta[:-1] + beta[1:])                # 1/rho_ref at the j+1/2 faces, (nz-1,)
    idx = lambda j, i: j * nx + i
    x_uniform = (np.ndim(dx) == 0)
    if x_uniform:
        ix2 = 1.0 / dx**2
    else:
        x = np.asarray(dx, float)
        hxp_i = x[2:] - x[1:-1]                              # h^+ for interior i=1..nx-2 (index q=i-1)
        hxm_i = x[1:-1] - x[:-2]                             # h^-
        dxc_i = 0.5 * (x[2:] - x[:-2])                       # cell thickness
    uniform = (np.ndim(dz) == 0)
    if uniform:
        iz2 = 1.0 / dz**2
    else:
        z = np.asarray(dz, float)
        hp_i = z[2:] - z[1:-1]                               # h^+ for interior j=1..nz-2 (index p=j-1)
        hm_i = z[1:-1] - z[:-2]                              # h^-
        dzc_i = 0.5 * (z[2:] - z[:-2])                       # cell thickness
    r, col, val = [], [], []
    for j in range(nz):
        for i in range(nx):
            k = idx(j, i)
            if j in (0, nz - 1) or i == 0:
                r.append(k); col.append(k); val.append(1.0)      # psi = 0 (z-walls incl. corners; axis)
            elif i == nx - 1:
                if open_outer_edge:                              # open outflow: Neumann dpsi/dx=0
                    r += [k, k]; col += [k, idx(j, nx - 2)]; val += [1.0, -1.0]
                else:
                    r.append(k); col.append(k); val.append(1.0)  # psi = 0 (closed outer wall)
            else:
                bjp = beta_zface[j]                              # j+1/2 face
                bjm = beta_zface[j - 1]                          # j-1/2 face
                if uniform:
                    cjp, cjm = bjp * iz2, bjm * iz2              # uniform: grouped as (bjp+bjm)*iz2 below
                    zpart = (bjp + bjm) * iz2
                else:
                    p = j - 1
                    cjp = bjp / (hp_i[p] * dzc_i[p])             # j+1 coeff: beta_{j+1/2}/(h^+ dzc)
                    cjm = bjm / (hm_i[p] * dzc_i[p])             # j-1 coeff: beta_{j-1/2}/(h^- dzc)
                    zpart = cjp + cjm
                if x_uniform:
                    bxp = bxm = beta[j] * ix2
                    center = -2 * beta[j] * ix2 - zpart
                else:
                    q = i - 1
                    bxp = beta[j] / (hxp_i[q] * dxc_i[q])        # i+1 coeff: beta_j/(h^+ dxc)
                    bxm = beta[j] / (hxm_i[q] * dxc_i[q])        # i-1 coeff: beta_j/(h^- dxc)
                    center = -(bxp + bxm) - zpart
                r += [k] * 5
                col += [k, idx(j, i + 1), idx(j, i - 1), idx(j + 1, i), idx(j - 1, i)]
                val += [center, bxp, bxm, cjp, cjm]
    return splu(sp.csc_matrix((val, (r, col)), shape=(nz * nx, nz * nx)))


class SeparablePoisson:
    """Fast SEPARABLE (DST x tridiagonal) inverter for the closed-wall anelastic streamfunction Poisson,
    a DROP-IN for the ``splu`` object ``build_anelastic_poisson`` returns: same ``.solve(b_flat) -> psi_flat``
    interface, machine-precision-identical result for ANY ``b`` (interior AND walls), at a small fraction of
    the cost on the production-size grids.

    WHY IT IS SEPARABLE. The operator ``M = div((1/rho_ref) grad psi)`` has ``beta = 1/rho_ref`` a function of
    z ONLY, and the x-direction is uniform spacing with psi=0 Dirichlet walls. In operator form on the interior
    points ``M = diag(beta_z) (x) Lxx + Lzz``, where ``Lxx`` is the standard ``[1,-2,1]/dx**2`` Dirichlet
    second-difference on the nx-2 interior x-points. ``Lxx``'s eigenvectors are the DST-I sine basis
    ``sin(m*pi*i/(nx-1))`` with EXACT discrete eigenvalues ``mu_m = -4 sin(m*pi/(2(nx-1)))**2 / dx**2``,
    m=1..nx-2 (NOT the continuous -k**2). So a DST-I along x decouples the system into nx-2 INDEPENDENT
    tridiagonals in z (interior j=1..nz-2, psi=0 at j=0,nz-1):
        sub (to j-1): cjm[j];  diag (j): beta[j]*mu_m - (cjp[j]+cjm[j]);  super (to j+1): cjp[j]
    where ``cjp/cjm`` are EXACTLY ``build_anelastic_poisson``'s z-face coefficients (uniform OR stretched dz).
    The sub/super are the same for every m; only the diagonal carries m (via beta[j]*mu_m).

    NORMALIZATION. ``M = S diag(mu) S^{-1}`` on x with S the (unnormalized) sine transform; because
    ``idst(dst(x, type=1), type=1) == x`` for scipy's type-1 transforms (the forward/inverse normalization
    cancels exactly), the inversion is ``psi = idst( tridiag_solve( dst(rhs) / -- per mode -- ) )`` with NO
    hand-inserted scale factor. scipy ``dst(..., type=1)`` output index k corresponds to m = k+1.

    WALL FOLDING (true drop-in for ANY b). The splu rows for the walls (i in {0,nx-1} or j in {0,nz-1}) are
    identity, so ``lu.solve(b)`` returns ``psi_wall = b_wall`` verbatim and the interior couples to those wall
    values. We reproduce this exactly: copy ``psi_wall = b_wall`` on all four edges+corners, and SUBTRACT each
    wall coupling from the interior RHS before the transform (i=1: -beta[j]/dx**2 * b[j,0]; i=nx-2: likewise to
    b[j,nx-1]; j=1: -cjm[j]*b[0,i]; j=nz-2: -cjp[j]*b[nz-1,i]). A corner-adjacent interior cell gets BOTH an
    x-wall and a z-wall correction (different neighbours -- not double counting). In production ``b`` is
    wall-zeroed, so these terms vanish; folding them keeps the inverter byte-equivalent to ``lu.solve`` regardless.

    STABILITY (no pivoting needed). The per-mode z-tridiag is weakly diagonally dominant -- |diag| =
    |beta[j]*mu_m| + cjp + cjm >= cjp + cjm, strict for every m>=1 since mu_m < 0 -- so the pivot-free Thomas
    sweep is unconditionally stable here. Do NOT add pivoting "to be safe": it is unnecessary and only slower.

    VECTORIZED BATCHED THOMAS (the speedup). The forward-elimination multipliers and inverse pivots depend on
    (m, beta, cjp, cjm) but NOT on the RHS, so they are precomputed ONCE here as (nz-2, nx-2) arrays. At solve
    time only the RHS sweeps run: a Python loop over the SMALL z-axis (~nz), each step a numpy op across the
    WHOLE nx-2 wavenumber axis at once. The loop is never over wavenumber (that would erase the win).

    Closed-wall global core ONLY -- this inverter is DST-I, which diagonalizes the Dirichlet-Dirichlet
    x-operator; the chimney's ``open_outer_edge`` Neumann BC is NOT supported here (guarded by the build-site
    assert). That is an implementation gap, not a mathematical one: the open operator separates as well
    (eliminate the trivial Neumann column -> the Dirichlet-Neumann tridiagonal, identical for every interior
    level, whose orthogonal eigenbasis is the odd-denominator quarter-wave sine transform, eigenvalues
    ``-4 sin^2((2m-1) pi / (2(2n+1))) / dx^2`` with ``n = nx-2``). Verified to 4e-14 against the open-edge splu
    at production nx/nz -- see ``CHIMNEY_BC_SENSITIVITY_RESULTS.md`` arm G."""

    def __init__(self, rho_ref, nx, nz, dx, dz, workers=1):
        self.nx, self.nz = int(nx), int(nz)
        # FFT thread count for the DST/IDST (speedup #3): 1 (default) reproduces the single-threaded call
        # bit-identically; >1 (or -1 for all cores) splits the batched 1-D transforms across the wavenumber
        # axis under scipy.fft.set_workers -- a pure wall-clock lever, each 1-D transform unchanged.
        self.workers = int(workers)
        beta = 1.0 / np.asarray(rho_ref, float)                  # 1/rho_ref at the levels, (nz,)
        beta_zface = 0.5 * (beta[:-1] + beta[1:])                # 1/rho_ref at the j+1/2 faces, (nz-1,)
        self.beta = beta
        self.ix2 = 1.0 / dx**2

        # z-face coefficients cjp[j], cjm[j] for the interior j=1..nz-2 -- EXACTLY build_anelastic_poisson's.
        ji = np.arange(1, nz - 1)                                # interior z indices
        if np.ndim(dz) == 0:
            iz2 = 1.0 / dz**2
            cjp = beta_zface[ji] * iz2
            cjm = beta_zface[ji - 1] * iz2
        else:
            z = np.asarray(dz, float)
            p = ji - 1                                           # offset matches build's hp_i/hm_i/dzc_i
            hp_i = z[2:] - z[1:-1]
            hm_i = z[1:-1] - z[:-2]
            dzc_i = 0.5 * (z[2:] - z[:-2])
            cjp = beta_zface[ji] / (hp_i[p] * dzc_i[p])
            cjm = beta_zface[ji - 1] / (hm_i[p] * dzc_i[p])
        self.cjp = cjp                                           # (nz-2,) super-diagonal (to j+1)
        self.cjm = cjm                                           # (nz-2,) sub-diagonal (to j-1)

        # EXACT discrete Lxx eigenvalues, m=1..nx-2  (scipy dst type-1 index k -> m=k+1).
        m = np.arange(1, nx - 1)                                 # (nx-2,)
        self.mu = -4.0 * np.sin(m * np.pi / (2.0 * (nx - 1)))**2 / dx**2   # (nx-2,)

        # Per-mode tridiagonal diagonals: diag[j,m] = beta[j]*mu_m - (cjp[j]+cjm[j]).  (nz-2, nx-2)
        diag = (beta[ji][:, None] * self.mu[None, :]) - (cjp + cjm)[:, None]

        # Precompute the batched Thomas factors (independent of the RHS): forward-elimination multipliers and
        # inverse pivots, swept down the z-axis once. cprime[j] = super[j] / pivot[j] (only needed for j<nz-3
        # in the back-substitution); inv_pivot[j] = 1/pivot[j]. Both (nz-2, nx-2).
        nzi = nz - 2
        nxi = nx - 2
        inv_pivot = np.empty((nzi, nxi))
        cprime = np.zeros((nzi, nxi))                            # super/pivot for the back-substitution
        pivot = diag[0].copy()                                   # (nxi,)
        inv_pivot[0] = 1.0 / pivot
        for jj in range(1, nzi):
            cprime[jj - 1] = cjp[jj - 1] / pivot                 # super of row jj-1 over its pivot
            pivot = diag[jj] - cjm[jj] * cprime[jj - 1]          # eliminate sub-diagonal cjm[jj]
            inv_pivot[jj] = 1.0 / pivot
        self.inv_pivot = inv_pivot
        self.cprime = cprime
        self._nzi, self._nxi = nzi, nxi

    def solve(self, b_flat):
        """Invert M psi = b with the same semantics as ``splu.solve`` on the closed-wall operator: returns the
        flat (nz*nx,) psi with psi_wall = b_wall and the interior the DST x tridiag solution. Accepts a flat
        (nz*nx,) array (or any shape raveling to it)."""
        nx, nz = self.nx, self.nz
        b = np.asarray(b_flat, float).reshape(nz, nx)

        psi = np.empty((nz, nx))
        # Walls (identity rows): psi_wall = b_wall, verbatim -- all four edges + corners.
        psi[0, :] = b[0, :]
        psi[-1, :] = b[-1, :]
        psi[:, 0] = b[:, 0]
        psi[:, -1] = b[:, -1]

        # Interior RHS with the wall couplings folded in (physical space, before the transform).
        rhs = b[1:-1, 1:-1].copy()                               # (nz-2, nx-2)
        bx = self.beta[1:-1] * self.ix2                          # (nz-2,)  x-coupling beta[j]/dx**2
        rhs[:, 0] -= bx * b[1:-1, 0]                             # i=1 couples to the i=0 axis wall
        rhs[:, -1] -= bx * b[1:-1, -1]                           # i=nx-2 couples to the i=nx-1 outer wall
        rhs[0, :] -= self.cjm[0] * b[0, 1:-1]                    # j=1 couples to the j=0 bottom wall
        rhs[-1, :] -= self.cjp[-1] * b[-1, 1:-1]                 # j=nz-2 couples to the j=nz-1 top wall

        # DST-I along x (axis=1, the interior x-points). Eigenvalue index k -> m=k+1, matched by self.mu.
        # The transform is single-threaded unless workers>1; set_workers(n) splits the batched 1-D DSTs over
        # the wavenumber axis -> BIT-IDENTICAL (each 1-D transform is unchanged). workers==1 takes the bare
        # call (no context) so the default path is structurally the original single-threaded code.
        w = self.workers
        if w == 1:
            bhat = dst(rhs, type=1, axis=1)                      # (nz-2, nx-2)
        else:
            with set_workers(w):
                bhat = dst(rhs, type=1, axis=1)

        # Batched Thomas across the wavenumber axis: forward sweep + back-substitution loop the small z-axis.
        nzi = self._nzi
        d = np.empty_like(bhat)
        d[0] = bhat[0] * self.inv_pivot[0]
        for jj in range(1, nzi):
            d[jj] = (bhat[jj] - self.cjm[jj] * d[jj - 1]) * self.inv_pivot[jj]
        x = np.empty_like(bhat)
        x[-1] = d[-1]
        for jj in range(nzi - 2, -1, -1):
            x[jj] = d[jj] - self.cprime[jj] * x[jj + 1]

        # Inverse DST-I along x (the normalization cancels exactly -- no manual factor).
        if w == 1:
            psi[1:-1, 1:-1] = idst(x, type=1, axis=1)
        else:
            with set_workers(w):
                psi[1:-1, 1:-1] = idst(x, type=1, axis=1)
        return psi.ravel()


class StretchedSeparablePoisson:
    """Fast SEPARABLE inverter for the closed-wall anelastic Poisson on a STRETCHED (non-uniform) x grid --
    the eigendecomposition + GEMM generalization of ``SeparablePoisson``, a drop-in for the ``splu`` object
    ``build_anelastic_poisson(rho_ref, nx, nz, x_centers, dz)`` returns (same ``.solve(b_flat)`` interface,
    same wall semantics).

    WHY IT STILL SEPARATES (the arm-G argument generalized -- CHIMNEY_BC_SENSITIVITY_RESULTS.md arm G proved
    the DST restriction was an implementation gap, not mathematics). On the interior points the operator is
    ``M = diag(beta_z) (x) Lxx + Lzz`` exactly as in the uniform case; non-uniform spacing only changes WHAT
    ``Lxx`` is, not the tensor structure. Here ``Lxx = W^-1 A`` with ``W = diag(dxc_i)`` (the centers-derived
    cell thickness, EXACTLY ``build_anelastic_poisson``'s / ``_div_grad_x_arm``'s) and ``A`` the SYMMETRIC
    tridiagonal with off-diagonal ``1/h_{i+1/2}`` (shared by both neighbours -- that is why A is symmetric)
    and diagonal ``-(1/h^- + 1/h^+)``. ``Lxx`` is not symmetric, but the diagonal similarity

        S = W^{1/2} Lxx W^{-1/2} = W^{-1/2} A W^{-1/2}

    is symmetric tridiagonal and negative definite under the psi=0 Dirichlet walls, so ``eigh_tridiagonal``
    gives ``S = Q diag(lam) Q^T`` with ORTHONORMAL ``Q`` and all ``lam_m < 0``, once at init. Substituting
    ``psi = W^{-1/2} Q xi`` per level and projecting decouples the system into nx-2 independent z-tridiagonals

        sub cjm[j];   diag beta[j]*lam_m - (cjp[j]+cjm[j]);   super cjp[j]

    -- VERBATIM ``SeparablePoisson``'s per-mode system with ``mu_m -> lam_m`` (the z-face coefficients, the
    diagonal-dominance/no-pivoting argument, and the batched-Thomas precompute/sweep are reused unchanged).
    The transforms are two dense (nx-2)x(nx-2) GEMMs per solve,

        forward  V_f = diag(sqrt(dxc)) Q      (rhs_hat = rhs_int @ V_f)
        backward V_b = Q^T diag(1/sqrt(dxc))  (psi_int = xi @ V_b)

    -- at production nx ~ 870 that is ~4e8 FLOP/solve in fp64: negligible on datacenter GPUs, ~1-2 ms on a
    GA102-class card, and cheap on CPU BLAS. UNIFORM LIMIT = free cross-validation: with equal spacing
    ``S = tridiag(1,-2,1)/dx^2``, ``Q`` is the DST-I sine basis and ``lam_m == mu_m`` analytically, so this
    solver must match ``SeparablePoisson`` to near machine precision on a uniform grid (tested).

    WALL FOLDING: identical in form to ``SeparablePoisson`` -- the only change is that the x-wall coupling
    coefficients are the stretched ``bxm(i=1) = beta[j]/(h^-_1 dxc_1)`` and ``bxp(i=nx-2) =
    beta[j]/(h^+_{nx-2} dxc_{nx-2})`` instead of the shared ``beta[j]/dx^2``.

    CONDITIONING (stated honestly): the x-operator's condition number grows like ``(2 x_max/(pi dx_min))^2``
    ~ 1e10 at the production stretch (2 km / 3.2e8 m), so fp64 backward error implies ~1e-6 RELATIVE solution
    error worst-case -- fine for psi (the Poisson is deliberately fp64 even on the fp32 core), fatal in fp32.
    Parity gates against splu are RESIDUAL-based at production scale, not naive 1e-13 (the arm-G 4e-14 was
    nx=200 uniform)."""

    def __init__(self, rho_ref, nx, nz, x_centers, dz):
        self.nx, self.nz = int(nx), int(nz)
        beta = 1.0 / np.asarray(rho_ref, float)                  # 1/rho_ref at the levels, (nz,)
        beta_zface = 0.5 * (beta[:-1] + beta[1:])                # 1/rho_ref at the j+1/2 faces, (nz-1,)
        self.beta = beta

        # x geometry: EXACTLY build_anelastic_poisson's stretched-x spacings (interior i=1..nx-2).
        x = np.asarray(x_centers, float)
        if x.shape != (self.nx,):
            raise ValueError(f"x_centers must be the (nx,) CENTERS array; got {x.shape} vs nx={nx}")
        hxp = x[2:] - x[1:-1]                                    # h^+_i (nx-2,)
        hxm = x[1:-1] - x[:-2]                                   # h^-_i
        dxc = 0.5 * (x[2:] - x[:-2])                             # centers-derived cell thickness
        self._bxm1 = 1.0 / (hxm[0] * dxc[0])                     # x-wall couplings / beta[j] (scalars)
        self._bxp_last = 1.0 / (hxp[-1] * dxc[-1])

        # z-face coefficients cjp/cjm -- verbatim SeparablePoisson (uniform OR stretched dz).
        ji = np.arange(1, nz - 1)
        if np.ndim(dz) == 0:
            iz2 = 1.0 / dz**2
            cjp = beta_zface[ji] * iz2
            cjm = beta_zface[ji - 1] * iz2
        else:
            z = np.asarray(dz, float)
            p = ji - 1
            hp_i = z[2:] - z[1:-1]
            hm_i = z[1:-1] - z[:-2]
            dzc_i = 0.5 * (z[2:] - z[:-2])
            cjp = beta_zface[ji] / (hp_i[p] * dzc_i[p])
            cjm = beta_zface[ji - 1] / (hm_i[p] * dzc_i[p])
        self.cjp = cjp
        self.cjm = cjm

        # The symmetrized x-operator S = W^{-1/2} A W^{-1/2}: eigendecompose ONCE (fp64 host).
        s_diag = -(1.0 / hxm + 1.0 / hxp) / dxc                  # (nx-2,)
        s_off = 1.0 / (hxp[:-1] * np.sqrt(dxc[:-1] * dxc[1:]))   # (nx-3,)  1/(h_{i+1/2} sqrt(dxc_i dxc_{i+1}))
        lam, Q = eigh_tridiagonal(s_diag, s_off)                 # S = Q diag(lam) Q^T, Q orthonormal
        if lam.max() >= 0.0:
            raise AssertionError("stretched-x Poisson: the symmetrized x-operator must be negative definite "
                                 "under Dirichlet walls -- non-negative eigenvalue found (grid bug?)")
        self.lam = lam
        sq = np.sqrt(dxc)
        self._Vf = sq[:, None] * Q                               # forward GEMM: rhs_hat = rhs_int @ Vf
        self._Vb = Q.T * (1.0 / sq)[None, :]                     # backward GEMM: psi_int = xi @ Vb

        # Per-mode z-tridiagonal batched-Thomas precompute -- verbatim SeparablePoisson with mu -> lam.
        diag = (beta[ji][:, None] * lam[None, :]) - (cjp + cjm)[:, None]
        nzi = nz - 2
        nxi = nx - 2
        inv_pivot = np.empty((nzi, nxi))
        cprime = np.zeros((nzi, nxi))
        pivot = diag[0].copy()
        inv_pivot[0] = 1.0 / pivot
        for jj in range(1, nzi):
            cprime[jj - 1] = cjp[jj - 1] / pivot
            pivot = diag[jj] - cjm[jj] * cprime[jj - 1]
            inv_pivot[jj] = 1.0 / pivot
        self.inv_pivot = inv_pivot
        self.cprime = cprime
        self._nzi = nzi

    def solve(self, b_flat):
        """Invert M psi = b with ``splu.solve`` semantics on the closed-wall stretched-x operator: returns
        the flat (nz*nx,) psi with psi_wall = b_wall and the interior the eigenbasis x tridiag solution."""
        nx, nz = self.nx, self.nz
        b = np.asarray(b_flat, float).reshape(nz, nx)

        psi = np.empty((nz, nx))
        psi[0, :] = b[0, :]
        psi[-1, :] = b[-1, :]
        psi[:, 0] = b[:, 0]
        psi[:, -1] = b[:, -1]

        # Interior RHS with the wall couplings folded in (physical space, before the transform).
        rhs = b[1:-1, 1:-1].copy()                               # (nz-2, nx-2)
        bi = self.beta[1:-1]                                     # (nz-2,)
        rhs[:, 0] -= (bi * self._bxm1) * b[1:-1, 0]              # i=1 couples to the i=0 axis wall
        rhs[:, -1] -= (bi * self._bxp_last) * b[1:-1, -1]        # i=nx-2 couples to the outer wall
        rhs[0, :] -= self.cjm[0] * b[0, 1:-1]                    # j=1 couples to the bottom wall
        rhs[-1, :] -= self.cjp[-1] * b[-1, 1:-1]                 # j=nz-2 couples to the top wall

        bhat = rhs @ self._Vf                                    # forward GEMM (the DST's replacement)

        nzi = self._nzi
        d = np.empty_like(bhat)
        d[0] = bhat[0] * self.inv_pivot[0]
        for jj in range(1, nzi):
            d[jj] = (bhat[jj] - self.cjm[jj] * d[jj - 1]) * self.inv_pivot[jj]
        x = np.empty_like(bhat)
        x[-1] = d[-1]
        for jj in range(nzi - 2, -1, -1):
            x[jj] = d[jj] - self.cprime[jj] * x[jj + 1]

        psi[1:-1, 1:-1] = x @ self._Vb                           # backward GEMM (the IDST's replacement)
        return psi.ravel()


def velocities_from_psi(psi, beta, dx, dz):
    """Anelastic velocities from the mass streamfunction: u = (1/rho_ref) d psi/dz,
    w = -(1/rho_ref) d psi/dx. ``beta`` = 1/rho_ref, shape (nz,) (broadcast over x). For constant
    rho_ref this is Stage B's u = d psi/dz, w = -d psi/dx (Boussinesq). ``dx``/``dz`` may each
    independently be a scalar uniform spacing or the x CENTERS / z LEVELS array (flows through
    ``_ddz``/``_ddx``)."""
    xp = get_xp(psi)
    b = xp.asarray(beta, psi.dtype)[:, None]   # beta (1/rho_ref) follows psi's MODULE and DTYPE (tiny (nz,));
    #   dtype-following (was hardcoded fp64) so an fp32 psi keeps the velocity recovery fp32 -- on an fp64
    #   psi this is identical to xp.asarray(beta, float) (byte-identical fp64 path).
    return b * _ddz(psi, dz), -b * _ddx(psi, dx)


def strain_rate_magnitude(u, w, dx, dz):
    """The 2-D strain-rate magnitude |S| = sqrt(2 S_ij S_ij) = sqrt(2 u_x^2 + 2 w_z^2 + (u_z + w_x)^2) on
    the height grid. Shared by the Smagorinsky eddy viscosity (which forms (C_s*Delta)^2 |S|) and the
    frictional-dissipation-heating rate (rho_ref nu_t |S|^2, M3.4b Task 2), so both read ONE strain field."""
    xp = get_xp(u)
    ux = _ddx(u, dx); wz = _ddz(w, dz); uz = _ddz(u, dz); wx = _ddx(w, dx)
    return xp.sqrt(2.0 * ux**2 + 2.0 * wz**2 + (uz + wx)**2)


def _mixing_dz(dz):
    """The LOCAL vertical mixing length for the Smagorinsky grid scale Delta. ``dz`` SCALAR -> returns it
    unchanged (the uniform Delta, byte-identical). ``dz`` the z LEVELS array -> the per-level cell thickness
    as an (nz, 1) column (interior 1/2 (z[j+1]-z[j-1]); edges the local interval), so fine BL layers get a
    SMALL Delta and a correspondingly small eddy viscosity instead of inheriting a global constant -- if the
    refined layers got the coarse-grid Delta they would over-smooth and erase the very BL jet we refine to
    resolve (BL_REFINEMENT_SPEC.md #4B.6)."""
    if np.ndim(dz) == 0:
        return dz
    # device-generic (Phase E): on the GPU dynamics path ``dz`` is the device z-levels array, so coerce via the
    # array's OWN module (np.asarray on a CuPy array raises). On CPU ``get_xp`` is numpy -> byte-identical.
    xp = get_xp(dz)
    z = xp.asarray(dz, float)
    dzc = xp.empty_like(z)
    dzc[1:-1] = 0.5 * (z[2:] - z[:-2])
    dzc[0] = z[1] - z[0]; dzc[-1] = z[-1] - z[-2]
    return dzc[:, None]


def _mixing_dx(dx):
    """The LOCAL horizontal mixing length for the Smagorinsky grid scale Delta -- the x-axis mirror of
    ``_mixing_dz``. ``dx`` SCALAR -> returns it unchanged (the uniform Delta, byte-identical). ``dx`` the
    x CENTERS array -> the per-cell dual width as a (1, nx) ROW (interior 1/2 (x[i+1]-x[i-1]); edges the
    local interval), so fine axis cells get a SMALL Delta and a correspondingly small eddy viscosity
    instead of inheriting a global constant -- if the refined axis cells got the coarse-grid Delta they
    would over-smooth and erase the very equatorial updraft the refinement resolves."""
    if np.ndim(dx) == 0:
        return dx
    # device-generic, mirroring _mixing_dz: coerce via the array's OWN module (np.asarray on a CuPy array
    # raises). On CPU get_xp is numpy -> byte-identical.
    xp = get_xp(dx)
    x = xp.asarray(dx, float)
    dxc = xp.empty_like(x)
    dxc[1:-1] = 0.5 * (x[2:] - x[:-2])
    dxc[0] = x[1] - x[0]; dxc[-1] = x[-1] - x[-2]
    return dxc[None, :]


def smagorinsky_viscosity(u, w, dx, dz, C_s, nu_floor):
    """State-dependent Smagorinsky eddy viscosity (decision #4): nu_t = (C_s * Delta)^2 |S| + nu_floor,
    Delta^2 = dx_local * dz_local the grid scale (dx_local/dz_local are the LOCAL horizontal/vertical
    cells on a stretched grid; scalar dx/dz on a uniform grid -> byte-identical), |S| = sqrt(2 S_ij S_ij)
    the strain-rate magnitude (``strain_rate_magnitude``). ``dx``/``dz`` may each independently be a
    scalar uniform spacing or the x CENTERS / z LEVELS array. Setting C_s = 0 gives the constant molecular
    floor nu_floor (the DNS limit used by the Rayleigh-Benard benchmark). nu_t is what M3 perturbs
    (C_s, nu_floor +-2x) while pushing resolution."""
    return (C_s**2 * _mixing_dx(dx) * _mixing_dz(dz)) * strain_rate_magnitude(u, w, dx, dz) + nu_floor


def smagorinsky_viscosity_aniso(u, w, dx, dz, C_s, nu_floor):
    """ANISOTROPIC Smagorinsky eddy viscosity (issue #81): a directional-Delta split returning
    ``(nu_h, nu_v)`` with the SHARED isotropic strain-rate magnitude ``|S|``:

        nu_h = (C_s * dx_local)^2 |S| + nu_floor   (horizontal eddy viscosity, acts on the x-face stencil)
        nu_v = (C_s * dz_local)^2 |S| + nu_floor   (vertical   eddy viscosity, acts on the z-face stencil)

    The directional-Delta split is the load-bearing part; a directional |S| split (u-shear vs w-shear) is
    a refinement not worth the complexity here (issue #81 implementation guide step 1.2). On a near-
    ISOTROPIC grid (dx == dz) this returns ``(nu_t, nu_t)`` EQUAL to ``smagorinsky_viscosity`` (Delta^2 =
    dx*dz = dx^2 = dz^2), so the closure choice is invisible where the grid is square; on the ~1000:1
    fleet grid nu_v drops ~(dz/dx)^2 ~ 10^-6 relative to the isotropic Delta^2 on the vertical limb,
    which is the whole point (the isotropic form carried the huge horizontal scale into the vertical
    mixing). ``C_s = 0`` collapses BOTH to the molecular floor ``nu_floor``. ``dx``/``dz`` may each
    independently be a scalar uniform spacing or the x CENTERS / z LEVELS array (``_mixing_dx``/
    ``_mixing_dz``) -- e.g. a stretched-x grid gives fine axis cells a proportionally small ``nu_h``."""
    S = strain_rate_magnitude(u, w, dx, dz)
    ldx = _mixing_dx(dx)                                       # local horizontal scale (scalar uniform | (1,nx))
    ldz = _mixing_dz(dz)                                       # local vertical scale (scalar uniform | (nz,1))
    nu_h = (C_s**2 * ldx * ldx) * S + nu_floor
    nu_v = (C_s**2 * ldz * ldz) * S + nu_floor
    return nu_h, nu_v


def vorticity_tendency(omega, b, u, w, dlnrho_dz, nu_t, dx, dz, nu_t_z=None):
    """The f=0 ANELASTIC vorticity tendency d omega/dt (decisions #0, #2, #4):

        d omega/dt = -u.grad(omega)              advection
                     - d b/dx                    buoyancy torque (b = g (T - T_ref)/T_ref, [m/s^2])
                     + omega * w * dlnrho_dz      anelastic vortex stretching, -omega(div u),
                                                  div u = -w d(ln rho_ref)/dz from div(rho_ref u)=0
                     + div(nu_t grad omega)       Smagorinsky/eddy diffusion of vorticity

    The pressure gradient drops out under the curl (anelastic -grad(p'/rho_ref) is a perfect gradient).
    For a CONSTANT rho_ref (dlnrho_dz = 0) and constant nu_t this is exactly Stage B's dimensional
    vorticity equation -u.grad(omega) - db/dx + nu lap(omega) -- the Boussinesq limit. ``dlnrho_dz`` is
    the (nz,) profile d(ln rho_ref)/dz (negative; the density scale height is ~1/|dlnrho_dz|).

    ANISOTROPIC CLOSURE (issue #81): ``nu_t_z=None`` (the default) keeps the ISOTROPIC eddy term
    ``div(nu_t grad omega)`` -- byte-for-byte the pre-#81 form. Passing a vertical eddy viscosity
    ``nu_t_z`` switches the eddy term to the directional-split ``_div_grad_aniso(omega, nu_t, nu_t_z)``
    (``nu_t`` then plays the role of the HORIZONTAL nu_h). ``dx``/``dz`` may each independently be a
    scalar uniform spacing or the x CENTERS / z LEVELS array."""
    xp = get_xp(omega)
    eddy = (_div_grad(omega, nu_t, dx, dz) if nu_t_z is None
            else _div_grad_aniso(omega, nu_t, nu_t_z, dx, dz))
    return (-_upwind(omega, u, w, dx, dz)
            - _ddx(b, dx)
            + omega * w * xp.asarray(dlnrho_dz, omega.dtype)[:, None]   # dtype-following (was hardcoded fp64):
            #   keeps the vortex-stretching term fp32 under an fp32 omega; identical on fp64 (byte-identical).
            + eddy)


def _vertical_eddy_term(F, coef_z, dz):
    """The interior VERTICAL arm of ``div(coef_z grad F)`` as a full (nz, nx) field (edges left 0) -- i.e.
    exactly the z-limb of ``_div_grad``/``_div_grad_aniso`` (same face-averaged ``coef_z``). Used to (a)
    subtract the vertical eddy limb from the explicit tendency and (b) define the operator the implicit
    vertical-diffusion solve inverts (``_implicit_vertical_diffusion``), so the two are the SAME discrete
    operator. ``coef_z`` is a scalar or (nz, nx) field; ``dz`` scalar or the z-levels array."""
    xp = get_xp(F)
    cz = coef_z * xp.ones_like(F)
    czp = 0.5 * (cz[1:-1, 1:-1] + cz[2:, 1:-1])               # z-face coefficient at j+1/2
    czm = 0.5 * (cz[1:-1, 1:-1] + cz[:-2, 1:-1])              # at j-1/2
    o = xp.zeros_like(F)
    o[1:-1, 1:-1] = _div_grad_z_arm(F, czp, czm, dz)
    return o


def _solve_tridiag_columns(sub, diag, sup, rhs):
    """Solve the per-column tridiagonal systems with bands ``(sub, diag, sup)`` and right-hand side
    ``rhs``, all shape (nz, nx): for each column i, ``sub[j]`` multiplies x[j-1], ``sup[j]`` multiplies
    x[j+1] (``sub[0]``/``sup[-1]`` unused). Thomas algorithm, vectorized over the column axis (the j-loop
    is over the small nz; each step is an (nx,) vector op). Host numpy -- the CPU-path sweep; on the
    GPU dynamics path the implicit solves instead run in a device Thomas RawKernel
    (``_gpu_kernels.solve_tridiag_columns_device``), like the Poisson's device DST port."""
    sub = np.asarray(sub, float); diag = np.asarray(diag, float)
    sup = np.asarray(sup, float); rhs = np.asarray(rhs, float)
    nz = diag.shape[0]
    cp = np.empty_like(diag); dp = np.empty_like(rhs)
    cp[0] = sup[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for j in range(1, nz):
        m = diag[j] - sub[j] * cp[j - 1]
        cp[j] = sup[j] / m
        dp[j] = (rhs[j] - sub[j] * dp[j - 1]) / m
    x = np.empty_like(rhs)
    x[-1] = dp[-1]
    for j in range(nz - 2, -1, -1):
        x[j] = dp[j] - cp[j] * x[j + 1]
    return x


# ======================================================================================
# THE SHARED COLUMN-THERMODYNAMICS KERNEL (M2b.3): hydrostatic heights, the wetness/albedo surface
# inputs, the enthalpy-conserving moist convective adjustment, and the independent single-column RCE
# anchor. As with the dynamical kernel, these are MODULE-LEVEL pure functions so that the SAME leaf
# physics is used by both ``StageADCore`` and the standalone ``single_column_rce`` anchor -- the
# anchor's INDEPENDENCE is in the *assembly/relaxation loop* (a different driver), per the roadmap's
# tier-2/3 distinction, not in re-deriving radiation/thermo (which are separately unit-tested leaves).
# All operate on the PRESSURE grid (last axis TOP->SURFACE), the column-thermodynamics convention.
# ======================================================================================
def hydrostatic_height(T_layers, p_edge):
    """Layer-center physical heights z [m] from the hydrostatic (hypsometric) relation, for a column or
    chain of columns. ``T_layers`` (..., nz) ordered TOP->SURFACE; ``p_edge`` (nz+1,) the interface
    pressures TOP->SURFACE. Returns (..., nz) heights, TOP->SURFACE (so z[...,nz-1] ~ 0 at the surface).
    The bridge between the pressure grid (thermo) and the height grid (dynamics); lifted from Stage A and
    generalized to an arbitrary batch shape so the 1-column RCE anchor and the 2-D core share it."""
    T = np.asarray(T_layers, float)
    p_edge = np.asarray(p_edge, float)
    *batch, nz = T.shape
    Tedge = np.empty((*batch, nz + 1))
    Tedge[..., 1:-1] = 0.5 * (T[..., :-1] + T[..., 1:])
    Tedge[..., 0] = T[..., 0]; Tedge[..., -1] = T[..., -1]
    z_edge = np.zeros((*batch, nz + 1))
    for k in range(nz - 1, -1, -1):                            # surface interface = nz (z=0) upward
        dz = (C.R_DRY * Tedge[..., k + 1] / C.G_ACCEL) * (np.log(p_edge[k + 1]) - np.log(p_edge[k]))
        z_edge[..., k] = z_edge[..., k + 1] + dz
    return 0.5 * (z_edge[..., :-1] + z_edge[..., 1:])


def _interp_columns(x, xp, fp, force_generic=False):
    """Batched per-row linear interpolation: ``np.interp(x[i], xp[i], fp[i])`` for every row i in ONE
    vectorized shot -- the issue-#99 replacement for the per-column ``np.interp`` Python loops of the
    dual-grid reconciliation (``_pressure_to_height`` / ``_height_to_pressure``), which the Step-0 fleet
    profile flagged as a `_thermo_substep` hot spot. Inputs are 2-D with broadcastable row dims: ``x``
    (B, M) the query points; ``xp`` (B, N) the sample points, each row STRICTLY INCREASING (the z grids
    are strictly monotone); ``fp`` (B, N) the sample values. Any of them may have a single row that
    broadcasts. Returns (B, M). Host-numpy only, like the np.interp loops it replaces.

    BIT-IDENTICAL BY CONSTRUCTION (the issue-#99 house rule): ``np.interp``'s C kernel computes, per
    query, the bracket j with xp[j] <= x < xp[j+1] and then EXACTLY
        slope * (x - xp[j]) + fp[j],   slope = (fp[j+1] - fp[j]) / (xp[j+1] - xp[j]),
    with short-circuits for an exact node hit (x == xp[j] -> fp[j]), the top sample (x >= xp[N-1] ->
    fp[N-1]) and the left clamp (x < xp[0] -> fp[0]). This function reproduces that exact value chain:
    the bracket is exact COMPARISONS only (one C-level ``searchsorted`` when the sample row is shared --
    the ``_height_to_pressure`` shape -- else the vectorized comparison count, == ``searchsorted(
    side='right') - 1`` per row), the gathers are flat ``take``s (cheaper than ``take_along_axis`` at
    fleet size), and the interpolant uses the same grouping of IEEE-exact elementwise ops (+,-,*,/ and
    selection only; no transcendentals, so SIMD-vs-scalar code paths cannot diverge). NOT reproduced
    (PR #102 review): the C kernel's recompute-from-the-right-endpoint fallback, which fires whenever
    the interpolant lands NaN -- from NaN inputs, but ALSO from FINITE inputs at extreme magnitude
    (slope overflow at |fp| ~ 1e308); and an N == 1 sample row (np.interp returns fp[0]; this would
    IndexError on the j+1 gather). Both are unreachable here -- the model's z/T/q/u fields are finite
    at physical magnitudes and the grids have nz >= 4 -- but they bound any general-purpose reuse.
    Gated bitwise against per-column ``np.interp`` on realistic AND adversarial states by
    ``tests/test_stage_ad_core.py`` (issue #99).

    DEVICE (Phase D): array-module-generic via ``get_xp`` (the array module is ``am`` here -- ``xp`` is the
    SAMPLE-points parameter name, not the module). numpy in -> the verbatim numpy ops (BIT-IDENTICAL); CuPy
    in -> the SAME bracket (searchsorted/bincount/cumsum/take) + lerp on the GPU. PROBED bit-for-bit on all
    three branches (CuPy fp64 == numpy fp64): the integer-bracket path is exact on CuPy, so the dual-grid
    reconciliation -- the ONE seam the port audit flagged as host-bound -- carries no parity loss on device."""
    am = get_xp(x if is_device(x) else (xp if is_device(xp) else fp))
    x = am.asarray(x, float)
    xp = am.asarray(xp, float)
    fp = am.asarray(fp, float)
    n = xp.shape[-1]
    if xp.shape[0] == 1:                                         # shared sample row: ONE C searchsorted
        j = am.searchsorted(xp[0], x.reshape(-1), side="right").reshape(x.shape) - 1
    # ``force_generic`` (Phase E): route to the GENERIC comparison-count branch (the ``else`` below) on the
    # CUDA-graph capture path. Branch 2 here (the searchsorted+BINCOUNT transpose trick) is capture-HOSTILE:
    # CuPy's ``bincount`` does an INTERNAL device->host sync (``if (x<0).any()``), illegal during capture and
    # unsuppressable (it is inside cupyx, not our code). ``not force_generic`` SHORT-CIRCUITS before the
    # ``bool((x[0,1:]>=x[0,:-1]).all())`` check too, so its OWN D2H sync is also skipped. The else branch is a
    # pure broadcast-compare + reduction + gather (all kernels, no sync) and is BIT-IDENTICAL to branch 2 (the
    # bracket ``j`` is the same ``searchsorted(side='right')-1`` per row, by construction; the eager ON-vs-OFF
    # parity check is the empirical discriminator). OFF -> verbatim shipped live-check + bincount (byte-id).
    elif x.shape[0] == 1 and not force_generic and bool((x[0, 1:] >= x[0, :-1]).all()):
        # Per-row samples against ONE shared ascending query row (the ``_pressure_to_height`` shape):
        # the transpose trick. Locate every sample among the queries with ONE C searchsorted
        # (xp[i,k] <= x[m]  <=>  m >= L[i,k], L = searchsorted(x, xp, 'left'); needs only x ascending),
        # then count per query by bincount + cumsum: c[i,m] = #{k: L[i,k] <= m} = #{k: xp[i,k] <= x[m]}
        # EXACTLY -- comparisons and integer counting only, so the bracket stays exact. O(B*(N+M))
        # instead of the O(B*N*M) comparison matrix (which loses to the np.interp loop at production
        # resolution; measured, issue #99).
        B_xp, M = xp.shape[0], x.shape[-1]
        L = am.searchsorted(x[0], xp.reshape(-1), side="left")
        L += am.repeat(am.arange(B_xp) * (M + 1), n)
        hist = am.bincount(L, minlength=B_xp * (M + 1)).reshape(B_xp, M + 1)
        j = am.cumsum(hist[:, :M], axis=1) - 1
    else:                                                        # generic: exact comparison count
        j = (xp[:, :, None] <= x[:, None, :]).sum(axis=1, dtype=am.intp) - 1
    j = am.maximum(am.minimum(j, n - 2), 0)                      # raw ufuncs (np.clip wrapper is slower)
    B = max(x.shape[0], xp.shape[0], fp.shape[0])
    if j.shape[0] != B:
        j = am.broadcast_to(j, (B,) + j.shape[1:])
    jf = None                                                    # per-row flat indices, built at most once
    if xp.shape[0] == 1:                                         # gathers: flat ``take`` (cheaper than
        x0 = xp[0].take(j); x1 = xp[0].take(j + 1)               #   take_along_axis at fleet size)
    else:
        jf = j + (am.arange(B) * n)[:, None]
        xpr = xp.reshape(-1)                                     # copies only if non-contiguous (a view slice)
        x0 = xpr.take(jf); x1 = xpr.take(jf + 1)
    if fp.shape[0] == 1:
        f0 = fp[0].take(j); f1 = fp[0].take(j + 1)
    else:
        if jf is None:
            jf = j + (am.arange(B) * n)[:, None]
        fpr = fp.reshape(-1)
        f0 = fpr.take(jf); f1 = fpr.take(jf + 1)
    out = ((f1 - f0) / (x1 - x0)) * (x - x0) + f0                # np.interp's slope*(x-xp[j])+fp[j]
    out = am.where(x == x0, f0, out)                             # exact node hit (C short-circuit)
    out = am.where(x < xp[:, :1], fp[:, :1], out)                # left clamp -> fp[0]
    out = am.where(x >= xp[:, -1:], fp[:, -1:], out)             # top sample / right clamp -> fp[N-1]
    return out


def _interp_columns_pair(x, xp, fp1, fp2, fp3=None, fp4=None, *, force_generic=False):
    """Two (or three, or four) batched per-row linear interpolations sharing the SAME query/sample grids (``x``,
    ``xp``) but DIFFERENT value rows (``fp1``, ``fp2``, optionally ``fp3``): the ``_update_buoyancy``
    pattern where ``T`` and ``q`` are both
    mapped p->h on the identical column heights. The bracket ``j`` and the sample bounds ``x0``/``x1`` depend
    ONLY on (``x``, ``xp``), so they are computed ONCE and reused for both gathers -- halving the bracket cost
    (the per-row searchsorted+bincount+cumsum), the heaviest part. BIT-IDENTICAL to calling ``_interp_columns``
    twice: every value is the SAME IEEE expression with the SAME operand grouping, only the leaf ``fp`` differs.
    Returns ``(out1, out2)``. Same input contract as ``_interp_columns`` (``fp1``/``fp2`` share ``xp``'s row dim,
    or are single-row broadcast).

    THIRD FIELD (PHY-19 condensate loading): pass ``fp3`` to share the SAME single bracket across three
    gathers and get ``(out1, out2, out3)`` back. ``fp3=None`` (the default) is the shipped two-field call
    in every respect -- the bracket, the bounds and both gathers are the identical expressions on the
    identical operands, so the two-field result is byte-identical to before this argument existed.
    ``force_generic`` is KEYWORD-ONLY: it sits behind ``fp3`` now, and a positional 5th argument would
    otherwise bind to ``fp3`` and be silently interpolated as a value row.

    FOURTH FIELD (PHY-22 B', the buoyancy pressure term): ``fp4`` adds a fourth gather on the SAME
    bracket -- the layer pressures ``p_mid``, mapped onto the dynamics heights to give the live column's
    hydrostatic pressure there. It is passed as a SINGLE ROW (1, N), since ``p_mid`` is x-independent, so
    it takes the cheap ``fp[0].take(j)`` gather and allocates no (B, N) broadcast. ``fp4`` may be given
    only with ``fp3``; ``fp4=None`` leaves every expression above untouched, so the two- and three-field
    results stay byte-identical to before this argument existed. Returns a 4-tuple when given.

    DEVICE (Phase D): array-module-generic via ``get_xp`` (module ``am``; ``xp`` is the sample-points param),
    numpy bit-identical / CuPy on the GPU -- same probe as ``_interp_columns``."""
    am = get_xp(x if is_device(x) else (xp if is_device(xp) else (fp1 if is_device(fp1) else fp2)))
    x = am.asarray(x, float)
    xp = am.asarray(xp, float)
    fp1 = am.asarray(fp1, float)
    fp2 = am.asarray(fp2, float)
    if fp3 is not None:
        fp3 = am.asarray(fp3, float)
    if fp4 is not None:
        if fp3 is None:                       # a 4th value row with no 3rd would silently return a pair
            raise ValueError("_interp_columns_pair: fp4 requires fp3 (pass fp3 explicitly).")
        fp4 = am.asarray(fp4, float)
    n = xp.shape[-1]
    if xp.shape[0] == 1:                                         # shared sample row: ONE C searchsorted
        j = am.searchsorted(xp[0], x.reshape(-1), side="right").reshape(x.shape) - 1
    elif x.shape[0] == 1 and not force_generic and bool((x[0, 1:] >= x[0, :-1]).all()):  # Phase E: route to generic (no bincount)
        B_xp, M = xp.shape[0], x.shape[-1]                       # per-row samples vs shared ascending query
        L = am.searchsorted(x[0], xp.reshape(-1), side="left")
        L += am.repeat(am.arange(B_xp) * (M + 1), n)
        hist = am.bincount(L, minlength=B_xp * (M + 1)).reshape(B_xp, M + 1)
        j = am.cumsum(hist[:, :M], axis=1) - 1
    else:                                                        # generic: exact comparison count
        j = (xp[:, :, None] <= x[:, None, :]).sum(axis=1, dtype=am.intp) - 1
    j = am.maximum(am.minimum(j, n - 2), 0)
    B = max(x.shape[0], xp.shape[0], fp1.shape[0], fp2.shape[0],
            *((fp3.shape[0],) if fp3 is not None else ()),
            *((fp4.shape[0],) if fp4 is not None else ()))
    if j.shape[0] != B:
        j = am.broadcast_to(j, (B,) + j.shape[1:])
    # shared sample-bound gathers (x0, x1) -- identical for both fields
    if xp.shape[0] == 1:
        x0 = xp[0].take(j); x1 = xp[0].take(j + 1); jf = None
    else:
        jf = j + (am.arange(B) * n)[:, None]
        xpr = xp.reshape(-1)
        x0 = xpr.take(jf); x1 = xpr.take(jf + 1)
    inv_slope_arg = (x - x0)                                     # (x - xp[j]); shared
    left = xp[:, :1]; right = xp[:, -1:]
    node_hit = (x == x0)

    def _gather(fp):
        if fp.shape[0] == 1:
            f0 = fp[0].take(j); f1 = fp[0].take(j + 1)
        else:
            jjf = jf if jf is not None else j + (am.arange(B) * n)[:, None]
            fpr = fp.reshape(-1)
            f0 = fpr.take(jjf); f1 = fpr.take(jjf + 1)
        out = ((f1 - f0) / (x1 - x0)) * inv_slope_arg + f0       # EXACT np.interp expression (same grouping)
        out = am.where(node_hit, f0, out)
        out = am.where(x < left, fp[:, :1], out)
        out = am.where(x >= right, fp[:, -1:], out)
        return out

    if fp3 is None:
        return _gather(fp1), _gather(fp2)
    if fp4 is None:
        return _gather(fp1), _gather(fp2), _gather(fp3)
    return _gather(fp1), _gather(fp2), _gather(fp3), _gather(fp4)


# ======================================================================================================
# COLUMN-PARALLEL (numba prange) batched interp -- the CPU-fuse #3 advect+buoy speedup (Task 2b).
#
#   The advect+buoy limb's hot part is the per-column linear interp (_height_to_pressure /
#   _pressure_to_height; ~66% of the limb). The numpy `_interp_columns` is already vectorized, but the
#   per-column work (bracket searchsorted + lerp) is embarrassingly parallel across columns. These numba
#   kernels prange over columns -- a per-column binary search (side='right' semantics, matching np.interp
#   and `_interp_columns`) + the IDENTICAL lerp/clamp value chain, so they are BIT-IDENTICAL to
#   `_interp_columns` in fp64 (pure +,-,*,/ and integer comparisons; no transcendentals, so no SIMD-vs-
#   scalar divergence). Default OFF (the numpy path); ON behind COSMO_ADVECT_PARALLEL. numba is imported
#   LAZILY (only on first ON use) and the kernels are @njit(cache=True) so the JIT is paid once per process.
#
#   Two kernels for the two PRODUCTION shapes (the only ones the advect+buoy interps hit):
#     - SHARED-xp  (`_height_to_pressure`): one sample grid `xp1d` (N,) for all columns, per-column queries
#       x (B, M) and values fp (B, N).  -> out (B, M).
#     - SHARED-x   (`_pressure_to_height`): one ascending query row `x1d` (M,) for all columns, per-column
#       sample grids xp (B, N) and values fp (B, N).  -> out (B, M).
#   Anything else (the generic comparison-count branch, single-row fp broadcast in odd combos) falls back
#   to the numpy `_interp_columns` -- correctness over coverage.
# ======================================================================================================
_INTERP_NUMBA = None        # (shared_xp_kernel, shared_x_kernel), built once on first ON use


def _build_interp_numba():
    """Compile (once) the two prange-over-columns interp kernels. Pure fp64 arithmetic -> bit-identical to
    `_interp_columns` (same bracket, same `(f1-f0)/(x1-x0)*(x-x0)+f0` grouping, same node-hit/clamp order)."""
    from numba import njit, prange   # lazy: only imported when COSMO_ADVECT_PARALLEL is ON

    @njit(cache=True, inline="always")
    def _bisect(arr, n, v):
        # searchsorted(side='right')-1 on the ascending row arr[0:n]: largest j with arr[j] <= v.
        lo = 0
        hi = n
        while lo < hi:
            mid = (lo + hi) // 2
            if arr[mid] <= v:                                   # 'right': advance past equal nodes
                lo = mid + 1
            else:
                hi = mid
        return lo - 1

    @njit(parallel=True, cache=True)
    def _interp_shared_xp(x, xp1d, fp, out):
        # SHARED sample grid xp1d (N,); per-column queries x (B, M) and values fp (B, N). out (B, M).
        B, M = x.shape
        N = xp1d.shape[0]
        x0lo = xp1d[0]
        x0hi = xp1d[N - 1]
        for i in prange(B):                                    # COLUMNS INDEPENDENT -> parallel
            for m in range(M):
                xm = x[i, m]
                if xm < x0lo:                                  # left clamp -> fp[0]
                    out[i, m] = fp[i, 0]
                    continue
                if xm >= x0hi:                                 # right clamp / top sample -> fp[N-1]
                    out[i, m] = fp[i, N - 1]
                    continue
                j = _bisect(xp1d, N, xm)
                if j < 0:
                    j = 0
                elif j > N - 2:
                    j = N - 2
                a0 = xp1d[j]
                if xm == a0:                                   # exact node hit (C short-circuit)
                    out[i, m] = fp[i, j]
                    continue
                a1 = xp1d[j + 1]
                f0 = fp[i, j]
                f1 = fp[i, j + 1]
                out[i, m] = ((f1 - f0) / (a1 - a0)) * (xm - a0) + f0

    @njit(parallel=True, cache=True)
    def _interp_shared_x(x1d, xp, fp, out):
        # SHARED ascending query row x1d (M,); per-column sample grids xp (B, N) and values fp (B, N). out (B, M).
        B, N = xp.shape
        M = x1d.shape[0]
        for i in prange(B):                                    # COLUMNS INDEPENDENT -> parallel
            lo_i = xp[i, 0]
            hi_i = xp[i, N - 1]
            for m in range(M):
                xm = x1d[m]
                if xm < lo_i:                                  # left clamp -> fp[0]
                    out[i, m] = fp[i, 0]
                    continue
                if xm >= hi_i:                                 # right clamp / top sample -> fp[N-1]
                    out[i, m] = fp[i, N - 1]
                    continue
                j = _bisect(xp[i], N, xm)
                if j < 0:
                    j = 0
                elif j > N - 2:
                    j = N - 2
                a0 = xp[i, j]
                if xm == a0:                                   # exact node hit
                    out[i, m] = fp[i, j]
                    continue
                a1 = xp[i, j + 1]
                f0 = fp[i, j]
                f1 = fp[i, j + 1]
                out[i, m] = ((f1 - f0) / (a1 - a0)) * (xm - a0) + f0

    return _interp_shared_xp, _interp_shared_x


def _interp_columns_parallel(x, xp, fp):
    """Numba prange-over-columns drop-in for `_interp_columns` on the two PRODUCTION shapes (shared-xp and
    shared-x). BIT-IDENTICAL in fp64; FALLS BACK to `_interp_columns` for any other shape/dtype combo (so it
    is never silently wrong). Same (x, xp, fp) contract."""
    x = np.asarray(x, float)
    xp = np.asarray(xp, float)
    fp = np.asarray(fp, float)
    B = max(x.shape[0], xp.shape[0], fp.shape[0])
    M = x.shape[-1]
    # SHARED-xp: one sample row, per-column queries + values (the `_height_to_pressure` shape).
    if (xp.shape[0] == 1 and x.shape[0] == B and fp.shape[0] == B
            and x.dtype == np.float64 and xp.dtype == np.float64 and fp.dtype == np.float64):
        global _INTERP_NUMBA
        if _INTERP_NUMBA is None:
            _INTERP_NUMBA = _build_interp_numba()
        out = np.empty((B, M), dtype=np.float64)
        _INTERP_NUMBA[0](np.ascontiguousarray(x), np.ascontiguousarray(xp[0]),
                         np.ascontiguousarray(fp), out)
        return out
    # SHARED-x: one ascending query row, per-column sample grids + values (the `_pressure_to_height` shape).
    if (x.shape[0] == 1 and xp.shape[0] == B and fp.shape[0] == B
            and bool((x[0, 1:] >= x[0, :-1]).all())
            and x.dtype == np.float64 and xp.dtype == np.float64 and fp.dtype == np.float64):
        if _INTERP_NUMBA is None:
            _INTERP_NUMBA = _build_interp_numba()
        out = np.empty((B, M), dtype=np.float64)
        _INTERP_NUMBA[1](np.ascontiguousarray(x[0]), np.ascontiguousarray(xp),
                         np.ascontiguousarray(fp), out)
        return out
    return _interp_columns(x, xp, fp)                          # any other shape -> numpy reference
