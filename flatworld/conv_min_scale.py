"""flatworld.conv_min_scale -- the PHY-19 §5(b) MINIMUM HORIZONTAL SCALE filter for the convective
adjustment's tendency.

WHAT IT IS. A conservative, mass-weighted horizontal smoothing applied to the *tendency* the moist
convective adjustment produces (its change in temperature ``dT`` and vapour ``dq`` across the
adjustment call), with a width ``W`` in METRES. It does not change *how much* convection happens; it
changes *how narrow* the heating and drying are allowed to be.

WHY. On the production stretched-x grid the sun-line axis cells are 2 km wide and the convective
adjustment is column-by-column, so the scheme is free to put an entire convecting column's heating
and drying into one 2 km cell -- which it does (``PHY19_AXIS_CONVECTION.md`` §2: a 12-16x peak/strip
rain ratio at a one-cell half-max width, against cap 64's 1.5-3 at 26-33 cells). A 2-D slab cannot
represent the two things that stop a real convecting column from being one cell wide: the COLD-POOL
OUTFLOW that spreads from a downdraft and seeds convection at a distance, and the ALONG-LINE BREAKUP
of a line updraft into discrete cells across the missing third dimension. Both put a FLOOR on the
horizontal scale over which convective heating and drying are delivered, and both land at tens of
kilometres (§4.1). ``W`` is that floor. It was fixed at **30 km** (range 10-60 km) on 2026-09-17 HST,
before any run read it.

THE OPERATOR, and the properties it keeps and trades away.

Per level ``k``, the filtered tendency is the MASS-WEIGHTED smoothing

    filt(d)_ik = ( A @ (dm_.k * d_.k) )_i / dm_ik

with a fixed ``A`` built once from the cell centres. With the column-sum normalization below,

    sum_i dx_i * dm_ik * filt(d)_ik  ==  sum_i dx_i * dm_ik * d_ik      EXACTLY, per level,

so the total column enthalpy ``sum c_p dx dm dT`` and total water ``sum dx dm dq`` are conserved to
round-off. Two consequences that belong in the code, not in a later discovery:

  * **Per-COLUMN conservation is deliberately broken.** Moving adjustment work between columns is the
    entire point of the term.
  * **The operator is conservative but NOT uniform-invariant.** Row sums are not 1 on a stretched
    grid, so a horizontally uniform tendency is reshaped into a slight gradient. The bound is
    measured on the built operator (``operator_diagnostics``) and reported at init, because it
    depends on the reflection detail. This is the same pair of properties the tracer-advection
    mass-consistency work had to choose between; this design picks CONSERVATION and bounds
    uniform-invariance.

BUILD ORDER IS LOAD-BEARING -- NORMALIZE LAST. The extent decision (which columns the operator spans)
is made FIRST and the normalization LAST, because normalizing a kernel and *then* changing its extent
leaves the column sums it was normalized for. The note measured that error at
``max|sum_i dx_i A_ij - dx_j| / dx_j = 1.59e-2`` against ``4.55e-16`` the right way round; that figure
is the note's measurement of the note's own wrong order and is not re-derived here. What IS asserted
here, at build time and on every grid, is the right-order result: ``COLSUM_RTOL`` below, which the
built operator meets at ~2e-16 on a production-like grid. The trade this order makes is real and
measurable in the other direction: the alternative that keeps uniform-invariance instead
(row-normalize to row sums of 1) misses conservation by **1.5e-1** on that same grid.

ONE CORRECTION TO THE NOTE'S SKETCH (§5(b) step 2.2). The note says to zero every ROW with
``dx_i >= W`` and put 1 on its diagonal. That alone does not conserve: an ACTIVE row ``i`` may still
reach an INACTIVE column ``j`` within ``W``, and column ``j`` then sums to ``dx_j`` (from its own
identity row) PLUS ``dx_i A_ij`` -- a leak of exactly the kind the build-order table is about. The
operator therefore has to be **exactly block-diagonal**: a dense conservative block on the active
columns, the identity everywhere else, and no entry crossing between them. The cutoff radius is then
a NO-FLUX boundary on BOTH sides of the active block (at the ``x = 0`` symmetry wall it is physical;
at the outer edge it is the choice the note names). Built this way the no-op outside the active set
is EXACT and not merely small.

THE WALL. ``x = 0`` is a psi = 0 symmetry wall (``stage_ad_core`` header item #5), so the stencil
REFLECTS: the mirror image of column ``j`` sits at ``-x_j`` with the same width, and its contribution
is added back inside. Physically the domain continues as its own mirror, so a cold pool spreading
"through" the axis is a cold pool spreading from the other side.

GRID INDEPENDENCE. ``W`` is in METRES, never cells. Two grids with the same ``W`` smooth the same
physical distance, and a delta's response is a triangle of half-width ``W`` whatever the spacing.

This module is PURE: numpy only, no core, no config, no device. The core builds the operator once at
construction and applies it with ``xp`` ops; everything here is testable without a world.
"""
from __future__ import annotations

import numpy as np

# The column-sum residual the built operator must meet. The right build order lands at round-off
# (the note measured 4.55e-16 on the production grid); this bound is loose enough for a big grid and
# far below the 1.59e-2 the wrong order produces, so it cannot pass a normalize-first operator.
COLSUM_RTOL = 1e-12


def active_count(dx_widths, W):
    """How many leading columns the filter touches: those with cell width ``dx_i < W``.

    STRICTLY ``<``: a column whose cell is already at least ``W`` wide cannot be narrower than the
    minimum scale, so the filter has nothing to do there and must be EXACTLY the identity (§5(b)
    step 5). Raises if the active columns are not a contiguous PREFIX -- on this world's grids the
    cells are finest at the ``x = 0`` axis and widen monotonically outward, and the whole
    fixed-buffer apply below assumes a contiguous leading block."""
    dx = np.asarray(dx_widths, float)
    act = dx < float(W)
    n = int(np.count_nonzero(act))
    if n and not np.all(act[:n]):
        bad = np.flatnonzero(~act[:n])
        raise ValueError(
            f"the min-scale filter's active columns must be a contiguous prefix of the grid "
            f"(finest cells at the x=0 axis), but columns {bad[:5].tolist()} inside the leading "
            f"{n} are already >= W = {float(W):g} m. This grid is not the stretched-x family the "
            f"term was designed for; build the operator on an explicit index set before using it.")
    return n


def build_min_scale_operator(x_centers, dx_widths, W):
    """``(n_act, A)`` -- the filter's active column count and its dense ``(n_act, n_act)`` fp64
    operator, or ``(0, None)`` when the grid is everywhere coarser than ``W`` (an exact no-op).

    ``x_centers`` and ``dx_widths`` are the cell CENTRES and TRUE widths in metres (the core's
    ``self.x`` and ``self._dx_cell``; on a uniform grid pass ``np.full(nx, dx)``). ``W`` is the
    minimum horizontal scale in METRES.

    Build order, exactly as §5(b) requires and with the block correction above:

      1. form the raw triangular kernel AND its reflection about ``x = 0``, on the ACTIVE BLOCK only
         (so no entry ever crosses the active/inactive boundary -- the identity outside is exact);
      2. normalize LAST, scaling each COLUMN so that ``sum_i dx_i A_ij == dx_j``.

    The column scaling is what makes the operator conservative; it is applied after the extent is
    fixed, never before."""
    x = np.asarray(x_centers, float)
    dx = np.asarray(dx_widths, float)
    W = float(W)
    if x.ndim != 1 or dx.shape != x.shape:
        raise ValueError(f"x_centers {x.shape} and dx_widths {dx.shape} must be the same 1-D shape")
    if not np.all(np.isfinite(x)) or not np.all(dx > 0.0):
        raise ValueError("x_centers must be finite and dx_widths strictly positive")
    if not (W > 0.0):
        raise ValueError(f"W must be > 0 m (got {W:g}); W = 0 is the OFF path and is a PYTHON-level "
                         f"branch in the core, never a zero-width operator")
    n = active_count(dx, W)
    if n == 0:
        return 0, None                      # nothing narrower than W anywhere -> exactly the identity

    xa, dxa = x[:n], dx[:n]
    d_direct = np.abs(xa[:, None] - xa[None, :])          # |x_i - x_j|
    d_mirror = np.abs(xa[:, None] + xa[None, :])          # |x_i - (-x_j)|, the image across the wall
    #   both distances use abs() so the kernel is well defined even if a centre were placed at x < 0
    K = np.maximum(0.0, 1.0 - d_direct / W) + np.maximum(0.0, 1.0 - d_mirror / W)
    # NORMALIZE LAST: one scale per COLUMN so that sum_i dx_i A_ij == dx_j exactly.
    colsum = dxa @ K                                       # (n,) -- sum_i dx_i K_ij
    if not np.all(colsum > 0.0):
        raise ValueError("a min-scale kernel column summed to zero -- W is smaller than a cell "
                         "spacing, which cannot happen for active columns (dx_i < W)")
    A = K * (dxa / colsum)[None, :]
    resid = np.max(np.abs(dxa @ A - dxa) / dxa)
    if not resid < COLSUM_RTOL:
        raise AssertionError(f"min-scale operator column sums off by {resid:.3e} (> {COLSUM_RTOL:g}) "
                             f"-- the normalization is not last, or the block is not closed")
    return n, A


def operator_diagnostics(A, dx_widths, W, n_act=None):
    """``dict`` of the numbers the init log and the design note quote for a BUILT operator:
    the conservation residual, the uniform-invariance bound (row sums, which are NOT 1 on a
    stretched grid), and the physical span the active block covers.

    These are measured on the operator that will actually run, never assumed -- the uniform-invariance
    bound in particular depends on the reflection detail (§5(b) step 3)."""
    dx = np.asarray(dx_widths, float)
    if A is None:
        return dict(n_act=0, span_m=0.0, colsum_resid=0.0, rowsum_min=1.0, rowsum_max=1.0)
    n = int(A.shape[0]) if n_act is None else int(n_act)
    dxa = dx[:n]
    rows = A.sum(axis=1)
    return dict(n_act=n,
                span_m=float(dxa.sum()),
                colsum_resid=float(np.max(np.abs(dxa @ A - dxa) / dxa)),
                rowsum_min=float(rows.min()),
                rowsum_max=float(rows.max()))


def describe(A, dx_widths, W, n_act=None, nx=None):
    """One human-readable init line (the driver prints its configuration on line 1, OPERATIONS §3.9).
    Every abbreviation is spelled out: this is read by a person, not parsed."""
    d = operator_diagnostics(A, dx_widths, W, n_act=n_act)
    total = int(np.asarray(dx_widths).size if nx is None else nx)
    if d["n_act"] == 0:
        return (f"conv_min_scale_m = {float(W):g} m: NO column is narrower than W on this grid "
                f"-> the filter is exactly the identity everywhere (a built no-op)")
    return (f"conv_min_scale_m = {float(W):g} m: the minimum-horizontal-scale filter touches "
            f"{d['n_act']} of {total} columns, spanning {d['span_m'] / 1e3:.0f} km, and is EXACTLY "
            f"the identity elsewhere; column-sum residual (the conservation gate) "
            f"{d['colsum_resid']:.2e}; row sums (uniform-invariance, NOT 1 on a stretched grid) "
            f"{d['rowsum_min']:.4f}-{d['rowsum_max']:.4f}")
