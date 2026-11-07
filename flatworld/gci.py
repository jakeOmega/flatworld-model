"""flatworld.gci -- Grid Convergence Index (GCI) automation for the resolution ladders.

The house stance is that *convergence is correctness*: a scalar observable that still moves with the
grid is an artifact, not physics. This module turns a three-rung resolution ladder into a quantitative
verdict instead of an eyeballed "looks converged" -- it implements the Roache Grid Convergence Index
and the Richardson-extrapolation machinery from Celik et al. (2008), "Procedure for Estimation and
Reporting of Uncertainty Due to Discretization in CFD Applications" (ASME J. Fluids Eng. 130, 078001).

Given three grid spacings ``h1 < h2 < h3`` (rung 1 = finest) and one scalar observable ``f1, f2, f3``
it returns:

  * ``p``            -- the OBSERVED (apparent) order of convergence, solved from the three values (not
                        assumed from the scheme's formal order);
  * ``f_exact``      -- the Richardson-extrapolated ``h -> 0`` value;
  * ``gci_fine``     -- the fractional Grid Convergence Index on the finest grid: a *reported numerical
                        uncertainty band*, ``f1 * (1 +/- gci_fine)``, with a built-in factor of safety;
  * ``asymptotic_ratio`` -- ``GCI_32 / (r21**p * GCI_21)``; ~1 means the three rungs sit in the
                        asymptotic range where Richardson extrapolation is valid (Roache's check);
  * ``regime``       -- monotonic / oscillatory / divergent, from the convergence ratio ``R``.

**Scope caveat (read this).** GCI is only meaningful for *converged-family* observables -- ones whose
rung-to-rung change is dominated by DISCRETIZATION error (band width, an edge slope, M_cond, a
detrainment-profile norm). It is meaningless for *placement-type* observables whose spread comes from a
physical choice rather than the grid (theta_max cold-boundary placement, kappa-sensitive band latitude,
a bistable equatorial state). Feeding those a grid ladder error-estimates noise. This module computes
the numbers regardless -- deciding an observable is converged-family is the caller's job (the
`/convergence-study` checklist, step 1). It also does NOT create resolution: three rungs that are not
yet in the asymptotic range (``asymptotic_ratio`` far from 1, or ``regime != 'monotonic'``) give an
UNRELIABLE ``p`` and band -- the module flags that via ``in_asymptotic_range`` / ``regime`` rather than
hiding it.

Pure numpy + math (no scipy); the observed-order solve is a short fixed-point iteration. Runnable and
unit-tested against a known-order synthetic (``tests/test_gci.py``).

    from flatworld.gci import grid_convergence_index
    res = grid_convergence_index(h=[1, 2, 4], f=[1.010, 1.040, 1.160])   # h any order; sorted internally
    print(res.report())
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# Roache's recommended factor of safety. 1.25 when the order is estimated from >= 3 grids (the case
# here); 3.0 is the conservative two-grid value (an assumed order). We always have three rungs.
FS_THREE_GRID = 1.25
FS_TWO_GRID = 3.0

# Tolerance on |asymptotic_ratio - 1| within which the three rungs are declared to be in the
# asymptotic range (Celik et al. treat ~10-15% as acceptable; a real ladder rarely does better).
ASYMPTOTIC_TOL = 0.15

# math.exp(709.8) overflows float64; keep a margin so no ``r ** p`` ever raises OverflowError. A near-
# degenerate refinement ratio (r -> 1, i.e. two rungs almost equally spaced) sends the observed-order
# seed/iteration to a huge |p|; past this exponent bound the triplet is simply not in a convergent
# regime, so we report "no order" (NaN) instead of crashing a print-only ladder block.
_MAX_EXP = 700.0


@dataclass
class GCIResult:
    """The verdict of a three-rung GCI analysis for one scalar observable. All error/GCI fields are
    FRACTIONAL (multiply by 100 for %, or by |f1| for the observable's own units)."""

    # inputs, sorted fine -> coarse (rung 1 = finest)
    h1: float
    h2: float
    h3: float
    f1: float
    f2: float
    f3: float
    r21: float           # refinement ratio h2/h1 (>1)
    r32: float           # refinement ratio h3/h2 (>1)
    fs: float            # factor of safety used

    # results
    p: float             # observed order of convergence
    f_exact: float       # Richardson-extrapolated (h->0) value, from the fine pair
    gci_fine: float      # fractional GCI on grid 1 (the reported uncertainty band)
    gci_coarse: float    # fractional GCI on grid 2 (from the coarse pair) -- used by the asymptotic check
    e_approx: float      # approximate relative error |(f1-f2)/f1|
    e_extrap: float      # extrapolated relative error |(f_exact-f1)/f_exact|
    asymptotic_ratio: float  # GCI_32 / (r21**p * GCI_21); ~1 => asymptotic range
    R: float             # convergence ratio eps21/eps32 (sign+magnitude => regime)
    regime: str          # 'monotonic' | 'oscillatory' | 'divergent' | 'converged'
    in_asymptotic_range: bool

    @property
    def band(self) -> tuple[float, float]:
        """The reported numerical-uncertainty interval on the finest grid: f1 +/- gci_fine*|f1|."""
        half = self.gci_fine * abs(self.f1)
        return (self.f1 - half, self.f1 + half)

    @property
    def band_abs(self) -> float:
        """Half-width of the uncertainty band in the observable's own units (gci_fine*|f1|)."""
        return self.gci_fine * abs(self.f1)

    def report(self, name: str = "observable", unit: str = "") -> str:
        """A one-block human-readable summary line for the ladder scripts."""
        u = f" {unit}" if unit else ""
        if self.regime == "converged":
            return (f"  {name:<20s}: f1={self.f1:.4g}{u}  already converged "
                    f"(finest rungs agree; GCI~0)")
        if not math.isfinite(self.f_exact):
            # p<=0 / non-finite: no valid convergence order, so the band below is the raw fine-pair
            # fallback (fs*e_approx), NOT a Richardson-extrapolated GCI. Label it as such (finding 4).
            note = f"  [!] no valid order [{self.regime}] -- band is a raw fine-pair fallback (fs*e_approx), not a GCI"
        elif not self.in_asymptotic_range:
            note = "  [!] NOT in asymptotic range -- p/band unreliable"
        else:
            note = ""
        return (f"  {name:<20s}: f1={self.f1:.4g}{u}  p={self.p:.2f}  "
                f"GCI={self.gci_fine * 100:.2f}%  (+/-{self.band_abs:.3g}{u})  "
                f"f_exact={self.f_exact:.4g}{u}  R={self.R:+.2f} [{self.regime}]  "
                f"asympt={self.asymptotic_ratio:.2f}{note}")


def refinement_ratio(h_coarse: float, h_fine: float) -> float:
    """The grid refinement ratio r = h_coarse/h_fine (> 1). Raises if not a genuine refinement."""
    r = float(h_coarse) / float(h_fine)
    if not (r > 1.0):
        raise ValueError(f"refinement ratio must be > 1 (h_coarse={h_coarse}, h_fine={h_fine} -> r={r})")
    return r


def observed_order(eps21: float, eps32: float, r21: float, r32: float,
                   max_iter: int = 200, tol: float = 1e-12) -> float:
    """The observed order of convergence p from the two differences and two refinement ratios.

    Solves Celik et al. (2008) eq. (3) by fixed-point iteration:

        p = (1/ln(r21)) * | ln|eps32/eps21| + q(p) |
        q(p) = ln( (r21**p - s) / (r32**p - s) ),   s = sign(eps32/eps21)

    ``q`` vanishes for a constant refinement ratio (r21 == r32), leaving the closed form
    ``p = ln|eps32/eps21| / ln(r21)``. The iteration is seeded from that closed form and converges in a
    few steps for well-behaved ladders. ``eps21 = f2 - f1``, ``eps32 = f3 - f2`` (fine pair, coarse pair).
    """
    if eps21 == 0.0:
        return math.nan                                   # fine pair identical -> order undefined
    ratio = eps32 / eps21
    s = math.copysign(1.0, ratio) if ratio != 0.0 else 1.0
    ln_r21 = math.log(r21)
    ln_r32 = math.log(r32)
    a = abs(ratio)
    if a == 0.0:
        return math.nan

    def _diverged(pp):
        # p so large that ``r ** p`` would overflow float64 -> the iteration is diverging (a
        # near-degenerate refinement ratio r->1 sends the seed to +/-inf). Bail to NaN so the caller
        # takes the graceful "no valid order" path instead of raising OverflowError.
        return (not math.isfinite(pp)) or abs(pp) * ln_r21 > _MAX_EXP or abs(pp) * ln_r32 > _MAX_EXP

    p = math.log(a) / ln_r21                              # constant-r seed
    if _diverged(p):
        return math.nan
    if abs(r21 - r32) < 1e-12:
        return p                                          # exact closed form, no iteration needed
    for _ in range(max_iter):
        d21 = r21 ** p - s
        d32 = r32 ** p - s
        if d32 == 0.0 or d21 == 0.0:
            break                                         # r**p == s (p~0): keep the current iterate
        p_new = abs(math.log(a) + math.log(abs(d21 / d32))) / ln_r21
        if _diverged(p_new):
            return math.nan
        if abs(p_new - p) < tol:
            p = p_new
            break
        p = p_new
    return p


def richardson_value(f1: float, f2: float, r21: float, p: float) -> float:
    """The Richardson-extrapolated ``h -> 0`` estimate from the fine pair: (r21**p f1 - f2)/(r21**p - 1)."""
    rp = r21 ** p
    return (rp * f1 - f2) / (rp - 1.0)


def grid_convergence_index(h, f, fs: float = FS_THREE_GRID, zero_tol: float = 1e-30) -> GCIResult:
    """Full three-rung GCI analysis for one scalar observable.

    Parameters
    ----------
    h : length-3 sequence of grid spacings (any order; sorted internally so rung 1 = finest).
        Use a *representative* spacing -- dx, 1/nx, dt, etc. Only ratios enter, so units/normalization
        cancel; ``gci_from_counts`` is the convenience for grid-point counts.
    f : length-3 sequence of the observable's value at the matching spacing.
    fs : factor of safety (default 1.25, the >=3-grid value).

    Returns a :class:`GCIResult`. Handles the degenerate cases (already-converged pair, oscillatory or
    divergent ladder) without raising -- they are reported through ``regime`` / ``in_asymptotic_range``.
    """
    h = [float(x) for x in h]
    f = [float(x) for x in f]
    if len(h) != 3 or len(f) != 3:
        raise ValueError("grid_convergence_index needs exactly 3 rungs (h and f length 3)")
    # sort by spacing ascending -> (h1,f1) finest ... (h3,f3) coarsest
    order = sorted(range(3), key=lambda i: h[i])
    h1, h2, h3 = (h[i] for i in order)
    f1, f2, f3 = (f[i] for i in order)
    if not (h1 < h2 < h3):
        raise ValueError(f"grid spacings must be distinct (got sorted {h1}, {h2}, {h3})")

    r21 = refinement_ratio(h2, h1)
    r32 = refinement_ratio(h3, h2)
    eps21 = f2 - f1
    eps32 = f3 - f2
    scale = max(abs(f1), abs(f2), abs(f3), 1.0)

    # Convergence ratio (Stern et al.): R = eps21/eps32 (fine-pair change over coarse-pair change).
    if eps32 != 0.0:
        R = eps21 / eps32
    elif eps21 != 0.0:
        R = math.copysign(math.inf, eps21)
    else:
        R = math.nan

    # Fine-pair agreement: the two FINEST rungs are identical to ~zero, so the fine-grid GCI band is
    # ~0 (converged at the fine end). The observed ORDER is undefined here -- there is no fine-pair
    # change to fit a slope to -- so p stays NaN; f_exact defaults to f1. This subsumes the
    # all-three-identical case. A differing coarse rung only says the coarse grid is coarse; it does
    # not falsify the fine-pair agreement.
    if abs(eps21) <= zero_tol * scale:
        return GCIResult(h1, h2, h3, f1, f2, f3, r21, r32, fs,
                         p=math.nan, f_exact=f1, gci_fine=0.0, gci_coarse=0.0,
                         e_approx=0.0, e_extrap=0.0, asymptotic_ratio=math.nan,
                         R=R, regime="converged", in_asymptotic_range=True)

    # Regime from R, sign-first (the SIGN takes precedence over the magnitude):
    #   R < 0     -> oscillatory (an alternating sequence, whether damping |R|<1 or growing |R|>1);
    #   0 < R < 1 -> monotonic convergence (the fine grid is closest to the limit);
    #   R >= 1    -> monotonic divergence (refining makes it WORSE -- no useful estimate).
    if R < 0.0:
        regime = "oscillatory"
    elif R < 1.0:
        regime = "monotonic"
    else:
        regime = "divergent"

    p = observed_order(eps21, eps32, r21, r32)
    if not math.isfinite(p) or p <= 0.0:
        # no valid order (oscillatory/divergent or noise): report what we can, band from the raw pair
        e_approx = abs((f1 - f2) / f1) if f1 != 0.0 else abs(f1 - f2) / scale
        return GCIResult(h1, h2, h3, f1, f2, f3, r21, r32, fs,
                         p=(p if math.isfinite(p) else math.nan),
                         f_exact=math.nan, gci_fine=fs * e_approx, gci_coarse=math.nan,
                         e_approx=e_approx, e_extrap=math.nan, asymptotic_ratio=math.nan,
                         R=R, regime=regime, in_asymptotic_range=False)

    f_exact = richardson_value(f1, f2, r21, p)
    e_approx = abs((f1 - f2) / f1) if f1 != 0.0 else abs(f1 - f2) / scale
    e_extrap = abs((f_exact - f1) / f_exact) if f_exact != 0.0 else math.nan
    gci_fine = fs * e_approx / (r21 ** p - 1.0)

    # coarse-pair GCI and the asymptotic-range check: GCI_32 / (r21**p * GCI_21) ~ 1.
    e_approx_32 = abs((f2 - f3) / f2) if f2 != 0.0 else abs(f2 - f3) / scale
    gci_coarse = fs * e_approx_32 / (r32 ** p - 1.0)
    denom = (r21 ** p) * gci_fine
    asymptotic_ratio = gci_coarse / denom if denom != 0.0 else math.nan
    in_asymptotic = (regime == "monotonic"
                     and math.isfinite(asymptotic_ratio)
                     and abs(asymptotic_ratio - 1.0) <= ASYMPTOTIC_TOL)

    return GCIResult(h1, h2, h3, f1, f2, f3, r21, r32, fs,
                     p=p, f_exact=f_exact, gci_fine=gci_fine, gci_coarse=gci_coarse,
                     e_approx=e_approx, e_extrap=e_extrap, asymptotic_ratio=asymptotic_ratio,
                     R=R, regime=regime, in_asymptotic_range=in_asymptotic)


def gci_from_counts(n, f, length: float = 1.0, **kw) -> GCIResult:
    """GCI convenience for grid-POINT-COUNT ladders (nx, nz, ne): spacing h_i = length/n_i.

    Only ratios of h enter the GCI, so ``length`` is optional (it cancels) -- pass the physical domain
    size if you want ``h`` in real units for the report, else leave it 1. Example::

        gci_from_counts(n=[512, 256, 128], f=[w512, w256, w128])   # nx ladder
    """
    n = [float(x) for x in n]
    h = [length / x for x in n]
    return grid_convergence_index(h, f, **kw)


def gci_table(rungs, observables, spacing_key, **kw) -> dict:
    """Run GCI for several observables off a list of per-rung result dicts (the ladder-script shape).

    ``rungs`` is a list of dicts (>=3), each with a ``spacing_key`` entry (dx, dt, or a count) and the
    observable keys. ``observables`` maps a display name -> the dict key (and optional unit) to analyse.
    Uses the three FINEST rungs when more than three are present. Returns {name: GCIResult}; a
    non-finite/degenerate observable is skipped with a ``None`` value so a caller can note it.

        gci_table(rows, {"band width [deg]": "width_deg", "cold edge [deg]": "theta_cold"}, "dx_km")
    """
    rr = sorted(rungs, key=lambda d: float(d[spacing_key]))   # ascending spacing = fine..coarse
    fine3 = rr[:3]                                             # the three finest rungs
    h = [float(d[spacing_key]) for d in fine3]
    out = {}
    for name, key in observables.items():
        try:
            f = [float(d[key]) for d in fine3]
            if not all(math.isfinite(v) for v in f) or not all(math.isfinite(v) for v in h):
                out[name] = None
                continue
            out[name] = grid_convergence_index(h, f, **kw)
        except (KeyError, ValueError, ZeroDivisionError, OverflowError, ArithmeticError):
            # a malformed/degenerate observable must never crash a print-only ladder block appended to
            # expensive compute -- report it as skipped, not raised (finding 1 belt-and-suspenders).
            out[name] = None
    return out


def format_table(results: dict, unit: str = "") -> str:
    """Format a {name: GCIResult|None} mapping (from ``gci_table``) into a printable block."""
    lines = ["  -- Grid Convergence Index (fine-grid GCI = reported numerical-uncertainty band) --"]
    for name, res in results.items():
        if res is None:
            lines.append(f"  {name:<20s}: (skipped -- non-finite / degenerate observable)")
        else:
            lines.append(res.report(name=name, unit=unit))
    return "\n".join(lines)


if __name__ == "__main__":
    # tiny self-demo on the Celik et al. (2008) worked example (their Table, p from a 1D grid triplet):
    #   h1<h2<h3 with r=1.5/... ; here a clean second-order synthetic f = 2 + 3*h**2.
    demo_h = [1.0, 2.0, 4.0]
    demo_f = [2.0 + 3.0 * hh ** 2 for hh in demo_h]
    res = grid_convergence_index(demo_h, demo_f)
    print("self-demo: f = 2 + 3 h^2  (exact p=2, f_exact=2)")
    print(res.report(name="demo f", unit=""))
