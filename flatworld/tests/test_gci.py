"""Tests for flatworld.gci -- the Grid Convergence Index automation (ROADMAP INF-4).

The gate is a KNOWN-ORDER SYNTHETIC: build a ladder from ``f(h) = f_exact + C*h**p`` for a chosen order
``p`` and refinement, then assert the module recovers ``p``, recovers ``f_exact`` by Richardson
extrapolation, brackets ``f_exact`` inside the reported GCI band, and reads the asymptotic-range /
regime flags correctly. Synthetics are exact power laws so the recovered order is analytic, not fitted.
Fast (pure arithmetic)."""
import math

import numpy as np
import pytest

from flatworld import gci


def synthetic(h, f_exact, C, p):
    """A single-term power-law observable f(h) = f_exact + C h^p, evaluated on the spacings ``h``."""
    return [f_exact + C * hh ** p for hh in h]


# ---------------------------------------------------------------------------------------------------
# observed order + Richardson extrapolation on a known-order synthetic
# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("p_true", [1.0, 1.5, 2.0, 3.0, 4.0])
@pytest.mark.parametrize("r", [2.0, 1.5])
def test_recovers_known_order_constant_r(p_true, r):
    """Constant refinement ratio: a pure power law of order p_true is recovered analytically (the
    constant-r closed form p = ln|eps32/eps21|/ln r), and Richardson recovers f_exact exactly."""
    h = [1.0, r, r * r]                                    # fine -> coarse, constant ratio r
    f_exact, C = 100.0, 0.1
    f = synthetic(h, f_exact, C, p_true)
    res = gci.grid_convergence_index(h, f)
    assert res.p == pytest.approx(p_true, abs=1e-9)
    assert res.f_exact == pytest.approx(f_exact, rel=1e-9)
    assert res.regime == "monotonic"


@pytest.mark.parametrize("p_true", [1.3, 2.0, 2.7])
def test_recovers_order_nonconstant_r(p_true):
    """Non-constant refinement ratio (r21 != r32) exercises the fixed-point q(p) iteration; a pure
    power law is still recovered to tight tolerance."""
    h = [1.0, 1.5, 3.0]                                    # r21=1.5, r32=2.0
    f_exact, C = 50.0, 0.05
    f = synthetic(h, f_exact, C, p_true)
    res = gci.grid_convergence_index(h, f)
    assert res.r21 == pytest.approx(1.5) and res.r32 == pytest.approx(2.0)
    assert res.p == pytest.approx(p_true, abs=1e-6)
    assert res.f_exact == pytest.approx(f_exact, rel=1e-8)


def test_richardson_value_direct():
    """richardson_value inverts the fine pair exactly given the true order."""
    p, r21, f_exact, C = 2.0, 2.0, 10.0, 0.5
    f1 = f_exact + C * 1.0 ** p
    f2 = f_exact + C * 2.0 ** p
    assert gci.richardson_value(f1, f2, r21, p) == pytest.approx(f_exact, rel=1e-12)


# ---------------------------------------------------------------------------------------------------
# the GCI band: a reported numerical-uncertainty interval that brackets the exact value
# ---------------------------------------------------------------------------------------------------
def test_gci_band_brackets_exact():
    """For a monotone, well-resolved second-order ladder the reported band f1 +/- GCI*|f1| contains
    the Richardson f_exact (the factor of safety 1.25 is exactly there to guarantee this)."""
    h = [1.0, 2.0, 4.0]
    f_exact, C, p = 100.0, 0.1, 2.0
    res = gci.grid_convergence_index(h, synthetic(h, f_exact, C, p))
    lo, hi = res.band
    assert lo <= f_exact <= hi
    # the band half-width equals the fractional GCI times |f1|
    assert res.band_abs == pytest.approx(res.gci_fine * abs(res.f1), rel=1e-12)


def test_finer_grid_shrinks_gci():
    """Refining the whole ladder (same order, smaller spacings) shrinks the reported GCI band ~r^p."""
    f_exact, C, p = 20.0, 0.2, 2.0
    coarse = gci.grid_convergence_index([1.0, 2.0, 4.0], synthetic([1.0, 2.0, 4.0], f_exact, C, p))
    fine = gci.grid_convergence_index([0.25, 0.5, 1.0], synthetic([0.25, 0.5, 1.0], f_exact, C, p))
    assert fine.gci_fine < coarse.gci_fine
    # both recover the same order and limit
    assert fine.p == pytest.approx(coarse.p, abs=1e-9)
    assert fine.f_exact == pytest.approx(f_exact, rel=1e-9)


# ---------------------------------------------------------------------------------------------------
# asymptotic-range check
# ---------------------------------------------------------------------------------------------------
def test_asymptotic_range_true_for_small_error():
    """When the discretization error is a small fraction of the value (f1~f2~f3), the Roache check
    GCI_32/(r^p GCI_21) sits near 1 and the ladder is flagged in the asymptotic range."""
    h = [1.0, 2.0, 4.0]
    res = gci.grid_convergence_index(h, synthetic(h, 1000.0, 0.05, 2.0))
    assert res.asymptotic_ratio == pytest.approx(1.0, abs=gci.ASYMPTOTIC_TOL)
    assert res.in_asymptotic_range


def test_asymptotic_range_false_for_large_error():
    """A ladder with huge relative discretization error (coarse grids far from the limit) is correctly
    flagged as NOT yet asymptotic even though the order is recovered -- the reported band is large and
    the check reads well below 1 (Celik's relative-error normalization)."""
    h = [1.0, 2.0, 4.0]
    res = gci.grid_convergence_index(h, synthetic(h, 2.0, 3.0, 2.0))   # error dwarfs f_exact
    assert res.p == pytest.approx(2.0, abs=1e-9)                       # order still exact
    assert not res.in_asymptotic_range
    assert res.gci_fine > 0.1                                          # and the band is honestly large


# ---------------------------------------------------------------------------------------------------
# convergence regimes
# ---------------------------------------------------------------------------------------------------
def test_monotonic_convergence_regime():
    h = [1.0, 2.0, 4.0]
    res = gci.grid_convergence_index(h, synthetic(h, 5.0, 0.1, 2.0))
    assert res.regime == "monotonic"
    assert 0.0 < res.R < 1.0


def test_oscillatory_regime():
    """Alternating rung-to-rung changes -> R<0 -> oscillatory, regardless of magnitude; not asymptotic."""
    res = gci.grid_convergence_index([1.0, 2.0, 4.0], [10.0, 10.5, 10.2])
    assert res.regime == "oscillatory"
    assert res.R < 0.0 and not res.in_asymptotic_range


def test_divergent_regime():
    """Refining makes it WORSE (fine grid farthest from the limit) -> R>=1 -> divergent; no f_exact."""
    res = gci.grid_convergence_index([1.0, 2.0, 4.0], [13.5, 11.0, 10.0])
    assert res.regime == "divergent"
    assert res.R >= 1.0 and not res.in_asymptotic_range
    assert math.isnan(res.f_exact)


def test_already_converged():
    """Identical values across rungs -> exactly resolved: regime 'converged', zero GCI, no NaNs in band."""
    res = gci.grid_convergence_index([1.0, 2.0, 4.0], [7.0, 7.0, 7.0])
    assert res.regime == "converged"
    assert res.gci_fine == 0.0 and res.f_exact == 7.0
    assert res.band == (7.0, 7.0)


def test_fine_pair_agreement_is_converged():
    """The two FINEST rungs agree exactly while the coarse one differs -> the fine-grid band is ~0, so
    report 'converged' (not a misleading 'monotonic, GCI 0.00%' with an undefined order). Regression for
    review finding 3."""
    res = gci.grid_convergence_index([1.0, 2.0, 4.0], [10.0, 10.0, 11.0])
    assert res.regime == "converged"
    assert res.gci_fine == 0.0 and res.f_exact == 10.0
    assert "converged" in res.report("m")


def test_no_overflow_on_near_degenerate_ratio():
    """A near-degenerate refinement ratio (r21 ~ 1) sends the observed-order iteration toward a huge p;
    the module must report 'no valid order' (NaN) rather than raise OverflowError from r**p, so a
    print-only ladder block cannot crash an expensive run. Regression for review finding 1 (its exact
    reproducer), through both the direct call and the gci_table retrofit path."""
    h, f = [60.0, 60.5, 240.0], [1.0, 21.0, 25.0]              # r21 = 60.5/60 ~ 1.008, r32 ~ 3.97
    res = gci.grid_convergence_index(h, f)                     # must not raise
    assert math.isnan(res.p) and math.isnan(res.f_exact)
    assert not res.in_asymptotic_range
    out = gci.gci_table([dict(dt=h[i], m=f[i]) for i in range(3)], {"m": "m"}, "dt")
    assert out["m"] is not None and math.isnan(out["m"].p)     # reported, not raised
    # observed_order itself bails to NaN on the divergent iteration
    assert math.isnan(gci.observed_order(20.0, 4.0, 1.008, 3.97))


def test_fallback_band_is_labelled():
    """When there is no valid convergence order, the reported band is the raw fine-pair fallback
    (fs*e_approx), and report() says so rather than mislabelling it a GCI. Regression for finding 4."""
    res = gci.grid_convergence_index([1.0, 2.0, 4.0], [13.5, 11.0, 10.0])   # divergent -> no order
    assert math.isnan(res.f_exact)
    txt = res.report("w")
    assert "not a GCI" in txt and "no valid order" in txt


# ---------------------------------------------------------------------------------------------------
# API robustness
# ---------------------------------------------------------------------------------------------------
def test_sort_order_invariance():
    """The result is independent of the order the three rungs are passed in (sorted internally)."""
    h, f = [1.0, 2.0, 4.0], synthetic([1.0, 2.0, 4.0], 100.0, 0.1, 2.0)
    a = gci.grid_convergence_index(h, f)
    b = gci.grid_convergence_index(list(reversed(h)), list(reversed(f)))
    assert a.p == pytest.approx(b.p) and a.f_exact == pytest.approx(b.f_exact)
    assert a.gci_fine == pytest.approx(b.gci_fine)


def test_gci_from_counts_matches_direct():
    """gci_from_counts(n) == grid_convergence_index(1/n): only spacing ratios enter."""
    n, f = [512, 256, 128], [9.20, 9.24, 9.35]
    a = gci.gci_from_counts(n, f)
    b = gci.grid_convergence_index([1.0 / x for x in n], f)
    assert a.p == pytest.approx(b.p) and a.gci_fine == pytest.approx(b.gci_fine)
    # a physical domain length only rescales the reported h, never the (ratio-based) verdict
    c = gci.gci_from_counts(n, f, length=8.0e7)
    assert c.p == pytest.approx(a.p) and c.gci_fine == pytest.approx(a.gci_fine)


def test_refinement_ratio_validation():
    with pytest.raises(ValueError):
        gci.refinement_ratio(1.0, 2.0)                    # coarse/fine must be > 1
    with pytest.raises(ValueError):
        gci.grid_convergence_index([2.0, 2.0, 4.0], [1.0, 2.0, 3.0])   # duplicate spacing


def test_wrong_length_raises():
    with pytest.raises(ValueError):
        gci.grid_convergence_index([1.0, 2.0], [1.0, 2.0])


def test_factor_of_safety_scales_band():
    """The reported GCI scales linearly with the factor of safety."""
    h, f = [1.0, 2.0, 4.0], synthetic([1.0, 2.0, 4.0], 100.0, 0.1, 2.0)
    a = gci.grid_convergence_index(h, f, fs=1.25)
    b = gci.grid_convergence_index(h, f, fs=3.0)
    assert b.gci_fine == pytest.approx(a.gci_fine * (3.0 / 1.25), rel=1e-12)


# ---------------------------------------------------------------------------------------------------
# gci_table -- the ladder-script retrofit shape (list of per-rung dicts)
# ---------------------------------------------------------------------------------------------------
def test_gci_table_uses_three_finest_and_skips_bad():
    """gci_table pulls the three FINEST rungs from a list of per-rung dicts, analyses each named
    observable, and returns None for a non-finite/degenerate one instead of raising."""
    rows = [
        dict(dx_km=64.0, width=9.9, edge=np.nan),          # coarsest -> dropped (only 3 finest used)
        dict(dx_km=32.0, width=9.5, edge=52.0),
        dict(dx_km=16.0, width=9.42, edge=52.3),
        dict(dx_km=8.0, width=9.40, edge=52.4),            # finest
    ]
    out = gci.gci_table(rows, {"width": "width", "edge": "edge"}, "dx_km")
    assert set(out) == {"width", "edge"}
    # width is finite across the 3 finest -> a real result recovering a positive order
    assert out["width"] is not None and out["width"].regime in ("monotonic", "oscillatory", "converged", "divergent")
    # edge is finite across the 3 finest rungs (the NaN was in the dropped coarsest) -> also a result
    assert out["edge"] is not None
    # a table whose observable is NaN in the retained window is skipped, not raised
    bad = gci.gci_table(rows[:3] + [dict(dx_km=8.0, width=9.40, edge=np.nan)],
                        {"edge": "edge"}, "dx_km")
    assert bad["edge"] is None


def test_report_strings_render():
    """report()/format_table produce strings for each regime without raising (used by the ladders)."""
    h = [1.0, 2.0, 4.0]
    good = gci.grid_convergence_index(h, synthetic(h, 100.0, 0.1, 2.0))
    conv = gci.grid_convergence_index(h, [7.0, 7.0, 7.0])
    assert "GCI=" in good.report("width", "deg")
    assert "converged" in conv.report("width")
    block = gci.format_table({"width": good, "edge": None, "flat": conv}, unit="deg")
    assert "width" in block and "skipped" in block
