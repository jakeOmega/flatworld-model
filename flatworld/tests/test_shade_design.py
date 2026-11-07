"""Gates for the lantern-shade design tool (``_shade_design``): the kernel must reproduce the
hand-probe contrast numbers (the noon-lingering ~2.5x at ±2° slats), the inversion must respect the
transmission bounds and the ribbon noon cap, and the linear structure must hold. Coarse grids."""
import numpy as np

from flatworld._shade_design import ALPHA_EDGES, CANDIDATES, design, kernel

THETA = np.linspace(0.5, 84.5, 36)


def test_kernel_positive_and_lingering_contrast():
    K = kernel(theta_deg=THETA, n=1501)
    assert K.shape == (len(THETA), len(ALPHA_EDGES) - 1)
    assert np.all(K >= 0.0) and np.isfinite(K).all()
    centers = 0.5 * (ALPHA_EDGES[:-1] + ALPHA_EDGES[1:])
    i_band = np.abs(THETA - 45.0).argmin()
    th_i = THETA[i_band]
    j0 = int(np.searchsorted(ALPHA_EDGES, th_i + 0.5))     # slats just ABOVE the observer (alpha >= theta)
    sl = slice(j0, j0 + 3)
    band = K[i_band, sl].sum()
    eq = K[0, sl].sum()
    assert 1.5 < band / eq < 4.0                           # the stationary-phase enhancement
    # geometry theorem: light at alpha < theta NEVER reaches theta (alpha >= theta always)
    i70 = np.abs(THETA - 70.0).argmin()
    low = centers < THETA[i70] - 1.5
    assert np.all(K[i70, low] == 0.0)


def test_design_bounds_cap_and_linearity():
    K = kernel(theta_deg=THETA, n=1201)
    res = design(K=K, theta=THETA, noon_cap=1400.0, **CANDIDATES["D_monotone_ribbon"])
    assert np.all(res["shade"] >= 0.0) and np.all(res["shade"] <= 1.0 + 1e-9)
    assert res["I0_mult"] > 0 and np.isfinite(res["residual"])
    assert res["noon_band"] <= 1400.0 * 1.05               # the habitability cap holds
    assert res["T_eq"] > 100.0                             # the furnace strip delivers boiling
    # linearity: doubling T doubles S
    T = res["shade"] * res["I0_mult"]
    assert np.allclose(K @ (2 * T), 2 * (K @ T))


def test_bright_recipe_tracks_committed_canonical_d4():
    """The D4_bright provenance recipe must regenerate the AUTHORITATIVE committed canonical-world
    shade to within tolerance (the exact lsq design is not bit-recoverable, so this guards the recipe
    from silently drifting away from the fixture the sh_b23L8 checkpoint was run under -- it does NOT
    assert bit-identity; the committed npz stays the source of truth)."""
    import os

    import pytest

    from flatworld._shade_design import BRIGHT_CANDIDATES, BRIGHT_FILES, design, kernel
    committed = os.path.join("m33_runs", BRIGHT_FILES["D4_bright"] + ".npz")
    if not os.path.exists(committed):
        pytest.skip("authoritative shade fixture not present (run from cosmology_models/)")
    ref = np.load(committed)
    res = design(K=kernel(), noon_cap=None, **BRIGHT_CANDIDATES["D4_bright"])   # full default grid
    T_eff = res["shade"] * res["I0_mult"]
    assert T_eff.shape == np.asarray(ref["T_eff"]).shape
    assert abs(res["I0_mult"] - float(ref["I0_mult"])) < 0.02 * float(ref["I0_mult"])   # I0 within 2 %
    assert np.max(np.abs(T_eff - np.asarray(ref["T_eff"]))) < 0.4                       # T_eff close
