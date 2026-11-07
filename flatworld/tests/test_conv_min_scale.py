"""PHY-19 remedy (b) -- the MINIMUM HORIZONTAL SCALE filter on the convective tendency.

What these tests defend, in the order the merge bar asks for them:

  1. **OFF is byte-identical.** The flag defaults to 0.0, and a stepped toy world with the flag absent,
     with it explicitly 0.0, are bit-for-bit the same run over the WHOLE prognostic state. (The
     against-``origin/main`` half of that gate is a cross-worktree hash quoted in the PR body -- a test
     inside the branch cannot import the pre-change module.)
  2. **ON conserves.** Column enthalpy ``sum c_p dx dm dT`` and column water ``sum dx dm dq`` survive the
     filter to round-off, per level, on a random tendency -- at the operator level AND through the core's
     own wiring.
  3. **ON is EXACTLY a no-op outside the active set** -- bitwise, not merely small.
  4. **The wall reflects**, and the reflection is the exact fold of the mirrored domain.
  5. **The width is in METRES**, so two grids with the same ``W`` smooth the same physical distance.
  6. **The fingerprint records the key**, so a checkpoint cannot hide which side of the switch it ran on.
  7. **numpy and CuPy agree** (behind the repo's GPU skip idiom).

Terminology, on first use: ``W`` is the minimum horizontal scale in metres (``conv_min_scale_m``, env
``COSMO_CONV_MIN_SCALE_M``); the ACTIVE SET is the columns whose cell width is below ``W``; ``dT``/``dq``
are the change in temperature [K] and specific humidity [kg/kg] the convective adjustment makes across
one call.
"""

import hashlib
import json
import warnings

import numpy as np
import pytest

from flatworld import conv_min_scale as cms
from flatworld import device as _dev
from flatworld.stage_ad_core import StageADConfig, StageADCore

gpu = pytest.mark.skipif(not _dev.gpu_available(), reason="no CUDA device / CuPy")

W30 = 30.0e3            # the value fixed on 2026-09-17 HST, before any run read it (note section 4.1)


# =================================================================================================
# The pure operator (flatworld/conv_min_scale.py) -- no core, no config, no device
# =================================================================================================
def _stretched(dx_min=2.0e3, n_fine=24, ratio=1.05, n=300, dx_max=4.0e6):
    """A production-shaped cross-track grid: ``n_fine`` cells of exactly ``dx_min`` at the x = 0 axis,
    then a geometric transition capped at ``dx_max``. Returns ``(x_centres, dx_widths)`` in metres."""
    w = [float(dx_min)] * int(n_fine)
    while len(w) < int(n):
        w.append(min(w[-1] * float(ratio), float(dx_max)))
    w = np.array(w[:int(n)], float)
    edges = np.concatenate([[0.0], np.cumsum(w)])
    return 0.5 * (edges[:-1] + edges[1:]), w


def test_the_operator_conserves_the_mass_weighted_column_sum_to_round_off():
    """The defining property: ``sum_i dx_i A_ij == dx_j``, which is what makes
    ``sum_i dx_i dm_ik filt(d)_ik == sum_i dx_i dm_ik d_ik`` exact per level. Checked directly on a
    RANDOM tendency, not only on the column sums, so a sign or transpose error cannot slip through."""
    x, dx = _stretched()
    n, A = cms.build_min_scale_operator(x, dx, W30)
    assert n > 0
    rng = np.random.default_rng(0)
    d = rng.standard_normal((n, 40))                    # a random (active columns, levels) tendency
    filt = A @ d
    before = dx[:n] @ d                                 # per level, the mass-weighted area total
    after = dx[:n] @ filt
    assert np.max(np.abs(after - before)) <= 1e-13 * np.max(np.abs(before))


def test_the_normalization_is_last_and_the_build_asserts_it():
    """The build-order gate. ``COLSUM_RTOL`` is asserted inside the builder, so this pins the number it
    actually reaches -- and pins the trade in the other direction: the alternative that keeps
    uniform-invariance instead (row sums of 1) misses conservation by orders of magnitude more."""
    x, dx = _stretched()
    n, A = cms.build_min_scale_operator(x, dx, W30)
    d = cms.operator_diagnostics(A, dx, W30)
    assert d["colsum_resid"] < 1e-14                     # round-off, per the note's 4.55e-16
    # the row-normalized alternative, on the same grid and the same kernel
    xa = x[:n]
    K = (np.maximum(0.0, 1.0 - np.abs(xa[:, None] - xa[None, :]) / W30)
         + np.maximum(0.0, 1.0 - np.abs(xa[:, None] + xa[None, :]) / W30))
    A_row = K / K.sum(axis=1)[:, None]                   # row sums exactly 1 -> uniform-invariant
    resid_row = np.max(np.abs(dx[:n] @ A_row - dx[:n]) / dx[:n])
    assert resid_row > 1e-3, "the trade should be visible: row-normalizing loses conservation"
    assert np.allclose(A_row.sum(axis=1), 1.0)           # ... and this is what it buys instead


def test_the_operator_is_not_uniform_invariant_and_says_so():
    """The property the design deliberately trades away, measured on the BUILT operator (it depends on
    the reflection detail, so it must be measured rather than assumed): a horizontally uniform tendency
    comes out reshaped into a slight gradient, bounded and small."""
    x, dx = _stretched()
    n, A = cms.build_min_scale_operator(x, dx, W30)
    rows = A.sum(axis=1)
    assert not np.allclose(rows, 1.0, atol=1e-12)        # NOT uniform-invariant ...
    assert np.all(np.abs(rows - 1.0) < 0.05)             # ... but bounded well inside a few percent
    inner = rows[: min(24, n)]                           # the constant-2 km cells the feature lives in
    assert np.all(np.abs(inner - 1.0) < 2e-3)


def test_the_operator_is_exactly_the_identity_where_cells_are_at_least_W_wide():
    """The no-op claim is EXACT, not small: the operator is block-diagonal, so a column at least ``W``
    wide neither gives nor receives. Pinned by construction (the operator has no entries there at all)
    and by the touched-column count on a production-shaped grid."""
    x, dx = _stretched(n=975)
    n, A = cms.build_min_scale_operator(x, dx, W30)
    assert A.shape == (n, n)                             # nothing outside the active block exists
    assert np.all(dx[:n] < W30) and np.all(dx[n:] >= W30)
    assert 60 < n < 100                                  # ~79 of 975 on the production grid (note 4.1)
    # the range endpoints bracket it and still stop far short of the whole grid
    n10, _ = cms.build_min_scale_operator(x, dx, 10.0e3)
    n60, _ = cms.build_min_scale_operator(x, dx, 60.0e3)
    assert n10 < n < n60 < 200


def test_the_wall_reflection_is_the_exact_fold_of_the_mirrored_domain():
    """x = 0 is a psi = 0 SYMMETRY wall, so the stencil must behave as though the domain continues as
    its own mirror. The check is the fold itself: build the UNREFLECTED kernel on the doubled domain
    (-x_n..-x_1, x_1..x_n), apply it to a symmetric pair of deltas, fold the halves together, and
    require the reflected one-sided operator to reproduce it."""
    x, dx = _stretched(n=120)
    n, A = cms.build_min_scale_operator(x, dx, W30)
    xa, dxa = x[:n], dx[:n]
    xm = np.concatenate([-xa[::-1], xa])                 # mirrored centres, ascending
    dxm = np.concatenate([dxa[::-1], dxa])
    Km = np.maximum(0.0, 1.0 - np.abs(xm[:, None] - xm[None, :]) / W30)   # NO reflection term
    Am = Km * (dxm / (dxm @ Km))[None, :]
    for k in (0, 1, 3, 8):                               # columns at and near the wall
        d = np.zeros(2 * n)
        d[n + k] = 1.0                                   # the delta ...
        d[n - 1 - k] = 1.0                               # ... and its mirror image, equally weighted
        folded = (Am @ d)[n:]                            # the response on the physical half
        one_sided = A[:, k]                              # the reflected operator's response to one delta
        assert np.allclose(one_sided, folded, rtol=0, atol=1e-12), f"k={k}"


def test_the_width_is_in_metres_and_the_spread_is_grid_independent():
    """``W`` is a physical length, never a cell count. Two grids whose axis spacing differs by 2x must
    spread a delta to the same PHYSICAL half-width -- within one cell, which is all a discrete grid can
    promise."""
    half_widths = []
    for dx_min, n_fine in ((1.0e3, 48), (2.0e3, 24)):
        x, dx = _stretched(dx_min=dx_min, n_fine=n_fine, n=400)
        n, A = cms.build_min_scale_operator(x, dx, W30)
        j = int(np.searchsorted(x[:n], 100.0e3))         # a column well inside the active block ...
        assert 0 < j < n - 1                             # ... and far from the wall
        resp = A[:, j]
        peak = resp.max()
        assert abs(x[int(np.argmax(resp))] - x[j]) <= dx[j]     # the peak sits on the delta
        # INTERPOLATED half-max crossing on each flank -- a nearest-cell read would quantize the
        # answer by the local spacing, which is the very thing this test is trying to vary
        xa = x[:n]                                       # the response lives on the ACTIVE block only
        lo = np.interp(0.5 * peak, resp[: j + 1], xa[: j + 1])
        hi = np.interp(0.5 * peak, resp[j:][::-1], xa[j:][::-1])
        half_widths.append(0.5 * (hi - lo))
    assert abs(half_widths[0] - half_widths[1]) < 2.0e3         # same physical spread, within a cell
    for hw in half_widths:
        assert abs(hw - 0.5 * W30) < 1.0e3                      # ... and it IS the triangle's W/2


def test_a_grid_coarser_than_W_everywhere_builds_no_operator_at_all():
    """On a uniform grid whose cells already exceed ``W`` the term has nothing to do, and the builder
    says so by returning no operator -- an exact no-op, not a matrix that happens to be the identity."""
    dx = np.full(40, 50.0e3)
    x = (np.arange(40) + 0.5) * 50.0e3
    n, A = cms.build_min_scale_operator(x, dx, W30)
    assert n == 0 and A is None
    assert "exactly the identity everywhere" in cms.describe(A, dx, W30)


def test_the_builder_refuses_a_grid_whose_fine_cells_are_not_a_leading_block():
    """The fixed-buffer apply assumes the active columns are a contiguous PREFIX (finest cells at the
    x = 0 axis, as every grid in this family is). A grid that violates it must fail loudly at build
    time, not silently filter the wrong columns."""
    dx = np.array([50.0e3, 2.0e3, 2.0e3, 50.0e3])
    x = np.cumsum(dx) - 0.5 * dx
    with pytest.raises(ValueError, match="contiguous prefix"):
        cms.build_min_scale_operator(x, dx, W30)


# =================================================================================================
# The core wiring (StageADCore) -- the gate, conservation through the real buffers, the OFF path
# =================================================================================================
def _core(minscale=None, nx=48, nz=10, seed=3, **extra):
    kw = dict(nx=nx, nz=nz, theta_max_deg=80.0, x_grid_type="stretched", x_dx_min=2.0e3, x_n_fine=12,
              prognostic_condensate=True, prognostic_rain=True, conv_tau=7200.0)
    kw.update(extra)
    if minscale is not None:
        kw["conv_min_scale_m"] = minscale
    rng = np.random.default_rng(seed)
    core = StageADCore(StageADConfig(**kw))
    core.run_dynamics = True
    core.couple_heat = True
    core.T_atm = core.T_atm + 0.7 * rng.standard_normal(core.T_atm.shape)
    core.q = np.clip(core.q + 1e-4 * rng.standard_normal(core.q.shape), 0.0, 0.5)
    return core


def _state_hash(core):
    h = hashlib.sha256()
    for name in ("T_atm", "q", "q_c", "q_r", "b", "psi", "omega", "T_surf", "W_surf", "S_dep"):
        a = getattr(core, name, None)
        if a is None:
            continue
        arr = np.ascontiguousarray(np.asarray(a))
        h.update(name.encode()); h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode()); h.update(arr.tobytes())
    return h.hexdigest()


def _stepped_hash(minscale, steps=5):
    core = _core(minscale)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(steps):
            core.step()
    return _state_hash(core)


def test_the_flag_defaults_off_and_off_is_byte_identical(monkeypatch):
    """The merge bar. The flag defaults to 0.0, and a five-step run with the flag ABSENT from the config
    is bit-for-bit a run with it explicitly 0.0 -- over the whole prognostic state, not just T."""
    monkeypatch.delenv("COSMO_CONV_MIN_SCALE_M", raising=False)
    assert StageADConfig(nx=8, nz=6).conv_min_scale_m == 0.0
    assert _stepped_hash(None) == _stepped_hash(0.0)


def test_the_env_key_reaches_the_config(monkeypatch):
    monkeypatch.setenv("COSMO_CONV_MIN_SCALE_M", "30000")
    assert StageADConfig(nx=8, nz=6).conv_min_scale_m == pytest.approx(30.0e3)


def test_the_fingerprint_records_the_key():
    """A checkpoint must not be able to hide which side of the switch it ran on. The fingerprint's
    physics set is INVERTED (everything not named volatile is physics), so this rides that for free --
    which is exactly why it needs a test rather than a claim."""
    core = _core(W30, nx=24, nz=6)
    fp = json.loads(core.fingerprint())
    assert fp["config"]["conv_min_scale_m"] == pytest.approx(W30)
    from flatworld import _ckpt_fingerprint as ckfp
    assert "conv_min_scale_m" not in ckfp.VOLATILE_FIELDS        # i.e. a difference is reported


def test_the_core_builds_the_operator_and_reports_what_it_touches(capsys):
    core = _core(W30)
    assert core._minsc_n > 0
    assert np.all(np.asarray(core._dx_cell)[core._minsc_n:] >= W30)
    line = capsys.readouterr().out
    assert "[PHY-19]" in line and "EXACTLY the identity elsewhere" in line


def test_the_gray_zone_gate_combination_is_refused_at_init():
    """Under the gray-zone gate the Betts-Miller blend fraction is an (nx, 1) PER-COLUMN array and the
    telescoping argument for filtering a tendency before it has not been checked (note 5(b) step 3).
    Refuse at init rather than run an unverified conservation claim."""
    with pytest.raises(NotImplementedError, match="gray-zone"):
        _core(W30, conv_dx_off=4.0e3, conv_dx_full=40.0e3)


def _filter_probe(core, rng):
    """Drive ``_apply_min_scale_filter`` on a hand-made tendency -> ``(T_in, q_in, T_out, q_out)``, the
    whole-domain (nx, nz) STATE each side of the filter (states, not tendencies: ``(Tpre + dT) - Tpre``
    is not bit-for-bit ``dT``, so a bitwise no-op claim has to be made on the state the filter saw)."""
    nx, nz = core.T_atm.shape
    Tpre = np.array(core.T_atm, float)
    qpre = np.clip(np.array(core.q, float), 0.0, 0.4)
    core._conv_Tpre = Tpre.copy()
    core._conv_qpre = qpre.copy()
    core.T_atm = Tpre + rng.standard_normal((nx, nz)) * 0.5
    q = qpre + rng.standard_normal((nx, nz)) * 1e-4
    T_in, q_in = core.T_atm.copy(), q.copy()
    core._apply_min_scale_filter(q)
    return T_in, q_in, np.asarray(core.T_atm).copy(), q.copy()


def test_on_conserves_column_enthalpy_and_water_to_round_off_through_the_core():
    """Conservation through the REAL buffers and the real grid weights, not only the pure operator:
    ``sum_i dx_i C_layer_k dT_ik`` (enthalpy [J/m]) and ``sum_i dx_i dm_k dq_ik`` (water [kg/m]) are
    unchanged per level. ``dm`` is horizontally uniform here, so it factors out -- the point of the test
    is that the DX weights and the operator agree."""
    core = _core(W30)
    T_in, q_in, T_out, q_out = _filter_probe(core, np.random.default_rng(11))
    Tpre, qpre = np.asarray(core._conv_Tpre), np.asarray(core._conv_qpre)
    dT0, dq0 = T_in - Tpre, q_in - qpre                   # the tendency the adjustment produced ...
    dT1, dq1 = T_out - Tpre, q_out - qpre                 # ... and the filtered one
    dx = np.asarray(core._dx_cell, float)[:, None]
    for before, after, label in ((dT0, dT1, "enthalpy"), (dq0, dq1, "water")):
        b = (dx * before).sum(axis=0)                     # (nz,) per-level area total
        a = (dx * after).sum(axis=0)
        assert np.max(np.abs(a - b)) <= 1e-12 * np.max(np.abs(b)), label
    # column enthalpy and water as single scalars, with the real per-level weights
    C_layer = np.asarray(core.C_layer, float)[None, :]
    dm = np.asarray(core.dm, float)[None, :]
    assert (dx * C_layer * dT1).sum() == pytest.approx((dx * C_layer * dT0).sum(), rel=1e-12)
    assert (dx * dm * dq1).sum() == pytest.approx((dx * dm * dq0).sum(), rel=1e-12)


def test_on_is_bitwise_a_no_op_outside_the_active_set():
    """EXACT, not small: every column at least ``W`` wide keeps the adjustment's own tendency
    bit-for-bit. This is what lets the term be turned on without touching the habitable band, where
    cells are hundreds of km wide."""
    core = _core(W30)
    n = core._minsc_n
    T_in, q_in, T_out, q_out = _filter_probe(core, np.random.default_rng(12))
    assert np.array_equal(T_in[n:], T_out[n:])
    assert np.array_equal(q_in[n:], q_out[n:])
    assert not np.allclose(T_in[:n], T_out[:n])           # ... and it DID something inside


def test_on_spreads_a_one_column_tendency_over_at_least_W():
    """The term's whole purpose, at the core level: a tendency confined to the single 2 km axis-side
    column comes out spread over a physical width of order ``W``, and the area total is unchanged."""
    core = _core(W30)
    nx, nz = core.T_atm.shape
    n = core._minsc_n
    j = 6                                                  # a 2 km cell, inside the fine block
    Tpre = np.array(core.T_atm, float)
    qpre = np.clip(np.array(core.q, float), 0.0, 0.4)
    dT = np.zeros((nx, nz)); dT[j, :] = 1.0
    core._conv_Tpre = Tpre.copy(); core._conv_qpre = qpre.copy()
    core.T_atm = Tpre + dT
    q = qpre.copy()
    core._apply_min_scale_filter(q)
    out = np.asarray(core.T_atm)[:, 0] - Tpre[:, 0]
    dx = np.asarray(core._dx_cell, float)
    assert (dx * out).sum() == pytest.approx((dx * dT[:, 0]).sum(), rel=1e-12)
    x = np.asarray(core.x, float)
    above = np.flatnonzero(out >= 0.5 * out.max())
    assert x[above[-1]] - x[above[0]] >= 0.8 * W30         # full width at half max, order W
    assert out.max() < 0.2 * dT[j, 0]                      # the one-column peak is gone
    assert np.count_nonzero(out) > 10 and np.all(out[n:] == 0.0)


@gpu
def test_numpy_and_cupy_agree_on_the_filtered_tendency():
    """Device parity, through the path the core actually takes. CuPy's cuBLAS binding REFUSES calls
    during CUDA-graph capture, so the device product goes through this package's capture-safe tiled
    GEMM RawKernel (``_gpu_kernels.gemm_device``, already shipped for the stretched-Poisson transforms)
    rather than ``cupy.matmul`` -- this checks that kernel reproduces numpy on the filter's shapes."""
    import cupy as cp

    from flatworld._gpu_kernels import build_gemm_kernel, gemm_device

    x, dx = _stretched(n=200)
    n, A = cms.build_min_scale_operator(x, dx, W30)
    rng = np.random.default_rng(7)
    d = rng.standard_normal((n, 32))
    host = A @ d
    out = cp.empty((n, 32), dtype=cp.float64)
    gemm_device(build_gemm_kernel(np.float64), cp.ascontiguousarray(cp.asarray(A)),
                cp.ascontiguousarray(cp.asarray(d)), out)
    assert np.allclose(cp.asnumpy(out), host, rtol=0, atol=1e-11 * np.max(np.abs(host)))


def test_a_post_build_width_change_takes_effect():
    """The documented warm-start route sets ``cfg.conv_min_scale_m`` on a BUILT core (a COSMO_* key the
    run17 checkpoint never recorded is refused by the checkpoint world guard). The call site re-reads
    the width, so the operator is rebuilt on the next eager substep."""
    core = _core(0.0)
    assert core._minsc_n == 0
    core.cfg.conv_min_scale_m = W30                   # exactly the line the config comment documents
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        core._thermo_substep(60.0)
    assert core._minsc_n > 0 and core._minsc_W == pytest.approx(W30)


def test_the_warm_start_route_cannot_walk_around_the_gray_zone_refusal():
    """The gate refusal must live where the operator is BUILT, not only in ``__init__`` -- otherwise the
    documented warm-start route (set the width on a built core) reaches the combination the init
    refuses."""
    core = _core(None, conv_dx_off=4.0e3, conv_dx_full=40.0e3)
    assert core._conv_gate_col is not None            # the gate really engaged on this grid
    core.cfg.conv_min_scale_m = W30
    with pytest.raises(NotImplementedError, match="gray-zone"):
        core._thermo_substep(60.0)
