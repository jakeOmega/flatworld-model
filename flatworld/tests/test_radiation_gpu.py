"""GPU (CuPy) parity tests for the array-module-generic gray-band radiation (Phase D port).

Two guarantees the radiation port must hold:
  * numpy path BIT-IDENTICAL -- the ported ``np.->xp.`` substitution is the SAME ops on numpy, so the
    CPU regression is untouched. We assert ``array_equal`` (atol=0, rtol=0) of the ported functions vs an
    in-test PRISTINE reimplementation of the load-bearing recurrence (``two_stream_lw``), and lean on the
    existing ``test_radiation.py`` suite + the committed ``archive/_gpu_rad_port_RESULTS.md`` snapshot proof for the
    full-surface byte-identity.
  * CuPy path correct -- the ported clear-sky path runs on a CuPy array with NO host-wall error; fp64 GPU
    matches numpy to machine precision (~1e-13); fp32 GPU (under the ``set_thermo_dtype`` gate, mirroring
    production) matches fp64 numpy to single-call round-off (~1e-6) and FOLLOWS the input dtype (fp32 in ->
    fp32 out).

The GPU tests SKIP cleanly via ``device.gpu_available()`` on a CPU-only box (the portable-gate pattern from
``test_device_gpu.py``), so this file is a no-op everywhere except a CuPy+CUDA session.

SCOPE / BOUNDARY: only the THERMO-FREE radiation surface is GPU-tested -- ``two_stream_lw``,
``two_stream_lw_windowed``, ``shortwave``, the optical-depth builders, and CLEAR-SKY ``column_radiation``
(clouds=False, lw_window=False). The ``clouds`` and ``lw_window`` paths transit ``flatworld.thermo``
(``qsat``/``vapor_pressure_from_q``/``esat``), which is NOT yet array-module-generic, so they still raise on
CuPy input; that boundary is asserted explicitly (``test_thermo_dependent_paths_raise_on_gpu``) so a future
thermo GPU port flips these from raise to parity. See ``archive/_gpu_rad_port_RESULTS.md``.
"""
import numpy as np
import pytest

from flatworld import device as dev
from flatworld import radiation as R
from flatworld import thermo as th
from flatworld import constants as C

gpu = pytest.mark.skipif(not dev.gpu_available(),
                         reason="no CuPy/CUDA device (optional GPU backend: requirements-gpu.txt)")


# ------------------------------------------------------------------- fixtures (CPU-safe; built from numpy)
def _inputs(nx=48, N=40, seed=20260622):
    """A representative fp64 column chain (nx columns x N layers, TOP->BOTTOM)."""
    rng = np.random.default_rng(seed)
    p_edge = np.broadcast_to(np.linspace(200.0, C.P_SURF, N + 1), (nx, N + 1)).copy()
    T_layers = 220.0 + 80.0 * rng.random((nx, N))
    T_surf = 280.0 + 40.0 * rng.random(nx)
    q = np.abs(rng.standard_normal((nx, N))) * 1e-3 + 1e-5
    S_toa = 300.0 + 100.0 * rng.random(nx)
    alpha_surf = 0.05 + 0.2 * rng.random(nx)
    dtau = np.abs(rng.standard_normal((nx, N))) * 0.3
    dtau_win = np.abs(rng.standard_normal((nx, N))) * 0.1
    S_diffuse = 40.0 + 10.0 * rng.random(nx)
    return dict(p_edge=p_edge, T_layers=T_layers, T_surf=T_surf, q=q, S_toa=S_toa,
                alpha_surf=alpha_surf, dtau=dtau, dtau_win=dtau_win, S_diffuse=S_diffuse)


def _two_stream_lw_pristine(dtau, T_layers, T_surf, eps_surf=1.0, D=R.D_DIFF, band_frac=1.0):
    """A faithful PRE-PORT numpy reimplementation of the two-stream recurrence (bare ``np.``), to anchor
    the numpy byte-identity of the ported ``two_stream_lw`` independently of the committed snapshot."""
    dtau = np.asarray(dtau, np.float64)
    T_layers = np.asarray(T_layers, np.float64)
    *batch, N = dtau.shape
    t = np.exp(-D * dtau)
    e = 1.0 - t
    B = band_frac * C.SIGMA_SB * T_layers ** 4
    Bs = band_frac * C.SIGMA_SB * np.asarray(T_surf, np.float64) ** 4
    eB = e * B
    Dn = np.zeros((*batch, N + 1), eB.dtype)
    for k in range(N):
        Dn[..., k + 1] = Dn[..., k] * t[..., k] + eB[..., k]
    D_surf = Dn[..., N]
    U_surf = eps_surf * Bs + (1.0 - eps_surf) * D_surf
    Up = np.zeros((*batch, N + 1), eB.dtype)
    Up[..., N] = U_surf
    for k in range(N - 1, -1, -1):
        Up[..., k] = Up[..., k + 1] * t[..., k] + eB[..., k]
    Fnet = Up - Dn
    H = Fnet[..., 1:] - Fnet[..., :-1]
    return dict(OLR=Up[..., 0], D_surf=D_surf, Fnet=Fnet, H=H, surf_net_lw=Bs - D_surf)


# ------------------------------------------------------------------- numpy BIT-IDENTITY (CPU-safe, always runs)
def test_numpy_two_stream_bit_identical_to_pristine():
    """The ported ``two_stream_lw`` on numpy is BIT-IDENTICAL (array_equal, atol=0 rtol=0) to the pre-port
    bare-numpy recurrence -- the np.->xp. substitution is the same ops on numpy."""
    inp = _inputs()
    got = R.two_stream_lw(inp["dtau"], inp["T_layers"], inp["T_surf"], eps_surf=0.95, band_frac=0.3)
    ref = _two_stream_lw_pristine(inp["dtau"], inp["T_layers"], inp["T_surf"], eps_surf=0.95, band_frac=0.3)
    for k in ref:
        assert got[k].dtype == ref[k].dtype, f"{k}: dtype drift {ref[k].dtype} -> {got[k].dtype}"
        assert np.array_equal(got[k], ref[k]), f"{k}: numpy output not byte-identical"


def test_numpy_clearsky_column_radiation_dtype_fp64():
    """Clear-sky ``column_radiation`` on a numpy fp64 input stays fp64 throughout (no silent cast)."""
    inp = _inputs()
    out = R.column_radiation(inp["p_edge"], inp["T_layers"], inp["q"], inp["T_surf"],
                             inp["S_toa"], inp["alpha_surf"], sw_bands=True, sw_diffuse=True,
                             S_diffuse=inp["S_diffuse"])
    for k in ("H", "surf_net", "toa_net", "OLR", "dtau_lw", "dtau_sw"):
        assert np.asarray(out[k]).dtype == np.float64, f"{k}: not fp64 ({np.asarray(out[k]).dtype})"


# ------------------------------------------------------------------------------ CuPy parity (GPU only)
def _relerr(host_ref, dev_out):
    a = np.asarray(host_ref, float)
    b = dev.to_host(dev_out).astype(float)
    return float(np.abs(a - b).max() / (np.abs(a).max() + 1e-30))


@gpu
def test_clearsky_fp64_gpu_matches_numpy_machine_precision():
    """fp64 CuPy clear-sky radiation == numpy to machine precision (~1e-13), with the result genuinely on
    the device. Covers the whole thermo-free surface: the two LW two-streams, shortwave (two-band +
    diffuse), and clear-sky column_radiation."""
    import cupy as cp
    inp = _inputs(nx=96, N=60)
    g = lambda a: cp.asarray(a)

    ref_lw = R.two_stream_lw(inp["dtau"], inp["T_layers"], inp["T_surf"])
    ref_lww = R.two_stream_lw_windowed(inp["dtau"], inp["dtau_win"], inp["T_layers"], inp["T_surf"])
    ref_sw = R.shortwave(inp["S_toa"], inp["dtau"] * 0.02, inp["alpha_surf"], f_vis=0.5, S_diffuse=inp["S_diffuse"])
    ref_cr = R.column_radiation(inp["p_edge"], inp["T_layers"], inp["q"], inp["T_surf"], inp["S_toa"],
                                inp["alpha_surf"], sw_bands=True, sw_diffuse=True, S_diffuse=inp["S_diffuse"])

    out_lw = R.two_stream_lw(g(inp["dtau"]), g(inp["T_layers"]), g(inp["T_surf"]))
    out_lww = R.two_stream_lw_windowed(g(inp["dtau"]), g(inp["dtau_win"]), g(inp["T_layers"]), g(inp["T_surf"]))
    out_sw = R.shortwave(g(inp["S_toa"]), g(inp["dtau"] * 0.02), g(inp["alpha_surf"]), f_vis=0.5,
                         S_diffuse=g(inp["S_diffuse"]))
    out_cr = R.column_radiation(g(inp["p_edge"]), g(inp["T_layers"]), g(inp["q"]), g(inp["T_surf"]),
                                g(inp["S_toa"]), g(inp["alpha_surf"]), sw_bands=True, sw_diffuse=True,
                                S_diffuse=g(inp["S_diffuse"]))

    # genuinely on the device (the port did not silently fall back to host)
    assert dev.is_device(out_lw["OLR"]) and dev.is_device(out_cr["H"])

    for ref, out, name in ((ref_lw, out_lw, "two_stream_lw"), (ref_lww, out_lww, "two_stream_lw_windowed"),
                           (ref_sw, out_sw, "shortwave"), (ref_cr, out_cr, "column_radiation")):
        for k in ref:
            if k not in out:
                continue
            e = _relerr(ref[k], out[k])
            assert e < 1e-11, f"{name}.{k}: fp64 GPU-vs-numpy rel {e:.3e} exceeds 1e-11"


@gpu
def test_clearsky_fp32_gpu_follows_dtype_and_matches_to_roundoff():
    """fp32 CuPy clear-sky radiation, under the ``set_thermo_dtype(np.float32)`` gate (production's path),
    FOLLOWS the input dtype (fp32 in -> fp32 out) and matches fp64 numpy to single-call round-off (~1e-6)."""
    import cupy as cp
    inp = _inputs(nx=96, N=60)

    ref_lw = R.two_stream_lw(inp["dtau"], inp["T_layers"], inp["T_surf"])
    ref_cr = R.column_radiation(inp["p_edge"], inp["T_layers"], inp["q"], inp["T_surf"], inp["S_toa"],
                                inp["alpha_surf"], sw_bands=True, sw_diffuse=True, S_diffuse=inp["S_diffuse"])

    g32 = lambda a: cp.asarray(a, dtype=cp.float32)
    prev = th.set_thermo_dtype(np.float32)
    try:
        out_lw = R.two_stream_lw(g32(inp["dtau"]), g32(inp["T_layers"]), g32(inp["T_surf"]))
        out_cr = R.column_radiation(g32(inp["p_edge"]), g32(inp["T_layers"]), g32(inp["q"]),
                                    g32(inp["T_surf"]), g32(inp["S_toa"]), g32(inp["alpha_surf"]),
                                    sw_bands=True, sw_diffuse=True, S_diffuse=g32(inp["S_diffuse"]))
    finally:
        th.set_thermo_dtype(prev)

    # dtype-following: fp32 input -> fp32 output
    assert out_lw["OLR"].dtype == cp.float32, f"fp32 in did not yield fp32 out ({out_lw['OLR'].dtype})"
    assert out_cr["H"].dtype == cp.float32

    for k in ref_lw:
        e = _relerr(ref_lw[k], out_lw[k])
        assert e < 5e-5, f"two_stream_lw.{k}: fp32 GPU-vs-fp64-numpy rel {e:.3e} exceeds 5e-5"
    for k in ("H", "surf_net", "toa_net", "OLR", "dtau_lw", "dtau_sw", "surf_abs_sw", "planetary_reflected"):
        e = _relerr(ref_cr[k], out_cr[k])
        assert e < 5e-5, f"column_radiation.{k}: fp32 GPU-vs-fp64-numpy rel {e:.3e} exceeds 5e-5"


@gpu
def test_thermo_dependent_paths_match_on_gpu():
    """BOUNDARY CLOSED: the cloud / LW-window paths transit ``flatworld.thermo`` (qsat /
    vapor_pressure_from_q), which is now array-module-generic (thermo GPU port merged), so they run on
    CuPy and MATCH numpy to machine precision. This previously asserted a ``TypeError`` raise (the honest
    pre-thermo-port boundary); the thermo port flipped it from raise to parity, and we verify the values
    are correct on-device, not merely that the host wall is gone."""
    import cupy as cp
    inp = _inputs(nx=8, N=20)
    g = lambda a: cp.asarray(a)
    np_args = (inp["p_edge"], inp["T_layers"], inp["q"], inp["T_surf"], inp["S_toa"], inp["alpha_surf"])
    cp_args = tuple(g(a) for a in np_args)
    for kwargs, label in (({"lw_window": True}, "lw_window"), ({"clouds": True}, "clouds")):
        ref = R.column_radiation(*np_args, **kwargs)
        out = R.column_radiation(*cp_args, **kwargs)
        assert dev.is_device(out["H"]), f"{label}: column_radiation silently fell back to host"
        for k in ref:
            if k not in out or not hasattr(ref[k], "shape"):
                continue
            e = _relerr(ref[k], out[k])
            assert e < 1e-11, f"column_radiation[{label}].{k}: fp64 GPU-vs-numpy rel {e:.3e} exceeds 1e-11"
