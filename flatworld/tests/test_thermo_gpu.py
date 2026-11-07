"""GPU (Phase D) device-backend tests for the column thermodynamics / surface elementwise leaf-math
(``thermo.py`` + ``surface.py``).

The Phase-D port made the elementwise thermo/surface helpers array-module-generic (numpy or CuPy): a
CuPy array input runs on the GPU with NO host round-trip, while a numpy caller stays BYTE-IDENTICAL to
the pre-port code (the COSMO_THERMO_FP32 dtype gate, #196, is orthogonal and preserved).

Two layers, mirroring ``test_device_gpu.py``:
  * a numpy bit-identical regression that runs on ANY box (the port must not move a single CPU bit);
  * GPU parity tests guarded by ``@gpu`` -- they SKIP cleanly when CuPy/a CUDA device is absent.

GPU targets (the prompt's minimum + the elementwise dependency chain they reach):
  thermo: esat, qsat, desat_dT, dqsat_dT, moist_adiabat_lapse_rate, saturation_mse, mse,
          virtual_temperature, density_temperature, vapor_pressure_from_q,
          lifting_condensation_level,
          sensible_heat_flux, latent_heat_flux;  surface: turbulent_fluxes, radiative_mean_temperature.
HOST-ONLY (Python-scalar integrators, NOT ported, NOT on the AD-core hot path):
  thermo.temperature_on_moist_adiabat, surface.solve_periodic_surface.
"""
import numpy as np
import pytest

from flatworld import device as dev
from flatworld import thermo as th
from flatworld import surface as S

gpu = pytest.mark.skipif(not dev.gpu_available(),
                         reason="no CuPy/CUDA device (optional GPU backend: requirements-gpu.txt)")


# ----------------------------------------------------------------------- representative fp64 inputs
def _inputs(xp):
    """Representative thermo inputs on the given array module (numpy or CuPy)."""
    rng = np.random.default_rng(7)
    T = rng.uniform(230.0, 360.0, 200)
    p = rng.uniform(2.0e4, 1.05e5, 200)
    z = rng.uniform(0.0, 1.5e4, 200)
    q = rng.uniform(1.0e-4, 3.0e-2, 200)
    return (xp.asarray(T), xp.asarray(p), xp.asarray(z), xp.asarray(q))


def _all_outputs(T, p, z, q):
    """Every ported elementwise thermo/surface output, keyed by name. Works on numpy OR CuPy inputs."""
    o = {}
    o["esat"] = th.esat(T)
    o["qsat"] = th.qsat(T, p)
    o["desat_dT"] = th.desat_dT(T)
    o["dqsat_dT"] = th.dqsat_dT(T, p)
    o["moist_lapse"] = th.moist_adiabat_lapse_rate(T, p)
    o["sat_mse"] = th.saturation_mse(T, p, z)
    o["mse"] = th.mse(T, p, z, q)
    o["vtemp"] = th.virtual_temperature(T, q, 0.5 * q)
    o["dtemp"] = th.density_temperature(T, q, 0.5 * q)          # PHY-19 condensate loading
    # PHY-20: the EXACT specific-humidity branch T(1 + 0.608 q_v - q_c). A separate Python branch, so it
    # needs its own parity entry -- an array-module bug there would not show in the default form.
    o["vtemp_exact"] = th.virtual_temperature(T, q, 0.5 * q, exact=True)
    o["dtemp_exact"] = th.density_temperature(T, q, 0.5 * q, exact=True)
    o["vpfromq"] = th.vapor_pressure_from_q(q, p)
    plcl, tlcl = th.lifting_condensation_level(T, 0.5 * th.qsat(T, p), p)
    o["plcl"] = plcl
    o["tlcl"] = tlcl
    o["shf"] = th.sensible_heat_flux(T, T - 5.0, 5.0)
    o["lhf"] = th.latent_heat_flux(T, q, p, 5.0)
    SH, LE = S.turbulent_fluxes(T, T - 5.0, q, p, 5.0, 0.5)
    o["turb_SH"] = SH
    o["turb_LE"] = LE
    return o


# ------------------------------------------------------ numpy bit-identical regression (CPU-safe)
def test_numpy_path_unchanged_by_get_xp():
    """The get_xp-ification must be a no-op on numpy: ``_F(x, np)`` == ``np.asarray(x, float64)``, and
    every ``xp.<op>`` on a numpy array is the bare ``np.<op>``. Sanity that the helpers return numpy and
    finite values for a numpy caller (the byte-for-byte SHA proof lives in the port RESULTS doc)."""
    T, p, z, q = _inputs(np)
    out = _all_outputs(T, p, z, q)
    for name, v in out.items():
        v = np.asarray(v)
        assert v.dtype == np.float64, f"{name}: numpy path lost float64 ({v.dtype})"
        assert np.all(np.isfinite(v)), f"{name}: numpy path produced non-finite"
    # radiative_mean_temperature (surface, pure-arithmetic) round-trips its definition
    assert S.radiative_mean_temperature(0.0, C_SIGMA_300()) == pytest.approx(300.0, rel=1e-9)


def C_SIGMA_300():
    from flatworld import constants as C
    return C.SIGMA_SB * 300.0 ** 4


def test_fp32_gate_still_downcasts_numpy_input():
    """#196 regression guard: the COSMO_THERMO_FP32 gate must still DOWNCAST a float64 numpy input to
    float32 (``_F`` hard-coerces to the GLOBAL ``_FLOAT``, NOT input-dtype-following). Restored in a
    finally so it cannot poison the bit-identical default elsewhere."""
    prev = th.set_thermo_dtype(np.float32)
    try:
        out = th.qsat(np.float64(300.0), np.float64(9.0e4))
        assert np.asarray(out).dtype == np.float32
    finally:
        th.set_thermo_dtype(prev)
    assert th._FLOAT is np.float64                     # global restored


# --------------------------------------------------------------------------- GPU parity (skip on CPU)
@gpu
def test_fp64_cupy_matches_numpy_machine_precision():
    """fp64 CuPy outputs live on the device and match the numpy reference to machine precision (~1e-13).
    Covers the full ported elementwise chain."""
    import cupy as cp
    ref = _all_outputs(*_inputs(np))
    g = _all_outputs(*_inputs(cp))
    for name in ref:
        assert dev.is_device(g[name]), f"{name}: not on device"
        a = np.asarray(ref[name])
        b = dev.to_host(g[name])
        rel = float(np.abs(a - b).max() / (np.abs(a).max() + 1e-30))
        assert rel < 1e-12, f"{name}: fp64 CuPy-vs-numpy rel {rel:.3e} exceeds 1e-12"


@gpu
def test_fp32_cupy_matches_numpy_to_1e6():
    """With the fp32 dtype gate on, fp32 CuPy outputs are float32 and match the fp64 numpy reference to
    ~1e-6 (single-precision floor). Restored in a finally."""
    import cupy as cp
    ref = _all_outputs(*_inputs(np))
    prev = th.set_thermo_dtype(np.float32)
    try:
        T, p, z, q = (cp.asarray(x, cp.float32) for x in _inputs(np))
        g = _all_outputs(T, p, z, q)
        for name in ref:
            assert dev.is_device(g[name]) and g[name].dtype == cp.float32, \
                f"{name}: expected fp32 device array, got {g[name].dtype}"
            a = np.asarray(ref[name])
            b = dev.to_host(g[name])
            rel = float(np.abs(a - b).max() / (np.abs(a).max() + 1e-30))
            assert rel < 1e-5, f"{name}: fp32 CuPy-vs-fp64-numpy rel {rel:.3e} exceeds 1e-5"
    finally:
        th.set_thermo_dtype(prev)
    assert th._FLOAT is np.float64


@gpu
def test_mixed_cupy_array_plus_python_scalar_stays_on_device():
    """The latent multi-arg trap: a CuPy array argument + a Python/numpy scalar argument must run on the
    DEVICE (one xp per function, all data args routed through it) -- a per-arg numpy 0-d coercion would
    collide with the CuPy array. The all-cupy parity run cannot catch this; this case is the guard."""
    import cupy as cp
    Tg = cp.asarray(np.linspace(250.0, 340.0, 64))
    cases = {
        "qsat": th.qsat(Tg, 9.0e4),
        "dqsat_dT": th.dqsat_dT(Tg, 9.0e4),
        "moist_lapse": th.moist_adiabat_lapse_rate(Tg, 9.0e4),
        "sat_mse": th.saturation_mse(Tg, 9.0e4, 1000.0),
        "mse": th.mse(Tg, 9.0e4, 1000.0, 0.01),
        "vtemp": th.virtual_temperature(Tg, 0.02, 0.005),
        "dtemp": th.density_temperature(Tg, 0.02, 0.005),
        "vtemp_exact": th.virtual_temperature(Tg, 0.02, 0.005, exact=True),      # PHY-20
        "dtemp_exact": th.density_temperature(Tg, 0.02, 0.005, exact=True),      # PHY-20
        "shf": th.sensible_heat_flux(Tg, 280.0, 5.0),
        "lhf": th.latent_heat_flux(Tg, 4.0e-3, 1.0e5, 5.0),
    }
    plcl, tlcl = th.lifting_condensation_level(Tg, 0.01, 1.0e5)
    cases["plcl"], cases["tlcl"] = plcl, tlcl
    SH, LE = S.turbulent_fluxes(Tg, 280.0, 4.0e-3, 1.0e5, 5.0, 0.5)
    cases["turb_SH"], cases["turb_LE"] = SH, LE
    for name, o in cases.items():
        assert dev.is_device(o), f"{name}: mixed (cupy array + scalar) input fell off the device"
    # value parity on the mixed qsat vs the numpy reference
    ref = th.qsat(dev.to_host(Tg), 9.0e4)
    rel = float(np.abs(ref - dev.to_host(cases["qsat"])).max() / (np.abs(ref).max() + 1e-30))
    assert rel < 1e-12, f"mixed qsat rel vs numpy {rel:.3e}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
