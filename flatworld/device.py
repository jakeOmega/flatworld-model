"""flatworld.device -- the numpy/CuPy array-module shim for the M3 GPU backend (M3.1).

The fused anelastic core (``stage_ad_core``) runs its hot dynamical kernels on either the CPU (numpy)
or an NVIDIA GPU (CuPy) behind a single dispatch point, so a converged resolution x closure sweep (M3.5)
is affordable. This module is that dispatch point; ``stage_ad_core`` imports ``get_xp``/``to_host``/
``to_device`` from here and is otherwise device-agnostic.

DESIGN (per M3_SUBTASKS.md M3.1 + the array-module review):
  * CuPy is an OPTIONAL dependency (``requirements-gpu.txt``). The import is GUARDED -- ``flatworld``
    must import, and the whole CPU test gate must run, with NO CuPy installed. ``_HAVE_CUPY`` records
    whether it is present; nothing here raises at import time.
  * The kernels infer their array module FROM their array argument (``get_xp(F)``): a numpy array in ->
    numpy out (BIT-IDENTICAL to the pre-M3.1 code, so the CPU gate is unchanged), a CuPy array in ->
    CuPy out. This is less invasive than threading an explicit ``xp=`` through every signature and
    removes the "xp says cupy but the array is numpy" footgun.
  * Host<->device transfers are EXPLICIT (``to_host``/``to_device``) and confined to the few real
    seams: the splu Poisson solve (kept on host in M3.1 -- ``omega`` -> host -> solve -> ``psi`` ->
    device), checkpoint/state I/O, and the per-substep CFL reduction (a scalar pulled to the host).

fp64 is the default working precision (decision in M3.1): conservation is structural/precision-
independent, but the ~1e-10 consistency floor that catches a discretization leak is ``kappa(op)*eps`` --
an fp32 ~1e-3 floor would bury it. See ``M31_RESULTS.md`` for the measured fp64-on-3080 throughput verdict.
"""
from __future__ import annotations
import numpy as np

try:                                  # optional GPU backend; absent on CPU-only sessions
    import cupy as _cp
    _HAVE_CUPY = True
except Exception:                     # ImportError, or a CuPy that fails to load on a driver-less box
    _cp = None
    _HAVE_CUPY = False


def have_cupy() -> bool:
    """True iff the CuPy package imported (it may still lack a runnable device -- see ``gpu_available``)."""
    return _HAVE_CUPY


def gpu_available() -> bool:
    """True iff CuPy imported AND a CUDA device is actually reachable. Tests use this to SKIP the GPU
    parity/throughput checks cleanly on a CPU-only box (rather than fail), so the gate is portable."""
    if not _HAVE_CUPY:
        return False
    try:
        return _cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def array_module(device: str = "cpu"):
    """Explicit module selection for a configured device string: numpy for ``"cpu"``, CuPy for
    ``"gpu"`` (raising a friendly install hint if CuPy is unavailable). Used once, in
    ``StageADCore.__init__``, to set ``self.xp`` and place the dynamics state."""
    if device == "gpu":
        if not _HAVE_CUPY:
            raise RuntimeError(
                "StageADConfig.device='gpu' but CuPy is not installed. Install the optional GPU "
                "backend:  .venv/bin/pip install -r requirements-gpu.txt  (see M3_SUBTASKS.md M3.1).")
        return _cp
    if device == "cpu":
        return np
    raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}")


def get_xp(arr):
    """Infer the array module FROM an array: CuPy iff ``arr`` is a CuPy ndarray, else numpy. This is the
    one-liner the module-level kernels open with (``xp = get_xp(F)``); a numpy caller gets the numpy
    branch and so is bit-identical to the pre-M3.1 code."""
    if _HAVE_CUPY and isinstance(arr, _cp.ndarray):
        return _cp
    return np


def is_device(arr) -> bool:
    """True iff ``arr`` lives on the GPU (a CuPy ndarray)."""
    return _HAVE_CUPY and isinstance(arr, _cp.ndarray)


def to_host(arr):
    """Bring an array to host numpy: a device->host copy for a CuPy array, an (alias) no-op for numpy
    (``np.asarray`` returns an ndarray unchanged -- so a numpy caller sees no copy and no behavior change)."""
    if _HAVE_CUPY and isinstance(arr, _cp.ndarray):
        return _cp.asnumpy(arr)
    return np.asarray(arr)


def host_array(arr, dtype=float):
    """A core attribute as a host NUMPY array of ``dtype`` -- ``to_host`` plus the cast, in one call.

    THE READ THE SEED PATH NEEDS. On a GPU core the fixed grid arrays are device-resident --
    ``StageADCore._place_thermo_on_device`` places ``p_edge``, ``p_mid``, ``dm``, ``sigma``,
    ``C_layer``, ``S``, ``T_ref``, ``z``, ``_dlnp``, ``RH``, and the stretched-x block places
    ``_dx_op`` / ``_dx_cell`` / ``_dx_w`` / ``_inv_dxw_row``. CuPy refuses implicit conversion, so a
    bare ``np.asarray`` on any of them raises ``TypeError: Implicit conversion to a NumPy array is
    not allowed`` -- which is exactly how the cross-track warm start failed to seed a GPU run at all
    (``core.dm``; PR #845). Anything that reads a core attribute into host arithmetic goes through
    here, so the same defect cannot return the next time a field is placed on the device.

    A no-copy alias on host input (``to_host`` is ``np.asarray`` there, and ``np.asarray`` with a
    matching dtype returns the same buffer), so the CPU path is unchanged."""
    return np.asarray(to_host(arr), dtype)


def to_device(arr, xp):
    """Place ``arr`` onto ``xp``'s memory (``xp.asarray``). For ``xp is numpy`` this is the no-op alias
    above; for CuPy it is a host->device copy (or a no-op if ``arr`` is already a CuPy array)."""
    return xp.asarray(arr)


def synchronize() -> None:
    """Block until queued device work completes (a no-op on the CPU). REQUIRED for honest GPU timing:
    CuPy kernel launches are asynchronous, so a benchmark must synchronize before stopping the clock.
    A genuine sync failure is NOT swallowed — a silently-skipped sync would under-count GPU time and
    bias the benchmark optimistically, so let it raise (it cannot fire on a working CuPy/device)."""
    if _HAVE_CUPY:
        _cp.cuda.Stream.null.synchronize()
