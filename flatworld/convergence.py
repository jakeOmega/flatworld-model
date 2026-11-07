"""
flatworld.convergence -- shared runtime convergence monitoring, guards, and checkpointing.

The task requires that production runs converge by their OWN criterion: monitor a steady-state
diagnostic (the normalized rate of change of an integrated/conserved quantity) and terminate when it
falls below a documented tolerance, with a max-runtime / max-iteration guard as a backstop, and
periodic checkpoints so a guarded run can resume rather than restart. This module provides those as
reusable pieces so every stage enforces convergence as a runtime stopping condition, not an offline
afterthought.
"""
from __future__ import annotations
import time
import os
import warnings
import numpy as np


class ConvergenceMonitor:
    """Tracks a steady-state diagnostic and decides when to stop.

    The diagnostic is the normalized change of a scalar ``metric`` between checks, per unit of model
    time (or per check): ``rate = |metric - prev| / (scale * d_units)``. Converged when ``rate`` stays
    below ``tol`` for ``patience`` consecutive checks. Stops (not converged) if ``max_iter`` or
    ``max_seconds`` is exceeded.
    """

    def __init__(self, tol=1e-4, patience=3, max_iter=None, max_seconds=None,
                 scale=1.0, name="metric", absolute=False):
        self.tol = float(tol)
        self.patience = int(patience)
        self.max_iter = max_iter
        self.max_seconds = max_seconds
        self.scale = float(scale)
        self.name = name
        self.absolute = absolute   # if True, the metric IS the steady-state diagnostic (converge when
                                   # metric < tol); else converge when its rate of change < tol
        self.prev = None
        self.history = []          # (iter, metric, rate)
        self.below = 0
        self.t0 = time.time()
        self.converged = False
        self.stop_reason = None

    def update(self, iteration, metric, d_units=1.0):
        """Record a new metric value; return True if the run should STOP (converged OR guard hit)."""
        metric = float(metric)
        if self.absolute:
            rate = metric / self.scale          # the metric already is a (normalized) rate/residual
            self.below = self.below + 1 if rate < self.tol else 0
        else:
            rate = np.nan
            if self.prev is not None:
                rate = abs(metric - self.prev) / (self.scale * max(d_units, 1e-30))
                self.below = self.below + 1 if rate < self.tol else 0
        self.history.append((iteration, metric, rate))
        self.prev = metric
        if self.below >= self.patience:
            self.converged = True
            self.stop_reason = f"converged: {self.name} rate < {self.tol:g} for {self.patience} checks"
            return True
        if self.max_iter is not None and iteration >= self.max_iter:
            self.stop_reason = f"GUARD: max_iter={self.max_iter} reached (NOT converged, rate={rate:.2e})"
            return True
        if self.max_seconds is not None and (time.time() - self.t0) >= self.max_seconds:
            self.stop_reason = f"GUARD: max_seconds={self.max_seconds} reached (NOT converged, rate={rate:.2e})"
            return True
        return False

    @property
    def last_rate(self):
        return self.history[-1][2] if self.history else np.nan

    @property
    def elapsed(self):
        return time.time() - self.t0

    def summary(self):
        return dict(converged=self.converged, stop_reason=self.stop_reason,
                    iterations=self.history[-1][0] if self.history else 0,
                    last_metric=self.prev, last_rate=self.last_rate, elapsed_s=self.elapsed)


def save_checkpoint(path, state: dict, meta: dict | None = None):
    """Atomically save a model state (dict of arrays + scalars) to ``path`` (.npz). Resumable."""
    d = os.path.dirname(path)                  # create parent dir if missing (e.g. per-run plots/<run>/data/)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp.npz"
    payload = {}
    for k, v in state.items():
        if v is None:                        # a None sentinel (e.g. atm_total_reset "baseline unknowable")
            continue                         # would save as array(None, object) -- UNLOADABLE under the
        payload["state__" + k] = np.asarray(v)   # loader's allow_pickle=False. load_state re-derives it
    if meta:                                     # from the absent key via .get(), so a written-None and an
        for k, v in meta.items():                # absent key are equivalent on load -- just never write it.
            if v is None:
                continue
            payload["meta__" + k] = np.asarray(v)
    np.savez(tmp, **payload)
    src = tmp if tmp.endswith(".npz") else tmp + ".npz"
    dst = path if path.endswith(".npz") else path + ".npz"
    # /mnt (Windows) mount: os.replace fails with EACCES if a READER holds dst open -- a donor/snapshot/gallery
    # ``load_checkpoint`` keeps the npz open for its whole with-block (it materializes every array first), and
    # this writer's ``np.savez`` is uncompressed, so a multi-MB core state on a slow mount under full
    # fleet+gallery load can be held for SECONDS. Retry with an escalating wait (~18 s total budget) rather than
    # crash the run: a writer blocking a few extra seconds on a rare contended checkpoint is harmless. This is
    # the ONLY protection for live-checkpoint readers other than the gallery snapshot (box-member donors,
    # _cloud_smoke, _autoconv_sweep), so the budget is deliberately generous.
    #
    # This retry is ENVIRONMENT-SPECIFIC (Jake's Windows/OneDrive mount). It is gated on COSMO_CKPT_RETRY,
    # DEFAULT ON so that existing Windows-mount workflows are preserved unchanged; set COSMO_CKPT_RETRY=0
    # (also: false/no/off/empty) on a POSIX/local FS where os.replace is atomic and never EACCES-blocks, to
    # let a genuine write error surface immediately instead of being masked by an ~18 s retry budget.
    if os.environ.get("COSMO_CKPT_RETRY", "1").strip().lower() not in ("0", "false", "no", "off", ""):
        for _attempt in range(20):
            try:
                os.replace(src, dst); break
            except PermissionError:
                if _attempt == 19:
                    raise
                time.sleep(min(0.2 * (_attempt + 1), 1.0))
    else:
        os.replace(src, dst)


def load_checkpoint(path, required=False):
    """Load a checkpoint saved by ``save_checkpoint``. Returns (state, meta) dicts.

    A MISSING FILE returns ``(None, None)`` by default, and that is a deliberate contract, not an
    oversight: the fleet supervisors, watchers and dashboards poll for checkpoints that do not exist yet
    and must not crash on the poll (``_bl_fleet``, ``_allon_fleet``, ``_b5d_watch``, ``driver_support``,
    and ~20 more call sites all test ``st is None``).

    It is the WRONG default for a consumer that immediately dereferences the state: there the None
    travels and blows up somewhere else entirely, with no path in the message -- a warm-start driver
    pointed at a checkpoint that was not there failed inside ``core.load_state(None)`` forty lines
    downstream. Such a caller passes ``required=True`` and gets a ``FileNotFoundError`` naming the path,
    AT THE LOAD.
    """
    p = path if path.endswith(".npz") else path + ".npz"
    if not os.path.exists(p):
        if required:
            raise FileNotFoundError(f"checkpoint not found: {p!r} (resolved from {path!r}; "
                                    f"cwd {os.getcwd()!r})")
        return None, None

    def _read(allow_pickle):
        with np.load(p, allow_pickle=allow_pickle) as z:
            state = {k[len("state__"):]: z[k] for k in z.files if k.startswith("state__")}
            meta = {k[len("meta__"):]: z[k] for k in z.files if k.startswith("meta__")}
        return state, meta

    try:
        state, meta = _read(allow_pickle=False)      # STRICT by default -- an object array is a save-side bug
    except ValueError as e:
        if "allow_pickle" not in str(e):
            raise
        # LEGACY-CHECKPOINT RECOVERY. A pre-fix writer stored a None sentinel (e.g. atm_total_reset,
        # "baseline unknowable") as array(None, object); save_checkpoint no longer writes Nones, but
        # files already on disk carry the object array. These are self-generated LOCAL checkpoints, so
        # the allow_pickle=True re-read is a bounded relaxation -- reached only for a file that already
        # failed strict loading. The 0-d .item() unwrap below turns the object-None back into None, which
        # load_state re-derives via .get(); re-saving the state then writes it back clean.
        warnings.warn(f"[ckpt] {p}: legacy object-array field (a pre-fix None sentinel); re-reading with "
                      f"allow_pickle=True. Re-saving this checkpoint will clean it.")
        state, meta = _read(allow_pickle=True)
    # unwrap 0-d arrays back to python scalars
    for d in (state, meta):
        for k, v in list(d.items()):
            if isinstance(v, np.ndarray) and v.ndim == 0:
                d[k] = v.item()
    return state, meta
