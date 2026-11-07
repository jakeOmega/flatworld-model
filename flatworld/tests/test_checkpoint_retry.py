"""save_checkpoint os.replace retry path (PR #153 review #2).

On the /mnt (Windows) mount, ``os.replace`` raises ``PermissionError`` while a reader (a donor/snapshot/
gallery ``load_checkpoint``) holds the target npz open. ``save_checkpoint`` must RETRY rather than crash the
run, and must still re-raise if the contention never clears (so a real, persistent failure is not swallowed).
"""
import os
import numpy as np
import pytest

from flatworld import convergence as conv


def test_save_checkpoint_retries_os_replace_then_succeeds(tmp_path, monkeypatch):
    path = str(tmp_path / "ckpt.npz")
    real_replace = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2:                        # the first two renames hit the held-file EACCES
            raise PermissionError(13, "Permission denied")
        return real_replace(src, dst)

    monkeypatch.setattr(conv.os, "replace", flaky)
    monkeypatch.setattr(conv.time, "sleep", lambda s: None)   # don't actually wait out the backoff
    conv.save_checkpoint(path, {"x": np.arange(5)}, {"iter": 7})
    assert calls["n"] == 3                          # 2 failures + 1 success (the retry recovered)
    st, meta = conv.load_checkpoint(path)
    assert np.array_equal(st["x"], np.arange(5)) and int(meta["iter"]) == 7


def test_save_checkpoint_reraises_when_contention_never_clears(tmp_path, monkeypatch):
    path = str(tmp_path / "ckpt.npz")

    def always_fail(src, dst):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(conv.os, "replace", always_fail)
    monkeypatch.setattr(conv.time, "sleep", lambda s: None)
    with pytest.raises(PermissionError):            # the budget is finite -- a persistent failure still raises
        conv.save_checkpoint(path, {"x": np.arange(3)}, {"iter": 1})


def test_save_checkpoint_retry_is_env_gated_default_on(tmp_path, monkeypatch):
    # C7(3): the mount-contention retry is now gated on COSMO_CKPT_RETRY, DEFAULT ON (preserving Jake's
    # Windows/OneDrive workflow). With the flag OFF, os.replace is attempted exactly ONCE and a real
    # error surfaces immediately (no ~18 s backoff loop) -- the env-specific behavior is opt-out.
    path = str(tmp_path / "ckpt.npz")
    calls = {"n": 0}

    def counting_fail(src, dst):
        calls["n"] += 1
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(conv.os, "replace", counting_fail)
    monkeypatch.setattr(conv.time, "sleep", lambda s: None)
    monkeypatch.setenv("COSMO_CKPT_RETRY", "0")
    with pytest.raises(PermissionError):
        conv.save_checkpoint(path, {"x": np.arange(3)}, {"iter": 1})
    assert calls["n"] == 1                          # flag OFF -> single attempt, no retry loop


# ── None-valued state fields must not poison the npz (the #360 atm_total_reset regression) ──────────
# A None sentinel serialized via np.asarray(None) becomes an object array, which the loader's
# allow_pickle=False refuses. save_checkpoint must SKIP Nones (write-side prevention), and
# load_checkpoint must RECOVER files already on disk that carry the object array (read-side fallback).

def test_save_checkpoint_skips_none_no_object_array_written(tmp_path):
    path = str(tmp_path / "ckpt.npz")
    conv.save_checkpoint(path, {"x": np.arange(4), "sentinel": None},
                         {"iter": 3, "note": None})
    # the raw npz must contain NO object-dtype array (so a strict allow_pickle=False reader never trips)
    with np.load(path, allow_pickle=False) as z:
        assert [k for k in z.files if z[k].dtype == object] == []
        assert "state__sentinel" not in z.files          # the None state field was skipped, not written
        assert "meta__note" not in z.files                # ... and likewise on the meta side
    st, meta = conv.load_checkpoint(path)                 # loads clean; the None key is simply absent
    assert np.array_equal(st["x"], np.arange(4)) and "sentinel" not in st
    assert int(meta["iter"]) == 3 and "note" not in meta


def test_load_checkpoint_recovers_legacy_object_none(tmp_path):
    # emulate a PRE-FIX writer: np.savez a state that contains array(None, object) directly (what the old
    # save_checkpoint produced for a None). The strict allow_pickle=False path raises ValueError; the
    # loader must fall back, warn, and return the sentinel as Python None (which load_state re-derives).
    path = str(tmp_path / "legacy.npz")
    np.savez(path, state__x=np.arange(4), state__atm_total_reset=np.asarray(None),
             meta__iter=np.asarray(9))
    # sanity: this file genuinely trips a strict reader (guards against the emulation going stale)
    with pytest.raises(ValueError, match="allow_pickle"):
        with np.load(path, allow_pickle=False) as z:
            _ = {k: z[k] for k in z.files}
    with pytest.warns(UserWarning, match="legacy object-array"):
        st, meta = conv.load_checkpoint(path)
    assert np.array_equal(st["x"], np.arange(4))
    assert st["atm_total_reset"] is None                  # 0-d object-None unwrapped back to None
    assert int(meta["iter"]) == 9


def test_load_checkpoint_missing_path_polls_silently_but_can_be_made_to_raise(tmp_path):
    """The two contracts of a MISSING checkpoint, both asserted, because they pull opposite ways.

    ``(None, None)`` is deliberate: the fleet supervisors, watchers and dashboards poll for checkpoints
    that do not exist yet and must not crash on the poll, and some two dozen call sites test for it. It
    is also the wrong answer for a consumer that immediately dereferences the state -- a warm-start
    driver pointed at a checkpoint that was not there died inside ``core.load_state(None)`` forty lines
    downstream, with no path in the message. ``required=True`` is for those, and the PATH must be in the
    error: that is the whole point of raising at the load rather than later.
    """
    missing = str(tmp_path / "not_here.npz")
    assert conv.load_checkpoint(missing) == (None, None)
    assert conv.load_checkpoint(str(tmp_path / "not_here")) == (None, None)   # the .npz is appended
    with pytest.raises(FileNotFoundError) as e:
        conv.load_checkpoint(missing, required=True)
    assert "not_here.npz" in str(e.value)
