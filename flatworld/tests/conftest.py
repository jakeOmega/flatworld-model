"""Process-environment isolation for the flatworld test suite (INF-16).

WHY THIS FILE EXISTS
--------------------
Many scripts in ``cosmology_models/`` self-configure the CANONICAL world by calling
``canonical_env.setdefault(...)`` at MODULE (import) scope. It has to land before ``flatworld`` loads
-- ``flatworld/constants.py`` and ``flatworld/geometry.py`` read ``COSMO_L_SUN`` /
``COSMO_SHADE_FILE`` / ``COSMO_SHADE2D_FILE`` at import, and a script that sets them afterwards runs
the DEFAULT world silently (the project's #1 documented foot-gun -- OPERATIONS.md section 3,
CANONICAL_WORLD.md).

Inside pytest the same call is a **global side effect on a shared process**. pytest imports every
test module during COLLECTION, before a single test runs, so a test module that imports such a
script applies canonical ``COSMO_*`` values to ``os.environ`` for the whole session.

That is what made the `nightly` full-suite gate red from its first real run after the August CI
blackout: 633 failures, ~92% of them env-shaped. The two that did it then were
``test_band_window.py`` (-> ``build_band_window`` / ``build_coarse_base``, ``COSMO_L_SUN``) and
``test_run_driver_cadence.py`` (-> ``run_driver``, the other 54 keys); both import clean since PR
#772 guarded those drivers (see "THE DRIVERS ARE NO LONGER THE DEMONSTRATOR" below). Most
``flatworld`` config fields are
``field(default_factory=lambda: _env_*("COSMO_...", <default>))``, evaluated at CONFIG
CONSTRUCTION, so with the canonical env resident every later test builds a config with
``x_grid_type='stretched'``, ``advection='vertical_implicit_strang_e'``, the surface-water/salt
flags ON, and a stretched-grid tail at ``COSMO_X_TAIL_FROM_DEG=60`` -- against a box the test sized
for the DEFAULT world. Hence the two headline messages, ``domain too small: ... exceed x_max=...``
and ``tail_from=... must lie strictly inside (0, x_max=...)``, plus every ``test_..._defaults_off``
assertion.

The leak is invisible to the per-PR `tests` gate because that gate names narrow FILE LISTS, none of
which include either polluting module -- so nothing collects them and nothing sets the env. Only a
whole-directory collection (the nightly) trips it, in serial as much as under ``-n auto``: the
parallelism was never the cause, it only reshuffled which tests landed downstream of the leak.

WHAT THIS FILE DOES
-------------------
Two guards, both restoring ``os.environ`` to a snapshot -- neither pins test ORDER and neither
serializes the suite:

  1. ``pytest_configure`` -> ``pytest_collection_finish``: snapshot before collection, restore
     after it. This undoes import-scope driver pollution without changing WHEN anything is
     imported, so no module's import-baked constants move.
  2. an autouse function-scoped fixture: snapshot/restore around every test, so a driver imported
     LAZILY inside a test body (or a test that sets a knob and raises before its own ``finally``)
     cannot leak into the next test either.

Both restore the ENVIRONMENT, and only the environment. Neither un-freezes module state that was
already read from it -- a module that imported (or reloaded) while a knob was set keeps the value it
read, and no later restore can reach back for it. That is a deliberate limit, not an oversight (see
the next section), and the half it cannot reach is pinned separately by
``test_env_isolation.py::test_the_import_baked_world_still_matches_the_pre_collection_environment``.

WHAT THIS FILE DELIBERATELY DOES NOT DO
---------------------------------------
It does not touch the scripts. Guarding one is a per-file change with its own consequences (see the
next section), and this file has to hold for whatever the tree currently contains. It also does not
un-bake anything already frozen at import: a module that read ``COSMO_L_SUN`` while it was set keeps
the value it read, and restoring the env after collection leaves those import-time reads exactly as
they were. That is a deliberate limit, not an oversight -- the half it cannot reach is pinned
separately, by the second test in ``test_env_isolation.py``.

THE SCRIPTS ARE THE DEFECT, AND SEVEN OF THEM ARE FIXED (PR #772)
-----------------------------------------------------------------
This file used to argue that the import-scope call was "right for the driver" and that only the
shared ``os.environ`` was at fault. That was wrong, and PR #772 is the disproof: wrapping the call --
AT ITS EXISTING POSITION, above the module's own flatworld imports -- in ``if __name__ == "__main__":``
leaves the RUN path byte-identical (the guard still executes in file order when the file is run)
while making an import write nothing. Seven drivers now do that: ``build_band_window``,
``build_coarse_base``, ``run_driver``, ``run_supervisor``, ``diagnostic_wind_w0``,
``stage_c_gallery_nsub30`` and ``_window_survey_selfcheck``. So ``test_band_window.py`` and
``test_run_driver_cadence.py``, the two collection-time polluters named above, now import clean.

**The guards in this file are not thereby redundant, but as of INF-16b tranche 2 they are no longer
DEMONSTRATED by the suite.** Since INF-16b tranche 3 the census finds 6 live files calling
``canonical_env.setdefault`` (or a bare ``os.environ.setdefault("COSMO_...")``) in a module body, and
each of the 6 is a deliberate seed — see ``flatworld/OPEN_ISSUES.md`` for which and why. Inside
pytest, moreover, NO test module still imports an unguarded driver at
collection: ``climate_classes`` -- reached at collection by ``test_classify_npz_law.py``,
``test_cloud_cover.py`` and ``test_surface_rh.py``, and the last demonstrator after PR #772 -- is
guarded too now. ``test_env_isolation.py``'s primary invariant is therefore an INVARIANT with no
current exemplar: it still fails the moment a new collection-time polluter appears, which is exactly
what it is for, but do not expect to be able to reproduce a failure by deleting these hooks alone.
The remaining sweep -- and the price the guard charges the scripts that imported a driver precisely
to win the import-timing race -- is tracked in ``flatworld/OPEN_ISSUES.md`` (ROADMAP INF-16b).

THE ONE REMAINING SHARP EDGE (guarded by test_env_isolation.py)
---------------------------------------------------------------
Restoring the environment cannot un-freeze what a module already read from it, so the residual risk
is anything that reads a ``COSMO_*`` knob at import or under ``importlib.reload`` while a stale value
is resident. There are two such windows, and the pin named above closes both by comparing the FROZEN
world (``constants.L_SUN``, ``geometry._shade_path``, ``geometry._shade2d_path``) against the
pre-collection snapshot rather than against the live environment.

``importlib.reload(flatworld.constants)`` RE-READS the environment. ``test_spectrum_env.py`` and
``test_geometry.py`` legitimately use the reload idiom to re-freeze module-level givens. With the
canonical env resident, those reloads flipped ``constants.L_SUN`` from the default 2.0e7 to the
canonical 5.652e7 for the REST of the worker -- which is why the same nightly log carries
``x_max=3.4641e+07`` (= 2.0e7*tan 60 deg) and ``x_max=9.78955e+07`` (= 5.652e7*tan 60 deg) side by
side, and why the geology asserts in ``test_tectonics.py`` / ``test_topo_contract.py`` moved: their
domain scales with L. With the env restored the reload re-reads a CLEAN environment and re-freezes
the same values it had, so the amplifier is disarmed at its input rather than by editing the reload
idiom.
"""
from __future__ import annotations

import os

import pytest

_SNAPSHOT_ATTR = "_flatworld_env_snapshot"


def _snapshot() -> "dict[str, str]":
    return dict(os.environ)


def _restore(saved: "dict[str, str]") -> "list[str]":
    """Put ``os.environ`` back to ``saved`` in place; return the keys that had drifted.

    In place (``os.environ`` itself, never a rebind) because ``os.environ`` writes through to the C
    environment that ``subprocess`` children inherit -- several tests here assert on a child's
    behaviour under a given knob, so the child's view has to move with ours."""
    drifted = []
    for key in [k for k in os.environ if k not in saved]:
        drifted.append(key)
        del os.environ[key]
    for key, value in saved.items():
        if os.environ.get(key) != value:
            drifted.append(key)
            os.environ[key] = value
    return drifted


def pytest_configure(config):
    # Runs BEFORE collection imports any test module, so this is the pristine session environment.
    setattr(config, _SNAPSHOT_ATTR, _snapshot())


def pytest_collection_finish(session):
    saved = getattr(session.config, _SNAPSHOT_ATTR, None)
    if saved is not None:
        _restore(saved)


@pytest.fixture(autouse=True)
def _flatworld_env_isolation():
    """Restore ``os.environ`` after every test (a dict copy per test; unmeasurable next to the
    physics). Complements ``monkeypatch`` rather than replacing it: this catches the cases
    monkeypatch never saw -- a bare ``os.environ[...] = ...``, and a driver imported inside a test
    body whose import-scope ``canonical_env.setdefault`` fires there."""
    saved = _snapshot()
    try:
        yield
    finally:
        _restore(saved)
