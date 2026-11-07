"""
flatworld.climate_anchors -- the SINGLE source for the geology stack's climatology |theta|-anchors (issue #83).

The geology stack keys several |theta|-band thresholds on the zonal-mean climate: the wet habitable RIBBON, the
rain BAND peak, the hot SALT-flat edge, and the cold ICE / glaciation edge. These are consumed across
``tectonics`` (erosion climatology ``default_w_clim`` + the salt/ice albedo bands), ``glacial`` (the ice line),
``topo_contract`` (the albedo hint) and ``rivers`` (enclosed-basin fate) -- so they live here ONCE, and the
later re-pin is a single edit instead of a scatter hunt across four modules.

*** RE-PINNED 2026-07-18 to the CANONICAL WORLD ``lens_diurnal`` *** (run17: L=5.652e7, I0=2.5, 2-D
lantern lens + resolved diurnal sun, stretched grid to theta 89.5 -- ``CANONICAL_WORLD.md``). Derived by
``run_from_checkpoint.emit_geology`` from the checkpoint's day-mean ``clim_*`` EMAs (the DOC-1 one-command
re-pin, executed at the DOC-9 promotion); crossings match the verified values in
``cosmology_models/CANON_DOC_NUMBERS_lens_diurnal.md`` (the doc's single source of truth). It SUPERSEDES
the ``nsub30`` pins (ribbon (27,54.6) on L=8e7): the brighter lensed sun contracts the habitable band
~12x and moves it poleward (warm edge 27->45.4, cold edge 54.6->48.2 -- a ~6,000 km lane ~60,000 km out).

**THE ZONAL STRUCTURE (load-bearing for any consumer logic).** From the sun-line outward:

  * theta < ~19 (the ``salt`` anchor = run17's innermost open-water column) is DRY EVAPORITE FLAT under
    the steam-greenhouse axis (day-mean up to ~330 C near theta~3): salt crust, flash-boiling virga,
    no standing water.
  * theta ~19-45 is the HOT SEA (day-mean 74 -> 30 C): drizzly (P ~1-1.4 mm/day), the evaporative
    source; the SST 55 C evaporation cap crosses at theta~37.6. The LIQUID-runoff peak of the
    climatology sits over its cool flank at theta~40 (norm 1.32 mm/day).
  * the habitable ribbon theta ~45.4-48.2 is ARID (~0.5 mm/day) and bright -- its land hydrology is
    MELTWATER-FED (exotic rivers from the ice margin), not rain-fed.
  * poleward of the ice line (ice = ribbon[1] = 48.2) the near-flank is nearly rainless (~0.08 mm/day
    to theta~55), then snow rises through the designed near-margin window to the ``band`` divide
    (60.2, the window edge), and keeps rising to the FAR snow dome at theta~72-75 (~3x the liquid
    peak) under the lens-donor warm shelf (theta 65-71). The far dome is DELIBERATELY outside the
    band-search window (its cold-based ice, 100,000+ km out at -70..-120 C, is not assumed to feed
    the warm margin); revisit ``band_search_pole_deg`` if that judgment changes (PR-3, 2026-07-18).

``tectonics`` assembles its ``default_w_clim`` fallback knots from these AT IMPORT and asserts strict
monotonicity (``salt < ribbon[0] < interior < ribbon[1] <= ice < band < 66 < 76``), so a bad re-pin
fails loudly rather than silently returning ``np.interp`` garbage.

**DOC-1 CLOSED (2026-07-18).** The precip-SHAPE inputs (``data/bl_precip_clim.npz`` + the fallback knots
in ``tectonics``/``meltwater``) are now rebuilt from the SAME run17 checkpoint as these anchors -- the
bl_dz20 half-migration is over. To promote a future canonical world: run
``run_from_checkpoint.py --checkpoint <ckpt> --emit-geology --write-geology``, paste the reported anchors
here, re-sample the two fallback knot sets from the rebuilt npz, and keep the geology magnitude tests
strong.

**B-INVARIANT (reviewed 2026-06-19).** These anchors are pinned to the **AD-core zonal climate** (the BL-core
``*_profhist`` + checkpoint), which is UPSTREAM of Stage C on the one-way cascade. The J1 "flip B on"
(``config.stage_c("production")`` -> ``resolved_supply=True``) is a *Stage-C* config that dries the band's
*(x,e)* precip where continents are upstream -- but it changes neither the **temperature** (Stage-C ``T_map_C``
is the static AD-core backbone, so ``ribbon``/``salt``/``ice`` -- the 30/55/0 C crossings -- are untouched)
nor the **AD-core zonal precip** (so ``band`` -- the snow peak -- is untouched). So **do NOT re-pin these for
B**; B's downstream is the Köppen classification of the Stage-C deliverable + the doc, tracked in
the root ``ROADMAP.md`` (DOC-3). (A future LIGHT re-pin may still happen for a *canonical-world* refinement -- a
different reason -- per the note above.)
"""
from __future__ import annotations

CLIMATE_ANCHORS = dict(
    ribbon=(45.4, 48.2),   # habitable ribbon [deg]: the lens_diurnal T_surf 30 C -> 0 C day-mean crossings
    #                        (warm edge theta~45.4 / x~57,300 km; cold edge theta~48.2 / x~63,300 km;
    #                        ribbon-mean ~18.5 C; edges carry the run's ~+-0.2 deg drift bar)
    band=60.2,             # near-margin SNOW-accumulation divide [deg] = the argmax of w_snow in the
    #                        DESIGNED (ice, ice+12] search window (anchors_from_profiles band_search_pole_deg
    #                        -- the window deliberately EXCLUDES the far lens-donated snow dome at
    #                        theta~72-75, whose cold-based ice is not assumed to feed the warm margin).
    #                        On run17 the near-margin snow RISES toward the window edge, so the derived
    #                        band sits AT that edge; see the PR-3 note (promotion 2026-07-18).
    salt=19.0,             # |theta| below which hot-zone land is evaporite salt flat -- run17's innermost
    #                        wetness>=0.5 column (the coastline; x~19,500 km); equatorward is crusted flat
    #                        under the steam-greenhouse axis
    ice=48.2,              # |theta| above which land is ice-mantled -- the ribbon's own cold edge (0 C): the
    #                        glacial zone starts AT the ribbon's cold edge and is snow-fed, out to the deep-
    #                        cold N2 cryosphere far beyond (out of geology scope)
)


# =====================================================================================================
# DERIVE the anchors from an AD-core zonal climate -- so a NEW canonical world is a one-command re-pin
# (``run_from_checkpoint.emit_geology``) instead of hand-editing the literals above. The literal
# ``CLIMATE_ANCHORS`` remains the COMMITTED current-canonical value; ``anchors_from_profiles`` recomputes
# the same four keys from a checkpoint's zonal T_surf (+ snow) so promotion re-derives them.
# =====================================================================================================
import numpy as _np   # noqa: E402  -- kept below the literal so a bare `from climate_anchors import CLIMATE_ANCHORS` stays numpy-free


def _first_down_crossing(theta, tsurf_c, level, start_theta=0.0):
    """First ``theta`` (scanning OUTWARD from ``start_theta``) where the surface temperature falls THROUGH
    ``level`` [deg C] -- the warm->cold pass that defines a ribbon edge. ``None`` if there is no such
    crossing. Linear-interpolated between the bracketing grid points. Restricting to DOWNWARD crossings
    (warm side above, cold side below) ignores any poleward RE-warming (e.g. a lens-donated far-field
    bump), so the ribbon edge is the inner cooling limb, not a spurious outer crossing."""
    th = _np.asarray(theta, float); T = _np.asarray(tsurf_c, float)
    for i in range(1, len(th)):
        if th[i] < start_theta:
            continue
        if T[i - 1] >= level > T[i]:
            return float(th[i - 1] + (th[i] - th[i - 1]) * (level - T[i - 1]) / (T[i] - T[i - 1]))
    return None


def anchors_from_profiles(theta_deg, tsurf_c, *, w_snow=None, wetness=None,
                          warm_c=30.0, cold_c=0.0, salt_c=100.0, band_pole_deg=2.0,
                          band_search_pole_deg=12.0):
    """Derive the geology band anchors (``ribbon``/``ice``/``salt``/``band``) from an AD-core ZONAL climate.

    Inputs are the core's ``theta_deg`` [deg, any order] and day-mean ``tsurf_c`` [deg C]; optionally the
    zonal ``w_snow`` (snow-accumulation proxy, for a data-driven ``band``) and ``wetness`` [0..1] (open-
    water fraction, for a data-driven ``salt`` sill). Rules:

      * ``ribbon`` = (30 C down-crossing, 0 C down-crossing) -- the habitable-temperature edges (``warm_c``
        / ``cold_c``). ``ice`` = the 0 C crossing (the ribbon's own cold edge, where melt_runoff gates).
      * ``salt`` = the hot open-water edge. From ``wetness`` when given (innermost theta whose wetness
        has climbed to >= 0.5 -- equatorward of it the surface is bare evaporite), else the ``salt_c``
        (~boiling) down-crossing of T_surf (rain flashes off a >100 C flat), else the committed literal.
      * ``band`` = the SNOW-accumulation peak in the NEAR-MARGIN window ``(ice, ice+band_search_pole_deg]``
        (from ``w_snow``) -- the calving divide / accumulation dome just poleward of ice; the window
        bound keeps a far-field snow bump (e.g. a lens-donated poleward re-warming) from being mistaken
        for the margin dome. Absent ``w_snow`` (or an empty window), held ``band_pole_deg`` poleward of ``ice``.

    Returns a dict shaped like ``CLIMATE_ANCHORS``; raises if the invariant
    ``salt < ribbon[0] < ribbon[1] <= ice < band`` fails (a non-physical climate or a bad crossing)."""
    th = _np.asarray(theta_deg, float); T = _np.asarray(tsurf_c, float)
    order = _np.argsort(th); th, T = th[order], T[order]

    warm = _first_down_crossing(th, T, warm_c)
    if warm is None:
        raise ValueError(f"no {warm_c} C (ribbon warm-edge) down-crossing in T_surf -- the world may have no "
                         f"habitable ribbon, or T_surf is not in deg C")
    cold = _first_down_crossing(th, T, cold_c, start_theta=warm)
    if cold is None:
        raise ValueError(f"no {cold_c} C (ribbon cold-edge / ice line) down-crossing poleward of {warm:.1f} deg")
    ice = cold

    # salt sill (hot open-water edge) -- soft; prefer wetness, then a hot-T crossing, then the literal
    salt = None
    if wetness is not None:
        w = _np.asarray(wetness, float)[order]
        wet = _np.where(w >= 0.5)[0]
        if wet.size:
            salt = float(th[wet[0]])                       # innermost open-water column
    if salt is None and salt_c is not None:
        salt = _first_down_crossing(th, T, salt_c)         # T falls below the salt-flat/boiling temperature
    if salt is None:
        salt = float(CLIMATE_ANCHORS["salt"])

    # snow-accumulation peak in the NEAR-MARGIN window poleward of the ice line -> the calving divide.
    # bounded so a far-field snow bump (deep-cold / lens-donated re-warming) is not read as the dome.
    if w_snow is not None:
        wsn = _np.asarray(w_snow, float)[order]
        win = (th > ice) & (th <= ice + band_search_pole_deg)
        band = float(th[win][int(_np.argmax(wsn[win]))]) if (win.any() and wsn[win].max() > 0) \
            else ice + band_pole_deg
        if band <= ice:                                    # degenerate: peak sat at the margin -> nudge poleward
            band = ice + band_pole_deg
    else:
        band = ice + band_pole_deg

    anchors = dict(ribbon=(warm, cold), band=band, salt=salt, ice=ice)
    assert_anchors_monotonic(anchors)
    return anchors


def assert_anchors_monotonic(anchors):
    """The load-bearing invariant every consumer (``tectonics``/``meltwater``/``glacial``) assumes:
    ``salt < ribbon[0] < ribbon[1] <= ice < band``. Raises ``ValueError`` on violation so a bad re-pin
    (hand or derived) fails loudly rather than silently returning ``np.interp`` garbage on non-monotone xp."""
    r0, r1 = anchors["ribbon"]
    seq = [("salt", anchors["salt"]), ("ribbon[0]", r0), ("ribbon[1]", r1),
           ("ice", anchors["ice"]), ("band", anchors["band"])]
    # ribbon[1] <= ice (equal allowed: ice IS the 0 C cold edge), everything else strictly increasing
    ok = (anchors["salt"] < r0 < r1 <= anchors["ice"] < anchors["band"])
    if not ok:
        raise ValueError(f"anchors violate salt < ribbon[0] < ribbon[1] <= ice < band: "
                         f"{[(n, round(v, 2)) for n, v in seq]}")
    return True
