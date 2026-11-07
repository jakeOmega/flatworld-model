"""
flatworld.lantern -- the LOCKED projector-day premise as a FORCING FAMILY (Jake 2026-06-11): one
16-h half-cosine day across the habitable band, noon scaled to the local day-mean budget S(theta),
C1-morphing outside the band to the scaled natural sun-passage day. Climate day-means are preserved
EXACTLY by construction (both components are renormalized to S(theta) numerically), so the lantern
changes nothing the zonal climate model sees -- it changes the DIURNAL structure, which is exactly
what Stage C's resolved land diurnal cycle (Issue #21) consumes via
``cascade.land_surface_diurnal(day_cycle=...)``.

UPDATE (2026-07-18): the analytic 16-h plateau above is the FAMILY / doc-figure shape, but it is a
*flatter* day than the committed 2-D lens actually paints. The realized lens day is PEAKIER
(kappa~2.96 vs the plateau's ~2.1) -- a higher noon over sharper dawn/dusk shoulders -- at nearly the
SAME dark-night length (both ~9.5-9.9 h; the plateau's nominal 16 h is mostly sub-120-W/m^2 erfc tail,
so its radiometric daylight is only ~13 h). The correction is PEAKEDNESS, not night length. Forcing the
LAND diurnal solve with the flatter plateau biased the land extremes: its softer evening shoulders held
the surface warmer into the night (biasing minima / frost / dewfall WARM) and its lower noon
under-represented the daytime peak. So ``day_flux_cycle`` now DEFAULTS to the REALIZED lens day
(``geometry.diurnal_flux_cycle``, the same curve the AD core is forced by) whenever a genuine 2-D optic
is loaded, keeping the plateau as a byte-identical escape (``COSMO_LANTERN_PLATEAU`` /
``realized=False``, and automatically in lens-free test worlds). See ``day_flux_cycle`` below and
``REALIZED_DAY_RESULTS.md``.

NB the preservation claim is vs the CORE's S (direct + diffuse): the legacy ``day_cycle=None`` path
(``geometry.diurnal_flux_cycle``) is direct-beam only, so switching ``lantern_day=True`` raises the
land backbone's day-MEAN forcing by the diffuse fraction as well as reshaping the day -- the two
paths differ in mean, not just shape (see ``tests/test_lantern.py`` on the shape gap).

The geometry mirrors the doc figure (``cosmology_models/lantern_wall_figure.py``): the band day
writes on the lantern wall's band crescent; outside the band the wall is free, and the locked
continuous-wall choice paints the scaled NATURAL day there. The plateau spans the habitable ribbon
``BAND_THETA`` (read from ``climate_anchors``; see below), morphing to the natural day just outside it.
The natural component uses the UNSHADED one-sun optics (``_shade_design._one_sun_total``), so this
module is safe to call in a shade-env process (COSMO_SHADE_FILE does not leak in).

Units note: ``S_of_theta`` is the world's day-mean budget [W/m^2] -- in the checkpoint-backbone
cascade it comes straight from the AD core's own ``S`` diagnostic, so the painted day is anchored
to the canonical world with no kernel computation.
"""
from __future__ import annotations
import os
import numpy as np

from scipy.special import erfc

from . import geometry as g
from ._shade_design import _one_sun_total
from .climate_anchors import CLIMATE_ANCHORS

DAYLEN_H = 16.0                  # [h] nominal lit span of the band day (twilight reaches the floor ~±8 h)
# BAND_THETA is the SINGLE SOURCE OF TRUTH for the lantern plateau's latitude span; EVERY lantern script
# (this module, lantern_wall_figure.py, plot_lantern_optic.py, the figure builders) must read it from HERE.
# It is tied to the habitable-ribbon anchor so it can never go stale -- the plateau spans exactly the ribbon
# the rest of the model defines, and follows it automatically when the canonical world moves. (2026-06-28:
# this had been a hard-coded (45,50) from an older, narrower/poleward canonical, so after the bm_tau2h ribbon
# moved to theta 27-54.6 the warm half of the ribbon was left on the morph-to-natural SPIKE day instead of
# the plateau -- over-baking it in the surface diurnal solve and the lethal-class gates. See OPEN_ISSUES.)
BAND_THETA = tuple(float(v) for v in CLIMATE_ANCHORS["ribbon"])   # [deg] = the habitable ribbon (27.0, 54.6)
TRANS_HOT, TRANS_COLD = 4.0, 5.0  # [deg] C1 morph zones to the natural day, just OUTSIDE BAND_THETA (warm:
                                  # BAND_THETA[0]-TRANS_HOT..BAND_THETA[0]; cool: BAND_THETA[1]..+TRANS_COLD)
# The band day is an ANALYTIC plateau (Jake 2026-06-26): a flat bright core with erfc (Gaussian-
# tailed) dawn/dusk edges that fade smoothly into a dark night (no kink), kappa=peak/day-mean~2.1
# (warm-edge noon ~1.7x Earth's above-atmosphere sunlight = the 1361 W/m^2 solar constant; these
# geometry fluxes are the INCIDENT direct beam -- pre-Rayleigh/vapour/cloud, VALIDATION.md "Finding
# (flux basis)". The prior "~2.5x Earth" here divided by Earth's SURFACE 1000 W/m^2, a basis mismatch.)
# These shape constants (T0_DAY/EDGE_SIG, with BAND_THETA/TRANS_* above)
# are the single source: lantern_wall_figure.py imports them from here, so the doc figure can't drift.
T0_DAY, EDGE_SIG = 5.70, 0.55     # [h] plateau half-power half-width / Gaussian edge width


def _band_shape(tau_h):
    """Normalised band-day shape (peak 1) vs circular hours-from-noon (the analytic plateau).

    NOTE (night floor): this is the PLATEAU ONLY -- the pure erfc fades to ~0 at midnight, with NO ember.
    The doc's day FIGURE (day_cycle.png = sections/07_day.md Fig 6) is generated by the more recent
    ``cosmology_models/lantern_kappa_design.py``, which ADDS a dim direct-beam ember ``F_EMBER*grey``
    (F_EMBER=5e-5) so midnight dips to ~1e-4 W/m^2 (clear-starlight, below moonlight), NOT pure black.
    Anything reproducing the doc's day (e.g. the canonical 2-D lens ``_lens_2d_canonical_design.py``)
    matches lantern_kappa_design.py's ember, not this bare plateau."""
    return 0.5 * erfc((np.asarray(tau_h, float) - T0_DAY) / (np.sqrt(2.0) * EDGE_SIG))


_TT = np.linspace(0.0, 12.0, 10001)
DUTY = float(2.0 * np.trapezoid(_band_shape(_TT), _TT) / 24.0)   # day-mean / noon of the plateau


def _smooth01(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def band_weight(th):
    """Day-shape blend: 1 in the band (locked 16-h cosine), C1-smoothly -> 0 outside (natural day).
    Mirrors the wall figure's construction."""
    th = np.abs(np.asarray(th, float))
    up = _smooth01((th - (BAND_THETA[0] - TRANS_HOT)) / TRANS_HOT)
    dn = 1.0 - _smooth01((th - BAND_THETA[1]) / TRANS_COLD)
    return np.minimum(up, dn)


REALIZED_N_SUNS = 6               # nearest suns summed for the realized lens day (geometry's default)


def _use_realized(realized=None):
    """Whether ``day_flux_cycle`` paints the REALIZED lens day (True) or the analytic plateau escape
    (False). ``realized`` (True/False) forces it. Otherwise the plateau is used ONLY when no genuine
    2-D optic is loaded (bare/test world; ``geometry._SHADE2D is None``) or ``COSMO_LANTERN_PLATEAU``
    is set -- so the canonical lens world gets the realized day automatically while lens-free tests
    stay byte-identical to the plateau. (Env read at CALL time, per the import-timing rule.)"""
    if realized is not None:
        return bool(realized)
    if os.environ.get("COSMO_LANTERN_PLATEAU"):
        return False
    return g._SHADE2D is not None


def day_flux_cycle(x, S_of_theta, n_steps=144, clock=None, n_suns=2, realized=None):
    """The lantern day at cross-track x [m]: returns ``(t [s], F [W/m^2])`` over ONE day, t=0 at
    noon (the ``geometry.diurnal_flux_cycle`` convention), with the 24-h mean equal to
    ``S_of_theta(|theta|)`` to floating point (renormalized numerically). Drop-in for
    ``cascade.land_surface_diurnal``'s ``day_cycle`` hook.

    TWO shapes, SAME day-mean (the lantern only reshapes the day, never its budget):
      * REALIZED (default when a genuine 2-D optic ``geometry._SHADE2D`` is loaded -- the canonical
        lens world): the day the loaded lens ACTUALLY paints at x, i.e. the SAME
        ``geometry.diurnal_flux_cycle`` curve the AD core is forced by, renormalized to S. Physically
        consistent: the land surface then rides the same PEAKIER (~12.6-h daylight, kappa~2.96) day the
        core saw -- not a flatter idealization. The night LENGTH is ~unchanged (both ~9.5-9.9 h dark);
        it is the plateau's softer evening SHOULDERS that biased land minima / frost / dewfall warm
        (REALIZED_DAY_RESULTS.md), not a longer/shorter night.
      * PLATEAU (the escape): the analytic 16-h erfc plateau family (``_band_shape`` in the band,
        C1-morphing to the natural day outside). BYTE-IDENTICAL to the pre-2026-07-18 behavior. Used
        automatically when no 2-D optic is loaded, or forced by ``COSMO_LANTERN_PLATEAU`` /
        ``realized=False``. Retained for the doc's lantern-wall figure and regression.
    ``realized``: None -> auto per the rule above; True/False -> force. (``n_suns`` is read only by the
    PLATEAU branch's natural-morph component; the REALIZED branch uses ``REALIZED_N_SUNS`` regardless.)"""
    clock = g.CLOCK if clock is None else clock
    th = float(np.abs(np.degrees(g.theta_of_x(float(x)))))
    S = float(S_of_theta(th))
    if _use_realized(realized):
        # the realized painted day the loaded optic delivers at x, renormalized so 24-h mean = S
        t, F = g.diurnal_flux_cycle(float(x), n_steps=n_steps, clock=clock, n_suns=REALIZED_N_SUNS)
        m = float(np.trapezoid(F, t) / clock.T_day)
        return t, (F * (S / m) if m > 0 else F)
    # ---- escape: the analytic plateau + natural-morph family (byte-identical, pre-2026-07-18) ----
    t = np.linspace(0.0, clock.T_day, n_steps + 1)
    tau_h = np.minimum(t, clock.T_day - t) / 3600.0          # circular hours from noon
    # the painted band component (the analytic plateau), renormalized to day-mean S
    Fc = _band_shape(tau_h)
    mc = np.trapezoid(Fc, t) / clock.T_day
    Fc = Fc * (S / mc) if mc > 0 else Fc
    # the natural component: UNSHADED nearest-suns passage, scaled to the same day-mean
    G = np.zeros_like(t)
    for k in range(-n_suns, n_suns + 1):
        G += _one_sun_total(g.slant_distance(float(x), clock.speed * t - k * clock.spacing))
    mg = np.trapezoid(G, t) / clock.T_day
    Gn = G * (S / mg) if mg > 0 else G
    w = float(band_weight(th))
    return t, w * Fc + (1.0 - w) * Gn
