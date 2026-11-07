"""flatworld.bl_cloud -- the boundary-layer / stratocumulus closure under a subsidence inversion
(CLOUD_PROGNOSTIC_ROADMAP Chunk 4).

The one regime the prognostic-cloud effort exists *for*: **radiatively-driven low cloud held under a
subsidence inversion** -- the marine-stratocumulus deck the grid-mean RH + bulk condensation
(Chunks 1-3) under-resolve. On the canonical world the habitable ribbon sits in the single cell's
**subsiding** branch (`cosmology_models/CLAUDE.md`; the low deck σ≈0.7-0.9 of `CLOUDS_RESULTS.md` §4a),
where the free troposphere is warm/dry and descending. The bulk scheme sees a sub-saturated grid-mean
column there and makes no persistent cloud. But a thin, well-mixed boundary layer capped by the
inversion can sustain stratocumulus by **cloud-top radiative cooling**: the cloud top radiates LW to
space, cooling itself, which (a) condenses water and (b) drives the turbulence that mixes moisture up --
balanced by **entrainment** of warm dry air from above the inversion, which dries/thins the deck. At
the coarse vertical grid (nz≈18, dz≈1.2 km) the inversion and the thin deck are SUB-GRID, so this is a
**parameterisation**: it sources/sinks the *existing* prognostic ``q_c`` (no second condensate -- the
budget stays single-reservoir), it is **direction-firm / magnitude-soft** (the deck depth/LWP is a
parameter choice, not a resolved quantity -- roadmap rule 4), and it is bounded so it cannot run away.

This module is **pure functions** (the leaf physics), like the dynamical/column-thermo kernels in
``stage_ad_core``; the state update + the coupled-sink positivity limiter live in
``StageADCore._bl_stratocumulus``.

THE CLOSURE (a simpler cloud-top-cooling source + entrainment sink, the level the roadmap calls for):

  * **Deck detection** (``deck_mask``): a cell is a stratocumulus-deck cell iff it is (1) within the
    low-cloud σ-band [``sigma_lo``, ``sigma_hi``] (keyed on σ = p/p_surf, a PHYSICAL coordinate, so the
    detection is resolution-robust -- not on a grid index), (2) **subsiding** (resolved w < 0), and
    (3) capped by a **stable inversion** (potential-temperature lapse dθ/dz at the cell top ≥
    ``stability_min``). An ascending or statically-unstable column fails (3)/(2) -> the closure is inert
    where there is no subsidence inversion.

  * **Maintenance source** (condensation, vapour -> ``q_c``). Cloud-top radiative cooling is a SINGLE
    surface process -- only the deck's TOP radiates to space; interior deck cells are LW-shielded by the
    cloud above -- so the closure computes ONE cloud-top cooling flux PER COLUMN and distributes the
    condensation it sustains across the deck cells by mass (NOT a per-cell flux, which would scale the
    integrated source with vertical resolution -- a box-test failure; PR #145 review #1). Per column:
        F_cool = ctop_cool_eff · ε_deck · f_win · σ T_top⁴                [W/m²]  (the window emission to space;
                                                                            an upper estimate -- no downwelling
                                                                            subtracted -- hence magnitude-soft)
        S_k    = F_cool / Σ_deck(L_eff·dm)   on each deck cell k           [kg/kg/s]
    so the column-integrated LATENT HEATING Σ_k L_eff,k S_k dm_k = F_cool tracks ONE cloud-top flux at ANY
    resolution (and is phase-exact across the freezing line; single-phase -> Σ L_eff dm = L·M_deck). The **deck emissivity** ε_deck = max(1 − exp(−kabs · LWP_deck), eps_floor) uses the
    DECK-INTEGRATED in-cloud water path LWP_deck = Σ_deck ρ q_c dz / deck_cover and the SAME ``kabs`` as
    ``radiation.cloud_optical_properties`` -- so the source "closes with" the radiation it feeds (no
    separate radiative constant); ``eps_floor`` is the nascent-cloud seed so a deck can NUCLEATE from
    q_c = 0; T_top is the cloud-top (topmost deck cell) temperature. ε_deck saturates to 1 once the deck is
    optically thick, so the source SATURATES (the first half of the runaway guard).

  * **Entrainment sink** (warm dry air mixed down -> evaporates ``q_c`` back to vapour):
        D = entrain_eff · q_c / tau_entrain                            [kg/kg/s]
    a relaxation on the entrainment-drying timescale ``tau_entrain``; ``entrain_eff`` is the **reported
    sensitivity** (more entrainment -> a thinner deck). LINEAR in q_c, so as the deck thickens the sink
    grows while the source saturates -> a FINITE steady deck S(q_c)=D(q_c) (the second half of the guard;
    a hard ``source_max`` cap backs it up).

ENERGY / WATER: the source CONDENSES vapour (latent warming) and the sink EVAPORATES condensate (latent
cooling) at the SAME level -- water moves between q and q_c only (total water q+q_c conserved within the
column; NO new precip sink), exactly mirroring the Chunk-2 condensation/re-evaporation. The cloud-top
radiative COOLING itself is applied by ``column_radiation`` (the deck's q_c feeds ``dtau_lw_cloud``); the
closure uses the cooling flux only as the SCALE that sets the maintainable condensate -- it applies NO
radiative tendency of its own, so the cooling is not double-counted.
"""
from __future__ import annotations
import numpy as np
from . import constants as C
from . import radiation as rad
from . import thermo as th                                  # _F (dtype gate) + _xp_of (array module)
from .device import get_xp

# Module defaults (observational best estimates / nascent-cloud seed -- NOT fitted to a climate target;
# the magnitude is reported as a sensitivity, roadmap rule 4). Overridable via the function args / config.
BL_SC_EPS_FLOOR = 0.05         # nascent-cloud-top emissivity floor [-] so a deck can nucleate from q_c=0
                               # (a thin initial cloud still cools its top); ε saturates to 1 as q_c grows.


# DEVICE/dtype (Phase D): bl_cloud is array-module-generic + fp32-gate-following via ``th._F`` -- a numpy
# fp64 input is the verbatim ``np.asarray(x, float64)`` (BYTE-IDENTICAL), a CuPy input stays on the GPU, and
# under ``set_thermo_dtype(fp32)`` it runs single precision. Reached on the device coupled path via
# ``StageADCore._bl_stratocumulus`` (the production cloud stack). The deck reductions (sum/argmax/
# take_along_axis) all have exact CuPy analogues (probed).
def potential_temperature(T, p_mid, p_ref=C.P_SURF):
    """Potential temperature θ = T (p_ref/p)^κ [K], shape matching ``T`` (..., nz)."""
    xp = th._xp_of(T, p_mid)
    return th._F(T, xp) * (p_ref / th._F(p_mid, xp)) ** C.KAPPA_POISSON


def static_stability(T, p_mid, z, p_ref=C.P_SURF):
    """Potential-temperature lapse dθ/dz [K/m] at each cell's TOP (the level above, index k-1, is higher
    in z on the TOP->SURFACE pressure grid). Shape (..., nz); the top cell (k=0, no level above) -> 0.
    Positive = statically STABLE (θ increases upward); the inversion cap is a large positive value."""
    xp = th._xp_of(T, p_mid, z)
    theta = potential_temperature(T, p_mid, p_ref)
    z = th._F(z, xp)
    dthdz = xp.zeros_like(theta)
    dz_up = z[..., :-1] - z[..., 1:]                       # z[k-1]-z[k] > 0 (height of cell k's top)
    dthdz[..., 1:] = (theta[..., :-1] - theta[..., 1:]) / xp.where(dz_up != 0.0, dz_up, 1.0)
    return dthdz


def deck_mask(T, w_p, p_mid, z, *, sigma_lo=0.70, sigma_hi=0.90, stability_min=2.0, p_surf=C.P_SURF):
    """Boolean (..., nz) mask of stratocumulus-deck cells: within the σ-band [``sigma_lo``, ``sigma_hi``]
    (σ = p_mid/p_surf, a PHYSICAL coordinate -> resolution-robust), SUBSIDING (``w_p`` < 0), and capped by
    a STABLE inversion (dθ/dz at the cell top ≥ ``stability_min`` [K/km]). ``w_p`` is the resolved vertical
    velocity on the pressure grid (positive up); ``z`` the layer-centre heights (..., nz, TOP->SURFACE)."""
    xp = th._xp_of(T, w_p, p_mid, z)
    sigma = th._F(p_mid, xp) / p_surf
    band = (sigma >= sigma_lo) & (sigma <= sigma_hi)
    subsiding = th._F(w_p, xp) < 0.0
    stable = static_stability(T, p_mid, z, p_surf) >= (stability_min / 1000.0)   # K/km -> K/m
    return band & subsiding & stable


def emissivity_from_lwp(lwp, *, kabs=rad.CLOUD_KABS_LW, eps_floor=BL_SC_EPS_FLOOR):
    """Cloud LW emissivity ε = max(1 − exp(−kabs · LWP), eps_floor) [-] from an in-cloud water path
    ``lwp`` [kg/m^2]. Uses the SAME ``kabs`` the radiation optics use (``CLOUD_KABS_LW``, whose
    two-stream layer emissivity is the physical 1 − exp(−kabs·LWP)), so the closure closes with the
    radiation; ``eps_floor`` is the nascent-cloud seed (nucleation from q_c = 0)."""
    xp = th._xp_of(lwp)
    return xp.maximum(1.0 - xp.exp(-kabs * th._F(lwp, xp)), eps_floor)


def deck_emissivity(q_c, T, p_mid, dm, *, deck_cover=1.0, kabs=rad.CLOUD_KABS_LW, eps_floor=BL_SC_EPS_FLOOR):
    """PER-CELL cloud LW emissivity from the in-cloud water path LWP_in = ρ q_c dz / deck_cover =
    q_c dm / deck_cover (dz = dm/ρ); see ``emissivity_from_lwp``. Shape matching ``q_c``. (The maintenance
    source uses the DECK-INTEGRATED emissivity -- one cloud top per column; this per-cell form is the
    single-cell limit / the emissivity-law cross-check.)"""
    xp = th._xp_of(q_c, dm)
    lwp_in = th._F(q_c, xp) * th._F(dm, xp) / max(deck_cover, 1e-9)
    return emissivity_from_lwp(lwp_in, kabs=kabs, eps_floor=eps_floor)


def stratocumulus_rates(T, q_c, mask, p_mid, dm, *, deck_cover=1.0, ctop_cool_eff=1.0, entrain_eff=1.0,
                        tau_entrain=3600.0, source_max=1.0e-6, f_win=rad.F_LW_WINDOW,
                        kabs=rad.CLOUD_KABS_LW, eps_floor=BL_SC_EPS_FLOOR):
    """The stratocumulus maintenance source ``S`` and entrainment sink ``D`` [kg/kg/s] per cell, nonzero
    only on the deck (``mask``). 2-D ``(nx, nz)`` (last axis vertical, TOP->SURFACE). Returns ``(S, D)``
    (both ≥ 0); the caller applies them with the matching latent heat under a coupled-sink positivity limiter.

    The source is ONE cloud-top cooling flux PER COLUMN, distributed across the deck by mass (so the
    integrated source tracks one cloud top at any vertical resolution -- NOT a per-cell flux; PR #145 #1):

      F_cool = ctop_cool_eff · ε_deck · f_win · σ T_top⁴      (ε_deck from the DECK-INTEGRATED in-cloud LWP;
                                                               T_top the cloud-top / topmost deck cell)
      S_k    = min(F_cool / Σ_deck(L_eff·dm), source_max)     on each deck cell k
      D_k    = entrain_eff · q_c_k / tau_entrain

    The source's uniform mixing-ratio rate uses the deck's L-WEIGHTED mass Σ_deck(L_eff·dm) as the
    denominator, so the per-cell latent heat (applied with each cell's own phase L_eff in the caller)
    integrates EXACTLY to F_cool even across the freezing line (single-phase -> L·M_deck, byte-identical).
    L_eff is L_VAP above freezing, L_SUB below (deposition), the SAME phase split the optics/microphysics
    use. ε_deck SATURATES to 1 once the deck is optically thick while the sink is LINEAR in q_c -> a finite
    steady deck (Σ_deck q_c dm = F_cool·tau/(L·entrain_eff), resolution-robust); the ``source_max`` cap is
    the backstop runaway guard. Magnitude-soft (rule 4): ``entrain_eff`` is the reported sensitivity; the
    closure pins the deck's EXISTENCE/LOCATION and its INTEGRATED strength's scaling, not its absolute level."""
    xp = th._xp_of(T, q_c, dm)
    T = th._F(T, xp)
    qc = th._F(q_c, xp)
    dm_b = xp.broadcast_to(th._F(dm, xp), T.shape)
    m = (mask if get_xp(mask) is xp else xp.asarray(mask)).astype(bool)       # mask onto T's module
    # per-column deck reductions (axis=-1 = vertical): deck air mass and the DECK-INTEGRATED in-cloud LWP
    M_deck = xp.sum(xp.where(m, dm_b, 0.0), axis=-1, keepdims=True)           # [kg/m^2] per column
    lwp_cell = qc * dm_b / max(deck_cover, 1e-9)                              # per-cell in-cloud LWP [kg/m^2]
    LWP_deck = xp.sum(xp.where(m, lwp_cell, 0.0), axis=-1, keepdims=True)     # deck-integrated [kg/m^2]
    eps_deck = emissivity_from_lwp(LWP_deck, kabs=kabs, eps_floor=eps_floor)  # ONE cloud-top emissivity / column
    # cloud-top temperature = the topmost deck cell (smallest index = highest); 0-index for deck-free columns
    # (masked out below, so its value is irrelevant). T is (nx, nz).
    idx_top = xp.argmax(m, axis=-1)                                           # (nx,) first True per column
    T_top = xp.take_along_axis(T, idx_top[:, None], axis=-1)                  # (nx, 1)
    F_cool = ctop_cool_eff * eps_deck * f_win * C.SIGMA_SB * T_top ** 4        # (nx, 1) cloud-top LW cooling [W/m^2]
    # Distribute as a uniform mixing-ratio rate whose PER-CELL latent heat (applied with each cell's own
    # phase L_eff in _bl_stratocumulus) integrates EXACTLY to F_cool -- even across the freezing line: the
    # denominator is the deck's L-WEIGHTED mass Sigma_deck L_eff*dm, not L_top*M_deck (which would be off by
    # the mass-weighted (L_eff-L_top)/L_top on a mixed-phase deck; PR #145 review follow-up). Single-phase ->
    # Sigma L_eff*dm = L*M_deck, byte-identical.
    L_eff = xp.where(T < C.T_FREEZE, C.L_SUB, C.L_VAP)
    LdM_deck = xp.sum(xp.where(m, L_eff * dm_b, 0.0), axis=-1, keepdims=True)  # (nx, 1) deck L-weighted mass [J/m^2]
    S_col = F_cool / xp.maximum(LdM_deck, 1e-9)                               # (nx, 1) condensation rate [kg/kg/s]
    S_col = xp.minimum(S_col, source_max)
    S = xp.where(m, xp.broadcast_to(S_col, T.shape), 0.0)                     # uniform-by-mass across the deck
    D = xp.where(m, entrain_eff * qc / max(tau_entrain, 1e-9), 0.0)
    return S, D
