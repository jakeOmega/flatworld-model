"""
flatworld.climate_map -- the CLIMATE-MAP stage: the DYNAMICAL 2-D (x, e) climate map with topology.

(Renamed from ``stage_c_map`` -- ROADMAP GEO-22: "Stage C" named a lettered A->B->C sequence whose
other members were fused into the AD core, and the operation is a *dimension* change ((x,z)->(x,e)),
not a downscaling step. ``flatworld.stage_c_map`` remains importable as a module alias; the class/
config names keep their ``StageC`` prefix so every existing caller, checkpoint, and results doc still
reads correctly.)

The deliverable: full 2-D maps of time-averaged surface temperature T(x,e) and precipitation P(x,e),
where topology breaks the along-track (e) symmetry. This is NOT a kinematic overlay on a prescribed
wind (that would be the retired plan-view model with the moisture direction patched). The horizontal
boundary-layer wind is a DIAGNOSTIC (damped) BALANCED wind, f=0 -- the Lindzen-Nigam/Gill limit
V = u_bg - grad(Phi)/r. It is topology-RESPONSIVE: the geopotential Phi carries the land/sea
diabatic-heating contrast (warm land -> thermal low -> convergence -> ascent -> rain) and the terrain,
and the wind flows down grad(Phi), so where the rain lands is set by SOLVED convergence, not by an
imposed wind. Only the two-layer moisture (W_s, W_a, P) is prognostic in time; the wind is the
diagnostic balance recomputed from Phi each step.

LIMITATION (stated plainly, not hidden): Phi is built ONCE in __init__ from a static temperature map
and is NOT updated as the moisture/latent heating evolve -- so there is NO feedback from the evolving
moisture field back onto the circulation. For a time-mean f=0 map a damped balanced wind is a standard,
adequate choice (the deliverable needs a topology-responsive wind, which this is; the water budget
closes and precip carries real along-track structure), but the absent circulation-moisture feedback is
a genuine first-order limitation. This was previously mislabeled "prognostic horizontal momentum"; it
is a diagnostic balance, corrected here (PR #4 review P1).

Moisture is TWO-LAYER (fixing the prior single-layer direction error, which advected near-surface
moisture equatorward while the real flux is poleward aloft -- confirmed in Stage B):
  * an ALOFT layer W_a carrying the poleward-delivered moisture supply F(theta) (recharged over sea),
    raining its saturation excess where the air has cooled enough (the stratiform band) and where
    forced/dynamical ascent lifts it (orographic + convergence);
  * a SURFACE layer W_s fed by sea/lake evaporation and land evapotranspiration (recycling), uplifted
    into the aloft layer where the dynamical flow converges.
The water budget closes (int E == int P at steady state).

Temperature carries maritime buffering AND a resolved DIURNAL CYCLE (Issue #21): sea cells hold the
(buffered) SST(theta) backbone with ~no diurnal swing (the 50 m mixed layer's huge heat capacity); land
cells carry a day-mean PLUS a per-cell diurnal range -> T_min(x,e)/T_max(x,e). The land backbone blends a
damp-coast and an arid-interior surface energy balance by continentality f=1-exp(-d_coast/L_cont): both the
day-mean and the diurnal range interpolate coast<->interior, and the min/max sit asymmetrically about the
mean (arid land has a hot noon spike over a radiatively-floored night). In the cascade the backbones come
from flatworld.surface's diurnally-resolved solve (the same machinery Stage A uses); standalone they fall
back to the same solve on a cost-bounded coarse grid. So the deliverable map now carries the ~10-60 K
land day/night swing -- the land/sea contrast and the frost margins the day-mean alone omits (the prior model
used only a day-mean minus a fixed continentality offset). The map carries BOTH the SURFACE (active-layer)
extremes T_min/T_max from the surface energy balance AND a screen-level AIR range T_air_min/T_air_max (Issue #21
follow-up): the air swing is a boundary-layer-DAMPED version of the surface swing (g<1, arid interior damping
more than the coast), anchored on the day-mean, with the night air min lifted above the skin min -- so air frost
is rarer than ground frost. The frost-aware Koppen E family + the frost-prone habitability class key on the AIR
min; the surface extremes are kept as the right quantity for the surface/land-process story.

Backbone inputs (SST(theta), T_land(theta), the aloft supply, the mean equatorward wind) are passed as
callables, so Stage C runs standalone with documented defaults OR consumes Stage A/B outputs (the
cascade). Equator side of the box is the warm edge; the cold side is poleward; the along-track (e)
boundary is periodic (a slab of the endless ribbon), and box-size independence is tested by widening e.
"""
from __future__ import annotations
import warnings as _warnings
import numpy as np
from scipy import ndimage as _ndi
from dataclasses import dataclass, field, replace
from . import constants as C
from . import geometry as g
from . import thermo as th
from . import surface as surf
from . import convergence as conv
from . import stagec_convergence as scv
from . import operators as ops
from . import bl_momentum as blm
from .topography import TopoGrid


# saturation column water [mm] proxy: rho_air * H_moist * qsat(T, p)  (kg/m^2 == mm)
def _wsat_column(T_K, p=C.P_SURF, H_moist=2500.0):
    return C.RHO_AIR0 * H_moist * th.qsat(np.asarray(T_K, float), p)


WSAT_ALOFT_CC_PER_K = 0.07     # Clausius-Clapeyron fractional capacity change per K of aloft warming
                               # (d ln qsat/dT ~ 6.8-7.5 %/K over the model's aloft range; the standard 7 %)


def _wa_supersat(W, Wsat, rh_crit):
    """PDF-integrated supersaturation tail [mm] for a top-hat sub-grid total-water PDF of half-width
    Delta = (1 - rh_crit) * Wsat about the grid-mean column W -- the vertically-integrated (mm) analog of
    ``radiation.cloud_pdf_partition``'s ``qc_eq``. With Q1 = (W - Wsat)/Delta the saturated-tail integral
    E[(W - Wsat)^+] is 0 for Q1 <= -1, (Delta/4)(1+Q1)^2 for -1 < Q1 < 1, and (W - Wsat) for Q1 >= 1.

    Unlike the hard-saturation excess max(W - Wsat, 0), this fires for grid-mean RH = W/Wsat > rh_crit
    (not only RH >= 1), so a rate-limited rainout built on it lets the aloft column equilibrate BELOW
    saturation with supply-dependent 2-D longitude structure (the RH=1 pinning fix, OPEN_ISSUES 2026-06-27).
    Reduces EXACTLY to max(W - Wsat, 0) as rh_crit -> 1 (hard-saturation limit, byte-identical default)."""
    if rh_crit >= 1.0:
        return np.maximum(W - Wsat, 0.0)                       # hard-saturation limit (byte-identical)
    Delta = np.maximum((1.0 - rh_crit) * Wsat, 1e-9)
    Q1 = (W - Wsat) / Delta
    return np.where(Q1 >= 1.0, W - Wsat,
                    np.where(Q1 <= -1.0, 0.0, 0.25 * Delta * (1.0 + Q1) ** 2))


# --- the physical geopotential-smoothing length (Refinement 2) -------------------------------------
# The ascent is w = -H_bl*div V = (H_bl/r) * del^2(Phi) -- a SECOND derivative of the geopotential, so
# sharp coastline/terrain features make tall narrow ascent spikes (and they SHARPEN with resolution, they
# do NOT converge away -- the staged-production warning). The physical fix is to smooth Phi over the length
# the boundary-layer flow actually integrates over, not respond to grid-scale coastline jumps. On this
# NON-rotating world (f=0) that scale is NOT a Rossby deformation radius -- L_d = c/f -> infinity here --
# but a FRICTIONAL / ADVECTIVE adjustment length: L_fric = V_char / r_fric, the fetch over which the damped
# flow relaxes a perturbation (equivalently V_char * tau_eq, the fetch over which air thermally adjusts to
# the local surface). The input is the characteristic balanced-wind V_char. An a-priori estimate is
# V_char ~ 20 m/s (the whole-sea median; -> ~200 km), BUT the operative wind in the sharp band/ribbon zone
# -- exactly where the del^2(Phi) spikes live and the smoothing must act -- is MEASURED cap-saturated at
# wind_cap = 40 m/s (~80% of ribbon-sea cells, robust across phi_smooth = 150-800 km), because grad(Phi)
# is steep there. So the operative V_char is wind_cap = 40 m/s and L_fric = wind_cap / r_fric = 400 km --
# the SELF-CONSISTENT frictional length (at L=400 the wind is still capped, so L=V/r holds; a fixed point).
# This CONFIRMS the existing 400 km and supersedes the a-priori V~20 -> 200: a phi_smooth sweep shows
# tightening to 200 km slides the band POLEWARD (theta~35 -> 37, off the circulation-set F-peak ~36.6) and
# RAISES the per-cell concentration without de-skewing the (already drain-gated) ribbon sea -- so tightening
# is contraindicated. DERIVED, not fitted: set by friction/advection at the measured operative wind, never
# swept until the skew vanishes (the cosmetic-de-skew trap). See OPEN_ISSUES (ribbon-sea refinements)/DESIGN.
V_CHAR_BL = 40.0           # operative balanced-wind speed [m/s] = wind_cap (the diagnostic wind saturates
                           # at the cap over the sharp-gradient band/ribbon zone where the smoothing acts;
                           # measured robust across phi_smooth). The whole-sea median ~20 is the broad-flow
                           # value, but it is NOT the operative wind for the spikes -- see the note above.
def frictional_smoothing_km(V_char=V_CHAR_BL, r_fric=1.0e-4):
    """Frictional/advective adjustment length L_fric = V_char / r_fric, in km (the f=0 smoothing scale).
    With the operative (cap-saturated) V_char = 40 m/s and r_fric = 1e-4/s this is 400 km."""
    return V_char / r_fric / 1.0e3


@dataclass
class StageCConfig:
    # --- backbone callables (theta_deg -> field); defaults make C runnable standalone ---
    sst_C_of_theta: object = None          # ocean surface temperature [C] vs latitude (maritime)
    Tland_C_of_theta: object = None        # land day-mean reference T [C] vs latitude
    aloft_supply_of_theta: object = None   # poleward-aloft moisture supply F [mm] vs latitude
    ubg_of_theta: object = None            # mean equatorward surface wind [m/s] (negative = toward eq)
    wetness_of_theta: object = None        # evaporation efficiency in [0,1] from Stage A's wetness ramp
                                           # (ocean=1, salt flat=0); the cascade wires A's own _wetness so
                                           # the theta<40 salt-flat hot zone does NOT evaporate as open
                                           # ocean (A->C consistency; same mask Stage B uses). Standalone:
                                           # falls back to A's 55->70 C SST ramp on this stage's backbone.
    # --- dynamics (f=0 boundary layer: diagnostic balanced wind; prognostic two-layer moisture) ---
    H_bl: float = 1000.0                   # boundary-layer depth [m] (sets pumping w = -H_bl * div V)
    beta_T: float = 30.0                   # thermal-low strength: dPhi = -beta_T * dT  [m^2/s^2 / K]
                                           # (principled: the BL thermal low ~ g*H_bl/T ~ 9.8*1000/300 ~ 33)
    g_terrain: float = 0.0                 # terrain geopotential factor (was 0.6; recalibrated 2026-06-17).
                                           # The orographic high is ALREADY carried by the lapse-cooled T_map
                                           # (Tmean_land -= lapse_land*h, ~6.5 K/km) -> via -beta_T*(T-Tref) it
                                           # adds beta_T*lapse*h ~ 0.195*h to Phi. An explicit g_terrain*g*h
                                           # (=5.88*h at 0.6) DOUBLE-COUNTED that ~30x and used the full
                                           # gravitational g*h scaling (which f=0 BL flow does NOT feel as a
                                           # horizontal PGF), pegging the diagnostic wind at wind_cap over even
                                           # moderate (~2 km) land. Orographic PRECIP is the separate oro_k
                                           # upslope term (and is minor here anyway -- moisture is lofted, not
                                           # low-level forced up a barrier; GH issue #36). Default 0 -> orography
                                           # steers through the (physically-scaled) lapse-thermal high only.
    r_fric: float = 1.0e-4                 # boundary-layer drag [1/s]; balanced wind V = u_bg - grad(Phi)/r
    phi_smooth_km: float | None = None     # geopotential smoothing length [km] = the DERIVED frictional/
                                           # advective adjustment length L_fric = V_char/r_fric = 400 km
                                           # (Refinement 2), with the operative cap-saturated band/ribbon wind
                                           # V_char = wind_cap = 40 m/s. NOT a Rossby deformation radius (f=0,
                                           # L_d->inf). Left None here and DERIVED in __post_init__ as
                                           # wind_cap/r_fric so overriding either keeps the documented fixed
                                           # point in sync (issue #88 item 4: the old hard-coded default
                                           # silently de-synced when r_fric/wind_cap were tuned). An explicit
                                           # float is honored as-is. PRINCIPLED, not a free fit (the prior
                                           # comment mis-called it a "deformation-radius scale"; the a-priori
                                           # V~20 -> 200 km is superseded -- tightening there slides the
                                           # circulation-set band). A grid-INDEPENDENT physical length, so the
                                           # del^2(Phi) spikes do not sharpen with resolution. See
                                           # frictional_smoothing_km. WARNING: because __post_init__ resolves
                                           # this field, dataclasses.replace() on a built config re-passes the
                                           # RESOLVED value (skipping re-derivation) -- to re-sync under a
                                           # wind_cap/r_fric change via replace(), pass phi_smooth_km=None
                                           # explicitly (config.stage_c does this).
    wind_cap: float = 40.0                 # cap |wind| [m/s] (numerical safety on sharp gradients)
    # --- moisture (two-layer) ---
    tau_evap: float = 2.0 * 86400.0        # sea-surface evaporation timescale [s]
    tau_rain: float = 6.0 * 3600.0         # large-scale (stratiform) rainout timescale [s]
    wa_capacity_factor: float = 1.0        # aloft CAPACITY ceiling = wa_capacity_factor * Wa_sat: after the
                                           # rate-limited drizzle, the aloft layer is rained back to at most
                                           # this multiple of saturation WITHIN the step, so a convergence/
                                           # detrainment node cannot park the column super-saturated (the
                                           # capacity-limited rainout, OPEN_ISSUES Brief 1). Default 1.0 =
                                           # the principled "a vapor column cannot stay super-saturated"
                                           # (rain to Wa_sat); >1 allows a small cloud-condensate holding
                                           # margin. FROZEN on physics, swept only to REPORT sensitivity --
                                           # it is NOT a band-magnitude fitter (it could be tuned to put the
                                           # ribbon-sea mean anywhere, so it is set blind to that number).
    wa_rh_crit: float = 1.0                # critical RH for the aloft stratiform precip ONSET (sub-grid
                                           # condensation). 1.0 = hard-saturation closure: rain only the
                                           # excess above RH=1 (and the capacity ceiling clips W_a to Wa_sat),
                                           # which pins RH=1 wherever convergence outpaces the drizzle and
                                           # erases the 2-D longitude structure of RH_aloft (OPEN_ISSUES
                                           # 2026-06-27). <1 routes both the drizzle and the ceiling through a
                                           # PDF supersaturation tail (`_wa_supersat`, the column analog of the
                                           # AD core's `radiation.cloud_pdf_partition`, rh_crit=CLOUD_RH_CRIT):
                                           # precip onsets at RH>rh_crit with PARTIAL rainout, so W_a settles
                                           # at a supply-dependent RH in (rh_crit,1) and RH_aloft keeps its
                                           # along-track structure. 1.0 is byte-identical to the prior closure.
    tau_cond: float = 1.0e3                # PDF stratiform condensation->rain timescale [s] for the
                                           # wa_rh_crit<1 path. PHYSICALLY ANCHORED to the AD core's
                                           # `qc_autoconv_tau` (=1e3 s, Kessler-class) -- the SAME condensation
                                           # rate the authoritative core uses, NOT a tuned knob. The aloft
                                           # column rains its PDF supersaturation tail at rate 1/tau_cond,
                                           # evaluated from the START-OF-STEP column, so dW_a/dt = supply -
                                           # qc(W_a)/tau_cond is a genuine O(dt) tendency with a dt-INDEPENDENT
                                           # fixed point (there is no dt in reality; the legacy /dt ceiling's
                                           # converged RH drifts with dt -> not converged physics, PR #231
                                           # review). Unused at wa_rh_crit>=1 (the legacy byte-identical path).
    tau_conv: float = 6.0 * 3600.0         # convective boundary-layer drainage timescale [s]
    conv_sst_C: float = 30.0               # T above which deep convection rains in place (hot inner sea)
    drain_subsidence_gate: bool = True     # gate the convective DRAIN on large-scale subsidence (Refinement
                                           # 1): deep convection is suppressed where the single cell's flow
                                           # SUBSIDES (descending air warms and caps the column from above),
                                           # even over a warm surface -- why Earth's subsidence zones stay dry
                                           # despite warm oceans. Without it the drain fires on surface warmth
                                           # alone (conv_active = T_map>conv_sst_C | ascent>thr) and rains
                                           # spurious local convection over the warm SUBSIDING ribbon sea
                                           # (the PR-2 finding). When True AND subsidence_of_theta is supplied,
                                           # the drain is multiplied by the per-row subsidence weight so the
                                           # subsiding ribbon stops raining its un-lofted moisture in place (it
                                           # advects equatorward into the cell's ascending branch instead). The
                                           # subsidence pattern is Stage B's w(theta) (cascade.subsidence_gate),
                                           # so this makes ribbon convection consistent with B's vertical motion
                                           # -- a real improvement that still inherits B's vertical-motion
                                           # softness. NOTE this is the DRAIN sibling of the loft-fraction gate
                                           # (f_loft_of_theta, Brief 2): Brief 2 gated only WHERE moisture lofts;
                                           # this gates the FATE of the drained moisture. The drain (the W_s
                                           # SINK) stays UNGATED so it always bounds W_s; under subsidence its
                                           # output is routed to surface-vapor EXPORT (vapor_export) -- the
                                           # equatorial-pathway proxy (rides equatorward, lofts off-map, returns
                                           # as F(theta)) -- instead of local rain. NOT precipitated (PE does not
                                           # apply under subsidence) and NOT re-injected in-domain (the drained
                                           # moisture under ascent is the PE fraction = surface-reaching RAIN, so
                                           # re-feeding it would double-count F(theta)'s already-airborne supply;
                                           # OPEN_ISSUES Ribbon-sea refinements 1b).
    subsidence_of_theta: object = None     # optional callable theta_deg -> [0,1] convection-allowed weight
                                           # (1 where the single cell ASCENDS / w>=0, 0 where it SUBSIDES /
                                           # w<0), from Stage B's mid-height w sign (cascade.subsidence_gate;
                                           # mirrors wetness_of_theta / f_loft_of_theta). When None the drain
                                           # gate is a NO-OP (subs_gate==1 everywhere), so the standalone/test
                                           # configs are bit-identical to the pre-refinement code -- the gate
                                           # only bites in the cascade, where Stage B's w is available.
    f_loft: float = 0.5                    # fraction of convective drainage detrained ALOFT vs rained in
                                           # place -- the convective/stratiform split (doc's chief unknown)
    f_loft_of_theta: object = None         # optional callable theta_deg -> f_loft in [0,1], the
                                           # PHYSICALLY-ANCHORED split from the entraining-plume model
                                           # (flatworld.plume_floft, OPEN_ISSUES Brief 2 / c3). When set,
                                           # it OVERRIDES the scalar f_loft per cross-track row (mirrors the
                                           # wetness_of_theta pattern); the cascade wires it from a Stage-A
                                           # column run with the FROZEN, LOCKED (eps, PE). When None the
                                           # scalar f_loft is used (the standalone/test default).
    # --- salt-aerosol CCN warm-rain coupling (GitHub #237, Stage-C side; the AD-core autoconv limb #243's
    #     sibling on the deliverable map) ---
    salt_ccn: bool = False                 # master switch; default OFF -> the precip code keeps its verbatim
                                           # lines, BYTE-IDENTICAL to the pre-feature map. When ON, scale Stage
                                           # C's WARM-rain conversion -- the convective precipitation efficiency
                                           # PE=(1-f_loft) AND the stratiform rainout DEMAND -- by (N_d/N_ref)^beta
                                           # per cross-track row, so the deliverable map P(x,e) carries the salt
                                           # CCN band-magnitude effect the AD core's autoconv limb (#243) measures
                                           # in the column. SIGN-AMBIGUOUS: beta<0 SUPPRESSION (sub-um salt CCN
                                           # delay coalescence -> slower warm rain), beta>0 INVIGORATION (giant
                                           # salt CCN seed large drops). Only the ratio N_d/N_ref enters; N_d==N_ref
                                           # is the clean zero-point (factor 1.0, exact no-op). Reported
                                           # direction-firm / magnitude NAMED-WALL-soft (aerosol microphysics +
                                           # crust-state erodibility), SWEPT -- never tuned to a target.
    salt_ccn_nd_of_theta: object = None    # callable theta_deg -> N_d [cm^-3] (mirrors f_loft_of_theta): the
                                           # AD core's emergent/prescribed N_d field, wired by the cascade
                                           # (stage_c_inputs) so the map's salt loading is SELF-CONSISTENT with the
                                           # column. When None the scalar salt_ccn_nd is broadcast (standalone/test).
    salt_ccn_nd: float | None = None       # scalar prescribed N_d [cm^-3] when the callable is None; None -> N_ref
                                           # (so master ON at the default is itself a no-op -- the byte-identical
                                           # zero-point that mirrors the AD core's salt_ccn_nd convention).
    salt_ccn_nd_ref: float = 100.0         # reference clean-maritime droplet number N_ref [cm^-3] (the today-
                                           # anchor; matches the AD core's salt_ccn_nd_ref). Only N_d/N_ref matters.
    salt_ccn_beta: float = -0.8            # warm-rain susceptibility exponent on (N_d/N_ref) (|beta| ~ the
                                           # susceptibility So in 0.3-1.2; matches the AD core's
                                           # salt_ccn_autoconv_beta). beta<0 suppression, beta>0 invigoration.
    tau_supply: float = 1.0 * 86400.0      # timescale to deliver the poleward-aloft band supply F(theta)
    wa_supply_sat_cap: float = 1.0         # cap the supply TARGET at cap*Wsat_aloft(theta) (default ON at the
                                           # physical value 1.0: the relaxation may fill a column TO its aloft
                                           # saturation capacity, never THROUGH it). Guards the boundary-condition
                                           # device against becoming a moisture pump where the target (esp. the
                                           # legacy 8 mm F-floor) exceeds a cold column's capacity -- the
                                           # lens_diurnal wet-belt artifact (2026-07-18, MODELING_LESSONS): 4-9
                                           # mm/day of forced rainout over theta 47-60 where Wsat ~ 0-2 mm. Never
                                           # binds where Wsat >= F (the band and equatorward; the prior L8e7
                                           # world's wet flank), so validated-world behavior is unchanged there.
                                           # inf -> the uncapped legacy law (byte-identical escape).
    # --- Track B / B1: resolved aloft supply (conserve-and-redistribute) ---
    resolved_supply: bool = False          # True: the poleward-aloft supply target is F_resolved(x,e) =
                                           # F_zonal(x)*blend(e), longitude-resolved by the local surface-vapor
                                           # export (the off-map equatorial return), instead of the prescribed
                                           # zonal F(theta). The DATACLASS default is False (byte-identical zonal
                                           # supply -- preserved for standalone/test/byte-identical paths). NB
                                           # the PRODUCTION profile flips this ON (J1 default-flip 2026-06-19,
                                           # total-reduction with config.SUPPLY_EREF_FROZEN): config.stage_c(
                                           # "production") sets resolved_supply=True, so the SHIPPED model has
                                           # the moisture-source teleconnection. See config.SUPPLY_EREF_FROZEN.
    supply_zonal_mixing: float = 0.0       # alpha in [0,1]: zonal MIXING of the export at the off-map
                                           # equatorial updraft. 0 (DEFAULT, the principled physical value) ->
                                           # fully localized: f=0 gives no zonal deflection of the cell's
                                           # return, and the diffusive zonal-homogenization time over the long
                                           # off-map equatorward transit is >> the transit (so negligible
                                           # off-map mixing; the resolved zonal mixing is already in kappa_q /
                                           # advWa). 1 -> zonally uniform (teleconnection damped). NOT a free
                                           # fitter: a frozen physical default + a one-sided SWEEP bounding the
                                           # unresolved off-map ADVECTIVE eddy mixing (kappa-like sensitivity).
    supply_ref_export: float = 0.0         # frozen all-sea reference export E_ref [mm/s, per-longitude cross-
                                           # track MEAN -- nx-portable, matches _export_col's units] that
                                           # produces the validated zonal F(theta) (the AD-core reference). 0
                                           # (DEFAULT) -> normalize by the RUNNING mean_e(export) == B1 (zonal-
                                           # mean supply conserved, no total reduction; byte-identical fallback).
                                           # >0 -> normalize by E_ref: a source block then drops the supply
                                           # BELOW F(theta) and reduces the TOTAL (the rest stays at F, not
                                           # enriched) -- the physical "less sea -> less moisture". Calibrated
                                           # from an all-sea reference run, frozen (the f_loft pattern), never fit.
    supply_export_ema: float = 0.01        # EMA weight for the per-longitude export feeding the blend (matches
                                           # the self.P / vapor_export EMAs; damps the supply<->export feedback)
    supply_diffusion_km: float = 0.0       # PHYSICAL along-track eddy diffusion of the off-map supply pathway,
                                           # as the diffusion length sigma [km]. 0 (DEFAULT) -> byte-identical (no
                                           # smear). The exported surface vapor diffuses along-track during its
                                           # off-map equatorward->loft->poleward transit; smearing g(e) by the
                                           # heat-kernel of that diffusion is the EXACT solution of d_t g = K d_ee g
                                           # over the transit time tau, with sigma = sqrt(2*K*tau). Physical estimate:
                                           # K_eddy ~ 1e5-1e6 m^2/s (the f=0 plane has NO baroclinic eddies -> low
                                           # end), tau ~ 1.2-2e7 s (the ~160,000 km off-map round trip at ~8 m/s)
                                           # -> sigma ~ 1,500-6,300 km. This is the PRINCIPLED replacement for the
                                           # box-referenced supply_zonal_mixing (which blends toward the finite-box
                                           # mean -- a box artifact on the infinite plane); a LOCAL diffusion has no
                                           # box dependence. A continent's equatorward reach NARROWER than sigma has
                                           # its evaporation shadow substantially washed; wider reaches keep a
                                           # shadow core with sigma-blurred edges. Periodic along e (mode="wrap").
    # --- Track C / C3: prognostic-Phi (moisture<->circulation feedback), default OFF -> static Gaussian Phi ---
    prognostic_phi: bool = False           # True: recompute the BL geopotential each `phi_update_every` steps
                                           # through C1's screened-Poisson inverse (bl_phi_screened.
                                           # geopotential_screened -- the UNIFIED thermal+latent operator),
                                           # under-relaxed by `phi_relax`, so the wind reorganizes around the
                                           # resolved latent heating (the C3 loop: P -> src_latent -> Phi ->
                                           # wind -> convergence -> moisture). False (DEFAULT) -> the shipped
                                           # STATIC Gaussian-smoothed thermal Phi (`_geopotential`), built once
                                           # in __init__ and never touched -> byte-identical. NOTE turning this
                                           # ON swaps Gaussian->screened-Poisson AND (if gms_geopot != 0) adds
                                           # the latent feedback; the J1 matched controls isolate the two
                                           # (gms_geopot=0 here = the operator-swap-only control). Lives on the
                                           # day-mean step() path only (NOT the diurnal _step_with_wind, which
                                           # carries its own breeze wind) -- C3 is the time-mean standing feedback.
    phi_relax: float = 0.1                 # under-relaxation of the Phi update: Phi <- (1-phi_relax)*Phi_old +
                                           # phi_relax*Phi_screened. The loop is a positive feedback, so this is
                                           # the engineering stabilizer (with the convergence diagnostic). 0 ->
                                           # frozen Phi; 1 -> full replacement each update. Small (~0.1) lets the
                                           # moisture relax between Phi kicks (the C3 convergence gate tunes it).
    phi_update_every: int = 1              # recompute Phi every N steps (Phi responds on the frictional/advective
                                           # fetch, slow vs the moisture sub-steps). Fires at iter 0, N, 2N, ...
                                           # (iter is incremented at the END of _step_with_wind). C3 FINDING
                                           # (`c3_close_loop refine`): a LARGE N adds a lagged-update SAWTOOTH that
                                           # inflates the residual convergence drift (~10x at N=25 vs N=1) WITHOUT
                                           # moving the fixed point. DEFAULT is N=1 (the validated-clean coupling --
                                           # the screened-Poisson solve is a cheap FFT, PR #163); RAISE it as a cost
                                           # lever only if the per-step inverse becomes a bottleneck, accepting the
                                           # bounded sawtooth (a discretization artifact, not a moisture-mode instability).
    gms_geopot: float = 0.0                # the C2 latent-source coefficient L*GMS [m^2/s^2 per mm/day] driving
                                           # src_latent = -gms_geopot*(P - P_baseline). 0 (DEFAULT) -> latent
                                           # inert even when prognostic_phi is on (the operator-swap-only path /
                                           # J1 control #2). C3 wires moist_stability.GMS_GEOPOT_FROZEN (~4.6);
                                           # FROZEN+LOCKED there, NOT a knob to fit here. Still default-OFF until
                                           # the J1 production verdict. Requires `_P_baseline` set when != 0.
    phi_source_smooth_km: float | None = None  # de-singularization length [km] for the screened-Poisson source
                                           # (caps the K_0 log-cusp at sharp rain cells / coastlines). None ->
                                           # C1's default SOURCE_DESING_FRAC/lam (~106 km, small vs the ~400 km
                                           # screening length). A FIXED PHYSICAL length, not grid-scale (issue #82).
    conv_gate_ramp_K: float = 0.0          # smooth the hard convective on/off gate (conv_active = T>conv_sst_C |
                                           # ascent>thr) to a tanh ramp of this width [K] -- the C3 gate-chatter
                                           # diagnostic (a hard switch in the feedback loop can limit-cycle at a
                                           # threshold and MASQUERADE as a structural runaway; ramping it
                                           # distinguishes gate-chatter from a genuine moisture-mode runaway).
                                           # 0 (DEFAULT) -> the exact hard boolean -> byte-identical. >0 -> the
                                           # drain weight ramps smoothly across the SST/ascent thresholds.
    # --- W2 (DIAGNOSTIC_WIND_ROADMAP): advect the moisture on the SELF-LIMITING momentum wind ---
    # Default OFF -> the moisture rides the diagnostic damped-balance wind `_balanced_wind` (byte-identical).
    # ON -> `step()` re-solves the nonlinear `bl_momentum` surface field (Bernoulli-self-limiting, NO cap) on
    # a cadence (warm-started) and advects the two-layer moisture on it via `_step_with_wind` -- which rebuilds
    # the poleward-aloft return as the kinematic departure off THIS surface wind (W1 scope (a)), so the swap is
    # a single re-wire at the advecting-wind level, the aloft layer following automatically. The momentum wind
    # reads its OWN virtual-temperature wind-Phi (`bl_momentum.build_wind_field`, decoupled from the moisture-Phi
    # `_Phi`) -- UNLESS `prognostic_phi` is ALSO on, the W3 (J1b) FUSION, in which case the momentum wind reads
    # the SAME prognostic screened-Poisson `_Phi` (thermal + latent) that `_update_phi_prognostic` evolves,
    # closing moisture->P->src_latent->Phi->momentum wind->convergence->moisture. NB this is the
    # MOISTURE-ADVECTING wind (distinct from cascade's `bl_momentum_wind`, which only switches the REPORTED wind
    # post-hoc). When ON, `cfg.phi_smooth_km` / `_balanced_wind` become inert (the balance wind no longer
    # advects); `_Phi` is inert too in W2 (prognostic_phi OFF) but is the LIVE forcing in the W3 fusion.
    momentum_advection: bool = False
    momentum_update_every: int = 50        # re-solve the momentum wind every N moisture steps (the COST lever;
                                           # warm-started from the prev field via u0/v0 -> a near-static wind-Phi
                                           # re-converges in a few iters). The wind-Phi drifts only weakly here
                                           # (static T_map; the virtual-T W_s feedback is second-order), so a
                                           # modest cadence is ample; W3's prognostic-Phi feedback is what makes
                                           # the re-solve load-bearing. Fires at iter 0, N, 2N, ...
    momentum_courant_target: float = 0.5   # moisture-CFL target the momentum-wind dt is sized to (DIRECTIONAL
                                           # Courant = max(umax/dx, vmax/de)*dt over BOTH layers -- cross-track
                                           # over dx, along-track over de; PR #245 note 1). The momentum wind has
                                           # NO cap, so dt is RE-DERIVED against its actual peak speed, not
                                           # wind_cap (W2 risk item): each step() advects at a CFL-safe dt =
                                           # cfg.dt/n_sub (a smaller step, more steps -- the W0 approach),
                                           # shrinking only when the peak would breach the target. On the
                                           # canonical world the momentum peak (~22 m/s aloft) << 40 m/s cap, so
                                           # Courant RELAXES and dt = cfg.dt (n_sub=1, no shrink -- W0 confirmed).
    momentum_cfg: object = None            # an explicit bl_momentum.BLMomentumConfig for the in-loop solve, or
                                           # None -> a default BLMomentumConfig (quadratic drag, the swept arm).
    momentum_phi_smooth_km: float | None = None  # the momentum wind-Phi smoothing length [km]. None (DEFAULT)
                                           # -> RE-DERIVE from the momentum field's own drag closure via
                                           # `bl_momentum.momentum_smoothing_km` (= H_bl/C_D ~ 833 km for the
                                           # quadratic default), NOT the cap-saturated wind_cap/r_fric=400 km the
                                           # diagnostic balance uses (an inherited 400 would silently re-impose
                                           # the diagnostic-wind length scale -- W2 risk item). A float overrides.
    H_aloft_km: float = 4.0                # height the aloft layer's saturation is evaluated at [km]
    lapse_aloft: float = 6.5e-3            # lapse to the aloft layer [K/m]
    # --- resolved aloft thermodynamics from the AD core (Workstream C) ---------------------------------
    # By default Stage C derives the aloft-layer saturation moisture path Wsat_aloft [mm] from a FAKE
    # FIXED-LAPSE aloft temperature (T_aloft = SST - lapse_aloft*H_aloft; the two knots above). The AD core
    # already RESOLVES the aloft profile and exposes it through cascade.stage_c_inputs(); these optional
    # callables (theta_deg -> value) carry it in so Stage C can consume the real profile instead of the
    # fixed-lapse stand-in. They are wired by the cascade's core="ad" branch and IGNORED unless
    # `aloft_thermo_from_core` is ON, so supplying them alone changes nothing (byte-identical).
    wsat_aloft_gamma: float = 0.25          # aloft-capacity LAND modulation (2026-07-18): the fraction of a
                                            # column's SURFACE temperature anomaly (vs the row's SST -- the
                                            # all-sea AD backbone's reference, so sea cells are EXACTLY
                                            # unperturbed) that reaches the aloft moisture layer:
                                            #   Wsat_map = Wsat_row * exp(CC * gamma * (T_map - SST_row)).
                                            # Without it the aloft capacity is a pure zonal profile and every
                                            # capacity-set transition (the cold-edge stratiform shutoff) is a
                                            # razor-straight line in e; physically, standing land-forced aloft
                                            # warmth should meander it by a fraction of a degree around the
                                            # continents. Best guess 0.25 from the Earth anchor ~0.3 (lower
                                            # free troposphere over strong stationary heating; WTG tropics
                                            # ~0.1, midlat stationary waves ~0.3-0.4, deep desert BL ~0.5)
                                            # shaded DOWN for f=0: infinite deformation radius -> gravity-wave
                                            # homogenization of the free troposphere at full strength -- and
                                            # UP for the standing, seasonless +20-40 K land forcing. SWEPT
                                            # 0.1-0.5; magnitude charged to the named wall = aloft thermal
                                            # structure unresolved (2-D (theta,z) core carries no e-structure).
                                            # 0/None -> byte-identical per-row legacy capacity.
    Wsat_aloft_of_theta: object = None      # resolved aloft saturation moisture path Wsat_aloft [mm] vs latitude
    T_aloft_of_theta: object = None         # resolved aloft-layer temperature [C] vs latitude (Wsat fallback source)
    RH_aloft_of_theta: object = None        # resolved aloft relative humidity [0,1] vs latitude (provenance only)
    aloft_thermo_from_core: bool = field(   # master switch (env COSMO_ALOFT_THERMO_FROM_CORE); default OFF ->
        default_factory=lambda: C._env_flag("COSMO_ALOFT_THERMO_FROM_CORE", False))  # keep the fixed-lapse path
                                            # (byte-identical). ON + a supplied Wsat_aloft_of_theta (or, as a
                                            # fallback, T_aloft_of_theta) -> Wsat_aloft is taken from the core's
                                            # resolved aloft profile rather than the SST-minus-fixed-lapse stand-in.
    recycle: float = 0.3                   # land evapotranspiration recycling fraction (Earth ~0.3-0.5)
    oro_k: float = 4.0e-4                  # orographic rainout per (upslope * wind) [mm/(m) scaling]
    conv_k: float = 0.5                    # convective uplift efficiency (W_s -> aloft where ascending)
    kappa_q: float = 6.0e3                 # moisture eddy diffusivity [m^2/s]
    u_aloft_frac: float = 1.0              # aloft poleward wind = u_aloft_frac * |ubg| (opposite surface)
    aloft_departure_sign: float = 1.0      # sign of the DYNAMICAL departure carried by the aloft layer
                                           # (issue #88 item 1). The aloft wind = poleward mean + sign*(V -
                                           # V_bg). +1.0 (DEFAULT, the shipped choice): the aloft departure is
                                           # surface-signed, so the aloft layer CONVERGES where the surface
                                           # converges -- a convergence node receives moisture in BOTH layers
                                           # plus local detrainment, with NO upper-level venting (plausibly
                                           # amplifies the top-5%-cell precip concentration). -1.0: the aloft
                                           # departure is the thermally-direct RETURN branch (divergent over a
                                           # surface convergence node), so the node VENTS aloft. A FLAGGED
                                           # modeling choice, NOT a tuned knob: flip to A/B the node skew. Band
                                           # PLACEMENT is F(theta)-set and unaffected; this only moves WHERE the
                                           # per-latitude rain concentrates. Default keeps the production map
                                           # byte-identical (see _issue88_aloft_ab.py for the A/B).
    # --- temperature map: day-mean backbone + the resolved DIURNAL CYCLE (Issue #21) ---
    lapse_land: float = 6.5e-3             # surface-air lapse with altitude [K/m]
    continentality_K: float = 8.0          # interior day-mean OFFSET [K] vs the coastal mean: the interior
                                           # day-mean defaults to coast - continentality_K when
                                           # Tland_interior_C_of_theta is not given. NOTE the cascade
                                           # SUPERSEDES this with the surface-EB-solved interior mean, which on
                                           # a SEASONLESS world runs slightly WARMER than the coast (no cold
                                           # winter; the real continental signal is the big diurnal SWING, not a
                                           # mean cooling). Kept as the standalone default so the standalone
                                           # day-mean map is bit-unchanged (see _build_temperature_map).
    L_cont_km: float = 400.0               # continentality e-folding distance from coast [km]
    maritime_L_km: float = 720.0           # MARITIME MODERATION e-folding [km] (2026-07-26): marine air
                                           # advected inland (the sea-/land-breeze exchange the doc's day
                                           # appendix derives) pulls the coastal land DAY-MEAN toward the
                                           # row's SST by m_mean = exp(-d_coast/maritime_L_km). This length
                                           # governs the day-MEAN only; the diurnal RANGE moderates over the
                                           # SHORTER, separate maritime_range_L_km (see below). Without it
                                           # the shoreline is a
                                           # raw where(sea, SST, T_land) step -- invisible while land~sea,
                                           # a +20 K zero-width cliff on the hot-land lens_diurnal world
                                           # (which also sharpened the coastal thermal low via Phi(T)).
                                           # CALIBRATED ON-WORLD, 720 ADOPTED (Jake, 2026-07-26; the
                                           # AS-1 FAIL path -- flatworld/MARITIME_L_CALIBRATION_RESULTS.md):
                                           # the #113 breeze intrusion re-measured on the CANONICAL
                                           # lens_diurnal world reads 722/725 km in the habitable band
                                           # across a 1.5x grid ladder (GRID-STABLE, unlike the
                                           # grid-labile full-domain 638->724) -- the deliverable's own
                                           # latitudes pin the value. NAMED WALL (recorded, not
                                           # resolved): the intrusion is a PROXY -- it measures the
                                           # breeze-reversal flow reach, not the day-mean moderation
                                           # this term applies; the direct-construction observable
                                           # (the marine-parcel day-fetch, ~420 km band) sits 1.7x
                                           # below, the physical preference between them is unsettled,
                                           # and the two ends differ by ~23% of band-land Koeppen
                                           # cells on the 2026-08-04 edition (720<->420: 23.2% band
                                           # flips / 5.96 K p95 -- as1_uncertainty/RESULTS.json; the
                                           # adoption-era A/B vs the old 610 ref, 8-14%, is preserved
                                           # at as1_uncertainty/RESULTS_2026-07-26_ref610.json).
                                           # (History: 610 was the b23/L=8e7 measurement inherited
                                           # untested across the L change; the AS-1 gate caught it --
                                           # as1_sweep/RESULTS.json quantifies the map's sensitivity.)
                                           # A derived scale, swept for
                                           # robustness, never tuned to a target. d_coast is distance to
                                           # NON-SALT-FLAT (wet-ocean) sea -- NOT "liquid": frozen sea IS
                                           # included, which is exactly what lets frozen sea chill its coast.
                                           # Only salt-flat sea (wetness<0.5) is excluded, so it moderates
                                           # nothing (a dry barrier); the frozen coast then runs COLDER than
                                           # the interior poleward of the ice line -- inverted from Earth,
                                           # physically right for arid-interior EB). The blend
                                           # acts at SEA LEVEL (before the altitude lapse) so coastal
                                           # highlands stay lapse-cooled. 0/None -> OFF, byte-identical
                                           # legacy sharp-shoreline map.
    maritime_range_L_km: "float | None" = 150.0   # MARITIME MODERATION of the diurnal RANGE, e-folding [km],
                                           # DECOUPLED from the day-mean length above (coastal-diurnal-range
                                           # fix). The day-MEAN is set by the deep advected marine air MASS
                                           # (~maritime_L_km fetch), but the 2 m diurnal SWING is a LOCAL
                                           # surface-layer response the land skin regenerates every day and
                                           # the (shallow, daily-reversing) sea breeze only PARTIALLY
                                           # flushes -- a much shorter reach. Moderating the range over the
                                           # 610 km mean-fetch collapsed the coastal swing to ~0 (a coast
                                           # read high==low), losing the real Bermuda-like diurnal range;
                                           # ~150 km (a few x Earth's sea-breeze penetration, scaled for the
                                           # stronger seasonless onshore flow) keeps it. DISPLAY-GRADE /
                                           # soft: the direction (coasts keep a real swing) is firm, the km
                                           # is a swept scale. None -> reuse maritime_L_km (legacy coupling,
                                           # byte-identical old range with diurnal_range_sea_K=0); 0 -> no
                                           # range moderation at all (lean on the bl_damp_g BL damping).
    # The land DIURNAL CYCLE, resolved per cell (Issue #21): sea ~0 swing (huge mixed-layer heat capacity);
    # land swings 10-60 K with insolation (latitude) and surface heat capacity (damp coast -> arid interior).
    # The per-cross-track-row backbones below are blended coast<->interior by the SAME continentality ramp
    # f = 1 - exp(-d_coast/L_cont). The cascade fills them from surface.solve_periodic_surface (reusing Stage
    # A's diurnally-resolved surface EB); standalone they fall back to the linearized-EBM estimate.
    Tland_interior_C_of_theta: object = None     # arid-interior land day-mean [C]; default coast - continentality_K
    dT_diurnal_coast_of_theta: object = None     # coastal (damp) land diurnal RANGE [K]; default linearized-EBM
    dT_diurnal_interior_of_theta: object = None  # arid-interior land diurnal RANGE [K]; default linearized-EBM (bigger)
    diurnal_skew_coast: float = 0.58       # fraction of the diurnal range ABOVE the day-mean for damp land:
                                           # Tmax = mean + skew*range, Tmin = mean - (1-skew)*range (~symmetric)
    diurnal_skew_interior: float = 0.72    # arid land is noon-spike heavy (hot peak over a radiatively-floored night)
    diurnal_range_sea_K: float = 1.0       # ocean 2 m AIR diurnal range [K]. The SST swing is ~0.1 K (50 m
                                           # mixed layer), but the near-surface marine AIR is not slaved to
                                           # SST -- direct shortwave absorption in the marine boundary layer,
                                           # the small diurnal warm layer, and daytime BL deepening give the
                                           # 2 m air ~0.5-1.5 K over open ocean. Was 0.0 (the SST number
                                           # mis-applied to air, so sea cells read high==low EXACTLY); ~1 K
                                           # is the observed marine-air swing. Also floors the fully-
                                           # moderated coastal range. Soft/display-grade. Drives the AIR
                                           # extremes only -- the sea SKIN uses the separate constant below.
    diurnal_range_sea_surface_K: float = 0.1   # ocean SKIN (SST) diurnal range [K], driving T_min_C/T_max_C
                                           # over sea. Kept SEPARATE from the air swing above (PR #389
                                           # review round 2, finding 3): the two are physically different
                                           # numbers -- a 50 m mixed layer swings ~0.1 K while the 2 m air
                                           # above it swings ~1 K -- and driving both from one field just
                                           # inverts the original bug, mis-applying the AIR number to the
                                           # SKIN (10x too large) instead of the SST number to the air.
                                           # T_max_C/T_min_C are NOT internal: the sea-breeze figures
                                           # (make_seabreeze_fig, _bl_diurnal_breeze_plots) drive the
                                           # density-current scale sqrt(2*beta*dT) off this swing, and
                                           # climate_classes / habitable_land_band gate on T_max_C.
    # --- screen-level (2 m) AIR diurnal range: a boundary-layer DAMPING of the surface range (Issue #21
    # follow-up). The emitted surface (skin) min/max OVERSTATE the 2 m AIR swing: a deep daytime convective
    # boundary layer mixes the sensible-heat flux through a large air mass (so the air max rises far less than
    # the skin max), and a shallow STABLE night layer decouples the air from the radiatively-cooling skin (so
    # the air night MIN sits WARMER than the skin min -- exactly why AIR frost is rarer than ground frost). The
    # arid INTERIOR damps MORE than the coast: dry bare soil super-heats and decouples strongly (air/skin range
    # ratio ~0.4-0.5), while a moist, well-coupled coast keeps skin and air close (~0.7-0.8), so g_interior <
    # g_coast. dT_air = g*dT_surf with g continentality-blended (the SAME f ramp the surface blend uses); the
    # air cycle is ANCHORED on the day-mean (symmetric air_skew) so its centre cannot drift warm. These are
    # documented INPUT knobs, SWEPT (the coast/interior contrast is weakly constrained -- sweep wide), not
    # tuned to a target; T_map_C (the deliverable day-mean) is untouched, so this is strictly additive.
    bl_damp_g_coast: float = 0.76          # range-damping g at the (moist, well-coupled) coast: WEAKER damping
    bl_damp_g_interior: float = 0.42       # range-damping g in the (arid, decoupled) interior: STRONGER damping
    air_skew: float = 0.5                  # frac of the air range ABOVE the day-mean; 0.5 = symmetric (centre at
                                           # Tmean, no warm drift). <0.5 encodes a bounded COLD skew; never >0.5.
    # --- GEO-22 nested refinement (default OFF -> byte-identical) ---
    nest_bc: object = None                 # a climate_nest.NestBC (duck-typed; no import) carrying the
                                           # COARSE full-domain solution interpolated onto THIS grid, the
                                           # boundary-rim weights/normals, the rim relaxation timescale,
                                           # and (optionally) the full-WINDOW land mask window_land/e_off
                                           # so a tiled solve's coast distance is computed window-wide
                                           # (one coastline for every tile, no T seams).
                                           # None (DEFAULT, and every non-nested caller) -> no warm start,
                                           # no rim term -- the step is verbatim. When set: W_s/W_a (and P)
                                           # WARM-START from the coarse fields, and each step relaxes the
                                           # boundary rim toward them on INFLOW only (outflow left free --
                                           # the moisture advection is hyperbolic, so only inbound
                                           # characteristics carry information the nest cannot compute).
                                           # Built by climate_nest.build_nest_bc; driver
                                           # refine_climate_window.py. NB the water_closure_* diagnostics
                                           # are NOT meaningful under a nest (open domain -- the budget
                                           # closes only jointly with the coarse parent).
    # --- time stepping / convergence ---
    # CFL note (issue #88 item 4): the explicit upwind moisture advection needs Courant = wind_cap*dt/de < 1.
    # Production (dt=600 s, de~30 km, wind_cap=40 m/s) sits at worst-case Courant ~0.84 -- stable but THIN.
    # Raising wind_cap OR dt independently crosses 1 (instability); raise one only with the other lowered, or
    # refine de. The standalone default below (dt=900) is paired with the coarser test grids that keep de large.
    dt: float = 900.0
    tol: float = 5e-4                      # steady-state tol on normalized max|d W_a|/step
    check_every: int = 200
    patience: int = 4
    max_iter: int = 3_000_000
    max_seconds: float = 9.0 * 3600
    # --- convergence RULE (PR #852 fix; flatworld/stagec_convergence.py has the argument) ---
    # "per_step" (DEFAULT, bit-identical): ``tol`` on max|dW_a|/<|W_a|> PER STEP, checked every
    # ``check_every`` steps -- the historical rule, whose physical strictness scales with ``dt`` (a nest at
    # dt=127 s admits ~7x the drift a parent at dt=900 s does). "model_time": the same change divided by the
    # check interval's MODEL SECONDS, tested against ``conv_tol_per_s`` [1/s], with the checks spaced
    # ``conv_check_s`` model seconds apart -- one physical standard at any dt. Deliverable routes select it
    # (stagec_convergence.deliverable_overrides(): the production profile's intent per unit model time, no
    # wall clock, and the ``conv_max_model_s`` backstop, which caps max_iter at ceil(conv_max_model_s/dt)).
    conv_rule: str = "per_step"
    conv_tol_per_s: float = 1e-4 / 600.0
    conv_check_s: float = 3.0e5
    conv_max_model_s: float | None = None
    checkpoint_path: str | None = None
    checkpoint_every: int = 20000

    def __post_init__(self):
        # Derive phi_smooth_km from the friction/cap physics unless explicitly set, so the documented
        # frictional fixed point L_fric = wind_cap / r_fric stays consistent when r_fric or wind_cap is
        # overridden (issue #88 item 4). Default (wind_cap=40, r_fric=1e-4) -> 400 km, byte-identical to the
        # prior hard-coded default. NOT a Rossby radius (f=0); see frictional_smoothing_km.
        if self.phi_smooth_km is None:
            self.phi_smooth_km = frictional_smoothing_km(V_char=self.wind_cap, r_fric=self.r_fric)
        if self.conv_rule not in ("per_step", "model_time"):
            raise ValueError(f"StageCConfig.conv_rule must be 'per_step' or 'model_time', got {self.conv_rule!r}")


class StageCMap:
    def __init__(self, topo: TopoGrid, cfg: StageCConfig | None = None):
        self.topo = topo
        self.cfg = cfg or StageCConfig()
        c = self.cfg
        self.x = topo.x; self.e = topo.e
        self.nx, self.ne = topo.land.shape
        self.dx = topo.dx; self.de = topo.de
        self.theta_deg = topo.theta_deg
        self.land = topo.land
        self.h = topo.height
        self.sea = ~self.land
        # --- backbone fields on the grid (per cross-track row, broadcast over e) ---
        th_d = self.theta_deg
        self.SST_C = self._call(c.sst_C_of_theta, self._default_sst, th_d)
        self.Tland_C = self._call(c.Tland_C_of_theta, self._default_tland, th_d)   # coastal/damp land day-mean
        # --- DIURNAL CYCLE backbones (Issue #21): interior day-mean + the coast/interior diurnal ranges ---
        # interior (arid) land day-mean [C]: cascade supplies the surface-EB-solved value; standalone default is
        # the legacy coast - continentality_K (kept so the standalone day-mean map is bit-identical -- see
        # _build_temperature_map). On a seasonless world the solved interior actually runs slightly WARMER.
        if c.Tland_interior_C_of_theta is not None:
            self.Tland_int_C = np.asarray(c.Tland_interior_C_of_theta(th_d), float)
        else:
            self.Tland_int_C = self.Tland_C - c.continentality_K
        # land diurnal RANGE [K] per cross-track row, damp-coast and arid-interior endpoints: the resolved
        # diurnal-amplitude field. Cascade -> surface.solve_periodic_surface; standalone -> linearized-EBM.
        self.dT_coast = self._diurnal_range_row(c.dT_diurnal_coast_of_theta, th_d, C.C_MOIST_LAND, 0.40, self.Tland_C)
        self.dT_int = self._diurnal_range_row(c.dT_diurnal_interior_of_theta, th_d, C.C_ARID_LAND, 0.10, self.Tland_int_C)
        self.F = self._call(c.aloft_supply_of_theta, self._default_supply, th_d)   # [mm]
        self.ubg = self._call(c.ubg_of_theta, self._default_ubg, th_d)             # [m/s] (<0 eqward)
        # evaporation efficiency [0,1] (ocean<->salt flat): gates sea-surface evaporation so the hot
        # salt-flat zone (SST>70 C, where Stage A boiled the sea to evaporite) does NOT evaporate as
        # open ocean -- the A->C consistency fix for the hot-zone precip blow-up (OPEN_ISSUES c2).
        self.wetness = self._wetness_col(c.wetness_of_theta, th_d)                 # (nx,) in [0,1]
        self.wet_col = self.wetness[:, None]                                       # broadcast over e
        # Continentality distance: to LIQUID OCEAN only. A hot SALT-FLAT sea (wetness < 0.5; SST > ~62.5 C,
        # the SAME midpoint classify_npz uses to RENDER it as salt flat) has dried out and cut itself off
        # from the ocean, so it provides NO ocean buffering: land beside it is effectively deep interior and
        # should bake (-> salt flat), not inherit a damped coastal diurnal range that spuriously keeps its
        # surface peak below the SF gate (Jake, 2026-06-27). Salt-flat sea is a dry BARRIER, not a coast.
        # Byte-identical to the geometric distance when no sea is salt-flat (the default/test worlds have
        # wetness == 1 everywhere), so this only bites on the hot cascade -- and it leaves genuine liquid-sea
        # coasts (their nearest wet ocean is unchanged) correctly damped.
        saltflat_sea = self.sea & (self.wet_col < 0.5)
        wet_ocean = self.sea & ~saltflat_sea
        if saltflat_sea.any() and wet_ocean.any():
            self.d_coast = _ndi.distance_transform_edt(~wet_ocean, sampling=[topo.dx / 1e3, topo.de / 1e3])
        else:
            self.d_coast = topo.distance_to_coast_km()         # no salt flats (or no liquid ocean): geometric
        # --- GEO-22 nested refinement: full-WINDOW coast distance (default absent -> verbatim). An EDT
        # cannot see coastline beyond its domain edge, so a TILE's own d_coast ends at its ~rim_km
        # overlap -- less than one maritime_L_km -- and per-tile distances would disagree across tile
        # seams (seams in the T map, in deliverable cells). When the nest BC carries the full-window
        # land mask (climate_nest.NestBC.window_land, this domain at column offset e_off), recompute
        # d_coast on the WHOLE window with the SAME salt-flat-aware convention (wet_col is theta-keyed,
        # so it broadcasts over the window's columns identically for every tile) and slice -- every
        # tile then sees ONE coastline. Must run BEFORE _build_temperature_map (the only d_coast
        # consumer). ---
        if c.nest_bc is not None and getattr(c.nest_bc, "window_land", None) is not None:
            wl = np.asarray(c.nest_bc.window_land, bool)
            e0 = int(getattr(c.nest_bc, "e_off", 0))
            if wl.shape[0] != self.nx or not (0 <= e0 and e0 + self.ne <= wl.shape[1]):
                raise ValueError(f"nest_bc.window_land {wl.shape} with e_off={e0} does not contain "
                                 f"this {(self.nx, self.ne)} domain -- BC built for a different window?")
            sea_w = ~wl
            saltflat_w = sea_w & (self.wet_col < 0.5)
            wet_ocean_w = sea_w & ~saltflat_w
            if saltflat_w.any() and wet_ocean_w.any():
                d_w = _ndi.distance_transform_edt(~wet_ocean_w, sampling=[topo.dx / 1e3, topo.de / 1e3])
            elif not wl.any():
                d_w = np.zeros(wl.shape)                       # no land anywhere: distance is moot
            else:
                d_w = _ndi.distance_transform_edt(wl, sampling=[topo.dx / 1e3, topo.de / 1e3])
            self.d_coast = d_w[:, e0:e0 + self.ne]
        # convective<->stratiform split f_loft per cross-track row: the plume-anchored f_loft_of_theta
        # (OPEN_ISSUES Brief 2) when supplied, else the scalar default broadcast. Clipped to [0,1].
        if c.f_loft_of_theta is not None:
            self.f_loft = np.clip(np.asarray(c.f_loft_of_theta(th_d), float), 0.0, 1.0)
        else:
            self.f_loft = np.full(self.nx, float(c.f_loft))
        self.f_loft_col = self.f_loft[:, None]                                     # broadcast over e
        # salt-aerosol CCN warm-rain factor per cross-track row (GitHub #237, Stage-C side): scale the warm-rain
        # conversion (convective PE + stratiform rainout demand) by (N_d/N_ref)^beta. None when OFF -> the precip
        # code keeps its verbatim lines (byte-identical). N_d from the wired callable (the AD-core field, so the
        # map's loading is self-consistent with the column) or the scalar fallback. At N_d==N_ref the factor is
        # exactly 1.0 -- the clean zero-point (master ON at the default is a no-op).
        if c.salt_ccn:
            if c.salt_ccn_nd_of_theta is not None:
                nd = np.asarray(c.salt_ccn_nd_of_theta(th_d), float)
            else:
                nd = np.full(self.nx, float(c.salt_ccn_nd if c.salt_ccn_nd is not None else c.salt_ccn_nd_ref))
            self.N_d = nd
            self.salt_warm_col = ((nd / c.salt_ccn_nd_ref) ** c.salt_ccn_beta)[:, None]   # (nx,1), over e
        else:
            self.N_d = None
            self.salt_warm_col = None
        # convective-drain subsidence gate (Refinement 1): per cross-track row in [0,1], 1 where the single
        # cell ASCENDS (convection allowed), 0 where it SUBSIDES (deep convection capped from above). From
        # Stage B's w(theta) sign when supplied; a NO-OP (all ones) standalone, so the test/standalone
        # configs are unchanged. Clipped to [0,1].
        if c.subsidence_of_theta is not None and c.drain_subsidence_gate:
            self.subs_gate = np.clip(np.asarray(c.subsidence_of_theta(th_d), float), 0.0, 1.0)
        else:
            self.subs_gate = np.ones(self.nx)
        self.subs_gate_col = self.subs_gate[:, None]                               # broadcast over e
        # aloft saturation moisture path Wsat_aloft [mm] per row (cooler air aloft). DEFAULT: the fixed-lapse
        # stand-in T_aloft = SST - lapse_aloft*H_aloft. Workstream C: when aloft_thermo_from_core is ON and the
        # cascade wired the AD core's RESOLVED aloft profile, take Wsat_aloft from that profile instead --
        # directly from the resolved column (Wsat_aloft_of_theta) or, failing that, from the resolved aloft
        # temperature (T_aloft_of_theta [C]) via the same _wsat_column law. The OFF path is byte-identical: the
        # branch is not entered unless the flag is on AND a callable is present, so supplying the callables with
        # the flag off leaves this line verbatim. RH_aloft is carried for provenance/diagnostics only.
        self.RH_aloft_core = (np.asarray(c.RH_aloft_of_theta(th_d), float)
                              if c.RH_aloft_of_theta is not None else None)
        if c.aloft_thermo_from_core and c.Wsat_aloft_of_theta is not None:
            self.Wsat_aloft = np.maximum(np.asarray(c.Wsat_aloft_of_theta(th_d), float), 0.0)  # [mm], resolved
        elif c.aloft_thermo_from_core and c.T_aloft_of_theta is not None:
            # T-only fallback: recompute Wsat via Stage C's _wsat_column, which need NOT be the identical
            # saturation law the core used to produce Wsat_aloft_mm. Prefer supplying Wsat_aloft_of_theta
            # (the branch above) when enabling this flag for real, so Wsat matches the core exactly.
            T_aloft_K = np.asarray(c.T_aloft_of_theta(th_d), float) + C.T_FREEZE   # resolved aloft T [C]->[K]
            self.Wsat_aloft = _wsat_column(np.maximum(T_aloft_K, 180.0))           # [mm], per row
        else:
            T_aloft_K = (self.SST_C + C.T_FREEZE) - c.lapse_aloft * (c.H_aloft_km * 1e3)   # fixed-lapse stand-in
            self.Wsat_aloft = _wsat_column(np.maximum(T_aloft_K, 180.0))           # [mm], per row
        # --- temperature map (diagnostic, set once; maritime sea + continental land) ---
        self.T_map_C = self._build_temperature_map()
        # --- 2-D aloft capacity map (wsat_aloft_gamma): per-row Wsat modulated by the column's own
        # surface-T anomaly vs the row's SST (the all-sea backbone reference -> sea cells exactly
        # unperturbed; lapse-cooled highlands hold LESS aloft). gamma=0/None keeps the verbatim
        # per-row (nx,1) broadcast -- byte-identical legacy. ---
        g_wsat = float(c.wsat_aloft_gamma) if c.wsat_aloft_gamma else 0.0
        if g_wsat > 0.0:
            dT_col = self.T_map_C - self.SST_C[:, None]
            self.Wsat_aloft_map = self.Wsat_aloft[:, None] * np.exp(WSAT_ALOFT_CC_PER_K * g_wsat * dT_col)
        else:
            self.Wsat_aloft_map = self.Wsat_aloft[:, None]
        # --- prognostic state ---
        self.u = np.broadcast_to(self.ubg[:, None], (self.nx, self.ne)).astype(float).copy()  # x-wind
        self.v = np.zeros((self.nx, self.ne))                                       # e-wind
        self.W_s = np.zeros((self.nx, self.ne))                                     # surface moisture [mm]
        self.W_a = np.zeros((self.nx, self.ne))         # aloft moisture [mm]: self-generated from evap
        self.P = np.zeros((self.nx, self.ne))                                       # precip rate [mm/day]
        self.vapor_export = 0.0     # EMA of the total surface-vapor EXPORT [mm/day, summed]: the moisture
                                    # removed under subsidence and routed to the equatorial pathway (rides
                                    # equatorward, lofts off-map, returns aloft as F(theta)) rather than
                                    # precipitated locally (Refinement 1). A budget SINK (leaves the domain
                                    # as vapor); NOT on the precip map. Zero unless the subsidence gate fires.
        # --- GEO-22 nested-refinement BC (default None -> the block is skipped, byte-identical) ---
        # Validate the duck-typed NestBC's grids against THIS grid, then WARM-START the prognostic
        # moisture (and the precip EMA) from the coarse solution -- a nest converges to the same steady
        # state from zero, but starting on the parent's fields spends the iterations on the fine-scale
        # departure, not on re-growing the zonal structure. The rim term itself lives in _step_with_wind.
        self._nest = c.nest_bc
        self._nest_g = None          # the parent's inherited aloft-supply profile g(e); see _supply_target
        if self._nest is not None:
            nb = self._nest
            for name in ("W_s", "W_a", "rim_w", "n_x", "n_e"):
                arr = np.asarray(getattr(nb, name), float)
                if arr.shape != (self.nx, self.ne):
                    raise ValueError(f"nest_bc.{name} shape {arr.shape} != the nest grid "
                                     f"{(self.nx, self.ne)} -- BC built for a different topo?")
            self.W_s = np.asarray(nb.W_s, float).copy()
            self.W_a = np.asarray(nb.W_a, float).copy()
            if getattr(nb, "P0", None) is not None:
                self.P = np.asarray(nb.P0, float).copy()
            # the INHERITED aloft-supply longitude profile g(e) (GEO-22). Resolved and validated ONCE
            # here, not per step, so the missing-profile warning fires at construction rather than
            # 10^5 times through the step loop. NB Python's default "once per location" filter then
            # dedupes it across MAPS too, so a multi-tile run warns once for the whole sweep, not once
            # per tile -- the per-tile `[refine] ... aloft supply:` log line is what covers each tile.
            sg = getattr(nb, "supply_g", None)
            if sg is not None:
                sg = np.asarray(sg, float)
                if sg.shape != (self.ne,):
                    raise ValueError(f"nest_bc.supply_g shape {sg.shape} != ({self.ne},) -- the inherited "
                                     f"aloft-supply profile must be one value per NEST along-track column "
                                     f"(build_nest_bc interpolates the parent's onto topo.e)")
                if not np.all(np.isfinite(sg)) or np.any(sg < 0.0):
                    raise ValueError("nest_bc.supply_g must be finite and non-negative (it multiplies the "
                                     "zonal aloft supply F(theta); a negative weight is a moisture sink "
                                     "masquerading as a boundary condition)")
                self._nest_g = sg.copy()
            elif c.resolved_supply:
                # The parent ran the RESOLVED supply but stored no profile (a pre-GEO-22-fix npz). The nest
                # cannot reconstruct it -- its own export is a sub-domain mean with no equatorial source --
                # so it falls back to the zonal F(theta) and says so, loudly, once.
                _warnings.warn(
                    "nest_bc carries no supply_g (the parent npz has no 'supply_g_col'): the nest is "
                    "running on the ZONAL aloft supply F(theta) -- the longitude structure of the off-map "
                    "supply (the evaporation shadow) is LOST. Re-run the parent solve with the GEO-22 fix "
                    "so it stores supply_g_col, or accept a zonally-uniform supply in the refined map.",
                    RuntimeWarning, stacklevel=2)
        self._Phi = self._geopotential()
        self.iter = 0
        self.last_run = None                  # the stop record of the latest run() (stagec_convergence)
        self._cum_E = 0.0; self._cum_P = 0.0                                        # water-budget tallies
        # Track B / B1: EMA of the per-longitude surface-vapor export feeding the resolved aloft supply
        # (the off-map equatorial return). Zero -> the blend falls back to zonal F until export spins up.
        self._export_col = np.zeros(self.ne)                          # [mm/s], per-longitude cross-track MEAN
        # --- S0 supply-split instrumentation (default OFF -> byte-identical) ---
        # Knock out ONE aloft-supply limb to decompose the poleward-band supply into the zonal F(theta)
        # delivery S_band and the longitude-resolved local-detrainment U -- the Approach-B+C S0 pre-check
        # (ROADMAP_BC_moisture_circulation_feedback.md). Each kills only the limb's contribution to the
        # ALOFT layer W_a (the W_s drain sink is untouched, so the W_s budget and P_conv are unchanged);
        # with both False the step is bit-for-bit the production step.
        self._s0_kill_F = False   # zero the F(theta) aloft delivery S_band into W_a
        self._s0_kill_U = False   # zero the local convective detrainment U into W_a
        # --- Track C / C3: prognostic-Phi anomaly zero-point ---
        # P_baseline is the precip whose latent heating is ALREADY baked into T_map_C (so the latent source
        # uses the ANOMALY P - P_baseline, not full P, or the gain is biased high -- roadmap C/C2). It is NOT
        # known at construction (the cascade exposes the AD-core precip, or it is the spun-up static-Phi P);
        # the caller sets `m._P_baseline = <field>` after spin-up. Required when gms_geopot != 0 (the
        # _update_phi_prognostic guard); None leaves the latent source off.
        self._P_baseline = None
        # --- W2: momentum-advection state (the self-limiting wind the moisture rides when momentum_advection) ---
        # _u_mom/_v_mom cache the (warm-startable) momentum SURFACE field; _mom_dt is the CFL-safe step it is
        # advected at. Built lazily on the first step() (None until then) so the default-OFF __init__ is
        # unchanged. The in-loop momentum config resolves phi_smooth_km to the momentum field's OWN adjustment
        # length (momentum_smoothing_km), not the diagnostic 400 km (W2 risk item).
        self._u_mom = None; self._v_mom = None; self._mom_dt = c.dt
        if c.momentum_advection:
            # W3 (DIAGNOSTIC_WIND_ROADMAP): momentum_advection + prognostic_phi is the J1b FUSION -- the
            # momentum wind reads the SAME prognostic screened-Poisson Phi that `_update_phi_prognostic`
            # evolves (thermal + latent), closing moisture -> P -> src_latent -> Phi -> momentum wind ->
            # convergence -> moisture. No longer rejected (see step() / _solve_momentum_wind, which branch on
            # cfg.prognostic_phi at RUNTIME so a driver can spin static then arm the feedback). With
            # prognostic_phi OFF this stays the W2 path: the momentum solve carries its OWN virtual-T Gaussian
            # wind-Phi (momentum_smoothing_km below), decoupled from `_Phi`. In the FUSED path that built Phi
            # is unused -- the screened-Poisson `_Phi` carries its own physically-pinned screening length
            # lam^-1 ~ 424 km (NOT the cap-saturated 400 km; bl_phi_screened), so `momentum_smoothing_km` /
            # `_mom_cfg.phi_smooth_km` go inert without re-imposing the diagnostic length (the W2 risk item).
            base_mom = c.momentum_cfg or blm.BLMomentumConfig()
            psm = c.momentum_phi_smooth_km
            if psm is None:
                psm = blm.momentum_smoothing_km(base_mom)
            self._mom_cfg = replace(base_mom, phi_smooth_km=psm)
        else:
            self._mom_cfg = None

    # ---- defaults (standalone backbone; replaced by Stage A/B in the cascade) ----
    @staticmethod
    def _call(fn, default, th_d):
        return np.asarray((fn or default)(th_d), float)

    def _wetness_col(self, fn, th_d):
        """Evaporation efficiency in [0,1] from Stage A's wetness ramp (ocean=1, salt flat=0). The
        cascade wires A's own _wetness(theta); standalone falls back to A's 55->70 C SST ramp applied
        to THIS stage's SST backbone, so the mask is A-consistent either way. Matches
        stage_a_column._wetness and stage_b_overturning._wetness_of_theta (the same 55/70 C cutoffs).
        Note: the standalone default SST backbone never exceeds 55 C, so the fallback is identically 1
        (the gate is a no-op for the default/test configs and only bites in the cascade, where A's
        hot-zone SST reaches 70-126 C)."""
        if fn is not None:
            return np.clip(np.asarray(fn(th_d), float), 0.0, 1.0)
        SST_WET_HI, SST_DRY = C.SST_WET_HI, C.SST_DRY   # shared ramp endpoints (constants.py); fallback only
        return np.clip((SST_DRY - self.SST_C) / (SST_DRY - SST_WET_HI), 0.0, 1.0)

    @staticmethod
    def _default_sst(th):   # buffered ocean SST [C]: warm equatorward, freezing poleward (placeholder)
        return np.interp(th, [30, 40, 45, 50, 55, 60, 62], [55, 42, 33, 22, 2, -8, -12])

    @staticmethod
    def _default_tland(th):  # land day-mean reference [C] (cooler than SST off the warm edge)
        return np.interp(th, [30, 40, 45, 50, 55, 60, 62], [60, 40, 28, 16, -6, -28, -40])

    @staticmethod
    def _default_supply(th):
        # poleward-aloft moisture supply F [mm]. DEAD on the cascade path -- run_cascade ALWAYS wires the
        # real backbone F via aloft_supply_of_theta, so this placeholder fires only for a bare standalone
        # StageCMap / a diagnostic. NOTE the corrected understanding (PR #7 / OPEN_ISSUES Issue A): the band
        # is CIRCULATION-SET at theta~36 -- the single cell lofts at the equator, returns poleward aloft and
        # dew-points equatorward of the habitable ribbon (theta~45-51), which is dry/supply-limited, NOT the
        # old "wet ribbon" premise. The knot values below are a legacy stand-in whose ribbon peak (~48 mm at
        # theta45) PREDATES that correction; they are retained only so a standalone StageCMap yields a
        # non-degenerate field -- do NOT read them as the model's F (the cascade's actual F is the real one).
        return np.interp(th, [30, 38, 42, 45, 48, 51, 55, 60],
                         [6, 10, 30, 48, 45, 30, 12, 5])

    @staticmethod
    def _default_ubg(th):   # mean equatorward surface wind [m/s] (toward the equator, negative x)
        return np.full_like(th, -4.0)

    # ---- temperature map (day-mean + resolved diurnal cycle, Issue #21) ----
    def _diurnal_range_row(self, fn, th_d, C_surf, wetness, Tref_C):
        """Per-cross-track-row land diurnal RANGE [K] for one continentality endpoint. If a backbone callable
        is supplied (the cascade's surface.solve_periodic_surface result on Stage A's column), use it. Else
        (standalone) fall back to the SAME surface.solve_periodic_surface, on a COARSE cross-track grid
        (cost-bounded, interpolated to th_d), driven by each row's insolation cycle and a downwelling-LW / air-
        temperature proxy synthesized from the backbone day-mean (Stage A is absent standalone). Reuses the
        real diurnal machinery rather than a divergent estimator -- a rough placeholder, like the other
        _default_* backbones; the cascade's is the physical one."""
        if fn is not None:
            return np.maximum(np.asarray(fn(th_d), float), 0.0)
        th_arr = np.asarray(th_d, float)
        nc = int(min(th_arr.size, 18))                                     # coarse grid bounds the solve count
        th_c = np.linspace(float(th_arr.min()), float(th_arr.max()), nc)
        Tref_c = np.interp(th_c, th_arr, np.asarray(Tref_C, float))
        rng = np.empty(nc)
        for i, (thi, trefc) in enumerate(zip(th_c, Tref_c)):
            Tair_K = max(float(trefc), -40.0) + C.T_FREEZE                 # near-surface air ~ the backbone mean
            LW_down = 0.75 * C.SIGMA_SB * Tair_K**4                        # clear-ish-sky downwelling-LW proxy
            x = float(g.x_of_theta(np.radians(thi)))
            t, S_inst = g.diurnal_flux_cycle(x, n_steps=96, n_suns=6)
            SW_abs = 0.9 * (1.0 - C.ALBEDO_LAND) * S_inst                  # land absorbed SW, clear-ish sky
            r = surf.solve_periodic_surface(t, SW_abs, LW_down, Tair_K, 3e-3, C.P_SURF, wind=5.0,
                                            C_surf=C_surf, wetness=wetness, max_days=60, tol=2e-3)
            rng[i] = r["Ts_range"]
        return np.interp(th_arr, th_c, rng)

    def _build_temperature_map(self):
        """Diagnostic (x,e) temperature: maritime sea + continental land WITH the resolved diurnal cycle.
        Sets self.T_min_C / self.T_max_C / self.dT_diurnal_C and returns the day-mean map (the classification
        backbone). Land blends a damp-coast and an arid-interior backbone by continentality
        f = 1 - exp(-d_coast/L_cont): the day-mean AND the diurnal range both interpolate coast<->interior, the
        altitude lapse rides the whole cycle, and the min/max sit asymmetrically about the mean (skew). On top
        of that, MARITIME MODERATION pulls the coastal land day-mean toward the row's SST over maritime_L_km
        and damps the diurnal range toward the sea's over the SEPARATE, shorter maritime_range_L_km (the mean
        is the deep advected air mass; the 2 m swing is a local surface response the sea breeze only partly
        flushes), removing the raw shoreline step. Sea cells hold the buffered SST with a small (~1 K) air
        swing. With the default interior mean = coast - continentality_K
        the day-MEAN is algebraically identical to the prior `mean - continentality_K*(1-exp(...))` model
        (so the standalone map is bit-unchanged); the NEW content is the per-cell min/max."""
        c = self.cfg
        f = 1.0 - np.exp(-self.d_coast / c.L_cont_km)                      # 0 at coast -> 1 deep interior
        # boundary-layer damping surface-range -> 2 m AIR range (used far below for the air cycle, but
        # needed HERE so the maritime RANGE moderation can target the sea's AIR swing; see there).
        g_air = (1.0 - f) * c.bl_damp_g_coast + f * c.bl_damp_g_interior
        lapse = c.lapse_land * self.h
        sst = self.SST_C[:, None]
        half_sea = 0.5 * c.diurnal_range_sea_K                 # sea 2 m AIR half-range
        half_sea_surf = 0.5 * c.diurnal_range_sea_surface_K    # sea SKIN (SST) half-range -- much smaller
        Tmean_land = (1.0 - f) * self.Tland_C[:, None] + f * self.Tland_int_C[:, None]
        dT_land = (1.0 - f) * self.dT_coast[:, None] + f * self.dT_int[:, None]
        # MARITIME MODERATION (see maritime_L_km / maritime_range_L_km in the config): marine air pulls the
        # coastal land day-mean toward the row's SST (nearest-ocean SST approximated by the row backbone --
        # SST(theta) varies slowly on the ~600 km scale), m_mean=1 at the shoreline -> 0 deep inland. Applied
        # at SEA LEVEL, before the lapse, so coastal highlands stay cold. The diurnal RANGE is moderated
        # SEPARATELY, over the shorter maritime_range_L_km: the day-mean is the deep advected marine air mass,
        # but the 2 m diurnal SWING is a LOCAL surface-layer response the land skin regenerates daily and the
        # shallow sea breeze only PARTIALLY flushes -- so it must not collapse to the sea's swing over the full
        # mean-fetch (that read a coast high==low; see maritime_range_L_km).
        # CAVEAT (2nd order at f=0): the pull target is the cell's OWN row SST, but where the nearest wet
        # ocean lies at a meaningfully different latitude (e.g. land poleward of the ice line whose nearest
        # sea is equatorward), the moderating air is drawn from a different water than the SST pulled
        # toward. m ~ exp(-d_coast/maritime_L_km) is already tiny in those geometries, so the error stays small.
        if c.maritime_L_km is not None and float(c.maritime_L_km) > 0.0:
            m_mean = np.exp(-self.d_coast / float(c.maritime_L_km))
            Tmean_land = m_mean * sst + (1.0 - m_mean) * Tmean_land
        # RANGE moderation over the decoupled length. This sits OUTSIDE the day-mean block on purpose: the
        # whole point of maritime_range_L_km is that the two lengths are independent, so switching the
        # day-MEAN moderation off (maritime_L_km = 0/None) must not silently switch the RANGE moderation
        # off too (PR #389 review round 2, finding 4 -- it did).
        #   maritime_range_L_km: None -> reuse maritime_L_km (legacy single-length coupling; if the mean is
        #   also off there is no length to reuse, so the range moderation is off as well -- that IS the
        #   legacy configuration); 0 -> no range moderation (lean on the g_air BL damping alone).
        L_rng = (float(c.maritime_L_km or 0.0) if c.maritime_range_L_km is None
                 else float(c.maritime_range_L_km))
        if L_rng > 0.0:
            m_rng = np.exp(-self.d_coast / L_rng)
            # diurnal_range_sea_K is the sea's 2 m AIR swing, but dT_land here is a SURFACE range that
            # is damped to air by g_air below. Moderating toward sea_K directly would land the fully-
            # flushed first land cell on g_air*sea_K ~ 0.76 K of AIR swing against the sea's 1.0 K --
            # a step at the shoreline, i.e. "the swing land is damped toward" would not be the swing
            # the sea actually has (review round 1, finding 4). Target the SURFACE-equivalent sea_K/g_air
            # so that after the g_air damping the shoreline air range is exactly sea_K: continuous.
            # (With diurnal_range_sea_K=0 the term vanishes either way, so the byte-identical legacy
            # escape is unaffected.)
            #   g_air is a DELIBERATELY-WIDE swept knob, so floor it before dividing: at g -> 0 the
            # surface-equivalent target diverges and would blow up the coastal SKIN range, which feeds
            # the sea-breeze figures and the salt-flat/habitability gates (review round 2, finding 4).
            g_safe = np.maximum(g_air, 0.05)
            dT_land = m_rng * (c.diurnal_range_sea_K / g_safe) + (1.0 - m_rng) * dT_land
        Tmean_land = Tmean_land - lapse
        skew = (1.0 - f) * c.diurnal_skew_coast + f * c.diurnal_skew_interior   # frac of range above the mean
        Tmax_land = Tmean_land + skew * dT_land
        Tmin_land = Tmean_land - (1.0 - skew) * dT_land
        self.T_min_C = np.where(self.sea, sst - half_sea_surf, Tmin_land)   # SKIN: the ~0.1 K SST swing
        self.T_max_C = np.where(self.sea, sst + half_sea_surf, Tmax_land)
        self.dT_diurnal_C = self.T_max_C - self.T_min_C
        # screen-level (2 m) AIR cycle: a boundary-layer damping of the SURFACE range, anchored on the
        # day-mean so its centre cannot drift (Issue #21 follow-up). g<1 blends coast (weak damp) <->
        # interior (strong damp) by the SAME continentality ramp f; the symmetric air_skew keeps the midpoint
        # at the day-mean (T_map_C untouched -> strictly additive).
        #   Night-min lift (the frost asymmetry) is NOT "purely g<1": the surface cycle is hot-SKEWED
        # (diurnal_skew>0.5, noon-heavy) while the air cycle is re-centred SYMMETRIC (air_skew=0.5), so the
        # skew raises the skin min toward the mean and works AGAINST the lift. Algebraically
        # T_air_min >= T_min requires (1-air_skew)*g <= (1-skew_surf), i.e. g <= (1-skew_surf)/(1-air_skew)
        # (coast ceiling ~0.84 vs the 0.76 default -- a thin margin), and T_air_max <= T_max requires
        # air_skew*g <= skew_surf. Since g is a DELIBERATELY-WIDE swept knob, rather than rely on that
        # parameter margin we ENFORCE the physical ordering by construction below: the screen-level air cycle
        # must sit INSIDE the skin cycle (a decoupled night air min cannot fall below the radiating skin min,
        # nor a BL-damped day air max exceed the skin max). For the default/cascade knobs the raw air extremes
        # are already strictly inside, so the clamp is a NO-OP; it only bites if a wide g-sweep (or a larger
        # surface skew) pushes a raw extreme past the skin -- keeping air-frost a subset of ground-frost
        # ROBUSTLY, not by a thin margin. Sea is unchanged (~0 swing).
        dT_air_land = g_air * dT_land                                      # g_air computed at the top
        T_air_max_land = np.minimum(Tmean_land + c.air_skew * dT_air_land, Tmax_land)
        T_air_min_land = np.maximum(Tmean_land - (1.0 - c.air_skew) * dT_air_land, Tmin_land)
        self.T_air_min_C = np.where(self.sea, sst - half_sea, T_air_min_land)
        self.T_air_max_C = np.where(self.sea, sst + half_sea, T_air_max_land)
        self.dT_air_C = self.T_air_max_C - self.T_air_min_C
        return np.where(self.sea, sst, Tmean_land)

    # ---- finite differences (periodic in e, one-sided in x): the shared operators (flatworld.operators),
    #      kept as thin wrappers so the wall/periodic conventions live in ONE place (importable by the
    #      #111 momentum solve + its #112/#114/#115 siblings). ----
    def _dx(self, F):
        return ops.ddx(F, self.dx)

    def _de(self, F):
        return ops.dde(F, self.de)

    def _lap(self, F):
        return ops.lap(F, self.dx, self.de)

    def _div(self, U, V):
        return ops.div(U, V, self.dx, self.de)

    def _adv_flux(self, F, U, V):
        # A NEST's cross-track walls are OPEN (outflow leaves, nothing enters); the full domain keeps
        # the closed zero-flux walls -- see flatworld.operators.adv_flux and climate_nest's LIMITS.
        return ops.adv_flux(F, U, V, self.dx, self.de, open_x=(self._nest is not None))

    def _geopotential(self):
        """Boundary-layer geopotential anomaly Phi [m^2/s^2]: warm surface -> LOW Phi (thermal low),
        high terrain -> HIGH Phi (air rides over). Smoothed over the FRICTIONAL/ADVECTIVE adjustment
        length L_fric = wind_cap/r_fric = 400 km (phi_smooth_km; f=0, so NOT a Rossby deformation radius) --
        the boundary layer integrates over that fetch and responds to the smoothed heating, not grid-scale
        coastline jumps that would spike the ascent w ~ del^2(Phi). The balanced wind flows down grad(Phi):
        into warm/low-Phi land (convergence -> ascent -> rain)."""
        import scipy.ndimage as ndi
        c = self.cfg
        Tref = np.mean(self.T_map_C, axis=1, keepdims=True)        # zonal-mean reference per row
        Phi = -c.beta_T * (self.T_map_C - Tref) + c.g_terrain * C.G_ACCEL * self.h
        sig = max(c.phi_smooth_km / (self.dx / 1e3), 1.0), max(c.phi_smooth_km / (self.de / 1e3), 1.0)
        # boundary modes MUST match the solver's: clamped at the cross-track walls ("nearest") but
        # PERIODIC along-track ("wrap") -- _de/_lap/_adv_flux are all np.roll-periodic in e, so a
        # clamped smooth would leave the land/sea contrast across the e-seam unsmoothed and the
        # periodic derivative would then difference the raw jump: exactly the grid-scale del^2(Phi)
        # ascent spike this smoothing exists to prevent, reborn at the seam (a 2-4x spurious precip
        # stripe on unpadded topologies -- issue #82).
        return ndi.gaussian_filter(Phi, sigma=sig, mode=("nearest", "wrap"))

    def _update_phi_prognostic(self):
        """Track C / C3: recompute the BL geopotential through C1's screened-Poisson inverse (the UNIFIED
        thermal + latent operator, `bl_phi_screened.geopotential_screened`) and under-relax it in:

            Phi <- (1 - phi_relax) * Phi_old + phi_relax * Phi_screened.

        The thermal source matches the static build's `Phi_eq = -beta_T*(T_map_C - Tref) + g_terrain*g*h`
        (same `beta_T`/`g_terrain`) but is inverted through the screened-Poisson `K_0` kernel rather than
        Gaussian-smoothed -- so `gms_geopot=0` is a pure operator swap (the J1 control). With `gms_geopot != 0`
        the latent source `-gms_geopot*(P - P_baseline)` adds on the SAME footing: a precip ANOMALY lowers Phi
        (a thermal low -> more ascent -> the positive moisture<->circulation feedback C3 closes). The ANOMALY
        (not full P) is required because the baseline latent heating is already in `T_map_C` (double-count
        guard). Called from `step()` every `phi_update_every` steps; the diurnal `_step_with_wind` path does
        NOT use it (C3 is the time-mean standing feedback)."""
        c = self.cfg
        from . import bl_phi_screened as bps
        use_latent = (c.gms_geopot != 0.0)
        if use_latent and self._P_baseline is None:
            raise ValueError(
                "prognostic_phi with gms_geopot != 0 requires the anomaly zero-point `_P_baseline` (the "
                "precip whose latent heating is baked into T_map_C). Set `m._P_baseline = <P field>` after "
                "spin-up (the C3 double-count guard); pass a zeros field to drive on full P deliberately.")
        Phi_new = bps.geopotential_screened(
            self.T_map_C, self.h, self.dx, self.de,
            P=(self.P if use_latent else None), P_baseline=(self._P_baseline if use_latent else None),
            gms_geopot=c.gms_geopot, beta_T=c.beta_T, g_terrain=c.g_terrain, r_fric=c.r_fric,
            source_smooth_km=c.phi_source_smooth_km, bc_x="neumann")
        self._Phi = (1.0 - c.phi_relax) * self._Phi + c.phi_relax * Phi_new

    def _balanced_wind(self):
        """Diagnostic damped boundary-layer wind (f=0, Lindzen-Nigam/Gill limit):
        V = u_bg*x_hat - grad(Phi)/r. Topology-responsive (flows down grad(Phi), which carries the
        heating/terrain) and unconditionally stable (no inertial CFL). NOT prognostic momentum: there is
        no du/dt and Phi is static, so no circulation-moisture feedback (see module docstring / DESIGN
        P1). Capped for numerical safety on sharp gradients."""
        c = self.cfg
        ubg2d = np.broadcast_to(self.ubg[:, None], (self.nx, self.ne))
        u = ubg2d - self._dx(self._Phi) / c.r_fric
        v = -self._de(self._Phi) / c.r_fric
        sp = np.sqrt(u**2 + v**2)
        scale = np.minimum(1.0, c.wind_cap / np.maximum(sp, 1e-9))
        return u * scale, v * scale

    # ---- Track B / B1: the poleward-aloft supply target ----
    def _resolved_supply_g(self):
        """The resolved-supply LONGITUDE PROFILE `g(e)`, shape (ne,) -- the single implementation of the
        `w`/`g`/along-track-diffusion algebra that `_supply_target` multiplies onto the zonal `F(theta)`
        and that `supply_g_col` publishes for a nest to inherit (one law, two readers -- they cannot
        drift). Returns None in exactly the cases where the supply falls back to the plain zonal `F`:
        `resolved_supply` off, or no resolved export anywhere to normalize against."""
        c = self.cfg
        if not c.resolved_supply:
            return None
        ref = float(c.supply_ref_export)
        denom = ref if ref > 0.0 else float(np.mean(self._export_col))
        if denom <= 1e-12:                                     # no resolved export -> nothing to redistribute
            return None
        a = float(c.supply_zonal_mixing)
        w = self._export_col / denom                           # per-longitude export weight (mean 1 iff ref==0)
        g = (1.0 - a) * w + a * float(np.mean(w))              # ref==0 -> (1-a)w+a (B1); ref>0 -> tracks the total
        if c.supply_diffusion_km > 0.0:                        # physical along-track eddy diffusion of the off-map
            sig = c.supply_diffusion_km / (self.de / 1e3)      # supply pathway: heat-kernel smear, sigma in cells.
            g = _ndi.gaussian_filter1d(g, sigma=max(sig, 1e-6), mode="wrap")  # periodic e; conserves mean_e(g)
        return g

    def supply_g_col(self):
        """PUBLIC: the longitude profile `g(e)` THIS SOLVE'S aloft-supply target actually uses, shape
        (ne,) -- or None when the target is the plain zonal `F(theta)` and there is no profile at all.
        The contract is that identity, in every configuration:

        * NEST with an inherited profile -> the inherited `NestBC.supply_g`;
        * NEST on the zonal fallback (no profile inherited) -> None;
        * full domain under `resolved_supply` -> the live post-diffusion `g` from the export blend;
        * otherwise (`resolved_supply` off, or no export to normalize against) -> None.

        The nest cases must be answered from `_nest_g`, NOT from the export blend: a nest never
        accumulates `_export_col`, so under a frozen `supply_ref_export > 0` the blend would compute
        `w = 0/E_ref` and hand back a profile of ZEROS -- a silent starvation instruction, the exact
        inverse of the wall-railed pump (MODELING_LESSONS, "a frozen full-domain supply reference is not
        sub-domain-portable"). None is the honest answer there, and it is what tells a caller there is
        nothing to inherit.

        This is the quantity a NESTED refinement must INHERIT from its coarse parent (GEO-22). A nest is
        a sub-domain of the parent's cross-track extent, so it contains none of the equatorial evaporation
        that feeds the off-map return. Callers writing a parent solution to disk should store this on the
        DELIVERED grid -- i.e. cropped the same way the 2-D maps' e-axis is cropped.

        MEANINGFUL AFTER CONVERGENCE: on a full domain the profile is the blend of the live `_export_col`
        EMA, so a mid-run call publishes a not-yet-converged profile. Call it on a converged solve (every
        current caller does -- the gallery writes it post-solve)."""
        if self._nest is not None:
            return None if self._nest_g is None else self._nest_g.copy()
        g = self._resolved_supply_g()
        return None if g is None else np.asarray(g, float).copy()

    def _supply_target(self):
        """The aloft-supply target `F` the relaxation `S_band` relaxes `W_a` toward, as a (nx, ne) field.

        NEST (`nest_bc` set) -- takes precedence over everything below: the target is `F_zonal(x)` times
        the longitude profile `g(e)` INHERITED from the coarse parent (`NestBC.supply_g`), or the plain
        zonal `F` when the parent carried none. The nest NEVER computes `g` from its own export: the
        supply is a boundary-condition device fed by evaporation outside the window, so a sub-domain mean
        normalized against the frozen full-domain reference is not a physical weight (MODELING_LESSONS,
        "a frozen full-domain supply reference is not sub-domain-portable"). The saturation cap below
        still applies, on the NEST's own (colder, better-resolved) `Wsat_aloft_map`.

        Default (`resolved_supply=False`): the prescribed ZONAL `F(theta)` broadcast across longitude
        (byte-identical to the pre-B1 `self.F[:, None]`).

        Resolved (`resolved_supply=True`): `F_resolved(x,e) = F_zonal(x) * g(e)`,
        `g(e) = (1-a)*w(e) + a*mean_e(w)`, `w(e) = export_col(e)/denom`, `a = supply_zonal_mixing`. The
        denominator sets the conservation:

        * `supply_ref_export = 0` (DEFAULT): `denom = mean_e(export_col)` -> `mean_e(w) = 1` so `mean_e(g) = 1`
          EXACTLY: the ZONAL-MEAN target reproduces the validated zonal `F(theta)` for any export pattern
          (B1), and a source block REDISTRIBUTES the finite supply (the blocked longitude drops, the rest
          rise) -- the zonal mean is conserved.
        * `supply_ref_export = E_ref > 0`: `denom = E_ref`, the frozen all-sea reference export that produces
          `F(theta)` (the AD-core zonal reference). Then `mean_e(g) = mean_e(export_col)/E_ref`: an export AT
          the reference reproduces `F(theta)`, and a source block drops the blocked longitude BELOW `F` while
          the rest stay AT `F` (not enriched) -> the TOTAL supply falls (the physical "less sea -> less
          moisture"). `a` damps only the LOCALIZATION (a=1 -> zonally uniform), never the total.

        `a = supply_zonal_mixing` is frozen at 0 by default (the principled f=0 longitude-preserving value;
        see the config). No export anywhere (`denom`/export ~ 0) -> fall back to zonal `F`.

        SATURATION CAP (`wa_supply_sat_cap`, default 1.0 = ON): the returned target is clipped at
        `cap * Wsat_aloft(theta)` -- the relaxation may fill a column TO its local aloft saturation
        capacity, never THROUGH it. The relaxation is a boundary-condition device (moisture arriving
        from outside the window via the unresolved poleward-aloft limb); a target above the local
        capacity turns it into an unphysical moisture pump: on the lens_diurnal world the legacy 8 mm
        F-floor against the collapsed cold-flank Wsat (~0-2 mm past the band's cold edge) force-fed
        4-9 mm/day of spurious rainout across theta 47-60 (sea AND land; the 2026-07-18 wet-belt
        artifact -- MODELING_LESSONS). Where Wsat >= F (the band and equatorward; the whole wet flank
        of the prior L8e7 world) the clip never binds and the validated supply is unchanged. Genuine
        cold-ward rainout is untouched: a near-saturated column ADVECTED into a lower-Wsat row still
        sheds its excess (that physics lives in advWa, not here). `cap = inf` -> the uncapped legacy
        law, byte-identical."""
        Fz = self.F[:, None]
        cap = float(self.cfg.wa_supply_sat_cap)
        # the 2-D capacity map (per-row Wsat x the gamma land modulation; (nx,1) verbatim at gamma=0)
        Fcap = (cap * np.asarray(self.Wsat_aloft_map, float)) if np.isfinite(cap) else None
        if self._nest is not None:                             # NEST: inherit g(e); never self-normalize
            g = self._nest_g                                   # validated (ne,) >= 0 at construction, or None
            F2 = (np.broadcast_to(Fz, (self.nx, self.ne)) if g is None else Fz * g[None, :])
            return F2 if Fcap is None else np.minimum(F2, Fcap)
        g = self._resolved_supply_g()
        if g is None:                                          # zonal F (resolved_supply off, or no export)
            F2 = np.broadcast_to(Fz, (self.nx, self.ne))
            return F2 if Fcap is None else np.minimum(F2, Fcap)
        F2 = Fz * np.maximum(g, 0.0)[None, :]
        return F2 if Fcap is None else np.minimum(F2, Fcap)

    # ---- W2: the self-limiting momentum wind the moisture rides under momentum_advection ----
    def _solve_momentum_wind(self):
        """Re-solve the nonlinear `bl_momentum` SURFACE field on the current state and cache it.

        W2 (prognostic_phi OFF): build the momentum wind-Phi from the (static) T_map and the CURRENT W_s
        (virtual temperature, the weak moisture->wind channel), Gaussian-smoothed over the momentum field's
        own adjustment length (`_mom_cfg.phi_smooth_km`).

        W3 fusion (prognostic_phi ON): read the PROGNOSTIC screened-Poisson Phi `self._Phi` -- the SAME field
        `_update_phi_prognostic` evolves (thermal + LATENT, under-relaxed) -- so the momentum wind reorganizes
        around the resolved latent heating (the J1b coupling). This replaces W2's weak virtual-T channel with
        the dominant latent channel; the screened-Poisson `_Phi` carries its own physically-pinned screening
        length (lam^-1 ~ 424 km), so the momentum wind-Phi Gaussian build (and `_mom_cfg.phi_smooth_km`) is
        bypassed -- NOT re-imposing the diagnostic 400 km (the W2 risk item). NB the latent-heating channel
        SUPERSEDES the W2 virtual-T (vapor-buoyancy) channel here; folding both onto one footing is a noted
        refinement, not the J1b question.

        Inverts the steady self-limiting balance warm-started from the previous field, and stores `_u_mom/
        _v_mom` + the CFL-safe `_mom_dt`. The aloft return is NOT solved here -- `_step_with_wind` rebuilds it
        as the kinematic departure off this surface wind (W1 scope (a)), so only the surface field is cached."""
        from . import bl_forcing as blf
        mc = self._mom_cfg
        if self.cfg.prognostic_phi:                                 # W3 fusion: the momentum wind reads _Phi
            Phi = self._Phi
        else:                                                       # W2: the momentum solve's own virtual-T Phi
            W_s = self.W_s if mc.virtual_temp else None
            Tv = blf.virtual_T_map(self.T_map_C, W_s)
            Phi = blf.geopotential(Tv, self.h, self.dx, self.de, beta_T=mc.phi_beta_T,
                                   g_terrain=mc.phi_g_terrain, phi_smooth_km=mc.phi_smooth_km)
        sol = blm.solve_surface_wind(Phi, self.ubg, self.dx, self.de, mc,
                                     T_map_C=self.T_map_C, height=self.h,
                                     u0=self._u_mom, v0=self._v_mom, verbose=False)
        self._u_mom, self._v_mom = sol["u"], sol["v"]
        self._mom_dt = self._momentum_cfl_dt(self._u_mom, self._v_mom)
        return self._u_mom, self._v_mom

    def _momentum_cfl_dt(self, u, v):
        """CFL-safe moisture timestep for an externally-supplied advecting wind, bounding BOTH the surface AND
        the poleward-aloft layer (the aloft departure u-2*ubg can exceed the surface speed -- W1 measured aloft
        22.5 vs surface 19.4 m/s on nsub30). The momentum wind has NO 40 m/s cap, so this is RE-DERIVED against
        the field's ACTUAL peak (W2 risk item), not wind_cap.

        DIRECTIONAL (PR #245 review note 1): the cross-track component advects over dx, the along-track over de,
        so the limit is `target*min(dx/umax, de/vmax)` with umax/vmax the per-direction peaks over both layers --
        parity with `bl_momentum._cfl_dt` and `stage_c_diurnal`. (The earlier magnitude-on-de form `max|V|*dt/de`
        was only safe under dx>=de -- true on every grid in play, canonical dx~348/de~30 km, but an unstated
        assumption; the directional form removes it.) Returns the largest dt <= cfg.dt that satisfies the limit;
        equals cfg.dt when the wind is gentle enough (the canonical-world case -> no shrink)."""
        c = self.cfg
        u_al, v_al = blm.aloft_departure_wind(u, v, self.ubg, c.u_aloft_frac, c.aloft_departure_sign)
        umax = max(float(np.max(np.abs(u))), float(np.max(np.abs(u_al))), 1e-9)   # cross-track (advects over dx)
        vmax = max(float(np.max(np.abs(v))), float(np.max(np.abs(v_al))), 1e-9)   # along-track  (advects over de)
        dt_cfl = c.momentum_courant_target * min(self.dx / umax, self.de / vmax)
        if dt_cfl >= c.dt:
            return c.dt
        n_sub = int(np.ceil(c.dt / dt_cfl))
        return c.dt / n_sub

    # ---- one step ----
    def step(self):
        """Advance one step on the diagnostic day-mean balanced wind (the deliverable path). When
        `prognostic_phi` is on (Track C / C3), the geopotential is re-inverted (screened-Poisson, under-
        relaxed) every `phi_update_every` steps BEFORE the wind is diagnosed -- closing the moisture<->
        circulation loop. Default OFF -> the static Phi is untouched and this is byte-identical.

        When `momentum_advection` is on (W2), the moisture instead rides the SELF-LIMITING `bl_momentum`
        surface field, re-solved (warm-started) every `momentum_update_every` steps and advected at the
        CFL-safe `_mom_dt`; `_step_with_wind` follows with the aloft departure off that surface wind.

        When BOTH `momentum_advection` and `prognostic_phi` are on (W3, the J1b fusion), the screened-Poisson
        Phi is re-inverted (thermal + latent, under-relaxed) every `phi_update_every` steps BEFORE the
        momentum wind is re-solved, so the momentum wind reads the SAME prognostic Phi -- closing
        moisture -> P -> src_latent -> Phi -> momentum wind -> convergence -> moisture."""
        c = self.cfg
        if c.momentum_advection:
            # W3 fusion: evolve the prognostic screened-Poisson Phi first (thermal + latent), then the
            # momentum re-solve below reads it (`_solve_momentum_wind` branches on cfg.prognostic_phi). With
            # prognostic_phi OFF this is a no-op -> the W2 path (momentum wind on its own virtual-T Gaussian Phi).
            if c.prognostic_phi and (self.iter % c.phi_update_every == 0):
                self._update_phi_prognostic()
            if self._u_mom is None or (self.iter % c.momentum_update_every == 0):
                self._solve_momentum_wind()
            return self._step_with_wind(self._u_mom, self._v_mom, dt=self._mom_dt)
        if c.prognostic_phi and (self.iter % c.phi_update_every == 0):
            self._update_phi_prognostic()
        return self._step_with_wind(*self._balanced_wind())

    def _step_with_wind(self, u, v, dt=None):
        """Advance the two-layer moisture one step on an EXTERNALLY supplied wind (u, v), at timestep
        ``dt`` (default ``self.cfg.dt``). ``step()`` calls this with the static balanced wind and the
        config dt -> byte-for-byte identical to the pre-refactor code. The diurnal-precip post-process
        (#131, ``stage_c_diurnal``) calls it with the time-varying sea/land-breeze wind and a sub-cycled
        CFL ``dt`` so the afternoon convergence is resolved rather than day-mean-averaged. The ``dt``
        threading is load-bearing: the capacity-ceiling rainout below divides by dt, so a verbatim fixed-dt
        body could not sub-cycle (a strong-breeze CFL would be unstable). The EMA (``self.P``) and the
        water-budget tallies are applied internally as before; the INSTANTANEOUS pre-EMA precip and the
        convergence are RETURNED so the diurnal solver can bin the resolved per-phase rate (it ignores the
        EMA-smoothed ``self.P``). Returns ``(P_now [mm/day], conv = -div(u,v) [1/s])``."""
        c = self.cfg
        dt = c.dt if dt is None else dt
        self.u, self.v = u, v
        # --- vertical pumping from convergence: ascent where the flow converges (div V < 0) ---
        div = self._div(self.u, self.v)
        w = -c.H_bl * div                                          # >0 = ascent
        ascent = np.maximum(w, 0.0)
        # --- aloft wind: poleward mean (opposite the surface) + sign*(dynamical departure) (issue #88 item 1).
        #     aloft_departure_sign = +1 (default): surface-signed departure (node converges aloft too, no vent);
        #     -1: the thermally-direct RETURN branch (node vents aloft). Sign applies to the FULL departure
        #     vector (u AND v; the background wind is purely cross-track so v_bg=0 -> the v departure IS self.v). ---
        # the poleward-aloft return as the kinematic departure off the surface wind (bl_momentum is the
        # single source of truth for this algebra -- W1; byte-identical to the prior inline expression,
        # locked by test_aloft_departure_wind_matches_stage_c_inline).
        u_aloft, v_aloft = blm.aloft_departure_wind(self.u, self.v, self.ubg,
                                                    c.u_aloft_frac, c.aloft_departure_sign)
        # === conservative two-layer moisture ===
        # sources into the surface layer W_s: sea evaporation + land evapotranspiration (recycled rain)
        Wsat_sfc = _wsat_column(self.T_map_C + C.T_FREEZE)
        Wa_sat = self.Wsat_aloft_map                                            # cold aloft saturation
                                                                                # (2-D gamma map; (nx,1) at gamma=0)
        # sea evaporation, GATED by A's wetness ramp: no open-ocean evaporation over the salt-flat hot
        # zone (wetness->0 where SST>70 C), the A->C consistency fix that removes the hot-zone blow-up.
        E_sea = self.wet_col * np.where(self.sea, np.maximum(Wsat_sfc - self.W_s, 0.0) / c.tau_evap, 0.0)  # mm/s
        E_land = np.where(self.land, c.recycle * self.P / 86400.0, 0.0)         # mm/s (ET = recycle*P)
        advWs = self._adv_flux(self.W_s, self.u, self.v)
        self.W_s = np.maximum(self.W_s + dt * (advWs + E_sea + E_land + c.kappa_q * self._lap(self.W_s)), 0.0)
        # Convection drains the boundary layer wherever it is warm-humid (the inner/outer sea) or in dynamical
        # ascent, at rate 1/tau_conv. The drain is the W_s SINK and is left UNGATED so it always bounds W_s
        # (gating it to 0 under subsidence removes the only sink at the diagnostic wind's convergence nodes and
        # runs W_s away -- a verified runaway, OPEN_ISSUES). What Refinement 1 changes is the FATE of the
        # drained moisture, partitioned by the large-scale vertical motion (subs_gate, Stage B's w sign):
        #   * ASCENT (subs_gate->1): the drainage splits by f_loft = (1-PE) (PER-ROW; the cascade's CELL-
        #     ASCENT-GATED plume efficiency, Brief 2) -- a fraction (1-PE) DETRAINS aloft (U, rides poleward),
        #     the PE fraction PRECIPITATES to the surface (P_conv). So P_conv is SURFACE-REACHING rain
        #     (precipitation efficiency applied), NOT total condensate.
        #   * SUBSIDENCE (subs_gate->0): deep convection is capped from above, so there is NO local rain and
        #     NO aloft detrainment. The moisture is surface-layer vapor swept into the equatorward return flow
        #     and EXPORTED to the equatorial pathway (vapor_export): it rides into the salt flats where it
        #     CANNOT rain (no surface source -> RH falls), lofts only at the off-map equatorial updraft, and
        #     returns poleward aloft as F(theta). It is NOT precipitated here (PE does not apply under
        #     subsidence -- removing the spurious local convective rain over the warm subsiding ribbon sea, the
        #     PR-2 finding) and NOT re-injected in-domain (the falsified Refinement-1b premise: re-feeding it
        #     would double-count F(theta)'s already-airborne supply). It is removed in-place at the drain rate
        #     as a bounded PROXY for the equatorward export -- the literal "hold as W_s and advect out the
        #     theta~28 edge" RUNS AWAY at the diagnostic wind's convergence nodes (W_s ~3e5 mm; verified), the
        #     diagnostic-wind limitation surfacing again. subs_gate==1 standalone (gate off) -> all drainage is
        #     loft+rain, vapor_export==0 -> bit-identical to the pre-refinement code.
        # The convective on/off gate: convection fires where the surface is warm (T>conv_sst_C) OR the flow
        # ascends (ascent>ASC_THR). Hard boolean by default; a tanh RAMP when conv_gate_ramp_K > 0 (the C3
        # gate-chatter diagnostic -- a hard switch inside the prognostic-Phi feedback loop can limit-cycle at
        # a threshold and look like a structural runaway). The ramp is a smooth OR (max) of the SST-excess and
        # ascent-excess sigmoids; ramp -> 0 recovers the boolean exactly (the byte-identical limit).
        ASC_THR = 1e-4    # ascent [m/s] above which convection is allowed (shared by both gate branches)
        if c.conv_gate_ramp_K > 0.0:
            # one dimensionless ramp sharpness r applied to BOTH thresholds (which live in different units),
            # so ramp -> 0 recovers the boolean for both gates: the SST gate ramps over `conv_gate_ramp_K`
            # [K] (= r*conv_sst_C) and the ascent gate over the same FRACTION r of its ASC_THR threshold.
            r = c.conv_gate_ramp_K / c.conv_sst_C
            g_sst = 0.5 * (1.0 + np.tanh((self.T_map_C - c.conv_sst_C) / (r * c.conv_sst_C)))
            g_asc = 0.5 * (1.0 + np.tanh((ascent - ASC_THR) / (r * ASC_THR)))
            conv_w = np.maximum(g_sst, g_asc)                                 # smooth OR -> [0,1]
            drain = conv_w * self.W_s / c.tau_conv                            # smoothly-gated W_s sink
        else:
            conv_active = (self.T_map_C > c.conv_sst_C) | (ascent > ASC_THR)
            drain = np.where(conv_active, self.W_s / c.tau_conv, 0.0)         # UNGATED W_s sink (bounds W_s)
        if self.salt_warm_col is None:
            U = self.subs_gate_col * self.f_loft_col * drain                 # detrain ALOFT (ascent only; 1-PE)
            P_conv = self.subs_gate_col * (1.0 - self.f_loft_col) * drain    # SURFACE rain (ascent only; PE)
        else:
            # salt CCN (#237): scale the convective PRECIPITATION EFFICIENCY PE=(1-f_loft) by (N_d/N_ref)^beta;
            # the rain that no longer falls is detrained ALOFT, so the convective<->stratiform SPLIT shifts
            # (suppression -> more lofted, less in-place convective rain). Written as the ORIGINAL split plus the
            # salt delta (P_conv0 - P_conv), so at factor 1 (N_d==N_ref) P_conv reproduces the original arithmetic
            # EXACTLY (pe == 1-f_loft elementwise) and U == U0 BIT-for-bit -- the clean zero-point then holds for a
            # NON-constant f_loft(theta) too (the production wiring), not just the scalar default (a plain
            # U=(1-pe)*drain would differ at ~1e-14 because 1-(1-f_loft) != f_loft in IEEE float). PE clipped to
            # [0,1]; the drain mass U+P_conv == U0+P_conv0 == subs_gate*drain is conserved by construction.
            U0 = self.subs_gate_col * self.f_loft_col * drain
            P_conv0 = self.subs_gate_col * (1.0 - self.f_loft_col) * drain
            pe = np.clip((1.0 - self.f_loft_col) * self.salt_warm_col, 0.0, 1.0)
            P_conv = self.subs_gate_col * pe * drain
            U = U0 + (P_conv0 - P_conv)
        vapor_export = (1.0 - self.subs_gate_col) * drain                   # surface-vapor export (subsidence)
        vapor_export_rate = float(np.sum(vapor_export))                      # mm/s summed (>=0)
        # Track B / B1: EMA the per-longitude export (cross-track MEAN, not sum) -> the resolved aloft
        # supply's blend (the off-map equatorial return is fed by this upstream evaporation). A MEAN (per
        # cross-track area) not a Σ_x so a FROZEN `supply_ref_export` reference is nx-PORTABLE (a sum would
        # scale with nx and silently de-calibrate at a different grid -- the L/nx trap, CLAUDE.md). The
        # `export/denom` ratio the blend uses is MATHEMATICALLY unchanged by mean-vs-sum (the 1/nx cancels) --
        # so the supply target / b1/b2 results are identical to within float rounding (~1e-15), not a behaviour
        # change. (The default-OFF path is untouched: the EMA is not updated unless resolved_supply.)
        # NOT accumulated under a NEST: a cross-track mean over a sub-domain that holds no equatorial
        # evaporation is not the quantity the supply blend needs, and the nest inherits its g(e) from the
        # coarse parent instead (_supply_target). Leaving it at zero also keeps the dead path unreadable.
        if c.resolved_supply and self._nest is None:
            self._export_col = ((1.0 - c.supply_export_ema) * self._export_col
                                + c.supply_export_ema * np.mean(vapor_export, axis=0))
        self.W_s = np.maximum(self.W_s - dt * (U + P_conv + vapor_export), 0.0)   # = -dt*drain (full sink)
        # --- aloft layer W_a: the POLEWARD-ALOFT moisture supply delivered from the inner sea (Stage B's
        #     derived band F(theta), CIRCULATION-SET at theta~36 -- the single cell lofts at the equator and
        #     the air returns poleward aloft to dew-point there, equatorward of the habitable ribbon; PR #7)
        #     -- delivered to the aloft layer and rained at/poleward of delivery (the two-layer fix: NOT
        #     advected equatorward at the surface as the retired single-layer model did). The local
        #     cell-ascent-gated convective detrainment U adds on top (secondary; see above). The
        #     dynamical wind + topology (orography, continentality, upwind-land shadow) then modulate
        #     WHERE within each latitude the delivery rains -- the actual 2-D map. ---
        S_band = np.maximum(self._supply_target() - self.W_a, 0.0) / c.tau_supply   # mm/s aloft delivery
        advWa = self._adv_flux(self.W_a, u_aloft, v_aloft)
        # S0 limb knockout (default OFF -> the two np.where are no-ops, bit-identical). Removes a limb's
        # source into W_a ONLY; the W_s drain sink (U + P_conv + vapor_export, above) is already applied.
        U_aloft = np.zeros_like(U) if self._s0_kill_U else U
        if self._s0_kill_F:
            S_band = np.zeros_like(S_band)
        src_a = advWa + U_aloft + S_band + c.kappa_q * self._lap(self.W_a)   # aloft source tendency [mm/s], from W0
        upslope = np.maximum(self.u * self._dx(self.h) + self.v * self._de(self.h), 0.0)
        W_cap = c.wa_capacity_factor * Wa_sat                                # capacity ceiling [mm]
        # --- aloft rainout: rate-limited drizzle + orographic, then a CAPACITY-LIMITED ceiling ---
        if c.wa_rh_crit >= 1.0:
            # LEGACY (default) byte-identical path: add the sources, then drizzle + orographic + a within-step
            # /dt HARD CLIP of the surplus above wa_capacity_factor*Wa_sat. The drizzle/orographic terms are
            # rate-limited; the /dt ceiling pulls any convergence/detrainment node back to the ceiling WITHIN
            # the step so the column cannot park super-saturated (OPEN_ISSUES c2 / Brief 1). At wa_rh_crit=1.0
            # `_wa_supersat` reduces EXACTLY to max(W_a-Wa_sat,0), so this is the prior closure verbatim.
            self.W_a = np.maximum(self.W_a + dt * src_a, 0.0)
            P_strat = _wa_supersat(self.W_a, Wa_sat, c.wa_rh_crit) / c.tau_rain   # mm/s (drizzle)
            P_oro = c.oro_k * upslope * np.maximum(self.W_a - 0.5 * Wa_sat, 0.0)  # mm/s
            P_rate = P_strat + P_oro
            if self.salt_warm_col is not None:                                   # #237: salt CCN scale the warm-
                P_rate = P_rate * self.salt_warm_col                             # rain rainout DEMAND, BEFORE the
                                                                                 # capacity ceiling (so the ceiling-
                                                                                 # masking stays diagnosable)
            P_excess = _wa_supersat(self.W_a - dt * P_rate, W_cap, c.wa_rh_crit) / dt   # surplus above capacity
            P_strat = np.minimum(P_rate + P_excess, self.W_a / dt)           # total aloft rain (mass-limited)
            self.W_a = np.maximum(self.W_a - dt * P_strat, 0.0)
        else:
            # dt-INDEPENDENT critical-RH path (PR #231 review). The legacy split -- add sources, THEN let the
            # rainout read the post-source column -- leaks dt into the converged RH level (fixed point
            # S = Sink(W + dt*S); there is no dt in reality). Here ALL tendencies are evaluated from the
            # START-OF-STEP column W0 and applied together, so dW_a/dt = src - qc(W0)/tau_cond - oro is a genuine
            # O(dt) tendency whose fixed point carries no dt -- the SAME timestep-convergent form the AD-core
            # microphysics uses. The PDF stratiform rainout (rate 1/tau_cond, physically anchored) onsets at
            # RH>rh_crit with PARTIAL rainout, so W0 equilibrates at a supply-dependent RH in (rh_crit,1) -> the
            # 2-D longitude structure of RH_aloft is preserved (the latitude-locked cloud-boundary fix,
            # OPEN_ISSUES 2026-06-27). A hard CLIP to the ceiling then bounds RH<=cap at the residual strong
            # convergence nodes (instantaneous -> dt-invariant; a column cannot sustain super-saturation).
            W0 = self.W_a
            P_pdf = _wa_supersat(W0, Wa_sat, c.wa_rh_crit) / c.tau_cond          # mm/s (PDF stratiform, from W0)
            P_oro = c.oro_k * upslope * np.maximum(W0 - 0.5 * Wa_sat, 0.0)       # mm/s
            demand = P_pdf + P_oro                                              # warm-rain rate demand
            if self.salt_warm_col is not None:                                  # #237: salt CCN scale the warm-
                demand = demand * self.salt_warm_col                           # rain DEMAND BEFORE the mass-limit /
                                                                               # capacity clip (ceiling-masking stays
                                                                               # diagnosable; both signs bounded)
            P_rate = np.minimum(demand, np.maximum(W0 / dt + src_a, 0.0))       # rate sinks (mass-limited)
            W1 = np.maximum(W0 + dt * (src_a - P_rate), 0.0)                     # single explicit-Euler update
            clip = np.maximum(W1 - W_cap, 0.0)                                  # hard cap RH<=cap (dt-invariant)
            self.W_a = W1 - clip
            P_strat = P_rate + clip / dt                                        # total aloft precip rate [mm/s]
        # --- GEO-22 nested rim relaxation (default OFF -> byte-identical: self._nest is None) ---
        # Relax the boundary rim toward the coarse parent solution on INFLOW only: the moisture
        # advection is hyperbolic, so information enters the nest only along inbound characteristics --
        # W_s gated on the SURFACE wind, W_a on its own ALOFT wind, each against the outward normal of
        # the nearest edge. Outflow edges stay free (clamping them would reflect outbound information
        # back into the domain). The relaxed water is deliberately NOT added to the E/P tallies: a nest
        # is an OPEN domain whose budget closes only jointly with the coarse parent, so water_closure_*
        # is not a meaningful diagnostic here (climate_nest module docstring).
        if self._nest is not None:
            nb = self._nest
            a_rim = -np.expm1(-dt / float(nb.tau_s))
            gate_s = (self.u * nb.n_x + self.v * nb.n_e) < 0.0
            gate_a = (u_aloft * nb.n_x + v_aloft * nb.n_e) < 0.0
            self.W_s += (a_rim * nb.rim_w * gate_s) * (nb.W_s - self.W_s)
            self.W_a += (a_rim * nb.rim_w * gate_a) * (nb.W_a - self.W_a)
        S_band_src = float(np.sum(S_band) * dt)                            # count delivery as a source
        # total LOCAL precip rate [mm/day], EMA-smoothed for a steady diagnostic (the deliverable map -- rain
        # only; the surface-vapor export below leaves laterally and is NOT on the map)
        P_now = (P_conv + P_strat) * 86400.0
        self.P = 0.99 * self.P + 0.01 * P_now
        self.vapor_export = 0.99 * self.vapor_export + 0.01 * (vapor_export_rate * 86400.0)  # mm/day (Refinement 1)
        # water-budget tallies. cum_E: local evaporation + the aloft band delivery F(theta). cum_P: ALL moisture
        # SINKS -- the local rain (P_conv + P_strat) PLUS the surface-vapor export. The export is a genuine SINK
        # (the moisture is removed from W_s by the drain and leaves the domain as vapor, returning aloft as
        # F(theta)) -- NOT re-fed in-domain, so it does NOT double-count F; the budget closes by conservation
        # (it is the same drain sink as the pre-refinement code, only re-LABELLED from off-map rain to vapor
        # export under subsidence). Standalone (gate off) vapor_export==0, so cum_P is bit-identical.
        self._cum_E += float(np.sum(E_sea + E_land) * dt) + S_band_src
        self._cum_P += float(np.sum(P_conv + P_strat) * dt) + vapor_export_rate * dt
        # update geopotential is static (T map fixed); recompute only if T evolves (it doesn't here)
        self.iter += 1
        # pre-EMA instantaneous precip [mm/day] + convergence [1/s] for the diurnal-precip post-process (#131)
        return P_now, -div

    # ---- diagnostics ----
    def diagnostics(self):
        # water-budget closure over the run-to-date (E and P area-integrated)
        closure = self._cum_P / max(self._cum_E, 1e-12)
        # instantaneous area-integrated E,P for the steady closure
        c = self.cfg
        Wsat_sfc = _wsat_column(self.T_map_C + C.T_FREEZE)
        E_sea = self.wet_col * np.where(self.sea, np.maximum(Wsat_sfc - self.W_s, 0.0) / c.tau_evap, 0.0) * 86400.0
        E_land = np.where(self.land, c.recycle * self.P / 86400.0, 0.0) * 86400.0
        # The aloft-band relaxation S_band = max(F - W_a, 0)/tau_supply is the THIRD moisture SOURCE (the
        # poleward-aloft delivery F(theta) from the inner sea); step() already counts it in the cumulative
        # `_cum_E`. The instantaneous closure must count it too, else it sits structurally above 1 at steady
        # state (S_band rains out via P_strat but was missing from the inst denominator -- measured 1.59 -> 0.999
        # once added). This, NOT the EMA lag, is why the old inst reading sat ~1.6-1.7 (issue #88 item 2).
        S_band = np.maximum(self._supply_target() - self.W_a, 0.0) / c.tau_supply * 86400.0   # mm/day aloft delivery
        # instantaneous P includes the surface-vapor export (a lateral sink leaving the warm edge) so the inst
        # closure reflects the FULL sink (local rain + exported vapor), not just what lands on the map.
        E_tot = float(np.sum(E_sea + E_land + S_band))
        P_tot = float(np.sum(self.P)) + float(self.vapor_export)
        # diurnal-cycle diagnostics (Issue #21 + follow-up): the surface AND screen-level air min/max maps and
        # frost statistics. frost_frac/frost_prone_frac now key on the AIR minimum (the Koppen-relevant quantity:
        # air frost is rarer than ground frost); the SURFACE-frost numbers are kept alongside under surf_* names
        # (the right quantity for the surface/land-process story). "frost-prone" is the diurnal cycle's NEW
        # information -- land livable in the day-MEAN (T_map_C>=0) but whose NIGHT dips below 0 C.
        land = self.land; nl = max(int(land.sum()), 1)
        air_frost = land & (self.T_air_min_C < 0.0)
        air_frost_prone = air_frost & (self.T_map_C >= 0.0)
        surf_frost = land & (self.T_min_C < 0.0)
        surf_frost_prone = surf_frost & (self.T_map_C >= 0.0)
        return dict(theta_deg=self.theta_deg, T_map_C=self.T_map_C, P_mm_day=self.P.copy(),
                    T_min_C=self.T_min_C.copy(), T_max_C=self.T_max_C.copy(),
                    dT_diurnal_C=self.dT_diurnal_C.copy(),
                    T_air_min_C=self.T_air_min_C.copy(), T_air_max_C=self.T_air_max_C.copy(),
                    dT_air_C=self.dT_air_C.copy(),
                    vapor_export_mm_day=float(self.vapor_export),
                    u=self.u.copy(), v=self.v.copy(), W_a=self.W_a.copy(), W_s=self.W_s.copy(),
                    # the wind panel's honesty overlays: the cap (so cap-saturated cells can be flagged
                    # in renders of old caches without guessing the run's config) and the background
                    # wind row profile (so Phi-driven structure can be separated from ubg(theta) bands).
                    wind_cap=float(c.wind_cap), ubg=self.ubg.copy(),
                    land=self.land, height=self.h, water_closure_cum=closure,
                    water_closure_inst=P_tot / max(E_tot, 1e-12),
                    P_land_mean=float(self.P[self.land].mean()) if self.land.any() else 0.0,
                    P_sea_mean=float(self.P[self.sea].mean()) if self.sea.any() else 0.0,
                    dT_land_mean=float(self.dT_diurnal_C[land].mean()) if land.any() else 0.0,
                    dT_sea_mean=float(self.dT_diurnal_C[self.sea].mean()) if self.sea.any() else 0.0,
                    dT_air_land_mean=float(self.dT_air_C[land].mean()) if land.any() else 0.0,
                    frost_frac=float(air_frost.sum()) / nl, frost_prone_frac=float(air_frost_prone.sum()) / nl,
                    surf_frost_frac=float(surf_frost.sum()) / nl,
                    surf_frost_prone_frac=float(surf_frost_prone.sum()) / nl)

    def state(self):
        # u, v are NOT persisted: they are a diagnostic balance recomputed from the static Phi at the top
        # of every step(), so they are fully determined by __init__ and would be overwritten on resume.
        # The prognostic moisture fields + iter are the real state; the water-budget accumulators
        # (_cum_E/_cum_P) and BOTH export EMAs (the scalar `vapor_export` AND the per-longitude `_export_col`
        # that feeds the resolved-supply blend) are persisted too so a RESUMED production run reports closure
        # over the WHOLE run (not just the post-resume window) and neither export EMA re-spins from 0 over
        # ~100 steps (issue #88 item 4; the `_export_col` array is the resolved-supply extension of that fix --
        # without it a resumed resolved_supply run re-spins the supply target, and under total-reduction it
        # restarts at ZERO supply).
        st = dict(W_s=self.W_s, W_a=self.W_a, P=self.P, iter=self.iter,
                  _cum_E=self._cum_E, _cum_P=self._cum_P, vapor_export=self.vapor_export,
                  _export_col=self._export_col)
        # Track C / C3: under prognostic_phi the geopotential is EVOLVING state (no longer recomputable from
        # __init__), so persist it + the anomaly zero-point; else the resume re-derives the static Phi (the
        # OFF checkpoint format is unchanged -- no _Phi key, byte-compatible with the shipped format). The
        # _Phi/_export_col resume fix's Track-C sibling.
        if self.cfg.prognostic_phi:
            st["_Phi"] = self._Phi
            if self._P_baseline is not None:
                st["_P_baseline"] = self._P_baseline
        return st

    def load_state(self, st):
        for k in ("W_s", "W_a", "P"):
            if k in st:                                  # tolerant of older checkpoints (which also held u, v,
                setattr(self, k, np.asarray(st[k]))      # and/or a now-removed P_export field -- ignored)
        self.iter = int(st.get("iter", 0))
        # accumulators: default to the freshly-initialized value (0.0) when absent, so resuming an OLDER
        # checkpoint simply reports closure for the post-resume window as before -- no crash, no behavior change.
        self._cum_E = float(st.get("_cum_E", self._cum_E))
        self._cum_P = float(st.get("_cum_P", self._cum_P))
        self.vapor_export = float(st.get("vapor_export", self.vapor_export))
        if "_export_col" in st:                          # resolved-supply export EMA; absent in older ckpts
            self._export_col = np.asarray(st["_export_col"], float)
        if "_Phi" in st:                                 # prognostic-Phi evolving state (C3); absent when OFF
            self._Phi = np.asarray(st["_Phi"], float)
        if "_P_baseline" in st:
            self._P_baseline = np.asarray(st["_P_baseline"], float)

    def run(self, verbose=True, on_check=None):
        """Relax to the steady state under ``cfg.conv_rule``; return the monitor summary merged with the
        product stop record (``stagec_convergence.stop_record``), which is also kept as ``self.last_run``.
        ``on_check(self, info)``, when given, is called after every convergence check (tracing only)."""
        c = self.cfg
        if c.checkpoint_path:
            st, _ = conv.load_checkpoint(c.checkpoint_path)
            if st is not None:
                self.load_state(st)
                if verbose: print(f"[StageC] resumed at iter {self.iter}")
        rule = c.conv_rule
        model_time = rule == "model_time"
        n_check = scv.check_steps(rule, c.dt, c.check_every, c.conv_check_s)
        tol = float(c.conv_tol_per_s) if model_time else c.tol
        max_iter = c.max_iter
        if model_time and c.conv_max_model_s is not None:
            cap = int(np.ceil(float(c.conv_max_model_s) / float(c.dt)))
            max_iter = cap if max_iter is None else min(int(max_iter), cap)
        mon = conv.ConvergenceMonitor(tol=tol, patience=c.patience, max_iter=max_iter,
                                      max_seconds=c.max_seconds,
                                      name=("max|dWa|/<Wa>/model-second" if model_time
                                            else "max|dWa|/<Wa>/step"), absolute=True)
        Wa_prev = self.W_a.copy()
        drift = float("nan")
        while True:
            for _ in range(n_check):
                self.step()
            if model_time:
                drift = scv.drift_metric(np.max(np.abs(self.W_a - Wa_prev)), np.mean(np.abs(self.W_a)),
                                         n_check, c.dt, rule)
            else:                                      # the historical per-step expression, verbatim
                denom = max(np.mean(np.abs(self.W_a)), 1e-9)
                drift = float(np.max(np.abs(self.W_a - Wa_prev))) / denom / c.check_every
            Wa_prev = self.W_a.copy()
            stop = mon.update(self.iter, drift)
            if on_check is not None:
                on_check(self, dict(iter=int(self.iter), drift=drift, below=int(mon.below),
                                    stop=bool(stop), elapsed_s=float(mon.elapsed)))
            if verbose and (self.iter % (n_check * 10) == 0 or stop):
                d = self.diagnostics()
                print(f"[StageC] it={self.iter:>8} P_land={d['P_land_mean']:.1f} P_sea={d['P_sea_mean']:.1f} "
                      f"mm/day  closure={d['water_closure_inst']:.3f}  drift={drift:.2e}")
            # checkpoint once per checkpoint_every WINDOW: iter advances by n_check each loop, so an
            # ``== 0`` test silently never fires unless n_check divides checkpoint_every -- ``< n_check``
            # fires on the first check that crosses the window boundary, robust to any (n_check,
            # checkpoint_every) pairing (matches Stage A / the AD core; a guarded long run always gets a
            # periodic resume point).
            if c.checkpoint_path and self.iter % c.checkpoint_every < n_check:
                conv.save_checkpoint(c.checkpoint_path, self.state(), {"iter": self.iter})
            if stop:
                break
        if verbose:
            print(f"[StageC] STOP: {mon.stop_reason} ({mon.elapsed:.1f}s)")
        summary = mon.summary()
        d = self.diagnostics()
        self.last_run = scv.stop_record(
            mon_summary=summary, rule=rule, tol=tol, tol_units=("1/s" if model_time else "1/step"),
            check_steps_=n_check, patience=c.patience, dt=c.dt, iterations=self.iter, max_iter=max_iter,
            max_seconds=c.max_seconds, final_drift=drift, water_closure_inst=d["water_closure_inst"],
            water_closure_cum=d["water_closure_cum"], nest=self._nest is not None,
            max_model_s=(c.conv_max_model_s if model_time else None))
        return {**summary, **self.last_run}
