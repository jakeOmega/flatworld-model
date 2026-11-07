"""
flatworld.topo_contract -- the **Topography Contract**: the single, source-agnostic interface every
downstream consumer (the climate cascade, viz_topo, and the future biology/human layers) binds to,
instead of binding to *how* a topography was generated. This is the **G1 spine** of the geology roadmap
(GEOLOGY_ROADMAP.md): providers FILL the contract, consumers never look behind it, so the
"procedural skeleton vs. kinematic plates" question stays open forever (a kinematic-plate provider can be
swapped in later behind the same fields with zero downstream change).

THE CONTRACT (the fields a consumer may rely on, on the uniform-in-distance ``(x, e)`` grid):

  * ``mask``               -- bool land/sea (single connected ocean + enclosed basins).
  * ``elevation`` [m]      -- surface height vs. the sea-level datum, land **and** seafloor in ONE field
                             (sea cells carry their bathymetry, <=0; land cells >0). On the G5 eroded path the
                             sea cells carry REAL bathymetry (ridge + sqrt(age), trenches, graded shelves;
                             ``seafloor.build_seafloor``); on the G1 finished-map fallback they are the flat
                             basement. The cascade-facing ``height`` (0 over sea) is derived by ``to_topogrid``,
                             so the seafloor bathymetry is a worldbuilding/contract enrichment that does NOT
                             change the cascade input.
  * ``seafloor_age`` [Myr] -- seafloor age on sea cells (``nan`` on land); filled by G5
                             (``erosion.contract_from_skeleton``), ``None`` on the finished-map fallback. A
                             proxy age now (a kinematic provider's native age could fill the same field later).
  * ``rivers``             -- extracted drainage network (a ``rivers.RiverNetwork``: channels, Strahler
                             order, discharge, perennial/ephemeral, lakes/endorheic/salt flats); filled by
                             G4 (``erosion.contract_from_skeleton``), ``None`` on the finished-map fallback.
  * ``lakes``              -- bool standing water (enclosed/endorheic basins, rift lakes). On the G1
                             finished-map fallback, DERIVED from the mask's non-boundary sea components
                             (``enclosed_basins``); on the G4 eroded path, the REFINED standing-water field
                             (the inland-sea evaporation rule applied -- wet/cold basins kept, hot-dry ones
                             flagged salt flats and excluded).
  * ``sediment_thickness`` [m] -- depositional cover; zeros until G3 (erosion+deposition) fills it.
  * ``province``           -- int lithology/tectonic-province tags (oceanic/craton/orogen/arc/rift), for
                             erodibility (G2) and resources/soils (biology layer) later.
  * ``albedo_hint``        -- salt-flat / ice / land hint [0,1] (informational; NOT wired into Stage A).

WHY a thin layer over ``TopoGrid``. ``TopoGrid`` (topography.py) is the cascade's grid carrier (x, e,
land, height, meta). The contract is a SUPERSET that (a) names the richer fields the LEM stages will
populate (rivers/lakes/sediment/seafloor_age/province) and (b) keeps a clean ``to_topogrid()`` adapter so
the EXISTING cascade runs unchanged from a contract. In G1 the contract is filled by the **finished-map
fallback** (``contract_from_topogrid``): ``elevation`` is the finished tectonic height (sea=0), so
``to_topogrid()`` reproduces today's map **bit-for-bit** (the G1 regression). G2+ will instead fill the
contract by ERODING the skeleton (tectonics.skeleton), at which point ``elevation`` carries real
bathymetry and ``rivers``/``sediment_thickness`` become non-trivial -- with no change to any consumer.

This module is closed-form numpy/scipy only; it imports ``topography`` (for ``TopoGrid``) but NOT
``cascade`` (consumers import it, not the reverse).
"""
from __future__ import annotations
import numpy as np
import scipy.ndimage as ndi
from dataclasses import dataclass, field
from . import geometry as g
from .climate_anchors import CLIMATE_ANCHORS
from .topography import TopoGrid

# ---- province tags (small, stable; extend cautiously -- downstream binds to these) ------------------
OCEANIC = 0       # oceanic crust (sea floor)
CRATON = 1        # stable continental interior
OROGEN = 2        # mountain belt (convergent uplift)
ARC = 3           # volcanic island / arc
RIFT = 4          # rift valley / strait floor
PROVINCE_NAMES = {OCEANIC: "oceanic", CRATON: "craton", OROGEN: "orogen", ARC: "arc", RIFT: "rift"}

# ---- defaults for the (provider-agnostic) province heuristic ----------------------------------------
OROGEN_HEIGHT_M = 2500.0   # land above this elevation is tagged OROGEN when no explicit orogen mask is given
SALT_THETA = CLIMATE_ANCHORS["salt"]   # |theta| below which hot-zone land is salt flat. SINGLE SOURCE:
                           # climate_anchors.CLIMATE_ANCHORS (issue #83; matches tectonics/glacial/rivers).
ICE_THETA = CLIMATE_ANCHORS["ice"]     # |theta| above which land is ice-mantled. SINGLE SOURCE: climate_anchors.
ALBEDO_LAND, ALBEDO_SALT, ALBEDO_ICE, ALBEDO_SEA = 0.20, 0.40, 0.62, 0.08


@dataclass
class TopographyContract:
    """The source-agnostic topography interface (see module docstring for the field meanings). ``mask``
    and ``elevation`` are required (nx, ne) arrays on the uniform-in-distance grid (``x`` cross-track [m],
    ``e`` along-track [m]); the rest are optional/derived and may be ``None`` until a later stage fills
    them. ``meta`` carries the provider's metadata (seed, theta_lo/hi, Le_km, central_e_slice, ...), so a
    contract built from a tectonic map keeps the cascade's crop and viz's axis labels working unchanged."""
    x: np.ndarray
    e: np.ndarray
    mask: np.ndarray                                   # bool (nx, ne)
    elevation: np.ndarray                              # [m] (nx, ne); land>0, sea<=0
    seafloor_age: np.ndarray | None = None             # [Myr] on sea, nan on land (G5); None on the finished-map fallback
    rivers: object | None = None                       # rivers.RiverNetwork drainage network (G4)
    lakes: np.ndarray | None = None                    # bool standing water (refined inland-sea rule, G4)
    sediment_thickness: np.ndarray | None = None       # [m] (G3)
    province: np.ndarray | None = None                 # int tags (PROVINCE_NAMES)
    albedo_hint: np.ndarray | None = None              # [0,1]
    meta: dict = field(default_factory=dict)

    # -- grid properties (mirror TopoGrid, so a contract is a drop-in for grid reasoning) --
    @property
    def theta_deg(self):
        return np.degrees(g.theta_of_x(self.x))

    @property
    def dx(self):
        return float(self.x[1] - self.x[0])

    @property
    def de(self):
        return float(self.e[1] - self.e[0])

    def distance_to_coast_km(self):
        """Euclidean distance from each land cell to the nearest sea cell [km] (continentality)."""
        if not self.mask.any():
            return np.zeros(self.mask.shape)
        return ndi.distance_transform_edt(self.mask, sampling=[self.dx / 1e3, self.de / 1e3])

    # -- the cascade-facing adapter --
    def to_topogrid(self, h_max_m=None):
        """Project the contract onto a ``TopoGrid`` (land = mask; height = clip(elevation,0,h_max) over
        land, 0 over sea -- the Stage-C convention). The meta is carried through verbatim, so the cascade's
        central-crop and viz's axis labels keep working. For a contract built by ``contract_from_topogrid``
        (elevation = the finished height, sea=0) this reproduces the source map **bit-for-bit**."""
        if h_max_m is None:
            h_max_m = float(self.meta.get("h_max_m", np.inf))
        height = np.where(self.mask, np.clip(self.elevation, 0.0, h_max_m), 0.0)
        return TopoGrid(np.asarray(self.x, float), np.asarray(self.e, float),
                        np.asarray(self.mask, bool), np.asarray(height, float), dict(self.meta)).validate()

    def central_crop(self):
        """The DELIVERED central crop (the central Le_km of a padded tectonic provider), as a standalone
        contract with ``e`` re-based to 0 and ALL 2-D fields cropped consistently. Mirrors
        tectonics.central_crop for the TopoGrid; an unpadded contract returns an equivalent copy.

        ``rivers`` (the G4 ``RiverNetwork``) is DROPPED on a crop (set ``None``), NOT sliced: its
        ``receiver`` holds flat node indices into the FULL grid, and discharge/Strahler/up_length are
        path-integrals over the whole domain that would be invalid at a new crop boundary -- so a naive
        column-slice would silently corrupt it (and break the ``lakes == rivers.lakes`` invariant). A
        consumer that needs the network on the crop must re-extract (``rivers.extract_network``) on the
        cropped surface; G6 wires that in. (No current consumer reads ``.rivers`` post-crop.)"""
        m = dict(self.meta)
        lo, hi = m.get("central_e_slice", [0, self.e.size])
        ne_d = int(hi - lo)
        de = float(self.e[1] - self.e[0])
        e = (np.arange(ne_d) + 0.5) * de                       # re-based: the crop starts at e=0
        cut = lambda a: a[:, lo:hi].copy() if (a is not None and getattr(a, "ndim", 0) == 2) else a
        m.update(ne=ne_d, ne_full=ne_d, pad_cells=0, central_e_slice=[0, ne_d],
                 Le_km_full=float(ne_d * de / 1e3), cropped=True)
        if "land_fraction" in m:
            m["land_fraction"] = float(cut(self.mask).mean())
            m["land_fraction_full"] = m["land_fraction"]
        return TopographyContract(
            self.x.copy(), e, cut(self.mask), cut(self.elevation),
            seafloor_age=cut(self.seafloor_age), rivers=None, lakes=cut(self.lakes),
            sediment_thickness=cut(self.sediment_thickness), province=cut(self.province),
            albedo_hint=cut(self.albedo_hint), meta=m)

    def validate(self, rtol=1e-6):
        """Check the contract against the same uniform-grid assumptions Stage C relies on (1-D increasing,
        uniformly-spaced x/e; (nx, ne) mask/elevation; finite elevation). Returns self for inline use."""
        x, e, mask, elev = self.x, self.e, self.mask, self.elevation
        if x.ndim != 1 or e.ndim != 1:
            raise ValueError(f"x and e must be 1-D (got {x.ndim}-D, {e.ndim}-D)")
        nx, ne = x.size, e.size
        if mask.shape != (nx, ne) or elev.shape != (nx, ne):
            raise ValueError(f"mask and elevation must be (nx, ne)=({nx}, {ne}); got {mask.shape}, {elev.shape}")
        for name, coord in (("x", x), ("e", e)):
            if coord.size < 2:
                raise ValueError(f"{name} needs at least 2 cells")
            d = np.diff(coord)
            if d[0] <= 0 or not np.all(d > 0):
                raise ValueError(f"{name} must be strictly increasing")
            if not np.allclose(d, d[0], rtol=rtol, atol=0.0):
                raise ValueError(f"{name} must be uniformly spaced (Stage C reads a single d{name})")
        if not np.all(np.isfinite(elev)):
            raise ValueError("elevation contains non-finite values")
        return self


# =====================================================================================================
# building a contract -- the finished-map fallback (G1) and the field derivations
# =====================================================================================================
def enclosed_basins(mask):
    """Bool mask of standing inland water: sea components that do NOT touch a grid edge (the single
    connected ocean is the boundary-reaching component). The provider-agnostic baseline ``lakes`` field --
    derived from the mask alone, so any provider's map gets it for free; G4 (``rivers.extract_network``)
    refines it with the inland-sea evaporation rule (wet/cold kept, hot-dry flagged salt flats)."""
    sea = ~np.asarray(mask, bool)
    lab, n = ndi.label(sea)
    if n == 0:
        return np.zeros(mask.shape, bool)
    edge = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    edge.discard(0)
    return np.isin(lab, [i for i in range(1, n + 1) if i not in edge])


def derive_albedo_hint(theta_deg, mask, salt_theta=SALT_THETA, ice_theta=ICE_THETA):
    """Latitude-banded surface-albedo hint (salt-flat white in the hot zone, ice-white in the cold zone,
    vegetated/rock elsewhere; sea its own low value) -- the provider-agnostic version of tectonics' albedo
    field, derivable from |theta| + the mask alone. Informational (matches tectonics' note: NOT wired into
    Stage A, which is 1-D cross-track and cannot consume a 2-D albedo)."""
    abst = np.abs(np.asarray(theta_deg, float))[:, None] * np.ones(mask.shape)
    alb = np.full(mask.shape, ALBEDO_SEA)
    alb[mask] = ALBEDO_LAND
    alb[mask & (abst < salt_theta)] = ALBEDO_SALT
    alb[mask & (abst > ice_theta)] = ALBEDO_ICE
    return alb


def derive_province(mask, elevation, theta_deg, land_volc=None, rift_mask=None,
                    orogen_height_m=OROGEN_HEIGHT_M):
    """Tag each cell with a tectonic province (PROVINCE_NAMES). Provider-agnostic by construction: sea is
    OCEANIC, land is CRATON, and high land is OROGEN (a clean elevation heuristic that needs no provider
    internals). If a provider supplies extra masks they REFINE it: ``land_volc`` -> ARC (volcanic islands),
    ``rift_mask`` -> RIFT (rift/strait floor). Used for erodibility (G2) and resources/soils later."""
    prov = np.full(mask.shape, OCEANIC, np.int8)
    prov[mask] = CRATON
    prov[mask & (np.asarray(elevation) > orogen_height_m)] = OROGEN
    if land_volc is not None:
        prov[np.asarray(land_volc, bool) & mask] = ARC
    if rift_mask is not None:
        prov[np.asarray(rift_mask, bool) & mask] = RIFT
    return prov


def contract_from_topogrid(topo, fields=None):
    """Build a Topography Contract from a finished ``TopoGrid`` (the G1 fallback / regression path).
    ``elevation`` is the finished height (sea=0), so ``to_topogrid()`` reproduces the source map exactly;
    ``lakes``/``province``/``albedo_hint`` are derived; ``seafloor_age``/``rivers`` stay ``None`` and
    ``sediment_thickness`` is zeros (later stages fill them). If the optional tectonics ``fields`` dict is
    passed (from ``generate(..., return_fields=True)``), it REFINES the derivations: ``land_volc`` -> ARC
    provinces, ``rift_carve`` -> RIFT provinces, and ``albedo`` is taken verbatim rather than re-derived."""
    mask = np.asarray(topo.land, bool)
    elevation = np.asarray(topo.height, float)                 # finished height, 0 over sea -> exact round-trip
    theta = topo.theta_deg
    land_volc = rift_mask = albedo = None
    if fields is not None:
        land_volc = fields.get("land_volc")
        rc = fields.get("rift_carve")
        rift_mask = (np.asarray(rc) > 0.5) if rc is not None else None
        albedo = fields.get("albedo")
    province = derive_province(mask, elevation, theta, land_volc=land_volc, rift_mask=rift_mask)
    albedo_hint = np.asarray(albedo, float) if albedo is not None else derive_albedo_hint(theta, mask)
    return TopographyContract(
        x=np.asarray(topo.x, float), e=np.asarray(topo.e, float), mask=mask, elevation=elevation,
        lakes=enclosed_basins(mask), sediment_thickness=np.zeros(mask.shape),
        province=province, albedo_hint=albedo_hint,
        meta={**dict(topo.meta), "contract": "topogrid"}).validate()


# =====================================================================================================
# the downsample interface (G6 spine, defined in G1) + the land-fraction re-pin hook (for G3 erosion)
# =====================================================================================================
def _block_mean(a, nx_c, ne_c):
    """Area-mean a fine (nx_f, ne_f) field into a coarse (nx_c, ne_c) grid that spans the SAME physical
    extent (both grids uniform-in-distance). Each fine cell is binned to its coarse cell by index ratio --
    correct for any (not only integer) refinement ratio, and an exact block average when it is integer."""
    a = np.asarray(a, float)
    nx_f, ne_f = a.shape
    ix = np.minimum((np.arange(nx_f) * nx_c) // nx_f, nx_c - 1)
    ie = np.minimum((np.arange(ne_f) * ne_c) // ne_f, ne_c - 1)
    flat = ix[:, None] * ne_c + ie[None, :]
    s = np.bincount(flat.ravel(), weights=a.ravel(), minlength=nx_c * ne_c)
    c = np.bincount(flat.ravel(), minlength=nx_c * ne_c)
    return (s / np.maximum(c, 1)).reshape(nx_c, ne_c)


def downsample(contract, nx_c, ne_c, orography_preserving=True, target_land_fraction=None):
    """Coarse-grain a (fine, worldbuilding-resolution) contract to the cascade's climate grid (nx_c, ne_c)
    over the same physical extent -- the **downsample interface** the geology roadmap puts in G1 and wires
    in G6. ``elevation`` is the area-averaged **land height** (so coarse-grained orography -- the large-scale
    mountains and coasts the climate actually feels -- is PRESERVED: the coarse height equals the fine block
    mean, the "preserve coarse orography" default), and ``mask`` is a land-fraction-preserving **majority
    vote** (coarse cell is land iff >=50% of its fine cells are land). Province/lakes/albedo are re-derived on
    the coarse grid; rivers/seafloor_age/sediment are not carried (resolved at the fine scale only). Returns a
    new coarse ``TopographyContract`` ready for ``to_topogrid()`` -> the cascade.

    **The averaged field is the ``to_topogrid`` LAND HEIGHT (G6 fix), not the raw signed elevation.** On a G5
    contract the sea cells carry deep bathymetry (to -11 km at trenches), so block-meaning the *signed*
    elevation over land AND sea pulls a coarse *coastal* (majority-land) cell down toward its deep sub-cells --
    and ``to_topogrid`` then clips that at 0, silently losing coastal land height (G5_RESULTS flagged this).
    We instead block-mean ``where(mask, clip(elevation,0,h_max), 0)`` -- the cascade-facing height, already 0
    over sea -- so the coarse land height is >=0, immune to the trench depth, and EQUAL to what downsampling
    the finished-map (sea=0) target would give (the "preserved-orography target" the G6 gate checks). The
    coarse ``elevation`` is therefore >=0 (sea cells carry a small positive value from any minority land
    sub-cells; ``to_topogrid`` zeroes them via the sea mask), which is also the continuous field the G6
    post-downsample ``repin_land_fraction`` cut operates on.

    ``target_land_fraction`` (G6): if given, the coarse ``mask`` is RE-PINNED on ``elevation`` to hit that
    land fraction (the prescribed downsample -> ``repin_land_fraction`` step), instead of the bare majority
    vote (which drifts ~1-2% on a coarse grid). The LEM cascade driver passes the fine map's own land
    fraction so the climate grid carries the same land area the worldbuilding map delivers."""
    x_lo = float(contract.x[0] - 0.5 * contract.dx); x_hi = float(contract.x[-1] + 0.5 * contract.dx)
    Le = float(contract.e[-1] + 0.5 * contract.de)
    xc = x_lo + (np.arange(nx_c) + 0.5) * (x_hi - x_lo) / nx_c
    ec = (np.arange(ne_c) + 0.5) * Le / ne_c
    h_max = float(contract.meta.get("h_max_m", np.inf))
    land_height = np.where(contract.mask, np.clip(contract.elevation, 0.0, h_max), 0.0)   # the to_topogrid height
    elev_c = _block_mean(land_height, nx_c, ne_c)                 # orography-preserving, trench-immune (>=0)
    land_frac_c = _block_mean(contract.mask.astype(float), nx_c, ne_c)
    mask_c = land_frac_c >= 0.5                                   # land-fraction-preserving majority vote
    if target_land_fraction is not None:
        # G6 post-downsample RE-PIN (the prescribed downsample -> repin_land_fraction step): the majority vote
        # drifts the land fraction by a percent or two on a coarse grid, so re-pin the coarse waterline on the
        # orography-preserving height field to hit ``target_land_fraction`` exactly (default: the fine map's
        # own land fraction, so the climate sees the same land area the worldbuilding map delivers).
        mask_c, _ = repin_land_fraction(elev_c, float(target_land_fraction))
    theta_c = np.degrees(g.theta_of_x(xc))
    # Carry the PADDED-vs-delivered split proportionally. The fine map is generated on a domain padded by
    # ``pad_km`` each along-track side (so Stage C's periodic-e seam stays in the pad); the cascade crops to the
    # central delivered region after Stage C (``_deliver_crop``). Coarse-graining scales the crop slice by
    # ne_c/ne_fine so the coarse grid keeps the same pad isolation. An UNPADDED fine contract (slice = full grid)
    # gives the identity slice [0, ne_c] -- the pre-G6 behavior.
    ne_fine = contract.mask.shape[1]
    flo, fhi = contract.meta.get("central_e_slice", [0, ne_fine])
    clo = max(0, min(int(round(flo * ne_c / ne_fine)), ne_c - 1))
    chi = max(clo + 1, min(int(round(fhi * ne_c / ne_fine)), ne_c))
    de_c = float(ec[1] - ec[0]) if ne_c > 1 else Le
    m = dict(contract.meta)
    m.update(nx=nx_c, ne=int(chi - clo), downsampled_from=tuple(contract.mask.shape),
             central_e_slice=[clo, chi], pad_cells=int(clo), ne_full=ne_c, Le_km_full=float(ne_c * de_c / 1e3),
             land_fraction=float(mask_c[:, clo:chi].mean()), land_fraction_full=float(mask_c.mean()),
             contract="downsampled", orography_preserving=bool(orography_preserving))
    return TopographyContract(
        x=xc, e=ec, mask=mask_c, elevation=elev_c, lakes=enclosed_basins(mask_c),
        sediment_thickness=np.zeros(mask_c.shape),
        province=derive_province(mask_c, elev_c, theta_c),
        albedo_hint=derive_albedo_hint(theta_c, mask_c), meta=m).validate()


def repin_land_fraction(elevation, target_fraction, cont_mask=None, sea_level=0.0):
    """Re-pin the waterline so the land fraction equals ``target_fraction`` -- the **hook** the erosion
    stage (G3) calls AFTER erosion moves coastlines (erosion is not area-neutral; the adaptive-quantile sea
    level that pinned the original land fraction must be re-applied). Returns ``(land_mask, new_sea_level)``,
    with ``land = elevation > new_sea_level``.

    Without ``cont_mask`` this is a global quantile cut (``new_sea_level = quantile(elevation, 1-f)``). With
    ``cont_mask`` it re-pins ONLY within continental crust (preserving the pinned continental land COUNT,
    the area-neutral cut tectonics uses), leaving non-continental cells (volcanic/ocean) at the absolute
    ``sea_level`` datum -- mirroring tectonics' Stage-H freeboard pin so heights stay physical. The
    continental pin can only place land on continental cells, so it raises ``ValueError`` if the target
    land count does not fit inside ``cont_mask`` (rather than silently flooding to the datum and missing
    the target)."""
    elevation = np.asarray(elevation, float)
    f = float(np.clip(target_fraction, 0.0, 1.0))
    if cont_mask is None:
        lvl = float(np.quantile(elevation, 1.0 - f))
        return elevation > lvl, lvl
    cont_mask = np.asarray(cont_mask, bool)
    n_pin = int(round(f * elevation.size))                     # target land COUNT (global)
    zc = elevation[cont_mask]
    if not 0 < n_pin < zc.size:                                # the pin lands ON continental cells only:
        raise ValueError(                                     # outside (0, n_cont) it cannot hit the target
            f"repin_land_fraction: target land count {n_pin} (fraction {f:.3f} of {elevation.size} cells) "
            f"does not fit the {int(zc.size)} continental cells; the continental-area-preserving pin cannot "
            f"reach this land fraction (widen cont_mask, or use the global cut with cont_mask=None)")
    lvl = float(np.partition(zc, zc.size - n_pin)[zc.size - n_pin])
    land = (cont_mask & (elevation > lvl)) | ((~cont_mask) & (elevation > sea_level))
    return land, lvl
