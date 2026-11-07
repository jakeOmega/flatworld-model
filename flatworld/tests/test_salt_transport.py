"""Acceptance tests for the wind-keyed salt-aerosol SOURCE/TRANSPORT scheme (GitHub #237, the
source/transport checkbox -- ``flatworld.salt_transport``).

The scheme turns the salt-CCN droplet-number field ``N_d(theta)`` from a PRESCRIBED scalar (#243) into an
EMERGENT field: wind-keyed saltation/dust emission off the equatorial salt flats (threshold + u*^3),
size-binned, lofted by the cell and settled along the poleward-aloft return so coarse re-deposits near
the line and the fine accumulation mode reaches the rain band as CCN. Tests cover the emission physics
(against the ``../aeolian_salt.py`` BOE relations), the size-selective transport, the field assembly, the
soft-knob monotonicity, and the AD-core wiring (default OFF byte-identical; transport mode emergent from
the resolved wind). All fast (pure-numpy or low-res column physics).
"""
import numpy as np
import pytest

from flatworld import constants as C
from flatworld import geometry as g
from flatworld import salt_transport as st
from flatworld.stage_ad_core import StageADCore, StageADConfig

PROG = dict(prognostic_condensate=True)


def _grid(nx=120, theta_max=58.0):
    x_max = g.x_of_theta(np.radians(theta_max))
    x = (np.arange(nx) + 0.5) * (x_max / nx)
    return np.degrees(g.theta_of_x(x)), x


def _flats_wind(theta_deg, peak=12.0, floor=2.0):
    """A plausible single-cell surface branch: ~peak m/s over the equatorial flats, decaying poleward."""
    return peak * np.exp(-(theta_deg - 10.0) ** 2 / (2 * 15.0 ** 2)) + floor


# ============================================================ emission physics (BOE cross-checks)
def test_threshold_is_supercritical_over_the_flats():
    """The headline FIRM finding (issue #237): the ~12 m/s flats wind is steadily supercritical for halite
    entrainment, u*/u*t ~ 1.4-2.0 (not a rare gust)."""
    rho = st.air_density(125.0)                                   # hot flats air (~0.89 kg/m^3)
    ustar = st.friction_velocity(12.0, z0=1e-3)                  # 12 m/s 10-m wind
    ustar_t = st.threshold_ustar(125e-6, rho)                    # optimal halite grain
    assert 1.3 < ustar / ustar_t < 2.1


def test_air_density_matches_ideal_gas_at_125C():
    """Flats air density is a real atmosphere (~26% below the 1.2 reference), not thin (BOE input)."""
    rho = st.air_density(125.0)
    assert abs(rho - C.P_SURF / (C.R_DRY * (125.0 + C.T_FREEZE))) < 1e-9
    assert 0.85 < rho < 0.95


def test_saltation_flux_zero_below_threshold_and_cubic_above():
    """Owen's law: Q == 0 below threshold, and Q grows ~u*^3 above it (the u*^3 form the issue specifies)."""
    rho = 0.89
    ut = 0.3
    assert st.saltation_flux(0.25, ut, rho) == 0.0               # sub-threshold -> no emission
    q1 = st.saltation_flux(0.6, ut, rho)
    q2 = st.saltation_flux(1.2, ut, rho)                         # double u*
    assert q1 > 0.0
    # ~2^3 = 8 baseline (cubic), modulated UP by the (1-(ut/u*)^2) gate (it rises toward 1 far from threshold)
    assert 8.0 <= q2 / q1 < 11.0


def test_settling_velocity_is_stokes_and_grows_as_d_squared():
    """Stokes terminal velocity ~ d^2: a 10x grain settles 100x faster (the size-selective transport driver)."""
    v1 = st.settling_velocity(1e-6)
    v10 = st.settling_velocity(10e-6)
    assert np.isclose(v10 / v1, 100.0, rtol=1e-6)


def test_emission_only_on_the_flats():
    """Emission is zero poleward of the salt sill (no bare evaporite pan to erode there) regardless of wind."""
    theta, x = _grid()
    u10 = np.full_like(theta, 14.0)                              # strong wind everywhere
    cfg = st.SaltTransportConfig(flats_theta_hi=25.0)
    _, diag = st.build_nd_field(theta, x, u10, cfg=cfg)
    assert np.all(diag["Q"][theta > 25.0] == 0.0)               # no saltation poleward of the flats
    assert np.any(diag["Q"][theta <= 25.0] > 0.0)               # but the flats erode under a 14 m/s wind


# ============================================================ size-selective transport (the three fates)
def test_size_fates_coarse_to_flats_fine_to_band():
    """The FIRM size split (issue #237): coarse grains re-deposit on the flats (the sun-line dunes), an
    intermediate size leaks over the cool sea, and only the fine accumulation mode reaches the rain band."""
    theta, x = _grid()
    _, diag = st.build_nd_field(theta, x, _flats_wind(theta))
    d_um = diag["d_um"]
    fate = diag["fate"]
    # the coarsest bin stays near the line; the finest reaches the band
    assert fate[np.argmax(d_um)] == "flats"
    assert fate[np.argmin(d_um)] == "band"
    # fate is monotone in size: once you're "flats" you never become "band" at a larger size
    order = np.argsort(d_um)
    rank = {"band": 0, "leak": 1, "flats": 2}
    seq = [rank[f] for f in fate[order]]
    assert seq == sorted(seq)                                    # non-decreasing with diameter


def test_band_loading_is_continental():
    """The headline precip-relevant result: the band crosses from clean-maritime (~100 cm^-3) into the
    continental-polluted regime (>= 1000 cm^-3) under the eroding flats -- the CCN-sensitive jump."""
    theta, x = _grid()
    Nd, diag = st.build_nd_field(theta, x, _flats_wind(theta))
    assert diag["N_d_band"] >= 1000.0                           # continental loading at the band
    assert diag["N_d_peak"] >= diag["N_d_band"]                 # peak over the source, decaying poleward


def test_field_decays_poleward_of_the_loft_point():
    """The aloft return concentration decays monotonically poleward (gravitational + dry-dep settling), so
    N_d is highest near the source and falls toward the cold edge."""
    theta, x = _grid()
    Nd, _ = st.build_nd_field(theta, x, _flats_wind(theta))
    pole = theta > 5.0                                          # poleward of the loft point
    assert np.all(np.diff(Nd[pole]) <= 1e-9)


def test_clean_when_wind_subcritical_everywhere():
    """A calm wind never crosses threshold -> Q == 0 -> N_d == N_ref everywhere (the exact clean no-op that
    keeps the master coupling inert until the flats actually erode)."""
    theta, x = _grid()
    cfg = st.SaltTransportConfig(n_d_ref=100.0)
    Nd, diag = st.build_nd_field(theta, x, np.full_like(theta, 0.5), cfg=cfg)
    assert np.all(diag["Q"] == 0.0)
    assert np.allclose(Nd, 100.0)


# ============================================================ soft-knob monotonicity (swept, not tuned)
def test_band_loading_monotone_in_sandblast_efficiency():
    """alpha (the dominant emission-magnitude knob) raises the band loading monotonically -- a clean SWEEP
    axis (the issue's 1e-5..1e-4 Owens-Lake/Bodele range), reported not tuned."""
    theta, x = _grid()
    u10 = _flats_wind(theta)
    bands = [st.build_nd_field(theta, x, u10, cfg=st.SaltTransportConfig(alpha_sandblast=a))[1]["N_d_band"]
             for a in (1e-5, 3e-5, 1e-4)]
    assert bands[0] < bands[1] < bands[2]


def test_band_loading_monotone_decreasing_in_deposition_floor():
    """A larger dry-deposition floor shortens the accumulation-mode lifetime -> LESS reaches the band. This
    is the lifetime knob that bounds the band loading (the BOE's lumped 'surviving fraction')."""
    theta, x = _grid()
    u10 = _flats_wind(theta)
    bands = [st.build_nd_field(theta, x, u10, cfg=st.SaltTransportConfig(v_dep_min=v))[1]["N_d_band"]
             for v in (2e-4, 5e-4, 1e-3)]
    assert bands[0] > bands[1] > bands[2]


def test_delivered_band_fraction_is_field_truth_not_gravity_split():
    """The gravity-only size-fate (`fine_to_band_frac`) over-states delivery: the FIELD's `delivered_band_frac`
    (on reach_dep = gravity + the dry-dep floor) is strictly smaller, and it is what sets N_d_band. Locks the
    two-delivery-stories distinction (review #237 comment 1)."""
    theta, x = _grid()
    Nd, diag = st.build_nd_field(theta, x, _flats_wind(theta))
    assert 0.0 < diag["delivered_band_frac"] < diag["fine_to_band_frac"]
    # N_d_band is the clean background lifted by the delivered number: consistent within float round-off
    assert diag["N_d_band"] > st.SaltTransportConfig().n_d_ref


def test_delivered_band_fraction_tracks_deposition_floor():
    """A larger dry-deposition floor shortens reach_dep -> less of the lofted number survives to the band ->
    `delivered_band_frac` falls (the lifetime knob acting on the FIELD, not the gravity split)."""
    theta, x = _grid()
    u10 = _flats_wind(theta)
    fr = [st.build_nd_field(theta, x, u10, cfg=st.SaltTransportConfig(v_dep_min=v))[1]["delivered_band_frac"]
          for v in (2e-4, 5e-4, 1e-3)]
    assert fr[0] > fr[1] > fr[2]


def test_emission_size_distribution_normalized():
    """The lognormal emission mass fractions sum to 1 (number is then f_i*F/m_i -> weighted to the fine mode)."""
    d, f = st.SaltTransportConfig().bin_diameters()
    assert np.isclose(f.sum(), 1.0)
    assert np.all(f >= 0.0)
    assert d[0] < d[-1]                                          # ascending diameters


# ============================================================ u_surf reduction wrapper
def test_nd_field_from_u_surf_uses_speed_not_sign():
    """The core-facing wrapper keys emission on wind SPEED -- a signed (equatorward) u_surf gives the same
    field as its absolute value."""
    theta, x = _grid()
    u = _flats_wind(theta)
    a, _ = st.nd_field_from_u_surf(theta, x, -u)
    b, _ = st.nd_field_from_u_surf(theta, x, u)
    assert np.array_equal(a, b)


# ============================================================ AD-core wiring (default OFF byte-identical)
def _base(**kw):
    return dict(nx=16, nz=16, theta_max_deg=55.0, **kw)


def test_source_defaults_prescribed():
    assert StageADConfig(**_base()).salt_ccn_source == "prescribed"


def test_bad_source_raises():
    with pytest.raises(ValueError):
        StageADCore(StageADConfig(salt_ccn=True, salt_ccn_source="bogus", **PROG, **_base()))


def test_transport_refresh_zero_warns_about_silent_noop():
    """salt_ccn_source='transport' with refresh_every=0 (the default) warns that N_d stays clean until a manual
    rebuild — the footgun guard (review #237 comment 3): a production transport run that never refreshes is a
    silent no-op."""
    with pytest.warns(UserWarning, match="silent no-op"):
        StageADCore(StageADConfig(salt_ccn=True, salt_ccn_source="transport", **PROG, **_base()))


@pytest.mark.filterwarnings("ignore:salt_ccn_source='transport'")
def test_transport_cold_start_is_clean_noop():
    """Transport mode at construction (no resolved wind yet) builds the CLEAN reference field -> an exact
    no-op until refreshed. The master ON at the reference must stay byte-identical to OFF."""
    core = StageADCore(StageADConfig(salt_ccn=True, salt_ccn_source="transport", salt_ccn_nd_ref=100.0,
                                     **PROG, **_base()))
    assert core.N_d is not None
    assert np.allclose(np.asarray(core.N_d), 100.0)


def test_transport_off_is_byte_identical_to_baseline():
    """salt_ccn OFF (default) -> N_d None -> the core is byte-identical to the pre-feature core, source knob
    notwithstanding (the load-bearing OFF guard for this PR)."""
    base = _base(clouds=True, **PROG)
    off = StageADCore(StageADConfig(salt_ccn=False, **base))
    # a transport-configured core with the MASTER off must also be inert (knobs set but salt_ccn False)
    knobbed = StageADCore(StageADConfig(salt_ccn=False, salt_ccn_source="transport",
                                        salt_ccn_transport_alpha=1e-4, **base))
    assert off.N_d is None and knobbed.N_d is None
    for m in (off, knobbed):
        m.run_dynamics = False
    for _ in range(15):
        off.step(); knobbed.step()
    assert np.array_equal(off.T_atm, knobbed.T_atm)
    assert np.array_equal(off.q, knobbed.q)
    assert np.array_equal(off._precip, knobbed._precip)


@pytest.mark.filterwarnings("ignore:salt_ccn_source='transport'")
def test_transport_builds_emergent_field_from_resolved_wind():
    """With a resolved lowest-level wind injected and the field rebuilt, transport mode loads the band above
    the clean reference (emergent N_d) and the field decays poleward."""
    core = StageADCore(StageADConfig(salt_ccn=True, salt_ccn_source="transport", grid_type="stretched",
                                     dz_min=20.0, salt_ccn_nd_ref=100.0, salt_ccn_transport_alpha=3e-5,
                                     **PROG, **_base()))
    core._u_surf = np.full(core.cfg.nx, 12.0)                    # ~12 m/s flats wind (the sim peak)
    core._build_salt_ccn_field()
    nd = np.asarray(core.N_d)
    assert nd.max() > 1000.0                                     # continental loading emerged
    assert nd[0] > nd[-1]                                        # equatorward (flats) source decays poleward


@pytest.mark.filterwarnings("ignore:salt_ccn_source='transport'")
def test_transport_at_reference_alpha_zero_is_clean():
    """alpha == 0 (no sandblasting) emits no dust -> N_d == N_ref even with a strong wind: the scheme's own
    clean zero-point, so a mis-set transport run still reduces to the no-op."""
    core = StageADCore(StageADConfig(salt_ccn=True, salt_ccn_source="transport", grid_type="stretched",
                                     dz_min=20.0, salt_ccn_nd_ref=100.0, salt_ccn_transport_alpha=0.0,
                                     **PROG, **_base()))
    core._u_surf = np.full(core.cfg.nx, 15.0)
    core._build_salt_ccn_field()
    assert np.allclose(np.asarray(core.N_d), 100.0)


def test_transport_refresh_runs_in_loop_when_enabled():
    """salt_ccn_transport_refresh_every > 0 rebuilds N_d from the live wind during step() (emergent in a
    single run); default 0 leaves it built-once. Here a hand-set wind + a refresh cadence updates the field."""
    core = StageADCore(StageADConfig(salt_ccn=True, salt_ccn_source="transport", grid_type="stretched",
                                     dz_min=20.0, salt_ccn_nd_ref=100.0, salt_ccn_transport_refresh_every=1,
                                     **PROG, **_base()))
    core.run_dynamics = False                                   # isolate the refresh hook from the dynamics
    core._u_surf = np.full(core.cfg.nx, 13.0)                   # inject a resolved wind
    assert np.allclose(np.asarray(core.N_d), 100.0)            # still clean (built at cold start)
    core.step()                                                 # refresh_every=1 -> rebuild from the live wind
    assert np.asarray(core.N_d).max() > 1000.0                 # the field went emergent mid-run
