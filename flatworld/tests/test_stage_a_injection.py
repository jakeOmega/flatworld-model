"""Issue-A Part 2 tests: dimensionalizing Stage B's transport and injecting -dF_B/dx into Stage A.

The coupling replaces A's diffusive MSE-transport closure with B's diagnosed flux-divergence PROFILE
(shape from nondim B, magnitude from a swept anchor). These tests pin the mechanics that the four
acceptance measurements rely on: zero-flux walls (energy closure), zero-net-energy injection, a
non-circular recomputable diffusive flux for the corr metric, and an untouched diffusive default path.
"""
import numpy as np
import pytest
from flatworld import constants as C
from flatworld.stage_a_column import StageA, StageAConfig
from flatworld.stage_b_overturning import (StageBAtmosphere, StageBConfig,
                                           dimensional_poleward_flux)


def test_dimensional_poleward_flux_walls_and_anchor():
    """B's dimensional poleward flux F_B [W/m]: shape from nondim int(u*b)dz, magnitude = anchor, and
    ZERO at both walls (equator symmetry + cold wall -- the energy-closure requirement for injection)."""
    B = StageBAtmosphere(StageBConfig(nx=80, nz=25, moist=True))
    for _ in range(2500):
        B.step()
    theta, F = dimensional_poleward_flux(B, anchor=1234.0)
    assert F.shape == theta.shape
    assert F[0] == 0.0 and F[-1] == 0.0                       # zero-flux walls
    assert np.max(np.abs(F)) == pytest.approx(1234.0, rel=1e-9)   # anchor sets the peak magnitude
    # the shape is B's heat transport int(u*b)dz, not A's diffusive shape
    u, w = B.velocities()
    g_nd = np.sum(u * B.b, axis=0) * B.dze
    assert np.argmax(np.abs(F)) == np.argmax(np.abs(g_nd))    # same peak latitude as B's transport


def test_injected_transport_conserves_energy_and_applies():
    """Injecting -dF_B/dx adds ZERO net energy when the flux vanishes at both walls (the increment
    telescopes to F[0]-F[-1]=0), so A's energy closure is preserved -- and it actually moves T."""
    m = StageA(StageAConfig(nx=24, nz=16, dt=7200.0, transport_mode="injected"))
    xf = np.arange(m.cfg.nx + 1)
    F = np.sin(np.pi * xf / m.cfg.nx) * 5.0e8                 # 0 at both ends, a poleward bump
    m.set_injected_flux(F)
    inc = m._injected_transport_increment()                   # per-column T increment [K], (nx,)
    M = m.dm.sum()
    heating = inc * (M * C.CP_AIR) / m.cfg.dt                  # back out the W/m^2 forcing
    net = np.sum(heating) * m.dx                               # int(-dF/dx) dx over the closed domain
    assert abs(net) < 1e-6 * np.max(np.abs(F))                # zero net energy added
    assert np.any(inc != 0.0)                                 # the transport does something


def test_set_injected_flux_zeros_walls_and_diffusive_recompute():
    """set_injected_flux forces both walls to zero (guard); recompute_diffusive_flux returns A's OWN
    -kappa*dH/dx regardless of transport_mode (the non-circular F_A for the Issue-A corr metric)."""
    m = StageA(StageAConfig(nx=24, nz=16, dt=7200.0, transport_mode="injected"))
    m.set_injected_flux(np.ones(m.cfg.nx + 1) * 7.0)          # nonzero walls on purpose
    assert m._injected_F[0] == 0.0 and m._injected_F[-1] == 0.0
    Fd = m.recompute_diffusive_flux()
    assert Fd.shape == (m.cfg.nx + 1,)
    assert Fd[0] == 0.0 and Fd[-1] == 0.0                     # closed-domain interfaces


def test_injected_mode_no_flux_is_local_rce_and_default_is_diffusive():
    """Before B exists (no flux installed) injected mode reduces to local RCE (zero transport); the
    default transport_mode stays 'diffusive' so the 64-test gate + energy closure run the legacy path."""
    m = StageA(StageAConfig(nx=24, nz=16, transport_mode="injected"))
    assert np.all(m._mse_transport_increment() == 0.0)
    assert StageAConfig().transport_mode == "diffusive"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
