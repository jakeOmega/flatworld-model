"""
flatworld.operators -- the shared (x, e) finite-difference operators for the 2-D maps.

The boundary-layer maps (Stage C, and the #111 momentum solve plus its #112/#114/#115 siblings) all
discretize the same (x, e) plane: cross-track x has WALLS (equator at x[0], pole at x[-1]) and is
one-sided there; along-track e is PERIODIC (a slab of the endless ribbon), done with ``np.roll``.
Factoring these operators out of ``stage_c_map`` keeps the wall/periodic conventions in ONE place so
every consumer differences the grid identically (a divergent boundary-layer wind needs a conservative
flux-form advection for a closed budget, and the periodic e-seam must be smoothed/differenced
consistently or a grid-scale del^2 spike is reborn at the seam -- issue #82).

These are the byte-identical free-function forms of the former ``StageCMap._dx/_de/_lap/_div/_adv_flux``
methods; ``StageCMap`` now calls them.

The closed cross-track walls are the CLOSED-DOMAIN convention. A NESTED sub-domain (GEO-22,
``flatworld.climate_nest``) sits mid-band with live wind crossing its x-walls, so its conservative
advection takes ``adv_flux(..., open_x=True)`` -- outflow leaves, nothing enters. That is the only
operator with an open-boundary form; everything else here is closed/periodic as described above.
"""
from __future__ import annotations
import numpy as np


def ddx(F, dx):
    """Cross-track derivative dF/dx: centered in the interior, one-sided at the x-walls (equator/pole)."""
    o = np.zeros_like(F)
    o[1:-1, :] = (F[2:, :] - F[:-2, :]) / (2 * dx)
    o[0, :] = (F[1, :] - F[0, :]) / dx
    o[-1, :] = (F[-1, :] - F[-2, :]) / dx
    return o


def dde(F, de):
    """Along-track derivative dF/de: centered, PERIODIC in e (np.roll)."""
    return (np.roll(F, -1, axis=1) - np.roll(F, 1, axis=1)) / (2 * de)   # periodic in e


def lap(F, dx, de):
    """Laplacian. The cross-track (x) part is zeroed on the x-walls (d2x[0]=d2x[-1]=0) -- one-sided /
    NOT strictly flux-conserving there, used only for the small eddy-diffusion term (the conservative
    transport rides on adv_flux). e is periodic (np.roll)."""
    d2x = np.zeros_like(F)
    d2x[1:-1, :] = (F[2:, :] - 2 * F[1:-1, :] + F[:-2, :]) / dx**2
    d2e = (np.roll(F, -1, axis=1) - 2 * F + np.roll(F, 1, axis=1)) / de**2
    return d2x + d2e


def div(U, V, dx, de):
    """Divergence dU/dx + dV/de (same wall/periodic conventions as ddx/dde)."""
    return ddx(U, dx) + dde(V, de)


def upwind_adv(F, U, V, dx, de):
    """ADVECTIVE (non-conservative) upwind material derivative -(V.grad)F = -(U dF/dx + V dF/de),
    one-sided in the upwind direction (backward where the velocity is positive, forward where negative).
    This is the right form for MOMENTUM under a DIVERGENT flow (the flux form -div(VF) would inject a
    spurious F*div(V)), and the consistent single-discretization upwinding is numerically stable (unlike
    flux-form + a centered divergence correction, whose discretizations don't cancel). Walled in x
    (one-sided collapses to the available interior face at the equator/pole), periodic in e (np.roll)."""
    # x (walls): backward/forward one-sided, no flux across the walls
    dF_back = np.zeros_like(F); dF_fwd = np.zeros_like(F)
    dF_back[1:, :] = (F[1:, :] - F[:-1, :]) / dx       # valid for i>=1; row 0 stays 0 (wall)
    dF_fwd[:-1, :] = (F[1:, :] - F[:-1, :]) / dx        # valid for i<=nx-2; row -1 stays 0 (wall)
    dFdx = np.where(U > 0, dF_back, dF_fwd)
    # e (periodic)
    dFde_back = (F - np.roll(F, 1, axis=1)) / de
    dFde_fwd = (np.roll(F, -1, axis=1) - F) / de
    dFde = np.where(V > 0, dFde_back, dFde_fwd)
    return -(U * dFdx + V * dFde)


def adv_flux(F, U, V, dx, de, open_x=False):
    """Conservative FLUX-FORM upwind advection -div(V F) = -[d(UF)/dx + d(VF)/de]. Periodic in e.

    CROSS-TRACK WALLS, two forms:

    * ``open_x=False`` (DEFAULT, the closed domain): ZERO flux through both x-walls, so the domain
      integral of F is conserved under transport (only sources/sinks change it) -- the right law for
      the full gallery domain, whose x-walls are the equator and the far cold field, and what a
      closed water budget needs under a divergent boundary-layer wind.
    * ``open_x=True`` (the NEST's open boundary): one-sided upwind OUTFLOW through both x-walls.
      The boundary-face flux is ``min(U[0], 0)*F[0]`` at x-lo and ``max(U[-1], 0)*F[-1]`` at x-hi:
      F leaves only where the wall-cell velocity points OUT, and nothing ever enters (there is no
      fluid outside the window -- inflow information arrives through the rim relaxation instead).
      The domain integral then falls by exactly the outbound boundary flux, and an INBOUND wall
      velocity reduces to the closed form identically. A nested sub-domain whose x-lo wall sits
      mid-band against a live equatorward wind needs this: sealed, that wall accumulates every
      outbound parcel into a few-cell strip (GEO-22 -- MODELING_LESSONS).

      PRECONDITION for "the open wall only ever removes F": ``F >= 0`` in the two wall ROWS. The wall
      term is ``velocity x F``, so a NEGATIVE wall cell under an outbound velocity carries a negative
      flux out, i.e. it ADDS to the domain integral. Every current caller satisfies this -- Stage C's
      ``W_s``/``W_a`` are clamped non-negative every step -- but a signed tracer (a temperature
      anomaly, a velocity component) would need the sign argument redone before using ``open_x``.
    """
    # x-faces (between i,i+1), upwind on the face velocity
    uf = 0.5 * (U[:-1, :] + U[1:, :])
    Fx = uf * np.where(uf > 0, F[:-1, :], F[1:, :])          # (nx-1, ne)
    dFx = np.zeros_like(F)
    dFx[1:-1, :] = (Fx[1:, :] - Fx[:-1, :]) / dx
    dFx[0, :] = Fx[0, :] / dx                                # wall: only the interior face flux
    dFx[-1, :] = -Fx[-1, :] / dx
    if open_x:
        # add the WALL-face fluxes back in with the interior faces' sign convention: dFx[i] is
        # (right-face flux - left-face flux)/dx, so the x-lo wall face SUBTRACTS and the x-hi ADDS.
        dFx[0, :] -= np.minimum(U[0, :], 0.0) * F[0, :] / dx
        dFx[-1, :] += np.maximum(U[-1, :], 0.0) * F[-1, :] / dx
    # e-faces (periodic)
    vf = 0.5 * (V + np.roll(V, -1, axis=1))
    Fe = vf * np.where(vf > 0, F, np.roll(F, -1, axis=1))
    dFe = (Fe - np.roll(Fe, 1, axis=1)) / de
    return -(dFx + dFe)
