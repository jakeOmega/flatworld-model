"""Device (CuPy) linear-solve kernels for the AD-core dynamics limb — the two host linear solves put
ON DEVICE (Phase C of the GPU port; see ``archive/_gpu_port_audit_RESULTS.md`` §4 and ``archive/_gpu_phase_c_RESULTS.md``).

This module holds the GPU analogues of the two host solves in the dynamics substep:

  1. ``GPUSeparablePoisson`` — the device port of ``stage_ad_core.SeparablePoisson`` (the separable
     DST-I (x) x batched-Thomas (z) anelastic streamfunction Poisson). DST-I is NOT in cupyx
     (``cupyx.scipy.fft.dst`` is type-II/III only), so the DST-I is computed via the standard
     ODD-EXTENSION rFFT identity, which CuPy supports and which reproduces scipy's ``dst(type=1)`` to
     machine precision in fp64. Validated vs the CPU ``SeparablePoisson`` (and originally as the
     ``_gpu_dst_probe`` / ``_gpu_graph_probe`` go/no-go GREEN).

  2. ``build_tridiag_cols_kernel`` — a one-thread-per-column Thomas RawKernel for the per-column
     tridiagonals of ``_implicit_vertical_diffusion`` (drop-in for ``_solve_tridiag_columns``). The
     internal forward+back nz march is a SINGLE kernel launch (the fast shape), validated vs the
     production ``_solve_tridiag_columns`` (NaN-poison gated in the probe).

It imports **nothing from ``stage_ad_core``** so it can be imported from ``stage_ad_core.__init__``
without a cycle (the probes — ``_gpu_dst_probe``/``_gpu_graph_probe``/``_gpu_substep_full`` — import the
kernels from HERE, and import the CPU reference/harness from ``stage_ad_core`` as before).

CuPy is an OPTIONAL dependency: this module imports without it (guarded ``_cp``); the kernel builders
raise a friendly error if called with no device, and the DST-I helpers are array-module-generic
(``xp`` passed in), so the same ``_dst1`` runs under numpy in a CPU unit test."""
from __future__ import annotations
import numpy as np

try:                                  # optional GPU backend; absent on CPU-only sessions
    import cupy as _cp
    _HAVE_CUPY = True
except Exception:                     # ImportError, or a CuPy that fails to load on a driver-less box
    _cp = None
    _HAVE_CUPY = False


# --------------------------------------------------------------------------- DST-I via odd-extension rFFT
def _dst1(x, xp):
    """DST-I along the last axis (length M) via an odd-extension rFFT of length 2(M+1). Reproduces
    ``scipy.fft.dst(x, type=1, axis=-1)`` exactly (fp64) and to ~1e-7 (fp32); dtype-preserving. Works for
    xp = numpy OR cupy (CuPy lacks a native DST-I).

        DST-I_M(x)_k = -Im( rFFT( [0, x_0..x_{M-1}, 0, -x_{M-1}..-x_0] ) )_{k+1},  k=0..M-1   (len 2(M+1))
    """
    M = x.shape[-1]
    N = 2 * (M + 1)
    y = xp.zeros(x.shape[:-1] + (N,), dtype=x.dtype)
    y[..., 1:M + 1] = x
    y[..., M + 2:] = -x[..., ::-1]
    Y = xp.fft.rfft(y, axis=-1)
    return (-Y.imag)[..., 1:M + 1]


def _idst1(x, xp):
    """Inverse DST-I = forward DST-I / (2(M+1)) (scipy normalization), so it cancels exactly with the
    forward transform in the separable inversion (the unnormalized-pair identity the CPU solver relies on)."""
    M = x.shape[-1]
    return _dst1(x, xp) / (2.0 * (M + 1))


# --------------------------------------------------------------------------- GPU separable Poisson solver
class GPUSeparablePoisson:
    """Device (CuPy) port of ``stage_ad_core.SeparablePoisson``: a DROP-IN separable inverter for the
    closed-wall anelastic streamfunction Poisson, with the EXACT same math (same discrete Lxx eigenvalues,
    same z-face coefficients, same batched-Thomas factors), running entirely on the GPU in the requested
    dtype. The Thomas factors (inv_pivot/cprime) are precomputed ONCE on device (fixed topology; production
    reuses them every substep), so per-solve cost is DST + the z-swept Thomas sweeps + iDST only.

    The batched Thomas sweep mirrors the CPU structure faithfully — it loops over the SMALL z-axis
    (sequential, ~2*nz tiny kernel launches per solve) and vectorizes across the nx-2 wavenumber modes.

    ``dtype`` is ``cp.float64`` (the production seam casts the Poisson RHS up to fp64) or ``cp.float32``.
    Inputs/outputs stay on device. ``solve`` accepts and returns an (nz, nx) device array (NOT flat — the
    production caller already has the 2-D omega/psi)."""

    def __init__(self, rho_ref, nx, nz, dx, dz, dtype):
        if not _HAVE_CUPY:
            raise RuntimeError("GPUSeparablePoisson requires CuPy (the GPU backend). Install "
                               "requirements-gpu.txt or use the CPU SeparablePoisson.")
        self.nx, self.nz = int(nx), int(nz)
        self.dtype = dtype
        beta = 1.0 / np.asarray(rho_ref, float)
        beta_zface = 0.5 * (beta[:-1] + beta[1:])
        ix2 = 1.0 / dx ** 2

        ji = np.arange(1, nz - 1)
        if np.ndim(dz) == 0:
            iz2 = 1.0 / dz ** 2
            cjp = beta_zface[ji] * iz2
            cjm = beta_zface[ji - 1] * iz2
        else:
            z = np.asarray(dz, float)
            p = ji - 1
            hp_i = z[2:] - z[1:-1]
            hm_i = z[1:-1] - z[:-2]
            dzc_i = 0.5 * (z[2:] - z[:-2])
            cjp = beta_zface[ji] / (hp_i[p] * dzc_i[p])
            cjm = beta_zface[ji - 1] / (hm_i[p] * dzc_i[p])

        m = np.arange(1, nx - 1)
        mu = -4.0 * np.sin(m * np.pi / (2.0 * (nx - 1))) ** 2 / dx ** 2

        diag = (beta[ji][:, None] * mu[None, :]) - (cjp + cjm)[:, None]   # (nz-2, nx-2)
        nzi, nxi = nz - 2, nx - 2
        inv_pivot = np.empty((nzi, nxi))
        cprime = np.zeros((nzi, nxi))
        pivot = diag[0].copy()
        inv_pivot[0] = 1.0 / pivot
        for jj in range(1, nzi):
            cprime[jj - 1] = cjp[jj - 1] / pivot
            pivot = diag[jj] - cjm[jj] * cprime[jj - 1]
            inv_pivot[jj] = 1.0 / pivot

        # Move the (RHS-independent) factors to device ONCE, in the working dtype.
        self.beta = _cp.asarray(beta, dtype)
        self.ix2 = dtype(ix2)
        self.cjp = _cp.asarray(cjp, dtype)
        self.cjm = _cp.asarray(cjm, dtype)
        self.inv_pivot = _cp.asarray(inv_pivot, dtype)
        self.cprime = _cp.asarray(cprime, dtype)
        self._nzi, self._nxi = nzi, nxi
        # FUSED z-Thomas sweep (one thread per wavenumber mode): collapse the ~2*(nz-2) tiny launches of the
        # Python forward/back loops into ONE kernel. Built ONCE here (NVRTC compile happens at construction,
        # in warmup -- never inside CUDA capture). ``fused_thomas`` toggles it off for the parity probe (the
        # Python loop is the round-off reference). nzi must fit the kernel scratch (_TRIDIAG_MAXNZ).
        # The kernel is templated on ``dtype`` and is CORRECT at fp32 too (identical arithmetic in float), but
        # the production solver is fp64 (the Poisson is deliberately fp64 -- ill-conditioned) and
        # ``_gpu_dst_thomas_parity`` only covers fp64; an fp32 construction would run the fused path WITHOUT
        # parity coverage, so add an fp32 case to that probe before relying on a fp32 GPUSeparablePoisson.
        self.fused_thomas = nzi <= _TRIDIAG_MAXNZ
        self._thomas_kernel = build_dst_thomas_kernel(dtype) if self.fused_thomas else None

    def solve(self, b):
        """Invert M psi = b on device (b an (nz, nx) CuPy array of self.dtype). Returns psi (nz, nx) on
        device, same semantics as ``SeparablePoisson.solve``/``splu.solve`` on the closed-wall operator
        (psi_wall = b_wall + the interior DST x tridiag solution)."""
        nx, nz = self.nx, self.nz
        psi = _cp.empty((nz, nx), dtype=self.dtype)
        psi[0, :] = b[0, :]
        psi[-1, :] = b[-1, :]
        psi[:, 0] = b[:, 0]
        psi[:, -1] = b[:, -1]

        rhs = b[1:-1, 1:-1].copy()
        bx = self.beta[1:-1] * self.ix2
        rhs[:, 0] -= bx * b[1:-1, 0]
        rhs[:, -1] -= bx * b[1:-1, -1]
        rhs[0, :] -= self.cjm[0] * b[0, 1:-1]
        rhs[-1, :] -= self.cjp[-1] * b[-1, 1:-1]

        bhat = _dst1(rhs, _cp)                                  # DST-I along x (device)

        nzi = self._nzi
        if self.fused_thomas:
            # FUSED: the whole forward+back z-march in ONE kernel (one thread per wavenumber mode). Identical
            # fp64 arithmetic in the identical order as the loop below -> round-off-equivalent (no
            # transcendentals; typically bit-identical). Collapses the ~2*nzi tiny launches.
            x = dst_thomas_sweep_device(self._thomas_kernel, bhat, self.inv_pivot, self.cprime, self.cjm)
        else:
            d = _cp.empty_like(bhat)
            d[0] = bhat[0] * self.inv_pivot[0]
            for jj in range(1, nzi):                           # forward sweep (sequential in z)
                d[jj] = (bhat[jj] - self.cjm[jj] * d[jj - 1]) * self.inv_pivot[jj]
            x = _cp.empty_like(bhat)
            x[-1] = d[-1]
            for jj in range(nzi - 2, -1, -1):                 # back-substitution (sequential in z)
                x[jj] = d[jj] - self.cprime[jj] * x[jj + 1]

        psi[1:-1, 1:-1] = _idst1(x, _cp)                      # inverse DST-I along x (device)
        return psi


# --------------------------------------------------------------------------- capture-safe tiled GEMM
_GEMM_TILE = 16
_GEMM_SRC = r"""
extern "C" __global__ void tiled_gemm_{CTYPE}(const {CTYPE}* __restrict__ A,
                                              const {CTYPE}* __restrict__ B,
                                              {CTYPE}* __restrict__ C,
                                              const int M, const int K, const int N) {
    // C(M,N) = A(M,K) @ B(K,N), row-major, shared-memory tiled. Written for the stretched-Poisson
    // transforms because CuPy's cuBLAS binding REFUSES calls during CUDA-graph capture
    // ("calling cuBLAS API during stream capture is currently unsupported", cublas.pyx _setStream) --
    // a RawKernel is capture-safe by construction and keeps ONE deterministic code path (captured and
    // eager runs execute identical arithmetic in identical order).
    const int TILE = %TILE%;
    __shared__ {CTYPE} As[%TILE%][%TILE%];
    __shared__ {CTYPE} Bs[%TILE%][%TILE% + 1];   // +1 pad: kill shared-memory bank conflicts on column reads
    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    {CTYPE} acc = ({CTYPE})0;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        const int ak = t * TILE + threadIdx.x;
        const int bk = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? A[row * K + ak] : ({CTYPE})0;
        Bs[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? B[bk * N + col] : ({CTYPE})0;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
"""


def build_gemm_kernel(dtype):
    """Compile the capture-safe tiled GEMM RawKernel for ``dtype`` (NVRTC at construction/warmup, never
    inside capture). fp64 for the Poisson transforms; the template is dtype-generic."""
    if not _HAVE_CUPY:
        raise RuntimeError("build_gemm_kernel requires CuPy")
    ctype = {"float64": "double", "float32": "float"}[np.dtype(dtype).name]
    src = _GEMM_SRC.replace("{CTYPE}", ctype).replace("%TILE%", str(_GEMM_TILE))
    return _cp.RawKernel(src, f"tiled_gemm_{ctype}")


def gemm_device(kernel, A, B, out):
    """out(M,N) = A(M,K) @ B(K,N) via the tiled RawKernel (row-major C-contiguous fp-typed device
    arrays). Grid/block from the fixed tile; safe inside CUDA-graph capture."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2 and out.shape == (M, N)
    # The kernel indexes raw row-major memory: a strided/F-ordered operand reads as its TRANSPOSE
    # (numpy's eigh returns F-ordered eigenvectors -- exactly the bug this assert exists to catch).
    ok = A.flags.c_contiguous and B.flags.c_contiguous and out.flags.c_contiguous
    assert ok, "gemm_device requires C-contiguous operands"
    grid = ((N + _GEMM_TILE - 1) // _GEMM_TILE, (M + _GEMM_TILE - 1) // _GEMM_TILE)
    kernel(grid, (_GEMM_TILE, _GEMM_TILE), (A, B, out, np.int32(M), np.int32(K), np.int32(N)))
    return out


# ------------------------------------------------------------- GPU STRETCHED-x separable Poisson solver
class GPUStretchedPoisson:
    """Device (CuPy) twin of ``stage_ad_kernels.StretchedSeparablePoisson`` — the eigendecomposition +
    GEMM separable inverter for the closed-wall anelastic Poisson on a STRETCHED (non-uniform) x grid.
    Same ``solve(b) -> psi`` semantics as ``GPUSeparablePoisson`` (device (nz, nx) in/out, psi_wall =
    b_wall), replacing the DST-I pair with two dense GEMM transforms (``rhs @ Vf`` / ``xi @ Vb``) run on
    the capture-safe tiled ``RawKernel`` (NOT cuBLAS -- see the capture note below), and reusing the
    SAME fused z-Thomas RawKernel (its per-mode factors are RHS-independent — only their values differ,
    mu_m -> lam_m).

    ZERO MATH DUPLICATION: __init__ instantiates the CPU ``StretchedSeparablePoisson`` (host fp64 eigh +
    Thomas precompute) and moves its factors to device once — the two solvers are identical by
    construction, so the CPU/GPU parity gate tests only the GEMM/kernel arithmetic, not a re-derivation.

    fp64 ONLY: the stretched x-operator's conditioning is ~1e10 at the production stretch (see the CPU
    class docstring) — fp32 would lose the solution entirely, so an fp32 construction raises instead of
    silently degrading (the production seam builds the Poisson fp64 unconditionally anyway).

    CUDA-graph capture: CuPy's cuBLAS binding REFUSES calls during stream capture ("calling cuBLAS
    API during stream capture is currently unsupported"), so ``@``/cuBLAS must NOT appear in solve() --
    both transforms run on the tiled GEMM RawKernel in eager AND captured modes (one deterministic code
    path; do not "simplify" back to ``@``, it breaks capture). All NVRTC compiles happen HERE at
    construction (warmup), never inside capture — the ``GPUSeparablePoisson`` discipline unchanged."""

    def __init__(self, rho_ref, nx, nz, x_centers, dz, dtype):
        if not _HAVE_CUPY:
            raise RuntimeError("GPUStretchedPoisson requires CuPy (the GPU backend). Install "
                               "requirements-gpu.txt or use the CPU StretchedSeparablePoisson.")
        if dtype != _cp.float64:
            raise ValueError("GPUStretchedPoisson is fp64-only: the stretched x-operator's conditioning "
                             "(~1e10 at the production stretch) makes an fp32 solve meaningless.")
        from .stage_ad_kernels import StretchedSeparablePoisson   # host init: eigh + Thomas factors
        cpu = StretchedSeparablePoisson(rho_ref, nx, nz, x_centers, dz)
        self.nx, self.nz = cpu.nx, cpu.nz
        self.dtype = dtype
        self.beta = _cp.asarray(cpu.beta, dtype)
        self._bxm1 = float(cpu._bxm1)                          # x-wall couplings / beta[j] (python floats)
        self._bxp_last = float(cpu._bxp_last)
        self.cjp = _cp.asarray(cpu.cjp, dtype)
        self.cjm = _cp.asarray(cpu.cjm, dtype)
        self.inv_pivot = _cp.asarray(cpu.inv_pivot, dtype)
        self.cprime = _cp.asarray(cpu.cprime, dtype)
        # np.ascontiguousarray: eigh_tridiagonal returns F-ORDERED eigenvectors (LAPACK), and numpy
        # broadcasting preserves that layout into _Vf/_Vb -- the RawKernel reads raw row-major, so an
        # F-ordered factor would silently transpose (caught by gemm_device's contiguity assert).
        self._Vf = _cp.asarray(np.ascontiguousarray(cpu._Vf), dtype)   # forward GEMM factor (nx-2, nx-2)
        self._Vb = _cp.asarray(np.ascontiguousarray(cpu._Vb), dtype)   # backward GEMM factor
        self._nzi = cpu._nzi
        self.fused_thomas = self._nzi <= _TRIDIAG_MAXNZ
        self._thomas_kernel = build_dst_thomas_kernel(dtype) if self.fused_thomas else None
        # Capture-safe tiled GEMM for the two transforms: CuPy's cuBLAS binding refuses calls during
        # CUDA-graph capture, so ``@`` is off the table inside the captured step. ONE code path (the
        # RawKernel) for eager AND captured -> identical arithmetic, identical order, no rtol seam
        # between the two run modes. NVRTC compile here (construction = warmup), never inside capture.
        self._gemm_kernel = build_gemm_kernel(dtype)

    def solve(self, b):
        """Invert M psi = b on device (b an (nz, nx) fp64 CuPy array). Returns psi (nz, nx) on device,
        same wall semantics as the CPU twin / splu."""
        nx, nz = self.nx, self.nz
        psi = _cp.empty((nz, nx), dtype=self.dtype)
        psi[0, :] = b[0, :]
        psi[-1, :] = b[-1, :]
        psi[:, 0] = b[:, 0]
        psi[:, -1] = b[:, -1]

        rhs = b[1:-1, 1:-1].copy()
        bi = self.beta[1:-1]
        rhs[:, 0] -= (bi * self._bxm1) * b[1:-1, 0]            # i=1 couples to the i=0 axis wall
        rhs[:, -1] -= (bi * self._bxp_last) * b[1:-1, -1]      # i=nx-2 couples to the outer wall
        rhs[0, :] -= self.cjm[0] * b[0, 1:-1]
        rhs[-1, :] -= self.cjp[-1] * b[-1, 1:-1]

        bhat = gemm_device(self._gemm_kernel, rhs, self._Vf,
                           _cp.empty_like(rhs))                # forward GEMM (capture-safe RawKernel)

        nzi = self._nzi
        if self.fused_thomas:
            x = dst_thomas_sweep_device(self._thomas_kernel, bhat, self.inv_pivot, self.cprime, self.cjm)
        else:
            d = _cp.empty_like(bhat)
            d[0] = bhat[0] * self.inv_pivot[0]
            for jj in range(1, nzi):
                d[jj] = (bhat[jj] - self.cjm[jj] * d[jj - 1]) * self.inv_pivot[jj]
            x = _cp.empty_like(bhat)
            x[-1] = d[-1]
            for jj in range(nzi - 2, -1, -1):
                x[jj] = d[jj] - self.cprime[jj] * x[jj + 1]

        # backward GEMM into a CONTIGUOUS temp (the RawKernel writes raw row-major; psi[1:-1,1:-1] is a
        # strided view), then one strided device copy into the interior.
        psi[1:-1, 1:-1] = gemm_device(self._gemm_kernel, x, self._Vb, _cp.empty_like(x))
        return psi


# --------------------------------------------------------------------------- one-thread-per-column Thomas
# PERF NOTE (PR #198 review item 3, DEFERRED): the ``cp``/``dp`` thread-local scratch is sized to
# ``_TRIDIAG_MAXNZ`` (512); at the production nz<=256 that over-allocates 2x of local memory per thread,
# which can cap occupancy. This kernel is the VERTICAL IMPLICIT DIFFUSION (omega) solve -- the DYNAMICS
# limb, the model's NEXT bottleneck (the convadjust limb is what PR #198 just optimized), and it runs in
# FP64 here (the omega tridiag is fp64; see ``_implicit_vertical_diffusion``), so the scratch is double[512]
# = 4 KB/thread regardless of the fp32 thermo path. Tightening it is left to the dynamics-limb pass, NOT
# this review PR: a hard 512->256 leaves ZERO headroom for the live nx4096/nz256 grid, and proper per-nz
# scratch templating means per-nz kernel COMPILATION, which belongs with the dynamics-limb optimization.
_TRIDIAG_MAXNZ = 512


def _tridiag_src(real):
    """Source for the one-thread-per-column Thomas kernel in C-precision ``real`` ('double' or 'float').
    Column-innermost (nz, nx) device layout: element (k, i) at k*nx + i (so the warp coalesces across the
    nx columns). The internal forward+back nz march is a SINGLE launch. Mirrors ``_solve_tridiag_columns``
    exactly (cp[0]=sup[0]/diag[0], dp[0]=rhs[0]/diag[0]; m=diag[j]-sub[j]*cp[j-1], cp[j]=sup[j]/m,
    dp[j]=(rhs[j]-sub[j]*dp[j-1])/m; back: x[-1]=dp[-1], x[j]=dp[j]-cp[j]*x[j+1]). cp/dp are thread-local
    scratch sized to a MAX nz (the systems are short, nz<=256)."""
    return f"""
extern "C" __global__
void tridiag_cols(const int nx, const int nz,
                  const {real}* __restrict__ sub, const {real}* __restrict__ diag,
                  const {real}* __restrict__ sup, const {real}* __restrict__ rhs,
                  {real}* __restrict__ x) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // column
    if (i >= nx) return;
    {real} cp[{_TRIDIAG_MAXNZ}];
    {real} dp[{_TRIDIAG_MAXNZ}];
    // layout (nz, nx): element (k,i) at k*nx + i
    {real} d0 = diag[0*nx + i];
    cp[0] = sup[0*nx + i] / d0;
    dp[0] = rhs[0*nx + i] / d0;
    for (int j = 1; j < nz; ++j) {{
        {real} m = diag[j*nx + i] - sub[j*nx + i] * cp[j-1];
        cp[j] = sup[j*nx + i] / m;
        dp[j] = (rhs[j*nx + i] - sub[j*nx + i] * dp[j-1]) / m;
    }}
    x[(nz-1)*nx + i] = dp[nz-1];
    for (int j = nz - 2; j >= 0; --j) {{
        x[j*nx + i] = dp[j] - cp[j] * x[(j+1)*nx + i];
    }}
}}
"""


# Cache compiled modules per C-precision so a core builds the kernel once.
_TRIDIAG_KERNEL_CACHE: dict = {}


def build_tridiag_cols_kernel(dtype):
    """Return the compiled one-thread-per-column Thomas ``tridiag_cols`` RawKernel for the given device
    dtype (``cp.float64`` or ``cp.float32``). Cached per C-precision. Requires CuPy.

    The systems must have nz <= ``_TRIDIAG_MAXNZ`` (the thread-local scratch size)."""
    if not _HAVE_CUPY:
        raise RuntimeError("build_tridiag_cols_kernel requires CuPy (the GPU backend).")
    real = "double" if dtype == np.float64 or dtype == _cp.float64 else "float"
    fn = _TRIDIAG_KERNEL_CACHE.get(real)
    if fn is None:
        fn = _cp.RawModule(code=_tridiag_src(real)).get_function("tridiag_cols")
        _TRIDIAG_KERNEL_CACHE[real] = fn
    return fn


def solve_tridiag_columns_device(kernel, sub, diag, sup, rhs, threads_per_block=128):
    """Run the Thomas RawKernel over the per-column tridiagonals, all (nz, nx) device arrays of one dtype.
    Returns x (nz, nx) device. The four band arrays must be C-contiguous (nz, nx) — the kernel indexes
    k*nx+i. ``kernel`` is from ``build_tridiag_cols_kernel`` for the SAME dtype."""
    nz, nx = diag.shape
    if nz > _TRIDIAG_MAXNZ:
        raise ValueError(f"tridiag kernel scratch sized for nz<={_TRIDIAG_MAXNZ}, got nz={nz}")
    sub = _cp.ascontiguousarray(sub)
    diag = _cp.ascontiguousarray(diag)
    sup = _cp.ascontiguousarray(sup)
    rhs = _cp.ascontiguousarray(rhs)
    x = _cp.empty((nz, nx), dtype=diag.dtype)
    tpb = int(threads_per_block)
    bpg = (nx + tpb - 1) // tpb
    kernel((bpg,), (tpb,), (np.int32(nx), np.int32(nz), sub, diag, sup, rhs, x))
    return x


# ===================================================================== DST-Poisson z-Thomas (FUSED sweep)
# The separable DST Poisson's z-sweep (``GPUSeparablePoisson.solve``) was a Python forward+back ``for jj``
# loop over the nz-2 interior levels -- ~2*(nz-2) tiny (nx-2,) kernel launches per solve, and the solve runs
# TWICE per dynamics substep (psi-from-omega + the operator-KE-removal). At nz144 that is ~568 sequential
# launches the CUDA graph cannot collapse (graphs amortize the PYTHON dispatch, NOT each kernel's on-device
# launch latency) -- the same launch-bound regime convadjust was (PR #198). Profiling the captured ~20 ms
# step: the z-Thomas loop is ~1.23 ms/call x2 = ~2.5 ms/step (60% of the 2.0 ms solve; the cuFFT DST/iDST is
# the efficient 37% remainder). The wavenumber MODES are independent (the DST diagonalizes x), so one thread
# per mode runs the whole sequential z-march in-thread, collapsing the ~568 launches into ONE kernel.
#
# This REUSES the solver's precomputed factorization (inv_pivot/cprime) -- the SAME "factor once, solve many"
# the CPU SeparablePoisson relies on -- and does the IDENTICAL fp64 arithmetic in the IDENTICAL order as the
# Python loop, so it is round-off-equivalent (no transcendentals to reassociate -> typically bit-identical).
# fp64 only: the solve is deliberately fp64 (the ill-conditioned operator amplifies kappa(op)*eps); the
# recurrence is trivial arithmetic, so fp64 costs nothing on GA102 here -- the win is the launch collapse.


def _dst_thomas_src(real):
    """Source for the one-thread-per-MODE fused DST-Poisson z-Thomas sweep in C-precision ``real``.
    Operates on the post-DST RHS ``bhat`` (nzi, nxi) = (nz-2, nx-2), C-contiguous -> element (jj, m) at
    jj*nxi + m, so the warp coalesces across the nxi modes for a fixed level jj. One thread per mode m runs
    the precomputed-factor Thomas recurrence (forward then back) over the nzi levels, writing ``x`` (nzi,
    nxi). Mirrors ``GPUSeparablePoisson.solve`` EXACTLY: d[0]=bhat[0]*inv_pivot[0];
    d[jj]=(bhat[jj]-cjm[jj]*d[jj-1])*inv_pivot[jj]; x[nzi-1]=d[nzi-1]; x[jj]=d[jj]-cprime[jj]*x[jj+1].
    ``inv_pivot``/``cprime`` are (nzi, nxi); ``cjm`` is (nzi,) (shared across modes). ``d`` is thread-local
    scratch sized to ``_TRIDIAG_MAXNZ`` (the systems are short, nzi<=nz<=256)."""
    return f"""
extern "C" __global__
void dst_thomas(const int nzi, const int nxi,
                const {real}* __restrict__ bhat, const {real}* __restrict__ inv_pivot,
                const {real}* __restrict__ cprime, const {real}* __restrict__ cjm,
                {real}* __restrict__ x) {{
    int m = blockIdx.x * blockDim.x + threadIdx.x;   // wavenumber mode (DST column)
    if (m >= nxi) return;
    {real} d[{_TRIDIAG_MAXNZ}];
    d[0] = bhat[0*nxi + m] * inv_pivot[0*nxi + m];
    for (int jj = 1; jj < nzi; ++jj) {{               // forward sweep (sequential in z)
        d[jj] = (bhat[jj*nxi + m] - cjm[jj] * d[jj-1]) * inv_pivot[jj*nxi + m];
    }}
    x[(nzi-1)*nxi + m] = d[nzi-1];
    for (int jj = nzi - 2; jj >= 0; --jj) {{          // back-substitution (sequential in z)
        x[jj*nxi + m] = d[jj] - cprime[jj*nxi + m] * x[(jj+1)*nxi + m];
    }}
}}
"""


_DST_THOMAS_KERNEL_CACHE: dict = {}


def build_dst_thomas_kernel(dtype):
    """Return the compiled one-thread-per-mode ``dst_thomas`` RawKernel for the given device dtype
    (typically ``cp.float64`` -- the DST Poisson is fp64). Cached per C-precision. Requires CuPy; systems
    must have nzi = nz-2 <= ``_TRIDIAG_MAXNZ`` (the thread-local scratch size)."""
    if not _HAVE_CUPY:
        raise RuntimeError("build_dst_thomas_kernel requires CuPy (the GPU backend).")
    real = "double" if dtype == np.float64 or dtype == _cp.float64 else "float"
    fn = _DST_THOMAS_KERNEL_CACHE.get(real)
    if fn is None:
        fn = _cp.RawModule(code=_dst_thomas_src(real)).get_function("dst_thomas")
        _DST_THOMAS_KERNEL_CACHE[real] = fn
    return fn


def dst_thomas_sweep_device(kernel, bhat, inv_pivot, cprime, cjm, threads_per_block=128):
    """Run the fused DST-Poisson z-Thomas sweep over the wavenumber modes. ``bhat``/``inv_pivot``/``cprime``
    are (nzi, nxi) device arrays of one dtype, ``cjm`` is (nzi,); all must be C-contiguous and match the
    ``kernel`` dtype. Returns ``x`` (nzi, nxi) device (the solved interior, pre-iDST)."""
    nzi, nxi = bhat.shape
    if nzi > _TRIDIAG_MAXNZ:
        raise ValueError(f"dst_thomas kernel scratch sized for nz-2<={_TRIDIAG_MAXNZ}, got nzi={nzi}")
    bhat = _cp.ascontiguousarray(bhat)
    inv_pivot = _cp.ascontiguousarray(inv_pivot)
    cprime = _cp.ascontiguousarray(cprime)
    cjm = _cp.ascontiguousarray(cjm)
    x = _cp.empty((nzi, nxi), dtype=bhat.dtype)
    tpb = int(threads_per_block)
    bpg = (nxi + tpb - 1) // tpb
    kernel((bpg,), (tpb,), (np.int32(nzi), np.int32(nxi), bhat, inv_pivot, cprime, cjm, x))
    return x


# ============================================================================= convective adjustment
# A one-thread-per-column fused RawKernel for ``moist_convective_adjust`` (the AD-core's GAUSS-SEIDEL moist
# convective adjustment), the ~84%-of-the-captured-step bottleneck (``archive/_gpu_phase_e_profile_RESULTS.md``).
#
# The numpy/CuPy ``moist_convective_adjust`` is vectorized ACROSS columns but its k-loop is SEQUENTIAL in k
# (pair k-1 reads what pair k just wrote -> Gauss-Seidel). On the GPU that means ``max_sweeps``x``(nz-1)`` =
# (4x143 at prod) ~572 SEQUENTIAL iterations, each issuing ~6-10 tiny (nx,) CuPy kernels -> ~5700 launches
# that the CUDA graph cannot collapse (it amortizes the PYTHON dispatch, not the on-device launch latency of
# each tiny kernel). The nx COLUMNS ARE INDEPENDENT, so one thread per column collapses all ~5700 launches
# into ONE kernel -- the same restructure the numba ``_conv_adjust_numba`` (prange over columns) already does
# on the CPU. This kernel MIRRORS that numba kernel exactly (same predicate ``Tk1 < (Tk - Gdz) - 1e-9``, same
# enthalpy-/water-conserving mix, same ``(mk1*Gamma)*dz`` grouping, same Tetens esat coefficients), so it is
# round-off-equivalent to the reference (NOT bit-identical -- the leaf transcendentals reassociate); the ship
# gate is the 60-step climate-match. It supports the two production closures (q_homogenize / rh_detrain).
#
# PRECISION: templated on ``real`` ('float' or 'double'). On the captured fp32 path the convadjust inputs are
# fp32 (T_atm/q/dm are fp32 working copies and th._F coerces the leaf math to fp32) -- so the 'float' kernel
# (with ``expf``) matches that path's dominant precision; any residual vs the current path's few fp64
# intermediates (dz, p_mid) is at the fp32 floor the climate-match validates. The 'double' kernel is the
# numba-fp64-equivalent path (kept for a fp64 device run / parity probe).
_CONVADJ_MAXNZ = 512


def _convadj_src(real):
    """Source for the one-thread-per-column moist-convective-adjustment kernel in C-precision ``real``.
    Row-major (nx, nz) layout (the AD-core's native ``T_atm`` shape): column i's data is the contiguous
    nz block at i*nz. One thread per column runs the full sequential sweep loop IN PLACE on T_atm and q.
    ``RHt`` (nz,), ``dm`` (nz,), ``p_mid`` (nz,), ``z`` (nx, nz) are inputs; ``detrain`` selects the closure
    (1 = rh_detrain, 0 = q_homogenize). Physical constants are compile-time literals (injected from the
    ``constants`` module by the builder, so they cannot drift)."""
    e = "expf" if real == "float" else "exp"
    suf = "f" if real == "float" else ""
    return f"""
extern "C" __global__
void convadj_cols(const int nx, const int nz, const int max_sweeps, const int detrain,
                  {real}* __restrict__ T_atm, {real}* __restrict__ q,
                  const {real}* __restrict__ dm, const {real}* __restrict__ z,
                  const {real}* __restrict__ p_mid, const {real}* __restrict__ RHt) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // column
    if (i >= nx) return;
    const {real} T_FREEZE = {{C_T_FREEZE}}{suf};
    const {real} EPS_MW   = {{C_EPS_MW}}{suf};
    const {real} L_VAP    = {{C_L_VAP}}{suf};
    const {real} R_DRY    = {{C_R_DRY}}{suf};
    const {real} CP_AIR   = {{C_CP_AIR}}{suf};
    const {real} G_ACCEL  = {{C_G_ACCEL}}{suf};
    {real}* Ti = T_atm + (long)i * nz;               // this column's contiguous nz block
    {real}* qi = q + (long)i * nz;
    const {real}* zi = z + (long)i * nz;
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {{
        int changed = 0;
        for (int k = nz - 1; k >= 1; --k) {{          // pair (k below/higher-p, k-1 above); Gauss-Seidel
            {real} Tk = Ti[k];
            {real} Tk1 = Ti[k-1];
            {real} pk = p_mid[k];
            // --- moist_adiabat_lapse_rate(Tk, pk): qsat then the Gamma_m formula ---
            {real} Tc = Tk - T_FREEZE;
            {real} esat = (Tc >= 0.0{suf})
                ? 611.2{suf} * {e}(17.67{suf} * Tc / (Tc + 243.5{suf}))
                : 611.2{suf} * {e}(21.87{suf} * Tc / (Tc + 265.5{suf}));
            {real} cap = 0.999{suf} * pk;
            {real} ee = (esat < cap) ? esat : cap;
            {real} qs = EPS_MW * ee / (pk - (1.0{suf} - EPS_MW) * ee);
            {real} num = 1.0{suf} + L_VAP * qs / (R_DRY * Tk);
            {real} den = 1.0{suf} + L_VAP * L_VAP * qs * EPS_MW / (CP_AIR * R_DRY * Tk * Tk);
            {real} Gamma = (G_ACCEL / CP_AIR) * num / den;
            {real} dz = zi[k-1] - zi[k];
            {real} Gdz = Gamma * dz;
            // NB (PR #198 review item 2): the -1e-9 instability-test tolerance is a NO-OP on the 'float'
            // (fp32) kernel -- at ~300 K the fp32 ULP is 2^(8-23) ~= 3e-5 K >> 1e-9, so (Tk - Gdz) - 1e-9f
            // rounds back to (Tk - Gdz). It is KEPT (not stripped) so the source MIRRORS the numpy/numba
            // reference's predicate verbatim (the ship gate is the climate-match, which validated this kernel
            // AS-IS; a scaled fp32 tolerance would CHANGE which pairs convect, i.e. change results). On the
            // 'double' kernel the constant is meaningful (fp64 ULP ~6e-14 K at 300 K), so this is not dead code.
            if (Tk1 < (Tk - Gdz) - 1e-9{suf}) {{
                {real} mk = dm[k];
                {real} mk1 = dm[k-1];
                {real} Ctot = mk * Tk + mk1 * Tk1;
                {real} Tk_new = (Ctot + mk1 * Gamma * dz) / (mk + mk1);   // (mk1*Gamma)*dz grouping
                {real} Tk1_new = Tk_new - Gdz;
                Ti[k] = Tk_new;
                Ti[k-1] = Tk1_new;
                {real} qk = qi[k];
                {real} qk1 = qi[k-1];
                if (detrain == 0) {{                  // q_homogenize
                    {real} q_mix = (mk * qk + mk1 * qk1) / (mk + mk1);
                    qi[k] = q_mix;
                    qi[k-1] = q_mix;
                }} else {{                             // rh_detrain (production default)
                    {real} W = mk * qk + mk1 * qk1;
                    // qsat at the NEW temperatures (below k, above k-1)
                    {real} Tc_k  = Tk_new - T_FREEZE;
                    {real} es_k  = (Tc_k >= 0.0{suf})
                        ? 611.2{suf} * {e}(17.67{suf} * Tc_k / (Tc_k + 243.5{suf}))
                        : 611.2{suf} * {e}(21.87{suf} * Tc_k / (Tc_k + 265.5{suf}));
                    {real} cap_k = 0.999{suf} * pk;
                    {real} ee_k  = (es_k < cap_k) ? es_k : cap_k;
                    {real} qs_k  = EPS_MW * ee_k / (pk - (1.0{suf} - EPS_MW) * ee_k);
                    {real} pk1 = p_mid[k-1];
                    {real} Tc_k1 = Tk1_new - T_FREEZE;
                    {real} es_k1 = (Tc_k1 >= 0.0{suf})
                        ? 611.2{suf} * {e}(17.67{suf} * Tc_k1 / (Tc_k1 + 243.5{suf}))
                        : 611.2{suf} * {e}(21.87{suf} * Tc_k1 / (Tc_k1 + 265.5{suf}));
                    {real} cap_k1 = 0.999{suf} * pk1;
                    {real} ee_k1 = (es_k1 < cap_k1) ? es_k1 : cap_k1;
                    {real} qs_k1 = EPS_MW * ee_k1 / (pk1 - (1.0{suf} - EPS_MW) * ee_k1);
                    {real} gk  = RHt[k]   * qs_k;
                    {real} gk1 = RHt[k-1] * qs_k1;
                    {real} denom = mk * gk + mk1 * gk1;
                    if (denom > 0.0{suf}) {{
                        {real} A = W / denom;
                        qi[k] = A * gk;
                        qi[k-1] = A * gk1;
                    }} else {{
                        {real} homog = W / (mk + mk1);
                        qi[k] = homog;
                        qi[k-1] = homog;
                    }}
                }}
                changed = 1;
            }}
        }}
        if (changed == 0) break;                       // per-column early-out (value-neutral vs fixed-topology)
    }}
}}
"""


_CONVADJ_KERNEL_CACHE: dict = {}


def build_convadj_cols_kernel(dtype):
    """Return the compiled one-thread-per-column ``convadj_cols`` RawKernel for the given device dtype
    (``cp.float64`` or ``cp.float32``). Cached per C-precision. Constants are injected from ``constants``
    at build time. Requires CuPy; systems must have nz <= ``_CONVADJ_MAXNZ`` (no scratch; this is a soft
    cap mirroring the tridiag kernel's nz bound)."""
    if not _HAVE_CUPY:
        raise RuntimeError("build_convadj_cols_kernel requires CuPy (the GPU backend).")
    real = "double" if dtype == np.float64 or dtype == _cp.float64 else "float"
    fn = _CONVADJ_KERNEL_CACHE.get(real)
    if fn is None:
        from . import constants as C
        src = _convadj_src(real)
        src = src.replace("{C_T_FREEZE}", repr(float(C.T_FREEZE))) \
                 .replace("{C_EPS_MW}", repr(float(C.EPS_MW))) \
                 .replace("{C_L_VAP}", repr(float(C.L_VAP))) \
                 .replace("{C_R_DRY}", repr(float(C.R_DRY))) \
                 .replace("{C_CP_AIR}", repr(float(C.CP_AIR))) \
                 .replace("{C_G_ACCEL}", repr(float(C.G_ACCEL)))
        fn = _cp.RawModule(code=src).get_function("convadj_cols")
        _CONVADJ_KERNEL_CACHE[real] = fn
    return fn


def convadj_columns_device(kernel, T_atm, q, dm, z, p_mid, RHt, max_sweeps, detrain,
                           threads_per_block=128):
    """Run the fused convective-adjustment RawKernel IN PLACE on (nx, nz) device ``T_atm`` and ``q``.
    All float arrays must be the SAME dtype as ``kernel`` was built for and C-contiguous (the kernel
    indexes i*nz + k). ``RHt`` (nz,): the Manabe RH target (rh_detrain only; pass any (nz,) for homogenize).
    ``detrain``: 1 = rh_detrain, 0 = q_homogenize. Mutates T_atm and q in place (returns nothing)."""
    nx, nz = T_atm.shape
    if nz > _CONVADJ_MAXNZ:
        raise ValueError(f"convadj kernel sized for nz<={_CONVADJ_MAXNZ}, got nz={nz}")
    for _n, _a in (("T_atm", T_atm), ("q", q)):
        if not _a.flags.c_contiguous:
            # In-place contract: a silent ascontiguousarray copy would drop the update (see micro runner).
            raise ValueError(f"convadj kernel requires C-contiguous {_n} (in-place contract); "
                             f"got strides {_a.strides} for shape {_a.shape}")
    tpb = int(threads_per_block)
    bpg = (nx + tpb - 1) // tpb
    kernel((bpg,), (tpb,), (np.int32(nx), np.int32(nz), np.int32(max_sweeps), np.int32(detrain),
                            T_atm, q, dm, z, p_mid, RHt))


# ========================================================= converged convective adjustment (SEGMENT MIXER)
# A one-thread-per-column fused RawKernel for ``stage_ad_column.moist_convective_segment_mix`` (ROADMAP
# PHY-17 step 4): the CONVERGED moist convective adjustment, reached by mixing whole unstable SEGMENTS
# rather than by relaxing pair by pair.
#
# WHY IT EXISTS. ``convadj_cols`` above runs the shipped Gauss-Seidel sweep, and production caps it at
# ``conv_max_sweeps`` = 4. A pairwise sweep moves information one layer pair per sweep, so the sweeps needed
# grow with the SQUARE of the unstable layer's depth in levels; on the canonical 144-level grid the cap is
# nowhere near the fixed point (one call on run17's map-band columns leaves 35.8 / 29.7 / 18.1 / 3.6 / 0.0 K
# of a 43.1 K potential-temperature deficit after 4 / 16 / 64 / 256 / 1,024 sweeps -- ``flatworld/
# OPEN_ISSUES.md``, PHY-17). Raising the cap is LINEAR in the cap on this kernel (the sweep loop was
# measured at 0.70 ms of a 20.3 ms captured step at cap 4, ``archive/_gpu_phase_e_profile_RESULTS.md``), so
# a converged sweep would cost ~180 ms. This kernel reaches the same fixed point in a bounded number of
# whole-column passes.
#
# It MIRRORS the numpy ``moist_convective_segment_mix`` op-for-op -- the same pairwise predicate
# ``T_{k-1} < T_k - Gamma*dz - 1e-9``, the same monotonically-grown active-pair set, the same closed-form
# enthalpy-conserving block solve, the same Gamma fixed-point iteration, the same single ``A*`` water
# normalization over a block, the same Tetens esat coefficients as ``convadj_cols`` -- so it is
# round-off-equivalent, NOT bit-identical (the leaf transcendentals reassociate, and the numpy body sums
# one intermediate in the opposite index order; see the launcher's note). Both production closures
# (rh_detrain / q_homogenize) are covered.
#
# PRECISION: templated on ``real`` ('float'/'double'), exactly as ``convadj_cols``, with the constants
# injected as compile-time literals from ``constants`` so they cannot drift.
#
# The per-column ``act`` scratch is thread-local, the ``_micro_src`` ``src_ice`` pattern (one byte per
# level, so 512 B per thread against that kernel's 4 KB of ``real`` scratch).
_SEGMIX_MAXNZ = 512


def _segmix_src(real):
    """Source for the one-thread-per-column CONVERGED convective adjustment in C-precision ``real``.
    Row-major (nx, nz) layout (the AD-core's native ``T_atm`` shape): column i's data is the contiguous
    nz block at i*nz, TOP(0) -> SURFACE(nz-1). One thread per column runs the whole merge-and-rescan loop
    IN PLACE on T_atm and q. ``RHt`` (nz,), ``dm`` (nz,), ``p_mid`` (nz,), ``z`` (nx, nz) are inputs;
    ``detrain`` selects the closure (1 = rh_detrain, 0 = q_homogenize)."""
    e = "expf" if real == "float" else "exp"
    suf = "f" if real == "float" else ""
    # FLT_EPSILON / DBL_EPSILON as literals (no <float.h> in an NVRTC translation unit), and the
    # settled bar as a literal too, so the fp32 rendering keeps every constant single-precision.
    eps = "1.1920929e-07" if real == "float" else "2.220446049250313e-16"
    from .stage_ad_column import _SEGMIX_SETTLED_ULP          # ONE home for the bar (the numpy body's)
    ulp = repr(float(_SEGMIX_SETTLED_ULP))
    return f"""
__device__ __forceinline__ {real} _sm_esat({real} T) {{
    const {real} T_FREEZE = {{C_T_FREEZE}}{suf};
    {real} Tc = T - T_FREEZE;
    if (Tc >= 0.0{suf}) return 611.2{suf} * {e}(17.67{suf} * Tc / (Tc + 243.5{suf}));
    return 611.2{suf} * {e}(21.87{suf} * Tc / (Tc + 265.5{suf}));
}}
__device__ __forceinline__ {real} _sm_qsat({real} T, {real} p) {{
    const {real} EPS_MW = {{C_EPS_MW}}{suf};
    {real} ee = _sm_esat(T);
    {real} cap = 0.999{suf} * p;
    if (ee > cap) ee = cap;
    return EPS_MW * ee / (p - (1.0{suf} - EPS_MW) * ee);
}}
__device__ __forceinline__ {real} _sm_gamma({real} T, {real} p) {{
    // thermo.moist_adiabat_lapse_rate [K/m]; same grouping as _convadj_src's inline form
    const {real} EPS_MW = {{C_EPS_MW}}{suf};
    const {real} L_VAP  = {{C_L_VAP}}{suf};
    const {real} R_DRY  = {{C_R_DRY}}{suf};
    const {real} CP_AIR = {{C_CP_AIR}}{suf};
    const {real} G_ACCEL = {{C_G_ACCEL}}{suf};
    {real} qs = _sm_qsat(T, p);
    {real} num = 1.0{suf} + L_VAP * qs / (R_DRY * T);
    {real} den = 1.0{suf} + L_VAP * L_VAP * qs * EPS_MW / (CP_AIR * R_DRY * T * T);
    return (G_ACCEL / CP_AIR) * num / den;
}}

// Mix ONE block (layers lo..hi, lo above hi) to the moist adiabat at fixed block enthalpy, then
// redistribute its water. Enthalpy is conserved for ANY Gammas, so every iterate conserves it.
// Returns 1 if the LAST Gamma iterate still moved the profile by more than {ulp} units
// in the last place (stopping rule (ii): "the profile solve has not settled"), else 0.
__device__ int _sm_mix({real}* Ti, {real}* qi, const {real}* zi, const {real}* dm,
                       const {real}* p_mid, const {real}* RHt,
                       const int lo, const int hi, const int gamma_iters, const int detrain) {{
    const {real} SETTLED = {ulp}{suf} * {eps}{suf};
    int moving = 0;
    {real} M = 0.0{suf}, H = 0.0{suf}, W = 0.0{suf};
    for (int i = lo; i <= hi; ++i) {{                  // lo->hi: the numpy reduceat's own summation order
        M += dm[i];
        H += dm[i] * Ti[i];
        W += dm[i] * qi[i];
    }}
    for (int it = 0; it < gamma_iters; ++it) {{
        {real} d = 0.0{suf}, S = 0.0{suf};
        if (it < gamma_iters - 1) {{
            // ONE Gamma sweep per iterate (PR #830). The two loops below the `else` evaluate Gamma at
            // the SAME Ti[i] -- the second reads T_old BEFORE its own write, which is exactly the value
            // the first used -- so the offsets d_i computed on the way down can simply be STASHED in
            // Ti[i] and read back, instead of being rebuilt from a second Gamma evaluation. Bitwise
            // identical: same d sequence, same S, same T_bot + d. The LAST iterate cannot do this --
            // stopping rule (ii) needs T_old to compare against T_nu -- so it keeps the two-loop form
            // below. Do not "simplify" the two branches back together.
            for (int i = hi; i >= lo; --i) {{
                S += dm[i] * d;
                {real} T_old = Ti[i];
                Ti[i] = d;                             // stash the offset in place of the temperature
                if (i > lo) d -= _sm_gamma(T_old, p_mid[i]) * (zi[i-1] - zi[i]);
            }}
            {real} T_bot = (H - S) / M;                // enthalpy-conserving in closed form
            for (int i = hi; i >= lo; --i) Ti[i] = T_bot + Ti[i];
        }} else {{
            for (int i = hi; i >= lo; --i) {{          // offsets d_i, built upward from the block bottom
                S += dm[i] * d;
                if (i > lo) d -= _sm_gamma(Ti[i], p_mid[i]) * (zi[i-1] - zi[i]);
            }}
            {real} T_bot = (H - S) / M;                // enthalpy-conserving in closed form
            d = 0.0{suf};
            for (int i = hi; i >= lo; --i) {{
                {real} T_old = Ti[i];                  // read BEFORE the write: Gamma at the same T the
                {real} T_nu = T_bot + d;               //   S-accumulation above used
                {real} dif = T_nu - T_old;             // rule (ii): did the LAST iterate still move it?
                if (dif < 0.0{suf}) dif = -dif;
                {real} mag = T_nu < 0.0{suf} ? -T_nu : T_nu;
                if (dif > SETTLED * mag) moving = 1;
                Ti[i] = T_nu;
                if (i > lo) d -= _sm_gamma(T_old, p_mid[i]) * (zi[i-1] - zi[i]);
            }}
        }}
    }}
    if (detrain != 0) {{                                // rh_detrain: ONE A* over the whole block
        // ONE qsat per layer (PR #830): the block's water W is already summed, so the denominator loop
        // can PARK its qsat in the q slot and the assignment loop read it back instead of evaluating it
        // a second time. Bitwise identical -- the raw qsat is stashed (not RHt*qsat), so both
        // expressions keep their original (dm*RHt)*qs and (A*RHt)*qs groupings.
        {real} denom = 0.0{suf};
        for (int i = lo; i <= hi; ++i) {{
            {real} qs = _sm_qsat(Ti[i], p_mid[i]);
            qi[i] = qs;
            denom += dm[i] * RHt[i] * qs;
        }}
        if (denom > 0.0{suf}) {{
            {real} A = W / denom;
            for (int i = lo; i <= hi; ++i) qi[i] = A * RHt[i] * qi[i];
        }} else {{                                      // every qsat underflowed (fp32: T <= ~54 K): homogenize
            {real} homog = W / M;
            for (int i = lo; i <= hi; ++i) qi[i] = homog;
        }}
    }} else {{                                          // q_homogenize == the g_j == 1 special case
        {real} homog = W / M;
        for (int i = lo; i <= hi; ++i) qi[i] = homog;
    }}
    return moving;
}}

extern "C" __global__
void segmix_cols(const int nx, const int nz, const int max_passes, const int gamma_iters,
                 const int detrain,
                 {real}* __restrict__ T_atm, {real}* __restrict__ q,
                 const {real}* __restrict__ dm, const {real}* __restrict__ z,
                 const {real}* __restrict__ p_mid, const {real}* __restrict__ RHt,
                 int* __restrict__ status) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // column
    if (i >= nx) return;
    {real}* Ti = T_atm + (long)i * nz;               // this column's contiguous nz block
    {real}* qi = q + (long)i * nz;
    const {real}* zi = z + (long)i * nz;
    unsigned char act[{_SEGMIX_MAXNZ}];              // act[k]: pair k (layers k-1, k) is in the mixed set
    for (int k = 0; k < nz; ++k) act[k] = 0;
    int nact_prev = -1, stall = 0, used = 0, done = 0;
    for (int pass = 0; pass < max_passes; ++pass) {{
        // ---- SCAN: the shipped sweep's predicate, on the whole column, BEFORE any mixing ----
        int work = 0;
        for (int k = nz - 1; k >= 1; --k) {{
            {real} Tk = Ti[k];
            {real} Gdz = _sm_gamma(Tk, p_mid[k]) * (zi[k-1] - zi[k]);
            // NB (as in convadj_cols): the -1e-9 K tolerance is a NO-OP on the 'float' kernel -- at ~300 K
            // the fp32 ULP is ~3e-5 K >> 1e-9 -- and is KEPT so the source mirrors the numpy reference's
            // predicate verbatim. On the 'double' kernel it is meaningful (fp64 ULP ~6e-14 K at 300 K).
            if (Ti[k-1] < (Tk - Gdz) - 1e-9{suf}) {{
                act[k] = 1;                          // MONOTONE: a pair, once active, stays active
                work = 1;
            }}
        }}
        // ---- STOPPING RULE, per column: (i) no pair unstable -- the sweep's OWN rule -- or (ii) the
        // active set has not grown for two consecutive passes, so the block partition is final. (ii) is
        // what makes THIS kernel terminate in 'float': the -1e-9 K tolerance above is a no-op at fp32
        // ULP ~3e-5 K, so under (i) alone every column runs max_passes with the residual stuck at the
        // ULP (measured on run17 columns in a host binary32 transliteration; the active-pair count was
        // identical to fp64 pass for pass, so (ii) exits at the SAME partition). Value-neutral either
        // way: a pass with nothing left to mix leaves the column untouched, and the launch count is 1
        // regardless, so CUDA-graph capture sees a constant op chain.
        int nact = 0, nblk = 0;                              // nblk = maximal runs of active pairs
        for (int k = 1; k < nz; ++k) {{
            nact += (int)act[k];
            if (act[k] != 0 && act[k-1] == 0) ++nblk;         // act[0] is never set (pair 0 is a dummy)
        }}
        if (nact > nact_prev) stall = 0; else stall += 1;
        nact_prev = nact;
        if (work == 0) {{ done = 1; used = pass; break; }}     // (i) the sweep's own rule
        used = pass + 1;
        // ---- MIX every maximal run of active pairs (a run k = j..b spans layers j-1..b), ITERATED BY
        // BLOCK ORDINAL, not by position (PR #830). The blocks mixed are the same ones, in the same
        // order (bottom-most first), with the same arguments -- this is a loop-nest change, not an
        // algorithm change. It matters because ONE THREAD PER COLUMN means a warp executes, masked,
        // every lane's _sm_mix call: with the call sited at the block's own k, neighbouring columns'
        // block boundaries do not line up and the warp pays, at EVERY k, the largest block any lane
        // starts there. Counted on the canonical columns (phy17_mixer_cost_model.py): 6.0x the useful
        // work cold, 16x on a state the mixer has recently mixed. Issued at block ORDINAL the warp pays
        // only the largest b-th block among its lanes -- 1.7x -- because every column has a handful of
        // blocks and they align by count, not by height. The cursor walk below is the same single
        // descent over act[] the position loop did, just suspended between blocks.
        int moving = 0, kcur = nz - 1, bot = -1;
        for (int b = 0; b < nblk; ++b) {{
            int lo = -1, hi = -1;
            while (kcur >= 1) {{
                if (act[kcur] != 0) {{
                    if (bot < 0) bot = kcur;
                    --kcur;
                    continue;
                }}
                if (bot >= 0) {{ lo = kcur; hi = bot; bot = -1; --kcur; break; }}
                --kcur;
            }}
            if (lo < 0) {{                                    // the run that reaches pair 1 closes at layer 0
                if (bot < 0) break;                          // (defensive: nblk says there is one)
                lo = 0; hi = bot; bot = -1;
            }}
            moving |= _sm_mix(Ti, qi, zi, dm, p_mid, RHt, lo, hi, gamma_iters, detrain);
        }}
        // (ii) the partition is final AND the profile solve has settled to the arithmetic's own
        // resolution. Both conjuncts: the growth test alone truncates a still-converging fp64 column
        // (measured, see the numpy body), and a bitwise settled test never fires at single precision
        // (the iteration limit-cycles in the last place).
        if (stall >= 2 && moving == 0) {{ done = 1; break; }}
    }}
    // Passes used, NEGATED if the bound was hit with the active set still growing (a TRUNCATED
    // adjustment -- the state is not the fixed point). One int per column, written once; the driver
    // reads it at a batch boundary, never per step.
    status[i] = done ? used : -used;
}}
"""


_SEGMIX_KERNEL_CACHE: dict = {}


def build_segmix_cols_kernel(dtype):
    """Return the compiled one-thread-per-column ``segmix_cols`` RawKernel for the given device dtype
    (``cp.float64`` or ``cp.float32``). Cached per C-precision. Constants are injected from ``constants``
    at build time. Requires CuPy; systems must have nz <= ``_SEGMIX_MAXNZ`` (the thread-local active-pair
    scratch)."""
    if not _HAVE_CUPY:
        raise RuntimeError("build_segmix_cols_kernel requires CuPy (the GPU backend).")
    real = "double" if dtype == np.float64 or dtype == _cp.float64 else "float"
    fn = _SEGMIX_KERNEL_CACHE.get(real)
    if fn is None:
        fn = _cp.RawModule(code=segmix_src_resolved(real)).get_function("segmix_cols")
        _SEGMIX_KERNEL_CACHE[real] = fn
    return fn


def segmix_src_resolved(real):
    """``_segmix_src(real)`` with the physical constants substituted in from ``constants`` -- the EXACT
    text the RawModule compiles. Split out so it can be rendered (and syntax-checked with a host C
    compiler) on a box with no CuPy and no nvcc; ``tests/test_gpu_kernel_source.py`` does exactly that."""
    from . import constants as C
    src = _segmix_src(real)
    return (src.replace("{C_T_FREEZE}", repr(float(C.T_FREEZE)))
               .replace("{C_EPS_MW}", repr(float(C.EPS_MW)))
               .replace("{C_L_VAP}", repr(float(C.L_VAP)))
               .replace("{C_R_DRY}", repr(float(C.R_DRY)))
               .replace("{C_CP_AIR}", repr(float(C.CP_AIR)))
               .replace("{C_G_ACCEL}", repr(float(C.G_ACCEL))))


def segmix_columns_device(kernel, T_atm, q, dm, z, p_mid, RHt, max_passes, gamma_iters, detrain,
                          status, threads_per_block=128):
    """Run the fused CONVERGED convective adjustment RawKernel IN PLACE on (nx, nz) device ``T_atm`` and
    ``q``. All float arrays must be the SAME dtype as ``kernel`` was built for and C-contiguous (the kernel
    indexes i*nz + k). ``RHt`` (nz,): the Manabe RH target (rh_detrain only; pass any (nz,) for
    homogenize). ``detrain``: 1 = rh_detrain, 0 = q_homogenize. ``status`` (nx,) int32: written in place
    with each column's passes used, NEGATED where the column hit ``max_passes`` with its active set still
    growing (a truncated adjustment -- read it at a batch boundary). Mutates T_atm, q and status in place
    (returns nothing).

    ROUND-OFF vs the numpy body, not bit-identity: besides the leaf transcendentals, the kernel accumulates
    the block's mass-weighted offset sum from the block BOTTOM upward while the numpy reduceat sums it from
    the top down. That is a summation-order difference worth ~1e-12 K in the solved block temperature --
    three orders inside the 1e-9 K instability tolerance both paths stop on."""
    nx, nz = T_atm.shape
    if nz > _SEGMIX_MAXNZ:
        raise ValueError(f"segmix kernel scratch sized for nz<={_SEGMIX_MAXNZ}, got nz={nz}")
    for _n, _a in (("T_atm", T_atm), ("q", q), ("status", status)):
        if not _a.flags.c_contiguous:
            # In-place contract: a silent ascontiguousarray copy would drop the update (see micro runner).
            raise ValueError(f"segmix kernel requires C-contiguous {_n} (in-place contract); "
                             f"got strides {_a.strides} for shape {_a.shape}")
    if status.shape != (nx,) or status.dtype != np.int32:
        raise ValueError(f"segmix status must be (nx,) int32; got {status.shape} {status.dtype}")
    tpb = int(threads_per_block)
    bpg = (nx + tpb - 1) // tpb
    kernel((bpg,), (tpb,), (np.int32(nx), np.int32(nz), np.int32(max_passes), np.int32(gamma_iters),
                            np.int32(detrain), T_atm, q, dm, z, p_mid, RHt, status))


# ================================================================== LW two-stream layer recurrence (FUSED)
# A one-thread-per-column fused RawKernel for the diffuse-gray LW two-stream recurrence (``two_stream_lw``),
# the radiation limb's launch-bound part. ``two_stream_lw`` is a per-column SEQUENTIAL nz-sweep: a top->surface
# downwelling pass + a surface->top upwelling pass (each a Python ``for k`` over the N layers), and the
# production window scheme runs it TWICE (background + window bands). At nz144 that is ~2 bands x 2 sweeps x
# 144 layers ~= 570 tiny (nx,) launches every captured step -- the same launch-bound regime convadjust/DST were
# (the CUDA graph amortizes the PYTHON dispatch, not each kernel's on-device launch latency). Profiling the
# captured step: this recurrence is ~2.5 ms (~65% of the 4 ms _radiate; the cloud optics / PDF are the
# efficient full-array remainder). The nx COLUMNS ARE INDEPENDENT, so one thread per column runs the whole
# transfer in-thread, collapsing the ~570 launches into ONE kernel -- the exact restructure the numba
# ``two_stream_lw_parallel`` (prange over columns) already does on the CPU; this kernel MIRRORS that numba
# reference op-for-op (transmissivity ``exp(-D*dtau)``, Planck ``band_frac*sigma*T^4``, both recurrences,
# Fnet/H), so it is round-off-equivalent (NOT bit-identical -- the leaf exp/T^4 reassociate); the ship gate is
# the 60-step climate-match. Templated on ``real`` ('float' for the captured fp32 radiation path / 'double').


def _two_stream_lw_src(real):
    """Source for the one-thread-per-column LW two-stream kernel in C-precision ``real``. Row-major (nx, N)
    layout: column i's per-layer data is the contiguous N block at i*N; Fnet is (nx, N+1) at i*(N+1).
    ``SIGMA_SB`` is a compile-time literal (injected by the builder). Mirrors ``_two_stream_lw_numba`` exactly:
    down pass Dn[k+1]=Dn[k]*t[k]+eB[k] (Fnet[k]=-Dn[k]); surface U_surf=eps*Bs+(1-eps)*D_surf; up pass
    Up[k]=Up[k+1]*t[k]+eB[k] (Fnet[k]+=Up[k]); H[k]=Fnet[k+1]-Fnet[k]. Outputs OLR/D_surf/surf_net_lw (nx,),
    Fnet (nx,N+1), H (nx,N). No thread scratch (the sweeps are single-pass; Fnet doubles as the down-pass
    store the up pass reads back)."""
    e = "expf" if real == "float" else "exp"
    suf = "f" if real == "float" else ""
    return f"""
extern "C" __global__
void two_stream_lw_cols(const int nx, const int N, const {real} eps_surf, const {real} D,
                        const {real} band_frac,
                        const {real}* __restrict__ dtau, const {real}* __restrict__ T_layers,
                        const {real}* __restrict__ T_surf,
                        {real}* __restrict__ OLR, {real}* __restrict__ D_surf_out,
                        {real}* __restrict__ surf_net_lw, {real}* __restrict__ Fnet,
                        {real}* __restrict__ H) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // column
    if (i >= nx) return;
    const {real} SIGMA_SB = {{C_SIGMA_SB}}{suf};
    const {real}* dtau_i = dtau + (long)i * N;
    const {real}* T_i = T_layers + (long)i * N;
    {real}* Fnet_i = Fnet + (long)i * (N + 1);
    {real}* H_i = H + (long)i * N;
    {real} Ts = T_surf[i];
    {real} Bs = band_frac * SIGMA_SB * Ts * Ts * Ts * Ts;
    // down pass: Dn[k+1] = Dn[k]*t[k] + eB[k], Dn[0]=0 at TOA (top->surface, sequential in k)
    {real} Dn_k = 0.0{suf};
    for (int k = 0; k < N; ++k) {{
        {real} tk = {e}(-D * dtau_i[k]);
        {real} Tk = T_i[k];
        {real} Bk = band_frac * SIGMA_SB * Tk * Tk * Tk * Tk;
        {real} eBk = (1.0{suf} - tk) * Bk;
        {real} Dn_kp1 = Dn_k * tk + eBk;
        Fnet_i[k] = -Dn_k;                            // net upward at interface k (Up added below)
        Dn_k = Dn_kp1;
    }}
    {real} D_surf_i = Dn_k;                           // = Dn[..., N]
    Fnet_i[N] = -D_surf_i;
    D_surf_out[i] = D_surf_i;
    surf_net_lw[i] = Bs - D_surf_i;
    // up pass: Up[N]=U_surf, Up[k] = Up[k+1]*t[k] + eB[k] (surface->top, sequential in k)
    {real} Up_kp1 = eps_surf * Bs + (1.0{suf} - eps_surf) * D_surf_i;   // U_surf
    Fnet_i[N] += Up_kp1;                              // net upward at surface = Up_surf - Dn_surf
    for (int k = N - 1; k >= 0; --k) {{
        {real} tk = {e}(-D * dtau_i[k]);
        {real} Tk = T_i[k];
        {real} Bk = band_frac * SIGMA_SB * Tk * Tk * Tk * Tk;
        {real} eBk = (1.0{suf} - tk) * Bk;
        {real} Up_k = Up_kp1 * tk + eBk;
        Fnet_i[k] += Up_k;                            // net upward at interface k = Up_k - Dn_k
        Up_kp1 = Up_k;
    }}
    OLR[i] = Up_kp1;                                  // upwelling at TOA = Up[..., 0]
    for (int k = 0; k < N; ++k) {{                    // H[k] = Fnet[k+1] - Fnet[k]
        H_i[k] = Fnet_i[k + 1] - Fnet_i[k];
    }}
}}
"""


_TWOSTREAM_LW_KERNEL_CACHE: dict = {}


def build_two_stream_lw_kernel(dtype):
    """Return the compiled one-thread-per-column ``two_stream_lw_cols`` RawKernel for the given device dtype
    (``cp.float32`` on the captured fp32 radiation path, ``cp.float64`` for a fp64 device run / parity).
    Cached per C-precision; ``SIGMA_SB`` injected from ``constants`` at build time. Requires CuPy."""
    if not _HAVE_CUPY:
        raise RuntimeError("build_two_stream_lw_kernel requires CuPy (the GPU backend).")
    real = "double" if dtype == np.float64 or dtype == _cp.float64 else "float"
    fn = _TWOSTREAM_LW_KERNEL_CACHE.get(real)
    if fn is None:
        from . import constants as C
        src = _two_stream_lw_src(real).replace("{C_SIGMA_SB}", repr(float(C.SIGMA_SB)))
        fn = _cp.RawModule(code=src).get_function("two_stream_lw_cols")
        _TWOSTREAM_LW_KERNEL_CACHE[real] = fn
    return fn


def two_stream_lw_cols_device(kernel, dtau, T_layers, T_surf, eps_surf, D, band_frac,
                              threads_per_block=128):
    """Run the fused single-band LW two-stream RawKernel over the (nx, N) device columns. Returns the same
    dict as ``two_stream_lw`` (OLR, D_surf, Fnet, H, surf_net_lw), all device arrays of ``dtau``'s dtype.
    All float inputs must match the ``kernel`` dtype and be C-contiguous (the kernel indexes i*N + k)."""
    nx, N = dtau.shape
    dtau = _cp.ascontiguousarray(dtau)
    T_layers = _cp.ascontiguousarray(T_layers)
    dt = dtau.dtype
    sc = dt.type                                   # the scalar constructor (dt itself is not callable)
    Ts = _cp.ascontiguousarray(_cp.broadcast_to(_cp.asarray(T_surf, dt), (nx,)))
    OLR = _cp.empty(nx, dt); D_surf = _cp.empty(nx, dt); surf_net_lw = _cp.empty(nx, dt)
    Fnet = _cp.empty((nx, N + 1), dt); H = _cp.empty((nx, N), dt)
    tpb = int(threads_per_block)
    bpg = (nx + tpb - 1) // tpb
    kernel((bpg,), (tpb,), (np.int32(nx), np.int32(N), sc(eps_surf), sc(D), sc(band_frac),
                            dtau, T_layers, Ts, OLR, D_surf, surf_net_lw, Fnet, H))
    return dict(OLR=OLR, D_surf=D_surf, Fnet=Fnet, H=H, surf_net_lw=surf_net_lw)


def two_stream_lw_windowed_device(kernel, dtau_bg, dtau_win, T_layers, T_surf, f_win, eps_surf, D,
                                  threads_per_block=128):
    """Device drop-in for ``two_stream_lw_windowed``: two fused-kernel band calls (background fraction
    ``1-f_win`` on ``dtau_bg``, window fraction ``f_win`` on ``dtau_win``) summed on device, same dict + the
    per-band OLR (``OLR_bg``/``OLR_win``). Mirrors ``two_stream_lw_windowed_parallel`` (the numba host path)."""
    bg = two_stream_lw_cols_device(kernel, dtau_bg, T_layers, T_surf, eps_surf, D, 1.0 - f_win, threads_per_block)
    win = two_stream_lw_cols_device(kernel, dtau_win, T_layers, T_surf, eps_surf, D, f_win, threads_per_block)
    return dict(OLR=bg["OLR"] + win["OLR"], D_surf=bg["D_surf"] + win["D_surf"],
                Fnet=bg["Fnet"] + win["Fnet"], H=bg["H"] + win["H"],
                surf_net_lw=bg["surf_net_lw"] + win["surf_net_lw"],
                OLR_bg=bg["OLR"], OLR_win=win["OLR"])


# =============================================================== prognostic microphysics + reevap (FUSED)
# A one-thread-per-column fused RawKernel for the CANONICAL prognostic-microphysics + below-cloud-reevap
# limb -- the captured step's single LARGEST limb (the realizable-ceiling A/B: stubbing it cuts the captured
# step ~18 -> ~11 ms, a ~7 ms / 39% ceiling). Unlike convadjust/DST/LW (latency-bound tiny launches), this
# limb is the long ELEMENTWISE chain of ``_prognostic_microphysics`` (+ ``_below_cloud_reevap``): ~30-40
# full (nx,nz) kernels, several with transcendentals (qsat/dqsat/the PDF), each writing a (nx,nz) temporary
# to global memory and reading it back. The fuse wins on BOTH axes -- it collapses those ~30-40 launches into
# ONE and keeps every intermediate (qs, denom, qc_eq, cond, evap, the ceiling, the falling fluxes) in
# registers, eliminating the global round-trips. The cells/columns are independent (microphysics is
# per-(x,z); the reevap sweep is sequential DOWN one column), so one thread per column runs the full
# two-phase transfer in-thread.
#
# This MIRRORS ``stage_ad_core._micro_numba`` (the proven canonical CPU-fuse: PDF source, dqsat denom, ice
# phase, kessler autoconversion, decoupled retained ceiling, reevaporation, the two-flux snow/rain reevap
# with melt) op-for-op, so it is round-off-equivalent to the numpy ``_prognostic_microphysics`` +
# ``_below_cloud_reevap`` reference (NOT bit-identical -- the leaf exp/T^4 reassociate); the ship gate is the
# 60-step climate-match. Templated on ``real`` ('float' for the captured fp32 path / 'double' fp64 parity).
# It covers ONLY the canonical production config (the same gate the numba fuse uses); any other config falls
# back to the numpy-generic ``_prognostic_microphysics``.
_MICRO_MAXNZ = 512


def _micro_src(real):
    """Source for the one-thread-per-column canonical microphysics+reevap kernel in C-precision ``real``.
    Row-major (nx, nz): column i's contiguous nz block at i*nz. Updates T_atm/q/q_c IN PLACE and writes the
    per-column surface precip into ``precip_out`` (nx,). Mirrors ``_micro_numba`` exactly (same predicates,
    same multiply groupings, same Tetens esat). ``src_ice``/``precip_qc_buf`` are thread-local scratch
    carrying the pre-heating ice mask + the per-layer precip source from phase 1 to phase 2."""
    e = "expf" if real == "float" else "exp"
    lg = "logf" if real == "float" else "log"
    suf = "f" if real == "float" else ""
    return f"""
__device__ __forceinline__ {real} _esat_s({real} T) {{
    const {real} T_FREEZE = {{C_T_FREEZE}}{suf};
    {real} Tc = T - T_FREEZE;
    if (Tc >= 0.0{suf}) return 611.2{suf} * {e}(17.67{suf} * Tc / (Tc + 243.5{suf}));
    return 611.2{suf} * {e}(21.87{suf} * Tc / (Tc + 265.5{suf}));
}}
__device__ __forceinline__ {real} _T_boil_s({real} p) {{
    // thermo.T_boil: closed-form inverse of the warm-branch Bolton esat at e = 0.999 p, so
    // T >= _T_boil_s(p) <=> the boiling mask _esat_s(T) >= 0.999 p (warm branch; #294)
    const {real} T_FREEZE = {{C_T_FREEZE}}{suf};
    {real} y = {lg}(0.999{suf} * p / 611.2{suf});
    return T_FREEZE + 243.5{suf} * y / (17.67{suf} - y);
}}
__device__ __forceinline__ {real} _qsat_s({real} T, {real} p) {{
    const {real} EPS_MW = {{C_EPS_MW}}{suf};
    {real} ee = _esat_s(T);
    {real} cap = 0.999{suf} * p;
    if (ee > cap) ee = cap;
    return EPS_MW * ee / (p - (1.0{suf} - EPS_MW) * ee);
}}
__device__ __forceinline__ {real} _desat_dT_s({real} T) {{
    const {real} T_FREEZE = {{C_T_FREEZE}}{suf};
    {real} Tc = T - T_FREEZE;
    {real} ee = _esat_s(T);
    if (Tc >= 0.0{suf}) return ee * 17.67{suf} * 243.5{suf} / ((Tc + 243.5{suf}) * (Tc + 243.5{suf}));
    return ee * 21.87{suf} * 265.5{suf} / ((Tc + 265.5{suf}) * (Tc + 265.5{suf}));
}}
__device__ __forceinline__ {real} _dqsat_dT_s({real} T, {real} p) {{
    const {real} EPS_MW = {{C_EPS_MW}}{suf};
    {real} ee = _esat_s(T);
    {real} cap = 0.999{suf} * p;
    if (ee >= cap) return 0.0{suf};
    {real} denom = p - (1.0{suf} - EPS_MW) * ee;
    return EPS_MW * p * _desat_dT_s(T) / (denom * denom);
}}

extern "C" __global__
void micro_cols(const int nx, const int nz, const {real}* __restrict__ dt_ptr, const {real} rh_crit,
                const {real} autoconv_tau, const {real} autoconv_crit,
                const {real} autoconv_tau_ice, const {real} autoconv_crit_ice,
                const {real} cloud_f_min, const {real} reevap_rate,
                const {real} cond_tau, const int apply_warm,   // #280 finite-tau cond / #237 salt N_d autoconv
                const int virga_flash_boil, const int boil_suppress,   // #294 boiling-layer ints (0/0 = shipped)
                {real}* __restrict__ T_atm, {real}* __restrict__ q, {real}* __restrict__ q_c,
                const {real}* __restrict__ dm, const {real}* __restrict__ p_mid,
                const {real}* __restrict__ warm_factor,        // (nx,) (N_d/N_ref)^beta; read ONLY iff apply_warm
                {real}* __restrict__ precip_out) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // column
    if (i >= nx) return;
    // dt is passed as a DEVICE pointer (not a host scalar): on the capture path dt is a 0-d device scalar
    // (kept on device to avoid the forbidden host sync), so read it through the pointer rather than baking a
    // host value -- and the same launch works for the eager path (dt wrapped into a 0-d device array).
    const {real} dt = dt_ptr[0];
    const {real} T_FREEZE = {{C_T_FREEZE}}{suf};
    const {real} L_VAP = {{C_L_VAP}}{suf};
    const {real} L_SUB = {{C_L_SUB}}{suf};
    const {real} L_FUS = {{C_L_FUS}}{suf};
    const {real} R_DRY = {{C_R_DRY}}{suf};
    const {real} CP_AIR = {{C_CP_AIR}}{suf};
    const {real} IWC_PLAUSIBLE = {{C_IWC_PLAUSIBLE}}{suf};
    const {real} LWC_PLAUSIBLE = {{C_LWC_PLAUSIBLE}}{suf};
    const {real} PDF_WIDTH_FLOOR = {{C_PDF_WIDTH_FLOOR}}{suf};
    {real}* Ti = T_atm + (long)i * nz;
    {real}* qi = q + (long)i * nz;
    {real}* qci = q_c + (long)i * nz;
    char src_ice[{_MICRO_MAXNZ}];
    {real} precip_qc_buf[{_MICRO_MAXNZ}];
    // ---- PHASE 1: saturation-adjustment microphysics (canonical: pdf source, dqsat denom, ice phase,
    //               kessler autoconv, decoupled retained ceiling, reevaporation) ----
    for (int k = 0; k < nz; ++k) {{
        {real} T = Ti[k]; {real} p = p_mid[k];
        int ice = (T < T_FREEZE) ? 1 : 0;                  // PRE-heating production ice (used by reevap)
        src_ice[k] = (char)ice;
        {real} L_eff = ice ? L_SUB : L_VAP;
        {real} qs = _qsat_s(T, p);
        {real} denom = 1.0{suf} + (L_eff / CP_AIR) * _dqsat_dT_s(T, p);
        {real} q_t = qi[k] + qci[k];                        // PDF partition on total water q+q_c
        {real} delta = (1.0{suf} - rh_crit) * qs;
        if (delta < PDF_WIDTH_FLOOR) delta = PDF_WIDTH_FLOOR;
        {real} Q1 = (q_t - qs) / delta;
        {real} f = 0.5{suf} * (Q1 + 1.0{suf});
        if (f < 0.0{suf}) f = 0.0{suf}; else if (f > 1.0{suf}) f = 1.0{suf};
        {real} qc_eq;
        if (Q1 >= 1.0{suf}) qc_eq = q_t - qs;
        else if (Q1 <= -1.0{suf}) qc_eq = 0.0{suf};
        else qc_eq = 0.25{suf} * delta * (1.0{suf} + Q1) * (1.0{suf} + Q1);
        if (qc_eq < 0.0{suf}) qc_eq = 0.0{suf};
        // #294 (boil_suppress): no equilibrium condensate above boiling -- qc_eq -> 0 in a boiling layer,
        // so net below kills the condensation source AND drains suspended q_c to vapour (denom == 1 there:
        // the clamped dqsat_dT is 0). ``boiling`` also zeroes precip_demand below.
        int boiling = 0;
        if (boil_suppress && _esat_s(T) >= 0.999{suf} * p) {{
            qc_eq = 0.0{suf};
            boiling = 1;
        }}
        {real} net = (qc_eq - qci[k]) / denom;
        {real} cond = (net > 0.0{suf}) ? net : 0.0{suf};
        {real} evap_demand = (net < 0.0{suf}) ? (-net) : 0.0{suf};   // qc_reevaporation ON
        if (cond_tau > 0.0{suf}) {{                          // #280: finite-tau saturation adjustment. Scale BOTH
            {real} cfrac = dt / cond_tau;                    // signs of the SAME ``net`` by frac=min(dt/tau,1) ->
            if (cfrac > 1.0{suf}) cfrac = 1.0{suf};          // conjugate q<->q_c exchange preserved. tau=0 skips
            cond *= cfrac; evap_demand *= cfrac;             // (byte-identical to the shipped instantaneous path).
        }}
        {real} qc_avail = qci[k] + cond;
        {real} tau, crit;
        if (ice) {{ tau = autoconv_tau_ice; crit = autoconv_crit_ice; }}
        else {{ tau = autoconv_tau; crit = autoconv_crit; }}
        {real} pa = qc_avail - crit;
        {real} precip_demand = (pa > 0.0{suf}) ? (pa / tau * dt) : 0.0{suf};
        if (apply_warm && !ice) precip_demand *= warm_factor[i];   // #237: salt warm-rain autoconv DEMAND scaling
        //   (N_d/N_ref)^beta, LIQUID only, applied to the rate demand BEFORE the retained-water ceiling below.
        {real} rho_air = p / (R_DRY * T);                   // retained-water ceiling (decoupled PLAUSIBLE bound)
        {real} lwc_ceiling = ice ? IWC_PLAUSIBLE : LWC_PLAUSIBLE;
        {real} rho_den = (rho_air > 1e-9{suf}) ? rho_air : 1e-9{suf};
        {real} qc_ceiling = lwc_ceiling * f / rho_den;
        if (f >= cloud_f_min) {{
            {real} ce = qc_avail - qc_ceiling;
            if (ce > 0.0{suf}) precip_demand += ce;
        }}
        if (boiling) precip_demand = 0.0{suf};              // #294: no precip forms in a boiling layer
        {real} evap = (evap_demand < qc_avail) ? evap_demand : qc_avail;   // coupled-sink positivity limiter
        {real} rem = qc_avail - evap;
        {real} precip_qc = (precip_demand < rem) ? precip_demand : rem;
        qci[k] = qc_avail - evap - precip_qc;
        qi[k] = qi[k] - cond + evap;
        Ti[k] = T + L_eff * (cond - evap) / CP_AIR;
        precip_qc_buf[k] = precip_qc;                       // carry the precip SOURCE [kg/kg] into phase 2
    }}
    // ---- PHASE 2: below-cloud reevap sweep (snow Fs / rain Fr two downward fluxes + melt, flux- AND
    //               deficit-limited reevaporation) ----
    {real} Fs = 0.0{suf}; {real} Fr = 0.0{suf};
    for (int k = 0; k < nz; ++k) {{
        {real} T = Ti[k]; {real} p = p_mid[k];
        {real} qs = _qsat_s(T, p);
        int warm = (T >= T_FREEZE) ? 1 : 0;
        {real} L_layer = warm ? L_VAP : L_SUB;
        {real} denom = 1.0{suf} + (L_layer / CP_AIR) * _dqsat_dT_s(T, p);
        {real} deficit = qs - qi[k];
        if (deficit < 0.0{suf}) deficit = 0.0{suf};
        {real} rate = reevap_rate * deficit * dm[k];
        {real} sat_cap = deficit / denom * dm[k] / dt;
        {real} evap_cap = (rate < sat_cap) ? rate : sat_cap;
        if (virga_flash_boil && _esat_s(T) >= 0.999{suf} * p) {{
            // #294 flash-boil virga: a boiling layer cannot hold falling liquid -- FLOOR the cap at the
            // ENERGY cap (full-flux flash down to the layer's own boiling point), keeping the ordinary
            // deficit-driven rate as the floor's other arm (PR #296 review: a pure replace PINS a
            // rain-cooled layer at T_boil into permanent pass-through). Replaces the #295 sat_cap bound
            // (wrong above boiling: ~0 in a steam-saturated layer).
            {real} bc = T - _T_boil_s(p);
            if (bc < 0.0{suf}) bc = 0.0{suf};
            bc = bc * CP_AIR * dm[k] / (L_VAP * dt);
            if (bc > evap_cap) evap_cap = bc;
        }}
        {real} src = precip_qc_buf[k] * dm[k] / dt;
        if (src_ice[k]) Fs += src; else Fr += src;          // production phase: snow vs rain source
        if (warm) {{                                        // snow melts to rain (absorbs L_FUS)
            {real} melt = Fs;
            Fs = 0.0{suf}; Fr += melt;
            Ti[k] = Ti[k] - L_FUS * (melt * dt / dm[k]) / CP_AIR;
        }}
        {real} Ftot = Fs + Fr;
        {real} E = (Ftot < evap_cap) ? Ftot : evap_cap;     // total reevap, flux- AND deficit-limited
        {real} fdiv = (Ftot > 1e-30{suf}) ? Ftot : 1e-30{suf};
        {real} fr = E / fdiv;                               // split by falling snow/rain fractions
        {real} Es = fr * Fs; {real} Er = fr * Fr;
        Fs -= Es; Fr -= Er;
        qi[k] = qi[k] + (Es + Er) * dt / dm[k];             // reevaporated vapour rejoins q
        Ti[k] = Ti[k] - (L_SUB * Es + L_VAP * Er) * dt / (dm[k] * CP_AIR);
    }}
    precip_out[i] = Fs + Fr;                                // surface-reaching precip [kg/m^2/s]
}}
"""


_MICRO_KERNEL_CACHE: dict = {}


def build_micro_cols_kernel(dtype):
    """Return the compiled one-thread-per-column ``micro_cols`` RawKernel for the given device dtype
    (``cp.float32`` captured path / ``cp.float64`` parity). Cached per C-precision; constants injected from
    ``constants``/``radiation`` at build time. Requires CuPy; nz <= ``_MICRO_MAXNZ`` (the thread scratch)."""
    if not _HAVE_CUPY:
        raise RuntimeError("build_micro_cols_kernel requires CuPy (the GPU backend).")
    real = "double" if dtype == np.float64 or dtype == _cp.float64 else "float"
    fn = _MICRO_KERNEL_CACHE.get(real)
    if fn is None:
        from . import constants as C
        from . import radiation as rad
        src = _micro_src(real)
        for k, v in (("{C_T_FREEZE}", C.T_FREEZE), ("{C_EPS_MW}", C.EPS_MW), ("{C_L_VAP}", C.L_VAP),
                     ("{C_L_SUB}", C.L_SUB), ("{C_L_FUS}", C.L_FUS), ("{C_R_DRY}", C.R_DRY),
                     ("{C_CP_AIR}", C.CP_AIR), ("{C_IWC_PLAUSIBLE}", rad.CLOUD_IWC_PLAUSIBLE),
                     ("{C_LWC_PLAUSIBLE}", rad.CLOUD_LWC_PLAUSIBLE),
                     ("{C_PDF_WIDTH_FLOOR}", rad.CLOUD_PDF_WIDTH_FLOOR)):
            src = src.replace(k, repr(float(v)))
        fn = _cp.RawModule(code=src).get_function("micro_cols")
        _MICRO_KERNEL_CACHE[real] = fn
    return fn


def micro_cols_device(kernel, T_atm, q, q_c, dm, p_mid, dt, rh_crit, autoconv_tau, autoconv_crit,
                      autoconv_tau_ice, autoconv_crit_ice, cloud_f_min, reevap_rate,
                      cond_tau=0.0, apply_warm=0, warm_factor=None, virga_flash_boil=0, boil_suppress=0,
                      threads_per_block=128):
    """Run the fused canonical microphysics+reevap RawKernel IN PLACE on (nx, nz) device T_atm/q/q_c. Returns
    the per-column surface precip (nx,). All float arrays must match the ``kernel`` dtype and be C-contiguous
    (the kernel indexes i*nz + k). nz <= ``_MICRO_MAXNZ``.

    ``cond_tau`` > 0 turns the saturation adjustment into the finite-tau relaxation (#280; 0 = the shipped
    instantaneous path, branch skipped -> byte-identical). ``apply_warm`` (0/1) gates the salt warm-rain
    autoconv scaling by ``warm_factor`` (nx,) = (N_d/N_ref)^beta (#237; 0 -> the array is never read, so the
    canonical path is byte-identical). ``warm_factor`` must be a (nx,) device array of the kernel dtype when
    ``apply_warm`` -- the caller owns it (persistent/address-stable for capture). ``virga_flash_boil`` /
    ``boil_suppress`` (0/1) gate the #294 boiling-layer physics (full-flux energy-capped virga /
    no-condensation-above-boiling); 0/0 -> the branches are skipped, byte-identical."""
    nx, nz = T_atm.shape
    if nz > _MICRO_MAXNZ:
        raise ValueError(f"micro kernel scratch sized for nz<={_MICRO_MAXNZ}, got nz={nz}")
    for _n, _a in (("T_atm", T_atm), ("q", q), ("q_c", q_c)):
        if not _a.flags.c_contiguous:
            # LOUD, not silent: the kernel updates these IN PLACE via flat i*nz+k indexing, so an
            # ascontiguousarray copy here would silently drop the update -- and passing the view garbles
            # every column (the 2026-07-02 nx1500 F-ordered-seed failure). The caller owns the layout.
            raise ValueError(f"micro kernel requires C-contiguous {_n} (in-place contract); "
                             f"got strides {_a.strides} for shape {_a.shape}")
    dt_dtype = T_atm.dtype
    sc = dt_dtype.type
    # dt may be a python float (eager/parity) OR a 0-d DEVICE scalar (capture path, where float(dt) would be
    # an illegal host sync). Wrap it into a 0-d device array of the kernel dtype and pass it as a pointer --
    # capture-safe either way (a tiny cast kernel when dt is already device, a H2D when it is a host float).
    dt_dev = _cp.asarray(dt, dt_dtype)
    precip_out = _cp.empty(nx, dtype=dt_dtype)
    # warm_factor: read only when apply_warm; still must be a valid (nx,) device arg of the kernel dtype (CuPy
    # marshals every pointer arg). When inactive, pass a cheap 1-elem placeholder (never dereferenced).
    if apply_warm:
        if warm_factor is None or warm_factor.shape != (nx,):
            raise ValueError(f"micro kernel apply_warm needs a (nx={nx},) warm_factor; got {warm_factor}")
        wf = _cp.ascontiguousarray(warm_factor, dtype=dt_dtype)
    else:
        wf = warm_factor if warm_factor is not None else _cp.ones(1, dtype=dt_dtype)
    tpb = int(threads_per_block)
    bpg = (nx + tpb - 1) // tpb
    kernel((bpg,), (tpb,), (np.int32(nx), np.int32(nz), dt_dev, sc(rh_crit), sc(autoconv_tau),
                            sc(autoconv_crit), sc(autoconv_tau_ice), sc(autoconv_crit_ice),
                            sc(cloud_f_min), sc(reevap_rate), sc(cond_tau), np.int32(apply_warm),
                            np.int32(virga_flash_boil), np.int32(boil_suppress),
                            T_atm, q, q_c, dm, p_mid, wf, precip_out))
    return precip_out


# ================================================================== diurnal-sun phase gather (chunk 4.1)
# The CAPTURE-path leg of the DIURNAL SUN coupling (COSMO_DIURNAL_SUN; stage_ad_core._advance_diurnal_sun):
# copy row ``floor(t_clock[0] / row_dur) % n_phase`` of the precomputed one-day (n_phase, nx)
# instantaneous-sun table into the FIXED (nx,) held buffer the in-loop radiation reads. The phase is
# carried by a (1,) float64 DEVICE TIME scalar (the diurnal clock), which the caller accumulates IN PLACE
# by the REALIZED dt at the end of each step -- so the phase is EXACT under the CFL-adaptive dt (rows
# repeat/skip as true time dictates; PR #301 review finding 1: the previous one-row-per-step counter
# assumed the dt cap binds, which the canonical diurnal state contradicts -- realized dt ran 30-39 s under
# the 60 s cap, i.e. a ~2x-fast sun). All device ops, capture-legal, no host sync, no per-step H2D.
# Grid-stride over nx (any nx). Templated on ``real`` ('double'/'float') like the other builders; the
# tables and the clock are fp64 in production (they set the absolute forcing scale / true time, mirroring
# the S/S_diffuse placement and the fp64 dynamics solves).


def _diurnal_gather_src(real):
    """Source for the diurnal row-gather kernel in C-precision ``real``. Row-major (n_phase, nx) table;
    ``t_clock`` is the (1,) float64 device diurnal clock [s] (read-only here -- the caller accumulates the
    realized dt into it separately at end-of-step, so the direct and diffuse gathers of one step read the
    SAME start-of-step time); ``row_dur`` = T_day / n_phase [s]."""
    return rf"""
extern "C" __global__
void diurnal_gather(const {real}* __restrict__ table, const double* __restrict__ t_clock,
                    const double row_dur, const long long n_phase, const long long nx,
                    {real}* __restrict__ out)
{{
    long long row = (long long)floor(t_clock[0] / row_dur);
    row = ((row % n_phase) + n_phase) % n_phase;               // int64-safe non-negative modulo
    const {real}* src = table + row * nx;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < nx;
         i += (long long)gridDim.x * blockDim.x)
        out[i] = src[i];
}}
"""


_DIURNAL_KERNEL_CACHE: dict = {}


def build_diurnal_gather_kernel(dtype):
    """Return the compiled ``diurnal_gather`` RawKernel for the given device dtype (``cp.float64`` or
    ``cp.float32``). Cached per C-precision. Requires CuPy. Launch shape: caller passes
    ``((nx+255)//256,), (256,)`` with args ``(table, t_clock, row_dur, n_phase, nx, out)`` -- see
    ``stage_ad_core._advance_diurnal_sun``."""
    if not _HAVE_CUPY:
        raise RuntimeError("build_diurnal_gather_kernel requires CuPy (the GPU backend).")
    real = "double" if dtype == np.float64 or dtype == _cp.float64 else "float"
    fn = _DIURNAL_KERNEL_CACHE.get(real)
    if fn is None:
        fn = _cp.RawModule(code=_diurnal_gather_src(real)).get_function("diurnal_gather")
        _DIURNAL_KERNEL_CACHE[real] = fn
    return fn
