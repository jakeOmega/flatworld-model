"""Static gates on the CUDA RawKernel SOURCES, runnable on a box with no GPU, no CuPy and no compiler.

WHY THIS EXISTS. The device kernels in ``flatworld/_gpu_kernels.py`` are C strings compiled by NVRTC at
run time, so on a CPU-only machine nothing ever looks at them -- a missing semicolon, a typo'd helper
name, an unsubstituted ``{C_...}`` template slot or an unsuffixed double literal in the fp32 variant all
survive every local gate and surface only on the rented GPU box. These tests close the cheapest part of
that gap **without** a GPU: the rendered source is de-CUDA-ed (the attributes stripped, the builtins
stubbed) and handed to ``pycparser``, which is a real C parser, and then walked for a few hygiene
properties a parser alone cannot see.

WHAT IT DOES AND DOES NOT PROVE. It proves the source PARSES as C, that every function called is one the
translation unit (or the stub prelude) declares, and that the host launcher passes exactly as many
arguments as the ``__global__`` signature declares (a mismatch NVRTC does not diagnose -- it is undefined
behaviour at launch, not a compile error). It does NOT prove it COMPILES as CUDA -- no NVRTC,
no type checking, no register/local-memory budget, and no check that the arithmetic is right. The
correctness gates for that are the numpy-vs-CUDA parity tests in ``test_stage_ad_core.py``, which skip
here and run on the GPU box.

CONTROL. Every check runs first on the SHIPPED ``convadj_cols`` source, which is known to compile on a
GPU box, so a failure of the method itself shows up as a failed control rather than as a false alarm on
the new kernel; and five deliberately-broken variants (three damaged sources, one typo'd call, one
dropped parameter) must be REJECTED, so a vacuous "it passes everything" harness fails too.
"""
import re

import pytest

from flatworld import _gpu_kernels as gk

pycparser = pytest.importorskip("pycparser")
from pycparser import c_ast, c_parser        # noqa: E402  (after importorskip, by design)

# Stubs for what NVRTC provides and a host C parser does not: the launch-configuration builtins and the
# math leaves the kernels call. Everything else the sources use is plain C.
_PRELUDE = """
typedef struct { int x; int y; int z; } dim3;
dim3 blockIdx; dim3 blockDim; dim3 threadIdx;
double exp(double); float expf(float);
double log(double); float logf(float);
double pow(double, double); float powf(float, float);
double sqrt(double); float sqrtf(float);
"""
_PRELUDE_FUNCS = {"exp", "expf", "log", "logf", "pow", "powf", "sqrt", "sqrtf"}


def _strip_comments(src):
    return re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", " ", src, flags=re.S))


def _to_c(src):
    """The rendered kernel source as parseable C: comments out, ``extern "C"`` and the CUDA attributes
    stripped, the builtins declared."""
    s = _strip_comments(src).replace('extern "C"', " ")
    s = re.sub(r"__device__|__global__|__forceinline__|__restrict__|__shared__", " ", s)
    return _PRELUDE + s


def _parse(src):
    return c_parser.CParser().parse(_to_c(src))


def _resolve_convadj(real):
    """``_convadj_src`` with its constants substituted -- the same substitution ``build_convadj_cols_kernel``
    does, lifted here so the control needs no CuPy."""
    from flatworld import constants as C
    src = gk._convadj_src(real)
    for name, val in (("T_FREEZE", C.T_FREEZE), ("EPS_MW", C.EPS_MW), ("L_VAP", C.L_VAP),
                      ("R_DRY", C.R_DRY), ("CP_AIR", C.CP_AIR), ("G_ACCEL", C.G_ACCEL)):
        src = src.replace("{C_%s}" % name, repr(float(val)))
    return src


SOURCES = {
    "convadj_cols[double]": lambda: _resolve_convadj("double"),      # CONTROL (ships, compiles on GPU)
    "convadj_cols[float]": lambda: _resolve_convadj("float"),        # CONTROL
    "segmix_cols[double]": lambda: gk.segmix_src_resolved("double"),
    "segmix_cols[float]": lambda: gk.segmix_src_resolved("float"),
}


@pytest.mark.parametrize("name", list(SOURCES))
def test_kernel_source_parses_as_c(name):
    """Every rendered kernel source parses. The two ``convadj_cols`` entries are the control."""
    _parse(SOURCES[name]())


@pytest.mark.parametrize("break_it", [
    lambda s: s.replace("int work = 0;", "int work = 0", 1),                   # dropped semicolon
    lambda s: s.replace("used = pass; break; }", "used = pass; break;", 1),    # unbalanced brace
    lambda s: s.replace("for (int pass = 0;", "for (int pass = 0", 1),         # malformed for-header
])
def test_the_parser_gate_rejects_broken_source(break_it):
    """NON-VACUOUS: the gate above must actually fail on damaged source, or it proves nothing."""
    with pytest.raises(c_parser.ParseError):
        _parse(break_it(gk.segmix_src_resolved("double")))


@pytest.mark.parametrize("name", list(SOURCES))
def test_every_called_function_is_defined(name):
    """A typo'd helper name (``_sm_gamma`` -> ``_sm_gamm``) PARSES fine and only fails at NVRTC, so walk
    the syntax tree: every function called must be defined in the same translation unit or declared in the
    stub prelude. This is the check that a parser alone does not give."""
    ast = _parse(SOURCES[name]())
    defined = {n.decl.name for n in ast.ext if isinstance(n, c_ast.FuncDef)}
    declared = {n.name for n in ast.ext if isinstance(n, c_ast.Decl)}
    known = defined | declared | _PRELUDE_FUNCS

    called = set()

    class _V(c_ast.NodeVisitor):
        def visit_FuncCall(self, node):
            if isinstance(node.name, c_ast.ID):
                called.add(node.name.name)
            self.generic_visit(node)

    _V().visit(ast)
    assert called, f"{name}: no calls found -- the walk is vacuous"
    assert called <= known, f"{name}: undefined function(s) called: {sorted(called - known)}"


def test_the_called_function_gate_rejects_a_typo():
    """NON-VACUOUS twin of the walk above."""
    ast = _parse(gk.segmix_src_resolved("double").replace("_sm_gamma(Tk,", "_sm_gamm(Tk,", 1))
    defined = {n.decl.name for n in ast.ext if isinstance(n, c_ast.FuncDef)}
    called = set()

    class _V(c_ast.NodeVisitor):
        def visit_FuncCall(self, node):
            if isinstance(node.name, c_ast.ID):
                called.add(node.name.name)
            self.generic_visit(node)

    _V().visit(ast)
    assert "_sm_gamm" in called - (defined | _PRELUDE_FUNCS)


def test_segmix_kernel_keeps_the_loop_nest_that_costs_a_warp_least():
    """PR #830's three warp-cost changes are ALGEBRAIC IDENTITIES -- they move no bit, so
    ``test_segmix_kernel_loop_nest_is_bitwise_neutral`` passes whether they are there or not, and so
    does the on-box parity gate. Nothing else would notice a revert. Anchor them in the source, in both
    precisions, with the reason beside each; the measured costs are in
    ``cosmology_models/phy17_mixer_cost_model.py``.

      (O1) blocks issued by ORDINAL, so a warp pays only the largest b-th block among its lanes
           (divergence 6.0x -> 1.7x on the canonical columns; 16x -> 3.5x on a recently-mixed state);
      (O2) the offsets stashed in place for every iterate but the last, so Gamma is evaluated once per
           iterate instead of twice;
      (O3) the water closure's qsat parked in the q slot instead of evaluated a second time.

    The LAST Gamma iterate must still evaluate Gamma twice -- stopping rule (ii) compares the new
    temperature against the pre-write one -- so the two ``_sm_gamma(T_old, ...)`` sites are the O2 stash
    loop and that last iterate, and there must be exactly two."""
    for real in ("double", "float"):
        src = gk.segmix_src_resolved(real)
        assert "if (act[k] != 0 && act[k-1] == 0) ++nblk;" in src, f"{real}: O1 block count gone"
        assert "for (int b = 0; b < nblk; ++b)" in src, f"{real}: O1 ordinal loop gone"
        assert "if (it < gamma_iters - 1)" in src, f"{real}: O2 branch gone"
        assert "Ti[i] = d;" in src, f"{real}: O2 offset stash gone"
        assert "qi[i] = qs;" in src, f"{real}: O3 qsat park gone"
        assert "qi[i] = A * RHt[i] * qi[i];" in src, f"{real}: O3 read-back gone"
        assert src.count("_sm_gamma(T_old, p_mid[i])") == 2, f"{real}: the last iterate changed shape"


@pytest.mark.parametrize("name", list(SOURCES))
def test_no_template_slot_survives_rendering(name):
    """An unsubstituted ``{C_L_VAP}`` (a constant added to the source but not to the builder's ``replace``
    chain) is a compile error on the GPU box and invisible here; and a stray ``{real}`` means the f-string
    template lost a brace. Neither may reach the module."""
    src = _strip_comments(SOURCES[name]())
    assert not re.findall(r"\{C_[A-Z_0-9]+\}", src), f"{name}: unsubstituted constant slot"
    assert not re.findall(r"\{[a-z_]+\}", src), f"{name}: stray template slot"


@pytest.mark.parametrize("name", [n for n in SOURCES if n.endswith("[float]")])
def test_fp32_sources_keep_every_literal_and_math_call_single_precision(name):
    """In the fp32 variant a bare ``1.0`` or a call to ``exp`` (rather than ``1.0f`` / ``expf``) silently
    promotes the expression to double: not wrong, but it costs the precision gate its meaning and the
    kernel its speed. The shipped ``convadj_cols[float]`` is the control that this bar is met-able."""
    src = _strip_comments(SOURCES[name]())
    # Tokenize WHOLE numeric literals (mantissa + optional exponent + optional 'f'), then require the
    # suffix on anything float-shaped. Do NOT write this as one negative lookahead after an optional
    # exponent group: the engine backtracks out of the exponent and reports ``1.1920929`` inside a
    # perfectly good ``1.1920929e-07f`` -- which it did, on the settled-bar constant.
    toks = re.findall(r"(?<![0-9A-Za-z_.])(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?f?(?![0-9A-Za-z_.])", src)
    bare = [t for t in toks if not t.endswith("f") and re.search(r"[.eE]", t)]
    assert not bare, f"{name}: unsuffixed double literal(s) {sorted(set(bare))}"
    assert any(t.endswith("f") for t in toks), f"{name}: no suffixed literal found -- the walk is vacuous"
    assert not re.search(r"(?<![A-Za-z_])(exp|log|pow|sqrt)\s*\(", src), f"{name}: double-precision math call"


def _kernel_params(src, name):
    """The parameter list of ``__global__ void <name>`` as parsed from the rendered source."""
    ast = _parse(src)
    for node in ast.ext:
        if isinstance(node, c_ast.FuncDef) and node.decl.name == name:
            return list(node.decl.type.args.params)
    raise AssertionError(f"{name} not found in the rendered source")


def _launcher_argc(launcher, **kw):
    """Number of arguments the host launcher actually passes to the kernel, captured with a fake kernel.

    The launcher only reads ``.shape`` / ``.dtype`` / ``.flags`` off its arrays, so numpy stands in for
    CuPy and this runs with no GPU."""
    seen = {}

    def _fake(grid, block, args):
        seen["n"] = len(args)
    launcher(_fake, **kw)
    return seen["n"]


def test_segmix_kernel_signature_matches_its_launcher():
    """THE ERROR A HAND-EDITED KERNEL MAKES, and the one thing a parser CAN catch without NVRTC: the host
    launcher passing a different number of arguments than the ``__global__`` signature declares. NVRTC does
    not type-check the argument tuple -- a mismatch is undefined behaviour at launch, not a compile error --
    so nothing downstream of this test would notice until the GPU box produced garbage. Counted from the
    parsed C on one side and from a fake kernel that records the tuple on the other, so neither side is a
    transcription of the other."""
    import numpy as np
    nx, nz = 4, 6
    kw = dict(T_atm=np.zeros((nx, nz)), q=np.zeros((nx, nz)), dm=np.zeros(nz), z=np.zeros((nx, nz)),
              p_mid=np.zeros(nz), RHt=np.zeros(nz), max_passes=8, gamma_iters=2, detrain=1,
              status=np.zeros(nx, np.int32))
    params = _kernel_params(gk.segmix_src_resolved("double"), "segmix_cols")
    assert len(params) == _launcher_argc(gk.segmix_columns_device, **kw)
    assert params[-1].name == "status"                     # the diagnostic buffer is the trailing arg


def test_the_arity_gate_rejects_a_dropped_parameter():
    """NON-VACUOUS: drop a parameter from the kernel signature and the check above must disagree."""
    import numpy as np
    nx, nz = 4, 6
    kw = dict(T_atm=np.zeros((nx, nz)), q=np.zeros((nx, nz)), dm=np.zeros(nz), z=np.zeros((nx, nz)),
              p_mid=np.zeros(nz), RHt=np.zeros(nz), max_passes=8, gamma_iters=2, detrain=1,
              status=np.zeros(nx, np.int32))
    src = gk.segmix_src_resolved("double").replace("const double* __restrict__ RHt,", "", 1)
    assert len(_kernel_params(src, "segmix_cols")) != _launcher_argc(gk.segmix_columns_device, **kw)
