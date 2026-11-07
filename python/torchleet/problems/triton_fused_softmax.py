"""triton-fused-softmax — online (single-pass) softmax, then a Triton kernel.

Device policy: the PyTorch part is real work and is checkable on CPU, so it is
graded everywhere. The Triton kernel runs on a GPU, or on CPU when the solver
started Python with TRITON_INTERPRET=1 (Triton's interpreter, which must be on
before the kernel is defined); otherwise it is reported SKIPPED — never passed.
A plain CPU run therefore reports partial verification, which is the honest
answer; it does not claim the kernel is correct.

The kernel is probed with a row length that is not a power of two AND a row
stride larger than the row (a column slice), so a missing mask, a wrong `other`
fill value, or row offsets computed from n_cols instead of the stride all fail.
"""
import os

import torch

from torchleet.runner import Skip

ENTRIES = ["online_softmax_pytorch"]
DEVICE = "cpu"          # the CPU-checkable portion; the kernel check gates itself
EXTRAS = []             # triton is only needed by the optional kernel check

HINTS = [
    "Online softmax keeps a running max and running sum in one pass.",
    "When a new max arrives, rescale the accumulated sum by exp(old_max - new_max).",
    "Subtracting the running max before exp() is what keeps it stable on large inputs.",
]


def check_matches_torch_softmax(ns):
    """torch.softmax is the oracle — it already ships with the user's PyTorch."""
    torch.manual_seed(0)
    for shape in [(4, 8), (1, 128), (16, 33)]:
        x = torch.randn(*shape)
        got = ns.online_softmax_pytorch(x)
        exp = torch.softmax(x, dim=-1)
        assert got.shape == exp.shape, \
            f"shape {tuple(got.shape)} != expected {tuple(exp.shape)} for input {shape}"
        assert torch.allclose(got, exp, atol=1e-6), (
            f"disagrees with torch.softmax on input {shape} "
            f"(max diff {(got - exp).abs().max().item():.2e})")


def check_rows_sum_to_one(ns):
    x = torch.randn(8, 64)
    out = ns.online_softmax_pytorch(x)
    sums = out.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), \
        f"softmax rows must sum to 1, got range [{sums.min():.4f}, {sums.max():.4f}]"


def check_numerically_stable(ns):
    """The point of online softmax: large inputs must not overflow."""
    x = torch.tensor([[1e4, 1e4 + 1.0, 1e4 + 2.0], [-1e4, -1e4 - 1.0, -1e4 - 2.0]])
    out = ns.online_softmax_pytorch(x)
    assert not torch.isnan(out).any(), "NaN on large inputs — subtract the running max"
    assert not torch.isinf(out).any(), "Inf on large inputs — subtract the running max"
    assert torch.allclose(out, torch.softmax(x, dim=-1), atol=1e-6), \
        "wrong values on large inputs (numerical stability)"


def _triton_device():
    """Where a Triton kernel can run here: a GPU, or the CPU under the interpreter."""
    try:
        import triton  # noqa: F401
    except ImportError:
        raise Skip("needs `triton` — pip install triton")
    if torch.cuda.is_available():
        return "cuda"
    if os.environ.get("TRITON_INTERPRET") == "1":
        return "cpu"
    raise Skip("requires CUDA — or set TRITON_INTERPRET=1 before importing triton "
               "to run the kernel on CPU; Triton kernel not verified on this machine")


def check_triton_kernel(ns):
    """Skipped, never passed, when it cannot actually be verified."""
    fn = next((getattr(ns, name) for name in ("fused_softmax_triton", "fused_softmax")
               if getattr(ns, name, None) is not None), None)
    if fn is None:
        raise Skip("pass fused_softmax_triton=<your launcher> to check the kernel")
    device = _triton_device()
    torch.manual_seed(0)
    wide = torch.randn(8, 1024, device=device) * 5
    cases = {
        "1000 columns sliced from a 1024-wide tensor": wide[:, :1000],
        # every logit < 0: a padded lane loaded as 0 instead of -inf would win the max
        "1000 all-negative columns": (wide[:, :1000] - 30).contiguous(),
        "512 columns": wide[:, :512].contiguous(),
        "a single column": wide[:3, :1].contiguous(),
        "values near 1e4": torch.tensor([[1e4, 1e4 + 1.0, 1e4 + 2.0]], device=device),
    }
    for name, x in cases.items():
        got = fn(x)
        expected = torch.softmax(x, dim=-1)
        assert tuple(got.shape) == tuple(x.shape), \
            f"{name}: expected output {tuple(x.shape)}, got {tuple(got.shape)}"
        diff = (got - expected).abs().max().item()
        assert torch.allclose(got, expected, atol=1e-5), (
            f"Triton kernel disagrees with torch.softmax on {name} (max diff "
            f"{diff:.2e}) — check the column mask, that padded lanes load -inf, and "
            f"that rows are addressed with the row stride")


CHECKS = [
    check_matches_torch_softmax,
    check_rows_sum_to_one,
    check_numerically_stable,
    check_triton_kernel,
]
