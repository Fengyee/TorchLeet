"""softmax — numerically stable softmax from scratch.

The built-in softmax / log_softmax kernels are disabled while the solver's
function runs (caught through a TorchFunctionMode, however they are reached);
logsumexp, exp, max and sum are all fine.
"""
import torch
from torch.overrides import TorchFunctionMode

ENTRIES = ["softmax"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "exp() of a large number overflows. What can you subtract without changing the result?",
    "softmax(x) == softmax(x - c) for any constant c. Pick c = max(x) along `dim`.",
    "Keep dims when you take the max and the sum so broadcasting lines up.",
]


def check_matches_torch_softmax(ns):
    torch.manual_seed(0)
    for shape in [(5,), (4, 8), (2, 3, 7)]:
        x = torch.randn(*shape)
        got, exp = ns.softmax(x), torch.softmax(x, dim=-1)
        assert got.shape == exp.shape, \
            f"shape {tuple(got.shape)} != {tuple(exp.shape)} for input {shape}"
        assert torch.allclose(got, exp, atol=1e-6), \
            f"disagrees with torch.softmax on {shape} (max diff {(got - exp).abs().max():.2e})"


def check_respects_dim_argument(ns):
    x = torch.randn(4, 6)
    got = ns.softmax(x, dim=0)
    assert torch.allclose(got, torch.softmax(x, dim=0), atol=1e-6), \
        "softmax(x, dim=0) should normalize down columns, not across rows"
    # A middle dimension, by positive and negative index: the max and the sum must
    # both be taken along `dim` (with keepdim) for this to broadcast correctly.
    x = torch.randn(2, 5, 3)
    for dim in (1, -2):
        got = ns.softmax(x, dim=dim)
        assert torch.allclose(got, torch.softmax(x, dim=dim), atol=1e-6), \
            f"softmax(x, dim={dim}) on a (2, 5, 3) tensor disagrees with torch.softmax"


def check_sums_to_one(ns):
    s = ns.softmax(torch.randn(6, 10)).sum(dim=-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-6), \
        f"rows must sum to 1, got range [{s.min():.4f}, {s.max():.4f}]"


def check_numerically_stable(ns):
    x = torch.tensor([[1000.0, 1001.0, 1002.0], [-1000.0, -1001.0, -1002.0]])
    out = ns.softmax(x)
    assert not torch.isnan(out).any(), "NaN on large inputs — subtract the max first"
    assert not torch.isinf(out).any(), "Inf on large inputs — subtract the max first"
    # Looser tolerance than elsewhere: at |x| ~ 1000 a float32 ulp is ~6e-5, so the
    # equally valid exp(x - logsumexp(x)) form is only accurate to ~1e-5 here.
    assert torch.allclose(out, torch.softmax(x, dim=-1), atol=1e-4), \
        "wrong values on large inputs"


def check_does_not_modify_input(ns):
    """`x -= x.max(...)` would shift the caller's tensor in place."""
    x = torch.randn(3, 4) * 5
    before = x.clone()
    ns.softmax(x)
    assert torch.equal(x, before), \
        "softmax modified its input in place — compute x - max into a new tensor"


def check_gradient_matches_torch(ns):
    """Backprop through it must agree with torch.softmax (no in-place op breaking autograd)."""
    torch.manual_seed(1)
    x = torch.randn(4, 7)
    w = torch.randn(4, 7)
    a = x.clone().requires_grad_(True)
    (ns.softmax(a, dim=-1) * w).sum().backward()
    b = x.clone().requires_grad_(True)
    (torch.softmax(b, dim=-1) * w).sum().backward()
    assert a.grad is not None, "no gradient flowed back to the input"
    assert torch.allclose(a.grad, b.grad, atol=1e-6), \
        f"gradient disagrees with torch.softmax (max diff {(a.grad - b.grad).abs().max():.2e})"


_BUILTINS = {"softmax", "_softmax", "log_softmax", "_log_softmax",
             "special_softmax", "special_log_softmax"}


class _NoBuiltinSoftmax(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        name = getattr(func, "__name__", "")
        if name in _BUILTINS:
            raise AssertionError(
                f"the built-in `{name}` was called — the exercise is to write softmax "
                "from exp, max and sum yourself")
        return func(*args, **(kwargs or {}))


def check_does_not_call_the_builtin(ns):
    """torch.softmax / F.softmax / x.log_softmax().exp() would pass every other check."""
    with _NoBuiltinSoftmax():
        ns.softmax(torch.randn(3, 5))
        ns.softmax(torch.randn(2, 4, 3), dim=1)


CHECKS = [check_matches_torch_softmax, check_respects_dim_argument,
          check_sums_to_one, check_numerically_stable, check_does_not_modify_input,
          check_gradient_matches_torch, check_does_not_call_the_builtin]
