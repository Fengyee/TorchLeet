"""cnn-scratch — convolution and max-pool implemented by hand.

nn.Conv2d / F.max_pool2d are the oracles. The solver's own weights are copied
into the reference layer, so only the convolution arithmetic is compared — on
non-square inputs (so H and W cannot be swapped unnoticed), several kernel /
stride / padding combinations, overlapping pools and the default pool stride.
Gradients are compared too, which catches a hand-written backward that is off.
The built-in conv / pool kernels are disabled while the solver's layers run —
caught through a TorchFunctionMode, so `from torch.nn.functional import conv2d`,
torch.convolution or an nn.Conv2d built inside forward() are refused too: the
point of the problem is to implement them, not to wrap them.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

ENTRIES = ["Conv2dCustom", "MaxPool2dCustom"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "Output size is (H + 2*padding - kernel) // stride + 1 per spatial dim.",
    "F.unfold turns sliding windows into columns if you want to avoid Python loops.",
    "Do not forget to add the bias, once per output channel.",
]

# (kernel, stride, padding, H, W) — non-square inputs on purpose
CONV_CASES = ((3, 1, 0, 12, 12), (3, 2, 1, 11, 14), (5, 3, 2, 13, 9), (1, 1, 0, 5, 7))
# (kernel, stride, H, W); stride None means "default to the kernel size"
POOL_CASES = ((2, 2, 8, 8), (3, 2, 9, 12), (3, None, 10, 7), (2, 1, 5, 6))


def check_conv_output_shape(ns):
    for k, stride, pad, h, w in CONV_CASES:
        m = ns.Conv2dCustom(3, 8, k, stride=stride, padding=pad)
        out = m(torch.randn(2, 3, h, w))
        exp = (2, 8, (h + 2 * pad - k) // stride + 1, (w + 2 * pad - k) // stride + 1)
        assert tuple(out.shape) == exp, (
            f"kernel={k} stride={stride} padding={pad} on a {h}x{w} input: "
            f"expected {exp}, got {tuple(out.shape)}")


def check_conv_matches_torch(ns):
    """Same weights in, same numbers (and gradients) out."""
    torch.manual_seed(0)
    for k, stride, pad, h, w in CONV_CASES:
        m = ns.Conv2dCustom(3, 4, k, stride=stride, padding=pad)
        ref = nn.Conv2d(3, 4, k, stride=stride, padding=pad)
        with torch.no_grad():
            ref.weight.copy_(m.weight)
            ref.bias.copy_(m.bias)
        x = torch.randn(2, 3, h, w, requires_grad=True)
        got, exp = m(x), ref(x)
        where = f"kernel={k} stride={stride} padding={pad} on a {h}x{w} input"
        assert got.shape == exp.shape, \
            f"{where}: expected {tuple(exp.shape)}, got {tuple(got.shape)}"
        assert torch.allclose(got, exp, atol=1e-5), (
            f"{where}: disagrees with nn.Conv2d "
            f"(max diff {(got - exp).abs().max():.2e})")
        g = torch.randn_like(exp)
        gx, gw = torch.autograd.grad(got, (x, m.weight), g)
        ex, ew = torch.autograd.grad(exp, (x, ref.weight), g)
        assert torch.allclose(gx, ex, atol=1e-4) and torch.allclose(gw, ew, atol=1e-4), (
            f"{where}: the forward pass matches nn.Conv2d but the gradients do not")


def check_conv_gradients_flow(ns):
    m = ns.Conv2dCustom(3, 4, 3)
    m(torch.randn(1, 3, 10, 10)).sum().backward()
    assert m.weight.grad is not None and torch.any(m.weight.grad != 0), \
        "no gradient reached the conv weights"
    assert m.bias.grad is not None and torch.any(m.bias.grad != 0), \
        "no gradient reached the conv bias"


def check_maxpool_matches_torch(ns):
    torch.manual_seed(0)
    for k, stride, h, w in POOL_CASES:
        p = ns.MaxPool2dCustom(k) if stride is None else ns.MaxPool2dCustom(k, stride=stride)
        x = torch.randn(2, 3, h, w)
        got, exp = p(x), F.max_pool2d(x, k, stride=stride)
        where = f"kernel={k} stride={stride} on a {h}x{w} input"
        assert tuple(got.shape) == tuple(exp.shape), \
            f"{where}: expected {tuple(exp.shape)}, got {tuple(got.shape)}"
        assert torch.allclose(got, exp, atol=1e-6), \
            f"{where}: disagrees with F.max_pool2d (max diff {(got - exp).abs().max():.2e})"


_BUILTINS = {
    "conv1d", "conv2d", "conv3d", "convolution", "_convolution",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
    "max_pool1d", "max_pool2d", "max_pool3d", "max_pool1d_with_indices",
    "max_pool2d_with_indices", "max_pool3d_with_indices",
    "adaptive_max_pool1d", "adaptive_max_pool2d", "fractional_max_pool2d",
}


class _NoBuiltinConvPool(TorchFunctionMode):
    """Fails on any call to a built-in convolution or max-pool, however it is
    reached (F.conv2d, a directly imported conv2d, torch.convolution, nn.Conv2d)."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        name = getattr(func, "__name__", "")
        if name in _BUILTINS:
            raise AssertionError(
                f"the built-in `{name}` was called — implement the operation yourself "
                "(slicing, Tensor.unfold / F.unfold and a matmul or einsum)")
        return func(*args, **(kwargs or {}))


def check_does_not_wrap_the_builtins(ns):
    """The layers must do the work themselves, not call nn.Conv2d / F.conv2d."""
    conv = ns.Conv2dCustom(3, 4, 3, stride=1, padding=1)
    pool = ns.MaxPool2dCustom(2, stride=2)
    for layer in (conv, pool):
        inner = [type(x).__name__ for x in layer.modules()
                 if isinstance(x, (nn.Conv2d, nn.MaxPool2d))]
        assert not inner, (
            f"{type(layer).__name__} contains {', '.join(inner)} — implement the "
            "operation with tensor ops instead of wrapping the built-in layer")
    with _NoBuiltinConvPool():
        conv(torch.randn(1, 3, 8, 8))
        pool(torch.randn(1, 3, 8, 8))


CHECKS = [check_conv_output_shape, check_conv_matches_torch,
          check_conv_gradients_flow, check_maxpool_matches_torch,
          check_does_not_wrap_the_builtins]
