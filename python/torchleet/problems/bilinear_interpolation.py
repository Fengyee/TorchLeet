"""bilinear-interpolation — resize an (N, C, H, W) batch like F.interpolate.

Every value check is an oracle comparison against F.interpolate on many shapes
(up/down-sampling, non-integer scales, size-1 inputs and outputs), in both
align_corners modes, plus the gradient through the resize. Vectorization is
checked by counting torch calls with a TorchFunctionMode: a gather-based resize
makes the same number of calls for an 8x8 and a 64x64 output, while a Python
loop over output pixels (or rows) makes many more. The same counter flags any
call into PyTorch's own resizers (interpolate / upsample / grid_sample).

resize_nearest is compared on size pairs where PyTorch agrees with the exact
rule floor(dst * in / out). PyTorch computes the scale in float32, which for a
few odd pairs (e.g. 6 -> 74) floors one index lower than exact integer
arithmetic would; those pairs are skipped so neither convention is penalised.
"""
import random

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

ENTRIES = ["resize_bilinear", "resize_nearest"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Bilinear is separable: work out, for each output row, which two input rows it "
    "reads and with what weights, do the same for columns, then gather and blend. "
    "Those index/weight vectors depend only on the sizes, never on pixel values.",
    "align_corners=False maps output index d to src = (d + 0.5) * in/out - 0.5 "
    "(pixel centres line up) and PyTorch clamps a negative src to 0; "
    "align_corners=True maps d to d * (in - 1)/(out - 1), and to 0 when out == 1.",
    "From src: i0 = floor(src) clamped to in-1, i1 = min(i0 + 1, in-1), w = src - i0. "
    "Index whole rows with x[:, :, i0, :] and x[:, :, i1, :], blend with (1-w) and w "
    "reshaped to broadcast over the row axis, then repeat along the last axis.",
]

_BANNED = ("interpolate", "upsample", "grid_sample", "affine_grid", "resize")


def _cases(n, seed, max_in=24, max_out=40):
    rng = random.Random(seed)
    fixed = [((4, 4), (8, 8)), ((8, 8), (4, 4)), ((5, 7), (13, 3)), ((6, 9), (4, 14)),
             ((1, 1), (3, 5)), ((6, 6), (1, 1)), ((7, 5), (1, 9)), ((3, 8), (11, 1)),
             ((1, 6), (4, 6)), ((10, 10), (10, 10)), ((9, 13), (22, 31))]
    rand = [((rng.randint(1, max_in), rng.randint(1, max_in)),
             (rng.randint(1, max_out), rng.randint(1, max_out))) for _ in range(n)]
    return fixed + rand


def _compare_bilinear(ns, align_corners, dtype=torch.float32, n=40, seed=0, tol=1e-5):
    g = torch.Generator().manual_seed(seed)
    for (H, W), (oh, ow) in _cases(n, seed):
        x = torch.randn(2, 3, H, W, generator=g, dtype=dtype)
        exp = F.interpolate(x, size=(oh, ow), mode="bilinear", align_corners=align_corners)
        got = ns.resize_bilinear(x.clone(), (oh, ow), align_corners=align_corners)
        assert torch.is_tensor(got) and tuple(got.shape) == (2, 3, oh, ow), (
            f"resizing (2, 3, {H}, {W}) to size=({oh}, {ow}) should give (2, 3, {oh}, {ow}), "
            f"got {tuple(got.shape) if torch.is_tensor(got) else type(got).__name__}")
        assert got.dtype == dtype, \
            f"output dtype should match the input ({dtype}), got {got.dtype}"
        diff = (got - exp).abs().max().item()
        if not diff <= tol:
            hint = ("with align_corners=False the source coordinate is "
                    "(d + 0.5) * in/out - 0.5, clamped at 0 — clamp src itself, not just the index"
                    if not align_corners else
                    "with align_corners=True the source coordinate is d * (in-1)/(out-1), "
                    "and 0 for every output when out == 1")
            raise AssertionError(
                f"(2, 3, {H}, {W}) -> ({oh}, {ow}), align_corners={align_corners}: max |diff| "
                f"vs F.interpolate = {diff:.2e} (tolerance {tol:g}). Hint: {hint}")


def check_matches_interpolate_align_corners_false(ns):
    _compare_bilinear(ns, align_corners=False, seed=0)


def check_matches_interpolate_align_corners_true(ns):
    _compare_bilinear(ns, align_corners=True, seed=1)


def check_size_one_edge_cases(ns):
    for ac in (False, True):
        for (H, W), (oh, ow) in [((5, 8), (1, 1)), ((1, 1), (4, 6)), ((1, 7), (3, 1)),
                                 ((6, 1), (1, 5)), ((2, 2), (1, 3))]:
            x = torch.randn(1, 2, H, W, generator=torch.Generator().manual_seed(H * 31 + W))
            exp = F.interpolate(x, size=(oh, ow), mode="bilinear", align_corners=ac)
            got = ns.resize_bilinear(x, (oh, ow), align_corners=ac)
            assert torch.isfinite(got).all(), (
                f"({H}, {W}) -> ({oh}, {ow}) with align_corners={ac} produced NaN/Inf — "
                f"(in-1)/(out-1) divides by zero when out == 1; PyTorch uses scale 0 there")
            diff = (got - exp).abs().max().item()
            assert diff <= 1e-5, (
                f"({H}, {W}) -> ({oh}, {ow}) with align_corners={ac}: max |diff| vs "
                f"F.interpolate = {diff:.2e}. Size-1 axes: with align_corners=True an output "
                f"of size 1 reads input index 0; an input of size 1 is just copied")


def check_same_size_is_identity(ns):
    x = torch.randn(2, 3, 9, 14, generator=torch.Generator().manual_seed(3))
    for ac in (False, True):
        got = ns.resize_bilinear(x, (9, 14), align_corners=ac)
        diff = (got - x).abs().max().item()
        assert diff <= 1e-6, (
            f"resizing to the same size with align_corners={ac} must return the input "
            f"(max |diff| = {diff:.2e}) — the coordinate map should send d to exactly d")


def check_constant_image_stays_constant(ns):
    for ac in (False, True):
        for size in [(17, 5), (3, 3), (1, 20)]:
            x = torch.full((1, 2, 6, 9), 3.75)
            got = ns.resize_bilinear(x, size, align_corners=ac)
            diff = (got - 3.75).abs().max().item()
            assert diff <= 1e-5, (
                f"a constant image must stay constant after resizing to {size} "
                f"(align_corners={ac}); max deviation {diff:.2e} — the two weights "
                f"along each axis must sum to 1")


def check_gradient_matches_interpolate(ns):
    g = torch.Generator().manual_seed(5)
    for (H, W), (oh, ow) in [((5, 7), (12, 4)), ((8, 8), (3, 13)), ((4, 6), (4, 6)),
                             ((6, 5), (1, 1))]:
        for ac in (False, True):
            x = torch.randn(2, 2, H, W, generator=g)
            up = torch.randn(2, 2, oh, ow, generator=g)
            xa = x.clone().requires_grad_(True)
            got = ns.resize_bilinear(xa, (oh, ow), align_corners=ac)
            assert got.requires_grad, \
                "the output is detached from x — the resize must be differentiable"
            (got * up).sum().backward()
            assert xa.grad is not None, "no gradient reached x"
            xb = x.clone().requires_grad_(True)
            (F.interpolate(xb, size=(oh, ow), mode="bilinear", align_corners=ac) * up) \
                .sum().backward()
            diff = (xa.grad - xb.grad).abs().max().item()
            assert diff <= 1e-5, (
                f"gradient w.r.t. x disagrees with F.interpolate's for ({H}, {W}) -> "
                f"({oh}, {ow}), align_corners={ac} (max |diff| {diff:.2e})")


def check_float64_supported(ns):
    _compare_bilinear(ns, align_corners=False, dtype=torch.float64, n=10, seed=7)
    _compare_bilinear(ns, align_corners=True, dtype=torch.float64, n=10, seed=8)


def _nearest_exact_rule_agrees(i, o):
    ref = F.interpolate(torch.arange(i, dtype=torch.float32).view(1, 1, 1, i),
                        size=(1, o), mode="nearest").view(-1).long()
    return torch.equal(ref, (torch.arange(o) * i) // o)


def check_nearest_matches_interpolate(ns):
    rng = random.Random(11)
    g = torch.Generator().manual_seed(11)
    cases = [((4, 4), (8, 8)), ((8, 8), (4, 4)), ((5, 7), (13, 3)), ((1, 1), (3, 2)),
             ((7, 9), (1, 1)), ((6, 10), (6, 10))]
    while len(cases) < 40:
        H, W, oh, ow = (rng.randint(1, 24), rng.randint(1, 24),
                        rng.randint(1, 40), rng.randint(1, 40))
        if _nearest_exact_rule_agrees(H, oh) and _nearest_exact_rule_agrees(W, ow):
            cases.append(((H, W), (oh, ow)))
    for (H, W), (oh, ow) in cases:
        x = torch.randn(2, 3, H, W, generator=g)
        exp = F.interpolate(x, size=(oh, ow), mode="nearest")
        got = ns.resize_nearest(x, (oh, ow))
        assert torch.is_tensor(got) and tuple(got.shape) == (2, 3, oh, ow), (
            f"resize_nearest (2, 3, {H}, {W}) -> ({oh}, {ow}) gave "
            f"{tuple(got.shape) if torch.is_tensor(got) else type(got).__name__}")
        if not torch.equal(got, exp):
            raise AssertionError(
                f"resize_nearest (2, 3, {H}, {W}) -> ({oh}, {ow}) differs from "
                f"F.interpolate(mode='nearest'). PyTorch's 'nearest' uses "
                f"src = floor(d * in/out) (no half-pixel offset, no rounding), clamped "
                f"to in-1 — rounding to the closest centre is 'nearest-exact'")


class _CallCounter(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.n = 0
        self.banned = set()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.n += 1
        name = getattr(func, "__name__", str(func))
        if any(b in name for b in _BANNED):
            self.banned.add(name)
        return func(*args, **(kwargs or {}))


def _count(fn, x, size, **kw):
    with _CallCounter() as c:
        fn(x, size, **kw)
    return c


def check_vectorized_and_no_builtin_resize(ns):
    x = torch.randn(1, 2, 8, 8, generator=torch.Generator().manual_seed(2))
    for name, fn, kw in [("resize_bilinear", ns.resize_bilinear, {"align_corners": False}),
                         ("resize_bilinear", ns.resize_bilinear, {"align_corners": True}),
                         ("resize_nearest", ns.resize_nearest, {})]:
        small = _count(fn, x, (8, 8), **kw)
        big = _count(fn, x, (64, 64), **kw)
        banned = small.banned | big.banned
        assert not banned, (
            f"{name} calls PyTorch's own resizer ({', '.join(sorted(banned))}) — "
            f"build it from indexing and arithmetic")
        assert big.n <= 1.5 * small.n + 20, (
            f"{name} made {small.n} torch calls for an 8x8 output but {big.n} for 64x64 — "
            f"the work is looping over output pixels/rows in Python. Compute index and "
            f"weight vectors once and gather all rows/columns in one indexing op")


CHECKS = [
    check_matches_interpolate_align_corners_false,
    check_matches_interpolate_align_corners_true,
    check_size_one_edge_cases,
    check_same_size_is_identity,
    check_constant_image_stays_constant,
    check_gradient_matches_interpolate,
    check_float64_supported,
    check_nearest_matches_interpolate,
    check_vectorized_and_no_builtin_resize,
]
