"""vit-components — the parts of a Vision Transformer that other problems take for granted:
Conv2d patch embedding (and its unfold + matmul equivalent), resampling learned position
embeddings to a new grid, per-sample stochastic depth (DropPath), a pre-norm block with
LayerScale, and a small ViT classifier.

Conventions graded here, with their sources:

* PatchEmbed — ViT (Dosovitskiy et al., ICLR 2021), sec. 3.1: flatten P x P patches and project
  linearly, implemented as nn.Conv2d(in_chans, D, kernel_size=P, stride=P) named `proj` (timm
  PatchEmbed). Tokens are in ROW-MAJOR grid order: token r * (W/P) + c is the patch at grid row r,
  column c (what conv(...).flatten(2).transpose(1, 2) gives). Attributes grid_size = (H/P, W/P) and
  num_patches. forward accepts any H, W that are multiples of P.
* patchify_linear(x, weight, bias, P) — the same map with F.unfold / reshape + one matmul, given the
  Conv2d weight (D, C, P, P): F.unfold orders each column as (c, i, j), which matches
  weight.reshape(D, -1). No conv call allowed.
* interpolate_pos_embed — the DeiT / MAE fine-tuning recipe (deit/main.py and mae/util/pos_embed.py,
  `interpolate_pos_embed`): keep the num_prefix_tokens leading rows (CLS, registers) unchanged,
  reshape the rest (row-major) to (1, gh, gw, D) -> permute to (1, D, gh, gw), F.interpolate with
  mode="bicubic", align_corners=False, permute back and flatten. Non-square grids are (h, w)
  tuples. EITHER antialias setting is accepted: False (PyTorch's default, DeiT/MAE) or True (timm's
  resample_abs_pos_embed, which switches to a PIL-style kernel with a = -0.5, widened when
  shrinking, with renormalised border weights, so it differs even when upsampling). The point of
  the exercise is the prefix / reshape / permute bookkeeping, not the kernel. The oracles below
  are independent separable 1-D resampling matrices, not calls to F.interpolate: Keys bicubic
  (a = -0.75, half-pixel centres, clamped borders = PyTorch's upsample_bicubic2d with
  align_corners=False), and PIL's bicubic filter (= PyTorch's antialias=True path).
* DropPath — timm's drop_path (timm/layers/drop.py): per-SAMPLE Bernoulli(keep_prob) mask of shape
  (B, 1, ..., 1), kept samples scaled by 1 / keep_prob (no scaling when keep_prob == 0),
  identity in eval mode and when drop_prob == 0. Stochastic depth itself is Huang et al. (ECCV
  2016), who drop a block for the whole mini-batch and rescale at test time instead; the
  per-sample, inverted-scaling form is torchvision's stochastic_depth(mode="row") and timm's.
  The probability is stored as `drop_prob`.
* ViTBlock — pre-norm (ViT eqs. 2-3): x = x + DropPath(LS1(Attn(LN1(x)))); x = x + DropPath(LS2(
  MLP(LN2(x)))). LayerScale (CaiT, Touvron et al., ICCV 2021): a learnable per-channel vector gamma
  of shape (dim,), initialised to init_values, multiplying the branch output. init_values=None
  means no LayerScale; any float, INCLUDING 0.0, creates it (timm's Block tests truthiness, so
  0.0 turns LayerScale off there — this problem uses `is not None`).
* ViT — patch_embed, cls_token (1, 1, D), pos_embed (1, 1 + N, D) added after prepending CLS
  (original ViT/DeiT, timm's no_embed_class=False), blocks, final LayerNorm `norm`, linear `head`
  on the CLS token. Stochastic depth decay rule: block i gets drop_path = torch.linspace(0,
  drop_path_rate, depth)[i] (timm calculate_drop_path_rates; DeiT).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchleet.runner import Skip

ENTRIES = ["PatchEmbed", "patchify_linear", "interpolate_pos_embed", "DropPath", "ViTBlock", "ViT"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "A patch embedding is a convolution whose kernel equals its stride, and its output grid "
    "flattened row by row is the token sequence. Position embeddings are an image of D-channel "
    "pixels plus a few prefix rows that are not on the grid. DropPath drops a whole residual "
    "branch for a whole sample, not individual activations.",
    "patchify_linear: F.unfold with kernel = stride = p returns (B, C*p*p, N), each column ordered "
    "(channel, row, col) - the same order as the Conv2d weight flattened to (D, C*p*p). "
    "interpolate_pos_embed: the grid rows are row-major (h, w, D) but F.interpolate wants "
    "(1, D, h, w), so reshape AND permute (a reshape alone scrambles them); bicubic with "
    "align_corners=False. DropPath: one Bernoulli(keep_prob) draw per sample, mask shape "
    "(B, 1, ..., 1), kept samples divided by keep_prob.",
    "PatchEmbed: conv, flatten the two spatial dims, move channels last. interpolate_pos_embed: "
    "split off the prefix rows, reshape + permute the grid, interpolate to (new_h, new_w), permute "
    "back, flatten, and put the untouched prefix in front again. DropPath: identity unless training "
    "and drop_prob > 0, otherwise multiply by the scaled mask (no division when keep_prob is 0). "
    "ViTBlock: add DropPath(LayerScale(attention of LN1(x))) to x, then the same with LN2 and the "
    "MLP; create LayerScale whenever init_values is not None. ViT: CLS token in front of the "
    "patch tokens, add pos_embed, blocks, final LayerNorm, head on token 0; block i drops with the "
    "i-th of `depth` evenly spaced rates from 0 to drop_path_rate.",
]


# ---------------------------------------------------------------- helpers

def _build(cls, what, *args, **kwargs):
    try:
        return cls(*args, **kwargs)
    except TypeError as e:
        raise AssertionError(f"could not construct {what} ({e})")


def _patch_embed(ns, img_size, p, c, d):
    pe = _build(ns.PatchEmbed, "PatchEmbed(img_size, patch_size, in_chans, embed_dim)",
                img_size=img_size, patch_size=p, in_chans=c, embed_dim=d)
    assert isinstance(getattr(pe, "proj", None), nn.Conv2d), \
        "PatchEmbed must hold its projection as an nn.Conv2d named `proj` (timm's naming)"
    return pe


def _patches_brute(x, weight, bias, p):
    """token r*gw + c = <weight, patch at grid (r, c)> + bias, one patch at a time."""
    b, _, h, w = x.shape
    gh, gw = h // p, w // p
    out = torch.zeros(b, gh * gw, weight.shape[0], dtype=x.dtype)
    wf = weight.reshape(weight.shape[0], -1)
    for r in range(gh):
        for c in range(gw):
            patch = x[:, :, r * p:(r + 1) * p, c * p:(c + 1) * p].reshape(b, -1)
            out[:, r * gw + c] = patch @ wf.T + (0 if bias is None else bias)
    return out


def _col_major(tokens, gh, gw):
    """The same tokens listed column by column (the classic transposed-order bug)."""
    b, n, d = tokens.shape
    return tokens.reshape(b, gh, gw, d).transpose(1, 2).reshape(b, n, d)


def _cubic(t, a=-0.75):
    t = abs(t)
    if t <= 1:
        return ((a + 2) * t - (a + 3)) * t * t + 1
    if t < 2:
        return ((a * t - 5 * a) * t + 8 * a) * t - 4 * a
    return 0.0


def _bicubic_matrix(n_in, n_out):
    """(n_out, n_in) 1-D resampling matrix: half-pixel centres (align_corners=False), Keys cubic
    with a = -0.75, out-of-range taps clamped to the border."""
    m = torch.zeros(n_out, n_in, dtype=torch.float64)
    scale = n_in / n_out
    for i in range(n_out):
        src = (i + 0.5) * scale - 0.5
        f = math.floor(src)
        t = src - f
        for k in range(-1, 3):
            m[i, min(max(f + k, 0), n_in - 1)] += _cubic(k - t)
    return m


def _antialias_matrix(n_in, n_out):
    """(n_out, n_in) 1-D matrix of PIL's bicubic filter (PyTorch's antialias=True path): Keys cubic
    with a = -0.5, stretched by n_in / n_out when shrinking; taps outside the grid are dropped and
    the remaining weights renormalised to sum to 1 (no clamping)."""
    m = torch.zeros(n_out, n_in, dtype=torch.float64)
    scale = n_in / n_out
    s = max(scale, 1.0)
    for i in range(n_out):
        centre = (i + 0.5) * scale
        lo = max(int(centre - 2 * s + 0.5), 0)
        hi = min(int(centre + 2 * s + 0.5), n_in)
        w = [_cubic((j - centre + 0.5) / s, a=-0.5) for j in range(lo, hi)]
        for j, wj in zip(range(lo, hi), w):
            m[i, j] = wj / sum(w)
    return m


def _interp_oracle(pos, old, new, n_prefix, antialias=False):
    d = pos.shape[-1]
    mat = _antialias_matrix if antialias else _bicubic_matrix
    grid = pos[0, n_prefix:].double().reshape(old[0], old[1], d)
    out = torch.einsum("ih,hwd,jw->ijd", mat(old[0], new[0]), grid, mat(old[1], new[1]))
    return torch.cat([pos[0, :n_prefix].double(), out.reshape(-1, d)])[None]


def _matches_bicubic(got, pos, old, new, n_prefix, atol=1e-5):
    """True if `got` equals bicubic resampling with antialias=False or with antialias=True."""
    return any(torch.allclose(got.double(), _interp_oracle(pos, old, new, n_prefix, aa), atol=atol)
               for aa in (False, True))


def _interp_variants(pos, old, new, n_prefix):
    """Classic mistakes, via F.interpolate — only used to explain a failure."""
    d = pos.shape[-1]
    pre, grid = pos[:, :n_prefix], pos[:, n_prefix:]
    img = grid.reshape(1, old[0], old[1], d).permute(0, 3, 1, 2)

    def back(t):
        return torch.cat([pre, t.permute(0, 2, 3, 1).reshape(1, -1, d)], 1)

    out = {
        "align_corners=True (use align_corners=False)":
            back(F.interpolate(img, size=new, mode="bicubic", align_corners=True)),
        "bilinear interpolation (use mode='bicubic')":
            back(F.interpolate(img, size=new, mode="bilinear", align_corners=False)),
        "grid.reshape(1, D, h, w) without a permute — the tokens are rows, so reshape to "
        "(1, h, w, D) and then permute":
            torch.cat([pre, F.interpolate(grid.reshape(1, d, old[0], old[1]), size=new,
                                          mode="bicubic", align_corners=False)
                       .reshape(1, -1, d)], 1),
    }
    if new[0] != new[1]:
        out["height and width swapped in size=(...)"] = back(
            F.interpolate(img, size=(new[1], new[0]), mode="bicubic", align_corners=False))
    return out


# ---------------------------------------------------------------- PatchEmbed / patchify

def check_patch_embed_shapes_and_attributes(ns):
    pe = _patch_embed(ns, (16, 24), 4, 3, 10)
    assert tuple(pe.grid_size) == (4, 6), \
        f"grid_size for a 16x24 image and patch 4 should be (4, 6) (rows, cols), got {pe.grid_size}"
    assert pe.num_patches == 24, f"num_patches should be 24, got {pe.num_patches}"
    k, s = pe.proj.kernel_size, pe.proj.stride
    assert tuple(k) == (4, 4) and tuple(s) == (4, 4), \
        f"proj must have kernel_size = stride = patch_size = 4, got kernel {k}, stride {s}"
    out = pe(torch.randn(2, 3, 16, 24))
    assert tuple(out.shape) == (2, 24, 10), \
        f"PatchEmbed((2, 3, 16, 24)) should return (B, N, D) = (2, 24, 10), got {tuple(out.shape)}"
    sq = _patch_embed(ns, 32, 8, 3, 6)
    assert tuple(sq.grid_size) == (4, 4) and sq.num_patches == 16, \
        "an int img_size means a square image"


def check_patch_embed_row_major_order(ns):
    torch.manual_seed(0)
    pe = _patch_embed(ns, (16, 24), 4, 3, 10).double()
    for shape in ((2, 3, 16, 24), (1, 3, 8, 20)):           # the second is not img_size
        x = torch.randn(*shape, dtype=torch.float64)
        try:
            got = pe(x)
        except Exception as e:
            raise AssertionError(
                f"PatchEmbed failed on an input of shape {shape} ({type(e).__name__}: {e}) — forward "
                f"must accept any H, W that are multiples of the patch size")
        want = _patches_brute(x, pe.proj.weight.detach(), pe.proj.bias.detach(), 4)
        assert tuple(got.shape) == tuple(want.shape), (
            f"input {shape}: expected {tuple(want.shape)}, got {tuple(got.shape)} — forward must "
            f"accept any H, W that are multiples of the patch size")
        if not torch.allclose(got, want, atol=1e-6):
            gh, gw = shape[2] // 4, shape[3] // 4
            extra = (" — your tokens are in COLUMN-major order; flatten(2) of the conv output "
                     "(B, D, h, w) lists row 0 first" if torch.allclose(got, _col_major(want, gh, gw),
                                                                         atol=1e-6) else "")
            raise AssertionError(
                f"input {shape}: token r*(W/p) + c must be the projection of the patch at grid row "
                f"r, column c{extra}")


def check_patchify_linear_matches_conv(ns):
    g = torch.Generator().manual_seed(1)
    for (b, c, h, w, p, d) in ((2, 3, 12, 20, 4, 7), (1, 2, 6, 6, 3, 5), (2, 1, 8, 8, 8, 4)):
        x = torch.randn(b, c, h, w, generator=g, dtype=torch.float64)
        wt = torch.randn(d, c, p, p, generator=g, dtype=torch.float64)
        bias = torch.randn(d, generator=g, dtype=torch.float64)
        want = F.conv2d(x, wt, bias, stride=p).flatten(2).transpose(1, 2)
        got = ns.patchify_linear(x, wt, bias, p)
        assert torch.is_tensor(got) and tuple(got.shape) == (b, (h // p) * (w // p), d), \
            f"patchify_linear should return (B, N, D) = {(b, (h // p) * (w // p), d)}, got " \
            f"{tuple(got.shape) if torch.is_tensor(got) else type(got).__name__}"
        if not torch.allclose(got, want, atol=1e-6):
            hint = ""
            alt = F.conv2d(x, wt.permute(0, 2, 3, 1).reshape(d, c, p, p), bias, stride=p)
            if c > 1 and torch.allclose(got, alt.flatten(2).transpose(1, 2), atol=1e-6):
                hint = (" — your patch vectors and the reshaped weight disagree on the order "
                        "inside a patch: F.unfold gives (channel, row, col), which matches "
                        "weight.reshape(D, -1); a (row, col, channel) patch layout needs "
                        "weight.permute(0, 2, 3, 1).reshape(D, -1)")
            elif torch.allclose(got, _col_major(want, h // p, w // p), atol=1e-6):
                hint = " — tokens are in column-major order; list the patches row by row"
            raise AssertionError(
                f"patchify_linear disagrees with conv2d(x, weight, bias, stride=p) for x "
                f"{(b, c, h, w)}, p={p}{hint}")
    x = torch.randn(1, 3, 8, 8, generator=g)
    wt = torch.randn(4, 3, 4, 4, generator=g)
    got = ns.patchify_linear(x, wt, None, 4)
    assert torch.allclose(got, F.conv2d(x, wt, None, stride=4).flatten(2).transpose(1, 2),
                          atol=1e-5), "bias=None must mean no bias"


def check_patchify_linear_equals_patch_embed_without_conv(ns):
    torch.manual_seed(2)
    pe = _patch_embed(ns, (8, 12), 4, 3, 6)
    x = torch.randn(2, 3, 8, 12)
    with torch.no_grad():
        a = pe(x)
        b = ns.patchify_linear(x, pe.proj.weight, pe.proj.bias, 4)
    assert torch.allclose(a, b, atol=1e-5), \
        "patchify_linear(x, pe.proj.weight, pe.proj.bias, p) must equal pe(x)"
    try:
        from torch.overrides import TorchFunctionMode
    except ImportError:
        raise Skip("needs torch.overrides.TorchFunctionMode (torch >= 2.0)")

    class _Spy(TorchFunctionMode):
        def __init__(self):
            super().__init__()
            self.names = []

        def __torch_function__(self, func, types, args=(), kwargs=None):
            self.names.append(getattr(func, "__name__", str(func)))
            return func(*args, **(kwargs or {}))

    with _Spy() as spy:
        ns.patchify_linear(x, pe.proj.weight.detach(), pe.proj.bias.detach(), 4)
    convs = sorted({n for n in spy.names if "conv" in n.lower()})
    assert not convs, (
        f"patchify_linear called {', '.join(convs)} — the point is to show the conv IS an "
        f"unfold/reshape followed by a matmul")


# ---------------------------------------------------------------- interpolate_pos_embed

def check_interpolate_pos_embed_same_grid_is_identity(ns):
    g = torch.Generator().manual_seed(3)
    pos = torch.randn(1, 1 + 4 * 5, 8, generator=g)
    out = ns.interpolate_pos_embed(pos.clone(), (4, 5), (4, 5), 1)
    assert torch.is_tensor(out) and tuple(out.shape) == (1, 21, 8), \
        f"same grid: expected shape (1, 21, 8), got " \
        f"{tuple(out.shape) if torch.is_tensor(out) else type(out).__name__}"
    assert torch.allclose(out, pos, atol=1e-6), \
        "resampling a 4x5 grid to 4x5 must give the input back"


def check_interpolate_pos_embed_matches_reference(ns):
    g = torch.Generator().manual_seed(4)
    cases = [((4, 4), (6, 6), 1), ((3, 5), (6, 4), 1), ((5, 3), (2, 7), 5), ((4, 6), (8, 9), 0),
             ((7, 7), (3, 3), 1)]
    for old, new, n_pre in cases:
        d = 6
        pos = torch.randn(1, n_pre + old[0] * old[1], d, generator=g, dtype=torch.float64)
        before = pos.clone()
        got = ns.interpolate_pos_embed(pos, old, new, n_pre)
        assert torch.equal(pos, before), "interpolate_pos_embed modified its input in place"
        want_shape = (1, n_pre + new[0] * new[1], d)
        assert torch.is_tensor(got) and tuple(got.shape) == want_shape, (
            f"{old} -> {new} with {n_pre} prefix tokens: expected shape {want_shape}, got "
            f"{tuple(got.shape) if torch.is_tensor(got) else type(got).__name__}")
        assert torch.allclose(got[:, :n_pre].double(), pos[:, :n_pre], rtol=1e-6, atol=1e-7), (
            f"the {n_pre} prefix (CLS / register) rows must be copied unchanged — interpolate only "
            f"the grid rows that follow them")
        if not _matches_bicubic(got, pos, old, new, n_pre):
            want = _interp_oracle(pos, old, new, n_pre)
            hint = ""
            for what, val in _interp_variants(pos, old, new, n_pre).items():
                if tuple(val.shape) == tuple(got.shape) and torch.allclose(got.double(), val.double(),
                                                                         atol=1e-5):
                    hint = f" — this is what {what} gives"
                    break
            raise AssertionError(
                f"grid {old} -> {new}, {n_pre} prefix tokens: values differ from bicubic resampling "
                f"(align_corners=False; antialias False or True) of the (h, w) grid (max abs diff "
                f"from the antialias=False result {(got.double() - want).abs().max():.2e}){hint}")


# ---------------------------------------------------------------- DropPath

def _drop_path(ns, p):
    m = _build(ns.DropPath, "DropPath(drop_prob)", p)
    assert isinstance(m, nn.Module), "DropPath must be an nn.Module (it needs .train()/.eval())"
    return m


def check_drop_path_identity_in_eval_and_at_zero(ns):
    x = torch.randn(8, 5, 4)
    m = _drop_path(ns, 0.5).eval()
    assert torch.equal(m(x), x), "DropPath must be the identity in eval mode"
    m = _drop_path(ns, 0.0).train()
    assert torch.equal(m(x), x), "DropPath(0.0) must be the identity in train mode too"


def check_drop_path_drops_whole_samples_and_rescales(ns):
    torch.manual_seed(5)
    p, keep = 0.25, 0.75
    m = _drop_path(ns, p).train()
    for shape in ((64, 5, 6), (64, 3, 4, 4), (64, 7)):
        x = torch.rand(*shape) + 0.5                        # no zeros in the input
        out = m(x)
        assert tuple(out.shape) == shape, f"DropPath changed the shape {shape} -> {tuple(out.shape)}"
        flat_x, flat_o = x.reshape(shape[0], -1), out.reshape(shape[0], -1)
        zero = (flat_o == 0).all(1)
        kept = torch.isclose(flat_o, flat_x / keep, rtol=1e-6, atol=0).all(1)
        if not (zero | kept).all():
            b = int((~(zero | kept)).nonzero()[0])
            if torch.isclose(flat_o[b], flat_x[b], rtol=1e-6).all():
                raise AssertionError(
                    f"input {shape}: sample {b} was kept but not rescaled — divide kept samples by "
                    f"keep_prob = 1 - drop_prob so the expected output equals the input")
            raise AssertionError(
                f"input {shape}: sample {b} is neither all zeros nor x[{b}] / keep_prob — DropPath "
                f"drops a WHOLE sample (mask shape (B, 1, ..., 1)), it is not element-wise dropout")
        assert zero.any() and kept.any(), \
            f"input {shape} with drop_prob={p}: expected some samples dropped and some kept"
    m1 = _drop_path(ns, 1.0).train()
    out = m1(torch.rand(6, 3) + 0.5)
    assert torch.isfinite(out).all() and (out == 0).all(), \
        "DropPath(1.0) in train mode must zero every sample (and not divide by keep_prob = 0)"


def check_drop_path_preserves_expectation(ns):
    torch.manual_seed(6)
    p = 0.3
    m = _drop_path(ns, p).train()
    n = 20000
    out = m(torch.ones(n, 3))
    frac = float((out[:, 0] == 0).float().mean())
    # std of the dropped fraction ~ sqrt(p(1-p)/n) = 0.0032; 0.02 is > 6 sigma.
    assert abs(frac - p) < 0.02, (
        f"{frac:.3f} of the samples were dropped with drop_prob={p} — drop with probability "
        f"drop_prob (keep with 1 - drop_prob)")
    mean = float(out.mean())
    # std of the mean ~ sqrt(p / (1 - p) / n) = 0.0046; 0.03 is > 6 sigma.
    assert abs(mean - 1.0) < 0.03, (
        f"E[DropPath(x)] should equal x: the mean output for x = 1 is {mean:.3f} — scale the kept "
        f"samples by 1 / keep_prob")


def check_drop_path_gradient(ns):
    torch.manual_seed(7)
    m = _drop_path(ns, 0.4).train()
    x = (torch.rand(40, 3, 5) + 0.5).requires_grad_()
    out = m(x)
    out.sum().backward()
    assert x.grad is not None, "no gradient reached the DropPath input"
    want = (out.detach() != 0).to(x.dtype) / 0.6
    assert torch.allclose(x.grad, want, rtol=1e-6), \
        "d out / d x must be the (scaled) mask: 1/keep_prob for kept samples, 0 for dropped ones"


# ---------------------------------------------------------------- ViTBlock

def _block(ns, dim=16, heads=4, **kw):
    return _build(ns.ViTBlock, "ViTBlock(dim, num_heads, mlp_ratio=4.0, drop_path=0.0, "
                  "init_values=None)", dim, heads, **kw)


def check_vit_block_shape_and_token_mixing(ns):
    torch.manual_seed(8)
    blk = _block(ns).eval()
    x = torch.randn(2, 5, 16, requires_grad=True)
    out = blk(x)
    assert tuple(out.shape) == (2, 5, 16), \
        f"ViTBlock must map (B, N, D) -> (B, N, D) (batch first), got {tuple(out.shape)}"
    out[:, 0].sum().backward()
    assert x.grad[:, 1:].abs().sum() > 0, (
        "token 0's output does not depend on the other tokens — the attention branch is missing, or "
        "it attends over the batch dimension (nn.MultiheadAttention needs batch_first=True for "
        "(B, N, D) inputs)")


def check_vit_block_is_pre_norm(ns):
    # Pre-norm: every branch sees LayerNorm(x), which ignores adding the same constant to all
    # channels, and the residual stream carries the constant through. So
    # block(x + c) = block(x) + c. A post-norm block LN(x + f(x)) does not satisfy this.
    torch.manual_seed(9)
    blk = _block(ns).double().eval()
    x = torch.randn(3, 6, 16, dtype=torch.float64)
    with torch.no_grad():
        a = blk(x)
        b = blk(x + 3.0)
    assert torch.allclose(b, a + 3.0, atol=1e-6), (
        "block(x + 3) != block(x) + 3: use the PRE-norm residual form x + f(LN(x)) for both "
        "branches (no LayerNorm on the residual stream itself)")


def check_vit_block_layerscale(ns):
    torch.manual_seed(10)
    dim = 16
    x = torch.randn(2, 5, dim)
    zero = _block(ns, dim, init_values=0.0)
    for mode in ("train", "eval"):
        getattr(zero, mode)()
        out = zero(x)
        assert torch.equal(out, x), (
            f"init_values=0.0 ({mode} mode): both residual branches are multiplied by gamma = 0, so "
            f"the block must return x exactly — LayerScale must exist whenever init_values is not "
            f"None (0.0 included), and the residual stream must not be normalised")
    zero.train()
    zero(x).pow(2).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in zero.parameters()), \
        "with gamma = 0 the gammas themselves must still get a gradient — make them nn.Parameters"
    plain = _block(ns, dim)
    scaled = _block(ns, dim, init_values=1.0)
    n_plain = sum(p.numel() for p in plain.parameters())
    n_scaled = sum(p.numel() for p in scaled.parameters())
    assert n_scaled - n_plain == 2 * dim, (
        f"LayerScale adds one learnable per-channel vector of size dim to each branch: expected "
        f"{2 * dim} extra parameters, got {n_scaled - n_plain}")
    res = scaled.load_state_dict(plain.state_dict(), strict=False)
    assert not res.unexpected_keys, (
        f"the block without LayerScale has parameters the one with LayerScale lacks: "
        f"{res.unexpected_keys} — use the same submodule names in both configurations")
    plain.eval(), scaled.eval()
    with torch.no_grad():
        assert torch.allclose(plain(x), scaled(x), atol=1e-6), \
            "init_values=1.0 must give the same output as no LayerScale (gamma = 1)"


def check_vit_block_drop_path_only_in_training(ns):
    torch.manual_seed(11)
    x = torch.randn(4, 5, 16)
    always = _block(ns, drop_path=1.0)
    always.train()
    out = always(x)
    assert torch.equal(out, x), (
        "drop_path=1.0 in train mode drops both branches for every sample, so the block must "
        "return x — apply DropPath to the branch output, not to x + branch")
    never = _block(ns, drop_path=0.0)
    never.load_state_dict(always.state_dict())
    always.eval(), never.eval()
    with torch.no_grad():
        assert torch.allclose(always(x), never(x), atol=1e-6), \
            "in eval mode a block with drop_path=1.0 must equal the same block with drop_path=0.0"
    half = _block(ns, drop_path=0.5).train()
    xb = torch.randn(64, 3, 16)
    with torch.no_grad():
        ob = half(xb)
    unchanged = (ob == xb).all(-1).all(-1).sum()
    assert 0 < unchanged < 64, (
        f"drop_path=0.5: {int(unchanged)} of 64 samples came out unchanged (both branches dropped "
        f"for that sample happens with probability 1/4) — drop per sample, not per batch")


# ---------------------------------------------------------------- ViT

def _vit(ns, **kw):
    cfg = dict(img_size=32, patch_size=8, in_chans=3, num_classes=7, embed_dim=32, depth=2,
               num_heads=4)
    cfg.update(kw)
    return _build(ns.ViT, "ViT(img_size, patch_size, in_chans, num_classes, embed_dim, depth, "
                  "num_heads, mlp_ratio=4.0, drop_path_rate=0.0, init_values=None)", **cfg)


def check_vit_forward_shapes_and_parameters(ns):
    torch.manual_seed(12)
    vit = _vit(ns).eval()
    out = vit(torch.randn(3, 3, 32, 32))
    assert tuple(out.shape) == (3, 7), f"ViT should return (B, num_classes) = (3, 7), got {tuple(out.shape)}"
    for name, shape in (("cls_token", (1, 1, 32)), ("pos_embed", (1, 17, 32))):
        t = getattr(vit, name, None)
        assert isinstance(t, nn.Parameter), f"ViT needs a learnable nn.Parameter `{name}`"
        assert tuple(t.shape) == shape, \
            f"{name} should be {shape} (pos_embed covers CLS + 16 patches), got {tuple(t.shape)}"
    rect = _vit(ns, img_size=(16, 32)).eval()
    assert tuple(rect.pos_embed.shape) == (1, 1 + 2 * 4, 32), \
        f"img_size=(16, 32) with patch 8 is a 2x4 grid: pos_embed (1, 9, 32), got {tuple(rect.pos_embed.shape)}"
    assert tuple(rect(torch.randn(2, 3, 16, 32)).shape) == (2, 7)


def check_vit_classifies_from_cls_token(ns):
    torch.manual_seed(13)
    vit = _vit(ns).eval()
    for name in ("norm", "head"):
        assert isinstance(getattr(vit, name, None), nn.Module), \
            f"ViT needs a final LayerNorm `norm` and a linear classifier `head`"
    seen = {}
    h1 = vit.norm.register_forward_hook(lambda m, i, o: seen.__setitem__("norm", o))
    h2 = vit.head.register_forward_hook(lambda m, i, o: seen.__setitem__("head_in", i[0]))
    try:
        with torch.no_grad():
            vit(torch.randn(2, 3, 32, 32))
    finally:
        h1.remove(), h2.remove()
    assert "norm" in seen and "head_in" in seen, "ViT.forward must call self.norm and self.head"
    normed = seen["norm"]
    cls = normed[:, 0] if normed.dim() == 3 else normed
    assert tuple(seen["head_in"].shape) == tuple(cls.shape) and torch.allclose(seen["head_in"], cls), (
        "the head's input must be the final-LayerNorm output of the CLS token (token 0), not a mean "
        "over tokens or a pre-norm feature")


def check_vit_gradients_reach_every_parameter(ns):
    torch.manual_seed(14)
    vit = _vit(ns, init_values=1e-2)
    vit(torch.randn(2, 3, 32, 32)).sum().backward()
    dead = [n for n, p in vit.named_parameters()
            if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
    assert not dead, f"no gradient reached: {', '.join(dead[:5])} — not on the forward path"


def check_vit_stochastic_depth_decay_rule(ns):
    vit = _vit(ns, depth=4, drop_path_rate=0.3)
    blocks = list(getattr(vit, "blocks", []))
    assert len(blocks) == 4, "ViT.blocks must hold the `depth` ViTBlocks (nn.ModuleList / Sequential)"
    want = torch.linspace(0, 0.3, 4).tolist()
    for i, blk in enumerate(blocks):
        probs = [float(m.drop_prob) for m in blk.modules()
                 if isinstance(m, ns.DropPath) or hasattr(m, "drop_prob")]
        assert all(abs(q - want[i]) < 1e-6 for q in probs) and (probs or want[i] == 0), (
            f"block {i}: DropPath drop_prob {probs or 'none'}, expected {want[i]:.3f} — the rate rises "
            f"linearly from 0 at the first block to drop_path_rate at the last "
            f"(torch.linspace(0, drop_path_rate, depth))")
    torch.manual_seed(15)
    x = torch.randn(8, 3, 32, 32)
    vit.train()
    a, b = vit(x), vit(x)
    assert not torch.allclose(a, b), "with drop_path_rate > 0, two train-mode passes should differ"
    vit.eval()
    with torch.no_grad():
        assert torch.equal(vit(x), vit(x)), "eval mode must be deterministic"


def check_vit_runs_at_a_new_resolution(ns):
    torch.manual_seed(16)
    src = _vit(ns)                                          # 32x32 images, 4x4 grid
    sd = {k: v.detach().clone() for k, v in src.state_dict().items()}
    old_pos = sd["pos_embed"]
    sd["pos_embed"] = ns.interpolate_pos_embed(old_pos, (4, 4), (6, 8), 1)
    assert torch.is_tensor(sd["pos_embed"]) and _matches_bicubic(sd["pos_embed"], old_pos, (4, 4), (6, 8), 1), \
        "interpolate_pos_embed (4, 4) -> (6, 8) disagrees with the bicubic reference"
    dst = _vit(ns, img_size=(48, 64))                       # 6x8 grid
    try:
        dst.load_state_dict(sd)
    except RuntimeError as e:
        raise AssertionError(f"a 48x64 ViT could not load the 32x32 checkpoint after resampling "
                             f"pos_embed to the 6x8 grid ({str(e)[:200]})")
    dst.eval()
    with torch.no_grad():
        out = dst(torch.randn(2, 3, 48, 64))
    assert tuple(out.shape) == (2, 7) and torch.isfinite(out).all(), \
        "the fine-tuning-resolution ViT should produce finite (2, 7) logits"
    # Same grid -> same model: resampling to 4x4 must reproduce the original logits.
    same = _vit(ns)
    sd2 = {k: v.detach().clone() for k, v in src.state_dict().items()}
    sd2["pos_embed"] = ns.interpolate_pos_embed(sd2["pos_embed"], (4, 4), (4, 4), 1)
    same.load_state_dict(sd2)
    src.eval(), same.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(src(x), same(x), atol=1e-5), \
            "resampling pos_embed to the same grid changed the model's output"


CHECKS = [
    check_patch_embed_shapes_and_attributes,
    check_patch_embed_row_major_order,
    check_patchify_linear_matches_conv,
    check_patchify_linear_equals_patch_embed_without_conv,
    check_interpolate_pos_embed_same_grid_is_identity,
    check_interpolate_pos_embed_matches_reference,
    check_drop_path_identity_in_eval_and_at_zero,
    check_drop_path_drops_whole_samples_and_rescales,
    check_drop_path_preserves_expectation,
    check_drop_path_gradient,
    check_vit_block_shape_and_token_mixing,
    check_vit_block_is_pre_norm,
    check_vit_block_layerscale,
    check_vit_block_drop_path_only_in_training,
    check_vit_forward_shapes_and_parameters,
    check_vit_classifies_from_cls_token,
    check_vit_gradients_reach_every_parameter,
    check_vit_stochastic_depth_decay_rule,
    check_vit_runs_at_a_new_resolution,
]
