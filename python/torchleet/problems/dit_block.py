"""dit-block — a Diffusion Transformer block with adaLN-Zero (Peebles & Xie, 2023).

Graded behaviourally, so nn.MultiheadAttention and hand-written attention both pass:

* adaLN-Zero: the final Linear of adaLN_modulation is zero-initialised, so at init
  the block is exactly the identity for any conditioning c, yet that Linear still
  receives a gradient.
* chunk order and modulate are pinned by *setting the adaLN bias* (weight = 0) to
  chosen values. With scale = -1, modulate(norm(x), shift, -1) == shift for every
  token, so (out - x) must be the same vector for every token and every input: that
  only holds if the chunks are (shift_msa, scale_msa, gate_msa, shift_mlp,
  scale_mlp, gate_mlp), modulate uses (1 + scale), and the branch is gated.
* pre-norm: with only one gate open, (out - x) must be invariant to a per-token
  affine change of x (LayerNorm removes it).
* attention: mixes tokens within a sample and never across samples (catches
  nn.MultiheadAttention without batch_first=True), is permutation equivariant, and
  equals softmax(Q K^T / sqrt(head_dim)) V with the block's own projection weights
  for num_heads heads (located from nn.MultiheadAttention, a fused Linear(D, 3D),
  or separate q/k/v Linear(D, D) layers).
* patchify uses the MAE / DiT layout: row-major patches, features ordered
  (row-in-patch, col-in-patch, channel); unpatchify is its exact inverse.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchleet.runner import Skip

ENTRIES = ["modulate", "TimestepEmbedder", "DiTBlock", "patchify", "unpatchify"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "adaLN-Zero turns the conditioning vector c into six (B, D) vectors with one "
    "Linear(D, 6D) after a SiLU: a shift, scale and gate for the attention branch and "
    "for the MLP branch. Initialise that Linear's weight and bias to zero — every gate "
    "is then 0, so each residual branch contributes nothing and the block starts as "
    "the identity.",
    "Shapes: x is (B, N, D) but shift/scale/gate are (B, D) — unsqueeze(1) so one "
    "vector applies to every token of its sample. modulate is "
    "x * (1 + scale) + shift (the 1 + keeps scale = 0 neutral). Each branch is "
    "x = x + gate * f(modulate(LayerNorm(x), shift, scale)) with a LayerNorm that "
    "has no affine parameters (adaLN supplies them).",
    "Chunk order: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = "
    "adaLN_modulation(c).chunk(6, dim=1). patchify: reshape (B, C, H/p, p, W/p, p), "
    "permute to (B, H/p, W/p, p, p, C), flatten to (B, H*W/p^2, p*p*C); unpatchify "
    "undoes exactly those steps in reverse.",
]

D, HEADS, B, T = 32, 4, 3, 6


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _block(ns, hidden=D, heads=HEADS, mlp_ratio=None, seed=0):
    torch.manual_seed(seed)
    kw = {} if mlp_ratio is None else {"mlp_ratio": mlp_ratio}
    try:
        return ns.DiTBlock(hidden, heads, **kw)
    except TypeError:
        return ns.DiTBlock(hidden_size=hidden, num_heads=heads, **kw)


def _ada_linear(block, hidden=D):
    ada = getattr(block, "adaLN_modulation", None)
    assert ada is not None, \
        "DiTBlock needs an `adaLN_modulation` module: nn.Sequential(nn.SiLU(), nn.Linear(D, 6*D))"
    lin = ada[-1] if isinstance(ada, nn.Sequential) else ada
    assert isinstance(lin, nn.Linear) and lin.in_features == hidden \
        and lin.out_features == 6 * hidden, (
            f"adaLN_modulation must end in nn.Linear({hidden}, {6 * hidden}) producing the six "
            f"modulation vectors, found {lin}")
    assert lin.bias is not None, "the adaLN Linear needs a bias"
    return lin


def _set_bias(block, chunks, hidden=D):
    """Zero the adaLN weight and set its bias to six chunks (scalars or (D,) vectors).
    Then adaLN_modulation(c) == bias for every c."""
    lin = _ada_linear(block, hidden)
    with torch.no_grad():
        lin.weight.zero_()
        lin.bias.copy_(torch.cat([torch.as_tensor(v, dtype=lin.bias.dtype).expand(hidden)
                                  for v in chunks]))


def _randomize(block, seed=1, std=0.5):
    lin = _ada_linear(block)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(lin.weight.shape, generator=g) * std)
        lin.bias.copy_(torch.randn(lin.bias.shape, generator=g) * std)


def _inputs(seed=0, b=B, t=T, d=D):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, t, d, generator=g), torch.randn(b, d, generator=g)


def _run(block, x, c):
    out = block(x, c)
    assert torch.is_tensor(out) and tuple(out.shape) == tuple(x.shape), (
        f"DiTBlock(x, c) must return a tensor of x's shape {tuple(x.shape)}, got "
        f"{tuple(out.shape) if torch.is_tensor(out) else type(out).__name__}")
    return out


def _spread(d):
    """How far (out - x) is from being one constant vector across batch and tokens."""
    d = d.detach()
    return float((d - d[:1, :1]).abs().max())


def _ln(x):
    return F.layer_norm(x, (x.shape[-1],), eps=1e-6)


def _mha(x, wq, bq, wk, bk, wv, bv, wo, bo, heads, scale_dim=None):
    b, n, d = x.shape
    hd = d // heads

    def proj(w, bias):
        return F.linear(x, w, bias).reshape(b, n, heads, hd).transpose(1, 2)

    q, k, v = proj(wq, bq), proj(wk, bk), proj(wv, bv)
    s = q @ k.transpose(-2, -1) / math.sqrt(scale_dim or hd)
    o = (s.softmax(-1) @ v).transpose(1, 2).reshape(b, n, d)
    return F.linear(o, wo, bo)


def _attention_weights(block, d=D):
    """Locate (wq, bq, wk, bk, wv, bv, wo, bo) layouts to try, or None."""
    mods = list(block.modules())
    mha = [m for m in mods if isinstance(m, nn.MultiheadAttention)]
    if mha:
        m = mha[0]
        if m.in_proj_weight is None:
            w = (m.q_proj_weight, m.k_proj_weight, m.v_proj_weight)
        else:
            w = m.in_proj_weight.chunk(3, 0)
        bs = m.in_proj_bias.chunk(3, 0) if m.in_proj_bias is not None else (None,) * 3
        return [(w[0], bs[0], w[1], bs[1], w[2], bs[2], m.out_proj.weight, m.out_proj.bias)]
    lins = [m for m in mods if isinstance(m, nn.Linear)]
    fused = [m for m in lins if m.in_features == d and m.out_features == 3 * d]
    square = [m for m in lins if m.in_features == d and m.out_features == d]
    if len(fused) == 1 and len(square) == 1:
        f, o = fused[0], square[0]
        w = f.weight
        bias = f.bias if f.bias is not None else torch.zeros(3 * d)
        hd = d // HEADS
        # standard (3, heads, hd) layout, as in timm / DiT and nn.MultiheadAttention
        std = (w[:d], bias[:d], w[d:2 * d], bias[d:2 * d], w[2 * d:], bias[2 * d:], o.weight, o.bias)
        # alternative (heads, 3, hd) layout: qkv(x).reshape(B, N, heads, 3 * hd)
        idx = torch.arange(3 * d).reshape(HEADS, 3, hd)
        qi, ki, vi = (idx[:, j].reshape(-1) for j in range(3))
        alt = (w[qi], bias[qi], w[ki], bias[ki], w[vi], bias[vi], o.weight, o.bias)
        return [std, alt]
    if len(square) == 4:
        q, k, v, o = square
        return [(q.weight, q.bias, k.weight, k.bias, v.weight, v.bias, o.weight, o.bias)]
    return None


# --------------------------------------------------------------------------
# modulate
# --------------------------------------------------------------------------
def check_modulate_formula(ns):
    g = torch.Generator().manual_seed(0)
    # B == N on purpose: forgetting unsqueeze(1) broadcasts (B, D) against (N, D)
    # without an error and silently applies sample i's shift to token i.
    x = torch.randn(4, 4, 8, generator=g)
    shift, scale = torch.randn(4, 8, generator=g), torch.randn(4, 8, generator=g)
    got = ns.modulate(x, shift, scale)
    want = x * (1 + scale[:, None]) + shift[:, None]
    assert tuple(got.shape) == (4, 4, 8), f"modulate must keep x's shape (4, 4, 8), got {tuple(got.shape)}"
    if torch.allclose(got, want, atol=1e-6):
        return
    if torch.allclose(got, x * scale[:, None] + shift[:, None], atol=1e-6):
        raise AssertionError(
            "modulate is x * scale + shift; DiT uses x * (1 + scale) + shift so that a "
            "zero scale leaves the normalised activations unchanged")
    if torch.allclose(got, x * (1 + scale) + shift, atol=1e-6):
        raise AssertionError(
            "shift/scale are (B, D) but x is (B, N, D): without unsqueeze(1) they broadcast "
            "over the token axis, so token i gets sample i's modulation")
    raise AssertionError(f"modulate should be x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1) "
                         f"(max diff {(got - want).abs().max():.3e})")


# --------------------------------------------------------------------------
# TimestepEmbedder
# --------------------------------------------------------------------------
def _freq_fn(ns):
    fn = getattr(ns.TimestepEmbedder, "timestep_embedding", None)
    assert fn is not None, (
        "TimestepEmbedder needs a `timestep_embedding(t, dim, max_period=10000)` "
        "(static)method returning the (N, dim) sinusoidal features")

    def call(t, dim, **kw):
        try:
            return fn(t, dim, **kw)
        except TypeError:
            return ns.TimestepEmbedder(8).timestep_embedding(t, dim, **kw)
    return call


def _freq_refs(t, dim, max_period=10000):
    half = dim // 2
    out = {}
    for den, tag in ((half, "/half"), (half - 1, "/(half-1)")):
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float64) / den)
        a = t.double()[:, None] * freqs[None]
        out[f"[cos, sin] {tag}"] = torch.cat([a.cos(), a.sin()], -1)
        out[f"[sin, cos] {tag}"] = torch.cat([a.sin(), a.cos()], -1)
    return out


def check_timestep_frequency_embedding(ns):
    emb = _freq_fn(ns)
    t = torch.tensor([0.0, 1.0, 10.5, 500.0, 999.0])
    got = emb(t, 64)
    assert tuple(got.shape) == (5, 64), \
        f"timestep_embedding(t, 64) for 5 timesteps must be (5, 64), got {tuple(got.shape)}"
    refs = _freq_refs(t, 64)
    match = [k for k, r in refs.items() if torch.allclose(got.double(), r, atol=5e-4)]
    assert match, (
        "timestep_embedding should be the DiT sinusoidal features: freqs = "
        "exp(-ln(max_period) * arange(half) / half), args = t[:, None] * freqs, "
        "emb = cat([cos(args), sin(args)], -1) (fractional t allowed)")
    try:
        got = emb(t, 16, max_period=100)
    except TypeError:
        raise AssertionError("timestep_embedding must accept max_period (default 10000)")
    want = _freq_refs(t, 16, max_period=100)[match[0]]
    assert torch.allclose(got.double(), want, atol=5e-4), \
        "max_period=100 must change the frequency range to exp(-ln(100) * i / half)"
    ints = emb(torch.tensor([3, 7]), 32)
    assert tuple(ints.shape) == (2, 32) and ints.is_floating_point(), \
        "integer (long) timesteps must work too — convert t to float"


def check_timestep_embedder_is_frequency_features_then_mlp(ns):
    torch.manual_seed(0)
    m = ns.TimestepEmbedder(32, 64)
    lins = [mod for mod in m.modules() if isinstance(mod, nn.Linear)]
    assert len(lins) == 2 and (lins[0].in_features, lins[0].out_features) == (64, 32) \
        and (lins[1].in_features, lins[1].out_features) == (32, 32), (
            "TimestepEmbedder(hidden_size=32, freq_dim=64) should hold exactly "
            "Linear(64, 32) -> SiLU -> Linear(32, 32), found "
            f"{[(l.in_features, l.out_features) for l in lins]}")
    t = torch.tensor([0.0, 3.0, 250.5, 999.0])
    out = m(t)
    assert tuple(out.shape) == (4, 32), f"TimestepEmbedder(32)(t) must be (4, 32), got {tuple(out.shape)}"
    want = lins[1](F.silu(lins[0](_freq_fn(ns)(t, 64).float())))
    assert torch.allclose(out, want, atol=1e-5), (
        "forward(t) should be Linear -> SiLU -> Linear applied to "
        "timestep_embedding(t, freq_dim)")
    assert torch.cdist(out, out).add(torch.eye(4) * 1e9).min() > 1e-4, \
        "different timesteps gave the same embedding"
    torch.manual_seed(0)
    first = next(mod for mod in ns.TimestepEmbedder(32).modules() if isinstance(mod, nn.Linear))
    assert first.in_features == 256, f"freq_dim should default to 256, got {first.in_features}"


# --------------------------------------------------------------------------
# DiTBlock
# --------------------------------------------------------------------------
def check_adaln_modulation_is_zero_initialized(ns):
    blk = _block(ns)
    ada = getattr(blk, "adaLN_modulation", None)
    assert isinstance(ada, nn.Sequential) and isinstance(ada[0], nn.SiLU), \
        "adaLN_modulation should be nn.Sequential(nn.SiLU(), nn.Linear(D, 6*D))"
    lin = _ada_linear(blk)
    assert torch.all(lin.weight == 0) and torch.all(lin.bias == 0), (
        "adaLN-Zero: the adaLN Linear's weight AND bias must be initialised to zero "
        "(nn.init.zeros_), so all gates start at 0")


def check_block_is_identity_at_init(ns):
    blk = _block(ns)
    x, c = _inputs()
    for cc in (c, torch.zeros_like(c), 10 * c):
        out = _run(blk, x, cc)
        assert torch.allclose(out, x, atol=1e-6), (
            f"at initialisation the block must return x exactly for any c (max deviation "
            f"{(out - x).abs().max():.3e}): with zero-initialised adaLN every gate is 0, so "
            f"both residual branches are multiplied by 0. Check the zero init and that "
            f"each branch is x + gate * f(...)")


def check_output_depends_on_c_once_adaln_is_trained(ns):
    blk = _block(ns)
    _randomize(blk)
    x, c = _inputs()
    a = _run(blk, x, c)
    b = _run(blk, x, c.flip(0))
    assert (a - b).abs().max() > 1e-3, \
        "with non-zero adaLN weights the output must depend on the conditioning c"
    c2 = c.clone()
    c2[1] += 1.0
    o2 = _run(blk, x, c2)
    assert torch.allclose(o2[0], a[0], atol=1e-6) and torch.allclose(o2[2], a[2], atol=1e-6), \
        "changing c for sample 1 changed the output of other samples — c[i] conditions sample i only"
    assert (o2[1] - a[1]).abs().max() > 1e-4, "changing c[1] must change sample 1's output"


def check_adaln_chunk_order_and_modulation(ns):
    """Bias-only adaLN with scale = -1 makes each branch's input the constant `shift`."""
    x, _ = _inputs(seed=1)
    _, c = _inputs(seed=2)
    x2 = 3.0 * torch.randn_like(x) + 1.0
    s = torch.linspace(-1.0, 1.0, D)
    order = ("chunks are (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) "
             "in that order, modulate is x * (1 + scale) + shift")
    for name, chunks in (("MLP", (0, 0, 0, s, -1, 1)), ("attention", (s, -1, 1, 0, 0, 0))):
        blk = _block(ns)
        _set_bias(blk, chunks)
        d1, d2 = _run(blk, x, c) - x, _run(blk, x2, c) - x2
        assert max(_spread(d1), _spread(torch.cat([d1, d2]))) < 1e-4, (
            f"bias-only adaLN with gate_{'mlp' if name == 'MLP' else 'msa'} = 1, its scale = -1 "
            f"and shift = s (other gates 0) should make the {name} branch see the constant "
            f"input modulate(norm(x), s, -1) = s, so out - x must be the same vector for "
            f"every token and every x (spread {max(_spread(d1), _spread(torch.cat([d1, d2]))):.3e}). "
            f"Check that the {order}")
        assert d1.abs().max() > 1e-5, f"the {name} branch contributed nothing with its gate open"
        gate_idx = 5 if name == "MLP" else 2
        doubled = list(chunks)
        doubled[gate_idx] = 2
        _set_bias(blk, doubled)
        d3 = _run(blk, x, c) - x
        assert torch.allclose(d3, 2 * d1, atol=1e-5), (
            f"doubling gate_{'mlp' if name == 'MLP' else 'msa'} must double the {name} "
            f"branch's contribution — the gate multiplies the branch output: x + gate * f(...)")


def check_both_branches_are_pre_norm(ns):
    """With one gate open and shift = scale = 0, out - x = f(LayerNorm(x)): invariant to
    per-token rescaling and shifting of x."""
    x, c = _inputs(seed=3)
    g = torch.Generator().manual_seed(4)
    a = torch.rand(B, T, 1, generator=g) * 1.5 + 0.5
    b = torch.randn(B, T, 1, generator=g)
    for name, chunks in (("attention", (0, 0, 1, 0, 0, 0)), ("MLP", (0, 0, 0, 0, 0, 1))):
        blk = _block(ns)
        _set_bias(blk, chunks)
        d = _run(blk, x, c) - x
        x2 = a * x + b
        d2 = _run(blk, x2, c) - x2
        assert torch.allclose(d, d2, atol=1e-4), (
            f"the {name} branch is not invariant to a per-token affine change of x "
            f"(max diff {(d - d2).abs().max():.3e}) — it must read LayerNorm(x) (pre-norm, "
            f"applied inside the branch), not x itself, and the norm must not be applied "
            f"after the residual add")


def check_attention_mixes_tokens_within_a_sample_only(ns):
    x, c = _inputs(seed=5)
    xp = x.clone()
    # a random perturbation (a constant shift would be erased by the LayerNorm)
    xp[0, 2] += torch.randn(D, generator=torch.Generator().manual_seed(0))
    blk = _block(ns)
    _set_bias(blk, (0, 0, 1, 0, 0, 0))           # attention branch only
    o, op = _run(blk, x, c), _run(blk, xp, c)
    assert torch.allclose(o[1:], op[1:], atol=1e-6), (
        "perturbing a token of sample 0 changed the output of samples 1..B-1 — attention is "
        "mixing across the batch. nn.MultiheadAttention needs batch_first=True for (B, N, D)")
    others = [i for i in range(T) if i != 2]
    assert (o[0, others] - op[0, others]).abs().max() > 1e-4, (
        "perturbing token 2 did not change any other token of the same sample — the attention "
        "branch must mix information across tokens (softmax over the key/token axis)")
    blk = _block(ns)
    _set_bias(blk, (0, 0, 0, 0, 0, 1))           # MLP branch only
    o, op = _run(blk, x, c), _run(blk, xp, c)
    assert torch.allclose(o[0, others], op[0, others], atol=1e-6), \
        "the MLP branch must act on each token independently"


def check_attention_is_multi_head_softmax_attention(ns):
    """Attention branch output vs softmax(QK^T/sqrt(head_dim))V built from the block's weights."""
    blk = _block(ns)
    layouts = _attention_weights(blk)
    if layouts is None:
        raise Skip("could not locate the attention projections (expected nn.MultiheadAttention, "
                   "one Linear(D, 3D) + Linear(D, D), or four Linear(D, D)); head count not verified")
    _set_bias(blk, (0, 4, 1, 0, 0, 0))           # attention input = 5 * LN(x): peaked softmax
    x, c = _inputs(seed=6)
    with torch.no_grad():
        d = _run(blk, x, c) - x
        h = 5.0 * _ln(x)
        for w in layouts:
            if torch.allclose(d, _mha(h, *w, heads=HEADS), atol=1e-4):
                return
        w = layouts[0]
        if torch.allclose(d, _mha(h, *w, heads=1), atol=1e-4) or \
                torch.allclose(d, _mha(h, *w, heads=1, scale_dim=D // HEADS), atol=1e-4):
            raise AssertionError(f"the attention behaves like a single head — split D = {D} into "
                                 f"num_heads = {HEADS} heads of size {D // HEADS}")
        if torch.allclose(d, _mha(h, *w, heads=HEADS, scale_dim=D), atol=1e-4):
            raise AssertionError(f"scores are scaled by 1/sqrt(D = {D}); scale by "
                                 f"1/sqrt(head_dim = {D // HEADS})")
        if torch.allclose(d, _mha(h, *w, heads=HEADS, scale_dim=1), atol=1e-4):
            raise AssertionError("scores are not scaled — divide Q K^T by sqrt(head_dim)")
    raise AssertionError(
        "the attention branch does not match multi-head softmax(Q K^T / sqrt(head_dim)) V "
        "followed by the output projection, using the block's own q/k/v/out weights "
        "(softmax over the key axis; with a fused qkv Linear, q, k, v are its first, "
        "second and third D outputs)")


def check_permutation_equivariance_over_tokens(ns):
    blk = _block(ns)
    _randomize(blk, seed=7)
    x, c = _inputs(seed=8)
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    a = _run(blk, x, c)[:, perm]
    b = _run(blk, x[:, perm], c)
    assert torch.allclose(a, b, atol=1e-5), (
        "permuting the tokens must permute the output the same way — the block has no "
        "positional information (positions are added once, before the first block) and no mask")


def check_gradients_reach_every_parameter(ns):
    blk = _block(ns)
    x, c = _inputs(seed=9)
    r = torch.randn(x.shape, generator=torch.Generator().manual_seed(1))
    (_run(blk, x, c) * r).sum().backward()
    lin = _ada_linear(blk)
    assert lin.weight.grad is not None and lin.weight.grad.abs().max() > 0, (
        "at init the adaLN Linear must still get a gradient (d out / d gate = branch output "
        "is non-zero) — otherwise the zero-initialised gates could never open")
    blk.zero_grad()
    _randomize(blk, seed=10)
    (_run(blk, x, c) * r).sum().backward()
    # Biases may legitimately get an exactly-zero gradient (a key bias shifts every score
    # in a row equally, which softmax ignores), so only weight matrices must be non-zero.
    dead = [n for n, p in blk.named_parameters() if p.requires_grad and
            (p.grad is None or (p.dim() >= 2 and torch.all(p.grad == 0)))]
    assert not dead, f"no gradient reached {', '.join(dead[:4])} once the gates are non-zero"


def check_structure_mlp_ratio_norms_and_gelu(ns):
    def shapes(blk):
        return {(m.in_features, m.out_features) for m in blk.modules() if isinstance(m, nn.Linear)}

    s = shapes(_block(ns))
    assert (D, 4 * D) in s and (4 * D, D) in s, \
        f"with the default mlp_ratio=4.0 the MLP should be Linear({D}, {4 * D}) -> GELU -> Linear({4 * D}, {D})"
    blk = _block(ns, mlp_ratio=2.0)
    s = shapes(blk)
    assert (D, 2 * D) in s and (2 * D, D) in s, \
        f"mlp_ratio=2.0 must give an MLP hidden width of int({D} * 2.0) = {2 * D}"
    for m in blk.modules():
        if isinstance(m, nn.LayerNorm):
            assert not m.elementwise_affine, (
                "the block's LayerNorms must have elementwise_affine=False — adaLN provides "
                "the per-sample scale and shift instead")
            assert abs(m.eps - 1e-6) < 1e-12, f"DiT uses LayerNorm eps=1e-6, found {m.eps}"
        if isinstance(m, nn.GELU):
            assert m.approximate == "tanh", "DiT's MLP uses nn.GELU(approximate='tanh')"


# --------------------------------------------------------------------------
# patchify / unpatchify
# --------------------------------------------------------------------------
def check_patchify_is_row_major_channel_last(ns):
    g = torch.Generator().manual_seed(0)
    imgs = torch.randn(2, 3, 8, 12, generator=g)
    p = 4
    tok = ns.patchify(imgs, p)
    assert tuple(tok.shape) == (2, 6, p * p * 3), (
        f"(2, 3, 8, 12) images with patch_size 4 give a 2 x 3 grid of patches, each with "
        f"4*4*3 = 48 values: expected (2, 6, 48), got {tuple(tok.shape)}")
    for i in range(2):
        for j in range(3):
            block = imgs[:, :, i * p:(i + 1) * p, j * p:(j + 1) * p]        # (B, C, p, p)
            got = tok[:, i * 3 + j]
            if torch.equal(got, block.permute(0, 2, 3, 1).reshape(2, -1)):
                continue
            if torch.equal(got, block.reshape(2, -1)):
                raise AssertionError(
                    "features inside a token are ordered (channel, row, col) — the "
                    "F.unfold / Conv2d-weight layout. Use (row-in-patch, col-in-patch, "
                    "channel) as in MAE/DiT: permute to (B, H/p, W/p, p, p, C) before flattening")
            hint = ""
            if torch.equal(tok[:, j * 2 + i], block.permute(0, 2, 3, 1).reshape(2, -1)):
                hint = " — patches are in column-major order; token k must be i * (W/p) + j"
            raise AssertionError(
                f"token {i * 3 + j} should hold the pixels [{i * p}:{(i + 1) * p}, "
                f"{j * p}:{(j + 1) * p}] (patch row {i}, col {j}, row-major), flattened as "
                f"(row, col, channel){hint}")


def check_unpatchify_inverts_patchify(ns):
    g = torch.Generator().manual_seed(1)
    for shape, p in (((2, 3, 8, 12), 4), ((2, 3, 8, 12), 2), ((1, 1, 4, 4), 4), ((3, 4, 6, 6), 3)):
        imgs = torch.randn(*shape, generator=g)
        tok = ns.patchify(imgs, p)
        back = ns.unpatchify(tok, p, shape[1], shape[2], shape[3])
        assert tuple(back.shape) == shape, \
            f"unpatchify(tokens, {p}, {shape[1]}, {shape[2]}, {shape[3]}) must return {shape}, got {tuple(back.shape)}"
        assert torch.equal(back, imgs), \
            f"unpatchify(patchify(x)) != x for x of shape {shape}, patch_size {p}"
        t2 = torch.randn(tok.shape, generator=g)
        assert torch.equal(ns.patchify(ns.unpatchify(t2, p, shape[1], shape[2], shape[3]), p), t2), \
            "patchify(unpatchify(tokens)) must give the tokens back"


CHECKS = [
    check_modulate_formula,
    check_timestep_frequency_embedding,
    check_timestep_embedder_is_frequency_features_then_mlp,
    check_adaln_modulation_is_zero_initialized,
    check_block_is_identity_at_init,
    check_output_depends_on_c_once_adaln_is_trained,
    check_adaln_chunk_order_and_modulation,
    check_both_branches_are_pre_norm,
    check_attention_mixes_tokens_within_a_sample_only,
    check_attention_is_multi_head_softmax_attention,
    check_permutation_equivariance_over_tokens,
    check_gradients_reach_every_parameter,
    check_structure_mlp_ratio_norms_and_gelu,
    check_patchify_is_row_major_channel_last,
    check_unpatchify_inverts_patchify,
]
