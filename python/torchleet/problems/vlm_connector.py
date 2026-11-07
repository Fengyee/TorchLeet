"""vlm-connector — getting vision features into an LLM.

Three pieces, each checked by properties rather than stored outputs:

  * MLPProjector (LLaVA-1.5 "mlp2x_gelu"): Linear(vision_dim, llm_dim) -> GELU ->
    Linear(llm_dim, llm_dim). The check finds the solver's two nn.Linear layers
    and recomputes fc2(gelu(fc1(x))) with them, so any initialization passes;
    both the exact (erf) and tanh GELU are accepted.
  * PerceiverResampler (Flamingo): learned latent queries cross-attend to the
    image tokens. Checked properties: output is (B, num_queries, dim) for any
    number of image tokens; no positional information on image tokens
    (permutation invariance); pre-LayerNorm on the image tokens (rescaling them
    leaves the output unchanged); samples do not mix across the batch; the
    latents are a learned parameter that gets gradient. Both K/V variants —
    image tokens only, or Flamingo's [image tokens ; latents] — pass.
  * merge_image_features: scatter image features into the placeholder positions
    (k-th feature -> k-th placeholder of that sample), build LM labels (no shift)
    with image and padding positions set to ignore_index, and raise ValueError
    when a sample's placeholder count differs from the image token count.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

ENTRIES = ["MLPProjector", "PerceiverResampler", "merge_image_features"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "The projector is applied to every image token independently. The resampler "
    "turns a variable number of image tokens into a fixed number by letting "
    "num_queries learned vectors (an nn.Parameter) be the attention queries, with "
    "the image tokens as keys and values.",
    "Resampler layer: LayerNorm the image tokens and the latents, q = W_q(latents), "
    "k/v from the image tokens (Flamingo also appends the latents to k/v), "
    "multi-head attention, residual add, then a pre-LN MLP with a second residual. "
    "Expand the latents over the batch dim; never add positions to the image tokens.",
    "Merge: a (B, L) boolean mask of placeholder positions tells you everything. Check "
    "its per-row count equals T (else raise ValueError), then write the flattened "
    "(B*T, D) image features into a clone of text_embeds through that mask — boolean "
    "indexing visits positions in row-major order, which is exactly 'k-th feature of "
    "sample b -> k-th placeholder of sample b'. Labels = input_ids with placeholder and "
    "attention_mask == 0 positions replaced by ignore_index (no shift).",
]

V_DIM, L_DIM = 24, 40
DIM, NQ, HEADS = 32, 6, 4


# ----------------------------------------------------------------- projector

def check_projector_output_shape(ns):
    torch.manual_seed(0)
    m = ns.MLPProjector(V_DIM, L_DIM)
    for shape in ((2, 7, V_DIM), (1, 1, V_DIM), (3, 16, V_DIM)):
        out = m(torch.randn(*shape))
        want = shape[:-1] + (L_DIM,)
        assert tuple(out.shape) == want, \
            f"MLPProjector({V_DIM}, {L_DIM}) on {shape} should give {want}, got {tuple(out.shape)}"


def check_projector_is_linear_gelu_linear(ns):
    torch.manual_seed(0)
    m = ns.MLPProjector(V_DIM, L_DIM)
    linears = [mod for mod in m.modules() if isinstance(mod, nn.Linear)]
    assert len(linears) == 2, (
        f"LLaVA-1.5's projector has exactly two nn.Linear layers "
        f"(Linear -> GELU -> Linear); found {len(linears)}")
    fc1, fc2 = linears
    assert (fc1.in_features, fc1.out_features) == (V_DIM, L_DIM), (
        f"first Linear should map vision_dim -> llm_dim ({V_DIM} -> {L_DIM}), "
        f"got {fc1.in_features} -> {fc1.out_features}")
    assert (fc2.in_features, fc2.out_features) == (L_DIM, L_DIM), (
        f"second Linear should map llm_dim -> llm_dim ({L_DIM} -> {L_DIM}), "
        f"got {fc2.in_features} -> {fc2.out_features}")
    x = torch.randn(2, 5, V_DIM)
    with torch.no_grad():
        got = m(x)
        ok = any(torch.allclose(got, fc2(F.gelu(fc1(x), approximate=a)), atol=1e-5)
                 for a in ("none", "tanh"))
        linear_only = torch.allclose(got, fc2(fc1(x)), atol=1e-5)
    assert not linear_only, \
        "the projector is affine — put a GELU between the two Linear layers"
    assert ok, ("output is not fc2(GELU(fc1(x))) — the forward pass should be exactly "
                "Linear -> GELU -> Linear, applied to every token independently")


# ----------------------------------------------------------------- resampler

def _resampler(ns, num_layers=1, seed=0):
    torch.manual_seed(seed)
    m = ns.PerceiverResampler(DIM, NQ, HEADS, num_layers=num_layers)
    m.eval()
    return m


def check_resampler_output_count_independent_of_image_tokens(ns):
    m = _resampler(ns)
    for b, n in ((2, 49), (1, 5), (3, 1), (2, 144)):
        out = m(torch.randn(b, n, DIM))
        assert tuple(out.shape) == (b, NQ, DIM), (
            f"{n} image tokens (batch {b}) should be resampled to {(b, NQ, DIM)}, got "
            f"{tuple(out.shape)} — the latent queries, not the image tokens, set the "
            f"output length")


def check_resampler_permutation_invariant(ns):
    """No positional encoding on the image tokens in this simplified version, so
    shuffling the token order must not change the output."""
    m = _resampler(ns, num_layers=2)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 20, DIM, generator=g)
    perm = torch.randperm(20, generator=g)
    with torch.no_grad():
        a, b = m(x), m(x[:, perm])
    assert torch.allclose(a, b, atol=1e-5), (
        f"shuffling the image tokens changed the output (max diff "
        f"{(a - b).abs().max():.2e}). Cross-attention is a set operation over keys/"
        f"values — no positional embeddings on image tokens, no pooling by position, "
        f"no slicing of the first tokens.")


def check_resampler_layernorms_image_tokens(ns):
    """Pre-LN on the image tokens: rescaling every token by a constant cannot move
    the output, because LayerNorm(c * x) == LayerNorm(x)."""
    m = _resampler(ns)
    x = torch.randn(2, 11, DIM, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        base = m(x)
        for c in (0.5, 3.0):
            out = m(x * c)
            assert torch.allclose(base, out, atol=1e-4), (
                f"scaling the image tokens by {c} changed the output (max diff "
                f"{(base - out).abs().max():.2e}) — LayerNorm the image tokens (and the "
                f"latents) before the attention projections, as in Flamingo")


def check_resampler_no_mixing_across_batch(ns):
    m = _resampler(ns)
    x = torch.randn(3, 9, DIM, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        full = m(x)
        single = m(x[1:2])
    assert torch.allclose(full[1:2], single, atol=1e-5), (
        "sample 1's output depends on the other samples in the batch — attention must "
        "run per sample (with nn.MultiheadAttention pass batch_first=True)")


def check_resampler_latents_are_learned_queries(ns):
    m = _resampler(ns)
    lat = getattr(m, "latents", None)
    assert isinstance(lat, nn.Parameter), (
        "store the learned queries as an nn.Parameter named `latents` of shape "
        f"(num_queries, dim) = ({NQ}, {DIM})")
    assert lat.numel() == NQ * DIM and lat.shape[-2:] == (NQ, DIM), \
        f"latents should be ({NQ}, {DIM}), got {tuple(lat.shape)}"
    m.train()
    x = torch.randn(2, 10, DIM, generator=torch.Generator().manual_seed(4))
    out = m(x)
    w = torch.randn(out.shape, generator=torch.Generator().manual_seed(5))
    (out * w).sum().backward()
    assert lat.grad is not None and lat.grad.abs().sum() > 0, \
        "no gradient reached `latents` — they must be the attention queries"
    rows = out.detach()[0]
    assert (rows[1:] - rows[:1]).abs().max() > 1e-4, (
        "all query outputs are identical — initialize the latents randomly (e.g. "
        "torch.randn), not to a constant")
    with torch.no_grad():
        other = m(torch.randn(2, 10, DIM, generator=torch.Generator().manual_seed(6)))
    assert (other - out.detach()).abs().max() > 1e-4, \
        "the output does not depend on the image tokens"


def check_resampler_gradients_reach_every_parameter(ns):
    for layers in (1, 2):
        m = _resampler(ns, num_layers=layers)
        m.train()
        out = m(torch.randn(2, 13, DIM, generator=torch.Generator().manual_seed(7)))
        w = torch.randn(out.shape, generator=torch.Generator().manual_seed(8))
        (out * w).sum().backward()      # a weighted sum: a plain .sum() is ~0 after a LayerNorm
        dead = [n for n, p in m.named_parameters()
                if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
        assert not dead, f"num_layers={layers}: no gradient reached {dead[:4]}"


def check_resampler_num_layers_adds_layers(ns):
    n1 = sum(p.numel() for p in _resampler(ns, 1).parameters())
    n3 = sum(p.numel() for p in _resampler(ns, 3).parameters())
    assert n3 > n1, (
        f"num_layers=3 has {n3} parameters, num_layers=1 has {n1} — stack num_layers "
        f"cross-attention + MLP blocks")


# --------------------------------------------------------------------- merge

IMG_ID, PAD, T, D_EMB = 99, 0, 4, 8


def _merge_inputs(dtype=torch.float32):
    I = IMG_ID
    input_ids = torch.tensor([
        [1, I, I, I, I, 2, 3, 4, 5, PAD, PAD, PAD],        # right-padded
        [PAD, PAD, 6, 7, 8, I, I, I, I, 9, 10, 11],        # left-padded
        [12, 0, 13, 14, 15, 16, 17, I, I, I, I, 18],       # token 0 here is real text
    ])
    attention_mask = torch.tensor([
        [1] * 9 + [0] * 3,
        [0] * 2 + [1] * 10,
        [1] * 12,
    ])
    g = torch.Generator().manual_seed(0)
    text_embeds = torch.randn(3, 12, D_EMB, generator=g).to(dtype)
    # feature k of sample b is the constant 10*b + k + 1 (plus a ramp): unmistakable
    image_features = (torch.arange(3).view(3, 1, 1) * 10 + torch.arange(T).view(1, T, 1) + 1
                      + 0.01 * torch.arange(D_EMB).view(1, 1, D_EMB)).float()
    return input_ids, text_embeds, image_features, attention_mask


def _call_merge(ns, *args, **kw):
    out = ns.merge_image_features(*args, **kw)
    assert isinstance(out, (tuple, list)) and len(out) == 2, \
        "merge_image_features should return (inputs_embeds, labels)"
    return out


def check_merge_puts_kth_feature_at_kth_placeholder(ns):
    ids, emb, feats, mask = _merge_inputs()
    out, _ = _call_merge(ns, ids, emb, feats, IMG_ID, attention_mask=mask)
    assert tuple(out.shape) == tuple(emb.shape), \
        f"inputs_embeds should keep text_embeds' shape {tuple(emb.shape)}, got {tuple(out.shape)}"
    for b in range(ids.shape[0]):
        pos = (ids[b] == IMG_ID).nonzero().flatten().tolist()
        for k, p in enumerate(pos):
            if not torch.allclose(out[b, p], feats[b, k], atol=1e-6):
                seen = out[b, p, 0].item()
                raise AssertionError(
                    f"sample {b}, placeholder #{k} (position {p}) should hold image "
                    f"feature {k} of sample {b} (value {feats[b, k, 0].item():.2f}...), "
                    f"found {seen:.2f}... — features go in order, per sample, at that "
                    f"sample's own placeholder positions")


def check_merge_leaves_text_untouched_and_inputs_unmodified(ns):
    ids, emb, feats, mask = _merge_inputs()
    ids0, emb0, feats0 = ids.clone(), emb.clone(), feats.clone()
    out, _ = _call_merge(ns, ids, emb, feats, IMG_ID, attention_mask=mask)
    text = ids != IMG_ID
    assert torch.equal(out[text], emb0[text]), \
        "text (non-placeholder) positions must keep their text embeddings unchanged"
    assert torch.equal(emb, emb0), \
        "text_embeds was modified in place — work on a clone (callers reuse the tensor)"
    assert torch.equal(ids, ids0) and torch.equal(feats, feats0), \
        "input_ids / image_features were modified in place"


def check_merge_labels(ns):
    ids, emb, feats, mask = _merge_inputs()
    _, labels = _call_merge(ns, ids, emb, feats, IMG_ID, attention_mask=mask)
    want = ids.clone()
    want[ids == IMG_ID] = -100
    want[mask == 0] = -100
    assert torch.is_tensor(labels) and tuple(labels.shape) == tuple(ids.shape), \
        f"labels should be a tensor of shape {tuple(ids.shape)} (no shift)"
    assert labels.dtype == torch.long, f"labels should be int64, got {labels.dtype}"
    if not torch.equal(labels, want):
        if torch.equal(labels[:, :-1], want[:, 1:]) or torch.equal(labels[:, 1:], want[:, :-1]):
            raise AssertionError(
                "labels are shifted — return them aligned with input_ids; the shift "
                "happens later, inside the LM loss")
        bad = (labels != want).nonzero().tolist()[:4]
        raise AssertionError(
            f"labels wrong at (sample, position) {bad}: expected input_ids, with image "
            f"placeholders and attention_mask == 0 positions set to ignore_index "
            f"(token 0 at sample 2 position 1 is real text — use attention_mask, not "
            f"input_ids == pad id, to find padding)")
    _, labels = _call_merge(ns, ids, emb, feats, IMG_ID, attention_mask=mask, ignore_index=-1)
    want1 = torch.where(want == -100, torch.full_like(want, -1), want)
    assert torch.equal(labels, want1), "ignore_index argument is not respected"
    _, labels = _call_merge(ns, ids, emb, feats, IMG_ID)
    want2 = ids.masked_fill(ids == IMG_ID, -100)
    assert torch.equal(labels, want2), \
        "with attention_mask=None only the image placeholder positions are ignored"


def _expect_value_error(fn, what):
    try:
        fn()
    except ValueError:
        return
    except AssertionError as e:
        raise AssertionError(
            f"{what}: raise ValueError, not assert (asserts vanish under python -O): {e}")
    except Exception as e:
        raise AssertionError(
            f"{what}: expected a clear ValueError, got {type(e).__name__}: {e}")
    raise AssertionError(f"{what}: no error raised — the features would be silently misaligned")


def check_merge_raises_on_placeholder_count_mismatch(ns):
    ids, emb, feats, mask = _merge_inputs()
    # sample 0 gets one extra placeholder, sample 2 one fewer: the TOTAL still matches
    ids_a = ids.clone()
    ids_a[0, 5] = IMG_ID
    ids_a[2, 10] = 18
    _expect_value_error(
        lambda: ns.merge_image_features(ids_a, emb, feats, IMG_ID, attention_mask=mask),
        "sample 0 has 5 placeholders and sample 2 has 3 (total still 12 = 3 x 4); "
        "check the count per sample")
    ids_b = ids.masked_fill(ids == IMG_ID, 7)
    ids_b[:, 0] = IMG_ID                      # one placeholder per sample, T = 4 features
    _expect_value_error(
        lambda: ns.merge_image_features(ids_b, emb, feats, IMG_ID, attention_mask=mask),
        "each sample has 1 placeholder but 4 image features")


def check_merge_is_differentiable(ns):
    ids, emb, feats, mask = _merge_inputs()
    emb = emb.clone().requires_grad_(True)
    feats = feats.clone().requires_grad_(True)
    out, _ = _call_merge(ns, ids, emb, feats, IMG_ID, attention_mask=mask)
    assert out.requires_grad, (
        "inputs_embeds is detached from the graph (.detach(), .data or torch.no_grad?) — "
        "the projector upstream is trained through this merge")
    w = torch.randn(out.shape, generator=torch.Generator().manual_seed(9))
    (out * w).sum().backward()
    assert feats.grad is not None, \
        "no gradient reached image_features — the projector could not be trained"
    is_img = ids == IMG_ID
    assert torch.allclose(feats.grad, w[is_img].view_as(feats), atol=1e-6), \
        "gradient w.r.t. image_features is wrong — each feature should receive the " \
        "gradient of exactly its own placeholder position"
    assert emb.grad is not None and torch.allclose(emb.grad[~is_img], w[~is_img], atol=1e-6), \
        "gradient should flow to text_embeds at text positions"
    assert emb.grad[is_img].abs().max() == 0, \
        "text_embeds at placeholder positions are overwritten, so their gradient must be 0"


def check_merge_casts_to_text_dtype(ns):
    """Vision tower in fp32, LLM in bf16: the merged sequence is in the LLM's dtype."""
    ids, emb, feats, mask = _merge_inputs(dtype=torch.bfloat16)
    out, _ = _call_merge(ns, ids, emb, feats.float(), IMG_ID, attention_mask=mask)
    assert out.dtype == torch.bfloat16, (
        f"inputs_embeds should have text_embeds' dtype (bfloat16), got {out.dtype} — "
        f"cast image_features with .to(text_embeds.dtype) before the assignment")
    is_img = ids == IMG_ID
    assert torch.equal(out[is_img], feats.reshape(-1, D_EMB).to(torch.bfloat16)), \
        "image features were not placed correctly in the bfloat16 case"


CHECKS = [
    check_projector_output_shape,
    check_projector_is_linear_gelu_linear,
    check_resampler_output_count_independent_of_image_tokens,
    check_resampler_permutation_invariant,
    check_resampler_layernorms_image_tokens,
    check_resampler_no_mixing_across_batch,
    check_resampler_latents_are_learned_queries,
    check_resampler_gradients_reach_every_parameter,
    check_resampler_num_layers_adds_layers,
    check_merge_puts_kth_feature_at_kth_placeholder,
    check_merge_leaves_text_untouched_and_inputs_unmodified,
    check_merge_labels,
    check_merge_raises_on_placeholder_count_mismatch,
    check_merge_is_differentiable,
    check_merge_casts_to_text_dtype,
]
