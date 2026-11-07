"""sliding-window-attention.

Self-consistency anchor: with a window wide enough to cover the whole sequence,
sliding-window attention must reduce to full attention. That check uses the
solver's own full_attention, so it holds whatever masking style they chose.

Oracle checks compare against F.scaled_dot_product_attention given the expected
boolean band mask, which pins the 1/sqrt(d) scale, masking before the softmax
(not zeroing weights after it) and the window semantics:
  * bidirectional (default): query i sees key j iff |i - j| <= window_size;
  * causal=True (Mistral-style): additionally j <= i, i.e. i - window_size <= j <= i.
"""
import torch
import torch.nn.functional as F

ENTRIES = ["create_sliding_window_mask", "full_attention", "sliding_window_attention"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Position i may attend to j when |i - j| <= window_size; with causal=True, only "
    "when also j <= i.",
    "Build the mask from pairwise offsets: pos = torch.arange(seq_len), then "
    "(pos[:, None] - pos[None, :]).abs() <= window_size (rows are queries, columns keys). "
    "For causal=True, AND it with pos[:, None] >= pos[None, :].",
    "Masked positions get -inf BEFORE the softmax, not zero after it, and the scores are "
    "scaled by 1/sqrt(d) as in ordinary attention.",
]

B, S, D = 2, 12, 16


def _qkv(seed=0):
    torch.manual_seed(seed)
    return (torch.randn(B, S, D) for _ in range(3))


def _band(n, w, causal=False):
    pos = torch.arange(n)
    rel = pos[:, None] - pos[None, :]           # query i (row) minus key j (column)
    mask = rel.abs() <= w
    return mask & (rel >= 0) if causal else mask


def _call_causal(fn, *args):
    try:
        return fn(*args, causal=True)
    except TypeError as e:
        if "causal" not in str(e):
            raise
        raise AssertionError(
            f"{fn.__name__} should accept causal=True (Mistral-style window: query i "
            f"sees keys i - window_size .. i)") from None


def check_mask_shape_and_dtype(ns):
    m = ns.create_sliding_window_mask(S, 2)
    assert tuple(m.shape) == (S, S), f"mask should be {(S, S)}, got {tuple(m.shape)}"
    assert m.dtype == torch.bool, f"mask should be boolean, got {m.dtype}"


def check_mask_marks_the_right_positions(ns):
    w = 2
    m = ns.create_sliding_window_mask(S, w)
    expected = _band(S, w)
    assert torch.equal(m, expected), (
        "mask does not match |i - j| <= window_size "
        f"({int((m != expected).sum())} positions differ)")


def check_causal_mask_marks_the_right_positions(ns):
    for w in (0, 2, S):
        m = _call_causal(ns.create_sliding_window_mask, S, w)
        expected = _band(S, w, causal=True)
        assert torch.equal(torch.as_tensor(m).bool(), expected), (
            f"causal mask (window_size={w}) should allow query i (row) to see key j "
            f"(column) iff i - window_size <= j <= i "
            f"({int((torch.as_tensor(m).bool() != expected).sum())} positions differ)")


def check_diagonal_always_visible(ns):
    m = ns.create_sliding_window_mask(S, 0)
    assert bool(m.diagonal().all()), "every position must attend to itself"


def check_output_shape(ns):
    q, k, v = _qkv()
    out = ns.sliding_window_attention(q, k, v, window_size=3)
    assert tuple(out.shape) == (B, S, D), \
        f"expected {(B, S, D)}, got {tuple(out.shape)}"


def check_full_attention_matches_sdpa(ns):
    q, k, v = _qkv(3)
    got = ns.full_attention(q, k, v)
    want = F.scaled_dot_product_attention(q, k, v)
    diff = (got - want).abs().max().item()
    assert diff < 1e-5, (
        f"full_attention differs from softmax(q k^T / sqrt(d)) v "
        f"(F.scaled_dot_product_attention), max diff {diff:.2e} — check the 1/sqrt(d) scale")


def check_wide_window_equals_full_attention(ns):
    """Window >= seq_len masks nothing, so it must equal full attention."""
    q, k, v = _qkv(1)
    wide = ns.sliding_window_attention(q, k, v, window_size=S)
    full = ns.full_attention(q, k, v)
    diff = (wide - full).abs().max().item()
    assert torch.allclose(wide, full, atol=1e-5), (
        f"with window_size >= seq_len nothing is masked, so this must equal "
        f"full_attention (max diff {diff:.2e})")


def check_narrow_window_differs_from_full(ns):
    """Guards against ignoring the mask entirely."""
    q, k, v = _qkv(2)
    narrow = ns.sliding_window_attention(q, k, v, window_size=1)
    full = ns.full_attention(q, k, v)
    assert not torch.allclose(narrow, full, atol=1e-4), \
        "window_size=1 gives the same result as full attention — the mask is not applied"


def check_matches_sdpa_with_band_mask(ns):
    q, k, v = _qkv(4)
    for w in (0, 1, 3, 5):
        got = ns.sliding_window_attention(q, k, v, window_size=w)
        want = F.scaled_dot_product_attention(q, k, v, attn_mask=_band(S, w))
        diff = (got - want).abs().max().item()
        assert diff < 1e-5, (
            f"window_size={w}: differs from F.scaled_dot_product_attention with the "
            f"|i - j| <= window_size mask (max diff {diff:.2e}). Scale by 1/sqrt(d) and "
            f"set masked scores to -inf before the softmax (zeroing weights after it "
            f"leaves rows that no longer sum to 1).")


def check_causal_matches_sdpa(ns):
    q, k, v = _qkv(5)
    for w in (0, 2, 4):
        got = _call_causal(ns.sliding_window_attention, q, k, v, w)
        want = F.scaled_dot_product_attention(q, k, v, attn_mask=_band(S, w, causal=True))
        diff = (got - want).abs().max().item()
        assert diff < 1e-5, (
            f"causal=True, window_size={w}: differs from F.scaled_dot_product_attention "
            f"with the mask i - window_size <= j <= i (max diff {diff:.2e})")
    got = _call_causal(ns.sliding_window_attention, q, k, v, S)
    want = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    diff = (got - want).abs().max().item()
    assert diff < 1e-5, (
        f"causal=True with window_size >= seq_len must equal plain causal attention "
        f"(max diff {diff:.2e})")


CHECKS = [check_mask_shape_and_dtype, check_mask_marks_the_right_positions,
          check_causal_mask_marks_the_right_positions,
          check_diagonal_always_visible, check_output_shape,
          check_full_attention_matches_sdpa,
          check_wide_window_equals_full_attention, check_narrow_window_differs_from_full,
          check_matches_sdpa_with_band_mask, check_causal_matches_sdpa]
