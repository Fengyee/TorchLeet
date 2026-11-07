"""grouped-query-attention — GQA shares K/V heads across groups of query heads.

The projection weights are arguments, so the checks compare against exact
oracles: nn.MultiheadAttention when there are as many K/V groups as query heads,
and F.scaled_dot_product_attention(..., enable_gqa=True) otherwise, which uses
the standard grouping — query head h reads K/V head h // (num_query_heads //
num_query_groups), i.e. K/V head j serves the contiguous query heads
j*g .. j*g + g - 1 (repeat_interleave, not repeat).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

ENTRIES = ["grouped_query_attention"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Queries keep all num_query_heads heads; keys and values only have "
    "num_query_groups heads, each shared by a group of g = num_query_heads // "
    "num_query_groups query heads.",
    "W_k and W_v project to num_query_groups * d_head features, not d_model — that "
    "is where the KV-cache saving comes from. Split them into num_query_groups heads.",
    "Expand K/V with repeat_interleave(g, dim=1) so query head h uses K/V head h // g "
    "(a plain repeat/tile would pair it with h % num_query_groups), then attend as "
    "in MHA.",
]

B, S, D_MODEL, HEADS = 2, 6, 32, 4
D_HEAD = D_MODEL // HEADS


def _out(r):
    return r[0] if isinstance(r, tuple) else r


def _weights(groups, seed=0, heads=HEADS):
    torch.manual_seed(seed)
    d_head = D_MODEL // heads
    w = lambda rows: torch.randn(rows, D_MODEL) / D_MODEL ** 0.5
    return w(D_MODEL), w(groups * d_head), w(groups * d_head), w(D_MODEL)


def _qkv(seed=0):
    torch.manual_seed(seed)
    return torch.randn(B, S, D_MODEL), torch.randn(B, S, D_MODEL), torch.randn(B, S, D_MODEL)


def _reference(q, k, v, heads, groups, W_q, W_k, W_v, W_o, keep=None):
    d_head = D_MODEL // heads
    split = lambda x, h: x.view(B, S, h, d_head).transpose(1, 2)
    o = F.scaled_dot_product_attention(
        split(q @ W_q.T, heads), split(k @ W_k.T, groups), split(v @ W_v.T, groups),
        attn_mask=keep, enable_gqa=True)
    return o.transpose(1, 2).reshape(B, S, D_MODEL) @ W_o.T


def _call(ns, q, k, v, groups, w, mask=None, heads=HEADS):
    return _out(ns.grouped_query_attention(q, k, v, heads, groups, *w, mask=mask))


def check_output_shape(ns):
    q, k, v = _qkv()
    for groups in (1, 2, 4):
        out = _call(ns, q, k, v, groups, _weights(groups))
        assert tuple(out.shape) == (B, S, D_MODEL), \
            f"num_query_groups={groups}: expected {(B, S, D_MODEL)}, got {tuple(out.shape)}"


def check_equals_mha_when_groups_equal_heads(ns):
    """num_query_groups == num_query_heads is plain multi-head attention."""
    torch.manual_seed(1)
    mha = nn.MultiheadAttention(D_MODEL, HEADS, bias=False, batch_first=True)
    W_q, W_k, W_v = (w.detach() for w in mha.in_proj_weight.chunk(3))
    w = (W_q, W_k, W_v, mha.out_proj.weight.detach())
    q, k, v = _qkv(1)
    with torch.no_grad():
        got = _call(ns, q, k, v, HEADS, w)
        exp, _ = mha(q, k, v, need_weights=False)
    assert torch.allclose(got, exp, atol=1e-5), (
        "with num_query_groups == num_query_heads, GQA must equal "
        f"nn.MultiheadAttention given the same weights (max diff {(got - exp).abs().max():.2e})")


def check_matches_sdpa_with_grouped_heads(ns):
    """MQA (1 group) and GQA (2 groups) against SDPA(enable_gqa=True)."""
    q, k, v = _qkv(2)
    for groups in (1, 2):
        w = _weights(groups, seed=groups)
        got = _call(ns, q, k, v, groups, w)
        exp = _reference(q, k, v, HEADS, groups, *w)
        assert torch.allclose(got, exp, atol=1e-5), (
            f"num_query_groups={groups}: disagrees with scaled_dot_product_attention("
            f"enable_gqa=True) (max diff {(got - exp).abs().max():.2e}) — query head h "
            f"must use K/V head h // {HEADS // groups} (repeat_interleave, not repeat), "
            "and scores are scaled by 1/sqrt(d_head)")


def check_causal_mask(ns):
    q, k, v = _qkv(3)
    w = _weights(2, seed=3)
    keep = torch.ones(S, S, dtype=torch.bool).tril()
    got = _call(ns, q, k, v, 2, w, mask=keep)
    exp = _reference(q, k, v, HEADS, 2, *w, keep=keep)
    assert torch.allclose(got, exp, atol=1e-5), (
        "with a causal mask (True = may attend) the output disagrees with causal "
        f"attention (max diff {(got - exp).abs().max():.2e}) — blocked scores must be "
        "-inf before the softmax")


def check_groups_must_divide_heads(ns):
    """3 K/V groups cannot evenly serve 4 query heads."""
    q, k, v = _qkv()
    try:
        _call(ns, q, k, v, 3, _weights(3))
    except Exception:
        return
    raise AssertionError(
        "num_query_groups=3 with num_query_heads=4 should be rejected — "
        "every K/V group must serve the same number of query heads")


def check_gradients_flow(ns):
    q, k, v = _qkv(4)
    q.requires_grad_()
    w = [t.requires_grad_() for t in _weights(2, seed=4)]
    _call(ns, q, k, v, 2, w).sum().backward()
    assert q.grad is not None and torch.any(q.grad != 0), \
        "no gradient reached the queries"
    dead = [n for n, t in zip(("W_q", "W_k", "W_v", "W_o"), w)
            if t.grad is None or not torch.any(t.grad != 0)]
    assert not dead, f"no gradient reached {', '.join(dead)}"


CHECKS = [check_output_shape, check_equals_mha_when_groups_equal_heads,
          check_matches_sdpa_with_grouped_heads, check_causal_mask,
          check_groups_must_divide_heads, check_gradients_flow]
