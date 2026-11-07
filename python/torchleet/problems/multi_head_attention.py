"""multi-head-attention.

The projection weights are arguments, so torch.nn.MultiheadAttention is an exact
oracle: its in_proj_weight splits into W_q, W_k, W_v and out_proj.weight is W_o.
The mask follows F.scaled_dot_product_attention's boolean convention (True = may
attend), the opposite of nn.MultiheadAttention's, so the oracle gets ~mask.
"""
import torch
import torch.nn as nn

ENTRIES = ["multi_head_attention"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "Project q, k and v with their weights (x @ W.T), split d_model into num_heads "
    "chunks of d_head = d_model // num_heads, attend within each head, then merge "
    "the heads and apply W_o.",
    "Reshape (batch, seq, d_model) to (batch, num_heads, seq, d_head) with view + "
    "transpose(1, 2), and scale the scores by 1/sqrt(d_head), not 1/sqrt(d_model).",
    "After softmax(scores) @ V, undo the split with transpose(1, 2) and "
    ".contiguous().view(batch, seq_q, d_model) before W_o. Positions where the mask "
    "is False get -inf before the softmax.",
]

B, S_Q, S_K, D_MODEL, HEADS = 2, 5, 7, 32, 4


def _out(r):
    return r[0] if isinstance(r, tuple) else r


def _setup(seed, heads=HEADS, s_q=S_Q, s_k=S_K):
    torch.manual_seed(seed)
    mha = nn.MultiheadAttention(D_MODEL, heads, bias=False, batch_first=True)
    W_q, W_k, W_v = (w.detach().clone() for w in mha.in_proj_weight.chunk(3))
    weights = (W_q, W_k, W_v, mha.out_proj.weight.detach().clone())
    q = torch.randn(B, s_q, D_MODEL)
    k, v = torch.randn(B, s_k, D_MODEL), torch.randn(B, s_k, D_MODEL)
    return mha, weights, q, k, v


def check_output_shape(ns):
    _, w, q, k, v = _setup(0)
    out = _out(ns.multi_head_attention(q, k, v, HEADS, *w))
    assert tuple(out.shape) == (B, S_Q, D_MODEL), (
        f"expected {(B, S_Q, D_MODEL)} for queries of length {S_Q} and keys of "
        f"length {S_K}, got {tuple(out.shape)}")


def check_matches_nn_multihead_attention(ns):
    """Same weights in, same output out — for 1, 4 and 8 heads (cross-attention)."""
    for heads in (1, HEADS, 8):
        mha, w, q, k, v = _setup(heads, heads=heads)
        with torch.no_grad():
            got = _out(ns.multi_head_attention(q, k, v, heads, *w))
            exp, _ = mha(q, k, v, need_weights=False)
        assert torch.allclose(got, exp, atol=1e-5), (
            f"with num_heads={heads}, disagrees with nn.MultiheadAttention given the "
            f"same weights (max diff {(got - exp).abs().max():.2e}) — check the head "
            "split (view + transpose), the 1/sqrt(d_head) scale and the W_o projection")


def check_causal_mask(ns):
    mha, w, x, _, _ = _setup(1, s_q=6, s_k=6)
    keep = torch.ones(6, 6, dtype=torch.bool).tril()        # True = may attend
    with torch.no_grad():
        got = _out(ns.multi_head_attention(x, x, x, HEADS, *w, mask=keep))
        exp, _ = mha(x, x, x, attn_mask=~keep, need_weights=False)
    assert torch.allclose(got, exp, atol=1e-5), (
        "with a causal mask (True = may attend) this must equal causal self-attention "
        f"(max diff {(got - exp).abs().max():.2e}) — blocked scores become -inf "
        "before the softmax")


def check_key_padding_mask(ns):
    """A (batch, 1, 1, seq_k) mask must broadcast over heads and queries."""
    mha, w, q, k, v = _setup(2)
    lengths = torch.tensor([S_K, 3])
    keep = torch.arange(S_K) < lengths[:, None]               # (B, S_K)
    with torch.no_grad():
        got = _out(ns.multi_head_attention(q, k, v, HEADS, *w, mask=keep[:, None, None, :]))
        exp, _ = mha(q, k, v, key_padding_mask=~keep, need_weights=False)
    assert torch.allclose(got, exp, atol=1e-5), (
        "with a key-padding mask of shape (batch, 1, 1, seq_k) the output disagrees "
        f"with nn.MultiheadAttention (max diff {(got - exp).abs().max():.2e})")


def check_gradients_reach_weights(ns):
    _, w, q, k, v = _setup(3)
    w = [t.clone().requires_grad_() for t in w]
    _out(ns.multi_head_attention(q, k, v, HEADS, *w)).sum().backward()
    dead = [n for n, t in zip(("W_q", "W_k", "W_v", "W_o"), w)
            if t.grad is None or not torch.any(t.grad != 0)]
    assert not dead, f"no gradient reached {', '.join(dead)} — is it used on the forward path?"


CHECKS = [check_output_shape, check_matches_nn_multihead_attention,
          check_causal_mask, check_key_padding_mask, check_gradients_reach_weights]
