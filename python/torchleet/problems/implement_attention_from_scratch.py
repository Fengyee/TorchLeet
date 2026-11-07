"""implement-attention-from-scratch — scaled dot-product attention.

F.scaled_dot_product_attention is the oracle. The mask follows its boolean
attn_mask convention: True = may attend, False = blocked.
"""
import torch
import torch.nn.functional as F

ENTRIES = ["scaled_dot_product_attention"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "scores = q @ k.transpose(-2,-1) / sqrt(d_k), then softmax, then @ v.",
    "Divide by sqrt(d_k) or the softmax saturates as dimension grows.",
    "Masked positions become -inf BEFORE the softmax, not zero after it.",
]

B, H, S, D = 2, 4, 6, 16


def _qkv(seed=0):
    torch.manual_seed(seed)
    return torch.randn(B, H, S, D), torch.randn(B, H, S, D), torch.randn(B, H, S, D)


def _out(r):
    return r[0] if isinstance(r, tuple) else r


def check_matches_torch_attention(ns):
    q, k, v = _qkv()
    got = _out(ns.scaled_dot_product_attention(q, k, v))
    exp = F.scaled_dot_product_attention(q, k, v)
    assert tuple(got.shape) == tuple(exp.shape), \
        f"expected {tuple(exp.shape)}, got {tuple(got.shape)}"
    assert torch.allclose(got, exp, atol=1e-5), \
        f"disagrees with F.scaled_dot_product_attention (max diff {(got - exp).abs().max():.2e})"


def check_scaling_is_applied(ns):
    """Without the 1/sqrt(d_k) factor the output diverges as d_k grows."""
    torch.manual_seed(1)
    q = torch.randn(1, 1, 4, 64) * 3
    k, v = torch.randn(1, 1, 4, 64) * 3, torch.randn(1, 1, 4, 64)
    got = _out(ns.scaled_dot_product_attention(q, k, v))
    exp = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(got, exp, atol=1e-4), \
        "output diverges on large d_k — is the 1/sqrt(d_k) scaling applied?"


def check_mask_blocks_positions(ns):
    """Masked-out keys must not influence the output."""
    q, k, v = _qkv(2)
    mask = torch.ones(S, S, dtype=torch.bool).tril()
    got = _out(ns.scaled_dot_product_attention(q, k, v, mask=mask))
    exp = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(got, exp, atol=1e-5), (
        "with a lower-triangular mask this must equal causal attention "
        f"(max diff {(got - exp).abs().max():.2e})")


def check_padding_mask_and_cross_attention(ns):
    """A key-padding mask of shape (B, 1, 1, S_k) must broadcast over heads and
    queries, and queries and keys may have different lengths (cross-attention)."""
    torch.manual_seed(3)
    s_q, s_k, d_v = 5, 9, 12
    q, k = torch.randn(B, H, s_q, D), torch.randn(B, H, s_k, D)
    v = torch.randn(B, H, s_k, d_v)
    lengths = torch.tensor([s_k, 4])                     # second sequence: 5 padded keys
    keep = (torch.arange(s_k) < lengths[:, None])[:, None, None, :]
    got = _out(ns.scaled_dot_product_attention(q, k, v, mask=keep))
    exp = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
    assert tuple(got.shape) == (B, H, s_q, d_v), \
        f"expected {(B, H, s_q, d_v)} for S_q={s_q}, S_k={s_k}, d_v={d_v}, got {tuple(got.shape)}"
    assert torch.allclose(got, exp, atol=1e-5), (
        "with a key-padding mask (True = attend) the output disagrees with "
        f"F.scaled_dot_product_attention (max diff {(got - exp).abs().max():.2e})")


CHECKS = [check_matches_torch_attention, check_scaling_is_applied,
          check_mask_blocks_positions, check_padding_mask_and_cross_attention]
