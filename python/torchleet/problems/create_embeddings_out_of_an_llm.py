"""create-embeddings-out-of-an-llm — pool a causal LM's hidden states into sentence embeddings.

The notebook runs SmolLM2-135M; grading never downloads it. The graded units are
the two pooling functions, checked on random hidden states whose padded
positions hold huge garbage values (so any leak shows), with right- and
left-padded batches, a length-1 sequence, and integer or bool masks:

  * mean_pool: the mean of each row's real-token states — not of all positions,
    and divided by the real-token count, not seq_len;
  * last_token_pool: the state of each row's last real token, wherever the
    padding is — index len-1 is only right for right padding.

End to end, a tiny randomly initialised causal LM defined here (masked causal
attention; learned absolute positions counted from the mask, as HF's generate()
does for left padding — a plain HF forward numbers positions 0..L-1, which RoPE
models such as SmolLM2 tolerate because attention sees only relative offsets) embeds
each sequence alone and inside right- and left-padded batches: both poolings
must give the same embedding every time.
"""
import math

import torch
import torch.nn as nn

ENTRIES = ["mean_pool", "last_token_pool"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "Both poolings must ignore padding, and attention_mask (1 = real token, 0 = "
    "padding) says where it is. Turn it into a float of shape (batch, seq_len, 1) "
    "so it broadcasts against the (batch, seq_len, hidden) states.",
    "Mean pooling: zero the padded states, sum over the sequence dim, and divide "
    "by each row's number of real tokens (the mask summed over seq_len, clamped "
    "to at least 1) — not by seq_len. Last-token pooling: find one position per "
    "row, then index hidden_states[arange(batch), positions].",
    "The last real token sits at the last index where the mask is 1. With right "
    "padding that is (number of real tokens - 1); with left padding it is always "
    "seq_len - 1. seq_len - 1 minus the position of the first 1 in the mask "
    "flipped along seq_len covers both (argmax returns the first maximum).",
]

GARBAGE = 1e4


def _batch(lengths, L, D, side, seed, mask_dtype=torch.long):
    """Random hidden states (B, L, D) with garbage at padded positions, and the mask."""
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(len(lengths), L, D, generator=g)
    mask = torch.zeros(len(lengths), L, dtype=torch.long)
    for i, n in enumerate(lengths):
        if side == "right":
            mask[i, :n] = 1
        else:
            mask[i, L - n:] = 1
    h = torch.where(mask[..., None].bool(), h, torch.full_like(h, GARBAGE))
    return h, mask.to(mask_dtype)


def _ref_mean(h, mask):
    m = mask.bool()
    return torch.stack([h[i][m[i]].mean(0) for i in range(h.shape[0])])


def _ref_last(h, mask):
    m = mask.bool()
    return torch.stack([h[i][m[i]][-1] for i in range(h.shape[0])])


CASES = [  # (lengths, seq_len, side, mask dtype, description)
    ([7, 3, 5, 1], 7, "right", torch.long, "right padding"),
    ([7, 3, 5, 1], 7, "left", torch.long, "left padding"),
    ([4, 6, 2], 6, "right", torch.bool, "right padding, bool mask"),
    ([4, 6, 2], 6, "left", torch.int32, "left padding, int32 mask"),
    ([5, 5], 5, "right", torch.long, "no padding"),
]


def _out(fn, name, h, mask):
    before = h.clone()
    out = fn(h, mask)
    assert torch.is_tensor(out), f"{name} should return a tensor, got {type(out).__name__}"
    assert torch.equal(h, before), f"{name} modified hidden_states in place"
    return out.detach().float()


def check_mean_pool_averages_real_tokens_only(ns):
    for k, (lengths, L, side, dt, where) in enumerate(CASES):
        h, mask = _batch(lengths, L, 8, side, seed=k, mask_dtype=dt)
        got = _out(ns.mean_pool, "mean_pool", h, mask)
        want = _ref_mean(h, mask)
        assert tuple(got.shape) == tuple(want.shape), \
            f"{where}: expected (batch, hidden) = {tuple(want.shape)}, got {tuple(got.shape)}"
        if torch.allclose(got, want, atol=1e-4):
            continue
        zeroed = torch.where(mask[..., None].bool(), h, torch.zeros_like(h))
        if torch.allclose(got, h.mean(1), rtol=1e-4, atol=1e-2):
            why = "padding positions are averaged in — use the attention mask"
        elif torch.allclose(got, zeroed.sum(1) / L, atol=1e-4):
            why = ("divided by seq_len — divide each row by its number of real tokens "
                   "(the mask summed over seq_len)")
        else:
            why = (f"max abs error {float((got - want).abs().max()):.3g}"
                   + (" — padded positions hold huge values and leak into the result"
                      if float(got.abs().max()) > 100 else ""))
        raise AssertionError(f"{where} (lengths {lengths}): mean_pool is not the mean of "
                             f"the real-token states — {why}")


def check_last_token_pool_finds_the_last_real_token(ns):
    for k, (lengths, L, side, dt, where) in enumerate(CASES):
        h, mask = _batch(lengths, L, 8, side, seed=10 + k, mask_dtype=dt)
        got = _out(ns.last_token_pool, "last_token_pool", h, mask)
        want = _ref_last(h, mask)
        assert tuple(got.shape) == tuple(want.shape), \
            f"{where}: expected (batch, hidden) = {tuple(want.shape)}, got {tuple(got.shape)}"
        if torch.allclose(got, want, atol=1e-6):
            continue
        n = mask.long().sum(1)
        at_len = h[torch.arange(h.shape[0]), (n - 1).clamp_min(0)]
        if side == "left" and torch.equal(got, at_len):
            why = ("index (number of real tokens - 1) is only right for right padding; "
                   "with left padding the last real token is at position seq_len - 1")
        elif side == "right" and torch.equal(got, h[:, -1]):
            why = ("the final position is padding for every shorter sequence under right "
                   "padding — use the last position where the mask is 1")
        else:
            rows = [i for i in range(h.shape[0]) if not torch.equal(got[i], want[i])]
            why = f"rows {rows} hold the wrong token's state"
        raise AssertionError(f"{where} (lengths {lengths}): last_token_pool is not the "
                             f"state of each row's last real token — {why}")


class _TinyCausalLM(nn.Module):
    """Embeddings + one masked causal self-attention block. Positions are counted from
    the mask, so a sequence's real-token states do not depend on its padding."""

    def __init__(self, vocab=40, d=16):
        super().__init__()
        self.d = d
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(64, d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, d))

    def forward(self, ids, mask):
        L = ids.shape[1]
        pos = (mask.long().cumsum(1) - 1).clamp_min(0)
        h = self.tok(ids) + self.pos(pos)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        allowed = torch.ones(L, L, dtype=torch.bool).tril() & mask.bool()[:, None, :]
        allowed |= torch.eye(L, dtype=torch.bool)          # padded rows attend to themselves
        s = (q @ k.transpose(1, 2) / math.sqrt(self.d)).masked_fill(~allowed, float("-inf"))
        h = h + self.proj(s.softmax(-1) @ v)
        return h + self.mlp(h)                             # (B, L, d), pads hold junk too


def _pad(seqs, side, pad_id=0):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        sl = slice(0, len(s)) if side == "right" else slice(L - len(s), L)
        ids[i, sl] = torch.tensor(s)
        mask[i, sl] = 1
    return ids, mask


def check_embeddings_do_not_depend_on_padding(ns):
    """What the notebook needs: a review's embedding is the same alone or in a batch."""
    torch.manual_seed(0)
    lm = _TinyCausalLM().eval()
    g = torch.Generator().manual_seed(1)
    seqs = [torch.randint(1, 40, (n,), generator=g).tolist() for n in (9, 2, 5, 1)]
    with torch.no_grad():
        alone = []
        for s in seqs:
            ids, mask = _pad([s], "right")
            alone.append(lm(ids, mask))
        for name in ("mean_pool", "last_token_pool"):
            fn = getattr(ns, name)
            solo = torch.cat([fn(h, torch.ones(h.shape[:2], dtype=torch.long)).float()
                              for h in alone])
            for side in ("right", "left"):
                ids, mask = _pad(seqs, side)
                batch = fn(lm(ids, mask), mask).float()
                assert tuple(batch.shape) == tuple(solo.shape), \
                    f"{name}: expected {tuple(solo.shape)}, got {tuple(batch.shape)}"
                bad = [i for i in range(len(seqs))
                       if not torch.allclose(batch[i], solo[i], atol=1e-5)]
                assert not bad, (
                    f"{name}: with {side} padding, the embeddings of sequences {bad} "
                    f"(lengths {[len(seqs[i]) for i in bad]}) differ from embedding each "
                    "one on its own — the padding leaks into the pooled vector")


CHECKS = [
    check_mean_pool_averages_real_tokens_only,
    check_last_token_pool_finds_the_last_real_token,
    check_embeddings_do_not_depend_on_padding,
]
