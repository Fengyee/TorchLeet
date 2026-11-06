"""Regression test for the property-based grader.

The invariant that matters: a CORRECT solution written differently from ours must
pass. An earlier design diffed user output against a bundled reference, which
fails that case for anything involving model construction. If this file ever goes
red on a `divergent` case, the grader has regressed to output-diffing.

Run: python3 test_grader.py     (needs torch; no test framework)
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from torchleet import check  # noqa: E402


# ------------------------------------------------------------------ kv-cache
class KVCache:
    def __init__(self):
        self.k = self.v = None

    def update(self, nk, nv):
        self.k = nk if self.k is None else torch.cat([self.k, nk], dim=2)
        self.v = nv if self.v is None else torch.cat([self.v, nv], dim=2)
        return self.k, self.v

    def reset(self):
        self.k = self.v = None


class Attn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.d_model, self.num_heads, self.d_head = d, h, d // h
        for n in "qkvo":
            setattr(self, f"W_{n}", nn.Linear(d, d, bias=False))

    def forward(self, x, kv_cache=None):
        B, S, _ = x.shape
        q, k, v = (t.view(B, S, self.num_heads, self.d_head).transpose(1, 2)
                   for t in (self.W_q(x), self.W_k(x), self.W_v(x)))
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)
        s = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        L = k.shape[2]                       # cached + new positions
        if S > 1:                            # a multi-token chunk (prefill) is causal,
            m = torch.ones(S, L).tril(diagonal=L - S).bool()   # offset by the cache length
            s = s.masked_fill(~m, float("-inf"))
        o = (s.softmax(-1) @ v).transpose(1, 2).reshape(B, S, self.d_model)
        return self.W_o(o)


class ListKVCache(KVCache):
    """Correct, but stores a list and concatenates on demand."""

    def __init__(self):
        self._k, self._v = [], []

    def update(self, nk, nv):
        self._k.append(nk)
        self._v.append(nv)
        return torch.cat(self._k, dim=2), torch.cat(self._v, dim=2)

    def reset(self):
        self._k, self._v = [], []


class SdpaAttn(Attn):
    """Correct, but xavier init + F.linear + fused SDPA."""

    def __init__(self, d, h):
        super().__init__(d, h)
        for n in "qkvo":
            nn.init.xavier_uniform_(getattr(self, f"W_{n}").weight)

    def forward(self, x, kv_cache=None):
        B, S, _ = x.shape
        q, k, v = (F.linear(x, getattr(self, f"W_{n}").weight)
                   .reshape(B, S, self.num_heads, self.d_head).permute(0, 2, 1, 3)
                   for n in "qkv")
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)
        L = k.shape[2]
        # is_causal=True aligns the mask top-left, which is wrong once a cache
        # makes L > S; pass the bottom-right-aligned mask explicitly.
        mask = torch.ones(S, L, dtype=torch.bool).tril(diagonal=L - S)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        o = o.permute(0, 2, 1, 3).reshape(B, S, self.d_model)
        return F.linear(o, self.W_o.weight)


class NoPastAttn(Attn):
    """Broken: updates the cache then ignores the past."""

    def forward(self, x, kv_cache=None):
        B, S, _ = x.shape
        q, k, v = (t.view(B, S, self.num_heads, self.d_head).transpose(1, 2)
                   for t in (self.W_q(x), self.W_k(x), self.W_v(x)))
        if kv_cache is not None:
            kv_cache.update(k, v)
        s = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        if kv_cache is None:
            s = s.masked_fill(~torch.tril(torch.ones(S, S)).bool(), float("-inf"))
        o = (s.softmax(-1) @ v).transpose(1, 2).reshape(B, S, self.d_model)
        return self.W_o(o)


class DecodeOnlyMaskAttn(Attn):
    """Broken: masks only when there is no cache, so a multi-token prefill into
    a cache attends to future tokens (the bug the original solution had)."""

    def forward(self, x, kv_cache=None):
        B, S, _ = x.shape
        q, k, v = (t.view(B, S, self.num_heads, self.d_head).transpose(1, 2)
                   for t in (self.W_q(x), self.W_k(x), self.W_v(x)))
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)
        s = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        if kv_cache is None:
            s = s.masked_fill(~torch.tril(torch.ones(S, S)).bool(), float("-inf"))
        o = (s.softmax(-1) @ v).transpose(1, 2).reshape(B, S, self.d_model)
        return self.W_o(o)


# ------------------------------------------------------------------- softmax
def online_softmax(x):
    m = torch.full((x.shape[0], 1), float("-inf"))
    s = torch.zeros(x.shape[0], 1)
    for j in range(x.shape[1]):
        xj = x[:, j:j + 1]
        mn = torch.maximum(m, xj)
        s = s * torch.exp(m - mn) + torch.exp(xj - mn)
        m = mn
    return torch.exp(x - m) / s


def unstable_softmax(x):
    e = torch.exp(x)
    return e / e.sum(-1, keepdim=True)


# ----------------------------------------------------------------------- cnn
class Cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(3, 32, 3, 1, 1), nn.Conv2d(32, 64, 3, 1, 1)
        self.p, self.f1, self.f2 = nn.MaxPool2d(2, 2), nn.Linear(64 * 16 * 16, 128), nn.Linear(128, 10)

    def forward(self, x):
        x = self.p(F.relu(self.c2(F.relu(self.c1(x)))))
        return self.f2(F.relu(self.f1(x.flatten(1))))


class DivergentCnn(nn.Module):
    """Correct, but a completely different architecture."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=2), nn.BatchNorm2d(16), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(4),
            nn.Flatten(), nn.Linear(32 * 16, 10))

    def forward(self, x):
        return self.net(x)


class WrongClassCount(Cnn):
    def __init__(self):
        super().__init__()
        self.f2 = nn.Linear(128, 100)


# ---------------------------------------------------------------------- rope
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_apply(q, k, cos, sin):
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def rope_noop(q, k, cos, sin):
    """Broken: ignores cos/sin and returns the inputs untouched."""
    return q, k


def rope_query_only(q, k, cos, sin):
    """Broken: rotates the query but forgets the key."""
    return q * cos + rotate_half(q) * sin, k


def rotate_every_two(x):
    """Correct, but the interleaved (RoFormer / GPT-J) pairing: (x_2i, x_2i+1)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def rope_apply_interleaved(q, k, cos, sin):
    return q * cos + rotate_every_two(q) * sin, k * cos + rotate_every_two(k) * sin


# ----------------------------------------------------------------------- gqa
def _gqa(q, k, v, num_query_heads, num_query_groups, W_q, W_k, W_v, W_o, mask, expand):
    if num_query_heads % num_query_groups:
        raise ValueError("num_query_groups must divide num_query_heads")
    B, S, D = q.shape
    dh, g = D // num_query_heads, num_query_heads // num_query_groups
    Q = F.linear(q, W_q).view(B, S, num_query_heads, dh).transpose(1, 2)
    K = expand(F.linear(k, W_k).view(B, S, num_query_groups, dh).transpose(1, 2), g)
    V = expand(F.linear(v, W_v).view(B, S, num_query_groups, dh).transpose(1, 2), g)
    s = Q @ K.transpose(-2, -1) / math.sqrt(dh)
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf"))
    o = (s.softmax(-1) @ V).transpose(1, 2).reshape(B, S, D)
    return F.linear(o, W_o)


def gqa(q, k, v, nh, ng, W_q, W_k, W_v, W_o, mask=None):
    """K/V head j serves query heads j*g .. j*g + g - 1."""
    return _gqa(q, k, v, nh, ng, W_q, W_k, W_v, W_o, mask,
                lambda t, g: t.repeat_interleave(g, dim=1))


def gqa_tiled(q, k, v, nh, ng, W_q, W_k, W_v, W_o, mask=None):
    """Broken: .repeat tiles the K/V heads, so query head h gets K/V head h % ng."""
    return _gqa(q, k, v, nh, ng, W_q, W_k, W_v, W_o, mask,
                lambda t, g: t.repeat(1, g, 1, 1))


# --------------------------------------------------------- gradient checkpointing
class CkptFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fn, *args):
        ctx.fn = fn
        ctx.rng_state = torch.get_rng_state()   # so dropout in fn replays the same mask
        ctx.save_for_backward(*args)
        with torch.no_grad():
            return fn(*args)

    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs = [t.detach().requires_grad_(t.requires_grad) for t in ctx.saved_tensors]
        with torch.random.fork_rng(devices=[]), torch.enable_grad():
            torch.set_rng_state(ctx.rng_state)
            out = ctx.fn(*inputs)
        outs = out if isinstance(out, tuple) else (out,)
        pairs = [(o, g) for o, g in zip(outs, grad_outputs) if o.requires_grad]
        if pairs:
            torch.autograd.backward([o for o, _ in pairs], [g for _, g in pairs])
        return (None, *(t.grad if t.requires_grad else None for t in inputs))


def ckpt(fn, *args):
    if torch.is_grad_enabled() and not any(
            isinstance(a, torch.Tensor) and a.requires_grad for a in args):
        dummy = torch.empty(0, requires_grad=True)
        return CkptFn.apply(lambda _d, *a: fn(*a), dummy, *args)
    return CkptFn.apply(fn, *args)


def ckpt_no_dummy(fn, *args):
    """Broken: with a plain (no-grad) input backward never runs, so fn never trains."""
    return CkptFn.apply(fn, *args)


class CkptFnInputsOnly(CkptFn):
    """Broken: autograd.grad w.r.t. the inputs, so fn's parameters get nothing."""

    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs = [t.detach().requires_grad_(True) for t in ctx.saved_tensors]
        with torch.enable_grad():
            out = ctx.fn(*inputs)
        grads = torch.autograd.grad(out, inputs, grad_outputs, allow_unused=True)
        return (None, *grads)


def ckpt_inputs_only(fn, *args):
    return CkptFnInputsOnly.apply(fn, *args)


# ------------------------------------------------------------------ sequence-packing
def _pack(seqs, max_len, pad_id, order):
    for t in seqs:
        if len(t) == 0 or len(t) > max_len:
            raise ValueError("every sequence needs 1..max_len tokens")
    bins = []                                          # [used, [doc indices]]
    for i in order:
        for b in bins:
            if b[0] + len(seqs[i]) <= max_len:
                b[0] += len(seqs[i]); b[1].append(i); break
        else:
            bins.append([len(seqs[i]), [i]])
    ids, pos, doc, seg = [], [], [], []
    for used, members in bins:
        pad = max_len - used
        ids.append([t for i in members for t in seqs[i].tolist()] + [pad_id] * pad)
        pos.append([p for i in members for p in range(len(seqs[i]))] + [0] * pad)
        doc.append([i for i in members for _ in range(len(seqs[i]))] + [-1] * pad)
        seg += [len(seqs[i]) for i in members]
    mk = lambda x: torch.tensor(x, dtype=torch.long).view(-1, max_len)
    cu = torch.tensor([0] + list(torch.tensor(seg, dtype=torch.long).cumsum(0).tolist()),
                      dtype=torch.int32)
    return mk(ids), mk(pos), mk(doc), cu


def pack_unstable_argsort(seqs, max_len, pad_id=0):
    """FFD whose tie-break among equal lengths is whatever an unstable sort gives."""
    lengths = torch.tensor([len(t) for t in seqs], dtype=torch.long)
    return _pack(seqs, max_len, pad_id, torch.argsort(lengths, descending=True).tolist())


def packed_labels(input_ids, doc_ids):
    nxt = F.pad(input_ids[:, 1:], (0, 1), value=-100)
    nxt_doc = F.pad(doc_ids[:, 1:], (0, 1), value=-2)
    return nxt.masked_fill(~((nxt_doc == doc_ids) & (doc_ids >= 0)), -100)


def document_causal_mask(doc_ids):
    i = torch.arange(doc_ids.shape[1])
    return (doc_ids[:, :, None] == doc_ids[:, None, :]) & (i[:, None] >= i[None, :])


def mask_padding_sees_all(doc_ids):
    """Padding rows attend everywhere (HF-style unmasking) — not this problem's convention."""
    return document_causal_mask(doc_ids) | (doc_ids[:, :, None] < 0)


# ------------------------------------------------------------------ create-embeddings
def mean_pool(h, m):
    m = m.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1)


def last_token_pool_e5(h, m):
    """E5-Mistral's reference pooling: detect left padding, else index the last real token."""
    if m[:, -1].sum() == m.shape[0]:
        return h[:, -1]
    return h[torch.arange(h.shape[0]), m.long().sum(1) - 1]


def last_token_pool_row0(h, m):
    """Decides the padding side from row 0 only."""
    if m[0, -1] == 1:
        return h[:, -1]
    return h[torch.arange(h.shape[0]), m.long().sum(1) - 1]


# ------------------------------------------------------------------ preloaded-dataset
class KeepRawDataset(torch.utils.data.Dataset):
    """Correct: converts once, but also keeps a reference to the caller's arrays."""
    def __init__(self, X, y):
        self.raw = (X, y)
        self.X, self.y = torch.as_tensor(X), torch.as_tensor(y)
        if len(self.X) != len(self.y):
            raise ValueError("X and y lengths differ")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class PerItemDataset(torch.utils.data.Dataset):
    """Wrong: keeps a tensor for len() but converts numpy on every __getitem__."""
    def __init__(self, X, y):
        if len(X) != len(y):
            raise ValueError("X and y lengths differ")
        self.X_raw, self.y_raw = X, y
        self.X = torch.as_tensor(X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.as_tensor(self.X_raw[i]), torch.as_tensor(self.y_raw[i])


CASES = [
    # (label, expected, problem_id, args, kwargs)
    ("kv-cache canonical", True, "kv-cache", (), dict(KVCache=KVCache, CachedAttention=Attn)),
    ("kv-cache divergent", True, "kv-cache", (), dict(KVCache=ListKVCache, CachedAttention=SdpaAttn)),
    ("kv-cache broken", False, "kv-cache", (), dict(KVCache=KVCache, CachedAttention=NoPastAttn)),
    ("kv-cache no prefill mask", False, "kv-cache", (), dict(KVCache=KVCache, CachedAttention=DecodeOnlyMaskAttn)),
    ("softmax correct", True, "triton-fused-softmax", (online_softmax,), {}),
    ("softmax unstable", False, "triton-fused-softmax", (unstable_softmax,), {}),
    ("cnn canonical", True, "cnn", (Cnn,), {}),
    ("cnn divergent", True, "cnn", (DivergentCnn,), {}),
    ("cnn wrong classes", False, "cnn", (WrongClassCount,), {}),
    ("rope correct", True, "rotary-positional-embedding", (rotate_half, rope_apply), {}),
    ("rope no-op", False, "rotary-positional-embedding", (rotate_half, rope_noop), {}),
    ("rope query only", False, "rotary-positional-embedding", (rotate_half, rope_query_only), {}),
    ("rope interleaved (GPT-J)", True, "rotary-positional-embedding",
     (rotate_every_two, rope_apply_interleaved), {}),
    ("gqa canonical", True, "grouped-query-attention", (gqa,), {}),
    ("gqa tiled K/V heads", False, "grouped-query-attention", (gqa_tiled,), {}),
    ("ckpt correct", True, "gradient-checkpointing", (CkptFn, ckpt), {}),
    ("ckpt no dummy input", False, "gradient-checkpointing", (CkptFn, ckpt_no_dummy), {}),
    ("ckpt inputs-only grads", False, "gradient-checkpointing", (CkptFnInputsOnly, ckpt_inputs_only), {}),
    ("packing: unstable FFD tie-break", True, "sequence-packing", (),
     dict(pack_sequences=pack_unstable_argsort, packed_labels=packed_labels,
          document_causal_mask=document_causal_mask)),
    ("packing: padding attends to all", False, "sequence-packing", (),
     dict(pack_sequences=pack_unstable_argsort, packed_labels=packed_labels,
          document_causal_mask=mask_padding_sees_all)),
    ("embeddings: E5 pooling", True, "create-embeddings-out-of-an-llm", (),
     dict(mean_pool=mean_pool, last_token_pool=last_token_pool_e5)),
    ("embeddings: padding side from row 0", False, "create-embeddings-out-of-an-llm", (),
     dict(mean_pool=mean_pool, last_token_pool=last_token_pool_row0)),
    ("preloaded: keeps caller's arrays", True, "preloaded-dataset", (KeepRawDataset,), {}),
    ("preloaded: converts per item", False, "preloaded-dataset", (PerItemDataset,), {}),
]


def main() -> int:
    bad = []
    for label, expected, pid, args, kwargs in CASES:
        got = check(pid, *args, **kwargs)
        if got != expected:
            bad.append(f"{label}: expected {expected}, got {got}")
    print("=" * 60)
    for b in bad:
        print("MISMATCH", b)
    print(f"{len(CASES) - len(bad)}/{len(CASES)} grader cases behaved as expected")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
