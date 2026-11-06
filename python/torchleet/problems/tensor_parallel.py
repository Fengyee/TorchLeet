"""tensor-parallel — Megatron-style tensor parallelism, simulated in one process.

W tensor-parallel ranks are Python lists: xs[r] is rank r's copy of the block
input, shards[r] is a dict of rank r's weight shards. The only collective is
`comm.all_reduce(list_of_W_tensors) -> list_of_W_tensors` (SUM). The grader's
communicator (re-created below, same API as the notebook's `SimComm`) counts
every call and records its inputs; like NCCL it is NOT autograd-aware, so the
solver has to wrap it in the conjugate operators of the Megatron paper.

Conventions (and where they come from):

  * nn.Linear layout. Weights are (out_features, in_features) and
    y = x W^T + b (torch.nn.functional.linear). Megatron writes Y = XA with
    A = W^T. "Column-parallel" (A split by columns, A = [A_1, ..., A_p]) is
    therefore a split of W along dim 0, the OUTPUT features; "row-parallel"
    (A split by rows) is a split of W along dim 1, the INPUT features.
    Megatron-LM stores its weights the same way: ColumnParallelLinear holds
    (output_size_per_partition, input_size) and partitions dim 0,
    RowParallelLinear holds (output_size, input_size_per_partition) and
    partitions dim 1 (megatron/core/tensor_parallel/layers.py).
    Shard r is the r-th contiguous block (torch.chunk order); a size that is not
    divisible by world_size raises ValueError (Megatron asserts the same with
    `divide` / `ensure_divisibility` in megatron/core/utils.py).
  * Biases. The column-parallel bias is split with its outputs. The
    row-parallel bias is full size and replicated on every rank; it is added
    ONCE, after the all-reduce (RowParallelLinear.forward:
    output_ = reduce_from_tensor_model_parallel_region(output_parallel);
    output = output_ + self.bias).
  * MLP (Shoeybi et al. 2019, "Megatron-LM", arXiv:1909.08053, Sec. 3 and
    Fig. 3a): Z = act(X W1^T + b1) W2^T + b2 with W1 column-parallel (act runs
    locally — it is elementwise) and W2 row-parallel: one all-reduce in forward.
  * Attention (Fig. 3b): heads are split across ranks — rank r owns the
    contiguous heads [r*H/W, (r+1)*H/W). The dense fused projection is
    w_qkv = [W_q; W_k; W_v] (3d, d), the nn.MultiheadAttention in_proj_weight
    layout, so shard_qkv must split Q, K and V separately and return per-rank
    [W_q,r; W_k,r; W_v,r]. Splitting the fused matrix into W contiguous pieces
    would give rank 0 mostly query rows and the last rank mostly value rows, so
    a rank's q, k, v would not belong to the same heads. (Megatron avoids this
    by storing the fused QKV grouped by head / query group, [ng, (np/ng + 2) *
    hn] in megatron/core/transformer/attention.py, so its plain column split is
    head-aligned.) Attention is causal, softmax scale 1/sqrt(head_dim). The
    output projection is row-parallel: one all-reduce in forward. H % W != 0
    raises ValueError.
  * f and g (paper Sec. 3, "Code 1"): f is identity forward / all-reduce
    backward, g is all-reduce forward / identity backward
    (megatron/core/tensor_parallel/mappings.py: _CopyToModelParallelRegion and
    _ReduceFromModelParallelRegion). Simulation of backward: every rank computes
    the same loss from its replicated output, so the grader sums the W per-rank
    losses and calls backward once. With g's identity backward each rank's
    partial receives exactly dL/dZ, so every shard's gradient equals the matching
    slice of the dense model's gradient and every xs[r].grad equals the dense
    dL/dX (f's backward all-reduce adds the ranks' partial input gradients).
    Plain autograd through a sum would instead give W x the gradients.
  * Communication count: exactly one all-reduce in forward and one in backward
    per MLP and per attention block (Sec. 3: "only two all-reduces in the forward
    path and two in the backward path"; Fig. 4: 4 per transformer layer). With
    world_size 1 the collective is a no-op and may be skipped (Megatron's
    _reduce bypasses it), so counts are only enforced for W >= 2.

Checks: shard shapes/values/independence and ValueError; head-aligned QKV
sharding; f and g forward/backward semantics and comm counts; MLP and attention
outputs equal the dense block for W in {1, 2, 4} (and with the solver's own
sharding functions), the tensor each rank hands to the all-reduce is exactly its
local partial (no bias, no averaging), the output really comes from the
all-reduce result (a tampering communicator), and all shard / input / bias
gradients equal the dense gradients.
"""
import math

import torch
import torch.nn.functional as F

ENTRIES = ["shard_column", "shard_row", "shard_qkv", "copy_to_tp", "reduce_from_tp",
           "tp_mlp_forward", "tp_attention_forward"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Translate Megatron's Y = XA into PyTorch first: nn.Linear stores W = A^T with "
    "shape (out, in). Column-parallel = each rank owns a slice of the OUTPUT features "
    "(rows of W, dim 0, plus that slice of the bias); row-parallel = each rank owns a "
    "slice of the INPUT features (columns of W, dim 1) and produces a partial sum of "
    "the full output, so the bias cannot be split.",
    "f and g are torch.autograd.Functions whose forward takes (ctx, comm, *tensors) and "
    "returns a tuple; backward must return one gradient per forward input, so its first "
    "value is None (for comm). f: forward passes the W tensors through unchanged, "
    "backward makes ONE all-reduce of the W incoming gradients. g: forward makes the one "
    "all-reduce, backward passes the W gradients straight through. comm.all_reduce on "
    "its own is not differentiable — exactly like NCCL.",
    "MLP: run f on the inputs; on each rank apply the local column-parallel linear and "
    "the activation, then multiply by that rank's row-parallel weight WITHOUT a bias; run "
    "g on the W partial sums; add the replicated b2 to each result. Attention: same "
    "skeleton — split the local fused projection along its last dim into this rank's q, "
    "k, v, view them as n_heads // W heads of size d // n_heads, run causal attention, "
    "merge the heads and multiply by the rank's w_o slice. shard_qkv: column-shard W_q, "
    "W_k and W_v (and the three bias thirds) separately, then concatenate rank r's three "
    "pieces.",
]


# ----------------------------------------------------------------- simulator

class _Comm:
    """Same API as the notebook's SimComm; `tamper` is added to every result."""

    def __init__(self, world_size, tamper=0.0):
        self.world_size = world_size
        self.calls = 0
        self.log = []                 # inputs of every call (detached clones)
        self.tamper = tamper

    def all_reduce(self, tensors):
        tensors = list(tensors)
        if len(tensors) != self.world_size:
            raise ValueError(f"all_reduce needs one tensor per rank ({self.world_size}), "
                             f"got {len(tensors)}")
        if any(t.shape != tensors[0].shape for t in tensors):
            raise ValueError("all_reduce: every rank must contribute a tensor of the same "
                             f"shape, got {[tuple(t.shape) for t in tensors]}")
        self.calls += 1
        with torch.no_grad():
            self.log.append([t.detach().clone() for t in tensors])
            total = torch.stack([t.detach() for t in tensors]).sum(0) + self.tamper
            return [total.clone() for _ in tensors]


def _count_ok(calls, want, W):
    return calls == want if W >= 2 else calls <= want


def _bwd_count_msg(block, W, n):
    why = ("is the block input going through copy_to_tp (f)?" if n == 0 else
           "only f's backward communicates — g's backward must be the identity")
    return (f"world_size={W}: the {block} backward made {n} all-reduce call(s), expected "
            f"exactly 1 — {why}")


# ----------------------------------------------------------------- dense references

def _rand(*shape, g):
    return torch.randn(*shape, generator=g, dtype=torch.float64)


def _mlp_params(d, ffn, seed):
    g = torch.Generator().manual_seed(seed)
    return {"w1": _rand(ffn, d, g=g) / math.sqrt(d), "b1": _rand(ffn, g=g),
            "w2": _rand(d, ffn, g=g) / math.sqrt(ffn), "b2": _rand(d, g=g)}


def _attn_params(d, seed):
    g = torch.Generator().manual_seed(seed)
    return {"w_qkv": _rand(3 * d, d, g=g) / math.sqrt(d), "b_qkv": _rand(3 * d, g=g),
            "w_o": _rand(d, d, g=g) / math.sqrt(d), "b_o": _rand(d, g=g)}


def _dense_mlp(x, p, act):
    return act(x @ p["w1"].T + p["b1"]) @ p["w2"].T + p["b2"]


def _heads_attention(q, k, v):
    """q, k, v: (B, h, T, hd). Causal softmax attention written out by hand."""
    T = q.shape[-2]
    s = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    return s.masked_fill(~causal, float("-inf")).softmax(-1) @ v


def _attn_core(x, w_qkv, b_qkv, h):
    B, T, _ = x.shape
    q, k, v = (x @ w_qkv.T + b_qkv).chunk(3, dim=-1)
    hd = q.shape[-1] // h
    q, k, v = (t.reshape(B, T, h, hd).transpose(1, 2) for t in (q, k, v))
    return _heads_attention(q, k, v).transpose(1, 2).reshape(B, T, h * hd)


def _dense_attention(x, p, n_heads):
    return _attn_core(x, p["w_qkv"], p["b_qkv"], n_heads) @ p["w_o"].T + p["b_o"]


# ----------------------------------------------------------------- reference sharding

def _ref_mlp_shards(p, W):
    w1, b1 = p["w1"].chunk(W, 0), p["b1"].chunk(W, 0)
    w2 = p["w2"].chunk(W, 1)
    return [{"w1": w1[r].clone(), "b1": b1[r].clone(), "w2": w2[r].clone(),
             "b2": p["b2"].clone()} for r in range(W)]


def _ref_qkv(w_qkv, b_qkv, W):
    wq, wk, wv = (t.chunk(W, 0) for t in w_qkv.chunk(3, 0))
    bq, bk, bv = (t.chunk(W, 0) for t in b_qkv.chunk(3, 0))
    return [(torch.cat([wq[r], wk[r], wv[r]]), torch.cat([bq[r], bk[r], bv[r]]))
            for r in range(W)]


def _ref_attn_shards(p, W):
    qkv = _ref_qkv(p["w_qkv"], p["b_qkv"], W)
    wo = p["w_o"].chunk(W, 1)
    return [{"w_qkv": qkv[r][0].clone(), "b_qkv": qkv[r][1].clone(),
             "w_o": wo[r].clone(), "b_o": p["b_o"].clone()} for r in range(W)]


def _grad_leaves(shards):
    return [{k: v.detach().clone().requires_grad_() for k, v in s.items()} for s in shards]


def _as_list(out, W, fn):
    assert out is not None, f"{fn} returned None — return the list of W per-rank outputs"
    out = list(out)
    assert len(out) == W and all(torch.is_tensor(o) for o in out), (
        f"{fn} should return a list of {W} tensors (rank r's copy of the block output), "
        f"got {len(out)} item(s)")
    return out


def _close(a, b, tol=1e-10):
    return a.shape == b.shape and torch.allclose(a, b, rtol=tol, atol=tol)


def _same_values(got, want):
    """Like _close, but a flattened (e.g. (B*T, d)) all-reduce payload is fine too."""
    return got.numel() == want.numel() and _close(got.reshape(want.shape), want)


# ----------------------------------------------------------------- sharding

def check_shard_column_and_shard_row(ns):
    g = torch.Generator().manual_seed(0)
    for W in (1, 2, 4):
        weight, bias = _rand(8, 12, g=g), _rand(8, g=g)
        keep_w, keep_b = weight.clone(), bias.clone()
        out = ns.shard_column(weight, bias, W)
        out = list(out) if out is not None else None      # a generator is consumed only once
        assert out is not None and len(out) == W, (
            f"shard_column(W(8, 12), b, world_size={W}) should return a list of {W} "
            f"(weight_shard, bias_shard) pairs")
        for r, pair in enumerate(out):
            w_r, b_r = pair
            assert tuple(w_r.shape) == (8 // W, 12), (
                f"world_size={W}: column-parallel weight shard {r} has shape "
                f"{tuple(w_r.shape)}, expected {(8 // W, 12)}. Megatron's A = W^T is split "
                f"by COLUMNS, i.e. nn.Linear's weight (out, in) is split along dim 0 — "
                f"each rank owns a slice of the output features")
            assert torch.equal(w_r, weight[r * 8 // W:(r + 1) * 8 // W]), (
                f"world_size={W}: weight shard {r} should be rows "
                f"[{r * 8 // W}, {(r + 1) * 8 // W}) of W (contiguous blocks in rank order)")
            assert b_r is not None and torch.equal(b_r, bias[r * 8 // W:(r + 1) * 8 // W]), (
                f"world_size={W}: the column-parallel bias is split together with the "
                f"output features — rank {r} needs b[{r * 8 // W}:{(r + 1) * 8 // W}]")
        for w_r, b_r in out:
            w_r.add_(1.0), b_r.add_(1.0)
        assert torch.equal(weight, keep_w) and torch.equal(bias, keep_b), (
            "editing a shard changed the full weight/bias — every rank owns its own "
            "memory, so return copies (.clone()), not views")

        none_b = list(ns.shard_column(weight, None, W))
        assert all(p[1] is None for p in none_b), (
            "shard_column(W, None, world_size) should return bias shards of None")

        w2 = _rand(12, 8, g=g)
        keep = w2.clone()
        rows = ns.shard_row(w2, W)
        rows = list(rows) if rows is not None else None
        assert rows is not None and len(rows) == W and all(torch.is_tensor(t) for t in rows), (
            f"shard_row(W(12, 8), world_size={W}) should return a list of {W} tensors")
        for r, w_r in enumerate(rows):
            assert tuple(w_r.shape) == (12, 8 // W), (
                f"world_size={W}: row-parallel shard {r} has shape {tuple(w_r.shape)}, "
                f"expected {(12, 8 // W)}. Megatron's A = W^T is split by ROWS, i.e. "
                f"nn.Linear's weight (out, in) is split along dim 1 — each rank owns a "
                f"slice of the INPUT features and produces a partial sum of all outputs")
            assert torch.equal(w_r, w2[:, r * 8 // W:(r + 1) * 8 // W]), (
                f"world_size={W}: row-parallel shard {r} should be columns "
                f"[{r * 8 // W}, {(r + 1) * 8 // W}) of W")
        for w_r in rows:
            w_r.add_(1.0)
        assert torch.equal(w2, keep), "shard_row must return copies, not views of W"

    for call, what in ((lambda: ns.shard_column(_rand(6, 4, g=g), None, 4), "shard_column(W(6, 4), None, 4)"),
                       (lambda: ns.shard_row(_rand(4, 6, g=g), 4), "shard_row(W(4, 6), 4)")):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError(f"{what} should raise ValueError: 6 features cannot be split "
                             f"evenly over 4 ranks")


def check_shard_qkv_keeps_heads_together(ns):
    for d, H, W in ((8, 4, 1), (8, 4, 2), (8, 4, 4), (12, 6, 3), (16, 4, 2)):
        p = _attn_params(d, seed=d * 10 + W)
        keep = p["w_qkv"].clone()
        out = ns.shard_qkv(p["w_qkv"], p["b_qkv"], H, W)
        out = list(out) if out is not None else None
        assert out is not None and len(out) == W, (
            f"shard_qkv should return {W} (w_qkv_shard, b_qkv_shard) pairs")
        ref = _ref_qkv(p["w_qkv"], p["b_qkv"], W)
        naive = list(zip(p["w_qkv"].chunk(W, 0), p["b_qkv"].chunk(W, 0)))
        for r, ((w_r, b_r), (rw, rb)) in enumerate(zip(out, ref)):
            assert tuple(w_r.shape) == tuple(rw.shape), (
                f"d={d}, n_heads={H}, world_size={W}: rank {r}'s fused QKV shard has shape "
                f"{tuple(w_r.shape)}, expected {tuple(rw.shape)} = (3 * d / world_size, d)")
            if W > 1 and torch.equal(w_r, naive[r][0]):
                raise AssertionError(
                    f"d={d}, n_heads={H}, world_size={W}: rank {r} got rows "
                    f"[{r * 3 * d // W}, {(r + 1) * 3 * d // W}) of the fused [W_q; W_k; W_v] "
                    f"— a contiguous split of the fused matrix gives rank 0 mostly query rows "
                    f"and the last rank mostly value rows, so a rank's q, k and v belong to "
                    f"different heads. Split Q, K and V separately by heads and concatenate "
                    f"rank r's three pieces")
            assert torch.equal(w_r, rw) and torch.equal(b_r, rb), (
                f"d={d}, n_heads={H}, world_size={W}: rank {r}'s shard should be "
                f"[W_q rows of heads {r * H // W}..{(r + 1) * H // W - 1}; the same rows of "
                f"W_k; of W_v] (and the matching bias entries), in that order")
        for w_r, _ in out:
            w_r.add_(1.0)
        assert torch.equal(p["w_qkv"], keep), "shard_qkv must return copies, not views"

    for d, H, W in ((12, 4, 3), (24, 6, 4)):
        # d % W == 0 here, so only a head-count check can catch it.
        p = _attn_params(d, seed=1)
        try:
            ns.shard_qkv(p["w_qkv"], p["b_qkv"], H, W)
        except ValueError:
            continue
        raise AssertionError(
            f"shard_qkv(..., n_heads={H}, world_size={W}) should raise ValueError: "
            f"{H} heads cannot be split evenly over {W} ranks (d={d} is divisible by {W}, "
            f"but that would cut a head in half)")


# ----------------------------------------------------------------- f and g

def check_copy_to_tp_is_identity_then_allreduce(ns):
    for W in (1, 2, 4):
        g = torch.Generator().manual_seed(W)
        xs = [_rand(2, 3, 4, g=g).requires_grad_() for _ in range(W)]
        G = [_rand(2, 3, 4, g=g) for _ in range(W)]
        comm = _Comm(W)
        ys = _as_list(ns.copy_to_tp(xs, comm), W, "copy_to_tp")
        assert comm.calls == 0, (
            f"world_size={W}: copy_to_tp (f) made {comm.calls} all-reduce call(s) in "
            f"forward — f is the identity in forward")
        for r in range(W):
            assert torch.equal(ys[r].detach(), xs[r].detach()), (
                f"world_size={W}: f's forward must return rank {r}'s input unchanged")
            assert ys[r].requires_grad, (
                f"world_size={W}: f's output is cut off from autograd — return the inputs "
                f"from an autograd.Function forward (e.g. x.view_as(x)), don't detach")
        sum((y * gr).sum() for y, gr in zip(ys, G)).backward()
        assert _count_ok(comm.calls, 1, W), (
            f"world_size={W}: backward through f made {comm.calls} all-reduce call(s), "
            f"expected exactly 1 (one call with all W gradients)")
        total = sum(G)
        for r in range(W):
            assert xs[r].grad is not None, f"world_size={W}: no gradient reached xs[{r}]"
            if W > 1 and _close(xs[r].grad, G[r]):
                raise AssertionError(
                    f"world_size={W}: xs[{r}].grad is only rank {r}'s own gradient. f's "
                    f"backward must ALL-REDUCE: rank r's input feeds every rank's shard, so "
                    f"its gradient is the sum of all ranks' partial input gradients")
            assert _close(xs[r].grad, total), (
                f"world_size={W}: xs[{r}].grad should be the SUM over ranks of the "
                f"gradients flowing into f's outputs")


def check_reduce_from_tp_is_allreduce_then_identity(ns):
    for W in (1, 2, 4):
        g = torch.Generator().manual_seed(10 + W)
        ps = [_rand(2, 3, 4, g=g).requires_grad_() for _ in range(W)]
        G = [_rand(2, 3, 4, g=g) for _ in range(W)]
        comm = _Comm(W)
        zs = _as_list(ns.reduce_from_tp(ps, comm), W, "reduce_from_tp")
        assert _count_ok(comm.calls, 1, W), (
            f"world_size={W}: reduce_from_tp (g) made {comm.calls} all-reduce call(s) in "
            f"forward, expected exactly 1 — and route the data through comm.all_reduce")
        total = sum(p.detach() for p in ps)
        for r in range(W):
            if W > 1 and _close(zs[r].detach(), total / W):
                raise AssertionError(
                    f"world_size={W}: g returned the MEAN of the partials. Row-parallel "
                    f"partials are pieces of one matmul (sum over the split input "
                    f"features), so they must be SUMMED — averaging is for data-parallel "
                    f"gradients")
            assert _close(zs[r].detach(), total), (
                f"world_size={W}: rank {r}'s output of g should be the sum of all ranks' "
                f"partials")
            assert zs[r].requires_grad, (
                f"world_size={W}: g's output does not require grad. comm.all_reduce is not "
                f"differentiable (like NCCL) — call it inside an autograd.Function forward")
        sum((z * gr).sum() for z, gr in zip(zs, G)).backward()
        assert _count_ok(comm.calls, 1, W), (
            f"world_size={W}: backward through g made {comm.calls - 1} all-reduce call(s); "
            f"g's backward is the identity — no communication")
        for r in range(W):
            assert ps[r].grad is not None, f"world_size={W}: no gradient reached partial {r}"
            if W > 1 and _close(ps[r].grad, sum(G)):
                raise AssertionError(
                    f"world_size={W}: g's backward all-reduced the gradients. Every rank "
                    f"computes the same loss from its replica of Z, so rank r's partial "
                    f"must receive exactly its own dL/dZ (identity backward) — summing "
                    f"gives W x the true gradient")
            assert _close(ps[r].grad, G[r]), (
                f"world_size={W}: partial {r}'s gradient should equal the gradient of rank "
                f"{r}'s output (g's backward is the identity)")


# ----------------------------------------------------------------- MLP

ACTS = [("gelu", F.gelu), ("tanh", torch.tanh)]


def _mlp_case(W, seed, d=8, ffn=16, B=2, T=3):
    p = _mlp_params(d, ffn, seed)
    x = _rand(B, T, d, g=torch.Generator().manual_seed(seed + 1))
    return p, x


def check_mlp_forward_matches_dense(ns):
    for W in (1, 2, 4):
        for name, act in ACTS:
            p, x = _mlp_case(W, seed=W)
            dense = _dense_mlp(x, p, act)
            shards = _ref_mlp_shards(p, W)
            xs = [x.clone() for _ in range(W)]
            comm = _Comm(W)
            outs = _as_list(ns.tp_mlp_forward(xs, shards, act, comm), W, "tp_mlp_forward")
            assert _count_ok(comm.calls, 1, W), (
                f"world_size={W}: the TP MLP forward made {comm.calls} all-reduce call(s); "
                f"Megatron's MLP needs exactly ONE (after the row-parallel matmul). The "
                f"column-parallel first linear and the activation need no communication")
            for r, o in enumerate(outs):
                assert tuple(o.shape) == tuple(dense.shape), (
                    f"world_size={W}: rank {r}'s output has shape {tuple(o.shape)}, "
                    f"expected {tuple(dense.shape)}")
                diag = ""
                if W > 1 and _close(o.detach(), dense + (W - 1) * p["b2"]):
                    diag = (" — it is off by exactly (W-1)*b2: the row-parallel bias was "
                            "added on every rank before the reduction. Add it once, after")
                elif W > 1 and _close(o.detach(), (dense - p["b2"]) / W + p["b2"]):
                    diag = " — the partial sums were averaged instead of summed"
                assert _close(o.detach(), dense), (
                    f"world_size={W}, act={name}: rank {r}'s TP MLP output differs from "
                    f"the dense MLP (max diff "
                    f"{(o.detach() - dense).abs().max().item():.2e}){diag}")
            for r in range(W):
                assert torch.equal(xs[r], x), "tp_mlp_forward modified its input in place"
            if W >= 2:
                s = shards
                logged = comm.log[0]
                for r in range(W):
                    want = act(x @ s[r]["w1"].T + s[r]["b1"]) @ s[r]["w2"].T
                    if (_same_values(logged[r], want + p["b2"])
                            or _same_values(logged[r], want + p["b2"] / W)):
                        raise AssertionError(
                            f"world_size={W}: rank {r} put the row-parallel bias into its "
                            f"partial sum before the all-reduce. The bias is replicated on "
                            f"every rank, so it must be added exactly once, after the "
                            f"reduction (Megatron's RowParallelLinear does output_ + bias)")
                    assert _same_values(logged[r], want), (
                        f"world_size={W}: the tensor rank {r} handed to all_reduce should be "
                        f"its partial sum act(x W1_r^T + b1_r) W2_r^T computed from its OWN "
                        f"shards (shape {tuple(want.shape)}), got shape "
                        f"{tuple(logged[r].shape)}")
                tamper = _Comm(W, tamper=0.25)
                outs2 = _as_list(ns.tp_mlp_forward([x.clone() for _ in range(W)], shards, act,
                                                   tamper), W, "tp_mlp_forward")
                assert all(_close(o.detach(), dense + 0.25) for o in outs2), (
                    f"world_size={W}: with a communicator that adds 0.25 to every all-reduce "
                    f"result, the output did not shift by 0.25 — the output must be built "
                    f"from what all_reduce returns (plus b2), not computed some other way")

    # End to end with the solver's own sharding functions.
    for W in (2, 4):
        p, x = _mlp_case(W, seed=20 + W, d=8, ffn=32)
        cols = list(ns.shard_column(p["w1"], p["b1"], W))
        rows = list(ns.shard_row(p["w2"], W))
        shards = [{"w1": w1, "b1": b1, "w2": w2, "b2": p["b2"].clone()}
                  for (w1, b1), w2 in zip(cols, rows)]
        outs = _as_list(ns.tp_mlp_forward([x.clone() for _ in range(W)], shards, F.gelu,
                                          _Comm(W)), W, "tp_mlp_forward")
        dense = _dense_mlp(x, p, F.gelu)
        assert all(_close(o.detach(), dense) for o in outs), (
            f"world_size={W}: shard_column + shard_row + tp_mlp_forward together do not "
            f"reproduce the dense MLP")


def _check_grads(W, shards, dense_p, xs, x_dense, keys, slicer, what):
    for r in range(W):
        for k in keys:
            got = shards[r][k].grad
            want = slicer(k, dense_p[k].grad, r)
            assert got is not None, (
                f"world_size={W}: no gradient reached rank {r}'s {what} '{k}'. "
                f"comm.all_reduce is not differentiable (like NCCL), so the forward must "
                f"go through your reduce_from_tp / copy_to_tp autograd Functions")
            if W > 1 and _close(got, W * want):
                raise AssertionError(
                    f"world_size={W}: rank {r}'s gradient of '{k}' is W x the dense "
                    f"gradient — g's backward must be the identity (each rank's loss "
                    f"replica already gives it dL/dZ; all-reducing it multiplies by W)")
            assert _close(got, want, tol=1e-9), (
                f"world_size={W}: rank {r}'s gradient of '{k}' should equal the matching "
                f"slice of the dense model's gradient (max diff "
                f"{(got - want).abs().max().item() if got.shape == want.shape else float('nan'):.2e})")
        gx = xs[r].grad
        assert gx is not None, (
            f"world_size={W}: no gradient reached xs[{r}] — the block input must flow "
            f"through copy_to_tp (f)")
        assert _close(gx, x_dense.grad, tol=1e-9), (
            f"world_size={W}: xs[{r}].grad should equal the dense dL/dX on every rank. Each "
            f"rank only sees part of dL/dX (its shard's contribution); f's backward "
            f"all-reduce adds them up — is the input going through copy_to_tp?")


def _mlp_slice(W):
    def s(k, gfull, r):
        if k in ("w1", "b1"):
            return gfull.chunk(W, 0)[r]
        if k == "w2":
            return gfull.chunk(W, 1)[r]
        return gfull                                   # b2 is replicated
    return s


def check_mlp_backward_gives_dense_gradients(ns):
    for W in (1, 2, 4):
        p, x = _mlp_case(W, seed=30 + W)
        G = _rand(*x.shape, g=torch.Generator().manual_seed(99))
        dp = {k: v.clone().requires_grad_() for k, v in p.items()}
        xd = x.clone().requires_grad_()
        (_dense_mlp(xd, dp, F.gelu) * G).sum().backward()

        shards = _grad_leaves(_ref_mlp_shards(p, W))
        xs = [x.clone().requires_grad_() for _ in range(W)]
        comm = _Comm(W)
        outs = _as_list(ns.tp_mlp_forward(xs, shards, F.gelu, comm), W, "tp_mlp_forward")
        n_fwd = comm.calls
        # Every rank computes the same loss from its replica; backward once on the sum.
        sum((o * G).sum() for o in outs).backward()
        assert _count_ok(comm.calls - n_fwd, 1, W), _bwd_count_msg("MLP", W, comm.calls - n_fwd)
        _check_grads(W, shards, dp, xs, xd, ["w1", "b1", "w2", "b2"], _mlp_slice(W), "MLP shard")


# ----------------------------------------------------------------- attention

def _attn_case(seed, d=8, B=2, T=5):
    p = _attn_params(d, seed)
    x = _rand(B, T, d, g=torch.Generator().manual_seed(seed + 1))
    return p, x


def check_attention_forward_matches_dense(ns):
    for d, H, W in ((8, 4, 1), (8, 4, 2), (8, 4, 4), (12, 6, 2), (16, 8, 4)):
        p, x = _attn_case(seed=d + H + W, d=d)
        dense = _dense_attention(x, p, H)
        shards = _ref_attn_shards(p, W)
        comm = _Comm(W)
        xs = [x.clone() for _ in range(W)]
        outs = _as_list(ns.tp_attention_forward(xs, shards, H, comm), W, "tp_attention_forward")
        assert _count_ok(comm.calls, 1, W), (
            f"world_size={W}: the TP attention forward made {comm.calls} all-reduce "
            f"call(s), expected exactly ONE (after the row-parallel output projection). "
            f"Each rank's heads are independent, so QKV, softmax and the per-head outputs "
            f"need no communication")
        for r, o in enumerate(outs):
            diag = ""
            if W > 1 and _close(o.detach(), dense + (W - 1) * p["b_o"]):
                diag = (" — off by exactly (W-1)*b_o: the output-projection bias was added on "
                        "every rank before the all-reduce")
            elif W > 1 and _close(o.detach(), (dense - p["b_o"]) / W + p["b_o"]):
                diag = " — the partial outputs were averaged instead of summed"
            assert tuple(o.shape) == tuple(dense.shape) and _close(o.detach(), dense), (
                f"d={d}, n_heads={H}, world_size={W}: rank {r}'s TP attention output differs "
                f"from dense causal multi-head attention (softmax scale 1/sqrt(head_dim))"
                f"{diag}. Rank r holds heads "
                f"{r * H // W}..{(r + 1) * H // W - 1}: split its local qkv into q, k, v "
                f"(chunk(3, -1)) and use n_heads // world_size heads of size d // n_heads")
        for r in range(W):
            assert torch.equal(xs[r], x), "tp_attention_forward modified its input in place"
        if W >= 2:
            for r in range(W):
                s = shards[r]
                want = _attn_core(x, s["w_qkv"], s["b_qkv"], H // W) @ s["w_o"].T
                assert _same_values(comm.log[0][r], want), (
                    f"d={d}, n_heads={H}, world_size={W}: the tensor rank {r} handed to "
                    f"all_reduce should be its partial output (its heads' attention output "
                    f"@ w_o_r.T, no bias)")
            tamper = _Comm(W, tamper=-0.5)
            outs2 = _as_list(ns.tp_attention_forward([x.clone() for _ in range(W)], shards, H,
                                                     tamper), W, "tp_attention_forward")
            assert all(_close(o.detach(), dense - 0.5) for o in outs2), (
                f"world_size={W}: with a communicator that subtracts 0.5 from every "
                f"all-reduce result, the attention output did not shift — build the output "
                f"from what all_reduce returns (plus b_o)")

    # End to end with the solver's own sharding functions.
    for d, H, W in ((8, 4, 2), (16, 8, 4)):
        p, x = _attn_case(seed=40 + W, d=d)
        qkv = list(ns.shard_qkv(p["w_qkv"], p["b_qkv"], H, W))
        wo = list(ns.shard_row(p["w_o"], W))
        shards = [{"w_qkv": w, "b_qkv": b, "w_o": o, "b_o": p["b_o"].clone()}
                  for (w, b), o in zip(qkv, wo)]
        outs = _as_list(ns.tp_attention_forward([x.clone() for _ in range(W)], shards, H,
                                                _Comm(W)), W, "tp_attention_forward")
        dense = _dense_attention(x, p, H)
        assert all(_close(o.detach(), dense) for o in outs), (
            f"d={d}, n_heads={H}, world_size={W}: shard_qkv + shard_row + "
            f"tp_attention_forward together do not reproduce dense attention")

    # Heads must divide over the ranks.
    d, H, W = 12, 4, 3
    p, x = _attn_case(seed=5, d=d)
    bad = [{"w_qkv": torch.cat([t.chunk(W, 0)[r] for t in p["w_qkv"].chunk(3, 0)]),
            "b_qkv": torch.cat([t.chunk(W, 0)[r] for t in p["b_qkv"].chunk(3, 0)]),
            "w_o": p["w_o"].chunk(W, 1)[r], "b_o": p["b_o"]} for r in range(W)]
    try:
        ns.tp_attention_forward([x.clone() for _ in range(W)], bad, H, _Comm(W))
    except ValueError:
        pass
    else:
        raise AssertionError(
            f"tp_attention_forward with n_heads={H} on world_size={W} should raise ValueError: "
            f"heads cannot be split evenly (even though d={d} divides by {W})")


def _attn_slice(W):
    def s(k, gfull, r):
        if k in ("w_qkv", "b_qkv"):
            return torch.cat([t.chunk(W, 0)[r] for t in gfull.chunk(3, 0)])
        if k == "w_o":
            return gfull.chunk(W, 1)[r]
        return gfull                                   # b_o is replicated
    return s


def check_attention_backward_gives_dense_gradients(ns):
    for d, H, W in ((8, 4, 1), (8, 4, 2), (8, 4, 4)):
        p, x = _attn_case(seed=60 + W, d=d)
        G = _rand(*x.shape, g=torch.Generator().manual_seed(98))
        dp = {k: v.clone().requires_grad_() for k, v in p.items()}
        xd = x.clone().requires_grad_()
        (_dense_attention(xd, dp, H) * G).sum().backward()

        shards = _grad_leaves(_ref_attn_shards(p, W))
        xs = [x.clone().requires_grad_() for _ in range(W)]
        comm = _Comm(W)
        outs = _as_list(ns.tp_attention_forward(xs, shards, H, comm), W, "tp_attention_forward")
        n_fwd = comm.calls
        sum((o * G).sum() for o in outs).backward()
        assert _count_ok(comm.calls - n_fwd, 1, W), _bwd_count_msg("attention", W,
                                                                  comm.calls - n_fwd)
        _check_grads(W, shards, dp, xs, xd, ["w_qkv", "b_qkv", "w_o", "b_o"], _attn_slice(W),
                     "attention shard")


CHECKS = [
    check_shard_column_and_shard_row,
    check_shard_qkv_keeps_heads_together,
    check_copy_to_tp_is_identity_then_allreduce,
    check_reduce_from_tp_is_allreduce_then_identity,
    check_mlp_forward_matches_dense,
    check_mlp_backward_gives_dense_gradients,
    check_attention_forward_matches_dense,
    check_attention_backward_gives_dense_gradients,
]
