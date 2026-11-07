"""ring-allreduce — the bandwidth-optimal collective behind DDP, simulated in one process.

W ranks live in one Python process and may only exchange data through a
`Network` (re-created below, identical to the notebook's) that logs every
message as (src, dst, numel). The checks verify

  * the maths: every rank ends with the elementwise SUM, and all ranks hold
    bitwise-identical results (all-gather copies, it never re-adds);
  * the pattern that makes it a *ring*: rank r only ever sends to (r+1) % W,
    at most 2(W-1) messages of at most ceil(N/W) elements, and the whole ring
    moves exactly 2(W-1)·N elements (every chunk travels W-1 hops in
    reduce-scatter and W-1 hops in all-gather, whatever the chunking). A
    gather-to-rank-0 or pass-the-whole-vector scheme fails these;
  * locality: on a network that delivers only zeros, rank r's output must not
    change when other ranks' inputs change — the data really flows through
    recv() rather than being read out of the shared Python list;
  * DDP: one fused all-reduce of the flattened trainable grads, divided by W,
    reproduces single-process full-batch gradients; frozen params are skipped
    and a None grad counts as zeros.
"""
import copy
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F

ENTRIES = ["ring_allreduce", "ddp_average_gradients"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Give every rank a private working copy split into W chunks with "
    "torch.tensor_split (sizes differ by at most one, empty chunks are fine). "
    "Phase 1 (reduce-scatter) leaves each rank owning one fully-summed chunk; "
    "phase 2 (all-gather) passes those finished chunks around the ring.",
    "Reduce-scatter step s: every rank r first sends chunk (r - s) % W to rank "
    "(r + 1) % W, then every rank receives from rank (r - 1) % W and ADDS the "
    "message into chunk (r - 1 - s) % W. After W - 1 steps rank r owns the "
    "complete sum of chunk (r + 1) % W.",
    "All-gather step s: rank r sends chunk (r + 1 - s) % W, receives into chunk "
    "(r - s) % W and OVERWRITES it (copy_, never +=). For DDP: torch.cat the "
    "trainable grads (zeros_like(p) where p.grad is None) into one flat vector "
    "per replica, all-reduce once, divide by W, then torch.split by p.numel() "
    "and view_as(p) to write the grads back.",
]


class Network:
    """Same API as the notebook's simulator: send(src, dst, t) / recv(src, dst)."""

    def __init__(self, world_size):
        self.world_size = world_size
        self.log = []                       # one (src, dst, numel) per send
        self._wire = defaultdict(deque)     # (src, dst) -> FIFO of tensors

    def _deliver(self, tensor):
        return tensor.detach().clone()

    def send(self, src, dst, tensor):
        W = self.world_size
        if not (0 <= src < W and 0 <= dst < W):
            raise ValueError(f"invalid ranks src={src}, dst={dst} for world_size={W}")
        if src == dst:
            raise ValueError(f"rank {src} tried to send a message to itself")
        if not torch.is_tensor(tensor):
            raise TypeError(f"send() takes a tensor, got {type(tensor).__name__}")
        self.log.append((src, dst, tensor.numel()))
        self._wire[(src, dst)].append(self._deliver(tensor))

    def recv(self, src, dst):
        q = self._wire[(src, dst)]
        if not q:
            raise RuntimeError(
                f"rank {dst} is waiting for a message from rank {src} but none was "
                f"sent (in a real cluster this is a deadlock)")
        return q.popleft()

    def pending(self):
        return sum(len(q) for q in self._wire.values())


class _ZeroNetwork(Network):
    """Delivers a zero tensor of the right shape instead of the payload."""

    def _deliver(self, tensor):
        return torch.zeros_like(tensor.detach())


def _inputs(W, N, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(N, generator=g, dtype=torch.float64) for _ in range(W)]


def _run(ns, tensors, net):
    out = ns.ring_allreduce(tensors, net)
    assert out is not None, "ring_allreduce returned None — return the list of W results"
    out = list(out)
    assert len(out) == len(tensors), (
        f"ring_allreduce should return one result per rank ({len(tensors)}), got {len(out)}")
    return out


SIZES = [(1, 5), (2, 1), (2, 9), (3, 2), (3, 10), (4, 3), (4, 13), (4, 16),
         (7, 5), (7, 50), (7, 101)]


def check_every_rank_gets_the_sum(ns):
    for i, (W, N) in enumerate(SIZES):
        tensors = _inputs(W, N, seed=i)
        expected = torch.stack(tensors).sum(0)
        out = _run(ns, tensors, Network(W))
        for r, res in enumerate(out):
            assert torch.is_tensor(res) and tuple(res.shape) == (N,), (
                f"W={W}, N={N}: rank {r}'s result should be a ({N},) tensor, got "
                f"{tuple(res.shape) if torch.is_tensor(res) else type(res).__name__}")
            assert res.dtype == torch.float64, (
                f"W={W}, N={N}: rank {r}'s result is {res.dtype} but the inputs are "
                f"float64 — keep the input dtype")
            diff = (res - expected).abs().max().item() if N else 0.0
            assert torch.allclose(res, expected, rtol=1e-12, atol=1e-12), (
                f"W={W}, N={N}: rank {r}'s result is not the elementwise sum of all "
                f"{W} inputs (max diff {diff:.2e}). Check the chunk indices of both "
                f"phases, and that the all-gather COPIES finished chunks instead of "
                f"adding them"
                + (". N % W != 0 here: if only such sizes fail, split with "
                   "torch.tensor_split, not torch.chunk" if N % W else ""))
        for r in range(1, W):
            assert torch.equal(out[0], out[r]), (
                f"W={W}, N={N}: ranks 0 and {r} hold different bits. In a ring "
                f"all-reduce every rank ends with identical values because the "
                f"all-gather phase copies finished chunks instead of re-summing them "
                f"— otherwise DDP replicas would slowly drift apart")


def check_messages_follow_the_ring(ns):
    """Ring = only right-neighbour traffic, 2(W-1) chunk-sized messages per rank."""
    for i, (W, N) in enumerate(SIZES):
        net = Network(W)
        _run(ns, _inputs(W, N, seed=100 + i), net)
        if W == 1:
            assert not net.log, (
                f"with a single rank there is nobody to talk to, but "
                f"{len(net.log)} messages were sent")
            continue
        assert net.log, (
            f"W={W}, N={N}: no messages went through `net` — every byte a rank "
            f"receives must be sent with net.send and picked up with net.recv")
        ceil = -(-N // W)
        for src, dst, numel in net.log:
            assert dst == (src + 1) % W, (
                f"W={W}: rank {src} sent a message to rank {dst}. In a ring, rank r "
                f"only talks to its right neighbour (r + 1) % W — sending to rank 0 "
                f"is the parameter-server pattern this algorithm replaces")
            assert numel <= ceil, (
                f"W={W}, N={N}: rank {src} sent a {numel}-element message, but a "
                f"ring only ever moves one chunk of at most ceil(N/W) = {ceil} "
                f"elements per step — don't pass whole vectors around")
        for r in range(W):
            sent = [n for s, _, n in net.log if s == r]
            assert len(sent) <= 2 * (W - 1), (
                f"W={W}, N={N}: rank {r} sent {len(sent)} messages; a ring "
                f"all-reduce needs exactly 2(W-1) = {2 * (W - 1)} "
                f"(W-1 reduce-scatter + W-1 all-gather steps)")
            if N >= W:
                assert len(sent) == 2 * (W - 1), (
                    f"W={W}, N={N}: rank {r} sent {len(sent)} messages, expected "
                    f"2(W-1) = {2 * (W - 1)} (W-1 reduce-scatter + W-1 all-gather)")
        total = sum(n for _, _, n in net.log)
        assert total == 2 * (W - 1) * N, (
            f"W={W}, N={N}: the ring moved {total} elements in total, expected "
            f"exactly 2(W-1)·N = {2 * (W - 1) * N}: every chunk travels W-1 hops "
            f"while being reduced and W-1 hops while being gathered (no padding)")
        assert net.pending() == 0, (
            f"W={W}, N={N}: {net.pending()} message(s) were sent but never "
            f"received — every send needs a matching recv")


def check_data_really_flows_through_the_network(ns):
    """On a network that delivers zeros, rank r can only know its own input."""
    W, N = 4, 10
    base = _inputs(W, N, seed=7)
    ref = _run(ns, [t.clone() for t in base], _ZeroNetwork(W))
    g = torch.Generator().manual_seed(8)
    for r in range(W):
        other = [t.clone() if j == r else t + torch.randn(N, generator=g, dtype=torch.float64)
                 for j, t in enumerate(base)]
        out = _run(ns, other, _ZeroNetwork(W))
        assert torch.equal(out[r], ref[r]), (
            f"with a network that delivers only zeros, rank {r}'s result still "
            f"changed when the OTHER ranks' inputs changed — rank {r} is reading "
            f"tensors[j] for j != {r} directly. Each rank may only use its own "
            f"tensor plus what net.recv hands it")


def check_inputs_untouched_and_results_independent(ns):
    W, N = 4, 13
    tensors = _inputs(W, N, seed=3)
    originals = [t.clone() for t in tensors]
    out = _run(ns, tensors, Network(W))
    for r in range(W):
        assert torch.equal(tensors[r], originals[r]), (
            f"rank {r}'s input tensor was modified in place — work on a copy "
            f"(callers such as DDP still need their local gradients)")
    snap = [o.clone() for o in out]
    for r in range(W):
        out[r].add_(1000.0)
        for j in range(W):
            if j != r:
                assert torch.equal(out[j], snap[j]), (
                    f"changing rank {r}'s result also changed rank {j}'s — each rank "
                    f"owns separate memory, so return W independent tensors")
        assert all(torch.equal(tensors[j], originals[j]) for j in range(W)), (
            f"rank {r}'s result aliases an input tensor — return new tensors")
        out[r].copy_(snap[r])


class _Model(nn.Module):
    """Params of several shapes (2-D, 1-D, 0-dim) plus a frozen one."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 8)
        self.fc2 = nn.Linear(8, 3, bias=False)
        self.gain = nn.Parameter(torch.ones(3))
        self.temperature = nn.Parameter(torch.tensor(1.5))
        self.frozen = nn.Parameter(torch.randn(3), requires_grad=False)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x))) * self.gain / self.temperature + self.frozen


def _ddp_setup(W, per_rank=3, seed=0):
    torch.manual_seed(seed)
    base = _Model().double()
    X = torch.randn(W * per_rank, 5, dtype=torch.float64)
    Y = torch.randn(W * per_rank, 3, dtype=torch.float64)
    single = copy.deepcopy(base)
    F.mse_loss(single(X), Y).backward()
    replicas = [copy.deepcopy(base) for _ in range(W)]
    for m, xs, ys in zip(replicas, X.chunk(W), Y.chunk(W)):
        F.mse_loss(m(xs), ys).backward()       # mean over the local shard
    return base, single, replicas


def check_ddp_matches_single_process_training(ns):
    for W in (1, 2, 3, 4):
        base, single, replicas = _ddp_setup(W, seed=W)
        net = Network(W)
        ns.ddp_average_gradients(replicas, net)
        names = [n for n, _ in single.named_parameters()]
        for r, m in enumerate(replicas):
            for name, ps, pm, pb in zip(names, single.parameters(), m.parameters(),
                                         base.parameters()):
                assert torch.equal(pm.detach(), pb.detach()), (
                    f"W={W}: replica {r}'s parameter '{name}' changed — only .grad "
                    f"should be written, the optimizer step comes later")
                if ps.grad is None:
                    assert pm.grad is None, (
                        f"W={W}: '{name}' is frozen (requires_grad=False) but replica "
                        f"{r} now has a .grad — skip frozen params entirely (an "
                        f"optimizer with weight decay would otherwise move them)")
                    continue
                assert pm.grad is not None, f"W={W}: replica {r}'s '{name}'.grad is None"
                assert tuple(pm.grad.shape) == tuple(pm.shape), (
                    f"W={W}: replica {r}'s '{name}'.grad has shape "
                    f"{tuple(pm.grad.shape)}, expected {tuple(pm.shape)}")
                diff = (pm.grad - ps.grad).abs().max().item()
                hint = (" — did you forget to divide the sum by W?"
                        if torch.allclose(pm.grad, W * ps.grad, atol=1e-10) and W > 1 else "")
                assert torch.allclose(pm.grad, ps.grad, rtol=1e-10, atol=1e-12), (
                    f"W={W}: after averaging, replica {r}'s grad of '{name}' differs "
                    f"from single-process full-batch training (max diff {diff:.2e})"
                    + hint)
        n_train = sum(p.numel() for p in base.parameters() if p.requires_grad)
        for r in range(W):
            sent = sum(1 for s, _, _ in net.log if s == r)
            assert sent == 2 * (W - 1), (
                f"W={W}: rank {r} sent {sent} messages for one gradient sync, expected "
                f"2(W-1) = {2 * (W - 1)}: flatten all grads into ONE buffer and call "
                f"ring_allreduce on it once"
                + (" — one all-reduce per parameter pays the per-message latency once "
                   "per parameter" if sent > 2 * (W - 1) else ""))
        total = sum(n for _, _, n in net.log)
        assert total == 2 * (W - 1) * n_train, (
            f"W={W}: {total} gradient elements moved over the ring, expected "
            f"2(W-1)·{n_train} — the flat buffer should hold exactly the trainable "
            f"parameters (skip requires_grad=False)")


def check_ddp_treats_missing_grads_as_zero(ns):
    W = 3
    _, _, replicas = _ddp_setup(W, seed=11)
    replicas[1].fc1.bias.grad = None           # e.g. this param was unused on rank 1
    expected = sum(m.fc1.bias.grad for i, m in enumerate(replicas) if i != 1) / W
    expected_w = sum(m.fc2.weight.grad for m in replicas) / W
    ns.ddp_average_gradients(replicas, Network(W))
    for r, m in enumerate(replicas):
        g = m.fc1.bias.grad
        assert g is not None, (
            f"replica {r}'s fc1.bias.grad is None after the sync — a parameter "
            f"whose grad is None on some replica counts as zeros there, and every "
            f"replica must receive the averaged grad")
        assert torch.allclose(g, expected, rtol=1e-10, atol=1e-12), (
            f"replica {r}'s fc1.bias.grad should be (sum of the other replicas' "
            f"grads + zeros) / W when one replica had grad=None")
        assert torch.allclose(m.fc2.weight.grad, expected_w, rtol=1e-10, atol=1e-12), (
            f"replica {r}'s fc2.weight.grad is wrong once another parameter's grad "
            f"was None — the flat-buffer offsets must not shift when a grad is "
            f"missing (substitute zeros_like(p), don't skip it)")


CHECKS = [
    check_every_rank_gets_the_sum,
    check_messages_follow_the_ring,
    check_data_really_flows_through_the_network,
    check_inputs_untouched_and_results_independent,
    check_ddp_matches_single_process_training,
    check_ddp_treats_missing_grads_as_zero,
]
