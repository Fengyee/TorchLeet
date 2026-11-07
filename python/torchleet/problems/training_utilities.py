"""training-utilities — the training-loop machinery written from memory: SGD with
momentum / Nesterov / weight decay, a linear-warmup + cosine LR schedule, global-norm
gradient clipping, and an exponential moving average (EMA) of the weights.

Oracles: torch.optim.SGD and torch.nn.utils.clip_grad_norm_ run side by side with the
solver's versions on identical copies of one model; the LR schedule and the EMA are
compared against their closed forms.
"""
import copy
import math

import torch
import torch.nn as nn

ENTRIES = ["SGD", "warmup_cosine_lr", "clip_grad_norm_", "EMA"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Write each update rule down before coding: what does PyTorch store in the momentum "
    "buffer, and is weight decay added before or after momentum? For the schedule, sketch "
    "lr(step) and mark warmup_steps and total_steps.",
    "SGD: g = grad + wd*p; buf = g on the first step, else buf = mu*buf + g; the step "
    "direction is g + mu*buf for Nesterov, buf otherwise; skip params whose .grad is None. "
    "Clipping: ONE norm over all gradients together, sqrt(sum of squared entries).",
    "model.parameters() is a one-shot generator: list() it if you iterate twice. Clip "
    "coefficient = max_norm / (total_norm + 1e-6), applied only when it is < 1; return "
    "the pre-clip norm. EMA: keep detached clones, update under torch.no_grad() with "
    "s.mul_(decay).add_(p, alpha=1 - decay), copy buffers as-is.",
]


# ---------------------------------------------------------------- helpers

class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 8)
        self.fc2 = nn.Linear(8, 3)
        self.unused = nn.Parameter(torch.randn(4))   # never in the graph: grad stays None

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


def _named(fn, name):
    fn.__name__ = name
    return fn


def _set_lr(opt, lr):
    """How a schedule sets the LR: param_groups for an Optimizer subclass, self.lr for a
    plain class (the notebook's set_lr does the same)."""
    groups = getattr(opt, "param_groups", None)
    if groups:
        for group in groups:
            group["lr"] = lr
    if hasattr(opt, "lr"):
        opt.lr = lr
    if not groups and not hasattr(opt, "lr"):
        raise AssertionError(
            "cannot change the learning rate of your SGD between steps: subclass "
            "torch.optim.Optimizer (lr lives in param_groups) or store it as self.lr")


# A warmup-then-cosine LR sequence for the changing-LR checks (all > 0).
_CHANGING_LRS = [0.05 * (t + 1) / 5 if t < 5 else
                 0.005 + 0.5 * (0.05 - 0.005) * (1 + math.cos(math.pi * (t - 5) / 15))
                 for t in range(20)]


def _sgd_trajectory_check(label, momentum, weight_decay, nesterov, zero_inplace=False,
                          lrs=None):
    """Run solver SGD and torch.optim.SGD for 20 steps on identical nets and compare
    every parameter after every step. With `lrs`, the LR is changed before every step."""
    def check(ns):
        torch.manual_seed(0)
        net = _Net()
        ref = copy.deepcopy(net)
        g = torch.Generator().manual_seed(1)
        xs, ys = torch.randn(20, 6, 5, generator=g), torch.randn(20, 6, 3, generator=g)
        lr = 0.05
        cfg = (f"(lr={lr if lrs is None else 'changing'}, momentum={momentum}, "
               f"weight_decay={weight_decay}, nesterov={nesterov})")
        # net.parameters() is a generator on purpose: a real call site passes one.
        opt = ns.SGD(net.parameters(), lr=lr, momentum=momentum,
                     weight_decay=weight_decay, nesterov=nesterov)
        ref_opt = torch.optim.SGD(ref.parameters(), lr=lr, momentum=momentum,
                                  weight_decay=weight_decay, nesterov=nesterov)
        prev = {n: p.detach().clone() for n, p in net.named_parameters()}
        for t in range(20):
            if zero_inplace:
                # zero_grad(set_to_none=False) style: the SAME grad tensors are zeroed in
                # place, which exposes a momentum buffer that aliases p.grad.
                for m in (net, ref):
                    for p in m.parameters():
                        if p.grad is not None:
                            p.grad.zero_()
            else:
                opt.zero_grad()
                ref_opt.zero_grad()
            nn.functional.mse_loss(net(xs[t]), ys[t]).backward()
            nn.functional.mse_loss(ref(xs[t]), ys[t]).backward()
            if lrs is not None:
                _set_lr(opt, lrs[t])
                _set_lr(ref_opt, lrs[t])
            grads = {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}
            opt.step()
            ref_opt.step()
            for n, p in net.named_parameters():
                if n in grads:
                    assert p.grad is not None and torch.equal(p.grad, grads[n]), (
                        f"step() modified p.grad of {n} at step {t + 1} {cfg}. Compute "
                        "g = grad + weight_decay*p out of place, and clone the first momentum "
                        "buffer (a buffer that IS p.grad gets changed by buf.mul_())")
            cur = dict(net.named_parameters())
            want = dict(ref.named_parameters())
            if all(torch.equal(cur[n].detach(), prev[n]) for n in cur if n != "unused"):
                raise AssertionError(
                    f"step() changed nothing on call {t + 1} {cfg}. If you stored `params` "
                    "as given, note model.parameters() is a generator that is exhausted "
                    "after one pass — keep list(params).")
            assert torch.equal(cur["unused"].detach(), want["unused"].detach()), (
                f"parameter whose .grad is None was modified at step {t + 1} {cfg} — "
                "PyTorch skips params without a gradient (no weight decay either)")
            for n in cur:
                d = (cur[n].detach() - want[n].detach()).abs().max().item()
                assert torch.allclose(cur[n].detach(), want[n].detach(), atol=1e-5, rtol=1e-4), (
                    f"{n} differs from torch.optim.SGD{cfg} after step {t + 1} "
                    f"(max diff {d:.2e})" + ("" if lrs is None else
                    ". The LR changes before every step here: PyTorch keeps lr OUT of the "
                    "buffer and applies the CURRENT lr to it (p -= lr*buf). A velocity "
                    "v = mu*v - lr*g matches only while the LR is constant"))
            prev = {n: p.detach().clone() for n, p in net.named_parameters()}
    return _named(check, f"check_sgd_matches_torch_{label}")


def check_sgd_zero_grad_clears_gradients(ns):
    torch.manual_seed(0)
    net = _Net()
    opt = ns.SGD(net.parameters(), lr=0.1)
    net(torch.randn(4, 5)).sum().backward()
    opt.zero_grad()
    stale = [n for n, p in net.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert not stale, f"zero_grad() left gradients on {stale} (set them to None or zero)"


# ---------------------------------------------------------------- LR schedule

def _lr(ns, step, total, warmup, base, min_lr=None):
    kw = {} if min_lr is None else {"min_lr": min_lr}
    return float(ns.warmup_cosine_lr(step, total, warmup, base, **kw))


def check_lr_warmup_is_linear_from_base_over_warmup(ns):
    base, total, warm = 0.4, 110, 10
    got0 = _lr(ns, 0, total, warm, base, 0.01)
    if abs(got0) < 1e-12:
        raise AssertionError(
            "lr at step 0 is 0 — this problem uses lr = base_lr*(step+1)/warmup_steps, "
            "so step 0 already trains with base_lr/warmup_steps")
    for s in (0, 3, 9):
        want = base * (s + 1) / warm
        got = _lr(ns, s, total, warm, base, 0.01)
        assert math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-9), \
            f"warmup: lr({s}) should be base_lr*(step+1)/warmup = {want:.6f}, got {got:.6f}"
    vals = [_lr(ns, s, total, warm, base, 0.01) for s in range(warm)]
    assert all(b > a for a, b in zip(vals, vals[1:])), "lr must increase strictly during warmup"


def check_lr_cosine_decay_key_points(ns):
    base, total, warm, lo = 0.4, 110, 10, 0.01
    span = total - warm                       # 100 steps of cosine

    def cos_lr(progress, floor=lo):
        return floor + 0.5 * (base - floor) * (1 + math.cos(math.pi * progress))

    for s in (warm, warm + span // 4, warm + span // 2, total - 1, total, total + 50):
        want = cos_lr(min(1.0, (s - warm) / span))
        got = _lr(ns, s, total, warm, base, lo)
        if math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-9):
            continue
        why = ""
        if s > total:
            why = " — clamp to min_lr after total_steps"
        elif math.isclose(got, cos_lr(s / total), rel_tol=1e-6):
            why = " — the cosine must run over (total_steps - warmup_steps), starting at warmup_steps"
        elif math.isclose(got, cos_lr((s - warm) / span, floor=0.0), rel_tol=1e-6):
            why = " — decay towards min_lr, not towards 0"
        raise AssertionError(
            f"lr({s}) with warmup={warm}, total={total}, base={base}, min_lr={lo} should be "
            f"{want:.6f}, got {got:.6f}{why}")
    vals = [_lr(ns, s, total, warm, base, lo) for s in range(warm, total + 30)]
    assert all(b <= a + 1e-12 for a, b in zip(vals, vals[1:])), \
        "lr must be non-increasing after warmup (and stay at min_lr past total_steps)"


def check_lr_edge_cases(ns):
    got = _lr(ns, 0, 50, 0, 1.0, 0.0)
    assert math.isclose(got, 1.0, rel_tol=1e-6), \
        f"warmup_steps=0 should start at base_lr (no warmup), got lr(0)={got}"
    got = _lr(ns, 50, 50, 5, 1.0)
    assert abs(got) < 1e-9, f"min_lr defaults to 0: lr(total_steps) should be 0, got {got}"
    got = _lr(ns, 25, 50, 0, 1.0, 0.0)
    assert math.isclose(got, 0.5, rel_tol=1e-6), f"lr at the cosine midpoint should be 0.5, got {got}"


# ---------------------------------------------------------------- clipping

def _grad_net(seed, scale):
    torch.manual_seed(seed)
    net = _Net()
    g = torch.Generator().manual_seed(seed + 100)
    for n, p in net.named_parameters():
        if n != "unused":                      # 'unused' keeps grad=None on purpose
            p.grad = torch.randn(p.shape, generator=g) * scale
    return net


def check_clip_matches_torch_when_clipping(ns):
    net = _grad_net(0, 3.0)
    ref = copy.deepcopy(net)                    # deepcopy does not copy .grad of Parameters
    for p, q in zip(net.parameters(), ref.parameters()):
        q.grad = None if p.grad is None else p.grad.clone()
    want = torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0)
    got = ns.clip_grad_norm_(net.parameters(), 1.0)   # a generator, like a real call site
    assert got is not None, "clip_grad_norm_ must return the total norm"
    assert math.isclose(float(got), float(want), rel_tol=1e-5), (
        f"returned norm {float(got):.5f} != {float(want):.5f} — return the global L2 norm "
        "of all gradients BEFORE clipping")
    for (n, p), q in zip(net.named_parameters(), ref.parameters()):
        if q.grad is None:
            assert p.grad is None, f"{n} had no grad and must keep grad=None"
            continue
        assert torch.allclose(p.grad, q.grad, atol=1e-6, rtol=1e-5), (
            f"clipped grad of {n} differs from torch.nn.utils.clip_grad_norm_ "
            f"(max diff {(p.grad - q.grad).abs().max():.2e}). Scale every grad by the same "
            "max_norm/(total_norm+1e-6); if nothing changed, did a generator get exhausted "
            "by the norm pass?")
    total = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in net.parameters() if p.grad is not None))
    assert total <= 1.0 + 1e-4, f"global grad norm after clipping is {total:.4f} > max_norm=1.0"


def check_clip_leaves_small_gradients_alone(ns):
    net = _grad_net(1, 0.01)
    before = {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}
    want = math.sqrt(sum(float(g.pow(2).sum()) for g in before.values()))
    got = ns.clip_grad_norm_(net.parameters(), 5.0)
    assert math.isclose(float(got), want, rel_tol=1e-5), \
        f"returned norm {float(got):.6f} != total grad norm {want:.6f}"
    for n, p in net.named_parameters():
        if n in before:
            assert torch.allclose(p.grad, before[n], rtol=1e-5, atol=1e-8), (
                f"{n}: grads with total norm {want:.3f} < max_norm=5 must not be rescaled "
                "(clip, don't normalise)")


def check_clip_zero_and_missing_grads(ns):
    net = _Net()
    for p in net.parameters():
        p.grad = None
    got = ns.clip_grad_norm_(net.parameters(), 1.0)
    assert got is not None and float(got) == 0.0, \
        f"no parameter has a gradient: expected norm 0, got {got}"
    for n, p in net.named_parameters():
        if n != "unused":
            p.grad = torch.zeros_like(p)
    got = ns.clip_grad_norm_(net.parameters(), 1.0)
    assert float(got) == 0.0, f"all-zero gradients: expected norm 0, got {got}"
    bad = [n for n, p in net.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"all-zero gradients became NaN/Inf on {bad} — division by a zero norm?"


# ---------------------------------------------------------------- EMA

def _bn_net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 6), nn.BatchNorm1d(6), nn.ReLU(), nn.Linear(6, 2))


def _run_ema(ns, decay=0.9, k=5):
    """Returns (model, ema, expected_params) after k updates with fresh weights each time."""
    model = _bn_net()
    expected = {n: p.detach().clone() for n, p in model.named_parameters()}
    ema = ns.EMA(model, decay)
    g = torch.Generator().manual_seed(7)
    for _ in range(k):
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(torch.randn(p.shape, generator=g))
        model.train()
        model(torch.randn(8, 4, generator=g) * 3 + 1)    # moves BN running stats
        ema.update(model)
        for n, p in model.named_parameters():
            expected[n] = decay * expected[n] + (1 - decay) * p.detach()
    return model, ema, expected


def _fresh_target(seed=123):
    t = _bn_net()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in t.parameters():
            p.copy_(torch.randn(p.shape, generator=g))
    return t


def check_ema_matches_closed_form(ns):
    model, ema, expected = _run_ema(ns, decay=0.9, k=5)
    target = _fresh_target()
    ema.copy_to(target)
    live = dict(model.named_parameters())
    for n, p in target.named_parameters():
        if torch.allclose(p.detach(), live[n].detach()) and not torch.allclose(p.detach(), expected[n], atol=1e-6):
            raise AssertionError(
                f"EMA of {n} equals the live weights — the shadow must hold detached CLONES, "
                "not references to the model's parameters")
        d = (p.detach() - expected[n]).abs().max().item()
        assert torch.allclose(p.detach(), expected[n], atol=1e-6), (
            f"EMA of {n} after 5 updates differs from decay^k*p0 + (1-decay)*sum decay^(k-i)*p_i "
            f"(max diff {d:.2e}). shadow starts as a copy of the weights and each update is "
            "shadow = decay*shadow + (1-decay)*param")


def check_ema_extreme_decays(ns):
    for decay in (0.0, 1.0):
        model, ema, expected = _run_ema(ns, decay=decay, k=3)
        target = _fresh_target()
        ema.copy_to(target)
        for n, p in target.named_parameters():
            assert torch.allclose(p.detach(), expected[n], atol=1e-6), (
                f"decay={decay}: EMA of {n} should be "
                + ("the latest weights" if decay == 0.0 else "the initial weights (frozen)"))


def check_ema_copies_buffers(ns):
    model, ema, _ = _run_ema(ns)
    target = _fresh_target()
    ema.copy_to(target)
    live = dict(model.named_buffers())
    for n, b in target.named_buffers():
        assert torch.equal(b, live[n]), (
            f"buffer {n} after copy_to should equal the model's latest buffer (buffers such as "
            "BatchNorm running stats are copied on update, not averaged)")


def _graph_tensors(obj, seen, depth=0):
    if depth > 4 or id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, nn.Module):
        yield from obj.parameters()
        yield from obj.buffers()
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _graph_tensors(v, seen, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _graph_tensors(v, seen, depth + 1)
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        for v in vars(obj).values():
            yield from _graph_tensors(v, seen, depth + 1)


def check_ema_state_is_detached_and_independent(ns):
    model, ema, _ = _run_ema(ns)
    with_graph = [t for t in _graph_tensors(ema, set()) if t.grad_fn is not None]
    assert not with_graph, (
        "EMA state carries an autograd graph (tensor with grad_fn) — update under "
        "torch.no_grad(), otherwise every update keeps the previous graph alive")
    target = _fresh_target()
    ema.copy_to(target)
    for p in target.parameters():
        assert isinstance(p, nn.Parameter) and p.is_leaf and p.requires_grad, \
            "copy_to must copy values into the existing Parameters (they must stay trainable leaves)"
    snapshot = {n: p.detach().clone() for n, p in target.named_parameters()}
    with torch.no_grad():                          # training the target must not touch the EMA
        for p in target.parameters():
            p.add_(1.0)
    target2 = _fresh_target(seed=5)
    ema.copy_to(target2)
    for n, p in target2.named_parameters():
        assert torch.allclose(p.detach(), snapshot[n], atol=1e-6), (
            f"modifying a model after copy_to changed the EMA ({n}) — copy values, don't "
            "share storage between the shadow and the model")


CHECKS = [
    _sgd_trajectory_check("plain", 0.0, 0.0, False),
    _sgd_trajectory_check("momentum", 0.9, 0.0, False),
    _sgd_trajectory_check("weight_decay", 0.0, 0.1, False),
    _sgd_trajectory_check("momentum_and_weight_decay", 0.9, 0.1, False),
    _sgd_trajectory_check("nesterov", 0.9, 0.0, True),
    _sgd_trajectory_check("nesterov_and_weight_decay", 0.9, 0.1, True),
    # weight_decay=0 here on purpose: then g IS p.grad, so a momentum buffer that is not
    # cloned on the first step aliases p.grad and gets wiped by the in-place zeroing.
    _sgd_trajectory_check("momentum_when_grads_are_zeroed_in_place", 0.9, 0.0, False, True),
    _sgd_trajectory_check("nesterov_when_grads_are_zeroed_in_place", 0.9, 0.0, True, True),
    # The LR changes every step (warmup then cosine): catches the textbook velocity form
    # v = mu*v - lr*g, which equals PyTorch's convention only at a constant LR.
    _sgd_trajectory_check("momentum_with_changing_lr", 0.9, 0.1, False, lrs=_CHANGING_LRS),
    _sgd_trajectory_check("nesterov_with_changing_lr", 0.9, 0.0, True, lrs=_CHANGING_LRS),
    check_sgd_zero_grad_clears_gradients,
    check_lr_warmup_is_linear_from_base_over_warmup,
    check_lr_cosine_decay_key_points,
    check_lr_edge_cases,
    check_clip_matches_torch_when_clipping,
    check_clip_leaves_small_gradients_alone,
    check_clip_zero_and_missing_grads,
    check_ema_matches_closed_form,
    check_ema_extreme_decays,
    check_ema_copies_buffers,
    check_ema_state_is_detached_and_independent,
]
