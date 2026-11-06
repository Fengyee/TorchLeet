"""normalization-layers — BatchNorm2d, LayerNorm, GroupNorm and (inverted) Dropout.

The norms are compared with nn.BatchNorm2d / nn.LayerNorm / nn.GroupNorm holding
the *same* (randomized) affine parameters, in float64: forward output AND the
gradients w.r.t. the input, weight and bias (driven by a random upstream
gradient, since the gradient of sum(BN(x)) w.r.t. x is ~0 and would hide bugs),
in train and eval mode. BatchNorm's running statistics are compared after
several training steps on batches of different shapes. Dropout is checked
statistically (keep-rate, unbiasedness) and on its edge cases.
"""
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

ENTRIES = ["BatchNorm2d", "LayerNorm", "GroupNorm", "Dropout"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "All three norms compute (x - mean) / sqrt(var + eps) followed by a learnable per-feature "
    "affine; they differ only in WHICH axes the statistics are taken over. For an (N, C, H, W) "
    "input: BatchNorm over (N, H, W) for each channel, LayerNorm over the trailing "
    "normalized_shape dims of each sample, GroupNorm over (C/G, H, W) for each sample and group.",
    "Normalize with the biased variance (unbiased=False). BatchNorm in training also updates "
    "running = (1 - momentum) * running + momentum * batch_stat under torch.no_grad(), using the "
    "UNBIASED variance (x n/(n-1), n = N*H*W) for running_var; in eval it normalizes with the "
    "running stats. Store them with register_buffer, and branch on self.training.",
    "GroupNorm: reshape to (N, G, -1) so each group holds C/G consecutive channels, normalize over "
    "the last dim, reshape back, then apply weight/bias of shape (C,) viewed as (1, C, 1, 1). "
    "Dropout: draw a keep-mask with P(keep) = 1 - p, return x * mask / (1 - p) in training, x "
    "itself in eval, and zeros when p == 1 (dividing by 1 - p = 0 would give NaN).",
]

F64 = torch.float64


# ----------------------------------------------------------------- helpers

def _close(got, want, what, rtol=1e-6, atol=1e-8):
    assert got is not None, f"{what} is None (did the parameter receive a gradient?)"
    assert tuple(got.shape) == tuple(want.shape), \
        f"{what} has shape {tuple(got.shape)}, expected {tuple(want.shape)}"
    got = got.to(F64)
    assert torch.isfinite(got).all(), f"{what} contains NaN/Inf"
    err = (got - want).abs().max().item()
    assert torch.allclose(got, want, rtol=rtol, atol=atol), \
        f"{what} disagrees with PyTorch (max abs diff {err:.3e})"


def _pair(mine, ref, seed):
    """Give both modules the same random affine params, in float64."""
    mine, ref = mine.double(), ref.double()
    g = torch.Generator().manual_seed(seed)
    for name in ("weight", "bias"):
        p = getattr(mine, name, None)
        assert isinstance(p, nn.Parameter), \
            f"{type(mine).__name__} must register a learnable nn.Parameter named `{name}`"
        q = getattr(ref, name)
        assert tuple(p.shape) == tuple(q.shape), \
            f"{type(mine).__name__}.{name} has shape {tuple(p.shape)}, PyTorch's is {tuple(q.shape)}"
        val = torch.randn(q.shape, generator=g, dtype=F64) * 0.5 + (1.0 if name == "weight" else 0.0)
        with torch.no_grad():
            p.copy_(val)
            q.copy_(val)
    return mine, ref


def _fwd_bwd(mod, x, gy):
    for p in mod.parameters():
        p.grad = None
    xx = x.clone().requires_grad_()
    y = mod(xx)
    assert tuple(y.shape) == tuple(x.shape), \
        f"{type(mod).__name__} output shape {tuple(y.shape)} != input shape {tuple(x.shape)}"
    y.backward(gy)
    return y.detach(), xx.grad, mod.weight.grad, mod.bias.grad


def _compare(mine, ref, x, label, seed=0):
    gy = torch.randn(x.shape, generator=torch.Generator().manual_seed(seed + 99), dtype=F64)
    y1, gx1, gw1, gb1 = _fwd_bwd(mine, x, gy)
    y2, gx2, gw2, gb2 = _fwd_bwd(ref, x, gy)
    _close(y1, y2, f"{label}: output")
    _close(gx1, gx2, f"{label}: grad w.r.t. the input")
    _close(gw1, gw2, f"{label}: grad w.r.t. weight")
    _close(gb1, gb2, f"{label}: grad w.r.t. bias")


def _x(shape, seed, scale=1.0, shift=0.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=F64) * scale + shift


_BUILTINS = {
    F: ["batch_norm", "layer_norm", "group_norm", "instance_norm", "dropout", "dropout1d",
        "dropout2d", "dropout3d", "alpha_dropout", "feature_alpha_dropout"],
    torch: ["batch_norm", "layer_norm", "group_norm", "instance_norm", "dropout",
            "native_batch_norm", "native_layer_norm", "native_group_norm", "feature_dropout"],
}


@contextlib.contextmanager
def _record_builtin_calls():
    used, saved = [], []

    def spy(label, fn):
        def wrapper(*a, **k):
            used.append(label)
            return fn(*a, **k)
        return wrapper

    for mod, names in _BUILTINS.items():
        for n in names:
            if hasattr(mod, n):
                orig = getattr(mod, n)
                saved.append((mod, n, orig))
                setattr(mod, n, spy(f"{'F' if mod is F else 'torch'}.{n}", orig))
    try:
        yield used
    finally:
        for mod, n, orig in saved:
            setattr(mod, n, orig)


# ------------------------------------------------------------------ checks

def check_parameter_and_buffer_conventions(ns):
    bn = ns.BatchNorm2d(5)
    params, buffers = dict(bn.named_parameters()), dict(bn.named_buffers())
    for name, want in (("weight", torch.ones(5)), ("bias", torch.zeros(5))):
        assert name in params, f"BatchNorm2d needs a learnable nn.Parameter `{name}` (found {list(params)})"
        assert torch.equal(params[name].detach().float(), want), \
            f"BatchNorm2d.{name} should start as {want.tolist()}"
    for name, want in (("running_mean", torch.zeros(5)), ("running_var", torch.ones(5))):
        assert name in buffers, (
            f"BatchNorm2d.{name} must be registered with self.register_buffer(...) so it moves "
            f"with .to(device) and is saved in state_dict() (buffers found: {list(buffers)})")
        assert torch.equal(buffers[name].float(), want), f"{name} should start as {want.tolist()}"
    assert "running_mean" not in params, "running stats are buffers, not trainable parameters"

    ln = ns.LayerNorm((3, 4))
    w = dict(ln.named_parameters()).get("weight")
    assert w is not None and tuple(w.shape) == (3, 4), \
        "LayerNorm((3, 4)).weight should be an nn.Parameter of shape (3, 4)"
    w = dict(ns.LayerNorm(6).named_parameters()).get("weight")
    assert w is not None and tuple(w.shape) == (6,), \
        "LayerNorm(6) must accept an int and create weight of shape (6,)"
    gn = ns.GroupNorm(2, 6)
    w = dict(gn.named_parameters()).get("weight")
    assert w is not None and tuple(w.shape) == (6,), \
        "GroupNorm(2, 6).weight should have one entry per CHANNEL: shape (6,)"


def check_batchnorm_train_mode_matches_torch(ns):
    for k, shape in enumerate([(4, 3, 5, 6), (2, 4, 3, 3), (8, 2, 1, 1)]):
        C = shape[1]
        mine, ref = _pair(ns.BatchNorm2d(C), nn.BatchNorm2d(C), seed=k)
        mine.train(), ref.train()
        x = _x(shape, seed=10 + k, scale=2.0, shift=1.5)
        _compare(mine, ref, x, f"BatchNorm2d train, x{shape}", seed=k)


def check_batchnorm_running_stats_match_torch(ns):
    shapes = [(4, 3, 5, 5), (2, 3, 4, 7), (6, 3, 2, 2), (3, 3, 6, 1), (5, 3, 3, 3)]
    for momentum in (0.1, 0.3):
        mine, ref = _pair(ns.BatchNorm2d(3, momentum=momentum), nn.BatchNorm2d(3, momentum=momentum), 7)
        mine.train(), ref.train()
        for step, shape in enumerate(shapes):
            x = _x(shape, seed=100 + step, scale=1 + step, shift=step - 2.0)
            mine(x)
            ref(x)
            for name in ("running_mean", "running_var"):
                got, want = getattr(mine, name), getattr(ref, name)
                hint = (" (update it with the UNBIASED batch variance, x n/(n-1))"
                        if name == "running_var" else "")
                assert torch.allclose(got.to(F64), want, rtol=1e-6, atol=1e-8), (
                    f"momentum={momentum}: {name} after training step {step + 1} is "
                    f"{[round(v, 5) for v in got.tolist()]}, PyTorch has "
                    f"{[round(v, 5) for v in want.tolist()]} — running = (1 - momentum) * "
                    f"running + momentum * batch_stat{hint}")
        # After training, eval mode must normalize with these running stats.
        mine.eval(), ref.eval()
        _compare(mine, ref, _x((4, 3, 3, 5), seed=5, scale=3.0, shift=-1.0),
                 f"BatchNorm2d eval after training (momentum={momentum})")


def check_batchnorm_eval_uses_running_stats_without_updating(ns):
    mine, ref = _pair(ns.BatchNorm2d(4), nn.BatchNorm2d(4), seed=3)
    rm, rv = torch.tensor([0.5, -1.0, 2.0, 0.0], dtype=F64), torch.tensor([0.25, 4.0, 1.0, 9.0], dtype=F64)
    with torch.no_grad():
        for m in (mine, ref):
            m.running_mean.copy_(rm)
            m.running_var.copy_(rv)
    mine.eval(), ref.eval()
    x = _x((3, 4, 4, 4), seed=21, scale=2.0, shift=3.0)
    _compare(mine, ref, x, "BatchNorm2d eval (must use running_mean/running_var, not batch stats)")
    assert torch.equal(mine.running_mean.to(F64), rm) and torch.equal(mine.running_var.to(F64), rv), \
        "running stats changed during an eval-mode forward — only training batches may update them"


def check_layernorm_matches_torch(ns):
    cases = [((4, 7, 16), 16), ((4, 7, 16), (7, 16)), ((2, 3, 5, 5), (3, 5, 5)), ((5, 1), 1)]
    for k, (shape, norm_shape) in enumerate(cases):
        mine, ref = _pair(ns.LayerNorm(norm_shape), nn.LayerNorm(norm_shape), seed=k)
        x = _x(shape, seed=30 + k, scale=3.0, shift=-2.0)
        for mode in ("train", "eval"):
            getattr(mine, mode)(), getattr(ref, mode)()
            _compare(mine, ref, x, f"LayerNorm({norm_shape}) {mode}, x{shape}", seed=k)


def check_groupnorm_matches_torch(ns):
    cases = [((3, 6, 4, 5), 1), ((3, 6, 4, 5), 2), ((3, 6, 4, 5), 3), ((3, 6, 4, 5), 6),
             ((2, 8, 7), 4), ((2, 4, 2, 3, 2), 2)]
    for k, (shape, G) in enumerate(cases):
        C = shape[1]
        mine, ref = _pair(ns.GroupNorm(G, C), nn.GroupNorm(G, C), seed=k)
        # per-channel offsets so that grouping the wrong channels changes the statistics
        x = _x(shape, seed=40 + k, scale=2.0)
        x = x + torch.arange(C, dtype=F64).reshape((1, C) + (1,) * (len(shape) - 2)) * 0.7
        for mode in ("train", "eval"):
            getattr(mine, mode)(), getattr(ref, mode)()
            _compare(mine, ref, x, f"GroupNorm(num_groups={G}, num_channels={C}) {mode}, x{shape}", seed=k)


def check_eps_inside_the_square_root(ns):
    # With variance ~1e-6 (comparable to eps), sqrt(var + eps) and sqrt(var) + eps differ a lot.
    x4 = _x((4, 4, 3, 3), seed=50, scale=1e-3)
    mine, ref = _pair(ns.BatchNorm2d(4, eps=1e-5), nn.BatchNorm2d(4, eps=1e-5), seed=1)
    _compare(mine, ref, x4, "BatchNorm2d train on tiny-variance input (eps goes inside sqrt)")
    mine, ref = _pair(ns.LayerNorm((3, 3), eps=1e-3), nn.LayerNorm((3, 3), eps=1e-3), seed=2)
    _compare(mine, ref, x4, "LayerNorm(eps=1e-3) on tiny-variance input (eps goes inside sqrt; use self.eps)")
    mine, ref = _pair(ns.GroupNorm(2, 4, eps=1e-4), nn.GroupNorm(2, 4, eps=1e-4), seed=3)
    _compare(mine, ref, x4, "GroupNorm(eps=1e-4) on tiny-variance input (eps goes inside sqrt; use self.eps)")


def check_dropout_eval_identity_and_edge_cases(ns):
    x = _x((64, 32), seed=60)
    d = ns.Dropout(0.5)
    d.eval()
    assert torch.equal(d(x), x), "Dropout in eval mode must return the input unchanged (no scaling)"

    d0 = ns.Dropout(0.0)
    d0.train()
    assert torch.equal(d0(x), x), "Dropout(p=0) in training must be the identity"

    d1 = ns.Dropout(1.0)
    d1.train()
    xx = x.clone().requires_grad_()
    y = d1(xx)
    assert not torch.isnan(y).any(), "Dropout(p=1) produced NaN — 0 / (1 - p) divides by zero; special-case p == 1"
    assert torch.equal(y.detach(), torch.zeros_like(x)), "Dropout(p=1) in training must output all zeros"
    if y.requires_grad:
        y.sum().backward()
        assert xx.grad is None or not torch.isnan(xx.grad).any(), "Dropout(p=1) produced NaN gradients"


def check_dropout_train_statistics(ns):
    p = 0.3
    torch.manual_seed(0)
    d = ns.Dropout(p)
    d.train()
    x = _x((400, 500), seed=61, shift=0.1)
    xx = x.clone().requires_grad_()
    x_before = x.clone()
    y = d(xx)
    assert tuple(y.shape) == tuple(x.shape), "Dropout must preserve the input shape"
    assert torch.equal(xx.detach(), x_before), "Dropout must not modify its input in place"
    yd = y.detach()
    kept = yd != 0
    scaled = x / (1 - p)
    assert torch.allclose(yd[kept], scaled[kept], rtol=1e-6, atol=1e-9), (
        "in training every surviving element must equal x / (1 - p) (inverted dropout scales at "
        "train time so that eval needs no change)")
    rate = kept.double().mean().item()
    assert abs(rate - (1 - p)) < 0.01, (
        f"Dropout(p={p}) kept {rate:.3f} of the elements; it should keep 1 - p = {1 - p:.2f} "
        f"(p is the probability of DROPPING an element)")
    ones = torch.ones(400, 500, dtype=F64)
    mean = d(ones).double().mean().item()
    assert abs(mean - 1.0) < 0.02, f"E[Dropout(x)] must equal x; for x = 1 the mean output was {mean:.3f}"
    y.backward(torch.ones_like(y))
    want = kept.to(F64) / (1 - p)
    assert xx.grad is not None and torch.allclose(xx.grad, want), \
        "the gradient of dropout must be mask / (1 - p): zero where dropped, 1/(1-p) where kept"
    again = d(x).detach() != 0
    assert not torch.equal(again, kept), "each training forward must draw a fresh random mask"


def check_does_not_call_builtin_norm_or_dropout(ns):
    x = _x((2, 4, 3, 3), seed=70)
    mods = [ns.BatchNorm2d(4), ns.LayerNorm((3, 3)), ns.GroupNorm(2, 4), ns.Dropout(0.5)]
    with _record_builtin_calls() as used:
        for m in mods:
            m.double()
            for mode in ("train", "eval"):
                getattr(m, mode)()
                m(x)
    assert not used, (
        f"your modules called {sorted(set(used))} — implement the normalization / dropout with "
        f"tensor ops (mean, var, sqrt, rand) instead of the built-in")


CHECKS = [
    check_parameter_and_buffer_conventions,
    check_batchnorm_train_mode_matches_torch,
    check_batchnorm_running_stats_match_torch,
    check_batchnorm_eval_uses_running_stats_without_updating,
    check_layernorm_matches_torch,
    check_groupnorm_matches_torch,
    check_eps_inside_the_square_root,
    check_dropout_eval_identity_and_edge_cases,
    check_dropout_train_statistics,
    check_does_not_call_builtin_norm_or_dropout,
]
