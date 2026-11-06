"""adamw — AdamW built on torch.optim.Optimizer.

MyAdamW is graded against torch's own torch.optim.AdamW, run step-for-step
from the same init on the same quadratic bowl — the trajectory must match
to float tolerance. The decoupled decay is pinned by a zero-gradient
parameter: it must still shrink by exactly (1 - lr * weight_decay). An
implementation that routes the decay through the gradient (Adam + L2) moves
it by the wrong amount, and also diverges from torch.optim.AdamW on the
trajectory check. A second lockstep run uses two param groups (one with
weight_decay=0), and a first step with |grad| ~ eps pins where eps goes.
"""
import torch

ENTRIES = ["MyAdamW"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Subclass torch.optim.Optimizer and keep per-parameter state in "
    "self.state[p] — it starts as an empty dict, so initialize it on the "
    "first step. Do the whole update under @torch.no_grad() with in-place ops "
    "(mul_, add_, addcmul_, addcdiv_).",
    "AdamW: m = b1*m + (1-b1)*g and v = b2*v + (1-b2)*g^2, then correct both "
    "by (1 - beta**step) and update p -= lr * m_hat / (sqrt(v_hat) + eps). "
    "Weight decay is p.mul_(1 - lr * weight_decay) applied directly to the "
    "parameter — do NOT add weight_decay * p to the gradient, that is L2 "
    "regularization and the adaptive denominator rescales it.",
]


def _quadratic(d=16, seed=0):
    """A quadratic bowl f(x) = mean((x - target)^2) with a fixed init."""
    g = torch.Generator().manual_seed(seed)
    target = torch.randn(d, generator=g)
    x0 = torch.randn(d, generator=g)
    return x0, target


def _run(opt_cls, x0, target, steps, **kw):
    """Drive an optimizer for `steps` steps on the bowl; return final params."""
    x = x0.clone().requires_grad_(True)
    opt = opt_cls([x], **kw)
    for _ in range(steps):
        loss = ((x - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return x.detach()


def _two_group_run(opt_cls, groups_kw, steps=50, seed=0):
    """A tiny 2-layer net whose 3 parameter tensors (matrix, vector, matrix)
    are split across two param groups with different hyperparameters."""
    g = torch.Generator().manual_seed(seed)
    init = (torch.randn(6, 4, generator=g), torch.randn(6, generator=g),
            torch.randn(2, 6, generator=g))
    X, Y = torch.randn(32, 4, generator=g), torch.randn(32, 2, generator=g)
    ps = [t.clone().requires_grad_(True) for t in init]
    opt = opt_cls([dict(params=ps[:2], **groups_kw[0]),
                   dict(params=ps[2:], **groups_kw[1])])
    for _ in range(steps):
        loss = ((torch.tanh(X @ ps[0].T + ps[1]) @ ps[2].T - Y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return [p.detach() for p in ps]


def check_adamw_matches_torch_builtin(ns):
    x0, target = _quadratic(seed=1)
    steps = 30
    kw = dict(lr=0.05, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.05)
    mine = _run(ns.MyAdamW, x0, target, steps, **kw)
    ref = _run(torch.optim.AdamW, x0, target, steps, **kw)
    dev = float((mine - ref).abs().max())
    assert torch.allclose(mine, ref, atol=1e-5, rtol=1e-4), (
        f"after {steps} steps your AdamW is {dev:.2e} away from "
        "torch.optim.AdamW run in lockstep. AdamW = Adam plus "
        "p.mul_(1 - lr * weight_decay) applied directly to the parameter — "
        "adding weight_decay * p to the gradient instead is L2 regularization "
        "and drifts away from this trajectory")


def check_adamw_decays_params_with_zero_gradient(ns):
    """Decoupled decay works even with NO gradient signal: the moments stay
    zero, so the whole step is p *= (1 - lr * weight_decay)."""
    g = torch.Generator().manual_seed(2)
    p0 = torch.randn(8, generator=g)
    lr, wd = 0.1, 0.5
    p = p0.clone().requires_grad_(True)
    opt = ns.MyAdamW([p], lr=lr, weight_decay=wd)
    p.grad = torch.zeros_like(p)
    opt.step()
    expected = p0 * (1 - lr * wd)
    assert torch.allclose(p.detach(), expected, atol=1e-7), (
        f"a parameter with a ZERO gradient must still decay to "
        f"p * (1 - lr*weight_decay) = {float(expected[0]):.6f} on coordinate 0, "
        f"got {float(p.detach()[0]):.6f}. The decay is "
        "p.mul_(1 - lr * weight_decay) on the parameter itself — if it goes "
        "through the gradient, a zero gradient means no decay (or a decay "
        "warped by the adaptive denominator)")


def check_adamw_matches_torch_with_param_groups(ns):
    """Two param groups with different lr, betas and weight decay (one group
    with none, as for biases and norm gains in practice), 50 steps."""
    groups = [dict(lr=0.02, betas=(0.9, 0.999), weight_decay=0.1),
              dict(lr=0.005, betas=(0.8, 0.99), weight_decay=0.0)]
    mine = _two_group_run(ns.MyAdamW, groups)
    ref = _two_group_run(torch.optim.AdamW, groups)
    for name, a, b in zip(("W1", "b1", "W2"), mine, ref):
        dev = float((a - b).abs().max())
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4), (
            f"with two param groups and three parameters, {name} ends up "
            f"{dev:.2e} away from torch.optim.AdamW after 50 steps. Read lr, "
            "betas, eps and weight_decay from each `group`, and keep m, v AND "
            "the step count in self.state[p] for every parameter separately")


def check_adamw_eps_goes_outside_the_sqrt(ns):
    """Hand-derived: from zero state m_hat = g and v_hat = g^2, so the first
    step is exactly -lr * g / (|g| + eps). With |g| comparable to eps this pins
    where eps goes: inside the sqrt, or added before the bias correction (the
    paper's 'eps-hat' form), the step comes out far too small."""
    g = torch.Generator().manual_seed(11)
    p0 = torch.randn(8, generator=g)
    grad = torch.randn(8, generator=g) * 1e-6     # |g| ~ eps
    lr, eps = 0.01, 1e-6
    p = p0.clone().requires_grad_(True)
    opt = ns.MyAdamW([p], lr=lr, betas=(0.9, 0.999), eps=eps, weight_decay=0.0)
    p.grad = grad.clone()
    opt.step()
    delta = p.detach() - p0
    expected = -lr * grad / (grad.abs() + eps)
    assert torch.allclose(delta, expected, atol=1e-7, rtol=1e-3), (
        f"with |grad| ~ eps the first step should be -lr * g / (|g| + eps) = "
        f"{float(expected[0]):.3e} on coordinate 0, got {float(delta[0]):.3e}. "
        "eps is added to sqrt(v_hat) — after the square root and after the "
        "bias correction: p -= lr * m_hat / (sqrt(v_hat) + eps)")


CHECKS = [
    check_adamw_matches_torch_builtin,
    check_adamw_matches_torch_with_param_groups,
    check_adamw_decays_params_with_zero_gradient,
    check_adamw_eps_goes_outside_the_sqrt,
]
