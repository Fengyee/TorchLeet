"""mlp-backprop — an MLP trained with hand-written backprop (no autograd).

Every gradient is compared with torch.autograd on an equivalent graph that the
spec builds from the *same* weights (written into the solver's layers), in
float64. The solver's code always runs under torch.no_grad() on tensors with
requires_grad=False, so an implementation that quietly calls .backward() or
torch.autograd.grad cannot pass.
"""
import math

import torch
import torch.nn.functional as F

ENTRIES = ["Linear", "ReLU", "softmax_cross_entropy", "MLP"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Write the chain rule for one element first: y[n, o] = sum_i x[n, i] * W[o, i] + b[o], so "
    "dL/dW[o, i] = sum over the batch of dL/dy[n, o] * x[n, i]. Every gradient has exactly the "
    "shape of the thing it is the gradient of — use that to pick the matmul.",
    "Linear.forward must cache its input. backward: grad_W = grad_out.T @ x  (out, in), "
    "grad_b = grad_out summed over the batch, grad_x = grad_out @ W. ReLU lets the gradient "
    "through only where the forward input was strictly > 0.",
    "Softmax-CE: subtract each row's max before exp (log-sum-exp), loss = mean(logsumexp(z) - "
    "z[target]); its gradient w.r.t. the logits is (softmax(z) - onehot(target)) / N. "
    "MLP.backward walks self.layers in reverse; step() does W -= lr * grad_W, b -= lr * grad_b.",
]

F64 = torch.float64


# ----------------------------------------------------------------- helpers

def _no_autograd(fn, *args):
    """Run solver code the way the problem demands: grad mode off, plain tensors."""
    try:
        with torch.no_grad():
            return fn(*args)
    except RuntimeError as e:
        msg = str(e)
        if "require grad" in msg or "grad_fn" in msg:
            raise AssertionError(
                "your code tried to use autograd (" + msg.splitlines()[0][:90] + ") — "
                "this problem must compute every gradient by hand; it is run under "
                "torch.no_grad() on tensors with requires_grad=False") from None
        raise


def _close(got, want, what, rtol=1e-7, atol=1e-9):
    assert got is not None, f"{what} is None"
    got = torch.as_tensor(got)
    assert tuple(got.shape) == tuple(want.shape), \
        f"{what} has shape {tuple(got.shape)}, expected {tuple(want.shape)}"
    got = got.to(F64)
    assert torch.isfinite(got).all(), f"{what} contains NaN/Inf"
    err = (got - want).abs().max().item()
    assert torch.allclose(got, want, rtol=rtol, atol=atol), \
        f"{what} disagrees with torch.autograd (max abs diff {err:.3e})"


def _linears(mlp):
    layers = getattr(mlp, "layers", None)
    assert layers is not None, \
        "MLP must keep its layers, in forward order, in a list attribute `self.layers`"
    return [layer for layer in layers if hasattr(layer, "W")]


def _ref_mlp(x, Ws, bs):
    """Equivalent autograd graph: Linear -> ReLU -> ... -> Linear (no ReLU on logits)."""
    h = x
    for k, (W, b) in enumerate(zip(Ws, bs)):
        h = F.linear(h, W, b)
        if k < len(Ws) - 1:
            h = F.relu(h)
    return h


def _spiral(n=100, noise=0.15, turns=1.25, seed=0):
    """Two interleaved spirals in [-1, 1]^2 — not linearly separable."""
    g = torch.Generator().manual_seed(seed)
    X, Y = [], []
    for c in range(2):
        r = torch.linspace(0.1, 1.0, n)
        t = (torch.linspace(0, turns * 2 * math.pi, n) + c * math.pi
             + noise * torch.randn(n, generator=g))
        X.append(torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=1))
        Y.append(torch.full((n,), c, dtype=torch.long))
    return torch.cat(X), torch.cat(Y)


# ------------------------------------------------------------------ checks

def check_linear_init_shapes_and_he_variance(ns):
    torch.manual_seed(0)
    lin = ns.Linear(256, 512)
    W, b = getattr(lin, "W", None), getattr(lin, "b", None)
    assert W is not None and b is not None, "Linear must store its parameters as .W and .b"
    assert tuple(W.shape) == (512, 256), \
        f"W must be (out_features, in_features) = (512, 256) like nn.Linear, got {tuple(W.shape)}"
    assert tuple(b.shape) == (512,), f"b must have shape (out_features,) = (512,), got {tuple(b.shape)}"
    var = W.to(F64).var().item()
    target = 2.0 / 256
    assert 0.85 * target < var < 1.15 * target, (
        f"Var(W) = {var:.2e}, but He/Kaiming init for ReLU nets has Var(W) = 2 / fan_in = "
        f"{target:.2e} (uniform: U(-sqrt(6/fan_in), sqrt(6/fan_in)))")
    assert abs(W.to(F64).mean().item()) < 0.05 * math.sqrt(target), "W should be zero-mean"
    assert torch.isfinite(b).all() and b.abs().max().item() <= math.sqrt(6 / 256), \
        "b should start at zero (or at least small)"


def check_linear_forward_backward_match_autograd(ns):
    torch.manual_seed(1)
    N, I, O = 7, 5, 4
    lin = ns.Linear(I, O)
    W, b = torch.randn(O, I, dtype=F64), torch.randn(O, dtype=F64)
    lin.W, lin.b = W.clone(), b.clone()
    x, g = torch.randn(N, I, dtype=F64), torch.randn(N, O, dtype=F64)

    y = _no_autograd(lin.forward, x)
    gx = _no_autograd(lin.backward, g)

    Wr, br, xr = (t.clone().requires_grad_() for t in (W, b, x))
    yr = F.linear(xr, Wr, br)
    yr.backward(g)
    _close(y, yr.detach(), "Linear.forward output")
    _close(getattr(lin, "grad_W", None), Wr.grad, "Linear.grad_W (expected layout (out, in))")
    _close(getattr(lin, "grad_b", None), br.grad, "Linear.grad_b")
    _close(gx, xr.grad, "the input gradient returned by Linear.backward")

    # backward() stores (overwrites) the gradient of the latest call; it must not accumulate.
    x2, g2 = torch.randn(N, I, dtype=F64), torch.randn(N, O, dtype=F64)
    _no_autograd(lin.forward, x2)
    _no_autograd(lin.backward, g2)
    _close(lin.grad_W, g2.T @ x2,
           "grad_W after a second forward/backward (it must be overwritten, not accumulated, "
           "and computed from the input cached by the latest forward)")


def check_relu_matches_autograd_including_exact_zero(ns):
    x = torch.tensor([[-2.0, -0.5, 0.0, 0.5, 3.0],
                      [0.0, 1e-12, -1e-12, 7.0, -7.0]], dtype=F64)
    g = torch.arange(1.0, 11.0, dtype=F64).reshape(2, 5)
    relu = ns.ReLU()
    y = _no_autograd(relu.forward, x)
    gx = _no_autograd(relu.backward, g)
    xr = x.clone().requires_grad_()
    F.relu(xr).backward(g)
    _close(y, F.relu(x), "ReLU.forward output")
    gx = torch.as_tensor(gx).to(F64)
    zero = x == 0
    assert torch.equal(gx[zero], xr.grad[zero]), (
        f"ReLU gradient at x == 0 exactly is {gx[zero].tolist()}; PyTorch's convention is 0 "
        f"(the mask is x > 0, not x >= 0)")
    _close(gx, xr.grad, "ReLU.backward input gradient")


def check_softmax_cross_entropy_matches_torch(ns):
    torch.manual_seed(2)
    for N, C in ((6, 5), (1, 3), (32, 10)):
        z = 3 * torch.randn(N, C, dtype=F64)
        t = torch.randint(0, C, (N,))
        loss, grad = _no_autograd(ns.softmax_cross_entropy, z, t)
        zr = z.clone().requires_grad_()
        ref = F.cross_entropy(zr, t)
        ref.backward()
        assert torch.as_tensor(loss).numel() == 1, "loss must be a scalar (mean over the batch)"
        got = float(torch.as_tensor(loss))
        assert abs(got - ref.item()) <= 1e-9 * max(1.0, abs(ref.item())), (
            f"loss {got:.6f} != F.cross_entropy {ref.item():.6f} for N={N}, C={C} "
            f"(mean over the batch of -log softmax(z)[target])")
        _close(grad, zr.grad, f"grad_logits for N={N}, C={C} (should be (softmax - onehot) / N)")


def check_softmax_cross_entropy_stable_on_extreme_logits(ns):
    z = torch.tensor([[1000.0, -1000.0, 0.0],
                      [-1000.0, 1000.0, 1000.0],
                      [1000.0, 1000.0, 1000.0],
                      [-1000.0, -1000.0, -1000.0],
                      [999.0, 1000.0, -1000.0]], dtype=F64)
    t = torch.tensor([1, 0, 2, 1, 0])
    loss, grad = _no_autograd(ns.softmax_cross_entropy, z, t)
    zr = z.clone().requires_grad_()
    ref = F.cross_entropy(zr, t)
    ref.backward()
    got = float(torch.as_tensor(loss))
    assert math.isfinite(got), (
        f"loss is {got} on logits of magnitude 1e3 — exp(1000) overflows; subtract the row max "
        f"(log-sum-exp) and never take log of a softmax probability that can underflow to 0")
    assert abs(got - ref.item()) <= 1e-9 * abs(ref.item()), \
        f"loss {got:.6f} != F.cross_entropy {ref.item():.6f} on extreme logits"
    _close(grad, zr.grad, "grad_logits on extreme logits")


def check_mlp_forward_backward_match_autograd(ns):
    sizes = [6, 9, 7, 4]
    torch.manual_seed(3)
    mlp = ns.MLP(sizes)
    lins = _linears(mlp)
    assert len(lins) == len(sizes) - 1, \
        f"MLP({sizes}) should contain {len(sizes) - 1} Linear layers, found {len(lins)}"
    for k, lin in enumerate(lins):
        assert tuple(lin.W.shape) == (sizes[k + 1], sizes[k]), \
            f"layer {k}: W is {tuple(lin.W.shape)}, expected {(sizes[k + 1], sizes[k])}"
    Ws = [0.6 * torch.randn(o, i, dtype=F64) for i, o in zip(sizes[:-1], sizes[1:])]
    bs = [0.3 * torch.randn(o, dtype=F64) for o in sizes[1:]]
    for lin, W, b in zip(lins, Ws, bs):
        lin.W, lin.b = W.clone(), b.clone()

    N = 10
    x = torch.randn(N, sizes[0], dtype=F64)
    y = torch.randint(0, sizes[-1], (N,))
    Wr = [W.clone().requires_grad_() for W in Ws]
    br = [b.clone().requires_grad_() for b in bs]
    xr = x.clone().requires_grad_()
    logits_ref = _ref_mlp(xr, Wr, br)
    logits_ref.retain_grad()
    F.cross_entropy(logits_ref, y).backward()

    logits = _no_autograd(mlp.forward, x)
    _close(logits, logits_ref.detach(),
           "MLP.forward logits (Linear -> ReLU -> ... -> Linear, no ReLU after the last layer)")
    gx = _no_autograd(mlp.backward, logits_ref.grad.clone())
    for k, lin in enumerate(lins):
        _close(getattr(lin, "grad_W", None), Wr[k].grad, f"grad_W of Linear layer {k}")
        _close(getattr(lin, "grad_b", None), br[k].grad, f"grad_b of Linear layer {k}")
    assert gx is not None, "MLP.backward should return the gradient w.r.t. the MLP input"
    _close(gx, xr.grad, "the input gradient returned by MLP.backward")


def check_mlp_step_is_plain_sgd(ns):
    sizes = [3, 5, 2]
    torch.manual_seed(4)
    mlp = ns.MLP(sizes)
    lins = _linears(mlp)
    for lin in lins:
        lin.W = torch.randn_like(lin.W, dtype=F64)
        lin.b = torch.randn_like(lin.b, dtype=F64)
    x, y = torch.randn(4, 3, dtype=F64), torch.tensor([0, 1, 1, 0])
    logits = _no_autograd(mlp.forward, x)
    p = torch.softmax(torch.as_tensor(logits), dim=1)
    _no_autograd(mlp.backward, (p - F.one_hot(y, 2)) / 4)
    before = [(lin.W.clone(), lin.b.clone(), lin.grad_W.clone(), lin.grad_b.clone()) for lin in lins]
    lr = 0.1
    _no_autograd(mlp.step, lr)
    for k, (lin, (W, b, gW, gb)) in enumerate(zip(lins, before)):
        assert torch.allclose(lin.W, W - lr * gW, atol=1e-12), \
            f"layer {k}: after step(lr) W should equal W - lr * grad_W"
        assert torch.allclose(lin.b, b - lr * gb, atol=1e-12), \
            f"layer {k}: after step(lr) b should equal b - lr * grad_b"


def check_mlp_learns_two_spirals(ns):
    X, Y = _spiral()
    torch.manual_seed(0)
    mlp = ns.MLP([2, 32, 32, 2])
    lr, steps = 0.5, 300

    def train():
        first, _ = ns.softmax_cross_entropy(mlp.forward(X), Y)
        for _ in range(steps):
            loss, grad = ns.softmax_cross_entropy(mlp.forward(X), Y)
            mlp.backward(grad)
            mlp.step(lr)
        logits = mlp.forward(X)
        last, _ = ns.softmax_cross_entropy(logits, Y)
        return float(torch.as_tensor(first)), float(torch.as_tensor(last)), logits

    first, last, logits = _no_autograd(train)
    acc = (torch.as_tensor(logits).argmax(dim=1) == Y).float().mean().item()
    assert math.isfinite(last), f"training diverged (loss {last}) — check the gradient scale and step sign"
    assert last < 0.2 and last < 0.3 * first, (
        f"full-batch SGD (lr={lr}, {steps} steps) on two spirals only took the loss from "
        f"{first:.3f} to {last:.3f} (accuracy {acc:.0%}); a correct MLP([2, 32, 32, 2]) gets below 0.05")
    assert acc >= 0.95, f"training accuracy on two spirals is only {acc:.0%}"


CHECKS = [
    check_linear_init_shapes_and_he_variance,
    check_linear_forward_backward_match_autograd,
    check_relu_matches_autograd_including_exact_zero,
    check_softmax_cross_entropy_matches_torch,
    check_softmax_cross_entropy_stable_on_extreme_logits,
    check_mlp_forward_backward_match_autograd,
    check_mlp_step_is_plain_sgd,
    check_mlp_learns_two_spirals,
]
