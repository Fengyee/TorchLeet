"""logistic-regression — sigmoid, BCE loss, and a training loop."""
import torch
import torch.nn.functional as F

ENTRIES = ["sigmoid", "bce_loss", "train_logistic_regression"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "sigmoid(z) = 1 / (1 + exp(-z)).",
    "BCE = -mean(y*log(p) + (1-y)*log(1-p)); clamp p away from 0 and 1 first.",
    "Train with gradient descent on the BCE of sigmoid(Xw + b).",
]


def check_sigmoid_matches_torch(ns):
    z = torch.linspace(-8, 8, 40)
    got = ns.sigmoid(z)
    assert torch.allclose(got, torch.sigmoid(z), atol=1e-6), \
        f"disagrees with torch.sigmoid (max diff {(got - torch.sigmoid(z)).abs().max():.2e})"


def check_sigmoid_range_and_midpoint(ns):
    out = ns.sigmoid(torch.linspace(-50, 50, 101))
    assert bool(((out >= 0) & (out <= 1)).all()), "sigmoid must stay within [0, 1]"
    assert abs(float(ns.sigmoid(torch.zeros(1))) - 0.5) < 1e-6, "sigmoid(0) must be 0.5"


def check_sigmoid_stable_for_large_inputs(ns):
    """exp(z) / (1 + exp(z)) is inf/inf = NaN once exp(z) overflows (z > ~88 in float32)."""
    z = torch.tensor([-200.0, -100.0, 100.0, 200.0])
    out = ns.sigmoid(z)
    assert torch.isfinite(out).all(), \
        f"sigmoid gives {out.tolist()} for z = {z.tolist()} — an exp() overflowed"
    assert torch.allclose(out, torch.sigmoid(z), atol=1e-6), \
        f"sigmoid should saturate to 0 and 1 for large |z|, got {out.tolist()}"


def check_bce_matches_torch(ns):
    torch.manual_seed(0)
    p = torch.rand(50).clamp(0.02, 0.98)
    y = (torch.rand(50) > 0.5).float()
    got = ns.bce_loss(y, p)
    exp = F.binary_cross_entropy(p, y)
    assert torch.allclose(torch.as_tensor(got).float(), exp, atol=1e-4), \
        f"disagrees with F.binary_cross_entropy ({float(got):.5f} vs {float(exp):.5f})"


def check_bce_finite_at_hard_predictions(ns):
    """A sigmoid that saturates to exactly 0 or 1 must not turn the loss into inf/NaN."""
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    p = torch.tensor([0.0, 1.0, 1.0, 0.0])
    loss = torch.as_tensor(ns.bce_loss(y, p)).float()
    assert torch.isfinite(loss).all(), (
        "bce_loss is inf/NaN when a prediction is exactly 0 or 1 — keep log() away "
        "from 0 with eps (or clamp p to [eps, 1 - eps])")
    assert float(loss) > 1.0, \
        "confidently wrong predictions (p=0 for y=1) must still get a large loss"


def check_bce_is_lower_when_confident_and_right(ns):
    y = torch.tensor([1.0, 1.0, 0.0, 0.0])
    good = ns.bce_loss(y, torch.tensor([0.95, 0.93, 0.05, 0.02]))
    bad = ns.bce_loss(y, torch.tensor([0.10, 0.20, 0.85, 0.90]))
    assert float(good) < float(bad), \
        "confident correct predictions must score a lower loss than confident wrong ones"


def check_training_separates_two_classes(ns):
    torch.manual_seed(0)
    X = torch.cat([torch.randn(60, 2) - 2.5, torch.randn(60, 2) + 2.5])
    y = torch.cat([torch.zeros(60), torch.ones(60)])
    out = ns.train_logistic_regression(X, y)
    w, b = (out + (None,))[:2] if isinstance(out, tuple) else (out, None)
    assert w is not None, "train_logistic_regression returned nothing usable"
    logits = X @ torch.as_tensor(w).reshape(-1) + (float(b) if b is not None else 0.0)
    acc = ((torch.sigmoid(logits) > 0.5).float() == y).float().mean()
    assert float(acc) > 0.9, \
        f"only {float(acc):.0%} accuracy on linearly separable data — training is not converging"


def _wb(out):
    """(w, b) as flat float tensors from whatever train_logistic_regression returned."""
    assert isinstance(out, (tuple, list)) and len(out) >= 2, \
        "train_logistic_regression should return (w, b, losses)"
    w, b = out[0], out[1]
    return (torch.as_tensor(w).detach().float().reshape(-1).clone(),
            torch.as_tensor(b).detach().float().reshape(-1).clone())


def check_gradient_step_is_exact(ns):
    """Epoch 2 must be exactly one gradient-descent step on the mean BCE away from
    epoch 1 — whatever the init. Catches a flipped sign, a missing 1/N, or a wrong
    bias gradient that still happens to separate easy data."""
    torch.manual_seed(0)
    X = torch.randn(40, 3)
    y = (torch.rand(40) > 0.5).float()
    lr = 0.5

    def after(n_epochs):
        torch.manual_seed(123)        # identical init for both runs, even a random one
        return _wb(ns.train_logistic_regression(X, y, lr=lr, n_epochs=n_epochs))

    w1, b1 = after(1)
    w2, b2 = after(2)
    w = w1.clone().requires_grad_(True)
    b = b1.clone().requires_grad_(True)
    loss = F.binary_cross_entropy_with_logits(X @ w + b, y)  # mean BCE of sigmoid(Xw + b)
    gw, gb = torch.autograd.grad(loss, (w, b))
    assert torch.allclose(w2, w1 - lr * gw, atol=1e-5), (
        "the weight update is not w - lr * X^T (p - y) / N "
        f"(max diff {(w2 - (w1 - lr * gw)).abs().max():.2e}) — check the sign and the 1/N")
    assert torch.allclose(b2, b1 - lr * gb, atol=1e-5), (
        "the bias update is not b - lr * mean(p - y) "
        f"(got {b2.tolist()}, expected {(b1 - lr * gb).tolist()})")


CHECKS = [check_sigmoid_matches_torch, check_sigmoid_range_and_midpoint,
          check_sigmoid_stable_for_large_inputs,
          check_bce_matches_torch, check_bce_finite_at_hard_predictions,
          check_bce_is_lower_when_confident_and_right,
          check_gradient_step_is_exact, check_training_separates_two_classes]
