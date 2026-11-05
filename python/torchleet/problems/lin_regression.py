"""lin-regression — linear regression with nn.Linear, MSE and gradient descent.

Beyond "it has the right shapes and can train", the checks pin down what makes
it *linear regression*: the model is affine in its input (no hidden
nonlinearity), and it has an intercept, so it can fit y = 2x + 3 exactly.
"""
import torch
import torch.nn as nn
from torchleet.checks import common as c

ENTRIES = ["LinearRegressionModel"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "A single nn.Linear(1, 1) is the whole model.",
    "forward() just applies that layer to the input.",
    "Train with MSELoss and an SGD/Adam optimizer.",
]


def check_model_is_affine(ns):
    """f(a*x1 + (1-a)*x2) == a*f(x1) + (1-a)*f(x2) holds only for an affine map."""
    m = c.build(ns, "LinearRegressionModel")
    torch.manual_seed(1)
    x1, x2 = torch.randn(32, 1) * 50, torch.randn(32, 1) * 50
    a = torch.rand(32, 1)
    with torch.no_grad():
        lhs = m(a * x1 + (1 - a) * x2)
        rhs = a * m(x1) + (1 - a) * m(x2)
    assert torch.allclose(lhs, rhs, atol=1e-3, rtol=1e-4), (
        "the model is not affine in its input — linear regression is y = w*x + b, "
        "with no activation or hidden layer")


def check_fits_a_line_with_intercept(ns):
    """Noiseless y = 2x + 3 must be fit almost exactly — impossible without a bias."""
    m = c.build(ns, "LinearRegressionModel")
    x = torch.linspace(-2, 2, 64).unsqueeze(1)
    y = 2 * x + 3
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    for _ in range(600):
        opt.zero_grad()
        nn.functional.mse_loss(m(x), y).backward()
        opt.step()
    with torch.no_grad():
        err = (m(x) - y).abs().max().item()
    assert err < 0.05, (
        f"after training on noiseless y = 2x + 3 the max error is still {err:.3f} — "
        "does the layer have a bias term?")


CHECKS = [
    c.output_shape("LinearRegressionModel", (1,), (1,)),
    c.batch_agnostic("LinearRegressionModel", (1,), (1,)),
    c.gradients_flow("LinearRegressionModel", (1,)),
    c.can_learn("LinearRegressionModel", (1,), (1,)),
    check_model_is_affine,
    check_fits_a_line_with_intercept,
]
