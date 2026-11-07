"""custom-dnn — a feedforward network (Linear -> ReLU -> Linear) for regression."""
import torch
from torchleet.checks import common as c

ENTRIES = ["DNNModel"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "Stack Linear layers with a nonlinearity between them.",
    "Without an activation between layers the whole stack collapses to one linear map.",
    "The last layer's out_features is your output dimension.",
]


def check_is_not_a_linear_map(ns):
    """Linear layers with no activation between them compose to one affine map,
    which cannot fit the notebook's quadratic target."""
    m = c.build(ns, "DNNModel")
    m.eval()
    torch.manual_seed(1)
    x1, x2 = torch.rand(256, 2) * 10, torch.rand(256, 2) * 10
    with torch.no_grad():
        gap = (m((x1 + x2) / 2) - (m(x1) + m(x2)) / 2).abs().max().item()
    assert gap > 1e-4, (
        "the network is affine in its input (f((a+b)/2) == (f(a)+f(b))/2 everywhere) — "
        "put a nonlinearity such as ReLU between the linear layers")


CHECKS = [
    c.output_shape("DNNModel", (2,), (1,)),
    c.batch_agnostic("DNNModel", (2,), (1,)),
    c.gradients_flow("DNNModel", (2,)),
    check_is_not_a_linear_map,
    c.can_learn("DNNModel", (2,), (1,)),
]
