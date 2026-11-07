"""custom-activation — define tanh(x) + x and use it after a linear layer."""
import torch
import torch.nn as nn
from torchleet.checks import common as c

ENTRIES = ["CustomActivationModel"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "An activation is just a function of the tensor; write it with torch ops "
    "(as a method or a small nn.Module) so autograd can differentiate it.",
    "This one has no learnable parameters: it is torch.tanh(x) + x, applied elementwise.",
    "forward(x) is custom_activation(self.linear(x)) — the activation goes on the "
    "output of the single linear layer.",
]


def check_activation_is_nonlinear(ns):
    """A model whose activation is the identity is not doing anything."""
    m = c.build(ns, "CustomActivationModel")
    x = torch.linspace(-3, 3, 50).unsqueeze(1)
    with torch.no_grad():
        y = m(x)
    slopes = (y[1:] - y[:-1]).squeeze()
    assert float(slopes.std()) > 1e-6, \
        "output is affine in the input — the activation appears to be the identity"


def check_computes_tanh_plus_identity(ns):
    """forward(x) must equal tanh(z) + z with z the linear layer's output."""
    m = c.build(ns, "CustomActivationModel")
    linears = [mod for mod in m.modules() if isinstance(mod, nn.Linear)]
    assert len(linears) == 1, (
        f"expected a single nn.Linear layer (the problem's self.linear), found {len(linears)}")
    x = torch.linspace(-4, 4, 41).unsqueeze(1)
    with torch.no_grad():
        z = linears[0](x)
        got, want = m(x), torch.tanh(z) + z
    assert got.shape == want.shape and torch.allclose(got, want, atol=1e-5), (
        "forward(x) should be tanh(z) + z where z = self.linear(x) "
        f"(max diff {(got - want).abs().max():.2e})")


CHECKS = [
    c.output_shape("CustomActivationModel", (1,), (1,)),
    c.batch_agnostic("CustomActivationModel", (1,), (1,)),
    c.gradients_flow("CustomActivationModel", (1,)),
    check_activation_is_nonlinear,
    check_computes_tanh_plus_identity,
    c.can_learn("CustomActivationModel", (1,), (1,)),
]
