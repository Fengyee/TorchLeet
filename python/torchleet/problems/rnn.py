"""rnn — a recurrent network, written from scratch, that folds a whole sequence
into one prediction.

The problem asks for the recurrence in raw tensor ops, so a model that wraps
nn.RNN / nn.RNNCell (or LSTM/GRU) is rejected; beyond that, parameter names and
initialization are free, so nothing is compared against a reference. The checks
pin the contract — (batch, seq_len, 1) in, (batch, 1) out for any batch and any
sequence length — and then verify the properties that make a network
*recurrent*: the prediction depends on the whole sequence, not just its last
step, and on the order of the steps, not just on which values occurred.
"""
import torch
import torch.nn as nn
from torchleet.checks import common as c

ENTRIES = ["RNNModel"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "Keep a hidden state h of shape (batch, hidden_dim), start it at zeros on every "
    "forward call, and update it once per timestep in a Python loop over seq_len.",
    "The update is h = tanh(x_t W_ih + h W_hh + b) with the same W_ih, W_hh and b at "
    "every step (store them as nn.Parameter); x_t is x[:, t] because the input is "
    "(batch, seq_len, input_dim).",
    "After the loop, the final h summarises the whole sequence: map it with "
    "nn.Linear(hidden_dim, 1) to get one prediction per sequence, shape (batch, 1).",
]

SEQ, FEAT = 10, 1


def _model(ns, seed=0):
    """Built in eval mode: every check below is an inference-time property, and a
    model with BatchNorm legitimately refuses a batch of 1 while training."""
    torch.manual_seed(seed)
    m = ns.RNNModel()
    m.eval()
    return m


def check_output_shape(ns):
    m = _model(ns)
    out = m(torch.randn(4, SEQ, FEAT))
    assert out.ndim == 2, (
        f"expected a 2D (batch, 1) prediction for a (4, {SEQ}, {FEAT}) input, "
        f"got a {out.ndim}D tensor of shape {tuple(out.shape)} — collapse the "
        "sequence dimension before the output layer")
    assert tuple(out.shape) == (4, 1), \
        f"expected (4, 1), got {tuple(out.shape)}"


def check_batch_agnostic(ns):
    m = _model(ns)
    for b in (1, 3, 8):
        out = m(torch.randn(b, SEQ, FEAT))
        assert tuple(out.shape) == (b, 1), \
            f"batch {b} gave {tuple(out.shape)}, expected {(b, 1)}"


def check_any_sequence_length(ns):
    """A recurrence has no fixed length; a flattened Linear would."""
    m = _model(ns)
    for t in (3, 10, 25):
        out = m(torch.randn(2, t, FEAT))
        assert tuple(out.shape) == (2, 1), (
            f"sequence length {t} gave {tuple(out.shape)}, expected {(2, 1)} — "
            "the model should accept any number of timesteps")


def check_whole_sequence_is_used(ns):
    """Two sequences ending on the same value must not give the same prediction.

    This is what separates a recurrence from a model that only looks at x[:, -1].
    """
    m = _model(ns)
    m.eval()
    a = torch.zeros(1, SEQ, FEAT)
    b = torch.zeros(1, SEQ, FEAT)
    a[0, :-1, 0] = 1.0          # same last step, very different history
    b[0, :-1, 0] = -1.0
    with torch.no_grad():
        ya, yb = m(a), m(b)
    assert not torch.allclose(ya, yb, atol=1e-5), (
        "two sequences that share only their final timestep produced the same "
        f"output ({ya.flatten().tolist()}) — earlier timesteps are being ignored, "
        "so the hidden state is not carrying history forward")


def check_order_of_steps_matters(ns):
    """Same values, different order -> different prediction.

    Catches a model that pools over time (mean/sum of per-step features) instead
    of carrying a state forward: pooling cannot tell these two sequences apart.
    """
    m = _model(ns)
    a = torch.zeros(1, SEQ, FEAT)
    b = torch.zeros(1, SEQ, FEAT)
    a[0, 1, 0] = 3.0            # a spike early in the sequence ...
    b[0, SEQ - 2, 0] = 3.0      # ... or late; both end on the same value (0)
    with torch.no_grad():
        ya, yb = m(a), m(b)
    assert not torch.allclose(ya, yb, atol=1e-5), (
        "moving a value to a different timestep did not change the output "
        f"({ya.flatten().tolist()}) — the model is pooling over time instead of "
        "updating a hidden state step by step")


def check_written_from_scratch(ns):
    """The exercise is the recurrence itself, so no built-in recurrent module."""
    m = _model(ns)
    builtin = [n or type(mod).__name__ for n, mod in m.named_modules()
               if isinstance(mod, (nn.RNNBase, nn.RNNCellBase))]
    assert not builtin, (
        f"found built-in recurrent module(s): {', '.join(builtin)} — implement the "
        "update h_t = tanh(x_t W_ih + h_(t-1) W_hh + b) with your own parameters")


def check_deterministic_in_eval(ns):
    m = _model(ns)
    m.eval()
    x = torch.randn(2, SEQ, FEAT)
    with torch.no_grad():
        first, second = m(x), m(x)
    assert torch.allclose(first, second, atol=1e-6), (
        "two forward passes on the same input disagree — the hidden state is "
        "probably initialised randomly instead of with zeros")


CHECKS = [
    check_output_shape,
    check_batch_agnostic,
    check_any_sequence_length,
    check_whole_sequence_is_used,
    check_order_of_steps_matters,
    check_written_from_scratch,
    check_deterministic_in_eval,
    c.gradients_flow("RNNModel", (SEQ, FEAT)),
    c.can_learn("RNNModel", (SEQ, FEAT), (1,), steps=60, lr=1e-2),
]
