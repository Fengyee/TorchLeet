"""benchmark — add benchmarking to a training loop.

Timing code is not gradeable; the model under test is. The checks pin the model
contract the problem states (flattened 28x28 image in, 10 logits out, at least
two hidden nn.Linear layers) and that it trains.
"""
import torch
import torch.nn as nn
from torchleet.checks import common as c

ENTRIES = ["SimpleNN"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "Flatten the 28x28 image (x.flatten(1)) and stack at least two hidden nn.Linear "
    "layers with a ReLU after each, ending in a Linear to 10 logits.",
    "Time with time.perf_counter(), not time.time(), and expect the first epoch to be "
    "slower: it pays one-off warm-up costs.",
    "CUDA kernels launch asynchronously: call torch.cuda.synchronize() before starting "
    "and before stopping the clock. torch.utils.benchmark.Timer does the warm-up, "
    "synchronization and repetition for you (but defaults to num_threads=1).",
]

SHAPE = (1, 28, 28)


def check_has_two_hidden_layers(ns):
    """The problem asks for at least two hidden layers: three nn.Linear in total."""
    m = c.build(ns, "SimpleNN")
    n_linear = sum(isinstance(mod, nn.Linear) for mod in m.modules())
    assert n_linear >= 3, (
        f"found {n_linear} nn.Linear layer(s); at least two hidden layers plus an "
        "output layer means at least 3")


def check_returns_logits_not_probabilities(ns):
    """nn.CrossEntropyLoss applies log-softmax itself, so the model must return
    raw logits; a softmax at the end still lets the loss go down, so only this
    catches it."""
    m = c.build(ns, "SimpleNN")
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(8, *SHAPE))
    rows = out.sum(dim=1)
    is_softmax = bool((out >= 0).all()) and torch.allclose(
        rows, torch.ones_like(rows), atol=1e-4)
    assert not is_softmax, (
        "every output row is non-negative and sums to 1, so the model ends in a "
        "softmax. Return the 10 raw logits: nn.CrossEntropyLoss applies "
        "log-softmax itself")


CHECKS = [
    c.output_shape("SimpleNN", SHAPE, (10,)),
    c.batch_agnostic("SimpleNN", SHAPE, (10,)),
    check_has_two_hidden_layers,
    check_returns_logits_not_probabilities,
    c.gradients_flow("SimpleNN", SHAPE),
    c.can_learn("SimpleNN", SHAPE, (10,), classification=True),
]
