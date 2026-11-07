"""alexnet — AlexNet from scratch.

Shape and gradient checks alone would pass any CNN, so the layout that makes it
AlexNet is pinned as well: five convolutions with the 11x11 / stride-4 stem,
three fully connected layers, and dropout that is active only in training.
Channel widths are free — the paper's 96-256-384-384-256 and torchvision's
64-192-384-256-256 both pass.
"""
import torch
import torch.nn as nn

from torchleet.checks import common as c

ENTRIES = ["AlexNet"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "AlexNet expects 3x224x224 input and ends in a classifier over num_classes.",
    "Five conv layers with max-pooling, then three fully connected layers.",
    "Track the spatial size through each stride and pool to size the first Linear: "
    "with an 11x11 / stride-4 / padding-2 first conv and three 3x3 / stride-2 "
    "max-pools, 224 -> 55 -> 27 -> 13 -> 6, so the first Linear takes 256*6*6.",
]


def check_five_convs_and_three_linears(ns):
    m = c.build(ns, "AlexNet")
    convs = [x for x in m.modules() if isinstance(x, nn.Conv2d)]
    fcs = [x for x in m.modules() if isinstance(x, nn.Linear)]
    assert len(convs) == 5, (
        f"found {len(convs)} nn.Conv2d layers; AlexNet has five convolutional layers")
    assert len(fcs) == 3, (
        f"found {len(fcs)} nn.Linear layers; AlexNet has three fully connected "
        "layers (two hidden 4096-wide ones, then the classifier)")
    stems = [x for x in convs if x.in_channels == 3]
    assert stems, "no convolution takes the 3-channel RGB image as input"
    stem = stems[0]
    assert tuple(stem.kernel_size) == (11, 11) and tuple(stem.stride) == (4, 4), (
        f"the first convolution is {tuple(stem.kernel_size)} with stride "
        f"{tuple(stem.stride)}; AlexNet's is 11x11 with stride 4")


def check_dropout_only_in_training(ns):
    m = c.build(ns, "AlexNet")
    x = torch.randn(2, 3, 224, 224)
    m.train()
    with torch.no_grad():
        a, b = m(x), m(x)
    assert not torch.equal(a, b), (
        "two training-mode passes on the same input gave identical outputs — "
        "AlexNet applies dropout (p=0.5) in its fully connected layers")
    m.eval()
    with torch.no_grad():
        a, b = m(x), m(x)
    assert torch.equal(a, b), \
        "two eval-mode passes disagree — dropout must switch off in eval mode"


CHECKS = [
    c.output_shape("AlexNet", (3, 224, 224), (10,), batch=2),
    c.batch_agnostic("AlexNet", (3, 224, 224), (10,), sizes=(1, 2)),
    c.gradients_flow("AlexNet", (3, 224, 224)),
    check_five_convs_and_three_linears,
    check_dropout_only_in_training,
]
