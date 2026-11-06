"""cnn-param-init — a factory of weight-initialisation strategies.

`config_init(name)` returns something you can hand to `Module.apply`, so the
checks apply it to a throwaway CNN and measure what came out. Nothing is compared
against a reference implementation: each strategy is defined by the *distribution*
it produces (the uniform and normal flavours of Xavier and Kaiming have the same
standard deviation, so both pass). On square layers (fan_in == fan_out == n) the
two must give sqrt(1/n) and sqrt(2/n); on non-square layers Xavier must use
2 / (fan_in + fan_out), which rules out a plain 1/fan_in (LeCun) scale, and
Kaiming may use either fan_in or fan_out, as He et al. allow.
"""
import math

import torch
import torch.nn as nn

ENTRIES = ["config_init"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "config_init returns the initializer *function*, it does not run it — you use "
    "it as `model.apply(config_init('xavier'))`.",
    "Each initializer takes one module and should only touch it when it is an "
    "nn.Conv2d or nn.Linear; apply() walks containers and activations too.",
    "torch.nn.init has all four: zeros_, normal_, xavier_normal_ (variance "
    "2/(fan_in + fan_out)) and kaiming_normal_ with nonlinearity='relu' (variance "
    "2/fan_in). Do not forget the bias.",
]

STRATEGIES = ("zeros", "random", "xavier", "kaiming")
CONV_FAN = 32 * 3 * 3        # fan_in == fan_out for a 32->32 3x3 conv
FC_FAN = 32 * 4 * 4          # ...and for the Linear that follows the pool


class _Probe(nn.Module):
    """Square layers, plus modules an initializer must learn to ignore."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(FC_FAN, FC_FAN)

    def forward(self, x):
        return self.fc(self.pool(self.relu(self.conv(x))).flatten(1))


def _initialised(ns, strategy):
    torch.manual_seed(0)
    model = _Probe()
    init = ns.config_init(strategy)
    assert init is not None, (
        f"config_init({strategy!r}) returned None — every strategy in "
        f"{list(STRATEGIES)} needs an entry in the dict")
    assert callable(init), (
        f"config_init({strategy!r}) returned a {type(init).__name__}; it should "
        "return a function that Module.apply can call on each submodule")
    model.apply(init)
    return model


def check_every_strategy_returns_an_initializer(ns):
    for strategy in STRATEGIES:
        init = ns.config_init(strategy)
        assert init is not None and callable(init), (
            f"config_init({strategy!r}) gave {init!r}; each of "
            f"{list(STRATEGIES)} must map to a callable initializer")


def check_survives_apply_on_a_whole_model(ns):
    """apply() visits ReLU, MaxPool and the container itself, not just the layers."""
    for strategy in STRATEGIES:
        model = _initialised(ns, strategy)
        out = model(torch.randn(2, 32, 8, 8))
        assert tuple(out.shape) == (2, FC_FAN), (
            f"after {strategy!r} initialisation the model returned "
            f"{tuple(out.shape)} — an initializer must not change any shapes")


def check_zeros_sets_everything_to_zero(ns):
    model = _initialised(ns, "zeros")
    for name, p in model.named_parameters():
        assert torch.all(p == 0), (
            f"{name} is not all zero after the zeros strategy "
            f"(max |value| {float(p.abs().max()):.4f}) — weights *and* biases go "
            "to zero, for both Conv2d and Linear")


def check_random_is_a_unit_normal(ns):
    model = _initialised(ns, "random")
    for name, w in (("conv.weight", model.conv.weight), ("fc.weight", model.fc.weight)):
        std = float(w.detach().std())
        assert 0.75 < std < 1.35, (
            f"{name} has std {std:.4f} after the random strategy; the problem asks "
            "for a plain standard normal (nn.init.normal_ defaults, std 1.0)")


def _expect_std(ns, strategy, factor):
    model = _initialised(ns, strategy)
    for name, w, fan in (("conv.weight", model.conv.weight, CONV_FAN),
                         ("fc.weight", model.fc.weight, FC_FAN)):
        want = math.sqrt(factor / fan)
        std = float(w.detach().std())
        assert 0.75 * want < std < 1.3 * want, (
            f"{name} has std {std:.4f} after the {strategy!r} strategy, but "
            f"variance {factor}/n with n={fan} means std {want:.4f} — "
            f"off by {std / want:.2f}x")


def check_xavier_scales_by_one_over_n(ns):
    """Xavier: variance 1/n, i.e. 2/(fan_in + fan_out)."""
    _expect_std(ns, "xavier", 1.0)


def check_kaiming_scales_by_two_over_n(ns):
    """Kaiming He: variance 2/n — twice Xavier, to survive ReLU."""
    _expect_std(ns, "kaiming", 2.0)


class _Rect(nn.Module):
    """Non-square layers, where fan_in != fan_out tells the formulas apart."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 64, kernel_size=3)    # fan_in 144, fan_out 576
        self.fc = nn.Linear(512, 64)                     # fan_in 512, fan_out 64

    def fans(self):
        return (("conv.weight", self.conv.weight, 16 * 9, 64 * 9),
                ("fc.weight", self.fc.weight, 512, 64))


def _rect_initialised(ns, strategy):
    torch.manual_seed(0)
    model = _Rect()
    model.apply(ns.config_init(strategy))
    return model


def check_xavier_uses_fan_in_plus_fan_out(ns):
    """Glorot: variance 2 / (fan_in + fan_out), not 1 / fan_in."""
    model = _rect_initialised(ns, "xavier")
    for name, w, fan_in, fan_out in model.fans():
        want = math.sqrt(2.0 / (fan_in + fan_out))
        std = float(w.detach().std())
        assert abs(std / want - 1) < 0.06, (
            f"{name} (fan_in={fan_in}, fan_out={fan_out}) has std {std:.4f} after "
            f"'xavier'; Glorot's variance 2/(fan_in + fan_out) means std {want:.4f}. "
            f"1/fan_in alone would give {math.sqrt(1 / fan_in):.4f}")


def check_kaiming_uses_one_fan(ns):
    """He: variance 2 / fan_in (or 2 / fan_out), with the ReLU gain of 2."""
    model = _rect_initialised(ns, "kaiming")
    for name, w, fan_in, fan_out in model.fans():
        std = float(w.detach().std())
        options = {"fan_in": math.sqrt(2.0 / fan_in), "fan_out": math.sqrt(2.0 / fan_out)}
        assert any(abs(std / v - 1) < 0.06 for v in options.values()), (
            f"{name} (fan_in={fan_in}, fan_out={fan_out}) has std {std:.4f} after "
            f"'kaiming'; variance 2/fan_in means std {options['fan_in']:.4f} "
            f"(2/fan_out: {options['fan_out']:.4f})")


def check_strategies_actually_differ(ns):
    """Four names must not quietly be the same initializer."""
    stds = {}
    for strategy in STRATEGIES:
        model = _initialised(ns, strategy)
        stds[strategy] = float(model.conv.weight.detach().std())
    assert len(set(round(v, 4) for v in stds.values())) == len(STRATEGIES), (
        f"different strategies produced the same weight spread: {stds} — each "
        "name should map to its own initializer")


CHECKS = [
    check_every_strategy_returns_an_initializer,
    check_survives_apply_on_a_whole_model,
    check_zeros_sets_everything_to_zero,
    check_random_is_a_unit_normal,
    check_xavier_scales_by_one_over_n,
    check_kaiming_scales_by_two_over_n,
    check_xavier_uses_fan_in_plus_fan_out,
    check_kaiming_uses_one_fan,
    check_strategies_actually_differ,
]
