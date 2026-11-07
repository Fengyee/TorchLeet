"""temperature-sampling — sample from softmax(logits / T).

The load-bearing check is distributional: on a 5-token vocabulary the empirical
frequencies over thousands of draws must match softmax(logits / T) for T < 1 and
T > 1. That rejects multiplying by T, applying T twice, or ignoring it, which the
greedy/diverse smoke checks alone would let through.
"""
import torch

ENTRIES = ["temperature_sample"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Divide the logits by the temperature before the softmax.",
    "Low temperature sharpens the distribution toward the argmax; high flattens it. "
    "temperature == 0 is the limit of that: return the argmax instead of dividing by zero.",
    "Sample with torch.multinomial on the resulting probabilities.",
]

V = 40


def _tok(x):
    return int(x.item()) if torch.is_tensor(x) else int(x)


def check_returns_valid_token(ns):
    torch.manual_seed(0)
    logits = torch.randn(V)
    for _ in range(20):
        assert 0 <= _tok(ns.temperature_sample(logits, temperature=1.0)) < V, \
            "sampled token outside the vocabulary"


def check_low_temperature_is_greedy(ns):
    torch.manual_seed(0)
    logits = torch.randn(V)
    best = int(logits.argmax())
    for _ in range(15):
        t = _tok(ns.temperature_sample(logits, temperature=0.01))
        assert t == best, \
            f"temperature 0.01 should be effectively greedy; expected {best}, got {t}"


def check_high_temperature_is_diverse(ns):
    torch.manual_seed(0)
    logits = torch.randn(V) * 5      # very peaked before temperature is applied
    seen = {_tok(ns.temperature_sample(logits, temperature=100.0)) for _ in range(200)}
    assert len(seen) > 5, \
        f"temperature 100 should flatten the distribution, but only {len(seen)} token(s) appeared"


def _tv(a, b):
    return 0.5 * float((a - b).abs().sum())


def check_samples_follow_softmax_of_scaled_logits(ns):
    """Many draws on a tiny vocabulary: frequencies must match softmax(logits / T).
    Sampling noise keeps the total-variation distance under ~0.03 at n=4000;
    T*logits, T**2 or a forgotten T all land at >= 0.10."""
    logits = torch.tensor([2.0, 1.0, 0.5, 0.0, -1.0])
    n = 4000
    torch.manual_seed(0)
    for t in (0.5, 2.0):
        want = torch.softmax(logits / t, dim=-1)
        counts = torch.zeros(5)
        for _ in range(n):
            counts[_tok(ns.temperature_sample(logits, temperature=t))] += 1
        d = _tv(counts / n, want)
        assert d < 0.05, (
            f"at temperature {t} the sampled frequencies {[round(x, 3) for x in (counts / n).tolist()]} "
            f"are {d:.3f} (total variation) away from softmax(logits / T) = "
            f"{[round(x, 3) for x in want.tolist()]} — divide the logits by T, once, before the softmax")


def check_zero_temperature_is_handled(ns):
    """T -> 0 is greedy decoding. temperature=0 must either return the argmax or be
    rejected explicitly — never divide by zero and sample from NaNs."""
    torch.manual_seed(0)
    logits = torch.randn(V)
    best = int(logits.argmax())
    try:
        got = _tok(ns.temperature_sample(logits, temperature=0.0))
    except (ValueError, AssertionError):
        return                         # an explicit "T must be > 0" is acceptable
    assert got == best, (
        f"temperature=0 should mean greedy decoding (the argmax, {best}); got {got}")


CHECKS = [check_returns_valid_token, check_low_temperature_is_greedy,
          check_high_temperature_is_diverse, check_samples_follow_softmax_of_scaled_logits,
          check_zero_temperature_is_handled]
