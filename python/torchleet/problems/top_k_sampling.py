"""top-k-sampling — keep the k largest logits, renormalise, sample.

Besides "only top-k tokens are ever sampled", the load-bearing check is
distributional: on an 8-token vocabulary the empirical frequencies must match
softmax(logits / T) renormalised over the top k. Sampling uniformly among the
top k, or forgetting the temperature, fails it.
"""
import torch

ENTRIES = ["top_k_sample"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Keep the k largest logits and make the rest impossible to select.",
    "Setting the discarded logits to -inf before the softmax does that cleanly.",
    "torch.topk gives you both the values and their indices. Clamp k to the vocabulary "
    "size first — torch.topk raises if k > vocab_size.",
]

V = 60


def _tok(x):
    return int(x.item()) if torch.is_tensor(x) else int(x)


def check_returns_valid_token(ns):
    torch.manual_seed(0)
    logits = torch.randn(V)
    for _ in range(20):
        t = _tok(ns.top_k_sample(logits, k=10))
        assert 0 <= t < V, f"token {t} outside [0, {V})"


def check_samples_only_from_top_k(ns):
    torch.manual_seed(0)
    logits = torch.randn(V)
    for k in (1, 5, 20):
        allowed = set(torch.topk(logits, k).indices.tolist())
        for _ in range(60):
            t = _tok(ns.top_k_sample(logits, k=k))
            assert t in allowed, f"k={k}: sampled {t}, which is not among the top {k}"


def check_k_of_one_is_argmax(ns):
    torch.manual_seed(0)
    logits = torch.randn(V)
    best = int(logits.argmax())
    for _ in range(15):
        assert _tok(ns.top_k_sample(logits, k=1)) == best, \
            f"k=1 must always return the argmax ({best})"


def check_larger_k_is_more_diverse(ns):
    torch.manual_seed(0)
    logits = torch.zeros(V)
    seen = {_tok(ns.top_k_sample(logits, k=V)) for _ in range(200)}
    assert len(seen) > 1, "k=vocab on a uniform distribution returned a single token"


def _tv(a, b):
    return 0.5 * float((a - b).abs().sum())


def _freqs(ns, logits, n, **kw):
    counts = torch.zeros(logits.numel())
    for _ in range(n):
        counts[_tok(ns.top_k_sample(logits, **kw))] += 1
    return counts / n


def check_samples_follow_renormalised_top_k_softmax(ns):
    """Within the top k, tokens keep their relative softmax(logits / T) weights.
    Sampling noise keeps the TV distance under ~0.03 at n=4000; uniform-in-top-k
    or a forgotten temperature are >= 0.10 away."""
    logits = torch.tensor([1.5, 1.0, 0.2, -0.3, -1.0, 0.7, -2.0, 0.0])
    torch.manual_seed(0)
    for t in (1.0, 0.5):
        want = torch.softmax(logits / t, dim=-1)
        keep = torch.zeros_like(want, dtype=torch.bool)
        keep[torch.topk(logits, 3).indices] = True
        want = torch.where(keep, want, torch.zeros_like(want))
        want = want / want.sum()
        got = _freqs(ns, logits, 4000, k=3, temperature=t)
        d = _tv(got, want)
        assert d < 0.05, (
            f"k=3, temperature={t}: sampled frequencies {[round(x, 3) for x in got.tolist()]} are "
            f"{d:.3f} (total variation) from softmax(logits / T) renormalised over the "
            f"top 3, {[round(x, 3) for x in want.tolist()]}")


def check_k_at_least_vocab_keeps_every_token(ns):
    """k >= vocab_size filters nothing: it is plain softmax sampling (clamp k —
    torch.topk raises when k > vocab_size)."""
    logits = torch.tensor([1.0, 0.5, 0.0, -0.5, -1.0])
    want = torch.softmax(logits, dim=-1)
    torch.manual_seed(0)
    for k in (5, 12):
        try:
            got = _freqs(ns, logits, 3000, k=k, temperature=1.0)
        except RuntimeError as e:
            raise AssertionError(
                f"k={k} on a 5-token vocabulary raised {e!r} — clamp k to the "
                f"vocabulary size before torch.topk")
        d = _tv(got, want)
        assert d < 0.05, (
            f"k={k} on a 5-token vocabulary should keep every token (plain softmax "
            f"sampling), but the frequencies are {d:.3f} (total variation) away")


CHECKS = [check_returns_valid_token, check_samples_only_from_top_k,
          check_k_of_one_is_argmax, check_larger_k_is_more_diverse,
          check_samples_follow_renormalised_top_k_softmax,
          check_k_at_least_vocab_keeps_every_token]
