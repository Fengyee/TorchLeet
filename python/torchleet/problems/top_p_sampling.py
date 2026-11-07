"""top-p-sampling — nucleus sampling.

The nucleus is the smallest prefix of the descending-sorted distribution whose
mass reaches p, so the token that CROSSES p belongs to it. The most common bug
masks `cumsum > p` and silently drops that token; a nucleus-membership check
cannot see it (a subset of the nucleus is still inside the nucleus), so the
load-bearing checks here are distributional: with probabilities
[0.5, 0.3, 0.15, 0.05] and p=0.6 the sampled frequencies must be
[0.625, 0.375, 0, 0], and the temperature must be applied before the nucleus is
chosen.
"""
import torch

ENTRIES = ["top_p_sample"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Sort probabilities descending, then take the smallest prefix whose cumulative mass >= p "
    "— the token that crosses p is IN the nucleus.",
    "Always keep at least the top token, renormalize the survivors, and map the sampled "
    "position back through the sort indices.",
    "Drop a token when the mass ranked ABOVE it already reaches p: mask (cumsum - prob) >= p, "
    "i.e. the cumsum shifted right by one.",
]

V = 50


def _nucleus(logits, p, temperature=1.0):
    probs = torch.softmax(logits / temperature, dim=-1)
    s, idx = probs.sort(descending=True)
    keep = (s.cumsum(-1) - s) < p          # prefix strictly before mass p is exceeded
    keep[0] = True                          # never drop the top token
    return set(idx[keep].tolist())


def check_returns_valid_token(ns):
    torch.manual_seed(0)
    logits = torch.randn(V)
    for _ in range(20):
        t = ns.top_p_sample(logits, p=0.9)
        t = int(t.item()) if torch.is_tensor(t) else int(t)
        assert 0 <= t < V, f"token {t} outside [0, {V})"


def check_samples_only_from_nucleus(ns):
    torch.manual_seed(0)
    logits = torch.randn(V)
    allowed = _nucleus(logits, 0.5)
    for _ in range(80):
        t = ns.top_p_sample(logits, p=0.5)
        t = int(t.item()) if torch.is_tensor(t) else int(t)
        assert t in allowed, \
            f"sampled token {t} is outside the top-p=0.5 nucleus {sorted(allowed)}"


def check_tiny_p_gives_argmax(ns):
    torch.manual_seed(0)
    logits = torch.randn(V)
    best = int(logits.argmax())
    for _ in range(15):
        t = ns.top_p_sample(logits, p=0.01)
        t = int(t.item()) if torch.is_tensor(t) else int(t)
        assert t == best, \
            f"with p tiny only the top token survives; expected {best}, got {t}"


def check_large_p_can_reach_beyond_top_token(ns):
    torch.manual_seed(0)
    logits = torch.zeros(V)      # uniform: every token is plausible
    seen = set()
    for _ in range(150):
        t = ns.top_p_sample(logits, p=1.0)
        seen.add(int(t.item()) if torch.is_tensor(t) else int(t))
    assert len(seen) > 1, \
        "p=1.0 on a uniform distribution always returned one token — is it sampling?"


def _tv(a, b):
    return 0.5 * float((a - b).abs().sum())


def _freqs(ns, logits, n, **kw):
    counts = torch.zeros(logits.numel())
    for _ in range(n):
        t = ns.top_p_sample(logits, **kw)
        counts[int(t.item()) if torch.is_tensor(t) else int(t)] += 1
    return counts / n


def _expected(logits, p, temperature=1.0):
    probs = torch.softmax(logits / temperature, dim=-1)
    keep = torch.zeros_like(probs, dtype=torch.bool)
    keep[list(_nucleus(logits, p, temperature))] = True
    probs = torch.where(keep, probs, torch.zeros_like(probs))
    return probs / probs.sum()


def check_token_that_crosses_p_is_kept(ns):
    """probs [0.5, 0.3, 0.15, 0.05]: the top token alone has 0.5 < p=0.6, so the
    nucleus is {0, 1} and the samples must split 0.625 / 0.375. Masking
    `cumsum > p` instead of `cumsum - prob >= p` only ever returns token 0."""
    logits = torch.tensor([0.5, 0.3, 0.15, 0.05]).log()
    torch.manual_seed(0)
    for p, n in ((0.6, 3000), (0.9, 3000)):
        want = _expected(logits, p)
        got = _freqs(ns, logits, n, p=p)
        d = _tv(got, want)
        assert d < 0.05, (
            f"probabilities [0.5, 0.3, 0.15, 0.05], p={p}: expected sampled frequencies "
            f"{[round(x, 3) for x in want.tolist()]} (the nucleus includes the token "
            f"whose cumulative mass crosses p, renormalised), got "
            f"{[round(x, 3) for x in got.tolist()]} — TV distance {d:.3f}")


def check_temperature_is_applied_before_the_nucleus(ns):
    """T reshapes the distribution, and the nucleus is chosen on the reshaped one.
    Here T=2 flattens [0.5, 0.3, 0.15, 0.05] so p=0.7 needs three tokens, not two."""
    logits = torch.tensor([0.5, 0.3, 0.15, 0.05]).log()
    want = _expected(logits, 0.7, temperature=2.0)
    torch.manual_seed(0)
    got = _freqs(ns, logits, 3000, p=0.7, temperature=2.0)
    d = _tv(got, want)
    assert d < 0.05, (
        f"temperature=2, p=0.7: expected frequencies {[round(x, 3) for x in want.tolist()]}, "
        f"got {[round(x, 3) for x in got.tolist()]} (TV {d:.3f}) — divide the logits by "
        f"the temperature, then softmax, sort and cut the nucleus")


CHECKS = [check_returns_valid_token, check_samples_only_from_nucleus,
          check_tiny_p_gives_argmax, check_large_p_can_reach_beyond_top_token,
          check_token_that_crosses_p_is_kept, check_temperature_is_applied_before_the_nucleus]
