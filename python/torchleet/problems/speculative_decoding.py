"""speculative-decoding — draft K tokens with a small model, verify them in one
target pass, accept/reject so the output distribution is unchanged.

The load-bearing checks come from the algorithm's own guarantee rather than from
any reference implementation:

* run speculative_decode with the draft model *being* the target model. Then
  p(x)/q(x) == 1 for every proposal, so a correct rejection rule accepts all K,
  and at a near-zero temperature the emitted tokens must be exactly what the
  solver's own standard_decode produces greedily. That is "identical output,
  fewer target calls" stated as an executable test.
* with a genuinely different draft, the first emitted token is sampled many
  times and its empirical distribution must sit far closer to the *target*
  model's next-token distribution than to the draft's. An implementation that
  just trusts the draft, or that drops the corrected-residual resampling, fails.
* the full guarantee: with hand-built bigram draft/target tables on a 3-token
  vocabulary, the JOINT distribution of the first three generated tokens (K=2,
  so a fully accepted round reaches the bonus token) must equal the target's
  exact joint, computed by enumeration. This is what catches a bonus token drawn
  from the wrong distribution, a residual that is p instead of max(0, p - q), or
  a round that keeps going after its first rejection — bugs the first-token
  check cannot see.
"""
import itertools
import torch
import torch.nn.functional as F

ENTRIES = ["DraftModel", "TargetModel", "standard_decode", "speculative_decode"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Draft phase: sample K tokens from the draft one at a time, keeping the draft "
    "probability q(x_i) of each. Verification: ONE target forward over "
    "prompt + all K drafts gives every p(x_i) at once.",
    "Accept token i with probability min(1, p(x_i)/q(x_i)). On rejection, stop and "
    "sample the replacement from the normalised residual max(0, p - q) — that "
    "correction is what keeps the output distribution exactly the target's.",
    "The target logits that score draft token i live at position "
    "prompt_len - 1 + i. If all K are accepted you get a free bonus token from "
    "position prompt_len + K - 1.",
]

VOCAB, D_DRAFT, D_TARGET = 24, 16, 32
K = 4


def _models(ns, seed=0):
    torch.manual_seed(seed)
    draft = ns.DraftModel(VOCAB, D_DRAFT).eval()
    target = ns.TargetModel(VOCAB, D_TARGET).eval()
    return draft, target


def _prompt(n=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (1, n), generator=g)


def _next_probs(model, ids, temperature=1.0):
    with torch.no_grad():
        return F.softmax(model(ids)[0, -1, :] / temperature, dim=-1)


# --------------------------------------------------------------------------
# DraftModel / TargetModel
# --------------------------------------------------------------------------
def check_model_output_shapes(ns):
    draft, target = _models(ns)
    ids = torch.randint(0, VOCAB, (2, 7))
    for name, m in (("DraftModel", draft), ("TargetModel", target)):
        with torch.no_grad():
            out = m(ids)
        assert tuple(out.shape) == (2, 7, VOCAB), (
            f"{name} must return next-token logits for every position, "
            f"(batch, seq_len, vocab_size) = {(2, 7, VOCAB)}; got {tuple(out.shape)}")
        assert torch.isfinite(out).all(), f"{name} produced NaN/Inf logits"
        assert any(p.requires_grad for p in m.parameters()), \
            f"{name} has no learnable parameters"


def check_models_are_prefix_consistent(ns):
    """Verification runs the target once over prompt + drafts and reads several
    positions out of it, so position j's logits must not depend on tokens > j."""
    draft, target = _models(ns, seed=2)
    ids = torch.randint(0, VOCAB, (1, 8))
    for name, m in (("DraftModel", draft), ("TargetModel", target)):
        with torch.no_grad():
            full = m(ids)
            for cut in (3, 5, 7):
                part = m(ids[:, :cut])
                assert torch.allclose(part, full[:, :cut], atol=1e-5), (
                    f"{name}: scoring the first {cut} tokens alone gave different "
                    f"logits than scoring the whole sequence. Speculative decoding "
                    f"verifies K drafts from ONE target pass, so the logits at "
                    f"position j must depend only on tokens 0..j; max deviation "
                    f"{float((part - full[:, :cut]).abs().max()):.6f}")


def check_models_respond_to_their_input(ns):
    draft, target = _models(ns, seed=3)
    a = torch.randint(0, VOCAB, (1, 5))
    b = a.clone()
    b[0, -1] = (int(b[0, -1]) + 1) % VOCAB
    for name, m in (("DraftModel", draft), ("TargetModel", target)):
        with torch.no_grad():
            assert not torch.allclose(m(a)[0, -1], m(b)[0, -1], atol=1e-6), \
                f"{name}'s last-position logits did not change when the last token changed"


# --------------------------------------------------------------------------
# standard_decode
# --------------------------------------------------------------------------
def check_standard_decode_extends_the_prompt(ns):
    _, target = _models(ns, seed=4)
    prompt = _prompt(3)
    with torch.no_grad():
        out = ns.standard_decode(target, prompt, 6, temperature=1.0)
    assert tuple(out.shape) == (1, 3 + 6), (
        f"standard_decode returns the full sequence, prompt + num_tokens = "
        f"{(1, 9)}; got {tuple(out.shape)}")
    assert torch.equal(out[:, :3], prompt), \
        "the prompt must be preserved unchanged at the front of the output"
    assert int(out.min()) >= 0 and int(out.max()) < VOCAB, \
        f"generated ids must be in [0, {VOCAB}), got [{int(out.min())}, {int(out.max())}]"


def check_standard_decode_is_greedy_at_low_temperature(ns):
    """Temperature -> 0 collapses the softmax onto the argmax."""
    _, target = _models(ns, seed=5)
    prompt = _prompt(3, seed=6)
    with torch.no_grad():
        got = ns.standard_decode(target, prompt, 6, temperature=1e-3)
        cur = prompt.clone()
        for _ in range(6):
            nxt = target(cur)[:, -1, :].argmax(-1, keepdim=True)
            cur = torch.cat([cur, nxt], dim=1)
    assert torch.equal(got, cur), (
        "at temperature 1e-3 every next-token distribution is effectively one-hot, "
        "so decoding must follow the model's argmax path. Expected "
        f"{cur[0].tolist()}, got {got[0].tolist()} — check that the temperature "
        "divides the LOGITS before the softmax.")


def check_standard_decode_samples_at_temperature_one(ns):
    _, target = _models(ns, seed=7)
    prompt = _prompt(3, seed=8)
    torch.manual_seed(0)
    with torch.no_grad():
        runs = [ns.standard_decode(target, prompt, 10, temperature=1.0) for _ in range(4)]
    assert any(not torch.equal(runs[0], r) for r in runs[1:]), (
        "four decodes at temperature 1.0 were identical — standard_decode must "
        "sample from the distribution (torch.multinomial), not take the argmax")


# --------------------------------------------------------------------------
# speculative_decode
# --------------------------------------------------------------------------
def _spec(ns, draft, target, prompt, k=K, temperature=1.0):
    out = ns.speculative_decode(draft, target, prompt, K=k, temperature=temperature)
    assert isinstance(out, (tuple, list)) and len(out) == 2, \
        "speculative_decode must return (accepted_tokens, num_accepted)"
    toks, n = out
    toks = [int(t) for t in toks]
    return toks, int(n)


def check_speculative_output_is_well_formed(ns):
    draft, target = _models(ns, seed=9)
    prompt = _prompt(4, seed=10)
    for _ in range(20):
        with torch.no_grad():
            toks, n = _spec(ns, draft, target, prompt)
        assert 0 <= n <= K, \
            f"num_accepted counts draft tokens, so it must be in [0, {K}]; got {n}"
        assert n <= len(toks) <= n + 1, (
            f"a round emits the {n} accepted draft tokens plus one extra — either "
            f"the corrected resample after a rejection or the bonus token when all "
            f"{K} were accepted; got {len(toks)} tokens for num_accepted={n}")
        assert all(0 <= t < VOCAB for t in toks), \
            f"emitted ids must be in [0, {VOCAB}), got {toks}"


def check_identical_models_accept_every_draft(ns):
    """p == q makes the acceptance ratio exactly 1, so nothing may be rejected."""
    _, target = _models(ns, seed=11)
    prompt = _prompt(4, seed=12)
    torch.manual_seed(0)
    with torch.no_grad():
        for temp in (1.0, 0.15):
            for trial in range(25):
                toks, n = _spec(ns, target, target, prompt, temperature=temp)
                assert n == K, (
                    f"with the draft model and the target model being the SAME "
                    f"model, p(x)/q(x) = 1 for every proposal, so all {K} drafts "
                    f"must be accepted (min(1, 1) = 1 and the uniform draw is "
                    f"< 1). At temperature {temp}, trial {trial} accepted only "
                    f"{n}. The usual cause is reading the target logits from the "
                    f"wrong position: draft token i is scored by position "
                    f"prompt_len - 1 + i.")
                assert len(toks) == K + 1, (
                    f"all {K} drafts accepted means a free bonus token from the "
                    f"target's last position, so {K + 1} tokens should come back; "
                    f"got {len(toks)}")


def check_speculative_reproduces_greedy_decoding(ns):
    """The guarantee, in its cleanest form: same model + no sampling noise means
    speculative decoding and plain decoding must emit the same tokens."""
    _, target = _models(ns, seed=13)
    prompt = _prompt(4, seed=14)
    with torch.no_grad():
        toks, n = _spec(ns, target, target, prompt, temperature=1e-3)
        ref = ns.standard_decode(target, prompt, len(toks), temperature=1e-3)
    expected = ref[0, prompt.shape[1]:].tolist()
    assert toks == expected, (
        "with the draft equal to the target and temperature 1e-3 both decoders are "
        "deterministic and must agree token for token — that is the whole promise "
        f"of speculative decoding. standard_decode gave {expected}, "
        f"speculative_decode gave {toks}.")


def _tv(a, b):
    return 0.5 * float((a - b).abs().sum())


def check_emitted_token_follows_the_target_distribution(ns):
    """Sample the first emitted token many times: it must be distributed as the
    TARGET's next-token distribution, not the draft's."""
    temp = 0.2                       # sharpens both models so they disagree clearly
    draft, target = _models(ns, seed=15)
    prompt = _prompt(4, seed=16)
    p = _next_probs(target, prompt, temp)
    q = _next_probs(draft, prompt, temp)
    assert _tv(p, q) > 0.25, "internal: draft and target should differ enough to be telling"

    trials = 3000
    counts = torch.zeros(VOCAB)
    torch.manual_seed(0)
    with torch.no_grad():
        for _ in range(trials):
            toks, _ = _spec(ns, draft, target, prompt, temperature=temp)
            counts[toks[0]] += 1
    emp = counts / trials
    to_target, to_draft = _tv(emp, p), _tv(emp, q)
    assert to_target < to_draft, (
        f"over {trials} rounds the first emitted token looked more like the DRAFT "
        f"model's distribution (total-variation distance {to_draft:.3f}) than the "
        f"TARGET model's ({to_target:.3f}). Speculative decoding must not shift the "
        f"output distribution: reject with probability 1 - p/q and resample from "
        f"the normalised residual max(0, p - q).")
    assert to_target < 0.12, (
        f"the first emitted token's empirical distribution is {to_target:.3f} away "
        f"(total variation) from the target model's next-token distribution over "
        f"{trials} rounds; sampling noise alone should keep this under ~0.06. The "
        f"accept/reject rule is biased.")


def check_speculative_accepts_less_when_the_draft_disagrees(ns):
    """A mismatched draft must actually get rejected sometimes."""
    draft, target = _models(ns, seed=17)
    prompt = _prompt(4, seed=18)
    torch.manual_seed(1)
    with torch.no_grad():
        accepted = [_spec(ns, draft, target, prompt)[1] for _ in range(60)]
    assert min(accepted) < K, (
        "an independently initialised draft model proposed 60 rounds of tokens and "
        f"every single one of the {K} drafts was accepted every time — the "
        "rejection test is not doing anything (it should reject with probability "
        "1 - min(1, p/q))")
    assert max(accepted) > 0, (
        "not a single draft token was ever accepted over 60 rounds — the "
        "acceptance ratio is probably inverted (it is p/q, target over draft)")


_P = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6], [0.1, 0.7, 0.2]])   # target bigram
_Q = torch.tensor([[0.2, 0.3, 0.5], [0.5, 0.4, 0.1], [0.4, 0.2, 0.4]])   # draft bigram


class _BigramLM(torch.nn.Module):
    """(B, S) ids -> (B, S, 3) logits; position t's logits depend on token t only."""

    def __init__(self, table):
        super().__init__()
        self.log_table = torch.nn.Parameter(table.log(), requires_grad=False)

    def forward(self, input_ids):
        return self.log_table[input_ids.long()]


def check_output_distribution_equals_the_target_joint(ns):
    """Generate 3 tokens with repeated speculative rounds (K=2) and compare the
    empirical joint over all 27 outcomes with the target's exact joint.
    Sampling noise keeps the TV distance under ~0.045 at n=5000 (99.99th
    percentile); a residual of p instead of max(0, p - q) lands at ~0.2, a bonus
    token from the draft at ~0.15, a greedy bonus at ~0.12."""
    n_tok, k, trials = 3, 2, 5000
    prompt = torch.tensor([[1, 0]])
    exact = torch.zeros(3 ** n_tok)
    for i, xs in enumerate(itertools.product(range(3), repeat=n_tok)):
        prob, prev = 1.0, int(prompt[0, -1])
        for x in xs:
            prob, prev = prob * float(_P[prev, x]), x
        exact[i] = prob

    draft, target = _BigramLM(_Q).eval(), _BigramLM(_P).eval()
    counts = torch.zeros(3 ** n_tok)
    torch.manual_seed(0)
    with torch.no_grad():
        for _ in range(trials):
            ids, out = prompt, []
            while len(out) < n_tok:
                toks, _ = _spec(ns, draft, target, ids, k=k, temperature=1.0)
                assert toks, "speculative_decode emitted no tokens"
                out += toks
                ids = torch.cat([ids, torch.tensor([toks])], dim=1)
            idx = 0
            for x in out[:n_tok]:
                idx = idx * 3 + x
            counts[idx] += 1
    emp = counts / trials
    d = _tv(emp, exact)
    worst = int((emp - exact).abs().argmax())
    seq = list(itertools.product(range(3), repeat=n_tok))[worst]
    assert d < 0.07, (
        f"the first {n_tok} generated tokens are {d:.3f} away (total variation) from "
        f"the target model's exact joint distribution over {trials} runs; sampling "
        f"noise alone stays under ~0.045. Worst sequence {list(seq)}: "
        f"{float(emp[worst]):.3f} sampled vs {float(exact[worst]):.3f} exact. Check "
        f"the three pieces: accept with prob min(1, p/q); on the first rejection "
        f"sample from normalise(max(0, p - q)) and STOP the round; if all K are "
        f"accepted, sample the bonus token from the target's last position.")


CHECKS = [
    check_model_output_shapes,
    check_models_are_prefix_consistent,
    check_models_respond_to_their_input,
    check_standard_decode_extends_the_prompt,
    check_standard_decode_is_greedy_at_low_temperature,
    check_standard_decode_samples_at_temperature_one,
    check_speculative_output_is_well_formed,
    check_identical_models_accept_every_draft,
    check_speculative_reproduces_greedy_decoding,
    check_emitted_token_follows_the_target_distribution,
    check_speculative_accepts_less_when_the_draft_disagrees,
    check_output_distribution_equals_the_target_joint,
]
