"""beam-search — keep the top-k partial sequences by cumulative log-probability.

The load-bearing checks use a tiny deterministic language model built as a *trap*:
the winning path loses on its first step and on its last, and only wins on the
total. Two properties follow from the definition of beam search alone, so any
correct implementation passes regardless of how it stores or prunes its beams:

  * beam_width=1 is greedy decoding (oracle: argmax of the model's own logits),
  * beam_width>=2 returns the sequence with the highest cumulative log-prob,
    where "highest" is settled by brute-force enumeration, not by a second beam
    search. Scoring candidates on the current step instead of the running total
    returns the greedy path here, half a nat short, so that bug cannot pass.

A second table checks the two things a production beam search adds — EOS and
length normalisation. One hypothesis ends early with EOS and wins on the raw sum
of log-probs; a long one wins once scores are divided by length. Both answers
are robust to whether a solver counts the start token or the EOS token in the
length, and to whether it stops early once beam_width hypotheses have finished.
"""
import torch

ENTRIES = ["beam_search"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Keep each beam as (sequence, cumulative_log_prob) and start from "
    "([start_token], 0.0) — the start token counts towards max_len.",
    "Score a candidate with the RUNNING total: score + log_softmax(logits)[token]. "
    "Comparing only the current step's log-prob turns beam search into greedy.",
    "Pool the candidates from every beam into one ranking (or flatten a (beams, vocab) "
    "score matrix and topk it), keep the top beam_width. A candidate ending in "
    "eos_token is finished: set it aside, never extend it.",
    "At the end rank finished hypotheses and the beams still alive at max_len by "
    "score / num_generated ** length_penalty and return the best sequence.",
]

VOCAB = 8
MAX_LEN = 4


def _table():
    """Logits row t = the model's output for token t. Two paths, deliberately crossed.

        0 -> 1 -> 4 -> 7   log-probs -0.82, -1.60, -0.02   total -2.44
        0 -> 2 -> 5 -> 7   log-probs -1.22, -0.01, -0.67   total -1.90

    The second path wins on the total while losing on both the first and the last
    step, so it is found only by ranking beams on their CUMULATIVE score. Greedy,
    and any scoring that looks at one step at a time, ends up on the first path.
    """
    t = torch.zeros(VOCAB, VOCAB)               # unlisted rows are flat: no strong move
    t[0] = -6.0
    t[0, 1], t[0, 2], t[0, 3] = 3.0, 2.6, 2.5   # start: 1 looks best, 2 is the real win
    t[1] = -0.571
    t[1, 4] = 0.0                               # token 1 -> 4, but only just
    t[2] = -6.0
    t[2, 5] = 0.5                               # token 2 -> 5, nearly certain
    t[4] = -6.0
    t[4, 7] = 0.0                               # 4 -> 7, nearly certain
    t[5] = -2.0
    t[5, 7] = 0.0                               # 5 -> 7, likely
    g = torch.Generator().manual_seed(1234)
    return t + torch.rand(VOCAB, VOCAB, generator=g) * 0.01  # break every tie


TABLE = _table()
LOGP = torch.log_softmax(TABLE, dim=-1)


class _TableLM(torch.nn.Module):
    """model(token_id) -> (1, vocab_size) logits, like the notebook's DummyLM."""

    def __init__(self):
        super().__init__()
        self.vocab_size = VOCAB

    def forward(self, token_id):
        return TABLE[_last(token_id)].unsqueeze(0)


def _last(token_id):
    """Accept an int, a list or a tensor of ids; the model is a bigram table."""
    if torch.is_tensor(token_id):
        return int(token_id.reshape(-1)[-1].item())
    if isinstance(token_id, (list, tuple)):
        return int(token_id[-1])
    return int(token_id)


def _as_ints(seq, where):
    assert seq is not None, f"{where}: beam_search returned None"
    try:
        out = [int(t) for t in seq]
    except TypeError:
        raise AssertionError(
            f"{where}: expected a list of token ids, got {type(seq).__name__}")
    return out


def _score(seq):
    """Cumulative log-probability of a decoded sequence under the test model."""
    return sum(LOGP[seq[i], seq[i + 1]].item() for i in range(len(seq) - 1))


def _brute_force_best(max_len):
    """The true argmax over every sequence — an oracle, not a beam search."""
    best, best_s = None, float("-inf")
    stack = [([0], 0.0)]
    while stack:
        seq, s = stack.pop()
        if len(seq) == max_len:
            if s > best_s:
                best, best_s = seq, s
            continue
        for nxt in range(VOCAB):
            stack.append((seq + [nxt], s + LOGP[seq[-1], nxt].item()))
    return best, best_s


def check_shape_and_start_token(ns):
    model = _TableLM()
    for bw in (1, 2, 4):
        for max_len in (2, 5, 8):
            got = _as_ints(ns.beam_search(model, 0, beam_width=bw, max_len=max_len),
                           f"beam_width={bw}, max_len={max_len}")
            assert len(got) == max_len, (
                f"beam_width={bw}: expected a sequence of length max_len={max_len} "
                f"(the start token counts), got length {len(got)}")
            assert got[0] == 0, \
                f"beam_width={bw}: the first token must be start_token=0, got {got[0]}"
            assert all(0 <= t < VOCAB for t in got), \
                f"decoded token ids outside [0, {VOCAB}): {got}"


def check_beam_width_one_is_greedy(ns):
    """With one beam there is nothing to prune, so it must be argmax decoding."""
    model = _TableLM()
    greedy = [0]
    for _ in range(MAX_LEN - 1):
        greedy.append(int(TABLE[greedy[-1]].argmax()))
    got = _as_ints(ns.beam_search(model, 0, beam_width=1, max_len=MAX_LEN), "beam_width=1")
    assert got == greedy, (
        f"beam_width=1 must reproduce greedy decoding: expected {greedy}, got {got}")


def check_wider_beam_finds_the_best_sequence(ns):
    """The defining property: keep enough beams and you find the best-scoring path."""
    model = _TableLM()
    best, best_score = _brute_force_best(MAX_LEN)
    greedy_score = _score(_as_ints(
        ns.beam_search(model, 0, beam_width=1, max_len=MAX_LEN), "beam_width=1"))
    for bw in (2, 3, 4):
        got = _as_ints(ns.beam_search(model, 0, beam_width=bw, max_len=MAX_LEN),
                       f"beam_width={bw}")
        got_score = _score(got)
        assert abs(got_score - best_score) < 1e-4, (
            f"beam_width={bw} returned {got} scoring {got_score:.3f}; the best "
            f"sequence is {best} scoring {best_score:.3f}. Greedy scores "
            f"{greedy_score:.3f} here, so a beam that big should not settle for "
            f"the greedy path — are candidates scored by their cumulative log-prob?")


def check_returns_the_top_scoring_beam(ns):
    """Different beam widths explore differently but must never score worse than greedy."""
    model = _TableLM()
    greedy_score = _score(_as_ints(
        ns.beam_search(model, 0, beam_width=1, max_len=6), "beam_width=1"))
    for bw in (2, 4):
        got = _as_ints(ns.beam_search(model, 0, beam_width=bw, max_len=6),
                       f"beam_width={bw}")
        s = _score(got)
        assert s >= greedy_score - 1e-4, (
            f"beam_width={bw} returned a sequence scoring {s:.3f}, worse than "
            f"greedy's {greedy_score:.3f} — the final answer should be the beam "
            f"with the highest cumulative score")


def check_deterministic(ns):
    model = _TableLM()
    a = _as_ints(ns.beam_search(model, 0, beam_width=3, max_len=MAX_LEN), "run 1")
    b = _as_ints(ns.beam_search(model, 0, beam_width=3, max_len=MAX_LEN), "run 2")
    assert a == b, (
        f"beam search is deterministic, but two identical calls returned {a} and {b} "
        f"— are you sampling instead of taking the top-k?")


EOS = 7
SHORT = [0, 1, EOS]                      # wins on the raw sum of log-probs
LONG = [0, 2, 3, 4, 5, 6, 3, 4]          # wins once scores are divided by length


def _eos_table():
    """0 -> 1 -> EOS totals -1.13; 0 -> 2 -> 3 -> 4 -> 5 -> 6 -> 3 -> 4 totals -1.94
    over 7 tokens (-0.28 per token). EOS is ~impossible everywhere else."""
    p = torch.zeros(VOCAB, VOCAB)

    def put(row, choices):
        others = [t for t in range(VOCAB) if t not in choices and t != EOS]
        eos = 0.0 if EOS in choices else 1e-4
        rest = 1.0 - sum(choices.values()) - eos
        for t in others:
            p[row, t] = rest / len(others)
        p[row, EOS] = eos
        for t, v in choices.items():
            p[row, t] = v

    put(0, {1: 0.46, 2: 0.44})
    put(1, {EOS: 0.70})
    for a, b in ((2, 3), (3, 4), (4, 5), (5, 6), (6, 3)):
        put(a, {b: 0.83})
    p[EOS] = 1.0 / VOCAB
    g = torch.Generator().manual_seed(4321)
    return p.log() + torch.rand(VOCAB, VOCAB, generator=g) * 1e-3


EOS_TABLE = _eos_table()


class _EosTableLM(_TableLM):
    def forward(self, token_id):
        return EOS_TABLE[_last(token_id)].unsqueeze(0)


def _eos_search(ns, bw, length_penalty):
    try:
        out = ns.beam_search(_EosTableLM(), 0, beam_width=bw, max_len=len(LONG),
                             eos_token=EOS, length_penalty=length_penalty)
    except TypeError as e:
        raise AssertionError(
            f"beam_search(..., eos_token=..., length_penalty=...) raised {e!r} — "
            f"the function must accept both keyword arguments")
    got = _as_ints(out, f"beam_width={bw}, eos_token={EOS}, length_penalty={length_penalty}")
    return SHORT if got == SHORT[:-1] else got   # keeping or stripping the EOS are both fine


def check_eos_finishes_a_hypothesis(ns):
    """Raw scoring (length_penalty=0): the short hypothesis that emitted EOS is best.
    It must be kept once finished, and never extended past EOS."""
    for bw in (2, 3):
        got = _eos_search(ns, bw, 0.0)
        assert EOS not in got[:-1], (
            f"beam_width={bw}: the returned sequence {got} continues after EOS={EOS} — "
            f"a beam that emits EOS is finished and must not be extended")
        assert got == SHORT, (
            f"beam_width={bw}, length_penalty=0: expected the finished hypothesis "
            f"{SHORT} (sum of log-probs -1.13), got {got}. "
            f"Set beams that emit EOS aside as finished hypotheses and let them compete "
            f"at the end with the beams still alive at max_len.")


def check_length_penalty_normalises_scores(ns):
    """length_penalty=1 ranks by average log-prob per token: the long hypothesis wins."""
    for bw in (2, 3):
        got = _eos_search(ns, bw, 1.0)
        assert got == LONG, (
            f"beam_width={bw}, length_penalty=1.0: expected {LONG} (-1.94 over 7 "
            f"tokens = -0.28 per token) to beat {SHORT} (-1.13 over 2 tokens = -0.57 "
            f"per token), got {got}. Rank final hypotheses by "
            f"score / length ** length_penalty.")


CHECKS = [
    check_shape_and_start_token,
    check_beam_width_one_is_greedy,
    check_wider_beam_finds_the_best_sequence,
    check_returns_the_top_scoring_beam,
    check_deterministic,
    check_eos_finishes_a_hypothesis,
    check_length_penalty_normalises_scores,
]
