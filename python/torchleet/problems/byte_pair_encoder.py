"""byte-pair-encoder — the BPE merge loop over a character-level vocabulary.

Everything here is an invariant of BPE itself: merging never changes the text it
spells, every merge removes symbols, and a fixed corpus gives a fixed answer. The
solver's own four functions are fed to each other, so any end-of-word marker,
container type or tie-break rule is fine as long as they agree.
"""
ENTRIES = ["get_vocab", "get_stats", "merge_vocab", "byte_pair_encoding"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "get_vocab splits each word into characters plus an end-of-word marker, and counts it.",
    "get_stats counts adjacent symbol pairs weighted by each word's frequency.",
    "A merge replaces every adjacent (a, b) with the single symbol 'ab' — the spelling "
    "of each word must not change, only how it is chopped up.",
]

# "low" is repeated on purpose: a get_stats that counts each distinct word once
# instead of weighting by its frequency scores the same on a corpus of unique
# words, and would sail through every check below.
CORPUS = ["low", "low", "low", "low", "lowest", "newer", "wider", "newest"]

# Built so that a merge can go wrong in the two classic ways: "abab" holds the same
# pair twice (a merge must replace every occurrence, not the first), and "st"/"es"
# share an 's', so after one of them is merged, "best" has adjacent symbols like
# ('e', 'st') that a string-replace on "e s t" would wrongly glue together.
REPLAY_CORPUS = ["st"] * 5 + ["es"] * 4 + ["best", "abab", "abab", "nest"]
REPLAY_MERGES = 8


def _items(vocab, who):
    assert hasattr(vocab, "items"), \
        f"{who} should return a dict-like mapping of word -> frequency, got {type(vocab).__name__}"
    return list(vocab.items())


def _symbols(vocab):
    """Total symbol count, weighted by word frequency. Every merge shrinks this."""
    return sum(len(word) * freq for word, freq in vocab.items())


def _spelling(vocab):
    """The text each entry spells, with its frequency. A merge must preserve this."""
    return sorted(("".join(word), freq) for word, freq in vocab.items())


def _get(stats, pair):
    return dict(stats).get(pair, 0)


def _split(result, n_merges):
    assert isinstance(result, tuple) and len(result) == 2, (
        "byte_pair_encoding(corpus, num_merges) should return (vocab, merges), "
        f"got {type(result).__name__}")
    vocab, merges = result
    _items(vocab, "byte_pair_encoding's first return value")
    assert len(merges) <= n_merges, \
        f"asked for at most {n_merges} merges, got {len(merges)} back"
    return vocab, merges


def check_get_vocab_splits_words_into_symbols(ns):
    corpus = ["low", "newest", "wider", "low"]
    vocab = ns.get_vocab(corpus)
    items = _items(vocab, "get_vocab")

    total = sum(f for _, f in items)
    assert total == len(corpus), \
        f"frequencies should sum to {len(corpus)} (one per word in the corpus), got {total}"

    for word, want in [("low", 2), ("newest", 1), ("wider", 1)]:
        hits = [(k, f) for k, f in items if "".join(k).startswith(word)]
        assert len(hits) == 1, \
            f"expected exactly one vocabulary entry spelling '{word}', found {len(hits)}"
        key, freq = hits[0]
        assert freq == want, \
            f"'{word}' appears {want}x in the corpus but has frequency {freq}"
        assert len(key) >= len(word), (
            f"'{word}' should start out split into at least {len(word)} symbols "
            f"(one per character, plus an end-of-word marker), got {len(key)}: {key}")


def check_get_stats_counts_adjacent_pairs(ns):
    vocab = ns.get_vocab(CORPUS)
    stats = ns.get_stats(vocab)
    pairs = dict(stats)
    assert pairs, "get_stats returned nothing for a non-empty vocabulary"

    # Every reported pair must really be adjacent somewhere, with the right count.
    for pair, count in pairs.items():
        assert isinstance(pair, tuple) and len(pair) == 2, \
            f"keys of get_stats should be (left, right) symbol pairs, got {pair!r}"
        actual = sum(freq for word, freq in vocab.items()
                     for i in range(len(word) - 1) if (word[i], word[i + 1]) == pair)
        assert count == actual, (
            f"pair {pair} occurs {actual} times across the vocabulary "
            f"(counting each word's frequency), but get_stats reported {count}")

    expected_total = sum((len(w) - 1) * f for w, f in vocab.items())
    assert sum(pairs.values()) == expected_total, (
        f"counts should total {expected_total} adjacent pairs, got {sum(pairs.values())} "
        "— is each word weighted by its frequency?")


def check_merge_vocab_applies_the_merge(ns):
    vocab = ns.get_vocab(CORPUS)
    stats = ns.get_stats(vocab)
    best = max(dict(stats), key=lambda p: dict(stats)[p])
    before = _get(stats, best)

    merged = ns.merge_vocab(best, vocab)
    _items(merged, "merge_vocab")

    after = _get(ns.get_stats(merged), best)
    assert after < before, (
        f"after merging {best} it should no longer be an adjacent pair, but "
        f"get_stats still counts it {after} times (was {before})")

    assert _symbols(merged) < _symbols(vocab), (
        f"merging should join symbols together: total symbol count stayed at "
        f"{_symbols(vocab)}")


def check_merge_vocab_preserves_the_words(ns):
    """A merge re-chunks the words; it must never change what they spell."""
    vocab = ns.get_vocab(CORPUS)
    stats = dict(ns.get_stats(vocab))
    best = max(stats, key=stats.get)
    merged = ns.merge_vocab(best, vocab)

    assert _spelling(merged) == _spelling(vocab), (
        f"merging {best} changed the corpus itself — joining each entry's symbols "
        "must give back the same words with the same frequencies. Expected\n  "
        f"{_spelling(vocab)}\ngot\n  {_spelling(merged)}")


def check_merge_picks_a_most_frequent_pair(ns):
    """BPE is greedy: the pair it merges first has to be one of the most frequent
    ones. Which of several tied pairs it picks is up to the solver."""
    vocab = ns.get_vocab(CORPUS)
    stats = dict(ns.get_stats(vocab))
    top = max(stats.values())

    _, merges = _split(ns.byte_pair_encoding(CORPUS, 1), 1)
    assert len(merges) == 1, f"one merge was requested, got {len(merges)}"
    chosen = tuple(merges[0])

    count = stats.get(chosen, None)
    assert count is not None, \
        f"merged {chosen}, which get_stats does not report as an adjacent pair at all"
    assert count == top, (
        f"merged {chosen}, which occurs {count} times, but the most frequent pair "
        f"occurs {top} times — BPE must merge the most frequent pair "
        "(counted with each word's frequency, not once per distinct word)")


def check_merges_reduce_symbols_monotonically(ns):
    counts = []
    for n in range(4):
        vocab, merges = _split(ns.byte_pair_encoding(CORPUS, n), n)
        assert len(merges) == n, \
            f"asked for {n} merges on a corpus that supports them, got {len(merges)}"
        counts.append(_symbols(vocab))

    for n in range(1, len(counts)):
        assert counts[n] < counts[n - 1], (
            f"{n} merges left {counts[n]} symbols but {n - 1} merges left "
            f"{counts[n - 1]} — each merge must strictly shrink the symbol count")


def check_bpe_preserves_the_corpus(ns):
    base = ns.get_vocab(CORPUS)
    vocab, _ = _split(ns.byte_pair_encoding(CORPUS, 6), 6)
    assert _spelling(vocab) == _spelling(base), (
        "after 6 merges the vocabulary no longer spells the original corpus — "
        "merges may only regroup symbols, never add or drop characters")


def check_bpe_is_deterministic(ns):
    a_vocab, a_merges = _split(ns.byte_pair_encoding(CORPUS, 5), 5)
    b_vocab, b_merges = _split(ns.byte_pair_encoding(CORPUS, 5), 5)
    assert list(a_merges) == list(b_merges), (
        f"the same corpus gave different merges on two runs:\n  {list(a_merges)}\n  "
        f"{list(b_merges)}\n— iterating a set, or a tie-break on unordered data, "
        "makes BPE non-reproducible")
    assert _spelling(a_vocab) == _spelling(b_vocab), \
        "the same corpus gave two different final vocabularies"


def _pair_counts(vocab):
    counts = {}
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            counts[(word[i], word[i + 1])] = counts.get((word[i], word[i + 1]), 0) + freq
    return counts


def _merge(pair, vocab):
    """Reference merge on symbol tuples: left to right, every occurrence."""
    a, b = pair
    out = {}
    for word, freq in vocab.items():
        merged, i = [], 0
        while i < len(word):
            if i + 1 < len(word) and word[i] == a and word[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(word[i])
                i += 1
        out[tuple(merged)] = out.get(tuple(merged), 0) + freq
    return out


def check_every_merge_is_greedy_and_exact(ns):
    """Replay the solver's merge list on its own starting vocabulary. Every step
    must merge a most frequent pair *of the vocabulary at that step*, and the final
    vocabulary must be exactly what applying those merges symbol-by-symbol gives.
    Catches merging only the first occurrence in a word, and string-replace merges
    that glue together symbols which merely spell the pair (e.g. 'e','st' when
    merging ('e', 's'))."""
    vocab = {tuple(w): f for w, f in ns.get_vocab(REPLAY_CORPUS).items()}
    final, merges = _split(ns.byte_pair_encoding(REPLAY_CORPUS, REPLAY_MERGES),
                           REPLAY_MERGES)
    for step, pair in enumerate(merges, 1):
        pair = tuple(pair)
        counts = _pair_counts(vocab)
        top = max(counts.values())
        assert counts.get(pair, 0) == top, (
            f"merge {step} was {pair}, which occurs {counts.get(pair, 0)} times at that "
            f"point, but the most frequent pair then occurs {top} times — recount the "
            "pairs on the current (already merged) vocabulary before every merge")
        want = _merge(pair, vocab)
        got = {tuple(w): f for w, f in ns.merge_vocab(pair, dict(vocab)).items()}
        assert got == want, (
            f"merge_vocab({pair}) on\n  {sorted(vocab.items())}\nshould give\n  "
            f"{sorted(want.items())}\ngot\n  {sorted(got.items())}\n— replace every "
            "adjacent occurrence of the two *symbols* (left to right), and never join "
            "symbols that only spell the pair as text (e.g. 'e','st' for ('e', 's'))")
        vocab = want
    got = {tuple(w): f for w, f in final.items()}
    assert got == vocab, (
        f"after the merges {list(merges)} the vocabulary should be\n  "
        f"{sorted(vocab.items())}\ngot\n  {sorted(got.items())}\n— a merge must "
        "replace every adjacent occurrence of the two *symbols*, and must not join "
        "symbols that only spell the pair as text")


CHECKS = [check_get_vocab_splits_words_into_symbols,
          check_get_stats_counts_adjacent_pairs,
          check_merge_vocab_applies_the_merge,
          check_merge_vocab_preserves_the_words,
          check_merge_picks_a_most_frequent_pair,
          check_merges_reduce_symbols_monotonically,
          check_bpe_preserves_the_corpus,
          check_bpe_is_deterministic,
          check_every_merge_is_greedy_and_exact]
