"""buffered-loading — a shuffle buffer over a streaming IterableDataset.

Checked properties, none of which depend on the RNG or the exact algorithm:
every element comes out exactly once (also for a buffer of 1 or one larger than
the stream); the first element is a uniform draw from the first buffer_size
stream elements (a buffer that always picks the same slot, or never picks the
last one, fails); no element comes out more than buffer_size positions before
its stream position; and on an infinite stream the buffer never reads more than
buffer_size elements ahead of what it has yielded — the O(buffer_size) memory
contract, which a "list() everything and shuffle" implementation breaks.
"""
from collections import Counter

import torch
from torch.utils.data import DataLoader, IterableDataset

ENTRIES = ["ShuffleBufferDataset"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "Pull buffer_size items from iter(base_dataset) into a list first; when the "
    "stream runs dry before that, you simply start with a smaller buffer.",
    "Then, for every further item from the stream: pick a uniformly random slot, "
    "yield what is in it, and put the new item in that slot.",
    "When the stream is exhausted, the buffer still holds up to buffer_size "
    "items: random.shuffle it and yield them all — otherwise they are lost.",
]


class _Stream(IterableDataset):
    """Yields (index, value) pairs 0..n-1 in order; n=None streams forever."""

    def __init__(self, n=None, limit=None):
        self.n, self.limit, self.pulled = n, limit, 0

    def __iter__(self):
        i = 0
        while self.n is None or i < self.n:
            self.pulled = i + 1
            if self.limit is not None and self.pulled > self.limit:
                raise AssertionError(
                    f"the shuffle buffer read {self.limit} items from an infinite "
                    "stream without yielding enough — it should hold at most "
                    "buffer_size items, not load the whole stream")
            yield torch.tensor(i), torch.tensor(float(i))
            i += 1


def _order(ns, n, buffer_size):
    out = []
    for item in ns.ShuffleBufferDataset(_Stream(n), buffer_size):
        assert isinstance(item, (tuple, list)) and len(item) == 2, (
            "the buffer should yield the base dataset's items unchanged — here "
            f"(index, value) pairs — got {type(item).__name__}")
        out.append(int(item[0]))
    return out


def check_every_element_exactly_once(ns):
    ds = ns.ShuffleBufferDataset(_Stream(10), 3)
    assert isinstance(ds, IterableDataset), \
        "ShuffleBufferDataset should subclass torch.utils.data.IterableDataset"
    for n, b in ((200, 16), (200, 1), (37, 37), (5, 100), (0, 4)):
        order = _order(ns, n, b)
        counts = Counter(order)
        dup = sorted(i for i, c in counts.items() if c > 1)
        missing = sorted(set(range(n)) - set(counts))
        assert not dup and not missing and len(order) == n, (
            f"{n} items through a buffer of {b}: {len(order)} came out"
            + (f", duplicated {dup[:5]}" if dup else "")
            + (f", missing {missing[:5]}" if missing else "")
            + " — every item must be yielded exactly once (do not forget to "
            "drain the buffer at the end)")


def check_works_with_dataloader(ns):
    loader = DataLoader(ns.ShuffleBufferDataset(_Stream(100), 20), batch_size=32)
    got = torch.cat([idx for idx, _ in loader])
    assert sorted(got.tolist()) == list(range(100)), \
        "batched through a DataLoader, the items are not each seen once"


def check_first_item_is_uniform_over_the_buffer(ns):
    b, trials = 8, 1600
    torch.manual_seed(0)
    firsts = Counter(_first(ns, b) for _ in range(trials))
    # An append-then-pop buffer briefly holds buffer_size + 1 items; accept that too.
    support = set(firsts)
    assert support in (set(range(b)), set(range(b + 1))), (
        f"over {trials} runs the first item was only ever one of {sorted(firsts)}; "
        f"with buffer_size={b} it should be a uniform draw from items 0..{b - 1} "
        "(pick the slot with random.randint(0, len(buffer) - 1))")
    k = len(support)
    lo, hi = min(firsts.values()), max(firsts.values())
    assert trials / (2 * k) < lo and hi < 2 * trials / k, (
        f"the first item is far from uniform over the buffer: {dict(sorted(firsts.items()))}")


def _first(ns, b):
    item = next(iter(ns.ShuffleBufferDataset(_Stream(40), b)))
    return int(item[0])


def check_items_never_jump_ahead_of_the_stream(ns):
    """Item i cannot come out before it has been read: position >= i - buffer_size."""
    for b in (1, 10, 40):
        order = _order(ns, 300, b)
        early = max(i - pos for pos, i in enumerate(order))
        assert early <= b, (
            f"with buffer_size={b}, item {order[[i - p for p, i in enumerate(order)].index(early)]} "
            f"came out {early} positions before its place in the stream — the "
            "buffer must only hold buffer_size items, so it cannot have read it yet")


def check_buffer_reads_lazily(ns):
    """On an infinite stream, k yields need at most buffer_size + k reads."""
    b = 20
    base = _Stream(None, limit=50 * b)
    it = iter(ns.ShuffleBufferDataset(base, b))
    for k in range(1, 3 * b):
        next(it)
        assert base.pulled <= b + k + 1, (
            f"after {k} items came out, {base.pulled} had been read from the stream "
            f"— with buffer_size={b} that should be at most {b + k + 1}")


def check_bigger_buffer_mixes_more(ns):
    def displacement(b):
        order = _order(ns, 400, b)
        return sum(abs(i - p) for p, i in enumerate(order)) / len(order)
    small, large = displacement(4), displacement(200)
    assert large > 3 * small, (
        f"average displacement is {small:.1f} with buffer_size=4 and {large:.1f} "
        "with buffer_size=200 — a larger buffer should shuffle much more")


CHECKS = [
    check_every_element_exactly_once,
    check_works_with_dataloader,
    check_first_item_is_uniform_over_the_buffer,
    check_items_never_jump_ahead_of_the_stream,
    check_buffer_reads_lazily,
    check_bigger_buffer_mixes_more,
]
