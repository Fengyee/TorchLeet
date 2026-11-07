"""data-parallel-loading — a DistributedSampler written from scratch.

Every rank's sampler is built in one process, so the cross-rank contract can be
checked directly: equal length ceil(N / R) on every rank, shards that together
cover the dataset with only the R * ceil(N / R) - N padding indices repeated
(also when shuffling — which fails a sampler seeded per rank — and when there
are more ranks than samples), reproducibility for the same seed + epoch, and a
new order after set_epoch. torch's DistributedSampler is not used as an oracle
for the exact order: strided or contiguous shards and any seeded RNG all pass.
"""
import math
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

ENTRIES = ["SimpleDistributedSampler"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "Every rank gets num_samples = ceil(len(dataset) / num_replicas) indices, so "
    "the padded index list has total_size = num_samples * num_replicas entries.",
    "In __iter__, build one index list that is identical on every rank: "
    "torch.randperm(n, generator=g) with g.manual_seed(seed + epoch) when "
    "shuffling, else range(n) — never seed with the rank.",
    "Pad by appending indices from the start (repeat the list if padding > n), "
    "then take indices[rank::num_replicas] as this rank's shard.",
]


class _Range(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


def _sampler(ns, n, reps, rank, shuffle, seed=0, epoch=None):
    s = ns.SimpleDistributedSampler(_Range(n), num_replicas=reps, rank=rank,
                                    shuffle=shuffle, seed=seed)
    if epoch is not None:
        s.set_epoch(epoch)
    return s


def _shard(s):
    return [int(i) for i in s]


def _check_cover(ns, n, reps, shuffle, seed=0, epoch=None):
    per = math.ceil(n / reps)
    shards = [_shard(_sampler(ns, n, reps, r, shuffle, seed, epoch)) for r in range(reps)]
    where = f"{n} samples on {reps} ranks (shuffle={shuffle})"
    for r, sh in enumerate(shards):
        assert len(sh) == per, f"{where}: rank {r} got {len(sh)} indices, expected ceil({n}/{reps}) = {per}"
        bad = [i for i in sh if not 0 <= i < n]
        assert not bad, f"{where}: rank {r} yielded out-of-range indices {bad[:5]}"
    flat = [i for sh in shards for i in sh]
    missing = sorted(set(range(n)) - set(flat))
    assert not missing, (
        f"{where}: no rank got indices {missing[:6]}"
        + (" — every rank must shuffle with the same permutation (seed + epoch, "
           "not the rank), or the shards overlap" if shuffle else ""))
    repeats = sum(c - 1 for c in Counter(flat).values())
    assert repeats == per * reps - n, (
        f"{where}: {repeats} indices are handed out twice, but only the "
        f"{per * reps - n} padding indices should be")


def check_every_rank_gets_the_same_count(ns):
    for n, reps in ((103, 4), (100, 4), (10, 3), (7, 1)):
        for r in range(reps):
            s = _sampler(ns, n, reps, r, shuffle=False)
            want = math.ceil(n / reps)
            assert len(s) == want, \
                f"len() on rank {r} of {reps} for {n} samples is {len(s)}, expected {want}"
            assert len(_shard(s)) == want, (
                f"rank {r} of {reps} yielded {len(_shard(s))} indices for {n} "
                f"samples but len() says {want}")


def check_shards_cover_the_dataset(ns):
    for n, reps in ((103, 4), (100, 4), (10, 3), (7, 1), (5, 5)):
        _check_cover(ns, n, reps, shuffle=False)


def check_shuffled_shards_cover_the_dataset(ns):
    for n, reps in ((103, 4), (64, 8)):
        for seed, epoch in ((0, None), (5, 2)):
            _check_cover(ns, n, reps, shuffle=True, seed=seed, epoch=epoch)


def check_more_ranks_than_samples(ns):
    _check_cover(ns, 3, 8, shuffle=False)
    _check_cover(ns, 3, 8, shuffle=True, seed=1, epoch=1)


def check_shuffle_is_reproducible(ns):
    for r in (0, 2):
        a = _shard(_sampler(ns, 103, 4, r, True, seed=42, epoch=3))
        b = _shard(_sampler(ns, 103, 4, r, True, seed=42, epoch=3))
        assert a == b, "two samplers with the same seed and epoch gave different orders"
    s = _sampler(ns, 103, 4, 0, True, seed=42)
    assert _shard(s) == _shard(s), \
        "iterating the same sampler twice in one epoch gave different orders"


def check_set_epoch_reshuffles(ns):
    s = _sampler(ns, 103, 4, 0, True, seed=42)
    orders = []
    for epoch in (0, 1, 2):
        s.set_epoch(epoch)
        orders.append(_shard(s))
    assert orders[0] != orders[1] and orders[1] != orders[2], (
        "set_epoch() did not change the order — the permutation should be "
        "generated in __iter__ from seed + epoch")
    a = _shard(_sampler(ns, 103, 4, 0, True, seed=1))
    b = _shard(_sampler(ns, 103, 4, 0, True, seed=2))
    assert a != b, "different seeds gave the same order"


def check_shuffle_actually_shuffles(ns):
    shards = [_shard(_sampler(ns, 200, 2, r, True, seed=0)) for r in range(2)]
    plain = [_shard(_sampler(ns, 200, 2, r, False)) for r in range(2)]
    assert shards != plain, \
        "shuffle=True produced the same shards as shuffle=False"


def check_works_as_a_dataloader_sampler(ns):
    s = _sampler(ns, 103, 4, 1, shuffle=True, seed=0)
    assert isinstance(s, Sampler) or hasattr(s, "__iter__"), "not usable as a sampler"
    loader = DataLoader(_Range(103), batch_size=8, sampler=s)
    got = torch.cat(list(loader)).tolist()
    assert got == _shard(s), \
        "a DataLoader using this sampler did not return the sampler's indices"


CHECKS = [
    check_every_rank_gets_the_same_count,
    check_shards_cover_the_dataset,
    check_shuffled_shards_cover_the_dataset,
    check_more_ranks_than_samples,
    check_shuffle_is_reproducible,
    check_set_epoch_reshuffles,
    check_shuffle_actually_shuffles,
    check_works_as_a_dataloader_sampler,
]
