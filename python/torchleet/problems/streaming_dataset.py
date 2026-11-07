"""streaming-dataset — an IterableDataset that streams a CSV line by line.

The checks write their own CSV in which row i has features i + 0.25*j and target
-i, so every yielded sample says which row it came from. Nothing is compared
with a stored output; the properties are: every row comes back once, in file
order, as (x, y) tensors; a second epoch works; the shards that DataLoader
workers would read are disjoint and cover the file (the worker context is faked
through the hook get_worker_info() reads, so no processes are spawned); and the
first sample arrives without the file being read into memory, measured with
tracemalloc. Line-modulo sharding, contiguous chunks and byte-range sharding all
pass.
"""
import os
import tempfile
import tracemalloc
from contextlib import contextmanager

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.data._utils import worker as _worker
from torchleet.runner import Skip

ENTRIES = ["StreamingDataset"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "Open the file inside __iter__ and loop over it — a Python file object is a "
    "lazy line iterator. Parse each line into x (all but the last value) and a "
    "scalar y (the last one) and yield them.",
    "Every DataLoader worker runs its own copy of __iter__. "
    "torch.utils.data.get_worker_info() is None in the main process; inside a "
    "worker it has .id and .num_workers.",
    "Shard by line number: enumerate the lines and keep only those with "
    "line_idx % num_workers == worker_id, treating the main process as worker 0 "
    "of 1.",
]

N, D = 301, 4          # not divisible by 2, 3 or 4 workers


def _write_csv(path, n=N, d=D):
    with open(path, "w") as f:
        for i in range(n):
            f.write(",".join(f"{i + 0.25 * j:.2f}" for j in range(d)) + f",{-i}\n")
    return path


@contextmanager
def _csv(n=N, d=D):
    with tempfile.TemporaryDirectory() as tmp:
        yield _write_csv(os.path.join(tmp, "data.csv"), n, d)


@contextmanager
def _as_worker(worker_id, num_workers, dataset):
    """What a DataLoader worker sees: get_worker_info() reports worker k of n.

    This sets the private module global that get_worker_info() returns. If a
    future torch stores it elsewhere, the check is skipped rather than failing a
    correct solution."""
    if not hasattr(_worker, "_worker_info") or not hasattr(_worker, "WorkerInfo"):
        raise Skip("cannot fake a DataLoader worker with this torch version")
    saved = _worker._worker_info
    _worker._worker_info = _worker.WorkerInfo(
        id=worker_id, num_workers=num_workers, seed=0, dataset=dataset)
    try:
        info = get_worker_info()
        if info is None or info.id != worker_id or info.num_workers != num_workers:
            raise Skip("cannot fake a DataLoader worker with this torch version")
        yield
    finally:
        _worker._worker_info = saved


def _rows(dataset):
    """The row index of every sample, read back from its target."""
    return [int(round(-float(y))) for _, y in dataset]


def check_yields_every_row_in_order(ns):
    with _csv() as path:
        ds = ns.StreamingDataset(path)
        assert isinstance(ds, IterableDataset), (
            f"StreamingDataset is a {type(ds).__mro__[1].__name__}; it should "
            "subclass torch.utils.data.IterableDataset")
        items = list(ds)
    assert len(items) == N, f"a {N}-row file gave {len(items)} samples"
    want_x = torch.arange(N, dtype=torch.float32)[:, None] + 0.25 * torch.arange(D)
    for i, item in enumerate(items):
        assert isinstance(item, (tuple, list)) and len(item) == 2, \
            f"each sample should be an (x, y) pair, got {type(item).__name__}"
        x, y = item
        assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor), \
            "x and y should be tensors"
        assert tuple(x.shape) == (D,), \
            f"x should hold the {D} feature columns, got shape {tuple(x.shape)}"
        assert y.numel() == 1, \
            f"y should be the single last column, got shape {tuple(y.shape)}"
        assert torch.allclose(x.float(), want_x[i]) and abs(float(y) + i) < 1e-4, (
            f"sample {i} is not row {i} of the file (x={x.tolist()}, y={float(y)}) "
            "— rows should come back in file order, last column as the target")


def check_batches_with_dataloader(ns):
    with _csv() as path:
        loader = DataLoader(ns.StreamingDataset(path), batch_size=64)
        batches = list(loader)
    bx, by = batches[0]
    assert tuple(bx.shape) == (64, D) and tuple(by.shape) == (64,), (
        f"DataLoader(batch_size=64) gave batches X={tuple(bx.shape)}, "
        f"y={tuple(by.shape)}; expected (64, {D}) and (64,) — yield x as a 1-D "
        "tensor and y as a scalar tensor")
    total = sum(len(b[1]) for b in batches)
    assert total == N, f"one epoch through the DataLoader gave {total} samples, expected {N}"


def check_second_epoch_works(ns):
    with _csv() as path:
        ds = ns.StreamingDataset(path)
        first, second = _rows(ds), _rows(ds)
    assert first == second == list(range(N)), (
        f"two passes over the same dataset gave {len(first)} and {len(second)} "
        "samples — open the file inside __iter__, so each epoch gets a fresh handle")


def check_workers_get_disjoint_shards(ns):
    with _csv() as path:
        for n in (2, 3, 4):
            shards = []
            for k in range(n):
                ds = ns.StreamingDataset(path)
                with _as_worker(k, n, ds):
                    shards.append(_rows(ds))
            seen = sorted(r for s in shards for r in s)
            sizes = [len(s) for s in shards]
            assert len(seen) == N, (
                f"with {n} workers the shards hold {len(seen)} samples in total "
                f"(sizes {sizes}) for a {N}-row file — "
                + ("every worker re-reads the whole file; use get_worker_info() to "
                   "give each worker its own share" if len(seen) > N else
                   "some rows belong to no worker"))
            assert seen == list(range(N)), \
                f"with {n} workers some rows are read twice and others never"
            assert min(sizes) > 0, (
                f"with {n} workers the shard sizes are {sizes} — every worker "
                "should get part of the file")


def check_streams_without_loading_the_file(ns):
    """Pulling the first sample must not read the whole file into memory."""
    with _csv(n=40_000, d=8) as path:
        size = os.path.getsize(path)
        ds = ns.StreamingDataset(path)
        started = not tracemalloc.is_tracing()
        if started:
            tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            base, _ = tracemalloc.get_traced_memory()
            it = iter(ds)
            next(it)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            if started:
                tracemalloc.stop()
        del it
    used = peak - base
    assert used < size / 4, (
        f"producing the first sample allocated {used / 1e6:.1f} MB for a "
        f"{size / 1e6:.1f} MB file — read it line by line instead of loading it "
        "(readlines(), read() and pandas.read_csv all load everything)")


CHECKS = [
    check_yields_every_row_in_order,
    check_batches_with_dataloader,
    check_second_epoch_works,
    check_workers_get_disjoint_shards,
    check_streams_without_loading_the_file,
]
