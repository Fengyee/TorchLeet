"""preloaded-dataset — a map-style Dataset that holds all samples in memory as tensors.

Checked: the Dataset protocol; len() is the number of samples (not features);
item i is exactly (X[i], y[i]) as a tuple of tensors, for tensor and numpy
inputs; the data is converted to tensors once in __init__ (some tensor attribute
holds all N samples, and __getitem__ makes no torch.as_tensor / torch.tensor /
torch.from_numpy / torch.asarray call on numpy or list data — checked by
behaviour, so keeping a reference to the caller's arrays or an index list is
fine); X and y of different lengths are rejected; and a
DataLoader batches it in order and, shuffled, yields every sample exactly once
with X and y still paired. The data have unique rows, so pairing and coverage
are exact. How the tensors are stored (two attributes, one tuple as in
TensorDataset, or one stacked tensor) is not prescribed.
"""
import torch
from torch.utils.data import DataLoader, Dataset
from torchleet.runner import Skip

try:                                  # optional: torch itself does not need numpy
    import numpy as np
except ImportError:
    np = None

ENTRIES = ["PreloadedDataset"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "A map-style dataset needs exactly two methods besides __init__: __len__ "
    "(how many samples) and __getitem__(idx) (one sample). DataLoader does the "
    "batching and shuffling by calling them.",
    "Convert once, in __init__: torch.as_tensor accepts tensors, numpy arrays and "
    "lists. Check that X and y have the same number of samples there, and raise "
    "if they do not.",
    "__len__ is the number of rows of X (its first dimension, not the feature "
    "count); __getitem__ returns the tuple (self.X[idx], self.y[idx]) — no dict, "
    "no batching inside the dataset.",
]

N, D = 50, 3


def _data(seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(N, D, generator=g)
    X[:, 0] = torch.arange(N, dtype=torch.float32)        # unique, identifies each row
    y = torch.arange(N, dtype=torch.float32) * 10 + 0.5   # y[i] identifies sample i
    return X, y


def _item(ds, i, where):
    item = ds[i]
    assert isinstance(item, tuple) and len(item) == 2, (
        f"{where}: ds[{i}] should be the tuple (X[{i}], y[{i}]), got "
        f"{type(item).__name__}" + (f" of length {len(item)}" if hasattr(item, "__len__") else ""))
    for name, t in zip(("x", "y"), item):
        assert torch.is_tensor(t), (
            f"{where}: the {name} half of ds[{i}] is a {type(t).__name__}, expected a "
            "torch.Tensor — convert the data to tensors in __init__")
    return item


def check_is_a_map_style_dataset(ns):
    cls = ns.PreloadedDataset
    assert isinstance(cls, type), f"PreloadedDataset should be a class, got {type(cls).__name__}"
    assert issubclass(cls, Dataset), "PreloadedDataset must subclass torch.utils.data.Dataset"
    for method in ("__len__", "__getitem__"):
        assert getattr(cls, method, None) is not getattr(Dataset, method, None), \
            f"PreloadedDataset does not implement {method}"


def check_length_is_number_of_samples(ns):
    X, y = _data()
    n = len(ns.PreloadedDataset(X, y))
    assert n == N, (
        f"{N} samples with {D} features each, but len(dataset) is {n}"
        + (" — that is the feature count; use the first dimension" if n == D else ""))


def check_items_are_the_original_pairs(ns):
    X, y = _data()
    ds = ns.PreloadedDataset(X, y)
    for i in (0, 7, N - 1):
        xi, yi = _item(ds, i, "tensor input")
        assert tuple(xi.shape) == (D,), f"ds[{i}][0] has shape {tuple(xi.shape)}, expected ({D},)"
        assert tuple(yi.shape) == (), f"ds[{i}][1] has shape {tuple(yi.shape)}, expected a scalar ()"
        assert torch.equal(xi, X[i]) and torch.equal(yi, y[i]), \
            f"ds[{i}] is not (X[{i}], y[{i}])"


def check_numpy_input_is_stored_as_tensors(ns):
    if np is None:
        raise Skip("needs numpy")
    X, y = _data(1)
    ds = ns.PreloadedDataset(X.numpy(), y.numpy())
    xi, yi = _item(ds, 3, "numpy input")
    assert torch.equal(xi, X[3]) and torch.equal(yi, y[3]), \
        "with numpy input, ds[3] does not hold the values of X[3], y[3]"


def _attrs(obj, depth=2):
    """Values reachable from the instance's attributes (through tuples/lists/dicts)."""
    out, todo = [], [(v, depth) for v in vars(obj).values()]
    while todo:
        v, d = todo.pop()
        out.append(v)
        if d and isinstance(v, (tuple, list)) and len(v) <= 8:
            todo += [(u, d - 1) for u in v]
        elif d and isinstance(v, dict):
            todo += [(u, d - 1) for u in v.values()]
    return out


def check_data_is_preloaded_as_tensors(ns):
    """Convert once in __init__, not on every __getitem__ call."""
    X, y = _data(2)
    ds = (ns.PreloadedDataset(X.numpy(), y.numpy()) if np is not None
          else ns.PreloadedDataset(X, y))
    held = [v for v in _attrs(ds) if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == N]
    assert held, (
        "no attribute of the dataset holds the samples as a tensor — convert X and y "
        "with torch.as_tensor (or torch.tensor) in __init__ and store the tensors")
    if np is None:
        return
    # Behaviour, not attributes: watch the conversion functions while items are read.
    names = ("as_tensor", "tensor", "from_numpy", "asarray")
    originals = {name: getattr(torch, name) for name in names if hasattr(torch, name)}
    converted = []

    def watch(name, fn):
        def wrapper(*args, **kwargs):
            data = args[0] if args else next(iter(kwargs.values()), None)
            if isinstance(data, (np.ndarray, np.generic, list, tuple)):
                converted.append(name)
            return fn(*args, **kwargs)
        return wrapper

    try:
        for name, fn in originals.items():
            setattr(torch, name, watch(name, fn))
        for i in (0, 3, N - 1):
            ds[i]
    finally:
        for name, fn in originals.items():
            setattr(torch, name, fn)
    assert not converted, (
        f"__getitem__ calls torch.{converted[0]} on numpy/list data for every sample — "
        "convert X and y to tensors once in __init__, so __getitem__ only indexes")


def check_mismatched_lengths_are_rejected(ns):
    X, y = _data(3)
    try:
        ns.PreloadedDataset(X, y[:-5])
    except Exception:
        return
    raise AssertionError(
        f"PreloadedDataset accepted X with {N} rows and y with {N - 5} — raise an error "
        "in __init__ when X and y have different numbers of samples")


def _pairs(batches):
    for b in batches:
        assert isinstance(b, (tuple, list)) and len(b) == 2 and all(map(torch.is_tensor, b)), (
            f"a DataLoader batch is a {type(b).__name__}, expected (X_batch, y_batch) — "
            "__getitem__ should return the tuple (X[idx], y[idx])")
    return batches


def check_dataloader_batches_in_order(ns):
    X, y = _data(4)
    batches = _pairs(list(DataLoader(ns.PreloadedDataset(X, y), batch_size=16, shuffle=False)))
    assert len(batches) == 4, \
        f"batch_size=16 over {N} samples should give 4 batches (16+16+16+2), got {len(batches)}"
    bx, by = batches[0]
    assert tuple(bx.shape) == (16, D) and tuple(by.shape) == (16,), (
        f"the first batch has shapes {tuple(bx.shape)} and {tuple(by.shape)}, expected "
        f"(16, {D}) and (16,)")
    assert torch.equal(bx, X[:16]) and torch.equal(by, y[:16]), \
        "without shuffling, the first batch should be X[:16], y[:16]"


def check_shuffled_epoch_sees_every_pair_once(ns):
    X, y = _data(5)
    loader = DataLoader(ns.PreloadedDataset(X, y), batch_size=16, shuffle=True,
                        generator=torch.Generator().manual_seed(0))
    xs, ys = zip(*_pairs(list(loader)))
    xs, ys = torch.cat(xs), torch.cat(ys)
    assert len(ys) == N, f"one shuffled epoch yielded {len(ys)} samples, expected {N}"
    idx = xs[:, 0].round().long()
    assert sorted(idx.tolist()) == list(range(N)), \
        "one shuffled epoch did not yield every sample exactly once"
    assert torch.equal(ys, y[idx]) and torch.equal(xs, X[idx]), \
        "after shuffling, X and y rows are no longer paired"
    assert not torch.equal(idx, torch.arange(N)), "shuffle=True left the samples in order"


CHECKS = [
    check_is_a_map_style_dataset,
    check_length_is_number_of_samples,
    check_items_are_the_original_pairs,
    check_numpy_input_is_stored_as_tensors,
    check_data_is_preloaded_as_tensors,
    check_mismatched_lengths_are_rejected,
    check_dataloader_batches_in_order,
    check_shuffled_epoch_sees_every_pair_once,
]
