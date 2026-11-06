"""debug-training — find and fix the silent bugs in a small image-classification
pipeline (HWC->CHW conversion, normalisation stats, model head, training loop,
evaluation loop).

Each check exercises ONE function's behaviour in isolation, on inputs built here, so
a remaining bug produces one pointed failure instead of a vague "accuracy is low".
The last check runs the solver's whole pipeline end to end on the same synthetic
shapes dataset as the notebook.
"""
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ENTRIES = ["to_tensor", "compute_normalization_stats", "normalize", "make_model",
           "train_one_epoch", "evaluate"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Debug from symptoms. Look at an image after every transform, print the per-channel "
    "mean/std of the normalised training set, and evaluate the same model twice. For each "
    "line of the training loop, ask which state it depends on and who resets that state.",
    "Data side: reshape() never moves axes (only permute/transpose do), and that includes "
    "a reshape you use to group values by channel. Model side: what does "
    "nn.CrossEntropyLoss apply to its input itself? Loops: what puts a model into "
    "train/eval mode, what clears .grad, and what keeps an autograd graph alive?",
    "There is one bug each in to_tensor, compute_normalization_stats and make_model, none "
    "in normalize, and three each in train_one_epoch (mode, gradient reset, loss "
    "bookkeeping) and evaluate (mode, autograd, how the predicted class is extracted).",
]

NUM_CLASSES = 4


def make_shapes_dataset(n, seed, size=16):
    """Identical to the notebook's generator: uint8 (n, size, size, 3) HWC images, labels (n,)."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, NUM_CLASSES, (n,), generator=g)
    background = torch.tensor([0.15, 0.45, 0.70])
    imgs = background + 0.08 * torch.randn(n, size, size, 3, generator=g)
    yy = torch.arange(size).view(1, size, 1)
    xx = torch.arange(size).view(1, 1, size)
    r = torch.randint(4, size - 4, (n, 1, 1), generator=g)
    c = torch.randint(4, size - 4, (n, 1, 1), generator=g)
    dy, dx = yy - r, xx - c
    shapes = torch.stack([
        (dy.abs() <= 1) & (dx.abs() <= 4),
        (dx.abs() <= 1) & (dy.abs() <= 4),
        (dy.abs() <= 3) & (dx.abs() <= 3) & ~((dy.abs() <= 1) & (dx.abs() <= 1)),
        ((dy - dx).abs() <= 1) & (dy.abs() <= 4),
    ], dim=1)
    mask = shapes[torch.arange(n), labels]
    color = (1 - background) + 0.15 * torch.randn(n, 1, 1, 3, generator=g)
    imgs = torch.where(mask[..., None], color, imgs)
    return (imgs.clamp(0, 1) * 255).round().to(torch.uint8), labels


# ---------------------------------------------------------------- data

def check_to_tensor_layout_and_scale(ns):
    g = torch.Generator().manual_seed(0)
    raw = torch.randint(0, 256, (3, 5, 7, 3), generator=g, dtype=torch.uint8)   # N, H, W, C
    before = raw.clone()
    out = ns.to_tensor(raw)
    assert torch.equal(raw, before), "to_tensor must not modify its input"
    assert tuple(out.shape) == (3, 3, 5, 7), \
        f"(N, H, W, C) = (3, 5, 7, 3) should become (N, C, H, W) = (3, 3, 5, 7), got {tuple(out.shape)}"
    assert out.dtype == torch.float32, f"expected float32, got {out.dtype}"
    want = raw.permute(0, 3, 1, 2).float() / 255
    if not torch.allclose(out, want, atol=1e-6):
        bad = (out - want).abs().flatten().argmax().item()
        n, c, h, w = (int(i) for i in torch.unravel_index(torch.tensor(bad), out.shape))
        raise AssertionError(
            f"pixels land in the wrong place: out[{n}, {c}, {h}, {w}] = {out[n, c, h, w].item():.4f} "
            f"but images[{n}, {h}, {w}, {c}] / 255 = {want[n, c, h, w].item():.4f}. "
            "The shape is right but the values are scrambled")


def check_normalization_stats_are_per_channel(ns):
    g = torch.Generator().manual_seed(1)
    means = torch.tensor([0.1, 0.5, 0.9])
    stds = torch.tensor([0.05, 0.2, 0.01])
    imgs = means.view(1, 3, 1, 1) + stds.view(1, 3, 1, 1) * torch.randn(10, 3, 4, 5, generator=g)
    out = ns.compute_normalization_stats(imgs)
    assert isinstance(out, (tuple, list)) and len(out) == 2, "return (mean, std)"
    mean, std = (torch.as_tensor(t).float() for t in out)
    assert tuple(mean.shape) == (3,) and tuple(std.shape) == (3,), \
        f"expected per-channel stats of shape (C,) = (3,), got {tuple(mean.shape)} and {tuple(std.shape)}"
    want_m, want_s = imgs.mean(dim=(0, 2, 3)), imgs.std(dim=(0, 2, 3))
    assert torch.allclose(mean, want_m, atol=1e-5), (
        f"channel means should be {[round(v, 4) for v in want_m.tolist()]} (each channel "
        f"averaged over N, H and W), got {[round(v, 4) for v in mean.tolist()]}")
    assert torch.allclose(std, want_s, rtol=5e-3), (
        f"channel stds should be {[round(v, 4) for v in want_s.tolist()]}, "
        f"got {[round(v, 4) for v in std.tolist()]}")


def check_normalize_standardises_each_channel(ns):
    g = torch.Generator().manual_seed(2)
    imgs = torch.rand(6, 3, 4, 4, generator=g)
    mean = torch.tensor([0.2, 0.5, 0.7])
    std = torch.tensor([0.1, 0.3, 0.05])
    out = ns.normalize(imgs, mean, std)
    want = (imgs - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
    assert tuple(out.shape) == tuple(imgs.shape), f"shape changed to {tuple(out.shape)}"
    assert torch.allclose(out, want, atol=1e-5), "normalize must compute (x - mean[c]) / std[c] per channel"


# ---------------------------------------------------------------- model

def check_model_outputs_raw_logits(ns):
    torch.manual_seed(0)
    model = ns.make_model(NUM_CLASSES)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(6, 3, 16, 16) * 2)
    assert tuple(out.shape) == (6, NUM_CLASSES), \
        f"make_model({NUM_CLASSES}) should map (6, 3, 16, 16) to (6, {NUM_CLASSES}), got {tuple(out.shape)}"
    in_unit = bool(((out >= 0) & (out <= 1)).all())
    sums_to_one = torch.allclose(out.sum(dim=1), torch.ones(6), atol=1e-4)
    assert not (in_unit and sums_to_one), (
        "the model outputs probabilities (every row is in [0, 1] and sums to 1), but "
        "nn.CrossEntropyLoss expects raw logits: it applies log-softmax itself")
    assert not in_unit, "every output is in [0, 1]: return unbounded logits, not probabilities"
    torch.manual_seed(0)
    out7 = ns.make_model(7).eval()(torch.randn(2, 3, 16, 16))
    assert tuple(out7.shape) == (2, 7), "make_model(num_classes) must honour num_classes"


# ---------------------------------------------------------------- training loop

def _tiny_problem(seed=3, n=12, batch=4):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 4, 4, generator=g)
    y = torch.randint(0, NUM_CLASSES, (n,), generator=g)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=False)
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Flatten(), nn.Linear(48, NUM_CLASSES))   # no dropout / BN
    return model, loader


def check_train_one_epoch_matches_clean_sgd_steps(ns):
    import copy
    model, loader = _tiny_problem()
    ref = copy.deepcopy(model)
    loss_fn = nn.CrossEntropyLoss()
    ns.train_one_epoch(model, loader, torch.optim.SGD(model.parameters(), lr=0.5), loss_fn)
    ref_opt = torch.optim.SGD(ref.parameters(), lr=0.5)
    for x, y in loader:
        ref_opt.zero_grad()
        loss_fn(ref(x), y).backward()
        ref_opt.step()
    for (n, p), q in zip(model.named_parameters(), ref.parameters()):
        d = (p - q).abs().max().item()
        assert torch.allclose(p, q, atol=1e-5), (
            f"after one epoch (3 batches) parameter {n} differs from 3 clean SGD steps by {d:.3g}: "
            "gradients from earlier batches leak into later updates")


def check_train_one_epoch_returns_detached_mean_loss(ns):
    model, loader = _tiny_problem()
    loss_fn = nn.CrossEntropyLoss()
    # lr = 0: parameters never move, so the expected per-batch losses are known exactly
    ret = ns.train_one_epoch(model, loader, torch.optim.SGD(model.parameters(), lr=0.0), loss_fn)
    if isinstance(ret, torch.Tensor):
        assert ret.grad_fn is None and not ret.requires_grad, (
            f"train_one_epoch returned a tensor attached to the autograd graph "
            f"(grad_fn={type(ret.grad_fn).__name__}). Summing `loss` tensors keeps every "
            "batch's graph alive, so memory grows with every step. Return a Python float")
    with torch.no_grad():
        want = sum(loss_fn(model(x), y).item() for x, y in loader) / len(loader)
    assert ret is not None and math.isclose(float(ret), want, rel_tol=1e-5), \
        f"expected the mean of the per-batch losses ({want:.5f}), got {ret}"


def check_train_one_epoch_uses_train_mode(ns):
    model, loader = _tiny_problem()
    model.eval()                                        # e.g. left in eval mode by evaluate()
    modes = []
    h = model.register_forward_hook(lambda m, i, o: modes.append(m.training))
    try:
        ns.train_one_epoch(model, loader, torch.optim.SGD(model.parameters(), lr=0.1),
                           nn.CrossEntropyLoss())
    finally:
        h.remove()
    assert modes, "train_one_epoch never called the model"
    assert all(modes), (
        "train_one_epoch ran forward passes in eval mode (the model arrived in eval mode, e.g. "
        "after evaluate()). Dropout is then off and BatchNorm neither uses batch statistics "
        "nor updates its running stats")


# ---------------------------------------------------------------- evaluation loop

def _eval_problem():
    torch.manual_seed(4)
    model = nn.Sequential(nn.Flatten(), nn.Linear(48, NUM_CLASSES))
    g = torch.Generator().manual_seed(5)
    x = torch.randn(50, 3, 4, 4, generator=g)
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
    y = pred.clone()
    wrong = list(range(6)) + [48, 49]                   # 42/50 right; the 2-sample last batch is all wrong
    y[wrong] = (pred[wrong] + 1) % NUM_CLASSES
    return model, DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False), 42 / 50


def check_evaluate_accuracy_is_correct(ns):
    model, loader, want = _eval_problem()
    got = ns.evaluate(model, loader)
    assert got is not None, "evaluate must return the accuracy"
    got = float(got)
    assert math.isclose(got, want, abs_tol=1e-6), (
        f"evaluate returned {got:.4f}, but the model's predicted classes (argmax over the class "
        f"dimension) match {want:.0%} of the labels"
        + (" — accuracy is the fraction of ALL samples, not the mean of per-batch accuracies"
           if math.isclose(got, (10 / 16 + 1 + 1 + 0) / 4, abs_tol=1e-6) else "")
        + (" — compare predicted class INDICES with the labels" if got < 0.1 else ""))


def check_evaluate_uses_eval_mode(ns):
    torch.manual_seed(6)
    model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
                          nn.Flatten(), nn.Dropout(0.5), nn.Linear(8 * 16, NUM_CLASSES))
    g = torch.Generator().manual_seed(7)
    x = torch.randn(64, 3, 4, 4, generator=g) * 2 + 1
    y = torch.randint(0, NUM_CLASSES, (64,), generator=g)
    loader = DataLoader(TensorDataset(x, y), batch_size=16)
    model.train()                                        # as left by train_one_epoch
    stats = [b.clone() for b in model.buffers()]
    modes = []
    h = model.register_forward_hook(lambda m, i, o: modes.append(m.training))
    try:
        a = float(ns.evaluate(model, loader))
        b = float(ns.evaluate(model, loader))
    finally:
        h.remove()
    assert modes and not any(modes), (
        "evaluate ran forward passes in training mode: dropout is active (the accuracy changes "
        "from call to call) and BatchNorm normalises with test-batch statistics")
    assert all(torch.equal(s, t) for s, t in zip(stats, model.buffers())), \
        "evaluate changed the model's BatchNorm running statistics (test data leaked into the model)"
    assert a == b, f"evaluating the same model twice gave {a:.4f} and {b:.4f}"


def check_evaluate_disables_autograd(ns):
    model, loader, _ = _eval_problem()
    grad_on = []
    h = model.register_forward_hook(lambda m, i, o: grad_on.append(torch.is_grad_enabled()))
    try:
        ns.evaluate(model, loader)
    finally:
        h.remove()
    assert grad_on and not any(grad_on), (
        "evaluate runs the model with autograd enabled: every batch records a graph and keeps "
        "its activations for a backward pass that never happens (wasted memory and time)")


# ---------------------------------------------------------------- end to end

def check_pipeline_end_to_end(ns):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)            # tiny convs: 1 thread is fastest and immune to CPU contention
    try:
        _end_to_end(ns)
    finally:
        torch.set_num_threads(threads)


def _end_to_end(ns):
    torch.manual_seed(0)
    train_raw, train_y = make_shapes_dataset(1024, seed=0)
    test_raw, test_y = make_shapes_dataset(512, seed=1)
    train_x, test_x = ns.to_tensor(train_raw), ns.to_tensor(test_raw)
    mean, std = ns.compute_normalization_stats(train_x)
    train_x, test_x = ns.normalize(train_x, mean, std), ns.normalize(test_x, mean, std)
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=256)
    model = ns.make_model(NUM_CLASSES)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for _ in range(4):
        losses.append(float(ns.train_one_epoch(model, train_loader, opt, loss_fn)))
        acc = float(ns.evaluate(model, test_loader))
    assert acc >= 0.9, (
        f"the full pipeline reaches only {acc:.1%} test accuracy after 4 epochs (train losses "
        f"{[round(l, 3) for l in losses]}); a healthy pipeline gets > 95% on this task")


CHECKS = [
    check_to_tensor_layout_and_scale,
    check_normalization_stats_are_per_channel,
    check_normalize_standardises_each_channel,
    check_model_outputs_raw_logits,
    check_train_one_epoch_matches_clean_sgd_steps,
    check_train_one_epoch_returns_detached_mean_loss,
    check_train_one_epoch_uses_train_mode,
    check_evaluate_accuracy_is_correct,
    check_evaluate_uses_eval_mode,
    check_evaluate_disables_autograd,
    check_pipeline_end_to_end,
]
