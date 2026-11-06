"""save-model — state_dict save/load for inference, and a resumable training checkpoint.

The test model has a BatchNorm layer (buffers that are not parameters, so a
"save the named_parameters" shortcut loses them) and Dropout (so forgetting
eval() changes the predictions). Files are written to a temporary directory and
read back with torch.load(weights_only=True) — the format a state dict must have.
Resuming is checked by behaviour: continuing training from the restored model and
optimizer must give bit-identical parameters to continuing the original run, for
SGD with momentum and for Adam, whatever the fresh model's initialisation.
Loading a file that carries a pickled object whose unpickling runs code must not
run it (weights_only=True). map_location cannot be exercised without a GPU, so it
is not checked.
"""
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn

ENTRIES = ["save_weights", "load_for_inference", "save_checkpoint", "load_checkpoint"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "Everything here is torch.save / torch.load plus the state_dict methods. A "
    "state dict maps names to tensors — parameters and buffers such as BatchNorm's "
    "running statistics — so it loads without your class's code being pickled.",
    "load_for_inference builds a fresh model with model_class(), loads the file with "
    "torch.load(path, map_location='cpu', weights_only=True), calls load_state_dict "
    "and then eval() — Dropout and BatchNorm behave differently in training mode.",
    "A checkpoint is one dict with the keys 'epoch', 'model_state_dict' and "
    "'optimizer_state_dict'. load_checkpoint calls load_state_dict on the model and "
    "on the optimizer it was given (not a new one) and returns checkpoint['epoch'] "
    "— without the optimizer state, momentum and Adam's moments restart from zero.",
]


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.bn = nn.BatchNorm1d(8)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(self.drop(torch.relu(self.bn(self.fc1(x)))))


_X = torch.randn(64, 4, generator=torch.Generator().manual_seed(0))
_Y = torch.randn(64, 2, generator=torch.Generator().manual_seed(1))


def _trained(seed=0, steps=5, opt="sgd"):
    torch.manual_seed(seed)
    net = _Net()
    optim = (torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9) if opt == "sgd"
             else torch.optim.Adam(net.parameters(), lr=0.01))
    _train(net, optim, steps, seed=100)
    return net, optim


def _train(net, optim, steps, seed):
    net.train()
    torch.manual_seed(seed)                     # same dropout masks for both runs
    for i in range(steps):
        xb, yb = _X[8 * i % 64:][:8], _Y[8 * i % 64:][:8]
        optim.zero_grad()
        nn.functional.mse_loss(net(xb), yb).backward()
        optim.step()


_DIR = None


def _tmp(name):
    """A fresh path in one temporary directory that is removed at interpreter exit."""
    global _DIR
    if _DIR is None:
        _DIR = tempfile.TemporaryDirectory(prefix="torchleet-save-model-")
    d = Path(tempfile.mkdtemp(dir=_DIR.name))
    return str(d / name)


def _load(path, what):
    assert Path(path).exists(), f"{what} did not write a file at the path it was given"
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise AssertionError(
            f"torch.load(path, weights_only=True) cannot read the file {what} wrote "
            f"({type(e).__name__}) — save state dicts and plain dicts of tensors and "
            "numbers, not the module or optimizer object itself") from None


def _same_tensors(got, want, what):
    missing = [k for k in want if k not in got]
    extra = [k for k in got if k not in want]
    assert not missing, (
        f"{what} is missing {', '.join(missing[:4])}"
        + (" — buffers such as BatchNorm's running statistics are in "
           "model.state_dict() but not in named_parameters()"
           if any("running" in k or "num_batches" in k for k in missing) else ""))
    assert not extra, f"{what} has unexpected keys {', '.join(map(str, extra[:4]))}"
    for k, v in want.items():
        g = got[k]
        assert torch.is_tensor(g) and g.shape == v.shape and torch.equal(g, v), \
            f"{what}['{k}'] does not hold the model's current values"


def _call(what, fn, *args):
    """Run solver code; turn an exception into a one-line failure message."""
    try:
        return fn(*args)
    except Exception as e:
        first = re.sub(r"\x1b\[[0-9;]*m", "", (str(e).strip().splitlines() or [""])[0])[:200]
        raise AssertionError(f"{what} raised {type(e).__name__}: {first}") from None


# --------------------------------------------------------------------- checks
def check_save_weights_writes_a_state_dict(ns):
    net, _ = _trained()
    path = _tmp("model.pth")
    ns.save_weights(net, path)
    sd = _load(path, "save_weights")
    assert isinstance(sd, Mapping), (
        f"the saved file holds a {type(sd).__name__}, expected the model's state dict "
        "(a mapping from names to tensors) — save model.state_dict()")
    _same_tensors(sd, net.state_dict(), "the saved state dict")


def check_load_for_inference_restores_an_eval_model(ns):
    net, _ = _trained(seed=1)
    path = _tmp("model.pth")
    torch.save(net.state_dict(), path)
    torch.manual_seed(999)                       # the fresh model starts elsewhere
    model = _call("load_for_inference", ns.load_for_inference, _Net, path)
    assert isinstance(model, _Net), (
        f"load_for_inference should return a model built by model_class(), got "
        f"{type(model).__name__}")
    assert model is not net
    training = [n or "model" for n, m in model.named_modules() if m.training]
    assert not training, (
        f"the loaded model is still in training mode ({', '.join(training[:3])}) — "
        "call model.eval(): Dropout and BatchNorm behave differently at inference")
    net.eval()
    x = torch.randn(16, 4, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        assert torch.equal(model(x), net(x)), \
            "the loaded model's predictions differ from the original's (both in eval mode)"


def check_save_then_load_round_trip(ns):
    net, _ = _trained(seed=2)
    path = _tmp("model.pth")
    ns.save_weights(net, path)
    torch.manual_seed(123)
    model = _call("load_for_inference on a file written by save_weights",
                  ns.load_for_inference, _Net, path)
    net.eval()
    x = torch.randn(16, 4, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        assert torch.equal(model(x), net(x)), \
            "save_weights followed by load_for_inference does not reproduce the model"


_TRIPPED = []


def _trip():
    _TRIPPED.append(True)
    return "this ran while unpickling"


class _Payload:
    """Unpickling this calls _trip(): stands in for code hidden in a .pth file."""

    def __reduce__(self):
        return (_trip, ())


def check_loading_does_not_run_pickled_code(ns):
    net, _ = _trained(seed=3)
    path = _tmp("downloaded.pth")
    torch.save({**net.state_dict(), "payload": _Payload()}, path)
    _TRIPPED.clear()
    try:
        ns.load_for_inference(_Net, path)
    except Exception:
        pass                                     # refusing the file is the right outcome
    assert not _TRIPPED, (
        "load_for_inference unpickled an arbitrary Python object from the file, which "
        "can run any code (weights_only=False) — keep torch.load's weights_only=True "
        "for checkpoints you did not write yourself")


def check_checkpoint_file_layout(ns):
    net, opt = _trained(seed=4, opt="sgd")
    path = _tmp("checkpoint.pth")
    ns.save_checkpoint(net, opt, 7, path)
    ck = _load(path, "save_checkpoint")
    assert isinstance(ck, Mapping), f"the checkpoint should be a dict, got {type(ck).__name__}"
    keys = ("epoch", "model_state_dict", "optimizer_state_dict")
    missing = [k for k in keys if k not in ck]
    assert not missing, (
        f"the checkpoint dict is missing {missing} — use the keys 'epoch', "
        f"'model_state_dict' and 'optimizer_state_dict' (got {sorted(map(str, ck))})")
    assert ck["epoch"] == 7, f"checkpoint['epoch'] is {ck['epoch']!r}, expected the epoch passed in (7)"
    _same_tensors(ck["model_state_dict"], net.state_dict(), "checkpoint['model_state_dict']")
    osd = ck["optimizer_state_dict"]
    assert isinstance(osd, Mapping) and "state" in osd and "param_groups" in osd, \
        "checkpoint['optimizer_state_dict'] should be optimizer.state_dict()"
    want = opt.state_dict()["state"]
    assert len(osd["state"]) == len(want) and all(
        torch.equal(osd["state"][i]["momentum_buffer"], want[i]["momentum_buffer"])
        for i in want), "the optimizer state in the checkpoint lacks the momentum buffers"


def _resume(ns, opt_name, seed):
    a, opt_a = _trained(seed=seed, opt=opt_name)
    path = _tmp("checkpoint.pth")
    ns.save_checkpoint(a, opt_a, 5, path)

    torch.manual_seed(seed + 1000)               # a different starting point
    b = _Net()
    opt_b = (torch.optim.SGD(b.parameters(), lr=0.05, momentum=0.9) if opt_name == "sgd"
             else torch.optim.Adam(b.parameters(), lr=0.01))
    epoch = _call("load_checkpoint on a file written by save_checkpoint",
                  ns.load_checkpoint, b, opt_b, path)
    assert epoch == 5, (
        f"load_checkpoint returned {epoch!r}; it should return the epoch stored in the "
        "checkpoint (5)")
    held = {id(p) for g in opt_b.param_groups for p in g["params"]}
    assert all(id(p) in held for p in b.parameters()), (
        "after load_checkpoint the optimizer no longer holds the model's parameters — "
        "load_state_dict(..., assign=True) replaces the Parameter objects, so the "
        "optimizer would update tensors the model no longer uses; load without assign")
    for (name, pa), pb in zip(a.named_parameters(), b.parameters()):
        assert torch.equal(pa, pb), \
            f"after load_checkpoint the model's {name} differs from the checkpoint — load it into the model passed in"
    assert opt_b.state_dict()["state"], (
        f"the {opt_name.upper()} optimizer passed to load_checkpoint still has no state "
        "— call optimizer.load_state_dict(checkpoint['optimizer_state_dict']) on it")
    _train(a, opt_a, 4, seed=7)
    _train(b, opt_b, 4, seed=7)
    for (name, pa), pb in zip(a.named_parameters(), b.parameters()):
        assert torch.equal(pa, pb), (
            f"with {opt_name.upper()}, training resumed from the checkpoint diverges from "
            f"the original run ({name} differs after 4 more steps) — restore the "
            "optimizer's state as well as the weights")


def check_resume_matches_uninterrupted_training_sgd_momentum(ns):
    _resume(ns, "sgd", seed=5)


def check_resume_matches_uninterrupted_training_adam(ns):
    _resume(ns, "adam", seed=6)


CHECKS = [
    check_save_weights_writes_a_state_dict,
    check_load_for_inference_restores_an_eval_model,
    check_save_then_load_round_trip,
    check_loading_does_not_run_pickled_code,
    check_checkpoint_file_layout,
    check_resume_matches_uninterrupted_training_sgd_momentum,
    check_resume_matches_uninterrupted_training_adam,
]
