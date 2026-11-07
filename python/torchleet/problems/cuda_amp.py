"""cuda-amp — automatic mixed precision: autocast + GradScaler, checked on the CPU.

No GPU is needed. GradScaler's logic (scale the loss, unscale before clipping,
skip a step whose gradients hold inf/NaN, halve the scale after an overflow) is
the same on every device, and recent PyTorch runs float16 autocast and an enabled
GradScaler on the CPU too. So train_step is driven with CPU tensors through:

  * bfloat16 with a disabled scaler (the notebook's CPU path) and float16 with an
    enabled scaler (the CUDA path), several steps each, against a reference step
    on an identical copy: forward + loss under autocast, scaled backward,
    unscale_, clip_grad_norm_(max_norm=1.0), scaler.step, scaler.update.
    Common mistakes are named by matching the result against them (no clipping,
    clipping before unscale_, gradients never zeroed, no scaler at all);
  * a batch whose float16 gradients underflow unless the loss is scaled;
  * a batch that overflows float16: the step must be skipped and the scale halved;
  * forward-hook dtypes: the layers ran in the low-precision dtype while the
    parameters stayed float32.
amp_config is checked for "cpu" (bfloat16, disabled GradScaler) and for "cuda"
(float16; whether its scaler is enabled is only checkable on a machine with CUDA,
because GradScaler disables itself without one). Without float16 CPU autocast or
a CPU GradScaler (PyTorch < 2.3) the float16 checks are skipped.
"""
import warnings

import torch
import torch.nn as nn
from torchleet.runner import Skip

ENTRIES = ["amp_config", "train_step"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "float16 has a narrow exponent range: small gradients underflow to zero, so it "
    "needs loss scaling (an enabled GradScaler). bfloat16 has float32's range and "
    "needs none — return a GradScaler built with enabled=False so train_step can "
    "call it the same way on both paths.",
    "Only the forward pass and the loss go inside torch.autocast(device_type="
    "inputs.device.type, dtype=amp_dtype); backward runs outside it. Zero the "
    "gradients every step, and never cast the model itself to half precision.",
    "Order matters: backward through the scaled loss, unscale the optimizer's "
    "gradients, clip them to max_norm=1.0, step through the scaler, then update the "
    "scale. Clipping before unscaling clips the scaled gradients (S times too big), "
    "and stepping the optimizer directly applies inf/NaN updates after an overflow "
    "that the scaler would have skipped.",
]

MAX_NORM = 1.0


def _scaler(**kw):
    """A CPU GradScaler; torch.amp.GradScaler (device-generic) exists since torch 2.3."""
    if getattr(torch.amp, "GradScaler", None) is None:
        raise Skip("needs torch >= 2.3 (torch.amp.GradScaler)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.amp.GradScaler("cpu", **kw)


def _net():
    return nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1))


def _step(kind, model, criterion, optimizer, scaler, inputs, labels, amp_dtype):
    """The reference AMP step, or one of the common mistakes (for diagnosis)."""
    if kind != "no zero_grad":
        optimizer.zero_grad()
    with torch.autocast(device_type=inputs.device.type, dtype=amp_dtype):
        loss = criterion(model(inputs), labels)
    if kind == "no scaler":
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
        optimizer.step()
        return loss
    scaler.scale(loss).backward()
    if kind == "clip before unscale_":
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
        scaler.unscale_(optimizer)
    else:
        scaler.unscale_(optimizer)
        if kind != "no clipping":
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
    scaler.step(optimizer)
    scaler.update()
    return loss


WHY = {
    "no clipping": "the gradients are not clipped — clip_grad_norm_(model.parameters(), "
                   "max_norm=1.0) after scaler.unscale_(optimizer)",
    "clip before unscale_": "the gradients are clipped while still scaled — call "
                            "scaler.unscale_(optimizer) before clip_grad_norm_",
    "no zero_grad": "gradients accumulate across steps — call optimizer.zero_grad() "
                    "every step",
    "no scaler": "the loss is not scaled — backward through scaler.scale(loss) and step "
                 "with scaler.step(optimizer), then scaler.update()",
}


def _student(ns):
    def step(model, criterion, optimizer, scaler, inputs, labels, amp_dtype):
        try:
            return ns.train_step(model, criterion, optimizer, scaler, inputs, labels, amp_dtype)
        except RuntimeError as e:
            if "since the last update" in str(e) or "called after step()" in str(e):
                raise AssertionError(
                    "the GradScaler refused the next step's unscale_ because the previous "
                    "step did not end with scaler.update() — call scaler.update() once per "
                    "step, after scaler.step(optimizer)") from None
            raise
    return step


def _run(step, amp_dtype, enabled, init_scale, batches, seed, lr=0.1, momentum=0.9):
    torch.manual_seed(seed)
    model = _net()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scaler = _scaler(init_scale=init_scale, enabled=enabled)
    states, losses = [], []
    for x, y in batches:
        losses.append(step(model, nn.MSELoss(), opt, scaler, x, y, amp_dtype))
        states.append(([p.detach().clone() for p in model.parameters()],
                       scaler.get_scale() if enabled else None))
    return states, losses


def _same(a, b):
    (pa, sa), (pb, sb) = a, b
    return sa == sb and all(x.shape == y.shape and torch.allclose(x, y, rtol=1e-5, atol=1e-7)
                            for x, y in zip(pa, pb))


def _compare(ns, amp_dtype, enabled, init_scale, batches, seed, where, **opt):
    got, _ = _run(_student(ns), amp_dtype, enabled, init_scale, batches, seed, **opt)
    want, _ = _run(lambda *a: _step("ref", *a), amp_dtype, enabled, init_scale, batches, seed,
                   **opt)
    for k, (g, w) in enumerate(zip(got, want)):
        if _same(g, w):
            continue
        for kind, why in WHY.items():
            try:
                alt, _ = _run(lambda *a, kind=kind: _step(kind, *a), amp_dtype, enabled,
                              init_scale, batches, seed, **opt)
            except Exception:
                continue
            if _same(g, alt[k]):
                raise AssertionError(f"{where}, step {k + 1}: {why}")
        detail = ""
        if g[1] != w[1]:
            detail = f" (loss scale {g[1]} instead of {w[1]})"
        elif not all(torch.isfinite(p).all() for p in g[0]):
            detail = " (the weights became inf/NaN)"
        raise AssertionError(
            f"{where}, step {k + 1}: the weights after train_step differ from a reference "
            f"AMP step{detail} — forward and loss under autocast, "
            "scaler.scale(loss).backward(), scaler.unscale_, clip_grad_norm_(max_norm=1.0), "
            "scaler.step, scaler.update")


def _batches(n, seed, scale=5.0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(32, 10, generator=g), torch.randn(32, 1, generator=g) * scale)
            for _ in range(n)]


def _need_fp16_cpu():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.autocast("cpu", dtype=torch.float16):
                out = nn.Linear(2, 2)(torch.randn(1, 2))
            torch.amp.GradScaler("cpu")
        ok = out.dtype == torch.float16
    except Exception:
        ok = False
    if not ok:
        raise Skip("this PyTorch has no float16 CPU autocast / CPU GradScaler "
                   "(needs torch >= 2.3)")


# --------------------------------------------------------------------- checks
def check_amp_config_picks_dtype_and_scaler(ns):
    _scaler()                                     # Skip on torch < 2.3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cpu = ns.amp_config("cpu")
        cuda = ns.amp_config("cuda")
    for name, out in (("cpu", cpu), ("cuda", cuda)):
        assert isinstance(out, (tuple, list)) and len(out) == 2, \
            f'amp_config("{name}") should return (amp_dtype, scaler), got {out!r}'
    dt, sc = cpu
    assert dt == torch.bfloat16, f'amp_config("cpu") should use torch.bfloat16, got {dt}'
    assert isinstance(sc, torch.amp.GradScaler), (
        f'amp_config("cpu") should return a torch.amp.GradScaler (disabled), got '
        f"{type(sc).__name__} — a disabled scaler passes everything through, so the "
        "training step is the same code on both paths")
    assert not sc.is_enabled(), (
        'amp_config("cpu"): bfloat16 has float32\'s exponent range and needs no loss '
        "scaling — build the GradScaler with enabled=False")
    dt, sc = cuda
    assert dt == torch.float16, f'amp_config("cuda") should use torch.float16, got {dt}'
    assert isinstance(sc, torch.amp.GradScaler), \
        f'amp_config("cuda") should return a torch.amp.GradScaler, got {type(sc).__name__}'
    if torch.cuda.is_available():
        assert sc.is_enabled(), 'amp_config("cuda"): float16 needs an enabled GradScaler'


def check_forward_runs_under_autocast(ns):
    torch.manual_seed(0)
    model = _net()
    seen = []
    for m in model:
        if isinstance(m, nn.Linear):
            m.register_forward_hook(lambda mod, i, o: seen.append(o.dtype))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = _scaler(enabled=False)
    (x, y), = _batches(1, 1)
    loss = ns.train_step(model, nn.MSELoss(), opt, scaler, x, y, torch.bfloat16)
    assert seen, "train_step never ran the model"
    assert all(d == torch.bfloat16 for d in seen), (
        f"the Linear layers produced {sorted(set(map(str, seen)))}, expected "
        "torch.bfloat16 — run the forward pass inside "
        "torch.autocast(device_type=inputs.device.type, dtype=amp_dtype) "
        "(torch.cuda.amp.autocast does nothing for CPU tensors)")
    bad = [n for n, p in model.named_parameters() if p.dtype != torch.float32]
    assert not bad, (
        f"parameters {bad[:3]} are no longer float32 — autocast casts per operation; "
        "the weights (master copy) stay float32, so do not call model.half()")
    assert loss is not None, "train_step should return the loss"
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        torch.manual_seed(0)
        want = nn.MSELoss()(_net()(x), y).float()
    assert abs(float(loss) - float(want)) <= 1e-3 * max(1.0, abs(float(want))), (
        f"train_step returned {float(loss):.5f}, expected the batch loss before the "
        f"update ({float(want):.5f})")


def check_bfloat16_steps_match_reference(ns):
    _compare(ns, torch.bfloat16, False, 2.0 ** 16, _batches(3, 2), seed=1,
             where="bfloat16, scaler disabled")


def check_float16_scaled_steps_match_reference(ns):
    _need_fp16_cpu()
    _compare(ns, torch.float16, True, 2.0 ** 10, _batches(3, 3), seed=2,
             where="float16 with an enabled GradScaler")


def check_float16_small_gradients_survive(ns):
    """Labels a hair away from the predictions: dL/dy ~ 6e-8, at the bottom of float16's
    subnormal range. Unscaled, the float16 backward rounds most of it away. A large
    learning rate (plain SGD, one step) makes the tiny update measurable in float32."""
    _need_fp16_cpu()
    torch.manual_seed(4)
    model = _net()
    x = torch.randn(32, 10, generator=torch.Generator().manual_seed(5))
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
        y = model(x).float() + 1e-6
    _compare(ns, torch.float16, True, 2.0 ** 16, [(x, y)], seed=4,
             where="float16 with tiny gradients", lr=1e4, momentum=0.0)


def check_float16_overflow_skips_the_step(ns):
    _need_fp16_cpu()
    (x, y), normal = _batches(2, 6)
    overflow = (x * 1e5, y)                       # beyond float16's max (65504): inf
    got, _ = _run(_student(ns), torch.float16, True, 2.0 ** 10, [normal, overflow], seed=6)
    before, after = got[0], got[1]
    assert all(torch.isfinite(p).all() for p in after[0]), (
        "the optimizer stepped on inf/NaN gradients after a float16 overflow — step "
        "with scaler.step(optimizer), which skips the update when unscale_ found inf/NaN")
    assert all(torch.equal(a, b) for a, b in zip(before[0], after[0])), (
        "a step whose float16 forward overflowed must be skipped, but the weights changed "
        "(is the forward pass running under float16 autocast?)")
    assert after[1] == before[1] / 2, (
        f"after an overflow the loss scale should halve ({before[1]} -> {before[1] / 2}), "
        f"got {after[1]} — call scaler.update() after scaler.step()")
    _compare(ns, torch.float16, True, 2.0 ** 10, [normal, overflow, normal], seed=6,
             where="float16, a normal step after an overflow")


CHECKS = [
    check_amp_config_picks_dtype_and_scaler,
    check_forward_runs_under_autocast,
    check_bfloat16_steps_match_reference,
    check_float16_scaled_steps_match_reference,
    check_float16_small_gradients_survive,
    check_float16_overflow_skips_the_step,
]
