"""conv-bn-fusion — fold eval-mode BatchNorm into the preceding conv / linear, and merge
a RepVGG block into one 3×3 conv.

Conventions graded here:
  * fuse_conv_bn(conv, bn) returns a NEW module of conv's type whose output equals
    bn(conv(x)) with bn in eval mode (running statistics). With
    s = gamma / sqrt(running_var + eps) (per OUTPUT channel):
        W' = W * s  (s broadcast along dim 0 of the weight, also for grouped/depthwise)
        b' = (b - running_mean) * s + beta,   b = 0 if conv.bias is None,
    gamma = 1, beta = 0 when bn.affine is False, eps = bn.eps. This is exactly
    torch.nn.utils.fusion.fuse_conv_bn_weights (torch 2.x), which the student must not
    call: the checks watch Python-level calls (sys.setprofile) made by all three entries
    into torch/nn/utils/fusion.py and torch.ao's fuser modules.
    Every hyperparameter of conv (stride, padding, dilation, groups, padding_mode) is
    kept, the result always has a bias Parameter, and conv / bn are not modified (state
    dicts and train/eval flags compared before and after; the result shares no storage
    with conv).
  * fuse_linear_bn(linear, bn1d): the same with W' = W * s[:, None]. (torch's
    fuse_linear_bn_eval does NOT handle affine=False BatchNorm1d in torch 2.15 —
    `None * tensor` — so that case is graded here too.)
  * repvgg_merge((conv3, bn3), (conv1, bn1), bn_id=None): RepVGG (Ding et al., CVPR 2021,
    "RepVGG: Making VGG-style ConvNets Great Again", §3.3, Eqs. 1–3 and Fig. 4). Each
    branch is fused with its BN, the fused 1×1 kernel is zero-padded to 3×3 at the CENTRE,
    the identity BN is written as a 3×3 kernel with K[o, o % (C/groups), 1, 1] = 1 (official
    repo's `_fuse_bn_tensor`), and kernels and biases are summed. Supported: 3×3 conv with
    padding 1 and 1×1 conv with padding 0, same stride (1 or 2) and groups (1, 2, depthwise),
    identity only when in_channels == out_channels and stride == 1 — otherwise a given
    bn_id must raise ValueError. The result equals bn3(conv3(x)) + bn1(conv1(x)) [+ bn_id(x)]
    in eval mode.

All equivalence checks use randomized running statistics (running_var spread over
orders of magnitude) and randomized affine parameters, in float64 with rtol = atol =
1e-10, plus float32 cases at a float32 tolerance. Tiny running_var with large eps
separates sqrt(var + eps) from sqrt(var) + eps; non-square in/out channel counts and a
Conv1d expose a scale broadcast along the wrong axis or a hard-coded 4-D view.
"""
import sys

import torch
import torch.nn as nn

ENTRIES = ["fuse_conv_bn", "fuse_linear_bn", "repvgg_merge"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "In eval mode BatchNorm is a fixed per-channel affine map y = s * x + t, with "
    "s = gamma / sqrt(running_var + eps) and t = beta - s * running_mean. A per-OUTPUT-channel "
    "scale and shift commute with the convolution, so they can be pushed into the filters and "
    "the bias.",
    "Dim 0 of a conv (or linear) weight is the output channel, also for grouped and depthwise "
    "convs: W' = W * s reshaped to (C_out, 1, 1, ...), b' = (b - mean) * s + beta. Use b = 0 "
    "when conv.bias is None, gamma = 1 and beta = 0 when bn.affine is False, and bn.eps INSIDE "
    "the square root. Start from copy.deepcopy(conv) and assign new nn.Parameters.",
    "RepVGG: fuse each branch with its own BN; zero-pad the fused 1x1 kernel by one pixel on "
    "every side so it lands at index [1, 1] of a 3x3; the identity BN is the fusion of a 3x3 "
    "'identity conv' whose kernel is 1 at [o, o % (C // groups), 1, 1] and 0 elsewhere; then add "
    "the three kernels and the three biases into one conv with kernel 3, padding 1 and the "
    "branches' stride and groups.",
]

F64 = torch.float64
_BANNED_FILES = ("nn/utils/fusion.py", "ao/quantization/fuser_method_mappings.py",
                 "ao/quantization/fuse_modules.py")


# ----------------------------------------------------------------- helpers

def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _randomize(module, seed, dtype=F64):
    """Random conv/linear weights; random BN affine params and running statistics."""
    g = _gen(seed)
    module = module.to(dtype)
    with torch.no_grad():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            C = module.num_features
            module.running_mean.copy_(torch.randn(C, generator=g, dtype=dtype) * 2.0)
            # variances spread over ~3 orders of magnitude
            module.running_var.copy_(torch.exp(torch.randn(C, generator=g, dtype=dtype) * 1.5))
            if module.affine:
                module.weight.copy_(torch.randn(C, generator=g, dtype=dtype) * 0.8 + 0.5)
                module.bias.copy_(torch.randn(C, generator=g, dtype=dtype))
        else:
            module.weight.copy_(torch.randn(module.weight.shape, generator=g, dtype=dtype) * 0.3)
            if module.bias is not None:
                module.bias.copy_(torch.randn(module.bias.shape, generator=g, dtype=dtype))
    return module.eval()


def _snapshot(*mods):
    return [({k: v.detach().clone() for k, v in m.state_dict().items()}, m.training,
             {k: v is None for k, v in m._parameters.items()}) for m in mods]


def _assert_unchanged(mods, snaps, fn_name):
    for m, (state, training, none_params) in zip(mods, snaps):
        name = type(m).__name__
        now = m.state_dict()
        assert set(now) == set(state), (
            f"{fn_name} changed the entries of the input {name}'s state_dict "
            f"({sorted(state)} -> {sorted(now)}) — build the result on a copy")
        for k, v in state.items():
            assert torch.equal(now[k], v), (
                f"{fn_name} modified the input {name}'s `{k}` — the original modules must stay "
                f"untouched (work on copy.deepcopy(conv), never write into conv.weight / .data)")
        assert m.training == training, (
            f"{fn_name} changed the input {name}'s train/eval mode")
        for k, was_none in none_params.items():
            assert (m._parameters.get(k) is None) == was_none, (
                f"{fn_name} replaced the input {name}'s `{k}` parameter — build the result on "
                f"copy.deepcopy(...), not on a shallow copy that shares the parameter dict")


def _calls_into_torch_fusion(fn, *args):
    """Run fn(*args) and report any Python function entered from torch's fusion helpers."""
    hits = []

    def prof(frame, event, arg):
        if event == "call":
            f = frame.f_code.co_filename.replace("\\", "/")
            if any(f.endswith(b) for b in _BANNED_FILES):
                hits.append(f"{frame.f_code.co_name} ({'/'.join(f.split('/')[-3:])})")

    old = sys.getprofile()
    sys.setprofile(prof)
    try:
        out = fn(*args)
    finally:
        sys.setprofile(old)
    return out, sorted(set(hits))


def _close(got, want, what, rtol=1e-10, atol=1e-10):
    assert torch.is_tensor(got), f"{what}: expected a tensor, got {type(got).__name__}"
    assert tuple(got.shape) == tuple(want.shape), (
        f"{what}: output shape {tuple(got.shape)}, expected {tuple(want.shape)}")
    err = (got.double() - want.double()).abs().max().item()
    tol = atol + rtol * want.abs().max().item()
    assert err <= tol, f"{what}: max |fused(x) - reference| = {err:.3e} (tolerance {tol:.1e})"


def _hyper(conv):
    return {k: getattr(conv, k) for k in ("in_channels", "out_channels", "kernel_size", "stride",
                                          "padding", "dilation", "groups", "padding_mode")}


def _fuse(ns, conv, bn, label):
    fused = ns.fuse_conv_bn(conv, bn)
    assert isinstance(fused, type(conv)), (
        f"{label}: fuse_conv_bn should return a new {type(conv).__name__}, got "
        f"{type(fused).__name__}")
    assert fused is not conv, (
        f"{label}: fuse_conv_bn returned the input conv itself — return a NEW module "
        f"(copy.deepcopy(conv)) so the original network stays usable")
    for name in ("weight", "bias"):
        p = getattr(fused, name, None)
        assert isinstance(p, nn.Parameter), (
            f"{label}: the fused conv's `{name}` should be an nn.Parameter, got "
            f"{type(p).__name__}" + (" — the fused conv ALWAYS needs a bias, even when the "
                                     "original had bias=False, because BN adds a shift"
                                     if name == "bias" and p is None else ""))
    return fused


def _diagnose_conv(conv, bn, fused):
    """Name the most likely bug from the fused parameters (float64 inputs only)."""
    w = conv.weight.detach()
    C = w.shape[0]
    mean, var, eps = bn.running_mean, bn.running_var, bn.eps
    gamma = bn.weight.detach() if bn.weight is not None else torch.ones(C, dtype=w.dtype)
    beta = bn.bias.detach() if bn.bias is not None else torch.zeros(C, dtype=w.dtype)
    b = conv.bias.detach() if conv.bias is not None else torch.zeros(C, dtype=w.dtype)
    view = (-1,) + (1,) * (w.dim() - 1)
    s = gamma / torch.sqrt(var + eps)
    fw, fb = fused.weight.detach().double(), fused.bias.detach().double()
    if fw.shape != w.shape:
        return (f" The fused weight has shape {tuple(fw.shape)} but the conv weight is "
                f"{tuple(w.shape)} — scale along dim 0 (output channels) only.")
    notes = []
    if not torch.allclose(fw, w * s.reshape(view), rtol=1e-9, atol=1e-12):
        s_bad = gamma / (torch.sqrt(var) + eps)
        if torch.allclose(fw, w * s_bad.reshape(view), rtol=1e-9, atol=1e-12):
            notes.append("the weight uses gamma / (sqrt(var) + eps); eps goes INSIDE the root")
        elif torch.allclose(fw, w, rtol=1e-9, atol=1e-12):
            notes.append("the weight was not scaled at all")
        elif w.shape[1] == C and torch.allclose(fw, w * s.reshape(1, -1, *view[2:]),
                                                rtol=1e-9, atol=1e-12):
            notes.append("the weight was scaled along the INPUT-channel axis (dim 1) — BN "
                         "acts on the conv's output channels (dim 0)")
        else:
            notes.append("the fused weight != W * gamma / sqrt(running_var + bn.eps) per "
                         "output channel")
    if not notes and not torch.allclose(fb, (b - mean) * s + beta, rtol=1e-9, atol=1e-12):
        if conv.bias is not None and torch.allclose(fb, -mean * s + beta, rtol=1e-9, atol=1e-12):
            notes.append("the bias ignores the conv's own bias b: b' = (b - mean) * s + beta")
        else:
            notes.append("the fused bias != (b - running_mean) * s + beta")
    return (" Likely: " + "; ".join(notes) + ".") if notes else ""


def _conv_cases(dtype=F64):
    """(label, conv, bn, input) — every case has nontrivial running statistics."""
    cases = [
        ("Conv2d(4, 6, 3, padding=1)", nn.Conv2d(4, 6, 3, padding=1), nn.BatchNorm2d(6),
         (2, 4, 7, 7)),
        ("Conv2d(3, 5, (3, 5), stride=2, padding=(1, 2), dilation=(1, 2), bias=False)",
         nn.Conv2d(3, 5, (3, 5), stride=2, padding=(1, 2), dilation=(1, 2), bias=False),
         nn.BatchNorm2d(5), (2, 3, 9, 11)),
        ("depthwise Conv2d(6, 6, 3, padding=1, groups=6)",
         nn.Conv2d(6, 6, 3, padding=1, groups=6), nn.BatchNorm2d(6), (2, 6, 6, 6)),
        ("grouped Conv2d(4, 8, 1, groups=2, bias=False)",
         nn.Conv2d(4, 8, 1, groups=2, bias=False), nn.BatchNorm2d(8), (3, 4, 5, 5)),
        ("Conv2d(3, 3, 3, padding=1, padding_mode='replicate')",
         nn.Conv2d(3, 3, 3, padding=1, padding_mode="replicate"), nn.BatchNorm2d(3), (2, 3, 6, 6)),
        ("Conv1d(3, 4, 5, padding=2) + BatchNorm1d", nn.Conv1d(3, 4, 5, padding=2),
         nn.BatchNorm1d(4), (2, 3, 12)),
    ]
    out = []
    for k, (label, conv, bn, shape) in enumerate(cases):
        conv, bn = _randomize(conv, 10 + k, dtype), _randomize(bn, 20 + k, dtype)
        x = torch.randn(shape, generator=_gen(30 + k), dtype=dtype) * 2.0 + 0.5
        out.append((label, conv, bn, x))
    return out


# ------------------------------------------------------------------ checks

def check_fuse_conv_bn_matches_eval_mode_conv_bn(ns):
    for label, conv, bn, x in _conv_cases():
        with torch.no_grad():
            want = bn(conv(x))
        fused = _fuse(ns, conv, bn, label)
        assert _hyper(fused) == _hyper(conv), (
            f"{label}: the fused conv's hyperparameters {_hyper(fused)} differ from the "
            f"original's {_hyper(conv)} — keep stride/padding/dilation/groups/padding_mode "
            f"(copy.deepcopy(conv) does)")
        with torch.no_grad():
            got = fused(x)
        try:
            _close(got, want, f"{label}, float64")
        except AssertionError as e:
            raise AssertionError(str(e) + _diagnose_conv(conv, bn, fused)) from None


def check_fuse_conv_bn_returns_fresh_module_and_leaves_inputs_untouched(ns):
    for label, conv, bn, x in _conv_cases()[:4]:
        snaps = _snapshot(conv, bn)
        with torch.no_grad():
            y_before = conv(x)
        fused = _fuse(ns, conv, bn, label)
        _assert_unchanged([conv, bn], snaps, "fuse_conv_bn")
        assert fused.weight.data_ptr() != conv.weight.data_ptr(), (
            f"{label}: the fused weight shares memory with the original conv's weight")
        with torch.no_grad():
            fused.weight.add_(1.0)                      # editing the copy ...
            fused.bias.add_(1.0)
            y_after = conv(x)
        assert torch.equal(y_before, y_after), (       # ... must not touch the original
            f"{label}: editing the fused conv in place changed the ORIGINAL conv's output — "
            f"the two share tensors; build the result on copy.deepcopy(conv)")
        _assert_unchanged([conv, bn], snaps, "fuse_conv_bn")


def check_fuse_conv_bn_with_affine_false_and_custom_eps(ns):
    cases = [
        ("BatchNorm2d(6, affine=False)", nn.Conv2d(4, 6, 3, padding=1),
         nn.BatchNorm2d(6, affine=False), None),
        ("BatchNorm2d(6, affine=False) after Conv2d(bias=False)",
         nn.Conv2d(4, 6, 3, padding=1, bias=False), nn.BatchNorm2d(6, affine=False), None),
        ("BatchNorm2d(6, eps=1e-3) with running_var ~ 1e-4", nn.Conv2d(4, 6, 3, padding=1),
         nn.BatchNorm2d(6, eps=1e-3), 1e-4),
        ("BatchNorm2d(6, eps=0.1)", nn.Conv2d(4, 6, 3, padding=1),
         nn.BatchNorm2d(6, eps=0.1), 0.05),
    ]
    for k, (label, conv, bn, var_scale) in enumerate(cases):
        conv, bn = _randomize(conv, 40 + k), _randomize(bn, 50 + k)
        if var_scale is not None:
            with torch.no_grad():
                bn.running_var.mul_(var_scale / bn.running_var.mean())
        x = torch.randn(2, 4, 5, 5, generator=_gen(60 + k), dtype=F64)
        with torch.no_grad():
            want = bn(conv(x))
        fused = _fuse(ns, conv, bn, label)
        with torch.no_grad():
            got = fused(x)
        try:
            _close(got, want, label)
        except AssertionError as e:
            raise AssertionError(str(e) + _diagnose_conv(conv, bn, fused)) from None


def check_fuse_conv_bn_float32(ns):
    conv = _randomize(nn.Conv2d(8, 16, 3, padding=1), 70, torch.float32)
    bn = _randomize(nn.BatchNorm2d(16), 71, torch.float32)
    x = torch.randn(4, 8, 10, 10, generator=_gen(72))
    with torch.no_grad():
        want = bn(conv(x))
        fused = _fuse(ns, conv, bn, "float32 Conv2d(8, 16, 3)")
        got = fused(x)
    assert fused.weight.dtype == torch.float32 and fused.bias.dtype == torch.float32, (
        f"float32 inputs should give float32 fused parameters, got {fused.weight.dtype} / "
        f"{fused.bias.dtype}")
    _close(got, want, "float32 Conv2d(8, 16, 3)", rtol=1e-5, atol=1e-5)


def check_fuse_conv_bn_does_not_call_torch_fusion(ns):
    conv = _randomize(nn.Conv2d(3, 4, 3, padding=1), 80)
    bn = _randomize(nn.BatchNorm2d(4), 81)
    _, hits = _calls_into_torch_fusion(ns.fuse_conv_bn, conv, bn)
    lin = _randomize(nn.Linear(5, 3), 82)
    bn1 = _randomize(nn.BatchNorm1d(3), 83)
    _, hits_lin = _calls_into_torch_fusion(ns.fuse_linear_bn, lin, bn1)
    conv3, bn3, conv1, bn1x1, bn_id = _repvgg(4, 4, 1, 1, True, 84)
    _, hits_rep = _calls_into_torch_fusion(ns.repvgg_merge, (conv3, bn3), (conv1, bn1x1), bn_id)
    for fn_name, used in (("fuse_conv_bn", hits), ("fuse_linear_bn", hits_lin),
                          ("repvgg_merge", hits_rep)):
        assert not used, (
            f"your {fn_name} calls PyTorch's own fusion helpers ({', '.join(used)}) — derive and "
            f"compute W' and b' yourself (repvgg_merge may call your own fuse_conv_bn); use "
            f"torch.nn.utils.fusion only to check your answer")


def check_fuse_linear_bn(ns):
    cases = [
        ("Linear(7, 5) + BatchNorm1d(5)", nn.Linear(7, 5), nn.BatchNorm1d(5)),
        ("Linear(6, 4, bias=False) + BatchNorm1d(4)", nn.Linear(6, 4, bias=False), nn.BatchNorm1d(4)),
        ("Linear(5, 5) + BatchNorm1d(5, affine=False)", nn.Linear(5, 5),
         nn.BatchNorm1d(5, affine=False)),
        ("Linear(3, 6) + BatchNorm1d(6, eps=1e-2)", nn.Linear(3, 6), nn.BatchNorm1d(6, eps=1e-2)),
    ]
    for k, (label, lin, bn) in enumerate(cases):
        lin, bn = _randomize(lin, 90 + k), _randomize(bn, 100 + k)
        x = torch.randn(9, lin.in_features, generator=_gen(110 + k), dtype=F64)
        snaps = _snapshot(lin, bn)
        with torch.no_grad():
            want = bn(lin(x))
        fused = ns.fuse_linear_bn(lin, bn)
        assert isinstance(fused, nn.Linear) and fused is not lin, (
            f"{label}: fuse_linear_bn should return a NEW nn.Linear, got "
            f"{'the input module itself' if fused is lin else type(fused).__name__}")
        assert isinstance(fused.bias, nn.Parameter), (
            f"{label}: the fused Linear needs a bias Parameter (BN adds a shift), got "
            f"{type(fused.bias).__name__}")
        assert (fused.in_features, fused.out_features) == (lin.in_features, lin.out_features), (
            f"{label}: fused Linear is ({fused.in_features} -> {fused.out_features}), expected "
            f"({lin.in_features} -> {lin.out_features})")
        _assert_unchanged([lin, bn], snaps, "fuse_linear_bn")
        with torch.no_grad():
            got = fused(x)
        _close(got, want, label)


# ----------------------------------------------------------------- RepVGG

def _repvgg(cin, cout, stride, groups, identity, seed, dtype=F64, conv_bias=False):
    conv3 = _randomize(nn.Conv2d(cin, cout, 3, stride, 1, groups=groups, bias=conv_bias),
                       seed, dtype)
    conv1 = _randomize(nn.Conv2d(cin, cout, 1, stride, 0, groups=groups, bias=conv_bias),
                       seed + 1, dtype)
    bn3 = _randomize(nn.BatchNorm2d(cout), seed + 2, dtype)
    bn1 = _randomize(nn.BatchNorm2d(cout), seed + 3, dtype)
    bn_id = _randomize(nn.BatchNorm2d(cin), seed + 4, dtype) if identity else None
    return conv3, bn3, conv1, bn1, bn_id


def _repvgg_ref(conv3, bn3, conv1, bn1, bn_id, x):
    with torch.no_grad():
        y = bn3(conv3(x)) + bn1(conv1(x))
        if bn_id is not None:
            y = y + bn_id(x)
    return y


def _merge_and_compare(ns, label, cin, cout, stride, groups, identity, seed, dtype=F64,
                       conv_bias=False, rtol=1e-10, atol=1e-10):
    conv3, bn3, conv1, bn1, bn_id = _repvgg(cin, cout, stride, groups, identity, seed, dtype,
                                            conv_bias)
    mods = [m for m in (conv3, bn3, conv1, bn1, bn_id) if m is not None]
    snaps = _snapshot(*mods)
    x = torch.randn(2, cin, 9, 8, generator=_gen(seed + 5), dtype=dtype) + 0.3
    want = _repvgg_ref(conv3, bn3, conv1, bn1, bn_id, x)
    merged = ns.repvgg_merge((conv3, bn3), (conv1, bn1), bn_id)
    assert isinstance(merged, nn.Conv2d), (
        f"{label}: repvgg_merge should return an nn.Conv2d, got {type(merged).__name__}")
    got_h = {k: getattr(merged, k) for k in ("in_channels", "out_channels", "kernel_size",
                                             "stride", "padding", "groups")}
    want_h = {"in_channels": cin, "out_channels": cout, "kernel_size": (3, 3),
              "stride": (stride, stride), "padding": (1, 1), "groups": groups}
    assert got_h == want_h, (
        f"{label}: the merged conv should be {want_h}, got {got_h}")
    assert isinstance(merged.bias, nn.Parameter), (
        f"{label}: the merged conv needs a bias Parameter (the sum of the three branch biases)")
    _assert_unchanged(mods, snaps, "repvgg_merge")
    with torch.no_grad():
        got = merged(x)
    try:
        _close(got, want, f"{label}", rtol=rtol, atol=atol)
    except AssertionError as e:
        hint = ""
        if bn_id is not None:
            no_id = _repvgg_ref(conv3, bn3, conv1, bn1, None, x)
            if torch.allclose(got.double(), no_id.double() + x.double(), rtol=1e-8, atol=1e-8):
                hint = (" Your identity branch adds x itself — the identity branch is bn_id(x): "
                        "fuse the BN into the identity kernel too.")
            elif torch.allclose(got.double(), no_id.double(), rtol=1e-8, atol=1e-8):
                hint = " The identity branch is missing from the merged conv."
        raise AssertionError(
            str(e) + hint + " Check: the 1x1 kernel is padded to the CENTRE of the 3x3 "
            "(F.pad by 1 on every side), each branch uses its OWN BN, and the identity kernel "
            "is 1 at [o, o % (C // groups), 1, 1].") from None


def check_repvgg_merge_with_identity_branch(ns):
    _merge_and_compare(ns, "RepVGG block C=6 -> 6, stride 1, with identity", 6, 6, 1, 1, True, 200)
    _merge_and_compare(ns, "RepVGG block C=4 -> 4 with biased branch convs", 4, 4, 1, 1, True, 210,
                       conv_bias=True)
    _merge_and_compare(ns, "float32 RepVGG block C=8 -> 8, with identity", 8, 8, 1, 1, True, 220,
                       dtype=torch.float32, rtol=1e-5, atol=1e-5)


def check_repvgg_merge_without_identity_branch(ns):
    _merge_and_compare(ns, "RepVGG block 4 -> 8 (in != out, no identity)", 4, 8, 1, 1, False, 300)
    _merge_and_compare(ns, "RepVGG block 6 -> 6, stride 2 (no identity)", 6, 6, 2, 1, False, 310)
    _merge_and_compare(ns, "RepVGG block 6 -> 6, stride 1, identity_bn=None", 6, 6, 1, 1, False, 320)


def check_repvgg_merge_grouped_and_depthwise(ns):
    _merge_and_compare(ns, "RepVGG block C=8, groups=2, with identity", 8, 8, 1, 2, True, 400)
    _merge_and_compare(ns, "RepVGG block C=6, groups=6 (depthwise), with identity", 6, 6, 1, 6,
                       True, 410)
    _merge_and_compare(ns, "RepVGG block 4 -> 8, groups=4, stride 2", 4, 8, 2, 4, False, 420)


def check_repvgg_merge_rejects_an_impossible_identity_branch(ns):
    for label, cin, cout, stride in (("in_channels 4 != out_channels 8", 4, 8, 1),
                                     ("stride 2", 6, 6, 2)):
        conv3, bn3, conv1, bn1, _ = _repvgg(cin, cout, stride, 1, False, 500)
        bn_id = _randomize(nn.BatchNorm2d(cin), 505)
        try:
            ns.repvgg_merge((conv3, bn3), (conv1, bn1), bn_id)
        except ValueError:
            continue
        except Exception as e:
            raise AssertionError(
                f"with {label} an identity branch cannot exist (its output would not match the "
                f"conv branches' shape); repvgg_merge should raise ValueError, but raised "
                f"{type(e).__name__}: {e}") from None
        raise AssertionError(
            f"with {label} an identity branch cannot exist (RepVGG only adds it when "
            f"in_channels == out_channels and stride == 1); repvgg_merge should raise "
            f"ValueError when given identity_bn, but returned normally")


CHECKS = [
    check_fuse_conv_bn_matches_eval_mode_conv_bn,
    check_fuse_conv_bn_returns_fresh_module_and_leaves_inputs_untouched,
    check_fuse_conv_bn_with_affine_false_and_custom_eps,
    check_fuse_conv_bn_float32,
    check_fuse_conv_bn_does_not_call_torch_fusion,
    check_fuse_linear_bn,
    check_repvgg_merge_with_identity_branch,
    check_repvgg_merge_without_identity_branch,
    check_repvgg_merge_grouped_and_depthwise,
    check_repvgg_merge_rejects_an_impossible_identity_branch,
]
