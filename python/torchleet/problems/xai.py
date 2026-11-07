"""xai — Grad-CAM (Selvaraju et al., 2017) on tiny random CNNs; nothing is downloaded.

Every test network is split into `features` (input -> target layer output A) and
`classify` (A -> logits), so the reference gets A and dy^c/dA straight from
torch.autograd.grad — no hooks. The solver's grad_cam only sees
(model, target_layer, x, target_class), as it would for a torchvision backbone.

Checked:
  * the raw map (N, h, w) is ReLU(sum_k alpha_k A^k), alpha_k the spatial mean of
    the gradient of the class *logit*, per sample — for an int class, None
    (argmax) and one class per sample; on non-square inputs; with layers after the
    target layer and a skip connection, so the gradient is not spatially constant
    (where per-pixel "gradient x activation" differs from Grad-CAM). Common
    mistakes (softmax score, summed gradients, normalising, no ReLU, ...) are named;
  * an analytic oracle: with global-average-pooling + a linear layer after the
    target layer, Grad-CAM is ReLU(CAM) / (h*w) (Selvaraju et al., Sec. 3.1);
  * batching: each sample gets its own weights;
  * every hook is removed, also when grad_cam raises; a second call agrees;
  * it works for a frozen model inside torch.no_grad() and leaves requires_grad
    and the grad mode as they were;
  * upsample_cam: bilinear (align_corners=False) to a non-square size, each map
    divided by its own maximum, an all-zero map stays zero.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module as _module_mod

ENTRIES = ["grad_cam", "upsample_cam"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "You need two tensors from the target layer: its output A (forward) and the "
    "gradient of the class score with respect to A (backward). A forward hook on "
    "target_layer can keep A; from there A.register_hook or torch.autograd.grad "
    "gives dy^c/dA. Keep the handle that register_forward_hook returns so you can "
    "remove the hook again.",
    "Score each sample by the logit of its own class (gather along dim 1, never the "
    "softmax) and sum those scores, so one backward pass gives every sample its "
    "own gradient. alpha is that gradient averaged over the spatial dims (2, 3) "
    "only — not over the batch — and the map is ReLU of the alpha-weighted sum of "
    "A over the channel dim: shape (N, h, w), not normalised.",
    "For a frozen model inside torch.no_grad(): run the forward under "
    "torch.enable_grad() on a detached copy of x that requires grad, so a graph "
    "exists without touching the parameters. Remove the hook in a finally block. "
    "upsample_cam: F.interpolate on cam.unsqueeze(1) with mode='bilinear', "
    "align_corners=False, then divide each map by its own amax over (H, W) plus a "
    "small eps such as 1e-8.",
]


# --------------------------------------------------------------- test networks
class _Net(nn.Module):
    """stem -> target -> head. `target` is the layer to explain; its output is A.

    head="late": a conv + tanh and a skip connection after the target layer, so
    dy/dA varies over space and A feeds the head twice (as in a ResNet block).
    head="gap": global average pooling + linear right after the target layer,
    the CAM architecture, where Grad-CAM has a closed form."""

    def __init__(self, k=6, n_classes=5, head="late", target_relu=True):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        layers = [nn.Conv2d(8, k, 3, padding=1)] + ([nn.ReLU()] if target_relu else [])
        self.target = nn.Sequential(*layers)
        self.head = head
        if head == "late":
            self.post = nn.Conv2d(k, k, 3, padding=1)
        self.fc = nn.Linear(k, n_classes)

    def features(self, x):
        return self.target(self.stem(x))

    def classify(self, a):
        if self.head == "late":
            a = torch.tanh(self.post(a)) + 0.5 * a
        return self.fc(a.mean(dim=(2, 3)))

    def forward(self, x):
        return self.classify(self.features(x))


def _net(seed, **kw):
    torch.manual_seed(seed)
    net = _Net(**kw).eval()
    with torch.no_grad():                      # bigger logits: classes differ clearly
        net.fc.weight.mul_(3.0)
    return net


def _x(seed, n, h, w):
    return torch.randn(n, 3, h, w, generator=torch.Generator().manual_seed(seed))


def _classes(net, x, target_class):
    n = x.shape[0]
    if target_class is None:
        with torch.no_grad():
            return net(x).argmax(dim=1)
    return torch.as_tensor(target_class, dtype=torch.long).reshape(-1).expand(n).clone()


def _parts(net, x, cls, score="logit"):
    """A, dScore/dA for the per-sample class `cls` (N,), computed without hooks."""
    with torch.enable_grad():
        a = net.features(x).detach().requires_grad_(True)
        logits = net.classify(a)
        if score == "softmax":
            logits = logits.softmax(dim=1)
        (g,) = torch.autograd.grad(logits.gather(1, cls[:, None]).sum(), a)
    return a.detach(), g


def _reference(net, x, cls):
    a, g = _parts(net, x, cls)
    alpha = g.mean(dim=(2, 3), keepdim=True)
    return torch.relu((alpha * a).sum(dim=1))


def _close(got, want, rtol=1e-4):
    scale = max(1.0, float(want.abs().max()))
    return got.shape == want.shape and torch.allclose(got, want, rtol=rtol, atol=1e-5 * scale)


def _diagnose(net, x, cls, got):
    """Name the mistake if `got` matches a common wrong variant of Grad-CAM."""
    a, g = _parts(net, x, cls)
    _, g_soft = _parts(net, x, cls, score="softmax")
    h, w = a.shape[-2:]
    alpha = g.mean(dim=(2, 3), keepdim=True)
    lin = (alpha * a).sum(dim=1)
    ref = torch.relu(lin)
    peak = ref.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    low = ref.amin(dim=(1, 2), keepdim=True)
    with torch.no_grad():
        pred = net(x).argmax(dim=1)
    variants = [
        (ref / peak, "the map is normalised by its maximum — grad_cam returns the raw "
                     "map; normalising is upsample_cam's job"),
        ((ref - low) / (peak - low).clamp_min(1e-12),
         "the map is min-max normalised — grad_cam returns the raw map"),
        (ref * (h * w), "alpha is the gradient summed over (h, w) — Grad-CAM averages "
                        "it (the 1/Z in the formula)"),
        (ref / a.shape[1], "the channels are averaged — the map is the alpha-weighted "
                           "*sum* over channels"),
        (torch.relu((g_soft.mean(dim=(2, 3), keepdim=True) * a).sum(dim=1)),
         "the gradient is taken of the softmax probability — backpropagate the class "
         "logit y^c instead"),
        (torch.relu((g * a).sum(dim=1)),
         "this is per-pixel gradient x activation — Grad-CAM first global-average-pools "
         "the gradient into one weight per channel"),
        (torch.relu((torch.relu(g).mean(dim=(2, 3), keepdim=True) * a).sum(dim=1)),
         "a ReLU was applied to the gradients — alpha uses the signed gradients; the "
         "ReLU goes on the final weighted sum"),
        (lin, "the ReLU is missing — negative evidence must be clipped to 0"),
        (lin.abs(), "abs() is not the ReLU — negative evidence must become 0"),
        (torch.relu((g.mean(dim=(0, 2, 3), keepdim=True) * a).sum(dim=1)),
         "the gradients are averaged over the batch too — average over dims (2, 3) "
         "only, so every sample gets its own weights"),
    ]
    if x.shape[0] > 1:
        variants.append((ref / x.shape[0],
                         "the class scores are averaged over the batch (.mean()), which "
                         "scales every gradient by 1/N — sum them instead"))
    if not torch.equal(pred, cls):
        variants.append((_reference(net, x, pred),
                         "this is the map of the predicted class — use target_class "
                         "when it is given"))
    for want, why in variants:
        if _close(got, want, rtol=1e-3):
            return why
    return None


def _cam(ns, net, x, target_class):
    out = ns.grad_cam(net, net.target, x, target_class)
    assert torch.is_tensor(out), \
        f"grad_cam should return a tensor, got {type(out).__name__}"
    return out.detach().float()


def _expect(ns, net, x, target_class, where):
    _compare(net, x, _classes(net, x, target_class), _cam(ns, net, x, target_class), where)


def _compare(net, x, cls, got, where):
    want = _reference(net, x, cls)
    n, h, w = want.shape
    assert tuple(got.shape) == (n, h, w), (
        f"{where}: expected the raw map at the target layer's resolution, (N, h, w) = "
        f"{(n, h, w)}, got {tuple(got.shape)}"
        + (" — drop the channel dim after summing over it" if got.dim() == 4 else ""))
    if not _close(got, want):
        why = _diagnose(net, x, cls, got)
        err = float((got - want).abs().max())
        raise AssertionError(
            f"{where}: the map is not ReLU(sum_k alpha_k A^k) (max abs error {err:.3g}, "
            f"map peak {float(want.max()):.3g})" + (f" — {why}" if why else ""))


# --------------------------------------------------------------------- checks
def check_grad_cam_shape_and_sign(ns):
    net = _net(0)
    cam = _cam(ns, net, _x(1, 2, 12, 16), 1)
    assert tuple(cam.shape) == (2, 6, 8), (
        f"a (2, 3, 12, 16) input gives target-layer maps of size 6x8, so grad_cam "
        f"should return (2, 6, 8); got {tuple(cam.shape)}")
    assert torch.isfinite(cam).all(), "the map contains NaN/inf"
    assert (cam >= 0).all(), \
        "the map has negative values — apply a ReLU to the weighted sum"


def check_grad_cam_matches_reference(ns):
    cases = [
        (dict(), _x(2, 1, 16, 16), 2, "one image, target_class=2"),
        (dict(), _x(3, 2, 14, 20), None, "two 14x20 images, target_class=None"),
        (dict(target_relu=False), _x(4, 3, 18, 12), torch.tensor([4, 0, 2]),
         "three 18x12 images, one class each, target layer without ReLU"),
        (dict(k=9, n_classes=7), _x(5, 2, 10, 22), 6, "two 10x22 images, target_class=6"),
    ]
    for i, (kw, x, tc, where) in enumerate(cases):
        _expect(ns, _net(10 + i, **kw), x, tc, where)


def check_grad_cam_equals_cam_for_gap_linear_head(ns):
    """Selvaraju et al. Sec. 3.1: with A -> global average pool -> linear W, the
    gradient dy^c/dA^k_ij is W[c, k] / (h*w) everywhere, so Grad-CAM is
    ReLU(sum_k W[c, k] A^k) / (h*w) — CAM (Zhou et al., 2016) up to the ReLU and scale.
    This oracle uses no autograd at all."""
    net = _net(20, head="gap", target_relu=False, k=8)
    x = _x(21, 2, 16, 12)
    for c in (0, 3):
        with torch.no_grad():
            a = net.features(x)
            want = torch.relu(torch.einsum("k,nkij->nij", net.fc.weight[c], a)) / (
                a.shape[-2] * a.shape[-1])
        got = _cam(ns, net, x, c)
        assert tuple(got.shape) == tuple(want.shape), \
            f"expected {tuple(want.shape)}, got {tuple(got.shape)}"
        assert _close(got, want), (
            f"for a global-average-pool + linear head, Grad-CAM of class {c} must equal "
            f"ReLU(sum_k W[c, k] A^k) / (h*w) (max abs error "
            f"{float((got - want).abs().max()):.3g})"
            + (f" — {why}" if (why := _diagnose(net, x, _classes(net, x, c), got)) else ""))


def check_batch_gives_each_sample_its_own_map(ns):
    net = _net(30)
    x = _x(31, 4, 16, 16)
    cls = torch.tensor([0, 3, 1, 3])
    batch = _cam(ns, net, x, cls)
    for i in range(4):
        alone = _cam(ns, net, x[i:i + 1], int(cls[i]))
        assert batch.shape[0] == 4 and _close(batch[i:i + 1], alone), (
            f"sample {i}'s map changes when it is computed inside a batch — score each "
            "sample by its own class logit and average the gradients over (h, w) only")


def _hooks(model):
    n = 0
    for m in model.modules():
        for attr in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks",
                     "_backward_pre_hooks"):
            n += len(getattr(m, attr, None) or {})
    for attr in ("_global_forward_hooks", "_global_forward_pre_hooks",
                 "_global_backward_hooks", "_global_backward_pre_hooks"):
        n += len(getattr(_module_mod, attr, None) or {})
    return n


def check_hooks_are_removed(ns):
    net = _net(40)
    x = _x(41, 2, 12, 12)
    before = _hooks(net)
    first = _cam(ns, net, x, 1)
    assert _hooks(net) == before, (
        "grad_cam left a hook registered on the model — keep the handle that "
        "register_*_hook returns and call handle.remove() before returning; a "
        "forgotten hook runs (and stores tensors) on every later forward pass")
    second = _cam(ns, net, x, 1)
    assert torch.allclose(first, second, atol=1e-6), \
        "calling grad_cam twice on the same input gives different maps — stale state?"
    try:
        ns.grad_cam(net, net.target, x, 10**6)       # no such class: should raise
    except Exception:
        pass
    assert _hooks(net) == before, (
        "when grad_cam raises (here: target_class out of range) its hook stays "
        "registered — remove it in a finally block")


def check_works_for_frozen_model_under_no_grad(ns):
    """Explaining a frozen pretrained backbone, from inference code, is the common case."""
    net = _net(50)
    for p in net.parameters():
        p.requires_grad_(False)
    x = _x(51, 2, 16, 12)
    try:
        with torch.no_grad():
            cam = _cam(ns, net, x, None)
            still_off = not torch.is_grad_enabled()
    except RuntimeError as e:
        raise AssertionError(
            "grad_cam fails for a frozen model (requires_grad=False) called inside "
            f"torch.no_grad(): {e} — build the graph under torch.enable_grad() from an "
            "input that requires grad") from None
    assert still_off, \
        "grad_cam switched gradients on globally — use torch.enable_grad() as a context"
    assert not any(p.requires_grad for p in net.parameters()), \
        "grad_cam unfroze the model's parameters — leave requires_grad as it was"
    _compare(net, x, _classes(net, x, None), cam,
             "frozen model called inside torch.no_grad()")


def check_upsample_cam(ns):
    g = torch.Generator().manual_seed(60)
    cam = torch.rand(3, 5, 4, generator=g) + 0.1    # min > 0: max- and min-max-scaling differ
    cam[1] = 0.0                                     # nothing positive: must stay 0
    cam[2] *= 7.0
    original = cam.clone()
    size = (23, 17)
    out = ns.upsample_cam(cam, size)
    assert torch.is_tensor(out), f"upsample_cam should return a tensor, got {type(out).__name__}"
    out = out.detach().float()
    assert torch.equal(cam, original), "upsample_cam modified its input in place"
    assert tuple(out.shape) == (3, *size), (
        f"cam (3, 5, 4) upsampled to size={size} should be (3, 23, 17), got "
        f"{tuple(out.shape)}" + (" — size is (H, W)" if tuple(out.shape) == (3, 17, 23) else ""))
    assert torch.isfinite(out).all(), \
        "an all-zero map gave NaN/inf — add a small eps to the max before dividing"
    assert torch.all(out[1] == 0), "an all-zero map must stay all zero"

    def up(mode, **kw):
        return F.interpolate(cam[:, None], size=size, mode=mode, **kw)[:, 0]

    bil = up("bilinear", align_corners=False)
    want = bil / bil.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    if torch.allclose(out, want, atol=1e-5):
        return
    peak_all = bil.amax().clamp_min(1e-12)
    lo = bil.amin(dim=(1, 2), keepdim=True)
    variants = [
        (bil, "the maps are not normalised — divide each by its own maximum"),
        (bil / peak_all, "the whole batch is divided by one global maximum — normalise "
                         "each map by its own maximum"),
        ((bil - lo) / (bil.amax(dim=(1, 2), keepdim=True) - lo).clamp_min(1e-12),
         "min-max normalisation — the statement divides each map by its maximum"),
        (up("bilinear", align_corners=True), None),
        (up("nearest"), None),
        (up("bicubic", align_corners=False), None),
    ]
    for v, why in variants:
        vn = v / v.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        if why is None and torch.allclose(out, vn, atol=1e-5):
            raise AssertionError(
                "the upsampling is not bilinear with align_corners=False "
                "(F.interpolate(cam.unsqueeze(1), size, mode='bilinear', "
                "align_corners=False))")
        if why is not None and torch.allclose(out, v, atol=1e-5):
            raise AssertionError(why)
    raise AssertionError(
        "upsample_cam disagrees with bilinear upsampling (align_corners=False) followed "
        f"by per-map max normalisation (max abs error {float((out - want).abs().max()):.3g})")


CHECKS = [
    check_grad_cam_shape_and_sign,
    check_grad_cam_matches_reference,
    check_grad_cam_equals_cam_for_gap_linear_head,
    check_batch_gives_each_sample_its_own_map,
    check_hooks_are_removed,
    check_works_for_frozen_model_under_no_grad,
    check_upsample_cam,
]
