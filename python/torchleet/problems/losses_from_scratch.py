"""losses-from-scratch — cross-entropy (class weights, ignore_index, label smoothing,
K-dim inputs), numerically stable BCE-with-logits (pos_weight), sigmoid focal loss
(RetinaNet / torchvision.ops.sigmoid_focal_loss) and soft Dice loss.

Oracles: F.cross_entropy and F.binary_cross_entropy_with_logits for values AND
gradients; the focal and Dice references are written out below from their papers'
formulas (the focal one mirrors torchvision.ops.sigmoid_focal_loss line by line).
"""
import torch
import torch.nn.functional as F

ENTRIES = ["cross_entropy", "binary_cross_entropy_with_logits", "sigmoid_focal_loss", "dice_loss"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Work in log space. log_softmax(x) = x - logsumexp(x) never forms exp(large). For BCE, "
    "-log(sigmoid(x)) = softplus(-x) = log(1 + exp(-x)): compute it with torch.logaddexp(0, -x) "
    "or F.softplus(-x). Move the class axis last and flatten, so (N, C, d1, ...) becomes (M, C) "
    "with M targets.",
    "Mean reduction is a WEIGHTED mean: divide by the sum of w[target] over non-ignored "
    "targets, not by N. Replace ignored targets by a valid index before gather(), then "
    "zero them out. PyTorch's label smoothing = (1-eps) * NLL + (eps/C) * sum_c w_c * "
    "(-log p_c), with both terms normalised by the same denominator.",
    "BCE: loss = (1-y)*x + (1 + (pos_weight-1)*y) * softplus(-x). The hand-written "
    "max(-x,0) + log1p(exp(-|x|)) has the right value, but autograd gets the slope wrong at "
    "exactly x = 0. Focal: p_t = p*y + (1-p)*(1-y), loss = alpha_t * (1-p_t)^gamma * BCE "
    "with alpha_t = alpha*y + (1-alpha)*(1-y) (none when alpha < 0), averaged over ALL "
    "elements. Dice per sample: 1 - (2*sum(p*t) + eps) / (sum(p) + sum(t) + eps), then "
    "the batch mean.",
]


# ---------------------------------------------------------------- references

def _focal_ref(x, y, alpha, gamma, reduction):
    """torchvision.ops.sigmoid_focal_loss, verbatim semantics."""
    p = torch.sigmoid(x)
    ce = F.binary_cross_entropy_with_logits(x, y, reduction="none")
    p_t = p * y + (1 - p) * (1 - y)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * y + (1 - alpha) * (1 - y)
        loss = alpha_t * loss
    return {"none": loss, "mean": loss.mean(), "sum": loss.sum()}[reduction]


def _dice_ref(x, t, eps):
    p = torch.sigmoid(x).flatten(1)
    t = t.flatten(1)
    dice = (2 * (p * t).sum(1) + eps) / (p.sum(1) + t.sum(1) + eps)
    return (1 - dice).mean()


def _compare(name, fn, ref, x, *args, rtol=1e-5, atol=1e-5, hint="", nan_hint="", **kw):
    """Value and gradient (w.r.t. x) of fn vs ref. `hint` is appended to every failure,
    `nan_hint` only to NaN/Inf failures."""
    xa = x.detach().clone().requires_grad_(True)
    xb = x.detach().clone().requires_grad_(True)
    got = fn(xa, *args, **kw)
    want = ref(xb, *args, **kw)
    assert isinstance(got, torch.Tensor), f"{name} must return a tensor, got {type(got).__name__}"
    assert got.shape == want.shape, f"{name}: expected shape {tuple(want.shape)}, got {tuple(got.shape)}"
    assert torch.isfinite(got).all(), f"{name} returned NaN/Inf{nan_hint or hint}"
    assert torch.allclose(got, want, rtol=rtol, atol=atol), f"{name}: {_diff(got, want)}{hint}"
    (ga,) = torch.autograd.grad(got.sum(), xa)
    (gb,) = torch.autograd.grad(want.sum(), xb)
    assert torch.isfinite(ga).all(), f"{name}: gradient has NaN/Inf{nan_hint or hint}"
    assert torch.allclose(ga, gb, rtol=rtol, atol=atol), \
        f"{name}: gradient w.r.t. the logits is wrong — {_diff(ga, gb)}{hint}"


def _diff(got, want):
    got, want = got.detach(), want.detach()
    if got.numel() == 1:
        return f"got {got.item():.6f}, expected {want.item():.6f}"
    i = (got - want).abs().flatten().argmax().item()
    return (f"max diff {(got - want).abs().max().item():.3g} at flat index {i} "
            f"(got {got.flatten()[i].item():.6g}, expected {want.flatten()[i].item():.6g})")


def _gen(seed):
    return torch.Generator().manual_seed(seed)


# ---------------------------------------------------------------- cross-entropy

def _ce_case(seed, shape, ignore_frac=0.0, ignore_index=-100):
    g = _gen(seed)
    x = torch.randn(*shape, generator=g) * 3
    C = shape[1]
    t = torch.randint(0, C, (shape[0],) + tuple(shape[2:]), generator=g)
    if ignore_frac:
        t[torch.rand(t.shape, generator=g) < ignore_frac] = ignore_index
        t.view(-1)[0] = 1                                   # never everything ignored
    return x, t


def check_ce_matches_torch_basic(ns):
    for seed, shape in [(0, (8, 5)), (1, (1, 3)), (2, (16, 10))]:
        x, t = _ce_case(seed, shape)
        _compare(f"cross_entropy on {shape}", ns.cross_entropy, F.cross_entropy, x, t)


def check_ce_ignore_index(ns):
    x, t = _ce_case(3, (12, 4), ignore_frac=0.4)
    _compare("cross_entropy with ignore_index=-100 targets", ns.cross_entropy, F.cross_entropy, x, t,
             hint=" — ignored targets contribute nothing and are not counted in the mean")
    x, t = _ce_case(4, (12, 6), ignore_frac=0.3, ignore_index=255)
    _compare("cross_entropy with ignore_index=255", ns.cross_entropy, F.cross_entropy, x, t,
             ignore_index=255, hint=" — honour the ignore_index argument")
    x, t = _ce_case(5, (12, 4))
    _compare("cross_entropy with ignore_index=2 (a real class id)", ns.cross_entropy,
             F.cross_entropy, x, t, ignore_index=2)


def check_ce_class_weights(ns):
    w = torch.tensor([0.2, 1.0, 3.0, 0.7, 1.5])
    x, t = _ce_case(6, (10, 5))
    _compare("cross_entropy with weight", ns.cross_entropy, F.cross_entropy, x, t, weight=w,
             hint=" — with class weights the mean divides by sum(weight[target]), not by N")
    x, t = _ce_case(7, (10, 5), ignore_frac=0.3)
    _compare("cross_entropy with weight + ignore_index", ns.cross_entropy, F.cross_entropy, x, t,
             weight=w, hint=" — the denominator is sum(weight[target]) over NON-ignored targets")


def check_ce_label_smoothing(ns):
    x, t = _ce_case(8, (10, 5))
    for eps in (0.1, 0.3):
        _compare(f"cross_entropy with label_smoothing={eps}", ns.cross_entropy, F.cross_entropy,
                 x, t, label_smoothing=eps,
                 hint=" — smoothing mixes the one-hot target with the uniform distribution "
                      "eps/C over ALL C classes (target class included)")
    w = torch.tensor([0.2, 1.0, 3.0, 0.7, 1.5])
    x, t = _ce_case(9, (10, 5), ignore_frac=0.3)
    _compare("cross_entropy with weight + ignore_index + label_smoothing=0.2", ns.cross_entropy,
             F.cross_entropy, x, t, weight=w, label_smoothing=0.2,
             hint=" — PyTorch weights the smoothing term per class too: (eps/C) * sum_c w_c*(-log p_c),"
                  " divided by the same sum(weight[target])")


def check_ce_k_dim_inputs(ns):
    w = torch.tensor([0.5, 2.0, 1.0, 1.5])
    for shape in [(2, 4, 7), (2, 4, 3, 5), (1, 4, 2, 3, 2)]:
        x, t = _ce_case(10 + len(shape), shape, ignore_frac=0.2)
        _compare(f"cross_entropy on {shape}", ns.cross_entropy, F.cross_entropy, x, t,
                 hint=" — the class axis is dim 1: move it last before flattening "
                      "(a plain reshape(-1, C) mixes classes with positions)")
        _compare(f"cross_entropy on {shape} with weight + label_smoothing", ns.cross_entropy,
                 F.cross_entropy, x, t, weight=w, label_smoothing=0.1)


def check_ce_extreme_logits(ns):
    x = torch.tensor([[1e4, -1e4, 0.0], [-1e4, 1e4, 5.0], [3e3, 3e3, -3e3], [0.0, 0.0, 0.0]])
    t = torch.tensor([1, 1, 2, 0])
    hint = " — compute log-softmax as x - logsumexp(x), never log(softmax(x))"
    _compare("cross_entropy on logits of size 1e4", ns.cross_entropy, F.cross_entropy, x, t,
             nan_hint=hint)
    _compare("cross_entropy on logits of size 1e4 with label_smoothing", ns.cross_entropy,
             F.cross_entropy, x, t, label_smoothing=0.1, nan_hint=hint)


# ---------------------------------------------------------------- BCE with logits

def check_bce_matches_torch(ns):
    g = _gen(20)
    x = torch.randn(9, 4, generator=g) * 5
    y_hard = (torch.rand(9, 4, generator=g) > 0.6).float()
    y_soft = torch.rand(9, 4, generator=g)
    pw = torch.tensor([1.0, 3.0, 0.5, 10.0])
    for y, kind in [(y_hard, "0/1"), (y_soft, "soft")]:
        _compare(f"binary_cross_entropy_with_logits ({kind} targets)",
                 ns.binary_cross_entropy_with_logits, F.binary_cross_entropy_with_logits, x, y)
        _compare(f"binary_cross_entropy_with_logits with pos_weight ({kind} targets)",
                 ns.binary_cross_entropy_with_logits, F.binary_cross_entropy_with_logits, x, y,
                 pos_weight=pw,
                 hint=" — pos_weight multiplies only the positive term: "
                      "-[pos_weight*y*log(sigmoid(x)) + (1-y)*log(1-sigmoid(x))]")


def check_bce_reduction_modes(ns):
    g = _gen(21)
    x, y = torch.randn(5, 3, generator=g) * 4, torch.rand(5, 3, generator=g)
    for red in ("none", "sum"):
        _compare(f"binary_cross_entropy_with_logits(reduction='{red}')",
                 ns.binary_cross_entropy_with_logits, F.binary_cross_entropy_with_logits, x, y,
                 reduction=red)


def check_bce_extreme_logits(ns):
    x = torch.tensor([1e4, -1e4, 1e4, -1e4, 50.0, -50.0, 30.0, -30.0, 3.0])
    y = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.3, 0.7, 0.0, 1.0, 1.0])
    for pw in (None, torch.tensor(2.0)):
        out = ns.binary_cross_entropy_with_logits(x, y, pos_weight=pw, reduction="none")
        # x = 1e4 with y = 0 costs 1e4; a clamped-probability version caps it near -log(eps)
        clamped = (isinstance(out, torch.Tensor) and out.shape == x.shape
                   and bool(torch.isfinite(out).all()) and out[2].item() < 100)
        _compare(f"binary_cross_entropy_with_logits on |logits| up to 1e4 (pos_weight={pw})",
                 ns.binary_cross_entropy_with_logits, F.binary_cross_entropy_with_logits, x, y,
                 pos_weight=pw, reduction="none",
                 nan_hint=" — log(sigmoid(x)) underflows to log(0); use (1-y)*x + softplus(-x)",
                 hint=" — the loss is capped for large |x|: clamping probabilities limits it to "
                      "-log(eps) and zeroes the gradient; use (1-y)*x + softplus(-x)" if clamped else "")


def check_bce_gradient_at_zero_logits(ns):
    # A zero-initialised classifier head outputs exactly 0 on the first step.
    x = torch.zeros(6)
    y = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.5, 1.0])
    _compare("binary_cross_entropy_with_logits at logits == 0", ns.binary_cross_entropy_with_logits,
             F.binary_cross_entropy_with_logits, x, y, reduction="none",
             hint=" — the true slope is sigmoid(x) - y. max(-x,0) + log1p(exp(-|x|)) has kinks at 0 "
                  "that cancel in value but NOT in autograd's subgradients; use torch.logaddexp(0, -x) "
                  "or F.softplus(-x)")


# ---------------------------------------------------------------- focal

def _focal_case(seed, shape=(6, 7)):
    g = _gen(seed)
    x = torch.randn(*shape, generator=g) * 3
    y = (torch.rand(*shape, generator=g) < 0.25).float()
    return x, y


def check_focal_matches_reference(ns):
    x, y = _focal_case(30)
    for alpha in (0.25, 0.5, 0.9):
        for gamma in (0.0, 1.5, 2.0):
            for red in ("mean", "sum", "none"):
                kw = dict(alpha=alpha, gamma=gamma, reduction=red)
                got = ns.sigmoid_focal_loss(x, y, **kw)
                want = _focal_ref(x, y, **kw)
                hint = ""
                if got.shape == want.shape and torch.allclose(
                        got, _focal_ref(x, y, 1 - alpha, gamma, red), rtol=1e-5, atol=1e-6):
                    hint = " — alpha weights the POSITIVES (y=1) and 1-alpha the negatives"
                elif red == "mean" and got.numel() == 1 and torch.allclose(
                        got, _focal_ref(x, y, alpha, gamma, "sum") / y.sum(), rtol=1e-5):
                    hint = (" — reduction='mean' averages over ALL elements (torchvision); "
                            "RetinaNet's division by the number of positive anchors happens "
                            "outside this function")
                _compare(f"sigmoid_focal_loss(alpha={alpha}, gamma={gamma}, reduction='{red}')",
                         ns.sigmoid_focal_loss, _focal_ref, x, y, hint=hint, **kw)


def check_focal_negative_alpha_disables_weighting(ns):
    x, y = _focal_case(31)
    _compare("sigmoid_focal_loss(alpha=-1, gamma=2)", ns.sigmoid_focal_loss, _focal_ref, x, y,
             alpha=-1.0, gamma=2.0, reduction="mean",
             hint=" — alpha < 0 means no alpha weighting at all")


def check_focal_gamma0_no_alpha_equals_bce(ns):
    x, y = _focal_case(32)
    got = ns.sigmoid_focal_loss(x, y, alpha=-1.0, gamma=0.0, reduction="mean")
    want = F.binary_cross_entropy_with_logits(x, y)
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-6), (
        f"with gamma=0 and alpha<0 focal loss must equal BCE-with-logits ({got.item():.6f} vs "
        f"{want.item():.6f})")


def check_focal_extreme_logits(ns):
    x = torch.tensor([[1e4, -1e4, 1e4, -1e4, 40.0, -40.0]])
    y = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
    _compare("sigmoid_focal_loss on |logits| up to 1e4", ns.sigmoid_focal_loss, _focal_ref, x, y,
             alpha=0.25, gamma=2.0, reduction="none",
             nan_hint=" — build the cross-entropy part from a stable BCE-with-logits, not log(sigmoid)")


# ---------------------------------------------------------------- dice

def _dice_case(seed, n=4, h=8, w=8):
    g = _gen(seed)
    x = torch.randn(n, 1, h, w, generator=g) * 2
    t = (torch.rand(n, 1, h, w, generator=g) < 0.3).float()
    t[0] = 0.0                                              # one sample with an empty mask
    return x, t


def check_dice_matches_formula(ns):
    x, t = _dice_case(40)
    tt = t.flatten(1)

    def dice_of(p, per_sample=True):
        if not per_sample:
            return 1 - (2 * (p * tt).sum() + eps) / (p.sum() + tt.sum() + eps)
        return (1 - (2 * (p * tt).sum(1) + eps) / (p.sum(1) + tt.sum(1) + eps)).mean()

    for eps in (1.0, 1e-3):
        got = ns.dice_loss(x, t, eps=eps)
        hint = " — per sample: 1 - (2*sum(p*t) + eps) / (sum(p) + sum(t) + eps), p = sigmoid(logits)"
        if isinstance(got, torch.Tensor) and got.numel() == 1:
            if torch.allclose(got, dice_of(torch.sigmoid(x).flatten(1), per_sample=False), rtol=1e-5):
                hint = " — compute Dice PER SAMPLE (sum over C,H,W), then average over the batch"
            elif torch.allclose(got, dice_of(x.flatten(1)), rtol=1e-5):
                hint = " — apply sigmoid to the logits first"
        _compare(f"dice_loss(eps={eps})", ns.dice_loss, _dice_ref, x, t, eps=eps, hint=hint)


def check_dice_perfect_and_empty(ns):
    g = _gen(41)
    t = (torch.rand(3, 1, 8, 8, generator=g) < 0.4).float()
    t[2] = 0.0                                              # empty mask, predicted empty
    perfect = (t * 2 - 1) * 20.0                            # +20 inside the mask, -20 outside
    got = ns.dice_loss(perfect, t)
    assert torch.isfinite(got).all() and abs(got.item()) < 1e-4, (
        f"a perfect prediction must give dice_loss ~ 0, got {got.item():.6f} (the eps in the "
        "numerator is what makes an empty mask predicted empty score Dice = 1)")
    bad = ns.dice_loss(-perfect, t)
    assert bad.item() > 0.5, f"a fully inverted prediction should be close to 1, got {bad.item():.4f}"


def check_dice_extreme_logits_finite(ns):
    x = ((torch.rand(2, 1, 6, 6, generator=_gen(42)) < 0.5).float() * 2 - 1) * 1e4
    t = (torch.rand(2, 1, 6, 6, generator=_gen(43)) < 0.5).float()
    x.requires_grad_(True)
    loss = ns.dice_loss(x, t)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all(), \
        "dice_loss must stay finite (value and gradient) for logits of size 1e4"
    want = _dice_ref(x.detach(), t, 1.0)
    assert torch.allclose(loss, want, rtol=1e-5, atol=1e-6), \
        f"dice_loss on logits of size 1e4: got {loss.item():.6f}, expected {want.item():.6f}"


# ---------------------------------------------------------------- no built-ins

_BUILTINS = ["cross_entropy", "nll_loss", "binary_cross_entropy", "binary_cross_entropy_with_logits"]


def check_does_not_call_builtin_losses(ns):
    saved = {n: getattr(F, n) for n in _BUILTINS}

    def trap(name):
        def f(*a, **k):
            raise AssertionError(f"called F.{name} — implement the loss yourself "
                                 "(torch.logsumexp / softplus / sigmoid are fine)")
        return f

    x, t = _ce_case(50, (4, 3))
    xb, yb = _focal_case(51)
    xd, td = _dice_case(52)
    try:
        for n in _BUILTINS:
            setattr(F, n, trap(n))
        ns.cross_entropy(x, t, label_smoothing=0.1)
        ns.binary_cross_entropy_with_logits(xb, yb)
        ns.sigmoid_focal_loss(xb, yb)
        ns.dice_loss(xd, td)
    finally:
        for n, f in saved.items():
            setattr(F, n, f)


CHECKS = [
    check_ce_matches_torch_basic,
    check_ce_ignore_index,
    check_ce_class_weights,
    check_ce_label_smoothing,
    check_ce_k_dim_inputs,
    check_ce_extreme_logits,
    check_bce_matches_torch,
    check_bce_reduction_modes,
    check_bce_extreme_logits,
    check_bce_gradient_at_zero_logits,
    check_focal_matches_reference,
    check_focal_negative_alpha_disables_weighting,
    check_focal_gamma0_no_alpha_equals_bce,
    check_focal_extreme_logits,
    check_dice_matches_formula,
    check_dice_perfect_and_empty,
    check_dice_extreme_logits_finite,
    check_does_not_call_builtin_losses,
]
