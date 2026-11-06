"""siglip-loss — the pairwise sigmoid loss of SigLIP (Zhai et al., 2023).

The paper's Algorithm 1, which every check here is measured against:

    t      = exp(t_prime)                       # logit_scale is the log-temperature
    zimg   = l2_normalize(img_emb)              # (N, D)
    ztxt   = l2_normalize(txt_emb)              # (N, D)
    logits = zimg @ ztxt.T * t + b              # (N, N)
    labels = 2 * eye(N) - ones(N)               # +1 on the diagonal, -1 elsewhere
    loss   = -sum(log_sigmoid(labels * logits)) / N     # divide by N, NOT N^2

The oracle is a brute-force double loop over (i, j) in float64 with a scalar,
overflow-free log-sigmoid, so it shares no code path with a vectorized answer.
Gradients are compared against autograd through an independent float64
vectorized version of the same formula (it agrees with the loop to ~1e-12).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ENTRIES = ["siglip_loss", "SigLIPHead"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "There is no softmax over the batch: every (image i, text j) pair is its own "
    "binary classification. The N matching pairs on the diagonal are positives "
    "(z = +1); the other N^2 - N pairs are negatives (z = -1).",
    "L2-normalize both embedding sets along the feature dim, then "
    "logits = exp(logit_scale) * img @ txt.T + logit_bias. Build the +/-1 label "
    "matrix from torch.eye (2*I - 1).",
    "loss = -(sum of log-sigmoid(z * logits)) / N — divide by the batch size N, "
    "not by N^2, so it is not a .mean() over the matrix. Use F.logsigmoid: "
    "log(sigmoid(x)) underflows to -inf for very negative x. SigLIPHead stores "
    "logit_scale = log(10) and logit_bias = -10 as nn.Parameters.",
]

D = 12


def _pairs(n, d=D, seed=0, corr=0.0):
    g = torch.Generator().manual_seed(seed)
    img = torch.randn(n, d, generator=g)
    txt = torch.randn(n, d, generator=g) + corr * img
    # un-normalized on purpose: varied row norms
    img = img * (torch.rand(n, 1, generator=g) * 3 + 0.2)
    txt = txt * (torch.rand(n, 1, generator=g) * 3 + 0.2)
    return img, txt


def _log_sigmoid_scalar(x: float) -> float:
    """log(sigmoid(x)) = -softplus(-x), written so neither branch can overflow."""
    return -(max(-x, 0.0) + math.log1p(math.exp(-abs(x))))


def _reference_loop(img, txt, t_prime, b):
    """Brute force: one log-sigmoid per (i, j) pair, float64, divided by N."""
    img = img.detach().double()
    txt = txt.detach().double()
    n = img.shape[0]
    t = math.exp(float(t_prime))
    total = 0.0
    for i in range(n):
        x = img[i] / img[i].norm()
        for j in range(n):
            y = txt[j] / txt[j].norm()
            z = 1.0 if i == j else -1.0
            logit = t * float(x @ y) + float(b)
            total += -_log_sigmoid_scalar(z * logit)
    return total / n


def _reference_autograd(img, txt, t_prime, b):
    """Vectorized float64 version used only to get reference gradients."""
    zi = F.normalize(img, dim=-1)
    zt = F.normalize(txt, dim=-1)
    logits = zi @ zt.T * t_prime.exp() + b
    n = logits.shape[0]
    labels = 2 * torch.eye(n, dtype=logits.dtype) - 1
    return -F.logsigmoid(labels * logits).sum() / n


def _loss(ns, img, txt, t_prime, b):
    out = ns.siglip_loss(img, txt, torch.tensor(float(t_prime)), torch.tensor(float(b)))
    assert torch.is_tensor(out), \
        f"siglip_loss should return a tensor, got {type(out).__name__}"
    assert out.numel() == 1, \
        f"siglip_loss should return a scalar, got shape {tuple(out.shape)}"
    return float(out.reshape(()))


def check_matches_brute_force_double_loop(ns):
    """Several batch sizes, scales and biases against the (i, j) double loop."""
    cases = [(5, math.log(10.0), -10.0, 0), (8, 0.0, 0.0, 1), (8, math.log(10.0), -10.0, 2),
             (13, 1.7, -2.5, 3), (3, math.log(30.0), 4.0, 4)]
    for n, tp, b, seed in cases:
        img, txt = _pairs(n, seed=seed, corr=1.0)
        got = _loss(ns, img, txt, tp, b)
        exp = _reference_loop(img, txt, tp, b)
        assert abs(got - exp) <= 1e-4 * max(1.0, abs(exp)), (
            f"N={n}, logit_scale={tp:.3f}, logit_bias={b}: got {got:.5f}, the pairwise "
            f"sigmoid loss is {exp:.5f}. Check: L2-normalize both inputs, "
            f"logits = exp(logit_scale) * img @ txt.T + logit_bias, labels +1 on the "
            f"diagonal and -1 elsewhere, loss = -sum(log sigmoid(labels * logits)) / N.")


def check_normalized_by_batch_size_not_pairs(ns):
    """The paper divides the double sum by N (the batch size), not by N^2."""
    n = 8
    img, txt = _pairs(n, seed=11, corr=1.0)
    tp, b = math.log(10.0), -10.0
    per_n = _reference_loop(img, txt, tp, b)
    total = per_n * n
    got = _loss(ns, img, txt, tp, b)
    if abs(got - total / n ** 2) < 1e-4 * max(1.0, total / n ** 2):
        raise AssertionError(
            f"loss = {got:.5f} = sum / N^2 — that is the mean over all N^2 pairs "
            f"(e.g. .mean() or BCE's default reduction). SigLIP divides the double sum "
            f"by the batch size N: expected {per_n:.5f}.")
    if abs(got - total) < 1e-4 * max(1.0, total):
        raise AssertionError(
            f"loss = {got:.5f} is the raw double sum; divide it by the batch size N "
            f"(expected {per_n:.5f}).")
    assert abs(got - per_n) <= 1e-4 * max(1.0, per_n), \
        f"expected sum / N = {per_n:.5f}, got {got:.5f}"


def check_finite_for_huge_logits(ns):
    """exp(logit_scale) = 1000 drives |logits| into the hundreds: log(sigmoid(x))
    underflows to log(0) = -inf there, F.logsigmoid does not."""
    img, txt = _pairs(6, seed=5, corr=0.5)
    img = img.clone().requires_grad_(True)
    tp = torch.tensor(math.log(1000.0), requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    out = ns.siglip_loss(img, txt, tp, b)
    val = float(out.detach().reshape(()))
    assert math.isfinite(val), (
        f"loss is {val} with |logits| ~ 1000 — log(sigmoid(x)) underflows to -inf for "
        f"x below about -89 in float32. Use F.logsigmoid (or -F.softplus(-x)).")
    exp = _reference_loop(img, txt, math.log(1000.0), 0.0)
    assert abs(val - exp) <= 1e-4 * max(1.0, abs(exp)), \
        f"with huge logits expected {exp:.3f}, got {val:.3f}"
    out.reshape(()).backward()
    for name, t in (("img_emb", img), ("logit_scale", tp), ("logit_bias", b)):
        assert t.grad is not None and torch.isfinite(t.grad).all(), \
            f"gradient w.r.t. {name} is missing or non-finite for huge logits"


def check_gradients_match_autograd_reference(ns):
    """d loss / d (logit_scale, logit_bias, embeddings) against float64 autograd."""
    img, txt = _pairs(7, seed=21, corr=1.0)
    tp0, b0 = 1.9, -3.0
    img_s = img.clone().requires_grad_(True)
    txt_s = txt.clone().requires_grad_(True)
    tp_s = torch.tensor(tp0, requires_grad=True)
    b_s = torch.tensor(b0, requires_grad=True)
    out = ns.siglip_loss(img_s, txt_s, tp_s, b_s)
    out.reshape(()).backward()
    for name, t in (("logit_scale", tp_s), ("logit_bias", b_s), ("img_emb", img_s),
                    ("txt_emb", txt_s)):
        assert t.grad is not None, (
            f"no gradient reached {name} — keep it in the graph (no .item(), float() "
            f"or math.exp on a tensor that must be learned)")

    img_r = img.double().requires_grad_(True)
    txt_r = txt.double().requires_grad_(True)
    tp_r = torch.tensor(tp0, dtype=torch.float64, requires_grad=True)
    b_r = torch.tensor(b0, dtype=torch.float64, requires_grad=True)
    _reference_autograd(img_r, txt_r, tp_r, b_r).backward()
    for name, s, r in (("logit_bias", b_s, b_r), ("logit_scale", tp_s, tp_r),
                       ("img_emb", img_s, img_r), ("txt_emb", txt_s, txt_r)):
        diff = (s.grad.double() - r.grad).abs().max().item()
        scale = max(1.0, r.grad.abs().max().item())
        assert diff <= 1e-4 * scale, (
            f"d loss / d {name} is off by {diff:.2e} from autograd of the reference — "
            f"is {name} entering the logits exactly as in "
            f"exp(logit_scale) * cos_sim + logit_bias?")


def check_embeddings_are_l2_normalized(ns):
    """Rescaling any embedding row (by its own factor) must not move the loss."""
    img, txt = _pairs(9, seed=31, corr=1.0)
    g = torch.Generator().manual_seed(3)
    ri = torch.rand(9, 1, generator=g) * 20 + 0.05
    rt = torch.rand(9, 1, generator=g) * 20 + 0.05
    base = _loss(ns, img, txt, math.log(10.0), -10.0)
    scaled = _loss(ns, img * ri, txt * rt, math.log(10.0), -10.0)
    assert abs(base - scaled) <= 1e-4 * max(1.0, abs(base)), (
        f"rescaling embedding rows changed the loss ({base:.5f} -> {scaled:.5f}); "
        f"L2-normalize each embedding along the feature dim (dim=-1) first")


def check_labels_are_plus_minus_one(ns):
    """Orthonormal pairs at t = 1, b = 0.5: cos-sim = I, so every diagonal logit is
    1.5 and every off-diagonal logit 0.5 — a closed form that pins the +1 diagonal
    / -1 off-diagonal labels (0/1 labels or a flipped sign give other numbers)."""
    sp = lambda x: math.log1p(math.exp(x))          # softplus; -log sigmoid(x) = sp(-x)
    for n in (2, 5):
        img = torch.eye(D)[:n] * 2.0
        txt = torch.eye(D)[:n] * 0.5
        got = _loss(ns, img, txt, 0.0, 0.5)
        exp = (n * sp(-1.5) + (n * n - n) * sp(0.5)) / n
        assert abs(got - exp) <= 1e-4, (
            f"N={n}, cos-sim = I, t=1, b=0.5 (diagonal logits 1.5, off-diagonal 0.5): "
            f"expected {exp:.5f}, got {got:.5f}. Positives (diagonal) get z=+1, every "
            f"other pair z=-1, and the loss term is -log sigmoid(z * logit).")


def _head_params(head):
    for name in ("logit_scale", "logit_bias"):
        p = getattr(head, name, None)
        assert isinstance(p, nn.Parameter), (
            f"SigLIPHead.{name} should be an nn.Parameter "
            f"(got {type(p).__name__})")
        assert p.numel() == 1, f"SigLIPHead.{name} should hold one scalar, has {p.numel()}"
        assert p.requires_grad, f"SigLIPHead.{name} must be learnable (requires_grad)"
    return head.logit_scale, head.logit_bias


def check_head_init_values(ns):
    head = ns.SigLIPHead()
    s, b = _head_params(head)
    assert abs(float(s.detach().reshape(())) - math.log(10.0)) < 1e-5, (
        f"default logit_scale should be log(10) = {math.log(10.0):.5f} (so the "
        f"temperature exp(logit_scale) starts at 10), got {float(s.detach().reshape(())):.5f}")
    assert abs(float(b.detach().reshape(())) + 10.0) < 1e-5, (
        f"default logit_bias should be -10 (the paper's init), got "
        f"{float(b.detach().reshape(())):.5f}")
    head = ns.SigLIPHead(init_scale=math.log(5.0), init_bias=-2.0)
    s, b = _head_params(head)
    assert abs(float(s.detach().reshape(())) - math.log(5.0)) < 1e-5 and \
        abs(float(b.detach().reshape(())) + 2.0) < 1e-5, \
        "SigLIPHead(init_scale, init_bias) must store those values as given"


def check_head_forward_uses_its_parameters(ns):
    head = ns.SigLIPHead()
    s, b = _head_params(head)
    with torch.no_grad():
        s.fill_(1.2)
        b.fill_(-1.5)
    img, txt = _pairs(6, seed=41, corr=1.0)
    out = head(img, txt)
    assert torch.is_tensor(out) and out.numel() == 1, \
        "SigLIPHead(img_emb, txt_emb) should return the scalar loss"
    exp = _reference_loop(img, txt, 1.2, -1.5)
    assert abs(float(out.detach().reshape(())) - exp) <= 1e-4 * max(1.0, abs(exp)), (
        f"SigLIPHead forward gave {float(out.detach().reshape(())):.5f}, expected {exp:.5f} for its "
        f"current logit_scale=1.2, logit_bias=-1.5 — read the parameters at call time")
    out.reshape(()).backward()
    for name, p in (("logit_scale", s), ("logit_bias", b)):
        assert p.grad is not None and p.grad.abs().sum() > 0, \
            f"no gradient reached SigLIPHead.{name}"


CHECKS = [
    check_matches_brute_force_double_loop,
    check_normalized_by_batch_size_not_pairs,
    check_labels_are_plus_minus_one,
    check_finite_for_huge_logits,
    check_gradients_match_autograd_reference,
    check_embeddings_are_l2_normalized,
    check_head_init_values,
    check_head_forward_uses_its_parameters,
]
