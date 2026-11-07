"""self-supervised-losses — SimCLR's NT-Xent, DINO's centred + sharpened self-distillation
loss with multi-crop, and the in-place EMA ("momentum") teacher update.

Conventions graded here, with their sources:

* nt_xent_loss(z1, z2, temperature) — SimCLR (Chen et al., "A Simple Framework for
  Contrastive Learning of Visual Representations", ICML 2020), eq. (1) and the loss line of
  Algorithm 1:
      l(i, j) = -log  exp(sim(z_i, z_j) / tau) / sum_{k = 1..2N, k != i} exp(sim(z_i, z_k) / tau)
      L       = 1/(2N) * sum_{k = 1..N} [ l(2k-1, 2k) + l(2k, 2k-1) ]
  with sim(u, v) = u.v / (|u| |v|). Here the 2N views are the rows of cat([z1, z2]) (the paper
  interleaves them; the loss does not depend on the order); row i of z1 and row i of z2 are a
  positive pair. The anchor itself is excluded (k != i), so each anchor's softmax runs over
  2N - 1 candidates: 1 positive + 2(N - 1) negatives. The loss is the MEAN over all 2N anchors.
  `temperature` is tau and DIVIDES the cosine (the v3 contrastive-loss-clip problem passes a
  multiplier instead). The official TF code (google-research/simclr, objective.py,
  add_contrastive_loss) returns loss_a + loss_b, the SUM of the means over the anchors of each
  view half, i.e. 2x the paper's average; this problem grades the paper's average.

* DINOLoss — DINO (Caron et al., "Emerging Properties in Self-Supervised Vision Transformers",
  ICCV 2021), Algorithm 1, and class DINOLoss in facebookresearch/dino, main_dino.py:
      P_t = softmax((t - C) / tau_t)  (centre, then sharpen; no gradient)
      log P_s = log_softmax(s / tau_s)
      loss = [ sum over pairs (iq in {0, 1}, v in 0..ncrops-1, v != iq) of
               mean over the batch of  -sum_k P_t[k] log P_s[k] ] / n_loss_terms,
      n_loss_terms = 2 * (ncrops - 1).
  Layout as in the official code (MultiCropWrapper concatenates per-crop outputs):
  student_output is (ncrops * B, K), crop v in rows [v*B, (v+1)*B), crops 0 and 1 being the two
  global crops; teacher_output is (2 * B, K), the same two global crops in the same order. After
  the loss, without gradient: C <- m*C + (1 - m) * teacher_output.mean(0) (raw teacher logits,
  all 2B rows; the official all_reduce is a no-op on one process). `center` is a registered
  buffer of shape (1, K), zeros at init. The official teacher-temperature warmup schedule (an
  `epoch` argument) is left out: tau_t is fixed.

* ema_update_(teacher, student, momentum) — theta_t <- m * theta_t + (1 - m) * theta_s, in place,
  outside autograd, PARAMETERS ONLY, as in main_dino.py's training loop
  (`param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)` over .parameters()). Buffers
  (e.g. BatchNorm running stats) are left as they are; DINO never calls teacher.eval(), so a BN
  teacher keeps its own batch statistics. MoCo v3's builder also averages .parameters() only, but
  with `param.data = ...`, which allocates new storage; this problem requires the existing storage
  to be updated in place (the trailing underscore).

Oracles below are brute-force loops in float64, written independently of any vectorised form.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchleet.runner import Skip

ENTRIES = ["nt_xent_loss", "DINOLoss", "ema_update_"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "NT-Xent: stack both views into one (2N, D) batch so every view is an anchor and every other "
    "view is a candidate; the positive of view i is the other augmentation of the same image. "
    "DINO: the teacher only sees the 2 global crops, the student sees all crops, and a crop is "
    "never compared with itself. The EMA update is not part of training: keep autograd out of it.",
    "NT-Xent: L2-normalise, build the (2N, 2N) cosine matrix divided by tau, and remove the "
    "diagonal from the softmax by setting it to -inf (not 0). Then it is cross-entropy with target "
    "i + N for i < N and i - N otherwise, averaged over 2N rows. DINO: teacher probabilities = "
    "softmax((t - center) / teacher_temp) computed on detached outputs; student log-probabilities = "
    "log_softmax(s / student_temp); the center update uses the TEACHER outputs and happens after "
    "the loss.",
    "DINO: chunk the student output into ncrops blocks and the teacher's into 2; for each teacher "
    "block iq and student block v with v != iq, add the batch mean of the cross-entropy "
    "-sum_k q_k log p_k; divide by the number of terms, 2 * (ncrops - 1). Then, under no_grad, "
    "move the center: m times the old center plus (1 - m) times the mean of all 2B teacher rows. "
    "EMA: under no_grad, scale each teacher parameter by m in place and add (1 - m) times the "
    "matching student parameter; leave buffers alone.",
]


# ---------------------------------------------------------------- helpers

def _scalar(x, what):
    assert torch.is_tensor(x), f"{what} must return a tensor, got {type(x).__name__}"
    assert x.dim() == 0, f"{what} must return a 0-dim (scalar) tensor, got shape {tuple(x.shape)}"
    return x


def _count_torch_calls(fn, *args):
    try:
        from torch.overrides import TorchFunctionMode
    except ImportError:  # very old torch
        raise Skip("needs torch.overrides.TorchFunctionMode (torch >= 2.0)")

    class _Counter(TorchFunctionMode):
        def __init__(self):
            super().__init__()
            self.n = 0

        def __torch_function__(self, func, types, args=(), kwargs=None):
            self.n += 1
            return func(*args, **(kwargs or {}))

    with _Counter() as c:
        fn(*args)
    return c.n


# ---------------------------------------------------------------- NT-Xent oracles

def _nt_xent_brute(z1, z2, tau):
    """Paper eq. (1), one anchor and one candidate at a time, float64. Differentiable."""
    z = torch.cat([z1, z2]).double()
    z = z / z.norm(dim=1, keepdim=True)
    n2 = z.shape[0]
    n = n2 // 2
    total = 0.0
    for i in range(n2):
        j = (i + n) % n2                                   # the other view of the same image
        cands = torch.stack([(z[i] * z[k]).sum() / tau for k in range(n2) if k != i])
        total = total + torch.logsumexp(cands, 0) - (z[i] * z[j]).sum() / tau
    return total / n2


def _nt_xent_variants(z1, z2, tau):
    """Values of the classic wrong implementations, used only to explain a failure."""
    z1, z2 = z1.double(), z2.double()
    n = z1.shape[0]
    u = F.normalize(torch.cat([z1, z2]), dim=1)
    s = u @ u.T / tau
    tgt = torch.cat([torch.arange(n) + n, torch.arange(n)])
    eye = torch.eye(2 * n, dtype=torch.bool)
    good = F.cross_entropy(s.masked_fill(eye, float("-inf")), tgt)
    a, b = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    clip_like = 0.5 * (F.cross_entropy(a @ b.T / tau, torch.arange(n))
                       + F.cross_entropy(b @ a.T / tau, torch.arange(n)))
    raw = torch.cat([z1, z2])
    return {
        "the anchor's own similarity left in the denominator (diagonal not masked)":
            F.cross_entropy(s, tgt),
        "the diagonal set to 0 instead of -inf (exp(0) = 1 still sits in every denominator)":
            F.cross_entropy(s.masked_fill(eye, 0.0), tgt),
        "only the N anchors of z1 (one direction) instead of all 2N views":
            F.cross_entropy(s.masked_fill(eye, float("-inf"))[:n], tgt[:n]),
        "CLIP's N x N cross-view loss: the 2(N-1) negatives must include the other views in "
        "the SAME view batch as well":
            clip_like,
        "no L2 normalisation (dot products instead of cosines)":
            F.cross_entropy((raw @ raw.T / tau).masked_fill(eye, float("-inf")), tgt),
        "the temperature multiplying the cosines instead of dividing them":
            F.cross_entropy((u @ u.T * tau).masked_fill(eye, float("-inf")), tgt),
        "a sum over the 2N anchors instead of the mean": good * 2 * n,
        "the official TF code's loss_a + loss_b (2x the paper's average); this problem uses the "
        "paper's mean over 2N anchors": good * 2,
    }


def _explain_nt_xent(got, z1, z2, tau):
    for what, val in _nt_xent_variants(z1, z2, tau).items():
        if abs(got - float(val)) <= 1e-6 * max(1.0, abs(float(val))):
            return f" — that is the value you get with {what}"
    return ""


def _views(seed, n, d, noise=0.3, dtype=torch.float64):
    """Two noisy 'augmentations' of n random images, with very different row norms."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(n, d, generator=g, dtype=dtype)
    z1 = (base + noise * torch.randn(n, d, generator=g, dtype=dtype)) \
        * (torch.rand(n, 1, generator=g, dtype=dtype) * 5 + 0.2)
    z2 = (base + noise * torch.randn(n, d, generator=g, dtype=dtype)) \
        * (torch.rand(n, 1, generator=g, dtype=dtype) * 5 + 0.2)
    return z1, z2


# ---------------------------------------------------------------- NT-Xent checks

def check_nt_xent_matches_brute_force(ns):
    for seed, (n, d, tau) in enumerate([(4, 8, 0.5), (7, 3, 0.1), (16, 32, 0.07), (2, 5, 1.0)]):
        z1, z2 = _views(seed, n, d)
        z1_before, z2_before = z1.clone(), z2.clone()
        got = _scalar(ns.nt_xent_loss(z1, z2, tau), "nt_xent_loss")
        assert torch.equal(z1, z1_before) and torch.equal(z2, z2_before), \
            "nt_xent_loss modified its inputs in place"
        want = float(_nt_xent_brute(z1, z2, tau))
        g = float(got)
        assert math.isclose(g, want, rel_tol=1e-5, abs_tol=1e-5), (
            f"N={n}, D={d}, tau={tau}: got {g:.6f}, expected {want:.6f} (SimCLR eq. 1, mean over "
            f"all 2N anchors){_explain_nt_xent(g, z1, z2, tau)}")


def check_nt_xent_gradient_matches_brute_force(ns):
    z1, z2 = _views(11, 6, 5)
    a = z1.clone().requires_grad_()
    b = z2.clone().requires_grad_()
    out = ns.nt_xent_loss(a, b, 0.2)
    assert torch.is_tensor(out) and out.requires_grad, \
        "nt_xent_loss must be differentiable (no .item(), .tolist(), torch.tensor(...) re-wrapping)"
    out.backward()
    ra = z1.clone().requires_grad_()
    rb = z2.clone().requires_grad_()
    _nt_xent_brute(ra, rb, 0.2).backward()
    for name, got, want in (("z1", a.grad, ra.grad), ("z2", b.grad, rb.grad)):
        assert got is not None, f"no gradient reached {name}"
        assert torch.isfinite(got).all(), \
            f"non-finite gradient w.r.t. {name} — e.g. an in-place edit of a tensor autograd needs"
        assert torch.allclose(got, want, rtol=1e-5, atol=1e-6), (
            f"gradient w.r.t. {name} differs from the brute-force loss's gradient (max abs diff "
            f"{(got - want).abs().max():.2e})")


def check_nt_xent_symmetric_in_views_and_batch_order(ns):
    z1, z2 = _views(3, 9, 7)
    base = float(ns.nt_xent_loss(z1, z2, 0.3))
    swapped = float(ns.nt_xent_loss(z2, z1, 0.3))
    assert math.isclose(base, swapped, rel_tol=1e-6, abs_tol=1e-7), (
        f"nt_xent_loss(z1, z2) = {base:.6f} but nt_xent_loss(z2, z1) = {swapped:.6f}: all 2N views "
        f"are anchors, so swapping the two view batches cannot change the loss")
    perm = torch.randperm(9, generator=torch.Generator().manual_seed(0))
    permuted = float(ns.nt_xent_loss(z1[perm], z2[perm], 0.3))
    assert math.isclose(base, permuted, rel_tol=1e-6, abs_tol=1e-7), \
        "shuffling the batch (the same permutation for both views) changed the loss"


def check_nt_xent_invariant_to_embedding_scale(ns):
    z1, z2 = _views(4, 8, 6)
    g = torch.Generator().manual_seed(1)
    s1 = torch.rand(8, 1, generator=g, dtype=torch.float64) * 100 + 0.01
    s2 = torch.rand(8, 1, generator=g, dtype=torch.float64) * 100 + 0.01
    a = float(ns.nt_xent_loss(z1, z2, 0.5))
    b = float(ns.nt_xent_loss(z1 * s1, z2 * s2, 0.5))
    assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7), (
        f"rescaling the rows of z1 and z2 changed the loss ({a:.6f} -> {b:.6f}): sim() is the "
        f"COSINE similarity, so L2-normalise both views (dim=1) first")


def check_nt_xent_self_similarity_excluded(ns):
    # N = 1: the only candidate left for each anchor is its positive, so the loss is exactly 0.
    one = torch.tensor([[1.0, 2.0, -1.0]], dtype=torch.float64)
    got = float(ns.nt_xent_loss(one, torch.tensor([[0.5, -1.0, 2.0]], dtype=torch.float64), 0.5))
    assert abs(got) < 1e-6, (
        f"with N = 1 each anchor has 2N - 1 = 1 candidate (its positive) once the anchor itself is "
        f"excluded, so the loss must be 0; got {got:.6f}")
    # Identical views of mutually orthogonal images at a low temperature: the positive has cosine
    # 1 and every negative cosine 0, so the loss is log(1 + 2(N-1) e^{-1/tau}) ~ 0. Keeping the
    # anchor's own cosine (also 1) in the denominator would give ~log 2 instead.
    n, tau = 5, 0.05
    z = torch.eye(8, dtype=torch.float64)[:n] * torch.arange(1, n + 1, dtype=torch.float64)[:, None]
    got = float(ns.nt_xent_loss(z, z.clone(), tau))
    want = math.log1p(2 * (n - 1) * math.exp(-1 / tau))
    assert math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-6), (
        f"two identical views of {n} orthogonal images at tau={tau} should give loss "
        f"log(1 + 2(N-1)e^(-1/tau)) = {want:.3g}; got {got:.3g}"
        + (" ~ log 2: the anchor's similarity with itself is still in the softmax"
           if abs(got - math.log(2)) < 0.01 else ""))


def check_nt_xent_temperature_limits_and_stability(ns):
    z1, z2 = _views(5, 6, 4)
    # tau -> infinity: every logit -> 0, each anchor is uniform over its 2N - 1 candidates.
    hot = float(ns.nt_xent_loss(z1, z2, 1e6))
    assert math.isclose(hot, math.log(2 * 6 - 1), rel_tol=1e-5), (
        f"at a huge temperature the loss must approach log(2N - 1) = {math.log(11):.4f} (2N - 1 "
        f"candidates per anchor); got {hot:.4f}"
        + (" = log(2N): the anchor itself is still a candidate" if abs(hot - math.log(12)) < 1e-3 else ""))
    # tau = 0.005 in float32: logits up to 200, exp() overflows -- the loss must stay finite.
    for tau in (0.01, 0.005):
        a, b = z1.float(), z2.float()
        got = ns.nt_xent_loss(a, b, tau)
        assert torch.isfinite(got), (
            f"loss is {float(got)} for float32 inputs at tau={tau} (logits up to {1 / tau:.0f}): use "
            f"F.cross_entropy / logsumexp on masked logits rather than exp() and a division")
        want = float(_nt_xent_brute(z1, z2, tau))
        assert math.isclose(float(got), want, rel_tol=1e-4, abs_tol=1e-3), \
            f"float32, tau={tau}: got {float(got):.5f}, expected {want:.5f}"


def check_nt_xent_is_vectorised(ns):
    small = _views(6, 4, 8)
    large = _views(6, 64, 8)
    n_small = _count_torch_calls(ns.nt_xent_loss, *small, 0.5)
    n_large = _count_torch_calls(ns.nt_xent_loss, *large, 0.5)
    assert n_large <= n_small + 8, (
        f"nt_xent_loss makes {n_small} torch calls for N=4 but {n_large} for N=64 — loop-free "
        f"please: one (2N, 2N) similarity matmul, a diagonal mask and one cross-entropy")


# ---------------------------------------------------------------- DINO oracles

def _dino_brute(s, t, center, ncrops, tau_s, tau_t):
    """DINO loss one (teacher crop, student crop, sample) term at a time, float64."""
    s, t, center = s.double(), t.double(), center.double().reshape(-1)
    b = t.shape[0] // 2
    total, n = 0.0, 0
    for iq in range(2):
        for v in range(ncrops):
            if v == iq:
                continue                                   # same view: skipped
            for i in range(b):
                q = torch.softmax((t[iq * b + i] - center) / tau_t, 0)
                log_p = torch.log_softmax(s[v * b + i] / tau_s, 0)
                total = total - (q * log_p).sum()
                n += 1
    return total / n


def _dino_inputs(seed, ncrops, b, k, dtype=torch.float64, s_scale=1.0, t_scale=1.0):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(ncrops * b, k, generator=g, dtype=dtype) * s_scale
    # a per-dimension offset in the teacher logits makes the center matter
    bias = torch.randn(k, generator=g, dtype=dtype) * 0.5
    t = torch.randn(2 * b, k, generator=g, dtype=dtype) * 0.1 * t_scale + bias * t_scale
    c0 = torch.randn(1, k, generator=g, dtype=dtype) * 0.1
    return s, t, c0


def _make_dino(ns, k, ncrops, tau_s, tau_t, m, dtype=torch.float64):
    try:
        mod = ns.DINOLoss(k, ncrops, student_temp=tau_s, teacher_temp=tau_t, center_momentum=m)
    except TypeError as e:
        raise AssertionError(
            "construct as DINOLoss(out_dim, ncrops, student_temp=0.1, teacher_temp=0.04, "
            f"center_momentum=0.9) ({e})")
    assert isinstance(mod, nn.Module), "DINOLoss must be an nn.Module"
    return mod.to(dtype)


def _center_of(mod):
    bufs = dict(mod.named_buffers())
    assert "center" in bufs, (
        "DINOLoss needs a registered buffer named `center` (self.register_buffer('center', "
        "torch.zeros(1, out_dim))) — a buffer is saved in state_dict and moves with .to(device)")
    return bufs["center"]


def _set_center(mod, value):
    with torch.no_grad():
        _center_of(mod).copy_(value.reshape(_center_of(mod).shape))


def _explain_dino(got, s, t, c0, ncrops, ts, tt, m):
    b = t.shape[0] // 2
    new_c = m * c0 + (1 - m) * t.mean(0, keepdim=True)
    cands = {
        "the center updated BEFORE the loss (the loss must use the old center; update it "
        "afterwards)": _dino_brute(s, t, new_c, ncrops, ts, tt),
        "no centering of the teacher": _dino_brute(s, t, torch.zeros_like(c0), ncrops, ts, tt),
        "the two temperatures swapped": _dino_brute(s, t, c0, ncrops, tt, ts),
    }
    # same-view pairs included
    tot = 0.0
    for iq in range(2):
        for v in range(ncrops):
            q = torch.softmax((t[iq * b:(iq + 1) * b] - c0) / tt, -1)
            tot = tot + (-(q * torch.log_softmax(s[v * b:(v + 1) * b] / ts, -1)).sum(-1)).mean()
    cands["same-view pairs (teacher crop iq vs student crop iq) included"] = tot / (2 * ncrops)
    cands["the right terms divided by 2 * ncrops instead of the number of terms 2 * (ncrops - 1)"] = \
        _dino_brute(s, t, c0, ncrops, ts, tt) * (2 * (ncrops - 1)) / (2 * ncrops)
    # centering applied to the student instead of the teacher
    cands["the center subtracted from the student instead of the teacher"] = \
        _dino_brute(s - c0, t, torch.zeros_like(c0), ncrops, ts, tt)
    for what, val in cands.items():
        if abs(got - float(val)) <= 1e-6 * max(1.0, abs(float(val))):
            return f" — that is the value you get with {what}"
    return ""


# ---------------------------------------------------------------- DINO checks

def check_dino_center_is_a_zero_buffer(ns):
    mod = _make_dino(ns, 12, 4, 0.1, 0.04, 0.9, dtype=torch.float32)
    c = _center_of(mod)
    assert tuple(c.shape) in ((1, 12), (12,)), \
        f"center must have shape (1, out_dim) = (1, 12), got {tuple(c.shape)}"
    assert torch.count_nonzero(c) == 0, "center must start at zeros"
    assert not isinstance(c, nn.Parameter) and not c.requires_grad, \
        "center is a statistic, not a learned weight: register it as a buffer, not a Parameter"
    assert not list(mod.parameters()), "DINOLoss has no learnable parameters"


def check_dino_loss_matches_brute_force(ns):
    for seed, (ncrops, b, k) in enumerate([(2, 4, 8), (3, 5, 6), (6, 3, 16)]):
        ts, tt, m = 0.1, 0.04, 0.9
        s, t, c0 = _dino_inputs(seed, ncrops, b, k)
        mod = _make_dino(ns, k, ncrops, ts, tt, m)
        _set_center(mod, c0)
        got = _scalar(mod(s, t), "DINOLoss.forward")
        want = float(_dino_brute(s, t, c0, ncrops, ts, tt))
        g = float(got)
        assert math.isclose(g, want, rel_tol=1e-5, abs_tol=1e-8), (
            f"2 global + {ncrops - 2} local crops, batch {b}: got {g:.6f}, expected {want:.6f} "
            f"(mean over the {2 * (ncrops - 1)} (teacher global crop, other student crop) pairs and "
            f"the batch){_explain_dino(g, s, t, c0, ncrops, ts, tt, m)}")


def check_dino_center_update_after_loss(ns):
    ts, tt, m, ncrops, b, k = 0.1, 0.04, 0.8, 4, 5, 6
    s, t, c0 = _dino_inputs(7, ncrops, b, k)
    mod = _make_dino(ns, k, ncrops, ts, tt, m)
    _set_center(mod, c0)
    mod(s, t.clone().requires_grad_())
    c1 = _center_of(mod)
    want = m * c0 + (1 - m) * t.mean(0, keepdim=True)
    msg = ""
    if torch.allclose(c1.reshape(1, -1), m * c0 + (1 - m) * s.mean(0, keepdim=True), atol=1e-9):
        msg = " — you averaged the STUDENT outputs; the center tracks the teacher's"
    elif torch.allclose(c1.reshape(1, -1), (1 - m) * c0 + m * t.mean(0, keepdim=True), atol=1e-9):
        msg = " — momentum m must weight the OLD center: c <- m*c + (1-m)*batch_mean"
    elif torch.allclose(c1.reshape(1, -1), c0, atol=1e-12):
        msg = " — the center did not change; update the buffer at the end of forward()"
    assert torch.allclose(c1.reshape(1, -1), want, rtol=1e-6, atol=1e-7), (
        f"after one call center should be m*c + (1-m)*teacher_output.mean(0) (over all 2B "
        f"teacher rows){msg}")
    assert c1.grad_fn is None and not c1.requires_grad, \
        "the center update must happen under torch.no_grad() (center has an autograd history)"
    # The next call must use the updated center: the buffer is the module's state.
    s2, t2, _ = _dino_inputs(8, ncrops, b, k)
    got = float(mod(s2, t2))
    want2 = float(_dino_brute(s2, t2, want, ncrops, ts, tt))
    assert math.isclose(got, want2, rel_tol=1e-5, abs_tol=1e-8), (
        f"second call: got {got:.6f}, expected {want2:.6f} with the center updated by the first "
        f"call — keep the center in the buffer between calls")


def check_dino_no_gradient_reaches_teacher(ns):
    ts, tt, m, ncrops, b, k = 0.1, 0.04, 0.9, 4, 3, 7
    s, t, c0 = _dino_inputs(9, ncrops, b, k)
    s_ = s.clone().requires_grad_()
    t_ = t.clone().requires_grad_()
    mod = _make_dino(ns, k, ncrops, ts, tt, m)
    _set_center(mod, c0)
    mod(s_, t_).backward()
    assert t_.grad is None or torch.count_nonzero(t_.grad) == 0, (
        "gradient flowed into teacher_output: the teacher is a stop-gradient target — detach it "
        "(or compute its softmax under torch.no_grad())")
    rs = s.clone().requires_grad_()
    _dino_brute(rs, t, c0, ncrops, ts, tt).backward()
    assert s_.grad is not None, "no gradient reached student_output"
    assert torch.allclose(s_.grad, rs.grad, rtol=1e-5, atol=1e-6), (
        f"gradient w.r.t. student_output differs from the brute-force loss's gradient (max abs "
        f"diff {(s_.grad - rs.grad).abs().max():.2e})")


def check_dino_centering_is_on_teacher_logits_before_softmax(ns):
    # Centering subtracts C from the teacher LOGITS before the softmax, so shifting the teacher
    # outputs and the center by the same vector leaves the loss unchanged ...
    ts, tt, m, ncrops, b, k = 0.1, 0.05, 0.9, 3, 4, 5
    s, t, c0 = _dino_inputs(10, ncrops, b, k)
    shift = torch.linspace(-2, 3, k, dtype=torch.float64)[None]
    losses = []
    for tt_out, c in ((t, c0), (t + shift, c0 + shift)):
        mod = _make_dino(ns, k, ncrops, ts, tt, m)
        _set_center(mod, c)
        losses.append(float(mod(s, tt_out)))
    assert math.isclose(losses[0], losses[1], rel_tol=1e-6, abs_tol=1e-6), (
        "shifting teacher_output and center by the same vector changed the loss: the center must "
        "be subtracted from the teacher logits, before the sharpened softmax")
    # ... and the center must actually change the result.
    mod = _make_dino(ns, k, ncrops, ts, tt, m)
    no_center = float(mod(s, t))
    assert abs(no_center - losses[0]) > 1e-4, \
        "the center had no effect on the loss: subtract it from the teacher logits"


def check_dino_stable_for_large_logits(ns):
    ts, tt, m, ncrops, b, k = 0.1, 0.04, 0.9, 4, 3, 10
    s, t, c0 = _dino_inputs(12, ncrops, b, k, s_scale=60.0, t_scale=8.0)
    mod = _make_dino(ns, k, ncrops, ts, tt, m, dtype=torch.float32)
    _set_center(mod, c0.float())
    s32 = s.float().requires_grad_()
    out = mod(s32, t.float())
    assert torch.is_tensor(out) and torch.isfinite(out), (
        f"loss is {float(out.detach())} for student logits ~ +-60 (so +-600 after / 0.1): use "
        f"F.log_softmax, not torch.log(F.softmax(...))")
    out.backward()
    assert torch.isfinite(s32.grad).all(), "non-finite gradient for large student logits"
    want = float(_dino_brute(s, t, c0, ncrops, ts, tt))
    got = float(out.detach())
    assert math.isclose(got, want, rel_tol=1e-4, abs_tol=1e-2), \
        f"large logits: got {got:.4f}, expected {want:.4f}"


def check_dino_is_vectorised_over_the_batch(ns):
    def run(b):
        s, t, _ = _dino_inputs(13, 4, b, 6, dtype=torch.float32)
        mod = _make_dino(ns, 6, 4, 0.1, 0.04, 0.9, dtype=torch.float32)
        return _count_torch_calls(mod, s, t)
    n_small, n_large = run(2), run(32)
    assert n_large <= n_small + 8, (
        f"DINOLoss makes {n_small} torch calls for batch 2 but {n_large} for batch 32 — a loop "
        f"over the crop pairs is fine, a loop over the samples is not")


# ---------------------------------------------------------------- EMA

class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 6)
        self.bn = nn.BatchNorm1d(6)
        self.fc2 = nn.Linear(6, 3)

    def forward(self, x):
        return self.fc2(torch.relu(self.bn(self.fc1(x))))


def _pair(seed=0, teacher_requires_grad=False):
    torch.manual_seed(seed)
    student = _Net().double()
    teacher = _Net().double()
    teacher.load_state_dict(student.state_dict())
    with torch.no_grad():
        for p in student.parameters():
            p.add_(torch.randn_like(p))                 # the student has moved on
    student.train()
    student(torch.randn(16, 4, dtype=torch.float64) * 3 + 1)   # and its BN stats too
    for p in teacher.parameters():
        p.requires_grad_(teacher_requires_grad)
    return teacher, student


def check_ema_update_values(ns):
    for m in (0.9, 0.996, 0.0, 1.0):
        teacher, student = _pair(0)
        t0 = {n: p.detach().clone() for n, p in teacher.named_parameters()}
        s0 = {n: p.detach().clone() for n, p in student.named_parameters()}
        ns.ema_update_(teacher, student, m)
        for n, p in teacher.named_parameters():
            want = m * t0[n] + (1 - m) * s0[n]
            if not torch.allclose(p, want, rtol=1e-12, atol=1e-12):
                swapped = (1 - m) * t0[n] + m * s0[n]
                hint = (" — the momentum must weight the TEACHER: t <- m*t + (1-m)*s"
                        if torch.allclose(p, swapped, atol=1e-12) else "")
                raise AssertionError(
                    f"momentum={m}: teacher.{n} should be m*teacher + (1-m)*student{hint}")
        for n, p in student.named_parameters():
            assert torch.equal(p, s0[n]), f"ema_update_ changed the student's {n}"
    # k updates in a row = the closed form m^k t0 + (1 - m^k) s   (student fixed)
    teacher, student = _pair(1)
    t0 = {n: p.detach().clone() for n, p in teacher.named_parameters()}
    for _ in range(5):
        ns.ema_update_(teacher, student, 0.7)
    for (n, p), q in zip(teacher.named_parameters(), student.parameters()):
        want = 0.7 ** 5 * t0[n] + (1 - 0.7 ** 5) * q.detach()
        assert torch.allclose(p, want, rtol=1e-10, atol=1e-12), \
            f"five updates with m=0.7 did not compound to m^5*t0 + (1-m^5)*s for {n}"


def check_ema_update_is_in_place_without_autograd(ns):
    for teacher_rg in (False, True):
        teacher, student = _pair(2, teacher_requires_grad=teacher_rg)
        student(torch.randn(8, 4, dtype=torch.float64)).sum().backward()   # student has .grad
        grads = {n: p.grad.clone() for n, p in student.named_parameters()}
        ids = {n: (id(p), p.data_ptr()) for n, p in teacher.named_parameters()}
        try:
            ns.ema_update_(teacher, student, 0.99)
        except RuntimeError as e:
            raise AssertionError(
                f"ema_update_ raised with teacher.requires_grad={teacher_rg} ({e}) — do the "
                f"update under torch.no_grad()")
        for n, p in teacher.named_parameters():
            assert id(p) == ids[n][0], (
                f"teacher.{n} was replaced by a new Parameter: update the existing one in place "
                f"(p.mul_(m).add_(...)) so everything holding it still sees the update")
            assert p.data_ptr() == ids[n][1], (
                f"teacher.{n} now points to new storage: `p.data = m * p.data + ...` allocates a "
                f"fresh tensor every step (MoCo v3's reference code does this). ema_update_ promises "
                f"an in-place update of the existing storage: p.mul_(m).add_(...) under no_grad")
            assert p.grad_fn is None and p.is_leaf, (
                f"teacher.{n} now has an autograd history ({type(p.grad_fn).__name__}): run the "
                f"update under torch.no_grad()")
            assert p.requires_grad == teacher_rg, \
                f"ema_update_ changed teacher.{n}.requires_grad"
            assert p.grad is None, f"ema_update_ wrote a .grad into teacher.{n}"
        for n, p in student.named_parameters():
            assert torch.equal(p.grad, grads[n]), f"ema_update_ changed student.{n}.grad"


def check_ema_update_leaves_buffers_alone(ns):
    teacher, student = _pair(3)
    before = {n: b.clone() for n, b in teacher.named_buffers()}
    assert not torch.equal(before["bn.running_mean"], dict(student.named_buffers())["bn.running_mean"])
    ns.ema_update_(teacher, student, 0.9)
    for n, b in teacher.named_buffers():
        assert torch.equal(b, before[n]), (
            f"teacher buffer {n} changed: update PARAMETERS only, as DINO's training loop does — "
            f"running statistics belong to the teacher's own forward passes")


CHECKS = [
    check_nt_xent_matches_brute_force,
    check_nt_xent_gradient_matches_brute_force,
    check_nt_xent_symmetric_in_views_and_batch_order,
    check_nt_xent_invariant_to_embedding_scale,
    check_nt_xent_self_similarity_excluded,
    check_nt_xent_temperature_limits_and_stability,
    check_nt_xent_is_vectorised,
    check_dino_center_is_a_zero_buffer,
    check_dino_loss_matches_brute_force,
    check_dino_center_update_after_loss,
    check_dino_no_gradient_reaches_teacher,
    check_dino_centering_is_on_teacher_logits_before_softmax,
    check_dino_stable_for_large_logits,
    check_dino_is_vectorised_over_the_batch,
    check_ema_update_values,
    check_ema_update_is_in_place_without_autograd,
    check_ema_update_leaves_buffers_alone,
]
