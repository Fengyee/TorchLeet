"""knowledge-distillation — soft-target loss (Hinton et al., 2015).

The soft term is pinned through its gradient with respect to the student logits,
T * (softmax(s/T) - softmax(t/T)) / batch, which is the same for
T^2 * KL(teacher_T || student_T) and for Hinton's soft-target cross-entropy (the
two differ only by the teacher's entropy, a constant). That gradient catches a
missing T^2, reduction='mean' instead of 'batchmean', a reversed KL and an
unsoftened teacher, without preferring one of the two valid formulations.
"""
import torch
import torch.nn.functional as F

ENTRIES = ["distillation_loss"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Two terms: a KL divergence between the temperature-softened teacher and student "
    "distributions, and ordinary cross-entropy against the hard labels; alpha blends "
    "them as alpha * soft + (1 - alpha) * hard.",
    "The soft term is KL(teacher_T || student_T): F.kl_div takes the student's "
    "log_softmax(s / T) as its input and the teacher's softmax(t / T) as its target, "
    "with reduction='batchmean' ('mean' would also divide by the number of classes).",
    "Multiply the soft term by T**2: its gradients scale like 1/T^2, so without the "
    "factor the soft targets fade as T grows. The hard-label cross-entropy uses the "
    "raw student logits (T = 1).",
]

B, C = 8, 5


def _batch(seed=0):
    torch.manual_seed(seed)
    return (torch.randn(B, C), torch.randn(B, C) * 2, torch.randint(0, C, (B,)))


def check_returns_scalar(ns):
    s, t, y = _batch()
    loss = ns.distillation_loss(s, t, y, temperature=2.0, alpha=0.5)
    loss = torch.as_tensor(loss)
    assert loss.ndim == 0, f"loss should be a scalar, got shape {tuple(loss.shape)}"
    assert float(loss) >= 0, f"loss should be non-negative, got {float(loss):.4f}"


def check_alpha_zero_is_pure_cross_entropy(ns):
    s, t, y = _batch()
    got = torch.as_tensor(ns.distillation_loss(s, t, y, temperature=2.0, alpha=0.0))
    assert torch.allclose(got.float(), F.cross_entropy(s, y), atol=1e-4), \
        (f"alpha=0 should leave only the hard-label term "
         f"({float(got):.4f} vs cross_entropy {float(F.cross_entropy(s, y)):.4f})")


def check_matching_the_teacher_lowers_the_soft_term(ns):
    _, t, y = _batch()
    same = torch.as_tensor(ns.distillation_loss(t.clone(), t, y, temperature=2.0, alpha=1.0))
    diff = torch.as_tensor(ns.distillation_loss(-t, t, y, temperature=2.0, alpha=1.0))
    assert float(same) < float(diff), \
        ("a student matching the teacher exactly must score lower than one "
         f"disagreeing with it ({float(same):.4f} vs {float(diff):.4f})")


def check_gradients_reach_the_student(ns):
    s, t, y = _batch()
    s.requires_grad_(True)
    t.requires_grad_(True)
    torch.as_tensor(ns.distillation_loss(s, t, y, temperature=2.0, alpha=0.5)).backward()
    assert s.grad is not None and torch.any(s.grad != 0), \
        "no gradient reached the student logits"


def _soft_value_and_grad(ns, s, t, y, temperature):
    s = s.clone().requires_grad_(True)
    loss = torch.as_tensor(ns.distillation_loss(s, t, y, temperature=temperature, alpha=1.0))
    loss.backward()
    return loss.item(), s.grad


def check_soft_term_matches_hinton(ns):
    """alpha=1: T^2 * KL(p_teacher^T || p_student^T), averaged over the batch."""
    for seed, T in ((3, 1.0), (4, 2.0), (5, 4.0)):
        s, t, y = _batch(seed)
        value, grad = _soft_value_and_grad(ns, s, t, y, T)
        assert grad is not None, "no gradient reached the student logits"
        log_q, p = F.log_softmax(s / T, dim=-1), F.softmax(t / T, dim=-1)
        kl = (T * T * (p * (p.log() - log_q)).sum(-1).mean()).item()
        soft_ce = (T * T * -(p * log_q).sum(-1).mean()).item()
        assert min(abs(value - kl), abs(value - soft_ce)) < 1e-4, (
            f"T={T}, alpha=1: got {value:.4f}, expected T^2 * KL(teacher_T || student_T) "
            f"= {kl:.4f} (batch mean). KL's input is the student's log_softmax(s/T), its "
            f"target the teacher's softmax(t/T), with reduction='batchmean'.")
        want = T * (F.softmax(s / T, dim=-1) - p) / s.size(0)
        diff = (grad - want).abs().max().item()
        assert diff < 1e-5, (
            f"T={T}, alpha=1: d(loss)/d(student_logits) should be "
            f"T * (softmax(s/T) - softmax(t/T)) / batch (max diff {diff:.2e}) — check the "
            f"T^2 factor, the KL direction and reduction='batchmean' (not 'mean')")


def check_alpha_blends_the_two_terms(ns):
    """The loss is linear in alpha: alpha * soft + (1 - alpha) * hard."""
    s, t, y = _batch(6)
    def at(a):
        return float(torch.as_tensor(ns.distillation_loss(s, t, y, temperature=3.0, alpha=a)))
    soft, hard, mid = at(1.0), at(0.0), at(0.3)
    want = 0.3 * soft + 0.7 * hard
    assert abs(mid - want) < 1e-4, (
        f"alpha=0.3 gave {mid:.4f}, but 0.3 * loss(alpha=1) + 0.7 * loss(alpha=0) = {want:.4f}")


CHECKS = [check_returns_scalar, check_alpha_zero_is_pure_cross_entropy,
          check_matching_the_teacher_lowers_the_soft_term,
          check_gradients_reach_the_student, check_soft_term_matches_hinton,
          check_alpha_blends_the_two_terms]
