"""muon — Muon built on torch.optim.Optimizer.

Muon's Newton–Schulz step does not return an exactly orthogonal matrix: each
iteration applies the odd polynomial p(s) = a*s + b*s^3 + c*s^5 to every
singular value of X and leaves the singular vectors alone. With Muon's tuned
coefficients (3.4445, -4.7750, 2.0315) the singular values land in roughly
[0.7, 1.2] after 5 steps; with the textbook (15/8, -5/4, 3/8) they converge to
1 but more slowly. So newton_schulz is graded as geometry that both satisfy:
the output keeps X's singular vectors, its singular values are pulled into a
band around 1, and it does not depend on the scale of X (normalize first).

Muon is graded by the geometry of its updates on hand-set gradients: the step
on a matrix must be lr * (shape scale) * an orthogonalized matrix sharing the
singular vectors of the momentum update (so plain momentum fails — its step
keeps the gradient's spread of singular values), the second step must follow
g2 + mu*(mu*g1 + g2) with Nesterov and mu*g1 + g2 without, conv filters are
flattened to 2D, and decoupled decay hits matrices too. Any of the published
shape scalings (Keller's sqrt(max(1, rows/cols)), Moonshot's
0.2*sqrt(max(rows, cols)), sqrt(rows/cols)) passes. The 1D fallback is pinned
by the hand-derived first AdamW step, and the whole optimizer must halve the
loss of a small linear model.
"""
import torch

ENTRIES = ["newton_schulz", "MyMuon"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Subclass torch.optim.Optimizer and keep per-parameter state in "
    "self.state[p] — it starts as an empty dict, so initialize it on the "
    "first step. Do the whole update under @torch.no_grad() with in-place ops "
    "(mul_, add_, addcmul_, addcdiv_).",
    "newton_schulz: transpose a tall X so X @ X.T is the small Gram matrix, "
    "divide by the Frobenius norm (+ eps) so every singular value is <= 1, "
    "then repeat A = X @ X.T; X = a*X + (b*A + c*A@A) @ X with Muon's "
    "(a, b, c) = (3.4445, -4.7750, 2.0315). The singular values end up in "
    "about [0.7, 1.2] — that is expected, not a bug.",
    "Muon on a matrix: buf = momentum*buf + grad, update = grad + momentum*buf "
    "(Nesterov) or buf; flatten it to (len(p), -1), orthogonalize with "
    "newton_schulz, scale by sqrt(max(1, rows/cols)), decay p by "
    "(1 - lr*weight_decay), then p -= lr * scale * update reshaped to p.shape. "
    "Only ndim >= 2 — 1D params get the AdamW fallback.",
]


# Exact arithmetic preserves singular vectors perfectly; this slack admits a
# Newton–Schulz run in bfloat16 (as in Keller Jordan's GPU code, ~0.03 here)
# while every wrong update direction tried lands above 0.18.
_TOL = 0.08


def _with_spectrum(m, n, svals, seed):
    """An m x n matrix with prescribed singular values and random vectors."""
    g = torch.Generator().manual_seed(seed)
    k = min(m, n)
    U, _ = torch.linalg.qr(torch.randn(m, k, generator=g))
    V, _ = torch.linalg.qr(torch.randn(n, k, generator=g))
    return U @ torch.diag(svals) @ V.T


def _spread(k, lo=1.0, hi=10.0):
    return torch.logspace(torch.log10(torch.tensor(lo)), torch.log10(torch.tensor(hi)), k)


def _coords(M, O):
    """Express O in the singular basis of M: D = U^T O V, plus the share of O
    that lies outside span(U) x span(V)."""
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    D = U.T @ O @ Vh.T
    outside = float((O - U @ D @ Vh).norm() / O.norm().clamp_min(1e-30))
    return D, outside


def _assert_orthogonalized(M, O, what, lo=0.5, hi=1.5):
    """O must be U diag(d) V^T for M = U S V^T, with every d in [lo, hi]."""
    assert torch.isfinite(O).all(), f"{what} contains NaN/Inf"
    D, outside = _coords(M, O)
    diag = D.diag()
    off = (D - torch.diag(diag)).abs().max()
    off_ratio = float(off / diag.abs().max().clamp_min(1e-30))
    assert outside < _TOL and off_ratio < _TOL, (
        f"{what} does not keep the singular vectors of its input "
        f"(off-diagonal {off_ratio:.3f}, outside the span {outside:.3f}) — "
        "every Newton–Schulz step must be X -> a*X + b*(X X^T) X + "
        "c*(X X^T)^2 X, which only reshapes the singular values")
    d_lo, d_hi = float(diag.min()), float(diag.max())
    assert lo <= d_lo and d_hi <= hi, (
        f"{what} has singular values in [{d_lo:.3f}, {d_hi:.3f}]; after 5 "
        f"Newton–Schulz steps they should all sit in [{lo}, {hi}] (Muon's "
        "coefficients give roughly [0.7, 1.2]). Normalize by the Frobenius "
        "norm first and check the coefficients (3.4445, -4.7750, 2.0315)")


def _step(opt, p, grad):
    p.grad = grad.clone()
    before = p.detach().clone()
    opt.step()
    return before - p.detach()          # = lr * scale * orthogonalized update


def _muon(ns, p, **kw):
    defaults = dict(lr=0.05, momentum=0.9, nesterov=True, ns_steps=5,
                    weight_decay=0.0)
    defaults.update(kw)
    return ns.MyMuon([p], **defaults)


# --------------------------------------------------------------- newton_schulz


def check_newton_schulz_keeps_singular_vectors_and_flattens_singular_values(ns):
    for (m, n), seed in (((16, 32), 1), ((32, 16), 2), ((24, 8), 3), ((16, 16), 4)):
        k = min(m, n)
        X = _with_spectrum(m, n, _spread(k), seed)   # condition number 10
        O = ns.newton_schulz(X, steps=5)
        assert tuple(O.shape) == (m, n), (
            f"newton_schulz must keep the input shape {(m, n)}, got "
            f"{tuple(O.shape)}")
        _assert_orthogonalized(X, O, f"newton_schulz of a {m}x{n} matrix")


def check_newton_schulz_is_scale_invariant(ns):
    """Orthogonalizing X or a multiple of X must give the same answer — only
    possible if the first step normalizes away the input scale."""
    g = torch.Generator().manual_seed(5)
    X = torch.randn(16, 32, generator=g)
    O1 = ns.newton_schulz(X, steps=5).float()
    O2 = ns.newton_schulz(7.3 * X, steps=5).float()
    dev = float((O1 - O2).norm() / O1.norm())
    assert dev < 0.05, (
        f"newton_schulz(X) and newton_schulz(7.3*X) differ by {dev:.1%} — the "
        "iteration only converges for spectral norm <= 1, so the input must be "
        "divided by its Frobenius norm (a cheap upper bound) BEFORE iterating")


# --------------------------------------------------------------- Muon, matrices


def check_muon_matrix_step_is_an_orthogonalized_update(ns):
    """From zero state the update is proportional to the gradient, so the
    step must be lr * scale * U d V^T with the gradient's singular vectors and
    d ~ 1. The gradient's singular values span 1..10: plain momentum without
    orthogonalization keeps that 10x spread and fails."""
    lr = 0.05
    for (m, n), seed in (((16, 8), 7), ((8, 16), 8)):
        grad = _with_spectrum(m, n, _spread(min(m, n)), seed)
        p = torch.randn(m, n).requires_grad_(True)
        delta = _step(_muon(ns, p, lr=lr), p, grad)
        # Any published shape scale (1, sqrt(2), 0.2*sqrt(16), 1/sqrt(2) here)
        # keeps delta / lr inside this band; a missing lr or normalization does not.
        _assert_orthogonalized(grad, delta / lr,
                               f"one Muon step on a {m}x{n} matrix, divided by lr,",
                               lo=0.3, hi=2.5)


def check_muon_momentum_and_nesterov(ns):
    """Second step from hand-set gradients g1, g2 (momentum 0.9):
    buf = 0.9*g1 + g2, and the orthogonalized update is that of
    g2 + 0.9*buf with Nesterov, of buf without."""
    mu = 0.9
    g = torch.Generator().manual_seed(9)
    g1, g2 = torch.randn(12, 8, generator=g), torch.randn(12, 8, generator=g)
    for nesterov in (True, False):
        p = torch.randn(12, 8, generator=g).requires_grad_(True)
        opt = _muon(ns, p, momentum=mu, nesterov=nesterov)
        _step(opt, p, g1)
        delta2 = _step(opt, p, g2)
        buf = mu * g1 + g2
        want = g2 + mu * buf if nesterov else buf
        D, outside = _coords(want, delta2)
        diag = D.diag()
        off_ratio = float((D - torch.diag(diag)).abs().max() / diag.abs().max())
        assert outside < _TOL and off_ratio < _TOL, (
            f"with nesterov={nesterov} the second step should orthogonalize "
            f"{'g2 + momentum*buf' if nesterov else 'buf'} where "
            "buf = momentum*g1 + g2, but its singular vectors do not match "
            f"(off-diagonal {off_ratio:.3f}) — check the momentum buffer update "
            "and which tensor you orthogonalize")


def check_muon_flattens_conv_filters(ns):
    """A conv weight (out, in, kh, kw) is orthogonalized as an
    (out, in*kh*kw) matrix and the step keeps the 4D shape."""
    lr = 0.05
    grad2d = _with_spectrum(8, 36, _spread(8), 10)
    grad = grad2d.reshape(8, 4, 3, 3)
    p = torch.randn(8, 4, 3, 3).requires_grad_(True)
    delta = _step(_muon(ns, p, lr=lr), p, grad)
    assert tuple(delta.shape) == (8, 4, 3, 3), \
        f"the conv weight changed shape to {tuple(delta.shape)}"
    _assert_orthogonalized(grad2d, delta.reshape(8, -1) / lr,
                           "one Muon step on an (8, 4, 3, 3) conv weight, "
                           "flattened to (8, 36) and divided by lr,",
                           lo=0.3, hi=2.5)


def check_muon_decoupled_weight_decay_on_matrices(ns):
    """With a zero gradient the momentum update is zero, so all that is left
    is the decoupled decay p *= (1 - lr * weight_decay)."""
    lr, wd = 0.05, 0.2
    p0 = torch.randn(6, 4)
    p = p0.clone().requires_grad_(True)
    _step(_muon(ns, p, lr=lr, weight_decay=wd), p, torch.zeros(6, 4))
    expected = p0 * (1 - lr * wd)
    got = p.detach()
    assert torch.isfinite(got).all(), (
        "a zero gradient produced NaN — add a small eps to the norm in "
        "newton_schulz so 0 / 0 cannot happen")
    assert torch.allclose(got, expected, atol=1e-6), (
        f"with a zero gradient a matrix should just decay to "
        f"p * (1 - lr*weight_decay) = {float(expected[0, 0]):.6f}, got "
        f"{float(got[0, 0]):.6f} — apply p.mul_(1 - lr * weight_decay) to "
        "matrices too")


# --------------------------------------------------------------- Muon, 1D + end-to-end


def check_muon_falls_back_to_adamw_for_1d_params(ns):
    """1D parameters (biases, norm gains) cannot be orthogonalized — Muon
    falls back to AdamW for them, and the first AdamW step from zero state is
    -lr * sign(grad)."""
    g = torch.Generator().manual_seed(8)
    p0 = torch.randn(8, generator=g)
    grad = torch.randn(8, generator=g) * 5.0
    p = p0.clone().requires_grad_(True)
    opt = ns.MyMuon([p], lr=0.02, weight_decay=0.0,
                    adamw_betas=(0.9, 0.95), adamw_eps=1e-8)
    p.grad = grad
    opt.step()
    delta = p.detach() - p0
    expected = -0.02 * grad.sign()
    assert torch.allclose(delta, expected, atol=1e-6), (
        f"a 1D parameter should get the AdamW fallback: from zero state the "
        f"first step is -lr*sign(grad) = {float(expected[0]):.6f} on "
        f"coordinate 0, got {float(delta[0]):.6f}. newton_schulz is only for "
        "ndim >= 2; biases and norm gains need m/v moment updates with bias "
        "correction, exactly like AdamW")


def check_muon_reduces_loss_on_a_small_linear_model(ns):
    import torch.nn as nn
    g = torch.Generator().manual_seed(9)
    model = nn.Linear(16, 8)  # 2D weight (Muon branch) + 1D bias (AdamW branch)
    with torch.no_grad():
        model.weight.copy_(torch.randn(8, 16, generator=g))
        model.bias.copy_(torch.randn(8, generator=g))
    opt = ns.MyMuon(model.parameters(), lr=0.02, momentum=0.95,
                    weight_decay=1e-3)
    X = torch.randn(32, 16, generator=g)
    y = torch.randn(32, 8, generator=g)
    with torch.no_grad():
        first = float(((model(X) - y) ** 2).mean())
    for _ in range(200):
        loss = ((model(X) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        final = float(((model(X) - y) ** 2).mean())
    assert final < first * 0.5, (
        f"200 Muon steps should at least halve the MSE of a tiny linear "
        f"model: loss went {first:.4f} -> {final:.4f}. Check both branches — "
        "the 2D weight needs momentum + orthogonalization + the "
        "sqrt(max(m,n)/min(m,n)) scale, the 1D bias needs the AdamW fallback")


CHECKS = [
    check_newton_schulz_keeps_singular_vectors_and_flattens_singular_values,
    check_newton_schulz_is_scale_invariant,
    check_muon_matrix_step_is_an_orthogonalized_update,
    check_muon_momentum_and_nesterov,
    check_muon_flattens_conv_filters,
    check_muon_decoupled_weight_decay_on_matrices,
    check_muon_falls_back_to_adamw_for_1d_params,
    check_muon_reduces_loss_on_a_small_linear_model,
]
