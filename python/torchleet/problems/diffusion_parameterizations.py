"""diffusion-parameterizations — noise schedules, eps / x0 / v / score conversions, SNR,
min-SNR-gamma loss weights, zero-terminal-SNR rescaling and the DDPM posterior.

Conventions graded here (the notebook states them; sources in brackets):

* Indexing: arrays are 0-based, array index i holds the paper's timestep t = i + 1, so
  betas[i] = beta_{i+1} and alpha_bar[i] = abar_{i+1} = prod_{s <= i+1} (1 - beta_s); abar_0 := 1.
  Schedules are float64 (Ho et al.'s and Nichol & Dhariwal's reference code use np.float64).
* linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02) = linspace(beta_start, beta_end, T)
  [Ho, Jain & Abbeel 2020, Sec. 4: "beta_1 = 1e-4 to beta_T = 0.02"].
* cosine_alpha_bar(T, s=0.008, max_beta=0.999) [Nichol & Dhariwal 2021, eq. 17 and the text
  after it]: f(t) = cos^2(((t/T + s) / (1 + s)) * pi/2), beta_t = min(1 - f(t)/f(t-1), max_beta)
  for t = 1..T, abar_t = prod_{s<=t} (1 - beta_s) — i.e. f(t)/f(0) except where a beta was
  clipped (for the defaults only beta_T, whose unclipped value is 1). This is what
  improved-diffusion's betas_for_alpha_bar + cumprod computes.
* With alpha_t := sqrt(abar_t) and sigma_t := sqrt(1 - abar_t) (signal / noise scales, VDM
  notation — NOT DDPM's alpha_t = 1 - beta_t), x_t = alpha_t x0 + sigma_t eps:
    eps_to_x0 = (x_t - sigma eps) / alpha,     x0_to_eps = (x_t - alpha x0) / sigma,
    v = alpha eps - sigma x0                   [Salimans & Ho 2022, "Progressive Distillation
                                                for Fast Sampling of Diffusion Models"; it is
                                                d x_phi / d phi on x_phi = cos(phi) x0 + sin(phi) eps],
    (x0, eps) from v = (alpha x_t - sigma v, sigma x_t + alpha v),
    score = grad_x log q(x_t | x0) = -(x_t - alpha x0) / sigma^2 = -eps / sigma.
  Every conversion takes alpha_bar of shape (B,), one value per sample, broadcast over the
  trailing dims of the tensors.
* snr(alpha_bar) = abar / (1 - abar) (0 at abar = 0, inf at abar = 1).
* min_snr_weight(alpha_bar, gamma=5, target) [Hang et al., ICCV 2023, "Efficient Diffusion
  Training via Min-SNR Weighting Strategy", arXiv 2303.09556]: the x0-space weight is min(SNR, gamma); expressed on the
  target's own MSE it is
    target "x0":  min(SNR, gamma)
    target "eps": min(SNR, gamma) / SNR            (since ||x0 - x0^||^2 = ||eps - eps^||^2 / SNR)
    target "v":   min(SNR, gamma) / (SNR + 1)      (since ||x0 - x0^||^2 = ||v - v^||^2 / (SNR + 1))
  (Hang et al. state the eps form as min(gamma/SNR, 1) and the v form, "divided by (SNR + 1)",
  in Sec. 4.2 "Min-SNR-gamma for Different Prediction Targets", derived in Appendix B; diffusers'
  examples/text_to_image/train_text_to_image.py uses the same mse_loss_weights / (snr + 1)).
  Checked against diffusers 0.31 compute_snr + that script's weights.
  The eps weight takes its limit 1 at SNR = 0 (= min(1, gamma / SNR)). The check verifies the
  defining property directly: w_eps*mse_eps = w_x0*mse_x0 = w_v*mse_v for consistent predictions.
* rescale_zero_terminal_snr(betas) [Lin, Liu, Li & Yang, WACV 2024, "Common Diffusion Noise
  Schedules and Sample Steps are Flawed", arXiv 2305.08891, Algorithm 1; identical to diffusers'
  rescale_zero_terminal_snr]: shift sqrt(abar) so its last value is 0, scale so
  its first value is unchanged, convert back to betas (beta_T becomes exactly 1).
* q_posterior(x0, x_t, t, betas) with 0-based t of shape (B,) [Ho et al. 2020, eqs. 6-7]:
    mean = sqrt(abar_{t-1}) beta_t / (1 - abar_t) * x0 + sqrt(1 - beta_t) (1 - abar_{t-1}) / (1 - abar_t) * x_t
    var  = (1 - abar_{t-1}) / (1 - abar_t) * beta_t                (shape (B,))
  At the first step (index 0, abar_0 = 1) this is the point mass mean = x0, var = 0.
  The oracle here is an independent product-of-Gaussians computation.
"""
import math

import torch

ENTRIES = ["linear_beta_schedule", "cosine_alpha_bar", "eps_to_x0", "x0_to_eps",
           "v_from_x0_eps", "v_to_x0_eps", "eps_to_score", "snr", "min_snr_weight",
           "rescale_zero_terminal_snr", "q_posterior"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Write everything in terms of the signal scale alpha_t = sqrt(abar_t) and the noise scale "
    "sigma_t = sqrt(1 - abar_t), with alpha^2 + sigma^2 = 1: x_t = alpha x0 + sigma eps is one "
    "linear equation, so knowing x_t and any one of (x0, eps, v) gives the others. Keep arrays "
    "0-based (index i is the paper's t = i + 1) and give each sample its own abar.",
    "Reshape the (B,) abar to (B, 1, ..., 1) before mixing with (B, C, H, W) tensors. "
    "v = alpha*eps - sigma*x0 is a rotation of (x0, eps) by the angle phi = atan2(sigma, alpha), "
    "so the inverse is the transposed rotation. Cosine schedule: evaluate f at t = 0..T, take "
    "betas = 1 - f[1:]/f[:-1], clip at max_beta, then cumprod(1 - betas). For min-SNR, convert "
    "the x0-space weight min(SNR, gamma) to the eps / v losses by how their errors scale.",
    "eps weight = min(SNR, gamma)/SNR written as clamp(gamma/SNR, max=1) (finite at SNR = 0); "
    "v weight = min(SNR, gamma)/(SNR + 1). Zero terminal SNR: s = sqrt(cumprod(1 - betas)); "
    "s = (s - s[-1]) * s[0] / (s[0] - s[-1]); abar = s^2; alphas = [abar[0], abar[1:]/abar[:-1]]; "
    "betas = 1 - alphas. Posterior: build abar_prev = [1, abar[:-1]] and gather everything at t "
    "(index 0 must give mean x0 and variance 0 — do not index abar[t - 1]).",
]

F64 = torch.float64


# --------------------------------------------------------------------------
# float64 references (written independently of the vectorized solution)
# --------------------------------------------------------------------------
def _linear_ref(T, b0=1e-4, b1=0.02):
    if T == 1:
        return torch.tensor([b0], dtype=F64)
    return torch.tensor([b0 + i * (b1 - b0) / (T - 1) for i in range(T)], dtype=F64)


def _cosine_ref(T, s=0.008, max_beta=0.999):
    """improved-diffusion's betas_for_alpha_bar loop, then a running product."""
    f = lambda u: math.cos((u + s) / (1 + s) * math.pi / 2) ** 2
    out, ab = [], 1.0
    for i in range(T):
        beta = min(1 - f((i + 1) / T) / f(i / T), max_beta)
        ab *= 1 - beta
        out.append(ab)
    return torch.tensor(out, dtype=F64)


def _abar(betas):
    out, ab = [], 1.0
    for b in betas.tolist():
        ab *= 1 - b
        out.append(ab)
    return torch.tensor(out, dtype=F64)


def _zero_snr_ref(betas):
    """Lin et al. 2024, Algorithm 1, in plain float64 Python."""
    s = [math.sqrt(a) for a in _abar(betas).tolist()]
    s0, sT = s[0], s[-1]
    s = [(x - sT) * (s0 / (s0 - sT)) for x in s]
    ab = [x * x for x in s]
    alphas = [ab[0]] + [ab[i] / ab[i - 1] for i in range(1, len(ab))]
    return torch.tensor([1 - a for a in alphas], dtype=F64)


def _bc(a, x):
    return a.reshape(-1, *([1] * (x.dim() - 1)))


def _close(got, want, what, rtol=1e-9, atol=1e-12):
    assert torch.is_tensor(got), f"{what} must be a tensor, got {type(got).__name__}"
    assert tuple(got.shape) == tuple(want.shape), \
        f"{what} has shape {tuple(got.shape)}, expected {tuple(want.shape)}"
    got = got.to(F64)
    ok = torch.allclose(got, want, rtol=rtol, atol=atol, equal_nan=False)
    return ok, (got - want).abs().max().item() if torch.isfinite(got).all() else float("nan")


def _batch(seed=0, B=5, shape=(3, 4, 5)):
    """float64 (x0, eps, abar) with B == last dim, so a (B,) abar broadcast along the last
    axis runs without error but gives wrong numbers."""
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(B, *shape, generator=g, dtype=F64)
    eps = torch.randn(B, *shape, generator=g, dtype=F64)
    ab = torch.tensor([0.9999, 0.93, 0.5, 0.07, 3e-4], dtype=F64)[:B]
    return x0, eps, ab


# --------------------------------------------------------------------------
# schedules
# --------------------------------------------------------------------------
def check_linear_beta_schedule(ns):
    for args, T in (((1000,), 1000), ((10, 1e-3, 0.05), 10)):
        got = ns.linear_beta_schedule(*args)
        assert torch.is_tensor(got), "linear_beta_schedule must return a tensor"
        assert tuple(got.shape) == (T,), (
            f"linear_beta_schedule({T}) must have T = {T} entries beta_1..beta_T, got shape "
            f"{tuple(got.shape)}")
        assert got.dtype == F64, (
            f"return float64 (got {got.dtype}): products of 1000 factors and 1 - abar near "
            f"abar = 1 lose ~1e-4 relative accuracy in float32 (SNR at t = 1 is off by ~2e-4)")
        want = _linear_ref(T, *args[1:])
        ok, err = _close(got, want, "betas", rtol=1e-12, atol=1e-15)
        if not ok and torch.allclose(got, want, rtol=1e-6, atol=0):
            raise AssertionError(
                f"betas are right only to float32 precision (max diff {err:.3e}): build them in "
                f"float64 (torch.linspace(..., dtype=torch.float64)) instead of casting a float32 "
                f"result")
        assert ok, (f"betas must be linspace(beta_start, beta_end, T) with beta_1 = beta_start and "
                    f"beta_T = beta_end (max diff {err:.3e})")


def check_cosine_alpha_bar_matches_improved_ddpm(ns):
    for kw in (dict(T=1000), dict(T=50), dict(T=4), dict(T=30, s=0.1), dict(T=20, max_beta=0.5)):
        T = kw["T"]
        s = kw.get("s", 0.008)
        mb = kw.get("max_beta", 0.999)
        try:
            got = ns.cosine_alpha_bar(**kw)
        except TypeError:
            got = ns.cosine_alpha_bar(T, s, mb)
        want = _cosine_ref(T, s, mb)
        assert torch.is_tensor(got) and got.dtype == F64, \
            f"cosine_alpha_bar must return a float64 tensor, got {getattr(got, 'dtype', type(got))}"
        f = lambda u: math.cos((u + s) / (1 + s) * math.pi / 2) ** 2
        if tuple(got.shape) == (T + 1,):
            raise AssertionError(
                f"cosine_alpha_bar({T}) returned T + 1 values: return abar_1..abar_T (index i = "
                f"paper t = i + 1); abar_0 = 1 is implicit")
        ok, err = _close(got, want, f"cosine_alpha_bar(**{kw})", rtol=1e-10, atol=0)
        if ok:
            continue
        if torch.allclose(got.to(F64), want, rtol=1e-3, atol=0):
            raise AssertionError(
                f"cosine_alpha_bar(**{kw}) matches the schedule only to float32 precision (max "
                f"diff {err:.3e}): compute f, the betas and the cumprod in float64 (e.g. "
                f"torch.arange(T + 1, dtype=torch.float64)) instead of casting a float32 result")
        shifted = torch.tensor([f(i / T) / f(0) for i in range(T)], dtype=F64)
        if torch.allclose(got.to(F64), shifted, rtol=1e-8):
            raise AssertionError(
                "off by one: got[0] = 1 is abar_0. Index i must hold abar_{i+1} = f((i+1)/T)/f(0), "
                "so got[0] < 1 and got[-1] is abar_T")
        raw = torch.tensor([f(i / T) / f(0) for i in range(1, T + 1)], dtype=F64)
        if torch.allclose(got.to(F64), raw, rtol=1e-8, atol=1e-30):
            raise AssertionError(
                f"no clipping: f(T)/f(0) = {raw[-1].item():.2e} is (numerically) 0, i.e. beta_T = 1. "
                f"Nichol & Dhariwal clip every beta_t = 1 - abar_t/abar_(t-1) at max_beta = {mb}, "
                f"so abar_T = {want[-1].item():.3e} here; derive the betas, clip, then cumprod")
        if mb != 0.999:
            raise AssertionError(
                f"with max_beta = {mb} every beta_t above it must be clipped (here more than the "
                f"last step) — clip the betas, not only beta_T; max diff {err:.3e}")
        raise AssertionError(
            f"cosine_alpha_bar(**{kw}) differs from Nichol & Dhariwal's schedule (max diff "
            f"{err:.3e}): f(t) = cos^2(((t/T + s)/(1 + s)) pi/2), beta_t = min(1 - f(t)/f(t-1), "
            f"max_beta), abar_t = prod (1 - beta)")
    ab = ns.cosine_alpha_bar(1000)
    assert bool((ab[1:] < ab[:-1]).all()) and 0 < ab[-1].item() and ab[0].item() < 1, \
        "abar must be strictly decreasing inside (0, 1)"


# --------------------------------------------------------------------------
# parameterizations
# --------------------------------------------------------------------------
def check_eps_x0_conversions(ns):
    x0, eps, ab = _batch()
    a, s = _bc(ab.sqrt(), x0), _bc((1 - ab).sqrt(), x0)
    x_t = a * x0 + s * eps
    got = ns.eps_to_x0(x_t, eps, ab)
    ok, err = _close(got, x0, "eps_to_x0", rtol=1e-8, atol=1e-10)
    if not ok:
        abb = _bc(ab, x0)
        if torch.allclose(got.to(F64), (x_t - (1 - abb) * eps) / abb, rtol=1e-6):
            raise AssertionError(
                "eps_to_x0 used the variances 1 - abar and abar where the scales sqrt(1 - abar) "
                "and sqrt(abar) belong: x0 = (x_t - sqrt(1 - abar) eps) / sqrt(abar)")
        raise AssertionError(
            f"eps_to_x0 must invert x_t = sqrt(abar) x0 + sqrt(1 - abar) eps (max diff {err:.3e}); "
            f"abar has shape (B,) — reshape it to (B, 1, ..., 1), broadcasting a (B,) tensor "
            f"against (B, C, H, W) pairs it with the LAST axis")
    got = ns.x0_to_eps(x_t, x0, ab)
    ok, err = _close(got, eps, "x0_to_eps", rtol=1e-8, atol=1e-10)
    assert ok, (f"x0_to_eps must give eps = (x_t - sqrt(abar) x0) / sqrt(1 - abar) (max diff "
                f"{err:.3e}); check the per-sample broadcasting of abar")
    # round trip on (B, D) inputs too
    x2 = torch.randn(5, 7, dtype=F64, generator=torch.Generator().manual_seed(1))
    e2 = torch.randn(5, 7, dtype=F64, generator=torch.Generator().manual_seed(2))
    back = ns.x0_to_eps(x2, ns.eps_to_x0(x2, e2, ab), ab)
    assert torch.allclose(back.to(F64), e2, rtol=1e-7, atol=1e-9), \
        "x0_to_eps(x_t, eps_to_x0(x_t, eps, abar), abar) must return eps for (B, D) inputs"


def check_v_from_x0_eps(ns):
    x0, eps, ab = _batch(seed=3)
    a, s = _bc(ab.sqrt(), x0), _bc((1 - ab).sqrt(), x0)
    want = a * eps - s * x0
    got = ns.v_from_x0_eps(x0, eps, ab)
    ok, err = _close(got, want, "v_from_x0_eps")
    if not ok:
        g = got.to(F64)
        abb = _bc(ab, x0)
        for why, v in {
            "the sign is flipped: Salimans & Ho define v = alpha*eps - sigma*x0 (it points toward "
            "noise as phi grows)": -want,
            "alpha and sigma are swapped: v = sqrt(abar) eps - sqrt(1 - abar) x0": s * eps - a * x0,
            "variances were used instead of scales: v = sqrt(abar) eps - sqrt(1 - abar) x0":
                abb * eps - (1 - abb) * x0,
        }.items():
            if torch.allclose(g, v, rtol=1e-8, atol=1e-10):
                raise AssertionError(f"v_from_x0_eps: {why}")
        raise AssertionError(f"v_from_x0_eps must be sqrt(abar)*eps - sqrt(1-abar)*x0 (max diff {err:.3e})")
    # independent oracle: v is d x / d phi on x(phi) = cos(phi) x0 + sin(phi) eps
    phi = torch.atan2((1 - ab).sqrt(), ab.sqrt()).requires_grad_(True)
    x_phi = _bc(torch.cos(phi), x0) * x0 + _bc(torch.sin(phi), x0) * eps
    up = torch.randn(x0.shape, dtype=F64, generator=torch.Generator().manual_seed(4))
    (dphi,) = torch.autograd.grad((x_phi * up).sum(), phi)
    assert torch.allclose(dphi, (got.to(F64) * up).flatten(1).sum(1), rtol=1e-8), \
        "v must equal d x_phi / d phi for x_phi = cos(phi) x0 + sin(phi) eps"


def check_v_to_x0_eps(ns):
    x0, eps, ab = _batch(seed=5)
    a, s = _bc(ab.sqrt(), x0), _bc((1 - ab).sqrt(), x0)
    x_t, v = a * x0 + s * eps, a * eps - s * x0
    out = ns.v_to_x0_eps(x_t, v, ab)
    assert isinstance(out, (tuple, list)) and len(out) == 2, "v_to_x0_eps must return a tuple (x0, eps)"
    for name, got, want in (("x0", out[0], x0), ("eps", out[1], eps)):
        ok, err = _close(got, want, f"v_to_x0_eps -> {name}", rtol=1e-8, atol=1e-10)
        if not ok:
            if name == "x0" and torch.allclose(got.to(F64), a * x_t + s * v, rtol=1e-8):
                raise AssertionError("x0 = alpha x_t - sigma v (minus): you added sigma v")
            raise AssertionError(
                f"v_to_x0_eps returned the wrong {name} (max diff {err:.3e}): with alpha^2 + "
                f"sigma^2 = 1, x0 = alpha x_t - sigma v and eps = sigma x_t + alpha v")
    # the endpoints: v stays well defined where eps- or x0-prediction divide by zero
    x_t = torch.randn(2, 3, 4, dtype=F64, generator=torch.Generator().manual_seed(6))
    v = torch.randn(2, 3, 4, dtype=F64, generator=torch.Generator().manual_seed(7))
    edge = torch.tensor([0.0, 1.0], dtype=F64)
    x0e, ee = ns.v_to_x0_eps(x_t, v, edge)
    assert torch.isfinite(x0e).all() and torch.isfinite(ee).all(), \
        "v_to_x0_eps must stay finite at abar = 0 and abar = 1 (no division anywhere)"
    assert torch.allclose(x0e[0], -v[0]) and torch.allclose(ee[0], x_t[0]), \
        "at abar = 0 (pure noise) x_t = eps and v = -x0, so v_to_x0_eps must return (-v, x_t)"
    assert torch.allclose(x0e[1], x_t[1]) and torch.allclose(ee[1], v[1]), \
        "at abar = 1 (no noise) x_t = x0 and v = eps, so v_to_x0_eps must return (x_t, v)"


def check_eps_to_score(ns):
    x0, eps, ab = _batch(seed=8)
    a, s = _bc(ab.sqrt(), x0), _bc((1 - ab).sqrt(), x0)
    x_t = (a * x0 + s * eps).requires_grad_(True)
    logq = torch.distributions.Normal(a * x0, s).log_prob(x_t).sum()
    (score,) = torch.autograd.grad(logq, x_t)                 # grad of log q(x_t | x0)
    got = ns.eps_to_score(eps, ab)
    ok, err = _close(got, score, "eps_to_score", rtol=1e-8, atol=1e-10)
    if ok:
        return
    g = got.to(F64)
    if torch.allclose(g, -eps / (s * s), rtol=1e-8):
        raise AssertionError(
            "eps_to_score divided by the variance: grad log q = -(x_t - alpha x0) / sigma^2 and "
            "x_t - alpha x0 = sigma eps, so the score is -eps / sigma (one power of sigma)")
    if torch.allclose(g, eps / s, rtol=1e-8):
        raise AssertionError("eps_to_score has the wrong sign: the score points back toward the mean, -eps / sigma")
    raise AssertionError(
        f"eps_to_score must equal grad_x log N(x_t; alpha x0, sigma^2) = -eps / sqrt(1 - abar) "
        f"(max diff {err:.3e})")


# --------------------------------------------------------------------------
# SNR and min-SNR weighting
# --------------------------------------------------------------------------
def check_snr(ns):
    ab = torch.tensor([0.9999, 0.9, 0.5, 0.1, 1e-6], dtype=F64)
    got = ns.snr(ab)
    ok, err = _close(got, ab / (1 - ab), "snr", rtol=1e-12)
    if not ok:
        if torch.allclose(got.to(F64), (1 - ab) / ab, rtol=1e-9):
            raise AssertionError("snr is inverted: SNR = signal / noise variance = abar / (1 - abar)")
        if torch.allclose(got.to(F64), ab.sqrt() / (1 - ab).sqrt(), rtol=1e-9):
            raise AssertionError("snr is a ratio of VARIANCES, alpha^2 / sigma^2 = abar / (1 - abar), not of scales")
        raise AssertionError(f"snr(abar) must be abar / (1 - abar) (max diff {err:.3e})")
    edge = ns.snr(torch.tensor([0.0, 1.0], dtype=F64))
    assert edge[0].item() == 0.0 and math.isinf(edge[1].item()), \
        "snr must be exactly 0 at abar = 0 and inf at abar = 1"
    sched = ns.snr(_cosine_ref(100))
    assert tuple(sched.shape) == (100,) and bool((sched[1:] < sched[:-1]).all()), \
        "snr over a schedule must keep its shape and decrease with t"


def _w(ns, ab, gamma, target):
    try:
        out = ns.min_snr_weight(ab, gamma=gamma, target=target)
    except TypeError:
        out = ns.min_snr_weight(ab, gamma, target)
    assert torch.is_tensor(out) and tuple(out.shape) == tuple(ab.shape), \
        f"min_snr_weight must return one weight per abar, shape {tuple(ab.shape)}"
    return out.to(F64)


def check_min_snr_weights(ns):
    ab = torch.tensor([0.9999, 0.95, 0.8, 0.5, 0.2, 0.01], dtype=F64)
    r = ab / (1 - ab)
    for gamma in (5.0, 1.0):
        ref = {"x0": r.clamp(max=gamma), "eps": r.clamp(max=gamma) / r,
               "v": r.clamp(max=gamma) / (r + 1)}
        for target, want in ref.items():
            got = _w(ns, ab, gamma, target)
            if torch.allclose(got, want, rtol=1e-10):
                continue
            for other, val in ref.items():
                if other != target and torch.allclose(got, val, rtol=1e-10):
                    raise AssertionError(
                        f"min_snr_weight(target='{target}') returned the '{other}' formula. The "
                        f"x0-space weight min(SNR, gamma) becomes min(SNR, gamma)/SNR on the eps "
                        f"MSE and min(SNR, gamma)/(SNR + 1) on the v MSE")
            if torch.allclose(got, torch.maximum(r, torch.tensor(gamma, dtype=F64)) * want / r.clamp(max=gamma), rtol=1e-10):
                raise AssertionError("use min(SNR, gamma), not max")
            raise AssertionError(
                f"min_snr_weight(target='{target}', gamma={gamma}) is wrong (max diff "
                f"{(got - want).abs().max().item():.3e})")
    zero = torch.tensor([0.0], dtype=F64)
    got = {t: _w(ns, zero, 5.0, t).item() for t in ("eps", "x0", "v")}
    assert got["eps"] == got["eps"] and abs(got["eps"] - 1.0) < 1e-12, (
        f"at SNR = 0 (zero terminal SNR) the eps weight min(SNR, gamma)/SNR is 0/0; return its "
        f"limit 1 (write it as min(1, gamma/SNR)), got {got['eps']}")
    assert got["x0"] == 0.0 and got["v"] == 0.0, \
        f"at SNR = 0 the x0 and v weights are min(0, gamma) = 0 and 0/(0 + 1) = 0, got {got}"


def check_min_snr_weights_equalize_the_three_losses(ns):
    """The defining property: the same prediction error, measured as an eps / x0 / v MSE and
    weighted by the target's own min-SNR weight, gives the same weighted loss."""
    x0, eps, ab = _batch(seed=9)
    a, s = _bc(ab.sqrt(), x0), _bc((1 - ab).sqrt(), x0)
    x_t = a * x0 + s * eps
    eps_hat = eps + 0.3 * torch.randn(eps.shape, dtype=F64, generator=torch.Generator().manual_seed(10))
    x0_hat = (x_t - s * eps_hat) / a
    v, v_hat = a * eps - s * x0, a * eps_hat - s * x0_hat
    mse = lambda p, q: ((p - q) ** 2).flatten(1).mean(1)
    l_eps = _w(ns, ab, 5.0, "eps") * mse(eps_hat, eps)
    l_x0 = _w(ns, ab, 5.0, "x0") * mse(x0_hat, x0)
    l_v = _w(ns, ab, 5.0, "v") * mse(v_hat, v)
    assert torch.allclose(l_eps, l_x0, rtol=1e-7), (
        "w_eps * ||eps - eps^||^2 must equal w_x0 * ||x0 - x0^||^2 for the same prediction: "
        f"per-sample {l_eps.tolist()} vs {l_x0.tolist()}")
    assert torch.allclose(l_v, l_x0, rtol=1e-7), (
        "w_v * ||v - v^||^2 must equal w_x0 * ||x0 - x0^||^2 for the same prediction "
        f"(||x0 - x0^||^2 = sigma^2 ||v - v^||^2 = ||v - v^||^2 / (SNR + 1)): "
        f"{l_v.tolist()} vs {l_x0.tolist()}")


# --------------------------------------------------------------------------
# zero terminal SNR
# --------------------------------------------------------------------------
def check_rescale_zero_terminal_snr(ns):
    cos_ab = _cosine_ref(200)
    cos_betas = 1 - torch.cat([cos_ab[:1], cos_ab[1:] / cos_ab[:-1]])
    sd_betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=F64) ** 2   # SD "scaled_linear"
    for name, betas in (("linear T=1000", _linear_ref(1000)), ("SD scaled-linear", sd_betas),
                        ("cosine T=200", cos_betas), ("linear T=10", _linear_ref(10, 1e-3, 0.2))):
        keep = betas.clone()
        got = ns.rescale_zero_terminal_snr(betas)
        assert torch.equal(betas, keep), "rescale_zero_terminal_snr must not modify its input in place"
        assert torch.is_tensor(got) and tuple(got.shape) == tuple(betas.shape), \
            f"{name}: must return betas of shape {tuple(betas.shape)}"
        got = got.to(F64)
        ab_new, ab_old = _abar(got), _abar(betas)
        assert ab_new[-1].item() == 0.0, (
            f"{name}: terminal abar must be exactly 0 (SNR_T = 0), got {ab_new[-1].item():.3e} — "
            f"beta_T must come out as exactly 1")
        assert abs(ab_new[0].item() - ab_old[0].item()) <= 1e-12 * ab_old[0].item(), (
            f"{name}: abar_1 (hence SNR at t = 1) must be preserved: {ab_old[0].item():.12f} -> "
            f"{ab_new[0].item():.12f}. Algorithm 1 rescales sqrt(abar) so its first value is unchanged")
        want = _zero_snr_ref(betas)
        if not torch.allclose(got, want, rtol=1e-9, atol=1e-12):
            # shift/scale applied to abar instead of sqrt(abar)?
            o = ab_old
            var_space = (o - o[-1]) * (o[0] / (o[0] - o[-1]))
            vb = 1 - torch.cat([var_space[:1], var_space[1:] / var_space[:-1]])
            if torch.allclose(got, vb, rtol=1e-9, atol=1e-12):
                raise AssertionError(
                    f"{name}: you shifted and scaled abar; Lin et al. shift and scale sqrt(abar) "
                    f"(the signal scale), which changes the intermediate SNRs differently")
            raise AssertionError(
                f"{name}: betas differ from Lin et al.'s Algorithm 1 (max diff "
                f"{(got - want).abs().max().item():.3e}): sqrt(abar) -> (sqrt(abar) - sqrt(abar_T)) "
                f"* sqrt(abar_1) / (sqrt(abar_1) - sqrt(abar_T)), then back to betas")
        assert bool((ab_new[1:] < ab_new[:-1]).all()) and bool((got > 0).all()) and bool((got <= 1).all()), \
            f"{name}: rescaled betas must lie in (0, 1] and abar must strictly decrease"
        sq_new, sq_old = ab_new.sqrt(), ab_old.sqrt()
        lin = (sq_old - sq_old[-1]) * sq_old[0] / (sq_old[0] - sq_old[-1])
        assert torch.allclose(sq_new, lin, rtol=1e-9, atol=1e-12), \
            f"{name}: sqrt(abar) after rescaling must be an affine function of sqrt(abar) before"


# --------------------------------------------------------------------------
# posterior q(x_{t-1} | x_t, x0)
# --------------------------------------------------------------------------
def _posterior_oracle(x0, x_t, t, betas):
    """Product of Gaussians: q(x_{t-1}|x0) = N(sqrt(abar_prev) x0, 1 - abar_prev) times
    q(x_t|x_{t-1}) = N(sqrt(1 - beta_t) x_{t-1}, beta_t), seen as a function of x_{t-1}."""
    ab = _abar(betas)
    means, vars_ = [], []
    for i, ti in enumerate(t.tolist()):
        beta = betas[ti].item()
        ab_prev = 1.0 if ti == 0 else ab[ti - 1].item()
        if ti == 0:                                   # x_0 is known: a point mass
            means.append(x0[i])
            vars_.append(0.0)
            continue
        prec = 1 / (1 - ab_prev) + (1 - beta) / beta
        var = 1 / prec
        means.append(var * (math.sqrt(ab_prev) * x0[i] / (1 - ab_prev)
                            + math.sqrt(1 - beta) * x_t[i] / beta))
        vars_.append(var)
    return torch.stack(means), torch.tensor(vars_, dtype=F64)


def _unpack_post(out):
    assert isinstance(out, (tuple, list)) and len(out) == 2, "q_posterior must return (mean, var)"
    return out


def check_q_posterior(ns):
    T = 50
    betas = _linear_ref(T, 1e-3, 0.2)
    g = torch.Generator().manual_seed(11)
    t = torch.tensor([0, 1, 7, 30, T - 1])                      # B = 5 == last dim
    x0 = torch.randn(5, 2, 3, 5, dtype=F64, generator=g)
    x_t = torch.randn(5, 2, 3, 5, dtype=F64, generator=g)
    mean, var = _unpack_post(ns.q_posterior(x0, x_t, t, betas))
    want_m, want_v = _posterior_oracle(x0, x_t, t, betas)
    assert tuple(mean.shape) == tuple(x_t.shape), f"mean must have x_t's shape {tuple(x_t.shape)}"
    assert torch.is_tensor(var) and tuple(var.shape) == (5,), \
        f"var must have shape (B,) = (5,), got {tuple(getattr(var, 'shape', ()))}"
    mean, var = mean.to(F64), var.to(F64)
    ab = _abar(betas)
    if not torch.allclose(var, want_v, rtol=1e-10, atol=1e-15):
        wrap = (1 - ab[-1]) / (1 - ab[0]) * betas[0]
        if abs(var[0].item() - wrap.item()) < 1e-12:
            raise AssertionError(
                "at index 0 you used abar[t - 1] = abar[-1], which wraps around to abar_T. "
                "abar_{t-1} for the first step is abar_0 = 1: prepend a 1 (abar_prev = [1, abar[:-1]])")
        if torch.allclose(var, betas[t], rtol=1e-10):
            raise AssertionError(
                "var is beta_t — that is DDPM's other *sampling* choice for sigma_t^2; the posterior "
                "variance of q(x_{t-1}|x_t, x0) is beta~_t = (1 - abar_{t-1}) / (1 - abar_t) * beta_t "
                "(0 at the first step, where x_{t-1} = x_0 is known)")
        if abs(var[0].item()) > 1e-15 and torch.allclose(var[1:], want_v[1:], rtol=1e-10):
            raise AssertionError(
                f"at the first step (index 0) x_{{t-1}} = x_0 is known, so the variance must be 0 "
                f"(abar_0 = 1), got {var[0].item():.3e}")
        raise AssertionError(
            f"posterior variance must be (1 - abar_(t-1)) / (1 - abar_t) * beta_t per sample "
            f"(max diff {(var - want_v).abs().max().item():.3e}) — each row uses its own t")
    if not torch.allclose(mean, want_m, rtol=1e-9, atol=1e-11):
        # isolate the coefficients by linearity
        one, zero = torch.ones_like(x0), torch.zeros_like(x0)
        c0 = ns.q_posterior(one, zero, t, betas)[0].to(F64).flatten(1)[:, 0]
        ct = ns.q_posterior(zero, one, t, betas)[0].to(F64).flatten(1)[:, 0]
        ab_prev = torch.cat([torch.ones(1, dtype=F64), ab[:-1]])[t]
        w0 = ab_prev.sqrt() * betas[t] / (1 - ab[t])
        wt = (1 - betas[t]).sqrt() * (1 - ab_prev) / (1 - ab[t])
        if torch.allclose(c0, wt, rtol=1e-9) and torch.allclose(ct, w0, rtol=1e-9):
            raise AssertionError("the x0 and x_t coefficients of the posterior mean are swapped")
        fmt = lambda v: [round(x, 6) for x in v.tolist()]
        first = ("" if torch.allclose(mean[0], x0[0], atol=1e-12) else
                 " At the first step (index 0) the mean must be exactly x0.")
        raise AssertionError(
            f"posterior mean is wrong (max diff {(mean - want_m).abs().max().item():.3e}).{first} "
            f"Coefficient of x0: sqrt(abar_(t-1)) beta_t / (1 - abar_t) = {fmt(w0)}, got {fmt(c0)}; "
            f"of x_t: sqrt(1 - beta_t) (1 - abar_(t-1)) / (1 - abar_t) = {fmt(wt)}, got {fmt(ct)}")
    # zero-terminal-SNR schedule: the last step's posterior ignores x_t and stays finite
    zb = _zero_snr_ref(_linear_ref(T, 1e-3, 0.2))
    tl = torch.tensor([T - 1, T - 2])
    m2, v2 = _unpack_post(ns.q_posterior(x0[:2], x_t[:2], tl, zb))
    wm, wv = _posterior_oracle(x0[:2], x_t[:2], tl, zb)
    assert torch.isfinite(m2).all() and torch.allclose(m2.to(F64), wm, rtol=1e-9, atol=1e-11) \
        and torch.allclose(v2.to(F64), wv, rtol=1e-9), (
        "with a zero-terminal-SNR schedule (beta_T = 1) the last posterior is "
        "N(sqrt(abar_(T-1)) x0, 1 - abar_(T-1)) — independent of x_T — and must stay finite")


CHECKS = [
    check_linear_beta_schedule,
    check_cosine_alpha_bar_matches_improved_ddpm,
    check_eps_x0_conversions,
    check_v_from_x0_eps,
    check_v_to_x0_eps,
    check_eps_to_score,
    check_snr,
    check_min_snr_weights,
    check_min_snr_weights_equalize_the_three_losses,
    check_rescale_zero_terminal_snr,
    check_q_posterior,
]
