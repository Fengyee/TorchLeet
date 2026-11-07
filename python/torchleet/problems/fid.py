"""fid — Fréchet Inception Distance from feature vectors (no Inception network).

FID = ||mu1 - mu2||^2 + Tr(S1 + S2 - 2 (S1 S2)^{1/2}). The checks never compare to
a stored number; they use closed forms and an independent reference:

* feature_statistics: mean and *unbiased* covariance, against torch.cov;
* identical statistics give 0; a translation gives exactly ||delta mu||^2; diagonal
  covariances give sum((sqrt(a) - sqrt(b))^2) + ||delta mu||^2 (also with exact zeros
  on the diagonal, i.e. singular covariances); the distance is symmetric;
* random SPD pairs match a reference written here a different way — complex
  eigenvalues of S1 @ S2 (torch.linalg.eigvals), principal sqrt, real part summed;
* rank-deficient covariances (N < D features) must give a finite, non-negative
  (up to round-off) value that also matches the reference;
* float32 inputs with large, nearly equal traces must still be computed to
  float64 accuracy: the answer is a small difference of ~1e5-sized traces, which
  float32 arithmetic gets wrong by tens of percent.
"""
import torch

ENTRIES = ["feature_statistics", "frechet_distance"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "FID fits a Gaussian to each feature set, so feature_statistics is just the mean "
    "and covariance of the rows. Use the unbiased covariance (divide by N - 1), the "
    "same convention as torch.cov(feats.T) and numpy.cov.",
    "The only hard term is Tr((S1 S2)^{1/2}). S1 S2 is not symmetric, but it is similar "
    "to the symmetric PSD matrix S1^{1/2} S2 S1^{1/2}, so both have the same "
    "non-negative eigenvalues lambda_i and the trace is sum(sqrt(lambda_i)).",
    "In float64: S1^{1/2} = V diag(sqrt(clamp(w, 0))) V^T from torch.linalg.eigh(S1); "
    "M = S1^{1/2} S2 S1^{1/2}; symmetrise M; then "
    "tr_covmean = eigvalsh(M).clamp(min=0).sqrt().sum() and "
    "FID = |mu1 - mu2|^2 + tr(S1) + tr(S2) - 2 * tr_covmean. Don't call eigvalsh on "
    "S1 @ S2 itself — it is not symmetric.",
]


def _f(v, what="frechet_distance"):
    if torch.is_tensor(v):
        assert v.numel() == 1, f"{what} must return a scalar, got shape {tuple(v.shape)}"
        v = v.reshape(())
        assert not v.is_complex(), (
            f"{what} returned a complex number — take a real symmetric route "
            f"(eigh / eigvalsh on a symmetric PSD matrix) instead of a general matrix sqrt")
    return float(v)


def _spd(d, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(d, 2 * d, generator=g, dtype=torch.float64)
    return scale * a @ a.T / (2 * d)


def _ref(mu1, s1, mu2, s2):
    """Independent reference: eigenvalues of the non-symmetric product, principal sqrt."""
    mu1, s1, mu2, s2 = (t.double() for t in (mu1, s1, mu2, s2))
    tr = torch.linalg.eigvals(s1 @ s2).sqrt().real.sum()
    d = mu1 - mu2
    return float(d @ d + s1.trace() + s2.trace() - 2 * tr)


def _alternatives(mu1, s1, mu2, s2):
    """Plausible wrong formulas, for diagnosis only."""
    mu1, s1, mu2, s2 = (t.double() for t in (mu1, s1, mu2, s2))
    d = mu1 - mu2
    tr = torch.linalg.eigvals(s1 @ s2).sqrt().real.sum()
    t12 = s1.trace() + s2.trace()
    return {
        "the cross term needs the factor 2: - 2 * Tr((S1 S2)^{1/2})": float(d @ d + t12 - tr),
        "the mean term is the *squared* distance ||mu1 - mu2||^2": float(d.norm() + t12 - 2 * tr),
        "Tr((S1 S2)^{1/2}) is a matrix square root, not Tr(S1 S2)":
            float(d @ d + t12 - 2 * (s1 @ s2).trace()),
        "Tr((S1 S2)^{1/2}) needs a matrix square root, not an elementwise sqrt of S1 @ S2":
            float(d @ d + t12 - 2 * (s1 @ s2).diagonal().clamp(min=0).sqrt().sum()),
        "S1 @ S2 is not symmetric, so eigvalsh/eigh on it is wrong (they read one triangle); "
        "use S1^{1/2} S2 S1^{1/2}":
            float(d @ d + t12 - 2 * torch.linalg.eigvalsh(s1 @ s2).clamp(min=0).sqrt().sum()),
    }


def _diagnose(got, want, args, tol):
    for why, v in _alternatives(*args).items():
        if abs(got - v) <= tol and abs(v - want) > tol:
            return f": {why}"
    return ""


# --------------------------------------------------------------------------
# feature_statistics
# --------------------------------------------------------------------------
def check_feature_statistics_match_torch_cov(ns):
    g = torch.Generator().manual_seed(0)
    feats = torch.randn(50, 7, generator=g, dtype=torch.float64) * torch.linspace(0.5, 3, 7) + 2.0
    mu, sigma = ns.feature_statistics(feats)
    assert tuple(mu.shape) == (7,), f"mu must be (D,) = (7,), got {tuple(mu.shape)}"
    assert tuple(sigma.shape) == (7, 7), (
        f"sigma must be the (D, D) = (7, 7) feature covariance, got {tuple(sigma.shape)} — "
        f"rows are samples, columns are features (torch.cov wants feats.T)")
    assert torch.allclose(mu.double(), feats.mean(0), atol=1e-10), "mu must be the mean over samples (dim 0)"
    want = torch.cov(feats.T)
    if torch.allclose(sigma.double(), want, atol=1e-10):
        pass
    elif torch.allclose(sigma.double(), want * 49 / 50, atol=1e-10):
        raise AssertionError("sigma divides by N; use the unbiased estimate (divide by N - 1) "
                             "like torch.cov and numpy.cov")
    elif torch.allclose(sigma.double(), feats.T @ feats / 49, atol=1e-8):
        raise AssertionError("sigma is not centred — subtract the mean before X^T X")
    else:
        raise AssertionError(f"sigma differs from torch.cov(feats.T) "
                             f"(max diff {(sigma.double() - want).abs().max():.3e})")
    feats32 = torch.randn(20, 12, generator=g)
    mu, sigma = ns.feature_statistics(feats32)
    assert torch.allclose(sigma.double(), torch.cov(feats32.double().T), atol=1e-5), \
        "float32 features must give the same covariance as torch.cov"
    few = torch.randn(5, 12, generator=g, dtype=torch.float64)
    mu, sigma = ns.feature_statistics(few)
    assert tuple(sigma.shape) == (12, 12) and torch.allclose(sigma, torch.cov(few.T), atol=1e-10), \
        "with fewer samples than features (N = 5 < D = 12) sigma is still the (D, D) covariance (rank <= N - 1)"


# --------------------------------------------------------------------------
# frechet_distance
# --------------------------------------------------------------------------
def check_identical_statistics_give_zero(ns):
    g = torch.Generator().manual_seed(1)
    feats = torch.randn(200, 32, generator=g, dtype=torch.float64) * 2 + 1
    mu, s = feats.mean(0), torch.cov(feats.T)
    d = _f(ns.frechet_distance(mu, s, mu, s))
    assert abs(d) < 1e-6, f"FID of a distribution with itself must be 0, got {d:.3e}"


def check_translation_gives_squared_mean_distance(ns):
    s = _spd(16, seed=2)
    mu1 = torch.randn(16, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    delta = torch.linspace(-1, 2, 16, dtype=torch.float64)
    d = _f(ns.frechet_distance(mu1, s, mu1 + delta, s))
    want = float(delta @ delta)
    assert abs(d - want) < 1e-6 * want, (
        f"same covariance, means shifted by delta: FID must be ||delta||^2 = {want:.6f}, got {d:.6f}"
        + (" — the mean term is the *squared* norm" if abs(d - float(delta.norm())) < 1e-6 else ""))


def check_diagonal_covariances_closed_form(ns):
    g = torch.Generator().manual_seed(4)
    a = torch.rand(10, generator=g, dtype=torch.float64) * 4 + 0.1
    b = torch.rand(10, generator=g, dtype=torch.float64) * 4 + 0.1
    mu1 = torch.randn(10, generator=g, dtype=torch.float64)
    mu2 = torch.randn(10, generator=g, dtype=torch.float64)
    args = (mu1, torch.diag(a), mu2, torch.diag(b))
    d = _f(ns.frechet_distance(*args))
    want = float(((a.sqrt() - b.sqrt()) ** 2).sum() + (mu1 - mu2) @ (mu1 - mu2))
    assert abs(d - want) < 1e-8 * max(1.0, want), (
        f"for diagonal covariances FID = sum((sqrt(a) - sqrt(b))^2) + ||mu1 - mu2||^2 = "
        f"{want:.6f}, got {d:.6f}{_diagnose(d, want, args, 1e-6)}")
    # exact zeros on the diagonal: singular covariances must be handled
    a0 = torch.tensor([0.0, 0.0, 1.0, 4.0, 2.0], dtype=torch.float64)
    b0 = torch.tensor([1.0, 0.0, 0.0, 9.0, 2.0], dtype=torch.float64)
    z = torch.zeros(5, dtype=torch.float64)
    d0 = _f(ns.frechet_distance(z, torch.diag(a0), z, torch.diag(b0)))
    assert abs(d0 - 3.0) < 1e-6, (
        f"singular diagonal covariances diag(0, 0, 1, 4, 2) vs diag(1, 0, 0, 9, 2) have "
        f"FID = 1 + 0 + 1 + 1 + 0 = 3, got {d0} — clamp tiny/negative eigenvalues to 0 "
        f"before taking square roots")


def check_symmetric_in_its_arguments(ns):
    s1, s2 = _spd(16, seed=5), _spd(16, seed=6, scale=3.0)
    g = torch.Generator().manual_seed(7)
    m1, m2 = torch.randn(16, generator=g, dtype=torch.float64), torch.randn(16, generator=g, dtype=torch.float64)
    d12 = _f(ns.frechet_distance(m1, s1, m2, s2))
    d21 = _f(ns.frechet_distance(m2, s2, m1, s1))
    assert abs(d12 - d21) < 1e-8 * max(1.0, abs(d12)), \
        f"FID must be symmetric: d(1, 2) = {d12:.8f} but d(2, 1) = {d21:.8f}"


def check_matches_reference_on_random_spd_pairs(ns):
    for dim, seed, scale in ((8, 10, 1.0), (32, 11, 5.0), (64, 12, 0.2)):
        s1, s2 = _spd(dim, seed), _spd(dim, seed + 100, scale)
        g = torch.Generator().manual_seed(seed)
        m1 = torch.randn(dim, generator=g, dtype=torch.float64)
        m2 = m1 + 0.3 * torch.randn(dim, generator=g, dtype=torch.float64)
        args = (m1, s1, m2, s2)
        d = _f(ns.frechet_distance(*args))
        want = _ref(*args)
        assert abs(d - want) < 1e-6 * max(1.0, want), (
            f"D = {dim}: FID = {d:.6f}, reference (eigenvalues of S1 S2, principal sqrt) "
            f"= {want:.6f}{_diagnose(d, want, args, 1e-6 * max(1.0, want))}")


def check_rank_deficient_covariances(ns):
    """N < D: both covariances are singular; the result must stay finite, real and >= 0."""
    g = torch.Generator().manual_seed(13)
    f1 = torch.randn(10, 32, generator=g, dtype=torch.float64)
    f2 = torch.randn(10, 32, generator=g, dtype=torch.float64) * 1.3 + 0.2
    m1, s1, m2, s2 = f1.mean(0), torch.cov(f1.T), f2.mean(0), torch.cov(f2.T)
    scale = float(s1.trace() + s2.trace())
    try:
        d = _f(ns.frechet_distance(m1, s1, m2, s2))
    except (RuntimeError, ValueError) as e:      # torch.linalg.LinAlgError is a RuntimeError
        raise AssertionError(
            f"frechet_distance failed on rank-deficient covariances ({type(e).__name__}: {e}) — "
            f"round-off makes some eigenvalues slightly negative; clamp them to 0 before sqrt "
            f"so no NaN reaches the next eigendecomposition")
    assert d == d and abs(d) != float("inf"), (
        "FID is NaN/Inf for rank-deficient covariances — round-off makes some eigenvalues "
        "slightly negative; clamp them to 0 before sqrt")
    want = _ref(m1, s1, m2, s2)
    assert abs(d - want) < 1e-6 * scale, \
        f"rank-deficient covariances: FID = {d:.6f}, reference {want:.6f}"
    same = _f(ns.frechet_distance(m1, s1, m1, s1))
    assert same > -1e-6 * scale and abs(same) < 1e-5 * scale, (
        f"a rank-deficient distribution compared with itself should give ~0 (round-off only), "
        f"got {same:.3e}")


def check_float32_inputs_are_computed_in_float64(ns):
    """Traces ~7e4, answer ~0.5: float32 arithmetic loses it, float64 does not."""
    g = torch.Generator().manual_seed(14)
    dim = 64
    q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=g, dtype=torch.float64))
    s1 = (q * torch.logspace(0, 4, dim, dtype=torch.float64)) @ q.T
    q2, _ = torch.linalg.qr(torch.randn(dim, dim, generator=g, dtype=torch.float64))
    s2 = s1 + (q2 * torch.rand(dim, generator=g, dtype=torch.float64)) @ q2.T
    s1, s2 = ((s1 + s1.T) / 2).float(), ((s2 + s2.T) / 2).float()
    mu = torch.zeros(dim)
    want = _ref(mu, s1, mu, s2)          # exact answer for these float32 inputs
    d = _f(ns.frechet_distance(mu, s1, mu, s2))
    assert abs(d - want) < 1e-4 * max(1.0, want), (
        f"FID = {d:.6f} but the exact value for these float32 inputs is {want:.6f}. The "
        f"result is a small difference of traces around 7e4, so cast mu/sigma to float64 "
        f"before the eigendecompositions")


CHECKS = [
    check_feature_statistics_match_torch_cov,
    check_identical_statistics_give_zero,
    check_translation_gives_squared_mean_distance,
    check_diagonal_covariances_closed_form,
    check_symmetric_in_its_arguments,
    check_matches_reference_on_random_spd_pairs,
    check_rank_deficient_covariances,
    check_float32_inputs_are_computed_in_float64,
]
