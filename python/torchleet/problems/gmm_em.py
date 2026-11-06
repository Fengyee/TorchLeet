"""gmm-em — Gaussian mixture model (full covariances) fitted with EM.

EM depends on its initialization, so nothing is compared against stored
parameters. The checks verify what any correct EM must satisfy: the likelihood
never decreases, a converged fit is a fixed point of the closed-form M-step,
score() agrees with a torch.distributions.MultivariateNormal reference evaluated at
the solver's own parameters, K=1 reduces to the sample mean/covariance, and on
well-separated blobs the best of a few seeds recovers the true means and weights.
"""
import itertools

import torch

ENTRIES = ["GaussianMixture"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Alternate two steps. E-step: responsibilities r[n, k] = p(component k | x_n). "
    "M-step: re-estimate weights, means and covariances as responsibility-weighted "
    "averages. Do the E-step in log space.",
    "log r[n, k] = log pi_k + log N(x_n | mu_k, Sigma_k) - logsumexp over k of the same. "
    "For log N use a Cholesky factor: Sigma = L L^T, solve L z = x - mu, then "
    "log N = -0.5 * (D log 2pi + 2 sum log diag(L) + |z|^2).",
    "With N_k = sum_n r[n, k]: pi_k = N_k / N, mu_k = sum_n r[n, k] x_n / N_k, Sigma_k = "
    "sum_n r[n, k] (x_n - mu_k)(x_n - mu_k)^T / N_k + reg_covar * I. Record the mean "
    "log-likelihood (the mean over n of the logsumexp) at init and after every M-step.",
]


# ---------------------------------------------------------------- data

_MEANS = torch.tensor([[0.0, 0.0], [7.0, 1.0], [1.0, 7.0]])
_COVS = torch.tensor([[[1.0, 0.5], [0.5, 0.8]],
                      [[0.6, -0.3], [-0.3, 1.0]],
                      [[0.5, 0.0], [0.0, 0.5]]])
_COUNTS = (150, 90, 60)


def _blobs(dtype=torch.float32, seed=0):
    """Three well-separated, differently shaped clusters with weights 0.5/0.3/0.2."""
    g = torch.Generator().manual_seed(seed)
    parts = [m + torch.randn(n, 2, generator=g) @ torch.linalg.cholesky(c).T
             for m, c, n in zip(_MEANS, _COVS, _COUNTS)]
    return torch.cat(parts).to(dtype), parts


def _overlapping(seed=1):
    """Two overlapping clusters in float64; fitting K=3 makes EM crawl for many steps."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(120, 2, generator=g, dtype=torch.float64)
    b = torch.randn(80, 2, generator=g, dtype=torch.float64) * 1.5 + torch.tensor([2.0, 1.0], dtype=torch.float64)
    return torch.cat([a, b])


def _fit(ns, X, **kw):
    gm = ns.GaussianMixture(**kw)
    out = gm.fit(X)
    assert out is gm, "fit() must return self"
    return gm


def _params(gm):
    return (torch.as_tensor(gm.weights_).double(), torch.as_tensor(gm.means_).double(),
            torch.as_tensor(gm.covariances_).double())


def _history(gm):
    return [float(v) for v in gm.log_likelihood_history_]


def _reference_score(gm, X):
    w, mu, cov = _params(gm)
    mvn = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)
    log_joint = mvn.log_prob(X.double().unsqueeze(1)) + w.log()       # (N, K)
    return float(torch.logsumexp(log_joint, dim=1).mean())


def _best_of_seeds(ns, X, K, seeds=range(5), **kw):
    """Like sklearn's n_init: keep the restart with the highest likelihood."""
    fits = [_fit(ns, X, n_components=K, seed=s, **kw) for s in seeds]
    return max(fits, key=lambda gm: float(gm.score(X)))


# ---------------------------------------------------------------- checks

def check_fitted_parameter_shapes(ns):
    X, _ = _blobs()
    gm = _fit(ns, X, n_components=3)
    w, mu, cov = _params(gm)
    assert tuple(w.shape) == (3,), f"weights_ must be (K,) = (3,), got {tuple(w.shape)}"
    assert tuple(mu.shape) == (3, 2), f"means_ must be (K, D) = (3, 2), got {tuple(mu.shape)}"
    assert tuple(cov.shape) == (3, 2, 2), \
        f"covariances_ must be full (K, D, D) = (3, 2, 2), got {tuple(cov.shape)}"
    assert abs(float(w.sum()) - 1) < 1e-5 and bool((w > 0).all()), \
        f"weights_ must be positive and sum to 1, got {w.tolist()}"
    assert torch.allclose(cov, cov.mT, atol=1e-6), "covariances_ must be symmetric"
    assert bool((torch.linalg.eigvalsh(cov) > 0).all()), "covariances_ must be positive definite"


def check_responsibilities_sum_to_one_even_far_away(ns):
    X, _ = _blobs()
    gm = _fit(ns, X, n_components=3)
    far = torch.tensor([[200.0, 200.0], [-150.0, 300.0], [1000.0, -1000.0]])
    for name, Z in (("training", X), ("far-away", far)):
        R = torch.as_tensor(gm.predict_proba(Z)).double()
        assert tuple(R.shape) == (len(Z), 3), \
            f"predict_proba must be (N, K) = ({len(Z)}, 3), got {tuple(R.shape)}"
        assert bool(torch.isfinite(R).all()), (
            f"predict_proba is NaN/inf on {name} points — every density underflows to 0 "
            f"there, so normalize in log space with torch.logsumexp")
        assert bool((R >= 0).all()) and torch.allclose(R.sum(1), torch.ones(len(Z), dtype=R.dtype), atol=1e-5), \
            f"responsibilities must be non-negative and sum to 1 over components ({name} points)"
    pred = torch.as_tensor(gm.predict(X)).long()
    R = torch.as_tensor(gm.predict_proba(X))
    assert torch.equal(pred, R.argmax(1)), "predict must return the argmax of predict_proba"


def check_log_likelihood_never_decreases(ns):
    X = _overlapping()
    for seed in (0, 1):
        gm = _fit(ns, X, n_components=3, max_iter=60, tol=0.0, seed=seed)
        h = _history(gm)
        assert len(h) >= 2, "log_likelihood_history_ should hold the init value and one entry per M-step"
        drops = [(i, h[i - 1] - h[i]) for i in range(1, len(h)) if h[i] < h[i - 1] - 1e-6]
        assert not drops, (
            f"EM must never decrease the log-likelihood, but step {drops[0][0]} dropped it "
            f"by {drops[0][1]:.2e} — check the M-step formulas (normalize by N_k, use the "
            f"new means in the covariance)")


def check_history_and_stopping_rule(ns):
    X, _ = _blobs()
    gm = _fit(ns, X, n_components=3, max_iter=2, tol=0.0)
    h = _history(gm)
    assert len(h) == 3, (
        f"with max_iter=2 the history should be [init, after M-step 1, after M-step 2], "
        f"got {len(h)} entries")
    s = float(gm.score(X))
    assert abs(h[-1] - s) < 1e-4 * max(1.0, abs(s)), (
        f"log_likelihood_history_[-1] = {h[-1]:.5f} must be the mean log-likelihood of "
        f"the final parameters, score(X) = {s:.5f}")

    tol = 1e-3
    gm = _fit(ns, X, n_components=3, max_iter=200, tol=tol)
    h = _history(gm)
    gains = [b - a for a, b in zip(h[:-1], h[1:])]
    assert 2 <= len(h) <= 201, f"history has {len(h)} entries; expected 2..max_iter+1"
    assert all(gn >= tol for gn in gains[:-1]), \
        f"fit stopped late: an earlier step already improved by less than tol={tol}"
    assert len(h) == 201 or gains[-1] < tol, \
        f"fit stopped early: the last step improved by {gains[-1]:.2e} >= tol={tol}"


def check_score_matches_multivariate_normal_reference(ns):
    X, _ = _blobs()
    gm = _fit(ns, X, n_components=3)
    g = torch.Generator().manual_seed(5)
    X_new = 4 * torch.randn(50, 2, generator=g) + 3
    for name, Z in (("training", X), ("new", X_new)):
        want = _reference_score(gm, Z)
        got = float(gm.score(Z))
        assert abs(got - want) < 1e-4 * max(1.0, abs(want)), (
            f"score on {name} data = {got:.5f}, but the mean over samples of "
            f"logsumexp_k(log pi_k + MultivariateNormal(mu_k, Sigma_k).log_prob(x)) at "
            f"your fitted parameters is {want:.5f}")


def check_single_component_is_sample_mean_and_covariance(ns):
    g = torch.Generator().manual_seed(2)
    X = torch.randn(50, 3, generator=g) @ torch.tensor([[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.3, -0.2, 0.5]]) + 2
    reg = 0.05
    gm = _fit(ns, X, n_components=1, reg_covar=reg)
    w, mu, cov = _params(gm)
    Xd = X.double()
    Xc = Xd - Xd.mean(0)
    want_cov = Xc.T @ Xc / len(X) + reg * torch.eye(3, dtype=torch.float64)
    assert torch.allclose(w, torch.ones(1, dtype=torch.float64), atol=1e-5), \
        f"K=1: weights_ must be [1.0], got {w.tolist()}"
    assert torch.allclose(mu[0], Xd.mean(0), atol=1e-4), "K=1: the mean must be the sample mean"
    if torch.allclose(cov[0], Xc.T @ Xc / (len(X) - 1) + reg * torch.eye(3, dtype=torch.float64), atol=1e-4):
        raise AssertionError("K=1: covariance divides by N - 1; the maximum-likelihood "
                             "estimate (the M-step) divides by N_k = N")
    assert torch.allclose(cov[0], want_cov, atol=1e-4), (
        "K=1: the covariance must be the biased sample covariance + reg_covar * I "
        f"(reg_covar={reg} added to the diagonal)")


def check_converged_fit_is_an_m_step_fixed_point(ns):
    X, _ = _blobs(dtype=torch.float64)
    reg = 1e-2
    # Any converged EM run is a fixed point (even a poor local optimum), so one seed is enough.
    gm = _fit(ns, X, n_components=3, tol=1e-10, max_iter=300, reg_covar=reg)
    w, mu, cov = _params(gm)
    R = torch.as_tensor(gm.predict_proba(X)).double()
    nk = R.sum(0)
    assert torch.allclose(w, nk / len(X), atol=1e-4), \
        "at convergence weights_ must equal the mean responsibility N_k / N"
    for k in range(3):
        if nk[k] < 5:
            continue
        m = (R[:, k:k + 1] * X).sum(0) / nk[k]
        d = X - m
        c = (R[:, k, None] * d).T @ d / nk[k] + reg * torch.eye(2, dtype=torch.float64)
        assert torch.allclose(mu[k], m, atol=1e-3), \
            f"at convergence means_[{k}] must be the responsibility-weighted mean of X"
        assert torch.allclose(cov[k], c, atol=1e-3), (
            f"at convergence covariances_[{k}] must be sum_n r_nk (x_n - mu_k)(x_n - mu_k)^T "
            f"/ N_k + reg_covar * I")


def check_recovers_separated_blobs(ns):
    X, parts = _blobs()
    gm = _best_of_seeds(ns, X, 3)
    w, mu, _ = _params(gm)
    true_mu = torch.stack([p.mean(0) for p in parts]).double()
    true_w = torch.tensor(_COUNTS, dtype=torch.float64) / sum(_COUNTS)
    perm = min(itertools.permutations(range(3)),
               key=lambda p: float((mu[list(p)] - true_mu).norm(dim=1).sum()))
    mu, w = mu[list(perm)], w[list(perm)]
    err = (mu - true_mu).norm(dim=1)
    assert bool((err < 0.1).all()), (
        f"best of 5 seeds should put a mean on every blob (true means "
        f"{[[round(v, 2) for v in r] for r in true_mu.tolist()]}), got "
        f"{[[round(v, 2) for v in r] for r in mu.tolist()]}")
    assert torch.allclose(w, true_w, atol=0.02), \
        f"mixture weights should be ~{true_w.tolist()}, got {[round(v, 3) for v in w.tolist()]}"


CHECKS = [check_fitted_parameter_shapes, check_responsibilities_sum_to_one_even_far_away,
          check_log_likelihood_never_decreases, check_history_and_stopping_rule,
          check_score_matches_multivariate_normal_reference,
          check_single_component_is_sample_mean_and_covariance,
          check_converged_fit_is_an_m_step_fixed_point, check_recovers_separated_blobs]
