"""pca — principal component analysis via the SVD of the centered data.

The oracle is the eigendecomposition of the unbiased sample covariance
(torch.cov + torch.linalg.eigh, in float64). Components are compared up to sign,
since v and -v are equally valid eigenvectors. explained_variance uses the N - 1
convention, which the checks can tell apart from 1/N (N = 40 here, a 2.5% gap).
"""
import torch

ENTRIES = ["pca_fit", "pca_transform", "pca_inverse_transform"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Center the data first: PCA finds directions of maximum variance around the mean, "
    "not around the origin.",
    "Take the thin SVD of the centered data, Xc = U S V^T. The rows of V^T are the "
    "principal directions, already sorted by decreasing singular value.",
    "The covariance is Xc^T Xc / (N - 1) = V diag(S^2 / (N - 1)) V^T, so the explained "
    "variances are S^2 / (N - 1). Project with (X - mean) @ components.T and reconstruct "
    "with Z @ components + mean.",
]


# ---------------------------------------------------------------- helpers

def _data(N=40, D=6, seed=0):
    """Anisotropic Gaussian with distinct, well-separated variances and an offset mean."""
    g = torch.Generator().manual_seed(seed)
    scales = torch.tensor([5.0, 3.0, 2.0, 1.2, 0.6, 0.25])[:D]
    Q, _ = torch.linalg.qr(torch.randn(D, D, generator=g))
    return torch.randn(N, D, generator=g) * scales @ Q.T + 3.0


def _eig_desc(X):
    """Eigenvalues/vectors of the unbiased covariance, largest first (float64)."""
    C = torch.cov(X.double().T)                 # correction=1, i.e. divides by N - 1
    evals, evecs = torch.linalg.eigh(C)
    return evals.flip(0), evecs.flip(1).T       # (D,), (D, D) with eigenvectors as rows


def _fit(ns, X, k):
    out = ns.pca_fit(X, k)
    assert isinstance(out, (tuple, list)) and len(out) == 3, \
        "pca_fit must return (components, explained_variance, mean)"
    comps, ev, mean = (torch.as_tensor(t) for t in out)
    D = X.shape[1]
    assert tuple(comps.shape) == (k, D), \
        f"components must be (k, D) = ({k}, {D}) with one direction per row, got {tuple(comps.shape)}"
    assert tuple(ev.shape) == (k,), f"explained_variance must be (k,) = ({k},), got {tuple(ev.shape)}"
    assert tuple(mean.shape) == (D,), f"mean must be (D,) = ({D},), got {tuple(mean.shape)}"
    return comps, ev, mean


# ---------------------------------------------------------------- checks

def check_components_are_orthonormal_rows(ns):
    for N, D, k in ((40, 6, 3), (40, 6, 6), (5, 6, 4)):
        X = _data(N, D)
        comps, _, _ = _fit(ns, X, k)
        gram = comps.double() @ comps.double().T
        err = float((gram - torch.eye(k, dtype=torch.float64)).abs().max())
        assert err < 1e-4, (
            f"components @ components.T must be the identity (N={N}, D={D}, k={k}); "
            f"max deviation {err:.2e}")


def check_mean_is_the_feature_mean(ns):
    X = _data()
    _, _, mean = _fit(ns, X, 3)
    assert torch.allclose(mean.double(), X.double().mean(0), atol=1e-5), \
        "mean must be the per-feature mean of X, shape (D,)"


def check_explained_variance_matches_covariance_eigenvalues(ns):
    for N, D, k in ((40, 6, 4), (5, 6, 4)):       # second case: fewer samples than features
        X = _data(N, D, seed=1)
        _, ev, _ = _fit(ns, X, k)
        want = _eig_desc(X)[0][:k]
        ev = ev.double()
        assert bool((ev[:-1] >= ev[1:] - 1e-6 * ev[0]).all()), \
            f"explained_variance must be sorted in decreasing order, got {ev.tolist()}"
        if torch.allclose(ev, want * (N - 1) / N, rtol=1e-3):
            raise AssertionError(
                "explained_variance is the biased (divide-by-N) variance; this problem "
                "uses the unbiased convention S^2 / (N - 1), matching torch.cov")
        assert torch.allclose(ev, want, rtol=1e-3, atol=1e-5), (
            f"explained_variance should equal the top-{k} eigenvalues of the sample "
            f"covariance {[round(v, 4) for v in want.tolist()]}, got "
            f"{[round(v, 4) for v in ev.tolist()]} (N={N}, D={D}) — did you center X?")


def check_components_match_eigenvectors_up_to_sign(ns):
    X = _data(seed=2)
    comps, _, _ = _fit(ns, X, 4)
    _, evecs = _eig_desc(X)
    cos = (comps.double() * evecs[:4]).sum(1).abs()
    bad = [i for i, c in enumerate(cos.tolist()) if c < 1 - 1e-3]
    assert not bad, (
        f"component(s) {bad} are not the matching covariance eigenvectors "
        f"(|cos| = {[round(c, 4) for c in cos.tolist()]}); expect ±v_i in order of "
        f"decreasing eigenvalue")


def check_transform_decorrelates_and_mean_maps_to_zero(ns):
    X = _data(seed=3)
    comps, ev, mean = _fit(ns, X, 4)
    Z = torch.as_tensor(ns.pca_transform(X, comps, mean)).double()
    assert tuple(Z.shape) == (X.shape[0], 4), \
        f"pca_transform must return (N, k) = ({X.shape[0]}, 4), got {tuple(Z.shape)}"
    z0 = torch.as_tensor(ns.pca_transform(mean.reshape(1, -1), comps, mean)).double()
    assert float(z0.abs().max()) < 1e-4, f"the mean must map to the origin, got {z0.tolist()}"
    cov_z = torch.cov(Z.T)
    assert torch.allclose(cov_z, torch.diag(ev.double()), atol=1e-3 * float(ev[0])), (
        "the projected data must be uncorrelated with variance explained_variance[i] "
        "along axis i (do not whiten or rescale in pca_transform)")


def check_new_data_is_centered_with_the_training_mean(ns):
    X = _data(seed=6)
    comps, _, mean = _fit(ns, X, 3)
    g = torch.Generator().manual_seed(7)
    X_new = 2 * torch.randn(7, X.shape[1], generator=g) - 1.0   # its own mean is not `mean`
    Z = torch.as_tensor(ns.pca_transform(X_new, comps, mean)).double()
    assert tuple(Z.shape) == (7, 3), \
        f"pca_transform must return (N, k) = (7, 3) for 7 new rows, got {tuple(Z.shape)}"
    want = (X_new.double() - mean.double()) @ comps.double().T
    assert torch.allclose(Z, want, atol=1e-4), (
        "pca_transform on new data must subtract the TRAINING mean passed in as `mean`, "
        "not the mean of the batch being transformed: Z = (X - mean) @ components.T")
    X_back = torch.as_tensor(ns.pca_inverse_transform(Z.to(comps.dtype), comps, mean)).double()
    assert torch.allclose(X_back, want @ comps.double() + mean.double(), atol=1e-4), \
        "pca_inverse_transform must return Z @ components + mean"


def check_reconstruction_error_equals_discarded_variance(ns):
    X = _data(seed=4)
    N = X.shape[0]
    evals = _eig_desc(X)[0]
    for k in (1, 3, 5):
        comps, _, mean = _fit(ns, X, k)
        Z = ns.pca_transform(X, comps, mean)
        X_hat = torch.as_tensor(ns.pca_inverse_transform(Z, comps, mean)).double()
        assert X_hat.shape == X.shape, \
            f"pca_inverse_transform must return (N, D) = {tuple(X.shape)}, got {tuple(X_hat.shape)}"
        err = float(((X.double() - X_hat) ** 2).sum()) / (N - 1)
        want = float(evals[k:].sum())
        assert abs(err - want) < 1e-3 * float(evals.sum()), (
            f"k={k}: sum of squared reconstruction errors / (N-1) = {err:.4f} should equal "
            f"the sum of the discarded eigenvalues {want:.4f}")


def check_full_rank_reconstructs_exactly(ns):
    X = _data(seed=5)
    comps, _, mean = _fit(ns, X, X.shape[1])
    X_hat = torch.as_tensor(ns.pca_inverse_transform(ns.pca_transform(X, comps, mean), comps, mean))
    err = float((X_hat.double() - X.double()).abs().max())
    assert err < 1e-4, f"with k = D the reconstruction must be exact, max error {err:.2e}"


CHECKS = [check_components_are_orthonormal_rows, check_mean_is_the_feature_mean,
          check_explained_variance_matches_covariance_eigenvalues,
          check_components_match_eigenvectors_up_to_sign,
          check_transform_decorrelates_and_mean_maps_to_zero,
          check_new_data_is_centered_with_the_training_mean,
          check_reconstruction_error_equals_discarded_variance,
          check_full_rank_reconstructs_exactly]
