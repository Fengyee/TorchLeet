"""kmeans — K-Means clustering.

K-Means starts from a random init, so nothing here compares against a stored
result. The checks verify the properties any converged clustering must satisfy:
every point sits with its nearest centroid, and each centroid is the mean of
its members. Lloyd's algorithm only reaches a local optimum, and roughly 30% of
random inits on three blobs end in one, so the cluster-recovery check takes the
best of several runs (by inertia) instead of trusting a single draw — otherwise a
correct solution that consumes the RNG differently would fail by bad luck.
"""
import torch

ENTRIES = ["kmeans"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Alternate two steps: assign each point to its nearest centroid, then move each centroid to the mean of its points.",
    "torch.cdist(data, centroids) gives you all the distances at once.",
    "Stop when the centroids stop moving by more than `tol`.",
]


def _blobs(k=3, n=60, d=2, spread=0.15, seed=0):
    """Well-separated clusters, so any sane initialization converges."""
    torch.manual_seed(seed)
    centers = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])[:k]
    pts = torch.cat([c + spread * torch.randn(n // k, d) for c in centers])
    return pts, centers


def check_output_shapes(ns):
    data, _ = _blobs()
    centroids, assignments = ns.kmeans(data, k=3)
    assert tuple(centroids.shape) == (3, 2), \
        f"centroids should be (k, D) = (3, 2), got {tuple(centroids.shape)}"
    assert tuple(assignments.shape) == (data.shape[0],), \
        f"assignments should be (N,) = {(data.shape[0],)}, got {tuple(assignments.shape)}"
    torch.manual_seed(0)
    data = torch.randn(40, 5)
    centroids, assignments = ns.kmeans(data, k=4)
    assert tuple(centroids.shape) == (4, 5) and tuple(assignments.shape) == (40,), (
        f"for N=40, D=5, k=4 expected centroids (4, 5) and assignments (40,), got "
        f"{tuple(centroids.shape)} and {tuple(assignments.shape)}")


def check_assignments_in_range(ns):
    data, _ = _blobs()
    _, a = ns.kmeans(data, k=3)
    assert int(a.min()) >= 0 and int(a.max()) < 3, \
        f"assignments must be in [0, 3), got [{int(a.min())}, {int(a.max())}]"


def check_points_belong_to_nearest_centroid(ns):
    """The invariant that defines a converged assignment step."""
    data, _ = _blobs()
    centroids, a = ns.kmeans(data, k=3)
    nearest = torch.cdist(data, centroids).argmin(dim=1)
    wrong = int((nearest != a).sum())
    assert wrong == 0, \
        f"{wrong} point(s) are not assigned to their nearest centroid"


def check_centroids_are_means_of_members(ns):
    data, _ = _blobs()
    centroids, a = ns.kmeans(data, k=3)
    for c in range(3):
        members = data[a == c]
        if len(members) == 0:
            continue
        assert torch.allclose(centroids[c], members.mean(0), atol=1e-3), \
            f"centroid {c} is not the mean of the points assigned to it"


def _inertia(data, centroids, a):
    return float(((data - centroids[a.long()]) ** 2).sum())


def check_recovers_separated_clusters(ns):
    """Three far-apart blobs must come back as three distinct groups — in the best
    of several restarts, since any single random init can hit a local optimum."""
    data, centers = _blobs()
    runs = [ns.kmeans(data, k=3) for _ in range(10)]
    centroids, a = min(runs, key=lambda r: _inertia(data, *r))
    assert len(set(a.tolist())) == 3, (
        f"expected 3 non-empty clusters on well-separated data, got "
        f"{len(set(a.tolist()))} (best of 10 runs)")
    for c in centers:
        assert (torch.cdist(c.unsqueeze(0), centroids).min() < 1.0), (
            f"no centroid landed near the true cluster center {c.tolist()} in any of "
            "10 runs — check the update step")


def check_empty_cluster_does_not_produce_nan(ns):
    """Four centroids over three distinct locations: two initial centroids must
    coincide, one of them wins every tie, and the other is left with no points."""
    data = torch.tensor([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]]).repeat(10, 1)
    torch.manual_seed(0)
    centroids, a = ns.kmeans(data, k=4)
    assert torch.isfinite(centroids).all(), (
        "a centroid became NaN/Inf — when a cluster has no points, keep its previous "
        "centroid (or re-seed it) instead of taking the mean of nothing")
    assert int(a.min()) >= 0 and int(a.max()) < 4, "assignments must be in [0, 4)"


CHECKS = [check_output_shapes, check_assignments_in_range,
          check_points_belong_to_nearest_centroid,
          check_centroids_are_means_of_members, check_recovers_separated_clusters,
          check_empty_cluster_does_not_produce_nan]
