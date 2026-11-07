"""vector-search — chunked exact top-k, an IVF index, and recall@k.

Conventions the checks hold the solver to:

  * metric "cosine" / "dot": higher is better, results sorted descending.
    metric "l2": scores are SQUARED L2 distances, lower is better, results sorted
    ascending, never negative (the ||q||^2 - 2 q.d + ||d||^2 expansion is clamped).
  * IVF padding when a query has fewer than k candidates: index -1 and the worst
    possible score (-inf for cosine/dot, +inf for l2), always at the end of the row.
  * IVF ids are global row numbers in the order vectors were add()-ed.
  * IVF train() runs seeded Lloyd k-means and stores self.centroids; after enough
    iterations every non-empty centroid is the mean of its cluster (a fixed point).

The oracle is a float64 full sort over the complete score matrix (l2 computed from
explicit differences, not the expansion). Comparisons are tie-tolerant: an index
may differ from the oracle's only where the two scores agree to within tolerance.
A TorchFunctionMode watches every tensor exact_topk produces to verify that it
never materializes a (Q, N) score matrix when given a small chunk_size.
"""
import math

import torch
from torch.overrides import TorchFunctionMode

ENTRIES = ["exact_topk", "IVFIndex", "recall_at_k"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Exact search is one matrix multiply per database chunk: compute a (Q, chunk) "
    "score block, keep its best k, and merge that with the best k found so far "
    "(concatenate the two candidate sets and take top-k again). Remember to add the "
    "chunk's start offset to the indices.",
    "cosine = dot product of L2-normalized vectors; l2 = ||q||^2 - 2 q.d + ||d||^2, "
    "clamped at 0, and you want the SMALLEST k (topk(largest=False)). A chunk can be "
    "smaller than k, so take min(k, chunk) there.",
    "IVF: k-means (random initial centroids from the data, alternate assign / "
    "recompute the mean, keep a centroid whose cluster went empty) gives nlist "
    "centroids; add() appends each vector's global id to its nearest centroid's "
    "list; search() scores only the vectors in the nprobe nearest lists and pads "
    "with id -1 and the worst score. recall@k = |approx ∩ exact| / k averaged over "
    "queries (order does not matter, -1 never counts).",
]

METRICS = ("cosine", "dot", "l2")


# ------------------------------------------------------------------- oracle

def _full_scores(q, db, metric):
    q, db = q.double(), db.double()
    if metric == "cosine":
        q = q / q.norm(dim=1, keepdim=True)
        db = db / db.norm(dim=1, keepdim=True)
        return q @ db.T
    if metric == "dot":
        return q @ db.T
    return ((q[:, None, :] - db[None, :, :]) ** 2).sum(-1)


def _ref_topk(q, db, k, metric):
    full = _full_scores(q, db, metric)
    order = full.argsort(dim=1, descending=(metric != "l2"), stable=True)[:, :k]
    return full.gather(1, order), order, full


def _assert_matches_reference(got, q, db, k, metric, ctx, valid=None):
    """Tie-tolerant comparison of (scores, indices) against the full-sort oracle."""
    assert isinstance(got, (tuple, list)) and len(got) == 2, \
        f"{ctx}: should return (scores, indices)"
    scores, idx = got
    nq = q.shape[0]
    assert tuple(scores.shape) == (nq, k) and tuple(idx.shape) == (nq, k), (
        f"{ctx}: scores and indices should both be (Q, k) = {(nq, k)}, got "
        f"{tuple(scores.shape)} and {tuple(idx.shape)}")
    assert not idx.is_floating_point(), f"{ctx}: indices should be integer (long), got {idx.dtype}"
    ref_s, ref_i, full = _ref_topk(q, db, k, metric)
    idx = idx.long()
    assert bool(((idx >= 0) & (idx < db.shape[0])).all()), \
        f"{ctx}: indices must be database row numbers in [0, {db.shape[0]})"
    srt = idx.sort(dim=1).values
    assert bool((srt[:, 1:] != srt[:, :-1]).all()), f"{ctx}: a row contains the same id twice"
    tol = 1e-4 * max(1.0, full.abs().max().item())
    s = scores.double()
    step = s[:, 1:] - s[:, :-1]
    if metric == "l2":
        assert bool((step >= -tol).all()), (
            f"{ctx}: l2 results must be sorted by ascending distance (nearest first) — "
            f"take the k SMALLEST distances (topk(..., largest=False))")
    else:
        assert bool((step <= tol).all()), \
            f"{ctx}: {metric} results must be sorted by descending score (best first)"
    if not bool(((s - ref_s).abs() <= tol).all()):
        why = {"cosine": "cosine = dot product of L2-normalized queries AND database rows",
               "dot": "dot = raw inner product, no normalization",
               "l2": "l2 scores are SQUARED distances ||q||^2 - 2 q.d + ||d||^2 (clamped at 0), "
                     "and you want the k smallest"}[metric]
        r = int(((s - ref_s).abs() > tol).any(1).nonzero()[0])
        raise AssertionError(
            f"{ctx}: query {r} top-{k} scores {[round(v, 4) for v in s[r, :4].tolist()]}... "
            f"differ from the exact {[round(v, 4) for v in ref_s[r, :4].tolist()]}... — {why}")
    at_idx = full.gather(1, idx)
    assert bool(((s - at_idx).abs() <= tol).all()), (
        f"{ctx}: the scores are right but the ids do not point at the database rows that "
        f"have them — ids must be global row numbers (add each chunk's start offset)")
    wrong = (idx != ref_i) & ((at_idx - ref_s).abs() > tol)
    if bool(wrong.any()):
        r, c = wrong.nonzero()[0].tolist()
        raise AssertionError(
            f"{ctx}: query {r}, rank {c}: got id {idx[r, c].item()} (score "
            f"{at_idx[r, c].item():.5f}) but the exact top-{k} has id {ref_i[r, c].item()} "
            f"(score {ref_s[r, c].item():.5f})")


def _data(nq=7, n=203, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    db = torch.randn(n, d, generator=g) * (torch.rand(n, 1, generator=g) * 2 + 0.3)
    q = torch.randn(nq, d, generator=g) * (torch.rand(nq, 1, generator=g) * 2 + 0.3)
    return q, db


def _clustered(n_per=60, n_clusters=8, d=16, nq=24, seed=0, spread=0.6):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_clusters, d, generator=g) * 3
    lab = torch.arange(n_clusters).repeat_interleave(n_per)
    db = centers[lab] + spread * torch.randn(lab.numel(), d, generator=g)
    db = db[torch.randperm(db.shape[0], generator=g)]
    train = centers[lab] + spread * torch.randn(lab.numel(), d, generator=g)
    qlab = torch.randint(0, n_clusters, (nq,), generator=g)
    q = centers[qlab] + spread * torch.randn(nq, d, generator=g)
    return q, db, train


# -------------------------------------------------------------- exact_topk

def check_exact_topk_matches_full_sort(ns):
    q, db = _data()
    for metric in METRICS:
        got = ns.exact_topk(q, db, 10, metric=metric)
        _assert_matches_reference(got, q, db, 10, metric, f"exact_topk(metric={metric!r})")


def check_exact_topk_any_chunk_size(ns):
    """Chunk sizes that do not divide N=203, and chunks smaller than k=10."""
    q, db = _data(seed=1)
    for metric in METRICS:
        for cs in (1, 7, 64, 202, 203, 1000):
            ctx = f"exact_topk(metric={metric!r}, k=10, chunk_size={cs})"
            try:
                got = ns.exact_topk(q, db, 10, metric=metric, chunk_size=cs)
            except RuntimeError as e:
                raise AssertionError(
                    f"{ctx} crashed: {e} — a chunk (and the last chunk) can hold fewer "
                    f"than k rows; take min(k, chunk rows) inside a chunk")
            _assert_matches_reference(got, q, db, 10, metric, ctx)


def check_exact_topk_k_equals_database_size(ns):
    q, db = _data(nq=3, n=23, seed=2)
    for metric in METRICS:
        got = ns.exact_topk(q, db, 23, metric=metric, chunk_size=5)
        _assert_matches_reference(got, q, db, 23, metric,
                                  f"exact_topk(k=N=23, metric={metric!r}, chunk_size=5)")


def check_exact_topk_l2_clamped_and_self_match(ns):
    g = torch.Generator().manual_seed(3)
    # far from the origin: ||q||^2 + ||d||^2 ~ 3e5 while true distances are ~0.3, so
    # the expansion cancels catastrophically and self-distances come out slightly < 0
    db = 100.0 + 0.1 * torch.randn(256, 16, generator=g)
    scores, _ = ns.exact_topk(db.clone(), db, 3, metric="l2", chunk_size=64)
    assert torch.isfinite(scores).all(), "l2 distances must be finite"
    assert bool((scores >= 0).all()), (
        f"negative squared distance {scores.min().item():.3e} — the expansion "
        f"||q||^2 - 2 q.d + ||d||^2 suffers cancellation in float32; clamp it at 0")
    q2, db2 = _data(nq=6, n=50, seed=4)
    q2 = db2[[3, 8, 15, 23, 42, 49]].clone()
    s2, i2 = ns.exact_topk(q2, db2, 3, metric="l2", chunk_size=16)
    assert i2[:, 0].tolist() == [3, 8, 15, 23, 42, 49], \
        "a query that is a copy of database row r must have row r as its nearest l2 neighbour"
    assert bool((s2[:, 0].abs() < 1e-3).all()), \
        f"distance of a vector to itself should be ~0, got {s2[:, 0].tolist()}"


class _WatchShapes(TorchFunctionMode):
    """Records the largest 2-D tensor exact_topk creates that has a dim of size nq."""

    def __init__(self, nq):
        super().__init__()
        self.nq, self.biggest, self.shape = nq, 0, None

    def _scan(self, o):
        if isinstance(o, torch.Tensor):
            if o.dim() >= 2 and self.nq in o.shape and o.numel() > self.biggest:
                self.biggest, self.shape = o.numel(), tuple(o.shape)
        elif isinstance(o, (tuple, list)):
            for x in o:
                self._scan(x)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self._scan(out)
        return out


def check_exact_topk_bounded_memory(ns):
    """With chunk_size=100 over N=3000 rows no (Q, N) score matrix may be built."""
    g = torch.Generator().manual_seed(5)
    nq, n = 3, 3000
    db = torch.randn(n, 16, generator=g)
    q = torch.randn(nq, 16, generator=g)
    for metric in METRICS:
        watch = _WatchShapes(nq)
        with watch:
            got = ns.exact_topk(q, db, 5, metric=metric, chunk_size=100)
        _assert_matches_reference(got, q, db, 5, metric,
                                  f"exact_topk(metric={metric!r}, N=3000, chunk_size=100)")
        assert watch.biggest <= nq * n // 4, (
            f"metric={metric!r}: a {watch.shape} tensor was built for Q={nq}, N={n}, "
            f"chunk_size=100 — score the database chunk by chunk and keep only a "
            f"running (Q, k) top-k, never the whole (Q, N) matrix")


# --------------------------------------------------------------------- IVF

def _index(ns, nlist, nprobe, metric, train, add_batches, seed=0):
    idx = ns.IVFIndex(nlist, nprobe, metric=metric, n_iter=20, seed=seed)
    idx.train(train)
    for b in add_batches:
        idx.add(b)
    return idx


def check_ivf_full_probe_equals_exact(ns):
    """nprobe = nlist scans every list, so IVF must return exactly the exact result."""
    q, db, train = _clustered(seed=0)
    for metric in METRICS:
        index = _index(ns, 8, 8, metric, train, [db])
        got = index.search(q, 10)
        _assert_matches_reference(got, q, db, 10, metric,
                                  f"IVFIndex(nlist=8, nprobe=8, metric={metric!r}).search")


def check_ivf_ids_are_database_rows_across_adds(ns):
    """Ids are global row numbers in add() order, also over several add() calls,
    and the training set need not be the database."""
    q, db, train = _clustered(seed=1)
    for metric in ("cosine", "l2"):
        index = _index(ns, 6, 6, metric, train, [db[:100], db[100:101], db[101:]])
        _assert_matches_reference(
            index.search(q, 7), q, db, 7, metric,
            f"IVF ({metric}) after add(db[:100]); add(db[100:101]); add(db[101:]) — ids "
            f"must continue from the number of vectors already added")


def _recall(approx, exact):
    hits = (approx.unsqueeze(2) == exact.unsqueeze(1)).any(2) & (approx >= 0)
    return hits.float().sum(1).div(exact.shape[1]).mean().item()


def check_ivf_recall_non_decreasing_in_nprobe(ns):
    q, db, train = _clustered(n_per=50, n_clusters=10, seed=2, spread=2.5)
    k, nlist = 10, 16
    for metric in ("l2", "cosine"):
        exact = _ref_topk(q, db, k, metric)[1]
        index = _index(ns, nlist, 1, metric, train, [db])
        recalls = []
        for p in range(1, nlist + 1):
            index.nprobe = p
            recalls.append(_recall(index.search(q, k)[1].long(), exact))
        assert recalls[-1] > 1 - 1e-9, \
            f"{metric}: nprobe = nlist must give recall 1.0, got {recalls[-1]:.3f}"
        for p in range(1, nlist):
            assert recalls[p] >= recalls[p - 1] - 1e-9, (
                f"{metric}: recall fell from {recalls[p - 1]:.3f} (nprobe={p}) to "
                f"{recalls[p]:.3f} (nprobe={p + 1}). Probe the nprobe NEAREST centroids "
                f"(set index.nprobe is read at search time), so a larger nprobe scans a "
                f"superset of lists.")
        assert recalls[0] >= 0.5, (
            f"{metric}: recall@{k} with nprobe=1 is only {recalls[0]:.2f} on well-clustered "
            f"data — are vectors added to and queries routed to their NEAREST centroid, "
            f"and does k-means actually update the centroids?")


def check_ivf_pads_when_fewer_than_k_candidates(ns):
    g = torch.Generator().manual_seed(6)
    db = torch.randn(6, 8, generator=g)
    q = torch.randn(3, 8, generator=g)
    for metric in METRICS:
        worst = math.inf if metric == "l2" else -math.inf
        index = _index(ns, 2, 2, metric, db, [db])
        scores, ids = index.search(q, 10)
        assert tuple(ids.shape) == (3, 10) and tuple(scores.shape) == (3, 10), \
            f"{metric}: search must return (Q, k) even when only 6 vectors exist"
        assert bool((ids[:, 6:] == -1).all()), \
            f"{metric}: with 6 vectors and k=10 the last 4 ids must be -1, got {ids[0].tolist()}"
        assert bool((scores[:, 6:] == worst).all()), (
            f"{metric}: padded scores must be {worst} (the worst possible "
            f"{'distance' if metric == 'l2' else 'score'}), got {scores[0, 6:].tolist()}")
        _assert_matches_reference((scores[:, :6], ids[:, :6]), q, db, 6, metric,
                                  f"IVF ({metric}) valid part of a padded result")
        # one probed list: fewer than N candidates, padding must sit at the end
        index.nprobe = 1
        scores, ids = index.search(q, 6)
        for r in range(3):
            valid = ids[r] >= 0
            n_valid = int(valid.sum())
            assert bool(valid[:n_valid].all()) and not bool(valid[n_valid:].any()), \
                f"{metric}: padding (-1) must come after all real results, got {ids[r].tolist()}"
            assert bool((scores[r, n_valid:] == worst).all()), \
                f"{metric}: padded scores must be {worst}"
            assert bool(torch.isfinite(scores[r, :n_valid]).all()), \
                f"{metric}: real results must have finite scores"


def check_ivf_kmeans_centroids_are_cluster_means(ns):
    """After enough Lloyd iterations k-means sits at a fixed point: every non-empty
    centroid is the mean of the training vectors nearest to it. Random initial rows
    that are never updated (or one iteration only) are far from that."""
    for metric in ("l2", "cosine"):
        for seed in (0, 1):
            _, _, train = _clustered(seed=10 + seed)
            index = ns.IVFIndex(8, 1, metric=metric, n_iter=50, seed=seed)
            index.train(train)
            cent = getattr(index, "centroids", None)
            assert torch.is_tensor(cent) and tuple(cent.shape) == (8, train.shape[1]), (
                f"train() should store the k-means centroids in self.centroids as an "
                f"(nlist, D) = (8, {train.shape[1]}) tensor")
            c = cent.detach().double()
            x = train.double()
            if metric == "cosine":
                x = x / x.norm(dim=1, keepdim=True)
            assign = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)
            for j in range(8):
                members = assign == j
                if not bool(members.any()):
                    continue
                mean = x[members].mean(0)
                err = (c[j] - mean).abs().max().item()
                if metric == "cosine":   # spherical k-means (unit-norm centroids) is fine too
                    err = min(err, (c[j] - mean / mean.norm()).abs().max().item())
                assert err <= 1e-3, (
                    f"metric={metric!r}, n_iter=50: centroid {j} is {err:.3f} away from the mean "
                    f"of the {int(members.sum())} training vectors nearest to it. After train() "
                    f"k-means should have converged — each iteration assigns every vector to its "
                    f"nearest centroid and then MOVES each centroid to its cluster mean, n_iter "
                    f"times (an empty cluster keeps its centroid).")


def check_ivf_deterministic_for_a_seed(ns):
    # Unclustered data and a single Lloyd iteration: the result depends on the initial
    # centroids, so an init drawn from the global RNG shows up as a difference (on
    # well-separated clusters different inits often converge to the same partition).
    q, db = _data(nq=20, n=400, d=8, seed=3)
    built = []
    for _ in range(2):
        index = ns.IVFIndex(16, 2, metric="l2", n_iter=1, seed=7)
        index.train(db)
        index.add(db)
        built.append((index, index.search(q, 5)))
    (ia, a), (ib, b) = built
    same_centroids = torch.equal(torch.as_tensor(ia.centroids), torch.as_tensor(ib.centroids))
    assert same_centroids and torch.equal(a[1], b[1]), (
        "two indexes built with the same seed gave different centroids or results — draw the "
        "k-means init from torch.Generator().manual_seed(seed), not from the global RNG")


# ------------------------------------------------------------------ recall

def check_recall_at_k_known_values(ns):
    def r(a, e):
        out = ns.recall_at_k(torch.tensor(a), torch.tensor(e))
        return float(out)
    cases = [
        ([[1, 2, 3]], [[1, 2, 3]], 1.0, "identical results"),
        ([[3, 1, 2]], [[1, 2, 3]], 1.0, "same set, different order (recall ignores order)"),
        ([[4, 5, 6]], [[1, 2, 3]], 0.0, "disjoint"),
        ([[1, -1, -1]], [[1, 2, 3]], 1 / 3, "padding ids (-1) never count as hits"),
        ([[1, 2, 3], [9, 7, 0]], [[3, 2, 1], [7, 8, 5]], (1 + 1 / 3) / 2,
         "averaged over queries"),
    ]
    for a, e, want, what in cases:
        got = r(a, e)
        assert abs(got - want) < 1e-6, \
            f"recall_at_k({a}, {e}) should be {want:.4f} ({what}), got {got:.4f}"


CHECKS = [
    check_exact_topk_matches_full_sort,
    check_exact_topk_any_chunk_size,
    check_exact_topk_k_equals_database_size,
    check_exact_topk_l2_clamped_and_self_match,
    check_exact_topk_bounded_memory,
    check_ivf_full_probe_equals_exact,
    check_ivf_ids_are_database_rows_across_adds,
    check_ivf_recall_non_decreasing_in_nprobe,
    check_ivf_pads_when_fewer_than_k_candidates,
    check_ivf_kmeans_centroids_are_cluster_means,
    check_ivf_deterministic_for_a_seed,
    check_recall_at_k_known_values,
]
