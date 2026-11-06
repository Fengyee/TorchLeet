"""decision-tree — CART classification tree (Gini impurity) from scratch.

The load-bearing check builds the best depth-1 stump by brute force (every feature,
every midpoint threshold, Gini counted with plain Python loops) and requires the
solver's max_depth=1 tree to reach the same weighted Gini and to put its threshold
at the same midpoint. The tree's internals are never inspected: leaves are recovered
from predict_proba, and depth from the required get_depth() method plus a leaf count.
"""
import torch

ENTRIES = ["DecisionTreeClassifier"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Grow the tree recursively: at each node try every feature and every candidate "
    "threshold, keep the split with the lowest weighted child Gini, and stop at "
    "max_depth, below min_samples_split, or when no split strictly lowers the impurity.",
    "Sort one feature column once. A cut after sorted position i sends the first i+1 "
    "samples left, so the left class counts for every cut are a cumsum of the one-hot "
    "labels in sorted order, and the right counts are total minus left.",
    "With counts c_L and c_R, n_L*Gini_L + n_R*Gini_R = n - (|c_L|^2/n_L + |c_R|^2/n_R). "
    "Mask cuts between equal values, take the argmax over (cut, feature), and use the "
    "midpoint of the two neighbouring sorted values as the threshold (x <= t goes left).",
]


# ---------------------------------------------------------------- helpers

def _fit(ns, X, y, **kw):
    tree = ns.DecisionTreeClassifier(**kw)
    out = tree.fit(X, y)
    assert out is tree, "fit() must return self"
    return tree


def _proba(tree, X):
    P = torch.as_tensor(tree.predict_proba(X)).double()
    assert P.ndim == 2 and P.shape[0] == X.shape[0], \
        f"predict_proba should return (N, C) = ({X.shape[0]}, C), got {tuple(P.shape)}"
    return P


def _pred(tree, X):
    p = torch.as_tensor(tree.predict(X)).reshape(-1).long()
    assert p.shape[0] == X.shape[0], \
        f"predict should return (N,) = ({X.shape[0]},), got {tuple(p.shape)}"
    return p


def _depth(tree):
    d = tree.get_depth()
    try:
        return int(d)
    except (TypeError, ValueError):
        raise AssertionError(f"get_depth() must return an int, got {d!r}")


def _gini(labels, C):
    if len(labels) == 0:
        return 0.0
    p = torch.bincount(labels, minlength=C).double() / len(labels)
    return float(1 - (p ** 2).sum())


def _weighted_gini(groups, y, C):
    """groups: (N,) leaf id per sample."""
    total = 0.0
    for g in groups.unique():
        m = groups == g
        total += int(m.sum()) * _gini(y[m], C)
    return total / len(y)


def _brute_force_stumps(X, y, C):
    """Every legal split as (weighted_gini, feature, threshold, gap), loops only."""
    out = []
    for f in range(X.shape[1]):
        vals = sorted(set(X[:, f].tolist()))
        for a, b in zip(vals[:-1], vals[1:]):
            t = (a + b) / 2
            m = X[:, f] <= t
            wg = (int(m.sum()) * _gini(y[m], C) + int((~m).sum()) * _gini(y[~m], C)) / len(y)
            out.append((wg, f, t, b - a))
    return sorted(out)


def _stump_data():
    """3 classes, 3 features: one integer-valued (many ties), two continuous."""
    g = torch.Generator().manual_seed(7)
    N = 60
    y = torch.randint(0, 3, (N,), generator=g)
    f0 = (y + torch.randint(-1, 2, (N,), generator=g)).clamp(0, 3).float()  # ties galore
    f1 = 0.9 * y.float() + torch.randn(N, generator=g)
    f2 = torch.randn(N, generator=g)
    return torch.stack([f0, f1, f2], dim=1), y


def _random_label_data(N=60, seed=1, ties=True):
    """Labels independent of X. Feature 0 has heavy ties unless ties=False; feature 1
    is continuous, so every impure node has a split that strictly lowers the Gini."""
    g = torch.Generator().manual_seed(seed)
    f0 = torch.randint(0, 4, (N,), generator=g).float()
    f1 = torch.randn(N, generator=g)
    y = torch.randint(0, 3, (N,), generator=g)
    if not ties:
        f0 = f0 + 0.1 * torch.randn(N, generator=g)
    return torch.stack([f0, f1], dim=1), y


def _xor_data():
    """Four noisy corner clusters labelled XOR(x0 > 0, x1 > 0), unequal sizes.

    Sizes are chosen so the greedy root split lands in the gap between clusters; with
    many other sizes it cuts through a cluster and depth 2 is then not enough."""
    g = torch.Generator().manual_seed(3)
    corners = [((-1.0, -1.0), 0, 16), ((1.0, 1.0), 0, 8), ((-1.0, 1.0), 1, 4), ((1.0, -1.0), 1, 12)]
    Xs, ys = [], []
    for (cx, cy), label, n in corners:
        Xs.append(torch.tensor([cx, cy]) + 0.2 * torch.randn(n, 2, generator=g))
        ys.append(torch.full((n,), label))
    return torch.cat(Xs), torch.cat(ys)


# ---------------------------------------------------------------- checks

def check_fit_returns_self_and_output_shapes(ns):
    X, y = _stump_data()
    tree = _fit(ns, X, y, max_depth=3)
    P = _proba(tree, X)
    assert P.shape == (X.shape[0], 3), \
        f"predict_proba must have one column per class (C = y.max()+1 = 3), got {tuple(P.shape)}"
    _pred(tree, X)
    _depth(tree)


def check_depth_one_tree_is_the_best_stump(ns):
    """Same weighted Gini as the brute-force optimum, same leaves, same midpoint."""
    X, y = _stump_data()
    C = 3
    stumps = _brute_force_stumps(X, y, C)
    best_wg, f, t, gap = stumps[0]
    assert stumps[1][0] - best_wg > 1e-3  # spec sanity: the optimum is unique

    tree = _fit(ns, X, y, max_depth=1)
    P = _proba(tree, X)
    _, leaf = torch.unique(P, dim=0, return_inverse=True)
    n_leaves = int(leaf.max()) + 1
    assert n_leaves <= 2, f"a max_depth=1 tree has at most 2 leaves, found {n_leaves}"
    got = _weighted_gini(leaf, y, C)
    assert abs(got - best_wg) < 1e-6, (
        f"your depth-1 split has weighted Gini {got:.5f}, but the best possible split "
        f"(feature {f}, threshold {t:.4f}) has {best_wg:.5f} — is every feature and "
        f"every threshold scored, and are ties between equal values handled?")
    for g in range(n_leaves):
        m = leaf == g
        want = torch.bincount(y[m], minlength=C).double() / int(m.sum())
        assert torch.allclose(P[m][0], want, atol=1e-6), (
            f"a leaf predicts {P[m][0].tolist()} but its training samples have class "
            f"distribution {want.tolist()} — a leaf must store the class frequencies")

    # The threshold must sit at the midpoint between neighbouring training values.
    m_left = X[:, f] <= t
    p_left = torch.bincount(y[m_left], minlength=C).double() / int(m_left.sum())
    p_right = torch.bincount(y[~m_left], minlength=C).double() / int((~m_left).sum())
    probe = X[:8].clone()
    for side, want, x in (("left", p_left, t - 0.25 * gap), ("right", p_right, t + 0.25 * gap)):
        probe[:, f] = x
        got_p = _proba(tree, probe)
        assert torch.allclose(got_p, want.expand_as(got_p), atol=1e-6), (
            f"a point with feature {f} = {x:.4f} should fall in the {side} leaf of the "
            f"split at {t:.4f} (the midpoint of two consecutive distinct values; "
            f"x <= threshold goes left)")


def check_unlimited_depth_memorizes_training_data(ns):
    # Continuous features: a tree that mishandles tied values fails the (finite-depth)
    # stump / constant-feature checks instead of looping forever here.
    X, y = _random_label_data(ties=False)
    tree = _fit(ns, X, y)
    acc = float((_pred(tree, X) == y).double().mean())
    assert acc == 1.0, (
        f"with max_depth=None the tree must fit every training point (labels here are "
        f"random and no two rows conflict), got {acc:.1%} train accuracy")


def check_max_depth_is_respected(ns):
    X, y = _random_label_data()
    for d in (1, 2, 3):
        tree = _fit(ns, X, y, max_depth=d)
        depth = _depth(tree)
        assert depth == d, (
            f"max_depth={d} on random labels should grow to depth exactly {d} "
            f"(the root is depth 0), got get_depth() = {depth}")
        n_leaves = len(torch.unique(_proba(tree, X), dim=0))
        assert n_leaves <= 2 ** d, \
            f"max_depth={d} allows at most {2 ** d} leaves, but predictions show {n_leaves}"
    tree = _fit(ns, X, y, max_depth=0)
    assert _depth(tree) == 0 and len(torch.unique(_proba(tree, X), dim=0)) == 1, \
        "max_depth=0 must give a single leaf"


def check_predict_proba_is_a_distribution(ns):
    X, y = _stump_data()
    tree = _fit(ns, X, y, max_depth=3)
    g = torch.Generator().manual_seed(11)
    X_new = 2 * torch.randn(40, X.shape[1], generator=g)
    for name, Z in (("training", X), ("unseen", X_new)):
        P = _proba(tree, Z)
        assert bool(torch.isfinite(P).all()), \
            f"predict_proba returned NaN/inf on {name} data — did a split create an empty child?"
        assert bool((P >= 0).all()), f"negative probabilities on {name} data"
        assert torch.allclose(P.sum(1), torch.ones(len(Z), dtype=P.dtype), atol=1e-5), \
            f"predict_proba rows must sum to 1 on {name} data, got {P.sum(1)[:4].tolist()}..."
        top2 = P.topk(2, dim=1).values
        unique = top2[:, 0] > top2[:, 1] + 1e-9
        pred = _pred(tree, Z)
        assert torch.equal(pred[unique], P.argmax(1)[unique]), \
            f"predict must be the argmax of predict_proba ({name} data)"


def check_single_class_and_pure_data(ns):
    X, _ = _stump_data()
    y = torch.full((X.shape[0],), 2)
    tree = _fit(ns, X, y, max_depth=5)
    assert _depth(tree) == 0, "a pure node must not be split: expected depth 0"
    P = _proba(tree, X[:5])
    assert P.shape == (5, 3) and torch.allclose(P, torch.tensor([[0.0, 0.0, 1.0]]).double().expand(5, 3)), \
        f"all labels are 2, so C = 3 and every row should be [0, 0, 1], got {P[0].tolist()}"
    assert bool((_pred(tree, X) == 2).all()), "all labels are 2; predict must return 2"


def check_constant_features_make_a_leaf(ns):
    """Identical rows with different labels: no threshold exists, so the root is a leaf."""
    X = torch.ones(10, 2)
    y = torch.tensor([0, 1, 1, 0, 1, 1, 1, 0, 1, 1])
    tree = _fit(ns, X, y, max_depth=4)
    assert _depth(tree) == 0, (
        "all rows are identical, so no threshold separates them — the root must stay "
        "a leaf (only cut between *distinct* sorted values)")
    P = _proba(tree, X[:1])
    assert torch.allclose(P[0], torch.tensor([0.3, 0.7]).double(), atol=1e-6), \
        f"leaf should hold the class frequencies [0.3, 0.7], got {P[0].tolist()}"


def check_min_samples_split(ns):
    X, y = _random_label_data()
    N = len(y)
    tree = _fit(ns, X, y, max_depth=2, min_samples_split=N + 1)
    assert _depth(tree) == 0, \
        f"min_samples_split={N + 1} > N={N}: the root must not be split"
    freq = torch.bincount(y, minlength=3).double() / N
    assert torch.allclose(_proba(tree, X[:1])[0], freq, atol=1e-6), \
        "a root leaf must predict the overall class frequencies"
    tree = _fit(ns, X, y, max_depth=2, min_samples_split=N)
    assert _depth(tree) >= 1, \
        f"min_samples_split={N} == N: a node with exactly min_samples_split samples may split"


def check_xor_needs_depth_two(ns):
    X, y = _xor_data()
    acc1 = float((_pred(_fit(ns, X, y, max_depth=1), X) == y).double().mean())
    acc2 = float((_pred(_fit(ns, X, y, max_depth=2), X) == y).double().mean())
    assert acc1 <= 0.75, \
        f"no single axis-aligned split can solve XOR, but max_depth=1 scored {acc1:.1%}"
    assert acc2 == 1.0, (
        f"two greedy levels separate these XOR clusters exactly; max_depth=2 scored "
        f"{acc2:.1%}")


CHECKS = [check_fit_returns_self_and_output_shapes, check_depth_one_tree_is_the_best_stump,
          check_unlimited_depth_memorizes_training_data, check_max_depth_is_respected,
          check_predict_proba_is_a_distribution, check_single_class_and_pure_data,
          check_constant_features_make_a_leaf, check_min_samples_split,
          check_xor_needs_depth_two]
