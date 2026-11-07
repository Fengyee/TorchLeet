"""gradient-boosting — gradient-boosted regression trees for squared loss.

Boosting is deterministic given the data (no sampling), so the checks pin down the
algorithm exactly without a stored reference: one stage of depth 1 with
learning_rate=1 must equal the brute-force best regression stump (computed here
with plain loops); staged_predict()[i] must equal a model trained with i+1 trees;
the learning rate must scale the first stage's update exactly; and training MSE can
never rise from one stage to the next (each leaf moves its residuals toward their mean).
"""
import torch

ENTRIES = ["GradientBoostingRegressor"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Start from the constant that minimizes squared error, F_0 = mean(y). For squared "
    "loss the negative gradient at each point is just the residual y - F(x), so every "
    "stage fits a small regression tree to the current residuals.",
    "Write a regression tree whose splits minimize SSE_left + SSE_right and whose leaves "
    "predict the mean target. After sorting a feature, SSE_L + SSE_R = sum(r^2) - "
    "(S_L^2/n_L + S_R^2/n_R), where S_L is a prefix sum (cumsum) of the sorted targets.",
    "Update F <- F + learning_rate * tree(X) after each stage and predict "
    "F_0 + learning_rate * sum of all tree outputs. staged_predict yields that running "
    "sum after each tree; return a new tensor each time rather than mutating one in place.",
]


# ---------------------------------------------------------------- helpers

def _fit(ns, X, y, **kw):
    gb = ns.GradientBoostingRegressor(**kw)
    out = gb.fit(X, y)
    assert out is gb, "fit() must return self"
    return gb


def _pred(gb, X):
    p = torch.as_tensor(gb.predict(X)).double()
    assert tuple(p.shape) == (X.shape[0],), \
        f"predict must return (N,) = ({X.shape[0]},), got {tuple(p.shape)}"
    return p


def _staged(gb, X):
    stages = list(gb.staged_predict(X))       # materialize first, as a caller would
    return [torch.as_tensor(p).double() for p in stages]


def _mse(p, y):
    return float(((p - y.double()) ** 2).mean())


def _data(N=80, seed=0):
    """Smooth nonlinear target of 2 continuous features plus noise."""
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(N, 2, generator=g) * 4 - 2
    y = torch.sin(2 * X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.2 * torch.randn(N, generator=g)
    return X, y


def _stump_data():
    """3 features: an integer-valued one (many ties) that carries a step, two continuous."""
    g = torch.Generator().manual_seed(4)
    N = 50
    f0 = torch.randint(0, 5, (N,), generator=g).float()
    f1 = torch.randn(N, generator=g)
    f2 = torch.randn(N, generator=g)
    y = 2.0 * (f0 >= 3).float() + 0.8 * f1 + 0.3 * torch.randn(N, generator=g)
    return torch.stack([f0, f1, f2], dim=1), y


def _brute_force_stumps(X, y):
    """Every legal split as (sse, feature, threshold, gap), with plain loops."""
    y = y.double()
    out = []
    for f in range(X.shape[1]):
        vals = sorted(set(X[:, f].tolist()))
        for a, b in zip(vals[:-1], vals[1:]):
            t = (a + b) / 2
            m = X[:, f] <= t
            sse = float(((y[m] - y[m].mean()) ** 2).sum() + ((y[~m] - y[~m].mean()) ** 2).sum())
            out.append((sse, f, t, b - a))
    return sorted(out)


# ---------------------------------------------------------------- checks

def check_initial_prediction_is_mean(ns):
    X, y = _data()
    gb = _fit(ns, X, y, n_estimators=5)
    assert abs(float(gb.init_) - float(y.mean())) < 1e-5, \
        f"init_ must be mean(y) = {float(y.mean()):.5f}, got {float(gb.init_):.5f}"
    gb0 = _fit(ns, X, y, n_estimators=3, learning_rate=0.0)
    X_new = torch.rand(10, 2) * 4 - 2
    p = _pred(gb0, X_new)
    assert torch.allclose(p, torch.full_like(p, float(y.mean())), atol=1e-5), \
        "with learning_rate=0 every prediction must be the initial constant mean(y)"


def check_staged_predict_matches_shorter_models(ns):
    X, y = _data()
    gb = _fit(ns, X, y, n_estimators=6, learning_rate=0.3, max_depth=2)
    staged = _staged(gb, X)
    assert len(staged) == 6, (
        f"staged_predict must yield one prediction per stage (n_estimators=6), got "
        f"{len(staged)} — do not include the initial constant")
    assert torch.allclose(staged[-1], _pred(gb, X), atol=1e-5), \
        "the last staged prediction must equal predict(X)"
    for i in (0, 3):
        short = _fit(ns, X, y, n_estimators=i + 1, learning_rate=0.3, max_depth=2)
        assert torch.allclose(staged[i], _pred(short, X), atol=1e-5), (
            f"staged_predict()[{i}] must equal the prediction of a model trained with "
            f"{i + 1} tree(s) (yield a fresh tensor at every stage, not one mutated in place)")


def check_training_mse_never_increases(ns):
    X, y = _data(seed=1)
    gb = _fit(ns, X, y, n_estimators=40, learning_rate=0.3, max_depth=2)
    mses = [_mse(torch.full((len(y),), float(y.mean()), dtype=torch.float64), y)]
    mses += [_mse(p, y) for p in _staged(gb, X)]
    for i in range(1, len(mses)):
        assert mses[i] <= mses[i - 1] * (1 + 1e-5) + 1e-7, (
            f"training MSE rose from {mses[i - 1]:.5f} to {mses[i]:.5f} at stage {i}; with "
            f"squared loss each tree must be fit to the current residuals y - F(x)")
    assert mses[-1] < 0.5 * mses[0], \
        f"40 stages only took training MSE from {mses[0]:.4f} to {mses[-1]:.4f}"


def check_single_stump_is_the_best_regression_stump(ns):
    """n_estimators=1, learning_rate=1, max_depth=1: F = mean(y) + stump(y - mean(y))."""
    X, y = _stump_data()
    stumps = _brute_force_stumps(X, y)
    best_sse, f, t, gap = stumps[0]
    assert stumps[1][0] - best_sse > 1e-2  # spec sanity: the optimum is unique
    gb = _fit(ns, X, y, n_estimators=1, learning_rate=1.0, max_depth=1)
    p = _pred(gb, X)
    sse = float(((p - y.double()) ** 2).sum())
    assert len(torch.unique(p.round(decimals=6))) <= 2, \
        "one depth-1 tree can only produce 2 distinct predictions"
    assert abs(sse - best_sse) < 1e-4 * best_sse, (
        f"one stage of depth 1 with learning_rate=1 should be the best regression stump "
        f"(feature {f}, threshold {t:.4f}, SSE {best_sse:.4f}); yours has SSE {sse:.4f} — "
        f"are splits chosen by SSE reduction and leaves set to the mean residual?")
    m = X[:, f] <= t
    yd = y.double()
    probe = X[:6].clone()
    for side, want, x in (("left", yd[m].mean(), t - 0.25 * gap), ("right", yd[~m].mean(), t + 0.25 * gap)):
        probe[:, f] = x
        got = _pred(gb, probe)
        assert torch.allclose(got, torch.full_like(got, float(want)), atol=1e-4), (
            f"a point with feature {f} = {x:.4f} should get the {side} leaf value "
            f"{float(want):.4f} of the split at {t:.4f} (thresholds are midpoints between "
            f"consecutive distinct values; x <= threshold goes left)")


def check_learning_rate_scales_first_stage(ns):
    X, y = _data(seed=2)
    X_new = torch.rand(30, 2, generator=torch.Generator().manual_seed(9)) * 4 - 2
    full = _fit(ns, X, y, n_estimators=1, learning_rate=1.0, max_depth=3)
    part = _fit(ns, X, y, n_estimators=1, learning_rate=0.25, max_depth=3)
    base = float(y.mean())
    for Z in (X, X_new):
        step_full, step_part = _pred(full, Z) - base, _pred(part, Z) - base
        assert torch.allclose(step_part, 0.25 * step_full, atol=1e-5), (
            "with one stage, prediction - mean(y) must scale exactly with learning_rate "
            "(F_1 = F_0 + lr * h_1, and h_1 does not depend on lr)")


def check_max_depth_is_respected(ns):
    """y = x on 64 evenly spaced points: the greedy tree halves every node, so a depth-d
    tree has exactly 2^d leaves, i.e. 2^d distinct predictions."""
    X = torch.linspace(0, 1, 64).unsqueeze(1)
    y = X[:, 0].clone()
    for d in (1, 2, 3):
        gb = _fit(ns, X, y, n_estimators=1, learning_rate=1.0, max_depth=d)
        n = len(torch.unique(_pred(gb, X).round(decimals=6)))
        assert n == 2 ** d, (
            f"max_depth={d} (root = depth 0) should give a tree with {2 ** d} leaves here, "
            f"but predictions take {n} distinct values")


def check_min_samples_split(ns):
    X, y = _data()
    N = len(y)
    gb = _fit(ns, X, y, n_estimators=1, learning_rate=1.0, max_depth=2, min_samples_split=N + 1)
    p = _pred(gb, X)
    assert torch.allclose(p, torch.full_like(p, float(y.mean())), atol=1e-5), \
        f"min_samples_split={N + 1} > N: the tree must be a single leaf (predicting mean(y))"
    gb = _fit(ns, X, y, n_estimators=1, learning_rate=1.0, max_depth=1, min_samples_split=N)
    assert len(torch.unique(_pred(gb, X).round(decimals=6))) == 2, \
        f"min_samples_split={N} == N: a node with exactly min_samples_split samples may split"


def check_fits_noisy_sine(ns):
    g = torch.Generator().manual_seed(0)
    X = torch.rand(200, 1, generator=g) * 2 * torch.pi
    y = torch.sin(X[:, 0]) + 0.1 * torch.randn(200, generator=g)
    gb = _fit(ns, X, y)                                    # the defaults: 100 x depth-3, lr 0.1
    train = _mse(_pred(gb, X), y)
    grid = torch.linspace(0.2, 2 * torch.pi - 0.2, 100).unsqueeze(1)
    clean = _mse(_pred(gb, grid), torch.sin(grid[:, 0]))
    assert train < 0.015, f"train MSE {train:.4f} on a noisy sine (noise variance 0.01) — expected < 0.015"
    assert clean < 0.01, f"MSE against the clean sine curve is {clean:.4f}, expected < 0.01"


CHECKS = [check_initial_prediction_is_mean, check_staged_predict_matches_shorter_models,
          check_training_mse_never_increases, check_single_stump_is_the_best_regression_stump,
          check_learning_rate_scales_first_stage, check_max_depth_is_respected,
          check_min_samples_split, check_fits_noisy_sine]
