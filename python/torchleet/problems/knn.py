"""knn — k-nearest neighbours in PyTorch."""
import torch

ENTRIES = ["knn_predict"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "torch.cdist gives every test-to-train distance in one call.",
    "topk(..., largest=False) picks the nearest neighbours.",
    "Predict the majority label among those neighbours (torch.mode helps).",
]


def _data(seed=0):
    torch.manual_seed(seed)
    Xtr = torch.cat([torch.randn(30, 2) - 3, torch.randn(30, 2) + 3])
    ytr = torch.cat([torch.zeros(30, dtype=torch.long), torch.ones(30, dtype=torch.long)])
    return Xtr, ytr


def check_output_shape_and_dtype(ns):
    Xtr, ytr = _data()
    Xte = torch.randn(7, 2)
    out = ns.knn_predict(Xtr, ytr, Xte, k=3)
    assert tuple(out.shape) == (7,), f"expected (7,), got {tuple(out.shape)}"
    assert set(out.tolist()) <= set(ytr.tolist()), "predicted a label not present in y_train"


def check_k_of_one_returns_nearest_label(ns):
    Xtr, ytr = _data()
    Xte = Xtr[:10] + 1e-4          # essentially on top of known points
    got = ns.knn_predict(Xtr, ytr, Xte, k=1)
    assert torch.equal(got, ytr[:10]), \
        "k=1 must return the label of the single nearest training point"


def check_separable_data_is_classified(ns):
    Xtr, ytr = _data()
    Xte = torch.cat([torch.randn(20, 2) - 3, torch.randn(20, 2) + 3])
    yte = torch.cat([torch.zeros(20, dtype=torch.long), torch.ones(20, dtype=torch.long)])
    acc = (ns.knn_predict(Xtr, ytr, Xte, k=5) == yte).float().mean()
    assert float(acc) > 0.9, \
        f"only {float(acc):.0%} accuracy on well-separated clusters"


def check_majority_vote_over_k_neighbours(ns):
    """Points on a line, query at the origin. The nearest neighbour is class 0, the
    3 nearest are mostly class 1, the 5 nearest mostly class 0 — so a solution that
    ignores k, or only looks at the closest point, gets one of these wrong. No ties."""
    Xtr = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0],
                        [10.0, 0.0], [11.0, 0.0], [12.0, 0.0]])
    ytr = torch.tensor([0, 1, 1, 0, 0, 0])
    Xte = torch.zeros(1, 2)
    for k, want in ((1, 0), (3, 1), (5, 0)):
        got = int(ns.knn_predict(Xtr, ytr, Xte, k=k).reshape(-1)[0])
        assert got == want, (
            f"k={k}: the {k} nearest labels are {ytr[:k].tolist()}, so the majority "
            f"vote is {want}, got {got}")


def check_uses_euclidean_distance(ns):
    """(2, 2) is nearer the origin than (3, 0) under L2 (2.83 < 3) but farther
    under L1 (4 > 3)."""
    Xtr = torch.tensor([[3.0, 0.0], [2.0, 2.0]])
    ytr = torch.tensor([0, 1])
    got = int(ns.knn_predict(Xtr, ytr, torch.zeros(1, 2), k=1).reshape(-1)[0])
    assert got == 1, (
        "the nearest training point to the origin by Euclidean distance is (2, 2) "
        "(label 1); (3, 0) is only nearer under L1/Manhattan distance")


CHECKS = [check_output_shape_and_dtype, check_k_of_one_returns_nearest_label,
          check_separable_data_is_classified, check_majority_vote_over_k_neighbours,
          check_uses_euclidean_distance]
