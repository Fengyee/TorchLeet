"""classification-metrics — confusion matrix, P/R/F1, ROC-AUC, average precision, ECE.

Conventions graded here (sklearn's, unless noted):
  * confusion_matrix: (C, C) int64, rows = TRUE class, cols = PREDICTED class.
  * precision_recall_f1: 0/0 := 0. average=None -> per-class (C,) tensors;
    "micro" pools TP/FP/FN over classes (P = R = F1 = accuracy for single-label);
    "macro" is the unweighted mean of the per-class values over the classes that
    occur in y_true or y_pred (TP + FP + FN > 0) — what sklearn does with
    labels=None and torchmetrics does for multiclass macro. Macro-F1 is the mean of
    per-class F1, NOT the harmonic mean of macro-P and macro-R.
  * roc_auc: P(score of a random positive > score of a random negative), ties
    count 1/2 (Mann-Whitney U / average ranks), and must be O(n log n).
  * average_precision: sklearn's step-wise AP = sum_n (R_n - R_{n-1}) P_n over the
    DISTINCT score thresholds in decreasing order (tied scores form one step),
    R_0 = 0, no interpolation, no trapezoid.
  * expected_calibration_error: confidence = max prob, prediction = argmax,
    n_bins equal-width bins (k/n, (k+1)/n] (right-closed, so conf = 1.0 is in the
    last bin), ECE = sum_b |B_b|/N * |acc(B_b) - conf(B_b)| (Guo et al., 2017).

Oracles are brute force: explicit loops, all-pairs AUC, thresholding at every
distinct score.
"""
import torch

from torchleet.runner import Skip

ENTRIES = ["confusion_matrix", "precision_recall_f1", "roc_auc", "average_precision",
           "expected_calibration_error"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Everything except ECE falls out of counting: a confusion matrix gives per-class TP/FP/FN, and "
    "for ranking metrics one sort of the scores gives every threshold at once. Encode each "
    "(true, pred) pair as a single integer so one counting op builds the whole matrix.",
    "TP is the diagonal, FP the column sums minus the diagonal, FN the row sums minus the diagonal. "
    "For AUC, rank all scores (tied scores share their average rank); the rank-sum of the positives "
    "gives the Mann-Whitney U. For AP, sort descending, cumsum the labels, and evaluate precision and "
    "recall only at the LAST position of each run of equal scores.",
    "AUC = (sum of positive ranks - n_pos(n_pos+1)/2) / (n_pos * n_neg). AP = sum over distinct "
    "thresholds of (recall step) x precision. ECE: bin index from the max probability with "
    "torch.bucketize on the interior edges (right-closed bins), then per-bin sums of correct and "
    "confidence via bincount with weights; ECE = sum_b |sum(correct) - sum(conf)| / N.",
]


# ---------------------------------------------------------------- helpers

def _trace(fn, *args, **kwargs):
    """Run fn under a TorchFunctionMode; return (#torch calls, largest tensor produced)."""
    try:
        from torch.overrides import TorchFunctionMode
    except ImportError:
        raise Skip("needs torch.overrides.TorchFunctionMode (torch >= 2.0)")

    class _Tracer(TorchFunctionMode):
        def __init__(self):
            super().__init__()
            self.calls, self.biggest = 0, 0

        def __torch_function__(self, func, types, args=(), kwargs=None):
            self.calls += 1
            out = func(*args, **(kwargs or {}))
            stack = [out]
            while stack:
                o = stack.pop()
                if isinstance(o, torch.Tensor):
                    self.biggest = max(self.biggest, o.numel())
                elif isinstance(o, (tuple, list)):
                    stack.extend(o)
            return out

    with _Tracer() as t:
        fn(*args, **kwargs)
    return t.calls, t.biggest


def _f(x):
    return float(torch.as_tensor(x).double())


def _vec(x):
    return torch.as_tensor(x).double().reshape(-1)


def _cm_oracle(y, p, C):
    cm = [[0] * C for _ in range(C)]
    for t, q in zip(y.tolist(), p.tolist()):
        cm[t][q] += 1
    return torch.tensor(cm, dtype=torch.long)


def _prf_oracle(y, p, C):
    y, p = y.tolist(), p.tolist()
    P, R, F, present = [], [], [], []
    TP = FP = FN = 0
    for c in range(C):
        tp = sum(1 for a, b in zip(y, p) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, p) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
        TP, FP, FN = TP + tp, FP + fp, FN + fn
        P.append(tp / (tp + fp) if tp + fp else 0.0)
        R.append(tp / (tp + fn) if tp + fn else 0.0)
        F.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
        present.append(tp + fp + fn > 0)
    k = sum(present)
    macro = tuple(sum(v for v, m in zip(vals, present) if m) / k if k else 0.0 for vals in (P, R, F))
    micro = (TP / (TP + FP) if TP + FP else 0.0, TP / (TP + FN) if TP + FN else 0.0,
             2 * TP / (2 * TP + FP + FN) if TP + FP + FN else 0.0)
    return (P, R, F), macro, micro


def _auc_oracle(s, y):
    s, y = s.double(), y.bool()
    pos, neg = s[y], s[~y]
    gt = (pos[:, None] > neg[None, :]).double()
    eq = (pos[:, None] == neg[None, :]).double()
    return float((gt + 0.5 * eq).mean())


def _ap_oracle(s, y):
    s, y = s.double().tolist(), [int(v) for v in y.tolist()]
    npos = sum(y)
    ap, r_prev = 0.0, 0.0
    for t in sorted(set(s), reverse=True):
        sel = [yi for si, yi in zip(s, y) if si >= t]
        tp = sum(sel)
        prec, rec = tp / len(sel), tp / npos
        ap += (rec - r_prev) * prec
        r_prev = rec
    return ap


def _ece_oracle(probs, labels, n_bins):
    conf, pred = probs.max(dim=1)
    conf, correct = conf.tolist(), (pred == labels).tolist()
    N, ece = len(conf), 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, c in enumerate(conf) if lo < c <= hi]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
            avg_conf = sum(conf[i] for i in idx) / len(idx)
            ece += len(idx) / N * abs(acc - avg_conf)
    return ece


def _informative_scores(g, y, levels=None):
    """Scores that rank positives higher on average. With `levels`, scores take only
    `levels` + a few distinct values, built from integers so ties are exact and occur
    both within and ACROSS the two classes."""
    n = y.numel()
    u = torch.rand(n, generator=g, dtype=torch.float64)
    if levels is None:
        return u + 0.15 * y.double()
    bump = max(1, levels // 4)
    return ((u * levels).floor() + bump * y.double()) / levels


# ---------------------------------------------------------------- confusion matrix

def check_confusion_matrix_matches_loop(ns):
    got = ns.confusion_matrix(torch.tensor([0, 0, 1]), torch.tensor([1, 1, 1]), 2)
    assert torch.is_tensor(got) and got.tolist() == [[0, 2], [0, 1]], (
        f"y_true=[0,0,1], y_pred=[1,1,1] must give [[0, 2], [0, 1]] (rows = TRUE class, columns = "
        f"PREDICTED), got {got.tolist() if torch.is_tensor(got) else got!r}")
    g = torch.Generator().manual_seed(0)
    for C, n in ((5, 200), (3, 1), (7, 50)):
        y = torch.randint(0, C, (n,), generator=g)
        p = torch.randint(0, C, (n,), generator=g)
        if C == 7:
            y[y == 6] = 0                       # class 6 never occurs in y_true or y_pred
            p[p == 6] = 1
        got = ns.confusion_matrix(y, p, C)
        assert torch.is_tensor(got) and tuple(got.shape) == (C, C), \
            f"confusion_matrix(num_classes={C}) must be a ({C}, {C}) tensor, got " \
            f"{tuple(got.shape) if torch.is_tensor(got) else type(got).__name__}"
        assert got.dtype == torch.long, f"confusion_matrix should hold int64 counts, got {got.dtype}"
        assert torch.equal(got, _cm_oracle(y, p, C)), \
            f"confusion_matrix disagrees with an explicit counting loop (C={C}, n={n})"


def check_confusion_matrix_is_vectorized(ns):
    g = torch.Generator().manual_seed(1)
    small = (torch.randint(0, 4, (20,), generator=g), torch.randint(0, 4, (20,), generator=g), 4)
    big = (torch.randint(0, 4, (3000,), generator=g), torch.randint(0, 4, (3000,), generator=g), 4)
    c_small, _ = _trace(ns.confusion_matrix, *small)
    c_big, _ = _trace(ns.confusion_matrix, *big)
    assert c_big <= c_small + 5, (
        f"confusion_matrix made {c_small} torch calls for 20 samples but {c_big} for 3000 — there "
        f"is a Python loop over samples. Encode each pair as y_true * C + y_pred and count once "
        f"(torch.bincount).")


# ---------------------------------------------------------------- precision / recall / F1

def check_precision_recall_f1_matches_oracle(ns):
    g = torch.Generator().manual_seed(2)
    C = 6
    y = torch.randint(0, 5, (300,), generator=g)        # class 5 never occurs
    p = torch.randint(0, 5, (300,), generator=g)
    p[p == 4] = 3                                        # class 4 occurs but is never predicted
    (P, R, F), macro, micro = _prf_oracle(y, p, C)
    out = ns.precision_recall_f1(y, p, C, None)
    assert isinstance(out, tuple) and len(out) == 3, "return a tuple (precision, recall, f1)"
    for name, got, want in zip(("precision", "recall", "f1"), out, (P, R, F)):
        got = _vec(got)
        assert got.numel() == C, f"average=None: per-class {name} must have {C} entries, got {got.numel()}"
        assert not torch.isnan(got).any(), \
            f"per-class {name} has NaN — class 4 is never predicted and class 5 never occurs; define 0/0 = 0"
        assert torch.allclose(got, torch.tensor(want, dtype=torch.float64), atol=1e-6), \
            f"per-class {name} {[round(v, 4) for v in got.tolist()]} != expected {[round(v, 4) for v in want]}"
    macro_all_classes = tuple(sum(v) / C for v in (P, R, F))
    for avg, want in (("macro", macro), ("micro", micro)):
        got = tuple(_f(v) for v in ns.precision_recall_f1(y, p, C, avg))
        why = ""
        if avg == "macro" and all(abs(a - b) < 1e-6 for a, b in zip(got, macro_all_classes)):
            why = (" — that averages over all num_classes; average only over classes that occur in "
                   "y_true or y_pred (class 5 occurs in neither; class 4 counts, with precision 0)")
        elif avg == "macro" and abs(got[0] - want[0]) < 1e-6 and abs(got[1] - want[1]) < 1e-6:
            why = " — macro-F1 must be the mean of the per-class F1 scores"
        elif avg == "micro":
            why = " — pool TP, FP and FN over all classes first, then divide once"
        assert all(abs(a - b) < 1e-6 for a, b in zip(got, want)), (
            f"average='{avg}': expected P/R/F1 {tuple(round(v, 4) for v in want)}, got "
            f"{tuple(round(v, 4) for v in got)}" + why)


def check_macro_f1_is_mean_of_per_class_f1(ns):
    y = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])
    p = torch.tensor([0, 0, 1, 1, 1, 2, 2, 0])
    P, R, F = (_f(v) for v in ns.precision_recall_f1(y, p, 3, "macro"))
    want = (4 / 7 + 2 / 5 + 1 / 2) / 3
    assert abs(F - want) < 1e-6, (
        f"macro-F1 is the mean of per-class F1 = (0.5714 + 0.4 + 0.5) / 3 = {want:.4f}, got {F:.4f}"
        + (" — that is the harmonic mean of macro-P and macro-R, a different (and non-standard) number"
           if abs(F - 0.5) < 1e-6 else ""))
    assert abs(P - 0.5) < 1e-6 and abs(R - 0.5) < 1e-6, \
        f"macro precision and recall should both be 0.5 here, got {P:.4f}, {R:.4f}"


def check_micro_f1_equals_accuracy(ns):
    g = torch.Generator().manual_seed(3)
    for C, n in ((4, 100), (10, 257)):
        y = torch.randint(0, C, (n,), generator=g)
        p = torch.where(torch.rand(n, generator=g) < 0.6, y, torch.randint(0, C, (n,), generator=g))
        acc = float((y == p).double().mean())
        P, R, F = (_f(v) for v in ns.precision_recall_f1(y, p, C, "micro"))
        assert max(abs(P - acc), abs(R - acc), abs(F - acc)) < 1e-6, (
            f"for single-label multiclass, micro precision = recall = F1 = accuracy ({acc:.4f}); got "
            f"P={P:.4f} R={R:.4f} F1={F:.4f}. Pool TP/FP/FN over classes before dividing.")


# ---------------------------------------------------------------- ROC-AUC

def check_roc_auc_matches_pairwise_oracle(ns):
    g = torch.Generator().manual_seed(4)
    for trial, (n, levels) in enumerate(((300, None), (300, 10), (101, 3), (64, None), (500, 25))):
        y = torch.rand(n, generator=g) < 0.3
        y[0], y[1] = True, False
        s = _informative_scores(g, y, levels)
        want = _auc_oracle(s, y)
        for labels, kind in ((y.long(), "int64"), (y, "bool")):
            got = _f(ns.roc_auc(s.clone(), labels.clone()))
            assert abs(got - want) < 1e-6, (
                f"trial {trial} (n={n}, {'continuous scores' if levels is None else f'{levels} distinct score values -> many ties'}, "
                f"{kind} labels): AUC {got:.6f} != all-pairs reference {want:.6f}. A tie between a "
                f"positive and a negative counts 1/2 — give tied scores their AVERAGE rank.")


def check_roc_auc_special_cases(ns):
    y = torch.tensor([0, 0, 1, 0, 1, 1, 0, 1])
    s = torch.tensor([0.1, 0.2, 0.8, 0.3, 0.9, 0.7, 0.4, 0.6])
    assert abs(_f(ns.roc_auc(s, y)) - 1.0) < 1e-6, "a perfect ranking must give AUC 1.0"
    assert abs(_f(ns.roc_auc(-s, y))) < 1e-6, "a perfectly reversed ranking must give AUC 0.0"
    assert abs(_f(ns.roc_auc(torch.full((8,), 0.5), y)) - 0.5) < 1e-6, \
        "constant scores must give AUC 0.5 (every positive/negative pair is a tie worth 1/2)"
    g = torch.Generator().manual_seed(5)
    logits = torch.randn(200, generator=g, dtype=torch.float64)
    yy = (torch.rand(200, generator=g) < 0.4).long()
    a, b = _f(ns.roc_auc(logits, yy)), _f(ns.roc_auc(torch.sigmoid(3 * logits), yy))
    assert abs(a - b) < 1e-6, f"AUC only depends on the ranking: logits gave {a:.6f}, sigmoid(3*logits) {b:.6f}"


def _subquadratic(ns, name):
    fn = getattr(ns, name)
    g = torch.Generator().manual_seed(6)
    small_y = (torch.arange(100) % 2).long()
    big_y = (torch.arange(4000) % 2).long()
    small_s = torch.rand(100, generator=g, dtype=torch.float64)
    big_s = (torch.rand(4000, generator=g, dtype=torch.float64) * 500).floor()   # with ties
    c_small, _ = _trace(fn, small_s, small_y)
    c_big, biggest = _trace(fn, big_s, big_y)
    assert biggest <= 50 * 4000, (
        f"{name} built a tensor with {biggest:,} elements for n=4000 — that is the O(n^2) "
        f"all-pairs / all-thresholds approach. Sort once (O(n log n)) and use cumulative sums.")
    assert c_big <= c_small + 10, (
        f"{name} made {c_small} torch calls for n=100 but {c_big} for n=4000 — a Python loop over "
        f"samples or thresholds. Sort once and use cumulative sums.")


def check_roc_auc_is_n_log_n(ns):
    _subquadratic(ns, "roc_auc")


# ---------------------------------------------------------------- average precision

def check_average_precision_matches_threshold_oracle(ns):
    g = torch.Generator().manual_seed(7)
    for trial, (n, levels) in enumerate(((200, None), (200, 8), (57, 3), (300, 40), (30, None))):
        y = torch.rand(n, generator=g) < 0.35
        y[0] = True
        s = _informative_scores(g, y, levels)
        want = _ap_oracle(s, y)
        got = _f(ns.average_precision(s.clone(), y.long()))
        assert abs(got - want) < 1e-6, (
            f"trial {trial} (n={n}, {'continuous' if levels is None else f'{levels}-level (tied)'} scores): "
            f"AP {got:.6f} != reference {want:.6f} (threshold at every distinct score, "
            f"AP = sum (R_n - R_(n-1)) * P_n, no interpolation)")


def check_average_precision_properties(ns):
    y = torch.tensor([1, 0, 1, 0])
    s = torch.tensor([0.9, 0.8, 0.7, 0.6])
    got = _f(ns.average_precision(s, y))
    assert abs(got - (0.5 * 1 + 0.5 * 2 / 3)) < 1e-6, (
        f"scores .9 .8 .7 .6 with labels 1 0 1 0: AP = .5*1 + .5*(2/3) = 0.8333, got {got:.4f}"
        + (" — 0.7917 is the trapezoidal area under the PR curve; AP is the step-wise sum "
           "sum (R_n - R_(n-1)) * P_n" if abs(got - 0.7916667) < 1e-4 else ""))
    got = _f(ns.average_precision(torch.tensor([0.9, 0.8, 0.3, 0.2]), torch.tensor([1, 1, 0, 0])))
    assert abs(got - 1.0) < 1e-6, f"a perfect ranking must give AP 1.0, got {got:.4f}"
    yy = torch.tensor([1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    got = _f(ns.average_precision(torch.full((10,), 0.3), yy))
    assert abs(got - 0.2) < 1e-6, (
        f"all scores tied -> a single threshold with precision 0.2 and recall 1, so AP = 0.2 (the "
        f"positive rate); got {got:.4f}. Tied scores must be one step, not one step per sample.")
    g = torch.Generator().manual_seed(8)
    yb = (torch.rand(20000, generator=g) < 0.1).long()
    rate = float(yb.double().mean())
    got = _f(ns.average_precision(torch.rand(20000, generator=g, dtype=torch.float64), yb))
    assert abs(got - rate) < 0.02, \
        f"random scores should give AP close to the positive rate {rate:.3f}, got {got:.4f}"


def check_average_precision_is_n_log_n(ns):
    _subquadratic(ns, "average_precision")


# ---------------------------------------------------------------- ECE

def _calib_data(g, n, C, n_bins):
    """float64 softmax outputs whose confidences stay away from bin edges."""
    while True:
        logits = torch.randn(n, C, generator=g, dtype=torch.float64) * 2
        probs = torch.softmax(logits, dim=1)
        labels = torch.where(torch.rand(n, generator=g) < 0.7, probs.argmax(1),
                             torch.randint(0, C, (n,), generator=g))
        conf = probs.max(1).values * n_bins
        if float((conf - conf.round()).abs().min()) > 1e-6:
            return probs, labels


def check_ece_matches_loop_oracle(ns):
    g = torch.Generator().manual_seed(9)
    probs, labels = _calib_data(g, 500, 4, 15)
    want = _ece_oracle(probs, labels, 15)
    got = _f(ns.expected_calibration_error(probs.clone(), labels.clone()))
    assert abs(got - want) < 1e-6, \
        f"ECE with the default n_bins=15: got {got:.6f}, expected {want:.6f} (sum_b |B_b|/N * |acc - conf|)"
    for n, C, nb in ((300, 10, 10), (64, 3, 5)):
        probs, labels = _calib_data(g, n, C, nb)
        want = _ece_oracle(probs, labels, nb)
        got = _f(ns.expected_calibration_error(probs, labels, n_bins=nb))
        assert abs(got - want) < 1e-6, (
            f"ECE with n_bins={nb} (N={n}, {C} classes): got {got:.6f}, expected {want:.6f}. "
            f"Weight each bin by |B_b|/N — not a plain mean over bins.")


def check_ece_bin_edges_are_right_closed(ns):
    # conf .5 correct (+.5), conf .75 correct (+.25), conf 1.0 wrong (-1); n_bins = 4.
    # Bins (0,.25] (.25,.5] (.5,.75] (.75,1]: each sample alone -> ECE = (.5 + .25 + 1) / 3.
    probs = torch.tensor([[0.5, 0.25, 0.25], [0.75, 0.25, 0.0], [0.0, 1.0, 0.0]])
    labels = torch.tensor([0, 0, 0])
    got = _f(ns.expected_calibration_error(probs, labels, n_bins=4))
    assert abs(got - 1.75 / 3) < 1e-6, (
        f"confidences 0.5, 0.75 and 1.0 with n_bins=4 must fall in bins (.25,.5], (.5,.75], (.75,1] "
        f"-> ECE 0.5833; got {got:.4f}"
        + (" — 0.4167 means left-closed [lo, hi) bins (1.0 clamped into the last bin)"
           if abs(got - 1.25 / 3) < 1e-6 else ""))


def check_ece_extremes(ns):
    probs = torch.tensor([[0.75, 0.25]] * 4 + [[0.0, 1.0]] * 2)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    got = _f(ns.expected_calibration_error(probs, labels))
    assert abs(got) < 1e-6, \
        f"75%-confident predictions that are right 3 times out of 4 (plus certain, correct ones) are " \
        f"perfectly calibrated: ECE must be 0, got {got:.4f}"
    wrong = torch.tensor([[1.0, 0.0, 0.0]] * 5)
    got = _f(ns.expected_calibration_error(wrong, torch.tensor([1, 2, 1, 2, 1])))
    assert abs(got - 1.0) < 1e-6, f"100%-confident predictions that are all wrong give ECE 1.0, got {got:.4f}"


CHECKS = [check_confusion_matrix_matches_loop, check_confusion_matrix_is_vectorized,
          check_precision_recall_f1_matches_oracle, check_macro_f1_is_mean_of_per_class_f1,
          check_micro_f1_equals_accuracy, check_roc_auc_matches_pairwise_oracle,
          check_roc_auc_special_cases, check_roc_auc_is_n_log_n,
          check_average_precision_matches_threshold_oracle, check_average_precision_properties,
          check_average_precision_is_n_log_n, check_ece_matches_loop_oracle,
          check_ece_bin_edges_are_right_closed, check_ece_extremes]
