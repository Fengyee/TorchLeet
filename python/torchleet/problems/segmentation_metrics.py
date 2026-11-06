"""segmentation-metrics — confusion matrix, IoU / mIoU, Dice, pixel and mean class accuracy,
and a streaming dataset-level accumulator.

Conventions graded here:
  * confusion_matrix(pred, target, C, ignore_index): (C, C) int64, ROWS = ground truth
    (target), COLUMNS = prediction (n_ij in Long et al., FCN 2015, Sec. 5; same as sklearn and
    torchmetrics). A pixel is counted iff its target is a class id in [0, C) and != ignore_index
    (ignore_index=None: only the range test). ignore_index only removes pixels by TARGET value
    (torchmetrics' _multiclass_confusion_matrix_format and MMSegmentation's
    IoUMetric.intersect_and_union do the same), so with ignore_index inside [0, C) a prediction
    of that class on a counted pixel still lands in its column. A counted pixel whose prediction
    is outside [0, C) is a caller bug: raise ValueError (preds on dropped pixels are never read).
  * per class, with TP = diag, FN = row sum - TP, FP = column sum - TP:
      IoU  = TP / (TP + FP + FN)          (Jaccard; FCN's "IU"),
      Dice = 2 TP / (2 TP + FP + FN)      (= F1 = 2 IoU / (1 + IoU)),
    NaN when the class is absent from both target and prediction (denominator 0). A class that
    is only in the target or only in the prediction scores 0, not NaN.
  * mean_iou = nanmean of IoU (absent classes excluded) — MMSegmentation's np.nanmean of
    total_area_intersect / total_area_union; the Cityscapes paper's IoU uses TP/FP/FN
    "determined over the whole test set".
  * pixel_accuracy = trace / total ("aAcc"); mean_class_accuracy = nanmean_c TP_c / rowsum_c
    (FCN's "mean accuracy", MMSegmentation's "Acc" averaged with nanmean) — a class with no
    ground-truth pixels is excluded even if predicted. Everything is NaN if no pixel is counted.
  * SegMetric sums confusion matrices over update() calls; compute() applies the formulas to the
    summed matrix (dataset-level mIoU, as MMSegmentation's IoUMetric.compute_metrics does), which
    is NOT the mean of per-image mIoUs.

Oracles are brute-force Python loops over pixels / classes, written below.
"""
import math

import torch

from torchleet.runner import Skip

ENTRIES = ["confusion_matrix", "iou_per_class", "mean_iou", "dice_per_class", "pixel_accuracy",
           "mean_class_accuracy", "SegMetric"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Every metric here is a function of one C x C matrix of pixel counts. Build it once: flatten, "
    "keep only the pixels whose target is a real class (in range and not ignore_index), and turn "
    "each (target, pred) pair into ONE integer so a single counting op fills the whole matrix. "
    "After that, every metric is a combination of the diagonal, the row sums and the column sums.",
    "index = target * C + pred, counted with torch.bincount(index, minlength=C * C) and reshaped "
    "to (C, C) so rows are targets; cast to int64 first (uint8 PNG masks overflow). TP = diag, "
    "FN = row sum - TP, FP = column sum - TP; IoU = TP / (TP + FP + FN), Dice = 2TP / (2TP + FP + FN). "
    "Divide in float64 and let 0 / 0 become NaN, then average with torch.nanmean.",
    "keep = (target >= 0) & (target < C) & (target != ignore_index); if pred[keep] has anything "
    "outside [0, C) raise ValueError; cm = bincount(C * t + p, minlength=C*C).view(C, C). "
    "iou = diag / (rows + cols - diag); mean_iou = nanmean(iou); pixel accuracy = diag.sum() / "
    "cm.sum(); mean class accuracy = nanmean(diag / rows). SegMetric holds a (C, C) int64 buffer, "
    "adds each batch's matrix in update(), and compute() runs the metric functions on the sum.",
]

_NAN = float("nan")


# ---------------------------------------------------------------- oracles

def _cm_loop(pred, target, C, ignore_index):
    cm = [[0] * C for _ in range(C)]
    for p, t in zip(pred.reshape(-1).tolist(), target.reshape(-1).tolist()):
        if ignore_index is not None and t == ignore_index:
            continue
        if not 0 <= t < C:
            continue
        cm[t][p] += 1
    return torch.tensor(cm, dtype=torch.long).reshape(C, C)


def _metrics_loop(cm):
    cm = cm.tolist()
    C = len(cm)
    rows = [sum(cm[i]) for i in range(C)]
    cols = [sum(cm[j][i] for j in range(C)) for i in range(C)]
    iou, dice, acc = [], [], []
    for i in range(C):
        tp = cm[i][i]
        fp, fn = cols[i] - tp, rows[i] - tp
        iou.append(tp / (tp + fp + fn) if tp + fp + fn else _NAN)
        dice.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else _NAN)
        acc.append(tp / rows[i] if rows[i] else _NAN)

    def nanmean(v):
        v = [x for x in v if x == x]
        return sum(v) / len(v) if v else _NAN

    total = sum(rows)
    return {"iou": torch.tensor(iou, dtype=torch.float64), "mean_iou": nanmean(iou),
            "dice": torch.tensor(dice, dtype=torch.float64), "mean_dice": nanmean(dice),
            "pixel_accuracy": sum(cm[i][i] for i in range(C)) / total if total else _NAN,
            "mean_class_accuracy": nanmean(acc)}


# ---------------------------------------------------------------- helpers

def _labels(g, shape, C, p_ignore=0.0, ignore=255, dtype=torch.long):
    t = torch.randint(0, C, shape, generator=g)
    if p_ignore:
        t = torch.where(torch.rand(shape, generator=g) < p_ignore, torch.full_like(t, ignore), t)
    return t.to(dtype)


def _noisy_pred(g, target, C, p_correct=0.6):
    rnd = torch.randint(0, C, target.shape, generator=g)
    valid = (target >= 0) & (target < C)
    keep = (torch.rand(target.shape, generator=g) < p_correct) & valid
    return torch.where(keep, target.long().clamp(0, C - 1), rnd)


def _as_cm(got, C, what):
    assert torch.is_tensor(got), f"{what} must return a tensor, got {type(got).__name__}"
    assert tuple(got.shape) == (C, C), f"{what} must return a ({C}, {C}) matrix, got {tuple(got.shape)}"
    assert got.dtype == torch.long, f"{what} must return int64 counts, got {got.dtype}"
    return got


def _call_cm(ns, pred, target, C, ignore_index=255, what=None):
    p0, t0 = pred.clone(), target.clone()
    got = ns.confusion_matrix(pred, target, C, ignore_index=ignore_index)
    assert torch.equal(pred, p0) and torch.equal(target, t0), "confusion_matrix must not modify its inputs"
    return _as_cm(got, C, what or "confusion_matrix")


def _cm_mismatch_msg(got, want, pred, target, C, ignore_index, what):
    hint = ""
    if torch.equal(got, want.T):
        hint = (" Your matrix is the TRANSPOSE of the expected one: rows must be the ground truth "
                "(target) and columns the prediction — careful, the arguments come as (pred, target).")
    elif ignore_index is not None and torch.equal(got, _cm_loop(pred.clamp(0, C - 1),
                                                                target.clamp(0, C - 1), C, None)):
        hint = " Ignored / out-of-range target pixels were counted; drop them before counting."
    elif int(got.sum()) != int(want.sum()):
        hint = (f" You counted {int(got.sum())} pixels but {int(want.sum())} are valid (target in "
                f"[0, {C}) and != ignore_index={ignore_index}).")
    i, j = divmod(int((got - want).abs().argmax()), C)
    return (f"{what}: confusion matrix differs from a pixel loop at [target={i}, pred={j}]: "
            f"expected {int(want[i, j])}, got {int(got[i, j])}.{hint}")


def _close(got, want, tol=1e-6):
    """Elementwise equality with NaN == NaN."""
    got = torch.as_tensor(got, dtype=torch.float64).reshape(-1)
    want = torch.as_tensor(want, dtype=torch.float64).reshape(-1)
    if got.shape != want.shape:
        return False
    nan_g, nan_w = torch.isnan(got), torch.isnan(want)
    if not torch.equal(nan_g, nan_w):
        return False
    return bool(((got - want).abs()[~nan_w] <= tol).all())


def _scalar(x, what):
    if torch.is_tensor(x):
        assert x.numel() == 1, f"{what} must be a scalar (0-dim tensor or float), got shape {tuple(x.shape)}"
        return float(x.reshape(-1)[0])
    assert isinstance(x, (float, int)), f"{what} must return a scalar, got {type(x).__name__}"
    return float(x)


def _random_cm(g, C, absent=(), pred_only=(), gt_only=()):
    cm = torch.randint(0, 40, (C, C), generator=g)
    cm += torch.diag(torch.randint(0, 200, (C,), generator=g))
    for c in absent:
        cm[c, :] = 0
        cm[:, c] = 0
    for c in pred_only:   # predicted somewhere, never in the ground truth
        cm[c, :] = 0
        cm[(c + 1) % C, c] = 7
    for c in gt_only:     # in the ground truth, never predicted
        cm[:, c] = 0
        cm[c, (c + 1) % C] = 5
    return cm


# ---------------------------------------------------------------- confusion matrix

def check_confusion_matrix_matches_pixel_loop(ns):
    g = torch.Generator().manual_seed(0)
    for C, shape in ((2, (64,)), (5, (3, 17, 23)), (21, (40, 50)), (7, (2, 3, 9, 11))):
        target = _labels(g, shape, C, p_ignore=0.15)
        pred = _noisy_pred(g, target, C)
        want = _cm_loop(pred, target, C, 255)
        got = _call_cm(ns, pred, target, C)
        if not torch.equal(got, want):
            raise AssertionError(_cm_mismatch_msg(got, want, pred, target, C, 255,
                                                  f"C={C}, label maps of shape {shape}"))


def check_confusion_matrix_rows_are_ground_truth(ns):
    # 3 pixels of class 0 predicted as 1, 1 pixel of class 1 predicted as 1.
    target = torch.tensor([0, 0, 0, 1])
    pred = torch.tensor([1, 1, 1, 1])
    got = _call_cm(ns, pred, target, 2)
    want = torch.tensor([[0, 3], [0, 1]])
    assert torch.equal(got, want), (
        f"target=[0,0,0,1], pred=[1,1,1,1] must give [[0, 3], [0, 1]] (row = true class, column = "
        f"predicted class), got {got.tolist()}"
        + (" — that is the transpose; the call is confusion_matrix(pred, target, ...)"
           if torch.equal(got, want.T) else ""))


def check_confusion_matrix_ignore_index_variants(ns):
    g = torch.Generator().manual_seed(1)
    C = 6
    target = _labels(g, (30, 40), C, p_ignore=0.1)
    pred = _noisy_pred(g, target, C)
    # ignore_index inside [0, C): only target == 0 pixels are dropped; predictions of 0 elsewhere count.
    for ign in (0, 3):
        got = _call_cm(ns, pred, target, C, ignore_index=ign)
        want = _cm_loop(pred, target, C, ign)
        if not torch.equal(got, want):
            extra = ""
            if (int(got[:, ign].sum()) == 0 and int(want[:, ign].sum()) > 0
                    and not torch.equal(got, want.T)):
                extra = (f" Column {ign} is empty: ignore_index only drops pixels whose TARGET is "
                         f"{ign}; a prediction of class {ign} on another pixel is still an error to count.")
            raise AssertionError(_cm_mismatch_msg(got, want, pred, target, C, ign,
                                                  f"ignore_index={ign} (a valid class id)") + extra)
        assert int(got[ign].sum()) == 0, f"row {ign} must be empty when ignore_index={ign}"
    # ignore_index=None: nothing is dropped by value; target 255 is still out of range -> dropped.
    got = _call_cm(ns, pred, target, C, ignore_index=None)
    want = _cm_loop(pred, target, C, None)
    if not torch.equal(got, want):
        raise AssertionError(_cm_mismatch_msg(got, want, pred, target, C, None, "ignore_index=None"))
    # ignore_index not present in the data -> every pixel counts
    full = _labels(g, (10, 10), C)
    got = _call_cm(ns, _noisy_pred(g, full, C), full, C, ignore_index=255)
    assert int(got.sum()) == 100, f"no pixel carries ignore_index, so all 100 must be counted, got {int(got.sum())}"
    # negative ignore_index (CrossEntropyLoss uses -100)
    t = full.clone()
    t[0] = -100
    got = _call_cm(ns, full, t, C, ignore_index=-100)
    assert int(got.sum()) == 90, f"ignore_index=-100: 10 pixels are ignored, so 90 count, got {int(got.sum())}"


def check_confusion_matrix_out_of_range_labels(ns):
    C = 4
    target = torch.tensor([0, 1, 2, 3, -1, 4, 254, 255, 2, 1])
    pred = torch.tensor([0, 1, 1, 3, 2, 0, 3, 255, 2, 1])   # pred 255 only where target is void
    try:
        got = _call_cm(ns, pred, target, C, ignore_index=255)
    except ValueError as e:
        raise AssertionError(
            f"confusion_matrix raised ValueError ({e}) although the only out-of-range prediction "
            f"(255) sits on a pixel whose target is ignore_index. Predictions on dropped pixels "
            f"are never counted, so drop first and validate only the kept predictions.") from None
    want = _cm_loop(pred, target, C, 255)
    if not torch.equal(got, want):
        raise AssertionError(
            _cm_mismatch_msg(got, want, pred, target, C, 255, "targets -1, 4 and 254 with C=4")
            + " Target values outside [0, C) are void and must be dropped like ignore_index "
              "(otherwise target * C + pred lands in the wrong cell or out of the matrix).")
    for bad, where in ((torch.tensor([0, 1, 4, 3]), "4 (== num_classes)"),
                       (torch.tensor([0, -1, 2, 3]), "-1")):
        try:
            ns.confusion_matrix(bad, torch.tensor([0, 1, 2, 3]), C, ignore_index=255)
        except ValueError:
            continue
        except Exception as e:  # noqa: BLE001
            raise AssertionError(
                f"a prediction of {where} on a counted pixel must raise ValueError (checked "
                f"explicitly), got {type(e).__name__}: {e}")
        raise AssertionError(
            f"a prediction of {where} on a counted pixel must raise ValueError — with "
            f"index = target * C + pred it would silently land in another class's cell")


def check_confusion_matrix_dtypes_and_empty_inputs(ns):
    g = torch.Generator().manual_seed(2)
    C = 19   # Cityscapes: 18 * 19 + 18 = 360 does not fit in uint8
    target = _labels(g, (2, 24, 32), C, p_ignore=0.1)
    pred = _noisy_pred(g, target, C)
    want = _cm_loop(pred, target, C, 255)
    got = _call_cm(ns, pred, target, C)
    if not torch.equal(got, want):
        raise AssertionError(_cm_mismatch_msg(got, want, pred, target, C, 255, f"C={C}, int64 inputs"))
    for dt in (torch.uint8, torch.int32, torch.int16):
        got = _call_cm(ns, pred.to(dt), target.to(dt), C, what=f"confusion_matrix on {dt} inputs")
        assert torch.equal(got, want), (
            f"{dt} label maps (C={C}) give the wrong counts — cast to int64 BEFORE computing "
            f"target * C + pred (in uint8, 18 * 19 + 18 = 360 wraps around to 104)")
    empty = _call_cm(ns, torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long), C,
                     what="confusion_matrix on empty inputs")
    assert int(empty.sum()) == 0, "empty inputs must give an all-zero matrix"
    allign = _call_cm(ns, torch.zeros(5, 5, dtype=torch.long), torch.full((5, 5), 255), C)
    assert int(allign.sum()) == 0, "if every pixel is ignored the matrix must be all zeros"


def _count_torch_calls(fn, *args, **kwargs):
    try:
        from torch.overrides import TorchFunctionMode
    except ImportError:
        raise Skip("needs torch.overrides.TorchFunctionMode (torch >= 2.0)")

    class _Counter(TorchFunctionMode):
        def __init__(self):
            super().__init__()
            self.n = 0

        def __torch_function__(self, func, types, args=(), kwargs=None):
            self.n += 1
            return func(*args, **(kwargs or {}))

    with _Counter() as c:
        fn(*args, **kwargs)
    return c.n


def check_confusion_matrix_is_vectorized(ns):
    g = torch.Generator().manual_seed(3)
    t_small = _labels(g, (8, 8), 3, p_ignore=0.1)
    p_small = _noisy_pred(g, t_small, 3)
    t_big = _labels(g, (4, 60, 80), 50, p_ignore=0.1)
    p_big = _noisy_pred(g, t_big, 50)
    n_small = _count_torch_calls(ns.confusion_matrix, p_small, t_small, 3, ignore_index=255)
    n_big = _count_torch_calls(ns.confusion_matrix, p_big, t_big, 50, ignore_index=255)
    assert n_big <= n_small + 5, (
        f"confusion_matrix made {n_small} torch calls for 64 pixels / 3 classes but {n_big} for "
        f"19200 pixels / 50 classes — the work grows with a Python loop over classes or pixels. "
        f"Encode each pair as target * C + pred and count them all with one torch.bincount.")


# ---------------------------------------------------------------- metric functions

def _cm_cases():
    g = torch.Generator().manual_seed(4)
    return [
        _random_cm(g, 5),
        _random_cm(g, 6, absent=(2,)),
        _random_cm(g, 7, absent=(0, 6), pred_only=(3,), gt_only=(4,)),
        torch.tensor([[50, 0], [0, 0]]),            # class 1 absent everywhere
        torch.tensor([[0, 9], [4, 0]]),             # everything wrong: IoU 0, not NaN
        torch.diag(torch.tensor([3, 0, 8, 1])),     # perfect, class 1 absent
    ]


def check_iou_and_dice_per_class(ns):
    for k, cm in enumerate(_cm_cases()):
        want = _metrics_loop(cm)
        got_iou = ns.iou_per_class(cm.clone())
        got_dice = ns.dice_per_class(cm.clone())
        for name, got, w in (("iou_per_class", got_iou, want["iou"]), ("dice_per_class", got_dice, want["dice"])):
            assert torch.is_tensor(got) and tuple(got.shape) == (cm.shape[0],), (
                f"{name} must return a ({cm.shape[0]},) tensor, got "
                f"{tuple(got.shape) if torch.is_tensor(got) else type(got).__name__}")
            assert got.is_floating_point(), f"{name} must return floats (use float64), got {got.dtype}"
            if _close(got, w):
                continue
            g64 = got.double()
            hint = ""
            absent = torch.isnan(w)
            if absent.any() and not torch.isnan(g64[absent]).all():
                v = g64[absent][~torch.isnan(g64[absent])][0].item()
                hint = (f" A class absent from both target and prediction (TP + FP + FN = 0) must be "
                        f"NaN, got {v:g} — treating it as {'a perfect 1' if v == 1 else v} biases "
                        f"the mean.")
            elif name == "dice_per_class" and _close(got, want["iou"]):
                hint = " That is the IoU: Dice = 2TP / (2TP + FP + FN) = 2 IoU / (1 + IoU)."
            elif name == "iou_per_class":
                d = cm.diag().double()
                if _close(got, d / (cm.sum(0) + cm.sum(1)).double()):
                    hint = " The union is rows + cols - TP (TP is in both sums once)."
                elif _close(got, d / cm.sum(1).double()):
                    hint = " That is per-class recall (TP / row sum), not IoU."
            raise AssertionError(
                f"{name} on case {k} (C={cm.shape[0]}): expected {[round(x, 4) for x in w.tolist()]}, "
                f"got {[round(x, 4) for x in g64.tolist()]}.{hint}")
        # a class only in the target or only in the prediction scores 0, not NaN
    cm = _random_cm(torch.Generator().manual_seed(5), 5, pred_only=(1,), gt_only=(3,))
    iou = ns.iou_per_class(cm).double()
    assert iou[1].item() == 0.0 and iou[3].item() == 0.0, (
        f"a class that is only predicted (class 1) or only in the ground truth (class 3) is a pure "
        f"error and has IoU 0, not NaN; got {iou[1].item()} and {iou[3].item()}")


def check_mean_iou_excludes_absent_classes(ns):
    for k, cm in enumerate(_cm_cases()):
        want = _metrics_loop(cm)["mean_iou"]
        got = _scalar(ns.mean_iou(cm.clone()), "mean_iou")
        if abs(got - want) <= 1e-6:
            continue
        iou = _metrics_loop(cm)["iou"]
        hint = ""
        if abs(got - torch.nan_to_num(iou, nan=0.0).mean().item()) <= 1e-6:
            hint = " You counted absent classes as IoU 0; skip them (torch.nanmean)."
        elif abs(got - torch.nan_to_num(iou, nan=1.0).mean().item()) <= 1e-6:
            hint = " You counted absent classes as IoU 1; skip them (torch.nanmean)."
        raise AssertionError(f"mean_iou on case {k}: expected {want:.6f}, got {got:.6f}.{hint}")
    got = _scalar(ns.mean_iou(torch.zeros(3, 3, dtype=torch.long)), "mean_iou")
    assert math.isnan(got), f"with no counted pixels every class is absent, so mean_iou must be NaN, got {got}"


def check_pixel_and_mean_class_accuracy(ns):
    for k, cm in enumerate(_cm_cases()):
        want = _metrics_loop(cm)
        pa = _scalar(ns.pixel_accuracy(cm.clone()), "pixel_accuracy")
        assert abs(pa - want["pixel_accuracy"]) <= 1e-6, (
            f"pixel_accuracy on case {k}: expected {want['pixel_accuracy']:.6f} (trace / total), got {pa:.6f}")
        mca = _scalar(ns.mean_class_accuracy(cm.clone()), "mean_class_accuracy")
        if abs(mca - want["mean_class_accuracy"]) <= 1e-6:
            continue
        d, rows, cols = cm.diag().double(), cm.sum(1).double(), cm.sum(0).double()
        hint = ""
        if abs(mca - torch.nanmean(d / cols).item()) <= 1e-6:
            hint = (" That is mean precision (TP / column sum). Rows are the ground truth: class "
                    "accuracy is TP / row sum.")
        elif abs(mca - torch.nan_to_num(d / rows, nan=0.0).mean().item()) <= 1e-6:
            hint = " Classes with no ground-truth pixels must be skipped (NaN + nanmean), not counted as 0."
        elif abs(mca - torch.nan_to_num(d / rows, nan=0.0)[(rows + cols) > 0].mean().item()) <= 1e-6:
            hint = (" A class that is predicted but has no ground-truth pixels must be skipped, not "
                    "scored 0 (that is torchmetrics' macro accuracy; FCN's and MMSegmentation's "
                    "mean accuracy is the nanmean of TP / row sum).")
        raise AssertionError(
            f"mean_class_accuracy on case {k}: expected {want['mean_class_accuracy']:.6f}, got {mca:.6f}.{hint}")
    z = torch.zeros(4, 4, dtype=torch.long)
    for name in ("pixel_accuracy", "mean_class_accuracy"):
        v = _scalar(getattr(ns, name)(z), name)
        assert math.isnan(v), f"{name} of an empty confusion matrix is 0/0 and must be NaN, got {v}"


# ---------------------------------------------------------------- streaming accumulator

_KEYS = ("confusion_matrix", "iou", "mean_iou", "dice", "mean_dice", "pixel_accuracy",
         "mean_class_accuracy")


def _dataset(seed=6, C=4):
    """Images of different sizes where per-image averaging and dataset-level pooling disagree:
    class 3 is tiny and missed in most images, large and found in one."""
    g = torch.Generator().manual_seed(seed)
    imgs = []
    for k, (h, w) in enumerate(((20, 30), (12, 12), (25, 18), (8, 40), (30, 30), (16, 16))):
        t = torch.randint(0, 3, (h, w), generator=g)
        t[torch.rand(h, w, generator=g) < 0.08] = 255
        if k == 4:
            t[5:25, 5:25] = 3                        # a big class-3 region, segmented well
            p = _noisy_pred(g, t, C, p_correct=0.9)
            p[5:25, 5:25] = 3
        else:
            t[0, 0] = 3                              # a single class-3 pixel, missed
            p = _noisy_pred(g, t, C, p_correct=0.75)
            p[0, 0] = 0
            p[p == 3] = 1                            # never predict 3 here
        imgs.append((p, t))
    return imgs, C


def _compare_result(res, want_cm, what):
    assert isinstance(res, dict), f"compute() must return a dict, got {type(res).__name__}"
    missing = [k for k in _KEYS if k not in res]
    assert not missing, f"compute() result is missing keys {missing} (expected {list(_KEYS)})"
    cm = res["confusion_matrix"]
    assert torch.is_tensor(cm) and cm.dtype == torch.long, (
        f"{what}: compute()['confusion_matrix'] must be an int64 count matrix, got "
        f"{cm.dtype if torch.is_tensor(cm) else type(cm).__name__} — float32 counts stop being "
        f"exact above 2**24 = 16.8M, and one full-size Cityscapes image already has 2.1M pixels")
    assert torch.equal(cm, want_cm), (
        f"{what}: compute()['confusion_matrix'] must be the SUM of the per-batch confusion "
        f"matrices (built with the constructor's num_classes and ignore_index); "
        f"total {int(want_cm.sum())} pixels expected, got "
        f"{int(cm.sum()) if torch.is_tensor(cm) else type(cm).__name__}")
    want = _metrics_loop(want_cm)
    for key in ("iou", "dice"):
        assert _close(res[key], want[key]), (
            f"{what}: compute()['{key}'] = {[round(x, 4) for x in torch.as_tensor(res[key]).tolist()]}, "
            f"expected {[round(x, 4) for x in want[key].tolist()]} (computed on the summed matrix)")
    for key in ("mean_iou", "mean_dice", "pixel_accuracy", "mean_class_accuracy"):
        got = _scalar(res[key], f"compute()['{key}']")
        assert abs(got - want[key]) <= 1e-6, f"{what}: compute()['{key}'] = {got:.6f}, expected {want[key]:.6f}"


def check_segmetric_is_dataset_level(ns):
    imgs, C = _dataset()
    want_cm = sum(_cm_loop(p, t, C, 255) for p, t in imgs)
    per_image = [_metrics_loop(_cm_loop(p, t, C, 255))["mean_iou"] for p, t in imgs]
    per_image_mean = sum(per_image) / len(per_image)
    dataset_miou = _metrics_loop(want_cm)["mean_iou"]
    assert abs(per_image_mean - dataset_miou) > 0.05  # the data must tell the two apart

    m = ns.SegMetric(C, ignore_index=255)
    for p, t in imgs:
        m.update(p, t)
    res = m.compute()
    got = _scalar(res["mean_iou"], "compute()['mean_iou']") if isinstance(res, dict) and "mean_iou" in res else None
    if got is not None and abs(got - per_image_mean) <= 1e-6:
        raise AssertionError(
            f"SegMetric returned mIoU {got:.4f}, the AVERAGE OF PER-IMAGE mIoUs. The standard "
            f"(Cityscapes / ADE20K / MMSegmentation) mIoU sums the confusion matrices over the "
            f"whole dataset first and computes IoU once: expected {dataset_miou:.4f}")
    _compare_result(res, want_cm, "6 images, one update per image")

    # batching must not matter: same images, other groupings (flattened, mixed shapes)
    m2 = ns.SegMetric(C, ignore_index=255)
    m2.update(torch.cat([imgs[0][0].reshape(-1), imgs[1][0].reshape(-1)]),
              torch.cat([imgs[0][1].reshape(-1), imgs[1][1].reshape(-1)]))
    for p, t in imgs[2:]:
        m2.update(p.unsqueeze(0), t.unsqueeze(0))
    _compare_result(m2.compute(), want_cm, "same images, different batching")


def check_segmetric_state_handling(ns):
    imgs, C = _dataset(seed=7)
    m = ns.SegMetric(C, ignore_index=255)
    m.update(*imgs[0])
    first = m.compute()
    again = m.compute()
    assert torch.equal(torch.as_tensor(first["confusion_matrix"]), torch.as_tensor(again["confusion_matrix"])), \
        "calling compute() twice must give the same result (compute must not modify the state)"
    snapshot = torch.as_tensor(first["confusion_matrix"]).clone()
    m.update(*imgs[1])
    assert torch.equal(torch.as_tensor(first["confusion_matrix"]), snapshot), (
        "a later update() changed the confusion matrix returned by an earlier compute(): "
        "compute() must return a copy (e.g. self.cm.clone()), not the live buffer that "
        "update() adds into in place")
    _compare_result(m.compute(), _cm_loop(*imgs[0], C, 255) + _cm_loop(*imgs[1], C, 255),
                    "update() after compute() keeps accumulating")
    m.reset()
    m.update(*imgs[2])
    _compare_result(m.compute(), _cm_loop(*imgs[2], C, 255), "after reset()")
    # the constructor's ignore_index is used (here a valid class id, 0)
    m0 = ns.SegMetric(C, ignore_index=0)
    for p, t in imgs[:3]:
        m0.update(p, t)
    want = sum(_cm_loop(p, t, C, 0) for p, t in imgs[:3])
    _compare_result(m0.compute(), want, "SegMetric(num_classes, ignore_index=0)")


CHECKS = [check_confusion_matrix_matches_pixel_loop, check_confusion_matrix_rows_are_ground_truth,
          check_confusion_matrix_ignore_index_variants, check_confusion_matrix_out_of_range_labels,
          check_confusion_matrix_dtypes_and_empty_inputs, check_confusion_matrix_is_vectorized,
          check_iou_and_dice_per_class, check_mean_iou_excludes_absent_classes,
          check_pixel_and_mean_class_accuracy, check_segmetric_is_dataset_level,
          check_segmetric_state_handling]
