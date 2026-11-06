"""detection-map — all-point interpolated Average Precision and detection mAP@IoU.

Conventions graded here:
  * average_precision(recall, precision): the PASCAL VOC >= 2010 all-point
    definition (devkit `voc_ap(..., use_07_metric=False)`):
        add sentinels  r = [0, r_1..r_K, 1],  p = [0, p_1..p_K, 0]
        make p monotonically non-increasing from the right (p_i = max_{j>=i} p_j)
        AP = sum_i (r_{i+1} - r_i) * p_{i+1}   over points where recall changes
    i.e. AP = sum_k (r_k - r_{k-1}) * max_{j>=k} p_j with r_0 = 0. Recall that is
    never reached contributes nothing (the sentinel precision at recall 1 is 0).
    Empty input -> 0.0.
  * mean_average_precision(...): per class, all predictions of that class are
    sorted by score across ALL images; each is matched greedily to the
    highest-IoU *still-unmatched* GT of the same class in the same image with
    IoU >= iou_threshold (COCO-style matching and inclusive threshold; the VOC
    devkit instead takes the max-IoU GT even if matched, and uses IoU > t).
    Unmatched predictions (incl. duplicates of a matched GT) are false positives.
    Recall denominator = #GT of the class. mAP = mean AP over classes with >= 1 GT;
    classes that only appear in predictions are ignored (absent from the dict).
    No GT at all -> (0.0, {}).

Oracles are written out below with plain Python loops. Random boxes are on a
quarter-pixel grid in float64 so IoU threshold comparisons are exact.
"""
import torch

ENTRIES = ["average_precision", "mean_average_precision"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Split it in two: (1) turn a ranked list of TP/FP flags into precision and recall arrays, "
    "(2) turn those arrays into one number. AP is the area under the precision-recall curve "
    "after replacing each precision by the best precision achievable at that recall or higher.",
    "For AP: prepend recall 0 and append recall 1 with precision 0, take a reverse running max of "
    "precision (flip, cummax, flip), and sum (recall step) x (interpolated precision). For mAP: "
    "per class, sort ALL its predictions across images by score, keep a 'used' flag per GT, and "
    "only compare a prediction with GTs of the same class in the same image.",
    "A prediction is a TP if, among the unmatched same-class GTs in its image, the one with the "
    "highest IoU has IoU >= threshold (mark it used); otherwise FP. Cumulative sums of the TP and FP "
    "flags give precision = TP / (TP + FP) and recall = TP / num_gt_of_class. Average AP only over "
    "classes that have at least one GT.",
]


# ---------------------------------------------------------------- oracles

def _iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _voc_ap(rec, prec):
    """PASCAL VOC devkit voc_ap(use_07_metric=False), with explicit loops."""
    mrec = [0.0] + [float(r) for r in rec] + [1.0]
    mpre = [0.0] + [float(p) for p in prec] + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    return sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]
               for i in range(len(mrec) - 1) if mrec[i + 1] != mrec[i])


def _map_oracle(pb, ps, pl, gb, gl, thr=0.5):
    n_img = len(gb)
    classes = sorted({c for img in range(n_img) for c in gl[img].tolist()})
    aps = {}
    for c in classes:
        dets = [(s, img, j) for img in range(n_img)
                for j, (s, l) in enumerate(zip(ps[img].tolist(), pl[img].tolist())) if l == c]
        dets.sort(key=lambda t: -t[0])
        gts = {img: [j for j, l in enumerate(gl[img].tolist()) if l == c] for img in range(n_img)}
        npos = sum(len(v) for v in gts.values())
        used, flags = set(), []
        for _, img, j in dets:
            box = pb[img][j].tolist()
            best, best_iou = None, -1.0
            for gj in gts[img]:
                if (img, gj) in used:
                    continue
                iou = _iou(box, gb[img][gj].tolist())
                if iou >= thr and iou > best_iou:
                    best, best_iou = gj, iou
            if best is None:
                flags.append(0)
            else:
                used.add((img, best))
                flags.append(1)
        tp = fp = 0
        rec, prec = [], []
        for f in flags:
            tp += f
            fp += 1 - f
            rec.append(tp / npos)
            prec.append(tp / (tp + fp))
        aps[c] = _voc_ap(rec, prec)
    return (sum(aps.values()) / len(aps) if aps else 0.0), aps


# ---------------------------------------------------------------- data

def _t(rows):
    return torch.tensor(rows, dtype=torch.float64).reshape(-1, 4)


def _l(vals):
    return torch.tensor(vals, dtype=torch.long)


def _s(vals):
    return torch.tensor(vals, dtype=torch.float64)


def _grid_boxes(g, n, lo=0, hi=200, min_wh=8, max_wh=40):
    x1 = torch.randint(lo * 4, hi * 4, (n,), generator=g)
    y1 = torch.randint(lo * 4, hi * 4, (n,), generator=g)
    w = torch.randint(min_wh * 4, max_wh * 4 + 1, (n,), generator=g)
    h = torch.randint(min_wh * 4, max_wh * 4 + 1, (n,), generator=g)
    return torch.stack([x1, y1, x1 + w, y1 + h], 1).double() / 4


def _jitter(g, box, px=6):
    b = box + torch.randint(-4 * px, 4 * px + 1, (4,), generator=g).double() / 4
    return torch.stack([torch.minimum(b[0], b[2]), torch.minimum(b[1], b[3]),
                        torch.maximum(b[0], b[2]), torch.maximum(b[1], b[3])])


def _random_dataset(g, n_img=6, n_cls=3, extra_class=True):
    """Noisy detector output: jittered hits, duplicates, wrong-label hits, background
    boxes, and (optionally) a class n_cls that has predictions but no GT."""
    pb, ps, pl, gb, gl = [], [], [], [], []
    for _ in range(n_img):
        G = int(torch.randint(0, 6, (1,), generator=g))
        gt = _grid_boxes(g, G)
        glab = torch.randint(0, n_cls, (G,), generator=g)
        boxes, labels = [], []
        for j in range(G):
            if float(torch.rand(1, generator=g)) < 0.8:
                boxes.append(_jitter(g, gt[j])); labels.append(int(glab[j]))
            if float(torch.rand(1, generator=g)) < 0.3:
                boxes.append(_jitter(g, gt[j])); labels.append(int(glab[j]))
            if float(torch.rand(1, generator=g)) < 0.15:
                boxes.append(_jitter(g, gt[j])); labels.append((int(glab[j]) + 1) % n_cls)
        n_bg = int(torch.randint(0, 4, (1,), generator=g))
        for b in _grid_boxes(g, n_bg):
            boxes.append(b)
            labels.append(int(torch.randint(0, n_cls + (1 if extra_class else 0), (1,), generator=g)))
        P = len(boxes)
        pb.append(torch.stack(boxes) if P else torch.zeros(0, 4, dtype=torch.float64))
        pl.append(_l(labels))
        ps.append(torch.rand(P, generator=g, dtype=torch.float64))
        gb.append(gt)
        gl.append(glab)
    return pb, ps, pl, gb, gl


def _call_map(ns, pb, ps, pl, gb, gl, **kw):
    out = ns.mean_average_precision([x.clone() for x in pb], [x.clone() for x in ps],
                                    [x.clone() for x in pl], [x.clone() for x in gb],
                                    [x.clone() for x in gl], **kw)
    assert isinstance(out, tuple) and len(out) == 2, \
        "mean_average_precision must return a tuple (mAP, per_class_ap_dict)"
    m, per = out
    assert isinstance(per, dict), f"the second return value must be a dict, got {type(per).__name__}"
    return float(m), {int(k): float(v) for k, v in per.items()}


def _why(got, known):
    """Append the diagnosis for a known wrong value, if `got` is one of them."""
    for value, reason in known:
        if abs(got - value) < 1e-4:
            return f" — {reason}"
    return ""


def _fmt(d):
    return "{" + ", ".join(f"{k}: {v:.4f}" for k, v in sorted(d.items())) + "}"


# ---------------------------------------------------------------- average_precision

def _ap(ns, rec, prec):
    return float(ns.average_precision(torch.tensor(rec, dtype=torch.float32),
                                      torch.tensor(prec, dtype=torch.float32)))


def check_average_precision_hand_examples(ns):
    # 4 GTs, ranked flags TP FP TP TP -> recall .25 .25 .5 .75, precision 1 .5 .667 .75
    rec, prec = [0.25, 0.25, 0.5, 0.75], [1.0, 0.5, 2 / 3, 0.75]
    got = _ap(ns, rec, prec)
    assert abs(got - 0.625) < 1e-5, (
        f"recall={rec}, precision=[1, .5, .667, .75]: expected AP 0.625 = .25*1 + .25*.75 + .25*.75, "
        f"got {got:.4f}" + _why(got, [
            (0.625 - 0.25 * 0.75 + 0.25 * 2 / 3, "no precision envelope (p_hat_i = max_{j>=i} p_j)"),
            (0.75, "the envelope runs left to right; it must be a running max from the RIGHT"),
            (0.8125, "the last precision was extrapolated to recall 1; the sentinel there is 0"),
            (0.375, "the first segment, from recall 0 to the first recall, is missing")]))
    got = _ap(ns, [0.5], [1.0])
    assert abs(got - 0.5) < 1e-6, \
        f"a single point (recall .5, precision 1) gives AP 0.5 — unreached recall adds nothing; got {got:.4f}"
    got = _ap(ns, [0.5, 1.0], [1.0, 1.0])
    assert abs(got - 1.0) < 1e-6, f"precision 1 up to recall 1 must give AP 1.0, got {got:.4f}"
    got = _ap(ns, [0.0, 0.0, 0.5], [0.0, 0.0, 1 / 3])
    assert abs(got - 1 / 6) < 1e-6, \
        f"flags FP FP TP with 2 GTs: expected AP 1/6 (recall .5 at precision 1/3), got {got:.4f}"
    empty = float(ns.average_precision(torch.zeros(0), torch.zeros(0)))
    assert empty == 0.0, f"no predictions -> AP 0.0, got {empty}"


def check_average_precision_matches_voc_reference(ns):
    g = torch.Generator().manual_seed(0)
    for trial in range(60):
        k = int(torch.randint(1, 40, (1,), generator=g))
        if trial % 2 == 0:  # realistic: from a ranked TP/FP sequence
            flags = (torch.rand(k, generator=g) < 0.6).double()
            npos = int(flags.sum()) + int(torch.randint(0, 4, (1,), generator=g))
            if npos == 0:
                continue
            tp = flags.cumsum(0)
            rec = (tp / npos).tolist()
            prec = (tp / torch.arange(1, k + 1)).tolist()
        else:               # arbitrary: non-decreasing recall with repeats, any precision
            steps = torch.rand(k, generator=g) * (torch.rand(k, generator=g) < 0.6)
            rec = (steps.cumsum(0) / (steps.sum() + 0.5)).tolist()
            prec = torch.rand(k, generator=g).tolist()
        want = _voc_ap(rec, prec)
        got = _ap(ns, rec, prec)
        assert abs(got - want) < 1e-5, (
            f"trial {trial}: AP {got:.6f} != VOC all-point reference {want:.6f} for recall="
            f"{[round(r, 3) for r in rec[:8]]}..., precision={[round(p, 3) for p in prec[:8]]}...")


# ---------------------------------------------------------------- mean_average_precision

def check_map_perfect_predictions(ns):
    g = torch.Generator().manual_seed(1)
    gb = [_grid_boxes(g, n) for n in (3, 4, 2)]
    gl = [torch.randint(0, 3, (len(b),), generator=g) for b in gb]
    ps = [torch.rand(len(b), generator=g, dtype=torch.float64) for b in gb]
    m, per = _call_map(ns, gb, ps, gl, gb, gl)
    want = sorted({c for x in gl for c in x.tolist()})
    assert sorted(per) == want, f"per-class dict keys should be the GT classes {want}, got {sorted(per)}"
    assert abs(m - 1.0) < 1e-6 and all(abs(v - 1.0) < 1e-6 for v in per.values()), \
        f"predictions identical to the GT must give mAP 1.0; got {m:.4f}, per-class {_fmt(per)}"


def _hand_example():
    # cat = 0, dog = 1, bird = 2 (bird has no GT -> ignored)
    gb = [_t([[0, 0, 10, 10], [20, 20, 30, 30]]),
          _t([[0, 0, 10, 10], [50, 50, 60, 60], [70, 70, 80, 80]])]
    gl = [_l([0, 1]), _l([0, 0, 1])]
    pb = [_t([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30], [60, 60, 70, 70], [0, 0, 10, 10]]),
          _t([[30, 30, 40, 40], [50, 50, 60, 60], [70, 70, 80, 80]])]
    ps = [_s([0.9, 0.8, 0.7, 0.95, 0.99]), _s([0.85, 0.6, 0.5])]
    pl = [_l([0, 0, 1, 1, 2]), _l([0, 0, 1])]
    return pb, ps, pl, gb, gl


def check_map_hand_computed_example(ns):
    """
    Image 0 GT: cat [0,0,10,10], dog [20,20,30,30]
    Image 1 GT: cat [0,0,10,10], cat [50,50,60,60], dog [70,70,80,80]
    cat (3 GT), predictions by score:
        .90 img0 [0,0,10,10]   TP
        .85 img1 [30,30,40,40] FP (no overlap)
        .80 img0 [1,1,10,10]   FP (IoU .81 with img0's cat, already matched -> duplicate)
        .60 img1 [50,50,60,60] TP
      recall 1/3 1/3 1/3 2/3, precision 1 1/2 1/3 1/2 -> envelope 1 1/2 1/2 1/2
      AP = 1/3*1 + 1/3*1/2 = 0.5   (img1's cat [0,0,10,10] is never found)
    dog (2 GT): .95 img0 [60,60,70,70] FP, .70 img0 TP, .50 img1 TP
      recall 0 1/2 1, precision 0 1/2 2/3 -> envelope 2/3 2/3 2/3 -> AP = 2/3
    bird: predicted at .99 but has no GT -> ignored.
    mAP = (1/2 + 2/3) / 2 = 7/12 = 0.5833
    """
    m, per = _call_map(ns, *_hand_example())   # default iou_threshold=0.5
    assert sorted(per) == [0, 1], \
        f"per-class dict must contain exactly the classes with GT ({{0, 1}}); bird (2) has none — got keys {sorted(per)}"
    assert abs(per[0] - 0.5) < 1e-5, (
        f"cat AP should be 0.5 (TP, FP, duplicate FP, TP over 3 GTs), got {per[0]:.4f}" + _why(per[0], [
            (5 / 6, "the duplicate [1,1,10,10] in image 0 was counted as a TP: either GTs are not marked "
                    "as matched, or it was matched to the cat in image 1 (match only within the same image)"),
            (0.75, "recall must be divided by the number of GTs of the class (3), not by the number found"),
            (2 / 3, "check the precision envelope direction and the recall-1 sentinel")]))
    assert abs(per[1] - 2 / 3) < 1e-5, (
        f"dog AP should be 0.6667 (FP, TP, TP over 2 GTs with the precision envelope), got {per[1]:.4f}"
        + _why(per[1], [(7 / 12, "the precision envelope is missing")]))
    assert abs(m - 7 / 12) < 1e-5, f"mAP should be (0.5 + 0.6667) / 2 = 0.5833, got {m:.4f}"


def check_map_duplicate_detection_is_false_positive(ns):
    gb = [_t([[0, 0, 10, 10], [40, 40, 50, 50]])]
    gl = [_l([0, 0])]
    pb = [_t([[0, 0, 10, 10], [0, 0, 10, 10], [40, 40, 50, 50]])]
    m, per = _call_map(ns, pb, [_s([0.9, 0.8, 0.7])], [_l([0, 0, 0])], gb, gl)
    assert abs(m - 5 / 6) < 1e-5, (
        f"GT A, B; predictions A(.9), A again(.8), B(.7): the second A is a duplicate -> FP, so "
        f"recall .5 .5 1, precision 1 .5 .667 and AP = .5*1 + .5*.667 = 0.8333; got {m:.4f}. "
        f"Each GT may be matched at most once.")


def check_map_missed_gt_caps_recall(ns):
    gb = [_t([[0, 0, 10, 10], [40, 40, 50, 50]]), _t([[5, 5, 25, 25]])]
    gl = [_l([1, 1]), _l([1])]
    pb = [_t([[0, 0, 10, 10]]), _t([[5, 5, 25, 25]])]
    m, per = _call_map(ns, pb, [_s([0.9]), _s([0.4])], [_l([1]), _l([1])], gb, gl)
    assert abs(per.get(1, -1) - 2 / 3) < 1e-5, (
        f"3 GTs, 2 found with perfect precision -> recall tops out at 2/3 and AP = 0.6667; got "
        f"{_fmt(per)}. The recall denominator is the number of GTs of the class, not of predictions.")
    gb2 = gb + [_t([[0, 0, 4, 4]])]
    gl2 = gl + [_l([5])]
    m2, per2 = _call_map(ns, pb + [torch.zeros(0, 4, dtype=torch.float64)],
                         [_s([0.9]), _s([0.4]), _s([])], [_l([1]), _l([1]), _l([])], gb2, gl2)
    assert abs(per2.get(5, -1)) < 1e-9 and abs(m2 - (2 / 3) / 2) < 1e-5, (
        f"a class with GT but no predictions has AP 0 and still counts in the mean: expected "
        f"{{1: 0.6667, 5: 0.0}} and mAP 0.3333, got {_fmt(per2)}, mAP {m2:.4f}")


def check_map_iou_threshold_is_inclusive(ns):
    # IoU([0,0,1,1], [0,0,2,1]) = 1/2 exactly.
    gb, gl = [_t([[0, 0, 2, 1]])], [_l([0])]
    pb, ps, pl = [_t([[0, 0, 1, 1]])], [_s([0.9])], [_l([0])]
    m, _ = _call_map(ns, pb, ps, pl, gb, gl, iou_threshold=0.5)
    assert abs(m - 1.0) < 1e-6, (
        f"a prediction with IoU exactly 0.5 must count as a match at iou_threshold=0.5 "
        f"(IoU >= threshold, COCO convention); got mAP {m:.4f}")
    m, _ = _call_map(ns, pb, ps, pl, gb, gl, iou_threshold=0.75)
    assert abs(m) < 1e-9, f"the same prediction is a FP at iou_threshold=0.75; got mAP {m:.4f}"


def check_map_matches_highest_iou_unmatched_gt(ns):
    # (a) Take the HIGHEST-IoU GT, not the first one above the threshold.
    #     G1=[0,10], G2=[4,14] (x-extent); P1=[3,13] IoU .54 with G1, .82 with G2 -> G2.
    #     P2=[5,15]: G2 is used, IoU(P2, G1) = .33 < .5 -> FP.  AP = 0.5.
    gb = [_t([[0, 0, 10, 10], [4, 0, 14, 10]])]
    gl = [_l([0, 0])]
    pb = [_t([[3, 0, 13, 10], [5, 0, 15, 10]])]
    m, _ = _call_map(ns, pb, [_s([0.9, 0.8])], [_l([0, 0])], gb, gl)
    assert abs(m - 0.5) < 1e-5, (
        f"prediction 1 overlaps GT1 (IoU .54) and GT2 (IoU .82): it must take GT2, its highest-IoU "
        f"match, which leaves prediction 2 (IoU .82 with GT2, .33 with GT1) as a FP -> AP 0.5; "
        f"got {m:.4f}" + _why(m, [(1.0, "prediction 1 took the first GT above the threshold, or a GT "
                                        "was matched twice")]))
    # (b) Among UNMATCHED GTs: P2's best GT is already taken, but another GT still has IoU >= .5.
    gb = [_t([[0, 0, 10, 10], [2, 0, 12, 10]])]
    pb = [_t([[0, 0, 10, 10], [0.75, 0, 10.75, 10]])]
    m, _ = _call_map(ns, pb, [_s([0.9, 0.8])], [_l([0, 0])], gb, gl)
    assert abs(m - 1.0) < 1e-5, (
        f"prediction 2 has IoU .86 with an already-matched GT and .78 with an unmatched one: it "
        f"must match the unmatched GT (TP) -> AP 1.0; got {m:.4f}" + _why(m, [
            (0.5, "that is VOC-devkit matching ('best GT overall, FP if already taken'); this problem "
                  "matches the best *unmatched* GT")]))


def check_map_is_order_invariant(ns):
    g = torch.Generator().manual_seed(2)
    data = _random_dataset(g, n_img=6)
    base_m, base_per = _call_map(ns, *data)
    pb, ps, pl, gb, gl = data
    img_perm = torch.randperm(len(gb), generator=g).tolist()
    sh = [[], [], [], [], []]
    for i in img_perm:
        p = torch.randperm(len(ps[i]), generator=g)
        q = torch.randperm(len(gl[i]), generator=g)
        sh[0].append(pb[i][p]); sh[1].append(ps[i][p]); sh[2].append(pl[i][p])
        sh[3].append(gb[i][q]); sh[4].append(gl[i][q])
    m, per = _call_map(ns, *sh)
    assert abs(m - base_m) < 1e-6 and per.keys() == base_per.keys() and \
        all(abs(per[k] - base_per[k]) < 1e-6 for k in per), (
        f"shuffling predictions/GT within images and the image order changed the result: "
        f"{base_m:.4f} {_fmt(base_per)} -> {m:.4f} {_fmt(per)}. Sort predictions by score across "
        f"all images, and don't rely on GT order.")


def check_map_ignores_classes_without_gt(ns):
    g = torch.Generator().manual_seed(3)
    pb, ps, pl, gb, gl = _random_dataset(g, n_img=5, extra_class=False)
    m0, per0 = _call_map(ns, pb, ps, pl, gb, gl)
    # add confident predictions of class 9, which has no GT anywhere
    pb2 = [torch.cat([b, _t([[1, 1, 30, 30]])]) for b in pb]
    ps2 = [torch.cat([s, _s([0.999])]) for s in ps]
    pl2 = [torch.cat([l, _l([9])]) for l in pl]
    m1, per1 = _call_map(ns, pb2, ps2, pl2, gb, gl)
    assert 9 not in per1, \
        "class 9 has predictions but no GT: its AP is undefined, so leave it out of the per-class dict"
    assert abs(m1 - m0) < 1e-6, (
        f"adding predictions of a class with no GT changed mAP {m0:.4f} -> {m1:.4f}; average only "
        f"over classes that have at least one GT")
    none = ns.mean_average_precision([_t([[0, 0, 1, 1]])], [_s([0.5])], [_l([0])],
                                     [torch.zeros(0, 4, dtype=torch.float64)], [_l([])])
    assert isinstance(none, tuple) and float(none[0]) == 0.0 and dict(none[1]) == {}, \
        f"with no GT at all, return (0.0, {{}}); got {none!r}"


def check_map_matches_oracle_on_random_detections(ns):
    g = torch.Generator().manual_seed(4)
    for trial in range(16):
        thr = (0.5, 0.75)[trial % 2]
        data = _random_dataset(g, n_img=int(torch.randint(1, 8, (1,), generator=g)))
        want_m, want_per = _map_oracle(*data, thr=thr)
        m, per = _call_map(ns, *data, iou_threshold=thr)
        assert sorted(per) == sorted(want_per), \
            f"trial {trial}: per-class keys {sorted(per)} != GT classes {sorted(want_per)}"
        bad = [k for k in per if abs(per[k] - want_per[k]) > 1e-5]
        assert not bad and abs(m - want_m) < 1e-5, (
            f"trial {trial} (iou_threshold={thr}, {len(data[0])} images, jittered hits + "
            f"duplicates + wrong-label hits + background boxes): expected mAP {want_m:.5f} "
            f"{_fmt(want_per)}, got {m:.5f} {_fmt(per)}")


CHECKS = [check_average_precision_hand_examples, check_average_precision_matches_voc_reference,
          check_map_perfect_predictions, check_map_hand_computed_example,
          check_map_duplicate_detection_is_false_positive, check_map_missed_gt_caps_recall,
          check_map_iou_threshold_is_inclusive, check_map_matches_highest_iou_unmatched_gt,
          check_map_is_order_invariant, check_map_ignores_classes_without_gt,
          check_map_matches_oracle_on_random_detections]
