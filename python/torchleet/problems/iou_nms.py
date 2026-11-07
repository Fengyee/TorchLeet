"""iou-nms — pairwise box IoU, greedy non-maximum suppression, class-aware (batched) NMS.

Conventions graded here (they match torchvision.ops except where noted):
  * boxes are (x1, y1, x2, y2) with x1 <= x2 and y1 <= y2; zero width/height is
    allowed and coordinates MAY be negative (raw regressed boxes often are);
  * box_iou(a, b) is (N, M), differentiable, and 0 (not NaN) when the union is 0;
  * nms suppresses a box when IoU > iou_threshold (strictly greater), only kept
    boxes suppress, and kept indices come back sorted by decreasing score;
  * batched_nms never suppresses across labels. torchvision's coordinate trick
    offsets by `max + 1`, which is only safe for coordinates >= 0 (its documented
    precondition) — here coordinates can be negative, so that offset fails.

Oracles are brute-force Python loops written below. Random boxes live on a
quarter-pixel grid in float64, so IoU is computed exactly the same way by any
correct implementation and threshold comparisons cannot flip on rounding.
"""
import torch

from torchleet.runner import Skip

ENTRIES = ["box_iou", "nms", "batched_nms"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "box_iou: give boxes1 a new axis so it is (N, 1, 4) and boxes2 (1, M, 4); elementwise "
    "max/min of the corners then broadcast to (N, M) with no loop. NMS itself is inherently "
    "sequential: sort by score and walk the list once.",
    "Intersection width = (min of the right edges - max of the left edges), clamped at 0; same "
    "for height. Union = area1 + area2 - intersection, and it can be 0 for two zero-area boxes. "
    "In NMS only a box you KEEP may suppress others, and it suppresses when IoU > threshold "
    "(strictly).",
    "nms: compute the full IoU matrix once, visit boxes in descending-score order, skip any already "
    "suppressed, otherwise keep it and mark everything with IoU above the threshold as suppressed. "
    "batched_nms: either run nms per label, or shift every box by label * (max - min + 1) so boxes "
    "of different labels can never overlap and call nms once.",
]


# ---------------------------------------------------------------- oracles

def _iou(a, b):
    """IoU of two boxes given as Python lists; 0 when the union is 0."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _iou_matrix(b1, b2):
    b1, b2 = b1.tolist(), b2.tolist()
    return torch.tensor([[_iou(a, b) for b in b2] for a in b1],
                        dtype=torch.float64).reshape(len(b1), len(b2))


def _greedy_nms(boxes, scores, thr):
    """Textbook greedy NMS: a box is kept iff its IoU with every already-kept,
    higher-scoring box is <= thr."""
    boxes, scores = boxes.tolist(), scores.tolist()
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    keep = []
    for i in order:
        if all(_iou(boxes[i], boxes[k]) <= thr for k in keep):
            keep.append(i)
    return keep


def _per_class_nms(boxes, scores, labels, thr):
    keep = []
    for c in sorted(set(labels.tolist())):
        idx = [i for i, l in enumerate(labels.tolist()) if l == c]
        sub = _greedy_nms(boxes[idx], scores[idx], thr)
        keep += [idx[j] for j in sub]
    return sorted(keep, key=lambda i: -float(scores[i]))


def _grid_boxes(g, n, lo=0, hi=100, max_wh=40, min_wh=0):
    """Random boxes on a quarter-pixel grid (exact in float64), zero-size allowed."""
    x1 = torch.randint(lo * 4, hi * 4, (n,), generator=g)
    y1 = torch.randint(lo * 4, hi * 4, (n,), generator=g)
    w = torch.randint(min_wh * 4, max_wh * 4 + 1, (n,), generator=g)
    h = torch.randint(min_wh * 4, max_wh * 4 + 1, (n,), generator=g)
    return torch.stack([x1, y1, x1 + w, y1 + h], 1).double() / 4


def _clustered_boxes(g, n, lo=0, hi=100):
    """Boxes jittered around a few centres, so NMS has real decisions to make."""
    k = int(torch.randint(1, 5, (1,), generator=g))
    centres = _grid_boxes(g, k, lo, hi, max_wh=40, min_wh=8)
    pick = torch.randint(0, k, (n,), generator=g)
    jitter = torch.randint(-24, 25, (n, 4), generator=g).double() / 4
    b = centres[pick] + jitter
    x1 = torch.minimum(b[:, 0], b[:, 2])
    x2 = torch.maximum(b[:, 0], b[:, 2])
    y1 = torch.minimum(b[:, 1], b[:, 3])
    y2 = torch.maximum(b[:, 1], b[:, 3])
    return torch.stack([x1, y1, x2, y2], 1)


def _as_index_list(keep, what):
    assert torch.is_tensor(keep), f"{what} must return a tensor, got {type(keep).__name__}"
    assert keep.dtype == torch.long, f"{what} must return a LongTensor (int64), got {keep.dtype}"
    assert keep.dim() == 1, f"{what} must return a 1-D tensor of indices, got shape {tuple(keep.shape)}"
    return keep.tolist()


def _count_torch_calls(fn, *args):
    try:
        from torch.overrides import TorchFunctionMode
    except ImportError:  # very old torch
        raise Skip("needs torch.overrides.TorchFunctionMode (torch >= 2.0)")

    class _Counter(TorchFunctionMode):
        def __init__(self):
            super().__init__()
            self.n = 0

        def __torch_function__(self, func, types, args=(), kwargs=None):
            self.n += 1
            return func(*args, **(kwargs or {}))

    with _Counter() as c:
        fn(*args)
    return c.n


# ---------------------------------------------------------------- box_iou

def check_box_iou_matches_brute_force(ns):
    g = torch.Generator().manual_seed(0)
    a = _grid_boxes(g, 13)
    b = _grid_boxes(g, 9)
    b[0] = a[0]                                   # identical pair -> 1
    b[1] = torch.tensor([500.0, 500, 510, 510])   # disjoint from everything -> 0
    a[1] = torch.tensor([10.0, 10, 10, 30])       # zero width
    b[2] = torch.tensor([5.0, 20, 25, 20])        # zero height
    a[2] = torch.tensor([7.0, 7, 7, 7])           # a single point
    want = _iou_matrix(a, b)
    for dtype, atol in ((torch.float64, 1e-7), (torch.float32, 1e-5)):
        got = ns.box_iou(a.to(dtype), b.to(dtype))
        assert torch.is_tensor(got) and tuple(got.shape) == (13, 9), \
            f"box_iou((13,4), (9,4)) must return a (13, 9) tensor, got " \
            f"{tuple(got.shape) if torch.is_tensor(got) else type(got).__name__}"
        err = (got.double() - want).abs()
        # Relative slack as well: inter / (union + eps) with a tiny eps (common in YOLO-style
        # code) is fine. Every wrong convention is off by far more than 1e-5 relative.
        wrong = torch.isfinite(err) & (err > atol + 1e-5 * want)
        if wrong.any():
            i, j = divmod(int(torch.where(wrong, err, torch.zeros_like(err)).argmax()), 9)
            hint = ("Clamp intersection width/height at 0 so disjoint boxes give 0."
                    if float(want[i, j]) == 0.0 else
                    "IoU = inter / (area1 + area2 - inter), with areas (x2 - x1) * (y2 - y1) "
                    "(no '+1 pixel').")
            raise AssertionError(
                f"box_iou disagrees with a brute-force double loop ({dtype}): "
                f"boxes {a[i].tolist()} vs {b[j].tolist()} -> expected {float(want[i, j]):.8f}, "
                f"got {float(got[i, j]):.8f}. {hint}")
        if torch.isnan(got).any():
            i, j = divmod(int(torch.isnan(got).flatten().nonzero()[0]), 9)
            raise AssertionError(
                f"box_iou returned NaN for boxes {a[i].tolist()} vs {b[j].tolist()} — both have "
                f"zero area, so the union is 0; define IoU = 0 there")


def check_box_iou_degenerate_boxes_give_zero_not_nan(ns):
    pt = torch.tensor([[3.0, 3.0, 3.0, 3.0]])
    line = torch.tensor([[0.0, 5.0, 10.0, 5.0]])
    box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    for x, y, what in ((pt, pt, "identical point boxes"), (line, line, "identical zero-height boxes"),
                       (pt, box, "a point inside a box"), (line, box, "a zero-height box inside a box")):
        got = ns.box_iou(x, y)
        assert not torch.isnan(got).any(), \
            f"IoU of {what} is NaN — the union is 0 (or the intersection is 0); define IoU = 0 there"
        assert float(got.reshape(-1)[0]) == 0.0, f"IoU of {what} should be 0, got {float(got.reshape(-1)[0])}"
    same = ns.box_iou(box, box)
    assert abs(float(same.reshape(-1)[0]) - 1.0) < 1e-6, f"IoU of a box with itself must be 1, got {float(same)}"


def check_box_iou_handles_empty_inputs(ns):
    a = torch.zeros(0, 4)
    b = torch.tensor([[0.0, 0, 1, 1], [2, 2, 3, 3], [0, 0, 5, 5]])
    for x, y, want in ((a, b, (0, 3)), (b, a, (3, 0)), (a, a, (0, 0))):
        got = ns.box_iou(x, y)
        assert tuple(got.shape) == want, \
            f"box_iou with shapes {tuple(x.shape)} and {tuple(y.shape)} must return {want}, got {tuple(got.shape)}"


def check_box_iou_is_vectorized_and_differentiable(ns):
    g = torch.Generator().manual_seed(1)
    small_a, small_b = _grid_boxes(g, 4), _grid_boxes(g, 3)
    big_a, big_b = _grid_boxes(g, 60), _grid_boxes(g, 50)
    n_small = _count_torch_calls(ns.box_iou, small_a, small_b)
    n_big = _count_torch_calls(ns.box_iou, big_a, big_b)
    assert n_big <= n_small + 5, (
        f"box_iou made {n_small} torch calls for 4x3 boxes but {n_big} for 60x50 — the work grows "
        f"with a Python loop over boxes. Broadcast (N, 1, 4) against (1, M, 4) instead.")
    # Continuous (off-grid) overlapping boxes: no max/min ties, so finite differences are valid.
    gen = torch.Generator().manual_seed(2)
    xy = torch.rand(5, 2, generator=gen, dtype=torch.float64) * 4
    wh = 4 + torch.rand(5, 2, generator=gen, dtype=torch.float64) * 4
    a = torch.cat([xy, xy + wh], 1).requires_grad_()
    xy2 = torch.rand(4, 2, generator=gen, dtype=torch.float64) * 4
    wh2 = 4 + torch.rand(4, 2, generator=gen, dtype=torch.float64) * 4
    b = torch.cat([xy2, xy2 + wh2], 1).requires_grad_()
    out = ns.box_iou(a, b)
    assert torch.is_tensor(out) and out.requires_grad, \
        "box_iou output is detached from its inputs — avoid .item()/.tolist()/torch.tensor(...) " \
        "so it can serve as an IoU loss"
    ok = torch.autograd.gradcheck(lambda p, q: ns.box_iou(p, q), (a, b), eps=1e-6, atol=1e-5,
                                  raise_exception=False)
    assert ok, "autograd gradient of box_iou does not match finite differences"


# ---------------------------------------------------------------- nms

def check_nms_matches_greedy_reference(ns):
    g = torch.Generator().manual_seed(3)
    thresholds = [0.0, 0.3, 0.5, 0.7, 1.0]
    for trial in range(40):
        n = int(torch.randint(1, 60, (1,), generator=g))
        boxes = _clustered_boxes(g, n)
        scores = torch.rand(n, generator=g, dtype=torch.float64)
        thr = thresholds[trial % len(thresholds)]
        want = _greedy_nms(boxes, scores, thr)
        got = _as_index_list(ns.nms(boxes.clone(), scores.clone(), thr), "nms")
        assert got == want, (
            f"trial {trial} ({n} boxes, iou_threshold={thr}): expected keep={want[:12]}"
            f"{'...' if len(want) > 12 else ''}, got {got[:12]}{'...' if len(got) > 12 else ''}. "
            f"Indices refer to the ORIGINAL boxes, ordered by decreasing score; only kept boxes "
            f"suppress; suppress when IoU > threshold.")


def check_nms_output_format(ns):
    g = torch.Generator().manual_seed(4)
    boxes = _clustered_boxes(g, 30).float()
    scores = torch.rand(30, generator=g)
    b0, s0 = boxes.clone(), scores.clone()
    keep = ns.nms(boxes, scores, 0.5)
    idx = _as_index_list(keep, "nms")
    assert torch.equal(boxes, b0) and torch.equal(scores, s0), "nms must not modify its inputs in place"
    assert len(set(idx)) == len(idx), "nms returned duplicate indices"
    assert all(0 <= i < 30 for i in idx), "nms returned an index outside [0, N)"
    s = scores[keep]
    assert bool((s[:-1] >= s[1:]).all()), "kept indices must be sorted by decreasing score"


def check_nms_threshold_is_strictly_greater(ns):
    # IoU([0,0,2,1], [0,0,1,1]) = 1 / 2 exactly.
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 2.0, 1.0]])
    scores = torch.tensor([0.4, 0.9])
    got = _as_index_list(ns.nms(boxes, scores, 0.5), "nms")
    assert got == [1, 0], (
        f"two boxes with IoU exactly 0.5 and iou_threshold=0.5 must BOTH survive (suppress only "
        f"when IoU > threshold, as torchvision does); got {got}")
    got = _as_index_list(ns.nms(boxes, scores, 0.49), "nms")
    assert got == [1], f"with iou_threshold=0.49 the lower-scoring box must be suppressed; got {got}"


def check_nms_suppressed_boxes_do_not_suppress(ns):
    # A-B IoU 0.538, B-C IoU 0.538, A-C IoU 0.25; scores A > B > C.
    # B is suppressed by A, so B must NOT suppress C.
    C, A, B = [6.0, 0, 16, 10], [0.0, 0, 10, 10], [3.0, 0, 13, 10]
    boxes = torch.tensor([C, A, B])
    scores = torch.tensor([0.7, 0.9, 0.8])
    got = _as_index_list(ns.nms(boxes, scores, 0.5), "nms")
    assert got == [1, 0], (
        f"chain A-B-C (A suppresses B; B would suppress C, but B is already gone): expected "
        f"[1, 0] (A then C), got {got}. Only boxes you keep may suppress others.")


def check_nms_edge_cases(ns):
    empty = ns.nms(torch.zeros(0, 4), torch.zeros(0), 0.5)
    assert torch.is_tensor(empty) and empty.dtype == torch.long and tuple(empty.shape) == (0,), \
        f"nms on no boxes must return an empty LongTensor of shape (0,), got {empty!r}"
    one = _as_index_list(ns.nms(torch.tensor([[1.0, 1, 4, 4]]), torch.tensor([0.3]), 0.5), "nms")
    assert one == [0], f"a single box must be kept, got {one}"
    same = torch.tensor([[2.0, 2, 8, 9]]).repeat(5, 1)
    sc = torch.tensor([0.2, 0.7, 0.9, 0.1, 0.5])
    got = _as_index_list(ns.nms(same, sc, 0.5), "nms")
    assert got == [2], f"5 identical boxes must collapse to the highest-scoring one (index 2), got {got}"
    pts = torch.tensor([[3.0, 3, 3, 3]]).repeat(3, 1)
    got = _as_index_list(ns.nms(pts, torch.tensor([0.1, 0.3, 0.2]), 0.0), "nms")
    assert got == [1, 2, 0], (
        f"zero-area boxes have IoU 0 with everything, so even iou_threshold=0 keeps all of them; "
        f"got {got}")


# ---------------------------------------------------------------- batched_nms

def check_batched_nms_equals_per_label_nms(ns):
    g = torch.Generator().manual_seed(5)
    label_set = torch.tensor([0, 3, 7])
    for trial in range(25):
        n = int(torch.randint(1, 50, (1,), generator=g))
        boxes = _clustered_boxes(g, n, lo=-60, hi=60)  # negative coordinates on purpose
        scores = torch.rand(n, generator=g, dtype=torch.float64)
        labels = label_set[torch.randint(0, 3, (n,), generator=g)]
        thr = [0.3, 0.5, 0.7][trial % 3]
        want = _per_class_nms(boxes, scores, labels, thr)
        got = _as_index_list(ns.batched_nms(boxes.clone(), scores.clone(), labels.clone(), thr),
                             "batched_nms")
        assert got == want, (
            f"trial {trial} ({n} boxes, labels in {{0,3,7}}, some coordinates negative, "
            f"iou_threshold={thr}): "
            f"expected {want[:12]}{'...' if len(want) > 12 else ''}, got {got[:12]}"
            f"{'...' if len(got) > 12 else ''}. batched_nms must equal running nms separately "
            f"per label, merged and sorted by decreasing score.")


def check_batched_nms_never_suppresses_across_labels(ns):
    box = torch.tensor([[0.0, 0, 10, 10]]).repeat(3, 1)
    got = _as_index_list(ns.batched_nms(box, torch.tensor([0.9, 0.8, 0.7]), torch.tensor([0, 1, 2]), 0.5),
                         "batched_nms")
    assert got == [0, 1, 2], f"identical boxes with different labels must all survive, got {got}"
    # torchvision's offset is label * (max + 1). With negative coordinates that is
    # not enough separation: shifted copies of these two boxes still overlap.
    neg = torch.tensor([[-50.0, -50, 50, 50], [-50.0, -50, 50, 50]])
    got = _as_index_list(ns.batched_nms(neg, torch.tensor([0.9, 0.8]), torch.tensor([0, 1]), 0.1),
                         "batched_nms")
    assert got == [0, 1], (
        f"boxes with coordinates in [-50, 50] and different labels were suppressed across labels "
        f"(got {got}). An offset of label * (max + 1) only separates classes when every "
        f"coordinate is >= 0; use label * (max - min + 1), or loop over labels.")
    empty = ns.batched_nms(torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.long), 0.5)
    assert torch.is_tensor(empty) and empty.dtype == torch.long and tuple(empty.shape) == (0,), \
        f"batched_nms on no boxes must return an empty LongTensor, got {empty!r}"


CHECKS = [check_box_iou_matches_brute_force, check_box_iou_degenerate_boxes_give_zero_not_nan,
          check_box_iou_handles_empty_inputs, check_box_iou_is_vectorized_and_differentiable,
          check_nms_matches_greedy_reference, check_nms_output_format,
          check_nms_threshold_is_strictly_greater, check_nms_suppressed_boxes_do_not_suppress,
          check_nms_edge_cases, check_batched_nms_equals_per_label_nms,
          check_batched_nms_never_suppresses_across_labels]
