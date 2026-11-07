"""roi-align — RoIAlign (Mask R-CNN) and RoIPool (Fast R-CNN) with torchvision.ops semantics.

Both oracles below are line-by-line Python transcriptions of torchvision's CPU kernels
(github.com/pytorch/vision, torchvision/csrc/ops/cpu/roi_align_kernel.cpp + roi_align_common.h,
and roi_pool_kernel.cpp), evaluated in float64. They were cross-checked against
torchvision 0.29.2's compiled ops (torch 2.15, CPU) on the same random cases used here.

RoIAlign conventions (roi_align_forward_kernel_impl / pre_calc_for_bilinear_interpolate):
  * boxes are (K, 5) rows [batch_index, x1, y1, x2, y2] in input-image coordinates; x is the
    width (column) axis. start = coord * spatial_scale - offset, offset = 0.5 iff aligned
    ("Do not using rounding; this implementation detail is critical").
  * roi_w = x2' - x1', roi_h = y2' - y1'; ONLY when aligned=False both are clamped to >= 1
    ("Force malformed ROIs to be 1x1").
  * bin = roi / out; grid = sampling_ratio if > 0 else ceil(roi / out) (per axis, per box);
    sample y = y1' + ph * bin_h + (iy + .5) * bin_h / grid_h, same for x.
  * a sample with y < -1 or y > H or x < -1 or x > W contributes 0 (all four weights 0);
    otherwise y <= 0 -> 0, y_low = (int) y, and if y_low >= H - 1 then y_low = y_high = H - 1
    (and y = y_low), else y_high = y_low + 1; same for x; standard 4-tap bilinear weights.
  * output = sum of the samples / max(grid_h * grid_w, 1) — an average; an empty grid
    (aligned=True, zero-size box, adaptive) gives 0, not NaN.
  * backward (roi_align_backward_kernel_impl) scatters grad * w / count to the same four taps,
    i.e. it is exactly the transpose of the (linear) forward; the oracle gradient here is M^T g
    for the dense weight matrix M of the oracle.

RoIPool conventions (roi_pool_forward_kernel_impl):
  * start/end = C++ round(coord * spatial_scale) — round half AWAY from zero (torch.round
    rounds half to even); roi_w = max(x2r - x1r + 1, 1) (inclusive end, legacy +1).
  * bin = roi / out (floating); bin p covers [floor(p * bin), ceil((p + 1) * bin)) shifted by
    the start, each end clipped to [0, H] (resp. W); an empty bin outputs 0, otherwise the max.
  * floor/ceil of p * (roi / out) in floating point can differ from exact integer arithmetic
    for a few (roi, out) pairs (e.g. ceil(7 * (29 / 7)) = 30 in float64); the random cases
    only use boxes for which float32, float64 and exact rational arithmetic agree, so both
    styles pass.

Random RoIAlign cases are filtered so that no sample lies within 1e-4 of the -1 / H / W
cut-offs and no adaptive ratio roi/out lies within 1e-4 of an integer: those are the only
places where the output is discontinuous in the box, so any correct op order agrees there.
The exact cut-off rule is graded separately on dyadic (exactly representable) coordinates.
"""
import math

import torch

from torchleet.runner import Skip

ENTRIES = ["roi_align", "roi_pool"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Each output bin is the AVERAGE of a small regular grid of bilinear samples, and nothing is "
    "rounded: the box is mapped to continuous feature-map coordinates (coord * spatial_scale, "
    "minus 0.5 when aligned). The sample positions are a tensor product of a y-grid and an "
    "x-grid, so compute them per axis, once per box, and reuse the two-tap gather from your "
    "bilinear-interpolation solution.",
    "Per box and axis: roi = end - start (clamped to >= 1 only when aligned=False), "
    "bin = roi / out, grid = sampling_ratio if > 0 else ceil(roi / out), sample "
    "y = start + p * bin + (i + 0.5) * bin / grid. A sample with y < -1 or y > H contributes 0; "
    "otherwise clamp y to [0, H - 1] and use taps floor(y) and min(floor(y) + 1, H - 1). Divide "
    "the sum by max(grid_h * grid_w, 1).",
    "For one box: build y of shape (out_h * grid_h,) and x of shape (out_w * grid_w,), turn each "
    "into (low index, high index, low weight, high weight) with the weights multiplied by the "
    "in-range mask; gather rows img[:, y_low] and img[:, y_high] and blend -> (C, out_h*grid_h, W); "
    "gather columns the same way -> (C, out_h*grid_h, out_w*grid_w); view as "
    "(C, out_h, grid_h, out_w, grid_w), sum dims 2 and 4, divide. roi_pool: round each scaled "
    "coordinate half away from zero (sign(v) * floor(|v| + 0.5)), roi = end - start + 1 (>= 1), "
    "bin p spans [floor(p*bin), ceil((p+1)*bin)) + start clipped to [0, H], max over it, 0 if empty.",
]


# ------------------------------------------------------------------ RoIAlign oracle

def _taps(y, x, n, H, W, clamp_outside=False, zero_pad=False):
    """Bilinear taps of one sample, as [(flat index into (N, H, W), weight)].

    Mirrors pre_calc_for_bilinear_interpolate. The two flags are deliberately wrong variants
    used only to explain a failure: `clamp_outside` clamps samples beyond the -1 / H cut-off
    instead of dropping them, `zero_pad` mimics grid_sample(padding_mode="zeros").
    """
    base = n * H * W
    if zero_pad:  # treat everything outside [0, H-1] x [0, W-1] as zeros, no clamping
        y0, x0 = math.floor(y), math.floor(x)
        ly, lx = y - y0, x - x0
        out = []
        for yy, wy in ((y0, 1 - ly), (y0 + 1, ly)):
            for xx, wx in ((x0, 1 - lx), (x0 + 1, lx)):
                if 0 <= yy < H and 0 <= xx < W:
                    out.append((base + yy * W + xx, wy * wx))
        return out
    if not clamp_outside and (y < -1.0 or y > H or x < -1.0 or x > W):
        return []
    if y <= 0:
        y = 0.0
    if x <= 0:
        x = 0.0
    y_low, x_low = int(y), int(x)
    if y_low >= H - 1:
        y_high = y_low = H - 1
        y = float(y_low)
    else:
        y_high = y_low + 1
    if x_low >= W - 1:
        x_high = x_low = W - 1
        x = float(x_low)
    else:
        x_high = x_low + 1
    ly, lx = y - y_low, x - x_low
    hy, hx = 1.0 - ly, 1.0 - lx
    return [(base + y_low * W + x_low, hy * hx), (base + y_low * W + x_high, hy * lx),
            (base + y_high * W + x_low, ly * hx), (base + y_high * W + x_high, ly * lx)]


def _box_geometry(box, out_h, out_w, scale, sampling_ratio, aligned, min_size=True,
                  scale_after_offset=False, offset=None):
    """Per-box start, bin size and grid size, exactly as in roi_align_forward_kernel_impl."""
    _, x1, y1, x2, y2 = box
    off = (0.5 if aligned else 0.0) if offset is None else offset
    if scale_after_offset:  # wrong variant: (x - 0.5) * scale
        sw, sh, ew, eh = (x1 - off) * scale, (y1 - off) * scale, (x2 - off) * scale, (y2 - off) * scale
    else:
        sw, sh, ew, eh = x1 * scale - off, y1 * scale - off, x2 * scale - off, y2 * scale - off
    rw, rh = ew - sw, eh - sh
    if not aligned and min_size:
        rw, rh = max(rw, 1.0), max(rh, 1.0)
    bh, bw = rh / out_h, rw / out_w
    gh = sampling_ratio if sampling_ratio > 0 else math.ceil(rh / out_h)
    gw = sampling_ratio if sampling_ratio > 0 else math.ceil(rw / out_w)
    return sh, sw, bh, bw, gh, gw, rh, rw


def _align_samples(inp_shape, boxes, output_size, scale, sampling_ratio, aligned, **variant):
    """For every bin (k-major, then ph, pw): (list of samples, count); a sample is a tap list."""
    N, C, H, W = inp_shape
    out_h, out_w = output_size
    clamp_outside = variant.pop("clamp_outside", False)
    zero_pad = variant.pop("zero_pad", False)
    ignore_batch = variant.pop("ignore_batch", False)
    no_count_clamp = variant.pop("no_count_clamp", False)
    bins = []
    for box in boxes.tolist():
        n = 0 if ignore_batch else int(box[0])
        sh, sw, bh, bw, gh, gw, _, _ = _box_geometry(box, out_h, out_w, scale, sampling_ratio,
                                                      aligned, **variant)
        count = gh * gw if no_count_clamp else max(gh * gw, 1)
        for ph in range(out_h):
            for pw in range(out_w):
                samples = []
                for iy in range(gh):
                    yy = sh + ph * bh + (iy + 0.5) * bh / gh
                    for ix in range(gw):
                        xx = sw + pw * bw + (ix + 0.5) * bw / gw
                        samples.append(_taps(yy, xx, n, H, W, clamp_outside, zero_pad))
                bins.append((samples, count))
    return bins


def _align_matrix(inp_shape, boxes, output_size, scale, sampling_ratio, aligned, **variant):
    """Dense (K * out_h * out_w, N * H * W) float64 matrix M with out = M @ input."""
    N, C, H, W = inp_shape
    bins = _align_samples(inp_shape, boxes, output_size, scale, sampling_ratio, aligned, **variant)
    M = torch.zeros(len(bins), N * H * W, dtype=torch.float64)
    for r, (samples, count) in enumerate(bins):
        for taps in samples:
            for idx, w in taps:
                M[r, idx] += w / count if count else float("nan")
        if count == 0:
            M[r] = float("nan")
    return M


def _flat(inp):
    N, C, H, W = inp.shape
    return inp.detach().double().permute(0, 2, 3, 1).reshape(N * H * W, C)


def _apply(M, inp, K, output_size):
    out_h, out_w = output_size
    C = inp.shape[1]
    return (M @ _flat(inp)).reshape(K, out_h, out_w, C).permute(0, 3, 1, 2)


def _ref_align(inp, boxes, output_size, scale=1.0, sampling_ratio=-1, aligned=False, **variant):
    M = _align_matrix(tuple(inp.shape), boxes, output_size, scale, sampling_ratio, aligned, **variant)
    return _apply(M, inp, boxes.shape[0], output_size)


def _ref_align_max(inp, boxes, output_size, scale=1.0, sampling_ratio=-1, aligned=False):
    """Wrong variant: max over the samples of a bin instead of the average."""
    N, C, H, W = inp.shape
    X = _flat(inp)
    bins = _align_samples(tuple(inp.shape), boxes, output_size, scale, sampling_ratio, aligned)
    out = torch.zeros(len(bins), C, dtype=torch.float64)
    for r, (samples, _) in enumerate(bins):
        vals = [sum((X[i] * w for i, w in taps), torch.zeros(C, dtype=torch.float64))
                for taps in samples]
        if vals:
            out[r] = torch.stack(vals).amax(0)
    out_h, out_w = output_size
    return out.reshape(boxes.shape[0], out_h, out_w, C).permute(0, 3, 1, 2)


# ------------------------------------------------------------------ RoIPool oracle

def _c_round(v):
    """C/C++ round(): halfway cases away from zero (Python's round() is half-to-even)."""
    return int(math.copysign(math.floor(abs(v) + 0.5), v))


def _pool_bins(box, out_h, out_w, scale, H, W, rnd=_c_round, plus_one=True):
    _, x1, y1, x2, y2 = box
    sw, sh, ew, eh = rnd(x1 * scale), rnd(y1 * scale), rnd(x2 * scale), rnd(y2 * scale)
    extra = 1 if plus_one else 0
    rw, rh = max(ew - sw + extra, 1), max(eh - sh + extra, 1)
    bh, bw = rh / out_h, rw / out_w
    for ph in range(out_h):
        hs = min(max(math.floor(ph * bh) + sh, 0), H)
        he = min(max(math.ceil((ph + 1) * bh) + sh, 0), H)
        for pw in range(out_w):
            ws = min(max(math.floor(pw * bw) + sw, 0), W)
            we = min(max(math.ceil((pw + 1) * bw) + sw, 0), W)
            yield ph, pw, hs, he, ws, we


def _ref_pool(inp, boxes, output_size, scale=1.0, **variant):
    N, C, H, W = inp.shape
    out_h, out_w = output_size
    x = inp.detach().double()
    out = torch.zeros(boxes.shape[0], C, out_h, out_w, dtype=torch.float64)
    for k, box in enumerate(boxes.tolist()):
        n = int(box[0])
        for ph, pw, hs, he, ws, we in _pool_bins(box, out_h, out_w, scale, H, W, **variant):
            if he > hs and we > ws:
                out[k, :, ph, pw] = x[n, :, hs:he, ws:we].amax(dim=(1, 2))
    return out


def _pool_arith_agrees(box, out_h, out_w, scale):
    """True if float32, float64 and exact integer arithmetic give the same bin boundaries."""
    _, x1, y1, x2, y2 = box
    for s, e, out in ((_c_round(x1 * scale), _c_round(x2 * scale), out_w),
                      (_c_round(y1 * scale), _c_round(y2 * scale), out_h)):
        r = max(e - s + 1, 1)
        b64 = r / out
        b32 = torch.tensor(float(r), dtype=torch.float32) / out
        for p in range(out):
            exact = ((p * r) // out, -((-(p + 1) * r) // out))
            f64 = (math.floor(p * b64), math.ceil((p + 1) * b64))
            f32 = (int(torch.floor(p * b32)), int(torch.ceil((p + 1) * b32)))
            if not exact == f64 == f32:
                return False
    return True


# ------------------------------------------------------------------ case generation

def _align_safe(box, H, W, out_h, out_w, scale, sampling_ratio, aligned, margin=1e-4):
    """No sample within `margin` of a discontinuity (the -1 / H / W cut-offs, or an integer
    ratio roi/out that decides the adaptive grid size)."""
    sh, sw, bh, bw, gh, gw, rh, rw = _box_geometry(box, out_h, out_w, scale, sampling_ratio, aligned)
    if sampling_ratio <= 0:
        for r in (rh / out_h, rw / out_w):
            if r != 0.0 and abs(r - round(r)) < margin:   # exact 0 (zero-size box) is exact anywhere
                return False
    for start, b, g, out, size in ((sh, bh, gh, out_h, H), (sw, bw, gw, out_w, W)):
        for p in range(out):
            for i in range(g):
                s = start + p * b + (i + 0.5) * b / g
                if abs(s + 1.0) < margin or abs(s - size) < margin:
                    return False
    return True


def _random_boxes(g, K, N, H, W, scale, out_hw, sampling_ratio, aligned, kind="inside"):
    """K boxes in IMAGE coordinates ((K, 5) float64) for a feature map of H x W at `scale`."""
    Himg, Wimg = H / scale, W / scale
    rows, tries = [], 0
    while len(rows) < K:
        tries += 1
        if tries > 10_000 * max(K, 1):   # a config whose boxes are all near a discontinuity
            raise RuntimeError(f"grader bug: no safe {kind!r} boxes for H={H} W={W} "
                               f"out={out_hw} sr={sampling_ratio} aligned={aligned}")
        u = torch.rand(6, generator=g, dtype=torch.float64).tolist()
        n = int(u[0] * N)
        if kind == "inside":
            x1, y1 = u[1] * Wimg * 0.7, u[2] * Himg * 0.7
            w = (0.1 + 0.9 * u[3]) * (Wimg - 1 - x1)
            h = (0.1 + 0.9 * u[4]) * (Himg - 1 - y1)
        elif kind == "outside":   # partly or fully outside the image, on any side
            x1 = (u[1] * 1.8 - 0.6) * Wimg
            y1 = (u[2] * 1.8 - 0.6) * Himg
            w, h = u[3] * Wimg * 0.9, u[4] * Himg * 0.9
        elif kind == "tiny":      # smaller than one feature-map pixel, some exactly zero-size
            x1, y1 = u[1] * (Wimg - 1), u[2] * (Himg - 1)
            w = 0.0 if u[5] < 0.25 else u[3] * 0.9 / scale
            h = 0.0 if u[5] > 0.75 else u[4] * 0.9 / scale
        else:                     # "any": mixture
            x1 = (u[1] * 1.3 - 0.2) * Wimg
            y1 = (u[2] * 1.3 - 0.2) * Himg
            w, h = u[3] * Wimg * 0.8, u[4] * Himg * 0.8
        box = [float(n), x1, y1, x1 + w, y1 + h]
        if _align_safe(box, H, W, out_hw[0], out_hw[1], scale, sampling_ratio, aligned):
            rows.append(box)
    return torch.tensor(rows, dtype=torch.float64).reshape(K, 5)


def _feature_map(g, N, C, H, W, dtype=torch.float64):
    return torch.randn(N, C, H, W, generator=g, dtype=torch.float64).to(dtype)


# ------------------------------------------------------------------ comparison helpers

def _call_align(ns, inp, boxes, output_size, scale, sr, aligned):
    got = ns.roi_align(inp.clone(), boxes.clone(), output_size, spatial_scale=scale,
                       sampling_ratio=sr, aligned=aligned)
    K, C = boxes.shape[0], inp.shape[1]
    want_shape = (K, C, output_size[0], output_size[1])
    assert torch.is_tensor(got) and tuple(got.shape) == want_shape, (
        f"roi_align on input {tuple(inp.shape)} with {K} boxes and output_size={output_size} must "
        f"return shape {want_shape}, got "
        f"{tuple(got.shape) if torch.is_tensor(got) else type(got).__name__}")
    assert got.dtype == inp.dtype, f"roi_align output dtype should match the input ({inp.dtype}), got {got.dtype}"
    return got


def _diagnose_align(got, inp, boxes, output_size, scale, sr, aligned, tol):
    """Name the classic bug that `got` reproduces, if any."""
    cands = []
    if aligned:
        cands.append(("you did not shift the box by -0.5 pixel: with aligned=True subtract 0.5 "
                      "AFTER multiplying by spatial_scale (and skip the min-size-1 clamp)",
                      lambda: _ref_align(inp, boxes, output_size, scale, sr, False)))
        cands.append(("the -0.5 shift is missing (the min-size rule is right)",
                      lambda: _ref_align(inp, boxes, output_size, scale, sr, True, offset=0.0)))
        if scale != 1.0:
            cands.append(("you subtracted 0.5 before scaling: it is coord * spatial_scale - 0.5, "
                          "not (coord - 0.5) * spatial_scale",
                          lambda: _ref_align(inp, boxes, output_size, scale, sr, True,
                                             scale_after_offset=True)))
    else:
        cands.append(("aligned=False must NOT shift the box by -0.5",
                      lambda: _ref_align(inp, boxes, output_size, scale, sr, True)))
        cands.append(("with aligned=False the RoI width/height is clamped to at least 1 "
                      "(torchvision: 'Force malformed ROIs to be 1x1')",
                      lambda: _ref_align(inp, boxes, output_size, scale, sr, False, min_size=False)))
    if sr <= 0:
        for fixed in (1, 2):
            cands.append((f"sampling_ratio={sr} means an ADAPTIVE grid of ceil(roi_h / out_h) x "
                          f"ceil(roi_w / out_w) samples per bin (per box), but you used a fixed "
                          f"{fixed} x {fixed}",
                          lambda fixed=fixed: _ref_align(inp, boxes, output_size, scale, fixed, aligned)))
    cands.append(("samples beyond the cut-off (y < -1 or y > H, x < -1 or x > W) must contribute "
                  "0 — you clamped them to the border instead",
                  lambda: _ref_align(inp, boxes, output_size, scale, sr, aligned, clamp_outside=True)))
    cands.append(("samples with -1 <= y < 0 (or H - 1 < y <= H) must be CLAMPED to the border "
                  "row/column, not blended with zeros (that is grid_sample's padding_mode='zeros'; "
                  "use 'border' plus a mask for the cut-off)",
                  lambda: _ref_align(inp, boxes, output_size, scale, sr, aligned, zero_pad=True)))
    cands.append(("each bin is the AVERAGE of its samples, not the max",
                  lambda: _ref_align_max(inp, boxes, output_size, scale, sr, aligned)))
    if inp.shape[0] > 1:
        cands.append(("boxes[:, 0] is the batch index: crop from input[batch_index], not input[0]",
                      lambda: _ref_align(inp, boxes, output_size, scale, sr, aligned, ignore_batch=True)))
    if scale != 1.0:
        cands.append(("box coordinates are in image space: multiply them by spatial_scale",
                      lambda: _ref_align(inp, boxes, output_size, 1.0, sr, aligned)))
    g = got.detach().double()
    for msg, fn in cands:
        try:
            alt = fn()
        except Exception:
            continue
        if torch.isfinite(alt).all() and (g - alt).abs().max().item() <= tol:
            return msg
    return None


def _compare_align(ns, inp, boxes, output_size, scale, sr, aligned, what, tol=1e-8):
    got = _call_align(ns, inp, boxes, output_size, scale, sr, aligned)
    want = _ref_align(inp, boxes, output_size, scale, sr, aligned)
    err = (got.detach().double() - want).abs()
    if not torch.isfinite(got).all():
        k = int((~torch.isfinite(got)).flatten().nonzero()[0]) // (got[0].numel())
        raise AssertionError(
            f"{what}: roi_align returned NaN/Inf for box {boxes[k].tolist()} — "
            f"divide by max(grid_h * grid_w, 1), never by 0")
    if err.max().item() > tol:
        k = int(err.flatten().argmax()) // got[0].numel()
        diag = _diagnose_align(got, inp, boxes, output_size, scale, sr, aligned, max(tol, 1e-6))
        raise AssertionError(
            f"{what} (output_size={output_size}, spatial_scale={scale}, sampling_ratio={sr}, "
            f"aligned={aligned}): max |diff| vs a line-by-line port of torchvision's kernel = "
            f"{err.max().item():.3e}, worst box [batch, x1, y1, x2, y2] = "
            f"{[round(v, 4) for v in boxes[k].tolist()]}."
            + (f" Likely cause: {diag}." if diag else
               " Check the sample positions start + p*bin + (i + 0.5)*bin/grid, the cut-off rule "
               "and the averaging. If you use F.grid_sample, its normalised coordinate must match "
               "its align_corners flag: (2x + 1)/W - 1 for False, 2x/(W - 1) - 1 for True."))


# ------------------------------------------------------------------ RoIAlign checks

def check_roi_align_shape_dtype_and_no_boxes(ns):
    g = torch.Generator().manual_seed(0)
    inp = _feature_map(g, 2, 3, 9, 11)
    boxes = torch.tensor([[1.0, 1.0, 2.0, 7.5, 6.0], [0.0, 0.0, 0.0, 10.0, 8.0]], dtype=torch.float64)
    for oh, ow in ((3, 3), (2, 5), (1, 1)):
        _call_align(ns, inp, boxes, (oh, ow), 1.0, 2, False)
    for dtype in (torch.float32, torch.float64):
        got = ns.roi_align(inp.to(dtype), boxes.to(dtype), (4, 2), spatial_scale=1.0,
                           sampling_ratio=-1, aligned=True)
        assert got.dtype == dtype, f"output dtype should match the input ({dtype}), got {got.dtype}"
    empty = ns.roi_align(inp, torch.zeros(0, 5, dtype=torch.float64), (7, 7))
    assert torch.is_tensor(empty) and tuple(empty.shape) == (0, 3, 7, 7), (
        f"with no boxes roi_align must return an empty (0, C, out_h, out_w) tensor, got "
        f"{tuple(empty.shape) if torch.is_tensor(empty) else type(empty).__name__} "
        f"(torch.stack of an empty list fails — handle K = 0)")


def check_roi_align_fixed_sampling_ratio(ns):
    g = torch.Generator().manual_seed(1)
    inp = _feature_map(g, 1, 3, 12, 14)
    for oh, ow, sr in ((2, 2, 2), (3, 5, 1), (4, 3, 3), (1, 1, 2)):
        boxes = _random_boxes(g, 8, 1, 12, 14, 1.0, (oh, ow), sr, False, "inside")
        _compare_align(ns, inp, boxes, (oh, ow), 1.0, sr, False,
                       "boxes inside a single image, aligned=False, fixed sampling grid")


def check_roi_align_aligned_half_pixel_shift(ns):
    g = torch.Generator().manual_seed(2)
    inp = _feature_map(g, 1, 2, 11, 13)
    for oh, ow, sr in ((2, 2, 2), (3, 4, 1), (5, 2, 3)):
        boxes = _random_boxes(g, 8, 1, 11, 13, 1.0, (oh, ow), sr, True, "inside")
        _compare_align(ns, inp, boxes, (oh, ow), 1.0, sr, True, "aligned=True (half-pixel shift)")


def check_roi_align_adaptive_sampling_grid(ns):
    g = torch.Generator().manual_seed(3)
    inp = _feature_map(g, 1, 2, 16, 18)
    for oh, ow, sr, aligned in ((2, 2, -1, False), (3, 2, 0, False), (2, 3, -1, True), (1, 1, 0, True)):
        boxes = _random_boxes(g, 8, 1, 16, 18, 1.0, (oh, ow), sr, aligned, "inside")
        _compare_align(ns, inp, boxes, (oh, ow), 1.0, sr, aligned,
                       "adaptive sampling grid (sampling_ratio <= 0)")


def check_roi_align_spatial_scale_and_batch_index(ns):
    g = torch.Generator().manual_seed(4)
    inp = _feature_map(g, 3, 3, 10, 12)
    for scale, sr, aligned in ((0.5, 2, False), (0.25, -1, True), (0.125, 2, True), (0.5, 0, False)):
        boxes = _random_boxes(g, 10, 3, 10, 12, scale, (3, 3), sr, aligned, "any")
        _compare_align(ns, inp, boxes, (3, 3), scale, sr, aligned,
                       "three images in the batch, spatial_scale != 1")


def check_roi_align_boxes_outside_the_image(ns):
    g = torch.Generator().manual_seed(5)
    inp = _feature_map(g, 2, 2, 9, 10)
    for scale, sr, aligned, out in ((1.0, 2, False, (3, 3)), (1.0, -1, True, (2, 4)),
                                    (0.5, 3, True, (4, 2)), (0.5, -1, False, (2, 2))):
        boxes = _random_boxes(g, 12, 2, 9, 10, scale, out, sr, aligned, "outside")
        _compare_align(ns, inp, boxes, out, scale, sr, aligned,
                       "boxes partly or fully outside the feature map")


def check_roi_align_exact_cutoff_rule(ns):
    # Dyadic coordinates: every sample position below is computed exactly in any op order.
    # A 1x1 output with sampling_ratio=1 and a box of height/width 1 samples its centre.
    H, W = 4, 5
    inp = (torch.arange(H * W, dtype=torch.float64).view(1, 1, H, W) + 1) ** 1.5
    f = inp[0, 0]

    def at(yc, xc):
        b = torch.tensor([[0.0, xc - 0.5, yc - 0.5, xc + 0.5, yc + 0.5]], dtype=torch.float64)
        return float(ns.roi_align(inp, b, (1, 1), spatial_scale=1.0, sampling_ratio=1,
                                  aligned=False).reshape(-1)[0])

    cases = [
        ((-1.0, 2.0), float(f[0, 2]), "y = -1 exactly is still inside the cut-off (only y < -1 "
                                      "is dropped); it is clamped to row 0"),
        ((-1.0625, 2.0), 0.0, "y = -1.0625 < -1 must contribute 0 (dropped, not clamped)"),
        ((-0.5, 2.5), float(0.5 * (f[0, 2] + f[0, 3])),
         "y = -0.5 must be clamped to row 0 (not blended with a zero row)"),
        ((float(H), 1.0), float(f[H - 1, 1]), "y = H exactly is still inside (only y > H is dropped); "
                                              "it reads row H - 1"),
        ((H + 0.0625, 1.0), 0.0, "y = H + 0.0625 > H must contribute 0"),
        ((H - 0.5, 1.0), float(f[H - 1, 1]), "H - 1 < y < H must read row H - 1 with full weight"),
        ((1.5, -1.0), float(0.5 * (f[1, 0] + f[2, 0])), "x = -1 exactly is clamped to column 0"),
        ((1.5, -1.25), 0.0, "x = -1.25 < -1 must contribute 0"),
        ((2.0, float(W)), float(f[2, W - 1]), "x = W exactly reads column W - 1"),
        ((2.0, W + 0.25), 0.0, "x = W + 0.25 > W must contribute 0"),
        ((1.25, 2.75), float(0.75 * 0.25 * f[1, 2] + 0.75 * 0.75 * f[1, 3]
                             + 0.25 * 0.25 * f[2, 2] + 0.25 * 0.75 * f[2, 3]),
         "an interior sample is the ordinary 4-tap bilinear blend"),
    ]
    for (yc, xc), want, why in cases:
        got = at(yc, xc)
        assert abs(got - want) <= 1e-9, (
            f"single sample at (y={yc}, x={xc}) on a {H}x{W} map: expected {want:.6f}, got "
            f"{got:.6f}. Rule: {why}.")


def check_roi_align_tiny_and_degenerate_boxes(ns):
    g = torch.Generator().manual_seed(6)
    inp = _feature_map(g, 2, 3, 8, 9)
    for sr, aligned in ((-1, False), (2, False), (-1, True), (2, True)):
        boxes = _random_boxes(g, 10, 2, 8, 9, 1.0, (2, 2), sr, aligned, "tiny")
        _compare_align(ns, inp, boxes, (2, 2), 1.0, sr, aligned,
                       "boxes smaller than a pixel (some zero-width or zero-height)")
    # aligned=True + adaptive grid + zero-size box -> ceil(0 / out) = 0 samples -> output 0.
    z = torch.tensor([[1.0, 3.0, 2.0, 3.0, 6.0], [0.0, 2.0, 4.0, 5.0, 4.0]], dtype=torch.float64)
    got = ns.roi_align(inp, z, (2, 2), spatial_scale=1.0, sampling_ratio=-1, aligned=True)
    assert torch.isfinite(got).all(), (
        "aligned=True with a zero-width (or zero-height) box and sampling_ratio=-1 gives a "
        "0-sample grid; you divided by 0. torchvision divides by max(grid_h * grid_w, 1) and "
        "returns 0 for such boxes")
    assert got.abs().max().item() == 0.0, (
        f"aligned=True, zero-size box, adaptive grid: the grid has ceil(0 / out) = 0 samples, so "
        f"the output must be exactly 0, got max |value| {got.abs().max().item():.4f}")


def check_roi_align_float32(ns):
    g = torch.Generator().manual_seed(7)
    inp = _feature_map(g, 2, 3, 10, 12)
    for scale, sr, aligned in ((1.0, -1, False), (0.5, 2, True), (0.5, -1, True)):
        boxes = _random_boxes(g, 10, 2, 10, 12, scale, (3, 2), sr, aligned, "any")
        _compare_align(ns, inp.float(), boxes.float(), (3, 2), scale, sr, aligned,
                       "float32 input and boxes", tol=2e-4)


def check_roi_align_gradient_wrt_input(ns):
    g = torch.Generator().manual_seed(8)
    for scale, sr, aligned, out, kind in ((1.0, 2, False, (3, 3), "inside"),
                                          (0.5, -1, True, (2, 3), "any"),
                                          (1.0, -1, False, (2, 2), "outside")):
        inp = _feature_map(g, 2, 2, 8, 9)
        boxes = _random_boxes(g, 7, 2, 8, 9, scale, out, sr, aligned, kind)
        boxes[1] = boxes[0]                       # two identical boxes: gradients must add up
        x = inp.clone().requires_grad_(True)
        got = ns.roi_align(x, boxes.clone(), out, spatial_scale=scale, sampling_ratio=sr,
                           aligned=aligned)
        assert torch.is_tensor(got) and got.requires_grad, (
            "roi_align output is detached from the input — avoid .item()/.tolist()/numpy on "
            "feature values; gather with tensor indexing so autograd can scatter the gradient back")
        up = torch.randn(got.shape, generator=g, dtype=torch.float64)
        (got * up).sum().backward()
        assert x.grad is not None, "no gradient reached the input feature map"
        M = _align_matrix(tuple(inp.shape), boxes, out, scale, sr, aligned)
        K, C = boxes.shape[0], inp.shape[1]
        G = up.permute(0, 2, 3, 1).reshape(K * out[0] * out[1], C)
        want = (M.T @ G).reshape(2, 8, 9, C).permute(0, 3, 1, 2)
        err = (x.grad - want).abs().max().item()
        assert err <= 1e-8, (
            f"d(sum(out * upstream))/d(input) disagrees with the transpose of the RoIAlign "
            f"weights (max |diff| {err:.2e}; spatial_scale={scale}, sampling_ratio={sr}, "
            f"aligned={aligned}). Every input pixel must collect weight * upstream / count from "
            f"every sample of every box that reads it")


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


def check_roi_align_is_vectorized(ns):
    g = torch.Generator().manual_seed(9)
    boxes = torch.tensor([[0.0, 1.0, 1.5, 9.0, 8.0], [1.0, 0.5, 2.0, 6.0, 9.5],
                          [0.0, -2.0, 3.0, 5.0, 11.0]], dtype=torch.float64)
    small = _feature_map(g, 2, 1, 12, 12)
    big = _feature_map(g, 2, 6, 12, 12)
    for sr_small, sr_big in ((1, 3), (-1, -1)):
        n_small = _count_torch_calls(ns.roi_align, small, boxes, (2, 2), spatial_scale=1.0,
                                     sampling_ratio=sr_small, aligned=True)
        n_big = _count_torch_calls(ns.roi_align, big, boxes, (7, 7), spatial_scale=1.0,
                                   sampling_ratio=sr_big, aligned=True)
        assert n_big <= n_small + 12, (
            f"roi_align made {n_small} torch calls for 3 boxes with C=1, a 2x2 output and "
            f"sampling_ratio={sr_small}, but {n_big} for the same 3 boxes with C=6, a 7x7 output "
            f"and sampling_ratio={sr_big} — the work grows with a Python loop over channels, bins "
            f"or sampling points. Only the loop over boxes is allowed; build all sample positions "
            f"of a box at once and gather with tensor indexing.")


# ------------------------------------------------------------------ RoIPool checks

def _pool_boxes(g, K, N, H, W, scale, out, kind="any"):
    """Boxes on a 1/8-pixel grid in image space (so coord * scale is exact)."""
    Himg, Wimg = H / scale, W / scale
    rows = []
    while len(rows) < K:
        u = torch.rand(5, generator=g, dtype=torch.float64).tolist()
        q = lambda v: round(v * 8) / 8
        if kind == "inside":
            x1, y1 = q(u[1] * Wimg * 0.6), q(u[2] * Himg * 0.6)
            x2, y2 = q(x1 + u[3] * (Wimg - 1 - x1)), q(y1 + u[4] * (Himg - 1 - y1))
        else:
            x1, y1 = q((u[1] * 1.6 - 0.4) * Wimg), q((u[2] * 1.6 - 0.4) * Himg)
            x2, y2 = q(x1 + u[3] * Wimg * 0.9), q(y1 + u[4] * Himg * 0.9)
        box = [float(int(u[0] * N)), x1, y1, x2, y2]
        if _pool_arith_agrees(box, out[0], out[1], scale):
            rows.append(box)
    return torch.tensor(rows, dtype=torch.float64)


def _compare_pool(ns, inp, boxes, out, scale, what):
    got = ns.roi_pool(inp.clone(), boxes.clone(), out, spatial_scale=scale)
    want_shape = (boxes.shape[0], inp.shape[1], out[0], out[1])
    assert torch.is_tensor(got) and tuple(got.shape) == want_shape, (
        f"roi_pool must return shape {want_shape}, got "
        f"{tuple(got.shape) if torch.is_tensor(got) else type(got).__name__}")
    if not torch.isfinite(got).all():
        raise AssertionError(
            f"{what}: roi_pool returned -inf/NaN — a bin that is clipped away entirely (the box "
            f"hangs off the feature map) is empty and must be 0, not the max of nothing")
    want = _ref_pool(inp, boxes, out, scale)
    err = (got.detach().double() - want).abs()
    if err.max().item() > 1e-6:
        k = int(err.flatten().argmax()) // got[0].numel()
        diag = None
        for msg, kw in (("round with C's round() (half AWAY from zero): torch.round / Python "
                         "round() send 2.5 to 2, torchvision sends it to 3",
                         dict(rnd=lambda v: int(torch.round(torch.tensor(v))))),
                        ("the RoI size is end - start + 1 (inclusive pixel ends, the legacy "
                         "+1 convention)", dict(plus_one=False)),
                        ("box coordinates must be rounded (not truncated/floored) after scaling",
                         dict(rnd=math.floor))):
            if (got.detach().double() - _ref_pool(inp, boxes, out, scale, **kw)).abs().max() <= 1e-6:
                diag = msg
                break
        raise AssertionError(
            f"{what} (output_size={out}, spatial_scale={scale}): max |diff| vs torchvision's "
            f"roi_pool kernel = {err.max().item():.3e}, worst box {boxes[k].tolist()}."
            + (f" Likely cause: {diag}." if diag else
               " Bin p spans [floor(p * bin), ceil((p + 1) * bin)) shifted by the rounded start "
               "and clipped to [0, H]; take the max; an empty bin is 0."))


def check_roi_pool_matches_reference(ns):
    g = torch.Generator().manual_seed(10)
    inp = _feature_map(g, 3, 3, 10, 13)
    for scale, out, kind in ((1.0, (2, 2), "inside"), (1.0, (3, 4), "any"), (0.5, (2, 3), "any"),
                             (0.25, (4, 4), "any"), (0.125, (1, 1), "any")):
        boxes = _pool_boxes(g, 12, 3, 10, 13, scale, out, kind)
        _compare_pool(ns, inp, boxes, out, scale, "roi_pool on random boxes")
    _compare_pool(ns, inp.float(), _pool_boxes(g, 8, 3, 10, 13, 0.5, (3, 3)).float(), (3, 3), 0.5,
                  "roi_pool with float32 input")


def check_roi_pool_rounding_and_empty_bins(ns):
    # A shuffled ramp: distinct values with no monotone structure, so a wrong bin cannot
    # accidentally contain the same maximum as the right one.
    inp = torch.randperm(2 * 6 * 7, generator=torch.Generator().manual_seed(11)) \
        .to(torch.float64).view(2, 1, 6, 7)
    # Scaled coordinates hit .5 exactly: 2.5 -> 3 and 0.5 -> 1 with C round (torch.round: 2, 0).
    boxes = torch.tensor([[0.0, 1.0, 1.0, 5.0, 5.0],      # scale 0.5: x/y 0.5 -> 1, 2.5 -> 3
                          [1.0, 3.0, 1.0, 9.0, 5.0]], dtype=torch.float64)
    _compare_pool(ns, inp, boxes, (2, 2), 0.5, "coordinates that land exactly on .5 after scaling")
    # Negative halves round away from zero too: -0.5 -> -1 (then clipped to the image).
    neg = torch.tensor([[0.0, -1.0, -1.0, 6.0, 4.0]], dtype=torch.float64)
    _compare_pool(ns, inp, neg, (3, 3), 0.5, "negative coordinates that land on -0.5 after scaling")
    # Fully outside -> every bin empty -> zeros, not -inf / -FLT_MAX.
    out = ns.roi_pool(inp, torch.tensor([[1.0, 20.0, 15.0, 30.0, 25.0], [0.0, -9.0, -9.0, -3.0, -2.0]],
                                        dtype=torch.float64), (2, 2), spatial_scale=1.0)
    assert torch.isfinite(out).all() and out.abs().max().item() == 0.0, (
        f"a box entirely outside the feature map has only empty bins, which must be 0 "
        f"(got {out.flatten().tolist()[:4]}...)")
    # A box partly outside: bins that fall off the map are 0, the others are maxima.
    _compare_pool(ns, inp, torch.tensor([[1.0, 4.0, -6.0, 12.0, 3.0]], dtype=torch.float64), (4, 4),
                  1.0, "a box hanging over the top-right corner")


CHECKS = [check_roi_align_shape_dtype_and_no_boxes, check_roi_align_fixed_sampling_ratio,
          check_roi_align_aligned_half_pixel_shift, check_roi_align_adaptive_sampling_grid,
          check_roi_align_spatial_scale_and_batch_index, check_roi_align_boxes_outside_the_image,
          check_roi_align_exact_cutoff_rule, check_roi_align_tiny_and_degenerate_boxes,
          check_roi_align_float32, check_roi_align_gradient_wrt_input, check_roi_align_is_vectorized,
          check_roi_pool_matches_reference, check_roi_pool_rounding_and_empty_bins]
