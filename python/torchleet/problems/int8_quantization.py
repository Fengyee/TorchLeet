"""int8-quantization — post-training weight/activation quantization from scratch.

Quantizers are graded against the definitions, with the grader's own
dequantization ((q - zp) * scale in float64) so a bug in the solver's
`dequantize` cannot hide a bug in the quantizer:
  * symmetric: q = clamp(round(x / s), -qmax, qmax), s = max|x| / qmax with
    qmax = 2^(b-1) - 1, per tensor or per channel along `axis`; int8 storage;
    |dequant - x| <= s/2 everywhere; all-zero tensors/channels give a positive,
    finite scale and no NaN;
  * asymmetric: uint8 storage, range forced to include 0, integer zero point,
    0.0 round-trips exactly, |dequant - x| <= s/2;
  * per-channel beats per-tensor on a weight with one outlier channel;
  * int8_linear equals dequant(x) @ dequant(w)^T, and with unit scales it is
    the EXACT integer result rounded once — float32 accumulation of products
    that reach 2^24 is not exact and fails that check; int8 @ int8 (which
    wraps around in PyTorch) fails the oracle check.
"""
import torch

ENTRIES = ["quantize_symmetric", "quantize_asymmetric", "dequantize", "int8_linear"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Symmetric: qmax = 2^(b-1) - 1 (127 for 8 bits) and the scale maps the "
    "largest |x| exactly onto qmax. Asymmetric: the 2^b - 1 steps span "
    "[min(x.min(), 0), max(x.max(), 0)] and the zero point is the integer code "
    "that represents 0.0.",
    "Per-channel along `axis` means one amax per index of that dimension: reduce "
    "|x| over every OTHER dim, then reshape the (C,) scale to broadcast (size C "
    "at `axis`, 1 elsewhere). Guard amax == 0 so the scale stays positive. Round "
    "with torch.round before casting — .to(torch.int8) alone truncates toward "
    "zero.",
    "int8_linear: upcast before subtracting the zero point (uint8 - int "
    "underflows) and before the matmul (int8 @ int8 stays int8 and wraps "
    "around). (x_q.int() - zp) @ w_q.int().T is exact in int32; then one float "
    "multiply by x_scale * w_scale, where a per-channel w_scale of shape "
    "(out_features,) broadcasts over the last dim of the (M, out_features) "
    "result.",
]


def _f64(t):
    return torch.as_tensor(t).detach().to(torch.float64)


def _zp(z):
    if z is None:
        return 0
    z = torch.as_tensor(z)
    assert z.numel() == 1, f"zero_point should be a single integer, got shape {tuple(z.shape)}"
    zf = float(z)
    assert zf == round(zf), (
        f"zero_point must be an INTEGER (got {zf}) — otherwise 0.0 is not exactly "
        f"representable and zero-padding / ReLU zeros pick up an error")
    return int(round(zf))


def _bcast(scale, ndim, axis):
    s = _f64(scale)
    if axis is None:
        return s.reshape(())
    shape = [1] * ndim
    shape[axis] = -1
    return s.reshape(shape)


def _sym(ns, x, num_bits=8, axis=None):
    out = ns.quantize_symmetric(x, num_bits=num_bits, axis=axis)
    assert isinstance(out, tuple) and len(out) == 2, \
        "quantize_symmetric should return a tuple (q, scale)"
    return out


def _asym(ns, x, num_bits=8):
    out = ns.quantize_asymmetric(x, num_bits=num_bits)
    assert isinstance(out, tuple) and len(out) == 3, \
        "quantize_asymmetric should return a tuple (q, scale, zero_point)"
    return out


def _check_sym_result(x, q, scale, num_bits, axis, label):
    qmax = 2 ** (num_bits - 1) - 1
    assert torch.is_tensor(q) and q.dtype == torch.int8, (
        f"{label}: q should be a torch.int8 tensor, got "
        f"{q.dtype if torch.is_tensor(q) else type(q).__name__}")
    assert tuple(q.shape) == tuple(x.shape), \
        f"{label}: q has shape {tuple(q.shape)}, expected {tuple(x.shape)}"
    lo, hi = int(q.min()), int(q.max())
    assert -qmax <= lo and hi <= qmax, (
        f"{label}: q spans [{lo}, {hi}] but symmetric {num_bits}-bit codes live in "
        f"[-{qmax}, {qmax}] (the most negative code -{qmax + 1} is left unused so "
        f"the grid is symmetric)")
    s = _f64(scale)
    xd = x.to(torch.float64)
    if axis is None:
        assert s.numel() == 1, \
            f"{label}: per-tensor scale should be a single number, got shape {tuple(s.shape)}"
        amax = xd.abs().max()
        want = amax / qmax if amax > 0 else None
    else:
        C = x.shape[axis]
        assert tuple(s.shape) == (C,), (
            f"{label}: per-channel scale should have shape ({C},) — one scale per "
            f"index of dim {axis} — got {tuple(s.shape)}")
        dims = [d for d in range(x.ndim) if d != axis % x.ndim]
        amax = xd.abs().amax(dim=dims)
        want = torch.where(amax > 0, amax / qmax, s)       # zero channels: anything > 0
    assert torch.isfinite(s).all() and (s > 0).all(), (
        f"{label}: every scale must be positive and finite (got "
        f"{s.flatten()[:4].tolist()}...) — guard all-zero tensors/channels")
    if want is not None:
        assert torch.allclose(s.reshape(_f64(want).shape), _f64(want), rtol=1e-5, atol=0), (
            f"{label}: scale should be max|x| / {qmax} so the largest magnitude maps "
            f"exactly to ±{qmax} (got {s.flatten()[:3].tolist()}, expected "
            f"{_f64(want).flatten()[:3].tolist()})")
    sb = _bcast(s, x.ndim, None if axis is None else axis % x.ndim)
    err = (q.to(torch.float64) * sb - xd).abs()
    bad = err > sb / 2 * (1 + 1e-5) + 1e-12
    assert not bad.any(), (
        f"{label}: |dequant(q) - x| reached {(err / sb.expand_as(err)).max():.3f}·scale, "
        f"but rounding to the nearest code bounds it by scale/2 — use torch.round "
        f"(casting truncates toward zero)")


def check_symmetric_per_tensor(ns):
    torch.manual_seed(0)
    for num_bits, x in ((8, torch.randn(64, 33) * 3.0),
                        (8, torch.tensor([-2.5, 0.0, 0.7, 1.0, 2.5])),
                        (4, torch.randn(200))):
        q, s = _sym(ns, x, num_bits=num_bits, axis=None)
        _check_sym_result(x, q, s, num_bits, None, f"symmetric {num_bits}-bit per-tensor")
        qmax = 2 ** (num_bits - 1) - 1
        assert int(q.abs().max()) == qmax, (
            f"symmetric {num_bits}-bit: the largest |x| should map to ±{qmax}, but "
            f"max|q| = {int(q.abs().max())}")


def check_symmetric_per_channel(ns):
    torch.manual_seed(1)
    w = torch.randn(16, 24) * torch.logspace(-2, 1, 16).unsqueeze(1)    # rows differ in range
    for axis in (0, 1, -1):
        q, s = _sym(ns, w, axis=axis)
        _check_sym_result(w, q, s, 8, axis, f"per-channel axis={axis}")
    conv = torch.randn(6, 4, 3, 3)
    q, s = _sym(ns, conv, axis=0)
    _check_sym_result(conv, q, s, 8, 0, "per-channel axis=0 on a (6, 4, 3, 3) conv weight")
    x3 = torch.randn(5, 7, 3)
    q, s = _sym(ns, x3, axis=1)
    _check_sym_result(x3, q, s, 8, 1, "per-channel axis=1 on a (5, 7, 3) tensor")
    wz = torch.randn(4, 10)
    wz[2] = 0.0
    q, s = _sym(ns, wz, axis=0)
    _check_sym_result(wz, q, s, 8, 0, "per-channel with an all-zero channel")
    assert (q[2] == 0).all(), "an all-zero channel must quantize to all-zero codes"


def check_per_channel_beats_per_tensor_with_an_outlier_channel(ns):
    torch.manual_seed(2)
    w = torch.randn(32, 64) * 0.02
    w[5] *= 100.0                                   # one outlier output channel
    q_t, s_t = _sym(ns, w, axis=None)
    q_c, s_c = _sym(ns, w, axis=0)
    err_t = (q_t.double() * _f64(s_t) - w.double()).pow(2).mean(dim=1)
    err_c = (q_c.double() * _f64(s_c).reshape(-1, 1) - w.double()).pow(2).mean(dim=1)
    assert (err_c <= err_t * (1 + 1e-6) + 1e-15).all(), (
        "per-channel MSE should be <= per-tensor MSE for every channel — each "
        "channel's own scale is never larger than the global one")
    normal = torch.ones(32, dtype=torch.bool)
    normal[5] = False
    ratio = (err_t[normal].mean() / err_c[normal].mean()).item()
    assert ratio > 100, (
        f"on the 31 normal channels per-channel quantization should be orders of "
        f"magnitude more accurate (the outlier no longer sets their step size), "
        f"but the MSE only improved {ratio:.1f}x — is the per-channel amax reduced "
        f"over the right dims?")


def _check_asym(x, q, s, z, num_bits, label):
    qmax = 2 ** num_bits - 1
    assert torch.is_tensor(q) and q.dtype == torch.uint8, (
        f"{label}: q should be a torch.uint8 tensor, got "
        f"{q.dtype if torch.is_tensor(q) else type(q).__name__}")
    assert tuple(q.shape) == tuple(x.shape), \
        f"{label}: q has shape {tuple(q.shape)}, expected {tuple(x.shape)}"
    assert int(q.max()) <= qmax, f"{label}: q reaches {int(q.max())} > {qmax}"
    s = _f64(s)
    assert s.numel() == 1 and torch.isfinite(s).all() and float(s) > 0, (
        f"{label}: scale must be a single positive finite number, got {s.flatten()[:3].tolist()}")
    zp = _zp(z)
    assert 0 <= zp <= qmax, f"{label}: zero_point {zp} is outside [0, {qmax}]"
    xd = x.double()
    lo, hi = min(float(xd.min()), 0.0), max(float(xd.max()), 0.0)
    if hi > lo:
        want = (hi - lo) / qmax
        assert abs(float(s) - want) <= 1e-5 * want, (
            f"{label}: scale should be (max(x.max(), 0) - min(x.min(), 0)) / {qmax} "
            f"= {want:.6g}, got {float(s):.6g} — the range must be widened to "
            f"include 0")
    err = ((q.double() - zp) * float(s) - xd).abs()
    assert float(err.max()) <= float(s) / 2 * (1 + 1e-5) + 1e-12, (
        f"{label}: |dequant(q) - x| reached {float(err.max()) / float(s):.3f}·scale "
        f"(bound: scale/2) — q = clamp(round(x / scale) + zero_point, 0, {qmax})")
    return zp, float(s)


def check_asymmetric_quantization(ns):
    torch.manual_seed(3)
    cases = [
        ("mixed-sign input", torch.randn(300) * 2 + 0.7, 8),
        ("all-positive input in [2, 5]", torch.rand(100) * 3 + 2, 8),
        ("all-negative input", -torch.rand(100) - 1, 8),
        ("4-bit input", torch.randn(50, 4), 4),
    ]
    for label, x, bits in cases:
        q, s, z = _asym(ns, x, num_bits=bits)
        _check_asym(x, q, s, z, bits, f"asymmetric {label}")
    q, s, z = _asym(ns, torch.rand(64) * 3 + 2)
    assert _zp(z) == 0, (
        f"for an all-positive input the range is [0, max], so the zero point must "
        f"be 0, got {_zp(z)}")
    q, s, z = _asym(ns, -torch.rand(64) - 1)
    assert _zp(z) == 255, (
        f"for an all-negative input the range is [min, 0], so the zero point must "
        f"be 255, got {_zp(z)}")


def check_zero_round_trips_exactly(ns):
    torch.manual_seed(4)
    x = torch.randn(500) * 2 + 1.0
    x[::7] = 0.0                                      # exact zeros (padding, ReLU outputs)
    q, s, z = _asym(ns, x)
    zp, _ = _check_asym(x, q, s, z, 8, "asymmetric with planted zeros")
    assert (q[::7].long() == zp).all(), (
        "elements equal to 0.0 must quantize to exactly the zero point")
    deq = ns.dequantize(q, s, z)
    assert torch.is_tensor(deq) and deq.is_floating_point(), \
        "dequantize should return a floating-point tensor"
    assert (deq[::7] == 0).all(), (
        f"0.0 must round-trip EXACTLY through quantize_asymmetric -> dequantize, got "
        f"{deq[::7][deq[::7] != 0][:3].tolist()}")


def check_asymmetric_wins_on_relu_activations(ns):
    torch.manual_seed(5)
    x = torch.relu(torch.randn(4096))
    q_s, s_s = _sym(ns, x)
    q_a, s_a, z_a = _asym(ns, x)
    mse_s = (q_s.double() * _f64(s_s) - x.double()).pow(2).mean()
    mse_a = ((q_a.double() - _zp(z_a)) * _f64(s_a) - x.double()).pow(2).mean()
    assert mse_a < 0.5 * mse_s, (
        f"on non-negative (post-ReLU) activations asymmetric quantization should use "
        f"all 256 codes and roughly quarter the MSE of symmetric (which wastes the "
        f"negative half): got asym {mse_a:.3e} vs sym {mse_s:.3e}")


def check_dequantize(ns):
    g = torch.Generator().manual_seed(6)
    q8 = torch.randint(-127, 128, (5, 7), generator=g).to(torch.int8)
    out = ns.dequantize(q8, torch.tensor(0.03))
    assert torch.is_tensor(out) and out.is_floating_point() and tuple(out.shape) == (5, 7), \
        "dequantize should return a float tensor with q's shape"
    assert torch.allclose(out.double(), q8.double() * 0.03, rtol=1e-6, atol=0), \
        "symmetric per-tensor: dequantize must return q * scale"
    qu = torch.randint(0, 256, (5, 7), generator=g).to(torch.uint8)
    out = ns.dequantize(qu, 0.05, 131)
    assert torch.allclose(out.double(), (qu.double() - 131) * 0.05, rtol=1e-6, atol=1e-7), (
        "asymmetric: dequantize must return (q - zero_point) * scale — upcast q "
        "first, uint8 - 131 wraps around")
    for axis, C in ((0, 5), (1, 7), (-1, 7)):
        sc = torch.rand(C, generator=g) * 0.1 + 0.01
        out = ns.dequantize(q8, sc, axis=axis)
        shape = [1, 1]
        shape[axis] = C
        want = q8.double() * sc.double().reshape(shape)
        assert torch.allclose(out.double(), want, rtol=1e-6, atol=0), (
            f"per-channel dequantize with axis={axis}: the ({C},) scale must "
            f"broadcast along dim {axis}")


def _linear_inputs(M, K, N, seed):
    g = torch.Generator().manual_seed(seed)
    x_q = torch.randint(0, 256, (M, K), generator=g).to(torch.uint8)
    w_q = torch.randint(-127, 128, (N, K), generator=g).to(torch.int8)
    return x_q, w_q, g


def check_int8_linear_matches_dequantized_matmul(ns):
    M, K, N = 6, 40, 9
    x_q, w_q, g = _linear_inputs(M, K, N, seed=7)
    cases = [
        ("per-tensor w_scale", 0.02, 121, torch.tensor(0.004)),
        ("per-channel w_scale", torch.tensor(0.015), 7, torch.rand(N, generator=g) * 0.01 + 1e-3),
        ("zero_point 0", 0.5, 0, torch.rand(N, generator=g) * 0.01 + 1e-3),
    ]
    for label, xs, xz, ws in cases:
        y = ns.int8_linear(x_q, xs, xz, w_q, ws)
        assert torch.is_tensor(y) and tuple(y.shape) == (M, N), (
            f"{label}: int8_linear should return ({M}, {N}) for x_q {(M, K)} and "
            f"w_q {(N, K)}, got {tuple(y.shape) if torch.is_tensor(y) else type(y).__name__}")
        assert y.is_floating_point(), f"{label}: the rescaled output should be floating point"
        xd = (x_q.double() - xz) * float(xs)
        wd = w_q.double() * _bcast(ws, 2, 0 if _f64(ws).numel() > 1 else None)
        want = xd @ wd.T
        diff = (y.double() - want).abs().max().item()
        tol = 1e-5 * want.abs().max().item()
        assert diff <= tol, (
            f"{label}: int8_linear differs from dequant(x) @ dequant(w).T by {diff:.3e} "
            f"(tolerance {tol:.1e}). Subtract the zero point after upcasting, "
            f"accumulate in a wide integer type, then multiply by x_scale * w_scale "
            f"(per-channel w_scale broadcasts over the output features)")


def check_int8_linear_accumulates_exactly(ns):
    """Unit scales expose the accumulator: the answer must be the exact integer."""
    g = torch.Generator().manual_seed(8)
    M, K, N = 16, 4096, 32
    x_q = torch.randint(200, 256, (M, K), generator=g).to(torch.uint8)
    w_q = torch.randint(90, 128, (N, K), generator=g).to(torch.int8)
    zp = 3
    exact = (x_q.long() - zp) @ w_q.long().T          # up to ~1.3e8 > 2^24
    y = ns.int8_linear(x_q, 1.0, zp, w_q, torch.ones(N))
    assert torch.is_tensor(y) and tuple(y.shape) == (M, N), \
        f"int8_linear should return ({M}, {N}), got {tuple(y.shape) if torch.is_tensor(y) else y}"
    want = exact.to(y.dtype)                          # the exact integer, rounded once
    n_bad = int((y != want).sum())
    worst = (y.double() - exact.double()).abs().max().item()
    assert worst <= 1e-5 * exact.abs().max().item(), (
        f"with x_scale = w_scale = 1 the output should be the integer dot products "
        f"sum_k (x_q - {zp}) * w_q, but it is off by up to {worst:.3g} — check the "
        f"zero-point subtraction and that nothing wraps around in 8-bit arithmetic")
    assert n_bad == 0, (
        f"with x_scale = w_scale = 1 the output should be the exact integer dot "
        f"product (rounded once to {y.dtype}); {n_bad}/{M * N} entries are off by up "
        f"to {worst:.0f}. These sums exceed 2^24, where float32 can no longer "
        f"represent every integer, so accumulating in float32 drifts — accumulate in "
        f"int32 (or int64/float64), then rescale")


def check_all_zero_input_has_no_nan(ns):
    z = torch.zeros(8, 5)
    q, s = _sym(ns, z)
    _check_sym_result(z, q, s, 8, None, "symmetric all-zero tensor")
    qa, sa, za = _asym(ns, z)
    _check_asym(z, qa, sa, za, 8, "asymmetric all-zero tensor")
    for label, deq in (("symmetric", ns.dequantize(q, s)),
                       ("asymmetric", ns.dequantize(qa, sa, za))):
        assert torch.isfinite(deq).all() and (deq == 0).all(), (
            f"{label}: an all-zero tensor must dequantize back to exact zeros without "
            f"NaN/Inf — guard max|x| == 0 (scale = 1.0 works)")


CHECKS = [
    check_symmetric_per_tensor,
    check_symmetric_per_channel,
    check_per_channel_beats_per_tensor_with_an_outlier_channel,
    check_asymmetric_quantization,
    check_zero_round_trips_exactly,
    check_asymmetric_wins_on_relu_activations,
    check_dequantize,
    check_int8_linear_matches_dequantized_matmul,
    check_int8_linear_accumulates_exactly,
    check_all_zero_input_has_no_nan,
]
