"""unet — a small time-conditioned U-Net (the DDPM-style diffusion denoiser).

Checks, all on tiny configurations so they run in a few seconds on CPU:
- the sinusoidal embedding has the documented layout (first half sin, second
  half cos, f_i = exp(-ln(10000) * i / (half - 1))), with a diagnosis when it
  matches a neighbouring convention instead;
- ResBlock: zeroing its 3x3 convs leaves exactly skip(x) (x itself when the
  channel counts match), which proves the residual add and that nothing (norm,
  activation, rescale) is applied after it; the time embedding is per sample;
- UNet: output shape equals input shape across resolutions and depths; time
  conditioning changes the output; samples in a batch are independent in train
  mode (BatchNorm fails this, GroupNorm passes; the global seed is reset before each
  forward so dropout, an allowed extra, draws identical masks); every parameter gets a
  gradient; encoder features are concatenated (not added) along channels at
  every resolution above the bottleneck, seen by a TorchFunctionMode that
  watches torch.cat; and it can overfit one tiny batch.

Before the gradient and conditioning checks the parameters are jittered with
small noise, so an ADM-style zero-initialised output conv (a valid choice that
makes the output exactly 0 at init) does not cause a false failure.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

ENTRIES = ["SinusoidalTimeEmbedding", "ResBlock", "UNet"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Build bottom-up: the embedding is a fixed function of t (no parameters), "
    "ResBlock is norm-act-conv twice with the projected time embedding added in "
    "between, and the UNet only wires ResBlocks, downsamples and upsamples together.",
    "In ResBlock the time projection gives (N, out_ch); index it with [:, :, None, None] "
    "so it broadcasts over H and W. When in_ch != out_ch the residual needs a 1x1 conv "
    "so skip(x) + h has matching channels. Return the sum as is, nothing after it.",
    "In UNet.forward keep a list: after each encoder level's ResBlock append h, then "
    "downsample (except at the last level). In the decoder walk the levels in reverse: "
    "torch.cat([h, skips.pop()], dim=1), a ResBlock whose in_ch is the sum of both "
    "widths, then upsample (except at the top level).",
]


# ----------------------------------------------------------------------- helpers
def _unet(ns, in_channels=3, base_channels=16, channel_mults=(1, 2), time_dim=32, seed=0):
    torch.manual_seed(seed)
    try:
        return ns.UNet(in_channels=in_channels, base_channels=base_channels,
                       channel_mults=channel_mults, time_dim=time_dim)
    except TypeError as e:
        raise AssertionError(
            "could not build UNet(in_channels=..., base_channels=..., channel_mults=..., "
            f"time_dim=...): {e}")


def _jitter(m, seed=1, std=0.02):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn(p.shape, generator=g) * std)
    return m


def _ref_embedding(t, dim, half_minus_one=True, cos_first=False, interleave=False):
    half = dim // 2
    i = torch.arange(half, dtype=torch.float64)
    f = torch.exp(-math.log(10000.0) * i / ((half - 1) if half_minus_one else half))
    a = t.double()[:, None] * f[None, :]
    s, c = a.sin(), a.cos()
    if interleave:
        return torch.stack([s, c], dim=-1).reshape(len(t), dim)
    return torch.cat([c, s] if cos_first else [s, c], dim=-1)


# ---------------------------------------------------------------- embedding
def check_time_embedding_layout(ns):
    dim = 16
    emb = ns.SinusoidalTimeEmbedding(dim)
    t = torch.tensor([0, 1, 3, 250, 999])
    got = emb(t)
    assert tuple(got.shape) == (5, dim), \
        f"SinusoidalTimeEmbedding({dim}) on t of shape (5,) should give (5, {dim}), got {tuple(got.shape)}"
    got = got.detach().double()
    if torch.allclose(got, _ref_embedding(t, dim), atol=1e-3):
        return
    for kw, what in [(dict(half_minus_one=False), "frequencies exp(-ln(10000) i / half) — "
                      "this problem uses i / (half - 1), so the lowest frequency is exactly 1/10000"),
                     (dict(cos_first=True), "cos in the first half and sin in the second — "
                      "the documented layout is [sin | cos]"),
                     (dict(interleave=True), "sin/cos interleaved per frequency — "
                      "the documented layout is concatenated halves [sin | cos]")]:
        if torch.allclose(got, _ref_embedding(t, dim, **kw), atol=1e-3):
            raise AssertionError(f"the embedding uses {what}")
    raise AssertionError(
        "embedding values do not match [sin(t f_i) | cos(t f_i)] with "
        "f_i = exp(-ln(10000) * i / (half - 1)), i = 0..half-1 "
        f"(at t = 0 expect {dim // 2} zeros then {dim // 2} ones; got {got[0].tolist()})")


def check_time_embedding_long_float_and_distinct(ns):
    emb = ns.SinusoidalTimeEmbedding(32)
    tl = torch.tensor([0, 5, 17, 400, 401, 998])
    a, b = emb(tl), emb(tl.float())
    assert torch.allclose(a.float(), b.float(), atol=1e-5), \
        "integer and float timesteps with the same values must embed identically"
    assert tuple(emb(torch.tensor([7])).shape) == (1, 32), "a batch of one timestep should give (1, 32)"
    d = torch.cdist(a.float(), a.float()) + torch.eye(len(tl)) * 1e9
    assert d.min().item() > 1e-2, \
        "two different timesteps got (almost) the same embedding — the model could not tell them apart"
    assert not any(p.requires_grad for p in emb.parameters()), \
        "the sinusoidal embedding is a fixed function — it should have no trainable parameters"


# ---------------------------------------------------------------- ResBlock
def _zero_3x3_convs(block):
    n = 0
    with torch.no_grad():
        for m in block.modules():
            if isinstance(m, nn.Conv2d) and tuple(m.kernel_size) != (1, 1):
                m.weight.zero_()
                if m.bias is not None:
                    m.bias.zero_()
                n += 1
    return n


def check_resblock_residual_path(ns):
    torch.manual_seed(0)
    x = torch.randn(2, 16, 6, 5)
    temb = torch.randn(2, 24)
    blk = ns.ResBlock(16, 16, 24)
    out = blk(x, temb)
    assert tuple(out.shape) == (2, 16, 6, 5), \
        f"ResBlock(16, 16, 24) should keep the shape (2, 16, 6, 5), got {tuple(out.shape)}"
    assert _zero_3x3_convs(blk) >= 2, "ResBlock should contain two 3x3 nn.Conv2d layers"
    out = blk(x, temb)
    diff = (out - x).abs().max().item()
    assert diff <= 1e-6, (
        f"with both 3x3 convs zeroed, ResBlock(16, 16) must return exactly x (max |diff| "
        f"{diff:.2e}). The residual is out = x + h with nothing after the add — no "
        f"activation, norm or 1/sqrt(2) rescale — and the time term goes in *between* the convs")

    blk = ns.ResBlock(16, 32, 24)
    out = blk(x, temb)
    assert tuple(out.shape) == (2, 32, 6, 5), \
        f"ResBlock(16, 32, 24) should give (2, 32, 6, 5), got {tuple(out.shape)}"
    skips = [m for m in blk.modules() if isinstance(m, nn.Conv2d) and tuple(m.kernel_size) == (1, 1)]
    assert len(skips) == 1, \
        "when in_ch != out_ch the residual needs exactly one 1x1 nn.Conv2d to match channels"
    _zero_3x3_convs(blk)
    out = blk(x, temb)
    exp = F.conv2d(x, skips[0].weight, skips[0].bias)
    diff = (out - exp).abs().max().item()
    assert diff <= 1e-5, (
        f"with its 3x3 convs zeroed, ResBlock(16, 32) must return exactly the 1x1-conv "
        f"projection of x (max |diff| {diff:.2e})")


def check_resblock_time_embedding_is_per_sample(ns):
    torch.manual_seed(0)
    blk = _jitter(ns.ResBlock(16, 16, 8))
    x = torch.randn(2, 16, 4, 4)
    ta = torch.randn(2, 8)
    tb = ta.clone()
    tb[1] = torch.randn(8) * 3
    # Same seed before each call: any dropout (an allowed extra) then draws the same mask
    # both times, so only a real cross-sample interaction can change sample 0.
    with torch.no_grad():
        torch.manual_seed(1)
        a = blk(x, ta)
        torch.manual_seed(1)
        b = blk(x, tb)
    assert (a[1] - b[1]).abs().max().item() > 1e-4, \
        "changing a sample's time embedding did not change its output — is the time projection added?"
    assert (a[0] - b[0]).abs().max().item() <= 1e-6, (
        "changing sample 1's time embedding changed sample 0's output — add the (N, out_ch) "
        "projection as [:, :, None, None], one vector per sample, and use no layer that "
        "mixes samples (BatchNorm does; use GroupNorm)")


# ---------------------------------------------------------------- UNet
def check_unet_output_shape_at_several_resolutions(ns):
    cases = [(dict(in_channels=3, base_channels=16, channel_mults=(1, 2, 2), time_dim=32),
              [(2, 3, 16, 16), (1, 3, 8, 12), (2, 3, 32, 24)]),
             (dict(in_channels=1, base_channels=8, channel_mults=(1, 2), time_dim=16),
              [(2, 1, 10, 6)]),
             (dict(in_channels=3, base_channels=8, channel_mults=(1, 2, 2, 2), time_dim=16),
              [(1, 3, 16, 8)])]
    for kw, shapes in cases:
        m = _unet(ns, **kw)
        for shape in shapes:
            t = torch.randint(0, 1000, (shape[0],))
            with torch.no_grad():
                out = m(torch.randn(*shape), t)
            assert tuple(out.shape) == shape, (
                f"UNet(channel_mults={kw['channel_mults']}, in_channels={kw['in_channels']}) "
                f"mapped {shape} to {tuple(out.shape)}; a denoiser's output must match its input")


def check_unet_time_conditioning_changes_output(ns):
    m = _jitter(_unet(ns, channel_mults=(1, 2, 2))).eval()
    x = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        a = m(x, torch.tensor([3, 40]))
        b = m(x, torch.tensor([700, 950]))
        c = m(x, torch.tensor([3.0, 40.0]))
    scale = a.abs().max().item() + 1e-12
    assert (a - b).abs().max().item() > 1e-3 * scale, (
        "t = (3, 40) and t = (700, 950) give the same output — the time embedding never "
        "reaches the network")
    assert torch.allclose(a, c, atol=1e-5), "integer and float timesteps should give the same output"


def check_unet_samples_are_independent(ns):
    m = _jitter(_unet(ns))
    m.train()
    torch.manual_seed(3)
    a, b, c = torch.randn(3, 1, 3, 8, 8).unbind(0)
    # Re-seed before each forward so dropout (allowed) uses identical masks in both calls.
    torch.manual_seed(4)
    out1 = m(torch.cat([a, b]), torch.tensor([10, 500]))
    torch.manual_seed(4)
    out2 = m(torch.cat([a, c]), torch.tensor([10, 90]))
    diff = (out1[0] - out2[0]).abs().max().item()
    assert diff <= 1e-5, (
        f"in train mode, sample 0's output changed when sample 1 changed (max |diff| {diff:.2e}) "
        f"— something mixes samples. BatchNorm does (batch statistics; diffusion U-Nets use "
        f"GroupNorm, which normalises each sample on its own), and so does any reduction of "
        f"the time embedding over the batch dimension")


def check_unet_gradients_reach_every_parameter(ns):
    m = _jitter(_unet(ns, channel_mults=(1, 2, 2)))
    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    out = m(x, torch.tensor([4, 321]))
    (out * torch.randn_like(out)).sum().backward()
    dead = [n for n, p in m.named_parameters()
            if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
    assert not dead, (
        f"no gradient reached: {', '.join(dead[:4])}{'...' if len(dead) > 4 else ''} "
        f"— defined but not on the forward path")
    assert x.grad is not None and x.grad.abs().sum() > 0, "no gradient reached the input image"


class _CatWatcher(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.sizes = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if getattr(func, "__name__", "") in ("cat", "concat", "concatenate") and args:
            dim = kwargs.get("dim", kwargs.get("axis", args[1] if len(args) > 1 else 0))
            ts = list(args[0])
            if (len(ts) >= 2 and all(torch.is_tensor(t) and t.dim() == 4 for t in ts)
                    and dim in (1, -3)):
                self.sizes.append(tuple(out.shape[-2:]))
        return out


def check_unet_concatenates_skip_features(ns):
    mults = (1, 2, 2)
    m = _unet(ns, channel_mults=mults)
    H, W = 16, 24
    with _CatWatcher() as w, torch.no_grad():
        m(torch.randn(1, 3, H, W), torch.tensor([50]))
    want = [(H >> i, W >> i) for i in range(len(mults) - 1)]
    missing = [s for s in want if s not in w.sizes]
    assert not missing, (
        f"for a {H}x{W} input with channel_mults={mults}, expected the decoder to "
        f"torch.cat encoder features along channels at every resolution above the "
        f"bottleneck {want}; saw channel concatenations at {sorted(set(w.sizes)) or 'none'}. "
        f"A U-Net's skips are concatenated, not added — the decoder gets the encoder's "
        f"high-resolution detail as extra channels")


def check_unet_can_overfit_tiny_batch(ns):
    m = _unet(ns, base_channels=16, channel_mults=(1, 2), time_dim=32)
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(2, 3, 8, 8, generator=g)
    eps = torch.randn(2, 3, 8, 8, generator=g)
    t = torch.tensor([30, 600])
    abar = torch.tensor([0.9, 0.3])[:, None, None, None]
    xt = abar.sqrt() * x0 + (1 - abar).sqrt() * eps
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    first = last = None
    for _ in range(25):
        opt.zero_grad()
        loss = F.mse_loss(m(xt, t), eps)
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
        last = loss.item()
    assert last < 0.5 * first, (
        f"25 Adam steps on one fixed batch of 2 only took the noise-prediction loss from "
        f"{first:.3f} to {last:.3f} — expected it to at least halve. Check that gradients "
        f"flow and nothing is detached")


CHECKS = [
    check_time_embedding_layout,
    check_time_embedding_long_float_and_distinct,
    check_resblock_residual_path,
    check_resblock_time_embedding_is_per_sample,
    check_unet_output_shape_at_several_resolutions,
    check_unet_time_conditioning_changes_output,
    check_unet_samples_are_independent,
    check_unet_gradients_reach_every_parameter,
    check_unet_concatenates_skip_features,
    check_unet_can_overfit_tiny_batch,
]
