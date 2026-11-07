"""augmentation — CIFAR-10 train/test pipelines built with torchvision.transforms.

The pipelines are applied to synthetic 32x32 RGB PIL images (random pixels, so
every flip/shift of one image is distinguishable), so nothing is downloaded. The
spec itself needs only PIL — it never imports torchvision, and a hand-written or
transforms.v2 pipeline passes the same way.

  * test_transform: a float tensor (3, 32, 32) equal to (pixels / 255 - mean) / std
    with CIFAR-10 statistics, identical on every call. The mean and std are read
    off the output (it is affine in the pixels, per channel) and accepted within
    0.002 of mean (0.4914, 0.4822, 0.4465) and of either std in circulation,
    (0.2470, 0.2435, 0.2616) or (0.2023, 0.1994, 0.2010) — so rounded or
    more precise constants pass. Missing Normalize, ImageNet or CIFAR-100
    statistics, (0.5, 0.5, 0.5) and swapped mean/std are named.
  * train_transform: every draw must be exactly one of the 162 images
    "optionally mirrored left-right, then shifted by (dy, dx) in [-4, 4]^2 with
    black zero padding" — computed here from the pixels — then normalised with
    the statistics test_transform uses. Over 300 seeded draws, about half must be
    mirrored and every shift from -4 to 4 must occur in both directions. Flip-then-crop and
    crop-then-flip give the same set, so their order is not checked; padding in
    normalised space (Normalize before the crop) or a vertical flip is named.
"""
import random

import torch
from torchleet.runner import Skip

ENTRIES = ["train_transform", "test_transform"]
DEVICE = "cpu"
EXTRAS = ["PIL"]
HINTS = [
    "Both pipelines are a transforms.Compose list. The test pipeline only converts "
    "and normalises; the training pipeline adds the random augmentations in front.",
    "PIL-space operations (RandomHorizontalFlip, RandomCrop) go before ToTensor, "
    "and Normalize goes last because it needs a tensor. The pad value of the crop "
    "is then a black pixel, as in the original images' borders.",
    "Training: RandomHorizontalFlip(p=0.5), RandomCrop(32, padding=4), ToTensor(), "
    "Normalize(CIFAR10_MEAN, CIFAR10_STD). Test: ToTensor() and the same Normalize "
    "only — no random transform, so evaluation is deterministic.",
]

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
# Both CIFAR-10 std vectors in wide use are accepted, (0.2470, 0.2435, 0.2616) and
# (0.2023, 0.1994, 0.2010); rounding or extra digits in either are fine (STATS_TOL).
ACCEPTED_STD = [STD, torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)]
STATS_TOL = 2e-3
WRONG_STATS = {
    "the ImageNet statistics": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "the CIFAR-100 statistics": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "Normalize((0.5,)*3, (0.5,)*3)": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "no Normalize at all": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
}
PAD = 4
DRAWS = 300


def _image(seed):
    try:
        from PIL import Image
    except ImportError:
        raise Skip("needs Pillow — pip install pillow") from None
    g = torch.Generator().manual_seed(seed)
    px = torch.randint(0, 256, (32, 32, 3), generator=g, dtype=torch.uint8)
    img = Image.frombytes("RGB", (32, 32), bytes(px.flatten().tolist()))
    return img, px.permute(2, 0, 1).float() / 255.0        # (3, 32, 32) in [0, 1]


def _as_tensor(out, which):
    assert torch.is_tensor(out), (
        f"{which} returned a {type(out).__name__}, expected a tensor — end the "
        "pipeline with ToTensor() and Normalize(...)")
    assert out.dtype == torch.float32, f"{which} returned dtype {out.dtype}, expected float32"
    assert tuple(out.shape) == (3, 32, 32), (
        f"{which} returned shape {tuple(out.shape)}, expected (3, 32, 32)"
        + (" — the crop must keep the 32x32 size (RandomCrop(32, padding=4))"
           if out.dim() == 3 and out.shape[0] == 3 else ""))
    return out


def _fit_stats(out, x):
    """Per-channel (mean, std) with out = (x - mean) / std, by least squares on the
    pixels; and the largest residual (0 when out is such an affine map of x)."""
    xs, ys = x.reshape(3, -1).double(), out.reshape(3, -1).double()
    xc, yc = xs - xs.mean(1, keepdim=True), ys - ys.mean(1, keepdim=True)
    a = (xc * yc).sum(1) / (xc * xc).sum(1)                   # 1 / std
    b = ys.mean(1) - a * xs.mean(1)                           # -mean / std
    resid = float((ys - (a[:, None] * xs + b[:, None])).abs().max())
    return (-b / a).float().view(3, 1, 1), (1 / a).float().view(3, 1, 1), resid


def _fmt(t):
    return "(" + ", ".join(f"{v:.4f}" for v in t.flatten().tolist()) + ")"


def _cifar_stats(mean, std):
    return (torch.allclose(mean, MEAN, atol=STATS_TOL, rtol=0)
            and any(torch.allclose(std, s, atol=STATS_TOL, rtol=0) for s in ACCEPTED_STD))


def _test_stats(ns):
    """(mean, std, True) that test_transform normalises with, or the canonical
    (MEAN, STD, False) if it is not a CIFAR-10 normalisation (its own check says so)."""
    try:
        img, x = _image(0)
        out = ns.test_transform(img)
        if torch.is_tensor(out) and tuple(out.shape) == (3, 32, 32):
            mean, std, resid = _fit_stats(out.detach().float(), x)
            if resid < 1e-4 and _cifar_stats(mean, std):
                return mean, std, True
    except Exception:
        pass
    return MEAN, STD, False


def check_test_transform_normalises_with_cifar_stats(ns):
    for seed in (0, 1, 2):
        img, x = _image(seed)
        out = _as_tensor(ns.test_transform(img), "test_transform")
        mean, std, resid = _fit_stats(out, x)
        if resid > 1e-4:
            raise AssertionError(
                "test_transform(img) is not (pixels / 255 - mean) / std per channel "
                f"(off by up to {resid:.3g}) — the test pipeline is ToTensor() then "
                "Normalize, with no augmentation")
        if _cifar_stats(mean, std):
            continue
        if torch.allclose(mean, STD, atol=STATS_TOL) and torch.allclose(std, MEAN, atol=STATS_TOL):
            raise AssertionError("test_transform: mean and std are swapped")
        for name, (m, sd) in WRONG_STATS.items():
            if (torch.allclose(mean, torch.tensor(m).view(3, 1, 1), atol=STATS_TOL)
                    and torch.allclose(std, torch.tensor(sd).view(3, 1, 1), atol=STATS_TOL)):
                raise AssertionError(
                    f"test_transform uses {name} — use the CIFAR-10 mean "
                    f"{_fmt(MEAN)} and std {_fmt(STD)}")
        raise AssertionError(
            f"test_transform normalises with mean {_fmt(mean)} and std {_fmt(std)}; the "
            f"CIFAR-10 statistics are mean {_fmt(MEAN)} and std {_fmt(STD)} "
            f"(or the also common {_fmt(ACCEPTED_STD[1])})")


def check_test_transform_is_deterministic(ns):
    img, _ = _image(3)
    torch.manual_seed(0)
    random.seed(0)
    outs = [_as_tensor(ns.test_transform(img), "test_transform") for _ in range(20)]
    assert all(torch.equal(outs[0], o) for o in outs[1:]), (
        "test_transform gave different outputs for the same image — remove the random "
        "transforms from the test pipeline; evaluation must be deterministic")


def _candidates(x, mean=MEAN, std=STD, fill_normalised=False, vertical=False):
    """All (flip, shift) variants of x, normalised: (2 * 9 * 9, 3*32*32) and their labels."""
    base = (x - mean) / std
    fill = torch.zeros(3, 1, 1) if fill_normalised else (0.0 - mean) / std
    out, labels = [], []
    for flipped in (False, True):
        b = base.flip(-2 if vertical else -1) if flipped else base
        padded = fill.expand(3, 32 + 2 * PAD, 32 + 2 * PAD).clone()
        padded[:, PAD:PAD + 32, PAD:PAD + 32] = b
        for i in range(2 * PAD + 1):
            for j in range(2 * PAD + 1):
                out.append(padded[:, i:i + 32, j:j + 32].reshape(-1))
                labels.append((flipped, i - PAD, j - PAD))
    return torch.stack(out), labels


def _match(draws, cands, tol=1e-3):
    # (n_draws, 162) L2 distances; the exact mode, since the matmul shortcut
    # ||a||^2 + ||b||^2 - 2ab cancels badly at these norms
    d = torch.cdist(draws.double(), cands.double(),
                    compute_mode="donot_use_mm_for_euclid_dist")
    best, idx = d.min(dim=1)
    return best < tol, idx


def check_train_transform_is_flip_plus_padded_crop(ns):
    img, x = _image(4)
    torch.manual_seed(0)
    random.seed(0)
    draws = torch.stack([_as_tensor(ns.train_transform(img), "train_transform").reshape(-1)
                         for _ in range(DRAWS)])
    mean, std, test_ok = _test_stats(ns)
    cands, labels = _candidates(x, mean, std)
    ok, idx = _match(draws, cands)
    if not ok.all():
        for why, kw in (
            ("the padding is not black: flip and crop must act on the PIL image, before "
             "ToTensor and Normalize (padding after Normalize pads with 0 = the mean colour)",
             dict(fill_normalised=True)),
            ("the image is flipped upside down — use a horizontal (left-right) flip",
             dict(vertical=True)),
        ):
            if _match(draws, _candidates(x, mean, std, **kw)[0])[0].all():
                raise AssertionError(f"train_transform: {why}")
        others = [(MEAN, s) for s in ACCEPTED_STD] + [
            (torch.tensor(m).view(3, 1, 1), torch.tensor(sd).view(3, 1, 1))
            for m, sd in WRONG_STATS.values()]
        for m, sd in others:
            if _match(draws, _candidates(x, m, sd)[0])[0].all():
                raise AssertionError(
                    f"train_transform normalises with mean {_fmt(m)} and std {_fmt(sd)}"
                    + (f", test_transform with mean {_fmt(mean)} and std {_fmt(std)} — "
                       "both pipelines must use the same CIFAR-10 statistics" if test_ok
                       else f" — use the CIFAR-10 mean {_fmt(MEAN)} and std {_fmt(STD)}"))
        bad = int((~ok).sum())
        raise AssertionError(
            f"{bad} of {DRAWS} augmented images are not the CIFAR-normalised original, "
            "optionally mirrored left-right and shifted by at most 4 pixels with black "
            "padding — train_transform should be RandomHorizontalFlip, "
            "RandomCrop(32, padding=4), ToTensor, Normalize and nothing else")

    seen = [labels[i] for i in idx.tolist()]
    flips = sum(f for f, _, _ in seen) / DRAWS
    assert 0.35 <= flips <= 0.65, (
        f"{flips:.0%} of {DRAWS} training images were mirrored, expected about 50% "
        "(RandomHorizontalFlip(p=0.5))")
    for axis, k in (("vertical", 1), ("horizontal", 2)):
        shifts = sorted({s[k] for s in seen})
        assert shifts == list(range(-PAD, PAD + 1)), (
            f"over {DRAWS} draws the {axis} shifts were {shifts}; a 32x32 crop of the "
            f"image padded by 4 should shift by every amount from -4 to 4 "
            "(RandomCrop(32, padding=4))")


CHECKS = [
    check_test_transform_normalises_with_cifar_stats,
    check_test_transform_is_deterministic,
    check_train_transform_is_flip_plus_padded_crop,
]
