"""contrastive-loss-clip — InfoNCE over a cosine-similarity matrix, plus a CLIP model.

Every loss check is a property of InfoNCE itself rather than a comparison against
a stored implementation:

  * on orthonormal features the symmetric loss has a closed form,
    log(1 + (B-1)*exp(-temperature)), which pins down normalization, the diagonal
    labels and the symmetric average at once;
  * scaling either feature set must not move the loss (that is what L2
    normalization buys you);
  * swapping the two arguments must not move it either (that is the "symmetric"
    in symmetric cross-entropy);
  * on random features the value must equal symmetric cross-entropy over the
    scaled cosine-similarity matrix, recomputed here from torch built-ins.

The scalar argument is a logit scale (CLIP's exp(logit_scale) = 1/tau). Where a
value other than 1.0 is asserted, multiplying the similarities by it or dividing
by it — the two conventions in the wild — are both accepted.
"""
import math

import torch
import torch.nn.functional as F

ENTRIES = ["info_nce_loss", "SimpleCLIP"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "L2-normalize both feature sets first (F.normalize(x, dim=-1)); the cosine "
    "similarity matrix is then just image_features @ text_features.T.",
    "The positives sit on the diagonal, so the cross-entropy targets are "
    "torch.arange(batch_size) — image i must match text i.",
    "Symmetric means both directions: (cross_entropy(logits, labels) + "
    "cross_entropy(logits.T, labels)) / 2, with logits = cosine similarity * "
    "exp(log_temperature). That multiplier is CLIP's logit scale 1/tau (initialized "
    "to 1/0.07, clamped at 100); keep it learnable in log space.",
]

B_SIZE, IMG_DIM, TXT_DIM, EMB_DIM = 16, 12, 10, 8


def _orthonormal_pairs(n, dim, seed=0):
    """n rows of an identity basis, each row randomly rescaled.

    Cosine similarity is then exactly I: 1 on the diagonal, 0 off it, whatever
    the row scales are.
    """
    torch.manual_seed(seed)
    base = torch.eye(dim)[:n]
    img = base * (torch.rand(n, 1) * 3 + 0.5)
    txt = base * (torch.rand(n, 1) * 3 + 0.5)
    return img, txt


def _paired_data(seed=0):
    """Synthetic image/text features that share a latent factor, so they are learnable."""
    torch.manual_seed(seed)
    latent = torch.randn(B_SIZE, EMB_DIM)
    images = torch.randn(B_SIZE, IMG_DIM) * 0.2 + latent @ torch.randn(EMB_DIM, IMG_DIM)
    texts = torch.randn(B_SIZE, TXT_DIM) * 0.2 + latent @ torch.randn(EMB_DIM, TXT_DIM)
    return images, texts


def _loss_value(ns, img, txt, temp):
    out = ns.info_nce_loss(img, txt, torch.tensor(float(temp)))
    assert torch.is_tensor(out), \
        f"info_nce_loss should return a tensor, got {type(out).__name__}"
    assert out.dim() == 0 or out.numel() == 1, \
        f"info_nce_loss should return a scalar, got shape {tuple(out.shape)}"
    return out.reshape(()) if out.dim() else out


def check_closed_form_on_orthonormal_features(ns):
    """With cosine similarity = I the symmetric InfoNCE loss is known exactly."""
    for n in (4, 8):
        img, txt = _orthonormal_pairs(n, max(n, EMB_DIM), seed=n)
        got = _loss_value(ns, img, txt, 1.0).item()
        expected = math.log(1.0 + (n - 1) * math.exp(-1.0))
        assert abs(got - expected) < 1e-4, (
            f"batch of {n} orthonormal pairs at temperature=1: the diagonal "
            f"similarities are 1 and the off-diagonals 0, so the loss must be "
            f"log(1 + {n - 1}*e^-1) = {expected:.4f}, got {got:.4f}. Check that you "
            f"normalize the features, label the diagonal, and average both directions.")


def check_invariant_to_feature_scale(ns):
    """L2 normalization means the magnitude of each embedding cannot matter.

    Every row gets its own scale: a single global factor would also be absorbed
    by normalizing along the wrong axis (over the batch instead of the features).
    """
    img, txt = _paired_data(1)
    img, txt = img[:, :EMB_DIM], txt[:, :EMB_DIM]
    g = torch.Generator().manual_seed(1)
    row_img = torch.rand(B_SIZE, 1, generator=g) * 20 + 0.05
    row_txt = torch.rand(B_SIZE, 1, generator=g) * 20 + 0.05
    base = _loss_value(ns, img, txt, 1.0).item()
    scaled = _loss_value(ns, img * row_img, txt * row_txt, 1.0).item()
    assert abs(base - scaled) < 1e-4, (
        f"rescaling each feature vector changed the loss ({base:.4f} -> {scaled:.4f}); "
        f"L2-normalize every embedding along the feature dimension (dim=-1) before "
        f"the similarity matrix")


def check_symmetric_in_its_two_inputs(ns):
    """i2t + t2i averaged: swapping images and texts must give the same number."""
    img, txt = _paired_data(2)
    img, txt = img[:, :EMB_DIM], txt[:, :EMB_DIM]
    a = _loss_value(ns, img, txt, 1.0).item()
    b = _loss_value(ns, txt, img, 1.0).item()
    assert abs(a - b) < 1e-5, (
        f"loss(image, text)={a:.4f} but loss(text, image)={b:.4f} — the loss must "
        f"average the image->text and text->image cross-entropies")


def check_labels_follow_the_diagonal(ns):
    """Permuting both sets the same way keeps pair i matched to pair i."""
    img, txt = _paired_data(3)
    img, txt = img[:, :EMB_DIM], txt[:, :EMB_DIM]
    perm = torch.randperm(B_SIZE, generator=torch.Generator().manual_seed(0))
    base = _loss_value(ns, img, txt, 1.0).item()
    permuted = _loss_value(ns, img[perm], txt[perm], 1.0).item()
    assert abs(base - permuted) < 1e-4, (
        f"permuting image and text rows identically changed the loss "
        f"({base:.4f} -> {permuted:.4f}); the targets should be "
        f"torch.arange(batch_size), not a fixed index")
    shuffled = _loss_value(ns, img, txt[perm], 1.0).item()
    assert shuffled > base + 1e-3, (
        f"shuffling only the texts (breaking every pair) did not raise the loss "
        f"({base:.4f} -> {shuffled:.4f}) — is the similarity matrix being used at all?")


def _oracle(img, txt, scale):
    """Symmetric InfoNCE from torch built-ins, logits = cos-sim * scale."""
    logits = F.normalize(img, dim=-1) @ F.normalize(txt, dim=-1).T * scale
    labels = torch.arange(logits.size(0))
    return ((F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2).item()


def check_temperature_is_used(ns):
    img, txt = _orthonormal_pairs(8, EMB_DIM, seed=5)
    cold = _loss_value(ns, img, txt, 1.0).item()
    hot = _loss_value(ns, img, txt, 8.0).item()
    assert abs(cold - hot) > 1e-3, \
        "changing the temperature did not change the loss — it is being ignored"
    assert min(cold, hot) >= 0.0, \
        f"cross-entropy cannot be negative, got {min(cold, hot):.4f}"


def check_matches_symmetric_cross_entropy(ns):
    """Random (non-orthogonal) features, several scales, against torch built-ins."""
    img, txt = _paired_data(9)
    img, txt = img[:, :EMB_DIM], txt[:, :EMB_DIM]
    for temp in (1.0, 2.5, 14.0):
        got = _loss_value(ns, img, txt, temp).item()
        times, divided = _oracle(img, txt, temp), _oracle(img, txt, 1.0 / temp)
        assert min(abs(got - times), abs(got - divided)) < 1e-4, (
            f"temperature={temp}: got {got:.4f}, but the symmetric cross-entropy over "
            f"the cosine-similarity matrix is {times:.4f} when the similarities are "
            f"multiplied by it (CLIP's logit scale) or {divided:.4f} when divided by "
            f"it. Normalize each embedding (dim=-1), scale the logits exactly once, "
            f"and average the two directions.")


def check_loss_backpropagates(ns):
    img, txt = _paired_data(4)
    img = img[:, :EMB_DIM].clone().requires_grad_(True)
    txt = txt[:, :EMB_DIM].clone().requires_grad_(True)
    temp = torch.tensor(4.0, requires_grad=True)
    loss = ns.info_nce_loss(img, txt, temp)
    loss.backward()
    for name, t in (("image_features", img), ("text_features", txt), ("temperature", temp)):
        assert t.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(t.grad).all(), f"gradient w.r.t. {name} contains NaN/Inf"
    assert img.grad.abs().sum() > 0, "gradient w.r.t. image_features is all zeros"


def _clip_forward(model, images, texts):
    out = model(images, texts)
    assert isinstance(out, (tuple, list)) and len(out) == 3, (
        "SimpleCLIP.forward should return (loss, image_features, text_features), got "
        + (f"a {type(out).__name__} of length {len(out)}"
           if isinstance(out, (tuple, list)) else f"a bare {type(out).__name__}"))
    return out


def check_model_shapes_and_temperature(ns):
    torch.manual_seed(0)
    model = ns.SimpleCLIP(IMG_DIM, TXT_DIM, EMB_DIM)
    images, texts = _paired_data(6)
    loss, img_f, txt_f = _clip_forward(model, images, texts)
    assert loss.dim() == 0 or loss.numel() == 1, \
        f"loss should be a scalar, got shape {tuple(loss.shape)}"
    assert tuple(img_f.shape) == (B_SIZE, EMB_DIM), \
        f"image_features should be {(B_SIZE, EMB_DIM)}, got {tuple(img_f.shape)}"
    assert tuple(txt_f.shape) == (B_SIZE, EMB_DIM), \
        f"text_features should be {(B_SIZE, EMB_DIM)}, got {tuple(txt_f.shape)}"
    scalars = [p for p in model.parameters() if p.requires_grad and p.numel() == 1]
    assert scalars, (
        "SimpleCLIP has no learnable scalar parameter — the (log-)temperature "
        "should be an nn.Parameter, not a constant")


def check_gradients_reach_every_parameter(ns):
    torch.manual_seed(1)
    model = ns.SimpleCLIP(IMG_DIM, TXT_DIM, EMB_DIM)
    images, texts = _paired_data(7)
    _clip_forward(model, images, texts)[0].backward()
    dead = [n for n, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    assert not dead, (
        f"no gradient reached {dead} — every encoder weight and the temperature "
        f"should be trained by the contrastive loss")


def check_training_aligns_the_pairs(ns):
    """End to end: the loss must fall and the similarity matrix must go diagonal."""
    torch.manual_seed(2)
    model = ns.SimpleCLIP(IMG_DIM, TXT_DIM, EMB_DIM)
    images, texts = _paired_data(8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = last = None
    for step in range(150):
        opt.zero_grad()
        loss = _clip_forward(model, images, texts)[0]
        loss.backward()
        opt.step()
        last = loss.item()
        if step == 0:
            first = last
    assert last < first, (
        f"loss did not decrease over 150 Adam steps ({first:.4f} -> {last:.4f}); "
        f"the correct pairs should become easier to identify")
    with torch.no_grad():
        _, img_f, txt_f = _clip_forward(model, images, texts)
        sim = F.normalize(img_f, dim=-1) @ F.normalize(txt_f, dim=-1).T
    diag = sim.diag().mean().item()
    off = sim[~torch.eye(B_SIZE, dtype=torch.bool)].mean().item()
    assert diag > off, (
        f"after training, mean diagonal similarity ({diag:.4f}) should exceed the "
        f"mean off-diagonal similarity ({off:.4f})")


CHECKS = [
    check_closed_form_on_orthonormal_features,
    check_invariant_to_feature_scale,
    check_symmetric_in_its_two_inputs,
    check_labels_follow_the_diagonal,
    check_temperature_is_used,
    check_matches_symmetric_cross_entropy,
    check_loss_backpropagates,
    check_model_shapes_and_temperature,
    check_gradients_reach_every_parameter,
    check_training_aligns_the_pairs,
]
