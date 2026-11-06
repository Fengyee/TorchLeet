"""autoencoder — convolutional autoencoder for anomaly detection.

Two entries: the model, and the per-image anomaly score built on it. The model is
free-form, so the checks pin the contract the problem states (same shape out as
in, pixels in [0, 1], a latent smaller than the input) and that it trains. The
score is checked against its definition — per-image mean squared reconstruction
error — up to a positive constant, so a sum instead of a mean still passes while
the classic bug (one batch-mean loss for the whole batch) does not.
"""
import torch
import torch.nn.functional as F
from torchleet.checks import common as c
from torchleet.runner import Skip

ENTRIES = ["Autoencoder", "anomaly_scores"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "Without a bottleneck an autoencoder can learn the identity and reconstruct "
    "anomalies as well as normal digits — compress each 784-pixel image to a code "
    "much smaller than that (e.g. conv/pool down to 7x7, then a Linear to 32 numbers).",
    "Mirror the encoder: Linear back to the 7x7 feature map, then "
    "ConvTranspose2d(kernel_size=3, stride=2, padding=1, output_padding=1) twice "
    "(7 -> 14 -> 28) and a Sigmoid, because the pixels are in [0, 1].",
    "Score every image separately — average the squared error over C, H and W but "
    "not over the batch, so 64 images give 64 scores. Set the threshold to a high "
    "percentile (e.g. the 95th) of the scores on held-out normal images, never on "
    "test labels.",
]

SHAPE = (1, 28, 28)
N_PIXELS = 28 * 28


def check_reconstructs_input_shape(ns):
    """An autoencoder whose output shape differs from its input cannot reconstruct."""
    m = c.build(ns, "Autoencoder")
    out = m(torch.rand(4, *SHAPE))
    assert tuple(out.shape) == (4, *SHAPE), \
        f"output {tuple(out.shape)} does not match the input {(4, *SHAPE)}"


def check_output_is_in_pixel_range(ns):
    """The inputs are pixel intensities in [0, 1]; the problem asks for a Sigmoid."""
    m = c.build(ns, "Autoencoder")
    with torch.no_grad():
        out = m(torch.randn(4, *SHAPE) * 3)
    lo, hi = float(out.min()), float(out.max())
    assert lo >= 0.0 and hi <= 1.0, (
        f"reconstructions range over [{lo:.3f}, {hi:.3f}] but pixels are in [0, 1] — "
        "end the decoder with a Sigmoid")


def check_encoder_is_a_bottleneck(ns):
    """The latent code must hold fewer numbers than the image.

    An over-complete code (e.g. a 64x7x7 feature map = 3136 numbers for a 784-pixel
    image) lets the network approach the identity and reconstruct anomalies too,
    which is exactly what an anomaly detector must not do.
    """
    m = c.build(ns, "Autoencoder")
    enc = getattr(m, "encoder", None)
    if enc is None:
        raise Skip("no .encoder attribute to inspect")
    with torch.no_grad():
        z = enc(torch.rand(2, *SHAPE))
    per_image = z[0].numel()
    assert per_image < N_PIXELS, (
        f"the encoder's code has {per_image} numbers per image, not fewer than the "
        f"{N_PIXELS} input pixels — there is no bottleneck, so the model can learn "
        "to copy anomalies too")


def check_learns_to_reconstruct(ns):
    """Overfit a fixed batch of [0, 1] images to themselves; the MSE must fall."""
    torch.manual_seed(0)
    m = c.build(ns, "Autoencoder")
    x = (torch.rand(8, *SHAPE) > 0.8).float()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    first = F.mse_loss(m(x), x).item()
    loss = None
    for _ in range(30):
        opt.zero_grad()
        loss = F.mse_loss(m(x), x)
        loss.backward()
        opt.step()
    assert loss.item() < first, (
        f"reconstruction loss did not decrease over 30 steps on a fixed batch "
        f"({first:.4f} -> {loss.item():.4f}) — the model is not learning")


def check_anomaly_scores_are_per_image_mse(ns):
    """One score per image, equal to its mean squared reconstruction error (a
    constant multiple — e.g. the sum over pixels — is accepted)."""
    m = c.build(ns, "Autoencoder")
    m.eval()
    # Images with very different brightness, so their errors differ a lot and a
    # score that is not per-image MSE (batch mean, L1, max) cannot line up by luck.
    x = torch.rand(6, *SHAPE) * torch.linspace(0.05, 1.0, 6).view(6, 1, 1, 1)
    s = ns.anomaly_scores(m, x)
    s = torch.as_tensor(s).detach().float()
    assert s.ndim == 1 and s.shape[0] == 6, (
        f"anomaly_scores returned shape {tuple(s.shape)} for a batch of 6 — it must "
        "return one score per image, shape (6,); a single batch-mean loss cannot say "
        "which image is anomalous")
    with torch.no_grad():
        want = ((m(x) - x) ** 2).flatten(1).mean(dim=1)
    ratio = s / want
    assert torch.all(ratio > 0), "scores must be positive multiples of the reconstruction error"
    spread = float((ratio.max() - ratio.min()) / ratio.mean())
    assert spread < 1e-3, (
        "scores are not proportional to each image's mean squared reconstruction "
        f"error (score/MSE ranges over {[round(float(r), 4) for r in ratio]}) — square "
        "the error and average it over channel, height and width")


CHECKS = [
    check_reconstructs_input_shape,
    check_output_is_in_pixel_range,
    check_encoder_is_a_bottleneck,
    c.gradients_flow("Autoencoder", SHAPE),
    check_learns_to_reconstruct,
    check_anomaly_scores_are_per_image_mse,
]
