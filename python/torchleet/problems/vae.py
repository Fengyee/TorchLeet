"""vae — variational autoencoder: the reparameterization trick and the negative ELBO.

Conventions graded here (the notebook states them):

* the encoder outputs mu and logvar = log sigma^2 from two separate heads;
* reparameterize(mu, logvar) = mu + exp(0.5 * logvar) * eps with
  eps = torch.randn_like(...), so with a fixed seed the spec rebuilds eps exactly
  and can compare z and its gradients (d z / d mu = 1, d z / d logvar = 0.5*sigma*eps);
* the decoder returns Bernoulli *logits*; sample(n) returns sigmoid(decode(z)) with
  z = torch.randn(n, latent_dim);
* vae_loss = recon + beta * kl, where recon is BCE-with-logits summed over pixels and
  kl = -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) summed over latent dims, both
  averaged over the batch. KL is checked against torch.distributions.kl_divergence;
  the reduction convention, beta, numerical stability on saturated logits and the
  gradients are each pinned separately.

Training is checked with the spec's own reference loss so a bug in vae_loss and a bug
in the model are reported by different checks.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ENTRIES = ["VAE", "vae_loss"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "The encoder predicts logvar = log(sigma^2), not sigma, so the standard deviation "
    "is exp(0.5 * logvar). Sampling z = mu + sigma * eps with eps drawn separately "
    "keeps z a differentiable function of mu and logvar.",
    "Reconstruction: F.binary_cross_entropy_with_logits(logits, x, reduction='sum') "
    "divided by the batch size, so it is summed over pixels and averaged over "
    "samples. Use the same convention for the KL: sum over latent dims, mean over "
    "the batch.",
    "KL(N(mu, sigma^2) || N(0, 1)) per latent dim is -0.5 * (1 + logvar - mu^2 - "
    "exp(logvar)); sum it over latent dims, average over the batch, then "
    "loss = recon + beta * kl. sample(n) decodes z = torch.randn(n, latent_dim) "
    "and applies a sigmoid.",
]

D, HID, LAT = 64, 64, 4


def make_bars(n, size=8, seed=0):
    """Binary 'bars' images: each image has a random subset of full rows OR columns on."""
    g = torch.Generator().manual_seed(seed)
    lines = (torch.rand(n, size, generator=g) < 0.25).float()
    vertical = (torch.rand(n, 1, 1, generator=g) < 0.5).float()
    rows = lines[:, :, None].expand(n, size, size)
    cols = lines[:, None, :].expand(n, size, size)
    return (vertical * cols + (1 - vertical) * rows).reshape(n, size * size)


class _one_thread:
    """Tiny models train ~10x faster on one thread than on an oversubscribed pool."""

    def __enter__(self):
        self.n = torch.get_num_threads()
        torch.set_num_threads(1)

    def __exit__(self, *exc):
        torch.set_num_threads(self.n)


def _build(ns, d=D, h=HID, lat=LAT, seed=0):
    torch.manual_seed(seed)
    try:
        m = ns.VAE(d, h, lat)
    except TypeError:
        m = ns.VAE(input_dim=d, hidden_dim=h, latent_dim=lat)
    m.train()
    return m


def _need(m, name):
    assert callable(getattr(m, name, None)), (
        f"VAE needs a `{name}` method — the problem asks for encode(x) -> (mu, logvar), "
        f"reparameterize(mu, logvar) -> z, decode(z) -> logits, forward(x) and sample(n)")
    return getattr(m, name)


def _ref_kl(mu, logvar):
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()


def _ref_loss(logits, x, mu, logvar, beta=1.0):
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(-1).mean()
    kl = _ref_kl(mu, logvar)
    return recon + beta * kl, recon, kl


def _unpack(out, what="vae_loss"):
    assert isinstance(out, (tuple, list)) and len(out) == 3, (
        f"{what} must return a tuple (loss, recon, kl), got {type(out).__name__}")
    vals = []
    for name, v in zip(("loss", "recon", "kl"), out):
        assert torch.is_tensor(v), (
            f"{name} must be a tensor (keep it in the graph — no .item()), "
            f"got {type(v).__name__}")
        assert v.numel() == 1, f"{name} must be a scalar, got shape {tuple(v.shape)}"
        vals.append(v.reshape(()))
    return vals


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def check_encode_decode_forward_shapes(ns):
    m = _build(ns)
    x = make_bars(5)
    mu, logvar = _need(m, "encode")(x)
    assert tuple(mu.shape) == (5, LAT) and tuple(logvar.shape) == (5, LAT), (
        f"encode(x) must return mu and logvar of shape (batch, latent_dim) = {(5, LAT)}, "
        f"got {tuple(mu.shape)} and {tuple(logvar.shape)}")
    assert not torch.allclose(mu, logvar), (
        "mu and logvar are identical — they must come from two separate linear heads")
    out = m(x)
    assert isinstance(out, (tuple, list)) and len(out) == 3, \
        "forward(x) must return a tuple (recon_logits, mu, logvar)"
    recon, mu2, lv2 = out
    assert tuple(recon.shape) == (5, D), \
        f"recon_logits must have the input's shape {(5, D)}, got {tuple(recon.shape)}"
    assert tuple(mu2.shape) == (5, LAT) and tuple(lv2.shape) == (5, LAT), \
        f"forward's mu/logvar must be {(5, LAT)}, got {tuple(mu2.shape)}, {tuple(lv2.shape)}"
    torch.manual_seed(1)
    dec = _need(m, "decode")(3.0 * torch.randn(32, LAT))
    assert tuple(dec.shape) == (32, D), \
        f"decode(z) for z of shape {(32, LAT)} must return {(32, D)}, got {tuple(dec.shape)}"
    assert (dec < 0).any() or (dec > 1).any(), (
        "decode(z) only produced values in [0, 1] — it should return Bernoulli *logits* "
        "(no sigmoid at the end); vae_loss applies BCE-with-logits and sample() the sigmoid")


def check_reparameterize_is_mu_plus_sigma_eps(ns):
    m = _build(ns)
    rep = _need(m, "reparameterize")
    torch.manual_seed(0)
    mu, logvar = 2.0 * torch.randn(6, LAT), torch.randn(6, LAT)
    torch.manual_seed(123)
    z = rep(mu, logvar)
    torch.manual_seed(123)
    eps = torch.randn_like(mu)
    want = mu + torch.exp(0.5 * logvar) * eps
    assert tuple(z.shape) == tuple(mu.shape), \
        f"z must have mu's shape {tuple(mu.shape)}, got {tuple(z.shape)}"
    if torch.allclose(z, want, atol=1e-5):
        return
    if torch.allclose(z, mu, atol=1e-6):
        raise AssertionError(
            "reparameterize returned mu unchanged in training mode — z must be a sample "
            "mu + sigma * eps (returning mu at eval time is optional, but training needs noise)")
    if torch.allclose(z, mu + torch.exp(logvar) * eps, atol=1e-5):
        raise AssertionError(
            "z = mu + exp(logvar) * eps scales the noise by the *variance*. logvar is "
            "log(sigma^2), so sigma = exp(0.5 * logvar)")
    if torch.allclose(z, mu + logvar * eps, atol=1e-5):
        raise AssertionError("z = mu + logvar * eps uses logvar as the std; sigma = exp(0.5 * logvar)")
    raise AssertionError(
        f"z differs from mu + exp(0.5*logvar) * eps with eps = torch.randn_like(mu) drawn "
        f"right after torch.manual_seed (max diff {(z - want).abs().max():.3e}). Draw eps "
        f"with one torch.randn_like call and no other random op inside reparameterize")


def check_reparameterize_gradients(ns):
    """Pathwise gradients: dz/dmu = 1, dz/dlogvar = 0.5 * sigma * eps."""
    m = _build(ns)
    torch.manual_seed(0)
    mu = torch.randn(6, LAT, requires_grad=True)
    logvar = (0.5 * torch.randn(6, LAT)).requires_grad_(True)
    torch.manual_seed(7)
    z = _need(m, "reparameterize")(mu, logvar)
    assert z.requires_grad, (
        "z does not require grad — sampling broke the graph (e.g. Normal(...).sample(), "
        "torch.normal, or .detach()). The point of the trick is that z = mu + sigma*eps "
        "is differentiable in mu and logvar")
    z.sum().backward()
    torch.manual_seed(7)
    eps = torch.randn_like(mu)
    assert mu.grad is not None and torch.allclose(mu.grad, torch.ones_like(mu), atol=1e-5), (
        "d z / d mu should be exactly 1 for every element"
        + ("" if mu.grad is None else f" (got values in [{mu.grad.min():.3f}, {mu.grad.max():.3f}])"))
    want = 0.5 * torch.exp(0.5 * logvar.detach()) * eps
    assert logvar.grad is not None and torch.allclose(logvar.grad, want, atol=1e-5), (
        "d z / d logvar should be 0.5 * exp(0.5*logvar) * eps — check sigma = exp(0.5*logvar)")


def check_reparameterize_is_stochastic(ns):
    m = _build(ns)
    rep = _need(m, "reparameterize")
    torch.manual_seed(0)
    mu = torch.full((20000, 1), 1.5)
    logvar = torch.full((20000, 1), math.log(0.25))
    a, b = rep(mu, logvar), rep(mu, logvar)
    assert not torch.allclose(a, b), "two calls to reparameterize returned the same z — no fresh noise"
    assert abs(float(a.mean()) - 1.5) < 0.03 and abs(float(a.std()) - 0.5) < 0.03, (
        f"z for mu = 1.5, sigma = 0.5 should have mean 1.5 and std 0.5; got mean "
        f"{float(a.mean()):.3f}, std {float(a.std()):.3f}")


def check_forward_is_decode_of_reparameterized_encoding(ns):
    m = _build(ns)
    x = make_bars(8, seed=1)
    torch.manual_seed(5)
    recon, mu, logvar = m(x)
    mu2, lv2 = m.encode(x)
    assert torch.allclose(mu, mu2, atol=1e-6) and torch.allclose(logvar, lv2, atol=1e-6), \
        "forward's mu/logvar differ from encode(x)"
    torch.manual_seed(5)
    z = mu2 + torch.exp(0.5 * lv2) * torch.randn_like(mu2)
    want = m.decode(z)
    assert torch.allclose(recon, want, atol=1e-5), (
        "forward(x) should be decode(reparameterize(*encode(x))) — with the same seed "
        f"the logits differ by up to {(recon - want).abs().max():.3e}. Decode the sampled "
        "z, not mu, and draw no other random numbers in forward")


def check_sample_decodes_prior_draws(ns):
    m = _build(ns)
    m.eval()
    torch.manual_seed(3)
    s = _need(m, "sample")(10)
    assert torch.is_tensor(s) and tuple(s.shape) == (10, D), \
        f"sample(10) must return a (10, {D}) tensor, got {getattr(s, 'shape', type(s))}"
    assert bool(((s >= 0) & (s <= 1)).all()), (
        "sample(n) must return Bernoulli means sigmoid(decode(z)) in [0, 1], "
        "not raw logits")
    torch.manual_seed(3)
    with torch.no_grad():
        want = torch.sigmoid(m.decode(torch.randn(10, LAT)))
    assert torch.allclose(s.detach(), want, atol=1e-5), (
        "sample(n) should decode z = torch.randn(n, latent_dim) drawn from the prior "
        f"N(0, I) and apply a sigmoid (max diff {(s.detach() - want).abs().max():.3e})")
    assert not torch.allclose(s, m.sample(10)), "two calls to sample() returned the same images"


def check_gradients_reach_encoder_heads_and_decoder(ns):
    m = _build(ns)
    x = make_bars(16, seed=2)
    recon, mu, logvar = m(x)
    _ref_loss(recon, x, mu, logvar)[0].backward()
    dead = [n for n, p in m.named_parameters()
            if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
    assert not dead, (
        f"no gradient reached {', '.join(dead[:4])} — every encoder, mu/logvar head and "
        f"decoder parameter must be on the path of the ELBO")


def check_vae_learns_bars(ns):
    with _one_thread():
        _train_on_bars(ns)


def _train_on_bars(ns):
    m = _build(ns, d=D, h=128, lat=8, seed=0)
    data = make_bars(512, seed=3)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)

    def full_loss():
        torch.manual_seed(11)
        with torch.no_grad():
            r, mu, lv = m(data)
            return float(_ref_loss(r, data, mu, lv)[0])

    first = full_loss()
    g = torch.Generator().manual_seed(4)
    for _ in range(300):
        xb = data[torch.randint(0, 512, (64,), generator=g)]
        r, mu, lv = m(xb)
        loss = _ref_loss(r, xb, mu, lv)[0]
        opt.zero_grad()
        loss.backward()
        opt.step()
    last = full_loss()
    # A decoder that ignores z can at best reach the per-pixel marginal entropy
    # (~0.8 * first here); using the latent gets well below 0.7 * first.
    assert last < 0.7 * first, (
        f"the negative ELBO only went from {first:.2f} to {last:.2f} nats per image after "
        f"300 Adam steps on 8x8 bar images (expected < {0.7 * first:.2f}) — the VAE is not "
        f"learning, or the decoder is not using z")


# --------------------------------------------------------------------------
# loss
# --------------------------------------------------------------------------
def _loss_inputs(seed=0, b=7, lat=5):
    g = torch.Generator().manual_seed(seed)
    logits = 2.0 * torch.randn(b, D, generator=g)
    x = (torch.rand(b, D, generator=g) < 0.4).float()
    mu = 1.5 * torch.randn(b, lat, generator=g)
    logvar = torch.randn(b, lat, generator=g)
    return logits, x, mu, logvar


def check_kl_matches_torch_distributions(ns):
    from torch.distributions import Normal, kl_divergence
    logits, x, mu, logvar = _loss_inputs()
    _, _, kl = _unpack(ns.vae_loss(logits, x, mu, logvar))
    want = kl_divergence(Normal(mu, torch.exp(0.5 * logvar)), Normal(0.0, 1.0)).sum(-1).mean()
    if torch.allclose(kl, want, rtol=1e-5, atol=1e-5):
        return
    b, lat = mu.shape
    hints = {
        "summed over the batch instead of averaged": want * b,
        "averaged over latent dims instead of summed": want / lat,
        "missing the -0.5 factor": want * -2.0,
        "the sign is flipped": -want,
    }
    for why, v in hints.items():
        if torch.allclose(kl, v, rtol=1e-5, atol=1e-5):
            raise AssertionError(f"kl = {float(kl):.4f} but expected {float(want):.4f}: {why}")
    raise AssertionError(
        f"kl = {float(kl):.4f} but KL(N(mu, sigma^2) || N(0, I)) summed over latent dims and "
        f"averaged over the batch is {float(want):.4f}. Per dim it is "
        f"-0.5 * (1 + logvar - mu^2 - exp(logvar))")


def check_kl_is_zero_exactly_at_the_prior(ns):
    _, x, _, _ = _loss_inputs()
    logits = torch.zeros(7, D)
    _, _, kl0 = _unpack(ns.vae_loss(logits, x, torch.zeros(7, 3), torch.zeros(7, 3)))
    assert abs(float(kl0)) < 1e-6, f"q = N(0, I) equals the prior, so kl must be 0, got {float(kl0):.3e}"
    _, _, kl1 = _unpack(ns.vae_loss(logits, x, torch.full((7, 3), 0.1), torch.zeros(7, 3)))
    _, _, kl2 = _unpack(ns.vae_loss(logits, x, torch.zeros(7, 3), torch.full((7, 3), -0.5)))
    assert float(kl1) > 0 and float(kl2) > 0, \
        "kl must be positive whenever q differs from the prior (shifted mean or shrunk variance)"


def check_recon_is_summed_over_pixels_and_averaged_over_batch(ns):
    logits, x, mu, logvar = _loss_inputs(seed=1)
    _, recon, _ = _unpack(ns.vae_loss(logits, x, mu, logvar))
    per = F.binary_cross_entropy_with_logits(logits, x, reduction="none")
    want = per.sum(-1).mean()
    if torch.allclose(recon, want, rtol=1e-5, atol=1e-4):
        return
    if torch.allclose(recon, per.mean(), rtol=1e-5, atol=1e-5):
        raise AssertionError(
            f"recon = {float(recon):.4f} is the mean over *all* elements; the ELBO's log "
            f"p(x|z) is a sum over the {D} pixels, so sum over pixels and average only over "
            f"the batch (expected {float(want):.4f}). Mean-over-pixels silently multiplies "
            f"the KL weight by {D}")
    if torch.allclose(recon, per.sum(), rtol=1e-5, atol=1e-3):
        raise AssertionError(
            f"recon = {float(recon):.2f} is summed over the batch too; average over the "
            f"batch (expected {float(want):.4f})")
    raise AssertionError(
        f"recon = {float(recon):.4f}, expected BCE-with-logits summed over pixels and "
        f"averaged over the batch = {float(want):.4f}")


def check_loss_is_recon_plus_beta_kl(ns):
    logits, x, mu, logvar = _loss_inputs(seed=2)
    want_loss, want_recon, want_kl = _ref_loss(logits, x, mu, logvar)
    for beta in (1.0, 0.0, 0.25, 4.0):
        loss, recon, kl = _unpack(ns.vae_loss(logits, x, mu, logvar, beta=beta))
        assert torch.allclose(loss, recon + beta * kl, rtol=1e-5, atol=1e-4), (
            f"with beta = {beta}, loss = {float(loss):.4f} but recon + beta * kl = "
            f"{float(recon + beta * kl):.4f}")
        assert torch.allclose(loss, want_recon + beta * want_kl, rtol=1e-5, atol=1e-4), (
            f"with beta = {beta} the negative ELBO should be {float(want_recon + beta * want_kl):.4f}, "
            f"got {float(loss):.4f}")


def check_loss_gradients_flow_to_all_inputs(ns):
    logits, x, mu, logvar = _loss_inputs(seed=3)
    leaves = [t.clone().requires_grad_(True) for t in (logits, mu, logvar)]
    loss, _, _ = _unpack(ns.vae_loss(leaves[0], x, leaves[1], leaves[2], beta=0.5))
    assert loss.requires_grad, "the loss is not differentiable — something was detached or .item()-ed"
    loss.backward()
    ref = [t.clone().requires_grad_(True) for t in (logits, mu, logvar)]
    _ref_loss(ref[0], x, ref[1], ref[2], beta=0.5)[0].backward()
    for name, a, b in zip(("recon_logits", "mu", "logvar"), leaves, ref):
        assert a.grad is not None, f"no gradient reached {name}"
        assert torch.allclose(a.grad, b.grad, rtol=1e-4, atol=1e-6), (
            f"gradient w.r.t. {name} is wrong (max diff {(a.grad - b.grad).abs().max():.3e})")


def check_loss_is_stable_for_saturated_logits(ns):
    """sigmoid(50) == 1.0 in float32: log(sigmoid) + BCE gives inf or a clamped -100."""
    x = torch.tensor([[1.0, 0.0, 1.0, 0.0] * (D // 4)])
    logits = torch.tensor([[-50.0, 50.0, 60.0, -60.0] * (D // 4)])
    mu, logvar = torch.zeros(1, 2), torch.zeros(1, 2)
    _, recon, _ = _unpack(ns.vae_loss(logits, x, mu, logvar))
    want = F.binary_cross_entropy_with_logits(logits, x, reduction="sum")
    assert torch.isfinite(recon), (
        "recon is inf/NaN for large logits — compute BCE from logits "
        "(F.binary_cross_entropy_with_logits / log-sum-exp form), not log(sigmoid(.))")
    if torch.allclose(recon, want, rtol=1e-4):
        return
    if torch.allclose(recon, want / D, rtol=1e-4):
        raise AssertionError("recon is averaged over pixels — sum over pixels, average over the batch")
    raise AssertionError(
        f"with logits of +-50..60 the summed BCE is exactly {float(want):.1f} nats but you got "
        f"{float(recon):.1f} — sigmoid(50) rounds to 1.0 in float32, so sigmoid-then-BCE "
        f"saturates (F.binary_cross_entropy clamps log at -100, an eps inside log caps it "
        f"at ~16); compute BCE from the logits directly")


CHECKS = [
    check_encode_decode_forward_shapes,
    check_reparameterize_is_mu_plus_sigma_eps,
    check_reparameterize_gradients,
    check_reparameterize_is_stochastic,
    check_forward_is_decode_of_reparameterized_encoding,
    check_sample_decodes_prior_draws,
    check_gradients_reach_encoder_heads_and_decoder,
    check_vae_learns_bars,
    check_kl_matches_torch_distributions,
    check_kl_is_zero_exactly_at_the_prior,
    check_recon_is_summed_over_pixels_and_averaged_over_batch,
    check_loss_is_recon_plus_beta_kl,
    check_loss_gradients_flow_to_all_inputs,
    check_loss_is_stable_for_saturated_logits,
]
