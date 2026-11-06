"""vq-vae — vector quantization with straight-through gradients, an EMA codebook, perplexity.

Conventions graded here (the notebook states them; sources in brackets):

* VectorQuantizer(num_embeddings=K, embedding_dim=D, beta=0.25) keeps its codebook in
  `self.embedding = nn.Embedding(K, D)`. forward(z) takes a channel-first z of shape
  (B, D, H, W) and returns (z_q, loss, indices):
  - indices: (B, H, W) int64, indices[b, h, w] = argmin_k ||z[b, :, h, w] - e_k||_2
    [van den Oord et al. 2017, "Neural Discrete Representation Learning", eqs. 1-2];
  - z_q: shape of z, value e_{indices} at every position; gradient by the straight-through
    estimator z + (z_q - z).detach(), so d z_q / d z = I and the codebook gets no gradient
    through z_q [Sec. 3.2: "copy gradients from decoder input z_q(x) to encoder output z_e(x)"];
  - loss = ||sg[z] - e||^2 + beta * ||z - sg[e]||^2 [eq. 3 without the reconstruction term],
    each term reduced by the MEAN over all B*D*H*W elements (= the paper's squared norm per
    position divided by D, averaged over positions), as in DeepMind Sonnet's VectorQuantizer
    (tf.reduce_mean) and F.mse_loss. beta = 0.25 is the paper's value.
  The two terms have the same VALUE; they differ only in where gradients go, so the beta
  placement and the stop-gradients are checked through gradients, not the loss value.

* VectorQuantizerEMA(K, D, beta=0.25, decay=0.99, eps=1e-5): codebook weight frozen
  (requires_grad False), buffers `ema_cluster_size` (K,) = N and `ema_embed_sum` (K, D) = m.
  In training mode, after quantizing with the CURRENT codebook (z_q and loss use it):
      n_k = #{j : k_j = k},   s_k = sum_{j : k_j = k} z_j                (batch statistics)
      N  <- decay * N + (1 - decay) * n                                  [App. A.1, N_i^(t)]
      m  <- decay * m + (1 - decay) * s                                  [App. A.1, m_i^(t)]
      N~ = (N + eps) / (sum_k N_k + K * eps) * sum_k N_k                 [Laplace smoothing,
                                                                           Sonnet VectorQuantizerEMA]
      e  <- m / N~                                                       [App. A.1, e_i^(t)]
  N stores the RAW EMA (smoothing is applied only for the division), as in Sonnet v1/v2,
  rosinality/vq-vae-2-pytorch, lucidrains/vector-quantize-pytorch and taming-transformers'
  EmbeddingEMA; zalandoresearch/pytorch-vq-vae writes the smoothed copy back, which fails
  check_ema_one_update_matches_the_equations with a message naming the convention (its codebook
  after one step is identical, and over several steps differs only by O(eps)). The EMAs are
  plain (no zero-debiasing; Sonnet's ExponentialMovingAverage starts its hidden state at zero
  and divides by 1 - decay^t, so a faithful Sonnet port fails the update checks). Fresh state is consistent:
  m / N~ equals the initial codebook (the reference uses N = 1, m = e). loss = beta * mse(z, sg[e])
  (commitment only). Eval mode changes nothing.

* codebook_perplexity(indices, K) = exp(-sum_k p_k log p_k), p_k = fraction of positions assigned
  to code k, with 0 log 0 = 0 (Sonnet's perplexity; it adds 1e-10 inside the log, which the
  tolerance here also accepts).

All references are recomputed in float64 inside this file; brute-force distances are explicit
(z - e)^2 sums, not the ||z||^2 - 2 z.e + ||e||^2 expansion.
"""
import torch
import torch.nn as nn

ENTRIES = ["VectorQuantizer", "VectorQuantizerEMA", "codebook_perplexity"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "The quantizer is a nearest-neighbour lookup over the codebook, done for every spatial "
    "position at once: move the channel axis last and flatten so each row is one D-dimensional "
    "vector, compute all row-to-code distances in one shot, take the argmin, then undo the "
    "layout. The two loss terms have the same value; what makes them different is which side "
    "is detached.",
    "z.permute(0, 2, 3, 1).reshape(-1, D) gives (B*H*W, D) rows (a plain reshape of a "
    "channel-first tensor mixes channels and positions). ||z - e||^2 = ||z||^2 - 2 z.e + "
    "||e||^2 gives the (N, K) distance matrix with one matmul. loss = mse(z_q, z.detach()) + "
    "beta * mse(z, z_q.detach()); straight-through output z + (z_q - z).detach(). EMA: count "
    "the assignments n_k and sum the assigned vectors s_k (one_hot(idx).T @ z_flat), all "
    "under torch.no_grad().",
    "EMA step in training mode only, after computing z_q with the current codebook: N = decay*N "
    "+ (1-decay)*n; m = decay*m + (1-decay)*s; total = N.sum(); N_s = (N + eps) / (total + "
    "K*eps) * total; codebook = m / N_s[:, None] (keep the raw N in the buffer). Perplexity: "
    "p = bincount(indices.flatten(), minlength=K) / count; exp(-sum p log p) with 0 log 0 = 0 "
    "(torch.special.xlogy).",
]

F64 = torch.float64


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _vq(ns, k, d, beta=0.25, seed=0):
    torch.manual_seed(seed)
    try:
        m = ns.VectorQuantizer(k, d, beta=beta)
    except TypeError:
        m = ns.VectorQuantizer(num_embeddings=k, embedding_dim=d, beta=beta)
    m.train()
    return m


def _ema(ns, k, d, beta=0.25, decay=0.99, eps=1e-5, seed=0):
    torch.manual_seed(seed)
    try:
        m = ns.VectorQuantizerEMA(k, d, beta=beta, decay=decay, eps=eps)
    except TypeError:
        m = ns.VectorQuantizerEMA(num_embeddings=k, embedding_dim=d, beta=beta,
                                  decay=decay, eps=eps)
    m.train()
    return m


def _codebook(m, name):
    emb = getattr(m, "embedding", None)
    assert isinstance(emb, nn.Embedding), (
        f"{name} must keep its codebook in `self.embedding = nn.Embedding(num_embeddings, "
        f"embedding_dim)`, got {type(emb).__name__}")
    return emb


def _set_codebook(m, e):
    with torch.no_grad():
        m.embedding.weight.copy_(e)


def _unpack(out, z, what):
    assert isinstance(out, (tuple, list)) and len(out) == 3, (
        f"{what}.forward(z) must return a tuple (z_q, loss, indices), got {type(out).__name__}")
    z_q, loss, idx = out
    B, D, H, W = z.shape
    assert torch.is_tensor(z_q) and tuple(z_q.shape) == (B, D, H, W), (
        f"z_q must have z's channel-first shape {(B, D, H, W)}, got "
        f"{tuple(getattr(z_q, 'shape', ()))} — permute back after the lookup")
    assert torch.is_tensor(idx) and tuple(idx.shape) == (B, H, W), (
        f"indices must have shape (B, H, W) = {(B, H, W)}, got {tuple(getattr(idx, 'shape', ()))}")
    assert idx.dtype == torch.int64, f"indices must be int64 (torch.long), got {idx.dtype}"
    assert torch.is_tensor(loss) and loss.numel() == 1, (
        f"loss must be a scalar tensor (keep it in the graph — no .item()), got "
        f"{type(loss).__name__} {tuple(getattr(loss, 'shape', ()))}")
    return z_q, loss.reshape(()), idx


def _ref_indices(z, e):
    """Brute force in float64: explicit squared differences, (B, H, W, K) distances."""
    zz = z.to(F64).permute(0, 2, 3, 1)[..., None, :]          # (B, H, W, 1, D)
    d = ((zz - e.to(F64)) ** 2).sum(-1)                         # (B, H, W, K)
    return d.argmin(-1), d


def _layout(e, idx):
    """codebook rows at indices (B, H, W) laid out channel-first (B, D, H, W)."""
    return e[idx].permute(0, 3, 1, 2)


# A codebook whose codes have DIFFERENT norms (so argmax z.e != argmin ||z - e||) and are
# >= 3 apart; z planted at code + N(0, 0.3^2) noise is then unambiguous.
_E_PLANT = torch.tensor([[0.0, 0.0, 0.0],
                         [3.0, 0.0, 0.0],
                         [0.0, 3.5, 0.0],
                         [0.0, 0.0, -3.0],
                         [3.0, 3.0, 3.0],
                         [-3.0, 1.0, 0.0],
                         [-6.0, -6.0, 6.0]])


def _planted(b=2, h=4, w=5, seed=0, k=None, noise=0.3):
    g = torch.Generator().manual_seed(seed)
    e = _E_PLANT if k is None else _E_PLANT[:k]
    idx = torch.randint(0, e.shape[0], (b, h, w), generator=g)
    z = _layout(e, idx) + noise * torch.randn(b, e.shape[1], h, w, generator=g)
    return e.clone(), z, idx


# --------------------------------------------------------------------------
# VectorQuantizer
# --------------------------------------------------------------------------
def check_vq_codebook_and_output_shapes(ns):
    m = _vq(ns, 7, 3)
    emb = _codebook(m, "VectorQuantizer")
    assert tuple(emb.weight.shape) == (7, 3), \
        f"the codebook must be (num_embeddings, embedding_dim) = (7, 3), got {tuple(emb.weight.shape)}"
    assert emb.weight.requires_grad, \
        "VectorQuantizer's codebook is trained by the codebook loss, so its weight must require grad"
    for shape in ((2, 3, 4, 5), (1, 3, 1, 1), (3, 3, 2, 2)):
        z = torch.randn(*shape)
        keep = z.clone()
        z_q, loss, idx = _unpack(m(z), z, "VectorQuantizer")
        assert z_q.dtype == z.dtype, f"z_q must keep z's dtype {z.dtype}, got {z_q.dtype}"
        assert torch.equal(z, keep), "forward must not modify z in place"
        assert bool(((idx >= 0) & (idx < 7)).all()), "indices must lie in [0, num_embeddings)"
    m1 = _vq(ns, 4, 1)
    z = torch.randn(2, 1, 3, 3)
    _unpack(m1(z), z, "VectorQuantizer")


def check_vq_indices_on_planted_codes(ns):
    """Every z is a code plus small noise, so the right index is unambiguous."""
    m = _vq(ns, 7, 3)
    e, z, want = _planted()
    _set_codebook(m, e)
    _, _, idx = _unpack(m(z), z, "VectorQuantizer")
    if torch.equal(idx, want):
        return
    wrong = (idx != want).float().mean().item()
    B, D, H, W = z.shape
    flat_bug = _ref_indices(z.reshape(B, H, W, D).permute(0, 3, 1, 2), e)[0]
    if torch.equal(idx, flat_bug):
        raise AssertionError(
            "indices match a nearest-code search on z.reshape(-1, D) WITHOUT permuting: z is "
            "channel-first (B, D, H, W), so a plain reshape/view puts values of different "
            "channels and positions in one row. Use z.permute(0, 2, 3, 1).reshape(-1, D)")
    dot = (z.permute(0, 2, 3, 1) @ e.t()).argmax(-1)
    if torch.equal(idx, dot):
        raise AssertionError(
            "indices maximise the dot product z.e: the nearest code minimises ||z||^2 - 2 z.e "
            "+ ||e||^2, and the ||e||^2 term matters when codes have different norms")
    raise AssertionError(
        f"{wrong:.0%} of the indices are wrong on inputs planted at codebook vectors + small "
        f"noise (first mismatch: got {idx[idx != want][0].item()}, expected "
        f"{want[idx != want][0].item()}). indices[b, h, w] must be argmin_k ||z[b, :, h, w] - e_k||")


def check_vq_indices_match_brute_force(ns):
    m = _vq(ns, 16, 4)
    g = torch.Generator().manual_seed(1)
    e = torch.randn(16, 4, generator=g)
    z = 1.5 * torch.randn(3, 4, 6, 6, generator=g)
    _set_codebook(m, e)
    _, _, idx = _unpack(m(z), z, "VectorQuantizer")
    want, d = _ref_indices(z, e)
    got_d = d.gather(-1, idx[..., None]).squeeze(-1)
    best = d.min(-1).values
    excess = (got_d - best).max().item()
    assert excess < 1e-4 * (1 + best.max().item()), (
        f"some position was not assigned its nearest code: its squared distance exceeds the "
        f"minimum by {excess:.3e} (brute force over all {16} codes). "
        f"{(idx != want).float().mean().item():.0%} of indices differ from the brute-force argmin")
    top2 = d.topk(2, dim=-1, largest=False).values
    clear = (top2[..., 1] - top2[..., 0]) > 1e-3
    assert torch.equal(idx[clear], want[clear]), \
        "indices differ from the brute-force argmin at positions without near-ties"


def check_vq_zq_is_the_selected_code(ns):
    m = _vq(ns, 7, 3)
    e, z, _ = _planted(seed=2)
    _set_codebook(m, e)
    z_q, _, idx = _unpack(m(z), z, "VectorQuantizer")
    want = _layout(e, idx)
    if torch.allclose(z_q, want, atol=1e-5):
        return
    if torch.allclose(z_q, z, atol=1e-6):
        raise AssertionError(
            "z_q equals the input z — the straight-through output must have the VALUE of the "
            "quantized codes. `z_q + (z - z_q).detach()` evaluates to z (and sends the gradient "
            "to the codebook); write z + (z_q - z).detach()")
    B, D, H, W = z.shape
    if torch.allclose(z_q, e[idx].reshape(B, D, H, W), atol=1e-5):
        raise AssertionError(
            "z_q holds the right codes but in the wrong layout: the lookup gives (B, H, W, D); "
            "permute(0, 3, 1, 2) back to channel-first instead of reshaping/viewing")
    raise AssertionError(
        f"z_q[b, :, h, w] must equal codebook[indices[b, h, w]] (max diff "
        f"{(z_q - want).abs().max().item():.3e})")


def check_vq_straight_through_gradient(ns):
    """Backprop through z_q alone: d z_q / d z = I and the codebook gets nothing."""
    m = _vq(ns, 7, 3)
    e, z, _ = _planted(seed=3)
    _set_codebook(m, e)
    z = z.clone().requires_grad_(True)
    z_q, _, _ = _unpack(m(z), z, "VectorQuantizer")
    assert z_q.requires_grad, (
        "z_q does not require grad — argmin/lookup is not differentiable, so the encoder needs "
        "the straight-through estimator z + (z_q - z).detach()")
    up = torch.randn(z.shape, generator=torch.Generator().manual_seed(4))
    (z_q * up).sum().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0, (
        "no gradient reached z through z_q — the encoder would never learn from the "
        "reconstruction loss. Use z_q = z + (z_q - z).detach()")
    assert torch.allclose(z.grad, up, atol=1e-6), (
        "d z_q / d z must be the identity (the decoder's gradient is copied to the encoder "
        f"unchanged), but the upstream gradient was altered (max diff {(z.grad - up).abs().max():.3e})")
    g = m.embedding.weight.grad
    assert g is None or torch.all(g == 0), (
        "the codebook received gradient from z_q itself — with the straight-through estimator "
        "the reconstruction gradient goes ONLY to the encoder; the codebook learns from the "
        "codebook loss")


def check_vq_loss_value(ns):
    e, z, _ = _planted(seed=5, noise=0.6)
    for beta in (0.25, 0.0, 2.0):
        m = _vq(ns, 7, 3, beta=beta)
        _set_codebook(m, e)
        _, loss, idx = _unpack(m(z), z, "VectorQuantizer")
        mse = ((_layout(e, idx) - z) ** 2).mean()
        want = (1 + beta) * mse
        if torch.allclose(loss, want, rtol=1e-5, atol=1e-7):
            continue
        B, D, H, W = z.shape
        hints = {
            "a SUM instead of a mean over elements": (1 + beta) * mse * z.numel(),
            "the paper's per-position squared norm (summed over D) averaged over positions; "
            "here both terms are means over all B*D*H*W elements (F.mse_loss)": (1 + beta) * mse * D,
            "only the codebook term (beta * commitment is missing)": mse,
            "beta applied to both terms": 2 * beta * mse,
        }
        for why, v in hints.items():
            if torch.allclose(loss, v, rtol=1e-5, atol=1e-7):
                raise AssertionError(f"with beta = {beta}: loss = {loss.item():.5f} is {why}")
        raise AssertionError(
            f"with beta = {beta}: loss = {loss.item():.5f}, expected mse(sg[z], e) + beta * "
            f"mse(z, sg[e]) = (1 + beta) * mean((z_q - z)^2) = {want.item():.5f}")


def check_vq_loss_gradients_are_isolated(ns):
    """Backprop the loss alone: the encoder sees only beta * commitment, the codebook only the
    codebook term."""
    beta = 0.25
    m = _vq(ns, 7, 3, beta=beta)
    e, z, _ = _planted(seed=6, noise=0.6)
    _set_codebook(m, e)
    z = z.clone().requires_grad_(True)
    _, loss, idx = _unpack(m(z), z, "VectorQuantizer")
    assert loss.requires_grad, "loss does not require grad — keep it in the graph"
    loss.backward()
    n = z.numel()
    zq = _layout(e, idx)
    g_commit = 2 * (z.detach() - zq) / n                       # d/dz of mse(z, sg[e])
    gz = z.grad
    assert gz is not None, "no gradient reached z from the loss — the commitment term must depend on z"
    if not torch.allclose(gz, beta * g_commit, atol=1e-7, rtol=1e-4):
        for why, v in {
            "beta multiplies the codebook term instead of the commitment term (the encoder "
            "gets the full codebook-term gradient)": g_commit,
            "the codebook term is missing its stop-gradient on z (mse(z_q, z) instead of "
            "mse(z_q, z.detach())), so the encoder gets (1 + beta) times the commitment gradient":
                (1 + beta) * g_commit,
        }.items():
            if torch.allclose(gz, v, atol=1e-7, rtol=1e-4):
                raise AssertionError(f"d loss / d z is wrong: {why}")
        raise AssertionError(
            "d loss / d z must be beta * 2 (z - sg[z_q]) / numel — only the beta-weighted "
            f"commitment term reaches the encoder (max diff {(gz - beta * g_commit).abs().max():.3e})")
    ge = m.embedding.weight.grad
    assert ge is not None, "the codebook got no gradient from the loss — the codebook term must depend on e"
    want_e = torch.zeros_like(e).index_add_(
        0, idx.flatten(), (zq - z.detach()).permute(0, 2, 3, 1).reshape(-1, 3)) * 2 / n
    if not torch.allclose(ge, want_e, atol=1e-7, rtol=1e-4):
        for why, v in {
            "beta multiplies the codebook term (the codebook gets beta times its gradient)": beta * want_e,
            "the commitment term is missing its stop-gradient on e (mse(z, z_q) instead of "
            "mse(z, z_q.detach())), so the codebook gets (1 + beta) times its gradient":
                (1 + beta) * want_e,
        }.items():
            if torch.allclose(ge, v, atol=1e-7, rtol=1e-4):
                raise AssertionError(f"d loss / d codebook is wrong: {why}")
        raise AssertionError(
            "d loss / d e_k must be sum over positions assigned to k of 2 (e_k - sg[z]) / numel — "
            f"only the codebook term reaches the codebook (max diff {(ge - want_e).abs().max():.3e})")


# --------------------------------------------------------------------------
# VectorQuantizerEMA
# --------------------------------------------------------------------------
def _smoothed(N, eps):
    total = N.sum()
    return (N + eps) / (total + N.numel() * eps) * total


def _buffers(m):
    bufs = dict(m.named_buffers())
    for name in ("ema_cluster_size", "ema_embed_sum"):
        if name not in bufs:
            params = dict(m.named_parameters())
            if name in params or isinstance(getattr(m, name, None), torch.Tensor):
                raise AssertionError(
                    f"`{name}` must be registered with self.register_buffer (EMA state is "
                    f"saved in state_dict and moved by .to(), but is not a parameter an "
                    f"optimizer should see)")
            raise AssertionError(
                f"VectorQuantizerEMA needs a buffer named `{name}` "
                f"(ema_cluster_size: (K,) EMA of code counts N; ema_embed_sum: (K, D) EMA of "
                f"assigned-vector sums m)")
    return m.ema_cluster_size, m.ema_embed_sum


def _set_ema_state(m, e, N, M):
    with torch.no_grad():
        m.embedding.weight.copy_(e)
        m.ema_cluster_size.copy_(N)
        m.ema_embed_sum.copy_(M)


def _ema_step_ref(e, N, M, z, decay, eps):
    """One training step of the reference in float64. Returns idx, z_q, new (e, N, M)."""
    e, N, M, z = e.to(F64), N.to(F64), M.to(F64), z.to(F64)
    idx = _ref_indices(z, e)[0]
    zq = _layout(e, idx)
    K, D = e.shape
    flat = z.permute(0, 2, 3, 1).reshape(-1, D)
    n = torch.bincount(idx.flatten(), minlength=K).to(F64)
    s = torch.zeros(K, D, dtype=F64).index_add_(0, idx.flatten(), flat)
    N1 = decay * N + (1 - decay) * n
    M1 = decay * M + (1 - decay) * s
    e1 = M1 / _smoothed(N1, eps)[:, None]
    return idx, zq, e1, N1, M1, n, s


def check_ema_buffers_and_frozen_codebook(ns):
    m = _ema(ns, 6, 3)
    emb = _codebook(m, "VectorQuantizerEMA")
    assert tuple(emb.weight.shape) == (6, 3), f"codebook must be (6, 3), got {tuple(emb.weight.shape)}"
    assert not emb.weight.requires_grad, (
        "the EMA codebook is not trained by gradient descent: call "
        "self.embedding.weight.requires_grad_(False)")
    N, M = _buffers(m)
    assert tuple(N.shape) == (6,) and tuple(M.shape) == (6, 3), (
        f"ema_cluster_size must be (K,) = (6,) and ema_embed_sum (K, D) = (6, 3), got "
        f"{tuple(N.shape)} and {tuple(M.shape)}")
    assert M.data_ptr() != emb.weight.data_ptr(), (
        "ema_embed_sum shares memory with the codebook weight — register a .clone() of it")
    e_from_state = M.to(F64) / _smoothed(N.to(F64), 1e-5)[:, None]     # built with eps = 1e-5
    assert torch.isfinite(e_from_state).all() and torch.allclose(
            e_from_state, emb.weight.detach().to(F64), rtol=1e-4, atol=1e-6), (
        "a fresh module's EMA state must be consistent with its codebook, i.e. "
        "ema_embed_sum / smoothed(ema_cluster_size) == embedding.weight — otherwise the first "
        "update replaces every unused code by garbage (with N = 0 its m_k / N~_k is divided by "
        "~eps). Initialise ema_cluster_size to ones and ema_embed_sum to a clone of the weight")


def check_ema_forward_quantizes_with_current_codebook(ns):
    beta = 0.4
    m = _ema(ns, 7, 3, beta=beta, decay=0.5, eps=1e-3)
    _buffers(m)
    e, z, want = _planted(seed=7, noise=0.6)
    _set_ema_state(m, e, torch.ones(7), e.clone())
    z = z.clone().requires_grad_(True)
    z_q, loss, idx = _unpack(m(z), z, "VectorQuantizerEMA")
    assert torch.equal(idx, want), (
        "indices must be the nearest codes of the codebook as it was BEFORE this call's update")
    zq_ref = _layout(e, idx)
    assert torch.allclose(z_q, zq_ref, atol=1e-5), (
        "z_q must be looked up in the codebook BEFORE the EMA update (quantize, then update), "
        f"max diff {(z_q - zq_ref).abs().max().item():.3e}")
    mse = ((zq_ref - z.detach()) ** 2).mean()
    if not torch.allclose(loss, beta * mse, rtol=1e-5, atol=1e-7):
        msg = (f"loss = {loss.item():.5f}, expected beta * mse(z, sg[z_q]) = "
               f"{(beta * mse).item():.5f}: with an EMA codebook the loss is the commitment term only")
        if torch.allclose(loss, (1 + beta) * mse, rtol=1e-5):
            msg += " (you also added the codebook term)"
        raise AssertionError(msg)
    up = torch.randn(z.shape, generator=torch.Generator().manual_seed(8))
    (loss + (z_q * up).sum()).backward()
    want_g = up + beta * 2 * (z.detach() - zq_ref) / z.numel()
    assert z.grad is not None and torch.allclose(z.grad, want_g, atol=1e-6), (
        "d/dz of (loss + <z_q, g>) must be g + beta * 2 (z - sg[z_q]) / numel: straight-through "
        "for z_q plus the commitment gradient")
    gw = m.embedding.weight.grad
    assert gw is None or torch.all(gw == 0), "no gradient may reach the EMA codebook"


def check_ema_one_update_matches_the_equations(ns):
    decay, eps = 0.8, 0.1      # large eps so Laplace smoothing is visible
    m = _ema(ns, 7, 3, decay=decay, eps=eps)
    _buffers(m)
    e, z, _ = _planted(seed=9, k=6, noise=0.4)            # codes 0..5 can be used; ...
    e = torch.cat([e, torch.tensor([[40.0, 40.0, 40.0]])])  # ... code 6 is far away: unused
    g = torch.Generator().manual_seed(10)
    N0 = torch.tensor([2.0, 0.5, 1.0, 3.0, 0.2, 1.5, 0.0])  # unused code 6 has N = 0
    M0 = torch.randn(7, 3, generator=g)
    _set_ema_state(m, e, N0, M0)
    _unpack(m(z), z, "VectorQuantizerEMA")
    idx, _, e1, N1, M1, n, s = _ema_step_ref(e, N0, M0, z, decay, eps)
    N_got, M_got = (t.detach().to(F64) for t in _buffers(m))
    w_got = m.embedding.weight.detach().to(F64)

    if not torch.allclose(N_got, N1, rtol=1e-5, atol=1e-6):
        if torch.allclose(N_got, _smoothed(N1, eps), rtol=1e-5, atol=1e-6):
            raise AssertionError(
                "ema_cluster_size holds the Laplace-SMOOTHED counts (as Zalando's pytorch-vq-vae "
                "port does). The graded convention, like Sonnet's VectorQuantizerEMA, rosinality's "
                "and lucidrains' ports, keeps the raw EMA N = decay*N + (1-decay)*n in the buffer "
                "and smooths only a local copy for the division; writing it back re-adds the eps "
                "pseudo-count every step, so the buffer drifts from the EMA of the counts")
        if torch.allclose(N_got, (1 - decay) * N0.to(F64) + decay * n, rtol=1e-5, atol=1e-6):
            raise AssertionError("decay and 1 - decay are swapped: N = decay * N + (1 - decay) * n")
        if torch.allclose(N_got, N0.to(F64)):
            raise AssertionError("ema_cluster_size did not change in training mode")
        raise AssertionError(
            f"ema_cluster_size after one step should be decay * N + (1 - decay) * n_batch = "
            f"{[round(v, 4) for v in N1.tolist()]}, got {[round(v, 4) for v in N_got.tolist()]}")
    assert torch.allclose(M_got, M1, rtol=1e-5, atol=1e-5), (
        "ema_embed_sum after one step should be decay * m + (1 - decay) * s, where s_k is the SUM "
        f"of the z vectors assigned to code k (max diff {(M_got - M1).abs().max().item():.3e})")
    if torch.allclose(w_got, e1, rtol=1e-5, atol=1e-5):
        return
    if not torch.isfinite(w_got).all() or torch.allclose(w_got[:6], (M1 / N1[:, None])[:6], rtol=1e-5):
        raise AssertionError(
            "the codebook is m / N without Laplace smoothing — an unused code with N = 0 becomes "
            "inf/NaN. Use N~ = (N + eps) / (sum(N) + K * eps) * sum(N) and e = m / N~")
    if torch.allclose(w_got, M1 / (N1 + eps)[:, None], rtol=1e-5, atol=1e-5):
        raise AssertionError(
            "e = m / (N + eps) is additive smoothing; Laplace smoothing also renormalises so the "
            "total count is preserved: N~ = (N + eps) / (sum(N) + K * eps) * sum(N)")
    if torch.allclose(w_got, e.to(F64)):
        raise AssertionError("the codebook did not change in training mode — write m / N~ into embedding.weight")
    raise AssertionError(
        "codebook after one step should be m / N~ with N~ = (N + eps) / (sum(N) + K*eps) * sum(N) "
        f"(max diff {(w_got - e1).abs().max().item():.3e})")


def check_ema_several_updates(ns):
    decay, eps, K, D = 0.9, 1e-5, 8, 4
    m = _ema(ns, K, D, decay=decay, eps=eps, seed=3)
    _buffers(m)
    g = torch.Generator().manual_seed(11)
    centers = 2.0 * torch.randn(K, D, generator=g)
    e0 = centers + 0.5 * torch.randn(K, D, generator=g)      # a consistent state near the data
    _set_ema_state(m, e0, torch.ones(K), e0.clone())
    e, N, M = e0.to(F64), torch.ones(K, dtype=F64), e0.to(F64)
    for step in range(5):
        lab = torch.randint(0, K, (2, 5, 5), generator=g)
        z = _layout(centers, lab) + 0.5 * torch.randn(2, D, 5, 5, generator=g)
        _, _, idx = _unpack(m(z), z, "VectorQuantizerEMA")
        want_idx, _, e, N, M, _, _ = _ema_step_ref(e, N, M, z, decay, eps)
        assert torch.equal(idx, want_idx), f"step {step}: indices differ from the reference"
        w = m.embedding.weight.detach().to(F64)
        assert torch.allclose(w, e, rtol=1e-4, atol=1e-5), (
            f"after training step {step + 1} the codebook differs from the EMA equations "
            f"(max diff {(w - e).abs().max().item():.3e})")
    Ng, Mg = (t.detach().to(F64) for t in _buffers(m))
    assert torch.allclose(Ng, N, rtol=1e-4, atol=1e-6) and torch.allclose(Mg, M, rtol=1e-4, atol=1e-5), \
        "after 5 steps ema_cluster_size / ema_embed_sum differ from the EMA equations"


def check_ema_eval_mode_does_not_update(ns):
    m = _ema(ns, 7, 3, decay=0.5)
    _buffers(m)
    e, z, want = _planted(seed=12)
    _set_ema_state(m, e, torch.ones(7), e.clone())
    m.eval()
    before = [t.detach().clone() for t in (m.embedding.weight, m.ema_cluster_size, m.ema_embed_sum)]
    with torch.no_grad():
        z_q, _, idx = _unpack(m(z), z, "VectorQuantizerEMA")
    after = [m.embedding.weight, m.ema_cluster_size, m.ema_embed_sum]
    for name, b, a in zip(("embedding.weight", "ema_cluster_size", "ema_embed_sum"), before, after):
        assert torch.equal(b, a.detach()), (
            f"{name} changed in eval mode — the EMA update must run only when self.training is True")
    assert torch.equal(idx, want) and torch.allclose(z_q, _layout(e, idx), atol=1e-6), \
        "eval-mode forward must still quantize"


def check_ema_state_has_no_autograd_history(ns):
    m = _ema(ns, 7, 3)
    _buffers(m)
    e, z, _ = _planted(seed=13)
    _set_ema_state(m, e, torch.ones(7), e.clone())
    z = z.clone().requires_grad_(True)
    m(z)
    for name, t in (("ema_cluster_size", m.ema_cluster_size), ("ema_embed_sum", m.ema_embed_sum),
                    ("embedding.weight", m.embedding.weight)):
        assert t.grad_fn is None and not t.requires_grad, (
            f"{name} carries autograd history after a training step — compute the EMA statistics "
            f"under torch.no_grad() from z.detach(), or every step's graph stays alive")


def check_ema_with_zero_decay_is_a_kmeans_step(ns):
    """decay = 0 and eps = 0: every used code jumps to the mean of its assigned vectors."""
    m = _ema(ns, 7, 3, decay=0.0, eps=0.0)
    _buffers(m)
    e, z, _ = _planted(seed=14, b=4, noise=0.8)
    _set_ema_state(m, e, torch.ones(7), e.clone())
    _, _, idx = _unpack(m(z), z, "VectorQuantizerEMA")
    flat = z.permute(0, 2, 3, 1).reshape(-1, 3)
    w = m.embedding.weight.detach()
    for k in idx.unique().tolist():
        mean_k = flat[idx.flatten() == k].mean(0)
        assert torch.allclose(w[k], mean_k, atol=1e-5), (
            f"with decay = 0 and eps = 0 the EMA update is one k-means (Lloyd) step: code {k} "
            f"should move to the mean of its assigned vectors {mean_k.tolist()}, got {w[k].tolist()}")


# --------------------------------------------------------------------------
# perplexity
# --------------------------------------------------------------------------
def _ref_perplexity(idx, K):
    p = torch.bincount(idx.flatten(), minlength=K).to(F64)
    p = p / p.sum()
    return torch.exp(-torch.special.xlogy(p, p).sum())


def _perp(ns, idx, K):
    out = ns.codebook_perplexity(idx, K)
    assert torch.is_tensor(out) and out.numel() == 1, \
        f"codebook_perplexity must return a scalar tensor, got {type(out).__name__}"
    return float(out)


def check_perplexity_values(ns):
    K = 8
    uni = torch.arange(K).repeat(6).reshape(3, 4, 4)
    got = _perp(ns, uni, K)
    assert abs(got - K) < 1e-4, f"uniform use of all {K} codes must give perplexity {K}, got {got:.5f}"
    one = torch.full((2, 3, 3), 5)
    got = _perp(ns, one, K)
    assert got == got, (
        "perplexity is NaN when some codes are unused — define 0 * log 0 = 0 "
        "(torch.special.xlogy, or mask out zero probabilities)")
    assert abs(got - 1.0) < 1e-4, f"a single code in use must give perplexity 1, got {got:.5f}"
    idx = torch.tensor([2, 2, 2, 5]).reshape(1, 2, 2)
    want = float(_ref_perplexity(idx, K))        # exp(H(0.75, 0.25)) = 1.7548
    got = _perp(ns, idx, K)
    if abs(got - want) > 1e-4:
        H = -(0.75 * torch.log(torch.tensor(0.75)) + 0.25 * torch.log(torch.tensor(0.25))).item()
        for why, v in {"the entropy itself — perplexity is exp(entropy)": H,
                       "exp of the base-2 entropy (use natural log with exp, or log2 with 2**)":
                           float(torch.exp(torch.tensor(H / 0.6931471805599453))),
                       "2 ** (entropy in nats)": 2 ** H}.items():
            if abs(got - v) < 1e-4:
                raise AssertionError(f"perplexity {got:.4f} is {why}; expected {want:.4f}")
        raise AssertionError(
            f"codes used with frequencies (0.75, 0.25) must give exp(H) = {want:.4f}, got {got:.4f}")
    g = torch.Generator().manual_seed(15)
    idx = torch.randint(0, 11, (4, 6, 6), generator=g)
    got, want = _perp(ns, idx, 16), float(_ref_perplexity(idx, 16))
    assert abs(got - want) < 1e-4 * want, (
        f"perplexity of random usage over 11 of 16 codes should be {want:.4f}, got {got:.4f} "
        f"(p_k = count_k / total, exp(-sum p log p) with 0 log 0 = 0)")


CHECKS = [
    check_vq_codebook_and_output_shapes,
    check_vq_indices_on_planted_codes,
    check_vq_indices_match_brute_force,
    check_vq_zq_is_the_selected_code,
    check_vq_straight_through_gradient,
    check_vq_loss_value,
    check_vq_loss_gradients_are_isolated,
    check_ema_buffers_and_frozen_codebook,
    check_ema_forward_quantizes_with_current_codebook,
    check_ema_one_update_matches_the_equations,
    check_ema_several_updates,
    check_ema_eval_mode_does_not_update,
    check_ema_state_has_no_autograd_history,
    check_ema_with_zero_decay_is_a_kmeans_step,
    check_perplexity_values,
]
