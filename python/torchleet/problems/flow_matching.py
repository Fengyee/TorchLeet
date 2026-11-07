"""flow-matching — conditional flow matching / rectified flow with the linear path.

Convention (the notebook states it): x0 ~ N(0, I) is noise at t = 0, x1 is data at
t = 1, x_t = (1 - t) * x0 + t * x1, the target velocity is u = x1 - x0 and the loss
is mean((v(x_t, t) - u)^2). Note SD3's paper writes the same path with the time
axis reversed (t = 0 data, t = 1 noise); only the convention above is graded.

How each entry is pinned without a reference implementation:

* interpolate: endpoints, an explicit midpoint, and per-sample broadcasting on a
  square (N == D) batch, where broadcasting t against the last dim instead of the
  first would silently give a wrong answer, plus an image-shaped batch.
* flow_matching_loss: with x0 and t passed in, equal to a loss written in the spec;
  an oracle that returns x1 - x0 gets exactly zero. With x0 and t sampled inside,
  a recording model sees (x_t, t), from which the spec recovers x0 = (x_t - t x1) /
  (1 - t) and checks it is standard normal and independent of x1, that t is
  per-sample U(0, 1), and that the oracle (x1 - x_t) / (1 - t) gets zero loss.
* euler_sample: exact for a constant field, equals x0 * (1 - 1/n)^n (the Euler
  product, not e^-1) for v = -x, calls the model exactly at t_k = k/n with t of
  shape (N,), and does not modify x0 in place. A short training run (spec MLP,
  student loss and sampler) must move samples onto a shifted, anisotropic Gaussian.
"""
import math

import torch
import torch.nn as nn

ENTRIES = ["interpolate", "flow_matching_loss", "euler_sample"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Everything follows from one straight line per pair: x_t = (1 - t) * x0 + t * x1 "
    "moves from noise (t = 0) to data (t = 1) at constant speed, so its time "
    "derivative — the regression target — is x1 - x0, independent of t.",
    "t has shape (N,) but x0 may be (N, D) or (N, C, H, W): reshape t to "
    "(N, 1, ..., 1) before mixing. When x0 / t are None, draw x0 with "
    "torch.randn_like(x1) and t with torch.rand(N) (one time per sample), then "
    "return the mean squared error between model(x_t, t) and x1 - x0.",
    "Euler: dt = 1 / num_steps; for k in 0 .. num_steps-1, call the model at "
    "t = k * dt (a (N,) tensor filled with that value) and update "
    "x = x + dt * v. Do not update x0 in place and do not call the model at t = 1.",
]

N, DIM = 64, 3


class _one_thread:
    """Tiny models train ~10x faster on one thread than on an oversubscribed pool."""

    def __enter__(self):
        self.n = torch.get_num_threads()
        torch.set_num_threads(1)

    def __exit__(self, *exc):
        torch.set_num_threads(self.n)


def _bt(t, x):
    """Reshape a per-sample t to broadcast over x's trailing dims."""
    return torch.as_tensor(t, dtype=x.dtype).reshape(-1, *([1] * (x.dim() - 1)))


class _ToyVelocity(nn.Module):
    """A fixed nonlinear velocity field that depends on both x and t."""

    def __init__(self, d=DIM):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.W = torch.randn(d, d, generator=g) / math.sqrt(d)
        self.b = torch.randn(d, generator=g)

    def forward(self, x, t):
        tt = _bt(t, x)
        return torch.tanh(x @ self.W.to(x.dtype)) * (1 + tt) + tt * self.b.to(x.dtype)


class _Recorder(nn.Module):
    def __init__(self, fn=None):
        super().__init__()
        self.fn = fn
        self.calls = []

    def forward(self, x, t):
        self.calls.append((x.detach().clone(), t.detach().clone() if torch.is_tensor(t) else t))
        return self.fn(x, t) if self.fn is not None else torch.zeros_like(x)


class _MLP(nn.Module):
    def __init__(self, d=2, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + 1, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, d))

    def forward(self, x, t):
        t = torch.as_tensor(t, dtype=x.dtype).reshape(-1, 1).expand(x.shape[0], 1)
        return self.net(torch.cat([x, t], dim=-1))


def _scalar(v, what="flow_matching_loss"):
    assert torch.is_tensor(v) and v.numel() == 1, (
        f"{what} must return a scalar tensor, got "
        f"{type(v).__name__}{'' if not torch.is_tensor(v) else tuple(v.shape)}")
    return v.reshape(())


# --------------------------------------------------------------------------
# interpolate
# --------------------------------------------------------------------------
def _interp(ns, x0, x1, t):
    try:
        return ns.interpolate(x0, x1, t)
    except RuntimeError as e:
        raise AssertionError(
            f"interpolate failed for x0 of shape {tuple(x0.shape)} and t of shape "
            f"{tuple(t.shape)} ({e}) — t holds one time per sample; reshape it to "
            f"(N, 1, ..., 1) so it broadcasts over the trailing dims")


def check_interpolate_endpoints_and_midpoints(ns):
    g = torch.Generator().manual_seed(0)
    x0, x1 = torch.randn(6, 5, generator=g), torch.randn(6, 5, generator=g)
    a = _interp(ns, x0, x1, torch.zeros(6))
    b = _interp(ns, x0, x1, torch.ones(6))
    assert tuple(a.shape) == (6, 5), f"interpolate must keep x0's shape (6, 5), got {tuple(a.shape)}"
    if torch.allclose(a, x1, atol=1e-6) and torch.allclose(b, x0, atol=1e-6):
        raise AssertionError(
            "the path is reversed: with this convention x_t = (1 - t) * x0 + t * x1, so "
            "t = 0 gives the noise x0 and t = 1 gives the data x1")
    assert torch.allclose(a, x0, atol=1e-6), "interpolate(x0, x1, t=0) must return x0 (noise end)"
    assert torch.allclose(b, x1, atol=1e-6), "interpolate(x0, x1, t=1) must return x1 (data end)"
    t = torch.tensor([0.1, 0.25, 0.5, 0.6, 0.75, 0.9])
    got = _interp(ns, x0, x1, t)
    want = (1 - t)[:, None] * x0 + t[:, None] * x1
    assert torch.allclose(got, want, atol=1e-6), (
        f"x_t must be (1 - t) * x0 + t * x1 with row i using t[i] "
        f"(max diff {(got - want).abs().max():.3e})")


def check_interpolate_broadcasts_t_per_sample(ns):
    g = torch.Generator().manual_seed(1)
    # Square batch: broadcasting t (N,) against the LAST dim runs without error here
    # but mixes column j with t[j] instead of row i with t[i].
    x0, x1 = torch.randn(4, 4, generator=g), torch.randn(4, 4, generator=g)
    t = torch.tensor([0.0, 0.3, 0.7, 1.0])
    got = _interp(ns, x0, x1, t)
    want = (1 - t)[:, None] * x0 + t[:, None] * x1
    assert not torch.allclose(got, t[:, None] * x0 + (1 - t)[:, None] * x1, atol=1e-6), (
        "the path is reversed: x_t = (1 - t) * x0 + t * x1 (t = 0 is noise, t = 1 is data)")
    assert torch.allclose(got, want, atol=1e-6), (
        "with a (4, 4) batch, row i must be mixed with t[i]. A bare (1 - t) * x0 "
        "broadcasts t along the last dim (column j gets t[j]) — reshape t to (N, 1, ...)")
    imgs0, imgs1 = torch.randn(4, 3, 5, 5, generator=g), torch.randn(4, 3, 5, 5, generator=g)
    got = _interp(ns, imgs0, imgs1, t)
    want = (1 - t).view(4, 1, 1, 1) * imgs0 + t.view(4, 1, 1, 1) * imgs1
    assert tuple(got.shape) == (4, 3, 5, 5), \
        f"for (4, 3, 5, 5) images x_t must keep that shape, got {tuple(got.shape)}"
    assert torch.allclose(got, want, atol=1e-6), \
        "image batch: t must broadcast over (C, H, W) per sample — reshape t to (N, 1, 1, 1)"


# --------------------------------------------------------------------------
# flow_matching_loss
# --------------------------------------------------------------------------
def _fm_inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    x1 = 2.0 * torch.randn(N, DIM, generator=g) + 1.0
    x0 = torch.randn(N, DIM, generator=g)
    t = torch.rand(N, generator=g)
    return x0, x1, t


def check_loss_matches_reference_for_given_x0_and_t(ns):
    x0, x1, t = _fm_inputs()
    m = _ToyVelocity()
    got = _scalar(ns.flow_matching_loss(m, x1, x0=x0, t=t))
    xt = (1 - t)[:, None] * x0 + t[:, None] * x1
    v = m(xt, t)
    want = ((v - (x1 - x0)) ** 2).mean()
    if torch.allclose(got, want, rtol=1e-5, atol=1e-6):
        return
    xt_rev = t[:, None] * x0 + (1 - t)[:, None] * x1
    alts = {
        "the target sign is flipped — u = d x_t / dt = x1 - x0, not x0 - x1":
            ((v - (x0 - x1)) ** 2).mean(),
        "the target is x1 - x0, not the data x1 itself": ((v - x1) ** 2).mean(),
        "the target is x1 - x0, not the noise x0": ((v - x0) ** 2).mean(),
        "the path is reversed — x_t = (1 - t) * x0 + t * x1 (t = 0 is noise)":
            ((m(xt_rev, t) - (x1 - x0)) ** 2).mean(),
        "the model must be called with t, not 1 - t": ((m(xt, 1 - t) - (x1 - x0)) ** 2).mean(),
        "the loss is summed over dims — average over every element (F.mse_loss)":
            ((v - (x1 - x0)) ** 2).sum(-1).mean(),
        "the loss is summed — average over every element (F.mse_loss)": ((v - (x1 - x0)) ** 2).sum(),
    }
    for why, a in alts.items():
        if torch.allclose(got, a, rtol=1e-5, atol=1e-6):
            raise AssertionError(f"loss = {float(got):.5f}, expected {float(want):.5f}: {why}")
    raise AssertionError(
        f"loss = {float(got):.5f}, expected mean((model(x_t, t) - (x1 - x0))^2) = {float(want):.5f}")


def check_oracle_velocity_gets_zero_loss(ns):
    x0, x1, t = _fm_inputs(seed=1)
    oracle = lambda x, tt: (x1 - x0).clone()           # noqa: E731 — knows the pairing
    got = _scalar(ns.flow_matching_loss(oracle, x1, x0=x0, t=t))
    assert float(got) < 1e-10, (
        f"a model that returns exactly x1 - x0 must get zero loss, got {float(got):.3e} — "
        f"the regression target is the constant velocity x1 - x0 of the straight path")

    # Sampled x0 and t: the conditional velocity can be read back from (x_t, t)
    # alone, u = (x1 - x_t) / (1 - t). Zero loss iff x_t is built from these x1 rows.
    g = torch.Generator().manual_seed(2)
    y1 = torch.randn(1000, 2, generator=g, dtype=torch.float64) * 1.5 + 0.5

    def read_back(x, tt):
        tt = torch.as_tensor(tt, dtype=x.dtype).reshape(-1, 1)
        return (y1 - x) / (1 - tt)

    torch.manual_seed(0)
    got = _scalar(ns.flow_matching_loss(read_back, y1))
    assert float(got) < 1e-6, (
        f"with x0 and t sampled inside the loss, the oracle (x1 - x_t) / (1 - t) should get "
        f"~0 loss but got {float(got):.3e}: x_t, t and the target must all use the same "
        f"per-sample (x0, t) draw and the given x1 rows")


def check_loss_samples_standard_normal_noise_and_uniform_time(ns):
    g = torch.Generator().manual_seed(3)
    n = 4000
    x1 = (torch.randn(n, 2, generator=g) * 2.0 + 3.0).double()
    rec = _Recorder()
    torch.manual_seed(1)
    _scalar(ns.flow_matching_loss(rec, x1))
    assert len(rec.calls) == 1, f"the model should be called once per loss, got {len(rec.calls)} calls"
    xt, t = rec.calls[0]
    assert torch.is_tensor(t) and t.numel() == n, (
        f"the model must receive one time per sample (a ({n},) tensor), got "
        f"{tuple(t.shape) if torch.is_tensor(t) else type(t).__name__}")
    t = t.reshape(-1).double()
    assert float(t.min()) >= 0 and float(t.max()) <= 1, \
        f"t must lie in [0, 1], got range [{float(t.min()):.3f}, {float(t.max()):.3f}]"
    assert t.unique().numel() > n // 2, "t must be drawn independently for every sample, not shared"
    assert abs(float(t.mean()) - 0.5) < 0.03 and abs(float(t.std()) - 1 / math.sqrt(12)) < 0.03, (
        f"t should be U(0, 1) (mean 0.5, std 0.289), got mean {float(t.mean()):.3f}, "
        f"std {float(t.std()):.3f}")
    keep = t < 0.9
    x0 = (xt.double()[keep] - t[keep, None] * x1[keep]) / (1 - t[keep, None])
    mean, std = float(x0.mean()), float(x0.std())
    assert abs(mean) < 0.06 and abs(std - 1) < 0.06, (
        f"recovered x0 = (x_t - t x1) / (1 - t) has mean {mean:.3f}, std {std:.3f}; it "
        f"should be standard normal noise — use torch.randn_like(x1), not torch.rand_like")
    # ~3600 kept rows: the sample correlation of independent draws has std ~0.017, so
    # 0.1 is ~6 sigma (0.06 was 3.6 sigma: ~1 in 2000 correct implementations failed).
    c = torch.corrcoef(torch.stack([x0[:, 0], x1[keep, 0]]))[0, 1]
    assert abs(float(c)) < 0.1, \
        f"x0 must be fresh noise independent of x1, but corr(x0, x1) = {float(c):.3f}"


def check_loss_is_differentiable(ns):
    torch.manual_seed(0)
    m = _MLP(d=DIM, h=16)
    x0, x1, t = _fm_inputs(seed=4)
    loss = _scalar(ns.flow_matching_loss(m, x1, x0=x0, t=t))
    assert loss.requires_grad, "the loss does not require grad — keep it in the autograd graph"
    loss.backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None or torch.all(p.grad == 0)]
    assert not dead, f"no gradient reached the model parameters {dead}"


# --------------------------------------------------------------------------
# euler_sample
# --------------------------------------------------------------------------
def check_euler_is_exact_for_a_constant_field(ns):
    c = torch.tensor([0.5, -2.0, 3.0])
    g = torch.Generator().manual_seed(5)
    x0 = torch.randn(10, 3, generator=g)
    keep = x0.clone()
    for n in (1, 3, 10):
        out = ns.euler_sample(lambda x, t: c.expand_as(x), x0, n)
        assert torch.equal(x0, keep), \
            "euler_sample modified x0 in place — start from x = x0.clone() or use x = x + dt * v"
        assert tuple(out.shape) == (10, 3), f"output must have x0's shape (10, 3), got {tuple(out.shape)}"
        assert torch.allclose(out, x0 + c, atol=1e-5), (
            f"for a constant velocity c, integrating from t = 0 to 1 must give x0 + c exactly "
            f"(num_steps = {n}); max diff {(out - (x0 + c)).abs().max():.3e} — check dt = 1 / num_steps "
            f"and that the model is called exactly num_steps times")


def check_euler_matches_the_euler_product_for_v_eq_minus_x(ns):
    g = torch.Generator().manual_seed(6)
    x0 = torch.randn(8, 2, generator=g, dtype=torch.float64)
    for n in (1, 2, 5, 50):
        out = ns.euler_sample(lambda x, t: -x, x0, n)
        want = x0 * (1 - 1 / n) ** n
        if torch.allclose(out, want, rtol=1e-6, atol=1e-8):
            continue
        if torch.allclose(out, x0 * math.exp(-1), rtol=1e-4):
            raise AssertionError(
                "the result equals the exact ODE solution x0 * e^-1 — but explicit Euler with "
                f"{n} steps gives x0 * (1 - 1/n)^n; implement the Euler update, not a closed form")
        ratio = float((out / x0).mean())
        raise AssertionError(
            f"for v = -x, {n} Euler steps of size 1/{n} give x0 * (1 - 1/{n})^{n} = "
            f"x0 * {(1 - 1 / n) ** n:.5f}; got x0 * {ratio:.5f}")


def check_euler_uses_the_left_endpoint_time_grid(ns):
    g = torch.Generator().manual_seed(7)
    x0 = torch.randn(5, 2, generator=g)
    n = 4
    rec = _Recorder()
    ns.euler_sample(rec, x0, n)
    assert len(rec.calls) == n, f"num_steps = {n} must call the model exactly {n} times, got {len(rec.calls)}"
    times = []
    for _, t in rec.calls:
        assert torch.is_tensor(t) and tuple(t.shape) == (5,), (
            f"pass t to the model as a (N,) tensor like in training, got "
            f"{tuple(t.shape) if torch.is_tensor(t) else type(t).__name__}")
        assert torch.allclose(t, t[0].expand_as(t)), "all samples share the same t at a given step"
        times.append(float(t[0]))
    want = [k / n for k in range(n)]
    if all(abs(a - b) < 1e-6 for a, b in zip(times, [(k + 1) / n for k in range(n)])):
        raise AssertionError(
            f"the model was called at {times}: those are right endpoints. Explicit Euler "
            f"evaluates v at the start of each step, t_k = k / n = {want}")
    if not all(abs(a - b) < 1e-6 for a, b in zip(times, want)):
        lin = torch.linspace(0, 1, n).tolist()
        why = (" — torch.linspace(0, 1, n) includes t = 1 and has spacing 1/(n-1)"
               if all(abs(a - b) < 1e-6 for a, b in zip(times, lin)) else "")
        raise AssertionError(
            f"the model must be called at t_k = k / num_steps = {want}, got "
            f"{[round(x, 4) for x in times]}{why}")
    # dx/dt = t has exact solution x0 + 1/2; left-point Euler gives x0 + (n - 1) / (2n).
    out = ns.euler_sample(lambda x, t: _bt(t, x).expand_as(x).clone(), x0, n)
    assert torch.allclose(out, x0 + (n - 1) / (2 * n), atol=1e-5), \
        f"for v = t the {n}-step Euler result is x0 + {(n - 1) / (2 * n)}, got a different value"


def check_training_moves_samples_onto_the_data(ns):
    with _one_thread():
        _train_and_sample(ns)


def _train_and_sample(ns):
    target_mean = torch.tensor([2.0, -1.0])
    target_std = torch.tensor([0.5, 0.25])
    torch.manual_seed(0)
    model = _MLP(d=2, h=64)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(300):
        x1 = target_mean + target_std * torch.randn(256, 2)
        loss = _scalar(ns.flow_matching_loss(model, x1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.manual_seed(1)
    x0 = torch.randn(2000, 2)
    with torch.no_grad():
        xs = ns.euler_sample(model, x0, 20)
    mean, std = xs.mean(0), xs.std(0)
    assert torch.isfinite(xs).all(), "samples contain NaN/Inf"
    # Correct implementations differ in how they consume the RNG; over ~650 such runs the
    # worst errors were 0.144 (mean) and 0.099 (std), so these bounds leave real headroom
    # while a wrong noise scale or target is still off by > 0.3.
    assert torch.allclose(mean, target_mean, atol=0.25) and torch.allclose(std, target_std, atol=0.16), (
        f"after training on N(mean={target_mean.tolist()}, std={target_std.tolist()}), 20 Euler "
        f"steps from N(0, I) gave mean {[round(v, 3) for v in mean.tolist()]}, std "
        f"{[round(v, 3) for v in std.tolist()]} — the loss or the sampler is off")


CHECKS = [
    check_interpolate_endpoints_and_midpoints,
    check_interpolate_broadcasts_t_per_sample,
    check_loss_matches_reference_for_given_x0_and_t,
    check_oracle_velocity_gets_zero_loss,
    check_loss_samples_standard_normal_noise_and_uniform_time,
    check_loss_is_differentiable,
    check_euler_is_exact_for_a_constant_field,
    check_euler_matches_the_euler_product_for_v_eq_minus_x,
    check_euler_uses_the_left_endpoint_time_grid,
    check_training_moves_samples_onto_the_data,
]
