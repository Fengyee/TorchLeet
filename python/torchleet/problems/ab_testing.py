"""ab-testing — two-proportion z-test, Wald CI, bootstrap CI, sample size, SRM and CUPED.

Conventions graded here (arm A = control, arm B = treatment; diff = p_B - p_A):
  * two_proportion_ztest: z = diff / SE0 with the POOLED standard error under H0,
    SE0 = sqrt(p(1-p)(1/n_A + 1/n_B)), p = (x_A + x_B)/(n_A + n_B); two-sided
    p-value 2 * (1 - Phi(|z|)) = erfc(|z| / sqrt 2). SE0 = 0 (no or all successes)
    gives z = 0, p = 1. Identical to statsmodels.stats.proportion.proportions_ztest
    (pooled by default). Inputs broadcast elementwise; float64 outputs.
  * diff_ci: Wald interval diff -/+ z_{1-alpha/2} * SE1 with the UNPOOLED
    SE1 = sqrt(p_A(1-p_A)/n_A + p_B(1-p_B)/n_B) — statsmodels
    confint_proportions_2indep(method="wald"). The test and the CI can disagree near
    the boundary because they use different SEs (the notebook shows a case).
  * bootstrap_ci: percentile bootstrap (Efron & Tibshirani 1993, ch. 13) for
    mean(b) - mean(a): resample EACH arm independently with replacement (its own n),
    take the alpha/2 and 1 - alpha/2 quantiles of the n_boot differences. Graded
    statistically, so any RNG order, quantile interpolation or the Poisson /
    multinomial variants pass: deterministic for a seeded generator; width close to
    2 z SE with the plug-in SE sqrt(var_a/n_a + var_b/n_b) (1/n variances); centred
    on the point estimate (resampling the POOLED data centres it on 0); invariant to
    the order of units inside an arm (index-paired resampling of two sorted arms
    collapses the width); asymmetric on right-skewed data the way the percentile
    method is (a symmetric "normal" or reflected "basic" bootstrap fails); empirical
    coverage of 250 simulated experiments in [0.85, 0.995]; the number of torch
    calls must not grow with n_boot (no Python loop over resamples).
  * sample_size_per_arm: two-sided two-proportion test, equal allocation,
        n = ceil( (z_{1-a/2} sqrt(2 pbar(1-pbar)) + z_{power} sqrt(p1 q1 + p2 q2))^2 / d^2 )
    with p1 = p_base, p2 = p_base + mde_abs, pbar = (p1 + p2)/2: pooled variance under
    H0, unpooled under H1 (Fleiss, Levin & Paik, "Statistical Methods for Rates and
    Proportions", 3rd ed., the formula without continuity correction; the n that
    R's power.prop.test solves for; statsmodels
    samplesize_proportions_2indep_onetail(..., alternative="two-sided")). Graded
    exactly (the test cases' fractional parts are far from integers).
  * srm_pvalue: Pearson chi-square goodness of fit with 1 dof, no continuity
    correction; expected_ratio = expected share of arm A; p = erfc(sqrt(chi2 / 2))
    (= scipy.stats.chisquare(...).pvalue).
  * cuped (Deng, Xu, Kohavi & Walker, WSDM 2013): theta = cov(y, x) / var(x) with the
    same normalization in both; y_adj = y - theta (x - mean(x)); var(x) = 0 -> theta = 0.
    In-sample identities checked exactly: mean(y_adj) = mean(y) and
    var(y_adj) = var(y) (1 - rho^2).

Oracles are written out in float64 below; they were cross-checked against scipy and
statsmodels when the problem was written.
"""
import math

import torch

from torchleet.runner import Skip

ENTRIES = ["two_proportion_ztest", "diff_ci", "bootstrap_ci", "sample_size_per_arm",
           "srm_pvalue", "cuped"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Every function here is a few lines of float64 arithmetic. The standard normal CDF is "
    "torch.special.ndtr, its inverse (z_{1-alpha/2}) is torch.special.ndtri, and two-sided tail "
    "probabilities are torch.special.erfc(|z| / sqrt(2)). A chi-square with 1 degree of freedom is "
    "the square of a standard normal. Convert inputs with torch.as_tensor(x, dtype=torch.float64): "
    "int tensors divide into float32 otherwise.",
    "The test asks 'how surprising is this diff if p_A == p_B?', so its SE uses the POOLED rate "
    "(x_A + x_B) / (n_A + n_B). The CI is built around the observed rates, so it uses the "
    "UNPOOLED SE. Bootstrap: draw an (n_boot, n_a) index matrix for a and a separate (n_boot, n_b) "
    "one for b with torch.randint(..., generator=generator), gather, take the row means, then "
    "torch.quantile at alpha/2 and 1 - alpha/2. CUPED's theta is the OLS slope of y on x.",
    "Sample size: the test rejects when |diff| > z_{1-alpha/2} * SE0. Require that to happen with "
    "probability `power` when the true diff is d: sqrt(n) * |d| = z_{1-alpha/2} * sqrt(2 pbar "
    "(1 - pbar)) + z_power * sqrt(p1 q1 + p2 q2). Solve for n and take math.ceil. SRM: expected "
    "counts are N * r and N * (1 - r), chi2 = sum (obs - exp)^2 / exp, and p = erfc(sqrt(chi2 / 2)). "
    "CUPED: theta = mean(xc * yc) / mean(xc * xc) with xc = x - mean(x); return y - theta * xc.",
]

F64 = torch.float64
SQRT2 = math.sqrt(2.0)


# ----------------------------------------------------------------- helpers

def _f(v):
    return torch.as_tensor(v).detach().to(F64)


def _tuple(out, n, name):
    if torch.is_tensor(out) and out.dim() >= 1 and out.shape[0] == n:
        out = list(out.unbind(0))       # e.g. torch.quantile(d, q) -> tensor([lo, hi]): unpacks fine
    assert isinstance(out, (tuple, list)) and len(out) == n, (
        f"{name} should return a tuple of {n} values, got "
        f"{type(out).__name__}{'' if not isinstance(out, (tuple, list)) else f' of length {len(out)}'}")
    return [_f(o) for o in out]


def _z(q):
    return float(torch.special.ndtri(torch.tensor(q, dtype=F64)))


def _ztest_ref(xa, na, xb, nb):
    xa, na, xb, nb = (_f(v) for v in (xa, na, xb, nb))
    pa, pb = xa / na, xb / nb
    diff = pb - pa
    pp = (xa + xb) / (na + nb)
    se0 = torch.sqrt(pp * (1 - pp) * (1 / na + 1 / nb))
    se1 = torch.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    safe = torch.where(se0 > 0, se0, torch.ones_like(se0))
    z = torch.where(se0 > 0, diff / safe, torch.zeros_like(diff))
    p = torch.special.erfc(z.abs() / SQRT2)
    return diff, z, p, se0, se1


def _close(got, want, what, rtol=1e-9, atol=1e-12):
    got, want = _f(got), _f(want)
    assert tuple(got.shape) == tuple(want.shape) or got.numel() == want.numel(), (
        f"{what}: shape {tuple(got.shape)}, expected {tuple(want.shape)}")
    got = got.reshape(want.shape)
    assert torch.isfinite(got).all(), f"{what}: contains NaN/Inf ({got.flatten()[:4].tolist()})"
    ok = torch.allclose(got, want, rtol=rtol, atol=atol)
    return ok, (got - want).abs().max().item()


ZCASES = [  # (x_A, n_A, x_B, n_B) — unequal n and clearly different rates: pooled != unpooled
    (500, 10_000, 560, 10_000),
    (30, 400, 61, 1300),
    (5_100, 100_000, 5_350, 100_500),
    (12, 3000, 40, 2500),
    (420, 600, 300, 520),
]


# ------------------------------------------------------------------ z-test / CI

def check_ztest_matches_pooled_closed_form(ns):
    for xa, na, xb, nb in ZCASES:
        out = _tuple(ns.two_proportion_ztest(xa, na, xb, nb), 3, "two_proportion_ztest")
        diff, z, p = out
        rd, rz, rp, se0, se1 = _ztest_ref(xa, na, xb, nb)
        label = f"x_A={xa}, n_A={na}, x_B={xb}, n_B={nb}"
        ok, err = _close(diff, rd, label)
        if not ok:
            if abs(float(diff) + float(rd)) < 1e-9:
                raise AssertionError(f"{label}: diff = {float(diff):.6g}, expected p_B - p_A = "
                                     f"{float(rd):.6g} — B is the treatment, diff is B minus A")
            raise AssertionError(
                f"{label}: diff = {float(diff):.12g}, expected p_B - p_A = {float(rd):.12g} "
                f"(off by {err:.1e}) — compute in float64: torch.as_tensor(x, dtype=torch.float64); "
                f"dividing int tensors gives float32")
        ok, err = _close(z, rz, label)
        if not ok:
            msg = f"{label}: z = {float(z):.8f}, expected {float(rz):.8f}."
            if abs(float(z) - float(rd / se1)) < 1e-6 * max(1.0, abs(float(z))):
                msg += (" You used the UNPOOLED SE — the test statistic uses the pooled rate "
                        "p = (x_A + x_B) / (n_A + n_B) because under H0 both arms share one p.")
            elif abs(float(z) + float(rz)) < 1e-6 * max(1.0, abs(float(z))):
                msg += " The sign is flipped: diff and z are B minus A."
            else:
                msg += " z = (p_B - p_A) / sqrt(p(1-p)(1/n_A + 1/n_B)) with the pooled p, in float64."
            raise AssertionError(msg)
        ok, err = _close(p, rp, label, rtol=1e-6, atol=1e-13)
        if not ok:
            pf, rpf = float(p), float(rp)
            msg = f"{label}: p-value {pf:.6g}, expected the two-sided {rpf:.6g}."
            if abs(pf - rpf / 2) < 1e-6 * rpf or abs(pf - float(torch.special.ndtr(-rz))) < 1e-9:
                msg += " That is a ONE-sided p-value — use 2 * (1 - Phi(|z|)) = erfc(|z| / sqrt(2))."
            elif abs(pf - (1 - rpf / 2)) < 1e-6:
                msg += " That is Phi(|z|) — the p-value is the TAIL probability 2 * (1 - Phi(|z|))."
            raise AssertionError(msg)


def check_ztest_vectorized_symmetric_and_edge_cases(ns):
    xa = torch.tensor([500, 30, 5100, 12, 420])
    na = torch.tensor([10_000, 400, 100_000, 3000, 600])
    xb = torch.tensor([560, 61, 5350, 40, 300])
    nb = torch.tensor([10_000, 1300, 100_500, 2500, 520])
    out = _tuple(ns.two_proportion_ztest(xa, na, xb, nb), 3, "two_proportion_ztest")
    ref = _ztest_ref(xa, na, xb, nb)[:3]
    for name, got, want, rtol in zip(("diff", "z", "p"), out, ref, (1e-9, 1e-9, 1e-6)):
        assert got.numel() == 5, (
            f"called with length-5 integer tensors, `{name}` should have 5 entries (one per "
            f"experiment), got shape {tuple(got.shape)} — use elementwise tensor ops, no .item()")
        ok, err = _close(got, want, f"vectorized {name}", rtol=rtol, atol=1e-13)
        assert ok, (f"vectorized call on integer tensors: `{name}` is off by {err:.2e}"
                    + (" — convert integer tensors to float64 before dividing (int / int gives "
                       "float32)" if err < 1e-5 else " — see the scalar check above"))
    d1, z1, p1 = out
    d2, z2, p2 = _tuple(ns.two_proportion_ztest(xb, nb, xa, na), 3, "two_proportion_ztest")
    assert torch.allclose(d2.reshape(-1), -d1.reshape(-1), atol=1e-14) and \
        torch.allclose(z2.reshape(-1), -z1.reshape(-1), rtol=1e-12, atol=1e-12), \
        "swapping the arms must negate diff and z"
    assert torch.allclose(p2.reshape(-1), p1.reshape(-1), rtol=1e-12, atol=1e-15), \
        "swapping the arms must leave the two-sided p-value unchanged"
    assert ((p1 >= 0) & (p1 <= 1)).all(), f"p-values must lie in [0, 1], got {p1.tolist()}"
    for xa_, na_, xb_, nb_, what in ((0, 1000, 0, 1200, "no conversions in either arm"),
                                     (800, 800, 500, 500, "every unit converted in both arms"),
                                     (40, 1000, 40, 1000, "identical counts")):
        d, z, p = _tuple(ns.two_proportion_ztest(xa_, na_, xb_, nb_), 3, "two_proportion_ztest")
        assert torch.isfinite(z).all() and torch.isfinite(p).all(), (
            f"{what}: got z = {float(z)}, p = {float(p)} — the pooled SE is 0 here; return "
            f"z = 0 and p = 1 instead of 0/0 = NaN")
        assert abs(float(z)) < 1e-12 and abs(float(p) - 1) < 1e-12, (
            f"{what}: expected z = 0 and p = 1 (no evidence of a difference), got z = {float(z)}, "
            f"p = {float(p)}")


def check_diff_ci_is_the_unpooled_wald_interval(ns):
    for alpha in (0.05, 0.01, 0.2):
        for xa, na, xb, nb in ZCASES:
            lo, hi = _tuple(ns.diff_ci(xa, na, xb, nb, alpha=alpha), 2, "diff_ci")
            rd, _, _, se0, se1 = _ztest_ref(xa, na, xb, nb)
            zq = _z(1 - alpha / 2)
            want_lo, want_hi = rd - zq * se1, rd + zq * se1
            label = f"diff_ci(x_A={xa}, n_A={na}, x_B={xb}, n_B={nb}, alpha={alpha})"
            ok_lo, _ = _close(lo, want_lo, label)
            ok_hi, _ = _close(hi, want_hi, label)
            if ok_lo and ok_hi:
                continue
            half = (float(hi) - float(lo)) / 2
            msg = (f"{label} = ({float(lo):.6f}, {float(hi):.6f}), expected "
                   f"({float(want_lo):.6f}, {float(want_hi):.6f}).")
            if abs(half - zq * float(se0)) < 1e-9:
                msg += (" You used the POOLED SE — the CI is centred on the observed diff, where "
                        "p_A != p_B, so it uses sqrt(p_A(1-p_A)/n_A + p_B(1-p_B)/n_B).")
            elif abs(half - _z(1 - alpha) * float(se1)) < 1e-9:
                msg += f" You used z_(1-alpha) = {_z(1 - alpha):.3f}; a two-sided CI needs z_(1-alpha/2) = {zq:.3f}."
            raise AssertionError(msg)
    lo, hi = _tuple(ns.diff_ci(torch.tensor([30, 500]), torch.tensor([400, 10_000]),
                               torch.tensor([61, 560]), torch.tensor([1300, 10_000])), 2, "diff_ci")
    assert lo.numel() == 2 and hi.numel() == 2, \
        "diff_ci on length-2 tensors should return two length-2 tensors (elementwise)"


# ------------------------------------------------------------------ bootstrap

def _trace_calls(fn, *args, **kwargs):
    try:
        from torch.overrides import TorchFunctionMode
    except ImportError:
        raise Skip("needs torch.overrides.TorchFunctionMode (torch >= 2.0)")

    class _Counter(TorchFunctionMode):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def __torch_function__(self, func, types, args=(), kwargs=None):
            self.calls += 1
            return func(*args, **(kwargs or {}))

    with _Counter() as c:
        out = fn(*args, **kwargs)
    return c.calls, out


def _boot(ns, a, b, n_boot, alpha, seed):
    g = torch.Generator().manual_seed(seed)
    lo, hi = _tuple(ns.bootstrap_ci(a, b, n_boot=n_boot, alpha=alpha, generator=g), 2,
                    "bootstrap_ci")
    assert lo.numel() == 1 and hi.numel() == 1, (
        f"bootstrap_ci should return two scalars (lo, hi), got shapes {tuple(lo.shape)}, "
        f"{tuple(hi.shape)}")
    lo, hi = float(lo), float(hi)
    assert math.isfinite(lo) and math.isfinite(hi), f"bootstrap_ci returned ({lo}, {hi})"
    return lo, hi


def _plugin_se(a, b):
    return math.sqrt(a.var(unbiased=False).item() / a.numel() + b.var(unbiased=False).item() / b.numel())


def _width_case(seed):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(600, generator=g, dtype=F64)                       # control: sd 1
    b = torch.randn(300, generator=g, dtype=F64) * 3.0 + 0.6           # treatment: sd 3, n halved
    return a, b


def _check_width_and_centre(ns, data_seed=1, boot_seed=2, n_boot=4000):
    a, b = _width_case(data_seed)
    a0, b0 = a.clone(), b.clone()
    lo, hi = _boot(ns, a, b, n_boot, 0.05, boot_seed)
    assert torch.equal(a, a0) and torch.equal(b, b0), \
        "bootstrap_ci modified its input tensors (e.g. an in-place sort) — work on copies"
    est = (b.mean() - a.mean()).item()
    se = _plugin_se(a, b)
    width_ratio = (hi - lo) / (2 * _z(0.975) * se)
    centre_off = ((lo + hi) / 2 - est) / se
    return lo, hi, est, se, width_ratio, centre_off


def check_bootstrap_is_deterministic_and_well_formed(ns):
    g = torch.Generator().manual_seed(3)
    a = torch.rand(120, generator=g, dtype=F64) * 4
    b = torch.rand(90, generator=g, dtype=F64) * 4 + 0.5
    r1 = _boot(ns, a, b, 2000, 0.05, seed=11)
    r2 = _boot(ns, a, b, 2000, 0.05, seed=11)
    assert r1 == r2, (
        f"two calls with identically seeded generators gave {r1} and {r2} — draw every random "
        f"number from the `generator` argument (torch.randint(..., generator=generator))")
    assert r1[1] - r1[0] > 1e-9 * (1 + abs(r1[0])), (
        f"bootstrap_ci returned the zero-width interval {r1} — resample WITH replacement "
        f"(a permutation of the data has exactly the same mean every time)")
    r3 = _boot(ns, a, b, 2000, 0.05, seed=12)
    assert r3 != r1, (
        "different generator seeds gave exactly the same interval — the resampling must use "
        "the generator's random draws")
    lo, hi = r1
    est = (b.mean() - a.mean()).item()
    assert lo < est < hi, (
        f"the 95% percentile interval ({lo:.4f}, {hi:.4f}) should contain the observed "
        f"difference of means {est:.4f} (= mean(b) - mean(a); B is the treatment)")
    lo90, hi90 = _boot(ns, a, b, 2000, 0.10, seed=11)
    assert hi90 - lo90 < hi - lo, (
        f"alpha=0.10 should give a NARROWER interval than alpha=0.05, got widths "
        f"{hi90 - lo90:.4f} vs {hi - lo:.4f}")


def check_bootstrap_width_and_centre(ns):
    lo, hi, est, se, wr, co = _check_width_and_centre(ns)
    assert abs(co) < 1.0, (
        f"the interval ({lo:.4f}, {hi:.4f}) is centred {co:+.2f} standard errors away from the "
        f"observed difference {est:.4f}. Resample EACH arm from its own data — resampling the "
        f"pooled data simulates H0 and centres the interval on 0")
    assert 0.9 < wr < 1.1, (
        f"interval width {hi - lo:.4f} is {wr:.2f}x the expected 2 * 1.96 * SE = "
        f"{2 * _z(0.975) * se:.4f} (SE = sqrt(var_a/n_a + var_b/n_b) = {se:.4f}, arms of size "
        f"600 and 300 with sd 1 and 3). "
        + ("About 0.84x is what the quantiles alpha and 1 - alpha (instead of alpha/2 and "
           "1 - alpha/2) give. " if 0.8 < wr < 0.88 else "")
        + "Resample each arm independently with its own size, average each resample, and take "
          "the alpha/2 and 1 - alpha/2 quantiles of the mean differences")
    assert abs(co) < 0.25, (
        f"the interval ({lo:.4f}, {hi:.4f}) is centred {co:+.2f} standard errors away from the "
        f"observed difference {est:.4f}; with 600 and 300 roughly normal values the bootstrap "
        f"distribution is nearly symmetric, so the percentile interval should be centred on the "
        f"estimate (within ~0.1 SE)")
    # order of units inside an arm is meaningless for an unpaired comparison
    g = torch.Generator().manual_seed(5)
    a = torch.randn(400, generator=g, dtype=F64).sort().values
    b = (torch.randn(400, generator=g, dtype=F64) * 2 + 0.3).sort().values
    lo_s, hi_s = _boot(ns, a, b, 4000, 0.05, seed=6)
    wr_s = (hi_s - lo_s) / (2 * _z(0.975) * _plugin_se(a, b))
    assert 0.9 < wr_s < 1.1, (
        f"with two equal-size arms whose units happen to be sorted, the interval width is "
        f"{wr_s:.2f}x the expected one. The arms are independent samples: draw SEPARATE "
        f"indices for a and b — reusing one index for both (paired resampling) ties the "
        f"resampled means together and collapses the interval")


def check_bootstrap_is_the_percentile_interval(ns):
    a = torch.linspace(-0.01, 0.01, 300, dtype=F64)           # nearly constant control arm
    q = (torch.arange(1, 31, dtype=F64) - 0.5) / 30
    b = torch.exp(1.5 * torch.special.ndtri(q))                # 30 log-normal quantiles: right-skewed
    lo, hi = _boot(ns, a, b, 6000, 0.05, seed=8)
    est = (b.mean() - a.mean()).item()
    assert lo < est < hi, (
        f"the interval ({lo:.4f}, {hi:.4f}) should contain the observed difference {est:.4f}")
    ratio = (hi - est) / (est - lo)
    assert ratio > 1.15, (
        f"on right-skewed data (30 log-normal values) the percentile interval is asymmetric — "
        f"the upper arm (hi - est = {hi - est:.3f}) should be clearly longer than the lower arm "
        f"(est - lo = {est - lo:.3f}), ratio {ratio:.2f}. "
        + ("A symmetric interval suggests est +/- z * SE_boot (the 'normal' bootstrap). "
           if 0.87 < ratio < 1.15 else "")
        + ("A longer LOWER arm suggests the reflected 'basic' interval (2 est - q_hi, 2 est - q_lo). "
           if ratio <= 0.87 else "")
        + "Return the alpha/2 and 1 - alpha/2 quantiles of the bootstrap differences themselves")


def check_bootstrap_is_vectorized(ns):
    g = torch.Generator().manual_seed(9)
    a, b = torch.rand(20, generator=g, dtype=F64), torch.rand(25, generator=g, dtype=F64)
    c_small, _ = _trace_calls(ns.bootstrap_ci, a, b, n_boot=100, alpha=0.05,
                              generator=torch.Generator().manual_seed(1))
    c_big, _ = _trace_calls(ns.bootstrap_ci, a, b, n_boot=3000, alpha=0.05,
                            generator=torch.Generator().manual_seed(1))
    assert c_big <= c_small + 50, (
        f"bootstrap_ci made {c_small} torch calls for n_boot=100 but {c_big} for n_boot=3000 — "
        f"there is a Python loop over resamples. Draw all indices at once: "
        f"torch.randint(n, (n_boot, n), generator=generator)")


def _coverage(ns, seed=2024, K=250, n_boot=1000):
    """Fraction of K simulated experiments whose 95% interval covers the true difference."""
    g = torch.Generator().manual_seed(seed)
    true_diff = 1.25 - 1.0
    hits = 0
    for _ in range(K):
        a = torch.empty(60, dtype=F64).exponential_(1.0, generator=g)          # mean 1
        b = torch.empty(90, dtype=F64).exponential_(1 / 1.25, generator=g)     # mean 1.25
        lo, hi = _tuple(ns.bootstrap_ci(a, b, n_boot=n_boot, alpha=0.05, generator=g), 2,
                        "bootstrap_ci")
        hits += int(float(lo) <= true_diff <= float(hi))
    return hits / K


def check_bootstrap_coverage_by_simulation(ns):
    g = torch.Generator().manual_seed(9)
    a, b = torch.rand(20, generator=g, dtype=F64), torch.rand(25, generator=g, dtype=F64)
    c_small, _ = _trace_calls(ns.bootstrap_ci, a, b, n_boot=10, alpha=0.05,
                              generator=torch.Generator().manual_seed(1))
    c_big, _ = _trace_calls(ns.bootstrap_ci, a, b, n_boot=500, alpha=0.05,
                            generator=torch.Generator().manual_seed(1))
    assert c_big <= c_small + 50, (
        "skipped the coverage simulation: bootstrap_ci loops over resamples in Python "
        "(see the vectorization check) and 250 experiments would take minutes")
    cov = _coverage(ns)
    assert 0.85 <= cov <= 0.995, (
        f"over 250 simulated experiments (exponential data, n = 60 vs 90) the 95% interval "
        f"covered the true difference {cov:.1%} of the time; a percentile bootstrap gives "
        f"about 93-95%. "
        + ("Far too low: the interval is too narrow or not centred on the estimate. " if cov < 0.85
           else "Too high: the interval is far too wide — use the spread of resampled MEANS, not "
                "of individual values. "))


# ------------------------------------------------------------------ sample size

def _n_ref(p, d, alpha=0.05, power=0.8):
    p1, p2 = p, p + d
    pbar = (p1 + p2) / 2
    num = _z(1 - alpha / 2) * math.sqrt(2 * pbar * (1 - pbar)) + _z(power) * math.sqrt(
        p1 * (1 - p1) + p2 * (1 - p2))
    return (num / d) ** 2


NCASES = [  # (p_base, mde_abs, alpha, power); fractional parts of n are in [0.1, 0.9]
    (0.05, 0.001, 0.05, 0.8),
    (0.10, 0.02, 0.05, 0.8),
    (0.30, -0.05, 0.05, 0.8),
    (0.50, 0.08, 0.01, 0.9),
    (0.02, 0.004, 0.10, 0.95),
    (0.20, 0.03, 0.05, 0.8),
]


def check_sample_size_matches_formula(ns):
    for p, d, alpha, power in NCASES:
        exact = _n_ref(p, d, alpha, power)
        want = math.ceil(exact)
        got = ns.sample_size_per_arm(p, d, alpha=alpha, power=power)
        gf = float(_f(got))
        label = f"sample_size_per_arm(p_base={p}, mde_abs={d}, alpha={alpha}, power={power})"
        assert gf == round(gf), f"{label} returned {gf}; round UP to a whole number of units"
        if int(gf) == want:
            continue
        msg = f"{label} = {int(gf)}, expected {want} (unrounded {exact:.2f})."
        pbar = p + d / 2
        simple = ((_z(1 - alpha / 2) + _z(power)) ** 2
                  * (p * (1 - p) + (p + d) * (1 - p - d)) / d ** 2)
        one_sided = ((_z(1 - alpha) * math.sqrt(2 * pbar * (1 - pbar)) + _z(power)
                      * math.sqrt(p * (1 - p) + (p + d) * (1 - p - d))) / d) ** 2
        if int(gf) == math.floor(exact):
            msg += " Round UP (math.ceil) — rounding down leaves the test under-powered."
        elif abs(gf - math.ceil(one_sided)) <= 1:
            msg += " That is z_(1-alpha): a two-sided test at level alpha uses z_(1-alpha/2)."
        elif abs(gf - math.ceil(simple)) <= 1:
            msg += (" That is the all-unpooled formula; this problem uses the pooled variance "
                    "2*pbar*(1-pbar) under H0 (next to z_(1-alpha/2)) and the unpooled "
                    "p1q1 + p2q2 under H1 (next to z_power).")
        elif abs(gf - exact / 2) <= 2:
            msg += " That is half the answer — the formula already gives n PER ARM."
        raise AssertionError(msg)


def check_sample_size_monotonicity(ns):
    n = lambda d, power=0.8, alpha=0.05: int(float(_f(ns.sample_size_per_arm(
        0.1, d, alpha=alpha, power=power))))
    a, b, c = n(0.005), n(0.01), n(0.02)
    assert a > b > c, f"a larger MDE must need fewer units: n(0.005)={a}, n(0.01)={b}, n(0.02)={c}"
    assert 3.5 < a / b < 4.5, (
        f"halving the MDE should roughly quadruple n (n ~ 1/d^2), got ratio {a / b:.2f}")
    assert n(0.01, power=0.9) > n(0.01, power=0.8), "more power must need more units"
    assert n(0.01, alpha=0.01) > n(0.01, alpha=0.05), "a smaller alpha must need more units"


# ------------------------------------------------------------------ SRM

def _srm_ref(na, nb, r):
    na, nb = _f(na), _f(nb)
    N = na + nb
    chi2 = (na - N * r) ** 2 / (N * r) + (nb - N * (1 - r)) ** 2 / (N * (1 - r))
    return torch.special.erfc(torch.sqrt(chi2 / 2))


def check_srm_pvalue(ns):
    cases = [(50_000, 51_000, 0.5), (100_000, 100_400, 0.5), (9_000, 1_100, 0.9),
             (2_950, 7_050, 0.3), (1_000, 1_000, 0.5), (4_100, 900, 0.8)]
    for na, nb, r in cases:
        got = _f(ns.srm_pvalue(na, nb, expected_ratio=r))
        want = _srm_ref(na, nb, r)
        label = f"srm_pvalue({na}, {nb}, expected_ratio={r})"
        ok, _ = _close(got, want, label, rtol=1e-6, atol=1e-13)
        if ok:
            continue
        msg = f"{label} = {float(got):.6g}, expected {float(want):.6g}."
        if r != 0.5 and abs(float(got) - float(_srm_ref(na, nb, 1 - r))) < 1e-9:
            msg += " expected_ratio is the expected share of arm A (the first count), not of B."
        elif abs(float(got) - float(want) / 2) < 1e-9:
            msg += " That is half the chi-square tail — P(chi2_1 > s) = erfc(sqrt(s / 2))."
        else:
            msg += (" Pearson chi-square with expected counts N*r and N*(1-r), 1 degree of "
                    "freedom, no continuity correction.")
        raise AssertionError(msg)
    got = _f(ns.srm_pvalue(torch.tensor([50_000, 1_000]), torch.tensor([51_000, 1_000])))
    assert got.numel() == 2 and torch.allclose(
        got.reshape(-1), torch.stack([_srm_ref(50_000, 51_000, 0.5), _srm_ref(1000, 1000, 0.5)]),
        rtol=1e-6, atol=1e-13), "srm_pvalue should work elementwise on count tensors"


# ------------------------------------------------------------------ CUPED

def check_cuped(ns):
    g = torch.Generator().manual_seed(12)
    for n in (5000, 12):
        x = torch.randn(n, generator=g, dtype=F64) * 3 + 10                   # pre-period metric
        y = 0.8 * x + torch.randn(n, generator=g, dtype=F64) * 2.0 + 5         # in-experiment metric
        x0, y0 = x.clone(), y.clone()
        out = _tuple(ns.cuped(y, x), 2, "cuped")
        y_adj, theta = out
        assert torch.equal(x, x0) and torch.equal(y, y0), "cuped modified its inputs"
        assert tuple(y_adj.shape) == (n,), f"y_adj should have shape ({n},), got {tuple(y_adj.shape)}"
        xc, yc = x - x.mean(), y - y.mean()
        th = ((xc * yc).mean() / (xc * xc).mean()).item()
        assert theta.numel() == 1, f"theta should be a scalar, got shape {tuple(theta.shape)}"
        tf = float(theta)
        if abs(tf - th) > 1e-9 * abs(th):
            msg = f"n={n}: theta = {tf:.8f}, expected cov(y, x) / var(x) = {th:.8f}."
            if abs(tf - th * n / (n - 1)) < 1e-9 * abs(th) or abs(tf - th * (n - 1) / n) < 1e-9 * abs(th):
                msg += (" Off by n/(n-1): the covariance and the variance must use the same "
                        "normalization (both 1/n or both 1/(n-1)).")
            elif abs(tf - (xc * yc).mean().item() / (yc * yc).mean().item()) < 1e-9:
                msg += " You divided by var(y); theta is the regression slope of y on x: cov / var(x)."
            elif abs(tf - 1.0) < 1e-9:
                msg += (" theta = 1 is what you get by regressing y on ITSELF — the covariate must "
                        "be the pre-experiment metric x_pre, never an in-experiment metric.")
            raise AssertionError(msg)
        assert abs(y_adj.mean().item() - y.mean().item()) < 1e-9 * (1 + abs(y.mean().item())), (
            f"n={n}: mean(y_adj) = {y_adj.mean().item():.6f} but mean(y) = {y.mean().item():.6f} — "
            f"subtract theta * (x - mean(x)), CENTRED, so the metric's mean (and the treatment "
            f"effect) is unchanged")
        ok, _ = _close(y_adj, y - th * xc, f"n={n}: y_adj", rtol=1e-9, atol=1e-9)
        assert ok, f"n={n}: y_adj should be y - theta * (x - mean(x))"
        rho = ((xc * yc).mean() / torch.sqrt((xc * xc).mean() * (yc * yc).mean())).item()
        ratio = y_adj.var().item() / y.var().item()
        assert abs(ratio - (1 - rho ** 2)) < 1e-9, (
            f"n={n}: var(y_adj)/var(y) = {ratio:.6f}, the optimal theta gives exactly "
            f"1 - rho^2 = {1 - rho ** 2:.6f} in-sample")
    # a constant covariate carries no information: theta = 0, y unchanged, no NaN
    y = torch.randn(50, generator=g, dtype=F64)
    y_adj, theta = _tuple(ns.cuped(y, torch.full((50,), 3.0, dtype=F64)), 2, "cuped")
    assert torch.isfinite(theta).all() and abs(float(theta)) < 1e-12, (
        f"with a constant pre-period covariate var(x) = 0; return theta = 0, got {float(theta)}")
    assert torch.allclose(y_adj, y, atol=1e-12), "with a constant covariate y_adj must equal y"


CHECKS = [
    check_ztest_matches_pooled_closed_form,
    check_ztest_vectorized_symmetric_and_edge_cases,
    check_diff_ci_is_the_unpooled_wald_interval,
    check_bootstrap_is_deterministic_and_well_formed,
    check_bootstrap_width_and_centre,
    check_bootstrap_is_the_percentile_interval,
    check_bootstrap_is_vectorized,
    check_bootstrap_coverage_by_simulation,
    check_sample_size_matches_formula,
    check_sample_size_monotonicity,
    check_srm_pvalue,
    check_cuped,
]
