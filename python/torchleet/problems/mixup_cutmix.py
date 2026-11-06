"""mixup-cutmix — batch-level Mixup and CutMix with a reproducible generator.

The checks never need the solver's permutation: labels are one-hot over
distinct classes, so the partner of sample i can be read off y_mixed (the one
non-zero class that is not i's own). Given that partner, every claim is checked
exactly: Mixup's output is lam*x + (1-lam)*x[perm] for the returned lam;
CutMix's output takes every pixel from either x[i] or x[perm[i]], the borrowed
pixels form one axis-aligned box shared by the batch (a single scalar lam
implies one box), and the borrowed fraction equals 1 - lam_returned (so lam
must be recomputed from the clipped box). The lam distribution is tested
against Beta(alpha, alpha) at a small and a large alpha, and reproducibility is
tested by scrambling the global RNG between two calls with equal-seeded
generators.
"""
import torch

ENTRIES = ["mixup", "cutmix"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Both functions pick one lam per batch and one random permutation perm of the "
    "batch; sample i is mixed with sample perm[i]. Labels are mixed with the same "
    "lam and perm, so a one-hot row becomes lam on its class and 1 - lam on the partner's.",
    "torch.distributions.Beta cannot take a generator. A Beta(a, a) draw is "
    "G1 / (G1 + G2) with G1, G2 independent Gamma(a) draws, and "
    "torch._standard_gamma(concentration_tensor, generator=g) draws Gammas from your "
    "generator; torch.randperm and torch.randint take generator= too.",
    "CutMix: cut_h = int(H * sqrt(1 - lam)), cut_w likewise, centre (cy, cx) uniform "
    "over the image, clip the box edges to [0, H] and [0, W], copy that region from "
    "x[perm] into a clone of x, then set lam = 1 - box_area / (H * W) before mixing labels.",
]


def _onehot_batch(N, K, seed):
    g = torch.Generator().manual_seed(seed)
    classes = torch.randperm(K, generator=g)[:N]
    y = torch.zeros(N, K)
    y[torch.arange(N), classes] = 1.0
    return y, classes.tolist()


def _lam(v):
    lam = float(v)
    assert 0.0 <= lam <= 1.0, f"lam must lie in [0, 1], got {lam}"
    return lam


def _call(fn, name, x, y, alpha, seed):
    # Clones, so an in-place implementation fails only check_inputs_not_modified.
    out = fn(x.clone(), y.clone(), alpha, generator=torch.Generator().manual_seed(seed))
    assert isinstance(out, (tuple, list)) and len(out) == 3, \
        f"{name} should return (x_mixed, y_mixed, lam)"
    xm, ym, lam = out
    assert tuple(xm.shape) == tuple(x.shape), \
        f"{name}: x_mixed should keep the shape {tuple(x.shape)}, got {tuple(xm.shape)}"
    assert tuple(ym.shape) == tuple(y.shape), \
        f"{name}: y_mixed should keep the shape {tuple(y.shape)}, got {tuple(ym.shape)}"
    return xm, ym, _lam(lam)


def _partner(ym, classes, name):
    """Recover perm from mixed one-hot labels; perm[i] = i when no other class appears."""
    where = {c: i for i, c in enumerate(classes)}
    perm = []
    for i, c in enumerate(classes):
        row = ym[i].detach().clone().float()
        row[c] = 0.0
        if float(row.max()) <= 0.0:
            perm.append(i)
            continue
        j = int(row.argmax())
        assert j in where, (
            f"{name}: y_mixed[{i}] puts weight on class {j}, which no sample in the batch "
            f"has — labels must be mixed with y[perm], the partners' own labels")
        perm.append(where[j])
    return perm


def _check_label_mix(ym, y, lam, perm, name):
    exp = lam * y + (1.0 - lam) * y[perm]
    diff = (ym.float() - exp).abs().max().item()
    assert diff <= 1e-5, (
        f"{name}: y_mixed is not lam * y + (1 - lam) * y[perm] for the returned "
        f"lam = {lam:.4f} (max |diff| {diff:.2e})")
    sums = ym.float().sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
        f"{name}: every row of y_mixed must still sum to 1, got {sums.tolist()}"


# --------------------------------------------------------------------------- mixup
def check_mixup_is_lambda_blend_of_pairs(ns):
    N, K = 8, 11
    y, classes = _onehot_batch(N, K, seed=0)
    x = torch.randn(N, 3, 5, 4, generator=torch.Generator().manual_seed(1))
    nontrivial = 0
    for seed in range(12):
        for alpha in (0.4, 1.0):
            xm, ym, lam = _call(ns.mixup, "mixup", x, y, alpha, seed)
            perm = _partner(ym, classes, "mixup")
            if 0.0 < lam < 1.0:
                assert sorted(perm) == list(range(N)), (
                    f"mixup: the partners {perm} are not a permutation of the batch — "
                    f"pair samples with torch.randperm, not independent random indices")
            _check_label_mix(ym, y, lam, perm, "mixup")
            exp = lam * x + (1.0 - lam) * x[perm]
            diff = (xm - exp).abs().max().item()
            assert diff <= 1e-5, (
                f"mixup: x_mixed is not lam * x + (1 - lam) * x[perm] for the returned "
                f"lam = {lam:.4f} and the partners implied by the labels (max |diff| "
                f"{diff:.2e}) — are x and y mixed with the same lam and the same perm?")
            nontrivial += 0.01 < lam < 0.99 and perm != list(range(N))
    assert nontrivial >= 5, \
        "mixup almost never mixes anything (lam stuck at 0/1 or perm = identity)"


def check_mixup_lambda_follows_beta(ns):
    y, _ = _onehot_batch(2, 2, seed=0)
    x = torch.randn(2, 1, 2, 2)

    def lams(alpha, n=200):
        return torch.tensor([_call(ns.mixup, "mixup", x, y, alpha, s)[2] for s in range(n)])

    l1 = lams(1.0)
    m, ext = l1.mean().item(), ((l1 < 0.1) | (l1 > 0.9)).float().mean().item()
    assert 0.43 <= m <= 0.57 and 0.10 <= ext <= 0.32, (
        f"with alpha=1, Beta(1, 1) is Uniform(0, 1): expected mean ~0.5 and ~20% of lam "
        f"outside [0.1, 0.9], got mean {m:.3f} and {ext:.0%}")
    l02 = lams(0.2)
    ext = ((l02 < 0.1) | (l02 > 0.9)).float().mean().item()
    assert ext >= 0.5, (
        f"with alpha=0.2, Beta(0.2, 0.2) is U-shaped (~68% of lam within 0.1 of 0 or 1), "
        f"got only {ext:.0%} — is lam really drawn from Beta(alpha, alpha)?")
    l100 = lams(100.0, n=100)
    worst = (l100 - 0.5).abs().max().item()
    assert worst <= 0.2, (
        f"with alpha=100, Beta(100, 100) concentrates at 0.5 (std ~0.035), but a lam "
        f"{worst + 0.5:.3f} or {0.5 - worst:.3f} came out — is lam drawn from Beta(alpha, alpha)?")


# --------------------------------------------------------------------------- cutmix
def _cutmix_cases(ns, seeds=range(30), alpha=1.0, shape=(6, 3, 12, 20), K=9):
    N = shape[0]
    y, classes = _onehot_batch(N, K, seed=3)
    x = torch.randn(*shape, generator=torch.Generator().manual_seed(4))
    for seed in seeds:
        xm, ym, lam = _call(ns.cutmix, "cutmix", x, y, alpha, seed)
        perm = _partner(ym, classes, "cutmix")
        yield x, y, xm, ym, lam, perm


def _box_mask(x, xm, perm):
    """Boolean (H, W) mask of pixels borrowed from the partner, or None if no partner."""
    masks = []
    for i, j in enumerate(perm):
        from_self = xm[i] == x[i]
        from_partner = xm[i] == x[j]
        assert bool((from_self | from_partner).all()), (
            f"cutmix: some pixels of x_mixed[{i}] equal neither x[{i}] nor its partner "
            f"x[{j}] — CutMix copies pixels, it never blends them"
            + ("" if j != i else " (here the labels say sample i was paired with itself)"))
        if j == i:
            continue
        m = from_partner & ~from_self                     # (C, H, W)
        assert bool((m == m[0]).all()), \
            f"cutmix: sample {i}'s pasted region differs between channels — paste all channels"
        masks.append(m[0])
    if not masks:
        return None
    for k, m in enumerate(masks[1:], 1):
        assert torch.equal(m, masks[0]), (
            "cutmix: different samples got different boxes — use one box for the whole "
            "batch (lam is a single number, so the box area must be shared)")
    return masks[0]


def check_cutmix_pastes_one_box_from_the_partner(ns):
    boxes = 0
    for x, y, xm, ym, lam, perm in _cutmix_cases(ns):
        if 0.0 < lam < 1.0:
            assert sorted(perm) == list(range(len(perm))), (
                f"cutmix: the partners {perm} are not a permutation of the batch — "
                f"pair samples with torch.randperm")
        m = _box_mask(x, xm, perm)
        if m is None or not m.any():
            continue
        rows, cols = m.any(dim=1), m.any(dim=0)
        r, c = rows.nonzero().flatten(), cols.nonzero().flatten()
        contiguous = (r[-1] - r[0] + 1 == len(r)) and (c[-1] - c[0] + 1 == len(c))
        assert contiguous and torch.equal(m, rows[:, None] & cols[None, :]), \
            "cutmix: the pixels taken from the partner do not form one axis-aligned rectangle"
        boxes += 1
    assert boxes >= 10, (
        f"cutmix pasted a non-empty box in only {boxes} of 30 calls with alpha=1 — "
        f"lam ~ Uniform(0, 1) there, so most calls should paste something")


def check_cutmix_lambda_is_recomputed_from_clipped_box(ns):
    for x, y, xm, ym, lam, perm in _cutmix_cases(ns, seeds=range(40)):
        m = _box_mask(x, xm, perm)
        if m is None:
            assert torch.equal(xm, x), "cutmix changed pixels although no sample got a partner"
            continue
        frac = m.float().mean().item()
        assert abs(frac - (1.0 - lam)) <= 1e-6, (
            f"cutmix: {frac:.4f} of the pixels come from the partner but the returned lam "
            f"is {lam:.4f} (so 1 - lam = {1 - lam:.4f}). After clipping the box to the image, "
            f"recompute lam = 1 - box_area / (H * W) and mix the labels with that")
        _check_label_mix(ym, y, lam, perm, "cutmix")


def check_cutmix_box_area_follows_beta(ns):
    fracs = []
    for x, y, xm, ym, lam, perm in _cutmix_cases(ns, seeds=range(40), alpha=100.0,
                                                 shape=(4, 1, 32, 32), K=4):
        fracs.append(1.0 - lam)
    mean = sum(fracs) / len(fracs)
    assert 0.25 <= mean <= 0.6, (
        f"with alpha=100 (lam ~ 0.5) the box should cover about half the image before "
        f"clipping (side ratio sqrt(1 - lam) ~ 0.71), but on average {mean:.2f} of the "
        f"image was replaced — use cut_h = H * sqrt(1 - lam), cut_w = W * sqrt(1 - lam)")


# --------------------------------------------------------------------------- both
def check_same_generator_seed_is_reproducible(ns):
    y, _ = _onehot_batch(6, 8, seed=0)
    x = torch.randn(6, 3, 10, 14, generator=torch.Generator().manual_seed(2))
    for name in ("mixup", "cutmix"):
        fn = getattr(ns, name)
        torch.manual_seed(0)
        a = fn(x, y, 1.0, generator=torch.Generator().manual_seed(123))
        torch.manual_seed(999)
        torch.rand(17)
        b = fn(x, y, 1.0, generator=torch.Generator().manual_seed(123))
        same = torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and float(a[2]) == float(b[2])
        assert same, (
            f"{name}: two calls with identically seeded generators gave different results "
            f"once the global RNG changed — some randomness (lam? perm? box?) is not drawn "
            f"from `generator`")
        pairings, lams = set(), set()
        for s in range(8):
            xm, ym, lam = fn(x, y, 1.0, generator=torch.Generator().manual_seed(s))
            pairings.add(tuple((ym > 0).flatten().tolist()))
            lams.add(round(float(lam), 6))
        assert len(lams) > 1, f"{name}: every generator seed gives the same lam"
        assert len(pairings) > 1, (
            f"{name}: every generator seed pairs the samples the same way — pair with a "
            f"random permutation drawn from the generator (not a fixed flip/roll)")
        torch.manual_seed(7)
        c = fn(x, y, 1.0)
        d = fn(x, y, 1.0)
        torch.manual_seed(7)
        e = fn(x, y, 1.0)
        assert torch.equal(c[0], e[0]) and float(c[2]) == float(e[2]), \
            f"{name}(generator=None) should draw from the global RNG (torch.manual_seed)"
        assert not (torch.equal(c[0], d[0]) and float(c[2]) == float(d[2])), (
            f"{name}(generator=None) returned the same result twice in a row — with no "
            f"generator, use the global RNG, not a freshly created (always identically "
            f"seeded) torch.Generator()")


def check_inputs_not_modified(ns):
    y, _ = _onehot_batch(5, 7, seed=0)
    x = torch.randn(5, 2, 9, 9, generator=torch.Generator().manual_seed(5))
    for name in ("mixup", "cutmix"):
        x0, y0 = x.clone(), y.clone()
        for s in range(4):
            getattr(ns, name)(x, y, 1.0, generator=torch.Generator().manual_seed(s))
        assert torch.equal(x, x0) and torch.equal(y, y0), (
            f"{name} modified its input batch in place — clone before pasting, the caller "
            f"may still need the clean batch (logging, a teacher model, the next augmentation)")


def check_alpha_zero_disables_mixing(ns):
    y, _ = _onehot_batch(4, 6, seed=0)
    x = torch.randn(4, 2, 6, 6, generator=torch.Generator().manual_seed(6))
    for name in ("mixup", "cutmix"):
        xm, ym, lam = _call(getattr(ns, name), name, x, y, 0.0, 0)
        assert lam == 1.0 and torch.equal(xm, x) and torch.equal(ym, y), (
            f"{name} with alpha <= 0 should disable mixing: return x and y unchanged "
            f"with lam = 1.0 (got lam = {lam})")


CHECKS = [
    check_mixup_is_lambda_blend_of_pairs,
    check_mixup_lambda_follows_beta,
    check_cutmix_pastes_one_box_from_the_partner,
    check_cutmix_lambda_is_recomputed_from_clipped_box,
    check_cutmix_box_area_follows_beta,
    check_same_generator_seed_is_reproducible,
    check_inputs_not_modified,
    check_alpha_zero_disables_mixing,
]
