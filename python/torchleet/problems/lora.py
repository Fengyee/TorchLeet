"""lora — low-rank adaptation.

The strongest check here is self-consistency: merging LoRA weights back into the
base layer must not change what the model outputs. That property holds for any
correct implementation regardless of how A and B are initialized or scaled
internally, so it cannot false-fail a valid solution.

Merge consistency alone would accept a layer that forgets the alpha/r scaling
everywhere (forward and merge agree with each other), so the effective update is
also compared against (alpha/r) * A @ B, with A and B found by shape — that works
for (d_in, r)/(r, d_out) parameters and for PEFT-style nn.Linear adapters alike.
"""
import torch
import torch.nn as nn

ENTRIES = ["LoRALinear", "apply_lora", "merge_lora"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "LoRA adds a low-rank detour next to the frozen layer: "
    "output = linear(x) + (alpha/r) * (x @ A) @ B.",
    "A is (d_in, rank) and B is (rank, d_out). Start B at zero and A random "
    "(bound ~1/sqrt(d_in)): the adapter is a no-op at step 0 but still gets a gradient.",
    "Merging means W_new = W_old + (alpha/r) * (A @ B).T — the transpose matches "
    "nn.Linear's (d_out, d_in) weight layout.",
]

D_IN, D_OUT, RANK = 16, 8, 4


def _model():
    torch.manual_seed(0)
    m = nn.Sequential()
    m.add_module("target", nn.Linear(D_IN, D_OUT))
    return m


def check_lora_layer_shapes_are_low_rank(ns):
    base = nn.Linear(D_IN, D_OUT)
    layer = ns.LoRALinear(base, rank=RANK, alpha=1.0)
    params = dict(layer.named_parameters())
    lows = [p for n, p in params.items() if p.requires_grad and p.dim() == 2]
    assert lows, "LoRALinear exposes no trainable 2D adapter matrices"
    for p in lows:
        assert RANK in p.shape, \
            f"trainable matrix {tuple(p.shape)} has no dimension equal to rank={RANK}"


def check_base_weights_are_frozen(ns):
    base = nn.Linear(D_IN, D_OUT)
    layer = ns.LoRALinear(base, rank=RANK, alpha=1.0)
    assert not base.weight.requires_grad, \
        "the wrapped layer's weight must be frozen (requires_grad=False)"


def check_forward_shape(ns):
    layer = ns.LoRALinear(nn.Linear(D_IN, D_OUT), rank=RANK, alpha=1.0)
    out = layer(torch.randn(3, D_IN))
    assert tuple(out.shape) == (3, D_OUT), \
        f"expected output {(3, D_OUT)}, got {tuple(out.shape)}"


def check_apply_lora_replaces_target(ns):
    m = ns.apply_lora(_model(), rank=RANK, alpha=1.0, target_modules=["target"])
    assert isinstance(m.target, ns.LoRALinear), \
        "apply_lora did not wrap the layer named in target_modules"


def check_merge_preserves_output(ns):
    """Merged and unmerged models must agree — the property that defines a merge."""
    m = ns.apply_lora(_model(), rank=RANK, alpha=2.0, target_modules=["target"])
    # Give the adapter a non-trivial value, else the merge is trivially identity.
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad and p.dim() == 2 and RANK in p.shape:
                p.normal_(0, 0.3)
    x = torch.randn(5, D_IN)
    m.eval()
    with torch.no_grad():
        before = m(x)
        merged = ns.merge_lora(m)
        after = merged(x)
    diff = (before - after).abs().max().item()
    assert torch.allclose(before, after, atol=1e-5), (
        f"merging changed the model's output (max diff {diff:.2e}). "
        f"W_new should equal W_old + scaling * (A@B).T")


def check_merge_removes_adapter(ns):
    m = ns.apply_lora(_model(), rank=RANK, alpha=1.0, target_modules=["target"])
    merged = ns.merge_lora(m)
    assert not isinstance(merged.target, ns.LoRALinear), \
        "after merge_lora the layer should be a plain nn.Linear again"


def _adapter_factors(layer):
    """Return the adapter as (A_, B_) with A_ (D_IN, RANK) and B_ (RANK, D_OUT),
    whatever layout the solver stored them in."""
    lows = [p for p in layer.parameters() if p.requires_grad and p.dim() == 2]
    a = [p for p in lows if set(p.shape) == {D_IN, RANK}]
    b = [p for p in lows if set(p.shape) == {RANK, D_OUT}]
    if len(a) != 1 or len(b) != 1 or len(lows) != 2:
        from torchleet.runner import Skip
        raise Skip("could not identify exactly one A and one B adapter matrix")
    A = a[0] if tuple(a[0].shape) == (D_IN, RANK) else a[0].T
    B = b[0] if tuple(b[0].shape) == (RANK, D_OUT) else b[0].T
    return A, B


def check_adapter_starts_as_a_no_op(ns):
    """B = 0 at init: wrapping must not change the pretrained layer's output."""
    torch.manual_seed(1)
    base = nn.Linear(D_IN, D_OUT)
    w, b = base.weight.detach().clone(), base.bias.detach().clone()
    layer = ns.LoRALinear(base, rank=RANK, alpha=8.0)
    layer.eval()
    x = torch.randn(5, D_IN)
    with torch.no_grad():
        got = layer(x)
    expected = torch.nn.functional.linear(x, w, b)
    diff = (got - expected).abs().max().item()
    assert torch.allclose(got, expected, atol=1e-6), (
        f"a freshly wrapped layer must compute exactly what the pretrained layer did "
        f"(max diff {diff:.2e}) — initialise B to zeros so delta W starts at 0")


def check_adapter_gets_gradient_at_init(ns):
    """A must be random: with A = B = 0 neither factor ever receives a gradient."""
    torch.manual_seed(2)
    layer = ns.LoRALinear(nn.Linear(D_IN, D_OUT), rank=RANK, alpha=1.0)
    x = torch.randn(5, D_IN)
    (layer(x) * torch.randn(5, D_OUT)).sum().backward()
    trainable = [p for p in layer.parameters() if p.requires_grad]
    assert any(p.grad is not None and bool((p.grad != 0).any()) for p in trainable), (
        "no adapter parameter received a gradient on the first step — if A and B are "
        "both zero the adapter can never move; initialise A randomly and B to zero")


def check_update_is_scaled_by_alpha_over_rank(ns):
    """The effective weight update must be (alpha/r) * A @ B, not A @ B or alpha * A @ B."""
    torch.manual_seed(3)
    base = nn.Linear(D_IN, D_OUT)
    w, b = base.weight.detach().clone(), base.bias.detach().clone()
    alpha = 8.0
    layer = ns.LoRALinear(base, rank=RANK, alpha=alpha)
    A, B = _adapter_factors(layer)
    with torch.no_grad():
        for p in layer.parameters():
            if p.requires_grad:
                p.normal_(0, 0.5)
    layer.eval()
    eye = torch.eye(D_IN)
    with torch.no_grad():
        delta = layer(eye) - torch.nn.functional.linear(eye, w, b)  # (D_IN, D_OUT)
        expected = (alpha / RANK) * (A @ B)
    diff = (delta - expected).abs().max().item()
    assert torch.allclose(delta, expected, atol=1e-5), (
        f"the LoRA branch should add (alpha/r) * x @ A @ B with alpha={alpha}, "
        f"r={RANK} (scaling {alpha / RANK}); the measured update differs by up to "
        f"{diff:.2e}. Ratio of norms: {float(delta.norm() / expected.norm()):.3f}")


CHECKS = [check_lora_layer_shapes_are_low_rank, check_base_weights_are_frozen,
          check_forward_shape, check_adapter_starts_as_a_no_op,
          check_adapter_gets_gradient_at_init, check_update_is_scaled_by_alpha_over_rank,
          check_apply_lora_replaces_target,
          check_merge_preserves_output, check_merge_removes_adapter]
