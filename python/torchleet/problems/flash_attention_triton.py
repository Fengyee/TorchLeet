"""flash-attention-triton — tiled attention with online softmax, then the kernel.

Device policy, the same one triton-fused-softmax uses: the tiled PyTorch
implementation is real work and runs anywhere, so it is graded properly on CPU
with F.scaled_dot_product_attention as the oracle. The Triton kernel runs on a
GPU, or on CPU when the solver started Python with TRITON_INTERPRET=1; otherwise
its check reports SKIPPED and is never treated as passing. A plain CPU run
therefore reports partial verification, which is the honest answer. The kernel
is probed with a sequence length that is not a multiple of any block size: keys
past N are loaded as zeros, and a kernel that does not mask their scores to -inf
leaks them into the softmax denominator.

The tiling checks are self-consistency as much as oracle agreement: the block
size is a performance knob, so every block size — including one larger than the
sequence, and one that does not divide it — must produce the same output.
"""
import inspect
import os

import torch
import torch.nn.functional as F

from torchleet.runner import Skip

ENTRIES = ["flash_attention_pytorch"]
DEVICE = "cpu"          # the CPU-checkable portion; the kernel check gates itself
EXTRAS = []             # triton is only needed by the optional kernel check

HINTS = [
    "Outer loop over Q blocks, inner loop over K,V blocks; the only matrix you "
    "ever materialize is the (block_q, block_kv) tile of scores.",
    "Carry running_max, running_sum and running_output per Q block. When a tile "
    "raises the max, rescale BOTH the running sum and the running output by "
    "exp(old_max - new_max) before adding the tile's contribution.",
    "Normalize once, after the inner loop: output = running_output / running_sum. "
    "The last block may be short, so slice with min(start + block, N).",
    "Causal: a K,V tile that starts after the Q tile's last row is pure future — skip "
    "it. Only tiles straddling the diagonal need a mask (-inf where key > query); the "
    "first tile always contains key 0, so no row is ever fully masked.",
]

B, H, N, D = 2, 3, 40, 16


def _qkv(seed=0, shape=(B, H, N, D)):
    torch.manual_seed(seed)
    return (torch.randn(*shape) for _ in range(3))


def check_matches_standard_attention(ns):
    """F.scaled_dot_product_attention is the oracle — it ships with the user's PyTorch."""
    q, k, v = _qkv(0)
    got = ns.flash_attention_pytorch(q, k, v, block_size=16)
    assert got is not None, "flash_attention_pytorch returned None"
    assert tuple(got.shape) == (B, H, N, D), \
        f"expected output {(B, H, N, D)}, got {tuple(got.shape)}"
    expected = F.scaled_dot_product_attention(q, k, v)
    diff = (got - expected).abs().max().item()
    assert torch.allclose(got, expected, atol=1e-5), (
        f"tiled attention disagrees with full attention (max diff {diff:.2e}); "
        f"FlashAttention is exact, not an approximation — check the 1/sqrt(d) scale "
        f"and that the final division by running_sum happens once, at the end")


def check_block_size_does_not_change_the_answer(ns):
    """Tiling is a memory strategy: the result must be identical for every block size."""
    q, k, v = _qkv(1)
    expected = F.scaled_dot_product_attention(q, k, v)
    for bs in (8, 16, 32, 64, 128):
        got = ns.flash_attention_pytorch(q, k, v, block_size=bs)
        diff = (got - expected).abs().max().item()
        assert torch.allclose(got, expected, atol=1e-5), (
            f"block_size={bs} gives a different answer from full attention (max diff "
            f"{diff:.2e}); block_size={bs} means "
            f"{'the whole sequence is one tile' if bs >= N else 'several tiles'}, and "
            f"the online softmax must combine tiles exactly")


def check_sequence_not_divisible_by_block(ns):
    q, k, v = _qkv(2, shape=(1, 1, 50, 32))
    got = ns.flash_attention_pytorch(q, k, v, block_size=16)
    expected = F.scaled_dot_product_attention(q, k, v)
    assert tuple(got.shape) == (1, 1, 50, 32), \
        f"expected output {(1, 1, 50, 32)}, got {tuple(got.shape)}"
    diff = (got - expected).abs().max().item()
    assert torch.allclose(got, expected, atol=1e-5), (
        f"seq_len=50 with block_size=16 leaves a ragged last tile of 2 rows and it "
        f"is being mishandled (max diff {diff:.2e}) — clamp the block end with "
        f"min(start + block_size, N)")


def check_numerically_stable_on_large_scores(ns):
    """Why online softmax exists: exp() of raw scores would overflow here."""
    q, k, v = _qkv(3)
    q = q * 40
    got = ns.flash_attention_pytorch(q, k, v, block_size=16)
    assert torch.isfinite(got).all(), (
        "NaN/Inf on large scores — subtract the running max before exp() instead "
        "of exponentiating the raw scores")
    expected = F.scaled_dot_product_attention(q, k, v)
    diff = (got - expected).abs().max().item()
    assert torch.allclose(got, expected, atol=1e-4), \
        f"wrong values on large scores (max diff {diff:.2e})"


def check_attention_weights_are_a_distribution(ns):
    """Feeding V = I recovers the attention matrix: rows must be non-negative and sum to 1."""
    torch.manual_seed(4)
    n = 24
    q, k = torch.randn(1, 1, n, n), torch.randn(1, 1, n, n)
    v = torch.eye(n).view(1, 1, n, n)
    weights = ns.flash_attention_pytorch(q, k, v, block_size=8)[0, 0]
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
        f"with V = identity the output is the attention matrix itself, whose rows "
        f"must sum to 1; got a range of [{sums.min():.4f}, {sums.max():.4f}] — the "
        f"running sum is not accumulating every tile")
    assert (weights >= -1e-6).all(), \
        f"attention weights must be non-negative, got a minimum of {weights.min():.4e}"


def check_gradients_flow(ns):
    q, k, v = _qkv(5, shape=(1, 2, 16, 8))
    q, k, v = q.requires_grad_(True), k.requires_grad_(True), v.requires_grad_(True)
    ns.flash_attention_pytorch(q, k, v, block_size=8).sum().backward()
    for name, t in (("Q", q), ("K", k), ("V", v)):
        assert t.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(t.grad).all(), f"gradient w.r.t. {name} contains NaN/Inf"


def check_causal_matches_masked_attention(ns):
    """causal=True must equal SDPA(is_causal=True) — including ragged last tiles."""
    params = inspect.signature(ns.flash_attention_pytorch).parameters
    assert "causal" in params, (
        "flash_attention_pytorch needs a `causal` argument (default False) — "
        "decoder LLMs run FlashAttention with causal masking")
    for shape, bs in (((B, H, N, D), 16), ((1, 2, 50, 32), 16), ((1, 2, 50, 32), 64),
                      ((1, 1, 37, 16), 8)):
        q, k, v = _qkv(7, shape=shape)
        got = ns.flash_attention_pytorch(q, k, v, block_size=bs, causal=True)
        expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        assert torch.isfinite(got).all(), (
            f"NaN/Inf with causal=True (seq_len={shape[2]}, block_size={bs}) — a row "
            f"whose running max is still -inf gives exp(-inf - -inf) = NaN")
        diff = (got - expected).abs().max().item()
        assert torch.allclose(got, expected, atol=1e-5), (
            f"causal=True disagrees with causally masked attention (seq_len={shape[2]}, "
            f"block_size={bs}, max diff {diff:.2e}) — skip K,V tiles entirely in the "
            f"future and mask key > query inside tiles that straddle the diagonal")
    # non-causal must stay non-causal
    q, k, v = _qkv(8)
    got = ns.flash_attention_pytorch(q, k, v, block_size=16, causal=False)
    assert torch.allclose(got, F.scaled_dot_product_attention(q, k, v), atol=1e-5), \
        "causal=False must still give full (unmasked) attention"


def _triton_device():
    """Where a Triton kernel can run here: a GPU, or the CPU under the interpreter."""
    try:
        import triton  # noqa: F401
    except ImportError:
        raise Skip("needs `triton` — pip install triton")
    if torch.cuda.is_available():
        return "cuda"
    if os.environ.get("TRITON_INTERPRET") == "1":
        return "cpu"
    raise Skip("requires CUDA — or set TRITON_INTERPRET=1 before importing triton to run "
               "the kernel on CPU; the Triton kernel is not verified on this machine")


def check_triton_kernel(ns):
    """Skipped, never passed, when it cannot actually be verified."""
    fn = next((getattr(ns, name) for name in
               ("flash_attention_triton", "flash_attention_triton_wrapper", "flash_attention_gpu")
               if getattr(ns, name, None) is not None), None)
    if fn is None:
        raise Skip("pass flash_attention_triton=<host wrapper that launches your kernel> "
                   "to check the GPU path")
    device = _triton_device()
    # N = 100 is not a multiple of 16, 32, 64 or 128, so the last K,V tile is ragged
    q, k, v = (t.to(device) for t in _qkv(6, shape=(1, 2, 100, 64)))
    got = fn(q, k, v)
    expected = F.scaled_dot_product_attention(q, k, v)
    diff = (got - expected).abs().max().item()
    assert torch.allclose(got, expected, atol=1e-2), (
        f"the Triton kernel disagrees with full attention at seq_len=100 (max diff "
        f"{diff:.2e}) — keys past N are loaded as zeros and must be masked to -inf "
        f"before the softmax (tl.dot may run in tf32, hence the loose tolerance)")
    if "causal" in inspect.signature(fn).parameters:
        got = fn(q, k, v, causal=True)
        expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        diff = (got - expected).abs().max().item()
        assert torch.allclose(got, expected, atol=1e-2), (
            f"the Triton kernel with causal=True disagrees with causally masked "
            f"attention (max diff {diff:.2e})")


CHECKS = [
    check_matches_standard_attention,
    check_block_size_does_not_change_the_answer,
    check_sequence_not_divisible_by_block,
    check_numerically_stable_on_large_scores,
    check_attention_weights_are_a_distribution,
    check_gradients_flow,
    check_causal_matches_masked_attention,
    check_triton_kernel,
]
