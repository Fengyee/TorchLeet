"""memory-flops-estimation — the back-of-envelope maths behind "will it fit / how long".

Every formula is checked against something real rather than a restated formula:
  * transformer_params vs. sum(p.numel()) of a Llama-style decoder actually
    built from nn.Embedding / nn.Linear / nn.RMSNorm / nn.LayerNorm (on the meta
    device, so even 7B/8B configs instantiate instantly) — GQA, MQA, tied and
    untied heads, gated (SwiGLU) and plain MLPs, biases, both norms;
  * kv_cache_bytes vs. numel * element_size of K/V tensors actually allocated;
  * training_memory_gb vs. bytes of real weight / grad / master-weight /
    optimizer-state tensors after an optimizer step (AdamW keeps exp_avg and
    exp_avg_sq, SGD-momentum keeps momentum_buffer, plain SGD keeps nothing);
  * training_flops vs. torch.utils.flop_counter.FlopCounterMode on a forward +
    backward through a bias-free matmul network (6 = 2 forward + 4 backward).
"""
import torch
import torch.nn as nn

ENTRIES = ["transformer_params", "training_memory_gb", "kv_cache_bytes", "training_flops"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Count module by module: token embedding (vocab x d), then per layer the "
    "attention projections, the MLP and two norms, then a final norm and the LM "
    "head. With GQA, k_proj and v_proj only produce n_kv_heads * head_dim "
    "outputs, where head_dim = d_model // n_heads.",
    "A gated (SwiGLU) MLP has three d x d_ff matrices (gate, up, down), a plain "
    "MLP two. RMSNorm has a weight only (d); LayerNorm has weight + bias (2d). "
    "Tied embeddings mean the LM head reuses the embedding matrix, so count it "
    "once. bias=True adds one vector per Linear in attention and MLP (its "
    "out_features), never on the LM head.",
    "Memory per parameter: fp32 = 4 (weights) + 4 (grads) + optimizer; "
    "bf16/fp16-mixed = 2 + 2 + 4 (fp32 master copy) + optimizer; AdamW/Adam add "
    "8 (m and v in fp32), SGD with momentum 4, plain SGD 0. KV cache = 2 (K and "
    "V) * batch * seq_len * n_layers * n_kv_heads * head_dim * bytes. Training "
    "FLOPs = 6 * N * D: 2ND forward, 4ND backward.",
]


# --------------------------------------------------------------------------- params

class _Block(nn.Module):
    def __init__(self, d, n_heads, n_kv, d_ff, gated, bias, norm):
        super().__init__()
        hd = d // n_heads
        make_norm = {"rmsnorm": lambda: nn.RMSNorm(d), "layernorm": lambda: nn.LayerNorm(d)}[norm]
        self.attn_norm, self.mlp_norm = make_norm(), make_norm()
        self.q_proj = nn.Linear(d, n_heads * hd, bias=bias)
        self.k_proj = nn.Linear(d, n_kv * hd, bias=bias)
        self.v_proj = nn.Linear(d, n_kv * hd, bias=bias)
        self.o_proj = nn.Linear(n_heads * hd, d, bias=bias)
        self.gate_proj = nn.Linear(d, d_ff, bias=bias) if gated else None
        self.up_proj = nn.Linear(d, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d, bias=bias)


class _Decoder(nn.Module):
    """Parameter skeleton of a Llama-style decoder (RoPE: no position params)."""

    def __init__(self, vocab, d, n_layers, n_heads, n_kv, d_ff, tied, gated, bias, norm):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList(
            _Block(d, n_heads, n_kv, d_ff, gated, bias, norm) for _ in range(n_layers))
        self.norm = nn.RMSNorm(d) if norm == "rmsnorm" else nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        if tied:
            self.lm_head.weight = self.embed_tokens.weight


def _reference_count(cfg):
    with torch.device("meta"):
        m = _Decoder(cfg["vocab_size"], cfg["d_model"], cfg["n_layers"], cfg["n_heads"],
                     cfg["n_kv_heads"], cfg["d_ff"], cfg.get("tied_embeddings", True),
                     cfg.get("gated_mlp", True), cfg.get("bias", False),
                     cfg.get("norm", "rmsnorm"))
    return sum(p.numel() for p in m.parameters())      # shared (tied) params counted once


_CONFIGS = [
    ("tiny Llama, GQA, untied",
     dict(vocab_size=101, d_model=64, n_layers=3, n_heads=8, n_kv_heads=2, d_ff=172,
          tied_embeddings=False)),
    ("tiny Llama, GQA, tied",
     dict(vocab_size=101, d_model=64, n_layers=3, n_heads=8, n_kv_heads=2, d_ff=172,
          tied_embeddings=True)),
    ("MHA (n_kv_heads == n_heads), plain MLP",
     dict(vocab_size=50, d_model=48, n_layers=2, n_heads=6, n_kv_heads=6, d_ff=192,
          gated_mlp=False)),
    ("MQA (n_kv_heads == 1) with biases",
     dict(vocab_size=77, d_model=32, n_layers=4, n_heads=4, n_kv_heads=1, d_ff=96,
          bias=True, tied_embeddings=False)),
    ("GPT-style: LayerNorm, biases, plain MLP, tied",
     dict(vocab_size=120, d_model=40, n_layers=2, n_heads=5, n_kv_heads=5, d_ff=160,
          gated_mlp=False, bias=True, norm="layernorm")),
    ("LayerNorm + gated MLP + GQA, untied",
     dict(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=80,
          norm="layernorm", tied_embeddings=False)),
]

LLAMA2_7B = dict(vocab_size=32000, d_model=4096, n_layers=32, n_heads=32, n_kv_heads=32,
                 d_ff=11008, tied_embeddings=False)
LLAMA3_8B = dict(vocab_size=128256, d_model=4096, n_layers=32, n_heads=32, n_kv_heads=8,
                 d_ff=14336, tied_embeddings=False)


def _diagnose(cfg, got, want):
    """Name the most likely culprit by toggling one feature at a time."""
    def count(**over):
        c = dict(cfg)
        c.update(over)
        return _reference_count(c)
    tips = []
    if cfg["n_kv_heads"] != cfg["n_heads"] and got == count(n_kv_heads=cfg["n_heads"]):
        tips.append("k_proj/v_proj look full-width — with GQA they have n_kv_heads * head_dim outputs")
    if got == want + cfg["vocab_size"] * cfg["d_model"]:
        tips.append("the embedding seems counted twice — tied embeddings share one matrix")
    if got == want - cfg["vocab_size"] * cfg["d_model"]:
        tips.append("the LM head (vocab x d) is missing for untied embeddings")
    if cfg.get("norm", "rmsnorm") == "layernorm" and got == count(norm="rmsnorm"):
        tips.append("LayerNorm has a bias as well as a weight (2·d per norm)")
    n_norms = 2 * cfg["n_layers"] + 1
    per_norm = 2 if cfg.get("norm", "rmsnorm") == "layernorm" else 1
    if abs(got - want) in (per_norm * cfg["d_model"], n_norms * per_norm * cfg["d_model"],
                           2 * cfg["n_layers"] * per_norm * cfg["d_model"]):
        tips.append("check the norms: two per layer plus a final one")
    if got == count(gated_mlp=not cfg.get("gated_mlp", True)):
        tips.append("gated (SwiGLU) MLPs have 3 matrices, plain MLPs 2")
    if got - want == cfg["vocab_size"]:
        tips.append("the LM head has no bias")
    if got == count(bias=not cfg.get("bias", False)):
        tips.append("bias=True adds one vector of size out_features to every attention/MLP Linear")
    return (" Likely: " + "; ".join(tips) + ".") if tips else ""


def check_transformer_params_matches_a_real_model(ns):
    for label, cfg in _CONFIGS:
        want = _reference_count(cfg)
        got = ns.transformer_params(**cfg)
        assert int(got) == got and int(got) == want, (
            f"{label}: transformer_params gives {got}, but the same config built "
            f"from nn modules has {want} parameters (off by {int(got) - want})."
            + _diagnose(cfg, int(got), want))


def check_transformer_params_llama_scale(ns):
    for name, cfg, published in (("Llama-2-7B", LLAMA2_7B, 6_738_415_616),
                                 ("Llama-3-8B", LLAMA3_8B, 8_030_261_248)):
        want = _reference_count(cfg)
        assert want == published
        got = ns.transformer_params(**cfg)
        assert int(got) == want, (
            f"{name} ({cfg}) has {want:,} parameters; transformer_params gives "
            f"{int(got):,} (off by {int(got) - want:,})." + _diagnose(cfg, int(got), want))


# --------------------------------------------------------------------------- memory

def _tensor_bytes(ts):
    return sum(t.numel() * t.element_size() for t in ts)


def _measured_bytes_per_param(precision, optimizer):
    """Allocate the real training state for a small model and weigh it."""
    torch.manual_seed(0)
    master = nn.Sequential(nn.Linear(16, 12), nn.Linear(12, 5))
    params = list(master.parameters())
    n = sum(p.numel() for p in params)
    make_opt = {
        "adamw": lambda ps: torch.optim.AdamW(ps, lr=1e-3),
        "adam": lambda ps: torch.optim.Adam(ps, lr=1e-3),
        "sgd-momentum": lambda ps: torch.optim.SGD(ps, lr=1e-3, momentum=0.9),
        "sgd": lambda ps: torch.optim.SGD(ps, lr=1e-3),
    }[optimizer]
    x = torch.randn(4, 16)
    if precision == "fp32":
        master(x).sum().backward()
        held = _tensor_bytes(params) + _tensor_bytes(p.grad for p in params)
    else:
        low = {"bf16-mixed": torch.bfloat16, "fp16-mixed": torch.float16}[precision]
        work = nn.Sequential(nn.Linear(16, 12), nn.Linear(12, 5)).to(low)
        work.load_state_dict(master.state_dict())
        work(x.to(low)).float().sum().backward()        # grads land in low precision
        wparams = list(work.parameters())
        held = (_tensor_bytes(wparams) + _tensor_bytes(p.grad for p in wparams)
                + _tensor_bytes(params))                   # + fp32 master weights
        for p, w in zip(params, wparams):
            p.grad = w.grad.float()                        # transient upcast for the step
    opt = make_opt(params)
    opt.step()
    state = [t for p in params for t in opt.state[p].values()
             if torch.is_tensor(t) and t.shape == p.shape]  # per-element state only
    return (held + _tensor_bytes(state)) / n


def check_training_memory_matches_real_optimizer_state(ns):
    for precision in ("fp32", "bf16-mixed", "fp16-mixed"):
        for optimizer in ("adamw", "adam", "sgd-momentum", "sgd"):
            bpp = _measured_bytes_per_param(precision, optimizer)
            for n_params in (1_000_000, 6_738_415_616):
                got = ns.training_memory_gb(n_params, precision=precision, optimizer=optimizer)
                want = n_params * bpp / 1e9
                assert abs(float(got) - want) <= 1e-9 * want, (
                    f"training_memory_gb({n_params:,}, precision={precision!r}, "
                    f"optimizer={optimizer!r}) = {float(got):.6g} GB, expected "
                    f"{want:.6g} GB ({bpp:g} bytes/param, decimal GB = bytes / 1e9). "
                    + {"fp32": "fp32: 4 B weights + 4 B grads + optimizer state.",
                       "bf16-mixed": "bf16-mixed: 2 B weights + 2 B grads + 4 B fp32 master copy + optimizer state.",
                       "fp16-mixed": "fp16-mixed: 2 B weights + 2 B grads + 4 B fp32 master copy + optimizer state."}[precision]
                    + {"adamw": " AdamW keeps m and v in fp32 (8 B).",
                       "adam": " Adam keeps m and v in fp32 (8 B).",
                       "sgd-momentum": " SGD-momentum keeps one fp32 buffer (4 B).",
                       "sgd": " Plain SGD keeps no state."}[optimizer])


def check_training_memory_rejects_unknown_settings(ns):
    for kwargs in (dict(precision="fp64"), dict(optimizer="lion-x")):
        try:
            ns.training_memory_gb(1000, **kwargs)
        except ValueError:
            continue
        except Exception as e:
            raise AssertionError(
                f"training_memory_gb(1000, {kwargs}) raised {type(e).__name__}; raise "
                f"ValueError for an unknown setting") from None
        raise AssertionError(
            f"training_memory_gb(1000, {kwargs}) returned a number — raise ValueError "
            f"for an unknown setting instead of silently guessing")


# --------------------------------------------------------------------------- KV cache

def check_kv_cache_bytes_matches_allocated_tensors(ns):
    fp8 = getattr(torch, "float8_e4m3fn", torch.int8)
    for batch, seq, layers, kv, hd, dtype in ((2, 10, 3, 2, 8, torch.bfloat16),
                                             (1, 7, 2, 4, 16, torch.float32),
                                             (3, 5, 4, 1, 32, fp8),
                                             (4, 33, 1, 3, 4, torch.float16)):
        cache = [(torch.empty(batch, kv, seq, hd, dtype=dtype),
                  torch.empty(batch, kv, seq, hd, dtype=dtype)) for _ in range(layers)]
        want = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in cache)
        elem = torch.empty(0, dtype=dtype).element_size()
        got = ns.kv_cache_bytes(batch, seq, layers, kv, hd, bytes_per_elem=elem)
        assert int(got) == want, (
            f"kv_cache_bytes(batch={batch}, seq_len={seq}, n_layers={layers}, "
            f"n_kv_heads={kv}, head_dim={hd}, bytes_per_elem={elem}) = {got}, but "
            f"allocating the per-layer K and V tensors of shape "
            f"({batch}, {kv}, {seq}, {hd}) takes {want} bytes — don't forget the "
            f"factor 2 for K *and* V")
    got = ns.kv_cache_bytes(1, 4096, 32, 32, 128)
    assert int(got) == 2 ** 31, (
        f"Llama-2-7B at 4k context, batch 1, bf16 (default bytes_per_elem=2) needs "
        f"2·1·4096·32·32·128·2 = 2,147,483,648 bytes (2 GiB); got {got}")


# --------------------------------------------------------------------------- FLOPs

def check_training_flops_matches_flop_counter(ns):
    from torch.utils.flop_counter import FlopCounterMode
    torch.manual_seed(0)
    for dims, tokens in (((32, 64, 48, 16), 20), ((10, 30, 10), (3, 7))):
        net = nn.Sequential(*[nn.Linear(a, b, bias=False) for a, b in zip(dims, dims[1:])])
        shape = (tokens, dims[0]) if isinstance(tokens, int) else (*tokens, dims[0])
        x = torch.randn(*shape, requires_grad=True)
        with FlopCounterMode(display=False) as fc:
            net(x).sum().backward()
        measured = fc.get_total_flops()
        n_params = sum(p.numel() for p in net.parameters())
        n_tokens = x.numel() // dims[0]
        got = ns.training_flops(n_params, n_tokens)
        assert abs(float(got) - measured) <= 1e-9 * measured, (
            f"a forward + backward pass of a {n_params}-parameter matmul network over "
            f"{n_tokens} tokens costs {measured} FLOPs (measured with FlopCounterMode), "
            f"training_flops gives {got}. Each parameter is one multiply-add (2 FLOPs) "
            f"per token forward and two multiply-adds (grad wrt input and wrt weight) "
            f"backward")
    got = ns.training_flops(70e9, 1.4e12)
    assert abs(float(got) - 5.88e23) <= 1e-6 * 5.88e23, (
        f"Chinchilla (70B params, 1.4T tokens) is 6·N·D = 5.88e23 FLOPs, got {got}")


CHECKS = [
    check_transformer_params_matches_a_real_model,
    check_transformer_params_llama_scale,
    check_training_memory_matches_real_optimizer_state,
    check_training_memory_rejects_unknown_settings,
    check_kv_cache_bytes_matches_allocated_tensors,
    check_training_flops_matches_flop_counter,
]
