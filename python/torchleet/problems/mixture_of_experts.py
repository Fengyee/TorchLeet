"""mixture-of-experts — top-k routed MoE layer with load balancing.

Routing is read out without knowing how the solver dispatches tokens: forward
hooks replace expert e's output with the basis vector e, so the layer output
becomes each token's gate vector. From that the checks verify that every token
uses exactly top_k experts with gates that sum to 1, and that the real output
equals the gate-weighted sum of those experts computed one token at a time.

Checks that need the router (top-k selection, aux loss) find it as `router`, the
only d_model -> num_experts Linear, or the only (num_experts, d_model)-shaped
parameter outside the experts, and skip if there is none.
"""
from types import SimpleNamespace

import torch
import torch.nn as nn

from torchleet.runner import Skip

ENTRIES = ["Expert", "MoELayer"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Route each token to its top-k experts and weight their outputs by the gate probabilities.",
    "Renormalize the top-k gate values so each token's weights sum to 1 (a softmax over "
    "just the k selected logits). Dispatch per expert: find the (token, slot) pairs that "
    "chose it, run only those tokens, and index_add the weighted outputs back.",
    "Switch-Transformer aux loss: num_experts * sum_i f_i * P_i, where f_i is the fraction "
    "of routing assignments sent to expert i (a count, no gradient) and P_i is the mean "
    "softmax router probability of expert i (this carries the gradient). Uniform routing gives 1.",
]

B, S, D_MODEL, D_FF, N_EXP, TOP_K = 2, 8, 64, 128, 8, 2


def _cfg(**kw):
    base = dict(num_experts=N_EXP, top_k=TOP_K, d_model=D_MODEL, d_ff=D_FF,
                aux_loss_weight=0.01)
    base.update(kw)
    return SimpleNamespace(**base)


def _layer(ns, **kw):
    torch.manual_seed(0)
    return ns.MoELayer(_cfg(**kw))


def check_expert_shape(ns):
    e = ns.Expert(D_MODEL, D_FF)
    out = e(torch.randn(5, D_MODEL))
    assert tuple(out.shape) == (5, D_MODEL), \
        f"an expert must preserve d_model: expected {(5, D_MODEL)}, got {tuple(out.shape)}"


def check_layer_output_shape(ns):
    out = _layer(ns)(torch.randn(B, S, D_MODEL))
    out = out[0] if isinstance(out, tuple) else out
    assert tuple(out.shape) == (B, S, D_MODEL), \
        f"expected {(B, S, D_MODEL)}, got {tuple(out.shape)}"


def check_returns_aux_loss(ns):
    res = _layer(ns)(torch.randn(B, S, D_MODEL))
    assert isinstance(res, tuple) and len(res) == 2, \
        "MoELayer.forward should return (output, aux_loss)"
    aux = torch.as_tensor(res[1]).detach()
    assert aux.ndim == 0, f"aux_loss should be a scalar, got shape {tuple(aux.shape)}"
    assert float(aux) >= 0, f"aux_loss should be non-negative, got {float(aux):.4f}"


def check_creates_one_module_per_expert(ns):
    layer = _layer(ns, num_experts=5)
    found = sum(1 for m in layer.modules() if isinstance(m, ns.Expert))
    assert found == 5, f"config asked for 5 experts, layer built {found}"


def check_gradients_flow(ns):
    layer = _layer(ns)
    x = torch.randn(B, S, D_MODEL, requires_grad=True)
    res = layer(x)
    (res[0] if isinstance(res, tuple) else res).sum().backward()
    assert x.grad is not None and torch.any(x.grad != 0), \
        "no gradient reached the input — the routing path is detached"


def check_routing_is_input_dependent(ns):
    """A router that ignores its input is not routing."""
    layer = _layer(ns)
    a = layer(torch.randn(B, S, D_MODEL) * 5)
    b = layer(torch.randn(B, S, D_MODEL) * 5)
    a = a[0] if isinstance(a, tuple) else a
    b = b[0] if isinstance(b, tuple) else b
    assert not torch.allclose(a, b, atol=1e-4), \
        "different inputs produced identical outputs"


def _out(res):
    return res[0] if isinstance(res, tuple) else res


def _experts(ns, layer):
    return [m for m in layer.modules() if isinstance(m, ns.Expert)]


def _probe_gates(ns, layer, x):
    """(num_tokens, num_experts) gate weights, plus the output columns the experts
    cannot explain. Expert e is made to output the basis vector e, so the layer
    output is sum_e gate_e * basis_e whatever the dispatch strategy."""
    experts = _experts(ns, layer)
    fired = []

    def make_hook(e):
        def hook(module, inputs, out):
            fired.append(e)
            basis = torch.zeros_like(out)
            basis[..., e] = 1.0
            return basis
        return hook

    handles = [m.register_forward_hook(make_hook(e)) for e, m in enumerate(experts)]
    try:
        with torch.no_grad():
            out = _out(layer(x)).reshape(-1, x.shape[-1])
    finally:
        for h in handles:
            h.remove()
    if not fired:
        raise Skip("the experts are never called as modules, so routing cannot be probed")
    n = len(experts)
    return out[:, :n], out[:, n:]


class _Router:
    """The solver's router, whichever way it is stored."""

    def __init__(self, ns, layer):
        own = {id(p) for e in _experts(ns, layer) for p in e.parameters()}
        linears = [m for m in layer.modules() if isinstance(m, nn.Linear)
                   and m.in_features == D_MODEL and m.out_features == N_EXP
                   and id(m.weight) not in own]
        named = getattr(layer, "router", None)
        self.linear = self.param = None
        if any(named is m for m in linears):
            self.linear = named
        elif len(linears) == 1:
            self.linear = linears[0]
        else:
            params = [q for q in layer.parameters() if id(q) not in own and
                      tuple(q.shape) in ((N_EXP, D_MODEL), (D_MODEL, N_EXP))]
            if len(params) != 1:
                raise Skip("could not find the router (expected `self.router`, a "
                           "d_model -> num_experts nn.Linear)")
            self.param = params[0]

    def logits(self, x_flat):
        if self.linear is not None:
            return self.linear(x_flat)
        w = self.param
        return x_flat @ (w if w.shape[0] == D_MODEL else w.T)

    def weight(self):
        return self.linear.weight if self.linear is not None else self.param

    @torch.no_grad()
    def set(self, w_ed):
        """Overwrite with a (num_experts, d_model) matrix and zero any bias."""
        if self.linear is not None:
            self.linear.weight.copy_(w_ed)
            if self.linear.bias is not None:
                self.linear.bias.zero_()
        else:
            self.param.copy_(w_ed if self.param.shape[0] == N_EXP else w_ed.T)


def check_each_token_gets_top_k_gates_summing_to_one(ns):
    layer = _layer(ns).eval()
    torch.manual_seed(1)
    gates, rest = _probe_gates(ns, layer, torch.randn(B, S, D_MODEL))
    assert rest.abs().max() < 1e-5, (
        "with every expert returning a basis vector, the layer output still has other "
        "components — MoELayer should return only the gate-weighted sum of expert outputs")
    used = (gates != 0).sum(-1)
    assert torch.all(used == TOP_K), (
        f"every token should be sent to exactly top_k={TOP_K} experts; tokens used "
        f"{sorted(set(used.tolist()))} experts")
    assert torch.all(gates >= 0), "gate weights must be non-negative"
    sums = gates.sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
        f"each token's gate weights should sum to 1 (softmax over the selected top-k "
        f"logits); got sums in [{sums.min():.4f}, {sums.max():.4f}]")


def check_output_is_gate_weighted_sum_of_experts(ns):
    """Compare the real output against a slow per-token loop over the chosen experts."""
    layer = _layer(ns).eval()
    torch.manual_seed(2)
    x = torch.randn(B, S, D_MODEL)
    gates, _ = _probe_gates(ns, layer, x)
    experts = _experts(ns, layer)
    x_flat = x.reshape(-1, D_MODEL)
    with torch.no_grad():
        out = _out(layer(x)).reshape(-1, D_MODEL)
        ref = torch.zeros_like(x_flat)
        for t in range(x_flat.size(0)):
            for e in gates[t].nonzero().flatten().tolist():
                ref[t] += gates[t, e] * experts[e](x_flat[t:t + 1])[0]
    diff = (out - ref).abs().max().item()
    assert diff < 1e-5, (
        f"output differs from sum_e gate_e * expert_e(token), computed token by token "
        f"(max diff {diff:.2e}) — check that each expert's output is scattered back to "
        f"the right tokens with the weight of the right top-k slot")


def check_gates_follow_the_router_top_k(ns):
    layer = _layer(ns).eval()
    router = _Router(ns, layer)
    torch.manual_seed(3)
    x = torch.randn(B, S, D_MODEL)
    gates, _ = _probe_gates(ns, layer, x)
    with torch.no_grad():
        logits = router.logits(x.reshape(-1, D_MODEL))
        top_vals, top_idx = logits.topk(TOP_K, dim=-1)
        want = torch.zeros_like(logits).scatter(1, top_idx, top_vals.softmax(-1))
    chosen = (gates != 0)
    assert torch.equal(chosen, want != 0), (
        "tokens were not sent to the experts with the k largest router logits")
    diff = (gates - want).abs().max().item()
    assert diff < 1e-5, (
        f"gate weights should be the softmax over the k selected router logits "
        f"(max diff {diff:.2e})")


def _aux(layer, x):
    res = layer(x)
    assert isinstance(res, tuple) and len(res) == 2, \
        "MoELayer.forward should return (output, aux_loss)"
    return torch.as_tensor(res[1])


def check_aux_loss_is_switch_load_balancing(ns):
    """N * sum_i f_i * P_i: 1 for uniform routing (k if f counts per token),
    much larger when every token goes to the same experts."""
    layer = _layer(ns, aux_loss_weight=1.0).eval()
    router = _Router(ns, layer)
    torch.manual_seed(4)
    router.set(torch.zeros(N_EXP, D_MODEL))
    uniform = _aux(layer, torch.randn(B, S, D_MODEL)).item()
    assert min(abs(uniform - 1.0), abs(uniform - TOP_K)) < 1e-4, (
        f"with a uniform router (all logits equal) the Switch aux loss "
        f"num_experts * sum_i f_i * P_i is 1 (or top_k={TOP_K} if f_i counts tokens "
        f"rather than assignments); got {uniform:.4f}. Sum over experts, don't average.")
    v = torch.randn(D_MODEL)
    w = torch.zeros(N_EXP, D_MODEL)
    w[0] = v * (20.0 / v.dot(v))  # expert 0's logit is 20 for this token, others 0
    router.set(w)
    collapsed = _aux(layer, v.expand(B, S, D_MODEL).clone()).item()
    assert collapsed > 2 * uniform, (
        f"when every token is routed to the same experts the aux loss should rise well "
        f"above its uniform value ({uniform:.4f}); got {collapsed:.4f}")


def check_aux_loss_trains_the_router(ns):
    layer = _layer(ns)
    router = _Router(ns, layer)
    torch.manual_seed(5)
    aux = _aux(layer, torch.randn(B, S, D_MODEL))
    assert aux.requires_grad, (
        "aux_loss carries no gradient — compute P_i from the softmax router "
        "probabilities, not only from the (non-differentiable) top-k counts")
    aux.backward()
    g = router.weight().grad
    assert g is not None and g.abs().sum() > 0, (
        "aux_loss gives no gradient to the router — P_i (mean softmax probability "
        "per expert) is the differentiable factor")


CHECKS = [check_expert_shape, check_layer_output_shape, check_returns_aux_loss,
          check_creates_one_module_per_expert, check_gradients_flow,
          check_routing_is_input_dependent,
          check_each_token_gets_top_k_gates_summing_to_one,
          check_output_is_gate_weighted_sum_of_experts,
          check_gates_follow_the_router_top_k,
          check_aux_loss_is_switch_load_balancing,
          check_aux_loss_trains_the_router]
