"""lstm — the LSTM cell written out by hand, plus the nn.LSTM version.

The hand-rolled cell is not compared against a reference: gate order, parameter
layout and initialisation are all free. Instead it is driven two ways that must
agree — one pass over a whole sequence versus two passes with the returned
(H, C) threaded between them. Only a genuine recurrence with a correctly carried
state satisfies that.

The gate algebra is pinned black-box, one timestep at a time, by varying the
incoming cell state C0 while x and H0 stay fixed. For an LSTM (no peepholes, as
in nn.LSTM) C1 = f * C0 + i * g is affine in C0 with a slope f in (0, 1), the
remainder i * g lies in (-1, 1) and takes both signs (g is a tanh), and
H1 / tanh(C1) = o is a gate in (0, 1) that does not depend on C0. From a zero
state C1 = i * g, and driving x hard saturates the input gates shut for one
sign of x, so those units write ~0 — without an input gate a saturated tanh
writes ~+-1 either way. Together with |H| < 1 and "the gates read both x and H0",
that catches a dropped forget, input or output gate, a sigmoid where a tanh
belongs, and a missing recurrence. nn.LSTM / nn.LSTMCell (and the other built-in
recurrent kernels) are disabled while the custom cell runs.
"""
import torch
import torch.nn as nn
from torch.overrides import TorchFunctionMode

from torchleet.checks import common as c

ENTRIES = ["CustomLSTMModel", "LSTMModel"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "CustomLSTMModel(input_dim, hidden_units).forward(inputs, H_C=None) returns "
    "(predictions, (H, C)) with predictions shaped (batch, seq_len, 1).",
    "Per timestep: i, f, o = sigmoid(...), C~ = tanh(...), C = f*C + i*C~, "
    "H = o*tanh(C). Stack every H and push the stack through the Linear.",
    "When H_C is given, start from it instead of a fresh state — that is what "
    "makes the model resumable, and zeros (not randn) are the right default.",
]

FEAT, HID, SEQ = 1, 8, 6


def _custom(ns, seed=0, input_dim=FEAT, hidden=HID):
    torch.manual_seed(seed)
    return ns.CustomLSTMModel(input_dim, hidden)


def _zero_state(batch, hidden=HID):
    return torch.zeros(batch, hidden), torch.zeros(batch, hidden)


def _unpack(out, where):
    assert isinstance(out, (tuple, list)) and len(out) == 2, (
        f"{where}: forward should return (predictions, (H, C)), got "
        f"{type(out).__name__}")
    pred, state = out
    assert isinstance(state, (tuple, list)) and len(state) == 2, (
        f"{where}: the second return value should be the (H, C) pair, got "
        f"{type(state).__name__}")
    return pred, state


def check_custom_output_shape(ns):
    m = _custom(ns)
    m.eval()
    pred, (h, cell) = _unpack(m(torch.randn(3, SEQ, FEAT), _zero_state(3)), "CustomLSTMModel")
    assert tuple(pred.shape) == (3, SEQ, 1), (
        f"expected predictions of shape {(3, SEQ, 1)} — one value per timestep — "
        f"got {tuple(pred.shape)}")
    for name, t in (("H", h), ("C", cell)):
        assert tuple(t.shape) == (3, HID), (
            f"{name} should be (batch, hidden_units) = {(3, HID)}, got "
            f"{tuple(t.shape)}")


def check_custom_honours_its_hidden_size(ns):
    """hidden_units is a constructor argument, not a constant."""
    for hidden in (4, 16):
        m = _custom(ns, hidden=hidden)
        m.eval()
        _, (h, _) = _unpack(m(torch.randn(2, SEQ, FEAT), _zero_state(2, hidden)),
                            f"CustomLSTMModel(1, {hidden})")
        assert tuple(h.shape) == (2, hidden), (
            f"CustomLSTMModel(1, {hidden}) returned a hidden state of "
            f"{tuple(h.shape)}, expected {(2, hidden)}")


def check_custom_state_is_resumable(ns):
    """One pass over 6 steps must equal two passes of 3 with (H, C) threaded."""
    m = _custom(ns)
    m.eval()
    x = torch.randn(2, SEQ, FEAT)
    s0 = _zero_state(2)
    with torch.no_grad():
        full, (hf, cf) = _unpack(m(x, s0), "CustomLSTMModel")
        part1, s1 = _unpack(m(x[:, :3], s0), "CustomLSTMModel")
        part2, (hp, cp) = _unpack(m(x[:, 3:], tuple(s1)), "CustomLSTMModel")
    assert torch.allclose(full[:, :3], part1, atol=1e-5), (
        "the first 3 timesteps changed when the sequence was cut short — a "
        "timestep must only depend on the ones before it")
    assert torch.allclose(full[:, 3:], part2, atol=1e-5), (
        f"feeding steps 4-6 with the state returned after step 3 gave different "
        f"predictions than one pass over all {SEQ} steps (max diff "
        f"{(full[:, 3:] - part2).abs().max():.2e}) — the incoming H_C is being "
        "ignored or the state is not carried forward")
    assert torch.allclose(hf, hp, atol=1e-5) and torch.allclose(cf, cp, atol=1e-5), \
        "the final (H, C) differs between the one-pass and the resumed run"


def check_custom_hidden_state_is_bounded(ns):
    """H = o * tanh(C) with o in (0, 1), so |H| < 1 whatever the input is."""
    m = _custom(ns)
    m.eval()
    with torch.no_grad():
        _, (h, _) = _unpack(m(torch.randn(4, SEQ, FEAT) * 50, _zero_state(4)),
                            "CustomLSTMModel")
    assert torch.isfinite(h).all(), (
        "the hidden state is NaN/inf for large inputs — use torch.sigmoid / "
        "torch.tanh (a tanh written with exp() overflows)")
    assert float(h.abs().max()) <= 1.0 + 1e-4, (
        f"the hidden state reached {float(h.abs().max()):.3f}; H = o * tanh(C) "
        "with a sigmoid gate can never leave (-1, 1), so a squashing "
        "nonlinearity is missing")


def check_custom_default_state_is_zeros(ns):
    """H_C=None must mean a zero state — not a random one."""
    m = _custom(ns)
    m.eval()
    x = torch.randn(2, SEQ, FEAT)
    with torch.no_grad():
        a, _ = _unpack(m(x), "CustomLSTMModel (H_C=None)")
        b, _ = _unpack(m(x), "CustomLSTMModel (H_C=None)")
        z, _ = _unpack(m(x, _zero_state(2)), "CustomLSTMModel")
    assert torch.allclose(a, b, atol=1e-6), (
        "two calls with H_C=None on the same input disagree — the default initial "
        "state is random; start from zeros (like nn.LSTM) instead")
    assert torch.allclose(a, z, atol=1e-5), (
        "with H_C=None the predictions differ from an explicit all-zero (H, C) — "
        "the default initial state should be zeros")


def _one_step(m, x, h0, c0):
    with torch.no_grad():
        _, (h1, c1) = _unpack(m(x, (h0, c0)), "CustomLSTMModel")
    return h1, c1


def _step_probe(ns, batch=16):
    """One timestep from the same x and H0 but three different C0."""
    m = _custom(ns)
    m.eval()
    torch.manual_seed(1)
    x = torch.randn(batch, 1, FEAT)
    h0 = torch.rand(batch, HID) * 1.6 - 0.8
    c0 = torch.randn(batch, HID)
    outs = [_one_step(m, x, h0, c0 + d) for d in (0.0, 1.0, -2.0)]
    return m, x, h0, c0, outs


def check_custom_forget_gate_scales_the_old_cell(ns):
    """C1 = f * C0 + i * g: affine in C0, with a slope f strictly inside (0, 1)."""
    _, _, _, c0, ((_, ca), (_, cb), (_, cc)) = _step_probe(ns)
    f1 = cb - ca                 # C0 shifted by +1
    f2 = (ca - cc) / 2.0         # C0 shifted by -2
    assert torch.allclose(f1, f2, atol=1e-4), (
        "the new cell state is not an affine function of the old one — for fixed x "
        "and H, C_t = f_t * C_{t-1} + i_t * g_t, and no gate may read C_{t-1} "
        f"(max mismatch {(f1 - f2).abs().max():.2e})")
    assert float(f1.min()) > -1e-5 and float(f1.max()) < 1 + 1e-5, (
        f"dC_t/dC_(t-1) ranges over [{float(f1.min()):.3f}, {float(f1.max()):.3f}]; "
        "it is the forget gate, a sigmoid in (0, 1)")
    assert not torch.allclose(f1, torch.ones_like(f1), atol=1e-3), (
        "the old cell state passes through unscaled — the forget gate is missing "
        "(C_t = f_t * C_(t-1) + i_t * g_t)")
    rest = ca - f1 * c0           # = i * g
    assert float(rest.abs().max()) < 1 + 1e-4, (
        f"C_t - f_t * C_(t-1) reached {float(rest.abs().max()):.3f}; i_t * g_t is a "
        "sigmoid times a tanh and stays inside (-1, 1)")
    assert (rest < -1e-3).any() and (rest > 1e-3).any(), (
        "the write i_t * g_t never changes sign — the candidate g_t should be a "
        "tanh (values in (-1, 1)), not a sigmoid or ReLU")


def check_custom_output_gate_reads_tanh_of_the_cell(ns):
    """H1 = o * tanh(C1) with o in (0, 1) independent of C0."""
    _, _, _, _, ((ha, ca), (hb, cb), (hc, cc)) = _step_probe(ns)
    for h, cell in ((ha, ca), (hb, cb), (hc, cc)):
        t = torch.tanh(cell)
        assert float((h * t).min()) >= -1e-6 and bool((h.abs() <= t.abs() + 1e-5).all()), (
            "H_t is not a (0, 1)-gated copy of tanh(C_t) — expected H_t = o_t * tanh(C_t)")
    ta, tb, tc = torch.tanh(ca), torch.tanh(cb), torch.tanh(cc)
    assert torch.allclose(ha * tb, hb * ta, atol=1e-5) and \
        torch.allclose(ha * tc, hc * ta, atol=1e-5), (
        "H_t / tanh(C_t) changes when only C_(t-1) changes — the output gate must "
        "depend on x_t and H_(t-1) only, and H_t = o_t * tanh(C_t)")
    assert not torch.allclose(ha, ta, atol=1e-4), \
        "H_t equals tanh(C_t) — the output gate is missing"


def check_custom_input_gate_scales_the_write(ns):
    """C1 = i * g from a zero state: each unit's input gate saturates shut for one
    sign of a large x, so its write is ~0 there. Without an input gate (C1 = g) a
    saturated tanh writes ~+-1 for both signs."""
    hidden = 32
    m = _custom(ns, hidden=hidden)
    m.eval()
    # The largest input that stays finite (a hand-written exp() tanh can overflow);
    # the larger it is, the harder the gates saturate, whatever the init scale.
    for scale in (5000.0, 500.0, 50.0, 10.0):
        x = torch.full((2, 1, FEAT), scale)
        x[1] = -scale
        _, c1 = _one_step(m, x, *_zero_state(2, hidden))
        if torch.isfinite(c1).all():
            break
    else:
        raise AssertionError(
            "the cell state is NaN/inf for large inputs — use torch.sigmoid / "
            "torch.tanh (a tanh written with exp() overflows)")
    shut = (c1.abs().min(dim=0).values < 0.05).float().mean().item()
    assert shut >= 0.25, (
        f"from a zero state C_t = i_t * g_t; with x = +-{scale:g} some input gates "
        f"should close (i_t ~ 0) and block the write, but only {shut:.0%} of the "
        "units wrote ~0 for either sign — is C_t = f_t * C_(t-1) + g_t missing the "
        "input gate i_t?")


_RECURRENT_KERNELS = {"lstm", "lstm_cell", "gru", "gru_cell", "rnn_tanh", "rnn_relu",
                      "rnn_tanh_cell", "rnn_relu_cell", "_thnn_fused_lstm_cell",
                      "_thnn_fused_gru_cell"}


class _NoBuiltinRNN(TorchFunctionMode):
    """Fails on any call to a built-in recurrent kernel, however it is reached
    (nn.LSTM, nn.LSTMCell, torch._VF.lstm_cell, ...)."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        name = getattr(func, "__name__", "")
        if name in _RECURRENT_KERNELS:
            raise AssertionError(
                f"CustomLSTMModel called the built-in `{name}` kernel — write the gate "
                "equations yourself; nn.LSTM belongs only in LSTMModel")
        return func(*args, **(kwargs or {}))


def check_custom_does_not_wrap_nn_lstm(ns):
    m = _custom(ns)
    m.eval()
    with torch.no_grad(), _NoBuiltinRNN():
        m(torch.randn(2, SEQ, FEAT), _zero_state(2))


def check_custom_gates_read_input_and_hidden(ns):
    """The gates are functions of both x_t and H_(t-1)."""
    m, x, h0, c0, ((_, ca), _, _) = _step_probe(ns)
    _, c_h = _one_step(m, x, h0 * -1.0 + 0.1, c0)
    _, c_x = _one_step(m, x + 1.0, h0, c0)
    assert float((c_h - ca).abs().max()) > 1e-4, (
        "changing H_(t-1) did not change C_t — the gates ignore the previous hidden "
        "state, so this is not a recurrent cell")
    assert float((c_x - ca).abs().max()) > 1e-4, \
        "changing x_t did not change C_t — the gates ignore the input"


def check_custom_gradients_reach_all_parameters(ns):
    m = _custom(ns)
    pred, _ = _unpack(m(torch.randn(2, SEQ, FEAT), _zero_state(2)), "CustomLSTMModel")
    pred.sum().backward()
    dead = [n for n, p in m.named_parameters()
            if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
    assert not dead, (
        f"no gradient reached: {', '.join(dead[:4])}"
        f"{'...' if len(dead) > 4 else ''} — every gate weight should be on the "
        "forward path (and registered as an nn.Parameter)")


def check_custom_can_learn(ns):
    torch.manual_seed(0)
    m = _custom(ns)
    x = torch.randn(8, SEQ, FEAT)
    y = torch.randn(8, 1)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    lossf = nn.MSELoss()

    def step():
        pred, _ = _unpack(m(x, _zero_state(8)), "CustomLSTMModel")
        return lossf(pred[:, -1, :], y)

    first = step().item()
    loss = None
    for _ in range(60):
        opt.zero_grad()
        loss = step()
        loss.backward()
        opt.step()
    assert loss.item() < first, (
        f"loss did not decrease over 60 steps on a fixed batch "
        f"({first:.4f} -> {loss.item():.4f}) — the custom cell is not learning")


def check_inbuilt_output_shape(ns):
    torch.manual_seed(0)
    m = ns.LSTMModel()
    m.eval()
    out = m(torch.randn(5, SEQ, FEAT))
    assert tuple(out.shape) == (5, 1), (
        f"LSTMModel should return one prediction per sequence — {(5, 1)} for a "
        f"(5, {SEQ}, {FEAT}) input — got {tuple(out.shape)}")


def check_inbuilt_any_sequence_length(ns):
    torch.manual_seed(0)
    m = ns.LSTMModel()
    m.eval()
    for t in (3, 10, 25):
        out = m(torch.randn(2, t, FEAT))
        assert tuple(out.shape) == (2, 1), (
            f"sequence length {t} gave {tuple(out.shape)}, expected {(2, 1)} — "
            "an LSTM accepts any number of timesteps")


CHECKS = [
    check_custom_output_shape,
    check_custom_honours_its_hidden_size,
    check_custom_state_is_resumable,
    check_custom_hidden_state_is_bounded,
    check_custom_default_state_is_zeros,
    check_custom_forget_gate_scales_the_old_cell,
    check_custom_output_gate_reads_tanh_of_the_cell,
    check_custom_input_gate_scales_the_write,
    check_custom_gates_read_input_and_hidden,
    check_custom_does_not_wrap_nn_lstm,
    check_custom_gradients_reach_all_parameters,
    check_custom_can_learn,
    check_inbuilt_output_shape,
    check_inbuilt_any_sequence_length,
    c.gradients_flow("LSTMModel", (SEQ, FEAT)),
    c.can_learn("LSTMModel", (SEQ, FEAT), (1,), steps=60, lr=1e-2),
]
