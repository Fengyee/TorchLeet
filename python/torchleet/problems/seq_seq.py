"""seq-seq — an attention seq2seq model: LSTM encoder + Bahdanau-attention decoder.

The decoder consumes three things from the encoder: per-timestep outputs to
attend over, and the final (hidden, cell) pair to start decoding from. The
encoder checks pin exactly that interface and then use the one invariant that
ties the two return values together — for a left-to-right LSTM the last row of
`outputs` *is* the top layer's final hidden state. An encoder that returns the
states of the wrong layer, forgets batch_first, or hands back the embeddings
instead of the LSTM outputs fails it.

The decoder is checked through the property that defines content-based
attention: the context is a softmax-weighted sum over source positions scored
by their content, so shuffling the encoder outputs along the source axis must
not change the decoder's step at all, and any source length must work. A
"location" attention that predicts a fixed number of weights from the decoder
state alone (Linear(..., src_len)) fails both.
"""
import torch
import torch.nn as nn

ENTRIES = ["Encoder", "Decoder"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "Encoder(input_dim, embed_dim, hidden_dim, num_layers): nn.Embedding over the "
    "source vocabulary, then nn.LSTM(embed_dim, hidden_dim, num_layers, "
    "batch_first=True).",
    "forward returns (outputs, (hidden, cell)) — the attention needs every "
    "timestep, the decoder needs the final states to start from.",
    "With batch_first=True outputs is (batch, src_len, hidden_dim) while hidden "
    "and cell stay (num_layers, batch, hidden_dim).",
    "Decoder attention (Bahdanau): score every encoder output h_i against the "
    "decoder's top-layer state s = hidden[-1] with e_i = v^T tanh(W_s s + "
    "W_h h_i), giving (batch, src_len) scores; softmax over the source axis. "
    "Nothing in the decoder may assume a fixed source length.",
    "context = torch.bmm(weights.unsqueeze(1), encoder_outputs) is "
    "(batch, 1, hidden_dim). Concatenate it with the embedded token, project "
    "back to embed_dim with tanh, run ONE LSTM step from (hidden, cell), and "
    "map the LSTM output to vocabulary logits.",
]

VOCAB, EMBED, HIDDEN, LAYERS, SRC = 20, 12, 16, 2, 7


def _encoder(ns, seed=0, vocab=VOCAB, embed=EMBED, hidden=HIDDEN, layers=LAYERS):
    torch.manual_seed(seed)
    return ns.Encoder(vocab, embed, hidden, layers)


def _run(enc, batch=3, src=SRC, vocab=VOCAB):
    x = torch.randint(0, vocab, (batch, src))
    out = enc(x)
    assert isinstance(out, (tuple, list)) and len(out) == 2, (
        "Encoder.forward should return (outputs, (hidden, cell)), got "
        f"{type(out).__name__}"
        + (f" of length {len(out)}" if hasattr(out, "__len__") else ""))
    outputs, state = out
    assert isinstance(state, (tuple, list)) and len(state) == 2, (
        "the second return value should be the (hidden, cell) pair from the "
        f"LSTM, got {type(state).__name__}")
    return outputs, state[0], state[1]


def check_returns_outputs_and_states(ns):
    enc = _encoder(ns)
    enc.eval()
    outputs, hidden, cell = _run(enc)
    assert tuple(outputs.shape) == (3, SRC, HIDDEN), (
        f"outputs should be (batch, src_len, hidden_dim) = {(3, SRC, HIDDEN)} so "
        f"attention can score every source position, got {tuple(outputs.shape)}"
        + (" — pass batch_first=True to nn.LSTM"
           if tuple(outputs.shape) == (SRC, 3, HIDDEN) else ""))
    for name, t in (("hidden", hidden), ("cell", cell)):
        assert tuple(t.shape) == (LAYERS, 3, HIDDEN), (
            f"{name} should be (num_layers, batch, hidden_dim) = "
            f"{(LAYERS, 3, HIDDEN)}, got {tuple(t.shape)}")


def check_last_output_is_the_final_hidden_state(ns):
    """A left-to-right LSTM ends its last step in exactly h_n[-1]."""
    enc = _encoder(ns)
    enc.eval()
    with torch.no_grad():
        outputs, hidden, _ = _run(enc)
    assert torch.allclose(outputs[:, -1, :], hidden[-1], atol=1e-5), (
        "outputs[:, -1, :] does not equal hidden[-1] (max diff "
        f"{(outputs[:, -1, :] - hidden[-1]).abs().max():.2e}) — the two return "
        "values do not come from the same single-direction LSTM pass, or the "
        "batch and time axes are swapped")


def check_accepts_any_batch_and_source_length(ns):
    enc = _encoder(ns)
    enc.eval()
    for batch, src in ((1, 4), (5, 7), (2, 15)):
        outputs, hidden, cell = _run(enc, batch=batch, src=src)
        assert tuple(outputs.shape) == (batch, src, HIDDEN), (
            f"a ({batch}, {src}) source batch gave outputs {tuple(outputs.shape)}, "
            f"expected {(batch, src, HIDDEN)}")
        assert tuple(hidden.shape) == (LAYERS, batch, HIDDEN), \
            f"hidden was {tuple(hidden.shape)}, expected {(LAYERS, batch, HIDDEN)}"


def check_honours_its_constructor_arguments(ns):
    """hidden_dim and num_layers are arguments, not constants."""
    enc = _encoder(ns, hidden=32, layers=1)
    enc.eval()
    outputs, hidden, _ = _run(enc)
    assert outputs.shape[-1] == 32, (
        f"Encoder(..., hidden_dim=32, num_layers=1) produced outputs with "
        f"{outputs.shape[-1]} features, expected 32")
    assert hidden.shape[0] == 1, (
        f"num_layers=1 should give a hidden state with a leading 1, got "
        f"{tuple(hidden.shape)}")


def check_embeds_the_whole_vocabulary(ns):
    """input_dim is the source vocabulary size, so every id below it must work."""
    enc = _encoder(ns)
    enc.eval()
    ids = torch.arange(VOCAB).unsqueeze(0)          # 0 .. VOCAB-1
    with torch.no_grad():
        outputs, _, _ = _run(enc, batch=1, src=VOCAB)   # shape sanity first
        a, _ = enc(ids)
        b, _ = enc(ids.clone())
        changed, _ = enc(torch.cat([ids[:, :1] * 0 + (VOCAB - 1), ids[:, 1:]], dim=1))
    assert torch.allclose(a, b, atol=1e-6), \
        "the same token ids gave two different encodings"
    assert not torch.allclose(a, changed, atol=1e-6), (
        "changing the first source token left the encoding unchanged — the input "
        "ids are not reaching the embedding")


def check_gradients_reach_all_parameters(ns):
    enc = _encoder(ns)
    outputs, hidden, cell = _run(enc, batch=2)
    (outputs.sum() + hidden.sum() + cell.sum()).backward()
    dead = [n for n, p in enc.named_parameters()
            if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
    assert not dead, (
        f"no gradient reached: {', '.join(dead[:4])}"
        f"{'...' if len(dead) > 4 else ''} — these layers are not on the forward path")


def check_can_learn(ns):
    """Overfit one fixed batch through a tiny read-out head; loss must fall."""
    torch.manual_seed(0)
    enc = _encoder(ns)
    head = nn.Linear(HIDDEN, 1)
    x = torch.randint(0, VOCAB, (8, SRC))
    y = torch.randn(8, 1)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=1e-2)
    lossf = nn.MSELoss()

    def step():
        outputs, _ = enc(x)
        return lossf(head(outputs[:, -1, :]), y)

    first = step().item()
    loss = None
    for _ in range(40):
        opt.zero_grad()
        loss = step()
        loss.backward()
        opt.step()
    assert loss.item() < first, (
        f"loss did not decrease over 40 steps on a fixed batch "
        f"({first:.4f} -> {loss.item():.4f}) — the encoder is not trainable")


# --------------------------------------------------------------- decoder

TGT_VOCAB = 17


def _decoder(ns, seed=0, hidden=HIDDEN, layers=LAYERS):
    torch.manual_seed(seed)
    return ns.Decoder(TGT_VOCAB, EMBED, hidden, layers)


def _decoder_inputs(batch=3, src=SRC, hidden=HIDDEN, layers=LAYERS, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, TGT_VOCAB, (batch,), generator=g)
    enc_out = torch.randn(batch, src, hidden, generator=g)
    h = torch.randn(layers, batch, hidden, generator=g)
    c = torch.randn(layers, batch, hidden, generator=g)
    return x, enc_out, h, c


def _decode(dec, x, enc_out, h, c):
    out = dec(x, enc_out, h, c)
    assert isinstance(out, (tuple, list)) and len(out) == 3, (
        "Decoder.forward(x, encoder_outputs, hidden, cell) should return "
        "(output, hidden, cell)")
    return out


def check_decoder_step_shapes_for_any_source_length(ns):
    dec = _decoder(ns)
    dec.eval()
    for src in (3, SRC, 13):
        x, enc_out, h, c = _decoder_inputs(src=src)
        try:
            out, h2, c2 = _decode(dec, x, enc_out, h, c)
        except RuntimeError as e:
            raise AssertionError(
                f"the decoder failed on a source of length {src} ({e}) — real "
                "attention scores every encoder output, so it must not depend on "
                "a fixed src_len")
        assert tuple(out.shape) == (3, TGT_VOCAB), (
            f"output should be (batch, output_dim) = {(3, TGT_VOCAB)} logits, "
            f"got {tuple(out.shape)}")
        for name, t in (("hidden", h2), ("cell", c2)):
            assert tuple(t.shape) == (LAYERS, 3, HIDDEN), (
                f"the returned {name} should be (num_layers, batch, hidden_dim) = "
                f"{(LAYERS, 3, HIDDEN)}, got {tuple(t.shape)}")


def check_decoder_attention_is_content_based(ns):
    """softmax-weighted sum over positions scored by content: permuting the
    source positions permutes scores and values together and changes nothing."""
    dec = _decoder(ns)
    dec.eval()
    x, enc_out, h, c = _decoder_inputs(seed=2)
    perm = torch.randperm(SRC, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        out1, h1, _ = _decode(dec, x, enc_out, h, c)
        out2, h2, _ = _decode(dec, x, enc_out[:, perm], h, c)
    dev = float((out1 - out2).abs().max())
    assert dev < 1e-5 and torch.allclose(h1, h2, atol=1e-5), (
        f"shuffling the encoder outputs along the source axis changed the "
        f"decoder's step (max diff {dev:.2e}). Attention weights must come from "
        "scoring each encoder output against the decoder state (e.g. "
        "v^T tanh(W_s s + W_h h_i)), not from a layer that predicts one weight "
        "per fixed source position")


def check_decoder_reads_the_encoder_outputs(ns):
    dec = _decoder(ns)
    dec.eval()
    x, enc_out, h, c = _decoder_inputs(seed=4)
    with torch.no_grad():
        out1, _, _ = _decode(dec, x, enc_out, h, c)
        out2, _, _ = _decode(dec, x, enc_out + torch.randn_like(enc_out), h, c)
    assert not torch.allclose(out1, out2, atol=1e-6), (
        "changing the encoder outputs left the decoder's prediction unchanged — "
        "the context vector is not reaching the output")


def check_decoder_gradients_reach_all_parameters(ns):
    dec = _decoder(ns)
    x, enc_out, h, c = _decoder_inputs(seed=5)
    enc_out.requires_grad_(True)
    out, h2, c2 = _decode(dec, x, enc_out, h, c)
    (out.sum() + h2.sum() + c2.sum()).backward()
    dead = [n for n, p in dec.named_parameters()
            if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
    assert not dead, (
        f"no gradient reached: {', '.join(dead[:4])}"
        f"{'...' if len(dead) > 4 else ''} — these layers are not on the forward path")
    assert enc_out.grad is not None and torch.any(enc_out.grad != 0), \
        "no gradient flows back into the encoder outputs through the attention"


def check_seq2seq_can_learn_with_teacher_forcing(ns):
    """Overfit a fixed batch (copy task) with teacher forcing; loss must fall."""
    torch.manual_seed(0)
    enc = _encoder(ns, vocab=TGT_VOCAB)
    dec = _decoder(ns)
    src = torch.randint(1, TGT_VOCAB, (8, 5))
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-2)
    lossf = nn.CrossEntropyLoss()

    def step():
        enc_out, (h, c) = enc(src)
        inp = torch.zeros(8, dtype=torch.long)
        loss = 0
        for t in range(src.shape[1]):
            out, h, c = dec(inp, enc_out, h, c)
            loss = loss + lossf(out, src[:, t])
            inp = src[:, t]
        return loss / src.shape[1]

    first = step().item()
    loss = None
    for _ in range(40):
        opt.zero_grad()
        loss = step()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.8, (
        f"encoder + decoder could not fit a fixed copy batch with teacher "
        f"forcing over 40 steps ({first:.4f} -> {loss.item():.4f})")


CHECKS = [
    check_returns_outputs_and_states,
    check_last_output_is_the_final_hidden_state,
    check_accepts_any_batch_and_source_length,
    check_honours_its_constructor_arguments,
    check_embeds_the_whole_vocabulary,
    check_gradients_reach_all_parameters,
    check_can_learn,
    check_decoder_step_shapes_for_any_source_length,
    check_decoder_attention_is_content_based,
    check_decoder_reads_the_encoder_outputs,
    check_decoder_gradients_reach_all_parameters,
    check_seq2seq_can_learn_with_teacher_forcing,
]
