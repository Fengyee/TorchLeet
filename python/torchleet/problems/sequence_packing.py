"""sequence-packing — pack variable-length documents into fixed-length rows without
letting them see each other.

Conventions (all stated in the notebook):

  * pack_sequences(seqs, max_len, pad_id=0) is first-fit decreasing (FFD) bin
    packing with a fully deterministic tie-break: visit the sequences in the order
    sorted(range(n), key=lambda i: (-len(seqs[i]), i)) and put each one into the
    lowest-index row with enough free space, opening a new row at the end when
    none fits. Inside a row, documents are laid out left to right in the order
    they were placed, starting at column 0 with no gaps; the tail is padding.
    Every length must satisfy 1 <= len <= max_len, else ValueError.
    FFD's decisions depend only on the sequence of lengths it visits, so the
    tie-break among EQUAL lengths changes which document fills a slot but never
    which lengths share a row. The checks therefore compare every row's document
    LENGTHS (left to right) exactly against a reference FFD (written below) and
    accept any order among equal lengths: torch.argsort(..., descending=True)
    without stable=True, or numpy's default argsort, are valid FFD too. (The
    notebook still prescribes the (-length, index) order, for a reproducible
    layout.) FFD is a heuristic: on [4, 4, 3, 3, 3, 3] with
    max_len 10 it uses 3 rows although 2 suffice ({4, 3, 3} twice); its
    worst-case guarantee is FFD <= 11/9 OPT + 6/9 (Dosa 2007).
  * Outputs: input_ids (R, max_len) int64 (pad_id on padding); position_ids
    (R, max_len) int64, 0..n-1 inside every document and 0 on padding; doc_ids
    (R, max_len) int64 = the document's index in `seqs`, -1 on padding;
    cu_seqlens (num_docs + 1,) int32 in the flash-attn varlen convention
    (flash_attn_varlen_func docstring: "cu_seqlens_q: (batch_size + 1,), dtype
    torch.int32. The cumulative sequence lengths of the sequences in the batch,
    used to index into q", with q of shape (total_q, nheads, headdim)). The token
    stream it indexes is the unpadded, flattened batch input_ids[doc_ids >= 0]
    (row-major, the token order flash_attn.bert_padding.unpad_input produces), with
    one segment per DOCUMENT: cu_seqlens[0] = 0, segment k is the k-th document in
    that order, cu_seqlens[-1] = total real tokens. This is what flash-attn's
    unpad_input_for_concatenated_sequences returns for packed rows; plain
    unpad_input would give one segment per ROW, letting documents see each other.
  * packed_labels(input_ids, doc_ids): already-shifted next-token targets, i.e.
    the logits at position t are trained on labels[t]:
    labels[r, t] = input_ids[r, t + 1] if t + 1 < L and doc_ids[r, t + 1] ==
    doc_ids[r, t] != -1, else -100 (F.cross_entropy's default ignore_index). So
    the last token of every document, every padding position and the last column
    are ignored. (Hugging Face models take UNshifted labels and shift inside the
    loss; its DataCollatorWithFlattening gets the same effect by writing -100 at
    the first token of every packed sequence.)
  * document_causal_mask(doc_ids) -> (R, L, L) bool, True = may attend (the
    boolean attn_mask convention of F.scaled_dot_product_attention):
    mask[r, i, j] = (doc_ids[r, i] == doc_ids[r, j]) and (j <= i). Padding is
    treated as one more document (id -1), so a padding query attends to the
    padding keys at or before it, every row contains at least its diagonal, and a
    softmax over a row is never 0/0 = NaN; no real token ever attends to padding.
    This is the `document_masking` mask_mod of the PyTorch FlexAttention blog
    post (document_id[q_idx] == document_id[kv_idx]) AND-ed with a causal mask.

The key property check runs a small pre-LN transformer layer with LEARNED
absolute position embeddings (so position ids matter; with RoPE alone a correct
document mask would hide non-reset positions, because RoPE scores depend only on
relative offsets) and hand-written masked softmax attention (so an all-False row
gives NaN), and requires the packed batch to reproduce, token for token, the
logits and next-token losses of every document run on its own, with finite
logits and gradients everywhere.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ENTRIES = ["pack_sequences", "packed_labels", "document_causal_mask"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Packing is a bin-packing problem: rows are bins of capacity max_len. Sort longest "
    "first, then each document goes into the first row that still has room. Everything "
    "else (positions, labels, mask) can be derived from doc_ids, the per-token "
    "document index.",
    "Keep a list of row contents and a list of free space per row; a document of "
    "length n goes to the lowest r with free[r] >= n. cu_seqlens is a 0 followed by the "
    "cumulative sum of the document lengths in row-major order, as int32. Labels: "
    "compare each doc id with its right neighbour (a shifted slice of doc_ids). Mask: "
    "broadcast a (R, L, 1) view of doc_ids against a (R, 1, L) view.",
    "Labels: start from a tensor full of -100; for every column but the last, copy the "
    "NEXT token wherever the next position belongs to the same document and the current "
    "position is not padding. Mask: AND the same-document comparison with a lower-"
    "triangular (causal) L x L matrix. Keep the comparison -1 == -1 as True — padding "
    "then forms its own causal block, so every row has at least its diagonal and no "
    "softmax row is all -inf.",
]

IGNORE = -100


# ----------------------------------------------------------------- references

def _ffd_rows(lengths, max_len):
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    rows, free = [], []
    for i in order:
        for r in range(len(rows)):
            if free[r] >= lengths[i]:
                rows[r].append(i)
                free[r] -= lengths[i]
                break
        else:
            rows.append([i])
            free.append(max_len - lengths[i])
    return rows


def _ref_pack(seqs, max_len, pad_id):
    rows = _ffd_rows([len(s) for s in seqs], max_len)
    R = len(rows)
    ids = torch.full((R, max_len), pad_id, dtype=torch.long)
    pos = torch.zeros(R, max_len, dtype=torch.long)
    doc = torch.full((R, max_len), -1, dtype=torch.long)
    lens = []
    for r, row in enumerate(rows):
        t = 0
        for i in row:
            n = len(seqs[i])
            ids[r, t:t + n], pos[r, t:t + n], doc[r, t:t + n] = seqs[i], torch.arange(n), i
            lens.append(n)
            t += n
    cu = torch.tensor([0] + lens, dtype=torch.long).cumsum(0).to(torch.int32)
    return ids, pos, doc, cu, rows


def _ref_labels(ids, doc):
    R, L = ids.shape
    out = torch.full_like(ids, IGNORE)
    for r in range(R):
        for t in range(L - 1):
            if doc[r, t] != -1 and doc[r, t + 1] == doc[r, t]:
                out[r, t] = ids[r, t + 1]
    return out


def _ref_mask(doc):
    R, L = doc.shape
    m = torch.zeros(R, L, L, dtype=torch.bool)
    for r in range(R):
        for i in range(L):
            for j in range(i + 1):
                m[r, i, j] = bool(doc[r, i] == doc[r, j])
    return m


def _make_seqs(lengths, vocab=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(1, vocab, (n,), generator=g) for n in lengths]


def _random_lengths(n, max_len, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, max_len + 1, (n,), generator=g).tolist()


def _pack(ns, seqs, max_len, pad_id=0):
    out = ns.pack_sequences(seqs, max_len, pad_id)
    assert isinstance(out, (tuple, list)) and len(out) == 4, (
        "pack_sequences should return a tuple (input_ids, position_ids, doc_ids, cu_seqlens)")
    return out


def _rows_of(doc):
    """Documents of each row in left-to-right order, from a doc_ids tensor."""
    rows = []
    for row in doc.tolist():
        ids = []
        for v in row:
            if v != -1 and (not ids or ids[-1] != v):
                ids.append(v)
        rows.append(ids)
    return rows


# ----------------------------------------------------------------- packing

def check_packing_is_valid(ns):
    cases = [(_random_lengths(13, 16, s), 16, s) for s in range(4)] + [
        ([5, 5, 5], 5, 10), ([1] * 7, 3, 11), ([9], 12, 12)]
    for lengths, max_len, seed in cases:
        seqs = _make_seqs(lengths, seed=seed)
        ids, pos, doc, cu = _pack(ns, seqs, max_len, pad_id=0)
        for name, t in (("input_ids", ids), ("position_ids", pos), ("doc_ids", doc)):
            assert torch.is_tensor(t) and t.dim() == 2 and t.shape[1] == max_len, (
                f"{name} should be a (R, max_len={max_len}) tensor, got "
                f"{tuple(t.shape) if torch.is_tensor(t) else type(t).__name__}")
            assert t.dtype == torch.long, f"{name} should be int64 (torch.long), got {t.dtype}"
        assert ids.shape == pos.shape == doc.shape, "input_ids, position_ids and doc_ids differ in shape"
        seen = set()
        for r in range(doc.shape[0]):
            row = doc[r]
            n_real = int((row >= 0).sum())
            assert n_real > 0, f"row {r} is pure padding — never open an empty row"
            assert bool((row[:n_real] >= 0).all()) and bool((row[n_real:] == -1).all()), (
                f"row {r}: documents must sit left-aligned from column 0 with no gaps, and "
                f"padding (doc_id -1) only at the end of the row; got doc_ids {row.tolist()}")
            assert bool((ids[r, n_real:] == 0).all()), (
                f"row {r}: padding positions must hold pad_id")
        for i, s in enumerate(seqs):
            where = (doc == i).nonzero()
            assert len(where) == len(s), (
                f"sequence {i} (length {len(s)}) occupies {len(where)} positions — every "
                f"sequence must appear exactly once, untruncated")
            rows = where[:, 0].unique()
            assert len(rows) == 1, f"sequence {i} is split across rows {rows.tolist()}"
            r = int(rows[0])
            cols = where[:, 1]
            assert int(cols.max() - cols.min()) + 1 == len(s), (
                f"sequence {i} is not contiguous in row {r} (columns {cols.tolist()})")
            assert torch.equal(ids[r, cols], s), (
                f"the tokens at sequence {i}'s positions are not seqs[{i}] in order")
            seen.add(i)
        assert set(doc[doc >= 0].unique().tolist()) == seen, (
            "doc_ids contains ids that are not indices into seqs")


def check_ffd_layout_is_exact(ns):
    """FFD row structure, compared by LENGTHS: any tie-break among equal lengths passes
    (it only swaps which equal-length document fills a slot); validity is checked above."""
    cases = [
        ([4, 4, 3, 3, 3, 3], 10, "FFD is not optimal here (2 rows would do) — but the "
                                  "spec asks for FFD, which gives 3"),
        ([3, 4, 6, 8, 1, 4, 4, 5, 3], 10, ""),
        ([2, 5, 4, 7, 1, 3, 8, 6, 2, 4], 10, ""),
        ([3, 3, 3, 3, 4, 4], 7, ""),
        (_random_lengths(40, 32, 7), 32, ""),
        (_random_lengths(25, 9, 8), 9, ""),
    ]
    for lengths, max_len, note in cases:
        seqs = _make_seqs(lengths, seed=len(lengths))
        ids, pos, doc, cu = _pack(ns, seqs, max_len)
        rows = _ffd_rows(lengths, max_len)
        got = _rows_of(doc)
        assert len(got) == len(rows), (
            f"lengths {lengths}, max_len {max_len}: you used {len(got)} rows, first-fit "
            f"decreasing uses {len(rows)}. Sort by length (longest first) and put each "
            f"document into the FIRST row with enough room, opening a row only when none "
            f"fits" + (f" ({note})" if note else ""))
        got_lens = [[lengths[i] for i in row] for row in got]
        want_lens = [[lengths[i] for i in row] for row in rows]
        assert got_lens == want_lens, (
            f"lengths {lengths}, max_len {max_len}: the document lengths of your rows (left "
            f"to right) are {got_lens}, first-fit decreasing gives {want_lens}. Visit "
            f"documents longest first, place each into the lowest-index row that fits (first "
            f"fit, not best fit), and lay out a row's documents in the order they were placed"
            + (f" — {note}" if note else ""))


def check_position_ids_restart_per_document(ns):
    for seed in range(3):
        lengths = _random_lengths(12, 10, 20 + seed)
        seqs = _make_seqs(lengths, seed=seed)
        ids, pos, doc, cu = _pack(ns, seqs, 10)
        for i, s in enumerate(seqs):
            got = pos[doc == i]
            if len(got) and int(got[0]) != 0:
                raise AssertionError(
                    f"document {i}'s position ids are {got.tolist()} — they must restart at 0 "
                    f"for every document (the model must see each document exactly as if it "
                    f"were alone, so positions count from its own start, not the row's)")
            assert torch.equal(got, torch.arange(len(s))), (
                f"document {i}'s position ids should be 0..{len(s) - 1}, got {got.tolist()}")
        assert bool((pos[doc == -1] == 0).all()), "position_ids on padding should be 0"


def check_cu_seqlens_flash_attn_convention(ns):
    for seed in range(4):
        lengths = _random_lengths(11, 12, 40 + seed)
        seqs = _make_seqs(lengths, seed=seed)
        ids, pos, doc, cu = _pack(ns, seqs, 12)
        order = [i for row in _rows_of(doc) for i in row]
        want = torch.tensor([0] + [lengths[i] for i in order]).cumsum(0)
        assert torch.is_tensor(cu) and cu.dim() == 1, "cu_seqlens should be a 1-D tensor"
        assert cu.dtype == torch.int32, (
            f"cu_seqlens must be torch.int32 (flash_attn_varlen_func requires int32), got "
            f"{cu.dtype}")
        if cu.numel() == len(seqs) and torch.equal(cu.long(), want[1:]):
            raise AssertionError(
                "cu_seqlens is missing the leading 0: it has num_docs + 1 entries, "
                "[0, len_0, len_0 + len_1, ...], so that segment k is "
                "tokens[cu_seqlens[k]:cu_seqlens[k+1]]")
        assert cu.numel() == len(seqs) + 1, (
            f"cu_seqlens should have num_docs + 1 = {len(seqs) + 1} entries (one segment per "
            f"DOCUMENT, not per row), got {cu.numel()}")
        assert torch.equal(cu.long(), want), (
            f"cu_seqlens should be {want.tolist()} (0, then cumulative document lengths in "
            f"row-major order over the unpadded tokens input_ids[doc_ids >= 0]), got "
            f"{cu.tolist()}")
        flat = ids[doc >= 0]
        for k, i in enumerate(order):
            assert torch.equal(flat[int(cu[k]):int(cu[k + 1])], seqs[i]), (
                f"segment {k} of the unpadded token stream is not a whole document")


def check_edge_cases(ns):
    # pad_id is respected and real tokens may equal it.
    seqs = [torch.tensor([7, 0, 7]), torch.tensor([0, 0]), torch.tensor([5])]
    ids, pos, doc, cu = _pack(ns, seqs, 4, pad_id=7)
    rids, rpos, rdoc, rcu, _ = _ref_pack(seqs, 4, 7)
    assert torch.equal(ids, rids) and torch.equal(doc, rdoc), (
        f"with pad_id=7 and real tokens 0 and 7, expected input_ids {rids.tolist()} and "
        f"doc_ids {rdoc.tolist()}, got {ids.tolist()} / {doc.tolist()} — use the pad_id "
        f"argument, and derive padding from the layout, never from token values")
    # Every document exactly max_len long: one per row, no padding.
    seqs = _make_seqs([6, 6, 6], seed=3)
    ids, pos, doc, cu = _pack(ns, seqs, 6)
    assert tuple(ids.shape) == (3, 6) and bool((doc >= 0).all()), (
        "three documents of length max_len should fill three rows exactly")
    # No documents at all.
    ids, pos, doc, cu = _pack(ns, [], 5)
    assert tuple(ids.shape) == (0, 5) and tuple(doc.shape) == (0, 5), (
        f"an empty list of sequences should give (0, max_len) tensors, got {tuple(ids.shape)}")
    assert cu.tolist() == [0] and cu.dtype == torch.int32, (
        f"with no documents cu_seqlens should be tensor([0], dtype=int32), got {cu}")
    for bad, why in (([torch.arange(6)], "longer than max_len"),
                     ([torch.arange(3), torch.zeros(0, dtype=torch.long)], "empty")):
        try:
            ns.pack_sequences(bad, 5, 0)
        except ValueError:
            continue
        raise AssertionError(f"pack_sequences should raise ValueError for a sequence that is {why}")


# ----------------------------------------------------------------- labels and mask

def _label_cases():
    g = torch.Generator().manual_seed(5)
    cases = []
    for seed in range(3):
        seqs = _make_seqs(_random_lengths(9, 8, 60 + seed), seed=seed)
        ids, _, doc, _, _ = _ref_pack(seqs, 8, 0)
        cases.append((ids, doc))
    # Hand-made: a row with no padding, ids not starting at 0, a pure-padding row,
    # two neighbouring documents whose boundary tokens are equal.
    doc = torch.tensor([[5, 5, 2, 2, 2, -1],
                        [0, 1, 1, 1, 3, 3],
                        [-1, -1, -1, -1, -1, -1],
                        [4, 4, 4, 4, 4, 4]])
    ids = torch.randint(1, 9, doc.shape, generator=g)
    ids[0, 1] = ids[0, 2] = 3
    ids[doc == -1] = 0
    cases.append((ids, doc))
    return cases


def check_packed_labels(ns):
    for ids, doc in _label_cases():
        keep = ids.clone()
        got = ns.packed_labels(ids, doc)
        want = _ref_labels(ids, doc)
        assert torch.equal(ids, keep), "packed_labels modified input_ids in place"
        assert torch.is_tensor(got) and got.shape == ids.shape, (
            f"labels should have the shape of input_ids {tuple(ids.shape)}, got "
            f"{tuple(got.shape) if torch.is_tensor(got) else type(got).__name__}")
        assert got.dtype == torch.long, f"labels should be int64, got {got.dtype}"
        if torch.equal(got, torch.where(doc >= 0, ids, torch.full_like(ids, IGNORE))):
            raise AssertionError(
                "labels equal input_ids (unshifted). Here labels are the targets of the "
                "logits at the SAME position, so labels[:, t] must be the NEXT token "
                "input_ids[:, t + 1]")
        first = torch.ones_like(doc, dtype=torch.bool)
        first[:, 1:] = doc[:, 1:] != doc[:, :-1]
        hf = torch.where((doc >= 0) & ~first, ids, torch.full_like(ids, IGNORE))
        if torch.equal(got, hf):
            raise AssertionError(
                "these are Hugging Face-style UNSHIFTED labels (-100 at each document's "
                "first token, the model shifts inside its loss). This problem uses "
                "pre-shifted labels: labels[:, t] is the target of the logits at position "
                "t, i.e. the next token input_ids[:, t + 1], and -100 at each document's "
                "LAST token")
        boundary = (doc[:, :-1] >= 0) & (doc[:, 1:] != doc[:, :-1]) & (doc[:, 1:] >= 0)
        if bool(boundary.any()) and bool((got[:, :-1][boundary] != IGNORE).any()):
            raise AssertionError(
                "the last token of a document has a real label (the first token of the "
                "NEXT document). Nothing in document A can predict how document B starts — "
                "set -100 at every document's last token")
        assert torch.equal(got, want), (
            f"labels differ from the reference at {(got != want).nonzero()[:4].tolist()} "
            f"(row, col). labels[r, t] = input_ids[r, t+1] when position t+1 is in the same "
            f"(non-padding) document, else -100 — including the last column and all padding")


def check_document_causal_mask(ns):
    cases = [doc for _, doc in _label_cases()]
    for doc in cases:
        got = ns.document_causal_mask(doc)
        R, L = doc.shape
        assert torch.is_tensor(got) and tuple(got.shape) == (R, L, L), (
            f"mask should have shape (R, L, L) = {(R, L, L)}, got "
            f"{tuple(got.shape) if torch.is_tensor(got) else type(got).__name__}")
        assert got.dtype == torch.bool, (
            f"mask should be torch.bool with True = may attend (the attn_mask convention of "
            f"F.scaled_dot_product_attention), got {got.dtype}")
        want = _ref_mask(doc)
        if torch.equal(got, ~want):
            raise AssertionError("the mask is inverted — True must mean 'may attend'")
        empty = ~got.any(-1)
        if bool(empty.any()):
            r, i = empty.nonzero()[0].tolist()
            raise AssertionError(
                f"row {r}, query {i} (doc_id {int(doc[r, i])}) may attend to nothing. A "
                f"softmax over an all-masked row is 0/0 = NaN, and NaN activations poison "
                f"the gradients even though their labels are -100. Treat padding as its own "
                f"document so each padding query attends to itself (and earlier padding)")
        causal = torch.ones(L, L, dtype=torch.bool).tril().expand(R, L, L)
        same = doc[:, :, None] == doc[:, None, :]
        if bool((got & ~same & (doc[:, :, None] >= 0) & (doc[:, None, :] >= 0)).any()):
            raise AssertionError(
                "a token attends to a token of a DIFFERENT document in the same row "
                "(cross-document contamination) — a causal mask alone is not enough; AND "
                "it with doc_ids[:, :, None] == doc_ids[:, None, :]")
        if bool((got & (doc[:, :, None] >= 0) & (doc[:, None, :] < 0)).any()):
            raise AssertionError("a real token attends to padding")
        if bool((got & ~causal).any()):
            raise AssertionError("a token attends to a later position — the mask must be causal")
        real_q = (doc >= 0)[:, :, None].expand(R, L, L)
        if torch.equal(got[real_q], want[real_q]) and not torch.equal(got, want):
            raise AssertionError(
                "the rows of real tokens are right but the padding rows differ. By this "
                "problem's convention padding is one more document (id -1): a padding "
                "query attends to every padding key at or before it in its row — "
                "exactly what (doc_ids[:, :, None] == doc_ids[:, None, :]) & causal gives")
        assert torch.equal(got, want), (
            f"mask differs from the reference at (row, query, key) "
            f"{(got != want).nonzero()[:3].tolist()}: mask[r, i, j] = "
            f"(doc_ids[r, i] == doc_ids[r, j]) & (j <= i), padding included as document -1")


# ----------------------------------------------------------------- end to end

class _TinyLM(nn.Module):
    """Token + LEARNED absolute position embeddings, one pre-LN block, LM head."""

    def __init__(self, vocab, max_len, d=16, n_heads=2):
        super().__init__()
        self.h = n_heads
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(max_len, d)
        self.ln1, self.ln2, self.ln_f = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.head = nn.Linear(d, vocab)

    def forward(self, input_ids, position_ids, mask):
        R, L = input_ids.shape
        x = self.tok(input_ids) + self.pos(position_ids)
        q, k, v = self.qkv(self.ln1(x)).chunk(3, -1)
        q, k, v = (t.view(R, L, self.h, -1).transpose(1, 2) for t in (q, k, v))
        s = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        a = s.masked_fill(~mask[:, None], float("-inf")).softmax(-1)   # hand-written: NaN if a row is all False
        x = x + self.proj((a @ v).transpose(1, 2).reshape(R, L, -1))
        x = x + self.mlp(self.ln2(x))
        return self.head(self.ln_f(x))


def check_packed_batch_equals_separate_documents(ns):
    vocab, max_len = 40, 16
    lengths = [7, 3, 12, 5, 9, 2, 4, 6, 1, 16, 8]
    seqs = _make_seqs(lengths, vocab=vocab, seed=123)
    torch.manual_seed(0)
    model = _TinyLM(vocab, max_len).double()

    ids, pos, doc, cu = _pack(ns, seqs, max_len)
    labels = ns.packed_labels(ids, doc)
    mask = ns.document_causal_mask(doc)
    assert mask.dtype == torch.bool, "document_causal_mask must return a bool tensor"
    logits = model(ids, pos, mask)
    assert bool(torch.isfinite(logits).all()), (
        "the packed forward produced NaN/inf logits (at padding positions?) — some mask "
        "row has no True entry, so its softmax is 0/0")
    tok_loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=IGNORE,
                               reduction="none").view(labels.shape)

    total_sep, n_sep = 0.0, 0
    for i, s in enumerate(seqs):
        n = len(s)
        alone = model(s[None], torch.arange(n)[None],
                      torch.ones(n, n, dtype=torch.bool).tril()[None])[0]
        where = doc == i
        packed_i = logits[where]
        assert packed_i.shape == alone.shape, f"document {i} has the wrong number of tokens"
        diff = (packed_i - alone).abs().max().item()
        assert diff < 1e-9, (
            f"document {i}: the logits in the packed batch differ from running the document "
            f"alone (max diff {diff:.2e}). Either it can see another document (mask), or its "
            f"positions don't restart at 0 (position_ids)")
        sep_loss = F.cross_entropy(alone[:-1], s[1:], reduction="none")
        packed_loss = tok_loss[where][:-1]
        assert torch.allclose(packed_loss, sep_loss, rtol=1e-9, atol=1e-10), (
            f"document {i}: the per-token next-token losses in the packed batch differ from "
            f"the document run alone — check the labels")
        assert float(tok_loss[where][-1].detach()) == 0.0 and int(labels[where][-1]) == IGNORE, (
            f"document {i}'s last token must not be trained (label -100)")
        total_sep += float(sep_loss.sum().detach())
        n_sep += n - 1

    n_packed = int((labels != IGNORE).sum())
    assert n_packed == n_sep, (
        f"{n_packed} tokens carry a label in the packed batch, expected {n_sep} "
        f"(sum of (length - 1) over the documents)")
    mean_loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=IGNORE)
    assert abs(float(mean_loss.detach()) - total_sep / n_sep) < 1e-9, (
        "the packed mean loss should equal the token-weighted mean over all documents")
    mean_loss.backward()
    bad = [n for n, p in model.named_parameters()
           if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
    assert not bad, f"non-finite gradients in {bad} — a fully masked attention row?"


CHECKS = [
    check_packing_is_valid,
    check_ffd_layout_is_exact,
    check_position_ids_restart_per_document,
    check_cu_seqlens_flash_attn_convention,
    check_edge_cases,
    check_packed_labels,
    check_document_causal_mask,
    check_packed_batch_equals_separate_documents,
]
