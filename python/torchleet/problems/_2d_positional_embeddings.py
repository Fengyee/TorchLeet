"""2d-positional-embeddings — sinusoidal position encodings for image grids.

The defining property of a 2D sin-cos embedding is that it is separable: one half
of the channels is a 1D sinusoid of the row index alone, the other half a 1D
sinusoid of the column index alone. The checks pin that down without fixing
conventions that differ between correct implementations — which half holds the
row (MAE's code puts the column first) and whether sin/cos are interleaved or
concatenated inside a half (each half is compared as a multiset of values).
"""
import math

import torch

ENTRIES = ["create_2d_sinusoidal_embeddings"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Encode the row and the column separately, then concatenate the two halves.",
    "Each half is a standard 1D sinusoidal embedding of size d_half = d_model // 2, "
    "with frequencies 1 / 10000^(2i / d_half) — so d_model must be divisible by 4.",
    "Flatten the grid in row-major order so index r*width + c is position (r, c).",
]

H, W, D = 4, 6, 32


def _emb(ns):
    return ns.create_2d_sinusoidal_embeddings(H, W, D)


def check_shape(ns):
    e = _emb(ns)
    assert tuple(e.shape) == (H * W, D), \
        f"expected {(H * W, D)} (flattened grid), got {tuple(e.shape)}"


def check_deterministic(ns):
    assert torch.allclose(_emb(ns), _emb(ns)), \
        "embeddings must be a fixed function of position, not random"


def check_values_are_bounded(ns):
    e = _emb(ns)
    assert not torch.isnan(e).any(), "NaN in embeddings"
    assert float(e.abs().max()) <= 1.0 + 1e-6, \
        f"sin/cos outputs must stay within [-1, 1], got max |value| {float(e.abs().max()):.3f}"


def check_positions_are_distinct(ns):
    """Different grid cells must not collide, or attention cannot tell them apart."""
    e = _emb(ns)
    dup = torch.cdist(e, e) + torch.eye(H * W) * 10
    assert float(dup.min()) > 1e-6, \
        "two different grid positions produced identical embeddings"


def check_encodes_both_axes(ns):
    """Moving along a row and along a column must both change the embedding."""
    e = _emb(ns).reshape(H, W, D)
    assert not torch.allclose(e[0, 0], e[0, 1], atol=1e-6), \
        "changing the column did not change the embedding — width is not encoded"
    assert not torch.allclose(e[0, 0], e[1, 0], atol=1e-6), \
        "changing the row did not change the embedding — height is not encoded"


def _sincos_values(n, dim):
    """Per position, the sorted values of a dim-wide 1D sin-cos embedding."""
    freqs = torch.exp(torch.arange(0, dim, 2, dtype=torch.float64) * -(math.log(10000.0) / dim))
    angles = torch.arange(n, dtype=torch.float64)[:, None] * freqs
    return torch.cat([angles.sin(), angles.cos()], dim=1).sort(dim=1).values


def check_halves_are_1d_sinusoids_of_row_and_column(ns):
    """One half must depend only on the row, the other only on the column, and
    each must hold the sin/cos values of the standard 1D formula on d_model // 2."""
    e = _emb(ns).reshape(H, W, D).double()
    half = D // 2
    a, b = e[..., :half], e[..., half:]

    def row_only(x):   # constant along the width
        return torch.allclose(x, x[:, :1].expand_as(x), atol=1e-6)

    def col_only(x):   # constant along the height
        return torch.allclose(x, x[:1].expand_as(x), atol=1e-6)

    if row_only(a) and col_only(b):
        rows, cols = a[:, 0], b[0]
    elif col_only(a) and row_only(b):
        rows, cols = b[:, 0], a[0]
    else:
        raise AssertionError(
            "one half of the channels should depend only on the row and the other half "
            "only on the column (flattened row-major, index r*width + c). A 1D "
            "embedding of the flattened index, or a column-major flatten, breaks this.")
    for name, got, n in (("row", rows, H), ("column", cols, W)):
        want = _sincos_values(n, half)
        diff = (got.sort(dim=1).values - want).abs().max().item()
        assert diff < 1e-5, (
            f"the {name} half is not a standard 1D sinusoid of size d_model // 2 = {half} "
            f"(max diff {diff:.2e}): position p should give sin(p * w_i) and cos(p * w_i) "
            f"with w_i = 1 / 10000^(2i / {half})")


CHECKS = [check_shape, check_deterministic, check_values_are_bounded,
          check_positions_are_distinct, check_encodes_both_axes,
          check_halves_are_1d_sinusoids_of_row_and_column]
