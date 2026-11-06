"""rotary-positional-embedding (RoPE).

RoPE is a rotation, which gives exact properties to test: applying rotate_half
twice negates the vector, rotating with a unit (cos, sin) pair preserves the
norm, and the score <R_m q, R_n k> depends only on m - n. Neither the norm nor
the zero-angle check can tell a real rotation from returning q and k untouched,
so check_quarter_turn pins the angle: at cos=0, sin=1 the result must be exactly
rotate_half of the input.

Which features form a rotation pair is a convention: half-split (i, i + d/2,
GPT-NeoX / HF Llama) or interleaved (2i, 2i + 1, RoFormer / GPT-J). The checks
read the pairing off the solver's own rotate_half and lay out cos/sin to match,
so either convention passes.
"""
import torch

ENTRIES = ["rotate_half", "apply_rotary_pos_emb"]
DEVICE = "cpu"
EXTRAS = []
HINTS = [
    "RoPE turns each pair of features by an angle m * theta_i (position times a per-pair "
    "frequency). A rotation keeps norms, and q_m . k_n then depends only on m - n.",
    "Write the 2D rotation elementwise: x*cos + rotate_half(x)*sin, where rotate_half "
    "maps each pair (a, b) to (-b, a). Rotate q and k the same way.",
    "Half-split convention: rotate_half returns cat(-x2, x1) of the two halves, and "
    "cos/sin repeat the angles as cat(freqs, freqs); they broadcast against "
    "(batch, heads, seq, head_dim).",
]

B, H, S, D = 2, 4, 8, 16


def check_rotate_half_shape(ns):
    x = torch.randn(B, H, S, D)
    assert tuple(ns.rotate_half(x).shape) == (B, H, S, D), \
        f"rotate_half must preserve shape, got {tuple(ns.rotate_half(x).shape)}"


def check_rotate_half_twice_negates(ns):
    """(-x2, x1) applied twice gives (-x1, -x2) = -x."""
    x = torch.randn(B, H, S, D)
    twice = ns.rotate_half(ns.rotate_half(x))
    assert torch.allclose(twice, -x, atol=1e-6), \
        "applying rotate_half twice should negate the input"


def check_rotate_half_permutes_values(ns):
    x = torch.randn(2, 4)
    out = ns.rotate_half(x)
    assert torch.allclose(out.abs().sort(-1).values, x.abs().sort(-1).values, atol=1e-6), \
        "rotate_half should move values around and flip signs, not change magnitudes"


def check_identity_rotation(ns):
    """cos=1, sin=0 is a zero-angle rotation: q and k come back unchanged."""
    q, k = torch.randn(B, H, S, D), torch.randn(B, H, S, D)
    cos, sin = torch.ones(S, D), torch.zeros(S, D)
    rq, rk = ns.apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.allclose(rq, q, atol=1e-6), "cos=1, sin=0 must leave q unchanged"
    assert torch.allclose(rk, k, atol=1e-6), "cos=1, sin=0 must leave k unchanged"


def check_quarter_turn(ns):
    """cos=0, sin=1 is a 90-degree turn: q*0 + rotate_half(q)*1 = rotate_half(q).

    Uses the solver's own rotate_half, so any consistent pairing convention passes,
    but returning q or k unrotated (or ignoring sin) does not.
    """
    torch.manual_seed(1)
    q, k = torch.randn(B, H, S, D), torch.randn(B, H, S, D)
    cos, sin = torch.zeros(S, D), torch.ones(S, D)
    rq, rk = ns.apply_rotary_pos_emb(q, k, cos, sin)
    assert not torch.allclose(rq, q, atol=1e-4), (
        "with cos=0, sin=1 the query came back unchanged — apply_rotary_pos_emb "
        "must actually use cos and sin (q*cos + rotate_half(q)*sin)")
    assert torch.allclose(rq, ns.rotate_half(q), atol=1e-6), (
        "with cos=0, sin=1 the rotated query should equal rotate_half(q) — "
        "check the q*cos + rotate_half(q)*sin formula")
    assert torch.allclose(rk, ns.rotate_half(k), atol=1e-6), (
        "with cos=0, sin=1 the rotated key should equal rotate_half(k) — the key "
        "must be rotated exactly like the query")


def _tables(ns, ang):
    """Lay out per-pair angles (S, D/2) as (cos, sin) tables (S, D) that match the
    pairing the solver's rotate_half uses: feature j and the feature rotate_half
    moves it to must share one angle."""
    partner = ns.rotate_half(torch.eye(D)).abs().argmax(dim=-1).tolist()
    pair_of, idx = {}, []
    for j in range(D):
        key = min(j, partner[j])
        pair_of.setdefault(key, len(pair_of))
        idx.append(pair_of[key])
    assert len(pair_of) == D // 2 and all(partner[partner[j]] == j for j in range(D)), (
        "rotate_half should pair the features up: each one moves to a partner that "
        "moves back to it (e.g. i <-> i + d/2)")
    full = ang[:, idx]
    return full.cos(), full.sin()


def check_rotation_preserves_norm(ns):
    """A rotation cannot change a vector's length."""
    torch.manual_seed(0)
    q, k = torch.randn(B, H, S, D), torch.randn(B, H, S, D)
    cos, sin = _tables(ns, torch.rand(S, D // 2) * 6.28)
    rq, rk = ns.apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.allclose(rq.norm(dim=-1), q.norm(dim=-1), atol=1e-4), \
        "rotation changed the query norms — check the cos/sin pairing"
    assert torch.allclose(rk.norm(dim=-1), k.norm(dim=-1), atol=1e-4), \
        "rotation changed the key norms — check the cos/sin pairing"


def check_scores_depend_only_on_relative_position(ns):
    """RoPE's defining property: with angles m * theta_i, the score between the
    same q at position m and the same k at position n depends only on m - n."""
    torch.manual_seed(2)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, D, 2).float() / D))
    cos, sin = _tables(ns, torch.outer(torch.arange(S).float(), inv_freq))
    q = torch.randn(D).expand(1, 1, S, D).contiguous()
    k = torch.randn(D).expand(1, 1, S, D).contiguous()
    rq, rk = ns.apply_rotary_pos_emb(q, k, cos, sin)
    scores = rq[0, 0] @ rk[0, 0].T                      # scores[m, n] = <R_m q, R_n k>
    assert not torch.allclose(scores, scores[0, 0].expand(S, S), atol=1e-4), \
        "every query-key score is the same — the positions are not being rotated"
    drift = (scores[1:, 1:] - scores[:-1, :-1]).abs().max()
    assert drift < 1e-4, (
        f"shifting both positions by one changed the score (max change {float(drift):.2e}) "
        "— <R_m q, R_n k> must depend only on m - n; rotate q and k with the same "
        "formula and the same per-position angles")


CHECKS = [check_rotate_half_shape, check_rotate_half_twice_negates,
          check_rotate_half_permutes_values, check_identity_rotation,
          check_quarter_turn, check_rotation_preserves_norm,
          check_scores_depend_only_on_relative_position]
