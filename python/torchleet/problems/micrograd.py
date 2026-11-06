"""micrograd — a scalar reverse-mode autodiff engine.

Every check builds the same expression twice: once from the solver's Value
objects and once from float64 torch scalars with requires_grad=True, then
compares the forward value and every leaf's gradient with torch.autograd.
The random-DAG check deliberately reuses nodes (shared subexpressions), which is
where an engine that assigns instead of accumulates, or that does not visit
nodes in topological order, goes wrong.
"""
import math
import random
import sys

import torch

ENTRIES = ["Value"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "Every operation returns a new Value that remembers its inputs plus a tiny closure "
    "`_backward` which, given out.grad, adds (local derivative x out.grad) into each input's "
    ".grad. Ask yourself why it must ADD rather than assign.",
    "backward(): build a topological order of the graph (each node after all of its inputs), "
    "set self.grad = 1.0, then call _backward on the nodes in reverse order. Use an explicit "
    "stack instead of recursion so a chain 5000 nodes deep still works.",
    "Derive most ops from a few primitives: a - b = a + (-b), a / b = a * b**-1, -a = a * -1. "
    "The reflected ops wrap the Python number with the operands in the right order "
    "(1 - v is Value(1) - v, 1 / v is Value(1) / v). Local derivatives: tanh' = 1 - tanh^2, "
    "sigmoid' = s(1 - s), exp' = exp, log' = 1/x, (x^k)' = k x^(k-1).",
]

F64 = torch.float64


# ----------------------------------------------------------------- helpers

def _num(v, what):
    assert isinstance(v, (int, float)) and not isinstance(v, bool), (
        f"{what} must be a plain Python float, got {type(v).__name__} — the engine works on "
        f"Python scalars, not tensors")
    return float(v)


def _t(x):
    return torch.tensor(float(x), dtype=F64, requires_grad=True)


def _agree(got, want, what, rtol=1e-7, atol=1e-10):
    got = _num(got, what)
    want = float(want)
    assert math.isfinite(got), f"{what} is {got}, expected {want:.6g}"
    assert abs(got - want) <= atol + rtol * abs(want), \
        f"{what} = {got:.10g}, but torch (float64 autograd) gives {want:.10g}"


def _compare(ns, fn, inputs, label):
    """Run fn on Values and on torch scalars; compare output data and leaf grads."""
    vs = [ns.Value(x) for x in inputs]
    ts = [_t(x) for x in inputs]
    try:
        out_v = fn(*vs)
    except TypeError as e:
        if "unsupported operand" not in str(e):
            raise
        raise AssertionError(
            f"`{label}` raised TypeError ({e}) — implement the reflected operators "
            f"(__radd__, __rsub__, __rmul__, __rtruediv__) so a Python number can sit on the "
            f"left") from None
    out_t = fn(*ts)
    assert all(hasattr(out_v, a) for a in ("data", "grad", "backward")), \
        f"`{label}` returned {type(out_v).__name__}, not a Value"
    _agree(out_v.data, out_t.item(), f"`{label}`.data")
    out_v.backward()
    out_t.backward()
    names = "abcdefgh"
    for i, (v, t) in enumerate(zip(vs, ts)):
        want = 0.0 if t.grad is None else t.grad.item()
        _agree(v.grad, want, f"in `{label}`, {names[i]}.grad (a={inputs[0]}"
               + (f", b={inputs[1]})" if len(inputs) > 1 else ")"))


# ------------------------------------------------------------------ checks

def check_leaf_attributes(ns):
    v = ns.Value(2.5)
    assert _num(v.data, "Value(2.5).data") == 2.5, f"Value(2.5).data is {v.data}"
    assert _num(v.grad, "Value(2.5).grad") == 0.0, \
        f"a fresh Value must start with grad 0.0, got {v.grad}"
    v.backward()
    assert _num(v.grad, "leaf.grad after backward") == 1.0, \
        "calling .backward() on a leaf must seed its own grad with 1.0"


def check_arithmetic_between_values(ns):
    cases = [
        ("a + b", lambda a, b: a + b),
        ("a - b", lambda a, b: a - b),
        ("a * b", lambda a, b: a * b),
        ("a / b", lambda a, b: a / b),
        ("-a", lambda a, b: -a),
        ("a * b + b / a - a", lambda a, b: a * b + b / a - a),
    ]
    for label, fn in cases:
        for a, b in ((0.7, -1.3), (-2.5, 0.4)):
            _compare(ns, fn, (a, b), label)


def check_reflected_ops_with_python_numbers(ns):
    cases = [
        ("2 * a", lambda a: 2 * a),
        ("3 + a", lambda a: 3 + a),
        ("1 - a", lambda a: 1 - a),
        ("1 / a", lambda a: 1 / a),
        ("2.5 / a", lambda a: 2.5 / a),
        ("a + 2", lambda a: a + 2),
        ("a - 2", lambda a: a - 2),
        ("a * 3", lambda a: a * 3),
        ("a / 4", lambda a: a / 4),
        ("1 - 2 * a / 3 + 0.5", lambda a: 1 - 2 * a / 3 + 0.5),
    ]
    for label, fn in cases:
        for a in (0.7, -1.9):
            _compare(ns, fn, (a,), label)


def check_pow_with_negative_and_fractional_exponents(ns):
    for k in (2, 3, 0, 1, -1, -2):
        for a in (1.7, -0.6):
            _compare(ns, lambda x, k=k: x ** k, (a,), f"a ** {k}")
    for k in (0.5, -0.5, 1.5, -1.5, 2.25):
        _compare(ns, lambda x, k=k: x ** k, (1.7,), f"a ** {k}")
    _compare(ns, lambda x, y: (x * y) ** -0.5, (0.8, 2.0), "(a * b) ** -0.5")


def check_unary_functions(ns):
    cases = [
        ("a.exp()", lambda a: a.exp(), (0.3, -2.0, 4.0)),
        ("a.log()", lambda a: a.log(), (0.3, 2.0, 7.5)),
        ("a.tanh()", lambda a: a.tanh(), (0.3, -1.2, 3.0)),
        ("a.relu()", lambda a: a.relu(), (0.3, -1.2)),
        ("a.sigmoid()", lambda a: a.sigmoid(), (0.3, -2.5, 6.0)),
    ]
    for label, fn, points in cases:          # torch tensors have the same methods
        for a in points:
            _compare(ns, fn, (a,), label)
    v = ns.Value(0.0)
    r = v.relu()
    r.backward()
    assert _num(r.data, "relu(0).data") == 0.0, f"relu(0) should be 0, got {r.data}"
    assert _num(v.grad, "relu'(0)") == 0.0, (
        f"relu'(0) = {v.grad}; follow PyTorch's convention and use 0 at exactly x == 0 "
        f"(the local gradient is (x > 0), not (x >= 0))")


def check_reused_nodes_accumulate_gradients(ns):
    _compare(ns, lambda x: x * x, (3.0,), "x * x")
    _compare(ns, lambda x: x + x, (3.0,), "x + x")
    _compare(ns, lambda x: x * x * x, (-1.5,), "x * x * x")
    _compare(ns, lambda x: x - x, (2.0,), "x - x")
    _compare(ns, lambda x: x / x, (2.0,), "x / x")

    def shared(x, y):
        a = x + y
        return a * a

    _compare(ns, shared, (1.5, -0.5), "a = x + y; a * a")

    def diamond(x, y):
        a = x * y
        b = a.tanh()
        c = a.exp()
        d = b * c + a
        return d * a

    _compare(ns, diamond, (0.6, -0.9), "a = x*y; (a.tanh() * a.exp() + a) * a")


# --- random expression DAGs -------------------------------------------------

_BINARY = ["add", "sub", "mul", "div"]
_UNARY = ["neg", "exp", "log", "tanh", "relu", "sigmoid", "pow"]
_CONST = ["radd", "rsub", "rmul", "rdiv", "addc", "subc", "mulc", "divc"]
_EXPONENTS = [2, 3, -1, -2, 0.5, -0.5, 1.5]


def _apply(op, a, b, c, k):
    """One op on either Values or torch scalars; c is a Python constant, k an exponent."""
    return {
        "add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b, "div": lambda: a / b,
        "neg": lambda: -a, "exp": lambda: a.exp(), "log": lambda: a.log(),
        "tanh": lambda: a.tanh(), "relu": lambda: a.relu(), "sigmoid": lambda: a.sigmoid(),
        "pow": lambda: a ** k,
        "radd": lambda: c + a, "rsub": lambda: c - a, "rmul": lambda: c * a, "rdiv": lambda: c / a,
        "addc": lambda: a + c, "subc": lambda: a - c, "mulc": lambda: a * c, "divc": lambda: a / c,
    }[op]()


def _ok(op, a, b, k):
    """Keep every intermediate in a well-conditioned range."""
    if op == "log" or (op == "pow" and isinstance(k, float)):
        return a > 0.3
    if op == "pow" and k < 0:
        return abs(a) > 0.3
    if op == "div":
        return abs(b) > 0.3
    if op == "rdiv":
        return abs(a) > 0.3
    if op == "exp":
        return a < 3
    return True


def _random_dag(ns, rng, n_leaves, n_ops):
    leaves = [rng.uniform(-2, 2) for _ in range(n_leaves)]
    pool = [(ns.Value(x), _t(x)) for x in leaves]
    vleaves, tleaves = [p[0] for p in pool], [p[1] for p in pool]
    ops_used = []
    for _ in range(n_ops):
        for _attempt in range(50):
            op = rng.choice(_BINARY + _UNARY + _CONST)
            # Prefer recent nodes (depth) but allow any (shared subexpressions).
            pick = lambda: pool[-1 - min(int(rng.expovariate(0.4)), len(pool) - 1)]
            (va, ta), (vb, tb) = pick(), pick()
            c = rng.choice([-2.0, -0.5, 0.5, 1.5, 3])
            k = rng.choice(_EXPONENTS)
            if not _ok(op, ta.item(), tb.item(), k):
                continue
            tr = _apply(op, ta, tb, c, k)
            if not (1e-3 < abs(tr.item()) < 50):
                continue
            pool.append((_apply(op, va, vb, c, k), tr))
            ops_used.append(op)
            break
    out_v = pool[-1][0] + pool[-2][0] * pool[-3][0]
    out_t = pool[-1][1] + pool[-2][1] * pool[-3][1]
    return vleaves, tleaves, out_v, out_t, ops_used


def check_random_dags_with_shared_nodes_match_autograd(ns):
    rng = random.Random(0)
    for trial in range(40):
        vleaves, tleaves, out_v, out_t, ops = _random_dag(ns, rng, n_leaves=4, n_ops=20)
        what = f"random DAG #{trial} (ops: {' '.join(ops)})"
        _agree(out_v.data, out_t.item(), f"output of {what}")
        out_v.backward()
        out_t.backward()
        for i, (v, t) in enumerate(zip(vleaves, tleaves)):
            want = 0.0 if t.grad is None else t.grad.item()
            g = _num(v.grad, "leaf.grad")
            assert abs(g - want) <= 1e-9 + 1e-6 * abs(want), (
                f"leaf {i} of {what}: grad {g:.10g}, torch (float64 autograd) gives {want:.10g}. "
                f"Shared nodes must accumulate (+=) and _backward must run in reverse "
                f"topological order")


def check_deep_chain_has_no_recursion_limit(ns):
    depth = 5000
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(min(old, 1000))      # Python's default; don't let a raised limit hide recursion
    try:
        x = ns.Value(0.5)
        y = x
        for _ in range(depth):
            y = y + x
        try:
            y.backward()
        except RecursionError:
            raise AssertionError(
                f"RecursionError on a {depth}-deep chain at Python's default recursion limit "
                f"— build the topological order with an explicit stack, not recursion") from None
        assert abs(_num(x.grad, "x.grad") - (depth + 1)) < 1e-6, \
            f"y = x + x + ... ({depth + 1} terms): x.grad should be {depth + 1}, got {x.grad}"
        assert abs(y.data - 0.5 * (depth + 1)) < 1e-6, f"y.data should be {0.5 * (depth + 1)}"
    finally:
        sys.setrecursionlimit(old)


CHECKS = [
    check_leaf_attributes,
    check_arithmetic_between_values,
    check_reflected_ops_with_python_numbers,
    check_pow_with_negative_and_fractional_exponents,
    check_unary_functions,
    check_reused_nodes_accumulate_gradients,
    check_random_dags_with_shared_nodes_match_autograd,
    check_deep_chain_has_no_recursion_limit,
]
