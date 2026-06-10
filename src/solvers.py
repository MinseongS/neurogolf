"""Tiered automatic solvers.

Tier conv: a single Conv (with bias) computes output[o,r,c] from the KxK
one-hot neighborhood of (r,c). The official check is `result > 0.0`, so each
output channel is a linear threshold function -> fit with an integer
perceptron per channel. Integer weights on 0/1 inputs keep float32 sums exact,
so local numpy verification is bit-faithful to ONNX Runtime.

A pattern-conflict pre-check (same neighborhood, different target) cheaply
rejects kernel sizes for which NO single conv can work.
"""

import time

import numpy as np

from . import builders
from .analyze import usable_examples
from .harness import CHANNELS, HEIGHT, WIDTH, convert_to_numpy

# ordered by params (= 100 * kh * kw); smaller kernel -> higher score
KERNEL_LADDER = [
    (1, 1),
    (1, 3), (3, 1),
    (1, 5), (5, 1),
    (3, 3), (1, 9), (9, 1),
    (3, 5), (5, 3),
    (5, 5), (3, 9), (9, 3),
    (7, 7),
    (5, 9), (9, 5), (1, 59), (59, 1),
    (9, 9),
]
MAX_UPDATES = 60_000


def canvases(task):
    pairs = []
    for ex in usable_examples(task):
        bm = convert_to_numpy(ex)
        pairs.append((bm["input"][0].astype(bool), bm["output"][0].astype(bool)))
    return pairs


def neighborhood_features(x, kh, kw):
    """x: bool [10,30,30] -> bool [900, 10*kh*kw] of per-cell neighborhoods."""
    ph, pw = kh // 2, kw // 2
    xp = np.pad(x, ((0, 0), (ph, ph), (pw, pw)))
    win = np.lib.stride_tricks.sliding_window_view(xp, (kh, kw), axis=(1, 2))
    # win: [10, 30, 30, kh, kw] -> [30, 30, 10, kh, kw] -> [900, D]
    return win.transpose(1, 2, 0, 3, 4).reshape(HEIGHT * WIDTH, CHANNELS * kh * kw)


def collect_patterns(pairs, kh, kw):
    """Unique (neighborhood pattern -> target colors) over all canvas cells.

    Returns (X bool [U, D+1] with bias column, Y bool [U, 10]) or None on
    conflict (same pattern, different target = infeasible for ANY KxK conv).
    """
    feats, targets = [], []
    for x, y in pairs:
        f = neighborhood_features(x, kh, kw)
        t = y.transpose(1, 2, 0).reshape(HEIGHT * WIDTH, CHANNELS)
        keep = f.any(axis=1) | t.any(axis=1)
        feats.append(f[keep])
        targets.append(t[keep])
    f = np.concatenate(feats)
    t = np.concatenate(targets)
    packed = np.packbits(f, axis=1)
    _, idx, inv = np.unique(packed, axis=0, return_index=True, return_inverse=True)
    # conflict iff two cells with identical pattern want different targets
    tp = np.packbits(t, axis=1)
    if not (tp == tp[idx][inv]).all():
        return None
    X = f[idx]
    Y = t[idx]
    # explicit all-zero pattern: empty neighborhoods must stay empty (bias <= 0)
    X = np.vstack([X, np.zeros((1, X.shape[1]), bool)])
    Y = np.vstack([Y, np.zeros((1, CHANNELS), bool)])
    X = np.hstack([X, np.ones((X.shape[0], 1), bool)])  # bias column
    return X, Y


def perceptron(X, y, deadline, max_updates=MAX_UPDATES):
    """Integer perceptron: find integer w with X@w > 0 exactly where y.

    Negatives are satisfied at score <= 0. Returns int64 w or None.
    """
    Xf = X.astype(np.int64)
    w = np.zeros(X.shape[1], dtype=np.int64)
    sign = np.where(y, 1, -1)
    updates = 0
    rng = np.random.default_rng(0)
    while updates < max_updates and time.monotonic() < deadline:
        s = Xf @ w
        err = np.where(y, s <= 0, s > 0)
        bad = np.flatnonzero(err)
        if bad.size == 0:
            return w
        # batch of single-sample updates in random order, re-checking lazily
        rng.shuffle(bad)
        for i in bad[:256]:
            s_i = Xf[i] @ w
            if (y[i] and s_i <= 0) or (not y[i] and s_i > 0):
                w += sign[i] * Xf[i]
                updates += 1
    return None


def verify_conv_numpy(pairs, W, B, kh, kw):
    ph, pw = kh // 2, kw // 2
    for x, y in pairs:
        xp = np.pad(x.astype(np.int64), ((0, 0), (ph, ph), (pw, pw)))
        win = np.lib.stride_tricks.sliding_window_view(xp, (kh, kw), axis=(1, 2))
        out = np.tensordot(W, win, axes=([1, 2, 3], [0, 3, 4])) + B[:, None, None]
        # ORT may sum in any order: bound the worst-case partial sum, not
        # just the result, to guarantee float32 exactness.
        out_abs = np.tensordot(np.abs(W), win, axes=([1, 2, 3], [0, 3, 4])) + np.abs(B)[:, None, None]
        if out_abs.max() >= (1 << 24):
            return False
        if not (((out > 0) == y).all()):
            return False
    return True


def grid_to_codes(grid):
    """top-left aligned [30,30] int canvas, color c -> code c+1, empty -> 0."""
    canvas = np.zeros((30, 30), np.int64)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            canvas[r, c] = color + 1
    return canvas


def solve_memorizer(task, k_proj=4):
    """Exact-match lookup over all scoreable examples. Always succeeds unless
    the task itself is contradictory (same input, different outputs)."""
    exs = usable_examples(task)
    if not exs:
        return None
    seen = {}
    xs, os_ = [], []
    for ex in exs:
        xin = builders.pack4_codes(grid_to_codes(ex["input"]))
        xout = builders.pack6_codes(grid_to_codes(ex["output"]))
        key = xin.tobytes()
        if key in seen:
            if seen[key] != xout.tobytes():
                return None  # contradictory task
            continue
        seen[key] = xout.tobytes()
        xs.append(xin)
        os_.append(xout)
    X4 = np.array(xs, dtype=np.int64)             # [N,240]
    O = np.array(os_, dtype=np.float32)           # [N,150]
    N = X4.shape[0]

    # dedupe repeated outputs when the grouping matrix is cheaper than the rows
    G = None
    uniq, group_idx = np.unique(O, axis=0, return_inverse=True)
    U = uniq.shape[0]
    if U * (O.shape[1] + N) < O.shape[1] * N:
        G = np.zeros((N, U), np.float32)
        G[np.arange(N), group_idx] = 1.0
        O = uniq

    for k in (k_proj, k_proj + 4, k_proj + 8, k_proj + 16):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            R = rng.choice([-1.0, 1.0], size=(240, k)).astype(np.float32)
            Z = (X4 @ R.astype(np.int64)).astype(np.float32)
            if np.abs(Z).max() >= (1 << 24):
                continue
            if np.unique(Z, axis=0).shape[0] == N:
                model = builders.memorizer_network(Z, R, O, k, G=G)
                tag = f"memorizer(n={N},k={k}" + (f",u={U})" if G is not None else ")")
                return model, {"method": tag}
    return None


def solve_conv(task, task_budget=90.0, channel_budget=6.0):
    """Try the kernel ladder; return (model, meta) for the smallest success."""
    pairs = canvases(task)
    if not pairs:
        return None
    task_deadline = time.monotonic() + task_budget
    for kh, kw in KERNEL_LADDER:
        if time.monotonic() > task_deadline:
            return None
        pat = collect_patterns(pairs, kh, kw)
        if pat is None:
            continue
        X, Y = pat
        D = CHANNELS * kh * kw
        ws = np.zeros((CHANNELS, D + 1), dtype=np.int64)
        ok = True
        for o in range(CHANNELS):
            deadline = min(time.monotonic() + channel_budget, task_deadline)
            w = perceptron(X, Y[:, o], deadline)
            if w is None:
                ok = False
                break
            ws[o] = w
        if not ok:
            continue
        W = ws[:, :D].reshape(CHANNELS, CHANNELS, kh, kw)
        B = ws[:, D]
        if not verify_conv_numpy(pairs, W, B, kh, kw):
            continue
        use_bias = bool(B.any())
        model = builders.conv_network(W, kh, kw, bias=B if use_bias else None)
        meta = {"method": f"conv{kh}x{kw}" + ("+b" if use_bias else "")}
        return model, meta
    return None
