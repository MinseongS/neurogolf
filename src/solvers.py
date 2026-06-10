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

_DENSE = [
    (1, 1), (1, 3), (3, 1), (1, 5), (5, 1), (3, 3), (1, 9), (9, 1),
    (3, 5), (5, 3), (5, 5), (3, 9), (9, 3), (7, 7), (5, 9), (9, 5),
    (1, 59), (59, 1), (9, 9),
]
_DW = [
    (1, 3), (3, 1), (1, 5), (5, 1), (3, 3), (1, 9), (9, 1), (3, 5),
    (5, 3), (5, 5), (3, 9), (9, 3), (7, 7), (1, 59), (59, 1), (9, 9),
]
# (params, kind, kh, kw) sorted cheapest-first; depthwise = 10*k params vs 100*k
KERNEL_LADDER = sorted(
    [(10 * kh * kw, "dw", kh, kw) for kh, kw in _DW]
    + [(100 * kh * kw, "dense", kh, kw) for kh, kw in _DENSE])
MAX_UPDATES = 60_000


def canvases(task):
    pairs = []
    for ex in usable_examples(task):
        bm = convert_to_numpy(ex)
        pairs.append((bm["input"][0].astype(bool), bm["output"][0].astype(bool)))
    return pairs


def neighborhood_features(x, kh, kw):
    """x: bool [C,30,30] -> bool [900, C*kh*kw] of per-cell neighborhoods."""
    ph, pw = kh // 2, kw // 2
    xp = np.pad(x, ((0, 0), (ph, ph), (pw, pw)))
    win = np.lib.stride_tricks.sliding_window_view(xp, (kh, kw), axis=(1, 2))
    # win: [C, 30, 30, kh, kw] -> [30, 30, C, kh, kw] -> [900, D]
    return win.transpose(1, 2, 0, 3, 4).reshape(HEIGHT * WIDTH, x.shape[0] * kh * kw)


def collect_patterns(pairs, kh, kw, channel=None):
    """Unique (neighborhood pattern -> target colors) over all canvas cells.

    With channel=c, features are restricted to input channel c and targets to
    output channel c (depthwise fitting). Returns (X bool [U, D+1] with bias
    column, Y bool [U, n_targets]) or None on conflict (same pattern,
    different target = infeasible for ANY conv of this shape).
    """
    feats, targets = [], []
    for x, y in pairs:
        if channel is None:
            f = neighborhood_features(x, kh, kw)
            t = y.transpose(1, 2, 0).reshape(HEIGHT * WIDTH, CHANNELS)
        else:
            f = neighborhood_features(x[channel:channel + 1], kh, kw)
            t = y[channel].reshape(HEIGHT * WIDTH, 1)
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
    Y = np.vstack([Y, np.zeros((1, Y.shape[1]), bool)])
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
    # output bounding box across all stored examples
    rmax = max(len(ex["output"]) for ex in exs)
    wmax = max(len(ex["output"][0]) for ex in exs)
    w6 = (wmax + 5) // 6
    if rmax * w6 * 6 >= 600:
        rmax, w6 = 30, 5  # near-full bbox: the in-graph Pad costs more than it saves
    seen = {}
    xs, os_ = [], []
    for ex in exs:
        xin = builders.pack4_codes(grid_to_codes(ex["input"]))
        xout = builders.pack6_codes(grid_to_codes(ex["output"]), rmax, w6)
        key = xin.tobytes()
        if key in seen:
            if seen[key] != xout.tobytes():
                return None  # contradictory task
            continue
        seen[key] = xout.tobytes()
        xs.append(xin)
        os_.append(xout)
    X4 = np.array(xs, dtype=np.int64)             # [N,240]
    O = np.array(os_, dtype=np.float32)           # [N,rmax*w6]
    N = X4.shape[0]

    # dedupe repeated outputs when the grouping matrix is cheaper than the rows
    G = None
    uniq, group_idx = np.unique(O, axis=0, return_inverse=True)
    U = uniq.shape[0]
    if U * (O.shape[1] + N) < O.shape[1] * N:
        G = np.zeros((N, U), np.float32)
        G[np.arange(N), group_idx] = 1.0
        O = uniq

    for k in (k_proj, k_proj + 2, k_proj + 6, k_proj + 14):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            R = rng.choice([-1.0, 1.0], size=(240, k)).astype(np.float32)
            Z = (X4 @ R.astype(np.int64)).astype(np.float32)
            if np.abs(Z).max() >= (1 << 24):
                continue
            if np.unique(Z, axis=0).shape[0] == N:
                model = builders.memorizer_network(Z, R, O, k, G=G, rmax=rmax, w6=w6)
                tag = f"memorizer(n={N},k={k},bb={rmax}x{w6 * 6}" + \
                    (f",u={U})" if G is not None else ")")
                return model, {"method": tag}
    return None


def _fit_dense(pairs, kh, kw, channel_budget, task_deadline):
    pat = collect_patterns(pairs, kh, kw)
    if pat is None:
        return None
    X, Y = pat
    D = CHANNELS * kh * kw
    ws = np.zeros((CHANNELS, D + 1), dtype=np.int64)
    for o in range(CHANNELS):
        deadline = min(time.monotonic() + channel_budget, task_deadline)
        w = perceptron(X, Y[:, o], deadline)
        if w is None:
            return None
        ws[o] = w
    return ws[:, :D].reshape(CHANNELS, CHANNELS, kh, kw), ws[:, D]


def _fit_depthwise(pairs, kh, kw, channel_budget, task_deadline):
    D = kh * kw
    W = np.zeros((CHANNELS, 1, kh, kw), dtype=np.int64)
    B = np.zeros(CHANNELS, dtype=np.int64)
    for c in range(CHANNELS):
        pat = collect_patterns(pairs, kh, kw, channel=c)
        if pat is None:
            return None
        X, Y = pat
        deadline = min(time.monotonic() + channel_budget, task_deadline)
        w = perceptron(X, Y[:, 0], deadline)
        if w is None:
            return None
        W[c, 0] = w[:D].reshape(kh, kw)
        B[c] = w[D]
    return W, B


def solve_conv(task, task_budget=120.0, channel_budget=6.0, beat=0.0):
    """Try the kernel ladder; return (model, meta) for the smallest success.

    beat: current best points for this task — ladder steps whose best possible
    score (25 - ln(params)) can't exceed it are skipped.
    """
    import math
    pairs = canvases(task)
    if not pairs:
        return None
    task_deadline = time.monotonic() + task_budget
    for nparams, kind, kh, kw in KERNEL_LADDER:
        if 25.0 - math.log(nparams) <= beat + 1e-9:
            return None  # ladder is params-sorted; nothing further can win
        if time.monotonic() > task_deadline:
            return None
        if kind == "dw":
            fit = _fit_depthwise(pairs, kh, kw, channel_budget, task_deadline)
        else:
            fit = _fit_dense(pairs, kh, kw, channel_budget, task_deadline)
        if fit is None:
            continue
        W, B = fit
        # expand depthwise to dense for the exact numpy check
        W_full = W
        if kind == "dw":
            W_full = np.zeros((CHANNELS, CHANNELS, kh, kw), dtype=np.int64)
            for c in range(CHANNELS):
                W_full[c, c] = W[c, 0]
        if not verify_conv_numpy(pairs, W_full, B, kh, kw):
            continue
        use_bias = bool(B.any())
        groups = 10 if kind == "dw" else 1
        model = builders.conv_network(
            W, kh, kw, bias=B if use_bias else None, groups=groups)
        tag = ("dwconv" if kind == "dw" else "conv") + f"{kh}x{kw}"
        meta = {"method": tag + ("+b" if use_bias else "")}
        return model, meta
    return None
