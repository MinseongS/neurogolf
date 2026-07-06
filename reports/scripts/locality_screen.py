#!/usr/bin/env python3
"""Locality screener for the train-to-golf factory (S12).

A single Conv(10,10,k,k) net graded by (out>0.0) is a per-cell linear
classifier over the kxk one-hot input neighborhood.  A NECESSARY condition
for such a net to exist is that the output color of every cell is a
FUNCTION of its kxk input neighborhood (no contradictions across fresh
arc-gen instances).  This script measures that contradiction rate per task
per k in {3,5,7} so training effort is only spent where a solution can
exist.  Linear separability is then the trainer's problem.

Cell encoding: int 0..9 = color (one-hot channel), 10 = blank (all-zero
cell: outside the HxW grid but inside the 30x30 tensor, or conv zero-pad
outside the tensor -- identical to the conv's view).
Label per 30x30 cell: output color 0..9, or 10 = must be all-nonpositive
(outside the output grid).

Usage: locality_screen.py [--min-cost 910] [--n 300] [--tasks 1,2,...]
Writes reports/locality_screen.json + prints a ranked table.
"""
import argparse, importlib, json, os, signal, sys, time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
_ARCGEN = os.path.join(_ROOT, "arc-gen")
sys.path.append(_ARCGEN)

KS = (3, 5, 7)
BLANK = 10


def encode(grid):
    """HxW color grid -> 30x30 int grid with BLANK padding."""
    g = np.asarray(grid, dtype=np.int8)
    h, w = g.shape
    if max(h, w) > 30:
        return None
    full = np.full((30, 30), BLANK, dtype=np.int8)
    full[:h, :w] = g
    return full


def screen_task(tasknum, arc_id, n, time_cap=90.0):
    gen = importlib.import_module(f"tasks.task_{arc_id}")
    maps = {k: {} for k in KS}          # patch bytes -> label
    contra = {k: 0 for k in KS}
    cells = {k: 0 for k in KS}
    shape_mismatch = 0
    n_ok = 0
    t0 = time.time()
    for _ in range(n):
        if time.time() - t0 > time_cap:
            break
        try:
            # some generators wedge inside a single generate() call (task076
            # stalled 20min); SIGALRM enforces a hard per-call timeout.
            signal.alarm(5)
            ex = gen.generate()
        except Exception:
            continue
        finally:
            signal.alarm(0)
        gi, go = np.asarray(ex["input"]), np.asarray(ex["output"])
        if gi.ndim != 2 or go.ndim != 2:
            continue
        ei, eo = encode(gi), encode(go)
        if ei is None or eo is None:
            continue
        n_ok += 1
        if gi.shape != go.shape:
            shape_mismatch += 1
            continue
        for k in KS:
            p = k // 2
            pad = np.full((30 + 2 * p, 30 + 2 * p), BLANK, dtype=np.int8)
            pad[p:p + 30, p:p + 30] = ei
            # sliding windows over the padded plane -> one patch per cell
            win = np.lib.stride_tricks.sliding_window_view(pad, (k, k))
            flat = win.reshape(900, k * k)
            labels = eo.reshape(900)
            # skip all-blank patches with blank label (trivial, dominates)
            keep = ~((flat == BLANK).all(axis=1) & (labels == BLANK))
            m = maps[k]
            for patch, lab in zip(flat[keep], labels[keep]):
                key = patch.tobytes()
                prev = m.get(key)
                if prev is None:
                    m[key] = int(lab)
                elif prev != int(lab):
                    contra[k] += 1
                cells[k] += 1
    return {
        "n_ok": n_ok,
        "shape_mismatch": shape_mismatch,
        "contra": {str(k): contra[k] for k in KS},
        "cells": {str(k): cells[k] for k in KS},
        "rate": {str(k): (contra[k] / cells[k] if cells[k] else None) for k in KS},
    }


def _alarm(signum, frame):
    raise TimeoutError("generate() timed out")


def main():
    signal.signal(signal.SIGALRM, _alarm)
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-cost", type=int, default=910)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--tasks", type=str, default=None)
    ap.add_argument("--out", type=str,
                    default=os.path.join(_ROOT, "reports", "locality_screen.json"))
    args = ap.parse_args()

    census = json.load(open(os.path.join(_ROOT, "reports", "blocker_census.json")))
    cost = {t["task"]: t["manifest_cost"] for t in census["tasks"]}
    pts = {t["task"]: t["current_points"] for t in census["tasks"]}
    mapping = json.load(open(os.path.join(_ROOT, "reports", "arc_mapping.json")))

    if args.tasks:
        tasks = [int(x) for x in args.tasks.split(",")]
    else:
        tasks = sorted(t for t, c in cost.items() if c >= args.min_cost)

    results = {}
    for t in tasks:
        arc = mapping[str(t)]["arc_id"]
        try:
            r = screen_task(t, arc, args.n)
        except Exception as e:
            r = {"error": f"{type(e).__name__}: {e}"}
        r["cost"] = cost.get(t)
        r["points"] = pts.get(t)
        results[str(t)] = r
        rate = r.get("rate", {})
        tag = " ".join(f"k{k}={rate.get(str(k)) if rate.get(str(k)) is not None else '-'}"
                       for k in KS) if "rate" in r else r.get("error", "?")
        mm = r.get("shape_mismatch", "?")
        print(f"task{t:03d} cost={cost.get(t)} n_ok={r.get('n_ok','?')} "
              f"mismatch={mm} {tag}", flush=True)

    json.dump(results, open(args.out, "w"), indent=1)

    # ranked summary: same-shape + zero contradictions at some k
    print("\n=== TRAINABLE CANDIDATES (same-shape, contradiction-free at some k) ===")
    rows = []
    for t, r in results.items():
        if "rate" not in r or r.get("n_ok", 0) < 50 or r.get("shape_mismatch", 1) > 0:
            continue
        best_k = None
        for k in KS:
            rt = r["rate"].get(str(k))
            if rt == 0.0:
                best_k = k
                break
        if best_k is None:
            continue
        params = 100 * best_k * best_k + 10
        gain = (25 - np.log(max(1, params))) - r["points"]
        if gain <= 0:
            continue
        rows.append((gain, int(t), best_k, r["cost"]))
    rows.sort(reverse=True)
    for gain, t, k, c in rows:
        print(f"task{t:03d} k={k} cost={c} potential_gain={gain:+.3f}")


if __name__ == "__main__":
    main()
