#!/usr/bin/env python3
"""Phase-2 probe: can a 1-hidden-layer conv (Conv kxk 10->C, ReLU, Conv 1x1
C->10) solve a task's unique-patch set where the linear single-Conv could not?

Solvability-only — no ONNX is built.  Economics: the 2 counted intermediates
cost 7200*C bytes fp32, so this only pays for incumbents above ~7700 (C=1).

Usage: probe_hidden.py TASK K C [--ngen 12000] [--steps 60000]
"""
import argparse, importlib, json, os, sys, time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.append(os.path.join(_ROOT, "arc-gen"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_to_golf import collect_patches, BLANK  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", type=int)
    ap.add_argument("k", type=int)
    ap.add_argument("C", type=int)
    ap.add_argument("--ngen", type=int, default=12000)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--gen-time-cap", type=float, default=700.0)
    ap.add_argument("--train-time-cap", type=float, default=1500.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed)

    mapping = json.load(open(os.path.join(_ROOT, "reports", "arc_mapping.json")))
    arc = mapping[str(args.task)]["arc_id"]
    gen = importlib.import_module(f"tasks.task_{arc}")
    print(f"task{args.task:03d} arc={arc} k={args.k} C={args.C}: collecting...",
          flush=True)
    patches, labels, counts, val, err = collect_patches(
        gen, args.k, args.ngen, args.gen_time_cap)
    if err:
        print(f"FAIL: {err}")
        sys.exit(2)
    M = len(patches)
    k2 = args.k * args.k
    print(f"  unique patches={M}", flush=True)

    pos_np = np.zeros((M, 10), dtype=np.float32)
    inb = labels < 10
    pos_np[np.nonzero(inb)[0], labels[inb]] = 1.0
    base_w = np.sqrt(counts.astype(np.float32))
    base_w /= base_w.mean()
    weight_np = base_w.copy()

    C = args.C
    W1 = torch.randn(C, 10, k2, requires_grad=True)  # hidden conv
    b1 = torch.zeros(C, requires_grad=True)
    W2 = torch.randn(10, C, requires_grad=True)
    b2 = torch.zeros(10, requires_grad=True)
    with torch.no_grad():
        W1 *= 0.3
        W2 *= 0.3
    opt = torch.optim.Adam([W1, b1, W2, b2], lr=1e-2)
    M_POS, M_NEG = 1.0, 0.2
    zero1 = torch.zeros(C, 1, k2)
    cols = torch.arange(k2)

    def forward(idx_np):
        p = torch.from_numpy(patches[idx_np].astype(np.int64))
        Wp = torch.cat([W1, zero1], dim=1)              # [C, 11, k2]
        h = Wp[:, p, cols].sum(-1).T + b1               # [B, C]
        return torch.relu(h) @ W2.T + b2                # [B, 10]

    def full_violations():
        vp = vn = 0
        viol_mask = np.zeros(M, dtype=bool)
        with torch.no_grad():
            for s in range(0, M, 200_000):
                o = forward(np.arange(s, min(M, s + 200_000))).numpy()
                p = pos_np[s:s + 200_000]
                bad = ((o <= 0) & (p > 0)) | ((o > 0) & (p == 0))
                viol_mask[s:s + 200_000] = bad.any(axis=1)
                vp += int(((o <= 0) & (p > 0)).sum())
                vn += int(((o > 0) & (p == 0)).sum())
        return vp, vn, viol_mask

    BATCH = min(M, 16384)
    check_every = 1000 if M <= 300_000 else 2500
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    best_viol = None
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, M, BATCH) if BATCH < M else np.arange(M)
        out = forward(idx)
        pos = torch.from_numpy(pos_np[idx])
        w = torch.from_numpy(weight_np[idx])[:, None]
        loss = (w * (pos * torch.relu(M_POS - out)
                     + (1.0 - pos) * torch.relu(out + M_NEG))).sum() / len(idx)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % check_every == 0 or step == args.steps:
            vp, vn, viol_mask = full_violations()
            tot = vp + vn
            best_viol = tot if best_viol is None else min(best_viol, tot)
            print(f"  step {step}: loss={loss.item():.6f} viol=({vp},{vn})",
                  flush=True)
            if tot == 0:
                print("SOLVED: hidden layer separates the patch set")
                sys.exit(0)
            weight_np = base_w * np.where(viol_mask, 10.0, 1.0)
        if time.time() - t0 > args.train_time_cap:
            print("  (time cap)")
            break
    print(f"NOT SOLVED: best_total_viol={best_viol}")
    sys.exit(2)


if __name__ == "__main__":
    main()
