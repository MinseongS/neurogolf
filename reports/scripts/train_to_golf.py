#!/usr/bin/env python3
"""Train-to-golf factory trainer (S12).

Fits a SINGLE-NODE Conv(10,10,k,k) net on fresh arc-gen data to replace an
expensive hand-compiled incumbent.  Grading blind spot: a single-node net
has no intermediate tensors => mem 0, cost = params only (k=3 no-bias ->
900 elems -> 18.20pt).

Key reduction: the conv output at a cell depends ONLY on its kxk one-hot
patch, so training operates on the DEDUPLICATED patch->label set (a few
thousand rows instead of millions of redundant cells).  Zero hinge
violations on the unique-patch set => exact grading on EVERY input composed
of seen patches; fresh-gate failures can then only come from novel patches
(coverage), which more generation fixes, not more training.

Per patch x (flattened one-hot, 10*k*k) with label c in 0..9:
  w_c . x >= M_POS   and   w_j . x <= -M_NEG for j != c
label 10 (blank cell, e.g. outside the output grid):
  w_j . x <= -M_NEG for all j
The all-zero patch is exact-0 output for a no-bias conv (grades blank
correctly for free) and is excluded.  Margins trained large to kill the
0.0-threshold knife-edge (task220/230/294 lesson).

On success writes:
  reports/candidates/taskNNN_t2g.py   (build(task) module for fresh_verify)
  reports/candidates/taskNNN_t2g.onnx
and prints bundled-eval result + cost/points delta vs the incumbent.

Usage: train_to_golf.py TASK [--k 3] [--bias] [--ngen 8000] [--steps 20000]
Exit 0 = candidate written + bundled fail=0; 2 = gate failed.
"""
import argparse, importlib, json, math, os, signal, sys, time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
_ARCGEN = os.path.join(_ROOT, "arc-gen")
sys.path.append(_ARCGEN)

BLANK = 10


def encode(grid):
    g = np.asarray(grid, dtype=np.int8)
    h, w = g.shape
    if max(h, w) > 30:
        return None
    full = np.full((30, 30), BLANK, dtype=np.int8)
    full[:h, :w] = g
    return full


def collect_patches(gen, k, ngen, time_cap):
    """Dedup patch->label-histogram over ngen fresh instances.  Contradictory
    patches (task not perfectly k-local, or generator noise) are resolved by
    MAJORITY label — the val exact-fail rate then measures the real
    instance-level damage against the relaxed adoption gate (fresh >=98%).
    Returns (patches [M,k*k] int8, labels [M], val examples, err)."""
    p = k // 2
    seen = {}
    val = []
    t0 = time.time()
    n_ok = 0
    signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError()))
    while n_ok < ngen and time.time() - t0 < time_cap:
        try:
            signal.alarm(5)
            ex = gen.generate()
        except Exception:
            continue
        finally:
            signal.alarm(0)
        gi, go = np.asarray(ex["input"]), np.asarray(ex["output"])
        if gi.ndim != 2 or go.ndim != 2 or gi.shape != go.shape:
            return None, None, None, None, "shape-mismatch"
        ei, eo = encode(gi), encode(go)
        if ei is None or eo is None:
            continue
        n_ok += 1
        if n_ok % 10 == 0 and len(val) < 400:
            val.append((ei, eo))
            continue
        pad = np.full((30 + 2 * p, 30 + 2 * p), BLANK, dtype=np.int8)
        pad[p:p + 30, p:p + 30] = ei
        win = np.lib.stride_tricks.sliding_window_view(pad, (k, k))
        flat = win.reshape(900, k * k)
        labels = eo.reshape(900)
        keep = ~(flat == BLANK).all(axis=1)
        for patch, lab in zip(flat[keep], labels[keep]):
            key = patch.tobytes()
            hist = seen.get(key)
            if hist is None:
                hist = seen[key] = np.zeros(11, dtype=np.int64)
            hist[int(lab)] += 1
    patches = np.frombuffer(b"".join(seen.keys()), dtype=np.int8).reshape(len(seen), k * k)
    hists = np.stack(list(seen.values()))
    labels = hists.argmax(axis=1)
    n_contra = int((np.count_nonzero(hists, axis=1) > 1).sum())
    minority = int(hists.sum() - hists.max(axis=1).sum())
    print(f"  contradictory patches={n_contra}/{len(patches)} "
          f"(minority obs={minority}, majority-resolved)", flush=True)
    counts = hists.max(axis=1)
    return patches, labels, counts, val, None


def patches_to_onehot(patches, k):
    """[M, k*k] int8 -> [M, 10*k*k] float32 (BLANK -> all-zero column)."""
    M = len(patches)
    x = np.zeros((M, 10, k * k), dtype=np.float32)
    rows = np.repeat(np.arange(M), k * k)
    cols = np.tile(np.arange(k * k), M)
    ch = patches.reshape(-1).astype(np.int64)
    mask = ch < 10
    x[rows[mask], ch[mask], cols[mask]] = 1.0
    return x.reshape(M, 10 * k * k)


def conv_exact_fail(W, B, k, val):
    """#val examples where (conv(x)>0) != onehot(out). numpy direct conv via
    the same patch trick."""
    p = k // 2
    fails = 0
    Wm = W.reshape(10, 10 * k * k)
    for ei, eo in val:
        x = np.zeros((10, 30, 30), dtype=np.float32)
        m = ei < 10
        rr, cc = np.nonzero(m)
        x[ei[m].astype(np.int64), rr, cc] = 1.0
        xpad = np.zeros((10, 30 + 2 * p, 30 + 2 * p), dtype=np.float32)
        xpad[:, p:p + 30, p:p + 30] = x
        win = np.lib.stride_tricks.sliding_window_view(xpad, (k, k), axis=(1, 2))
        feat = win.transpose(1, 2, 0, 3, 4).reshape(900, 10 * k * k)
        out = feat @ Wm.T
        if B is not None:
            out = out + B
        pred = (out > 0.0)
        truth = np.zeros((900, 10), dtype=bool)
        lab = eo.reshape(900)
        inb = lab < 10
        truth[np.nonzero(inb)[0], lab[inb].astype(np.int64)] = True
        if not np.array_equal(pred, truth):
            fails += 1
    return fails


def lp_fit(patches, labels, counts, k, m_pos=1.0, m_neg=0.2,
           seed=0, max_rounds=40):
    """Exact per-channel linear feasibility via cutting-plane LP.

    Solves each output channel independently: rows with label c need
    w_c.x >= m_pos, all other rows need w_c.x <= -m_neg, bias <= -m_neg.
    Subset-infeasible => globally infeasible (definitive kill at this k);
    feasible with zero violations on the full set => definitive solve.
    Returns (W [10,10,k,k], B [10]) or None if any channel is infeasible.
    """
    from scipy.optimize import linprog

    M = len(patches)
    k2 = k * k
    rng = np.random.default_rng(seed)

    def features(idx):
        x = patches_to_onehot(patches[idx], k)
        return np.hstack([x, np.ones((len(x), 1), dtype=np.float32)])

    def channel_out(w):
        # w: [10*k2+1]; evaluate w.x for ALL patches chunked via the
        # embedding trick (blank cells contribute 0)
        table = np.vstack([w[:-1].reshape(10, k2),
                           np.zeros((1, k2), dtype=w.dtype)])  # [11,k2]
        out = np.empty(M, dtype=np.float64)
        cols = np.arange(k2)
        for s in range(0, M, 500_000):
            p = patches[s:s + 500_000].astype(np.int64)
            out[s:s + 500_000] = table[p, cols].sum(axis=1)
        return out + w[-1]

    nv = 10 * k2 + 1
    Ws = []
    for c in range(10):
        is_pos = labels == c
        # start with the highest-count rows (most instance-relevant)
        order = np.argsort(-counts)
        active = set(order[:min(M, 20_000)].tolist())
        sol = None
        for rnd in range(max_rounds):
            idx = np.fromiter(active, dtype=np.int64)
            X = features(idx)
            pos = is_pos[idx]
            A = np.vstack([-X[pos], X[~pos], np.eye(nv)[-1:]])
            b = np.concatenate([-np.full(int(pos.sum()), m_pos),
                                -np.full(int((~pos).sum()), m_neg), [-m_neg]])
            r = linprog(np.zeros(nv), A_ub=A, b_ub=b, bounds=(None, None),
                        method="highs")
            if r.status == 2:
                print(f"  LP ch{c}: INFEASIBLE (subset of {len(idx)}, "
                      f"round {rnd}) -> no solution at k={k}", flush=True)
                return None
            if r.status != 0:
                print(f"  LP ch{c}: solver status {r.status} ({r.message})",
                      flush=True)
                return None
            sol = r.x
            out = channel_out(sol)
            viol = np.nonzero(np.where(is_pos, out < m_pos * 0.999,
                                       out > -m_neg * 0.999))[0]
            viol = np.setdiff1d(viol, idx, assume_unique=False)
            if len(viol) == 0:
                print(f"  LP ch{c}: solved (round {rnd}, "
                      f"active={len(idx)})", flush=True)
                break
            take = viol[np.argsort(-counts[viol])][:20_000]
            active.update(take.tolist())
        else:
            print(f"  LP ch{c}: cutting-plane did not converge", flush=True)
            return None
        Ws.append(sol)
    Wmat = np.stack(Ws)                                   # [10, nv]
    W = Wmat[:, :-1].reshape(10, 10, k, k).astype(np.float32)
    B = Wmat[:, -1].astype(np.float32)
    return W, B


def write_candidate(task, W, B, k, out_py, out_onnx):
    import io, base64
    import onnx
    from onnx import helper, numpy_helper, TensorProto

    inputs = ["input", "W"] + (["B"] if B is not None else [])
    node = helper.make_node("Conv", inputs, ["output"],
                            kernel_shape=[k, k], pads=[k // 2] * 4)
    inits = [numpy_helper.from_array(W.astype(np.float32), "W")]
    if B is not None:
        inits.append(numpy_helper.from_array(B.astype(np.float32), "B"))
    graph = helper.make_graph(
        [node], f"task{task:03d}_t2g",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.save(model, out_onnx)

    def b64(a):
        buf = io.BytesIO()
        np.save(buf, a.astype(np.float32))
        return base64.b64encode(buf.getvalue()).decode()

    binit = "" if B is None else f"\n        tensor('B', arr_b64('{b64(B)}')),"
    bref = "" if B is None else ", 'B'"
    src = f'''"""Task {task:03d} — train-to-golf single-Conv net (S12 factory).

Fit on deduplicated fresh arc-gen patches with asymmetric margin hinge
(reports/scripts/train_to_golf.py). Grading: (out>0.0) one-hot match.
"""
from onnx import TensorProto, helper
from ._exact import arr_b64, model, tensor


def build(task):
    inits = [
        tensor('W', arr_b64('{b64(W)}')),{binit}
    ]
    nodes = [
        helper.make_node('Conv', ['input', 'W'{bref}], ['output'],
                         kernel_shape=[{k}, {k}], pads=[{k // 2}] * 4),
    ]
    return model('task{task:03d}_t2g', nodes, inits, output_dtype=1, opset=13,
                 value_infos=[])
'''
    with open(out_py, "w") as f:
        f.write(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", type=int)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--bias", action="store_true", default=False)
    ap.add_argument("--ngen", type=int, default=8000)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen-time-cap", type=float, default=420.0)
    ap.add_argument("--train-time-cap", type=float, default=420.0)
    ap.add_argument("--lp", action="store_true", default=False,
                    help="exact cutting-plane LP instead of SGD (definitive "
                         "feasible/infeasible; implies bias)")
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mapping = json.load(open(os.path.join(_ROOT, "reports", "arc_mapping.json")))
    arc = mapping[str(args.task)]["arc_id"]
    gen = importlib.import_module(f"tasks.task_{arc}")

    print(f"task{args.task:03d} arc={arc} k={args.k} bias={args.bias}: "
          f"collecting patches from {args.ngen} instances...", flush=True)
    patches, labels, counts, val, err = collect_patches(gen, args.k, args.ngen,
                                                        args.gen_time_cap)
    if err:
        print(f"FAIL: {err}")
        sys.exit(2)
    if args.bias:
        # the all-zero patch is excluded from collection (a no-bias conv is
        # exact-0 there), but with bias it outputs B itself -> constrain
        # B <= -M_NEG by adding the zero patch with a blank label.
        patches = np.concatenate([patches,
                                  np.full((1, args.k * args.k), BLANK, np.int8)])
        labels = np.concatenate([labels, np.array([BLANK], dtype=np.int64)])
        counts = np.concatenate([counts, np.array([counts.max()], dtype=counts.dtype)])
    M = len(patches)
    print(f"  unique patches={M} val_examples={len(val)} "
          f"blank_labels={(labels == BLANK).sum()}", flush=True)

    if args.lp:
        fit = lp_fit(patches, labels, counts, args.k, seed=args.seed)
        if fit is None:
            print("FAIL: LP infeasible/unconverged — definitive at this k "
                  "if INFEASIBLE was printed")
            sys.exit(2)
        W, B = fit
        return finish(args, W, B, val)

    # Embedding-gather formulation: patch cells index a [10, 11, k*k] weight
    # table whose slot 10 (BLANK) is welded to zero -- equivalent to the
    # one-hot linear model but O(k^2) per row instead of O(10*k^2), and the
    # patch bank stays int8 numpy (millions of unique patches at k=7).
    k2 = args.k * args.k
    pos_np = np.zeros((M, 10), dtype=np.float32)
    inb = labels < 10
    pos_np[np.nonzero(inb)[0], labels[inb]] = 1.0

    Wt = torch.zeros((10, 10, k2), requires_grad=True)
    params = [Wt]
    Bt = None
    if args.bias:
        Bt = torch.zeros(10, requires_grad=True)
        params.append(Bt)
    opt = torch.optim.Adam(params, lr=2e-2)
    M_POS, M_NEG = 1.0, 0.2
    zero_slot = torch.zeros((10, 1, k2))
    cols = torch.arange(k2)

    def forward(idx_np):
        p = torch.from_numpy(patches[idx_np].astype(np.int64))  # [B, k2]
        Wpad = torch.cat([Wt, zero_slot], dim=1)                # [10, 11, k2]
        # out[b, c] = sum_j Wpad[c, p[b, j], j]
        return Wpad[:, p, cols].sum(-1).T + (Bt if Bt is not None else 0.0)

    def full_violations():
        vp = vn = 0
        pmin, nmax = np.inf, -np.inf
        viol_mask = np.zeros(M, dtype=bool)
        with torch.no_grad():
            for s in range(0, M, 200_000):
                o = forward(np.arange(s, min(M, s + 200_000))).numpy()
                p = pos_np[s:s + 200_000]
                bad = ((o <= 0) & (p > 0)) | ((o > 0) & (p == 0))
                viol_mask[s:s + 200_000] = bad.any(axis=1)
                vp += int(((o <= 0) & (p > 0)).sum())
                vn += int(((o > 0) & (p == 0)).sum())
                if (p > 0).any():
                    pmin = min(pmin, float(o[p > 0].min()))
                nmax = max(nmax, float(o[p == 0].max()))
        return vp, vn, pmin, nmax, viol_mask

    # frequency-weighted loss: violations on common patches cost more
    # instance-level accuracy than violations on rare ones (sqrt tempering);
    # rows found violated at the last full check get a 10x boost to fight
    # mean-loss dilution in the endgame.
    base_w = np.sqrt(counts.astype(np.float32))
    base_w /= base_w.mean()
    weight_np = base_w.copy()

    BATCH = min(M, 16384)
    check_every = 500 if M <= 300_000 else 2000
    t0 = time.time()
    solved = False
    rng = np.random.default_rng(args.seed)
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
            viol_pos, viol_neg, pmin, nmax, viol_mask = full_violations()
            print(f"  step {step}: loss={loss.item():.6f} "
                  f"viol=({viol_pos},{viol_neg}) pos_min={pmin:.3f} "
                  f"neg_max={nmax:.3f}", flush=True)
            if viol_pos == 0 and viol_neg == 0 and pmin >= 0.5 * M_POS \
                    and nmax <= -0.5 * M_NEG:
                solved = True
                break
            weight_np = base_w * np.where(viol_mask, 10.0, 1.0)
        if time.time() - t0 > args.train_time_cap:
            print("  (train time cap)")
            break

    if not solved:
        # best-effort: a few stubborn patches may be rare enough that the
        # relaxed adoption gate (fresh >=98%) still passes -- val decides.
        print("NOTE: hinge not fully solved; gating on val exact-fail rate")

    W = Wt.detach().numpy().reshape(10, 10, args.k, args.k)
    B = Bt.detach().numpy() if Bt is not None else None
    return finish(args, W, B, val)


def finish(args, W, B, val):
    vf = conv_exact_fail(W, B, args.k, val)
    rate = vf / max(1, len(val))
    print(f"  val exact-fail = {vf}/{len(val)} ({100 * rate:.2f}%)")
    if rate > 0.015:
        # >1.5% is outside the relaxed adoption gate (fresh >=98%) with
        # margin for val noise. More --ngen (coverage) or larger k may fix.
        print("FAIL: val fail rate above relaxed gate")
        sys.exit(2)

    os.makedirs(os.path.join(_ROOT, "reports", "candidates"), exist_ok=True)
    out_py = os.path.join(_ROOT, "reports", "candidates", f"task{args.task:03d}_t2g.py")
    out_onnx = os.path.join(_ROOT, "reports", "candidates", f"task{args.task:03d}_t2g.onnx")
    write_candidate(args.task, W, B, args.k, out_py, out_onnx)

    from src.harness import load_task, evaluate
    res = evaluate(out_onnx, load_task(args.task))
    nparams = int(W.size + (B.size if B is not None else 0))
    census = json.load(open(os.path.join(_ROOT, "reports", "blocker_census.json")))
    inc = next(t for t in census["tasks"] if t["task"] == args.task)
    new_pts = 25 - math.log(max(1, nparams))
    print(f"BUNDLED: pass={res['pass']} fail={res['fail']} err={res['error']} "
          f"mem={res['memory']} params={res['params']}")
    print(f"COST: candidate params={nparams} pts={new_pts:.3f} vs incumbent "
          f"cost={inc['manifest_cost']} pts={inc['current_points']:.3f} "
          f"delta={new_pts - inc['current_points']:+.3f}")
    print(f"WROTE {out_py}")
    sys.exit(0 if res["fail"] == 0 and not res["error"] else 2)


if __name__ == "__main__":
    main()
