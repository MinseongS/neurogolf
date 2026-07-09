"""Overfit public-dump min-merge (S18 lever, LB-proven +4.46 in one pass from urad 7242.52).

WHY THIS BEATS THE OLD mine_public_bundles.py FOR OVERFIT MODE
  - Baselines against `submission/overfit_nets/` (the ACTIVE best submission), not the safe
    manifest. In overfit mode the gate is bundled fail=0 ONLY (constant dataset => permanent),
    so NO fresh-gate is needed -- every cheaper bundled-fail=0 net is pure, permanent upside.
  - STATIC-cost prefilter first (sum of counted node-output bytes + initializer elems) so we
    only pay a full grader eval on candidates that are plausibly cheaper -> fast.
  - Verifies each candidate in an ISOLATED evaluate() call (matches Kaggle per-task grading and
    dodges the knife-edge batch-eval undercount, memory neurogolf-knife-edge-conv-flip).

ROUTINE (end to end)
  1. Find new higher-scoring uploaders:
       kaggle kernels list -s neurogolf --sort-by dateRun --page-size 30
  2. Pull each into its own dir and extract the 400 onnx (dumps arrive as submission.zip via
     `kaggle kernels output <ref> -p DIR`; older base64-embedded notebooks -> use
     mine_public_bundles.py's extractor instead):
       kaggle kernels output <ref> -p mine/<name>; unzip mine/<name>/submission.zip -d mine/<name>_nets
  3. Min-merge:
       uv run python -m reports.scripts.mine_overfit_minmerge mine/urad_nets mine/poby_nets ...
     -> prints every task where a dump net is cheaper AND bundled fail=0, with point delta.
  4. Adopt: re-run with --apply to install winners into submission/overfit_nets/ (backs up
     originals to <dir>/.minmerge_backup/), then rebuild + submit:
       cd submission/overfit_nets && zip -j ../../submission.zip task*.onnx && cd ../..
       uv run python reports/scripts/scan_unsigned_topk.py submission/overfit_nets
       kaggle competitions submit -c neurogolf-2026 -f submission.zip -m "..."

S18 RESULT: urad 7242.52 -> 6 tasks (355 +1.56, 264 +0.83, 197 +0.77, 222 +0.66, 236 +0.63,
383 +0.005). franksunp/poby 7240.26, jonathan 7242, seddiktrk = 0 additional (urad dominated).
"""
import argparse, glob, os, shutil, sys
import numpy as np
import onnx
from onnx import shape_inference
from src.harness import load_task, evaluate

OURS = "submission/overfit_nets"
DT = {1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8, 12: 4, 13: 8, 16: 2}


def static_cost(p):
    """approx grader (memory+params): sum counted node-output bytes + initializer element count."""
    try:
        m = onnx.load(p)
        m = shape_inference.infer_shapes(m)
    except Exception:
        return None
    g = m.graph
    vi = {}
    for v in list(g.value_info) + list(g.output) + list(g.input):
        dims = tuple(d.dim_value if d.HasField("dim_value") else 0 for d in v.type.tensor_type.shape.dim)
        vi[v.name] = (v.type.tensor_type.elem_type, dims)
    mem = 0
    for n in g.node:
        for o in n.output:
            if o in ("input", "output"):
                continue
            et, dims = vi.get(o, (None, None))
            if not dims or any(d == 0 for d in dims):
                continue
            c = 1
            for d in dims:
                c *= d
            mem += c * DT.get(et, 4)
    params = sum(int(np.prod(i.dims)) for i in g.initializer)
    return mem + params


def find_net(dumpdir, t):
    for pat in (f"{dumpdir}/task{t:03d}.onnx", f"{dumpdir}/**/task{t:03d}.onnx"):
        g = glob.glob(pat, recursive=True)
        if g:
            return g[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+", help="one or more extracted dump dirs (each with taskNNN.onnx)")
    ap.add_argument("--apply", action="store_true", help="install winners into submission/overfit_nets")
    ap.add_argument("--margin", type=int, default=20, help="min static-cost saving to bother eval-ing")
    args = ap.parse_args()

    wins = []  # (delta_pts, task, dumpname, path, cand_mem, cand_params)
    for t in range(1, 401):
        op = f"{OURS}/task{t:03d}.onnx"
        if not os.path.exists(op):
            continue
        oc = static_cost(op)
        if oc is None:
            continue
        # collect statically-cheaper candidates across dumps
        cands = []
        for dd in args.dumps:
            cp = find_net(dd, t)
            if not cp:
                continue
            cc = static_cost(cp)
            if cc is not None and cc < oc - args.margin:
                cands.append((cc, dd, cp))
        if not cands:
            continue
        # isolated exact eval: ours + each cheaper candidate
        our = evaluate(op, load_task(t))
        best = None
        for cc, dd, cp in sorted(cands):
            r = evaluate(cp, load_task(t))
            if r["ok"] and r["fail"] == 0 and r["points"] > our["points"] + 1e-6:
                if best is None or r["points"] > best[0]:
                    best = (r["points"], dd, cp, r["memory"], r["params"])
        if best:
            wins.append((round(best[0] - our["points"], 4), t, os.path.basename(best[1]), best[2], best[3], best[4]))

    wins.sort(reverse=True)
    total = 0.0
    print(f"{'dP':>7} {'task':>4} {'dump':>28} {'mem':>6} {'par':>5}")
    for dp, t, dn, cp, mem, par in wins:
        total += dp
        print(f"{dp:7.4f} {t:4d} {dn:>28} {mem:6d} {par:5d}")
    print(f"\nTOTAL adoptable: +{round(total, 4)} across {len(wins)} tasks")

    if args.apply and wins:
        bdir = f"{OURS}/.minmerge_backup"
        os.makedirs(bdir, exist_ok=True)
        for dp, t, dn, cp, mem, par in wins:
            f = f"task{t:03d}.onnx"
            shutil.copy(f"{OURS}/{f}", f"{bdir}/{f}")
            shutil.copy(cp, f"{OURS}/{f}")
        print(f"APPLIED {len(wins)} nets to {OURS} (backups in {bdir}). "
              f"Now: rebuild submission.zip, scan_unsigned_topk, submit.", file=sys.stderr)


if __name__ == "__main__":
    main()
