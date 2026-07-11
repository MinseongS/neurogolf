#!/usr/bin/env python3
"""legacy_chain_scan.py — structural classifier over ALL 400 deployed nets.

Validated 2026-07-11 C/I lane discriminator (26/26): a rebuild WINS iff the deployed
incumbent is a LEGACY Slice/Where/Gather/Concat/Equal/Pad chain; it FLOORS iff the
incumbent is already an optimized Einsum/Conv/QLinear* net realizing the minimal
free-output construction.

This tool mechanizes the discriminator. For every submission/overfit_nets/taskNNN.onnx it:
  - attributes counted memory cost per node output to op-type buckets
  - computes legacy_frac = counted-cost through LEGACY ops / total counted memory
  - counts counted intermediate planes (node outputs that carry counted bytes)
  - flags presence of any Einsum
  - pulls deployed cost + points from state/manifest.json
Then applies the actionable filter (points<19.5, cost>400, not in any wall/floor/done list)
and prints a ranked table (highest legacy_frac * gap first).

Usage:  uv run python tools/legacy_chain_scan.py [--all] [--json OUT]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import shape_inference

ROOT = Path(__file__).resolve().parent.parent
OVERFIT = ROOT / "submission" / "overfit_nets"
MANIFEST = ROOT / "state" / "manifest.json"

# grader itemsize by onnx elem_type (matches minmerge.static_cost / scoring.DT)
DT = {1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8, 12: 4, 13: 8, 16: 2}

# op-type buckets
LEGACY_OPS = {"Slice", "Where", "Gather", "GatherElements", "GatherND", "Concat",
              "Equal", "Pad", "Scatter", "ScatterND", "ScatterElements"}
CONSOLIDATED_OPS = {"Einsum", "Conv", "ConvInteger", "QLinearConv", "QLinearMatMul",
                    "MatMul", "Gemm"}


def per_node_cost(model_path):
    """Return (total_counted_mem, bucket_costs, n_counted_planes, has_einsum, op_hist).

    bucket_costs = {'legacy': bytes, 'consolidated': bytes, 'other': bytes}.
    Attributes each node's counted output bytes to the bucket of the node op_type.
    """
    try:
        m = onnx.load(str(model_path))
        m = shape_inference.infer_shapes(m)
    except Exception as e:
        return None
    g = m.graph
    vi = {}
    for v in list(g.value_info) + list(g.output) + list(g.input):
        dims = tuple(d.dim_value if d.HasField("dim_value") else 0
                     for d in v.type.tensor_type.shape.dim)
        vi[v.name] = (v.type.tensor_type.elem_type, dims)

    buckets = {"legacy": 0, "consolidated": 0, "other": 0}
    total = 0
    n_planes = 0
    has_einsum = False
    op_hist = {}
    for n in g.node:
        op = n.op_type
        op_hist[op] = op_hist.get(op, 0) + 1
        if op == "Einsum":
            has_einsum = True
        node_bytes = 0
        for o in n.output:
            if o in ("input", "output") or not o:
                continue
            et, dims = vi.get(o, (None, None))
            if not dims or any(d == 0 for d in dims):
                continue
            c = 1
            for d in dims:
                c *= d
            b = c * DT.get(et, 4)
            node_bytes += b
        if node_bytes > 0:
            n_planes += 1
        total += node_bytes
        if op in LEGACY_OPS:
            buckets["legacy"] += node_bytes
        elif op in CONSOLIDATED_OPS:
            buckets["consolidated"] += node_bytes
        else:
            buckets["other"] += node_bytes
    params = sum(int(np.prod(i.dims)) for i in g.initializer)
    return total, buckets, n_planes, has_einsum, op_hist, params


# ---- exclusion lists (from worklists + levers.yaml, 2026-07-11) ----
# ci_triage DO-NOT-DISPATCH walls
CI_WALLS = {66, 76, 96, 101, 138, 145, 148, 157, 158, 173, 187, 191, 192, 198, 204,
            216, 243, 255, 279, 284, 328, 349, 350, 338, 324, 364, 366, 396, 25, 23,
            42, 44, 219, 319}
# ci_triage recently adopted / near-floor
CI_NEARFLOOR = {2, 5, 15, 17, 22, 80, 85, 86, 89, 92, 97, 98, 114, 120, 132, 133, 161,
                193, 234, 264, 256, 265, 294, 333, 342, 363}
# ci_triage 20 DRY floors (wave1+2 judged)
CI_DRY = {4, 29, 310, 372, 110, 365, 340, 202, 196, 125, 93, 9, 378, 323, 5, 346,
          228, 336, 58, 45, 237, 131, 332, 214, 43}
# ci_triage 6 wins already adopted
CI_WINS = {267, 72, 217, 304, 275, 123}
# profile_compile LANE CLOSED judged (wave1+2+S) + wins
PC_JUDGED = {208, 14, 359, 91, 177, 185, 84, 55, 363, 17, 232, 297, 335, 175, 244,
             301, 246, 293, 183, 60, 10, 239, 151, 259, 47, 278, 387, 377, 50}
# levers free-output-einsum-regime-crack done
FO_DONE = {329, 303, 141, 33, 341, 260, 75, 240, 159, 345, 109, 295, 392, 398, 61, 51,
           287, 94, 63, 199, 273, 224, 246, 190, 251, 268, 205, 222}
FO_FLOOR = {112, 163, 99, 102, 62, 12, 348, 124, 354, 90, 390, 154, 34, 168, 381, 71}

EXCLUDED = CI_WALLS | CI_NEARFLOOR | CI_DRY | CI_WINS | PC_JUDGED | FO_DONE | FO_FLOOR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="show all 400, not just actionable")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    manifest = json.load(open(MANIFEST))
    rows = []
    for t in range(1, 401):
        p = OVERFIT / f"task{t:03d}.onnx"
        if not p.exists():
            continue
        res = per_node_cost(p)
        if res is None:
            continue
        total, buckets, n_planes, has_einsum, op_hist, params = res
        mkey = f"{t:03d}"
        mrow = manifest.get(mkey, {})
        cost = mrow.get("cost")
        points = mrow.get("points")
        if cost is None:
            cost = total + params
        if points is None:
            points = max(1.0, 25 - math.log(max(cost, 1)))
        legacy_frac = buckets["legacy"] / total if total else 0.0
        consol_frac = buckets["consolidated"] / total if total else 0.0
        # gap = ln(cost) headroom below 25pt (how many points are on the table)
        gap = max(0.0, 25 - points)
        excluded = t in EXCLUDED
        actionable = (points < 19.5) and (cost > 400) and (not excluded)
        # legacy score: fraction of counted cost in legacy ops, weighted by gap
        score = legacy_frac * gap
        rows.append({
            "task": t, "cost": cost, "points": round(points, 3),
            "legacy_frac": round(legacy_frac, 3), "consol_frac": round(consol_frac, 3),
            "legacy_bytes": buckets["legacy"], "consol_bytes": buckets["consolidated"],
            "n_planes": n_planes, "has_einsum": has_einsum,
            "gap": round(gap, 3), "score": round(score, 3),
            "excluded": excluded, "actionable": actionable,
            "ops": op_hist,
        })

    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)

    shown = rows if args.all else [r for r in rows if r["actionable"]]
    shown.sort(key=lambda r: r["score"], reverse=True)

    print(f"{'task':>4} {'cost':>6} {'pts':>6} {'lfrac':>5} {'cfrac':>5} "
          f"{'plns':>4} {'eins':>4} {'gap':>5} {'score':>6}  actionable")
    for r in shown:
        print(f"{r['task']:>4} {r['cost']:>6} {r['points']:>6.2f} "
              f"{r['legacy_frac']:>5.2f} {r['consol_frac']:>5.2f} {r['n_planes']:>4} "
              f"{'Y' if r['has_einsum'] else '.':>4} {r['gap']:>5.2f} {r['score']:>6.2f}  "
              f"{'*' if r['actionable'] else ''}")
    print(f"\n{len(shown)} rows shown; {sum(r['actionable'] for r in rows)} actionable of "
          f"{len(rows)} nets scanned")


if __name__ == "__main__":
    main()
