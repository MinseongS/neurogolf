"""Deep-fold scan (runtime-timeout-spend lever, 2026-07-10).
Improves on src/neurogolf/scans/fold.py in exactly the three ways the old scanner
was blind:
  1. rank by dpts = ln(cost / (cost - plane_bytes))  (not absolute bytes)
  2. min plane 600B (was 1500B)
  3. indicator algebra: a plane produced/consumed by ReduceMax/MaxPool/Or/Where over
     {0,1}-valued data, whose every downstream path reaches the FREE graph output only
     through monotone ops, is a sum+(>0 free threshold) fold candidate — the old scanner
     classified these as unfoldable-nonlinear. Also adds Conv/ConvTranspose as
     sum-contraction consumers (missing from old REDUCERS).
Finder only — every hit needs mechanism-level verification before building.
"""
import json
import math
import sys
from pathlib import Path

import onnx
from onnx import shape_inference, TensorProto

ROOT = Path("/Users/minseong/project/neurogolf")
sys.path.insert(0, str(ROOT / "src"))
from neurogolf.manifest import load as load_manifest  # noqa: E402

NETS = ROOT / "submission" / "overfit_nets"
OUT = Path(__file__).parent / "deepfold_scan_20260711.json"

ITEMSIZE = {TensorProto.FLOAT: 4, TensorProto.FLOAT16: 2, TensorProto.DOUBLE: 8,
            TensorProto.INT64: 8, TensorProto.INT32: 4, TensorProto.INT16: 2,
            TensorProto.INT8: 1, TensorProto.UINT8: 1, TensorProto.UINT16: 2,
            TensorProto.UINT32: 4, TensorProto.UINT64: 8, TensorProto.BOOL: 1}

# sum-contraction consumers (linear fold against free input) — NOW including Conv family
SUM_REDUCERS = {"Einsum", "ReduceSum", "ReduceMean", "MatMul", "Gemm",
                "GlobalAveragePool", "ReduceL1", "Conv", "ConvTranspose"}
# ops through which a value's sign/positivity survives to the >0 output threshold
MONOTONE = {"Max", "ReduceMax", "MaxPool", "Or", "Add", "Mul", "Where", "Cast",
            "Concat", "Pad", "Reshape", "Unsqueeze", "Squeeze", "Transpose",
            "Identity", "Expand", "Tile", "Slice", "Gather", "Min", "Clip", "Relu"}
# producers whose output is linear in their (float) inputs
LINEAR_PRODUCERS = {"Einsum", "MatMul", "Gemm", "Conv", "ConvTranspose", "Add", "Sub",
                    "Mul", "ReduceSum", "ReduceMean", "Cast", "Reshape", "Transpose",
                    "Concat", "Pad", "Slice", "Gather", "GlobalAveragePool", "Resize",
                    "Unsqueeze", "Squeeze", "Expand", "Tile", "Identity"}
# indicator-ish producers: output is {0,1}/bool-like or a max over such
INDICATOR_PRODUCERS = {"Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual",
                       "And", "Or", "Not", "Xor", "Sign", "ReduceMax", "MaxPool",
                       "GlobalMaxPool", "Where", "Clip", "Min", "Max"}

# floored monsters — re-audited 2026-07-10 (233-lens 8/8 negative); keep excluded but ANNOTATE
FLOORED = {233, 366, 18, 285, 286, 133, 158, 54, 349, 173, 145, 191, 66, 216}


def plane_bytes(vi):
    tt = vi.type.tensor_type
    sz = ITEMSIZE.get(tt.elem_type, 4)
    n = 1
    for d in tt.shape.dim:
        n *= (d.dim_value if d.dim_value > 0 else 1)
    return n * sz, [d.dim_value for d in tt.shape.dim], tt.elem_type


def reaches_output_monotone(name, consumers_of, producer_of, depth=0):
    """Every downstream path from tensor `name` reaches graph output 'output'
    passing only through MONOTONE ops."""
    if depth > 12:
        return False
    cons = consumers_of.get(name, [])
    if not cons:
        return name == "output"
    for node in cons:
        if node.op_type not in MONOTONE:
            return False
        for o in node.output:
            if o == "output":
                continue
            if not reaches_output_monotone(o, consumers_of, producer_of, depth + 1):
                return False
    return True


man = load_manifest()
rows = []
for t in range(1, 401):
    row = man.get(f"{t:03d}")
    if not row or row.get("cost") is None or row["cost"] < 2000:
        continue
    cost = row["cost"]
    p = NETS / f"task{t:03d}.onnx"
    if not p.exists():
        continue
    try:
        m = shape_inference.infer_shapes(onnx.load(str(p)), strict_mode=False)
    except Exception:
        continue
    g = m.graph
    vis = {vi.name: vi for vi in list(g.value_info)}
    consumers_of = {}
    for n in g.node:
        for i in n.input:
            consumers_of.setdefault(i, []).append(n)
    producer_of = {o: n for n in g.node for o in n.output}

    for name, vi in vis.items():
        b, sh, et = plane_bytes(vi)
        if b < 600:
            continue
        prod = producer_of.get(name)
        if prod is None:
            continue
        cons_ops = [n.op_type for n in consumers_of.get(name, [])]
        if not cons_ops:
            continue
        dpts = math.log(cost / max(1, cost - b))

        kind = None
        # class 1: linear fold (old lane, widened reducers + lower min bytes)
        if prod.op_type in LINEAR_PRODUCERS and all(c in SUM_REDUCERS for c in cons_ops):
            kind = "linear_fold"
        # class 2: indicator fold — nonlinear-by-old-rules but max==sum>0 on indicators,
        # and the >0 output threshold is free if all downstream paths are monotone
        elif (prod.op_type in INDICATOR_PRODUCERS or
              any(c in {"ReduceMax", "MaxPool", "Max", "Or", "Where"} for c in cons_ops)):
            if reaches_output_monotone(name, consumers_of, producer_of):
                kind = "indicator_fold"
        if kind is None:
            continue
        rows.append({"task": t, "floored": t in FLOORED, "cost": cost, "bytes": b,
                     "dpts": round(dpts, 4), "shape": sh, "dtype": int(et),
                     "producer": prod.op_type, "consumers": cons_ops, "tensor": name,
                     "kind": kind})

rows.sort(key=lambda r: -r["dpts"])
OUT.write_text(json.dumps(rows, indent=1))
print(f"{len(rows)} candidate planes; top 40 by dpts:")
print(f"{'task':>5} {'flr':>3} {'kind':>15} {'dpts':>7} {'bytes':>6} {'cost':>6}  producer->consumers  shape")
for r in rows[:40]:
    print(f"{r['task']:>5} {'F' if r['floored'] else '':>3} {r['kind']:>15} {r['dpts']:>7} "
          f"{r['bytes']:>6} {r['cost']:>6}  {r['producer']}->{','.join(r['consumers'][:3])}  {r['shape']} {r['tensor'][:24]}")
