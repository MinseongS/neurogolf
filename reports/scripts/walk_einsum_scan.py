"""Global walk-einsum propagation scanner.

Ranks all 400 deployed nets by ITERATION-SUSPECT bytes: counted intermediate
tensors that occur in repeated same-shape groups (>=REPEAT_MIN members with the
same (shape, dtype) produced by the same op cycle) — the signature of unrolled
propagation/scan loops that the walk-einsum mechanism collapses
(insight_registry: walk_einsum_iteration_collapse; proven on 187/110/243/077/
002/018/364).

Outputs reports/walk_einsum_scan.json + a ranked table to stdout.

  PYTHONPATH=. .venv/bin/python reports/scripts/walk_einsum_scan.py
"""

import json
from collections import defaultdict
from pathlib import Path

import onnx
from onnx import shape_inference

REPEAT_MIN = 4
DTYPE_BYTES = {1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8, 12: 4, 13: 8}

ROOT = Path(__file__).resolve().parents[2]
manifest = json.load(open(ROOT / "reports" / "manifest.json"))["tasks"]

rows = []
for num in range(1, 401):
    p = ROOT / "networks" / f"task{num:03d}.onnx"
    if not p.exists():
        continue
    model = onnx.load(str(p))
    try:
        inferred = shape_inference.infer_shapes(model)
    except Exception:
        inferred = model
    g = inferred.graph
    inits = {i.name for i in g.initializer}
    graph_io = {vi.name for vi in list(g.input) + list(g.output)}
    shapes = {}
    for vi in list(g.value_info) + list(g.output):
        t = vi.type.tensor_type
        dims = [d.dim_value for d in t.shape.dim]
        if any(d <= 0 for d in dims):
            continue
        shapes[vi.name] = (tuple(dims), t.elem_type)
    producer_op = {}
    for node in g.node:
        for o in node.output:
            producer_op[o] = node.op_type

    groups = defaultdict(list)
    for name, (dims, et) in shapes.items():
        if name in inits or name in graph_io:
            continue
        nbytes = DTYPE_BYTES.get(et, 4)
        for d in dims:
            nbytes *= d
        if nbytes < 64:  # scalars/tiny vectors are not iteration planes
            continue
        groups[(dims, et, producer_op.get(name, "?"))].append((name, nbytes))

    repeat_bytes = 0
    top_groups = []
    for (dims, et, op), members in groups.items():
        if len(members) >= REPEAT_MIN:
            b = sum(nb for _, nb in members)
            repeat_bytes += b
            top_groups.append({"op": op, "shape": list(dims), "n": len(members), "bytes": b})
    top_groups.sort(key=lambda x: -x["bytes"])
    e = manifest[str(num)]
    total = e["memory"] + e["params"]
    rows.append({
        "task": num,
        "total": total,
        "points": round(e["points"], 3),
        "repeat_bytes": repeat_bytes,
        "repeat_frac": round(repeat_bytes / max(total, 1), 3),
        "groups": top_groups[:4],
    })

rows.sort(key=lambda r: -r["repeat_bytes"])
json.dump(rows, open(ROOT / "reports" / "walk_einsum_scan.json", "w"), indent=1)
print(f"{'task':>5} {'total':>7} {'pts':>7} {'repeat':>7} {'frac':>5}  top-group")
for r in rows[:45]:
    tg = r["groups"][0] if r["groups"] else {}
    print(f"{r['task']:>5} {r['total']:>7} {r['points']:>7.3f} {r['repeat_bytes']:>7} {r['repeat_frac']:>5.2f}  "
          f"{tg.get('op','-')}x{tg.get('n','-')} {tg.get('shape','')} {tg.get('bytes',0)}B")
