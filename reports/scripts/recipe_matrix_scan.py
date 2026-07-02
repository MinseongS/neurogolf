"""Unified recipe-applicability matrix: today's proven recipes × all 400 nets.

For each deployed net, detect the mechanical signatures of every S8-proven
recipe and estimate convertible bytes. Output = ranked per-task opportunity
matrix (reports/recipe_matrix.json) driving the exhaustive application sweep.

Signatures (recipe → playbook toolbox #):
- REDUCE_ONLY   (7/8): intermediate plane >=400B whose consumers are ALL in
                 {ReduceMax,ReduceMin,ReduceSum,ArgMax,ArgMin} → free-input
                 einsum profiles / moments. est = bytes - 240.
- TOPK_LOCATE   (8): plane whose consumer chain is only Reshape/Flatten/Cast
                 → TopK → moments closed form. est = bytes - 300.
- CODE_PLANE    (209-kit): fp32 [1,1,H,W] produced by 1x1 Conv, consumed only
                 by Equal/Reduce/Cast/MaxPool → u8 QLinearConv recast. est = 0.75*bytes.
- REPEAT_GROUP  (1/9/10): >=4 same-(shape,dtype,op) intermediates → walk einsum
                 / batched-K / conv-channel union. est = 0.7*group bytes.
- LABEL_EPILOGUE(13a): u8/int plane feeding Equal whose output is graph output,
                 PLUS >=1 other full plane (label/Where source) → epilogue fold.
                 est = 0.5*(those planes) [only when separate planes exist — task044 lesson].
- FP16_RECAST   (fp16 memory): fp32 intermediate >=800B whose consumers are all
                 elementwise/compare and values provably small ints NOT checkable
                 statically → flag only (est 0.4*bytes, confidence low).

Excludes tasks rebuilt in S8 and priced floors.

  PYTHONPATH=. .venv/bin/python reports/scripts/recipe_matrix_scan.py
"""

import json
from collections import defaultdict
from pathlib import Path

import onnx
from onnx import shape_inference

ROOT = Path(__file__).resolve().parents[2]
manifest = json.load(open(ROOT / "reports" / "manifest.json"))["tasks"]

DONE = {2,17,18,23,25,44,54,66,76,77,101,110,118,133,145,158,163,173,187,191,
        202,204,208,209,219,233,234,243,255,280,286,313,349,351,364,366,367}
DTYPE_BYTES = {1:4,2:1,3:1,4:2,5:2,6:4,7:8,9:1,10:2,11:8,12:4,13:8}
REDUCERS = {"ReduceMax","ReduceMin","ReduceSum","ArgMax","ArgMin"}
PASSTHRU = {"Reshape","Flatten","Cast","Squeeze","Unsqueeze"}

rows = []
for num in range(1, 401):
    if num in DONE:
        continue
    p = ROOT / "networks" / f"task{num:03d}.onnx"
    if not p.exists():
        continue
    model = onnx.load(str(p))
    try:
        g = shape_inference.infer_shapes(model).graph
    except Exception:
        g = model.graph
    inits = {i.name for i in g.initializer}
    outs_of_graph = {vi.name for vi in g.output}
    shapes = {}
    for vi in list(g.value_info) + list(g.output):
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        if dims and all(d > 0 for d in dims):
            shapes[vi.name] = (tuple(dims), vi.type.tensor_type.elem_type)
    def nbytes(name):
        if name not in shapes: return 0
        dims, et = shapes[name]
        b = DTYPE_BYTES.get(et, 4)
        for d in dims: b *= d
        return b
    consumers = defaultdict(list)
    producer = {}
    for node in g.node:
        for i in node.input:
            consumers[i].append(node)
        for o in node.output:
            producer[o] = node

    found = defaultdict(int)
    detail = defaultdict(list)

    for name, (dims, et) in shapes.items():
        if name in inits or name in outs_of_graph or name not in producer:
            continue
        b = nbytes(name)
        if b < 400:
            continue
        cons = consumers.get(name, [])
        if not cons:
            continue
        ops = {c.op_type for c in cons}
        if ops <= REDUCERS:
            found["REDUCE_ONLY"] += max(0, b - 240)
            detail["REDUCE_ONLY"].append((name, b))
            continue
        # TopK locate: passthru chain ending in TopK
        def leads_to_topk(nm, depth=0):
            if depth > 4: return False
            cc = consumers.get(nm, [])
            if not cc: return False
            for c in cc:
                if c.op_type == "TopK": continue
                elif c.op_type in PASSTHRU and leads_to_topk(c.output[0], depth+1): continue
                else: return False
            return True
        if leads_to_topk(name):
            found["TOPK_LOCATE"] += max(0, b - 300)
            detail["TOPK_LOCATE"].append((name, b))
            continue
        pr = producer[name]
        if et == 1 and len(dims) == 4 and dims[1] == 1 and pr.op_type == "Conv":
            k = [a for a in pr.attribute if a.name == "kernel_shape"]
            if k and list(k[0].ints) == [1, 1] and ops <= {"Equal","Cast","MaxPool"} | REDUCERS:
                found["CODE_PLANE"] += int(0.75 * b)
                detail["CODE_PLANE"].append((name, b))

    # repeat groups
    groups = defaultdict(int)
    for name, (dims, et) in shapes.items():
        if name in inits or name in outs_of_graph or name not in producer: continue
        b = nbytes(name)
        if b < 64: continue
        groups[(dims, et, producer[name].op_type)] += 1
    rep = 0
    for (dims, et, op), n in groups.items():
        if n >= 4:
            b = DTYPE_BYTES.get(et,4)
            for d in dims: b *= d
            rep += b * n
    if rep >= 1500:
        found["REPEAT_GROUP"] += int(0.7 * rep)

    total_est = sum(found.values())
    if total_est < 500:
        continue
    e = manifest[str(num)]
    tot = e["memory"] + e["params"]
    import math
    dpts = math.log(tot / max(tot - total_est, 100)) if tot > total_est else math.log(tot/100)
    rows.append({
        "task": num, "total": tot, "points": round(e["points"],3),
        "est_convertible": total_est,
        "est_dpts": round(min(dpts, 3.0), 3),
        "recipes": {k: v for k, v in found.items()},
        "top_planes": {k: v[:3] for k, v in detail.items()},
    })

rows.sort(key=lambda r: -r["est_dpts"])
json.dump(rows, open(ROOT / "reports" / "recipe_matrix.json", "w"), indent=1)
print(f"{len(rows)} tasks with >=500B estimated convertible bytes")
print(f"{'task':>5} {'total':>7} {'pts':>7} {'conv':>6} {'est+':>6}  recipes")
for r in rows[:40]:
    print(f"{r['task']:>5} {r['total']:>7} {r['points']:>7.3f} {r['est_convertible']:>6} {r['est_dpts']:>6.3f}  {','.join(r['recipes'])}")
print("sum est_dpts (all):", round(sum(r["est_dpts"] for r in rows), 1))
