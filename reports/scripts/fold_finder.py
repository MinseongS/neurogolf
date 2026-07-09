"""Time-for-cost fold finder for the free-input einsum lever (runtime headroom S16).
Find nets that MATERIALIZE a big counted node-output plane which is then CONTRACTED/REDUCED —
those are candidates to fold into ONE fat einsum against the FREE input (never materializing the
plane). Runtime is free for us (S16: 229ms/pass total), so a fatter/slower fold is affordable.

Ranks big reducible planes by bytes, annotates producer op + consumer reducer + whether the plane's
producer chain reads the FREE `input` (foldable straight off it). Excludes known-floor tasks.
Usage: uv run python -m reports.scripts.fold_finder [min_bytes]
"""
import json, sys
import onnx
from onnx import shape_inference, TensorProto

ITEMSIZE = {TensorProto.FLOAT:4, TensorProto.FLOAT16:2, TensorProto.DOUBLE:8,
            TensorProto.INT64:8, TensorProto.INT32:4, TensorProto.INT16:2, TensorProto.INT8:1,
            TensorProto.UINT8:1, TensorProto.UINT16:2, TensorProto.UINT32:4, TensorProto.UINT64:8,
            TensorProto.BOOL:1}
# Only SUM-contractions make a plane foldable into a free-input einsum (task387). ReduceMax/Min are
# NONLINEAR (presence/OR) and CANNOT be a sum-contraction (fold agent D verdict, task064/134).
REDUCERS = {"Einsum","ReduceSum","ReduceMean","MatMul","Gemm","GlobalAveragePool","ReduceL1"}
# a plane is foldable only if its VALUE is linear in the input: producer must NOT be a nonlinear op.
NONLINEAR_PRODUCERS = {"ReduceMax","ReduceMin","MaxPool","GlobalMaxPool","ArgMax","ArgMin","TopK",
                       "Greater","Less","Equal","And","Or","Where","Clip","Relu","Sign"}
MIN_BYTES = int(sys.argv[1]) if len(sys.argv) > 1 else 1500

man = json.load(open("reports/manifest.json"))["tasks"]
# skip tiny nets and known walk-einsum / floored families (algorithmic floor, cristianoc-validated)
FLOORED = {"233","366","18","285","286","133","158","54","319","349","173","145","110","243","77",
           "187","2","92","234","335","209","76"}

def plane_bytes(vi):
    et = vi.type.tensor_type.elem_type
    sz = ITEMSIZE.get(et, 4)
    n = 1
    for d in vi.type.tensor_type.shape.dim:
        n *= (d.dim_value if d.dim_value > 0 else 1)
    return n * sz, [d.dim_value for d in vi.type.tensor_type.shape.dim], et

rows = []
for t in range(1, 401):
    if str(t) in FLOORED:
        continue
    cost = man[str(t)]["memory"] + man[str(t)]["params"]
    if cost < MIN_BYTES:
        continue
    try:
        m = shape_inference.infer_shapes(onnx.load(f"networks/task{t:03d}.onnx"), strict_mode=False)
    except Exception:
        continue
    g = m.graph
    vis = {vi.name: vi for vi in list(g.value_info) + list(g.output)}
    # consumers map
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n.op_type)
    # producer map
    producer = {o: n for n in g.node for o in n.output}
    for n in g.node:
        for o in n.output:
            if o not in vis:
                continue
            b, sh, et = plane_bytes(vis[o])
            if b < MIN_BYTES:
                continue
            cons = consumers.get(o, [])
            red = [c for c in cons if c in REDUCERS]
            if not red:
                continue
            # does producer read the free input (directly or 1 hop)?
            prod = producer.get(o)
            # skip nonlinear-produced planes (presence/max/compare) — not sum-foldable
            if prod is not None and prod.op_type in NONLINEAR_PRODUCERS:
                continue
            reads_input = prod is not None and ("input" in prod.input or any(
                producer.get(i) is not None and "input" in producer[i].input for i in prod.input))
            rows.append({"task": t, "cost": cost, "plane_bytes": b, "shape": sh,
                         "producer": prod.op_type if prod else "?", "reducers": red,
                         "reads_input": reads_input, "elem": int(et)})

# rank by plane_bytes (bigger materialized plane = bigger potential fold win)
rows.sort(key=lambda r: -r["plane_bytes"])
print(f"{len(rows)} reducible big planes (>= {MIN_BYTES}B), floored-excluded\n")
print("task   cost  plane_B  shape                producer  reducer         reads_input")
seen = set()
for r in rows[:40]:
    key = r["task"]
    tag = "  <-2nd+" if key in seen else ""
    seen.add(key)
    print(f"task{r['task']:03d} {r['cost']:>6} {r['plane_bytes']:>6}  {str(r['shape']):<20} "
          f"{r['producer']:<9} {','.join(r['reducers'])[:14]:<14} {r['reads_input']}{tag}")
json.dump(rows, open("reports/fold_candidates.json","w"), indent=1)
print("\nwrote reports/fold_candidates.json")
