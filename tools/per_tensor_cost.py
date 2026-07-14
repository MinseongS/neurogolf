"""Per-tensor counted-cost inspector.

Runs a deployed net through the SAME grader trace path as neurogolf.scoring.calculate_memory,
but instead of summing, dumps each named node-output tensor's counted bytes (max of static-declared
and runtime-traced) sorted descending, annotated with producer op + consumer ops. This localizes
exactly which tensor(s) dominate a wall net's cost so we know where a fold/surrogate must bite.

Usage: uv run python tools/per_tensor_cost.py 349 173 158 138
"""
import json, math, sys, tempfile, pathlib, collections
import numpy as np, onnx, onnxruntime
from neurogolf.scoring import sanitize_model, load_task, convert_to_numpy
from neurogolf.paths import ROOT

NETS = ROOT / "networks"


def per_tensor(task):
    path = NETS / f"task{int(task):03d}.onnx"
    model = onnx.load(str(path))
    # producer/consumer map on ORIGINAL names (pre-sanitize) for readability
    prod = {}
    cons = collections.defaultdict(list)
    for n in model.graph.node:
        for o in n.output:
            if o:
                prod[o] = n.op_type
        for i in n.input:
            if i:
                cons[i].append(n.op_type)
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        san = sanitize_model(onnx.load(str(path)))
        # map safe_name back: sanitize renames deterministically; rebuild reverse via a fresh pass
        opts = onnxruntime.SessionOptions()
        opts.enable_profiling = True
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        opts.profile_file_prefix = str(tmp / "prof")
        sess = onnxruntime.InferenceSession(san.SerializeToString(), opts)
        ex = load_task(int(task))
        sub = ex.get("train", []) + ex.get("test", [])
        for e in sub[:3]:
            b = convert_to_numpy(e)
            if b:
                sess.run(["output"], {"input": b["input"]})
                break
        trace = sess.end_profiling()
    # static shapes from shape inference
    g = onnx.shape_inference.infer_shapes(model, strict_mode=False).graph
    ITEM = {1:4,10:2,11:8,7:8,6:4,5:2,3:1,2:1,9:1,12:4,13:8,4:2,8:8}  # elem_type->itemsize
    static = {}
    for vi in list(g.value_info) + list(g.output):
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            continue
        n = 1; ok = True
        for d in tt.shape.dim:
            if d.dim_value > 0:
                n *= d.dim_value
            else:
                ok = False
        if ok:
            static[vi.name] = (n * ITEM.get(tt.elem_type, 4), tt.elem_type)
    # runtime from trace: safe_name_k -> original via name_map is lost; instead read trace shapes
    # trace node names are sanitized (safe_name_*). We recover magnitude only, mapped by ORDER is unsafe.
    # Simpler: report STATIC counted cost per tensor (dominant driver) — runtime max only raises it.
    rows = []
    for name, (bytes_, et) in static.items():
        if name in ("input", "output"):
            continue
        rows.append((bytes_, name, et, prod.get(name, "?"), cons.get(name, [])))
    rows.sort(reverse=True)
    total = sum(r[0] for r in rows)
    print(f"\n=== task{int(task):03d}  static-counted total ~{total}  (top tensors) ===")
    for b, name, et, p, c in rows[:12]:
        cc = collections.Counter(c)
        print(f"  {b:8d}B  dtype={et}  <-{p:14s}  ->{dict(cc)}  {name[:40]}")


if __name__ == "__main__":
    for t in sys.argv[1:]:
        try:
            per_tensor(t)
        except Exception as e:
            print(f"task{t}: ERR {e}")
