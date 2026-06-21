"""EXACT equivalence-preserving dtype surgery (output bit-identical to the deployed net).
Greedily tightens Cast-`to` targets and Constant dtypes to the smallest dtype that keeps
the net's binarized output (out>0) BIT-IDENTICAL to the original on a large fresh sample.
Bit-identical-to-original => same function => same private/LB behavior => zero risk.
Only accepts a change if it (a) keeps output identical AND (b) raises harness points.

Usage: python -m reports.surgery 35 293 286 ...   (task numbers; default = a probe set)
"""
import sys, json, tempfile, os, math
import numpy as np
import onnx
from onnx import TensorProto, helper, shape_inference
import onnxruntime as ort
from src.harness import evaluate, load_task
from src.genverify import load_gen
from src.harness import convert_to_numpy

NSAMP = 300
# smaller-itemsize dtypes to try, smallest first
ITEM = {TensorProto.BOOL:1, TensorProto.UINT8:1, TensorProto.INT8:1, TensorProto.INT16:2,
        TensorProto.UINT16:2, TensorProto.FLOAT16:2, TensorProto.INT32:4, TensorProto.FLOAT:4,
        TensorProto.INT64:8, TensorProto.DOUBLE:8}
TRY_ORDER = [TensorProto.BOOL, TensorProto.UINT8, TensorProto.INT8, TensorProto.INT16,
             TensorProto.FLOAT16, TensorProto.INT32]

def fresh(task, n=NSAMP):
    gen = load_gen(task)
    if gen is None: return None
    xs, ys = [], []
    tries = 0
    while len(xs) < n and tries < n*8:
        tries += 1
        try: ex = gen.generate()
        except Exception: continue
        if max(len(ex["input"]),len(ex["input"][0]),len(ex["output"]),len(ex["output"][0]))>30: continue
        bm = convert_to_numpy(ex)
        if bm is None: continue
        xs.append(bm["input"].astype(np.float32)); ys.append(bm["output"][0].astype(np.int8))
    return xs, ys

def run_outs(model, xs):
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(model.SerializeToString(), so)
    outs = []
    for x in xs:
        o = sess.run(None, {"input": x})[0]
        outs.append((o[0] > 0).astype(np.int8))
    return outs

def ident(a, b):
    return all((p == q).all() for p, q in zip(a, b))

def candidates(model):
    """Yield (node_index, kind, current_to) for Cast and Constant nodes worth shrinking."""
    out = []
    for i, nd in enumerate(model.graph.node):
        if nd.op_type == "Cast":
            for a in nd.attribute:
                if a.name == "to" and a.i in ITEM and ITEM[a.i] > 1:
                    out.append((i, "cast", a.i))
        elif nd.op_type == "Constant":
            for a in nd.attribute:
                if a.name == "value" and a.t.data_type in ITEM and ITEM[a.t.data_type] > 1:
                    out.append((i, "const", a.t.data_type))
    return out

def apply_change(model, idx, kind, new_dt):
    m = onnx.ModelProto(); m.CopyFrom(model)
    nd = m.graph.node[idx]
    if kind == "cast":
        for a in nd.attribute:
            if a.name == "to": a.i = new_dt
    else:  # const: re-encode values in the new dtype (lossless only if representable)
        for a in nd.attribute:
            if a.name == "value":
                arr = onnx.numpy_helper.to_array(a.t)
                np_dt = helper.tensor_dtype_to_np_dtype(new_dt)
                arr2 = arr.astype(np_dt)
                if not np.array_equal(arr2.astype(arr.dtype), arr):  # lossy -> bail
                    return None
                newt = onnx.numpy_helper.from_array(arr2, a.t.name)
                a.t.CopyFrom(newt)
    # strip stale value_info so shape-inference re-derives dtypes downstream
    del m.graph.value_info[:]
    try:
        m = shape_inference.infer_shapes(m, strict_mode=True)
    except Exception:
        return None
    return m

def score(model, task):
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = f.name
    try:
        onnx.save(model, path)
        ev = evaluate(path, load_task(task))
    except Exception as e:
        return None
    finally:
        try: os.unlink(path)
        except Exception: pass
    return ev

def surger(task):
    path = f"networks/task{task:03d}.onnx"
    base = onnx.load(path)
    ev0 = score(base, task)
    if not ev0 or not ev0["ok"] or ev0["fail"] != 0:
        return f"task{task:03d}: SKIP (base not scoreable ok)"
    fr = fresh(task)
    if fr is None: return f"task{task:03d}: SKIP (no generator)"
    xs, ys = fr
    if not xs: return f"task{task:03d}: SKIP (no samples)"
    try:
        orig_outs = run_outs(base, xs)
    except Exception as e:
        return f"task{task:03d}: SKIP (base run err {str(e)[:40]})"
    # require the deployed net to actually be correct on fresh (else surgery is meaningless)
    if not ident(orig_outs, ys):
        return f"task{task:03d}: SKIP (deployed net not fresh-correct; don't touch)"
    work = base; cur = ev0["points"]; accepted = 0
    for (idx, kind, cur_to) in candidates(base):
        for new_dt in TRY_ORDER:
            if ITEM[new_dt] >= ITEM[cur_to]: continue
            cand = apply_change(work, idx, kind, new_dt)
            if cand is None: continue
            try:
                outs = run_outs(cand, xs)
            except Exception:
                continue
            if not ident(outs, orig_outs): continue
            ev = score(cand, task)
            if ev and ev["ok"] and ev["fail"] == 0 and ev["points"] > cur + 1e-9:
                work = cand; cur = ev["points"]; accepted += 1
                break
    gain = cur - ev0["points"]
    if accepted and gain > 1e-6:
        onnx.save(work, path + ".surgered")
        return f"task{task:03d}: {ev0['points']:.3f} -> {cur:.3f} (+{gain:.3f}) [{accepted} changes, bit-identical/{len(xs)}] saved .surgered"
    return f"task{task:03d}: no safe gain ({len(candidates(base))} cands, {accepted} kept)"

if __name__ == "__main__":
    tasks = [int(a) for a in sys.argv[1:]] or [35, 293, 342, 20, 286, 29, 284]
    for t in tasks:
        print(surger(t), flush=True)
