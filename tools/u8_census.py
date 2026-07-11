"""Census: fp16/fp32 counted planes >=200B holding small non-negative integers,
with producer+consumers in the u8-safe op set. Per-task worker, run via pool."""
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/Users/minseong/project/neurogolf")

WORKER = r'''
import sys, json
import numpy as np
import onnx
import onnxruntime as ort
from onnx import shape_inference, TensorProto as TP

SAFE = {"Add","Sub","Mul","Min","Max","Equal","Greater","Less","GreaterOrEqual","LessOrEqual",
        "Where","MaxPool","Pad","Concat","Slice","Gather","GatherElements","ReduceMax","ReduceMin",
        "Clip","Mod","Cast","Reshape","Transpose","Expand","ArgMax","ArgMin","Squeeze","Unsqueeze",
        "Flatten","Identity","Tile","ScatterElements","ScatterND","Not","And","Or","Xor","Abs"}
ITEM = {TP.FLOAT: 4, TP.FLOAT16: 2}

t = int(sys.argv[1])
path = f"submission/overfit_nets/task{t:03d}.onnx"
m = onnx.load(path)
opset = next((o.version for o in m.opset_import if o.domain == ""), None)
try:
    mi = shape_inference.infer_shapes(m, strict_mode=False)
except Exception:
    mi = m
dt = {}
shp = {}
for vi in list(mi.graph.value_info) + list(mi.graph.output) + list(mi.graph.input):
    tt = vi.type.tensor_type
    dt[vi.name] = tt.elem_type
    dims = [d.dim_value if d.HasField("dim_value") else -1 for d in tt.shape.dim]
    shp[vi.name] = dims
prod = {}
cons = {}
for n in mi.graph.node:
    for o in n.output:
        if o:
            prod[o] = n
    for i in n.input:
        cons.setdefault(i, []).append(n)
gout = {o.name for o in mi.graph.output}
ginp = {i.name for i in mi.graph.input}
inits = {i.name for i in mi.graph.initializer}

cands = []
for name, et in dt.items():
    if name in gout or name in ginp or name in inits or name not in prod:
        continue
    if et not in ITEM:
        continue
    dims = shp.get(name, [])
    if not dims or any(d < 0 for d in dims):
        continue
    nel = 1
    for d in dims:
        nel *= d
    byt = nel * ITEM[et]
    if byt < 200:
        continue
    p = prod[name]
    cs = cons.get(name, [])
    if p.op_type not in SAFE:
        continue
    if any(c.op_type not in SAFE for c in cs):
        continue
    cands.append((name, et, byt, p.op_type, [c.op_type for c in cs]))

result = {"task": t, "opset": opset, "planes": []}
if cands:
    # value-range check on up to 6 bundled examples
    try:
        data = json.load(open(f"data/task{t:03d}.json"))
        exs = (data.get("train", []) + data.get("test", []))[:3] + \
              data.get("arc-gen", data.get("arc_gen", []))[:3]
        m2 = onnx.load(path)
        want = [c[0] for c in cands]
        have = {o.name for o in m2.graph.output}
        for nm in want:
            if nm not in have:
                et = dt[nm]
                m2.graph.output.append(onnx.helper.make_tensor_value_info(nm, et, None))
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        s = ort.InferenceSession(m2.SerializeToString(), so, providers=["CPUExecutionProvider"])
        onames = [o.name for o in s.get_outputs()]
        mins = {n: np.inf for n in want}
        maxs = {n: -np.inf for n in want}
        isint = {n: True for n in want}
        for ex in exs:
            g = np.array(ex["input"])
            x = np.zeros((1, 10, 30, 30), np.float32)
            h, w = g.shape
            for c in range(10):
                x[0, c, :h, :w] = (g == c)
            outs = s.run(None, {s.get_inputs()[0].name: x})
            for nm, arr in zip(onames, outs):
                if nm in mins:
                    a = arr.astype(np.float64)
                    mins[nm] = min(mins[nm], float(a.min()))
                    maxs[nm] = max(maxs[nm], float(a.max()))
                    if not np.all(a == np.round(a)):
                        isint[nm] = False
        for (name, et, byt, po, co) in cands:
            ok = isint[name] and mins[name] >= 0 and maxs[name] <= 255
            result["planes"].append({"tensor": name, "dtype": int(et), "bytes": byt,
                                     "prod": po, "cons": co, "u8ok": bool(ok),
                                     "vmin": mins[name], "vmax": maxs[name]})
    except Exception as e:
        result["error"] = str(e)[:150]
print(json.dumps(result))
'''


def run(t):
    r = subprocess.run([sys.executable, "-c", WORKER, str(t)],
                       capture_output=True, text=True, cwd=ROOT, timeout=900)
    if r.returncode != 0 or not r.stdout.strip():
        return {"task": t, "error": (r.stderr or "none")[-200:]}
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    skip = {220, 230, 294}
    out = {}
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(run, t): t for t in range(1, 401) if t not in skip}
        for f in as_completed(futs):
            r = f.result()
            out[str(r["task"])] = r
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1, sort_keys=True))
    # summary
    import math
    manifest = json.load(open(ROOT / "state/manifest.json"))
    ent = manifest["tasks"] if "tasks" in manifest else manifest
    costs = {}
    it = ent.items() if isinstance(ent, dict) else [(e.get("task"), e) for e in ent]
    for k, v in it:
        costs[int(k)] = v.get("cost") or (v.get("mem", 0) + v.get("params", 0))
    rows = []
    opsets = {}
    for k, r in out.items():
        opsets[r.get("opset")] = opsets.get(r.get("opset"), 0) + 1
        save = sum(p["bytes"] * (1 - 1 / (4 if p["dtype"] == 1 else 2))
                   for p in r.get("planes", []) if p["u8ok"])
        if save >= 100:
            c = costs.get(int(k), 0)
            gain = math.log(c / max(1, c - save)) if c > save else 0
            rows.append((gain, int(k), c, int(save),
                         len([p for p in r.get("planes", []) if p["u8ok"]])))
    rows.sort(reverse=True)
    print("opset distribution:", dict(sorted(opsets.items(), key=lambda x: (x[0] is None, x[0]))))
    print(f"{'gain':>6} task {'cost':>7} {'save':>6} planes")
    for g, t, c, s, np_ in rows[:20]:
        print(f"{g:6.3f} {t:>4} {c:>7} {s:>6} {np_}")
    print(f"total tasks with u8-recastable mass: {len(rows)}, "
          f"sum gain: {sum(r[0] for r in rows):.2f}")


if __name__ == "__main__":
    main()
