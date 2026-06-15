"""Floor-break verifier: .venv/bin/python reports/verify_fb.py N
Builds src/custom/taskNNN.py, prints STORED ok/pts/memory/params and FRESH 200/200."""
import importlib, sys, numpy as np, onnx, onnxruntime as ort
sys.path.insert(0, "/tmp/arc-gen")
from src.harness import load_task, evaluate, convert_to_numpy
from src.genverify import load_gen

N = int(sys.argv[1])
task = load_task(N)
mod = importlib.import_module(f"src.custom.task{N:03d}"); importlib.reload(mod)
model = mod.build(task)
ev = evaluate(model, task)
print(f"STORED ok={ev['ok']} pts={ev.get('points')} memory={ev.get('memory')} "
      f"params={ev.get('params')} fail={ev.get('fail')} err={ev.get('error')}")
onnx.save(model, f"/tmp/_fb{N}.onnx")
gen = load_gen(N)
so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(f"/tmp/_fb{N}.onnx", so)
run = fail = tries = 0
while run < 200 and tries < 2000:
    tries += 1
    try: ex = gen.generate()
    except Exception: continue
    if max(len(ex["input"]), len(ex["input"][0]), len(ex["output"]), len(ex["output"][0])) > 30: continue
    bm = convert_to_numpy(ex)
    out = sess.run(None, {"input": bm["input"].astype(np.float32)})[0]
    if not ((out[0] > 0).astype(np.int8) == bm["output"][0].astype(np.int8)).all(): fail += 1
    run += 1
print(f"FRESH {run-fail}/{run} pass ({fail} fail)")
