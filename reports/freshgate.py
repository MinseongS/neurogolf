"""Fresh-3000 gate specific candidate nets before adopting (leak/overfit guard)."""
import sys, numpy as np, onnxruntime as ort
from src.harness import load_task, evaluate, convert_to_numpy
from src.genverify import load_gen

CANDS = [
    (193, "/tmp/h_chcsh/task193.onnx"),
    (157, "/tmp/h_chcsh/task157.onnx"),
    (286, "/tmp/h_koji7116/overrides/task286.onnx"),
]

def gate(path, task, N=3000):
    ev = evaluate(path, load_task(task))
    gen = load_gen(task)
    if gen is None:
        return "NOGEN", ev
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(path, so)
    run = tries = bad = 0
    while run < N and tries < N * 6:
        tries += 1
        try: ex = gen.generate()
        except Exception: continue
        if max(len(ex["input"]), len(ex["input"][0]), len(ex["output"]), len(ex["output"][0])) > 30: continue
        bm = convert_to_numpy(ex)
        try: out = sess.run(None, {"input": bm["input"].astype(np.float32)})[0]
        except Exception: bad += 1; run += 1; continue
        if not ((out[0] > 0).astype(np.int8) == bm["output"][0].astype(np.int8)).all(): bad += 1
        run += 1
    return f"n={run} bad={bad} {'PASS' if bad == 0 else 'FAIL'}", ev

for task, src in CANDS:
    r, ev = gate(src, task)
    print(f"task{task:03d} [{src.split('/')[2]}]: pts={ev['points']:.3f} ok={ev['ok']} fail={ev['fail']} -> FRESH {r}", flush=True)
print("DONE", flush=True)
