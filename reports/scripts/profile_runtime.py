"""One-time runtime profiler for all 400 nets (time-for-cost lever, [[neurogolf-runtime-timeout-dimension]]).
Measures median wall-clock ms/run per task under grader-style session options (ORT_DISABLE_ALL, no
profiling). Writes reports/runtime_profile.json = {task: {ms_median, ms_max, n_runs, err}}.
Usage: .venv/bin/python -m reports.scripts.profile_runtime [START END]
"""
import sys, os, json, time, pathlib
import numpy as np, onnx, onnxruntime as ort
from src.harness import load_task, sanitize_model, convert_to_numpy

ROOT = pathlib.Path(__file__).resolve().parents[2]
NET = ROOT / "networks"
OUT = ROOT / "reports" / "runtime_profile.json"

start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
end = int(sys.argv[2]) if len(sys.argv) > 2 else 400
N_TIMED = 5  # median over this many runs after 1 warmup

def bench_inputs(task):
    subs = task.get("train", []) + task.get("test", [])
    outs = []
    for ex in subs[:3]:
        b = convert_to_numpy(ex)
        if b:
            outs.append(b["input"])
    if not outs:
        outs = [np.zeros((1, 10, 30, 30), np.float32)]
    return outs

def profile(task_num):
    path = NET / f"task{task_num:03d}.onnx"
    if not path.exists():
        return {"err": "no onnx"}
    try:
        m = sanitize_model(onnx.load(str(path)))
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(m.SerializeToString(), opts)
        task = load_task(task_num)
        inps = bench_inputs(task)
        sess.run(["output"], {"input": inps[0]})  # warmup
        times = []
        for _ in range(N_TIMED):
            t0 = time.perf_counter()
            for x in inps:
                sess.run(["output"], {"input": x})
            times.append((time.perf_counter() - t0) / len(inps) * 1000.0)
        times.sort()
        return {"ms_median": round(times[len(times)//2], 3),
                "ms_max": round(max(times), 3), "n_inputs": len(inps)}
    except Exception as e:
        return {"err": str(e)[:120]}

def main():
    res = {}
    if OUT.exists() and (start, end) != (1, 400):
        res = json.loads(OUT.read_text())
    for t in range(start, end + 1):
        r = profile(t)
        res[str(t)] = r
        tag = r.get("ms_median", r.get("err"))
        print(f"task{t:03d}  {tag}", flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    ok = {k: v for k, v in res.items() if "ms_median" in v}
    tot = sum(v["ms_median"] for v in ok.values())
    print(f"\n=== {len(ok)}/{len(res)} profiled; sum median ms = {tot:.0f} ===")
    top = sorted(ok.items(), key=lambda kv: kv[1]["ms_median"], reverse=True)[:15]
    for k, v in top:
        print(f"  task{int(k):03d}  {v['ms_median']:.1f} ms")

if __name__ == "__main__":
    main()
