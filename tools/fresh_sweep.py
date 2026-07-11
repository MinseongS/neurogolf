#!/usr/bin/env python
"""Board-wide fresh sweep: run every deployed net on N fresh arc-gen instances,
report per-task fail counts. Private-LB risk triage (2026-07-11 silent-zero incident).

Usage: uv run python tools/fresh_sweep.py [--n 1500] [--tasks 1 2 3] [--jobs 8]
Output: state/fresh_sweep_<date>.json  {task: {"fails": F, "runs": R}}
Each task runs in its own subprocess (knife-edge isolation)."""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

WORKER = r'''
import sys, json, importlib
sys.path.insert(0, "src"); sys.path.insert(0, "arc-gen")
import numpy as np
import onnxruntime as ort
from neurogolf.scans.fresh import to_onehot
t = int(sys.argv[1]); n = int(sys.argv[2])
mapping = json.load(open("state/arc_mapping.json"))
gen = importlib.import_module(f"tasks.task_{mapping[str(t)]['arc_id']}")
so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(open(f"submission/overfit_nets/task{t:03d}.onnx", "rb").read(), so,
                            providers=["CPUExecutionProvider"])
fails = runs = 0
for _ in range(n):
    try:
        ex = gen.generate()
    except Exception:
        continue
    gi = np.array(ex["input"]); goa = np.array(ex["output"])
    # graded universe is <=30x30 (net input [1,10,30,30]); bundled pipeline filters sizes
    if max(gi.shape) > 30 or max(goa.shape) > 30:
        continue
    runs += 1
    x = to_onehot(ex["input"])
    go = np.array(ex["output"]); h, w = go.shape
    exp = np.zeros((10, 30, 30), bool)
    for c in range(10):
        exp[c, :h, :w] = (go == c)
    try:
        out = (sess.run(["output"], {"input": x})[0] > 0.0)[0]
    except Exception:
        fails += 1
        continue
    if not np.array_equal(out, exp):
        fails += 1
print(json.dumps({"task": t, "fails": fails, "runs": runs}))
'''


def run_task(t: int, n: int, timeout: int) -> dict:
    try:
        r = subprocess.run([sys.executable, "-c", WORKER, str(t), str(n)],
                           capture_output=True, text=True, cwd=ROOT, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"task": t, "error": f"timeout>{timeout}s"}
    except Exception as e:  # never kill the pool
        return {"task": t, "error": repr(e)[-300:]}
    if r.returncode != 0 or not r.stdout.strip():
        return {"task": t, "error": (r.stderr or "no output")[-300:]}
    return json.loads(r.stdout.strip().splitlines()[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--tasks", type=int, nargs="*", default=None)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--resume", action="store_true",
                    help="skip tasks already present (without error) in today's sweep file")
    args = ap.parse_args()
    path = ROOT / "state" / f"fresh_sweep_{date.today().isoformat()}.json"
    out = {}
    if args.resume and path.exists():
        out = {k: v for k, v in json.loads(path.read_text()).items() if not v.get("error")}
        print(f"resume: {len(out)} tasks already swept")
    tasks = [t for t in (args.tasks or range(1, 401)) if str(t) not in out]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_task, t, args.n, args.timeout): t for t in tasks}
        for f in as_completed(futs):
            r = f.result()
            out[str(r["task"])] = r
            path.write_text(json.dumps(out, indent=1, sort_keys=True))  # incremental
            if r.get("fails") or r.get("error"):
                print(f"task{r['task']:03d}: {r}", flush=True)
    bad = [r for r in out.values() if r.get("fails")]
    err = [r for r in out.values() if r.get("error")]
    print(f"\nswept {len(out)} tasks: {len(bad)} with fresh fails, {len(err)} errors -> {path}")


if __name__ == "__main__":
    main()
