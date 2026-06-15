"""Sweep the single-Conv fitter across many tasks; bank the mem-0 hits.

Usage: PYTHONPATH=. .venv/bin/python reports/conv_sweep.py [task,task,...|auto]
  auto  -> every task whose current real points < THRESH (most headroom first)

Fast triage per task (small train/verify, k=1 then 3). Any task that reaches
stored-ok + FRESH clean is re-verified at full strength and, if still clean and
it BEATS the current adopted points, the custom file is written (NOT adopted —
main runs src.adopt). Results logged to reports/conv_hits.json incrementally."""
import json, os, sys
import importlib.util

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("ORT_NUM_THREADS", "1")
sys.path.insert(0, "/tmp/arc-gen")
# load the solver module directly (reports/ is not a package)
spec = importlib.util.spec_from_file_location("conv_fit", "reports/conv_fit.py")
conv_fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv_fit)

THRESH = 18.0     # only bother with tasks below this (single-Conv ceiling ~18-20)

m = json.load(open("reports/manifest.json"))["tasks"]

def cur_pts(N):
    return m.get(str(N), {}).get("points", 0.0)

arg = sys.argv[1] if len(sys.argv) > 1 else "auto"
if arg == "auto":
    cand = sorted([int(k) for k, v in m.items() if v.get("points", 99) < THRESH],
                  key=cur_pts)
else:
    cand = [int(x) for x in arg.split(",")]

HITS_PATH = os.environ.get("CONV_HITS", "reports/conv_hits.json")
print(f"sweeping {len(cand)} tasks -> {HITS_PATH}", flush=True)
hits = {}
try:
    hits = json.load(open(HITS_PATH))
except Exception:
    pass

for ci, N in enumerate(cand):
    print(f"[{ci+1}/{len(cand)}] task{N:03d}...", flush=True)
    try:
        # fast triage: k=1 then k=3, small budgets
        res = conv_fit.solve(N, ks=(1, 3), n_train=200, n_verify=60,
                             rounds=4, write=False, verbose=False)
    except Exception as e:
        print(f"task{N:03d}: error {e}")
        continue
    if not res:
        continue
    # confirm at full strength
    full = conv_fit.solve(N, ks=(res["k"],), n_train=400, n_verify=200,
                          rounds=8, write=False, verbose=False)
    if not full:
        print(f"task{N:03d}: triage hit but full-verify failed (k={res['k']})")
        continue
    beat = full["points"] - cur_pts(N)
    tag = "WRITE" if (full["points"] > cur_pts(N) + 1e-9) else "no-gain"
    print(f"task{N:03d}: SINGLE-CONV k={full['k']} {full['points']:.2f} pts "
          f"(cur {cur_pts(N):.2f}, +{beat:.2f}) [{tag}]")
    hits[str(N)] = {"k": full["k"], "points": round(full["points"], 2),
                    "cur": round(cur_pts(N), 2), "gain": round(beat, 2)}
    json.dump(hits, open(HITS_PATH, "w"), indent=1)
    if full["points"] > cur_pts(N) + 1e-9:
        conv_fit.emit_custom(full["W"], full["B"], full["k"], N)

print(f"\ndone. {len(hits)} total hits logged in reports/conv_hits.json")
