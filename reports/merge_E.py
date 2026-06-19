"""Build the best submission: BASE = kojimar's audited 7113.80 blend (/tmp/koji_final,
known LB-valid). OVERLAY our current net ONLY where ours scores strictly higher on our
examples AND passes a fresh-200 generalization check (fresh-pass => generalizes => LB-safe).
Where kojimar's net fails our examples, fall back to ours. Guarantees ~>= 7113.80 + our
genuine fresh-verified wins. Writes networks/ + manifest in place (git-revertible)."""
import json, pathlib, shutil
import numpy as np
import onnxruntime as ort
from src.harness import load_task, evaluate, convert_to_numpy
from src.genverify import load_gen
from src.pipeline import MANIFEST, NETWORKS, load_manifest, write_scoreboard

KOJI = pathlib.Path("/tmp/koji_final")
OUR_BACKUP = pathlib.Path("/tmp/our7107")   # snapshot of our current (7107) networks
NF = 200

def fresh_ok(path, num, n=NF):
    gen = load_gen(num)
    if gen is None:
        return None
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(path, so)
    run = tries = 0
    while run < n and tries < n * 6:
        tries += 1
        try: ex = gen.generate()
        except Exception: continue
        if max(len(ex["input"]), len(ex["input"][0]), len(ex["output"]), len(ex["output"][0])) > 30:
            continue
        bm = convert_to_numpy(ex)
        try: out = sess.run(None, {"input": bm["input"].astype(np.float32)})[0]
        except Exception: return False
        if not ((out[0] > 0).astype(np.int8) == bm["output"][0].astype(np.int8)).all():
            return False
        run += 1
    return run > 0

our_man = load_manifest()   # our current 7107 manifest (pts per task)
new_man = {}
use_koji = use_ours = koji_broken = 0
for n in range(1, 401):
    kp = KOJI / f"task{n:03d}.onnx"
    op = OUR_BACKUP / f"task{n:03d}.onnx"
    o_pts = our_man.get(n, {}).get("points", 0.0)
    o_meta = our_man.get(n, {})
    try:
        ev = evaluate(str(kp), load_task(n))
    except Exception:
        ev = {"ok": False}
    if not ev["ok"]:
        # kojimar's net broken on our examples -> keep ours
        shutil.copy(op, NETWORKS / f"task{n:03d}.onnx")
        new_man[n] = o_meta; koji_broken += 1
        print(f"  task{n:03d}: koji FAILS examples -> KEEP ours ({o_pts:.2f})", flush=True)
        continue
    k_pts = ev["points"]
    # overlay ours only if strictly better AND fresh-verified
    if o_pts > k_pts + 1e-9 and fresh_ok(str(op), n) is True:
        shutil.copy(op, NETWORKS / f"task{n:03d}.onnx")
        new_man[n] = o_meta; use_ours += 1
        print(f"  task{n:03d}: OURS {o_pts:.2f} > koji {k_pts:.2f} (fresh OK) -> overlay ours", flush=True)
    else:
        shutil.copy(kp, NETWORKS / f"task{n:03d}.onnx")
        new_man[n] = {"points": k_pts, "memory": ev["memory"], "params": ev["params"], "method": "ext:kojimar7113"}
        use_koji += 1

with open(MANIFEST, "w") as f:
    json.dump({"tasks": {str(k): v for k, v in sorted(new_man.items())}}, f, indent=1)
total, solved = write_scoreboard(new_man)
print(f"\n=== E: {use_koji} kojimar + {use_ours} ours-overlay + {koji_broken} ours-fallback(koji broken) ===")
print(f"TOTAL local: {total:.2f} pts, {solved}/400 solved")
json.dump({"use_koji": use_koji, "use_ours": use_ours, "koji_broken": koji_broken, "total": total},
          open("reports/merge_E_result.json", "w"), indent=1)
