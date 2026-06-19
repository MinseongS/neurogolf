"""Fresh-gated keep-best merge of kojimar's 7113.80 public blend (/tmp/koji_final)
into our current manifest (already at 7107.01 with sajayr's good nets + our wins).
Adopts a kojimar net ONLY if it passes our stored examples, beats our current
pts, AND passes a fresh-200 generalization check. Keeps our better/genuine nets."""
import json, pathlib, shutil
import numpy as np
import onnxruntime as ort
from src.harness import load_task, evaluate, convert_to_numpy
from src.genverify import load_gen
from src.pipeline import MANIFEST, NETWORKS, load_manifest, write_scoreboard

SRC = pathlib.Path("/tmp/koji_final")
MARGIN = 1e-9
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

manifest = load_manifest()
adopted, rej_fresh, rej_ex, total_gain = [], [], 0, 0.0
rejf = 0
for n in range(1, 401):
    p = str(SRC / f"task{n:03d}.onnx")
    cur = manifest.get(n); cur_pts = cur["points"] if cur else 0.0
    try: ev = evaluate(p, load_task(n))
    except Exception: continue
    if not ev["ok"]:
        continue  # fails our examples -> keep ours
    if ev["points"] <= cur_pts + MARGIN:
        continue  # ours is better -> keep ours
    fk = fresh_ok(p, n)
    if fk is True:
        shutil.copy(p, NETWORKS / f"task{n:03d}.onnx")
        manifest[n] = {"points": ev["points"], "memory": ev["memory"],
                       "params": ev["params"], "method": "ext:kojimar7113"}
        total_gain += ev["points"] - cur_pts
        adopted.append((n, round(cur_pts,2), round(ev["points"],2)))
        print(f"  ADOPT task{n:03d}: {cur_pts:.2f} -> {ev['points']:.2f} (+{ev['points']-cur_pts:.2f})", flush=True)
    else:
        rejf += 1
        print(f"  REJECT-fresh task{n:03d}: theirs {ev['points']:.2f} FRESH={fk} (kept ours {cur_pts:.2f})", flush=True)

with open(MANIFEST, "w") as f:
    json.dump({"tasks": {str(k): v for k, v in sorted(manifest.items())}}, f, indent=1)
total, solved = write_scoreboard(manifest)
print(f"\n=== ADOPTED {len(adopted)} kojimar nets (+{total_gain:.1f} stored), {rejf} fresh-rejected ===")
print(f"TOTAL: {total:.2f} pts, {solved}/400 solved")
json.dump({"adopted": adopted, "rej_fresh": rejf, "gain": total_gain}, open("reports/merge_koji_result.json","w"), indent=1)
