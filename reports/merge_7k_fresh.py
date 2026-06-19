"""Adopt sajayr's public neurogolf-7k nets, FRESH-GENERALIZATION GATED.

Reads reports/compare_7k.json (candidates where theirs passes OUR stored
examples and beats ours by >= MARGIN). For each, re-evaluates on our examples
AND runs a fresh-instance generalization check (n=NF freshly generated arc-gen
instances). Adopts into networks/ + manifest ONLY if BOTH pass — this rejects
the ~4-5 nets sajayr says fail the private (held-out) tests, protecting real LB.
"""
import json, pathlib, shutil
import numpy as np
import onnxruntime as ort
from src.harness import load_task, evaluate, convert_to_numpy
from src.genverify import load_gen
from src.pipeline import MANIFEST, NETWORKS, load_manifest, write_scoreboard

THEIRS = pathlib.Path("/tmp/ng7k/extracted")
MARGIN = 0.3
NF = 200  # fresh instances to verify generalization

def fresh_ok(path, num, n=NF):
    gen = load_gen(num)
    if gen is None:
        return None  # can't verify -> treat as unknown
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(path, so)
    run = tries = 0
    while run < n and tries < n * 6:
        tries += 1
        try:
            ex = gen.generate()
        except Exception:
            continue
        if max(len(ex["input"]), len(ex["input"][0]),
               len(ex["output"]), len(ex["output"][0])) > 30:
            continue
        bm = convert_to_numpy(ex)
        try:
            out = sess.run(None, {"input": bm["input"].astype(np.float32)})[0]
        except Exception:
            return False
        if not ((out[0] > 0).astype(np.int8) == bm["output"][0].astype(np.int8)).all():
            return False
        run += 1
    return run > 0

cmp = json.load(open("reports/compare_7k.json"))
cands = [r for r in cmp if r.get("ok") and r.get("fail") == 0 and r.get("delta", 0) >= MARGIN]
cands.sort(key=lambda r: -r["delta"])
print(f"{len(cands)} stored-example win candidates; fresh-gating each (n={NF})...")

manifest = load_manifest()
adopted, rejected_fresh, total_gain = [], [], 0.0
for r in cands:
    n = r["task"]
    p = str(THEIRS / f"task{n:03d}.onnx")
    cur = manifest.get(n); cur_pts = cur["points"] if cur else 0.0
    ev = evaluate(p, load_task(n))
    if not ev["ok"] or ev["points"] <= cur_pts + 1e-9:
        continue
    fk = fresh_ok(p, n)
    if fk is True:
        shutil.copy(p, NETWORKS / f"task{n:03d}.onnx")
        manifest[n] = {"points": ev["points"], "memory": ev["memory"],
                       "params": ev["params"], "method": "ext:sajayr7k"}
        total_gain += ev["points"] - cur_pts
        adopted.append((n, cur_pts, ev["points"]))
        print(f"  ADOPT task{n:03d}: {cur_pts:.2f} -> {ev['points']:.2f} (+{ev['points']-cur_pts:.2f}) [fresh OK]", flush=True)
    else:
        rejected_fresh.append((n, cur_pts, ev["points"], fk))
        print(f"  REJECT task{n:03d}: theirs {ev['points']:.2f} FRESH={fk} (overfit/non-general — kept ours {cur_pts:.2f})", flush=True)

with open(MANIFEST, "w") as f:
    json.dump({"tasks": {str(k): v for k, v in sorted(manifest.items())}}, f, indent=1)
total, solved = write_scoreboard(manifest)
print(f"\n=== ADOPTED {len(adopted)} fresh-verified nets, +{total_gain:.1f} stored ===")
print(f"REJECTED {len(rejected_fresh)} that failed fresh generalization (protected real LB)")
print(f"TOTAL: {total:.2f} pts, {solved}/400 solved")
json.dump({"adopted": adopted, "rejected_fresh": rejected_fresh, "gain": total_gain},
          open("reports/merge_7k_result.json", "w"), indent=1)
