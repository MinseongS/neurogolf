"""Multi-source keep-best harvest SCAN (read-only; no adopt).
Scores 3 fresh public sources per task vs our current manifest, finds where any
source beats ours on OUR task definition (stored examples pass). Output = ranked
candidate wins per source + combined best. Adoption (fresh-200 gated) is a
separate step once we see the headroom.
"""
import json, pathlib
from src.harness import load_task, evaluate

SOURCES = {
    "koji7116": pathlib.Path("/tmp/h_koji7116/overrides"),
    "zeropad":  pathlib.Path("/tmp/h_zeropad/submission"),
    "chcsh":    pathlib.Path("/tmp/h_chcsh"),
    "seddik":   pathlib.Path("/tmp/h_seddik/submission"),
    "frank":    pathlib.Path("/tmp/h_frank/selected_submission"),
}
man = json.load(open("reports/manifest.json"))["tasks"]

out = {}
src_gain = {s: 0.0 for s in SOURCES}
combined_gain = 0.0
combined_wins = 0
for n in range(1, 401):
    ours = man.get(str(n), {}).get("points", 0.0)
    best_src, best_pts = None, ours
    rec = {"task": n, "ours": round(ours, 3)}
    for s, d in SOURCES.items():
        p = d / f"task{n:03d}.onnx"
        if not p.exists():
            continue
        try:
            r = evaluate(str(p), load_task(n))
        except Exception as e:
            rec[s] = f"ERR:{str(e)[:40]}"
            continue
        pts = r["points"] if r["ok"] and r["fail"] == 0 else 0.0
        rec[s] = round(pts, 3)
        if r["ok"] and r["fail"] == 0:
            if pts - ours >= 0.005:
                src_gain[s] += pts - ours
            if pts > best_pts + 1e-9:
                best_pts, best_src = pts, s
    if best_src:
        rec["best_src"] = best_src
        rec["best_delta"] = round(best_pts - ours, 3)
        combined_gain += best_pts - ours
        combined_wins += 1
    out[n] = rec
    if best_src:
        print(f"{n:>3} ours={ours:6.2f} -> {best_src} {best_pts:6.2f} (+{best_pts-ours:.3f})", flush=True)

json.dump(out, open("reports/harvest_multi.json", "w"), indent=1)
print("\n=== PER-SOURCE potential stored gain (theirs beats ours on OUR examples, pre-fresh) ===")
for s in SOURCES:
    print(f"  {s:8s}: +{src_gain[s]:.2f}")
print(f"\n=== COMBINED keep-best: {combined_wins} task-wins, +{combined_gain:.2f} stored (pre-fresh-gate) ===")
