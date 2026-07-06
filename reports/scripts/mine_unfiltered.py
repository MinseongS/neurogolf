"""Relaxed re-mine: grader-measure EVERY extracted source net (no byte prefilter).
The prefilter in mine_public_bundles skips candidate files larger than ours, but grader-mem
!= file size (big cheap initializers, different node structure) -> can miss real wins.
Usage: .venv/bin/python -m reports.scripts.mine_unfiltered <extracted_root>
"""
import glob, json, math, os, sys
import onnx
from src.harness import load_task, evaluate

root = sys.argv[1]
man = json.load(open("reports/manifest.json"))["tasks"]
sources = {os.path.basename(d): d for d in sorted(glob.glob(f"{root}/*")) if os.path.isdir(d)}
print(f"sources: {list(sources)}", file=sys.stderr)

cands = []
for t in range(1, 401):
    task = load_task(t)
    mine = man[str(t)]["memory"] + man[str(t)]["params"]
    best = None
    for s, d in sources.items():
        p = f"{d}/task{t:03d}.onnx"
        if not os.path.exists(p):
            continue
        try:
            r = evaluate(onnx.load(p), task)
        except Exception:
            continue
        if r["fail"] != 0:
            continue
        c = r["memory"] + r["params"]
        if best is None or c < best[0]:
            best = (c, s)
    if best and best[0] < mine:
        cands.append({"task": t, "mine": mine, "cost": best[0], "source": best[1],
                      "dpts": round(math.log(mine) - math.log(best[0]), 3)})
    if t % 50 == 0:
        print(f"  ...{t}/400", file=sys.stderr)
cands.sort(key=lambda x: -x["dpts"])
json.dump(cands, open("reports/unfiltered_candidates.json", "w"), indent=1)
tot = sum(c["dpts"] for c in cands)
print(f"\n{len(cands)} candidates (unfiltered), potential +{tot:.2f} pts")
for c in cands[:50]:
    print(f"  task{c['task']:03d}: {c['mine']:>7} -> {c['cost']:>7}  +{c['dpts']:.3f}  ({c['source'].split('_')[0]})")
