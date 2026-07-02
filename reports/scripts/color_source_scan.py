"""Classify all 400 tasks by OUTPUT-COLOUR SOURCE — the epilogue-cost knob.

Classes (cheapest→most expensive under the grader counting model):
- FIXED_DELTA: cells that change always take colours from one FIXED palette
  across every bundled example → epilogue = Where(mask, const-onehot, input),
  nearly free; prime walk-einsum targets (task002/077 class).
- SMALL_K: delta colours vary per example but ≤K distinct per example and
  present in that example's input → gates-as-einsum-operands selection works
  (task110/286 class).
- COPY: delta colours are many/per-cell data-dependent → uint8 label epilogue
  (~8-12KB) floor until the uniform-arity folding problem is solved (task187
  class).
- RESHAPE: output grid shape differs from input (crop/scale/tile tasks).

  PYTHONPATH=. .venv/bin/python reports/scripts/color_source_scan.py
Writes reports/color_source_scan.json and prints a queue-oriented summary.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
manifest = json.load(open(ROOT / "reports" / "manifest.json"))["tasks"]

SMALL_K = 2

rows = []
for num in range(1, 401):
    task = json.load(open(ROOT / "data" / f"task{num:03d}.json"))
    examples = task["train"] + task["test"] if isinstance(task, dict) and "train" in task else task
    delta_sets = []
    reshape = False
    subset_input = True
    for ex in examples:
        gi, go = ex["input"], ex["output"]
        if len(gi) != len(go) or len(gi[0]) != len(go[0]):
            reshape = True
            break
        dcol = set()
        icol = {c for row in gi for c in row}
        for r in range(len(gi)):
            for c in range(len(gi[0])):
                if gi[r][c] != go[r][c]:
                    dcol.add(go[r][c])
        delta_sets.append(frozenset(dcol))
        if not dcol <= icol:
            subset_input = False
    if reshape:
        cls = "RESHAPE"
    else:
        union = set().union(*delta_sets) if delta_sets else set()
        always_same = len(set(delta_sets)) == 1
        maxk = max((len(d) for d in delta_sets), default=0)
        if always_same or len(union) <= 2:
            cls = "FIXED_DELTA"
        elif maxk <= SMALL_K:
            cls = "SMALL_K"
        else:
            cls = "COPY"
    e = manifest[str(num)]
    rows.append({"task": num, "class": cls, "total": e["memory"] + e["params"],
                 "points": round(e["points"], 3)})

json.dump(rows, open(ROOT / "reports" / "color_source_scan.json", "w"), indent=0)

from collections import Counter, defaultdict
cnt = Counter(r["class"] for r in rows)
print("class counts:", dict(cnt))
by = defaultdict(list)
for r in rows:
    by[r["class"]].append(r)
for cls in ["FIXED_DELTA", "SMALL_K", "COPY", "RESHAPE"]:
    top = sorted(by[cls], key=lambda r: -r["total"])[:12]
    print(f"\n== {cls} — biggest nets (epilogue-cheap first is the queue) ==")
    for r in top:
        print(f"  task{r['task']:03d} total={r['total']:6d} pts={r['points']:.2f}")
