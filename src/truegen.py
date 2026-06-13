"""Reliable single-process genverify: REAL (fresh-generalizing) score per task.
Tasks where stored >> real are hidden goldmines (a generalizing custom gives a
big real jump). Uses many instances for low variance."""
import json, sys
sys.path.insert(0, "/tmp/arc-gen")
from src.genverify import fresh_pass

N = 80
m = json.load(open("reports/manifest.json"))["tasks"]
rows = []
for num in range(1, 401):
    v = m.get(str(num))
    if not v:
        rows.append((0.0, 0.0, num, "none")); continue
    ok, run = fresh_pass(num, N)
    real = v["points"] if (run == 0 or ok == run) else 0.0
    rows.append((v["points"] - real, real, num, v.get("method") or "?"))
rows.sort(reverse=True)
tot_real = sum(r[1] for r in rows)
print(f"TRUE total real (n={N}): {tot_real:.1f}  (Kaggle was 6277.51)")
print("TOP goldmines (stored >> real):")
for gap, real, num, meth in rows[:35]:
    if gap < 0.5: break
    print(f"  task{num:03d}: stored {gap+real:.2f} -> REAL {real:.2f} (gap {gap:.2f})  {meth[:22]}")
json.dump([(num, real, gap) for gap, real, num, meth in rows], open("reports/truegen.json","w"))
