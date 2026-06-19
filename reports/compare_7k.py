"""Score sajayr's public neurogolf-7k ONNX nets against OUR harness + examples,
compare to our current manifest pts. Output candidates where theirs beats ours
AND passes our stored examples (correctness on our task definition)."""
import json, pathlib, sys
from src.harness import load_task, evaluate

THEIRS = pathlib.Path("/tmp/ng7k/extracted")
man = json.load(open("reports/manifest.json"))["tasks"]

out = []
for n in range(1, 401):
    p = THEIRS / f"task{n:03d}.onnx"
    if not p.exists():
        continue
    ours = man.get(str(n), {}).get("points", 0.0)
    try:
        ex = load_task(n)
        r = evaluate(str(p), ex)
    except Exception as e:
        out.append({"task": n, "err": str(e)[:120], "ours": round(ours, 2)})
        print(f"{n:>3} ERR {str(e)[:80]}", flush=True)
        continue
    theirs = r["points"] if r["ok"] else 0.0
    rec = {"task": n, "ours": round(ours, 3), "theirs": round(theirs, 3),
           "delta": round(theirs - ours, 3), "ok": r["ok"],
           "fail": r["fail"], "mem": r["memory"], "par": r["params"],
           "err": r["error"]}
    out.append(rec)
    flag = "WIN" if (r["ok"] and theirs - ours >= 0.3) else ("eq" if r["ok"] else "FAILEX")
    print(f"{n:>3} ours={ours:6.2f} theirs={theirs:6.2f} d={theirs-ours:+6.2f} fail={r['fail']} {flag}", flush=True)

json.dump(out, open("reports/compare_7k.json", "w"), indent=1)
wins = [r for r in out if r.get("ok") and r.get("delta", 0) >= 0.3 and r.get("fail") == 0]
print(f"\n=== {len(wins)} candidate wins (theirs passes our examples & beats by >=0.3) ===")
tot = sum(r["delta"] for r in wins)
print(f"total potential stored gain (pre-fresh-verify): +{tot:.1f}")
for r in sorted(wins, key=lambda r: -r["delta"])[:60]:
    print(f"  {r['task']:>3} ours={r['ours']:6.2f} theirs={r['theirs']:6.2f} +{r['delta']:.2f}")
