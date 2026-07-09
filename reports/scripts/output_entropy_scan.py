"""output_entropy_scan (2026-07-08) — novel lens: OUTPUT simplicity vs NET cost.

Idea under test: "memorization is dead" is a GLOBAL claim. Per-task, if the 268 bundled
instances have few distinct outputs (K small) or trivially small outputs, maybe a cheap
codebook/selector beats the algorithm's counted cost. Overfit is permanently safe (constant
dataset), so a selector need only fit the 268 bundled examples with fail=0.

Two probes:
  (A) codebook lower bound: K distinct outputs x max_output_area vs current cost (manifest).
      gap>0 => codebook alone is cheaper than the net (necessary, not sufficient).
  (B) cheap-selector battery over ALWAYS-1x1-output tasks: does ANY O(1) color rule
      (corner / most|least-freq / unique / max|min color) fit all 268? 100% => real win.

VERDICT 2026-07-08 (this run, active set at BEST~7264.52): DRY.
  22 tasks have codebook-LB < cost, but the small output is the RESULT of irreducible input
  computation, not a cheap function. Battery over the 6 always-1x1 tasks: NO 100% selector.
  Closest = task346 least_freq 263/267 (4 misses need spatial detection for sprinkle-noise
  count flips). Same wall as memorization-dead: ARC computes a small answer FROM a complex
  input; the computation is the cost. Inspected 346 (spatial mono-block detect) + 355
  (speck-count-per-block) — both genuine, active nets already near closed-form.
REOPEN: a NEW active net that is heavy AND whose output truly is a cheap fixed function
  (e.g. a corner read) — re-run after big reformulations; or extend the battery with
  crop/reflect/transpose selectors for variable-size-output tasks (174/325/394 untested).

Usage: PYTHONPATH=. uv run python reports/scripts/output_entropy_scan.py
"""
import json, collections
import numpy as np
from src.harness import load_task

MAN = {t["task"]: t for t in json.load(open("reports/overfit_manifest.json"))["tasks"]}


def ghash(g):
    return (len(g), len(g[0]), tuple(tuple(r) for r in g))


def chist(g):
    h = [0] * 10
    for r in g:
        for c in r:
            h[c] += 1
    return tuple(h)


def selectors(g):
    g = np.array(g)
    H, W = g.shape
    nz = g[g > 0]
    out = {
        "corner_tl": g[0, 0], "corner_tr": g[0, W - 1],
        "corner_bl": g[H - 1, 0], "corner_br": g[H - 1, W - 1],
        "center": g[H // 2, W // 2],
    }
    if len(nz):
        vals, cnts = np.unique(nz, return_counts=True)
        out["most_freq"] = vals[np.argmax(cnts)]
        out["least_freq"] = vals[np.argmin(cnts)]
        u = vals[cnts == 1]
        out["unique1"] = u[0] if len(u) == 1 else -1
        out["max_color"] = vals.max()
        out["min_color"] = vals.min()
    return out


def main():
    rows = []
    onexone = []
    for n in range(1, 401):
        try:
            t = load_task(n)
        except Exception:
            continue
        exs = t.get("train", []) + t.get("test", []) + t.get("arc-gen", [])
        outs = [e["output"] for e in exs]
        N = len(outs)
        K = len({ghash(o) for o in outs})
        max_area = max(len(o) * len(o[0]) for o in outs)
        km, coll = {}, 0
        for e in exs:
            k = chist(e["input"])
            oh = ghash(e["output"])
            if k in km:
                coll += km[k] != oh
            else:
                km[k] = oh
        cur = MAN.get(n, {}).get("cost")
        rows.append((n, N, K, max_area, K * max_area, cur, coll == 0))
        if all(len(e["output"]) == 1 and len(e["output"][0]) == 1 for e in exs):
            onexone.append((n, exs))

    cand = [r for r in rows if r[5] and r[4] < r[5]]
    cand.sort(key=lambda r: r[5] - r[4], reverse=True)
    print("(A) codebook-LB < current cost:")
    print(f"{'task':>4} {'N':>4} {'K':>4} {'maxA':>5} {'memoLB':>7} {'cur':>6} {'gap':>6} hist")
    for n, N, K, mA, lb, cur, hs in cand:
        print(f"{n:4d} {N:4d} {K:4d} {mA:5d} {lb:7d} {cur:6d} {cur - lb:6d} {'YES' if hs else '-'}")

    print(f"\n(B) cheap-selector battery over {len(onexone)} always-1x1-output tasks:")
    print(f"{'task':>4} {'cur':>6} {'bestRule':>12} {'hit':>10}")
    for n, exs in onexone:
        N = len(exs)
        agg = collections.defaultdict(int)
        for e in exs:
            o = e["output"][0][0]
            for k, v in selectors(e["input"]).items():
                agg[k] += v == o
        best = max(agg.items(), key=lambda x: x[1]) if agg else ("none", 0)
        flag = "  <== 100% WIN" if best[1] == N else ""
        print(f"{n:4d} {str(MAN.get(n, {}).get('cost')):>6} {best[0]:>12} {best[1]:>4}/{N}{flag}")


if __name__ == "__main__":
    main()
