"""Floor-break target ranker — the work queue, recomputed from ground truth.

Usage: PYTHONPATH=. .venv/bin/python reports/fb_rank.py [--json]

For every adopted custom net it computes the realistic floor-break headroom
(points gained by dropping to the ~3600B label-map floor, capped at the mem-0
single-Conv ceiling). For public-base tasks in the feasible queue it shows the
current stored points (the build target is ~16.5). Sorted by headroom desc so
each build wave grabs the richest unbuilt target.

Floor reference: score = max(1, 25 - ln(mem+params)).
  mem 3600  -> 16.81 pts (label-map+Equal floor)
  mem 0     -> 18.42 pts (params ~900, single Conv)  [best realistic]
"""
import json, math, sys

m = json.load(open("reports/manifest.json"))["tasks"]
queue = json.load(open("reports/fb_publicnet_queue.json"))

FLOOR_MEM = 3600          # label-map+Equal realistic floor
def pts(memparams):
    return max(1.0, 25.0 - math.log(max(1.0, memparams)))

FLOOR_PTS = pts(FLOOR_MEM)

rows = []
for k, v in m.items():
    meth = v.get("method") or ""
    cur_mem = v.get("memory", 0) + v.get("params", 0)
    cur_pts = v.get("points", 0.0)
    if "custom" in meth:
        head = max(0.0, FLOOR_PTS - cur_pts)
        rows.append((head, int(k), cur_pts, v.get("memory"), v.get("params"), "custom"))

rows.sort(reverse=True)
print("=== ADOPTED CUSTOMS by floor-break headroom (potential pts to ~3600B floor) ===")
print(f"{'task':>4} {'cur_pts':>7} {'mem':>8} {'params':>6} {'->floor':>7} {'gain':>5}")
tot = 0.0
for head, k, cp, mem, pa, _ in rows:
    if head < 0.3:
        continue
    tot += head
    print(f"{k:>4} {cp:>7.2f} {mem:>8} {pa:>6} {FLOOR_PTS:>7.2f} {head:>5.2f}")
print(f"--- total reducible headroom on customs: ~{tot:.1f} pts (optimistic ceiling)")

print("\n=== PUBLIC-NET FEASIBLE QUEUE (unbuilt; build target ~16.5) ===")
done = set(queue.get("done_publicnet", []))
adopted_custom = {int(k) for k, v in m.items() if "custom" in (v.get("method") or "")}
seen = set()
for grp in ("feasible_queued", "building_now"):
    for k in queue.get(grp, []):
        if k in seen or k in done or k in adopted_custom:
            continue
        seen.add(k)
        v = m.get(str(k), {})
        cp = v.get("points", 0.0)
        gain = max(0.0, FLOOR_PTS - cp)
        print(f"{k:>4} cur_pts={cp:>6.2f} mem={v.get('memory'):>8} method={v.get('method')}  (~+{gain:.1f} if built)")

if "--json" in sys.argv:
    out = {"customs_headroom": [[k, round(h, 2), cp] for h, k, cp, *_ in rows if h >= 0.3]}
    json.dump(out, open("reports/fb_rank.json", "w"), indent=1)
    print("\nwrote reports/fb_rank.json")
