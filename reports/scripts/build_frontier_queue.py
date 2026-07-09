#!/usr/bin/env python3
"""Build a ranked "frontier review queue".

Produces the ordered list of tasks a human reviews to find semantic shortcuts
that could move a task into the 20+ score band (counted cost <= ~148 bytes).

Outputs:
  reports/frontier_queue.json  - full sorted list (rank_score desc)
  reports/frontier_queue.md    - human-readable top-60 + frontier seeds + summary

Run:  uv run python reports/scripts/build_frontier_queue.py
"""
import json
import math
import os
import statistics
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(REPO, "reports", "manifest.json")
SEMMAP = os.path.join(REPO, "reports", "semantic_task_map.json")
ORACLE_DIR = os.environ.get(
    "NEUROGOLF_ORACLE_DIR",
    str(Path(REPO) / "reports" / "oracles" / "arc-code-golf-solutions"),
)
OUT_JSON = os.path.join(REPO, "reports", "frontier_queue.json")
OUT_MD = os.path.join(REPO, "reports", "frontier_queue.md")

TARGET_POINTS = 25.0 - math.log(148.0)  # points a task would score at cost 148B

# Annotation sets
DEAD_CROP = {329, 150, 155, 289, 341, 239, 43, 345, 340, 335, 75}
DEAD_SCOUT = {273, 76, 54, 157}
WALL = {4, 18, 2, 44, 118, 209}
KNIFE_EDGE = {220, 230, 294}
FRONTIER_SEED = {92, 233, 349, 367, 396, 74, 138, 202, 204, 25, 64, 17,
                 222, 328, 208, 55, 366}


def oracle_len(task):
    """Length in characters of the golf solution, minus header comments and
    whitespace. Shorter = semantically simpler rule. Returns None if missing."""
    path = os.path.join(ORACLE_DIR, "task%03d.py" % task)
    if not os.path.exists(path):
        return None
    n = 0
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):  # full-line comment
                continue
            # count non-whitespace characters only (strip inner whitespace too)
            n += len("".join(s.split()))
    return n


def abbrev_class(cid):
    """Abbreviate a compiler_*/cost_* class id for the table."""
    if cid.startswith("compiler_"):
        return "c:" + cid[len("compiler_"):]
    if cid.startswith("cost_"):
        return "$:" + cid[len("cost_"):]
    return cid


def main():
    manifest = json.load(open(MANIFEST))["tasks"]
    semmap = json.load(open(SEMMAP))
    sem_by_task = {t["task"]: t for t in semmap["tasks"]}

    # --- oracle lengths for all 400 (compute median over present ones) ---
    lens = {}
    missing_oracle = []
    for t in range(1, 401):
        ol = oracle_len(t)
        if ol is None:
            missing_oracle.append(t)
        else:
            lens[t] = ol
    median_len = statistics.median(lens.values())

    rows = []
    for t in range(1, 401):
        key = str(t)
        if key not in manifest:
            # should not happen given 400 tasks; skip defensively
            continue
        m = manifest[key]
        points = m["points"]
        mem = m.get("memory", 0)
        params = m.get("params", 0)
        cost = mem + params
        method = m.get("method", "")

        gain = TARGET_POINTS - points
        if gain < 0:
            gain = 0.0

        ol = lens.get(t)
        if ol and ol > 0:
            simplicity = median_len / ol
        else:
            simplicity = 1.0  # neutral when oracle missing
        rank_score = gain * math.sqrt(simplicity)

        # classes: only compiler_*/cost_*, abbreviated
        sem = sem_by_task.get(t, {})
        classes = [c for c in sem.get("classes", [])
                   if c.startswith("compiler_") or c.startswith("cost_")]
        classes_abbr = [abbrev_class(c) for c in classes]

        flags = []
        if t in DEAD_CROP:
            flags.append("dead_crop")
        if t in DEAD_SCOUT:
            flags.append("dead_scout")
        if t in WALL:
            flags.append("wall")
        if t in KNIFE_EDGE:
            flags.append("knife_edge")
        if t in FRONTIER_SEED:
            flags.append("frontier_seed")

        rows.append({
            "task": t,
            "points": round(points, 4),
            "mem": mem,
            "params": params,
            "cost": cost,
            "method": method,
            "gain_to_20": round(gain, 4),
            "oracle_len": ol,
            "simplicity": round(simplicity, 4),
            "rank_score": round(rank_score, 4),
            "classes": classes_abbr,
            "flags": flags,
        })

    # queue excludes tasks already >= 20 (gain == 0)
    queue = [r for r in rows if r["gain_to_20"] > 0]
    queue.sort(key=lambda r: r["rank_score"], reverse=True)

    # --- JSON output ---
    out = {
        "meta": {
            "target_points_at_148B": round(TARGET_POINTS, 6),
            "median_oracle_len": median_len,
            "n_tasks": len(rows),
            "n_in_queue": len(queue),
            "n_already_20plus": sum(1 for r in rows if r["gain_to_20"] == 0),
            "missing_oracle": missing_oracle,
            "total_gain_to_20_available": round(sum(r["gain_to_20"] for r in rows), 4),
        },
        "queue": queue,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    # --- band summary (over queued tasks, by current points) ---
    def band(lo, hi):
        return [r for r in queue if lo <= r["points"] < hi]
    b13_16 = band(13, 16)
    b16_18 = band(16, 18)
    b18_20 = band(18, 20)
    total_gain = out["meta"]["total_gain_to_20_available"]

    # --- Markdown output ---
    def fmt_row(r):
        cls = ",".join(r["classes"])[:34]
        flg = ",".join(r["flags"])
        meth = r["method"].replace("custom:", "")
        return ("| %d | %.2f | %d | %d | %d | %s | %.3f | %s | %.2f | %.3f | %s | %s |"
                % (r["task"], r["points"], r["mem"], r["params"], r["cost"], meth,
                   r["gain_to_20"],
                   ("%d" % r["oracle_len"]) if r["oracle_len"] is not None else "-",
                   r["simplicity"], r["rank_score"], cls, flg))

    header = ("| task | pts | mem | params | cost | method | gain | ora_len | "
              "simpl | rank | classes(c:/$:) | flags |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|"

    lines = []
    lines.append("# Frontier review queue")
    lines.append("")
    lines.append("Ranked list of sub-20 tasks to human-review for a semantic "
                 "shortcut into the 20+ band (counted cost <= ~148B).")
    lines.append("")
    lines.append("- Target points at 148B: `%.4f` (= 25 - ln(148))" % TARGET_POINTS)
    lines.append("- `gain_to_20` = target - current points (clamped >=0; tasks "
                 ">=20 excluded from queue).")
    lines.append("- `simplicity` = median_oracle_len(%g) / oracle_len; "
                 ">1 = simpler-than-median rule." % median_len)
    lines.append("- `rank_score` = gain_to_20 * sqrt(simplicity).")
    lines.append("- classes abbreviate `compiler_*` as `c:` and `cost_*` as `$:`.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- Tasks in queue (sub-20): **%d** of %d; already 20+: **%d**."
                 % (len(queue), len(rows), out["meta"]["n_already_20plus"]))
    lines.append("- Current-points bands (queued tasks): "
                 "13-16 = **%d**, 16-18 = **%d**, 18-20 = **%d**."
                 % (len(b13_16), len(b16_18), len(b18_20)))
    lines.append("- Total `gain_to_20` available across all 400: **%.2f** points."
                 % total_gain)
    if missing_oracle:
        lines.append("- Missing oracle files: %s"
                     % ", ".join(str(x) for x in missing_oracle))
    else:
        lines.append("- Missing oracle files: none (all 400 covered).")
    lines.append("")
    lines.append("## Top 60 by rank_score")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for r in queue[:60]:
        lines.append(fmt_row(r))
    lines.append("")
    lines.append("## Frontier seeds (named in HIGH_SCORE_FRONTIER.md)")
    lines.append("")
    lines.append("Shown regardless of rank; some may already be 20+ (then absent "
                 "from the main queue).")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    seed_rows = [r for r in rows if r["task"] in FRONTIER_SEED]
    seed_rows.sort(key=lambda r: r["rank_score"], reverse=True)
    for r in seed_rows:
        lines.append(fmt_row(r))
    lines.append("")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")

    # --- console report ---
    print("Wrote %s" % OUT_JSON)
    print("Wrote %s" % OUT_MD)
    print("Median oracle len: %g" % median_len)
    print("Total gain_to_20 available: %.4f" % total_gain)
    print("Missing oracle: %s" % (missing_oracle or "none"))
    print("Top 15 by rank_score:")
    print("  %-4s %-8s %-6s %-8s %-8s" % ("task", "points", "cost", "ora_len", "rank"))
    for r in queue[:15]:
        print("  %-4d %-8.3f %-6d %-8s %-8.3f"
              % (r["task"], r["points"], r["cost"],
                 str(r["oracle_len"]), r["rank_score"]))


if __name__ == "__main__":
    main()
