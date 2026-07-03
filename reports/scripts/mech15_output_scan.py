#!/usr/bin/env python
"""Systematic finder for playbook mechanism 15 (signed-channel priority overlay).

Mechanism 15 (see reports/REBUILD_PLAYBOOK.md entry 15): the grader decodes
`(out > 0.0)` per channel, so overlap/paint-order priority is LINEAR — a task
qualifies iff its OUTPUT decomposes into a union of axis-aligned SEPARABLE fills
(solid monochrome rects / segments / frames-as-4-segments), optionally on top of
the copied input, with a CONSTANT colour roster across examples and a fixed
overlap ordering. All 3 hits so far (234/335/092) were found BY HAND.

This scan is a *structural, bundled-only, numpy* screen. It is
NECESSARY-NOT-SUFFICIENT: the generator may emit more rects / data-dependent
colours than the bundled train/test examples reveal, so a structural qualifier
still needs per-task mechanism verification before adopting.

Method per task 1..400:
  1. Load bundled examples (train+test) via src.harness.load_task.
  2. For same-shape examples: object = delta cells where output != input
     (fills ride on the copied input). For different-shape examples: object =
     the whole output grid.
  3. Greedy maximal axis-aligned monochrome-rectangle decomposition of the
     object; record rect count R (must be <= RMAX with zero residual — residual
     is impossible by construction, so R<=RMAX is the discriminator). Every rect
     is solid monochrome by construction.
  4. Constant-roster test: literal fill-colour set identical across all
     examples -> CONSTANT, else DATA_DEPENDENT (HARDER: colour must come from
     input via tiny einsum W, not discarded).
  5. Overlap proxy: any two different-colour rect bounding-boxes intersect in an
     example -> OVERLAP (needs the signed-priority trick), else DISJOINT (plain
     signed routing).
  6. Join incumbent cost (mem+params) from reports/manifest.json; apply the
     exclusion filters; emit qualifiers sorted by cost with est headroom.

Writes reports/mech15_output_scan.{md,json}. Scan-only, no ONNX inference.
"""

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.harness import load_task  # noqa: E402

MANIFEST = ROOT / "reports" / "manifest.json"
OUT_MD = ROOT / "reports" / "mech15_output_scan.md"
OUT_JSON = ROOT / "reports" / "mech15_output_scan.json"

RMAX = 8                     # max rects per example to qualify structurally
FLOOR = 1500                 # realistic signed-fill floor (bytes) for headroom est
MIN_COST = 1200              # nothing to win below this (mem+params)
MAX_POINTS = 20.0            # already >=20 pts -> nothing to win

# S11 killed / floored cohort + the 3 existing hits (092/234/335)
EXCLUDE_KILLED = {233, 285, 370, 133, 54, 366, 41, 84, 162, 177, 92, 234, 335}
# Known walls (floor-bound, don't re-grind)
EXCLUDE_WALLS = {4, 18, 2, 44, 118, 209, 76}
EXCLUDE = EXCLUDE_KILLED | EXCLUDE_WALLS


def to_grid(g):
    return np.array(g, dtype=np.int16)


def max_rect_at(anchor_r, anchor_c, grid, avail):
    """Largest axis-aligned monochrome rectangle anchored at top-left (anchor).

    avail = boolean mask of cells still eligible (in-object, uncovered).
    Returns (r0, c0, r1, c1, colour) inclusive bounds of the best rectangle,
    all of whose cells share grid colour and are available.
    """
    H, W = grid.shape
    col = grid[anchor_r, anchor_c]
    # max width in the anchor row
    maxw = 0
    c = anchor_c
    while c < W and avail[anchor_r, c] and grid[anchor_r, c] == col:
        maxw += 1
        c += 1
    best = None
    best_area = 0
    # For each candidate width, grow height as far as every row of that width
    # is fully same-colour & available.
    for w in range(1, maxw + 1):
        h = 0
        r = anchor_r
        while r < H:
            row_ok = True
            for cc in range(anchor_c, anchor_c + w):
                if not (avail[r, cc] and grid[r, cc] == col):
                    row_ok = False
                    break
            if not row_ok:
                break
            h += 1
            r += 1
        area = w * h
        if area > best_area:
            best_area = area
            best = (anchor_r, anchor_c, anchor_r + h - 1, anchor_c + w - 1, int(col))
    return best


def decompose(grid, object_mask, rmax_cap):
    """Greedy monochrome-rect cover of object_mask over grid.

    Returns list of rects [(r0,c0,r1,c1,colour)] or None if it needed more than
    rmax_cap rects (bailed early).
    """
    avail = object_mask.copy()
    rects = []
    H, W = grid.shape
    while avail.any():
        # top-left-most available cell
        ys, xs = np.where(avail)
        # lexicographic top-left
        order = np.lexsort((xs, ys))
        ar, ac = ys[order[0]], xs[order[0]]
        rect = max_rect_at(int(ar), int(ac), grid, avail)
        if rect is None:
            return None
        r0, c0, r1, c1, col = rect
        avail[r0:r1 + 1, c0:c1 + 1] = False
        rects.append(rect)
        if len(rects) > rmax_cap:
            return None
    return rects


def bboxes_overlap(a, b):
    ar0, ac0, ar1, ac1, _ = a
    br0, bc0, br1, bc1, _ = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def analyze_task(task_num):
    """Return a per-task structural record (dict) or None if unloadable."""
    try:
        t = load_task(task_num)
    except Exception as e:
        return {"task": task_num, "error": f"load: {e}"}

    examples = list(t.get("train", [])) + list(t.get("test", []))
    if not examples:
        return {"task": task_num, "error": "no bundled examples"}

    per_ex = []
    rosters = []
    r_counts = []
    any_delta = False
    any_overlap = False
    shape_modes = set()
    qualifies = True

    for ex in examples:
        gin = to_grid(ex["input"])
        gout = to_grid(ex["output"])
        if gin.shape == gout.shape:
            mode = "delta"
            object_mask = gout != gin
        else:
            mode = "whole"
            object_mask = np.ones(gout.shape, dtype=bool)
        shape_modes.add(mode)

        n_obj = int(object_mask.sum())
        if n_obj > 0 and mode == "delta":
            any_delta = True
        if mode == "whole":
            any_delta = True  # whole-output tasks always have an "object"

        if n_obj == 0:
            # pure copy example: 0 rects, contributes nothing, still valid
            per_ex.append({"mode": mode, "R": 0, "colours": []})
            rosters.append(frozenset())
            r_counts.append(0)
            continue

        rects = decompose(gout, object_mask, RMAX)
        if rects is None:
            qualifies = False
            per_ex.append({"mode": mode, "R": None, "colours": None,
                           "note": f">{RMAX} rects"})
            r_counts.append(999)
            rosters.append(None)
            continue

        R = len(rects)
        r_counts.append(R)
        cols = sorted({rc[4] for rc in rects})
        rosters.append(frozenset(cols))
        per_ex.append({"mode": mode, "R": R, "colours": cols})
        if R > RMAX:
            qualifies = False
        # overlap proxy
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                if rects[i][4] != rects[j][4] and bboxes_overlap(rects[i], rects[j]):
                    any_overlap = True

    if not any_delta:
        return {"task": task_num, "error": "identity / empty object"}

    # roster class
    valid_rosters = [r for r in rosters if r is not None and len(r) > 0]
    if not valid_rosters:
        roster_class = "EMPTY"
    elif all(r == valid_rosters[0] for r in valid_rosters):
        roster_class = "CONSTANT"
    else:
        roster_class = "DATA_DEPENDENT"

    r_max = max((r for r in r_counts if r != 999), default=0)
    if any(r == 999 for r in r_counts):
        r_max = 999

    return {
        "task": task_num,
        "qualifies": bool(qualifies),
        "R_max": int(r_max) if r_max != 999 else None,
        "roster_class": roster_class,
        "overlap": "OVERLAP" if any_overlap else "DISJOINT",
        "shape_mode": "+".join(sorted(shape_modes)),
        "n_examples": len(examples),
        "roster_union": sorted(set().union(*[set(r) for r in valid_rosters]))
        if valid_rosters else [],
        "per_example": per_ex,
    }


def main():
    manifest = json.load(open(MANIFEST))["tasks"]

    records = []
    for n in range(1, 401):
        rec = analyze_task(n)
        if rec is None:
            continue
        # attach cost
        mi = manifest.get(str(n))
        if mi is not None:
            rec["cost"] = mi["memory"] + mi["params"]
            rec["memory"] = mi["memory"]
            rec["params"] = mi["params"]
            rec["points"] = mi["points"]
            rec["method"] = mi["method"]
        else:
            rec["cost"] = None
        records.append(rec)

    # Qualifiers = structurally qualify AND pass economic/exclusion filters
    qualifiers = []
    for rec in records:
        if rec.get("error"):
            continue
        if not rec.get("qualifies"):
            continue
        n = rec["task"]
        if n in EXCLUDE:
            continue
        cost = rec.get("cost")
        if cost is None or cost < MIN_COST:
            continue
        if rec.get("points", 0.0) >= MAX_POINTS:
            continue
        rec["est_headroom"] = cost - FLOOR
        qualifiers.append(rec)

    qualifiers.sort(key=lambda r: r["cost"], reverse=True)

    # ---- JSON dump (full records + qualifiers) ----
    out = {
        "params": {"RMAX": RMAX, "FLOOR": FLOOR, "MIN_COST": MIN_COST,
                   "MAX_POINTS": MAX_POINTS,
                   "excluded": sorted(EXCLUDE)},
        "n_qualifiers": len(qualifiers),
        "qualifiers": qualifiers,
        "all_records": records,
    }
    json.dump(out, open(OUT_JSON, "w"), indent=2, default=str)

    # ---- Markdown report ----
    lines = []
    lines.append("# Mechanism 15 output-decomposition scan (signed-channel priority overlay)")
    lines.append("")
    lines.append("Structural, bundled-only (train+test), numpy screen for playbook mechanism 15.")
    lines.append("**NECESSARY-NOT-SUFFICIENT**: bundled decomposition can under-count rects and")
    lines.append("miss data-dependent colours the generator emits. A qualifier still needs")
    lines.append("per-task mechanism verification (render generator outputs) before adopting.")
    lines.append("")
    lines.append(f"- RMAX rects/example = {RMAX}; headroom floor = {FLOOR}B; "
                 f"min cost = {MIN_COST}; max points = {MAX_POINTS}")
    lines.append(f"- Excluded (S11 killed/floored + hits + walls): {sorted(EXCLUDE)}")
    lines.append(f"- **Structural qualifiers after filters: {len(qualifiers)}**")
    lines.append("")
    lines.append("Roster class: CONSTANT = literal fill colours identical across examples "
                 "(plain signed W). DATA_DEPENDENT = HARDER (colour must be read from input "
                 "via tiny einsum, not discarded). Overlap: OVERLAP needs the signed-priority "
                 "trick; DISJOINT = plain signed routing.")
    lines.append("")
    lines.append("> CAVEAT — the overlap flag is a WEAK proxy: it intersects the bounding "
                 "boxes of the greedy rects of the *resolved final output*, so a true "
                 "paint-order overlap that is already resolved in the output reads as "
                 "DISJOINT. All 3 known hits (092/234/335) show DISJOINT here. Do not use it "
                 "to reject a candidate; use it only as a hint that OVERLAP tasks are "
                 "definitely in the priority-trick regime.")
    lines.append("")
    lines.append("> The 3 hand-found hits 092/234/335 all pass this structural screen "
                 "(092 R_max=8 DATA_DEPENDENT, 234 R_max=5 DATA_DEPENDENT, 335 R_max=2 "
                 "CONSTANT) — that is the validation that the screen catches the mechanism. "
                 "They are excluded below only because they already use it.")
    lines.append("")
    lines.append("| task | cost | mem | params | pts | R_max | roster | overlap | shape | headroom | method |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in qualifiers:
        lines.append(
            f"| {r['task']} | {r['cost']} | {r['memory']} | {r['params']} | "
            f"{r['points']:.2f} | {r['R_max']} | {r['roster_class']} | {r['overlap']} | "
            f"{r['shape_mode']} | {r['est_headroom']} | {r['method']} |"
        )
    lines.append("")
    lines.append("## Top-10 by cost — per-example detail")
    lines.append("")
    for r in qualifiers[:10]:
        lines.append(f"### task{r['task']:03d}  (cost {r['cost']}, {r['roster_class']}, "
                     f"{r['overlap']}, R_max {r['R_max']})")
        lines.append(f"- roster union (literal colours): {r['roster_union']}")
        for i, ex in enumerate(r["per_example"]):
            lines.append(f"  - ex{i} [{ex['mode']}] R={ex['R']} colours={ex.get('colours')}")
        lines.append("")

    OUT_MD.write_text("\n".join(lines))

    # ---- console summary ----
    print(f"structural qualifiers (post-filter): {len(qualifiers)}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print()
    print("TOP-10 BY COST:")
    for r in qualifiers[:10]:
        print(f"  task{r['task']:03d}  cost={r['cost']:>6}  R_max={r['R_max']}  "
              f"{r['roster_class']:<14} {r['overlap']:<8} {r['shape_mode']:<11} "
              f"headroom~{r['est_headroom']}  {r['method']}")


if __name__ == "__main__":
    main()
