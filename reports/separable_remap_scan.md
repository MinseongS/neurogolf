# Separable Spatial Remap Scan (Playbook mechanism 14)

Scan date: 2026-07-03. Pure-numpy over bundled `data/taskNNN.json` (train + test + arc-gen).

## What this finds

Tasks whose rule is a **fixed content-independent separable spatial remap**: for ALL
examples and every colour channel,
`output[r,s] == input[rowmap[r], colmap[s]]` (or background 0 when a row/col is a
zero-pad slot). This is exactly the class killed near-free by mechanism 14
(single 5-operand Einsum `ra,ai,zcij,bj,sb->zcrs`, mem=0, params = |U|+|S| after
factoring P=U@S, Q=V@T). Covers fixed crops, tiles/repeats, nearest upscales/downscales,
sublattice reads, row/col permutations, reflections, mirror-tilings, zero-pad shifts.

### Solver
Per task: require constant input shape AND constant output shape across all examples
(a fixed P (out×in) / Q needs fixed dims). Then solve `rowmap`,`colmap` as a binary CSP
(AC-3 + backtracking on a 16-example spread), **verify reconstruction on ALL examples**,
and minimize K (distinct source indices) for the cost projection. NONE row/col = zero-pad.

Projected params (mem=0): for each axis `min(out*in, out*K + K*in)` where
K = number of distinct source rows/cols used. `proj = proj_P + proj_Q`.
Projected points = `25 - ln(proj)`. Gain = projected_points - current_points.

## Negative evidence (funnel over 400 tasks)

| stage | count |
|---|---|
| variable input shape (skip — no fixed P) | 196 |
| variable output shape (skip) | 14 |
| constant in+out shape (candidates) | 190 |
| AC-3 proved non-separable | 172 |
| separable on subset but fails full arc-gen (near-miss, rejected) | 2 (tasks 130, 287) |
| **qualified separable remaps** | **16** |

All 16 qualified maps were independently re-verified by direct reconstruction across
every example (train+test+arc-gen). task108 (already landed via mechanism 14) is detected
correctly with proj=300 == its current params.

## Actionable candidates (positive gain, excl. task108 + already-cheaper tasks)

Ranked by projected gain. `cur` = current mem+params from `reports/manifest.json`.

| task | cur (mem+par) | remap type | proj params | cur pts | proj pts | gain |
|---|---|---|---|---|---|---|
| 152 | 360+98 = 458 | 3×3→6×6 mirror-tile (row & col reflect-tile, K=3/3) | 36 | 18.873 | 21.416 | **+2.543** |
| 211 | 240+156 = 396 | 3×2→9×4 mirror-stack ×3 + col mirror (K=3/2) | 35 | 19.019 | 21.445 | **+2.426** |
| 83  | 376+71 = 447 | 3×4→6×8 mirror-tile (K=3/4) | 50 | 18.897 | 21.088 | **+2.191** |
| 142 | 144+138 = 282 | 3×3→6×6 mirror-tile (K=3/3) | 36 | 19.358 | 21.416 | **+2.058** |
| 135 | 0+200 = 200 | 9×9→3×3 fixed crop (top-right corner, K=3/3) | 54 | 19.702 | 21.011 | **+1.309** |
| 53  | 0+30 = 30 | 3×3→3×3 row-shift + zero-pad top row (K=2/3) | 18 | 21.599 | 22.110 | **+0.511** |
| 164 | 0+30 = 30 | col upscale, rows identity (K=3/3) | 27 | 21.599 | 21.704 | +0.105 |
| 172 | 0+30 = 30 | row upscale, cols identity (K=3/3) | 27 | 21.599 | 21.704 | +0.105 |
| 210 | 0+30 = 30 | row upscale, cols identity (K=3/3) | 27 | 21.599 | 21.704 | +0.105 |
| 311 | 0+30 = 30 | col upscale, rows identity (K=3/3) | 27 | 21.599 | 21.704 | +0.105 |

**Sum of positive gain ≈ +11.46 pts.** Top-5 (152, 211, 83, 142, 135) ≈ **+10.53 pts** and
are the clear wins — all currently carry real detection-plane bloat (mem 144–376) that
mechanism 14 dissolves to mem=0 + a ~35–54 param factored map. Tasks 53/164/172/210/311
are already at the ONNX param floor (0+30) so gains are marginal (+0.1–0.5).

### Exact maps for the top candidates
- **152 / 142**: `rowmap=colmap=[0,1,2,2,1,0]` — 2×2 block mirror-tiling of a 3×3 tile.
- **83**: `rowmap=[0,1,2,2,1,0]`, `colmap=[0,1,2,3,3,2,1,0]` — mirror-tile 3×4→6×8.
- **211**: `rowmap=[2,1,0,0,1,2,2,1,0]`, `colmap=[1,0,0,1]` — reversed 3-row mirror-stack, 2-col mirror.
- **135**: `rowmap=[0,1,2]`, `colmap=[6,7,8]` — fixed crop of the top-right 3×3 corner of a 9×9.
- **53**: `rowmap=[NONE,0,1]`, `colmap=[0,1,2]` — shift rows down by one, top row → background.

## Qualified-but-no-play (already cheaper than the remap projection — do NOT touch)
| task | cur | why skip |
|---|---|---|
| 108 | 0+300 | already landed via mechanism 14 (proj == current) |
| 87  | 0+5   | reflect — native flip is 5 params < 18 projected |
| 140 | 0+5   | reflect — native flip cheaper |
| 223 | 0+5   | small upscale already at 5 params < 54 projected |
| 116 | 0+30  | upscale already at 30 < 34 projected |
| 385 | 0+30  | remap already at 30 < 116 projected |

## Notes / caveats
- Projections assume the mechanism-14 factored Einsum reaches its theoretical
  `min(out*in, out*K+K*in)` param count with mem=0. Real ONNX export may add a few
  params; treat gains as upper bounds and re-measure grader mem/params post-build.
- Reflections (87/140) and tiny upscales are already golfed below the separable
  projection via native flip/resize ops — mechanism 14 is not the cheapest path there.
- Scan artifacts: `results.json`, `stats.json` in the session scratchpad
  (`.../scratchpad/remap_scan/`); solver `scan.py`.
- No candidate nets built — scan only, per mission.
