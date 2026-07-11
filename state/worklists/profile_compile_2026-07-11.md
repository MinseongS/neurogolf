# Profile-compile worklist (oracle info-structure triage, 2026-07-11)

Source: fable-fork triage of all 400 rule oracles (reference/arc-code-golf-solutions) against
deployed costs. Classes: P=profile-only, S=sparse-positional, C=per-cell (skip), I=iterative (skip).
Counts over the 223 fat-middle tasks: P=49 (32 above bar), S=61 (15 above bar), C≈85, I≈28.
Naive P-gap +58pt (realistic +20-35 after floors). Builders MUST check state/tasks/taskNNN.md for
binding negatives first, derive worst-case semantics from the GENERATOR, and fresh-A/B ≥3000.

## P-list (gap = ln(cost/naive bound))
| gap | task | cost | rule |
|-----|------|------|------|
| 2.81 | 208 | 4172 | least-color ring = 4 bbox scalars → two band outer-products [WAVE1] |
| 2.80 | 014 | 4097 | rows containing color-c + col-extent crop [WAVE1] |
| 2.74 | 359 | 3866 | per-row majority color → row fill (± transpose) [WAVE1] |
| 2.40 | 091 | 2764 | crop between two 5-columns / 5-row extents [WAVE1] |
| 2.33 | 177 | 2565 | bbox crop + horizontal flip [WAVE1] |
| 2.06 | 185 | 1955 | per-tile majority (separator-partitioned) [WAVE1] |
| 2.03 | 084 | 1895 | fixed diagonals from grid size only [WAVE1] |
| 2.01 | 184 | 1860 | block-partition → per-block first color |
| 2.00 | 055 | 1856 | three 8-lines → 6 region fills [WAVE1] |
| 2.00 | 213 | 1850 | border-color max fill + crop |
| 1.99 | 388 | 1829 | cols-containing-color → 8-fill + 2× tile |
| 1.94 | 224 | 1737 | inner ring of 5-bbox → C |
| 1.91 | 070 | 1682 | 1s inside 8-bbox → 3 |
| 1.81 | 063 | 1521 | empty-interior rows/cols → 3 |
| 1.80 | 398 | 1510 | staircase tiling of row pattern |
| 1.74 | 013 | 1424 | 2 cells → paint full columns |
| 1.73 | 232 | 1410 | per-row alternate [color,5,…] after first nonzero |
| 1.71 | 297 | 1389 | rows 2+ = broadcast tiled first-row |
| 1.71 | 335 | 1380 | L-connector 8→2 (regime-cracked before — check ledger) |
| 1.61 | 175 | 1255 | out=max(X,Xᵀ), 0→most-common, diag→v |
| 1.55 | 244 | 1179 | uniform-row/col partition compress + reverse |
| 1.52 | 301 | 1141 | histogram → sorted bar chart |
| 1.51 | 246 | 1128 | L-path 8s between the 2 and 3 |
| 1.43 | 293 | 1043 | per-row broadcast first cell (± transpose) |
| 1.41 | 183 | 1024 | shrink center, 8→corner color per quadrant |
| 1.40 | 060 | 1010 | content rows: left=row[0], right=row[-1], center 5 |
| 1.39 | 010 | 1002 | 5-cells → column-first-appearance rank color |
| 1.33 | 239 |  945 | histogram bar chart |
| 1.28 | 151 |  900 | full-uniform cross → 8-cell stamp |
| 1.27 | 259 |  892 | non-1 bbox crop + 1→0 |
| 1.24 | 047 |  862 | each cell paints full row+col |

## S-list primary
278 (4505, 2-pair→3-ring), 387 (3984, point pairs), 377 (3967, RLE row→rings), 050 (3849, rays
between 8-pairs), 069 (2919, template stamp at markers = 368-class runtime kernel), 270 (2858,
single-step moves), 397 (2751, 2×2 block detect). Secondary: 008 383 037 190 212 250 245 030.

## Caveats
- Several P tasks carry other-lens ledger entries (161 re-golfed 07-11; 335/303 regime-cracked;
  084 S5-optimized) — builder checks tasklog first.
- Oracle magic constants are NOT ground truth (bundled-tuned); the GENERATOR is.
- WAVE1 (8 builders, 2026-07-11): 208 014 359 091 177 185 084 055.

## 49th-place distribution bounds (2026-07-11, rigorous)
His histogram: 0 tasks <15, 11 in [15,16). Ours: 3 <15, 26 <16.
=> At least 15 of our 26 sub-16 tasks have a <=8103-cost realization at 49th place.
Under the minimal-gap assignment (his worst 11 = our costliest 11), the OTHER 15 carry
guaranteed gaps: 101 >=0.47, 076 >=0.46, 118 >=0.42, 138 >=0.37, 198 >=0.34, 173 >=0.33,
191 >=0.31, 145 >=0.26, 204 >=0.23, 066 >=0.22, 025 >=0.19, 350 >=0.11, 338 >=0.09,
324 >=0.06, 216 >=0.03 — SUM >=+3.9 (any other assignment gives MORE).
Also: our 3 sub-15 (233/018/366) are <=22026 for him (233 gap >=0.24).
IMPLICATION: today's "floor" verdicts on 076/173/101 are OUR-PHYSICS floors, falsified in
principle by his histogram. These 15 tasks = the reverse-engineering / community-intel target
list ("under 9000 possible?" threads).

## LANE CLOSED (2026-07-11 late)
Wave1 (8) + wave2 (8) + S-list (4) = 20 judged: 3 wins adopted (055 +0.065, 363* +0.048, 017* +0.028
[*tier1 probes], plus wave1 177/359), 17 empirical floors. Root causes across all floors: (1)
fp32-input-coupling (tensors born from / co-bound with the free fp32 input are dtype-pinned; escape
costs an 18KB cast), (2) the 900B non-separable label/route plane before the free Equal, (3)
params-element-count floors (398 mem=49!), (4) prior sessions already applied this exact idiom
(07-09 profile-compile adoptions + public min-merges are tighter than the naive oracle bounds).
Remaining P-rows (232/297/335/175/244/301/246/293/183/060/010/239/151/259/047) are LOWER-gap than
the judged set → expect the same floors; do NOT dispatch builders without a new physics ingredient.
All 20 incumbents measured fresh-clean (3000/3000) except previously-known tails.
