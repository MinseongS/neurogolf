# task067 — 2dee498d

**Rule:** Generator builds an input grid of (size) rows x (3*size) cols, size in 2..5,
as three side-by-side size x size blocks: block0 = colors[r][c], block1 = colors
optionally vertically flipped, block2 = colors[r][c]. The OUTPUT is the size x size
grid colors[r][c], i.e. exactly block0 (== block2) — the LEFT size columns of the
input cropped to size rows. Pure copy/crop. Off-grid cells are all-zero; in-grid
cells (incl. colour 0 -> channel 0 = 1) always occupy their column.
**Current:** 19.247 pts, ext:kojimar6275, mem 282, params 33
**Target tier:** S (pure spatial copy/crop, output is a column-subset of input)

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | prior: ReduceMax col_occ + cumsum-Conv + Greater + Where | S | 282 | 33 | 19.25 | — | baseline |
| 1 | scalar size^2 from ReduceSum(input,[1,2,3])/3; keep c iff c^2<size^2 via const squared col-ramp; Where | S | 38 | 32 | 20.75 | 200/200 | ADOPT-worthy |

## Best achieved
20.75 @ mem 38 params 32 — beats prior 19.25 by +1.50 (Y).

## Irreducible-floor analysis
Only full-width intermediate is `keep_b` bool [1,1,1,30] = 30B; rest are [1,1,1,1]
scalars (total, size2 = 8B). Output Where lands in the FREE output. The 30B bool
column mask is the floor for any "crop to a data-dependent column count" — you must
materialise one per-column keep flag to feed Where. Could shave ~0 more meaningfully.

## OPEN ANGLES (re-attack backlog)
- keep_b is already minimal (bool, 30 elems). A Slice-based crop would need a
  data-dependent size -> symbolic-dim "could not be measured" trap; bool keep+Where
  is the safe minimal form. Effectively at the practical floor for this task.

## INSIGHT (transferable)
⭐ For a "crop the input to its first `size` columns/rows" task where the grid width
is a known multiple of size (here 3*size) and EVERY in-grid cell occupies its row/col
(channel-0 set for colour-0), recover size as a pure SCALAR: total = ReduceSum(input,
[1,2,3]) = K*size^2; size^2 = total/K. Then keep column c iff c^2 < size^2 by comparing
a CONSTANT squared ramp const [0,1,4,...] against the scalar — NO per-column occupancy
plane and NO cumsum Conv. This collapses the public "ReduceMax-occ + cumsum-Conv +
threshold" idiom (two [1,1,1,30] fp32 planes, 240B) down to two scalars + one 30B bool
mask (282 -> 38B, 19.25 -> 20.75, +1.50). Comparing c^2<size^2 avoids a Sqrt entirely.
