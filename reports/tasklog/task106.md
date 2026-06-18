# task106 — 46442a0e (C4 rotational symmetrization of a small grid)

**Rule:** INPUT is a size×size grid (size∈{2,3}) of nonzero colours (1..9) at the TOP-LEFT.
OUTPUT is 2size×2size = the C4 (4-fold rotation) orbit of the input about the N×N centre
(N=2size): each cell (r,c)→{(r,c),(c,N-1-r),(N-1-r,N-1-c),(N-1-c,r)}. Equivalently OUTPUT =
OR over {G, rot90, rot180, rot270} where G = input placed top-left of an N×N grid; rot90 ccw =
F_N @ Gᵀ (F_N = N×N anti-identity). All colours nonzero ⇒ active region is exactly the top-left
6×6 (≤3 size). The four quadrants are disjoint (size×size each) so OR == max == sum.

**Current (prior stored):** 17.94 pts, ext:kojimar6275, mem 1039, params 125.
**Target tier:** A/B (data-dependent rotation). The STORED net is already an optimal closed-form
**constant GatherElements**: Slice→3×3 [1,10,3,3], ArgMax→colour-index, flatten to a 10-elem value
vector (9 cells + appended bg-0), then ONE GatherElements with a [1,1,36] permutation index that
realises the entire C4 symmetrization — TWO precomputed perm tables (size=2 / size=3) selected by a
size flag (the 144B int32 Where). Equal→one-hot [1,10,6,6] bool→Pad→output.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | colf Conv on 6×6 slice; rot90 = F_N@Tᵀ ×3 MatMul, Max-OR; Equal→Pad bool | A | 12690 | 66 | 15.55 | — | works, heavy |
| 2 | sentinel-bg, single-channel uint8 Pad→Equal | A | 3654 | 57 | 16.78 | — | trim |
| 3 | 6×6 uint8 one-hot → Pad as FREE uint8 output | A | 3438 | 57 | 16.84 | 200/200 | best (mine) |

## Best achieved
16.84 @ mem 3438 params 57 — adopted? **N**. Beats prior 17.94? **NO** — my MatMul-rotation
approach is structurally heavier than the stored constant-gather. Stored net remains best (17.94).
My net is correct & generalizes 200/200 but is +1.1 WORSE.

## Irreducible-floor analysis
The STORED net is at floor. Two ~360B planes dominate and are both structural:
(1) entry 10-channel slice [1,10,3,3] fp32 = 360B (ArgMax needs the multi-channel float; Slice
preserves fp32; 3×3 is the minimum that covers size=3); (2) the one-hot expansion [1,10,6,6] bool
= 360B (the 10-ch reduce/expand floor at the 6×6 working canvas). Plus the GatherElements index
machinery (~150B) + 125 params (two 36-elem perm tables + selector). Total 1164. To beat by +0.3
needs ≤862B; the two 720B floor planes alone forbid it. The ONLY loose piece is the size-select
Where (144B int32 [1,1,36] + ~36 params); a hypothetical single merged perm would save ~180,
landing ~985 → 17.99 (+0.05), still far short of +0.3. The two perm tables genuinely differ
(size=2 output is 4×4 with bg fill, size=3 is full 6×6) so they cannot be merged into one gather.

## OPEN ANGLES (re-attack backlog)
- Merge the two C4 perm tables into ONE size-agnostic gather (eliminate the size-flag Where). The
  obstacle: size=2 vs size=3 output extents differ (4×4 vs 6×6), so a fixed perm over the value
  vector cannot route both — would need a size-parametric index ARITHMETIC (e.g. clip/offset by N)
  that is itself ≥ the Where it replaces. Likely net-neutral, untried in detail. Even if free: +0.05.
- Shrink the [1,10,3,3] entry: impossible below 360B (ArgMax needs ≥ all colour channels over the
  3×3 region; fp32-forced by Slice).

## INSIGHT (transferable)
⭐ A fixed-position SMALL-grid C4/Cn symmetrization where input & output are pure colour COPIES is
NOT a MatMul-rotation task — it is a **constant GatherElements**: flatten the K×K input to a value
vector (+1 appended bg slot), precompute the [1,1,(2K)²] permutation index that maps each output
cell to its source cell under the rotation orbit, and emit the whole symmetrized grid in ONE gather
(NO rotation planes, NO transpose chain). For a variable grid size, select among per-size perm
tables with a size flag. This beats the F_N@Tᵀ MatMul approach by ~1.1 pts because it removes all
6 rotation intermediates. The MatMul/reverse-transpose idiom (task027/112) is for DATA-DEPENDENT-
centre rotations on the full canvas; a FIXED top-left small grid collapses to a constant gather.
