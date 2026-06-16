# task011 — 09629e4f

**Rule:** The 11x11 grid is a 3x3 array of 3x3 mini-cells separated by a gray (5)
"hollywood squares" frame at rows/cols 3 and 7. Each mini-cell holds rainbow
pixels (colours {2,3,4,6,8}); exactly ONE cell ("chosen") has 4 coloured pixels,
every other has 5. The output keeps the frame and fills each output mini-cell
block (mr,mc) SOLID with the colour of the chosen cell's pixel at interior
position (mr,mc) (bg if empty) — i.e. output = the chosen 3x3 cell upscaled 3x
onto the frame. NOT a flood-fill/connectivity task: it is count-discrimination
(unique min-count cell) + a position->fill transposition.

**Current:** 14.12 pts, gen:biohack_new, mem 50756, params 2331
**Target tier:** A/B — fixed geometry, separable, no data-dependent crop size; the
only forced full-canvas tensors are the colour plane and the 30x30 output label.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | slice[1,10,11,11]+Mul+ReduceSum colf, gather 9x9 compact, count<5 select, gather upscale | B | 14197 | 310 | 15.42 | — | works, but in11+kin = 9680B |
| 2 | replace slice+Mul with 1x1 Conv on FREE input -> colf30 | B | 8317 | 298 | 15.94 | 200/200 | ADOPT-candidate |

## Best achieved
15.94 @ mem 8317 params 298 — adopted? N (per instructions, not self-adopted).
Beats prior 14.12? Y by +1.82.

## Irreducible-floor analysis
Two ~3600B planes dominate: (a) colf = colour-index plane [1,1,30,30] fp32 (3600B)
from the 1x1 Conv — needed once to read both occupancy and colour; can't be fp16
because ORT Conv requires input/weight type match and the input is fp32-free
(casting the [1,10,30,30] input to fp16 costs 18000B). (b) L = padded uint8
[1,1,30,30] label (3600B) — the output-shaping floor; output is genuinely 30x30
one-hot so the 30x30 sentinel-padded label is unavoidable. Remaining ~1100B are
small (cr [1,1,9,30]=1080B from the two-step row/col gather, plus 484B label
copies). colf+L ~= 7200B is the practical floor for this construction.

## OPEN ANGLES (re-attack backlog)
- Drop colf30 entirely: gather the 9 interior rows on a fp16-cast of the conv
  plane, or contract the channel axis with a MatMul that lands the 11x11 colour
  directly — but every channel-contraction route I see materialises a >=3240B
  10-channel intermediate, so unlikely to beat 3600B. (~+0.1 at most.)
- Shrink cr (1080B) by gathering a flattened 81-index map in one Gather op
  instead of axis2-then-axis3 (~+0.05).
- These are marginal (<0.2); the 7200B colf+L floor caps this task around ~16.2.

## INSIGHT (transferable)
"Odd-one-out by pixel COUNT then upscale" is closed-form tier-B, NOT a detection
wall: the unique cell is `count < k` (here <5) — no ReduceMin/ArgMax needed when
exactly one cell differs. Select-the-cell collapses to `Sum_{R,C} (count<k)*block`,
and the 3x->block upscale reuses the task195 const-index-map Gather idiom (block
content = cellflat[(r//4)*3 + (c//4)], frame via a const Where). Computing the
colour plane with a 1x1 Conv on the FREE fp32 input (3600B) beats Slice+Mul+Sum
(9680B) whenever you need the whole plane anyway. ⭐ count-discriminate + Kronecker
upscale pattern.
