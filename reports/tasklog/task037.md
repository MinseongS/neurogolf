# task037 — 1f876c06

**Rule:** 3..6 diagonal (45°) segments on a fixed 10x10 grid, each a DISTINCT colour, length 3..7,
direction cdiff=±1. The INPUT shows only the TWO endpoints of each segment (both painted that colour);
the OUTPUT fills the whole diagonal between them. Segments never share a cell (generator bitmap guard).
**Current:** 14.12 pts, gen:biohack_new, mem 45000, params 8326
**Target tier:** B (label-map + Equal). NOT a flood-fill/connectivity wall — the fill is a bounded,
direction-separable diagonal prefix∧suffix per colour channel, closed-form.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | per-ch doubling SHIFT-OR via Pad+Slice | B | 126120 | 232 | 13.25 | - | Pad grows canvas, huge mem |
| 2 | SHIFT-OR via [10,10] matmul shifts | B | 76800 | 637 | 13.74 | - | ~40 [1,9,10,10] intermediates |
| 3 | flattened [100,100] reachability matmul, 2 dirs | B | 23900 | 20044 | 14.31 | - | params dominate |
| 4 | one [100,100] + col-flip reuse | B | 29200 | 10048 | 14.42 | - | still 10k params |
| 5 | diag-coord [19,10] Gather + [10,10] triangulars | B | 71900 | 738 | 13.81 | - | padded layout balloons mem |
| 6 | shared 7x7 diag Conv, batch-9, flips for 4 dirs | B | 39200 | 102 | 14.42 | - | flips add ~14 tensors |
| 7 | **2 kernels (main/anti) x 4 pad-sides, no flips** | B | 25700 | 146 | **14.84** | 200/200 | **ADOPTED candidate** |

## Best achieved
14.84 @ mem 25700 params 146 — adopted? candidate only. Beats prior 14.12 by **+0.72**? YES. fresh 200/200.

## Irreducible-floor analysis
Memory now bound by the four fp16 7x7-Conv outputs ([9,1,10,10]=1800B each) + the fp32 channel slice
(3600B). The four directional aggregates must coexist transiently before the two ANDs. Could shave ~1.8KB
by Min-ing conv pairs before thresholding, or by group=9 depthwise conv on [1,9,10,10] (mem 23900 but
params jump to 926 → 14.88, a wash). The 900B padded-L + Equal output path is already the cheapest 10-ch
route. ~14.8–14.9 is the practical floor for this content-dependent diagonal fill.

## OPEN ANGLES (re-attack backlog)
- Min(cv_ul,cv_dr)>0 instead of two Greaters+And: removes 2 bool planes (~1.8KB) → maybe ~14.95.
- Pack the 4 directional convs into ONE conv with 4 output channels (single Conv node, kernel [4,1,7,7])
  then split — fewer node intermediates, possibly lower traced mem.

## INSIGHT (transferable)
⭐ "Connect/fill between two same-colour endpoints along a 45° line" is NOT a connectivity bail: it is a
direction-separable per-channel diagonal prefix-OR ∧ suffix-OR. When the generator BOUNDS the segment
length (here ≤7, so endpoint distance ≤6), the unbounded prefix/suffix-OR collapses to a single bounded
**KxK diagonal Conv + >0 threshold** — far cheaper than doubling-shift chains (Pad grows the canvas) or a
flattened [100,100] reachability matmul (10k params). Reshape the colour channels onto the BATCH axis so
ONE small kernel serves all 9; reuse each diagonal kernel for both opposite directions by swapping the
asymmetric SAME-pad side (no axis flips). Disjoint colour segments ⇒ a [1,9] colour-weight MatMul collapses
the per-channel fill to a single colour-index plane with no [1,10,H,W] intermediate.
