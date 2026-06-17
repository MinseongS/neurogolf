# task234 — 98cf29f8 ("frog eats fly")

**Rule:** Two solid rectangles on a 0 background: the FROG (colour c0) and the
FLY (colour c1), plus a 1-wide LINE (the "tongue", colour c1) joining the fly to
the frog along one axis. The whole figure is then optionally vertical-flipped
and/or transposed (applied identically to input and output → orientation-
EQUIVARIANT). Output = frog kept fixed, tongue deleted, and the fly box slid
along the tongue axis until it is flush against the frog. Grid is embedded in a
30×30 canvas (true grid 12-20 × 15-20); off-grid cells are all-zero.
**Current:** 15.21 pts, custom:task234, mem 17698, params 97 (prior version of
this same file — per-channel spatial products).
**Target tier:** B (separable row/col label map → free Equal). Not S: the fly
translation is a data-dependent shift (tongue length), non-local. The output is
three separable rectangles (frog box, moved fly box, in-grid bg + off-grid
sentinel) so a Tier-B row/col label map is the highest admissible tier.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | (prior) per-channel R/C·isFrog/isFly products (6× [1,10,30,*] fp16) + label map | B | 17698 | 97 | 15.21 | n/a | committed baseline = P |
| 2 | recover frog/fly CHANNEL INDEX (== colour) as runtime int32 [1] scalar; **Gather** that single channel out of R / C / per-channel sums → [1,1,30,*] 60B slices instead of [1,10,30,*] 600B products | B | **13085** | 98 | **15.513** | 200/200 | WIN (+0.300) |

## Best achieved
**15.513 @ mem 13085 params 98 — stored 266/266, fresh 200/200.** Beats prior
15.21 by **+0.300** (exactly the ≥+0.3 bar). Adopted? **N** (main adopts via
`python -m src.adopt 234`).

## Irreducible-floor analysis (after attempt 2)
mem 13085 is spread, no single dominant plane:
- **2400 B** = two fp32 per-channel sums `rowSumChf [1,10,30,1]` + `colSumChf
  [1,10,1,30]` (1200 each). IRREDUCIBLE: ORT ReduceSum on the fp32 one-hot input
  emits fp32; both axes are needed — row+col occupancy of BOTH colours (for the
  frog/fly bbox-area solidity test) AND the fly's per-line counts (thickness =
  min-positive line count, which excludes the tongue). Casting the input to fp16
  first = 18000 B (worse).
- **5400 B** = label map: 3 bool box planes `[1,1,30,30]` (frogBox/flyBox/gridBox,
  900 each) + 3 uint8 Where outputs (Lg/Lf/L, 900 each). This is the floor for a
  "3 separable rectangles + off-grid sentinel → free Equal" output: the final
  `Equal(L, chan)` lands in the FREE bool output, but the single uint8 label plane
  it reads must distinguish 0(in-grid bg)/colour/10(off-grid). bool `[1,1,30,30]`
  (900) is already the cheapest full-canvas dtype. OR-ing colour planes into the
  output instead would materialise multiple non-free [1,10,30,30] = 9000 B each.
- **1200 B** = R, C fp16 `[1,10,30,1]`/`[1,10,1,30]` (600 each) — per-channel
  occupancy, needed for span/bbox and as the Gather source.
- **~3000 B** = the run-mask + edge scalar machinery (≈35 tiny 60 B `[1,1,30,1]`
  vectors + scalars).

## OPEN ANGLES (re-attack backlog)
- **Drop one fp32 per-channel sum (≈1200 B → ~+0.09).** The frog/fly solidity
  test needs bbox area (both row+col spans); thickness needs one axis of raw
  counts. If frog/fly could be discriminated WITHOUT bbox area (e.g. a tongue
  detector: the fly is the colour with a 1-wide protrusion), only one per-channel
  sum axis would be needed. Not yet found a cheap orientation-agnostic tongue
  signal.
- **Shrink the label map below 5400 B.** Folding the off-grid sentinel away by
  ANDing the in-grid mask into the final output forces a non-free [1,10,30,30]
  Equal intermediate (9000 B) — net worse. No path found below the 3-rect floor.
- Trim the ~35 tiny run-mask vectors by computing the moved-axis run as a single
  banded predicate — sub-200 B, marginal.

## INSIGHT (transferable)
⭐ **When exactly K colours/objects each occupy their own one-hot CHANNEL and the
channel index equals the colour, recover that index as a runtime int32 `[1]`
scalar (`Reshape(ReduceSum(sel·arange),[1])→Cast int32`) and `Gather(plane, idx,
axis=1)` to pull the single channel out as a [1,1,…] slice — instead of
`Mul(plane, sel[1,10,1,1])` which materialises a full [1,10,30,*] product plane.**
Here it replaced six 600 B per-channel products (RFrog/RFly/CFrog/CFly +
flyRowSum/flyColSum muls) with 60 B gathered slices, −3.6 KB → +0.30 pts, no
correctness change. A `[1]`-shaped (not scalar `[]`) Gather index keeps the
channel axis as dim-1 so downstream broadcasts stay aligned, and a runtime Gather
index keeps shapes STATIC (unlike a runtime Slice, which leaves symbolic dims →
mem unmeasurable).
