# task364 — e509e548

**Rule:** 1+max(H,W)//3 non-overlapping (gap>=1) GREEN boxes on a black canvas (grid H in
[10,20], W=H+[-2,2], so <=20x22). Each box is one of three sprite skeletons in any of 8
dihedral orientations: "el" (L-shape: a full col + a full perpendicular row), "aitch"
(H-shape: full left col + half right col + middle cross-row), "you" (U-shape: two full
parallel cols + one connecting row). Output recolours every green pixel by its box's shape
class: L->1 (blue), H->2 (red), U->6 (pink); background stays 0.
**Current:** 13.75 pts, custom flood (unique-label MaxPool flood + ScatterND histogram of
endpoint/turn counts), mem 75136, params 1402.
**Target tier:** detection (per-component shape classification requires component-level
aggregation = a flood; no per-pixel-local discriminator exists since a straight-arm cell is
locally identical across L/U/H).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | local-degree seeds + 3 MaxPool floods (8 iters) + uint8 Where chain | det | 76748 | 101 | 13.75 | 200/200 | correct, no gain |
| 2 | + Conv neighbour sums, drop ScatterND/int64, uint8 Where chain | det | 62060 | 65 | 13.96 | 200/200 | +0.21 |
| 3 | + per-seed iters (J=6, V/H=8), drop final flood gate, WHERE-PRIORITY drop notH | det | 55900 | 65 | 14.07 | 300/300 | **+0.317 WIN** |

## Best achieved
14.068 @ mem 55900 params 65 — adopted? N (write-only). Beats prior 13.75 by +0.317 (Y).

## Discriminator (verified 0 mismatch over thousands of fresh boxes)
Compute 4-neighbour degree on the green mask (two 3x3 Convs: vert=up+down, horiz=left+right;
deg=(vert+horiz)*mask). Three LOCAL seeds:
- J = deg>=3 (a T-junction) — only H has one.
- vend = deg==1 & vert==1 (lone neighbour vertical); hend = deg==1 & horiz==1.
Flood-MAX each seed through the mask (kernel-3 only — inter-box gap can be 1, so a bigger
kernel leaks across it; re-gate by mask each step). Per-seed iters = measured BFS reach from
that seed type to the farthest box cell: junction reach<=6, endpoint reach<=8. Classify:
- isH = Jf>0
- isU = hasV XOR hasH  (U's two tips point the SAME way -> only one orientation present)
- isL = hasV AND hasH  (L's tips are perpendicular -> both)  [chain default]
H Where applied LAST (priority) so isU needn't exclude H.

## Irreducible-floor analysis
Flood dominates: 36,080 B (J 6 + V 8 + H 8 iters x [pool+gate] fp16, last gate dropped).
Irreducible because (a) the colour is a per-COMPONENT property and a straight-arm cell is
locally indistinguishable across the 3 shapes -> aggregation over the whole component is
mandatory = a flood; (b) kernel must be 3 (1-wide inter-box gap forbids a bigger pool);
(c) measured BFS reach is 8 for endpoints, so 8 iters are necessary; (d) MaxPool requires
float, fp16 (2B) is its dtype floor (no uint8/bool MaxPool in ORT); (e) MAX cannot OR three
independent flags, so 3 separate floods are required (an H-shape shares U's (hasV,hasH)
XOR-signature, so J cannot be dropped). The two fp32 entry slices (green+bg, 3520 B) are the
Slice-dtype floor; bg is needed for the in-grid mask (off-grid-inside-the-20x22 region must
map to the 255 sentinel, not bg=0).

## OPEN ANGLES
- Reduce endpoint flood reach below 8 by seeding interior relay cells — but interior cells
  don't carry the endpoint-orientation flag, so no valid relay exists (explored, dead end).
- Single combined flood for L-vs-U via a MAX-survivable encoding — fails: MAX(1,2) loses the
  lower flag, can't recover both-present (L) vs one-present (U). 2 floods minimum.
- Eliminate the bg fp32 slice by recovering grid H,W as scalars — needs bg occupancy anyway.

## INSIGHT (transferable)
⭐ A "per-component shape classify + recolor" connectivity task is NOT at floor just because
the public net floods unique labels + ScatterND-counts: the discriminating per-component
counts often reduce to a few MAX-floodable BOOLEAN flags computed from LOCAL degree features
(junction = deg>=3; endpoint-orientation = deg==1 & (vert|horiz)==1). Flooding 2-3 booleans
(kernel-3, per-seed iters = measured BFS reach) + a uint8 Where-priority chain beats the
int64-label + ScatterND-histogram net by ~10-20 KB. ⭐ L-vs-U (both 2 endpoints, 0 junctions,
identical local stats) separates ONLY by endpoint ALIGNMENT = (hasVertEndpoint XOR
hasHorizEndpoint): U's tips share an axis (one orientation), L's are perpendicular (both).
⭐ Two free flood cuts: (1) per-seed iteration counts set to the measured BFS reach of each
seed type, not a single worst-case; (2) drop the re-gate on the FINAL flood step — leaked
gap-cell values are non-green and get discarded by the downstream green-gated Where chain
(gate the classify bools by the green mask once instead).
