# task191 — 7df24a62

**Rule:** A blue square (channel-1 frame) encloses a small yellow pattern (tall×wide, tall∈{1,2,3},
wide∈{2,3}, touches all 4 bbox edges, exactly max(tall,wide) yellow cells) on a 23×23 grid littered
with scattered yellow noise dots. For every grid position and every dihedral orientation (4 rot × 2
xpose) where the yellow noise EXACTLY equals the oriented pattern (all pattern-yellows present AND no
extra yellow inside the oriented bbox), draw a blue box = oriented-bbox dilated by 1. Overlay the
yellow dots on top. (The reference sprite reproduces itself.) Generator only emits non-illegal
instances, so boxes never collide with the sprite frame / off-grid.

**Current:** 11.53 pts (public 1037-node net), → **13.77 pts custom**, mem 74258, params 1122
**Target tier:** detection (8-orientation template match) — NOT a multi-object-correspondence BAIL:
the match is a pure binary correlation expressible as a single stacked Conv.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | fp32, two Convs (corr+tot), D5 ConvTranspose | det | 199974 | 1113 | 12.79 | 200/200 | ok |
| 2 | + fp16 working planes | det | 129296 | 1114 | 13.22 | 200/200 | ok |
| 3 | + crop conv canvas to 23×23 grid | det | 94864 | 1128 | 13.53 | 200/200 | ok |
| 4 | + stamp mask3 (3×3) then single 3×3 dilate | det | 93394 | 1120 | 13.54 | 200/200 | ok |
| 5 | + COMBINED match kernel (fold tot into corr) | det | 74258 | 1122 | 13.77 | 500/500 | **adopt-candidate** |

## Best achieved
13.77 @ mem 74258 params 1122 — beats prior 11.53 by **+2.24** (≥+0.3 ✓). fresh 500/500 (incl off-grid).

## Irreducible-floor analysis
Two 8-channel fp16 full-canvas planes dominate: `corr` Conv output [1,8,25,25]=10KB and the
`placed` ConvTranspose [1,8,27,27]=11KB. Both are intrinsic to running 8 orientations as stacked
Conv channels — the 8 dihedral kernels are data-dependent (built from the runtime pattern) so they
cannot be pre-reduced at build time even when many coincide for symmetric patterns. Plus ~3 fp32
[1,1,30,30] planes (B slice, Y slice, ingrid = 3.6KB each) for the output assembly.

## OPEN ANGLES (further compaction, not needed for the win)
- Fold the per-orientation ReduceMax(placed) earlier or stamp into ONE channel via summed
  ConvTranspose to drop the 8-ch `placed` plane (~11KB → ~1.5KB).
- fp16 the output-assembly fp32 planes (B/Y/ingrid 30×30) — compute ingrid on the 23×23 crop.
- Drop PAD top/left if a re-derivation shows edge anchors never go negative (would shrink the
  conv canvas 27→25).

## INSIGHT (transferable)
⭐ **8-orientation dihedral template matching is NOT a shape-correspondence BAIL** — it is a stacked
Conv: extract the small pattern as a 3×3, build the 8 oriented kernels as FIXED gather-permutations
of the 9 flattened elements (rot90/T = constant index maps PERMS), and run all 8 as the
output-channels of ONE Conv weight [8,1,3,3].
⭐ **Fold a two-predicate window match into ONE correlation kernel via signed weights:** to test
"all K pattern cells present AND no extra inside the bbox" in a single Conv, use
`combk = Ko*(1+B) - B*mask3` (pattern → 1+B, extra-in-bbox → −B); `Conv==npat` is exact (fp16-safe
for B=100, sums < 2048). Removes the separate "total-in-bbox" Conv plane.
⭐ **A data-dependent small-window readout (Gather a 3×3 at a runtime bbox corner) can over-read** —
the 3×3 frame exceeds a smaller tall×wide sprite and silently captures adjacent noise; mask
rows≥tall / cols≥wide (derived as scalars from the bbox extent) before using it.
⭐ ConvTranspose(M, stamp, group=C) is the clean "scatter a fixed stamp at every firing anchor"
primitive; reduce over channels then a single MaxPool dilates the union.
