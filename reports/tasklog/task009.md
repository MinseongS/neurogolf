# task009 — 06df4c85

**Rule:** The pixel grid is `common.create_linegrid(bitmap, spacing=2, linecolor)` of an underlying
bitmap of size n in [6,10]: bitmap cell (r,c) -> a 2x2 block at pixel rows/cols {3r,3r+1}x{3c,3c+1};
pixel rows/cols == 2 (mod 3) are gridlines (linecolor). The transform = `connect_bitmap`: for every
pair of SAME-colored bitmap cells sharing a row, fill the cells between them with that color (likewise
columns). Equivalently, per color v: fill the span [min,max] of v's occupied positions along each axis.
Verified that distinct colors' spans NEVER overlap (max coverage 1/cell over 261 instances), and that
lone cells survive (span of one cell = itself), so "start from input + per-color span union" is exact.

**Current:** 14.232 pts, ext:thbdh6285, mem 47400, params 79
**Target tier:** A — separable bbox/span-fill (boolean prefix/suffix-OR via triangular MatMul, task070
idiom) routed into the FREE bool output; not Tier S because output colors copy arbitrary input colors
through a Where/Equal, not a fixed Conv route.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | full-pixel 10-ch Where chain (f32 branches) | A | 117840 | 1156 | 13.31 | — | correct but f32 full planes |
| 2 | bitmap-scale span + upsample, uint8 Where chain | A | 62050 | 1176 | 13.95 | — | 3 full u8 planes |
| 3 | compose at bitmap scale, 1 upsample, 2 u8 Wheres | A | 53750 | 336 | 14.10 | — | separable gridline init |
| 4 | 1-CHANNEL color-index plane -> Equal(arange) bool output | A | 38682 | 337 | 14.43 | — | killed 10-ch full planes |
| 5 | slice ch1..9 only; pref*suf>0 product (one Greater) | A | 34182 | 340 | 14.55 | 200/200 | adopted-as-best |

## Best achieved
14.551 @ mem 34182 params 340 — adopted? N (per instructions). Beats prior 14.232? Y (+0.32).

## Irreducible-floor analysis
Two f32 [1,1,30,30]=3600B planes dominate: (a) the `input[:,1:10,::3,::3]` downsample Slice (Slice
preserves the fp32 input dtype) and (b) `ingrid = ReduceMax(input,[1])` (ReduceMax requires float; it is
the only any-channel off-grid detector and the off-grid bottom/right border is data-dependent so cannot
be statically cropped). The remaining bulk is ~7 fp16 [1,9,10,10]=1800B bitmap-scale planes (4 triangular
MatMuls + 2 span products + add). Everything past the entry slice is fp16/1-channel; the 10-ch expansion
lives only in the FREE Equal output.

## OPEN ANGLES (re-attack backlog)
- Eliminate the f32 `ingrid` ReduceMax (~3600B): derive the active-grid square size as a scalar
  (e.g. max nonzero pixel row via an index-weighted reduction) and build a separable rect mask from
  two [30] vectors; would need to correct the trailing-gridline-at-3n-1 edge (a naive bitmap-occupancy
  upsample misclassifies pixel 3n-1 as in-grid). Est ~+0.07.
- Fewer bitmap-scale MatMul planes: the 4 prefix/suffix MatMuls are the count floor for a 2-axis span;
  no obvious single-MatMul packing found (row side right-multiplies, col side left-multiplies).

## INSIGHT (transferable)
⭐ A "connect same-colored collinear cells" / span-fill task is the per-color generalization of the
task070 bbox-as-mask lever: per channel, in-span = (prefix-sum>0) AND (suffix-sum>0), and since the
two factors are non-negative this is just `prefix*suffix > 0` — ONE Greater instead of 2 Greater+1 And
per direction. ⭐ For a 2x-linegrid rendering (cell -> 2x2 block + 1px gridlines), do ALL work at the
≤10x10 BITMAP scale (downsample via strided Slice ::3), collapse to a SINGLE fp16 color-index plane,
then upsample that 1-channel plane with two Gathers (ridx[r]=r//3), reinsert gridlines via a static
separable gp mask + scalar linecolor index, sentinel off-grid to -1, and route the whole 10-channel
one-hot expansion into the FREE output via `Equal(L_final, arange[1,10,1,1])`. This keeps every full-
30x30 tensor at 1 channel (3600B f32 or 1800B f16) instead of the 9000B+ 10-channel floor.
