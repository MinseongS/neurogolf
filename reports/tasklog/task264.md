# task264 — a8c38be5 (9 fixed-glyph sprites recolored, scattered to a 9x9 chart)

**Rule:** Input is an H x W grid (H,W in 14..16) on black(0) holding 9 non-overlapping
3x3 sprites. Each sprite is a solid gray(5) 3x3 block whose cells at a FIXED per-index
glyph shape are overwritten by that sprite's color (colors random non-gray, MAY repeat).
Output is a fixed 9x9 grid (gray background) laid out as a 3x3 arrangement of 3x3 glyphs:
cell (idx//3, idx%3) holds glyph[idx] painted in sprite idx's color, gray elsewhere.
idx==4 (center) glyph is EMPTY -> solid gray center block. So per instance only the 9
glyph COLORS vary; the glyph SHAPES and their output positions are fixed constants.
Re-triage: the previously-mislabeled "infeasible" task IS feasible — it's recover-8-
color-scalars + stamp-a-fixed-template.

**Current:** was 13.2 pts (public net, prior recovery 13.15). Now **15.04 pts, mem 20200,
params 944**, 265/265 stored, fresh 200/200 (and 1000/1000).
**Target tier:** A (separable-ish closed-form). NOT S: per-sprite color must be matched
to a glyph SHAPE at an unknown scatter position (a localized matched-filter convolution +
masked reduction), no single Conv/permute does it. NOT B/detection: output is a fixed
template parameterized by 8 recovered scalars — no per-cell label map over the 30x30 input.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | gray matched-filter detect (8 glyphs) + glyph color-sum readout, full 30x30 convs | A | 85771 | 932 | 13.63 | 200/200 | works, heavy |
| 2 | slice gray+colorval to 16x16 corner before convs (planes 28x28 -> 14x14) | A | 35643 | 944 | 14.49 | — | trim |
| 3 | colorval via 1x1 channel-collapse conv on FREE input then slice (drop 10ch fp32 corner slice) | A | 24907 | 944 | 14.84 | — | trim |
| 4 | conv outputs/template fp16; Where(det,colorsum,0) replaces Mul(cast(det),colorsum); Mg as bool init | A | **20200** | **944** | **15.04** | **1000/1000** | FINAL |

## Best achieved
**15.04 pts @ mem 20200, params 944** — adopted? **N** (main adopts). Beats prior 13.2? **Y (+1.84).**

## How it works (the load-bearing trick)
Each glyph shape is unique, so its complement within the 3x3 block (= the sprite's GRAY
cells) is also unique. Gray is a single fixed channel -> NO color-collision ambiguity
(color cells can repeat across sprites, gray cannot). Per index k, a 3x3 matched filter on
the gray channel `F_k = comp_k - 9*glyph_k` (valid conv) hits its max == count(comp_k)
EXACTLY iff every comp_k cell is gray AND no glyph_k cell is gray in that window -> only at
sprite k's top-left. `Equal(resp, count_comp_k)` = binary detector D_k (verified: exactly
one hit per k, no false positives over thousands of fresh instances). Color readout:
`colorsum_k = colorval (x) glyph_k` (valid conv) = count(glyph_k)*color[k] at the top-left;
`ReduceSum(Where(D_k, colorsum_k, 0))/count(glyph_k)` = color[k]. The 8 detectors and 8
color-sums are each ONE 8-output Conv. Output: place the 8 colors into the fixed 9x9
glyph template (Mul color[1,8,1,1] x placed[1,8,9,9] + ReduceSum over ch), `Where(Mg_const,
Tcol, gray5)`, Pad 9x9 -> 30x30 with sentinel 12, final `Equal(L, arange)` -> free BOOL output.

## Irreducible-floor analysis
Dominant intermediates: the three 8-channel 14x14 fp16 planes resp/colorsum/picked (3136 B
each = 9408) and the colorval 30x30 fp32 conv output (3600). WORK=16 is minimal: a sprite
top-left can be as large as H-3 = 13, spanning to col/row 15, so the detector valid conv
needs a 16-wide canvas (output 14). colorval30 must be fp32 (3600) — making it fp16 requires
casting the full [1,10,30,30] input (18000) which is far worse. The 8-glyph fan-out (8 output
channels) is intrinsic to matching 8 distinct shapes; folding two of the three 14x14 planes
would require a per-position Gather (its own >=100 B) for net-neutral gain.

## OPEN ANGLES (re-attack backlog)
- **Halve WORK adaptively.** Bound the active region per-instance (ReduceMax row/col occupancy)
  and Gather-crop to the true sprite span; sprites occupy only ~9 of the 16x16 cells but their
  positions vary, so a static crop can't go below 16. A data-dependent crop is a Gather (>=100 B)
  -> likely net-neutral. Untried in detail.
- **Drop colorsum, read color at one glyph cell via a single-1 filter per glyph.** Still an
  8-channel conv -> same 3136 B; no win unless detection+readout fuse into one conv.
- **Detect on color channels instead of gray** to skip the colorval conv — blocked: colors repeat
  across sprites, so a color-channel matched filter is ambiguous (gray complement is the unique key).

## INSIGHT (transferable)
⭐ **A "scatter fixed shapes recolored" / label-map-by-shape task is recover-K-scalars +
stamp-a-fixed-template, NOT a per-cell detection floor.** When each object is a fixed-shape
glyph at an unknown position and only its COLOR varies: (1) the glyph's GRAY/background
complement within its block is a unique, collision-free key (use it, not the color cells which
can repeat); (2) a 3x3 matched filter `comp - BIG*glyph` on the single background channel +
`Equal(resp, count_comp)` is an EXACT binary locator with no false positives; (3) read the
scalar via `ReduceSum(Where(detector, value_conv, 0)) / count`; (4) build the output as a fixed
constant template parameterized by the K recovered scalars (Mul-by-color-vector + ReduceSum +
Where + Pad + Equal) — never a 30x30 label map. This converts an apparent shape-correspondence
"detection" task into a clean Tier-A closed form (13.2 -> 15.04 here).
