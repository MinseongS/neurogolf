# task009 — 06df4c85

**Rule:** A `create_linegrid(bitmap, spacing=2, linecolor)` rendering of a size-n bitmap
(n∈[6,10]): each bitmap cell → a 2×2 colour block, with 1-px `linecolor` gridlines at
pixel rows/cols ≡ 2 (mod 3). Transform = `connect_bitmap`: for every pair of SAME-coloured
bitmap cells sharing a row, fill the cells between them (inclusive) with that colour;
likewise for columns. (Distinct colours' spans never overlap — the generator only fills
FREE spans, max coverage 1 colour/cell.)
**Current:** 15.86 pts, ext:kojimar7113 (crowd net), mem 8249, params 1062
**Target tier:** B — closed-form span-fill + fixed linegrid re-render; no detection wall.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | (leftover src) per-channel triangular-MatMul span-fill, f16 | B | 34182 | 340 | 14.55 | — | superseded (huge) |
| 2 | kojimar Einsum span-fill, cast colors_sample→f16 | B | 9649 | 1060 | 15.72 | — | WORSE (cast ADDS a plane: 3600 f32 + 1800 f16) |
| 3 | strided-1×1-Conv colour-index + directional nearest-marker span-fill + DepthToSpace render | B | 8849 | 96 | 15.90 | — | works; many f16 carry planes |
| 4 | + in-grid = strided ones-Conv>0; drop >0 fill-gate | B | 8049 | 96 | 16.00 | — | trim |
| 5 | + bg→NEGBIG mask shared (drop per-pack Where) | B | 7449 | 97 | 16.07 | — | trim |
| 6 | + ch0 conv weight 0.5 → valid/dot from ONE conv (drop occ conv) | B | 7049 | 86 | 16.13 | 200/200 | **adopted** |

## Best achieved
16.127 @ mem 7049 params 86 — adopted? Y (src/custom/task009.py). Beats prior 15.86
by +0.27 (just under the +0.3 bar). Fresh 200/200 (isolated, generator-by-path).

## Irreducible-floor analysis
⭐ The public kojimar net's dominant plane is `colors_sample` = `input[:,1:10,::3,::3]`, a
[1,9,10,10] **f32 = 3600B** slice (Slice inherits the f32 input dtype; an explicit Cast to
f16 only ADDS a 1800B plane → net worse, attempt 2). The win is to NEVER materialise a
9-channel plane: a **STRIDED 1×1 Conv** `Conv(input, W[1,10,1,1], strides=3)` collapses the
10 colour channels AND sub-samples to the 10×10 bitmap in ONE op → a single [1,1,10,10]
colour-INDEX plane (400B f32). After that the dominant costs are: 2×900B u8 (DepthToSpace
input `scalar_blocks` [1,9,10,10] and output `color_grid` [1,1,30,30]) — both structural to
the 3×3 super-cell re-render — and ~14 f16 [1,1,10,10] carry planes (200B each) from the
directional span-fill. Remaining ~150B over the +0.3 bar; the carry arithmetic (pack /
prefix-MaxPool / suffix-MaxPool / 2× Mod-decode / Equal / Where, per axis) is plane-minimal
given that nearest-marker fill needs BOTH a value-decode (`lval`, fill colour) and an
equality (needs `rval`); all must be f16 (Add/MaxPool/Mod reject uint8).

## OPEN ANGLES (re-attack backlog)
- Gather-upscale render to drop `scalar_blocks` (900): blocked — correct gridline/off-grid
  boundary needs right/bottom SEPARATOR validity (cell i+1 / j+1), which an upscale by
  p//3 gets wrong at the trailing edge (row 3n-1 is off-grid but maps to valid cell n-1);
  fixing it costs ≥1 extra 30×30 plane, net-negative. DepthToSpace separator-tail trick is
  the cheaper correct render.
- Drop one f16 carry plane to cross +0.3: every reduction tried (diff-mod equality, slice-
  reverse shared pack, 2-channel conv for bg-mask) ADDS ≥1 plane. The bg-AND-off-grid mask
  (`Lbm_dot = Where(Lbm>0.75, Lbm, NEGBIG)`) is load-bearing — off-grid cells are 0 (not
  NEG) from the conv, so a conv-baked negative bg weight cannot replace the Where (off-grid
  would still pollute the suffix-max and kill right-edge spans).

## INSIGHT (transferable)
⭐ **STRIDED 1×1 Conv = collapse-channels + subsample in ONE op, killing the 9-channel f32
plane.** For any "downsample a one-hot grid to a coarse colour-INDEX bitmap" step, prefer
`Conv(input, W[1,C,1,1], strides=s)` over Slice-then-ReduceSum: the Slice inherits f32 and
materialises a [1,C,h,w] plane (3600B here), whereas the strided Conv emits [1,1,h,w]
directly (400B). Combine with the task004 fractional-ch0 lever (bg weight 0.5) to fold the
in-grid/dot masks into the SAME conv.
⭐ **Directional nearest-marker span-fill on a 1-channel index** (no per-colour channels):
when same-colour fills provably never interleave, "fill between two same-colour markers" =
pack `pos*16+colour` at markers (bg→NEGBIG so it loses), prefix-MaxPool = nearest-left pack,
suffix-MaxPool of the REVERSED ramp = nearest-right pack, fill iff decoded `lval==rval`.
Replaces the public Einsum's persistent 9-channel plane with a stack of 200B f16 planes.
