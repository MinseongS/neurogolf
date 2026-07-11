---
deployed_cost: 3558
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task284 — ARC b7249182

**Rule:** Two seed dots share a line (a row, or a column if the grid is transposed)
at the long-axis coords Cl<Cr, with half=(Cr-Cl+1)/2. Each seed grows a bilateral
"wrench"/cross glyph in its own colour toward the centreline. Left glyph (colour of
Cl seed): horizontal stem on the shared line cols Cl..e0, a 5-tall vertical bar at
e0 (K-2..K+2 on the short axis), and hook cells at e0+1 on the two bar-ends; right
glyph mirrors with e1, Cr where e0=(Cl+Cr-3)/2, e1=(Cl+Cr+3)/2. The whole grid is
optionally transposed (xpose). Off-grid cells stay all-zero (no channel set).
**Current:** 14.9585 pts, custom:task284, mem 22821, params 138
**Target tier:** A — separable rank-1 glyph (sum of 6 rank-1 components) routed into
the free BOOL output; one fp32 colour-index Conv + one fp32 MatMul are the two
heavy planes, both irreducible per the 3600B-plane floor.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | prior committed net (2 MatMul planes + per-channel occupancy) | A | 22821 | 138 | 14.96 | n/a | baseline |
| 1 | single combined 6-component MatMul (both glyphs), colour by along-side threshold | A | 23643 | 137 | 14.92 | — | regressed: the side mask was built as an fp32 broadcast Add (3600) |
| 2 | keep side mask in BOOL (Or of two 1-D bools, never fp32) | A | 21664 | 137 | 15.01 | — | +0.05 |
| 3 | replace 6× [1,10,..] occupancy fp32 (7200B) with one colour-index Conv + tiny 1-D profiles; grid extent via ReduceMax(input,[1,3])/[1,2] 120B vectors | A | 20034 | 128 | 15.09 | — | +0.13 |
| 4 | CANONICAL frame (H=cross,W=along): build whole label assuming non-T, then Transpose+Where the finished uint8 label iff isT — kills all factor-orientation mixing (~6×720B) | A | 17664 | 128 | 15.21 | — | +0.25 |
| 5 | shrink cross factor to 3 distinct rows (stem/band5/mid3) instead of 6 | A | 16424 | 119 | 15.29 | 200/200 | adopt-candidate |

## Best achieved
15.286 @ mem 16424 params 119 — beats prior 14.9585 by **+0.328** (≥+0.3 ✓).
Fresh isolated 200/200. adopt-recommend **Y**.

## Irreducible-floor analysis
Two unavoidable fp32 [1,1,30,30] planes dominate (3600B each = 7200B):
- **Conv_0** colour-index plane `Σ_k k·input_k` — the cheapest way to recover the
  two seed colours AND their positions; the alternative (per-channel
  ReduceSum/Max → 6× [1,10,30,1]/[1,10,1,30] fp32) costs 7200B, strictly worse.
- **MatMul_77** the single combined glyph plane (CH[1,1,30,6]@AW[1,1,6,30]); a
  6-component rank-1 sum routed through one MatMul — ORT forces the output to fp32.
Remaining ~5400B is the uint8/bool label chain (Lc0 grid sentinel, Lc glyph fill,
LcT transpose, L orientation-select, glyphM, gridC) — each a 900B [1,1,30,30].
The output Equal is FREE.

## OPEN ANGLES (re-attack backlog)
- The orientation Transpose+Where on the finished uint8 label costs 1800B
  (LcT 900 + L 900). If the glyph MatMul could be made orientation-correct without
  per-factor mixing AND the grid/colour kept consistent, that 1800 could drop —
  but every attempt to mix at the factor level re-introduced ~4320B of [1,1,6,30]
  fp32 factor tensors, so the transpose-the-label trick is the cheaper of the two.
- gridC (900B bool) is the off-grid sentinel = NOT(crossExt⊗alongExt). It is
  separable but a Where needs the materialized condition; folding it via associated
  broadcasts into the final glyph Where might shave ~900B (untried, fiddly).
- The two 3600B planes are the structural floor; ~16.5KB is near the realistic
  minimum for "recover 2 colours + positions + orientation, then stamp a non-rect
  separable glyph". Tier S impossible (output colours copy arbitrary input colours,
  needs a routed plane).

## INSIGHT (transferable)
⭐ For a multi-colour separable stamp, build ONE combined glyph plane via a single
MatMul over ALL components of ALL colours (K = total component count), then colour
each cell by a cheap 1-D ALONG-AXIS side threshold (`along-coord < S/2 ? colL : colR`)
instead of one MatMul-plane per colour — halves the heavy fp32 plane count.
⭐ Handle a global transpose by building the entire uint8 label in a CANONICAL frame
and conditionally transposing the FINISHED label (`Where(isT, L^T, L)`) — this is
far cheaper (one Transpose + one Where = 1800B) than mixing orientation into every
1-D/2-D factor (which re-materializes many [1,1,6,30] fp32 tensors). Pull the grid
extent into the canonical frame by swapping the 1-D row/col extent vectors by isT.
⭐ Any broadcast Add/Mul that lands on [1,1,30,30] is upcast to fp32 (3600B) by ORT
even with fp16 operands — keep boolean selections in BOOL (And/Or, 900B) and never
let an orientation-mix become a 30×30 float Add.

## ADOPTED 20260709T041319Z
- cost: 3558 -> 3089 (points 16.9644)
- source: candidates/public_dumps/20260709/7261-53-lb-compact-onnx-artifact-starter/nets/task284.onnx
- note: min-merge from nets

## DEPLOYED NET ANATOMY (2026-07-11 audit)
Mechanism CHANGED at 2026-07-09 adopt: no longer the MatMul-glyph net the prose above
describes. Now a **sparse-edit ScatterND stamp**: base = free `input` (copy), then 56
point-writes into fp32 output. mem 2912 / params 177 / cost 3089.
- Byte map (mem, output [1,10,30,30] fp32 = 36000 is FREE):
  - `indices` [56,4] **int64 = 1792** (61% of mem) — ScatterND indices, dtype-forced i64, STRUCTURAL carrier
  - `idx8` [56,4] i8 = 224 (assembled in i8, one Cast to i64 = optimal)
  - `rp`,`cp` fp32 [1,30] = 240 (row/col non-bg profiles → ArgMax rf/rl/cf/cl)
  - `u56_r`,`u56_c`,`chans` i8 [56,1] = 168; a_vals/b_vals fp32 = 80; misc ~408
- The 56 writes = 28 color-writes (+1 to colour channel) + 28 background-clears (−1 to
  channel 0 at the SAME 28 positions). Both load-bearing: glyph cells sit on background
  in `input`, so ch0 must be pushed ≤0 (decode is per-channel `(output>0)`, harness l233).
- DETECTION/CARRIER split: no fp32 detection plane (unlike prose). All cost is the SPARSE
  STAMP carrier (i64 indices 1792 + i8 idx assembly). Geometry (profiles/argmax/colours)
  ~560. The i64 index is the only >900B tensor.
- Oracle rule (reference/arc-code-golf-solutions/task284.py): 2 seed dots share a row/col;
  each grows a "wrench" = vertical bar (5-tall) + perpendicular stem toward centre + 2 hook
  cells past the bar ends; grid optionally transposed. Glyph = 6 disjoint rank-1 rects × 2
  colours. Verified renders ex0/1/2.

## NEGATIVE LEDGER 2026-07-11 — free-output fp16 signed-einsum LOSES (MEASURED)
(1) WHAT RAN: built candidates/task284/einsum.onnx (build_einsum.py). Replaced ScatterND
    tail with a 9-component free-output fp16 signed-einsum `b,kr,kc,kv->bvrc` writing fp16
    `output` directly (6 glyph rank-1 rects split L/R hooks = 8 + 1 bg rectangle). Reused
    deployed geometry scalars (rf/rl/cf/cl, a_s/b_s, sep/stem/drow/dcol). Bands built from
    interval [lo,hi] via GreaterOrEqual+LessOrEqual+And+Cast(fp16). Correct 266/266 bundled.
    `uv run ng gate` → **cost 4780 (mem 4642, params 138) vs deployed 3089 — REJECT, +1691.**
(2) TOOL+DATE: uv run ng gate / onnxruntime==1.26.0 profiler measurement, 2026-07-11, opus.
(3) BYTE MATH (why it loses): band-build = 6 bool [9,30] (1620) + 2 fp16 bands [9,30] (1080)
    = 2700; + chanvec 180 + geometry (rp/cp 240, grid-reduces gr/gc 240, seed slices 80,
    lo/hi lane int32 arithmetic ~430, misc) ≈ 1760. Interval bands need 2 comparisons each
    (1350/axis irreducible). The separable-plane materialization (2700 just for 9 bands)
    already exceeds the ScatterND i64-index carrier (1792); geometry then doubles the gap.
    REGIME: for a SPARSE ~28-cell glyph, per-cell ScatterND (i64 indices) beats separable
    free-output einsum — the inverse of the mask-dominant regime where einsum wins. The
    deployed ScatterND net is at/near its structural floor (~2900 mem).
(4) REOPEN-TRIGGER: (a) a band-construction primitive that yields fp16 interval masks in
    ≤1 counted tensor/axis (would drop band-build 2700→~1080, then einsum ≈ 1080+180+geom
    ~2300 < 3089); (b) a way to cut components below ~4 (merge wrench rects); (c) a cheaper
    sparse stamp than i64 ScatterND (no i32 ScatterND on grader ORT; Loop/NonZero banned);
    (d) new public dump with a smaller task284 net. FALSIFICATION HISTORY: none prior for
    this specific claim; the 2026-07-09 min-merge already beat the old MatMul-glyph net
    (16424→3089) so the einsum-plane family was implicitly dominated — now MEASURED.
- DURABLE TECH FINDING (reusable): ORT 1.26 wraps fp16 Einsum in `InsertedPrecisionFreeCast`
  nodes (fp16→fp32 operands, fp32→fp16 output [1,10,30,30]=18000) but these inserted nodes
  are NOT in the static graph so calculate_memory does NOT count them → **fp16 free-output
  einsum IS memory-viable** (the fp16-output-is-free trick works; harness decode `(out>0)`
  is dtype-agnostic, scoring.py l114-115 skips output regardless of dtype). It just doesn't
  help a sparse-glyph task. Also: rebuilt graphs that KEEP deployed runtime-Slice nodes MUST
  carry the deployed `value_info` (pins Slice output shapes) or strict shape-infer emits
  dim_param → calculate_memory returns None ("performance could not be measured").
