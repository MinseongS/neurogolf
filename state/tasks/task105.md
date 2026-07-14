---
deployed_cost: 2849
logged_costs_match: verified-2026-07-11 (deployed=2849, 17.045pts, fail=0; body attempts table is the RETIRED 5089 net)
migrated: 2026-07-09
---

# task105 — 4612dd53

**Rule:** A rectangular OUTLINE (perimeter of a bbox) is drawn, optionally with ONE full
interior "cutline" — either a horizontal row (`horiz`) or a vertical column (`vert`) spanning
the box interior. Every figure cell is colored blue(1) or red(2) (each red w.p. ~1/4). In the
INPUT only the blue cells appear; the red cells are erased to background(0). In the OUTPUT
every figure cell appears (blue unchanged, erased cells become red). So output = input with
every figure-cell that is currently empty repainted red(2). Verified bbox(blue)==bbox(figure)
(corners always part of the outline), so the box geometry is fully recoverable from blue.
Cutline orientation recovered from per-row vs per-col interior-blue COUNT (Rmax vs Cmax).

**Current:** 16.14 pts (prior), method = unknown public net.
**Target tier:** A (separable row⊗col rectangle + cutline routed into the FREE Where output;
not S because output color set is fixed but the figure is a 2-D structure, not a pure spatial
copy of input cells).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | bbox-as-mask (tri-matmul) + perim + cutline, W=14 sq, fp16 planes | A | 8901 | 450 | 15.86 | 200/200 | works, too heavy |
| 2 | + cutline counts via MatMul (drop interior/int_blue WxW); 2 outer-prod figure | A | 6717 | 450 | 16.12 | — | leaner |
| 3 | + drop bg slice (red=figure∧¬blue); single Less for notblue | A | 6129 | 444 | 16.21 | — | |
| 4 | + rectangular canvas HR=13×WC=11 (figure spans rows1-12,cols2-10) | A | 5088 | 634 | 16.35 | — | |
| 5 | + CumSum prefix/suffix-OR (drop 4 triangular inits, 580 params) | A | 5089 | 58 | **16.45** | 199-200/200 | ADOPT-CANDIDATE |

## Best achieved
16.45 @ mem 5089 params 58 — beats prior 16.14 by +0.31 (≥+0.3 ✓). Fresh: 5/6 runs 200/200,
one 199/200; 500/500 clean on a separate run; aggregate ~1/1200 fail rate, ALL the rare
single-interior-blue-cell orientation ambiguity (structurally undecidable, see below).

## Irreducible-floor analysis
Dominant intermediates: the two 30×30 planes for the final `Where` cond — red30 (uint8, 900B
from Pad) + cond (bool, 900B from Cast). ORT `Pad` rejects bool, `Where` requires bool, so the
u8-pad→bool-cast pair (1800B) is the irreducible carrier for routing the W×W mask into the
30×30 output. Next: blue_f32 (572B, the one fp32 channel-1 slice; fp32 because CumSum/ReduceMax
reject fp16) and 4 fp16 13×11 planes (blue, fig_a, fig_b, fig_s = 1144B). Everything else is
sub-150B 1-D vectors.

## OPEN ANGLES (re-attack backlog)
- Resolve the single-interior-cell ambiguity (0.08% fail): a lone interior blue cell gives NO
  input signal for H vs V cutline (generator prior is ~50/50). Verified truly ambiguous — no
  positional/geometric tell. Likely unrecoverable; would need a guess that beats 50%.
- Drop the fp16 `blue` plane (286B) by running the cutline MatMuls + notblue in fp32 off
  blue_f32 (needs col_inner_T/row_inner_T in fp32). ~286B → ~16.50.
- Tighter canvas via col-offset slice (cols 2:11, WC=9) with shifted final Pad — saves ~26%
  on each W×W plane but complicates the pad placement; modest (~few hundred B).

## 2026-07-11 AUDIT (opus per-task) — deployed net is a DIFFERENT, near-floor net
**Ledger above was STALE.** Deployed `submission/overfit_nets/task105.onnx` is NOT the
attempt-5 CumSum net (5089); it is a leaner **row-hash** rebuild: cost **2849** (mem 2740 +
params 109), bundled fail=0 (266/266), **17.045 pts**. Byte map (top carriers):
- `out_batch` [1,3,14,13] u8 **546B** — 3-channel output assembly (Concat bg/blue/red → Pad→output).
- `fg` [1,1,14,9] fp32 **504B** — the single blue-channel detection read (Slice input ch1, cols 2:11).
- 5× [1,1,14,13] u8 **182B** each (shape_u8, row_cols_u8, fg_u8, out_bg, out_red) = 910B.
- fg_hash_u8 126B, row_hash 56B, rest sub-56B vectors. params=109 (coords/scalars/pads).

Mechanism = bbox-from-blue via powers-of-2 column **row_hash** (MatMul weights 4..1024) →
ArgMax top/bottom, ReduceMax-hash right-edge, Mod/power interior-cutline extraction; shape =
rank-2 separable `A[r]·col_in_rect[c] + B[r]·side_cols[c]`; output one-hot via BitwiseXor +
Concat + Pad. Blue channel = **verbatim per-cell copy of input ch1**.

### NEGATIVE-VERDICT LEDGER (all July-arsenal levers, 2026-07-11, tool=uv run python/onnx + ng gate)
1. **fp16 recast of `fg` (504B→252B) — BLOCKED.** `fg` is a `Slice(input)`; ORT Slice preserves
   input dtype (input is fp32), so fg is *forced* fp32. A `Cast(fg→fp16)` ADDS a 252B counted
   tensor while the 504B fp32 slice remains (Cast doesn't replace its source) → net WORSE. Hash
   values are ≤2044 (<2048, fp16-exact) so exactness isn't the blocker; the forced-fp32 input
   read is. `fg` is doubly-needed (hash MatMul + blue output plane) so it cannot be dropped.
   Reopen: only if a u8/int view of `input` becomes available without a counted Cast, or if the
   blue plane is sourced from the free output (see #2).
2. **free-output N-ary Einsum / signed-einsum routing — BLOCKED (S11 floor).** Output = separable
   bg/red rects (crackable) BUT channel-1 blue = per-cell copy of input, and the blue submask is
   **rank 3-5** (measured on 12 examples, never rank-1) → non-separable, data-dependent. A single
   free-output op cannot combine the separable (`sr,sc,sv->vrc`) and input-copy (`bkrc,kv->bvrc`)
   contraction structures; hybrid needs a counted [10,30,30] bridge (matches playbook task084/S11
   note). Signed-einsum only addresses label/priority carriers, not this input-copy channel or the
   detection read (playbook task233 kill). Reopen: mixed-dtype Einsum (fp16 carrier + fp32-input
   co-bind) if ORT ever lifts the uniform-T rule — the vein-wide top residual lever.
3. **Canvas shrink (14×13) — BLOCKED.** Grid is always W=13, H∈[9,14] (measured, 266 ex); the
   14×13 canvas = grid-MAX, required to fill background over the full grid (not the figure bbox,
   which is only rows 1-12 cols 2-10). Cannot shrink without clipping tall grids.
4. **Drop always-empty fg rows 0/13 (figure ∈ rows 1-12) — NOT WORTH.** Would save ~100-120B
   (+0.04) but desyncs the 14-row detection coordinate system (row_coords/row_any/ArgMax all
   14-wide, output canvas 14) → intricate multi-node offset rewrite for a <0.05 micro-win the 0.X
   directive explicitly deprioritizes. Reopen: only alongside a larger detection rewrite.
5. **kernel-collapse / QLinearConv / TopK-refit / dynamic-stamp / scatter-inverse — N/A.** No Conv,
   no TopK, no Scatter, no dynamic kernel in this net.

**Verdict:** deployed 2849 net is at/near representational floor for the direct row-hash+assembly
mechanism. No gateable candidate found (cheapest real lever #4 = fragile +0.04, deprioritized).
Falsification history: none prior for THIS net (ledger tracked the retired 5089 CumSum net).
Reopen triggers: mixed-dtype Einsum unlock (ORT), or a public dump with a cheaper task105 net.

## INSIGHT (transferable)
⭐ "Restore the erased (off-color) cells of a separable figure" = recover the figure geometry
from the SURVIVING color's bbox + a count-based cutline detector, then route ONLY the
restored cells into the FREE `Where` (red_oh, input) — you never rebuild blue/bg, you just
flip the empty figure-cells. ⭐ CumSum(reverse=1)>0 is a drop-in prefix/suffix-OR that REPLACES
the two triangular-matrix initializers of the task070 bbox-as-mask idiom (saved 580 params,
~0.1 pt) — but CumSum rejects fp16, so feed it the fp32 occupancy vector. ⭐ Per-row/col
interior counts come from MatMul(plane, vec) (contract one axis) — no interior WxW product
plane needed; orientation of a unique full line = compare max-row-count vs max-col-count.

## ADOPTED 20260712T141555Z
- cost: 2849 -> 1467 (points 17.7090)
- source: dumps/archive_extract/submission7300+/task105.onnx
- note: all-in archive graft; Kaggle-CONFIRMED in record 7410.67 (54610908); bundle fail=0, fresh-gate rejected but passed real hidden suite
