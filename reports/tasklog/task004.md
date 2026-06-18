# task004 — 025d127b

**Rule:** Grid (size 8..16, variable H,W) holds 0..4 axis-stacked slanted parallelogram
outlines, one DISTINCT random colour each, separated by a full blank row (shapes are
ROW-SEPARABLE, never share a row). The output un-slants each shape's outline: every shape
row shifts RIGHT by +1, EXCEPT the shape's BOTTOM row (shift 0) and the rightmost pixel of
the SECOND-TO-LAST row (also shift 0). Colours simply copy the input colours. Reformulated
to a fully-local per-cell partition (verified exact 3000/3000, zero collisions):
rowany[r]=ReduceMax(occ,cols); below[r]=rowany[r+1]; below2[r]=rowany[r+2];
is_bottom=rowany∧¬below; is_2ndlast=rowany∧below∧¬below2;
special[r,c]=occ[r,c]∧is_2ndlast[r]∧occ[r+1,c] (occ pixel directly below);
copy_cell=occ∧(is_bottom∨special); shift_cell=occ∧¬copy_cell;
L_out=shiftR1(colf·shift_cell)+colf·copy_cell.
**Current:** 14.08 pts, gen:thbdh6332, mem 54000, params 1020
**Target tier:** A — output colours COPY arbitrary input colours (Tier S route blocked: a
fixed Conv can't route random per-instance colours), but the whole map is a separable
per-row shift collapsed into ONE colour-index value plane, no [1,10,H,W] product.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | full-30 fp16 planes, single L plane + ingrid mask | A | 43926 | 32 | 14.31 | n/a | works, MARGINAL (+0.22) |
| 2 | crop all working planes to 17×17 (active-region escape) | A | 25152 | 38 | 14.87 | 200/200 + 500/500 | ADOPT (+0.78) |
| 3 | fold in-grid into the colour Conv (ch0 weight=0.5 sentinel) — kills the redundant 30×30 fp32 ReduceMax `ingrid30` (3600B) + `ingrid_f` (1156B) | A | 20396 | 38 | 15.075 | 200/200 | +0.21 vs #2 |
| 4 | crop W=17→16 (measured input/output coloured extent ≤col15 over 30k fresh; col-16 shift overflow is a clamped no-op) | A | 19118 | 38 | 15.140 | 500/500 | +0.27 vs #2 |
| 5 | opset-11 int32 `Pad` (ORT accepts it!) — Cast Lmask→int32 at 16×16 then int32-Pad, dropping the 1800B fp16 30×30 bridge plane `L30` | A | 18342 | 72 | 15.179 | 200/200 | +0.31 vs #2 |
| 6 | `shift_cell = occ − copy_cell` (copy_cell⊆occ) drops the Sub(1)+Mul `notcopy` plane | A | 17830 | 72 | 15.207 | 500/500 | **ADOPT (+0.34)** |

## Best achieved
15.207 @ mem 17830 params 72 — recommend adopt: Y. Beats prior 14.87 by +0.34 (≥ +0.3).
Remaining dominant intermediates: `colf30` (3600B fp32 Conv entry — the one mandatory 10→1
reduction, fp32 forced by fp32 input) and `L30_i32` (3600B int32 — the mandatory Equal input,
opset-10/11 Equal accepts only int32/int64/bool). Everything else is a 16×16 fp16/bool working
plane (≤1024B). The two 3600B planes are the irreducible floor of this encoding.

## Irreducible-floor analysis
Dominant intermediates: colf30 (3600B fp32 Conv entry — the irreducible 10→1 colour-index
plane), ingrid30 (3600B fp32 ReduceMax-over-channels, needed to distinguish in-grid-bg from
off-grid since both have colf=0), L30 (1800B fp16 padded value plane) and its int32 cast
L30_i32 (3600B, forced because opset-10 Equal rejects fp16/uint8 — only int32/int64/bool).
Everything else is a 17×17 fp16/bool working plane (≈300–600B each). The 17×17 active-region
crop (generator bounds H,W ≤ 16, +1 col of shift headroom) is the lever that took it from
44k → 25k.

## OPEN ANGLES (re-attack backlog)
- DONE in #3: folded in-grid into the colour Conv (ch0 weight=0.5 sentinel) — killed `ingrid30`
  (3600B) + `ingrid_f` (1156B). off-grid=0 / in-grid-bg=0.5 / coloured=k from ONE plane.
- DONE in #5: int32 `Pad` IS legal under opset-11 ORT (the tasklog claim "Pad rejects int32" is
  opset-10-only) — Cast Lmask→int32 at WxW then int32-Pad, killing the 1800B fp16 30×30 bridge.
- The two remaining 3600B planes (`colf30` fp32 entry, `L30_i32` int32 Equal-input) are the hard
  floor: the entry must be fp32 (fp32 input → fp32 Conv), and opset-10/11 Equal accepts only
  int32/int64/bool, so the final colour-index 30×30 plane must be int32. Routing the one-hot
  expansion as a Concat-padded bool [1,10,WxW] measured WORSE (the 10-ch partial-pad bool plane
  ≥5100B). No cheaper path to the 30×30 int32 Equal input than fp16-mask→cast-at-WxW→int32-Pad.
- The fp32 `colf` (1024B WxW Slice) and `Lmask_i32` (1024B) are the next tier; both are
  structurally needed (Slice preserves fp32; Where→Cast for the int32 Pad). Marginal.

## INSIGHT (transferable)
⭐ A per-shape geometric SHEAR/"un-slant" that looks like it needs shape segmentation is
fully LOCAL when shapes are row-separable: classify each row by 1-cell vertical occupancy
neighbours (rowany[r], rowany[r±1,2]) and characterise edge-case pixels by their immediate
vertical neighbour (here the special "rightmost of 2nd-to-last row" = an occupied cell with
an occupied cell directly below in a 2nd-to-last row) — NO flood-fill, NO rightmost/argmax
scan. Then a colour-COPY remap collapses to ONE colour-index value plane
L=shiftR1(colf·shiftmask)+colf·copymask (masks partition the occupied cells with zero
collisions, so Add == Or). Pairs with the active-region crop (generator size bound) for the
big byte win — the 30×30 floor only really bites the two fp32 entry planes and the final
int32 Equal plane.
