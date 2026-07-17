---
deployed_cost: 2756
logged_costs_match: match
migrated: 2026-07-09
---

# task251 — a5313dff

**Rule:** Black(0) canvas with 1px-thick RED(2) rectangle outlines (wide/tall in {4,5,6}),
each optionally carrying an inner red core (inset 2). The 1-cell black "gap" ring just inside
each outline is recoloured BLUE(1) in the output — but ONLY for boxes fully inside the grid;
boxes drawn clipped (row/col = -1) keep their gap black. Boxes may abut, so a 4-dir
ray-enclosure / flood FALSE-POSITIVES on ~1-2 spurious cells at box junctions (112 across
stored exs) — the rule genuinely needs to MATCH the full box-outline template, not a local
enclosure test.

**Current:** 15.82 pts, ext:kojimar7113 (crowd template-match net), mem 8741, params 973
**Target tier:** detection/template-match — the size-specific box detect + paint is inherent;
the only lever is golfing the public mechanism, not a cheaper closed form.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | (prior session) bounded-unroll flood D=11 @ 13x13 | det | 12616 | 208 | 15.54 | 200/200 | WORSE than public (22 reach planes) |
| 2 | 4-dir ray-enclosure closed-form | — | — | — | — | — | 112 spurious cells at box junctions — WALL |
| 3 | template-match @ W=12, no-pad conv | det | 5949 | 996 | 16.15 | — | 58 stored fails (4x4 box at edge unreachable) |
| 4 | + conv pads=[0,0,2,2] (effective 14 window) | det | 7781 | 1002 | 15.92 | — | 0 fails, correct |
| 5 | single-channel combo=red-black detector | det | 7493 | 678 | 15.99 | — | drops 2-ch concat + halves detector params |
| 6 | + ConvTranspose output-crop pads=[0,0,2,2] | det | 7101 | 672 | **16.04** | 200/200 | BEST |
| 7 | bool-mask uint8 tail (derive masks from combo) | det | 7533 | 674 | 15.99 | — | extra small bool planes ate the uint8 saving — reverted |

## Best achieved
16.04 @ mem 7101 params 672 — adopted? N (agent does not adopt). Beats prior 15.82 by **+0.22**
(MARGINAL, < +0.3). Stored 266/266, fresh 200/200.

## Irreducible-floor analysis
The detection bank dominates and is the WALL: scores[1,9,9,9] fp16 (1458) + detect_bool[1,9,9,9]
bool (729) + detect[1,9,9,9] fp16 (1458) = **3645B**, ~51% of mem. Why irreducible:
1. **9 channels** — wide,tall each uniform in {4,5,6}, independent → all 9 combos occur with
   equal probability; each needs its own outline template AND its own gap-ring paint stamp, so
   no channel can be dropped or merged (the size determines both the detect kernel and the paint).
2. **9x9 anchor grid** — exactly the valid in-grid top-lefts for a width-4 box (col 0..size-4=8).
   Pinned; can't stride (anchors are per-cell).
3. **fp16 floor** — ORT Conv AND ConvTranspose have NO uint8/int8 kernel (verified: uint8
   ConvTranspose → INVALID_GRAPH), so scores+detect are fp16. The hard threshold (essential: max
   partial score 0.9375 vs perfect 1.0; painting raw fractional scores smears) needs Greater,
   which MUST emit a bool intermediate, then a fp16 Cast for the ConvTranspose. So bank ≥ 1458+729+1458.

Surround already cropped to the generator's hard 12-cell bound (slices/casts/combo/fill/out_black/
small3 ≈ 3456B); padding the 3-ch result DIRECTLY into the FREE output avoids any 30x30 plane.
Whole net floors near 7101 → 16.04.

## OPEN ANGLES (re-attack backlog)
- **Separable filled-interior paint**: blue gap = filled-interior AND not-red, and the FILLED
  interior IS row⊗col separable (verified). If the 9-channel detect could be replaced by detecting
  box row-extent (3 ch) and col-extent (3 ch) and the paint done as two 1-D dilations, the bank
  could drop ~40%. Blocker found this session: associating each cell with its box's (w,t) to drive
  the separable fill needs per-box labelling — no cheap construction found. Worth a fresh look at
  whether a CumSum/prefix construction can carry box extent per-cell without the 9-ch ConvTranspose.
- **Drop detect_bool**: if a single op could threshold fp16→fp16 {0,1} (no bool), saves 729B →
  ~16.10 (just over +0.3). ORT has no such op (Greater/Less/Equal all emit bool; Relu/Clip give
  fractional, needing 2 planes). Re-probe if a future ORT adds a fp16-out compare.

## INSIGHT (transferable)
- ⭐ **ConvTranspose `pads` CROP the output** (opset-11, fp16 OK): pads=[0,0,2,2] on a transpose-conv
  whose natural output is 14x14 emits 12x12 directly — kills the trailing fill-slice plane. Mirror of
  Conv pads but subtractive. Use whenever a paint/upsample lands in a sub-region of its natural extent.
- ⭐ **Single-channel `combo = red - black` replaces a 2-channel detector concat**: when a template
  needs "feature A present AND feature B absent" (here red outline present + gap black), encode both
  as a signed 1-channel field (A=+1, B=-1) with signed template weights — perfect match still sums to
  1.0, drops the concat plane AND halves the conv params. (Verified 0/262 fails vs 2-ch.)
- A box-fill / enclosure task with ABUTTING boxes is NOT closed-form via ray-enclosure or flood —
  junction cells get ray-enclosed by two different boxes' outlines (112 spurious here). Genuine
  per-size template matching (detect top-left + paint stamp) is required; the public crowd net's
  Conv+ConvTranspose mechanism is the right structure and golfs only ~ +0.2 (crop to active canvas).
- A documented "200/200 fresh" on-disk custom that was NOT adopted usually means it scored BELOW the
  public net (here a flood version at 15.54 < 15.82) — measure before trusting the docstring's tier.

## S8 (2026-07-02) — walk-einsum border flood (+0.432) ADOPTED, div 0
9-iter Min-gated QLC flood chain (17×144B u8 planes) → ONE 8-conn 14-slot walk einsum on the 12×12 crop (task002 pattern: ring seeds, S entries 1.0, Greater(non,W) epilogue). 2772+272 vs 4500+187 → 16.547→16.979. Fresh 2500 cached + 5000 uncached + 600 div 0.

## 2026-07-08 — fat-middle fresh-sweep ADOPTED (+0.098)
- Deep-attack of the "separable row⊗col paint" open angle → that angle is DEAD (targets the retired
  9-ch template-match formulation, floors ~6300 vs current flood champion). **Real win via fresh
  re-inspection**: the 8-conn border-flood walk-Einsum carries `red`/`non`/`W` as fp32 (576B each);
  `non`(=1-red, {0,1}) and `W`(small nonneg walk counts) are fp16-exact → recast to fp16 (288B each).
- Gate: bundled fail=0 (266/266), **DEPLOYED 3039 → 2756 (−283B)**, unsigned-TopK scan clean.
  (Note: measured vs DEPLOYED overfit net, not src — src baseline 3044 was staler.)
- ⭐ TRANSFERABLE: fp16 count-plane recast on flood/walk intermediates holding {0,1} masks + small
  walk counts. Floor after: `mask30`[1,1,30,30] bool 900B (output-welded, S10) + `red` fp32 576B (needed
  for off-grid=traversable via 1-red). Candidate: reports/candidates/task251/fatmid_attack.onnx.

## ADOPTED 20260710T064810Z
- cost: 2756 -> 1531 (points 17.6663)
- source: /tmp/task251_cand.onnx
- note: runtime-spend S8-port fanout (187-batch7 epsilon-J single-channel variant): Einsum(W)->Greater->Pad->Where(mask,BLUE,input) tail collapsed into ONE free-output einsum. Sole counted plane = t2 Conv [1,1,12,12] 576B (neg-pad 30->12 crop in-op, non-red traversability). Stack a∈{0,1} with t2 SHARED not stacked (halves vs 243): a=1 real 8-conn border flood (ring seed 2 rank-1 terms), a=0 J/12 rank-collapse constant K; signed T decode: out[1]=K−W (enclosed->BLUE), out[0]=W, out[2]=K. 14 transitions. 2756->1531 (+0.588). bundled 266/0, bit-identical, fresh 4000/4000. TRANSFERABLE: epsilon-J single-channel avoids the stacked-plane doubling whenever the Where paints the W=0 (unreached) side; border-flood/enclosure nets are the class.

## ADOPTED 20260715T021900Z
- cost: 1531 -> 1475 (points 17.7036)
- source: candidates/task251/factor_seeds.onnx
- note: exact shared rank-2 factorization of G/H seed routes in final free-output Einsum

## ADOPTED 20260715T120646Z
- cost: 1475 -> 1335 (points 17.8033)
- source: candidates/task251/channel_support_collapse.onnx
- note: factor-seed plus generator-supported input-channel collapse T[2,10,10] -> E[2,10]*Tprime[2,10,2]

## ADOPTED 20260715T133459Z
- cost: 1335 -> 1191 (points 17.9175)
- source: candidates/task251/walk_route_shared.onnx
- note: share the 14-step adjacency transition across epsilon-K and enclosure-W routes; scale all-seed K by 2^-60

## ADOPTED 20260715T141131Z
- cost: 1191 -> 1159 (points 17.9447)
- source: candidates/task251/channel_code_shared.onnx
- note: shared rank-2 channel code on reachable input and output axes (E/Tprime 60 -> D/C 28)

## ADOPTED 20260715T142736Z
- cost: 1159 -> 1155 (points 17.9481)
- source: candidates/task251/diagonal_channel_metric.onnx
- note: diagonalize the shared rank-2 channel metric (2x2x2 core 8 -> route weights 4)

## ADOPTED 20260715T144759Z
- cost: 1155 -> 1147 (points 17.9551)
- source: candidates/task251/seed_precontracted.onnx
- note: exact initializer-only seed precontraction abK*abL -> aKL (16 -> 8 params)
