---
deployed_cost: 1774
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task302 — c0f76784

**Rule:** 3 (sometimes 2) non-overlapping square gray(=5) box outlines of side L∈{3,4,5}
on a fixed 12×12 grid, each hollow (its (L-2)×(L-2) interior is background black=0). Output
keeps every gray frame and FILLS each hole solid with colour 5+(L-2)=3+L (L=3→6, L=4→7,
L=5→8). Every interior cell's fill = 5 + s where s = hole side (=L-2).
**Current:** 16.04 pts (prior public net)
**Target tier:** A (separable closed-form: per-cell colour-index plane routed into FREE bool output)

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | run-length(hrun) + 4-dir gray-bound, ingrid AND, fp16 30×30 | A | 25164 | 125 | 14.86 | 266/266 | works, too big |
| 2 | drop ingrid (sentinel-99 Pad handles off-canvas) | A | 12888 | 117 | 15.53 | — | win |
| 3 | windowed-gray enclosure (no run-bound) + presence-count value | A | 11556 | 108 | 15.64 | — | win |
| 4 | LINEAR weighted-conv value (2·G@1+G@2 both sides) | A | 8388 | 87 | 15.96 | — | big win |
| 5 | pack gl/gr/gu/gd into ONE [4,1,7,7] conv + ReduceProd; uint8 L | A | 6084 | 240 | 16.25 | — | win |
| 6 | L = Where(interior, fill, 5·G) (fold fillterm/intf/add) | A | 5508 | 240 | 16.34 | 200/200 | win |
| 7 | interior = (G==0)&encG (drop black-channel slice) | A | 4932 | 234 | 16.45 | 500/500 | **adopted** |

## Best achieved
16.45 @ mem 4932 params 234 — adopted? Y. Beats prior 16.04? Y (+0.41).

## Irreducible-floor analysis
Dominant intermediates: the [1,4,12,12] fp16 enclosure conv stack (1152B, four within-3
directional gray sums; needs all 4 → can't collapse to a sum because adjacent boxes put 2
grays in one within-3 arm, so a sum==4 test misfires — ReduceProd>0 is required) and the
[1,1,30,30] uint8 padded colour-index L (900B, the one unavoidable index plane; uint8 via
Pad halves the fp16 cost). Everything else is 12×12 (=144 cells) so fp16 planes are only
288B — the small-active-canvas escape (generator size bound 12) is what makes this Tier-A.

## OPEN ANGLES (re-attack backlog)
- Could fold valsum into the enclosure conv as a 5th channel (−288B mem, +49 params) — net
  neutral, skipped.
- The [4,1,7,7] enclosure kernel is 196 of the 234 params; a reshape-to-batch trick to share
  one 1-D kernel across the 4 arms would cut params but adds planes — not worth at 16.45.

## INSIGHT (transferable)
⭐ "Read a hole's SIZE from the distance to its flanking frames" beats run-length: when a
generator guarantees objects are ≥1 cell apart, within a small window along one axis there is
AT MOST ONE marker, so the nearest-marker distance becomes a single LINEAR weighted Conv
(weights 2,1,… by proximity) — NO argmin, NO product run-length chain. Pair it with a
ReduceProd-over-channels enclosure test (pack the 4 directional within-k sums into one conv,
product>0 ⇔ marker present in every direction; a sum-threshold FAILS under adjacency). And
`interior = (plane==0) AND enclosed` removes a whole second channel slice when the in-grid
palette is just {bg, marker}.

## 2026-07-01 (S7 re-run) — FLOOR re-confirmed
mem 1216/17.52; ch5 576B fp32 gray slice=min entry, features concat + ch5_u8 feed final QLinearConv (both needed). No safe reduction; all dominant intermediates structurally forced (fp32 entry crop / int32-64 index buffer / full-canvas routing mask).

## ADOPTED 20260713T143839Z
- cost: 1774 -> 1773 (points 17.5196)
- source: candidates/public_dumps/20260713_7281/extracted/task302.onnx
- note: Ryosuke 7281.18 public-LB confirmed per-task min-merge; bundled fail=0

## ADOPTED 20260713T151007Z
- cost: 1774 -> 1773 (points 17.5196)
- source: candidates/public_dumps/20260713_7281/extracted/task302.onnx
- note: Ryosuke-7281 isolation B; task047 explicitly excluded; bundled fail=0

## ADOPTED 20260713T151949Z
- cost: 1774 -> 1773 (points 17.5196)
- source: candidates/public_dumps/20260713_7281/extracted/task302.onnx
- note: Kaggle-isolated safe: group delta +2.05 exactly (sub 54651291 minus 54651270); task047 excluded

## ADOPTED 20260714T143609Z
- cost: 1773 -> 1385 (points 17.7665)
- source: candidates/task302/cand.onnx
- note: 4-state single-plane collapse: Max(u, 3x3-spread of 5x5-ring-centre) folds the 2-channel Concat into ONE u8 plane; final QLinearConv drops [10,2,5,5]->[10,1,5,5] (500->250 params). x_zero_point=10 supplies the off-grid 4th state free. Kernels = cutting-plane integer fit, clean on 6000 fresh (0 divergence vs incumbent).

### Mechanism (supersedes the "FLOOR re-confirmed" verdict above — that was wrong)
Prior public net: `Slice -> Cast -> QLinearConv(ring5x5) -> QLinearConv(fill3x3) -> Concat
-> QLinearConv[10,2,5,5]`, mem 1216 / params 558. The `Concat` existed only to give the
final conv two independently-weighted input channels (gray, and the L5-hole count). Replacing
`Concat` with `Max` merges both into ONE u8 plane carrying four distinguishable states:

    u    = QuantizeLinear(ch5, 1/14)        {0 black, 14 gray}
    ring = QLinearConv(u, 5x5 border, y_scale=14, B=-210)   = clamp(gray_count - 15)  [1,1,8,8]
    dsp  = QLinearConv(ring, ones3x3, w_scale=5, y_zp=7, pads=3) = {7, 12}             [1,1,12,12]
    m    = Max(u, dsp)                      {7 black, 12 L5-hole, 14 gray}
    out  = QLinearConv(m, [10,1,5,5], x_zp=10, pads=[2,2,20,20])   -> free output

mem 1072 (576 fp32 entry + 4x144 u8; `ring` stays 8x8=64 via pads=0) / params 313 / cost 1385.

Why one plane suffices, and why 3 states do NOT (LP-proved over the saturated window set):
out-ch8 needs `black == off-grid` so the ring signal is isolable by a single hyperplane;
out-ch0 needs them distinguishable. A 3-state plane can satisfy exactly one, so the pair is
unsatisfiable for EVERY encoding. The 3x3 spread of the ring centre (`dsp`) adds the fourth
state and breaks the tie. `x_zero_point` is what makes the off-grid state free: conv padding
contributes exactly 0, so the 4th state costs no bytes and no params.

### Negative-verdict ledger correction
- date: 2026-07-14
- ran: full LP/MILP separability sweep over all 3-state and 4-state single-plane encodings
  (5x5 and 3x3 final kernels), on a window set saturated with 40k arc-gen instances.
- verdict: the 2026-07-01 "FLOOR re-confirmed / no safe reduction / all dominant intermediates
  structurally forced" entry is FALSIFIED (-388 cost, +0.2469). What was forced is the 576B
  fp32 entry crop and the u8 gray plane; the `Concat` + [10,2,5,5] conv were NOT.
- reopen: the 576B fp32 entry is the remaining suspect. It is forced only while the final conv
  needs a u8 feature; a fp32-native detect->free-output route would delete it.

### ⭐ TRANSFERABLE
**Concat -> Max/Min collapse.** Any net ending in `Concat(A, B) -> QLinearConv[10,C,k,k]` is
paying `10*C*k*k` params for independent per-channel weights. If A and B take few discrete
values, re-scale them so `Max(A, B)` is injective on the reachable state combinations and
feed the final conv ONE plane: params halve (C 2->1) and the Concat's bytes vanish. The
final conv's `x_zero_point` then supplies one extra state for free (padding contributes 0),
which is exactly the state a cropped-canvas task needs for "off-grid".
Gate before building: LP-test single-plane separability per output channel over ALL state-value
directions. It is a 2-parameter search (values are scale-invariant), so it is cheap and gives a
proof, not a guess. Scan the public-dump nets for the `Concat -> QLinearConv` signature.

**METHOD WARNING (cost me one failed gate):** fitting the final conv's hyperplanes on the
bundled 266 examples' windows overfits — it passed bundled 266/266 but failed 127/2500 fresh.
The window set does NOT saturate by sampling (2972 windows @266 ex -> 8555 @40k, still rising).
Fit with a cutting-plane loop (fit -> harvest violated fresh windows -> refit) and confirm
`candidate != incumbent = 0` on >=6000 fresh. Directions that look feasible on a narrow window
set are artifacts: 32 "feasible" 4-state directions @266 ex collapsed to 0 for the chosen one
once the true distribution was included.
