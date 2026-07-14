---
deployed_cost: 3526
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task137 — 5c2c9af4

**Rule:** Input is a size×size grid (size 20..30) with exactly 3 pixels of one
colour at rows {row-s, row, row+s} on a ±1 diagonal (flip picks the diagonal),
spacing s in [2, size//4]. Output draws concentric square ring perimeters around
(row, col): cell (r,c) in-grid is `color` iff max(|r-row|,|c-col|) % s == 0, else
black; outside the (top-left) grid the canvas is all 0.
**Current:** 16.039 pts, custom:task137 (Gather idx-plane), mem 7106, params 686
**Target tier:** A — closed-form chebyshev rings; output colour COPIES the input
colour so a fixed Conv/route works, but it is a 3-state output (off-grid/black/
colour) so a colour-index route is needed (not pure separable bool).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | prior: Gather(palette, int32 idx[30,30]) + Less plane | A | 7106 | 686 | 16.039 | 200/200 | baseline |
| 2 | bool ring + Where(one-hot) | - | - | - | 0 | - | off-grid black leak + 9000B else-plane |
| 3 | uint8 idx, 10 separate [30,30] planes | A | 11071 | 685 | 15.63 | - | too many planes |
| 4 | per-axis 1-D idx vecs + dlt-select + offgrid override (4 planes) | A | 5791 | 685 | 16.22 | 200/200 | ok but <+0.3 |
| 5 | + fp16 vectors + fp16 fmod (drop int32 Mod) + 4-D (drop squeeze) | A | 4937 | 685 | 16.366 | 200/200 | beats +0.3 |
| 6 | fold off-grid into 1-D vecs by forcing off-axis to dominate (2 planes) | A | 3197 | 686 | **16.736** | 500/500 | ADOPTED |

## Best achieved
16.736 @ mem 3197 params 686 — beats prior 16.039 by **+0.70**. fresh 500/500.

## Irreducible-floor analysis
Only TWO canvas-sized intermediates remain: `dlt = Less(dr2,dc2)` (900B bool) and
`Lidx = Where(dlt, lc, lr)` (900B uint8). Chebyshev distance is genuinely 2-D so
≥1 canvas plane is unavoidable; the comparison is the second. Remaining mem is the
two fp32 Conv outputs (120B each) + small fp16 1-D vectors. Params (686) are now
the single biggest term — dominated by the two 300-element row/col occupancy Convs
[1,10,1,30]/[1,10,30,1] (ch0 weight 0). Halving them by slicing/reducing channels
instead trades 540 params for 2000-3600B of intermediate plane → strictly worse at
this memory scale, so 686 is the efficient choice.

## OPEN ANGLES (re-attack backlog)
- Collapse the two Convs to one if a single kernel could emit both row- and col-
  occupancy (different orientations block this with a plain Conv); a grouped /
  reshaped contraction might shave ~300 params (~+0.07).
- The 99-sentinel "force the off-grid axis to dominate" trick removed the offgrid
  OR plane AND the Lin plane at once — verify it transfers to other square-grid
  off-canvas-zeroing tasks.

## INSIGHT (transferable)
⭐ For a 3-state per-cell output (off-grid→all-zero / background→ch0 / value→ch_k)
route a uint8 index plane into a FREE BOOL output via Equal(Lidx, arange(10)) with
an OUT-OF-RANGE sentinel (10) for off-grid (matches no channel → all-off). To zero
the off-grid region WITHOUT a dedicated offgrid plane, push the sentinel into the
1-D per-axis index vectors and FORCE the off-grid axis to win a max/select by
setting that axis's distance to a huge value (99) on its 1-D bound mask — collapses
both the offgrid OR-plane and the intermediate select-plane, leaving just the
dominating-axis comparison + the index plane (2 canvas planes total).

## ADOPTED 20260709T041330Z
- cost: 3526 -> 3389 (points 16.8717)
- source: candidates/public_dumps/20260709/7261-53-lb-compact-onnx-artifact-starter/nets/task137.onnx
- note: min-merge from nets

## NEGATIVE LEDGER 2026-07-11 (deepfold indicator_fold flag = FALSE POSITIVE; floor)
Deployed net (min-merge, cost 3389 = mem 3263 + params 126, bundled 266/0) is the
Einsum-marker-detect form, NOT the old 686-param Conv form. Byte map: THREE 900B
canvas planes dominate — `cheb`=Max(dr_u8,dc_u8) uint8, `cheb_rem`=Mod(cheb,s) uint8,
`nonring`=Cast(cheb_rem)->bool — feeding final free Where(nonring,input,center_vec).
Marker DETECTION is tiny (ReduceSum+Equal+3 scalar Einsums ~150B); the 2700B is all
CARRIER = the Chebyshev-ring render. Remaining ~563B = the two 1-D row/col distance
chains + scalars.

- (1) WHAT RAN: deepfold indicator_fold candidate (nonring, dpts 0.3087). Ran the
  crack_condition 3-test + built/gated two plane-collapse candidates:
  (a) Gather-LUT: merge Mod+Cast into Gather(ring_lut[30], cheb) killing cheb_rem.
      REJECT — ORT 1.26 Gather rejects uint8 indices; casting cheb->int32 index = 3600B
      plane, +900B worse than the 900B it removes.
  (b) cmp+Where (drop Max): nonring=Where(dr>=dc, dr%s!=0, dc%s!=0) with bool branches.
      REJECT — ORT Where(9) with bool DATA is NOT_IMPLEMENTED. uint8 branches then need a
      trailing Cast->bool = back to 3 planes, no win.
  crack_condition verdict = FLOOR: #1 global-state routing HOLDS (mask is fn of ~4
  scalars cr,cc,s,size) and #2 Where-ROUTING HOLDS, but #3 FAILS — ring is at a
  DATA-DEPENDENT CENTER (positioned) and the paint predicate is a data-dependent 2-D
  COUPLING (chebyshev max) + run (mod s). Free-output einsum fold = telescoped
  nested-square rank-1 (task240 family) needs dynamic per-axis band matrices Rmat/Cmat
  [T,30], T=ceil(30/s)~30 worst-case (s=2) => 2x[30,30] fp16 = 3600B fp32-co-bound,
  >> current 2700B uint8/bool. Matches the lever boundary rule ("data-dependent
  run/coupling -> 900B index-plane is cheapest; every Einsum factorization prices
  higher"). Also mask FRACTION often <45% (s=7 ~14%).
  ORT-LEGAL PLANE FLOOR = 3: from (dr,dc) to the bool ring-mask needs Max, Mod, Cast =
  3 ops = 3 counted 900B planes; no single op does max-then-modular-test, Gather is
  int-index-taxed, Where-bool unimplemented, And/Or decomposition needs >=4 planes.
  Chains/scalars (~563B) and params (126, hsq/idx30 reused) are already minimal.
- (2) TOOL+DATE: uv run ng gate + hand-built onnx (onnx 1.21.0 / ort 1.26.0), 2026-07-11, opus.
- (3) REOPEN TRIGGER: an ORT build/opset that supports uint8/int8 Gather indices OR
  Where with bool data (would enable the 2-plane Gather-LUT / cmp+Where, ~+0.25); a
  fused max-modulo or mod-to-bool op; a new public teacher with a cheaper task137;
  a scorer change letting a mixed-dtype (fp16 carrier + fp32 input) einsum fold the ring.
- (4) FALSIFICATION HISTORY: none prior for the fold; this net already beat the
  self-built 686-param form via 2026-07-09 min-merge (3883->3389). 137 was NEVER in any
  free-output-einsum floor list (floor_tasks/batch6_floored/pending_unconfirmed) — this
  is its first honest crack attempt, verdict FLOOR by data-dependent-coupling economics.

## ADOPTED 20260713T143829Z
- cost: 3389 -> 3017 (points 16.9880)
- source: candidates/public_dumps/20260713_7281/extracted/task137.onnx
- note: Ryosuke 7281.18 public-LB confirmed per-task min-merge; bundled fail=0

## ADOPTED 20260713T150957Z
- cost: 3389 -> 3017 (points 16.9880)
- source: candidates/public_dumps/20260713_7281/extracted/task137.onnx
- note: Ryosuke-7281 isolation B; task047 explicitly excluded; bundled fail=0

## ADOPTED 20260713T151938Z
- cost: 3389 -> 3017 (points 16.9880)
- source: candidates/public_dumps/20260713_7281/extracted/task137.onnx
- note: Kaggle-isolated safe: group delta +2.05 exactly (sub 54651291 minus 54651270); task047 excluded
