# task178 — 746b3537

**Rule:** Input is a width×height grid of stacked solid horizontal colour bands;
band i has colour colors[i] (∈1..9; consecutive bands always differ) and thickness
thicks[i]∈1..3, with 3..5 bands, width∈1..5 ⇒ height=Σthicks≤15. Output is the
n=len(colors) × 1 column listing each band's colour once in order (run-length
de-duplication of the rows). If xpose, input AND output are transposed (bands run
down columns; output is a 1×n row).
**Current:** 15.478 pts, gen:thbdh6332 (public CumSum-scan net), mem 13569, params 87
**Target tier:** A (closed-form compaction; data-dependent output shape) — separable
single-axis run-length de-dup, no flood-fill / no connectivity.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | per-orientation full pipelines, separate Lnx/Lxp/Lsel fp16 planes | A | 18019 | 173 | 15.19 | 268/268 | too many planes |
| 2 | unify orientation BEFORE compaction (one [30,15] relation), 3 fp16 placement planes | A | 10487 | 145 | 15.73 | 200/200 | MARGINAL (+0.25) |
| 3 | fold axis-mask select via Not/And/Or (Or→active), one valsel + one Lsel | A | 9648 | 145 | **15.81** | 300/300 | **PASS (+0.333)** |

## Best achieved
**15.81 @ mem 9648 params 145** — adopted? N (per instructions). Beats prior 15.478 by **+0.333** ✅ (≥+0.3).

## Method
- Read band colour along BOTH long axes from one input line each, cropped to 15:
  col 0 → per-row colour `cvalR` (Conv kw=[0..9]); row 0 → per-col colour `cvalC`.
- band-start = (colour≠prev) AND (colour≠0); cumcount = inclusive CumSum.
- xpose = (#col band-starts > #row band-starts): non-xpose has col0 carrying all
  bands (≥3 starts) and row0 uniform (1 start); transposed flips. Removes any
  explicit distinct-colour-count plane.
- Select the active orientation's (colour, newstart, cumcount) length-15 vectors,
  run ONE compaction: oc[i] = colour at cell with cumcount==i+1, via a tiny
  [1,1,30,15] Equal relation × value, ReduceSum over the 15-axis → [1,1,30,1].
- Place that sequence down col0 (non-xpose) / along row0 (xpose) into ONE fp16
  colour-index plane (sentinel 99 elsewhere); axis-mask `active` built with
  Not/And/Or (bool Where is NOT_IMPLEMENTED in ORT), value selected with a fp16
  Where; final `Equal(Lsel, chan[0..9])` writes straight to the free BOOL output.
  Invalid cells equal no 0..9 channel ⇒ the data-dependent n×1 / 1×n output shape
  falls out automatically (band colours are always nonzero, so no channel-0 leak).

## Irreducible-floor analysis
Dominant intermediates: three [1,1,30,30] fp16 planes (`valsel`, `Lsel` @1800B,
`active` bool @900B) = ~4500B, plus the [1,1,30,15] relation/prod (~1350B) and the
two [1,10,15,1] input-line slices (600B each). The output MUST funnel through ONE
[1,1,30,30] plane before the single 10-ch Equal (the only allowed 10-ch expansion =
the free output), so at least one 30×30 fp16 plane (1800B) is structural. fp16 IS
counted at half here (calculate_memory uses the shape-inferred dtype even though ORT
inserts PrecisionFreeCast at runtime — the trace only supplies SHAPES). NOT at floor.

## OPEN ANGLES (re-attack backlog)
- Collapse `valsel`+`Lsel` into a single Where to drop ~1800B (would push to ~15.95):
  need the placed value at col0/row0 without a separate axis-selected value plane —
  e.g. Min of two gated small-broadcast placements, but naive Min overlays both
  orientations (wrong for NX along row0). Needs a clean gate that zeroes the inactive
  placement to sentinel without a second 30×30 plane.
- The [1,1,30,15] relation could shrink to [1,1,n_out≤5,15] if output rows were
  bounded to 5, but the placement axis still spans 30; marginal.

## INSIGHT (transferable)
⭐ "Run-length de-duplicate consecutive bands → compact line" is closed-form tier-A,
NOT a detection wall: band-start = (colour≠prev)∧(colour≠0) along ONE input line;
inclusive CumSum gives the output slot; out[i]=colour where cumcount==i+1 via a tiny
[out×in] Equal relation. Orientation (xpose) is recovered for FREE as #col-starts vs
#row-starts (no distinct-colour plane). Unify BOTH orientations into ONE compaction
by selecting the small length-L vectors with a scalar Where BEFORE the relation, then
place + transpose-free axis-select. ⭐ ORT under ORT_DISABLE_ALL has NO Where impl for
BOOL data inputs ("Where(9) NOT_IMPLEMENTED") — select between two bool masks with
Not/And/Or, not Where. ⭐ Data-dependent output shape is free: nonzero band colours +
99 sentinel ⇒ invalid cells match no 0..9 channel after the final Equal.
