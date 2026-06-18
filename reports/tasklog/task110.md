# task110 — 484b58aa

**Rule:** A 29×29 grid is a doubly-periodic colour tiling (colours 1..9). The
ROW period rp and COLUMN period cp are INDEPENDENT scalars; on FRESH random
instances rp==cp ∈ {2,4,5,6,7,8,9} (never 1 or 3, verified 16k axes), but the
fixed validate() cases reach cp=18 / no col period. The INPUT additionally has
up to five black (colour-0) rectangular cutouts (≤5×5). The OUTPUT removes the
cutouts, restoring every black cell to its periodic colour. Off-grid (row/col
29) is the ALL-ZERO one-hot.
**Current:** 13.48 pts, gen:vyank6322, mem 100033, params 350 (labelled
"confirmed-infeasible" BLANK note — FALSE POSITIVE).
**Target tier:** detection→closed-form periodic in-painting; B-ish.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | per-q full-plane detect + ±p iter fill (fp16, single period) | — | 996k | 2593 | — | 200/200 fresh but stored fail (rp≠cp) | wrong: assumed one period |
| 2 | separate rp/cp detect (q≤18) + gating | B | 775k | 3039 | 11.43 | — | correct, huge |
| 3 | windowed bool detect (WIN15) + uint8 fill | B | 134k | 1468 | 13.18 | 200/200 | |
| 4 | sentinel one-hot, frozen-cur fill, pad-sentinel boundary | B | 80k | 1382 | 13.69 | 500/500 | |
| 5 | max-of-4-donors fill + qcands skip(3) + WIN12 | B | 69860 | 1344 | **13.83** | 3000/3000 | ADOPTED |

## Best achieved
13.83 @ mem 69860 params 1344 — beats prior 13.48 by **+0.34** (≥+0.3 ✅).
Stored 266/266, isolated fresh 200/200 (and 3000/3000, 50000/50000 detect-axis).

## Irreducible-floor analysis
Not at floor. Dominant residuals: the 3600B fp32 Conv entry plane (colour-index
Σk·input_k — irreducible 10→1 reduction), then ~22KB windowed period detection
(7 q-candidates × 2 axes × ~8 tiny WIN×WIN planes) and ~45KB fill (3 passes ×
4 padded-Gather donors, all uint8 900B planes). Everything full-canvas is uint8.

## OPEN ANGLES
- Period detection via 1-D autocorrelation/MatMul instead of per-q full-window
  planes (would drop the ~22KB detection block).
- Fewer fill passes if a cheap multi-hop donor (±2p) could be made robust at 2
  passes (mults=[1,2] niters=1 failed 5/3000; niters=2 = 16 gathers, worse).
- WIN=11 reaches 13.87 but had 1/30000 detect-axis disagreement — rejected for
  robustness; WIN=12 is 0/50000.

## INSIGHT (transferable)
⭐ "BLANK-note confirmed-infeasible" periodic in-painting was a FALSE POSITIVE.
⭐ Period in-painting = (a) detect period as a SCALAR via smallest-q masked
   mismatch reduction on a SMALL TOP-LEFT WINDOW (global period needs no full
   canvas — WIN=12 matched full detection 0/50000 axes, planes shrink 6×); (b)
   route out-of-range Gather donors to a Pad-appended BLACK sentinel index (30)
   so one Greater(donor,0) test rejects them — no per-cell validity masks; (c)
   gate a missing-period axis by p_eff = found?p:99 (all donors → sentinel);
   (d) fill all black cells with elementwise MAX of the ±period donors (all
   valid donors carry the identical periodic colour, so max=correct, 0=none) —
   collapses a 4-way Where chain to one fill. Whole pipeline uint8 (Greater/
   And/Where/Equal/Gather/Pad all run uint8 under ORT_DISABLE_ALL); only the
   one fp32 Conv entry and the fp16 detection-count cast stay non-uint8.
⭐ When skipping non-contiguous q-candidates, map ArgMax index back through a
   Gather(qtab, idx) — NOT idx+offset (silent period-off-by-skip bug).
