# Carrier-crossover scan (Pad-then-Equal vs Equal-then-Pad)

Proven rule (S11, playbook mechanism 7 addendum, task259 vs task041): a net that emits its
final `[1,10,30,30]` one-hot via a padded single-channel index carrier (**Pad-then-Equal**,
fixed ~900B `[1,1,30,30]` uint8) can instead build the 10-channel one-hot at CONTENT resolution
and Pad it to the free output (**Equal-then-Pad**, `10·h·w` bytes of bool). Equal-then-Pad wins
iff pre-pad content area `h·w < 90`.

Scan: all 400 deployed `networks/taskNNN.onnx`. 134 nets have a comparison tail reading a
single-channel `[1,1,30,30]`-ish integer/bool carrier. Classification below (graph inspection +
`load_task` bundled metadata + arc-gen generator source; no ORT inference).

## Result: 1 actionable candidate

- **CANDIDATE: [174]**  total est **+0.0958 pts**
- Verdict counts: {'CANDIDATE': 1, 'NA_WHERE': 49, 'NA_ABOVE': 51, 'NA_FULLCANVAS': 16, 'NA_GEN': 1, 'MARGINAL': 16}

**task174** is the only Pad-fed Equal-decode net whose pre-pad content sits below the 90-cell
crossover (5×5=25) while still emitting Pad-then-Equal. Tail: `Lin[1,1,5,5]u8 -> Pad -> L[1,1,30,30]u8`
(900B carrier) `-> Equal -> output`. Drop-in swap builds `Equal(Lin,levels) -> [1,10,5,5]bool` (250B)
then `Pad -> output`, deleting the 900B carrier for a 250B one-hot (−650B counted mem). Generator
(72ca375d) output = bounding box of box-0, `wide∈[2,5], tall∈[2,7−wide]` → max area 12 < 90 (safe).
Est +0.096 pts drop-in; up to +0.116 if the 5×5 content is also tightened to the true ≤12-cell output.

**task377** looked like a candidate on bundled data (max out 81 < 90) but is REJECTED: generator
eb5a1d5d emits `(2·len(colors)−1)²` and nesting depth reaches ~13 colors → outputs up to ~27×27
(area ~729 » 90). Its graph also pre-pads 11×11 (=121), so a drop-in Equal-then-Pad would build
1210B > 900B — a loss. NOT_APPLICABLE.

## Why the other 133 don't qualify

- **NA_ABOVE (51)**: Equal-decode, Pad-fed, but pre-pad content ≥120 cells — already on the correct
  (Pad-then-Equal) side of the crossover; a 900B carrier is cheaper than their ≥1200B content one-hot.

- **MARGINAL (16)**: Equal-decode, Pad-fed, content area 90–120 (fifteen 10×10=100 nets + task075 9×13=117).
  Inside the spec's marginal band but drop-in still LOSES (content 100 → 1000B one-hot > 900B carrier);
  no 'other saving' materialises because the content index stays counted in both forms. Effectively no gain.

- **NA_WHERE (49)**: tail is `Where(mask[1,1,30,30]bool, …) -> FLOAT [1,10,30,30]` — a select-mask, not an
  index one-hot. The crossover doesn't apply: padding a content-res FLOAT result is far more expensive than
  the 900B bool mask (some, e.g. t034/t341/t345/t381, have small pre-pad masks but the mechanism differs).

- **NA_FULLCANVAS (16)**: Equal-decode whose carrier is a genuine full 30×30 computed index (not Pad-fed);
  these are full-canvas / upscaling tasks with large output.

## Flagged tasks (CANDIDATE + MARGINAL + rejected candidate), sorted by est delta

| task | dec | cur pts | carrier B | prepad area | max bundled out | gen max area | verdict | est Δpts |
|---|---|---|---|---|---|---|---|---|
| 174 | Equal | 16.130 | 900 | 25 | 12 | 12 | CANDIDATE | +0.0958 |
| 377 | Equal | 15.981 | 900 | 121 | 81 | 729 | NA_GEN | +0.0000 |
| 368 | Equal | 16.264 | 900 | 100 | 100 | — | MARGINAL | -0.0159 |
| 361 | Equal | 16.547 | 900 | 100 | 100 | — | MARGINAL | -0.0211 |
| 069 | Equal | 16.728 | 900 | 100 | 100 | — | MARGINAL | -0.0252 |
| 037 | Equal | 16.816 | 900 | 100 | 100 | — | MARGINAL | -0.0275 |
| 250 | Equal | 16.921 | 900 | 100 | 100 | — | MARGINAL | -0.0305 |
| 333 | Equal | 16.921 | 900 | 100 | 100 | — | MARGINAL | -0.0305 |
| 354 | Equal | 17.080 | 900 | 100 | 100 | — | MARGINAL | -0.0357 |
| 062 | Equal | 17.162 | 900 | 100 | 100 | — | MARGINAL | -0.0387 |
| 260 | Equal | 17.349 | 900 | 100 | 100 | — | MARGINAL | -0.0465 |
| 124 | Equal | 17.423 | 900 | 100 | 100 | — | MARGINAL | -0.0499 |
| 392 | Equal | 17.424 | 900 | 100 | 100 | — | MARGINAL | -0.0500 |
| 099 | Equal | 17.445 | 900 | 100 | 100 | — | MARGINAL | -0.0510 |
| 348 | Equal | 17.453 | 900 | 100 | 100 | — | MARGINAL | -0.0514 |
| 088 | Equal | 17.465 | 900 | 100 | 100 | — | MARGINAL | -0.0520 |
| 041 | Equal | 17.511 | 900 | 100 | 100 | — | MARGINAL | -0.0544 |
| 075 | Equal | 17.695 | 900 | 117 | 117 | — | MARGINAL | -0.1668 |

_Full 134-row classification in `reports/carrier_crossover_scan.json`._
