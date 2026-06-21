# task286 — b782dc8a

**Rule:** Maze of cyan(8) walls + black(0) corridors with a seed pair (two colours at adjacent,
opposite-parity cells; sometimes 2-4 painted cells). Output recolours the 4-connected black component
CONTAINING the seeds by a checkerboard parity map `out[r][c] = pair[(r+c)%2]` (pair[p] = the seed
colour whose own (r+c)%2 == p); cyan stays cyan, unreached black stays 0. Grid 10..25 each dim.
Reached fraction of black cells varies 1%→100% (median 88%, only 7% fully connected) so reachability
is genuinely required, not "colour all black". Input→output is a clean deterministic FUNCTION (seeds +
parities recoverable from input) — NOT a collision/ambiguity infeasibility. 8-connected flood == 4-
connected flood here (maze carved in 2-cell steps), max 4-conn geodesic depth ≈ 69.

**Current (RE-MEASURED 2026-06-21 — DEPLOYED NET WAS UPGRADED since the 13.10 log below):** the live
`networks/task286.onnx` is now **13.72 pts, mem 76279, params 2596** — a *pointer-jump* component-
labeller (CumSum + 24× ScatterElements/GatherElements + ArgMax), NOT the old MaxPool+Min flood. It is
~2× cheaper than the MaxPool flood. Fresh-measured: **1/300 bad** (the documented D-tail micro-leak —
its propagation depth is cut below the worst-case geodesic, silently failing ~0.06% fresh; passes all
stored). So the floor to beat rose from 13.10 → **13.72** and the technique improved from flood → jump-flood.

**Target tier:** detection/flood — bounded reachability is the only exact method; not closed-form.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | (a) closed-form/separable parity | — | — | — | — | — | INFEASIBLE: reachability essential (1%–100% coverage) |
| 2 | (b) 3x3 MaxPool+Min flood, 30x30, D=64 fp16 | flood | 279904 | 1851 | 12.45 | 200/200 | exact but heavy (prior session) |
| 3 | (b) + crop 25x25 D=60 / D=72-robust | flood | 199k–234k | ~1.3k | 12.63–12.79 | 200/200 | exact; below even old 13.10 |
| 4 | lattice downsample-by-2 (DATA-DEP PERIOD lever) | — | — | — | — | — | INFEASIBLE: corridors fill all 4 (r%2,c%2) classes |
| 5 | (2026-06-21) pointer-jump CC label, ~10 rounds × ~4 [900] fp16 planes | flood | ~72k est | — | ~13.8 | — | at best TIE with 13.72, NOT +0.3 |

## Best achieved
13.72 (current deployed jump-flood, unbeaten). Beats prior by +0.3? **N — INFEASIBLE.**

## Irreducible-floor analysis
Whole task collapses to "compute the 4-connected reachability mask M from the seeds"; output is then
trivial. M is a data-dependent connectivity flood. Two correct bounded encodings, BOTH ≥ current 76279:
- **MaxPool flood** (prior log): walls 1-thick ⇒ safe dilation radius 1/round, geodesic depth ≈69 ⇒
  ~69 rounds × (1800B fp16 dilate + 900B remask) ≈ 186k. ORT has no uint8/bool MaxPool so fp16 is the
  dtype floor. Strictly WORSE (≈12.4 pts) than current.
- **Pointer-jump CC labelling** (= what the live net already does): ~ceil(log2 N) rounds of
  Gather(label,ptr)+min-update+ptr-update on [900] vectors; fp16 labels exact (idx <2048). ≈10 rounds ×
  ~4 planes × 900 × 2B ≈ 72k ⇒ ≈13.8 — at best a TIE with 76279, and matching the live net's exactness
  (let alone improving to 3000/3000) eats any byte saving.
To gain +0.3 mem+params must drop 26% (78875 → ≤58432); neither encoding reaches that. The live jump-flood
sits at the structural floor for this connectivity task.

## OPEN ANGLES (low/zero expected value)
- Corridor-compaction crop to shrink planes → BLOCKED: data-dependent variable grid (10–25) leaves
  symbolic dims (calculate_memory → None trap).
- Sub-fp16 flood/label planes → BLOCKED by ORT (no uint8/bool MaxPool, no int8 Max).
- No separable / count-parametric / copy reformulation exists for a data-dependent CC flood.
- Gap-closer via robust-D adopt: the live net leaks ~0.06% fresh; a robust net raises REAL LB by
  ~0.0006×13≈0.008 while LOSING stored points (more rounds). Net-negative — do not adopt.

## INSIGHT (transferable)
⭐ task286 RE-CONFIRMED INFEASIBLE-TO-BEAT (2026-06-21): the deployed net was UPGRADED from a MaxPool+Min
flood (13.10) to a pointer-jump component-labeller (**13.72 / 76279 / 2596**) — so re-read the LIVE net
before trusting an old tasklog floor; the deployed technique can change between sessions. Any correct
rebuild (MaxPool ≈186k WORSE, competing pointer-jump ≈72k TIE) cannot reach the +0.3 target (needs
≤58432). The reachability mask is the sole load-bearing computation and is a genuine connectivity flood
(geodesic ≈69, coverage 1%–100%, no separable/period shortcut). Bonus: the live net still leaks 1/300
fresh (D below worst-case geodesic) — a latent gap-attribution micro-leak, not stored-relevant.
