# task286 — b782dc8a

**Rule:** Maze of cyan(8) walls + black(0) corridors with a seed pair (two colours at adjacent,
opposite-parity cells). Output recolours the 4-connected black component CONTAINING the seeds by a
checkerboard parity map `out[r][c] = pair[(r+c)%2]` (pair[p] = the seed colour whose own (r+c)%2 == p);
cyan stays cyan, unreached black stays 0. Grid 10..25 each dim. 94% of instances have ≥1 unreached
black region (so reachability is genuinely required, not "colour all black"). 8-connected flood == 4-
connected flood here (0/400 mismatch — maze carved in 2-cell steps, no diagonal bypasses a 1-thick
wall the orthogonal path misses), so dilation = a single 3x3 MaxPool (5x5 LEAKS, 376/400).

**Current:** 13.10 pts, mem 145879, params 1294. The deployed net is the SAME bounded-unroll flood:
59 MaxPool + 59 Min on a 25x25 fp16 region. It cuts D to 59 for the score and thereby FAILS the high-D
tail: **5/8000 fresh fails, all D>59 instances** — i.e. the deployed net is ~0.06% non-generalizing.

**Target tier:** detection/flood — bounded-iteration unrolling is the only exact method; reachability
is not closed-form and the corridor lattice is not cleanly downsamplable.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | (a) closed-form/separable parity | — | — | — | — | — | INFEASIBLE: 94% multi-component, reachability essential |
| 2 | (b) bounded unroll, 3x3 MaxPool+Min, 30x30, D=64 fp16 | flood | 279904 | 1851 | 12.45 | 200/200 | exact but heavy |
| 3 | (b) + crop to 25x25, D=60 | flood | 199679 | 1313 | 12.79 | 200/200 | exact; below 13.10 |
| 4 | (b) + folded-Conv entry, D=72 (robust to worst-case D=68/40000) | flood | 234529 | 1328 | 12.63 | 200/200 | exact + ROBUST; still below 13.10 |
| 5 | lattice downsample-by-2 (DATA-DEP PERIOD lever) | — | — | — | — | — | INFEASIBLE: corridors occupy ALL 4 (r%2,c%2) classes densely — not a clean lattice |

## Best achieved
12.63 @ mem 234529 params 1328 (D=72, fully robust) — NOT adopted. Beats prior 13.10? **NO.**
The deployed 13.10 net is the identical technique tuned to D=59; matching its robustness only adds rounds.

## Irreducible-floor analysis
Dominant intermediate = the flood iteration: **D rounds × 2 fp16 25x25 planes (1250B each) = the MaxPool
output + the Min(passable) re-mask.** Both planes are irreducible:
- **2 planes/round is the hard floor.** Masking must happen EVERY dilation step because walls are 1-thick:
  an unmasked MaxPool lights a wall cell, which spreads to the corridor cell behind it in the next pool
  (leak verified — 5x5 / 2-step dilation leaks 376/400). No ONNX op does masked-dilation atomically.
- **fp16 1250B/plane is the dtype floor.** ORT has NO uint8/bool MaxPool or Conv (INVALID_GRAPH), so the
  dilation MUST be float; fp16 is the smallest. uint8 (625B) is unreachable.
- **D≈59–68 is structural.** 8-conn worst-case geodesic = 68 over 40000 samples (margin to ~80). A robust
  net needs D≈72. Even at the deployed net's aggressive D=59, the flood alone is 59×2×1250 = 147.5KB ⇒
  25−ln(147500) = 13.10. To beat by +0.3 needs mem+params ≤ exp(25−13.4)=109KB < the bare 59-round flood.
  ⇒ **mathematically impossible to beat +0.3 with this technique, and it is the only exact technique.**

## OPEN ANGLES (low expected value — flood floor is structural)
- Compact the corridor cells into a dense sub-canvas to shrink planes below 25x25 → BLOCKED: data-dependent
  Slice/compaction leaves symbolic dims (calculate_memory→None trap).
- A sub-1250B {0,1} dilation op (uint8/bool MaxPool/Conv) → BLOCKED by ORT (no such kernels).
- Gap-closer via adopt: the deployed net mis-handles 0.06% of Kaggle; a robust net would raise REAL LB by
  ~0.0006×13 ≈ 0.008 while LOSING ~0.47 stored (13.10→12.63). Net-negative — NOT worth adopting.

## INSIGHT (transferable)
⭐ **Bounded-unrolled flood has a hard floor of `D × 2 × (Wk²·2)` bytes** and that floor frequently EXCEEDS
the deployed net's score, because: (1) masking is required every step (1-thick walls ⇒ no multi-step pool
or post-mask), and (2) ORT forbids uint8/bool MaxPool+Conv so each plane is fp16. For a flood with worst-
case geodesic D, score ≤ 25 − ln(2·D·2·Wk²). For Wk=25, D≥30 already caps below ~14.0; D≥59 caps at ~13.1.
**So flood tasks whose deployed net already uses MaxPool+Min unrolling at the size cap are AT FLOOR — do not
re-attack.** The deployed task286 net is exactly this and is essentially optimal (it even shaves D below the
true worst-case to buy score, eating a ~0.06% generalization miss). 8-conn==4-conn is a real simplification
(one 3x3 MaxPool not a plus) but does not change the 2-plane/round or fp16 floors.
