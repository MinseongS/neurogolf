# task243 — 9edfc990

**Rule:** size in [12,18]; grid filled with random colours (~50% black). Output copies
the input, then every BLUE (1) cell floods blue into ALL 4-connected cells currently 0
(black), propagating: any newly-blued black cell spreads to its black neighbours. Net:
every black cell 4-connected through black to a blue seed becomes blue; unreachable black
stays black; non-black colours are unchanged.

**Current (prior):** 13.86 pts, uint8 MaxPool flood (opset 18), mem 68616, params 102.
**Target tier:** detection/flood — genuine 4-connected flood-fill; the iterated dilation is
structural. Goal was constant-factor reduction of the established flood net.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | clean re-impl: single colf plane + uint8 cross-flood (re-inject) | flood | 66672 | 58 | 13.89 | 200/200 | correct, ~baseline |
| 2 | drop seed re-injection (Max(gated,seed)) | flood | 57600 | 46 | 14.04 | 400/400 | win |
| 3 | Conv on full input then crop 1-ch (kill 12960B 10-ch fp32 crop) | flood | 51840 | 58 | 14.14 | 200/200 | win |
| 4 | sequential alt V/H gated (2 planes/step) instead of cross | — | — | — | 0 | 27/400 | FAIL (loses multi-step same-dir runs) |
| 5 | single Conv weight=100+k (colf carries colour AND in-grid; kill occ conv + off-grid Where) | flood | 45324 | 48 | 14.28 | 200/200 | win |
| 6 | blue seed = (colf==101) reuse, drop blue slice; remove unused inits | flood | 44352 | 37 | 14.30 | 1000/1000 | **adopted** |

## Best achieved
14.30 @ mem 44352 params 37 — adopted? write-only (per instructions). Beats prior 13.86? **YES (+0.44)**.

## Irreducible-floor analysis
Flood dominates: 28 cross-dilation iters × 4 uint8 18×18 planes (pv, ph, Max, Min) = ~36KB
(80% of mem). 4-connectivity requires a cross dilate (2 MaxPool + Max) + a re-gate to the
black region (Min) per step; 1-cell walls force radius-1, and depth-28 is data-required
(official arc-gen has cases needing exactly 28; K<28 fails 1–2 official cases). Entry colour
plane is one unavoidable 3600B fp32 Conv. The flood is a structural floor — no sub-floor
escape (it is a true bounded BFS, not a collapsible separable/count form).

## OPEN ANGLES (re-attack backlog)
- Fewer planes/iter (<4): needs combining cross-Max with the black-gate-Min in one op —
  no ORT op does masked-dilation in a single pass; sequential V/H gating is INCORRECT
  (attempt 4, 27/400). Believed at the per-step floor.
- Fewer iters: depth is data-required (28). A larger MaxPool kernel leaks across 1-cell
  walls. No distance-doubling possible with walls.
- Tighter dynamic crop per-instance: blocked (static shape; data-dependent crop trips the
  symbolic-dim "could not be measured" trap).

## INSIGHT (transferable)
- ⭐ uint8 MaxPool/Max/Min REQUIRE opset>=18 (opset-11 MaxPool type-constraint rejects uint8
  → INVALID_GRAPH at load). The scorer checks DOMAIN not VERSION, so declaring opset 18 is
  free and unlocks the whole uint8 flood pipeline (1B/plane vs fp16 2B). The prior accepted
  243 net already used opset 18 — match it.
- ⭐ A SINGLE colour Conv with weight (BASE+k) packs colour-index AND in-grid-occupancy into
  one fp32 plane: off-grid one-hot is all-zero → Conv=0 → never equals any BASE+k channel
  constant at the final Equal, so off-grid auto-produces NO channel with ZERO extra ops (no
  occupancy conv, no sentinel Where). Saved a whole 3600B conv + Slice + Greater + a Where.
- ⭐ For a 4-connected gated flood, DROP seed re-injection: re-gating to the free region
  drops the (non-free) seeds, but their free neighbours captured on step 0 carry the BFS
  front, and seeds keep their colour via the underlying index plane in the final label.
  One fewer Max plane per iteration (~9KB here).
- Conv on the FULL 30×30 input then Slice the 1-channel result to the active region beats
  Slicing the 10-channel fp32 input first (3600B+1296B vs a 12960B 10-ch crop).
- Sequential alternating V/H gated dilation is NOT equivalent to cross-dilation for flood
  (it cannot advance corridors needing consecutive same-direction steps) — keep the
  symmetric cross (2 MaxPool + Max) per step.
