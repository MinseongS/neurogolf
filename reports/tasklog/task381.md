# task381 — ef135b50

**Rule:** size=10 grid, red(2) boxes on black(0).  For each row, the maximal run
of non-red cells BETWEEN two red cells is painted maroon(9) UNLESS any cell in
that run has a red directly above/below (then the whole run stays black). Red
copies through; outside the 10x10 active region is black.  Generator validation
forbids a red-black-red pattern on a row and forbids maroon in the top/bottom
rows.

**Current (prompt P):** 16.75 pts.  Adopted net in manifest: 16.84
(`ext:ghiotto_conv4`, mem 3350, params 146 — Slice/MaxPool/per-row ReduceMax).
**Target tier:** B (label-map + Equal; output is a whole-row reduce, not a
per-cell neighbourhood function, so no single-Conv Tier-S).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | iterative radius-1 run-spread of "danger" within non-red runs (8 then 7 iters) + label/Equal | B | 6200 | 433 | 16.20 | 500/500 | correct but 14 spread planes dominate; below P |
| 2 | doubling spread (offsets 1,2,4,8) | B | 6200 | 833 | 16.14 | 126/200 | WRONG — offset-o jump leaks across a single red separator |
| 3 | PER-ROW block test + "red above" only (no spread) + label/Equal | B | 3250 | 44 | 16.90 | 500/500 | best; beats P, edges adopted 16.84, short of +0.3 |

## Best achieved
16.90 @ mem 3250 params 44 — adopted? N (agent does not adopt). Beats prior P
16.75? YES (+0.15). Beats adopted 16.84? marginally (+0.06). Reaches +0.3
(17.05)? NO → MARGINAL.

## Irreducible-floor analysis
mem 3250 = L[1,1,30,30] uint8 label 900 (output is 30x30 → 30x30 label floor for
the final Equal) + ch2 fp32 Slice 400 (Slice preserves input fp32; the red plane
must be float for MaxPool/Conv) + two directional 1x10 MaxPools 400 (need both
"red-left" and "red-right" for the between-two-reds span) + 3 fp16 work planes
600 (ch2f, above_f, blocked_f) + ~8 bool 10x10 planes ~800 + tiny per-row vecs.
The 1700B of L+slice+MaxPools is structural.  To reach 17.05 needs ≤2840 total
(−410): no plane removal found that keeps exactness (Concat-output is 1000B >
900B L and trips ORT's bool-Pad rejection without sanitize; maroon-Where output
needs a 30x30 bool which costs a 1800B pad+cast).

## OPEN ANGLES (re-attack backlog)
- Single fused op for the "between two reds" span replacing the two MaxPools
  (−~300B) — e.g. one cumulative-red signal whose sign distinguishes left/right
  simultaneously; not found exact yet.
- A 30x30 label below 900B (output shape is fixed 30x30 so likely impossible
  without a sub-uint8 carrier, which ORT upcasts).
- Confirm whether the official sanitize_model path makes bool Concat→Pad legal
  (the adopted ghiotto net uses it) — if so a Where(maroon30,onehot9,input)
  output could drop the Lm/L10 label-build, but still needs a 30x30 plane.

## INSIGHT (transferable)
⭐ "Fill the run BETWEEN two markers, whole-run gated by a per-cell predicate"
is NOT necessarily an iterative-flood / per-run-reduce wall: if the GENERATOR
VALIDATION forbids re-entrant patterns on the scan axis (here: red-black-red is
rejected) then a PER-ROW (per-line) ReduceMax of the block predicate is exactly
equivalent to the per-run reduce, collapsing an O(width) radius-1 spread (14
planes, ~2800B) into ONE reduce. Also: checking only "red ABOVE" (not above OR
below) sufficed because the top-row exclusion + row symmetry make the two
directions redundant under the validator. DISCRIMINATOR vs a true flood: read
the generator's reject/validate clauses, not just the draw loop — they often
constrain the input distribution enough to make a global reduce exact.
Anti-lever: gap-DOUBLING (offset 2^i) LEAKS across a single barrier cell (jumps
over a red onto a non-red in the next run); barrier-bounded spread must step by
radius 1.
