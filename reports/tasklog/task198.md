# task198 — 83302e8f

**Rule:** A `size x size` cell grid (size in 3..7); each cell is `minisize x minisize` pixels
(minisize in 3..5), cells separated by single 1-px lines of `color`. pitch p=minisize+1,
actual_size = size*p-1 (<=29). Cell interiors default GREEN(3). "Permeable points" are black(0)
pixels sitting ON a line (where the line colour would otherwise be). A black-on-vertical-line
pixel (c%p==p-1, r%p!=p-1) connects cells (R,C) and (R,C+1) -> both YELLOW(4). A black-on-
horizontal-line pixel (r%p==p-1, c%p!=p-1) connects (R,C),(R+1,C). A pixel exactly on a line
crossing triggers nothing. Output keeps the line colour on every line position EXCEPT permeable
points (which become YELLOW), and cell interiors are YELLOW if their cell got marked else GREEN.
**NO flood-fill** — fully local closed-form (the BUILD_PROMPT "task198 infeasible / flood wall"
note was a MISCLASSIFICATION; the generator does a single direct marking pass, no propagation).
**Current (prior):** 14.203 pts, ext:thbdh6332, mem 47510, params 1345
**Target tier:** A (separable cell-space marking via selector MatMuls; no per-cell colour Conv).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | cell-space mark + double-MatMul down/upsample, route 10-ch to FREE bool out | A | 32712 | 133 | (logic bugs) | — | fixed ch0/extent/line-detect/sentinel |
| 2 | drop `anyset` full plane: extent+nonblack via direct axis reductions | A | 30460 | 135 | 14.671 | 200/200 | ADOPTED |

## Best achieved
14.671 @ mem 30460 params 135 — beats prior 14.203 by **+0.468** (well above +0.3 bar). Fresh 200/200.

## Irreducible-floor analysis
Dominant intermediate: ONE fp32 3600B black plane `ch0` (Slice of input channel 0 = in-grid
black; off-grid is all-zero so ch0==1 iff in-grid-black — irreducible since a per-pixel black
mask requires materializing a channel as fp32). Remaining cost is ~7 fp16 1800B full planes
(Gv/Gh gap masks, Ypix upsample, interiorL/lineL/gridL/L composition) + ~9 bool 900B broadcast
planes (online/site/perm/ingrid). The composition Wheres and the broadcast bool planes are the
next target but each tried fold either kept the plane count flat or broke off-grid line handling.

## OPEN ANGLES (re-attack backlog)
- Pack Gv+8*Gh into ONE fp16 plane + ONE pair of MatMuls, recover Vg/Hg by magnitude bands in
  the tiny SxS cell space (verified max gaps/cell = 4 < 8, mutually exclusive) — saves one 1800B
  fp16 plane (~ -0.05 pts). Skipped for complexity vs marginal payoff.
- Collapse the 4 composition Wheres (interiorL/lineL/gridL/L) — every nested-Where variant tried
  stayed at 4 planes or mishandled off-grid line positions (rmodp==p-1 fires off-grid -> must
  gate online by ingrid). Worth one more pass if chasing Tier-A floor.

## INSIGHT (transferable)
⭐ A "flood-fill / variable-size region" classification can be WRONG — re-read the generator for
an explicit propagation LOOP before bailing. task198's "permeable line-grid" looked like a flood
wall but is a one-pass local marking: each gap paints its 2 adjacent cells, computed exactly as a
separable cell-space double-MatMul (Rsel@G@Csel down, Rsel^T@Y@Csel^T up) keyed on Cidx=floor(c/p).
⭐ Off-grid pixels are ALL-ZERO across channels (NOT background ch0=1) — so `ch0==1` cleanly means
"in-grid black", but any positional mask (r%p==p-1) FIRES off-grid and must be gated by an ingrid
mask; route off-grid to a sentinel (99) so Equal(L,arange) yields all-False, matching the all-zero
target (NOT channel-0=True). ⭐ Line-pitch p recovers as a robust scalar via a COUNT threshold
(line-row nonblack >=10 vs cell-row <=size-1<=6) rather than exact "==W" (permeable gaps break ==W).
