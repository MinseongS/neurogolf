# task058 — 28e73c20

**Rule:** INPUT is an all-background square grid of side `size`∈[5,20]; there is
NOTHING spatial to detect. OUTPUT is a deterministic green (colour 3) inward
rectangular involute spiral that is a pure function of `size`: starts at (0,0)
going right, lays the top + right edges, then winds inward with stride 2.
Closed form (exact, all sizes 5..20): with r,c index, e=size-1-r, f=size-1-c,
ring distance d=min(r,c,e,f) — `green = (d even) XOR (r==c+1 AND c==d)` OR the
even-size termination cell `(2r==size AND 2c==size-2)`, then AND in-grid. The
`r==c+1 AND c==d` term is the single per-ring break/connector (gap on even rings,
connector on odd rings) that lives one cell below the diagonal on the left edge.
**Current (public):** 15.31 pts
**Target tier:** B — output is a genuinely 2-D, size-dependent ring pattern; the
ring distance `d=min(r,c,e,f)` couples r&c non-separably, forcing at least one
full 30×30 fp32 plane; no tier-A separable route exists.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 4-arm separable closed form, bool routing | B | 47472 | 194 | 14.23 | — | works, too many planes |
| 2 | same, fp16 single-label Equal | B | 39372 | 195 | 14.41 | — | better |
| 3 | XOR-ring (d even XOR diag-break) + extra cell, fp16 label | B | 25152 | 975 | 14.83 | — | far fewer planes |
| 4 | all-fp32 variant | B | 34152 | 975 | 14.53 | — | worse (more 3600B planes) |
| 5 | computed r==c+1 (drop 900 params) | B | 26052 | 105 | 14.83 | — | params↔mem neutral |
| 6 | nested-Where fp16 label (drops 2 fp32 label planes) | B | 19752 | 106 | 15.10 | 200/200 | BEST |

## Best achieved
15.10 @ mem 19752 params 106 — adopted? **N**. Beats prior 15.31? **N** (−0.21).
Fresh 200/200 exact (isolated, generated inline from the generator).

## Irreducible-floor analysis
Four full 30×30 fp32 planes dominate (4×3600 = 14400): `D` (the ring-min
min(r,c,e,f) — fp16 Min crashes under ORT_DISABLE_ALL so it must be fp32), `dmod`
(D mod 2 for the even-ring test), and the two nested-`Where` label planes (fp16
consts but ORT upcasts Where to fp32 via InsertedPrecisionFreeCast). The rest is
~9 bool 900B planes (comparisons / Ands / Ors). The public net already reaches
15.31; the closed-form structural floor here is ~15.1–15.2, BELOW the public
score. The colour-index plane / ring-min cannot be pushed under fp32, and the
pattern is inherently 2-D (no separable row⊗col escape), so the 3600B plane floor
is real and multiple instances are unavoidable.

## OPEN ANGLES (re-attack backlog)
- Drop the `dmod` plane: compute `d-even` from the separable per-row min(r,e) and
  per-col min(c,f) parities — but parity of a min is not separable, so this needs
  a non-obvious identity; unclear it exists.
- Collapse the two nested-Where label planes into one fused op that emits 3/0/−99
  without an intermediate full plane (no single ORT op does `a*3 + b*(-99)`).
- Gather a precomputed [16,30,30] spiral table by `size-5`: exact and trivial but
  14400 params caps score ≈15.36 (barely +0.05, still <+0.3). Not worth it.

## INSIGHT (transferable)
⭐ A "draw a deterministic shape whose ONLY free parameter is the recovered grid
size" task is a genuine tier-B 30×30-plane floor when the shape is non-separable
(here a ring-distance spiral). The clean closed form for an ARC inward square
spiral is `green = (min-ring-dist even) XOR (one diagonal-break cell per ring)`
plus an even-size termination cell — far cheaper than 4-arm segment unions. But
fp16 buys NOTHING for Min/Mod/Where here (ORT upcasts all three to fp32 under
ORT_DISABLE_ALL), so the floor is set by the COUNT of full fp32 planes, not their
dtype. nested-`Where(cond, a, Where(cond2, b, c))` is the cheapest 3-value
colour-index label (2 planes) vs arithmetic `a*k1 + b*k2` (3+ planes + cast
upcasts).
