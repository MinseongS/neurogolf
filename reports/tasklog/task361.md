# task361 — e40b9e2f

**Rule:** Input is a PARTIAL "pinwheel": a subset of a figure with 4-fold (90°)
rotational symmetry about a centre (R+b/2, C-b/2) (b = bump ∈ {0,1}). The OUTPUT
always places the FULL orbit: `output = input ∪ rot90 ∪ rot180 ∪ rot270` about
that centre. In integer terms the 90° forward rotation is (Y,X)→(s1+X, s2−Y) with
s1=R−C+b, s2=R+C. The centre is recovered EXACTLY by the (s1,s2) maximising the
number of input pixels whose orbit is in-grid with matching colour; a 2-step
AND-chain argmax is exact (verified 20000/20000). The completion is then a
first-non-zero over the 3 rotated source colours of the selected centre.
**Current (public):** 15.35 pts, ext:biohack_new.
**Target tier:** detection (non-local symmetry centre-finding) — NOT S/A/B: the
output is the union of 4 input-content-dependent rotations about a data-dependent
centre that must be SEARCHED for (no fixed Conv/permute/separable form).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | full-grid 90-cand × 100-cell orbit search, int32 indices | det | 1.2M | ~800 | 10.99 | — | over floor |
| 2 | precompute orbit index tables (params instead of arith) | det | — | — | — | — | params = same wall |
| 3 | window-relative (9×9, 25 const-orbit cands), arith completion | det | 33.6k | 3.0k | 14.60 | — | better |
| 4 | two-window (7×7 search 49 cells / 9×9 complete 81), 13 real cands | det | 25.6k | 1.8k | 14.78 | — | better |
| 5 | + fp16 index math, base-mask folds occupancy, Conv-then-slice | det | 22.8k | 2.0k | 14.94 | — | better |
| 6 | + precomputed SRC completion tables, unused-init cleanup | det | 17.3k | 4.98k | **14.99** | 2000/2000 | EXACT but below floor |

## Best achieved
**14.99 @ mem 17313 params 4979 — fresh 2000/2000 EXACT. Adopted? N.**
Does NOT beat public 15.35 (it is 0.36 BELOW). MARGINAL → do not adopt.

## Irreducible-floor analysis
The solution is exact and fully generalising but the SCORE floors ~22000
(mem+params) → ~15.0 pts, ~tied with the already-near-floor public net.
Dominant, irreducible costs:
- **Vbig 3600 B** (fp32 [1,1,30,30] colour-index Conv output). The colour index
  needs the 30×30 plane; Conv-then-slice (3600) is the cheapest path — slicing the
  10-channel corner first costs 4000 (incorner) and any fp16 route needs the fp32
  corner to exist first. Structural.
- **search block ~3.2k** = col0/col1/eq0/eq1/good01 [13,49] + goodf fp16. This is
  the brute-force argmax: candidates(13) × cells(49). The Gather index MUST be
  int32, so even one candidates×cells tensor is large; this is the detection floor.
- **ORB 1274 + SRC 3159 params** — constant orbit/source index tables (params count
  by ELEMENT COUNT regardless of dtype, so they cannot be shrunk by dtype).
- L 900 (final label map), ScatterND int64 index 648, window gathers ~2k.
The genuine wall: an EXACT solution must (a) search candidate centres × cells and
(b) carry an int32 index over that product — there is no separable/closed-form
centre recovery (median(y±x) ≈ 2650/3000; bbox/centroid all fail), so the search
cannot be removed. Beating 15.35 by +0.3 needs mem+params ≤ 11342, ~half the floor.

## OPEN ANGLES (re-attack backlog)
- Closed-form centre from a cheap robust statistic (would delete the whole search
  → could reach Tier-B ~16.8). Tried median(y+x)/median(y−x) (2650/3000), bbox-sum,
  centroid, 180°-only argmax (2665/3000), single rot90 step (4948/5000) — none
  exact. If a closed form exists it is the only path past the floor.
- Eliminate Vbig: a Conv variant that emits only the 10×10 corner (stride/region)
  would save 3200 B but ORT Conv has no crop. ~0.15 pts.
- Drop ScatterND: Pad the completion window to 30×30 with a dynamic offset (no ORT
  op does dynamic-offset Pad cleanly).

## INSIGHT (transferable)
⭐ A "complete the rotational symmetry" task = union of 4 rotations about a
data-dependent centre; the centre is found by **orbit-consistency argmax** (the
(s1,s2) maximising input pixels whose rotation orbit is in-grid & same-colour) —
a 2-step AND-chain is exact where naive 1-step / 180°-only / median heuristics are
not. Big compaction lever for these searches: **window-relative coordinates** —
translate to the bbox-centre so candidate centres collapse to a small FIXED set
(here 90→13) and the orbit permutations become CONSTANT index tables (no runtime
index arithmetic). Use a tight 7×7 SEARCH window (input cells) + a 9×9 COMPLETION
window (symmetric output radius ≤4). KEY scorer fact relearned: **params are
charged by ELEMENT COUNT, not bytes** — so precomputed int32 index tables cost the
same as uint8, and a candidates×cells search has a hard floor: even the single
mandatory int32 Gather-index over that product blew past the public floor. This
confirms the briefing: non-local detection tasks whose public net is already near
the memorizer/detection floor are MARGINAL — an exact net can be built and
generalises 100%, but cannot beat the floor by +0.3.
