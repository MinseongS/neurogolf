# task115 — 4be741c5

**Rule:** Input is a small grid (height 8..16) tiled left→right by K consecutive
vertical colour BANDS (K = 3 or 4 distinct non-bg colours). Band boundaries are
noisy — the "gap" columns mix the two neighbouring band colours per row — but every
row contains all K colours in the SAME left-to-right order `colors[0..K-1]`. If
`xpose=1` the figure is transposed (bands run top→bottom). OUTPUT = just that colour
sequence: a 1×K row (non-xpose) or a K×1 column (xpose); all other cells (incl
off-grid) are all-zero, NOT background ch0.
**Current (prior public):** 16.68 pts, ArgMax/TopK/ScatterElements reads of two
directional input slices, mem 4042, params 77.
**Target tier:** B-ish — output colours COPY input colours but band ORDER is a
data-dependent rank + orientation is data-dependent, so no fixed Conv/permute (tier S
blocked); the routing is separable (rank-1 per orientation) so the output never needs
a colour-index plane (escapes the 3600B floor) — but the floor is set elsewhere.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | centroid-rank + full-line orient + 30×30 uint8 label + Equal | B | 21323 | 131 | 15.03 | — | works, heavy |
| 2 | fp16 downstream, spread-based orientation (0/8000) | B | 13263 | 133 | 15.50 | — | |
| 3 | drop profile masks; single fp16 Lf plane + (-1 ch0) Equal | B | 9383 | 132 | 15.84 | — | |
| 4 | MatMul moments (no [1,10,30,1] product); fp32 small results | B | 7275 | 202 | 16.08 | — | |
| 5 | seq via MatMul; present-gate folded into rank=-1 | B | 5456 | 170 | 16.36 | 200/200 | |
| 6 | separable bool And routing in a tiny **4×4** grid + Pad→30×30 | B | 3932 | 100 | 16.70 | 500/500 | **best** |

## Best achieved
**16.70 @ mem 3912 params 100** — adopted? N (orchestrator gates). Beats prior 16.68
by **+0.02** → **MARGINAL** (bar is +0.30). Stored 266/266, ISOLATED fresh 500/500.

## Irreducible-floor analysis
Dominant intermediates: the TWO directional fp32 reductions of the input —
`rowprof32 [1,10,30,1]` and `colprof32 [1,10,1,30]`, 1200 B each = **2400 B**. Both are
required and cannot be removed or narrowed:
  * Orientation is data-dependent and ranking uses the BAND-AXIS centroid, which is
    the cols centroid for non-xpose but the rows centroid for xpose → both row- and
    col-first-moments are needed, each a one-direction weighted reduction of the
    [1,10,30,30] input → an irreducible [1,10,30,1]/[1,10,1,30] fp32 plane.
  * fp32 is mandatory: a row/col moment reaches ~14 000 (>2048), outside fp16's
    integer-exact range, so fp16 would round and silently flip a close band order.
  * Packing both moments into one MatMul `W[1,1,2,30]@input → [1,10,2,30]` is the SAME
    2400 B (no win), and `ReduceSum(input,[2,3])` only gives the unweighted count free.
Everything else is already tiny: routing happens in a 4×4 grid (≤160 B planes), the
rank is a [1,10,1,10] pairwise compare (~300 B), all scalars are [1,10,1,1] (40 B).
mem+params = 4012 ⇒ 16.70. The public net sits at the same wall (4042, reading two
directional slices). To clear +0.3 (≤3045 B) one full profile would have to vanish —
structurally impossible for this rule.

## OPEN ANGLES (re-attack backlog)
- Eliminate one directional reduction: would need orientation + BOTH band orders from a
  single axis read. No exact construction found (row counts carry no left↔right order;
  col counts carry no top↔bottom order). Believed impossible.
- Tier-S impossible (data-dependent order + orientation). Tier-A blocked (output is a
  rank-permutation, not a fixed row⊗col rectangle).

## INSIGHT (transferable)
⭐ ROUTE THE OUTPUT THROUGH A TINY GRID THEN PAD. When the real output occupies only a
small top-left block (here ≤4×4: K≤4 colours on one axis), do ALL the separable
bool-And / label routing in a [1,10,4,4] (≤160 B) space and `Pad(0)` up to
[1,10,30,30] as the FINAL op (output is free). This shrank the routing planes ~7.5×
(300 B → 40 B each) and was the single biggest win (16.36→16.70). ORT `Pad` rejects
bool, so Cast the 4×4 core to uint8 first and declare the graph output UINT8 (the
scorer's `result>0` accepts it).
⭐ CENTROID-RANK for ordered contiguous bands is exact (0/6000) and ORIENTATION via the
spread of per-colour centroids — band axis has large centroid spread, the other axis
≈0 spread because every colour spans its full extent — is exact (0/8000) using ONLY the
[1,10,1,1] centroid vectors (no full has-planes). Fold the present-gate into the rank
by setting absent colours' centroid to +big (so they're never "smaller") and their rank
to −1 (so they match no band position) — removes a separate And-mask plane.
⚠️ Two fp32 directional input reductions (one per axis) is a hard ~2400 B wall for any
rule whose answer needs a per-axis weighted profile on BOTH axes (orientation-dependent
band order). Caps the score ~16.7 — same wall the public net hits. Report MARGINAL fast
once both profiles are confirmed necessary.
