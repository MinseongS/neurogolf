# task084 — 3bd67248

**Rule (generator):** square n×n grid (n=3..21), left column painted colour c (1..9).
Output: col0 (rows 0..n-1) = c; bottom row (r=n-1, cols≥1) = yellow(4); anti-diagonal
(r+c==n-1, cols≥1) = red(2); other in-grid cells = bg(0); off-grid unset. The output is
a FIXED template keyed by two O(1) scalars (size n, colour c) — escape 1/3.

**Prior:** 16.55 @ mem 4362 params 300 (Where(cond, X, input) — full per-row colour plane
X[1,10,30,1]=1200B + three full bool planes yb/rb/cond=2700B; structural over-model).

## Attempts
| # | angle | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|
| 1 | label via int32 additive masks (Mul/Add) | 17045 | 78 | 15.25 | — | 3 full int32 planes (3600 each) blow up |
| 2 | uint8 Where-chain, full 30×30, gated isbot/iscol0 | 7545 | 86 | 16.06 | — | 8 full uint8/bool planes |
| 3 | body(cols1..29)+col0-strip Concat, uint8 | 5590 | 84 | 16.36 | — | 6 full planes |
| 4 | per-ROW base VECTOR + colin-gate (drops ingrid plane) | 3910 | 84 | 16.71 | — | 4 full 30-wide planes |
| 5 | **crop working block to [21,21] (n≤21) + Pad sentinel** | **2911** | **74** | **17.00** | **500/500** | **ADOPTED** |

## Best achieved
**17.00 @ mem 2911 params 74 — beats 16.55 by +0.45.** Fresh ISOLATED 500/500.

## Build (escape 1/3 = scalar-keyed fixed template → free output)
- n = `Sqrt(ReduceSum(input))` (every in-grid cell is one-hot ⇒ Σ = n²; exact for n≤21).
- c = `ArgMax_chan(chmask · ReduceSum(input,[2,3]))` — colour count = n is the unique max
  among non-bg channels (red/yellow only reach n-1; merges harmlessly if c∈{2,4}). 40B, no
  1200B col-0 slice.
- Working block cropped to rows 0..20 / body-cols 1..20 (everything beyond is always
  off-grid). Per-ROW base **vector** rowval[1,1,21,1] (yellow bottom / bg / sentinel) gated
  to in-grid columns by ONE full Where (`base`), red anti-diagonal `Equal(r, n-1-col)`
  overlaid (`body`). col0 is a [1,1,21,1] strip. Concat → Lblk[21,21], Pad→L[30,30] (sentinel
  10 off-grid), `Equal(L, chan-ramp[1,10,1,1])` routes the one-hot into the FREE bool output.

## Dominant intermediate B
L[1,1,30,30] uint8 = **900B** (the post-Pad label). Then Lblk[21,21]=441, and three cropped
[21,20] working planes isred/base/body=420 each. No per-row colour plane, no value plane, no
Where(.,.,input).

## Irreducible-floor analysis
A 30×30 uint8 label plane (900B) is required: output is data-dependent across the whole grid,
and the colour channel c is runtime (any of 1..9) so the 10-ch one-hot cannot be pre-pruned to
a few channels for a task399-style tiny-block Pad (the full [1,10,21,21] bool one-hot = 4410B
≫ 900). The per-cell pattern is inherently 2-D (anti-diagonal r+c==n-1) so ≥1 full comparison
plane (isred) is unavoidable; the colin in-grid gate is also 2-D. Three cropped [21,20] planes +
the 900B label is near the practical floor for a full-grid template task. Param floor is trivial
(74) — points are memory-bound.

## INSIGHT (transferable)
⭐ For a SCALAR-KEYED FIXED-TEMPLATE whose output covers the whole (variable n×n) grid (escape
1/3), the win chain is: (1) recover the O(1) scalars by reductions (n=√Σinput, colour=ArgMax of
masked per-channel counts) — never a full occupancy plane; (2) build a single uint8 LABEL plane
and route the 10-ch expansion into the FREE output via `Equal(L, chan-ramp)` — kills the per-row
colour plane AND the Where(.,.,input) carrier; (3) collapse every separable condition to a row/col
VECTOR and broadcast (per-ROW base value vector + one colin-gate Where replaces a full ingrid AND
+ a full bottom-Where); (4) CROP the working block to the max active extent (n≤21 ⇒ [21,21]) and
`Pad` the always-off-grid border with the sentinel — shrinks every working full plane from 900→420.
Net 4362→2911 (16.55→17.00, +0.45). The 30×30 uint8 label (900B) is the hard floor when the
colour channel is runtime (can't prune channels for a tiny-block Pad).
