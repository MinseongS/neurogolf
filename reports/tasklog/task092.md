# task092 — 40853293

**Rule:** A (10/20/30)x(10/20/30) grid holds up to 5 axis-aligned "sticks", each a DISTINCT
colour, drawn in the INPUT as their TWO endpoint pixels only. A stick is HORIZONTAL (both
endpoints in the SAME row, >=2 cols apart) or VERTICAL (both in the SAME column, >=2 rows
apart). The OUTPUT fills the whole segment between (and incl.) the two endpoints. Horizontal
sticks occupy distinct rows; vertical sticks distinct cols; a horizontal and vertical stick
may CROSS — at a crossing the VERTICAL stick's colour wins (generator draws all horizontal
sticks first, then all vertical, so the column over-writes). Verified 0/500 fresh:
rowfill_k = rowPrefixOR ∧ rowSuffixOR; colfill_k = colPrefixOR ∧ colSuffixOR; col wins.
**Current:** 14.56 pts, gen draft (this file's prior version), mem 32350, params 1830
**Target tier:** B/A — separable closed-form interval fill via 1-D row/col occupancy profiles
+ triangular prefix/suffix-OR; the per-cell colour-index plane is the floor.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | prior draft: per-ch tri-MatMul on 1-D profiles, fp32 ingrid ReduceMax + uint8 Where chain | B | 32350 | 1830 | 14.56 | — | stored baseline |
| 1 | drop fp32 [1,1,30,30] in-grid ReduceMax; recover H/W via ReduceMax(rowhas, ch-axis) | B | 29110 | 1829 | 14.66 | — | -3240B |
| 2 | fuse Sub+Mul+Add+Where sentinel into ONE Where(ingrid_bool, stickColor, 10) | B | 24670 | 1828 | 14.82 | — | -4440B |
| 3 | sentinel via 1-D offrow/offcol penalties in ONE variadic Sum (drops And bool plane) | B | 23950 | 1829 | 14.84 | — | -720B |
| 4 | **col prefix/suffix as RIGHT-multiply on [1,10,1,30] — kills the colhas Transpose** | B | 23350 | 1829 | **14.866** | 500/500 | best |

## Best achieved
14.866 @ mem 23350 params 1829 — adopted? N (per instructions, not self-adopted).
Beats prior 14.56 by **+0.306** (≥+0.3). official 3/3, fresh 500/500.

## Irreducible-floor analysis
Memory now bound by FOUR [30,30] f16 planes (1800 each = 7200): the two colour MatMuls
(rowColor, colColor), the col-priority Where (stickColor), and the off-grid sentinel Sum (L2f).
Plus two fp32 ReduceMax occupancy reductions ([1,10,30,1]/[1,10,1,30] = 1200 each) — FORCED
because ORT ReduceMax rejects bool/uint8 and the input is fp32. The two colour MatMuls are
irreducible: a crossing cell is covered by a row-stick AND a col-stick, so a single linear
channel-contraction would SUM the two colours; the split (iscol/isrow weights) + a Where is
required to realise the "col wins" priority. ~14.9 is the practical floor for this two-plane
colour-priority fill.

## OPEN ANGLES (re-attack backlog)
- Collapse the col-priority Where: encode col colours at magnitude 10·c and Max() with the
  row plane (col always wins), then decode c=combined/10 for col cells — but decode needs a
  Where anyway, so net wash unless the decode folds into the final Equal arange (untried).
- Eliminate one fp32 ReduceMax (2400B total): both row- and col-occupancy come from the same
  input; no single reduction yields both, but a per-channel batched MatVec contracting the
  free fp32 input directly (guide's MatMul(input,vec) lever) might dodge one fp32 transient.
- Merge the two colour MatMuls into one batched [2,30,30] MatMul + Slice (fewer nodes, same
  bytes) — no mem win but cleaner.

## INSIGHT (transferable)
⭐ The axis-aligned analog of task037's diagonal endpoint-fill collapses to 1-D row/col
occupancy profiles (ReduceMax over the col/row axis → [1,10,30,1]/[1,10,1,30]), so the fill
is a triangular prefix∧suffix-OR on tiny per-channel VECTORS (params, not 30×30 planes), and
the colour is recovered by a [channel]×[30] MatMul contraction — never a [1,10,30,30] plane.
⭐ Do col-direction prefix/suffix-OR as a RIGHT-multiply (colhas[1,10,1,30] @ Tri[30,30])
instead of transposing to [.,30,1] and left-multiplying — saves the whole Transpose plane.
⭐ Off-grid sentinel without an in-grid 30×30 plane: add 1-D penalties offrow[r]=10·(r>=H),
offcol[c]=10·(c>=W) to the colour plane in ONE variadic Sum (broadcast) — off-grid cells go
>=10 so Equal(L, arange[0..9]) matches nothing (all-zero target), in-grid background stays 0
(ch0=1). Recover H/W as ReduceMax of per-channel occupancy over the CHANNEL axis (bg ch0=1
fills every in-grid cell), no separate frame ReduceMax over the full input.
