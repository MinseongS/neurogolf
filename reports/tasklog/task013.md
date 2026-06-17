# task013 — 0a938d79

**Rule:** Grid W×H (W∈20..30, H∈6..12). Two seed pixels: `grid[bottoms[0]*(H-1)][start]=colors[0]`,
`grid[bottoms[1]*(H-1)][start+sep+1]=colors[1]`, start∈[1,W//2], sep∈[1,5] (period p=sep+1∈[2,6]).
Output paints FULL vertical stripes (whole columns) at start, start+p, start+2p, … (<W), alternating
colors[0],colors[1],colors[0],… If `xpose`, transpose (stripes become full rows).
**Current (prior):** 15.43 pts.
**Target tier:** A/B — closed-form separable reconstruction; no flood-fill, no global argmax.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | separable: per-axis colour profiles → period scalars → fp16 Lcolor plane → Equal→bool output | A/B | 13581 | 65 | 15.48 | 200/200 | fp32 Lcolor plane dominated |
| 2 | cast perpendicular vecs to fp16 BEFORE the broadcast Where (Lcolor born fp16) | A/B | 10101 | 65 | 15.77 | 500/500 | adopted |

## Best achieved
15.77 @ mem 10101 params 65 — beats prior 15.43 by **+0.34** (≥+0.3 ✓).

## Key construction
- Orientation: **xpose=1 IFF both seed columns ∈ {0, W-1}** (verified 0/3000 mismatches; in xpose=0
  seed cols are start≥1, never 0, never both at edges).
- Period-axis colour-weighted profile pval (colval for xpose=0 / rowval for xpose=1) — each seed sits at
  a distinct position (p≥2), so the profile value IS the seed colour index.
- firstpos/lastpos via masked-arange ReduceMax / Where-BIG ReduceMin; p=last-first; colours via Gather.
- stripe colour: m=(t-firstpos) mod 2p (fp16 Mod, exact); m==0→c0, m==p→c1, else bg=0; gate t≥firstpos.
- pvec[30] placed as colvec[1,1,1,30] or rowvec[1,1,30,1], **cast fp16 then ONE Where broadcasts the two
  perpendicular vectors to the [1,1,30,30] Lcolor plane directly in fp16** (the key 3600→1800 win).
- in-grid rect mask (rowin∧colin); off-grid→ -1 so `Equal(L16, arange[1,10,1,1])` → all-zero off-grid,
  channel-0 for in-grid bg, routing the 10-ch expansion into the FREE bool output.

## Irreducible-floor analysis
Dominant intermediates: two fp16 [1,1,30,30] planes — Lcolor16 (1800B, the broadcast build) and L16
(1800B, the off-grid-sentinel Where) — plus the in-grid bool plane (900B) and the two per-channel
column/row profiles perch_col/perch_row ([1,10,1,30] & [1,10,30,1], 1200B each). The two perch planes
are needed because pval=Where(xpose,rowval,colval) requires BOTH colour profiles in a static graph;
the second fp16 plane is the off-grid sentinel gate (a 2-D gate the perpendicular vectors can't encode).

## OPEN ANGLES (re-attack backlog)
- Collapse Lcolor16+L16 into one plane: bake the period-axis extent (-1 for ≥extent) into pvec and apply
  only the perpendicular gate, but the orientation-select of the perpendicular gate itself materializes a
  [1,1,30,30] bool — net-neutral as tried in head; may yield ~900B if the gate can ride the FREE output.
- Replace perch_col (1200B) with cheap colored-occupancy [1,1,1,30] (120B) for orientation only — but the
  colour VALUES still require a full profile on the chosen axis; no win unless orientation can be resolved
  before computing any colour profile (circular today).

## INSIGHT (transferable)
⭐ When a per-cell colour-index plane is built by broadcasting TWO PERPENDICULAR vectors via a single
Where(scalar, rowvec[1,1,30,1], colvec[1,1,1,30]), CAST THE VECTORS TO fp16 BEFORE the Where so the
[1,1,30,30] plane is born fp16 (1800B) — casting AFTER leaves the fp32 3600B plane in the trace
(task013: 13581→10101, +0.29 pts). Orientation (xpose) for "seeds-on-perpendicular-edges" tasks is
cleanly `both perpendicular-axis seed coords ∈ {0, extent-1}`, no spread/argmax heuristics.
