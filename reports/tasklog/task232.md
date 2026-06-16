# task232 — 97999447

**Rule:** Input is a W×H grid (W,H ∈ 7..14) with a few single coloured pixels, one per
distinct row, each in the left half (col ≤ width//2), colours random non-gray. For every
coloured pixel at (row, col, color) the output paints a horizontal trail from col to the
right grid edge: cell c gets `color` when c has the same parity as col, else gray(5).
Cells left of col, rows with no pixel, and off-grid cells stay background(0). Each row is
independent and holds at most one pixel.
**Current (stored before):** ~14.71 pts, public net, tier A.
**Target tier:** B (label-map + final Equal) — the per-cell colour depends on per-row
start-col parity (couples r and c) so it is NOT a pure row⊗col separable rectangle (tier A
blocked); a single Conv can't route random per-instance output colours (tier S blocked).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | colf Conv plane + label map, 2 slices | B | 11076 | 67 | 15.68 | — | works |
| 2 | merge row+col slice into one | B | 9396 | 68 | 15.84 | — | works |
| 3 | replace 30×30 colf plane with 1-D reductions + per-channel MatMul | B | 7032 | 365 | 16.09 | — | works |
| 4 | merge color+start into ONE [1,10,30,2] batched MatMul | B | 5944 | 659 | 16.20 | 200/200 | **best** |

## Best achieved
**16.20 @ mem 5944 params 659** — adopted? N (orchestrator gates). Beats prior 14.71 by
**+1.49** (≥+0.3 ✓). Stored 266/266, fresh ISOLATED 200/200.

## Irreducible-floor analysis
Dominant intermediate: the batched MatMul output `[1,10,30,2]` = 2400 B fp32. This is the
per-channel, per-row column-contraction that recovers BOTH per-row scalars at once
(col 0 = k·rowcount → colour, col 1 = Σ_c c·input → start-col). It is the price of needing
two independent per-row quantities; collapsing channels first would require the 3600 B
[1,1,30,30] colf plane, and casting input to fp16 to halve the MatMul costs 18000 B. The
final label plane `[1,1,30,30]` uint8 = 900 B (padded from a 14×14 working canvas) is the
other floor item. Everything else is ≤ 392 B (14×14 bool/uint8 masks built from broadcast
1-D vectors). Net ~5944 — near floor for a B-tier label-map task with two per-row params.

## OPEN ANGLES (re-attack backlog)
- Single-Conv tier-S attempt is blocked by random per-instance colours, BUT the parity
  structure is fixed: a Conv that emits a parity-banded label (e.g. 100·active + parity)
  then a per-row colour Gather might shave the MatMul — unlikely to beat 2400 B though.
- Drop the [1,10,30,2] to [1,10,30,1] if start-col could be inferred from colour+geometry
  (it can't here — start-col is independent random).
- Investigate whether ORT keeps the L chain in uint8 vs the static fp16 mislabel; if a
  genuine fp16 leak exists, forcing uint8 could save ~900 B.

## INSIGHT (transferable)
⭐ Two independent per-row scalars (here colour and start-column) can be recovered in ONE
batched `MatMul(input[1,10,30,30], W[1,10,30,2])` over the FREE fp32 input: pack each
scalar as a column of the contraction weight (col0 = per-channel constant k for a
k-weighted count; col1 = column-index ramp gated to k≥1 for a position centroid), then
`ReduceSum` over channels and Slice the [.,.,.,2] result into the two vectors. This
eliminates BOTH the 3600 B [1,1,30,30] colf plane and a second 1200 B contraction —
generalises any "recover N per-row params, then label-map" task. Parity coupling
(c%2 == start%2) is built separably as `Equal(Mod(col,2)[1,1,1,W], Mod(start,2)[1,1,W,1])`.
