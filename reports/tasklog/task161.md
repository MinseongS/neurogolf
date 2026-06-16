# task161 — 6cdd2623 (laser rows/cols)

**Rule:** H×W grid (W 15..25, H 10..20) with scattered pixels in 2 palette colours
plus "laser" markers in a 3rd colour (megacolor). A laser ROW r (1≤r≤H-2) has the
megacolor at grid[r][0] and grid[r][W-1]; a laser COL c (1≤c≤W-2) has it at
grid[0][c] and grid[H-1][c]. The megacolor appears ONLY at those border endpoints,
never in the interior. Output paints the ENTIRE laser row / laser column with the
megacolor; every other in-grid cell is background (0); off-grid empty.
**Current (prior):** 14.67 pts, ext:thbdh6285, mem 30554, params 78
**Target tier:** B (label map). NOT pure Tier A: the output one-hot needs channel-0
(background) = 1 at every in-grid non-laser cell, so a `paint AND mc_onehot` separable
form is insufficient — the standard L→Equal(L,arange) label map is required.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | separable paint·mc_onehot (no ch0 bg) | A | — | — | 0 | fail | WRONG: target ch0=1 at in-grid bg |
| 2 | label map, dynamic Slice for W-1/H-1 | B | None | — | 0 | — | "perf could not be measured": runtime-Slice → symbolic dims |
| 3 | label map, Gather lines, matched-pairs megacolor | B | 20531 | 97 | 15.07 | 4/4 | works; heavy |
| 4 | fp16 lines + gather-channel laser masks | B | 12603 | 84 | 15.55 | 200/200 | leaner |
| 5 | presence-both-ends megacolor (drop per-row products), fp32 lines, 4-plane L | B | **10293** | **84** | **15.7527** | **500/500** | **best** |

## Best achieved
**15.7527 @ mem 10293 params 84 — adopted? N (orchestrator gates).**
Beats prior 14.67 by **+1.08** (≥+0.3 ✓). Generalizes: fresh 500/500 + stored 266/266.

## Key construction
- in-grid extent H,W = ReduceSum of ReduceMax-occupancy (off-grid is all-zero one-hot).
- 4 border one-hot lines via **Gather** (scalar indices squeezed from [1] inits so every
  Gather drops its axis to a consistent rank-3 [1,10,30]; runtime-tensor Slice would leave
  symbolic dims → harness can't measure memory).
- megacolor = colour present at BOTH ends of a line ((L∧R)∨(T∧B)) AND not present in the
  strict interior. Presence = ReduceMax per line ([1,10]); interior = total−ring_count(>0)
  with 4-corner double-count correction. 0 err / 12000 numpy + 500/500 ONNX fresh.
- laser masks: Gather the megacolor CHANNEL out of the col-0 / row-0 one-hot lines (no
  colour Conv, no k-weighted reduce).
- L = Where(laser, mc, 0) then Where(ingrid, ·, 10) → Equal(L, arange[0..9]) → BOOL output.

## Irreducible-floor analysis
Two floors dominate the 10293:
- **4×[1,10,30] fp32 Gather lines = 4800.** Gather of the fp32 input yields fp32; all 4
  sides genuinely needed (presence both-ends, interior ring counts, laser col-0/row-0).
  Collapsing to a single-channel colf[1,1,30,30] (3600) + index lines would re-add 3600
  AND lose the per-colour interior/presence counting that needs one-hot → no net win.
- **4×[1,1,30,30] label planes = 3600** (linemask Or, ingrid And, inner Where, L Where),
  all bool/uint8 (1 B/elem, already minimal). A laser row mask is per-row [1,1,30,1] and
  must be clipped to cols<W → needs the in-grid 30×30 conjunction; 3-plane variants leak
  megacolor into off-grid columns of laser rows (verified failure). 4 planes is the floor.

## OPEN ANGLES (re-attack backlog)
- Eliminate one label plane if a way is found to clip the broadcast laser row/col to the
  in-grid rectangle without a full [1,1,30,30] And (1-D clips reintroduce a plane each).
- Fuse the 4 fp32 Gathers — no opset-11 op gathers two non-adjacent axes' end-slices at once.

## INSIGHT (transferable)
⭐ Two reusable levers confirmed here:
1. **"border-only colour at both line-ends" beats per-row matched-pair products** for
   isolating a marker colour: ReduceMax-presence per side ([1,10] each) ANDed pairwise, gated
   by (total−ring_count>0)=interior. All tiny [1,10] tensors, no [1,10,30] products.
2. **Recover a per-position boolean for a known scalar colour by Gathering that colour's
   CHANNEL** out of a one-hot line (`Gather(line[1,10,30], mc_idx, axis=1)`) instead of
   building a colour-index plane and comparing — skips the k-weighted channel reduce.
Also: runtime-tensor **Slice leaves symbolic output dims** → `calculate_memory` returns None
("performance could not be measured"); use **Gather with squeezed scalar indices** for
data-dependent border lines so shapes stay static. And: scoring compares the FULL 10-channel
one-hot — channel 0 must be 1 at in-grid background, so background-bearing outputs are
label-map (Tier B), not separable paint masks.
