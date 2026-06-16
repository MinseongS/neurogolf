# task365 — e50d258f

**Rule:** A 10x10 grid holds 2-3 solid, gap-separated, axis-aligned rectangles ("boxes")
filled with blue(1)/cyan(8) plus a few red(2) cells. Each box has a DISTINCT red count
(sampled without replacement from {1,2,3,4}); the output is the box with the MOST reds,
cropped to its bounding box at the top-left of a fresh grid (rest all-zero / off-grid).
**Current (prior):** 14.81 pts, gen:vyank6322, mem 26613, params 68
**Target tier:** detection/B — needs per-box red ARGMAX + variable-size crop; selection is a
global argmax over data-dependent-count components, so it lands in the run-sum / detection band
(B-ish), NOT a clean separable (A) or single-op (S) rule.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | linear 5-step run-sum recurrence (L,R,U,D), winmask, gather-crop | B/det | 21914 | 474 | 14.98 | 266 stored | works, heavy |
| 2 | + segmented-doubling run-sums (offsets 1,2,4) | B/det | 20314 | 674 | 15.05 | — | leaner |
| 3 | + fp16 colour slice, drop redundant *nb / And / eqmax masks | B/det | 19614 | 674 | **15.08** | **200/200 + 500/500** | best, generalizes |

## Best achieved
15.0822 @ mem 19614 params 674 — adopted? N (orchestrator gates). Beats prior 14.81 by **+0.27 (MARGINAL, <+0.3)**.
Fresh ISOLATED 200/200 AND 500/500 against freshly-generated instances (15 distinct output shapes seen) — GENERALIZES.

## Irreducible-floor analysis
Dominant memory = 68 fp16 [1,1,10,10] planes (13600 B) from the FOUR contiguous-run all-reduce
sweeps (L+R for horizontal red-run total, U+D for the vertical roll-up). Box-red total per cell
requires a segmented all-reduce on each axis = (forward scan)+(reverse scan)−self = 2 one-directional
run-sums per axis → 4 sweeps, each ~14 tensors even with log-step doubling (offsets 1,2,4; needed
because boxes reach 6 wide/tall). The ≥1-cell gap means a non-gated shift-by-2 would bleed across
adjacent boxes, so the per-step link/gate chain is mandatory.
Secondary: colf30 [1,1,30,30] fp32 = 3600 B — the standard "read colour from the 30x30 one-hot"
floor (slice-then-cast is cheaper than slicing input channels); + final padded uint8 label 900 B.

## OPEN ANGLES (re-attack backlog)
- **Halve the run-sums (the real lever, ~+0.3 to clear B threshold):** an integral-image of red via
  2 CumSum planes + box extents from run-LENGTHS (2 sweeps) would still be 4 sweeps; the genuine cut
  is a 2-CumSum area-sum evaluated per cell at its (corner, far-corner) — blocked by needing a
  data-dependent 2-D Gather (GatherND) of II at per-cell indices. If a cheap per-cell II-rectangle
  read exists, box-red drops from 4 sweeps (13.6 KB) to ~2 CumSum planes (→ ~mem 7 KB ≈ 16.3 pts).
- Reversal-batch trick (pack base+reverse on channel axis, share link) verified correct but
  BREAKS EVEN on bytes (2-ch plane = 2x size cancels the half-count).
- Avoid the 3600 colour read: only possible if the crop colour came from scalars (it doesn't —
  the box keeps its full 1/8/2 texture), so the 3600 stays.

## INSIGHT (transferable)
⭐ "global argmax over solid axis-aligned rectangles + variable crop" is FEASIBLE and beats the
gen-net WITHOUT flood-fill: per-component reductions become **contiguous-run all-reduces** (segmented
doubling, offsets 1,2,4 cover runs ≤8). Distinct per-box counts make the winning box the UNIQUE argmax,
so `winmask = (boxred == ReduceMax(boxred))` recovers its exact bbox FOR FREE (no run-length pass) —
1-D occupancy profiles of winmask give (min_row,min_col,H,W) scalars, then the task036 Gather-shift
crop idiom. The wall is that a 2-D segmented SUM still needs 4 one-directional sweeps (~13.6 KB of
fp16 10x10 planes); it lands at ~15.1, a MARGINAL +0.27 over the gen-net. To cross +0.3/reach B≈16.8
you need a CumSum integral image with a cheap per-cell rectangle read (blocked here by data-dependent
2-D Gather).
