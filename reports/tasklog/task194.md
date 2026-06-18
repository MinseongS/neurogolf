# task194 — 7fe24cdd

**Rule:** Fixed size=3 grid with 5..9 coloured pixels -> 2*size=6 output. Each
coloured pixel (r,c,v) is stamped at its 4-cell C4 rotation orbit about the 6x6
centre: out[r][c], out[2s-c-1][r], out[c][2s-r-1], out[2s-r-1][2s-c-1] = v.
Equivalently output = A + rot90(A) + rot180(A) + rot270(A) with A = input placed
in the top-left 6x6. The 4 orbit cells are always distinct, so it is a pure
FIXED coordinate scatter: each of the 36 output cells reads exactly ONE input
cell. No value plane, no flood/argmax, all instances fixed 3x3->6x6.
**Current:** 17.67 pts, GridSample(nearest)+Pad import (ext:wguesdon6304), mem 1440, params 81
**Target tier:** S (pure spatial copy) in principle — but the cheapest realisation is the GridSample at floor.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | GridSample(nearest,zeros) [1,6,6,2] grid + Pad 6x6->30x30 | S-ish | 1440 | 81 | 17.673 | 200/200 | TIES import (same approach, at floor) |

## Best achieved
17.673 @ mem 1440 params 81 — adopted? N (ties, not >=+0.3). Beats prior 17.673? N (exact tie).

## Irreducible-floor analysis
The dominant intermediate is the GridSample output [1,10,6,6] fp32 = 1440 B. It
cannot be shrunk:
- GridSample output dtype = input dtype = fp32 (ORT). To get a fp16 720 B sample
  you must first have fp16 data, but the harness input is fixed fp32. Casting the
  full input is [1,10,30,30] fp16 = 18000 B; slicing the active 6x6 region first
  is itself a [1,10,6,6] fp32 = 1440 B intermediate (>= the GridSample cost), so
  no cast/slice route reaches fp16 below 1440.
- All 10 channels are required (output one-hot uses arbitrary colours 0-9 + bg).
  Dropping ch0 to 9 channels needs an input slice ([1,9,30,30]=32400) — far worse.
- Making GridSample the FINAL op (writing directly to the FREE `output`) needs a
  full [1,30,30,2] grid = 1800 params (no intermediate) = 17.50 pts, WORSE than
  72-param grid + 1440 intermediate (1512 total, 17.679).
- The C4 orbit-sum (rot90/180/270 via 0-param Transpose+negative-Slice) is
  strictly worse: each rotation of the 6x6 region is its own 1440/720 plane and
  you still pay a 1440 slice to extract the region, summing 4 copies.
1440 fp32 + 72 grid = 1512 (17.679) is the genuine floor; the +0.006 over the
stored 17.673 (which carries 8 extra pad/padval params) is far below the +0.3 bar.

## OPEN ANGLES (re-attack backlog)
- If a future ORT op could sample 10-ch and emit fp16 directly (e.g. a Resize/
  Gather variant whose output dtype is controllable) the floor would drop to
  ~720+grid -> ~18.3. None exists under the opset/banned-op constraints today.
- GatherND/Gather over a flattened spatial axis still produces the same 1440
  fp32 6x6x10 result (or a 36000 reshape intermediate) — no improvement.

## INSIGHT (transferable)
A FIXED full-coverage coordinate scatter (every output cell reads exactly one
input cell, geometry data-independent) is optimally a single GridSample(nearest)
with a small per-output-cell grid + Pad. Its memory floor is the sampled plane
in fp32 (channels x out_cells x 4 B) and is UNSHRINKABLE: GridSample inherits the
fp32 input dtype, and any cast/slice to reach fp16 first costs at least as much as
the sample itself. A small-grid+intermediate beats a full-canvas-grid+free-output
whenever (out_cells x 2) > (channels x out_cells x 4) is false — i.e. almost
always, since params count element COUNT. ⇒ public GridSample scatter imports are
already at floor; treat them as MARGINAL unless the active region collapses the
channel count or a controllable-dtype sampling op appears.
