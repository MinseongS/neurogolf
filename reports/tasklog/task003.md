# task003 — 017c7c7b

**Rule:** A height=6, width=3 grid holds a vertically-PERIODIC blue (1) stencil with
period `steps`∈{2,3}; for steps==2 the column pattern may flip L-R once per period
(`flip`). Output is height=9, width=3: same stencil recoloured RED (2) and extended to
9 rows by the same offset+flip schedule. Off the 9×3 grid all channels are 0.
Output rows 0–5 == input rows 0–5. The continuation rows reuse rows the input already
shows, so NO flip handling is needed: steps3 (offsets 0,3,6) → out6,7,8 = in0,in1,in2;
steps2 (offsets 0,2,4,6,8, flip toggling) → out6,7,8 = in2,in3,in0. Only one scalar
needed: is3 = shift-by-3 matches (in3==in0,in4==in1,in5==in2).

**Current:** 17.92 pts, ext:kojimar6275, mem 1172, params 18
**Target tier:** S-ish (spatial copy + 1 scalar) — output is a pure recolour+periodic-copy
of input cells; the only data-dependent choice is steps2 vs steps3.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | uint8 copy block, period2-match detection | A | 745 | 33 | — | — | WRONG (flip breaks period-2 match) |
| 2 | shift-by-3 detection + flip-row select | A | 731 | 33 | — | — | WRONG (out6/flip parity wrong) |
| 3 | corrected extension table (out6,7,8 = in0/1/2 or in2/3/0) | A | 720 | 33 | 18.38 | 200/200 | PASS |
| 4 | fully-uint8 copy (bg/zeros via Equal not Sub) | A | 633 | 34 | 18.50 | 200/200 | ADOPT |

## Best achieved
18.50 @ mem 633 params 34 — beats prior 17.92 by **+0.58**. Fresh 200/200.

## Irreducible-floor analysis
Dominant intermediate = the pre-Pad [1,10,9,3] uint8 block (270B) — floor for "pad a
10-channel 9×3 one-hot block to the 30×30 output"; the 7 always-zero channels still
count but the channel dim must be 10 to match the output. Next is the fp32 ch1/6×3
slice (72B) — Slice inherits the fp32 input dtype, 18 elems × 4B, irreducible entry.
Everything downstream is uint8 (one-hot {0,1}, scored out>0). Detection runs on tiny
fp16 6×3 row slices (~6B each).

## OPEN ANGLES (re-attack backlog)
- Pad the small block is already the cheapest output route; building a colour-index
  plane + Equal-to-arange is WORSE here (the padded full-canvas [1,1,30,30] plane = 900B
  > 270B). No obvious sub-633 angle remains; the 270B 10-ch block is the wall.

## INSIGHT (transferable)
⭐ For a periodic-EXTENSION task, do NOT model the flip/offset parity abstractly — the
generator's continuation rows are usually IDENTICAL to rows the input already contains
(same offset-parity & flip-state), so the extension collapses to a fixed lookup table of
input rows selected by ONE period scalar (here shift-by-3 match). Detect the period by an
exact shift-match (Sub→sq→ReduceSum==0), NOT by a period-2 tiling match — the latter is
broken by per-period column flips even though the period IS 2.
⭐ UINT8 WHOLE-PIPELINE for a one-hot copy/recolour: build bg = Equal(red,0) and a zeros
plane = Equal(red, <unused value>) since uint8 Sub/Mul are rejected but uint8 Equal/Where/
Concat/Pad/Slice all run under ORT_DISABLE_ALL — keeps the whole copy block at itemsize 1.
