# task354 — ddf7fa4f

**Rule:** size-10 grid. Row 0 holds 3 coloured "light" pixels (gray excluded as a
colour), one per light column. Below sit 3 solid gray rectangular boxes; the
generator guarantees each box's COLUMN span contains exactly one light column.
Output: every light pixel passes through unchanged at row 0, and every gray box is
recoloured to the colour of the light whose column lies in that box's span; all
other in-grid cells stay background (0). Verified key reduction (5000/0): within
ANY single row each contiguous gray run is exactly one box's column extent and
contains exactly one light column, and the light column is gray through the whole
box. So planting L[c] on the gray cells of each box's light column then spreading
it HORIZONTALLY within gray runs reproduces the output exactly.

**Current (stored before me):** 15.26 pts, public CumSum/MaxPool net, mem 16860, params 147.
**Target tier:** A-ish (separable seed + horizontal run-fill) — output colour per
cell is a row-local run function, not a single per-cell linear map (two boxes may
share a column with different colours → no pure column propagation, no Tier S/one-op).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | full-30x30 Conv colour-index + seed + 4× gated 1×3 MaxPool fill (uint8) | B | — | — | — | — | ORT rejects uint8 MaxPool |
| 2 | same, fp16 fill planes | B | 7320 | 41 | 16.10 | — | ok; 3600B Conv dominates |
| 3 | drop full colf: gray=ch5 slice, lightrow=row0-slice Conv, lightplane | A | 3980 | 57 | 16.70 | — | ok; fill chain dominates |
| 4 | fold final regate into label Where (drop one plane) | A | 3780 | 57 | **16.75** | 200/200 + 600/600 | **best, exact** |
| 5 | crop fill to 8-row gray band (rows 2–9) | A | 3940 | 72 | 16.70 | — | WORSE: extra slice+Pad planes cost > band saving; reverted |

## Best achieved
**16.75** @ mem 3780 params 57 — adopted? **N** (orchestrator gates). Beats prior
15.26 by **+1.49**. GENERALIZES: official genverify fresh_pass 200/200 + isolated
600/600. (MARGINAL vs the P=17.0 par by −0.25; clear win vs the real stored 15.26.)

## Irreducible-floor analysis
Dominant intermediate = the horizontal run-fill chain: 4 gated dilation steps
(cur0 + pool0..3 + cur1..2) = 8 planes × 200B fp16 = ~1600B. Each step MUST be
MaxPool(1×3) THEN re-gate by gray: the minimum inter-box column gap is 1 (verified),
so radius-1 dilation with a re-gate after every pool is required to block leakage
across a single non-gray cell; radius-2 (1×5) leaks (2858/4000 fail). Max box width
is 5 with the light column possibly at a box edge ⇒ 4 cells of one-directional
spread ⇒ K=4 (K=3 fails 143/4000). MaxPool needs a float dtype (ORT rejects uint8
MaxPool / int8 Max), and fp16 (2B) is the smallest float ⇒ planes can't go to 1B.
Other costs: padded label L [1,1,30,30] uint8 = 900B (the final Equal must run on the
full 30×30; Pad rejects bool so L can't stay 10×10), the two fp32 input slices
gray_f + row0 = 800B (input is fp32, slices inherit it).

## OPEN ANGLES (re-attack backlog)
- **2-op run-fill via CumSum reset** (would drop fill 1600B→~400B → ~mem 2580 →
  ~17.1 pts): each run has exactly ONE non-zero seed, so a forward/backward
  segment-broadcast would fill it; the blocker is resetting the cumulative value at
  each run boundary (gap) without Scan/segment-id. If a cheap run-start index can be
  built from gray (e.g. gray AND NOT shifted-gray), a Gather-subtract of the prefix
  at the run start could replace the dilation. Untried — most promising lever.
- per-box column-extent as scalars (rmin/cmin/W via 1-D occupancy) then a separable
  Srow@P@Scol stamp — blocked here because boxes can share columns at different rows,
  so a single [1,1,10] col-occupancy can't separate two stacked boxes.
- shrink the 800B fp32 input slices by deriving gray/lightrow from one shared slice.

## INSIGHT (transferable)
⭐ Horizontal/vertical "spread one seed across a contiguous run" = iterated
`MaxPool(1×k) → re-gate-by-mask`; the re-gate after EVERY pool is what blocks
leakage across unit gaps, so kernel radius must be 1 when the minimum gap is 1.
Iteration count = max one-directional spread distance (worst seed-to-edge), NOT box
size. fp16 is the floor dtype for any MaxPool-based fill (ORT rejects uint8 MaxPool
and int8 element-wise Max). When non-target cells of a label only ever take ONE of
two trivial values (here: row-0 lights or background-0), you don't need a full
colour-index plane — build a tiny row/col vector and mask it into position, killing
the 3600B full-canvas Conv collapse entirely.
