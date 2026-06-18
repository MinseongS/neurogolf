# task192 — 7e0986d6

**Rule:** Background = colour 0; grid size 10..20 each axis. 3..5 SOLID rectangular boxes of one `boxcolor` (each >=3 wide & >=3 tall, separated by >=1 gap) plus sparse single "static" pixels of a second `color`. Static pixels (from `remove_neighbors(random_pixels)`) are never 4-adjacent to each other but MAY land on a box cell (overwriting it) or abut a box. OUTPUT = the boxes only, rendered uniformly in `boxcolor`: every static pixel deleted, every box hole a static pixel punched re-filled with `boxcolor`.
**Current:** 14.14 pts, gen:thbdh6332, mem 51898, params 70
**Target tier:** B — per-cell value is a single instance scalar (boxcolor) routed via Equal-expand into the FREE output; not S (mask needs 2x2-conv structure) but well below detection.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | task193 Where(keep,input,bg) | — | — | — | — | — | WRONG: copies noise colour into box holes (497/500 instances have noise-on-box) |
| 2 | keep=occ-2x2; nested fp32 Where (boxhot/bg/zero) | B | 52434 | 67 | 14.13 | — | exact but bgval [1,10,30,30] fp32 floor |
| 3 | same, fp16 planes | B | 34414 | 67 | 14.55 | — | fp16 halves the full plane |
| 4 | crop all full planes to 20x20 (size<=20) | B | 31314 | 67 | 14.65 |  |  |
| 5 | single banded Conv (bg=1, colf=10+k) → ingrid & occ from ONE 30x30 plane | B | 26114 | 67 | 14.83 |  |  |
| 6 | collapse 3-way select to 1-ch index L + Equal-expand into FREE output | B | 15980 | 68 | **15.32** | 500/500 | ADOPTED |

## Best achieved
15.32 @ mem 15980 params 68 — beats prior 14.14 by **+1.18**. Fresh 200/200 and 500/500 exact.

## Irreducible-floor analysis
Dominant intermediate = the 30x30 fp32 Conv output `g30` (3600B): a 10->1 Conv on the fp32 input MUST emit fp32. Everything downstream runs on the 20x20 active crop (size<=20) as fp16 (~800B each) plus the L30 Pad (1800B fp16). The 10-channel expansion costs ZERO storage — `Equal(L30[1,1,30,30], channel_ramp[1,10,1,1])` lands the [1,10,30,30] one-hot directly in the FREE bool output.

## OPEN ANGLES (re-attack backlog)
- g30 fp32 3600B: only removable by not running a fp32 Conv on the input (e.g. ReduceMax/Slice tricks), but those either keep fp32 or balloon a 10-ch crop. Likely at floor for ~15.3.
- L20 vs L30: padding L (1-ch) is cheaper than padding the 10-ch one-hot; already done.

## INSIGHT (transferable)
⭐ "delete isolated noise + re-fill the holes it punched with a UNIFORM recovered colour" is NOT task193's `Where(keep,input,bg)` — that copies the noise colour into box holes. Split into (a) keep MASK = part-of-filled-2x2 of OCCUPANCY (noise-on-box still reads occupied, so holes never appear in the mask), and (b) a single scalar fill colour `bc` = argmax-count channel. Then collapse the 3-way per-cell choice {boxcolor / background / off-grid} into ONE 1-channel index plane `L = keep*bc - (1-ingrid)` (off-grid = -1 ⇒ matches no channel) and EXPAND with `Equal(L, channel_ramp)` straight into the FREE output — turning what looks like a 2-full-plane Where-select into a single 1800B plane. Also ⭐ `bg=inp[0][0]` is WRONG when a box sits in the corner — `common.grids` background is always colour 0. And ⭐ a single banded Conv (weight ch0=1, ch_k=10+k) yields BOTH the grid-extent mask (g>0.5) and the occupancy mask (g>9.5) from one 30x30 plane.
