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
| 6 | collapse 3-way select to 1-ch index L + Equal-expand into FREE output | B | 15980 | 68 | 15.32 | 500/500 | prior ADOPTED |
| 7 | crop INPUT (10-ch) to 20x20 before Conv | — | 24279 | 68 | 14.90 | — | WORSE: 10-ch input slice = 16000B counted plane (input slice is NOT free) |
| 8 | keep full-input Conv; tail in UINT8 (L30 uint8 900B, Equal accepts uint8 here) | B | 11879 | 68 | 15.61 | — | L30 1800→900B; +0.30 |
| 9 | boxcolor via ArgMax (drop one-hot/ramp/ReduceSum chain) | B | 11789 | 58 | **15.62** | 200/200 | ADOPTED |

## Best achieved
15.62 @ mem 11789 params 58 — beats prior 15.32 by **+0.30** (and prior-prior 14.14 by +1.48). Fresh 200/200, stored 265/265 exact.

## Irreducible-floor analysis
Dominant intermediate = the 30x30 fp32 Conv output `g30` (3600B): a 10->1 Conv on the fp32 input MUST emit fp32, and any input narrowing balloons (a 10-ch 20x20 input slice = 16000B — input slices are NOT free, only the graph input is). Second is the 20x20 fp32 crop `g` (1600B) which serves BOTH thresholds (occ>9.5, ingrid>0.5) — measured cheaper than two 30x30 bool thresholds (900+900) sliced. The four 2x2-conv pipeline planes (occ/blockcnt/blockfull/keepcnt, fp16 800B each) are the genuine morphological-opening computation (erode-2x2 then dilate-2x2 = 2 convs, not fusible). Entry pair 3600+1600 = 5200B is the structural floor at ~15.6.

## OPEN ANGLES (re-attack backlog)
- g30 fp32 3600B: removable only by not running a fp32 Conv on the input; ReduceMax/Slice tricks keep fp32 or balloon a 10-ch crop. At floor for this rule.
- The 4 fp16 conv planes (3200B): "part-of-filled-2x2" is morphological opening (2 passes); no single-conv equivalent found.

## INSIGHT (transferable)
⭐ "delete isolated noise + re-fill the holes it punched with a UNIFORM recovered colour" is NOT task193's `Where(keep,input,bg)` — that copies the noise colour into box holes. Split into (a) keep MASK = part-of-filled-2x2 of OCCUPANCY (noise-on-box still reads occupied, so holes never appear in the mask), and (b) a single scalar fill colour `bc` = argmax-count channel. Then collapse the 3-way per-cell choice {boxcolor / background / off-grid} into ONE 1-channel index plane and EXPAND with `Equal(L, channel_ramp)` straight into the FREE output. Also ⭐ `bg=inp[0][0]` is WRONG when a box sits in the corner — `common.grids` background is always colour 0. And ⭐ a single banded Conv (weight ch0=1, ch_k=10+k) yields BOTH the grid-extent mask (g>0.5) and the occupancy mask (g>9.5) from one 30x30 plane.
⭐ RE-GOLF WINS (this session, +0.30): (1) the WHOLE TAIL can be UINT8 on THIS ORT build — `Equal(L_uint8, ramp_uint8)` AND opset-11 `Pad` with a uint8 constant BOTH run under ORT_DISABLE_ALL, so the colour-index plane L30 scores 900B (uint8) not 1800B (fp16); always test Equal/Pad uint8 directly rather than trusting the "Equal rejects uint8 / Pad rejects uint8" tasklog claims (FALSE here). (2) ⚠️ CROP-INPUT TRAP: slicing the 10-ch input to the active region is NET-NEGATIVE — an input Slice is a COUNTED node output (a 10-ch 20x20 slice = 16000B), only the graph `input` itself is free; crop the 1-channel PACKED plane after the Conv instead. (3) ArgMax over a channel-axis count gives a scalar colour index in ONE op (drops the one-hot/ramp/ReduceSum chain, ~90B mem + 10 params).
