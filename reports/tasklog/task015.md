# task015 — 0ca9ddb6 (twinkling stars)

**Rule:** 9x9 grid placed top-left on the 30x30 canvas (cells outside the 9x9 grid
are all-zero in input AND output). Every coloured pixel copies input->output. A
blue(1) "rook" star stamps colour 7 at its 4 orthogonal neighbours; a red(2)
"bishop" star stamps colour 4 at its 4 diagonal neighbours; colours 6 and 8 just
copy. The generator scooches twinklers into [1, size-2] and keeps every twinkler
at Chebyshev distance >=1 from all other kept pixels, so the 7/4 halos always stay
inside the grid and land ONLY on background cells (never on each other or on a
pixel).
**Current:** 18.197605 pts, ext:kojimar6275, mem 0, params 900.
**Target tier:** S (pure-param single Conv, mem 0) — already there; the question is
purely whether 900 params can be shrunk.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 2x chained Where(halo->onehot, input) | B | 52200 | 59 | 14.14 | 265/265 | 10-ch stage1 intermediate (36000B) dominates |
| 2 | 9x9 block: slice 10 ch fp32 + dilation conv + bool overlay + Concat10 + Pad | B | 6318 | 91 | 16.23 | 265/265 | 10 fp32 9x9 slices (3240B) + cat10 uint8 (810B) floor it |
| 3 | runtime-assembled rank-structured conv weight (3 distinct 10x10 planes -> Concat -> Reshape [10,10,3,3]) | S* | 10800 | 408 | 15.68 | 265/265 | assembled fp32 weight (3600B) + Concat (3600B) intermediates > 900-param saving |
| 4 | single dense Conv [10,10,3,3] (= clean reimpl of public closed form) | S | 0 | 900 | 18.20 | 200/200 | ties prior; mem 0 |

## Best achieved
18.197605 @ mem 0 params 900 — adopted? matches prior (no regression). Beats prior
18.197605? **N (exact tie / at floor).**

## Irreducible-floor analysis
The closed-form rule IS a single linear conv of the one-hot: out_k(centre)=in_k for
all 10 channels; out_7 += in_1 at the 4 rook offsets; out_4 += in_2 at the 4 bishop
offsets; out_0 = in_0 - in_1(rook) - in_2(bishop) (the two -1 terms cancel the bg bit
exactly where a halo lands so the halo cell is a clean single-colour one-hot). A
single ungrouped Conv on a 10-channel one-hot I/O FORCES weight shape
[out=10, in=10, kH=3, kW=3] = 900 elements; params count ELEMENTS not nonzeros, and
the 3x3 kernel is forced by the +-1 halo offsets. So 900 params / mem 0 is the hard
floor for this architecture. Every escape pays a full-canvas fp32 intermediate that
exceeds the saving:
  - runtime-assembled weight: Conv forces fp32 weight dtype, so the [10,10,3,3]
    intermediate is 3600B (+ the Concat that builds it) -> 10800B mem (attempt 3).
  - per-channel bool-overlay block: the [1,10,9,9] Concat is 810B uint8 AND the
    passthrough needs ~3240B of fp32 9x9 slices -> 16.2 ceiling (attempt 2).
  - grouped Conv: the cross-channel edges (out7<-in1, out0<-in1/in2, out4<-in2) cross
    every contiguous group split, so no valid group < 10 exists.
Budget check: beating 18.20 by +0.3 needs mem+params <= exp(25-18.5) = 665; no route
gets the copy+halo logic under 665 without a >665B intermediate.

## OPEN ANGLES (re-attack backlog)
- None with payoff. The only sub-900 single-op would need a Conv weight whose first
  two dims are < 10, which is impossible without slicing input channels (an
  intermediate) since I/O are fixed 10-channel one-hots. Confirmed dead.

## INSIGHT (transferable)
⭐ "Pure-param single-Conv" tasks (current net = one Conv, weight [10,10,3,3]=900,
mem 0) are AT FLOOR when the rule is an exact linear map of the one-hot (copy +
local-stamp halos). The dense weight's 900 element count is irreducible: the two
channel dims are pinned to 10 by the 10-channel I/O and the kernel by the stamp
radius. Runtime-assembling the weight from smaller inits LOSES because ORT forces
the assembled Conv weight to fp32 (>=3600B intermediate) — params drop but mem
balloons past the saving. Any block/Concat/Where reformulation introduces a
full-canvas intermediate (>=810B uint8 cat10 or >=3240B fp32 slices) that caps the
score ~16-18.3 < 18.5. So: a memory-0 single-Conv pure-param net with the channel
dims already at the one-hot count is a confirmed BAIL — do not chase rank-k weight
tricks (they help PARAMS for an INITIALIZER weight, never a runtime-built one).
