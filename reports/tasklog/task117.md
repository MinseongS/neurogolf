# task117 — 4c5c2cf0

**Rule:** A background-0 grid (size 12..15, so ≤15×15) holds two colours. The BODY is a
fixed 3×3 X-cross (corners + centre) in colour `color` at (rowoff,coloff); its centre is
(cr,cc)=(rowoff+1,coloff+1). The LEGS are an arbitrary Conway sprite in colour `legcolor`,
drawn in ONE quadrant relative to the cross centre (the input shows only that quadrant). The
OUTPUT keeps the body cross and reflects the legs 4-fold about the centre: a leg cell (r,c) →
{(r,c),(2cr−r,c),(r,2cc−c),(2cr−r,2cc−c)} OR'd. An overall horizontal/vertical flip is applied
to BOTH input and output, so it does not affect the transform. Body identification: the body
is the colour whose 5 pixels form a mono 3×3 X (corners+centre set, edge-mids empty) AND whose
total pixel count is exactly 5; the legs are the other colour.

**Current:** 15.2196 pts, `ext:biohack_new` (public base net; verified GENERALIZES 100/100 fresh),
mem 17527, params 156.
**Target tier:** B (label-map + final Equal, with a data-dependent 2-D reflection realised as
two boolean MatMuls — the task112 idiom). Tier A blocked: the relabel is a per-cell colour
function of a globally-detected centre, not a row⊗col separable rectangle; the body/leg colours
are random per instance so no fixed Conv can route them (Tier S blocked).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | count==5 ⇒ body, double-MatMul reflect | B | 36672 | 131 | — | — | WRONG: legs have count 5 ~36% of the time |
| 2 | geometric mono-X detect + own-colour-count==5, in-grid=input-occ | B | 45326 | 131 | fail | — | in-grid used coloured-occ, clipped reflected legs |
| 3 | in-grid = full grid square (incl ch0) | B | 45326 | 131 | 14.28 | — | correct 265/265 |
| 4 | colf via 1×1 Conv on free input (kill [1,10,W,W]); fp16 planes; W=15 | B | 19444 | 142 | 15.12 | 200/200 | correct |
| 5 | banded X-conv, mono via single corner-conv, 1-D centre+lcolor scalars | B | 16918 | 157 | 15.25 | 200/200 | correct, MARGINAL |

## Best achieved
15.2546 @ mem 16918 params 157 — adopted? **N** (orchestrator gates). Beats prior 15.2196?
**Y but only +0.035 → MARGINAL** (< +0.3 threshold). Fresh 200/200 (and 1000/1000 on the
weakened-mono check). Both the prior base net and this custom generalize, so this is a true
lateral, not a gap-closing swap.

## Irreducible-floor analysis
Dominant intermediate: `colf30` = the full-grid colour-index plane [1,1,30,30] fp32 = 3600 B,
produced by a 1×1 Conv contracting the 10 channels of the FREE input. This is the single
smallest representation of a per-cell colour value over the whole grid; any alternative that
reads all 10 channels at ≥15×15 (a spatial Slice of the input = [1,10,15,15] fp32 = 9000 B, or
a fp16 cast of the input = 18000 B) is strictly larger, and Conv cannot emit fp16 from an fp32
input nor restrict its output to a sub-window. colf is genuinely required for the mono-colour
discriminator (Conv(colf, corner-kernel)==0), which cuts the spurious-leg-X mis-detection rate
from ~0.45% (occupancy+count only) to ~0.005%. The second cost centre is the `own5` subsystem
(int32 Gather index plane 900 B + `ownc` 450 B): a per-cell "count-of-my-colour==5" test that
removes 27/30000 spurious leg-sub-X detections; it cannot use uint8 indices (ORT Gather rejects
them) and a MatMul/one-hot alternative would re-materialise a [1,10,*,*] plane. Together
colf30+colf32+colf+own5 ≈ 6.7 kB are structural; the remaining ~10 kB is ~16 fp16 [1,1,15,15]
working planes (2-D reflection matrices + 3 MatMul products + masks) that the data-dependent
2-D reflection inherently needs. Net floor ≈ 16.9 kB ⇒ ~15.25 pts.

## OPEN ANGLES (re-attack backlog)
- **Kill colf30 (3600 B → ~15.5+ pts):** find a channel-contraction that emits a [1,1,15,15]
  result directly from the free input without a 30×30 or [1,10,*,*] stop-over. (MatMul on a
  reshaped input creates a [10,900] intermediate = 36 kB; no cheap reshape found.)
- **Cheaper own5 (−~1.3 kB):** replace the per-cell colour-count Gather with the quadrant rule
  (legs all on one side of cr and cc) computed from 1-D leg-row/col profiles — but it is
  circular (needs bcolor→legf→center). A non-circular 1-D formulation would recover the bytes.
- **Fold the 3 reflection MatMul products:** lD = Rmat@legf@CmatT could perhaps be expressed so
  lB/lC are not both materialised (currently 4×450 B for the 4-fold OR).

## INSIGHT (transferable)
- ⭐ **Mono-colour X-centre detection without per-channel planes:** to find a fixed small
  same-colour stamp (here a 3×3 X) among other-colour clutter, combine on the colf plane:
  (a) a BANDED occupancy conv `10·(#X-cells)+1·(#edge-cells)` compared to a single constant
  (==50) detects "X full AND edges empty" in ONE conv; (b) a corner-vs-centre conv
  `corners=+1, centre=−4` compared to 0 enforces mono-colour (4·corner-sum==4·centre) without
  a separate ×5 plane; (c) a Gather of per-channel counts by the colf index gives a per-cell
  "count-of-my-colour" for an exact "this colour totals N" test. This trio uniquely locates the
  stamp centre at ~0.005% error with zero [1,10,H,W] materialisation.
- **`cnt==5` is NOT a body discriminator here** even though the body is always 5 px: the leg
  sprite has exactly 5 px ~36% of the time. Identify by SHAPE (mono isolated X) + count, not
  count alone.
- **In-grid mask must include channel 0:** the output extends beyond the input's coloured cells
  (reflected legs land on previously-background cells), so the grid extent has to come from the
  FULL grid square (any channel incl. ch0), never from coloured-occupancy — otherwise reflected
  pixels get clipped to the sentinel.
- **lcolor without a 2-D plane:** with exactly two present colours, `lcolor = Σ_k k·(cnt_k>0) −
  bcolor` from the tiny [1,10] counts, avoiding a 450 B colf·legmask reduction plane.
