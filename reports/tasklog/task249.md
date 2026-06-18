# task249 — a416b8f3

**Rule:** Input is a width×height grid (one-hot, top-left of 30×30). Output is a (2*width)×height
grid that is the input duplicated horizontally: out[:, c] = out[:, width+c] = in[:, c]. Equivalently
output column i maps to input column m[i] = i (if i<W) else i-W. This holds for ALL i in 0..29: i<W →
col i; W≤i<2W → col i-W; i≥2W → col i-W≥W = off-grid input col = all-zero one-hot = correct empty. So
NO clip is needed (W≥3 ⇒ i-W ∈ [0,29]). Rows pass through (Gather on the width axis only); off-grid
rows are all-zero. Pure spatial copy ⇒ Tier S, output is FREE.
**Current (prior):** 18.33 pts, ReduceMax+ReduceSum-scan width + Less/Sub/Where/Clip column-index +
int64 Gather, mem 758, params 34.
**Target tier:** S — output is a pure Gather of input columns; only the scalar W and a length-30 index
vector need materializing.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | drop redundant Clip; build index map in fp16 Where; Cast int64→int32 | S | 400 | 34 | 18.93 | — | works |
| 2 | drop redundant Reshape (ReduceSum axes=[1,2,3] keepdims=0 already gives [1]) | S | 396 | 33 | 18.94 | 200/200 | ADOPTED |

## Best achieved
18.94 @ mem 396 params 33 — beats prior 18.33 by +0.61 (≥+0.3 ✓). Adopt: out of scope (build-only).

## Irreducible-floor analysis
Dominant intermediates: colocc [1,1,1,30] fp32 (120B) + m int32 [30] (120B). colocc is irreducible —
ReduceMax inherits the fp32 input dtype (casting input to fp16 = 18000B). The int32 index plane (120B)
is the floor for axis-3 Gather indices (Gather rejects uint8; int32 < int64). Remaining: shifted f16 60,
m16 f16 60, lt bool 30, W32 f32 4, W16 f16 2. The index pipeline (Less→Sub→Where→Cast) needs at minimum
one bool + one fp16 + the final int32 over length-30. ~396B is near the practical Tier-S floor for a
runtime-width column gather.

## OPEN ANGLES (re-attack backlog)
- Eliminate `shifted` (60B): no single op produces Where(i<W, i, i-W) without materializing the i-W
  branch; arithmetic (mul+sub of a cast ge) costs MORE planes. Likely irreducible.
- W without the [1,1,1,30] colocc plane: W = (max occupied col)+1, but every alternative still needs a
  per-column reduction of the same shape. No cheaper route found.

## INSIGHT (transferable)
⭐ For a runtime-width HORIZONTAL DUPLICATION (out col i = in col m[i], m[i]=i if i<W else i-W), the
clip is REDUNDANT: i≥2W maps to input col i-W≥W which is off-grid = all-zero = the correct empty output,
and i<2W keeps i-W in-range since W≥3. Build the length-30 index in fp16 (Where) and Cast to int32 (not
int64) — halves both the index and the working vectors. Width = ReduceSum(ReduceMax(input,[1,2]),[1,2,3])
with keepdims=0 lands a clean [1] scalar (axis 0 survives), no Reshape needed. Net: 758→396, +0.61.
