# task396 (ARC-AGI fcb5c309) — crop largest hollow box, recolour to static colour

P (ext:kojimar7113) = 14.72 → **15.29** (mem 16323, params 149), fresh 300/300.
(Supersedes a leftover src/custom/task396.py that scored 14.49 with a k=2..7 run-conv army +
full vertical+horizontal run-length on fp32 18×18 planes.)

## Rule
2-3 hollow rectangular boxes (1px outline colour c0, black interior) + scattered single-pixel
static (colour c1, some inside boxes). `wides`/`talls` sorted DESC ⇒ box 0 is the LARGEST
(max width AND max height). Output = tall0×wide0 crop of box 0 with every NON-BLACK cell
(outline + interior static) painted c1, black interior stays black.

## Encoding (single-direction run-length, plane-eliminated)
- colf30 = Σk·input_k (1×1 Conv, fp32 entry 3600B), Slice→18×18, Cast→fp16 colf.
- HORIZONTAL same-colour adjacency pairs eqh = (colf[:, :-1]==colf[:, 1:]) & colf>0 (uint8).
- **Run-length-ending-here via CUMSUM-RESET** (replaces the old k=2..7 conv army): pad a leading
  zero, cs=CumSum(eq, axis); reset=where(eq, -BIG, cs); rl = cs − prefixmax(reset) where
  prefixmax = one-sided full-length MaxPool (ZERO params). maxH = ReduceMax(rl) = wide0−1.
  CumSum needs fp32 (rejects fp16/uint8/int8; int32 same size) → pay ONE fp32 cumsum + one
  fp32 cast-up, everything else fp16.
- Position from the horizontal map only: bcol0 = (min col with per-col max-run==maxH) − maxH;
  brow0 = min row with per-row max-run==maxH (top edge). Both reduced to [1,1,1,18]/[1,1,18,1]
  BEFORE masking (no full-plane Where).
- **tall0 by a 1-D probe** down box-0's left-edge column (Gather col bcol0 from colf →
  [1,1,18,1]): tall0 = (first row ≥ brow0 where colvec != c0) − brow0. ⚠️ box may reach the
  grid bottom edge ⇒ the no-stop fallback MUST be A(=18), not BIG, else tall0 overshoots by 1.
  This **kills the entire vertical cumsum machinery (~5KB, 9 planes)** — box 0 has BOTH max
  width and max height (same sorted index 0), so one direction + a column probe suffices.
- c0 = colf at (brow0,bcol0); c1 = present non-bg colour ≠ c0 (ArgMax over masked chramp).
- Gather-shift colf to (brow0,bcol0), crop WORK×WORK, paint non-black→c1, **uint8 sentinel-99
  Pad to 30×30** (opset-11 Pad+Equal accept uint8 → 900B not fp16 1800B), Equal(L30, chan u8)
  → FREE BOOL one-hot output.

## Dominant intermediates (irreducible)
Conv entry 3600B (input is fp32 ⇒ Conv can't keep fp16); cumsum cast-up+cumsum 2×1296B
(fp32-only op); colff slice 1296B (transient fp32 18×18 before fp16 cast); output Pad 900B.

## Levers used / transferable
- ⭐ Single-axis run-length suffices when the target object maximises BOTH axes (sorted DESC):
  detect on one axis, recover the other dimension by a tiny 1-D edge-column run probe.
- ⭐ CUMSUM-RESET run-length (1 fp32 cumsum + one-sided MaxPool, 0 params) replaces a k-value
  conv army for max-contiguous-run; conv-sum overcounts across gaps, cumsum-reset does not.
- uint8 sentinel-99 Pad+Equal for the output one-hot (opset-11) → 900B vs fp16 1800B.
- Reduce a full-plane argmin to a per-row/col profile BEFORE the masking Where.
- Pitfall: a "first stop below" probe must fall back to the canvas EDGE, not a huge sentinel,
  when the object touches the grid boundary (+1 overshoot bug, caught at 263/266).
