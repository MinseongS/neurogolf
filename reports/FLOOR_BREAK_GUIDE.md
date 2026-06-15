# Floor-break guide — rebuild high-memory nets to ~3600 bytes (agent briefing)

PROVEN 2026-06-15: task019 mem 43800→3365, pts 14.31→16.84, 200/200 fresh. Worked example:
`src/custom/task019.py`. Read it — it demonstrates every lever below.

## Scoring model (EXACT, validated)
`score = max(1, 25 - ln(memory + params))`. `memory = SUM over every intermediate tensor of
(elements × itemsize)` — cumulative, NOT peak. Tensors named `input`/`output` are FREE (any dtype).
itemsize: bool/uint8=1, fp16=2, fp32/int32=4. So [1,10,30,30]=9000 uint8/36000 fp32;
[1,1,30,30]=900 uint8/3600 fp32; [1,10,30,1] or [1,10,1,30]=300/1200; [1,1,12,12]=144/576.
Conv weights & initializers are PARAMS (added to memory in score, but you control them); a single
Conv `input→output` with no other tensors = memory 0.

## The core idea
The ~9k "floor" on size-dependent tasks comes from materializing a 10-channel [1,10,30,30]
intermediate. NEVER do that. Push the per-channel (10-way) expansion into the FINAL op whose result
is the free `output`. Three patterns, prefer earlier:

1. **Single op input→output (mem 0, ~18-25 pts):** rule is a per-cell neighbourhood-linear function
   → ONE Conv(+bias) into output. The scorer is per-channel `(out>0)`, so each output channel is a
   separating hyperplane over the local one-hot — hand-derive integer weights (see task344). Best.
2. **Separable one-hot (mem ~2400, ~17 pts):** output[ch,r,c] = rowcond[ch,r] AND/MUL colcond[ch,c].
   Build tiny [1,10,30,1] & [1,10,1,30] condition tensors from input (ReduceMax/compare), final
   `And`/`Mul` broadcasts into output. (Generalizes task239/246/082.)
3. **Label map + final Equal (mem ~3600 or less, ~16.8 pts):** GENERAL fallback. Build a SINGLE
   uint8 label map L[1,1,30,30] = output colour index (0-9) per cell; sentinel ≥10 where the cell
   is all-channels-off. Final op: `output = Equal(L, arange[1,10,1,1])`. Requires opset 11 (float/
   general Equal) and the graph `output` value_info declared dtype **BOOL** — else load fails.

## Levers (all measured on task019)
- **uint8 for labels/masks** (ORT implements Where/Equal for uint8; int8/int16 = NOT_IMPLEMENTED).
  uint8 plane = 900B vs 3600 fp32.
- **small working canvas + Pad at end.** If the active region is bounded (e.g. ≤12×12), do ALL
  per-cell work at [1,1,12,12] (144 elem) and `Pad` only the final L to [1,1,30,30] (sentinel fill)
  just before the Equal.
- **read a small input slice** (e.g. 6×6 of channel-0) when gathers only touch a sub-grid.
- **separable rectangles**: bbox/in-grid masks = outer product of 1-D bounds (`row<H & col<W`).
- Keep L the ONLY ~900B plane; everything else ≤144B (12×12 masks, 1-D aggregates).

## Constraints
- opset domain "" only; banned ops: Loop/Scan/NonZero/Unique/Compress/Script/Function (Gather/Mod OK).
- Float math integer-valued & < 2^24 (float32-exact). Avoid Slice on float (opset10 INVALID_GRAPH);
  int32 Equal rejected by ORT → use uint8 Equal or `1-Clip(Abs(d),0,1)`.
- Build the model directly with onnx.helper when you need BOOL output / opset 11
  (declare output value_info dtype=TensorProto.BOOL, opset_imports=[make_opsetid("",11)],
  ir_version=src.harness.IR_VERSION). For float output, the `src.builders._model` helper works.

## Workflow per task
1. `.venv/bin/python -m src.show N --gen` → read the generator = exact rule.
2. Decide the output colour per cell → is it a single Conv (1), separable (2), or a label map (3)?
   If the rule is sprite-scatter / shape-correspondence / non-deterministic, BAIL (report why).
3. Build `src/custom/taskNNN.py` `def build(task):`. Count bytes as you go.
4. Verify (must be STORED ok=True, memory ≪ current, FRESH 200/200) with the script in your task prompt.
5. Write ONLY src/custom/taskNNN.py. Do NOT adopt/commit/touch networks/manifest — main adopts.
