# Floor-break build agent — briefing (read this fully, then the guide)

You build ONE ONNX net for ONE NeuroGolf task, from scratch, at minimal memory.
Goal: a generalizing net at the ~3600B label-map floor (or lower) that scores
~16.5+ pts, replacing a high-memory public/base net.

## Hard rules (do not violate)
1. **Do NOT spawn sub-agents.** Do the work yourself, in this one agent.
2. **Write `src/custom/taskNNN.py` EARLY** (a first rough build), then iterate in
   place. Do not hold everything in your head and write once at the end — that is
   how agents stall and get killed. Save → verify → improve → save, repeatedly.
3. Write ONLY `src/custom/taskNNN.py` (3-digit zero-pad, e.g. task175.py with a
   `def build(task):` returning an onnx ModelProto). Do NOT run adopt, do NOT
   commit, do NOT touch networks/ or reports/manifest.json. The main session adopts.
4. If the rule is genuinely NOT a deterministic same-shape per-cell function
   (sprite-scatter / flood-fill / connectivity / shape-correspondence / object
   counting / ambiguous / global-search), BAIL: write nothing, return a one-line
   reason. Do not force an overfit.

## Workflow
1. `PYTHONPATH=. .venv/bin/python -m src.show N --gen` → the generator source IS
   the exact rule. Read it carefully. Also look at a couple of train examples:
   `PYTHONPATH=. .venv/bin/python -m src.show N`.
2. Read `reports/FLOOR_BREAK_GUIDE.md` (the patterns + scoring) and ONE worked
   example: `src/custom/task092.py` (separable interval-fill label map) or
   `src/custom/task205.py` (detection + separable label map). Match their style:
   build with onnx.helper, BOOL `output`, opset 11, `IR_VERSION` from src.harness.
3. Decide the cheapest pattern that fits (prefer earlier):
   - single Conv input→output (mem 0) if per-cell neighbourhood-linear;
   - separable one-hot (row-cond AND/MUL col-cond) → mem ~2400;
   - label map L[1,1,30,30] uint8 + `output = Equal(L, arange[1,10,1,1])` (BOOL).
4. Verify: `PYTHONPATH=. .venv/bin/python reports/verify_fb.py N`
   MUST print `STORED ok=True` with memory ≪ the current net AND `FRESH 200/200`.
   If FRESH < 200/200, your rule is wrong or incomplete — fix it (do not adopt
   an overfit; the main session's gate will reject it anyway).
5. Profile to cut bytes: `PYTHONPATH=. .venv/bin/python reports/mem_profile.py N`
   shows each tensor's byte cost. Eliminate the biggest ones.

## One-hot convention GOTCHAS (cost real bugs before)
- Input is one-hot [1,10,30,30] fp32. In-grid background = channel 0 = 1.
  **OFF-GRID = ALL channels 0** (not ch0=1). So "erase to background" (ch0=1) is
  NOT the same as off-grid. Use a label sentinel ≥10 for off-grid cells so it
  matches no channel in the final Equal.
- `ingrid = ReduceMax(input over channel axis) > 0.5` (any channel set).
- To COUNT coloured cells / get the colour index, use a Conv with weights
  `arange(10)` (ch0 weight = 0) — `ReduceSum` over all channels wrongly includes
  the ch0 background plane.
- The scorer checks `(output[0] > 0)` per channel vs the target one-hot. A BOOL
  label-Equal output passes. Declare graph output value_info dtype = BOOL.

## THE 3600 RULE — the single biggest memory lever (read this)
Reading the colour of every cell from the one-hot input costs ONE [1,1,30,30] fp32
plane = 3600 bytes (a 1x1 Conv with arange weights, or ReduceMax). This is the
UNAVOIDABLE floor for any rule that passes input colours through per-cell. Input
slicing does NOT help (a [1,10,10,10] 10-channel slice = 4000 > 3600). So a
passthrough/label-map task floors at ~3600 (colour read) + ~900 (uint8 label) ≈ 16.6 pts.
**ESCAPE IT when you can:** if the output colour at each cell comes from a SMALL set
of SCALARS (e.g. one detected field/background colour + a fixed/derived geometric
pattern), you do NOT need to read input colour per cell. Build the geometry as a
[1,1,30,30] bool/uint8 mask (900) and emit
  `output = Where(geom_mask, colourA_onehot[1,10,1,1], colourB_onehot[1,10,1,1])`
or `Equal(label_from_scalars, arange)` — NO 3600 colour Conv. Pure geometric tasks
(diagonals, borders, stripes, rectangles, centre-cross) hit ~900-1800 bytes ≈ 17.5-18 pts.
Detect the 1-2 needed colours with cheap channel-count reductions (ReduceSum over
spatial axes -> [1,10,1,1], ArgMax), never a full-grid colour read. ALWAYS ask first:
"does my output colour actually depend on per-cell input, or just on a few scalars?"

## Byte budget reminders (score = max(1, 25 - ln(mem+params)))
- [1,10,30,30]=9000 uint8 / 36000 fp32 — NEVER materialize this; route the
  10-way expansion into the final `output` op.
- [1,1,30,30]=900 uint8 / 1800 fp16 / 3600 fp32. Keep at most ONE ~900B plane.
- [1,10,30,1] or [1,10,1,30]=300 uint8 / 1200 fp32. 1-D aggregates are cheap.
- Use uint8 for labels/masks (ORT implements Where/Equal for uint8; int8/int16
  are NOT_IMPLEMENTED). Float math must be integer-valued < 2^24 (float32-exact).
- Constraints: opset domain "" only; banned ops Loop/Scan/NonZero/Unique/
  Compress/Function. Avoid Slice on float at opset≤10. Gather/Mod/Equal OK.
