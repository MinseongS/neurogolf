# boolpad scan — Cast-before-shape-op harvest (S11 task174 lever)

**Lever (task174):** a BOOL/uint8 tensor Cast to a wider dtype only to satisfy an
old-opset shape-only op (Pad illegal on bool at opset ≤11, legal at opset 13). Deleting
the Cast + bumping opset removes the counted wide plane. task174 landed +0.19 this way.

**Scan:** `reports/scripts/boolpad_scan.py` over all 400 `networks/taskNNN.onnx`
(skip 120/220/230/294 = mem≈0). Flags Cast(bool|uint8 → wider) whose output feeds ONLY
shape-only ops (Pad/Concat/Reshape/Transpose/Slice/Tile/Gather*/Scatter*/…) **on their
DATA slot** (index/shape/param slots legitimately require int and are excluded).
Raw hits + verdicts: `reports/boolpad_scan.json`.

## Result: 32 raw hits, **0 adoptable**. Lever is exhausted in this codebase.

Every hit fails at the mechanism level for one of three reasons:

### 1. Scatter onto the FLOAT graph-input canvas — dtype locked (126, 106, 293)
The top-ranked hits are `bool→float` casts on the **updates** input of a
`ScatterND`/`ScatterElements` whose **data slot 0 is the model's `input` FLOAT[1,10,30,30]
canvas**. ONNX scatter requires updates dtype == data dtype, so updates is locked to float.
Narrowing would require casting the whole 9000-byte canvas to bool — a large net loss, not
a saving. Not harvestable.

| task | plane | bytes | naive dpts | verdict |
|---|---|---|---|---|
| 126 | updates FLOAT[2,30] | 240 | 0.51 | FALSE+ (scatter onto float input) |
| 106 | updates FLOAT[1,1,6,6] | 144 | 0.26 | FALSE+ (scatter onto float input) |
| 293 | updates FLOAT[2,3,2] | 48 | 0.04 | FALSE+ (scatter onto float input) |

### 2. Cast pays for a downstream ARITHMETIC op, not the shape op (255, 158, 206, 092)
`bool→float` immediately feeds a shape-op (Reshape/Slice/Concat) whose output goes straight
into MatMul / AveragePool / ReduceSum / Mul — ops that genuinely require float. The float is
mandatory for the arithmetic, not for the reshape. The scan flags them because the *immediate*
consumer is a shape-op; mechanism inspection shows the real consumer is float-only. (All four
are also heavily-matured nets, incl. task092/task255 landed in prior sessions.)

### 3. The one true bool→Pad (task056) — REFUTED, opset-migration inverts the lever
Only `networks/task056.onnx` has a genuine `bool→fp16→Pad` (the task174 shape). **Built and
measured** at `reports/candidates/task056_boolpad.py`:

| | memory | params | points |
|---|---|---|---|
| incumbent (opset 9, fp16 Pad) | 34 | 0 | 21.4736 |
| boolpad (opset 13, bool Pad)  | 22 | 23 | 21.1933 |

Delta **−0.28 (LOSS)**, fail=0 both. The fp16 plane *is* removed (mem 34→22 ✓, bool Pad valid),
but task056 is **opset 9**, where Slice `starts/ends/axes` and Pad `pads` are FREE node
attributes. `calculate_params` (src/harness.py:139) counts **initializers**, not regular node
attributes — so bumping 9→13 (required for bool Pad; Slice-1→Slice-13 and Pad-2→Pad-13 both move
those attrs into int64 **inputs**) converts ~26 free attrs into counted params. That swamps the
12-byte fp16 saving.

**Why task174 paid but task056 doesn't:** task174 was already at opset 11, so its 11→13 bump forced
no attribute→initializer conversion (Pad already used input-form pads at 11). The lever only pays
when the source net is **already at opset ≥ 11 with input-form shape params** — a bool→Pad/shape-op
sitting at opset ≤ 10 will always lose the migration cost.

## Reusable takeaways
- **Slot-awareness is essential**: 22 of 54 initial hits were casts feeding index/shape slots
  (ScatterND indices are int64-only) — semantically required, not spurious. Filter to DATA slots.
- **Trace the true consumer past the shape op**: a shape-op consumer whose output feeds a
  float-only op (MatMul/Pool/Reduce/Conv/Mul) means the cast pays for the float op.
- **Check the source opset before costing a bool-Pad harvest**: opset ≤ 10 → the 9/2→13 bump
  turns free Slice/Pad/etc node-attributes into counted inits; net loss on small nets.
- **Scatter-onto-input canvases** (updates→ScatterND with data=float model input) are a common
  FALSE+ class: updates dtype is locked to the input canvas.

## Files
- scan: `reports/scripts/boolpad_scan.py` · inspectors: `boolpad_inspect.py`, `boolpad_slot0.py`
- data: `reports/boolpad_scan.json` (32 hits w/ verdicts)
- refuted build: `reports/candidates/task056_boolpad.py`
