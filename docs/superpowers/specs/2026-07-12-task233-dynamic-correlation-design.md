# Task233 Dynamic Correlation Design

## Goal

Reduce task233's deployed cost from 24,703 to below 22,026 while preserving
all bundled outputs, by replacing the 9-bit hash and inverse-table placement
lookup with direct runtime-kernel correlation.

## Current Cost Center

The deployed v3S graph extracts each outside sprite's 3x3 red mask, hashes the
mask, hashes every 3x3 black-hole window inside the cropped box, scatters the
324 window positions into a 512-entry inverse table, and gathers the placement
for each sprite hash. The counted tensors and initializers in this lookup cost
roughly 4.5K.

## Chosen Design

1. Keep the deployed packed-label detector, box extraction, crop, stamping,
   public override lanes, and final rendering unchanged.
2. Reshape the runtime `red [5,9]` sprite masks to `[5,1,3,3]` and cast them to
   uint8.
3. Use those masks as the runtime weight input of `QLinearConv`, with the
   existing uint8 `holm [1,1,20,20]` black-hole map as input. Unit scales and
   zero points make each `[1,5,18,18]` output value the exact overlap count.
4. Apply `MaxPool(kernel_shape=[18,18])` and consume its indices output. Reduce
   each absolute NCHW index modulo 324 to recover the flat placement in the
   18x18 window grid.
5. Convert the resulting five positions to the dtype and shape expected by the
   existing override and stamping tail.
6. Delete the sprite-hash MatMul, fp16 hole-hash Conv, 324-element int32 hash
   plane, 512-entry inverse table, scatter/gather lookup, and their constants.

The generator assigns distinct sprite pixel counts from 4 through 8 and keeps
target sprites separated. Therefore the correct target window reaches the
sprite's full overlap count; a shifted copy cannot contain the same finite
3x3 mask, and a different target has a different pixel count. This makes the
per-channel maximum the desired placement under the generator contract.

## Alternatives Rejected

- Reintroducing the old fp32 detector cannot feed one global TopK without a
  counted fp32 flatten/reshape, leaving less than 1K expected savings.
- Keeping the inverse table and applying local dtype or alias golf is expected
  to save less than 1K.
- Runtime fp16 Conv correlation is semantically similar but its five-channel
  response costs twice as much as the uint8 QLinearConv response.

## Verification

The implementation starts with tests that fail against a deliberately absent
candidate builder or old hash lookup. The candidate must then pass, in order:

1. ONNX checker and ORT session creation under the pinned onnx 1.21.0 and
   onnxruntime 1.26.0 environment.
2. A focused probe proving runtime QLinearConv weights and MaxPool indices have
   the assumed semantics.
3. Exact agreement on all 266 bundled task233 examples (`fail=0`).
4. Cost below both 24,703 and the histogram-derived target 22,026.
5. Isolated-process fresh A/B testing against the deployed v3S graph, with no
   candidate-only regression before adoption is considered.
6. `ng gate`, followed by `ng adopt` only if all normal project gates pass.

## Scope and Stop Conditions

All candidate source and ONNX artifacts stay under `candidates/task233/`.
Neither deployed networks nor `src/custom/task233.py` change during discovery.
If dynamic QLinearConv weights are unsupported, MaxPool indices disagree with
the expected per-channel offsets, bundled correctness fails for a structural
reason, or measured cost is at least 22,026, this design is recorded as a
dated negative result instead of being expanded into unrelated task233 work.

If successful, the runtime-kernel correlation mechanism is registered in
`state/insights.yaml` and the 400-task board is rescanned for sprite-pattern
placement tasks before any adoption or submission.
