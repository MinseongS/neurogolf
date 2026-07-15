# Task333 Cost-1586 Lowering Design

## Scope

Improve only task333 from the deployed cost-1786 incumbent. Do not reproduce the rejected
uint8 `CumSum` graph, the 2000-byte fp16 `CumSum` core, or cached public task333 models whose
best measured cost is 2401. Keep the ONNX 1.21.0 and ONNX Runtime 1.26.0 pins unchanged.

The candidate is successful only if all of the following hold:

- statically priced cost is at most 1616;
- `uv run ng gate ... --task 333` reports bundled fail=0;
- 1500 generated fresh examples report candidate fail=0 and divergence=0 against the oracle;
- adoption, if reached, runs through `uv run ng adopt` and never copies into deployment directly.

## Selected Architecture

Apply two exact algebraic folds to the current live graph while leaving its four directional
`MaxPool` carries and two prefix/suffix `Min` planes unchanged.

First, replace the three counted 10x10 planes `Hf`, `Vf`, and `merged` with two planes:

```text
hmerge = Where(rbox, Hmin, bar)
merged = Where(cbox, Vmin, hmerge)
```

This is exact on task333's generator. Outside a box row or column, `bar` is preserved. On a box
row, `Hmin` is at least `bar` and supplies the horizontal hull. On a box column, `Vmin` is at
least `bar` and supplies the vertical hull. The only simultaneous box-row/box-column cells are
the green 2x2 box, where `bar`, `Hmin`, and `Vmin` all equal the green code. The old zero branch
and final `Max` are therefore unnecessary.

Second, replace scalar `BitwiseAnd` followed by `Concat` with one broadcast `BitwiseAnd`:

```text
hull_features = BitwiseAnd(merged, masks=[255, 27])
```

The mask has shape `[1,2,1,1]`, so channel zero is `merged & 255 == merged` and channel one is
the incumbent's `merged & 27`. The existing terminal `QLinearConv`, weights, biases, scales, and
padding remain byte-for-byte unchanged.

## Static Price

The incumbent is memory 1740 plus 46 initializer elements, cost 1786.

- Removing one of the three old select/merge planes saves 100 bytes.
- Removing the separate `merged_bits` plane saves 100 bytes.
- The old scalar `z` initializer is removed.
- The old scalar bit mask is replaced by a two-element mask, so initializer count changes by
  `-1 +1 = 0` overall.

The candidate is therefore memory 1540 plus 46 initializer elements, cost **1586**. Its expected
score gain is `ln(1786 / 1586) = 0.1187633592`, and it clears the required cost-1616 ceiling by
30.

## Implementation Boundary

Create a task-local builder and candidate only under `candidates/task333/`. The builder must load
the deployed task333 graph, assert that the expected incumbent nodes and initializers are present,
perform only the two folds above, run ONNX checker plus strict shape inference, and save a new
candidate without modifying `submission/overfit_nets/`.

Do not change `src/custom/task333.py` until the candidate has passed bundled and fresh gates. After
a successful adoption, regenerate or update the exact source so it reproduces the adopted graph.

## Error Handling and Stop Conditions

- If ORT rejects broadcast `BitwiseAnd`, stop this lowering; do not substitute CumSum or fp16.
- If static scoring exceeds 1616, do not run the expensive fresh gate.
- If any bundled example fails, retain the incumbent and diagnose the first divergent tensor.
- If bundled passes but any of 1500 fresh examples fails or diverges, retain the incumbent.
- Do not weaken the success criteria and do not adopt a merely cheaper but inexact candidate.

## Verification

Use a red-green test cycle:

1. Add a task-local test that imports the not-yet-created builder and asserts the expected nodes,
   shapes, memory 1540, parameter count 46, and cost 1586; run it and observe the missing-builder
   failure.
2. Implement the minimal builder and rerun the structural test to green.
3. Run strict checker/shape inference and an isolated ORT smoke inference.
4. Run `uv run ng gate candidates/task333/fused_select_broadcast_hull.onnx --task 333` and require
   bundled fail=0 with cost 1586.
5. Compare incumbent, candidate, and generator oracle on 1500 fresh examples in a fresh process;
   require candidate fail=0 and divergence=0.
6. Only then run `uv run ng adopt ... --task 333 --note "nested directional select + broadcast
   bitwise hull feature fold"`.
