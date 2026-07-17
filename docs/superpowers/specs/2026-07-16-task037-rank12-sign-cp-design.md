# Task037 Rank-12 Sign-CP Stretch Design

## Objective

Improve only task037 from the adopted opset-20 cost1284 quadratic-product renderer while
preserving semantic equality on fresh generator examples and exact zero logits outside the
10x10 output grid.  The candidate must keep all six selected slots, must not reintroduce
`ScatterElements`, and may be adopted only through `ng gate` then `ng adopt`.

## Current cost and constraint

The deployed graph costs `642 memory + 642 params = 1284`.  Its terminal FREE-output Einsum
expands one slot predicate into fourteen coefficient terms and multiplies six independently
selected predicates.  Seven private Einsum labels are needed per slot, so the current equation
already consumes 42 of the 48 labels left after reserving `N,K,R,C`.  A direct four-branch
affine-root expression needs more than 52 ASCII labels and is therefore not a legal ONNX
Einsum lowering for six independent slots.

The exact predicate is

```text
q = (r-lo)(lo+span-r) + 1/4
    - 10(c-base-sign*r)^2
    - 10[k>0](k-colour)^2.
```

The renderer needs only the sign of each predicate.  The six-factor product remains exact when
every present and absent slot keeps the incumbent predicate sign, because generator segments do
not overlap and off-grid support is supplied separately by the row/column bases.

## Selected staged construction

### 1. Exact descriptor packing

Keep `selected_base` and `selected_sgn` as INT8.  Cast only `selected_lo_safe`, `selected_n`, and
`selected_kvec` to INT8, concatenate `[one, lo, span, base, sign, colour]` as a 36-byte INT8
descriptor, and cast that descriptor once to FLOAT.  This replaces five 24-byte FLOAT Cast
outputs plus a 144-byte FLOAT Concat with three 6-byte INT8 Cast outputs, a 36-byte INT8 Concat,
and one 144-byte FLOAT Cast.  The exact saving is 66 cost units.

### 2. Repeated affine row basis

Replace `[1,r,r^2]` with two reuses of `[1,r]`.  A rank component selects one affine row factor
from each copy.  Relabel the descriptor's two singleton axes as the already-output singleton
`N` (`NNas`) so one new private row label remains available for each of the six slots without
exceeding the 52-letter Einsum limit.  Fold each component weight into its left descriptor
factor.

### 3. Rank-12 sign-preserving CP tensor

Represent the coefficient tensor over

```text
[descriptor-left=6, descriptor-right=6,
 row-left=2, row-right=2, column=3, channel=4]
```

with twelve CP components.  Store six FLOAT factor matrices with shapes
`[12,6]`, `[12,6]`, `[12,2]`, `[12,2]`, `[12,3]`, and `[12,4]`.  Normalize every component
before float32 freezing so no single factor carries the CP scale ambiguity.  The predicted
rank-12 endpoint is:

| item | saving from 1284 |
|---|---:|
| INT8 descriptor packing | 66 |
| rank14/weight basis to rank12 repeated-row CP | 46 |
| `[1,r,r^2]` to `[1,r]` initializer | 30 |
| total | **142** |

Predicted cost is therefore **1142**.  If lower ranks pass the same proof, expected costs are
1119 at rank11, 1096 at rank10, and 1073 at rank9.  Rank12 is implemented first; lower ranks are
optional only after the rank12 endpoint is fully verified.

## Evidence before implementation

A float64 rank-12 fit over the six-mode coefficient tensor reached relative coefficient error
`1.1213378437483326e-06`.  After float32 reconstruction, a conservative finite audit covered
5,139 valid-or-absent descriptors and all 1000 `(row,column,class)` states per descriptor:

```text
states=5,139
predicate evaluations=5,139,000
sign flips=0
zero ties=0
minimum signed margin=0.19507336616516113
maximum absolute value error=0.15185546875
```

This is feasibility evidence, not adoption evidence.  The frozen factor operands must repeat
the audit, and pinned ORT inference must still prove that its contraction order keeps all signs.

## Test and adoption gates

Tests are written before the builder and must observe the missing-builder failure.  The builder
must not write ONNX until its frozen factors pass:

1. full checker and strict shape inference at opset20;
2. the 5,139-state / 5,139,000-predicate sign audit with zero flips and zero ties;
3. every bundled example with fail0, classes0..9 observed, finite logits, and exact raw off-grid
   zero;
4. a fresh incumbent/candidate differential with candidate fail0 and divergence0;
5. a fresh-process runtime comparison no worse than 2x the cost1284 incumbent;
6. actual scorer cost at most1142.

Only after those checks pass may `uv run ng gate` and then `uv run ng adopt` run.  After adoption,
`src/custom/task037.py`, the candidate, and the deployed ONNX must rebuild byte-identically.

## Failure handling

- A coefficient-small candidate with one predicate sign flip is rejected.
- ORT load failure, non-finite output, off-grid nonzero, bundled failure, fresh divergence, or
  runtime above2x rejects rank12.
- If rank12 fails, retain the deployed cost1284 while building the exact descriptor-packed
  rank14 control, predicted cost1188; adopt that control only if it passes the same external
  gates.
- Do not modify another task, change ONNX/ORT pins, pack, or submit.

