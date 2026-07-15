# Task329 Cost-117 Exact Route Design

## Objective

Improve only task329 from deployed cost 134 to cost at most 121 without repeating the
failed rank-2 Adam/hinge search.  The candidate must be an exact construction, must prove
all `4 * 30 * 10 * 10 = 12000` logical classifier states before an ONNX file is written,
and must have float32 margin at least 0.008.  Adoption must use the normal
`ng gate` then `ng adopt` path.

## Selected construction

Keep the current dynamic cell-count spatial code, but factor both quadratic colour
polynomials into repeated affine roots.  At the same time, preserve the reduced cell
count's dimensions so the graph no longer needs `Unsqueeze` or its axes initializer.

For input colour `k` and output colour `c`, use the shared two-feature basis

```text
C(x) = [1, x].
```

Two `2 x 2 x 2` cores produce the left and right affine roots.  Their branch-zero
matrices are

```text
L[0] = [[ 0.5,  1], [-1, 0]]  ->  0.5 - k + c
R[0] = [[ 0.5, -1], [ 1, 0]]  ->  0.5 + k - c,
```

so their product is exactly

```text
E(k,c) = 0.25 - (k-c)^2.
```

Their branch-one matrices are

```text
L[1] = [[10, -20], [0, 0]]  ->  10 - 20c
R[1] = [[10,  20], [0, 0]]  ->  10 + 20c,
```

so their product is exactly

```text
P(c) = 100 - 400c^2 = 400 * (0.25-c^2).
```

The existing spatial construction remains exact.  Legal cell counts are
`9, 25, 49, 81`; columns 1 through 4 store those codes and every other column stores
zero.  With `state=[cell_count,1]`, the shared spatial core produces branch roots
`1` and `column_code-cell_count`.  Reusing each root twice gives

```text
D(s,w) = (column_code(w)-cell_count(s))^2.
```

The final score remains

```text
score(k,c,s,w) = E(k,c) + D(s,w) * P(c).
```

At the selected middle column `D=0`, only `c=k` is positive.  At every other column
`D>=1`, only background channel `c=0` is positive.

## ONNX graph

The graph has three nodes:

1. `ReduceSum(input, keepdims=1) -> cell_count_keep[1,1,1,1]`.
2. `Concat(cell_count_keep, one_state, axis=1) -> state_vec[1,2,1,1]`.
3. One FREE-output `Einsum` that reuses the colour basis, spatial core, state vector,
   and width route.

The terminal equation is

```text
bkhw,ka,tad,cd,ke,tef,cf,tpq,bqij,pw,tur,brlm,uw->bchw
```

with inputs

```text
input,
colour_features, colour_left, colour_features,
colour_features, colour_right, colour_features,
root_core, state_vec, route_r,
root_core, state_vec, route_r.
```

The singleton `i,j,l,m` dimensions disappear inside the terminal contraction.  The
output remains FREE under the scorer.

## Exact cost

Initializer elements:

| initializer | shape | elements |
|---|---:|---:|
| `colour_features` | `[10,2]` | 20 |
| `colour_left` | `[2,2,2]` | 8 |
| `colour_right` | `[2,2,2]` | 8 |
| `one_state` | `[1,1,1,1]` | 1 |
| `root_core` | `[2,2,2]` | 8 |
| `route_r` | `[2,30]` | 60 |
| total params | | **105** |

Counted intermediates:

| tensor | dtype/shape | bytes |
|---|---:|---:|
| `cell_count_keep` | fp32 `[1,1,1,1]` | 4 |
| `state_vec` | fp32 `[1,2,1,1]` | 8 |
| total memory | | **12** |

Predicted total cost is therefore **117**, giving `ln(134/117) = +0.1356658652`.
This clears the required cost-121 threshold without depending on sparse-initializer
accounting or uncounted constants.

## Proof and test order

The builder exposes a proof-only mode and does not write ONNX until it has established:

1. Both affine-root products equal the historical `E` and `P` tables bit-exactly in
   float32 for all ten input and output colours.
2. The repeated spatial roots equal `[1,D]` for all four sizes and thirty columns.
3. Across all 12000 logical states, `desired > 0`, `wrong <= 0`, and margin is at
   least 0.008.
4. The predicted static cost is at most 121.

TDD starts with a new proof-first test whose missing builder failure is observed before
implementation.  After proof succeeds, the builder may write the candidate.  The
candidate then receives ONNX checker/strict shape inference, official bundled gate,
fresh1500 incumbent/candidate comparison, and only then `ng adopt`.

After adoption, `src/custom/task329.py` must rebuild the deployed protobuf byte-for-byte.
The current and historical task329 builders must remain idempotent when they encounter
the newer deployed graph.

## Failure handling

- Do not write an ONNX candidate if the logical proof or 0.008 margin assertion fails.
- Do not adopt if actual scorer cost exceeds 121, bundled fail is nonzero, or the
  official gate reports an error.
- Treat ORT rejection or unreasonable contraction runtime as a graph-lowering failure;
  retain deployed cost134 and historical rank4 cost270 fallback.
- Fresh validation is diagnostic in the project's 8000-overfit mode, but any candidate
  divergence is recorded before an adoption decision.
- Do not modify another task, upgrade ONNX/ORT, pack, or submit as part of this work.

## Alternatives not selected

- A rational/global solver over a new sign factor may find a smaller representation,
  but it is unnecessary for the current threshold and introduces margin and search-risk.
- A single shared colour-root core could potentially remove more parameters, but needs
  additional selectors or a coupled algebraic identity.  It is deferred until cost117
  becomes the incumbent.
- Restoring the explicit four-way selector or counted quadratic staging is strictly
  worse than the current dynamic-state representation and is prohibited for this step.
