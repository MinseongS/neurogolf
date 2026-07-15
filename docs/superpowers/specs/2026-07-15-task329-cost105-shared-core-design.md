# Task329 Cost-105 Shared-Core Design

## Objective

Improve only task329 from deployed cost117 to cost at most105.  The new route must not
repeat the failed rank-2 Adam/hinge search, must prove all 12000 logical states before
writing ONNX, and must retain float32 margin at least0.008.  Any adoption must use the
normal `ng gate` then `ng adopt` path.

## Selected construction

The cost117 graph stores three separate `2 x 2 x 2` tensors: two colour-root cores and
one spatial-root core.  The cost105 graph replaces all three with one shared core and
uses one `2 x 2` swap matrix both to negate colour features and to exchange the
preserve/background spatial branches.

Set `q=32` and define the shared colour feature

```text
F(x) = [32*(x+0.5), 0.5-x].
```

Define

```text
J = [[0, 1/32],
     [32,    0]].
```

For every colour `x`, `F(x) * J = F(-x)`.  The same initializer `J` therefore turns a
left affine root into its sign-reflected right root.  Its off-diagonal entries also map
colour branch zero to spatial branch one with weight `1/32`, and colour branch one to
spatial branch zero with weight `32`.

The shared, scaled core is

```text
core[0] = [[1/1024, -1/32],
           [   3/32,     1]]

core[1] = [[0, 1.25],
           [0,   40]].
```

Used with colour features, its two left/right products are exactly

```text
4 * E(k,c), where E(k,c) = 0.25-(k-c)^2
4 * P(c),   where P(c)   = 400*(0.25-c^2).
```

## Spatial reuse

The dynamic state is

```text
state = [cell_count, -1],
cell_count in {9,25,49,81}.
```

The same core's branch one produces the constant spatial root `2` when the two route
rows satisfy

```text
route0/32 + route1 = -1/20.
```

For a legal count `x`, define

```text
z(x) = (3*x-32) / (-1280*(x-32))
route0(x) = 1024*z(x)
route1(x) = -1/20-route0(x)/32.
```

Columns1..4 use the route for counts9/25/49/81 respectively.  Every other column uses
`z=1/5`.  Shared-core branch zero is zero at the matching state/column and nonzero at
all other logical state/column pairs.  Reusing that root twice forms the nonnegative
background correction.

Because `J[0,1]=1/32` and `J[1,0]=32`, the terminal score is

```text
0.5 * E(k,c) + 512 * P(c) * D(s,w)
= 0.5 * (E(k,c) + 1024*P(c)*D(s,w)),
```

where `D` is the square of the unscaled branch-zero spatial root.  At the matching
middle column `D` is zero up to bounded float32 roundoff.  Away from it, the smallest
logical root magnitude is greater than0.0326, so the amplified correction makes only
background channel zero positive.

## ONNX graph

The graph retains the cost117 three-node topology:

1. `ReduceSum(input, keepdims=1) -> cell_count_keep[1,1,1,1]`.
2. `Concat(cell_count_keep, one_state=-1, axis=1) -> state_vec[1,2,1,1]`.
3. One FREE-output terminal `Einsum`.

The terminal equation is

```text
bkhw,ka,tad,cd,ke,eg,tgi,cf,fi,tu,upq,bqmn,pw,urs,bsxy,rw->bchw
```

Its inputs are

```text
input,
colour_features, shared_core, colour_features,
colour_features, swap, shared_core, colour_features, swap,
swap,
shared_core, state_vec, route_r,
shared_core, state_vec, route_r.
```

Thus `colour_features`, `shared_core`, `swap`, `state_vec`, and `route_r` are all reused
inside the FREE-output contraction; no colour or spatial intermediate is materialized.

## Exact cost

Initializer elements:

| initializer | shape | elements |
|---|---:|---:|
| `colour_features` | `[10,2]` | 20 |
| `shared_core` | `[2,2,2]` | 8 |
| `swap` | `[2,2]` | 4 |
| `one_state` | `[1,1,1,1]` | 1 |
| `route_r` | `[2,30]` | 60 |
| total params | | **93** |

Counted intermediates remain:

| tensor | dtype/shape | bytes |
|---|---:|---:|
| `cell_count_keep` | fp32 `[1,1,1,1]` | 4 |
| `state_vec` | fp32 `[1,2,1,1]` | 8 |
| total memory | | **12** |

Predicted total cost is therefore **105**, giving `ln(117/105)=+0.1082135846`.

## Proof and acceptance order

The builder must expose `--prove-only --json` and must not write an ONNX file until it
has proved:

1. `F(x)J=F(-x)` and both colour products equal `4E` and `4P` bit-exactly in float32.
2. The branch-one spatial root is constant2 within float32 tolerance.
3. The four target branch-zero roots have magnitude at most `3e-8`, while every other
   logical state/column root is nonzero.
4. All 12000 classifier states have `desired > 0`, `wrong <= 0`, and margin at least
   0.008.  The read-only prototype measured desired min about0.125, wrong max about
   -0.375, and margin about0.125.
5. Every score is finite and predicted cost is at most105.

TDD begins with a missing-builder RED.  After proof succeeds, the candidate receives
full ONNX checking and strict shape inference, isolated scorer measurement, official
bundled gate, and fresh1500 comparison.  Only then may `ng adopt` run.

After adoption, `src/custom/task329.py` must reproduce the deployed protobuf byte for
byte, and the current plus historical task329 builders must remain idempotent when they
encounter the newer graph.

## Failure handling

- Do not write ONNX if any identity, logical-state, finite-value, margin, or predicted-cost
  assertion fails.
- Do not adopt if scorer cost exceeds105, bundled fail is nonzero, or official gate errors.
- Treat an ORT rejection or unreasonable 16-operand contraction runtime as a lowering
  failure and retain the deployed cost117 source.
- Fresh is diagnostic in 8000-overfit mode, but record any divergence before adoption.
- Do not modify another task, retry sparse initializers, upgrade ONNX/ORT, pack, or submit.
- Preserve historical rank4 cost270 as the independent fallback.

## Alternatives not selected

- Sharing only the two colour cores reaches predicted cost109 and misses the target.
- Sparse initializers would reduce dense-zero accounting but are already rejected by the
  pinned scorer's strict shape inference and are explicitly closed in project records.
- A new global sign solver could go below105, but is unnecessary while this rational,
  proof-first construction reaches the requested threshold with wide float margin.
