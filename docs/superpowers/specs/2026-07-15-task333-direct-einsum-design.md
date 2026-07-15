# Task333 Direct-Einsum Cost-1295 Design

## Scope and Success Criteria

Improve only task333 from the deployed cost-1586 incumbent. The next `+0.1` score threshold is
cost at most 1435; this design targets cost **1295**. Do not reproduce the rejected uint8
`CumSum` graph, the 2000-cost fp16 `CumSum` core, or cached public task333 models. Keep the
ONNX 1.21.0 and ONNX Runtime 1.26.0 pins unchanged.

The candidate is successful only if all of the following hold:

- static pricing reports memory 0, parameters 1295, and cost 1295;
- strict ONNX checking and shape inference succeed;
- an ONNX Runtime 1.26.0 smoke inference completes within the bounded planner search;
- `uv run ng gate candidates/task333/direct_einsum.onnx --task 333` reports bundled fail=0;
- 1500 generated fresh examples report candidate fail=0 and divergence=0 against the oracle;
- adoption, if reached, runs through `uv run ng adopt` and never copies into deployment directly.

No pack or submission is part of this work.

## Task Semantics

The generator places one green 2x2 box and zero or more colored source pixels. A source aligned
with either box row or either box column paints its color through the intervening black cells up
to, but not including, the nearest green cell. All other pixels are preserved.

The generator gives the direct renderer three properties:

1. there are exactly four green pixels;
2. each of the four directions has at most one aligned source;
3. every painted target between an aligned source and the box is black before painting.

These are generator properties, not approximations inferred from the bundled examples.

## Selected Architecture

Replace the complete deployed graph with one fp32 `Einsum`. The FREE input is read three times:
once as the candidate source pixel and twice as two independently selected green pixels. The
single node writes directly to the FREE output, so it creates no counted intermediate tensor.

Let `x[c,r,s]` be the active 10x10 crop of the one-hot input and let `y[o,h,w]` be the output.
The renderer sums five modes:

| Mode | Coordinate predicate | Channel action |
| --- | --- | --- |
| preserve | `(h,w) = (r,s)`; both green coordinates free | identity, weighted by `1/16` |
| horizontal-left | `r=h=u=p` and `s<w<v<q` | replace background with source color |
| horizontal-right | `r=h=u=p` and `s>w>v>q` | replace background with source color |
| vertical-up | `s=w=v=q` and `r<h<u<p` | replace background with source color |
| vertical-down | `s=w=v=q` and `r>h>u>p` | replace background with source color |

Here `(r,s)` is the source, `(h,w)` the target, and `(u,v)`, `(p,q)` the ordered green pair.
Strict ordering excludes both green cells from every ray. In a box row or column, ordering the
two green cells chooses exactly one pair and identifies the nearer edge of the box.

### Reused Relation Factors

Define `I[i,j] = 1[i=j]`, `L[i,j] = 1[i<j]`, `R = L^T`, and `J[i,j] = 1` over local coordinates
0 through 9. Store two banks:

- `A[3,10,10] = [I, L, R]` for the source-to-target edge;
- `B[4,10,10] = [I, J, L, R]` for both target-to-green1 and green1-to-green2 edges.

Four one-hot mode selectors choose an `A` and `B` relation for each axis:

| Mode | row A | row B | col A | col B |
| --- | --- | --- | --- | --- |
| preserve | I | J | I | J |
| horizontal-left | I | I | L | L |
| horizontal-right | I | I | R | R |
| vertical-up | L | L | I | I |
| vertical-down | R | R | I | I |

Because the same selected `B` relation is used on both later edges, these four bank selections
represent every three-edge chain without storing a mode-specific spatial tensor.

### Channel Factors

`green[10]` selects channel 3 for each green operand. `channel[2,10,10]` stores:

- preserve: `channel[0,c,o] = 1[c=o]`;
- ray: for source colors other than black 0 and green 3, `channel[1,c,o]` is `+1` at `o=c`,
  `-1` at `o=0`, and zero otherwise.

A `[5,2]` selector maps preserve to channel matrix 0 and all ray modes to matrix 1. A `[5]`
weight vector is `[1/16, 1, 1, 1, 1]`.

### Crop and Output Embedding

The model interface is `[1,10,30,30]`. A single reused selector `E[10,30]` maps local
coordinates 0 through 9 to the active crop. It is used for all six input coordinates and both
output coordinates. Consequently the output is zero outside the active 10x10 crop without a
counted crop, pad, or reshape node.

The conceptual 26-operand equation is:

```text
ncRS,rR,sS,ngUV,uU,vV,g,ndPQ,pP,qQ,d,
arh,ma,bhu,mb,bup,isw,mi,jwv,mj,jvq,
kco,mk,m,hH,wW->noHW
```

The operands, in order, are the source input and its two `E` factors; green input 1, its two `E`
factors, and `green`; green input 2, its two `E` factors, and `green`; row `A`, row-A selector,
row `B`, row-B selector, repeated row `B`; column `A`, column-A selector, column `B`, column-B
selector, repeated column `B`; channel matrix, mode-channel selector, mode weight; and the two
output `E` factors. Repeated operands refer to the same initializer or FREE input and add no
parameters.

## Exactness Argument

For preserve mode, the two independently summed green operands contribute `4 * 4 = 16`; the
`1/16` weight therefore copies every input one-hot value exactly once.

For a horizontal or vertical ray, equality on the fixed axis restricts both selected greens to
the aligned box row or column. There are exactly two such green pixels, and strict ordering gives
exactly one ordered pair. The three ordered edges select precisely the black cells between the
source and the nearer green pixel. The ray channel matrix adds the source-color logit and
subtracts the preserved background logit. Generator-side direction uniqueness prevents two ray
terms from painting the same target. Thus every output position remains one-hot and agrees with
the procedural draw operation.

The same direct formula has already been checked as a pure NumPy oracle on all 265 bundled cases
and 1500 fresh generated cases, with zero mismatches in both sets. This is diagnostic evidence;
the emitted ONNX must independently pass every gate below.

## Static Price

Only initializer element count is charged because the sole node writes to the FREE output:

| Initializer | Shape | Elements |
| --- | ---: | ---: |
| crop/output selector `E` | 10x30 | 300 |
| relation bank `A` | 3x10x10 | 300 |
| relation bank `B` | 4x10x10 | 400 |
| row-A selector | 5x3 | 15 |
| row-B selector | 5x4 | 20 |
| column-A selector | 5x3 | 15 |
| column-B selector | 5x4 | 20 |
| mode weights | 5 | 5 |
| channel matrices | 2x10x10 | 200 |
| mode-channel selector | 5x2 | 10 |
| green-channel selector | 10 | 10 |
| **Total** | | **1295** |

Expected score gain is `ln(1586 / 1295) = 0.202704428`, clearing the cost-1435 threshold by 140.

## Implementation Boundary

Create only task-local artifacts before adoption:

- `candidates/task333/test_direct_einsum.py`
- `candidates/task333/build_direct_einsum.py`
- `candidates/task333/direct_einsum.onnx`
- `candidates/task333/verify_direct_einsum.py`

The builder constructs a new model rather than rewriting the directional incumbent. It must use
opset and IR versions accepted by the pinned runtime, declare the existing input/output interface,
run ONNX checker plus strict shape inference, and assert the exact initializer count. Do not
modify `submission/overfit_nets/task333.onnx` or `src/custom/task333.py` before successful gates.
After adoption, regenerate exact source and verify its rebuilt SHA against the deployed model.

## Planner Risk and Stop Conditions

ORT `Einsum` contraction planning is operand-order dependent. Start with the semantic order above.
If session creation or one fixed bundled inference does not complete within 20 seconds, try a
deterministic bounded list of at most 11 additional operand orders that move small relation and
selector factors earlier while preserving the same equation. Persist only the fastest passing
order. If none of the 12 total orders completes within the limit, record the lowering as dormant
in the task333 lever ledger and stop; do not split it into counted directional planes or return to
`CumSum`.

Additional hard stops:

- checker or strict shape-inference failure: stop and diagnose before any gate;
- static cost above 1435: do not run fresh verification or adopt;
- any bundled failure: retain cost 1586 and diagnose the first divergent case;
- any fresh failure or candidate/oracle divergence: retain cost 1586;
- no gate weakening, direct deployment copy, or adoption of an inexact candidate.

## Verification Sequence

Use a red-green test cycle:

1. Add a task-local test importing the not-yet-created builder and asserting one `Einsum`, no
   counted intermediates, the exact interface, 1295 initializer elements, and cost 1295. Run it
   first and observe the missing-builder failure.
2. Implement the smallest builder that satisfies the structural test, then run checker and strict
   shape inference.
3. Run one fixed bundled sample through ORT under the 20-second planner limit. If needed, run only
   the bounded operand-order search described above.
4. Run the emitted ONNX against the already-validated direct NumPy formula on all bundled cases.
5. Run `uv run ng gate candidates/task333/direct_einsum.onnx --task 333`; require cost 1295 and
   bundled fail=0.
6. In a fresh process, compare candidate, incumbent, and generator oracle on 1500 fresh examples;
   require candidate fail=0 and divergence=0.
7. Only then run `uv run ng adopt candidates/task333/direct_einsum.onnx --task 333 --note
   "single free-output relational Einsum"`.
8. Regenerate the exact task333 source, rebuild it, and require source/deployed SHA equality.
