# Task019 Min-Basis Whole Renderer Design

Date: 2026-07-15 KST
Scope: task019 only

## Objective

Replace the deployed cost-1934 residual QLinearConv graph with an exact
H/W-conditioned whole renderer costing at most 1750. The candidate must pass
ONNX checker and strict shape inference, all 267 bundled examples, `ng gate`,
and a fresh A/B comparison with zero divergence before adoption through
`ng adopt`.

The design must not repeat the exhausted M/D/R, categorical E/D, literal
`E=R-5M`, or 1200-element dense halo attempts.

## Chosen architecture

Use one terminal float32 `Einsum` to fuse point placement, diagonal halo
placement, and channel classification. The terminal output is free under the
scorer, while all detector intermediates and initializer elements are priced
explicitly before model construction.

The contraction is:

```text
bvrc,p,ra,apgq,qR,w,cA,AwgQ,QC,gvV->bVRC
```

Indices have these meanings:

- `b`: batch
- `v`, `V`: input and output colour channel
- `r`, `c`: input row and column
- `R`, `C`: output row and column
- `p`, `w`: height and width state features
- `a`, `A`: six-dimensional source bases
- `q`, `Q`: twelve-dimensional output bases
- `g`: shared glyph type, point (`g=0`) or diagonal halo (`g=1`)

Sharing `g` between the row selector, column selector, and channel mixer
prevents point/halo cross terms. In particular, halo is the product of the
row ±1 relation and column ±1 relation, so it creates only the four diagonal
neighbours.

## Exact H/W detector

The input generator guarantees:

- `2 <= H,W <= 6`;
- each point row contains exactly one coloured point;
- point rows are separated by two or three rows.

Therefore:

- Height is recovered from background channel slice `[0:6,0:2]`. Every valid
  row has at least one background cell in the first two columns because
  `W>=2` and a row contains at most one point. `ReduceMax` across the two
  columns followed by `ReduceSum` gives H.
- Width is recovered from background channel slice `[0:2,0:6]`. At most one
  of the first two rows contains a point, so the maximum of their background
  row sums is W.

The two Slice nodes share `starts=[0,0,0,0]` and use separate ends. Axes are
omitted, saving one initializer element compared with the existing Slice.
The old `colored_fund` plane and zero initializer are not retained because
the whole renderer consumes the free input directly.

Convert H and W to exact five-element state features with:

```text
Min(H, [2,3,4,5,6])
Min(W, [2,3,4,5,6])
```

For H=2..6 the feature matrix is lower triangular with determinant two. Its
inverse contains only integers and halves, so the selector coefficients are
exactly representable in float32. In-memory analysis reconstructed every
selector coefficient with zero float32 error.

## Joint point/halo selector

Define `T[h,r,g,R]` as follows:

- `g=0`: coefficient one at `R=r` and `R=r+h` for `r<h`;
- `g=1`: sum of the clipped ±1 shifts of those two point locations;
- all other positions: zero.

Factor the tensor into:

- source basis `A[r,a]`, shape `[30,6]`;
- joint core `G[a,p,g,q]`, shape `[6,5,2,12]`;
- output basis `B[q,R]`, shape `[12,30]`.

The source and output bases are exact rectangular identities. The core uses
only the values `{-2,-1,0,0.5,1,1.5,2}` after transformation into the Min
state basis. Reusing the same three factors for rows and columns gives all 25
H/W states without the exhausted 1200-element ±1 dense halo extension.

## Channel mixer

Use a dense initializer `C[g,v,V]` with shape `[2,10,10]`.

For `g=0` (point/base relation):

- copy every input channel to the same output channel;
- subtract five from cyan output for each active non-background, non-cyan
  input point.

For `g=1` (halo relation), for active point colours 1..7 and 9:

- subtract one from the background output;
- add one to the cyan output.

This implements:

```text
background = tiled_background - diagonal_halo_count
cyan       = diagonal_halo_count - 5*tiled_point
colour k   = tiled_colour_k
```

True logits have minimum one. False logits have maximum zero. The generator
excludes cyan as an input point colour.

## Static cost gate

No ONNX candidate may be built unless this accounting remains at or below
1750.

Initializer parameters:

| Item | Elements |
|---|---:|
| shared Slice starts | 4 |
| height Slice ends | 4 |
| width Slice ends | 4 |
| Min caps | 5 |
| source basis `[30,6]` | 180 |
| joint core `[6,5,2,12]` | 720 |
| output basis `[12,30]` | 360 |
| channel mixer `[2,10,10]` | 200 |
| **Total params** | **1477** |

Counted runtime memory:

| Tensor | Bytes |
|---|---:|
| height probe `[1,1,6,2]` | 48 |
| height row max `[1,1,6,1]` | 24 |
| H scalar | 4 |
| width probe `[1,1,2,6]` | 48 |
| width row sums `[1,1,2,1]` | 8 |
| W scalar | 4 |
| H Min features | 20 |
| W Min features | 20 |
| **Total memory** | **176** |

Predicted total cost is `1477 + 176 = 1653`. This is 281 below the deployed
cost 1934 and implies a point gain of `ln(1934/1653) = 0.1569985782`.

## Padding and stored-bias proof

The candidate contains no QLinearConv node. Consequently:

- implicit QLinearConv padding tuple: `()`;
- stored QLinearConv bias set: `()`;
- every stored bias is non-positive vacuously.

The terminal Einsum has no additive bias. Canvas and footprint padding are
represented by exact zero support in the selector tensor. The output basis is
zero beyond coordinate 11, and each H/W selector is zero outside `2H x 2W`.
If implementation introduces any QLinearConv, the build must stop until its
implicit padding semantics and every stored bias are re-proved non-positive.

## Implementation and verification

Implementation stays under `candidates/task019/` until all gates pass.

1. Add a regression test that checks the 1653 accounting, exact reconstruction
   for all 25 H/W states, the empty QLinearConv/bias audit, and deterministic
   generation.
2. Observe the test failing before implementing the builder.
3. Build the candidate only because the predicted cost is below 1750.
4. Run ONNX checker, strict shape inference, and an ORT 1.26 load/probe.
5. Require bundled 267/267 and measured cost at most 1750.
6. Run `uv run ng gate <candidate> --task 019` and require PASS.
7. Run fresh A/B against the deployed/source-owned incumbent and require zero
   candidate failures, zero incumbent failures, and zero divergence.
8. Adopt only with `uv run ng adopt`; never copy into the deployment directory.
9. Regenerate the exact source-owned task019 builder after adoption and require
   its rebuilt ONNX SHA to match the adopted candidate.

No pack or submission action is in scope.

## Failure and pivot rule

If measured cost exceeds 1750, checker/strict inference fails, ORT rejects the
contraction, bundled evaluation is not 267/267, gate rejects, or fresh A/B
diverges, do not adopt. Record the concrete dated negative result in the
appropriate `state/levers.yaml` ledger with run, verdict, reopen trigger, and
falsification history.

After such a failure, do not repeat categorical E/D candidates. Pivot only to
a genuinely new fused carrier operation or a new exact factorization whose
full pre-build accounting is at most 1750.
