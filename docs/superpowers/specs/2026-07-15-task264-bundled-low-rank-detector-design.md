# Task 264 Bundled Low-Rank Detector Design

**Status:** Approved on 2026-07-15

## Goal

Improve task264 beyond the deployed cost-1227 model while preserving bundled
265/265 correctness. The first target is cost at most 1110, but the search must
continue below the first passing rank when a materially larger score gain is
available. Fresh examples are diagnostic only, consistent with the repository's
8000-overfit default.

## Current model and constraint

The deployed model has eight scalar high-arity detector Einsums followed by an
exact support-gated glyph renderer. Its detector shift representation costs 736
parameters:

- `shift_row`: `14 * 16 = 224`
- `shift_col`: `30 * 16 = 480`
- `shift_1`, `shift_2`: `2 * 16 = 32`

The renderer currently reuses this rank-16 shift basis for 126 parameters of
placement state. All eight glyph detectors encounter every possible 14-by-14
top-left coordinate in the 265 bundled examples, so coordinate pruning cannot
remove a useful part of the basis.

The three-state shift tensor has column-unfold rank 16 with nearly flat nonzero
singular values. Therefore an exact shared factor below rank 16 is unavailable.
A lower-rank candidate must be trained against the finite bundled behavior.

## Chosen architecture

### Shared initializer factor

Keep the eight detector nodes byte-identical: same node order, operation type,
equation, input names, output names, and attributes. Only replace the values and
rank dimensions of `shift_row`, `shift_col`, `shift_1`, and `shift_2`.

For rank `r`, the learned detector state costs:

```text
14r + 30r + r + r = 46r
```

Changing the detector rank makes the current reused renderer placement basis
invalid. Decouple placement into an exact block/local embedding costing 180
parameters. The remaining graph costs 365, so the first-stage candidate cost is:

```text
365 + 180 + 46r = 545 + 46r
```

Expected costs are:

| Rank | Cost | Approximate task score gain |
| ---: | ---: | ---: |
| 12 | 1097 | +0.11 |
| 10 | 1005 | +0.20 |
| 8 | 913 | +0.30 |
| 6 | 821 | +0.40 |
| 4 | 729 | +0.52 |

The rank sweep begins at 12 to establish a candidate beyond the next score bar,
then continues through 10, 8, 6, and 4. It does not stop merely because rank 12
passes.

### Training target and loss

Use all 265 bundled task264 examples. Export the deployed detector's eight scalar
glyph-color outputs as teacher targets. Train only the four shift initializers;
the detector equations and channel predicates `A`, `Cg`, and `Bc` stay frozen.

Optimize a staged objective:

1. teacher scalar regression for stable initialization;
2. exact rounded glyph-color agreement for every bundled example;
3. output margin maximization so float32 ONNX Runtime evaluation remains on the
   correct side of every renderer decision boundary.

Use multiple deterministic restarts per rank and retain only candidates that are
finite under float32 export. Warm-start smaller ranks from the nearest passing
higher rank when that is more stable than a fresh factorization.

### Exact renderer

The glyph masks, channel semantics, and chart support remain exact and frozen.
The replacement placement embedding must reconstruct the same 9-by-9 support and
must structurally force every off-chart output to zero. The removed
`Gather -> Equal -> Pad` tail must not be restored or reimplemented.

### Optional exact channel reduction

Only after a detector rank passes, attempt the independent repeated-root channel
factor. It may reduce the exact channel representation from 54 to about 22
parameters, saving another 32 cost. This is a separate stage so a channel change
cannot obscure detector-training failures.

## Fallbacks

### Detector-specific shift coefficients

If no shared rank-12 factor passes, continue to share the row and column bases but
give each of the eight glyph detectors its own `shift_1` and `shift_2`
coefficients. This costs approximately:

```text
545 + (44 + 16)r = 545 + 60r
```

Rank 8 is then expected to cost 1025 before the optional channel reduction. This
separates glyph conflicts while retaining a useful shared spatial basis.

### Detector-core/public graft

Pivot to detector-core or public-graft research if rank 12 cannot pass, any
candidate emits a nonzero off-chart value, or runtime regresses. The currently
available public task264 teachers have minimum static cost 2329, so they are idea
sources rather than direct byte-level graft candidates.

## Verification and adoption

Tests must be written before the new builder/trainer. They must assert:

- detector nodes are byte-identical to the deployed graph;
- only allowed initializer values and the exact renderer placement change;
- candidate generation is deterministic and byte-identical at its fixed point;
- static cost matches the rank formula;
- ONNX checker and strict shape inference pass;
- all 265 bundled outputs are exact;
- every off-chart value is exactly zero;
- runtime does not regress against cost 1227.

Run each promising rank through `uv run ng score 264`. Fresh evaluation is kept
as a diagnostic. Full evaluation, gate, and adoption use
`NG_EVAL_TIMEOUT_SECONDS=1800`. Adoption must go through `ng gate` and
`ng adopt`; no manifest edit or deployment copy may bypass the gate.

The winning rank is the cheapest candidate that retains bundled 265/265 and
acceptable runtime, not merely the first candidate to cross cost 1110.
