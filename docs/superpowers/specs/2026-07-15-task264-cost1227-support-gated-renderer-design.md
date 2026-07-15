# Task264 Cost1227 Support-Gated Renderer Design

## Objective

Reduce the deployed task264 graph from cost1361 to at most1231 while preserving the existing
eight scalar detector Einsums and the already-adopted FREE-output glyph renderer. The candidate
must keep off-chart output exactly zero, avoid the removed Gather/Equal/Pad tail, and pass the
normal bundled gate without changing its criteria.

## Chosen representation

The current terminal stores 180 parameters for exact block/local output embeddings and132 for an
exact matrix factor of `glyph_pattern`. Replace these with two coordinated changes.

First, decompose the two spatial pattern slices as a five-term CP factor: one rank-1 term for the
all-gray 9x9 chart and four exact terms for the rank-4 coloured-glyph mask. The factors
`glyph_row[5,3,3]`, `glyph_col[5,3,3]`, and `glyph_term[5,2]` cost100 parameters and reconstruct the
current pattern exactly, saving32.

Second, reuse the detector's existing `shift_col[30,16]` interpolation matrix for output placement.
Store only `block_shift[3,16]` for powers0/3/6 and `local_shift[3,16]` for powers0/1/2. Their product
with `shift_col` approximates the nine coordinate one-hots with maximum on-chart error0.00055.
Multiply both output axes by a shared exact `chart_support[30]` whose first nine values are1 and
the rest0. This makes every off-chart value exactly zero. The new placement parameters cost126,
saving54.

The final terminal remains one FREE-output Einsum. It keeps the current exact rank-3 channel
factor unchanged and grows from10 to15 operands.

## Detector carrier reductions

Delete `shift_0[16]` from every detector equation because it is exactly all ones; removing the
operand leaves the detector contraction unchanged and saves16 parameters.

Feed the eight pre-Round detector scalars directly into the existing colour Concat and remove the
eight Round nodes. Across all265 bundled examples, the maximum observed distance from the rounded
integer is0.04417. The equality separator remains sign-correct for deviations below0.5, and a
conservative endpoint analysis at deviation0.045 combined with the approximate renderer gives a
minimum signed output margin of0.2372. Removing the Round outputs saves32 memory bytes.

## Cost model

- Current: memory208 + params1153 = cost1361.
- Round-output removal: memory208 ->176.
- Delete `shift_0`: params -16.
- Glyph CP rank5: params132 ->100, saving32.
- Support-gated interpolated embedding: params180 ->126, saving54.
- Candidate: memory176 + params1051 = **cost1227**.
- Expected score gain: `ln(1361/1227)`, above the required0.1 threshold.

## Failure and pivot conditions

Reject this representation before a full gate if any of the following occurs:

- static cost exceeds1231;
- any off-chart raw output is nonzero;
- any bundled output sign differs from the incumbent/oracle;
- a fresh-process one-example run exceeds20 seconds, or the candidate's median over the same20
  bundled examples exceeds110% of the cost1361 incumbent median;
- the terminal contraction fails to load or finish under pinned ONNX1.21.0/ORT1.26.0.

On rejection, do not restore the removed Gather/Equal/Pad tail. Pivot to a detector-core rewrite or
a source-owned public-teacher mechanism with a separate cost proof at or below1231.

## Implementation and verification

1. Add a failing test that requires exact CP reconstruction, exact chart support, unchanged channel
   factors, removal of Round/`shift_0`, static cost1227, persistent artifact equality, and a
   byte-identical fixed point.
2. Implement a source-owned transform under `candidates/task264/` from the deployed cost1361 graph.
3. Verify checker and strict shape inference, then compare raw/sign output and runtime on bounded
   bundled examples.
4. Run the focused fixed-point tests.
5. Use `NG_EVAL_TIMEOUT_SECONDS=1800` for full evaluation, `ng gate`, and `ng adopt`; never bypass
   the normal bundled fail0 and cheaper-than-deployed conditions.
6. Treat fresh generated examples as diagnostic only; they do not replace the bundled adoption gate.
7. After adoption, regenerate `src/custom/task264.py`, rebuild `networks/task264.onnx`, and require
   source/candidate/deployed/network SHA equality.

## Files

- `candidates/task264/test_support_gated_renderer.py`: TDD, algebra, cost, and fixed-point checks.
- `candidates/task264/build_support_gated_renderer.py`: source-owned graph transform.
- `candidates/task264/support_gated_renderer.onnx`: persistent candidate artifact.
- `candidates/task264/DISCOVERY.md`: measurements and next threshold.
