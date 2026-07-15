# task124 next-tier design — exact rank-4 bank and QLinear row hash

## Objective and invariants

Reduce the adopted task124 graph from **memory878 + params61 = cost939** to
**cost <=849**, which is the next `+0.1` score boundary, while preserving all of
the following:

- task124 only; scratch artifacts remain under `candidates/task124/`;
- bundled gate fail 0 and candidate cheaper than the deployed model;
- fresh diagnostic fail 0, raw candidate/incumbent divergence 0, and off-grid
  positives 0;
- the padded uint8 `QLinearConv` remains the direct graph-output producer;
- no valid plane, runtime-colour crop, scalar `Pad`, or terminal `Equal` tail;
- adoption only through `ng gate` followed by `ng adopt`; no pack or submit.

## Approved approach

Compose two independent exact graph folds. Together they save exactly 90 cost:

| fold | memory delta | params delta | cost delta |
|---|---:|---:|---:|
| keep the compact row bank rank 4 | -80 | -1 | -81 |
| replace cellwise p3 equality with exact QLinear row hashes | -19 | +10 | -9 |
| **total** | **-99** | **+9** | **-90** |

The predicted result is **memory779 + params70 = cost849**. Shape inference and
the official scorer remain authoritative; any measured cost above 849 rejects
this candidate for the current tier.

## Fold 1: rank-4 compact row bank

The adopted graph reshapes three ten-cell rows to rank 1, concatenates a
46-element flat bank, slices five flat rows, concatenates a 50-element
`bottom_flat`, and reshapes it back to `[1,1,5,10]`. None of those rank changes
carry semantics.

The replacement is:

```text
false8 [1,1,1,8]
row0_4d, p3_rows_a, row2_4d [1,1,1,10]
  -> Concat(axis=3) = fg_pad4d [1,1,1,46]
  -> five Slice(axis=3) = bottom_row_i [1,1,1,10]
  -> Concat(axis=2) = bottom_fg [1,1,5,10]
```

This deletes `row0_flat`, `row1_flat`, and `row2_flat` (30 bytes), deletes
`bottom_flat` (50 bytes), and deletes the one-element `row_flat_shape`
initializer. Existing starts, ends, and steps are reused; only the existing
slice axis value changes from 0 to 3.

## Fold 2: exact QLinear p3 row hash

The adopted p3 detector compares ten cells and then reduces them:

```text
Equal [10] -> Cast [10] -> ReduceMin [1] -> Cast [1]
```

This costs 22 bytes of intermediates. Replace it with two scalar
`QLinearMatMul` fingerprints and one scalar `Equal`:

```text
p3_rows_a [1,1,1,10] --QLinearMatMul(hash_code [10,1])--> p3_hash_a [1,1,1,1]
p3_rows_b [1,1,1,10] --QLinearMatMul(hash_code [10,1])--> p3_hash_b [1,1,1,1]
Equal(p3_hash_a, p3_hash_b) -> is_p3 [1,1,1,1]
```

`p3b_idx` changes from scalar `4` to rank-1 `[4]` without changing parameter
count, so both gathered rows remain rank 4. The new uint8 initializer is:

```text
hash_code = [1, 2, 4, 8, 16, 32, 0, 0, 0, 0]^T
```

Both matrix multiplies reuse the existing scale `1.0` and uint8 zero point `0`
for A, B, and Y. Centered row values are uint8 `0/2`, so the result is the exact
integer dot product. Its maximum is `2*(1+2+4+8+16+32)=126`; therefore uint8
saturation cannot occur.

The rank-4 boolean is intentionally not squeezed. Broadcasting it through the
existing `Where` is legal; the following `Gather(source_offsets, candidate)`
gains rank, so its five-way `Split` axis moves to the row axis at index 4. The
five one-element results are already reshaped to the Slice start shape, so this
rank propagation adds no counted elements. Adding a `Squeeze` would cost one
byte and miss the target at cost850.

### Why the six-bit fingerprint is collision-free

The generator constrains sprite width and left offset to `wide <=3` and
`offset <=3`. Thus an unshifted sprite row occupies only columns 0 through 5.
For `tall=3`, the generator forces `diag=0`, so rows 1 and 4 are both unshifted
and the six-bit encoding covers every possible foreground cell. For `tall=2`:

- with `diag=0`, both compared rows are again wholly within columns 0 through 5;
- with `diag=1` and `wide=1`, the shift is zero and both rows remain wholly
  within the encoded range;
- with `diag=1` and `wide>=2`, row 1 is a nonempty unshifted sprite row, while
  row 4 is shifted right by `2*(wide-1)`. The two possible occupied column
  intervals are disjoint. The row-1 fingerprint is therefore nonzero and cannot
  equal the shifted row-4 fingerprint, even when part of row 4 lies beyond
  column 5.

The generator explicitly rejects sprites that do not occupy every row, so the
nonempty-row premise holds. Consequently hash equality is equivalent to the
adopted ten-cell equality for every reachable generator instance.

An independent exhaustive enumeration of all 1,428 legal combinations of
`tall`, `wide`, full-row/full-column sprite subsets, offsets, and allowed
diagonal modes found zero disagreements between full ten-cell equality and the
six-bit hash decision.

## TDD and validation contract

Before the builder exists, add a regression test that imports it and fails. The
test must then enforce:

1. ONNX checker and strict shape inference pass under the pinned environment.
2. `QLinearMatMul` fingerprints and the rank-4 row bank have the exact expected
   shapes and inputs.
3. Removed `Reshape`/cellwise p3-reduction intermediates are absent.
4. The graph output is uint8 and is produced directly by the padded
   `QLinearConv` with the existing centered-mask and runtime-colour weights.
5. Bundled evaluation is fail 0 at exactly memory779, params70, cost849.
6. Bundled raw output is byte-identical to the deployed incumbent and has zero
   positives outside the 10x10 valid output area.
7. An exhaustive parameter-domain or equivalent adversarial test proves the
   hash decision matches full ten-cell row equality, including diagonal cases
   whose shifted row reaches columns 6 through 9.
8. Fresh 2000 A/B has candidate fail 0, incumbent/candidate divergence 0, and
   off-grid positives 0.

Only after these pass should the candidate go through isolated `ng gate`. Run
`ng adopt` only if gate fail is 0 and the measured candidate is cheaper. After
adoption, rebuild from `src/custom/task124.py` and require SHA identity among
source-built, candidate, and deployed artifacts before synchronizing
`DISCOVERY.md`, task124 state, the live lever, insight, and `STATE.md`.

## Rejection and fallback

Reject this route without adoption if pinned ORT disagrees with the inferred
rank broadcasting, QLinearMatMul produces any non-exact fingerprint, measured
cost exceeds 849, or any bundled/fresh/off-grid check fails. The safe fallback
is the unchanged cost939 deployed graph; no partial fold is worth adopting in
this tier because rank-4 bank alone reaches only cost858 and does not cross the
next score boundary.
