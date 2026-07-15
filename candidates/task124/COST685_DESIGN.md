# task124 exact composite design — corrected cost686 target

## Status and objective

This design was approved on 2026-07-16. It continues only task124 from the deployed
**memory779 + params70 = cost849** graph and seeks the largest already-priced exact joint
reduction, rather than stopping at the next `+0.1` boundary.

The robust primary target is **memory623 + params63 = cost686**, worth an estimated
**18.469122 points**, or **+0.213182** from cost849. The official isolated scorer is
authoritative; these numbers are a pre-build tensor/initializer accounting contract.

The initially approved pre-build accounting was cost685. Implementation-plan self-review found
that opset12 makes every `QLinearConv`/`QLinearMatMul` zero-point input mandatory. A pinned local
schema/ORT probe also rejected sharing the rank-4 Scatter update as a QLinear zero point and
rejected a scalar Scatter update. Retaining the required scalar zero point adds one parameter to
every stage; this correction changes no graph semantics or implementation scope.

The following invariants remain mandatory:

- modify only task124 source, candidates, tests, and task124-specific records;
- preserve the padded uint8 `QLinearConv` as the direct graph-output producer;
- bundled gate fail=0 and candidate cost strictly below the deployed artifact;
- fresh raw candidate/incumbent divergence=0 and off-grid positives=0;
- preserve pinned `onnx==1.21.0` and `onnxruntime==1.26.0`;
- adopt only through `ng gate` followed by `ng adopt`;
- synchronize `src/custom/task124.py` and `candidates/task124/DISCOVERY.md` after adoption;
- do not run `ng pack` or `ng submit`.

## Chosen architecture

The design keeps the current 5x10 channel-0 float crop and compact dynamic-row renderer. It
changes the stored polarity of the centered mask, deletes two redundant materializations, narrows
the geometry controller, and makes the bounded row fingerprint consume only the six columns that
the generator proof requires.

The full graph remains a composition of four independently testable units:

1. **Inverse quantized mask encoder:** channel-0 values `0/1` become foreground/background
   uint8 values `0/2` in one `QuantizeLinear` operation.
2. **Compact period/shift controller:** separate row 0/2 reads recover the shift, while exact
   six-column QLinear hashes distinguish period 3.
3. **Rank-preserving row renderer:** the existing compact row bank and five dynamic Slices emit
   rows 5..9, which are concatenated directly with the top five rows.
4. **Runtime-colour output head:** the existing padded uint8 `QLinearConv` uses inverted runtime
   weights and remains the direct FREE graph-output producer.

No 3x8 sprite-core rewrite is included. Its extra dynamic Slice controls were not priced below
this design, so it would add implementation risk without a demonstrated lower endpoint.

## Exact quantized polarity

The input channel-0 crop contains only exact float values `0` for foreground and `1` for
background. Quantizing it with scale `0.5` and zero point `0` gives:

| spatial state | stored mask | effective value at `x_zero_point=1` |
|---|---:|---:|
| foreground | 0 | -1 |
| in-grid background | 2 | +1 |
| implicit QLinearConv padding | 1 | 0 |

This replaces `Less -> Where` with one `QuantizeLinear`, deleting the 50-byte bool `fg5`
intermediate. The compact bank's padding initializer changes from eight zeroes to eight stored
twos because those cells represent in-grid background, not convolution padding.

The former rank-4 stored-two initializer is repurposed as a rank-4 stored-zero Scatter update, and
one scalar `0.5` initializer is added for `QuantizeLinear`. `QuantizeLinear` may omit its optional
output zero point, but opset12 QLinear operators retain the mandatory scalar zero point 0. The
nonzero final input/weight zero point remains explicitly stored 1. The new half-scale therefore
adds one parameter relative to the initial cost685 estimate.

The runtime output weights invert in the same way:

| output row | stored weight | effective weight at `w_zero_point=1` |
|---|---:|---:|
| background channel 0 | 2 | +1 |
| selected foreground colour | 0 | -1 |
| every other colour | 1 | 0 |

`ScatterElements` therefore starts from `[2,1,1,1,1,1,1,1,1,1]` and writes stored zero at the
selected colour. The products are positive exactly for background-on-background and
foreground-on-selected-colour, negative for the two wrong pairings, zero for unused colours, and
zero for implicit padding. The final uint8 output is consequently identical to the incumbent,
including the off-grid region.

## Geometry and routing folds

### Separate row 0/2 reads

The current graph gathers rows 0 and 2 together, runs one `ArgMax`, then splits the original rows
back apart for the row bank. With the inverted mask, two rank-4 Gathers feed two `ArgMin` nodes:

```text
Gather(centered5, [0]) -> row0_4d -> ArgMin -> left0_i64
Gather(centered5, [2]) -> row2_4d -> ArgMin -> left2_i64
```

Both rows are guaranteed to contain foreground by the generator. For period 1/2 states,
`left2-left0` is exactly shift 0..2. Period-3 states may yield an arbitrary or wrapped difference,
but the independently exact period predicate selects candidate 3 before that value is used.

This deletes the 20-byte joint Gather output, the two 10-byte Split outputs, and the two-element
shift kernel while retaining the same 20 bytes of rank-4 source rows. Net change:
**memory -20, params -2**.

### Scalar uint8 candidate route

Each `ArgMin` result is in 0..9 and is cast to uint8. Their uint8 subtraction produces the legal
shift for period 1/2; underflow is permitted only in the period-3 branch that `Where` discards.
The period predicate and shift are squeezed to scalars, `Where` emits a scalar uint8 candidate,
and one final Cast produces the int32 index required by `Gather`.

Reshaping `source_offsets` from `[4,5,1]` to `[4,5]` does not change its 20 parameters. A scalar
candidate now gathers a rank-1 five-element start vector, so its five Split outputs already have
the `[1]` shape required by `Slice`. The five counted Reshape outputs and their shape initializer
are deleted. Relative to the separate-row graph, this fold is **memory -26, params -1**.

The uint8 underflow branch must be tested under pinned ORT. If that operator path is unsupported,
the exact fallback casts both coordinates to int32 before subtraction. The fallback is predicted
to end at cost697 after all other folds and remains preferable to weakening correctness.

## Six-tap QLinear period hash

The full legal generator enumeration already proves that comparing rows 1 and 4 only needs
columns 0..5: every smaller prefix has counterexamples, while the six-column prefix has zero
disagreements across all 1,428 legal sprite/offset/diagonal states.

Replace each ten-element `QLinearMatMul` fingerprint with a negative-pad `QLinearConv` whose
kernel is:

```text
hash6 = [1, 2, 4, 8, 16, 32]  # shape [1,1,1,6]
```

The row-1 hash consumes the already required `p3_rows_a` with right pad `-4`. The row-4 hash
consumes `centered5` directly with top pad `-4` and right pad `-4`, deleting the ten-byte
`p3_rows_b` Gather and its one-element index initializer. Both hashes have shape `[1,1,1,1]`.

The inverted stored rows contain `0/2`, so each background hash is
`2 * sum(selected powers)` and the maximum is `2*(1+2+4+8+16+32)=126`. No uint8 saturation is
possible. Complementing every encoded bit maps `h` to `126-h`, so hash equality is unchanged from
the already-proved foreground encoding.

Pinned ONNX 1.21 shape inference and ORT 1.26 execution must accept both negative-pad
`QLinearConv` forms. If they do not, retain the existing two `QLinearMatMul` hashes and row-4
Gather. The uint8-routing graph is then predicted at cost701, still an exact improvement.

## Direct final-mask concatenation

The current graph first concatenates five one-row Slice outputs into `bottom_fg [1,1,5,10]`, then
concatenates that tensor with `centered5`. The replacement sends all six inputs directly to the
final mask Concat:

```text
Concat(centered5, bottom_row_0, ..., bottom_row_4, axis=2)
  -> ng_centered_mask [1,1,10,10]
```

This deletes `bottom_fg` and saves 50 bytes without changing values, ordering, or shapes.

## Predicted cost ladder

Candidates are built and verified as a local chain before any deployment mutation. The cheapest
passing exact endpoint is the only candidate sent to final adoption.

| stage | memory | params | cost | estimated gain | purpose |
|---|---:|---:|---:|---:|---|
| deployed baseline | 779 | 70 | 849 | - | immutable control |
| inverse quantize + direct Concat + separate row reads | 659 | 69 | 728 | +0.153758 | conservative exact checkpoint |
| uint8 scalar routing | 633 | 68 | 701 | +0.191551 | remove rank/reshape control bytes |
| six-tap/direct-row4 QLinear hash | 623 | 63 | **686** | **+0.213182** | robust primary endpoint |

Fallbacks are explicit:

- int32 scalar routing plus the QLinear hash: predicted cost697;
- uint8 scalar routing with the existing QLinearMatMul hashes: predicted cost701;
- int32 scalar routing with the existing hashes: predicted cost712;
- conservative checkpoint only: predicted cost728.

The implementation must continue through the full ladder and must not stop merely because an
earlier stage crosses `+0.1`. A fallback may be adopted only if every cheaper attempted endpoint
is rejected for a recorded, reproducible schema/runtime/correctness reason.

## Validation contract

Tests are written before each persistent builder and initially fail on the missing interface. The
final focused suite must enforce all of the following:

1. ONNX checker and strict shape inference pass under the pinned environment.
2. Structural inspection proves `fg5`, `bottom_fg`, the joint row0/2 Gather, row Split, start
   Reshapes, ten-tap hash, and row-4 Gather are absent in the primary endpoint.
3. `QuantizeLinear` maps both possible in-crop channel-0 values, float `0/1`, exactly to stored
   uint8 `0/2`.
4. Runtime colours 1..9 give effective weights background `+1`, selected `-1`, others `0`.
5. The padded uint8 `QLinearConv` remains the direct graph-output producer.
6. All 1,428 legal generator states preserve full-row equality, six-column hash equality, and the
   period/shift candidate; every hash is at most 126. The handcrafted bundled period-1 case is
   tested separately because the randomized generator domain uses periods 2 and 3.
7. Every bundled example is raw byte-identical to the cost849 incumbent, passes the target, and
   has zero nonzero output outside the valid 10x10 area.
8. Fresh 2,000 A/B has incumbent fail=0, candidate fail=0, raw divergence=0, sign divergence=0,
   and off-grid positives=0.
9. The isolated scorer confirms the measured memory/params/cost for every retained checkpoint.

Only after local verification selects the cheapest exact candidate should the workflow run:

```text
ng gate -> ng adopt -> source rebuild -> SHA equality -> focused re-verification
```

No gate rejection may be bypassed. If no candidate passes, deployment and source remain unchanged
and the dated negative result is recorded only in the task124 lever ledger with the required
ran/tool-date/verdict/reopen/falsification evidence.

## Adoption and synchronization

On a successful final gate, `ng adopt` is the only operation allowed to replace the deployed
artifact. Then update `src/custom/task124.py` to construct the adopted graph semantically and
require byte-identical SHA equality among source build, candidate, and deployment.

Update `candidates/task124/DISCOVERY.md`, `state/tasks/task124.md`, the task124 lever entry, and any
reusable quantized-polarity/hash insight with measured rather than predicted values. Because the
recursive skill's legacy inventory/queue scripts are absent from this repository, perform only a
read-only 400-graph signature rescan using the currently available project mechanisms; do not
modify another task. Do not pack or submit.
