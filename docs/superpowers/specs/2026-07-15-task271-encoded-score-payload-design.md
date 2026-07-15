# Task271 encoded-score payload design

Date: 2026-07-15

## Objective and scope

Optimize only task271 from the adopted cost411 graph to the next `+0.1` threshold,
cost at most 371. Preserve bundled fail0, the unique-winner rule, correct selection and
payload decoding at all 49 candidate positions, fresh divergence0, deterministic two-pass generation, and exact
source/candidate/deployed SHA equality. Adoption must use `ng adopt`.

The pinned environment remains onnx 1.21.0 and onnxruntime 1.26.0. Previously removed
dual ReduceMax/ArgMax, `cyan_patch/crop8/Pad`, and full-canvas Cast routes are out of
scope.

## Current cost

The cost411 incumbent has memory302 and params109. Its dominant counted tensor is
`scores[1,1,7,7]` FLOAT (196B). After MaxPool, the coordinate and dynamic-patch route
costs another 102B of memory beyond `max_score`: MaxPool indices, INT32 coordinate
arithmetic, Slice bounds, FLOAT patch, UINT8 patch, and centered UINT8 patch.

## Chosen representation

Encode the winning patch in the same scalar used to rank candidate windows. For spatial
bit position `k` in row-major order, use the direct score Conv weights:

- background channel 0: `-4096` at every 3x3 kernel position;
- blue channel 1: `512 + 2^k` at position `k`;
- all other channels: zero.

For an exact box with blue count `b` and row-major binary patch code `p`, the score is
`512*b + p`. Since `0 <= p <= 511`, a one-count increase always dominates every
payload difference. The generator draws four distinct counts from 0 through 8, so the
maximum-count box remains the unique score winner.

Every non-box 3x3 window contains at least one background cell. Even the loose upper
bound of eight blue cells, maximal payload, and one background cell is
`8*512 + 511 - 4096 = 511`. Four distinct nonnegative counts imply that the true winner
has count at least three, so its score is at least `3*512 + 7 = 1543`. Invalid windows
therefore cannot win. All scores are exact integers well inside FLOAT32's exact range.

## Decoder graph

MaxPool emits only `max_score`; its indices output and all coordinate nodes are deleted.
The low nine bits of `max_score` are decoded as follows:

1. Cast the positive exact-integer scalar to UINT16.
2. Broadcast `Mod` against `moduli = [2,4,8,...,512]`, shaped `[1,1,3,3]`.
3. Compare the remainders with `thresholds = [1,2,4,...,256]` using
   `GreaterOrEqual`, producing the nine payload bits.
4. Cast BOOL to UINT8 and left-shift by the shared `one_u8` to obtain 0/2.
5. Feed the existing signed one-feature ConvInteger renderer. With input zero-point 1,
   0/2 becomes -1/+1; channel1 weight +1 and channel8 weight -1 reproduce the blue/cyan
   patch in the FREE 10x30x30 output.

The intended node sequence is:

`Conv -> MaxPool -> Cast -> Mod -> GreaterOrEqual -> Cast -> BitShift -> ConvInteger`.

UINT16 BitShift is not used: pinned ORT 1.26 has no kernel for it. A capability probe
confirmed UINT16 Mod and GreaterOrEqual plus UINT8 BitShift under opset12.

## Static price

Counted memory:

- encoded scores FLOAT32 `[1,1,7,7]`: 196
- max score FLOAT32 scalar: 4
- encoded score UINT16 scalar: 2
- UINT16 remainders `[1,1,3,3]`: 18
- BOOL payload bits `[1,1,3,3]`: 9
- UINT8 payload bits `[1,1,3,3]`: 9
- centered UINT8 payload `[1,1,3,3]`: 9

Total memory is 247.

Initializer elements:

- score kernel: 90
- moduli: 9
- thresholds: 9
- shared UINT8 one: 1
- signed renderer weight: 10

Total params are 119, so expected cost is exactly 366. This saves 45 from cost411 and
beats the cost371 threshold by five.

## Rejected alternatives

- UINT16 BitShift decoding passes ONNX schema checks but ORT 1.26 reports
  `NOT_IMPLEMENTED`.
- UINT16 integer Div is not admitted by ONNX shape inference. UINT32 BitShift works but
  widens the decoder and prices above the threshold.
- Cropping the 7x7 score grid loses legal fresh winner positions. Bundled winners cover
  34 positions and a 3000-case fresh sample covered 39, including rare central-edge
  positions.
- FP16 scoring requires a counted conversion of the FREE FLOAT32 input and is cost-worse.

## Test-first implementation

Before the builder exists, add one focused regression that pins the cost411 baseline
SHA and expects the encoded-payload builder and persistent artifact. The test must assert:

- the exact node sequence and opset12;
- exact encoded kernel, moduli, thresholds, dtypes, shapes, and renderer inputs;
- removal of every coordinate/Slice initializer and tensor;
- static cost366 with memory247 and params119;
- bundled267/267 fail0;
- byte-identical persistent generation and second-pass fixed point.

Run this test once and require an expected RED caused by the missing builder. Then add
the minimal builder and require GREEN.

## Adoption and verification

After the focused test is green:

1. Run all task271 focused tests.
2. Run official `ng gate`; require cost366, bundled267/267, fail0.
3. Re-run the 49-position MaxPool unique-winner probe.
   Also exercise the encoded score and payload decoder with a synthetic winning box at
   every one of the 49 positions.
4. Run fresh comparison with at least 2000 generated cases; require candidate fail0 and
   divergence0.
5. Generate exact source and rebuild `networks/task271.onnx`; require candidate,
   deployed-after-adopt, and source-rebuilt SHA equality.
6. Adopt only through `ng adopt`, update task271 DISCOVERY/insight/STATE, and repeat the
   focused score/fresh/SHA checks from the adopted artifact.

If the actual static price exceeds 371, any bundled case fails, or fresh divergence is
nonzero, do not adopt and pivot rather than weakening the gates.
