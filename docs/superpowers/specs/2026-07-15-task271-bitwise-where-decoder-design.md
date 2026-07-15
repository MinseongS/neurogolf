# Task271 Bitwise/Where Decoder Design

Date: 2026-07-15

## Objective and scope

Optimize only task271 from the adopted cost366 graph while preserving its encoded-score
selection proof. The target for this step is the cheapest exact lowering already demonstrated
under the pinned runtime: cost350, bundled267/267, all 49 legal winner positions, all 512
payload codes, and no off-grid positive output. Adoption must use `ng adopt`.

The pinned environment remains onnx 1.21.0 and onnxruntime 1.26.0. This step does not
reintroduce dual ReduceMax/ArgMax, `cyan_patch/crop8/Pad`, a full-canvas Cast, or a second
score carrier. The next `+0.1` threshold remains cost at most 331; cost350 is a safe
below-threshold improvement authorized by the user, not a claim that task271 is exhausted.

## Incumbent and unchanged proof

The cost366 incumbent has memory247 and params119. Its graph is:

`Conv -> MaxPool -> Cast(UINT16) -> Mod -> GreaterOrEqual -> Cast(UINT8) -> BitShift -> ConvInteger`.

The direct score Conv remains byte-for-byte unchanged. At row-major patch position `k`, its
blue-channel coefficient is `512 + 2^k`; every background-channel coefficient is `-4096`.
An exact box with blue count `b` and payload `p` therefore scores `512*b + p`.

The generator gives the four exact boxes distinct blue counts. A one-count difference
contributes 512, greater than the maximum payload difference 511, so the exact-box winner is
unique. Every invalid 3x3 window includes background and remains below every possible winning
exact box. The proposed rewrite changes only low-nine-bit decoding and rendering preparation;
it cannot change the 7x7 score values or MaxPool selection.

## Chosen decoder

Replace the four-node decoder segment

`Mod -> GreaterOrEqual -> Cast(UINT8) -> BitShift`

with

`BitwiseAnd -> Cast(BOOL) -> Where(UINT8)`.

The complete graph becomes:

`Conv -> MaxPool -> Cast(UINT16) -> BitwiseAnd -> Cast(BOOL) -> Where(UINT8) -> ConvInteger`.

The decoder uses these exact tensors:

- `payload_masks[1,1,3,3] UINT16 = [1,2,4,...,256]`;
- `two_u8[1] = 2`, the true branch of Where;
- `zero_u8[1] = 0`, the false branch of Where;
- `one_u8[1] = 1`, the ConvInteger input zero-point;
- the existing signed renderer weight with channel1 `+1`, channel8 `-1`, and all other
  channels zero.

`BitwiseAnd(encoded_u16, payload_masks)` produces either zero or the bit mask at each patch
position. Casting that tensor to BOOL performs the nonzero test without the nine threshold
parameters. `Where` directly emits centered UINT8 values 0/2, deleting both the counted
UINT8 bit plane and the separate BitShift result. ConvInteger subtracts zero-point 1, so the
valid patch becomes -1/+1. Its padded region remains zero-effective and produces no positive
off-grid class.

BitwiseAnd requires default opset18. The model keeps the standard ONNX domain only.

## Pinned-runtime evidence obtained before implementation

An in-memory model, without creating a candidate artifact, established all prerequisites
under onnx 1.21.0 and ORT 1.26.0 with graph optimizations disabled:

- UINT16 BitwiseAnd loads and runs at opset18.
- UINT8 Where loads and runs; INT8 Where is `NOT_IMPLEMENTED` and is rejected.
- The complete proposed graph passes full checker, strict shape inference, and ORT load.
- The decoder reproduces all 512 possible nine-bit payloads and has zero off-grid positives.
- A synthetic winning patch reproduces the expected output at all 49 legal top-left positions.
- The unchanged scorer has a unique maximum on all 267 bundled examples; the observed minimum
  top-two margin is 202.
- Official in-memory evaluation reports pass267, fail0, memory238, params112, cost350, and
  points19.1420668455.

This evidence satisfies the required pre-implementation MaxPool-position and unique-winner
proof. The persistent implementation must reproduce it rather than weakening it.

## Static price

Counted memory:

- FLOAT32 scores `[1,1,7,7]`: 196;
- FLOAT32 max score scalar: 4;
- UINT16 encoded scalar: 2;
- UINT16 masked payload `[1,1,3,3]`: 18;
- BOOL payload bits `[1,1,3,3]`: 9;
- UINT8 centered payload `[1,1,3,3]`: 9.

Total memory is 238.

Initializer elements:

- score kernel: 90;
- payload masks: 9;
- Where true, Where false, and ConvInteger zero-point scalars: 3;
- signed renderer weight: 10.

Total params are 112, so total cost is exactly 350. This saves 16 from cost366 and raises
task271 points by `ln(366/350) = 0.0447001789`.

## Rejected alternatives and pivot rules

- The initially preferred INT8 Where path would cost349, but pinned ORT has no implementation
  for `Where(16)` on INT8. It must not be built or retried.
- BitwiseAnd followed by the incumbent Cast/BitShift centering is legal but costs357, so it is
  dominated by the UINT8 Where route.
- Arithmetic INT8 centering is supported at opset18, but its extra counted patch plane removes
  the memory advantage and cannot beat cost350.
- Score-grid cropping, FP16 score conversion, and the previously removed coordinate/crop/full-
  canvas families remain out of scope.

If the persistent candidate costs more than 350, fails any bundled example, changes one of the
49 synthetic outputs, diverges on fresh data, or fails fixed-point reconstruction, do not adopt.

## Test-first implementation

Add a new focused test before adding the builder. The RED test must require:

- the pinned cost366 incumbent SHA;
- exact opset18 and node sequence;
- exact masks, branch scalars, zero-point, score kernel, and renderer weight;
- absence of `Mod`, `GreaterOrEqual`, payload UINT8 Cast, and BitShift;
- strict inference/full checker and pinned ORT loading;
- exhaustive 512-code decoder equivalence and 49-position full-graph equivalence;
- static cost350, memory238, params112, bundled267/267 fail0;
- persistent artifact equality and second-pass fixed point.

Run the focused test once and require RED because the builder is absent. Implement only the
minimal fixed-point builder, generate `candidates/task271/bitwise_where_decoder.onnx`, and make
the test GREEN.

## Adoption and completion verification

Before adoption, run all task271 focused tests, official `ng gate`, fresh comparison with at
least 2000 generated examples, and exact-source reconstruction. Require bundled fail0,
cost350, 49/49 position equivalence, 512/512 decoder equivalence, fresh candidate fail0 and
divergence0, and byte-identical candidate/source reconstruction.

Adopt only through `uv run ng adopt`. After adoption, rebuild `networks/task271.onnx` from
`src/custom/task271.py` and require candidate, deployed, network, and source-rebuilt SHA equality.
Update the task271 ledger, DISCOVERY handoff, and the existing payload-carry insight without
overwriting unrelated shared-worktree edits. The next `+0.1` threshold after cost350 is cost at
most 316.
