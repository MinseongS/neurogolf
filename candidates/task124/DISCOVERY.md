# task124 discovery — centered runtime-colour QLinear renderer

## Final adopted result

- Start: **cost 1953 = memory 1879 + params 74**, score 17.4228780691.
- Final: **cost 849 = memory 779 + params 70**, score 18.2559408137.
- Gain: **+0.8330627446**, saving 1104 cost (56.53%). The final 939->849 fold alone
  adds **+0.1007562929**.
- Final gate: **267/267**, fail 0. Fresh diagnostic: **2000/2000**, incumbent fail 0,
  candidate fail 0, candidate/incumbent divergence 0. Off-grid positives: 0.
- Candidate/deployed/source SHA-256:
  `4e4bafbb3d65046a1ec08a211de6c9951705b613777a2a3ece9f4c73f6041b25`.
- Final adoption: `ng adopt`, timestamp `20260715T144530Z`. No pack or submit was run.

## Final graph

The original terminal route materialized two scalar colour carriers before the FREE output:

```text
out_fg bool [1,1,10,10]
  -> Where(runtime colour, background) = color_crop u8, 100B
  -> Pad = color_full u8 [1,1,30,30], 900B
  -> Equal(channel ids) = FREE bool output
```

The final graph instead centers the top foreground mask once:

```text
Slice(input channel 0, 5x10) -> Less(..., shared scale=1.0) -> fg5 bool
  -> Where(fg5, 2, 0) = centered5 uint8 [1,1,5,10]
       ├─ period/shift detection through two scalar QLinear row hashes
       ├─ compact rank-4 source-row bank
       └─ top half of the final centered mask

runtime source offsets [5]
  -> five fixed-length rank-preserving Slice operations on the compact row bank
  -> bottom_fg uint8 [1,1,5,10]

Concat(centered5, bottom_fg) = ng_centered_mask uint8 [1,1,10,10]
  -> padded QLinearConv -> FREE uint8 graph output [1,10,30,30]
```

The five dynamic row Slices replace `Add(source_offset,col_idx)`, whose INT32
`fg_source_map [1,1,5,10]` alone cost 200B. The final fold keeps the row bank and every Slice
output rank 4, deleting three 10B row Reshapes, the 50B `bottom_flat`, and their one-element
shape initializer. Explicit Slice result value information fixes each runtime row at
`[1,1,1,10]` for strict shape inference and official memory accounting.

The former p3 test materialized ten-cell `Equal` and `Cast` outputs before scalar reduction.
Two uint8 `QLinearMatMul` operations now hash rows 1 and 4 with
`[1,2,4,8,16,32,0,0,0,0]^T`, followed by one scalar-element `Equal`. Centered entries are 0/2,
so the maximum hash is 126 and cannot saturate uint8. Generator-domain enumeration covered all
1,428 legal sprite/offset/diagonal combinations and found zero disagreements with full ten-cell
row equality.

## Exact quantized renderer

`ng_centered_mask` stores background/foreground as uint8 `0/2`. With shared
`x_zero_point=w_zero_point=1`, the effective input states are:

| state | stored input | effective input |
|---|---:|---:|
| in-grid background | 0 | -1 |
| foreground | 2 | +1 |
| implicit Conv padding | 1 (quantization padding state) | 0 |

`ReduceMax(input)->ArgMax` finds the runtime foreground colour. `ScatterElements` starts from
stored weights `[0,1,1,1,1,1,1,1,1,1]` and writes stored value 2 at the selected row. With
weight zero-point 1 this is exactly:

| output row | effective weight |
|---|---:|
| background | -1 |
| selected colour | +1 |
| every other colour | 0 |

Therefore the background channel is positive only for in-grid background, the selected channel
is positive only for foreground, all other channels are zero, and implicit padded positions stay
zero. The padded uint8 QLinearConv writes the graph output directly; there is no valid plane,
runtime colour crop, full scalar Pad, or terminal Equal.

## Adoption chain

| stage | memory | params | cost | points | key fold |
|---|---:|---:|---:|---:|---|
| initial | 1879 | 74 | 1953 | 17.422878 | `Where->Pad->Equal` label tail |
| dynamic two-feature | 1219 | 190 | 1409 | 17.749364 | `[mask,valid]` padded QLinearConv |
| centered one-feature | 999 | 79 | 1078 | 18.017137 | 0/2 mask, shared zero-point, no valid plane |
| scattered colour weight | 988 | 69 | 1057 | 18.036810 | replace `Cast->Equal->Where` with ScatterElements |
| centered Slice geometry | 878 | 67 | 945 | 18.148815 | reuse centered5; delete 200B index plane |
| shared initializers | 878 | 61 | 939 | 18.155185 | reuse scale, Slice step, and crop ends |
| rank-4 bank + row hash | 779 | 70 | 849 | 18.255941 | delete flat carriers; scalar QLinear fingerprints |

Every stage passed the official isolated gate and was adopted only through `ng adopt`.

## Verification and artifacts

- Tests were written before each persistent builder; each first failed on its missing builder.
- Exact colour/weight tests enumerate runtime colours 1..9 and assert `w_zero_point=1` gives
  background -1, selected +1, and all other rows 0.
- Structural tests require QLinearConv to remain the direct uint8 graph-output producer and
  require the old valid plane, per-cell index map, runtime colour selector plane, Pad, and Equal
  tail to be absent.
- Full checker, strict shape inference, pinned ONNX 1.21/ORT 1.26, bundled raw A/B, isolated
  `ng gate`, fresh 2000 A/B, and off-grid-positive checks passed.
- Final builder: `candidates/task124/build_rank4_qlinear_hash.py`.
- Final regression: `candidates/task124/test_rank4_qlinear_hash.py`.
- The final regression proves the bounded hash over all 1,428 legal generator states and the
  complete task124 focused suite remains 14/14 after adoption.
- Geometry regression: `candidates/task124/test_centered_slice_geometry.py`.
- Source owner: `src/custom/task124.py`; it rebuilds the final artifact byte-identically.

## Transfer and residual boundary

The renderer applies when a fixed compact plane has two semantic states, one runtime-selected
foreground colour, and implicit quantized padding can be assigned the zero effective state. The
dynamic-Slice geometry fold applies when a small number of fixed-length rows are selected from a
compact bank using runtime starts; preserve rank through Concat/Slice whenever flattening carries
no semantics. A bounded QLinear fingerprint can replace cellwise equality plus all-reduction when
reachable rows admit a collision-free integer code and the accumulated value cannot saturate.

Reject multi-colour runtime content, dynamic output row lengths, saturating/colliding row codes,
or cases where quantized padding cannot remain distinct from both semantic states. At final
cost849, another +0.1 requires **cost <=768** (saving at least 81). The remaining dominant tensor
is the load-bearing FREE-input channel-0 float Slice (200B). A 400-graph post-insight scan loaded
all models with errors0: task124 was the only scalar-QLinear-hash exact hit; task270 was a broad
flattened-Slice hit, and task174/233/343/363 were broad bounded Equal/Cast/ReduceMin hits requiring
independent semantic/cost proofs. No other task was modified. `ConvInteger` remains outside this
path because direct uint8 QLinearConv output is an explicit invariant.
