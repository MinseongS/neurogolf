---
deployed_cost: 904
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task197 — 82819916

## 2026-06-29 colour-kernel screen

Current source score: 17.135964 @ mem 2586 params 16.

Rule: infer a binary pattern from the first coloured row, then complete later rows
using each row's two observed colours.

The graph recovers two per-row colour sequences (`row_color0/1`) and uses
`ConvInteger` with runtime colour kernels to write directly to output.  Dominant
costs are the two [1,10,14,1] f32 slices (560 B each) and their u8/transposed
kernel forms.

No rewrite adopted.  A float Conv variant would make the runtime colour kernel
4x larger, and the 14-row maximum is required by the generator height bound.

## ADOPTED 20260709T080333Z
- cost: 904 -> 852 (points 18.2524)
- source: candidates/task197/task197_u8_where_tail.onnx
- note: u8 Where tail replaces fp32 Mul/Where source-index tail cost 904->852

## ADOPTED 20260709T094144Z
- cost: 852 -> 753 (points 18.3759)
- source: candidates/task197/task197_self_einsum.onnx
- note: self-einsum template row: compute match/active directly from free input, deleting counted T/g Slice outputs; trades 360 selector params for -440B memory

## ADOPTED 20260713T141429Z
- cost: 753 -> 130 (points 20.1325)
- source: candidates/task197/identity_vote.onnx
- note: onehot self-Einsum identity vote; bundled 267/0, fresh 50000/0
