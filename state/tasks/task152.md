---
deployed_cost: 90
logged_costs_match: stale-likely
migrated: 2026-07-09
---


## S9 (2026-07-03) — mechanism-14 separable-remap einsum (+0.934) ADOPTED
Single 5-operand Einsum 'ra,ai,zcij,bj,sb->zcrs', mem=0: mirror-tile 3x3->6x6 symmetric U/S shared, 458->180.
Gates: stored fail=0; uncached fresh 2000+600: 0/0/0 (bit-identical). No TopK.
NOTE: scan projection was ~8x optimistic — output axis must span the FULL 30 (grading
tensor [1,10,30,30]), so U tables are [30,K] not [out,K]. Backup task152_pre_s9.onnx.

## ADOPTED 20260712T140208Z
- cost: 90 -> 60 (points 20.9057)
- source: /Users/minseong/project/neurogolf/dumps/archive_extract/submission7300+/task152.onnx
- note: archive.zip submission7300+ net; fresh 2000/0 fail; mechanism-graft
