# Source ownership base

Status date: 2026-06-28

This report tracks whether each live submitted network has a local source
builder under `src/custom/taskNNN.py`.  Possessing `networks/taskNNN.onnx` is not
counted as source ownership.

## Current status

- Networks present: 400/400
- Source builder files present: 400/400
- Source-controlled `build()` available: 400/400
- Missing source builder: 0/400
- Special missing case: none
- External live with no local source builder: 0

## Missing-source tasks

None.

## Immediate base-first policy

1. Do not treat an ONNX-only task as controlled.
2. For each missing task, add a source builder file that can reproduce the
   current live behavior or a verified equivalent behavior.
3. Only after the builder passes official+arc-gen local evaluation should it
   count as source-owned.
4. If the builder is equivalent but not better than live, keep it for ownership
   but do not adopt a new network.
5. If the builder improves stored score and passes fresh verification, adopt and
   submit.

## Next source-ownership targets

Base is complete.  Future work should shift back to score improvements, with
semantic rewrites preferred over exact-source preservation for the remaining
wall-class tasks.
