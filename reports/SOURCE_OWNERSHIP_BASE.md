# Source ownership base

Status date: 2026-06-28

This report tracks whether each live submitted network has a local semantic
builder under `src/custom/taskNNN.py`.  Possessing `networks/taskNNN.onnx` is not
counted as source ownership.

## Current status

- Networks present: 400/400
- Semantic/custom source files present: 379/400
- Missing semantic source: 21/400
- Special missing case: task118 is live custom-derived but has no source file
- External live with no local semantic source: 26

## Missing-source tasks

Priority order is roughly: low points / high memory first, then live-custom
source recovery, then small public nets.

```text
002 005 018 044 046 054 066 076 101 118
133 157 173 209 216 219 255
285 286 319 366
```

## Immediate base-first policy

1. Do not treat an ONNX-only task as controlled.
2. For each missing task, add a semantic builder file that can reproduce the
   current live behavior or a verified equivalent behavior.
3. Only after the builder passes official+arc-gen local evaluation should it
   count as source-owned.
4. If the builder is equivalent but not better than live, keep it for ownership
   but do not adopt a new network.
5. If the builder improves stored score and passes fresh verification, adopt and
   submit.

## Next source-ownership targets

- task118: recover/replace the live `custom:task118_tail_where+onnxsim` source.
- task054: high-memory no-source closed-form star/flag reconstruction candidate.
- task319: high-memory no-source correspondence candidate; likely wall but source
  ownership still missing.
- task002/task018/task286/task209/task366: low-score/high-memory missing-source
  tasks, many previously classified as walls; require semantic builders or
  explicit wall-source stubs only if exact reproduction is infeasible.
