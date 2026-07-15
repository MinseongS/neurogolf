# Public-Zero Repair Design for Tasks 118 and 131

## Problem

The deployed task118 and task131 models pass every bundled example and are cheaper than their
predecessors, but isolated Kaggle submissions score 0.00:

- task118: submission 54732112
- task131: submission 54732114

Both failures were introduced by bundle-specific support reduction. Task118 replaced semantic
detection with a fixed input hash/edit lookup. Task131 reduced the legal 155-position probe support
to the 119 positions active in bundled data. These are local-score improvements but public-score
regressions.

## Goal

Restore nonzero public scoring for both tasks through an auditable gate path while preserving the
normal rule that ordinary adoptions must be strictly cheaper than the deployed model.

## Selected approach

Add an explicit public-zero repair mode to `ng gate` and `ng adopt`. The mode is separate from
ordinary score optimization and requires:

1. a user-supplied Kaggle submission reference proving the incumbent scored 0.00;
2. candidate bundled fail=0 in an isolated process;
3. no unsupported integer TopK;
4. a repository record mapping the task, zero-score submission, repair candidate, and reason;
5. an adoption stamp marked `REPAIRED`, not `ADOPTED`.

Public-zero repair may accept a more expensive candidate because the comparison is between an
effective public score of zero and a known-safe nonzero model. Without this explicit mode and
evidence, the existing strictly-cheaper check remains unchanged.

## Repair candidates

### task118

Restore `submission/.backups/task118_20260715T110407Z.onnx`:

- bundled: 267/267
- cost: 8699
- points if valid: 15.9290366
- provenance: the cost-8699 archive graft was included in Kaggle-confirmed record 54610908

This discards the later fixed-hash optimization chain. Individual topology ideas from that chain
may be reconsidered only if they can be applied to the semantic/Kaggle-positive model without
introducing input memorization.

### task131

Restore `submission/.backups/task131_20260715T115906Z.onnx`:

- bundled: 266/266
- cost: 3874
- points if valid: 16.7379572
- fresh diagnostic: 1500/1500 versus 76/1500 failures for the deployed pruned-support model

This discards only the 119-position support prune. Future compression may retain the rest of the
topology, but must preserve all 155 legal probe positions or replace them with a rule-complete
fixed-shape representation.

## Score effect

The local manifest decreases by about 0.951 points:

- task118: 16.8452124 -> 15.9290366 (-0.9161758)
- task131: 16.7728918 -> 16.7379572 (-0.0349346)

Those local gains are not real on Kaggle because both current artifacts score zero. The expected
public recovery is approximately 32.667 points if the restored artifacts reproduce their prior
safety evidence.

## Verification

Implementation follows test-driven development:

1. add failing tests proving ordinary gate still rejects a more-expensive candidate;
2. add failing tests proving public-zero repair requires matching evidence and rejects malformed,
   mismatched, or nonzero evidence;
3. implement the minimal gate/CLI/adoption path;
4. run focused and full gate/adoption tests;
5. gate and adopt task118 and task131 only through the new repair mode;
6. independently score both deployed artifacts, verify source/provenance records, and pack only
   after the workspace reaches a submission-qualified 400/400 state.

No submission is part of this repair unless the full-board verification and concurrent-submission
checks pass.
