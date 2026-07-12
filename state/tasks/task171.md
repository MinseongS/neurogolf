---
deployed_cost: 759
logged_costs_match: match
migrated: 2026-07-09
---

# task171 — mem0 border-from-black-rectangle

Rule from stored examples: the input grid is all black(0); output is cyan(8) on the
border of the in-grid rectangle and black inside. The deployed graph is a single
3x3 Conv from channel 0 to output channels 0 and 8, written directly into the free
10-channel output: mem 0, params 910.

## 2026-06-30 mem0 params assessment

Although the semantic rule is simple, reducing params while preserving mem0 is
blocked by the fixed output arity of Conv: a direct Conv to the official output has
weight shape `[10,10,3,3]` even though only channel 0 is read and only channels 0/8
are meaningful. A smaller Conv after slicing channel 0 would immediately materialize
a counted full-canvas slice (3600B), worse than the current `memory + params = 910`.

No adoption candidate.


## S15b (2026-07-06) — RE-ADOPTED from prvsiyan 7235.05 min-merge notebook (further golf): 910 -> 759 (+0.181)
Gate fresh_verify 1500: inc=0/0 (cand<=inc, safe rule). prvsiyan bundle = min-merge of public sources, had a cheaper variant than my prior net. Source-owned via live_to_exact_source, re-measured fail=0. See [[neurogolf-urad-7225-bundle-vein]].
## ADOPTED 20260709T123205Z
- cost: 759 -> 435 (points 18.9247)
- source: candidates/public_dumps/20260709_pm/biohack44_neurogolf-2026-championship-best-solution/_src_A/task171.onnx
- note: min-merge from biohack44_neurogolf-2026-championship-best-solution

## 2026-07-09 public-insight deep-lane autopsy (op-delta vs .backups)
Op-delta deployed(435) vs backup task171_20260709T123205Z(759): IDENTICAL weights (w1/b1/w2/b2),
ONE node deleted — the `Relu` between Conv1 (`input`->t1[1,1,9,9], 1x1, neg-pad crop) and Conv2
(t2->output, 3x3, pad expand). Deleting it removes the t2 fp32 plane (324B), the whole delta.
Why safe: b1=-0.039 so t1<0 for the all-black interior; the Relu clamp DOES change intermediate
values, but the grid is only 3-9 wide and the argmax decode over the VALID region is invariant —
the numeric diff lives entirely in the ignored 30x30 padding. This is dead-node pruning, NOT a
cheaper representation. Cost 759(mem648+par111) -> 435(mem324+par111).

⭐ TRANSFERABLE: registered insight `borrowed_net_redundant_branch_prune`. Does NOT generalize to
our own nets — naive Relu-drop gated on the two direct Conv-Relu-Conv twins FAILED (task266 fail=14,
task042 fail=196); systematic ablation of all 15 deployed Relu/Clip nets = 0 dead nodes. The win is a
per-net decode-invariance accident specific to the borrowed weights, reproducible only by ablating
freshly-borrowed public nets. Reopen: ablate each NEW min-merge net's pointwise nodes before trusting tight.

## ADOPTED 20260712T140251Z
- cost: 435 -> 414 (points 18.9741)
- source: /Users/minseong/project/neurogolf/dumps/archive_extract/submission7300+/task171.onnx
- note: archive.zip submission7300+ net; fresh 2000/0 fail; mechanism-graft
