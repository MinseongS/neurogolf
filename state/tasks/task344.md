---
deployed_cost: 910
logged_costs_match: match
migrated: 2026-07-09
---

# task344 — green/red adjacency local rewrite

Semantic source already exists in `src/custom/task344.py`.

Rule: a green(3) cell with at least one red(2) 4-neighbor becomes cyan(8); a red(2)
cell with at least one green(3) 4-neighbor becomes black(0); gray(5) and other cells
are unchanged. The deployed/source graph is a single 3x3 plus-neighborhood Conv
written directly into the free output: mem 0, params 910.

## 2026-06-30 mem0 params assessment

The dense Conv has only 22 nonzero weights, but the scorer counts the full dense
initializer shape `[10,10,3,3]` plus bias. The plus-shaped rule genuinely needs the
3x3 spatial support, so kernel bbox trimming cannot reduce params. Decomposing into
slices/counts would introduce full-canvas intermediate planes and loses the mem0
advantage.

No adoption candidate.

## ADOPTED 20260712T140232Z
- cost: 910 -> 692 (points 18.4604)
- source: /Users/minseong/project/neurogolf/dumps/archive_extract/submission7300+/task344.onnx
- note: archive.zip submission7300+ net; fresh 2000/0 fail; mechanism-graft
