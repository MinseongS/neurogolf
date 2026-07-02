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
