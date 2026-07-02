# User Review Workflow

The working mode has shifted from autonomous queue grinding to human-in-the-loop
task review.

## Goal

Let the user inspect individual tasks visually and mathematically, then turn good
semantic ideas into source-owned ONNX candidates.

## Per-Task Loop

1. Pick one task.
2. Show the visible data:
   - `train`, `test`, and `arc-gen` counts;
   - representative input/output grids;
   - shapes and colours;
   - current score, memory, params, method.
3. State the current semantic hypothesis in plain language.
4. Mark confidence:
   - `verified`: Python oracle or ONNX candidate passes stored examples;
   - `uncertain`: inferred from logs or partial examples;
   - `contradicted`: visible stored data disagrees with prior tasklog/generator notes.
5. Let the user challenge the rule.
6. Test the user's rule as a Python oracle before touching source.
7. If the oracle passes, build the cheapest plausible ONNX candidate.
8. Compare against incumbent with `src.harness.evaluate`.
9. Adopt only if it improves and passes the normal gates.
10. Record the result in `reports/tasklog/taskNNN.md`.

## Important Lessons From Recent Work

- `task002`: visible stored data supports the simple rectangle rule:
  enclosed black rectangles with green adjacent edges become yellow. Prior fresh-generator
  noise notes could not be reproduced because `/tmp/arc-gen` is absent in the current
  workspace. The semantic correction was important, but the incumbent bitset flood was
  already compact; only a tiny params improvement was adopted.
- `mem0` tasks: memory is already zero, so improvement requires reducing params. This can
  be valuable: halving params gives roughly +0.69 points, and reducing a 900-param Conv to
  about 100 params can give about +2.2 points. However single-Conv mem0 tasks often pay
  dense output-channel weights; sparse nonzero structure does not reduce params unless the
  dense initializer shape itself shrinks.

## Research Protocol

Use `reports/TASK_RESEARCH_PROTOCOL.md` as the detailed per-task operating
contract.  The tasklog template has been expanded so every task records:

- human-readable rule;
- current graph cost anatomy;
- mechanism hypotheses and kill criteria;
- attempts, verification, floor/wall reason, and transferable insight.

## What To Avoid

- Do not keep defending old tasklog conclusions when visible stored data contradicts them.
- Do not build ONNX before a simple Python oracle validates the user's semantic rule.
- Do not spend long cycles on a wall task after the likely improvement ceiling is tiny.
- Do not leave source/live parity broken.
