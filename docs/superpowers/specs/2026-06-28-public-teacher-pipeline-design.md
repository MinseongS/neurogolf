# Public teacher pipeline design

## Goal

Use public high-score models without surrendering design control.  A public ONNX
is treated as a teacher artifact: compare it, extract its structural insight,
rebuild or scaffold source ownership, then propagate the insight through the
recursive queue.

## Ownership policy

- Do not blindly overwrite `networks/taskNNN.onnx` with a public model.
- A public model may influence adoption only after a source-controlled builder
  exists or an explicit submission-test exception is recorded.
- A public model can be valuable even when its stored score is lower than ours if
  it uses a better mechanism: lower memory, fewer params, a better dtype/opset
  floor, or a reusable op pattern.
- Every imported mechanism must become either:
  - an exact source scaffold draft,
  - a semantic source implementation,
  - or an entry/draft for `reports/insight_registry.yaml`.

## Components

- `public_candidates/`
  - Local drop zone for public ONNX files.
  - Expected filenames include a task id, e.g. `task017.onnx`, `017.onnx`, or
    `some_source_task017.onnx`.
- `reports/scripts/public_teacher_scan.py`
  - Scans public candidate ONNX files.
  - Evaluates each candidate against stored examples.
  - Compares public vs current live vs source builder.
  - Flags candidates as score-up, memory-down, params-down, source-lag, or
    mechanism-teacher.
  - Writes `reports/public_teacher_report.json` and
    `reports/public_teacher_report.md`.
- `reports/scripts/public_teacher_extract.py`
  - Creates reviewable artifacts for one public teacher.
  - Generates an exact-preserve source scaffold draft under
    `reports/public_teacher_sources/`.
  - Generates a mechanism extraction note under
    `reports/public_teacher_insights/`.
  - Does not overwrite live source or live network unless a future explicit
    install flag is added and reviewed.

## Data flow

1. Drop public ONNX files into `public_candidates/`.
2. Run the teacher scan.
3. Review report rows with `score_up` or `mechanism_teacher`.
4. Run teacher extraction for selected tasks.
5. Convert extracted mechanisms into source code and registry entries.
6. Regenerate global inventory and recursive queue.
7. Adopt only through the normal stored/fresh/source gates.

## Success criteria

- Empty public candidate directory produces a valid empty report.
- Public candidate files with task ids are evaluated and compared.
- Lower-score but structurally smaller candidates are surfaced.
- Extraction produces source and insight drafts without mutating live models.
- The recursive improvement skill documents this workflow.
