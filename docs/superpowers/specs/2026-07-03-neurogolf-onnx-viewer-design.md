# NeuroGolf ONNX Viewer Design

## Goal

Build a local Streamlit tool for fast task inspection and candidate ONNX iteration.
The first version focuses on visual feedback: select a task, inspect examples,
run the current deployed ONNX or a scratch `cand.py`, and see output, diff, cost,
and score immediately.

## Scope

The app lives at `tools/onnx_viewer.py` and runs with:

```bash
PYTHONPATH=. .venv/bin/streamlit run tools/onnx_viewer.py
```

Version 1 includes:

- Sidebar task selector for `task001` through `task400`.
- Example count control, defaulting to 30 examples.
- Stored examples first, with optional fresh-cache examples when
  `reports/fresh_cache/taskNNN.npz` exists.
- Candidate source selector:
  - deployed `networks/taskNNN.onnx`;
  - source `src/custom/taskNNN.py`;
  - arbitrary scratch `cand.py` path exposing `build(task)`.
- Per-run metrics:
  - pass/fail count;
  - memory;
  - params;
  - points = `25 - ln(memory + params)`;
  - build/runtime error text.
- Visual comparison for each displayed example:
  - input;
  - expected output;
  - candidate output;
  - mismatch mask.

Version 1 deliberately does not include a node-by-node ONNX graph editor. It can
be added later after the core compare loop is useful.

## Architecture

Streamlit runs in one Python process and directly uses the existing NeuroGolf
harness.

- `src.harness.load_task` loads stored examples.
- `src.harness.convert_to_numpy` converts ARC grids to one-hot tensors.
- `src.harness.evaluate` computes official local score and cost for models.
- `src.harness.run_network` runs a prepared ONNX Runtime session and applies the
  free `>0` threshold.
- Scratch candidates are loaded with `importlib.util.spec_from_file_location` and
  called as `build(task)`.

For `.onnx` inputs, the app loads the model path directly. For source or scratch
Python candidates, the app builds an in-memory ONNX model and evaluates/runs it.

## UI

The sidebar contains task and data controls. The main page is a compact research
surface:

1. top metrics row for selected candidate;
2. candidate path/source controls;
3. scrollable example list with four fixed columns;
4. an error panel shown only when build, shape inference, evaluation, or runtime
   fails.

Grid rendering uses a stable 10-colour ARC palette. Cells are fixed-size squares
so comparisons do not shift layout. Mismatch views highlight only cells where the
candidate one-hot output differs from expected.

## Data Flow

1. User selects task and candidate.
2. App loads task JSON and candidate model.
3. App evaluates candidate once with `evaluate`.
4. App runs up to the selected number of examples through ONNX Runtime.
5. App renders input, expected, output, and diff.

Fresh-cache support is read-only: if `reports/fresh_cache/taskNNN.npz` exists,
the app may show those cached inputs and expected outputs. It does not generate
new fresh data in version 1.

## Error Handling

Errors are displayed inline and do not crash the Streamlit process:

- missing candidate path;
- missing `build(task)`;
- ONNX checker or shape inference failure;
- ONNX Runtime session failure;
- output shape mismatch;
- unsupported fresh-cache layout.

## Testing

Manual smoke checks:

- load `task187` deployed ONNX and show 30 stored examples;
- load `src/custom/task187.py` and compare metrics against `reports/manifest.json`;
- load the known scratch `task187fold/cand.py` if present;
- verify mismatch masks show no cells for passing stored examples;
- verify an invalid candidate path reports a clear error.

No production submission artifacts are modified by this tool.
