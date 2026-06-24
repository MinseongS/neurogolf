# NeuroGolf Custom Score Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and verify custom NeuroGolf solvers that improve score, starting with task017 and then expanding into new algorithmic primitive search.

**Architecture:** Keep one solver per `src/custom/taskNNN.py`, using the existing `build(task) -> onnx.ModelProto` convention. Use the project harness, fresh generator verification, and `src.adopt` as gates before touching submission artifacts. Separate immediate recovery work from exploratory algorithm search so failed research does not contaminate known-score changes.

**Tech Stack:** Python 3 via `.venv/bin/python`, ONNX helper APIs, NumPy, existing NeuroGolf modules `src.harness`, `src.genverify`, `src.adopt`, and project reports under `reports/`.

## Global Constraints

- Do not overwrite dirty user changes in `src/custom/task064.py`, `src/custom/task110.py`, `src/custom/task198.py`, or `src/custom/task370.py`.
- Do not make a Kaggle submission without explicit final confirmation of the exact zip file.
- Every accepted solver must pass stored harness evaluation, fresh generator verification, and `src.adopt`.
- If `/tmp/arc-gen` cannot be restored, stop before claiming a fresh-generalizing win.
- If task017 no longer beats the current base, record it as superseded and move to task182/new-search.
- Avoid ambiguous generators and connectivity/flood unrolling above the incumbent memory floor.

---

### Task 1: Restore and verify generator availability

**Files:**
- Read: `reports/arc_mapping.json`
- Read: `src/genverify.py`
- No source modification expected.

**Interfaces:**
- Consumes: `reports/arc_mapping.json` mapping task number to generator path.
- Produces: A working `/tmp/arc-gen` tree, or a clear blocker report.

- [ ] **Step 1: Check whether task017 generator path exists**

Run:

```bash
.venv/bin/python - <<'PY'
import json, pathlib
m=json.load(open('reports/arc_mapping.json'))
p=pathlib.Path(m['17']['generator'])
print(p)
print('exists=', p.exists())
PY
```

Expected if ready:

```text
/tmp/arc-gen/tasks/task_0dfd9992.py
exists= True
```

- [ ] **Step 2: If missing, search local temp and project caches**

Run:

```bash
find /private/tmp /tmp . -path '*arc-gen*' -o -name 'task_0dfd9992.py'
```

Expected useful output is either a path containing `task_0dfd9992.py` or no output.

- [ ] **Step 3: If a complete arc-gen tree is found outside `/tmp/arc-gen`, restore it by symlink or copy**

Use this only when the source directory visibly contains `tasks/task_0dfd9992.py` and `common.py`/generator support files:

```bash
ln -s /absolute/path/to/arc-gen /tmp/arc-gen
```

Expected:

```bash
test -f /tmp/arc-gen/tasks/task_0dfd9992.py
```

exits with status 0.

- [ ] **Step 4: If no local generator tree exists, stop and report the blocker**

Report exactly:

```text
Blocked: /tmp/arc-gen is missing, and no local copy of task_0dfd9992.py was found. Fresh verification cannot be trusted until the ARC-GEN repository is restored.
```

Do not continue to final adoption without this.

---

### Task 2: Implement task017 custom solver

**Files:**
- Create: `src/custom/task017.py`
- Read: `reports/tasklog/task017.md`
- Read: `src/custom/task029.py`
- Read: `src/custom/task096.py`

**Interfaces:**
- Consumes: Existing custom solver convention `build(task) -> onnx.ModelProto`.
- Produces: `src/custom/task017.py` with a `build(task)` function.

- [ ] **Step 1: Create `src/custom/task017.py` with the documented helper functions and model shell**

The file must expose `build(task)`. Implement these named graph stages:

```python
"""task017 (ARC-AGI 0dfd9992) — periodic cutout repair.

Recover one of the valid (mod, length, offset) tuples by fixed-sample template
matching, rebuild the 21x21 periodic colour-index grid, pad with a uint8
sentinel, and Equal into the free output tensor.
"""

import itertools
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

N = 30
SIDE = 21


def _valid_tuples():
    vals = []
    for mod in range(4, 10):
        for length in range(4, mod + 1):
            for offset in range(1, length + 1):
                vals.append((mod, length, offset))
    return vals


def _pattern(mod, length, offset):
    y = np.zeros((SIDE, SIDE), dtype=np.uint8)
    half = length // 2
    for r in range(SIDE):
        rr = (offset + r) % length - half
        for c in range(SIDE):
            cc = (offset + c) % length - half
            y[r, c] = ((rr * rr + cc * cc) % mod) + 1
    return y


def _sample_cells():
    # 15 cells from the tasklog-safe floor: spread over rows/cols so five
    # rectangular cutouts rarely erase all discriminating evidence.
    return np.array([
        [0, 0], [0, 5], [0, 10], [0, 15], [0, 20],
        [5, 2], [5, 8], [5, 14],
        [10, 0], [10, 6], [10, 12], [10, 18],
        [15, 4], [15, 16],
        [20, 20],
    ], dtype=np.int64)


def build(task):
    inits = []
    nodes = []

    def init(name, arr, dtype=None):
        arr = np.ascontiguousarray(arr, dtype=dtype) if dtype is not None else np.ascontiguousarray(arr)
        inits.append(numpy_helper.from_array(arr, name))
        return name

    def node(op, inputs, output, **attrs):
        nodes.append(helper.make_node(op, inputs, [output], **attrs))
        return output

    # The graph body is added in Step 2 before any smoke test is run.
    graph = helper.make_graph(
        nodes,
        "task017_periodic_repair",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, N, N])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, N, N])],
        inits,
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
        ir_version=IR_VERSION,
    )
```

- [ ] **Step 2: Add the ONNX graph construction before the `make_graph` call**

Required graph details:

```text
colour-index plane:
  Conv(input, arange(10).reshape(1,10,1,1)) -> colf [1,1,30,30] fp32

sample gather:
  Build int64 GatherND indices shaped [1, 15, 4] with entries [0,0,row,col].
  GatherND(colf, sample_indices) -> sample_colours [1,15] fp32

candidate table:
  cand_samples [106,15] fp32 generated by _pattern for every tuple and _sample_cells.
  Equal(sample_colours broadcast, cand_samples) -> match_bool [1,106,15]
  Cast(match_bool, FLOAT16) -> match_h [1,106,15]
  ReduceSum(match_h, axes=[2]) -> score [1,106]
  ArgMax(score, axis=1, keepdims=0) -> best_idx [1]

pattern gather:
  cand_patterns [106,21,21] uint8 generated by _pattern.
  Gather(cand_patterns, best_idx, axis=0) -> pat21 [1,21,21] uint8
  Unsqueeze to [1,1,21,21] if needed.

pad and output:
  Pad pat21 to [1,1,30,30] with uint8 sentinel 200.
  Equal(padded_label, arange(10).reshape(1,10,1,1).astype(uint8)) -> output.
```

The model output must be named exactly `output`, with input named exactly `input`, both `[1,10,30,30]` float32/bool-compatible as used in existing custom builders.

The resulting `build()` must create these initializers exactly once:

```python
tuples = _valid_tuples()
cells = _sample_cells()
patterns = np.stack([_pattern(*t) for t in tuples]).astype(np.uint8)
cand_samples = np.stack([p[cells[:, 0], cells[:, 1]] for p in patterns]).astype(np.float16)
init("w_colour", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
init("sample_idx", np.array([[[0, 0, int(r), int(c)] for r, c in cells]], dtype=np.int64))
init("cand_samples", cand_samples.reshape(1, len(tuples), len(cells)).astype(np.float16))
init("cand_patterns", patterns)
init("pad30", np.array([0, 0, 0, 0, 0, 0, N - SIDE, N - SIDE], dtype=np.int64))
init("sentinel_u8", np.array(200, dtype=np.uint8))
init("channels_u8", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1))
```

The resulting `build()` must add nodes with these names and dependencies:

```python
node("Conv", ["input", "w_colour"], "colf")
node("GatherND", ["colf", "sample_idx"], "sample_f")
node("Cast", ["sample_f"], "sample_h", to=TensorProto.FLOAT16)
node("Equal", ["sample_h", "cand_samples"], "match_b")
node("Cast", ["match_b"], "match_h", to=TensorProto.FLOAT16)
node("ReduceSum", ["match_h"], "score", axes=[2], keepdims=0)
node("ArgMax", ["score"], "best_idx", axis=1, keepdims=0)
node("Gather", ["cand_patterns", "best_idx"], "pat3", axis=0)
node("Unsqueeze", ["pat3"], "pat4", axes=[1])
node("Pad", ["pat4", "pad30", "sentinel_u8"], "label30", mode="constant")
node("Equal", ["label30", "channels_u8"], "output")
```

If ORT rejects `Unsqueeze` attribute syntax for the selected opset, use the same opset-13 input-form pattern used elsewhere in the repo and add an `axes1` int64 initializer with value `[1]`.

- [ ] **Step 3: Run import/build smoke test**

Run:

```bash
.venv/bin/python - <<'PY'
from src.harness import load_task
from src.custom.task017 import build
m = build(load_task(17))
print(m.ir_version, len(m.graph.node), len(m.graph.initializer))
assert m.graph.input[0].name == 'input'
assert m.graph.output[0].name == 'output'
PY
```

Expected: prints numeric model metadata and exits 0.

- [ ] **Step 4: Commit only `src/custom/task017.py` if smoke test passes**

Run:

```bash
git add src/custom/task017.py
git commit -m "feat: add task017 periodic repair custom solver"
```

Expected: one commit containing only `src/custom/task017.py`.

---

### Task 3: Verify and adopt task017

**Files:**
- Modify via adopt only if successful: `networks/task017.onnx`
- Modify via adopt only if successful: `reports/manifest.json`
- Read: `src/adopt.py`
- Read: `src/genverify.py`

**Interfaces:**
- Consumes: `src/custom/task017.py::build(task)`.
- Produces: adopted task017 network and manifest entry, or a superseded report.

- [ ] **Step 1: Run stored harness evaluation**

Run:

```bash
.venv/bin/python - <<'PY'
from src.harness import load_task, evaluate
from src.custom.task017 import build
task = load_task(17)
model = build(task)
res = evaluate(model, task)
print(res)
assert res['ok'], res
PY
```

Expected: `res['ok']` is true.

- [ ] **Step 2: Run fresh verification**

Run only after Task 1 confirms `/tmp/arc-gen` exists:

```bash
.venv/bin/python - <<'PY'
from src.genverify import fresh_pass
ok, run = fresh_pass(17, n=200)
print(ok, run)
assert run > 0
assert ok == run
PY
```

Expected: `200 200`.

- [ ] **Step 3: Run adopt gate**

Run:

```bash
.venv/bin/python -m src.adopt 017
```

Expected if successful:

```text
ADOPTED
```

If output says `REJECT` because incumbent is better, record task017 as superseded and do not manually copy the network.

- [ ] **Step 4: Review changed files**

Run:

```bash
git status --short
git diff -- reports/manifest.json
```

Expected allowed changes:

```text
 M reports/manifest.json
 M networks/task017.onnx
```

No dirty user files should be staged or modified by this task.

- [ ] **Step 5: Commit adoption result if adopted**

Run:

```bash
git add networks/task017.onnx reports/manifest.json
git commit -m "score: adopt task017 custom solver"
```

Expected: one commit containing only `networks/task017.onnx` and `reports/manifest.json`.

---

### Task 4: Create a safe submission zip after local adoption

**Files:**
- Create: `submission/submission_task017_custom.zip`
- Read: `networks/task*.onnx`

**Interfaces:**
- Consumes: current `networks/` directory after task017 adoption.
- Produces: a zip file containing exactly 400 `taskNNN.onnx` files.

- [ ] **Step 1: Count network files**

Run:

```bash
find networks -maxdepth 1 -name 'task*.onnx' | wc -l
```

Expected:

```text
     400
```

- [ ] **Step 2: Create submission directory**

Run:

```bash
mkdir -p submission
```

Expected: exits 0.

- [ ] **Step 3: Build a uniquely named zip**

Run from the `networks` directory:

```bash
zip -q ../submission/submission_task017_custom.zip task*.onnx
```

Expected: exits 0.

- [ ] **Step 4: Verify zip contents**

Run:

```bash
python - <<'PY'
import zipfile
p='submission/submission_task017_custom.zip'
with zipfile.ZipFile(p) as z:
    names=z.namelist()
print(len(names), names[:3], names[-3:])
assert len(names)==400
assert all(n.startswith('task') and n.endswith('.onnx') for n in names)
PY
```

Expected: prints `400` and exits 0.

Do not submit this zip until the user explicitly confirms the exact file.

---

### Task 5: New algorithm candidate triage

**Files:**
- Create: `reports/new_algo_candidates_2026-06-25.md`
- Read: `reports/sweep_ledger.json`
- Read: `reports/tasklog/*.md`
- Read: `reports/manifest.json`

**Interfaces:**
- Consumes: manifest/tasklog/ledger evidence.
- Produces: ranked candidate list for one next implementation target.

- [ ] **Step 1: Generate a ranked candidate table**

Run:

```bash
.venv/bin/python - <<'PY'
import json, re
manifest=json.load(open('reports/manifest.json'))['tasks']
ledger=json.load(open('reports/sweep_ledger.json'))
terms=re.compile(r'false|revisit|OPEN|untried|runtime|Gather|sentinel|template|RE-PROBE WIN|candidate', re.I)
rows=[]
dirty={64,110,198,370}
for e in ledger:
    n=e['task']
    if n in dirty:
        continue
    man=manifest.get(str(n),{})
    method=str(man.get('method', e.get('method','')))
    mem=int(man.get('memory', e.get('memory') or 0))
    pts=float(man.get('points', e.get('points') or 0))
    text=' '.join(str(e.get(k,'')) for k in ['status','verdict','note','class','sig'])
    if method.startswith('custom:') or mem < 8000:
        continue
    score=0
    if terms.search(text): score += 3
    if 'confirmed-infeasible' in text: score -= 4
    if re.search(r'\\bWALL\\b|ambiguous|non-deterministic|flood-fill', text, re.I): score -= 3
    if score > 0:
        rows.append((score, mem, pts, n, method, text[:300].replace('\\n',' ')))
for row in sorted(rows, reverse=True)[:20]:
    print('%03d score=%+d mem=%d pts=%.3f %s\\n  %s' % (row[3], row[0], row[1], row[2], row[4], row[5]))
PY
```

Expected: candidate lines excluding dirty tasks.

- [ ] **Step 2: Write `reports/new_algo_candidates_2026-06-25.md` from the ranked data**

Run:

```bash
.venv/bin/python - <<'PY'
import json, re, pathlib
manifest=json.load(open('reports/manifest.json'))['tasks']
ledger=json.load(open('reports/sweep_ledger.json'))
terms=re.compile(r'false|revisit|OPEN|untried|runtime|Gather|sentinel|template|RE-PROBE WIN|candidate', re.I)
dirty={64,110,198,370}
rows=[]
for e in ledger:
    n=e['task']
    if n in dirty:
        continue
    man=manifest.get(str(n),{})
    method=str(man.get('method', e.get('method','')))
    mem=int(man.get('memory', e.get('memory') or 0))
    pts=float(man.get('points', e.get('points') or 0))
    text=' '.join(str(e.get(k,'')) for k in ['status','verdict','note','class','sig'])
    if method.startswith('custom:') or mem < 8000:
        continue
    score=0
    if terms.search(text): score += 3
    if 'confirmed-infeasible' in text: score -= 4
    if re.search(r'\bWALL\b|ambiguous|non-deterministic|flood-fill', text, re.I): score -= 3
    if score > 0:
        hyp='Needs manual read of tasklog before build.'
        if n == 182:
            hyp='Runtime-kernel Conv shape match may still beat current base after reconciliation.'
        elif n == 25:
            hyp='Move-dot-beside-line scatter was previously stalled, not proven impossible.'
        elif n == 367:
            hyp='False infeasible; prior closed-form direction-carry idea may transfer.'
        rows.append((score, mem, pts, n, method, hyp))
rows=sorted(rows, reverse=True)[:10]
lines=[
    '# New algorithm candidates — 2026-06-25',
    '',
    '## Selection rule',
    '',
    'Candidates are external-method tasks with at least 8000 memory bytes, no dirty user source file, and evidence for a non-wall primitive such as runtime Conv, coordinate Gather, sentinel label carrier, parameter-template recovery, or a prior false infeasible verdict.',
    '',
    '## Ranked candidates',
    '',
    '| Rank | Task | Current method | Current points | Current memory | Hypothesis | Bail condition |',
    '|---:|---:|---|---:|---:|---|---|',
]
for i,(_,mem,pts,n,method,hyp) in enumerate(rows,1):
    bail='Reject if stored/fresh/adopt fails or if incumbent is already below the documented custom floor.'
    lines.append(f'| {i} | {n:03d} | {method} | {pts:.3f} | {mem} | {hyp} | {bail} |')
rec = f'Task {rows[0][3]:03d}' if rows else 'No candidate'
lines += [
    '',
    '## Recommended next implementation',
    '',
    f'{rec} after task017, unless task017 adoption already gives enough local improvement for the current submission cycle.',
    '',
]
pathlib.Path('reports/new_algo_candidates_2026-06-25.md').write_text('\n'.join(lines))
PY
```

Expected: file exists and contains a ranked table with concrete numeric values. It must not include `task064`, `task110`, `task198`, or `task370`.

- [ ] **Step 3: Commit the candidate report**

Run:

```bash
git add reports/new_algo_candidates_2026-06-25.md
git commit -m "docs: rank neurogolf new algorithm candidates"
```

Expected: one commit containing only the candidate report.

---

### Task 6: Stop point and review

**Files:**
- Read: `git status --short`
- Read: `git log --oneline -5`

**Interfaces:**
- Consumes: outputs from Tasks 1-5.
- Produces: final handoff summary and explicit next action request.

- [ ] **Step 1: Verify workspace state**

Run:

```bash
git status --short
git log --oneline -5
```

Expected:

```text
```

The only remaining dirty files should be the pre-existing user files and any untracked reports that existed before this plan.

- [ ] **Step 2: Summarize outcomes**

Report:

```text
task017: adopted / rejected / blocked
fresh verification: pass / blocked
submission zip: path or not created
new algorithm next target: task number and reason
dirty user files preserved: yes/no
```

- [ ] **Step 3: Ask before Kaggle submission**

If a zip exists and local verification passed, ask:

```text
Submit submission/submission_task017_custom.zip to Kaggle now?
```

Do not run `kaggle competitions submit` before the user confirms.
