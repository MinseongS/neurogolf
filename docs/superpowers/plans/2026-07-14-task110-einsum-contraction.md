# Task110 Einsum Contraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing bundled-exact task110 cost-340 Einsum finish the mandatory gate, then adopt and Kaggle-test it on the confirmed 7424.42 base.

**Architecture:** Generate algebraically identical one-node models by permuting operands and equation terms together, then benchmark each in an isolated ORT process. If every permutation times out, calculate the exact lower bound of the staged factorization and stop if it exceeds the approved cost ceiling.

**Tech Stack:** Python 3.13, ONNX 1.21.0, ONNX Runtime 1.26.0, pytest, NeuroGolf `ng` CLI.

## Global Constraints

- Work only in `/Users/minseong/.codex/worktrees/4c36/neurogolf`.
- Candidate artifacts stay under `candidates/task110/`.
- Preserve onnx==1.21.0 and onnxruntime==1.26.0.
- Adoption uses `ng adopt`; submission uses `ng pack` then `ng submit`.
- Bundled fail=0 and cost <=1668 are mandatory.
- Fresh evaluation is diagnostic and cannot block an overfit adoption.
- Never modify deployed task110 before `ng adopt` succeeds.

---

### Task 1: Algebra-Preserving Reorder Builder

**Files:**
- Create: `candidates/task110/build_reordered_einsum.py`
- Create: `tests/test_task110_reordered_einsum.py`
- Read: `candidates/task110/bundled_selfeinsum.onnx`

**Interfaces:**
- Consumes: the existing one-node bundled Einsum.
- Produces: `VARIANT_ORDERS`, `reorder_model(source, order)`, and `build_all(out_dir)`.

- [ ] **Step 1: Write the failing structural tests**

```python
from collections import Counter
from pathlib import Path
import onnx
from candidates.task110.build_reordered_einsum import SOURCE, VARIANT_ORDERS, reorder_model

def signature(model):
    node = model.graph.node[0]
    attr = next(a for a in node.attribute if a.name == "equation")
    lhs = onnx.helper.get_attribute_value(attr).decode().split("->", 1)[0].split(",")
    return Counter(zip(node.input, lhs, strict=True))

def test_every_order_is_a_full_permutation():
    operand_count = len(onnx.load(SOURCE).graph.node[0].input)
    for name, order in VARIANT_ORDERS.items():
        assert tuple(sorted(order)) == tuple(range(operand_count)), name

def test_reordering_preserves_operand_term_pairs():
    original = onnx.load(SOURCE)
    for order in VARIANT_ORDERS.values():
        candidate = reorder_model(SOURCE, order)
        onnx.checker.check_model(candidate, full_check=True)
        assert signature(candidate) == signature(original)
```

- [ ] **Step 2: Run the test and confirm it fails before implementation**

Run: `.venv/bin/pytest tests/test_task110_reordered_einsum.py -q`

Expected: collection fails because `build_reordered_einsum` does not exist.

- [ ] **Step 3: Implement the minimal builder**

```python
from pathlib import Path
import onnx

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "candidates/task110/bundled_selfeinsum.onnx"
OUT_DIR = ROOT / "candidates/task110/reordered"

PAIR_INPUT_FIRST = (0,1,15,23, 2,3,16,24, 4,5,17,25, 6,7,18,26,
                    8,9,19,27, 10,11,20,28, 12,13,21,29, 14,22)
PAIR_SUPPORT_FIRST = (15,23,0,1, 16,24,2,3, 17,25,4,5, 18,26,6,7,
                      19,27,8,9, 20,28,10,11, 21,29,12,13, 22,14)
COL_THEN_ROW = (18,26,6,7, 19,27,8,9, 20,28,10,11, 21,29,12,13,
                15,23,0,1, 16,24,2,3, 17,25,4,5, 22,14)
VARIANT_ORDERS = {
    "support_first": tuple(range(15,30)) + tuple(range(15)),
    "pair_input_first": PAIR_INPUT_FIRST,
    "pair_support_first": PAIR_SUPPORT_FIRST,
    "col_then_row": COL_THEN_ROW,
    "reverse": tuple(reversed(range(30))),
}

def reorder_model(source: Path, order: tuple[int, ...]) -> onnx.ModelProto:
    model = onnx.load(source)
    node = model.graph.node[0]
    operand_count = len(node.input)
    if tuple(sorted(order)) != tuple(range(operand_count)):
        raise ValueError(f"order must permute range({operand_count})")
    attr = next(a for a in node.attribute if a.name == "equation")
    lhs, rhs = onnx.helper.get_attribute_value(attr).decode().split("->", 1)
    terms, inputs = lhs.split(","), list(node.input)
    del node.input[:]
    node.input.extend(inputs[i] for i in order)
    attr.s = (",".join(terms[i] for i in order) + "->" + rhs).encode()
    onnx.checker.check_model(model, full_check=True)
    return model

def build_all(out_dir: Path = OUT_DIR) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, order in VARIANT_ORDERS.items():
        path = out_dir / f"{name}.onnx"
        onnx.save(reorder_model(SOURCE, order), path)
        result[name] = path
    return result

if __name__ == "__main__":
    for name, path in build_all().items():
        print(name, path)
```

- [ ] **Step 4: Run tests and generate five variants**

```bash
.venv/bin/pytest tests/test_task110_reordered_einsum.py -q
.venv/bin/python candidates/task110/build_reordered_einsum.py
```

Expected: two tests pass and five model paths print.

- [ ] **Step 5: Commit Task 1**

```bash
git add candidates/task110/build_reordered_einsum.py tests/test_task110_reordered_einsum.py
git commit -m "feat: generate task110 einsum reorder variants"
```

### Task 2: Isolated Runtime Benchmark

**Files:**
- Create: `candidates/task110/benchmark_reorders.py`
- Create at runtime: `candidates/task110/reorder_benchmark.json`
- Modify: `tests/test_task110_reordered_einsum.py`

**Interfaces:**
- Consumes: `build_all()`.
- Produces: `benchmark(path, timeout)` and JSON results containing `status`, `seconds`, `path`, and `error`.

- [ ] **Step 1: Add the failing result-contract test**

```python
from candidates.task110.benchmark_reorders import benchmark

def test_benchmark_reports_required_keys():
    result = benchmark(Path("candidates/task110/bundled_selfeinsum.onnx"), timeout=0.01)
    assert set(result) == {"status", "seconds", "path", "error"}
    assert result["status"] in {"ok", "timeout", "error"}
```

- [ ] **Step 2: Confirm the benchmark test fails before implementation**

Run: `.venv/bin/pytest tests/test_task110_reordered_einsum.py::test_benchmark_reports_required_keys -q`

Expected: import failure for `benchmark_reorders`.

- [ ] **Step 3: Implement the isolated benchmark**

```python
import json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from candidates.task110.build_reordered_einsum import ROOT, build_all

RUN_CODE = r'''import sys,time,onnxruntime as ort
from neurogolf.scoring import convert_to_numpy,load_task
d=load_task(110); e=(d["train"]+d["test"]+d["arc-gen"])[0]
x=convert_to_numpy(e)["input"]
s=ort.InferenceSession(sys.argv[1],providers=["CPUExecutionProvider"])
t=time.perf_counter(); s.run(["output"],{"input":x}); print(time.perf_counter()-t)'''

def benchmark(path: Path, timeout: float = 10.0) -> dict[str, object]:
    started = time.perf_counter()
    try:
        proc = subprocess.run([sys.executable,"-c",RUN_CODE,str(path)], cwd=ROOT,
                              text=True,capture_output=True,timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status":"timeout","seconds":time.perf_counter()-started,
                "path":str(path),"error":""}
    if proc.returncode:
        return {"status":"error","seconds":time.perf_counter()-started,
                "path":str(path),"error":proc.stderr[-500:]}
    return {"status":"ok","seconds":float(proc.stdout.strip().splitlines()[-1]),
            "path":str(path),"error":""}

def main():
    results={name:benchmark(path) for name,path in build_all().items()}
    out=ROOT/"candidates/task110/reorder_benchmark.json"
    out.write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(results,indent=2))

if __name__ == "__main__": main()
```

- [ ] **Step 4: Run tests and benchmark**

```bash
.venv/bin/pytest tests/test_task110_reordered_einsum.py -q
.venv/bin/python candidates/task110/benchmark_reorders.py
```

Expected: tests pass and at least one variant reports `status: ok`, `seconds < 10`.

- [ ] **Step 5: If all variants time out, prove the staged fallback is cost-dead**

```bash
.venv/bin/python - <<'PY'
row = 1*30*30*4
col = 1*30*30*4
cost = row + col + 340
print({"row":row,"col":col,"minimum_staged_cost":cost})
assert cost == 7540 and cost > 1668
PY
```

Expected: minimum staged cost 7540. The shared donor indices require `[n,r,p]` and `[n,c,q]`; stop without building a score-negative fallback.

- [ ] **Step 6: Commit Task 2**

```bash
git add candidates/task110/benchmark_reorders.py tests/test_task110_reordered_einsum.py candidates/task110/reorder_benchmark.json
git commit -m "test: benchmark task110 einsum contraction orders"
```

### Task 3: Gate and Adopt the Fastest Exact Variant

**Files:**
- Read: `candidates/task110/reorder_benchmark.json`
- Replace through CLI only: `submission/overfit_nets/task110.onnx`
- Modify through CLI only: deployment manifest and task110 ledger artifacts

**Interfaces:**
- Consumes: fastest candidate with `status: ok` and runtime below 10 seconds.
- Produces: a gate-approved deployed task110 model with bundled fail=0 and cost <=1668.

- [ ] **Step 1: Select the fastest successful candidate without editing deployment**

```bash
CANDIDATE=$(.venv/bin/python - <<'PY'
import json
from pathlib import Path
r = json.loads(Path("candidates/task110/reorder_benchmark.json").read_text())
ok = [(v["seconds"], v["path"]) for v in r.values() if v["status"] == "ok"]
if not ok:
    raise SystemExit("no successful reorder candidate")
print(min(ok)[1])
PY
)
printf '%s\n' "$CANDIDATE"
```

Expected: one path below `candidates/task110/reordered/`.

- [ ] **Step 2: Run the mandatory task gate**

Run: `.venv/bin/ng gate "$CANDIDATE" --task 110`

Expected: bundled fail=0, candidate cost 340, and cheaper than deployed cost 2751. Fresh results may be recorded diagnostically but do not block an overfit candidate.

- [ ] **Step 3: Adopt only through the project CLI**

Run: `.venv/bin/ng adopt "$CANDIDATE" --task 110`

Expected: adoption succeeds through the mandatory gate and replaces only task110 deployment state.

- [ ] **Step 4: Verify the deployed package remains complete and internally consistent**

```bash
.venv/bin/ng status
.venv/bin/ng verify --hash
```

Expected: 400/400 and HASH-OK; manifest total improves by about 2.091 points from 7424.2851 to about 7426.376.

- [ ] **Step 5: Preserve dirty-worktree ownership**

Inspect `git status --short` and `git diff -- submission/overfit_nets/task110.onnx state/tasks/task110.yaml state/manifest.json`. Do not stage any pre-existing modified ledger or manifest file merely because `ng adopt` touched it. Commit only newly created implementation/test files whose ownership is unambiguous.

### Task 4: Isolated Kaggle Confirmation and Handoff

**Files:**
- Modify by replacement: `state/STATE.md`
- Modify: `state/submissions.md`
- Read/modify if owned by this experiment: `state/tasks/task110.yaml`

**Interfaces:**
- Consumes: adopted, locally verified 400/400 package.
- Produces: one isolated leaderboard measurement and an exact rollback or confirmed new base.

- [ ] **Step 1: Check for concurrent submissions before packing**

Run: `.venv/bin/kaggle competitions submissions -c neuro-golf`

Expected: no unaccounted in-flight submission from another session. If one exists, wait and reconcile before submitting.

- [ ] **Step 2: Pack and submit the isolated task110 change**

```bash
.venv/bin/ng pack
.venv/bin/ng submit -m "7424.42 base + task110 reordered exact einsum"
```

Expected: one new submission ID. Record the ID immediately in `state/submissions.md`.

- [ ] **Step 3: Poll until the leaderboard score is final**

Run: `.venv/bin/kaggle competitions submissions -c neuro-golf`

Expected if the local model generalizes to the leaderboard contract: approximately 7426.3 to 7426.7. Treat the actual leaderboard result as authoritative.

- [ ] **Step 4: Keep or rollback based on the isolated result**

If the score exceeds 7424.42, retain the adopted model and make it the new baseline. If it does not, restore the pre-adoption task110 candidate using `ng adopt` on the saved deployed backup, then rerun `ng verify --hash`; never copy directly into `submission/overfit_nets/`.

- [ ] **Step 5: Replace the live handoff and update experiment ledgers**

Replace, do not append to, `state/STATE.md`. Record the candidate equation order, local gate result, submission ID, final leaderboard score, decision, and next live lever in the appropriate ledgers. Use only the four ledger fields for any negative/floor judgment and leave the lever dormant rather than dead.

- [ ] **Step 6: Run final verification**

```bash
.venv/bin/ng status
.venv/bin/ng verify --hash
.venv/bin/pytest tests/test_task110_reordered_einsum.py -q
git status --short
```

Expected: 400/400, HASH-OK, all task110 tests pass, and unrelated dirty-worktree changes remain untouched.
