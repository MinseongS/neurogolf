"""The single adoption gate.

Everything the project adopts from now on passes through gate(): three
checks encode hard-won catastrophe-prevention rules —
  1. isolated bundled fail == 0 (candidate actually works)
  2. strictly cheaper than the deployed net (cost = memory + params)
  3. unsupported-integer-TopK clean (a single uint8/int8-TopK net errors the ENTIRE
     Kaggle submission)

eval_isolated runs scoring in an isolated subprocess because ORT
weight-aliasing means only isolated per-task processes give true scores.
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from neurogolf.paths import ROOT, OVERFIT_NETS, STATE
from neurogolf.topk import find_unsigned_topk

_EVAL_CODE = """
import json, sys
from neurogolf.scoring import evaluate, load_task
task, path = int(sys.argv[1]), sys.argv[2]
r = evaluate(path, load_task(task), keep_failures=False)
row = {k: r.get(k) for k in ('ok','pass','fail','memory','params','points','error')}
row['cost'] = None if row['memory'] is None or row['params'] is None else int(row['memory']) + int(row['params'])
print(json.dumps(row))
"""


def _eval_timeout_seconds() -> int:
    raw = os.environ.get("NG_EVAL_TIMEOUT_SECONDS", "600")
    try:
        return max(1, int(raw))
    except ValueError:
        return 600

def eval_isolated(model_path: Path, task_num: int) -> dict:
    proc = subprocess.run([sys.executable, "-c", _EVAL_CODE, str(task_num), str(model_path)],
                          cwd=str(ROOT), text=True, capture_output=True,
                          timeout=_eval_timeout_seconds())
    if proc.returncode != 0 or not proc.stdout.strip():
        return {"ok": False, "fail": None, "cost": None, "error": (proc.stderr or "no output")[-500:]}
    return json.loads(proc.stdout.strip().splitlines()[-1])

def deployed_cost(task_num: int) -> int | None:
    mpath = STATE / "manifest.json"
    if mpath.exists():
        row = json.load(open(mpath)).get(f"{task_num:03d}")
        if row and row.get("cost") is not None:
            return int(row["cost"])
    res = eval_isolated(OVERFIT_NETS / f"task{task_num:03d}.onnx", task_num)
    return res.get("cost")

@dataclass
class GateResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    candidate: dict = field(default_factory=dict)
    incumbent_cost: int | None = None
    repairing_invalid_topk: bool = False

def gate(candidate: Path, task_num: int, *, repair_invalid: bool = False) -> GateResult:
    reasons: list[str] = []
    cand = eval_isolated(Path(candidate), task_num)
    if not cand.get("ok") or cand.get("fail") != 0:
        reasons.append(f"bundled fail != 0 (fail={cand.get('fail')}, error={cand.get('error')})")
    inc = deployed_cost(task_num)
    offenders = find_unsigned_topk(Path(candidate))
    incumbent_path = OVERFIT_NETS / f"task{task_num:03d}.onnx"
    incumbent_offenders = (
        find_unsigned_topk(incumbent_path)
        if repair_invalid and incumbent_path.exists()
        else []
    )
    repairing_invalid_topk = bool(
        repair_invalid and incumbent_offenders and not offenders
    )
    if (
        cand.get("cost") is None
        or inc is None
        or (cand["cost"] >= inc and not repairing_invalid_topk)
    ):
        reasons.append(f"not strictly cheaper (cand={cand.get('cost')}, deployed={inc})")
    if offenders:
        reasons.append("unsupported integer TopK: " + "; ".join(offenders))
    if repair_invalid and not incumbent_offenders:
        reasons.append("repair requested but incumbent has no unsupported integer TopK")
    return GateResult(
        ok=not reasons,
        reasons=reasons,
        candidate=cand,
        incumbent_cost=inc,
        repairing_invalid_topk=repairing_invalid_topk,
    )
