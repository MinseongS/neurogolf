"""Score every networks/taskNNN.onnx against data/taskNNN.json.

Rebuilds a manifest-shaped view of the deployed ONNX artifacts. Use when the
local nets are the source of truth (e.g. recovered from a submission zip with
no matching src/custom/ source).

    PYTHONPATH=. .venv/bin/python reports/scripts/score_networks.py
    PYTHONPATH=. .venv/bin/python reports/scripts/score_networks.py --tasks 47,127
"""

import argparse
import concurrent.futures
import json
import pathlib

from src import harness

ROOT = pathlib.Path(__file__).resolve().parents[2]
NETWORKS = ROOT / "networks"
OUT = ROOT / "reports" / "networks_score.json"


def score_one(task_num):
    path = NETWORKS / f"task{task_num:03d}.onnx"
    if not path.exists():
        return task_num, {"ok": False, "error": "missing net", "points": 0.0}
    try:
        res = harness.evaluate(path, harness.load_task(task_num))
    except Exception as e:  # noqa: BLE001 - report, never abort the sweep
        return task_num, {"ok": False, "error": f"crash: {e}", "points": 0.0}
    res.pop("failures", None)
    return task_num, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", help="comma-separated task numbers (default: all 400)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    tasks = (
        [int(t) for t in args.tasks.split(",")]
        if args.tasks
        else list(range(1, 401))
    )

    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for task_num, res in pool.map(score_one, tasks):
            results[task_num] = res

    total = sum(r["points"] for r in results.values())
    failed = {k: v for k, v in results.items() if not v["ok"]}

    OUT.write_text(json.dumps({
        "total": total,
        "tasks": {str(k): results[k] for k in sorted(results)},
    }, indent=1))

    print(f"scored {len(tasks)} tasks")
    print(f"total points: {total:.4f}")
    print(f"not ok: {len(failed)}")
    for k, v in sorted(failed.items()):
        print(f"  task{k:03d}: fail={v.get('fail')} error={v.get('error')}")


if __name__ == "__main__":
    main()
