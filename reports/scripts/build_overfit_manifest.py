#!/usr/bin/env python3
"""Measure the active 8000-mode submission set.

Writes:
  reports/overfit_manifest.json
  reports/overfit_manifest.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.harness import evaluate, load_task


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
DEFAULT_NETS = ROOT / "submission" / "overfit_nets"


def measure_task_once(task_num: int, nets_dir: Path) -> dict[str, Any]:
    path = nets_dir / f"task{task_num:03d}.onnx"
    row: dict[str, Any] = {
        "task": task_num,
        "path": str(path.relative_to(ROOT)) if path.exists() else str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        row.update({"ok": False, "pass": 0, "fail": None, "memory": None, "params": None, "points": 0.0, "error": "missing"})
        return row
    try:
        result = evaluate(str(path), load_task(task_num), keep_failures=False)
    except Exception as exc:
        row.update({"ok": False, "pass": 0, "fail": None, "memory": None, "params": None, "points": 0.0, "error": str(exc)})
        return row
    row.update({key: result.get(key) for key in ("ok", "pass", "fail", "memory", "params", "points", "error")})
    row["cost"] = None if row.get("memory") is None or row.get("params") is None else int(row["memory"]) + int(row["params"])
    return row


def measure_task(task_num: int, nets_dir: Path, retries: int = 1) -> dict[str, Any]:
    attempts = []
    for _ in range(retries + 1):
        row = measure_task_once(task_num, nets_dir)
        attempts.append({key: row.get(key) for key in ("ok", "pass", "fail", "memory", "params", "points", "error")})
        if row.get("ok") and row.get("fail") == 0:
            row["attempts"] = attempts
            return row
    isolated = measure_task_isolated(task_num, nets_dir)
    attempts.append({"isolated": True, **{key: isolated.get(key) for key in ("ok", "pass", "fail", "memory", "params", "points", "error")}})
    if isolated.get("ok") and isolated.get("fail") == 0:
        isolated["attempts"] = attempts
        isolated["isolated_recovered"] = True
        return isolated
    row["attempts"] = attempts
    return row


def measure_task_isolated(task_num: int, nets_dir: Path) -> dict[str, Any]:
    code = """
import json
from pathlib import Path
from src.harness import evaluate, load_task
task = int(__import__('sys').argv[1])
path = Path(__import__('sys').argv[2]) / f'task{task:03d}.onnx'
result = evaluate(str(path), load_task(task), keep_failures=False)
row = {'task': task, 'path': str(path), 'exists': path.exists()}
row.update({key: result.get(key) for key in ('ok', 'pass', 'fail', 'memory', 'params', 'points', 'error')})
row['cost'] = None if row.get('memory') is None or row.get('params') is None else int(row['memory']) + int(row['params'])
print(json.dumps(row, sort_keys=True))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code, str(task_num), str(nets_dir)],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "task": task_num,
            "path": str(nets_dir / f"task{task_num:03d}.onnx"),
            "exists": (nets_dir / f"task{task_num:03d}.onnx").exists(),
            "ok": False,
            "pass": 0,
            "fail": None,
            "memory": None,
            "params": None,
            "points": 0.0,
            "cost": None,
            "error": (proc.stderr or proc.stdout).strip(),
        }
    line = proc.stdout.strip().splitlines()[-1]
    row = json.loads(line)
    path = Path(row["path"])
    row["path"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    return row


def write_markdown(manifest: dict[str, Any], path: Path) -> None:
    rows = manifest["tasks"]
    failing = [row for row in rows if row.get("fail") not in (0, None) or not row.get("ok")]
    top_cost = sorted(rows, key=lambda row: row.get("cost") or -1, reverse=True)[:40]
    lines = [
        "# Overfit Manifest",
        "",
        "Active 8000-mode submission measurement for `submission/overfit_nets/`.",
        "",
        f"- tasks: `{manifest['n_tasks']}`",
        f"- ok tasks: `{manifest['n_ok']}`",
        f"- bundled failing/error tasks: `{len(failing)}`",
        f"- total points: `{manifest['total_points']:.6f}`",
        f"- total cost: `{manifest['total_cost']}`",
        f"- measured nets dir: `{manifest['nets_dir']}`",
        "",
    ]
    if failing:
        lines.extend([
            "## Failing / Error Tasks",
            "",
            "| task | ok | pass | fail | memory | params | points | error |",
            "|---:|---|---:|---:|---:|---:|---:|---|",
        ])
        for row in failing:
            lines.append(
                f"| {row['task']:03d} | {row.get('ok')} | {row.get('pass')} | {row.get('fail')} | "
                f"{row.get('memory')} | {row.get('params')} | {float(row.get('points') or 0):.6f} | {row.get('error') or ''} |"
            )
        lines.append("")
    lines.extend([
        "## Highest Cost Tasks",
        "",
        "| rank | task | points | memory | params | cost |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for rank, row in enumerate(top_cost, 1):
        lines.append(
            f"| {rank} | {row['task']:03d} | {float(row.get('points') or 0):.6f} | "
            f"{row.get('memory')} | {row.get('params')} | {row.get('cost')} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nets-dir", default=str(DEFAULT_NETS), help="Directory containing taskNNN.onnx files.")
    parser.add_argument("--out-json", default=str(REPORTS / "overfit_manifest.json"))
    parser.add_argument("--out-md", default=str(REPORTS / "overfit_manifest.md"))
    parser.add_argument("--retries", type=int, default=1, help="Retry a task this many times when bundled eval fails.")
    parser.add_argument(
        "--isolated-all",
        action="store_true",
        help="Measure every task in a fresh subprocess to avoid long-lived ORT session stalls.",
    )
    args = parser.parse_args()

    nets_dir = Path(args.nets_dir).resolve()
    if args.isolated_all:
        rows = [measure_task_isolated(task_num, nets_dir) for task_num in range(1, 401)]
    else:
        rows = [measure_task(task_num, nets_dir, retries=args.retries) for task_num in range(1, 401)]
    manifest = {
        "mode": "8000_overfit",
        "nets_dir": str(nets_dir.relative_to(ROOT)) if nets_dir.is_relative_to(ROOT) else str(nets_dir),
        "n_tasks": len(rows),
        "n_ok": sum(1 for row in rows if row.get("ok") and row.get("fail") == 0),
        "total_points": sum(float(row.get("points") or 0.0) for row in rows),
        "total_cost": sum(int(row.get("cost") or 0) for row in rows),
        "tasks": rows,
    }
    Path(args.out_json).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    write_markdown(manifest, Path(args.out_md))
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    print(f"tasks: {manifest['n_tasks']}")
    print(f"ok: {manifest['n_ok']}")
    print(f"total_points: {manifest['total_points']:.6f}")


if __name__ == "__main__":
    main()
