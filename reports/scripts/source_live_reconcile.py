#!/usr/bin/env python3
"""Compare local source builders against current live networks."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from src.harness import evaluate, load_task


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
NETWORKS = ROOT / "networks"


def parse_task_list(text: str) -> list[int]:
    tasks: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if start > end:
                raise argparse.ArgumentTypeError(f"invalid descending range: {part}")
            tasks.update(range(start, end + 1))
        else:
            tasks.add(int(part))
    bad = [task for task in tasks if task < 1 or task > 400]
    if bad:
        raise argparse.ArgumentTypeError(f"task out of range 1..400: {bad[0]}")
    if not tasks:
        raise argparse.ArgumentTypeError("task list is empty")
    return sorted(tasks)


def safe_eval_model(model_or_path: Any, task_num: int) -> dict[str, Any]:
    try:
        return evaluate(model_or_path, load_task(task_num), keep_failures=False)
    except Exception as exc:
        return {
            "ok": False,
            "filesize": None,
            "memory": None,
            "params": None,
            "points": 0.0,
            "pass": 0,
            "fail": None,
            "error": str(exc),
        }


def eval_source(task_num: int) -> dict[str, Any]:
    try:
        mod = importlib.import_module(f"src.custom.task{task_num:03d}")
        importlib.reload(mod)
        return safe_eval_model(mod.build(load_task(task_num)), task_num)
    except Exception as exc:
        return {
            "ok": False,
            "filesize": None,
            "memory": None,
            "params": None,
            "points": 0.0,
            "pass": 0,
            "fail": None,
            "error": str(exc),
        }


def differs(live: dict[str, Any], source: dict[str, Any]) -> bool:
    if live.get("ok") != source.get("ok"):
        return True
    for key in ("points", "memory", "params"):
        if live.get(key) != source.get(key):
            return True
    return False


def reconcile(task_nums: list[int]) -> list[dict[str, Any]]:
    rows = []
    for task_num in task_nums:
        live_path = NETWORKS / f"task{task_num:03d}.onnx"
        live = safe_eval_model(str(live_path), task_num)
        source = eval_source(task_num)
        if differs(live, source):
            rows.append(
                {
                    "task": task_num,
                    "live": live,
                    "source": source,
                    "delta_points": float(source.get("points") or 0) - float(live.get("points") or 0),
                    "delta_memory": None if source.get("memory") is None or live.get("memory") is None else int(source["memory"]) - int(live["memory"]),
                    "delta_params": None if source.get("params") is None or live.get("params") is None else int(source["params"]) - int(live["params"]),
                }
            )
    rows.sort(key=lambda row: (row["delta_points"], row["task"]))
    return rows


def write_markdown(rows: list[dict[str, Any]], path: Path, task_nums: list[int]) -> None:
    source_lags = [row for row in rows if row["delta_points"] < -1e-9]
    source_ahead = [row for row in rows if row["delta_points"] > 1e-9]
    if len(task_nums) == 400 and task_nums[0] == 1 and task_nums[-1] == 400:
        task_scope = "001-400"
    else:
        task_scope = ",".join(f"{task:03d}" for task in task_nums)
    lines = [
        "# Source/live reconcile",
        "",
        "This report distinguishes `build()` coverage from live-model parity.",
        "",
        f"- task scope: {task_scope}",
        f"- mismatches: {len(rows)}",
        f"- source lags live: {len(source_lags)}",
        f"- source ahead of live: {len(source_ahead)}",
        "",
        "| task | live pts | src pts | Δpts | live mem | src mem | live params | src params | note |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows[:120]:
        live = row["live"]
        source = row["source"]
        if row["delta_points"] < -1e-9:
            note = "source_lags_live"
        elif row["delta_points"] > 1e-9:
            note = "source_ahead_live"
        else:
            note = "metric_shape_diff"
        lines.append(
            f"| {row['task']:03d} | {float(live.get('points') or 0):.3f} | {float(source.get('points') or 0):.3f} | "
            f"{row['delta_points']:+.3f} | {live.get('memory')} | {source.get('memory')} | "
            f"{live.get('params')} | {source.get('params')} | {note} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=parse_task_list,
        help="Comma-separated task numbers and ranges, e.g. 1,7,20-25.",
    )
    parser.add_argument("--task", type=int, help="Single task number shortcut.")
    parser.add_argument("--start", type=int, default=1, help="First task number when --tasks/--task is omitted.")
    parser.add_argument("--end", type=int, default=400, help="Last task number when --tasks/--task is omitted.")
    parser.add_argument("--out-json", default=str(REPORTS / "source_live_reconcile.json"))
    parser.add_argument("--out-md", default=str(REPORTS / "source_live_reconcile.md"))
    args = parser.parse_args()
    if args.task is not None and args.tasks is not None:
        parser.error("--task and --tasks are mutually exclusive")
    if args.task is not None:
        task_nums = parse_task_list(str(args.task))
    elif args.tasks is not None:
        task_nums = args.tasks
    else:
        if args.start < 1 or args.end > 400 or args.start > args.end:
            parser.error("--start/--end must define an ascending range within 1..400")
        task_nums = list(range(args.start, args.end + 1))
    rows = reconcile(task_nums)
    Path(args.out_json).write_text(json.dumps(rows, indent=2, sort_keys=True, default=str))
    write_markdown(rows, Path(args.out_md), task_nums)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    print(f"tasks checked: {len(task_nums)}")
    print(f"mismatches: {len(rows)}")


if __name__ == "__main__":
    main()
