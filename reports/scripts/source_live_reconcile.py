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


def reconcile() -> list[dict[str, Any]]:
    rows = []
    for task_num in range(1, 401):
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


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    source_lags = [row for row in rows if row["delta_points"] < -1e-9]
    source_ahead = [row for row in rows if row["delta_points"] > 1e-9]
    lines = [
        "# Source/live reconcile",
        "",
        "This report distinguishes `build()` coverage from live-model parity.",
        "",
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
    parser.add_argument("--out-json", default=str(REPORTS / "source_live_reconcile.json"))
    parser.add_argument("--out-md", default=str(REPORTS / "source_live_reconcile.md"))
    args = parser.parse_args()
    rows = reconcile()
    Path(args.out_json).write_text(json.dumps(rows, indent=2, sort_keys=True, default=str))
    write_markdown(rows, Path(args.out_md))
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    print(f"mismatches: {len(rows)}")


if __name__ == "__main__":
    main()
