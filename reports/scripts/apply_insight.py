#!/usr/bin/env python3
"""Safe entrypoint for applying one insight to one task.

This intentionally starts as a guarded skeleton.  Insight-specific transformers
must be added explicitly; until then the command verifies the current source
builder and prints the next required manual action.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from src.harness import evaluate, load_task


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"


def load_registry() -> dict[str, dict]:
    rows = json.loads((REPORTS / "insight_registry.yaml").read_text())
    return {row["id"]: row for row in rows}


def verify_source(task: int) -> dict:
    mod = importlib.import_module(f"src.custom.task{task:03d}")
    return evaluate(mod.build(None), load_task(task), keep_failures=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("insight_id")
    parser.add_argument("task", type=int)
    parser.add_argument("--execute", action="store_true", help="reserved for future transformer execution")
    args = parser.parse_args()

    registry = load_registry()
    if args.insight_id not in registry:
        raise SystemExit(f"unknown insight: {args.insight_id}")
    insight = registry[args.insight_id]
    ev = verify_source(args.task)
    print(json.dumps({"task": args.task, "source_eval": ev}, indent=2, default=str))

    transformer = insight.get("transformer")
    if not transformer:
        print("No automated transformer registered for this insight yet.")
        print("Next step: inspect the task source and incumbent ONNX, implement a candidate builder, then register the reusable mechanism.")
        return
    if not args.execute:
        print(f"Transformer registered but not executed: {transformer}")
        print("Rerun with --execute after reviewing transformer safety.")
        return
    raise SystemExit("transformer execution is not implemented yet")


if __name__ == "__main__":
    main()
