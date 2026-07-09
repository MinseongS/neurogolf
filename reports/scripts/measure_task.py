"""Measure a custom task's mem/params/points from its Python builder.
Usage: uv run python -m reports.scripts.measure_task TASKNUM
"""
import sys, importlib
from src.harness import load_task, evaluate

TASK = int(sys.argv[1])
mod = importlib.import_module(f"src.custom.task{TASK:03d}")
task = load_task(TASK)
res = evaluate(mod.build(task), task)
print({k: res[k] for k in ("ok", "pass", "fail", "memory", "params", "points", "error")})
