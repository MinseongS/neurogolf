"""Rebuild deployed ONNX artifacts from source-owned task builders.

This repository treats `src/custom/taskNNN.py` as the source of truth.
`networks/taskNNN.onnx` files are local build/deploy artifacts and are ignored by git.

Usage:
  PYTHONPATH=. .venv/bin/python reports/scripts/rebuild_networks_from_source.py
  PYTHONPATH=. .venv/bin/python reports/scripts/rebuild_networks_from_source.py --tasks 1-10
  PYTHONPATH=. .venv/bin/python reports/scripts/rebuild_networks_from_source.py --tasks 17,92,233
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import onnx

from src.harness import ROOT, evaluate, load_task

NETWORKS = ROOT / "networks"
MANIFEST = ROOT / "reports" / "manifest.json"
N_TASKS = 400


def parse_tasks(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def load_manifest() -> dict[str, object]:
    if not MANIFEST.exists():
        return {}
    return json.load(open(MANIFEST))["tasks"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=f"1-{N_TASKS}")
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()

    NETWORKS.mkdir(exist_ok=True)
    manifest = load_manifest()
    rebuilt = failed = 0

    for task_num in parse_tasks(args.tasks):
        task = load_task(task_num)
        mod = importlib.import_module(f"src.custom.task{task_num:03d}")
        importlib.reload(mod)
        model = mod.build(task)
        if model is None:
            print(f"task{task_num:03d}: no source model")
            failed += 1
            continue
        ev = evaluate(model, task)
        if not ev["ok"]:
            print(
                f"task{task_num:03d}: FAIL stored eval "
                f"fail={ev.get('fail')} err={ev.get('error')}"
            )
            failed += 1
            continue
        out = NETWORKS / f"task{task_num:03d}.onnx"
        onnx.save(model, out)
        rebuilt += 1
        if args.write_manifest:
            manifest[str(task_num)] = {
                "points": ev["points"],
                "memory": ev["memory"],
                "params": ev["params"],
                "method": f"custom:task{task_num:03d}",
            }
        print(
            f"task{task_num:03d}: wrote {out} "
            f"pts={ev['points']:.4f} mem={ev['memory']} params={ev['params']}"
        )

    if args.write_manifest:
        MANIFEST.write_text(json.dumps({"tasks": manifest}, indent=1) + "\n")
        print(f"updated {MANIFEST}")

    print(f"rebuilt={rebuilt} failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
