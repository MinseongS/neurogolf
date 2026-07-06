#!/usr/bin/env python3
"""Per-(mechanism, task) outcome ledger. The matching engine reads resolved_tasks() to
exclude known floors/failures so an insight is never re-probed on dead tasks."""
from __future__ import annotations

import json
from pathlib import Path

REPORTS = Path(__file__).resolve().parents[1]
PATH = REPORTS / "mechanism_coverage.json"

SEED = {
    # NOTE: task numbers are UNPADDED strings ("77" not "077") to match the key format used by
    # reports/task_index.json (built from manifest.json's unpadded keys) and insight_registry.yaml
    # source_tasks. task-6 backfill validation caught "077"/"092" here as stale zero-padded typos
    # that could never match task_index.json rows -> fixed (same tasks, corrected representation).
    "walk_einsum": ["187", "110", "243", "77"],
    "einsum_vs_free_input": ["255", "208", "370"],
    "signed_rect": ["234", "335", "92"],
    "gridsample_warp": ["209"],
    "qlinearconv_render": ["133"],
    "topk_width": ["173"],
}


def load() -> dict:
    return json.loads(PATH.read_text()) if PATH.exists() else {}


def save(data: dict) -> None:
    PATH.write_text(json.dumps(data, indent=1))


def record(mechanism: str, task: str, outcome: str, **fields) -> None:
    data = load()
    mech = data.setdefault(mechanism, {"query": "", "tasks": {}})
    entry = mech["tasks"].setdefault(task, {})
    entry["outcome"] = outcome
    entry.update(fields)
    save(data)


def resolved_tasks(mechanism: str) -> set:
    data = load()
    tasks = data.get(mechanism, {}).get("tasks", {})
    return {t for t, e in tasks.items() if e.get("outcome") in ("applied", "floor", "failed")}


def seed() -> None:
    for mech, tasks in SEED.items():
        for t in tasks:
            if t not in resolved_tasks(mech):
                record(mech, t, "applied", session="seed")


if __name__ == "__main__":
    seed()
    print(f"seeded -> {PATH}")
