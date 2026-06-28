#!/usr/bin/env python3
"""Match reusable insights against the global layer inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"


def load_registry(path: Path) -> list[dict]:
    # The registry is JSON-compatible YAML to avoid adding a PyYAML dependency.
    return json.loads(path.read_text())


def any_intersects(values: list[str], target: set[str]) -> bool:
    return bool(set(values or []).intersection(target))


def all_in(values: list[str], target: set[str]) -> bool:
    return set(values or []).issubset(target)


def matches(task: dict, insight: dict) -> bool:
    if insight.get("status") != "active":
        return False
    if not task.get("source_has_build", False):
        return False
    if task.get("source_class") in {"no_build", "no_source"}:
        return False
    pred = insight.get("applies_when") or {}
    reject = insight.get("reject_when") or {}
    ops = set(task.get("ops", {}).keys())
    tags = set(task.get("tags", []))
    memory = int(task.get("memory") or 0)
    points = float(task.get("points") or 0.0)

    if pred.get("any_ops") and not any_intersects(pred["any_ops"], ops):
        return False
    if pred.get("all_ops") and not all_in(pred["all_ops"], ops):
        return False
    if pred.get("any_tags") and not any_intersects(pred["any_tags"], tags):
        return False
    if pred.get("all_tags") and not all_in(pred["all_tags"], tags):
        return False
    if pred.get("min_memory") is not None and memory < int(pred["min_memory"]):
        return False
    if pred.get("max_points") is not None and points > float(pred["max_points"]):
        return False

    if reject.get("any_ops") and any_intersects(reject["any_ops"], ops):
        return False
    if reject.get("any_tags") and any_intersects(reject["any_tags"], tags):
        return False
    return True


def candidate_score(task: dict, insight: dict) -> float:
    memory = int(task.get("memory") or 0)
    params = int(task.get("params") or 0)
    points = float(task.get("points") or 0.0)
    source_penalty = 0 if task.get("source_class") != "exact_preserve" else -500
    wall_penalty = -1500 if "documented_wall" in task.get("tags", []) else 0
    marginal_penalty = -40000 if "marginal_wall" in task.get("tags", []) else 0
    exact_wall_penalty = -100000 if (
        "information_loss_wall" in task.get("tags", [])
        or "ambiguity_wall" in task.get("tags", [])
        or "infeasible_exact_wall" in task.get("tags", [])
    ) else 0
    low_score_bonus = max(0.0, 16.0 - points) * 300
    return memory + params + low_score_bonus + source_penalty + wall_penalty + marginal_penalty + exact_wall_penalty


def build_applications(inventory: dict, registry: list[dict]) -> list[dict]:
    rows = []
    for insight in registry:
        matched = []
        for task in inventory["tasks"]:
            if matches(task, insight):
                matched.append(
                    {
                        "task": task["task"],
                        "score": round(candidate_score(task, insight), 3),
                        "points": task.get("points"),
                        "memory": task.get("memory"),
                        "params": task.get("params"),
                        "method": task.get("method"),
                        "source_class": task.get("source_class"),
                        "tags": task.get("tags", []),
                    }
                )
        matched.sort(key=lambda x: (-x["score"], x["task"]))
        rows.append(
            {
                "insight_id": insight["id"],
                "title": insight.get("title"),
                "source_tasks": insight.get("source_tasks", []),
                "expected": insight.get("expected", {}),
                "candidate_count": len(matched),
                "candidates": matched,
            }
        )
    return rows


def write_queue(applications: list[dict], path: Path, limit: int) -> None:
    lines = [
        "# Recursive improvement queue",
        "",
        "Generated from `reports/global_layer_inventory.json` and `reports/insight_registry.yaml`.",
        "",
        "Use this as a global queue.  When a deep task attempt creates a new mechanism, add it to the registry and regenerate this file.",
        "",
    ]
    for app in applications:
        lines.extend(
            [
                f"## {app['insight_id']}",
                "",
                app.get("title") or "",
                "",
                f"- source tasks: {' '.join(f'{t:03d}' for t in app.get('source_tasks', [])) or 'none'}",
                f"- candidates: {app['candidate_count']}",
                f"- expected: {app.get('expected', {})}",
                "",
                "| rank | task | score | pts | mem | params | source | tags |",
                "|---:|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for rank, cand in enumerate(app["candidates"][:limit], 1):
            tags = ",".join(cand.get("tags", [])[:8])
            lines.append(
                f"| {rank} | {cand['task']:03d} | {cand['score']:.1f} | "
                f"{float(cand['points'] or 0):.3f} | {cand['memory']} | {cand['params']} | "
                f"{cand['source_class']} | {tags} |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default=str(REPORTS / "global_layer_inventory.json"))
    parser.add_argument("--registry", default=str(REPORTS / "insight_registry.yaml"))
    parser.add_argument("--out-json", default=str(REPORTS / "insight_applications.json"))
    parser.add_argument("--out-md", default=str(REPORTS / "recursive_queue.md"))
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()
    inventory = json.loads(Path(args.inventory).read_text())
    registry = load_registry(Path(args.registry))
    apps = build_applications(inventory, registry)
    Path(args.out_json).write_text(json.dumps(apps, indent=2, sort_keys=True))
    write_queue(apps, Path(args.out_md), args.limit)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    for app in apps:
        print(f"{app['insight_id']}: {app['candidate_count']} candidates")


if __name__ == "__main__":
    main()
