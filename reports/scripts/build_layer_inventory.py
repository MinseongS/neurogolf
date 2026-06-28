#!/usr/bin/env python3
"""Build a global task/layer inventory for recursive NeuroGolf improvement."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import onnx


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
NETWORKS = ROOT / "networks"
CUSTOM = ROOT / "src" / "custom"


def load_manifest() -> dict:
    path = REPORTS / "manifest.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return data.get("tasks", data)


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except FileNotFoundError:
        return ""


def source_has_build(text: str) -> bool:
    return bool(
        re.search(r"^def\s+build\s*\(", text, flags=re.M)
        or re.search(r"^from\s+\.[\w_]+\s+import\s+build\b", text, flags=re.M)
        or re.search(r"^from\s+\.[\w_]+\s+import\s+.*\bbuild\b", text, flags=re.M)
    )


def op_counts(model: onnx.ModelProto) -> Counter:
    return Counter(node.op_type for node in model.graph.node)


def initializer_summary(model: onnx.ModelProto) -> dict:
    dtype_counts = Counter()
    total = 0
    largest = []
    for init in model.graph.initializer:
        elems = math.prod(init.dims) if init.dims else 1
        total += elems
        dtype_counts[str(init.data_type)] += elems
        largest.append({"name": init.name, "elems": elems, "dtype": init.data_type, "shape": list(init.dims)})
    largest.sort(key=lambda x: x["elems"], reverse=True)
    return {"total_elems": total, "dtype_elem_counts": dict(dtype_counts), "largest": largest[:8]}


def source_class(text: str) -> str:
    if not text:
        return "no_source"
    if not source_has_build(text):
        return "no_build"
    lower = text.lower()
    if "exact source reconstruction" in lower or "preserves the verified live behavior" in lower:
        return "exact_preserve"
    if "heuristic" in lower:
        return "heuristic"
    if "source-owned" in lower or "semantic" in lower or "task " in lower:
        return "semantic_or_handbuilt"
    return "unknown"


def has_actionable_open_angle(tasklog_text: str) -> bool:
    lower = tasklog_text.lower()
    if "untried" in lower or "re-attack" in lower or "reprobe" in lower:
        return True
    match = re.search(r"^##\s+open angles?.*?(?=^##\s+|\Z)", lower, flags=re.M | re.S)
    if not match:
        return False
    section = match.group(0)
    closed_markers = (
        "none",
        "no credible",
        "no remaining",
        "exhausted",
        "not worth",
        "not currently actionable",
    )
    return bool(section.strip()) and not any(marker in section[:700] for marker in closed_markers)


def tag_task(ops: Counter, manifest_row: dict, source_text: str, tasklog_text: str) -> list[str]:
    tags: set[str] = set()
    node_count = sum(ops.values())
    memory = int(manifest_row.get("memory") or 0)
    points = float(manifest_row.get("points") or 0.0)
    lower_src = source_text.lower()
    lower_log = tasklog_text.lower()

    src_class = source_class(source_text)
    if src_class == "no_build":
        tags.add("no_build")
    if src_class == "no_source":
        tags.add("no_source")
    if src_class == "exact_preserve":
        tags.add("exact_preserve")
    if "heuristic" in lower_src:
        tags.add("heuristic")
    if node_count == 1 and (ops.get("Conv") or ops.get("Gather")) and memory == 0:
        tags.add("mem0_single_op")
    if ops.get("Conv", 0) + ops.get("QLinearConv", 0) >= 2:
        tags.add("conv_heavy")
    if ops.get("Conv", 0) + ops.get("QLinearConv", 0) >= 1 and node_count <= 60:
        tags.add("local_stencil")
    if ops.get("QLinearConv") or ops.get("QLinearMatMul") or "qlinear" in lower_src or "qlinear" in lower_log:
        tags.add("qlinear")
    if ops.get("MatMul") or ops.get("MatMulInteger"):
        tags.add("matmul")
    if "lut" in lower_src or "lut" in lower_log or "lookup" in lower_src or "lookup" in lower_log:
        tags.add("lut_selection")
    if "template-match" in lower_src or "template match" in lower_src or "dihedral" in lower_src:
        tags.add("template_match")
    if "beats deployed" in lower_log or "adopt-candidate" in lower_log or "new custom" in lower_log:
        tags.add("custom_win")
    if ops.get("Equal") and ("channel_values" in source_text or "onehot" in lower_src):
        tags.add("onehot_final_equal")
    if ops.get("CumSum") or ops.get("ReduceMax", 0) + ops.get("ReduceMin", 0) >= 10:
        tags.add("scan")
    if ops.get("MaxPool", 0) >= 5:
        tags.add("maxpool_scan")
    if ops.get("Gather", 0) + ops.get("GatherND", 0) + ops.get("GatherElements", 0) >= 10:
        tags.add("gather_heavy")
    if ops.get("ScatterND") or ops.get("ScatterElements"):
        tags.add("scatter")
    if ops.get("BitShift") or ops.get("BitwiseAnd") or ops.get("BitwiseOr"):
        tags.add("bitwise_program")
    if "assignment" in lower_log or "correspondence" in lower_log:
        tags.add("assignment_wall")
    if "connect" in lower_log or "flood" in lower_log:
        tags.add("connectivity_wall")
    if "wall" in lower_log or "infeasible" in lower_log or "underdetermined" in lower_log:
        tags.add("documented_wall")
    if (
        "marginal" in lower_log
        and ("did not beat" in lower_log or "no beating net" in lower_log or "cannot beat" in lower_log)
    ):
        tags.add("marginal_wall")
    if "information loss" in lower_log or "information-theoretically" in lower_log:
        tags.add("information_loss_wall")
    if "infeasible for an exact" in lower_log or "infeasible for exact" in lower_log:
        tags.add("infeasible_exact_wall")
    if has_actionable_open_angle(tasklog_text):
        tags.add("open_angle")
    if memory >= 10000:
        tags.add("high_memory")
    if points < 15.0:
        tags.add("low_score")
    return sorted(tags)


def build_inventory() -> dict:
    manifest = load_manifest()
    tasks = []
    by_op: dict[str, list[int]] = defaultdict(list)
    by_tag: dict[str, list[int]] = defaultdict(list)
    failures = []

    for task in range(1, 401):
        path = NETWORKS / f"task{task:03d}.onnx"
        manifest_row = manifest.get(str(task), {})
        try:
            model = onnx.load(path)
        except Exception as exc:
            failures.append({"task": task, "error": str(exc)})
            continue

        ops = op_counts(model)
        source_text = read_text(CUSTOM / f"task{task:03d}.py")
        tasklog_text = read_text(REPORTS / "tasklog" / f"task{task:03d}.md")
        tags = tag_task(ops, manifest_row, source_text, tasklog_text)
        for op in ops:
            by_op[op].append(task)
        for tag in tags:
            by_tag[tag].append(task)

        tasks.append(
            {
                "task": task,
                "points": manifest_row.get("points"),
                "memory": manifest_row.get("memory"),
                "params": manifest_row.get("params"),
                "method": manifest_row.get("method"),
                "node_count": sum(ops.values()),
                "ops": dict(sorted(ops.items())),
                "op_sequence": [node.op_type for node in model.graph.node[:80]],
                "opset": [{"domain": op.domain, "version": op.version} for op in model.opset_import],
                "initializer_summary": initializer_summary(model),
                "source_class": source_class(source_text),
                "source_has_build": source_has_build(source_text),
                "tags": tags,
                "tasklog_exists": bool(tasklog_text),
            }
        )

    return {
        "version": 1,
        "task_count": len(tasks),
        "failures": failures,
        "tasks": tasks,
        "by_op": {k: v for k, v in sorted(by_op.items())},
        "by_tag": {k: v for k, v in sorted(by_tag.items())},
    }


def write_markdown(inv: dict, path: Path) -> None:
    source_rows = Counter(task["source_class"] for task in inv["tasks"])
    build_count = sum(1 for task in inv["tasks"] if task.get("source_has_build"))
    no_build = [task["task"] for task in inv["tasks"] if not task.get("source_has_build")]
    lines = [
        "# Global layer inventory summary",
        "",
        f"- tasks indexed: {inv['task_count']}",
        f"- failures: {len(inv['failures'])}",
        f"- source-controlled build(): {build_count}/{inv['task_count']}",
        f"- no build(): {' '.join(f'{t:03d}' for t in no_build) or 'none'}",
        "",
        "## Source classes",
        "",
    ]
    for cls, count in sorted(source_rows.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- `{cls}`: {count}")
    lines.extend([
        "",
        "## Largest op families",
        "",
    ])
    op_rows = sorted(((op, len(tasks)) for op, tasks in inv["by_op"].items()), key=lambda x: (-x[1], x[0]))
    for op, count in op_rows[:25]:
        lines.append(f"- `{op}`: {count} tasks")
    lines.extend(["", "## Key tags", ""])
    for tag, tasks in sorted(inv["by_tag"].items(), key=lambda x: (-len(x[1]), x[0]))[:30]:
        sample = " ".join(f"{t:03d}" for t in tasks[:20])
        lines.append(f"- `{tag}`: {len(tasks)} tasks — {sample}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPORTS / "global_layer_inventory.json"))
    parser.add_argument("--summary", default=str(REPORTS / "global_layer_inventory.md"))
    args = parser.parse_args()
    inv = build_inventory()
    Path(args.out).write_text(json.dumps(inv, indent=2, sort_keys=True))
    write_markdown(inv, Path(args.summary))
    print(f"wrote {args.out} ({inv['task_count']} tasks)")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
