#!/usr/bin/env python3
"""Compare high-score frontier mechanisms against active high-cost tasks.

This is a research report, not a gate.  It inspects the active overfit ONNX
set and summarizes which cheap tasks avoid counted full-canvas carriers, then
lists high-cost tasks with similar op families that may admit a rewrite.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
NETS = ROOT / "submission" / "overfit_nets"
MANIFEST = REPORTS / "overfit_manifest.json"
OUT_JSON = REPORTS / "frontier_mechanism_audit.json"
OUT_MD = REPORTS / "frontier_mechanism_audit.md"


DTYPE_NAMES = {
    1: "float32",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    9: "bool",
    10: "float16",
    11: "float64",
}


def elem_size(elem_type: int) -> int:
    sizes = {
        1: 4,
        2: 1,
        3: 1,
        4: 2,
        5: 2,
        6: 4,
        7: 8,
        9: 1,
        10: 2,
        11: 8,
    }
    return sizes.get(elem_type, 4)


def tensor_shape_and_type(value_info: onnx.ValueInfoProto) -> tuple[list[int] | None, int | None]:
    if not value_info.type.HasField("tensor_type"):
        return None, None
    tt = value_info.type.tensor_type
    if not tt.HasField("shape"):
        return None, tt.elem_type
    dims: list[int] = []
    for dim in tt.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            return None, tt.elem_type
        dims.append(dim.dim_value)
    return dims, tt.elem_type


def infer_tensor_map(model: onnx.ModelProto) -> dict[str, dict[str, Any]]:
    try:
        graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    except Exception:
        graph = model.graph
    out: dict[str, dict[str, Any]] = {}
    for item in list(graph.input) + list(graph.value_info) + list(graph.output):
        shape, elem_type = tensor_shape_and_type(item)
        if shape is None or elem_type is None:
            continue
        elems = math.prod(shape)
        out[item.name] = {
            "shape": shape,
            "dtype": DTYPE_NAMES.get(elem_type, str(elem_type)),
            "bytes": elems * elem_size(elem_type),
            "elems": elems,
        }
    return out


def initializer_summary(model: onnx.ModelProto) -> dict[str, Any]:
    largest = []
    dtype_counts: Counter[str] = Counter()
    total = 0
    for init in model.graph.initializer:
        elems = math.prod(init.dims) if init.dims else 1
        total += elems
        dtype_counts[DTYPE_NAMES.get(init.data_type, str(init.data_type))] += elems
        largest.append({"name": init.name, "shape": list(init.dims), "dtype": DTYPE_NAMES.get(init.data_type, str(init.data_type)), "elems": elems})
    largest.sort(key=lambda row: row["elems"], reverse=True)
    return {"total_elems": total, "dtype_elem_counts": dict(dtype_counts), "largest": largest[:10]}


def producer_maps(model: onnx.ModelProto) -> tuple[dict[str, str], dict[str, list[str]]]:
    producer: dict[str, str] = {}
    consumers: dict[str, list[str]] = defaultdict(list)
    for node in model.graph.node:
        label = node.name or (node.output[0] if node.output else node.op_type)
        for output in node.output:
            if output:
                producer[output] = f"{node.op_type}:{label}"
        for inp in node.input:
            if inp:
                consumers[inp].append(node.op_type)
    return producer, consumers


def classify(summary: dict[str, Any]) -> list[str]:
    ops = Counter(summary["ops"])
    tags = []
    if summary["memory"] == 0:
        tags.append("mem0")
    if summary["memory"] <= 140:
        tags.append("frontier_memory")
    if ops.get("Gather", 0) and summary["memory"] == 0:
        tags.append("direct_gather")
    if ops.get("Conv", 0) + ops.get("QLinearConv", 0) and summary["memory"] == 0:
        tags.append("single_op_stencil")
    if ops.get("Where", 0) and "input" in summary["output_inputs"]:
        tags.append("free_input_overlay")
    if ops.get("Equal", 0) and summary["memory"] <= 200:
        tags.append("final_equal")
    if ops.get("Einsum", 0) and summary["memory"] == 0:
        tags.append("einsum_direct_output")
    if ops.get("ScatterND", 0) or ops.get("ScatterElements", 0):
        tags.append("scatter")
    if ops.get("TopK", 0):
        tags.append("topk")
    return tags


def summarize_task(task: int, row: dict[str, Any]) -> dict[str, Any]:
    path = NETS / f"task{task:03d}.onnx"
    model = onnx.load(path)
    tensor_map = infer_tensor_map(model)
    producer, consumers = producer_maps(model)
    ops = Counter(node.op_type for node in model.graph.node)
    output_node = next((node for node in model.graph.node if "output" in node.output), None)
    counted = []
    for name, meta in tensor_map.items():
        if name in {"input", "output"}:
            continue
        counted.append({
            "name": name,
            "bytes": meta["bytes"],
            "shape": meta["shape"],
            "dtype": meta["dtype"],
            "producer": producer.get(name, ""),
            "consumers": consumers.get(name, [])[:6],
        })
    counted.sort(key=lambda item: item["bytes"], reverse=True)
    summary = {
        "task": task,
        "points": row.get("points"),
        "memory": row.get("memory"),
        "params": row.get("params"),
        "cost": row.get("cost"),
        "nodes": len(model.graph.node),
        "ops": dict(sorted(ops.items())),
        "output_op": output_node.op_type if output_node is not None else None,
        "output_inputs": list(output_node.input) if output_node is not None else [],
        "largest_counted_tensors": counted[:12],
        "initializers": initializer_summary(model),
    }
    summary["mechanism_tags"] = classify(summary)
    return summary


def load_manifest_rows() -> list[dict[str, Any]]:
    data = json.loads(MANIFEST.read_text())
    return data["tasks"]


def write_md(payload: dict[str, Any]) -> None:
    lines = [
        "# Frontier Mechanism Audit",
        "",
        "Compares current high-score/tiny-cost active ONNX mechanisms against high-cost active tasks.",
        "",
        "## Frontier Tasks",
        "",
        "| task | pts | mem | params | cost | nodes | output | tags | top ops |",
        "|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in payload["frontier"]:
        ops = ", ".join(f"{k}:{v}" for k, v in sorted(row["ops"].items(), key=lambda x: (-x[1], x[0]))[:5])
        tags = ", ".join(row["mechanism_tags"])
        lines.append(f"| {row['task']:03d} | {row['points']:.3f} | {row['memory']} | {row['params']} | {row['cost']} | {row['nodes']} | {row['output_op']} | {tags} | {ops} |")
    lines.extend([
        "",
        "## High-Cost Tasks",
        "",
        "| task | pts | mem | params | cost | nodes | output | tags | largest counted tensors |",
        "|---:|---:|---:|---:|---:|---:|---|---|---|",
    ])
    for row in payload["high_cost"]:
        largest = "; ".join(
            f"{t['bytes']}B {t['dtype']} {t['shape']} {t['producer'].split(':', 1)[0]}"
            for t in row["largest_counted_tensors"][:4]
        )
        tags = ", ".join(row["mechanism_tags"])
        lines.append(f"| {row['task']:03d} | {row['points']:.3f} | {row['memory']} | {row['params']} | {row['cost']} | {row['nodes']} | {row['output_op']} | {tags} | {largest} |")
    lines.extend([
        "",
        "## Rewrite Leads",
        "",
    ])
    for lead in payload["rewrite_leads"]:
        lines.append(f"- task{lead['task']:03d}: {lead['why']} Candidate mechanism: {lead['candidate_mechanism']}.")
    lines.append("")
    OUT_MD.write_text("\n".join(lines))


def build_leads(high_cost: list[dict[str, Any]]) -> list[dict[str, Any]]:
    leads = []
    for row in high_cost:
        ops = Counter(row["ops"])
        largest = row["largest_counted_tensors"][:4]
        full_canvas = [t for t in largest if t["bytes"] >= 900]
        if row["task"] in {349, 138, 74, 80, 370} and ops.get("QLinearConv", 0) + ops.get("Conv", 0):
            leads.append({
                "task": row["task"],
                "why": "local-stencil family with expensive counted canvas carriers; recursive queue already maps it to sparse_conv_single_op_floor.",
                "candidate_mechanism": "replace staged detector planes with one direct-output QLinearConv/Conv or final Where(input overlay).",
            })
        elif row["task"] in {101, 25, 64, 219} and (ops.get("Equal", 0) or ops.get("Where", 0)):
            leads.append({
                "task": row["task"],
                "why": "one-hot/final-equal family; expensive label or mask planes may be delayed into graph output.",
                "candidate_mechanism": "free_final_onehot_equal or free-input overlay.",
            })
        elif row["task"] == 54 and (ops.get("ScatterND", 0) or ops.get("ScatterElements", 0)):
            leads.append({
                "task": row["task"],
                "why": "scatter-heavy edit stream; active net pays for many full-canvas edit masks.",
                "candidate_mechanism": "sparse edit stream only if inactive duplicate writes are proven harmless.",
            })
        elif full_canvas:
            leads.append({
                "task": row["task"],
                "why": "largest tensors are counted full-canvas carriers, unlike the 20+ frontier.",
                "candidate_mechanism": "semantic direct-output rewrite; graph surgery unlikely to be enough.",
            })
    return leads[:20]


def main() -> None:
    rows = load_manifest_rows()
    by_task = {int(row["task"]): row for row in rows}
    frontier_tasks = [int(row["task"]) for row in sorted(rows, key=lambda x: (x["cost"], -x["points"]))[:50]]
    high_cost_tasks = [int(row["task"]) for row in sorted(rows, key=lambda x: x["cost"], reverse=True)[:40]]
    frontier = [summarize_task(task, by_task[task]) for task in frontier_tasks]
    high_cost = [summarize_task(task, by_task[task]) for task in high_cost_tasks]
    payload = {
        "frontier": frontier,
        "high_cost": high_cost,
        "rewrite_leads": build_leads(high_cost),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_md(payload)
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
