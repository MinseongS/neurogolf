#!/usr/bin/env python3
"""Search active nets for high-leverage representation changes.

This report is intentionally speculative.  It ranks places where a counted
single-channel label/mask carrier exists immediately before the free graph
output, because those are the only locations where a new compiler pass can
delete full-canvas memory rather than trim bytes.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import onnx


ROOT = Path(__file__).resolve().parents[2]
NETS = ROOT / "submission" / "overfit_nets"
MANIFEST = ROOT / "reports" / "overfit_manifest.json"
OUT_JSON = ROOT / "reports" / "new_representation_scan.json"
OUT_MD = ROOT / "reports" / "new_representation_scan.md"

DTYPE_SIZE = {1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8}
DTYPE_NAME = {1: "fp32", 2: "u8", 3: "i8", 4: "u16", 5: "i16", 6: "i32", 7: "i64", 9: "bool", 10: "fp16", 11: "fp64"}


def shape_dtype_map(model: onnx.ModelProto) -> dict[str, dict[str, Any]]:
    try:
        graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    except Exception:
        graph = model.graph
    out = {}
    for item in list(graph.input) + list(graph.value_info) + list(graph.output):
        if not item.type.HasField("tensor_type"):
            continue
        tt = item.type.tensor_type
        if not tt.HasField("shape"):
            continue
        dims = []
        ok = True
        for dim in tt.shape.dim:
            if not dim.HasField("dim_value") or dim.dim_value <= 0:
                ok = False
                break
            dims.append(dim.dim_value)
        if not ok:
            continue
        elems = math.prod(dims)
        out[item.name] = {
            "shape": dims,
            "dtype": DTYPE_NAME.get(tt.elem_type, str(tt.elem_type)),
            "bytes": elems * DTYPE_SIZE.get(tt.elem_type, 4),
            "elems": elems,
        }
    return out


def maps(model: onnx.ModelProto) -> tuple[dict[str, onnx.NodeProto], dict[str, list[onnx.NodeProto]]]:
    producers = {}
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for out in node.output:
            producers[out] = node
        for inp in node.input:
            consumers[inp].append(node)
    return producers, consumers


def node_label(node: onnx.NodeProto | None) -> str:
    if node is None:
        return ""
    return f"{node.op_type}:{node.name or (node.output[0] if node.output else '')}"


def op_counts(model: onnx.ModelProto) -> dict[str, int]:
    return dict(sorted(Counter(node.op_type for node in model.graph.node).items()))


def candidate_rows(task: int, row: dict[str, Any]) -> list[dict[str, Any]]:
    model = onnx.load(NETS / f"task{task:03d}.onnx")
    tmap = shape_dtype_map(model)
    producers, consumers = maps(model)
    ops = op_counts(model)
    rows = []
    output_node = next((node for node in model.graph.node if "output" in node.output), None)
    if output_node is None:
        return rows
    output_inputs = list(output_node.input)
    for inp in output_inputs:
        if inp in {"input"}:
            continue
        meta = tmap.get(inp)
        if not meta:
            continue
        prod = producers.get(inp)
        cons = consumers.get(inp, [])
        if meta["bytes"] < 500:
            continue
        kind = None
        if output_node.op_type == "Equal" and len(meta["shape"]) == 4 and meta["shape"][1] == 1:
            kind = "final_equal_label_plane"
        elif output_node.op_type == "Where" and meta["dtype"] == "bool":
            kind = "final_where_mask_plane"
        elif output_node.op_type in {"Pad", "Gather", "Einsum", "Conv", "QLinearConv"}:
            kind = "direct_output_op"
        if not kind:
            continue
        consumers_without_output = [c for c in cons if c is not output_node]
        rows.append({
            "task": task,
            "points": row["points"],
            "cost": row["cost"],
            "memory": row["memory"],
            "params": row["params"],
            "kind": kind,
            "output_op": output_node.op_type,
            "tensor": inp,
            "tensor_bytes": meta["bytes"],
            "shape": meta["shape"],
            "dtype": meta["dtype"],
            "producer": node_label(prod),
            "extra_consumers": [node_label(c) for c in consumers_without_output],
            "ops": ops,
            "hypothesis": hypothesis(kind, prod, consumers_without_output, meta, ops),
        })
    return rows


def hypothesis(kind: str, prod: onnx.NodeProto | None, extra: list[onnx.NodeProto], meta: dict[str, Any], ops: dict[str, int]) -> str:
    pop = prod.op_type if prod is not None else ""
    if kind == "final_equal_label_plane":
        if not extra and pop in {"Gather", "Pad", "Where", "Max", "Min", "Transpose"}:
            return "try direct one-hot output by fusing label producer with Equal; viable only if the producer can emit 10 channels without materialized 10x carrier"
        if pop in {"Conv", "QLinearConv"}:
            return "try producer out_channels=10 directly into output, or label-free signed scores; high risk but can delete label plane"
        return "label carrier before free Equal; needs semantic direct-output rewrite"
    if kind == "final_where_mask_plane":
        if pop == "Pad":
            return "try output-sized mask avoidance: move Pad into color branch or shrink active canvas with free-input overlay"
        if pop in {"Greater", "Equal", "And", "Or"}:
            return "try predicate fusion into final Where or replace mask with signed color scores"
        return "mask carrier before free Where; look for boolean predicate fusion"
    return "already direct output; inspect whether previous carrier can be renamed into output or deleted"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    rows = []
    for row in manifest["tasks"]:
        rows.extend(candidate_rows(int(row["task"]), row))
    rows.sort(key=lambda x: (x["tensor_bytes"], x["cost"]), reverse=True)
    payload = {"rows": rows}
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True))
    lines = [
        "# New Representation Scan",
        "",
        "Speculative candidates where a counted label/mask carrier feeds the free graph output.",
        "",
        "| rank | task | kind | pts | cost | tensor | bytes | dtype | shape | producer | hypothesis |",
        "|---:|---:|---|---:|---:|---|---:|---|---|---|---|",
    ]
    for idx, r in enumerate(rows[:100], 1):
        lines.append(
            f"| {idx} | {r['task']:03d} | {r['kind']} | {r['points']:.3f} | {r['cost']} | "
            f"{r['tensor']} | {r['tensor_bytes']} | {r['dtype']} | {r['shape']} | {r['producer']} | {r['hypothesis']} |"
        )
    lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    print(f"candidates={len(rows)}")


if __name__ == "__main__":
    main()
