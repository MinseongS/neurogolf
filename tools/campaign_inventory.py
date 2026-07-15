from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from google.protobuf.message import DecodeError


PROTECTED_TASKS = {118, 131, 209}


def point_gain(cost: int, saving: int) -> float:
    if cost <= 1 or saving <= 0 or saving >= cost:
        return 0.0
    return math.log(cost / (cost - saving))


def _shape_elements(value_info: onnx.ValueInfoProto) -> int | None:
    tensor = value_info.type.tensor_type
    if not tensor.HasField("shape"):
        return None
    dims = [d.dim_value for d in tensor.shape.dim]
    if not dims or any(d <= 0 for d in dims):
        return None
    return math.prod(dims)


def analyze_model(task: int, model: onnx.ModelProto, manifest_row: dict[str, Any]) -> dict[str, Any]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    initializers = {item.name for item in inferred.graph.initializer}
    producers = {out: node.op_type for node in inferred.graph.node for out in node.output if out}
    tensors = []
    for item in inferred.graph.value_info:
        if item.name in initializers or item.name in {"input", "output"}:
            continue
        elements = _shape_elements(item)
        if elements is None:
            continue
        dtype = item.type.tensor_type.elem_type
        saving = elements * np.dtype(onnx.helper.tensor_dtype_to_np_dtype(dtype)).itemsize
        tensors.append({"name": item.name, "saving": saving, "producer": producers.get(item.name, "")})
    banks = [
        {"name": item.name, "saving": math.prod(item.dims)}
        for item in inferred.graph.initializer
        if item.dims and all(d > 0 for d in item.dims)
    ]
    cost = int(manifest_row["cost"])
    tensors.sort(key=lambda row: (-row["saving"], row["name"]))
    banks.sort(key=lambda row: (-row["saving"], row["name"]))
    largest_tensor = tensors[0] if tensors else {"name": "", "saving": 0, "producer": ""}
    largest_bank = banks[0] if banks else {"name": "", "saving": 0}
    return {
        "task": task,
        "cost": cost,
        "points": float(manifest_row["points"]),
        "sha256": str(manifest_row["sha256"]),
        "largest_tensor": {**largest_tensor, "expected_gain": point_gain(cost, largest_tensor["saving"])},
        "largest_param_bank": {**largest_bank, "expected_gain": point_gain(cost, largest_bank["saving"])},
    }


def risk_flags(task: int, model: onnx.ModelProto) -> list[str]:
    flags = []
    if task in PROTECTED_TASKS:
        flags.append("protected-public-zero-repair")
    if any(node.op_type == "Einsum" and len(node.input) >= 12 for node in model.graph.node):
        flags.append("runtime-heavy-einsum")
    return flags


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (-row["optimistic_gain"], row["task"]))


def _opportunity_label(opportunity: dict[str, Any]) -> str:
    name = opportunity["name"] or "-"
    return f"{name} ({opportunity['saving']})"


def render_markdown(rows: list[dict[str, Any]], baseline: float) -> str:
    lines = [
        "# NeuroGolf 7455 Campaign Queue",
        "",
        f"Baseline: {baseline:.4f}",
        "",
        "| rank | task | cost | points | optimistic_gain | largest_tensor | largest_param_bank | risk_flags |",
        "| ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for rank, row in enumerate(rows, start=1):
        flags = ", ".join(row["risk_flags"]) or "-"
        lines.append(
            f"| {rank} | {row['task']:03d} | {row['cost']} | {row['points']:.6f} | "
            f"{row['optimistic_gain']:.6f} | {_opportunity_label(row['largest_tensor'])} | "
            f"{_opportunity_label(row['largest_param_bank'])} | {flags} |"
        )
    return "\n".join(lines) + "\n"


def _load_inventory(manifest_path: Path, nets_dir: Path) -> tuple[float, list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text())
    rows_by_task = {int(row["task"]): row for row in manifest.values()}
    expected_tasks = set(range(1, 401))
    if set(rows_by_task) != expected_tasks or len(manifest) != 400:
        raise ValueError("manifest must contain exactly one row for each task 001-400")

    rows = []
    mismatches = []
    for task in range(1, 401):
        manifest_row = rows_by_task[task]
        model_path = nets_dir / f"task{task:03d}.onnx"
        actual_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
        expected_sha = str(manifest_row["sha256"])
        if actual_sha != expected_sha:
            mismatches.append(f"task{task:03d}: expected {expected_sha}, got {actual_sha}")
            continue

        model = onnx.load(model_path)
        row = analyze_model(task, model, manifest_row)
        row["optimistic_gain"] = max(
            row["largest_tensor"]["expected_gain"],
            row["largest_param_bank"]["expected_gain"],
        )
        row["risk_flags"] = risk_flags(task, model)
        rows.append(row)

    if mismatches:
        raise ValueError("deployed model SHA-256 mismatch:\n" + "\n".join(mismatches))
    baseline = sum(float(row["points"]) for row in rows_by_task.values())
    return baseline, rank_rows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inventory and rank NeuroGolf campaign opportunities.")
    parser.add_argument("--manifest", required=True, type=Path, help="Path to the 400-task manifest JSON.")
    parser.add_argument("--nets", required=True, type=Path, help="Directory containing deployed taskNNN.onnx files.")
    parser.add_argument("--out", required=True, type=Path, help="Directory for inventory.json and queue.md.")
    args = parser.parse_args(argv)

    try:
        baseline, rows = _load_inventory(args.manifest, args.nets)
    except (OSError, ValueError, json.JSONDecodeError, DecodeError) as exc:
        parser.error(str(exc))

    payload = {"baseline": baseline, "task_count": len(rows), "rows": rows}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "inventory.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.out / "queue.md").write_text(render_markdown(rows, baseline))
    print(f"{len(rows)} models inventoried; 0 SHA mismatches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
