from __future__ import annotations

import math
from typing import Any

import numpy as np
import onnx


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
