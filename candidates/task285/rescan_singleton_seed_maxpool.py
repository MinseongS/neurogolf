#!/usr/bin/env python3
"""Scan deployed graphs for exact singleton-seed Mul -> MaxPool folds."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = pathlib.Path(__file__).resolve().parents[2]
NETS = ROOT / "submission/overfit_nets"
DTYPE_BYTES = {
    onnx.TensorProto.FLOAT: 4,
    onnx.TensorProto.UINT8: 1,
    onnx.TensorProto.INT8: 1,
    onnx.TensorProto.INT32: 4,
    onnx.TensorProto.INT64: 8,
    onnx.TensorProto.BOOL: 1,
    onnx.TensorProto.FLOAT16: 2,
}


def attribute_ints(node: onnx.NodeProto, name: str, default: list[int]) -> list[int]:
    for attribute in node.attribute:
        if attribute.name == name:
            return [int(value) for value in attribute.ints]
    return default


def attribute_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for attribute in node.attribute:
        if attribute.name == name:
            return int(attribute.i)
    return default


def scan(path: pathlib.Path) -> list[dict[str, object]]:
    model = onnx.shape_inference.infer_shapes(onnx.load(path), strict_mode=False)
    producers = {output: node for node in model.graph.node for output in node.output}
    initializers = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    infos = {
        value.name: value
        for value in list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    }
    rows: list[dict[str, object]] = []
    for pool in model.graph.node:
        if pool.op_type != "MaxPool" or not pool.input:
            continue
        mul = producers.get(pool.input[0])
        if mul is None or mul.op_type != "Mul":
            continue
        seed_names = [name for name in mul.input if name in initializers]
        if len(seed_names) != 1:
            continue
        seed_name = seed_names[0]
        seed = np.asarray(initializers[seed_name])
        nonzero = np.argwhere(seed != 0)
        if len(nonzero) != 1 or float(seed[tuple(nonzero[0])]) != 1.0:
            continue
        dynamic_name = next(name for name in mul.input if name != seed_name)
        info = infos.get(dynamic_name)
        if info is None:
            continue
        tensor_type = info.type.tensor_type
        shape = [int(dim.dim_value) for dim in tensor_type.shape.dim]
        if len(shape) != 4 or any(dimension <= 0 for dimension in shape):
            continue
        kernel = attribute_ints(pool, "kernel_shape", [])
        pads = attribute_ints(pool, "pads", [0, 0, 0, 0])
        strides = attribute_ints(pool, "strides", [1, 1])
        dilations = attribute_ints(pool, "dilations", [1, 1])
        if (
            len(kernel) != 2
            or len(pads) != 4
            or strides != [1, 1]
            or dilations != [1, 1]
            or attribute_int(pool, "ceil_mode", 0) != 0
        ):
            continue
        broadcast_seed = np.broadcast_to(seed, shape)
        broadcast_nonzero = np.argwhere(broadcast_seed != 0)
        spatial_positions = np.unique(broadcast_nonzero[:, 2:], axis=0)
        if len(spatial_positions) != 1:
            continue
        row, col = (int(value) for value in spatial_positions[0])
        height, width = shape[2:]
        top = max(0, row + pads[0] - kernel[0] + 1)
        bottom = min(height - 1, row + pads[0])
        left = max(0, col + pads[1] - kernel[1] + 1)
        right = min(width - 1, col + pads[1])
        block_height = bottom - top + 1
        block_width = right - left + 1
        itemsize = DTYPE_BYTES.get(int(tensor_type.elem_type))
        if itemsize is None:
            continue
        channels = shape[0] * shape[1]
        plane_elements = int(np.prod(shape))
        old_memory = 2 * plane_elements * itemsize
        new_memory = (
            channels + channels * block_height * block_width + plane_elements
        ) * itemsize
        old_params = int(seed.size)
        new_params = 6 + 4 + 8
        task_label: int | str
        if path.stem.startswith("task") and path.stem[-3:].isdigit():
            task_label = int(path.stem[-3:])
        else:
            task_label = path.stem
        rows.append(
            {
                "task": task_label,
                "mul_output": mul.output[0],
                "pool_output": pool.output[0],
                "shape": shape,
                "seed": [row, col],
                "kernel": kernel,
                "pads": pads,
                "block": [top, left, block_height, block_width],
                "estimated_cost_saving": (old_memory - new_memory)
                + (old_params - new_params),
            }
        )
    return rows


def main() -> None:
    rows = [row for path in sorted(NETS.glob("task*.onnx")) for row in scan(path)]
    print(json.dumps({"scanned": 400, "hits": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
