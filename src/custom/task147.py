"""Task 147 — recolour non-isolated green pixels to cyan.

Exact source for the live crop + local Conv + ScatterElements graph.
"""

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ..harness import IR_VERSION


def build(task):
    inits = []

    def init(name, arr, dt):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dt), name))

    init("slice_starts", np.array([0, 3, 0, 0], np.int64), np.int64)
    init("slice_ends", np.array([1, 4, 6, 6], np.int64), np.int64)
    delta_kernel = np.array(
        [
            [[[0, -1, 0], [-1, 0, -1], [0, -1, 0]]],
            [[[0, 1, 0], [1, 4, 1], [0, 1, 0]]],
        ],
        np.float32,
    )
    init("delta_kernel", delta_kernel, np.float32)
    init("delta_bias", np.array([0.0, -4.5], np.float32), np.float32)
    scatter_indices = np.zeros((1, 2, 6, 6), np.int64)
    scatter_indices[:, 0, :, :] = 3
    scatter_indices[:, 1, :, :] = 8
    init("scatter_indices", scatter_indices, np.int64)
    nodes = [
        helper.make_node("Slice", ["input", "slice_starts", "slice_ends"], ["c3_f"]),
        helper.make_node("Conv", ["c3_f", "delta_kernel", "delta_bias"], ["delta_updates"], pads=[1, 1, 1, 1]),
        helper.make_node(
            "ScatterElements",
            ["input", "scatter_indices", "delta_updates"],
            ["output"],
            axis=1,
            reduction="add",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "task147",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        inits,
    )
    return helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 16)])
