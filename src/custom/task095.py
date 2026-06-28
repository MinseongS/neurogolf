"""Task 095 — expand gray dots into 3x3 blue boxes with gray centers.

Exact source for the live crop + QuantizeLinear + ConvInteger graph.
"""

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ..harness import IR_VERSION


def build(task):
    inits = []

    def init(name, arr, dt):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dt), name))

    init("starts", np.array([0, 5, 1, 1], np.int64), np.int64)
    init("ends", np.array([1, 6, 8, 8], np.int64), np.int64)
    init("quant_scale", np.array(1.0 / 255.0, np.float32), np.float32)
    init("x_zero_point", np.array(1, np.uint8), np.uint8)
    weights = np.zeros((10, 1, 3, 3), np.int8)
    weights[0, 0] = -1
    weights[1, 0] = np.array([[1, 1, 1], [1, -1, 1], [1, 1, 1]], np.int8)
    weights[5, 0, 1, 1] = 1
    init("weights", weights, np.int8)
    nodes = [
        helper.make_node("Slice", ["input", "starts", "ends"], ["c5_f"]),
        helper.make_node("QuantizeLinear", ["c5_f", "quant_scale"], ["encoded_lattice"]),
        helper.make_node(
            "ConvInteger",
            ["encoded_lattice", "weights", "x_zero_point"],
            ["output"],
            kernel_shape=[3, 3],
            pads=[2, 2, 23, 23],
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "task095",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.INT32, [1, 10, 30, 30])],
        inits,
    )
    return helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 21)])
