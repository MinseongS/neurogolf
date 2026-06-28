"""Task 081 — complete isolated cyan L-trominoes on a 7x7 canvas.

The live graph uses two uint8 QLinearConv stages:
1. local detectors over the cyan channel;
2. direct emission/padding to the 30x30 output.
"""

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ..harness import IR_VERSION


def build(task):
    inits = []

    def init(name, arr, dt):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dt), name))

    init("x8_starts", np.array([0, 8, 0, 0], np.int64), np.int64)
    init("x8_ends", np.array([1, 9, 7, 7], np.int64), np.int64)
    init("q_scale", np.array(1.0, np.float32), np.float32)
    init("u8_zero", np.array(0, np.uint8), np.uint8)
    init("i8_zero", np.array(0, np.int8), np.int8)
    init("x1_zero", np.array(1, np.uint8), np.uint8)
    hidden_w = np.array(
        [
            [[[0, 0, 0], [6, -6, 6], [0, 12, 0]]],
            [[[0, 12, 0], [6, -6, 6], [0, 0, 0]]],
            [[[0, 0, 0], [0, 3, 0], [0, 0, 0]]],
        ],
        np.int8,
    )
    init("hidden_w", hidden_w, np.int8)
    init("hidden_b", np.array([-15, -15, 0], np.int32), np.int32)
    final_w = np.zeros((10, 3, 1, 1), np.int8)
    final_w[:, :, 0, 0] = np.array(
        [
            [-1, -1, -1],
            [1, 1, -1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [-1, 0, 1],
            [0, 0, 0],
        ],
        np.int8,
    )
    init("final_w", final_w, np.int8)

    nodes = [
        helper.make_node("Slice", ["input", "x8_starts", "x8_ends"], ["x8"]),
        helper.make_node("Cast", ["x8"], ["cyan_u8"], to=TensorProto.UINT8),
        helper.make_node(
            "QLinearConv",
            ["cyan_u8", "q_scale", "u8_zero", "hidden_w", "q_scale", "i8_zero", "q_scale", "u8_zero", "hidden_b"],
            ["features"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        ),
        helper.make_node(
            "QLinearConv",
            ["features", "q_scale", "x1_zero", "final_w", "q_scale", "i8_zero", "q_scale", "u8_zero"],
            ["output"],
            kernel_shape=[1, 1],
            pads=[0, 0, 23, 23],
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "task081",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 10, 30, 30])],
        inits,
    )
    return helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 21)])
