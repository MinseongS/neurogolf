"""Task 282 — small 7x7 local code emission with uint8 QLinearConv."""

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ..harness import IR_VERSION


def build(task):
    inits = []

    def init(name, arr, dt):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dt), name))

    init("starts", np.array([0, 5, 1, 1], np.int64), np.int64)
    init("ends", np.array([1, 6, 8, 8], np.int64), np.int64)
    init("scale", np.array(1.0, np.float32), np.float32)
    init("zero", np.array(0, np.uint8), np.uint8)
    init("one", np.array(1, np.uint8), np.uint8)
    code_kernel = np.array(
        [
            [[[1, 2, 1], [2, 0, 2], [1, 2, 1]]],
            [[[2, 1, 2], [1, 0, 1], [2, 1, 2]]],
        ],
        np.uint8,
    )
    init("code_kernel", code_kernel, np.uint8)
    init("emit_kernel", np.array([0, 2, 1, 1, 1, 2, 1, 1, 1, 1], np.uint8).reshape(10, 1, 1, 1), np.uint8)

    nodes = [
        helper.make_node("Slice", ["input", "starts", "ends"], ["v"]),
        helper.make_node("Cast", ["v"], ["c"], to=TensorProto.UINT8),
        helper.make_node(
            "QLinearConv",
            ["c", "scale", "zero", "code_kernel", "scale", "zero", "scale", "zero"],
            ["codes"],
            pads=[2, 2, 2, 2],
        ),
        helper.make_node(
            "QLinearConv",
            ["codes", "scale", "one", "emit_kernel", "scale", "one", "scale", "zero"],
            ["output"],
            group=2,
            pads=[0, 0, 21, 21],
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "task282",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 10, 30, 30])],
        inits,
    )
    return helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 21)])
