"""Task 385 — fixed row Gather."""

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ..harness import IR_VERSION


IDX = [9, 8, 7, 6, 5, 5, 6, 7, 8, 9] + list(range(10, 30))


def build(task):
    idx = numpy_helper.from_array(np.array(IDX, np.int64), "v0")
    node = helper.make_node("Gather", ["input", "v0"], ["output"], axis=2)
    graph = helper.make_graph(
        [node],
        "task385",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [idx],
    )
    return helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 11)])
