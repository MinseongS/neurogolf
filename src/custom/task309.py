"""Task 309 — fixed colour/channel permutation."""

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ..harness import IR_VERSION


IDX = [0, 1, 2, 3, 4, 7, 6, 5, 8, 9]


def build(task):
    idx = numpy_helper.from_array(np.array(IDX, np.int64), "idx")
    node = helper.make_node("Gather", ["input", "idx"], ["output"], axis=1)
    graph = helper.make_graph(
        [node],
        "task309",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [idx],
    )
    return helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 11)])
