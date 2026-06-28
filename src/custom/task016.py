"""Task 016 — fixed colour permutation.

Live network is a single Gather on the channel axis with index
`[0,5,6,4,3,1,2,7,9,8]`.  This builder owns that source-level transform.
"""

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ..harness import IR_VERSION


def build(task):
    idx = numpy_helper.from_array(
        np.array([0, 5, 6, 4, 3, 1, 2, 7, 9, 8], np.int64), "idx_val"
    )
    node = helper.make_node("Gather", ["input", "idx_val"], ["output"], axis=1)
    graph = helper.make_graph(
        [node],
        "task016",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [idx],
    )
    return helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 11)])
