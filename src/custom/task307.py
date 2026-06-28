"""Task 307 — compact spatial crop/scale via MaxRoiPool.

Matches the live one-node network.
"""

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ..harness import IR_VERSION


def build(task):
    rois = numpy_helper.from_array(np.array([[0.0, 0.0, 0.0, 14.0, 14.0]], np.float32), "rois")
    node = helper.make_node(
        "MaxRoiPool",
        ["input", "rois"],
        ["output"],
        pooled_shape=[30, 30],
        spatial_scale=1.0,
    )
    graph = helper.make_graph(
        [node],
        "task307",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [rois],
    )
    return helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 11)])
