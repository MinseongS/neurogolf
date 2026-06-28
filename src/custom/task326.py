"""Task 326 — keep only top-left 2x2 region via separable spatial mask."""

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ..harness import IR_VERSION


def build(task):
    mask = np.zeros((30,), np.float32)
    mask[:2] = 1.0
    inits = [
        numpy_helper.from_array(mask, "spatial_mask"),
    ]
    node = helper.make_node("Einsum", ["input", "spatial_mask", "spatial_mask"], ["output"], equation="nchw,h,w->nchw")
    graph = helper.make_graph(
        [node],
        "task326",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        inits,
    )
    return helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 12)])
