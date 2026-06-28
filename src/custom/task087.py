"""Task 087 — 3x3 rotate-180 via RoiAlign.

Matches the live one-node network.  The unusual ROI coordinates are the compact
no-intermediate way this project uses to route the top-left 3x3 rotated output
through the free graph output.
"""

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ..harness import IR_VERSION


def build(task):
    inits = [
        numpy_helper.from_array(np.array([[3.25, 3.25, -31.75, -31.75]], np.float32), "rois"),
        numpy_helper.from_array(np.array([0], np.int64), "batch_indices"),
    ]
    node = helper.make_node(
        "RoiAlign",
        ["input", "rois", "batch_indices"],
        ["output"],
        coordinate_transformation_mode="half_pixel",
        mode="avg",
        output_height=30,
        output_width=30,
        sampling_ratio=1,
        spatial_scale=1.0,
    )
    graph = helper.make_graph(
        [node],
        "task087",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        inits,
    )
    return helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 16)])
