"""Task 317 — two-stage nearest Resize."""

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ..harness import IR_VERSION


def build(task):
    inits = [
        numpy_helper.from_array(np.array([0.0, 0.03448276, 0.03448276, 0.5555556, 0.3448276, 0.3448276], np.float32), "sample_roi"),
        numpy_helper.from_array(np.array([0.2, 0.13333334, 0.13333334], np.float32), "sample_scales"),
        numpy_helper.from_array(np.array([0.0, 0.0, 0.0, 9.0, 3.2222223, 3.2222223], np.float32), "out_roi"),
        numpy_helper.from_array(np.array([5.0, 7.5, 7.5], np.float32), "out_scales"),
    ]
    nodes = [
        helper.make_node(
            "Resize",
            ["input", "sample_roi", "sample_scales"],
            ["centers"],
            axes=[1, 2, 3],
            coordinate_transformation_mode="tf_crop_and_resize",
            extrapolation_value=0.0,
            mode="nearest",
            nearest_mode="round_prefer_floor",
        ),
        helper.make_node(
            "Resize",
            ["centers", "out_roi", "out_scales"],
            ["output"],
            axes=[1, 2, 3],
            coordinate_transformation_mode="tf_crop_and_resize",
            extrapolation_value=0.0,
            mode="nearest",
            nearest_mode="floor",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "task317",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        inits,
    )
    return helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 18)])
