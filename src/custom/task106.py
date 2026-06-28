"""Task 106 — C4 symmetrization of a 2x2 or 3x3 colour grid.

The live graph flattens the top-left 3x3 non-background colour patch, chooses a
precomputed permutation table for size=2 vs size=3, then scatters the resulting
one-hot channels into the free output.
"""

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ..harness import IR_VERSION


IDX2 = np.array(
    [
        [0, 1, 3, 0, 8, 8],
        [3, 4, 4, 1, 8, 8],
        [1, 4, 4, 3, 8, 8],
        [0, 3, 1, 0, 8, 8],
        [8, 8, 8, 8, 8, 8],
        [8, 8, 8, 8, 8, 8],
    ],
    np.int32,
)
IDX3 = np.array(
    [
        [0, 1, 2, 6, 3, 0],
        [3, 4, 5, 7, 4, 1],
        [6, 7, 8, 8, 5, 2],
        [2, 5, 8, 8, 7, 6],
        [1, 4, 7, 5, 4, 3],
        [0, 3, 6, 2, 1, 0],
    ],
    np.int32,
)


def build(task):
    inits = []

    def init(name, arr, dt):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dt), name))

    init("patch_starts", np.array([0, 1, 0, 0], np.int64), np.int64)
    init("patch_ends", np.array([1, 10, 3, 3], np.int64), np.int64)
    init("color_w", np.arange(1, 10, dtype=np.float32).reshape(1, 9, 1, 1), np.float32)
    init("flat_shape", np.array([1, 1, 9], np.int64), np.int64)
    init("probe_idx", np.array([6], np.int32), np.int32)
    init("idx2", IDX2, np.int32)
    init("idx3", IDX3, np.int32)

    nodes = [
        helper.make_node("Slice", ["input", "patch_starts", "patch_ends"], ["patch"]),
        helper.make_node("Conv", ["patch", "color_w"], ["colors_f_1133"], kernel_shape=[1, 1]),
        helper.make_node("Cast", ["colors_f_1133"], ["colors_u8_1133"], to=TensorProto.UINT8),
        helper.make_node("Reshape", ["colors_u8_1133", "flat_shape"], ["colors"]),
        helper.make_node("Gather", ["colors", "probe_idx"], ["probe"], axis=2),
        helper.make_node("Cast", ["probe"], ["is3"], to=TensorProto.BOOL),
        helper.make_node("Gather", ["colors", "idx2"], ["channels2"], axis=2),
        helper.make_node("Gather", ["colors", "idx3"], ["channels3"], axis=2),
        helper.make_node("Where", ["is3", "channels3", "channels2"], ["channels_u8"]),
        helper.make_node("Cast", ["channels_u8"], ["channels"], to=TensorProto.INT32),
        helper.make_node("Cast", ["channels_u8"], ["updates"], to=TensorProto.FLOAT),
        helper.make_node("ScatterElements", ["input", "channels", "updates"], ["output"], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        "task106",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        inits,
    )
    return helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 16)])
