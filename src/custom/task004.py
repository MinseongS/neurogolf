"""Task 004 — compact row-shift rule with packed endpoint test.

The live teacher graph already uses a compact 16x16 scalar-color workspace.
This source-owned variant keeps the same rule but computes the bottom-row
endpoint predicate with one uint8 QLinearConv instead of two shifted planes.
"""

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ..harness import IR_VERSION


def _tensor(name, arr):
    arr = np.asarray(arr)
    if arr.ndim > 0:
        arr = np.ascontiguousarray(arr)
    return numpy_helper.from_array(arr, name)


def build(task):
    wc = np.zeros((1, 10, 2, 2), dtype=np.float32)
    for k in range(10):
        wc[0, k, 0, 0] = -255 + k

    inits = [
        _tensor("Wc", wc),
        _tensor("bc", np.array([255], dtype=np.float32)),
        _tensor("u255", np.array(255, dtype=np.uint8)),
        _tensor("u0", np.array(0, dtype=np.uint8)),
        _tensor("u1", np.array(1, dtype=np.uint8)),
        _tensor("u2", np.array(2, dtype=np.uint8)),
        _tensor("scale", np.array(1.0, dtype=np.float32)),
        _tensor("wend", np.array([[[[0, 1], [2, 0]]]], dtype=np.uint8)),
        _tensor("bidx", np.array(list(range(1, 16)) + [0], dtype=np.int64)),
        _tensor("sidx", np.array([15] + list(range(15)), dtype=np.int64)),
        _tensor("pads", np.array([0, 0, 0, 0, 0, 0, 14, 14], dtype=np.int64)),
        _tensor("palette", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)),
    ]
    nodes = [
        helper.make_node(
            "Conv",
            ["input", "Wc", "bc"],
            ["code_f"],
            dilations=[14, 14],
            kernel_shape=[2, 2],
            pads=[0, 0, 0, 0],
        ),
        helper.make_node("Cast", ["code_f"], ["code"], to=TensorProto.UINT8),
        helper.make_node("Less", ["code", "u255"], ["valid"]),
        helper.make_node("Where", ["valid", "code", "u0"], ["vals"]),
        helper.make_node("Min", ["vals", "u1"], ["any_color"]),
        helper.make_node(
            "QLinearConv",
            ["any_color", "scale", "u0", "wend", "scale", "u0", "scale", "u0"],
            ["end_code"],
            kernel_shape=[2, 2],
            pads=[0, 0, 1, 1],
        ),
        helper.make_node("Equal", ["end_code", "u2"], ["endpoint"]),
        helper.make_node("ReduceMax", ["vals"], ["row_has_u8"], axes=[3], keepdims=1),
        helper.make_node("Cast", ["row_has_u8"], ["row_has"], to=TensorProto.BOOL),
        helper.make_node("Gather", ["row_has", "bidx"], ["row_next"], axis=2),
        helper.make_node("Xor", ["row_next", "endpoint"], ["move_cond"]),
        helper.make_node("Where", ["move_cond", "vals", "u0"], ["move_vals"]),
        helper.make_node("Where", ["move_cond", "u0", "vals"], ["stay_vals"]),
        helper.make_node("Gather", ["move_vals", "sidx"], ["shifted"], axis=3),
        helper.make_node("Max", ["stay_vals", "shifted"], ["scalar_out"]),
        helper.make_node("Where", ["valid", "scalar_out", "u255"], ["scalar_valid"]),
        helper.make_node("Pad", ["scalar_valid", "pads", "u255"], ["scalar_full"], mode="constant"),
        helper.make_node("Equal", ["scalar_full", "palette"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes,
        "task004_qlinear_endpoint",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
        inits,
    )
    return helper.make_model(
        graph,
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 17)],
    )
