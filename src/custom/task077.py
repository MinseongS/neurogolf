"""Task 077 — red-support heuristic recolour.

This task is underdetermined by the visible grid; the live model uses a compact
heuristic around red-channel support.  It estimates supported rectangle cells
from red rows/columns with small max-pool expansions, then recolours predicted
static rectangle cells to yellow (channel 4).
"""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper

from ._exact import model, tensor


def build(task):
    inits = [
        tensor("slice_starts", np.array([2, 0, 0], dtype=np.int64)),
        tensor("slice_ends", np.array([3, 20, 21], dtype=np.int64)),
        tensor("slice_axes", np.array([1, 2, 3], dtype=np.int64)),
        tensor("row_values", np.arange(20, dtype=np.uint8).reshape(1, 1, 20, 1)),
        tensor("row_inv_values", (30 - np.arange(20, dtype=np.uint8)).reshape(1, 1, 20, 1)),
        tensor("row_min_values", (31 - np.arange(20, dtype=np.uint8)).reshape(1, 1, 20, 1)),
        tensor("row_max_values", (1 + np.arange(20, dtype=np.uint8)).reshape(1, 1, 20, 1)),
        tensor("zero_u8", np.array([0], dtype=np.uint8)),
        tensor("one_u8", np.array([1], dtype=np.uint8)),
        tensor("pad_pads", np.array([0, 0, 0, 0, 0, 0, 10, 9], dtype=np.int64)),
        tensor("channel4_vec", np.eye(10, dtype=np.float32)[4].reshape(1, 10, 1, 1)),
        tensor("q_scale", np.array(1.0, dtype=np.float32)),
        tensor("q_zero", np.array(0, dtype=np.uint8)),
        tensor("col_kernel", np.array([1, 2, 1], dtype=np.uint8).reshape(1, 1, 1, 3)),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "slice_starts", "slice_ends", "slice_axes"], ["x2_float"]),
        helper.make_node("Cast", ["x2_float"], ["x2_bool"], to=TensorProto.BOOL),
        helper.make_node("Cast", ["x2_float"], ["x2_u8"], to=TensorProto.UINT8),
        helper.make_node("Where", ["x2_bool", "row_min_values", "zero_u8"], ["rmin_neg0"]),
        helper.make_node("Where", ["x2_bool", "row_max_values", "zero_u8"], ["rmax0"]),
        helper.make_node("MaxPool", ["x2_u8"], ["vertical_red"], kernel_shape=[5, 1], pads=[2, 0, 2, 0], strides=[1, 1]),
        helper.make_node(
            "QLinearConv",
            ["vertical_red", "q_scale", "q_zero", "col_kernel", "q_scale", "q_zero", "q_scale", "q_zero"],
            ["col_score"],
            pads=[0, 1, 0, 1],
            strides=[1, 1],
        ),
        helper.make_node("Less", ["one_u8", "col_score"], ["col_supported"]),
        helper.make_node("MaxPool", ["rmin_neg0"], ["bbox1_rmin_neg_pool"], kernel_shape=[5, 5], pads=[2, 2, 2, 2], strides=[1, 1]),
        helper.make_node("Where", ["x2_bool", "bbox1_rmin_neg_pool", "zero_u8"], ["bbox1_rmin_neg"]),
        helper.make_node("MaxPool", ["rmax0"], ["bbox1_rmax_pool"], kernel_shape=[5, 5], pads=[2, 2, 2, 2], strides=[1, 1]),
        helper.make_node("Where", ["x2_bool", "bbox1_rmax_pool", "zero_u8"], ["bbox1_rmax"]),
        helper.make_node("MaxPool", ["bbox1_rmin_neg"], ["bbox2_rmin_neg_pool"], kernel_shape=[5, 5], pads=[2, 2, 2, 2], strides=[1, 1]),
        helper.make_node("Where", ["x2_bool", "bbox2_rmin_neg_pool", "zero_u8"], ["bbox2_rmin_neg"]),
        helper.make_node("MaxPool", ["bbox1_rmax"], ["bbox2_rmax_pool"], kernel_shape=[5, 5], pads=[2, 2, 2, 2], strides=[1, 1]),
        helper.make_node("Where", ["x2_bool", "bbox2_rmax_pool", "zero_u8"], ["bbox2_rmax"]),
        helper.make_node("MaxPool", ["bbox2_rmin_neg"], ["fill1_rmin_neg_pool"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]),
        helper.make_node("MaxPool", ["bbox2_rmax"], ["fill1_rmax_pool"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]),
        helper.make_node("Less", ["row_inv_values", "fill1_rmin_neg_pool"], ["fill1_row_ok_min"]),
        helper.make_node("Less", ["row_values", "fill1_rmax_pool"], ["fill1_row_ok_max"]),
        helper.make_node("And", ["fill1_row_ok_min", "fill1_row_ok_max"], ["fill1_b"]),
        helper.make_node("Where", ["fill1_b", "fill1_rmin_neg_pool", "zero_u8"], ["fill1_rmin_neg"]),
        helper.make_node("Where", ["fill1_b", "fill1_rmax_pool", "zero_u8"], ["fill1_rmax"]),
        helper.make_node("MaxPool", ["fill1_rmin_neg"], ["fill2_rmin_neg_pool"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]),
        helper.make_node("MaxPool", ["fill1_rmax"], ["fill2_rmax_pool"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]),
        helper.make_node("Less", ["row_inv_values", "fill2_rmin_neg_pool"], ["fill2_row_ok_min"]),
        helper.make_node("Less", ["row_values", "fill2_rmax_pool"], ["fill2_row_ok_max"]),
        helper.make_node("And", ["fill2_row_ok_min", "fill2_row_ok_max"], ["fill2_b"]),
        helper.make_node("And", ["fill2_b", "col_supported"], ["inside_supported"]),
        helper.make_node("Xor", ["x2_bool", "inside_supported"], ["fill_bool"]),
        helper.make_node("Pad", ["fill_bool", "pad_pads"], ["fill_full"], mode="constant"),
        helper.make_node("Where", ["fill_full", "channel4_vec", "input"], ["output"]),
    ]
    return model("task077", nodes, inits, output_dtype=TensorProto.FLOAT, opset=13)
