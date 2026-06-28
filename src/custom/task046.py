"""Task 046 — source-owned snake compaction heuristic.

The input is a height-3 coloured snake with separator columns and per-segment
vertical offsets.  This heuristic compresses visible columns, estimates segment
offset changes from gray junction rows, propagates segment colour across gray
cells, and shifts each column back into the canonical 3-row frame.
"""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper

from ._exact import model, tensor


def build(task):
    color_weights = np.zeros((1, 10, 2, 1), dtype=np.float32)
    for ch, value in enumerate([0, 2, 3, 4, 5, 1, 6, 7, 8, 9]):
        color_weights[0, ch, 0, 0] = value

    inits = [
        tensor("zero_f", np.array(0.0, dtype=np.float16)),
        tensor("path_weights", np.arange(30, 0, -1, dtype=np.float16).reshape(1, 30)),
        tensor("axis_one", np.array(1, dtype=np.int64)),
        tensor("topk_k", np.array([16], dtype=np.int64)),
        tensor("one_i8", np.array(1, dtype=np.int8)),
        tensor("zero_u8", np.array(0, dtype=np.uint8)),
        tensor("gray_u8", np.array(1, dtype=np.uint8)),
        tensor("outside_u8", np.array(10, dtype=np.uint8)),
        tensor("row_ids_u8", np.arange(3, dtype=np.uint8).reshape(1, 3, 1)),
        tensor("color_values", np.array([0, 2, 3, 4, 5, 11, 6, 7, 8, 9], dtype=np.uint8).reshape(1, 10, 1, 1)),
        tensor("color_weights", color_weights),
        tensor("path_color_axes", np.array([1, 2], dtype=np.int64)),
        tensor("comp_color_axes", np.array([1, 3], dtype=np.int64)),
        tensor("src_idx_head_starts", np.array([0], dtype=np.int64)),
        tensor("src_idx_head_ends", np.array([15], dtype=np.int64)),
        tensor("src_idx_head_axes", np.array([1], dtype=np.int64)),
        tensor("offset_delta_i8_pads", np.array([0, 1, 0, 0], dtype=np.int64)),
        tensor("visible_right_gated_pads", np.array([0, 0, 0, 1], dtype=np.int64)),
        tensor("offset_-2", np.array(-2.0, dtype=np.float16)),
        tensor("offset_-1", np.array(-1.0, dtype=np.float16)),
        tensor("offset_1", np.array(1.0, dtype=np.float16)),
        tensor("offset_2", np.array(2.0, dtype=np.float16)),
        tensor("output_pads", np.array([0, 0, 0, 0, 0, 0, 27, 14], dtype=np.int64)),
    ]

    nodes = [
        helper.make_node("Conv", ["input", "color_weights"], ["color_grid_f_full"], kernel_shape=[2, 1], dilations=[27, 1]),
        helper.make_node("Cast", ["color_grid_f_full"], ["color_grid_u"], to=TensorProto.UINT8),
        helper.make_node("ReduceMax", ["color_grid_u", "path_color_axes"], ["path_color"], keepdims=0),
        helper.make_node("Greater", ["path_color", "zero_u8"], ["path_bool"]),
        helper.make_node("Where", ["path_bool", "path_weights", "zero_f"], ["path_score"]),
        helper.make_node("TopK", ["path_score", "topk_k"], ["topk_values", "source_idx64_b"], axis=1, largest=1, sorted=1),
        helper.make_node("Greater", ["topk_values", "zero_f"], ["width_present"]),
        helper.make_node("Gather", ["color_grid_u", "source_idx64_b"], ["comp_color_u"], axis=3),
        helper.make_node("Squeeze", ["comp_color_u", "comp_color_axes"], ["comp_color"]),
        helper.make_node("Cast", ["source_idx64_b"], ["source_idx_i8"], to=TensorProto.INT8),
        helper.make_node("Slice", ["source_idx_i8", "src_idx_head_starts", "src_idx_head_ends", "src_idx_head_axes"], ["src_idx_head"]),
        helper.make_node("Slice", ["source_idx_i8", "src_idx_head_axes", "topk_k", "src_idx_head_axes"], ["src_idx_tail"]),
        helper.make_node("Sub", ["src_idx_tail", "src_idx_head"], ["src_idx_gap"]),
        helper.make_node("Greater", ["src_idx_gap", "one_i8"], ["boundary_tail"]),
        helper.make_node("Equal", ["comp_color", "gray_u8"], ["gray_cells"]),
        helper.make_node("Where", ["gray_cells", "row_ids_u8", "zero_u8"], ["gray_row_cells"]),
        helper.make_node("ReduceMax", ["gray_row_cells", "src_idx_head_axes"], ["gray_row_u8"], keepdims=0),
        helper.make_node("Cast", ["gray_row_u8"], ["gray_row_i8"], to=TensorProto.INT8),
        helper.make_node("Slice", ["gray_row_i8", "src_idx_head_starts", "src_idx_head_ends", "src_idx_head_axes"], ["gray_head"]),
        helper.make_node("Slice", ["gray_row_i8", "src_idx_head_axes", "topk_k", "src_idx_head_axes"], ["gray_tail"]),
        helper.make_node("Sub", ["gray_tail", "gray_head"], ["offset_delta_tail"]),
        helper.make_node("Cast", ["boundary_tail"], ["boundary_tail_i8"], to=TensorProto.INT8),
        helper.make_node("Mul", ["offset_delta_tail", "boundary_tail_i8"], ["offset_delta_tail_gated"]),
        helper.make_node("Pad", ["offset_delta_tail_gated", "offset_delta_i8_pads"], ["offset_delta_i8"], mode="constant"),
        helper.make_node("Cast", ["offset_delta_i8"], ["offset_delta"], to=TensorProto.FLOAT16),
        helper.make_node("CumSum", ["offset_delta", "axis_one"], ["offset_by_col"]),
        helper.make_node("ReduceMax", ["comp_color", "src_idx_head_axes"], ["visible_color"], keepdims=0),
        helper.make_node("Slice", ["visible_color", "src_idx_head_starts", "src_idx_head_ends", "src_idx_head_axes"], ["visible_left_core"]),
        helper.make_node("Where", ["boundary_tail", "zero_u8", "visible_left_core"], ["visible_left_core_gated"]),
        helper.make_node("Pad", ["visible_left_core_gated", "offset_delta_i8_pads"], ["visible_left_gated"], mode="constant"),
        helper.make_node("Slice", ["visible_color", "src_idx_head_axes", "topk_k", "src_idx_head_axes"], ["visible_right_core"]),
        helper.make_node("Where", ["boundary_tail", "zero_u8", "visible_right_core"], ["visible_right_core_gated"]),
        helper.make_node("Pad", ["visible_right_core_gated", "visible_right_gated_pads"], ["visible_right_gated"], mode="constant"),
        helper.make_node("Max", ["visible_color", "visible_left_gated", "visible_right_gated"], ["segment_color_cols"]),
        helper.make_node("Where", ["gray_cells", "segment_color_cols", "comp_color"], ["filled_comp_color"]),
        helper.make_node("Equal", ["offset_by_col", "offset_-2"], ["offset_-2_mask"]),
        helper.make_node("Equal", ["offset_by_col", "offset_-1"], ["offset_-1_mask"]),
        helper.make_node("Equal", ["offset_by_col", "zero_f"], ["offset_0_mask"]),
        helper.make_node("Equal", ["offset_by_col", "offset_1"], ["offset_1_mask"]),
        helper.make_node("Equal", ["offset_by_col", "offset_2"], ["offset_2_mask"]),
        helper.make_node("Split", ["filled_comp_color"], ["src_row_0", "src_row_1", "src_row_2"], axis=1, num_outputs=3),
        helper.make_node("Where", ["offset_0_mask", "src_row_0", "zero_u8"], ["shifted_row_0_off_0"]),
        helper.make_node("Where", ["offset_1_mask", "src_row_1", "shifted_row_0_off_0"], ["shifted_row_0_off_1"]),
        helper.make_node("Where", ["offset_2_mask", "src_row_2", "shifted_row_0_off_1"], ["shifted_row_0_off_2"]),
        helper.make_node("Where", ["offset_-1_mask", "src_row_0", "zero_u8"], ["shifted_row_1_off_-1"]),
        helper.make_node("Where", ["offset_0_mask", "src_row_1", "shifted_row_1_off_-1"], ["shifted_row_1_off_0"]),
        helper.make_node("Where", ["offset_1_mask", "src_row_2", "shifted_row_1_off_0"], ["shifted_row_1_off_1"]),
        helper.make_node("Where", ["offset_-2_mask", "src_row_0", "zero_u8"], ["shifted_row_2_off_-2"]),
        helper.make_node("Where", ["offset_-1_mask", "src_row_1", "shifted_row_2_off_-2"], ["shifted_row_2_off_-1"]),
        helper.make_node("Where", ["offset_0_mask", "src_row_2", "shifted_row_2_off_-1"], ["shifted_row_2_off_0"]),
        helper.make_node("Concat", ["shifted_row_0_off_2", "shifted_row_1_off_1", "shifted_row_2_off_0"], ["shifted_color"], axis=1),
        helper.make_node("Where", ["width_present", "shifted_color", "outside_u8"], ["output_ids"]),
        helper.make_node("Equal", ["output_ids", "color_values"], ["output_top"]),
        helper.make_node("Pad", ["output_top", "output_pads"], ["output"], mode="constant"),
    ]
    return model("task046", nodes, inits, output_dtype=TensorProto.BOOL, opset=20)
