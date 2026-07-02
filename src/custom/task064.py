"""task064 (2c608aff) candidate — marker-plane-free extremes + fused wrap range tests.

Two priced conversions vs the kojimar-formulation incumbent (mem 13368 / params 68):

1. marker_grid [1,1,30,30] fp32 (3600B) + 4 ArgMax int64 (960B) + 4 casts -> GONE.
   Per-row/per-col FIRST/LAST marker positions are recovered exactly from the FREE
   input via pow2-weighted einsums + Log floor-extraction:
     last  = trunc(log2(sum_w 2^w      * marker[.,w]))
     first = 23 - trunc(log2(sum_w 2^(23-w) * marker[.,w]))
   Grid is top-left anchored with width,height <= 24, so all sums are < 2^24 and
   exact in fp32; trunc via Cast(fp32->u8). Empty rows/cols give code 0 (masked by
   h_active/v_active downstream, same as the incumbent ArgMax-0 convention).

2. The GE/LE/And range pyramid (6 x 576B) -> u8 wraparound range test (4 x 576B):
     in_range = (coord_punched - low) mod 256 <= (high - low)
   with inactive-row sentinels low=200/high=201 (delta 1, coords wrap to >= 55).
   Punched (box) coords 255 wrap to >= 232 > delta, so the box stays excluded.

Everything else (rect/marker channel detection, box bounds, active masks, Pad,
output Where) is byte-identical to the incumbent graph.
"""

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8


def build(task):
    inits = []
    nodes = []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(np.asarray(arr, dtype=dtype), name))
        return name

    def node(op, inputs, outputs, **attrs):
        if isinstance(outputs, str):
            outputs = [outputs]
        nodes.append(helper.make_node(op, inputs, outputs, **attrs))

    # --- initializers (incumbent set) ---
    init("slice0", [0], np.int64)
    init("slice1", [1], np.int64)
    init("slice2", [2], np.int64)
    init("slice3", [3], np.int64)
    init("slice24", [24], np.int64)
    init("axis2", [2], np.int64)
    init("axes023", [0, 2, 3], np.int64)
    init("pad_pads", [0, 0, 0, 0, 0, 0, 6, 6], np.int64)
    init("invalid_2558", np.array(255, dtype=np.uint8), np.uint8)
    init("onehot_depth", np.array(10, dtype=np.int64), np.int64)
    init("onehot_values", [0.0, 1.0], np.float32)
    init("col_coord8", np.arange(24, dtype=np.uint8).reshape(1, 1, 1, 24), np.uint8)
    init("row_coord8", np.arange(24, dtype=np.uint8).reshape(1, 1, 24, 1), np.uint8)
    # --- new initializers ---
    w2 = np.zeros((2, 30), dtype=np.float32)
    w2[0, :24] = 2.0 ** np.arange(24)          # last  = leading bit
    w2[1, :24] = 2.0 ** (23 - np.arange(24))   # first = 23 - leading bit
    init("W2", w2, np.float32)
    init("one_pos", np.array(1.0, dtype=np.float32), np.float32)
    init("inv_ln2eps", np.array((1.0 / np.log(2.0)) * (1.0 + 1e-6), dtype=np.float32), np.float32)
    init("c23_u8", np.array(23, dtype=np.uint8), np.uint8)
    init("c200_u8", np.array(200, dtype=np.uint8), np.uint8)
    init("c201_u8", np.array(201, dtype=np.uint8), np.uint8)

    # --- front end: rect channel vs marker channel (incumbent, unchanged) ---
    node("ReduceSum", ["input", "axes023"], "count10", keepdims=0)
    node("ReduceMax", ["input"], "row_presence", axes=[3], keepdims=1)
    node("ReduceMax", ["input"], "col_presence", axes=[2], keepdims=1)
    node("TopK", ["count10", "slice3"], ["top_vals", "top_idx"], axis=0, largest=1, sorted=1)
    node("Slice", ["top_idx", "slice1", "slice3", "slice0"], "cand_idx")
    node("ReduceSum", ["row_presence", "axes023"], "row_count10", keepdims=0)
    node("ReduceSum", ["col_presence", "axes023"], "col_count10", keepdims=0)
    node("Gather", ["count10", "cand_idx"], "cand_count", axis=0)
    node("Gather", ["row_count10", "cand_idx"], "cand_row_count", axis=0)
    node("Gather", ["col_count10", "cand_idx"], "cand_col_count", axis=0)
    node("Mul", ["cand_row_count", "cand_col_count"], "cand_area")
    node("Equal", ["cand_count", "cand_area"], "cand_fills_box")
    node("Cast", ["cand_fills_box"], "cand_fills_u8", to=2)
    node("ArgMax", ["cand_fills_u8"], "rect_pos", axis=0, keepdims=1)
    node("Gather", ["cand_idx", "rect_pos"], "rect_idx", axis=0)
    node("ReduceSum", ["cand_idx", "slice0"], "cand_sum", keepdims=1, noop_with_empty_axes=0)
    node("Sub", ["cand_sum", "rect_idx"], "marker_idx")
    node("Unsqueeze", ["marker_idx", "slice1"], "marker_idx_11")
    node("OneHot", ["marker_idx_11", "onehot_depth", "onehot_values"], "marker_selector", axis=0)
    node("Gather", ["row_presence", "rect_idx"], "rect_row", axis=1)
    node("Gather", ["col_presence", "rect_idx"], "rect_col", axis=1)
    node("Gather", ["row_presence", "marker_idx"], "row_marker", axis=1)
    node("Gather", ["col_presence", "marker_idx"], "col_marker", axis=1)
    node("Cast", ["rect_row"], "rect_row_bool", to=9)
    node("Cast", ["rect_col"], "rect_col_bool", to=9)
    node("ArgMax", ["rect_col"], "box_left_i", axis=3, keepdims=1)
    node("ArgMax", ["rect_col"], "box_right_i", axis=3, keepdims=1, select_last_index=1)
    node("ArgMax", ["rect_row"], "box_top_i", axis=2, keepdims=1)
    node("ArgMax", ["rect_row"], "box_bottom_i", axis=2, keepdims=1, select_last_index=1)
    node("Cast", ["box_left_i"], "box_left", to=2)
    node("Cast", ["box_right_i"], "box_right", to=2)
    node("Cast", ["box_top_i"], "box_top", to=2)
    node("Cast", ["box_bottom_i"], "box_bottom", to=2)
    node("Slice", ["rect_col_bool", "slice0", "slice24", "slice3"], "rect_col_bool24")
    node("Slice", ["rect_row_bool", "slice0", "slice24", "axis2"], "rect_row_bool24")
    node("Where", ["rect_col_bool24", "invalid_2558", "col_coord8"], "col_coord_outside")
    node("Where", ["rect_row_bool24", "invalid_2558", "row_coord8"], "row_coord_outside")
    node("Cast", ["row_marker"], "row_has_marker", to=9)
    node("Cast", ["col_marker"], "col_has_marker", to=9)
    node("Slice", ["row_has_marker", "slice0", "slice24", "axis2"], "row_has_marker24")
    node("Slice", ["col_has_marker", "slice0", "slice24", "slice3"], "col_has_marker24")
    node("And", ["rect_row_bool24", "row_has_marker24"], "h_active24")
    node("And", ["rect_col_bool24", "col_has_marker24"], "v_active24")

    # --- conversion 1: per-row/per-col marker extremes via pow2 einsum + Log ---
    node("Einsum", ["input", "marker_selector", "W2"], "E_row", equation="bchw,cxy,dw->bdhy")  # [1,2,30,1]
    node("Max", ["E_row", "one_pos"], "E_row_c")
    node("Log", ["E_row_c"], "L_row")
    node("Mul", ["L_row", "inv_ln2eps"], "code_row_f")
    node("Cast", ["code_row_f"], "code_row", to=2)                                             # [1,2,30,1] u8
    node("Slice", ["code_row", "slice0", "slice1", "slice1"], "right_raw30")                   # [1,1,30,1]
    node("Slice", ["code_row", "slice1", "slice2", "slice1"], "left_code30")
    node("Sub", ["c23_u8", "left_code30"], "left_raw30")
    node("Einsum", ["input", "marker_selector", "W2"], "E_col", equation="bchw,cxy,dh->bdyw")  # [1,2,1,30]
    node("Max", ["E_col", "one_pos"], "E_col_c")
    node("Log", ["E_col_c"], "L_col")
    node("Mul", ["L_col", "inv_ln2eps"], "code_col_f")
    node("Cast", ["code_col_f"], "code_col", to=2)                                             # [1,2,1,30] u8
    node("Slice", ["code_col", "slice0", "slice1", "slice1"], "bottom_raw30")                  # [1,1,1,30]
    node("Slice", ["code_col", "slice1", "slice2", "slice1"], "top_code30")
    node("Sub", ["c23_u8", "top_code30"], "top_raw30")
    node("Slice", ["left_raw30", "slice0", "slice24", "axis2"], "left_raw24")
    node("Slice", ["right_raw30", "slice0", "slice24", "axis2"], "right_raw24")
    node("Slice", ["top_raw30", "slice0", "slice24", "slice3"], "top_raw24")
    node("Slice", ["bottom_raw30", "slice0", "slice24", "slice3"], "bottom_raw24")

    # --- span bounds (incumbent semantics) ---
    node("Min", ["left_raw24", "box_left"], "h_low_raw24")
    node("Max", ["right_raw24", "box_right"], "h_high24")
    node("Min", ["top_raw24", "box_top"], "v_low_raw24")
    node("Max", ["bottom_raw24", "box_bottom"], "v_high24")

    # --- conversion 2: fused u8 wraparound range tests ---
    node("Where", ["h_active24", "h_low_raw24", "c200_u8"], "h_low24")
    node("Where", ["h_active24", "h_high24", "c201_u8"], "h_high24m")
    node("Where", ["v_active24", "v_low_raw24", "c200_u8"], "v_low24")
    node("Where", ["v_active24", "v_high24", "c201_u8"], "v_high24m")
    node("Sub", ["h_high24m", "h_low24"], "dh")                       # [1,1,24,1]
    node("Sub", ["v_high24m", "v_low24"], "dv")                       # [1,1,1,24]
    node("Sub", ["col_coord_outside", "h_low24"], "subh")             # [1,1,24,24]
    node("Sub", ["row_coord_outside", "v_low24"], "subv")             # [1,1,24,24]
    node("LessOrEqual", ["subh", "dh"], "h_range")
    node("LessOrEqual", ["subv", "dv"], "v_range")
    node("Or", ["h_range", "v_range"], "fill_mask24")
    node("Pad", ["fill_mask24", "pad_pads"], "fill_mask", mode="constant")
    node("Where", ["fill_mask", "marker_selector", "input"], "output")

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task064_logextremes", [x], [y], inits)
    return helper.make_model(
        graph,
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 17)],
    )
