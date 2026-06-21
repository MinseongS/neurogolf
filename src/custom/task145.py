"""task145 (ARC-AGI 6455b5f5) — colour the min/max-area rooms of a red-walled partition.

Rule (verified closed-form, 300/300):
  The input shows only RED(2) walls that partition the top-left HxW grid into axis-aligned
  rectangular rooms (background=0); off-grid (bottom/right border) = all channels 0.
  For every background cell, its room AREA = (horizontal bg run-length) * (vertical bg run-length)
  -- exact because each room is a solid rectangle, so the bg run through any of its cells equals
  the room's width / height.  Let A = area at each bg cell.
    - bg cell whose area == global MAX area  -> BLUE(1)
    - bg cell whose area == global MIN area  -> CYAN(8)
    - other bg cells                         -> stay BG(0)
    - red walls                              -> stay RED(2)
    - off-grid cells                         -> all-zero one-hot (sentinel colour 10)

Construction (no flood-fill, fully separable):
  * z = bg-channel slice (ch0) on the WORK x WORK active crop.
  * width per cell via the directional MaxPool-carry trick (task350/367 idiom): mark wall/edge
    positions with an ascending uint8 ramp on the left and a descending ramp on the right; a
    one-sided full-length uint8 MaxPool carries the nearest-wall ramp position; then
    left_marker + right_marker - W gives -width (the deployed net's exact packing).  Same for
    height.  area = neg_width * neg_height (negatives multiply to the positive area).
  * global max / min area over bg cells via ReduceMax / (masked) ReduceMin.
  * in-grid mask from 1-D occupancy PROFILES: ReduceSum of the FREE input over the channel axis
    plus one spatial axis -> tiny [1,1,1,W] / [1,1,W,1] vectors, >0 -> Greater -> And.  This
    replaces the deployed net's two WxW 'inside' MaxPools (saves ~2.8KB of planes).
  * colour index assembled with a Where priority chain (uint8) and routed into the FREE output
    via Pad(->30x30, sentinel 10) + Equal(channel_ids) -> BOOL output.

Beats the deployed kojimar net (15.51) by removing the inside-detection planes.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I64 = TensorProto.INT64

N = 30
W = 20  # active canvas (grid <= 20x20, top-left anchored)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        if isinstance(ins, str):
            ins = [ins]
        nodes.append(helper.make_node(op, list(ins), [out], **attrs))
        return out

    # ---- bg-channel slice on WxW crop -------------------------------------
    init("z_s", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("z_e", np.array([1, 1, W, W], np.int64), np.int64)
    n("Slice", ["input", "z_s", "z_e"], "z_f32")     # [1,1,W,W] f32, bg=1
    n("Cast", ["z_f32"], "z_bool", to=BOOL)          # bg mask

    # ---- directional run-length markers (uint8 MaxPool) -------------------
    init("zero_u8", np.array(0, np.uint8), np.uint8)
    pos_left = np.arange(1, W + 1, dtype=np.uint8).reshape(1, 1, 1, W)
    pos_right = np.arange(W, 0, -1, dtype=np.uint8).reshape(1, 1, 1, W)
    pos_up = np.arange(1, W + 1, dtype=np.uint8).reshape(1, 1, W, 1)
    pos_down = np.arange(W, 0, -1, dtype=np.uint8).reshape(1, 1, W, 1)
    init("pos_left", pos_left, np.uint8)
    init("pos_right", pos_right, np.uint8)
    init("pos_up", pos_up, np.uint8)
    init("pos_down", pos_down, np.uint8)

    # source = ramp at non-bg (wall/offgrid) cells, 0 at bg cells
    n("Where", ["z_bool", "zero_u8", "pos_left"], "left_src")
    n("Where", ["z_bool", "zero_u8", "pos_right"], "right_src")
    n("Where", ["z_bool", "zero_u8", "pos_up"], "up_src")
    n("Where", ["z_bool", "zero_u8", "pos_down"], "down_src")

    n("MaxPool", ["left_src"], "left_marker", kernel_shape=[1, W], pads=[0, W - 1, 0, 0], strides=[1, 1])
    n("MaxPool", ["right_src"], "right_marker", kernel_shape=[1, W], pads=[0, 0, 0, W - 1], strides=[1, 1])
    n("MaxPool", ["up_src"], "up_marker", kernel_shape=[W, 1], pads=[W - 1, 0, 0, 0], strides=[1, 1])
    n("MaxPool", ["down_src"], "down_marker", kernel_shape=[W, 1], pads=[0, 0, W - 1, 0], strides=[1, 1])

    # lr_sum = left_marker + right_marker (uint8).  DequantizeLinear(x, scale=1, zp=W) fuses
    # the uint8->fp16 cast and the "- W" into one op (deployed-net idiom): -> (lr_sum - W) fp16.
    n("Add", ["left_marker", "right_marker"], "lr_sum")    # uint8
    n("Add", ["up_marker", "down_marker"], "ud_sum")
    init("scale_f16", np.array(1.0, np.float16), np.float16)
    init("W_u8", np.array(W, np.uint8), np.uint8)
    n("DequantizeLinear", ["lr_sum", "scale_f16", "W_u8"], "neg_width")
    n("DequantizeLinear", ["ud_sum", "scale_f16", "W_u8"], "neg_height")
    n("Mul", ["neg_width", "neg_height"], "area")          # = width*height (>0), fp16

    # ---- global max / min area over bg cells ------------------------------
    init("axes_hw", np.array([2, 3], np.int64), np.int64)
    n("ReduceMax", ["area", "axes_hw"], "max_area", keepdims=1)   # [1,1,1,1]
    # at non-bg cells, area is large/garbage; for the min, blank them to max_area.
    n("Where", ["z_bool", "area", "max_area"], "area_for_min")
    n("ReduceMin", ["area_for_min", "axes_hw"], "min_area", keepdims=1)
    n("Equal", ["area", "max_area"], "max_mask")           # bool
    n("Equal", ["area_for_min", "min_area"], "min_mask")

    # ---- in-grid mask from 1-D occupancy profiles -------------------------
    init("ax_col", np.array([1, 2], np.int64), np.int64)   # channel + rows -> [1,1,1,30]
    init("ax_row", np.array([1, 3], np.int64), np.int64)   # channel + cols -> [1,1,30,1]
    n("ReduceSum", ["input", "ax_col"], "colsum", keepdims=1)
    n("ReduceSum", ["input", "ax_row"], "rowsum", keepdims=1)
    init("v_s", np.array([0], np.int64), np.int64)
    init("v_e", np.array([W], np.int64), np.int64)
    init("v_ax_c", np.array([3], np.int64), np.int64)
    init("v_ax_r", np.array([2], np.int64), np.int64)
    n("Slice", ["colsum", "v_s", "v_e", "v_ax_c"], "colsumA")   # [1,1,1,W]
    n("Slice", ["rowsum", "v_s", "v_e", "v_ax_r"], "rowsumA")   # [1,1,W,1]
    init("zero_f32", np.array(0.0, np.float32), np.float32)
    n("Greater", ["colsumA", "zero_f32"], "col_in")            # [1,1,1,W] bool
    n("Greater", ["rowsumA", "zero_f32"], "row_in")            # [1,1,W,1] bool
    n("And", ["row_in", "col_in"], "inside")                  # [1,1,W,W] bool (broadcast)

    # ---- assemble colour index (uint8) ------------------------------------
    init("two_u8", np.array(2, np.uint8), np.uint8)
    init("eight_u8", np.array(8, np.uint8), np.uint8)
    init("one_u8", np.array(1, np.uint8), np.uint8)
    init("ten_u8", np.array(10, np.uint8), np.uint8)
    n("Where", ["z_bool", "zero_u8", "two_u8"], "color0")      # bg->0, wall->2
    n("Where", ["inside", "color0", "ten_u8"], "color1")       # off-grid -> 10 sentinel
    n("Where", ["max_mask", "one_u8", "color1"], "color2")     # max-area -> blue(1)
    n("Where", ["min_mask", "eight_u8", "color2"], "color_core")  # min-area -> cyan(8)

    # ---- route into FREE output -------------------------------------------
    init("core_pads", np.array([0, 0, 0, 0, 0, 0, N - W, N - W], np.int64), np.int64)
    n("Pad", ["color_core", "core_pads", "ten_u8"], "color30", mode="constant")  # [1,1,30,30] u8
    channel_ids = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("channel_ids_u8", channel_ids, np.uint8)
    n("Equal", ["color30", "channel_ids_u8"], "output")        # bool [1,10,30,30]

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task145", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 19)])
