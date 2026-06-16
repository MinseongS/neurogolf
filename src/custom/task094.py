"""Task 094 (41e4d17e): draw pink crosshairs through each blue-box centre.

Rule (from ARC-GEN generator):
  Grid is size 15, background cyan(8).  Input holds 1-2 blue(1) 5x5 box
  *outlines* (the perimeter of a 5x5 square), centred at (r,c) with r,c in
  3..11 and the boxes well separated (>=3 gap each axis, crosshairs never
  touch).  The OUTPUT keeps the blue boxes and additionally paints, for every
  box centre (r,c), the entire row r and entire column c pink(6).  Blue boxes
  are drawn AFTER the pink crosshairs, so blue overwrites pink where they
  overlap.

Encoding (Tier-A separable label-map):
  * One Conv over the input one-hot with a 5x5 kernel that matches the box
    outline on the blue channel only.  The response equals 16 EXACTLY at each
    box centre and <=8 everywhere else, so `resp > 12` isolates the centres as
    a [1,1,30,30] bool plane (never materialised as fp; reduced immediately).
  * Reduce that plane to 1-D centre profiles: is_centre_row = OR over cols
    ([1,1,30,1]), is_centre_col = OR over rows ([1,1,1,30]).  cross = row OR col
    broadcasts to the full crosshair mask in the FREE final ops.
  * blue mask = input channel 1 (>0).  Label map L (uint8, 15x15 canvas):
    base cyan(8); pink(6) where cross; blue(1) where blue (priority).  Pad to
    30x30 with sentinel 10 (off-grid -> all-channels-off) and a single final
    Equal(L, arange[0..9]) into the free BOOL output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

WORK = 15  # active grid side (generator size = 15)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- box-outline detection kernel (blue channel only) ----
    # 5x5 perimeter-of-square pattern; channel 1 (blue) carries it, rest zero.
    outline = np.array([[1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1]], dtype=np.float32)
    kw = np.zeros((1, 10, 5, 5), dtype=np.float32)
    kw[0, 1] = outline
    init("kw", kw, np.float32)            # response peaks at 16 at each centre
    init("thr", np.array(12.0, np.float32), np.float32)

    # colour-index constants
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    init("cyan", np.array(8, np.uint8), np.uint8)
    init("pink", np.array(6, np.uint8), np.uint8)
    init("blue", np.array(1, np.uint8), np.uint8)
    init("half", np.array(0.5, np.float32), np.float32)

    # crop / pad helpers
    init("c0", np.array([0], np.int64), np.int64)
    init("c15", np.array([WORK], np.int64), np.int64)
    init("ax2", np.array([2], np.int64), np.int64)
    init("ax3", np.array([3], np.int64), np.int64)
    init("blue_st", np.array([1, 0, 0], np.int64), np.int64)
    init("blue_en", np.array([2, WORK, WORK], np.int64), np.int64)
    init("blue_ax", np.array([1, 2, 3], np.int64), np.int64)
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64),
         np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)

    # ---- centre detection -> 1-D centre profiles ----
    n("Conv", ["input", "kw"], "resp", pads=[2, 2, 2, 2])  # [1,1,30,30] f32
    # reduce the full-grid response straight to 1-D centre profiles (no 15x15
    # fp plane materialised): ReduceMax over each axis, then crop to 15.
    n("ReduceMax", ["resp"], "row_max30", axes=[3], keepdims=1)  # [1,1,30,1]
    n("ReduceMax", ["resp"], "col_max30", axes=[2], keepdims=1)  # [1,1,1,30]
    n("Slice", ["row_max30", "c0", "c15", "ax2"], "row_max")  # [1,1,15,1]
    n("Slice", ["col_max30", "c0", "c15", "ax3"], "col_max")  # [1,1,1,15]
    n("Greater", ["row_max", "thr"], "is_crow")        # [1,1,15,1] bool
    n("Greater", ["col_max", "thr"], "is_ccol")        # [1,1,1,15] bool
    n("Or", ["is_crow", "is_ccol"], "cross")           # [1,1,15,15] bool

    # ---- blue mask (input channel 1) ----
    n("Slice", ["input", "blue_st", "blue_en", "blue_ax"], "blue_f")  # 15x15 f32
    n("Greater", ["blue_f", "half"], "is_blue")        # [1,1,15,15] bool

    # ---- label map: cyan base, pink on cross, blue on box (priority) ----
    n("Where", ["cross", "pink", "cyan"], "L_a")       # [1,1,15,15] uint8
    n("Where", ["is_blue", "blue", "L_a"], "L15")      # blue overrides
    n("Pad", ["L15", "padpads", "padval"], "L", mode="constant")  # 30x30 u8
    n("Equal", ["L", "chan"], "output")                # -> free BOOL output

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
