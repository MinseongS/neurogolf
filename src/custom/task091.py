"""Task 091 (3f7978a0): "glowsticks" zoom-crop.

Rule (from ARC-GEN generator): the input is an H x W grid (H,W in 9..15) with
scattered cyan(8) pixels plus a marked "zoom box": its four corners are cyan and
its left/right edges (interior rows) are grey(5).  The OUTPUT is exactly the
sub-grid of the input inside that box:

    output = input[row : row+zoom_h, col : col+zoom_w]

(verified byte-exact on 2000 fresh instances).  The box is recovered purely from
the grey edges (grey appears ONLY on the box's two vertical edges, interior
rows, and zoom_h >= 3 is guaranteed so grey always exists):

    col      = min grey column           col_hi = max grey column
    row      = min grey row - 1           row_hi = max grey row + 1
    zoom_w   = col_hi - col + 1           zoom_h = row_hi - row + 1

(verified: 0 mismatches / 5000 fresh).

Floor-break design (Tier B on a tiny canvas), mem 6344 / 90 params / 16.23 pts:
* All input content lives in the top-left 15x15 corner (H,W <= 15) and the whole
  output (zoom_h,zoom_w <= 14) lands in the top-left corner, so every per-cell
  intermediate is a 15x15 plane -- never a [1,10,30,30] stack.
* Only colours {0(black),5(grey),8(cyan)} ever occur (verified, 20k instances),
  so the per-cell colour is captured by TWO single-channel 15x15 slices of the
  input (channel 5 = grey, channel 8 = cyan), cast to uint8 (225 B each). The
  two float slices (900 B each) are the irreducible entry gateway: Slice keeps
  the input float dtype, so they cannot be made uint8 without first casting the
  whole [1,10,30,30] input (9000 B, worse).
* Box geometry comes purely from the grey edges: rowhas/colhas = ReduceMax of
  channel-5 over each axis; row=min_grey_row-1, col=min_grey_col, and the
  extents from max-min (grey only marks the box's vertical edges, zoom_h>=3
  guarantees grey exists). All box params are float SCALARS.
* Crop+shift the grey & cyan bit-planes to the top-left with two Gathers each
  (rows then cols), indices ri=row+i, ci=col+j (clipped valid).
* Build a uint8 label map L (8=cyan, 5=grey, 0=black-in-region, 10=outside the
  zoom_h x zoom_w region), Pad to 30x30 with sentinel 10, then final
  Equal(L, arange[0..9]) writes straight into the FREE bool `output`. Sentinel
  >=10 never matches channels 0..9 -> all-false (all-zero), exactly as the
  scorer expects for cells outside the output grid.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

WORK = 15  # working canvas side: H,W <= 15 so all content is in the top-left.


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- constants ----
    # Only colours {0(black), 5(grey), 8(cyan)} ever occur (verified on 20k
    # fresh instances), so the colour-index plane = 5*ch5 + 8*ch8 is built from
    # two single-channel 15x15 slices -- no full-grid Conv needed.
    init("c5_st", np.array([5, 0, 0], np.int64), np.int64)
    init("c5_en", np.array([6, WORK, WORK], np.int64), np.int64)
    init("c8_st", np.array([8, 0, 0], np.int64), np.int64)
    init("c8_en", np.array([9, WORK, WORK], np.int64), np.int64)
    init("crop_ax", np.array([1, 2, 3], np.int64), np.int64)
    init("five_u", np.array(5, np.uint8), np.uint8)
    init("eight_u", np.array(8, np.uint8), np.uint8)
    init("zero_u", np.array(0, np.uint8), np.uint8)
    init("ar_r", np.arange(WORK, dtype=np.float32).reshape(1, 1, WORK, 1),
         np.float32)
    init("ar_c", np.arange(WORK, dtype=np.float32).reshape(1, 1, 1, WORK),
         np.float32)
    init("ar_i", np.arange(WORK, dtype=np.float32), np.float32)  # [WORK] for idx
    init("big", np.array(99.0, np.float32), np.float32)
    init("neg", np.array(-1.0, np.float32), np.float32)
    init("one", np.array(1.0, np.float32), np.float32)
    init("zero_f", np.array(0.0, np.float32), np.float32)
    init("max_f", np.array(float(WORK - 1), np.float32), np.float32)
    init("chan_u", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    init("sent", np.array(10, np.uint8), np.uint8)
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)

    # ---- grey / cyan single-channel slices (15x15) ----
    # Only colours {0,5,8} occur, and they are mutually exclusive per cell.
    n("Slice", ["input", "c5_st", "c5_en", "crop_ax"], "c5")  # grey 0/1 [1,1,W,W] f32
    n("Slice", ["input", "c8_st", "c8_en", "crop_ax"], "c8")  # cyan 0/1 [1,1,W,W] f32
    n("Cast", ["c5"], "c5u", to=TensorProto.UINT8)           # uint8 0/1 grey
    n("Cast", ["c8"], "c8u", to=TensorProto.UINT8)           # uint8 0/1 cyan

    # ---- grey-edge plane (c5 is already the grey 0/1 mask) -> box geometry ----
    n("ReduceMax", ["c5"], "rowhas", axes=[3], keepdims=1)  # [1,1,WORK,1] f32
    n("ReduceMax", ["c5"], "colhas", axes=[2], keepdims=1)  # [1,1,1,WORK] f32

    # grey row range: min..max of rows carrying grey
    init("half", np.array(0.5, np.float32), np.float32)
    n("Greater", ["rowhas", "half"], "rmask")            # bool rows with grey
    n("Greater", ["colhas", "half"], "cmask")            # bool cols with grey
    n("Where", ["rmask", "ar_r", "big"], "rlo_w")        # idx where grey else 99
    n("ReduceMin", ["rlo_w"], "grlo", keepdims=0)        # scalar min grey row
    n("Where", ["rmask", "ar_r", "neg"], "rhi_w")        # idx where grey else -1
    n("ReduceMax", ["rhi_w"], "grhi", keepdims=0)        # scalar max grey row
    n("Where", ["cmask", "ar_c", "big"], "clo_w")
    n("ReduceMin", ["clo_w"], "gclo", keepdims=0)        # scalar min grey col
    n("Where", ["cmask", "ar_c", "neg"], "chi_w")
    n("ReduceMax", ["chi_w"], "gchi", keepdims=0)        # scalar max grey col

    # box origin / extent (float scalars, integer-valued)
    n("Sub", ["grlo", "one"], "row")                     # row = grlo - 1
    n("Add", ["grhi", "one"], "rowhi")                   # rowhi = grhi + 1
    n("Sub", ["rowhi", "row"], "h_m1")                   # zoom_h - 1
    n("Add", ["h_m1", "one"], "zh")                      # zoom_h
    n("Sub", ["gchi", "gclo"], "w_m1")                   # zoom_w - 1
    n("Add", ["w_m1", "one"], "zw")                      # zoom_w
    # col origin
    n("Identity", ["gclo"], "col")

    # ---- row / col gather indices ri = row + i, ci = col + j ----
    n("Add", ["ar_i", "row"], "ri_f")                    # [WORK] f32
    n("Add", ["ar_i", "col"], "ci_f")                    # [WORK] f32
    n("Clip", ["ri_f", "zero_f", "max_f"], "ri_c")       # valid 0..WORK-1 (float)
    n("Clip", ["ci_f", "zero_f", "max_f"], "ci_c")
    n("Cast", ["ri_c"], "ri", to=TensorProto.INT64)
    n("Cast", ["ci_c"], "ci", to=TensorProto.INT64)

    # ---- crop+shift grey & cyan bit-planes to the top-left (uint8 15x15) ----
    n("Gather", ["c5u", "ri"], "c5r", axis=2)            # rows
    n("Gather", ["c5r", "ci"], "c5g", axis=3)            # cols -> grey crop
    n("Gather", ["c8u", "ri"], "c8r", axis=2)
    n("Gather", ["c8r", "ci"], "c8g", axis=3)            # cyan crop

    # ---- in-grid rectangle (i < zoom_h) & (j < zoom_w) ----
    n("Less", ["ar_r", "zh"], "r_in")                    # [1,1,WORK,1] bool
    n("Less", ["ar_c", "zw"], "c_in")                    # [1,1,1,WORK] bool
    n("And", ["r_in", "c_in"], "rect")                   # [1,1,WORK,WORK] bool

    # ---- label map: 8=cyan, 5=grey, 0=black-in-region, 10=outside ----
    init("u1", np.array(1, np.uint8), np.uint8)
    n("Equal", ["c8g", "u1"], "isC")                     # bool cyan crop
    n("Equal", ["c5g", "u1"], "isG")                     # bool grey crop
    n("Where", ["isG", "five_u", "zero_u"], "L0")        # 5 if grey else 0
    n("Where", ["isC", "eight_u", "L0"], "L1")           # 8 if cyan (overrides)
    n("Where", ["rect", "L1", "sent"], "Lw")             # sentinel 10 outside region
    n("Pad", ["Lw", "padpads", "padval"], "L", mode="constant")  # uint8 [1,1,30,30]
    n("Equal", ["L", "chan_u"], "output")                # -> free BOOL output

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
