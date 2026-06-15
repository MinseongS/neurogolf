"""Task 035 (1f642eb9): project border pixels onto a cyan pool's edges.

Rule (from ARC-GEN generator, verified 20000/0 fresh):
  The grid (size 10) holds a cyan(8) rectangle ("pool"): rows 3..r1 (r1 in 4..7),
  cols c0..5 (c0 in 2..5).  Scattered non-cyan colour pixels sit on the four grid
  borders, each ALIGNED with a pool row or column.  In the output every border
  pixel is additionally copied inward onto the nearest pool edge cell:
    out[3 , c] = input[0 , c]   (top  border -> pool top    row, where nonzero)
    out[r1, c] = input[9 , c]   (bot  border -> pool bottom row, where nonzero)
    out[r , c0]= input[r , 0]   (left border -> pool left   col, where nonzero)
    out[r , 5] = input[r , 9]   (right border-> pool right  col, where nonzero)
  Everything else equals the input.  The four projection lines never collide
  (verified 50000/0), so the four overrides commute.  The projected colour is
  simply a COPY of the corresponding border row/column -- no colour detection of
  the border pixels is needed; only the two variable pool bounds r1, c0 (scalars).

Encoding (Tier B label-map, tiny 10x10 canvas):
  V = per-cell input colour index (one Conv [0..9] over the one-hot input,
      Cast->uint8, Slice to the active 10x10 corner).  toprow/botrow/leftcol/
      rightcol are 1-D slices of V (the four border lines).  Build the label map
      L by four broadcasting Where overrides (each: line-position mask AND
      projected-value != 0), then Pad to 30x30 with sentinel 10 and a single
      final Equal(L, arange[0..9]) writing straight into the free BOOL output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

WORK = 10  # active grid side (size = 10)


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
    # colour-index Conv weight: 1x1 kernel, channel k -> weight k  (one-hot -> k)
    init("kw", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    init("half", np.array(0.5, np.float32), np.float32)
    init("u0", np.array(0, np.uint8), np.uint8)

    # row / col index vectors on the 10x10 canvas
    init("ar_row", np.arange(WORK, dtype=np.float32).reshape(1, 1, WORK, 1),
         np.float32)
    init("ar_col", np.arange(WORK, dtype=np.float32).reshape(1, 1, 1, WORK),
         np.float32)
    init("three", np.array(3.0, np.float32), np.float32)   # pool top row (fixed)
    init("five", np.array(5.0, np.float32), np.float32)    # pool right col (fixed)

    # slices of V (uint8 10x10) for the four border lines
    init("r0_st", np.array([0], np.int64), np.int64)
    init("r0_en", np.array([1], np.int64), np.int64)
    init("r9_st", np.array([WORK - 1], np.int64), np.int64)
    init("r9_en", np.array([WORK], np.int64), np.int64)
    init("ax2", np.array([2], np.int64), np.int64)
    init("ax3", np.array([3], np.int64), np.int64)
    # crop the 30x30 colour-index plane to the active 10x10 corner
    init("crop_st", np.array([0, 0], np.int64), np.int64)
    init("crop_en", np.array([WORK, WORK], np.int64), np.int64)
    init("crop_ax", np.array([2, 3], np.int64), np.int64)
    # pad L (10x10 uint8) -> 30x30 with sentinel 10
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64),
         np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)

    # ---- V = per-cell input colour index (uint8 10x10) ----
    # Conv emits the full 30x30 colour index (f32, unavoidable: input is 30x30),
    # but we Slice to the active 10x10 corner BEFORE casting so the uint8 plane is
    # 100B not 900B.
    n("Conv", ["input", "kw"], "Vbig")                 # [1,1,30,30] f32 (one-hot->k)
    n("Slice", ["Vbig", "crop_st", "crop_en", "crop_ax"], "Vf")  # [1,1,10,10] f32
    n("Cast", ["Vf"], "V", to=TensorProto.UINT8)       # [1,1,10,10] uint8

    # ---- variable pool bounds (scalars): r1 = max cyan row, c0 = min cyan col ----
    # cyan presence from channel 8, sliced to the active 10x10 corner only
    # (axes [1,2,3] -> [1,1,10,10] = 400B, not the full 30x30 = 3600B).
    init("cy_st", np.array([8, 0, 0], np.int64), np.int64)
    init("cy_en", np.array([9, WORK, WORK], np.int64), np.int64)
    init("cy_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "cy_st", "cy_en", "cy_ax"], "cy")   # [1,1,10,10] f32 cyan
    n("ReduceMax", ["cy"], "cy_row", axes=[3], keepdims=1)   # [1,1,10,1] row has cyan
    n("ReduceMax", ["cy"], "cy_col", axes=[2], keepdims=1)   # [1,1,1,10] col has cyan
    init("big", np.array(99.0, np.float32), np.float32)
    init("neg", np.array(-1.0, np.float32), np.float32)
    n("Greater", ["cy_row", "half"], "cyr_b")
    n("Greater", ["cy_col", "half"], "cyc_b")
    # r1 = max row index with cyan
    n("Where", ["cyr_b", "ar_row", "neg"], "r_idx")          # [1,1,10,1]
    n("ReduceMax", ["r_idx"], "r1f", keepdims=1)             # [1,1,1,1]
    # c0 = min col index with cyan
    n("Where", ["cyc_b", "ar_col", "big"], "c_idx")          # [1,1,1,10]
    n("ReduceMin", ["c_idx"], "c0f", keepdims=1)             # [1,1,1,1]

    # ---- line-position masks on the 10x10 canvas (bool) ----
    n("Equal", ["ar_row", "three"], "is_top")               # [1,1,10,1] row==3
    n("Equal", ["ar_row", "r1f"], "is_bot")                 # [1,1,10,1] row==r1
    n("Equal", ["ar_col", "c0f"], "is_left")                # [1,1,1,10] col==c0
    n("Equal", ["ar_col", "five"], "is_right")              # [1,1,1,10] col==5

    # ---- the four border lines (copies of V's edge rows/cols), uint8 ----
    n("Slice", ["V", "r0_st", "r0_en", "ax2"], "toprow")    # [1,1,1,10]
    n("Slice", ["V", "r9_st", "r9_en", "ax2"], "botrow")    # [1,1,1,10]
    n("Slice", ["V", "r0_st", "r0_en", "ax3"], "leftcol")   # [1,1,10,1]
    n("Slice", ["V", "r9_st", "r9_en", "ax3"], "rightcol")  # [1,1,10,1]

    # nonzero-projected-value gates (bool): ORT Greater rejects uint8, so use
    # Equal(line, 0) then Not.
    n("Equal", ["toprow", "u0"], "top_z"); n("Not", ["top_z"], "top_nz")
    n("Equal", ["botrow", "u0"], "bot_z"); n("Not", ["bot_z"], "bot_nz")
    n("Equal", ["leftcol", "u0"], "left_z"); n("Not", ["left_z"], "left_nz")
    n("Equal", ["rightcol", "u0"], "right_z"); n("Not", ["right_z"], "right_nz")

    # combined override conditions (broadcast to [1,1,10,10])
    n("And", ["is_top", "top_nz"], "cond_top")
    n("And", ["is_bot", "bot_nz"], "cond_bot")
    n("And", ["is_left", "left_nz"], "cond_left")
    n("And", ["is_right", "right_nz"], "cond_right")

    # ---- build L by four commuting Where overrides (start from V) ----
    n("Where", ["cond_top", "toprow", "V"], "L1")
    n("Where", ["cond_bot", "botrow", "L1"], "L2")
    n("Where", ["cond_left", "leftcol", "L2"], "L3")
    n("Where", ["cond_right", "rightcol", "L3"], "L10")     # [1,1,10,10] uint8

    # ---- pad to 30x30 (sentinel 10 outside) and final Equal -> BOOL output ----
    n("Pad", ["L10", "padpads", "padval"], "L", mode="constant")  # [1,1,30,30] u8
    n("Equal", ["L", "chan"], "output")                     # -> free BOOL output

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
