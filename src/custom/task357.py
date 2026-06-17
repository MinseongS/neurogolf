"""task357 (ARC-AGI e179c5f4) — "bounce" zigzag ray.

Rule (from the ARC-GEN generator `common.bounce`, verified fresh):
  Grid is HEIGHT=10 rows by WIDTH=W cols, W in [2,10], anchored top-left of the
  30x30 canvas.  INPUT is all BLACK(0) with a single BLUE(1) pixel at the
  bottom-left cell (row 9, col 0).  OUTPUT is all CYAN(8) background with a BLUE(1)
  zig-zag "bounce" path: starting at (row 9, col 0) the path goes UP one row each
  step while its column bounces back and forth between 0 and W-1 (triangle wave).
  Out-of-grid cells (rows>=10 or cols>=W) are all-zero (no channel set).

  The path column for row r is a triangle wave of the step s = 9 - r with
  period 2*(W-1):
      pc(r) = (W-1) - | (s mod 2*(W-1)) - (W-1) |.
  The whole output is a function of W ONLY (the single input pixel is fixed at
  (9,0)); W is recovered as the number of occupied input columns.

Construction (no full-canvas planes for the per-row path):
  * W = number of columns with any input channel set (ReduceMax over ch+rows,
    then ReduceSum over cols).
  * pc is a per-ROW vector [1,1,30,1] (only rows 0..9 valid); built from the
    1-D row ramp via Mod + abs.  rows>=10 get a sentinel column (99) so no
    column ever matches => those rows are all-zero in the output.
  * path = Equal(colramp[1,1,1,30], pc[1,1,30,1]) -> ONE bool [1,1,30,30].
  * ingrid = (row<10) AND (col<W); since pc in [0,W-1] for valid rows, path is
    automatically inside the grid.
  * label L (uint8 [1,1,30,30]) = Where(path, 1, Where(ingrid, 8, 99)).
  * output = Equal(L, arange[1,10,1,1]) -> [1,10,30,30] BOOL routed into the
    FREE output (off-grid sentinel 99 matches no channel => all-zero).
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION


def build(task):
    inits, nodes = [], []

    def init(name, arr, dt):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dt), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    H = TensorProto.FLOAT16
    U8 = TensorProto.UINT8
    B = TensorProto.BOOL

    # ---- recover W (number of occupied columns) ----
    # occupancy over channels+rows -> [1,1,1,30] then sum cols -> scalar W.
    n("ReduceMax", ["input"], "colmax", axes=[1, 2], keepdims=1)   # [1,1,1,30] fp32 (0/1)
    n("ReduceSum", ["colmax"], "Wf32", axes=[3], keepdims=1)        # scalar fp32 = W
    n("Cast", ["Wf32"], "Wf", to=H)                                # fp16 scalar W

    # ---- 1-D ramps (fp16, small ints exact) ----
    init("RIv", np.arange(30, dtype=np.float16).reshape(1, 1, 30, 1), np.float16)  # row idx
    init("CIv", np.arange(30, dtype=np.float16).reshape(1, 1, 1, 30), np.float16)  # col idx
    init("nine", np.array(9.0, np.float16), np.float16)
    init("two", np.array(2.0, np.float16), np.float16)
    init("oneH", np.array(1.0, np.float16), np.float16)
    init("tenH", np.array(10.0, np.float16), np.float16)
    init("sent", np.array(99.0, np.float16), np.float16)

    # period m = 2*(W-1)
    n("Sub", ["Wf", "oneH"], "Wm1")          # W-1 (scalar fp16)
    n("Mul", ["two", "Wm1"], "m")            # 2*(W-1)

    # step s = 9 - r (per-row vector [1,1,30,1])
    n("Sub", ["nine", "RIv"], "srow")        # [1,1,30,1]
    # q = s mod m  (Mod fp16, fmod semantics; s>=0 for rows 0..9)
    n("Mod", ["srow", "m"], "q", fmod=1)     # [1,1,30,1]
    # pc = (W-1) - |q - (W-1)|
    n("Sub", ["q", "Wm1"], "qd")
    n("Abs", ["qd"], "qabs")
    n("Sub", ["Wm1", "qabs"], "pc_raw")      # path column per row [1,1,30,1]

    # rows >= 10 -> sentinel column (no col matches)
    n("Less", ["RIv", "tenH"], "rowin")      # bool [1,1,30,1] (r<10)
    n("Where", ["rowin", "pc_raw", "sent"], "pc")   # [1,1,30,1]

    # ---- path mask = (col == pc) ; broadcast to [1,1,30,30] ----
    n("Equal", ["CIv", "pc"], "path")        # bool [1,1,30,30]

    # ---- label L (uint8): path->1(blue), in-grid->8(cyan), else 99 ----
    # colcyan is a tiny [1,1,1,30] base (8 inside the W cols, 99 outside);
    # gating it by rowin (r<10) is the only [1,1,30,30] materialisation besides
    # `path` and the final L, so the whole net carries just 3 full planes.
    init("one_u8", np.array(1, np.uint8), np.uint8)
    init("eight_u8", np.array(8, np.uint8), np.uint8)
    init("sent_u8", np.array(99, np.uint8), np.uint8)
    init("chan_u8", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)

    n("Less", ["CIv", "Wf"], "colin")                        # bool [1,1,1,30] (c<W)
    n("Where", ["colin", "eight_u8", "sent_u8"], "colcyan")  # uint8 [1,1,1,30]
    n("Where", ["rowin", "colcyan", "sent_u8"], "csent")     # uint8 [1,1,30,30]
    n("Where", ["path", "one_u8", "csent"], "L")             # uint8 [1,1,30,30]
    n("Equal", ["L", "chan_u8"], "output")                   # [1,10,30,30] BOOL (free)

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task357", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
