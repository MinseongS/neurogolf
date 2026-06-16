"""task033 (ARC-AGI 1e32b0e9) — "stamp the top-left cell's shape (in linecolor) into every cell".

Rule (from the generator):
  A 17x17 grid is a 3x3 arrangement of 5x5 cells separated by full lines of `linecolor`
  at rows/cols 5 and 11 (hollywood_squares border).  Inside each cell a small shape made
  of `color` pixels is drawn (inner positions r,c in {1,2,3} relative to the cell origin
  mr*6, mc*6).  The reference shape = the `color` pixels in the top-left cell (megacell 0,0),
  occupying canvas rows/cols 1..3.
  OUTPUT = INPUT, plus: that reference 3x3 pattern P is stamped (in `linecolor`) into ALL 9
  cells (drawn first), then the original `color` pixels are overlaid on top (drawn last, so
  color wins where a stamp and a color pixel coincide).

  Equivalently, per pixel:
    - color pixels of the input are UNCHANGED (output color channel == input color channel).
    - a pixel becomes `linecolor` iff it was already a line OR it is a stamp position that is
      NOT a color pixel.
  So the ONLY change vs input is: certain background pixels (the stamp positions that are not
  color pixels) flip from background(0) -> linecolor.

Encoding (floor-break via Where, route the 10-ch expansion into the FREE output):
  newpix[R,C] = (M[R,C] AND NOT color_plane[R,C])   on the 17x17 canvas, where
    color_plane = nonbg AND NOT line_mask   (line_mask is a FIXED geometric constant:
                                             rows/cols 5,11),  nonbg = (input ch0 == 0),
    M = Srow @ P @ Scol^T   (boolean double-MatMul tiling, task250 idiom) with
      P = color_plane[1:4, 1:4]  (the top-left cell shape, 3x3),
      Srow[R,dr] = (R % 6 == dr+1) for dr in {0,1,2}  (places the 3 pattern rows into the
                   3 cell row-blocks {1,2,3},{7,8,9},{13,14,15}; 0 on gaps/lines/off-grid),
      Scol[C,dc] likewise.
  Pad newpix to 30x30 (fp16, then Greater->bool cond) and
    output = Where(newpix_cond, linecolor_onehot, input)
  where linecolor_onehot = input[:, :, 5:6, 0:1]  ([1,10,1,1], the one-hot of `linecolor`
  read off a guaranteed line pixel).  Where broadcasts the [1,1,30,30] cond and [1,10,1,1]
  value against the FREE [1,10,30,30] input -> output is the only 10-channel tensor and it is
  free; the dominant intermediate is the [1,1,30,30] padded plane (~1800B fp16).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64

W = 17  # active 17x17 canvas (grid is always 17x17 for this task)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- background channel slice (ch0) on the 17x17 canvas ----------------
    init("ch0_s", np.array([0, 0, 0], np.int64), np.int64)
    init("ch0_e", np.array([1, W, W], np.int64), np.int64)
    init("ch0_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "ch0_s", "ch0_e", "ch0_ax"], "ch0")  # [1,1,W,W] f32

    # nonbg = 1 - ch0  (bool: a pixel is non-background)
    init("ONEH", np.array(1.0, np.float16), np.float16)
    n("Cast", ["ch0"], "ch0_16", to=F16)
    n("Sub", ["ONEH", "ch0_16"], "nonbg")  # [1,1,W,W] fp16, {0,1}

    # ---- line_mask : FIXED geometric mask of the border lines (rows/cols 5,11)
    line_mask = np.zeros((1, 1, W, W), dtype=np.float16)
    for k in (5, 11):
        line_mask[0, 0, k, :] = 1.0
        line_mask[0, 0, :, k] = 1.0
    init("notline", 1.0 - line_mask, np.float16)        # fp16 {0,1}

    # color_plane = nonbg AND NOT line  (the `color` pixels of the input)
    n("Mul", ["nonbg", "notline"], "color_plane")       # [1,1,W,W] fp16 {0,1}

    # ---- P = top-left cell shape = color_plane[1:4, 1:4]  (3x3) -------------
    init("P_s", np.array([1, 1], np.int64), np.int64)
    init("P_e", np.array([4, 4], np.int64), np.int64)
    init("P_ax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["color_plane", "P_s", "P_e", "P_ax"], "P")  # [1,1,3,3] fp16

    # ---- tiling matrices Srow, Scol : [1,1,W,3] and [1,1,3,W] ---------------
    # Srow[R,dr] = 1 iff R % 6 == dr+1  (dr in {0,1,2}); 0 on gaps/lines/off-grid.
    Srow = np.zeros((1, 1, W, 3), dtype=np.float16)
    for R in range(W):
        for dr in range(3):
            if R % 6 == dr + 1:
                Srow[0, 0, R, dr] = 1.0
    ScolT = np.zeros((1, 1, 3, W), dtype=np.float16)
    for C in range(W):
        for dc in range(3):
            if C % 6 == dc + 1:
                ScolT[0, 0, dc, C] = 1.0
    init("Srow", Srow, np.float16)     # [1,1,W,3]
    init("ScolT", ScolT, np.float16)   # [1,1,3,W]

    # M = Srow @ P @ ScolT  -> [1,1,W,W] fp16 {0,1} (exactly one term per (R,C))
    n("MatMul", ["Srow", "P"], "SP")          # [1,1,W,3]@[1,1,3,3] -> [1,1,W,3]
    n("MatMul", ["SP", "ScolT"], "M")         # [1,1,W,3]@[1,1,3,W] -> [1,1,W,W]

    # ---- newpix = M AND NOT color_plane ------------------------------------
    # diff = M - color_plane in {1, 0, -1}; newpix = (diff > 0), which is 1 only
    # at stamp positions that are NOT color pixels (M=1, color=0).
    init("ZEROH", np.array(0.0, np.float16), np.float16)
    n("Sub", ["M", "color_plane"], "diff")          # [1,1,W,W] fp16 {1,0,-1}
    n("Greater", ["diff", "ZEROH"], "newpix_b")     # [1,1,W,W] bool

    # ---- pad to 30x30 (uint8, half the bytes of fp16) then -> bool cond -----
    U8 = TensorProto.UINT8
    n("Cast", ["newpix_b"], "newpix_u8", to=U8)      # [1,1,W,W] uint8 {0,1}
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64), np.int64)
    init("ZEROU8", np.array(0, np.uint8), np.uint8)
    n("Pad", ["newpix_u8", "pads", "ZEROU8"], "newpix30", mode="constant")  # [1,1,30,30] uint8
    n("Cast", ["newpix30"], "cond", to=BOOL)         # [1,1,30,30] bool

    # ---- linecolor one-hot read off a guaranteed line pixel (row 5, col 0) --
    init("lc_s", np.array([0, 5, 0], np.int64), np.int64)
    init("lc_e", np.array([10, 6, 1], np.int64), np.int64)
    init("lc_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "lc_s", "lc_e", "lc_ax"], "lc_onehot")  # [1,10,1,1] f32

    # ---- output = Where(cond, linecolor_onehot, input) : FREE [1,10,30,30] ---
    n("Where", ["cond", "lc_onehot", "input"], "output")

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task033", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
