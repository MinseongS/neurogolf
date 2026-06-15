"""Task 189 (ARC-AGI 7c008303): quadrant-legend recolor of a green stamp.

Generator (size is always 6 -> input 9x9, output 6x6):
  A 2x2 legend of four colors sits in a corner, separated from a size x size
  region by a cyan(8) cross at index 2 (a full cyan row and a full cyan column).
  Each green(3) pixel in the size x size region is recolored by the QUADRANT it
  falls in (r,c vs size//2=3 -> one of the four legend cells), then a global
  flip_horiz / flip_vert is applied to BOTH the grid and the output.

Because the flips transform grid and output identically, everything can be read
in the *flipped frame* with no un-flipping:
  out[R][C] (R,C in 0..5) = legend_cell[R//3][C//3]   if green at (R+orr, C+orc)
                          = 0 (background)             otherwise
where the green-block offset orr/orc in {0,3} and the legend corner lr/lc in
{0,7} are fixed by which side of the cyan cross the legend sits on -- i.e. by
flip_vert (full cyan row at index 2 vs 6) and flip_horiz (full cyan col 2 vs 6).
Verified exact on all 266 stored examples and 200/200 fresh arc-gen instances.

Memory floor-break (color-index label map + final Equal):
  Instead of materialising a [1,10,6,6] one-hot recolor stack, we collapse the
  legend to FOUR scalar color indices (L22idx[1,1,2,2] = sum_k k*onehot), expand
  them to a 6x6 quadrant grid (Kr @ . @ Kc on a single channel), mask by the
  green block, and emit a uint8 label L[6,6] (legend color where green, else 0).
  Padded to 30x30 (sentinel 10 outside the 6x6) the final op
      output = Equal(L, arange[1,10,1,1])
  writes straight into the free BOOL output -- no 10-channel plane after L22.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F = TensorProto.FLOAT
H = TensorProto.FLOAT16
U8 = TensorProto.UINT8
B = TensorProto.BOOL


SRC = 9  # the grid + legend live entirely in the top-left 9x9 corner


def _banks():
    # legend-corner extractors: [2(flip), 2(local), SRC] and [2, SRC, 2]
    Lr = np.zeros((2, 2, SRC), np.float32)
    Lc = np.zeros((2, SRC, 2), np.float32)
    # green-block extractors: [2, 6, SRC] and [2, SRC, 6]
    Sr = np.zeros((2, 6, SRC), np.float32)
    Sc = np.zeros((2, SRC, 6), np.float32)
    for q in range(2):
        Lr[0, q, 0 + q] = 1.0   # no flip: legend rows 0,1
        Lr[1, q, 7 + q] = 1.0   # flip:    legend rows 7,8
        Lc[0, 0 + q, q] = 1.0
        Lc[1, 7 + q, q] = 1.0
    for R in range(6):
        Sr[0, R, R + 3] = 1.0   # no flip: green rows 3..8
        Sr[1, R, R] = 1.0       # flip:    green rows 0..5
        Sc[0, R + 3, R] = 1.0
        Sc[1, R, R] = 1.0
    # legend 2x2 -> 6x6 quadrant expansion (flip-independent)
    Kr = np.zeros((6, 2), np.float32)
    Kc = np.zeros((2, 6), np.float32)
    for R in range(6):
        Kr[R, R // 3] = 1.0
        Kc[R // 3, R] = 1.0
    return Lr, Lc, Sr, Sc, Kr, Kc


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype=np.float32):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    # --- cyan(8) per-row / per-col counts -> vflip / hflip 0/1 scalars ---
    Wrow = np.zeros((1, 10, 1, 30), np.float32); Wrow[0, 8, 0, :] = 1.0
    init("Wrow", Wrow)
    Wcol = np.zeros((1, 10, 30, 1), np.float32); Wcol[0, 8, :, 0] = 1.0
    init("Wcol", Wcol)
    n("Conv", ["input", "Wrow"], "rowsum")                  # [1,1,30,1]
    n("Conv", ["input", "Wcol"], "colsum")                  # [1,1,1,30]

    # vflip = (cyan row count at index 6 == 9); hflip likewise on col 6.
    init("idx6", np.array(6, np.int64), dtype=np.int64)
    init("eight5", np.array(8.5, np.float32))
    n("Gather", ["rowsum", "idx6"], "row6", axis=2)         # [1,1,1,1]
    n("Gather", ["colsum", "idx6"], "col6", axis=3)         # [1,1,1,1]
    n("Greater", ["row6", "eight5"], "vbool")
    n("Greater", ["col6", "eight5"], "hbool")
    n("Cast", ["vbool"], "vflip4", to=TensorProto.INT64)
    n("Cast", ["hbool"], "hflip4", to=TensorProto.INT64)
    n("ReduceSum", ["vflip4"], "vflip", axes=[0, 1, 2, 3], keepdims=0)  # scalar
    n("ReduceSum", ["hflip4"], "hflip", axes=[0, 1, 2, 3], keepdims=0)  # scalar

    # --- pick flip-specific extractor matrices ---
    Lr, Lc, Sr, Sc, Kr, Kc = _banks()
    init("LrB", Lr); init("LcB", Lc); init("SrB", Sr); init("ScB", Sc)
    init("Kr", Kr, dtype=np.float16); init("Kc", Kc, dtype=np.float16)
    n("Gather", ["LrB", "vflip"], "lr", axis=0)            # [2,30]
    n("Gather", ["LcB", "hflip"], "lc", axis=0)            # [30,2]
    n("Gather", ["SrB", "vflip"], "sr", axis=0)            # [6,30]
    n("Gather", ["ScB", "hflip"], "sc", axis=0)            # [30,6]

    # --- legend 2x2 one-hot L22 -> 4 color indices L22idx [1,1,2,2] ---
    # Fold the channel-weighting (sum_k k*onehot) into the legend extraction so
    # the intermediate carries a single channel: kin = sum_k k*input[k] is a
    # [1,1,30,30] color-index plane, then lr @ kin @ lc -> [1,1,2,2] directly.
    # Slice the input to the 9x9 corner first, then a 1x1 conv collapses the 10
    # channels to a single color-index plane kin [1,1,9,9] (324 B).
    init("i_st", np.array([0, 0], np.int64), np.int64)
    init("i_en", np.array([SRC, SRC], np.int64), np.int64)
    init("i_ax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["input", "i_st", "i_en", "i_ax"], "in9")   # [1,10,SRC,SRC]
    Wk = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("Wk", Wk)
    n("Conv", ["in9", "Wk"], "kin")                        # [1,1,SRC,SRC] color idx
    n("MatMul", ["lr", "kin"], "lrow")                     # [1,1,2,SRC]
    n("MatMul", ["lrow", "lc"], "L22idx")                  # [1,1,2,2] color idx

    # --- expand 2x2 color indices to the 6x6 quadrant grid (single channel) ---
    n("Cast", ["L22idx"], "L22idxh", to=H)                 # [1,1,2,2] f16
    n("MatMul", ["Kr", "L22idxh"], "lemid")                # [1,1,6,2] f16
    n("MatMul", ["lemid", "Kc"], "leidx")                  # [1,1,6,6] f16 color idx

    # --- green-block mask gb [1,1,6,6], derived from the SAME color-index plane ---
    # green present at a cell <=> kin == 3. Extract the 6x6 green block, then
    # threshold: gb = (selected color-index == 3). One row/col selection picks
    # exactly one source cell per output cell, so the gathered value is that
    # cell's color index (3 iff green).
    n("MatMul", ["sr", "kin"], "krow")                     # [1,1,6,30] f32
    n("MatMul", ["krow", "sc"], "kblk")                    # [1,1,6,6] color idx
    init("two5", np.array(2.5, np.float32), np.float32)
    init("three5", np.array(3.5, np.float32), np.float32)
    n("Greater", ["kblk", "two5"], "g_lo")                 # > 2.5
    n("Less", ["kblk", "three5"], "g_hi")                  # < 3.5  -> == 3
    n("And", ["g_lo", "g_hi"], "gb")                       # [1,1,6,6] bool green

    # --- L = legend color where green, else 0; uint8 6x6 ---
    n("Cast", ["leidx"], "leidx_u8", to=U8)                # [1,1,6,6] uint8
    init("v0", np.array(0, np.uint8), np.uint8)
    n("Where", ["gb", "leidx_u8", "v0"], "Lwk")            # [1,1,6,6] uint8

    # --- pad 6x6 -> 30x30 (sentinel 10), final Equal -> free BOOL output ---
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 24, 24], np.int64), np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    n("Pad", ["Lwk", "padpads", "padval"], "L", mode="constant")  # [1,1,30,30]
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task189", [x], [y], inits)
    return helper.make_model(
        g, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
