"""Task 075 (363442ee): stamp the size x size template into marked grid blocks.

Rule (from ARC-GEN generator, size = 3; grid 9 rows x 13 cols):
  Left columns 0..2 rows 0..2 hold a 3x3 colour TEMPLATE P.  Column 3 is a full-
  height gray(5) separator.  Columns 4..12 form a 3x3 grid of 3x3 blocks; block
  (R,C) (R,C in 0..2) occupies rows 3R..3R+2, cols 3C+4..3C+6.  The INPUT marks
  selected blocks with a single blue(1) pixel at [3R+1, 3C+5].  The OUTPUT keeps
  the template + separator and copies P into every MARKED block (unmarked blocks
  stay background).

  output[row,col] = identity input for cols 0..3, and for each right-region cell
  (col 4..12): P[row%3,(col-4)%3] if its block is marked else background.

Encoding (Tier-S spatial copy via one data-dependent flat Gather):
  EVERY output cell is a copy of an input cell in columns 0..3 (the template P and
  the gray separator both live there; unmarked / right cells map to a guaranteed
  in-grid background cell row3/col0).  So the gather SOURCE is just input cols 0..3
  rows 0..8 -> [1,10,9,4] (1440B fp32 entry plane), cast fp16, flattened to
  [1,10,36].  Per output cell (117 of them) the source flat-index is
    base[p] + blue_marker[block(p)] * (template_flat(p) - base[p])
  base = identity for cols 0..3 else BG; blue_marker gathered from the blue channel
  at the block's marker pixel (a tiny [117] fp16 blue slice).  One Gather pulls the
  9x13 active output, then Pad to 30x30 (the free output).  No [1,10,30,30]
  intermediate is ever materialised; everything after the entry slice is fp16.
  Verified evaluate() ok + fresh 200/200.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

H, W = 9, 13          # active output region
SW = 4                # source width: input columns 0..3
NS = H * SW           # 36 source cells
NO = H * W            # 117 output cells
BG = 3 * SW + 0       # in-grid background sentinel (row 3, col 0) -> always bg


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, outs, **attrs):
        if isinstance(outs, str):
            outs = [outs]
        nodes.append(helper.make_node(op, ins, outs, **attrs))
        return outs[0]

    # ---- per-cell index recipe over the 36-cell source ----
    base = np.zeros(NO, dtype=np.float32)
    coef = np.zeros(NO, dtype=np.float32)
    mksrc = np.zeros(NO, dtype=np.int64)         # marker pixel (active 9x13 frame)
    for row in range(H):
        for col in range(W):
            p = row * W + col
            if col < SW:                          # left region: identity copy
                base[p] = row * SW + col
            else:                                 # right region: template / bg
                R, C = row // 3, (col - 4) // 3
                mr, mc = row % 3, (col - 4) % 3
                base[p] = BG
                coef[p] = (mr * SW + mc) - BG
                mksrc[p] = (3 * R + 1) * W + (3 * C + 5)

    init("base", base.reshape(1, 1, NO), np.float16)
    init("coef", coef.reshape(1, 1, NO), np.float16)
    init("mksrc", mksrc, np.int64)

    # slices / shapes
    init("ss", np.array([0, 0], np.int64), np.int64)
    init("se", np.array([H, SW], np.int64), np.int64)
    init("sa", np.array([2, 3], np.int64), np.int64)
    init("bs", np.array([0, 1, 0, 0], np.int64), np.int64)
    init("be", np.array([1, 2, H, W], np.int64), np.int64)
    init("ba", np.array([0, 1, 2, 3], np.int64), np.int64)
    init("src_shape", np.array([1, 10, NS], np.int64), np.int64)
    init("blue_shape", np.array([NO], np.int64), np.int64)
    init("idx_shape", np.array([H, W], np.int64), np.int64)
    # pad active [1,10,9,13] -> [1,10,30,30]
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - H, 30 - W], np.int64), np.int64)
    init("pad_val", np.array(0, np.float16), np.float16)

    # ---- entry: source = input cols 0..3, rows 0..8 ----
    n("Slice", ["input", "ss", "se", "sa"], "srcf")               # [1,10,9,4] f32
    n("Cast", ["srcf"], "srch", to=TensorProto.FLOAT16)           # [1,10,9,4] f16
    n("Reshape", ["srch", "src_shape"], "srcflat")                # [1,10,36] f16

    # ---- blue marker channel (active 9x13), tiny fp16 ----
    n("Slice", ["input", "bs", "be", "ba"], "bluef")              # [1,1,9,13] f32
    n("Cast", ["bluef"], "blueh", to=TensorProto.FLOAT16)         # f16
    n("Reshape", ["blueh", "blue_shape"], "bluev")                # [117] f16
    n("Gather", ["bluev", "mksrc"], "mk", axis=0)                 # [117] f16

    # ---- flat source index: base + mk*coef ----
    n("Mul", ["mk", "coef"], "mkc")                               # [1,1,117]
    n("Add", ["base", "mkc"], "fidxf")                            # [1,1,117]
    n("Reshape", ["fidxf", "idx_shape"], "fidx1d")                # [9,13]
    n("Cast", ["fidx1d"], "fidx", to=TensorProto.INT32)          # [9,13]

    # ---- gather output cells (2-D index -> [1,10,9,13]) then pad to 30x30 ----
    n("Gather", ["srcflat", "fidx"], "outact", axis=2)            # [1,10,9,13] f16
    n("Pad", ["outact", "pads", "pad_val"], "output")             # [1,10,30,30] f16

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
