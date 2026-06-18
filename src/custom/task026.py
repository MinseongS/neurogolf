"""task026 (ARC-AGI 1b2d62fb) — left/right half intersection -> cyan.

Rule (from the ARC-GEN generator `grid_intersect`, verified fresh):
  Input is a (2*width+1) x height grid, width=3 height=5 => 7 cols x 5 rows,
  placed at the top-left of the 30x30 canvas.  Colours:
    background fill = maroon (channel 9)
    placed pixels   = black  (channel 0)
    separator column at col `width` (=3) = blue (channel 1)
  The two halves are cols 0..2 (left) and cols 4..6 (right); col 3 is the blue
  separator.  Output is a width x height = 3 cols x 5 rows grid:
      output[r][c] = cyan(8)  iff  left[r][c] != maroon  AND  right[r][c] != maroon
                   = black(0) otherwise
  i.e. a cell is cyan iff BOTH the left-half cell and the matching right-half
  cell (col c+4) are non-background.  Off the 3x5 output grid everything is 0.

Encoding (no full-canvas plane ever materialises):
  maroon presence = input channel 9.  Slice the left half (rows 0:5, cols 0:3)
  and right half (rows 0:5, cols 4:7) of channel 9 -> two [1,1,5,3] fp32 blocks.
  cyan = Equal(Lm + Rm, 0)  (both maroon-channel values are 0 => both non-bg).
  Build a tiny [1,1,5,3] colour-index block M = cyan * 8 (cyan->8, else 0), one-hot
  it with Equal(M, arange[1,10,1,1]) -> [1,10,5,3] bool, cast uint8, and Pad the
  block spatially to [1,10,30,30] (zeros) directly as the FREE graph output.
  The padded zeros make every off-grid cell all-zero across channels (out>0 False),
  matching the expected one-hot.  Dominant intermediate is the 150-elem uint8
  block (150 B).
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

S = 30
H = 5   # output rows  (= height)
W = 3   # output cols  (= width)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dt):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dt), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    U8 = TensorProto.UINT8

    # ---- slice maroon channel (ch9) left & right halves --------------------
    # left  : rows 0:5, cols 0:3, channel 9
    init("lstart", np.array([9, 0, 0], np.int64), np.int64)
    init("lend",   np.array([10, H, W], np.int64), np.int64)
    init("axc23",  np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "lstart", "lend", "axc23"], "Lm")   # [1,1,5,3] fp32

    # right : rows 0:5, cols 4:7, channel 9
    init("rstart", np.array([9, 0, W + 1], np.int64), np.int64)
    init("rend",   np.array([10, H, 2 * W + 1], np.int64), np.int64)
    n("Slice", ["input", "rstart", "rend", "axc23"], "Rm")   # [1,1,5,3] fp32

    # cyan iff both maroon-channel cells are 0 (both halves non-background).
    # maroon channel is {0,1}; cyan = NOT(Lm>0 OR Rm>0).  Stay in bool (15 B
    # planes) instead of an fp32 sum (60 B).
    init("zero", np.array(0.0, np.float32), np.float32)
    n("Greater", ["Lm", "zero"], "Lp")                        # [1,1,5,3] bool
    n("Greater", ["Rm", "zero"], "Rp")                        # [1,1,5,3] bool
    n("Or", ["Lp", "Rp"], "either")                           # [1,1,5,3] bool
    n("Not", ["either"], "cyan")                              # [1,1,5,3] bool

    # one-hot 10-ch block in ONE Where: cyan -> ch8 one-hot, else ch0 one-hot.
    # broadcasts cond[1,1,5,3] against two [1,10,1,1] uint8 colour one-hots ->
    # [1,10,5,3] uint8 directly (no bool/fp intermediate, no per-channel cast).
    cyan_oh = np.zeros((1, 10, 1, 1), np.uint8); cyan_oh[0, 8] = 1
    black_oh = np.zeros((1, 10, 1, 1), np.uint8); black_oh[0, 0] = 1
    init("cyan_oh", cyan_oh, np.uint8)
    init("black_oh", black_oh, np.uint8)
    n("Where", ["cyan", "cyan_oh", "black_oh"], "oh8")        # [1,10,5,3] uint8
    init("pad", np.array([0, 0, 0, 0, 0, 0, S - H, S - W], np.int64), np.int64)
    init("padv", np.array(0, np.uint8), np.uint8)
    n("Pad", ["oh8", "pad", "padv"], "output")                # [1,10,30,30] uint8 FREE

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", U8, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task026", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 13)])
