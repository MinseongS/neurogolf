"""task149 (ARC-AGI 6773b310) — count pink pixels per 3x3 mini-block, blue if >=2.

Rule (from generator, verified fresh):
  Input is an 11x11 "hollywood_squares" board: a 3x3 arrangement of 3x3 mini-grids
  (minisize=3) separated by single cyan (8) gridlines at rows/cols 3 and 7.
  Each mini-block (r,c), r,c in {0,1,2}, sits at input rows/cols
  {0,1,2},{4,5,6},{8,9,10}.  Pink (6) pixels are scattered in the mini-blocks
  (each block holds exactly 1 or 2 pink pixels).  OUTPUT is a 3x3 grid where
  output[r][c] = blue (1) iff the mini-block (r,c) holds >= 2 pink pixels, else
  black (0).  Cells outside the 3x3 output are all zero (no channel set).

Encoding (sign-one-hot, near the irreducible floor):
  ONE strided no-pad Conv with NEGATIVE pads crops directly to [1,1,3,3]:
  kernel [1,10,3,3] selects channel 6 (pink) summed over each 3x3 window,
  stride 4 (= block period minisize+1), pads=[0,0,-19,-19] discards the trailing
  blocks so the output is exactly the 3x3 block-count grid.  bias = -1.5 folds the
  >=2 threshold in: conv = count - 1.5, which is >0 iff count >= 2.
  The harness scores (out > 0), so the 2-colour one-hot is built by SIGN with no
  Equal/Where/threshold consts: ch1 (blue) = conv (>0 when win), ch0 (black) =
  -conv (>0 when not win).  Cast the conv plane to fp16 first so Neg/Concat run on
  half-width planes; ONE Pad then adds the 8 trailing colour channels AND the
  30x30 spatial border (all zero) -> [1,10,30,30] output.  Off-3x3 cells stay
  all-zero across every channel, matching the sparse one-hot target.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

S = 30


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

    # ---- block pink-counts cropped to 3x3 by negative conv pads; bias = -1.5 ---
    W = np.zeros((1, 10, 3, 3), np.float32)
    W[0, 6, :, :] = 1.0
    init("cntW", W, np.float32)
    init("cntB", np.array([-1.5], np.float32), np.float32)
    n("Conv", ["input", "cntW", "cntB"], "cnt",
      kernel_shape=[3, 3], strides=[4, 4], pads=[0, 0, -19, -19])   # [1,1,3,3] fp32

    # half-width the two sign planes
    n("Cast", ["cnt"], "cntH", to=H)                # [1,1,3,3] fp16  (ch1: blue)
    n("Neg", ["cntH"], "negH")                      # [1,1,3,3] fp16  (ch0: black)
    n("Concat", ["negH", "cntH"], "oh", axis=1)     # [1,2,3,3] fp16  sign one-hot

    # one Pad: +8 trailing colour channels AND the 30x30 spatial border (all 0).
    # legacy (opset<=10) Pad takes pads/value as ATTRIBUTES (0 params) vs the
    # opset-11 pads-as-input form (which would cost an 8-elem initializer).
    n("Pad", ["oh"], "output", mode="constant", value=0.0,
      pads=[0, 0, 0, 0, 0, 8, S - 3, S - 3])         # [1,10,30,30] fp16

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", H, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task149", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 10)])
