"""Task 327 (d13f3404): propagate each coloured pixel down-right along its diagonal.

Rule (from ARC-GEN generator, size=3 fixed):
  Input is a 3x3 grid with up to 3 coloured pixels (colours 1..9), each on a
  DISTINCT diagonal `c-r` (generator skips repeated diagonals). The 6x6 output
  draws, from each coloured pixel at (r,c), a down-right 45 degree ray
  `output[r+idx][c+idx]=colour` for idx in range(2*size-max(r,c)) (i.e. until it
  leaves the 6x6 grid). Background stays 0; cells outside the 6x6 grid are empty.

Encoding (Tier S/A, diagonal cumulative-sum via one depthwise-free Conv):
  Because each diagonal carries at most ONE source pixel, the whole transform is
  a DIAGONAL PREFIX-SUM of the colour-index plane: out[R][C] = sum_{j>=0}
  Vin[R-j][C-j]. The sum has at most one nonzero term so it reproduces the source
  colour everywhere down-right of it, 0 elsewhere -- exact, no double-count.
  1. Slice input to the active [1,10,6,6] corner (rows/cols 0..5).
  2. Conv kw[1,10,1,1]=[0,1,..,9] -> colour-index plane Vin [1,1,6,6] (ch0 -> 0).
  3. Conv with a 6x6 diagonal-of-ones kernel, pad top/left=5 -> diagonal prefix
     sum Ldiag [1,1,6,6] = propagated colour index.
  4. Cast uint8, Pad to 30x30 with sentinel 10, final Equal(L, arange) -> the
     FREE bool output. Off-grid (L=10) matches no channel -> all-zero one-hot.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

WORK = 6  # active output side (2*size, size=3)


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
    # crop the 30x30 one-hot input to the 3x3 SOURCE corner (sources sit in
    # rows/cols 0..2). The colour-index Conv runs on this tiny [1,10,3,3] slice
    # (360B) instead of the full [1,10,6,6] (1440B); the diagonal Conv's top/left
    # pad of 5 then extends the ray across the full 6x6 output.
    SRC = 3
    init("crop_st", np.array([0, 0], np.int64), np.int64)
    init("crop_en", np.array([SRC, SRC], np.int64), np.int64)
    init("crop_ax", np.array([2, 3], np.int64), np.int64)

    # colour-index Conv weight: channel k -> weight k (ch0 -> 0 drops background)
    init("kw", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)

    # 6x6 diagonal-of-ones kernel (main diagonal). With pads=5 on top/left this
    # makes out[R][C] = sum_{j=0..5} Vin[R-j][C-j].
    dk = np.zeros((1, 1, WORK, WORK), dtype=np.float32)
    for j in range(WORK):
        dk[0, 0, WORK - 1 - j, WORK - 1 - j] = 1.0
    init("dk", dk, np.float32)

    # pad the 3x3 colour-index plane to the full 6x6 output canvas (bottom/right)
    init("vpads", np.array([0, 0, 0, 0, 0, 0, WORK - 3, WORK - 3], np.int64),
         np.int64)
    init("vzero", np.array(0.0, np.float32), np.float32)

    # final one-hot channel comparison + pad sentinel
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64),
         np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)

    # ---- graph ----
    # 1. 3x3 source corner of the one-hot input
    n("Slice", ["input", "crop_st", "crop_en", "crop_ax"], "Xs")  # [1,10,3,3]
    # 2. colour-index plane (ch0 weight 0 -> background stays 0)
    n("Conv", ["Xs", "kw"], "Vin3")                               # [1,1,3,3] f32
    # 2b. pad colour-index plane up to the 6x6 output canvas
    n("Pad", ["Vin3", "vpads", "vzero"], "Vin", mode="constant")  # [1,1,6,6] f32
    # 3. diagonal prefix sum (pad top/left by 5 so kernel reaches up-left)
    n("Conv", ["Vin", "dk"], "Ldiag", pads=[WORK - 1, WORK - 1, 0, 0])  # [1,1,6,6]
    # 4. label -> padded sentinel -> free bool output
    n("Cast", ["Ldiag"], "Lu", to=TensorProto.UINT8)             # [1,1,6,6] u8
    n("Pad", ["Lu", "padpads", "padval"], "L", mode="constant")  # [1,1,30,30] u8
    n("Equal", ["L", "chan"], "output")                          # free BOOL out

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
