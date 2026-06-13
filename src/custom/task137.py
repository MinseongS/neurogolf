"""Task 137 (5c2c9af4): concentric square rings from 3 diagonal pixels.

Rule (from ARC-GEN generator): input is a size x size grid (20..30) with 3
pixels of one color at (row-s, col+-s), (row, col), (row+s, col-+s). Output
draws concentric square rings around (row, col): cell (r, c) inside the grid
is `color` iff max(|r-row|, |c-col|) % s == 0, else black; outside the grid
the canvas is all-zero.

Graph: everything is computed on 1-D row/col vectors.
  - color one-hot v = Equal(per-channel count, 3)  -> palette col 0
  - palette data [1,10,3] = [v, black e0, zeros]
  - row/col pixel counts via two 300-param Convs (kernel 1x30 / 30x1 over
    channels 1..9); weighted sum/3 gives center, ReduceMax gives row+s -> s
  - size = Sqrt(total one-hot count); X(r) = |r-row| inside else 30 (Where),
    Y(c) likewise: the 30 sentinel dominates Max comparisons so grid bounds
    are folded into the 1-D vectors
  - zr = (X mod s == 0); per-row palette index pr = outside?2 : (zr?0:1),
    per-col pc likewise
  - chebyshev split: a = Less(X, Y) [30,30] bool; idx = Where(a, pc, pr)
    (max(X,Y) = Y when X<Y else X, so the ring test only needs the 1-D
    vector of the dominating axis)
  - output = Gather(data, idx, axis=2)  (free tensor)
Only two canvas-sized intermediates: a (900B bool) + idx (3600B int32).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ..builders import _model

I32 = onnx.TensorProto.INT32
F32 = onnx.TensorProto.FLOAT


def build(task):
    inits = []
    nodes = []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    # ---- constants ----
    wrp = np.zeros((1, 10, 1, 30), np.float32)
    wrp[0, 1:, 0, :] = 1.0
    init("Wrp", wrp, np.float32)                      # row pixel counts
    init("Wcp", wrp.reshape(1, 10, 30, 1), np.float32)  # col pixel counts
    init("iotaR", np.arange(30, dtype=np.float32).reshape(30, 1), np.float32)
    init("iotaC", np.arange(30, dtype=np.float32).reshape(1, 30), np.float32)
    extra = np.zeros((1, 10, 2), np.float32)
    extra[0, 0, 0] = 1.0                              # palette col 1: black
    init("EXTRA", extra, np.float32)                  # palette col 2: zeros
    init("c3i", np.array(3, np.int32), np.int32)
    init("c3f", np.array(3.0, np.float32), np.float32)
    init("c30f", np.array(30.0, np.float32), np.float32)
    init("c0i", np.array(0, np.int32), np.int32)
    init("c1i", np.array(1, np.int32), np.int32)
    init("c2i", np.array(2, np.int32), np.int32)

    # ---- palette: [color one-hot, black, zero] ----
    n("ReduceSum", ["input"], "chcnt", axes=[2, 3], keepdims=1)  # [1,10,1,1]
    n("Cast", ["chcnt"], "chcnt_i", to=I32)
    n("Equal", ["chcnt_i", "c3i"], "v_b")             # count==3 -> color
    n("Cast", ["v_b"], "v_f", to=F32)
    n("Squeeze", ["v_f"], "v_r", axes=[3])            # [1,10,1]
    n("Concat", ["v_r", "EXTRA"], "data", axis=2)     # [1,10,3]

    # ---- center (row,col) and spacing s from the 3 pixels ----
    n("Conv", ["input", "Wrp"], "rowpix")             # [1,1,30,1]
    n("Conv", ["input", "Wcp"], "colpix")             # [1,1,1,30]
    n("Mul", ["rowpix", "iotaR"], "rprod")            # [1,1,30,1]
    n("ReduceSum", ["rprod"], "srow", keepdims=0)     # 3*row
    n("ReduceMax", ["rprod"], "mrow", keepdims=0)     # row+s
    n("Div", ["srow", "c3f"], "rowc")
    n("Sub", ["mrow", "rowc"], "s_f")
    n("Mul", ["colpix", "iotaC"], "cprod")            # [1,1,1,30]
    n("ReduceSum", ["cprod"], "scol", keepdims=0)     # 3*col
    n("Div", ["scol", "c3f"], "colc")

    # ---- grid size and bounded chebyshev coordinates ----
    n("ReduceSum", ["input"], "ntot", keepdims=0)     # size^2
    n("Sqrt", ["ntot"], "size")
    n("Less", ["iotaR", "size"], "g_r")               # bool [30,1] in-grid row
    n("Less", ["iotaC", "size"], "g_c")               # bool [1,30]
    n("Sub", ["iotaR", "rowc"], "drs")
    n("Abs", ["drs"], "dr")                           # [30,1]
    n("Sub", ["iotaC", "colc"], "dcs")
    n("Abs", ["dcs"], "dc")                           # [1,30]
    n("Where", ["g_r", "dr", "c30f"], "X")            # |r-row| or 30 outside
    n("Where", ["g_c", "dc", "c30f"], "Y")

    # ---- ring test on each axis: dist mod s == 0 ----
    n("Cast", ["X"], "Xi", to=I32)
    n("Cast", ["Y"], "Yi", to=I32)
    n("Cast", ["s_f"], "si", to=I32)
    n("Mod", ["Xi", "si"], "mr")                      # [30,1] int32
    n("Mod", ["Yi", "si"], "mc")                      # [1,30]
    n("Equal", ["mr", "c0i"], "zr")
    n("Equal", ["mc", "c0i"], "zc")

    # ---- per-row / per-col palette indices ----
    n("Where", ["zr", "c0i", "c1i"], "prin")          # ring->0, bg->1
    n("Where", ["g_r", "prin", "c2i"], "pr")          # outside->2
    n("Where", ["zc", "c0i", "c1i"], "pcin")
    n("Where", ["g_c", "pcin", "c2i"], "pc")

    # ---- assemble: max(X,Y) picks the dominating axis ----
    n("Less", ["X", "Y"], "a")                        # bool [30,30]
    n("Where", ["a", "pc", "pr"], "idx")              # int32 [30,30]
    n("Gather", ["data", "idx"], "output", axis=2)    # [1,10,30,30] free

    return _model(nodes, inits)
