"""Task 199 (834ec97d): move pixel down one row + draw a yellow parity stripe.

Rule (ARC-GEN generator): input has a single non-yellow pixel of color cc at
(row, col) on a size x size grid.  Output:
  - the pixel moves down by one row -> (row+1, col), same color cc;
  - yellow (channel 4) fills every cell (r, c) with r <= row and
    c % 2 == col % 2 (and c < size);
  - everything else inside the grid is background (channel 0).

Every relevant region is a separable outer product of a row-vector (varies
along H) and a col-vector (varies along W):
  Gmask = (r < W)        x (c < W)              -> inside the grid
  Y     = (r <= row)     x (c%2==col%2 & c<W)   -> yellow stripe
  Pmask = (r == row+1)   x (c == col)           -> moved pixel
Stack the three row-vectors into R [1,3,30,1] and the three col-vectors into
C [1,3,1,30]; one broadcast Mul gives feat = R*C = [1,3,30,30] = [Gmask,Y,Pmask]
in a single tensor (the only canvas-sized intermediate).

A 1x1 Conv with a runtime weight [10,3,1,1] routes feat to the 10 output
channels and writes `output` directly (free):
  out0   = Gmask - Y - Pmask    (background minus the two specials)
  out4   = Y                    (yellow)
  out_cc = Pmask                (pixel color; cc = input pixel's channel, != 4)

W = sqrt(sum(input)); row/col come from channel-1..9 marginals of the input
(computed without any canvas-sized slice).  Memory is dominated by the single
[1,3,30,30] feat (10800); everything else is tiny vectors.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ..builders import _model

FLT = onnx.TensorProto.FLOAT
I32 = onnx.TensorProto.INT32
I64 = onnx.TensorProto.INT64


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **kw):
        nodes.append(helper.make_node(op, ins, [out], **kw))
        return out

    # ---- constants ----
    init("idxH", np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1), np.float32)
    init("idxW", np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30), np.float32)
    init("idxHi", np.arange(30, dtype=np.int32).reshape(1, 1, 30, 1), np.int32)
    init("idxWi", np.arange(30, dtype=np.int32).reshape(1, 1, 1, 30), np.int32)
    init("idxWmod2", (np.arange(30) % 2).astype(np.int32).reshape(1, 1, 1, 30), np.int32)
    init("one", np.array(1.0, np.float32), np.float32)
    init("two", np.array(2.0, np.float32), np.float32)

    # ---- grid width W = sqrt(total one-hot count) ----
    n("ReduceSum", ["input"], "tot", keepdims=1, axes=[1, 2, 3])    # [1,1,1,1]
    n("Sqrt", ["tot"], "W")                                         # grid width

    # ---- row/col occupancy (1 at pixel row/col) ----
    # rc[ch,r] = cells of colour ch in row r; rc_all = W per row<W,
    # rc_ch0 = background per row (= W except W-1 at pixel row); diff = pixel.
    init("s0", np.array([0], np.int64), np.int64)
    init("e1", np.array([1], np.int64), np.int64)
    init("ax1", np.array([1], np.int64), np.int64)
    n("ReduceSum", ["input"], "rc", keepdims=1, axes=[3])           # [1,10,30,1]
    n("ReduceSum", ["rc"], "rc_all", keepdims=1, axes=[1])          # [1,1,30,1]
    n("Slice", ["rc", "s0", "e1", "ax1"], "rc0")                    # [1,1,30,1] ch0
    n("Sub", ["rc_all", "rc0"], "rowocc")                           # [1,1,30,1] 1@row
    n("ReduceSum", ["input"], "cc_", keepdims=1, axes=[2])          # [1,10,1,30]
    n("ReduceSum", ["cc_"], "cc_all", keepdims=1, axes=[1])         # [1,1,1,30]
    n("Slice", ["cc_", "s0", "e1", "ax1"], "cc0")                   # [1,1,1,30] ch0
    n("Sub", ["cc_all", "cc0"], "colocc")                           # [1,1,1,30] 1@col
    # per-channel colour histogram (channel cc == 1) for the routing weight
    n("ReduceSum", ["rc"], "cnt", keepdims=1, axes=[2])             # [1,10,1,1]

    # ---- row / col scalars ----
    n("Mul", ["idxH", "rowocc"], "rw")
    n("ReduceSum", ["rw"], "row", keepdims=1, axes=[1, 2, 3])       # [1,1,1,1]
    n("Mul", ["idxW", "colocc"], "cw")
    n("ReduceSum", ["cw"], "col", keepdims=1, axes=[1, 2, 3])       # [1,1,1,1]
    n("Add", ["row", "one"], "rowp1")                               # row+1

    # ---- row-vectors (bool), concat then a single cast to float ----
    n("Less", ["idxH", "W"], "vH_g_b")                              # r < W
    n("Less", ["idxH", "rowp1"], "vH_y_b")                          # r <= row
    n("Cast", ["rowp1"], "rowp1i", to=I32)
    n("Equal", ["idxHi", "rowp1i"], "vH_p_b")                       # r == row+1
    n("Concat", ["vH_g_b", "vH_y_b", "vH_p_b"], "Rb", axis=1)      # [1,3,30,1] bool
    n("Cast", ["Rb"], "R", to=FLT)

    # ---- col-vectors (bool), concat then a single cast to float ----
    n("Less", ["idxW", "W"], "vW_g_b")                              # c < W
    # col % 2
    n("Cast", ["col"], "coli", to=I32)
    n("Div", ["col", "two"], "colhf")
    n("Floor", ["colhf"], "colfl")
    n("Mul", ["colfl", "two"], "col2fl")
    n("Sub", ["col", "col2fl"], "colm2f")
    n("Cast", ["colm2f"], "colm2", to=I32)
    n("Equal", ["idxWmod2", "colm2"], "par_b")                      # c%2 == col%2
    n("And", ["par_b", "vW_g_b"], "vW_y_b")                         # parity & c<W
    n("Equal", ["idxWi", "coli"], "vW_p_b")                         # c == col
    n("Concat", ["vW_g_b", "vW_y_b", "vW_p_b"], "Cb", axis=1)     # [1,3,1,30] bool
    n("Cast", ["Cb"], "C", to=FLT)

    # ---- feat = R * C = [Gmask, Y, Pmask] ----
    n("Mul", ["R", "C"], "feat")                                    # [1,3,30,30]

    # ---- runtime conv weight [10,3,1,1] ----
    col0 = np.zeros((10, 1, 1, 1), np.float32); col0[0] = 1.0         # +Gmask -> out0
    col1 = np.zeros((10, 1, 1, 1), np.float32); col1[0] = -1.0; col1[4] = 1.0
    init("wcol0", col0, np.float32)
    init("wcol1", col1, np.float32)
    init("maskno0", np.array([0.] + [1.] * 9, np.float32).reshape(1, 10, 1, 1), np.float32)
    init("negE0", np.array([-1.] + [0.] * 9, np.float32).reshape(10, 1, 1, 1), np.float32)
    init("shp10", np.array([10, 1, 1, 1], np.int64), np.int64)
    n("Mul", ["cnt", "maskno0"], "csel")                            # onehot(cc) [1,10,1,1]
    n("Reshape", ["csel", "shp10"], "csel2")                        # [10,1,1,1]
    n("Add", ["csel2", "negE0"], "wcol2")                           # +Pmask@cc, -Pmask@0
    n("Concat", ["wcol0", "wcol1", "wcol2"], "W3", axis=1)         # [10,3,1,1]

    # ---- final conv -> output (free) ----
    n("Conv", ["feat", "W3"], "output", kernel_shape=[1, 1])
    return _model(nodes, inits)
