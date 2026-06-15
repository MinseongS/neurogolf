"""Task 199 (834ec97d): move pixel down one row + draw a yellow parity stripe.

Rule (ARC-GEN generator): input is a size x size grid (size in 3..15) holding a
single non-yellow pixel of color cc at (row, col).  Output:
  - the pixel moves down by one row -> (row+1, col), same color cc;
  - yellow (channel 4) fills every cell (r, c) with r <= row and
    c % 2 == col % 2 and c < size;
  - everything else inside the grid is background (channel 0).

Memory floor-break (label map + final Equal, small working canvas).
Build a SINGLE uint8 label map L on a WORK x WORK canvas (WORK = 15 >= max size)
giving the output colour index per cell:
    L = (r==row+1 & c==col) ? cc
      : (r<=row & c%2==col%2 & c<W) ? 4
      : (r<W & c<W) ? 0
      : 10 (outside sentinel, never matches channels 0..9)
Then Pad L to 30x30 with sentinel 10 and emit `output = Equal(L, arange[1,10,1,1])`
(free BOOL output, opset 11).  Every per-cell plane is WORK x WORK uint8 (225B);
the row/col scalars come from 1-D marginals, never a canvas-sized float stack.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

WORK = 15  # working canvas side (>= max grid size 15)
F = TensorProto.FLOAT
I32 = TensorProto.INT32
U8 = TensorProto.UINT8


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
    init("idxH", np.arange(WORK, dtype=np.float32).reshape(1, 1, WORK, 1), np.float32)
    init("idxW", np.arange(WORK, dtype=np.float32).reshape(1, 1, 1, WORK), np.float32)
    init("idxHi", np.arange(WORK, dtype=np.int32).reshape(1, 1, WORK, 1), np.int32)
    init("idxWi", np.arange(WORK, dtype=np.int32).reshape(1, 1, 1, WORK), np.int32)
    init("idxWmod2", (np.arange(WORK) % 2).astype(np.int32).reshape(1, 1, 1, WORK), np.int32)
    init("one", np.array(1.0, np.float32), np.float32)
    init("two", np.array(2.0, np.float32), np.float32)
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)

    # ---- grid width W = sqrt(total one-hot count) ----
    n("ReduceSum", ["input"], "tot", keepdims=1, axes=[1, 2, 3])
    n("Sqrt", ["tot"], "W")                                         # scalar grid width

    # ---- row/col occupancy of the single (input) pixel ----
    # ch0 = background; non-background marginal isolates the pixel row/col.
    s0 = init("s0", np.array([0], np.int64), np.int64)
    e1 = init("e1", np.array([1], np.int64), np.int64)
    ax1 = init("ax1", np.array([1], np.int64), np.int64)
    n("ReduceSum", ["input"], "rc", keepdims=1, axes=[3])          # [1,10,30,1]
    n("ReduceSum", ["rc"], "rc_all", keepdims=1, axes=[1])         # [1,1,30,1]
    n("Slice", ["rc", s0, e1, ax1], "rc0")                         # ch0 row counts
    n("Sub", ["rc_all", "rc0"], "rowocc")                          # 1 @ pixel row
    n("ReduceSum", ["input"], "cc_", keepdims=1, axes=[2])         # [1,10,1,30]
    n("ReduceSum", ["cc_"], "cc_all", keepdims=1, axes=[1])        # [1,1,1,30]
    n("Slice", ["cc_", s0, e1, ax1], "cc0")
    n("Sub", ["cc_all", "cc0"], "colocc")                          # 1 @ pixel col

    # ---- row / col scalars (float, integer-valued) ----
    init("idxH30", np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1), np.float32)
    init("idxW30", np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30), np.float32)
    n("Mul", ["idxH30", "rowocc"], "rw")
    n("ReduceSum", ["rw"], "row", keepdims=1, axes=[1, 2, 3])      # scalar row
    n("Mul", ["idxW30", "colocc"], "cw")
    n("ReduceSum", ["cw"], "col", keepdims=1, axes=[1, 2, 3])      # scalar col
    n("Add", ["row", "one"], "rowp1")                              # row+1

    # ---- colour index cc (uint8 scalar) ----
    init("chvals", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    n("ReduceMax", ["input"], "pres", axes=[2, 3], keepdims=1)     # [1,10,1,1]
    n("Mul", ["pres", "chvals"], "ccparts")
    n("ReduceSum", ["ccparts"], "ccf", keepdims=1)                 # scalar cc float
    n("Cast", ["ccf"], "cc8", to=U8)                               # uint8 scalar cc

    # ---- row-condition vectors on WORK canvas ----
    n("Less", ["idxH", "rowp1"], "vH_y")                           # r <= row
    n("Cast", ["rowp1"], "rowp1i", to=I32)
    n("Equal", ["idxHi", "rowp1i"], "vH_p")                        # r == row+1
    n("Less", ["idxH", "W"], "vH_g")                               # r < W

    # ---- col-condition vectors on WORK canvas ----
    n("Less", ["idxW", "W"], "vW_g")                               # c < W
    n("Cast", ["col"], "coli", to=I32)
    # col % 2 = col - 2*floor(col/2)
    n("Div", ["col", "two"], "colhf")
    n("Floor", ["colhf"], "colfl")
    n("Mul", ["colfl", "two"], "col2fl")
    n("Sub", ["col", "col2fl"], "colm2f")
    n("Cast", ["colm2f"], "colm2", to=I32)
    n("Equal", ["idxWmod2", "colm2"], "par")                       # c%2 == col%2
    n("And", ["par", "vW_g"], "vW_y")                              # parity & c<W
    n("Equal", ["idxWi", "coli"], "vW_p")                          # c == col

    # ---- 2-D masks (broadcast And) ----
    n("And", ["vH_g", "vW_g"], "Gb")                               # in-grid
    n("And", ["vH_y", "vW_y"], "Yb")                               # yellow stripe
    n("And", ["vH_p", "vW_p"], "Pb")                               # moved pixel

    # ---- label map L (priority: pixel > yellow > in-grid > outside) ----
    init("v0", np.array(0, np.uint8), np.uint8)
    init("v4", np.array(4, np.uint8), np.uint8)
    init("vout", np.array(10, np.uint8), np.uint8)
    n("Where", ["Gb", "v0", "vout"], "L1")                         # 0 in-grid else 10
    n("Where", ["Yb", "v4", "L1"], "L2")                           # yellow = 4
    n("Where", ["Pb", "cc8", "L2"], "L12")                         # pixel = cc

    # ---- pad to 30x30 with sentinel 10, final Equal ----
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    n("Pad", ["L12", "padpads", "padval"], "L", mode="constant")   # [1,1,30,30] uint8
    n("Equal", ["L", "chan"], "output")                            # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
