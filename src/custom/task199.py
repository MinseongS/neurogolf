"""Task 199 (ARC-GEN 834ec97d): move pixel down one row + draw a yellow parity comb.

Rule (ARC-GEN generator, verified fresh): input is a size x size grid (size in
3..15) holding a single non-yellow pixel of colour cc at (row, col); row in
[0,size-2], col in [0,size-1].  Output:
  - the pixel moves down by one row -> output[row+1][col] = cc;
  - yellow (channel 4) fills every cell (r,c) with r <= row AND c%2 == col%2,
    for r in 0..row (same column parity as col);
  - everything else inside the grid is background (0), off-grid all-zero.
The colour point sits at row+1, outside the yellow region, so they never
overlap.

Label-map + final Equal (small working canvas).  Build a SINGLE uint8 label map
L on a WORK x WORK canvas (WORK = 15 >= max size):
    L = (r==row+1 & c==col) ? cc
      : (r<=row & c%2==col%2) ? 4
      : (r<size & c<size) ? 0
      : 10 (off-grid sentinel, matches no channel)
then Pad to 30x30 (sentinel 10) and `output = Equal(L, arange[1,10,1,1])`
(free BOOL output, opset 11).

Scalars without any per-channel [1,10,30,*] marginal: ONE no-pad Conv with a
width-collapsing kernel W[2,10,30,1] (kh=30, kw=1) contracts the channel axis
(weight 0 on ch0 so background drops out) and the row axis at once:
    conv -> [1,2,1,30]
      ch0  weight (k>=1)        -> presence vector p[c] = 1 at the pixel col
      ch1  weight (k>=1)*r      -> value    vector v[c] = pixel row at pixel col
    col  = ReduceSum(p * colramp)      (single nonzero)
    row  = ReduceSum(v)                (single nonzero == pixel row)
    cc   = ReduceSum(counts * chanramp), counts = ReduceSum(input, axes=[2,3]).
size = sqrt(total one-hot count) (every in-grid cell sets exactly one channel,
so the count is size*size).  Dominant intermediate: padded L (uint8 900B); the
conv output is only [1,2,1,30] = 240B and replaces the two 1200B marginals.
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

    # ---- grid size = sqrt(total one-hot count) ----
    n("ReduceSum", ["input"], "tot", keepdims=1, axes=[1, 2, 3])      # scalar size^2
    n("Sqrt", ["tot"], "W")                                          # scalar size

    # ---- ONE conv: row index + presence at the pixel column ----
    # kernel [out=2, in=10, kh=30, kw=1]; collapses channel + row, leaves col.
    wconv = np.zeros((2, 10, 30, 1), np.float32)
    for k in range(1, 10):           # ch0 (background) weight stays 0
        wconv[0, k, :, 0] = 1.0       # presence
        wconv[1, k, :, 0] = np.arange(30)  # row index
    init("wconv", wconv, np.float32)
    n("Conv", ["input", "wconv"], "pv", kernel_shape=[30, 1],
      pads=[0, 0, 0, 0], strides=[1, 1])                             # [1,2,1,30]

    s0 = init("s0", np.array([0], np.int64), np.int64)
    e1 = init("e1", np.array([1], np.int64), np.int64)
    e2 = init("e2", np.array([2], np.int64), np.int64)
    ax1 = init("ax1", np.array([1], np.int64), np.int64)
    n("Slice", ["pv", s0, e1, ax1], "pres")                          # [1,1,1,30] presence
    n("Slice", ["pv", e1, e2, ax1], "vrow")                          # [1,1,1,30] row@col

    init("idxW30", np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30), np.float32)
    n("Mul", ["pres", "idxW30"], "cw")
    n("ReduceSum", ["cw"], "col", keepdims=1, axes=[1, 2, 3])        # scalar col
    n("ReduceSum", ["vrow"], "row", keepdims=1, axes=[1, 2, 3])      # scalar row
    n("Add", ["row", "one"], "rowp1")                                # row+1

    # ---- colour index cc (uint8 scalar) ----
    init("chvals", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    n("ReduceSum", ["input"], "cnt", axes=[2, 3], keepdims=1)        # [1,10,1,1]
    n("Mul", ["cnt", "chvals"], "ccparts")
    n("ReduceSum", ["ccparts"], "ccf", keepdims=1)                   # scalar cc float
    n("Cast", ["ccf"], "cc8", to=U8)                                 # uint8 scalar cc

    # ---- row-condition vectors on WORK canvas ----
    n("Less", ["idxH", "rowp1"], "vH_y")                             # r <= row
    n("Cast", ["rowp1"], "rowp1i", to=I32)
    n("Equal", ["idxHi", "rowp1i"], "vH_p")                          # r == row+1
    n("Less", ["idxH", "W"], "vH_g")                                 # r < size

    # ---- col-condition vectors on WORK canvas ----
    n("Less", ["idxW", "W"], "vW_g")                                 # c < size
    n("Cast", ["col"], "coli", to=I32)
    # col % 2 = col - 2*floor(col/2)
    n("Div", ["col", "two"], "colhf")
    n("Floor", ["colhf"], "colfl")
    n("Mul", ["colfl", "two"], "col2fl")
    n("Sub", ["col", "col2fl"], "colm2f")
    n("Cast", ["colm2f"], "colm2", to=I32)
    n("Equal", ["idxWmod2", "colm2"], "vW_y")                        # c%2 == col%2
    n("Equal", ["idxWi", "coli"], "vW_p")                            # c == col

    # ---- 2-D masks (broadcast And) ----
    n("And", ["vH_g", "vW_g"], "Gb")                                 # in-grid
    n("And", ["vH_y", "vW_y"], "Yb0")                                # yellow comb (ungated)
    n("And", ["Yb0", "Gb"], "Yb")                                    # gate to in-grid
    n("And", ["vH_p", "vW_p"], "Pb")                                 # moved pixel

    # ---- label map L (priority: pixel > yellow > in-grid > outside) ----
    init("v0", np.array(0, np.uint8), np.uint8)
    init("v4", np.array(4, np.uint8), np.uint8)
    init("vout", np.array(10, np.uint8), np.uint8)
    n("Where", ["Gb", "v0", "vout"], "L1")                           # 0 in-grid else 10
    n("Where", ["Yb", "v4", "L1"], "L2")                             # yellow = 4
    n("Where", ["Pb", "cc8", "L2"], "L12")                           # pixel = cc

    # ---- pad to 30x30 with sentinel 10, final Equal ----
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    n("Pad", ["L12", "padpads", "padval"], "L", mode="constant")     # [1,1,30,30] uint8
    n("Equal", ["L", "chan"], "output")                              # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
