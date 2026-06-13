"""Task 109 (ARC 47c1f68c): 4-fold mirror of a top-left sprite, in linecolor.

Input is a (2s+1)x(2s+1) grid: a linecolor cross at row=s, col=s, and a
`color` sprite in the top-left quadrant (rows,cols < s). Output is the 2s x 2s
grid where the sprite mask is mirrored into all four quadrants and rendered in
linecolor (cross removed; background elsewhere inside the grid).

Method (all size-derived in-graph, exact for s in {3,4,5,6}):
  occupancy O = 1 - ch0.  n = max column-occupancy (the vertical cross col is
  the unique fully-occupied column).  t = n-2 = 2s-1, m = n-1 = 2s.
  sprite mask M = O restricted to top-left sxs (2*row < t & 2*col < t).
  symmetric fold matrix E[R,K] = I[R,K] + [R+K == t]; out_mask = E @ M @ E
  (E is symmetric, so E^T = E). out_mask is the 4-fold mirror.
  linecolor one-hot L: column-sum of input over the vertical cross column.
  Output channels: ch0 = gridmask*(1-out_mask) (m x m), channel L = out_mask.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ..builders import _model


def build(task):
    inits = []
    nodes = []

    def init(name, arr, dtype=np.float32):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    one = init("one", np.array([1.0], np.float32))
    two = init("two", np.array([2.0], np.float32))
    half = init("half", np.array([0.5], np.float32))

    # --- occupancy O = sum of color channels 1..9 (0 outside grid AND on bg) ---
    init("ax1", np.array([1], np.int64), np.int64)
    init("c0s", np.array([0], np.int64), np.int64)
    init("c0e", np.array([1], np.int64), np.int64)
    n("Slice", ["input", "c0s", "c0e", "ax1"], "ch0")          # [1,1,30,30]
    n("ReduceSum", ["input"], "ingrid", axes=[1], keepdims=1)  # [1,1,30,30]
    n("Sub", ["ingrid", "ch0"], "O")                           # color occupancy

    # --- n = max column occupancy (vertical cross column is uniquely full) ---
    n("ReduceSum", ["O"], "colsum", axes=[2], keepdims=1)       # [1,1,1,30]
    n("ReduceMax", ["colsum"], "nval", axes=[3], keepdims=1)    # [1,1,1,1] = n
    n("Sub", ["nval", "two"], "t")                              # t = n-2 = 2s-1
    n("Sub", ["nval", "one"], "m")                              # m = n-1 = 2s

    # --- vertical cross column mask (colsum == n) ---
    n("Cast", ["colsum"], "colsumI", to=onnx.TensorProto.INT32)
    n("Cast", ["nval"], "nvalI", to=onnx.TensorProto.INT32)
    n("Equal", ["colsumI", "nvalI"], "vmaskB")                  # [1,1,1,30] bool
    n("Cast", ["vmaskB"], "vmask", to=onnx.TensorProto.FLOAT)

    # --- linecolor one-hot L = (sum over cross column > 0) ---
    n("ReduceSum", ["input"], "Xc", axes=[2], keepdims=1)       # [1,10,1,30]
    n("Mul", ["Xc", "vmask"], "Xcv")                            # [1,10,1,30]
    n("ReduceSum", ["Xcv"], "Lcnt", axes=[3], keepdims=1)       # [1,10,1,1]
    n("Cast", ["Lcnt"], "LcntI", to=onnx.TensorProto.INT32)
    init("zeroI", np.array([0], np.int32), np.int32)
    n("Greater", ["LcntI", "zeroI"], "LB")                      # [1,10,1,1] bool

    # --- sprite mask M = O restricted to top-left sxs (2*row<t & 2*col<t) ---
    twoR = (2 * np.arange(30)).reshape(1, 1, 30, 1).astype(np.float32)
    twoC = (2 * np.arange(30)).reshape(1, 1, 1, 30).astype(np.float32)
    init("twoR", twoR)
    init("twoC", twoC)
    n("Less", ["twoR", "t"], "rmB")                             # [1,1,30,1] bool
    n("Less", ["twoC", "t"], "cmB")                             # [1,1,1,30] bool
    n("Cast", ["rmB"], "rm", to=onnx.TensorProto.FLOAT)
    n("Cast", ["cmB"], "cm", to=onnx.TensorProto.FLOAT)
    n("Mul", ["O", "rm"], "Or")
    n("Mul", ["Or", "cm"], "M")                                 # [1,1,30,30]

    # --- fold matrix E[R,K] = I[R,K] + [R+K == t] (symmetric) ---
    R = np.arange(30).reshape(30, 1)
    K = np.arange(30).reshape(1, 30)
    init("ImatB", (R == K), np.bool_)                           # [30,30] bool
    init("RpKI", (R + K).astype(np.int32), np.int32)            # [30,30] int
    n("Cast", ["t"], "tIs", to=onnx.TensorProto.INT32)          # [1,1,1,1]
    n("Reshape",
      ["tIs", init("scl", np.array([1], np.int64), np.int64)], "tscalar")  # [1]
    n("Equal", ["RpKI", "tscalar"], "mirrB")                    # [30,30] bool
    n("Or", ["ImatB", "mirrB"], "EB")                           # [30,30] bool
    n("Cast", ["EB"], "E", to=onnx.TensorProto.FLOAT)           # [30,30]

    # --- out_mask = E @ M @ E ---
    n("MatMul", ["E", "M"], "EM")                               # [1,1,30,30]
    n("MatMul", ["EM", "E"], "outm")                            # [1,1,30,30]

    n("Greater", ["outm", "half"], "outmB")                     # [1,1,30,30] bool

    # --- gridmask (row<m & col<m) as bool ---
    Ridx = np.arange(30).reshape(1, 1, 30, 1).astype(np.float32)
    Cidx = np.arange(30).reshape(1, 1, 1, 30).astype(np.float32)
    init("Ridx", Ridx)
    init("Cidx", Cidx)
    n("Less", ["Ridx", "m"], "grB")                             # [1,1,30,1] bool
    n("Less", ["Cidx", "m"], "gcB")                             # [1,1,1,30] bool
    n("And", ["grB", "gcB"], "gridB")                           # [1,1,30,30] bool
    n("Not", ["outmB"], "notout")
    n("And", ["gridB", "notout"], "ch0B")                       # [1,1,30,30] bool

    # --- assemble output (all bool) then single Cast into `output` ---
    # channel 0 = gridB & ~outm; channels 1..9 = outm & L[1:10]
    init("Ls1", np.array([1], np.int64), np.int64)
    init("Ls10", np.array([10], np.int64), np.int64)
    n("Slice", ["LB", "Ls1", "Ls10", "ax1"], "LsliceB")         # [1,9,1,1] bool
    n("And", ["outmB", "LsliceB"], "partLB")                    # [1,9,30,30] bool
    n("Concat", ["ch0B", "partLB"], "outB", axis=1)             # [1,10,30,30] bool
    n("Cast", ["outB"], "output", to=onnx.TensorProto.FLOAT)

    return _model(nodes, inits)
