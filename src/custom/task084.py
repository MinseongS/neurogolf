"""Task 084 (3bd67248): left column color s x s grid -> add red anti-diagonal
and yellow bottom row.

Rule (from ARC-GEN generator): square grid of size s (3..21) with the left
column painted some color c.  Output keeps the left column, paints cells
(s-1-j, j) red (2) for j = 1..s-1 (the anti-diagonal i+j = s-1, excluding
column 0) and cells (s-1, j) yellow (4) for j = 1..s-1 (bottom row excluding
column 0).  The new cells never overlap column 0, so even when c is 2 or 4
the one-hot planes stay exact.

Graph: s = ReduceSum(input) (one 1 per colored cell).  All geometry is done
with broadcast comparisons on tiny index vectors:
  eq    = (i+1 == s)              row s-1 indicator          [1,1,30,1]
  lb    = (i+1 <  s)              rows 0..s-2 indicator      [1,1,30,1]
  colb  = (colidx < s)            cols 1..s-1 (col0 -> 999)  [1,1,1,30]
  yb    = eq AND colb             yellow cells               [1,1,30,30]
  rb    = (i+1 == s - colidx)     red cells (i+j == s-1)     [1,1,30,30]
  cond  = yb OR rb
Per-row replacement colors X[1,10,30,1]: rows < s-1 -> e2 (red), row s-1 ->
e4 (yellow); built by Concat of zero blocks with Cast(lb)/Cast(eq).
Final op: output = Where(cond, X, input)  (output tensor is free).
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

    # row index + 1, shape [1,1,30,1]
    init("rp1", (np.arange(30) + 1).reshape(1, 1, 30, 1), np.int32)
    # column index, col 0 poisoned to 999 (must never match), [1,1,1,30]
    cj = np.arange(30)
    cj[0] = 999
    init("cj", cj.reshape(1, 1, 1, 30), np.int32)
    # zero channel blocks for X concat
    init("z2", np.zeros((1, 2, 30, 1)), np.float32)
    init("z1", np.zeros((1, 1, 30, 1)), np.float32)
    init("z5", np.zeros((1, 5, 30, 1)), np.float32)

    # every in-grid cell is one-hot (color 0 included) -> total = s*s
    n("ReduceSum", ["input"], "ss_f", keepdims=0)       # scalar f32 = s^2
    n("Sqrt", ["ss_f"], "s_f")                          # exact for s<=21
    n("Cast", ["s_f"], "s_i", to=I32)                   # scalar i32

    n("Equal", ["rp1", "s_i"], "eq")                    # i == s-1   [1,1,30,1]
    n("Less", ["rp1", "s_i"], "lb")                     # i <  s-1   [1,1,30,1]
    n("Less", ["cj", "s_i"], "colb")                    # 1<=j<=s-1  [1,1,1,30]
    n("And", ["eq", "colb"], "yb")                      # yellow     [1,1,30,30]
    n("Sub", ["s_i", "cj"], "smj")                      # s-j        [1,1,1,30]
    n("Equal", ["rp1", "smj"], "rb")                    # i+j==s-1   [1,1,30,30]
    n("Or", ["yb", "rb"], "cond")                       # [1,1,30,30]

    n("Cast", ["eq"], "eqf", to=F32)                    # [1,1,30,1]
    n("Cast", ["lb"], "lbf", to=F32)                    # [1,1,30,1]
    # X[c,i]: c==2 for rows<s-1 (red), c==4 for row s-1 (yellow)
    n("Concat", ["z2", "lbf", "z1", "eqf", "z5"], "X", axis=1)  # [1,10,30,1]

    n("Where", ["cond", "X", "input"], "output")
    return _model(nodes, inits)
