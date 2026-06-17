"""task045 (ARC-AGI 22eb0ac0) — "draw the row line where both endpoints match".

Rule (from the generator, size=10 grid):
  For each idx, row r = 2*idx+1 (rows 1,3,5,7,9) carries a LEFT endpoint colour at
  column 0 and a RIGHT endpoint colour at column size-1 (=9).  The input shows just
  those two endpoint cells.  OUTPUT = INPUT, plus: for every row whose LEFT colour
  EQUALS its RIGHT colour, the ENTIRE row is filled with that colour (interior cols
  1..8 get the endpoint colour).  Rows whose endpoints differ are left unchanged
  (only the two endpoint cells).

  Per channel k / row r / col c, the output one-hot is:
    output[k,r,c] = input[k,r,c]                                  (preserve input)
                    OR (c is interior AND row r qualifies AND k is the row colour)
  where "row r qualifies" = input[k,r,0]==1 AND input[k,r,9]==1 for some k, and the
  row colour one-hot is exactly  both[k,r] = input[k,r,0] * input[k,r,9].

Encoding (route the 10-ch expansion into the FREE output via Where):
  leftcol  = input[:,:,:,0:1]     [1,10,30,1]  (left endpoint one-hot per row)
  rightcol = input[:,:,:,9:10]    [1,10,30,1]  (right endpoint one-hot per row)
  both     = leftcol * rightcol   [1,10,30,1]  (row-colour one-hot; 0 if endpoints differ)
  qualify  = ReduceMax(both,[1])  [1,1,30,1]   (>0 iff row qualifies)  -> bool
  interior = const [1,1,1,30]     (1 on cols 1..8, the 10x10 interior) -> bool
  fillcond = And(qualify, interior) [1,1,30,30] bool                   (interior cell of a
                                                                        qualifying row)
  output   = Where(fillcond, both, input)   FREE [1,10,30,30]
    - fillcond true  -> both[1,10,30,1] broadcasts -> the row's colour one-hot
    - elsewhere      -> input (preserves endpoints, in-grid bg, all-zero off-grid)

  Dominant intermediate: fillcond [1,1,30,30] bool = 900B; the three [1,10,30,1] slices
  are 1200B each.  No [1,10,30,30] plane is ever materialised (it IS the free output).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- left / right endpoint one-hots per row -----------------------------
    init("l_s", np.array([0], np.int64), np.int64)
    init("l_e", np.array([1], np.int64), np.int64)
    init("l_ax", np.array([3], np.int64), np.int64)
    n("Slice", ["input", "l_s", "l_e", "l_ax"], "leftcol")    # [1,10,30,1]

    init("r_s", np.array([9], np.int64), np.int64)
    init("r_e", np.array([10], np.int64), np.int64)
    n("Slice", ["input", "r_s", "r_e", "l_ax"], "rightcol")   # [1,10,30,1]

    # both[k,r] = 1 iff channel k is present at BOTH ends of row r (row colour one-hot)
    n("Mul", ["leftcol", "rightcol"], "both")                 # [1,10,30,1] {0,1}

    # qualify[r] = >0 iff the row's endpoints match  (colours are 1..9, never bg)
    n("ReduceMax", ["both"], "qual_f", axes=[1], keepdims=1)  # [1,1,30,1] {0,1}
    init("ZERO", np.array(0.0, np.float32), np.float32)
    n("Greater", ["qual_f", "ZERO"], "qualify")               # [1,1,30,1] bool

    # interior mask: cols 1..8 of the 10x10 grid
    interior = np.zeros((1, 1, 1, 30), dtype=bool)
    interior[0, 0, 0, 1:9] = True
    init("interior", interior, np.bool_)

    # fillcond[r,c] = qualifying row AND interior column
    n("And", ["qualify", "interior"], "fillcond")             # [1,1,30,30] bool

    # output = Where(fillcond, row-colour one-hot, input)  -> FREE [1,10,30,30]
    n("Where", ["fillcond", "both", "input"], "output")

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task045", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
