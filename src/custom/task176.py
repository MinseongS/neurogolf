"""task176 (ARC-AGI 7447852a) — add a FIXED periodic yellow pattern to a red zigzag.

Rule (from the generator, height is ALWAYS 3):
  The input is a red (2) zigzag: red sits at (r,c) where r = c%4 folded to {0,1,2}
  (the triangle wave 0,1,2,1,0,1,2,1,...).  A `mode` counter (period 6) increments
  every time the zigzag touches row 0 or row 2 (every 2 columns).  Yellow (4) is
  filled BELOW the red when mode in {0,5} and ABOVE the red when mode in {2,3}.
  Both r and mode are functions of the column index c ALONE (independent of the
  grid width), so the entire yellow overlay is a FIXED pattern, periodic with
  period 12 in c, occupying only the top 3 rows.  The output is the input with
  this yellow pattern added, restricted to the in-grid columns (the grid spans
  columns 0..width-1; every in-grid column contains at least one red pixel).

Encoding (Tier-S/A, route the 10-ch expansion into the FREE Where output):
  yellow positions = FIXED const plane Y[1,1,30,30] (uint8 {0,1}, top-3-rows only)
                     AND colmask[1,1,1,30] (column has any input pixel).
  output = Where(cond, yellow_onehot[1,10,1,1], input).  Red is preserved because
  yellow never coincides with red.  The dominant intermediate is the [1,1,30,30]
  bool cond plane (~900B); everything else is tiny ([1,1,1,30] vectors).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I64 = TensorProto.INT64


def _yellow_pattern():
    """Fixed [3,30] yellow (color-4) overlay as a function of column index."""
    height = 3
    yel = np.zeros((3, 30), dtype=np.uint8)

    def modeof(c):
        mode = -1
        for cc in range(c + 1):
            r = cc % (2 * height - 2)
            r = r if r < height else 2 * height - r - 2
            mode = mode if r not in [0, height - 1] else (mode + 1) % 6
        return mode

    for c in range(30):
        r = c % (2 * height - 2)
        r = r if r < height else 2 * height - r - 2
        m = modeof(c)
        if m in [0, 5]:
            for i in range(r + 1, height):
                yel[i, c] = 1
        if m in [2, 3]:
            for i in range(0, r):
                yel[i, c] = 1
    return yel


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- FIXED yellow pattern plane Y[1,1,30,30] (uint8 -> bool) -------------
    Y = np.zeros((1, 1, 30, 30), dtype=np.uint8)
    Y[0, 0, :3, :] = _yellow_pattern()
    init("Y_u8", Y, np.uint8)
    n("Cast", ["Y_u8"], "Y_bool", to=BOOL)             # [1,1,30,30] bool

    # ---- colmask[1,1,1,30] : column has any input pixel ---------------------
    n("ReduceMax", ["input"], "col_any", axes=[1, 2], keepdims=1)  # [1,1,1,30] f32
    init("ZEROF", np.array(0.0, np.float32), np.float32)
    n("Greater", ["col_any", "ZEROF"], "colmask")      # [1,1,1,30] bool

    # ---- cond = Y AND colmask  (broadcast) ----------------------------------
    n("And", ["Y_bool", "colmask"], "cond")            # [1,1,30,30] bool

    # ---- yellow one-hot (color 4) as a [1,10,1,1] constant ------------------
    oh = np.zeros((1, 10, 1, 1), dtype=np.float32)
    oh[0, 4, 0, 0] = 1.0
    init("yellow_oh", oh, np.float32)

    # ---- output = Where(cond, yellow_onehot, input) : FREE [1,10,30,30] ------
    n("Where", ["cond", "yellow_oh", "input"], "output")

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task176", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
