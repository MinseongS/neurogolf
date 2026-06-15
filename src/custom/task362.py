"""Task 362 (ARC e48d4e1a): shift the cross axes by `offset`.

Rule (from ARC-GEN generator, size=10 always):
  Input: a full horizontal line at row `row` and a full vertical line at col
  `col`, both in `color`; plus `offset` gray (5) cells in the last column,
  rows 0..offset-1.  row,col in 1..8; offset in 1..3 (clamped so that
  row+offset <= 9 and col-offset >= 0).
  Output: the horizontal line moved DOWN by offset (now at row `row+offset`),
  the vertical line moved LEFT by offset (now at col `col-offset`); no gray.

Closed-form / label-map floor break:
  - color   = sum over channels k in {1,2,3,4,6,7,8,9} of presence[k]*k.
  - offset  = pixel count of channel 5 (gray).
  - row     = the row whose colored-cell count is 10 (the full horizontal
              line); every other in-grid row has count 1 (the vertical line).
  - col     = the col whose colored-cell count is 10.
  - R_out = row + offset ; C_out = col - offset (both always in 0..9).
  On a 10x10 working canvas the two output lines each span the whole grid, so
  linemask = (r == R_out) OR (c == C_out).  Build a uint8 label L10 = color on
  the lines else 0, Pad to 30x30 with sentinel 10 (off-grid), and the final op
  Equal(L, arange[0..9]) writes straight into the free BOOL output.

Memory: dominant intermediate is the fp32 channel-collapse Conv plane
[1,1,30,30] = 3600 B (needed to read per-row/col colored counts) plus the
900 B padded label.  All other tensors are <=120 B.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

WORK = 10  # grid side (size is always 10 in this generator)

# colorset = channels that can be the moving colour (exclude bg=0, gray=5)
_COLORSET = [1, 2, 3, 4, 6, 7, 8, 9]


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
    # 1x1 Conv weight that collapses the 10 channels to the colored-cell
    # indicator (1 for colorset channels, 0 for bg & gray).
    w_col = np.zeros((1, 10, 1, 1), np.float32)
    for k in _COLORSET:
        w_col[0, k, 0, 0] = 1.0
    init("Wcol", w_col, np.float32)

    # colour-value weights: presence[k] * k summed over colorset gives colour.
    w_val = np.zeros((1, 10, 1, 1), np.float32)
    for k in _COLORSET:
        w_val[0, k, 0, 0] = float(k)
    init("Wval", w_val, np.float32)

    # gray (channel 5) one-hot selector for the pixel-count -> offset.
    g_one = np.zeros((1, 10, 1, 1), np.float32)
    g_one[0, 5, 0, 0] = 1.0
    init("Gone", g_one, np.float32)

    init("ar30r", np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1),
         np.float32)
    init("ar30c", np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30),
         np.float32)
    init("ar10r", np.arange(WORK, dtype=np.float32).reshape(1, 1, WORK, 1),
         np.float32)
    init("ar10c", np.arange(WORK, dtype=np.float32).reshape(1, 1, 1, WORK),
         np.float32)
    init("five", np.array(5.0, np.float32), np.float32)
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    init("zero_u8", np.array(0, np.uint8), np.uint8)
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK],
                             np.int64), np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)

    # ---- recover colour (scalar uint8) ----
    n("ReduceMax", ["input"], "pres", axes=[2, 3], keepdims=1)   # [1,10,1,1]
    n("Mul", ["pres", "Wval"], "presv")                          # [1,10,1,1]
    n("ReduceSum", ["presv"], "colf", axes=[1, 2, 3], keepdims=1)  # [1,1,1,1]
    n("Cast", ["colf"], "color_u8", to=TensorProto.UINT8)        # scalar uint8

    # ---- recover offset (scalar float = gray pixel count) ----
    n("ReduceSum", ["input"], "cnt", axes=[2, 3], keepdims=1)    # [1,10,1,1]
    n("Mul", ["cnt", "Gone"], "graycnt")                         # [1,10,1,1]
    n("ReduceSum", ["graycnt"], "offset", axes=[1, 2, 3], keepdims=1)  # [1,1,1,1]

    # ---- colored-cell map -> row/col counts -> the cross-axis positions ----
    n("Conv", ["input", "Wcol"], "colored")                      # [1,1,30,30] f32
    n("ReduceSum", ["colored"], "rowcnt", axes=[3], keepdims=1)  # [1,1,30,1]
    n("ReduceSum", ["colored"], "colcnt", axes=[2], keepdims=1)  # [1,1,1,30]
    n("Greater", ["rowcnt", "five"], "rowsel")                   # bool [1,1,30,1]
    n("Greater", ["colcnt", "five"], "colsel")                   # bool [1,1,1,30]
    n("Cast", ["rowsel"], "rowselF", to=TensorProto.FLOAT)
    n("Cast", ["colsel"], "colselF", to=TensorProto.FLOAT)
    n("Mul", ["ar30r", "rowselF"], "rprod")                      # [1,1,30,1]
    n("Mul", ["ar30c", "colselF"], "cprod")                      # [1,1,1,30]
    n("ReduceSum", ["rprod"], "rowf", axes=[2], keepdims=1)      # [1,1,1,1] = row
    n("ReduceSum", ["cprod"], "colf2", axes=[3], keepdims=1)     # [1,1,1,1] = col

    n("Add", ["rowf", "offset"], "Rout")                         # row + offset
    n("Sub", ["colf2", "offset"], "Cout")                        # col - offset

    # ---- 10x10 line mask = (r == Rout) OR (c == Cout) ----
    n("Equal", ["ar10r", "Rout"], "rsel10")                      # bool [1,1,10,1]
    n("Equal", ["ar10c", "Cout"], "csel10")                      # bool [1,1,1,10]
    n("Or", ["rsel10", "csel10"], "linemask")                    # bool [1,1,10,10]

    # ---- label map: colour on lines, bg(0) elsewhere; sentinel via Pad ----
    n("Where", ["linemask", "color_u8", "zero_u8"], "L10")       # uint8 [1,1,10,10]
    n("Pad", ["L10", "padpads", "padval"], "L", mode="constant")  # uint8 [1,1,30,30]
    n("Equal", ["L", "chan"], "output")                          # -> free BOOL

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task362", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
