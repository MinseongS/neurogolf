"""task092 (ARC-AGI 40853293) — separable per-colour interval fill.

Rule (from the generator): the grid holds several axis-aligned "sticks", each a
UNIQUE colour, drawn as two endpoint pixels that are colinear (same row OR same
column).  The output fills every cell between (and including) the two endpoints
with that stick's colour.  Because the endpoints are colinear, the filled
segment is exactly the bounding box of the colour's pixels.  Sticks may CROSS;
the generator draws the column sticks AFTER the row sticks, so at a crossing the
COLUMN colour wins.

Memory floor-break (label map + final Equal, only `output` is 2-D over channels):
  Everything is computed as length-30 / 30x30 single planes; no [1,10,30,30]
  intermediate is ever materialised.  Per colour c we get a row-range vector
  rowin_c[r] = (minrow_c<=r<=maxrow_c) and a col-range vector colin_c[col],
  built with triangular prefix/suffix-OR MatMuls (params, not memory).  Each
  channel is a row-stick (spans one row) or a col-stick (spans one column).

  We contract over the channel axis with MatMuls to get two colour planes:
    rowColor[r,col] = sum_c (isrow_c * c) * rowin_c[r] * colin_c[col]
    colColor[r,col] = sum_c (iscol_c * c) * rowin_c[r] * colin_c[col]
  (each cell is covered by at most one row-stick and one col-stick, so each sum
  collapses to that single colour).  The uint8 label map is

    L = colColor          where a col-stick covers the cell   (column wins)
        rowColor          elif a row-stick covers the cell
        0                 elif in-grid                        (background)
        10                else (off-grid)                     (matches nothing)

  and the free BOOL output = Equal(L, arange[1,10,1,1]) (opset 11).  All values
  are small integers, exact in float32 / uint8.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    H = TensorProto.FLOAT16
    U8 = TensorProto.UINT8
    B = TensorProto.BOOL

    # --- per-colour has-row / has-col masks (ch0 kept harmlessly) -----------
    n("ReduceMax", ["input"], "rowhasf", axes=[3], keepdims=1)  # [1,10,30,1]
    n("ReduceMax", ["input"], "colhasf", axes=[2], keepdims=1)  # [1,10,1,30]
    n("Cast", ["rowhasf"], "rowhas", to=H)
    n("Cast", ["colhasf"], "colhas", to=H)

    Ltri = np.tril(np.ones((30, 30), np.float16))
    Utri = np.triu(np.ones((30, 30), np.float16))
    init("Ltri", Ltri, np.float16)
    init("Utri", Utri, np.float16)
    init("zero", np.array(0.0, np.float16), np.float16)
    init("half", np.array(0.5, np.float16), np.float16)
    init("oneh", np.array(1.5, np.float16), np.float16)

    # rowin_c[r] = (exists r'<=r with rowhas) AND (exists r'>=r with rowhas)
    n("MatMul", ["Ltri", "rowhas"], "rpre")
    n("MatMul", ["Utri", "rowhas"], "rsuf")
    n("Greater", ["rpre", "zero"], "rpre_b")
    n("Greater", ["rsuf", "zero"], "rsuf_b")
    n("And", ["rpre_b", "rsuf_b"], "rowin_b")
    n("Cast", ["rowin_b"], "rowin", to=H)           # [1,10,30,1]

    n("Transpose", ["colhas"], "colhasT", perm=[0, 1, 3, 2])    # [1,10,30,1]
    n("MatMul", ["Ltri", "colhasT"], "cpre")
    n("MatMul", ["Utri", "colhasT"], "csuf")
    n("Greater", ["cpre", "zero"], "cpre_b")
    n("Greater", ["csuf", "zero"], "csuf_b")
    n("And", ["cpre_b", "csuf_b"], "colinT_b")
    n("Cast", ["colinT_b"], "colinT", to=H)         # [1,10,30,1] (col axis here)

    # --- classify channel: col-stick spans exactly one column ---------------
    n("ReduceSum", ["colinT"], "colcount", axes=[2], keepdims=1)  # [1,10,1,1]
    n("Greater", ["colcount", "half"], "present")
    n("Less", ["colcount", "oneh"], "single_col")
    n("And", ["present", "single_col"], "iscol_b")
    n("Cast", ["iscol_b"], "iscol", to=H)           # [1,10,1,1]
    # row-stick = present and not col-stick
    n("Cast", ["present"], "presentf", to=H)
    n("Sub", ["presentf", "iscol"], "isrow")        # [1,10,1,1] (1 for row-stick)

    # channel index 0..9 as a [1,10,1,1] weight, zero on ch0 (background)
    carr = np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1)
    init("carr", carr, np.float16)

    # --- squeeze to [10,30] channel-major planes for contraction ------------
    n("Squeeze", ["rowin"], "RW", axes=[0, 3])      # [10,30] channel x row
    n("Squeeze", ["colinT"], "CW", axes=[0, 3])     # [10,30] channel x col
    n("Squeeze", ["iscol"], "iscolv", axes=[0, 2, 3])   # [10]
    n("Squeeze", ["isrow"], "isrowv", axes=[0, 2, 3])   # [10]
    n("Squeeze", ["carr"], "cv", axes=[0, 2, 3])        # [10]

    # weighted row planes:  Rrow[c,r] = isrow_c * c * rowin_c[r]
    n("Mul", ["isrowv", "cv"], "rowwt")             # [10]
    n("Unsqueeze", ["rowwt"], "rowwt1", axes=[1])   # [10,1]
    n("Mul", ["RW", "rowwt1"], "Rrow")              # [10,30]
    n("Transpose", ["Rrow"], "RrowT", perm=[1, 0])  # [30,10]
    n("MatMul", ["RrowT", "CW"], "rowColor")        # [30,30] colour of row-stick

    # weighted col planes:  Rcol[c,r] = iscol_c * c * rowin_c[r]
    n("Mul", ["iscolv", "cv"], "colwt")
    n("Unsqueeze", ["colwt"], "colwt1", axes=[1])
    n("Mul", ["RW", "colwt1"], "Rcol")
    n("Transpose", ["Rcol"], "RcolT", perm=[1, 0])  # [30,10]
    n("MatMul", ["RcolT", "CW"], "colColor")        # [30,30] colour of col-stick

    # coverage planes (1 if a row/col stick covers the cell): a covered cell has
    # colour >= 1, so colour > 0  <=>  covered (stick colours are 1..9).
    n("Greater", ["rowColor", "zero"], "rowcov_b")
    n("Greater", ["colColor", "zero"], "colcov_b")

    # --- assemble label map L[30,30] ---------------------------------------
    n("ReduceMax", ["input"], "ingridf", axes=[1], keepdims=1)  # [1,1,30,30]
    n("Squeeze", ["ingridf"], "ingrid2", axes=[0, 1])          # [30,30] f32
    init("zerof", np.array(0.0, np.float32), np.float32)
    n("Greater", ["ingrid2", "zerof"], "ingrid_b")

    n("Cast", ["rowColor"], "rowColorU8", to=U8)
    n("Cast", ["colColor"], "colColorU8", to=U8)
    init("v0", np.array(0, np.uint8), np.uint8)
    init("v10", np.array(10, np.uint8), np.uint8)
    # base: 0 in-grid else 10
    n("Where", ["ingrid_b", "v0", "v10"], "Lbase")             # [30,30]
    # row sticks on top of background
    n("Where", ["rowcov_b", "rowColorU8", "Lbase"], "Lrow")
    # col sticks win over everything
    n("Where", ["colcov_b", "colColorU8", "Lrow"], "L2")       # [30,30]
    init("Lshape", np.array([1, 1, 30, 30], np.int64), np.int64)
    n("Reshape", ["L2", "Lshape"], "L")                        # [1,1,30,30]

    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["L", "chan"], "output")                        # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task092", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
