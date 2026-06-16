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

    stickColor = colColor where a col-stick covers (column wins), else rowColor.
    L          = stickColor + offrow + offcol     where
      offrow[r] = 10*(r>=H), offcol[c] = 10*(c>=W)   are 1-D [30,1]/[1,30] penalties
      summed in ONE variadic Sum (broadcast).  In-grid cells keep stickColor (0 for
      background -> ch0=1); off-grid cells become >=10 so Equal(L, arange[0..9])
      matches NO channel -> all-zero, which is the target for cells outside the grid.

  H,W (the grid frame) are recovered offset-free from per-row/col occupancy:
    rowframe[r] = ReduceMax(rowhas, ch-axis) = 1 iff r<H  (bg ch0=1 fills every in-grid
    cell), colframe[c] similarly.  No [1,1,30,30] in-grid plane is built.

  The free BOOL output = Equal(L_u8[1,1,30,30], arange[0..9][1,10,1,1]) (opset 11).
  All values are small integers, exact in float32 / float16 / uint8.

  Verified: official 3/3, fresh 500/500.  pts 14.866, mem 23350, params 1829
  (vs prior stored 14.56 / mem 32350).  Dominant intermediates: 2 colour MatMul
  [30,30] f16 (1800 each) + the col-priority Where + the sentinel Sum [30,30] f16
  (1800 each) + the two fp32 ReduceMax occupancy reductions (1200 each, forced
  because ORT ReduceMax needs float and the input is fp32).
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
    n("ReduceMax", ["input"], "rowhasf", axes=[3], keepdims=1)  # [1,10,30,1] f32
    n("ReduceMax", ["input"], "colhasf", axes=[2], keepdims=1)  # [1,10,1,30] f32
    n("Cast", ["rowhasf"], "rowhas", to=H)
    n("Cast", ["colhasf"], "colhas", to=H)
    # grid frame (r<H / c<W) from the per-row/col occupancy over the channel axis:
    # any channel set (bg ch0=1 fills every in-grid cell) => 1 for in-grid rows/cols.
    n("ReduceMax", ["rowhas"], "rowframe", axes=[1], keepdims=1)  # [1,1,30,1] f16
    n("ReduceMax", ["colhas"], "colframe", axes=[1], keepdims=1)  # [1,1,1,30] f16

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

    # col side: prefix/suffix-OR along the COL axis (axis 3) by RIGHT-multiplying
    # colhas[1,10,1,30] by a triangular [30,30] (no transpose needed).
    n("MatMul", ["colhas", "Utri"], "cpre")    # prefix: c'<=c needs Utri[c',c]=1 iff c'<=c
    n("MatMul", ["colhas", "Ltri"], "csuf")    # suffix
    n("Greater", ["cpre", "zero"], "cpre_b")
    n("Greater", ["csuf", "zero"], "csuf_b")
    n("And", ["cpre_b", "csuf_b"], "colin_b")
    n("Cast", ["colin_b"], "colin4", to=H)          # [1,10,1,30]

    # --- classify channel: col-stick spans exactly one column ---------------
    n("ReduceSum", ["colin4"], "colcount", axes=[3], keepdims=1)  # [1,10,1,1]
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
    n("Squeeze", ["colin4"], "CW", axes=[0, 2])     # [10,30] channel x col
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

    # --- assemble a single f16 label plane, all in fp16 (no fp32 30x30) -----
    # stickColor = colColor where a col-stick covers, else rowColor (col wins).
    # A covered cell has colour 1..9 (>0).  colColor>0 => use colColor, else rowColor.
    n("Greater", ["colColor", "zero"], "colcov_b")             # [30,30] bool
    n("Where", ["colcov_b", "colColor", "rowColor"], "stickColor")  # [30,30] f16

    # off-grid sentinel via 1-D penalties: a cell with r>=H or c>=W must map to a
    # label that matches NO channel (>=10).  offrow[r] = 10*(1-rowframe[r]) [30,1],
    # offcol[c] = 10*(1-colframe[c]) [1,30].  L = stickColor + offrow + offcol in ONE
    # variadic Sum (broadcast) -> exactly stickColor in-grid (offrow=offcol=0), >=10
    # off-grid (so Equal matches nothing -> all-zero target).
    n("Squeeze", ["rowframe"], "rfr", axes=[0, 1, 3])          # [30] f16 {0,1}
    n("Squeeze", ["colframe"], "cfr", axes=[0, 1, 2])          # [30] f16 {0,1}
    init("ten", np.array(10.0, np.float16), np.float16)
    init("onef", np.array(1.0, np.float16), np.float16)
    n("Sub", ["onef", "rfr"], "nrfr")                          # [30] {0,1}
    n("Sub", ["onef", "cfr"], "ncfr")                          # [30]
    n("Mul", ["nrfr", "ten"], "offrow0")                       # [30]
    n("Mul", ["ncfr", "ten"], "offcol0")                       # [30]
    n("Unsqueeze", ["offrow0"], "offrow", axes=[1])            # [30,1]
    n("Unsqueeze", ["offcol0"], "offcol", axes=[0])            # [1,30]
    n("Sum", ["stickColor", "offrow", "offcol"], "L2f")        # [30,30] f16
    n("Cast", ["L2f"], "L2", to=U8)                            # [30,30] uint8
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
