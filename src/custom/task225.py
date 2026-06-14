"""Task 225 (ARC-AGI 93b581b8): stamp the 2x2 colour block at the four
diagonal corner offsets (+-2).

Rule (verified exact on all 265 stored examples + fresh arc-gen). The 6x6 grid
holds a single 2x2 block of four DISTINCT colours at top-left (row, col)
(row, col in 1..3). Writing the block cells as

      c0 c1            (c0 at (row,   col),   c1 at (row,   col+1),
      c2 c3             c2 at (row+1, col),   c3 at (row+1, col+1))

the output keeps the block and additionally stamps a 2x2 monochrome block of
each colour at a diagonal corner offset:

      c0 -> top-left (row+2, col+2)   c1 -> (row+2, col-2)
      c2 -> (row-2, col+2)            c3 -> (row-2, col-2)

(out-of-bounds parts are clipped away). The block and all four stamps are
pairwise disjoint for every (row, col), so the transform is exactly

      output = input + stamps.

Construction (no flood-fill, all aggregates 1-D):
  * occ = sum of colour channels -> per-row / per-col occupancy.
  * row, col scalars: the block fills exactly two consecutive rows {row,row+1};
    sum of occupied row indices = 2*row+1, so row = (sum-1)/2 (same for col).
  * corner colours: rowK[r] (sum of block colours in row r) gives the channel
    set {c0,c1} at row `row` and {c2,c3} at row+1; colK[c] gives {c0,c2} at col
    and {c1,c3} at col+1. As the four colours are DISTINCT (random_colors(4)),
    each corner one-hot is the intersection (elementwise product) of a row-set
    and a col-set, e.g. c0 = rowtop * colleft.
  * stamp region masks are outer products of row/col range indicators built from
    the scalar row/col and fixed index planes.
  * stamps = c0*Mpp + c1*Mpm + c2*Mmp + c3*Mmm, then output = input + stamps.

Intermediates are 1-D row/col vectors and bool [1,1,30,30] masks; the only
10-channel float canvases are the two stamp half-planes summed straight into
the free output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..builders import _model


def build(task):
    inits = []
    nodes = []
    vinfos = []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    def vi(name, dtype, shape):
        vinfos.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    # ---- occupancy occ[1,1,30,30] = sum of colour channels 1..9 ------------
    wocc = np.zeros((1, 10, 1, 1), np.float32)
    wocc[0, 1:, 0, 0] = 1.0
    init("Wocc", wocc, np.float32)
    n("Conv", ["input", "Wocc"], "occ")                       # [1,1,30,30] f32
    vi("occ", TensorProto.FLOAT, [1, 1, 30, 30])

    # ---- per-row / per-col occupancy counts -------------------------------
    n("ReduceSum", ["occ"], "rowcnt", axes=[3], keepdims=1)   # [1,1,30,1] f32
    vi("rowcnt", TensorProto.FLOAT, [1, 1, 30, 1])
    n("ReduceSum", ["occ"], "colcnt", axes=[2], keepdims=1)   # [1,1,1,30] f32
    vi("colcnt", TensorProto.FLOAT, [1, 1, 1, 30])

    init("zero_f", np.array(0.0, np.float32), np.float32)
    n("Greater", ["rowcnt", "zero_f"], "occr_b")              # [1,1,30,1] bool
    vi("occr_b", TensorProto.BOOL, [1, 1, 30, 1])
    n("Greater", ["colcnt", "zero_f"], "occc_b")              # [1,1,1,30] bool
    vi("occc_b", TensorProto.BOOL, [1, 1, 1, 30])
    n("Cast", ["occr_b"], "occr", to=TensorProto.INT32)
    vi("occr", TensorProto.INT32, [1, 1, 30, 1])
    n("Cast", ["occc_b"], "occc", to=TensorProto.INT32)
    vi("occc", TensorProto.INT32, [1, 1, 1, 30])

    # fixed index vectors (int32)
    ridx = np.arange(30, dtype=np.int32).reshape(1, 1, 30, 1)
    cidx = np.arange(30, dtype=np.int32).reshape(1, 1, 1, 30)
    init("RIDX", ridx, np.int32)                              # [1,1,30,1]
    init("CIDX", cidx, np.int32)                              # [1,1,1,30]

    # sum of occupied row indices = 2*row+1  -> row = (sum-1)/2
    n("Mul", ["occr", "RIDX"], "rowiw")                       # [1,1,30,1] i32
    vi("rowiw", TensorProto.INT32, [1, 1, 30, 1])
    n("ReduceSum", ["rowiw"], "rowsum", axes=[2], keepdims=1)  # [1,1,1,1]
    vi("rowsum", TensorProto.INT32, [1, 1, 1, 1])
    n("Mul", ["occc", "CIDX"], "coliw")                       # [1,1,1,30] i32
    vi("coliw", TensorProto.INT32, [1, 1, 1, 30])
    n("ReduceSum", ["coliw"], "colsum", axes=[3], keepdims=1)  # [1,1,1,1]
    vi("colsum", TensorProto.INT32, [1, 1, 1, 1])

    init("one_i", np.array(1, np.int32), np.int32)
    init("two_i", np.array(2, np.int32), np.int32)
    n("Sub", ["rowsum", "one_i"], "rowsm1")
    vi("rowsm1", TensorProto.INT32, [1, 1, 1, 1])
    n("Div", ["rowsm1", "two_i"], "ROW")                      # scalar row
    vi("ROW", TensorProto.INT32, [1, 1, 1, 1])
    n("Sub", ["colsum", "one_i"], "colsm1")
    vi("colsm1", TensorProto.INT32, [1, 1, 1, 1])
    n("Div", ["colsm1", "two_i"], "COL")                      # scalar col
    vi("COL", TensorProto.INT32, [1, 1, 1, 1])

    # ---- row/col equality indicators (block rows row, row+1; cols col,col+1)
    n("Add", ["ROW", "one_i"], "ROW1")
    vi("ROW1", TensorProto.INT32, [1, 1, 1, 1])
    n("Add", ["COL", "one_i"], "COL1")
    vi("COL1", TensorProto.INT32, [1, 1, 1, 1])

    # Equal(RIDX, scalar) -> bool [1,1,30,1] ; cast to float for reductions
    n("Equal", ["RIDX", "ROW"], "rTop_b")
    vi("rTop_b", TensorProto.BOOL, [1, 1, 30, 1])
    n("Equal", ["RIDX", "ROW1"], "rBot_b")
    vi("rBot_b", TensorProto.BOOL, [1, 1, 30, 1])
    n("Equal", ["CIDX", "COL"], "cLft_b")
    vi("cLft_b", TensorProto.BOOL, [1, 1, 1, 30])
    n("Equal", ["CIDX", "COL1"], "cRgt_b")
    vi("cRgt_b", TensorProto.BOOL, [1, 1, 1, 30])
    n("Cast", ["rTop_b"], "rTop", to=TensorProto.FLOAT)
    vi("rTop", TensorProto.FLOAT, [1, 1, 30, 1])
    n("Cast", ["rBot_b"], "rBot", to=TensorProto.FLOAT)
    vi("rBot", TensorProto.FLOAT, [1, 1, 30, 1])
    n("Cast", ["cLft_b"], "cLft", to=TensorProto.FLOAT)
    vi("cLft", TensorProto.FLOAT, [1, 1, 1, 30])
    n("Cast", ["cRgt_b"], "cRgt", to=TensorProto.FLOAT)
    vi("cRgt", TensorProto.FLOAT, [1, 1, 1, 30])

    # ---- corner colour one-hots (distinct colours -> intersection product) -
    # rowK[1,10,30,1] = per-row channel sums; colK[1,10,1,30] per-col.
    # (background channel 0 is summed too, but is killed below by masking the
    #  row-set vectors with CHM before the row/col intersection.)
    n("ReduceSum", ["input"], "rowK", axes=[3], keepdims=1)   # [1,10,30,1] f32
    vi("rowK", TensorProto.FLOAT, [1, 10, 30, 1])
    n("ReduceSum", ["input"], "colK", axes=[2], keepdims=1)   # [1,10,1,30] f32
    vi("colK", TensorProto.FLOAT, [1, 10, 1, 30])

    chm = np.ones((1, 10, 1, 1), np.float32); chm[0, 0, 0, 0] = 0.0
    init("CHM", chm, np.float32)

    # rowtop[1,10,1,1] = rowK at row `row` = {c0,c1}; rowbot = {c2,c3}
    n("Mul", ["rowK", "rTop"], "rowK_t")                      # [1,10,30,1]
    vi("rowK_t", TensorProto.FLOAT, [1, 10, 30, 1])
    n("ReduceSum", ["rowK_t"], "rowtop0", axes=[2], keepdims=1)  # [1,10,1,1]
    vi("rowtop0", TensorProto.FLOAT, [1, 10, 1, 1])
    n("Mul", ["rowtop0", "CHM"], "rowtop")                    # drop channel 0
    vi("rowtop", TensorProto.FLOAT, [1, 10, 1, 1])
    n("Mul", ["rowK", "rBot"], "rowK_b")
    vi("rowK_b", TensorProto.FLOAT, [1, 10, 30, 1])
    n("ReduceSum", ["rowK_b"], "rowbot0", axes=[2], keepdims=1)
    vi("rowbot0", TensorProto.FLOAT, [1, 10, 1, 1])
    n("Mul", ["rowbot0", "CHM"], "rowbot")                    # drop channel 0
    vi("rowbot", TensorProto.FLOAT, [1, 10, 1, 1])

    # colleft = {c0,c2}; colright = {c1,c3}
    n("Mul", ["colK", "cLft"], "colK_l")                      # [1,10,1,30]
    vi("colK_l", TensorProto.FLOAT, [1, 10, 1, 30])
    n("ReduceSum", ["colK_l"], "colleft", axes=[3], keepdims=1)  # [1,10,1,1]
    vi("colleft", TensorProto.FLOAT, [1, 10, 1, 1])
    n("Mul", ["colK", "cRgt"], "colK_r")
    vi("colK_r", TensorProto.FLOAT, [1, 10, 1, 30])
    n("ReduceSum", ["colK_r"], "colright", axes=[3], keepdims=1)
    vi("colright", TensorProto.FLOAT, [1, 10, 1, 1])

    # The four stamps factor as a single rank-1 outer product per channel:
    #   stampcolor[k,r,c] = R[k,r] * C[k,c]
    # with R = rowtop*rowP + rowbot*rowM   (the {c0,c1}/{c2,c3} colours live on
    # the +/- row regions) and C = colleft*colP + colright*colM. This collapses
    # the colour field into ONE canvas-sized Mul (built from 1-D vectors).

    # ---- stamp region row/col range indicators ----------------------------
    # rowP covers {row+2,row+3}; rowM covers {row-2,row-1}
    # colP covers {col+2,col+3}; colM covers {col-2,col-1}
    init("two_i2", np.array(2, np.int32), np.int32)
    init("three_i", np.array(3, np.int32), np.int32)
    n("Add", ["ROW", "two_i2"], "rP0")                        # row+2
    vi("rP0", TensorProto.INT32, [1, 1, 1, 1])
    n("Add", ["ROW", "three_i"], "rP1")                       # row+3
    vi("rP1", TensorProto.INT32, [1, 1, 1, 1])
    n("Sub", ["ROW", "two_i2"], "rM0")                        # row-2
    vi("rM0", TensorProto.INT32, [1, 1, 1, 1])
    n("Sub", ["ROW", "one_i"], "rM1")                         # row-1
    vi("rM1", TensorProto.INT32, [1, 1, 1, 1])
    n("Add", ["COL", "two_i2"], "cP0")
    vi("cP0", TensorProto.INT32, [1, 1, 1, 1])
    n("Add", ["COL", "three_i"], "cP1")
    vi("cP1", TensorProto.INT32, [1, 1, 1, 1])
    n("Sub", ["COL", "two_i2"], "cM0")
    vi("cM0", TensorProto.INT32, [1, 1, 1, 1])
    n("Sub", ["COL", "one_i"], "cM1")
    vi("cM1", TensorProto.INT32, [1, 1, 1, 1])

    def rangemask(idxname, lo, hi, axisname, shape, fprefix):
        # (idx >= lo) & (idx <= hi) -> float, naturally 0 outside [0,29]
        ge = n("Not", [n("Less", [idxname, lo], fprefix + "_lt")],
               fprefix + "_ge")
        vi(fprefix + "_lt", TensorProto.BOOL, shape)
        vi(fprefix + "_ge", TensorProto.BOOL, shape)
        # idx <= hi  <=> not (hi < idx)
        n("Less", [hi, idxname], fprefix + "_gt")
        vi(fprefix + "_gt", TensorProto.BOOL, shape)
        n("Not", [fprefix + "_gt"], fprefix + "_le")
        vi(fprefix + "_le", TensorProto.BOOL, shape)
        n("And", [fprefix + "_ge", fprefix + "_le"], fprefix + "_b")
        vi(fprefix + "_b", TensorProto.BOOL, shape)
        n("Cast", [fprefix + "_b"], fprefix, to=TensorProto.FLOAT)
        vi(fprefix, TensorProto.FLOAT, shape)
        return fprefix

    rshape = [1, 1, 30, 1]
    cshape = [1, 1, 1, 30]
    rangemask("RIDX", "rP0", "rP1", None, rshape, "rowP")
    rangemask("RIDX", "rM0", "rM1", None, rshape, "rowM")
    rangemask("CIDX", "cP0", "cP1", None, cshape, "colP")
    rangemask("CIDX", "cM0", "cM1", None, cshape, "colM")

    # ---- factored colour field: R[1,10,30,1] and C[1,10,1,30] -------------
    # R = rowtop*rowP + rowbot*rowM   (rowtop/rowbot broadcast over the 30 rows)
    n("Mul", ["rowtop", "rowP"], "Rp")                       # [1,10,30,1]
    vi("Rp", TensorProto.FLOAT, [1, 10, 30, 1])
    n("Mul", ["rowbot", "rowM"], "Rm")                       # [1,10,30,1]
    vi("Rm", TensorProto.FLOAT, [1, 10, 30, 1])
    n("Add", ["Rp", "Rm"], "Rrow")                           # [1,10,30,1]
    vi("Rrow", TensorProto.FLOAT, [1, 10, 30, 1])
    # C = colleft*colP + colright*colM
    n("Mul", ["colleft", "colP"], "Cp")                      # [1,10,1,30]
    vi("Cp", TensorProto.FLOAT, [1, 10, 1, 30])
    n("Mul", ["colright", "colM"], "Cm")                     # [1,10,1,30]
    vi("Cm", TensorProto.FLOAT, [1, 10, 1, 30])
    n("Add", ["Cp", "Cm"], "Ccol")                           # [1,10,1,30]
    vi("Ccol", TensorProto.FLOAT, [1, 10, 1, 30])
    # stampcolor = R * C  -> the ONLY canvas-sized colour tensor
    n("Mul", ["Rrow", "Ccol"], "stampcolor")                # [1,10,30,30] f32
    vi("stampcolor", TensorProto.FLOAT, [1, 10, 30, 30])

    # ---- stamp region mask = (rowP|rowM) & (colP|colM) --------------------
    # reuse the bool range masks built inside rangemask().
    n("Or", ["rowP_b", "rowM_b"], "rowPM_b")                 # [1,1,30,1] bool
    vi("rowPM_b", TensorProto.BOOL, [1, 1, 30, 1])
    n("Or", ["colP_b", "colM_b"], "colPM_b")                 # [1,1,1,30] bool
    vi("colPM_b", TensorProto.BOOL, [1, 1, 1, 30])
    # clip to the fixed 6x6 output grid (rows/cols 0..5); out-of-grid stamp
    # parts are dropped by the generator.
    init("six_i", np.array(6, np.int32), np.int32)
    n("Less", ["RIDX", "six_i"], "rin6")                     # [1,1,30,1] bool
    vi("rin6", TensorProto.BOOL, [1, 1, 30, 1])
    n("Less", ["CIDX", "six_i"], "cin6")                     # [1,1,1,30] bool
    vi("cin6", TensorProto.BOOL, [1, 1, 1, 30])
    n("And", ["rowPM_b", "rin6"], "rowok")                   # [1,1,30,1] bool
    vi("rowok", TensorProto.BOOL, [1, 1, 30, 1])
    n("And", ["colPM_b", "cin6"], "colok")                   # [1,1,1,30] bool
    vi("colok", TensorProto.BOOL, [1, 1, 1, 30])
    n("And", ["rowok", "colok"], "stampmask")               # [1,1,30,30] bool
    vi("stampmask", TensorProto.BOOL, [1, 1, 30, 30])

    # output = Where(stampmask, stampcolor, input) -> free output
    n("Where", ["stampmask", "stampcolor", "input"], "output")

    return _model(nodes, inits, vinfos)
