"""Task 062 (ARC-AGI 2bcee788): mirror-reconstruct a sprite, recolor, green bg.

Generator: a 3x3 sprite (always containing a column-0 cell) is drawn in `color`
on a 10x10 grid at offset (row in 1..6, col in 4..6). Each sprite cell (r,c)
paints grid[row+r][col+c]=color; the MIRROR cell grid[row+r][col-c-1] gets a
red(2) marker only for c==0 (c>0 mirror cells stay background in the input).
The OUTPUT paints BOTH the sprite and its mirror across the axis at col-0.5 in
`color`, on a green(3) background, then a random flip_horiz / transpose is
applied to grid and output identically (axis is vertical or horizontal, either
way).  Size is ALWAYS 10x10.

Compact rule (verified exact on all 267 stored examples and 200/200 fresh):
  - red R = channel 2 ;  colormask C = cells whose colour is not 0 or 2.
  - centroids: rc=(col,row centroid of R), cc=(col,row centroid of C).
    The red markers sit exactly on the mirror of the c==0 sprite column, so the
    axis is the perpendicular bisector between the red line and that column.
    d = cc - rc.  Axis is VERTICAL iff |d_col| > |d_row|.
    axis-sum s = 2*rc[active] + sign(d[active])  (reflection maps x -> s-x).
  - reflection col j -> s-j (vertical) / row i -> s-i (horizontal) via Gather on
    a tiny index vector (negative targets redirected to all-zero index 29);
    refl = vert ? reflV : reflH.
  - M = (C OR refl) restricted to the fixed 10x10 region.
  - colorvec[1,10,1,1] = per-channel ReduceMax of input, zeroed on ch 0,2,3
    -> one-hot of the sprite colour.
  - output = Where(M, colorvec, BG) with BG the fixed green-in-[0:10,0:10]
    constant plane; the single Where writes free `output`, so no [1,10,30,30]
    intermediate is materialised.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..builders import _model


def build(task):
    inits, nodes, vinfos = [], [], []

    def init(name, arr, dtype=np.float32):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    def vi(name, dtype, shape):
        vinfos.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    F = TensorProto.FLOAT
    F16 = TensorProto.FLOAT16
    B = TensorProto.BOOL
    I32 = TensorProto.INT32

    # ---- colormask C (needed full for the reflection Gather) ---------------
    Wc = np.ones((1, 10, 1, 1), np.float32); Wc[0, 0, 0, 0] = 0.0; Wc[0, 2, 0, 0] = 0.0
    init("Wc", Wc)
    n("Conv", ["input", "Wc"], "C"); vi("C", F, [1, 1, 30, 30])   # colormask

    # ---- red column / row profiles straight from `input` (no R canvas) -----
    # Wrc[1,10,30,1] sums channel-2 over the 30 rows -> [1,1,1,30] (red col prof)
    # Wrr[1,10,1,30] sums channel-2 over the 30 cols -> [1,1,30,1] (red row prof)
    Wrc = np.zeros((1, 10, 30, 1), np.float32); Wrc[0, 2, :, 0] = 1.0
    Wrr = np.zeros((1, 10, 1, 30), np.float32); Wrr[0, 2, 0, :] = 1.0
    init("Wrc", Wrc); init("Wrr", Wrr)
    n("Conv", ["input", "Wrc"], "Rcp"); vi("Rcp", F, [1, 1, 1, 30])
    n("Conv", ["input", "Wrr"], "Rrp"); vi("Rrp", F, [1, 1, 30, 1])

    colk = np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30)
    rowk = np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1)
    init("colk", colk); init("rowk", rowk)

    def centroid_from_profiles(cprof, rprof, pre):
        K = n("ReduceSum", [cprof], pre + "K", axes=[1, 2, 3], keepdims=1)
        vi(K, F, [1, 1, 1, 1])
        cs = n("Conv", [cprof, "colk"], pre + "cs"); vi(cs, F, [1, 1, 1, 1])
        rs = n("Conv", [rprof, "rowk"], pre + "rs"); vi(rs, F, [1, 1, 1, 1])
        ccol = n("Div", [cs, K], pre + "col"); vi(ccol, F, [1, 1, 1, 1])
        crow = n("Div", [rs, K], pre + "row"); vi(crow, F, [1, 1, 1, 1])
        return ccol, crow

    rcol, rrow = centroid_from_profiles("Rcp", "Rrp", "R")
    # C profiles via ReduceSum on the C canvas
    n("ReduceSum", ["C"], "Ccp", axes=[2], keepdims=1); vi("Ccp", F, [1, 1, 1, 30])
    n("ReduceSum", ["C"], "Crp", axes=[3], keepdims=1); vi("Crp", F, [1, 1, 30, 1])
    ccol, crow = centroid_from_profiles("Ccp", "Crp", "C")

    # d = cc - rc ; vertical iff |dcol| > |drow|
    n("Sub", [ccol, rcol], "dcol"); vi("dcol", F, [1, 1, 1, 1])
    n("Sub", [crow, rrow], "drow"); vi("drow", F, [1, 1, 1, 1])
    n("Abs", ["dcol"], "adcol"); vi("adcol", F, [1, 1, 1, 1])
    n("Abs", ["drow"], "adrow"); vi("adrow", F, [1, 1, 1, 1])
    n("Greater", ["adcol", "adrow"], "vertb"); vi("vertb", B, [1, 1, 1, 1])

    # active centroid r_act and active diff d_act (vert? col : row)
    n("Where", ["vertb", rcol, rrow], "ract"); vi("ract", F, [1, 1, 1, 1])
    n("Where", ["vertb", "dcol", "drow"], "dact"); vi("dact", F, [1, 1, 1, 1])
    # s = 2*ract + sign(dact)
    init("two", np.array(2.0, np.float32))
    n("Mul", ["ract", "two"], "r2"); vi("r2", F, [1, 1, 1, 1])
    n("Sign", ["dact"], "sgn"); vi("sgn", F, [1, 1, 1, 1])
    n("Add", ["r2", "sgn"], "s"); vi("s", F, [1, 1, 1, 1])

    # ---- reflection index vectors -> Gather on fp16 copy of C -------------
    # (reuse the centroid index kernels colk/rowk -- identical arange vectors)
    init("nhalf", np.array(-0.5, np.float32))
    init("c29", np.array(29.0, np.float32))
    init("shp30", np.array([30], np.int64), np.int64)

    n("Cast", ["C"], "Cf", to=F16); vi("Cf", F16, [1, 1, 30, 30])
    init("c95", np.array(9.5, np.float32))
    # NOT-vert (for gating the inactive reflection to all-zero)
    n("Not", ["vertb"], "horb"); vi("horb", B, [1, 1, 1, 1])

    # gather reflection; an index is redirected to the always-zero index 29 when
    # the target leaves the grid (s-j<0), the output cell is off-grid (j>=10), or
    # this axis is the INACTIVE orientation (gate).  So the inactive reflection
    # is all-zero and the two can simply be summed -- no blend Where, and the
    # 10x10 output clamp is folded in (C is already confined to the grid).
    def safe_index(basevec, gate, outname, shapeC, axis):
        n("Sub", ["s", basevec], outname + "g"); vi(outname + "g", F, shapeC)
        n("Less", [outname + "g", "nhalf"], outname + "neg"); vi(outname + "neg", B, shapeC)
        n("Greater", [basevec, "c95"], outname + "off"); vi(outname + "off", B, shapeC)
        n("Or", [outname + "neg", outname + "off"], outname + "b1"); vi(outname + "b1", B, shapeC)
        n("Or", [outname + "b1", gate], outname + "bad"); vi(outname + "bad", B, shapeC)
        n("Where", [outname + "bad", "c29", outname + "g"], outname + "ss"); vi(outname + "ss", F, shapeC)
        n("Cast", [outname + "ss"], outname + "i", to=I32); vi(outname + "i", I32, shapeC)
        n("Reshape", [outname + "i", "shp30"], outname + "f"); vi(outname + "f", I32, [30])
        n("Gather", ["Cf", outname + "f"], outname, axis=axis); vi(outname, F16, [1, 1, 30, 30])

    safe_index("colk", "horb", "reflV", [1, 1, 1, 30], 3)   # active when vertical
    safe_index("rowk", "vertb", "reflH", [1, 1, 30, 1], 2)  # active when horizontal

    # M = (C + reflV + reflH) > 0.5   (only the active reflection is nonzero)
    init("half16", np.array(0.5, np.float16), np.float16)
    n("Sum", ["Cf", "reflV", "reflH"], "Msum"); vi("Msum", F16, [1, 1, 30, 30])
    n("Greater", ["Msum", "half16"], "Mb"); vi("Mb", B, [1, 1, 30, 30])

    # ---- color one-hot vector ---------------------------------------------
    n("ReduceMax", ["input"], "chmax", axes=[2, 3], keepdims=1); vi("chmax", F, [1, 10, 1, 1])
    chmask = np.ones((1, 10, 1, 1), np.float32)
    chmask[0, 0, 0, 0] = chmask[0, 2, 0, 0] = chmask[0, 3, 0, 0] = 0.0
    init("chmask", chmask)
    n("Mul", ["chmax", "chmask"], "colorvec"); vi("colorvec", F, [1, 10, 1, 1])

    # ---- output = Where(Mb, colorvec, BG) ---------------------------------
    BG = np.zeros((1, 10, 30, 30), np.float32); BG[0, 3, :10, :10] = 1.0
    init("BG", BG)
    n("Where", ["Mb", "colorvec", "BG"], "output")

    return _model(nodes, inits, vinfos)
