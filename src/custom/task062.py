"""Task 062 (ARC-AGI 2bcee788): mirror-reconstruct a sprite, recolor, green bg.

Generator (size ALWAYS 10x10): a connected 3x3 sprite (always containing a
column-0 cell) is drawn in `color` (color in {1,4,5,6,7,8,9}) at offset (row in
1..6, col in 4..6).  Each sprite cell (r,c) paints grid[row+r][col+c]=color; the
MIRROR cell grid[row+r][col-c-1] gets a red(2) marker only for c==0 (c>0 mirror
cells stay background=0 in the input).  The OUTPUT paints BOTH the sprite and
its full mirror across the axis at col-0.5 in `color`, on a green(3)
background.  A random flip_horiz / transpose is then applied to grid and output
identically (so the reflection axis is vertical or horizontal).

Exact rule (verified 0/500 fresh in numpy, see tasklog):
  C = colored sprite mask (colour not in {0,2}); R = red(2) mask.
  centroids ccol/crow (of C), rcol/rrow (of R).  d = c-centroid - r-centroid.
  VERTICAL axis iff |dcol| > |drow|.
  axis-sum s = 2*r_active + sign(d_active)  (reflection maps x -> s-x).
  refl = reflect(C) across that axis (Gather on a tiny index vector; off-grid
  targets redirected to the always-zero index 9).  M = C OR refl.
  colour one-hot = per-channel ReduceMax(input) zeroed on channels 0,2,3.
  output (BOOL one-hot, routed into the FREE output) =
      offgrid -> ch0    | ingrid & ~M -> ch3(green)    | ingrid & M -> colour.
  Built as 3 associated And(spatialmask, channelvec) ORed together so no
  [1,10,30,30] carrier is ever materialised.

Memory is dominated by the 10x10 reflection planes (fp16, 200B each) -- the
whole working canvas is the 10x10 active grid, never 30x30.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..builders import _model

W = 10  # active grid is always 10x10


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

    colored_chans = [1, 4, 5, 6, 7, 8, 9]

    # ---- colored occupancy plane (10x10) for the reflection Gather ----------
    # 1x1 Conv contracts the 10 channels -> single occupancy plane at 30x30,
    # then a single Slice (both spatial axes at once) crops to the 10x10 active
    # grid.  C10 doubles as the source for the colored centroid profiles.
    Wocc = np.zeros((1, 10, 1, 1), np.float32)
    for k in colored_chans:
        Wocc[0, k, 0, 0] = 1.0
    init("Wocc", Wocc)
    init("z00", np.array([0, 0], np.int64), np.int64)
    init("t1010", np.array([10, 10], np.int64), np.int64)
    init("ax23", np.array([2, 3], np.int64), np.int64)
    n("Conv", ["input", "Wocc"], "C30"); vi("C30", F, [1, 1, 30, 30])
    n("Slice", ["C30", "z00", "t1010", "ax23"], "C10"); vi("C10", F, [1, 1, 10, 10])
    n("Cast", ["C10"], "Cf", to=F16); vi("Cf", F16, [1, 1, 10, 10])

    # ---- red 1-D profiles (channel 2 only), already cropped to 10 ----------
    # Wr_col [1,1,10,30]: sum ch2 over the first 10 rows -> [1,1,1,30] col prof
    # Wr_row [1,1,30,10]: sum ch2 over the first 10 cols -> [1,1,30,1] row prof
    # (we then Slice to 10).  Use spatial-cropping Convs that read only ch2.
    Wr_col = np.zeros((1, 10, 30, 1), np.float32); Wr_col[0, 2, :, 0] = 1.0
    Wr_row = np.zeros((1, 10, 1, 30), np.float32); Wr_row[0, 2, 0, :] = 1.0
    init("Wr_col", Wr_col); init("Wr_row", Wr_row)
    init("z0", np.array([0], np.int64), np.int64)
    init("t10", np.array([10], np.int64), np.int64)
    init("ax3", np.array([3], np.int64), np.int64)
    init("ax2", np.array([2], np.int64), np.int64)
    n("Conv", ["input", "Wr_col"], "Rcp30"); vi("Rcp30", F, [1, 1, 1, 30])
    n("Conv", ["input", "Wr_row"], "Rrp30"); vi("Rrp30", F, [1, 1, 30, 1])
    n("Slice", ["Rcp30", "z0", "t10", "ax3"], "Rcp"); vi("Rcp", F, [1, 1, 1, 10])
    n("Slice", ["Rrp30", "z0", "t10", "ax2"], "Rrp"); vi("Rrp", F, [1, 1, 10, 1])

    # colored profiles straight off C10 (ReduceSum, no extra kernels)
    n("ReduceSum", ["C10"], "Ccp", axes=[2], keepdims=1); vi("Ccp", F, [1, 1, 1, 10])
    n("ReduceSum", ["C10"], "Crp", axes=[3], keepdims=1); vi("Crp", F, [1, 1, 10, 1])

    # ---- centroids (all scalars) -------------------------------------------
    colk = np.arange(W, dtype=np.float32).reshape(1, 1, 1, W)
    rowk = np.arange(W, dtype=np.float32).reshape(1, 1, W, 1)
    init("colk", colk); init("rowk", rowk)

    def centroid(cprof, rprof, pre):
        K = n("ReduceSum", [cprof], pre + "K", axes=[1, 2, 3], keepdims=1)
        vi(K, F, [1, 1, 1, 1])
        cs = n("Conv", [cprof, "colk"], pre + "cs"); vi(cs, F, [1, 1, 1, 1])
        rs = n("Conv", [rprof, "rowk"], pre + "rs"); vi(rs, F, [1, 1, 1, 1])
        ccol = n("Div", [cs, K], pre + "col"); vi(ccol, F, [1, 1, 1, 1])
        crow = n("Div", [rs, K], pre + "row"); vi(crow, F, [1, 1, 1, 1])
        return ccol, crow

    ccol, crow = centroid("Ccp", "Crp", "C")
    rcol, rrow = centroid("Rcp", "Rrp", "R")

    # d = c-centroid - r-centroid; vertical iff |dcol| > |drow|
    n("Sub", [ccol, rcol], "dcol"); vi("dcol", F, [1, 1, 1, 1])
    n("Sub", [crow, rrow], "drow"); vi("drow", F, [1, 1, 1, 1])
    n("Abs", ["dcol"], "adcol"); vi("adcol", F, [1, 1, 1, 1])
    n("Abs", ["drow"], "adrow"); vi("adrow", F, [1, 1, 1, 1])
    n("Greater", ["adcol", "adrow"], "vertb"); vi("vertb", B, [1, 1, 1, 1])
    n("Not", ["vertb"], "horb"); vi("horb", B, [1, 1, 1, 1])

    # active centroid + active diff
    n("Where", ["vertb", rcol, rrow], "ract"); vi("ract", F, [1, 1, 1, 1])
    n("Where", ["vertb", "dcol", "drow"], "dact"); vi("dact", F, [1, 1, 1, 1])
    init("two", np.array(2.0, np.float32))
    n("Mul", ["ract", "two"], "r2"); vi("r2", F, [1, 1, 1, 1])
    n("Sign", ["dact"], "sgn"); vi("sgn", F, [1, 1, 1, 1])
    n("Add", ["r2", "sgn"], "s"); vi("s", F, [1, 1, 1, 1])

    # ---- reflection index vectors -> Gather --------------------------------
    init("nhalf", np.array(-0.5, np.float32))
    init("c95", np.array(9.5, np.float32))   # off-grid output position
    init("c9", np.array(9.0, np.float32))     # always-zero gather index (col/row 9 is bg)
    init("shp10", np.array([W], np.int64), np.int64)

    def safe_index(basevec, gate, outname, shapeC, axis):
        # target index = s - basevec ; redirect to col/row 9 (guaranteed bg)
        # when target leaves grid, output pos is off the 10-grid, or this axis
        # is the inactive orientation.
        n("Sub", ["s", basevec], outname + "g"); vi(outname + "g", F, shapeC)
        # target leaves grid: < -0.5  OR  > 9.5
        n("Less", [outname + "g", "nhalf"], outname + "neg"); vi(outname + "neg", B, shapeC)
        n("Greater", [outname + "g", "c95"], outname + "hi"); vi(outname + "hi", B, shapeC)
        # output position off the 10-grid (basevec >= 10): basevec > 9.5
        n("Greater", [basevec, "c95"], outname + "off"); vi(outname + "off", B, shapeC)
        n("Or", [outname + "neg", outname + "hi"], outname + "b0"); vi(outname + "b0", B, shapeC)
        n("Or", [outname + "b0", outname + "off"], outname + "b1"); vi(outname + "b1", B, shapeC)
        n("Or", [outname + "b1", gate], outname + "bad"); vi(outname + "bad", B, shapeC)
        n("Where", [outname + "bad", "c9", outname + "g"], outname + "ss"); vi(outname + "ss", F, shapeC)
        n("Cast", [outname + "ss"], outname + "i", to=I32); vi(outname + "i", I32, shapeC)
        n("Reshape", [outname + "i", "shp10"], outname + "f"); vi(outname + "f", I32, [W])
        n("Gather", ["Cf", outname + "f"], outname, axis=axis); vi(outname, F16, [1, 1, 10, 10])

    # reuse the centroid arange kernels colk[1,1,1,10] / rowk[1,1,10,1]
    safe_index("colk", "horb", "reflV", [1, 1, 1, W], 3)
    safe_index("rowk", "vertb", "reflH", [1, 1, W, 1], 2)

    # M = (Cf + reflV + reflH) > 0.5   (only the active reflection is nonzero)
    init("half16", np.array(0.5, np.float16), np.float16)
    n("Sum", ["Cf", "reflV", "reflH"], "Msum"); vi("Msum", F16, [1, 1, 10, 10])
    n("Greater", ["Msum", "half16"], "M10"); vi("M10", B, [1, 1, 10, 10])

    # ---- colour one-hot vector ([1,10,1,1] fp16) ---------------------------
    n("ReduceMax", ["input"], "chmax", axes=[2, 3], keepdims=1); vi("chmax", F, [1, 10, 1, 1])
    chmask = np.ones((1, 10, 1, 1), np.float32)
    chmask[0, 0, 0, 0] = chmask[0, 2, 0, 0] = chmask[0, 3, 0, 0] = 0.0
    init("chmask", chmask)
    n("Mul", ["chmax", "chmask"], "colvecf"); vi("colvecf", F, [1, 10, 1, 1])
    n("Cast", ["colvecf"], "colvec", to=F16); vi("colvec", F16, [1, 10, 1, 1])

    # ---- assemble the 10x10 coloured grid, then Pad -> FREE 30x30 output ----
    # green one-hot vector ch3=1 ([1,10,1,1]); colour where M else green.  The
    # whole [1,10,30,30] expansion is the FREE `output`: Pad(incolor10) places
    # the 10x10 in the top-left and zero-fills off-grid (matching the harness,
    # which leaves off-grid cells all-zero).  fp16 halves incolor10 (the colour
    # one-hot is {0,1}, exact in fp16); the harness reads the output as >0.
    green10 = np.zeros((1, 10, 1, 1), np.float16); green10[0, 3, 0, 0] = 1.0
    init("green10", green10, np.float16)
    n("Where", ["M10", "colvec", "green10"], "incolor10"); vi("incolor10", F16, [1, 10, 10, 10])
    n("Pad", ["incolor10"], "output", mode="constant",
      pads=[0, 0, 0, 0, 0, 0, 20, 20], value=0.0)

    model = _model(nodes, inits, vinfos)
    model.graph.output[0].type.tensor_type.elem_type = TensorProto.FLOAT16
    return model
