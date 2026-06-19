"""task387 (ARC-AGI f35d900a) — 4-corner box -> framed/grayscaled box.

Rule (from generator task_f35d900a.py):
  INPUT: a width x height grid (14..18 each) with EXACTLY 4 coloured pixels at the
  corners of an axis-aligned rectangle of size (wide x tall), 5..11 each:
      (row,col)=colors[0]  (row,col+wide)=colors[1]
      (row+tall,col)=colors[1]  (row+tall,col+wide)=colors[0]
  (two colours, never gray, on the two diagonals.)
  OUTPUT, deterministically from those 4 pixels:
    - at each corner a 3x3 block filled with the OTHER colour, centre cell = own colour;
    - gray (5) marks along the 4 frame edges at EVEN distances from the corners:
        top/bottom rows  (r==row or r==row+tall): cols col+dc & col+wide-dc, dc in 2,4..wide//2
        left/right cols  (c==col or c==col+wide): rows row+dr & row+tall-dr, dr in 2,4..tall//2
  Off-grid (>= width / >= height) stays all-zero.

  Per-cell function of scalars row,col,rowt(=row+tall),colw(=col+wide),c0,c1 and (R,C):
    block cell  = (|R-row|<=1 or |R-rowt|<=1) and (|C-col|<=1 or |C-colw|<=1)
    centre cell = R in {row,rowt} and C in {col,colw}, value = own colour
    grayV (c==col or c==colw): (dr even & 2<=dr<=tall//2) or (drt even & 2<=drt<=tall//2)
    grayH (r==row or r==rowt): (dc even & 2<=dc<=wide//2) or (dcw even & 2<=dcw<=wide//2)
    fill colours: TL->c1 ; TR->c0 ; BL->c0 ; BR->c1   (own colour is the OPPOSITE)
  in-grid extent = rows [0,height) x cols [0,width).

Encoding (RE-GOLF of the adopted 18176/15.19 net).  The old net carried the forced
fp32 30x30 colour-index Conv plane (3600B) AND its 18x18 fp32 crop (1296B); the whole
output is determined by the SIX scalars above + per-row/col in-grid extent, none of which
need a 2-D colour plane.  So the colour Conv is replaced by FOUR no-pad PROFILE Convs
(per-row / per-col pixel-count and in-grid-count, 120B each) — the ROW/COL-SUM-AS-CONV
lever — plus tiny Gathers of the input one-hot for c0/c1.  EVERY downstream working plane
is a tiny [1,1,18,1]/[1,1,1,18] vector or an 18x18 fp16/bool index plane; no full 30x30
plane survives except the forced uint8 output Pad.

  - rowpix/colpix (pixel rows/cols), rowin/colin (in-grid extent) via 1x10x1x30 /
    1x10x30x1 Convs (ch0 weight 0 for pixel, 1 for in-grid).
  - row,col,rowt,colw from masked-ramp ReduceMin/Max over rowpix/colpix.
  - c0 = colour at (row,col) = sum_k k*input[:, :, row, col]; c1 at (row,colw); both via
    two chained Gathers of the FREE input one-hot + an arange dot (tiny).
  - ingrid mask = rowin>0 (row) AND colin>0 (col), broadcast inside the L0 Where.
  - build the WORK=18 colour-index plane with a 4-Where priority chain (fp16), Cast to
    uint8, Pad WORK->30 with sentinel 50, Equal(Lpad, arange-uint8) -> FREE bool output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64


def build(task):
    WORK = 18  # width,height in [14,18]; frame always fits within [0,WORK)
    inits, nodes = [], []

    def init(name, arr, dtype=F16):
        np_dt = {F32: np.float32, F16: np.float16, I64: np.int64, BOOL: bool}[dtype]
        inits.append(numpy_helper.from_array(np.asarray(arr).astype(np_dt), name=name))
        return name

    ctr = [0]

    def nn(op, ins, **attrs):
        ctr[0] += 1
        out = f"{op}_{ctr[0]}"
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- profile-Conv weights (fp32) ----
    # per-ROW pixel count: collapse the 30-wide axis, ch0 (bg) weight 0 -> only pixels
    init("Wrpix", np.concatenate([np.zeros((1, 1, 1, 30)),
         np.ones((1, 9, 1, 30))], axis=1), F32)            # -> [1,1,30,1]
    init("Wcpix", np.concatenate([np.zeros((1, 1, 30, 1)),
         np.ones((1, 9, 30, 1))], axis=1), F32)            # -> [1,1,1,30]

    # ---- constants ----
    Rramp = init("Rramp", np.arange(WORK).reshape(1, 1, WORK, 1), F16)
    Cramp = init("Cramp", np.arange(WORK).reshape(1, 1, 1, WORK), F16)
    init("zero", [[[[0.0]]]], F16)
    init("one", [[[[1.0]]]], F16)
    init("two", [[[[2.0]]]], F16)
    init("BIG", [[[[999.0]]]], F16)
    init("NEG", [[[[-1.0]]]], F16)
    init("gray", [[[[5.0]]]], F16)
    init("OFFG", [[[[50.0]]]], F16)
    init("half", [[[[0.5]]]], F16)
    init("zero32", [[[[0.0]]]], F32)
    init("kdot", np.arange(10).reshape(1, 10, 1, 1), F32)   # colour-index dot weights
    init("shp1", np.array([1], dtype=np.int64), I64)
    init("s0", np.array([0], dtype=np.int64), I64)
    init("sW", np.array([WORK], dtype=np.int64), I64)
    init("ax2", np.array([2], dtype=np.int64), I64)
    init("ax3", np.array([3], dtype=np.int64), I64)
    init("pad_amt", np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], dtype=np.int64), I64)
    inits.append(numpy_helper.from_array(np.array(50, dtype=np.uint8), name="OFFGu"))
    inits.append(numpy_helper.from_array(
        np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), name="arange10u"))

    # ---- profiles (tiny: 120B fp32 each, then crop to WORK and fp16) ----
    rowpix = nn("Conv", ["input", "Wrpix"])   # [1,1,30,1] fp32: #pixels in row
    colpix = nn("Conv", ["input", "Wcpix"])   # [1,1,1,30]
    # in-grid count = sum over ALL channels+the other axis (free reduction, no params)
    rowin = nn("ReduceSum", ["input"], axes=[1, 3], keepdims=1)   # [1,1,30,1] fp32
    colin = nn("ReduceSum", ["input"], axes=[1, 2], keepdims=1)   # [1,1,1,30] fp32

    # in-grid extent masks (bool [1,1,30,1] / [1,1,1,30]); only first WORK matter (Rramp<WORK)
    rowInG = nn("Greater", [rowin, "zero32"])   # bool [1,1,30,1]
    colInG = nn("Greater", [colin, "zero32"])   # bool [1,1,1,30]

    # pixel-row / pixel-col masks (cast to fp16 for the masked-ramp recovery)
    rpx = nn("Greater", [rowpix, "zero32"])   # bool [1,1,30,1]
    cpx = nn("Greater", [colpix, "zero32"])   # bool [1,1,1,30]

    # ---- recover row,col,rowt,colw via masked-ramp ReduceMin/Max (fp16) ----
    # Rramp is [1,1,WORK,1]; rpx is [1,1,30,1] -> slice not needed, broadcast on the
    # first WORK rows (rows>=WORK are off-grid, rpx False there since no pixels).
    rpxW = nn("Slice", [rpx, "s0", "sW", "ax2"])   # [1,1,WORK,1] bool
    cpxW = nn("Slice", [cpx, "s0", "sW", "ax3"])   # [1,1,1,WORK] bool

    rlo = nn("Where", [rpxW, Rramp, "BIG"])
    row = nn("ReduceMin", [rlo], axes=[2], keepdims=1)     # [1,1,1,1] fp16
    rhi = nn("Where", [rpxW, Rramp, "NEG"])
    rowt = nn("ReduceMax", [rhi], axes=[2], keepdims=1)
    clo = nn("Where", [cpxW, Cramp, "BIG"])
    col = nn("ReduceMin", [clo], axes=[3], keepdims=1)
    chi = nn("Where", [cpxW, Cramp, "NEG"])
    colw = nn("ReduceMax", [chi], axes=[3], keepdims=1)

    isCol = nn("Equal", [Cramp, col])
    isColw = nn("Equal", [Cramp, colw])
    isRow = nn("Equal", [Rramp, row])
    isRowt = nn("Equal", [Rramp, rowt])

    # ---- c0 = colour at (row,col), c1 at (row,colw): tiny Gathers of the FREE input ----
    rowi = nn("Reshape", [nn("Cast", [row], to=I64), "shp1"])    # scalar
    coli = nn("Reshape", [nn("Cast", [col], to=I64), "shp1"])
    colwi = nn("Reshape", [nn("Cast", [colw], to=I64), "shp1"])
    grow = nn("Gather", ["input", rowi], axis=2)    # [1,10,1,30] fp32 (tiny)
    g0 = nn("Gather", [grow, coli], axis=3)         # [1,10,1,1]
    g1 = nn("Gather", [grow, colwi], axis=3)        # [1,10,1,1]
    # colour index = sum_k k * onehot_k  (Conv with kdot weights -> [1,1,1,1])
    c0f = nn("Conv", [g0, "kdot"])                  # [1,1,1,1] fp32
    c1f = nn("Conv", [g1, "kdot"])
    c0 = nn("Cast", [c0f], to=F16)                  # [1,1,1,1] fp16 scalar
    c1 = nn("Cast", [c1f], to=F16)

    # ---- geometric ramps (fp16) ----
    dr = nn("Sub", [Rramp, row])     # [1,1,WORK,1]
    drt = nn("Sub", [rowt, Rramp])
    dc = nn("Sub", [Cramp, col])     # [1,1,1,WORK]
    dcw = nn("Sub", [colw, Cramp])
    tall = nn("Sub", [rowt, row])    # [1,1,1,1]
    wide = nn("Sub", [colw, col])

    def absn(x):
        return nn("Abs", [x])

    atTop = nn("Not", [nn("Greater", [absn(dr), "one"])])   # [1,1,WORK,1]
    atBot = nn("Not", [nn("Greater", [absn(drt), "one"])])
    atLft = nn("Not", [nn("Greater", [absn(dc), "one"])])   # [1,1,1,WORK]
    atRgt = nn("Not", [nn("Greater", [absn(dcw), "one"])])
    nearR = nn("Or", [atTop, atBot])      # [1,1,WORK,1]
    nearC = nn("Or", [atLft, atRgt])      # [1,1,1,WORK]
    block = nn("And", [nearR, nearC])     # [1,1,WORK,WORK] bool

    # per-cell block fill colour via nested Where over the tiny row/col vectors
    fillTop = nn("Where", [atLft, c1, c0])  # top row: TL(left)->c1, TR(right)->c0
    fillBot = nn("Where", [atLft, c0, c1])  # bottom row: BL(left)->c0, BR(right)->c1
    fillCol = nn("Where", [atTop, fillTop, fillBot])  # [1,1,WORK,WORK] fp16
    # centre own colour = OPPOSITE of fill
    ownTop = nn("Where", [atLft, c0, c1])
    ownBot = nn("Where", [atLft, c1, c0])
    ownCol = nn("Where", [atTop, ownTop, ownBot])     # [1,1,WORK,WORK] fp16

    # ---- gray edges (fp16) ----
    def even(x):
        ctr[0] += 1
        mo = f"Mod_{ctr[0]}"
        nodes.append(helper.make_node("Mod", [x, "two"], [mo], fmod=1))
        return nn("Equal", [mo, "zero"])
    th = nn("Floor", [nn("Mul", [tall, "half"])])   # tall//2
    tw = nn("Floor", [nn("Mul", [wide, "half"])])

    def inrange(x, hi):  # 2<=x<=hi
        ge2 = nn("Not", [nn("Less", [x, "two"])])
        le = nn("Not", [nn("Greater", [x, hi])])
        return nn("And", [ge2, le])

    gV_a = nn("And", [even(dr), inrange(dr, th)])
    gV_b = nn("And", [even(drt), inrange(drt, th)])
    gV_dr = nn("Or", [gV_a, gV_b])             # [1,1,WORK,1]
    onEdgeC = nn("Or", [isCol, isColw])        # [1,1,1,WORK]
    grayV = nn("And", [gV_dr, onEdgeC])        # [1,1,WORK,WORK]

    gH_a = nn("And", [even(dc), inrange(dc, tw)])
    gH_b = nn("And", [even(dcw), inrange(dcw, tw)])
    gH_dc = nn("Or", [gH_a, gH_b])             # [1,1,1,WORK]
    onEdgeR = nn("Or", [isRow, isRowt])        # [1,1,WORK,1]
    grayH = nn("And", [gH_dc, onEdgeR])
    gray = nn("Or", [grayV, grayH])

    # centre cells = corners = onEdgeR AND onEdgeC (reuse the gray edge vectors)
    centre = nn("And", [onEdgeR, onEdgeC])     # [1,1,WORK,WORK] bool

    # ---- assemble colour-index plane (priority low->high), fp16 ----
    # L0: in-grid bg = 0, off-grid = sentinel 50.  ingrid = rowInG AND colInG.
    ingridW = nn("And", [nn("Slice", [rowInG, "s0", "sW", "ax2"]),
                         nn("Slice", [colInG, "s0", "sW", "ax3"])])  # [1,1,WORK,WORK] bool
    L0 = nn("Where", [ingridW, "zero", "OFFG"])
    L1 = nn("Where", [gray, "gray", L0])
    L2 = nn("Where", [block, fillCol, L1])     # block fill (per-cell c0/c1)
    Lout = nn("Where", [centre, ownCol, L2])   # centres = own colour

    # ---- cast to uint8, pad WORK->30 with sentinel, route to FREE bool output ----
    Lu8 = nn("Cast", [Lout], to=TensorProto.UINT8)
    Lpad = nn("Pad", [Lu8, "pad_amt", "OFFGu"], mode="constant")
    nn("Equal", [Lpad, "arange10u"])  # [1,10,30,30] bool
    nodes[-1].output[0] = "output"

    graph = helper.make_graph(
        nodes, "task387", [helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
