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

  Verified exactly (0 mismatch / 400 fresh) as a per-cell function of the recovered
  scalars row,col,rowt(=row+tall),colw(=col+wide),c0,c1 and the cell (R,C):
    block cell  = (|R-row|<=1 or |R-rowt|<=1) and (|C-col|<=1 or |C-colw|<=1)
    centre cell = R in {row,rowt} and C in {col,colw}
    grayV (c==col or c==colw): (dr even & 2<=dr<=tall//2) or (drt even & 2<=drt<=tall//2)
    grayH (r==row or r==rowt): (dc even & 2<=dc<=wide//2) or (dcw even & 2<=dcw<=wide//2)
    fill colours: TL->c1 centre c0 ; TR->c0 centre c1 ; BL->c0 centre c1 ; BR->c1 centre c0
  in-grid mask = (sum over input channels) > 0   (off-grid cells have no channel set).

Encoding (mem 18176, params 85, 15.19 pts, fresh 500/500):
  - ONE 1x1 Conv with weight [0.5,1,2..9] makes a colf plane that doubles as the in-grid
    signal: off-grid=0, in-grid bg=0.5, pixel k=k (so a separate ReduceSum in-grid plane
    is not needed; thresholds 0.25/0.75 split off-grid / bg / pixel).
  - Crop colf to the WORK=18 active region (width,height<=18) so every downstream full
    plane is 18x18 (1296B) not 30x30 (3600B).
  - Recover row,col,rowt,colw by masked-ramp ReduceMin/Max; c0,c1 by slicing the top row
    out of colf (Gather axis2) then Gather axis3 at col / colw (scalars, no full plane).
  - Build the colour-index plane Lout with a 5-Where priority chain over row/col ramps
    (off-grid sentinel 50 -> gray 5 -> c0 blocks -> c1 blocks -> centres=own colf).  Block
    fill colour = c1 on TL,BR <=> Xor(atTop,atLft)=False, c0 on TR,BL otherwise.
  - Cast Lout to uint8, Pad WORK->30 with sentinel 50, Equal(Lpad, arange-uint8) -> the
    FREE bool [1,10,30,30] output (off-grid 50 matches no colour 0..9 -> all-zero).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64


def build(task):
    WORK = 18  # width,height in [14,18]; frame always fits within [0,WORK)
    inits, nodes = [], []

    def init(name, arr, dtype=F32):
        inits.append(numpy_helper.from_array(np.asarray(arr).astype(
            {F32: np.float32, I64: np.int64, BOOL: bool}[dtype]), name=name))
        return name

    ctr = [0]

    def nn(op, ins, **attrs):
        ctr[0] += 1
        out = f"{op}_{ctr[0]}"
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- constants ----
    # colour weights: ch0(bg)=0.5 so the colf plane ALSO encodes in-grid (bg->0.5,
    # off-grid->0, pixel k->k); avoids a separate ReduceSum(input) in-grid plane.
    init("Wcol", np.array([0.5] + list(range(1, 10)), dtype=np.float32).reshape(1, 10, 1, 1))
    Rramp = init("Rramp", np.arange(WORK, dtype=np.float32).reshape(1, 1, WORK, 1))
    Cramp = init("Cramp", np.arange(WORK, dtype=np.float32).reshape(1, 1, 1, WORK))
    init("zero", np.array([[[[0.0]]]]))
    init("one", np.array([[[[1.0]]]]))
    init("two", np.array([[[[2.0]]]]))
    init("BIG", np.array([[[[999.0]]]]))
    init("NEG", np.array([[[[-1.0]]]]))
    init("gray", np.array([[[[5.0]]]]))
    init("OFFG", np.array([[[[50.0]]]]))
    init("q25", np.array([[[[0.25]]]]))
    init("q75", np.array([[[[0.75]]]]))
    init("shp1", np.array([1], dtype=np.int64), I64)
    init("shp4", np.array([1, 1, 1, 1], dtype=np.int64), I64)
    init("sl_start", np.array([0, 0], dtype=np.int64), I64)
    init("sl_end", np.array([WORK, WORK], dtype=np.int64), I64)
    init("sl_axes", np.array([2, 3], dtype=np.int64), I64)
    # pad WORKxWORK colour-index plane back to 30x30 with the off-grid sentinel
    init("pad_amt", np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], dtype=np.int64), I64)
    inits.append(numpy_helper.from_array(np.array(50, dtype=np.uint8), name="OFFGu"))
    inits.append(numpy_helper.from_array(
        np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), name="arange10u"))

    # ---- colour-index plane colf [1,1,30,30], then crop to the WORKxWORK active region ----
    colf30 = nn("Conv", ["input", "Wcol"])             # [1,1,30,30]: bg=0.5, pix=k, offg=0
    colf = nn("Slice", [colf30, "sl_start", "sl_end", "sl_axes"])  # [1,1,WORK,WORK]
    ingrid = nn("Greater", [colf, "q25"])              # bool: in-grid (bg or pixel)

    # ---- recover scalars ----
    # row profile rowany[r] = max colour over c ; colany[c] = max over r (>0 => pixel)
    rowany = nn("ReduceMax", [colf], axes=[3], keepdims=1)  # [1,1,30,1]
    colany = nn("ReduceMax", [colf], axes=[2], keepdims=1)  # [1,1,1,30]
    rmask = nn("Greater", [rowany, "q75"])    # rows with a PIXEL (bg=0.5 excluded)
    cmask = nn("Greater", [colany, "q75"])
    # row = min r with pixel ; rowt = max
    rlo = nn("Where", [rmask, Rramp, "BIG"])
    row = nn("ReduceMin", [rlo], axes=[2], keepdims=1)     # [1,1,1,1]
    rhi = nn("Where", [rmask, Rramp, "NEG"])
    rowt = nn("ReduceMax", [rhi], axes=[2], keepdims=1)
    clo = nn("Where", [cmask, Cramp, "BIG"])
    col = nn("ReduceMin", [clo], axes=[3], keepdims=1)
    chi = nn("Where", [cmask, Cramp, "NEG"])
    colw = nn("ReduceMax", [chi], axes=[3], keepdims=1)

    isRow = nn("Equal", [Rramp, row])     # [1,1,30,1] bool
    isRowt = nn("Equal", [Rramp, rowt])   # [1,1,30,1]
    isCol = nn("Equal", [Cramp, col])
    isColw = nn("Equal", [Cramp, colw])

    # ---- recover c0 = colf at (row,col), c1 = colf at (row,colw) via Gather ----
    # slice the TOP row (index row) out of colf -> [1,1,30] then index col / colw.
    rowi = nn("Reshape", [nn("Cast", [row], to=I64), "shp1"])    # scalar
    coli = nn("Reshape", [nn("Cast", [col], to=I64), "shp1"])
    colwi = nn("Reshape", [nn("Cast", [colw], to=I64), "shp1"])
    toprow = nn("Gather", [colf, rowi], axis=2)    # [1,1,1,30]
    c0g = nn("Gather", [toprow, coli], axis=3)     # [1,1,1,1]
    c1g = nn("Gather", [toprow, colwi], axis=3)
    c0 = nn("Reshape", [c0g, "shp4"])              # [1,1,1,1]
    c1 = nn("Reshape", [c1g, "shp4"])

    # ---- geometric ramps ----
    dr = nn("Sub", [Rramp, row])     # [1,1,30,1]
    drt = nn("Sub", [rowt, Rramp])
    dc = nn("Sub", [Cramp, col])     # [1,1,1,30]
    dcw = nn("Sub", [colw, Cramp])
    tall = nn("Sub", [rowt, row])    # [1,1,1,1]
    wide = nn("Sub", [colw, col])

    def absnode(x):
        return nn("Abs", [x])

    # near a corner row/col (chebyshev <=1)
    adr = absnode(dr)
    adrt = absnode(drt)
    adc = absnode(dc)
    adcw = absnode(dcw)
    atTop = nn("Not", [nn("Greater", [adr, "one"])])   # [1,1,30,1]
    atBot = nn("Not", [nn("Greater", [adrt, "one"])])
    atLft = nn("Not", [nn("Greater", [adc, "one"])])   # [1,1,1,30]
    atRgt = nn("Not", [nn("Greater", [adcw, "one"])])
    # a block cell is near a corner row AND a corner col.  tall,wide >=5 so within a block
    # exactly one of atTop/atBot and one of atLft/atRgt holds.  Fill colour is c1 on the
    # TL & BR corners <=> (atTop == atLft) and c0 on TR & BL <=> (atTop != atLft).
    nearR = nn("Or", [atTop, atBot])      # [1,1,30,1]
    nearC = nn("Or", [atLft, atRgt])      # [1,1,1,30]
    block = nn("And", [nearR, nearC])     # [1,1,30,30]
    xorTL = nn("Xor", [atTop, atLft])     # [1,1,30,30] : True on c0-fill corners (TR,BL)
    blkC0 = nn("And", [block, xorTL])     # block cells filled with c0 (TR,BL blocks)
    blkC1 = nn("And", [block, nn("Not", [xorTL])])  # block cells filled with c1 (TL,BR blocks)
    occ = nn("Greater", [colf, "q75"])    # the 4 input pixels = the centres (bg 0.5 excluded)

    # ---- gray edges ----
    # parity: dr even -> Mod(dr,2)==0
    def even(x):
        ctr[0] += 1
        mo = f"Mod_{ctr[0]}"
        nodes.append(helper.make_node("Mod", [x, "two"], [mo], fmod=1))
        return nn("Equal", [mo, "zero"])
    th = nn("Div", [tall, "two"])  # tall//2 ; tall is non-negative integer-valued float
    th = nn("Floor", [th])
    tw = nn("Div", [wide, "two"])
    tw = nn("Floor", [tw])

    def inrange(x, hi):  # 2<=x<=hi
        ge2 = nn("Not", [nn("Less", [x, "two"])])
        le = nn("Not", [nn("Greater", [x, hi])])
        return nn("And", [ge2, le])

    # vertical gray: condition on dr / drt, applied where c==col or c==colw
    gV_a = nn("And", [even(dr), inrange(dr, th)])
    gV_b = nn("And", [even(drt), inrange(drt, th)])
    gV_dr = nn("Or", [gV_a, gV_b])             # [1,1,30,1]
    onEdgeC = nn("Or", [isCol, isColw])        # [1,1,1,30]
    grayV = nn("And", [gV_dr, onEdgeC])        # [1,1,30,30]

    gH_a = nn("And", [even(dc), inrange(dc, tw)])
    gH_b = nn("And", [even(dcw), inrange(dcw, tw)])
    gH_dc = nn("Or", [gH_a, gH_b])             # [1,1,1,30]
    onEdgeR = nn("Or", [isRow, isRowt])        # [1,1,30,1]
    grayH = nn("And", [gH_dc, onEdgeR])
    gray = nn("Or", [grayV, grayH])

    # ---- assemble colour-index plane Lout (priority low->high) ----
    # start: in-grid bg = 0, off-grid = sentinel 50 (matches no colour 0..9 -> all-zero)
    L0 = nn("Where", [ingrid, "zero", "OFFG"])
    L1 = nn("Where", [gray, "gray", L0])
    L2 = nn("Where", [blkC0, c0, L1])
    L3 = nn("Where", [blkC1, c1, L2])
    Lout = nn("Where", [occ, colf, L3])   # [1,1,WORK,WORK] centres = own input colour

    # ---- cast to uint8, pad WORKxWORK -> 30x30 with sentinel, route to FREE bool output
    Lu8 = nn("Cast", [Lout], to=TensorProto.UINT8)         # [1,1,WORK,WORK] uint8
    Lpad = nn("Pad", [Lu8, "pad_amt", "OFFGu"], mode="constant")  # [1,1,30,30] uint8
    nn("Equal", [Lpad, "arange10u"])  # [1,10,30,30] bool (uint8 Equal OK under ORT)
    nodes[-1].output[0] = "output"

    graph = helper.make_graph(
        nodes, "task387", [helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
