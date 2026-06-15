"""Task 228 (ARC-AGI 952a094c) — box with interior corner pixels ejected.

Rule (from the generator):
  A solid rectangle of `bcolor` (outline ring; interior cleared to black) sits in
  the grid.  Inside, at the 4 interior corners of the ring, are 4 distinct colours:
    colors[0] at (r0+1, c0+1)  (interior top-left)
    colors[1] at (r0+1, c1-1)  (interior top-right)
    colors[2] at (r1-1, c0+1)  (interior bottom-left)
    colors[3] at (r1-1, c1-1)  (interior bottom-right)
  box outer bounds are rows [r0, r1], cols [c0, c1].

  Output: the same ring (interior all black), and the 4 colours EJECTED to the 4
  *outer* diagonal corners, each going to the diagonally OPPOSITE outer corner:
    (r0-1, c0-1) = colors[3]   (interior BR -> outer TL)
    (r0-1, c1+1) = colors[2]   (interior BL -> outer TR)
    (r1+1, c0-1) = colors[1]   (interior TR -> outer BL)
    (r1+1, c1+1) = colors[0]   (interior TL -> outer BR)
  i.e. the colour in interior quadrant Q moves to the outer corner of the OPPOSITE
  quadrant.

Memory floor-break (label map + final Equal):
  The grid is always 10x10, so all work is done on a WORK=10 canvas (the whole
  grid, no off-grid cells inside it) and Pad fills the rest of the 30x30 with the
  off-grid sentinel 10.  We detect bcolor (most frequent non-zero channel), the
  box bounds r0/r1/c0/c1, and the 4 interior corner colours (max of the colour
  plane in each box quadrant), then build a single uint8 label map L and emit
  output = Equal(L, arange[1,10,1,1]) (opset 11, BOOL).  No [1,10,30,30]
  intermediate is materialised; all values are small ints (exact fp16/uint8).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
B = TensorProto.BOOL

WORK = 10  # grid is always 10x10; work on exactly the grid, pad rest to sentinel


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- colour plane G (0 background AND off-grid), sliced to WORK in fp16 -
    init("Wg", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    n("Conv", ["input", "Wg"], "Gf")                  # [1,1,30,30] f32 colour
    init("st", np.array([0, 0, 0], np.int64), np.int64)
    init("en", np.array([1, WORK, WORK], np.int64), np.int64)
    init("ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["Gf", "st", "en", "ax"], "Gs")        # [1,1,WORK,WORK] f32
    n("Cast", ["Gs"], "G", to=F16)                    # fp16 (288B)
    init("Half", np.array(0.5, np.float16), np.float16)

    # ---- bcolor = argmax over channels 1..9 of cell counts -----------------
    n("ReduceSum", ["input"], "counts", axes=[2, 3], keepdims=1)  # [1,10,1,1]
    init("mask01", np.array([0] + [1] * 9, np.float32).reshape(1, 10, 1, 1), np.float32)
    n("Mul", ["counts", "mask01"], "counts1")          # zero ch0
    n("ArgMax", ["counts1"], "bidx", axis=1, keepdims=1)  # [1,1,1,1] int64
    n("Cast", ["bidx"], "bcolor_f", to=F16)            # scalar colour value
    n("Cast", ["bidx"], "bcolor_u", to=U8)             # uint8 scalar

    # ---- box mask = (G == bcolor) (interior is black -> not in box) --------
    # uint8 colour plane (100B) gives cheap exact Equal/comparisons.
    n("Cast", ["G"], "Gu", to=U8)                      # [1,1,WORK,WORK] uint8
    init("v0u", np.array(0, np.uint8), np.uint8)
    n("Equal", ["Gu", "bcolor_u"], "box_b")            # ring
    n("Equal", ["Gu", "v0u"], "bg_b"); n("Not", ["bg_b"], "nonbg_b")  # coloured
    n("Cast", ["box_b"], "box_f", to=F16)

    # ---- box bounds r0,r1,c0,c1 (scalars) ----------------------------------
    n("ReduceMax", ["box_f"], "rowocc", axes=[3], keepdims=1)  # [1,1,WORK,1]
    n("ReduceMax", ["box_f"], "colocc", axes=[2], keepdims=1)  # [1,1,1,WORK]
    init("Icol", np.arange(WORK, dtype=np.float16).reshape(1, 1, WORK, 1), np.float16)
    init("IcolH", (WORK - np.arange(WORK)).astype(np.float16).reshape(1, 1, WORK, 1), np.float16)
    init("Irow", np.arange(WORK, dtype=np.float16).reshape(1, 1, 1, WORK), np.float16)
    init("IrowH", (WORK - np.arange(WORK)).astype(np.float16).reshape(1, 1, 1, WORK), np.float16)
    init("CW", np.array(float(WORK), np.float16), np.float16)
    n("Mul", ["rowocc", "IcolH"], "r0a"); n("ReduceMax", ["r0a"], "r0m", keepdims=1)
    n("Sub", ["CW", "r0m"], "r0")                      # [1,1,1,1]
    n("Mul", ["rowocc", "Icol"], "r1a"); n("ReduceMax", ["r1a"], "r1", keepdims=1)
    n("Mul", ["colocc", "IrowH"], "c0a"); n("ReduceMax", ["c0a"], "c0m", keepdims=1)
    n("Sub", ["CW", "c0m"], "c0")
    n("Mul", ["colocc", "Irow"], "c1a"); n("ReduceMax", ["c1a"], "c1", keepdims=1)

    # ---- box centre (mid row / mid col) ------------------------------------
    init("Two", np.array(2.0, np.float16), np.float16)
    n("Add", ["r0", "r1"], "rs"); n("Div", ["rs", "Two"], "rmid")
    n("Add", ["c0", "c1"], "cs"); n("Div", ["cs", "Two"], "cmid")

    # ---- colour-pixel plane: non-background and non-bcolor (the 4 pixels) ---
    n("Not", ["box_b"], "notbc_b")                     # not bcolor
    n("And", ["nonbg_b", "notbc_b"], "pix_b")          # [1,1,WORK,WORK] the 4 pixels
    init("zeroG", np.array(0.0, np.float16), np.float16)
    n("Where", ["pix_b", "G", "zeroG"], "Gpix")        # colour where pixel else 0

    # ---- quadrant masks relative to box centre -----------------------------
    # Collapse to two 10x10 planes (top / bottom rows of Gpix), reduce over rows
    # to per-column [1,1,1,WORK] vectors, then split left/right -> 4 scalars.
    init("zero16", np.array(0.0, np.float16), np.float16)
    n("Less", ["Icol", "rmid"], "top_b")               # [1,1,WORK,1]
    n("Greater", ["Icol", "rmid"], "bot_b")
    n("Less", ["Irow", "cmid"], "lft_b")               # [1,1,1,WORK]
    n("Greater", ["Irow", "cmid"], "rgt_b")

    n("Where", ["top_b", "Gpix", "zero16"], "Gtop")    # [1,1,WORK,WORK]
    n("Where", ["bot_b", "Gpix", "zero16"], "Gbot")
    n("ReduceMax", ["Gtop"], "topcol", axes=[2], keepdims=1)  # [1,1,1,WORK]
    n("ReduceMax", ["Gbot"], "botcol", axes=[2], keepdims=1)

    def split(colvec, sidemask, name):
        n("Where", [sidemask, colvec, "zero16"], name + "_m")  # [1,1,1,WORK]
        n("ReduceMax", [name + "_m"], name + "_c", axes=[3], keepdims=1)  # scalar
        return name + "_c"

    cTL = split("topcol", "lft_b", "qtl")  # colors[0]
    cTR = split("topcol", "rgt_b", "qtr")  # colors[1]
    cBL = split("botcol", "lft_b", "qbl")  # colors[2]
    cBR = split("botcol", "rgt_b", "qbr")  # colors[3]

    # ---- output corner cells (single-cell masks) ---------------------------
    init("One", np.array(1.0, np.float16), np.float16)
    n("Sub", ["r0", "One"], "or0")     # outer-top row    = r0-1
    n("Add", ["r1", "One"], "or1")     # outer-bottom row = r1+1
    n("Sub", ["c0", "One"], "oc0")     # outer-left col   = c0-1
    n("Add", ["c1", "One"], "oc1")     # outer-right col  = c1+1

    def rowmask(val, name):
        n("Sub", ["Icol", val], name + "_d"); n("Abs", [name + "_d"], name + "_a")
        n("Greater", ["Half", name + "_a"], name + "_b")
        return name + "_b"

    def colmask(val, name):
        n("Sub", ["Irow", val], name + "_d"); n("Abs", [name + "_d"], name + "_a")
        n("Greater", ["Half", name + "_a"], name + "_b")
        return name + "_b"

    rTopB = rowmask("or0", "rtop")
    rBotB = rowmask("or1", "rbot")
    cLftB = colmask("oc0", "clft")
    cRgtB = colmask("oc1", "crgt")

    n("And", [rTopB, cLftB], "oTL_b")  # outer top-left  -> cBR
    n("And", [rTopB, cRgtB], "oTR_b")  # outer top-right -> cBL
    n("And", [rBotB, cLftB], "oBL_b")  # outer bot-left  -> cTR
    n("And", [rBotB, cRgtB], "oBR_b")  # outer bot-right -> cTL

    # ---- build uint8 label map L (in-grid background everywhere = 0) --------
    n("Cast", [cTL], "uTL", to=U8)
    n("Cast", [cTR], "uTR", to=U8)
    n("Cast", [cBL], "uBL", to=U8)
    n("Cast", [cBR], "uBR", to=U8)
    init("v0arr", np.zeros((1, 1, WORK, WORK), np.uint8), np.uint8)

    n("Where", ["box_b", "bcolor_u", "v0arr"], "L1")   # ring=bcolor else 0
    n("Where", ["oTL_b", "uBR", "L1"], "L2")
    n("Where", ["oTR_b", "uBL", "L2"], "L3")
    n("Where", ["oBL_b", "uTR", "L3"], "L4")
    n("Where", ["oBR_b", "uTL", "L4"], "L5")           # [1,1,WORK,WORK] uint8

    # ---- pad to 30x30 with sentinel 10, then final Equal -------------------
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    n("Pad", ["L5", "padpads", "padval"], "L", mode="constant")  # [1,1,30,30] uint8
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task228", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
