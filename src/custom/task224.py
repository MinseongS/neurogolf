"""Task 224 (928ad970): reconstruct the outer box from 4 gray markers.

Rule (from ARC-GEN generator):
  Input is an H x W grid (H, W <= 16) containing (a) a small inner rectangle
  perimeter drawn in a single non-gray colour c, and (b) exactly 4 gray (5)
  marker pixels.  Each marker sits one cell OUTSIDE one edge of a larger outer
  rectangle:
    one gray at row = top-1, one at row = bottom+1,
    one at col = left-1, one at col = right+1.
  Equivalently, over the 4 markers:
    top    = min(gray_row) + 1     bottom = max(gray_row) - 1
    left   = min(gray_col) + 1     right  = max(gray_col) - 1
  Output = input, plus the PERIMETER of the outer rectangle
  [top..bottom] x [left..right] painted in colour c.  Gray markers and the
  inner box are preserved.

Output colour per cell:
    5  where the input is gray,
    c  where the input has colour c (inner box) OR the cell is on the outer
       rectangle perimeter,
    0  otherwise (in-grid background),
   (outside the H x W grid: no channel set).

Tier B (label map + final Equal), pushed hard:

* The box position is a GLOBAL aggregate of the gray markers (min/max gray row
  & col), so the rule is non-local -> no single Conv (Tier S out).
* ONE Conv with weight [0,1,...,9] turns the one-hot input into a colour-value
  plane cval (0 bg, 5 gray, c inner box).  Casting it to uint8 IS the label
  base for every input-derived cell, so no separate gray/inner equality tests.
* The whole active grid is <= 16 x 16 (generator bound), so every per-cell plane
  is sliced to a 16 x 16 working canvas (256 elems) and the final label is
  Padded back to 30 x 30 with an outside sentinel just before the Equal.
* The outer-rectangle perimeter is built separably from 1-D row/col index
  conditions (inR/inC/edgeR/edgeC, each <=64 bytes) and OR'd in.
* The 10-way colour expansion is routed into the FREE bool `output` via the
  final Equal(L, arange[0..9]).

Dominant intermediate: the fp32 colour-value Conv plane sliced to 16 x 16
(1024 B).  It is the one place the 10 input channels collapse to a per-cell
scalar; nothing smaller carries both the gray and the inner-box colour at their
exact 2-D positions.  All values are small integers, exact in fp32/uint8.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

WORK = 16  # working-canvas side (max grid dim from the generator)


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
    cw = np.arange(10, dtype=np.float32)
    cw[0] = 0.0
    cw[5] = 0.0
    init("cw", cw.reshape(1, 10, 1, 1), np.float32)             # colour-scalar weights
    init("colorw", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1),
         np.float32)                                            # 1x1 Conv: cell -> colour

    init("ar_row", np.arange(WORK, dtype=np.float32).reshape(1, 1, WORK, 1),
         np.float32)
    init("ar_col", np.arange(WORK, dtype=np.float32).reshape(1, 1, 1, WORK),
         np.float32)
    init("BIG", np.array(99.0, np.float32), np.float32)
    init("NEG", np.array(-1.0, np.float32), np.float32)
    init("one", np.array(1.0, np.float32), np.float32)
    init("half", np.array(0.5, np.float32), np.float32)

    init("arange10", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1),
         np.uint8)
    init("v10", np.array(10, np.uint8), np.uint8)               # outside-grid sentinel

    # slice helpers
    init("ch5_st", np.array([5], np.int64), np.int64)
    init("ch5_en", np.array([6], np.int64), np.int64)
    init("ch_ax", np.array([1], np.int64), np.int64)
    init("w_st", np.array([0, 0], np.int64), np.int64)
    init("w_en_rc", np.array([WORK, WORK], np.int64), np.int64)
    init("w_ax_rc", np.array([2, 3], np.int64), np.int64)
    init("w_en_r", np.array([WORK], np.int64), np.int64)        # rows only
    init("ax2", np.array([2], np.int64), np.int64)
    init("ax3", np.array([3], np.int64), np.int64)
    init("st0", np.array([0], np.int64), np.int64)
    # pad L (16x16) -> 30x30, fill = sentinel 10
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)

    # ---- colour scalar c (float) ----
    n("ReduceMax", ["input"], "pres", axes=[2, 3], keepdims=1)   # [1,10,1,1]
    n("Mul", ["pres", "cw"], "cparts")
    n("ReduceSum", ["cparts"], "cf", keepdims=1)                 # [1,1,1,1] = c
    n("Cast", ["cf"], "cu8_b", to=TensorProto.UINT8)
    n("Squeeze", ["cu8_b"], "cu8", axes=[0, 1, 2, 3])            # scalar uint8 c

    # ---- colour-value plane (one Conv), sliced to the 16x16 working canvas ----
    n("Conv", ["input", "colorw"], "cval30")                     # [1,1,30,30] fp32
    n("Slice", ["cval30", "w_st", "w_en_rc", "w_ax_rc"], "cval")  # [1,1,16,16]
    n("Cast", ["cval"], "Lcast", to=TensorProto.UINT8)           # [1,1,16,16] label base

    # ---- gray mask + 1-D occupancy (from cval, all on the 16x16 canvas) ----
    # gray == colour 5: 4.5 < cval < 5.5.
    init("v45", np.array(4.5, np.float32), np.float32)
    init("v55", np.array(5.5, np.float32), np.float32)
    n("Greater", ["cval", "v45"], "g_lo")
    n("Less", ["cval", "v55"], "g_hi")
    n("And", ["g_lo", "g_hi"], "is_gray")                        # [1,1,16,16] bool
    n("Cast", ["is_gray"], "is_gray_f", to=TensorProto.FLOAT)    # 16x16 = 1024B
    n("ReduceMax", ["is_gray_f"], "grow", axes=[3], keepdims=1)  # [1,1,16,1]
    n("ReduceMax", ["is_gray_f"], "gcol", axes=[2], keepdims=1)  # [1,1,1,16]
    n("Greater", ["grow", "half"], "growb")
    n("Greater", ["gcol", "half"], "gcolb")

    # ---- min/max gray row & col (scalars) ----
    n("Where", ["growb", "ar_row", "BIG"], "rmin_in")
    n("ReduceMin", ["rmin_in"], "minrow", keepdims=0)
    n("Where", ["growb", "ar_row", "NEG"], "rmax_in")
    n("ReduceMax", ["rmax_in"], "maxrow", keepdims=0)
    n("Where", ["gcolb", "ar_col", "BIG"], "cmin_in")
    n("ReduceMin", ["cmin_in"], "mincol", keepdims=0)
    n("Where", ["gcolb", "ar_col", "NEG"], "cmax_in")
    n("ReduceMax", ["cmax_in"], "maxcol", keepdims=0)

    # bounds: row>=top == row > minrow+0.5 ; row<=bot == row < maxrow-0.5
    n("Add", ["minrow", "half"], "top_thr")
    n("Sub", ["maxrow", "half"], "bot_thr")
    n("Add", ["mincol", "half"], "left_thr")
    n("Sub", ["maxcol", "half"], "right_thr")
    n("Add", ["minrow", "one"], "top")     # exact integer edges
    n("Sub", ["maxrow", "one"], "bot")
    n("Add", ["mincol", "one"], "left")
    n("Sub", ["maxcol", "one"], "right")

    # ---- separable perimeter conditions (1-D) ----
    n("Greater", ["ar_row", "top_thr"], "rge")
    n("Less", ["ar_row", "bot_thr"], "rle")
    n("And", ["rge", "rle"], "inR")                              # [1,1,16,1]
    n("Greater", ["ar_col", "left_thr"], "cge")
    n("Less", ["ar_col", "right_thr"], "cle")
    n("And", ["cge", "cle"], "inC")                              # [1,1,1,16]
    n("Equal", ["ar_row", "top"], "r_top")
    n("Equal", ["ar_row", "bot"], "r_bot")
    n("Or", ["r_top", "r_bot"], "edgeR")                         # [1,1,16,1]
    n("Equal", ["ar_col", "left"], "c_left")
    n("Equal", ["ar_col", "right"], "c_right")
    n("Or", ["c_left", "c_right"], "edgeC")                      # [1,1,1,16]

    # perim = (edgeR & inC) | (inR & edgeC)   -> [1,1,16,16]
    n("And", ["edgeR", "inC"], "horiz")                          # top/bottom edges
    n("And", ["inR", "edgeC"], "vert")                           # left/right edges
    n("Or", ["horiz", "vert"], "perim")                          # [1,1,16,16] bool

    # ---- label map L (uint8 16x16, padded to 30x30) ----
    n("Where", ["perim", "cu8", "Lcast"], "Lin")                 # paint perimeter c
    # in-grid mask from 1-D occupancy (grid is a solid HxW rect anchored at 0,0).
    # Reduce over channels AND one spatial axis directly -> [1,1,30,1] = 120B.
    n("ReduceMax", ["input"], "rowany30", axes=[1, 3], keepdims=1)  # [1,1,30,1]
    n("Slice", ["rowany30", "st0", "w_en_r", "ax2"], "rowany")   # [1,1,16,1]
    n("ReduceMax", ["input"], "colany30", axes=[1, 2], keepdims=1)  # [1,1,1,30]
    n("Slice", ["colany30", "st0", "w_en_r", "ax3"], "colany")   # [1,1,1,16]
    n("Greater", ["rowany", "half"], "rowanyb")
    n("Greater", ["colany", "half"], "colanyb")
    n("And", ["rowanyb", "colanyb"], "ingrid")                   # [1,1,16,16]
    n("Where", ["ingrid", "Lin", "v10"], "L16")                  # uint8 [1,1,16,16]
    n("Pad", ["L16", "padpads", "padval"], "L", mode="constant")  # [1,1,30,30] uint8

    # ---- final 10-way expansion into FREE bool output ----
    n("Equal", ["L", "arange10"], "output")                      # bool [1,10,30,30]

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
