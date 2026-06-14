"""Task 288 (ARC-AGI b8cdaf2b): complete the "robot" figure by drawing the two
antenna diagonals.

Rule (verified exact on all 267 stored examples, derived from the ARC-GEN
generator). The s x s grid (s = neck + 2*shoulder, top-left on the canvas) holds
a figure whose bottom two rows are given:

  * bottom row  (s-1): shoulder shirt cells on each corner + neck antenna cells
                       in the middle
  * row  (s-2)       : neck shirt cells in the middle

The output ADDS two antenna-colored diagonals climbing up-and-outward from each
shoulder.  In closed form the added cells are:

  * left  diagonal: row - col == s-2-shoulder ,  0 <= col < shoulder
  * right diagonal: row + col == 2s-3-shoulder,  s-shoulder <= col < s

Both s and shoulder are read off the column-occupancy of the figure:
  * every column 0..s-1 is occupied (bottom row fills them) -> s = #occupied cols
  * the two shoulder corner columns each hold exactly one figure cell, the neck
    columns hold two -> #columns-with-exactly-one-cell == 2*shoulder.

The antenna color is the less-frequent of the two figure colors (shirt count =
neck+2*shoulder > neck = antenna count, always), recovered as the channel with
the minimal positive cell count.

Graph: compute the scalar s, shoulder, the two diagonal offsets, build the bool
diagonal mask from fixed row/col index planes, build the antenna one-hot color
vector A[1,10,1,1], then
  output = Where(mask[1,1,30,30], A[1,10,1,1], input)
which never materialises a 10-channel float canvas (the task166 single-Where
trick): on mask cells the cell becomes the antenna one-hot, elsewhere it is the
input untouched (mask cells are background in the input).
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

    # ---- occupancy plane occ[1,1,30,30] = sum of colour channels 1..9 (=1 on a
    #      figure cell, 0 on background / outside the grid) -------------------
    wocc = np.zeros((1, 10, 1, 1), np.float32)
    wocc[0, 1:, 0, 0] = 1.0
    init("Wocc", wocc, np.float32)
    n("Conv", ["input", "Wocc"], "occ")                       # [1,1,30,30] f32
    vi("occ", TensorProto.FLOAT, [1, 1, 30, 30])

    # ---- per-column figure count colcount[1,1,1,30] ------------------------
    n("ReduceSum", ["occ"], "colcnt", axes=[2], keepdims=1)   # [1,1,1,30] f32
    vi("colcnt", TensorProto.FLOAT, [1, 1, 1, 30])
    n("Cast", ["colcnt"], "colcnti", to=TensorProto.INT32)
    vi("colcnti", TensorProto.INT32, [1, 1, 1, 30])

    # s = number of occupied columns (colcount > 0)
    init("zero_i", np.array(0, np.int32), np.int32)
    n("Greater", ["colcnti", "zero_i"], "occcol")             # [1,1,1,30] bool
    vi("occcol", TensorProto.BOOL, [1, 1, 1, 30])
    n("Cast", ["occcol"], "occcoli", to=TensorProto.INT32)
    vi("occcoli", TensorProto.INT32, [1, 1, 1, 30])
    n("ReduceSum", ["occcoli"], "S", axes=[3], keepdims=1)    # [1,1,1,1] i32
    vi("S", TensorProto.INT32, [1, 1, 1, 1])

    # 2*shoulder = number of columns with exactly one figure cell
    init("one_i", np.array(1, np.int32), np.int32)
    n("Equal", ["colcnti", "one_i"], "iscorner")              # [1,1,1,30] bool
    vi("iscorner", TensorProto.BOOL, [1, 1, 1, 30])
    n("Cast", ["iscorner"], "iscorneri", to=TensorProto.INT32)
    vi("iscorneri", TensorProto.INT32, [1, 1, 1, 30])
    n("ReduceSum", ["iscorneri"], "twoSH", axes=[3], keepdims=1)  # [1,1,1,1]
    vi("twoSH", TensorProto.INT32, [1, 1, 1, 1])
    init("two_i", np.array(2, np.int32), np.int32)
    n("Div", ["twoSH", "two_i"], "SH")                        # [1,1,1,1] i32
    vi("SH", TensorProto.INT32, [1, 1, 1, 1])

    # ---- diagonal offsets ---------------------------------------------------
    # K_left  = S - 2 - SH ;  K_right = 2S - 3 - SH
    init("c2", np.array(2, np.int32), np.int32)
    init("c3", np.array(3, np.int32), np.int32)
    n("Sub", ["S", "SH"], "SmSH")                             # S - SH
    vi("SmSH", TensorProto.INT32, [1, 1, 1, 1])
    n("Sub", ["SmSH", "c2"], "Kleft")                         # S - SH - 2
    vi("Kleft", TensorProto.INT32, [1, 1, 1, 1])
    n("Add", ["S", "SmSH"], "twoSmSH")                        # 2S - SH
    vi("twoSmSH", TensorProto.INT32, [1, 1, 1, 1])
    n("Sub", ["twoSmSH", "c3"], "Kright")                     # 2S - SH - 3
    vi("Kright", TensorProto.INT32, [1, 1, 1, 1])
    # S - SH  (lower bound for right diagonal columns)
    # already have SmSH

    # ---- fixed row/col index planes ----------------------------------------
    rr = np.arange(30, dtype=np.int32).reshape(1, 1, 30, 1) * np.ones(
        (1, 1, 1, 30), np.int32)
    cc = np.ones((1, 1, 30, 1), np.int32) * np.arange(
        30, dtype=np.int32).reshape(1, 1, 1, 30)
    init("RmC", (rr - cc).astype(np.int32), np.int32)         # row - col
    init("RpC", (rr + cc).astype(np.int32), np.int32)         # row + col
    init("COL", cc.astype(np.int32), np.int32)                # col index

    # ---- left diagonal mask: (RmC == Kleft) & (COL < SH) -------------------
    n("Equal", ["RmC", "Kleft"], "onLleft")                   # [1,1,30,30] bool
    vi("onLleft", TensorProto.BOOL, [1, 1, 30, 30])
    n("Less", ["COL", "SH"], "ltSH")                          # COL < SH
    vi("ltSH", TensorProto.BOOL, [1, 1, 30, 30])
    n("And", ["onLleft", "ltSH"], "Lmask")
    vi("Lmask", TensorProto.BOOL, [1, 1, 30, 30])

    # ---- right diagonal mask: (RpC == Kright) & (COL >= S-SH) & (COL < S) ---
    n("Equal", ["RpC", "Kright"], "onLright")                 # [1,1,30,30] bool
    vi("onLright", TensorProto.BOOL, [1, 1, 30, 30])
    # COL >= S-SH  <=>  (S-SH-1) < COL
    init("one_i2", np.array(1, np.int32), np.int32)
    n("Sub", ["SmSH", "one_i2"], "SmSHm1")                    # S-SH-1
    vi("SmSHm1", TensorProto.INT32, [1, 1, 1, 1])
    n("Less", ["SmSHm1", "COL"], "geSmSH")                    # COL > S-SH-1
    vi("geSmSH", TensorProto.BOOL, [1, 1, 30, 30])
    n("Less", ["COL", "S"], "ltS")                            # COL < S
    vi("ltS", TensorProto.BOOL, [1, 1, 30, 30])
    n("And", ["onLright", "geSmSH"], "Rmask0")
    vi("Rmask0", TensorProto.BOOL, [1, 1, 30, 30])
    n("And", ["Rmask0", "ltS"], "Rmask")
    vi("Rmask", TensorProto.BOOL, [1, 1, 30, 30])

    n("Or", ["Lmask", "Rmask"], "mask")                       # [1,1,30,30] bool
    vi("mask", TensorProto.BOOL, [1, 1, 30, 30])

    # ---- antenna one-hot colour A[1,10,1,1] --------------------------------
    # per-channel total count (integer-valued) -> int32
    n("ReduceSum", ["input"], "chcntf", axes=[2, 3], keepdims=1)  # [1,10,1,1]
    vi("chcntf", TensorProto.FLOAT, [1, 10, 1, 1])
    n("Cast", ["chcntf"], "chcnt", to=TensorProto.INT32)     # [1,10,1,1] i32
    vi("chcnt", TensorProto.INT32, [1, 10, 1, 1])
    # adj = chcnt + (chcnt == 0) * BIG  so empty channels never win the min
    n("Greater", ["chcnt", "zero_i"], "nonempty")            # [1,10,1,1] bool
    vi("nonempty", TensorProto.BOOL, [1, 10, 1, 1])
    n("Not", ["nonempty"], "isempty")                        # [1,10,1,1] bool
    vi("isempty", TensorProto.BOOL, [1, 10, 1, 1])
    n("Cast", ["isempty"], "isempti", to=TensorProto.INT32)
    vi("isempti", TensorProto.INT32, [1, 10, 1, 1])
    init("big_i", np.array(1000000, np.int32), np.int32)
    n("Mul", ["isempti", "big_i"], "bigmask")               # [1,10,1,1] i32
    vi("bigmask", TensorProto.INT32, [1, 10, 1, 1])
    n("Add", ["chcnt", "bigmask"], "adj")                    # [1,10,1,1] i32
    vi("adj", TensorProto.INT32, [1, 10, 1, 1])
    n("ReduceMin", ["adj"], "minpos", axes=[1], keepdims=1)  # [1,1,1,1] i32
    vi("minpos", TensorProto.INT32, [1, 1, 1, 1])
    n("Equal", ["chcnt", "minpos"], "isant")                 # [1,10,1,1] bool
    vi("isant", TensorProto.BOOL, [1, 10, 1, 1])
    n("Cast", ["isant"], "A", to=TensorProto.FLOAT)          # [1,10,1,1] f32
    vi("A", TensorProto.FLOAT, [1, 10, 1, 1])

    # ---- output = Where(mask, A, input) ------------------------------------
    n("Where", ["mask", "A", "input"], "output")             # [1,10,30,30] f32

    return _model(nodes, inits, vinfos)
