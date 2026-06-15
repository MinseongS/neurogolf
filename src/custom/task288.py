"""Task 288 (ARC-AGI b8cdaf2b): complete the "robot" figure by drawing the two
antenna diagonals.

Rule (verified exact on all 267 stored examples). The s x s grid holds a figure
whose bottom two rows are: bottom row (s-1) has shoulder shirt cells on corners
and neck antenna cells in the middle; row (s-2) has neck shirt cells in middle.
The output ADDS two antenna-colored diagonals from each shoulder upward-outward.

  * left  diagonal: row - col == s-2-shoulder,  0 <= col < shoulder
  * right diagonal: row + col == 2s-3-shoulder,  s-shoulder <= col < s

s and shoulder are read from column occupancy:
  * s = number of occupied columns
  * 2*shoulder = number of columns with exactly one figure cell

Antenna color = channel with minimal positive cell count.

Memory floor-break (eliminate the [1,1,30,30] occ plane + 1D column masks):

  Old: Conv(input, Wocc) -> occ [1,1,30,30] (3600B) -> ReduceSum -> colcnt
  New: Single Conv with 30x1 kernel summing all figure channels over all rows
       -> colcnt [1,1,1,30] directly (no 2D intermediate).

  Old: COL [1,1,30,30] param (900 elements) for column comparisons
  New: COL1D [1,1,1,30] (30 elements); the column conditions ltSH, ltS, geSmSH
       are 1D [1,1,1,30] (30B each) and broadcast over the 2D diagonal masks.

  Old: separate geSmSH, ltS, Rmask0 as [1,1,30,30] planes
  New: right_cols = And(geSmSH1d, ltS1d) in 1D (30B), then
       Rmask = And(onLright, right_cols) directly (no Rmask0 intermediate).

Remaining: 5 bool [1,1,30,30] planes (onLleft, Lmask, onLright, Rmask, mask)
           = 4500B — unavoidable for 2D diagonal detection.
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

    # ---- per-column figure count colcnt [1,1,1,30] via a single Conv --------
    # W_col [1,10,30,1]: sums channels 1..9 across all 30 rows in a single conv.
    # input [1,10,30,30] * W_col[1,10,30,1] -> colcnt [1,1,1,30]
    W_col = np.zeros((1, 10, 30, 1), np.float32)
    W_col[0, 1:, :, 0] = 1.0
    init("W_col", W_col, np.float32)
    n("Conv", ["input", "W_col"], "colcnt",
      kernel_shape=[30, 1], pads=[0, 0, 0, 0])                   # [1,1,1,30] f32
    vi("colcnt", TensorProto.FLOAT, [1, 1, 1, 30])

    n("Cast", ["colcnt"], "colcnti", to=TensorProto.INT32)
    vi("colcnti", TensorProto.INT32, [1, 1, 1, 30])

    # s = number of occupied columns
    init("zero_i", np.array(0, np.int32), np.int32)
    n("Greater", ["colcnti", "zero_i"], "occcol")                 # [1,1,1,30] bool
    vi("occcol", TensorProto.BOOL, [1, 1, 1, 30])
    n("Cast", ["occcol"], "occcoli", to=TensorProto.INT32)
    vi("occcoli", TensorProto.INT32, [1, 1, 1, 30])
    n("ReduceSum", ["occcoli"], "S", axes=[3], keepdims=1)        # [1,1,1,1] i32
    vi("S", TensorProto.INT32, [1, 1, 1, 1])

    # 2*shoulder = number of columns with exactly one figure cell
    init("one_i", np.array(1, np.int32), np.int32)
    n("Equal", ["colcnti", "one_i"], "iscorner")                  # [1,1,1,30] bool
    vi("iscorner", TensorProto.BOOL, [1, 1, 1, 30])
    n("Cast", ["iscorner"], "iscorneri", to=TensorProto.INT32)
    vi("iscorneri", TensorProto.INT32, [1, 1, 1, 30])
    n("ReduceSum", ["iscorneri"], "twoSH", axes=[3], keepdims=1)  # [1,1,1,1]
    vi("twoSH", TensorProto.INT32, [1, 1, 1, 1])
    init("two_i", np.array(2, np.int32), np.int32)
    n("Div", ["twoSH", "two_i"], "SH")                           # [1,1,1,1] i32
    vi("SH", TensorProto.INT32, [1, 1, 1, 1])

    # ---- diagonal offsets ---------------------------------------------------
    # Kleft  = S - 2 - SH;  Kright = 2S - 3 - SH
    init("c2", np.array(2, np.int32), np.int32)
    init("c3", np.array(3, np.int32), np.int32)
    n("Sub", ["S", "SH"], "SmSH")                                 # S - SH
    vi("SmSH", TensorProto.INT32, [1, 1, 1, 1])
    n("Sub", ["SmSH", "c2"], "Kleft")                             # S - SH - 2
    vi("Kleft", TensorProto.INT32, [1, 1, 1, 1])
    n("Add", ["S", "SmSH"], "twoSmSH")                            # 2S - SH
    vi("twoSmSH", TensorProto.INT32, [1, 1, 1, 1])
    n("Sub", ["twoSmSH", "c3"], "Kright")                         # 2S - SH - 3
    vi("Kright", TensorProto.INT32, [1, 1, 1, 1])

    # ---- fixed row-minus-col and row-plus-col planes (2D params) -----------
    rr = np.arange(30, dtype=np.int32).reshape(1, 1, 30, 1) * np.ones(
        (1, 1, 1, 30), np.int32)
    cc = np.ones((1, 1, 30, 1), np.int32) * np.arange(
        30, dtype=np.int32).reshape(1, 1, 1, 30)
    init("RmC", (rr - cc).astype(np.int32), np.int32)            # [1,1,30,30]
    init("RpC", (rr + cc).astype(np.int32), np.int32)            # [1,1,30,30]

    # ---- 1D column index (replaces 2D COL param, saves 870 param elements) -
    init("COL1D", np.arange(30, dtype=np.int32).reshape(1, 1, 1, 30), np.int32)

    # ---- left diagonal mask: (RmC == Kleft) & (COL1D < SH) ----------------
    n("Equal", ["RmC", "Kleft"], "onLleft")                       # [1,1,30,30] bool
    vi("onLleft", TensorProto.BOOL, [1, 1, 30, 30])
    # COL1D < SH: [1,1,1,30] bool (1D, broadcasts)
    n("Less", ["COL1D", "SH"], "ltSH")                            # [1,1,1,30] bool
    vi("ltSH", TensorProto.BOOL, [1, 1, 1, 30])
    n("And", ["onLleft", "ltSH"], "Lmask")                        # [1,1,30,30] bool
    vi("Lmask", TensorProto.BOOL, [1, 1, 30, 30])

    # ---- right diagonal mask: (RpC == Kright) & (COL1D >= S-SH) & (COL1D < S)
    n("Equal", ["RpC", "Kright"], "onLright")                     # [1,1,30,30] bool
    vi("onLright", TensorProto.BOOL, [1, 1, 30, 30])
    # Compute right column conditions in 1D first, then And with 2D diagonal
    init("one_i2", np.array(1, np.int32), np.int32)
    n("Sub", ["SmSH", "one_i2"], "SmSHm1")                        # S-SH-1
    vi("SmSHm1", TensorProto.INT32, [1, 1, 1, 1])
    n("Less", ["SmSHm1", "COL1D"], "geSmSH")                      # [1,1,1,30] bool (1D)
    vi("geSmSH", TensorProto.BOOL, [1, 1, 1, 30])
    n("Less", ["COL1D", "S"], "ltS")                              # [1,1,1,30] bool (1D)
    vi("ltS", TensorProto.BOOL, [1, 1, 1, 30])
    # Combine right column conditions in 1D (30B) before the 2D And
    n("And", ["geSmSH", "ltS"], "right_cols")                     # [1,1,1,30] bool (1D)
    vi("right_cols", TensorProto.BOOL, [1, 1, 1, 30])
    n("And", ["onLright", "right_cols"], "Rmask")                 # [1,1,30,30] bool
    vi("Rmask", TensorProto.BOOL, [1, 1, 30, 30])

    n("Or", ["Lmask", "Rmask"], "mask")                           # [1,1,30,30] bool
    vi("mask", TensorProto.BOOL, [1, 1, 30, 30])

    # ---- antenna one-hot colour A[1,10,1,1] --------------------------------
    n("ReduceSum", ["input"], "chcntf", axes=[2, 3], keepdims=1)  # [1,10,1,1]
    vi("chcntf", TensorProto.FLOAT, [1, 10, 1, 1])
    n("Cast", ["chcntf"], "chcnt", to=TensorProto.INT32)
    vi("chcnt", TensorProto.INT32, [1, 10, 1, 1])
    n("Greater", ["chcnt", "zero_i"], "nonempty")
    vi("nonempty", TensorProto.BOOL, [1, 10, 1, 1])
    n("Not", ["nonempty"], "isempty")
    vi("isempty", TensorProto.BOOL, [1, 10, 1, 1])
    n("Cast", ["isempty"], "isempti", to=TensorProto.INT32)
    vi("isempti", TensorProto.INT32, [1, 10, 1, 1])
    init("big_i", np.array(1000000, np.int32), np.int32)
    n("Mul", ["isempti", "big_i"], "bigmask")
    vi("bigmask", TensorProto.INT32, [1, 10, 1, 1])
    n("Add", ["chcnt", "bigmask"], "adj")
    vi("adj", TensorProto.INT32, [1, 10, 1, 1])
    n("ReduceMin", ["adj"], "minpos", axes=[1], keepdims=1)
    vi("minpos", TensorProto.INT32, [1, 1, 1, 1])
    n("Equal", ["chcnt", "minpos"], "isant")
    vi("isant", TensorProto.BOOL, [1, 10, 1, 1])
    n("Cast", ["isant"], "A", to=TensorProto.FLOAT)
    vi("A", TensorProto.FLOAT, [1, 10, 1, 1])

    # ---- output = Where(mask, A, input) ------------------------------------
    n("Where", ["mask", "A", "input"], "output")

    return _model(nodes, inits, vinfos)
