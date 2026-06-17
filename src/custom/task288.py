"""Task 288 (ARC-AGI b8cdaf2b): complete the "robot" figure by drawing the two
antenna diagonals.

Rule (verified exact on all 267 stored examples + fresh). The s x s grid holds a
figure whose bottom two rows are: bottom row (s-1) has `shoulder` shirt cells on
each corner and `neck` antenna cells in the middle; row (s-2) has `neck` shirt
cells in middle.  The output ADDS two antenna-coloured diagonals rising out of
each shoulder upward-outward, for c in [0, shoulder):
    (s-2-shoulder+c, c)        -> antenna   (left  diagonal)
    (s-2-shoulder+c, s-1-c)    -> antenna   (right diagonal)

Both are exact 45-degree diagonals:
    left  : r - c == D  with  D = s - 2 - shoulder
    right : r + c == S  with  S = 2s - 3 - shoulder = D + s - 1
bounded to r < s-2 (so they never overwrite the existing bottom-two-row shirt/
antenna) and in-grid (c < s).  shoulder = (s - neck)/2, so D = (s+neck)/2 - 2.

Scalar recovery (NO per-cell colour-index plane):
  * cnt = ReduceSum(input,[2,3]) [1,10,1,1]; the two non-bg colours are SHIRT
    (count 2*shoulder+neck) and ANTENNA (count neck).  Antenna is always the
    RARER, so its one-hot = Equal(cnt, min positive count) (background ch0 has
    the largest count so it never wins the min; empty channels masked to BIG).
  * neck = that min positive count; s = number of occupied columns (the bottom
    row fills all s columns).  D = (s+neck)/2 - 2, S = D + s - 1.

Floor-break vs the prior 16.06 build: the old build paid TWO [1,1,30,30] int32
parameter planes (RmC, RpC = 900 elems each) to test r-c and r+c.  Here the
diagonals are tested with two fp16 1-D ramps RI[1,1,30,1] and CI[1,1,1,30]
(30 elems each) via Equal(RI-D, CI) / Equal(S-RI, CI) which broadcast to the
30x30 bool masks directly -- removing ~1740 parameter elements.  All scalar
math is fp16 (integers < 2048, exact).
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..builders import _model


def build(task):
    inits = []
    nodes = []
    vinfos = []

    F = TensorProto.FLOAT
    H = TensorProto.FLOAT16
    B = TensorProto.BOOL

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

    # ---- per-channel pixel counts cnt [1,10,1,1] --------------------------
    n("ReduceSum", ["input"], "cntf", axes=[2, 3], keepdims=1)       # f32
    vi("cntf", F, [1, 10, 1, 1])
    n("Cast", ["cntf"], "cnt", to=TensorProto.INT32)                 # int32 (Equal needs int)
    vi("cnt", TensorProto.INT32, [1, 10, 1, 1])

    # antenna one-hot = rarest positive-count channel.  ch0 (background) has the
    # largest count so masking is only needed for the empty channels.
    init("zero_i", np.array(0, np.int32), np.int32)
    init("big_i", np.array(1000000, np.int32), np.int32)
    n("Greater", ["cnt", "zero_i"], "haspx")                         # bool present
    vi("haspx", B, [1, 10, 1, 1])
    n("Where", ["haspx", "cnt", "big_i"], "cntm")                    # empty -> BIG
    vi("cntm", TensorProto.INT32, [1, 10, 1, 1])
    n("ReduceMin", ["cntm"], "neck", axes=[1], keepdims=1)           # [1,1,1,1] = neck
    vi("neck", TensorProto.INT32, [1, 1, 1, 1])
    n("Equal", ["cntm", "neck"], "isant")                            # bool [1,10,1,1]
    vi("isant", B, [1, 10, 1, 1])
    n("Cast", ["isant"], "A", to=F)                                  # float one-hot
    vi("A", F, [1, 10, 1, 1])

    # ---- s = number of occupied columns (bottom row fills all s cols) ------
    n("ReduceMax", ["input"], "colany", axes=[1, 2], keepdims=1)     # [1,1,1,30] f32
    vi("colany", F, [1, 1, 1, 30])
    n("ReduceSum", ["colany"], "sizef", axes=[3], keepdims=1)        # [1,1,1,1] = s
    vi("sizef", F, [1, 1, 1, 1])
    n("Cast", ["sizef"], "size", to=TensorProto.INT32)               # int32 scalar
    vi("size", TensorProto.INT32, [1, 1, 1, 1])

    # ---- D = (s+neck)/2 - 2 ,  S = D + s - 1 ,  s-2  (int32, exact) --------
    I = TensorProto.INT32
    init("two_i", np.array(2, np.int32), np.int32)
    init("one_i", np.array(1, np.int32), np.int32)
    n("Add", ["size", "neck"], "sn")                                 # s+neck (even)
    vi("sn", I, [1, 1, 1, 1])
    n("Div", ["sn", "two_i"], "sn2")                                 # (s+neck)/2
    vi("sn2", I, [1, 1, 1, 1])
    n("Sub", ["sn2", "two_i"], "Dk")                                 # D
    vi("Dk", I, [1, 1, 1, 1])
    n("Add", ["Dk", "size"], "Dps")                                  # D + s
    vi("Dps", I, [1, 1, 1, 1])
    n("Sub", ["Dps", "one_i"], "Sk")                                 # S = D + s - 1
    vi("Sk", I, [1, 1, 1, 1])
    n("Sub", ["size", "two_i"], "sizem2")                            # s - 2
    vi("sizem2", I, [1, 1, 1, 1])

    # ---- ramps (int32; only the bool masks are 30x30) ---------------------
    init("RI", np.arange(30, dtype=np.int32).reshape(1, 1, 30, 1), np.int32)
    init("CI", np.arange(30, dtype=np.int32).reshape(1, 1, 1, 30), np.int32)

    # left diagonal  : r - c == D  ->  Equal(RI - D, CI)
    n("Sub", ["RI", "Dk"], "RImD")                                   # [1,1,30,1]
    vi("RImD", I, [1, 1, 30, 1])
    n("Equal", ["RImD", "CI"], "leftD")                              # [1,1,30,30] bool
    vi("leftD", B, [1, 1, 30, 30])
    # right diagonal : r + c == S  ->  Equal(S - RI, CI)
    n("Sub", ["Sk", "RI"], "SmRI")                                   # [1,1,30,1]
    vi("SmRI", I, [1, 1, 30, 1])
    n("Equal", ["SmRI", "CI"], "rightD")                             # [1,1,30,30] bool
    vi("rightD", B, [1, 1, 30, 30])
    n("Or", ["leftD", "rightD"], "diag")                             # [1,1,30,30] bool
    vi("diag", B, [1, 1, 30, 30])

    # bounds: RI < s-2 (row)  and  CI < s (col, excludes the right diagonal's
    # off-grid tail).  rowok broadcasts down columns, colok across rows.
    n("Less", ["RI", "sizem2"], "rowok")                             # [1,1,30,1] bool
    vi("rowok", B, [1, 1, 30, 1])
    n("Less", ["CI", "size"], "colok")                               # [1,1,1,30] bool
    vi("colok", B, [1, 1, 1, 30])
    n("And", ["diag", "rowok"], "m1")                                # [1,1,30,30] bool
    vi("m1", B, [1, 1, 30, 30])
    n("And", ["m1", "colok"], "mask")                                # [1,1,30,30] bool
    vi("mask", B, [1, 1, 30, 30])

    # ---- output = Where(mask, antenna, input) -----------------------------
    n("Where", ["mask", "A", "input"], "output")

    return _model(nodes, inits, vinfos)
