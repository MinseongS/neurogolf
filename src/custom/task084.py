"""Task 084 (3bd67248) — scalar-keyed fixed template -> uint8 label -> FREE output.

Rule (generator): square n x n grid (n=3..21), left column painted colour c (1..9).
Output: col0 (rows 0..n-1) = c; bottom row (r=n-1, cols>=1) = yellow(4);
anti-diagonal (r+c==n-1, cols>=1) = red(2); other in-grid cells = bg(0); off-grid unset.

FIXED template keyed by two O(1) scalars (size n, colour c):
  n = sqrt(Sum(input))                         (one-hot per cell => Sum = n*n)
  c = ArgMax_chan(per-channel pixel counts)    (colour count n > red/yellow n-1)

Because n<=21, every meaningful cell sits in rows 0..20 / cols 0..20, so the working
label block is only [1,1,21,21]; rows/cols >=21 are always off-grid and are filled by
ONE Pad with the off-grid sentinel 10.  Column 0 is a [1,1,21,1] strip (colour c where
in-grid).  The cols-1..20 body is built from a per-ROW base VECTOR (yellow bottom / bg /
sentinel) gated to in-grid columns, with the red anti-diagonal overlaid.  Equal(L,
channel-ramp) routes the one-hot into the FREE bool output (no value/colour plane).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from src.harness import IR_VERSION

F32 = TensorProto.FLOAT
I32 = TensorProto.INT32
U8 = TensorProto.UINT8
B = TensorProto.BOOL

RW = 21   # working rows  (0..20  -> covers n up to 21)
CW = 20   # working body cols (1..20)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- n (size) scalar ----
    n("ReduceSum", ["input"], "ss_f", keepdims=0)         # n^2 (f32)
    n("Sqrt", ["ss_f"], "nf")
    n("Cast", ["nf"], "ni", to=I32)                       # n

    # ---- colour scalar c ----
    n("ReduceSum", ["input"], "cnt", axes=[2, 3], keepdims=0)  # [1,10] (40B)
    chmask = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1]], np.float32)
    init("chmask", chmask, np.float32)
    n("Mul", ["cnt", "chmask"], "cntm")
    n("ArgMax", ["cntm"], "ci64", axis=1, keepdims=0)     # [1] int64 = c
    n("Cast", ["ci64"], "cu8", to=U8)                     # [1] uint8

    # ---- ramps (cropped to the working block) & scalars ----
    init("rr", np.arange(RW, dtype=np.int32).reshape(1, 1, RW, 1), np.int32)     # 0..20
    init("ccb", np.arange(1, 1 + CW, dtype=np.int32).reshape(1, 1, 1, CW), np.int32)  # 1..20
    init("one_i", np.array(1, np.int32), np.int32)
    n("Sub", ["ni", "one_i"], "nm1")                      # n-1
    n("Sub", ["nm1", "ccb"], "diagt")                     # [1,1,1,CW] = n-1-col

    # vector conditions
    n("Equal", ["rr", "nm1"], "isbot")                    # [1,1,RW,1] r==n-1
    n("Less", ["rr", "ni"], "rowin")                      # [1,1,RW,1] r<n
    n("Less", ["ccb", "ni"], "colin")                     # [1,1,1,CW] c<n
    n("Equal", ["rr", "diagt"], "isred")                  # [1,1,RW,CW] anti-diagonal

    init("u0", np.array(0, np.uint8), np.uint8)
    init("u2", np.array(2, np.uint8), np.uint8)
    init("u4", np.array(4, np.uint8), np.uint8)
    init("u10", np.array(10, np.uint8), np.uint8)

    # per-ROW base value VECTOR: row n-1 -> 4, rows 0..n-2 -> 0, rows>=n -> 10
    n("Where", ["rowin", "u0", "u10"], "rowin_v")         # [1,1,RW,1]
    n("Where", ["isbot", "u4", "rowin_v"], "rowval")      # [1,1,RW,1] vector
    # gate to in-grid columns (off-grid col -> sentinel) -> ONE full block
    n("Where", ["colin", "rowval", "u10"], "base")        # [1,1,RW,CW]
    # overlay red anti-diagonal (2)
    n("Where", ["isred", "u2", "base"], "body")           # [1,1,RW,CW]

    # column-0 strip: colour c where in-grid row else sentinel (vector)
    n("Where", ["rowin", "cu8", "u10"], "col0")           # [1,1,RW,1]

    # assemble [1,1,RW,1+CW] then Pad to [1,1,30,30] with sentinel 10
    n("Concat", ["col0", "body"], "Lblk", axis=3)         # [1,1,RW,CW+1]
    pads = np.array([0, 0, 0, 0, 0, 0, 30 - RW, 30 - (CW + 1)], np.int64)
    init("pads", pads, np.int64)
    n("Pad", ["Lblk", "pads", "u10"], "L", mode="constant")  # [1,1,30,30] uint8

    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["L", "chan"], "output")                   # [1,10,30,30] bool

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task084", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
