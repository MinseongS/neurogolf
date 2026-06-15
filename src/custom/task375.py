"""Task 375 (ARC-AGI ea786f4a) — both diagonals black on a solid square.

Rule (from the generator): the input is a solid `size`x`size` square of a single
`color` placed at the top-left of the canvas (size is odd, 3..13+), with the
single centre cell turned black.  The output keeps the same square but paints
BOTH diagonals black (colour 0): out[r][c] = 0 if (r==c or r+c==size-1) else
color, for in-grid cells; off-grid cells stay all-channels-off.

Diagonal membership uses the canvas coords directly because the square is
anchored at the origin (r,c in 0..size-1): main diagonal r==c, anti-diagonal
r+c==(size-1).

Memory floor-break (label map + final Equal):
  * Conv arange weights -> colour plane G; Conv ones weights -> in-grid count.
  * color scalar = ReduceMax(G) (the lone black centre is 0, so the square's
    colour dominates).  s1 = size-1 = max occupied row index.
  * main-diag mask = constant identity plane (r==c); anti-diag mask =
    Equal(r+c, s1) from a constant coordinate-sum plane.
  * uint8 label L[1,1,30,30] = ingrid ? ((diag) ? 0 : color) : 10, and the free
    BOOL output = Equal(L, arange[1,10,1,1]) (opset 11).
All values are small integers, exact in float32 / uint8.  Only one ~900B uint8
plane (L) is ever materialised.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
B = TensorProto.BOOL


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- small working canvas: the square is anchored at the origin and the
    # generator size is 2*randint(2,7)+1 <= 15, so all live cells lie in the
    # top-left W x W block.  Do every per-cell op there (225 elem) and Pad the
    # final label to 30x30 with the off-grid sentinel. ----
    W = 15

    # ---- color scalar: corner (0,0) is always `color` (size>=3, centre black is
    # interior).  Slice the corner one-hot vector and contract with arange. ----
    init("c00s", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("c00e", np.array([1, 10, 1, 1], np.int64), np.int64)
    init("c00ax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "c00s", "c00e", "c00ax"], "corner")  # [1,10,1,1]
    Wc = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("Wc", Wc, np.float32)
    n("Mul", ["corner", "Wc"], "corner_w")          # [1,10,1,1]
    n("ReduceSum", ["corner_w"], "color", axes=[1], keepdims=1)  # [1,1,1,1] f32

    # ---- s1 = size-1 = max occupied row index.  occ_row[r] = any channel set in
    # row r (incl. black centre) = ReduceMax over channel+col axes. ----
    n("ReduceMax", ["input"], "rowocc_f", axes=[1, 3], keepdims=1)  # [1,1,30,1]
    init("st0", np.array([0], np.int64), np.int64)
    init("enW", np.array([W], np.int64), np.int64)
    init("ax2", np.array([2], np.int64), np.int64)
    n("Slice", ["rowocc_f", "st0", "enW", "ax2"], "rowocc")  # [1,1,W,1]
    Irow = np.arange(W, dtype=np.float32).reshape(1, 1, W, 1)
    init("Irow", Irow, np.float32)
    n("Mul", ["rowocc", "Irow"], "rowidx")          # [1,1,W,1]
    n("ReduceMax", ["rowidx"], "s1", keepdims=1)     # [1,1,1,1] = size-1 (f32)

    # ---- in-grid = (r <= s1) AND (c <= s1)  (separable 1-D bounds) ----
    Icol = np.arange(W, dtype=np.float32).reshape(1, 1, 1, W)
    init("Icol", Icol, np.float32)
    init("NegHalf", np.array(-0.5, np.float32), np.float32)
    n("Sub", ["s1", "Irow"], "rin_d")               # s1 - r  >= 0 inside
    n("Greater", ["rin_d", "NegHalf"], "rin_b")     # r <= s1  [1,1,W,1] bool
    n("Sub", ["s1", "Icol"], "cin_d")
    n("Greater", ["cin_d", "NegHalf"], "cin_b")     # c <= s1  [1,1,1,W] bool
    n("And", ["rin_b", "cin_b"], "ingrid_b")        # [1,1,W,W] bool

    # ---- diagonal masks (canvas == local coords; square anchored at origin) ----
    # main diagonal: r == c
    EyeM = np.eye(W, dtype=np.uint8).reshape(1, 1, W, W)
    init("EyeM", EyeM, np.uint8)
    init("u1", np.array(1, np.uint8), np.uint8)
    n("Equal", ["EyeM", "u1"], "maindiag_b")        # [1,1,W,W] bool
    # anti diagonal: r + c == s1
    RC = (np.arange(W).reshape(W, 1) + np.arange(W).reshape(1, W))
    RC = RC.astype(np.float32).reshape(1, 1, W, W)
    init("RC", RC, np.float32)
    n("Equal", ["RC", "s1"], "antidiag_b")          # broadcast [1,1,W,W] bool
    n("Or", ["maindiag_b", "antidiag_b"], "diag_b")

    # ---- assemble uint8 label map (W x W) ----
    n("Cast", ["color"], "color_u", to=U8)          # [1,1,1,1] uint8
    init("u0", np.array(0, np.uint8), np.uint8)      # black (diag colour)
    init("u10", np.array(10, np.uint8), np.uint8)    # off-grid sentinel
    # in-grid cell colour: diag -> 0 else color
    n("Where", ["diag_b", "u0", "color_u"], "Lin")  # [1,1,W,W] uint8
    n("Where", ["ingrid_b", "Lin", "u10"], "Lsm")   # off-grid -> 10

    # pad Lsm [1,1,W,W] -> L [1,1,30,30] with sentinel 10 (all pads off-grid)
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64), np.int64)
    n("Pad", ["Lsm", "pads", "u10"], "L", mode="constant")  # [1,1,30,30]

    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["L", "chan"], "output")             # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task375", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
