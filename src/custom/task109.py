"""Task 109 (ARC 47c1f68c): 4-fold mirror of a top-left sprite, in linecolor.

Input is a (2s+1)x(2s+1) grid: a linecolor cross at row=s, col=s, and a
`color` sprite in the top-left quadrant (rows,cols < s).  Output is the 2s x 2s
grid where the sprite mask is mirrored into all four quadrants and rendered in
linecolor (cross removed; background elsewhere inside the grid).

Memory floor-break (small canvas + label map + final Equal):
  The whole 2s x 2s output (s <= 6 -> 2s <= 12) fits in the top-left 12 x 12
  corner, so every per-cell intermediate is a 12 x 12 plane (144 elem).  The
  sprite mask M is the occupancy restricted to the top-left s x s; the 4-fold
  mirror is E @ M @ E with the symmetric fold matrix E[R,K]=I[R,K]+[R+K==t]
  (t=2s-1).  A single uint8 label map L (linecolor on the mirror, 0 on in-grid
  background, sentinel 10 outside the 2s x 2s region) is padded to 30 x 30 and
  finished with Equal(L, arange[1,10,1,1]) into the free BOOL `output`.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

WORK = 12  # output side bound = 2*max(s) = 12


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    init("one", np.array([1.0], np.float32), np.float32)
    init("two", np.array([2.0], np.float32), np.float32)

    w_nz = np.zeros((1, 10, 1, 1), np.float32)
    w_nz[0, 1:, 0, 0] = 1.0
    init("w_nz", w_nz, np.float32)

    # per-channel column sums; colour-occupancy column sums via nonzero conv.
    n("ReduceSum", ["input"], "Xc", axes=[2], keepdims=1)        # [1,10,1,30]
    n("Conv", ["Xc", "w_nz"], "colsum")                          # [1,1,1,30]
    n("ReduceMax", ["colsum"], "nval", axes=[3], keepdims=1)     # [1,1,1,1] = n
    n("Sub", ["nval", "two"], "t")                               # t = n-2 = 2s-1
    n("Sub", ["nval", "one"], "m")                               # m = n-1 = 2s
    n("Cast", ["nval"], "nvalI", to=TensorProto.INT32)

    # --- sprite occupancy O: the top-left SRC x SRC corner (sprite rows,cols
    # < s <= 6) lies wholly inside the grid (grid side 2s+1 >= 7), so there
    # O = 1 - ch0.  Slice only channel 0 ([1,1,6,6] = 144B). ---
    SRC = 6
    init("iwst", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("iwen", np.array([1, 1, SRC, SRC], np.int64), np.int64)
    init("iwax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "iwst", "iwen", "iwax"], "ch0src")     # [1,1,SRC,SRC]
    init("one16", np.array([1.0], np.float16), np.float16)
    n("Cast", ["ch0src"], "ch0src16", to=TensorProto.FLOAT16)
    n("Sub", ["one16", "ch0src16"], "Osrc")                     # [1,1,SRC,SRC] f16
    init("opad", np.array([0, 0, 0, 0, 0, 0, WORK - SRC, WORK - SRC], np.int64),
         np.int64)
    init("opv", np.array(0.0, np.float16), np.float16)
    n("Pad", ["Osrc", "opad", "opv"], "O", mode="constant")    # [1,1,WORK,WORK] f16

    # --- linecolor id (scalar uint8) ---
    # The cross column has linecolor count == n; for the linecolor channel the
    # max column-count == n, for every other channel it is < n.  So the
    # linecolor channel is the one whose per-channel max column-sum equals n.
    n("ReduceMax", ["Xc"], "Xcmax", axes=[3], keepdims=1)        # [1,10,1,1]
    n("Cast", ["Xcmax"], "XcmaxI", to=TensorProto.INT32)
    n("Equal", ["XcmaxI", "nvalI"], "LB")                        # [1,10,1,1] bool
    arc = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("arc", arc, np.float32)
    n("Cast", ["LB"], "LBf", to=TensorProto.FLOAT)
    n("Mul", ["LBf", "arc"], "Lidp2")
    n("ReduceMax", ["Lidp2"], "Lid", axes=[1, 2, 3], keepdims=1)  # [1,1,1,1]
    n("Cast", ["Lid"], "Lid_u8", to=TensorProto.UINT8)          # scalar uint8

    # --- sprite mask M = O restricted to top-left sxs (2*row<t & 2*col<t) ---
    twoR = (2 * np.arange(WORK)).reshape(1, 1, WORK, 1).astype(np.float32)
    twoC = (2 * np.arange(WORK)).reshape(1, 1, 1, WORK).astype(np.float32)
    init("twoR", twoR, np.float32)
    init("twoC", twoC, np.float32)
    n("Less", ["twoR", "t"], "rmB")                             # [1,1,WORK,1] bool
    n("Less", ["twoC", "t"], "cmB")                             # [1,1,1,WORK] bool
    n("Cast", ["rmB"], "rm", to=TensorProto.FLOAT16)
    n("Cast", ["cmB"], "cm", to=TensorProto.FLOAT16)
    n("Mul", ["O", "rm"], "Or")
    n("Mul", ["Or", "cm"], "M")                                 # [1,1,WORK,WORK] f16

    # --- fold matrix E[R,K] = I[R,K] + [R+K == t] (symmetric), WORK x WORK ---
    R = np.arange(WORK).reshape(WORK, 1)
    K = np.arange(WORK).reshape(1, WORK)
    init("ImatB", (R == K), np.bool_)                           # [WORK,WORK] bool
    init("RpKI", (R + K).astype(np.int32), np.int32)            # [WORK,WORK] int
    n("Cast", ["t"], "tIs", to=TensorProto.INT32)               # [1,1,1,1]
    n("Reshape",
      ["tIs", init("scl", np.array([1], np.int64), np.int64)], "tscalar")  # [1]
    n("Equal", ["RpKI", "tscalar"], "mirrB")                    # [WORK,WORK] bool
    n("Or", ["ImatB", "mirrB"], "EB")                           # [WORK,WORK] bool
    n("Cast", ["EB"], "E", to=TensorProto.FLOAT16)             # [WORK,WORK] f16

    # --- out_mask = E @ M @ E ---
    n("MatMul", ["E", "M"], "EM")                               # [1,1,WORK,WORK] f16
    n("MatMul", ["EM", "E"], "outm")                            # [1,1,WORK,WORK] f16
    init("half16", np.array([0.5], np.float16), np.float16)
    n("Greater", ["outm", "half16"], "outmB")                  # bool

    # --- gridmask (row<m & col<m), WORK canvas ---
    Ridx = np.arange(WORK).reshape(1, 1, WORK, 1).astype(np.float32)
    Cidx = np.arange(WORK).reshape(1, 1, 1, WORK).astype(np.float32)
    init("Ridx", Ridx, np.float32)
    init("Cidx", Cidx, np.float32)
    n("Less", ["Ridx", "m"], "grB")                            # [1,1,WORK,1] bool
    n("Less", ["Cidx", "m"], "gcB")                            # [1,1,1,WORK] bool
    n("And", ["grB", "gcB"], "gridB")                          # [1,1,WORK,WORK]

    # --- uint8 label map L: linecolor on mirror, 0 in-grid bg, 10 outside ---
    init("v0", np.array(0, np.uint8), np.uint8)
    init("v10", np.array(10, np.uint8), np.uint8)
    n("Where", ["gridB", "v0", "v10"], "L0")                   # 0 in-grid else 10
    n("Where", ["outmB", "Lid_u8", "L0"], "Lwk")               # mirror -> linecolor

    # pad L back to 30x30 (sentinel 10), then final Equal
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    n("Pad", ["Lwk", "padpads", "padval"], "L", mode="constant")  # [1,1,30,30]

    init("chan10", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan10"], "output")                       # -> free BOOL

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task109", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
