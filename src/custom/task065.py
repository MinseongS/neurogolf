"""Task 065 (2dc579da): fold the dotted quadrant onto an s x s grid.

Rule (ARC-GEN): input is a (2s+1)x(2s+1) grid (s in 1..7) split into four s x s
quadrants by a central cross of `linecolor` (row s and col s).  The whole grid is
background `b` except a single `dotcolor` pixel in one quadrant.  Output is an
s x s grid of `b` carrying that dot at the same in-quadrant position.  The three
colours (line/dot/b) are distinct, so:
  - dotcolor = the colour with cell-count 1
  - b        = the colour with cell-count 4s^2-1 = tot - 2*sqrt(tot)

Memory floor-break (label map + final Equal, small working canvas).
Find the dot's (row,col) by reducing the FREE input over each spatial axis once
([1,10,30,1]/[1,10,1,30] marginals) then GATHERING the dotcolour channel out
(120B each) — no 30x30 dotmask and no masked-copy marginals.  Fold the position
to (dr,dc) = (row mod (s+1), col mod (s+1)) and build a uint8 label map on a
WORK x WORK (=7x7) canvas:
    L = (r==dr & c==dc) ? dotcolor : b   for r<s,c<s ; sentinel 10 outside.
Pad L to 30x30 with the sentinel, then `output = Equal(L, arange[1,10,1,1])`
(free BOOL output, opset 11).  No canvas-sized float stack survives.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

WORK = 7  # working canvas side (>= max output size s = 7)
F = TensorProto.FLOAT
I32 = TensorProto.INT32
U8 = TensorProto.UINT8


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **kw):
        nodes.append(helper.make_node(op, ins, [out], **kw))
        return out

    # ---- constants ----
    init("c1f", np.array(1.0, np.float32), np.float32)
    init("c2f", np.array(2.0, np.float32), np.float32)
    init("chalf", np.array(0.5, np.float32), np.float32)
    init("idxH", np.arange(WORK, dtype=np.int32).reshape(1, 1, WORK, 1), np.int32)
    init("idxW", np.arange(WORK, dtype=np.int32).reshape(1, 1, 1, WORK), np.int32)
    init("idxH30", np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1), np.float32)
    init("idxW30", np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30), np.float32)
    init("chvals", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)

    # ---- size scalars: tot=(2s+1)^2, sq=2s+1, s ----
    n("ReduceSum", ["input"], "tot", keepdims=1)                   # [1,1,1,1]
    n("Sqrt", ["tot"], "sq")                                       # 2s+1
    n("Sub", ["sq", "c1f"], "sqm1")
    n("Mul", ["sqm1", "chalf"], "sf")                              # s (float)
    n("Add", ["sf", "c1f"], "sp1f")                                # s+1 (float)
    n("Cast", ["sf"], "s_i", to=I32)                               # s int [1,1,1,1]

    # ---- channel counts -> dotvec (count==1), b count = tot-2*sq -> bcolour ----
    n("ReduceSum", ["input"], "cntf", axes=[2, 3], keepdims=1)     # [1,10,1,1]
    init("c1i", np.array(1, np.int32), np.int32)
    n("Cast", ["cntf"], "cnt", to=I32)
    n("Equal", ["cnt", "c1i"], "dotvec_b")                         # [1,10,1,1] dot chan
    n("Cast", ["dotvec_b"], "dotvec", to=F)
    n("Mul", ["sq", "c2f"], "twosq")
    n("Sub", ["tot", "twosq"], "bcountf")                          # 4s^2-1
    n("Cast", ["bcountf"], "bcount", to=I32)
    n("Equal", ["cnt", "bcount"], "bvec_b")                        # [1,10,1,1] b chan

    # ---- dotcolor / b colour indices (uint8 scalars) ----
    n("Mul", ["dotvec", "chvals"], "dcparts")
    n("ReduceSum", ["dcparts"], "dcf", keepdims=1)                 # scalar dotcolor
    n("Cast", ["dcf"], "dot8", to=U8)
    n("Cast", ["dcf"], "dot_i", to=I32)
    init("shp1", np.array([1], np.int64), np.int64)
    nodes.append(helper.make_node("Reshape", ["dot_i", "shp1"], ["dot1"]))  # [1] int32 channel idx
    n("Cast", ["bvec_b"], "bvecf", to=F)
    n("Mul", ["bvecf", "chvals"], "bparts")
    n("ReduceSum", ["bparts"], "bf", keepdims=1)                   # scalar b colour
    n("Cast", ["bf"], "b8", to=U8)

    # ---- dot row/col from per-channel marginals (no 30x30 dotmask) ----
    # Reduce once over each spatial axis, then GATHER the dotcolour channel
    # (120B [1,1,30,1]/[1,1,1,30]) instead of Mul+ReduceSum masked copies.
    n("ReduceSum", ["input"], "rc", axes=[3], keepdims=1)          # [1,10,30,1] 1200B
    n("Gather", ["rc", "dot1"], "dotrow", axis=1)                  # [1,1,30,1] 1@row
    n("ReduceSum", ["input"], "cc_", axes=[2], keepdims=1)         # [1,10,1,30] 1200B
    n("Gather", ["cc_", "dot1"], "dotcol", axis=1)                 # [1,1,1,30] 1@col

    n("Mul", ["idxH30", "dotrow"], "rrw")
    n("ReduceSum", ["rrw"], "row", keepdims=1)                     # scalar row
    n("Mul", ["idxW30", "dotcol"], "ccw")
    n("ReduceSum", ["ccw"], "col", keepdims=1)                     # scalar col

    # ---- fold: dr = row mod (s+1), dc = col mod (s+1) ----
    def fold(val, out):
        n("Div", [val, "sp1f"], out + "_q")
        n("Floor", [out + "_q"], out + "_fl")
        n("Mul", [out + "_fl", "sp1f"], out + "_m")
        n("Sub", [val, out + "_m"], out)                           # scalar folded
    fold("row", "dr")
    fold("col", "dc")
    n("Cast", ["dr"], "dri", to=I32)                               # [1,1,1,1]
    n("Cast", ["dc"], "dci", to=I32)

    # ---- masks on WORK canvas ----
    n("Equal", ["idxH", "dri"], "rdot")                            # [1,1,WORK,1]
    n("Equal", ["idxW", "dci"], "cdot")                            # [1,1,1,WORK]
    n("And", ["rdot", "cdot"], "dot")                              # [1,1,WORK,WORK]
    n("Less", ["idxH", "s_i"], "rin")                              # r < s
    n("Less", ["idxW", "s_i"], "cin")                              # c < s
    n("And", ["rin", "cin"], "ingrid")                             # [1,1,WORK,WORK]

    # ---- label map L = ingrid ? (dot ? dotcolor : b) : sentinel ----
    init("vout", np.array(10, np.uint8), np.uint8)
    n("Where", ["dot", "dot8", "b8"], "Lin")                       # dotcolor / b
    n("Where", ["ingrid", "Lin", "vout"], "L7")                    # sentinel outside

    # ---- pad to 30x30 with sentinel, final Equal ----
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    n("Pad", ["L7", "padpads", "padval"], "L", mode="constant")    # [1,1,30,30] uint8
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                            # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
