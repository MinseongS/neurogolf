"""Task 109 (ARC 47c1f68c): 4-fold mirror of a top-left sprite, in linecolor.

Input is a (2s+1)x(2s+1) grid: a linecolor cross at row=s, col=s, and a
`color` sprite in the top-left quadrant (rows,cols < s).  Output is the 2s x 2s
grid where the sprite mask is mirrored into all four quadrants and rendered in
linecolor (cross removed; in-grid background elsewhere).

Memory floor-break (small canvas + index-gather mirror + label map + Equal):
  s in [3,6] -> sprite lives in the top-left 6x6 corner; output side 2s <= 12.
  The 4-fold mirror is SEPARABLE: out[r,c] = sprite[min(r,t-r), min(c,t-c)] with
  t = 2s-1, since min(i,t-i) in 0..s-1 folds each output index back into the
  s x s sprite quadrant.  Realised as TWO data-dependent Gathers on a uint8 6x6
  sprite (no fold-matrix MatMul, planes stay uint8/bool).  A single uint8 label
  map L (linecolor on the mirror, 0 on in-grid background, sentinel 10 outside
  the 2s x 2s region) is padded to 30 x 30 and finished with
  Equal(L, arange[1,10,1,1]) into the free BOOL `output`.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

WORK = 12   # output side bound = 2*max(s) = 12
SRC = 6     # sprite corner bound = max(s) = 6


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
    n("ReduceMax", ["colsum"], "nval", axes=[3], keepdims=1)     # [1,1,1,1] = n=2s+1
    n("Sub", ["nval", "two"], "t")                               # t = n-2 = 2s-1
    n("Sub", ["nval", "one"], "m")                               # m = n-1 = 2s
    n("Cast", ["nval"], "nvalI", to=TensorProto.INT32)

    # --- linecolor id (scalar uint8): channel (k>=1) whose max per-channel
    # column-sum == n (only the cross column fills a whole column). ---
    n("ReduceMax", ["Xc"], "Xcmax", axes=[3], keepdims=1)        # [1,10,1,1]
    n("Cast", ["Xcmax"], "XcmaxI", to=TensorProto.INT32)
    n("Equal", ["XcmaxI", "nvalI"], "LB")                        # [1,10,1,1] bool
    arc = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("arc", arc, np.float32)
    n("Cast", ["LB"], "LBf", to=TensorProto.FLOAT)
    n("Mul", ["LBf", "arc"], "Lidp2")
    n("ReduceMax", ["Lidp2"], "Lid", axes=[1, 2, 3], keepdims=1)  # [1,1,1,1]
    n("Cast", ["Lid"], "Lid_u8", to=TensorProto.UINT8)           # scalar uint8

    # --- sprite mask (uint8 {0,1}) on the SRC x SRC top-left corner ---
    # In the top-left s x s quadrant the only non-bg colour is the sprite
    # (cross & other quadrants are bg or outside the quadrant). Restrict to
    # rows<s & cols<s via 2*idx < t.
    init("iwst", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("iwen", np.array([1, 1, SRC, SRC], np.int64), np.int64)
    init("iwax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "iwst", "iwen", "iwax"], "ch0src")      # [1,1,SRC,SRC] f32
    init("zerof", np.array([0.0], np.float32), np.float32)
    n("Equal", ["ch0src", "zerof"], "nonbg_b")                   # bool: non-background
    # restrict to top-left s x s quadrant
    twoR6 = (2 * np.arange(SRC)).reshape(1, 1, SRC, 1).astype(np.float32)
    twoC6 = (2 * np.arange(SRC)).reshape(1, 1, 1, SRC).astype(np.float32)
    init("twoR6", twoR6, np.float32)
    init("twoC6", twoC6, np.float32)
    n("Less", ["twoR6", "t"], "qrB")                             # [1,1,SRC,1] bool
    n("Less", ["twoC6", "t"], "qcB")                             # [1,1,1,SRC] bool
    n("And", ["nonbg_b", "qrB"], "sp1")
    n("And", ["sp1", "qcB"], "sprite_b")                         # [1,1,SRC,SRC] bool
    n("Cast", ["sprite_b"], "sprite", to=TensorProto.UINT8)      # [1,1,SRC,SRC] u8

    # --- mirror via two data-dependent Gathers: idx[i]=clip(min(i,t-i),0,SRC-1)
    ramp = np.arange(WORK, dtype=np.float32).reshape(WORK)
    init("ramp", ramp, np.float32)
    init("shp1", np.array([1], np.int64), np.int64)
    n("Reshape", ["t", "shp1"], "t1")                            # [1] scalar
    n("Sub", ["t1", "ramp"], "tmi")                              # t - i, [WORK]
    n("Less", ["ramp", "tmi"], "use_i")                          # i < t-i ?
    n("Where", ["use_i", "ramp", "tmi"], "fmin")                 # min(i, t-i)
    init("zf", np.array(0.0, np.float32), np.float32)
    init("hi", np.array(float(SRC - 1), np.float32), np.float32)
    n("Max", ["fmin", "zf"], "fmin0")                            # clip low (fp32 ok)
    n("Min", ["fmin0", "hi"], "fidx")                            # clip high
    n("Cast", ["fidx"], "gidx", to=TensorProto.INT64)            # [WORK] int64
    n("Gather", ["sprite", "gidx"], "g_row", axis=2)             # [1,1,WORK,SRC] u8
    n("Gather", ["g_row", "gidx"], "outm_u8", axis=3)            # [1,1,WORK,WORK] u8
    init("u8_0", np.array(0, np.uint8), np.uint8)
    n("Greater", ["outm_u8", "u8_0"], "outmB")                   # bool [1,1,WORK,WORK]

    # --- gridmask (row<m & col<m), WORK canvas ---
    Ridx = np.arange(WORK).reshape(1, 1, WORK, 1).astype(np.float32)
    Cidx = np.arange(WORK).reshape(1, 1, 1, WORK).astype(np.float32)
    init("Ridx", Ridx, np.float32)
    init("Cidx", Cidx, np.float32)
    n("Less", ["Ridx", "m"], "grB")                              # [1,1,WORK,1] bool
    n("Less", ["Cidx", "m"], "gcB")                              # [1,1,1,WORK] bool
    n("And", ["grB", "gcB"], "gridB")                            # [1,1,WORK,WORK]
    # mirror only counts inside the 2s x 2s output region
    n("And", ["outmB", "gridB"], "mirrB")                        # [1,1,WORK,WORK]

    # --- uint8 label map L: linecolor on mirror, 0 in-grid bg, 10 outside ---
    init("v0", np.array(0, np.uint8), np.uint8)
    init("v10", np.array(10, np.uint8), np.uint8)
    n("Where", ["gridB", "v0", "v10"], "L0")                     # 0 in-grid else 10
    n("Where", ["mirrB", "Lid_u8", "L0"], "Lwk")                 # mirror -> linecolor

    # pad L back to 30x30 (sentinel 10), then final Equal
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    n("Pad", ["Lwk", "padpads", "padval"], "L", mode="constant")  # [1,1,30,30] u8

    init("chan10", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan10"], "output")                        # -> free BOOL

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task109", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
