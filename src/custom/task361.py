"""Task 361 (e40b9e2f): complete a 4-fold rotational "pinwheel".

Rule (from ARC-GEN generator, verified fresh):
  The input holds a partial pinwheel: a set of coloured pixels that are some
  subset of a figure with 4-fold (90 deg) rotational symmetry about a centre.
  Each generator pixel places 1 or all 4 of the rotation orbit; the OUTPUT
  always places all 4 (the full symmetric figure).  So:

      output = input  union  rot90(input) union rot180 union rot270

  about a centre (R + b/2, C - b/2)  (b = bump in {0,1}).  In integer terms the
  90 deg forward rotation is  (Y,X) -> (s1 + X, s2 - Y)  with
      s1 = R - C + b,  s2 = R + C.
  The centre is recovered EXACTLY (verified 2000/2000 fresh) by choosing the
  (s1,s2) that MAXIMISES the number of input pixels whose entire 4-orbit lies
  in-grid with matching colour (the "symmetric core" pins the pivot).
  Observed ranges: s1 in [-4..5], s2 in [6..14]  -> 90 candidate centres.

Encoding:
  Work on the active 10x10 corner, flattened to length 100.
  V  = per-cell colour index (Conv [0..9] -> Slice 10x10 -> Cast uint8 -> flat).
  Search: for each of 90 candidate centres, chain the forward rotation 3 times,
  gather the orbit colours and AND together (in-grid & equal-to-base); ReduceSum
  over the 100 cells -> score[90]; ArgMax -> best (s1,s2).
  Completion: pick best s1,s2 (Gather), chain the INVERSE rotation 3 times to get
  the 3 source-cell index maps [100], gather the source colours, and OR them onto
  V (first non-zero wins).  Reshape to 10x10, Pad to 30x30 with sentinel 10, and a
  single Equal(L, arange) writes straight into the free BOOL output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

W = 10           # active grid side
N = W * W        # 100 flattened cells
S1 = np.arange(-4, 6)     # 10 values
S2 = np.arange(6, 15)     # 9 values


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---------- candidate centre arrays ----------
    s1g, s2g = np.meshgrid(S1, S2, indexing="ij")
    s1c = s1g.reshape(-1).astype(np.int32)         # [90]
    s2c = s2g.reshape(-1).astype(np.int32)         # [90]
    C = s1c.shape[0]

    # flattened cell coords
    yy, xx = np.meshgrid(np.arange(W), np.arange(W), indexing="ij")
    yflat = yy.reshape(-1).astype(np.int32)        # [100]
    xflat = xx.reshape(-1).astype(np.int32)        # [100]

    init("kw", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    init("crop_st", np.array([0, 0], np.int64), np.int64)
    init("crop_en", np.array([W, W], np.int64), np.int64)
    init("crop_ax", np.array([2, 3], np.int64), np.int64)

    init("s1c", s1c.reshape(C, 1), np.int32)       # [90,1] broadcast over cells
    init("s2c", s2c.reshape(C, 1), np.int32)       # [90,1]
    init("yrow", yflat.reshape(1, N), np.int32)    # [1,100]
    init("xrow", xflat.reshape(1, N), np.int32)    # [1,100]
    init("zero32", np.array(0, np.int32), np.int32)
    init("Wm1", np.array(W - 1, np.int32), np.int32)
    init("Wc", np.array(W, np.int32), np.int32)
    init("oob", np.array(N, np.int32), np.int32)   # out-of-bounds gather slot -> bg
    init("u0", np.array(0, np.uint8), np.uint8)
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64),
         np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    init("ax0", np.array([0], np.int64), np.int64)

    # ---------- V = colour-index map, flattened, with a bg sentinel slot ----------
    n("Conv", ["input", "kw"], "Vbig")                          # [1,1,30,30] f32
    n("Slice", ["Vbig", "crop_st", "crop_en", "crop_ax"], "Vc")  # [1,1,10,10]
    n("Cast", ["Vc"], "Vu", to=TensorProto.UINT8)               # [1,1,10,10] u8
    n("Reshape", ["Vu", "shp_n"], "Vn")                         # [100] u8
    init("shp_n", np.array([N], np.int64), np.int64)
    # append a bg(0) slot at index N so clamped/OOB gathers read background
    n("Concat", ["Vn", "u0r"], "Vp", axis=0)                    # [101] u8
    init("u0r", np.array([0], np.uint8), np.uint8)
    # occupancy of each of the 100 real cells (bool)
    n("Equal", ["Vn", "u0vec"], "Vz")                           # [100] bool (bg)
    init("u0vec", np.zeros(N, np.uint8), np.uint8)
    n("Not", ["Vz"], "occ")                                     # [100] bool occupied

    # base colour per cell, broadcast over candidates: [1,100] u8
    n("Reshape", ["Vn", "shp_1n"], "Vrow")                      # [1,100] u8
    init("shp_1n", np.array([1, N], np.int64), np.int64)

    # ================= centre search =================
    # forward rotation chain over candidates; build [90,100] flat gather indices.
    # state cy,cx start as the per-cell coords broadcast to [90,100].
    cy, cx = "yrow", "xrow"      # [1,100] int32 (broadcast with s1c/s2c -> [90,100])
    good = None                  # running bool [90,100]
    for step in range(3):
        # ny = s1 + cx ;  nx = s2 - cy
        ny = n("Add", ["s1c", cx], f"ny{step}")                 # [90,100]
        nx = n("Sub", ["s2c", cy], f"nx{step}")                 # [90,100]
        # in-bounds test
        gy0 = n("GreaterOrEqual", [ny, "zero32"], f"gy0_{step}")
        gy1 = n("LessOrEqual", [ny, "Wm1"], f"gy1_{step}")
        gx0 = n("GreaterOrEqual", [nx, "zero32"], f"gx0_{step}")
        gx1 = n("LessOrEqual", [nx, "Wm1"], f"gx1_{step}")
        inb = n("And", [gy0, gy1], f"inb0_{step}")
        inb = n("And", [inb, gx0], f"inb1_{step}")
        inb = n("And", [inb, gx1], f"inb_{step}")               # [90,100] bool
        # flat index = ny*W + nx, sent OOB slot when not in bounds
        flat = n("Mul", [ny, "Wc"], f"fy{step}")
        flat = n("Add", [flat, nx], f"flat0_{step}")            # [90,100]
        flat = n("Where", [inb, flat, "oobC"], f"flat_{step}")  # OOB -> N
        # clamp into [0,N] for the gather
        flatc = n("Clip", [flat, "zero32", "oob"], f"flatc_{step}")
        idx64 = n("Cast", [flatc], f"idx{step}", to=TensorProto.INT64)
        col = n("Gather", ["Vp", idx64], f"col{step}", axis=0)  # [90,100] u8
        eq = n("Equal", [col, "Vrow"], f"eq{step}")             # vs base [1,100]
        cond = n("And", [eq, inb], f"cond{step}")               # [90,100] bool
        good = cond if good is None else n("And", [good, cond], f"good{step}")
        cy, cx = ny, nx
    init("oobC", np.full((C, 1), N, np.int32), np.int32)        # [90,1] OOB const

    # only count occupied base cells
    n("And", [good, "occ"], "goodocc")                          # [90,100] bool
    n("Cast", ["goodocc"], "goodf", to=TensorProto.INT32)
    init("redax", np.array([1], np.int64), np.int64)
    n("ReduceSum", ["goodf", "redax"], "score", keepdims=0)     # [90] int32
    n("ArgMax", ["score"], "best", axis=0, keepdims=0)          # scalar int64
    n("Reshape", ["best", "shp_1"], "best1")
    init("shp_1", np.array([1], np.int64), np.int64)
    n("Gather", ["s1flat", "best1"], "bs1", axis=0)             # [1] int32
    n("Gather", ["s2flat", "best1"], "bs2", axis=0)             # [1] int32
    init("s1flat", s1c, np.int32)
    init("s2flat", s2c, np.int32)

    # ================= completion =================
    # inverse rotation chain to get 3 source-index maps [100].
    # inverse of fwd (Y,X)->(s1+X, s2-Y) :  given (y,x), pre = (s2 - x, y - s1).
    Y, Xc = "yrow", "xrow"       # [1,100]
    outcol = "Vrow"              # [1,100] u8 running output colour (bg wins later)
    for step in range(3):
        nY = n("Sub", ["bs2", Xc], f"pY{step}")                 # [1,100] int32
        nX = n("Sub", [Y, "bs1"], f"pX{step}")                  # [1,100]
        gy0 = n("GreaterOrEqual", [nY, "zero32"], f"py0_{step}")
        gy1 = n("LessOrEqual", [nY, "Wm1"], f"py1_{step}")
        gx0 = n("GreaterOrEqual", [nX, "zero32"], f"px0_{step}")
        gx1 = n("LessOrEqual", [nX, "Wm1"], f"px1_{step}")
        inb = n("And", [gy0, gy1], f"pinb0_{step}")
        inb = n("And", [inb, gx0], f"pinb1_{step}")
        inb = n("And", [inb, gx1], f"pinb_{step}")              # [1,100] bool
        flat = n("Mul", [nY, "Wc"], f"pfy{step}")
        flat = n("Add", [flat, nX], f"pflat0_{step}")
        flat = n("Where", [inb, flat, "oob1"], f"pflat_{step}")
        flatc = n("Clip", [flat, "zero32", "oob"], f"pflatc_{step}")
        idx64 = n("Cast", [flatc], f"pidx{step}", to=TensorProto.INT64)
        scol = n("Gather", ["Vp", idx64], f"scol{step}", axis=0)  # [1,100] u8
        # first non-zero wins: where current output is bg, take source colour
        isbg = n("Equal", [outcol, "u0"], f"isbg{step}")        # [1,100] bool
        outcol = n("Where", [isbg, scol, outcol], f"out{step}")
        Y, Xc = nY, nX
    init("oob1", np.full((1, 1), N, np.int32), np.int32)

    # reshape to 10x10, pad to 30x30 (sentinel 10), final Equal -> BOOL output
    n("Reshape", [outcol, "shp_grid"], "L10")                   # [1,1,10,10] u8
    init("shp_grid", np.array([1, 1, W, W], np.int64), np.int64)
    n("Pad", ["L10", "padpads", "padval"], "L", mode="constant")  # [1,1,30,30]
    n("Equal", ["L", "chan"], "output")                         # -> free BOOL out

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 13)])
