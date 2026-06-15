"""Task 361 (e40b9e2f): complete a 4-fold rotational "pinwheel".

Rule (from ARC-GEN generator, verified fresh):
  The input holds a partial pinwheel -- a subset of a figure with 4-fold (90 deg)
  rotational symmetry about a centre.  The OUTPUT always places the full orbit:

      output = input  U  rot90(input) U rot180(input) U rot270(input)

  about centre (R + b/2, C - b/2)  (b = bump in {0,1}).  In integer terms the
  90 deg forward rotation is  (Y,X) -> (s1 + X, s2 - Y)  with
      s1 = R - C + b,   s2 = R + C .
  The centre is recovered EXACTLY by choosing the (s1,s2) maximising the number
  of input pixels whose 2-step rotation orbit (rot90 then rot180) is in-grid with
  matching colour (the symmetric core pins the pivot).  A 2-step AND-chain is
  exact -- verified 20000/20000 fresh.  Observed ranges: s1 in [-4..5],
  s2 in [6..14]  -> 90 candidate centres.

Encoding (work on the active 10x10 corner, flattened to 100 cells):
  V   = per-cell colour index (Conv [0..9] -> Slice 10x10 -> Cast uint8 -> flat),
        with a background sentinel appended at flat index 100 so out-of-bounds
        gathers read background (0).
  search : for each of 90 candidates, chain the forward rotation twice, gather
        orbit colours and AND with "equal to base" (OOB auto-fails since bg!=base);
        AND with occupancy; ReduceSum over the 100 cells -> score[90]; ArgMax.
  complete : Gather best (s1,s2); chain the INVERSE rotation 3 times to get the
        three source-cell index maps [100]; gather source colours and keep the
        first non-zero onto V.  Reshape 10x10, Pad to 30x30 (sentinel 10), and a
        single Equal(L, arange) writes straight into the free BOOL output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

W = 10                       # active grid side
N = W * W                    # 100 flattened cells
S1 = np.arange(-4, 6)        # 10 values
S2 = np.arange(6, 15)        # 9 values


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    s1g, s2g = np.meshgrid(S1, S2, indexing="ij")
    s1c = s1g.reshape(-1).astype(np.float32)         # [90]
    s2c = s2g.reshape(-1).astype(np.float32)
    C = s1c.shape[0]
    yy, xx = np.meshgrid(np.arange(W), np.arange(W), indexing="ij")
    yflat = yy.reshape(-1).astype(np.float32)        # [100]
    xflat = xx.reshape(-1).astype(np.float32)

    # constants
    init("kw", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    init("crop_st", np.array([0, 0], np.int64), np.int64)
    init("crop_en", np.array([W, W], np.int64), np.int64)
    init("crop_ax", np.array([2, 3], np.int64), np.int64)
    init("shp_n", np.array([N], np.int64), np.int64)
    init("shp_1n", np.array([1, N], np.int64), np.int64)
    init("shp_grid", np.array([1, 1, W, W], np.int64), np.int64)
    init("shp_1", np.array([1], np.int64), np.int64)

    init("s1c", s1c.reshape(C, 1), np.float32)       # [90,1]
    init("s2c", s2c.reshape(C, 1), np.float32)       # [90,1]
    init("yrow", yflat.reshape(1, N), np.float32)    # [1,100]
    init("xrow", xflat.reshape(1, N), np.float32)
    init("zerof", np.array(0.0, np.float32), np.float32)
    init("Wm1", np.array(float(W - 1), np.float32), np.float32)
    init("Wc", np.array(float(W), np.float32), np.float32)
    init("Nf", np.array(float(N), np.float32), np.float32)         # OOB flat slot
    init("u0", np.array(0, np.uint8), np.uint8)
    init("u0r", np.array([0], np.uint8), np.uint8)
    init("u0vec", np.zeros(N, np.uint8), np.uint8)
    init("redax", np.array([1], np.int64), np.int64)
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64),
         np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    init("s1flat", s1c, np.float32)                  # [90]
    init("s2flat", s2c, np.float32)

    # ---------- V = colour-index map, flattened (+ bg sentinel slot) ----------
    n("Conv", ["input", "kw"], "Vbig")                          # [1,1,30,30]
    n("Slice", ["Vbig", "crop_st", "crop_en", "crop_ax"], "Vc")  # [1,1,10,10]
    n("Cast", ["Vc"], "Vu", to=TensorProto.UINT8)
    n("Reshape", ["Vu", "shp_n"], "Vn")                         # [100] u8
    n("Concat", ["Vn", "u0r"], "Vp", axis=0)                    # [101] u8
    n("Equal", ["Vn", "u0vec"], "Vz")                           # [100] bool bg
    n("Not", ["Vz"], "occ")                                     # [100] bool occupied
    n("Reshape", ["Vn", "shp_1n"], "Vrow")                      # [1,100] u8 base

    # ================= centre search (2-step AND-chain) =================
    cy, cx = "yrow", "xrow"                # [1,100] f32 broadcast with [90,1]
    good = None
    for step in range(2):
        ny = n("Add", ["s1c", cx], f"ny{step}")                 # [90,100] f32
        nx = n("Sub", ["s2c", cy], f"nx{step}")
        gy0 = n("GreaterOrEqual", [ny, "zerof"], f"gy0_{step}")
        gy1 = n("LessOrEqual", [ny, "Wm1"], f"gy1_{step}")
        gx0 = n("GreaterOrEqual", [nx, "zerof"], f"gx0_{step}")
        gx1 = n("LessOrEqual", [nx, "Wm1"], f"gx1_{step}")
        inb = n("And", [gy0, gy1], f"inb0_{step}")
        inb = n("And", [inb, gx0], f"inb1_{step}")
        inb = n("And", [inb, gx1], f"inb_{step}")               # [90,100] bool
        flat = n("Mul", [ny, "Wc"], f"fy{step}")
        flat = n("Add", [flat, nx], f"flat0_{step}")
        flat = n("Where", [inb, flat, "Nf"], f"flat_{step}")    # OOB -> 100
        idx = n("Cast", [flat], f"idx{step}", to=TensorProto.INT32)
        col = n("Gather", ["Vp", idx], f"col{step}", axis=0)    # [90,100] u8
        eq = n("Equal", [col, "Vrow"], f"eq{step}")             # vs base
        good = eq if good is None else n("And", [good, eq], f"good{step}")
        cy, cx = ny, nx
    n("And", [good, "occ"], "goodocc")                          # [90,100] bool
    n("Cast", ["goodocc"], "goodf", to=TensorProto.FLOAT)       # f32 for reduce
    n("ReduceSum", ["goodf", "redax"], "score", keepdims=0)     # [90]
    n("ArgMax", ["score"], "best", axis=0, keepdims=0)          # scalar
    n("Reshape", ["best", "shp_1"], "best1")
    n("Gather", ["s1flat", "best1"], "bs1", axis=0)             # [1] f32
    n("Gather", ["s2flat", "best1"], "bs2", axis=0)             # [1] f32

    # ================= completion (inverse rotation, 3 sources) =================
    # inverse of fwd (Y,X)->(s1+X, s2-Y):  pre(y,x) = (s2 - x, y - s1).
    Y, Xc = "yrow", "xrow"                 # [1,100] f32
    outcol = "Vrow"                        # [1,100] u8
    for step in range(3):
        nY = n("Sub", ["bs2", Xc], f"pY{step}")                 # [1,100] f32
        nX = n("Sub", [Y, "bs1"], f"pX{step}")
        gy0 = n("GreaterOrEqual", [nY, "zerof"], f"py0_{step}")
        gy1 = n("LessOrEqual", [nY, "Wm1"], f"py1_{step}")
        gx0 = n("GreaterOrEqual", [nX, "zerof"], f"px0_{step}")
        gx1 = n("LessOrEqual", [nX, "Wm1"], f"px1_{step}")
        inb = n("And", [gy0, gy1], f"pinb0_{step}")
        inb = n("And", [inb, gx0], f"pinb1_{step}")
        inb = n("And", [inb, gx1], f"pinb_{step}")
        flat = n("Mul", [nY, "Wc"], f"pfy{step}")
        flat = n("Add", [flat, nX], f"pflat0_{step}")
        flat = n("Where", [inb, flat, "Nf"], f"pflat_{step}")
        idx = n("Cast", [flat], f"pidx{step}", to=TensorProto.INT32)
        scol = n("Gather", ["Vp", idx], f"scol{step}", axis=0)  # [1,100] u8
        isbg = n("Equal", [outcol, "u0"], f"isbg{step}")        # first non-zero wins
        outcol = n("Where", [isbg, scol, outcol], f"out{step}")
        Y, Xc = nY, nX

    n("Reshape", [outcol, "shp_grid"], "L10")                   # [1,1,10,10] u8
    n("Pad", ["L10", "padpads", "padval"], "L", mode="constant")
    n("Equal", ["L", "chan"], "output")                         # -> free BOOL out

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 13)])
