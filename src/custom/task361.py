"""Task 361 (e40b9e2f): complete a 4-fold rotational "pinwheel".

Rule (from ARC-GEN generator, verified 30000/30000 fresh):
  The input holds a partial pinwheel -- a subset of a figure with 4-fold (90 deg)
  rotational symmetry about a centre.  The OUTPUT always places the full orbit:

      output = input  U  rot90(input) U rot180(input) U rot270(input)

  about centre (R + b/2, C - b/2)  (b = bump).  In integer terms the 90 deg
  forward rotation is  (Y,X) -> (s1 + X, s2 - Y)  with s1 = R-C+b, s2 = R+C.
  The centre is recovered EXACTLY by choosing the (s1,s2) maximising the number
  of input pixels whose 2-step rotation orbit is in-grid with matching colour
  (a 2-step AND-chain argmax is exact).

Compaction -- window-relative coords (the key to fitting under the floor):
  The whole figure plus its completion lives within radius 4 of the input's
  bounding-box centre, and in that 9x9 WINDOW the candidate centres collapse to a
  FIXED 5x5 = 25-candidate set (s1_w in [-2..2], s2_w in [6..10]).  So the orbit
  permutations inside the window are *constants*, precomputed as integer index
  tables ORB[2,25,81] (search) and SRC[3,25,81] (completion).  No per-candidate
  index arithmetic at run time.

Graph:
  V    : colour-index of the full 10x10 corner, flattened to 100 (+bg sentinel).
  WY,WX: window top-left = floor(bbox-centre) - 4   (from 1-D occupancy maxima).
  win  : gather the 81 window colours from V at absolute indices (WY+y)*10+WX+x
         (out-of-grid -> bg).  base = win, occ = win>0.
  search : ORB-gather window colours, AND "==base" over 2 steps, AND occ,
           ReduceSum over 81 -> score[25]; ArgMax -> best candidate.
  complete : gather SRC[k, best] (3 source maps), keep first non-zero onto win.
  Scatter the 81 completed colours back to the 10x10 corner (ScatterND), Pad to
  30x30 (sentinel 10), final Equal(L, arange) -> free BOOL output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

W = 10
N = W * W                       # 100
WIN = 9
NW = WIN * WIN                  # 81
S1W = np.arange(-2, 3)          # 5
S2W = np.arange(6, 11)          # 5


def _tables():
    yw, xw = np.meshgrid(np.arange(WIN), np.arange(WIN), indexing="ij")
    ywf = yw.reshape(-1); xwf = xw.reshape(-1)
    s1g, s2g = np.meshgrid(S1W, S2W, indexing="ij")
    s1c = s1g.reshape(-1); s2c = s2g.reshape(-1); C = len(s1c)
    ORB = np.full((2, C, NW), NW, np.int64)
    for c in range(C):
        s1, s2 = s1c[c], s2c[c]; cy = ywf.copy(); cx = xwf.copy()
        for k in range(2):
            ny = s1 + cx; nx = s2 - cy
            inb = (ny >= 0) & (ny < WIN) & (nx >= 0) & (nx < WIN)
            ORB[k, c] = np.where(inb, ny * WIN + nx, NW); cy, cx = ny, nx
    SRC = np.full((3, C, NW), NW, np.int64)
    for c in range(C):
        s1, s2 = s1c[c], s2c[c]; Y = ywf.copy(); Xc = xwf.copy()
        for k in range(3):
            nY = s2 - Xc; nX = Y - s1; Y, Xc = nY, nX
            inb = (Y >= 0) & (Y < WIN) & (Xc >= 0) & (Xc < WIN)
            SRC[k, c] = np.where(inb, Y * WIN + Xc, NW)
    return ORB, SRC, ywf, xwf, C


def build(task):
    ORB, SRC, ywf, xwf, C = _tables()
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- constants ----
    init("kw", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    init("crop_st", np.array([0, 0], np.int64), np.int64)
    init("crop_en", np.array([W, W], np.int64), np.int64)
    init("crop_ax", np.array([2, 3], np.int64), np.int64)
    init("shp_n", np.array([N], np.int64), np.int64)
    init("shp_grid", np.array([1, 1, W, W], np.int64), np.int64)
    init("shp_1", np.array([1], np.int64), np.int64)
    init("u0r", np.array([0], np.uint8), np.uint8)
    init("u0", np.array(0, np.uint8), np.uint8)
    init("u0vecW", np.zeros(NW, np.uint8), np.uint8)
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64),
         np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    init("redax1", np.array([1], np.int64), np.int64)

    # occupancy row/col maxima -> bbox extremes -> window top-left
    init("ar_full", np.arange(W, dtype=np.float32).reshape(1, 1, 1, W), np.float32)
    init("ar_fullR", np.arange(W, dtype=np.float32).reshape(1, 1, W, 1), np.float32)
    init("half", np.array(0.5, np.float32), np.float32)
    init("negbig", np.array(-1.0, np.float32), np.float32)
    init("posbig", np.array(99.0, np.float32), np.float32)
    init("four", np.array(4.0, np.float32), np.float32)
    init("Wf", np.array(float(W), np.float32), np.float32)
    init("two", np.array(2.0, np.float32), np.float32)

    # window-relative coord vectors
    init("ywf", ywf.astype(np.float32).reshape(1, NW), np.float32)   # [1,81]
    init("xwf", xwf.astype(np.float32).reshape(1, NW), np.float32)
    init("zerof", np.array(0.0, np.float32), np.float32)
    init("Hm1", np.array(float(W - 1), np.float32), np.float32)
    init("Nf", np.array(float(N), np.float32), np.float32)

    # index tables (int32 for Gather). element COUNT is what the scorer charges.
    init("ORB0", ORB[0], np.int32)   # [25,81]
    init("ORB1", ORB[1], np.int32)
    # window-relative candidate (s1_w, s2_w) per candidate, for arithmetic completion
    s1g, s2g = np.meshgrid(S1W, S2W, indexing="ij")
    init("s1wc", s1g.reshape(-1).astype(np.float32), np.float32)   # [25]
    init("s2wc", s2g.reshape(-1).astype(np.float32), np.float32)

    # ---- V = colour index, flattened, + bg sentinel ----
    # slice the 10-channel input to the active 10x10 corner BEFORE the Conv so the
    # colour-index plane is [1,1,10,10] (400 B) not [1,1,30,30] (3600 B).
    init("in_st", np.array([0, 0], np.int64), np.int64)
    init("in_en", np.array([W, W], np.int64), np.int64)
    n("Slice", ["input", "in_st", "in_en", "crop_ax"], "incorner")  # [1,10,10,10]
    n("Conv", ["incorner", "kw"], "Vc")                             # [1,1,10,10]
    n("Cast", ["Vc"], "Vu", to=TensorProto.UINT8)
    n("Reshape", ["Vu", "shp_n"], "Vn")                             # [100]
    n("Concat", ["Vn", "u0r"], "Vp", axis=0)                        # [101]

    # ---- bbox extremes from occupancy (float) ----
    n("Greater", ["Vc", "zerof"], "occ2d")                          # [1,1,10,10] bool
    n("Cast", ["occ2d"], "occf2d", to=TensorProto.FLOAT)
    n("ReduceMax", ["occf2d"], "rowocc", axes=[3], keepdims=1)       # [1,1,10,1]
    n("ReduceMax", ["occf2d"], "colocc", axes=[2], keepdims=1)       # [1,1,1,10]
    n("Greater", ["rowocc", "half"], "rb")                          # [1,1,10,1]
    n("Greater", ["colocc", "half"], "cb")                          # [1,1,1,10]
    # ymin = min row idx occupied, ymax = max row idx occupied
    n("Where", ["rb", "ar_fullR", "posbig"], "rmin_i"); n("ReduceMin", ["rmin_i"], "ymin", keepdims=0)
    n("Where", ["rb", "ar_fullR", "negbig"], "rmax_i"); n("ReduceMax", ["rmax_i"], "ymax", keepdims=0)
    n("Where", ["cb", "ar_full", "posbig"], "cmin_i"); n("ReduceMin", ["cmin_i"], "xmin", keepdims=0)
    n("Where", ["cb", "ar_full", "negbig"], "cmax_i"); n("ReduceMax", ["cmax_i"], "xmax", keepdims=0)
    # WY = floor((ymin+ymax)/2) - 4   (integer arithmetic via float, exact small ints)
    n("Add", ["ymin", "ymax"], "ysum"); n("Div", ["ysum", "two"], "ymid")
    n("Floor", ["ymid"], "ymidf"); n("Sub", ["ymidf", "four"], "WY")   # scalar f32
    n("Add", ["xmin", "xmax"], "xsum"); n("Div", ["xsum", "two"], "xmid")
    n("Floor", ["xmid"], "xmidf"); n("Sub", ["xmidf", "four"], "WX")

    # ---- gather 81 window colours from V ----
    n("Add", ["ywf", "WY"], "ay")                                   # [1,81] f32
    n("Add", ["xwf", "WX"], "ax")
    gy0 = n("GreaterOrEqual", ["ay", "zerof"], "wy0")
    gy1 = n("LessOrEqual", ["ay", "Hm1"], "wy1")
    gx0 = n("GreaterOrEqual", ["ax", "zerof"], "wx0")
    gx1 = n("LessOrEqual", ["ax", "Hm1"], "wx1")
    wib = n("And", [gy0, gy1], "wib0"); wib = n("And", [wib, gx0], "wib1")
    wib = n("And", [wib, gx1], "wib")                               # [1,81] bool
    n("Mul", ["ay", "Wf"], "afy"); n("Add", ["afy", "ax"], "aflat0")
    n("Where", ["wib", "aflat0", "Nf"], "aflat")                    # OOB -> 100
    n("Cast", ["aflat"], "aidx", to=TensorProto.INT32)             # [1,81]
    n("Reshape", ["aidx", "shp_w"], "aidxv")                       # [81]
    init("shp_w", np.array([NW], np.int64), np.int64)
    n("Gather", ["Vp", "aidxv"], "win", axis=0)                    # [81] u8
    n("Concat", ["win", "u0r"], "winp", axis=0)                    # [82]
    n("Reshape", ["win", "shp_1w"], "base")                        # [1,81] u8
    init("shp_1w", np.array([1, NW], np.int64), np.int64)
    n("Equal", ["win", "u0vecW"], "winz"); n("Not", ["winz"], "occw")  # [81] bool
    n("Reshape", ["occw", "shp_1w"], "occwR")                      # [1,81]

    # ---- search (constant orbit tables) ----
    n("Gather", ["winp", "ORB0"], "col0", axis=0)                  # [25,81] u8
    n("Gather", ["winp", "ORB1"], "col1", axis=0)
    n("Equal", ["col0", "base"], "eq0")                            # [25,81] bool
    n("Equal", ["col1", "base"], "eq1")
    n("And", ["eq0", "eq1"], "good01")
    n("And", ["good01", "occwR"], "goodocc")                       # [25,81]
    n("Cast", ["goodocc"], "goodf", to=TensorProto.FLOAT16)        # fp16: 81<2048 exact
    n("ReduceSum", ["goodf", "redax1"], "score", keepdims=0)       # [25] fp16
    n("ArgMax", ["score"], "best", axis=0, keepdims=0)
    n("Reshape", ["best", "shp_1"], "best1")                       # [1]

    # ---- completion: inverse rotation arithmetic on the selected window centre ----
    n("Gather", ["s1wc", "best1"], "bs1", axis=0)                  # [1] f32
    n("Gather", ["s2wc", "best1"], "bs2", axis=0)
    # inverse of fwd (Y,X)->(s1+X, s2-Y): pre(y,x)=(s2-x, y-s1), in window coords.
    Y, Xc = "ywf", "xwf"                                           # [1,81] f32
    outcol = "base"                                                # [1,81] u8
    for k in range(3):
        nY = n("Sub", ["bs2", Xc], f"pY{k}")                       # [1,81] f32
        nX = n("Sub", [Y, "bs1"], f"pX{k}")
        gy0 = n("GreaterOrEqual", [nY, "zerof"], f"py0_{k}")
        gy1 = n("LessOrEqual", [nY, "Wm1w"], f"py1_{k}")
        gx0 = n("GreaterOrEqual", [nX, "zerof"], f"px0_{k}")
        gx1 = n("LessOrEqual", [nX, "Wm1w"], f"px1_{k}")
        inb = n("And", [gy0, gy1], f"pinb0_{k}"); inb = n("And", [inb, gx0], f"pinb1_{k}")
        inb = n("And", [inb, gx1], f"pinb_{k}")
        flat = n("Mul", [nY, "WINf"], f"pfy{k}"); flat = n("Add", [flat, nX], f"pf0_{k}")
        flat = n("Where", [inb, flat, "NWf"], f"pflat_{k}")
        idx = n("Cast", [flat], f"pidx{k}", to=TensorProto.INT32)  # [1,81]
        scol = n("Gather", ["winp", idx], f"scol{k}", axis=0)      # [1,81] u8
        isbg = n("Equal", [outcol, "u0"], f"isbg{k}")
        outcol = n("Where", [isbg, scol, outcol], f"oc{k}")
        Y, Xc = nY, nX
    init("Wm1w", np.array(float(WIN - 1), np.float32), np.float32)
    init("WINf", np.array(float(WIN), np.float32), np.float32)
    init("NWf", np.array(float(NW), np.float32), np.float32)

    # ---- scatter the 81 completed colours back to the 10x10 corner ----
    # L100[aidx] = outcol (only in-bounds entries; OOB aidx==100 dropped via clamp+mask)
    n("Reshape", [outcol, "shp_w"], "outv")                        # [81] u8
    # scatter indices: use aidxv but set OOB (==100) to a dummy slot in a 101 buffer
    init("z101", np.zeros(N + 1, np.uint8), np.uint8)
    n("Reshape", ["aidxv", "shp_w1"], "scidx")                     # [81,1] int64? need int64
    init("shp_w1", np.array([NW, 1], np.int64), np.int64)
    n("Cast", ["scidx"], "scidx64", to=TensorProto.INT64)
    n("ScatterND", ["z101", "scidx64", "outv"], "L101")            # [101] u8
    n("Slice", ["L101", "sl0", "sl100", "sl_ax"], "L100")          # [100]
    init("sl0", np.array([0], np.int64), np.int64)
    init("sl100", np.array([N], np.int64), np.int64)
    init("sl_ax", np.array([0], np.int64), np.int64)
    n("Reshape", ["L100", "shp_grid"], "L10")                      # [1,1,10,10] u8
    n("Pad", ["L10", "padpads", "padval"], "L", mode="constant")
    n("Equal", ["L", "chan"], "output")

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 13)])
