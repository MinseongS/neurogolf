"""Task 361 (e40b9e2f): complete a 4-fold rotational "pinwheel".

Rule (from ARC-GEN generator, verified 20000/20000 fresh):
  The input is a partial pinwheel -- a subset of a figure with 4-fold (90 deg)
  rotational symmetry about a centre.  The OUTPUT always places the full orbit:

      output = input  U  rot90 U rot180 U rot270   (about that centre).

  In integer terms the 90 deg forward rotation is  (Y,X)->(s1+X, s2-Y) with
  s1=R-C+b, s2=R+C.  The centre is recovered EXACTLY by the (s1,s2) maximising
  the number of input pixels whose 2-step orbit is in-grid with matching colour
  (a 2-step AND-chain argmax is exact).

Two-window compaction (keeps it under the public floor):
  Everything sits near the input's bbox centre.
   * SEARCH window  7x7 (radius 3, top-left SY=floor(bbox_y_mid)-3): contains every
     occupied input cell.  In this window the candidate centres collapse to a FIXED
     5x5=25 set (s1_w in [-2..2], s2_w in [4..8]); the orbit permutations are thus
     constant tables ORB[2,25,49] -> the search is two constant Gathers + Equal.
   * COMPLETION window 9x9 (radius 4, top-left SY-1): contains the full symmetric
     output.  Done by inverse-rotation ARITHMETIC on the SELECTED centre (no table),
     keeping the first non-zero of the three rotated source colours.
  Scatter the 81 completed colours back to the 10x10 corner, Pad to 30x30
  (sentinel 10), final Equal(L, arange) -> free BOOL output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

W = 10
SW = 7;  SNW = SW * SW          # search window 49
CW = 9;  CNW = CW * CW          # completion window 81
# The (s1_w, s2_w) centres that ACTUALLY occur in this 7x7 search window collapse
# to these 13 pairs (verified 40000/40000 fresh).  Each is a true pinwheel centre,
# so the argmax over just these 13 always contains -- and selects -- the correct one.
PAIRS = [(-2, 6), (-1, 5), (-1, 6), (-1, 7), (0, 4), (0, 5), (0, 6),
         (0, 7), (0, 8), (1, 5), (1, 6), (1, 7), (2, 6)]


def _orb():
    swy, swx = np.meshgrid(np.arange(SW), np.arange(SW), indexing="ij")
    swyf = swy.reshape(-1); swxf = swx.reshape(-1)
    s1c = np.array([p[0] for p in PAIRS]); s2c = np.array([p[1] for p in PAIRS])
    C = len(PAIRS)
    ORB = np.full((2, C, SNW), SNW, np.int64)
    for c in range(C):
        s1, s2 = s1c[c], s2c[c]; cy = swyf.copy(); cx = swxf.copy()
        for k in range(2):
            ny = s1 + cx; nx = s2 - cy
            ib = (ny >= 0) & (ny < SW) & (nx >= 0) & (nx < SW)
            ORB[k, c] = np.where(ib, ny * SW + nx, SNW); cy, cx = ny, nx
    # completion source tables on the 9x9 window (centre s1_c=s1, s2_c=s2+2)
    cwy, cwx = np.meshgrid(np.arange(CW), np.arange(CW), indexing="ij")
    cwyf = cwy.reshape(-1); cwxf = cwx.reshape(-1)
    SRC = np.full((3, C, CNW), CNW, np.int64)
    for c in range(C):
        s1, s2 = s1c[c], s2c[c] + 2; Y = cwyf.copy(); Xc = cwxf.copy()
        for k in range(3):
            nY = s2 - Xc; nX = Y - s1; Y, Xc = nY, nX
            ib = (Y >= 0) & (Y < CW) & (Xc >= 0) & (Xc < CW)
            SRC[k, c] = np.where(ib, Y * CW + Xc, CNW)
    return ORB, SRC, s1c, s2c, swyf, swxf, C


def build(task):
    ORB, SRC, s1c, s2c, swyf, swxf, C = _orb()
    cwy, cwx = np.meshgrid(np.arange(CW), np.arange(CW), indexing="ij")
    cwyf = cwy.reshape(-1).astype(np.float32); cwxf = cwx.reshape(-1).astype(np.float32)

    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name)); return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs)); return out

    # constants
    init("kw", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    init("crop_ax", np.array([2, 3], np.int64), np.int64)
    init("shp_n", np.array([W * W], np.int64), np.int64)
    init("shp_grid", np.array([1, 1, W, W], np.int64), np.int64)
    init("shp_1", np.array([1], np.int64), np.int64)
    init("u0r", np.array([0], np.uint8), np.uint8)
    init("u0", np.array(0, np.uint8), np.uint8)
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64), np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    init("redax1", np.array([1], np.int64), np.int64)

    init("ar_full", np.arange(W, dtype=np.float32).reshape(1, 1, 1, W), np.float32)
    init("ar_fullR", np.arange(W, dtype=np.float32).reshape(1, 1, W, 1), np.float32)
    init("half", np.array(0.5, np.float32), np.float32)
    init("negbig", np.array(-1.0, np.float32), np.float32)
    init("posbig", np.array(99.0, np.float32), np.float32)
    init("three", np.array(3.0, np.float32), np.float32)
    init("two", np.array(2.0, np.float32), np.float32)
    init("zerof", np.array(0.0, np.float32), np.float32)

    init("swyh", swyf.astype(np.float16).reshape(1, SNW), np.float16)   # [1,49] fp16
    init("swxh", swxf.astype(np.float16).reshape(1, SNW), np.float16)
    init("SNfh", np.array(float(SNW), np.float16), np.float16)
    init("u255s", np.full(SNW, 255, np.uint8), np.uint8)
    init("u0vecS", np.zeros(SNW, np.uint8), np.uint8)

    # fp16 completion-window coords for the index math (fp16-exact small ints)
    init("cwyh", cwyf.reshape(1, CNW), np.float16)
    init("cwxh", cwxf.reshape(1, CNW), np.float16)
    init("zeroh", np.array(0.0, np.float16), np.float16)

    init("ORB0", ORB[0], np.int32)                                     # [13,49]
    init("ORB1", ORB[1], np.int32)

    # ---- V = colour index (Conv full 30x30 -> slice corner -> uint8 -> flat) ----
    # the 30x30 Conv output (3600 B) is the cheapest colour-index path: Conv fuses
    # the 10-channel reduction without materialising a per-channel corner slice.
    n("Conv", ["input", "kw"], "Vbig")                                 # [1,1,30,30] f32
    init("crop_st", np.array([0, 0], np.int64), np.int64)
    init("crop_en", np.array([W, W], np.int64), np.int64)
    n("Slice", ["Vbig", "crop_st", "crop_en", "crop_ax"], "Vc")        # [1,1,10,10]
    n("Cast", ["Vc"], "Vu", to=TensorProto.UINT8)
    n("Reshape", ["Vu", "shp_n"], "Vn")                                # [100]
    n("Concat", ["Vn", "u0r"], "Vp", axis=0)                           # [101]

    # ---- bbox extremes -> window top-left scalars ----
    n("Greater", ["Vc", "zerof"], "occ2d")
    n("Cast", ["occ2d"], "occf2d", to=TensorProto.FLOAT)
    n("ReduceMax", ["occf2d"], "rowocc", axes=[3], keepdims=1)         # [1,1,10,1]
    n("ReduceMax", ["occf2d"], "colocc", axes=[2], keepdims=1)         # [1,1,1,10]
    n("Greater", ["rowocc", "half"], "rb"); n("Greater", ["colocc", "half"], "cb")
    n("Where", ["rb", "ar_fullR", "posbig"], "rmin_i"); n("ReduceMin", ["rmin_i"], "ymin", keepdims=0)
    n("Where", ["rb", "ar_fullR", "negbig"], "rmax_i"); n("ReduceMax", ["rmax_i"], "ymax", keepdims=0)
    n("Where", ["cb", "ar_full", "posbig"], "cmin_i"); n("ReduceMin", ["cmin_i"], "xmin", keepdims=0)
    n("Where", ["cb", "ar_full", "negbig"], "cmax_i"); n("ReduceMax", ["cmax_i"], "xmax", keepdims=0)
    # SY = floor((ymin+ymax)/2) - 3   ;  CY = SY-1 (completion window)
    n("Add", ["ymin", "ymax"], "ysum"); n("Div", ["ysum", "two"], "ymid")
    n("Floor", ["ymid"], "ymidf"); n("Sub", ["ymidf", "three"], "SY")
    n("Add", ["xmin", "xmax"], "xsum"); n("Div", ["xsum", "two"], "xmid")
    n("Floor", ["xmid"], "xmidf"); n("Sub", ["xmidf", "three"], "SX")

    # ---- SEARCH: gather 49 search-window colours (fp16 index math) ----
    n("Cast", ["SY"], "SYh", to=TensorProto.FLOAT16); n("Cast", ["SX"], "SXh", to=TensorProto.FLOAT16)
    n("Add", ["swyh", "SYh"], "say"); n("Add", ["swxh", "SXh"], "sax")   # [1,49] fp16
    g0 = n("GreaterOrEqual", ["say", "zeroh"], "sy0"); g1 = n("LessOrEqual", ["say", "Hm1h"], "sy1")
    g2 = n("GreaterOrEqual", ["sax", "zeroh"], "sx0"); g3 = n("LessOrEqual", ["sax", "Hm1h"], "sx1")
    sib = n("And", [g0, g1], "sib0"); sib = n("And", [sib, g2], "sib1"); sib = n("And", [sib, g3], "sib")
    n("Mul", ["say", "Wh"], "safy"); n("Add", ["safy", "sax"], "saf0")
    n("Where", ["sib", "saf0", "SNfh"], "saf")                          # OOB -> 49
    n("Cast", ["saf"], "saidx", to=TensorProto.INT32)
    n("Reshape", ["saidx", "shp_s"], "saidxv"); init("shp_s", np.array([SNW], np.int64), np.int64)
    # gather from a 50-slot search buffer (V window + bg sentinel)
    n("Gather", ["Vp", "saidxv"], "swin", axis=0)                      # [49] u8
    n("Concat", ["swin", "u0r"], "swinp", axis=0)                      # [50]
    n("Equal", ["swin", "u0vecS"], "swz")
    n("Where", ["swz", "u255s", "swin"], "sbasew")                     # [49] u8 (occ-fold)
    n("Reshape", ["sbasew", "shp_1s"], "base"); init("shp_1s", np.array([1, SNW], np.int64), np.int64)

    n("Gather", ["swinp", "ORB0"], "col0", axis=0)                     # [25,49] u8
    n("Gather", ["swinp", "ORB1"], "col1", axis=0)
    n("Equal", ["col0", "base"], "eq0"); n("Equal", ["col1", "base"], "eq1")
    n("And", ["eq0", "eq1"], "good01")                                 # [25,49] bool
    n("Cast", ["good01"], "goodf", to=TensorProto.FLOAT16)
    n("ReduceSum", ["goodf", "redax1"], "score", keepdims=0)           # [25] fp16
    n("ArgMax", ["score"], "best", axis=0, keepdims=0)
    n("Reshape", ["best", "shp_1"], "best1")

    # ---- COMPLETION: gather 81 completion-window colours ----
    # completion window top-left CY = SY-1, CX = SX-1  (fp16 index math)
    init("oneh", np.array(1.0, np.float16), np.float16)
    init("Wh", np.array(float(W), np.float16), np.float16)
    init("Hm1h", np.array(float(W - 1), np.float16), np.float16)
    init("CNoobh", np.array(float(W * W), np.float16), np.float16)
    n("Sub", ["SYh", "oneh"], "CY"); n("Sub", ["SXh", "oneh"], "CX")
    n("Add", ["cwyh", "CY"], "cay"); n("Add", ["cwxh", "CX"], "cax")
    cg0 = n("GreaterOrEqual", ["cay", "zeroh"], "cy0"); cg1 = n("LessOrEqual", ["cay", "Hm1h"], "cy1")
    cg2 = n("GreaterOrEqual", ["cax", "zeroh"], "cx0"); cg3 = n("LessOrEqual", ["cax", "Hm1h"], "cx1")
    cib = n("And", [cg0, cg1], "cib0"); cib = n("And", [cib, cg2], "cib1"); cib = n("And", [cib, cg3], "cib")
    n("Mul", ["cay", "Wh"], "cafy"); n("Add", ["cafy", "cax"], "caf0")
    n("Where", ["cib", "caf0", "CNoobh"], "caf")
    n("Cast", ["caf"], "caidx", to=TensorProto.INT32)
    n("Reshape", ["caidx", "shp_c"], "caidxv"); init("shp_c", np.array([CNW], np.int64), np.int64)
    n("Gather", ["Vp", "caidxv"], "cwin", axis=0)                      # [81] u8
    n("Concat", ["cwin", "u0r"], "cwinp", axis=0)                      # [82]
    n("Reshape", ["cwin", "shp_1c"], "cwinrow"); init("shp_1c", np.array([1, CNW], np.int64), np.int64)

    # source maps from precomputed SRC tables for the selected candidate
    init("SRC0", SRC[0], np.int32)                                    # [13,81]
    init("SRC1", SRC[1], np.int32)
    init("SRC2", SRC[2], np.int32)
    outcol = "cwinrow"                                                # [1,81] u8
    for k in range(3):
        srck = n("Gather", [f"SRC{k}", "best1"], f"src{k}", axis=0)   # [1,81] int32
        scol = n("Gather", ["cwinp", srck], f"scol{k}", axis=0)       # [1,81] u8
        isbg = n("Equal", [outcol, "u0"], f"isbg{k}")
        outcol = n("Where", [isbg, scol, outcol], f"oc{k}")

    # ---- scatter the 81 completed colours back to the 10x10 corner ----
    n("Reshape", [outcol, "shp_c"], "outv")                            # [81] u8
    init("z101", np.zeros(W * W + 1, np.uint8), np.uint8)
    n("Reshape", ["caidxv", "shp_c1"], "scidx"); init("shp_c1", np.array([CNW, 1], np.int64), np.int64)
    n("Cast", ["scidx"], "scidx64", to=TensorProto.INT64)
    n("ScatterND", ["z101", "scidx64", "outv"], "L101")                # [101] u8
    init("sl0", np.array([0], np.int64), np.int64); init("sl100", np.array([W * W], np.int64), np.int64)
    init("sl_ax", np.array([0], np.int64), np.int64)
    n("Slice", ["L101", "sl0", "sl100", "sl_ax"], "L100")              # [100]
    n("Reshape", ["L100", "shp_grid"], "L10")
    n("Pad", ["L10", "padpads", "padval"], "L", mode="constant")
    n("Equal", ["L", "chan"], "output")

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 13)])
