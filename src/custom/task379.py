"""task379 (ARC-AGI ecdecbb3) — "red dots shoot rays to cyan lines, stamp 3x3 boxes".

Rule (from the generator; canonical = horizontal-line orientation, xpose flips it):
  * 1-2 full-width horizontal CYAN(8) lines; each column has AT MOST ONE RED(2)
    dot (verified).  Each dot paints a red ray toward the NEAREST line above and
    NEAREST line below it (a nearer line blocks a farther one).  Where a ray
    reaches a line at (L,c) a 3x3 CYAN box is stamped centred there with a RED
    centre.  Priority: centre-red > box-cyan > ray-red > line-cyan > bg.

Closed-form, PROFILE-based (verified 0 bad / 3000 fresh, 0/266 stored):
  NO full red/cyan plane is ever materialised — everything is recovered as 1-D
  profiles via two no-pad Convs that collapse one spatial axis, each packing TWO
  profiles in disjoint magnitude bands (val = (dot_pos+1) + 64*cyan_count, all
  ints < 2048 so fp16/fp32-exact):
    cH = Conv collapse-height -> [1,1,1,30] = (dot_row+1) + 64*cyan_per_col
    cW = Conv collapse-width  -> [1,1,30,1] = (dot_col+1) + 64*cyan_per_row
  Decode: cyan_count = floor(val/64) ; dot_pos = val-64*cyan_count-1 ; presence
  = (val mod 64) > 0.  Canonical (lines=rows) profiles are picked from the two
  convs by a scalar orientation flag horB; only the final colour plane is
  transposed back.
  Line rows Lmin/Lmax (<=2 lines); per column Ldown/Lup = nearest line below/
  above the dot.  Ray = rlo[c] <= row <= rhi[c] (one full plane), where rlo/rhi
  collapse the up/down ray spans and auto-include the dot.
  Box (cyan ring) is built SEPARABLY per line: box rows = {L-1, L+1} ONLY (the
  line row itself stays cyan from lineB), AND'd with the ±1-column-widened
  "reaches this line" column profile.  Because the box never covers row L, the
  priority line<ray<box reproduces the RED box centre for free (the ray reds
  (L, dotcol)) — no separate centre layer.
  Compose = Where chain line(8) < ray(2) < box(8) into ONE uint8 colour-index
  plane; off-grid -> 99 sentinel; Pad to 30x30; 10-way expansion lands in the
  FREE bool output via Equal(Lpad, arange_ch).  All working planes are bool/uint8
  (no full fp16/fp32 plane).  Crop to WK=20 (generator width,height in [12,20]).
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F = TensorProto.FLOAT
B = TensorProto.BOOL
U8 = TensorProto.UINT8
F16 = TensorProto.FLOAT16


def build(task):
    inits, nodes = [], []

    def init(name, arr, npdtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=npdtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, list(ins), [out], **attrs))
        return out

    WK = 20
    RED, CYAN = 2, 8

    # ---- conv weights (collapse one spatial axis) ----
    # collapse HEIGHT (kernel [1,10,30,1]) -> [1,1,1,30] per-column
    # ONE conv per collapse-direction carries TWO profiles in disjoint magnitude
    # bands:  val = (dot_pos + 1) + BAND * cyan_count   with BAND=64 (all ints
    # stay < 2048 so fp16-exact).  Decode: cyan_count = floor(val/64);
    # dot_pos = (val mod 64) - 1 ; presence = (val mod 64) > 0.
    BAND = 64
    # collapse HEIGHT -> [1,1,1,30]: red carries (row+1), cyan carries BAND
    wH = np.zeros((1, 10, 30, 1), np.float32)
    wH[0, RED, :, 0] = np.arange(30) + 1.0
    wH[0, CYAN, :, 0] = float(BAND)
    init("wH", wH, np.float32)
    # collapse WIDTH -> [1,1,30,1]: red carries (col+1), cyan carries BAND
    wW = np.zeros((1, 10, 1, 30), np.float32)
    wW[0, RED, 0, :] = np.arange(30) + 1.0
    wW[0, CYAN, 0, :] = float(BAND)
    init("wW", wW, np.float32)
    init("band32", np.array(float(BAND), np.float32), np.float32)

    # ---- constants ----
    init("rowidx", np.arange(WK, dtype=np.float16).reshape(1, 1, WK, 1), np.float16)
    init("BIG", np.array(100.0, np.float16), np.float16)
    init("nBIG", np.array(-100.0, np.float16), np.float16)
    init("BIG2", np.array(50.0, np.float16), np.float16)
    init("half", np.array(0.5, np.float16), np.float16)
    init("half32", np.array(0.5, np.float32), np.float32)
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    init("padO", np.array([0, 0, 0, 0, 0, 0, 30 - WK, 30 - WK], np.int64), np.int64)
    init("sentU8", np.array(99, np.uint8), np.uint8)
    init("u0", np.array(0, np.uint8), np.uint8)
    init("u2", np.array(2, np.uint8), np.uint8)
    init("u8", np.array(8, np.uint8), np.uint8)
    init("one16", np.array(1.0, np.float16), np.float16)
    # crop a [1,1,1,30] vector to [1,1,1,WK]
    init("v0", np.array([0], np.int64), np.int64)
    init("vK", np.array([WK], np.int64), np.int64)
    init("ax2", np.array([2], np.int64), np.int64)
    init("ax3", np.array([3], np.int64), np.int64)

    def conv(name, w):
        return n("Conv", ["input", w], name)

    # ---- TWO packed convs (band 64): cH collapses height, cW collapses width ----
    conv("cH30", "wH")             # [1,1,1,30]  (dot_row+1) + 64*cyan_per_col
    conv("cW30", "wW")             # [1,1,30,1]  (dot_col+1) + 64*cyan_per_row

    # active extents
    n("ReduceMax", ["input"], "rowany30", axes=[1, 3], keepdims=1)  # [1,1,30,1]
    n("ReduceMax", ["input"], "colany30", axes=[1, 2], keepdims=1)  # [1,1,1,30]
    n("ReduceSum", ["rowany30"], "nH", axes=[2, 3], keepdims=1)     # scalar
    n("ReduceSum", ["colany30"], "nW", axes=[2, 3], keepdims=1)     # scalar

    # decode cyan-per-row (from cW: floor(/64)) for horiz-line + orientation
    n("Div", ["cW30", "band32"], "cyrowQ30")
    n("Floor", ["cyrowQ30"], "cyperrowH30")        # [1,1,30,1] cyan per row (horiz lines)
    n("ReduceMax", ["cyperrowH30"], "maxcyrow", axes=[2, 3], keepdims=1)
    n("Equal", ["maxcyrow", "nW"], "horB")                          # scalar bool

    # ---- canonical (lines=rows) dot profile [1,1,1,WK] ----
    # horiz dot = (cH mod 64) - 1 ; vert dot = (transpose(cW) mod 64) - 1
    n("Transpose", ["cW30"], "cWt", perm=[0, 1, 3, 2])             # [1,1,1,30]
    n("Where", ["horB", "cH30", "cWt"], "cDot30")                 # packed dot, canonical
    n("Slice", ["cDot30", "v0", "vK", "ax3"], "cDotF")            # [1,1,1,WK] fp32
    # decode low band in fp32 (exact): dotraw = cDot - 64*floor(cDot/64)
    n("Div", ["cDotF", "band32"], "cDotq")
    n("Floor", ["cDotq"], "cDotcy")
    n("Mul", ["cDotcy", "band32"], "cDotcyB")
    n("Sub", ["cDotF", "cDotcyB"], "dotrawF")                    # (dot+1) low band, fp32
    n("Cast", ["dotrawF"], "dotraw", to=F16)
    n("Greater", ["dotraw", "half"], "hasdotB")                  # presence
    n("Sub", ["dotraw", "one16"], "dotrow")                      # real dot row

    # ---- canonical cyan-per-row [1,1,WK,1] ----
    # horiz lines: floor(cW/64) ; vert lines: floor(transpose(cH)/64)
    n("Transpose", ["cH30"], "cHt", perm=[0, 1, 3, 2])           # [1,1,30,1]
    n("Div", ["cHt", "band32"], "cyperrowV30q")
    n("Floor", ["cyperrowV30q"], "cyperrowV30")
    n("Where", ["horB", "cyperrowH30", "cyperrowV30"], "cyperrow30")
    n("Slice", ["cyperrow30", "v0", "vK", "ax2"], "cyperrowF")      # [1,1,WK,1]
    # canonical active width (cols) = horiz? nW : nH
    n("Where", ["horB", "nW", "nH"], "Wc")
    n("Equal", ["cyperrowF", "Wc"], "lineB")                        # [1,1,WK,1] bool
    # in-grid mask in ORIGINAL orientation (rowany (x) colany), cropped to WK
    n("Slice", ["rowany30", "v0", "vK", "ax2"], "ranyF")            # [1,1,WK,1]
    n("Slice", ["colany30", "v0", "vK", "ax3"], "canyF")           # [1,1,1,WK]
    n("Greater", ["ranyF", "half32"], "ranyB")
    n("Greater", ["canyF", "half32"], "canyB")
    n("And", ["ranyB", "canyB"], "ingO")                            # [1,1,WK,WK] bool (orig frame)

    # ---- line rows Lmin/Lmax (<=2 lines) ----
    n("Where", ["lineB", "rowidx", "BIG"], "lminv")
    n("ReduceMin", ["lminv"], "Lmin", axes=[2, 3], keepdims=1)
    n("Where", ["lineB", "rowidx", "nBIG"], "lmaxv")
    n("ReduceMax", ["lmaxv"], "Lmax", axes=[2, 3], keepdims=1)

    # ---- per-column Ldown/Lup ----
    n("Less", ["dotrow", "Lmin"], "d_ltmin")
    n("Less", ["dotrow", "Lmax"], "d_ltmax")
    n("Where", ["d_ltmax", "Lmax", "BIG"], "Ldn0")
    n("Where", ["d_ltmin", "Lmin", "Ldn0"], "Ldown")
    n("Greater", ["dotrow", "Lmax"], "d_gtmax")
    n("Greater", ["dotrow", "Lmin"], "d_gtmin")
    n("Where", ["d_gtmin", "Lmin", "nBIG"], "Lup0")
    n("Where", ["d_gtmax", "Lmax", "Lup0"], "Lup")
    n("Less", ["Ldown", "BIG2"], "dexB")
    n("Greater", ["Lup", "nBIG"], "uexB")
    n("And", ["dexB", "hasdotB"], "hasdownB")
    n("And", ["uexB", "hasdotB"], "hasupB")
    # ray bounds rlo/rhi (per col, [1,1,1,WK])
    n("Where", ["hasdotB", "dotrow", "BIG"], "rlo0")
    n("Where", ["hasdotB", "dotrow", "nBIG"], "rhi0")
    n("Where", ["hasupB", "Lup", "rlo0"], "rlo")
    n("Where", ["hasdownB", "Ldown", "rhi0"], "rhi")

    # per-column "reaches Lmin / Lmax" (for the separable per-line box)
    n("Equal", ["Lup", "Lmin"], "up_is_min"); n("And", ["up_is_min", "hasupB"], "rmin_u")
    n("Equal", ["Ldown", "Lmin"], "dn_is_min"); n("And", ["dn_is_min", "hasdownB"], "rmin_d")
    n("Or", ["rmin_u", "rmin_d"], "reach_min")          # [1,1,1,WK] bool
    n("Equal", ["Lup", "Lmax"], "up_is_max"); n("And", ["up_is_max", "hasupB"], "rmax_u")
    n("Equal", ["Ldown", "Lmax"], "dn_is_max"); n("And", ["dn_is_max", "hasdownB"], "rmax_d")
    n("Or", ["rmax_u", "rmax_d"], "reach_max")
    # widen reach ±1 column (horizontal box extent), via 1-D MaxPool on tiny profile
    n("Cast", ["reach_min"], "rmin_f", to=F16)
    n("MaxPool", ["rmin_f"], "rmin_w", kernel_shape=[1, 3], pads=[0, 1, 0, 1], strides=[1, 1])
    n("Greater", ["rmin_w", "half"], "rmin_wb")          # [1,1,1,WK] bool widened
    n("Cast", ["reach_max"], "rmax_f", to=F16)
    n("MaxPool", ["rmax_f"], "rmax_w", kernel_shape=[1, 3], pads=[0, 1, 0, 1], strides=[1, 1])
    n("Greater", ["rmax_w", "half"], "rmax_wb")

    # ---- ray plane (one full plane) ----
    n("Less", ["rowidx", "rlo"], "lt_lo")
    n("Greater", ["rowidx", "rhi"], "gt_hi")
    n("Or", ["lt_lo", "gt_hi"], "out_r")
    n("Not", ["out_r"], "rayB")

    # ---- box ROW bands = {L-1, L+1} ONLY (NOT the line row itself: the line is
    # already cyan from lineB, and the ray reds the dot-centre at row L).  This
    # lets the priority line < ray < box reproduce the red box-centre for FREE
    # (ray red shows at (L, dotcol) since box never covers row L).
    n("Sub", ["Lmin", "one16"], "Lmin_m1"); n("Add", ["Lmin", "one16"], "Lmin_p1")
    n("Sub", ["Lmax", "one16"], "Lmax_m1"); n("Add", ["Lmax", "one16"], "Lmax_p1")
    n("Equal", ["rowidx", "Lmin_m1"], "rmin_a"); n("Equal", ["rowidx", "Lmin_p1"], "rmin_b")
    n("Or", ["rmin_a", "rmin_b"], "rows_min")             # [1,1,WK,1]
    n("Equal", ["rowidx", "Lmax_m1"], "rmax_a"); n("Equal", ["rowidx", "Lmax_p1"], "rmax_b")
    n("Or", ["rmax_a", "rmax_b"], "rows_max")

    # ---- box (CYAN ring top/bottom) = separable per-line (rows (x) widened cols) ----
    n("And", ["rows_min", "rmin_wb"], "box_min")          # [1,1,WK,WK]
    n("And", ["rows_max", "rmax_wb"], "box_max")
    n("Or", ["box_min", "box_max"], "boxB")

    # ---- compose colour-index uint8 (priority): line < ray < box ----
    n("Where", ["lineB", "u8", "u0"], "L0")     # cyan lines (broadcast over cols)
    n("Where", ["rayB", "u2", "L0"], "L1")      # red ray (reds the box centre)
    n("Where", ["boxB", "u8", "L1"], "L3")      # cyan ring top/bottom

    # ---- transpose back + offgrid sentinel + expand ----
    n("Transpose", ["L3"], "L3T", perm=[0, 1, 3, 2])
    n("Where", ["horB", "L3", "L3T"], "Lc")        # original-frame colour-index
    n("Where", ["ingO", "Lc", "sentU8"], "Lm")
    n("Pad", ["Lm", "padO", "sentU8"], "Lpad")
    n("Equal", ["Lpad", "chan"], "output")

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task379", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
