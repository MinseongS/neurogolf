"""task034 (ARC-AGI 1f0c79e5) — diagonal staircase sprouts from a 2x2 seed.

Rule (from the generator): a single 2x2 square of one `color` sits with top-left
corner (R,C) on a 9x9 grid. Its four corners map 1:1 to the four diagonal
directions:
    dir0 TL=(R,C)   delta(-1,-1)   dir1 TR=(R,C+1)   delta(-1,1)
    dir3 BL=(R+1,C) delta(1,-1)    dir2 BR=(R+1,C+1) delta(1,1)
In the INPUT a chosen corner is painted RED(2); unchosen corners keep `color`.
In the OUTPUT every corner keeps `color` (the seed square) and each chosen
corner grows an outward diagonal staircase to the edge.

Closed form (verified exact over thousands of fresh instances): relative to a
chosen corner (r0,c0) with delta (dr,dc), cell (r,c) is painted iff
    a>=0 AND b>=0 AND |a-b|<=1,  a=(r-r0)*dr, b=(c-c0)*dc.

MEMORY MODEL (this rebuild vs the 18666 original):
  * ONE fp32 colour plane is materialised (Conv over the 10-channel input ->
    [1,1,30,30]); it is immediately cropped to the 9x9 active region. Everything
    downstream lives on the 9x9 canvas in uint8/bool (81 B), never fp32 (324 B),
    and the per-direction band predicate is built separably along row [1,1,9,1]
    and col [1,1,1,9].
  * the four chosen flags are read from a real 9x9 red plane at the corner cells
    (a true 2-D lookup, NOT a row x col outer product, which would cross-talk
    when two corners are red).
  * the label map L9[1,1,9,9] uint8 = colour where painted else 0; Pad to
    [1,1,30,30] with sentinel 10; emit output = Equal(L, arange[0..9]) -> BOOL
    so the 10-channel expansion lands in the FREE output tensor.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

S = 9          # working grid size
G = 30         # full canvas
DELTAS = [(-1, -1), (-1, 1), (1, 1), (1, -1)]  # dir0..dir3


def build(task):
    inits, nodes, vinfos = [], [], []

    F = TensorProto.FLOAT
    U8 = TensorProto.UINT8
    B = TensorProto.BOOL

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    def vi(name, dtype, shape):
        vinfos.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    crop = G - S
    init("half", np.array(0.5, np.float32), np.float32)
    init("one", np.array(1.0, np.float32), np.float32)
    init("BIG", np.array(99.0, np.float32), np.float32)

    # ---------- colour scalar from channel presence (no colour plane) ----------
    # pres[0,k,0,0] = 1 iff colour k appears anywhere; colour = the lone non-bg
    # non-red colour present.  These reductions never touch a 30x30 plane
    # ([1,10,1,1] = 40 B).
    n("ReduceMax", ["input"], "pres", axes=[2, 3], keepdims=1)   # [1,10,1,1]
    vi("pres", F, [1, 10, 1, 1])
    cwv = np.arange(10, dtype=np.float32)
    cwv[0] = 0.0
    cwv[2] = 0.0
    init("colw", cwv.reshape(1, 10, 1, 1), np.float32)           # [1,10,1,1]
    n("Mul", ["pres", "colw"], "colparts")                       # [1,10,1,1]
    vi("colparts", F, [1, 10, 1, 1])
    n("ReduceSum", ["colparts"], "color", keepdims=1)            # [1,1,1,1] scalar
    vi("color", F, [1, 1, 1, 1])

    # ---------- crop a 3-channel PxP window, derive bg & red planes -----------
    # The seed 2x2 (corner R,C in [1,6]) lives in rows/cols 1..7, so a 7x7 window
    # at offset (1,1) captures every occupied & every red cell.  Only channels
    # 0..2 are needed: a cell is OCCUPIED iff background channel 0 is OFF (true
    # for every colour, not just 0..2), and RED iff channel 2 is ON.  Cropping to
    # [1,3,7,7] = 588 B (vs 1960 for all 10 channels) is the only per-cell input
    # tensor.  Cropped window index t maps to absolute coordinate t + OFF.
    P = S - 2   # 7
    OFF = 1
    KEEP = 3    # channels 0..2
    # Pad crops per side: begins remove OFF (top/left); also drop channels 3..9.
    # pad-spec layout = [b_n,b_c,b_h,b_w, e_n,e_c,e_h,e_w].
    init("croppadsX",
         np.array([0, 0, -OFF, -OFF, 0, -(10 - KEEP), -(G - P - OFF),
                   -(G - P - OFF)], np.int64), np.int64)
    n("Pad", ["input", "croppadsX"], "Xc", mode="constant")      # [1,3,7,7] f32
    vi("Xc", F, [1, KEEP, P, P])
    # two single-output Convs collapse channels to a bg plane and a red plane,
    # each [1,1,7,7] = 196 B (no [1,2,7,7] intermediate, no slices).
    bgw = np.zeros((1, KEEP, 1, 1), np.float32)
    bgw[0, 0, 0, 0] = 1.0   # background channel
    init("bgw", bgw, np.float32)
    n("Conv", ["Xc", "bgw"], "bgp")                              # [1,1,7,7] bg f32
    vi("bgp", F, [1, 1, P, P])
    redw = np.zeros((1, KEEP, 1, 1), np.float32)
    redw[0, 2, 0, 0] = 1.0  # red channel
    init("redw", redw, np.float32)
    n("Conv", ["Xc", "redw"], "redp")                            # [1,1,7,7] red f32
    vi("redp", F, [1, 1, P, P])

    # red (==1 in this window) plane, bool.
    n("Greater", ["redp", "half"], "red")                        # [1,1,7,7] bool
    vi("red", B, [1, 1, P, P])

    # ---------- R = min occupied row, C = min occupied col --------------------
    # occupancy = background OFF.  A row is occupied iff its minimum bg value < .5
    # (i.e. some cell has bg=0); reduce the fp32 bg plane directly (no occ plane).
    rrP = (np.arange(P, dtype=np.float32) + OFF).reshape(1, 1, P, 1)
    ccP = (np.arange(P, dtype=np.float32) + OFF).reshape(1, 1, 1, P)
    init("rrP", rrP, np.float32)                                 # [1,1,7,1] abs row
    init("ccP", ccP, np.float32)                                 # [1,1,1,7] abs col
    rrS = np.arange(S, dtype=np.float32).reshape(1, 1, S, 1)
    ccS = np.arange(S, dtype=np.float32).reshape(1, 1, 1, S)
    init("rrS", rrS, np.float32)                                 # [1,1,9,1]
    init("ccS", ccS, np.float32)                                 # [1,1,1,9]

    n("ReduceMin", ["bgp"], "rbgmin", axes=[3], keepdims=1)      # [1,1,7,1] f32
    vi("rbgmin", F, [1, 1, P, 1])
    n("Less", ["rbgmin", "half"], "roccb")                       # occupied row
    vi("roccb", B, [1, 1, P, 1])
    n("Where", ["roccb", "rrP", "BIG"], "ridx")
    vi("ridx", F, [1, 1, P, 1])
    n("ReduceMin", ["ridx"], "R", keepdims=1)                    # [1,1,1,1] abs R
    vi("R", F, [1, 1, 1, 1])

    n("ReduceMin", ["bgp"], "cbgmin", axes=[2], keepdims=1)      # [1,1,1,7] f32
    vi("cbgmin", F, [1, 1, 1, P])
    n("Less", ["cbgmin", "half"], "coccb")                       # occupied col
    vi("coccb", B, [1, 1, 1, P])
    n("Where", ["coccb", "ccP", "BIG"], "cidv")
    vi("cidv", F, [1, 1, 1, P])
    n("ReduceMin", ["cidv"], "C", keepdims=1)                    # [1,1,1,1] abs C
    vi("C", F, [1, 1, 1, 1])

    n("Add", ["R", "one"], "R1")
    vi("R1", F, [1, 1, 1, 1])
    n("Add", ["C", "one"], "C1")
    vi("C1", F, [1, 1, 1, 1])

    # ---------- gather the 4 corner red values (true 2-D lookup, scalars) -----
    # red9 = uint8 red plane; gather row Ri then col Ci to read red at a corner.
    n("Cast", ["red"], "red8", to=U8)                            # [1,1,7,7] uint8
    vi("red8", U8, [1, 1, P, P])
    n("Cast", ["R"], "Ri", to=TensorProto.INT64)                # scalar int (abs)
    vi("Ri", TensorProto.INT64, [1, 1, 1, 1])
    n("Cast", ["C"], "Ci", to=TensorProto.INT64)
    vi("Ci", TensorProto.INT64, [1, 1, 1, 1])
    n("Squeeze", ["Ri"], "Ris")                                  # scalar
    n("Squeeze", ["Ci"], "Cis")
    # window-relative corner indices: rows {R-OFF, R-OFF+1}, cols {C-OFF, C-OFF+1}.
    init("idx01w", np.array([-OFF, -OFF + 1], np.int64), np.int64)
    n("Add", ["Ris", "idx01w"], "rows2")                         # [2] = {R-OFF,..}
    n("Add", ["Cis", "idx01w"], "cols2")                         # [2] = {C-OFF,..}
    n("Gather", ["red8", "rows2"], "redr", axis=2)               # [1,1,2,7] uint8
    vi("redr", U8, [1, 1, 2, P])
    n("Gather", ["redr", "cols2"], "redq", axis=3)               # [1,1,2,2] uint8
    vi("redq", U8, [1, 1, 2, 2])
    # redq[0,0,i,j] = red at corner (R+i, C+j).  Map the 4 corners:
    #   dir0 (R,   C)   -> [0,0]   dir1 (R,   C+1) -> [0,1]
    #   dir2 (R+1, C+1) -> [1,1]   dir3 (R+1, C)   -> [1,0]
    CORNER_IJ = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}

    # ---------- the 2x2 seed square (painted in every instance) ---------------
    # seed = (R <= r <= R+1) & (C <= c <= C+1) — one 9x9 plane shared by all dirs.
    init("onep5n", np.array(-1.5, np.float32), np.float32)
    init("onep5", np.array(1.5, np.float32), np.float32)
    init("nhalf", np.array(-0.5, np.float32), np.float32)
    n("Sub", ["rrS", "R"], "rmR")                                # [1,1,9,1] r-R
    vi("rmR", F, [1, 1, S, 1])
    n("Greater", ["rmR", "nhalf"], "rlo")                        # r-R > -0.5 (>=0)
    vi("rlo", B, [1, 1, S, 1])
    n("Less", ["rmR", "onep5"], "rhi")                           # r-R < 1.5 (<=1)
    vi("rhi", B, [1, 1, S, 1])
    n("And", ["rlo", "rhi"], "rin")                              # [1,1,9,1]
    vi("rin", B, [1, 1, S, 1])
    n("Sub", ["ccS", "C"], "cmC")                                # [1,1,1,9] c-C
    vi("cmC", F, [1, 1, 1, S])
    n("Greater", ["cmC", "nhalf"], "clo")                        # c-C > -0.5 (>=0)
    vi("clo", B, [1, 1, 1, S])
    n("Less", ["cmC", "onep5"], "chi")                           # c-C < 1.5 (<=1)
    vi("chi", B, [1, 1, 1, S])
    n("And", ["clo", "chi"], "cin")                              # [1,1,1,9]
    vi("cin", B, [1, 1, 1, S])
    n("And", ["rin", "cin"], "seedsq")                           # [1,1,9,9]
    vi("seedsq", B, [1, 1, S, S])

    # ---------- per-direction band gated by the chosen flag -------------------
    # band predicate stays bool throughout: a>=0, b>=0 are 1-D row/col compares;
    # |a-b|<=1 <=> (a > b-1.5) AND (b > a-1.5).  The chosen flag is folded into
    # the 1-D `age` (a tiny [1,1,9,1] AND) so no separate 9x9 gate plane exists.
    corner_r = {0: "R", 1: "R", 2: "R1", 3: "R1"}
    corner_c = {0: "C", 1: "C1", 2: "C1", 3: "C"}

    L9_acc = "seedsq"      # bool "painted" accumulator (seed always on)
    for d in range(4):
        cr = corner_r[d]
        ccc = corner_c[d]
        dr, dc = DELTAS[d]
        i, j = CORNER_IJ[d]

        # chosen flag scalar = redq[0,0,i,j] (uint8) > 0.
        init(f"fst{d}", np.array([0, 0, i, j], np.int64), np.int64)
        init(f"fen{d}", np.array([1, 1, i + 1, j + 1], np.int64), np.int64)
        init(f"fax{d}", np.array([0, 1, 2, 3], np.int64), np.int64)
        n("Slice", ["redq", f"fst{d}", f"fen{d}", f"fax{d}"], f"fcell{d}")  # [1,1,1,1] u8
        vi(f"fcell{d}", U8, [1, 1, 1, 1])
        n("Cast", [f"fcell{d}"], f"flagf{d}", to=F)              # [1,1,1,1] f32
        vi(f"flagf{d}", F, [1, 1, 1, 1])
        n("Greater", [f"flagf{d}", "half"], f"flagb{d}")         # [1,1,1,1] bool
        vi(f"flagb{d}", B, [1, 1, 1, 1])

        # a = (r-r0)*dr  (1-D row vector);  b = (c-c0)*dc  (1-D col vector).
        n("Sub", ["rrS", cr], f"dra{d}")                         # [1,1,9,1]
        vi(f"dra{d}", F, [1, 1, S, 1])
        init(f"sdr{d}", np.array(float(dr), np.float32), np.float32)
        init(f"sdc{d}", np.array(float(dc), np.float32), np.float32)
        n("Mul", [f"dra{d}", f"sdr{d}"], f"a{d}")                # [1,1,9,1]
        vi(f"a{d}", F, [1, 1, S, 1])
        n("Sub", ["ccS", ccc], f"dcb{d}")                        # [1,1,1,9]
        vi(f"dcb{d}", F, [1, 1, 1, S])
        n("Mul", [f"dcb{d}", f"sdc{d}"], f"b{d}")                # [1,1,1,9]
        vi(f"b{d}", F, [1, 1, 1, S])

        # a >= 0 (1-D), gated by the chosen flag -> tiny [1,1,9,1] AND.
        n("Greater", [f"a{d}", "onep5n"], f"age0{d}")            # a > -1.5
        vi(f"age0{d}", B, [1, 1, S, 1])
        n("And", [f"age0{d}", f"flagb{d}"], f"age{d}")           # gated, [1,1,9,1]
        vi(f"age{d}", B, [1, 1, S, 1])
        n("Greater", [f"b{d}", "onep5n"], f"bge{d}")             # b > -1.5
        vi(f"bge{d}", B, [1, 1, 1, S])
        n("And", [f"age{d}", f"bge{d}"], f"ab{d}")               # [1,1,9,9] bool
        vi(f"ab{d}", B, [1, 1, S, S])

        # |a-b|<=1: am = a-1.5, bm = b-1.5; (b > am) AND (a > bm).
        n("Add", [f"a{d}", "onep5n"], f"am{d}")                  # [1,1,9,1] = a-1.5
        vi(f"am{d}", F, [1, 1, S, 1])
        n("Add", [f"b{d}", "onep5n"], f"bm{d}")                  # [1,1,1,9] = b-1.5
        vi(f"bm{d}", F, [1, 1, 1, S])
        n("Greater", [f"b{d}", f"am{d}"], f"bgta{d}")            # [1,1,9,9] bool
        vi(f"bgta{d}", B, [1, 1, S, S])
        n("Greater", [f"a{d}", f"bm{d}"], f"agtb{d}")            # [1,1,9,9] bool
        vi(f"agtb{d}", B, [1, 1, S, S])

        # cond = ab & bgta & agtb  (chained ANDs, no separate `band` plane).
        n("And", [f"ab{d}", f"bgta{d}"], f"cnd1{d}")             # [1,1,9,9] bool
        vi(f"cnd1{d}", B, [1, 1, S, S])
        n("And", [f"cnd1{d}", f"agtb{d}"], f"cond{d}")           # [1,1,9,9] bool
        vi(f"cond{d}", B, [1, 1, S, S])

        nn = f"onacc{d}"
        n("Or", [L9_acc, f"cond{d}"], nn)
        vi(nn, B, [1, 1, S, S])
        L9_acc = nn

    # ---------- label map: colour where painted else 0 (uint8 9x9) ------------
    n("Cast", ["color"], "coloru8", to=U8)                       # [1,1,1,1] uint8
    vi("coloru8", U8, [1, 1, 1, 1])
    init("zerou8", np.array(0, np.uint8), np.uint8)
    n("Where", [L9_acc, "coloru8", "zerou8"], "L9")              # [1,1,9,9] uint8
    vi("L9", U8, [1, 1, S, S])

    init("padpads", np.array([0, 0, 0, 0, 0, 0, crop, crop], np.int64), np.int64)
    init("sent", np.array(10, np.uint8), np.uint8)
    n("Pad", ["L9", "padpads", "sent"], "L", mode="constant")    # [1,1,30,30] u8
    vi("L", U8, [1, 1, G, G])

    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["L", "chan"], "output")                          # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F, [1, 10, G, G])
    y = helper.make_tensor_value_info("output", B, [1, 10, G, G])
    g = helper.make_graph(nodes, "task034", [x], [y], inits, value_info=vinfos)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
