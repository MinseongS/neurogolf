"""task367 (ARC-AGI e73095fd) — fill box-outline interiors over line clutter.

Rule: 2-4 gray(5) rectangle OUTLINES (>=3x3, black interior) plus straight gray LINES
connecting boxes/edges. A box may be clipped by exactly ONE column at left (col=-1) or
right (right wall off-grid). OUTPUT: every BLACK interior cell of a real box outline ->
YELLOW(4); gray & background unchanged. The discriminator between a real box wall and a
straight line that fakes a rectangle is HORIZONTAL CORNER-TERMINATION: a box wall's gray
run STOPS at the corner, a line passes straight through.

GATHER-FREE exact predicate (verified 1500/1500 fresh, replaces the prior int64 GatherND
wall reads).  For each black cell (r,c):
  - rt = nearest gray UP (col c),  rb = nearest gray DOWN (col c)
  - c0 = nearest gray LEFT (row r), c1 = nearest gray RIGHT (row r)  (clip side -> grid edge)
  - top wall: the gray run in row rt that sits at column c has endpoints exactly [c0,c1]
  - bottom wall: same with rb
  - left/right vertical walls span exactly [rt,rb] (skip a clipped side)
All four "wall info at the cell" values are obtained by a DIRECTIONAL CARRY (prefix-max of a
position-packed plane), NOT a per-cell gather.  Run endpoints are prefix-max / suffix-min of
edge marks.  Directional prefix/suffix max = MaxPool with a one-sided full-length 1-D kernel
(zero params).  Everything runs on the 20x20 active crop in fp16.

Packing for a carry: pack = prio*BIG + (val+OFF) at gray cells, sentinel elsewhere; prefix-max
along the axis; decode val = pack mod BIG - OFF, src = (pack//BIG) [reversed for up/left carries].
A=20 so all packed integers stay < 2048 (fp16-exact).  BIG=32, OFF=10, P=20.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I64 = TensorProto.INT64

A = 20            # active canvas (grid <= 20x20)
BIG = 32.0
OFF = 10.0
P = 20.0
SENT = -60000.0  # fp16-safe MaxPool sentinel (< all real packed values, > fp16 -inf)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        if isinstance(ins, str):
            ins = [ins]
        nodes.append(helper.make_node(op, list(ins), [out], **attrs))
        return out

    # ---- gray plane on the AxA crop, fp16 ---------------------------------
    init("g_s", np.array([5, 0, 0], np.int64), np.int64)
    init("g_e", np.array([6, A, A], np.int64), np.int64)
    init("g_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "g_s", "g_e", "g_ax"], "gray_f32")        # [1,1,A,A] f32
    n("Cast", ["gray_f32"], "g", to=F16)                          # {0,1} fp16

    # position ramps (fp16)
    colramp = np.arange(A, dtype=np.float16).reshape(1, 1, 1, A)
    rowramp = np.arange(A, dtype=np.float16).reshape(1, 1, A, 1)
    init("colramp", colramp, np.float16)
    init("rowramp", rowramp, np.float16)
    init("ZH", np.array(0.0, np.float16), np.float16)
    init("HALF", np.array(0.5, np.float16), np.float16)
    init("ONEH", np.array(1.0, np.float16), np.float16)
    init("SENTH", np.array(SENT, np.float16), np.float16)

    # ---- helper: prefix/suffix max via MaxPool (one-sided full-length kernel)
    def pmax(src, out, axis, rev):
        # axis: 2 (rows, vertical) or 3 (cols, horizontal). rev=True => suffix.
        if axis == 3:
            ks = [1, A]
            pads = [0, 0, 0, A - 1] if rev else [0, A - 1, 0, 0]
        else:
            ks = [A, 1]
            pads = [0, 0, A - 1, 0] if rev else [A - 1, 0, 0, 0]
        nodes.append(helper.make_node("MaxPool", [src], [out],
                                      kernel_shape=ks, pads=pads, strides=[1, 1]))
        return out

    # ---- shifted gray (neighbor masks) for run-edge detection -------------
    # left neighbor: gL[r,c] = g[r,c-1]; right neighbor gR[r,c]=g[r,c+1];
    # up gU[r,c]=g[r-1,c]; down gD[r,c]=g[r+1,c].  Build with Pad+Slice.
    def shift(src, out, axis, amt):
        # amt = +1 -> shift content toward larger index (neighbor = content at idx-1)
        # implement gL (g[r,c-1]) : pad left by 1, slice off the last col.
        # ONNX Pad pads = [b0,b1,b2,b3, e0,e1,e2,e3]
        if axis == 3:
            if amt == +1:      # gL: g[r,c-1] : pad width-begin
                pads = [0, 0, 0, 1, 0, 0, 0, 0]; ss = [0]; se = [A]; sa = [3]
            else:              # gR: g[r,c+1] : pad width-end
                pads = [0, 0, 0, 0, 0, 0, 0, 1]; ss = [1]; se = [A + 1]; sa = [3]
        else:
            if amt == +1:      # gU: g[r-1,c] : pad height-begin
                pads = [0, 0, 1, 0, 0, 0, 0, 0]; ss = [0]; se = [A]; sa = [2]
            else:              # gD: g[r+1,c] : pad height-end
                pads = [0, 0, 0, 0, 0, 0, 1, 0]; ss = [1]; se = [A + 1]; sa = [2]
        pn = out + "_p"
        nodes.append(helper.make_node("Pad", [src, init(out + "_pads", np.array(pads, np.int64), np.int64), "ZH"], [pn], mode="constant"))
        init(out + "_ss", np.array(ss, np.int64), np.int64)
        init(out + "_se", np.array(se, np.int64), np.int64)
        init(out + "_sa", np.array(sa, np.int64), np.int64)
        nodes.append(helper.make_node("Slice", [pn, out + "_ss", out + "_se", out + "_sa"], [out]))
        return out

    shift("g", "gL", 3, +1)
    shift("g", "gR", 3, -1)
    shift("g", "gU", 2, +1)
    shift("g", "gD", 2, -1)

    # ---- run-edge marks ---------------------------------------------------
    # is_gray = g; left edge of a horizontal run: g==1 & gL==0
    # leftedge_val = colramp where (g & ~gL) else SENT  (for prefix-max -> Lstart)
    # rightedge: g & ~gR -> for suffix-MIN of colramp -> negate & suffix-max.
    def edge_mark(g, gnb, ramp, out, neg):
        # mark = ramp where (g==1 & gnb==0) else SENT.  neg=True -> use -ramp (for min via max)
        # (1-gnb): gnb is {0,1}; notnb = 1-gnb
        n("Sub", ["ONEH", gnb], out + "_notnb")          # 1-gnb
        n("Mul", [g, out + "_notnb"], out + "_em")        # g & ~nb  -> {0,1}
        # value = (neg?-:+)ramp ; where mark else SENT:
        rr = ramp if not neg else out + "_nr"
        if neg:
            n("Sub", ["ZH", ramp], out + "_nr")           # -ramp
        # masked = mark*val + (1-mark)*SENT
        n("Mul", [out + "_em", rr], out + "_mv")
        n("Sub", ["ONEH", out + "_em"], out + "_im")
        n("Mul", [out + "_im", "SENTH"], out + "_sv")
        n("Add", [out + "_mv", out + "_sv"], out)
        return out

    edge_mark("g", "gL", "colramp", "leftedge", neg=False)   # prefix-max -> Lstart
    edge_mark("g", "gR", "colramp", "rightedge", neg=True)    # suffix-max(-ramp) -> -Rend
    edge_mark("g", "gU", "rowramp", "topedge", neg=False)     # prefix-max -> Tstart
    edge_mark("g", "gD", "rowramp", "botedge", neg=True)      # suffix-max(-ramp) -> -Bend

    pmax("leftedge", "Lstart", axis=3, rev=False)             # [1,1,A,A]
    pmax("rightedge", "Rend_neg", axis=3, rev=True)
    n("Sub", ["ZH", "Rend_neg"], "Rend")                      # Rend = -(-Rend)
    pmax("topedge", "Tstart", axis=2, rev=False)
    pmax("botedge", "Bend_neg", axis=2, rev=True)
    n("Sub", ["ZH", "Bend_neg"], "Bend")

    # ---- directional carries ----------------------------------------------
    init("BIGH", np.array(BIG, np.float16), np.float16)
    init("OFFH", np.array(OFF, np.float16), np.float16)
    init("PH", np.array(P, np.float16), np.float16)
    init("HALFSENT", np.array(SENT / 2.0, np.float16), np.float16)
    init("NEG99", np.array(-99.0, np.float16), np.float16)

    # precompute g as bool (for Where masking) and the four prio*BIG planes once
    n("Cast", ["g"], "gbool", to=BOOL)
    # prioBig planes with OFF folded in: prio*BIG + OFF
    n("Mul", ["rowramp", "BIGH"], "rowprioF0")
    n("Add", ["rowprioF0", "OFFH"], "rowprioF")          # row*BIG+OFF (fwd, axis2)
    n("Mul", ["colramp", "BIGH"], "colprioF0")
    n("Add", ["colprioF0", "OFFH"], "colprioF")          # col*BIG+OFF (fwd, axis3)
    n("Sub", ["PH", "rowramp"], "rowrampR")
    n("Mul", ["rowrampR", "BIGH"], "rowprioR0")
    n("Add", ["rowprioR0", "OFFH"], "rowprioR")          # (P-row)*BIG+OFF
    n("Sub", ["PH", "colramp"], "colrampR")
    n("Mul", ["colrampR", "BIGH"], "colprioR0")
    n("Add", ["colprioR0", "OFFH"], "colprioR")          # (P-col)*BIG+OFF

    def carry(val, axis, rev, outv, outsrc=None):
        if axis == 2:
            prioBig = "rowprioR" if rev else "rowprioF"
        else:
            prioBig = "colprioR" if rev else "colprioF"
        # packed = (prio*BIG+OFF) + val where gray else SENT
        n("Add", [prioBig, val], outv + "_pk0")
        n("Where", ["gbool", outv + "_pk0", "SENTH"], outv + "_pk")
        pmax(outv + "_pk", outv + "_pm", axis=axis, rev=rev)
        # val output: unseen decodes to a small value; correctness is gated by have_top/have_bot
        # and the src-side clip flags, so no per-val sentinel Where is needed.
        n("Mod", [outv + "_pm", "BIGH"], outv + "_md", fmod=1)
        n("Sub", [outv + "_md", "OFFH"], outv)            # carried val
        if outsrc is not None:
            n("Greater", [outv + "_pm", "HALFSENT"], outv + "_seen")  # bool seen
            n("Div", [outv + "_pm", "BIGH"], outv + "_dv")
            n("Floor", [outv + "_dv"], outv + "_pc")
            if rev:
                n("Sub", ["PH", outv + "_pc"], outv + "_sraw")
                n("Where", [outv + "_seen", outv + "_sraw", "NEG99"], outsrc)
            else:
                n("Where", [outv + "_seen", outv + "_pc", "NEG99"], outsrc)
        return outv

    carry("Lstart", 2, False, "topL", "topRow")
    carry("Rend",   2, False, "topR")
    carry("Lstart", 2, True,  "botL", "botRow")
    carry("Rend",   2, True,  "botR")
    carry("Tstart", 3, False, "leftT", "leftCol")
    carry("Bend",   3, False, "leftB")
    carry("Tstart", 3, True,  "rightT", "rightCol")
    carry("Bend",   3, True,  "rightB")

    # ---- predicate (all fp16 / bool elementwise) --------------------------
    # black cell: g==0 (interior is black). also need it to be background black, but
    # interior cells are black(0); we only paint where predicate holds AND cell is black.
    # blackcell = (input black plane). Use channel 0.
    init("k0_s", np.array([0, 0, 0], np.int64), np.int64)
    init("k0_e", np.array([1, A, A], np.int64), np.int64)
    n("Slice", ["input", "k0_s", "k0_e", "g_ax"], "black_f32")
    n("Cast", ["black_f32"], "black_b", to=BOOL)              # [1,1,A,A] bool

    # existence of top/bottom walls: topRow>=0 and botRow>=0 (src>=0 means a gray was seen)
    # since SENT-derived prio decodes to large negative -> floor gives big neg; seen => prio>=0.
    init("NEGHALF", np.array(-0.5, np.float16), np.float16)
    n("Greater", ["topRow", "NEGHALF"], "have_top")          # topRow>=0
    n("Greater", ["botRow", "NEGHALF"], "have_bot")

    # in-grid mask (grid is the solid top-left HxW rect; off-grid = all channels 0).
    # last in-grid column of each row = max colramp where in-grid -> prefix-max over cols.
    # input colours are only gray(5) and black(0); in-grid = black OR gray.
    # black_bh is the fp16 black plane (defined below); compute ingrid = max(black,g).
    n("Where", ["black_b", "ONEH", "g"], "ingrid")               # black -> 1 else g
    # colramp where ingrid else 0, ReduceMax over cols -> last in-grid col per row
    n("Mul", ["ingrid", "colramp"], "igcol")                     # [1,1,A,A]
    n("ReduceMax", ["igcol"], "lastcol", axes=[3], keepdims=1)   # [1,1,A,1] last in-grid col/row
    # c0 = left_clip? 0 : leftCol ; c1 = right_clip? lastcol : rightCol
    # left_clip = leftCol<0 ; right_clip = rightCol<0
    n("Less", ["leftCol", "ZH"], "left_clip")
    n("Less", ["rightCol", "ZH"], "right_clip")
    # c0 = Where(left_clip, 0, leftCol)
    n("Where", ["left_clip", "ZH", "leftCol"], "c0")
    n("Where", ["right_clip", "lastcol", "rightCol"], "c1")
    # rt=topRow, rb=botRow
    # cond top: topL<=c0 and topR>=c1
    def le(a, b, out):  # a<=b  ==  not(a>b)
        n("Greater", [a, b], out + "_g"); n("Not", [out + "_g"], out)
    def ge(a, b, out):  # a>=b == not(a<b)
        n("Less", [a, b], out + "_l"); n("Not", [out + "_l"], out)
    le("topL", "c0", "t1"); ge("topR", "c1", "t2")
    le("botL", "c0", "b1"); ge("botR", "c1", "b2")
    n("And", ["t1", "t2"], "topok")
    n("And", ["b1", "b2"], "botok")
    # left wall (skip if clipped): leftT<=rt and leftB>=rb
    le("leftT", "topRow", "l1"); ge("leftB", "botRow", "l2")
    n("And", ["l1", "l2"], "leftok0")
    n("Or", ["leftok0", "left_clip"], "leftok")
    le("rightT", "topRow", "r1"); ge("rightB", "botRow", "r2")
    n("And", ["r1", "r2"], "rightok0")
    n("Or", ["rightok0", "right_clip"], "rightok")
    # corner termination: right: topR==c1 and botR==c1 (skip if right_clip)
    n("Equal", ["topR", "c1"], "ct_tr"); n("Equal", ["botR", "c1"], "ct_br")
    n("And", ["ct_tr", "ct_br"], "ct_right0")
    n("Or", ["ct_right0", "right_clip"], "ct_right")
    n("Equal", ["topL", "c0"], "ct_tl"); n("Equal", ["botL", "c0"], "ct_bl")
    n("And", ["ct_tl", "ct_bl"], "ct_left0")
    n("Or", ["ct_left0", "left_clip"], "ct_left")

    # combine all
    n("And", ["have_top", "have_bot"], "p1")
    n("And", ["topok", "botok"], "p2")
    n("And", ["leftok", "rightok"], "p3")
    n("And", ["ct_right", "ct_left"], "p4")
    n("And", ["p1", "p2"], "pa")
    n("And", ["p3", "p4"], "pb")
    n("And", ["pa", "pb"], "pred0")
    n("And", ["pred0", "black_b"], "interior")               # [1,1,A,A] bool

    # ---- route into FREE output: yellow(4) where interior ------------------
    n("Cast", ["interior"], "int_u8", to=U8)
    init("pads30", np.array([0, 0, 0, 0, 0, 0, 30 - A, 30 - A], np.int64), np.int64)
    init("ZU8", np.array(0, np.uint8), np.uint8)
    n("Pad", ["int_u8", "pads30", "ZU8"], "int30", mode="constant")
    n("Cast", ["int30"], "cond", to=BOOL)
    oh = np.zeros((1, 10, 1, 1), np.float32); oh[0, 4, 0, 0] = 1.0
    init("yellow_oh", oh, np.float32)
    n("Where", ["cond", "yellow_oh", "input"], "output")

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task367", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
