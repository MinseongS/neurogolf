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
So per corner the painted set is the main diagonal ray plus its two adjacent
rays running to the edge — a width-3 diagonal band.

Net: ALL per-cell work is done on the bounded 9x9 region (negative-Pad crop, so
every canvas tensor is [1,1,9,9] = tiny). Read parameters (colour, R, C, the
four chosen flags) with cheap reductions; the painted band per direction is an
affine predicate on broadcast row/col index vectors. Build a uint8 label map
L9[1,1,9,9] = output colour index (0=bg), then Pad to [1,1,30,30] with the
off-grid sentinel 10 and emit output = Equal(L, arange[0..9]) -> BOOL.
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

    # --- crop input to the 9x9 region (negative pads = free crop) ---------
    crop = G - S  # 21
    init("croppads", np.array([0, 0, 0, 0, 0, 0, -crop, -crop], np.int64),
         np.int64)
    n("Pad", ["input", "croppads"], "X", mode="constant")           # [1,10,9,9]
    vi("X", F, [1, 10, S, S])

    # --- colour-index plane: cidx[r,c] = colour at (r,c) (0..9) -----------
    aw = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("aw", aw, np.float32)
    n("Conv", ["X", "aw"], "cidx")                  # [1,1,9,9] f32
    vi("cidx", F, [1, 1, S, S])

    init("half", np.array(0.5, np.float32), np.float32)
    init("two", np.array(2.0, np.float32), np.float32)
    n("Greater", ["cidx", "half"], "nz")            # bool
    vi("nz", B, [1, 1, S, S])
    n("Equal", ["cidx", "two"], "red")              # bool
    vi("red", B, [1, 1, S, S])
    n("Cast", ["nz"], "nzf", to=F)
    vi("nzf", F, [1, 1, S, S])
    n("Cast", ["red"], "redf", to=F)
    vi("redf", F, [1, 1, S, S])

    # --- R = min occupied row, C = min occupied col -----------------------
    init("BIG", np.array(99.0, np.float32), np.float32)
    init("one", np.array(1.0, np.float32), np.float32)
    init("nhalf", np.array(-0.5, np.float32), np.float32)
    init("onep5", np.array(1.5, np.float32), np.float32)
    rr = np.arange(S, dtype=np.float32).reshape(1, 1, S, 1)
    cc = np.arange(S, dtype=np.float32).reshape(1, 1, 1, S)
    init("rr", rr, np.float32)                      # [1,1,9,1]
    init("cc", cc, np.float32)                      # [1,1,1,9]

    n("ReduceMax", ["nzf"], "rowhas", axes=[3], keepdims=1)   # [1,1,9,1]
    vi("rowhas", F, [1, 1, S, 1])
    n("Greater", ["rowhas", "half"], "rowhasb")
    vi("rowhasb", B, [1, 1, S, 1])
    n("Where", ["rowhasb", "rr", "BIG"], "rowidx")
    vi("rowidx", F, [1, 1, S, 1])
    n("ReduceMin", ["rowidx"], "R", keepdims=1)               # [1,1,1,1]
    vi("R", F, [1, 1, 1, 1])

    n("ReduceMax", ["nzf"], "colhas", axes=[2], keepdims=1)   # [1,1,1,9]
    vi("colhas", F, [1, 1, 1, S])
    n("Greater", ["colhas", "half"], "colhasb")
    vi("colhasb", B, [1, 1, 1, S])
    n("Where", ["colhasb", "cc", "BIG"], "colidx")
    vi("colidx", F, [1, 1, 1, S])
    n("ReduceMin", ["colidx"], "C", keepdims=1)               # [1,1,1,1]
    vi("C", F, [1, 1, 1, 1])

    n("Add", ["R", "one"], "R1")
    vi("R1", F, [1, 1, 1, 1])
    n("Add", ["C", "one"], "C1")
    vi("C1", F, [1, 1, 1, 1])

    # --- colour scalar: max colour over non-red coloured cells ------------
    n("Sub", ["one", "redf"], "notred")             # [1,1,9,9]
    vi("notred", F, [1, 1, S, S])
    n("Mul", ["cidx", "notred"], "colorplane")      # [1,1,9,9] (red->0, bg 0)
    vi("colorplane", F, [1, 1, S, S])
    n("ReduceMax", ["colorplane"], "color", keepdims=1)       # [1,1,1,1]
    vi("color", F, [1, 1, 1, 1])

    corner_r = {0: "R", 1: "R", 2: "R1", 3: "R1"}
    corner_c = {0: "C", 1: "C1", 2: "C1", 3: "C"}

    m_acc = None
    seed_acc = None
    for d in range(4):
        cr = corner_r[d]
        ccc = corner_c[d]
        dr, dc = DELTAS[d]
        # corner indicator (rr==cr)&(cc==ccc) -> seed cell map
        n("Equal", ["rr", cr], f"rmatch{d}")        # [1,1,9,1] bool
        vi(f"rmatch{d}", B, [1, 1, S, 1])
        n("Equal", ["cc", ccc], f"cmatch{d}")       # [1,1,1,9] bool
        vi(f"cmatch{d}", B, [1, 1, 1, S])
        n("And", [f"rmatch{d}", f"cmatch{d}"], f"cornb{d}")
        vi(f"cornb{d}", B, [1, 1, S, S])
        n("Cast", [f"cornb{d}"], f"cornf{d}", to=F)
        vi(f"cornf{d}", F, [1, 1, S, S])
        if seed_acc is None:
            seed_acc = f"cornf{d}"
        else:
            nn = f"seedacc{d}"
            n("Add", [seed_acc, f"cornf{d}"], nn)
            vi(nn, F, [1, 1, S, S])
            seed_acc = nn
        # chosen flag = sum(red * corner) -> scalar
        n("Mul", ["redf", f"cornf{d}"], f"rc{d}")
        vi(f"rc{d}", F, [1, 1, S, S])
        n("ReduceSum", [f"rc{d}"], f"flag{d}", keepdims=1)    # [1,1,1,1]
        vi(f"flag{d}", F, [1, 1, 1, 1])
        # a, b (separable along row/col)
        n("Sub", ["rr", cr], f"da{d}")              # [1,1,9,1]
        vi(f"da{d}", F, [1, 1, S, 1])
        n("Sub", ["cc", ccc], f"db{d}")             # [1,1,1,9]
        vi(f"db{d}", F, [1, 1, 1, S])
        init(f"sdr{d}", np.array(float(dr), np.float32), np.float32)
        init(f"sdc{d}", np.array(float(dc), np.float32), np.float32)
        n("Mul", [f"da{d}", f"sdr{d}"], f"a{d}")    # [1,1,9,1]
        vi(f"a{d}", F, [1, 1, S, 1])
        n("Mul", [f"db{d}", f"sdc{d}"], f"b{d}")    # [1,1,1,9]
        vi(f"b{d}", F, [1, 1, 1, S])
        n("Greater", [f"a{d}", "nhalf"], f"age{d}")  # [1,1,9,1] bool
        vi(f"age{d}", B, [1, 1, S, 1])
        n("Greater", [f"b{d}", "nhalf"], f"bge{d}")  # [1,1,1,9] bool
        vi(f"bge{d}", B, [1, 1, 1, S])
        n("Sub", [f"a{d}", f"b{d}"], f"diff{d}")    # [1,1,9,9]
        vi(f"diff{d}", F, [1, 1, S, S])
        n("Abs", [f"diff{d}"], f"adiff{d}")         # [1,1,9,9]
        vi(f"adiff{d}", F, [1, 1, S, S])
        n("Less", [f"adiff{d}", "onep5"], f"band{d}")  # bool
        vi(f"band{d}", B, [1, 1, S, S])
        n("And", [f"age{d}", f"bge{d}"], f"ab{d}")   # bool [1,1,9,9]
        vi(f"ab{d}", B, [1, 1, S, S])
        n("And", [f"ab{d}", f"band{d}"], f"cond{d}")  # bool
        vi(f"cond{d}", B, [1, 1, S, S])
        n("Cast", [f"cond{d}"], f"condf{d}", to=F)
        vi(f"condf{d}", F, [1, 1, S, S])
        n("Mul", [f"condf{d}", f"flag{d}"], f"contrib{d}")    # [1,1,9,9]
        vi(f"contrib{d}", F, [1, 1, S, S])
        if m_acc is None:
            m_acc = f"contrib{d}"
        else:
            nn = f"macc{d}"
            n("Add", [m_acc, f"contrib{d}"], nn)
            vi(nn, F, [1, 1, S, S])
            m_acc = nn

    n("Add", [m_acc, seed_acc], "paintsum")         # [1,1,9,9]
    vi("paintsum", F, [1, 1, S, S])
    n("Greater", ["paintsum", "half"], "painted")   # bool
    vi("painted", B, [1, 1, S, S])

    # L9 = where(painted, color, 0) -> uint8, then Pad to 30x30 with sentinel 10
    init("zerof", np.array(0.0, np.float32), np.float32)
    n("Where", ["painted", "color", "zerof"], "Lf")  # [1,1,9,9] f
    vi("Lf", F, [1, 1, S, S])
    n("Cast", ["Lf"], "L9", to=U8)                   # [1,1,9,9] uint8
    vi("L9", U8, [1, 1, S, S])
    init("sent", np.array(10, np.uint8), np.uint8)
    n("Pad", ["L9", "padpads", "sent"], "L",
      mode="constant")                              # [1,1,30,30] uint8
    init("padpads", np.array([0, 0, 0, 0, 0, 0, crop, crop], np.int64),
         np.int64)
    vi("L", U8, [1, 1, G, G])

    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["L", "chan"], "output")             # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F, [1, 10, G, G])
    y = helper.make_tensor_value_info("output", B, [1, 10, G, G])
    g = helper.make_graph(nodes, "task034", [x], [y], inits, value_info=vinfos)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
