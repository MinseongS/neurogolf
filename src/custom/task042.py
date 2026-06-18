"""Task 042 (ARC-AGI 22233c11) — extend a green diagonal staircase with cyan.

Rule (from the generator): the input has one or two diagonal "staircases" of two
m x m green blocks (m = magnify, data-dependent in {1,2,3}).  A staircase is a
top green block at (r,col) and a bottom green block at (r+m, col±m) (diagonally
corner-touching).  The OUTPUT keeps all green and adds two cyan m x m blocks that
continue the zigzag: one above the top block and one below the bottom block.

Unified, fully separable rule (verified 0/5000):
  let s = +1 if the bottom block is to the RIGHT of the top block (flip0), else -1.
    cyan_top    = top    block translated by (-m, +2m*s)
    cyan_bottom = bottom block translated by (+m, -2m*s)
  A green cell is a TOP cell iff a green block sits diagonally below it
  (at (r+m,c+m) -> s=+1, or (r+m,c-m) -> s=-1); BOTTOM iff a green block sits
  diagonally above it (at (r-m,c-m) -> s=+1, or (r-m,c+m) -> s=-1).

Each cyan class collapses to an AND of two reads of the green plane (no separate
neighbour/gated planes ever materialise):
    cyan_top_r[i,j] = Gd[i+m, j-2m] & Gd[i+2m, j-m]
    cyan_top_l[i,j] = Gd[i+m, j+2m] & Gd[i+2m, j+m]
    cyan_bot_l[i,j] = Gd[i-m, j+2m] & Gd[i-2m, j+m]
    cyan_bot_r[i,j] = Gd[i-m, j-2m] & Gd[i-2m, j-m]
cyan = OR of the four classes (disjoint in practice).

ONNX construction (everything on the 10x10 active grid; output stays BOOL/free):
  * green plane Gd = ONE Slice of channel 3 + the 10x10 corner, bool [1,1,10,10]
    (never a 30x30 plane).
  * m in {1,2,3} from the green pixel COUNT alone (ranges don't overlap: m=1 ->
    2..4 px, m=2 -> 8, m=3 -> 18, since m>=2 forces a single staircase):
    cnt = ReduceSum(grn10); m = 1 + (cnt>5) + (cnt>12).
  * a read Gd[i+a*m, j+b*m] is two Gathers of a zero-padded 12x12 plane; per-m
    source-index vectors come from constant [4,10] tables Gather(table, m); table
    values are clamped into the zero pad so any off-grid read returns 0.  Row
    Gathers are cached (4 distinct row tables) so each serves two reads.
  * label L (uint8 10x10) = 3 where green, 8 where cyan; Pad to 30x30 (sentinel
    99 off-grid); free BOOL output = Equal(L, arange[0..9]).

Memory ~4990B: 900B padded 30x30 uint8 label (output-shaping floor) + 400B fp32
entry slice (the one fp32 plane) + the small 10x10/12x12 working set.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
I64 = TensorProto.INT64
U8 = TensorProto.UINT8
B = TensorProto.BOOL

PAD = 1          # zero border (off-grid reads clamp into the pad)
P = 10 + 2 * PAD  # padded plane size


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- green plane on the 10x10 active grid ----
    # input is [1,10,30,30]; channel 3 = green, top-left 10x10 holds everything.
    # Slice channel 3 AND the 10x10 corner in ONE op (never materialise a 30x30 plane).
    init("st", np.array([3, 0, 0], np.int64), np.int64)
    init("en", np.array([4, 10, 10], np.int64), np.int64)
    init("ax123", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "st", "en", "ax123"], "grn10")       # [1,1,10,10] f32
    init("zf", np.array(0.0, np.float32), np.float32)
    n("Greater", ["grn10", "zf"], "Gd")                       # [1,1,10,10] bool

    # ---- detect m in {1,2,3} from the green pixel COUNT (ranges don't overlap) ----
    # m=1 -> 2..4 green px; m=2 -> 8; m=3 -> 18 (single staircase for m>=2).
    # m = 1 + (cnt>5) + (cnt>12).
    # green is one-hot {0,1} on the slice -> count = ReduceSum(grn10), reusing the
    # fp32 entry plane (no extra cast plane).
    n("ReduceSum", ["grn10"], "gcnt", axes=[0, 1, 2, 3], keepdims=0)  # scalar f32
    init("t5", np.array(5.0, np.float32), np.float32)
    init("t12", np.array(12.0, np.float32), np.float32)
    n("Greater", ["gcnt", "t5"], "has2b")
    n("Greater", ["gcnt", "t12"], "has3b")
    n("Cast", ["has2b"], "has2", to=I64)
    n("Cast", ["has3b"], "has3", to=I64)
    init("one64", np.array(1, np.int64), np.int64)
    n("Add", ["one64", "has2"], "m_tmp")
    n("Add", ["m_tmp", "has3"], "m_v")                        # int64 in {1,2,3}
    # ensure a true 0-d scalar so table-Gather returns a [10] vector (not [1,10])
    n("Squeeze", ["m_v"], "m")

    # ---- zero-padded green plane (22x22) for all shifts ----
    n("Cast", ["Gd"], "Gd_u8", to=U8)
    init("padcfg", np.array([0, 0, PAD, PAD, 0, 0, PAD, PAD], np.int64), np.int64)
    init("u0", np.array(0, np.uint8), np.uint8)
    n("Pad", ["Gd_u8", "padcfg", "u0"], "Gpad", mode="constant")  # [1,1,22,22] u8

    # ---- per-m source-index tables ----------------------------------------
    # For a shift by (dr,dc): out[i] reads padded index (i + PAD - delta) where
    # delta is the shift in cells.  delta = d_units * m.  Build a [4,10] table
    # indexed by m (rows 0..3, only 1..3 used) for each needed d_units.
    base = np.arange(10)

    def idx_table(d_units, name):
        tbl = np.zeros((4, 10), np.int64)
        for mm in range(1, 4):
            tbl[mm] = base + PAD - d_units * mm
        # clamp out-of-pad indices into the zero-padding band (safe: off-grid
        # reads only ever clamp onto a guaranteed-zero pad cell; valid reads
        # land in the real [PAD, PAD+9] band and are never clamped).
        tbl = np.clip(tbl, 0, P - 1)
        init(name, tbl, np.int64)
        n("Gather", [name, "m"], name + "_v", axis=0)        # [10] int64
        return name + "_v"

    # row index vectors
    rv_m = idx_table(1, "tr_pm")     # shift dr = +m
    rv_nm = idx_table(-1, "tr_nm")   # dr = -m
    rv_2m = idx_table(2, "tr_p2m")   # dr = +2m
    rv_n2m = idx_table(-2, "tr_n2m") # dr = -2m
    # col index vectors (same tables, reuse)
    cv_m = rv_m
    cv_nm = rv_nm
    cv_2m = rv_2m
    cv_n2m = rv_n2m

    _rowcache = {}

    def shift(plane, rv, cv, out):
        # plane padded [1,1,22,22]; gather rows (cached per row-table) then cols.
        key = (plane, rv)
        if key not in _rowcache:
            rname = "rg_" + rv + "_" + plane
            n("Gather", [plane, rv], rname, axis=2)          # [1,1,10,22]
            _rowcache[key] = rname
        n("Gather", [_rowcache[key], cv], out, axis=3)
        return out

    # ---- cyan via double-shifts of the single padded green plane Gpad --------
    # Each cyan class = AND of two reads of Gd (no separate gated/neighbour planes):
    #   cyan_top_r[i,j] = Gd[i+m, j-2m] & Gd[i+2m, j-m]
    #   cyan_top_l[i,j] = Gd[i+m, j+2m] & Gd[i+2m, j+m]
    #   cyan_bot_l[i,j] = Gd[i-m, j+2m] & Gd[i-2m, j+m]
    #   cyan_bot_r[i,j] = Gd[i-m, j-2m] & Gd[i-2m, j-m]
    # A read Gd[i + a*m, j + b*m] = shift("Gpad", row d_units=a, col d_units=b)
    # (src index i+PAD + a*m -> the +a table).
    rtab = {1: rv_m, -1: rv_nm, 2: rv_2m, -2: rv_n2m}
    ctab = {1: cv_m, -1: cv_nm, 2: cv_2m, -2: cv_n2m}

    def read(a, b, out):                # Gd[i+a*m, j+b*m]
        return shift("Gpad", rtab[a], ctab[b], out)

    def cyan_class(a1, b1, a2, b2, out):
        r1 = read(a1, b1, out + "_p")   # uint8
        r2 = read(a2, b2, out + "_q")   # uint8
        n("Cast", [r1], out + "_pb", to=B)
        n("Cast", [r2], out + "_qb", to=B)
        n("And", [out + "_pb", out + "_qb"], out)
        return out

    cyan_class(1, -2, 2, -1, "cy1")     # top_r
    cyan_class(1, 2, 2, 1, "cy2")       # top_l
    cyan_class(-1, 2, -2, 1, "cy3")     # bot_l
    cyan_class(-1, -2, -2, -1, "cy4")   # bot_r
    n("Or", ["cy1", "cy2"], "cyA")
    n("Or", ["cy3", "cy4"], "cyB")
    n("Or", ["cyA", "cyB"], "cyan")                          # [1,1,10,10] bool

    # ---- label plane (uint8): 3 where green, 8 where cyan, else 0 ----
    init("u3", np.array(3, np.uint8), np.uint8)
    init("u8c", np.array(8, np.uint8), np.uint8)
    init("u0b", np.array(0, np.uint8), np.uint8)
    n("Where", ["Gd", "u3", "u0b"], "Lg")                    # green=3
    n("Where", ["cyan", "u8c", "Lg"], "L10")                 # cyan=8 over it

    # ---- pad to 30x30 (sentinel 99 off-grid) and Equal -> BOOL output ----
    init("pad30", np.array([0, 0, 0, 0, 0, 0, 20, 20], np.int64), np.int64)
    init("u99", np.array(99, np.uint8), np.uint8)
    n("Pad", ["L10", "pad30", "u99"], "L", mode="constant")  # [1,1,30,30] u8
    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["L", "chan"], "output")                      # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task042", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
