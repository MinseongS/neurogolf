"""task163 (ARC-AGI 6d0160f0) — copy the yellow cell's block into the block at
the yellow cell's mini-position.

Rule (from the generator):
  Grid is 11x11 "hollywood_squares": a 3x3 arrangement of 3x3 mini-blocks
  separated by single gray(5) lines at row/col index 3 and 7 (i.e. r%4==3 or
  c%4==3). Block (R,C) occupies rows R*4..R*4+2, cols C*4..C*4+2. Cells are
  filled with random colours; exactly one cell is yellow(4), at block (R,C) and
  mini-position (mr,mc) within that block.
  OUTPUT = the same gray-square grid, blank, EXCEPT the 3x3 contents of input
  block (R,C) are stamped into the output block at position (mr,mc) (same
  intra-block offsets).

Encoding (Tier B — separable block copy as a double boolean MatMul):
  V = colour-index plane (1x1 [0..9] Conv over the one-hot input), sliced to the
  11x11 active canvas and cast fp16.
  Yellow position: yr = Σ_o o*(row o has yellow), yc likewise (yellow is unique).
  mr = yr%4, mc = yc%4 (fp16 Mod, integer-exact); source block top-left
  R4 = yr-mr, C4 = yc-mc; target block top-left mr4=mr*4? no: target block index
  is (mr,mc) so its top-left is mr*4 etc. row offset = R4 - mr*4.
  copied = Rmat @ V @ CmatT, where
     Rmat[o,s] = (s == o + off_r) AND (mt_r <= o <= mt_r+2)   (mt_r = mr*4)
     CmatT[s,o] = (s == o + off_c) AND (mt_c <= o <= mt_c+2)  (mt_c = mc*4)
  copied is the source-block colour index placed at the target block, 0 else
  (and 0 on grid lines, since target cells are content cells only).
  Label L = copied + gray_background (gray=5 on lines, 0 elsewhere; no overlap).
  Pad to 30x30 with sentinel 99, Equal(L, arange[0..9]) -> BOOL output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64

W = 11  # active canvas (grid is exactly 11x11)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- colour-index plane V = Σ k * input_k  (1x1 Conv on full input) -----
    kw = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("convW", kw, np.float32)
    n("Conv", ["input", "convW"], "Vfull")          # [1,1,30,30] f32
    # slice to active 11x11
    init("v_s", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("v_e", np.array([1, 1, W, W], np.int64), np.int64)
    init("v_ax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["Vfull", "v_s", "v_e", "v_ax"], "V")  # [1,1,W,W] f32
    n("Cast", ["V"], "V16", to=F16)                    # [1,1,W,W] f16

    # ---- yellow position: colour index 4 is unique (only the yellow cell) ---
    init("FOURH", np.array(4.0, np.float16), np.float16)
    n("Equal", ["V16", "FOURH"], "yelb")             # bool [1,1,W,W]
    n("Cast", ["yelb"], "yel", to=F32)               # f32  [1,1,W,W]

    # per-row presence -> yr = Σ o * presence_row(o)
    n("ReduceMax", ["yel"], "rowpres", axes=[3], keepdims=1)  # [1,1,W,1]
    n("ReduceMax", ["yel"], "colpres", axes=[2], keepdims=1)  # [1,1,1,W]
    ramp2 = np.arange(W, dtype=np.float32).reshape(1, 1, W, 1)
    ramp3 = np.arange(W, dtype=np.float32).reshape(1, 1, 1, W)
    init("ramp2", ramp2, np.float32)
    init("ramp3", ramp3, np.float32)
    n("Mul", ["rowpres", "ramp2"], "yrcontrib")
    n("ReduceSum", ["yrcontrib"], "yr", axes=[2, 3], keepdims=1)  # [1,1,1,1]
    n("Mul", ["colpres", "ramp3"], "yccontrib")
    n("ReduceSum", ["yccontrib"], "yc", axes=[2, 3], keepdims=1)  # [1,1,1,1]

    # ---- mr=yr%4, mc=yc%4 ; offsets ; target block top-left -----------------
    init("FOUR", np.array(4.0, np.float32), np.float32)
    n("Mod", ["yr", "FOUR"], "mr", fmod=1)   # yr % 4  (f32 exact for ints)
    n("Mod", ["yc", "FOUR"], "mc", fmod=1)
    n("Sub", ["yr", "mr"], "R4")             # source block top row = yr - mr
    n("Sub", ["yc", "mc"], "C4")             # source block top col
    n("Mul", ["mr", "FOUR"], "mt_r")         # target block top row = mr*4
    n("Mul", ["mc", "FOUR"], "mt_c")
    n("Sub", ["R4", "mt_r"], "off_r")        # input_row = output_row + off_r
    n("Sub", ["C4", "mt_c"], "off_c")

    # ---- build Rmat / CmatT -------------------------------------------------
    # Rmat[o(axis2), s(axis3)] = (s == o + off_r) AND (mt_r <= o <= mt_r+2)
    # CmatT[s(axis2), o(axis3)] = (s == o + off_c) AND (mt_c <= o <= mt_c+2)
    init("THREEF", np.array(3.0, np.float32), np.float32)

    def selmat(off, mt, out_axis, tag):
        # out index `o` on `out_axis`, source index `s` on the other axis.
        ov = "ramp2" if out_axis == 2 else "ramp3"
        sv = "ramp3" if out_axis == 2 else "ramp2"
        # cond1 = Equal(s, o + off)  -> broadcasts to [1,1,W,W]
        n("Add", [ov, off], f"tgt_{tag}")
        n("Equal", [sv, f"tgt_{tag}"], f"eq_{tag}")
        # cond2 = (o >= mt) AND (o < mt+3)
        n("Less", [ov, mt], f"below_{tag}")
        n("Not", [f"below_{tag}"], f"ge_{tag}")            # o >= mt
        n("Add", [mt, "THREEF"], f"mtlim_{tag}")           # mt+3
        n("Less", [ov, f"mtlim_{tag}"], f"le_{tag}")       # o < mt+3
        n("And", [f"ge_{tag}", f"le_{tag}"], f"inr_{tag}")
        n("And", [f"eq_{tag}", f"inr_{tag}"], f"mb_{tag}")  # bool [1,1,W,W]
        n("Cast", [f"mb_{tag}"], f"m_{tag}", to=F16)
        return f"m_{tag}"

    Rmat = selmat("off_r", "mt_r", 2, "R")
    CmatT = selmat("off_c", "mt_c", 3, "C")

    # ---- copied = Rmat @ V @ CmatT -----------------------------------------
    n("MatMul", [Rmat, "V16"], "rowmapped")     # f16 [1,1,W,W]
    n("MatMul", ["rowmapped", CmatT], "copied")  # f16 [1,1,W,W]

    # ---- gray background (FIXED): 5 where line (o%4==3) ----------------------
    bg = np.zeros((1, 1, W, W), np.float16)
    for r in range(W):
        for c in range(W):
            if r % 4 == 3 or c % 4 == 3:
                bg[0, 0, r, c] = 5.0
    init("bg", bg, np.float16)
    n("Add", ["copied", "bg"], "L16")            # f16 label, lines=5 + copied

    n("Cast", ["L16"], "Lu", to=U8)              # uint8 [1,1,W,W]
    init("S99", np.array(99, np.uint8), np.uint8)
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64), np.int64)
    n("Pad", ["Lu", "pads", "S99"], "L", mode="constant")  # [1,1,30,30] uint8

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")          # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task163", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
