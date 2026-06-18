"""task333 (ARC-AGI d43fd935) — beams shot toward a green 2x2 box.

Rule (from the generator):
  The grid (always 10x10) holds ONE green(3) 2x2 box at (boxrow,boxcol) and a set
  of coloured pixels.  For each coloured pixel:
    * the pixel itself is always drawn;
    * if the pixel lies in one of the box's two ROWS (row in {boxrow,boxrow+1}),
      it shoots a horizontal beam of its colour TOWARD the box, filling every cell
      between the pixel and the box (stopping at the green edge);
    * if it lies in one of the box's two COLUMNS (col in {boxcol,boxcol+1}),
      it shoots a vertical beam TOWARD the box (stopping at the green edge).
  Per the generator's flags, at most ONE pixel sits on each of the four sides
  (left / right / up / down).  So output = input with four beam segments filled.

Floor-break encoding (single uint8 label map + final Equal; no [1,10,30,30] tensor):
  Work on a 10x10 canvas. C[1,1,M,M] = colour index via 1x1 arange Conv (ch0 weight
  0), cast to fp16 + crop.  Green plane G = (C==3); per-row/per-col green presence
  give the box-row / box-col masks and the boxrow/boxcol scalars.  Each directional
  beam is a triangular prefix/suffix MatMul of the colour plane (a single value's
  prefix sum IS that value).  The four disjoint beams are combined by a priority
  Where-CHAIN whose BASE CASE is the raw colour plane C (so the dot survives and the
  "beam>0 ? beam : C" select + all Adds fold into the chain for free).  Each beam's
  region restriction is a BOOL Where condition (separable row x col side vectors
  broadcast to [M,M]) — no fp16 *region Mul planes.  Green never enters a gated
  prefix sum (gated cells lie strictly before the barrier), so no green removal.
  L = Cast(chain, uint8); Pad to [1,1,30,30] sentinel 10; free BOOL output =
  Equal(L, arange[1,10,1,1]).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F = TensorProto.FLOAT
H = TensorProto.FLOAT16
U8 = TensorProto.UINT8
I64 = TensorProto.INT64
B = TensorProto.BOOL
N = 30
M = 10  # working canvas (grid is always 10x10)


def build(task):
    inits, nodes, _seen = [], [], set()

    def init(name, arr, dtype):
        if name in _seen:
            return name
        _seen.add(name)
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    init("one", np.array(1.0, np.float16), np.float16)
    init("linev", np.arange(M, dtype=np.float16).reshape(1, 1, M, 1), np.float16)
    init("lineh", np.arange(M, dtype=np.float16).reshape(1, 1, 1, M), np.float16)

    # ----- colour label map C[1,1,M,M] via 1x1 arange Conv (ch0 weight 0) -------
    wc = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("kcolor", wc, np.float32)
    n("Conv", ["input", "kcolor"], "Cc")          # [1,1,30,30] fp32 colour index
    init("c_starts", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("c_ends", np.array([1, 1, M, M], np.int64), np.int64)
    init("c_axes", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["Cc", "c_starts", "c_ends", "c_axes"], "Ccrop")  # [1,1,M,M] fp32
    n("Cast", ["Ccrop"], "C", to=H)               # [1,1,M,M] fp16 colour index

    # ----- green plane G = (C==3) -----------------------------------------------
    init("three", np.array(3.0, np.float16), np.float16)
    n("Equal", ["C", "three"], "G_b")             # [1,1,M,M] bool
    n("Cast", ["G_b"], "G", to=H)                 # [1,1,M,M] fp16

    # per-row / per-col green presence
    n("ReduceMax", ["G"], "grow", axes=[3], keepdims=1)   # [1,1,M,1]
    n("ReduceMax", ["G"], "gcol", axes=[2], keepdims=1)   # [1,1,1,M]

    # boxrow / boxcol = min index with green
    init("big", np.array(100.0, np.float16), np.float16)

    def lo_index(mask, axisvec, nm, redaxis):
        n("Sub", ["one", mask], nm + "_inv")
        n("Mul", [nm + "_inv", "big"], nm + "_pen")
        n("Add", [axisvec, nm + "_pen"], nm + "_w")
        n("ReduceMin", [nm + "_w"], nm, axes=[redaxis], keepdims=1)
        return nm

    lo_index("grow", "linev", "boxrow", 2)   # [1,1,1,1]
    lo_index("gcol", "lineh", "boxcol", 3)
    n("Add", ["boxcol", "one"], "boxcol1")
    n("Add", ["boxrow", "one"], "boxrow1")

    # region side masks over the line coordinate (broadcastable, separable) ------
    n("Less", ["lineh", "boxcol"], "leftcol")       # [1,1,1,M]
    n("Greater", ["lineh", "boxcol1"], "rightcol")  # [1,1,1,M]
    n("Less", ["linev", "boxrow"], "uprow")         # [1,1,M,1]
    n("Greater", ["linev", "boxrow1"], "downrow")   # [1,1,M,1]

    # box-row / box-col bool masks (green spans exactly the 2 box rows / cols)
    n("Cast", ["grow"], "growb", to=B)              # [1,1,M,1]
    n("Cast", ["gcol"], "gcolb", to=B)              # [1,1,1,M]

    # ----- directional fills via triangular MatMuls of the RAW colour plane -----
    # Feeding the raw C (not a box-row-restricted copy) is safe: the Where chain
    # below only writes a beam where (side AND box-line) holds, so any carry that
    # spills onto a non-box row/col is never selected.  Saves the two restrict Muls.
    Utri = np.triu(np.ones((M, M), np.float16))   # Utri[a,b]=1 if a<=b
    Ltri = np.tril(np.ones((M, M), np.float16))   # Ltri[a,b]=1 if a>=b
    init("Utri", Utri, np.float16)
    init("Ltri", Ltri, np.float16)
    n("MatMul", ["C", "Utri"], "leftfill")        # carry rightward over cols
    n("MatMul", ["C", "Ltri"], "rightfill")       # carry leftward over cols
    n("MatMul", ["Ltri", "C"], "upfill")          # carry downward over rows
    n("MatMul", ["Utri", "C"], "downfill")        # carry upward over rows

    # ----- combine via priority Where-CHAIN -------------------------------------
    # On a box row the ONLY coloured cells in the left region belong to the left
    # beam, so where fill==0 the colour plane C is also 0 — the condition needs
    # only (side AND box-line), no separate fill>0 test.
    def beam(fill, sidemask, linemask, prev, tag):
        n("And", [sidemask, linemask], tag + "_cond")
        n("Where", [tag + "_cond", fill, prev], tag + "_out")
        return tag + "_out"

    base = "C"
    base = beam("leftfill", "leftcol", "growb", base, "lb")
    base = beam("rightfill", "rightcol", "growb", base, "rb")
    base = beam("upfill", "uprow", "gcolb", base, "ub")
    base = beam("downfill", "downrow", "gcolb", base, "db")
    n("Cast", [base], "L10", to=U8)               # [1,1,M,M] uint8

    # ----- Pad to [1,1,30,30] with sentinel 10 (off-grid) ----------------------
    init("u10", np.array(10, np.uint8), np.uint8)
    init("padcfg", np.array([0, 0, 0, 0, 0, 0, N - M, N - M], np.int64), np.int64)
    n("Pad", ["L10", "padcfg", "u10"], "L", mode="constant")  # [1,1,N,N]

    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["L", "chan"], "output")           # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F, [1, 10, N, N])
    y = helper.make_tensor_value_info("output", B, [1, 10, N, N])
    g = helper.make_graph(nodes, "task333", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
