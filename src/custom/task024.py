"""task024 (ARC-AGI 178fcbfb) — "draw lines from scattered seed pixels".

Rule (from the generator task_178fcbfb.py), verified exactly (0/6000 fresh fails):
  A width x height grid (each in 6..15) sits at the top-left of the 30x30 canvas.
  Scattered seed pixels of colours {2(red), 1(blue), 3(green)}, each on a distinct
  ROW (rows sampled without replacement) and an arbitrary column:
    - colour 2 (red)   : fill that pixel's COLUMN entirely with red (vertical line).
    - colour 1 / 3     : fill that pixel's ROW entirely with that colour (horizontal line).
  At a crossing of a horizontal line and a red column the HORIZONTAL colour wins
  (verified: 12944/12944 crossings -> horizontal colour).  Off-grid is all-zero.

  Per-cell closed form, priority horizontal > red-col > in-grid-bg, all SEPARABLE
  into a row-condition AND a col-condition (per output channel):
    ch3 (green) = grn_row[r]            AND colany[c]
    ch1 (blue)  = blue_row[r]           AND colany[c]
    ch2 (red)   = (nothoriz[r]&rowany[r]) AND red_col[c]
    ch0 (bg)    = (nothoriz[r]&rowany[r]) AND (~red_col[c] & colany[c])
    ch4..9      = always 0
  where for row r: grn_row/blue_row = a green/blue seed in row r; nothoriz = neither;
  rowany = row r is in-grid;  and for col c: red_col = a red seed in col c;
  colany = col c is in-grid.  The priority (horiz beats red) is enforced by gating
  red/bg with nothoriz (no explicit subtraction needed).

Encoding (Tier A separable, NO full-canvas intermediate at all):
  rowmax = ReduceMax(input, axes=[3]) [1,10,30,1] ; colmax = ReduceMax(input,[2]) [1,10,1,30]
  build RC[1,10,30,1] (per-channel row condition) and CC[1,10,1,30] (col condition)
  by concatenating tiny per-channel bool vectors, then  output = And(RC, CC)  -> the
  only [1,10,30,30] tensor is the FREE BOOL output.  Dominant intermediate ~600B.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64

N = 30


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- per-channel row/col occupancy of the FREE input -------------------
    n("ReduceMax", ["input"], "rowmax_f", axes=[3], keepdims=1)   # [1,10,30,1] f32
    n("ReduceMax", ["input"], "colmax_f", axes=[2], keepdims=1)   # [1,10,1,30] f32

    init("ZF", np.array(0.0, np.float32), np.float32)

    def chslice(src, ch, out):
        s = init(out + "_s", np.array([ch], np.int64), np.int64)
        e = init(out + "_e", np.array([ch + 1], np.int64), np.int64)
        a = init(out + "_a", np.array([1], np.int64), np.int64)
        n("Slice", [src, s, e, a], out)
        return out

    # row conditions (tiny [1,1,30,1]) — slice the fp32 reduction then Greater
    chslice("rowmax_f", 1, "bluerow_f")  # [1,1,30,1] f32
    chslice("rowmax_f", 3, "grnrow_f")
    n("Greater", ["bluerow_f", "ZF"], "blue_row")   # bool
    n("Greater", ["grnrow_f", "ZF"], "grn_row")     # bool
    # rowany = any channel present in row r
    n("ReduceMax", ["rowmax_f"], "rowany_f", axes=[1], keepdims=1)  # [1,1,30,1] f32
    n("Greater", ["rowany_f", "ZF"], "rowany")                      # [1,1,30,1] bool
    # nothoriz = NOT (blue_row OR grn_row)
    n("Or", ["blue_row", "grn_row"], "horiz")     # [1,1,30,1] bool
    n("Not", ["horiz"], "nothoriz")               # [1,1,30,1] bool
    n("And", ["nothoriz", "rowany"], "redrow")    # in-grid & not-horiz row  [1,1,30,1]

    # col conditions (tiny [1,1,1,30])
    chslice("colmax_f", 2, "redcol_f")   # [1,1,1,30] f32
    n("Greater", ["redcol_f", "ZF"], "red_col")                    # [1,1,1,30] bool
    n("ReduceMax", ["colmax_f"], "colany_f", axes=[1], keepdims=1)  # [1,1,1,30] f32
    n("Greater", ["colany_f", "ZF"], "colany")                      # [1,1,1,30] bool
    n("Not", ["red_col"], "notred")               # [1,1,1,30] bool
    n("And", ["notred", "colany"], "bgcol")       # in-grid & not-red col   [1,1,1,30]

    # zero pad vectors
    init("zrow", np.zeros((1, 1, N, 1), bool), bool)   # [1,1,30,1] all-false
    init("zcol", np.zeros((1, 1, 1, N), bool), bool)   # [1,1,1,30] all-false

    # ---- assemble RC[1,10,30,1] and CC[1,10,1,30] --------------------------
    # channel order 0..9
    row_parts = ["redrow", "blue_row", "redrow", "grn_row",
                 "zrow", "zrow", "zrow", "zrow", "zrow", "zrow"]
    col_parts = ["bgcol", "colany", "red_col", "colany",
                 "zcol", "zcol", "zcol", "zcol", "zcol", "zcol"]
    n("Concat", row_parts, "RC", axis=1)   # [1,10,30,1] bool
    n("Concat", col_parts, "CC", axis=1)   # [1,10,1,30] bool

    # ---- output = And(RC, CC) : FREE [1,10,30,30] bool --------------------
    n("And", ["RC", "CC"], "output")

    x = helper.make_tensor_value_info("input", F32, [1, 10, N, N])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, N, N])
    graph = helper.make_graph(nodes, "task024", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
