"""task105 (ARC-AGI 4612dd53) — "restore the missing (red) cells of a rectangular box".

Rule (from the generator task_4612dd53.py):
  A rectangular OUTLINE (perimeter of a bbox) is drawn, optionally with ONE interior
  full "cutline" — either a horizontal row (`horiz`) or a vertical column (`vert`)
  spanning the box interior.  Every cell of this figure is colored blue(1) or red(2)
  (each cell red with prob ~1/4).  In the INPUT only the BLUE cells appear; the red
  cells are erased (background 0).  In the OUTPUT every figure cell appears (blue stays
  blue, the erased ones become red(2)).

  So:  output = input, but every figure-cell that is currently empty -> red(2).
  Blue cells are unchanged; on-grid background and off-grid cells are unchanged.

  Verified exactly: bbox(blue) == bbox(figure) (corners are always part of the outline
  and at least one of each corner-pair is blue, so the blue bbox recovers the box exactly).

  figure  = perimeter(bbox)  UNION  cutline
  cutline = a single full interior row OR a single full interior column.
  Detect cutline orientation by per-row vs per-col interior-blue COUNT:
    horizontal cutline -> exactly one interior ROW has >=1 interior blue cell (count = kh),
                          every interior COL has count <= 1.
    vertical   cutline -> exactly one interior COL has count kv, every interior ROW <= 1.
    Rmax = max_i (interior-blue count in row i);  Cmax = max_j (count in col j).
    horizontal iff Rmax >= Cmax (>0)  [default H on the rare single-cell tie];
    vertical   iff Cmax >  Rmax.
    fill the row(s) with count==Rmax (horizontal) -> (cutrow == Rmax) (x) col_in
    fill the col(s) with count==Cmax (vertical)   -> row_in (x) (cutcol == Cmax)

Encoding (route the 10-ch expansion into the FREE output; never materialize [1,10,H,W]):
  Everything works on a 14x14 active canvas in fp16.  Final op:
    output = Where(red_cond[1,1,30,30], red_onehot[1,10,1,1], input)
  red_cond = figure-cell AND empty(ch0).  Where flips those empty cells to red(2) and
  leaves blue / on-grid-bg / off-grid untouched.  No box/label plane is materialized in
  10 channels; the only full-size 10-ch tensor is the FREE output.
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

# Active canvas.  The figure spans rows 1..12, cols 2..10 across all instances,
# so rows 0:HR cols 0:WC cover it with margin; smaller planes = lower memory.
HR = 13  # rows
WC = 11  # cols


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    sax = init("sax", np.array([1, 2, 3], np.int64), np.int64)

    # ---- blue plane (channel 1) on the HRxWC canvas ------------------------
    init("bl_s", np.array([1, 0, 0], np.int64), np.int64)
    init("bl_e", np.array([2, HR, WC], np.int64), np.int64)
    n("Slice", ["input", "bl_s", "bl_e", "sax"], "blue_f32")   # [1,1,HR,WC] f32
    n("Cast", ["blue_f32"], "blue", to=F16)                    # [1,1,HR,WC] f16

    # ---- bbox occupancy (fp32 for CumSum) ----------------------------------
    n("ReduceMax", ["blue_f32"], "rowhas", axes=[3], keepdims=1)   # [1,1,HR,1] f32
    n("ReduceMax", ["blue_f32"], "colhas", axes=[2], keepdims=1)   # [1,1,1,WC] f32

    # prefix/suffix OR via CumSum (>0) — no triangular-matrix initializers.
    init("AX2", np.array(2, np.int64), np.int64)
    init("AX3", np.array(3, np.int64), np.int64)
    n("CumSum", ["rowhas", "AX2"], "pref_r_s")                 # blue row <= i
    n("CumSum", ["rowhas", "AX2"], "suf_r_s", reverse=1)       # blue row >= i
    n("CumSum", ["colhas", "AX3"], "pref_c_s")                 # blue col <= j
    n("CumSum", ["colhas", "AX3"], "suf_c_s", reverse=1)       # blue col >= j

    init("ZEROH", np.array(0.0, np.float16), np.float16)
    init("ZEROF", np.array(0.0, np.float32), np.float32)
    n("Greater", ["pref_r_s", "ZEROF"], "pref_r")
    n("Greater", ["suf_r_s", "ZEROF"], "suf_r")
    n("Greater", ["pref_c_s", "ZEROF"], "pref_c")
    n("Greater", ["suf_c_s", "ZEROF"], "suf_c")
    n("And", ["pref_r", "suf_r"], "row_in_b")   # [1,1,W,1] bool  (t..b)
    n("And", ["pref_c", "suf_c"], "col_in_b")   # [1,1,1,W] bool  (l..r)
    n("Cast", ["row_in_b"], "row_in", to=F16)   # [1,1,W,1] f16 {0,1}
    n("Cast", ["col_in_b"], "col_in", to=F16)   # [1,1,1,W] f16 {0,1}

    # ---- perimeter: edges of row_in / col_in -------------------------------
    # interior_row[i] = row_in[i-1] AND row_in[i+1]   (both neighbors in box)
    # edge_row        = row_in AND NOT interior_row
    # neighbors via Pad(+1 each side) then Slice back to W.
    init("pad_r", np.array([0, 0, 1, 0, 0, 0, 1, 0], np.int64), np.int64)
    n("Pad", ["row_in", "pad_r"], "row_pad", mode="constant")  # [1,1,HR+2,1]
    # prev[i] = row_pad[i]  (== row_in[i-1]);  next[i] = row_pad[i+2]
    init("ax2", np.array([2], np.int64), np.int64)
    init("ax3", np.array([3], np.int64), np.int64)
    init("r0", np.array([0], np.int64), np.int64)
    init("rHR", np.array([HR], np.int64), np.int64)
    init("r2", np.array([2], np.int64), np.int64)
    init("rHR2", np.array([HR + 2], np.int64), np.int64)
    init("rWC", np.array([WC], np.int64), np.int64)
    init("rWC2", np.array([WC + 2], np.int64), np.int64)
    n("Slice", ["row_pad", "r0", "rHR", "ax2"], "row_prev")    # row_in[i-1]
    n("Slice", ["row_pad", "r2", "rHR2", "ax2"], "row_next")   # row_in[i+1]
    n("Mul", ["row_prev", "row_next"], "row_inner")            # both neighbors
    init("pad_c", np.array([0, 0, 0, 1, 0, 0, 0, 1], np.int64), np.int64)
    n("Pad", ["col_in", "pad_c"], "col_pad", mode="constant")  # [1,1,1,WC+2]
    n("Slice", ["col_pad", "r0", "rWC", "ax3"], "col_prev")
    n("Slice", ["col_pad", "r2", "rWC2", "ax3"], "col_next")
    n("Mul", ["col_prev", "col_next"], "col_inner")

    # edge_row = row_in * (1 - row_inner);  edge_col similarly
    init("ONEH", np.array(1.0, np.float16), np.float16)
    n("Sub", ["ONEH", "row_inner"], "row_notinner")
    n("Mul", ["row_in", "row_notinner"], "edge_row")   # [1,1,W,1]
    n("Sub", ["ONEH", "col_inner"], "col_notinner")
    n("Mul", ["col_in", "col_notinner"], "edge_col")   # [1,1,1,W]

    # ---- cutline (per-row/col interior-blue counts WITHOUT a WxW plane) -----
    # cutrow[i] = row_inner[i] * sum_j blue[i,j]*col_inner[j]
    #           = row_inner * MatMul(blue, col_inner^T)
    # cutcol[j] = col_inner[j] * sum_i blue[i,j]*row_inner[i]
    #           = col_inner * MatMul(row_inner^T, blue)
    n("Transpose", ["col_inner"], "col_inner_T", perm=[0, 1, 3, 2])  # [1,1,W,1]
    n("Transpose", ["row_inner"], "row_inner_T", perm=[0, 1, 3, 2])  # [1,1,1,W]
    n("MatMul", ["blue", "col_inner_T"], "rcnt_raw")    # [1,1,W,1] interior-blue/row
    n("MatMul", ["row_inner_T", "blue"], "ccnt_raw")    # [1,1,1,W] interior-blue/col
    n("Mul", ["rcnt_raw", "row_inner"], "cutrow")       # gate to interior rows
    n("Mul", ["ccnt_raw", "col_inner"], "cutcol")       # gate to interior cols
    n("ReduceMax", ["cutrow"], "Rmax", axes=[2], keepdims=1)      # [1,1,1,1]
    n("ReduceMax", ["cutcol"], "Cmax", axes=[3], keepdims=1)      # [1,1,1,1]

    # horizontal active iff Rmax >= Cmax AND Rmax > 0  (default H on tie)
    n("Greater", ["Rmax", "ZEROH"], "has_cut")          # Rmax>0  (any cutline)
    n("Less", ["Rmax", "Cmax"], "R_lt_C")               # Rmax < Cmax
    n("Not", ["R_lt_C"], "R_ge_C")                      # Rmax >= Cmax
    n("And", ["has_cut", "R_ge_C"], "is_h")             # horizontal cutline
    n("Less", ["Rmax", "Cmax"], "is_v")                 # vertical cutline (Cmax>Rmax)

    # cutline row = (cutrow == Rmax) AND is_h ; cut col = (cutcol == Cmax) AND is_v
    n("Equal", ["cutrow", "Rmax"], "row_is_max")        # [1,1,W,1] bool
    n("And", ["row_is_max", "is_h"], "cut_row_b")       # broadcast is_h
    n("Equal", ["cutcol", "Cmax"], "col_is_max")        # [1,1,1,W] bool
    n("And", ["col_is_max", "is_v"], "cut_col_b")
    n("Cast", ["cut_row_b"], "cut_row", to=F16)         # [1,1,W,1]
    n("Cast", ["cut_col_b"], "cut_col", to=F16)         # [1,1,1,W]

    # ---- figure = perimeter UNION cutline, as 2 rank-1 outer products ------
    # rterm = edge_row + cut_row  (rows to fill full-width)
    # cterm = edge_col + cut_col  (cols to fill full-height)
    # figure = (rterm (x) col_in) OR (row_in (x) cterm)
    n("Add", ["edge_row", "cut_row"], "rterm")          # [1,1,W,1]
    n("Add", ["edge_col", "cut_col"], "cterm")          # [1,1,1,W]
    n("Mul", ["rterm", "col_in"], "fig_a")              # [1,1,W,W]
    n("Mul", ["row_in", "cterm"], "fig_b")              # [1,1,W,W]
    n("Add", ["fig_a", "fig_b"], "fig_s")               # [1,1,W,W]
    n("Greater", ["fig_s", "ZEROH"], "figure")          # [1,1,HR,WC] bool
    # red = figure AND NOT blue.  Every figure cell is either blue or (erased)
    # red, and figure lies inside the bbox (on-grid), so no separate bg gate is
    # needed.  notblue = blue < 0.5 in one Less op.
    init("HALFH", np.array(0.5, np.float16), np.float16)
    n("Less", ["blue", "HALFH"], "notblue")             # [1,1,HR,WC] bool
    n("And", ["figure", "notblue"], "red_b")            # [1,1,HR,WC] bool

    # ---- pad to 30x30 -> cond ----------------------------------------------
    n("Cast", ["red_b"], "red_u8", to=U8)
    init("pads30", np.array([0, 0, 0, 0, 0, 0, 30 - HR, 30 - WC], np.int64), np.int64)
    init("ZEROU8", np.array(0, np.uint8), np.uint8)
    n("Pad", ["red_u8", "pads30", "ZEROU8"], "red30", mode="constant")
    n("Cast", ["red30"], "cond", to=BOOL)               # [1,1,30,30] bool

    # ---- red one-hot (color 2) ; output = Where(cond, red_oh, input) -------
    oh = np.zeros((1, 10, 1, 1), np.float32)
    oh[0, 2, 0, 0] = 1.0
    init("red_oh", oh, np.float32)
    n("Where", ["cond", "red_oh", "input"], "output")

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task105", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
