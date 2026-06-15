"""Task 293 (ARC-AGI ba97ae07): swap line colours at the crossing.

Rule: two lines (H-band and V-band) cross; input draws one on top so its colour
shows at the intersection; output swaps so the OTHER colour shows there.
Outside the intersection the lines are unchanged.

Strategy (no [1,10,30,30] intermediates):
  H-row: rows fully covered (coloured_per_row > 2.5; V-only rows ≤ 2 thick).
  V-col: cols with NO black cells AND has coloured cells
         (black_per_col < 0.5 AND coloured_per_col > 0.5).

  intersection = H_row AND V_col  → [1,1,30,30] bool  (900 B)

  To find which colour shows at intersection:
    hrmax[c] = 1 iff colour c appears anywhere in H-rows  → [1,10,1,1]
    vcmax[c] = 1 iff colour c appears anywhere in V-cols  → [1,10,1,1]
    inter_color[c] = hrmax[c] AND vcmax[c]               → intersection colour
    all_colors[c]  = hrmax[c] OR  vcmax[c]               → H-color + V-color
    replacement    = all_colors - inter_color             → the OTHER colour

  output = Where(intersection, replacement, input)        → free [1,10,30,30]

Memory ledger (fp32 itemsize 4, bool 1):
  row_col_cnt [1,1,30,1]=120  Wrow_col params=300f
  col_col_cnt [1,1,1,30]=120  Wcol_col params=300f
  col_blk_cnt [1,1,1,30]=120  Wcol_blk params=300f
  H_row       [1,1,30,1]=30B
  col_has_col [1,1,1,30]=30B  col_no_blk [1,1,1,30]=30B  V_col [1,1,1,30]=30B
  inter_bool  [1,1,30,30]=900B
  row_max     [1,10,30,1]=1200B  H_row_f [1,1,30,1]=120B  row_max_masked [1,10,30,1]=1200B
  hrmax       [1,10,1,1]=40B
  col_max     [1,10,1,30]=1200B  V_col_f [1,1,1,30]=120B  col_max_masked [1,10,1,30]=1200B
  vcmax       [1,10,1,1]=40B
  hrmax_b vcmax_b inter_color_b all_col_b  4×[1,10,1,1]bool=40B
  inter_color_f all_col_f replacement  3×[1,10,1,1]float=120B
  ≈ 5780 B memory + ~3800 B params → ~25 - ln(9580) ≈ 15.84 pts
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..builders import _model


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype=np.float32):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **kw):
        nodes.append(helper.make_node(op, ins, [out], **kw))
        return out

    # ── Kernels ───────────────────────────────────────────────────────────────
    # Per-row: sum coloured (ch1-9) across 30 cols → [1,1,30,1]
    Wrow = np.zeros((1, 10, 1, 30), np.float32)
    Wrow[0, 1:, 0, :] = 1.0
    init("Wrow", Wrow)

    # Per-col: sum coloured (ch1-9) across 30 rows → [1,1,1,30]
    Wcol = np.zeros((1, 10, 30, 1), np.float32)
    Wcol[0, 1:, :, 0] = 1.0
    init("Wcol", Wcol)

    # Per-col: sum black (ch0) across 30 rows → [1,1,1,30]
    Wblk = np.zeros((1, 10, 30, 1), np.float32)
    Wblk[0, 0, :, 0] = 1.0
    init("Wblk", Wblk)

    init("thr25", np.array([2.5], np.float32))
    init("thr05", np.array([0.5], np.float32))

    # ── H-row detection (coloured count > 2.5; V-only rows ≤ 2) ─────────────
    n("Conv", ["input", "Wrow"], "row_col_cnt")          # [1,1,30,1]
    n("Greater", ["row_col_cnt", "thr25"], "H_row")      # [1,1,30,1] bool

    # ── V-col detection (has coloured cells AND no black cells) ──────────────
    n("Conv", ["input", "Wcol"], "col_col_cnt")          # [1,1,1,30]
    n("Conv", ["input", "Wblk"], "col_blk_cnt")          # [1,1,1,30]
    n("Greater", ["col_col_cnt", "thr05"], "col_has_col")
    n("Less", ["col_blk_cnt", "thr05"], "col_no_blk")
    n("And", ["col_has_col", "col_no_blk"], "V_col")     # [1,1,1,30] bool

    # ── Intersection mask ──────────────────────────────────────────────────────
    n("And", ["H_row", "V_col"], "inter_bool")            # [1,1,30,30] bool

    # ── hrmax: ReduceMax of input over H-rows ─────────────────────────────────
    n("ReduceMax", ["input"], "row_max", axes=[3], keepdims=1)     # [1,10,30,1]
    n("Cast", ["H_row"], "H_row_f", to=TensorProto.FLOAT)          # [1,1,30,1]
    n("Mul", ["row_max", "H_row_f"], "row_max_masked")              # [1,10,30,1]
    n("ReduceMax", ["row_max_masked"], "hrmax", axes=[2], keepdims=1)  # [1,10,1,1]

    # ── vcmax: ReduceMax of input over V-cols ─────────────────────────────────
    n("ReduceMax", ["input"], "col_max", axes=[2], keepdims=1)     # [1,10,1,30]
    n("Cast", ["V_col"], "V_col_f", to=TensorProto.FLOAT)          # [1,1,1,30]
    n("Mul", ["col_max", "V_col_f"], "col_max_masked")              # [1,10,1,30]
    n("ReduceMax", ["col_max_masked"], "vcmax", axes=[3], keepdims=1)  # [1,10,1,1]

    # ── Intersection colour and replacement ───────────────────────────────────
    n("Greater", ["hrmax", "thr05"], "hrmax_b")           # [1,10,1,1] bool
    n("Greater", ["vcmax", "thr05"], "vcmax_b")           # [1,10,1,1] bool

    # inter_color = hrmax AND vcmax  (the colour shown at intersection in input)
    n("And", ["hrmax_b", "vcmax_b"], "inter_color_b")     # [1,10,1,1] bool
    # all_colors  = hrmax OR vcmax   (H-color + V-color)
    n("Or", ["hrmax_b", "vcmax_b"], "all_col_b")          # [1,10,1,1] bool

    n("Cast", ["inter_color_b"], "inter_color_f", to=TensorProto.FLOAT)
    n("Cast", ["all_col_b"], "all_col_f", to=TensorProto.FLOAT)
    # replacement = all_colors - inter_color = the OTHER line's colour
    n("Sub", ["all_col_f", "inter_color_f"], "replacement")  # [1,10,1,1]

    # ── Final output ──────────────────────────────────────────────────────────
    n("Where", ["inter_bool", "replacement", "input"], "output")  # free

    return _model(nodes, inits)
