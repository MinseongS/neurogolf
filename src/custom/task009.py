"""task009 (ARC-AGI 06df4c85) — "connect same-colored cells sharing a row/column".

Rule (generator task_06df4c85.py + common.create_linegrid):
  The pixel grid is `create_linegrid(bitmap, spacing=2, linecolor)` of an underlying
  size-n bitmap (n in [6,10]): each bitmap cell (r,c) -> a 2x2 block at pixel
  rows/cols {3r,3r+1}x{3c,3c+1}; pixel rows/cols == 2 (mod 3) are gridlines (linecolor).
  Transform = connect_bitmap: for every pair of SAME-colored bitmap cells sharing a row,
  fill the cells between them (inclusive) with that colour; likewise columns.

Encoding — the public kojimar net pays a 3600B f32 [1,9,10,10] `colors_sample` plane
(an Einsum span-fill); this re-golf REMOVES it:
  * STRIDED 1x1 Conv  `Lbm = Conv(input, W[1,10,1,1]=[0..9], strides=3)` collapses the 10
    colour channels AND sub-samples to bitmap scale in ONE op -> [1,1,10,10] f32 (400B),
    a single colour-INDEX bitmap (0 on background) — no 9-channel plane ever materialises.
  * Span-fill on that 1-channel plane via DIRECTIONAL nearest-marker carries (the spans
    provably never interleave across colours — the generator only fills FREE spans, max
    coverage 1/cell — so the nearest dot left and nearest dot right of a span cell are the
    SAME colour).  pack = pos*16 + colour at dots; prefix-MaxPool = nearest-left pack,
    suffix-MaxPool of the reversed ramp = nearest-right pack; fill iff decoded l==r>0.
  * Re-render to the 30x30 linegrid with DepthToSpace(blocksize=3) of nine 1x10x10 u8
    sub-blocks (4 content + 5 gridline separators), then Equal vs colour ids -> FREE output.
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

N = 10  # bitmap canvas side (n in [6,10]; off-grid cells are all-zero)


def build(task):
    inits, nodes = [], []
    _np = {F32: np.float32, F16: np.float16, BOOL: np.bool_,
           U8: np.uint8, I64: np.int64}

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=_np[dtype]), name))
        return name

    def nd(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- 1) colour-index bitmap via strided 1x1 conv (NO 9-channel plane) ---
    # ch0 (background) weight = 0.5 so ONE conv encodes: off-grid=0 / in-grid-bg=0.5 /
    # coloured-dot=k>=1.  valid_cell = Lbm>0.25 ; dot = Lbm>0.75 (task004 fractional lever)
    Widx = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    Widx[0, 0] = 0.5
    init("Widx", Widx, F32)
    nd("Conv", ["input", "Widx"], "Lbm_f32", strides=[3, 3])      # [1,1,10,10] f32 colour idx
    nd("Cast", ["Lbm_f32"], "Lbm", to=F16)                        # [1,1,10,10] f16

    init("T25", np.array(0.25, np.float16), F16)
    init("T75", np.array(0.75, np.float16), F16)
    nd("Greater", ["Lbm", "T25"], "valid_cell_b")                # [1,1,10,10] bool in-grid
    nd("Greater", ["Lbm", "T75"], "dot")                          # [1,1,10,10] bool: a dot

    # ---- 2) span-fill via directional nearest-marker carries ---------------
    # pack pos*16 + colour at dot cells (large-neg elsewhere); pos = index along axis.
    colramp = (np.arange(N) * 16).astype(np.float16).reshape(1, 1, 1, N)
    colrrev = ((N - 1 - np.arange(N)) * 16).astype(np.float16).reshape(1, 1, 1, N)
    rowramp = (np.arange(N) * 16).astype(np.float16).reshape(1, 1, N, 1)
    rowrrev = ((N - 1 - np.arange(N)) * 16).astype(np.float16).reshape(1, 1, N, 1)
    init("colramp", colramp, F16); init("colrrev", colrrev, F16)
    init("rowramp", rowramp, F16); init("rowrrev", rowrrev, F16)
    init("S16", np.array(16.0, np.float16), F16)
    init("NEGBIG", np.array(-1000.0, np.float16), F16)

    # mask bg cells to a large-negative value so they always LOSE the prefix/suffix
    # max (then a simple Add of the position ramp is enough — no per-pack Where).
    nd("Where", ["dot", "Lbm", "NEGBIG"], "Lbm_dot")              # [1,1,10,10] f16

    def carries(axis, fwd_ramp, rev_ramp, tag):
        # packed = pos*16 + colour at dots, large-negative at bg cells
        nd("Add", ["Lbm_dot", fwd_ramp], "pkf_" + tag)
        nd("Add", ["Lbm_dot", rev_ramp], "pkr_" + tag)
        if axis == 3:
            pads_pre = [0, N - 1, 0, 0]; pads_suf = [0, 0, 0, N - 1]; ks = [1, N]
        else:
            pads_pre = [N - 1, 0, 0, 0]; pads_suf = [0, 0, N - 1, 0]; ks = [N, 1]
        nd("MaxPool", ["pkf_" + tag], "lpk_" + tag, kernel_shape=ks, pads=pads_pre)
        nd("MaxPool", ["pkr_" + tag], "rpk_" + tag, kernel_shape=ks, pads=pads_suf)
        # decode colour = pack mod 16 (for bg-only sides this stays negative -> loses below)
        nd("Mod", ["lpk_" + tag, "S16"], "lval_" + tag, fmod=1)
        nd("Mod", ["rpk_" + tag, "S16"], "rval_" + tag, fmod=1)
        # bracketed by SAME colour on both sides => fill with that colour.  A no-dot side
        # decodes negative, never equals a valid colour, and even a spurious equal fills a
        # NEGATIVE value which loses the final Max against Lbm>=0.
        nd("Equal", ["lval_" + tag, "rval_" + tag], "eq_" + tag)
        nd("Where", ["eq_" + tag, "lval_" + tag, "NEGBIG"], "fv_" + tag)  # fill colour
        return "fv_" + tag

    rowfill = carries(3, "colramp", "colrrev", "row")   # fill along columns (row spans)
    colfill = carries(2, "rowramp", "rowrrev", "col")   # fill along rows  (col spans)

    nd("Max", [rowfill, colfill, "Lbm"], "conn_f16")    # connected colour idx (incl dots)
    nd("Cast", ["conn_f16"], "content_u8", to=U8)       # [1,1,10,10] u8

    # ---- 3) content_grid: content where in-grid, sentinel 255 off-grid -----
    init("invalid_u8", np.array(255, np.uint8), U8)
    nd("Where", ["valid_cell_b", "content_u8", "invalid_u8"], "content_grid_u8")

    # ---- 4) gridline separator sub-blocks ----------------------------------
    init("mc_s", [1], I64); init("mc_e", [10], I64); init("mc_ax", [3], I64)
    init("mr_s", [1], I64); init("mr_e", [10], I64); init("mr_ax", [2], I64)
    init("m_st", [1], I64)
    nd("Slice", ["valid_cell_b", "mc_s", "mc_e", "mc_ax", "m_st"], "v_mask_tail")  # [1,1,10,9]
    nd("Slice", ["valid_cell_b", "mr_s", "mr_e", "mr_ax", "m_st"], "h_mask_tail")  # [1,1,9,10]
    init("false_col_shape", [1, 1, 10, 1], I64)
    init("false_row_shape", [1, 1, 1, 10], I64)
    nd("ConstantOfShape", ["false_col_shape"], "false_col",
       value=numpy_helper.from_array(np.array([0], np.bool_), "v"))
    nd("ConstantOfShape", ["false_row_shape"], "false_row",
       value=numpy_helper.from_array(np.array([0], np.bool_), "v"))
    nd("Concat", ["v_mask_tail", "false_col"], "v_sep_mask", axis=3)   # [1,1,10,10] bool
    nd("Concat", ["h_mask_tail", "false_row"], "h_sep_mask", axis=2)   # [1,1,10,10] bool

    # linecolour scalar from a guaranteed gridline pixel (2,0) (n>=6 so it exists)
    init("line_s", [2, 0], I64); init("line_e", [3, 1], I64); init("line_ax", [2, 3], I64)
    nd("Slice", ["input", "line_s", "line_e", "line_ax"], "line_color_onehot_f")  # [1,10,1,1]
    nd("ArgMax", ["line_color_onehot_f"], "line_color_i64", axis=1, keepdims=1)
    nd("Cast", ["line_color_i64"], "line_color_u8", to=U8)            # [1,1,1,1] u8

    nd("Where", ["v_sep_mask", "line_color_u8", "invalid_u8"], "v_sep_u8")
    nd("Where", ["h_sep_mask", "line_color_u8", "invalid_u8"], "h_sep_u8")
    nd("And", ["v_sep_mask", "h_sep_mask"], "x_sep_mask")
    nd("Where", ["x_sep_mask", "line_color_u8", "invalid_u8"], "x_sep_u8")

    # ---- 5) assemble 3x3 super-cell via DepthToSpace -----------------------
    nd("Concat", ["content_grid_u8", "content_grid_u8", "v_sep_u8",
                  "content_grid_u8", "content_grid_u8", "v_sep_u8",
                  "h_sep_u8", "h_sep_u8", "x_sep_u8"], "scalar_blocks_u8", axis=1)
    nd("DepthToSpace", ["scalar_blocks_u8"], "color_grid_u8", blocksize=3, mode="DCR")  # [1,1,30,30]

    # ---- 6) colour-index plane -> one-hot bool output (FREE) ---------------
    init("all_color_ids", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), U8)
    nd("Equal", ["color_grid_u8", "all_color_ids"], "output")    # FREE [1,10,30,30] bool

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task009", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
