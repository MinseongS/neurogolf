"""task148 (ARC-AGI 673ef223) — portal beam projection.

Rule (from the generator, numpy-oracle verified 500/500):
  Grid H in 16..24, W in 8..12.  TWO vertical red(2) "portals", each a run of
  length 4..6: one touching col 0 (left), the other col W-1 (right).  Equal run
  lengths; their tops are offset by delta=second-first (>=6, src run above dst).

  A few cyan(8) "markers" sit in the SOURCE portal's rows (the run whose cap is
  `srccol`).  Marker at row rs, col c.  Output:
    * Source row rs: cyan(8) beam from the cap inward up to the marker, yellow(4)
      at the marker col c (cap stays red; beyond the marker unchanged).
    * Dest row rd=rs+delta (aligned row in the OTHER portal): FULL cyan beam over
      all in-grid cols, dest cap (dstcol) stays red.
  Everything else == input.  flip mirrors columns -> source may be on left
  (srccol=0, fill rightward) or right (srccol=W-1, fill leftward); detected from
  which red run contains the marker rows.

Encoding (label-map on a small WORK=24x12 canvas):
  V = colour index (1x1 Conv sum_k k*input_k), Slice to 24x12, uint8.
  Per-row/col scalars: W-1, src_is_left, first, second, delta, marker col mcol[r],
  hasmark[r].  dstfill rows = hasmark shifted down by delta (Gather axis=2).
  L built by broadcasting Where overrides; Pad(sentinel 10) -> Equal -> BOOL out.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
I64 = TensorProto.INT64
B = TensorProto.BOOL

WR = 24  # active rows
WC = 12  # active cols
RED, CYAN, YEL = 2, 8, 4
BIG = 1000.0


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- constants ----
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    init("half", np.array(0.5, np.float32), np.float32)
    init("zero", np.array(0.0, np.float32), np.float32)
    init("rowramp", np.arange(WR, dtype=np.float32).reshape(1, 1, WR, 1), np.float32)
    init("colramp", np.arange(WC, dtype=np.float32).reshape(1, 1, 1, WC), np.float32)
    init("neg1", np.array(-1.0, np.float32), np.float32)
    init("bigf", np.array(BIG, np.float32), np.float32)
    init("clip_lo", np.array(0.0, np.float32), np.float32)
    init("clip_hi", np.array(float(WR - 1), np.float32), np.float32)
    init("u_cyan", np.array(CYAN, np.uint8), np.uint8)
    init("u_yel", np.array(YEL, np.uint8), np.uint8)

    # slices / pad
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - WR, 30 - WC], np.int64), np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)

    # ===== input has only colours {0,2,8}.  cyan(ch8): full 24x12 slice (markers
    # can be anywhere).  red(ch2): only at the two cap columns, so we never need a
    # full red plane — slice col0 (96B) and Gather the last col below. =====
    init("cy_st", np.array([0, CYAN, 0, 0], np.int64), np.int64)
    init("cy_en", np.array([1, CYAN + 1, WR, WC], np.int64), np.int64)
    init("redcol0_st", np.array([0, RED, 0, 0], np.int64), np.int64)
    init("redcol0_en", np.array([1, RED + 1, WR, 1], np.int64), np.int64)
    init("ax0123", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "cy_st", "cy_en", "ax0123"], "cyf24")      # [1,1,24,12] f32 1152B
    n("Greater", ["cyf24", "half"], "iscyan_b")                # bool 288B

    # ===== in-grid bounds from 1-D profiles of the FREE input (120B each) =====
    # Hm1 = max row with any channel ; Wm1 = max col with any channel.
    n("ReduceMax", ["input"], "rowhas", axes=[1, 3], keepdims=1)  # [1,1,30,1] 120B
    n("ReduceMax", ["input"], "colhas", axes=[1, 2], keepdims=1)  # [1,1,1,30] 120B
    init("rowramp30", np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1), np.float32)
    init("colramp30", np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30), np.float32)
    n("Greater", ["rowhas", "half"], "rowhas_b")
    n("Greater", ["colhas", "half"], "colhas_b")
    n("Where", ["rowhas_b", "rowramp30", "neg1"], "rowidx30")
    n("Where", ["colhas_b", "colramp30", "neg1"], "colidx30")
    n("ReduceMax", ["rowidx30"], "Hm1", keepdims=1)             # [1,1,1,1]
    n("ReduceMax", ["colidx30"], "Wm1", keepdims=1)             # [1,1,1,1] = W-1
    # row/col in-grid masks on the 24x12 ramps
    n("Greater", ["rowramp", "Hm1"], "row_gt_last")
    n("Not", ["row_gt_last"], "row_ingrid")                     # [1,1,24,1]
    n("Greater", ["colramp", "Wm1"], "col_gt_last")
    n("Not", ["col_gt_last"], "col_ingrid")                     # [1,1,1,12]
    n("And", ["row_ingrid", "col_ingrid"], "ingrid")           # [1,1,24,12] bool

    # ===== red-run rows (left cap col0 ; right cap col==W-1) — only tiny slices =====
    # leftred_row: red(ch2) at col0  -> [1,1,24,1]
    n("Slice", ["input", "redcol0_st", "redcol0_en", "ax0123"], "redcol0")  # [1,1,24,1] f32 96B
    n("Greater", ["redcol0", "half"], "leftred_b")            # bool
    n("Cast", ["leftred_b"], "leftred_row", to=F32)
    # rightred_row: red(ch2) at col W-1.  W in 8..12 -> W-1 in 7..11, so slice red
    # ch2 to cols 7..11 only ([1,1,24,5]=480B) and Gather col (Wm1-7).
    init("redwin_st", np.array([0, RED, 0, 7], np.int64), np.int64)
    init("redwin_en", np.array([1, RED + 1, WR, 12], np.int64), np.int64)
    n("Slice", ["input", "redwin_st", "redwin_en", "ax0123"], "redwin")  # [1,1,24,5] f32 480B
    init("seven", np.array(7.0, np.float32), np.float32)
    n("Sub", ["Wm1", "seven"], "lastoff_f")                   # Wm1-7 scalar
    n("Cast", ["lastoff_f"], "lastoff_i", to=I64)
    n("Reshape", ["lastoff_i", "scalar1"], "lastoff_v")       # [1]
    init("scalar1", np.array([1], np.int64), np.int64)
    n("Gather", ["redwin", "lastoff_v"], "redlast", axis=3)   # [1,1,24,1] f32
    n("Greater", ["redlast", "half"], "rightred_b")
    n("Cast", ["rightred_b"], "rightred_row", to=F32)

    # cyan rows (per-row any cyan) from the f32 cyan slice we already have
    n("ReduceMax", ["cyf24"], "cyanrow", axes=[3], keepdims=1)  # [1,1,24,1] f32

    # ===== src_is_left = any (cyanrow AND leftred_row) =====
    n("Mul", ["cyanrow", "leftred_row"], "cl")                  # [1,1,24,1]
    n("ReduceMax", ["cl"], "srcleft_f", keepdims=1)            # [1,1,1,1] {0,1}
    n("Greater", ["srcleft_f", "half"], "src_is_left")         # scalar bool

    # src red rows / dst red rows  (Where on scalar)
    n("Where", ["src_is_left", "leftred_row", "rightred_row"], "srcredrow")  # [1,1,24,1]
    n("Where", ["src_is_left", "rightred_row", "leftred_row"], "dstredrow")

    # first = min row of srcredrow ; second = min row of dstredrow
    n("Greater", ["srcredrow", "half"], "srcredrow_b")
    n("Where", ["srcredrow_b", "rowramp", "bigf"], "srcrow_idx")
    n("ReduceMin", ["srcrow_idx"], "first", keepdims=1)        # [1,1,1,1]
    n("Greater", ["dstredrow", "half"], "dstredrow_b")
    n("Where", ["dstredrow_b", "rowramp", "bigf"], "dstrow_idx")
    n("ReduceMin", ["dstrow_idx"], "second", keepdims=1)       # [1,1,1,1]
    n("Sub", ["second", "first"], "delta")                     # [1,1,1,1] f32

    # ===== per-row marker column mcol[r] (min cyan col, BIG if none), fp16 =====
    init("colramp16", np.arange(WC, dtype=np.float16).reshape(1, 1, 1, WC),
         np.float16)
    init("bigf16", np.array(BIG, np.float16), np.float16)
    n("Where", ["iscyan_b", "colramp16", "bigf16"], "cyancol_idx")  # fp16 [1,1,24,12] 576B
    n("ReduceMin", ["cyancol_idx"], "mcol16", axes=[3], keepdims=1)  # fp16 [1,1,24,1]
    n("Cast", ["mcol16"], "mcol", to=F32)                      # [1,1,24,1] f32
    n("Greater", ["cyanrow", "half"], "hasmark")               # [1,1,24,1] bool

    # ===== dstfill rows = hasmark shifted down by delta (Gather axis=2) =====
    n("Cast", ["hasmark"], "hasmark_f", to=F32)                 # [1,1,24,1]
    # idx[r] = clip(r - delta, 0, WR-1)
    n("Sub", ["rowramp", "delta"], "idx_f0")                   # [1,1,24,1] f32
    n("Max", ["idx_f0", "clip_lo"], "idx_f1")
    n("Min", ["idx_f1", "clip_hi"], "idx_f")
    n("Cast", ["idx_f"], "idx_i", to=I64)                      # [1,1,24,1]
    n("Reshape", ["idx_i", "reshape24"], "idx_v")              # [24]
    init("reshape24", np.array([WR], np.int64), np.int64)
    # gather along axis 2 of hasmark_f
    n("Gather", ["hasmark_f", "idx_v"], "dstfill_g", axis=2)   # [1,1,24,1,1]? -> need squeeze
    # Gather of [1,1,24,1] with indices shape [24] along axis2 -> [1,1,24,1] (replaces axis2)
    # Actually result shape: input shape with axis2 replaced by indices shape:
    # [1,1] + [24] + [1] = [1,1,24,1]. Good.
    n("Greater", ["dstfill_g", "half"], "dstfill")            # [1,1,24,1] bool

    # ===== SOURCE beam cyan cols: strictly between cap and marker =====
    #   left  src (cap col0):   lo=0,     hi=mcol[r]
    #   right src (cap col W-1): lo=mcol[r], hi=W-1
    # select lo,hi per row by the scalar src_is_left (f32 Where, tiny [1,1,24,1]).
    n("Not", ["src_is_left"], "src_is_right")
    n("Where", ["src_is_left", "zero", "mcol"], "lo")        # [1,1,24,1]
    n("Where", ["src_is_left", "mcol", "Wm1"], "hi")         # [1,1,24,1]
    n("Greater", ["colramp", "lo"], "col_gt_lo")             # [1,1,24,12]
    n("Less", ["colramp", "hi"], "col_lt_hi")
    n("And", ["col_gt_lo", "col_lt_hi"], "src_beam")
    n("And", ["src_beam", "hasmark"], "src_cyan_cond")       # in-grid at final Where
    # yellow cond: col==mcol AND hasmark
    n("Equal", ["colramp", "mcol"], "col_eq_mcol")           # [1,1,24,12]
    n("And", ["col_eq_mcol", "hasmark"], "yellow_cond")

    # ===== DEST beam cyan cols: all in-grid cols except the dest cap =====
    #   src left  -> dstcap=W-1 -> cyan for col < W-1
    #   src right -> dstcap=0   -> cyan for col > 0
    # per-orientation single col bound selected by scalar; build via col vs dbound.
    # src_is_left:  cyan = col < W-1   (dbound test: col != W-1 and col<W) handled by
    #               col<W-1; src right: col>0.  Use a col-vector chosen by scalar.
    n("Less", ["colramp", "Wm1"], "col_ltlast")              # [1,1,1,12]
    n("Greater", ["colramp", "zero"], "col_gt0")             # [1,1,1,12]
    n("And", ["src_is_left", "col_ltlast"], "dcl_a")
    n("And", ["src_is_right", "col_gt0"], "dcl_b")
    n("Or", ["dcl_a", "dcl_b"], "dst_cyan_cols")             # [1,1,1,12]
    n("And", ["dst_cyan_cols", "dstfill"], "dst_cyan_cond")  # [1,1,24,12]

    # ===== base label map: red(2) at the two cap columns (where the cap rows are),
    # else bg(0).  input has only {0,2,8}; input cyan markers become yellow via the
    # yellow override below, so they need no base. =====
    init("u_red2", np.array(RED, np.uint8), np.uint8)
    init("u_bg", np.array(0, np.uint8), np.uint8)
    # cap-column masks (col==0 and col==W-1)
    n("Equal", ["colramp", "zero"], "is_col0")               # [1,1,1,12]
    n("Equal", ["colramp", "Wm1"], "is_collast")             # [1,1,1,12]
    # leftred_row/rightred_row are bool-ish floats [1,1,24,1]; turn to bool
    n("Greater", ["leftred_row", "half"], "leftred_rb")
    n("Greater", ["rightred_row", "half"], "rightred_rb")
    n("And", ["is_col0", "leftred_rb"], "redbase_l")         # [1,1,24,12]
    n("And", ["is_collast", "rightred_rb"], "redbase_r")
    n("Or", ["redbase_l", "redbase_r"], "redbase")           # bool 288B
    n("Where", ["redbase", "u_red2", "u_bg"], "Vbase")       # uint8 [1,1,24,12]

    # ===== build L (start from Vbase) =====
    n("Where", ["dst_cyan_cond", "u_cyan", "Vbase"], "L1")   # dest beam
    n("Where", ["src_cyan_cond", "u_cyan", "L1"], "L2")      # source beam
    n("Where", ["yellow_cond", "u_yel", "L2"], "L3")         # yellow markers
    # off-grid (within WORK) -> sentinel 10 (all channels off)
    n("Where", ["ingrid", "L3", "padval"], "Lwork")

    # ===== pad + final one-hot =====
    n("Pad", ["Lwork", "padpads", "padval"], "L", mode="constant")
    n("Equal", ["L", "chan"], "output")

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task148", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
