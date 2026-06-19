"""task192 (ARC-AGI 7e0986d6) — "remove the isolated noise pixels, keep the boxes,
recolour the boxes to the box colour".

Rule (from the generator):
  The grid holds 3..5 SOLID rectangular boxes of one colour `boxcolor` (each
  wides/talls in [3,10] so every box is >=3x3) plus a sprinkling of single isolated
  "noise" pixels of a DIFFERENT colour `color` (generator runs remove_neighbors so
  no two noise pixels are 4-adjacent and none bridges two shapes; a noise pixel MAY
  land ON a box cell, overwriting its colour in the INPUT, but the OUTPUT keeps the
  box colour there).  OUTPUT = the boxes only, every box cell painted `boxcolor`,
  every noise pixel deleted (-> background), off-grid -> all background.

  Closed-form local discriminator (verified exact 500/500 fresh, derived empirically):
    occ(r,c) = any colour (>0) present.  Let vsum = occ(up)+occ(down),
    hsum = occ(left)+occ(right).  A cell is a BOX cell IFF
        occ(r,c) AND vsum>=1 AND hsum>=1.
    (Box cells, being inside a >=3x3 solid, always have >=1 vertical AND >=1
     horizontal occupied neighbour; an isolated noise pixel — even one sitting on a
     box, even abutting a box on one side — never has BOTH.)
    This is captured by ONE 3x3 Conv with kernel  [[0,3,0],[1,6,1],[0,3,0]] applied
    to every colour channel 1..9 (ch0 weight 0):
        score = 6*occ + 3*vsum + 1*hsum
      box cells take score in {10,11,13,14}; noise cells in {6,7,8,9,12}.
      => box  <=>  score>9.5 AND score!=12 .  (the lone collision value is 12 =
         6+3+3 = a vertical sliver with no horizontal support = noise.)
    boxcolor = most-frequent non-background colour = ArgMax over channels 1..9 of
    the per-channel pixel COUNT (boxes >=9 cells each, noise sparse).
    in-grid rectangle = row/col SUM-profiles >0 (rowany (X) colany), so no 30x30
    in-grid plane is needed.

Encoding (route the 10-ch one-hot into the FREE bool output, keep <=4 full planes):
  score   [1,1,30,30] f32 (3600B) -- the single forced full f32 plane.
  kov_b = score>9.5         (keep-or-vgap, bool 900B)
  vgap_b = score==12        (the noise-sliver collision, bool 900B)
  bg_or_out_u8 = Where(rowany, Where(colany, BG, OUT), OUT)  -- in-grid bg vs outside,
                 broadcast from tiny [1,1,30,1]/[1,1,1,30] profiles (uint8 900B).
  target_u8 = Where(kov_b, Where(vgap_b, bg_or_out, boxcode), bg_or_out)  -- box cells
              (kov & !vgap) -> boxcode; vgap or !kov -> bg/outside (uint8 900B).
  output = Equal(channel_codes, target_u8) -> BOOL [1,10,30,30] (FREE).
  Dominant intermediate: the 3600B f32 score plane (Conv inherits the fp32 input).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I64 = TensorProto.INT64


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- score = single 3x3 Conv: 6*occ + 3*vsum + 1*hsum --------------------
    # kernel [[0,3,0],[1,6,1],[0,3,0]] on every colour channel 1..9 (ch0 = 0).
    k = np.array([[0, 3, 0], [1, 6, 1], [0, 3, 0]], np.float32)
    w = np.zeros((1, 10, 3, 3), np.float32)
    for ch in range(1, 10):
        w[0, ch] = k
    init("SCORE_W", w, np.float32)
    n("Conv", ["input", "SCORE_W"], "score",
      kernel_shape=[3, 3], pads=[1, 1, 1, 1])                     # [1,1,30,30] f32

    init("NINE5", np.array(9.5, np.float32).reshape(1), np.float32)
    init("TWELVE", np.array(12.0, np.float32).reshape(1), np.float32)
    n("Greater", ["score", "NINE5"], "kov_b")                     # [1,1,30,30] bool
    n("Equal", ["score", "TWELVE"], "vgap_b")                     # [1,1,30,30] bool

    # ---- in-grid rectangle from row/col sum-profiles (no 30x30 ingrid plane) --
    # row_f[1,1,30,1] = sum over channels 1..9 & cols of occupancy; >0.5 -> row has
    # an in-grid cell.  Same for cols.  (Background ch0 excluded by axis-1 sum incl
    # ch0? -> use ReduceSum over ALL channels: off-grid cells are all-zero so they
    # don't contribute; in-grid bg cells set ch0=1 so the row/col is marked in-grid.)
    n("ReduceSum", ["input"], "row_f", axes=[1, 3], keepdims=1)   # [1,1,30,1] f32
    n("ReduceSum", ["input"], "col_f", axes=[1, 2], keepdims=1)   # [1,1,1,30] f32
    init("HALF", np.array(0.5, np.float32).reshape(1), np.float32)
    n("Greater", ["row_f", "HALF"], "row_b")                      # [1,1,30,1] bool
    n("Greater", ["col_f", "HALF"], "col_b")                      # [1,1,1,30] bool

    # bg_or_out: in-grid bg cell -> BG code (9), off-grid -> OUT code (10).
    init("BG_U8", np.array(9, np.uint8).reshape(1), np.uint8)
    init("OUT_U8", np.array(10, np.uint8).reshape(1), np.uint8)
    n("Where", ["col_b", "BG_U8", "OUT_U8"], "col_bg_u8")         # [1,1,1,30] uint8
    n("Where", ["row_b", "col_bg_u8", "OUT_U8"], "bg_or_out_u8")  # [1,1,30,30] uint8

    # ---- boxcolor SCALAR = argmax over channels 1..9 of pixel counts ----------
    n("ReduceSum", ["input"], "cnt_all", axes=[2, 3], keepdims=1)  # [1,10,1,1] f32
    init("cc_s", np.array([1], np.int64), np.int64)
    init("cc_e", np.array([10], np.int64), np.int64)
    init("cc_ax", np.array([1], np.int64), np.int64)
    n("Slice", ["cnt_all", "cc_s", "cc_e", "cc_ax"], "cnt19")      # [1,9,1,1] f32
    n("ArgMax", ["cnt19"], "boxcode0", axis=1, keepdims=1)         # [1,1,1,1] i64 (0..8)
    n("Cast", ["boxcode0"], "boxcode", to=U8)                      # [1,1,1,1] uint8 (0..8)

    # ---- target colour-index plane (uint8) -----------------------------------
    #   box (kov & !vgap)        -> boxcode (0..8 -> colour channel 1..9)
    #   vgap OR !kov, in-grid    -> 9  (background channel 0)
    #   off-grid                 -> 10 (matches no channel -> all-zero)
    n("Where", ["vgap_b", "bg_or_out_u8", "boxcode"], "box_or_bg") # [1,1,30,30] uint8
    n("Where", ["kov_b", "box_or_bg", "bg_or_out_u8"], "target_u8")# [1,1,30,30] uint8

    # ---- expand to one-hot via Equal vs channel codes -> FREE bool output -----
    # channel c is on where target == channel_codes[c].
    # code 9 -> ch0 (bg); code j (0..8) -> ch (j+1); code 10 -> none.
    codes = np.empty((1, 10, 1, 1), np.uint8)
    codes[0, 0, 0, 0] = 9
    for j in range(9):
        codes[0, j + 1, 0, 0] = j
    init("CODES", codes, np.uint8)
    n("Equal", ["CODES", "target_u8"], "output")                  # [1,10,30,30] bool FREE

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task192", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
