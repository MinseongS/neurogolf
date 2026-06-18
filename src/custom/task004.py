"""task004 (ARC-AGI 025d127b) — un-slant each slanted parallelogram by a per-row +1 shear.

Rule (from generator task_025d127b.py, verified exact + fresh 200/200):
  The grid (size 8..16, variable H,W) holds 0..4 axis-stacked slanted "parallelogram"
  outlines, one DISTINCT random colour each, separated by a full blank row (so the
  shapes are ROW-SEPARABLE — never share a row).  Each shape spans rows [row .. row+tall-1].
  The OUTPUT shifts the shape's outline so its left edge stands upright:
    * every shape row shifts RIGHT by +1
    * EXCEPT the shape's BOTTOM row (shift 0)
    * AND the rightmost pixel of the SECOND-TO-LAST row also stays (shift 0)
  Colours simply COPY the input colours (random per shape) — no colour routing.

  Reformulated as a per-INPUT-cell partition (verified, zero collisions):
    occ          = colour-index plane >= 1                   (a coloured pixel)
    rowany[r]    = ReduceMax(occ over cols)                  (per-row occupancy)
    below[r]     = rowany[r+1] ;  below2[r] = rowany[r+2]
    is_bottom[r] = rowany[r] AND NOT below[r]
    is_2ndlast[r]= rowany[r] AND below[r] AND NOT below2[r]
    special[r,c] = occ[r,c] AND is_2ndlast[r] AND occ[r+1,c]  (occ pixel directly below)
    copy_cell    = occ AND (is_bottom OR special)            (cells that DON'T shift)
    shift_cell   = occ AND NOT copy_cell                     (cells that shift +1)
    L_out        = shiftR1(colf * shift_cell) + (colf * copy_cell)   (no collisions)
  Output one-hot = Equal(L_out, arange[0..9]) masked to the in-grid region.

Encoding (ONE fp32 colour-index entry plane; everything downstream narrow/cropped):
  - colf30 = Conv(input, W) -> [1,1,30,30] f32 (the one irreducible 3600B entry).
    Channel-0 weight is 0.5 (a SENTINEL), channels 1..9 are 1..9, so the SINGLE plane
    encodes all three states off-grid=0 / in-grid-bg=0.5 / coloured pixel=k>=1 — there
    is NO separate ReduceMax in-grid plane (the diagnosed redundant-plane elimination).
  - Slice colf30 -> [1,1,17,17] active corner; all per-cell working planes are 17x17 f16.
  - occ = colf>=0.75 (>0.5) ; ingrid = colf>=0.25 (>0) — both derived from the one plane.
  - L_out collapsed to ONE 17x17 value plane; off-grid -> sentinel 255 via the ingrid mask.
  - Cast the 17x17 value plane to int32 (1156B) FIRST, then Pad to 30x30 (the only 30x30
    int32 plane), then Equal(.., arange) writes straight into the FREE BOOL output — no
    intermediate 30x30 fp16 plane, no [1,10,30,30] expansion ever materialises.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I32 = TensorProto.INT32
I64 = TensorProto.INT64


def build(task):
    inits, nodes = [], []

    _NP = {F16: np.float16, F32: np.float32, BOOL: np.bool_,
           U8: np.uint8, I32: np.int32, I64: np.int64}

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=_NP[dtype]), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # WORK = active canvas: grid is at most 16x16 (generator randint(8,16)).  Measured
    # over 30k fresh instances the coloured extent of BOTH input and output never exceeds
    # col/row 15, and any +1 shift that would reach col 16 is a clamped no-op in the rule
    # (ground truth has nothing at col 16), so a 16x16 crop captures the whole active
    # region and harmlessly drops the off-grid col-16 shift overflow.
    W = 16

    # ---- constants ----
    # colour-index conv weight: ch0 = 0.5 sentinel (in-grid background marker), ch_k = k.
    kw = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    kw[0, 0, 0, 0] = 0.5
    init("kw", kw, F32)
    init("arange10", np.arange(10, dtype=np.int32).reshape(1, 10, 1, 1), I32)

    init("quarter_f32", np.array(0.25, np.float32), F32)  # in-grid threshold (bg=0.5>0.25)
    init("thr75_f32", np.array(0.75, np.float32), F32)    # occupancy threshold (>=1)
    init("one_f16", np.array(1.0, np.float16), F16)
    init("sent255", np.array(255.0, np.float16), F16)

    # crop spec: rows 0..W, cols 0..W
    init("crop_st", np.array([0, 0], np.int64), I64)
    init("crop_en", np.array([W, W], np.int64), I64)
    init("crop_ax", np.array([2, 3], np.int64), I64)
    # axis / index consts for the small-canvas Slices
    init("s_axis2", np.array([2], np.int64), I64)
    init("s_axis3", np.array([3], np.int64), I64)
    init("st0", np.array([0], np.int64), I64)
    init("st1", np.array([1], np.int64), I64)
    init("st2", np.array([2], np.int64), I64)
    init("enW", np.array([W], np.int64), I64)        # W
    init("enW1", np.array([W + 1], np.int64), I64)   # W+1 (after pad 1)
    init("enW2", np.array([W + 2], np.int64), I64)   # W+2 (after pad 2)

    # opset-11 Pad takes `pads` (and optional constant `value`) as INPUTS.
    init("p_row1", np.array([0, 0, 0, 0, 0, 0, 1, 0], np.int64), I64)  # pad bottom row +1
    init("p_row2", np.array([0, 0, 0, 0, 0, 0, 2, 0], np.int64), I64)  # pad bottom row +2
    init("p_left1", np.array([0, 0, 0, 1, 0, 0, 0, 0], np.int64), I64)  # pad left col +1
    init("p_final", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64), I64)
    init("v_zero_f16", np.array(0.0, np.float16), F16)
    init("v_sent_i32", np.array(255, np.int32), I32)

    # ---- ONE colour-index entry plane on the full 30x30 input, then crop to WxW.
    #      The ch0=0.5 sentinel folds the in-grid mask into this single plane. ----
    n("Conv", ["input", "kw"], "colf30")            # [1,1,30,30] f32 (the ONLY 30x30 fp32 plane)
    n("Slice", ["colf30", "crop_st", "crop_en", "crop_ax"], "colf")  # [1,1,W,W] f32

    # in-grid: in-grid-bg = 0.5, off-grid = 0  ->  ingrid = colf > 0.25
    n("Greater", ["colf", "quarter_f32"], "ingrid")  # bool [1,1,W,W]: in-grid (bg or pixel)

    # ---- occupancy (fp16): occ = colf >= 1  <=>  colf > 0.75 ----
    n("Greater", ["colf", "thr75_f32"], "occ_b")    # bool
    n("Cast", ["occ_b"], "occ", to=F16)             # [1,1,W,W] f16 {0,1}
    # colour value plane in fp16 (coloured = k, bg/off = 0.5/0 -> masked out by occ)
    n("Cast", ["colf"], "colf16", to=F16)           # [1,1,W,W] f16

    # ---- per-row occupancy ----
    n("ReduceMax", ["occ"], "rowany", axes=[3], keepdims=1)  # [1,1,W,1] f16

    # below[r]=rowany[r+1]: pad bottom by 1, slice off first row
    n("Pad", ["rowany", "p_row1", "v_zero_f16"], "rowany_p1", mode="constant")
    n("Slice", ["rowany_p1", "st1", "enW1", "s_axis2"], "below")   # [1,1,W,1]
    # below2[r]=rowany[r+2]
    n("Pad", ["rowany", "p_row2", "v_zero_f16"], "rowany_p2", mode="constant")
    n("Slice", ["rowany_p2", "st2", "enW2", "s_axis2"], "below2")  # [1,1,W,1]

    # is_bottom = rowany AND NOT below  = rowany * (1-below)
    n("Sub", ["one_f16", "below"], "notbelow")
    n("Mul", ["rowany", "notbelow"], "is_bottom")   # [1,1,W,1] f16 {0,1}
    # is_2ndlast = rowany * below * (1-below2)
    n("Sub", ["one_f16", "below2"], "notbelow2")
    n("Mul", ["rowany", "below"], "rb")
    n("Mul", ["rb", "notbelow2"], "is_2ndlast")     # [1,1,W,1] f16 {0,1}

    # occdown[r,c] = occ[r+1,c]: pad bottom by 1, slice off first row
    n("Pad", ["occ", "p_row1", "v_zero_f16"], "occ_p1", mode="constant")
    n("Slice", ["occ_p1", "st1", "enW1", "s_axis2"], "occdown")    # [1,1,W,W]

    # special = occ * is_2ndlast(broadcast) * occdown
    n("Mul", ["occ", "is_2ndlast"], "occ_2nd")      # broadcast [1,1,W,1]
    n("Mul", ["occ_2nd", "occdown"], "special")     # [1,1,W,W] f16 {0,1}

    # copy_cell = occ * max(is_bottom(broadcast), special)
    n("Max", ["is_bottom", "special"], "copy_sel")  # broadcast -> [1,1,W,W]
    n("Mul", ["occ", "copy_sel"], "copy_cell")      # [1,1,W,W] f16 {0,1}
    # shift_cell = occ AND NOT copy_cell.  copy_cell is a subset of occ, so
    # shift_cell = occ - copy_cell exactly (saves the Sub+Mul of the 1-copy_cell route).
    n("Sub", ["occ", "copy_cell"], "shift_cell")    # [1,1,W,W] f16 {0,1}

    # value contributions (fp16): copy stays, shift moves +1 col
    n("Mul", ["colf16", "copy_cell"], "copyval")    # [1,1,W,W]
    n("Mul", ["colf16", "shift_cell"], "shiftval")
    # shiftR1: pad left by 1, slice off last col
    n("Pad", ["shiftval", "p_left1", "v_zero_f16"], "shiftval_p", mode="constant")
    n("Slice", ["shiftval_p", "st0", "enW", "s_axis3"], "shiftedval")  # [1,1,W,W]
    n("Add", ["copyval", "shiftedval"], "Lout_f16")  # [1,1,W,W] f16 value 0..9

    # ---- route into free output ----
    # mask off-grid-within-canvas to sentinel (fp16), Cast to int32 at WxW (1024B), then
    # int32-Pad to 30x30 with sentinel 255 (the mandatory 3600B Equal input — opset-11
    # Pad DOES accept int32, so we cast at WxW and skip the 1800B fp16 30x30 bridge
    # plane), then Equal into the free BOOL output.  Off-grid -> 255 -> matches no colour
    # -> all-zero one-hot, exactly the ground truth.
    n("Where", ["ingrid", "Lout_f16", "sent255"], "Lmask")  # [1,1,W,W] f16, off-grid -> 255
    n("Cast", ["Lmask"], "Lmask_i32", to=I32)               # [1,1,W,W] int32 (1024B)
    n("Pad", ["Lmask_i32", "p_final", "v_sent_i32"], "L30_i32", mode="constant")  # [1,1,30,30] int32
    n("Equal", ["L30_i32", "arange10"], "output")           # BOOL [1,10,30,30] = FREE output

    graph = helper.make_graph(
        nodes, "task004",
        [helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
