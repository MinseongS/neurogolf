"""task004 (ARC-AGI 025d127b) — un-slant each slanted parallelogram by a per-row +1 shear.

Rule (from generator task_025d127b.py, verified exact 3000/3000 + fresh 200/200):
  The grid (size 8..16, variable H,W) holds 0..4 axis-stacked slanted "parallelogram"
  outlines, one DISTINCT random colour each, separated by a full blank row (so the
  shapes are ROW-SEPARABLE — never share a row).  Each shape spans rows [row .. row+tall-1].
  The OUTPUT shifts the shape's outline so its left edge stands upright:
    * every shape row shifts RIGHT by +1
    * EXCEPT the shape's BOTTOM row (shift 0)
    * AND the rightmost pixel of the SECOND-TO-LAST row also stays (shift 0)
  Colours simply COPY the input colours (random per shape) — no colour routing.

  Reformulated as a per-INPUT-cell partition (verified 3000/3000, zero collisions):
    occ          = colour-index plane > 0
    rowany[r]    = ReduceMax(occ over cols)                  (per-row occupancy)
    below[r]     = rowany[r+1] ;  below2[r] = rowany[r+2]
    is_bottom[r] = rowany[r] AND NOT below[r]
    is_2ndlast[r]= rowany[r] AND below[r] AND NOT below2[r]
    special[r,c] = occ[r,c] AND is_2ndlast[r] AND occ[r+1,c]  (occ pixel directly below)
    copy_cell    = occ AND (is_bottom OR special)            (cells that DON'T shift)
    shift_cell   = occ AND NOT copy_cell                     (cells that shift +1)
    L_out        = shiftR1(colf * shift_cell) + (colf * copy_cell)   (no collisions)
  Output one-hot = Equal(L_out, arange[0..9]) masked to the in-grid region.

Encoding (single fp32 colour-index entry plane, everything else fp16/bool):
  - colf = Conv(input, [0..9])  -> [1,1,30,30] f32 (the one irreducible 3600B entry).
  - ingrid = ReduceMax(input over all 10 channels) -> in-grid (incl. bg) vs all-zero off-grid;
    off-grid stays all-zero in the output by writing sentinel 255 there.
  - all masks built on the fp16 occupancy plane (1800B planes); vertical neighbour shifts
    via Slice+Pad on the row axis; horizontal +1 shift via Slice+Pad on the col axis.
  - L_out collapsed to ONE [1,1,30,30] value plane; final Where(ingrid, L_out, 255)->uint8,
    then Equal(L_u8, arange[1,10,1,1]) writes straight into the FREE BOOL output —
    no [1,10,30,30] intermediate is ever materialized.
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

    # ---- constants ----
    init("kw", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), F32)  # colour-index conv
    init("arange10", np.arange(10, dtype=np.int32).reshape(1, 10, 1, 1), I32)
    init("zero_f16", np.array(0.0, np.float16), F16)
    init("zero_f32", np.array(0.0, np.float32), F32)
    init("sent255", np.array(255.0, np.float16), F16)

    # WORK = active canvas: grid is at most 16x16 (generator randint(8,16)); +1 col of
    # shift headroom -> 17 wide.  Crop colf/occ to WORK x WORK so every working fp16 plane
    # is 17x17x2 = 578B instead of the full 30x30 (1800B).
    W = 17
    init("one_f16", np.array(1.0, np.float16), F16)
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

    # ---- colour-index plane + in-grid mask on the full 30x30 input (these two
    #      30x30 fp32 planes are the only full-size intermediates), then crop both
    #      to the WxW active corner so all later working planes are WxW ----
    n("Conv", ["input", "kw"], "colf30")            # [1,1,30,30] f32, value 0..9
    n("Slice", ["colf30", "crop_st", "crop_en", "crop_ax"], "colf")  # [1,1,W,W] f32

    n("ReduceMax", ["input"], "ingrid30", axes=[1], keepdims=1)  # [1,1,30,30] f32 {0,1}
    n("Slice", ["ingrid30", "crop_st", "crop_en", "crop_ax"], "ingrid_f")  # [1,1,W,W]
    n("Greater", ["ingrid_f", "zero_f32"], "ingrid")             # bool [1,1,W,W]

    # ---- occupancy (fp16) ----
    n("Cast", ["colf"], "colf16", to=F16)           # [1,1,W,W] f16 value plane
    n("Greater", ["colf16", "zero_f16"], "occ_b")   # bool
    n("Cast", ["occ_b"], "occ", to=F16)             # [1,1,W,W] f16 {0,1}

    # ---- per-row occupancy ----
    n("ReduceMax", ["occ"], "rowany", axes=[3], keepdims=1)  # [1,1,W,1] f16

    # below[r]=rowany[r+1]: pad bottom by 1, slice off first row
    n("Pad", ["rowany"], "rowany_p1", mode="constant", value=0.0,
      pads=[0, 0, 0, 0, 0, 0, 1, 0])
    n("Slice", ["rowany_p1", "st1", "enW1", "s_axis2"], "below")   # [1,1,W,1]
    # below2[r]=rowany[r+2]
    n("Pad", ["rowany"], "rowany_p2", mode="constant", value=0.0,
      pads=[0, 0, 0, 0, 0, 0, 2, 0])
    n("Slice", ["rowany_p2", "st2", "enW2", "s_axis2"], "below2")  # [1,1,W,1]

    # is_bottom = rowany AND NOT below  = rowany * (1-below)
    n("Sub", ["one_f16", "below"], "notbelow")
    n("Mul", ["rowany", "notbelow"], "is_bottom")   # [1,1,W,1] f16 {0,1}
    # is_2ndlast = rowany * below * (1-below2)
    n("Sub", ["one_f16", "below2"], "notbelow2")
    n("Mul", ["rowany", "below"], "rb")
    n("Mul", ["rb", "notbelow2"], "is_2ndlast")     # [1,1,W,1] f16 {0,1}

    # occdown[r,c] = occ[r+1,c]: pad bottom by 1, slice off first row
    n("Pad", ["occ"], "occ_p1", mode="constant", value=0.0,
      pads=[0, 0, 0, 0, 0, 0, 1, 0])
    n("Slice", ["occ_p1", "st1", "enW1", "s_axis2"], "occdown")    # [1,1,W,W]

    # special = occ * is_2ndlast(broadcast) * occdown
    n("Mul", ["occ", "is_2ndlast"], "occ_2nd")      # broadcast [1,1,W,1]
    n("Mul", ["occ_2nd", "occdown"], "special")     # [1,1,W,W] f16 {0,1}

    # copy_cell = occ * max(is_bottom(broadcast), special)
    n("Max", ["is_bottom", "special"], "copy_sel")  # broadcast -> [1,1,W,W]
    n("Mul", ["occ", "copy_sel"], "copy_cell")      # [1,1,W,W] f16 {0,1}
    # shift_cell = occ * (1-copy_cell)
    n("Sub", ["one_f16", "copy_cell"], "notcopy")
    n("Mul", ["occ", "notcopy"], "shift_cell")      # [1,1,W,W] f16 {0,1}

    # value contributions (fp16): copy stays, shift moves +1 col
    n("Mul", ["colf16", "copy_cell"], "copyval")    # [1,1,W,W]
    n("Mul", ["colf16", "shift_cell"], "shiftval")
    # shiftR1: pad left by 1, slice off last col
    n("Pad", ["shiftval"], "shiftval_p", mode="constant", value=0.0,
      pads=[0, 0, 0, 1, 0, 0, 0, 0])
    n("Slice", ["shiftval_p", "st0", "enW", "s_axis3"], "shiftedval")  # [1,1,W,W]
    n("Add", ["copyval", "shiftedval"], "Lout_f16")  # [1,1,W,W] f16 value 0..9

    # ---- route into free output ----
    # mask off-grid-within-canvas to sentinel (fp16), pad to 30x30 with sentinel,
    # then cast->int32 and Equal into the free output.
    n("Where", ["ingrid", "Lout_f16", "sent255"], "Lmask")  # [1,1,W,W] f16, off-grid -> 255
    n("Pad", ["Lmask"], "L30", mode="constant", value=255.0,
      pads=[0, 0, 0, 0, 0, 0, 30 - W, 30 - W])          # [1,1,30,30] f16
    n("Cast", ["L30"], "L30_i32", to=I32)               # int32 (Equal needs int/bool)
    n("Equal", ["L30_i32", "arange10"], "output")       # BOOL [1,10,30,30] = FREE output

    graph = helper.make_graph(
        nodes, "task004",
        [helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
    model.ir_version = IR_VERSION
    return model
