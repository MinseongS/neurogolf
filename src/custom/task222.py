"""task222 (ARC-AGI 91714a58) -- "keep the one solid box, drop the noise".

Rule (from the generator):
  A `size`x`size` (size=16) grid holds one SOLID axis-aligned rectangle of a single
  `color` (width,height in [2,8], 9 <= area <= 16, placed strictly interior) on top of
  a field of random single-pixel noise of arbitrary colours.  The generator guarantees
  the box `color` has NO same-colour 4-neighbour anywhere outside the box (so noise of
  the box colour is always isolated).  OUTPUT = INPUT with everything but the box zeroed
  (keep box cells at their colour, blank everything else to background).

Exact local rule (brute-verified 0 fails / 50000 fresh instances):
  keep(r,c) iff cell (r,c) belongs to a fully-filled, single-non-zero-colour block of
  shape 3x3 OR 2x4 OR 4x2.
  Why this is exact AND robust:
    * Every valid box (area>=9, both dims>=2) CONTAINS such a sub-block: a box with a
      dimension==2 must be >=2x5 (area>=9) so it contains a 2x4 (or 4x2); otherwise both
      dims>=3 so it contains a 3x3.  Every box cell lies inside one of these windows.
    * Noise occasionally forms a spurious solid 2x2 (~1/60) and very rarely a 2x3; a
      3x3 / 2x4 / 4x2 (>=8 same-colour pixels in that exact shape among ~50%-density
      random noise) drives false positives to ~0.

Encoding (floor-break: route the 10-ch expansion into the FREE output, task193 idiom):
  colf30[1,1,30,30] f32 = Conv1x1(input,[0,1,..,9])  -> per-cell colour index (bg=0).
  Crop to the WxW active grid (the box lives in rows/cols 1..14), cast f16.
  For each block shape (h,w):
    mx = MaxPool(v,(h,w))      ; nmn = MaxPool(-v,(h,w))    (TL-anchored, no pad)
    seed = (mx + nmn == 0) AND (nmn < 0)   -> window uniform & non-background
    dil  = MaxPool(seed_f16,(h,w), pads=[h-1,w-1,0,0])      -> BR dilation back to WxW
           (MaxPool zero-pads, so this one op fuses the pad-restore + footprint dilation)
  keep = OR of the three dilations; Pad to 30x30 (True off-grid since input is all-zero
         there -> Where selects the all-zero input, not the bg one-hot).
  output = Where(keep, input, bg_onehot).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8

# Detection window.  The 16x16 grid's solid box is placed strictly interior
# (row,col >= 1 ; row+height,col+width <= 15) so every box cell AND every covering
# 3x3 / 2x4 / 4x2 window lies in rows/cols 1..14 -> a 14x14 crop at OFF=1 contains all
# detectable structure.  Noise outside that crop is never part of a valid block, so the
# keep mask is simply False there (in-grid border rows 0,15 blanked; off-grid >=16
# padded True so Where picks the all-zero input rather than the bg one-hot).
OFF = 1
W = 14


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- colour-index plane (the one forced f32 entry read of the input) ---
    wpack = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)  # [0,1,..,9]
    init("WPACK", wpack, np.float32)
    n("Conv", ["input", "WPACK"], "colf30", kernel_shape=[1, 1])  # [1,1,30,30] f32

    # crop to the WxW detection window at offset OFF.
    init("c_s", np.array([OFF, OFF], np.int64), np.int64)
    init("c_e", np.array([OFF + W, OFF + W], np.int64), np.int64)
    init("c_ax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["colf30", "c_s", "c_e", "c_ax"], "colf", )         # [1,1,W,W] f32
    n("Cast", ["colf"], "v", to=F16)                               # [1,1,W,W] f16

    init("ZERO0", np.array(0.0, np.float16), np.float16)

    keep_terms = []
    for h, w in [(3, 3), (2, 4), (4, 2)]:
        tag = f"{h}x{w}"
        k = float(h * w)
        # A TL-anchored h x w window is a fully-filled single-colour block iff every
        # cell equals the window max, i.e. sum == k*max, AND the colour is non-bg
        # (max>0).  Window sum via a no-pad ones-Conv (values <=81 -> fp16 exact); this
        # drops the shared Neg plane and the per-shape min-via-max MaxPool.
        n("MaxPool", ["v"], f"mx_{tag}", kernel_shape=[h, w], strides=[1, 1])
        init(f"ONES_{tag}", np.ones((1, 1, h, w), np.float16), np.float16)
        n("Conv", ["v", f"ONES_{tag}"], f"S_{tag}", kernel_shape=[h, w])  # window sum
        init(f"K_{tag}", np.array(k, np.float16), np.float16)
        n("Mul", [f"mx_{tag}", f"K_{tag}"], f"kmx_{tag}")          # k*max
        n("Equal", [f"S_{tag}", f"kmx_{tag}"], f"uni_{tag}")      # sum == k*max
        n("Greater", [f"mx_{tag}", "ZERO0"], f"nz_{tag}")         # colour non-bg
        n("And", [f"uni_{tag}", f"nz_{tag}"], f"seedb_{tag}")     # [1,1,oh,ow] bool
        n("Cast", [f"seedb_{tag}"], f"seed_{tag}", to=F16)        # f16 {0,1}
        # bottom-right-anchored MaxPool spreads each window-TL seed over the h x w cells
        # it certifies AND restores SZxSZ in ONE op: with symmetric pads=[h-1,..] the
        # output index o reads seed[o-h+1 .. o] (MaxPool zero-pads), so out[r,c] fires
        # iff some covering window (TL in r-h+1..r) was a valid seed.  No separate Pad
        # plane (saves a full SZxSZ f16 carrier per shape).
        n("MaxPool", [f"seed_{tag}"], f"dil_{tag}",
          kernel_shape=[h, w], strides=[1, 1],
          pads=[h - 1, w - 1, h - 1, w - 1])                      # [1,1,W,W] f16
        keep_terms.append(f"dil_{tag}")

    # keep = ANY block-shape dilation fired: ONE variadic Max over the three f16 maps.
    # The maps are exactly {0,1} (seeds cast 0/1 then MaxPool'd), so Cast f16->uint8 is
    # the keep mask directly -- no Greater/threshold plane needed.
    n("Max", keep_terms, "dilmax")                                # [1,1,W,W] f16 {0,1}
    n("Cast", ["dilmax"], "keep_u8", to=U8)                       # [1,1,W,W] uint8
    # Two-stage pad of the WxW keep window back to 30x30:
    #  (a) pad the OFF-cell in-grid border (rows/cols < OFF and the trailing in-grid
    #      rows up to 15) with 0/False -> noise there is blanked (keep=False).
    #  (b) pad the off-grid tail (>=16) with 1/True so Where selects the all-zero input
    #      there instead of the bg one-hot (task193 idiom).  uint8 Pad (opset-11) is
    #      allowed; bool Pad is not.
    end_pad = 16 - (OFF + W)
    pad_a = [0, 0, OFF, OFF, 0, 0, end_pad, end_pad]               # -> 16x16
    init("kpad_a", np.array(pad_a, np.int64), np.int64)
    init("ZEROU8", np.array(0, np.uint8), np.uint8)
    n("Pad", ["keep_u8", "kpad_a", "ZEROU8"], "keep16_u8", mode="constant")
    pad_b = [0, 0, 0, 0, 0, 0, 14, 14]                            # 16x16 -> 30x30
    init("kpad_b", np.array(pad_b, np.int64), np.int64)
    init("ONEU8", np.array(1, np.uint8), np.uint8)
    n("Pad", ["keep16_u8", "kpad_b", "ONEU8"], "keep30_u8", mode="constant")
    n("Cast", ["keep30_u8"], "keep_b", to=BOOL)                   # [1,1,30,30] bool

    # ---- single Where -> FREE [1,10,30,30] output --------------------------
    bg = np.zeros((1, 10, 1, 1), np.float32)
    bg[0, 0, 0, 0] = 1.0
    init("bg_onehot", bg, np.float32)
    n("Where", ["keep_b", "input", "bg_onehot"], "output")

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task222", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
