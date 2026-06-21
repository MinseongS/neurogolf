"""task222 (ARC-AGI 91714a58) -- "keep the one solid box, drop the noise".

Rule (from the generator, fixed 16x16 grid):
  The grid is a 16x16 field of random single-pixel colour noise (density 0.5) plus ONE
  SOLID axis-aligned rectangle (the "box") of a single colour, width,height in [2,8],
  9 <= area <= 16, placed strictly interior (box rows/cols within 1..14).  The generator
  guarantees the box colour has NO same-colour 4-neighbour outside the box (so box-colour
  noise is isolated) and that noise never extends the box.  OUTPUT = input restricted to
  the box rectangle; every other in-grid cell becomes background (channel 0); off-grid
  cells (rows/cols 16..29) stay all-zero.

Closed-form rule (brute-verified 0 fails / 20000 fresh):
  keep(r,c) iff (r,c) belongs to a fully-filled, SINGLE-NON-ZERO-COLOUR window of shape
  3x3 OR 2x5 OR 5x2.  Every valid box (both dims >=2, area >=9) contains at least one such
  window and these windows tile the box completely; ~50%-density noise essentially never
  produces 9-10 same-colour pixels in one of these exact shapes (measured 0/20000).  The
  smaller 2x3/3x2 set DOES leak (~3/20000), so a >=9-cell window is required for safety;
  {3x3,2x5,5x2} ties {3x3,2x4,4x2} on exactness but uses smaller valid-conv intermediates.

Encoding (all-UINT8 pipeline at opset 13; routes the 10-ch expansion into the FREE Where
output, task193 idiom + the kojimar QLinearConv idiom):
  colf30 [1,1,30,30] f32 = Conv1x1(input, [0,1,..,9]) -> per-cell colour index (bg=0).
    The one irreducible 3600B entry plane (Conv inherits the fp32 input dtype).  We then
    Slice to the 14x14 active grid (box lives in rows/cols 1..14) and Cast to UINT8 -> v.
  Per shape (h,w), the uniformity test runs in EXACT integer uint8 arithmetic (half the
  fp16 footprint; uint8 MaxPool/Max/Equal/QLinearConv all run under ORT_DISABLE_ALL at
  opset 13, NOT at opset 11):
    S   = window-sum  via QLinearConv(v, ones)  [scales 1, zp 0 -> exact integer sum,
          max 5*2*9=90 < 256 -> no overflow]
    mx  = window-max  via uint8 MaxPool
    kmx = k*mx        via 1x1 QLinearConv(mx, [k])  [exact, <=90 < 256]
    seed = Equal(S, kmx)  -> a window is uniform-nonzero iff S == k*mx.  (All-bg windows
          give 0==0 and also seed, but their dilation only ever covers bg cells -> Where
          picks the bg branch == correct, so no >0 gate is needed.)
  Cast seed -> uint8, dilate over the hxw footprint with a bottom-right-padded uint8
  MaxPool, then MAX the three dilations -> keep_u8 (box-cover mask).
  Cast keep -> bool; Pad active->16 (False: in-grid border -> bg) then 16->30 (True:
  off-grid -> Where picks the all-zero input).  Pad accepts BOOL at opset 13, so only ONE
  30x30 plane (keep_b) is paid.  output = Where(keep_b, input, bg_onehot).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8

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

    # ---- colour-index plane (one 3600B fp32 entry), crop, cast to UINT8 ---------
    init("WPACK", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    n("Conv", ["input", "WPACK"], "colf30", kernel_shape=[1, 1])   # [1,1,30,30] f32
    init("c_s", np.array([OFF, OFF], np.int64), np.int64)
    init("c_e", np.array([OFF + W, OFF + W], np.int64), np.int64)
    init("c_ax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["colf30", "c_s", "c_e", "c_ax"], "colf")           # [1,1,W,W] f32
    n("Cast", ["colf"], "v", to=U8)                                # [1,1,W,W] uint8 (0..9)

    # ---- per-shape uniform-nonzero seeds + dilation, ENTIRELY in UINT8 ----------
    # Uniformity test S == k*mx in exact integer uint8 arithmetic (see module docstring).
    # All scale/zero-point inputs are SCALARS (shape []) and shared across the six
    # QLinearConvs (ORT accepts a scalar weight-scale/zp; verified) -> 3 quant params total,
    # each quant conv computing an EXACT integer sum.
    init("QS1", np.array(1.0, np.float32), np.float32)        # scale 1.0 (x/weight/y)
    init("QZ0u", np.array(0, np.uint8), np.uint8)             # uint8 zero-point (x/y)
    init("QZ0i", np.array(0, np.int8), np.int8)               # int8 weight zero-point

    _wq_seen = {}

    def qconv(xname, wq, out, ks, wname=None):
        # QLinearConv with all scales=1, zero-points=0: y = exact integer sum(x*w).
        # Identical weights are shared (deduped) to keep the param count minimal.
        wq = wq.astype(np.int8)
        key = wq.tobytes() + bytes(wq.shape)
        if key in _wq_seen:
            wn = _wq_seen[key]
        else:
            wn = wname or (out + "_wq")
            init(wn, wq, np.int8)
            _wq_seen[key] = wn
        n("QLinearConv",
          [xname, "QS1", "QZ0u", wn, "QS1", "QZ0i", "QS1", "QZ0u"],
          out, kernel_shape=ks)

    keep_terms = []
    for h, w in [(3, 3), (2, 5), (5, 2)]:
        tag = f"{h}x{w}"
        k = h * w
        qconv("v", np.ones((1, 1, h, w)), f"S_{tag}", [h, w])     # uint8 window-sum
        n("MaxPool", ["v"], f"mx_{tag}", kernel_shape=[h, w], strides=[1, 1])  # uint8 max
        qconv(f"mx_{tag}", np.array([[[[k]]]]), f"kmx_{tag}", [1, 1],
              wname=f"KW_{k}")                                     # uint8 k*max (shared by k)
        n("Equal", [f"S_{tag}", f"kmx_{tag}"], f"seedb_{tag}")    # uniform window (bool)
        n("Cast", [f"seedb_{tag}"], f"seed_{tag}", to=U8)         # uint8 {0,1}
        n("MaxPool", [f"seed_{tag}"], f"dil_{tag}",               # uint8 dilation
          kernel_shape=[h, w], strides=[1, 1],
          pads=[h - 1, w - 1, h - 1, w - 1])
        keep_terms.append(f"dil_{tag}")
    n("Max", keep_terms, "keep_u8")                                # uint8 box-cover mask [WxW]
    n("Cast", ["keep_u8"], "keep_b14", to=BOOL)                    # bool at WxW

    # ---- pad active->16 (False: in-grid border -> bg) then ->30 (True: off-grid ->
    # Where picks all-zero input).  Pad accepts BOOL at opset 13, so we pad the bool mask
    # directly (one 30x30 plane instead of a uint8 + a bool one).
    end_pad = 16 - (OFF + W)
    init("kpad_a", np.array([0, 0, OFF, OFF, 0, 0, end_pad, end_pad], np.int64), np.int64)
    init("FALSEB", np.array(False, np.bool_), np.bool_)
    n("Pad", ["keep_b14", "kpad_a", "FALSEB"], "keep16_b", mode="constant")
    init("kpad_b", np.array([0, 0, 0, 0, 0, 0, 14, 14], np.int64), np.int64)
    init("TRUEB", np.array(True, np.bool_), np.bool_)
    n("Pad", ["keep16_b", "kpad_b", "TRUEB"], "keep_b", mode="constant")

    bg = np.zeros((1, 10, 1, 1), np.float32)
    bg[0, 0, 0, 0] = 1.0
    init("bg_onehot", bg, np.float32)
    n("Where", ["keep_b", "input", "bg_onehot"], "output")

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task222", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 13)])
