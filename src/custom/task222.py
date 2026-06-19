"""task222 (ARC-AGI 91714a58) -- "keep the one solid box, drop the noise".

Rule (from the generator):
  A 16x16 grid holds one SOLID axis-aligned rectangle of a single `color`
  (width,height in [2,8], 9 <= area <= 16, placed strictly interior so the box lives in
  rows/cols 1..14) on a field of random single-pixel noise of arbitrary colours.  The
  generator guarantees the box `color` has NO same-colour 4-neighbour anywhere outside
  the box.  OUTPUT = INPUT with everything but the box zeroed.

Exact local rule (brute-verified 0 fails / 50000 fresh):
  keep(r,c) iff (r,c) belongs to a fully-filled, single-non-zero-colour block of shape
  3x3 OR 2x4 OR 4x2.  Every valid box contains such a sub-block and these windows tile
  the box fully; ~50%-density noise needs >=8 same-colour pixels in that exact shape to
  false-fire (-> ~0 probability).

Encoding (route the 10-ch expansion into the FREE Where output, task193 idiom):
  colf30[1,1,30,30] f32 = Conv1x1(input,[0,1,..,9]) -> per-cell colour index (bg=0).
  Crop to the 14x14 active grid, cast f16 -> v.
  Per shape (h,w): window-sum S = Conv(v,ones), window-max mx = MaxPool(v); the window is
  uniform-nonzero iff S == (h*w)*mx and mx>0.  Dilate each TL seed over its hxw footprint
  with a padded BR-anchored MaxPool (zero-pad fuses the WxW restore + spread in one op).
  keep = MAX of the three dilations -> uint8; Pad to 16x16 (0 border) then 30x30 (1
  off-grid so Where picks the all-zero input there, not the bg one-hot).
  output = Where(keep_bool, input, bg_onehot).
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

    init("WPACK", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    n("Conv", ["input", "WPACK"], "colf30", kernel_shape=[1, 1])   # [1,1,30,30] f32

    init("c_s", np.array([OFF, OFF], np.int64), np.int64)
    init("c_e", np.array([OFF + W, OFF + W], np.int64), np.int64)
    init("c_ax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["colf30", "c_s", "c_e", "c_ax"], "colf")           # [1,1,W,W] f32
    n("Cast", ["colf"], "v", to=F16)                               # [1,1,W,W] f16

    # NOTE: the window-non-bg gate (mx>0) is DROPPED.  An all-background window also
    # satisfies S==k*mx (0==0) and would "seed" True, but its dilation only ever covers
    # background cells, and a kept background cell routes through Where to input==bg
    # (ch0=1 in-grid, all-zero off-grid) == the false branch -> harmless (verified 0 fails
    # / 12000 fresh).  This removes the per-shape Greater + And planes.
    keep_terms = []
    for h, w in [(3, 3), (2, 4), (4, 2)]:
        tag = f"{h}x{w}"
        k = float(h * w)
        # uniform window  <=>  window-sum == k * window-max  (both integer, fp16-exact
        # for sums <= k*9 <= 144 << 2048).  An all-bg window (0==0) also seeds True but
        # dilates only over bg -> Where->input==bg, harmless (no non-bg gate needed).
        n("MaxPool", ["v"], f"mx_{tag}", kernel_shape=[h, w], strides=[1, 1])
        init(f"ONES_{tag}", np.ones((1, 1, h, w), np.float16), np.float16)
        n("Conv", ["v", f"ONES_{tag}"], f"S_{tag}", kernel_shape=[h, w])
        init(f"K_{tag}", np.array(k, np.float16), np.float16)
        n("Mul", [f"mx_{tag}", f"K_{tag}"], f"kmx_{tag}")          # k*max
        n("Equal", [f"S_{tag}", f"kmx_{tag}"], f"seedb_{tag}")    # uniform window
        n("Cast", [f"seedb_{tag}"], f"seed_{tag}", to=F16)        # f16 {0,1}
        n("MaxPool", [f"seed_{tag}"], f"dil_{tag}",
          kernel_shape=[h, w], strides=[1, 1],
          pads=[h - 1, w - 1, h - 1, w - 1])
        keep_terms.append(f"dil_{tag}")

    n("Max", keep_terms, "dilmax")
    n("Cast", ["dilmax"], "keep_u8", to=U8)

    end_pad = 16 - (OFF + W)
    init("kpad_a", np.array([0, 0, OFF, OFF, 0, 0, end_pad, end_pad], np.int64), np.int64)
    init("ZEROU8", np.array(0, np.uint8), np.uint8)
    n("Pad", ["keep_u8", "kpad_a", "ZEROU8"], "keep16_u8", mode="constant")
    init("kpad_b", np.array([0, 0, 0, 0, 0, 0, 14, 14], np.int64), np.int64)
    init("ONEU8", np.array(1, np.uint8), np.uint8)
    n("Pad", ["keep16_u8", "kpad_b", "ONEU8"], "keep30_u8", mode="constant")
    n("Cast", ["keep30_u8"], "keep_b", to=BOOL)

    bg = np.zeros((1, 10, 1, 1), np.float32)
    bg[0, 0, 0, 0] = 1.0
    init("bg_onehot", bg, np.float32)
    n("Where", ["keep_b", "input", "bg_onehot"], "output")

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task222", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
