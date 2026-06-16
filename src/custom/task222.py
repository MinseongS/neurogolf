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
      dims>=3 so it contains a 3x3.  Every box cell lies inside one of these windows
      (3x3/2x4/4x2 windows tile a solid rectangle fully).
    * Noise occasionally forms a spurious solid 2x2 (~1/60) and very rarely a 2x3
      (~1/20000); requiring a 3x3 / 2x4 / 4x2 (>=8 same-colour pixels in that exact
      shape among ~50%-density random noise) drives false positives to ~0.

Encoding (floor-break -- route the 10-ch expansion into the FREE output, task193 idiom):
  colf[1,1,30,30] f32 = Conv1x1(input,[0,1,..,9]) -> per-cell colour index (bg/off-grid=0).
  crop to the 16x16 active region, cast f16.  For each block shape (h,w):
    mx = MaxPool(v,(h,w))      ; mn = -MaxPool(-v,(h,w))         (TL-anchored, no pad)
    seed = (mx==mn) AND (mn>0)  -> window is uniform & non-background
    pad seed back to 16x16 (TL); dilate = MaxPool(seed,(h,w)) bottom-right anchored
  keep = OR of the three dilations; pad to 30x30 (False outside the 16x16 grid).
  output = Where(keep, input, bg_onehot).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- colour-index plane (entry f32 read of the input) ------------------
    wpack = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)  # [0,1,..,9]
    init("WPACK", wpack, np.float32)
    n("Conv", ["input", "WPACK"], "colf30", kernel_shape=[1, 1])  # [1,1,30,30] f32

    # crop to the 16x16 active region (grid size is fixed 16; box is interior).
    init("c_s", np.array([0, 0], np.int64), np.int64)
    init("c_e", np.array([16, 16], np.int64), np.int64)
    init("c_ax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["colf30", "c_s", "c_e", "c_ax"], "colf16")        # [1,1,16,16] f32
    n("Cast", ["colf16"], "v", to=F16)                            # [1,1,16,16] f16
    n("Neg", ["v"], "nv")                                         # for min via max

    init("NHALF", np.array(-0.5, np.float16), np.float16)
    init("ZERO0", np.array(0.0, np.float16), np.float16)
    init("ZERO5", np.array(0.5, np.float16), np.float16)

    keep_terms = []
    SZ = 16
    for h, w in [(3, 3), (2, 4), (4, 2)]:
        tag = f"{h}x{w}"
        oh, ow = SZ - h + 1, SZ - w + 1
        # TL-anchored window max(v) and max(-v) = -min(v)   (no padding).
        n("MaxPool", ["v"], f"mx_{tag}", kernel_shape=[h, w], strides=[1, 1])
        n("MaxPool", ["nv"], f"nmn_{tag}", kernel_shape=[h, w], strides=[1, 1])
        # range = max - min = mx + nmn ; uniform-colour window iff range == 0
        # (Add keeps a plane the same size min-via-Neg would, so no extra cost).
        n("Add", [f"mx_{tag}", f"nmn_{tag}"], f"rng_{tag}")       # [1,1,oh,ow] f16
        n("Equal", [f"rng_{tag}", "ZERO0"], f"uni_{tag}")        # max==min
        # non-background: min > 0  <=>  nmn (=-min) < -0.5
        n("Less", [f"nmn_{tag}", "NHALF"], f"nz_{tag}")
        n("And", [f"uni_{tag}", f"nz_{tag}"], f"seedb_{tag}")     # [1,1,oh,ow] bool
        n("Cast", [f"seedb_{tag}"], f"seed_{tag}", to=F16)        # f16 {0,1}
        # pad seed back to 16x16 (place at top-left, fill 0).
        pad = np.array([0, 0, 0, 0, 0, 0, SZ - oh, SZ - ow], np.int64)
        init(f"spad_{tag}", pad, np.int64)
        init(f"z16_{tag}", np.array(0.0, np.float16), np.float16)
        n("Pad", [f"seed_{tag}", f"spad_{tag}", f"z16_{tag}"], f"seedpad_{tag}",
          mode="constant")                                        # [1,1,16,16] f16
        # dilate over the window footprint: bottom-right anchored MaxPool.
        n("MaxPool", [f"seedpad_{tag}"], f"dil_{tag}",
          kernel_shape=[h, w], strides=[1, 1], pads=[h - 1, w - 1, 0, 0])
        keep_terms.append(f"dil_{tag}")

    # keep = ANY block-shape dilation fired: Max over the three f16 maps, one Greater.
    n("Max", [keep_terms[0], keep_terms[1]], "dil01")             # [1,1,16,16] f16
    n("Max", ["dil01", keep_terms[2]], "dilmax")
    n("Greater", ["dilmax", "ZERO5"], "keep16b")                  # [1,1,16,16] bool
    n("Cast", ["keep16b"], "keep16_u8", to=TensorProto.UINT8)
    # pad to 30x30 with True outside the 16x16 grid: those off-grid cells are all-zero
    # in the input, so selecting `input` there yields the all-zero target (NOT the
    # bg one-hot, which would wrongly set channel-0 high off-grid).  task193 idiom:
    # selcond = keep OR off-grid.
    init("kpad", np.array([0, 0, 0, 0, 0, 0, 14, 14], np.int64), np.int64)
    init("ONEU8", np.array(1, np.uint8), np.uint8)
    n("Pad", ["keep16_u8", "kpad", "ONEU8"], "keep30_u8", mode="constant")
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
