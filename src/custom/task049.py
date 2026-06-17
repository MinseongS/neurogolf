"""task049 (ARC-AGI 23b5c85d) — emit a solid block of the rarest colour sized to
the smallest box.

Rule (from the generator):
  N (2..5) solid axis-aligned rectangles of DISTINCT colours are drawn on a
  width x height (10..20) black(0) canvas.  Boxes can overlap; the LAST box drawn
  is the one with the strictly-smallest width AND strictly-smallest height, and
  its colour `colors[-1]` is guaranteed (by generator rejection sampling) to be
  the RAREST colour by visible pixel count.  Because it is drawn last it is never
  occluded, so its full solid rectangle survives.
  OUTPUT = a solid rectangle of shape  talls[-1] x wides[-1]  filled with
  colours[-1], anchored at the top-left (0,0).  (The output grid is exactly that
  size; in the one-hot harness encoding every cell outside it is all-channels-0.)

Reformulation (fully separable, no per-cell colour plane at all):
  rare colour = argmin over foreground colours of pixel-count.
    cnt = ReduceSum(input, axes=[2,3])  -> [1,10,1,1]
    mask ch0 (background) and zero-count (unused) channels to +BIG, take ReduceMin
    -> rare = Equal(masked_cnt, min)  -> [1,10,1,1] bool one-hot  (unique winner).
  tall = number of rows the rare colour occupies; wide = number of cols.
    Extract the single rare-colour plane with a 1x1 Conv whose WEIGHT is the
    runtime rare one-hot [1,10,1,1] (out_ch=1, in_ch=10) -> rareplane[1,1,30,30].
    (ORT accepts a runtime-computed Conv weight.)  This collapses channel-select
    + plane-extract into ONE op, avoiding a 10-channel [1,10,30,*] occupancy plane.
    rowocc = ReduceMax(rareplane, axes=[3]) [1,1,30,1] ; tall = ReduceSum(rowocc).
    colocc = ReduceMax(rareplane, axes=[2]) [1,1,1,30] ; wide = ReduceSum(colocc).
    (the rare box is a solid contiguous rectangle so #occupied-rows == its height.)
  output[ch,r,c] = rare[ch] AND (r < tall) AND (c < wide)
    rowin = (rowramp < tall) [1,1,30,1] ; colin = (colramp < wide) [1,1,1,30]
    output = And(rare_bool[1,10,1,1], And(rowin, colin))  -> FREE [1,10,30,30] BOOL.

Dominant intermediate: the single fp32 rare plane [1,1,30,30] (3600B) — the
documented colour-plane floor; everything else is <=900B (the rect bool plane) or
scalar/1-D.  The 10-channel expansion lands in the free bool `output`.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64

BIG = 1.0e4  # bigger than any possible pixel count (<= 20*20=400)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- per-channel pixel counts -> rare colour one-hot --------------------
    n("ReduceSum", ["input"], "cnt", axes=[2, 3], keepdims=1)   # [1,10,1,1] f32

    # mask: +BIG on background channel 0 (always present), +BIG on unused (0-cnt)
    bgmask = np.zeros((1, 10, 1, 1), np.float32)
    bgmask[0, 0, 0, 0] = BIG
    init("bgmask", bgmask, np.float32)
    init("ZERO32", np.array(0.0, np.float32), np.float32)
    init("BIG32", np.array(BIG, np.float32), np.float32)

    n("Add", ["cnt", "bgmask"], "cnt_b")                        # ch0 -> >=BIG
    n("Equal", ["cnt_b", "ZERO32"], "is_zero")                  # unused (cnt==0)
    n("Where", ["is_zero", "BIG32", "cnt_b"], "cnt_m")          # [1,10,1,1] f32

    n("ReduceMin", ["cnt_m"], "mincnt", keepdims=1)             # [1,1,1,1] f32
    n("Equal", ["cnt_m", "mincnt"], "rare")                     # [1,10,1,1] bool

    # ---- extract the single rare-colour plane via a 1x1 Conv whose weight is
    # the runtime rare one-hot [1,10,1,1] (out_ch=1, in_ch=10) -> [1,1,30,30].
    # This avoids ever materialising a 10-channel [1,10,30,*] occupancy plane;
    # the single fp32 rare plane (3600B) is the only large intermediate.
    n("Cast", ["rare"], "rare_f", to=F32)                       # [1,10,1,1] f32 weight
    n("Conv", ["input", "rare_f"], "rareplane")                 # [1,1,30,30] f32

    # count occupied rows / cols of the solid rare rectangle
    n("ReduceMax", ["rareplane"], "rowocc", axes=[3], keepdims=1)  # [1,1,30,1]
    n("ReduceMax", ["rareplane"], "colocc", axes=[2], keepdims=1)  # [1,1,1,30]
    n("ReduceSum", ["rowocc"], "tall_f", axes=[1, 2, 3], keepdims=1)  # scalar
    n("ReduceSum", ["colocc"], "wide_f", axes=[1, 2, 3], keepdims=1)  # scalar
    n("Cast", ["tall_f"], "tall", to=F16)
    n("Cast", ["wide_f"], "wide", to=F16)

    # ---- separable origin-anchored rectangle --------------------------------
    init("rowramp", np.arange(30, dtype=np.float16).reshape(1, 1, 30, 1), np.float16)
    init("colramp", np.arange(30, dtype=np.float16).reshape(1, 1, 1, 30), np.float16)
    n("Less", ["rowramp", "tall"], "rowin")                     # [1,1,30,1] bool
    n("Less", ["colramp", "wide"], "colin")                     # [1,1,1,30] bool

    n("And", ["rowin", "colin"], "rect")                        # [1,1,30,30] bool
    n("And", ["rare", "rect"], "output")                        # FREE [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task049", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
