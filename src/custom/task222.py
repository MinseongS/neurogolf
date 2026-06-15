"""Task 222 (ARC-AGI 91714a58): keep only cells inside the solid box's bbox.

Rule (from generator): a 16x16 grid holds scattered random single pixels plus ONE
solid axis-aligned filled rectangle (the "box") of a single colour.  Output keeps
every cell INSIDE the box's bounding rectangle (its own colour) and blacks out
(0) everything else.

Robust box detection (verified on 2000 fresh samples):
  The box colour is the UNIQUE colour that forms 2x2 solid blocks: the box (area
  >= 9, both sides >= 2) always contains >= 4 distinct 2x2 solid same-colour
  blocks, whereas scattered random pixels (guaranteed by the generator to have no
  same-colour 4-neighbour) form at most 1.  So `cnt2x2[c] >= 2` (channels 1..9)
  isolates the box colour exactly.

  The box colour's presence plane is the solid rectangle plus possibly a few
  isolated box-colour singletons elsewhere.  Box rows are exactly the rows with a
  HORIZONTAL box-colour pair (box width >= 2, singletons are isolated -> no pair);
  box cols are the cols with a VERTICAL box-colour pair.  Their outer product is
  precisely the box bbox.

Floor-break: separable bbox = rowhas (30x1) x colhas (1x30); one final Where keeps
`input` inside the bbox, writes black (bg one-hot) for in-grid cells outside it,
and leaves off-grid cells all-zero.  All spatial intermediates are fp16 / uint8.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..builders import _model

F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype=np.float32):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **kw):
        nodes.append(helper.make_node(op, ins, [out], **kw))
        return out

    # ---- per-colour 2x2 solid-block count -> pick the box colour ----
    # Slice straight from the f32 input to the 9 coloured channels over the 16x16
    # active grid (blocks live only here), then cast that small slice to fp16: every
    # per-channel plane is [1,9,15,15] (2025 B) not [1,10,29,29].  Channel 0
    # (background, blocks everywhere) is dropped so argmax never picks it; the box
    # colour index is recovered as argmax + 1.
    init("sl_st", np.array([0, 1, 0, 0], np.int64), np.int64)
    init("sl_en", np.array([1, 10, 16, 16], np.int64), np.int64)
    init("sl_ax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "sl_st", "sl_en", "sl_ax"], "xc32")  # [1,9,16,16] f32
    n("Cast", ["xc32"], "xc", to=F16)

    init("W22", np.ones((9, 1, 2, 2), np.float16), np.float16)
    # conv bias -3.5 then Relu: 0.5 exactly where a 2x2 block (sum==4), else 0.
    init("B22", np.full((9,), -3.5, np.float16), np.float16)
    n("Conv", ["xc", "W22", "B22"], "blk_sum", group=9)       # [1,9,15,15] f16
    n("Relu", ["blk_sum"], "blkr")                            # 0.5 at blocks else 0
    n("ReduceSum", ["blkr"], "cnt2x2", axes=[2, 3], keepdims=1)   # [1,9,1,1] 0.5*count
    n("Squeeze", ["cnt2x2"], "cnt2x2s", axes=[2, 3])          # [1,9]
    n("ArgMax", ["cnt2x2s"], "argc", axis=1, keepdims=0)      # [1] in 0..8

    # ---- box-colour presence plane via channel Gather on the 16x16 slice ----
    n("Gather", ["xc", "argc"], "boxpres16", axis=1)          # [1,1,16,16] f16
    init("onehalf", np.array(1.5, np.float16), np.float16)
    init("half", np.array(0.5, np.float16), np.float16)

    # ---- horizontal pair -> box rows ; vertical pair -> box cols (16x16) ----
    init("Wh", np.ones((1, 1, 1, 2), np.float16), np.float16)
    n("Conv", ["boxpres16", "Wh"], "hsum")                    # [1,1,16,15] f16
    n("Greater", ["hsum", "onehalf"], "hpair")
    n("Cast", ["hpair"], "hpairf", to=F16)
    n("ReduceSum", ["hpairf"], "rowsum", axes=[3], keepdims=1)    # [1,1,16,1]
    n("Greater", ["rowsum", "half"], "rowhas")                # [1,1,16,1] bool box rows

    init("Wv", np.ones((1, 1, 2, 1), np.float16), np.float16)
    n("Conv", ["boxpres16", "Wv"], "vsum")                    # [1,1,15,16] f16
    n("Greater", ["vsum", "onehalf"], "vpair")
    n("Cast", ["vpair"], "vpairf", to=F16)
    n("ReduceSum", ["vpairf"], "colsum", axes=[2], keepdims=1)    # [1,1,1,16]
    n("Greater", ["colsum", "half"], "colhas")                # [1,1,1,16] bool box cols

    # ---- bbox = rowhas & colhas (16x16) ; pad to 30x30 ; keep input inside ----
    n("And", ["rowhas", "colhas"], "inbox16")                 # [1,1,16,16] bool
    n("Cast", ["inbox16"], "inbox16f", to=F16)
    # opset-10 Pad (attribute pads): pad rows/cols 16->30 with 0 (outside the box).
    n("Pad", ["inbox16f"], "inboxf", mode="constant", value=0.0,
      pads=[0, 0, 0, 0, 0, 0, 14, 14])                        # [1,1,30,30] f16
    n("Greater", ["inboxf", "half"], "inbox")                 # [1,1,30,30] bool

    # The grid is always exactly 16x16, so off-grid (= rows>=16 or cols>=16) is a
    # constant mask; off-grid cells (all-zero input) must stay all-zero, not black.
    offgrid = np.ones((1, 1, 30, 30), bool)
    offgrid[0, 0, :16, :16] = False
    init("offgrid", offgrid, bool)
    n("Or", ["inbox", "offgrid"], "keep")                     # input where in-box or off-grid
    bg = np.zeros((1, 10, 1, 1), np.float32); bg[0, 0, 0, 0] = 1.0
    init("bg", bg)
    n("Where", ["keep", "input", "bg"], "output")

    return _model(nodes, inits)
