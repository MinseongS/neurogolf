"""Task 207 (ARC-AGI 88a62173) — output the odd-one-out 2x2 sprite.

Rule (from the generator): a 5x5 grid holds four 2x2 sprites whose top-left
corners are (0,0),(0,3),(3,0),(3,3) (a 3-stride 2x2 layout). Three sprites are
identical ("same" pattern); exactly ONE ("diff") has a different pattern. Every
coloured pixel uses a single colour `color`. The 2x2 OUTPUT is the `diff`
sprite's pattern in that colour.

Closed-form recovery (NO argmax / NO block identification / NO gather):
  Stack the four 2x2 blocks and, PER CHANNEL PER CELL, count how many of the
  four blocks set that channel: cnt = block0+block1+block2+block3.
  Because 3 blocks are identical and 1 differs, at every (cell, channel) the
  count is in {0,1,3,4} (never 2 -- the 3 identical blocks contribute 0 or 3,
  the diff block contributes 0 or 1). The DIFF block's value at a cell/channel
  is 1 exactly when cnt in {1,4}:
    cnt=4 -> all four agree (incl diff)  -> diff=1
    cnt=1 -> only the diff sets it       -> diff=1
    cnt=3 -> the three "same" set it, diff does NOT -> diff=0
    cnt=0 -> none set it                 -> diff=0
  This holds for EVERY channel (incl. background ch0), so `(cnt==1)|(cnt==4)`
  reconstructs the diff sprite's one-hot directly. Pad to 30x30 (zeros).

Memory: only tiny [1,10,2,2] tensors (40 elems). The full-canvas [1,10,30,30]
result is the FREE `output` of the final Pad. Nothing else is full-size.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
B = TensorProto.BOOL


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- slice the four 2x2 sprite blocks (top-left corners on a 3-stride) ----
    # rows/cols {0:2} and {3:5}; each block [1,10,2,2] fp32 (40 elems, 160B).
    blocks = []
    for bi, (r0, c0) in enumerate([(0, 0), (0, 3), (3, 0), (3, 3)]):
        s = init(f"s{bi}", np.array([r0, c0], np.int64), np.int64)
        e = init(f"e{bi}", np.array([r0 + 2, c0 + 2], np.int64), np.int64)
        ax = init(f"ax{bi}", np.array([2, 3], np.int64), np.int64)
        n("Slice", ["input", s, e, ax], f"blk{bi}")  # [1,10,2,2] fp32
        blocks.append(f"blk{bi}")

    # ---- per-channel/per-cell count over the four blocks ----
    n("Sum", blocks, "cnt")  # [1,10,2,2] fp32, values in {0,1,3,4}

    # ---- diff one-hot = (cnt == 1) OR (cnt == 4) ----
    init("one", np.array(1.0, np.float32), np.float32)
    init("four", np.array(4.0, np.float32), np.float32)
    n("Equal", ["cnt", "one"], "eq1")     # [1,10,2,2] bool
    n("Equal", ["cnt", "four"], "eq4")    # [1,10,2,2] bool
    n("Or", ["eq1", "eq4"], "maskb")      # [1,10,2,2] bool
    # Pad rejects bool -> cast to fp16 (tiny), pad with 0, route into output.
    n("Cast", ["maskb"], "maskf", to=F16)  # [1,10,2,2] fp16 (80B)

    init("pads", np.array([0, 0, 0, 0, 0, 0, 28, 28], np.int64), np.int64)
    init("zf16", np.array(0.0, np.float16), np.float16)
    n("Pad", ["maskf", "pads", "zf16"], "output", mode="constant")  # FREE output

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F16, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task207", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
