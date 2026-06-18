"""task318 (ARC-AGI ce4f8723) — overlay two stacked half-grids, recolour to green.

Rule (from the ARC-GEN generator, verified fresh):
  A size=4 task.  INPUT is a `size` x (2*size+1) grid: the TOP block (rows 0..3)
  holds only colour-1 pixels, the BOTTOM block (rows 5..8) holds only colour-2
  pixels, and the single middle row (row 4) is a yellow separator.  OUTPUT is
  size x size: a cell (r,c) is GREEN (colour 3) iff there is a pixel at (r,c) in
  the TOP block OR the BOTTOM block; otherwise background (0).  Off-grid (>=4)
  is all-zero (the target is np.zeros, only the 4x4 region is set).

Encoding (no full 30x30 plane):
  top  = input[:, 1:2, 0:4, 0:4]   (the only colour present up top is ch1)
  bot  = input[:, 2:3, 5:9, 0:4]   (the only colour present below is ch2)
  green = (top > 0) OR (bot > 0)            [1,1,4,4] bool
  ch0   = NOT green
  Concat ch0, green into channels [0..9] (only 0 and 3 are non-zero) as uint8,
  then Pad to [1,10,30,30] (value 0 -> off-grid all zero).  Output declared uint8;
  the harness scores (out > 0), so the {0,1} planes pass identically.

  All working tensors live on the 4x4 active region, so the dominant intermediate
  is the [1,10,4,4] uint8 carrier (160B) -- far below the public net's ~734B.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

S = 30
G = 4  # active grid size


def build(task):
    inits, nodes = [], []
    seen = set()

    def init(name, arr, dt):
        if name in seen:
            return name
        seen.add(name)
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dt), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    U8 = TensorProto.UINT8
    B = TensorProto.BOOL
    NI = TensorProto.INT64

    # ---- slice the two relevant colour channels / blocks --------------------
    # top: ch1, rows 0..3, cols 0..3
    init("t_s", np.array([1, 0, 0], np.int64), np.int64)
    init("t_e", np.array([2, G, G], np.int64), np.int64)
    init("ax123", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "t_s", "t_e", "ax123"], "top")     # [1,1,4,4] fp32

    # bottom: ch2, rows 5..8, cols 0..3
    init("b_s", np.array([2, G + 1, 0], np.int64), np.int64)
    init("b_e", np.array([3, 2 * G + 1, G], np.int64), np.int64)
    n("Slice", ["input", "b_s", "b_e", "ax123"], "bot")     # [1,1,4,4] fp32

    # green occupancy = top>0 OR bot>0
    init("z0", np.array(0.0, np.float32), np.float32)
    n("Greater", ["top", "z0"], "tg")                       # bool [1,1,4,4]
    n("Greater", ["bot", "z0"], "bg")                       # bool [1,1,4,4]
    n("Or", ["tg", "bg"], "green")                          # bool [1,1,4,4]
    n("Not", ["green"], "ch0b")                             # bool [1,1,4,4]

    # cast to uint8 and assemble [1,10,4,4]
    n("Cast", ["green"], "greenu", to=U8)                   # uint8 [1,1,4,4]
    n("Cast", ["ch0b"], "ch0u", to=U8)                      # uint8 [1,1,4,4]
    # zero channel (uint8) for the unused colour channels (constant init)
    init("zu", np.zeros((1, 1, G, G), np.uint8), np.uint8)  # uint8 [1,1,4,4] all 0

    chans = ["ch0u", "zu", "zu", "greenu"] + ["zu"] * 6     # ch0=bg, ch3=green
    n("Concat", chans, "small", axis=1)                     # uint8 [1,10,4,4]

    # pad to [1,10,30,30] (value 0 -> off-grid all zero)
    init("pads", np.array([0, 0, 0, 0, 0, 0, S - G, S - G], np.int64), np.int64)
    init("pv", np.array(0, np.uint8), np.uint8)
    n("Pad", ["small", "pads", "pv"], "output", mode="constant")

    out_vi = helper.make_tensor_value_info("output", U8, [1, 10, S, S])
    in_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, S, S])
    graph = helper.make_graph(nodes, "task318", [in_vi], [out_vi], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = IR_VERSION
    return model
