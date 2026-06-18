"""task052 (ARC-AGI 25d8a9c8) — "per-row uniformity -> gray, else black (3x3)".

Rule (from the generator):
  Input is a 3x3 grid of colours. Output is a 3x3 grid where every cell of row r is
  gray(5) if input row r is monochrome (all 3 cells equal), otherwise black(0).
  Output value depends ONLY on the row (separable: per-row value broadcast across cols).
  Off-grid cells (r>=3 or c>=3) are all background (every channel 0).

Encoding (floor-break: never materialize a [1,10,30,30] plane; route the 10-ch
expansion into the FREE output as the final op, keep all working planes on the 3x3
active region):
  slice the active 3x3 block, cast fp16.
  count[k,r] = ReduceSum over the column axis -> [1,10,3,1]  (per-row per-channel count).
  uniform[r] = (ReduceMax_k count == 3)  -> u bool [1,1,3,1].
  chan_oh[1,10,3,1] = Where(u, gray_onehot[1,10,1,1], black_onehot[1,10,1,1])  -> picks
    channel 5 on uniform rows, channel 0 otherwise (per row, no column dependence).
  small[1,10,3,3] = And(chan_oh, colin[1,1,1,3])  (broadcast over the 3 columns) cast uint8.
  output = Pad(small, [.,.,30,30]) with zeros -> off-grid stays all-0.  uint8 output.

  The only non-free working planes are the fp16 3x3 slice (180B), the fp16 reduced
  count [1,10,3,1] (60B), and the uint8 small block [1,10,3,3] (90B). No 360B fp32
  10-ch plane and no Mul/Sum 10-ch planes (the v1 net had three of those => 1185B).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I64 = TensorProto.INT64


def build(task):
    inits, nodes, vis = [], [], []

    def init(name, arr, dtype=None):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr), name))
        return name

    def vi(name, dtype, shape):
        vis.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    def n(op, ins, out, dtype=None, shape=None, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        if dtype is not None:
            vi(out, dtype, shape)
        return out

    # ---- slice the active 3x3 block (channels all, rows 0:3, cols 0:3) -------
    init("s0", np.array([0, 0, 0], np.int64), np.int64)
    init("e0", np.array([10, 3, 3], np.int64), np.int64)
    init("a0", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "s0", "e0", "a0"], "blk", F32, [1, 10, 3, 3])
    n("Cast", ["blk"], "blk16", F16, [1, 10, 3, 3], to=F16)

    # ---- per-row per-channel count over the column axis ---------------------
    n("ReduceSum", ["blk16"], "cnt", F16, [1, 10, 3, 1], keepdims=1, axes=[3])

    # ---- uniform[r] = (max_k count == 3) ------------------------------------
    n("ReduceMax", ["cnt"], "cmax", F16, [1, 1, 3, 1], keepdims=1, axes=[1])
    init("three", np.array(3.0, np.float16), np.float16)
    n("Equal", ["cmax", "three"], "u", BOOL, [1, 1, 3, 1])

    # ---- per-row channel one-hot: gray(5) if uniform else black(0) ----------
    gray = np.zeros((1, 10, 1, 1), np.uint8); gray[0, 5, 0, 0] = 1
    blk0 = np.zeros((1, 10, 1, 1), np.uint8); blk0[0, 0, 0, 0] = 1
    init("gray_oh", gray, U8)
    init("blk_oh", blk0, U8)
    n("Where", ["u", "gray_oh", "blk_oh"], "chan", U8, [1, 10, 3, 1])

    # ---- broadcast across the 3 columns (in-grid) via Concat (Mul rejects uint8) --
    nodes.append(helper.make_node("Concat", ["chan", "chan", "chan"], ["small_i"], axis=3))
    vi("small_i", U8, [1, 10, 3, 3])

    # ---- pad off-grid to 30x30 with zeros -----------------------------------
    init("pads", np.array([0, 0, 0, 0, 0, 0, 27, 27], np.int64), np.int64)
    n("Pad", ["small_i", "pads"], "output", mode="constant")

    out_vi = helper.make_tensor_value_info("output", U8, [1, 10, 30, 30])
    inp_vi = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task052", [inp_vi], [out_vi], inits, value_info=vis)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
