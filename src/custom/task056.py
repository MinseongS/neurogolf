"""Task 056 (ARC-AGI 27a28665) — classify one of 4 fixed 3x3 shapes -> 1x1 idx.

Rule (from the generator): the input is a 3x3 grid containing a single shape
drawn in some colour at one of 4 fixed pixel patterns indexed idx in {1,2,3,6}:
    idx=1: (0,0)(0,1)(1,0)(1,2)(2,1)
    idx=2: (0,0)(0,2)(1,1)(2,0)(2,2)
    idx=3: (0,1)(0,2)(1,1)(1,2)(2,0)
    idx=6: (0,1)(1,0)(1,1)(1,2)(2,1)
The OUTPUT is a 1x1 grid whose single cell value == idx (colour-agnostic).

Encoding (positional fingerprint, NO ArgMax/Gather/templates):
 * Slice channel-0 of the top-left 3x3 -> bg occupancy [1,1,3,3]; binary shape
   occupancy = 1 - bg (fp16, tiny plane).
 * One no-pad Conv with a SINGLE [1,1,3,3] positional weight collapses the 3x3
   occupancy to ONE scalar fingerprint that is DISTINCT per pattern:
       weight = [[0,0,1],[2,2,2],[2,2,0]]  ->  idx1=6 idx2=5 idx3=7 idx6=8.
 * A "bank" const [1,10,1,1] holds each idx's fingerprint at its own channel
   (ch1=6 ch2=5 ch3=7 ch6=8) and a never-matching -1 elsewhere, so
   Equal(score, bank) is the 10-channel one-hot DIRECTLY (no ArgMax/Gather/
   output_bank). Cast uint8; Pad -> [1,10,30,30].

Params: 9 (weight) + 10 (bank) + tiny scalars.  All intermediates are tiny
([1,1,3,3] / [1,10,1,1]); the 10-channel expansion and 30x30 canvas land in
the FREE output via Pad.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
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

    # ---- bg occupancy of the top-left 3x3 (channel 0) -> [1,1,3,3] fp32 ----
    # Slice channel 0 AND spatial rows 0:3, cols 0:3 in one op -> [1,1,3,3].
    # We never form the binary 1-bg plane: with occ = 1-bg and weight sum Sw=11,
    # the positional fingerprint Conv(occ,w) = Sw - Conv(bg,w), so we run the
    # Conv DIRECTLY on bg (fp32) and fold the Sw offset into the bank constants.
    init("st", np.array([0, 0, 0], np.int64), np.int64)   # starts on axes 1,2,3
    init("en", np.array([1, 3, 3], np.int64), np.int64)   # ends   on axes 1,2,3
    init("ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "st", "en", "ax"], "bg")         # [1,1,3,3] fp32

    # ---- positional fingerprint Conv on bg -> scalar score [1,1,1,1] fp32 ----
    Wpos = np.array([[0, 0, 1], [2, 2, 2], [2, 2, 0]], np.float32).reshape(1, 1, 3, 3)
    init("W", Wpos, np.float32)
    n("Conv", ["bg", "W"], "score")                       # [1,1,1,1] fp32 = Conv(bg,w)

    # ---- bank holds Sw-fingerprint per idx channel; Equal -> one-hot directly --
    # occ-fingerprints: idx1=6 idx2=5 idx3=7 idx6=8 ; Sw=11 ; bg-score = 11-fp.
    Sw = 11.0
    bank = np.full((1, 10, 1, 1), -1.0, np.float32)
    occ_fp = {1: 6.0, 2: 5.0, 3: 7.0, 6: 8.0}             # idx -> occ fingerprint
    for k, v in occ_fp.items():
        bank[0, k, 0, 0] = Sw - v                         # = bg-score for idx
    init("bank", bank, np.float32)
    n("Equal", ["score", "bank"], "onehot")               # [1,10,1,1] bool (broadcast)
    n("Cast", ["onehot"], "onehot8", to=U8)               # [1,10,1,1] uint8
    pads = np.array([0, 0, 0, 0, 0, 0, 29, 29], np.int64)
    init("pads", pads, np.int64)
    init("zerov", np.array(0, np.uint8), np.uint8)
    n("Pad", ["onehot8", "pads", "zerov"], "output", mode="constant")

    out_vi = helper.make_tensor_value_info("output", U8, [1, 10, 30, 30])
    in_vi = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task056", [in_vi], [out_vi], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
