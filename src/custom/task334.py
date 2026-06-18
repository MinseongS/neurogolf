"""Task 334 (ARC-AGI d4469b4b) — color -> fixed 3x3 gray/black pattern.

Rule (from the generator): the input is a size x size grid holding random
pixels of ONE colour c in {1,2,3}.  The pixel POSITIONS are irrelevant; the
OUTPUT is ALWAYS a 3x3 grid whose gray(5)/black(0) pixels are a fixed pattern
keyed ONLY on the colour c:

    c=1: [[0,5,0],[5,5,5],[0,5,0]]
    c=2: [[5,5,5],[0,5,0],[0,5,0]]
    c=3: [[0,0,5],[0,0,5],[5,5,5]]

So the whole output is determined by ONE scalar (the colour c).  This is a pure
COUNT->FIXED-PATTERN task (cheapest tier).

Recover c: per-channel pixel counts cnts = ReduceSum(input,[2,3]) [1,10,1,1].
Slice the channels for colours {1,2,3} (channels 1:4) and ArgMax over the
channel axis -> a scalar index ci in {0,1,2} (colour = ci+1).  (Only one of
those three channels is ever non-empty, so ArgMax is exact and POSITION-robust.)

Build the gray indicator: onehot_c[1,3] = Equal(ci, arange3); the gray pattern
flattened to [1,9] is onehot_c @ table[3,9] (a tiny 27-elem const table whose
row ci is the flattened gray mask for colour ci+1).  Reshape to [1,1,3,3].

Route to the free output: gray must live in colour channel 5, black in channel
0.  Build ch0 = Where(gray, 0, 1) and ch5 = Where(gray, 1, 0) as [1,1,3,3]
uint8, place ch5 at channel position 5 by Concat'ing ch0 with four zero
channels then ch5 -> [1,6,3,3] uint8, and Pad (trailing 4 channels + the 30x30
spatial border) straight into the FREE output.  No 30x30 carrier plane is ever
materialised; declare the output uint8 (harness scores out>0).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
I64 = TensorProto.INT64
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

    # ---- recover colour index ci in {0,1,2} (colour = ci+1) ----
    n("ReduceSum", ["input"], "cnts", axes=[2, 3], keepdims=1)   # [1,10,1,1] f32
    init("s_a", np.array([1], np.int64), np.int64)
    init("s_b", np.array([4], np.int64), np.int64)
    init("s_ax", np.array([1], np.int64), np.int64)
    n("Slice", ["cnts", "s_a", "s_b", "s_ax"], "cnt3")           # [1,3,1,1]
    init("ci_sq", np.array([1], np.int64), np.int64)
    n("ArgMax", ["cnt3"], "ci", axis=1, keepdims=0)              # [1,1,1] int64
    n("Reshape", ["ci", "ci_sq"], "ci1")                         # [1] int64

    # Gather the 3x3 gray-onehot table by colour index ci.  table[ci] is the
    # 3x3 gray mask (1=gray, 0=black) for colour ci+1, so the gathered plane
    # IS the gray indicator directly (no MatMul / no extra full planes).
    table = np.array([
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],   # c=1
        [[1, 1, 1], [0, 1, 0], [0, 1, 0]],   # c=2
        [[0, 0, 1], [0, 0, 1], [1, 1, 1]],   # c=3
    ], np.uint8)                                                 # [3,3,3]
    init("table", table, np.uint8)
    n("Gather", ["table", "ci1"], "g3", axis=0)                  # [1,3,3] u8
    init("zerov0", np.array(0, np.uint8), np.uint8)
    n("Greater", ["g3", "zerov0"], "gray2")                      # [1,3,3] bool
    init("rsg", np.array([1, 1, 3, 3], np.int64), np.int64)
    n("Reshape", ["gray2", "rsg"], "gray")                       # [1,1,3,3] bool

    # build the colour channels: ch0=black-onehot, ch5=gray-onehot
    init("one3", np.ones((1, 1, 3, 3), np.uint8), np.uint8)
    init("zero3", np.zeros((1, 1, 3, 3), np.uint8), np.uint8)
    n("Where", ["gray", "zero3", "one3"], "ch0")                 # [1,1,3,3] u8
    n("Where", ["gray", "one3", "zero3"], "ch5")                 # [1,1,3,3] u8
    # channels: 0=ch0, 1..4=zero (reuse the single zero3 const four times), 5=ch5
    n("Concat", ["ch0", "zero3", "zero3", "zero3", "zero3", "ch5"],
      "oneh6", axis=1)                                           # [1,6,3,3] u8

    # Pad [1,6,3,3] -> [1,10,30,30] with 0 (this Pad IS the output)
    pads = np.array([0, 0, 0, 0, 0, 4, 27, 27], np.int64)
    init("pads", pads, np.int64)
    init("zerov", np.array(0, np.uint8), np.uint8)
    n("Pad", ["oneh6", "pads", "zerov"], "output", mode="constant")

    out_vi = helper.make_tensor_value_info("output", U8, [1, 10, 30, 30])
    in_vi = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task334", [in_vi], [out_vi], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
