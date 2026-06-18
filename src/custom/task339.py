"""Task 339 (ARC-AGI d631b094) — count colored pixels -> N-tall color column.

Rule (from the generator): input is a 3x3 grid holding `count` (1..9) pixels of
a single random color C at random positions.  The OUTPUT is a grid of shape
(count rows) x 1 col, every cell colored C.  In the [1,10,30,30] one-hot
encoding that means: channel C, column 0, rows 0..count-1 are 1; all else 0.

So the entire output is determined by TWO scalars:
    N = count = total number of non-background pixels
    C = the (unique) color  = ArgMax over per-channel pixel counts (chans 1..9)

This is the COUNT->FIXED-PATTERN tier (task399 idiom):
  * per-channel counts cnts = ReduceSum(input, [2,3])  -> [1,10,1,1] (40B)
  * N  = ReduceSum of the colored channels (slice chans 1..9)        -> scalar
  * C  = ArgMax(cnts over chans 1..9) + 1                            -> scalar
Build a TINY [1,10,9,1] uint8 one-hot:
    rowmask[r] = (r < N)              -> [1,1,9,1]
    colorhot   = (arange == C)        -> [1,10,1,1]
    small      = rowmask AND colorhot -> [1,10,9,1]
then Pad straight INTO the free [1,10,30,30] output (21 rows + 29 cols of 0).
No 30x30 carrier plane is ever materialised.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
I32 = TensorProto.INT32
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

    # ---- per-channel pixel counts [1,10,1,1] (fp32, 40B) ----
    n("ReduceSum", ["input"], "cnts", axes=[2, 3], keepdims=1)        # [1,10,1,1]

    # ---- only ONE color is present, so its channel is the only colored one
    #      with count>0.  keepmask zeroes background channel 0; multiply it onto
    #      cnts so both the color one-hot AND the count N come from this product:
    #          masked[k] = cnts[k] for k>=1, 0 for k==0
    keep = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1], np.float32).reshape(1, 10, 1, 1)
    init("keep", keep, np.float32)
    n("Mul", ["cnts", "keep"], "masked")                           # [1,10,1,1] fp32

    # color one-hot = (masked > 0)  -> true only at the present color channel
    init("zerof", np.array([[[[0.0]]]], np.float32), np.float32)
    n("Greater", ["masked", "zerof"], "colorhot")                  # [1,10,1,1] bool

    # N = total colored pixels = sum over channels of masked
    n("ReduceSum", ["masked"], "nf", axes=[1], keepdims=1)         # [1,1,1,1] fp32

    # ---- colmask [1,1,1,9] = (col_ramp < N)  (output is 1 row x N cols) ----
    ramp9 = np.arange(9, dtype=np.float32).reshape(1, 1, 1, 9)
    init("ramp9", ramp9, np.float32)
    n("Less", ["ramp9", "nf"], "colmask")                          # [1,1,1,9] bool

    # ---- small one-hot [1,10,1,9] = colorhot AND colmask ----
    n("And", ["colorhot", "colmask"], "small")                     # [1,10,1,9] bool
    n("Cast", ["small"], "small_u8", to=U8)                        # [1,10,1,9] uint8

    # ---- Pad [1,10,1,9] -> [1,10,30,30] with 0; this Pad IS the output ----
    pads = np.array([0, 0, 0, 0, 0, 0, 29, 21], np.int64)
    init("pads", pads, np.int64)
    init("zerov", np.array(0, np.uint8), np.uint8)
    n("Pad", ["small_u8", "pads", "zerov"], "output", mode="constant")  # [1,10,30,30]

    out_vi = helper.make_tensor_value_info("output", U8, [1, 10, 30, 30])
    in_vi = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task339", [in_vi], [out_vi], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
