"""task111 (ARC-AGI 48d8fb45) — output sprite-0's 3x3 block.

Rule (from the generator):
  4 conway sprites are stamped on a size=10 grid in a single colour `color`
  (gray excluded).  A gray (=5) marker is placed at (minirows[0]-1,
  minicols[0]+1).  Sprite 0 occupies the 3x3 block at top-left
  (minirows[0], minicols[0]); its pixels are `output[r][c] = color`.
  Everything else of the 3x3 output is background.

  So: gray pixel at (gr, gc)  =>  block top-left (gr+1, gc-1).
  output = the 3x3 crop of the input at rows [gr+1 .. gr+3),
  cols [gc-1 .. gc+2).  The block holds ONLY background (ch0) and `color`
  pixels (the gray marker sits one row ABOVE the block, never inside it), so
  the crop equals the output one-hot exactly.

Encoding (opset 11):
  1. Slice channel-5 (gray) to its possible bounding box [rows 0..6, cols 1..8]
     -> [1,1,7,8] fp32 (224B).  gr = ArgMax(rowsum) in 0..6 ;
     gcrel = ArgMax(colsum) in 0..7 (= gc-1 since cols start at 1).
  2. block top-left r0 = gr+1 , c0 = gcrel  (= gc-1).
  3. Data-dependent Slice of the full 10-ch input to [1,10,3,3] (the crop),
     value_info shape declared so static memory is measurable.
  4. Pad the 3x3 crop into the free 30x30 output (top-left).

Dominant intermediate: the [1,10,3,3] fp32 crop = 360B (irreducible fp32
10-channel 3x3 region; a uint8 cast would only ADD a plane, and a per-axis
Gather would pay a 3600B [1,10,3,30] intermediate).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
I64 = TensorProto.INT64


def build(task):
    inits, nodes = [], []

    np_of = {F32: np.float32, I64: np.int64}

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=np_of[dtype]), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    vis = []

    def vi(name, shape, dtype=F32):
        vis.append(helper.make_tensor_value_info(name, dtype, shape))

    # ---- 1. gray bounding-box slice -----------------------------------------
    # gray (channel 5) marker:  row in [0..6], col in [1..8].
    init("g_s", np.array([0, 5, 0, 1], np.int64), I64)
    init("g_e", np.array([1, 6, 7, 9], np.int64), I64)
    init("ax0123", np.array([0, 1, 2, 3], np.int64), I64)
    n("Slice", ["input", "g_s", "g_e", "ax0123"], "gray")  # [1,1,7,8] f32
    vi("gray", [1, 1, 7, 8])

    n("ReduceSum", ["gray"], "rp", axes=[3], keepdims=1)   # [1,1,7,1]
    vi("rp", [1, 1, 7, 1])
    n("ReduceSum", ["gray"], "cp", axes=[2], keepdims=1)   # [1,1,1,8]
    vi("cp", [1, 1, 1, 8])
    n("ArgMax", ["rp"], "gr", axis=2, keepdims=1)          # [1,1,1,1] int64 in 0..6
    vi("gr", [1, 1, 1, 1], I64)
    n("ArgMax", ["cp"], "gc1", axis=3, keepdims=1)         # [1,1,1,1] int64 in 0..7
    vi("gc1", [1, 1, 1, 1], I64)

    # ---- 2. block top-left & crop bounds ------------------------------------
    # starts = [r0, c0] = [gr+1, gc1] ; ends = starts + 3.
    init("one1", np.array([1], np.int64), I64)
    init("shp1", np.array([1], np.int64), I64)
    init("three2", np.array([3, 3], np.int64), I64)

    n("Reshape", ["gr", "shp1"], "gr1")                    # [1]
    vi("gr1", [1], I64)
    n("Reshape", ["gc1", "shp1"], "gc1f")                  # [1]  (= gc-1 = c0)
    vi("gc1f", [1], I64)
    n("Add", ["gr1", "one1"], "r0")                        # [1]  r0 = gr+1
    vi("r0", [1], I64)

    n("Concat", ["r0", "gc1f"], "starts", axis=0)          # [2]
    vi("starts", [2], I64)
    n("Add", ["starts", "three2"], "ends")                 # [2]  starts + 3
    vi("ends", [2], I64)
    init("ax23", np.array([2, 3], np.int64), I64)

    # ---- 3. crop the 3x3 block ----------------------------------------------
    n("Slice", ["input", "starts", "ends", "ax23"], "crop")  # [1,10,3,3] f32
    vi("crop", [1, 10, 3, 3])

    # ---- 4. pad into the free 30x30 output ----------------------------------
    init("pads", np.array([0, 0, 0, 0, 0, 0, 27, 27], np.int64), I64)
    n("Pad", ["crop", "pads"], "output")

    graph = helper.make_graph(
        nodes, "task111",
        [helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])],
        inits, value_info=vis,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
