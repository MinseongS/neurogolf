"""task079 (ARC-AGI 39a8645d) — output the 3x3 shape of the most-COPIED sprite.

Rule (from the generator, verified on fresh instances):
  The 14x14 grid contains 2-3 sprite TYPES, each a distinct 3x3 monochrome shape
  in its own colour.  Each type is placed `num` non-overlapping times where the
  per-type copy counts are DISTINCT (sampled from {1,2,3} without replacement)
  and sorted descending, so type 0 has the strictly MOST copies.  The 3x3 output
  is the shape of sprite type 0, drawn in its colour.

  copies(colour) = total_pixels(colour) / sprite_size(colour), and the dominant
  colour = argmax copies (verified 0 fails / 800 fresh).  NB: "most pixels" is
  WRONG (a 6-px shape x3 copies < an 8-px shape... etc); must divide by size.

  sprite_size(colour) = MAX over 3x3 windows of that colour's pixel count
  (copies are non-overlapping & >=1 cell apart, so one window == one aligned
  copy).  The 3x3 output = the best-scoring aligned window of the dominant
  colour (verified 0 fails / 500 fresh).

Pipeline (ONNX, opset 11):
  1. cnt = ReduceSum(input,[2,3]) [1,10,1,1] = per-colour total pixels.
  2. blk = depthwise 3x3 sum-conv over the 14x14 active region [1,10,12,12];
     blkmax = ReduceMax over spatial = per-colour sprite_size.
  3. copies = cnt / blkmax ; zero ch0 ; k = ArgMax -> dominant colour.
  4. Mk = the dominant-colour block-count plane (Gather blk by k); argmax over
     its flattened 12x12 grid -> a top-left of one aligned copy.
  5. crop the 3x3 of the dominant-colour mask at that top-left; one-hot the 3x3
     (bg->ch0, pixel->ch_k); Pad uint8 to [1,10,30,30] at the top-left.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
I64 = TensorProto.INT64
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

    # ---- slice input to the 14x14 active region, COLOUR channels 1..9 ------
    # (drop background ch0: it can't be the dominant sprite colour, and skipping
    #  it shaves a channel off the two big planes).
    init("s_start", np.array([1, 0, 0], np.int64), np.int64)
    init("s_end", np.array([10, 14, 14], np.int64), np.int64)
    init("s_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "s_start", "s_end", "s_ax"], "act")     # [1,9,14,14] f32

    # ---- 1. per-colour total pixel count -----------------------------------
    n("ReduceSum", ["act"], "cnt", axes=[2, 3], keepdims=1)      # [1,9,1,1] f32

    # ---- 2. per-colour sprite size = max 3x3 block count -------------------
    convw = np.ones((9, 1, 3, 3), np.float32)                    # depthwise
    init("convw", convw, np.float32)
    n("Conv", ["act", "convw"], "blk", group=9)                  # [1,9,12,12] f32
    n("ReduceMax", ["blk"], "blkmax", axes=[2, 3], keepdims=1)   # [1,9,1,1] f32

    # ---- 3. copies = cnt / blkmax ; dominant colour index (0..8 -> +1) -----
    init("eps", np.array(1e-3, np.float32), np.float32)
    n("Add", ["blkmax", "eps"], "blkden")                        # [1,9,1,1]
    n("Div", ["cnt", "blkden"], "copies")                        # [1,9,1,1] f32
    n("ArgMax", ["copies"], "k0", axis=1, keepdims=0)            # [1,1,1] int64
    init("scalar1", np.array([1], np.int64), np.int64)
    n("Reshape", ["k0", "scalar1"], "k0s")                       # [1] int64 (0..8)
    # actual colour channel = k0 + 1
    init("one_i64", np.array([1], np.int64), np.int64)
    n("Add", ["k0s", "one_i64"], "kscal")                        # [1] int64 (1..9)

    # ---- 4. dominant-colour block plane -> best aligned top-left -----------
    n("Gather", ["blk", "k0s"], "Mblk", axis=1)                  # [1,1,12,12] f32
    init("flat", np.array([1, 144], np.int64), np.int64)
    n("Reshape", ["Mblk", "flat"], "Mflat")                      # [1,144] f16
    n("ArgMax", ["Mflat"], "bestflat", axis=1, keepdims=0)       # [1] int64
    init("twelve", np.array([12], np.int64), np.int64)
    n("Div", ["bestflat", "twelve"], "best_r")                   # [1]
    n("Mod", ["bestflat", "twelve"], "best_c")                   # [1]

    # ---- dominant-colour mask plane (14x14) for the crop -------------------
    n("Gather", ["act", "k0s"], "Mk", axis=1)                    # [1,1,14,14] f32

    # ---- 5. crop the 3x3 block at (best_r,best_c) --------------------------
    init("r012", np.array([0, 1, 2], np.int64), np.int64)
    n("Add", ["best_r", "r012"], "ridx")                         # [3]
    n("Add", ["best_c", "r012"], "cidx")                         # [3]
    n("Gather", ["Mk", "ridx"], "rowcrop", axis=2)               # [1,1,3,14] f16
    n("Gather", ["rowcrop", "cidx"], "shape3", axis=3)           # [1,1,3,3] f16

    # ---- label & one-hot the 3x3 ------------------------------------------
    n("Cast", ["kscal"], "kfl", to=F32)                          # [1] f32
    init("rk", np.array([1, 1, 1, 1], np.int64), np.int64)
    n("Reshape", ["kfl", "rk"], "kfl4")                          # [1,1,1,1] f32
    n("Mul", ["shape3", "kfl4"], "L3")                           # [1,1,3,3] f32
    arange = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("arange", arange, np.float32)
    n("Equal", ["L3", "arange"], "oh3")                          # [1,10,3,3] bool
    n("Cast", ["oh3"], "oh3u", to=U8)                            # [1,10,3,3] u8
    init("pads", np.array([0, 0, 0, 0, 0, 0, 27, 27], np.int64), np.int64)
    n("Pad", ["oh3u", "pads"], "output", mode="constant")        # [1,10,30,30] u8

    graph = helper.make_graph(
        nodes, "task079", [
            helper.make_tensor_value_info("input", F32, [1, 10, 30, 30]),
        ], [
            helper.make_tensor_value_info("output", U8, [1, 10, 30, 30]),
        ], inits)
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
