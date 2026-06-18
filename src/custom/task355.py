"""task355 (ARC-AGI de1cd16c) — output the colour of the block with the most specks.

Rule (from the generator; generalization is scored against generate(), always 2x2):
  The grid is a 2x2 tiling of solid-colour rectangular blocks (each side 5..10,
  filling a <=20x20 canvas anchored top-left). The 4 block colours are DISTINCT.
  A unique `pcolor` (not any block colour) is scattered as `counts[idx]` single
  specks over block idx; counts is a DISTINCT sample of range(6), so there is a
  unique maximum. The 1x1 output = `mostest`, the colour of the block that
  received the MOST specks.

Pipeline (ONNX, opset 11) — closed-form bbox-band MatMul net:
  cnt = ReduceSum(input,[2,3]) per-channel counts; pcolor = present channel with
  the SMALLEST count (specks rarest among present colours). speck = Gather(input,
  pcolor). A SOLID block fills every row & col in its range so the raw 1-D
  per-channel occupancy ReduceMax(input,axis) IS the bbox band. boxcnt[k] =
  rowband_k @ speck @ colband_k (two batched MatMuls, broadcast 10-vs-1). Zero
  pcolor & absent channels, ArgMax -> mostest. Output one-hot at cell (0,0) only,
  Pad-routed into the FREE bool output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
I32 = TensorProto.INT32
I64 = TensorProto.INT64
B = TensorProto.BOOL


def build(task):
    inits, nodes = [], []

    NP = {F32: np.float32, U8: np.uint8, I32: np.int32, I64: np.int64,
          B: np.bool_, TensorProto.FLOAT16: np.float16}

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=NP[dtype]), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    BIG = 1e6

    # ---- per-channel pixel counts -> pcolor channel ---------------------------
    n("ReduceSum", ["input"], "cnt", axes=[2, 3], keepdims=1)  # [1,10,1,1]
    init("f32_zero", np.array([0.0]), F32)
    init("f32_big", np.array([BIG]), F32)
    n("Greater", ["cnt", "f32_zero"], "present")             # bool [1,10,1,1]
    n("Where", ["present", "cnt", "f32_big"], "cnt_masked")  # f32
    n("ArgMin", ["cnt_masked"], "pcolor_idx4", axis=1, keepdims=0)  # [1,1,1] i64
    # keep a [1]-shaped index so Gather(axis=1) yields [1,1,30,30] directly
    n("Squeeze", ["pcolor_idx4"], "pcolor_i1", axes=[0, 1])   # [1] i64
    n("Gather", ["input", "pcolor_i1"], "speck4", axis=1)    # [1,1,30,30] f32

    # ---- bbox bands straight from occupancy ----------------------------------
    n("ReduceMax", ["input"], "rowocc", axes=[3], keepdims=1)  # [1,10,30,1] f32
    n("ReduceMax", ["input"], "colocc", axes=[2], keepdims=1)  # [1,10,1,30] f32
    n("Transpose", ["rowocc"], "rowoccT", perm=[0, 1, 3, 2])   # [1,10,1,30] f32
    n("MatMul", ["rowoccT", "speck4"], "proj")                 # [1,10,1,30] f32
    n("Mul", ["proj", "colocc"], "projc")                      # [1,10,1,30] f32
    n("ReduceSum", ["projc"], "boxcnt", axes=[3], keepdims=1)  # [1,10,1,1] f32

    # ---- mask pcolor channel, ArgMax -> mostest ------------------------------
    init("chan_ids", np.arange(10).reshape(1, 10, 1, 1), I64)
    init("shp111", np.array([1, 1, 1, 1]), I64)
    n("Reshape", ["pcolor_idx4", "shp111"], "pcolor_b")        # [1,1,1,1] i64
    n("Equal", ["chan_ids", "pcolor_b"], "is_pcolor")          # bool [1,10,1,1]
    n("Where", ["is_pcolor", "f32_zero", "boxcnt"], "boxcnt2")
    n("ArgMax", ["boxcnt2"], "mostest4", axis=1, keepdims=0)    # [1,1,1] i64
    n("Reshape", ["mostest4", "shp111"], "mostest_b")
    n("Equal", ["chan_ids", "mostest_b"], "mostest_oh")        # bool [1,10,1,1]

    # ---- route one-hot to cell (0,0) only via uint8 Pad ----------------------
    # mostest_oh [1,10,1,1] -> uint8 -> Pad trailing dims to [1,10,30,30].
    n("Cast", ["mostest_oh"], "mostest_u8", to=U8)
    init("pad_amt", np.array([0, 0, 0, 0, 0, 0, 29, 29]), I64)
    init("zero_u8", np.array(0, dtype=np.uint8).reshape(1), U8)
    n("Pad", ["mostest_u8", "pad_amt", "zero_u8"], "output")  # uint8 [1,10,30,30]

    graph = helper.make_graph(nodes, "task355", [
        helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", U8, [1, 10, 30, 30])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
