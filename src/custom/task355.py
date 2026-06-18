"""task355 (ARC-AGI de1cd16c) — output the colour of the block with the most specks.

Rule (from the generator; generalization is scored against generate()):
  The grid is a tiling (2x2 by default; up to 2x3 / 3x2 in the displayed test) of
  solid-colour rectangular blocks (each side 5..10) filling a <=20x20 canvas
  anchored top-left. The block colours are DISTINCT. A unique `pcolor` (not any
  block colour) is scattered as `counts[idx]` single specks over block idx;
  `counts` is a DISTINCT sample of range(6), so the maximum is unique. The 1x1
  output = `mostest`, the colour of the block with the MOST specks.

KEY ESCAPE (no speck plane, no 30x30 working plane):
  A speck OVERWRITES the block colour, so inside block k's rectangle every cell
  is either block-colour-k or pcolor. Therefore
        area_k  ==  cnt_k  +  specks_k          (exact partition)
  where cnt_k = #pixels still showing colour k = ReduceSum(input,[2,3]) and
  area_k = (#rows block k occupies) x (#cols block k occupies). So
        specks_k = area_k - cnt_k
  and  mostest = argmax_k specks_k  (over present block channels, pcolor masked).
  Everything is a [1,10,1,1] scalar reduction; the only mid-size intermediates
  are the two 1-D occupancy bands (ReduceMax over one axis), ~1200B each. The
  speck plane and every [1,10,30,*] projection are eliminated.

Pipeline (ONNX, opset 11):
  cnt   = ReduceSum(input,[2,3])                        # [1,10,1,1] pixel counts
  pcolor channel = present channel with SMALLEST cnt (specks rarer than blocks)
  rowocc/colocc = ReduceMax(input,[3]) / (input,[2])    # binary occupancy bands
  rowcount = ReduceSum(rowocc,[2]); colcount = ReduceSum(colocc,[3])
  area = rowcount*colcount;  specks = area - cnt
  mask pcolor & absent channels -> ArgMax -> mostest one-hot at (0,0) -> Pad ->
  routed into the FREE uint8 output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
I64 = TensorProto.INT64
B = TensorProto.BOOL


def build(task):
    inits, nodes = [], []

    NP = {F32: np.float32, U8: np.uint8, I64: np.int64, B: np.bool_}

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=NP[dtype]), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    BIG = 1e6
    NEG = -1e6

    # ---- per-channel pixel counts -> pcolor channel ---------------------------
    n("ReduceSum", ["input"], "cnt", axes=[2, 3], keepdims=1)  # [1,10,1,1] f32
    init("f32_zero", np.array([0.0]), F32)
    init("f32_big", np.array([BIG]), F32)
    init("f32_neg", np.array([NEG]), F32)
    n("Greater", ["cnt", "f32_zero"], "present")               # bool [1,10,1,1]
    # pcolor = present channel with smallest count
    n("Where", ["present", "cnt", "f32_big"], "cnt_for_min")   # absent -> +BIG
    n("ArgMin", ["cnt_for_min"], "pcolor_idx", axis=1, keepdims=0)  # [1,1,1] i64

    # ---- occupancy bands -> per-channel row/col occupied counts ---------------
    n("ReduceMax", ["input"], "rowocc", axes=[3], keepdims=1)  # [1,10,30,1] f32
    n("ReduceMax", ["input"], "colocc", axes=[2], keepdims=1)  # [1,10,1,30] f32
    n("ReduceSum", ["rowocc"], "rowcount", axes=[2], keepdims=1)  # [1,10,1,1]
    n("ReduceSum", ["colocc"], "colcount", axes=[3], keepdims=1)  # [1,10,1,1]
    n("Mul", ["rowcount", "colcount"], "area")                 # [1,10,1,1] f32
    n("Sub", ["area", "cnt"], "specks")                        # [1,10,1,1] f32

    # ---- mask pcolor & absent channels, ArgMax -> mostest --------------------
    init("chan_ids", np.arange(10).reshape(1, 10, 1, 1), I64)
    init("shp111", np.array([1, 1, 1, 1]), I64)
    n("Reshape", ["pcolor_idx", "shp111"], "pcolor_b")         # [1,1,1,1] i64
    n("Equal", ["chan_ids", "pcolor_b"], "is_pcolor")          # bool [1,10,1,1]
    # invalid (pcolor OR absent) -> -BIG so ArgMax never picks them
    n("Not", ["present"], "absent")                            # bool
    n("Or", ["is_pcolor", "absent"], "invalid")               # bool
    n("Where", ["invalid", "f32_neg", "specks"], "specks2")    # f32
    n("ArgMax", ["specks2"], "mostest", axis=1, keepdims=0)    # [1,1,1] i64
    n("Reshape", ["mostest", "shp111"], "mostest_b")
    n("Equal", ["chan_ids", "mostest_b"], "mostest_oh")        # bool [1,10,1,1]

    # ---- route one-hot to cell (0,0) only via uint8 Pad ----------------------
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
