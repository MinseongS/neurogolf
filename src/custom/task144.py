"""Task 144 (ARC-AGI 6430c8c4) — per-cell NOR of two stacked 4x4 sub-grids.

Rule (from the generator): the input is a 9x4 grid = two 4x4 sub-grids stacked
vertically and separated by a yellow (4) divider row at row `size`=4.
  * top sub-grid    (rows 0..3): orange (7) pixels
  * bottom sub-grid (rows 5..8): red (2) pixels
The 4x4 OUTPUT colours cell (r,c) GREEN (3) iff BOTH the top cell (r,c) and the
bottom cell (r+5,c) are EMPTY (background); otherwise the cell stays BLACK (0).
Everything outside the 4x4 output grid is unset (all-zero one-hot).

So output is a pure per-cell NOR of the two halves' occupancy:
  green = top_empty AND bottom_empty,   black = NOT green.
A cell is empty iff input channel-0 (background one-hot) is 1 there.

Implementation (all working tensors stay 4x4 — the 10-channel + spatial
expansion lands in the FREE output via the final Pad):
  1. Slice channel-0 over the top 4x4 (rows 0..3) and the bottom 4x4 (rows 5..8).
  2. Cast each to BOOL; greenb = And(top_empty, bot_empty).
  3. inner = Where(greenb[1,1,4,4], green_onehot[1,4,1,1], black_onehot[1,4,1,1])
     -> [1,4,4,4] uint8.  The Where BROADCASTS the per-cell green/black mask
     against the tiny 4-channel one-hots in ONE op, so it simultaneously routes
     each cell to its colour channel AND emits all four channels — no separate
     black plane, no Not, no uint8 casts, no zero-channel init.
  4. Pad to [1,10,30,30] (uint8) — the Pad output IS the graph output (free);
     trailing colour channels and the spatial border zero-fill automatically.
Output value_info is UINT8 — the harness scores (out>0), identical to fp16/fp32.

mem 240B (two 4x4 fp32 slices 128B [Slice preserves the fp32 input dtype], three
4x4 bool planes 48B, the [1,4,4,4] uint8 Where result 64B), params 32 -> 19.39
(beats the public Concat net 18.94 by +0.45).
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
    inits, nodes, vis = [], [], []

    def I(name, arr, dt):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dt), name))

    def vi(name, dt, shape):
        vis.append(helper.make_tensor_value_info(name, dt, shape))

    # ---- 1. slice channel-0 (background) over the two 4x4 sub-grids ----
    # top sub-grid = rows 0..3 ; bottom sub-grid = rows 5..8 ; cols 0..3.
    # No `axes` input: starts/ends default to the first 4 axes.
    I("ts", np.array([0, 0, 0, 0], np.int64), np.int64)
    I("te", np.array([1, 1, 4, 4], np.int64), np.int64)
    I("bs", np.array([0, 0, 5, 0], np.int64), np.int64)
    I("be", np.array([1, 1, 9, 4], np.int64), np.int64)
    nodes.append(helper.make_node("Slice", ["input", "ts", "te"], ["topf"]))
    vi("topf", F32, [1, 1, 4, 4])
    nodes.append(helper.make_node("Slice", ["input", "bs", "be"], ["botf"]))
    vi("botf", F32, [1, 1, 4, 4])

    # ---- 2. NOR: green = top_empty AND bot_empty ----
    nodes.append(helper.make_node("Cast", ["topf"], ["topb"], to=B))
    vi("topb", B, [1, 1, 4, 4])
    nodes.append(helper.make_node("Cast", ["botf"], ["botb"], to=B))
    vi("botb", B, [1, 1, 4, 4])
    nodes.append(helper.make_node("And", ["topb", "botb"], ["greenb"]))
    vi("greenb", B, [1, 1, 4, 4])

    # ---- 3. Where broadcasts the mask against 4-channel one-hots ----
    # green cell -> channel 3 (green=3) ; otherwise channel 0 (black=0).
    I("gblk", np.array([0, 0, 0, 1], np.uint8).reshape(1, 4, 1, 1), np.uint8)
    I("kblk", np.array([1, 0, 0, 0], np.uint8).reshape(1, 4, 1, 1), np.uint8)
    nodes.append(helper.make_node("Where", ["greenb", "gblk", "kblk"], ["inner"]))
    vi("inner", U8, [1, 4, 4, 4])

    # ---- 4. Pad to the full 10x30x30 — Pad output IS the free graph output ----
    I("pads", np.array([0, 0, 0, 0, 0, 6, 26, 26], np.int64), np.int64)
    nodes.append(helper.make_node("Pad", ["inner", "pads"], ["output"],
                                  mode="constant"))

    graph = helper.make_graph(
        nodes, "task144",
        [helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", U8, [1, 10, 30, 30])],
        inits, value_info=vis)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = IR_VERSION
    return model
