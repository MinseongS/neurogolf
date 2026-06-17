"""task043 (ARC-AGI 2281f1f4) — "gray markers on top row + right column ->
red at every intersection".

Rule (from generator task_2281f1f4.py, grid is ALWAYS 10x10):
  Top row (row 0) carries gray(5) markers at a set of columns `cols` (cols in 0..8).
  Right column (col 9) carries gray(5) markers at a set of rows `rows` (rows in 1..9).
  Output = input (gray markers kept unchanged) PLUS red(2) at every intersection
  (r in rows, c in cols), i.e. output[r][c]=red for each r with a right-col gray and
  each c with a top-row gray. rows>=1 and cols<=8 so red never lands on a gray marker.

  => SEPARABLE row x col, routed into the FREE output (Tier A, no per-cell label plane):
     colmask[c] = (input row0 col c is gray)     -> [1,1,1,10]
     rowmask[r] = (input col9 row r is gray)      -> [1,1,10,1]
     red[r,c]   = rowmask[r] AND colmask[c]       -> [1,1,10,10] (broadcast)
     output     = Where(red, red_onehot, input)   (FREE [1,10,30,30])

Encoding: slice the gray channel's row0 (col presence) and col9 (row presence)
directly from the FREE input (no ReduceMax needed — these are single lines), AND
them with broadcast, Pad to 30x30 bool, Where with the fixed red one-hot. Dominant
intermediate is the [1,1,30,30] padded uint8 red mask (~900B).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I64 = TensorProto.INT64

W = 10  # active canvas (grid is always 10x10 for this task)
GRAY = 5
RED = 2


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- colmask: gray channel, row 0, cols 0:10  -> [1,1,1,10] -----------
    init("col_s", np.array([GRAY, 0, 0], np.int64), np.int64)
    init("col_e", np.array([GRAY + 1, 1, W], np.int64), np.int64)
    init("hw_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "col_s", "col_e", "hw_ax"], "col_f32")  # [1,1,1,W] f32
    n("Cast", ["col_f32"], "colmask", to=BOOL)                   # [1,1,1,W] bool

    # ---- rowmask: gray channel, col 9, rows 0:10  -> [1,1,10,1] -----------
    init("row_s", np.array([GRAY, 0, W - 1], np.int64), np.int64)
    init("row_e", np.array([GRAY + 1, W, W], np.int64), np.int64)
    n("Slice", ["input", "row_s", "row_e", "hw_ax"], "row_f32")  # [1,1,W,1] f32
    n("Cast", ["row_f32"], "rowmask", to=BOOL)                   # [1,1,W,1] bool

    # ---- red mask = rowmask AND colmask (broadcast)  -> [1,1,10,10] -------
    n("And", ["rowmask", "colmask"], "red_b")  # [1,1,W,W] bool

    # ---- pad to 30x30 (uint8) then -> bool cond ---------------------------
    n("Cast", ["red_b"], "red_u8", to=U8)  # [1,1,W,W] uint8 {0,1}
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64), np.int64)
    init("ZEROU8", np.array(0, np.uint8), np.uint8)
    n("Pad", ["red_u8", "pads", "ZEROU8"], "red30", mode="constant")  # [1,1,30,30] u8
    n("Cast", ["red30"], "cond", to=BOOL)  # [1,1,30,30] bool

    # ---- red one-hot (fixed color 2) --------------------------------------
    oh = np.zeros((1, 10, 1, 1), np.float32)
    oh[0, RED, 0, 0] = 1.0
    init("red_oh", oh, np.float32)  # [1,10,1,1] f32

    # ---- output = Where(cond, red_oh, input) : FREE [1,10,30,30] ----------
    n("Where", ["cond", "red_oh", "input"], "output")

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task043", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
