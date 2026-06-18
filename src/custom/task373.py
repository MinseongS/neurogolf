"""task373 (ARC-AGI e9afcf9a) — 2x6 two-row grid, swap rows on odd columns.

Rule (from the ARC-GEN generator, verified fresh):
  The grid is ALWAYS 2 rows x 6 cols.  Input row0 = colorA across all 6 cols,
  row1 = colorB across all 6 cols (each row a solid colour).  Output:
      output[(r+c)%2][c] = colors[r]
  i.e. even columns keep (A top, B bottom); odd columns swap (B top, A bottom).
  Everything outside the 2x6 region is background colour 0 (unchanged).

Construction (pure spatial copy / permutation, Tier-S, uint8 whole-pipeline):
  The two colours are CONSTANT across columns, so a single 2x1 slice of column 0
  carries the entire instance:  cols = input[:, :, 0:2, 0:1]  -> [1,10,2,1] = the
  one-hot stack [A;B].  Its row-reverse cols_rev = [B;A] is the swapped pair.
  The 2x6 block is then one broadcast Where over a fixed column-parity mask:
      block[ :, :, :, c] = cols      if c even   (A top, B bottom)
                         = cols_rev   if c odd    (B top, A bottom)
  Everything is cast to uint8 (one-hot is {0,1}, harness scores out>0), so the
  only fp32 tensor is the tiny 80B entry slice; the 120B uint8 block is Padded
  straight into the FREE 30x30 output.

  mem ~= 80 (fp32 slice) + 20 (uint8 cols) + 20 (uint8 rev) + 120 (uint8 block)
  vs the public GridSample-2x6 net's 480B fp32 sampled plane -> beats 18.76.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

S = 30


def build(task):
    inits, nodes = [], []
    seen = set()

    def init(name, arr, dt):
        if name in seen:
            return name
        seen.add(name)
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dt), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    U8 = TensorProto.UINT8
    B = TensorProto.BOOL

    # ---- grab the two solid colours from column 0, rows 0..1 ----------------
    init("s00", np.array([0, 0], np.int64), np.int64)   # starts (rows, cols)
    init("e21", np.array([2, 1], np.int64), np.int64)   # ends   (rows<2, col<1)
    init("ax23", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["input", "s00", "e21", "ax23"], "cols_f")   # [1,10,2,1] fp32, 80B
    n("Cast", ["cols_f"], "cols", to=U8)                    # [1,10,2,1] uint8, 20B

    # row-reverse along axis 2 (height): [A;B] -> [B;A], via step -1 Slice
    init("rs", np.array([1], np.int64), np.int64)           # start at row 1
    init("re", np.array([-3], np.int64), np.int64)          # down to (and incl) row 0
    init("rax", np.array([2], np.int64), np.int64)
    init("rst", np.array([-1], np.int64), np.int64)
    n("Slice", ["cols", "rs", "re", "rax", "rst"], "cols_rev")  # [1,10,2,1] uint8, 20B

    # ---- broadcast to the 2x6 block via fixed column-parity mask ------------
    # even columns -> cols ([A;B]); odd columns -> cols_rev ([B;A])
    even = np.array([1, 0, 1, 0, 1, 0], dtype=bool).reshape(1, 1, 1, 6)
    init("evenmask", even, np.bool_)
    n("Where", ["evenmask", "cols", "cols_rev"], "block")   # [1,10,2,6] uint8, 120B

    # ---- pad the block into the FREE 30x30 output ---------------------------
    # block sits at top-left; pad bottom rows (30-2=28) and right cols (30-6=24)
    pad_amt = np.array([0, 0, 0, 0, 0, 0, S - 2, S - 6], dtype=np.int64)
    init("pad_amt", pad_amt, np.int64)
    init("pad_val", np.array(0, np.uint8), np.uint8)
    n("Pad", ["block", "pad_amt", "pad_val"], "output", mode="constant")

    # ---- value_info ---------------------------------------------------------
    def vi(name, dt, shape):
        return helper.make_tensor_value_info(name, dt, shape)

    inp = vi("input", TensorProto.FLOAT, [1, 10, S, S])
    out = vi("output", U8, [1, 10, S, S])
    vinfos = [
        vi("cols_f", TensorProto.FLOAT, [1, 10, 2, 1]),
        vi("cols", U8, [1, 10, 2, 1]),
        vi("cols_rev", U8, [1, 10, 2, 1]),
        vi("block", U8, [1, 10, 2, 6]),
    ]

    graph = helper.make_graph(nodes, "task373", [inp], [out], inits, value_info=vinfos)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = IR_VERSION
    return model
