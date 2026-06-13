"""Task 398 (feca6190): anti-diagonal rays from a 1x5 color row.

Rule: input is a 1x5 row with n nonzero colors (n=1..5); output is an s x s
grid, s = 5n. Each input column c (color v, zeros included) draws the
anti-diagonal ray output[r][s-1+c-r] = v for r in c..s-1. Equivalently cell
(r,j) inside the grid shows input column c = r+j-(s-1) when 0 <= c <= 4,
black otherwise; outside the s x s grid the canvas is all-zero.

Graph: z = number of zero colors (sum of channel-0 over input row 0 cols
0..4); s = 25-5z, so z in {0..4} selects one of 5 precomputed [30,30] index
maps. The maps are stored row-deduplicated: ROWMAP [5,30] gathers per-row
pattern ids, ROWS [K,30] gathers the actual index rows -> idx [30,30] int32.
Data tensor [1,10,7] = input row columns 0..4 (one-hots; zero colors are
already the black one-hot) ++ const [black e0, all-zero]; idx values: c for
ray cells, 5 for inside-black, 6 for outside. Final Gather(data, idx,
axis=2) writes output directly (free tensor). Only one canvas-sized
intermediate (idx, 3600B).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ..builders import _model


def _index_tables():
    """ROWMAP [5,30] int32, ROWS [K,30] int32 (deduplicated row patterns)."""
    all_rows = []
    for z in range(5):
        s = 25 - 5 * z
        for i in range(30):
            row = np.full(30, 6, np.int32)          # outside -> zero column
            if i < s:
                j = np.arange(s)
                c = i + j - (s - 1)
                row[:s] = np.where((c >= 0) & (c <= 4), c, 5)
            all_rows.append(row)
    rows = np.array(all_rows, np.int32)              # [150, 30]
    uniq, inv = np.unique(rows, axis=0, return_inverse=True)
    rowmap = inv.astype(np.int32).reshape(5, 30)
    return rowmap, uniq.astype(np.int32)


def build(task):
    inits = []
    nodes = []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    rowmap, rows = _index_tables()
    init("ROWMAP", rowmap, np.int32)                 # [5,30]
    init("ROWS", rows, np.int32)                     # [K,30]

    # --- palette: input row 0, columns 0..4, plus [black, zero] columns ---
    init("sl_st", np.array([0, 0], np.int64), np.int64)
    init("sl_en", np.array([1, 5], np.int64), np.int64)
    init("sl_ax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["input", "sl_st", "sl_en", "sl_ax"], "rowslice")  # [1,10,1,5]
    n("Squeeze", ["rowslice"], "row3", axes=[2])                  # [1,10,5]
    extra = np.zeros((1, 10, 2), np.float32)
    extra[0, 0, 0] = 1.0                             # col 5: black one-hot
    init("EXTRA", extra, np.float32)                 # col 6: all-zero
    n("Concat", ["row3", "EXTRA"], "data", axis=2)                # [1,10,7]

    # --- z = number of zero colors -> scalar table index ---
    init("z_st", np.array([0], np.int64), np.int64)
    init("z_en", np.array([1], np.int64), np.int64)
    init("z_ax", np.array([1], np.int64), np.int64)
    n("Slice", ["rowslice", "z_st", "z_en", "z_ax"], "zslice")    # [1,1,1,5]
    n("ReduceSum", ["zslice"], "z", keepdims=0)                   # scalar f
    n("Cast", ["z"], "zi", to=onnx.TensorProto.INT32)             # scalar i32

    # --- index canvas via two-level gather ---
    n("Gather", ["ROWMAP", "zi"], "ridx", axis=0)                 # [30] i32
    n("Gather", ["ROWS", "ridx"], "idx", axis=0)                  # [30,30] i32

    # --- assemble output (free tensor) ---
    n("Gather", ["data", "idx"], "output", axis=2)                # [1,10,30,30]

    return _model(nodes, inits)
