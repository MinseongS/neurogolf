"""Task 303 (ARC-GEN c1d99e64): recolour all-black rows/cols to red.

Rule (from ARC-GEN generator, verified fresh):
  The input grid has background black (colour 0) and exactly one foreground
  colour.  A set of full ROWS and full COLS were overwritten to all-black
  ("straightaways").  The generator first builds the static pattern so that
  EVERY in-grid row and EVERY in-grid col contains >=2 colours, THEN paints the
  chosen rows/cols entirely black.  Therefore a "line" row/col is exactly an
  in-grid row/col whose every cell is black.
  Output: output[r][c] = red(2) if (row r is all-black OR col c is all-black)
  else input[r][c].

Encoding (Tier A, separable row/col line masks, NO [1,*,30,30] colour plane):
  A row r is a line iff (#black cells == #in-grid cells in the row) AND
  (#in-grid cells > 0).  Off-grid cells set NO channel, so a per-row count of
  channel-0 (black) vs the per-row count over ALL channels (in-grid)
  distinguishes an all-black IN-GRID row from an off-grid (all-zero) row.  Both
  counts come from one column-reduce rsum[1,10,30,1] (= ReduceSum(input,[3]),
  1200 B): in_row = ReduceSum(rsum, ch); black_row = rsum[:,0:1].  Symmetric for
  columns via csum[1,10,1,30].  rowline[1,1,30,1]/colline[1,1,1,30] are tiny;
  linemask = OR broadcasts to a [1,1,30,30] bool plane (900 B), and the single
  final op Where(linemask, red_onehot[1,10,1,1], input) writes straight into the
  FREE output (off-grid stays all-zero: input is zero there and off-grid
  rows/cols are never lines).

  Dominant intermediate: rsum/csum [1,10,30,1] = 1200 B each (irreducible -- a
  per-channel single-axis reduce splits the black-count from the in-grid count
  without ever materialising a [1,*,30,30] colour plane).
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    B = TensorProto.BOOL

    init("zero", np.array(0.0, np.float32), np.float32)
    init("ch_ax", np.array([0, 1, 2, 3], np.int64), np.int64)

    # --- per-row counts -----------------------------------------------------
    n("ReduceSum", ["input"], "rsum", axes=[3], keepdims=1)        # [1,10,30,1] f32
    n("ReduceSum", ["rsum"], "in_row", axes=[1], keepdims=1)       # [1,1,30,1] in-grid count
    init("r0_s", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("r0_e", np.array([1, 1, 30, 1], np.int64), np.int64)
    n("Slice", ["rsum", "r0_s", "r0_e", "ch_ax"], "black_row")     # [1,1,30,1] black count

    # --- per-col counts -----------------------------------------------------
    n("ReduceSum", ["input"], "csum", axes=[2], keepdims=1)        # [1,10,1,30] f32
    n("ReduceSum", ["csum"], "in_col", axes=[1], keepdims=1)       # [1,1,1,30] in-grid count
    init("c0_s", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("c0_e", np.array([1, 1, 1, 30], np.int64), np.int64)
    n("Slice", ["csum", "c0_s", "c0_e", "ch_ax"], "black_col")     # [1,1,1,30] black count

    # --- line masks ---------------------------------------------------------
    # row is a line iff black_row == in_row (all in-grid cells black) AND
    # in_row > 0 (the row is actually in-grid).
    n("Equal", ["black_row", "in_row"], "row_allblack")            # [1,1,30,1] bool
    n("Greater", ["in_row", "zero"], "row_ingrid")                 # [1,1,30,1] bool
    n("And", ["row_allblack", "row_ingrid"], "rowline")            # [1,1,30,1] bool

    n("Equal", ["black_col", "in_col"], "col_allblack")            # [1,1,1,30] bool
    n("Greater", ["in_col", "zero"], "col_ingrid")                 # [1,1,1,30] bool
    n("And", ["col_allblack", "col_ingrid"], "colline")            # [1,1,1,30] bool

    n("Or", ["rowline", "colline"], "lineraw")                     # [1,1,30,30] bool
    # gate to the in-grid rectangle: cell in-grid iff its row AND col are in-grid
    n("And", ["row_ingrid", "col_ingrid"], "ingrid")               # [1,1,30,30] bool
    n("And", ["lineraw", "ingrid"], "linemask")                    # [1,1,30,30] bool

    # --- route into FREE output --------------------------------------------
    red = np.zeros((1, 10, 1, 1), np.float32)
    red[0, 2, 0, 0] = 1.0
    init("red", red, np.float32)
    n("Where", ["linemask", "red", "input"], "output")             # FREE [1,10,30,30]

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task303", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
