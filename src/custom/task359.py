"""task359 (ARC-AGI e26a3af2) — "restore the striped background".

Rule (from the generator):
  * The grid (width=sum(wides) x height, anchored top-left) is tiled with
    consecutive VERTICAL stripes: band idx has a fixed stripe colour and spans
    `wide` consecutive columns.  (If xpose=1 the whole figure is transposed, so
    the stripes run HORIZONTALLY instead.)
  * Sparse "noise" pixels (prob 0.1) of arbitrary colours 1..9 are stamped on
    top of the stripes in the INPUT.
  * OUTPUT = the clean stripes (every cell restored to its band's stripe colour).

Recovery (verified 0 errors over thousands of fresh instances):
  Every in-grid cell is a colour 1..9 (background 0 only appears OFF the grid,
  where convert_to_numpy leaves ALL channels = 0).  So:
    * colcount[1,10,1,30] = ReduceSum(input, rows)  -> per-(colour,column) count
    * rowcount[1,10,30,1] = ReduceSum(input, cols)  -> per-(colour,row)    count
  Within a stripe column the stripe colour overwhelmingly dominates (noise is
  <10%), so its per-column ARGMAX over channels = the stripe colour (channel 0
  count is 0 everywhere, so argmax over all 10 channels already excludes bg).
    * col_idx[1,1,1,30] = ArgMax(colcount, ch)   (vertical-stripe hypothesis)
    * row_idx[1,1,30,1] = ArgMax(rowcount, ch)   (horizontal-stripe hypothesis)
  Orientation (xpose) is recovered WITHOUT any per-cell plane: the correct
  orientation maximises the total "peak" match count
    peak_col = sum_c max_ch colcount[.,c]   vs   peak_row = sum_r max_ch rowcount[r,.]
  (the wrong axis mixes several band colours per line, lowering its peak).
  Select the index vector by use_col = peak_col >= peak_row.
    selected[1,1,30,30] = Where(use_col, col_idx[1,1,1,30], row_idx[1,1,30,1])
  routes the chosen per-line colour across the whole canvas in ONE broadcast.
  In-grid mask is separable (rowany[1,1,30,1] x colany[1,1,1,30]); off-grid
  cells are set to sentinel 10 so the final Equal(L,[0..9]) leaves them all-zero
  (= background), matching the off-grid target.  No [1,10,*,*] plane is built.

Dominant intermediates: the two [1,10,1,30]/[1,10,30,1] fp32 reductions (1200 B
each, irreducible — both stripe orientations must be probed) and the [1,1,30,30]
uint8 label plane just before the FREE Equal->output.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION


def build(task):
    inits, nodes = [], []

    def init(name, arr, npdtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=npdtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    U8 = TensorProto.UINT8
    B = TensorProto.BOOL

    # ---- constants ----
    init("chan", np.arange(10).reshape(1, 10, 1, 1), np.uint8)   # [0..9] colour channels
    init("zerof", np.array(0.0, np.float32), np.float32)
    init("ten", np.array(10, np.uint8), np.uint8)                # off-grid sentinel

    # ---- per-(colour,column) and per-(colour,row) counts ----
    n("ReduceSum", ["input"], "colcount", axes=[2], keepdims=1)  # [1,10,1,30] fp32
    n("ReduceSum", ["input"], "rowcount", axes=[3], keepdims=1)  # [1,10,30,1] fp32

    # ---- per-line stripe colour = argmax over channels (ch0 count==0 always) ----
    n("ArgMax", ["colcount"], "col_arg", axis=1, keepdims=1)     # [1,1,1,30] int64
    n("ArgMax", ["rowcount"], "row_arg", axis=1, keepdims=1)     # [1,1,30,1] int64
    n("Cast", ["col_arg"], "col_idx", to=U8)                     # [1,1,1,30] uint8
    n("Cast", ["row_arg"], "row_idx", to=U8)                     # [1,1,30,1] uint8

    # ---- orientation discriminator (scalars only) ----
    n("ReduceMax", ["colcount"], "cpeak", axes=[1], keepdims=1)  # [1,1,1,30]
    n("ReduceSum", ["cpeak"], "cpeaksum", axes=[2, 3], keepdims=1)  # scalar
    n("ReduceMax", ["rowcount"], "rpeak", axes=[1], keepdims=1)  # [1,1,30,1]
    n("ReduceSum", ["rpeak"], "rpeaksum", axes=[2, 3], keepdims=1)  # scalar
    # use_col = cpeaksum >= rpeaksum  (opset-11 has no GreaterOrEqual)
    n("Less", ["cpeaksum", "rpeaksum"], "col_lt")
    n("Not", ["col_lt"], "use_col")                              # scalar bool

    # ---- select chosen per-line colour, broadcast across the canvas ----
    # Where([1,1,1,1], [1,1,1,30], [1,1,30,1]) -> [1,1,30,30]
    n("Where", ["use_col", "col_idx", "row_idx"], "selected")    # [1,1,30,30] uint8

    # ---- separable in-grid mask: a line is in-grid iff it has any colour ----
    n("ReduceSum", ["rowcount"], "rowany_s", axes=[1], keepdims=1)  # [1,1,30,1] fp32
    n("Greater", ["rowany_s", "zerof"], "rowany")                # [1,1,30,1] bool
    n("ReduceSum", ["colcount"], "colany_s", axes=[1], keepdims=1)  # [1,1,1,30] fp32
    n("Greater", ["colany_s", "zerof"], "colany")                # [1,1,1,30] bool
    n("And", ["rowany", "colany"], "ingrid")                     # [1,1,30,30] bool

    # ---- label plane: in-grid -> stripe colour, off-grid -> sentinel 10 ----
    n("Where", ["ingrid", "selected", "ten"], "L")               # [1,1,30,30] uint8

    # ---- one-hot expansion into the FREE output (BOOL) ----
    n("Equal", ["L", "chan"], "output")                          # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task359", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
