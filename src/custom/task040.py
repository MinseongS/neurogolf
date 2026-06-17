"""Task 040 (2204b7a8): recolour green markers toward their nearest border.

Rule (from ARC-GEN generator, size=10):
  Grid 10x10. Two non-green colours c0=colors[0], c1=colors[1].
  Non-transposed layout:
    col 0  = c0 (all rows),  col 9 = c1 (all rows).
    Green(3) markers sit in cols 1..8.  Each green cell at (r,c) becomes
      c0 if c < 5  else c1.
  Transposed layout (xpose=1): the whole grid is transposed, so
    row 0 = c0, row 9 = c1, green at (r,c) -> c0 if r < 5 else c1.
  Borders are unchanged; only green cells are recoloured.

  Observation: the replacement colour is a COPY of a border cell on the same
  line, so no colour detection is needed -- only the orientation (xpose) and the
  per-cell border-copy plane.
    non-xpose:  out[r,c] = leftcol[r]  if c<5 else rightcol[r]   (leftcol=V[:,0], rightcol=V[:,9])
    xpose:      out[r,c] = toprow[c]   if r<5 else botrow[c]

Encoding (Tier B label-map on the tiny 10x10 active canvas):
  V = colour-index plane (Conv [0..9] over one-hot input), sliced to 10x10, uint8.
  Orientation scalar is_xpose = (every cell of row 0 is nonzero)  -- only the
  transposed layout fills the whole top row with c0; the non-transposed top row
  has interior zeros.
  Build candidate target planes by broadcasting the four border slices, select
  by is_xpose, then override green cells.  Pad to 30x30 (sentinel 10) and a
  single Equal(L, arange) writes straight into the free BOOL output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

WORK = 10  # active grid side (size = 10)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- constants ----
    init("kw", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    init("half", np.array(0.5, np.float32), np.float32)
    init("five", np.array(5.0, np.float32), np.float32)

    init("ar_row", np.arange(WORK, dtype=np.float32).reshape(1, 1, WORK, 1), np.float32)
    init("ar_col", np.arange(WORK, dtype=np.float32).reshape(1, 1, 1, WORK), np.float32)

    # crop the 30x30 colour-index plane to the active 10x10 corner
    init("crop_st", np.array([0, 0], np.int64), np.int64)
    init("crop_en", np.array([WORK, WORK], np.int64), np.int64)
    init("crop_ax", np.array([2, 3], np.int64), np.int64)

    # edge-line slices
    init("e0_st", np.array([0], np.int64), np.int64)
    init("e0_en", np.array([1], np.int64), np.int64)
    init("e9_st", np.array([WORK - 1], np.int64), np.int64)
    init("e9_en", np.array([WORK], np.int64), np.int64)
    init("ax2", np.array([2], np.int64), np.int64)
    init("ax3", np.array([3], np.int64), np.int64)

    # pad L (10x10 uint8) -> 30x30 with sentinel 10
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)

    # ---- V = per-cell input colour index (uint8 10x10) ----
    n("Conv", ["input", "kw"], "Vbig")                            # [1,1,30,30] f32
    n("Slice", ["Vbig", "crop_st", "crop_en", "crop_ax"], "Vf")   # [1,1,10,10] f32
    n("Cast", ["Vf"], "V", to=TensorProto.UINT8)                  # [1,1,10,10] uint8

    # ---- border-line slices (uint8) ----
    n("Slice", ["V", "e0_st", "e0_en", "ax3"], "leftcol")    # [1,1,10,1]  V[:, :, :, 0]
    n("Slice", ["V", "e9_st", "e9_en", "ax3"], "rightcol")   # [1,1,10,1]  V[:, :, :, 9]
    n("Slice", ["V", "e0_st", "e0_en", "ax2"], "toprow")     # [1,1,1,10]  V[:, :, 0, :]
    n("Slice", ["V", "e9_st", "e9_en", "ax2"], "botrow")     # [1,1,1,10]  V[:, :, 9, :]

    # ---- candidate target planes ----
    # non-xpose: choose by column half (c<5 -> leftcol else rightcol)
    n("Less", ["ar_col", "five"], "col_lo")                  # [1,1,1,10] bool
    n("Where", ["col_lo", "leftcol", "rightcol"], "tgtA")    # [1,1,10,10] uint8
    # xpose: choose by row half (r<5 -> toprow else botrow)
    n("Less", ["ar_row", "five"], "row_lo")                  # [1,1,10,1] bool
    n("Where", ["row_lo", "toprow", "botrow"], "tgtB")       # [1,1,10,10] uint8

    # ---- orientation scalar: is_xpose = (row 0 fully nonzero AND row 9 fully) ----
    # Only the transposed layout fills BOTH the top and bottom rows entirely
    # (they are the two colour borders).  A non-transposed grid would need 8
    # greens in row 0 AND 8 in row 9 = 16 pixels to fake it, but at most 10
    # pixels are ever placed -> impossible.  Min over each row of the nonzero
    # indicator; both must be full.
    n("Cast", ["toprow"], "toprow_f", to=TensorProto.FLOAT)  # [1,1,1,10]
    n("Cast", ["botrow"], "botrow_f", to=TensorProto.FLOAT)  # [1,1,1,10]
    n("Greater", ["toprow_f", "half"], "top_nz")             # nonzero indicator
    n("Greater", ["botrow_f", "half"], "bot_nz")
    n("And", ["top_nz", "bot_nz"], "tb_nz")                  # [1,1,1,10] bool
    n("Cast", ["tb_nz"], "tb_nz_f", to=TensorProto.FLOAT)
    n("ReduceMin", ["tb_nz_f"], "all_tb", keepdims=1)        # [1,1,1,1] 1.0 iff both full
    n("Greater", ["all_tb", "half"], "is_xpose")             # [1,1,1,1] bool

    # ---- select target plane by orientation ----
    n("Where", ["is_xpose", "tgtB", "tgtA"], "tgt")          # [1,1,10,10] uint8

    # ---- override green cells only: green <=> colour index 3 (reuse V) ----
    init("u3", np.array(3, np.uint8), np.uint8)
    n("Equal", ["V", "u3"], "is_green")                       # [1,1,10,10] bool
    n("Where", ["is_green", "tgt", "V"], "L10")               # [1,1,10,10] uint8

    # ---- pad to 30x30 and final Equal -> BOOL output ----
    n("Pad", ["L10", "padpads", "padval"], "L", mode="constant")  # [1,1,30,30] u8
    n("Equal", ["L", "chan"], "output")                           # free BOOL output

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
