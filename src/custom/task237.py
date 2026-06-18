"""task237 (ARC-AGI 99fa7670) — "rightward ray from each seed + downward ray on the
last column".

Rule (from the generator):
  Grid is W x H, both in [3..9].  Background is 0.  At certain rows r (spaced >=2
  apart, so AT MOST ONE seed per row) a single coloured pixel sits at column c with
  colour v.  The OUTPUT draws, for each seed (r, c, v):
    * a horizontal ray:  output[r][c .. W-1] = v   (fill row r from c rightward)
    * a vertical ray:    output[r .. H-1][W-1] = v  (fill the LAST column downward)
  When several seeds hit the last column, the lowest seed at-or-above each row wins
  (later/lower seeds overwrite) -> a forward-fill (last-observed-colour) down the
  last column, NOT a running-max.

Encoding (small 9x9 active canvas; grid <=9x9 by the generator bounds):
  colf = sum_k k*input_k                      [1,1,9,9] colour index, 0=bg/off-grid
  ingrid = ReduceMax(input, ch axis)          occupancy (1 in-grid, 0 off-grid)
  HORIZONTAL ray = CumSum(colf, cols) (exactly one nonzero per row => prefix sum is
    the rightward fill), masked by ingrid so it stops at the grid's right edge.
  lastcol[i] = ReduceMax(hor, cols)           per-row seed colour (0 if no seed)
  FORWARD-FILL down the last column as a MatMul:  ff = M @ lastcol where
    grp = CumSum(lastcol!=0) (segment id, increments at each seed) and
    M[i][j] = (grp[i]==grp[j]) selects each row's segment-start seed colour.
  lastmask = ingrid AND NOT shift_left(ingrid)  -> the rightmost in-grid col per row.
  res = Where(lastmask, ff, hor)              colour index plane
  L = Where(ingrid, res, SENTINEL); Pad to 30x30 with SENTINEL; output =
    Equal(L, arange[0..9]) -> BOOL (off-grid + padding stay all-zero).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64

W = 9  # active canvas (grid is at most 9x9 by the generator bounds)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    init("ZEROF", np.array(0.0, np.float32), np.float32)

    # ---- colour-index plane  colf = sum_k k*input_k (channels 1..9 only) ---
    # ch0 weight would be 0, so slice it out: smaller 9-channel 9x9 slice.
    init("cs_s", np.array([0, 1, 0, 0], np.int64), np.int64)
    init("cs_e", np.array([1, 10, W, W], np.int64), np.int64)
    init("cs_ax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "cs_s", "cs_e", "cs_ax"], "x19")   # [1,9,W,W] f32
    cw = np.arange(1, 10, dtype=np.float32).reshape(1, 9, 1, 1)
    init("colw", cw, np.float32)
    n("Conv", ["x19", "colw"], "colf")                      # [1,1,W,W] f32

    # ---- separable in-grid masks (NO 2-D occupancy plane) ------------------
    # convert_to_numpy sets some channel=1 for every in-grid cell (ch0 for bg),
    # so ReduceMax over channels+the other spatial axis recovers the H/W extent.
    n("ReduceMax", ["input"], "rowmax", axes=[1, 3], keepdims=1)  # [1,1,30,1]
    n("ReduceMax", ["input"], "colmax", axes=[1, 2], keepdims=1)  # [1,1,1,30]
    # slice each to the 9-canvas extent
    init("rs_s", np.array([0], np.int64), np.int64)
    init("rs_e", np.array([W], np.int64), np.int64)
    init("rs_ax2", np.array([2], np.int64), np.int64)
    init("rs_ax3", np.array([3], np.int64), np.int64)
    n("Slice", ["rowmax", "rs_s", "rs_e", "rs_ax2"], "rowmax9")  # [1,1,W,1]
    n("Slice", ["colmax", "rs_s", "rs_e", "rs_ax3"], "colmax9")  # [1,1,1,W]
    n("Greater", ["rowmax9", "ZEROF"], "rowin")             # bool [1,1,W,1]
    n("Greater", ["colmax9", "ZEROF"], "colin")             # bool [1,1,1,W]
    n("And", ["rowin", "colin"], "ingrid")                  # bool [1,1,W,W]
    n("Cast", ["ingrid"], "ingridf", to=F32)                # f32 {0,1}

    # ---- horizontal ray = CumSum along columns (axis=3), then mask ---------
    init("ax3i", np.array(3, np.int64), np.int64)
    n("CumSum", ["colf", "ax3i"], "hcum")                   # [1,1,W,W] f32
    n("Mul", ["hcum", "ingridf"], "hor")                    # masked off-grid -> 0

    # ---- per-row seed colour (last in-grid column value) -------------------
    n("ReduceMax", ["hor"], "lastcol", axes=[3], keepdims=1)  # [1,1,W,1] f32

    # ---- forward-fill down the last column via segment MatMul --------------
    n("Greater", ["lastcol", "ZEROF"], "nz_b")              # bool [1,1,W,1]
    n("Cast", ["nz_b"], "nz_f", to=F32)
    init("ax2i", np.array(2, np.int64), np.int64)
    n("CumSum", ["nz_f", "ax2i"], "grp")                    # [1,1,W,1] f32 group id
    # grpT on axis3 -> Equal broadcasts to [1,1,W,W]
    n("Transpose", ["grp"], "grpT", perm=[0, 1, 3, 2])      # [1,1,1,W]
    n("Equal", ["grp", "grpT"], "M_b")                      # bool [1,1,W,W]
    n("Cast", ["M_b"], "M_f", to=F32)
    n("MatMul", ["M_f", "lastcol"], "ff")                   # [1,1,W,1] f32
    # mask to in-grid rows (reuse the separable rowin)
    n("Cast", ["rowin"], "rowinf", to=F32)                  # f32 [1,1,W,1]
    n("Mul", ["ff", "rowinf"], "ffm")                       # [1,1,W,1]

    # ---- last-in-grid-column = colin AND NOT shift_left(colin) (1-D) -------
    # shift_left along axis3: pad one zero col at right, then drop first col.
    n("Cast", ["colin"], "colinf", to=F32)                  # f32 [1,1,1,W]
    init("pad_sl", np.array([0, 0, 0, 0, 0, 0, 0, 1], np.int64), np.int64)
    n("Pad", ["colinf", "pad_sl"], "col_padR", mode="constant")  # [1,1,1,W+1]
    init("sl_s", np.array([1], np.int64), np.int64)
    init("sl_e", np.array([W + 1], np.int64), np.int64)
    init("sl_ax", np.array([3], np.int64), np.int64)
    n("Slice", ["col_padR", "sl_s", "sl_e", "sl_ax"], "col_shift")  # [1,1,1,W]
    n("Greater", ["col_shift", "ZEROF"], "col_shift_b")     # bool: has right nbr
    n("Not", ["col_shift_b"], "no_right")                   # bool [1,1,1,W]
    n("And", ["rowin", "no_right"], "lastmask")             # bool [1,1,W,W]

    # ---- combine: res = Where(lastmask, ff, hor) ---------------------------
    n("Where", ["lastmask", "ffm", "hor"], "res")           # [1,1,W,W] f32 index

    # ---- to uint8 label, sentinel off-grid, pad to 30x30 -------------------
    n("Cast", ["res"], "res_u8", to=U8)                     # [1,1,W,W] u8
    init("SENT", np.array(99, np.uint8), np.uint8)
    n("Where", ["ingrid", "res_u8", "SENT"], "Lw")          # u8, off-grid=99
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64), np.int64)
    n("Pad", ["Lw", "pads", "SENT"], "L", mode="constant")  # [1,1,30,30] u8

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                     # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task237", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
