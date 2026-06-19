"""task351 (ARC-AGI dc0a314f) — fill the green hole from D2 symmetry (Tier-S copy).

Rule (from generator):
  The grid is 2*size = 16 wide and FULLY D2-symmetric: an 8x8 quadrant is
  mirrored both horizontally and vertically, so
      value(r,c) == value(15-r,c) == value(r,15-c) == value(15-r,15-c).
  A 5x5 cutout at (row, col) is erased to GREEN(=3); green appears NOWHERE else
  (random_color excludes green).  The 5x5 OUTPUT is the original cutout content,
  reconstructed from the intact BOTTOM-RIGHT double mirror:

      output[i][j] = grid[15-row-i][15-col-j]

  Pure SPATIAL COPY of input cells.  The stored examples place the hole as far as
  (row=5, col=9), so the source can reach rows 6..15 / cols 2..15 — it is NOT a
  fixed corner quadrant; the random generate() path clamps row,col to {0..3}.

PLANE-ELIMINATED build (route the 10-channel expansion into the FREE output):
  Instead of gathering the 10-channel one-hot (forces a [1,10,5,30]=6000B fp32
  plane), collapse to a SINGLE colour-index plane first, gather that (1-channel,
  6x cheaper), then expand the 5x5 block back to a one-hot ON THE BLOCK and Pad
  that tiny uint8 block straight into the FREE output:

  1. colf = Conv(input, W[1,10,1,1] = [0,1,..,9])  -> [1,1,30,30] f32 (3600B,
     the one unavoidable fp32 colour-index entry plane; bg=ch0 -> value 0).
  2. recover (row,col): green channel (3) sliced to a tiny rectangular corner
     [1,1,7,11] (covers first-green row<=5 / col<=9 with margin); ArgMax of the
     {0,1} occupancy gives the FIRST green row/col = (row, col).
  3. Vr = Gather(colf, 15-row-arange(5), axis=2)  -> [1,1,5,30] (600B)
     Vs = Gather(Vr,   15-col-arange(5), axis=3)  -> [1,1,5,5]  (100B)
  4. oh = Equal(Vs, arange_ch[1,10,1,1])  -> [1,10,5,5] bool (Equal accepts fp32,
     integer-exact); Cast -> uint8; Pad to 30x30 (off-block all-zero, opset-11
     Pad accepts uint8).  Output declared UINT8 (scored as out>0).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
I64 = TensorProto.INT64
U8 = TensorProto.UINT8


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    WORK = 5

    # ---- recover hole top-left (row, col) -----------------------------------
    # Green hole's FIRST row<=5 and FIRST col<=9 (stored stress) / <=3 (random),
    # so slice GREEN channel (3) to a small rectangular corner [1,1,7,11].
    init("g_starts", np.array([3, 0, 0], np.int64), np.int64)
    init("g_ends", np.array([4, 7, 11], np.int64), np.int64)
    init("g_axes", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "g_starts", "g_ends", "g_axes"], "green")  # [1,1,7,11]
    n("ReduceMax", ["green"], "rowhas", axes=[3], keepdims=1)  # [1,1,7,1]
    n("ReduceMax", ["green"], "colhas", axes=[2], keepdims=1)  # [1,1,1,11]
    n("ArgMax", ["rowhas"], "row_i", axis=2, keepdims=1)       # [1,1,1,1] int64
    n("ArgMax", ["colhas"], "col_i", axis=3, keepdims=1)       # [1,1,1,1] int64
    init("shp1", np.array([1], np.int64), np.int64)
    n("Reshape", ["row_i", "shp1"], "row_s")                  # [1] int64
    n("Reshape", ["col_i", "shp1"], "col_s")                  # [1] int64

    # mirror source indices  idx = 15 - off - arange(WORK)
    base = (15 - np.arange(WORK)).astype(np.int64)            # [15,14,13,12,11]
    init("base", base, np.int64)                              # [WORK] int64
    n("Sub", ["base", "row_s"], "ridx")                       # [WORK] int64
    n("Sub", ["base", "col_s"], "cidx")                       # [WORK] int64

    # ---- single colour-index plane: colf = sum_k k * input_k -----------------
    cw = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("cw", cw, np.float32)
    n("Conv", ["input", "cw"], "colf")                        # [1,1,30,30] f32

    # ---- gather the mirrored 5x5 colour-index window -------------------------
    n("Gather", ["colf", "ridx"], "Vr", axis=2)              # [1,1,5,30] f32
    n("Gather", ["Vr", "cidx"], "Vs", axis=3)                # [1,1,5,5] f32

    # ---- expand to one-hot ON THE 5x5 BLOCK, route into the FREE output ------
    arange_ch = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("arange_ch", arange_ch, np.float32)
    n("Equal", ["Vs", "arange_ch"], "oh")                    # [1,10,5,5] bool
    n("Cast", ["oh"], "oh8", to=U8)                          # [1,10,5,5] uint8
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    init("zero", np.array(0, np.uint8), np.uint8)
    n("Pad", ["oh8", "padpads", "zero"], "output", mode="constant")  # [1,10,30,30] u8

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", U8, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task351", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
