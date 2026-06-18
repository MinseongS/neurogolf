"""task181 (ARC-AGI 760b3cac) — mirror the cyan sprite to the opposite outer side.

Rule (from the ARC-GEN generator, verified fresh):
  Active grid is 6 rows x 9 cols (placed top-left in the 30x30 canvas).
  A 3-wide cyan (colour 8) "conway" sprite sits in columns 3..5 and ROWS 0..2
  (cell grid[r][c+3] for sprite-col c in 0..2).  A FIXED yellow (colour 4) shape
  sits in ROWS 3..5, cols 3..5: pixels (3,3),(4,3),(4,4),(4,5),(5,4).
  A `flip` flag horizontally mirrors the WHOLE 9-wide grid (col j -> 8-j) at the
  end (applied to both input and output identically).  Cyan and yellow occupy
  DISJOINT row bands (0-2 vs 3-5).

  OUTPUT = INPUT plus a horizontally-reflected COPY of the cyan sprite on the
  opposite outer side:
    - flip = 0:  reflected block lands in cols 0..2  (out col m <- in col 5-m).
    - flip = 1:  reflected block lands in cols 6..8  (out col m <- in col 11-m).
  Yellow is unchanged.  The flip flag is recoverable from the yellow shape:
  yellow at (3,3) <=> flip=0 ;  yellow at (3,5) <=> flip=1.
  The cyan sprite always occupies input cols 3..5 regardless of flip.

Construction (6x9 active grid only; one-hot routed into the FREE output via Pad):
  Work entirely on the 6x9 active region (never a 30x30 intermediate until the
  final output Pad).  Cyan reflection is a column Gather on the 6x9 cyan mask with
  a runtime (flip-selected) column index that maps each output col to its
  reflected source col (or a guaranteed-empty cyan col 0).  OR with the original
  cyan.  Top 3 rows carry cyan (value 8), bottom 3 rows carry the unchanged yellow
  (value 4); build a [1,1,6,9] uint8 colour-index plane, Equal vs the colour ramp
  -> [1,10,6,9] one-hot, single final Pad to 30x30 = the FREE output.
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

    F = TensorProto.FLOAT
    U8 = TensorProto.UINT8
    NI = TensorProto.INT32
    B = TensorProto.BOOL

    init("zf", np.array(0.0, np.float32), np.float32)

    # ---- cyan mask on the 6x9 grid, rows 0..2 (cyan only lives there) -----------
    # Slice input[:, 8:9, 0:3, 0:9]  (cyan channel, top 3 rows, all 9 cols)
    init("cy_lo", np.array([0, 8, 0, 0], np.int64), np.int64)
    init("cy_hi", np.array([1, 9, 3, 9], np.int64), np.int64)
    init("ax0123", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "cy_lo", "cy_hi", "ax0123"], "cyf")   # [1,1,3,9] fp32
    n("Greater", ["cyf", "zf"], "cyb")                          # [1,1,3,9] bool

    # ---- flip scalar from yellow ch4 at (3,5) -----------------------------------
    # Slice input[:, 4:5, 3:4, 5:6] -> [1,1,1,1]
    init("fl_lo", np.array([0, 4, 3, 5], np.int64), np.int64)
    init("fl_hi", np.array([1, 5, 4, 6], np.int64), np.int64)
    n("Slice", ["input", "fl_lo", "fl_hi", "ax0123"], "flcell")  # [1,1,1,1] fp32
    n("Greater", ["flcell", "zf"], "flipb")                      # [1,1,1,1] bool
    init("one1", np.array([1], np.int64), np.int64)
    n("Reshape", ["flipb", "one1"], "flips")                     # [1] bool

    # ---- reflected-column index (runtime flip-selected) -------------------------
    # cyan source cols 3,4,5 ; reflect onto opposite side.
    # non-flip: out col 0<-5, 1<-4, 2<-3 ; others -> empty cyan col 0
    # flip:     out col 6<-5, 7<-4, 8<-3 ; others -> empty cyan col 0
    idxA = np.zeros(9, dtype=np.int32)
    idxA[0] = 5; idxA[1] = 4; idxA[2] = 3
    idxB = np.zeros(9, dtype=np.int32)
    idxB[6] = 5; idxB[7] = 4; idxB[8] = 3
    init("idxA", idxA, np.int32)
    init("idxB", idxB, np.int32)
    n("Where", ["flips", "idxB", "idxA"], "colidx")              # [9] int32
    n("Gather", ["cyb", "colidx"], "refl", axis=3)               # [1,1,3,9] bool
    n("Or", ["cyb", "refl"], "cy_out")                           # [1,1,3,9] bool

    # ---- top colour-index plane (cyan=8) ----------------------------------------
    init("c8", np.array(8, np.uint8), np.uint8)
    init("z8", np.array(0, np.uint8), np.uint8)
    n("Where", ["cy_out", "c8", "z8"], "top_scalar")             # [1,1,3,9] uint8

    # ---- bottom colour-index plane: yellow rows 3..5 unchanged (value 4) --------
    init("yl_lo", np.array([0, 4, 3, 0], np.int64), np.int64)
    init("yl_hi", np.array([1, 5, 6, 9], np.int64), np.int64)
    n("Slice", ["input", "yl_lo", "yl_hi", "ax0123"], "ylf")     # [1,1,3,9] fp32
    n("Greater", ["ylf", "zf"], "ylb")                           # [1,1,3,9] bool
    init("c4", np.array(4, np.uint8), np.uint8)
    n("Where", ["ylb", "c4", "z8"], "bot_scalar")                # [1,1,3,9] uint8

    # ---- assemble 6x9 colour-index, one-hot, pad to 30x30 -----------------------
    n("Concat", ["top_scalar", "bot_scalar"], "scalar6x9", axis=2)  # [1,1,6,9] uint8
    # only colours 0..8 ever appear (channel 9 / maroon is never used), so build a
    # 9-channel one-hot and pad the missing channel 9 to zero in the final Pad.
    cv = np.arange(9, dtype=np.uint8).reshape(1, 9, 1, 1)
    init("cvals", cv, np.uint8)
    n("Equal", ["scalar6x9", "cvals"], "onehot6x9")              # [1,9,6,9] bool
    # pad: +1 channel (9->10) at end, rows 6->30, cols 9->30
    init("pad30", np.array([0, 0, 0, 0, 0, 1, S - 6, S - 9], np.int64), np.int64)  # begins=0; ends: +1 ch, +24 row, +21 col
    init("falseb", np.array(False, np.bool_), np.bool_)
    n("Pad", ["onehot6x9", "pad30", "falseb"], "output")         # [1,10,30,30] bool (FREE)

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task181", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 13)])
