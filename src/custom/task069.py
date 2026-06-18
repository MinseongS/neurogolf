"""task069 (ARC-AGI 321b1fc6) — "recolour every cyan sprite from the one coloured sprite".

Rule (from the generator):
  The grid is always 10x10.  `num_boxes` (4) IDENTICAL connected sprites (an
  arbitrary connected subset of pixels inside a small box: width 2..4, height 2..3)
  are placed at random non-overlapping positions (separated with margin 1).  Exactly
  ONE sprite (the first drawn) is shown in its real per-pixel COLOURS; every other
  sprite is shown all-cyan (colour 8).  OUTPUT = erase the coloured sprite to black,
  and redraw EVERY cyan sprite in the real colour pattern, aligned to each sprite's
  own bounding-box top-left.

  Verified invariants (300+ fresh instances):
    * output nonzero  <=>  input == cyan(8)
    * the coloured input box -> black in the output
    * the colour at a cyan cell = colourmap[(r - bbox_top, c - bbox_left)], where the
      colourmap (offset -> colour, offset relative to the sprite's bounding-box top-left)
      is revealed by the single coloured sprite and is identical for every sprite.

Key structure used (canvas is fixed 10x10):
  * colf  = Sum_k k*input_k via a 1x1 Conv  -> colour-index plane (bg=0, cyan=8,
    palette colours = their channel index).
  * occ   = colf > 0  (any sprite pixel).
  * bbox top/left per cell: propagate the MINIMUM row/col index over 4-CONNECTED
    occupied cells (sprites are 4-connected, <=3x4, and their bounding boxes are
    separated with margin 1 so two sprites never share a 4-edge; 8-connectivity
    WOULD wrongly merge corner-adjacent sprites).  7 iterations of a PLUS-shaped
    min (re-masked to occupancy) recover each cell's bbox-corner exactly.  The
    plus-min = elementwise min of a vertical 1x3 min-pool and a horizontal 3x1
    min-pool (each = Neg -> MaxPool -> Neg, fp16, works under ORT_DISABLE_ALL);
    their elementwise Min is min(self,N,S,E,W) with NO diagonal coupling.
    dr = rowramp - topr   (0..2),  dc = colramp - topc   (0..3),  key = dr*4 + dc.
  * colour map: the coloured sprite cells (colf>0 AND colf!=8) carry colour = colf.
    tableColor[k] = Sum_cells colf*(key==k) over coloured-sprite cells, built as a
    [1,12] vector by MatMul(colf_weighted[1,N], keyOneHot[N,12]).
  * output label L[10x10] = Gather(tableColor, key)  ON CYAN CELLS ONLY (colf==8);
    coloured box and background -> 0.  Pad to 30x30 with sentinel 99 (off-grid ->
    all channels off), final  output = Equal(L, arange[10])  (BOOL, free output).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
I32 = TensorProto.INT32
I64 = TensorProto.INT64
U8 = TensorProto.UINT8

S = 10            # grid is always 10x10
N = S * S         # 100
NK = 12           # keys 0..11  (dr 0..2, dc 0..3 -> dr*4+dc)
BIG = 99.0        # sentinel for "unoccupied" in the min-propagation
ITERS = 7         # plus-min iterations (4-connected, covers <=3x4 sprite)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # --------------------------------------------------- colour-index plane (Conv)
    convw = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("convw", convw, np.float32)
    n("Conv", ["input", "convw"], "colf30")                  # [1,1,30,30] f32
    init("c_s", np.array([0, 0], np.int64), np.int64)
    init("c_e", np.array([S, S], np.int64), np.int64)
    init("c_ax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["colf30", "c_s", "c_e", "c_ax"], "colf")     # [1,1,10,10] f32

    # occ = colf > 0  -> fp16 {0,1}
    init("ZERO32", np.array(0.0, np.float32), np.float32)
    n("Greater", ["colf", "ZERO32"], "occ_b")
    n("Cast", ["occ_b"], "occ", to=F16)                      # [1,1,10,10] fp16

    # --------------------------------------------------- ramps (fp16)
    rowramp = np.arange(S, dtype=np.float16).reshape(1, 1, S, 1)
    colramp = np.arange(S, dtype=np.float16).reshape(1, 1, 1, S)
    init("rowramp", rowramp, np.float16)
    init("colramp", colramp, np.float16)
    init("BIGf", np.array(BIG, np.float16), np.float16)

    # --------------------------------------------------- bbox-top/left via plus-min
    # Work in NEGATED space: nval = -rowindex on occupied, -BIG elsewhere; then
    # "propagate the MIN row index over the plus neighbourhood" = "propagate the
    # MAX of nval", done with two 1-D MaxPools (vertical 3x1, horizontal 1x3) and
    # an elementwise Max (= max over self,N,S,E,W, no diagonal coupling).  Re-mask
    # to -BIG on unoccupied each iter so empties never leak.  One final Neg.
    nBIG = n("Neg", ["BIGf"], "nBIG")                        # -BIG scalar fp16

    def plus_max(cur, tag, it):                              # max(self,N,S,E,W)
        mv = n("MaxPool", [cur], f"{tag}_mv{it}", kernel_shape=[3, 1], pads=[1, 0, 1, 0])
        mh = n("MaxPool", [cur], f"{tag}_mh{it}", kernel_shape=[1, 3], pads=[0, 1, 0, 1])
        return n("Max", [mv, mh], f"{tag}_pm{it}")

    init("nrowramp", -np.arange(S, dtype=np.float16).reshape(1, 1, S, 1), np.float16)
    init("ncolramp", -np.arange(S, dtype=np.float16).reshape(1, 1, 1, S), np.float16)
    n("Where", ["occ_b", "nrowramp", "nBIG"], "rtop0")       # -rowindex / -BIG
    cur = "rtop0"
    for it in range(ITERS):
        pm = plus_max(cur, "rt", it)
        cur = n("Where", ["occ_b", pm, "nBIG"], f"rtop{it+1}")
    n("Neg", [cur], "topr")                                  # min row index
    topr = "topr"

    n("Where", ["occ_b", "ncolramp", "nBIG"], "ctop0")
    cur = "ctop0"
    for it in range(ITERS):
        pm = plus_max(cur, "ct", it)
        cur = n("Where", ["occ_b", pm, "nBIG"], f"ctop{it+1}")
    n("Neg", [cur], "topc")
    topc = "topc"

    # dr = rowramp - topr ; dc = colramp - topc  (only meaningful where occupied)
    n("Sub", ["rowramp", topr], "dr")                        # broadcast
    n("Sub", ["colramp", topc], "dc")
    # key = dr*4 + dc  (gate to occupied so unoccupied -> 0)
    init("FOURh", np.array(4.0, np.float16), np.float16)
    n("Mul", ["dr", "FOURh"], "dr4")
    n("Add", ["dr4", "dc"], "key_raw")                       # fp16
    n("Mul", ["key_raw", "occ"], "key")                      # 0 where unoccupied
    # defensive clamp to [0,11] (occupied keys are always in range; guards Gather)
    init("KLO", np.array(0.0, np.float16), np.float16)
    init("KHI", np.array(float(NK - 1), np.float16), np.float16)
    n("Max", ["key", "KLO"], "key_lo")
    n("Min", ["key_lo", "KHI"], "key_c")
    n("Cast", ["key_c"], "key_i", to=I32)                    # indices [1,1,10,10]
    init("key_flat_shape", np.array([N], np.int64), np.int64)
    n("Reshape", ["key_i", "key_flat_shape"], "key_iflat")   # [N]

    # --------------------------------------------------- colour-map table[12]
    # coloured-sprite cells: colf>0 AND colf!=8 (cyan).  weight = colf (the colour).
    # col_w = colf * (colf!=8): bg(colf=0)->0, cyan(colf=8)->0, coloured->colf.
    init("CYAN32", np.array(8.0, np.float32), np.float32)
    n("Equal", ["colf", "CYAN32"], "is_cyan_b")              # bool
    n("Cast", ["is_cyan_b"], "is_cyan", to=F32)
    init("ONE32", np.array(1.0, np.float32), np.float32)
    n("Sub", ["ONE32", "is_cyan"], "not_cyan")               # 1 if not cyan
    n("Mul", ["colf", "not_cyan"], "col_w")                  # colour value on coloured cells, else 0
    # double-MatMul histogram table[3,4] = colour at offset (dr,dc), learned from
    # the coloured box only (col_w = colour on coloured cells, 0 elsewhere):
    #   dr_oh_w[d,i] = (dr[i]==d) * col_w[i]        [3,N] fp16
    #   dc_oh[i,c]   = (dc[i]==c)                    [N,4] fp16
    #   table[d,c]   = dr_oh_w @ dc_oh = colour at (d,c)
    # fp16 exact: colours<=9, exactly one coloured cell per offset.
    n("Cast", ["col_w"], "col_w16", to=F16)                  # [1,1,10,10] fp16
    init("flatN", np.array([1, N], np.int64), np.int64)
    init("colN", np.array([N, 1], np.int64), np.int64)
    n("Reshape", ["col_w16", "flatN"], "col_w_row")          # [1,N] fp16
    n("Reshape", ["dr", "flatN"], "dr_row")                  # [1,N] fp16
    n("Reshape", ["dc", "colN"], "dc_col")                   # [N,1] fp16
    init("ar3col", np.array([[0.0], [1.0], [2.0]], np.float16), np.float16)  # [3,1]
    init("ar4row", np.array([[0.0, 1.0, 2.0, 3.0]], np.float16), np.float16)  # [1,4]
    n("Equal", ["dr_row", "ar3col"], "dr_oh_b")              # [3,N] bool
    n("Cast", ["dr_oh_b"], "dr_oh", to=F16)                  # [3,N] fp16
    n("Mul", ["dr_oh", "col_w_row"], "dr_oh_w")              # [3,N] fp16 (colour-wtd)
    n("Equal", ["dc_col", "ar4row"], "dc_oh_b")              # [N,4] bool
    n("Cast", ["dc_oh_b"], "dc_oh", to=F16)                  # [N,4] fp16
    n("MatMul", ["dr_oh_w", "dc_oh"], "tableColor")          # [3,4] fp16
    init("tc_shape", np.array([NK], np.int64), np.int64)
    n("Reshape", ["tableColor", "tc_shape"], "tableColor_flat")  # [12] fp16

    # --------------------------------------------------- output label plane
    # Lraw[10x10] = Gather(tableColor_flat, key) ; keep only on CYAN cells.
    n("Gather", ["tableColor_flat", "key_iflat"], "Lflat", axis=0)  # [N] fp16
    init("L2d_shape", np.array([1, 1, S, S], np.int64), np.int64)
    n("Reshape", ["Lflat", "L2d_shape"], "Lraw")             # [1,1,10,10] fp16
    n("Cast", ["is_cyan_b"], "is_cyan16", to=F16)            # [1,1,10,10] fp16
    n("Mul", ["Lraw", "is_cyan16"], "Lmask")                 # only cyan cells get colour
    n("Cast", ["Lmask"], "Lu8", to=U8)                       # [1,1,10,10] uint8

    # pad to 30x30 with sentinel 99 off-grid
    init("pad30", np.array([0, 0, 0, 0, 0, 0, 30 - S, 30 - S], np.int64), np.int64)
    init("SENT", np.array(99, np.uint8), np.uint8)
    n("Pad", ["Lu8", "pad30", "SENT"], "L30", mode="constant")  # [1,1,30,30] uint8
    arange_u8 = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("arange_u8", arange_u8, np.uint8)
    n("Equal", ["L30", "arange_u8"], "output")               # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task069", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
