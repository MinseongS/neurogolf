"""task141 (ARC-AGI 623ea044) — draw the full X (two 45-degree diagonals)
through a single coloured pixel, clipped to the square grid.

Rule (verified fresh): the INPUT is a size x size grid (odd size in 5..21,
anchored top-left in the 30x30 canvas) containing exactly ONE coloured pixel at
(row, col) of colour `color`; everything else is background.  The OUTPUT colours
`color` on every in-grid cell on either 45-degree diagonal through the pixel:
    r + c == row + col   OR   r - c == row - col .
Off-grid cells (all-zero in every input channel) stay all-zero.

Closed-form recovery (no detection floor) from the input only:
  * grid extent: an in-grid cell sets exactly one channel (bg sets ch0), an
    off-grid cell sets none.  rowany / colany ([1,1,30,1]/[1,1,1,30] booleans)
    from ReduceMax over channels mark the in-grid rows / cols (grid is a full
    top-left square so every in-grid row/col is occupied by bg).
  * pixel row/col: the foreground (non-ch0) row indicator =
    ReduceSum(input over all chan) - ReduceSum(ch0)  -> 1 at the pixel row, 0
    else (all 1-D, no 30x30 plane).  row = sum(rowfg * Rramp), col likewise.
  * colour: per-channel pixel COUNT cnt=ReduceSum(input,[2,3]); the single
    fg channel (k>=1) has cnt==1, others 0.  colour = sum_k k*(cnt_k>0) with a
    k-ramp whose entry 0 is 0 (so the always-present ch0 contributes nothing).
  * a = row+col, b = row-col.  Diagonal predicate is folded INTO the 1-D ramps
    so it is automatically clipped to the grid: Rclip = row index on in-grid
    rows, sentinel -100 off-grid; aCol = a-Cramp on in-grid cols, sentinel +100
    off-grid; bCol = b+Cramp on in-grid cols, sentinel +200 off-grid.
    onA = Equal(Rclip[1,1,30,1], aCol[1,1,1,30]) is true ONLY at in-grid cells on
    diagonal A (sentinels never coincide), onB likewise; diag = onA OR onB.
  * output = Where(diag[1,1,30,30], colour_onehot[1,10,1,1], input): on-diagonal
    in-grid cells get the colour, every other cell keeps `input` (in-grid bg ch0
    survives, off-grid stays all-zero, the original pixel sits on a diagonal so
    it is overwritten with the same colour).

Only three 30x30 intermediates ever materialise (onA, onB, diag — bool, 900 B
each); everything else is a tiny 1-D vector or scalar.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

N = 30  # canvas side


def build(task):
    inits, nodes = [], []

    def init(name, arr, dt):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dt), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    H = TensorProto.FLOAT16
    B = TensorProto.BOOL

    # ---- constants ----
    init("half", np.array(0.5, np.float32), np.float32)
    # 1-D ramps (fp16; values 0..29 exact in fp16)
    init("Rramp", np.arange(N, dtype=np.float16).reshape(1, 1, N, 1), np.float16)
    init("Cramp", np.arange(N, dtype=np.float16).reshape(1, 1, 1, N), np.float16)
    # diagonal sentinels (off-grid rows/cols can never match an in-grid index)
    init("sentR", np.array(-100.0, np.float16), np.float16)   # off-grid rows
    init("sentA", np.array(100.0, np.float16), np.float16)    # off-grid cols (A)
    init("sentB", np.array(200.0, np.float16), np.float16)    # off-grid cols (B)
    # colour one-hot comparator (channel arange, fp32 for the FREE Where output)
    init("kchan", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)

    # ---- pixel row / col via foreground (non-ch0) 1-D PROFILES from one Conv each ----
    # A no-pad Conv whose kernel spans the WHOLE row (1x30) / column (30x1) and
    # weights every NON-background channel 1.0 sums foreground cells per row/col:
    #   rowfg[r] = sum_{k>=1, c} input[k,r,c]  = 1 exactly at the pixel row, 0 else
    # The same Conv with ALL channels weighted 1.0 gives the total cells per
    # row/col (>0 <=> that row/col is in-grid).  This keeps everything at the 120 B
    # profile size -- no [1,10,30,1] (1200 B) per-channel intermediate.
    fgrow_w = np.ones((1, 10, 1, N), np.float32); fgrow_w[0, 0] = 0.0   # ch0 -> 0
    fgcol_w = np.ones((1, 10, N, 1), np.float32); fgcol_w[0, 0] = 0.0
    init("Wfgrow", fgrow_w, np.float32)
    init("Wfgcol", fgcol_w, np.float32)
    n("Conv", ["input", "Wfgrow"], "rowfg32")    # [1,1,30,1] fg count per row
    n("Conv", ["input", "Wfgcol"], "colfg32")    # [1,1,1,30] fg count per col
    # in-grid indicators: total cells per row/col (>0 <=> occupied); plain reduce
    n("ReduceSum", ["input"], "rowall", axes=[1, 3], keepdims=1)    # [1,1,30,1]
    n("ReduceSum", ["input"], "colall", axes=[1, 2], keepdims=1)    # [1,1,1,30]
    n("Greater", ["rowall", "half"], "rowany")                      # bool [1,1,30,1]
    n("Greater", ["colall", "half"], "colany")                      # bool [1,1,1,30]
    # row = sum(rowfg * Rramp), col = sum(colfg * Cramp), all fp16 (exact ints)
    n("Cast", ["rowfg32"], "rowfg", to=H)
    n("Cast", ["colfg32"], "colfg", to=H)
    n("Mul", ["rowfg", "Rramp"], "rowwt")
    n("Mul", ["colfg", "Cramp"], "colwt")
    n("ReduceSum", ["rowwt"], "rowidx", axes=[2, 3], keepdims=1)     # [1,1,1,1] fp16
    n("ReduceSum", ["colwt"], "colidx", axes=[2, 3], keepdims=1)
    n("Add", ["rowidx", "colidx"], "aval")     # a = row+col   (scalar fp16)
    n("Sub", ["rowidx", "colidx"], "bval")     # b = row-col

    # ---- colour scalar from per-channel pixel counts ----
    n("ReduceSum", ["input"], "cnt", axes=[2, 3], keepdims=1)        # [1,10,1,1] fp32
    n("Greater", ["cnt", "half"], "cntpos")                          # bool [1,10,1,1]
    n("Cast", ["cntpos"], "cntposf", to=F)
    n("Mul", ["cntposf", "kchan"], "kcontrib")                       # ch0 weight 0
    n("ReduceSum", ["kcontrib"], "colorv", axes=[1], keepdims=1)     # [1,1,1,1] fp32 colour

    # ---- diagonal predicate folded into 1-D ramps (auto grid-clipped) ----
    # in-grid row index, sentinel off-grid
    n("Where", ["rowany", "Rramp", "sentR"], "Rclip")               # [1,1,30,1] fp16
    # diagonal A:  r + c == a  ->  r == a - c
    n("Sub", ["aval", "Cramp"], "aCol_raw")                         # [1,1,1,30]
    n("Where", ["colany", "aCol_raw", "sentA"], "aCol")             # off-grid cols -> +100
    n("Equal", ["Rclip", "aCol"], "onA")                            # [1,1,30,30] bool
    # diagonal B:  r - c == b  ->  r == b + c
    n("Add", ["bval", "Cramp"], "bCol_raw")
    n("Where", ["colany", "bCol_raw", "sentB"], "bCol")             # off-grid cols -> +200
    n("Equal", ["Rclip", "bCol"], "onB")
    n("Or", ["onA", "onB"], "diag")                                 # [1,1,30,30] bool

    # ---- colour one-hot [1,10,1,1] (fp32 1.0 in the colour channel) ----
    n("Equal", ["kchan", "colorv"], "color_ohb")                    # [1,10,1,1] bool
    n("Cast", ["color_ohb"], "color_oh", to=F)                      # [1,10,1,1] fp32 {0,1}
    # ---- route into the FREE Where output ----
    # on-diagonal in-grid cells -> colour one-hot; every other cell -> input
    # (in-grid bg ch0 survives, off-grid stays all-zero, the pixel sits on a
    # diagonal so it is overwritten with its own colour).
    n("Where", ["diag", "color_oh", "input"], "output")             # [1,10,30,30] fp32 FREE

    x = helper.make_tensor_value_info("input", F, [1, 10, N, N])
    y = helper.make_tensor_value_info("output", F, [1, 10, N, N])
    graph = helper.make_graph(nodes, "task141", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
