"""Task 090 (ARC-AGI 3eda0437): paint the unique maximal empty rectangle pink.

Rule (from the generator):
  * Grid is height (2..5) rows x width (20..30) cols.  Cells are either black(0)
    (= empty) or static (blue 1 / gray 5).  A small cutout makes one all-black
    axis-aligned rectangle, and the generator guarantees there is EXACTLY ONE
    maximal-AREA all-empty (all-black) axis-aligned rectangle in the grid.
  * Output = input with that unique maximal empty rectangle painted pink(6).

Key facts exploited:
  * height <= 5, so the in-grid band is rows 0..4.
  * Channel 0 (black) is 1 ONLY for in-grid empty cells, 0 for off-grid cells, so a
    band-sum equal to band height marks "all rows empty AND in-grid" -- grid width
    is never recovered and empty runs naturally stop at the grid edge.
  * Work per ROW-BAND (r1<=r2, 10 pairs).  band-empty[c] over rows r1..r2.  The run
    of empties STARTING at column c = (next occupied/off-grid column at or after c)
    minus c, recovered with ONE suffix-min (negated MaxPool) -- no width
    enumeration.  areamap[band,c] = bandheight * runlen; global max M is unique; the
    single winning cell gives band (=> top-row, height) and column (=> left), and
    width = M / height.  The pink cover is a separable ramp comparison routed
    straight into the FREE output via one Where.  No [1,10,30,30] intermediate, no
    input cast, no ConvTranspose.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

H = 5          # in-grid row cap (height <= 5)
W = 30         # full width
BIG = 1000.0   # large sentinel for suffix-min over occupied columns

BANDS = [(r1, r2) for r2 in range(H) for r1 in range(r2)]   # r1<r2, 10 bands
NB = len(BANDS)
BAND_H = [r2 - r1 + 1 for (r1, r2) in BANDS]


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, outs, **attrs):
        if isinstance(outs, str):
            outs = [outs]
        nodes.append(helper.make_node(op, ins, outs, **attrs))
        return outs[0]

    # ---- slice channel 0 over the 5-row band ----
    init("e_st", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("e_en", np.array([1, 1, H, W], np.int64), np.int64)
    init("e_ax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "e_st", "e_en", "e_ax"], "e0")        # [1,1,5,30] f32
    n("Cast", ["e0"], "e", to=TensorProto.FLOAT16)             # [1,1,5,30] f16

    # ---- per-band empty mask: band-sum conv then == band height ----
    bker = np.zeros((NB, 1, H, 1), np.float16)
    bh_arr = np.zeros((1, NB, 1, 1), np.float16)
    for b, (r1, r2) in enumerate(BANDS):
        bker[b, 0, r1:r2 + 1, 0] = 1.0
        bh_arr[0, b, 0, 0] = BAND_H[b]
    init("bker", bker, np.float16)
    init("bh", bh_arr, np.float16)
    n("Conv", ["e", "bker"], "bandsum")            # [1,NB,1,W]
    n("Equal", ["bandsum", "bh"], "bemptyB")       # bool [1,NB,1,W]
    n("Cast", ["bemptyB"], "bempty", to=TensorProto.FLOAT16)   # {0,1}

    # ---- run length starting at each column via suffix-min of next-occupied index ----
    # occidx[c] = c if column is occupied/off-grid (band-empty==0) else c+BIG.
    colramp = np.arange(W, dtype=np.float16).reshape(1, 1, 1, W)
    init("colramp", colramp, np.float16)
    init("BIG", np.array(BIG, np.float16), np.float16)
    bigterm = n("Mul", ["bempty", "BIG"], "bigterm")           # BIG where empty
    occidx = n("Add", ["colramp", bigterm], "occidx")          # [1,NB,1,W]
    # suffix-min over j>=c : negate, MaxPool right window, negate back.
    negocc = n("Neg", [occidx], "negocc")
    # MaxPool kernel 1xW, pad right W-1 -> position c covers [c, c+W-1] (suffix).
    sufmax = n("MaxPool", [negocc], "sufmax",
               kernel_shape=[1, W], pads=[0, 0, 0, W - 1])      # [1,NB,1,W]
    nextocc0 = n("Neg", [sufmax], "nextocc0")                  # min next-occupied index
    # cap at W: a run reaching the right grid edge (full-width grid, no off-grid
    # terminator) has its boundary at column W, so next-occupied is at most W.
    init("Wc", np.array(float(W), np.float16), np.float16)
    nextocc = n("Min", [nextocc0, "Wc"], "nextocc")
    runlen0 = n("Sub", [nextocc, "colramp"], "runlen0")       # [1,NB,1,W] >=0
    # rectangles must have width >= 2 (generator uses c1 < c2): drop runlen 0/1.
    init("onehalf", np.array(1.5, np.float16), np.float16)
    init("zero16", np.array(0.0, np.float16), np.float16)
    w2B = n("Greater", [runlen0, "onehalf"], "w2B")          # runlen >= 2
    runlen = n("Where", [w2B, runlen0, "zero16"], "runlen")
    areamap = n("Mul", [runlen, "bh"], "areamap")            # bandheight * runlen

    Mname = n("ReduceMax", [areamap], "M", keepdims=1)       # [1,1,1,1]

    # ---- winning cell (band, col): areamap == M (unique) ----
    winB = n("Equal", [areamap, Mname], "winB")
    win = n("Cast", [winB], "win", to=TensorProto.FLOAT16)

    bandtop = np.array([r1 for (r1, r2) in BANDS], np.float16).reshape(1, NB, 1, 1)
    init("bandtop", bandtop, np.float16)
    tr = n("ReduceMax", [n("Mul", [win, "bandtop"], "wt")], "tr", keepdims=1)   # top row
    tc = n("ReduceMax", [n("Mul", [win, "colramp"], "wc")], "tc", keepdims=1)   # left col
    rhv = n("ReduceMax", [n("Mul", [win, "bh"], "wbh")], "rhv", keepdims=1)     # height
    rwv = n("Div", [Mname, rhv], "rwv")                                          # width = M/height
    br = n("Add", [tr, rhv], "br")
    bc = n("Add", [tc, rwv], "bc")

    # ---- cover = (R>=tr)&(R<br)&(C>=tc)&(C<bc) over 30x30 (off-band auto-excluded) ----
    fullrow = np.arange(W, dtype=np.float16).reshape(1, 1, W, 1)
    fullcol = np.arange(W, dtype=np.float16).reshape(1, 1, 1, W)
    init("fullrow", fullrow, np.float16)
    init("fullcol", fullcol, np.float16)
    rge = n("Not", [n("Less", ["fullrow", tr], "rlt")], "rge")
    rltbr = n("Less", ["fullrow", br], "rltbr")
    cge = n("Not", [n("Less", ["fullcol", tc], "clt")], "cge")
    cltbc = n("Less", ["fullcol", bc], "cltbc")
    rband = n("And", [rge, rltbr], "rband")
    cband = n("And", [cge, cltbc], "cband")
    cover = n("And", [rband, cband], "cover")      # [1,1,30,30] bool

    # ---- route to FREE output ----
    pink = np.zeros((1, 10, 1, 1), np.float32)
    pink[0, 6, 0, 0] = 1.0
    init("pink", pink, np.float32)
    n("Where", [cover, "pink", "input"], "output")

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task090", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
