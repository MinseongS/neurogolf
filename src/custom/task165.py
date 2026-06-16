"""task165 (ARC-AGI 6d58a25d) — "kite drips colour to the bottom".

Rule (from the generator, verified 0/1499 fresh on the reconstruction below):
  The grid (active 20x20) holds scattered single pixels in colour `color` and ONE
  fixed 10-cell "kite" shape in colour `kite`, anchored at (row,col) with row in
  [1,10], col in [5,14].  The kite cells (relative to its top cell (row,col)) are:
      (0,0),(1,-1),(1,0),(1,1),(2,-2),(2,-1),(2,1),(2,2),(3,-3),(3,3)
  For each kite column (col-3 .. col+3): if a `color` pixel lies BELOW the topmost
  kite cell of that column, the column "drips": every cell from just below the
  LOWEST kite cell of that column down to the bottom row (19) is painted `color`.

  Per-column kite-row offsets (dc = column - col):
    toprel (topmost kite row, rel): -3:3 -2:2 -1:1 0:0 +1:1 +2:2 +3:3
    lowrel (lowest  kite row, rel): -3:3 -2:2 -1:2 0:1 +1:2 +2:2 +3:3
  trigger(dc) = exists `color` pixel at column col+dc with r > row+toprel(dc).
  fill(dc)    = if trigger, rows [row+lowrel(dc)+1 .. 19] at column col+dc.

ONNX pipeline (opset 11, no flood-fill).  Score: mem=20809 params=254 -> 15.04.
  1. colf = sum_k k*input_k  (1x1 Conv) -> [1,1,30,30] fp32 value plane (the only
     3600B plane; irreducible since a 1x1 Conv over the 30x30 input is 30x30).
  2. Kite anchor by Cauchy-Schwarz equality.  Slice colf to the conv-relevant
     region [0:17,0:18] FIRST (anchor row in [1,10], col in [5,14]); S1=Conv(colfA,K),
     S2=Conv(colfA^2,K) over the 4x7 kite kernel K, valid -> [1,1,14,12].  Anchor is
     the UNIQUE cell where all 10 pattern cells are equal & nonzero:
     10*S2 == S1^2 AND S1>0  (verified unique 0/2000).  row,col = ReduceMax(iswin*
     rowramp / colramp+3); kite value = max(iswin*S1)/10.
  3. color = the present non-bg channel index != kite (per-channel ReduceMax).
  4. colormask = (colf20 == color).  lastcolor[1,1,1,20] = max row with a color
     pixel per column.  For each dc in -3..3 compute trigger (lastcolor[col+dc] >
     row+toprel) and write startrow = row+lowrel+1 into that column (else BIG).
     fill30 = (rowramp30 >= startrow30) AND (row < 20).
  5. output = Where(fill30, color_onehot[1,10,1,1] float, input)  (10-ch -> FREE out).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
I64 = TensorProto.INT64
B = TensorProto.BOOL

CELLS = [(0, 0), (1, -1), (1, 0), (1, 1), (2, -2), (2, -1),
         (2, 1), (2, 2), (3, -3), (3, 3)]
TOPREL = {-3: 3, -2: 2, -1: 1, 0: 0, 1: 1, 2: 2, 3: 3}
LOWREL = {-3: 3, -2: 2, -1: 2, 0: 1, 1: 2, 2: 2, 3: 3}
SIZE = 20  # active canvas


def build(task):
    inits, nodes, vinfo = [], [], []

    npmap = {F32: np.float32, U8: np.uint8, I64: np.int64, B: np.bool_}

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=npmap[dtype]), name))
        return name

    def vi(name, dtype, shape):
        vinfo.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    def n(op, ins, out, dtype=None, shape=None, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        if dtype is not None:
            vi(out, dtype, shape)
        return out

    # ---- 1. colf = sum_k k*input_k  (value plane) ----------------------------
    w_colf = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("w_colf", w_colf, F32)
    n("Conv", ["input", "w_colf"], "colf", F32, [1, 1, 30, 30])  # [1,1,30,30]

    # ---- 2. kite anchor via Cauchy-Schwarz ----------------------------------
    K = np.zeros((1, 1, 4, 7), np.float32)
    for dr, dc in CELLS:
        K[0, 0, dr, dc + 3] = 1.0
    init("K", K, F32)
    # The anchor lives in row in [1,10], valid-col index j in [2,11].  So the Conv
    # only needs colf rows 0..16, cols 0..17 (window rows i..i+3<=13+3, cols j..j+6).
    # Slice colf to that region FIRST -> conv output + colf2 shrink dramatically.
    CH, CW = 17, 18
    init("cst", np.array([0, 0, 0, 0], np.int64), I64)
    init("cen", np.array([1, 1, CH, CW], np.int64), I64)
    init("cax", np.array([0, 1, 2, 3], np.int64), I64)
    n("Slice", ["colf", "cst", "cen", "cax"], "colfA", F32, [1, 1, CH, CW])
    n("Mul", ["colfA", "colfA"], "colf2", F32, [1, 1, CH, CW])
    RH, RW = CH - 3, CW - 6  # valid-corr output size = 14 x 12
    n("Conv", ["colfA", "K"], "S1", F32, [1, 1, RH, RW])          # valid corr
    n("Conv", ["colf2", "K"], "S2", F32, [1, 1, RH, RW])
    init("ten", np.array(10.0, np.float32), F32)
    n("Mul", ["S2", "ten"], "tenS2", F32, [1, 1, RH, RW])
    n("Mul", ["S1", "S1"], "S1sq", F32, [1, 1, RH, RW])
    n("Sub", ["tenS2", "S1sq"], "diff", F32, [1, 1, RH, RW])
    # iswin = (diff == 0) AND (S1 > 0)
    init("zero", np.array(0.0, np.float32), F32)
    n("Equal", ["diff", "zero"], "eqz", B, [1, 1, RH, RW])
    n("Greater", ["S1", "zero"], "s1pos", B, [1, 1, RH, RW])
    n("And", ["eqz", "s1pos"], "iswin", B, [1, 1, RH, RW])
    n("Cast", ["iswin"], "iswinf", F32, [1, 1, RH, RW], to=F32)

    # row = max over (iswin * rowramp);  col = max over (iswin * (colramp+3))
    rr = np.arange(RH, dtype=np.float32).reshape(1, 1, RH, 1)
    cc = (np.arange(RW, dtype=np.float32) + 3.0).reshape(1, 1, 1, RW)
    init("rr27", rr, F32)
    init("cc24", cc, F32)
    n("Mul", ["iswinf", "rr27"], "roww", F32, [1, 1, RH, RW])
    n("ReduceMax", ["roww"], "row_s", F32, [1, 1, 1, 1], axes=[2, 3], keepdims=1)
    n("Mul", ["iswinf", "cc24"], "colw", F32, [1, 1, RH, RW])
    n("ReduceMax", ["colw"], "col_s", F32, [1, 1, 1, 1], axes=[2, 3], keepdims=1)
    # kite value = max over (iswin * S1) / 10
    n("Mul", ["iswinf", "S1"], "kw", F32, [1, 1, RH, RW])
    n("ReduceMax", ["kw"], "kw_s", F32, [1, 1, 1, 1], axes=[2, 3], keepdims=1)
    init("tenth", np.array(0.1, np.float32), F32)
    n("Mul", ["kw_s", "tenth"], "kite_s", F32, [1, 1, 1, 1])

    # ---- 3. color = present non-bg channel index != kite --------------------
    # per-channel presence [1,10,1,1]; channel ramp; pick max index that is
    # present and != kite.
    n("ReduceMax", ["input"], "pres", F32, [1, 10, 1, 1], axes=[2, 3], keepdims=1)
    init("half", np.array(0.5, np.float32), F32)
    n("Greater", ["pres", "half"], "presb", B, [1, 10, 1, 1])
    chramp = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("chramp", chramp, F32)
    # is this channel the kite? (chramp == kite_s) broadcast
    n("Equal", ["chramp", "kite_s"], "iskite", B, [1, 10, 1, 1])
    n("Not", ["iskite"], "notkite", B, [1, 10, 1, 1])
    # ch0 already excluded because its value is 0 and color>=1; but ensure index
    # selection: candidate = present AND notkite -> value = chramp else -1
    n("And", ["presb", "notkite"], "cand", B, [1, 10, 1, 1])
    init("neg1", np.array(-1.0, np.float32), F32)
    n("Where", ["cand", "chramp", "neg1"], "candv", F32, [1, 10, 1, 1])
    n("ReduceMax", ["candv"], "color_s", F32, [1, 1, 1, 1], axes=[1], keepdims=1)

    # ---- 4. fill mask --------------------------------------------------------
    # work on the active 20x20.  colormask20 = (colf20 == color)
    init("st20", np.array([0, 0, 0, 0], np.int64), I64)
    init("en20", np.array([1, 1, 20, 20], np.int64), I64)
    init("ax20", np.array([0, 1, 2, 3], np.int64), I64)
    n("Slice", ["colf", "st20", "en20", "ax20"], "colf20", F32, [1, 1, 20, 20])
    n("Equal", ["colf20", "color_s"], "colormask", B, [1, 1, 20, 20])

    rr20 = np.arange(SIZE, dtype=np.float32).reshape(1, 1, SIZE, 1)
    cc20 = np.arange(SIZE, dtype=np.float32).reshape(1, 1, 1, SIZE)
    init("rr20", rr20, F32)
    init("cc20", cc20, F32)
    BIG = 99.0
    init("BIG", np.array(BIG, np.float32), F32)

    # lastcolor[1,1,1,20] = max row index having a color pixel, per column
    # (=-1 where the column has none).  colormaskf * rowramp, ReduceMax over rows.
    # masked = where(colormask, rowramp, -1); ReduceMax over rows -> lastrow
    init("neg1b", np.array(-1.0, np.float32), F32)
    n("Where", ["colormask", "rr20", "neg1b"], "rmasked", F32, [1, 1, SIZE, SIZE])
    n("ReduceMax", ["rmasked"], "lastcolor", F32, [1, 1, 1, SIZE], axes=[2],
      keepdims=1)

    # build startrow[1,1,1,20] = fill-start row at each kite column (else BIG)
    startrow = "startrow_init"
    init("startrow_init", np.full((1, 1, 1, SIZE), BIG, np.float32), F32)
    for dc in range(-3, 4):
        tag = f"d{dc + 3}"
        init(f"dc_{tag}", np.array(float(dc), np.float32), F32)
        n("Add", ["col_s", f"dc_{tag}"], f"c0_{tag}", F32, [1, 1, 1, 1])
        n("Equal", ["cc20", f"c0_{tag}"], f"coleq_{tag}", B, [1, 1, 1, SIZE])
        # trigger: lastcolor at this column > row + toprel(dc)
        n("Where", [f"coleq_{tag}", "lastcolor", "neg1b"], f"lc_{tag}", F32,
          [1, 1, 1, SIZE])
        n("ReduceMax", [f"lc_{tag}"], f"lcmax_{tag}", F32, [1, 1, 1, 1], axes=[3],
          keepdims=1)
        init(f"tr_{tag}", np.array(float(TOPREL[dc]), np.float32), F32)
        n("Add", ["row_s", f"tr_{tag}"], f"trow_{tag}", F32, [1, 1, 1, 1])
        n("Greater", [f"lcmax_{tag}", f"trow_{tag}"], f"trigb_{tag}", B,
          [1, 1, 1, 1])
        # startval = row + lowrel(dc) + 1
        init(f"lr_{tag}", np.array(float(LOWREL[dc] + 1), np.float32), F32)
        n("Add", ["row_s", f"lr_{tag}"], f"lrow_{tag}", F32, [1, 1, 1, 1])
        # write start = trig ? lrow : BIG, but ONLY into this column.
        # set = coleq AND trig
        n("And", [f"coleq_{tag}", f"trigb_{tag}"], f"setcol_{tag}", B,
          [1, 1, 1, SIZE])
        n("Where", [f"setcol_{tag}", f"lrow_{tag}", startrow], f"startrow_{tag}",
          F32, [1, 1, 1, SIZE])
        startrow = f"startrow_{tag}"

    # pad startrow[1,1,1,20] -> [1,1,1,30] with BIG (cols 20-29 never fill).
    init("padcol", np.array([0, 0, 0, 0, 0, 0, 0, 30 - SIZE], np.int64), I64)
    n("Pad", [startrow, "padcol", "BIG"], "startrow30", F32, [1, 1, 1, 30],
      mode="constant")
    # fill30 = (rr30 >= startrow30) AND (rr30 < 20)
    rr30 = np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1)
    init("rr30", rr30, F32)
    rowvalid = (np.arange(30) < SIZE).reshape(1, 1, 30, 1)
    init("rowvalid", rowvalid, B)
    n("Less", ["rr30", "startrow30"], "filllt", B, [1, 1, 30, 30])
    n("Not", ["filllt"], "fillge", B, [1, 1, 30, 30])
    n("And", ["fillge", "rowvalid"], "fill30", B, [1, 1, 30, 30])

    # ---- 5. output = Where(fill30, color_onehot, input) ----------------------
    # color_onehot[1,10,1,1] float = (chramp == color); Where keeps input float.
    n("Equal", ["chramp", "color_s"], "color_ohb", B, [1, 10, 1, 1])
    n("Cast", ["color_ohb"], "color_oh", F32, [1, 10, 1, 1], to=F32)
    n("Where", ["fill30", "color_oh", "input"], "output")

    graph = helper.make_graph(
        nodes, "task165", [helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])],
        inits, value_info=vinfo)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
