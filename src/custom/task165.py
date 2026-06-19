"""task165 (ARC-AGI 6d58a25d) — "kite drips colour to the bottom".

Rule (from the generator, verified fresh on the reconstruction below):
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

ONNX pipeline (opset 11, no flood-fill).  RE-GOLF of the 15.04 net:
  - Slice the INPUT to the active 20x20 FIRST so the colour-index Conv emits a
    [1,1,20,20] fp32 entry plane (1600B) instead of the full 30x30 (3600B).
  - Cast that entry plane to fp16 ONCE; every downstream full/working plane
    (ring-conv inputs colf2, the S1/S2 ring convs, colormask source, lastcolor
    source) runs in fp16 -> counts at half (fp16 Conv keeps fp16 under
    ORT_DISABLE_ALL; integers <=8100 are exact in fp16 at the even multiples the
    Cauchy-Schwarz equality lands on).
  - Collapse the three 30x30 bool fill planes (Less/Not/And) into ONE: fold the
    rows<20 validity into the row ramp (off-grid rows -> -1 sentinel) and compare
    with Greater(rr30, startrow-1) == (rr30 >= startrow), a single bool plane.
  - 10-ch expansion routed into the FREE Where output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
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

    npmap = {F32: np.float32, F16: np.float16, U8: np.uint8,
             I64: np.int64, B: np.bool_}

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

    # ---- 1. colf = sum_k k*input_k  (value plane) ---------------------------
    # Conv over the FREE 10-ch input (slicing the 10-ch input would materialise a
    # 16000B plane); the 1-ch colf is the one forced fp32 entry plane (3600B).
    w_colf = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("w_colf", w_colf, F32)
    n("Conv", ["input", "w_colf"], "colf", F32, [1, 1, 30, 30])

    # ---- 2. kite anchor via Cauchy-Schwarz (MUST be fp32) -------------------
    # 10*S2 == S1^2 reaches 8100, where fp16 has step 4 -> the equality test
    # false-fires on most windows.  The ring-conv block is the forced fp32
    # intermediate here; keep it fp32.
    K = np.zeros((1, 1, 4, 7), np.float32)
    for dr, dc in CELLS:
        K[0, 0, dr, dc + 3] = 1.0
    init("K", K, F32)
    # Anchor row i in [1,10] (kernel top cell), col j+3 in [5,14] -> j in [2,11].
    # Windows touch colf rows 0..13 and cols 0..17, so slice colf to [0:14,0:18].
    CH, CW = 14, 18
    init("cst", np.array([0, 0, 0, 0], np.int64), I64)
    init("cen", np.array([1, 1, CH, CW], np.int64), I64)
    init("cax", np.array([0, 1, 2, 3], np.int64), I64)
    n("Slice", ["colf", "cst", "cen", "cax"], "colfA", F32, [1, 1, CH, CW])
    n("Mul", ["colfA", "colfA"], "colf2", F32, [1, 1, CH, CW])
    RH, RW = CH - 3, CW - 6  # valid-corr output size = 11 x 12
    n("Conv", ["colfA", "K"], "S1", F32, [1, 1, RH, RW])          # valid corr
    n("Conv", ["colf2", "K"], "S2", F32, [1, 1, RH, RW])
    init("ten", np.array(10.0, np.float32), F32)
    n("Mul", ["S2", "ten"], "tenS2", F32, [1, 1, RH, RW])
    n("Mul", ["S1", "S1"], "S1sq", F32, [1, 1, RH, RW])
    # iswin = (10*S2 == S1^2) AND (S1 > 0).  Compare directly (no diff plane).
    init("zero", np.array(0.0, np.float32), F32)
    n("Equal", ["tenS2", "S1sq"], "eqz", B, [1, 1, RH, RW])
    n("Greater", ["S1", "zero"], "s1pos", B, [1, 1, RH, RW])
    n("And", ["eqz", "s1pos"], "iswin", B, [1, 1, RH, RW])
    # cast the winner mask to fp16 for the (small, exact) index/value reductions.
    n("Cast", ["iswin"], "iswinf", F16, [1, 1, RH, RW], to=F16)
    n("Cast", ["S1"], "S1h", F16, [1, 1, RH, RW], to=F16)

    # row = max over (iswin * rowramp);  col = max over (iswin * (colramp+3))
    rr = np.arange(RH, dtype=np.float16).reshape(1, 1, RH, 1)
    cc = (np.arange(RW, dtype=np.float16) + 3.0).reshape(1, 1, 1, RW)
    init("rr27", rr, F16)
    init("cc24", cc, F16)
    n("Mul", ["iswinf", "rr27"], "roww", F16, [1, 1, RH, RW])
    n("ReduceMax", ["roww"], "row_s", F16, [1, 1, 1, 1], axes=[2, 3], keepdims=1)
    n("Mul", ["iswinf", "cc24"], "colw", F16, [1, 1, RH, RW])
    n("ReduceMax", ["colw"], "col_s", F16, [1, 1, 1, 1], axes=[2, 3], keepdims=1)
    # kw_s = max over (iswin * S1) = 10 * kite_colour (S1<=90, fp16 exact).
    # Avoid the /10 (0.1 is not fp16-exact, shifts 9->8.998); compare 10*chan.
    n("Mul", ["iswinf", "S1h"], "kw", F16, [1, 1, RH, RW])
    n("ReduceMax", ["kw"], "kw_s", F16, [1, 1, 1, 1], axes=[2, 3], keepdims=1)

    # colf cropped to the active 20x20 (fp32) for the fill block; no fp16 copy
    # needed -- colormask reads this fp32 slice directly against an fp32 colour.
    init("ist", np.array([0, 0, 0, 0], np.int64), I64)
    init("ien", np.array([1, 1, SIZE, SIZE], np.int64), I64)
    init("iax", np.array([0, 1, 2, 3], np.int64), I64)
    n("Slice", ["colf", "ist", "ien", "iax"], "colf20", F32, [1, 1, SIZE, SIZE])

    # ---- 3. color = present non-bg channel index != kite --------------------
    n("ReduceMax", ["input"], "pres", F32, [1, 10, 1, 1], axes=[2, 3], keepdims=1)
    init("half", np.array(0.5, np.float32), F32)
    n("Greater", ["pres", "half"], "presb", B, [1, 10, 1, 1])
    # is this channel the kite? 10*chan == kw_s (=10*kite), all fp16-exact ints
    chramp10 = (np.arange(10, dtype=np.float16) * 10.0).reshape(1, 10, 1, 1)
    init("chramp10", chramp10, F16)
    n("Equal", ["chramp10", "kw_s"], "iskite", B, [1, 10, 1, 1])
    n("Not", ["iskite"], "notkite", B, [1, 10, 1, 1])
    n("And", ["presb", "notkite"], "cand", B, [1, 10, 1, 1])
    chramp32 = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("chramp32", chramp32, F32)
    init("neg1_32", np.array(-1.0, np.float32), F32)
    n("Where", ["cand", "chramp32", "neg1_32"], "candv", F32, [1, 10, 1, 1])
    n("ReduceMax", ["candv"], "color_s", F32, [1, 1, 1, 1], axes=[1], keepdims=1)

    # ---- 4. fill mask (fp16 working planes on the 20x20 grid) ---------------
    # colormask = (colf20 == color), both fp32.
    n("Equal", ["colf20", "color_s"], "colormask", B, [1, 1, SIZE, SIZE])

    rr20 = np.arange(SIZE, dtype=np.float16).reshape(1, 1, SIZE, 1)
    cc20 = np.arange(SIZE, dtype=np.float16).reshape(1, 1, 1, SIZE)
    init("rr20", rr20, F16)
    init("cc20", cc20, F16)
    BIG = 99.0

    # lastcolor[1,1,1,20] = max row index having a color pixel, per column (-1 if none)
    init("neg1b", np.array(-1.0, np.float16), F16)
    n("Where", ["colormask", "rr20", "neg1b"], "rmasked", F16, [1, 1, SIZE, SIZE])
    n("ReduceMax", ["rmasked"], "lastcolor", F16, [1, 1, 1, SIZE], axes=[2],
      keepdims=1)

    # startrow[1,1,1,20] = fill-start row at each kite column (else BIG)
    startrow = "startrow_init"
    init("startrow_init", np.full((1, 1, 1, SIZE), BIG, np.float16), F16)
    for dc in range(-3, 4):
        tag = f"d{dc + 3}"
        init(f"dc_{tag}", np.array(float(dc), np.float16), F16)
        n("Add", ["col_s", f"dc_{tag}"], f"c0_{tag}", F16, [1, 1, 1, 1])
        n("Equal", ["cc20", f"c0_{tag}"], f"coleq_{tag}", B, [1, 1, 1, SIZE])
        # trigger: lastcolor at this column > row + toprel(dc)
        n("Where", [f"coleq_{tag}", "lastcolor", "neg1b"], f"lc_{tag}", F16,
          [1, 1, 1, SIZE])
        n("ReduceMax", [f"lc_{tag}"], f"lcmax_{tag}", F16, [1, 1, 1, 1], axes=[3],
          keepdims=1)
        init(f"tr_{tag}", np.array(float(TOPREL[dc]), np.float16), F16)
        n("Add", ["row_s", f"tr_{tag}"], f"trow_{tag}", F16, [1, 1, 1, 1])
        n("Greater", [f"lcmax_{tag}", f"trow_{tag}"], f"trigb_{tag}", B,
          [1, 1, 1, 1])
        # startval = row + lowrel(dc) + 1
        init(f"lr_{tag}", np.array(float(LOWREL[dc] + 1), np.float16), F16)
        n("Add", ["row_s", f"lr_{tag}"], f"lrow_{tag}", F16, [1, 1, 1, 1])
        n("And", [f"coleq_{tag}", f"trigb_{tag}"], f"setcol_{tag}", B,
          [1, 1, 1, SIZE])
        n("Where", [f"setcol_{tag}", f"lrow_{tag}", startrow], f"startrow_{tag}",
          F16, [1, 1, 1, SIZE])
        startrow = f"startrow_{tag}"

    # startrow - 1 so a single Greater gives (rr >= startrow); pad to 30 cols
    # with BIG (cols 20-29 never fill).  Compare in fp32 (Pad bridge).
    init("onef16", np.array(1.0, np.float16), F16)
    n("Sub", [startrow, "onef16"], "startm1", F16, [1, 1, 1, SIZE])
    init("padcol", np.array([0, 0, 0, 0, 0, 0, 0, 30 - SIZE], np.int64), I64)
    init("BIG16", np.array(BIG, np.float16), F16)
    n("Pad", ["startm1", "padcol", "BIG16"], "startrow30", F16, [1, 1, 1, 30],
      mode="constant")
    # row ramp with off-grid rows (>=20) folded to -1 so they never fill.
    rr30v = np.arange(30, dtype=np.float16)
    rr30v[SIZE:] = -1.0
    rr30 = rr30v.reshape(1, 1, 30, 1)
    init("rr30", rr30, F16)
    # fill30 = rr30 > (startrow - 1) == (rr30 >= startrow) ; one bool plane.
    n("Greater", ["rr30", "startrow30"], "fill30", B, [1, 1, 30, 30])

    # ---- 5. output = Where(fill30, color_onehot, input) ----------------------
    n("Equal", ["chramp32", "color_s"], "color_ohb", B, [1, 10, 1, 1])
    n("Cast", ["color_ohb"], "color_oh", F32, [1, 10, 1, 1], to=F32)
    n("Where", ["fill30", "color_oh", "input"], "output")

    graph = helper.make_graph(
        nodes, "task165", [helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])],
        inits, value_info=vinfo)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
