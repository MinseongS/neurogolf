"""Task 394 (f9012d9b): reconstruct the bitten-out window of a periodic grid.

Rule (from ARC-GEN generator): a `size x size` grid (size in 4..7) is tiled by
a `m x m` colour pattern (m = minisize = 2 if size < 7 else 3) drawn from 2
random colours: `grid[r][c] = colors[(r%m)*m + (c%m)]`.  A `bs x bs` "bite"
(bs = bitesize in 1..3, always <= m) at (row,col) is blacked out (set to 0) in
the INPUT.  The OUTPUT is that `bs x bs` window holding the ORIGINAL pattern
colours removed.  So output cell (r,c) = pattern value at phase
`((row+r)%m, (col+c)%m)`.

Construction — everything runs on a fixed 7x7 crop (size <= 7 always), so the
largest tensor is the [1,10,7,7] crop (1960 B); NO 30x30 plane is built.

1.  `xc = input[:, :, :7, :7]`.  Per-row/col reductions:
    * `s3 = ReduceSum(xc, [3])` ([1,10,7,1]), `s2 = ReduceSum(xc, [2])`.
    * in-grid count per row = `ReduceSum(s3,[1])` (each in-grid cell, colour or
      black, contributes 1); >0 => in-grid.  `size` = max in-grid index + 1,
      `m = 3 if size==7 else 2`.
    * colored count per row = in-grid count - (channel-0/black count) = a hole
      row loses exactly `bs` colours, so `0 < colored < size` flags hole rows;
      `bs` = #hole rows, `row`/`col` = first hole row/col.

2.  Pattern colour per phase WITHOUT a value plane: `Pr[pr,r] = (r%m==pr)`
    ([3,7]); `sp = Pr@xc` ([1,10,3,7]) then `@Pc` ([1,10,3,3]) counts in-grid
    cells of phase (pr,pc) and colour k.  `pat = (Sum_k k*cnt)/(Sum_{k>=1}cnt)`
    (den clamped to >=1 so empty phases give 0 not NaN — NaN would poison the
    gather MatMul).

3.  Window read = double-MatMul gather from runtime scalars:
    `Rsel[r,pr]=((row+r)%m==pr)`, `Csel[pc,c]=((col+c)%m==pc)`,
    `out3 = Rsel@pat@Csel` ([3,3]); cells with r>=bs or c>=bs -> -1 sentinel;
    Pad 3x3 -> 30x30 with -1; final `Equal(L, 0..9)` -> BOOL output (sentinel /
    off-grid cells match no channel => background, exactly as required).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

S = 7  # crop side (size <= 7 always)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    idxS = np.arange(S, dtype=np.float32)

    # ---- constants ----
    # crop channels 1..9 (drop black ch0) and the 7x7 active region in one Slice
    init("CROP_S", np.array([1, 0, 0], np.int64), np.int64)     # starts: ch1, row0, col0
    init("CROP_E", np.array([10, S, S], np.int64), np.int64)    # ends
    init("CROP_AX", np.array([1, 2, 3], np.int64), np.int64)    # axes 1,2,3
    init("WK", np.arange(1, 10, dtype=np.float32).reshape(1, 9, 1, 1), np.float32)  # 1x1 conv: Sum k*ch (k=1..9)
    init("KOUTu", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    init("PADVALu", np.array(200, np.uint8), np.uint8)
    init("SENT", np.array(200.0, np.float32), np.float32)
    init("IDXSr", idxS.reshape(1, 1, S, 1), np.float32)         # per-row index
    init("IDXSc", idxS.reshape(1, 1, 1, S), np.float32)         # per-col index
    init("ONE", np.array(1.0, np.float32), np.float32)
    init("TWO", np.array(2.0, np.float32), np.float32)
    init("FIVE5", np.array(5.5, np.float32), np.float32)
    init("EPS", np.array(0.5, np.float32), np.float32)
    init("BIG99", np.array(99.0, np.float32), np.float32)
    init("ZEROF", np.array(0.0, np.float32), np.float32)
    init("SIXF", np.array(float(S - 1), np.float32), np.float32)
    # output-index small consts
    init("R3F", np.arange(3, dtype=np.float32).reshape(3, 1), np.float32)
    init("C3F", np.arange(3, dtype=np.float32).reshape(1, 3), np.float32)
    init("SH3", np.array([3], np.int64), np.int64)
    init("SH_L3", np.array([1, 1, 3, 3], np.int64), np.int64)
    init("PADS", np.array([0, 0, 0, 0, 0, 0, 27, 27], np.int64), np.int64)

    # ---- crop ----
    n("Slice", ["input", "CROP_S", "CROP_E", "CROP_AX"], "xc")   # [1,9,7,7] fp32 entry (ch1..9, 7x7)
    # 1x1 Convs collapse the 9 channels with NO [1,9,7,7] product plane:
    n("Conv", ["xc", "WK"], "colf")                              # [1,1,7,7] = Sum_k k*ch  (value)

    # ---- per-row / per-col coloured counts (every in-grid row has >=1 coloured cell) ----
    # xc holds only ch1..9 (coloured), so a direct ReduceSum is the coloured count.
    n("ReduceSum", ["xc"], "crow", axes=[1, 3], keepdims=1)      # [1,1,7,1] coloured/row
    n("ReduceSum", ["xc"], "ccol", axes=[1, 2], keepdims=1)      # [1,1,1,7]

    # ---- size, m ----  (in-grid row/col <=> coloured count > 0)
    n("Greater", ["crow", "EPS"], "rhas_b")
    n("Cast", ["rhas_b"], "rhas", to=TensorProto.FLOAT)
    n("Greater", ["ccol", "EPS"], "chas_b")
    n("Cast", ["chas_b"], "chas", to=TensorProto.FLOAT)
    n("Mul", ["rhas", "IDXSr"], "rmaxv")
    n("ReduceMax", ["rmaxv"], "rmax", keepdims=0)
    n("Mul", ["chas", "IDXSc"], "cmaxv")
    n("ReduceMax", ["cmaxv"], "cmax", keepdims=0)
    n("Max", ["rmax", "cmax"], "maxidx")                         # size-1
    n("Add", ["maxidx", "ONE"], "sizef")
    n("Greater", ["maxidx", "FIVE5"], "is7b")
    n("Cast", ["is7b"], "is7", to=TensorProto.FLOAT)
    n("Add", ["is7", "TWO"], "mf")                               # m (2 or 3)

    # ---- hole: row,col,bs ----
    n("Less", ["crow", "sizef"], "rlt")
    n("Greater", ["crow", "EPS"], "rgt")
    n("And", ["rlt", "rgt"], "rhole_b")
    n("Cast", ["rhole_b"], "rhole", to=TensorProto.FLOAT)        # [1,1,7,1]
    n("Less", ["ccol", "sizef"], "clt")
    n("Greater", ["ccol", "EPS"], "cgt")
    n("And", ["clt", "cgt"], "chole_b")
    n("Cast", ["chole_b"], "chole", to=TensorProto.FLOAT)
    n("ReduceSum", ["rhole"], "bsf", keepdims=0)                 # bitesize
    n("Sub", ["ONE", "rhole"], "rnoth")
    n("Mul", ["rnoth", "BIG99"], "rpen")
    n("Add", ["IDXSr", "rpen"], "ridxp")
    n("ReduceMin", ["ridxp"], "rowf", keepdims=0)                # row
    n("Sub", ["ONE", "chole"], "cnoth")
    n("Mul", ["cnoth", "BIG99"], "cpen")
    n("Add", ["IDXSc", "cpen"], "cidxp")
    n("ReduceMin", ["cidxp"], "colfs", keepdims=0)               # col

    # ---- window read = spatial-shift gather (no pattern matmul needed) ----
    # The bite cell (row+r, col+c) is blacked, but the cell one period away has
    # the SAME phase and is guaranteed intact & on-grid: since size >= 2*m, every
    # index a in [0,size) satisfies a>=m OR a<size-m, so srcR(a)=a-m if a>=m else
    # a+m lands on an intact non-black cell with identical pattern colour.
    n("Add", ["mf", "mf"], "twom")                               # 2m
    n("Add", ["R3F", "rowf"], "ar")                              # [3,1] row+r
    n("Add", ["C3F", "colfs"], "ac")                             # [1,3] col+c
    # shift = -m where a>=m else +m ;  src = a + shift  (then clip to [0,6])
    n("Less", ["ar", "mf"], "arlt")                              # a<m
    n("Cast", ["arlt"], "arltf", to=TensorProto.FLOAT)
    n("Mul", ["arltf", "twom"], "arsh")                          # (a<m)?2m:0
    n("Sub", ["arsh", "mf"], "arshift")                          # -m + (a<m)*2m
    n("Add", ["ar", "arshift"], "src_r")                         # srcR(row+r)
    n("Less", ["ac", "mf"], "aclt")
    n("Cast", ["aclt"], "acltf", to=TensorProto.FLOAT)
    n("Mul", ["acltf", "twom"], "acsh")
    n("Sub", ["acsh", "mf"], "acshift")
    n("Add", ["ac", "acshift"], "src_c")
    # clip to [0,6] and cast to int for Gather
    n("Clip", ["src_r", "ZEROF", "SIXF"], "src_rc")
    n("Clip", ["src_c", "ZEROF", "SIXF"], "src_cc")
    n("Cast", ["src_rc"], "gr", to=TensorProto.INT32)            # [3,1]
    n("Cast", ["src_cc"], "gc", to=TensorProto.INT32)            # [1,3]
    n("Reshape", ["gr", "SH3"], "gr1")                           # [3]
    n("Reshape", ["gc", "SH3"], "gc1")                           # [3]
    n("Gather", ["colf", "gr1"], "g2", axis=2)                   # [1,1,3,7]
    n("Gather", ["g2", "gc1"], "out3", axis=3)                   # [1,1,3,3] colours

    # ---- sentinel mask (cells with r>=bs or c>=bs -> 200, matches no channel) ----
    n("Less", ["R3F", "bsf"], "rin")                             # [3,1]
    n("Less", ["C3F", "bsf"], "cin")                             # [1,3]
    n("And", ["rin", "cin"], "keep")                             # [3,3]
    n("Cast", ["keep"], "keepf", to=TensorProto.FLOAT)
    n("Reshape", ["keepf", "SH_L3"], "keepf4")                   # [1,1,3,3]
    n("Mul", ["out3", "keepf4"], "kept")
    n("Sub", ["ONE", "keepf4"], "notk")
    n("Mul", ["notk", "SENT"], "notks")                          # masked cells -> 200
    n("Add", ["kept", "notks"], "L34")                           # [1,1,3,3] keep?out3:200
    n("Cast", ["L34"], "L34u", to=TensorProto.UINT8)             # uint8 carrier (900B plane)
    n("Pad", ["L34u", "PADS", "PADVALu"], "L", mode="constant")  # [1,1,30,30] uint8
    n("Equal", ["L", "KOUTu"], "output")                         # -> BOOL output

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
