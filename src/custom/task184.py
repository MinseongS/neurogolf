"""task184 (ARC-AGI 780d0b14) — "downsample a tall x wide grid of solid colour patches".

Rule (from the generator):
  The input is a (tall x wide) arrangement of solid axis-aligned rectangular patches,
  each filled with ONE colour (1..9) with ~10% random dropout to background (0).  The
  patches are separated by exactly ONE all-background row between block-rows and ONE
  all-background column between block-cols (no leading/trailing separators).  tall,wide
  in {2,3}.  OUTPUT is the (tall x wide) grid whose cell (bi,bj) is the patch colour of
  the block at block-row bi, block-col bj.  Every block is single-coloured & non-empty,
  so the block colour = (sum of colour-index over the block) / (count of non-bg cells).

Encoding (data-dependent downsample as a double weighted MatMul, no flood-fill):
  colf = sum_k k*input_k  (colour-index plane, 0 on bg); occ = colf>0.
  Per-row/col occupancy -> in-grid extent H,W (last content row/col) and all-zero rows/cols.
  Interior separator row  seprow[r] = allzero[r] AND r<H.   block-row index  bri = exclusive
  CumSum(seprow).  Selector  RselO[R,r] = (bri[r]==R) AND (in-grid, non-separator) for R in
  0..2 (tall<=3).  Likewise CselO[C,c] for C in 0..2.
  Snum = RselO @ colf @ CselO^T,  Sden = RselO @ occ @ CselO^T  (both [3,3]).
  Block colour  L = round(Snum/Sden) where Sden>0, sentinel 99 elsewhere; Pad to 30x30 with
  the sentinel; output = Equal(L, arange[1,10,1,1]) -> BOOL.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64

N = 30
K = 3   # max block count per axis (tall, wide in {2,3})


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- colour-index plane colf = sum_k k*input_k  --------------------------
    w = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("convw", w, np.float32)
    n("Conv", ["input", "convw"], "colf32")             # [1,1,30,30] f32, 0 on bg

    init("ZEROF", np.array(0.0, np.float32), np.float32)
    n("Greater", ["colf32", "ZEROF"], "occ_b")          # bool [1,1,30,30]

    # ---- 1-D occupancy profiles (cheap bool reductions via cast to f16) ------
    n("Cast", ["occ_b"], "occ", to=F16)                 # f16 {0,1} [1,1,30,30]
    n("ReduceMax", ["occ"], "rowhas", axes=[3], keepdims=1)   # [1,1,30,1]
    n("ReduceMax", ["occ"], "colhas", axes=[2], keepdims=1)   # [1,1,1,30]

    rampN = np.arange(N, dtype=np.float16)
    init("ramp_r", rampN.reshape(1, 1, N, 1), np.float16)     # [1,1,30,1]
    init("ramp_c", rampN.reshape(1, 1, 1, N), np.float16)     # [1,1,1,30]
    n("Mul", ["rowhas", "ramp_r"], "rr")
    n("ReduceMax", ["rr"], "Hm1", axes=[2], keepdims=1)       # [1,1,1,1]
    n("Mul", ["colhas", "ramp_c"], "cc")
    n("ReduceMax", ["cc"], "Wm1", axes=[3], keepdims=1)       # [1,1,1,1]

    n("Not", [n("Greater", ["ramp_r", "Hm1"], "rgt")], "ingrid_r")  # bool [1,1,30,1]
    n("Not", [n("Greater", ["ramp_c", "Wm1"], "cgt")], "ingrid_c")  # bool [1,1,1,30]

    init("ZEROH", np.array(0.0, np.float16), np.float16)
    n("Not", [n("Greater", ["rowhas", "ZEROH"], "rh_b")], "allzero_r")  # [1,1,30,1]
    n("Not", [n("Greater", ["colhas", "ZEROH"], "ch_b")], "allzero_c")  # [1,1,1,30]

    # separator = allzero AND ingrid ; nonsep = ingrid AND NOT allzero
    n("And", ["allzero_r", "ingrid_r"], "seprow_b")          # [1,1,30,1]
    n("And", ["allzero_c", "ingrid_c"], "sepcol_b")          # [1,1,1,30]
    n("Cast", ["seprow_b"], "seprow", to=F32)
    n("Cast", ["sepcol_b"], "sepcol", to=F32)

    n("Not", ["allzero_r"], "naz_r")
    n("Not", ["allzero_c"], "naz_c")
    n("And", ["ingrid_r", "naz_r"], "nonsep_r_b")            # [1,1,30,1]
    n("And", ["ingrid_c", "naz_c"], "nonsep_c_b")            # [1,1,1,30]

    # ---- block index via exclusive cumsum of separators (fp32, CumSum needs fp)
    init("AX2", np.array(2, np.int64), np.int64)
    init("AX3", np.array(3, np.int64), np.int64)
    n("CumSum", ["seprow", "AX2"], "bri", exclusive=1)       # [1,1,30,1] f32
    n("CumSum", ["sepcol", "AX3"], "bci", exclusive=1)       # [1,1,1,30] f32

    # ---- selector matrices  RselO[R,r], CselO[C,c]  (R,C in 0..K-1) ---------
    # selectors built directly as 4D so MatMul broadcasts over leading [1,1].
    init("rshp4r", np.array([1, 1, 1, N], np.int64), np.int64)  # -> [1,1,1,30]
    init("RidxR", np.arange(K, dtype=np.float32).reshape(1, 1, K, 1), np.float32)  # [1,1,K,1]
    init("RidxC", np.arange(K, dtype=np.float32).reshape(1, 1, 1, K), np.float32)  # [1,1,1,K]

    # block-row selector (R on axis2, r on axis3). Build bool then cast to BOTH
    # f32 (colour chain, reuses fp32 colf32 -> no colf16 plane) and f16 (count chain).
    n("Reshape", ["bri", "rshp4r"], "bri_1r")                # [1,1,1,30] f32
    n("Equal", ["bri_1r", "RidxR"], "Reqb")                  # bool [1,1,K,30]
    n("Reshape", ["nonsep_r_b", "rshp4r"], "nsr_b")          # bool [1,1,1,30]
    n("And", ["Reqb", "nsr_b"], "Rsel_b")                    # bool [1,1,K,30]
    n("Cast", ["Rsel_b"], "RselF", to=F32)                   # f32 [1,1,K,30]
    n("Cast", ["Rsel_b"], "RselH", to=F16)                   # f16 [1,1,K,30]

    # block-col selector transposed (c on axis2, C on axis3)
    init("rshp4c", np.array([1, 1, N, 1], np.int64), np.int64)  # -> [1,1,30,1]
    n("Reshape", ["bci", "rshp4c"], "bci_c1")                # [1,1,30,1] f32
    n("Equal", ["bci_c1", "RidxC"], "Ceqb")                  # bool [1,1,30,K] (c,C)
    n("Reshape", ["nonsep_c_b", "rshp4c"], "nsc_b")          # bool [1,1,30,1]
    n("And", ["Ceqb", "nsc_b"], "Csel_b")                    # bool [1,1,30,K]
    n("Cast", ["Csel_b"], "CselF", to=F32)                   # f32 [1,1,30,K]
    n("Cast", ["Csel_b"], "CselH", to=F16)                   # f16 [1,1,30,K]

    # ---- Snum = RselF @ colf32 @ CselF ; Sden = RselH @ occ @ CselH ----------
    n("MatMul", ["RselF", "colf32"], "num1")                 # f32 [1,1,K,30] (R,c)
    n("MatMul", ["num1", "CselF"], "Snum")                   # f32 [1,1,K,K] (R,C)
    n("MatMul", ["RselH", "occ"], "den1")                    # f16 [1,1,K,30] (R,c)
    n("MatMul", ["den1", "CselH"], "Sden")                   # f16 [1,1,K,K] (R,C)

    init("ONEH", np.array(1.0, np.float16), np.float16)
    n("Cast", ["Snum"], "SnumH", to=F16)                     # f16 [1,1,K,K] (<2048 exact)
    n("Greater", ["Sden", "ZEROH"], "valid_b")               # bool [1,1,K,K]
    n("Where", ["valid_b", "Sden", "ONEH"], "den_safe")      # avoid /0
    n("Div", ["SnumH", "den_safe"], "colr0")                 # f16 [1,1,K,K] colour
    n("Round", ["colr0"], "colr")                            # exact integer colour
    init("SENT", np.array(99.0, np.float16), np.float16)
    n("Where", ["valid_b", "colr", "SENT"], "Lf")            # f16 [1,1,K,K]
    n("Cast", ["Lf"], "Lsmall", to=TensorProto.UINT8)        # uint8 [1,1,K,K]

    # Pad K x K -> 30 x 30 with sentinel
    init("SENTU", np.array(99, np.uint8), np.uint8)
    init("pads", np.array([0, 0, 0, 0, 0, 0, N - K, N - K], np.int64), np.int64)
    n("Pad", ["Lsmall", "pads", "SENTU"], "L", mode="constant")  # uint8 [1,1,30,30]

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                      # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task184", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
