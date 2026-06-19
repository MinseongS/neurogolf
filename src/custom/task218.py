"""task218 (ARC-AGI 90c28cc7) — "compress a quilt of solid colour patches to its block grid".

Rule (from the generator):
  The 21x21 input holds a single rectangular "quilt": a (tall x wide) arrangement of
  solid axis-aligned colour patches (each one colour 1..9, NO dropout — fully filled),
  placed at (rowoffset, coloffset).  Block-row i has height depths[i]; block-col j has
  width lengths[j].  tall,wide in {2,3}.  The generator guarantees no two block-rows are
  identical and no two block-cols are identical (so ADJACENT block-rows/cols always
  differ in at least one cell — block boundaries are detectable from colour changes).
  OUTPUT is the (tall x wide) grid whose cell (bi,bj) is the colour of that block.

Encoding (plane-eliminated re-golf, 2026-06-19):
  colf30 = sum_k k*input_k  (the ONE fp32 colour-index plane, on the full 30x30 input).
  EVERYTHING downstream is computed from colf30 directly — no 21x21 fp32/fp16 colour
  plane and no occupancy plane is ever materialised:
    - row/col block boundaries via two weighted SIGNATURE MatMuls straight off colf30
      (tiny [1,2,30,1] / [1,2,1,30] results); a boundary is where the signature changes.
    - in-quilt extent from ReduceMax(colf30) 1-D profiles (no occupancy plane).
    - block index = inclusive-cumsum(newblock); the 0-basing -1 is FOLDED into the
      selector ramp (Equal(cumsum, R+1)) so no bri/bci plane is materialised; gated
      to in-quilt rows/cols to drop trailing-bg rows that retain the cumsum value.
    - downsample colour: Snum = Rsel @ colf30 @ Csel  (Rsel [1,1,K,30] fp32, the extra
      off-quilt rows/cols are zeroed by the in-quilt gate). The per-block AREA (divisor)
      comes from the selectors themselves — Sden = ReduceSum(Rsel) * ReduceSum(Csel) —
      so NO occupancy plane is needed. block colour = round(Snum/Sden); empty -> 99.
    - the 10-ch one-hot is expanded on the TINY KxK colour grid (Equal vs arange) and
      that small one-hot is Pad-ed straight into the FREE 30x30 uint8 output — no 30x30
      carrier/label plane is ever materialised (invalid blocks -> 99 -> all-zero).
  Mem 11651->8434, params 146->178, pts 15.62->15.94 (+0.31). Fresh 500/500 (all 4
  tall x wide shapes). Dominant intermediate: colf30 3600B fp32 (the irreducible
  10->1 colour-index entry plane; Conv output inherits fp32 input dtype).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64
U8 = TensorProto.UINT8

M = 30      # work on the full 30x30 canvas (the Conv output width)
OUT = 30    # output grid is 30x30 padded
K = 3       # max block count per axis (tall, wide in {2,3})


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- colour-index plane colf30 = sum_k k*input_k (Conv on the FREE input) -
    # The single fp32 entry plane (the 3600B colour-index floor). Everything below
    # works off it directly; no 21x21 colour plane and no occupancy plane.
    w = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("convw", w, np.float32)
    n("Conv", ["input", "convw"], "colf30")                   # [1,1,30,30] f32

    init("ZEROF", np.array(0.0, np.float32), np.float32)

    # ---- 1-D occupancy profiles -> in-grid quilt extent (no occ plane) --------
    n("ReduceMax", ["colf30"], "rowmax", axes=[3], keepdims=1)  # [1,1,30,1] f32
    n("ReduceMax", ["colf30"], "colmax", axes=[2], keepdims=1)  # [1,1,1,30] f32
    n("Greater", ["rowmax", "ZEROF"], "rq_b")                 # bool [1,1,30,1] row in quilt
    n("Greater", ["colmax", "ZEROF"], "cq_b")                 # bool [1,1,1,30] col in quilt

    # ---- new-block indicator via weighted SIGNATURE vectors (off colf30) ------
    # sigrow[r] = sum_c (c+1)*colf30[r,c]; adjacent distinct block-rows differ in
    # >=1 column -> distinct weighted sums. Two independent weight vectors so a
    # boundary is missed only if BOTH collide (verified astronomically unlikely).
    w1 = np.arange(M, dtype=np.float32) + 1.0
    w2 = (np.arange(M, dtype=np.float32) + 1.0) ** 2          # distinct, fp32-exact
    Wc = np.stack([w1, w2], axis=1).reshape(1, 1, M, 2)        # [1,1,M,2]
    Wr = np.stack([w1, w2], axis=0).reshape(1, 1, 2, M)        # [1,1,2,M]
    init("Wcol", Wc, np.float32)
    init("Wrow", Wr, np.float32)
    n("MatMul", ["colf30", "Wcol"], "sigrow")                 # [1,1,30,2] f32
    n("MatMul", ["Wrow", "colf30"], "sigcol")                 # [1,1,2,30] f32

    init("st1", np.array([1], np.int64), np.int64)
    init("enM", np.array([M], np.int64), np.int64)
    init("ax2", np.array([2], np.int64), np.int64)
    init("ax3", np.array([3], np.int64), np.int64)
    init("st0a", np.array([0], np.int64), np.int64)
    init("enMm1", np.array([M - 1], np.int64), np.int64)
    init("TRUE1", np.array([[[[True]]]], np.bool_), np.bool_)

    # rows: boundary where EITHER signature differs from the previous row.
    # Compute the difference of adjacent rows (one [1,1,29,2] plane) and test !=0
    # per hash channel; OR the 2 channels.  Adjacent-row diff is small-valued.
    init("h0", np.array([0], np.int64), np.int64)
    init("h1", np.array([1], np.int64), np.int64)
    init("h2", np.array([2], np.int64), np.int64)
    n("Slice", ["sigrow", "st1", "enM", "ax2"], "sr_cur")     # [1,1,29,2]
    n("Slice", ["sigrow", "st0a", "enMm1", "ax2"], "sr_prev") # [1,1,29,2]
    n("Equal", ["sr_cur", "sr_prev"], "sr_same2")             # bool [1,1,29,2]
    # row same iff BOTH hash channels match: And of the two bool channel-slices
    # (avoids the fp16 ReduceMin bridge plane).
    n("Slice", ["sr_same2", "h0", "h1", "ax3"], "sr_s0")      # bool [1,1,29,1]
    n("Slice", ["sr_same2", "h1", "h2", "ax3"], "sr_s1")      # bool [1,1,29,1]
    n("And", ["sr_s0", "sr_s1"], "sr_same")                   # bool [1,1,29,1]
    n("Not", ["sr_same"], "diffrow_b")                        # bool [1,1,29,1]
    n("Concat", ["TRUE1", "diffrow_b"], "newrow_b", axis=2)   # bool [1,1,30,1]
    n("And", ["newrow_b", "rq_b"], "newrow_q_b")
    n("Cast", ["newrow_q_b"], "newrow_f", to=F32)             # f32 (CumSum needs fp32)

    # cols: boundary where EITHER signature differs from the previous col
    n("Slice", ["sigcol", "st1", "enM", "ax3"], "sc_cur")     # [1,1,2,29]
    n("Slice", ["sigcol", "st0a", "enMm1", "ax3"], "sc_prev")
    n("Equal", ["sc_cur", "sc_prev"], "sc_same2")             # bool [1,1,2,29]
    n("Slice", ["sc_same2", "h0", "h1", "ax2"], "sc_s0")      # bool [1,1,1,29]
    n("Slice", ["sc_same2", "h1", "h2", "ax2"], "sc_s1")      # bool [1,1,1,29]
    n("And", ["sc_s0", "sc_s1"], "sc_same")                   # bool [1,1,1,29]
    n("Not", ["sc_same"], "diffcol_b")                        # [1,1,1,29]
    n("Concat", ["TRUE1", "diffcol_b"], "newcol_b", axis=3)   # bool [1,1,1,30]
    n("And", ["newcol_b", "cq_b"], "newcol_q_b")
    n("Cast", ["newcol_q_b"], "newcol_f", to=F32)

    # ---- block index = inclusive cumsum(newblock); the -1 (0-basing) is folded
    # into the selector ramp constants below, so no separate bri/bci plane. -----
    init("AX2", np.array(2, np.int64), np.int64)
    init("AX3", np.array(3, np.int64), np.int64)
    n("CumSum", ["newrow_f", "AX2"], "cr")                    # inclusive, [1,1,30,1] f32
    n("CumSum", ["newcol_f", "AX3"], "cc")

    # ---- selector matrices Rsel[R,r] (R axis2), Csel[c,C] (C axis3) ----------
    init("rshp4r", np.array([1, 1, 1, M], np.int64), np.int64)  # -> [1,1,1,30]
    init("rshp4c", np.array([1, 1, M, 1], np.int64), np.int64)  # -> [1,1,30,1]
    # ramps are arange(K)+1 so Equal(inclusive_cumsum, R+1) == (block_index==R)
    init("RidxR", (np.arange(K, dtype=np.float32) + 1.0).reshape(1, 1, K, 1), np.float32)
    init("RidxC", (np.arange(K, dtype=np.float32) + 1.0).reshape(1, 1, 1, K), np.float32)

    # block-row selector (R on axis2, r on axis3); gate to in-quilt rows so a
    # trailing bg row (which retains the last inclusive-cumsum value) is excluded.
    n("Reshape", ["cr", "rshp4r"], "cr_1r")                   # [1,1,1,30] f32
    n("Equal", ["cr_1r", "RidxR"], "Reqb")                    # bool [1,1,K,30]
    n("Reshape", ["rq_b", "rshp4r"], "rq_1r")                 # bool [1,1,1,30]
    n("And", ["Reqb", "rq_1r"], "Rsel_b")                     # bool [1,1,K,30]
    n("Cast", ["Rsel_b"], "RselF", to=F32)                    # f32 [1,1,K,30]

    # block-col selector transposed (c on axis2, C on axis3)
    n("Reshape", ["cc", "rshp4c"], "cc_c1")                   # [1,1,30,1] f32
    n("Equal", ["cc_c1", "RidxC"], "Ceqb")                    # bool [1,1,30,K]
    n("Reshape", ["cq_b", "rshp4c"], "cq_c1")                 # bool [1,1,30,1]
    n("And", ["Ceqb", "cq_c1"], "Csel_b")                     # bool [1,1,30,K]
    n("Cast", ["Csel_b"], "CselF", to=F32)

    # ---- Snum = Rsel @ colf30 @ Csel  (all fp32, tiny after 1st MatMul) -------
    n("MatMul", ["RselF", "colf30"], "num1")                  # f32 [1,1,K,30]
    n("MatMul", ["num1", "CselF"], "Snum")                    # f32 [1,1,K,K]

    # ---- per-block AREA from the selectors themselves (NO occupancy plane) ----
    # rowcount[bi] = #rows assigned to block-row bi = ReduceSum(Rsel, cols);
    # colcount[bj] = ReduceSum(Csel, rows).  Sden = rowcount * colcount (full rect).
    n("ReduceSum", ["RselF"], "rowcount", axes=[3], keepdims=1)  # f32 [1,1,K,1]
    n("ReduceSum", ["CselF"], "colcount", axes=[2], keepdims=1)  # f32 [1,1,1,K]
    n("Mul", ["rowcount", "colcount"], "Sden")                # f32 [1,1,K,K]

    # ---- block colour = round(Snum / Sden); empty block -> 99 sentinel --------
    init("ONEF1", np.array(1.0, np.float32), np.float32)
    n("Greater", ["Sden", "ZEROF"], "valid_b")               # bool [1,1,K,K]
    n("Where", ["valid_b", "Sden", "ONEF1"], "den_safe")
    n("Div", ["Snum", "den_safe"], "colr0")
    n("Round", ["colr0"], "colr")
    init("SENT", np.array(99.0, np.float32), np.float32)
    n("Where", ["valid_b", "colr", "SENT"], "Lf")             # f32 [1,1,K,K]
    n("Cast", ["Lf"], "Lsmall", to=U8)                        # uint8 [1,1,K,K]

    # PLANE-ELIMINATION: expand the 10-ch one-hot on the TINY KxK colour grid,
    # then Pad that small one-hot straight into the FREE 30x30 output. No 30x30
    # carrier plane (the old 900B uint8 L) is ever materialised — invalid blocks
    # carry colour 99 (==no channel) so they expand to all-zero, and the Pad's
    # zero fill leaves every off-grid cell all-zero (matches the generator output).
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["Lsmall", "chan"], "oh_b")                    # bool [1,10,K,K]
    n("Cast", ["oh_b"], "oh", to=U8)                          # uint8 [1,10,K,K]
    init("pads", np.array([0, 0, 0, 0, 0, 0, OUT - K, OUT - K], np.int64), np.int64)
    init("ZEROU", np.array(0, np.uint8), np.uint8)
    n("Pad", ["oh", "pads", "ZEROU"], "output", mode="constant")  # uint8 [1,10,30,30]

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", U8, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task218", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
