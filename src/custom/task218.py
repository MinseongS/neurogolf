"""task218 (ARC-AGI 90c28cc7) — "compress a quilt of solid colour patches to its block grid".

Rule (from the generator):
  The 21x21 input holds a single rectangular "quilt": a (tall x wide) arrangement of
  solid axis-aligned colour patches (each one colour 1..9, NO dropout — fully filled),
  placed at (rowoffset, coloffset).  Block-row i has height depths[i]; block-col j has
  width lengths[j].  tall,wide in {2,3}.  The generator guarantees no two block-rows are
  identical and no two block-cols are identical (so ADJACENT block-rows/cols always
  differ in at least one cell — block boundaries are detectable from colour changes).
  OUTPUT is the (tall x wide) grid whose cell (bi,bj) is the colour of that block.

Encoding (data-dependent downsample as a double weighted MatMul, no flood-fill):
  colf = sum_k k*input_k  (colour-index plane on the 21x21 canvas, 0 on bg).
  In-grid quilt extent from 1-D occupancy.  A row r (r>=1, in quilt) starts a NEW block
  iff its colour profile differs from row r-1's; first quilt row also starts a block.
  bri[r] = inclusive-cumsum(newblock_r) - 1  -> 0-based block-row index (gated to quilt).
  Likewise bci[c].  Build [1,1,K,21] / [1,1,21,K] one-hot selectors, downsample
  Snum = Rsel @ colf @ Csel, Sden = Rsel @ occ @ Csel; block colour = Snum/Sden (exact,
  every block single-coloured & non-empty).  K=3.  KxK colour -> Pad(99) -> Equal(arange).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64

N = 21      # input is fixed 21x21
OUT = 30    # output grid is still 30x30 padded
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

    # ---- colour-index plane colf = sum_k k*input_k (Conv on the FREE input) --
    # never slice the 10-ch input (that would materialize a [1,10,21,21]=17640B
    # plane); run the Conv on the full input then crop the SINGLE-channel result.
    w = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("convw", w, np.float32)
    n("Conv", ["input", "convw"], "colf30")                   # [1,1,30,30] f32
    # crop the single-channel colour plane to the 21x21 active canvas (fp32), then
    # cast to fp16 for the downstream full-plane ops (half the static-mem cost).
    init("st0", np.array([0, 0], np.int64), np.int64)
    init("en21", np.array([N, N], np.int64), np.int64)
    init("ax23", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["colf30", "st0", "en21", "ax23"], "colf32")   # [1,1,21,21] f32
    n("Cast", ["colf32"], "colf", to=F16)                     # [1,1,21,21] f16

    init("ZEROH", np.array(0.0, np.float16), np.float16)
    n("Greater", ["colf", "ZEROH"], "occ_b")                  # bool [1,1,21,21]
    n("Cast", ["occ_b"], "occ", to=F16)                       # f16 {0,1}

    # ---- 1-D occupancy profiles -> in-grid quilt extent ----------------------
    n("ReduceMax", ["occ"], "rowhas", axes=[3], keepdims=1)   # [1,1,21,1] f16
    n("ReduceMax", ["occ"], "colhas", axes=[2], keepdims=1)   # [1,1,1,21] f16
    n("Greater", ["rowhas", "ZEROH"], "rq_b")                 # bool [1,1,21,1] row in quilt
    n("Greater", ["colhas", "ZEROH"], "cq_b")                 # bool [1,1,1,21] col in quilt

    # ---- new-block indicator via cheap per-row/col SIGNATURE vectors ---------
    # Each row of a block-row is identical; adjacent block-rows always differ in
    # >=1 column.  A weighted row signature sigrow[r] = sum_c (c+1)*colf32[r,c]
    # (fp32, exact) gives a per-row scalar that changes exactly at block-row
    # boundaries (distinct profiles -> distinct weighted sums; verified on fresh).
    # MatMul contracts the column axis -> [1,1,21,1]; col signature contracts rows.
    # use TWO independent weight vectors (W=[w1|w2]) so a boundary is missed only
    # if BOTH weighted sums collide for an adjacent distinct pair — astronomically
    # unlikely (verified on fresh).  One MatMul yields both signatures as columns.
    w1 = np.arange(N, dtype=np.float32) + 1.0
    w2 = (np.arange(N, dtype=np.float32) + 1.0) ** 2          # distinct, fp32-exact
    Wc = np.stack([w1, w2], axis=1).reshape(1, 1, N, 2)        # [1,1,N,2]
    Wr = np.stack([w1, w2], axis=0).reshape(1, 1, 2, N)        # [1,1,2,N]
    init("Wcol", Wc, np.float32)
    init("Wrow", Wr, np.float32)
    n("MatMul", ["colf32", "Wcol"], "sigrow")                 # [1,1,21,2] f32
    n("MatMul", ["Wrow", "colf32"], "sigcol")                 # [1,1,2,21] f32

    init("st1", np.array([1], np.int64), np.int64)
    init("enN", np.array([N], np.int64), np.int64)
    init("ax2", np.array([2], np.int64), np.int64)
    init("ax3", np.array([3], np.int64), np.int64)
    init("st0a", np.array([0], np.int64), np.int64)
    init("enNm1", np.array([N - 1], np.int64), np.int64)

    # rows: boundary where EITHER signature differs from the previous row
    n("Slice", ["sigrow", "st1", "enN", "ax2"], "sr_cur")     # [1,1,20,2]
    n("Slice", ["sigrow", "st0a", "enNm1", "ax2"], "sr_prev") # [1,1,20,2]
    n("Equal", ["sr_cur", "sr_prev"], "sr_same2")             # bool [1,1,20,2]
    n("Cast", ["sr_same2"], "sr_same2f", to=F16)
    n("ReduceMin", ["sr_same2f"], "sr_samef", axes=[3], keepdims=1)  # [1,1,20,1]
    n("Greater", ["sr_samef", "ZEROH"], "sr_same")            # bool [1,1,20,1]
    n("Not", ["sr_same"], "diffrow_b")                        # bool [1,1,20,1]
    init("TRUE1", np.array([[[[True]]]], np.bool_), np.bool_)
    n("Concat", ["TRUE1", "diffrow_b"], "newrow_b", axis=2)   # bool [1,1,21,1]
    n("And", ["newrow_b", "rq_b"], "newrow_q_b")
    n("Cast", ["newrow_q_b"], "newrow_f", to=F32)             # f32 (CumSum needs fp32)

    # cols: boundary where EITHER signature differs from the previous col
    n("Slice", ["sigcol", "st1", "enN", "ax3"], "sc_cur")     # [1,1,2,20]
    n("Slice", ["sigcol", "st0a", "enNm1", "ax3"], "sc_prev")
    n("Equal", ["sc_cur", "sc_prev"], "sc_same2")             # bool [1,1,2,20]
    n("Cast", ["sc_same2"], "sc_same2f", to=F16)
    n("ReduceMin", ["sc_same2f"], "sc_samef", axes=[2], keepdims=1)  # [1,1,1,20]
    n("Greater", ["sc_samef", "ZEROH"], "sc_same")            # bool [1,1,1,20]
    n("Not", ["sc_same"], "diffcol_b")                        # [1,1,1,20]
    n("Concat", ["TRUE1", "diffcol_b"], "newcol_b", axis=3)   # bool [1,1,1,21]
    n("And", ["newcol_b", "cq_b"], "newcol_q_b")
    n("Cast", ["newcol_q_b"], "newcol_f", to=F32)

    # ---- block index = inclusive cumsum(newblock) - 1  -----------------------
    init("AX2", np.array(2, np.int64), np.int64)
    init("AX3", np.array(3, np.int64), np.int64)
    init("ONEF", np.array(1.0, np.float32), np.float32)
    n("CumSum", ["newrow_f", "AX2"], "cr")                    # inclusive, [1,1,21,1] f32
    n("Sub", ["cr", "ONEF"], "bri")                           # 0-based (bg rows -> -1)
    n("CumSum", ["newcol_f", "AX3"], "cc")
    n("Sub", ["cc", "ONEF"], "bci")

    # ---- selector matrices RselO[R,r], CselO[c,C] (R,C in 0..K-1) -----------
    init("rshp4r", np.array([1, 1, 1, N], np.int64), np.int64)  # -> [1,1,1,21]
    init("rshp4c", np.array([1, 1, N, 1], np.int64), np.int64)  # -> [1,1,21,1]
    init("RidxR", np.arange(K, dtype=np.float32).reshape(1, 1, K, 1), np.float32)
    init("RidxC", np.arange(K, dtype=np.float32).reshape(1, 1, 1, K), np.float32)

    # block-row selector (R on axis2, r on axis3); gate to in-quilt rows so a
    # trailing bg row (which retains the last inclusive-cumsum value) is excluded.
    n("Reshape", ["bri", "rshp4r"], "bri_1r")                 # [1,1,1,21] f16
    n("Equal", ["bri_1r", "RidxR"], "Reqb")                   # bool [1,1,K,21]
    n("Reshape", ["rq_b", "rshp4r"], "rq_1r")                 # bool [1,1,1,21]
    n("And", ["Reqb", "rq_1r"], "Rsel_b")                     # bool [1,1,K,21]
    n("Cast", ["Rsel_b"], "RselH", to=F16)                    # f16 [1,1,K,21]

    # block-col selector transposed (c on axis2, C on axis3)
    n("Reshape", ["bci", "rshp4c"], "bci_c1")                 # [1,1,21,1] f16
    n("Equal", ["bci_c1", "RidxC"], "Ceqb")                   # bool [1,1,21,K]
    n("Reshape", ["cq_b", "rshp4c"], "cq_c1")                 # bool [1,1,21,1]
    n("And", ["Ceqb", "cq_c1"], "Csel_b")                     # bool [1,1,21,K]
    n("Cast", ["Csel_b"], "CselH", to=F16)

    # ---- Snum = Rsel @ colf @ Csel ; Sden = Rsel @ occ @ Csel (all fp16) -----
    n("MatMul", ["RselH", "colf"], "num1")                    # f16 [1,1,K,21]
    n("MatMul", ["num1", "CselH"], "SnumH")                   # f16 [1,1,K,K]
    n("MatMul", ["RselH", "occ"], "den1")                     # f16 [1,1,K,21]
    n("MatMul", ["den1", "CselH"], "Sden")                    # f16 [1,1,K,K]

    init("ONEH16", np.array(1.0, np.float16), np.float16)
    n("Greater", ["Sden", "ZEROH"], "valid_b")               # bool [1,1,K,K]
    n("Where", ["valid_b", "Sden", "ONEH16"], "den_safe")
    n("Div", ["SnumH", "den_safe"], "colr0")
    n("Round", ["colr0"], "colr")
    init("SENT", np.array(99.0, np.float16), np.float16)
    n("Where", ["valid_b", "colr", "SENT"], "Lf")             # f16 [1,1,K,K]
    n("Cast", ["Lf"], "Lsmall", to=TensorProto.UINT8)

    # Pad K x K -> 30 x 30 with sentinel
    init("SENTU", np.array(99, np.uint8), np.uint8)
    init("pads", np.array([0, 0, 0, 0, 0, 0, OUT - K, OUT - K], np.int64), np.int64)
    n("Pad", ["Lsmall", "pads", "SENTU"], "L", mode="constant")  # uint8 [1,1,30,30]

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                       # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task218", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
