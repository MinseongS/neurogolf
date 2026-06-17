"""task190 (ARC-AGI 7ddcd7ec) — extend each diagonal seed of a 2x2 box into a full 45 ray.

Rule (generator task_7ddcd7ec.py, size=10 fixed grid):
  A solid 2x2 box of one COLOUR k sits at (row,col).  Up to three of the four diagonal
  corners get a SEED pixel (same colour) one cell out from the matching box corner:
      d0 up-left   seed (row-1,col-1)   d1 up-right  seed (row-1,col+2)
      d3 down-left seed (row+2,col-1)   d2 down-right seed (row+2,col+2)
  The INPUT shows box + the (present) seeds (one cell each).  The OUTPUT extends each
  present seed into a FULL 45 ray from that corner out to the grid edge; box preserved.

  Closed-form (verified 0/5000 fresh):
    - main diagonal value  Dmain = row-col   (carries d0 up-left ray and d2 down-right ray)
    - anti  diagonal value  Aanti = row+col+1 (carries d1 up-right ray and d3 down-left ray)
    - upper half = rows <= row-1 ;  lower half = rows >= row+2  (box rows row,row+1 split them)
    - ray present iff occupancy exists on that diagonal in that half:
        has_d2 = any(occ & ondiag & lower)   has_d0 = any(occ & ondiag & upper)
        has_d3 = any(occ & onanti & lower)   has_d1 = any(occ & onanti & upper)
    - fill = box | ondiag&((lower&has_d2)|(upper&has_d0)) | onanti&((lower&has_d3)|(upper&has_d1))
  The box cells on each diagonal (row,col)&(row+1,col+1) sit in NEITHER half so they never
  trigger a flag; a half-flag fires only when a seed extends occupancy past the box.

Encoding (route 10-ch expansion into the FREE bool output):
  - colf = 1x1 Conv (w[0,k]=k) -> colour-index plane; crop to 10x10; k = ReduceMax.
  - occ = colf>0 ; box top-left = 2x2 sum-Conv==4 (unique); row/col = ReduceSum(btl*ramp).
  - dval=rr-cc, aval=rr+cc are the only irreducible 2-D planes (broadcast 1-D fp16 ramps);
    ondiag/onanti = Equal to scalar Dmain/Aanti; halves are 1-D column vectors.
  - flags = ReduceMax over (occ & half & on...) > 0 (scalar bool, no Gather).
  - L = fill*k -> Pad to 30x30 sentinel 255 (uint8) -> output = Equal(L, arange) BOOL FREE.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I64 = TensorProto.INT64

N = 10  # active grid is always 10x10


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- occupancy from background channel-0 (avoids the 3600B colour plane) -
    init("b_s", np.array([0, 0, 0], np.int64), np.int64)
    init("b_e", np.array([1, N, N], np.int64), np.int64)
    init("b_a", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "b_s", "b_e", "b_a"], "ch0")     # [1,1,10,10] f32 bg
    init("HALF32", np.array(0.5, np.float32), np.float32)
    n("Less", ["ch0", "HALF32"], "occ")                   # bool: bg==0 -> occupied
    n("Cast", ["occ"], "occf", to=F16)                    # f16 {0,1}

    # colour scalar k = ReduceSum(present_channel * arange) (no full colour plane)
    init("arange10", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    n("ReduceMax", ["input"], "presf", axes=[2, 3], keepdims=1)  # [1,10,1,1] f32
    n("Mul", ["presf", "arange10"], "kparts")
    n("ReduceSum", ["kparts"], "k_f32", axes=[1], keepdims=1)    # [1,1,1,1] f32
    n("Cast", ["k_f32"], "k16", to=F16)

    # ---- 2x2 box top-left via sum-Conv == 4 ---------------------------------
    init("K22", np.ones((1, 1, 2, 2), np.float16), np.float16)
    n("Conv", ["occf", "K22"], "bc")                      # [1,1,9,9] f16
    init("FOUR", np.array(4.0, np.float16), np.float16)
    n("Equal", ["bc", "FOUR"], "btl_b")                   # [1,1,9,9] bool (unique)
    n("Cast", ["btl_b"], "btl", to=F16)                   # f16 {0,1}

    rr9 = np.arange(9, dtype=np.float16).reshape(1, 1, 9, 1)
    cc9 = np.arange(9, dtype=np.float16).reshape(1, 1, 1, 9)
    init("rr9", rr9, np.float16)
    init("cc9", cc9, np.float16)
    # reduce btl to 1-D row/col profiles first (avoid two full 9x9 product planes)
    n("ReduceSum", ["btl"], "btlrow", axes=[3], keepdims=1)  # [1,1,9,1]
    n("ReduceSum", ["btl"], "btlcol", axes=[2], keepdims=1)  # [1,1,1,9]
    n("Mul", ["btlrow", "rr9"], "btlr_w")
    n("ReduceSum", ["btlr_w"], "row16", axes=[2], keepdims=1)  # [1,1,1,1] = row
    n("Mul", ["btlcol", "cc9"], "btlc_w")
    n("ReduceSum", ["btlc_w"], "col16", axes=[3], keepdims=1)  # [1,1,1,1] = col

    # scalars
    n("Sub", ["row16", "col16"], "Dmain")
    n("Add", ["row16", "col16"], "rpc")
    init("ONE16", np.array(1.0, np.float16), np.float16)
    n("Add", ["rpc", "ONE16"], "Aanti")
    init("TWO16", np.array(2.0, np.float16), np.float16)
    n("Add", ["row16", "TWO16"], "rlo")                   # row+2
    n("Add", ["row16", "ONE16"], "rowp1")                 # row+1
    n("Add", ["col16", "ONE16"], "colp1")                 # col+1

    # ---- 2-D diagonal planes (the only irreducible full planes) -------------
    rrf = np.arange(N, dtype=np.float16).reshape(1, 1, N, 1)
    ccf = np.arange(N, dtype=np.float16).reshape(1, 1, 1, N)
    init("rrf", rrf, np.float16)                          # [1,1,10,1]
    init("ccf", ccf, np.float16)                          # [1,1,1,10]
    # ondiag: rr-cc==Dmain  <=>  rr == cc+Dmain ; build the [1,1,1,10] target row
    # vector so the Equal broadcasts straight to bool (no f16 dval/aval plane).
    n("Add", ["ccf", "Dmain"], "cd")                      # [1,1,1,10] = cc+Dmain
    n("Equal", ["rrf", "cd"], "ondiag")                   # [1,1,10,10] bool
    n("Sub", ["Aanti", "ccf"], "ac")                      # [1,1,1,10] = Aanti-cc
    n("Equal", ["rrf", "ac"], "onanti")                   # [1,1,10,10] bool

    # half-plane column vectors (broadcast over cols)
    n("Not", [n("Less", ["rrf", "rlo"], "lt_lo")], "lower")  # rr >= row+2
    upper = n("Less", ["rrf", "row16"], "upper")             # rr <= row-1 (== rr < row)

    # ---- ray-present flags via per-row occupancy profiles -------------------
    # od/oa = occupied cells ON the (anti)diagonal; reduce over columns to a row
    # profile [1,1,10,1], then split by half with the 1-D lower/upper vectors.
    init("Z32f16", np.array(0.0, np.float16), np.float16)
    n("Cast", ["lower"], "lowerf", to=F16)
    n("Cast", [upper], "upperf", to=F16)
    n("Where", ["ondiag", "occf", "Z32f16"], "od")        # occ on main diag, f16
    n("Where", ["onanti", "occf", "Z32f16"], "oa")        # occ on anti diag, f16
    n("ReduceMax", ["od"], "odrow", axes=[3], keepdims=1)  # [1,1,10,1] f16
    n("ReduceMax", ["oa"], "oarow", axes=[3], keepdims=1)  # [1,1,10,1] f16

    def flag(prof, halff, tag):
        n("Mul", [prof, halff], f"pm_{tag}")              # [1,1,10,1]
        n("ReduceMax", [f"pm_{tag}"], f"fl_{tag}", axes=[2], keepdims=1)
        n("Greater", [f"fl_{tag}", "Z32f16"], f"has_{tag}")
        return f"has_{tag}"

    has_d2 = flag("odrow", "lowerf", "d2")
    has_d0 = flag("odrow", "upperf", "d0")
    has_d3 = flag("oarow", "lowerf", "d3")
    has_d1 = flag("oarow", "upperf", "d1")

    # ---- assemble fill mask -------------------------------------------------
    n("And", ["lower", has_d2], "ld2")
    n("And", [upper, has_d0], "ud0")
    n("Or", ["ld2", "ud0"], "dhalf")                      # [1,1,10,1] bool
    n("And", ["ondiag", "dhalf"], "diagfill")             # [1,1,10,10] bool
    n("And", ["lower", has_d3], "ld3")
    n("And", [upper, has_d1], "ud1")
    n("Or", ["ld3", "ud1"], "ahalf")
    n("And", ["onanti", "ahalf"], "antifill")

    # box mask
    n("Not", [n("Less", ["rrf", "row16"], "blt0")], "br_ge")     # rr >= row
    n("Not", [n("Greater", ["rrf", "rowp1"], "bgt0")], "br_le")  # rr <= row+1
    n("And", ["br_ge", "br_le"], "brow")                  # [1,1,10,1]
    n("Not", [n("Less", ["ccf", "col16"], "bltc")], "bc_ge")     # cc >= col
    n("Not", [n("Greater", ["ccf", "colp1"], "bgtc")], "bc_le")  # cc <= col+1
    n("And", ["bc_ge", "bc_le"], "bcol")                  # [1,1,1,10]
    n("And", ["brow", "bcol"], "boxmask")                 # [1,1,10,10]

    # fill = box | diagfill | antifill
    n("Or", ["diagfill", "antifill"], "rays")
    n("Or", ["boxmask", "rays"], "fill")                  # [1,1,10,10] bool

    # ---- L = k where filled -> Pad 30x30 sentinel 255 -> Equal(arange) ----
    n("Where", ["fill", "k16", "Z32f16"], "L16")          # [1,1,10,10] f16 (k or 0)
    n("Cast", ["L16"], "Lu8", to=U8)                      # uint8 0..9
    init("Lpads", np.array([0, 0, 0, 0, 0, 0, 30 - N, 30 - N], np.int64), np.int64)
    init("SENT", np.array(255, np.uint8), np.uint8)
    n("Pad", ["Lu8", "Lpads", "SENT"], "L30", mode="constant")   # [1,1,30,30] u8
    init("arange", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L30", "arange"], "output")               # [1,10,30,30] BOOL FREE

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task190", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
