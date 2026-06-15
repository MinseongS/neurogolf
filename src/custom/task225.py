"""Task 225 (ARC-AGI 93b581b8): diagonal-corner 2x2 stamp expansion.

Rule (verified exact on stored + fresh arc-gen). Grid is ALWAYS 6x6.
A single 2x2 block of four DISTINCT random colours sits at top-left (row, col)
with row, col in 1..3:

      c0 c1            c0 @ (row,  col)   c1 @ (row,  col+1)
      c2 c3            c2 @ (row+1,col)   c3 @ (row+1,col+1)

The input is that block alone.  The output keeps the block and additionally
stamps a 2x2 monochrome block of each colour at a diagonal corner offset
(out-of-grid parts clipped):

      c0 -> top-left (row+2, col+2)   c1 -> (row+2, col-2)
      c2 -> (row-2, col+2)            c3 -> (row-2, col-2)

The block and the four stamps are pairwise disjoint, so output = block + stamps.
Every painted cell lies inside the 6x6 grid (rows/cols 0..5), so the whole
construction runs on a 6x6 label canvas and is Pad'd to 30x30 (sentinel 10 =>
all channels off) before the final Equal that routes the 10-channel expansion
into the FREE `output`.

Memory plan (all big tensors avoided):
  * rowK[1,10,30,1] / colK[1,10,1,30] reductions of FREE `input` -> 1200B each.
  * colours recovered as channel one-hots [1,10,1,1] (intersection of distinct
    row-set and col-set) -> colour INDEX scalars [1,1,1,1].
  * geometry scalars row, col from occupied-index sums.
  * label built on a 6x6 canvas (144B) from 1-D region range masks * scalar
    colour indices; Pad -> [1,1,30,30] uint8 (900B); Equal -> bool `output`.
Dominant intermediates: the two 1200B per-channel reductions + 900B padded
label.  No 10-channel and no 30x30 colour plane is ever materialised.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def build(task):
    inits = []
    nodes = []
    vinfos = []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    def vi(name, dtype, shape):
        vinfos.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    # =====================================================================
    # 1. per-channel row / col sums of the (free) input one-hot
    # =====================================================================
    n("ReduceSum", ["input"], "rowK", axes=[3], keepdims=1)
    vi("rowK", TensorProto.FLOAT, [1, 10, 30, 1])
    n("ReduceSum", ["input"], "colK", axes=[2], keepdims=1)
    vi("colK", TensorProto.FLOAT, [1, 10, 1, 30])

    chm = np.ones((1, 10, 1, 1), np.float32)
    chm[0, 0, 0, 0] = 0.0
    init("CHM", chm, np.float32)

    n("Mul", ["rowK", "CHM"], "rowKc")
    vi("rowKc", TensorProto.FLOAT, [1, 10, 30, 1])
    n("Mul", ["colK", "CHM"], "colKc")
    vi("colKc", TensorProto.FLOAT, [1, 10, 1, 30])

    n("ReduceSum", ["rowKc"], "occr", axes=[1], keepdims=1)   # [1,1,30,1]
    vi("occr", TensorProto.FLOAT, [1, 1, 30, 1])
    n("ReduceSum", ["colKc"], "occc", axes=[1], keepdims=1)   # [1,1,1,30]
    vi("occc", TensorProto.FLOAT, [1, 1, 1, 30])

    # =====================================================================
    # 2. geometry scalars: row = (sum occupied row idx - 1) / 2 ; col idem
    # =====================================================================
    ridx = np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1)
    cidx = np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30)
    init("RIDX", ridx, np.float32)
    init("CIDX", cidx, np.float32)
    init("zero_f", np.array(0.0, np.float32), np.float32)
    init("one_f", np.array(1.0, np.float32), np.float32)
    init("two_f", np.array(2.0, np.float32), np.float32)

    n("Greater", ["occr", "zero_f"], "occr_b")
    vi("occr_b", TensorProto.BOOL, [1, 1, 30, 1])
    n("Cast", ["occr_b"], "occr_f", to=TensorProto.FLOAT)
    vi("occr_f", TensorProto.FLOAT, [1, 1, 30, 1])
    n("Greater", ["occc", "zero_f"], "occc_b")
    vi("occc_b", TensorProto.BOOL, [1, 1, 1, 30])
    n("Cast", ["occc_b"], "occc_f", to=TensorProto.FLOAT)
    vi("occc_f", TensorProto.FLOAT, [1, 1, 1, 30])

    n("Mul", ["occr_f", "RIDX"], "riw")
    vi("riw", TensorProto.FLOAT, [1, 1, 30, 1])
    n("ReduceSum", ["riw"], "rsum", axes=[2], keepdims=1)
    vi("rsum", TensorProto.FLOAT, [1, 1, 1, 1])
    n("Mul", ["occc_f", "CIDX"], "ciw")
    vi("ciw", TensorProto.FLOAT, [1, 1, 1, 30])
    n("ReduceSum", ["ciw"], "csum", axes=[3], keepdims=1)
    vi("csum", TensorProto.FLOAT, [1, 1, 1, 1])

    n("Sub", ["rsum", "one_f"], "rsm1")
    vi("rsm1", TensorProto.FLOAT, [1, 1, 1, 1])
    n("Div", ["rsm1", "two_f"], "ROW")
    vi("ROW", TensorProto.FLOAT, [1, 1, 1, 1])
    n("Sub", ["csum", "one_f"], "csm1")
    vi("csm1", TensorProto.FLOAT, [1, 1, 1, 1])
    n("Div", ["csm1", "two_f"], "COL")
    vi("COL", TensorProto.FLOAT, [1, 1, 1, 1])

    n("Add", ["ROW", "one_f"], "ROW1")
    vi("ROW1", TensorProto.FLOAT, [1, 1, 1, 1])
    n("Add", ["COL", "one_f"], "COL1")
    vi("COL1", TensorProto.FLOAT, [1, 1, 1, 1])

    # =====================================================================
    # 3. recover the four colours as channel one-hots [1,10,1,1]
    # =====================================================================
    n("Equal", ["RIDX", "ROW"], "rTop_b")
    vi("rTop_b", TensorProto.BOOL, [1, 1, 30, 1])
    n("Cast", ["rTop_b"], "rTop", to=TensorProto.FLOAT)
    vi("rTop", TensorProto.FLOAT, [1, 1, 30, 1])
    n("Equal", ["RIDX", "ROW1"], "rBot_b")
    vi("rBot_b", TensorProto.BOOL, [1, 1, 30, 1])
    n("Cast", ["rBot_b"], "rBot", to=TensorProto.FLOAT)
    vi("rBot", TensorProto.FLOAT, [1, 1, 30, 1])
    n("Equal", ["CIDX", "COL"], "cLft_b")
    vi("cLft_b", TensorProto.BOOL, [1, 1, 1, 30])
    n("Cast", ["cLft_b"], "cLft", to=TensorProto.FLOAT)
    vi("cLft", TensorProto.FLOAT, [1, 1, 1, 30])
    n("Equal", ["CIDX", "COL1"], "cRgt_b")
    vi("cRgt_b", TensorProto.BOOL, [1, 1, 1, 30])
    n("Cast", ["cRgt_b"], "cRgt", to=TensorProto.FLOAT)
    vi("cRgt", TensorProto.FLOAT, [1, 1, 1, 30])

    n("Mul", ["rowKc", "rTop"], "rk_top")
    vi("rk_top", TensorProto.FLOAT, [1, 10, 30, 1])
    n("ReduceSum", ["rk_top"], "setTop", axes=[2], keepdims=1)   # {c0,c1}
    vi("setTop", TensorProto.FLOAT, [1, 10, 1, 1])
    n("Mul", ["rowKc", "rBot"], "rk_bot")
    vi("rk_bot", TensorProto.FLOAT, [1, 10, 30, 1])
    n("ReduceSum", ["rk_bot"], "setBot", axes=[2], keepdims=1)   # {c2,c3}
    vi("setBot", TensorProto.FLOAT, [1, 10, 1, 1])

    n("Mul", ["colKc", "cLft"], "ck_lft")
    vi("ck_lft", TensorProto.FLOAT, [1, 10, 1, 30])
    n("ReduceSum", ["ck_lft"], "setLft", axes=[3], keepdims=1)   # {c0,c2}
    vi("setLft", TensorProto.FLOAT, [1, 10, 1, 1])
    n("Mul", ["colKc", "cRgt"], "ck_rgt")
    vi("ck_rgt", TensorProto.FLOAT, [1, 10, 1, 30])
    n("ReduceSum", ["ck_rgt"], "setRgt", axes=[3], keepdims=1)   # {c1,c3}
    vi("setRgt", TensorProto.FLOAT, [1, 10, 1, 1])

    n("Mul", ["setTop", "setLft"], "c0oh")
    vi("c0oh", TensorProto.FLOAT, [1, 10, 1, 1])
    n("Mul", ["setTop", "setRgt"], "c1oh")
    vi("c1oh", TensorProto.FLOAT, [1, 10, 1, 1])
    n("Mul", ["setBot", "setLft"], "c2oh")
    vi("c2oh", TensorProto.FLOAT, [1, 10, 1, 1])
    n("Mul", ["setBot", "setRgt"], "c3oh")
    vi("c3oh", TensorProto.FLOAT, [1, 10, 1, 1])

    kidx = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("KIDX", kidx, np.float32)
    for name in ["c0", "c1", "c2", "c3"]:
        n("Mul", [name + "oh", "KIDX"], name + "_kw")
        vi(name + "_kw", TensorProto.FLOAT, [1, 10, 1, 1])
        n("ReduceSum", [name + "_kw"], name + "idx", axes=[1], keepdims=1)
        vi(name + "idx", TensorProto.FLOAT, [1, 1, 1, 1])

    # =====================================================================
    # 4. build the 6x6 label canvas from separable region rectangles
    # =====================================================================
    r6 = np.arange(6, dtype=np.float32).reshape(1, 1, 6, 1)
    c6 = np.arange(6, dtype=np.float32).reshape(1, 1, 1, 6)
    init("R6", r6, np.float32)
    init("C6", c6, np.float32)
    init("half_f", np.array(0.5, np.float32), np.float32)
    init("p2_f", np.array(2.0, np.float32), np.float32)
    init("p3_f", np.array(3.0, np.float32), np.float32)
    init("m1_f", np.array(1.0, np.float32), np.float32)
    init("m2_f", np.array(2.0, np.float32), np.float32)

    def rangemask(idx_plane, lo_scalar, hi_scalar, tag, shape):
        n("Sub", [lo_scalar, "half_f"], tag + "_lo")
        vi(tag + "_lo", TensorProto.FLOAT, [1, 1, 1, 1])
        n("Add", [hi_scalar, "half_f"], tag + "_hi")
        vi(tag + "_hi", TensorProto.FLOAT, [1, 1, 1, 1])
        n("Greater", [idx_plane, tag + "_lo"], tag + "_g")
        vi(tag + "_g", TensorProto.BOOL, shape)
        n("Less", [idx_plane, tag + "_hi"], tag + "_l")
        vi(tag + "_l", TensorProto.BOOL, shape)
        n("And", [tag + "_g", tag + "_l"], tag + "_b")
        vi(tag + "_b", TensorProto.BOOL, shape)
        n("Cast", [tag + "_b"], tag + "_f", to=TensorProto.FLOAT)
        vi(tag + "_f", TensorProto.FLOAT, shape)
        return tag + "_f"

    rshape = [1, 1, 6, 1]
    cshape = [1, 1, 1, 6]

    # stamp row/col ranges (each 2 wide)
    n("Add", ["ROW", "p2_f"], "rP2")
    vi("rP2", TensorProto.FLOAT, [1, 1, 1, 1])
    n("Add", ["ROW", "p3_f"], "rP3")
    vi("rP3", TensorProto.FLOAT, [1, 1, 1, 1])
    rPlus = rangemask("R6", "rP2", "rP3", "rPlus", rshape)
    n("Sub", ["ROW", "m2_f"], "rM2")
    vi("rM2", TensorProto.FLOAT, [1, 1, 1, 1])
    n("Sub", ["ROW", "m1_f"], "rM1")
    vi("rM1", TensorProto.FLOAT, [1, 1, 1, 1])
    rMinus = rangemask("R6", "rM2", "rM1", "rMinus", rshape)
    n("Add", ["COL", "p2_f"], "cP2")
    vi("cP2", TensorProto.FLOAT, [1, 1, 1, 1])
    n("Add", ["COL", "p3_f"], "cP3")
    vi("cP3", TensorProto.FLOAT, [1, 1, 1, 1])
    cPlus = rangemask("C6", "cP2", "cP3", "cPlus", cshape)
    n("Sub", ["COL", "m2_f"], "cM2")
    vi("cM2", TensorProto.FLOAT, [1, 1, 1, 1])
    n("Sub", ["COL", "m1_f"], "cM1")
    vi("cM1", TensorProto.FLOAT, [1, 1, 1, 1])
    cMinus = rangemask("C6", "cM2", "cM1", "cMinus", cshape)

    # single-row / single-col masks for the 4 center cells
    rTop_s = rangemask("R6", "ROW", "ROW", "rTopS", rshape)
    rBot_s = rangemask("R6", "ROW1", "ROW1", "rBotS", rshape)
    cLft_s = rangemask("C6", "COL", "COL", "cLftS", cshape)
    cRgt_s = rangemask("C6", "COL1", "COL1", "cRgtS", cshape)

    def rect(rmask, cmask, tag):
        n("Mul", [rmask, cmask], tag)
        vi(tag, TensorProto.FLOAT, [1, 1, 6, 6])
        return tag

    blk00 = rect(rTop_s, cLft_s, "blk00")   # c0
    blk01 = rect(rTop_s, cRgt_s, "blk01")   # c1
    blk10 = rect(rBot_s, cLft_s, "blk10")   # c2
    blk11 = rect(rBot_s, cRgt_s, "blk11")   # c3
    stp0 = rect(rPlus, cPlus, "stp0")       # c0
    stp1 = rect(rPlus, cMinus, "stp1")      # c1
    stp2 = rect(rMinus, cPlus, "stp2")      # c2
    stp3 = rect(rMinus, cMinus, "stp3")     # c3

    terms = [(blk00, "c0idx"), (blk01, "c1idx"), (blk10, "c2idx"),
             (blk11, "c3idx"), (stp0, "c0idx"), (stp1, "c1idx"),
             (stp2, "c2idx"), (stp3, "c3idx")]
    acc = None
    for i, (reg, cidx_name) in enumerate(terms):
        tname = "term%d" % i
        n("Mul", [reg, cidx_name], tname)
        vi(tname, TensorProto.FLOAT, [1, 1, 6, 6])
        if acc is None:
            acc = tname
        else:
            aname = "acc%d" % i
            n("Add", [acc, tname], aname)
            vi(aname, TensorProto.FLOAT, [1, 1, 6, 6])
            acc = aname

    # =====================================================================
    # 5. Pad to 30x30 with sentinel 10, Cast uint8, Equal vs arange -> output
    # =====================================================================
    pads = np.array([0, 0, 0, 0, 0, 0, 24, 24], dtype=np.int64)
    init("PADS", pads, np.int64)
    init("pad10", np.array(10.0, np.float32), np.float32)
    n("Pad", [acc, "PADS", "pad10"], "L30f", mode="constant")
    vi("L30f", TensorProto.FLOAT, [1, 1, 30, 30])
    n("Cast", ["L30f"], "L30", to=TensorProto.UINT8)
    vi("L30", TensorProto.UINT8, [1, 1, 30, 30])

    arange = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("ARANGE", arange, np.uint8)
    n("Equal", ["L30", "ARANGE"], "output")

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task225", [x], [y], inits, value_info=vinfos)
    model = helper.make_model(
        graph, ir_version=10,
        opset_imports=[helper.make_opsetid("", 11)])
    return model
