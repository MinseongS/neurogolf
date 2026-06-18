"""Task 253 (ARC-AGI a61ba2ce): four L-trominoes -> fixed 4x4 corner layout.

Rule (from generator, verified fresh):
  A 13x13 input contains exactly four L-trominoes (three cells of one solid
  colour each), one per colour.  Each L is a 2x2 box with ONE corner missing,
  and every instance uses all four distinct missing-corner orientations:
      missing BR  (cells TL,TR,BL present)
      missing BL  (cells TL,TR,BR present)
      missing TR  (cells TL,BL,BR present)
      missing TL  (cells TR,BL,BR present)
  The 4x4 output is a FIXED layout; only the four colours vary.  Each colour is
  placed in the output corner DIAGONALLY OPPOSITE to its missing corner:
      missing BR -> output TL corner   (cells (0,0),(0,1),(1,0))
      missing BL -> output TR corner   (cells (0,2),(0,3),(1,3))
      missing TR -> output BL corner   (cells (2,0),(3,0),(3,1))
      missing TL -> output BR corner   (cells (2,3),(3,2),(3,3))
  The output L's mirror the input orientations (each fills its corner's L).

Encoding (Tier B, tiny intermediates):
  1. colf30 = Conv(input, w=k) -> [1,1,30,30] fp32 colour-index plane (the one
     allowed fp32 entry).  Slice to the active 14x14 region, cast to fp16.
  2. occ = (cf>0) fp16.  Four no-pad 2x2 convs on occ each yield value 3 only
     at the top-left of the matching orientation's box (kernel = pattern of the
     3 present cells); Equal(.,3) gives a one-hot box-TL indicator per
     orientation [1,1,13,13] fp16.
  3. Each orientation's colour = ReduceSum(indicator * colour-at-a-present-cell).
     For BR/BL/TR the box-TL cell is present so multiply by cf directly; for TL
     the box-TL cell is empty so read box-TR (cf shifted one col left).  Sum is
     the bare colour scalar (indicator is a single 1).
  4. Build the fixed 4x4 colour-index label map by scattering the four scalars
     into their L positions via constant masks, Pad to 30x30 with sentinel 10,
     and Equal(L, arange[0..9]) writes straight into the FREE BOOL output.

  Dominant intermediate: colf30 (fp32 [1,1,30,30] = 3600 B) -- irreducible 10->1
  colour reduction.  Everything downstream is fp16 14x14 (392 B) or smaller.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

W = 14  # active working region (markers live in 0..12)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    f16 = TensorProto.FLOAT16

    # 1) colour-index plane: Conv(input[1,10,30,30], w[1,10,1,1] = k) -> [1,1,30,30]
    kw = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("cw", kw, np.float32)
    n("Conv", ["input", "cw"], "colf30")  # [1,1,30,30] fp32

    # slice to active 14x14 and cast to fp16
    init("s0", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("s1", np.array([1, 1, W, W], np.int64), np.int64)
    init("sax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["colf30", "s0", "s1", "sax"], "cf32")
    n("Cast", ["cf32"], "cf", to=f16)  # [1,1,14,14] fp16

    # occupancy fp16
    init("zero16", np.array(0.0, np.float16), np.float16)
    n("Greater", ["cf", "zero16"], "occ_b")
    n("Cast", ["occ_b"], "occ", to=f16)  # [1,1,14,14] fp16 {0,1}

    # 2) ONE 4-channel orientation conv on occ: out[1,4,13,13]; out[o] = sum of
    # occ over orientation o's 3 present cells, == 3 only at the box top-left of
    # an exact L of that orientation.  Channel order: BR, BL, TR, TL.
    orient = ["BR", "BL", "TR", "TL"]
    pat = {
        "BR": [[1, 1], [1, 0]],   # missing BR
        "BL": [[1, 1], [0, 1]],   # missing BL
        "TR": [[1, 0], [1, 1]],   # missing TR
        "TL": [[0, 1], [1, 1]],   # missing TL
    }
    ker = np.stack([np.array(pat[o], np.float16) for o in orient])[:, None]  # [4,1,2,2]
    init("oker", ker, np.float16)
    n("Conv", ["occ", "oker"], "resp")  # [1,4,13,13] fp16
    init("three16", np.array(3.0, np.float16), np.float16)
    n("Equal", ["resp", "three16"], "ind_b")  # [1,4,13,13] bool box-TL one-hots
    n("Cast", ["ind_b"], "ind", to=f16)

    # 3) colour scalar per orientation.  cfsum = sum of cf over each 2x2 box =
    # 3*colour at an L box (missing cell is 0).  color_o = sum(cfsum*ind_o)/3.
    init("ones2", np.ones((1, 1, 2, 2), np.float16), np.float16)
    n("Conv", ["cf", "ones2"], "cfsum")  # [1,1,13,13] fp16, = 3*colour at L boxes
    n("Mul", ["ind", "cfsum"], "prod")    # [1,4,13,13] broadcast
    n("ReduceSum", ["prod"], "colsum", axes=[2, 3], keepdims=1)  # [1,4,1,1] = 3*colour
    init("third", np.array(1.0 / 3.0, np.float16), np.float16)
    n("Mul", ["colsum", "third"], "colors")  # [1,4,1,1] colour index per orientation

    # 4) scatter the four colours into the fixed 4x4 layout.
    # build a [1,4,4,4] mask M[o] marking orientation o's output L cells; the
    # label map = ReduceSum over the orientation axis of colors*M.
    corner_cells = {
        "BR": [(0, 0), (0, 1), (1, 0)],            # output TL
        "BL": [(0, 2), (0, 3), (1, 3)],            # output TR
        "TR": [(2, 0), (3, 0), (3, 1)],            # output BL
        "TL": [(2, 3), (3, 2), (3, 3)],            # output BR
    }
    M = np.zeros((1, 4, 4, 4), np.float16)
    for i, o in enumerate(orient):
        for (r, c) in corner_cells[o]:
            M[0, i, r, c] = 1.0
    init("scatM", M, np.float16)
    n("Mul", ["colors", "scatM"], "labelled")  # [1,4,4,4]
    n("ReduceSum", ["labelled"], "L4", axes=[1], keepdims=1)  # [1,1,4,4] colour-index

    # pad to 30x30 with sentinel 10 (off-grid -> all channels 0), stay fp16
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - 4, 30 - 4], np.int64), np.int64)
    init("sent", np.array(10.0, np.float16), np.float16)
    n("Pad", ["L4", "pads", "sent"], "L30", mode="constant")  # [1,1,30,30] fp16

    # final Equal against colour ramp (fp16, exact for small ints) -> BOOL output
    init("ramp", np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1), np.float16)
    n("Equal", ["L30", "ramp"], "output")

    graph = helper.make_graph(
        nodes, "task253",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
