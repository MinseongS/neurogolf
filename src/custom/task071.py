"""task071 (ARC-AGI 3345333e) — mirror-complete the sprite, delete the box.

Rule (verified 0/3000 fresh):
  The 16x16 grid holds two colours.  One ("sprite", colors[0]) is a continuous
  creature drawn MIRROR-SYMMETRICALLY about a vertical axis (every pixel placed
  at both scol+c and scol-c-off => axis2 = 2*scol-off).  The other ("box",
  colors[1]) is a SOLID width-4 rectangle drawn ON TOP, occluding part of the
  sprite.  Output = the sprite mirror-completed (sprite OR reflect(sprite) about
  the axis), with the box removed entirely.

  Box channel = the non-bg channel that is SOLID (cnt == bboxW*bboxH) AND has
  bboxW == 4 (verified 0/5000).  Sprite channel = the other non-bg channel.

  Axis recovery (occlusion-robust, no flood-fill):
    valid = sprite OR box  (box fills the occluded mirror cells).
    For candidate axes a, reflect via R_a[k,c]=(k+c==a):
      overlap_a = sum( sprite * (valid @ R_a) ).
    The true axis is the UNIQUE one where every visible sprite cell has an
    in-grid mirror landing on a valid cell => overlap_a is maximal.
    axis = argmax_a overlap_a  (verified unique 0/2000).
  Reconstruct: outmask = sprite OR (sprite @ R_axis); colour = sprite colour.

Memory: harness pads grids to 30x30; the generator's active grid is always the
16x16 top-left corner.  All heavy planes are cropped to WK=16x16 (and the axis
stack to [NA,16,16]); the final colour-index plane is Pad-ed back to 30x30 with
a 99 sentinel (off-grid -> NO channel on, matching the all-zero target padding).
Output one-hot via Equal(L, arange10) -> BOOL into the FREE output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
I64 = TensorProto.INT64
U8 = TensorProto.UINT8
B = TensorProto.BOOL

WK = 16                          # active working canvas (generator size is fixed 16)
PAD = 30 - WK                    # padding back to the 30x30 harness canvas
AXES = np.arange(9, 21)          # candidate symmetry axes; fresh axis2 in [10,20],
                                 # stored (validate) examples reach 9 -> cover [9,20]
NA = len(AXES)
BIG = 1000.0


def build(task):
    inits, nodes = [], []
    np_of = {F32: np.float32, I64: np.int64, F16: np.float16, U8: np.uint8}

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=np_of[dtype]), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- constants -----------------------------------------------------------
    init("colidx", np.arange(10).reshape(1, 10, 1, 1), F32)
    init("colramp", np.arange(WK).reshape(1, 1, 1, WK), F32)        # active-region ramp
    init("rowramp", np.arange(WK).reshape(1, 1, WK, 1), F32)

    init("one", np.ones((1, 1, 1, 1), np.float32), F32)
    init("zero", np.array(0.0, np.float32).reshape(1, 1, 1, 1), F32)
    init("BIGc", np.array(BIG, np.float32).reshape(1, 1, 1, 1), F32)
    init("four", np.array(4.0, np.float32).reshape(1, 1, 1, 1), F32)

    # ---- per-channel scalars (operate on FREE full input directly) -----------
    n("ReduceSum", ["input"], "cnt", axes=[2, 3], keepdims=1)        # [1,10,1,1]
    n("ReduceMax", ["input"], "rowocc30", axes=[3], keepdims=1)      # [1,10,30,1]
    n("ReduceMax", ["input"], "colocc30", axes=[2], keepdims=1)      # [1,10,1,30]
    # crop occupancy profiles to the active WK extent (cheaper ramp math)
    init("c0a", np.array([0], np.int64), I64)
    init("cWKa", np.array([WK], np.int64), I64)
    init("ax2", np.array([2], np.int64), I64)
    init("ax3", np.array([3], np.int64), I64)
    n("Slice", ["rowocc30", "c0a", "cWKa", "ax2"], "rowocc")         # [1,10,WK,1]
    n("Slice", ["colocc30", "c0a", "cWKa", "ax3"], "colocc")         # [1,10,1,WK]

    # column extent -> width
    n("Mul", ["colocc", "colramp"], "colpos")
    n("ReduceMax", ["colpos"], "cmax", axes=[3], keepdims=1)
    n("Sub", ["one", "colocc"], "ncolocc")
    n("Mul", ["ncolocc", "BIGc"], "ncolBIG")
    n("Add", ["colpos", "ncolBIG"], "cminpos")
    n("ReduceMin", ["cminpos"], "cmin", axes=[3], keepdims=1)
    n("Sub", ["cmax", "cmin"], "wm1")
    n("Add", ["wm1", "one"], "width")

    # row extent -> height
    n("Mul", ["rowocc", "rowramp"], "rowpos")
    n("ReduceMax", ["rowpos"], "rmax", axes=[2], keepdims=1)
    n("Sub", ["one", "rowocc"], "nrowocc")
    n("Mul", ["nrowocc", "BIGc"], "nrowBIG")
    n("Add", ["rowpos", "nrowBIG"], "rminpos")
    n("ReduceMin", ["rminpos"], "rmin", axes=[2], keepdims=1)
    n("Sub", ["rmax", "rmin"], "hm1")
    n("Add", ["hm1", "one"], "height")

    # ---- box channel: solid AND width==4 -------------------------------------
    n("Mul", ["width", "height"], "area")
    n("Equal", ["cnt", "area"], "issolid")
    n("Equal", ["width", "four"], "isw4")
    n("And", ["issolid", "isw4"], "boxsel_b")
    n("Greater", ["cnt", "zero"], "haspx")
    n("And", ["boxsel_b", "haspx"], "boxsel_bb")
    n("Cast", ["boxsel_bb"], "boxsel", to=F32)                       # [1,10,1,1]

    # sprite channel: has pixels AND not box AND not background (ch0)
    n("Not", ["boxsel_bb"], "notbox")
    n("And", ["haspx", "notbox"], "spsel_b0")
    init("nonbg", np.array([0] + [1] * 9).reshape(1, 10, 1, 1), F32)
    n("Cast", ["nonbg"], "nonbg_b", to=B)
    n("And", ["spsel_b0", "nonbg_b"], "spsel_b")
    n("Cast", ["spsel_b"], "spsel", to=F32)

    # ---- colour-index plane via 1x1 Conv (no [1,10,*] product), then crop ----
    # colf30[1,1,30,30] = sum_k k * input_k ; Conv weight [1,10,1,1] = colidx.
    n("Conv", ["input", "colidx"], "colf30")                        # [1,1,30,30]
    init("c0", np.array([0, 0], np.int64), I64)
    init("cWK", np.array([WK, WK], np.int64), I64)
    init("ax23", np.array([2, 3], np.int64), I64)
    n("Slice", ["colf30", "c0", "cWK", "ax23"], "colf")             # [1,1,WK,WK]

    # sprite / box colour scalars
    n("Mul", ["spsel", "colidx"], "spc_v")
    n("ReduceSum", ["spc_v"], "spcolor", axes=[1], keepdims=1)      # [1,1,1,1]
    n("Mul", ["boxsel", "colidx"], "bxc_v")
    n("ReduceSum", ["bxc_v"], "boxcolor", axes=[1], keepdims=1)     # [1,1,1,1]

    # Isp / valid masks from the index plane
    n("Equal", ["colf", "spcolor"], "Isp_b")                        # [1,1,WK,WK]
    n("Equal", ["colf", "boxcolor"], "Ibox_b")
    n("Or", ["Isp_b", "Ibox_b"], "valid_b")
    n("Cast", ["Isp_b"], "Isp32", to=F32)
    n("Cast", ["valid_b"], "valid32", to=F32)
    init("WWshape", np.array([WK, WK], np.int64), I64)
    n("Reshape", ["valid32", "WWshape"], "valid2")                  # [WK,WK]
    n("Reshape", ["Isp32", "WWshape"], "Isp2")                      # [WK,WK]

    # ---- axis via column-coincidence anti-diagonal sums (NO [NA,WK,WK]) -------
    # overlap_a = sum_{c+c'=a} C[c,c'], C[c,c'] = sum_r Isp[r,c]*valid[r,c'].
    n("Transpose", ["Isp2"], "IspT", perm=[1, 0])                   # [WK,WK]
    n("MatMul", ["IspT", "valid2"], "C")                            # [WK,WK]  C[c,c']
    init("Cflatshape", np.array([1, WK * WK], np.int64), I64)
    n("Reshape", ["C", "Cflatshape"], "Cflat")                      # [1, WK*WK]
    # D[flat(c,c'), a] = (c+c' == AXES[a])  -> overlap = Cflat @ D  [1, NA]
    D = np.zeros((WK * WK, NA), np.float32)
    for c in range(WK):
        for cp in range(WK):
            s = c + cp
            for a in range(NA):
                if AXES[a] == s:
                    D[c * WK + cp, a] = 1.0
    init("Dmat", D, F32)
    n("MatMul", ["Cflat", "Dmat"], "overlap")                      # [1, NA]
    n("ArgMax", ["overlap"], "axis_idx", axis=1, keepdims=0)       # [1] int64
    # map index -> actual axis2 value = AXES[axis_idx]; AXES is contiguous so
    # axis2 = axis_idx + AXES[0]
    n("Cast", ["axis_idx"], "axis_f", to=F32)
    init("axoff", np.array(float(AXES[0])).reshape(1), F32)
    n("Add", ["axis_f", "axoff"], "axis2")                         # [1] = axis2

    # Rsel[k,c] = (k + c == axis2)  built directly from the scalar (no stack)
    KC = (np.arange(WK)[:, None] + np.arange(WK)[None, :]).astype(np.float32)
    init("KCsum", KC, F32)                                          # [WK,WK]
    init("shp11", np.array([1, 1], np.int64), I64)
    n("Reshape", ["axis2", "shp11"], "axis2_2")                     # [1,1]
    n("Equal", ["KCsum", "axis2_2"], "Rsel_b")                      # [WK,WK]
    n("Cast", ["Rsel_b"], "Rsel", to=F32)

    # ---- reconstruct ---------------------------------------------------------
    n("MatMul", ["Isp2", "Rsel"], "mirror2")                       # [WK,WK]
    n("Add", ["Isp2", "mirror2"], "uni2")
    n("Greater", ["uni2", "zero"], "outmask_b")                    # [WK,WK]
    n("Cast", ["outmask_b"], "outmask", to=F32)
    init("shp11WW", np.array([1, 1, WK, WK], np.int64), I64)
    n("Reshape", ["outmask", "shp11WW"], "outmask4")                 # [1,1,WK,WK]

    # L = sprite colour where outmask else 0  (in-grid bg -> ch0 True via 0)
    n("Mul", ["outmask4", "spcolor"], "Lk")                          # [1,1,WK,WK] f32
    n("Cast", ["Lk"], "Lk8", to=U8)                                  # uint8 (1B/elem)

    # Pad back to 30x30 with sentinel 99 (off-grid -> Equal all-False -> all-zero)
    init("padval", np.array(99, np.uint8), U8)
    init("pads", np.array([0, 0, 0, 0, 0, 0, PAD, PAD], np.int64), I64)
    n("Pad", ["Lk8", "pads", "padval"], "L")                        # [1,1,30,30] uint8
    # uint8 one-hot: Equal(L, arange10_u8) -> [1,10,30,30] BOOL (free output)
    init("arange10u8", np.arange(10).reshape(1, 10, 1, 1), U8)
    n("Equal", ["L", "arange10u8"], "output")

    graph = helper.make_graph(
        nodes, "task071",
        [helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", B, [1, 10, 30, 30])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
