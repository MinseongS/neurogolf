"""Task 013 (0a938d79): periodic two-color column (or row) stripes from 2 seeds.

Rule (ARC-GEN generator): two seed dots mark two colors. Starting at index
`start` along the stripe axis, every (sep+1)-th line (full perpendicular extent)
is painted, alternating colors[0], colors[1], colors[0], ...  The seed at the
smaller index is colors[0]; the next (start+sep+1) is colors[1].  `xpose` may
transpose rows<->cols.

Construction (fully separable, one Mul into `output`):
  output[ch,r,c] = Urow[ch,r] * Vcol[ch,c]    ([1,10,30,1] * [1,10,1,30])

xpose detection: dr = |seed row diff|, H = grid height.  xpose=0 iff
dr==0 or dr==H-1 (seeds share a row or span full height); else row separation
is the small step -> xpose=1.  (step<=6 << H-1>=19 when transposed.)
We select the canonical "stripe axis" seeds / in-grid mask up front (Where on
1-D tensors), run the stripe builder ONCE, then route the colored stripe S and
the plain perpendicular mask to Urow/Vcol per orientation.

Stripe builder along an axis (index k=0..29), per channel:
  - color channels (1..9): painted iff used[ch] and k>=pos[ch] and
    (k-pos[ch]) % P == 0 and in-grid, P = 2*step, step = (2nd-1st seed index).
  - channel 0 (background): in-grid AND not any-stripe.

Memory floor-break: all the [10,30] / [1,30] arithmetic planes run in fp16
(indices 0..29, deltas -29..29 and the Mod results are all exact in fp16), so
each big plane is 600 B instead of 1200 B; the per-element Mod runs directly in
fp16 (no int32 cast planes), and the final separable `Mul` writes into the free
`output`.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype=np.float16):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    # ---- constants (fp16 unless noted) ----
    init("halff", np.array(0.5, np.float16))
    init("onef", np.array(1.0, np.float16))
    init("neghalf", np.array(-0.5, np.float16))
    init("sh30", np.array([30], np.int64), np.int64)
    init("sh1030", np.array([10, 30], np.int64), np.int64)
    init("sh130", np.array([1, 30], np.int64), np.int64)
    init("ar30f_col", np.arange(30, dtype=np.float16).reshape(30, 1))   # [30,1]
    init("ar30row", np.arange(30, dtype=np.float16).reshape(1, 30))     # [1,30]
    init("ar30f", np.arange(30, dtype=np.float16))                      # [30]
    init("bigvec", np.full(30, 1e4, np.float16))
    init("negvec", np.full(30, -1.0, np.float16))
    init("big_col", np.full((10, 1), 1e4, np.float16))
    init("neg_col", np.full((10, 1), -1.0, np.float16))
    init("two_s", np.array([[2.0]], np.float16))
    init("one_col", np.array([[1.0]], np.float16))
    ch_keepb = np.ones((10, 1), bool); ch_keepb[0, 0] = False
    init("ch_keepb", ch_keepb, bool)
    ch0_onlyb = np.zeros((10, 1), bool); ch0_onlyb[0, 0] = True
    init("ch0_onlyb", ch0_onlyb, bool)
    chkeepb4 = np.ones((1, 10, 1), bool); chkeepb4[0, 0, 0] = False
    init("chkeepb4", chkeepb4, bool)

    # ---- in-grid masks (input is fp32; reduce then Cast to fp16) ----
    n("ReduceSum", ["input"], "colsum32", axes=[1, 2], keepdims=0)  # [1,30] f32
    n("ReduceSum", ["input"], "rowsum32", axes=[1, 3], keepdims=0)  # [1,30] f32
    n("Cast", ["colsum32"], "colsum", to=F16)
    n("Cast", ["rowsum32"], "rowsum", to=F16)
    n("Reshape", ["colsum", "sh30"], "colsum1")                    # [30]
    n("Reshape", ["rowsum", "sh30"], "rowsum1")                    # [30]
    n("Greater", ["colsum1", "halff"], "colmask_b")               # [30] bool
    n("Greater", ["rowsum1", "halff"], "rowmask_b")               # [30] bool
    n("Cast", ["colmask_b"], "colmask", to=F16)                   # [30]
    n("Cast", ["rowmask_b"], "rowmask", to=F16)                   # [30]
    n("ReduceMax", ["colsum1"], "H_f", axes=[0], keepdims=0)      # scalar H

    # ---- seed indicators per channel (exclude background channel 0) ----
    n("ReduceMax", ["input"], "cvcol32", axes=[2], keepdims=0)    # [1,10,30] f32
    n("ReduceMax", ["input"], "cvrow32", axes=[3], keepdims=0)    # [1,10,30] f32
    n("Greater", ["cvcol32", "halff32"], "cvcolb4")               # [1,10,30] bool
    n("Greater", ["cvrow32", "halff32"], "cvrowb4")               # [1,10,30] bool
    init("halff32", np.array(0.5, np.float32), np.float32)
    n("And", ["cvcolb4", "chkeepb4"], "scolb4")                   # zero ch0 (bool)
    n("And", ["cvrowb4", "chkeepb4"], "srowb4")                   # zero ch0 (bool)
    n("Reshape", ["scolb4", "sh1030"], "scolb")                   # [10,30] bool
    n("Reshape", ["srowb4", "sh1030"], "srowb")                   # [10,30] bool

    # ---- xpose detection: dr = |seed row diff|, isx0 = dr==0 or dr==H-1 ----
    n("Cast", ["srowb"], "srowf", to=F16)                         # [10,30] f16
    n("ReduceMax", ["srowf"], "rowseed", axes=[0], keepdims=0)    # [30] f16
    n("Cast", ["rowseed"], "rowseed_b", to=BOOL)
    n("Where", ["rowseed_b", "ar30f", "bigvec"], "rs_lo")
    n("ReduceMin", ["rs_lo"], "minrow", axes=[0], keepdims=0)     # scalar
    n("Where", ["rowseed_b", "ar30f", "negvec"], "rs_hi")
    n("ReduceMax", ["rs_hi"], "maxrow", axes=[0], keepdims=0)     # scalar
    n("Sub", ["maxrow", "minrow"], "dr")                          # scalar
    n("Sub", ["H_f", "onef"], "Hm1")
    n("Cast", ["dr"], "dr_i", to=TensorProto.INT32)
    n("Cast", ["Hm1"], "Hm1_i", to=TensorProto.INT32)
    init("c0i", np.array(0, np.int32), np.int32)
    n("Equal", ["dr_i", "c0i"], "dr_is0")
    n("Equal", ["dr_i", "Hm1_i"], "dr_isH")
    n("Or", ["dr_is0", "dr_isH"], "isx0_b")                       # scalar bool

    # ---- choose canonical stripe-axis seeds (fp16) + in-grid mask ----
    n("Cast", ["scolb"], "scolf", to=F16)                         # [10,30] f16
    n("Where", ["isx0_b", "scolf", "srowf"], "seeds")            # [10,30] f16
    n("Where", ["isx0_b", "colmask", "rowmask"], "axmask")       # [30]

    # ---- stripe builder (run once) ----
    n("MatMul", ["seeds", "ar30f_col"], "pos")                   # [10,1]
    n("ReduceSum", ["seeds"], "cnt", axes=[1], keepdims=1)       # [10,1]
    n("Greater", ["cnt", "halff"], "used_b")                     # [10,1] bool
    n("Where", ["used_b", "pos", "big_col"], "plo")
    n("ReduceMin", ["plo"], "minp", axes=[0], keepdims=1)        # [1,1]
    n("Where", ["used_b", "pos", "neg_col"], "phi")
    n("ReduceMax", ["phi"], "maxp", axes=[0], keepdims=1)        # [1,1]
    n("Sub", ["maxp", "minp"], "step")
    n("Mul", ["step", "two_s"], "P")
    n("Max", ["P", "one_col"], "Ps")                             # clamp >=1
    n("Sub", ["ar30row", "pos"], "delta")                        # [10,30] f16
    n("Mod", ["delta", "Ps"], "modr", fmod=1)                    # [10,30] f16
    init("ahalf", np.array(0.5, np.float16))
    n("Less", ["modr", "ahalf"], "onmod_b")                      # modr==0 bool
    n("Greater", ["delta", "neghalf"], "fwd_b")                  # delta>=0 bool
    n("And", ["onmod_b", "fwd_b"], "mf_b")                       # [10,30] bool
    n("And", ["mf_b", "used_b"], "m_b")                          # & used [10,30]
    n("Reshape", ["axmask", "sh130"], "axrow")                   # [1,30] f16
    n("Cast", ["axrow"], "axrow_b", to=BOOL)                     # [1,30] bool
    n("And", ["m_b", "axrow_b"], "colorpat_b")                   # [10,30] stripes

    # any-stripe column (either color): spacing = step (= maxp-minp)
    n("Max", ["step", "one_col"], "step_c")                      # clamp >=1
    n("Sub", ["ar30row", "minp"], "dany")                        # [1,30] f16
    n("Mod", ["dany", "step_c"], "danym", fmod=1)                # [1,30] f16
    n("Less", ["danym", "ahalf"], "danym0_b")                    # danym==0
    n("Greater", ["dany", "neghalf"], "danyf_b")
    n("And", ["danym0_b", "danyf_b"], "stcol_b")                 # [1,30] bool
    n("And", ["stcol_b", "axrow_b"], "anyst_b")                  # in-grid
    n("Not", ["anyst_b"], "notany_b")                            # [1,30] bool
    n("And", ["notany_b", "axrow_b"], "bg_b")                    # [1,30] background

    # assemble S[10,30] bool: ch0 = bg, ch1..9 = colorpat
    n("And", ["colorpat_b", "ch_keepb"], "colorz")               # zero ch0
    n("And", ["bg_b", "ch0_onlyb"], "bgch")                      # only ch0
    n("Or", ["colorz", "bgch"], "S_b")                           # [10,30] bool
    n("Cast", ["S_b"], "S", to=F16)                              # [10,30] f16

    # ---- perpendicular plain mask (fp16) ----
    n("Where", ["isx0_b", "rowmask", "colmask"], "plain")        # [30] f16

    # ---- route to Urow / Vcol per orientation ----
    n("Where", ["isx0_b", "plain", "S"], "Urow2d")               # [10,30] (bcast)
    n("Where", ["isx0_b", "S", "plain"], "Vcol2d")               # [10,30]
    init("shU", np.array([1, 10, 30, 1], np.int64), np.int64)
    init("shV", np.array([1, 10, 1, 30], np.int64), np.int64)
    n("Reshape", ["Urow2d", "shU"], "Urow")
    n("Reshape", ["Vcol2d", "shV"], "Vcol")
    n("Mul", ["Urow", "Vcol"], "output")                         # [1,10,30,30] free

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F16, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task013", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
