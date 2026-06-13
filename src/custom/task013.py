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
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ..builders import _model

I32 = onnx.TensorProto.INT32
F32 = onnx.TensorProto.FLOAT
BOOL = onnx.TensorProto.BOOL


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype=np.float32):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    # ---- constants ----
    init("halff", np.array(0.5, np.float32))
    init("onef", np.array(1.0, np.float32))
    init("neghalf", np.array(-0.5, np.float32))
    init("c0i", np.array(0, np.int32), np.int32)
    init("sh30", np.array([30], np.int64), np.int64)
    init("sh1030", np.array([10, 30], np.int64), np.int64)
    init("sh130", np.array([1, 30], np.int64), np.int64)
    init("ar30f_col", np.arange(30, dtype=np.float32).reshape(30, 1))   # [30,1]
    init("ar30row", np.arange(30, dtype=np.float32).reshape(1, 30))     # [1,30]
    init("ar30f", np.arange(30, dtype=np.float32))                      # [30]
    init("bigvec", np.full(30, 1e6, np.float32))
    init("negvec", np.full(30, -1.0, np.float32))
    init("big_col", np.full((10, 1), 1e6, np.float32))
    init("neg_col", np.full((10, 1), -1.0, np.float32))
    init("two_s", np.array([[2.0]], np.float32))
    init("one_col", np.array([[1.0]], np.float32))
    ch_keepb = np.ones((10, 1), bool); ch_keepb[0, 0] = False
    init("ch_keepb", ch_keepb, bool)
    ch0_onlyb = np.zeros((10, 1), bool); ch0_onlyb[0, 0] = True
    init("ch0_onlyb", ch0_onlyb, bool)
    chkeepb4 = np.ones((1, 10, 1), bool); chkeepb4[0, 0, 0] = False
    init("chkeepb4", chkeepb4, bool)

    # ---- in-grid masks; background channel 0 fills the whole grid ----
    n("ReduceSum", ["input"], "colsum", axes=[1, 2], keepdims=0)   # [1,30]
    n("ReduceSum", ["input"], "rowsum", axes=[1, 3], keepdims=0)   # [1,30]
    n("Reshape", ["colsum", "sh30"], "colsum1")                    # [30]
    n("Reshape", ["rowsum", "sh30"], "rowsum1")                    # [30]
    n("Greater", ["colsum1", "halff"], "colmask_b")                # [30] bool
    n("Greater", ["rowsum1", "halff"], "rowmask_b")                # [30] bool
    n("Cast", ["colmask_b"], "colmask", to=F32)                    # [30]
    n("Cast", ["rowmask_b"], "rowmask", to=F32)                    # [30]
    n("ReduceMax", ["colsum1"], "H_f", axes=[0], keepdims=0)       # scalar H

    # ---- seed indicators per channel (exclude background channel 0) ----
    # color-channel seed indicators (exclude background ch0 by slicing 1..9)
    n("ReduceMax", ["input"], "cvcol4", axes=[2], keepdims=0)      # [1,10,30]
    n("ReduceMax", ["input"], "cvrow4", axes=[3], keepdims=0)      # [1,10,30]
    n("Greater", ["cvcol4", "halff"], "cvcolb4")                   # [1,10,30] bool
    n("Greater", ["cvrow4", "halff"], "cvrowb4")                   # [1,10,30] bool
    n("And", ["cvcolb4", "chkeepb4"], "scolb4")                    # zero ch0 (bool)
    n("And", ["cvrowb4", "chkeepb4"], "srowb4")                    # zero ch0 (bool)
    n("Reshape", ["scolb4", "sh1030"], "scolb")                    # [10,30] bool
    n("Reshape", ["srowb4", "sh1030"], "srowb")                    # [10,30] bool

    # ---- xpose detection: dr = |seed row diff|, isx0 = dr==0 or dr==H-1 ----
    n("Cast", ["srowb"], "srowf", to=F32)                          # [10,30] f
    n("ReduceMax", ["srowf"], "rowseed", axes=[0], keepdims=0)     # [30] f
    n("Cast", ["rowseed"], "rowseed_b", to=BOOL)
    n("Where", ["rowseed_b", "ar30f", "bigvec"], "rs_lo")
    n("ReduceMin", ["rs_lo"], "minrow", axes=[0], keepdims=0)      # scalar
    n("Where", ["rowseed_b", "ar30f", "negvec"], "rs_hi")
    n("ReduceMax", ["rs_hi"], "maxrow", axes=[0], keepdims=0)      # scalar
    n("Sub", ["maxrow", "minrow"], "dr")                           # scalar
    n("Sub", ["H_f", "onef"], "Hm1")
    n("Cast", ["dr"], "dr_i", to=I32)
    n("Cast", ["Hm1"], "Hm1_i", to=I32)
    n("Equal", ["dr_i", "c0i"], "dr_is0")
    n("Equal", ["dr_i", "Hm1_i"], "dr_isH")
    n("Or", ["dr_is0", "dr_isH"], "isx0_b")                        # scalar bool

    # ---- choose canonical stripe-axis seeds (float) + in-grid mask ----
    n("Cast", ["scolb"], "scolf", to=F32)                          # [10,30] f
    n("Where", ["isx0_b", "scolf", "srowf"], "seeds")             # [10,30] f
    n("Where", ["isx0_b", "colmask", "rowmask"], "axmask")        # [30]

    # ---- stripe builder (run once) ----
    n("MatMul", ["seeds", "ar30f_col"], "pos")                    # [10,1]
    n("ReduceSum", ["seeds"], "cnt", axes=[1], keepdims=1)        # [10,1]
    n("Greater", ["cnt", "halff"], "used_b")                      # [10,1] bool
    n("Where", ["used_b", "pos", "big_col"], "plo")
    n("ReduceMin", ["plo"], "minp", axes=[0], keepdims=1)         # [1,1]
    n("Where", ["used_b", "pos", "neg_col"], "phi")
    n("ReduceMax", ["phi"], "maxp", axes=[0], keepdims=1)         # [1,1]
    n("Sub", ["maxp", "minp"], "step")
    n("Mul", ["step", "two_s"], "P")
    n("Max", ["P", "one_col"], "Ps")                              # clamp >=1
    n("Cast", ["Ps"], "Pi", to=I32)                               # [1,1] i32
    n("Sub", ["ar30row", "pos"], "delta")                         # [10,30] f
    n("Cast", ["delta"], "delta_i", to=I32)
    n("Mod", ["delta_i", "Pi"], "modr")                           # [10,30] i32
    n("Equal", ["modr", "c0i"], "onmod_b")                        # bool
    n("Greater", ["delta", "neghalf"], "fwd_b")                   # delta>=0 bool
    n("And", ["onmod_b", "fwd_b"], "mf_b")                        # [10,30] bool
    n("And", ["mf_b", "used_b"], "m_b")                           # & used [10,30] bool
    n("Reshape", ["axmask", "sh130"], "axrow")                    # [1,30] f
    n("Cast", ["axrow"], "axrow_b", to=BOOL)                      # [1,30] bool
    n("And", ["m_b", "axrow_b"], "colorpat_b")                    # [10,30] bool stripes

    # any-stripe column (either color): minp + j*step0, step0 = maxp-minp
    # (per-color period is P=2*step0; spacing of EITHER color is step0).
    # anyst[k] = (k>=minp) & ((k-minp)%step0==0) & in-grid.
    n("Max", ["step", "one_col"], "step_c")                       # clamp >=1 (float)
    n("Cast", ["step_c"], "step_is", to=I32)                      # [1,1] = step0
    n("Sub", ["ar30row", "minp"], "dany")                         # [1,30] f
    n("Cast", ["dany"], "dany_i", to=I32)
    n("Mod", ["dany_i", "step_is"], "danym")                      # [1,30] i32
    n("Equal", ["danym", "c0i"], "danym0_b")
    n("Greater", ["dany", "neghalf"], "danyf_b")
    n("And", ["danym0_b", "danyf_b"], "stcol_b")                  # [1,30] bool
    n("And", ["stcol_b", "axrow_b"], "anyst_b")                   # in-grid
    n("Not", ["anyst_b"], "notany_b")                             # [1,30] bool
    n("And", ["notany_b", "axrow_b"], "bg_b")                     # [1,30] background

    # assemble S[10,30] bool: ch0 = bg, ch1..9 = colorpat
    n("And", ["colorpat_b", "ch_keepb"], "colorz")                # bool zero ch0
    n("And", ["bg_b", "ch0_onlyb"], "bgch")                       # bool only ch0
    n("Or", ["colorz", "bgch"], "S_b")                            # [10,30] bool
    n("Cast", ["S_b"], "S", to=F32)                               # [10,30] f

    # ---- perpendicular plain mask (float, broadcast in the Where below) ----
    n("Where", ["isx0_b", "rowmask", "colmask"], "plain")         # [30] f

    # ---- route to Urow / Vcol per orientation ----
    # xpose=0: Urow=plain(rows), Vcol=S(col stripes)
    # xpose=1: Urow=S(row stripes), Vcol=plain(cols)
    n("Where", ["isx0_b", "plain", "S"], "Urow2d")                # [10,30] f (bcast)
    n("Where", ["isx0_b", "S", "plain"], "Vcol2d")                # [10,30] f
    init("shU", np.array([1, 10, 30, 1], np.int64), np.int64)
    init("shV", np.array([1, 10, 1, 30], np.int64), np.int64)
    n("Reshape", ["Urow2d", "shU"], "Urow")
    n("Reshape", ["Vcol2d", "shV"], "Vcol")
    n("Mul", ["Urow", "Vcol"], "output")                          # [1,10,30,30] free

    return _model(nodes, inits)
