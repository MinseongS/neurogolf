"""Task 234 (ARC 98cf29f8): "frog eats fly".

A solid rectangle (the frog, color[0]) and a smaller rectangle (the fly,
color[1]) plus a 1-wide line of the fly's color (the tongue) connecting the
fly to the frog. The output keeps the frog fixed, deletes the tongue, and
slides the fly box along the tongue axis until it is flush against the frog.

Key facts (verified on all 266 examples):
- the frog is the colour whose pixel count equals its bounding-box area
  (solid); the fly's count is strictly less (its bbox swallows the tongue).
- the output fly box is always a rectangle, so it factors as the outer
  product of a row-mask and a col-mask -> the whole output (frog box + fly
  box) is computed as outRow[1,10,30,1] * outCol[1,10,1,30] straight into
  `output` (free).
- the moved axis is the one on which the frog and fly projections are
  disjoint; on the perpendicular axis the fly mask is unchanged, on the moved
  axis it becomes a run of length T (= box thickness = min positive
  per-perpendicular-line fly count) flush against the frog edge.

All intermediates are tiny 1-D / scalar tensors of small integers, so the
graph is exact in float32 and cheap in memory.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ..builders import _model

BIG = 1000.0


def build(task):
    inits = []
    nodes = []

    # All mask math runs in float16 (every value is a small integer < 2048,
    # so fp16 is exact) -> the many [1,10,30,1] mask tensors cost 600B each
    # instead of 1200B.  Only the final basis vectors are cast back to fp32
    # for the MatMuls into `output` (which must be fp32).
    F16 = np.float16

    def init(name, arr, dtype=F16):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    # index vectors / scalars (fp16)
    init("ri", np.arange(30).reshape(1, 1, 30, 1))   # rows
    init("ci", np.arange(30).reshape(1, 1, 1, 30))   # cols
    init("one", np.array(1.0))
    init("big", np.array(BIG))
    m0 = np.ones((1, 10, 1, 1)); m0[0, 0] = 0.0
    init("m0", m0)

    # --- per-channel projections (reduce fp32 `input`, then cast the small
    #     outputs to fp16 -- never materialise a fp16 copy of the canvas) ---
    n("ReduceMax", ["input"], "Rf", axes=[3], keepdims=1)   # [1,10,30,1] fp32
    n("Cast", ["Rf"], "R", to=onnx.TensorProto.FLOAT16)
    n("ReduceMax", ["input"], "Cf", axes=[2], keepdims=1)   # [1,10,1,30]
    n("Cast", ["Cf"], "C", to=onnx.TensorProto.FLOAT16)
    n("ReduceSum", ["input"], "cntf", axes=[2, 3], keepdims=1)  # [1,10,1,1]
    n("Cast", ["cntf"], "cnt", to=onnx.TensorProto.FLOAT16)

    # --- per-channel bbox area to find the solid (frog) channel -------
    # both boxes are contiguous, so #occupied-rows == row-extent; bbox area
    # = (sum of R) * (sum of C) -- no per-channel min/max needed.
    n("ReduceSum", ["R"], "rspan", axes=[2], keepdims=1)          # [1,10,1,1]
    n("ReduceSum", ["C"], "cspan", axes=[3], keepdims=1)          # [1,10,1,1]
    n("Mul", ["rspan", "cspan"], "bbox")                          # [1,10,1,1]

    # isFrog = (cnt == bbox) & (cnt>0); isFly = (cnt>0) & !isFrog
    n("Cast", ["cnt"], "cnt_i", to=onnx.TensorProto.INT32)
    n("Cast", ["bbox"], "bbox_i", to=onnx.TensorProto.INT32)
    n("Equal", ["cnt_i", "bbox_i"], "eqsolid")                    # bool
    n("Cast", ["eqsolid"], "solid", to=onnx.TensorProto.FLOAT16)    # [1,10,1,1]
    # pos = cnt>0
    init("zero", np.array(0.0, np.float32))
    n("Greater", ["cnt", "zero"], "pos_b")
    n("Cast", ["pos_b"], "pos", to=onnx.TensorProto.FLOAT16)
    n("Mul", ["solid", "pos"], "frog_raw")
    n("Mul", ["frog_raw", "m0"], "isFrog")                        # [1,10,1,1]
    n("Sub", ["pos", "isFrog"], "fly_raw")
    n("Mul", ["fly_raw", "m0"], "isFly")

    # --- global frog / fly axis projections ---------------------------
    n("Mul", ["R", "isFrog"], "RFrog")
    n("ReduceSum", ["RFrog"], "frogR", axes=[1], keepdims=1)      # [1,1,30,1]
    n("Mul", ["R", "isFly"], "RFly")
    n("ReduceSum", ["RFly"], "flyR", axes=[1], keepdims=1)
    n("Mul", ["C", "isFrog"], "CFrog")
    n("ReduceSum", ["CFrog"], "frogC", axes=[1], keepdims=1)      # [1,1,1,30]
    n("Mul", ["C", "isFly"], "CFly")
    n("ReduceSum", ["CFly"], "flyC", axes=[1], keepdims=1)

    # overlaps -> moved-axis selectors (scalars)
    n("Mul", ["frogR", "flyR"], "ovR")
    n("ReduceSum", ["ovR"], "ovRs", axes=[2, 3], keepdims=1)      # [1,1,1,1]
    n("Mul", ["frogC", "flyC"], "ovC")
    n("ReduceSum", ["ovC"], "ovCs", axes=[2, 3], keepdims=1)
    # moved_rows = 1 - (ovRs>0); but rows-disjoint => moved.  moved_rows = (ovRs==0)
    n("Greater", ["ovRs", "zero"], "ovR_b")
    n("Cast", ["ovR_b"], "ovR_f", to=onnx.TensorProto.FLOAT16)
    n("Sub", ["one", "ovR_f"], "moved_rows")                     # scalar [1,1,1,1]
    n("Greater", ["ovCs", "zero"], "ovC_b")
    n("Cast", ["ovC_b"], "ovC_f", to=onnx.TensorProto.FLOAT16)
    n("Sub", ["one", "ovC_f"], "moved_cols")

    # --- frog edges (global) ------------------------------------------
    n("Mul", ["frogR", "ri"], "fRri")
    n("ReduceMax", ["fRri"], "fr_rmax", axes=[2, 3], keepdims=1)  # [1,1,1,1]
    n("Sub", ["one", "frogR"], "frogRinv")
    n("Mul", ["frogRinv", "big"], "frogRbig")
    n("Add", ["fRri", "frogRbig"], "fRrib")
    n("ReduceMin", ["fRrib"], "fr_rmin", axes=[2, 3], keepdims=1)
    n("Mul", ["frogC", "ci"], "fCci")
    n("ReduceMax", ["fCci"], "fr_cmax", axes=[2, 3], keepdims=1)
    n("Sub", ["one", "frogC"], "frogCinv")
    n("Mul", ["frogCinv", "big"], "frogCbig")
    n("Add", ["fCci", "frogCbig"], "fCcib")
    n("ReduceMin", ["fCcib"], "fr_cmin", axes=[2, 3], keepdims=1)

    # fly near edges
    n("Mul", ["flyR", "ri"], "flRri")
    n("Sub", ["one", "flyR"], "flyRinv")
    n("Mul", ["flyRinv", "big"], "flyRbig")
    n("Add", ["flRri", "flyRbig"], "flRrib")
    n("ReduceMin", ["flRrib"], "fl_rmin", axes=[2, 3], keepdims=1)
    n("Mul", ["flyC", "ci"], "flCci")
    n("Sub", ["one", "flyC"], "flyCinv")
    n("Mul", ["flyCinv", "big"], "flyCbig")
    n("Add", ["flCci", "flyCbig"], "flCcib")
    n("ReduceMin", ["flCcib"], "fl_cmin", axes=[2, 3], keepdims=1)

    # --- T = box thickness on the moved axis --------------------------
    # per-channel column/row sums first (avoids a [1,10,30,30] intermediate)
    n("ReduceSum", ["input"], "colSumChf", axes=[2], keepdims=1)  # [1,10,1,30]
    n("Cast", ["colSumChf"], "colSumCh", to=onnx.TensorProto.FLOAT16)
    n("ReduceSum", ["input"], "rowSumChf", axes=[3], keepdims=1)  # [1,10,30,1]
    n("Cast", ["rowSumChf"], "rowSumCh", to=onnx.TensorProto.FLOAT16)
    n("Mul", ["colSumCh", "isFly"], "colSumFly")
    n("ReduceSum", ["colSumFly"], "flyColSum", axes=[1], keepdims=1)  # [1,1,1,30]
    n("Mul", ["rowSumCh", "isFly"], "rowSumFly")
    n("ReduceSum", ["rowSumFly"], "flyRowSum", axes=[1], keepdims=1)  # [1,1,30,1]
    # min positive (replace zeros with BIG via Where)
    n("Greater", ["flyColSum", "zero"], "fcs_pos")
    n("Where", ["fcs_pos", "flyColSum", "big"], "fcs_w")
    n("ReduceMin", ["fcs_w"], "Tr", axes=[2, 3], keepdims=1)   # thickness rows
    n("Greater", ["flyRowSum", "zero"], "frs_pos")
    n("Where", ["frs_pos", "flyRowSum", "big"], "frs_w")
    n("ReduceMin", ["frs_w"], "Tc", axes=[2, 3], keepdims=1)   # thickness cols

    # --- moved-axis run masks -----------------------------------------
    # rows: below = fl_rmin > fr_rmax ; run = below?[fr_rmax+1, fr_rmax+Tr]
    #                                            : [fr_rmin-Tr, fr_rmin-1]
    n("Greater", ["fl_rmin", "fr_rmax"], "below_b")
    n("Cast", ["below_b"], "below", to=onnx.TensorProto.FLOAT16)   # scalar
    # below run bounds
    n("Add", ["fr_rmax", "one"], "lo_below_r")           # fr_rmax+1
    n("Add", ["fr_rmax", "Tr"], "hi_below_r")            # fr_rmax+Tr
    # above run bounds
    n("Sub", ["fr_rmin", "Tr"], "lo_above_r")            # fr_rmin-Tr
    n("Sub", ["fr_rmin", "one"], "hi_above_r")           # fr_rmin-1
    # select lo/hi by 'below'
    n("Sub", ["one", "below"], "above")
    n("Mul", ["below", "lo_below_r"], "_lb")
    n("Mul", ["above", "lo_above_r"], "_la")
    n("Add", ["_lb", "_la"], "lo_r")
    n("Mul", ["below", "hi_below_r"], "_hb")
    n("Mul", ["above", "hi_above_r"], "_ha")
    n("Add", ["_hb", "_ha"], "hi_r")
    # runRow[r] = (ri>=lo_r) & (ri<=hi_r)   over [1,1,30,1]
    # opset10 has no GreaterOrEqual; use Greater with -1 offsets
    n("Sub", ["lo_r", "one"], "lo_r_m")
    n("Add", ["hi_r", "one"], "hi_r_p")
    n("Greater", ["ri", "lo_r_m"], "geR")                # ri > lo-1  == ri>=lo
    n("Greater", ["hi_r_p", "ri"], "leR")                # hi+1 > ri  == ri<=hi
    n("And", ["geR", "leR"], "runRow_b")
    n("Cast", ["runRow_b"], "runRow", to=onnx.TensorProto.FLOAT16)  # [1,1,30,1]

    # cols
    n("Greater", ["fl_cmin", "fr_cmax"], "belowc_b")
    n("Cast", ["belowc_b"], "belowc", to=onnx.TensorProto.FLOAT16)
    n("Add", ["fr_cmax", "one"], "lo_below_c")
    n("Add", ["fr_cmax", "Tc"], "hi_below_c")
    n("Sub", ["fr_cmin", "Tc"], "lo_above_c")
    n("Sub", ["fr_cmin", "one"], "hi_above_c")
    n("Sub", ["one", "belowc"], "abovec")
    n("Mul", ["belowc", "lo_below_c"], "_lbc")
    n("Mul", ["abovec", "lo_above_c"], "_lac")
    n("Add", ["_lbc", "_lac"], "lo_c")
    n("Mul", ["belowc", "hi_below_c"], "_hbc")
    n("Mul", ["abovec", "hi_above_c"], "_hac")
    n("Add", ["_hbc", "_hac"], "hi_c")
    n("Sub", ["lo_c", "one"], "lo_c_m")
    n("Add", ["hi_c", "one"], "hi_c_p")
    n("Greater", ["ci", "lo_c_m"], "geC")
    n("Greater", ["hi_c_p", "ci"], "leC")
    n("And", ["geC", "leC"], "runCol_b")
    n("Cast", ["runCol_b"], "runCol", to=onnx.TensorProto.FLOAT16)  # [1,1,1,30]

    # --- global fly output row / col masks (single vectors) -----------
    # flyOutRow = moved_rows ? runRow : flyR   (perpendicular axis kept)
    n("Mul", ["moved_rows", "runRow"], "mr_run")
    n("Sub", ["one", "moved_rows"], "keep_rows")
    n("Mul", ["keep_rows", "flyR"], "kr_R")
    n("Add", ["mr_run", "kr_R"], "flyOutRow")            # [1,1,30,1]
    n("Mul", ["moved_cols", "runCol"], "mc_run")
    n("Sub", ["one", "moved_cols"], "keep_cols")
    n("Mul", ["keep_cols", "flyC"], "kc_C")
    n("Add", ["mc_run", "kc_C"], "flyOutCol")            # [1,1,1,30]

    # --- fused bilinear assembly straight into `output` ---------------
    # Only 3 basis vectors are ever non-zero (frog box, moved fly box, grid),
    # so the canvas is two tiny MatMuls into `output` (free):
    #   VrT3[1,1,30,3] = [frogRow | flyOutRow | gridRow]
    #   VcT3[1,1,3,30] = [frogCol | flyOutCol | gridCol]
    #   output[ch] = sum_{k,l} VrT3[:,k] My3[ch,k,l] VcT3[l,:]
    # My3 routes basis 0->frog colour, 1->fly colour, 2(grid)->bg, with the
    # coloured cells subtracted from the background channel:
    #   My3[ch] = isFrog[ch]*E00 + isFly[ch]*E11 + e0[ch]*(E22 - E00 - E11)
    n("ReduceMax", ["input"], "gridRowf", axes=[1, 3], keepdims=1)  # [1,1,30,1]
    n("Cast", ["gridRowf"], "gridRow", to=onnx.TensorProto.FLOAT16)
    n("ReduceMax", ["input"], "gridColf", axes=[1, 2], keepdims=1)  # [1,1,1,30]
    n("Cast", ["gridColf"], "gridCol", to=onnx.TensorProto.FLOAT16)
    n("Concat", ["frogR", "flyOutRow", "gridRow"], "VrT3h", axis=3)  # [1,1,30,3]
    n("Concat", ["frogC", "flyOutCol", "gridCol"], "VcT3h", axis=2)  # [1,1,3,30]
    # final MatMuls run in fp32 so the last op writes straight into `output`
    n("Cast", ["VrT3h"], "VrT3", to=onnx.TensorProto.FLOAT)
    n("Cast", ["VcT3h"], "VcT3", to=onnx.TensorProto.FLOAT)

    E00 = np.zeros((1, 1, 3, 3)); E00[0, 0, 0, 0] = 1.0
    E11 = np.zeros((1, 1, 3, 3)); E11[0, 0, 1, 1] = 1.0
    Ec = np.zeros((1, 1, 3, 3))
    Ec[0, 0, 2, 2] = 1.0; Ec[0, 0, 0, 0] = -1.0; Ec[0, 0, 1, 1] = -1.0
    init("E00", E00); init("E11", E11); init("Ec", Ec)   # fp16 like the masks
    e0 = np.zeros((1, 10, 1, 1)); e0[0, 0] = 1.0
    init("e0", e0)
    n("Mul", ["isFrog", "E00"], "M_frog")    # [1,10,3,3] fp16
    n("Mul", ["isFly", "E11"], "M_fly")
    n("Mul", ["e0", "Ec"], "M_bg")
    n("Add", ["M_frog", "M_fly"], "M_ff")
    n("Add", ["M_ff", "M_bg"], "My3h")       # [1,10,3,3] fp16
    n("Cast", ["My3h"], "My3", to=onnx.TensorProto.FLOAT)
    n("MatMul", ["My3", "VcT3"], "Rmid")     # [1,10,3,30]
    n("MatMul", ["VrT3", "Rmid"], "output")  # [1,10,30,30] (free)

    return _model(nodes, inits)
