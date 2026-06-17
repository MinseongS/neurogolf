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
  box) is computed via a separable row/col label map straight into the free
  BOOL output.
- the moved axis is the one on which the frog and fly projections are
  disjoint; on the perpendicular axis the fly mask is unchanged, on the moved
  axis it becomes a run of length T (= box thickness) flush against the frog.

Memory lever (vs the prior 17698B version): the frog/fly are exactly TWO
colour channels, and the channel index of each equals its colour.  So instead
of materialising per-channel spatial products (R*isFrog, C*isFly, ...) -- six
[1,10,30,*] fp16 planes at 600B each -- we recover each colour's channel index
as a runtime int32 [1] scalar and Gather that single channel out of R / C / the
per-channel sums (each a 60-120B [1,1,30,1] slice).  This removes ~3.6KB of
per-channel product planes.

All intermediates are tiny 1-D / scalar tensors of small integers, so the
graph is exact in float16 (every value < 2048) and float32.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ..harness import IR_VERSION

BIG = 1000.0


def build(task):
    inits = []
    nodes = []
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
    init("zero", np.array(0.0))
    init("zero32", np.array(0.0, np.float32), np.float32)
    init("half16", np.array(0.5), np.float16)
    init("shape1", np.array([1], np.int64), np.int64)
    cidx = np.arange(10).reshape(1, 10, 1, 1)
    init("cidx", cidx)                                # fp16 [1,10,1,1]
    m0 = np.ones((1, 10, 1, 1)); m0[0, 0] = 0.0
    init("m0", m0)

    # --- per-channel projections (reduce fp32 input) ------------------
    n("ReduceSum", ["input"], "rowSumChf", axes=[3], keepdims=1)  # [1,10,30,1] f32
    n("ReduceSum", ["input"], "colSumChf", axes=[2], keepdims=1)  # [1,10,1,30] f32
    # per-channel occupancy (bool->fp16) [1,10,30,1] / [1,10,1,30]
    n("Greater", ["rowSumChf", "zero32"], "R_b")
    n("Cast", ["R_b"], "R", to=onnx.TensorProto.FLOAT16)          # [1,10,30,1]
    n("Greater", ["colSumChf", "zero32"], "C_b")
    n("Cast", ["C_b"], "C", to=onnx.TensorProto.FLOAT16)          # [1,10,1,30]
    n("ReduceSum", ["input"], "cntf", axes=[2, 3], keepdims=1)    # [1,10,1,1]
    n("Cast", ["cntf"], "cnt", to=onnx.TensorProto.FLOAT16)

    # --- per-channel bbox area to find the solid (frog) channel -------
    n("ReduceSum", ["R"], "rspan", axes=[2], keepdims=1)          # [1,10,1,1]
    n("ReduceSum", ["C"], "cspan", axes=[3], keepdims=1)          # [1,10,1,1]
    n("Mul", ["rspan", "cspan"], "bbox")                          # [1,10,1,1]

    # isFrog = (cnt == bbox) & (cnt>0) & (chan!=0); isFly = (cnt>0) & !isFrog
    n("Cast", ["cnt"], "cnt_i", to=onnx.TensorProto.INT32)
    n("Cast", ["bbox"], "bbox_i", to=onnx.TensorProto.INT32)
    n("Equal", ["cnt_i", "bbox_i"], "eqsolid")                    # bool
    n("Cast", ["eqsolid"], "solid", to=onnx.TensorProto.FLOAT16)
    n("Greater", ["cnt", "zero"], "pos_b")
    n("Cast", ["pos_b"], "pos", to=onnx.TensorProto.FLOAT16)
    n("Mul", ["solid", "pos"], "frog_raw")
    n("Mul", ["frog_raw", "m0"], "isFrog")                        # [1,10,1,1]
    n("Sub", ["pos", "isFrog"], "fly_raw")
    n("Mul", ["fly_raw", "m0"], "isFly")

    # --- channel indices (== colour) as runtime int32 [1] scalars -----
    def chan_index(sel, base):
        n("Mul", [sel, "cidx"], base + "_p")
        n("ReduceSum", [base + "_p"], base + "_s", axes=[1], keepdims=0)  # [] scalar
        n("Reshape", [base + "_s", "shape1"], base + "_r")               # [1]
        n("Cast", [base + "_r"], base + "_idx", to=onnx.TensorProto.INT32)
        return base + "_idx"

    frogIdx = chan_index("isFrog", "frog")
    flyIdx = chan_index("isFly", "fly")

    # Gather the frog/fly channel directly out of the per-channel planes:
    n("Gather", ["R", frogIdx], "frogR", axis=1)                  # [1,1,30,1]
    n("Gather", ["R", flyIdx], "flyR", axis=1)
    n("Gather", ["C", frogIdx], "frogC", axis=1)                  # [1,1,1,30]
    n("Gather", ["C", flyIdx], "flyC", axis=1)
    # fly per-row / per-col pixel counts (for thickness) -> fp16 [1,1,..]
    n("Gather", ["rowSumChf", flyIdx], "flyRowSumf", axis=1)      # f32 [1,1,30,1]
    n("Cast", ["flyRowSumf"], "flyRowSum", to=onnx.TensorProto.FLOAT16)
    n("Gather", ["colSumChf", flyIdx], "flyColSumf", axis=1)      # f32 [1,1,1,30]
    n("Cast", ["flyColSumf"], "flyColSum", to=onnx.TensorProto.FLOAT16)

    # --- moved-axis selectors: axis where frog/fly projections disjoint
    n("Mul", ["frogR", "flyR"], "ovR")
    n("ReduceSum", ["ovR"], "ovRs", axes=[2, 3], keepdims=1)      # [1,1,1,1]
    n("Greater", ["ovRs", "zero"], "ovR_b")
    n("Cast", ["ovR_b"], "ovR_f", to=onnx.TensorProto.FLOAT16)
    n("Sub", ["one", "ovR_f"], "moved_rows")                     # scalar

    # --- frog edges (global) ------------------------------------------
    n("Mul", ["frogR", "ri"], "fRri")
    n("ReduceMax", ["fRri"], "fr_rmax", axes=[2, 3], keepdims=1)
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

    # fly near edge (min row / col)
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
    n("Greater", ["flyColSum", "zero"], "fcs_pos")
    n("Where", ["fcs_pos", "flyColSum", "big"], "fcs_w")
    n("ReduceMin", ["fcs_w"], "Tr", axes=[2, 3], keepdims=1)   # thickness rows
    n("Greater", ["flyRowSum", "zero"], "frs_pos")
    n("Where", ["frs_pos", "flyRowSum", "big"], "frs_w")
    n("ReduceMin", ["frs_w"], "Tc", axes=[2, 3], keepdims=1)   # thickness cols

    # --- moved-axis run masks -----------------------------------------
    n("Greater", ["fl_rmin", "fr_rmax"], "below_b")
    n("Cast", ["below_b"], "below", to=onnx.TensorProto.FLOAT16)
    n("Add", ["fr_rmax", "one"], "lo_below_r")
    n("Add", ["fr_rmax", "Tr"], "hi_below_r")
    n("Sub", ["fr_rmin", "Tr"], "lo_above_r")
    n("Sub", ["fr_rmin", "one"], "hi_above_r")
    n("Sub", ["one", "below"], "above")
    n("Mul", ["below", "lo_below_r"], "_lb")
    n("Mul", ["above", "lo_above_r"], "_la")
    n("Add", ["_lb", "_la"], "lo_r")
    n("Mul", ["below", "hi_below_r"], "_hb")
    n("Mul", ["above", "hi_above_r"], "_ha")
    n("Add", ["_hb", "_ha"], "hi_r")
    n("Sub", ["lo_r", "one"], "lo_r_m")
    n("Add", ["hi_r", "one"], "hi_r_p")
    n("Greater", ["ri", "lo_r_m"], "geR")
    n("Greater", ["hi_r_p", "ri"], "leR")
    n("And", ["geR", "leR"], "runRow_b")
    n("Cast", ["runRow_b"], "runRow", to=onnx.TensorProto.FLOAT16)  # [1,1,30,1]

    # cols (moved_cols == 1 - moved_rows, but recompute below_c independently)
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

    # --- output fly row / col masks -----------------------------------
    # flyOutRow = moved_rows ? runRow : flyR   (perpendicular axis kept)
    n("Mul", ["moved_rows", "runRow"], "mr_run")
    n("Sub", ["one", "moved_rows"], "keep_rows")
    n("Mul", ["keep_rows", "flyR"], "kr_R")
    n("Add", ["mr_run", "kr_R"], "flyOutRow")            # [1,1,30,1]
    # moved_cols = 1 - moved_rows  (exactly one axis moves)
    n("Sub", ["one", "moved_rows"], "moved_cols")
    n("Mul", ["moved_cols", "runCol"], "mc_run")
    n("Sub", ["one", "moved_cols"], "keep_cols")
    n("Mul", ["keep_cols", "flyC"], "kc_C")
    n("Add", ["mc_run", "kc_C"], "flyOutCol")            # [1,1,1,30]

    # --- label map + final Equal (floor-break) ------------------------
    n("ReduceMax", ["input"], "gridRowf", axes=[1, 3], keepdims=1)  # [1,1,30,1] f32
    n("Cast", ["gridRowf"], "gridRow", to=onnx.TensorProto.FLOAT16)
    n("ReduceMax", ["input"], "gridColf", axes=[1, 2], keepdims=1)  # [1,1,1,30]
    n("Cast", ["gridColf"], "gridCol", to=onnx.TensorProto.FLOAT16)

    n("Greater", ["frogR", "half16"], "frB")            # [1,1,30,1] bool
    n("Greater", ["frogC", "half16"], "fcB")            # [1,1,1,30] bool
    n("And", ["frB", "fcB"], "frogBox")                 # [1,1,30,30] bool
    n("Greater", ["flyOutRow", "half16"], "flrB")
    n("Greater", ["flyOutCol", "half16"], "flcB")
    n("And", ["flrB", "flcB"], "flyBox")                # [1,1,30,30] bool
    n("Greater", ["gridRow", "half16"], "grB")
    n("Greater", ["gridCol", "half16"], "gcB")
    n("And", ["grB", "gcB"], "gridBox")                 # [1,1,30,30] bool

    # frog / fly colour ids (uint8 scalars): channel index == colour
    n("Cast", ["frog_r"], "frogColU", to=onnx.TensorProto.UINT8)  # [1]
    n("Cast", ["fly_r"], "flyColU", to=onnx.TensorProto.UINT8)

    init("u0", np.array(0, np.uint8), np.uint8)
    init("u10", np.array(10, np.uint8), np.uint8)
    n("Where", ["gridBox", "u0", "u10"], "Lg")          # 0 in-grid else 10
    n("Where", ["flyBox", "flyColU", "Lg"], "Lf")       # fly box -> fly colour
    n("Where", ["frogBox", "frogColU", "Lf"], "L")      # frog box -> frog colour

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                 # -> free BOOL output

    x = helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info(
        "output", onnx.TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task234", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
