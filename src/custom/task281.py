"""Task 281 (b548a754): a hollow box (outer border + inner fill) is stretched
along the axis of an isolated cyan dot until its edge reaches the dot; the dot
is erased.  xpose/flip are applied to BOTH input and output, so the rule is the
orientation-agnostic stretch.

Key facts (verified exact over 50k arc-gen instances):
  - the stretched frame is exactly the FILLED bounding box of all nonzero cells
    (box + dot): rows min..max occupied, cols min..max occupied.
  - bbox border ring -> outer color; eroded interior -> inner color.
  - a cell shows the INNER color iff it has all 4 orthogonal neighbors nonzero.
  - outer color = the nonzero, non-cyan color that is NOT the inner color.

Graph: row/col occupancy -> cummax span (filled bbox); 3x1/1x3 erosion conv for
the interior; a 4-neighbour conv flags interior cells.  A single colour-id plane
(Conv with weights 0..9) lets us read the inner/outer colour ids as scalars and
build a runtime [10,3] selector (cols = gridmask, interiorE, border coeffs).
That selector becomes a runtime 1x1 Conv weight applied straight into `output`:
  output[c] = gridmask*[c==0] + interiorE*([c==inner]-[c==0])
                              + border*([c==outer]-[c==0])
so channel 0 (background) = gridmask - interiorE - border (1 inside grid only).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ..builders import _model


def build(task):
    inits = []
    nodes = []

    def init(name, arr, dtype=np.float32):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    # --- grid region, nonzero & colour-id planes (one 1x1 Conv, 2 channels) ---
    n("ReduceMax", ["input"], "gridmask", axes=[1], keepdims=1)   # [1,1,30,30]
    w_id = np.zeros((2, 10, 1, 1), np.float32)
    w_id[0, :, 0, 0] = np.arange(10, dtype=np.float32)            # colour id
    w_id[1, 1:, 0, 0] = 1.0                                       # nonzero (ch1..9)
    init("w_id", w_id)
    n("Conv", ["input", "w_id"], "idch")                         # [1,2,30,30]
    init("sl_st", np.array([0], np.int64), np.int64)
    init("sl_en", np.array([1], np.int64), np.int64)
    init("sl_ax", np.array([1], np.int64), np.int64)
    init("sl_en2", np.array([2], np.int64), np.int64)
    n("Slice", ["idch", "sl_st", "sl_en", "sl_ax"], "colorid")   # [1,1,30,30]
    n("Slice", ["idch", "sl_en", "sl_en2", "sl_ax"], "nzf")      # nonzero colour

    # --- filled bounding box (cummax span on row & col occupancy) ---
    n("ReduceMax", ["nzf"], "rowo", axes=[3], keepdims=1)        # [1,1,30,1]
    n("ReduceMax", ["nzf"], "colo", axes=[2], keepdims=1)        # [1,1,1,30]
    n("MaxPool", ["rowo"], "rpre", kernel_shape=[30, 1], pads=[29, 0, 0, 0])
    n("MaxPool", ["rowo"], "rsuf", kernel_shape=[30, 1], pads=[0, 0, 29, 0])
    n("Mul", ["rpre", "rsuf"], "rowocc")                         # [1,1,30,1]
    n("MaxPool", ["colo"], "cpre", kernel_shape=[1, 30], pads=[0, 29, 0, 0])
    n("MaxPool", ["colo"], "csuf", kernel_shape=[1, 30], pads=[0, 0, 0, 29])
    n("Mul", ["cpre", "csuf"], "colocc")                         # [1,1,1,30]

    # interior rows/cols: span[r-1]&span[r]&span[r+1] (3-wide erosion)
    init("kr", np.ones((1, 1, 3, 1), np.float32))
    init("be", np.array([-2.0], np.float32))
    n("Conv", ["rowocc", "kr", "be"], "rint_r", pads=[1, 0, 1, 0])
    n("Relu", ["rint_r"], "rowint")                              # [1,1,30,1]
    init("kc", np.ones((1, 1, 1, 3), np.float32))
    n("Conv", ["colocc", "kc", "be"], "cint_c", pads=[0, 1, 0, 1])
    n("Relu", ["cint_c"], "colint")                              # [1,1,1,30]

    n("Mul", ["rowint", "colint"], "interiorE")                  # [1,1,30,30]
    n("Mul", ["rowocc", "colocc"], "bbox")                       # [1,1,30,30]

    # --- inner-cell mask: cell with all 4 orthogonal neighbours nonzero ---
    kpl = np.zeros((1, 1, 3, 3), np.float32)
    kpl[0, 0, 0, 1] = kpl[0, 0, 2, 1] = kpl[0, 0, 1, 0] = kpl[0, 0, 1, 2] = 1.0
    init("kpl", kpl)
    init("bpl", np.array([-3.0], np.float32))
    n("Conv", ["nzf", "kpl", "bpl"], "nbr_r", pads=[1, 1, 1, 1])
    n("Relu", ["nbr_r"], "innernb")                              # [1,1,30,30]

    # inner colour id (scalar): max colour-id over inner cells
    n("Mul", ["colorid", "innernb"], "in_id_p")
    n("ReduceMax", ["in_id_p"], "inner_id", axes=[1, 2, 3], keepdims=0)  # [1]
    # innerVec via Equal(arange(10), inner_id)
    init("ar10", np.arange(10, dtype=np.int32).reshape(10, 1), np.int32)
    n("Cast", ["inner_id"], "inner_i", to=onnx.TensorProto.INT32)
    n("Equal", ["ar10", "inner_i"], "iv_b")                      # [10,1] bool
    n("Cast", ["iv_b"], "iv", to=onnx.TensorProto.FLOAT)         # [10,1]

    # outerVec (tiny): present colour, not inner, not cyan/bg  ([1,10] ops)
    n("ReduceMax", ["input"], "present", axes=[2, 3], keepdims=0)  # [1,10]
    init("v_shape", np.array([10, 1], np.int64), np.int64)
    n("Reshape", ["present", "v_shape"], "presentv")             # [10,1]
    notc = np.ones((10, 1), np.float32)
    notc[0, 0] = 0.0   # background
    notc[8, 0] = 0.0   # cyan
    init("notc", notc)
    init("one1", np.array([[1.0]], np.float32))
    n("Sub", ["one1", "iv"], "notiv")
    n("Mul", ["presentv", "notiv"], "ptmp")
    n("Mul", ["ptmp", "notc"], "ov")                             # [10,1] outerVec

    # output = gridmask*e0 + interiorE*(iv-ov) + bbox*(ov-e0)
    # (border = bbox - interiorE -> outer; ch0 = gridmask - bbox)
    e0 = np.zeros((10, 1), np.float32)
    e0[0, 0] = 1.0
    init("e0", e0)
    n("Sub", ["iv", "ov"], "c1")                                 # [10,1]
    n("Sub", ["ov", "e0"], "c2")
    n("Concat", ["e0", "c1", "c2"], "Wsel", axis=1)              # [10,3]
    init("w_shape", np.array([10, 3, 1, 1], np.int64), np.int64)
    n("Reshape", ["Wsel", "w_shape"], "Wconv")                   # [10,3,1,1]

    # mask stack [1,3,30,30] = (gridmask, interiorE, bbox); 1x1 Conv -> output
    n("Concat", ["gridmask", "interiorE", "bbox"], "Mstack", axis=1)
    n("Conv", ["Mstack", "Wconv"], "output")                     # free

    return _model(nodes, inits)
