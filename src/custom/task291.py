"""task291 (ARC-AGI b9b7f026) — donut-box colour readout (1x1 output).

Rule (from the generator):
  An H×W grid (12..18 each) holds 4..7 disjoint solid axis-aligned rectangles,
  each a distinct colour. The FIRST box (boxes[0], colour colors[0]) has a
  rectangular "donut hole" of black cells carved out of its STRICT interior
  (drow/dcol/dwide/dtall chosen so the hole never touches the box edge — the
  colour-0 ring stays solid all the way round). Every other box is a perfectly
  SOLID rectangle. The OUTPUT is a 1x1 grid whose single cell holds colors[0] —
  the colour of the holed box.

Discriminator (closed-form, NO ramps / argmax / 30x30 plane):
  Each box is a contiguous solid rectangle, so for a SOLID colour k its pixel
  count equals (#occupied rows)*(#occupied cols). The donut box's hole is STRICTLY
  interior, so it still occupies every row & col of its bounding box, yet removes
  >=1 interior cell -> count < nrows*ncols. The donut colour is the UNIQUE k>=1
  with count < nrows*ncols.
      nrows_k = sum_r [colour k present in row r]   (ReduceSum of row-occupancy)
      ncols_k = sum_c [colour k present in col c]
      donut_k = (cnt_k < nrows_k*ncols_k) AND (k>=1)   -> exactly one true
      colidx  = sum_k k*donut_k   (scalar)
  Output one-hot placed at cell (0,0), routed into the FREE output.

Memory: dominant intermediates are the two fp16 occupancy planes
  rowoc [1,10,30,1] and coloc [1,10,1,30] (600B each). Everything else is
  [1,10,1,1] / [1,1,1,1] scalars; the final Pad carrier is [1,1,30,30] fp16.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- per-colour pixel count: [1,10,1,1] (fp32) ------------------------
    n("ReduceSum", ["input"], "cnt", axes=[2, 3], keepdims=1)  # [1,10,1,1]

    # ---- per-colour row / col occupancy (fp32) ----------------------------
    n("ReduceMax", ["input"], "rowoc", axes=[3], keepdims=1)     # [1,10,30,1] f32
    n("ReduceMax", ["input"], "coloc", axes=[2], keepdims=1)     # [1,10,1,30] f32

    # ---- #occupied rows / cols per colour (fp32, <=30 exact) --------------
    n("ReduceSum", ["rowoc"], "nrows", axes=[2], keepdims=1)     # [1,10,1,1] f32
    n("ReduceSum", ["coloc"], "ncols", axes=[3], keepdims=1)     # [1,10,1,1] f32
    n("Mul", ["nrows", "ncols"], "area")                         # [1,10,1,1] f32 (<=900 exact)

    # ---- donut = (cnt < area) AND (k>=1) ----------------------------------
    n("Less", ["cnt", "area"], "holed")                          # bool [1,10,1,1]
    kge1 = (np.arange(10) >= 1).reshape(1, 10, 1, 1)
    init("kge1", kge1, np.bool_)
    n("And", ["holed", "kge1"], "donut")                         # bool [1,10,1,1]

    # ---- colidx = sum_k k*donut (scalar) ----------------------------------
    n("Cast", ["donut"], "donutf", to=F32)                       # [1,10,1,1] f32
    kvec = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("kvec", kvec, np.float32)
    n("Mul", ["donutf", "kvec"], "kd")                           # [1,10,1,1]
    n("ReduceSum", ["kd"], "colidx", axes=[1], keepdims=1)       # [1,1,1,1] f32

    # ---- SEPARABLE one-hot at cell (0,0): NO 30x30 carrier ----------------
    # output[ch,r,c] = (ch==colidx) AND (r==0) AND (c==0).
    # chsel = Equal(colidx, arange) -> [1,10,1,1] bool (tiny).
    # rowsel/colsel are const (index==0). Associate so the largest intermediate
    # is And(colsel,chsel)=[1,10,1,30] bool=300B; final And -> [1,10,30,30] FREE.
    arange = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("arange", arange, np.float32)
    n("Equal", ["colidx", "arange"], "chsel")                    # [1,10,1,1] bool
    rowsel = (np.arange(30) == 0).reshape(1, 1, 30, 1)
    colsel = (np.arange(30) == 0).reshape(1, 1, 1, 30)
    init("rowsel", rowsel, np.bool_)
    init("colsel", colsel, np.bool_)
    n("And", ["chsel", "colsel"], "chc")                         # [1,10,1,30] bool (300B)
    n("And", ["chc", "rowsel"], "output")                        # [1,10,30,30] bool FREE

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task291", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
