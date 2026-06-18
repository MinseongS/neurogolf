"""task350 (ARC-AGI dbc1a6ce) — connect collinear blue pixels with cyan spans.

Rule (from generator):
  Grid is height x width (width in 8..24, height in [width-2, width+2], so height<=26,
  width<=24), placed top-left of the 30x30 canvas (everything else background color 0).
  Random blue(1) pixels are scattered.  In the OUTPUT, for every pair of blue pixels
  sharing a ROW, the cells strictly between them are filled cyan(8) (unless already blue);
  likewise for every pair sharing a COLUMN.  Per row the closed span [min blue col,
  max blue col] becomes blue-or-cyan; same per column.  Blue pixels stay blue.

  cell becomes cyan iff (it is NOT blue) AND it lies in some row-span OR some col-span:
    h_span[r,c] = (blue in row r at col<=c) AND (blue in row r at col>=c)
    v_span[r,c] = (blue in col c at row<=r) AND (blue in col c at row>=r)
  (A span cell that is not blue must be background, so no separate bg test is needed.)
  Off-grid is background with no blue, so spans never leak off the grid.

Encoding (route the 10-ch expansion into the FREE Where output):
  Work on the active 26x24 (rows x cols) canvas.  B = blue plane, cast f16.
  prefix/suffix-OR per row & per col INDEPENDENTLY via triangular {0,1} MatMul (task070
  lever); the rule is NOT separable into row(x)col so full 2-D planes are required:
    leftOR  = B @ Uc , rightOR = B @ Lc     (col triangulars, contract cols)
    upOR    = Lr @ B , downOR  = Ur @ B     (row triangulars, contract rows)
  hspan = (leftOR*rightOR)>0 ; vspan = (upOR*downOR)>0 ; span = hspan|vspan.
  fill = span AND (B==0) ; output = Where(fill_padded_to_30, cyan_onehot, input).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8

N = 30
HR = 26  # active rows  (height <= 26)
WC = 24  # active cols   (width  <= 24)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- blue plane (channel 1) on the HRxWC active canvas, cast to f16 -----
    init("bl_s", np.array([1, 0, 0], np.int64), np.int64)
    init("bl_e", np.array([2, HR, WC], np.int64), np.int64)
    init("bl_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "bl_s", "bl_e", "bl_ax"], "blue_f32")  # [1,1,HR,WC] f32
    n("Cast", ["blue_f32"], "B", to=F16)  # [1,1,HR,WC] f16

    # prefix/suffix-OR along each axis via fp16 MaxPool with asymmetric padding
    # (full-length 1-D window; one-sided pad = directional running max). No params.
    #   leftOR[r,c]  = max_{k<=c} B[r,k]   (pad col-begin)
    #   rightOR[r,c] = max_{k>=c} B[r,k]   (pad col-end)
    n("MaxPool", ["B"], "leftOR", kernel_shape=[1, WC], pads=[0, WC - 1, 0, 0])
    n("MaxPool", ["B"], "rightOR", kernel_shape=[1, WC], pads=[0, 0, 0, WC - 1])
    n("Mul", ["leftOR", "rightOR"], "hprod")  # >0 iff in h-span
    #   upOR[r,c]    = max_{k<=r} B[k,c]   (pad row-begin)
    #   downOR[r,c]  = max_{k>=r} B[k,c]   (pad row-end)
    n("MaxPool", ["B"], "upOR", kernel_shape=[HR, 1], pads=[HR - 1, 0, 0, 0])
    n("MaxPool", ["B"], "downOR", kernel_shape=[HR, 1], pads=[0, 0, HR - 1, 0])
    n("Mul", ["upOR", "downOR"], "vprod")     # >0 iff in v-span

    # span = hprod>0 OR vprod>0 ; a cyan cell is a non-blue span cell.
    init("ZH", np.array(0.0, np.float16), np.float16)
    n("Greater", ["hprod", "ZH"], "hb")       # [1,1,HR,WC] bool
    n("Greater", ["vprod", "ZH"], "vb")
    n("Or", ["hb", "vb"], "span")
    init("ONEH", np.array(1.0, np.float16), np.float16)
    n("Less", ["B", "ONEH"], "notblue")       # B==0 (bool)
    n("And", ["span", "notblue"], "fill_s")   # [1,1,HR,WC] bool

    # ---- pad back to 30x30 for the Where cond ------------------------------
    n("Cast", ["fill_s"], "fill_u8", to=U8)
    init("pads", np.array([0, 0, 0, 0, 0, 0, N - HR, N - WC], np.int64), np.int64)
    init("ZU8", np.array(0, np.uint8), np.uint8)
    n("Pad", ["fill_u8", "pads", "ZU8"], "fill30", mode="constant")  # [1,1,30,30] u8
    n("Cast", ["fill30"], "fill", to=BOOL)

    # ---- cyan one-hot (color 8) --------------------------------------------
    oh = np.zeros((1, 10, 1, 1), np.float32)
    oh[0, 8, 0, 0] = 1.0
    init("cyan_oh", oh, np.float32)

    n("Where", ["fill", "cyan_oh", "input"], "output")

    x = helper.make_tensor_value_info("input", F32, [1, 10, N, N])
    y = helper.make_tensor_value_info("output", F32, [1, 10, N, N])
    graph = helper.make_graph(nodes, "task350", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
