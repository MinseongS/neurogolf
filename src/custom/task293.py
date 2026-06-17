"""task293 (ARC-AGI ba97ae07) — swap which crossing line is on top at the intersection.

Rule (from the generator):
  A H x W grid (5..15) placed top-left of the 30x30 canvas has ONE vertical line
  (cols [off0,off0+thick0), spanning all H rows, colour colors[0]) and ONE horizontal
  line (rows [off1,off1+thick1), spanning all W cols, colour colors[1]); thick in {1,2}.
  INPUT draws `first` then `second`; OUTPUT swaps the draw order.  The ONLY difference
  input->output is the intersection rectangle (horiz_rows x vert_cols): the colour there
  flips from the input's top colour to the OTHER line's colour.  (Off-grid cells are
  all-zero, not background-channel — convert_to_numpy never touches them.)

  So output = input EXCEPT the intersection cells, which take the colour of the line NOT
  currently shown there (the two non-bg colours are colors[0],colors[1]; the intersection
  shows one of them, the output shows the other).

Encoding (route the 10-ch expansion into the FREE Where output, task033 idiom):
  - colcount[c] = #coloured cells in column c via ONE no-pad Conv (kernel 0 on ch0, 1 on
    ch1..9) -> [1,1,1,30].  rowcount[r] likewise -> [1,1,30,1].  (no 30x30 plane)
  - H = #nonzero rows = grid height; W = #nonzero cols = grid width.
  - vert_col[c]  = (colcount==H)  (the vertical line is the only fully-filled column).
  - horiz_row[r] = (rowcount==W).
  - inter_mask[1,1,30,30] = vert_col[1,1,1,30] AND horiz_row[1,1,30,1] (bool; ~900B).
  - present[k] = (ReduceSum(input,[2,3])>0) for k=1..9 -> the two line colours [1,10,1,1].
  - inter_count[k] = sum over the intersection per channel via TWO MatMuls (contract cols
    by vert_col -> [1,10,30,1]; contract rows by horiz_row -> [1,10,1,1]).  inter = >0.
  - bottom[k] = present AND NOT inter  (the colour to paint at the intersection).
  - output = Where(inter_mask, bottom[1,10,1,1], input).
  Dominant intermediate: the single [1,10,30,1] MatMul step (1200B fp32) + the 900B bool
  inter_mask; everything else is <=120B vectors -> mem ~2.4KB.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- per-column / per-row coloured counts via no-pad Conv ---------------
    Wcol = np.ones((1, 10, 30, 1), np.float32); Wcol[0, 0, :, :] = 0.0
    Wrow = np.ones((1, 10, 1, 30), np.float32); Wrow[0, 0, :, :] = 0.0
    init("Wcol", Wcol, np.float32)
    init("Wrow", Wrow, np.float32)
    n("Conv", ["input", "Wcol"], "colcount")   # [1,1,1,30] f32
    n("Conv", ["input", "Wrow"], "rowcount")   # [1,1,30,1] f32

    # ---- vert_col / horiz_row = the column/row with MAX coloured count -------
    # A vertical-line column is fully filled (count = H = grid height), strictly more
    # than any other column (thick<=2 < H); likewise the horizontal line row = max.
    init("ZF", np.array(0.0, np.float32), np.float32)
    n("ReduceMax", ["colcount"], "colmax", axes=[2, 3], keepdims=1)  # = H
    n("ReduceMax", ["rowcount"], "rowmax", axes=[2, 3], keepdims=1)  # = W
    n("Equal", ["colcount", "colmax"], "vert_col_b")    # [1,1,1,30] bool
    n("Equal", ["rowcount", "rowmax"], "horiz_row_b")   # [1,1,30,1] bool
    n("And", ["vert_col_b", "horiz_row_b"], "inter_mask")  # [1,1,30,30] bool

    # ---- present colours: ReduceSum over space, >0, mask ch0 ----------------
    n("ReduceSum", ["input"], "cnt", axes=[2, 3], keepdims=1)  # [1,10,1,1] f32
    n("Greater", ["cnt", "ZF"], "present_b")                   # [1,10,1,1] bool
    notbg = np.ones((1, 10, 1, 1), bool); notbg[0, 0, 0, 0] = False
    init("NOTBG", notbg, bool)
    n("And", ["present_b", "NOTBG"], "present")                # [1,10,1,1] bool

    # ---- inter colour: sum over the intersection per channel (2 MatMuls) -----
    n("Cast", ["vert_col_b"], "vert_col_f", to=F32)       # [1,1,1,30]
    n("Cast", ["horiz_row_b"], "horiz_row_f", to=F32)     # [1,1,30,1]
    n("Transpose", ["vert_col_f"], "vert_col_col", perm=[0, 1, 3, 2])  # [1,1,30,1]
    n("MatMul", ["input", "vert_col_col"], "byrow")       # [1,10,30,1]
    n("Transpose", ["horiz_row_f"], "horiz_row_rowT", perm=[0, 1, 3, 2])  # [1,1,1,30]
    n("MatMul", ["horiz_row_rowT", "byrow"], "inter_count")  # [1,10,1,1]
    n("Greater", ["inter_count", "ZF"], "inter_present")     # [1,10,1,1] bool

    # ---- bottom colour = present AND NOT inter_present ----------------------
    n("Not", ["inter_present"], "not_inter")
    n("And", ["present", "not_inter"], "bottom_b")        # [1,10,1,1] bool
    n("Cast", ["bottom_b"], "bottom", to=F32)             # [1,10,1,1] f32 one-hot

    # ---- output = Where(inter_mask, bottom, input) --------------------------
    n("Where", ["inter_mask", "bottom", "input"], "output")

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task293", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
