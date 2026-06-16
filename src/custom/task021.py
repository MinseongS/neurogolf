"""task021 (ARC-AGI 1190e5a7) — count the grid lines, emit a bg-filled box.

Rule (from the generator, verified 0/5961 valid fresh instances):
  The input is a 2-colour grid: background `colors[0]` with some FULL horizontal
  lines and FULL vertical lines drawn in `colors[1]`.  The output is a grid of
  shape  (#horizontal-lines + 1) x (#vertical-lines + 1)  filled entirely with the
  BACKGROUND colour `colors[0]`.

  Key invariants:
    * The top-left corner cell (0,0) is NEVER on a line (the first line is at
      index >= 1), so  bg = input[0][0]  exactly.  (The most-frequent colour is
      NOT a safe proxy: the line colour can out-number the background.)
    * A "full horizontal line" is a row that contains ZERO background pixels and
      is in-grid (has any pixel).  oh = (#such rows) + 1.  Same for columns -> ow.

Pipeline (ONNX, opset 11), no full [1,10,30,30] intermediate materialised:
  1. bg one-hot  = input[:, :, 0:1, 0:1]               -> [1,10,1,1]
  2. bgmask[1,1,30,30] = ReduceSum_c(input * bgonehot) (1 where cell == bg)
     tot[1,1,30,30]    = ReduceSum_c(input)            (1 where in-grid)
  3. per-row: bgrow = ReduceSum_c-cols(bgmask); totrow = ReduceSum(tot)
     full-line row  = (totrow > 0) AND (bgrow == 0); oh = ReduceSum(rows)+1
     same on the column axis -> ow.
  4. boxmask[1,1,30,30] = (rowramp < oh) AND (colramp < ow)
  5. L[1,1,30,30] = Where(boxmask, bgidx, 10);  output = Equal(L, arange[0..9]).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
I64 = TensorProto.INT64
B = TensorProto.BOOL


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    f32, u8, i64 = np.float32, np.uint8, np.int64

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- 1. background one-hot = corner cell -------------------------------
    # Slice input[:, :, 0:1, 0:1] -> [1,10,1,1] one-hot of the bg colour.
    init("s_starts", np.array([0, 0], np.int64), np.int64)
    init("s_ends", np.array([1, 1], np.int64), np.int64)
    init("s_axes", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["input", "s_starts", "s_ends", "s_axes"], "bgonehot")  # [1,10,1,1]

    # bg colour INDEX (scalar) = sum_k k * bgonehot ; also as a [1] int64 index.
    init("chvec", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), f32)
    n("Mul", ["bgonehot", "chvec"], "bgidx_e")          # [1,10,1,1]
    n("ReduceSum", ["bgidx_e"], "bgidx_f", axes=[1], keepdims=1)  # [1,1,1,1] f32
    init("shp1", np.array([1], np.int64), np.int64)
    n("Cast", ["bgidx_f"], "bgidx_i", to=I64)
    n("Reshape", ["bgidx_i", "shp1"], "bgidx1")         # [1] int64

    # ---- 2. per-channel row / col profiles ---------------------------------
    n("ReduceSum", ["input"], "rowprof", axes=[3], keepdims=1)  # [1,10,30,1] f32
    n("ReduceSum", ["input"], "colprof", axes=[2], keepdims=1)  # [1,10,1,30] f32

    init("zero", np.array(0.0, np.float32), f32)
    init("half", np.array(0.5, np.float32), f32)

    # ---- 3a. horizontal full-line count ------------------------------------
    # bg count per row = Gather bg channel of rowprof; total per row = sum_c.
    n("Gather", ["rowprof", "bgidx1"], "bgrow", axis=1)          # [1,1,30,1] f32
    n("ReduceSum", ["rowprof"], "totrow", axes=[1], keepdims=1)  # [1,1,30,1] f32
    n("Less", ["bgrow", "half"], "no_bg_row")           # bgrow == 0 (int counts)
    n("Greater", ["totrow", "zero"], "ingrid_row")
    n("And", ["no_bg_row", "ingrid_row"], "hline")      # [1,1,30,1] bool
    n("Cast", ["hline"], "hline_f", to=F32)
    n("ReduceSum", ["hline_f"], "nh", axes=[2], keepdims=1)  # [1,1,1,1] f32

    # ---- 3b. vertical full-line count --------------------------------------
    n("Gather", ["colprof", "bgidx1"], "bgcol", axis=1)          # [1,1,1,30] f32
    n("ReduceSum", ["colprof"], "totcol", axes=[1], keepdims=1)  # [1,1,1,30] f32
    n("Less", ["bgcol", "half"], "no_bg_col")
    n("Greater", ["totcol", "zero"], "ingrid_col")
    n("And", ["no_bg_col", "ingrid_col"], "vline")      # [1,1,1,30] bool
    n("Cast", ["vline"], "vline_f", to=F32)
    n("ReduceSum", ["vline_f"], "nv", axes=[3], keepdims=1)  # [1,1,1,1] f32

    # ---- 4. output dims oh = nh+1, ow = nv+1 -------------------------------
    init("one", np.array(1.0, np.float32), f32)
    n("Add", ["nh", "one"], "oh")                       # [1,1,1,1] f32
    n("Add", ["nv", "one"], "ow")

    # box mask: rowramp < oh AND colramp < ow
    rampr = np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1)
    rampc = np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30)
    init("rampr", rampr, f32)
    init("rampc", rampc, f32)
    n("Less", ["rampr", "oh"], "rowin")                 # [1,1,30,1] bool
    n("Less", ["rampc", "ow"], "colin")                 # [1,1,1,30] bool

    # ---- 5. final one-hot output (routed into the FREE output tensor) -------
    # output[k,r,c] = (r<oh) AND (c<ow) AND (k==bg).  Associate the broadcasts so
    # no [1,1,30,30] box plane is materialised: fuse (c<ow) with (k==bg) first
    # ([1,10,1,30] = 300B), then AND with (r<oh) straight into the free output.
    n("Greater", ["bgonehot", "half"], "bgbool")        # [1,10,1,1] bool (==bg ch)
    n("And", ["colin", "bgbool"], "colbg")              # [1,10,1,30] bool
    n("And", ["rowin", "colbg"], "output")              # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task021", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
