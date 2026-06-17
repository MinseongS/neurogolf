"""task100 (ARC-AGI 445eab21) — output the colour of the LARGER-area box.

Rule (from the generator):
  The grid (active size=10, on a 30x30 canvas) holds TWO hollow rectangle
  OUTLINES drawn in two distinct colours `colors[0]`, `colors[1]`.  Each box has
  width `wide` and height `tall`; the outline spans the box's full bounding box.
  The output is a 2x2 grid in which ALL FOUR cells take the colour of the box
  with the LARGER area (wide*tall).  `xpose` may transpose the grid but does not
  change which box is larger.

  Therefore: per colour channel c=1..9 the bbox AREA = rowspan*colspan exactly
  equals wide*tall (the outline reaches every edge of its bbox).  The winning
  colour `w` = argmax over present channels of the bbox area.  Output = a bool
  one-hot that is True only at channel w and only at the 4 top-left cells
  (r<2 and c<2); all other cells / channels are False (matches the harness,
  which leaves out-of-grid cells all-zero).

Pipeline (ONNX, opset 11):
  1. rowocc[1,10,30,1] / colocc[1,10,1,30] = ReduceMax(input) over each axis.
  2. per-channel rmin/rmax/cmin/cmax via ramp-Where + ReduceMin/Max (fp16).
  3. area = (rmax-rmin+1)*(cmax-cmin+1) ; zero for ch0 and absent channels.
  4. w = ArgMax(area) over channel axis ; build channel one-hot [1,10,1,1].
  5. cellmask[1,1,30,30] = (r<2) AND (c<2) ; output = And(onehot, cellmask).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
I64 = TensorProto.INT64
B = TensorProto.BOOL


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- per-channel occupancy profiles --------------------------------------
    n("ReduceMax", ["input"], "rowocc", axes=[3], keepdims=1)   # [1,10,30,1] f32
    n("ReduceMax", ["input"], "colocc", axes=[2], keepdims=1)   # [1,10,1,30] f32

    # active grid is size=10, so all boxes live in rows/cols 0..9 -> slice the
    # fp32 occupancy profiles to the 10-wide active region before thresholding;
    # this shrinks every downstream fp16 ramp-Where plane 3x ([..30..]->[..10..]).
    init("s0", np.array([0], np.int64), np.int64)
    init("s10", np.array([10], np.int64), np.int64)
    init("ax2", np.array([2], np.int64), np.int64)
    init("ax3", np.array([3], np.int64), np.int64)
    n("Slice", ["rowocc", "s0", "s10", "ax2"], "rowocc10")      # [1,10,10,1] f32
    n("Slice", ["colocc", "s0", "s10", "ax3"], "colocc10")      # [1,10,1,10] f32

    # rowspan = #occupied rows = tall ; colspan = #occupied cols = wide.  Because
    # the box outline reaches every edge of its bbox, every row in [row,row+tall)
    # and every col in [col,col+wide) is occupied, so a simple ReduceSum of the
    # 0/1 occupancy profile yields the exact tall / wide -> NO ramp/min/max needed.
    n("ReduceSum", ["rowocc10"], "rsp", axes=[2], keepdims=1)   # [1,10,1,1] f32 (tall)
    n("ReduceSum", ["colocc10"], "csp", axes=[3], keepdims=1)   # [1,10,1,1] f32 (wide)
    n("Mul", ["rsp", "csp"], "area")                            # [1,10,1,1] f32

    # ch0 is background: every active cell has ch0=1 so it spans the full grid and
    # would falsely win.  Force ch0 area := -1 so it never wins; absent channels
    # have area 0 which already loses to any present box (area >= 9).
    ch0kill = np.zeros((1, 10, 1, 1), np.bool_)
    ch0kill[0, 0, 0, 0] = True
    init("ch0kill", ch0kill, np.bool_)
    init("neg1", np.array(-1.0, np.float32), np.float32)
    n("Where", ["ch0kill", "neg1", "area"], "area2")            # [1,10,1,1] f32

    # ---- winner channel = argmax area over channel axis ----------------------
    n("ArgMax", ["area2"], "w_i", axis=1, keepdims=1)           # [1,1,1,1] int64
    # one-hot over the 10 channels: chan-ramp == w
    init("chan", np.arange(10, dtype=np.int64).reshape(1, 10, 1, 1), np.int64)
    n("Equal", ["chan", "w_i"], "onehot")                       # [1,10,1,1] bool

    # ---- 2x2 top-left cell mask ----------------------------------------------
    init("rr2", np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1), np.float32)
    init("rc2", np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30), np.float32)
    init("two", np.array(2.0, np.float32), np.float32)
    n("Less", ["rr2", "two"], "rin")                            # [1,1,30,1] bool
    n("Less", ["rc2", "two"], "cin")                            # [1,1,1,30] bool

    # final: True only at channel w AND the 2x2 cells.  Associate broadcasts so
    # only a small [1,10,30,1] intermediate forms, never a [1,1,30,30] cellmask.
    n("And", ["onehot", "rin"], "ow")                           # [1,10,30,1] bool
    n("And", ["ow", "cin"], "output")                           # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task100", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
