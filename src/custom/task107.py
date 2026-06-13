"""Task 107 (469497ad): zoom-upscale a 5x5 grid by a variable factor and
overlay red (color 2) corner-ray decorations around the 2x2 "box".

Rule (from the ARC-GEN generator): the input is always 5x5; the output is
(5*factor)x(5*factor) where factor = (#distinct colors in the last row) + 1.
The output is the kron upscale of the input by factor*factor, overlaid with red
at four diagonal corner-rays of length `factor` emanating from the corners of
the (upscaled) 2x2 box. Red is only ever drawn on background cells.

Single-gather design (one int8 canvas):
  Flatten the one-hot input to [1,10,900] and append a constant "red column"
  [1,10,1] (channel 2 = 1, all else 0) -> src [1,10,901] int8. A precomputed
  index map idx[30,30] (looked up from a [15,30,30] table by the key
  boxcase*5 + (factor-2)) gives, for each output cell, the flat input position
  to copy -- or 900 (the red column) for red cells. Gather(src, idx, axis=2)
  produces the finished [1,10,30,30] int8 answer in one shot; a final Cast to
  float writes `output` (free). Pointing red cells at the red column both sets
  channel 2 and clears channel 0 automatically.

Two scalars index the table:
  * factor-2 in {0..4}: factor = transitions_in_last_row + 2; transitions =
    4 - sum_i dot(onehot[i], onehot[i+1]) over the last-row cells, so
    factor-2 = 4 - dotsum.
  * boxcase in {0,1,2} for box at (0,1)/(1,0)/(1,1): cell (0,1) has a color =>
    0; else cell (2,0) has a color => 1; else 2. (occ = sum over color
    channels 1..9; background cells have channel 0 = 1 and must be excluded.)
    boxcase = 2 - 2*occ(0,1) - occ(2,0).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ..builders import _model


def _idx_table():
    """[15,30,30] int32 flat-gather indices keyed by boxcase*5 + (factor-2).

    idx[R,C] = 900 (red column) for red cells, else (R//f)*30 + (C//f)
    (the flattened position of input cell (R//f, C//f); rows/cols >= 5*f land on
    background input cells, which are all zero).
    """
    poses = {0: (0, 1), 1: (1, 0), 2: (1, 1)}
    tab = np.zeros((15, 30, 30), np.int32)
    for bc in range(3):
        row, col = poses[bc]
        for fi, f in enumerate(range(2, 7)):
            n = 5 * f
            red = np.zeros((30, 30), bool)
            for k in range(f):
                lorow, hirow = row * f - k - 1, (row + 2) * f + k
                locol, hicol = col * f - k - 1, (col + 2) * f + k
                for r, c in [(lorow, locol), (lorow, hicol),
                             (hirow, locol), (hirow, hicol)]:
                    if 0 <= r < n and 0 <= c < n:
                        red[r, c] = True
            m = tab[bc * 5 + fi]
            for R in range(30):
                for C in range(30):
                    m[R, C] = 900 if red[R, C] else (R // f) * 30 + (C // f)
    return tab


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    init("IDX", _idx_table(), np.int32)                 # [15,30,30]
    redcol = np.zeros((1, 10, 1), np.int8); redcol[0, 2, 0] = 1
    init("REDCOL", redcol, np.int8)                     # [1,10,1]
    init("c4", np.array(4, np.int32), np.int32)
    init("c2i", np.array(2, np.int32), np.int32)
    init("c5", np.array(5, np.int32), np.int32)
    init("flat", np.array([1, 10, 900], np.int64), np.int64)

    # Slice operand initializers (opset 10: starts/ends/axes are inputs)
    init("lr_s", np.array([4, 0], np.int64), np.int64)
    init("lr_e", np.array([5, 5], np.int64), np.int64)
    init("lr_ax", np.array([2, 3], np.int64), np.int64)
    init("a_s", np.array([0], np.int64), np.int64)
    init("a_e", np.array([4], np.int64), np.int64)
    init("b_s", np.array([1], np.int64), np.int64)
    init("b_e", np.array([5], np.int64), np.int64)
    init("col_ax", np.array([3], np.int64), np.int64)
    init("p01_s", np.array([0, 1], np.int64), np.int64)
    init("p01_e", np.array([1, 2], np.int64), np.int64)
    init("p_ax", np.array([2, 3], np.int64), np.int64)
    init("p20_s", np.array([2, 0], np.int64), np.int64)
    init("p20_e", np.array([3, 1], np.int64), np.int64)
    init("ch_s", np.array([1], np.int64), np.int64)
    init("ch_e", np.array([10], np.int64), np.int64)
    init("ch_ax", np.array([1], np.int64), np.int64)

    # ----- factor-2 = 4 - sum_i dot(lr[i], lr[i+1]) -----
    n("Slice", ["input", "lr_s", "lr_e", "lr_ax"], "lr")      # [1,10,1,5]
    n("Slice", ["lr", "a_s", "a_e", "col_ax"], "lra")         # cols 0..3
    n("Slice", ["lr", "b_s", "b_e", "col_ax"], "lrb")         # cols 1..4
    n("Mul", ["lra", "lrb"], "lrm")                     # [1,10,1,4]
    n("ReduceSum", ["lrm"], "dotsum_f", keepdims=0)     # scalar f32
    n("Cast", ["dotsum_f"], "dotsum", to=onnx.TensorProto.INT32)
    n("Sub", ["c4", "dotsum"], "fidx")                  # int32 scalar in 0..4

    # ----- boxcase = 2 - 2*occ(0,1) - occ(2,0) -----
    n("Slice", ["input", "p01_s", "p01_e", "p_ax"], "p01")    # [1,10,1,1]
    n("Slice", ["p01", "ch_s", "ch_e", "ch_ax"], "p01c")      # channels 1..9
    n("ReduceSum", ["p01c"], "occ01_f", keepdims=0)
    n("Cast", ["occ01_f"], "occ01", to=onnx.TensorProto.INT32)
    n("Slice", ["input", "p20_s", "p20_e", "p_ax"], "p20")
    n("Slice", ["p20", "ch_s", "ch_e", "ch_ax"], "p20c")
    n("ReduceSum", ["p20c"], "occ20_f", keepdims=0)
    n("Cast", ["occ20_f"], "occ20", to=onnx.TensorProto.INT32)
    n("Mul", ["occ01", "c2i"], "occ01x2")
    n("Sub", ["c2i", "occ01x2"], "bc_t")
    n("Sub", ["bc_t", "occ20"], "boxcase")              # int32 scalar 0..2

    # ----- table key = boxcase*5 + (factor-2) -----
    n("Mul", ["boxcase", "c5"], "bc5")
    n("Add", ["bc5", "fidx"], "rkey_t")                 # int32, possibly [1]
    n("Squeeze", ["rkey_t"], "rkey")                    # 0-D scalar
    n("Gather", ["IDX", "rkey"], "idx2d", axis=0)       # [30,30] int32

    # ----- build src [1,10,901] int8 and gather -----
    n("Cast", ["input"], "xi", to=onnx.TensorProto.INT8)      # [1,10,30,30]
    n("Reshape", ["xi", "flat"], "f1")                        # [1,10,900] int8
    n("Concat", ["f1", "REDCOL"], "src", axis=2)              # [1,10,901] int8
    n("Gather", ["src", "idx2d"], "ans", axis=2)              # [1,10,30,30] i8
    n("Cast", ["ans"], "output", to=onnx.TensorProto.FLOAT)   # free

    return _model(nodes, inits)
