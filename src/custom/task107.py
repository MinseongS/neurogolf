"""Task 107 (469497ad): zoom-upscale a 5x5 grid by a variable factor and
overlay red (color 2) corner-ray decorations around the 2x2 "box".

Rule (from the ARC-GEN generator): the input is always 5x5; the output is
(5*factor)x(5*factor) where factor = (#distinct colors in the last row) + 1.
The output is the kron upscale of the input by factor*factor, overlaid with red
at four diagonal corner-rays of length `factor` emanating from the corners of
the (upscaled) 2x2 box. Red is only ever drawn on background cells.

Memory floor-break (label map + final Equal):
  The previous design gathered the one-hot `src` [1,10,901] into a finished
  [1,10,30,30] int8 plane (9000B) using a [15,30,30] int32 index table
  (54000 param-bytes). Here instead:
    * Build a flat COLOUR-LABEL source lab[1,1,901] uint8 by reshaping `input`
      to [1,10,900] and reducing k*onehot over channels -> [1,1,900]; no
      [1,1,30,30] float plane is ever materialised. A red column = 2 is
      appended.
    * The upscale gather index (R//f)*30+(C//f) is computed arithmetically;
      a small uint8 RED table [15,30,30] (13500B) marks the red corner-ray
      cells, where the gather index is forced to 900 (the red column).
    * Gather(lab, idx) -> a single L[1,1,30,30] uint8 colour label. A sentinel
      10 is written outside the 5*factor x 5*factor output region (those cells
      are all-channels-off in the target). Final Equal(L, arange[0..9]) writes
      the free BOOL `output`.

Two scalars index the red table (boxcase*5 + (factor-2)):
  * factor-2 = 4 - sum_i dot(onehot[i], onehot[i+1]) over the last-row cells.
  * boxcase = 2 - 2*occ(0,1) - occ(2,0)   (box at (0,1)/(1,0)/(1,1)).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION


RED_SZ = 24  # red cells never exceed (row+2)*f+(f-1) <= 23 for row<=1, f<=6


def _red_table():
    """[15,RED_SZ,RED_SZ] uint8 red-cell mask keyed by boxcase*5 + (factor-2)."""
    poses = {0: (0, 1), 1: (1, 0), 2: (1, 1)}
    tab = np.zeros((15, RED_SZ, RED_SZ), np.uint8)
    for bc in range(3):
        row, col = poses[bc]
        for fi, f in enumerate(range(2, 7)):
            n = 5 * f
            m = tab[bc * 5 + fi]
            for k in range(f):
                lorow, hirow = row * f - k - 1, (row + 2) * f + k
                locol, hicol = col * f - k - 1, (col + 2) * f + k
                for r, c in [(lorow, locol), (lorow, hicol),
                             (hirow, locol), (hirow, hicol)]:
                    if 0 <= r < n and 0 <= c < n:
                        m[r, c] = 1
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

    I32 = TensorProto.INT32
    init("RED", _red_table(), np.uint8)                 # [15,30,30] uint8
    init("c4", np.array(4, np.int32), np.int32)
    init("c2i", np.array(2, np.int32), np.int32)
    init("c5", np.array(5, np.int32), np.int32)

    # Slice operand initializers (opset 11: starts/ends/axes are inputs)
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
    n("Cast", ["dotsum_f"], "dotsum", to=I32)
    n("Sub", ["c4", "dotsum"], "fidx")                  # int32 scalar in 0..4

    # ----- boxcase = 2 - 2*occ(0,1) - occ(2,0) -----
    n("Slice", ["input", "p01_s", "p01_e", "p_ax"], "p01")    # [1,10,1,1]
    n("Slice", ["p01", "ch_s", "ch_e", "ch_ax"], "p01c")      # channels 1..9
    n("ReduceSum", ["p01c"], "occ01_f", keepdims=0)
    n("Cast", ["occ01_f"], "occ01", to=I32)
    n("Slice", ["input", "p20_s", "p20_e", "p_ax"], "p20")
    n("Slice", ["p20", "ch_s", "ch_e", "ch_ax"], "p20c")
    n("ReduceSum", ["p20c"], "occ20_f", keepdims=0)
    n("Cast", ["occ20_f"], "occ20", to=I32)
    n("Mul", ["occ01", "c2i"], "occ01x2")
    n("Sub", ["c2i", "occ01x2"], "bc_t")
    n("Sub", ["bc_t", "occ20"], "boxcase")              # int32 scalar 0..2

    # ----- table key = boxcase*5 + (factor-2); fetch red mask [30,30] -----
    n("Mul", ["boxcase", "c5"], "bc5")
    n("Add", ["bc5", "fidx"], "rkey_t")                 # int32, possibly [1]
    n("Squeeze", ["rkey_t"], "rkey")                    # 0-D scalar
    n("Gather", ["RED", "rkey"], "red_sm", axis=0)      # [RED_SZ,RED_SZ] u8
    # pad red mask to [30,30] with 0 (no red outside the small region)
    init("rpad", np.array([0, 0, 30 - RED_SZ, 30 - RED_SZ], np.int64), np.int64)
    init("rpv", np.array(0, np.uint8), np.uint8)
    n("Pad", ["red_sm", "rpad", "rpv"], "red2d", mode="constant")  # [30,30]

    # ----- factor f (float scalar) -----
    init("c2f", np.array(2.0, np.float32), np.float32)
    n("Cast", ["fidx"], "fidx_f", to=TensorProto.FLOAT)
    n("Add", ["fidx_f", "c2f"], "fscal")                # scalar float = factor

    # ----- arithmetic upscale index (R//f)*30 + (C//f) over [30,30] -----
    Rg = np.arange(30, dtype=np.float32).reshape(30, 1)
    Cg = np.arange(30, dtype=np.float32).reshape(1, 30)
    init("Rg", Rg, np.float32)                          # [30,1]
    init("Cg", Cg, np.float32)                          # [1,30]
    n("Div", ["Rg", "fscal"], "Rdiv")
    n("Floor", ["Rdiv"], "Rf")                          # [30,1] floor(R/f)
    n("Div", ["Cg", "fscal"], "Cdiv")
    n("Floor", ["Cdiv"], "Cf")                          # [1,30] floor(C/f)
    init("c30f", np.array(30.0, np.float32), np.float32)
    n("Mul", ["Rf", "c30f"], "Rf30")                    # [30,1]
    n("Add", ["Rf30", "Cf"], "base_f")                  # [30,30] = Rf*30+Cf
    n("Cast", ["base_f"], "base_i", to=I32)             # [30,30] int32 0..624

    # ----- where red: index 900, else base index -----
    init("c900", np.array(900, np.int32), np.int32)
    n("Cast", ["red2d"], "red_b", to=TensorProto.BOOL)  # [30,30] bool
    n("Where", ["red_b", "c900", "base_i"], "idx2d")    # [30,30] int32

    # ----- colour-label source lab[1,1,901] uint8 -----
    # colour index per cell = sum_k k*onehot[k]  (1x1 Conv with kvec); the
    # [1,1,30,30] label plane is the only canvas-sized float here.
    init("kvec", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1),
         np.float32)
    n("Conv", ["input", "kvec"], "lab_f", kernel_shape=[1, 1])  # [1,1,30,30] f32
    n("Cast", ["lab_f"], "lab_u8", to=TensorProto.UINT8)        # [1,1,30,30] u8
    init("flat", np.array([1, 1, 900], np.int64), np.int64)
    n("Reshape", ["lab_u8", "flat"], "labflat")                 # [1,1,900] u8
    redcol = np.array(2, np.uint8).reshape(1, 1, 1)
    init("REDCOL", redcol, np.uint8)                            # [1,1,1]
    n("Concat", ["labflat", "REDCOL"], "labsrc", axis=2)        # [1,1,901] u8

    # ----- gather colour labels into L[1,1,30,30] -----
    n("Gather", ["labsrc", "idx2d"], "Lraw", axis=2)            # [1,1,30,30] u8

    # ----- sentinel 10 outside the 5*factor x 5*factor output region -----
    init("c5f", np.array(5.0, np.float32), np.float32)
    n("Mul", ["fscal", "c5f"], "nlim")                          # scalar = 5*factor
    Rcol = np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1)
    Crow = np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30)
    init("Rcol", Rcol, np.float32)
    init("Crow", Crow, np.float32)
    n("Less", ["Rcol", "nlim"], "rin")                          # [1,1,30,1] bool
    n("Less", ["Crow", "nlim"], "cin")                          # [1,1,1,30] bool
    n("And", ["rin", "cin"], "ingrid")                          # [1,1,30,30] bool
    init("v10", np.array(10, np.uint8), np.uint8)
    n("Where", ["ingrid", "Lraw", "v10"], "L")                  # [1,1,30,30] u8
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                         # free BOOL

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task107", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
