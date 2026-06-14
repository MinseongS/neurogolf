"""Task 256 (ARC-AGI a65b410d): staircase triangles from a red line.

Rule (verified exact on all 266 stored examples, derived from the ARC-GEN
generator). The input has a single horizontal red(2) line of `length` cells in
row `row`, columns 0..length-1. The generator builds the output with:
  * red    on row == row,   cols 0..length-1
  * green  on rows r < row, cols 0..(length+row-r)-1
  * blue   on rows r > row, cols 0..(length+row-r)-1
All three families share one width formula W(r) = length + row - r, so a cell
(r,c) is coloured iff 0 <= c < W(r) (W(r) > 0 is implied: c>=0). The generator
guarantees row+length = triangle < grid size, so the coloured region never spills
past the grid into off-canvas background.

Recover the two scalars from the input red line:
  * length = total red cells.
  * row    = sum_r r * (row r contains red)  (exactly one row has red).

Graph: per-row/per-channel counts via MatMul(input, ones) -> red row counts
[1,1,30,1] -> length + row scalars -> width plane W(r) -> mask = (COL < W(r))
[1,1,30,30] bool -> per-row colour code in {1,2,3} -> Gather a [4,10] one-hot
table -> Cvec[1,10,30,1] -> single Where into the free output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..builders import _model


def build(task):
    inits = []
    nodes = []
    vinfos = []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    def vi(name, dtype, shape):
        vinfos.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    # ---- red (channel 2) per-row counts in ONE width-30 valid Conv ---------
    # weight [out=1, in=10, kh=1, kw=30] picks channel 2 and sums the whole row;
    # valid padding collapses the width to 1 -> rowcnt[1,1,30,1].
    Wrow = np.zeros((1, 10, 1, 30), np.float32)
    Wrow[0, 2, 0, :] = 1.0
    init("Wrow", Wrow, np.float32)
    n("Conv", ["input", "Wrow"], "rowcnt")                    # [1,1,30,1] f32
    vi("rowcnt", TensorProto.FLOAT, [1, 1, 30, 1])

    # ---- length = sum of red row counts (scalar int) -----------------------
    n("ReduceSum", ["rowcnt"], "lenf", axes=[2], keepdims=1)  # [1,1,1,1] f32
    vi("lenf", TensorProto.FLOAT, [1, 1, 1, 1])
    n("Cast", ["lenf"], "LEN", to=TensorProto.INT32)          # [1,1,1,1] i32
    vi("LEN", TensorProto.INT32, [1, 1, 1, 1])

    # ---- row = index of the (unique) row containing red --------------------
    init("zero_f", np.array(0.0, np.float32), np.float32)
    n("Greater", ["rowcnt", "zero_f"], "rowhasb")             # [1,1,30,1] bool
    vi("rowhasb", TensorProto.BOOL, [1, 1, 30, 1])
    n("Cast", ["rowhasb"], "rowhas", to=TensorProto.INT32)    # [1,1,30,1] i32
    vi("rowhas", TensorProto.INT32, [1, 1, 30, 1])
    iota_r = np.arange(30, dtype=np.int32).reshape(1, 1, 30, 1)
    init("Ir", iota_r, np.int32)                              # [1,1,30,1] = r
    n("Mul", ["rowhas", "Ir"], "rowidxm")                     # [1,1,30,1] i32
    vi("rowidxm", TensorProto.INT32, [1, 1, 30, 1])
    n("ReduceSum", ["rowidxm"], "ROW", axes=[2], keepdims=1)  # [1,1,1,1] i32
    vi("ROW", TensorProto.INT32, [1, 1, 1, 1])

    # ---- width plane W(r) = LEN + ROW - r ----------------------------------
    n("Add", ["LEN", "ROW"], "LpR")                           # [1,1,1,1] i32
    vi("LpR", TensorProto.INT32, [1, 1, 1, 1])
    n("Sub", ["LpR", "Ir"], "Wr")                            # [1,1,30,1] i32
    vi("Wr", TensorProto.INT32, [1, 1, 30, 1])

    # ---- coloured mask = (COL < W(r)) --------------------------------------
    iota_c = np.arange(30, dtype=np.int32).reshape(1, 1, 1, 30)
    init("Ic", iota_c, np.int32)                              # [1,1,1,30] = c
    n("Less", ["Ic", "Wr"], "mask")                          # [1,1,30,30] bool
    vi("mask", TensorProto.BOOL, [1, 1, 30, 30])

    # ---- per-row colour code in {1,2,3}: green(3) r<ROW, red(2) r==ROW,
    #      blue(1) r>ROW.  code = 2 + (r<ROW) - (r>ROW). ---------------------
    n("Greater", ["Ir", "ROW"], "belowb")                    # r>ROW [1,1,30,1]
    vi("belowb", TensorProto.BOOL, [1, 1, 30, 1])
    n("Less", ["Ir", "ROW"], "aboveb")                       # r<ROW [1,1,30,1]
    vi("aboveb", TensorProto.BOOL, [1, 1, 30, 1])
    n("Cast", ["belowb"], "belowi", to=TensorProto.INT32)   # [1,1,30,1] i32
    vi("belowi", TensorProto.INT32, [1, 1, 30, 1])
    n("Cast", ["aboveb"], "abovei", to=TensorProto.INT32)   # [1,1,30,1] i32
    vi("abovei", TensorProto.INT32, [1, 1, 30, 1])
    init("two_i", np.array(2, np.int32), np.int32)
    n("Sub", ["abovei", "belowi"], "adiff")                 # (r<ROW)-(r>ROW)
    vi("adiff", TensorProto.INT32, [1, 1, 30, 1])
    n("Add", ["adiff", "two_i"], "codes4")                  # 2 + diff
    vi("codes4", TensorProto.INT32, [1, 1, 30, 1])
    # flatten to [30] for Gather
    init("shp30", np.array([30], np.int64), np.int64)
    n("Reshape", ["codes4", "shp30"], "codes")             # [30] i32
    vi("codes", TensorProto.INT32, [30])

    # ---- colour one-hot table T[4,10] (rows index by code 1/2/3) ----------
    T = np.zeros((4, 10), np.float16)
    T[1, 1] = 1.0   # blue
    T[2, 2] = 1.0   # red
    T[3, 3] = 1.0   # green
    init("T", T, np.float16)                                # [4,10] fp16
    n("Gather", ["T", "codes"], "Crows", axis=0)           # [30,10] fp16
    vi("Crows", TensorProto.FLOAT16, [30, 10])
    n("Transpose", ["Crows"], "Ct", perm=[1, 0])           # [10,30] fp16
    vi("Ct", TensorProto.FLOAT16, [10, 30])
    init("shpC", np.array([1, 10, 30, 1], np.int64), np.int64)
    n("Reshape", ["Ct", "shpC"], "Cvec16")                 # [1,10,30,1] fp16
    vi("Cvec16", TensorProto.FLOAT16, [1, 10, 30, 1])
    n("Cast", ["Cvec16"], "Cvec", to=TensorProto.FLOAT)    # [1,10,30,1] f32
    vi("Cvec", TensorProto.FLOAT, [1, 10, 30, 1])

    # ---- output = Where(mask, Cvec, input) ---------------------------------
    n("Where", ["mask", "Cvec", "input"], "output")         # [1,10,30,30] f32

    return _model(nodes, inits, vinfos)
