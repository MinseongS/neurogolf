"""task121 (ARC-AGI 5117e062) — extract the cyan-marked 3x3 sprite, recolored.

Rule (from the generator):
  A 13x13 grid holds 3-4 non-overlapping 3x3 "conway" sprites, each drawn in its
  own colour `colors[idx]` (colours are sampled from 1..9 excluding cyan=8).
  Sprite 0 is special: the CENTER cell of its 3x3 block, (brows[0]+1, bcols[0]+1),
  is overwritten with cyan (colour 8) as a marker.
  The 3x3 output is sprite 0's shape painted entirely in sprite 0's colour, with
  the center cell forced to that colour as well (output = the sprite-0 block:
  every occupied OR center cell -> colors[0], background -> 0).

  Sprites are placed with spacing>=1, so no other sprite intrudes into sprite 0's
  3x3 block: within that block the only non-background colours are colors[0] and
  the single cyan marker at the center.

Pipeline (ONNX, opset 11) — single full-grid plane:
  1. colf = sum_k w_k * input_k  (1x1 Conv), w_8 = 1000 (cyan marker), w_k = k
     otherwise.  One [1,1,30,30] f32 plane (the only big tensor).
  2. Marker (row,col): reductions of colf give per-row/col max; the row/col whose
     max is ~1000 is the marker line.  Dot with an index ramp -> scalar (r,c).
  3. Gather a 3x3 window of colf around the marker (rows r-1..r+1, c-1..c+1).
     Window values are in {0, colors[0], 1000}.
  4. colors[0] = ReduceMax over the window after mapping the 1000 (cyan) -> 0.
  5. occ = window > 0  (non-background, incl. the cyan center).
     L = Where(occ, colors[0], 0)  -> [1,1,3,3] uint8 label.
  6. Pad to 30x30 with sentinel 10 (so cells beyond the 3x3 output are all-zero),
     Equal(L, arange[0..9]) -> free BOOL one-hot output.
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

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    MARK = 1000.0  # cyan-marker magnitude (fp32-exact)

    # ---- 1. single colour/marker plane colf = sum_k w_k * input_k -----------
    w = np.arange(10, dtype=np.float32)
    w[8] = MARK
    cw = w.reshape(1, 10, 1, 1)               # Conv weight [O=1,I=10,1,1]
    init("cw", cw, np.float32)
    n("Conv", ["input", "cw"], "colf")        # [1,1,30,30] f32  (only big tensor)

    # ---- 2. marker (row, col) scalars: argmax of per-row / per-col maxima ----
    # The marker line is the unique row/col whose max == MARK (>= everything else).
    n("ReduceMax", ["colf"], "rmax", axes=[3], keepdims=1)  # [1,1,30,1]
    n("ReduceMax", ["colf"], "cmax", axes=[2], keepdims=1)  # [1,1,1,30]
    n("ArgMax", ["rmax"], "cy_r", axis=2, keepdims=1)       # [1,1,1,1] int64 = row
    n("ArgMax", ["cmax"], "cy_c", axis=3, keepdims=1)       # [1,1,1,1] int64 = col

    # ---- 3. gather a 3x3 window of colf around the marker -------------------
    init("off3", np.array([-1, 0, 1], np.int64), np.int64)
    init("shp1", np.array([1], np.int64), np.int64)

    n("Reshape", ["cy_r", "shp1"], "cy_r1")   # [1] int64
    n("Add", ["off3", "cy_r1"], "ridx")       # [3] int64  (marker in [1,11] => in-bounds)
    n("Reshape", ["cy_c", "shp1"], "cy_c1")   # [1] int64
    n("Add", ["off3", "cy_c1"], "cidx")       # [3] int64

    n("Gather", ["colf", "ridx"], "Wr", axis=2)   # [1,1,3,30]
    n("Gather", ["Wr", "cidx"], "W3", axis=3)     # [1,1,3,3] (vals 0, c0, 1000)

    # ---- 4. colors[0] = max(window with cyan-marker -> 0) -------------------
    init("zero", np.array(0.0, np.float32), np.float32)
    init("half_mark", np.array(MARK / 2.0, np.float32), np.float32)
    n("Greater", ["W3", "half_mark"], "ismark")    # [1,1,3,3] bool (the center)
    n("Where", ["ismark", "zero", "W3"], "W3n")    # marker -> 0
    n("ReduceMax", ["W3n"], "c0v", axes=[2, 3], keepdims=1)  # [1,1,1,1] = colors[0]
    n("Cast", ["c0v"], "c0u8", to=U8)

    # ---- 5. label: occupied(>0) cell -> colors[0], else 0 -------------------
    init("halff", np.array(0.5, np.float32), np.float32)
    n("Greater", ["W3", "halff"], "occ")      # [1,1,3,3] bool (incl. cyan center)
    init("u0", np.array(0, np.uint8), np.uint8)
    n("Where", ["occ", "c0u8", "u0"], "Lout")  # [1,1,3,3] uint8

    # ---- 6. one-hot the 3x3 label, then Pad straight into the FREE output ----
    # Build the one-hot at 3x3 (tiny), then Pad with 0 to [1,10,30,30]: cells
    # beyond the 3x3 output grid become all-zero (no channel set) automatically.
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["Lout", "chan"], "oh3b")      # [1,10,3,3] BOOL
    n("Cast", ["oh3b"], "oh3", to=U8)         # [1,10,3,3] uint8 (Pad needs non-bool)
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 27, 27], np.int64), np.int64)
    n("Pad", ["oh3", "padpads", "u0"], "output", mode="constant")  # [1,10,30,30] u8

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", U8, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task121", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
