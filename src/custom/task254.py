"""task254 (ARC-AGI a61f2674) — keep the shortest bar (red) and tallest bar (blue).

Rule (from the generator):
  A `size`x`size` (size=9) grid holds `num` (4 or 5) vertical GRAY (colour 5) bars.
  Bar `idx` sits in column `2*idx + offset` and is a solid bottom-anchored run of
  height `val` (the gray cells fill rows `size-1` down to `size-val`).  Bar heights
  `vals` are sampled DISTINCT from 1..size, so the min and max heights are UNIQUE.
  Output: the bar of MINIMUM height is recoloured RED (colour 2), the bar of MAXIMUM
  height is recoloured BLUE (colour 1), and every other bar is erased (background).

Encoding (ONNX, opset 11) — fully closed-form, no detection wall:
  Each bar = one solid vertical run of gray, so the bar height = per-column gray
  pixel COUNT.  The gray cells of a bar are exactly the input gray pixels in its
  column, so we never reconstruct positions: we just recolour the gray plane.

  1. gray = input channel 5  [1,1,30,30].
  2. colcnt = ReduceSum(gray, rows) -> [1,1,1,30] bar heights (0 in empty columns).
  3. maxcol mask = Equal(colcnt, ReduceMax(colcnt))  (unique tallest bar; empties are 0<max).
  4. mincol mask = Equal(colcnt, ReduceMin(Where(colcnt>0, colcnt, BIG)))  (min over real bars).
  5. blue cells = gray AND maxcol ; red cells = gray AND mincol  (broadcast col mask over rows).
  6. Label L (uint8): in-grid bg=0, blue=1, red=2, off-grid=99.  output = Equal(L, arange[0..9])
     -> FREE bool one-hot output; off-grid (L=99) yields all-zero channels.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
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

    # ---- gray plane (channel 5), cropped to the 9x9 active grid --------------
    # The whole task lives in the top-left 9x9 region, so we keep every working
    # plane at 9x9 (81 elems) instead of 30x30 (900) — escape (3) small canvas.
    W = 9
    init("starts", np.array([0, 5, 0, 0], np.int64), np.int64)
    init("ends", np.array([1, 6, W, W], np.int64), np.int64)
    init("axes", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "starts", "ends", "axes"], "gray")    # [1,1,9,9] f32
    init("half", np.array(0.5, np.float32), np.float32)
    n("Greater", ["gray", "half"], "graymask")                 # [1,1,9,9] bool

    # ---- per-column gray count = bar heights --------------------------------
    n("ReduceSum", ["gray"], "colcnt", axes=[2], keepdims=1)   # [1,1,1,9] f32

    # ---- tallest bar (blue) : Equal to global max ----------------------------
    n("ReduceMax", ["colcnt"], "maxv", axes=[3], keepdims=1)   # [1,1,1,1]
    n("Equal", ["colcnt", "maxv"], "maxcol")                   # [1,1,1,9] bool

    # ---- shortest bar (red) : min over columns with count>0 ------------------
    init("zero", np.array(0.0, np.float32), np.float32)
    init("BIG", np.array(1000.0, np.float32), np.float32)
    n("Greater", ["colcnt", "zero"], "hascol")                 # bool [1,1,1,9]
    n("Where", ["hascol", "colcnt", "BIG"], "cnt2")            # empties -> BIG
    n("ReduceMin", ["cnt2"], "minv", axes=[3], keepdims=1)     # [1,1,1,1]
    n("Equal", ["colcnt", "minv"], "mincol")                   # [1,1,1,9] bool

    # ---- recoloured cells (broadcast col mask over rows) ---------------------
    n("And", ["graymask", "maxcol"], "blue")                   # [1,1,9,9] bool
    n("And", ["graymask", "mincol"], "red")                    # [1,1,9,9] bool

    # ---- label map L (uint8) on 9x9: bg=0, blue=1, red=2 ---------------------
    init("u1", np.array(1, np.uint8), np.uint8)
    init("u2", np.array(2, np.uint8), np.uint8)
    init("u0", np.array(0, np.uint8), np.uint8)
    n("Where", ["red", "u2", "u0"], "Lred")                    # [1,1,9,9] u8
    n("Where", ["blue", "u1", "Lred"], "Lin")                  # [1,1,9,9] u8

    # ---- pad 9x9 -> 30x30 with sentinel 99 (off-grid -> all channels 0) ------
    init("padv", np.array(99, np.uint8), np.uint8)
    init("pads",
         np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64), np.int64)
    n("Pad", ["Lin", "pads", "padv"], "L", mode="constant")    # [1,1,30,30] u8

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                        # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task254", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
