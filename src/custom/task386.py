"""Task 386 (ARC-AGI f2829549) — NOR of the two half-grids -> green.

Rule (from the generator): the 4x7 input has a blue separator column at col 3.
Left half (cols 0..2) carries orange (colour 7) pixels; right half (cols 4..6)
carries gray (colour 5) pixels.  The 4x3 OUTPUT is green (colour 3) at cell
(r,c) iff BOTH the left cell (r,c) is empty AND the right cell (r,c) is empty:
    output[r][c] = green  iff  left[r][c]==0  AND  right[r][c]==0   (else 0).
Geometry is fixed (width=3, height=4) for every instance of this task.

Encoding (minimal):
  * a = orange plane (channel 7) over rows 0..3, cols 0..2  -> [1,1,4,3] fp32
  * b = gray   plane (channel 5) over rows 0..3, cols 4..6  -> [1,1,4,3] fp32
  * green mask = NOT(a OR b)  (bool, [1,1,4,3])
  * label L (4x3 uint8) = 3 where green else 0 (0 == background colour, which
    every non-green output cell carries).  Pad L spatially to 30x30 with a
    sentinel 10 (off-grid cells are NOT a valid colour -> all-false one-hot).
  * free BOOL output = Equal(L, arange[1,10,1,1]):  in-grid green cells set
    channel 3, in-grid non-green cells set channel 0 (background), off-grid
    cells (sentinel 10) match no channel -> all-zero, exactly the one-hot.
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

    # ---- slice the two half-grids (orange ch7 left, gray ch5 right) ----
    init("a_s", np.array([7, 0, 0], np.int64), np.int64)
    init("a_e", np.array([8, 4, 3], np.int64), np.int64)
    init("ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "a_s", "a_e", "ax"], "a")     # [1,1,4,3] fp32 (48B)

    init("b_s", np.array([5, 0, 4], np.int64), np.int64)
    init("b_e", np.array([6, 4, 7], np.int64), np.int64)
    n("Slice", ["input", "b_s", "b_e", "ax"], "b")     # [1,1,4,3] fp32 (48B)

    # ---- green mask = NOT(a OR b) ----
    n("Cast", ["a"], "ab", to=B)                       # [1,1,4,3] bool
    n("Cast", ["b"], "bb", to=B)                       # [1,1,4,3] bool
    n("Or", ["ab", "bb"], "occ")                       # [1,1,4,3] bool
    n("Not", ["occ"], "grn")                           # [1,1,4,3] bool

    # ---- label L (4x3 uint8): 3 where green else 0 ----
    init("u3", np.array(3, np.uint8), np.uint8)
    init("u0", np.array(0, np.uint8), np.uint8)
    n("Where", ["grn", "u3", "u0"], "Lsm")             # [1,1,4,3] uint8

    # one-hot expand to 10 channels on the SMALL 4x3 block only (120 elems),
    # then Pad spatially -> 30x30 (off-grid zero-fill = all-channel zero).
    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["Lsm", "chan"], "ohsm")                # [1,10,4,3] BOOL (120B)

    # opset-13 Pad accepts bool -> drop the uint8 cast; output BOOL directly.
    init("pads", np.array([0, 0, 0, 0, 0, 0, 26, 27], np.int64), np.int64)
    init("bf", np.array(False, np.bool_), np.bool_)
    n("Pad", ["ohsm", "pads", "bf"], "output", mode="constant")  # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task386", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 13)])
