"""Task 167 (ARC-AGI 6e02f1e3) — distinct-colour COUNT -> fixed 3x3 gray pattern.

Rule (from the generator):
  The input is a `size`x`size` grid (size=3) whose cells are coloured
  `idxs[r*size+c] + color_offset`, idx in {0,1,2}, color_offset=2  ->  colours
  in {2,3,4}.  The OUTPUT is a 3x3 grid that is all background (0) except for
  three GRAY (5) cells whose positions are keyed ONLY on
      nc = len(set(idxs)) = number of DISTINCT colours present (1..3):
    nc==1 -> top row   : (0,0)(0,1)(0,2)
    nc==2 -> main diag : (0,0)(1,1)(2,2)
    nc==3 -> anti-diag : (0,2)(1,1)(2,0)
  Everything outside the 3x3 grid is background.

  This is the COUNT->FIXED-PATTERN tier: the whole output is determined by ONE
  scalar nc.  Recover nc with NO 30x30 plane:
    cnts = ReduceSum(input, axes=[2,3])              -> [1,10,1,1] fp32 (40B)
    presence = Greater(cnts, 0)                      -> [1,10,1,1] bool
    pres_fg  = Slice(presence, ch 1..10)             -> drop background ch0
    nc = ReduceSum(Cast(pres_fg, fp32), axis=1)      -> [1,1,1,1] scalar
  (Counting distinct PRESENT non-background channels = num_colors exactly, and
   is independent of color_offset, so it generalises.)

  The gray 3x3 mask is selected among three constant patterns by nc via a nested
  Where (Equal on the scalar).  Build the tiny [1,2,3,3] uint8 one-hot (ch0=black,
  ch5=gray) and Pad it INTO the FREE 30x30 output: no carrier plane materialises.
"""

import numpy as np
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

    # ---- nc = number of distinct present non-background colours (scalar) ----
    n("ReduceSum", ["input"], "cnts", axes=[2, 3], keepdims=1)   # [1,10,1,1] fp32
    init("zero", np.array([[[[0.0]]]], np.float32), np.float32)
    n("Greater", ["cnts", "zero"], "pres")                       # [1,10,1,1] bool
    # drop background channel 0
    init("c1", np.array([1], np.int64), np.int64)
    init("c10", np.array([10], np.int64), np.int64)
    init("ax1", np.array([1], np.int64), np.int64)
    n("Slice", ["pres", "c1", "c10", "ax1"], "presfg")           # [1,9,1,1] bool
    n("Cast", ["presfg"], "presf", to=F32)                       # [1,9,1,1] fp32
    n("ReduceSum", ["presf"], "nc", axes=[1], keepdims=1)        # [1,1,1,1] fp32

    # ---- select the gray 3x3 mask among the three fixed patterns by nc ----
    p1 = np.array([[1, 1, 1],
                   [0, 0, 0],
                   [0, 0, 0]], np.float32).reshape(1, 1, 3, 3)   # top row
    p2 = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]], np.float32).reshape(1, 1, 3, 3)   # main diag
    p3 = np.array([[0, 0, 1],
                   [0, 1, 0],
                   [1, 0, 0]], np.float32).reshape(1, 1, 3, 3)   # anti diag
    init("p1", p1, np.float32)
    init("p2", p2, np.float32)
    init("p3", p3, np.float32)

    # is2 = (nc == 2) ; is3 = (nc == 3)  (both bool scalars)
    init("two", np.array([[[[2.0]]]], np.float32), np.float32)
    init("three", np.array([[[[3.0]]]], np.float32), np.float32)
    n("Equal", ["nc", "two"], "is2")
    n("Equal", ["nc", "three"], "is3")
    # gray_f = is3 ? p3 : (is2 ? p2 : p1)  (fp32 branches; ORT-supported)
    n("Where", ["is2", "p2", "p1"], "g12")    # [1,1,3,3] fp32
    n("Where", ["is3", "p3", "g12"], "grayf")  # [1,1,3,3] fp32 gray mask (0/1)
    init("half", np.array([[[[0.5]]]], np.float32), np.float32)
    n("Greater", ["grayf", "half"], "gray")    # [1,1,3,3] bool gray mask

    # ---- tiny [1,6,3,3] one-hot: ch0=black(=NOT gray), ch5=gray; ch1..4 zero ----
    n("Not", ["gray"], "notgray")                  # bool: black cells
    n("Cast", ["notgray"], "ch0", to=U8)           # ch0 = black (uint8)
    n("Cast", ["gray"], "grayu8", to=U8)           # ch5 = gray (uint8)
    # channels 1..4 are all zero
    init("z4", np.zeros((1, 4, 3, 3), np.uint8), np.uint8)
    # ch5 = gray itself
    n("Concat", ["ch0", "z4", "grayu8"], "oneh", axis=1)   # [1,6,3,3] uint8

    # ---- Pad [1,6,3,3] -> [1,10,30,30] with 0; this Pad IS the output ----
    pads = np.array([0, 0, 0, 0, 0, 4, 27, 27], np.int64)
    init("pads", pads, np.int64)
    init("zerov", np.array(0, np.uint8), np.uint8)
    n("Pad", ["oneh", "pads", "zerov"], "output", mode="constant")

    out_vi = helper.make_tensor_value_info("output", U8, [1, 10, 30, 30])
    in_vi = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task167", [in_vi], [out_vi], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
