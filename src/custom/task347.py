"""Task 347 (ARC-AGI dae9d2b5) — overlay two half-grids -> pink presence map.

Rule (from the generator): the input is a (2*size)-row x size-col grid built
from two size x size sub-grids stacked HORIZONTALLY in the harness 30x30 layout:
  - color 4 pixels live at (r, c)        (left  block, columns 0..size-1)
  - color 3 pixels live at (r, c+size)   (right block, columns size..2*size-1)
The OUTPUT is a size x size grid where out[r][c] = pink (6) iff a 4 pixel was at
(r,c) in the left block OR a 3 pixel was at (r,c) in the right block; every other
in-grid cell is background (0); everything outside the size x size grid is unset.

size is always 3 (common.random_pixels(size,size); validate uses size=3).

Encoding (Pad-into-free-output, uint8 whole-pipeline):
  fg(3x3 bool) = Or( ch4[rows0:3,cols0:3] , ch3[rows0:3,cols3:6] )
  fgu = Cast(fg, uint8)        # pink presence (-> channel 6)
  bgu = Cast(Not(fg), uint8)   # background    (-> channel 0)
  oneh[1,7,3,3] = Concat( bgu, zeros[1,5,3,3], fgu )   # bg@0 ... pink@6
  output = Pad(oneh -> [1,10,30,30], 0)                # the FREE uint8 output

No Conv, no 30x30 carrier plane; every working tensor is <= [1,7,3,3].  Output
declared UINT8 (harness scores out>0, so a {0,1} one-hot passes identically).

Result: mem 189, params 69 -> 19.45 pts (beats public import 18.53 by +0.92),
isolated fresh 200/200.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
B = TensorProto.BOOL

SIZE = 3


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- slice the two source blocks (the only fp32 working planes, 36B each) ----
    init("ax", np.array([1, 2, 3], np.int64), np.int64)
    init("a_st", np.array([4, 0, 0], np.int64), np.int64)
    init("a_en", np.array([5, SIZE, SIZE], np.int64), np.int64)
    n("Slice", ["input", "a_st", "a_en", "ax"], "a")          # ch4 left  [1,1,3,3]
    init("b_st", np.array([3, 0, SIZE], np.int64), np.int64)
    init("b_en", np.array([4, SIZE, 2 * SIZE], np.int64), np.int64)
    n("Slice", ["input", "b_st", "b_en", "ax"], "b")          # ch3 right [1,1,3,3]

    # ---- fg presence (bool), then bg/fg one-hot planes (uint8) ----
    n("Cast", ["a"], "ab", to=B)
    n("Cast", ["b"], "bb", to=B)
    n("Or", ["ab", "bb"], "fg")                               # [1,1,3,3] bool
    n("Not", ["fg"], "bgb")                                   # [1,1,3,3] bool
    n("Cast", ["fg"], "fgu", to=U8)                           # [1,1,3,3] u8 -> ch6
    n("Cast", ["bgb"], "bgu", to=U8)                          # [1,1,3,3] u8 -> ch0

    # ---- assemble channels 0..6 (bg@0, pink@6), then Pad into the FREE output ----
    init("z5", np.zeros((1, 5, SIZE, SIZE), np.uint8), np.uint8)
    n("Concat", ["bgu", "z5", "fgu"], "oneh", axis=1)         # [1,7,3,3] u8

    pads = np.array([0, 0, 0, 0, 0, 3, 30 - SIZE, 30 - SIZE], np.int64)
    init("pads", pads, np.int64)
    init("padv", np.array(0, np.uint8), np.uint8)
    n("Pad", ["oneh", "pads", "padv"], "output", mode="constant")  # [1,10,30,30]

    in_vi = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    out_vi = helper.make_tensor_value_info("output", U8, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task347", [in_vi], [out_vi], inits)
    model = helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_operatorsetid("", 13)])
    return model
