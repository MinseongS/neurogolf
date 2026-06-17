"""Task 068 (31aa019c): 3x3 red halo around the unique (count-1) pixel.

Rule (from ARC-GEN generator, 10x10 grid):
  Several distinct colours are placed on a 10x10 grid.  Index-0's colour gets
  EXACTLY one pixel; every other colour gets >=2 pixels.  Distinct colours
  (random_colors samples without replacement) so per-colour count == per-index
  count.  Hence there is exactly ONE colour with count 1 -- that is min_color.
  Output is blank except: for the single min_color pixel at (r0,c0), stamp a
  3x3 RED (colour 2) box centred there, and put min_color back in the centre.
  The unique pixel is generated with r,c in [1, size-2] = [1,8] so the 3x3 box
  never clips the grid boundary.

Encoding (label-map, Tier-B):
  cnt_k    = ReduceSum(input,[2,3])            per-channel pixel count [1,10,1,1]
  uniq_k   = Equal(cnt_k, 1)                   the single count-1 channel (bg
             ch0 count is large; other colours >=2; exactly one channel ==1)
  ucolor   = Sum_k k*uniq_k                    scalar colour index of the centre
  colf     = Conv(input, w=k)                  colour-index plane [1,1,30,30]
  center   = Equal(colf, ucolor)               the single centre cell (bg=0 and
             other colours never equal ucolor>=1, so exactly one True)
  box      = Conv(center_f, ones3x3, pad=1)>0  the 3x3 box mask
  border   = box AND NOT center                the 8 surrounding cells -> red
  L (uint8): 2 on border, ucolor on centre, 0 elsewhere
  output   = Equal(L, arange[0..9])            free BOOL one-hot

  (If ucolor==2 the centre is also red, which is correct: border|center fills
  the full 3x3 box in channel 2 and L=2 there too -- the centre branch sets
  L=ucolor=2, consistent.)
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # constants
    init("w_k", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    init("w_kr", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    init("ones3", np.ones((1, 1, 3, 3), np.float16), np.float16)
    init("one_f", np.array(1.0, np.float32), np.float32)
    init("half16", np.array(0.5, np.float16), np.float16)
    init("two_u", np.array(2, np.uint8), np.uint8)
    init("zero_u", np.array(0, np.uint8), np.uint8)
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    # slice colf to the 10x10 active region, then pad L back with a >=10
    # sentinel so the off-grid 30x30 region matches the all-zero target there.
    init("sl_st", np.array([0, 0], np.int64), np.int64)
    init("sl_en", np.array([10, 10], np.int64), np.int64)
    init("sl_ax", np.array([2, 3], np.int64), np.int64)
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 20, 20], np.int64), np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)

    # per-channel counts -> the unique (count==1) channel
    n("ReduceSum", ["input"], "cnt", axes=[2, 3], keepdims=1)        # [1,10,1,1] f32
    n("Equal", ["cnt", "one_f"], "uniq_b")                           # bool [1,10,1,1]
    n("Cast", ["uniq_b"], "uniq_f", to=TensorProto.FLOAT)
    n("Mul", ["uniq_f", "w_kr"], "uk")
    n("ReduceSum", ["uk"], "ucolor_f", axes=[1], keepdims=1)         # [1,1,1,1] f32

    # colour-index plane (single Conv input->plane), sliced to 10x10 active
    n("Conv", ["input", "w_k"], "colf30")                            # [1,1,30,30] f32
    n("Slice", ["colf30", "sl_st", "sl_en", "sl_ax"], "colf")        # [1,1,10,10] f32
    n("Equal", ["colf", "ucolor_f"], "center_b")                     # bool [1,1,10,10]
    n("Cast", ["center_b"], "center_f", to=TensorProto.FLOAT16)

    # 3x3 box via ones conv, then border = box & ~center
    n("Conv", ["center_f", "ones3"], "boxsum", pads=[1, 1, 1, 1])    # fp16 [1,1,10,10]
    n("Greater", ["boxsum", "half16"], "box_b")                      # bool box
    n("Not", ["center_b"], "ncenter_b")
    n("And", ["box_b", "ncenter_b"], "border_b")                     # bool 8-nbrs

    # label map (uint8): border->2, centre->ucolor, else 0  (10x10)
    n("Cast", ["ucolor_f"], "ucolor_u", to=TensorProto.UINT8)        # [1,1,1,1] u8
    n("Where", ["border_b", "two_u", "zero_u"], "L0")                # 0/2 [1,1,10,10]
    n("Where", ["center_b", "ucolor_u", "L0"], "L10")                # centre colour
    n("Pad", ["L10", "padpads", "padval"], "L", mode="constant")     # [1,1,30,30] u8
    n("Equal", ["L", "chan"], "output")                              # free BOOL

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task068", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
