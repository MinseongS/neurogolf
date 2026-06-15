"""Task 290 (b94a9452): "cookie/creme" square -> cropped, colour-swapped square.

Rule (from ARC-GEN generator, verified fresh):
  The input grid holds ONE solid square of side `size = thicks[0] + 2*thicks[1]`
  (thicks[i] in {1,2}, so size in {3,4,5,6}) placed at (row,col).  The whole square
  is the "cookie" colour colors[1]; a CENTRED inner thicks[0]xthicks[0] block (the
  "creme") is colour colors[0].  Because size = thicks[0] + 2*thicks[1], the inner
  block is exactly centred (margin thicks[1] on every side).
  The OUTPUT is that square cropped to the top-left corner (0,0) of a
  size x size grid with the TWO colours SWAPPED:
    output outer  = colors[0]  (input's inner / rarer colour)
    output inner  = colors[1]  (input's outer / more-frequent colour)

  Everything the output needs collapses to FOUR scalars, all read from the
  per-channel pixel COUNTS (no spatial plane at all):
    cnt[k] = #cells of colour k  (ReduceSum over H,W -> [1,10,1,1]).
    Among channels 1..9 with cnt>0 there are exactly two colours:
      c0 (output outer) = the RARER one  (count = thicks[0]^2 in {1,4})
      c1 (output inner) = the more frequent one (count = size^2 - thicks[0]^2)
    size = sqrt(cnt[c0] + cnt[c1])   (whole square = size^2 cells)
    t0   = sqrt(cnt[c0])             (inner block side, 1 or 2)
    t1   = (size - t0)/2             (centred margin)  -> inner = [t1, t1+t0)

Encoding (label-map + final Equal; NO 30x30 working plane):
  Build the size x size label map on an 8x8 canvas via separable row/col masks:
  L = c1 on the centred inner block, c0 on the rest of the square, sentinel
  (>=10) outside.  Pad 8x8 -> 30x30 (sentinel) and one final Equal(L, arange)
  writes straight into the FREE bool output.  The dominant intermediate is the
  900 B uint8 Pad feeding Equal; counts/scalars are all <= a few hundred bytes.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

WORK = 8  # active canvas side (size <= 6, inner offset <= 2 -> fits in 8)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- constants ----
    init("chidx", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    init("half_f", np.array(0.5, np.float32), np.float32)
    init("big", np.array(1e6, np.float32), np.float32)
    init("neg", np.array(-1.0, np.float32), np.float32)
    init("two_f", np.array(2.0, np.float32), np.float32)
    # 1-D index ramp on the 8x8 canvas (row & col)
    init("ar_row", np.arange(WORK, dtype=np.float32).reshape(1, 1, WORK, 1), np.float32)
    init("ar_col", np.arange(WORK, dtype=np.float32).reshape(1, 1, 1, WORK), np.float32)
    # final Equal channel comparator + sentinel
    init("chan_u8", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    init("sent_u8", np.array(99, np.uint8), np.uint8)  # outside-grid sentinel
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64),
         np.int64)

    # ---- per-channel pixel counts (cheap: [1,10,1,1] = 40 B) ----
    n("ReduceSum", ["input"], "cnt", axes=[2, 3], keepdims=1)   # [1,10,1,1] f32

    # mask out background (channel 0) and absent channels
    n("Greater", ["chidx", "half_f"], "is_colour")              # ch>=1
    n("Greater", ["cnt", "half_f"], "present")                  # cnt>=1
    n("And", ["is_colour", "present"], "valid")                 # bool [1,10,1,1]

    # c0 = rarer present colour: min count -> its count & channel index
    n("Where", ["valid", "cnt", "big"], "cnt_lo")               # absent -> +big
    n("ReduceMin", ["cnt_lo"], "c0_cnt", keepdims=1)            # [1,1,1,1] = t0^2
    # c1 = more-frequent present colour
    n("Where", ["valid", "cnt", "neg"], "cnt_hi")               # absent -> -1
    n("ReduceMax", ["cnt_hi"], "c1_cnt", keepdims=1)            # [1,1,1,1]

    # recover colour channel indices (argmin/argmax via count match)
    n("Equal", ["cnt", "c0_cnt"], "is_c0")                      # bool [1,10,1,1]
    n("And", ["is_c0", "valid"], "is_c0v")
    n("Where", ["is_c0v", "chidx", "neg"], "c0_idx_each")
    n("ReduceMax", ["c0_idx_each"], "c0_idx", keepdims=1)       # [1,1,1,1] f32
    n("Equal", ["cnt", "c1_cnt"], "is_c1")
    n("And", ["is_c1", "valid"], "is_c1v")
    n("Where", ["is_c1v", "chidx", "neg"], "c1_idx_each")
    n("ReduceMax", ["c1_idx_each"], "c1_idx", keepdims=1)       # [1,1,1,1] f32

    # ---- geometry scalars ----
    n("Add", ["c0_cnt", "c1_cnt"], "sq_cnt")                    # size^2
    n("Sqrt", ["sq_cnt"], "size_f")                             # size (3..6)
    n("Sqrt", ["c0_cnt"], "t0_f")                               # inner side (1 or 2)
    # t1 = (size - t0)/2
    n("Sub", ["size_f", "t0_f"], "two_t1")
    n("Div", ["two_t1", "two_f"], "t1_f")                       # inner_lo
    n("Add", ["t1_f", "t0_f"], "inner_hi")                      # inner_lo+t0

    # ---- separable masks on the 8x8 canvas ----
    # in-square: row<size & col<size
    n("Less", ["ar_row", "size_f"], "row_sq")                   # [1,1,8,1] bool
    n("Less", ["ar_col", "size_f"], "col_sq")                   # [1,1,1,8] bool
    n("And", ["row_sq", "col_sq"], "in_sq")                     # [1,1,8,8] bool
    # inner block: inner_lo<=r<inner_hi  (Less has no >=, use Not(Less(r,lo)))
    n("Less", ["ar_row", "t1_f"], "row_below")
    n("Not", ["row_below"], "row_ge_lo")
    n("Less", ["ar_row", "inner_hi"], "row_lt_hi")
    n("And", ["row_ge_lo", "row_lt_hi"], "row_in")             # [1,1,8,1]
    n("Less", ["ar_col", "t1_f"], "col_below")
    n("Not", ["col_below"], "col_ge_lo")
    n("Less", ["ar_col", "inner_hi"], "col_lt_hi")
    n("And", ["col_ge_lo", "col_lt_hi"], "col_in")             # [1,1,1,8]
    n("And", ["row_in", "col_in"], "inner")                     # [1,1,8,8] bool

    # ---- colour value per cell (float -> uint8), then sentinel outside ----
    # inside square: inner -> c1, else -> c0 (both are small 0..9 floats, exact).
    n("Where", ["inner", "c1_idx", "c0_idx"], "sq_val")         # [1,1,8,8] f32
    n("Cast", ["sq_val"], "sq_u8", to=TensorProto.UINT8)        # [1,1,8,8] uint8 (0..9)
    # outside the square -> uint8 sentinel (Equal then yields all-channels-off).
    n("Where", ["in_sq", "sq_u8", "sent_u8"], "L8")             # [1,1,8,8] uint8

    # ---- pad 8x8 -> 30x30 with sentinel, final Equal -> free BOOL output ----
    n("Pad", ["L8", "padpads", "sent_u8"], "L", mode="constant")  # [1,1,30,30] u8
    n("Equal", ["L", "chan_u8"], "output")                       # -> free bool output

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
