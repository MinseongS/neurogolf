"""Task 265 (ARC-AGI a8d7556c) — paint qualifying 2x2 black holes red.

Rule (faithful to the ARC-GEN generator, size=18 grid of gray(5)/black(0)).
The generator runs two stripe passes, mutating `output` between them, so the
result is order-dependent (stripe order [0,1]):

  Pass 0 (downstripe), on the ORIGINAL grid:
    a 2x2 all-black hole (r,c) is painted red(2) unless one of its horizontal
    side-columns is also fully black:
        fail0 = ( col c-1 both rows black )  OR  ( col c+2 both rows black )
    (the c==0 / c==W-2 boundary "skip" is the same as those shifted terms
     simply not existing -> treated as pass.)

  Pass 1 (sidestripe), on the grid AFTER pass-0 reds are filled in:
    a 2x2 hole counts as empty only if all four cells are STILL black (none
    was painted red in pass 0); it is painted unless a vertical side-row is
    fully "black-still":
        fail1 = ( row r-1 both cols black-still ) OR ( row r+2 both cols black-still )

  A cell is red iff it lies in any painted hole (pass 0 or pass 1).

Everything is local, so it is a short cascade of convolutions on single-channel
planes over the 18x18 active corner, ending in a uint8 label map L (0/5 base,
2 red, 10 off-grid) and a final Equal(L, arange) into the free BOOL output.
The only [1,10,30,30] tensor is the free `output`.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

W = 18  # active grid side (grid sits in the top-left 18x18 corner)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT16

    init("half", np.array(0.5, np.float16), np.float16)
    init("oneh", np.array(1.5, np.float16), np.float16)
    init("three_f", np.array(3.5, np.float16), np.float16)
    init("k22", np.ones((1, 1, 2, 2), np.float16), np.float16)
    init("k21", np.ones((1, 1, 2, 1), np.float16), np.float16)
    init("k12", np.ones((1, 1, 1, 2), np.float16), np.float16)

    # ---- in-grid black plane (channel 0) of the 18x18 corner ----
    # The grid is always exactly 18x18, so the whole 18x18 corner is in-grid;
    # cells outside it become the sentinel 10 via the final Pad.  channel 0 == 1
    # only for in-grid black cells (off-grid is all-zero, but unused here).
    init("s_st", np.array([0, 0, 0], np.int64), np.int64)
    init("s_en", np.array([1, W, W], np.int64), np.int64)
    init("s_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "s_st", "s_en", "s_ax"], "blk32")  # [1,1,18,18] f32 0/1
    n("Cast", ["blk32"], "blkf", to=F)                      # fp16 0/1
    n("Greater", ["blkf", "half"], "blk")                   # bool black mask

    # =====================================================================
    # PASS 0 (downstripe) on the original grid
    # =====================================================================
    # hole e0[r,c] = 2x2 all black (sum of 2x2 ones over blk == 4), pad br
    n("Conv", ["blkf", "k22"], "e0_s", pads=[0, 0, 1, 1])   # [1,1,18,18]
    n("Greater", ["e0_s", "three_f"], "e0")

    # cpb[r,c] = blk[r,c] & blk[r+1,c]  (vertical 2x1 ones, pad bottom)
    n("Conv", ["blkf", "k21"], "cpb_s", pads=[0, 0, 1, 0])
    n("Greater", ["cpb_s", "oneh"], "cpb")                  # ==2
    n("Cast", ["cpb"], "cpbf", to=F)

    # fail0[r,c] = cpb[r,c-1] OR cpb[r,c+2]  (width-4 kernel [1,0,0,1])
    init("kds", np.array([1., 0., 0., 1.], np.float16).reshape(1, 1, 1, 4),
         np.float16)
    n("Conv", ["cpbf", "kds"], "f0_s", pads=[0, 1, 0, 2])   # out[c]=cpb[c-1]+cpb[c+2]
    n("Greater", ["f0_s", "half"], "fail0")
    n("Not", ["fail0"], "pass0")
    n("And", ["e0", "pass0"], "h0")                         # painted holes (pass 0)

    # expand holes -> red0 cells: cell red0[r,c] iff any of the 4 holes
    # (r-1,c-1),(r-1,c),(r,c-1),(r,c) is in h0  -> 2x2 ones conv, pad top/left.
    n("Cast", ["h0"], "h0f", to=F)
    n("Conv", ["h0f", "k22"], "red0_s", pads=[1, 1, 0, 0])
    n("Greater", ["red0_s", "half"], "red0")               # bool red0 cells

    # =====================================================================
    # PASS 1 (sidestripe) on the pass-0-mutated grid
    # =====================================================================
    # black-still = black AND NOT red0  (cells still 0 after pass 0)
    n("Not", ["red0"], "notred0")
    n("And", ["blk", "notred0"], "blk1")                   # bool
    n("Cast", ["blk1"], "blk1f", to=F)

    # e1[r,c] = 2x2 all black-still
    n("Conv", ["blk1f", "k22"], "e1_s", pads=[0, 0, 1, 1])
    n("Greater", ["e1_s", "three_f"], "e1")

    # rpb1[r,c] = blk1[r,c] & blk1[r,c+1]  (horizontal 1x2 ones, pad right)
    n("Conv", ["blk1f", "k12"], "rpb1_s", pads=[0, 0, 0, 1])
    n("Greater", ["rpb1_s", "oneh"], "rpb1")               # ==2
    n("Cast", ["rpb1"], "rpb1f", to=F)

    # fail1[r,c] = rpb1[r-1,c] OR rpb1[r+2,c]  (height-4 kernel [1,0,0,1])
    init("kss", np.array([1., 0., 0., 1.], np.float16).reshape(1, 1, 4, 1),
         np.float16)
    n("Conv", ["rpb1f", "kss"], "f1_s", pads=[1, 0, 2, 0])
    n("Greater", ["f1_s", "half"], "fail1")
    n("Not", ["fail1"], "pass1")
    n("And", ["e1", "pass1"], "h1")

    n("Cast", ["h1"], "h1f", to=F)
    n("Conv", ["h1f", "k22"], "red1_s", pads=[1, 1, 0, 0])
    n("Greater", ["red1_s", "half"], "red1")

    # ---- final red cells = red0 OR red1 ----
    n("Or", ["red0", "red1"], "red")

    # ---- label map L (uint8): off-grid 10, red 2, gray 5, black 0 ----
    init("v0", np.array(0, np.uint8), np.uint8)
    init("v2", np.array(2, np.uint8), np.uint8)
    init("v5", np.array(5, np.uint8), np.uint8)
    n("Where", ["blk", "v0", "v5"], "base")                # original colour (0/5)
    n("Where", ["red", "v2", "base"], "L18")               # red overrides

    # ---- pad back to 30x30 (sentinel 10), final Equal -> BOOL output ----
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64),
         np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    n("Pad", ["L18", "padpads", "padval"], "L", mode="constant")
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task265", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
