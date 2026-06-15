"""Task 372 (ARC-GEN e98196ab): vertical fold of two 5-row bands across a gray line.

Rule (from ARC-GEN generator, verified fresh 200/200):
  Input is an 11-col x 11-row grid (width=11, height=5 -> 2*height+1 = 11 rows).
  Row 5 is an all-gray(5) separator.  The TOP band is rows 0..4, the BOTTOM band
  is rows 6..10.  A pixel with idx=0 sits in the top band at row r; a pixel with
  idx=1 sits in the bottom band at row r+height+1 = r+6.  The two colours used are
  always non-gray and the output (11 cols x 5 rows) folds the bands onto each
  other:
      output[r][c] = top[r][c] colour  if the top band has a pixel there
                     else bottom[r+6][c] colour
  i.e. the output is the union of the top band and the bottom band.

Encoding (label-map + final Equal, Tier B, mem ~5160 -> 16.44 pts):
  The benchmark feeds a one-hot [1,10,30,30] tensor (channel=colour, axis2=row,
  axis3=col) and scores pred = out>0.  A 1x1 Conv with weights k=0..9 collapses
  the one-hot stack to a SINGLE colour-index map idxmap[1,1,30,30] (= the colour
  index 0..9 at each cell, 0 at background) -- this avoids ever materialising a
  10-channel band slice.  Slice the top band rows 0..4 and the bottom band rows
  6..10 of idxmap (both [1,1,5,11]), then fold: L = top_idx where the top band is
  non-zero else bottom_idx.  Cast L to uint8, Pad to 30x30 with sentinel 10
  (>=10 matches no colour channel -> all background outside the grid), and the
  final op Equal(L, arange[0..9]) writes straight into the free BOOL output, so
  the 10-channel expansion never costs memory.

  Dominant intermediates: idxmap (f32 [1,1,30,30] = 3600 B) and the padded label
  map Lp (uint8 [1,1,30,30] = 900 B); everything else is <=220 B small-band
  tensors.  Both are irreducible: idxmap is the smallest single-plane colour
  recovery from the one-hot input, and Lp must be 30x30 to broadcast against the
  10 colour channels in the final Equal.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

H = 5    # band height (rows per band)
W = 11   # grid width (cols 0..10)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # 1x1 Conv: idxmap = sum_k k * input_k = colour index (0..9), 0 at background.
    # weight [out=1, in=10, 1, 1] = [0,1,2,...,9].  Collapses the one-hot stack to
    # a single colour-index plane without any 10-channel intermediate.
    init("w", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    n("Conv", ["input", "w"], "idxmap")  # [1,1,30,30] f32, values 0..9

    # slice the two bands (rows only; cols 0..10) -> [1,1,5,11] each
    init("rc_ax", np.array([2, 3], np.int64), np.int64)
    init("T_st", np.array([0, 0], np.int64), np.int64)
    init("T_en", np.array([H, W], np.int64), np.int64)
    init("B_st", np.array([H + 1, 0], np.int64), np.int64)         # bottom row start = 6
    init("B_en", np.array([2 * H + 1, W], np.int64), np.int64)     # bottom row end = 11 (excl)
    n("Slice", ["idxmap", "T_st", "T_en", "rc_ax"], "topi")  # [1,1,5,11] f32
    n("Slice", ["idxmap", "B_st", "B_en", "rc_ax"], "boti")  # [1,1,5,11] f32

    # fold: L = top colour if the top band has a pixel here, else bottom colour
    init("half", np.array(0.5, np.float32), np.float32)
    n("Greater", ["topi", "half"], "tpos")                   # bool: top present
    n("Cast", ["topi"], "topu", to=TensorProto.UINT8)        # [1,1,5,11] uint8
    n("Cast", ["boti"], "botu", to=TensorProto.UINT8)        # [1,1,5,11] uint8
    n("Where", ["tpos", "topu", "botu"], "L")                # [1,1,5,11] uint8 label

    # pad to 30x30 with sentinel 10 (matches no colour channel), then final Equal
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - H, 30 - W], np.int64),
         np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    n("Pad", ["L", "padpads", "padval"], "Lp", mode="constant")  # [1,1,30,30] uint8
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["Lp", "chan"], "output")  # -> free BOOL [1,10,30,30]

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
