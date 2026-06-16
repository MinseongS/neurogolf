"""Task 321 (ARC-GEN cf98881b): priority overlay of three 4x4 colour panels.

Rule (from ARC-GEN generator, verified fresh 200/200):
  Input is a 14-col x 4-row grid (size=4 -> width 3*size+2 = 14, height size = 4).
  Three 4x4 panels are laid out left to right separated by single red(2) columns:
      panel 0 = cols 0..3   (pixels drawn in colour 4, idx 0)
      [red separator column at col 4]
      panel 1 = cols 5..8   (pixels drawn in colour 9, idx 1)
      [red separator column at col 9]
      panel 2 = cols 10..13 (pixels drawn in colour 1, idx 2)
  Each panel holds a sparse set of coloured pixels.  The output (4x4) overlays the
  three panels with a FIXED priority order panel0 > panel1 > panel2 (the generator
  writes idx=2 first, then idx=1, then idx=0, so the lowest idx wins):
      output[r][c] = panel0[r][c] (=4) if panel0 has a pixel there,
                     else panel1[r][c] (=9) if panel1 has a pixel there,
                     else panel2[r][c] (=1) if panel2 has a pixel there,
                     else background 0.
  Colours are always the fixed triple (4, 9, 1); the separator columns are red(2).

Encoding (channel-slice + final Equal, Tier B):
  The benchmark feeds a one-hot [1,10,30,30] tensor (channel=colour, axis2=row,
  axis3=col) and scores pred = out>0.  Because the three panel colours are FIXED
  (4, 9, 1), each panel's presence mask is just a single input channel sliced to
  that panel's 4x4 region -- no colour recovery / Conv needed at all:
      m0 = input[ch=4, rows 0..3, cols 0..3]    (panel 0 presence)
      m1 = input[ch=9, rows 0..3, cols 5..8]    (panel 1 presence)
      m2 = input[ch=1, rows 0..3, cols 10..13]  (panel 2 presence)
  Each mask is a tiny [1,1,4,4] tensor sliced directly from the FREE input -- the
  full [1,1,30,30] colour-index plane (3600 B) is never materialised.  Fold by the
  fixed priority panel0 > panel1 > panel2 into a uint8 label L on [1,1,4,4]:
      L = 4 where m0>0 else (9 where m1>0 else (1 where m2>0 else 0)).
  Pad L to 30x30 with sentinel 10 (matches no colour channel -> background outside
  the 4x4 output), then the final op Equal(L, arange[0..9]) writes straight into
  the free BOOL output, so the 10-channel expansion never costs memory.

  Dominant intermediate: Lp (uint8 [1,1,30,30] = 900 B) -- it must be 30x30 to
  broadcast against the 10 colour channels in the final Equal.  Everything else is
  tiny [1,1,4,4] panel tensors (<=64 B each).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

S = 4   # panel size (4x4)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # Slice each panel's presence mask straight from the FREE one-hot input:
    # axis 1 = colour channel, axis 2 = row, axis 3 = col.  Colours are fixed
    # (panel0=4, panel1=9, panel2=1), so a single-channel slice over the panel's
    # 4x4 region is exactly that panel's presence mask -> [1,1,4,4] f32.
    init("ax123", np.array([1, 2, 3], np.int64), np.int64)
    init("m0_st", np.array([4, 0, 0], np.int64), np.int64)
    init("m0_en", np.array([5, S, S], np.int64), np.int64)            # ch4, cols 0..3
    init("m1_st", np.array([9, 0, S + 1], np.int64), np.int64)
    init("m1_en", np.array([10, S, 2 * S + 1], np.int64), np.int64)   # ch9, cols 5..8
    init("m2_st", np.array([1, 0, 2 * S + 2], np.int64), np.int64)
    init("m2_en", np.array([2, S, 3 * S + 2], np.int64), np.int64)    # ch1, cols 10..13
    n("Slice", ["input", "m0_st", "m0_en", "ax123"], "m0")  # [1,1,4,4] f32
    n("Slice", ["input", "m1_st", "m1_en", "ax123"], "m1")
    n("Slice", ["input", "m2_st", "m2_en", "ax123"], "m2")

    # priority fold into a uint8 colour label on [1,1,4,4]:
    #   L = 4 where m0>0 else (9 where m1>0 else (1 where m2>0 else 0))
    init("half", np.array(0.5, np.float32), np.float32)
    n("Greater", ["m0", "half"], "p0pos")
    n("Greater", ["m1", "half"], "p1pos")
    n("Greater", ["m2", "half"], "p2pos")
    init("c4", np.array(4, np.uint8).reshape(1, 1, 1, 1), np.uint8)
    init("c9", np.array(9, np.uint8).reshape(1, 1, 1, 1), np.uint8)
    init("c1", np.array(1, np.uint8).reshape(1, 1, 1, 1), np.uint8)
    init("c0", np.array(0, np.uint8).reshape(1, 1, 1, 1), np.uint8)
    n("Where", ["p2pos", "c1", "c0"], "L2")      # panel2 (=1) else bg
    n("Where", ["p1pos", "c9", "L2"], "L12")     # panel1 (=9) over panel2
    n("Where", ["p0pos", "c4", "L12"], "L")      # panel0 (=4) over the rest

    # pad to 30x30 with sentinel 10 (matches no colour channel), then final Equal
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - S, 30 - S], np.int64),
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
