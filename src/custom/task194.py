"""Task 194 (7fe24cdd): C4 rotational symmetrization of a 3x3 grid -> 6x6.

Rule (from ARC-GEN generator):
  Input is a size x size grid (size = 3) with 5..9 coloured pixels.  Output is
  2*size x 2*size = 6x6.  Each input pixel (r,c,v) is stamped at its 4-cell C4
  rotation orbit about the 6x6 centre:
      out[r][c]            = v
      out[2s-c-1][r]       = v
      out[c][2s-r-1]       = v
      out[2s-r-1][2s-c-1]  = v
  The 4 orbit cells are always distinct, so it is a pure FIXED coordinate scatter:
  EACH of the 36 output cells reads exactly ONE input cell.  We invert the scatter
  to a per-output-cell SOURCE-index map src[R][C] = r*3+c (full coverage, no
  conflicts), making the whole output a single flat Gather of the 9 input cells.

Encoding (pure copy, uint8 working planes, uint8 free output):
  Slice input to the active 3x3 corner (the only fp32 plane, 360B), Cast->uint8
  (one-hot is exactly {0,1}, loss-free, 4x smaller), Reshape the 3x3 spatial block
  to a flat 9-vector, Gather 36 source indices (axis=2) -> [1,10,36], Reshape to
  [1,10,6,6], then Pad to 30x30 with zeros (the FREE uint8 output).  Outside the
  6x6 block all channels are zero (= background), matching the generator.

  This beats the public GridSample net (mem 1440) and the prior at-floor verdict:
  the C4 orbit is a SINGLE 2-D-indexed Gather (a [6,6] index tensor on a flat-9
  axis yields [1,10,6,6] in one op), not a GridSample sampled plane.  The
  per-output-cell single-source-cell structure means no value plane, no rotation
  sum, no fp32 6x6 sample -- the dominant intermediates are the 360B fp32 input
  slice (unshrinkable: input is fp32, all 10 colour channels possible) and the
  360B uint8 6x6 output one-hot.  mem 900, params 54 -> 18.14 pts (+0.47 over the
  prior 17.67 GridSample), isolated fresh 200/200.  The earlier verdict was wrong:
  it assumed GridSample was the only realisation; a flat Gather handles the
  non-separable rotation at the SAME 360B copy floor as the separable D2 mirror.

  Output is declared UINT8: the harness scores (out > 0) booleans, so a uint8
  {0,1} one-hot passes identically.  ORT supports uint8 Gather/Reshape/Pad under
  ORT_DISABLE_ALL.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

SIZE = 3
OUT = 2 * SIZE  # 6


def _src_map():
    s = SIZE
    src = -np.ones((2 * s, 2 * s), dtype=np.int64)
    for r in range(s):
        for c in range(s):
            fi = r * s + c
            for (R, C) in [(r, c), (2 * s - c - 1, r),
                           (c, 2 * s - r - 1), (2 * s - r - 1, 2 * s - c - 1)]:
                src[R][C] = fi
    assert (src >= 0).all(), "orbit map incomplete"
    return src  # [6,6] source indices into the flat 9 input cells


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # crop input to the active 3x3 corner (all 10 channels kept)
    init("crop_st", np.array([0, 0], np.int64), np.int64)
    init("crop_en", np.array([SIZE, SIZE], np.int64), np.int64)
    init("crop_ax", np.array([2, 3], np.int64), np.int64)

    # flatten spatial 3x3 -> 9, then ONE 2-D-indexed Gather emits 6x6 directly
    init("flat9", np.array([1, 10, SIZE * SIZE], np.int64), np.int64)
    init("srcidx", _src_map(), np.int64)                       # [6,6] -> 6x6 out

    # pad the 6x6 block out to 30x30 with zeros -> free uint8 output
    init("pad_pads", np.array([0, 0, 0, 0, 0, 0, 30 - OUT, 30 - OUT], np.int64),
         np.int64)
    init("pad_val", np.array(0, np.uint8), np.uint8)

    n("Slice", ["input", "crop_st", "crop_en", "crop_ax"], "g33")  # [1,10,3,3] f32
    n("Cast", ["g33"], "g33u", to=TensorProto.UINT8)               # [1,10,3,3] u8
    n("Reshape", ["g33u", "flat9"], "flat")                        # [1,10,9]   u8
    # Gather with a [6,6] index tensor along axis=2 yields [1,10,6,6] in one op
    n("Gather", ["flat", "srcidx"], "g66", axis=2)                 # [1,10,6,6] u8
    n("Pad", ["g66", "pad_pads", "pad_val"], "output", mode="constant")

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
