"""Task 55 (ARC-GEN 272f95fa): fill a cyan 3x3 grid's plus-shaped cells.

Rule (from generator, verified fresh 200/200):
  The grid is partitioned into a 3x3 arrangement of variable-size blocks by two
  full horizontal cyan(8) lines and two full vertical cyan(8) lines (always all
  four lines present). The input already contains the cyan cross. The output
  keeps the input cyan lines and fills 5 of the 9 blocks (a plus shape) with
  FIXED colours by (rowband, colband):
      (0,1)=red 2   (1,0)=yellow 4   (1,1)=magenta 6   (1,2)=green 3   (2,1)=blue 1
  The four corner blocks and everything off-grid stay background 0.

Encoding (separable band partition + double-MatMul LUT routed into FREE BOOL output):
  Block boundaries are data-dependent (block sizes random 1..10), so the rowband
  and colband indices are recovered per-row/per-col as EXCLUSIVE CumSums of the
  cyan-line indicator. col 0 is never a vertical line and row 0 never a
  horizontal line, so the cyan channel along the first column gives the
  horizontal-line indicator (hline[1,1,30,1]) and along the first row the
  vertical-line indicator (vline[1,1,1,30]); their exclusive cumsums are the band
  indices {0,1,2,...}. The (rowband,colband)->colour LUT is NOT rank-1, so it is
  applied as the double-MatMul idiom Ronehot[1,1,30,3] @ LUT[3,3] @ Conehot[1,1,3,30]
  -> a single [1,1,30,30] band-colour plane. Two overlays then fix the special
  regions: where a cell is on a cyan line (hline_r OR vline_c) force colour 8;
  where a cell is off-grid (NOT (rowin AND colin), from separable channel-sum
  occupancy profiles) force sentinel 10 (matches no colour channel). The final
  Equal(L, arange[0..9]) writes straight into the FREE BOOL output so the
  10-channel expansion costs nothing.

  Everything past the one fp16 MatMul output runs in fp16 (Where/overlays), so
  the three full-canvas value planes count at half. Band cumsums and the
  occupancy profiles stay as ~120 B row/col vectors (no full occupancy plane).

  Dominant intermediate: the three [1,1,30,30] fp16 value planes (Lband, Lg, L
  = 1800 B each). Irreducible because the LUT is rank>1 (a sum of 5 rank-1
  blocks, needing the matmul) and the line/offgrid overrides are 2-D OR/AND of
  row & col vectors, so each override genuinely materialises a 30x30 selection.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

# LUT[rowband][colband] -> colour index. bands 0,1,2.
LUT = np.array([[0, 2, 0],
                [4, 6, 3],
                [0, 1, 0]], dtype=np.float32)
F16 = TensorProto.FLOAT16


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    init("ax", np.array([0, 1, 2, 3], np.int64), np.int64)
    # cyan(8) along first column -> per-row horizontal-line indicator
    init("hs", np.array([0, 8, 0, 0], np.int64), np.int64)
    init("he", np.array([1, 9, 30, 1], np.int64), np.int64)
    n("Slice", ["input", "hs", "he", "ax"], "hline")  # [1,1,30,1] f32
    # cyan(8) along first row -> per-col vertical-line indicator
    init("ve", np.array([1, 9, 1, 30], np.int64), np.int64)
    n("Slice", ["input", "hs", "ve", "ax"], "vline")  # [1,1,1,30] f32

    # band indices = exclusive cumsum of the line indicators (fp32, int-exact)
    init("ax2", np.array(2, np.int64), np.int64)
    init("ax3", np.array(3, np.int64), np.int64)
    n("CumSum", ["hline", "ax2"], "rowband", exclusive=1)  # [1,1,30,1]
    n("CumSum", ["vline", "ax3"], "colband", exclusive=1)  # [1,1,1,30]

    # band one-hots (fp16)
    init("kr", np.arange(3, dtype=np.float32).reshape(1, 1, 1, 3), np.float32)
    n("Equal", ["rowband", "kr"], "Reqb")
    n("Cast", ["Reqb"], "Ronehot", to=F16)  # [1,1,30,3]
    init("kc", np.arange(3, dtype=np.float32).reshape(1, 1, 3, 1), np.float32)
    n("Equal", ["colband", "kc"], "Ceqb")
    n("Cast", ["Ceqb"], "Conehot", to=F16)  # [1,1,3,30]

    init("lut", LUT.reshape(1, 1, 3, 3), np.float16)
    n("MatMul", ["Ronehot", "lut"], "RL")    # [1,1,30,3] f16
    n("MatMul", ["RL", "Conehot"], "Lband")  # [1,1,30,30] f16 band colour

    # line overlay -> 8
    init("half", np.array(0.5, np.float32), np.float32)
    n("Greater", ["hline", "half"], "hb")  # [1,1,30,1] bool
    n("Greater", ["vline", "half"], "vb")  # [1,1,1,30] bool
    n("Or", ["hb", "vb"], "isline")        # [1,1,30,30] bool
    init("c8", np.array(8, np.float16), np.float16)
    n("Where", ["isline", "c8", "Lband"], "Lg")  # [1,1,30,30] f16

    # in-grid gate (separable occupancy profiles) -> off-grid sentinel 10
    n("ReduceSum", ["input"], "rowsum", axes=[1, 3], keepdims=1)  # [1,1,30,1]
    n("ReduceSum", ["input"], "colsum", axes=[1, 2], keepdims=1)  # [1,1,1,30]
    n("Greater", ["rowsum", "half"], "rin")  # bool
    n("Greater", ["colsum", "half"], "cin")  # bool
    n("And", ["rin", "cin"], "ingrid")       # [1,1,30,30] bool
    init("c10", np.array(10, np.float16), np.float16)
    n("Where", ["ingrid", "Lg", "c10"], "L")  # [1,1,30,30] f16

    init("chan", np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1), np.float16)
    n("Equal", ["L", "chan"], "output")  # FREE bool [1,10,30,30]

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "g55", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
