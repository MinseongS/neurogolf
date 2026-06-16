"""task358 (ARC-AGI e21d9049) — reconstruct a diagonally-striped CROSS.

Rule (from the ARC-GEN generator, verified fresh):
  The grid is H x W (W in 10..20, H = W or W+1).  A "cross" is drawn: every
  cell on row `row` OR column `col` is coloured  colors[(r + c) % n]  where
  n = len(colors) in {3,4}.  Along each arm the colour cycles with period n
  (a diagonal-stripe colouring restricted to the cross).  The INPUT shows only
  a contiguous n x n window of that cross around the intersection; the OUTPUT
  redraws the FULL cross.  An optional horizontal flip mirrors both.

Flip-agnostic reconstruction (verified 0 / 5000 fresh instances):
  The colour on each arm is PERIODIC with period n, so flip/offset/colour-list
  are never recovered.  From the input only:
    * in-grid mask = any channel hot (off-grid cells are all-channels-off, so
      grid extent is visible; off-grid output must be all-off).
    * colour-index plane G = sum_k k * input_k (1x1 Conv).
    * per-row / per-col coloured counts; the arms are the row & column whose
      count == n = max(rowcount, colcount).
    * the n coloured cells on an arm give one full period; periodic extension
          prof[i] = sum_j armcolour[j] * (i % n == j % n)
      (one shared (arange%n) equality matrix, applied by MatMul).
  Assemble L = where(rowarm, rowprof, where(colarm, colprof, bg)); force a
  sentinel off-grid; final Equal(L, 0..9) writes the BOOL output for free.

Memory: the only 30x30 intermediates are G (fp16), the in-grid mask, the
(arange%n) equality matrix (bool), the two arm-select products, and the uint8
label.  The periodic extension uses MatMul (no 30x30 product plane).
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

S = 30  # full canvas (input is always zero-padded to 30x30)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dt):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dt), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    H16 = TensorProto.FLOAT16
    U8 = TensorProto.UINT8
    B = TensorProto.BOOL

    # ---- coloured indicator plane nz = sum_{k>=1} input_k  (1x1 Conv) ----
    # weights [0,1,1,...,1]: one-hot bg(ch0)->0, any colour->1.  fp32 (input is fp32).
    init("indW", np.array([0] + [1] * 9, dtype=np.float32).reshape(1, 10, 1, 1),
         np.float32)
    n("Conv", ["input", "indW"], "nz")            # [1,1,30,30] fp32 {0,1} coloured

    # ---- in-grid mask (separable: grid is the top-left H x W rectangle) ----
    # rows/cols that contain ANY in-grid cell (off-grid is all-channels-off).
    init("half", np.array(0.5, np.float32), np.float32)
    n("ReduceMax", ["input"], "rowany", axes=[1, 3], keepdims=1)   # [1,1,30,1] fp32
    n("ReduceMax", ["input"], "colany", axes=[1, 2], keepdims=1)   # [1,1,1,30] fp32
    n("Greater", ["rowany", "half"], "rowinB")    # [1,1,30,1] bool
    n("Greater", ["colany", "half"], "colinB")    # [1,1,1,30] bool
    n("And", ["rowinB", "colinB"], "ingridB")     # [1,1,30,30] bool (separable)

    # ---- coloured-cell counts (per row / per col) ----
    init("zeroh", np.array(0.0, np.float16), np.float16)
    n("ReduceSum", ["nz"], "rowcnt", axes=[3], keepdims=1)   # [1,1,30,1] fp32
    n("ReduceSum", ["nz"], "colcnt", axes=[2], keepdims=1)   # [1,1,1,30] fp32

    n("ReduceMax", ["rowcnt"], "nrow", axes=[2, 3], keepdims=1)   # scalar
    n("ReduceMax", ["colcnt"], "ncol", axes=[2, 3], keepdims=1)
    n("Max", ["nrow", "ncol"], "nval")            # scalar fp32 = n (period)

    # arm lines: coloured count == n
    n("Equal", ["rowcnt", "nval"], "rowmaskB")    # [1,1,30,1] bool
    n("Equal", ["colcnt", "nval"], "colmaskB")    # [1,1,1,30] bool
    n("Cast", ["rowmaskB"], "rowmaskC", to=F)     # [1,1,30,1] fp32
    n("Cast", ["colmaskB"], "colmaskR", to=F)     # [1,1,1,30] fp32

    # ---- arm colour periods (1-D), from the reduced ONE-HOT (no colour plane) ----
    # Reduce the one-hot input onto the arm row / col, then weight-sum channels.
    init("to_rowvec", np.array([1, 1, 1, S], np.int64), np.int64)
    init("to_colvec", np.array([1, 1, S, 1], np.int64), np.int64)
    init("chW", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    n("Reshape", ["rowmaskC", "to_rowvec"], "rowmaskV")     # [1,1,1,30]
    # rowarm one-hot: MatMul(rowmaskV, input) contracts the row axis -> [1,10,1,30]
    n("MatMul", ["rowmaskV", "input"], "rowOH")             # [1,10,1,30] fp32 (1200B)
    n("Conv", ["rowOH", "chW"], "rowcolorsF")               # [1,1,1,30] fp32
    n("Cast", ["rowcolorsF"], "rowcolors", to=H16)          # [1,1,1,30] fp16
    # colarm one-hot: MatMul(input, colmaskV) contracts the col axis -> [1,10,30,1]
    n("Reshape", ["colmaskR", "to_colvec"], "colmaskV")     # [1,1,30,1]
    n("MatMul", ["input", "colmaskV"], "colOH")             # [1,10,30,1] fp32 (1200B)
    n("Conv", ["colOH", "chW"], "colcolorsF")               # [1,1,30,1] fp32
    n("Cast", ["colcolorsF"], "colcolors", to=H16)          # [1,1,30,1] fp16

    # ---- periodic extension by Gather (no 30x30 plane) ----
    # The n coloured cells on an arm are n CONSECUTIVE positions; for output index
    # i, the colour is that of position  first + ((i - first) mod n),  where
    # `first` = the smallest coloured position on the arm.  All indices are 1-D.
    NI = TensorProto.INT32
    init("arI", np.arange(S, dtype=np.int32), np.int32)            # [30] int32
    init("bigI", np.array(999, np.int32), np.int32)
    n("Cast", ["nval"], "nI", to=NI)                              # scalar int32 = n
    # squeeze arm-colour vectors to 1-D [30]
    init("flat", np.array([S], np.int64), np.int64)
    n("Reshape", ["rowcolors", "flat"], "rowcV")                  # [30] fp16
    n("Reshape", ["colcolors", "flat"], "colcV")                  # [30] fp16
    # first coloured index on each arm
    n("Greater", ["rowcV", "zeroh"], "rcNZ")                      # [30] bool
    n("Greater", ["colcV", "zeroh"], "ccNZ")
    n("Where", ["rcNZ", "arI", "bigI"], "rcIdxOrBig")             # [30] int32
    n("ReduceMin", ["rcIdxOrBig"], "firstC", axes=[0], keepdims=1)   # [1] int32
    n("Where", ["ccNZ", "arI", "bigI"], "ccIdxOrBig")
    n("ReduceMin", ["ccIdxOrBig"], "firstR", axes=[0], keepdims=1)
    # idx[i] = firstC + ((i - firstC) mod n)   (Mod fmod=0 -> non-negative for n>0)
    n("Sub", ["arI", "firstC"], "dC")                            # [30] int32
    n("Mod", ["dC", "nI"], "dCmod")                              # [30] int32 (0..n-1)
    n("Add", ["dCmod", "firstC"], "idxC")                        # [30] int32
    n("Sub", ["arI", "firstR"], "dR")
    n("Mod", ["dR", "nI"], "dRmod")
    n("Add", ["dRmod", "firstR"], "idxR")
    # gather the period colours
    n("Gather", ["rowcV", "idxC"], "rowprofF", axis=0)           # [30] fp16
    n("Gather", ["colcV", "idxR"], "colprofF", axis=0)           # [30] fp16
    n("Reshape", ["rowprofF", "to_rowvec"], "rowprof")          # [1,1,1,30]
    n("Reshape", ["colprofF", "to_colvec"], "colprof")          # [1,1,30,1]

    # ---- assemble label plane (uint8 throughout: colours 0..9 + sentinel) ----
    n("Cast", ["rowprof"], "rowprofU", to=U8)            # [1,1,1,30] uint8
    n("Cast", ["colprof"], "colprofU", to=U8)            # [1,1,30,1] uint8
    init("zerou", np.array(0, np.uint8), np.uint8)
    init("sentu", np.array(99, np.uint8), np.uint8)
    n("Where", ["colmaskB", "colprofU", "zerou"], "Lc")  # colline arm [1,1,30,30] uint8
    n("Where", ["rowmaskB", "rowprofU", "Lc"], "Lrc")    # rowline overrides
    n("Where", ["ingridB", "Lrc", "sentu"], "L")         # off-grid -> sentinel uint8

    # ---- final Equal into the FREE bool output ----
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")          # [1,10,30,30] BOOL (free)

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task358", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
