"""Task 198: lattice grid of line color L with gaps.

Rule (verified on all 266 examples):
- grids are square (n x n), single nonzero color L forming separator rows/cols
- gaps (0-cells on separator lines, inside grid) are filled with 4
- a cell block becomes 4 if a non-corner gap lies on its border, else 3
  (a gap on a horizontal sep row marks the blocks above+below in that column
  band; a gap on a vertical sep col marks blocks left+right in that row band;
  gaps at line intersections mark nothing)

Implementation: occupancy stats -> sep row/col masks -> band indices P/Q via
triangular MatMul -> band-equality matrices -> gap dilation + Peq@d@Heq spread
-> bool mask logic -> dynamic-weight 1x1 Conv writes channels {L,3,4} straight
into `output`.
"""

import numpy as np
import onnx
from onnx import TensorProto

from ..builders import _model


def build(task):
    nodes, inits = [], []

    def init(name, arr, dtype=np.float32):
        inits.append(onnx.numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(onnx.helper.make_node(op, inputs, [out], **attrs))
        return out

    # constants
    w_occ = np.ones((1, 10, 1, 1), np.float32)
    w_occ[0, 0] = 0.0
    init("w_occ", w_occ)
    tril = np.tril(np.ones((30, 30), np.float32))      # tril[i,r]=1 if r<=i
    init("triL", tril)                                  # P = triL @ seprowf
    init("triU", tril.T.copy())                         # Q = sepcolf @ triU
    init("arangeR", np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1))
    init("arangeC", np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30))
    mask0 = np.ones((1, 10, 1, 1), np.float32)
    mask0[0, 0] = 0.0
    init("mask0", mask0)
    e34 = np.zeros((10, 2, 1, 1), np.float32)
    e34[3, 0] = 1.0
    e34[4, 1] = 1.0
    init("e34", e34)
    init("half", np.array(0.5, np.float32))
    init("twoc", np.array(2.0, np.float32))
    init("shapeW", np.array([10, 1, 1, 1], np.int64), np.int64)

    # occupancy and grid size
    n("Conv", ["input", "w_occ"], "occ")                       # [1,1,30,30] f
    n("ReduceSum", ["occ"], "rowcnt", axes=[3], keepdims=1)    # [1,1,30,1]
    n("ReduceSum", ["occ"], "colcnt", axes=[2], keepdims=1)    # [1,1,1,30]
    n("ReduceMax", ["occ"], "rowhas", axes=[3], keepdims=1)
    n("ReduceMax", ["occ"], "colhas", axes=[2], keepdims=1)
    n("MaxPool", ["rowhas"], "rsuf", kernel_shape=[30, 1], pads=[0, 0, 29, 0])
    n("MaxPool", ["colhas"], "csuf", kernel_shape=[1, 30], pads=[0, 0, 0, 29])
    n("ReduceSum", ["rsuf"], "hsum", axes=[2, 3], keepdims=1)  # [1,1,1,1]
    n("ReduceSum", ["csuf"], "wsum", axes=[2, 3], keepdims=1)
    n("Max", ["hsum", "wsum"], "nsz")                          # grid size n

    # in-grid and separator masks
    n("Less", ["arangeR", "nsz"], "rowinb")                    # [1,1,30,1] b
    n("Less", ["arangeC", "nsz"], "colinb")                    # [1,1,1,30] b
    n("Mul", ["rowcnt", "twoc"], "rc2")
    n("Greater", ["rc2", "nsz"], "seprowb")
    n("Mul", ["colcnt", "twoc"], "cc2")
    n("Greater", ["cc2", "nsz"], "sepcolb")
    n("Cast", ["seprowb"], "seprowf", to=TensorProto.FLOAT)
    n("Cast", ["sepcolb"], "sepcolf", to=TensorProto.FLOAT)

    # band indices and equality matrices
    n("MatMul", ["triL", "seprowf"], "P")                      # [1,1,30,1]
    n("MatMul", ["sepcolf", "triU"], "Q")                      # [1,1,1,30]
    n("Cast", ["P"], "Pint", to=TensorProto.INT32)
    n("Transpose", ["Pint"], "Pt", perm=[0, 1, 3, 2])
    n("Equal", ["Pint", "Pt"], "Peqb")                         # [1,1,30,30] b
    n("Cast", ["Peqb"], "Peqf", to=TensorProto.FLOAT)
    n("Cast", ["Q"], "Qint", to=TensorProto.INT32)
    n("Transpose", ["Qint"], "Qt", perm=[0, 1, 3, 2])          # [1,1,30,1]
    n("Equal", ["Qt", "Qint"], "Heqb")                         # [1,1,30,30] b
    n("Cast", ["Heqb"], "Heqf", to=TensorProto.FLOAT)

    # gaps
    n("Greater", ["occ", "half"], "occb")
    n("And", ["rowinb", "colinb"], "ingridb")                  # [1,1,30,30] b
    n("Or", ["seprowb", "sepcolb"], "sep2b")
    n("Not", ["occb"], "nocc")
    n("And", ["nocc", "ingridb"], "g1")
    n("And", ["g1", "sep2b"], "gapb")
    n("Not", ["sepcolb"], "nsepc")
    n("And", ["seprowb", "nsepc"], "honly")
    n("And", ["gapb", "honly"], "gapHb")
    n("Cast", ["gapHb"], "gapHf", to=TensorProto.FLOAT)
    n("Not", ["seprowb"], "nsepr")
    n("And", ["sepcolb", "nsepr"], "vonly")
    n("And", ["gapb", "vonly"], "gapVb")
    n("Cast", ["gapVb"], "gapVf", to=TensorProto.FLOAT)

    # spread gaps into adjacent blocks
    n("MaxPool", ["gapHf"], "dV", kernel_shape=[3, 1], pads=[1, 0, 1, 0])
    n("MaxPool", ["gapVf"], "dH", kernel_shape=[1, 3], pads=[0, 1, 0, 1])
    n("Add", ["dV", "dH"], "dsum")
    n("MatMul", ["Peqf", "dsum"], "t1")
    n("MatMul", ["t1", "Heqf"], "AB")
    n("Greater", ["AB", "half"], "fourb")

    # fill masks
    n("Not", ["sep2b"], "nsep2")
    n("And", ["ingridb", "nsep2"], "intb")
    n("And", ["fourb", "intb"], "f4a")
    n("Or", ["f4a", "gapb"], "fill4b")
    n("Not", ["fourb"], "nfour")
    n("And", ["intb", "nfour"], "fill3b")

    # output assembly: channels [L, 3, 4] via dynamic-weight 1x1 conv
    n("Concat", ["occb", "fill3b", "fill4b"], "catb", axis=1)  # [1,3,30,30] b
    n("Cast", ["catb"], "G", to=TensorProto.FLOAT)
    n("ReduceSum", ["input"], "counts", axes=[2, 3], keepdims=1)
    n("Mul", ["counts", "mask0"], "cm")
    n("Greater", ["cm", "half"], "Lb")
    n("Cast", ["Lb"], "Lf", to=TensorProto.FLOAT)
    n("Reshape", ["Lf", "shapeW"], "Lr")                       # [10,1,1,1]
    n("Concat", ["Lr", "e34"], "Wd", axis=1)                   # [10,3,1,1]
    n("Conv", ["G", "Wd"], "output")

    return _model(nodes, inits)
