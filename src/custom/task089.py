"""task089 (ARC-AGI 3e980e27) — "complete every marker into its full sprite".

Rule (from the generator, grid is always 13x13):
  Two 3x3 sprites are drawn.  Sprite idx0 has one RED(2) marker cell + body cells of
  one colour; sprite idx1 has one GREEN(3) marker cell + body cells of another colour.
  Each sprite is stamped at several non-overlapping mega-positions (the bounding boxes
  are separated by >=1 empty row/col so sprites are never even diagonally touching).
  The FIRST occurrence of each sprite is drawn in FULL in the input; every later
  occurrence shows ONLY its marker pixel.  idx0's later copies are COLUMN-MIRRORED
  (c -> 2-c); idx1 is never mirrored.  OUTPUT = draw the full sprite at every marker.

Encoding (all work on the 13x13 active canvas; 30x30 is all-background):
  colf = Sum_k k*input_k                                  (1x1 Conv -> colour index)
  For each marker colour m in {RED(flip), GREEN(noflip)}:
    mmap = (colf==m)
    d     = dilate^4(mmap) gated by occ each step  -> each marker grows into its sprite;
            lone markers (isolated by the >=1 gap) stay single cells.
    body  = d AND NOT mmap                          -> body cells of the FULL sprite only
            (lone markers have no body next to them).
    full  = dilate^4(body) gated by occ             -> the full sprite incl. its marker;
            lone markers are gone.
    fmark = mmap AND full                            -> the single full marker cell
    (fy,fx) = Sum(fmark*rowramp), Sum(fmark*colramp) -> full-marker position (scalars)
    fsp   = colf*full                                -> clean full-sprite colour plane
    K[5,5] = 5x5 window of fsp centred at (fy,fx)    -> template anchored on its marker
    For RED also mirror K's columns.
    usemap = lone markers only (RED) / all markers (GREEN)
    stamp  = Conv(usemap, rot180(K), pad=2)          -> places the template at every marker
                                                        (stamps never overlap so the sum
                                                         per cell is a single tap = colour)
    L = where(stamp>0, stamp, L)                     -> overlay onto the colour-index plane
  Route the 10-channel one-hot into the FREE output: Pad L to 30x30 with sentinel 99,
  then Equal(L30, arange10).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64
U8 = TensorProto.UINT8

S = 13  # grid is always 13x13


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out] if isinstance(out, str) else out, **attrs))
        return out

    # ---- colour-index plane ----
    init("convw", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), F32)
    n("Conv", ["input", "convw"], "colf30")                       # [1,1,30,30] f32
    init("c_s", np.array([0, 0], np.int64), I64)
    init("c_e", np.array([S, S], np.int64), I64)
    init("c_ax", np.array([2, 3], np.int64), I64)
    n("Slice", ["colf30", "c_s", "c_e", "c_ax"], "colf")          # [1,1,13,13] f32

    init("ZERO32", np.array(0.0, np.float32), F32)
    n("Greater", ["colf", "ZERO32"], "occ_b")                     # bool
    n("Cast", ["occ_b"], "occ", to=F16)                           # [1,1,13,13] fp16

    n("Cast", ["colf"], "colf16", to=F16)                          # [1,1,13,13] fp16

    # ramps for marker position
    rowr = np.arange(S, dtype=np.float16).reshape(1, 1, S, 1)
    colr = np.arange(S, dtype=np.float16).reshape(1, 1, 1, S)
    init("rowr", rowr, F16)
    init("colr", colr, F16)

    # dilation kernel constants (MaxPool 3x3 pad1)
    def dilate(src, occ, tag, k=4):
        cur = src
        for i in range(k):
            o = f"dl_{tag}_{i}"
            nodes.append(helper.make_node(
                "MaxPool", [cur], [o + "_p"],
                kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]))
            n("Mul", [o + "_p", occ], o)
            cur = o
        return cur

    init("ONE16", np.array(1.0, np.float16), F16)
    init("ZERO16", np.array(0.0, np.float16), F16)

    def stamp_for(m, flip, tag):
        # mmap = (colf == m)
        init(f"mc_{tag}", np.array(float(m), np.float32), F32)
        n("Equal", ["colf", f"mc_{tag}"], f"mmap_b_{tag}")
        n("Cast", [f"mmap_b_{tag}"], f"mmap_{tag}", to=F16)        # [1,1,13,13] fp16

        d = dilate(f"mmap_{tag}", "occ", f"d1_{tag}", 4)
        n("Sub", ["ONE16", f"mmap_{tag}"], f"notm_{tag}")
        n("Mul", [d, f"notm_{tag}"], f"body_{tag}")                # full-sprite body only
        full = dilate(f"body_{tag}", "occ", f"d2_{tag}", 4)
        n("Mul", [f"mmap_{tag}", full], f"fmark_{tag}")            # single full marker

        # full-marker position (scalars)
        n("Mul", [f"fmark_{tag}", "rowr"], f"fyp_{tag}")
        n("ReduceSum", [f"fyp_{tag}"], f"fy_{tag}", axes=[2, 3], keepdims=0)  # [1,1] fp16
        n("Mul", [f"fmark_{tag}", "colr"], f"fxp_{tag}")
        n("ReduceSum", [f"fxp_{tag}"], f"fx_{tag}", axes=[2, 3], keepdims=0)

        # clean full-sprite colour plane, padded by 2 each side
        n("Mul", ["colf16", full], f"fsp_{tag}")                   # [1,1,13,13] fp16
        init(f"pad2_{tag}", np.array([0, 0, 2, 2, 0, 0, 2, 2], np.int64), I64)
        n("Pad", [f"fsp_{tag}", f"pad2_{tag}", "ZERO16"], f"fspp_{tag}", mode="constant")  # [1,1,17,17]

        # window rows [fy : fy+5] (Gather axis2), then cols [fx : fx+5] (Gather axis3)
        n("Cast", [f"fy_{tag}"], f"fyi_{tag}", to=I64)             # [1,1]
        n("Cast", [f"fx_{tag}"], f"fxi_{tag}", to=I64)
        # build index vectors fy+[0..4], fx+[0..4]
        init(f"ar5_{tag}", np.arange(5, dtype=np.int64), I64)      # [5]
        n("Reshape", [f"fyi_{tag}", f"sc1_{tag}"], f"fysc_{tag}")  # scalar [1]
        # we need scalar ints; reshape [1,1]->[1]
        # gather rows
        n("Add", [f"fysc_{tag}", f"ar5_{tag}"], f"ridx_{tag}")     # [5]
        n("Reshape", [f"fxi_{tag}", f"sc1_{tag}"], f"fxsc_{tag}")
        n("Add", [f"fxsc_{tag}", f"ar5_{tag}"], f"cidx_{tag}")     # [5]
        n("Gather", [f"fspp_{tag}", f"ridx_{tag}"], f"krow_{tag}", axis=2)  # [1,1,5,17]
        n("Gather", [f"krow_{tag}", f"cidx_{tag}"], f"K_{tag}", axis=3)     # [1,1,5,5]

        # flip columns for RED, then rot180 to form the conv weight
        if flip:
            # column reverse: Slice axis3 step -1
            init(f"fs_{tag}", np.array([4], np.int64), I64)
            init(f"fe_{tag}", np.array([-6], np.int64), I64)
            init(f"fa_{tag}", np.array([3], np.int64), I64)
            init(f"fst_{tag}", np.array([-1], np.int64), I64)
            n("Slice", [f"K_{tag}", f"fs_{tag}", f"fe_{tag}", f"fa_{tag}", f"fst_{tag}"], f"Kf_{tag}")
            Ksrc = f"Kf_{tag}"
        else:
            Ksrc = f"K_{tag}"
        # rot180: reverse both axis2 and axis3 (step -1)
        init(f"rs_{tag}", np.array([4, 4], np.int64), I64)
        init(f"re_{tag}", np.array([-6, -6], np.int64), I64)
        init(f"ra_{tag}", np.array([2, 3], np.int64), I64)
        init(f"rst_{tag}", np.array([-1, -1], np.int64), I64)
        n("Slice", [Ksrc, f"rs_{tag}", f"re_{tag}", f"ra_{tag}", f"rst_{tag}"], f"W_{tag}")  # [1,1,5,5]

        # usemap: RED -> lone markers (mmap - fmark); GREEN -> all markers
        if flip:
            n("Sub", [f"mmap_{tag}", f"fmark_{tag}"], f"use_{tag}")
            usemap = f"use_{tag}"
        else:
            usemap = f"mmap_{tag}"

        # stamp = Conv(usemap, W, pad=2)
        n("Conv", [usemap, f"W_{tag}"], f"stamp_{tag}", pads=[2, 2, 2, 2])  # [1,1,13,13] fp16
        return f"stamp_{tag}"

    init("sc1_R", np.array([1], np.int64), I64)
    init("sc1_G", np.array([1], np.int64), I64)

    stampR = stamp_for(2, True, "R")
    stampG = stamp_for(3, False, "G")

    # overlay onto colour-index plane: L = where(stamp>0, stamp, L)
    n("Greater", [stampR, "ZERO16"], "sR_b")
    n("Where", ["sR_b", stampR, "colf16"], "L1")
    n("Greater", [stampG, "ZERO16"], "sG_b")
    n("Where", ["sG_b", stampG, "L1"], "L")                       # [1,1,13,13] fp16

    n("Cast", ["L"], "Lu8", to=U8)
    init("pad30", np.array([0, 0, 0, 0, 0, 0, 30 - S, 30 - S], np.int64), I64)
    init("SENT", np.array(99, np.uint8), U8)
    n("Pad", ["Lu8", "pad30", "SENT"], "L30", mode="constant")    # [1,1,30,30] uint8
    init("arange_u8", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), U8)
    n("Equal", ["L30", "arange_u8"], "output")                     # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task089", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
