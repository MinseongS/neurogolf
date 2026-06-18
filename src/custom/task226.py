"""Task 226 (ARC-GEN 941d9a10): fill 3 diagonal blocks of a gray-line grid.

Rule (from ARC-GEN generator, verified fresh):
  The 10x10 input is a black canvas partitioned into a (R x C) array of blocks
  by fully-gray separator rows/cols (gray = colour 5).  R and C are each 3 or 5
  (i.e. 2 or 4 separators per axis).  The output copies the input and then fills
  three diagonal blocks with solid colour:
      block (0, 0)                 -> blue   (1)   top-left
      block (R//2, C//2)           -> red    (2)   centre
      block (R-1, C-1)             -> green  (3)   bottom-right
  Each filled block covers only the background interior of that block (the gray
  separator lines are untouched).  Background stays 0, gray lines stay 5.

Encoding (separable 1-D block masks -> single colour-index plane -> Equal, Tier B):
  Everything is computed on the active 10x10 region.
    gray10 = input[:, 5:6, 0:10, 0:10]                    [1,1,10,10] fp32
    sr[r]  = (sum_c gray10 == 10)   separator-row indicator   [1,1,10,1]
    sc[c]  = (sum_r gray10 == 10)   separator-col indicator   [1,1,1,10]
  Inclusive prefix counts via a lower-triangular MatMul:
    incR = Tincl @ sr                (separators at or before row r)  [1,1,10,1]
    excR = incR - sr                 (separators strictly above r = block index)
    sufR = nsepR - incR              (separators strictly after r)
  Block-membership 1-D masks (a non-separator row r):
    rowblk0   = (incR == 0)                       rows before the 1st separator
    rowblkmid = (excR == nsepR/2) AND not sr      the centre block's rows
    rowblklast= (sufR == 0)        AND not sr      rows after the last separator
  (nsepR is always even (2 or 4) so nsepR/2 is exact; block index of the red
  block = R//2 = (nsepR+1)//2 = nsepR/2 for nsepR in {2,4}.)  Same for cols.
  The three filled rectangles are separable AND-broadcasts:
    blue  = rowblk0   AND colblk0
    red   = rowblkmid AND colblkmid
    green = rowblklast AND colblklast      (all [1,1,10,10] bool, disjoint)
  All four regions (gray lines, blue, red, green) are pairwise disjoint, so the
  colour-index plane is built by a uint8 Where chain (order irrelevant), base =
  gray lines (5) painted with blue(1)/red(2)/green(3) blocks:
    L10 = Where(blue, 1, Where(red, 2, Where(green, 3, Where(gray, 5, 0))))
  Pad L10 to 30x30 with sentinel 10 (matches no colour channel -> off-grid 0)
  and the final op Equal(Lp, arange[0..9]) writes straight into the FREE BOOL
  output, so the 10-channel expansion costs no memory.

  Dominant intermediate: padded label plane Lp (uint8 [1,1,30,30] = 900 B),
  irreducible because it must be 30x30 to broadcast against the 10 colour
  channels in the final Equal.  gray10 (fp32 [1,1,10,10] = 400 B) is the only
  other non-tiny tensor; all 1-D masks are <=40 B.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

N = 10  # active grid is always 10x10


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # --- gray plane over the active 10x10 region ----------------------------
    init("g_s", np.array([0, 5, 0, 0], np.int64), np.int64)
    init("g_e", np.array([1, 6, N, N], np.int64), np.int64)
    init("g_ax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "g_s", "g_e", "g_ax"], "gray10")  # [1,1,10,10] fp32 0/1

    # --- separator-row / separator-col indicators --------------------------
    n("ReduceSum", ["gray10"], "rowsum", axes=[3], keepdims=1)  # [1,1,10,1]
    n("ReduceSum", ["gray10"], "colsum", axes=[2], keepdims=1)  # [1,1,1,10]
    init("ten", np.array(float(N), np.float32), np.float32)
    n("Equal", ["rowsum", "ten"], "sr_b")  # [1,1,10,1] bool
    n("Equal", ["colsum", "ten"], "sc_b")  # [1,1,1,10] bool
    n("Cast", ["sr_b"], "sr", to=TensorProto.FLOAT)
    n("Cast", ["sc_b"], "sc", to=TensorProto.FLOAT)

    # inclusive prefix counts via lower-triangular MatMuls
    tril = np.tril(np.ones((N, N), np.float32))            # row i: 1 for j<=i
    init("TinclR", tril.reshape(1, 1, N, N), np.float32)   # [1,1,10,10]
    # for cols: inc_c[j] = sum_{i<=j} sc[i]  -> Tc[i,j]=1 if i<=j (upper-tri)
    init("TinclC", np.triu(np.ones((N, N), np.float32)).reshape(1, 1, N, N),
         np.float32)
    n("MatMul", ["TinclR", "sr"], "incR")        # [1,1,10,1]
    n("MatMul", ["sc", "TinclC"], "incC")        # [1,1,1,10]

    n("ReduceSum", ["sr"], "nsepR", axes=[2], keepdims=1)  # [1,1,1,1]
    n("ReduceSum", ["sc"], "nsepC", axes=[3], keepdims=1)  # [1,1,1,1]

    n("Sub", ["incR", "sr"], "excR")             # separators strictly above
    n("Sub", ["incC", "sc"], "excC")
    n("Sub", ["nsepR", "incR"], "sufR")          # separators strictly after
    n("Sub", ["nsepC", "incC"], "sufC")

    init("zero", np.array(0.0, np.float32), np.float32)
    init("half", np.array(0.5, np.float32), np.float32)
    n("Mul", ["nsepR", "half"], "midR")          # nsepR/2 = red block index
    n("Mul", ["nsepC", "half"], "midC")

    # not-separator helpers
    n("Not", ["sr_b"], "nsr")
    n("Not", ["sc_b"], "nsc")

    # 1-D block-membership masks (bool)
    n("Equal", ["incR", "zero"], "r0")           # rows before first separator
    n("Equal", ["incC", "zero"], "c0")
    n("Equal", ["excR", "midR"], "rmid_raw")
    n("And", ["rmid_raw", "nsr"], "rmid")
    n("Equal", ["excC", "midC"], "cmid_raw")
    n("And", ["cmid_raw", "nsc"], "cmid")
    n("Equal", ["sufR", "zero"], "rlast_raw")
    n("And", ["rlast_raw", "nsr"], "rlast")
    n("Equal", ["sufC", "zero"], "clast_raw")
    n("And", ["clast_raw", "nsc"], "clast")

    # separable rectangle masks (broadcast [1,1,10,1] x [1,1,1,10] -> [1,1,10,10])
    n("And", ["r0", "c0"], "blue_b")
    n("And", ["rmid", "cmid"], "red_b")
    n("And", ["rlast", "clast"], "green_b")

    # colour-index plane via a uint8 Where chain (regions disjoint -> order
    # irrelevant).  Base = gray lines (5), painted with blue/red/green blocks.
    init("g_half", np.array(0.5, np.float32), np.float32)
    n("Greater", ["gray10", "g_half"], "gray_b")          # [1,1,10,10] bool
    init("u0", np.array(0, np.uint8).reshape(1, 1, 1, 1), np.uint8)
    init("u1", np.array(1, np.uint8).reshape(1, 1, 1, 1), np.uint8)
    init("u2", np.array(2, np.uint8).reshape(1, 1, 1, 1), np.uint8)
    init("u3", np.array(3, np.uint8).reshape(1, 1, 1, 1), np.uint8)
    init("u5", np.array(5, np.uint8).reshape(1, 1, 1, 1), np.uint8)
    n("Where", ["gray_b", "u5", "u0"], "Lg")              # 5 where gray, else 0
    n("Where", ["green_b", "u3", "Lg"], "Lgr")            # paint green (3)
    n("Where", ["red_b", "u2", "Lgr"], "Lrd")             # paint red (2)
    n("Where", ["blue_b", "u1", "Lrd"], "L10")            # paint blue (1) -> u8

    # pad to 30x30 with off-grid sentinel 10, then Equal into FREE bool output
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - N, 30 - N], np.int64), np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    n("Pad", ["L10", "pads", "padval"], "Lp", mode="constant")  # [1,1,30,30] u8

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["Lp", "chan"], "output")          # FREE BOOL [1,10,30,30]

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
