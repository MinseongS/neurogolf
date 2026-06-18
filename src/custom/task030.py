"""Task 030 (1caeab9d): vertically align same-shape colour clusters to colour-1's top row.

Rule (from ARC-GEN generator, verified fresh 200/200 & 500/500):
  The grid (height 5 or 10, width always 10, background colour 0) holds three
  IDENTICAL-shape clusters of colours 1, 2, 4 at distinct (megarow, megacol)
  positions.  In the OUTPUT every cluster is moved VERTICALLY so its top row aligns
  with colour-1's top row (megarows[0]); columns are unchanged; colour-1 never moves;
  the three colours occupy DISJOINT columns (verified).  Per colour k:
      out_k = in_k shifted by delta_k = top1 - top_k  (top_k = first/min row of k)
  delta_1 = 0 (colour-1 is the alignment target).

Encoding (Tier B, per-colour boolean shift MatMul folding colour into the matrix):
  - Slice colours 1, 2, 4 to three [1,1,10,10] fp32 planes (colours are
    non-contiguous so two/three small slices are the minimum-byte grab).
  - top_k = ArgMax over rows of per-row presence (ReduceMax over cols); all three
    colours are always present so ArgMax is exact -- no Where/ReduceMin sentinel.
  - For the two MOVING colours (2, 4): build a COLOUR-CARRYING shift matrix
    S[R,r] = colour * (r + delta_k == R) via Where(Equal(rowramp+delta, rowramp^T),
    colour, 0), cast fp16, and MatMul S @ in -> the shifted, already-coloured plane.
  - Colour-1 (delta 0) skips its matrix: its raw fp16 slice IS its coloured plane.
  - Disjoint columns => Sum the three coloured planes -> colour-index Lf (fp16) ->
    one uint8 Cast.
  - In-grid mask: height 5 or 10; a row is in-grid iff r<5 OR height==10, the latter
    detected from a 40B background row-9 strip.  Off-grid rows -> sentinel 99.
  - Pad to 30x30 (sentinel 99) and Equal(L_uint8, arange) -> free BOOL output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

WORK = 10  # active grid side


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- colour-channel slice constants (non-contiguous: 1, 2, 4) ----
    init("st1", np.array([1, 0, 0], np.int64), np.int64)
    init("en1", np.array([2, WORK, WORK], np.int64), np.int64)
    init("st2", np.array([2, 0, 0], np.int64), np.int64)
    init("en2", np.array([3, WORK, WORK], np.int64), np.int64)
    init("st4", np.array([4, 0, 0], np.int64), np.int64)
    init("en4", np.array([5, WORK, WORK], np.int64), np.int64)
    init("ax_c", np.array([1, 2, 3], np.int64), np.int64)

    # row ramps for the shift matrix S[R,r] = colour*(r + delta == R)
    init("OUTR", np.arange(WORK, dtype=np.int64).reshape(WORK, 1), np.int64)  # R
    init("SRCR", np.arange(WORK, dtype=np.int64).reshape(1, WORK), np.int64)  # r
    init("ZERO16", np.array(0.0, np.float16), np.float16)
    init("col2", np.array(2.0, np.float16), np.float16)
    init("col4", np.array(4.0, np.float16), np.float16)

    # in-grid (height) constants
    init("st0", np.array([0, 0, 9, 0], np.int64), np.int64)
    init("en0", np.array([1, 1, 10, WORK], np.int64), np.int64)
    init("ax0", np.array([0, 1, 2, 3], np.int64), np.int64)
    init("ROWIDX64", np.arange(WORK, dtype=np.int64).reshape(1, 1, WORK, 1),
         np.int64)
    init("FIVE64", np.array(5, np.int64), np.int64)
    init("halff", np.array(0.5, np.float32), np.float32)

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    init("SENT", np.array(99, np.uint8), np.uint8)
    init("pads30", np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64),
         np.int64)

    # ---- three colour planes (fp16) ----
    for cn, st, en in (("1", "st1", "en1"), ("2", "st2", "en2"),
                       ("4", "st4", "en4")):
        n("Slice", ["input", st, en, "ax_c"], f"s{cn}")        # [1,1,10,10] fp32
        n("Cast", [f"s{cn}"], f"p{cn}", to=TensorProto.FLOAT16)  # fp16

    # ---- top row per colour via ArgMax of per-row presence ----
    tops = {}
    for cn in ("1", "2", "4"):
        n("ReduceMax", [f"p{cn}"], f"rm{cn}", axes=[3], keepdims=1)   # [1,1,10,1]
        n("ArgMax", [f"rm{cn}"], f"top{cn}", axis=2, keepdims=1)      # [1,1,1,1] i64
        tops[cn] = f"top{cn}"

    # ---- shift the two MOVING colours (2, 4); colour-1 passes through raw ----
    coloured = ["p1"]  # colour-1: raw fp16 slice = its coloured plane (delta 0)
    for cn, colc in (("2", "col2"), ("4", "col4")):
        # delta = top1 - topk  (scalar int64)
        n("Sub", ["top1", f"top{cn}"], f"d{cn}")                     # [1,1,1,1]
        # S[R,r] = (r + delta == R) ; build via broadcast ramps -> [10,10]
        n("Add", ["SRCR", f"d{cn}"], f"srcd{cn}")                    # [1,10] + bcast
        n("Equal", [f"srcd{cn}", "OUTR"], f"Seq{cn}")                # [10,10] bool
        # fold colour: Where(Seq, colour, 0) -> fp16 colour-carrying matrix
        n("Where", [f"Seq{cn}", colc, "ZERO16"], f"S{cn}")           # [10,10] fp16
        n("MatMul", [f"S{cn}", f"p{cn}"], f"sh{cn}")                 # [1,1,10,10] fp16
        coloured.append(f"sh{cn}")

    # disjoint columns -> Sum the coloured planes into a colour-index plane
    n("Sum", coloured, "Lf")                                         # [1,1,10,10] fp16
    n("Cast", ["Lf"], "Lcol", to=TensorProto.UINT8)                 # uint8

    # ---- in-grid mask: row in-grid iff r<5 OR height==10 ----
    n("Slice", ["input", "st0", "en0", "ax0"], "ch0r9")             # [1,1,1,10] fp32
    n("ReduceMax", ["ch0r9"], "bg9", axes=[3], keepdims=1)          # [1,1,1,1]
    n("Greater", ["bg9", "halff"], "h10")                           # scalar bool
    n("Less", ["ROWIDX64", "FIVE64"], "rlt5")                       # [1,1,10,1] bool
    n("Or", ["rlt5", "h10"], "ingrid")                              # [1,1,10,1] bool

    n("Where", ["ingrid", "Lcol", "SENT"], "Lm")                    # [1,1,10,10] uint8
    n("Pad", ["Lm", "pads30", "SENT"], "Lpad", mode="constant")     # [1,1,30,30]
    n("Equal", ["Lpad", "chan"], "output")                          # [1,10,30,30] bool

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
