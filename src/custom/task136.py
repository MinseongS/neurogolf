"""task136 (ARC-AGI 5c0a986e) — two 2x2 boxes, each emits a 45-degree ray.

Rule (from the ARC-GEN generator, grid is always 10x10 top-left):
  Two solid 2x2 boxes are placed (non-overlapping, diagonals >=3 apart):
    * box value 1 at top-left (R0,C0)  (input channel 1)
    * box value 2 at top-left (R1,C1)  (input channel 2)
  The OUTPUT keeps both boxes and adds two 45-degree diagonal rays:
    * value 1 ray: from (R0,C0) up-left to the top-left edge:
        r-c == R0-C0  AND  r <= R0
    * value 2 ray: from (R1,C1) down-right to the bottom-right edge:
        r-c == R1-C1  AND  r >= R1
  Boxes/rays of the two colours never collide (diagonals >=3 apart).

Recovery from the INPUT only:
  * box1 = channel 1, box2 = channel 2 (sliced to 10x10).
  * (R0,C0) = (min row, min col) of box1; (R1,C1) likewise for box2 — scalars
    from 1-D occupancy profiles, no 2-D argmin plane.
  * diagonal predicate uses RmC = rowramp - colramp (one 10x10 fp16 plane).
  * label L (uint8): value1 mask -> 1, value2 mask -> 2 (disjoint).  Pad to
    30x30 with sentinel; Equal(L, arange) -> FREE bool output.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

W = 10  # active canvas side (grid is always 10x10 anchored top-left)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dt):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dt), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    H = TensorProto.FLOAT16
    U8 = TensorProto.UINT8
    B = TensorProto.BOOL

    init("half", np.array(0.5, np.float32), np.float32)
    init("bigH", np.array(99.0, np.float16), np.float16)
    # 1-D ramps (fp16, exact small ints)
    init("RIv", np.arange(W, dtype=np.float16).reshape(1, 1, W, 1), np.float16)
    init("CIv", np.arange(W, dtype=np.float16).reshape(1, 1, 1, W), np.float16)
    init("RIf", np.arange(W, dtype=np.float32).reshape(1, 1, W, 1), np.float32)
    init("CIf", np.arange(W, dtype=np.float32).reshape(1, 1, 1, W), np.float32)
    # final output machinery
    init("chan_u8", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    init("sent_u8", np.array(99, np.uint8), np.uint8)
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64), np.int64)
    init("axes4", np.array([0, 1, 2, 3], np.int64), np.int64)

    # ---- slice box1 (ch1) and box2 (ch2) to 10x10 (fp32) ----
    init("s1s", np.array([0, 1, 0, 0], np.int64), np.int64)
    init("s1e", np.array([1, 2, W, W], np.int64), np.int64)
    init("s2s", np.array([0, 2, 0, 0], np.int64), np.int64)
    init("s2e", np.array([1, 3, W, W], np.int64), np.int64)
    n("Slice", ["input", "s1s", "s1e", "axes4"], "B1")   # [1,1,10,10] fp32
    n("Slice", ["input", "s2s", "s2e", "axes4"], "B2")   # [1,1,10,10] fp32
    n("Greater", ["B1", "half"], "box1")   # bool
    n("Greater", ["B2", "half"], "box2")   # bool

    # ---- recover (R0,C0) and (R1,C1) as scalars from 1-D profiles ----
    # row profile = sum over cols; col profile = sum over rows.
    for tag, plane in (("1", "B1"), ("2", "B2")):
        n("ReduceSum", [plane], "rp" + tag, axes=[3], keepdims=1)  # [1,1,10,1]
        n("ReduceSum", [plane], "cp" + tag, axes=[2], keepdims=1)  # [1,1,1,10]
        # rows where present -> rowramp, else +big ; min -> first row
        n("Greater", ["rp" + tag, "half"], "rpm" + tag)            # bool
        n("Greater", ["cp" + tag, "half"], "cpm" + tag)
        n("Where", ["rpm" + tag, "RIf", "bigF"], "rmin_src" + tag)
        n("Where", ["cpm" + tag, "CIf", "bigF"], "cmin_src" + tag)
        n("ReduceMin", ["rmin_src" + tag], "R" + tag, axes=[2, 3], keepdims=1)
        n("ReduceMin", ["cmin_src" + tag], "C" + tag, axes=[2, 3], keepdims=1)
    init("bigF", np.array(99.0, np.float32), np.float32)

    # diagonal constants R-C (fp16 exact small int range)
    n("Cast", ["R1"], "R1h", to=H)
    n("Cast", ["C1"], "C1h", to=H)
    n("Cast", ["R2"], "R2h", to=H)
    n("Cast", ["C2"], "C2h", to=H)
    n("Sub", ["R1h", "C1h"], "d1")   # R0-C0
    n("Sub", ["R2h", "C2h"], "d2")   # R1-C1

    # ---- diagonal masks ----
    n("Sub", ["RIv", "CIv"], "RmC")  # [1,1,10,10] fp16
    # diag1: RmC==d1 AND row<=R0
    n("Equal", ["RmC", "d1"], "ond1")
    n("Sub", ["R1h", "RIv"], "ler1")    # R0 - row  (>=0 means row<=R0)
    n("Greater", ["ler1", "neghalfH"], "rowle1")  # row<=R0  (R0-row > -0.5)
    n("And", ["ond1", "rowle1"], "diag1")
    # diag2: RmC==d2 AND row>=R1
    n("Equal", ["RmC", "d2"], "ond2")
    n("Sub", ["RIv", "R2h"], "ger2")    # row - R1 (>=0 means row>=R1)
    n("Greater", ["ger2", "neghalfH"], "rowge2")
    n("And", ["ond2", "rowge2"], "diag2")
    init("neghalfH", np.array(-0.5, np.float16), np.float16)

    # ---- value masks = box OR diag ----
    n("Or", ["box1", "diag1"], "m1")
    n("Or", ["box2", "diag2"], "m2")

    # ---- label L (uint8): disjoint masks; priority m2 over m1 (no overlap) ----
    init("z_u8", np.array(0, np.uint8), np.uint8)
    init("one_u8", np.array(1, np.uint8), np.uint8)
    init("two_u8", np.array(2, np.uint8), np.uint8)
    n("Where", ["m1", "one_u8", "z_u8"], "lab1")
    n("Where", ["m2", "two_u8", "lab1"], "L10")

    # ---- pad 10x10 -> 30x30 with sentinel, route to FREE bool output ----
    n("Pad", ["L10", "padpads", "sent_u8"], "L", mode="constant")
    n("Equal", ["L", "chan_u8"], "output")

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task136", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
