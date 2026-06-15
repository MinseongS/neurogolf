"""Task 012 (0962bcdd): plus -> X+plus stamp, two centers.

Rule (from ARC-GEN generator, size=12, then a gravity reflect/transpose applied
EQUALLY to input and output so the input->output map is gravity-invariant):

Two colours c0,c1 (distinct, 1..9).  Two centres are placed.  In the INPUT each
centre is a 5-cell plus: centre = c0, the 4 orthogonal neighbours = c1.  In the
OUTPUT each centre grows a 5x5 stamp:
  * c0 at the centre and the 8 diagonal cells (dist 1 and 2):
        (0,0),(+-1,+-1),(+-2,+-2)
  * c1 at the 8 orthogonal cells (dist 1 and 2):
        (+-1,0),(0,+-1),(+-2,0),(0,+-2)
The two stamps never overlap (centres are 6 rows apart, stamps reach +-2).

So output colour per cell is a deterministic LOCAL function of the input ->
Tier B label map.  Tier S is blocked: the output colours c0,c1 are random per
instance, so a fixed Conv cannot route them to the correct output channel.
Tier A blocked: the 5x5 X/plus stamp is not a row(x)col separable rectangle.

Build (label map + final Equal, 12x12 working canvas):
 1. A = colour-value image via 1x1 Conv(weights=arange) -> [1,1,30,30] f32,
    Slice to [1,1,12,12] (the whole active region fits the top-left corner:
    centres sit at rows 2 & 8 pre-gravity; gravity is a reflect/transpose of a
    12x12 grid so every used cell stays in 0..11).
 2. counts[k] = ReduceSum input over space.  c0colour = sum_k k*(count==2);
    c1colour = sum_k k*(count==8).  (each centre is 1 c0 cell -> count 2 total;
    each centre is 4 c1 arms -> count 8 total.)
 3. M = Equal(A12, c0colour) -> centre mask.
 4. Conv M with the c0 stamp kernel and the c1 stamp kernel; (>0) -> stamps.
 5. L = Where(c1stamp, c1colour_u8, Where(c0stamp, c0colour_u8, sentinel=10)).
 6. Pad L to 30x30 (sentinel), output = Equal(L, arange_u8[1,10,1,1]) -> BOOL.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

WORK = 12  # active region (12x12 grid) fits the top-left corner


def _c0_kernel():
    k = np.zeros((5, 5), np.float32)
    k[2, 2] = 1.0
    for dr, dc in [(1, 1), (-1, -1), (1, -1), (-1, 1)]:
        k[2 + dr, 2 + dc] = 1.0
        k[2 + 2 * dr, 2 + 2 * dc] = 1.0
    return k.reshape(1, 1, 5, 5)


def _c1_kernel():
    k = np.zeros((5, 5), np.float32)
    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        k[2 + dr, 2 + dc] = 1.0
        k[2 + 2 * dr, 2 + 2 * dc] = 1.0
    return k.reshape(1, 1, 5, 5)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- constants ----
    init("colw", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    init("kc0", _c0_kernel(), np.float32)
    init("kc1", _c1_kernel(), np.float32)
    init("karange", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1),
         np.float32)
    init("two", np.array(2.0, np.float32), np.float32)
    init("eight", np.array(8.0, np.float32), np.float32)
    init("one", np.array(1.0, np.float32), np.float32)
    init("zero", np.array(0.0, np.float32), np.float32)
    init("half", np.array(0.5, np.float32), np.float32)
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    init("sent", np.array(10, np.uint8), np.uint8)
    # slice A (30x30) -> 12x12 top-left
    init("sl_st", np.array([0, 0], np.int64), np.int64)
    init("sl_en", np.array([WORK, WORK], np.int64), np.int64)
    init("sl_ax", np.array([2, 3], np.int64), np.int64)
    # pad L 12x12 -> 30x30 with sentinel
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64),
         np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)

    # ---- colour-value image A, cropped to 12x12 ----
    n("Conv", ["input", "colw"], "A30")                  # [1,1,30,30] f32 colour
    n("Slice", ["A30", "sl_st", "sl_en", "sl_ax"], "A")  # [1,1,12,12] f32

    # ---- per-colour counts -> c0 (count 2) and c1 (count 8) scalar colours ----
    n("ReduceSum", ["input"], "cnt", axes=[2, 3], keepdims=1)   # [1,10,1,1] f32
    # is2 = (|cnt-2| < 0.5) ; is8 = (|cnt-8| < 0.5)
    n("Sub", ["cnt", "two"], "d2")
    n("Abs", ["d2"], "ad2")
    n("Less", ["ad2", "half"], "is2b")
    n("Cast", ["is2b"], "is2", to=TensorProto.FLOAT)
    n("Sub", ["cnt", "eight"], "d8")
    n("Abs", ["d8"], "ad8")
    n("Less", ["ad8", "half"], "is8b")
    n("Cast", ["is8b"], "is8", to=TensorProto.FLOAT)
    # colour = sum_k k * mask[k]
    n("Mul", ["is2", "karange"], "c0parts")
    n("ReduceSum", ["c0parts"], "c0f", axes=[1], keepdims=1)    # [1,1,1,1]
    n("Mul", ["is8", "karange"], "c1parts")
    n("ReduceSum", ["c1parts"], "c1f", axes=[1], keepdims=1)    # [1,1,1,1]
    n("Cast", ["c0f"], "c0u", to=TensorProto.UINT8)
    n("Cast", ["c1f"], "c1u", to=TensorProto.UINT8)

    # ---- centre mask M = (A == c0colour) ----
    n("Sub", ["A", "c0f"], "dA")                # broadcast scalar
    n("Abs", ["dA"], "adA")
    n("Less", ["adA", "half"], "Mb")            # [1,1,12,12] bool centre
    n("Cast", ["Mb"], "Mf", to=TensorProto.FLOAT)

    # ---- stamps via Conv ----
    n("Conv", ["Mf", "kc0"], "c0raw", pads=[2, 2, 2, 2])  # [1,1,12,12]
    n("Conv", ["Mf", "kc1"], "c1raw", pads=[2, 2, 2, 2])
    n("Greater", ["c0raw", "half"], "c0st")
    n("Greater", ["c1raw", "half"], "c1st")

    # ---- label map ----
    n("Where", ["c0st", "c0u", "sent"], "L0")   # uint8 [1,1,12,12]
    n("Where", ["c1st", "c1u", "L0"], "L12")    # c1 over c0 (disjoint anyway)
    n("Pad", ["L12", "padpads", "padval"], "L", mode="constant")  # [1,1,30,30]
    n("Equal", ["L", "chan"], "output")         # -> free BOOL output

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
