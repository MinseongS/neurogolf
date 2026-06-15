"""Task 204 (ARC 868de0fa): hollow blue squares; fill interior with 7 (odd
side) or 2 (even side).  Output is the SAME-size grid: blue walls stay blue,
the interior of each square is recoloured, everything else stays background 0.

Memory floor-break.  The old net materialised ~14 fp32 [1,1,30,30] planes
(c1/Hm/cu/pu/Pup/Pdn/ps + a 2-channel ch01 + a 4-channel cat) for the final
1x1 Conv -> 64800 bytes.  Instead we build a SINGLE uint8 label map
L[1,1,30,30] = output colour (0/1/2/7) per cell and emit
`output = Equal(L, arange[1,10,1,1])` (opset 11, BOOL output) -- so the
10-channel expansion lives only in the free output.

Detection is unchanged in spirit but kept tiny / fp16:
  Hm = horizontal-wall interior cell (a blue cell with a blue left & right
       neighbour) via Conv+Relu.
  enclosed = odd count of Hm strictly above in the column (ray-cast parity)
       via a strict-lower-triangular 0.5 MatMul then frac()==0.5.
  odd = interior height parity from nearest H-wall row above/below
       (index-weighted cummax MaxPool, frac()==0.5).
The interior cells are exactly the enclosed, non-blue cells; colour 7 if the
height is odd else 2.  Blue walls (input ch1) override to 1; background is 0.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F16 = TensorProto.FLOAT16


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    # ---- Hm = horizontal-wall interior cell (blue with blue L & R nbr) ----
    # 1x3 ones over channel 1, bias -2, Relu -> 1 where all three are blue.
    # Convs read the fp32 `input` directly (no 10-channel fp16 cast -> saves 18KB).
    W1 = np.zeros((1, 10, 1, 3), np.float32)
    W1[0, 1, 0, :] = 1.0
    init("W1", W1, np.float32)
    init("b1", np.array([-2.0], np.float32), np.float32)
    n("Conv", ["input", "W1", "b1"], "c1f", pads=[0, 1, 0, 1])  # [1,1,30,30] f32
    n("Relu", ["c1f"], "Hmf")
    n("Cast", ["Hmf"], "Hm", to=F16)                           # fp16 0/1

    # ---- enclosed: parity of Hm strictly above (ray-cast) ----
    Tl = np.tril(np.full((30, 30), 0.5, np.float16), -1)
    init("Tl", Tl, np.float16)
    n("MatMul", ["Tl", "Hm"], "cu")                            # 0.5*count above
    n("Floor", ["cu"], "fl")
    n("Sub", ["cu", "fl"], "pu")                               # 0.5 iff odd
    init("Q", np.array(0.25, np.float16), np.float16)
    n("Greater", ["pu", "Q"], "enc_b")                         # bool enclosed

    # ---- interior height parity from nearest H-wall above/below ----
    j = np.arange(30, dtype=np.float16)
    init("idx1", ((j + 1) / 2).reshape(30, 1), np.float16)
    init("idx2", ((30 - j) / 2).reshape(30, 1), np.float16)
    n("Mul", ["Hm", "idx1"], "v1")
    n("MaxPool", ["v1"], "Pup", kernel_shape=[30, 1], pads=[29, 0, 0, 0])
    n("Mul", ["Hm", "idx2"], "v2")
    n("MaxPool", ["v2"], "Pdn", kernel_shape=[30, 1], pads=[0, 0, 29, 0])
    n("Add", ["Pup", "Pdn"], "sf")
    n("Floor", ["sf"], "fl2")
    n("Sub", ["sf", "fl2"], "ps")                              # 0.5 iff odd ht
    n("Greater", ["ps", "Q"], "odd_b")                         # bool odd side

    # ---- blue mask (input channel 1) ----
    Wb = np.zeros((1, 10, 1, 1), np.float32)
    Wb[0, 1, 0, 0] = 1.0
    init("Wb", Wb, np.float32)
    n("Conv", ["input", "Wb"], "blue_f")                       # [1,1,30,30] f32
    init("Half", np.array(0.5, np.float32), np.float32)
    n("Greater", ["blue_f", "Half"], "blue_b")                 # bool blue

    # ---- uint8 label map L ----
    init("v0u", np.array(0, np.uint8), np.uint8)
    init("v1u", np.array(1, np.uint8), np.uint8)
    init("v2u", np.array(2, np.uint8), np.uint8)
    init("v7u", np.array(7, np.uint8), np.uint8)
    init("v10u", np.array(10, np.uint8), np.uint8)             # off-grid sentinel
    # in-grid mask: ReduceSum over channels == 1 inside the grid, 0 outside
    n("ReduceSum", ["input"], "occ_f", axes=[1], keepdims=1)   # [1,1,30,30] f32
    n("Greater", ["occ_f", "Half"], "grid_b")                 # bool inside grid
    # base label inside grid: off-grid sentinel 10, in-grid background 0
    n("Where", ["grid_b", "v0u", "v10u"], "Lg")              # 0 in-grid else 10
    # interior cell = enclosed AND not blue ; colour 7 if odd else 2
    n("Not", ["blue_b"], "nblue_b")
    n("And", ["enc_b", "nblue_b"], "int_b")                    # interior cell
    n("Where", ["odd_b", "v7u", "v2u"], "fill")               # 7 or 2
    n("Where", ["int_b", "fill", "Lg"], "Li")                # interior overrides bg
    n("Where", ["blue_b", "v1u", "Li"], "L")                 # blue overrides

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                        # -> BOOL output

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
