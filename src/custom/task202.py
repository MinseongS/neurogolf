"""task202 (855e0971) candidate — profile-only rebuild, no per-cell code/black planes.

Drops code_f (3600B fp32), iszero_b (900B bool), black (900B u8) entirely:
* rowcolor/colcolor = per-row/col colour-weighted count ratio Div(sum k*cnt, sum cnt)
  — exact band colour in the used orientation (integer/int division, exact in fp32).
* orientation via Cauchy uniformity: sum_k cnt_k^2 == (sum_k cnt_k)^2 per row iff the
  row is single-coloured (horizontal bands); one 3-operand einsum with the free input.
* band colour at each black cell recovered WITHOUT a black plane:
  hbH[w] = sum_h rowcolor[h]*black[h,w] = S_h - sum_h rowcolor[h]*occ[h,w]
  (occ contracted against the FREE input inside one einsum).
* mark = outer Equal of sentinel-masked profiles (incumbent trick), output Where free.

Only counted 30x30 carrier left: `mark` (900B bool).
"""

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT


def build(task):
    inits = []
    nodes = []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(np.asarray(arr, dtype=dtype), name))
        return name

    def node(op, inputs, output, **attrs):
        nodes.append(helper.make_node(op, inputs, [output], **attrs))
        return output

    init("codes_y", np.arange(10, dtype=np.float32).reshape(10, 1), np.float32)
    init("sel_y", np.array([0] + [1] * 9, dtype=np.float32).reshape(10, 1), np.float32)
    init("one_f", np.array(1.0, dtype=np.float32), np.float32)
    init("half_f", np.array(0.5, dtype=np.float32), np.float32)
    init("c99", np.array(99.0, dtype=np.float32), np.float32)
    init("c100", np.array(100.0, dtype=np.float32), np.float32)
    init("onehot0", np.array([1] + [0] * 9, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)

    # per-row / per-col colour-weighted counts and non-black counts (free-input einsums)
    node("Einsum", ["input", "codes_y"], "num_r", equation="bchw,cy->bhy")   # [1,30,1]
    node("Einsum", ["input", "sel_y"], "den_r", equation="bchw,cy->bhy")     # [1,30,1]
    node("Max", ["den_r", "one_f"], "den_r1")
    node("Div", ["num_r", "den_r1"], "rowcolor")                              # [1,30,1]
    node("Einsum", ["input", "codes_y"], "num_c", equation="bchw,cy->byw")   # [1,1,30]
    node("Einsum", ["input", "sel_y"], "den_c", equation="bchw,cy->byw")     # [1,1,30]
    node("Max", ["den_c", "one_f"], "den_c1")
    node("Div", ["num_c", "den_c1"], "colcolor")                              # [1,1,30]

    # orientation: horizontal iff every row single-coloured (Cauchy equality)
    node("Einsum", ["input", "input", "sel_y"], "sq_r", equation="bchw,bchv,cy->bhy")  # [1,30,1]
    node("Mul", ["den_r", "den_r"], "den2")
    node("Sub", ["den2", "sq_r"], "cdiff")
    node("ReduceSum", ["cdiff"], "ctot", keepdims=1)                          # [1,1,1]
    node("Greater", ["ctot", "half_f"], "isvert")

    node("Greater", ["rowcolor", "half_f"], "rpos")                           # [1,30,1]
    node("Greater", ["colcolor", "half_f"], "cpos")                           # [1,1,30]

    # band colour of the black in each column / row (no black plane)
    node("ReduceSum", ["rowcolor"], "S_h", keepdims=1)                        # [1,1,1]
    node("Einsum", ["input", "rowcolor", "sel_y"], "E1", equation="bchw,bhy,cy->byw")  # [1,1,30]
    node("Sub", ["S_h", "E1"], "hbH")
    node("ReduceSum", ["colcolor"], "S_c", keepdims=1)                        # [1,1,1]
    node("Einsum", ["input", "colcolor", "sel_y"], "E2", equation="bchw,byw,cy->bhy")  # [1,30,1]
    node("Sub", ["S_c", "E2"], "hbV")

    node("Where", ["isvert", "hbV", "rowcolor"], "A")                         # [1,30,1]
    node("Where", ["rpos", "A", "c99"], "Am")
    node("Where", ["isvert", "colcolor", "hbH"], "B")                         # [1,1,30]
    node("Where", ["cpos", "B", "c100"], "Bm")
    node("Equal", ["Am", "Bm"], "mark")                                       # [1,30,30] bool

    node("Where", ["mark", "onehot0", "input"], "output")

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task202_profile_only", [x], [y], inits)
    return helper.make_model(
        graph,
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 17)],
    )
