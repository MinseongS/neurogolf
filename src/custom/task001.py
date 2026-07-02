"""Task 001 — fixed-corner 3x3 self-tiling (ARC fractal) as one factorized Einsum.

Kronecker self-product of the top-left 3x3 sprite, emitted DIRECTLY to the free
`output` tensor (zero counted intermediates). Input is monochrome (exactly one
non-background colour), so the foreground rule is the diagonal x_c*y_c.

Param breakdown (cost 240, mem 0):
  - `macro` [3,30] + `micro` [3,30] = 180 coordinate selectors. The 30-width is
    forced because the Einsum emits the [.,.,30,30] output directly (mem 0).
    `micro` doubles as the input 3x3 selector (input one-hot is zero beyond the
    corner, so micro[i,s]=delta(i,s) for s<3) and is reused 6x.
  - channel factor = `u` [3,10] + `ab` [3,10] = 60. The output threshold is
    `value > 0`, so the channel separator only needs SIGN-correctness over the
    reachable monochrome channel pairs (0,0),(0,k),(k,0),(k,k), k=1..9. A
    SYMMETRIC rank-3 sign factor (`ab` reused for both the block and within
    inputs, like `micro`) is feasible over that restricted domain:
        score[c] = sum_k u[k,c] * ab[k,cb] * ab[k,cw]
    with float32 worst-case sign margin ~2.0. This replaces the previous
    asymmetric rank-3 `u/a/b` factor (90 -> 60 params; cost 270 -> 240).
    rank<3 (sym or asym) is numerically infeasible over this domain.
"""

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ..harness import IR_VERSION


def build(task):
    macro = np.zeros((3, 30), dtype=np.float32)
    micro = np.zeros((3, 30), dtype=np.float32)
    for out in range(9):
        macro[out // 3, out] = 1.0
        micro[out % 3, out] = 1.0

    # Symmetric rank-3 sign factor (ab reused for block + within inputs).
    u = np.array(
        [
            [4.713938236236572, -5.281449317932129, -12.889695167541504,
             -17.848506927490234, -0.7993064522743225, -2.604400634765625,
             -2.1753878593444824, -0.3526121973991394, -4.357834815979004,
             -12.100116729736328],
            [-7.518928527832031, 4.174498081207275, -4.446011066436768,
             3.0661802291870117, 1.813381314277649, -1.9427882432937622,
             1.1782419681549072, -9.045491218566895, 5.854343414306641,
             -10.67162799835205],
            [-13.805724143981934, -5.821502685546875, 4.347726821899414,
             0.09081657975912094, -7.182774066925049, 1.4969850778579712,
             -0.7884160280227661, 1.6365774869918823, -13.454136848449707,
             7.511240005493164],
        ],
        dtype=np.float32,
    )
    ab = np.array(
        [
            [-20.628700256347656, -4.419224262237549, -4.865675449371338,
             -6.44936466217041, -7.27122688293457, -2.893186330795288,
             -1.1490651369094849, -6.739649772644043, -8.790376663208008,
             -9.811230659484863],
            [0.25487709045410156, 6.542076110839844, -11.40894603729248,
             -15.397046089172363, -5.573111534118652, 5.166507720947266,
             2.627955675125122, 0.12062016874551773, -12.533860206604004,
             9.03586483001709],
            [-0.1918373554944992, 3.553530693054199, 14.297358512878418,
             15.88792610168457, -1.1588215827941895, -7.148657321929932,
             2.028015613555908, -8.26154613494873, 6.555273532867432,
             -16.472551345825195],
        ],
        dtype=np.float32,
    )

    # micro doubles as the input 3x3 selector (in_sel); ab doubles as both the
    # block and within channel factors.
    nodes = [
        helper.make_node(
            "Einsum",
            [
                "input",
                "input",
                "micro",  # ay  — selects input1 row
                "micro",  # bx  — selects input1 col
                "micro",  # ez  — selects input2 row
                "micro",  # fw  — selects input2 col
                "macro",  # ar  — output row block
                "macro",  # bd  — output col block
                "micro",  # er  — output row within-block
                "micro",  # fd  — output col within-block
                "u",      # kc
                "ab",     # kp
                "ab",     # kq
            ],
            ["output"],
            equation="npyx,nqzw,ay,bx,ez,fw,ar,bd,er,fd,kc,kp,kq->ncrd",
        )
    ]
    inits = [
        numpy_helper.from_array(macro, "macro"),
        numpy_helper.from_array(micro, "micro"),
        numpy_helper.from_array(u, "u"),
        numpy_helper.from_array(ab, "ab"),
    ]
    graph = helper.make_graph(
        nodes,
        "task001_factorized_product",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        inits,
    )
    return helper.make_model(
        graph,
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 13)],
    )
