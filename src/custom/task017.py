"""Task 017 — 0dfd9992 — cost-10 self-einsum (one-hot identity-vote fill).

The input is a 21x21 doubly-periodic pattern P[r,c] = ((rr^2 + cc^2) % mod) + 1
(period = length, 4..9) with 5 black rectangle cutouts stamped over it; the output
is the same pattern with the cutouts filled back in.

Mechanism (no period recovery, no formula rebuild, cost 10):

  * One-hot I/O is FREE (graph input/output not counted). Output is decoded as
    (out > 0), so a cell's colour is the single channel that is strictly positive.

  * Cells at the same phase (r'==r, c'==c mod length) hold the SAME pattern colour.
    Summing (voting) the one-hots of every same-phase cell therefore piles ALL
    positive mass onto the correct channel with zero colour contamination.

  * "Row r and row r' are the same pattern row" is an EXACT self-contraction:
        S[r,r'] = sum_{k>=1, x} I[k,r,x] * I[k,r',x] > 0
    Different-phase rows disagree at EVERY column (a_r != a_r' mod mod ->
    P[r,c] != P[r',c] for all c), so S = 0 exactly — a clean nonnegative identity
    gate needing no threshold. The transposed contraction gives the column gate Sc.

  * The full 2D vote  O[k,r,c] = sum_{p,q} S[r,p]*Sc[c,q]*I[k,p,q]  folds — after
    substituting S = einsum(I,I) and Sc = einsum(I,I) — into ONE degree-5 Einsum on
    the input reused five times. ONNX Einsum is variadic, so no intermediate node
    output is materialised (mem = 0).

  * A length-10 mask [0,1,...,1] on each channel-indexed operand (a, b, k) excludes
    the black cutout channel 0 from both identity gates and the output. One [10]
    initializer reused 3x = 10 params -> cost 10, points 22.697.

Exact on 5000/5000 fresh generator draws; the only theoretical failure is an entire
2D same-phase orbit being cut, impossible with 5 small rectangles.
"""

from onnx import TensorProto, helper

from ..harness import IR_VERSION


def build(task):
    mask = helper.make_tensor(
        "mask", TensorProto.FLOAT, [10], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    )
    node = helper.make_node(
        "Einsum",
        ["input", "input", "input", "input", "input", "mask", "mask", "mask"],
        ["output"],
        equation="narx,napx,nbyc,nbyq,nkpq,a,b,k->nkrc",
    )
    graph = helper.make_graph(
        [node],
        "task017",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [mask],
    )
    return helper.make_model(
        graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 12)]
    )
