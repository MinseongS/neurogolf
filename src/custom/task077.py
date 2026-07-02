"""Task 077 candidate — rectangle restore as ONE checkpoint-walk Einsum
(walk_einsum_iteration_collapse).

Exact rule (validated stored 266/266 + fresh 20000/20000):
- Reds cluster by Chebyshev<=2 connectivity; generator gap>=2 => cross-rect red
  distance >=3, so clusters never bridge rects.
- Rect == bbox(red cluster); cell (r,c) filled yellow iff it is static (not red,
  not empty) and lies in some cluster's bbox.
- bbox membership as ONE walk through red cells (jumps Chebyshev<=2, penta band J)
  with 4 checkpoint constraints against the output indices r,c via ONE shared
  triangular matrix T (subscript order swapped for >=):
    row(p0)<=r, row(p4)>=r, col(p8)<=c, col(p12)>=c.
  Measured max intra-cluster jump diameter = 3 -> segments of 4 have margin.
- Red plane is read straight from the FREE graph input via channel selector e2,
  so counted tensors are only walk (3600B) and fill bool (900B).
- All operands nonnegative, J dyadic 0.5 -> fp32 rounding cannot flip >0.
"""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper

from ._exact import model, tensor

STEPS = 12          # 3 segments of 4 jumps; measured max needed segment = 3
CHECK1, CHECK2 = 4, 8
LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build(task):
    # penta-band jump matrix (Chebyshev<=2 per axis), dyadic 0.5, shared row/col
    J = np.zeros((30, 30), np.float32)
    for i in range(30):
        for j in range(max(0, i - 2), min(30, i + 3)):
            J[i, j] = 0.5

    # upper triangular incl. diagonal: T[i, j] = 1 iff i <= j
    T = np.triu(np.ones((30, 30), np.float32))

    e2 = np.zeros(10, np.float32)
    e2[2] = 1.0
    stat = np.ones(10, np.float32)   # static = not empty, not red
    stat[0] = 0.0
    stat[2] = 0.0

    inits = [
        tensor("J", J),
        tensor("T", T),
        tensor("e2", e2),
        tensor("stat", stat),
        tensor("zero_f", np.array(0.0, dtype=np.float32)),
        tensor("yellow", np.eye(10, dtype=np.float32)[4].reshape(1, 10, 1, 1)),
    ]

    pool = list(LETTERS)
    n = pool.pop(0)
    r = pool.pop(0)
    c = pool.pop(0)
    a = [pool.pop(0) for _ in range(STEPS + 1)]
    b = [pool.pop(0) for _ in range(STEPS + 1)]
    q = [pool.pop(0) for _ in range(STEPS + 2)]

    subs = [n + q[0] + a[0] + b[0], q[0], a[0] + r]     # red at p0, row(p0) <= r
    ops = ["input", "e2", "T"]
    for j in range(1, STEPS + 1):
        subs += [a[j - 1] + a[j], b[j - 1] + b[j], n + q[j] + a[j] + b[j], q[j]]
        ops += ["J", "J", "input", "e2"]
        if j == CHECK1:
            subs.append(r + a[j])       # r <= row(p4)
            ops.append("T")
        elif j == CHECK2:
            subs.append(b[j] + c)       # col(p8) <= c
            ops.append("T")
    subs.append(c + b[STEPS])           # c <= col(p12)
    ops.append("T")
    subs += [n + q[STEPS + 1] + r + c, q[STEPS + 1]]    # static at (r,c)
    ops += ["input", "stat"]
    equation = ",".join(subs) + "->" + n + r + c

    nodes = [
        helper.make_node("Einsum", ops, ["walk"], equation=equation),
        helper.make_node("Greater", ["walk", "zero_f"], ["fill"]),
        helper.make_node("Where", ["fill", "yellow", "input"], ["output"]),
    ]
    return model("task077_walk_einsum", nodes, inits, output_dtype=TensorProto.FLOAT, opset=17)
