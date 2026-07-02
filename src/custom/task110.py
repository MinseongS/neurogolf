"""Task 110 candidate: doubly-periodic pattern restoration collapsed to Einsums.

Rule: 29x29 grid holds a doubly-periodic colour pattern (per-axis periods; fresh
instances have equal row/col period in {2..9}); <=5 black cutout rects (<=5x5)
punch holes; output restores the full pattern.

Mechanism (walk-einsum lens: grader counts only NODE OUTPUTS, 'input'/'output'
free, >0 threshold at 'output'):
- A_stack[p,x,r] (initializer, 6x30x30): p=0 identity, p=1..5 congruence
  matrices [x==r (mod L)], L in {5,6,7,8,9}, rows/cols limited to 0..28 so the
  off-canvas row/col 29 can never be written.
- Per-axis period validity as ONE einsum producing a [6] vector: candidate L is
  valid iff no two visible cells in the same orthogonal line and same residue
  class have different colours:
    conf_row[p] = sum input[n,v,x,y] input[n,w,z,y] A[p,x,z] D[v,w]
  with D[v,w] = [v!=w, v>=1, w>=1]. All terms nonneg -> ==0 test is exact.
  Identity (p=0) is trivially conflict-free -> fallback.
- Selection = largest valid index (priority 9>8>7>6>5>identity), computed on [6]
  vectors: g = valid * [sum of higher-priority valid == 0].
- Reconstruction straight to the free 'output' in ONE einsum:
    output[n,v,r,c] = sum_{p,q,x,y} gr[p] gc[q] m[v] input[n,v,x,y] A[p,x,r] A[q,y,c]
  = #visible cells of colour v congruent to (r,c) -> >0 iff colour v present in
  the (r,c) congruence class. m masks the black channel.

Verified in numpy (proto.py): 266/266 stored, 20000 fresh (see tasklog).
"""

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ..harness import IR_VERSION


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT

    # A_stack: index 0 = identity(30), 1..5 = congruence mod 5,6,7,8,9 on 0..28
    A = np.zeros((6, 30, 30), np.float32)
    A[0] = np.eye(30, dtype=np.float32)
    idx = np.arange(29)
    for i, p in enumerate([5, 6, 7, 8, 9], start=1):
        A[i, :29, :29] = (idx[:, None] % p == idx[None, :] % p).astype(np.float32)
    init("A", A, np.float32)

    # D[v,w] = 1 iff v != w and both nonblack
    D = np.ones((10, 10), np.float32)
    np.fill_diagonal(D, 0.0)
    D[0, :] = 0.0
    D[:, 0] = 0.0
    init("D", D, np.float32)

    # U[q,p] = 1 iff q > p  (sum of strictly-higher-priority entries)
    U = np.tril(np.ones((6, 6), np.float32), -1)
    init("U", U, np.float32)

    # black-channel mask
    m = np.ones(10, np.float32)
    m[0] = 0.0
    init("m", m, np.float32)

    init("zero", np.array(0.0, np.float32), np.float32)

    # per-axis conflict vectors [6]
    nodes.append(helper.make_node(
        "Einsum", ["input", "input", "A", "D"], ["confr"],
        equation="nvxy,nwzy,pxz,vw->p"))
    nodes.append(helper.make_node(
        "Einsum", ["input", "input", "A", "D"], ["confc"],
        equation="nvxy,nwxz,pyz,vw->p"))

    def select(conf, tag):
        vb = n("Equal", [conf, "zero"], f"vb_{tag}")
        vf = n("Cast", [vb], f"vf_{tag}", to=F)
        nodes.append(helper.make_node("Einsum", [f"vf_{tag}", "U"], [f"s_{tag}"],
                                      equation="q,qp->p"))
        zb = n("Equal", [f"s_{tag}", "zero"], f"zb_{tag}")
        zf = n("Cast", [zb], f"zf_{tag}", to=F)
        return n("Mul", [f"vf_{tag}", f"zf_{tag}"], f"g_{tag}")

    gr = select("confr", "r")
    gc = select("confc", "c")

    # reconstruction straight to free 'output' (>0 threshold applied by grader)
    nodes.append(helper.make_node(
        "Einsum", [gr, gc, "m", "input", "A", "A"], ["output"],
        equation="p,q,v,nvxy,pxr,qyc->nvrc"))

    graph = helper.make_graph(
        nodes,
        "task110_period_einsum",
        [helper.make_tensor_value_info("input", F, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", F, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = IR_VERSION
    return model
