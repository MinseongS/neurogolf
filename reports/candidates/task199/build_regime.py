#!/usr/bin/env python
"""task199 regime-crack: fold yellow-comb + moved-pixel into one free-output Einsum.

Rule: bg=0 grid (size 3..15) with ONE pixel colour cc at (row,col).
Output: pixel at (row+1,col); yellow(4) at all (r,c) with r<=row and c%2==col%2.

out[k,h,w] = sum_u input[u,h,w] * sum_t K[t,k]*A[t,h]*B[t,w]
  t0 base  : e0        x ones          x ones
  t1 yellow: (e4-e0)   x [h<=row]      x paritymatch(w,col)
  t2 pixel : (ecc-e0)  x d_{h,row+1}   x d_{w,col}
"""
import numpy as np
import onnx
from onnx import helper, TensorProto as TP

F = TP.FLOAT

def init(name, arr):
    arr = np.asarray(arr)
    return helper.make_tensor(name, F if arr.dtype == np.float32 else TP.INT64,
                              arr.shape, arr.flatten().tolist())

inits, nodes = [], []

# ---- initializers ----
m = np.zeros((1, 10), np.float32); m[0, 1:] = 1.0            # non-bg channel picker
inits.append(init("m", m))                                    # 10
alt = ((-1.0) ** np.arange(30)).astype(np.float32).reshape(1, 1, 30)
inits.append(init("alt", alt))                                # 30 ((-1)^w, reused 2x)
inits.append(init("ones30", np.ones((1, 1, 30), np.float32))) # 30
e0 = np.zeros((1, 1, 10), np.float32); e0[0, 0, 0] = 1.0
inits.append(init("e0", e0))                                  # 10 (K row0 + Sub, reused)
K1 = np.zeros((1, 1, 10), np.float32)
K1[0, 0, 0] = -1.0; K1[0, 0, 4] = 1.0                         # t1: e4-e0
inits.append(init("K1", K1))                                  # 10
inits.append(init("convW", np.array([[[1.0, 0.0]]], np.float32)))  # 2 (shift down 1)
inits.append(init("ax2", np.array([2], np.int64)))            # 1 (cumsum axis)

# ---- nodes ----
nodes.append(helper.make_node("Einsum", ["input", "m"], ["rowvec"], equation="bshw,us->buh"))
nodes.append(helper.make_node("Einsum", ["input", "m"], ["colvec"], equation="bshw,us->buw"))
nodes.append(helper.make_node("Einsum", ["input", "m"], ["ccvec"], equation="bshw,us->bus"))
nodes.append(helper.make_node("CumSum", ["rowvec", "ax2"], ["le"], exclusive=0, reverse=1))
nodes.append(helper.make_node("Conv", ["rowvec", "convW"], ["sh"], kernel_shape=[2], pads=[1, 0]))
# pmsign[w] = (-1)^(w+col) = sum_c colvec[c]*alt[c]*alt[w]: +1 iff parity match
nodes.append(helper.make_node("Einsum", ["colvec", "alt", "alt"], ["pmsign"], equation="buc,vuc,xuw->buw"))
nodes.append(helper.make_node("Sub", ["ccvec", "e0"], ["ccm"]))
nodes.append(helper.make_node("Concat", ["ones30", "le", "sh"], ["A"], axis=1))
nodes.append(helper.make_node("Concat", ["ones30", "pmsign", "colvec"], ["B"], axis=1))
nodes.append(helper.make_node("Concat", ["e0", "K1", "ccm"], ["K"], axis=1))
nodes.append(helper.make_node("Einsum", ["input", "K", "A", "B"], ["output"],
                              equation="buhw,btk,bth,btw->bkhw"))

graph = helper.make_graph(
    nodes, "task199_regime",
    [helper.make_tensor_value_info("input", F, [1, 10, 30, 30])],
    [helper.make_tensor_value_info("output", F, [1, 10, 30, 30])],
    inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model, full_check=True)
onnx.save(model, "/Users/minseong/project/neurogolf/reports/candidates/task199/regime.onnx")
print("saved")
