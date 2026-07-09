#!/usr/bin/env python
"""task063 regime-crack: fold Where(greenMask, green, input) into one free-output Einsum.

Deployed mask semantics: rowVal[h] > colVal[w] with
  rowVal = 2 if totRow==2 elif totRow<cnt -> 1 else 0
  colVal = 0 if totCol==2 elif totCol<cnt -> 1 else 2
mask <=> rFree&(cInt|cFree) | (rInt&~rFree)&cFree
Under signed >0 decode, multiplicity is harmless:
  mask' = rFree*cInt + rFree*cFree + rInt*cFree   (mask'>0 <=> mask)
output[m,h,w] = input[m,h,w] + mask'(h,w) * (2*e3[m]*occ(h,w) - 2*input[m,h,w])
One Einsum: 'bkhw,uh,vw,zuv,zkm->bmhw' with X=[ones,rFree,rInt,cInt,cFree].
"""
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

F32 = TensorProto.FLOAT

def init(name, arr):
    return numpy_helper.from_array(arr, name=name)

wk = np.zeros(10, np.float32); wk[2] = 1.0; wk[8] = 1.0
two = np.array([2.0], np.float32)
trueB = np.ones((1, 30), np.bool_)

# A[z,u,v]: z=0 identity (ones x ones), z=1 mask products
A = np.zeros((2, 5, 5), np.float32)
A[0, 0, 0] = 1.0        # ones * ones -> identity route
A[1, 1, 3] = 1.0        # rFree * cInt
A[1, 1, 4] = 1.0        # rFree * cFree
A[1, 2, 4] = 1.0        # rInt  * cFree

# D[z,k,m]: z=0 pass-through delta_km ; z=1 edit 2*e3[m] - 2*delta_km
D = np.zeros((2, 10, 10), np.float32)
D[0] = np.eye(10, dtype=np.float32)
D[1, :, 3] += 2.0
D[1] -= 2.0 * np.eye(10, dtype=np.float32)

nodes = [
    helper.make_node('Einsum', ['input', 'wk'], ['totRow'], equation='bkrc,k->br'),
    helper.make_node('Einsum', ['input', 'wk'], ['totCol'], equation='bkrc,k->bc'),
    helper.make_node('ReduceSum', ['input'], ['cnt'], axes=[1, 3], keepdims=0),
    helper.make_node('Less', ['totRow', 'cnt'], ['rIntB']),
    helper.make_node('Less', ['totCol', 'cnt'], ['cIntB']),
    helper.make_node('Equal', ['totRow', 'two'], ['rFreeB']),
    helper.make_node('Equal', ['totCol', 'two'], ['cFreeB']),
    helper.make_node('Concat', ['trueB', 'rFreeB', 'rIntB', 'cIntB', 'cFreeB'], ['Xb'], axis=0),
    helper.make_node('Cast', ['Xb'], ['X'], to=F32),
    helper.make_node('Einsum', ['input', 'X', 'X', 'A', 'D'], ['output'],
                     equation='bkhw,uh,vw,zuv,zkm->bmhw'),
]

graph = helper.make_graph(
    nodes, 'task063_regime',
    [helper.make_tensor_value_info('input', F32, [1, 10, 30, 30])],
    [helper.make_tensor_value_info('output', F32, [1, 10, 30, 30])],
    initializer=[init('wk', wk), init('two', two), init('trueB', trueB),
                 init('A', A), init('D', D)],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 12)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, 'reports/candidates/task063/regime.onnx')
print('saved, params =', sum(np.prod(t.dims) for t in model.graph.initializer))
