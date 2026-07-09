#!/usr/bin/env python
"""task094 regime-crack: fold Where(line_bg30, pink, input) into ONE free-output Einsum.

Rule: 15x15 grid, cyan(8) bg, blue(1) 5x5 box outlines centered (r,c) in 3..11;
paint full row r and col c pink(6); blue overwrites pink.

Mechanism:
  - detection: row_score[i] = sum_hw input[b,1,h,w] * D[i,h]  (D = 2@h=i+1, 1@h=i+3, 2@h=i+5)
    == deployed dilated-Conv score * 13; threshold 20.5. Same for cols (shared D).
  - Less(score, 20.5) -> NOT-center bool -> Cast -> Pad(value=1) builds P[2,30]:
      P[0,h] = 1 (identity term), P[1,h] = 1 - R(h)  (R = center-row indicator at h)
  - final: output[b,c,h,w] = sum_{t,s} input[b,s,h,w] * W[t,s,c] * P[t,h] * Q[t,w]
      t=0: W[0]=eye, W[0][8,8]=0, W[0][8,6]=1  -> input + input[8]*(e6-e8)
      t=1: W[1][8,6]=-1, W[1][8,8]=+1          -> -input[8]*(e6-e8)*(1-R)(1-C)
    => output = input + (1-(1-R)(1-C)) * input[8] * (e6-e8)
    Crosshair overlap: output[6]=1>0, output[8]=0 (t=0 kills it) -> sign-correct.
    Off-grid: linear in input -> zeros.
"""
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto as TP

def init(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)

# D [9,30]: detection taps (shared row/col)
D = np.zeros((9, 30), dtype=np.float32)
for i in range(9):
    D[i, i + 1] = 2.0
    D[i, i + 3] = 1.0
    D[i, i + 5] = 2.0

e1 = np.zeros(10, dtype=np.float32); e1[1] = 1.0

W = np.zeros((2, 10, 10), dtype=np.float32)
W[0] = np.eye(10, dtype=np.float32)
W[0, 8, 8] = 0.0
W[0, 8, 6] = 1.0
W[1, 8, 6] = -1.0
W[1, 8, 8] = 1.0

inits = [
    init('D', D),
    init('e1', e1),
    init('thr', np.float32(20.5)),          # rank-0
    init('padv', np.float32(1.0)),          # rank-0 pad constant
    init('pads', np.array([1, 3, 0, 18], dtype=np.int64)),  # [1,9] -> [2,30]
    init('W', W),
]

nodes = [
    helper.make_node('Einsum', ['input', 'e1', 'D'], ['rs'], equation='bshw,s,ih->bi'),
    helper.make_node('Einsum', ['input', 'e1', 'D'], ['cs'], equation='bshw,s,iw->bi'),
    helper.make_node('Less', ['rs', 'thr'], ['rn']),
    helper.make_node('Less', ['cs', 'thr'], ['cn']),
    helper.make_node('Cast', ['rn'], ['rnf'], to=TP.FLOAT),
    helper.make_node('Cast', ['cn'], ['cnf'], to=TP.FLOAT),
    helper.make_node('Pad', ['rnf', 'pads', 'padv'], ['P'], mode='constant'),
    helper.make_node('Pad', ['cnf', 'pads', 'padv'], ['Q'], mode='constant'),
    helper.make_node('Einsum', ['input', 'W', 'P', 'Q'], ['output'],
                     equation='bshw,tsc,th,tw->bchw'),
]

graph = helper.make_graph(
    nodes, 'task094_regime',
    [helper.make_tensor_value_info('input', TP.FLOAT, [1, 10, 30, 30])],
    [helper.make_tensor_value_info('output', TP.FLOAT, [1, 10, 30, 30])],
    inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, 'reports/candidates/task094/regime.onnx')
print('saved. params =', sum(np.prod(i.dims) if i.dims else 1 for i in model.graph.initializer))
