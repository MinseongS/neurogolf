#!/usr/bin/env python
"""task273 regime crack: corner-rectangle interior fill -> ONE free-output Einsum.

Rule: yellow(4) corner quadruples mark rectangles on a 10x10 grid; fill strict
interior with red(2).  Interior(h,w) <=> yellow strictly NW & NE & SW & SE.
Each quadrant count is multilinear in input: sum_{s,a,b} input[b,s,a,b]*e4[s]*T[a,h]*T[b,w]
with T[a,h]>0 iff a<h, factored T = P('ak') @ Q('kh'), k=9 staircase
(P[a,k]=[a<=k], Q[k,h]=[k<h]).  t-axis: t=0 identity (P/Q rank-1 ones, E=1/128
so each factor = 100/128 = 25/32, product (25/32)^4 ~ 0.3725 <= 1, W[0]=I),
t=1 fill (E=e4, W[1,x,c]=V[c]*(1-[x==4]) folds the not-yellow guard + red
routing via one-hot input; V=+1 at c==2 else -1).  Decode is (output>0):
non-fill cell -> 0.3725*onehot; fill cell -> ch2=+count>0, others<=0.3725-1<0;
off-grid -> 0 (every group carries an input factor).  memory=0, params=1300.
"""
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

P = np.zeros((2, 30, 9), np.float32)
Q = np.zeros((2, 9, 30), np.float32)
P[0, :, 0] = 1.0
Q[0, 0, :] = 1.0
P[1] = (np.arange(30)[:, None] <= np.arange(9)[None, :]).astype(np.float32)
Q[1] = (np.arange(9)[:, None] < np.arange(30)[None, :]).astype(np.float32)
E = np.zeros((2, 10), np.float32)
E[0, :] = 1.0 / 128.0
E[1, 4] = 1.0
W = np.zeros((2, 10, 10), np.float32)
W[0] = np.eye(10, dtype=np.float32)
V = -np.ones(10, np.float32)
V[2] = 1.0
ny = np.ones(10, np.float32)
ny[4] = 0.0
W[1] = ny[:, None] * V[None, :]

# Operand order tuned for ORT's left-to-right pairwise contraction: after G1
# reduces to a [t,h,w]-shaped accumulator, each later group joins its
# h/w-linked T-factor FIRST so the running intermediate never blows past
# ~[t,h,w,30] (v0 naive order: 99.5 ms/inf; this order: 3.0 ms/inf, identical).
eq = ('bijg,ti,tjk,tkh,tgl,tlw,'   # G1 NW: yellow(j,g), j<h, g<w
      'tph,tnp,bmno,tm,twq,tqo,'   # G2 NE: yellow(n,o), n<h, w<o
      'thv,tvs,brsu,tr,tuy,tyw,'   # G3 SW: yellow(s,u), h<s, u<w
      'the,tea,bzad,tz,twf,tfd,'   # G4 SE: yellow(a,d), h<a, w<d
      'bxhw,txc->bchw')            # identity/value slot

inputs = (['input', 'E', 'P', 'Q', 'P', 'Q']
          + ['Q', 'P', 'input', 'E', 'P', 'Q']
          + ['P', 'Q', 'input', 'E', 'P', 'Q']
          + ['P', 'Q', 'input', 'E', 'P', 'Q']
          + ['input', 'W'])

node = helper.make_node('Einsum', inputs, ['output'], equation=eq)
graph = helper.make_graph(
    [node], 'task273_regime',
    [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 10, 30, 30])],
    [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10, 30, 30])],
    [numpy_helper.from_array(P, 'P'), numpy_helper.from_array(Q, 'Q'),
     numpy_helper.from_array(E, 'E'), numpy_helper.from_array(W, 'W')])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
model.ir_version = 8
onnx.checker.check_model(model, full_check=True)
onnx.save(model, 'reports/candidates/task273/regime.onnx')
print('saved; params =', P.size + Q.size + E.size + W.size)
