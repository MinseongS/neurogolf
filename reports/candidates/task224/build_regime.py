"""task224 regime-crack: fold rect1/rect2/XOR/Pad/Where into one free-output Einsum.

output[b,k,h,w] = count[k] * ( input[k,h,w]*(O(h)O(w) - R1C1 + R2C2)
                               + mask[k]*ingrid(h,w)*(R1C1 - R2C2) )
  count[k]  = sum_xy input[k,x,y]        (2nd input occurrence, free)
  ingrid    = sum_s input[s,h,w]         (1st occurrence contracted vs ones)
  O         = ones16 zero-padded to 30 (identity gate; grids <= 16)
  R1,R2/C1,C2 = nested bbox spans of the color-5 markers (exact bools)
frame = R1C1 - R2C2 is exact 0/1 because span2 is built nested
(ge2: grid > rmin+1, le2: grid+1 < rmax  -- no u8 underflow).
Decode-equivalent to deployed Where(frame_full, paint, input) under >0.
"""
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto as TP

F32, U8, I64, BOOL = TP.FLOAT, TP.UINT8, TP.INT64, TP.BOOL

inits = []
def init(name, arr):
    inits.append(numpy_helper.from_array(arr, name))
    return name

e5 = np.zeros(10, np.float32); e5[5] = 1
init('e5', e5)
init('grid', np.arange(16, dtype=np.uint8).reshape(1, 1, 16, 1))
init('grid2', (np.arange(16, dtype=np.uint8) + 1).reshape(1, 1, 16, 1))
init('one_u8', np.array(1, np.uint8))
init('ones16', np.ones((1, 1, 16, 1), bool))
init('pads', np.array([0, 0, 0, 0, 0, 0, 14, 0], np.int64))

Fp = np.zeros((5, 5, 2), np.float32)
Fp[0, 0, 0] = 1                      # identity: O(h)*O(w), v0
Fp[1, 3, 0] = -1; Fp[2, 4, 0] = 1    # v0: -R1C1 + R2C2
Fp[1, 3, 1] = 1;  Fp[2, 4, 1] = -1   # v1: +R1C1 - R2C2
init('Fp', Fp)

mask = np.array([0, 1, 1, 1, 1, 0, 1, 1, 1, 1], np.float32)
G = np.zeros((2, 10, 10), np.float32)
G[0] = np.eye(10, dtype=np.float32)          # v0: delta_ks
G[1] = mask[:, None] * np.ones(10, np.float32)  # v1: mask[k] * 1[s]
init('G', G)

N = []
N.append(helper.make_node('Einsum', ['input', 'e5'], ['row_sum'], equation='bchw,c->bh'))
N.append(helper.make_node('Einsum', ['input', 'e5'], ['col_sum'], equation='bchw,c->bw'))
for ax in ('row', 'col'):
    N.append(helper.make_node('Cast', [f'{ax}_sum'], [f'{ax}_hb'], to=BOOL))
    N.append(helper.make_node('Cast', [f'{ax}_hb'], [f'{ax}_has'], to=U8))
N.append(helper.make_node('ArgMax', ['row_has'], ['rmin_i'], axis=1, keepdims=1))
N.append(helper.make_node('ArgMax', ['row_has'], ['rmax_i'], axis=1, keepdims=1, select_last_index=1))
N.append(helper.make_node('ArgMax', ['col_has'], ['cmin_i'], axis=1, keepdims=1))
N.append(helper.make_node('ArgMax', ['col_has'], ['cmax_i'], axis=1, keepdims=1, select_last_index=1))
for s in ('rmin', 'rmax', 'cmin', 'cmax'):
    N.append(helper.make_node('Cast', [f'{s}_i'], [s], to=U8))
N.append(helper.make_node('Add', ['rmin', 'one_u8'], ['top1']))
N.append(helper.make_node('Add', ['cmin', 'one_u8'], ['left1']))
for pre, lo, hi, lo2 in (('r', 'rmin', 'rmax', 'top1'), ('c', 'cmin', 'cmax', 'left1')):
    N.append(helper.make_node('Greater', ['grid', lo], [f'{pre}ge1']))
    N.append(helper.make_node('Less', ['grid', hi], [f'{pre}le1']))
    N.append(helper.make_node('And', [f'{pre}ge1', f'{pre}le1'], [f'{pre}s1']))
    N.append(helper.make_node('Greater', ['grid', lo2], [f'{pre}ge2']))
    N.append(helper.make_node('Less', ['grid2', hi], [f'{pre}le2']))
    N.append(helper.make_node('And', [f'{pre}ge2', f'{pre}le2'], [f'{pre}s2']))
N.append(helper.make_node('Concat', ['ones16', 'rs1', 'rs2', 'cs1', 'cs2'], ['S_b'], axis=1))
N.append(helper.make_node('Pad', ['S_b', 'pads'], ['S_p']))
N.append(helper.make_node('Cast', ['S_p'], ['S'], to=F32))
N.append(helper.make_node('Einsum', ['input', 'input', 'S', 'S', 'Fp', 'G'], ['output'],
                          equation='bshw,bkxy,bphi,bqwj,pqv,vks->bkhw'))

graph = helper.make_graph(
    N, 'task224_regime',
    [helper.make_tensor_value_info('input', F32, [1, 10, 30, 30])],
    [helper.make_tensor_value_info('output', F32, [1, 10, 30, 30])],
    inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 16)])
model.ir_version = 10
onnx.checker.check_model(model)
onnx.save(model, '/Users/minseong/project/neurogolf/reports/candidates/task224/regime.onnx')
print('saved. params =', sum(int(np.prod(t.dims)) if t.dims else 1 for t in inits))
