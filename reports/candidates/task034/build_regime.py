import onnx, numpy as np
from onnx import helper, numpy_helper, TensorProto

m = onnx.load('submission/overfit_nets/task034.onnx')
g = m.graph

# Keep nodes up to and including mask9 (Max), present (ReduceMax), paint (Mul).
# Drop: mask9b (Cast), mask30 (Pad), Where. Replace tail with einsum.
keep = []
for n in g.node:
    if n.output[0] in ('mask9b','mask30','output'):
        continue
    keep.append(n)

# consts
def C(name, arr):
    return numpy_helper.from_array(arr, name)

Pr = np.zeros((9,30), np.float32); 
for i in range(9): Pr[i,i]=1.0
Pc = Pr.copy()
Dc = np.zeros((2,10,10), np.float32)
Dc[0] = np.eye(10, dtype=np.float32)   # identity for keep
Dc[1] = np.ones((10,10), np.float32)   # ones for paint
ones110 = np.ones((1,10), np.float32)
one_u8 = np.ones((1,1,9,9), np.uint8)

new_inits = [C('Pr',Pr), C('Pc',Pc), C('Dc',Dc), C('ones110',ones110), C('one_u8',one_u8)]

# nodes
nodes = list(keep)
# invmask9 = one_u8 - mask9  (uint8)
nodes.append(helper.make_node('Sub', ['one_u8','mask9'], ['invmask9']))
# G_u8 = Concat([invmask9, mask9], axis=1) -> [1,2,9,9]
nodes.append(helper.make_node('Concat', ['invmask9','mask9'], ['G_u8'], axis=1))
# G = Cast fp32
nodes.append(helper.make_node('Cast', ['G_u8'], ['G_f32'], to=TensorProto.FLOAT))
# paint2 = Reshape(paint,[1,10])
nodes.append(C('shp110', np.array([1,10],np.int64)) if False else helper.make_node('Reshape',['paint','shp110'],['paint2']))
new_inits.append(C('shp110', np.array([1,10],np.int64)))
# P = Concat([ones110, paint2], axis=0) -> [2,10]
nodes.append(helper.make_node('Concat', ['ones110','paint2'], ['P'], axis=0))
# einsum
nodes.append(helper.make_node('Einsum',
    ['input','Pr','Pc','G_f32','Dc','P'], ['output'],
    equation='bkHW,hH,wW,bshw,skc,sc->bcHW'))

del g.node[:]
g.node.extend(nodes)
g.initializer.extend(new_inits)

onnx.save(m, 'reports/candidates/task034/regime.onnx')
print('saved')
