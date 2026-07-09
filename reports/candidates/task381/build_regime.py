import numpy as np, onnx
from onnx import helper, numpy_helper, TensorProto
F32 = TensorProto.FLOAT

m = onnx.load('submission/overfit_nets/task381.onnx')
g = m.graph

# keep nodes up to maroon8 (drop Pad, Where)
keep = []
for n in g.node:
    if n.op_type in ('Pad','Where'):
        continue
    keep.append(n)
# maroon8 is [1,1,8,10] bool

# keep predicate inits, drop pad/maroon9
drop_inits = {'pads_output','pad_axes','maroon9'}
inits = [i for i in g.initializer if i.name not in drop_inits]

def add_init(name, arr):
    inits.append(numpy_helper.from_array(arr, name))

# Maug construction operands
# Reshape maroon8 [1,1,8,10] -> [1,8,10]
add_init('shp_1_8_10', np.array([1,8,10], np.int64))
# Pad to [1,9,11]: pads over 3 dims begin[0,0,0] end[0,1,1]
add_init('pads_maug', np.array([0,0,0, 0,1,1], np.int64))
# M0 [1,9,11] fp32: corner (0,8,10)=1
M0 = np.zeros((1,9,11), np.float32); M0[0,8,10]=1.0
add_init('M0', M0)

# Sh [9,30]
Sh = np.zeros((9,30), np.float32)
for i in range(8): Sh[i, i+1]=1.0
Sh[8,:]=1.0
add_init('Sh', Sh)
# Sw [11,30]
Sw = np.zeros((11,30), np.float32)
for a in range(10): Sw[a,a]=1.0
Sw[10,:]=1.0
add_init('Sw', Sw)
# D [2,10,10]
D = np.zeros((2,10,10), np.float32)
D[0]=np.eye(10, dtype=np.float32)
D[1,:,9]=1.0; D[1,:,0]=-1.0
add_init('D', D)

nodes = list(keep)
nodes.append(helper.make_node('Cast', ['maroon8'], ['mh'], to=F32))
nodes.append(helper.make_node('Reshape', ['mh','shp_1_8_10'], ['mr']))  # [1,8,10]
nodes.append(helper.make_node('Pad', ['mr','pads_maug'], ['mpad'], mode='constant'))  # [1,9,11]
nodes.append(helper.make_node('Concat', ['M0','mpad'], ['Maug'], axis=0))  # [2,9,11]
nodes.append(helper.make_node('Einsum', ['input','D','Maug','Sh','Sw'], ['output'],
                              equation='bkhw,zkm,zia,ih,aw->bmhw'))

g2 = helper.make_graph(nodes, 'task381_regime',
    [helper.make_tensor_value_info('input', F32, [1,10,30,30])],
    [helper.make_tensor_value_info('output', F32, [1,10,30,30])],
    initializer=inits)
mm = helper.make_model(g2, opset_imports=[helper.make_opsetid('',18)])
mm.ir_version=9
onnx.save(mm, 'reports/candidates/task381/regime.onnx')
print('saved; params=', sum(int(np.prod(t.dims)) for t in inits))
