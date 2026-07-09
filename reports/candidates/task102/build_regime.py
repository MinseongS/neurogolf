import onnx, numpy as np, copy
from onnx import helper, numpy_helper, TensorProto
from src.harness import load_task, evaluate

base = onnx.load('submission/overfit_nets/task102.onnx')
g = base.graph
# Keep detection subgraph producing 'fill_core' [1,1,10,10] uint8. Drop Cast->bool, Pad, Where.
keep_ops = []
for n in g.node:
    if n.op_type in ('Cast',) and 'fill_core_bool' in n.output: continue
    if n.op_type=='Pad': continue
    if n.op_type=='Where': continue
    keep_ops.append(n)

nodes = list(keep_ops)
inits = [i for i in g.initializer if i.name not in ('pad_fill30','pad_axes_hw','red_vec')]

# fill_core [1,1,10,10] u8 -> fp32 [10,10]
nodes.append(helper.make_node('Cast',['fill_core'],['fc_f'],to=TensorProto.FLOAT))
nodes.append(helper.make_node('Squeeze',['fc_f','sq_axes'],['fc10']))  # [10,10]
inits.append(numpy_helper.from_array(np.array([0,1],dtype=np.int64),'sq_axes'))
# delta init [10,10] = e_{0,0}
delta = np.zeros((10,10),np.float32); delta[0,0]=1.0
inits.append(numpy_helper.from_array(delta[None],'delta_row'))     # [1,10,10]
nodes.append(helper.make_node('Unsqueeze',['fc10','un_axes'],['fc_row']))  # [1,10,10]
inits.append(numpy_helper.from_array(np.array([0],dtype=np.int64),'un_axes'))
nodes.append(helper.make_node('Concat',['delta_row','fc_row'],['FC'],axis=0))  # [2,10,10]

# Placement S [2,30,10]: t0 = col0 all ones ; t1 = shift +1 identity
S = np.zeros((2,30,10),np.float32)
S[0,:,0]=1.0
for h in range(10):
    S[1,h+1,h]=1.0
inits.append(numpy_helper.from_array(S,'S_place'))
# Channel mix M [2,10,10]: M0=I, M1=A (A[0,2]=1,A[0,0]=-1)
M = np.zeros((2,10,10),np.float32)
M[0]=np.eye(10)
M[1,0,2]=1.0; M[1,0,0]=-1.0
inits.append(numpy_helper.from_array(M,'M_mix'))

nodes.append(helper.make_node('Einsum',['input','M_mix','S_place','S_place','FC'],['output'],
                              equation='bkHW,tkc,tHh,tWw,thw->bcHW'))

newg = helper.make_graph(nodes, g.name, list(g.input), list(g.output), inits)
m = helper.make_model(newg, opset_imports=list(base.opset_import), ir_version=base.ir_version)
onnx.save(m,'reports/candidates/task102/regime.onnx')
r = evaluate('reports/candidates/task102/regime.onnx', load_task(102))
print({k:r[k] for k in ('ok','pass','fail','memory','params','error')})
print('deployed total 2527; this total', (r['memory'] or 0)+(r['params'] or 0))
