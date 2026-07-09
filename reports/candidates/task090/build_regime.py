import onnx, numpy as np
from onnx import helper, numpy_helper, TensorProto
from src import harness

m = onnx.load('submission/overfit_nets/task090.onnx')
g = m.graph

# Drop the final And (safe_name_607) and Where(->output) nodes; keep strips 604 (row bool [1,1,30,1]) and 606 (col bool [1,1,1,30])
keep=[]
for n in g.node:
    if n.output and n.output[0] in ('safe_name_607','output'):
        continue
    keep.append(n)

# Remove unused pink init safe_name_7
inits=[i for i in g.initializer if i.name!='safe_name_7']

def np_init(name, arr):
    return numpy_helper.from_array(arr.astype(arr.dtype), name=name)

new_inits=[]
# W[s,c,k] : k=0 identity, k=1 = δ6[c]-δ(s,c)
W=np.zeros((10,10,2),dtype=np.float32)
for s in range(10):
    W[s,s,0]=1.0
    W[s,:,1]= -np.eye(10)[s]
    W[s,6,1]+=1.0
new_inits.append(np_init('rc_W', W))
new_inits.append(np_init('rc_rs_shape', np.array([1,30],dtype=np.int64)))  # reshape target
# pad amounts [begin_r, begin_c, end_r, end_c] for 2D tensor -> pad 1 row at top
new_inits.append(np_init('rc_pads', np.array([1,0,0,0],dtype=np.int64)))
new_inits.append(np_init('rc_padval', np.array(1.0,dtype=np.float32)))

nodes=list(keep)
def N(op,ins,outs,**kw):
    nodes.append(helper.make_node(op,ins,outs,**kw))

# Row strip
N('Cast',['safe_name_604'],['rc_Rf'],to=TensorProto.FLOAT)
N('Reshape',['rc_Rf','rc_rs_shape'],['rc_Rr'])
N('Pad',['rc_Rr','rc_pads','rc_padval'],['rc_Rk'])
# Col strip
N('Cast',['safe_name_606'],['rc_Cf'],to=TensorProto.FLOAT)
N('Reshape',['rc_Cf','rc_rs_shape'],['rc_Cr'])
N('Pad',['rc_Cr','rc_pads','rc_padval'],['rc_Ck'])
# Fold
N('Einsum',['input','rc_W','rc_Rk','rc_Ck'],['output'],equation='bshw,sck,kh,kw->bchw')

ng=helper.make_graph(nodes,g.name,g.input,g.output,list(inits)+new_inits)
nm=helper.make_model(ng,opset_imports=m.opset_import)
nm.ir_version=m.ir_version
onnx.save(nm,'reports/candidates/task090/regime.onnx')
print('saved')
