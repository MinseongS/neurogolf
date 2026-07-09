import onnx, numpy as np
from onnx import helper, numpy_helper, TensorProto as TP
m=onnx.load('submission/overfit_nets/task124.onnx')
g=m.graph
keep=[]
for n in g.node:
    keep.append(n)
    if 'out_fg' in n.output: break
used=set()
for n in keep: used.update(n.input)
keep_inits=[i for i in g.initializer if i.name in used]
f16=np.float16
new_inits=list(keep_inits)
P=np.zeros((30,10),f16)
for i in range(10): P[i,i]=1
new_inits.append(numpy_helper.from_array(P,'P'))
new_inits.append(numpy_helper.from_array(np.ones((1,1,10,10),f16),'ones_plane'))
e0=np.zeros((1,10,1),f16); e0[0,0,0]=1
new_inits.append(numpy_helper.from_array(e0,'e0col'))
t3=np.zeros((1,10,1),f16); t3[0,0,0]=3
new_inits.append(numpy_helper.from_array(t3,'three_e0'))
new_inits.append(numpy_helper.from_array(np.array([1,10,1],np.int64),'presh'))
nodes=list(keep)
nodes.append(helper.make_node('ReduceMax',['input'],['present_f'],axes=[2,3],keepdims=1))
nodes.append(helper.make_node('Cast',['present_f'],['present_h'],to=TP.FLOAT16))
nodes.append(helper.make_node('Reshape',['present_h','presh'],['present_c']))
nodes.append(helper.make_node('Sub',['present_c','three_e0'],['colB']))
nodes.append(helper.make_node('Concat',['e0col','colB'],['CMAP'],axis=2))
nodes.append(helper.make_node('Cast',['out_fg'],['outfg_h'],to=TP.FLOAT16))
nodes.append(helper.make_node('Concat',['ones_plane','outfg_h'],['K'],axis=1))
nodes.append(helper.make_node('Einsum',['P','P','CMAP','K'],['output'],equation='Hh,Ww,bck,bkhw->bcHW'))
out_vi=helper.make_tensor_value_info('output',TP.FLOAT16,[1,10,30,30])
gg=helper.make_graph(nodes,'regime',[g.input[0]],[out_vi],new_inits)
mm=helper.make_model(gg,opset_imports=[helper.make_opsetid('',12)])
mm.ir_version=m.ir_version
onnx.save(mm,'reports/candidates/task124/regime.onnx')
print('saved')
