import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper as nh
F=TensorProto.FLOAT; F16=TensorProto.FLOAT16
def const(n,a): return nh.from_array(np.asarray(a),name=n)
nodes=[];inits=[]
def N(op,i,o,**k):
    nodes.append(helper.make_node(op,i,o,**k)); return o[0]

e2r=np.zeros((1,10),np.float32); e2r[0,2]=1.0
e3r=np.zeros((1,10),np.float32); e3r[0,3]=1.0
rowc2=np.arange(30,dtype=np.float32).reshape(1,30)
colc2=np.arange(30,dtype=np.float32).reshape(1,30)
Dmat=np.zeros((10,10),np.float32); Dmat[0,8]=1.0; Dmat[0,0]=-1.0
Dfull=np.stack([np.eye(10,dtype=np.float32),Dmat,Dmat],0)
ones=np.ones((1,30),np.float32)
for n,a in [('e2r',e2r),('e3r',e3r),('rowc2',rowc2),('colc2',colc2),
            ('Dfull',Dfull),('ones',ones)]:
    inits.append(const(n,a))

N('Einsum',['input','e2r'],['Rr'],equation='bchw,kc->kh')   # [1,30] red-row
N('Einsum',['input','e3r'],['Gc'],equation='bchw,kc->kw')   # [1,30] green-col
# scalar positions [1]
N('Einsum',['input','e2r','rowc2'],['r_red'],equation='bchw,mc,nh->')
N('Einsum',['input','e3r','rowc2'],['r_grn'],equation='bchw,mc,nh->')
N('Einsum',['input','e2r','colc2'],['c_red'],equation='bchw,mc,nw->')
N('Einsum',['input','e3r','colc2'],['c_grn'],equation='bchw,mc,nw->')

# polynomial factors: q_row=(rowc-r_red)(r_grn-rowc), q_col=(colc-c_red)(c_grn-colc)
N('Sub',['rowc2','r_red'],['ar']); N('Sub',['r_grn','rowc2'],['br']); N('Mul',['ar','br'],['q_row'])
N('Sub',['colc2','c_red'],['ac']); N('Sub',['c_grn','colc2'],['bc']); N('Mul',['ac','bc'],['q_col'])
N('Add',['q_col','Gc'],['cHq'])   # +green-col makes corner cell positive

N('Concat',['ones','Rr','q_row'],['RF'],axis=0)
N('Concat',['ones','cHq','Gc'],['CF'],axis=0)
N('Einsum',['input','Dfull','RF','CF'],['output'],equation='bchw,tcC,th,tw->bChw')

g=helper.make_graph(nodes,'r246',
    [helper.make_tensor_value_info('input',F,[1,10,30,30])],
    [helper.make_tensor_value_info('output',F,[1,10,30,30])],inits)
m=helper.make_model(g,opset_imports=[helper.make_opsetid('',18)]); m.ir_version=9
onnx.save(m,'reports/candidates/task246/regime.onnx'); print('saved')
