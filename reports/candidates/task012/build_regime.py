import numpy as np, onnx
from onnx import helper as h, TensorProto as TP, numpy_helper as nh

# stamp kernels
K_C=np.zeros((5,5),np.float32); K_A=np.zeros((5,5),np.float32)
for dr in range(-2,3):
  for dc in range(-2,3):
    if abs(dr)==abs(dc): K_C[dr+2,dc+2]=1
    elif (dr==0)!=(dc==0): K_A[dr+2,dc+2]=1
Kall=K_C+K_A
delta00=np.zeros((5,5),np.float32); delta00[2,2]=1
negKall=-Kall
eye10=np.eye(10,dtype=np.float32)
e0=np.zeros(10,np.float32); e0[0]=1
ones10=np.ones(10,np.float32)

I=[
 nh.from_array(np.array([3],np.int64),'k3'),
 nh.from_array(eye10,'eye10'),
 nh.from_array(np.array(1,np.int64),'i1'),
 nh.from_array(np.array(2,np.int64),'i2'),
 nh.from_array(K_C,'K_C'), nh.from_array(K_A,'K_A'),
 nh.from_array(delta00,'delta00'), nh.from_array(negKall,'negKall'),
 nh.from_array(e0,'e0'), nh.from_array(ones10,'ones10'),
]
N=[]
# color counts per channel
N.append(h.make_node('Einsum',['input'],['counts'],equation='bchw->c'))
N.append(h.make_node('TopK',['counts','k3'],['tkv','tkc'],axis=0,largest=1,sorted=1))
# arm = index1, center = index2
N.append(h.make_node('Gather',['tkc','i1'],['arm_i'],axis=0))
N.append(h.make_node('Gather',['tkc','i2'],['ctr_i'],axis=0))
N.append(h.make_node('Gather',['eye10','arm_i'],['eA'],axis=0))   # [10]
N.append(h.make_node('Gather',['eye10','ctr_i'],['eC'],axis=0))   # [10]
# build W[10,10,5,5] float
N.append(h.make_node('Einsum',['eC','eC','K_C'],['wCC'],equation='o,i,de->oide'))
N.append(h.make_node('Einsum',['eA','eC','K_A'],['wAC'],equation='o,i,de->oide'))
N.append(h.make_node('Einsum',['e0','ones10','delta00'],['wOG'],equation='o,i,de->oide'))
N.append(h.make_node('Einsum',['e0','eC','negKall'],['wNS'],equation='o,i,de->oide'))
N.append(h.make_node('Sum',['wCC','wAC','wOG','wNS'],['wF']))
N.append(h.make_node('Cast',['wF'],['W'],to=TP.INT8))
# input -> uint8
N.append(h.make_node('Cast',['input'],['xu8'],to=TP.UINT8))
# ConvInteger -> output (free), int32
N.append(h.make_node('ConvInteger',['xu8','W'],['output'],pads=[2,2,2,2],strides=[1,1],dilations=[1,1]))

g=h.make_graph(N,'t012',
  [h.make_tensor_value_info('input',TP.FLOAT,[1,10,30,30])],
  [h.make_tensor_value_info('output',TP.INT32,[1,10,30,30])], I)
m=h.make_model(g,opset_imports=[h.make_opsetid('',13)])
m.ir_version=10
onnx.save(m,'reports/candidates/task012/regime.onnx')
print('saved')
