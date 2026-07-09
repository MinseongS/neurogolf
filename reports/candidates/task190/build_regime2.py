"""task190 REGIME-CRACK v2: box-gated diagonal-delta fold.
Rule: 2x2 block + up to 4 diagonal corner tips; each tipped corner spawns an
outward 45-deg ray to the grid edge, painted the grid colour.
Reparametrize by u=r-c (anti-diag), w=r+c (main-diag); the two block diagonals
never share an integer cell (D,S opposite parity) -> cross-pollution free.
Each ray = degree-2 diagonal delta (1-(r-c-D)^2 or 1-(r+c-S)^2) gated by a
DISJOINT row-box x col-box quadrant (axis-aligned steps). Disjoint quadrants =>
at most one delta active per cell. Fold Where(ray,colour,input) into ONE
free-output Einsum:  output = input + input[0]*RAY*(e_v - e_0).
"""
import onnx, numpy as np
from onnx import helper, numpy_helper, TensorProto as TP
from src.harness import load_task, evaluate

F = TP.FLOAT
nodes, inits = [], []
def ini(arr, name):
    inits.append(numpy_helper.from_array(np.asarray(arr), name)); return name
def nd(op, i, o, **kw):
    nodes.append(helper.make_node(op, list(i), [o], **kw)); return o

# ---- constants / static einsum operands ----
ini(np.float32(1.0),'one'); ini(np.float32(0.0),'zero')
ini(np.arange(9,dtype=np.float32),'ids9')
ini(np.arange(10,dtype=np.float32),'ids10')
ini(np.arange(30,dtype=np.float32),'ids30')
ini(np.float32(1.0),'c1'); ini(np.float32(2.0),'c2'); ini(np.float32(9.0),'nine')
ini(np.array([0.]+[1.]*9,dtype=np.float32),'chmask')
ini(np.array([1.]+[0.]*9,dtype=np.float32),'e0')
ini(np.ones(30,dtype=np.float32),'ones30')
ini(np.ones((10,1),dtype=np.float32),'onescol')
P = np.stack([np.ones(30),np.arange(30.),np.arange(30.)**2]).astype(np.float32); ini(P,'P')
Ax = np.zeros((10,5),np.float32); Ax[0,:4]=1.0; Ax[:,4]=1.0; ini(Ax,'Ax')
Cxe = np.ones((10,10,5),np.float32); Cxe[:,:,4]=np.eye(10); ini(Cxe,'Cxe')
ini(np.array([[1,0,-1],[0,2,0],[-1,0,0]],np.float32),'MaC0')
ini(np.array([[0,-2,0],[2,0,0],[0,0,0]],np.float32),'MaC1')
ini(np.array([[-1,0,0],[0,0,0],[0,0,0]],np.float32),'Cq')
ini(np.array([[1,0,-1],[0,-2,0],[-1,0,0]],np.float32),'MmC0')
ini(np.array([[0,2,0],[2,0,0],[0,0,0]],np.float32),'MmC1')
d00=np.zeros((3,3),np.float32); d00[0,0]=1; ini(d00,'delta00')
ini(np.array([0,0,0],np.int64),'s000'); ini(np.array([1,10,10],np.int64),'e11010'); ini(np.array([1,2,3],np.int64),'a123')
ini(np.array([0,1],np.int64),'sq01')
ini(np.array([0],np.int64),'z1'); ini(np.array([9],np.int64),'n9'); ini(np.array([1],np.int64),'ax1')
ini(np.array([10],np.int64),'t10'); ini(np.array([1],np.int64),'s1'); ini(np.array([0],np.int64),'ax0')

# ---- extraction ----
nd('Slice',['input','s000','e11010','a123'],'bg10')
nd('Sub',['one','bg10'],'occ10')
nd('Squeeze',['occ10','sq01'],'occ')                    # [10,10]
nd('Slice',['occ','z1','n9','ax1'],'occ_c0')
nd('Slice',['occ','s1','t10','ax1'],'occ_c1')
nd('Mul',['occ_c0','occ_c1'],'hpair')                   # [10,9]
nd('Slice',['hpair','z1','n9','ax0'],'hp_r0')
nd('Slice',['hpair','s1','t10','ax0'],'hp_r1')
nd('Mul',['hp_r0','hp_r1'],'blk')                       # [9,9]
nd('Einsum',['blk','ids9'],'r0',equation='rc,r->')
nd('Einsum',['blk','ids9'],'c0',equation='rc,c->')
nd('Sub',['r0','c0'],'D'); nd('Mul',['D','D'],'D2')
nd('Add',['r0','c0'],'rc0'); nd('Add',['rc0','c1'],'S'); nd('Mul',['S','S'],'S2')
nd('Sub',['r0','c1'],'r0m1'); nd('Add',['r0','c2'],'r0p2')
nd('Sub',['c0','c1'],'c0m1'); nd('Add',['c0','c2'],'c0p2')
for nm,val in [('ohRm1','r0m1'),('ohRp2','r0p2'),('ohCm1','c0m1'),('ohCp2','c0p2')]:
    nd('Equal',['ids10',val],nm+'_b'); nd('Cast',[nm+'_b'],nm,to=F)
nd('Einsum',['occ','ohRm1','ohCm1'],'gNW',equation='rc,r,c->')
nd('Einsum',['occ','ohRp2','ohCp2'],'gSE',equation='rc,r,c->')
nd('Einsum',['occ','ohRm1','ohCp2'],'gNE',equation='rc,r,c->')
nd('Einsum',['occ','ohRp2','ohCm1'],'gSW',equation='rc,r,c->')
ini(np.array([0,2,3],np.int64),'rax023')
nd('ReduceSum',['input','rax023'],'ccount',keepdims=0)  # [10]
nd('Greater',['ccount','zero'],'gt_b'); nd('Cast',['gt_b'],'gt',to=F)
nd('Mul',['gt','chmask'],'ev'); nd('Sub',['ev','e0'],'cd')  # [10]

# ---- row/col gate vectors ----
nd('Less',['ids30','r0'],'rU_b'); nd('Cast',['rU_b'],'rU',to=F)
nd('GreaterOrEqual',['ids30','r0p2'],'rDa_b'); nd('Cast',['rDa_b'],'rDa',to=F)
nd('LessOrEqual',['ids30','nine'],'le9_b'); nd('Cast',['le9_b'],'le9',to=F)
nd('Mul',['rDa','le9'],'rD')
nd('Less',['ids30','c0'],'cL_b'); nd('Cast',['cL_b'],'cL',to=F)
nd('GreaterOrEqual',['ids30','c0p2'],'cRa_b'); nd('Cast',['cRa_b'],'cRa',to=F)
nd('Mul',['cRa','le9'],'cR')
for src in ['rU','rD','ones30','cL','cR']:
    nd('Unsqueeze',[src,'ax0'],src+'_u')
nd('Concat',['rU_u','rD_u','rU_u','rD_u','ones30_u'],'RG',axis=0)   # [5,30]
nd('Concat',['cL_u','cR_u','cR_u','cL_u','ones30_u'],'CG',axis=0)   # [5,30]

# ---- diag-mix DM[5,3,3] ----
nd('Mul',['D','MaC1'],'Dm1'); nd('Mul',['D2','Cq'],'Dm2')
nd('Add',['MaC0','Dm1'],'Ma_a'); nd('Add',['Ma_a','Dm2'],'Ma')
nd('Mul',['S','MmC1'],'Sm1'); nd('Mul',['S2','Cq'],'Sm2')
nd('Add',['MmC0','Sm1'],'Mm_a'); nd('Add',['Mm_a','Sm2'],'Mm')
nd('Mul',['gNW','Ma'],'qNW'); nd('Mul',['gSE','Ma'],'qSE')
nd('Mul',['gNE','Mm'],'qNE'); nd('Mul',['gSW','Mm'],'qSW')
for src in ['qNW','qSE','qNE','qSW','delta00']:
    nd('Unsqueeze',[src,'ax0'],src+'_du')
nd('Concat',['qNW_du','qSE_du','qNE_du','qSW_du','delta00_du'],'DM',axis=0)  # [5,3,3]

# ---- Be[10,5] ----
nd('Unsqueeze',['cd','ax1'],'cd_col')                  # [10,1]
nd('Concat',['cd_col','cd_col','cd_col','cd_col','onescol'],'Be',axis=1)  # [10,5]

# ---- final free einsum ----
nd('Einsum',['input','P','P','RG','CG','DM','Ax','Be','Cxe'],'output',
   equation='bxrc,ir,jc,qr,qc,qij,xq,eq,xeq->berc')

g = helper.make_graph(nodes,'task190_regime2',
    [helper.make_tensor_value_info('input',F,[1,10,30,30])],
    [helper.make_tensor_value_info('output',F,[1,10,30,30])], initializer=inits)
m = helper.make_model(g, opset_imports=[helper.make_opsetid('',17)]); m.ir_version=8
onnx.save(m,'reports/candidates/task190/regime2.onnx')
r = evaluate('reports/candidates/task190/regime2.onnx', load_task(190))
print({k:r[k] for k in ('ok','pass','fail','memory','params','error')})
print('deployed 2525; concurrent 2377; this total', (r['memory'] or 0)+(r['params'] or 0))
