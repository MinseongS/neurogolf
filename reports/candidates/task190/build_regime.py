"""task190 REGIME-CRACK: fold the 900B ray30 Pad-plane + Where epilogue into ONE
free-output Einsum. Rule: 2x2 block + 1..4 diagonal corner satellites; each
satellite spawns an outward diagonal ray to the grid edge, painted the grid color.

Mechanism:
- R (block top row) / C: pair-count moment einsums (nz(u,x)&nz(u+1,x) = 2*delta(u=R)).
- Corner gates: diag/anti-diag pair-count moments M0,M1 -> linear solve
  (2gNW = M0(1+R)-1-M1, 2gSE = M0(1-R)+M1-1; anti family for NE/SW).
- Per-axis side distances V_p[u] = Relu(+-(u-R)+off) stacked [1,2,10]; ray
  polynomial F6(V,X) = VX(1-(V-X)^2) (>0 iff V=X>=1, 0 if V=0 or X=0 -> quadrant
  disjoint, no cross-corner interference). Powers via ONE Pow -> [5,2,10].
- Color: MaxPool presence - 2*e0 (C[0]=-1, C[v]=+1, absent 0).
- Final 11-operand Einsum: identity (k=0, eye mixer) + paint (k=1, delta_d0
  bg-gate mixer) with [10,30] eye-slab bridges; output = FREE.
"""
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

F = TensorProto.FLOAT


def init(name, arr):
    return numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name)


inits = []
nodes = []

# ---------- static initializers ----------
N = 10
B = np.zeros((10, 30), np.float32)          # bridge delta(y=u)
for u in range(10):
    B[u, u] = 1.0
B1 = np.zeros((10, 30), np.float32)         # shift delta(z=u+1)
for u in range(10):
    B1[u, u + 1] = 1.0
inits += [init('B', B), init('B1', B1)]

nzk = np.ones(10, np.float32); nzk[0] = 0.0            # colored-channel mask
idshalf = (np.arange(10) / 2.0).astype(np.float32)     # u/2 (pair-count is 2*delta)
ids10 = np.arange(10).astype(np.float32)
inits += [init('nzk', nzk), init('idshalf', idshalf), init('ids10', ids10)]

E2 = np.zeros((2, 10, 10), np.float32)
E2[0] = np.eye(10)                                     # identity mixer
E2[1, 0, :] = 1.0                                      # bg-gate delta_d0
inits += [init('E2', E2)]

UNF = np.zeros((4, 2, 2), np.float32)                  # P -> (p,q)
for P in range(4):
    UNF[P, P // 2, P % 2] = 1.0
inits += [init('UNF', UNF)]

KJ = np.array([[1, 0, 0, 0, 0],
               [0, .5, -.5, 1, -.5]], np.float32)      # F6 coefs /2 (gates are 2g)
inits += [init('KJ', KJ)]

# SGI2[p,u] = sign_p*(-u) + off_p : row0 = -u-1 (V0=R-1-u after +R), row1 = u-2 (V1=u-R-2 after -R)
SGI2 = np.stack([-ids10 - 1.0, ids10 - 2.0])[None]     # [1,2,10]
sgnpair = np.array([1.0, -1.0], np.float32).reshape(1, 2, 1)
inits += [init('SGI2', SGI2), init('sgnpair', sgnpair)]

expsR = np.array([0, 1, 3, 2, 1], np.float32).reshape(5, 1, 1)
expsC = np.array([0, 1, 1, 2, 3], np.float32).reshape(5, 1, 1)
inits += [init('expsR', expsR), init('expsC', expsC)]

inits += [init('one', np.array([1.0], np.float32))]
twoe0 = np.zeros((1, 10, 1, 1), np.float32); twoe0[0, 0] = 2.0
oneskc = np.ones((1, 10, 1, 1), np.float32)
inits += [init('twoe0', twoe0), init('oneskc', oneskc)]
idrow = np.array([[1, 0, 0, 0]], np.float32)
inits += [init('idrow', idrow)]
rshape = np.array([1, 4], np.int64)
inits.append(numpy_helper.from_array(rshape, 'rshape'))


def n(op, ins, outs, **kw):
    nodes.append(helper.make_node(op, ins, outs, **kw))


# ---------- R, C (block top-left) ----------
# q[u] = sum_x nz(u,x)*nz(u+1,x) = 2*delta(u=R);  R = sum idshalf[u]*q[u]
n('Einsum', ['input', 'input', 'nzk', 'nzk', 'B', 'B1', 'idshalf'], ['R'],
  equation='bkyx,bmzx,k,m,uy,uz,u->b')
n('Einsum', ['input', 'input', 'nzk', 'nzk', 'B', 'B1', 'idshalf'], ['Cc'],
  equation='bkyx,bmyw,k,m,ux,uw,u->b')

# ---------- corner gates via diag / anti-diag pair moments ----------
# dg[u] = sum_t nz(u,t)nz(u+1,t+1):  gNW@R-1, 1@R, gSE@R+1
n('Einsum', ['input', 'input', 'nzk', 'nzk', 'B', 'B1', 'B', 'B1'], ['M0d'],
  equation='bkyx,bmzw,k,m,uy,uz,tx,tw->b')
n('Einsum', ['input', 'input', 'nzk', 'nzk', 'B', 'B1', 'B', 'B1', 'ids10'], ['M1d'],
  equation='bkyx,bmzw,k,m,uy,uz,tx,tw,u->b')
# ag[u] = sum_s nz(u,s+1)nz(u+1,s):  gNE@R-1, 1@R, gSW@R+1
n('Einsum', ['input', 'input', 'nzk', 'nzk', 'B', 'B1', 'B1', 'B'], ['M0a'],
  equation='bkyx,bmzw,k,m,uy,uz,sx,sw->b')
n('Einsum', ['input', 'input', 'nzk', 'nzk', 'B', 'B1', 'B1', 'B', 'ids10'], ['M1a'],
  equation='bkyx,bmzw,k,m,uy,uz,sx,sw,u->b')

# 2g solve:  2gNW = M0(1+R)-1-M1 ; 2gSE = M0(1-R)+M1-1  (same for anti: NE/SW)
n('Add', ['R', 'one'], ['Rp1'])
n('Sub', ['one', 'R'], ['Rm1n'])
for fam, M0, M1, gA, gB in (('d', 'M0d', 'M1d', 'gNW2', 'gSE2'),
                            ('a', 'M0a', 'M1a', 'gNE2', 'gSW2')):
    n('Mul', [M0, 'Rp1'], [f'u1{fam}'])
    n('Add', [M1, 'one'], [f'v1{fam}'])
    n('Sub', [f'u1{fam}', f'v1{fam}'], [gA])
    n('Mul', [M0, 'Rm1n'], [f'u2{fam}'])
    n('Sub', [M1, 'one'], [f'v2{fam}'])
    n('Add', [f'u2{fam}', f'v2{fam}'], [gB])

n('Concat', ['gNW2', 'gNE2', 'gSW2', 'gSE2'], ['gcat'], axis=0)   # [4]
n('Reshape', ['gcat', 'rshape'], ['grow'])                        # [1,4]
n('Concat', ['idrow', 'grow'], ['G2'], axis=0)                    # [2,4]

# ---------- color vector ----------
n('MaxPool', ['input'], ['pooled'], kernel_shape=[30, 30])        # [1,10,1,1]
n('Sub', ['pooled', 'twoe0'], ['Cvec'])                           # pres - 2*e0
n('Concat', ['oneskc', 'Cvec'], ['SVc'], axis=0)                  # [2,10,1,1]

# ---------- per-axis side-distance stacks ----------
for ax, Rname in (('r', 'R'), ('c', 'Cc')):
    n('Mul', ['sgnpair', Rname], [f'sm{ax}'])       # [1,2,1]
    n('Add', ['SGI2', f'sm{ax}'], [f'D{ax}'])       # [1,2,10]
    n('Relu', [f'D{ax}'], [f'V{ax}'])               # [1,2,10]
exp_of = {'r': 'expsR', 'c': 'expsC'}
n('Pow', ['Vr', 'expsR'], ['RS'])                   # [5,2,10]
n('Pow', ['Vc', 'expsC'], ['CS'])                   # [5,2,10]

# ---------- final FREE einsum ----------
n('Einsum', ['input', 'E2', 'SVc', 'G2', 'UNF', 'RS', 'B', 'CS', 'B', 'KJ'],
  ['output'],
  equation='bdyx,kdc,kcef,ka,apq,jpu,uy,jqw,wx,kj->bcyx')

graph = helper.make_graph(
    nodes, 'task190_regime',
    [helper.make_tensor_value_info('input', F, [1, 10, 30, 30])],
    [helper.make_tensor_value_info('output', F, [1, 10, 30, 30])],
    initializer=inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
model.ir_version = 8
onnx.save(model, '/Users/minseong/project/neurogolf/reports/candidates/task190/regime.onnx')
print('saved')
