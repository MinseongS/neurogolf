"""task084 candidate — full free-output Einsum (signed-channel, mechanism 15 + epilogue fold).

Goal of this file: EMPIRICALLY price the dossier's "route the overlay through a
free einsum" idea and expose why it cannot beat the incumbent's scatter (1817B).

Single-einsum form (epilogue fold, playbook 13a): every term carries an `input`
factor, so off-canvas cells self-silence and fill cells (background => w=0) pick
the w=0 row of the mixer:

    output[b,v,r,c] = sum_{w,s} input[b,w,r,c] * A[s,r,c] * M[s,w,v]
      s=0 pass  : A[0]=ones,  M[0]=Id            -> copies input colour (col0=k, bg=0)
      s=1 red   : A[1]=D,      M[1,0,:]=e2 - e0   -> +red, -bg on the anti-diagonal
      s=2 yellow: A[2]=B,      M[2,0,:]=e4 - e0   -> +yellow, -bg on the bottom row

FORCED COST: the einsum's operands must share the FREE input's fp32 dtype (casting
input to fp16 would itself materialise an 18000B counted plane).  A[1]=D and A[2]=B
are A-dependent, so A must be a COUNTED node output [3,30,30] fp32 = 10800B, plus the
[30,30] transients that build D/B.  That is why the free-einsum route is far MORE
expensive than the incumbent scatter, which encodes the anti-diagonal in a compact
[.,.,.,21] index plane via row = last_row - c arithmetic (no [30,30] carrier).
"""
import numpy as np
from onnx import TensorProto, helper
from ._exact import model, tensor

F = TensorProto.FLOAT


def build(task):
    # ---- mixer M[s,w,v] : [3,10,10] fp32 ----
    Id = np.eye(10, dtype=np.float32)
    red = np.zeros((10, 10), np.float32); red[0, 2] = 1.0; red[0, 0] = -1.0   # e2 - e0
    yel = np.zeros((10, 10), np.float32); yel[0, 4] = 1.0; yel[0, 0] = -1.0   # e4 - e0
    M = np.stack([Id, red, yel]).astype(np.float32)              # [3,10,10]

    coords_r = np.arange(30, dtype=np.float32).reshape(30, 1)
    coords_c = np.arange(30, dtype=np.float32).reshape(1, 30)
    cge1 = (np.arange(30) >= 1).astype(np.float32).reshape(1, 30)   # column>=1 mask (fixed)
    ones_plane = np.ones((1, 30, 30), np.float32)

    inits = [
        tensor('M_mix', M),
        tensor('coords_r', coords_r),
        tensor('coords_c', coords_c),
        tensor('cge1', cge1),
        tensor('ones_plane', ones_plane),
        tensor('total_axes', np.array([0, 1, 2, 3], np.int64)),
        tensor('one_f', np.float32(1.0)),
        tensor('half', np.float32(0.5)),
        tensor('shp1', np.array([1, 30, 30], np.int64)),
    ]

    nodes = [
        # recover last_row = A-1 = sqrt(sum(input)) - 1   (sum = A^2)
        helper.make_node('ReduceSum', ['input', 'total_axes'], ['tsum'], keepdims=0),
        helper.make_node('Sqrt', ['tsum'], ['A_f']),
        helper.make_node('Sub', ['A_f', 'one_f'], ['lastrow']),           # scalar A-1

        # D = (r + c == lastrow) & (c >= 1)
        helper.make_node('Add', ['coords_r', 'coords_c'], ['rc']),        # [30,30]
        helper.make_node('Sub', ['rc', 'lastrow'], ['rc_off']),
        helper.make_node('Mul', ['rc_off', 'rc_off'], ['rc_off2']),
        helper.make_node('Less', ['rc_off2', 'half'], ['D_eq']),          # bool [30,30]
        helper.make_node('Cast', ['D_eq'], ['D_eqf'], to=1),
        helper.make_node('Mul', ['D_eqf', 'cge1'], ['D']),                # [30,30]

        # B = (r == lastrow) & (1 <= c <= lastrow)
        helper.make_node('Sub', ['coords_r', 'lastrow'], ['r_off']),      # [30,1]
        helper.make_node('Mul', ['r_off', 'r_off'], ['r_off2']),
        helper.make_node('Less', ['r_off2', 'half'], ['r_eq']),           # bool [30,1]
        helper.make_node('Cast', ['r_eq'], ['r_eqf'], to=1),
        helper.make_node('LessOrEqual', ['coords_c', 'lastrow'], ['cle']),# bool [1,30] (c<=A-1)
        helper.make_node('Cast', ['cle'], ['clef'], to=1),
        helper.make_node('Mul', ['clef', 'cge1'], ['cband']),             # [1,30]
        helper.make_node('Mul', ['r_eqf', 'cband'], ['B']),               # [30,30]

        # A = stack(ones, D, B) -> [3,30,30]
        helper.make_node('Reshape', ['D', 'shp1'], ['D1']),
        helper.make_node('Reshape', ['B', 'shp1'], ['B1']),
        helper.make_node('Concat', ['ones_plane', 'D1', 'B1'], ['A_op'], axis=0),  # [3,30,30]

        # free final einsum -> graph output
        helper.make_node('Einsum', ['input', 'A_op', 'M_mix'], ['output'],
                         equation='bwrc,src,swv->bvrc'),
    ]

    return model('task084_einsum_full', nodes, inits, output_dtype=F, opset=13)
