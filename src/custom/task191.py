"""task191 candidate — grader-counting-model rebuild.

Replaces the fp16 detect/threshold/stamp block (corrm 8464 + M 8464 + Yg 1058 +
boxsum 1058 + combk/stamp glue ~3.9k) with ONE multi-operand Einsum against the
FREE 'input' (walk-einsum mechanism):

  box[y,x] = sum_{o,i,j,g,h} rfill[o,g]*Q[g,i,y]*cfill[o,h]*Q[h,j,x]
             * prod_{c in 3x3 window} ( alpha_c[o]*CONST + beta_c[o]*Y[i-1+di, j-1+dj] )

Each window-cell factor is a product-of-sums pair: u_c[o,s] (s=0 const branch,
s=1 yellow branch) x W[s,ch] x RS_di[s,i,p] x RS_dj[s,j,q] x input[z,ch,p,q].
The s=0 branch reads input at the fixed on-grid cell (0,0) with all-ones channel
weights (== exactly 1), so don't-care cells never touch off-grid reads; the s=1
branch reads the yellow channel at the shifted cell (zero off-grid = incumbent's
pad-black semantics).  alpha = 1-K, beta = 2K-M (exact per-cell equality test:
match iff every footprint cell equals the oriented pattern).  The stamp
(oriented-bbox dilated by 1) folds into the same einsum via Q[g,i,y] =
[-2 <= y-i-g <= 0] contracted with the footprint runs rfill/cfill.
Everything is nonneg except beta=-1 entries; all values are small exact
integers in fp32, and the only consumer is Greater(box, 0).
Front-end (blue-frame bbox, K3, Kconv, Mconv) and epilogue are the incumbent's.
"""
import numpy as np
from onnx import TensorProto, helper
from ._exact import arr_b64, model, tensor

# ---- incumbent front-end initializer payloads (unchanged) ----
_B64 = {
 'y_s': 'k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDMsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=',
 'y_e': 'k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDMsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoFAAAAAAAAABcAAAAAAAAAFwAAAAAAAAA=',
 'y_ax': 'k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDMsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoBAAAAAAAAAAIAAAAAAAAAAwAAAAAAAAA=',
}

def build(task):
    f32, i64 = np.float32, np.int64

    # blue-channel row/col profile conv weights
    wrow = np.zeros((1, 10, 1, 30), dtype=f32); wrow[0, 1, 0, :] = 1.0
    wcol = np.zeros((1, 10, 30, 1), dtype=f32); wcol[0, 1, :, 0] = 1.0
    ramp = np.arange(30, dtype=f32)
    # 8 dihedral permutations of the flattened 3x3 (identical to incumbent 'perm')
    idx9 = np.arange(9).reshape(3, 3)
    perms = []
    for xpose in (0, 1):
        for rot in range(4):
            m = np.rot90(idx9, -rot) if not xpose else np.rot90(idx9.T, -rot)
            perms.append(m.reshape(9))
    # incumbent's exact perm ordering (decoded from live net):
    perm = np.array([0,1,2,3,4,5,6,7,8, 0,3,6,1,4,7,2,5,8, 2,5,8,1,4,7,0,3,6,
                     2,1,0,5,4,3,8,7,6, 8,7,6,5,4,3,2,1,0, 8,5,2,7,4,1,6,3,0,
                     6,3,0,7,4,1,8,5,2, 6,7,8,3,4,5,0,1,2], dtype=i64)

    tril = np.array([[1,0,0],[1,1,0],[1,1,1]], dtype=f32)
    triu = np.array([[1,1,1],[0,1,1],[0,0,1]], dtype=f32)

    # ---- mega-einsum constant operands ----
    # RS_d[s, i, p]: s=0 -> [p == 0] (const read); s=1 -> [p == i-1+d]
    RS = []
    for d in range(3):
        t = np.zeros((2, 23, 30), dtype=f32)
        t[0, :, 0] = 1.0
        for i in range(23):
            p = i - 1 + d
            if 0 <= p < 30:
                t[1, i, p] = 1.0
        RS.append(t)
    W2 = np.zeros((2, 10), dtype=f32); W2[0, :] = 1.0; W2[1, 4] = 1.0
    Q = np.zeros((3, 23, 23), dtype=f32)
    gg, ii, yy = np.meshgrid(np.arange(3), np.arange(23), np.arange(23), indexing='ij')
    Q[(yy - ii - gg >= -2) & (yy - ii - gg <= 0)] = 1.0

    inits = [
        tensor('y_s', arr_b64(_B64['y_s'])),
        tensor('y_e', arr_b64(_B64['y_e'])),
        tensor('y_ax', arr_b64(_B64['y_ax'])),
        tensor('wrow', wrow),
        tensor('wcol', wcol),
        tensor('zero1111b', np.zeros((1, 1, 1, 1), dtype=f32)),
        tensor('rampr', ramp.reshape(1, 1, 30, 1)),
        tensor('rampc', ramp.reshape(1, 1, 1, 30)),
        tensor('BIG', np.full((1, 1, 1, 1), 999.0, dtype=f32)),
        tensor('one', np.ones((1, 1, 1, 1), dtype=f32)),
        tensor('one11', np.ones((1, 1, 1), dtype=f32)),
        tensor('scalar1', np.array([1], dtype=i64)),
        tensor('off012', np.array([1.0, 2.0, 3.0], dtype=f32)),
        tensor('gmax', np.array([22.0], dtype=f32)),
        tensor('gzero', np.array([0.0], dtype=f32)),
        tensor('a3r', np.arange(3, dtype=f32).reshape(1, 1, 3, 1)),
        tensor('a3c', np.arange(3, dtype=f32).reshape(1, 1, 1, 3)),
        tensor('sh1111', np.array([1, 1, 1, 1], dtype=i64)),
        tensor('sh9', np.array([9], dtype=i64)),
        tensor('perm', perm),
        tensor('sh8133', np.array([8, 1, 3, 3], dtype=i64)),
        tensor('tril', tril),
        tensor('triu', triu),
        tensor('sh83', np.array([8, 3], dtype=i64)),
        tensor('zero', np.zeros((1, 1), dtype=f32)),
        tensor('sh8131', np.array([8, 1, 3, 1], dtype=i64)),
        tensor('sh8113', np.array([8, 1, 1, 3], dtype=i64)),
        tensor('sh829', np.array([8, 2, 9], dtype=i64)),
        tensor('RS0', RS[0]),
        tensor('RS1', RS[1]),
        tensor('RS2', RS[2]),
        tensor('W2', W2),
        tensor('Q', Q),
        tensor('c0u', np.zeros((1, 1, 1, 1), dtype=np.uint8)),
        tensor('c1u', np.ones((1, 1, 1, 1), dtype=np.uint8)),
        tensor('c4u', np.full((1, 1, 1, 1), 4, dtype=np.uint8)),
        tensor('pad99', np.array([0, 0, 0, 0, 0, 0, 7, 7], dtype=i64)),
        tensor('cv99', np.array([99], dtype=np.uint8)),
        tensor('arange_ch', np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)),
    ] + [tensor('ci%d' % c, np.asarray(c, dtype=i64)) for c in range(9)]

    # ---- front-end: blue-frame bbox -> K3 -> Kconv/Mconv/rfill/cfill (incumbent) ----
    nodes = [
        helper.make_node('Slice', ['input', 'y_s', 'y_e', 'y_ax'], ['Y']),
        helper.make_node('Conv', ['input', 'wrow'], ['rowcnt']),
        helper.make_node('Conv', ['input', 'wcol'], ['colcnt']),
        helper.make_node('Greater', ['rowcnt', 'zero1111b'], ['rowhas_b']),
        helper.make_node('Greater', ['colcnt', 'zero1111b'], ['colhas_b']),
        helper.make_node('Cast', ['rowhas_b'], ['rowhas'], to=1),
        helper.make_node('Cast', ['colhas_b'], ['colhas'], to=1),
        helper.make_node('Mul', ['rowhas', 'rampr'], ['rr_a']),
        helper.make_node('Sub', ['one', 'rowhas'], ['rr_n']),
        helper.make_node('Mul', ['rr_n', 'BIG'], ['rr_b']),
        helper.make_node('Add', ['rr_a', 'rr_b'], ['rr_v']),
        helper.make_node('ReduceMin', ['rr_v'], ['minr'], axes=[2], keepdims=0),
        helper.make_node('Mul', ['colhas', 'rampc'], ['cc_a']),
        helper.make_node('Sub', ['one', 'colhas'], ['cc_n']),
        helper.make_node('Mul', ['cc_n', 'BIG'], ['cc_b']),
        helper.make_node('Add', ['cc_a', 'cc_b'], ['cc_v']),
        helper.make_node('ReduceMin', ['cc_v'], ['minc'], axes=[3], keepdims=0),
        helper.make_node('ReduceMax', ['rr_a'], ['maxr'], axes=[2], keepdims=0),
        helper.make_node('ReduceMax', ['cc_a'], ['maxc'], axes=[3], keepdims=0),
        helper.make_node('Sub', ['maxr', 'minr'], ['tw_r']),
        helper.make_node('Sub', ['maxc', 'minc'], ['tw_c']),
        helper.make_node('Sub', ['tw_r', 'one11'], ['tall']),
        helper.make_node('Sub', ['tw_c', 'one11'], ['wide']),
        helper.make_node('Reshape', ['minr', 'scalar1'], ['minr_s']),
        helper.make_node('Reshape', ['minc', 'scalar1'], ['minc_s']),
        helper.make_node('Add', ['minr_s', 'off012'], ['ridx_f0']),
        helper.make_node('Add', ['minc_s', 'off012'], ['cidx_f0']),
        helper.make_node('Clip', ['ridx_f0', 'gzero', 'gmax'], ['ridx_f']),
        helper.make_node('Clip', ['cidx_f0', 'gzero', 'gmax'], ['cidx_f']),
        helper.make_node('Cast', ['ridx_f'], ['ridx'], to=6),
        helper.make_node('Cast', ['cidx_f'], ['cidx'], to=6),
        helper.make_node('Gather', ['Y', 'ridx'], ['Yr'], axis=2),
        helper.make_node('Gather', ['Yr', 'cidx'], ['K3raw'], axis=3),
        helper.make_node('Reshape', ['tall', 'sh1111'], ['tall4']),
        helper.make_node('Reshape', ['wide', 'sh1111'], ['wide4']),
        helper.make_node('Less', ['a3r', 'tall4'], ['rkeep_b']),
        helper.make_node('Less', ['a3c', 'wide4'], ['ckeep_b']),
        helper.make_node('Cast', ['rkeep_b'], ['rkeep'], to=1),
        helper.make_node('Cast', ['ckeep_b'], ['ckeep'], to=1),
        helper.make_node('Mul', ['K3raw', 'rkeep'], ['K3a']),
        helper.make_node('Mul', ['K3a', 'ckeep'], ['K3']),
        helper.make_node('Reshape', ['K3', 'sh9'], ['k3flat']),
        helper.make_node('Gather', ['k3flat', 'perm'], ['ko_flat'], axis=0),
        helper.make_node('Reshape', ['ko_flat', 'sh8133'], ['Kconv']),
        helper.make_node('ReduceMax', ['Kconv'], ['rh3'], axes=[3], keepdims=1),
        helper.make_node('ReduceMax', ['Kconv'], ['ch3'], axes=[2], keepdims=1),
        helper.make_node('Reshape', ['rh3', 'sh83'], ['rh2']),
        helper.make_node('MatMul', ['rh2', 'tril'], ['rpre']),
        helper.make_node('MatMul', ['rh2', 'triu'], ['rsuf']),
        helper.make_node('Greater', ['rpre', 'zero'], ['rpre_b']),
        helper.make_node('Greater', ['rsuf', 'zero'], ['rsuf_b']),
        helper.make_node('And', ['rpre_b', 'rsuf_b'], ['rfill_b']),
        helper.make_node('Cast', ['rfill_b'], ['rfill'], to=1),
        helper.make_node('Reshape', ['rfill', 'sh8131'], ['rfill4']),
        helper.make_node('Reshape', ['ch3', 'sh83'], ['ch2']),
        helper.make_node('MatMul', ['ch2', 'tril'], ['cpre']),
        helper.make_node('MatMul', ['ch2', 'triu'], ['csuf']),
        helper.make_node('Greater', ['cpre', 'zero'], ['cpre_b']),
        helper.make_node('Greater', ['csuf', 'zero'], ['csuf_b']),
        helper.make_node('And', ['cpre_b', 'csuf_b'], ['cfill_b']),
        helper.make_node('Cast', ['cfill_b'], ['cfill'], to=1),
        helper.make_node('Reshape', ['cfill', 'sh8113'], ['cfill4']),
        helper.make_node('Mul', ['rfill4', 'cfill4'], ['Mconv']),
        # ---- per-cell coefficients: alpha = 1-K, beta = 2K-M ----
        helper.make_node('Sub', ['one', 'Kconv'], ['alpha']),
        helper.make_node('Add', ['Kconv', 'Kconv'], ['kk2']),
        helper.make_node('Sub', ['kk2', 'Mconv'], ['beta']),
        helper.make_node('Concat', ['alpha', 'beta'], ['ucat'], axis=1),
        helper.make_node('Reshape', ['ucat', 'sh829'], ['uflat']),
    ]
    for c in range(9):
        nodes.append(helper.make_node('Gather', ['uflat', 'ci%d' % c], ['u%d' % c], axis=2))

    # ---- mega-einsum ----
    reserved = 'yxoijghz'
    pool = [ch for ch in 'abcdefklmnpqrstuvwABCDEFGHIJKLMNOPQRSTUVWXYZ' if ch not in reserved]
    terms, feeds = [], []
    for c in range(9):
        di, dj = divmod(c, 3)
        s, chl, p, q = pool.pop(0), pool.pop(0), pool.pop(0), pool.pop(0)
        # order chosen so ORT's left-to-right pairwise contraction keeps temps small
        terms += [s + 'j' + q,
                  'z' + chl + p + q,
                  s + 'i' + p,
                  s + chl,
                  'o' + s]
        feeds += ['RS%d' % dj, 'input', 'RS%d' % di, 'W2', 'u%d' % c]
    terms += ['og', 'giy', 'oh', 'hjx']
    feeds += ['rfill', 'Q', 'cfill', 'Q']
    eq = ','.join(terms) + '->yx'
    nodes.append(helper.make_node('Einsum', feeds, ['box'], equation=eq))

    # ---- epilogue (incumbent) ----
    nodes += [
        helper.make_node('Greater', ['box', 'zero1111b'], ['box23']),
        helper.make_node('Greater', ['Y', 'zero1111b'], ['yel23']),
        helper.make_node('Where', ['box23', 'c1u', 'c0u'], ['idx_bb']),
        helper.make_node('Where', ['yel23', 'c4u', 'idx_bb'], ['colidx23']),
        helper.make_node('Pad', ['colidx23', 'pad99', 'cv99'], ['colidx'], mode='constant'),
        helper.make_node('Equal', ['colidx', 'arange_ch'], ['output']),
    ]
    return model('task191_walk_einsum', nodes, inits, output_dtype=TensorProto.BOOL, opset=17)
