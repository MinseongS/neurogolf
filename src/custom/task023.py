"""Task 023 candidate — 150deff5: decompose gray blobs into 2x2 boxes (cyan 8)
and 3-sticks (red 2).

Mechanism (counting-model rebuild, replaces the 5-round / 56-QLinearConv
unrolled unique-cover peel of the incumbent):
  * 8x8 u8 window (rows 0:8, cols 1:9) — same as incumbent.
  * 3 full peel rounds with a TYPE-AGNOSTIC unique-coverage cell plane
    (funi = (coverage count == 1)); the unique tile's type is implied by which
    detector fires, so one Equal replaces the incumbent's two.
  * Confirmation fused into one QLinearConv per tile type via the threshold
    trick: conv over (5*u + funi) with i32 bias (-20 for 2x2 / -15 for 3-taps)
    saturates to "anchor complete AND contains a unique-cover cell" — kills the
    incumbent's per-type sel/Mul chains and the Div nodes (i32-bias fold).
    The all-type scatter kernel Wnt doubles as coverage counter AND removal
    scatter (dec), so no separate Ladd path exists.
  * Final square-only confirm round + NEW residual majority rule on still
    ambiguous cells: box iff sat_u8(2*nSq - nL) >= 1, computed by a single
    QLinearConv with weight zero_point 128 (signed weights).  This residual
    resolves most fixpoint-ambiguous blobs the incumbent gives up on:
    bit-exact sim fresh fail 2.0% vs incumbent ~6% (8000 fresh), stored 266/266.
"""
import numpy as np
from onnx import TensorProto, helper
from ._exact import model, tensor


def _k(entries, shape):
    w = np.zeros(shape, dtype=np.uint8)
    for idx, v in entries:
        w[idx] = v
    return w


def build(task):
    # 3x3 detect kernels (anchor at top-left of footprint), pads [0,0,2,2]
    # channel 0: 2x2 square; 1: 1x3 horizontal; 2: 3x1 vertical
    wdet = np.zeros((3, 1, 3, 3), dtype=np.uint8)
    wdet[0, 0, 0:2, 0:2] = 1
    wdet[1, 0, 0, 0:3] = 1
    wdet[2, 0, 0:3, 0] = 1
    # scatter (coverage) kernels, pads [2,2,0,0]: taps at offsets -2..0
    # nT = sum of all three coverages: input is the 3-ch anchor tensor
    wnt = np.zeros((1, 3, 3, 3), dtype=np.uint8)
    wnt[0, 0, 1:3, 1:3] = 1   # square coverage: offsets -1..0
    wnt[0, 1, 2, 0:3] = 1     # horizontal: offsets (0, -2..0)
    wnt[0, 2, 0:3, 2] = 1     # vertical: offsets (-2..0, 0)
    # Sadd: scatter of confirmed square channel only
    wsadd = np.zeros((1, 3, 3, 3), dtype=np.uint8)
    wsadd[0, 0, 1:3, 1:3] = 1
    # residual: sat(2*nSq - nL) with weight zero point 128
    wres = np.full((1, 3, 3, 3), 128, dtype=np.uint8)
    wres[0, 0, 1:3, 1:3] = 130          # +2 per square anchor
    wres[0, 1, 2, 0:3] = 127            # -1 per h anchor
    wres[0, 2, 0:3, 2] = 127            # -1 per v anchor

    inits = [
        tensor('SC', np.asarray(np.float32(1.0))),
        tensor('ZP', np.asarray(np.uint8(0))),
        tensor('ZW', np.asarray(np.uint8(128))),   # weight zero point for wres
        tensor('Wdet', wdet),
        tensor('Bdet', np.array([-3, -2, -2], dtype=np.int32)),
        tensor('Wnt', wnt),
        tensor('Bconf', np.array([-20, -15, -15], dtype=np.int32)),
        tensor('Wsadd', wsadd),
        tensor('Wres', wres),
        tensor('ONE', np.asarray(np.uint8(1))),
        tensor('FIVE', np.asarray(np.uint8(5))),
        # slices
        tensor('ns', np.array([5, 0, 1], dtype=np.int64)),
        tensor('ne', np.array([6, 8, 9], dtype=np.int64)),
        tensor('na', np.array([1, 2, 3], dtype=np.int64)),
        tensor('bs', np.array([0, 0, 0], dtype=np.int64)),
        tensor('be', np.array([1, 9, 11], dtype=np.int64)),
        # pads: 8x8 window -> 9x11 (rows 0..7 -> pad bottom 1; cols 1..8 -> pad l1 r2)
        tensor('spad', np.array([0, 0, 0, 1, 0, 0, 1, 2], dtype=np.int64)),
        tensor('zc', np.zeros((1, 1, 9, 11), dtype=np.uint8)),
        tensor('pads', np.array([0, 0, 0, 0, 0, 0, 21, 19], dtype=np.int64)),
    ]

    DET = dict(kernel_shape=[3, 3], pads=[0, 0, 2, 2])
    SCT = dict(kernel_shape=[3, 3], pads=[2, 2, 0, 0])

    def qlc(x, w, y, bias=None, attrs=DET, zw='ZP'):
        ins = [x, 'SC', 'ZP', w, 'SC', zw, 'SC', 'ZP']
        if bias is not None:
            ins.append(bias)
        return helper.make_node('QLinearConv', ins, [y], **attrs)

    nodes = [
        helper.make_node('Slice', ['input', 'bs', 'be', 'na'], ['blk_f']),
        helper.make_node('Cast', ['blk_f'], ['blk'], to=2),
        helper.make_node('Slice', ['input', 'ns', 'ne', 'na'], ['nz_f']),
        helper.make_node('Cast', ['nz_f'], ['u0'], to=2),
    ]

    S_prev = None
    u = 'u0'
    for r in range(3):
        sfx = str(r)
        nodes += [
            qlc(u, 'Wdet', 'A' + sfx, bias='Bdet'),
            qlc('A' + sfx, 'Wnt', 'nT' + sfx, attrs=SCT),
            helper.make_node('Equal', ['nT' + sfx, 'ONE'], ['eq' + sfx]),
            helper.make_node('Cast', ['eq' + sfx], ['funi' + sfx], to=2),
            helper.make_node('Mul', [u, 'FIVE'], ['u5' + sfx]),
            helper.make_node('Add', ['u5' + sfx, 'funi' + sfx], ['in' + sfx]),
            qlc('in' + sfx, 'Wdet', 'C' + sfx, bias='Bconf'),
            qlc('C' + sfx, 'Wsadd', 'Sadd' + sfx, attrs=SCT),
            qlc('C' + sfx, 'Wnt', 'dec' + sfx, attrs=SCT),
            helper.make_node('Min', ['dec' + sfx, 'ONE'], ['m' + sfx]),
            helper.make_node('Sub', [u, 'm' + sfx], ['u' + str(r + 1)]),
        ]
        if S_prev is None:
            S_prev = 'Sadd' + sfx
        else:
            nodes.append(helper.make_node('Add', [S_prev, 'Sadd' + sfx], ['S' + sfx]))
            S_prev = 'S' + sfx
        u = 'u' + str(r + 1)

    # final round: square-only confirm + signed-weight majority residual
    nodes += [
        qlc(u, 'Wdet', 'AF', bias='Bdet'),
        qlc('AF', 'Wnt', 'nTF', attrs=SCT),
        helper.make_node('Equal', ['nTF', 'ONE'], ['eqF']),
        helper.make_node('Cast', ['eqF'], ['funiF'], to=2),
        helper.make_node('Mul', [u, 'FIVE'], ['u5F']),
        helper.make_node('Add', ['u5F', 'funiF'], ['inF']),
        qlc('inF', 'Wdet', 'CF', bias='Bconf'),
        qlc('CF', 'Wsadd', 'SaddF', attrs=SCT),
        helper.make_node('Add', [S_prev, 'SaddF'], ['SF']),
        qlc('AF', 'Wres', 'Draw', attrs=SCT, zw='ZW'),
        helper.make_node('Mul', ['Draw', u], ['resid']),
        helper.make_node('Add', ['SF', 'resid'], ['SR']),
        # epilogue
        helper.make_node('Min', ['SR', 'ONE'], ['Smin']),
        helper.make_node('Sub', ['u0', 'Smin'], ['Lm']),
        helper.make_node('Pad', ['Smin', 'spad'], ['Sg'], mode='constant'),
        helper.make_node('Pad', ['Lm', 'spad'], ['Lg'], mode='constant'),
        helper.make_node('Concat', ['blk', 'zc', 'Lg', 'zc', 'zc', 'zc', 'zc', 'zc', 'Sg', 'zc'],
                         ['small'], axis=1),
        helper.make_node('Pad', ['small', 'pads'], ['output'], mode='constant'),
    ]
    return model('task023_peel_resid', nodes, inits, output_dtype=2, opset=17)
