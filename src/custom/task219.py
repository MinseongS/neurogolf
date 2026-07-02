"""task219 candidate — batched-band + einsum-collapsed rebuild.

Same algorithm as incumbent (band detect -> band-0 pattern -> per-later-band
A-anchor alignment + best-column-shift match -> place shifted band-0 pattern,
blue = placed & empty), but:
  * the 5 unrolled per-band blocks become ONE batched block with a K-band axis
  * per-band Pad[21,10]/Gather[15,10]/Mul/Max placement+accumulation collapse
    into ONE Einsum 'kjr,ks,jsc,k->rc' producing the [15,10] blue mask directly
  * shift variants built by one Gather with an initializer index table
    (indices = params, not memory)
  * epilogue: colour = Max(8*cyan, blue) in uint8
K = 6 later bands (incumbent only handled 5; tall=1 grids can have 7 bands)
S = 5 shifts {+2,+1,0,-1,-2} (incumbent had 4; -2 occurs when col0=2*awide, colk=awide)
Priority order of the first 4 shifts matches incumbent's ArgMax tie-break.
"""
import numpy as np
from onnx import TensorProto, helper
from ._exact import model, tensor

K = 6   # later bands handled (band indices 2..K+1)
S = 4   # column-shift candidates, order [+2,+1,0,-1] (incumbent parity; a 5th
        # shift -2 was tried and REJECTED: left-shifts pull band-0's dense C
        # region over the A|B scoring window and win spuriously -> worse fresh)

F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
B = TensorProto.BOOL


def build(task):
    offs = [0, 1, 2, 3, 4][:S]           # B0p col index offset per shift (pad-left=2)
    ci = np.array([[c + o for c in range(10)] for o in offs], dtype=np.int64)  # [S,10]
    inits = [
        tensor('t219_st4', np.array([0, 8, 0, 0], np.int64)),
        tensor('t219_en4', np.array([1, 9, 15, 10], np.int64)),
        tensor('t219_pads_cy', np.array([0, 0, 0, 0, 0, 0, 3, 0], np.int64)),
        tensor('t219_pads_b0', np.array([0, 2, 0, 2], np.int64)),
        tensor('t219_pads_out', np.array([0, 0, 0, 0, 0, 0, 15, 20], np.int64)),
        tensor('t219_hi255', np.uint8(255)),
        tensor('t219_eight', np.uint8(8)),
        tensor('t219_zero1', np.zeros(1, np.float16)),
        tensor('t219_onef', np.float16(1.0)),
        tensor('t219_one1', np.ones(1, np.float16)),
        tensor('t219_cax', np.array(0, np.int64)),
        tensor('t219_i0', np.array([0], np.int64)),
        tensor('t219_i1', np.array([1], np.int64)),
        tensor('t219_i2', np.array([2], np.int64)),
        tensor('t219_i3', np.array([3], np.int64)),
        tensor('t219_i14', np.array([14], np.int64)),
        tensor('t219_i15', np.array([15], np.int64)),
        tensor('t219_iK', np.array([K], np.int64)),
        tensor('t219_k012', np.array([0, 1, 2], np.int64)),
        tensor('t219_r18', np.array([18], np.int64)),
        tensor('t219_r310', np.array([3, 10], np.int64)),
        tensor('t219_rm4', np.array([1, 1, 15, 10], np.int64)),
        tensor('t219_kvals', np.arange(2, 2 + K, dtype=np.float16).reshape(K, 1)),
        tensor('t219_onesK', np.ones((K, 1), np.float16)),
        tensor('t219_arS', np.arange(S, dtype=np.int64)),
        tensor('t219_ar15', np.arange(15, dtype=np.int64)),
        tensor('t219_ci', ci),
        tensor('t219_ar10', np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)),
    ]
    n = helper.make_node
    nodes = [
        # ---- head: cyan plane, padded, row occupancy, band tops, band index
        n('Slice', ['input', 't219_st4', 't219_en4'], ['cy4']),                    # [1,1,15,10] f32
        n('Cast', ['cy4'], ['cyh'], to=F16),                                        # [1,1,15,10] f16
        n('Pad', ['cyh', 't219_pads_cy'], ['cyp'], mode='constant'),                # [1,1,18,10]
        n('ReduceMax', ['cyp'], ['rocc'], axes=[3], keepdims=0),                    # [1,1,18]
        n('Reshape', ['rocc', 't219_r18'], ['occ18']),                              # [18]
        n('Slice', ['occ18', 't219_i0', 't219_i15'], ['occ15']),                    # [15]
        n('Slice', ['occ15', 't219_i0', 't219_i14'], ['p14']),                      # [14]
        n('Concat', ['t219_zero1', 'p14'], ['prev'], axis=0),                       # [15]
        n('Sub', ['t219_onef', 'prev'], ['inv']),                                   # [15]
        n('Mul', ['occ15', 'inv'], ['btop']),                                       # [15] band-top ind.
        n('CumSum', ['btop', 't219_cax'], ['bidx']),                                # [15] band index 1..
        # ---- band 0 window, validity, masked pattern B0
        n('ArgMax', ['btop'], ['top1'], axis=0, keepdims=0),                        # []
        n('Unsqueeze', ['top1', 't219_i0'], ['top1u']),                             # [1]
        n('Add', ['top1u', 't219_k012'], ['rows0']),                                # [3]
        n('Gather', ['cyp', 'rows0'], ['g0'], axis=2),                              # [1,1,3,10]
        n('Reshape', ['g0', 't219_r310'], ['g0r']),                                 # [3,10]
        n('Gather', ['occ18', 'rows0'], ['occ0'], axis=0),                          # [3]
        n('Slice', ['occ0', 't219_i1', 't219_i2'], ['o01']),                        # [1]
        n('Slice', ['occ0', 't219_i2', 't219_i3'], ['o02']),                        # [1]
        n('Min', ['o01', 'o02'], ['o0m']),                                          # [1]
        n('Concat', ['t219_one1', 'o01', 'o0m'], ['v0'], axis=0),                   # [3]
        n('Unsqueeze', ['v0', 't219_i1'], ['v0u']),                                 # [3,1]
        n('Mul', ['g0r', 'v0u'], ['B0']),                                           # [3,10]
        # ---- anchor column + band-0 anchor row
        n('ReduceMax', ['B0'], ['cocc'], axes=[0], keepdims=0),                     # [10]
        n('ArgMax', ['cocc'], ['c0'], axis=0, keepdims=0),                          # []
        n('Unsqueeze', ['c0', 't219_i0'], ['c0u']),                                 # [1]
        n('Gather', ['B0', 'c0u'], ['colB'], axis=1),                               # [3,1]
        n('Squeeze', ['colB', 't219_i1'], ['colBs']),                               # [3]
        n('ArgMax', ['colBs'], ['a0'], axis=0, keepdims=0),                         # []
        # ---- shift variants of B0 via pad + index-table gather
        n('Pad', ['B0', 't219_pads_b0'], ['B0p'], mode='constant'),                 # [3,14]
        n('Gather', ['B0p', 't219_ci'], ['shifts'], axis=1),                        # [3,S,10]
        # ---- batched later bands: exists, top, occupancy, validity
        n('Equal', ['t219_kvals', 'bidx'], ['kone']),                               # [K,15] bool
        n('Cast', ['kone'], ['kf'], to=F16),                                        # [K,15]
        n('Mul', ['kf', 'btop'], ['bt5']),                                          # [K,15]
        n('ReduceMax', ['bt5'], ['ex5'], axes=[1], keepdims=0),                     # [K]
        n('ArgMax', ['bt5'], ['top5'], axis=1, keepdims=0),                         # [K]
        n('Unsqueeze', ['top5', 't219_i1'], ['top5u']),                             # [K,1]
        n('Add', ['top5u', 't219_k012'], ['rows5']),                                # [K,3]
        n('Gather', ['occ18', 'rows5'], ['occk'], axis=0),                          # [K,3]
        n('Slice', ['occk', 't219_i1', 't219_i2', 't219_i1'], ['ok1']),             # [K,1]
        n('Slice', ['occk', 't219_i2', 't219_i3', 't219_i1'], ['ok2']),             # [K,1]
        # incumbent parity: validity slot 1 uses BAND-0's occ[1] for the k=2 row
        # (its con_171 = [1, band0_occ1, min(band0_occ1, band2_occ2)]) but the
        # band's OWN occ[1] for k>=3 rows (its con_247 pattern). Load-bearing
        # for gapped bands (arows+brows = {0,2}).
        n('Unsqueeze', ['o01', 't219_i0'], ['o01u']),                               # [1,1]
        n('Slice', ['ok1', 't219_i1', 't219_iK', 't219_i0'], ['ok1t']),             # [K-1,1]
        n('Concat', ['o01u', 'ok1t'], ['s1'], axis=0),                              # [K,1]
        n('Min', ['s1', 'ok2'], ['okm']),                                           # [K,1]
        n('Concat', ['t219_onesK', 's1', 'okm'], ['v5'], axis=1),                   # [K,3]
        # ---- per-band anchor row + aligned rows
        n('Gather', ['cyp', 'c0u'], ['colp'], axis=3),                              # [1,1,18,1]
        n('Reshape', ['colp', 't219_r18'], ['colpr']),                              # [18]
        n('Gather', ['colpr', 'rows5'], ['a5g'], axis=0),                           # [K,3]
        n('Mul', ['a5g', 'v5'], ['a5m']),                                           # [K,3]
        n('ArgMax', ['a5m'], ['a5'], axis=1, keepdims=0),                           # [K]
        n('Sub', ['a5', 'a0'], ['d5']),                                             # [K]
        n('Add', ['top5', 'd5'], ['al5']),                                          # [K]
        n('Unsqueeze', ['al5', 't219_i1'], ['al5u']),                               # [K,1]
        n('Add', ['al5u', 't219_k012'], ['rowsal']),                                # [K,3]
        # ---- band content at aligned rows, shift scoring
        n('Gather', ['cyp', 'rowsal'], ['W5'], axis=2),                             # [1,1,K,3,10]
        n('Einsum', ['shifts', 'W5'], ['scores'], equation='jsc,xykjc->ks'),        # [K,S]
        n('ArgMax', ['scores'], ['s5'], axis=1, keepdims=0),                        # [K]
        n('Unsqueeze', ['s5', 't219_i1'], ['s5u']),                                 # [K,1]
        n('Equal', ['s5u', 't219_arS'], ['sone']),                                  # [K,S] bool
        n('Cast', ['sone'], ['sf'], to=F16),                                        # [K,S]
        # ---- placement one-hots + THE einsum -> blue mask [15,10]
        n('Unsqueeze', ['rowsal', 't219_i2'], ['rau']),                             # [K,3,1]
        n('Equal', ['rau', 't219_ar15'], ['pone']),                                 # [K,3,15] bool
        n('Cast', ['pone'], ['pf'], to=F16),                                        # [K,3,15]
        n('Einsum', ['pf', 'sf', 'shifts', 'ex5'], ['mask'],
          equation='kjr,ks,jsc,k->rc'),                                             # [15,10]
        # ---- epilogue: colour = max(8*cyan, blue&empty); pad; equal -> onehot
        n('Cast', ['mask'], ['mb'], to=B),                                          # [15,10] bool
        n('Cast', ['mb'], ['mu'], to=U8),                                           # [15,10] u8
        n('Reshape', ['mu', 't219_rm4'], ['mu4']),                                  # [1,1,15,10]
        n('Cast', ['cyh'], ['cyu'], to=U8),                                         # [1,1,15,10]
        n('Mul', ['cyu', 't219_eight'], ['cy8']),                                   # [1,1,15,10]
        n('Max', ['cy8', 'mu4'], ['col15']),                                        # [1,1,15,10] max(8*cyan, mask)
        n('Pad', ['col15', 't219_pads_out', 't219_hi255'], ['col30'],
          mode='constant'),                                                         # [1,1,30,30]
        n('Equal', ['col30', 't219_ar10'], ['output']),                             # [1,10,30,30] FREE
    ]
    return model('task219_cand', nodes, inits, output_dtype=TensorProto.BOOL, opset=17)
