"""Task 370 — REDESIGN CANDIDATE: runtime-parameterized stamp (RTS).

Replaces the 4-branch dilated-QLinearConv candidate bank + mux chain with:
  1. Gather-based `d` detection: d = max{ d in {2,3,4,5} : black20[hint - d*dir] is
     a black sprite pixel }.  (direction `dir` from hint-vs-sprite-centroid sign.)
  2. A single runtime kernel assembled by ScatterND: a CENTERED (S x S) diagonal
     line kernel, dilation=1, with 1s at (C - k*d*dir_r, C - k*d*dir_c), k=0..8.
     Direction is baked into the tap positions (no input flips needed).
  3. ONE QLinearConv (pads=[C,C,C,C]) -> repeat plane -> existing epilogue.

S = 2*C+1.  C is the max stamp offset that must be representable.
Measurement / gate harness only; not adopted here.
"""
import numpy as np
from onnx import TensorProto, helper
from src.custom._exact import tensor, model

# Kernel half-size (max stamp offset).  C=15 is the fresh minimum: the densest
# fresh spacing d=3 reaches stamp offset 15 (0/2500 fresh fail; C=14 fails 2).
C = 15
S = 2 * C + 1          # 31x31 kernel
KMIN, KMAX = 1, 8      # stamp indices k = 1..8 (k=0 = origin sprite, excluded)


def build(task):
    inits = [
        tensor('false_b', np.array(False, dtype=np.bool_)),
        tensor('ax13', np.array([1, 3], dtype=np.int64)),
        tensor('ax12', np.array([1, 2], dtype=np.int64)),
        tensor('ax23', np.array([2, 3], dtype=np.int64)),
        tensor('ax123', np.array([1, 2, 3], dtype=np.int64)),
        tensor('q_scale', np.array(1.0, dtype=np.float32)),
        tensor('q_zp', np.array(0, dtype=np.uint8)),
        tensor('row20', np.arange(20, dtype=np.float32)),
        tensor('row30', np.arange(30, dtype=np.float32)),
        tensor('sel_idx_0', np.array([0], dtype=np.int64)),
        tensor('slice_starts', np.array([0, 0, 0, 0], dtype=np.int64)),
        tensor('slice_ends', np.array([1, 1, 20, 20], dtype=np.int64)),
        tensor('pad_to_30', np.array([0, 0, 0, 0, 0, 0, 10, 10], dtype=np.int64)),
        tensor('vec_starts', np.array([0, 0], dtype=np.int64)),
        tensor('vec_ends20', np.array([1, 20], dtype=np.int64)),
        # detection
        tensor('dvals', np.array([2, 3, 4, 5], dtype=np.int64)),
        tensor('pos1', np.array([1], dtype=np.int64)),
        tensor('neg1', np.array([-1], dtype=np.int64)),
        tensor('two_i', np.array(2, dtype=np.int64)),
        tensor('zeros4', np.zeros((4, 1), dtype=np.int64)),
        tensor('clip_lo', np.array(0, dtype=np.int64)),
        tensor('clip_hi', np.array(19, dtype=np.int64)),
        tensor('u8_zero', np.array(0, dtype=np.uint8)),
        # kernel build
        tensor('base_k', np.arange(KMIN, KMAX + 1, dtype=np.int64)),
        tensor('cC', np.array(C, dtype=np.int64)),
        tensor('cS1', np.array(S - 1, dtype=np.int64)),
        tensor('trash_c', np.array(1, dtype=np.int64)),   # (0,1): off both diagonals
        tensor('lo0', np.array(0, dtype=np.int64)),
        tensor('zerosN', np.zeros((KMAX - KMIN + 1, 1), dtype=np.int64)),
        tensor('onesN_u8', np.ones((KMAX - KMIN + 1,), dtype=np.uint8)),
        tensor('zerosN_u8', np.zeros((KMAX - KMIN + 1,), dtype=np.uint8)),
        tensor('zeroS', np.zeros((1, 1, S, S), dtype=np.uint8)),
    ]
    nodes = [
        # valid rows/cols
        helper.make_node('ReduceSum', ['input', 'ax13'], ['vrc'], keepdims=0),
        helper.make_node('ReduceSum', ['input', 'ax12'], ['vcc'], keepdims=0),
        helper.make_node('Slice', ['vrc', 'vec_starts', 'vec_ends20'], ['vr20']),
        helper.make_node('Slice', ['vcc', 'vec_starts', 'vec_ends20'], ['vc20']),
        helper.make_node('Greater', ['vr20', 'q_scale'], ['vrb']),
        helper.make_node('Greater', ['vc20', 'q_scale'], ['vcb']),
        helper.make_node('Unsqueeze', ['vrb', 'ax13'], ['valid_rows20']),
        helper.make_node('Unsqueeze', ['vcb', 'ax12'], ['valid_cols20']),
        # hint colour
        helper.make_node('ReduceSum', ['input', 'ax23'], ['counts'], keepdims=0),
        helper.make_node('Equal', ['counts', 'q_scale'], ['hcb']),
        helper.make_node('Cast', ['hcb'], ['hcv'], to=1),
        helper.make_node('Unsqueeze', ['hcv', 'ax23'], ['hint_color']),
        # black20
        helper.make_node('Slice', ['input', 'slice_starts', 'slice_ends'], ['black20']),
        helper.make_node('Cast', ['black20'], ['black20_u8'], to=2),
        # direction sign
        helper.make_node('Einsum', ['black20', 'row20'], ['brs'], equation='nchw,h->n'),
        helper.make_node('Einsum', ['black20', 'row20'], ['bcs'], equation='nchw,w->n'),
        helper.make_node('ReduceSum', ['black20', 'ax123'], ['bcnt'], keepdims=0),
        helper.make_node('Einsum', ['input', 'hcv', 'row30'], ['hr'], equation='nchw,nc,h->n'),
        helper.make_node('Einsum', ['input', 'hcv', 'row30'], ['hc'], equation='nchw,nc,w->n'),
        helper.make_node('Mul', ['hr', 'bcnt'], ['hrs']),
        helper.make_node('Mul', ['hc', 'bcnt'], ['hcs']),
        helper.make_node('Greater', ['hrs', 'brs'], ['row_pos']),
        helper.make_node('Greater', ['hcs', 'bcs'], ['col_pos']),
        helper.make_node('Cast', ['hr'], ['hri'], to=7),
        helper.make_node('Cast', ['hc'], ['hci'], to=7),
        helper.make_node('Where', ['row_pos', 'pos1', 'neg1'], ['dir_r']),
        helper.make_node('Where', ['col_pos', 'pos1', 'neg1'], ['dir_c']),
        # d detection
        helper.make_node('Mul', ['dvals', 'dir_r'], ['dro']),
        helper.make_node('Mul', ['dvals', 'dir_c'], ['dco']),
        helper.make_node('Sub', ['hri', 'dro'], ['prr']),
        helper.make_node('Sub', ['hci', 'dco'], ['pcr']),
        helper.make_node('Clip', ['prr', 'clip_lo', 'clip_hi'], ['pr']),
        helper.make_node('Clip', ['pcr', 'clip_lo', 'clip_hi'], ['pc']),
        helper.make_node('Unsqueeze', ['pr', 'pos1'], ['prc']),
        helper.make_node('Unsqueeze', ['pc', 'pos1'], ['pcc']),
        helper.make_node('Concat', ['zeros4', 'zeros4', 'prc', 'pcc'], ['pidx'], axis=1),
        helper.make_node('GatherND', ['black20_u8', 'pidx'], ['pvals']),
        helper.make_node('Greater', ['pvals', 'u8_zero'], ['pvb']),
        helper.make_node('Cast', ['pvb'], ['pvi'], to=7),
        helper.make_node('Mul', ['pvi', 'dvals'], ['vdv']),
        helper.make_node('ReduceMax', ['vdv'], ['d_raw'], keepdims=0),
        helper.make_node('Max', ['d_raw', 'two_i'], ['d_scalar']),
        helper.make_node('Unsqueeze', ['d_scalar', 'sel_idx_0'], ['d1']),
        # kernel build: tap = C - k*d*dir
        helper.make_node('Mul', ['base_k', 'd1'], ['kd']),
        helper.make_node('Mul', ['kd', 'dir_r'], ['kdr']),
        helper.make_node('Mul', ['kd', 'dir_c'], ['kdc']),
        helper.make_node('Sub', ['cC', 'kdr'], ['tr']),
        helper.make_node('Sub', ['cC', 'kdc'], ['tc']),
        helper.make_node('Less', ['tr', 'lo0'], ['olr']),
        helper.make_node('Greater', ['tr', 'cS1'], ['ohr']),
        helper.make_node('Or', ['olr', 'ohr'], ['oob']),
        helper.make_node('Where', ['oob', 'lo0', 'tr'], ['tr_s']),
        helper.make_node('Where', ['oob', 'trash_c', 'tc'], ['tc_s']),
        helper.make_node('Unsqueeze', ['tr_s', 'pos1'], ['trc']),
        helper.make_node('Unsqueeze', ['tc_s', 'pos1'], ['tcc']),
        helper.make_node('Concat', ['zerosN', 'zerosN', 'trc', 'tcc'], ['kidx'], axis=1),
        helper.make_node('Where', ['oob', 'zerosN_u8', 'onesN_u8'], ['kupd']),
        helper.make_node('ScatterND', ['zeroS', 'kidx', 'kupd'], ['w_runtime']),
        # single conv
        helper.make_node('QLinearConv',
                         ['black20_u8', 'q_scale', 'q_zp', 'w_runtime', 'q_scale', 'q_zp', 'q_scale', 'q_zp'],
                         ['repeat20'], dilations=[1, 1], pads=[C, C, C, C]),
        # epilogue
        helper.make_node('Where', ['valid_rows20', 'repeat20', 'q_zp'], ['rrv']),
        helper.make_node('Where', ['valid_cols20', 'rrv', 'q_zp'], ['rv']),
        helper.make_node('Greater', ['rv', 'q_zp'], ['rvb']),
        helper.make_node('Pad', ['rvb', 'pad_to_30', 'false_b'], ['rvb30'], mode='constant'),
        helper.make_node('Where', ['rvb30', 'hint_color', 'input'], ['output']),
    ]
    return model('task370_rts', nodes, inits, output_dtype=1, opset=13)
