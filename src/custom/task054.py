"""task054 candidate — constructive rebuild (free-input einsum profiles +
sparse edit chain), mirrors scratchpad proto2 exactly.

Chain: label Conv+Cast -> E1 5-row ref wipe ScatterND -> E2 8-cell ring
ScatterND -> E3-H row ScatterND(reduction=max, 255-encoded lines) ->
E3-V ScatterElements(max) -> free Equal against 255-adjusted colour table.

Seeds are found from the free 'input' via multi-operand Einsum row profiles
of the star colour (motif 3x3 zone masked out for rows within +-1 of the
motif center; sound because box cells are Chebyshev>=4 from the center).
Line spans use the incumbent's CumSum segment trick, batched [1,1,4,30].
Duplicate-row scatter conflicts are resolved by reduction='max' since line
cells are encoded as 255 (> any label); the final Equal matches channel
line_color against 255 instead (input never contains line_color cells
outside the wiped motif).
"""
import numpy as np
from onnx import TensorProto, helper
from ._exact import model, tensor

I64 = np.int64
I32 = np.int32
U8 = np.uint8
F32 = np.float32


def build(task):
    n = helper.make_node
    inits = [
        tensor('W10', np.arange(10, dtype=F32).reshape(1, 10, 1, 1)),
        tensor('COLORS', np.arange(10, dtype=U8).reshape(1, 10, 1, 1)),
        tensor('IOTA_W', np.arange(30, dtype=I32).reshape(1, 1, 1, 30)),
        tensor('IOTA_H', np.arange(30, dtype=I32).reshape(1, 1, 30, 1)),
        tensor('IOTA30_I32', np.arange(30, dtype=I32)),
        tensor('IOTA30_U8', np.arange(30, dtype=U8)),
        tensor('WOFF5', np.array([-2, -1, 0, 1, 2], dtype=I64)),
        tensor('Z52', np.zeros((5, 2), dtype=I64)),
        tensor('Z42', np.zeros((4, 2), dtype=I64)),
        tensor('Z481', np.zeros((4, 8, 1), dtype=I64)),
        tensor('FOFFR', np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=I64).reshape(1, 8, 1)),
        tensor('FOFFC', np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=I64).reshape(1, 8, 1)),
        tensor('EXPSH', np.array([1, 1, 30, 4], dtype=I64)),
        # reshape shapes
        tensor('SH1', np.array([1], dtype=I64)),
        tensor('SH4', np.array([4], dtype=I64)),
        tensor('SH10', np.array([10], dtype=I64)),
        tensor('SH1130', np.array([1, 1, 1, 30], dtype=I64)),
        tensor('SH41', np.array([4, 1], dtype=I64)),
        tensor('SH411', np.array([4, 1, 1], dtype=I64)),
        tensor('SH1141', np.array([1, 1, 4, 1], dtype=I64)),
        tensor('SH1114', np.array([1, 1, 1, 4], dtype=I64)),
        tensor('SH4_30', np.array([4, 30], dtype=I64)),
        tensor('SH5_30', np.array([5, 30], dtype=I64)),
        tensor('SH5_1', np.array([5, 1], dtype=I64)),
        # slice constants
        tensor('S00', np.array([0, 0], dtype=I64)),
        tensor('S11', np.array([1, 1], dtype=I64)),
        tensor('S22', np.array([2, 2], dtype=I64)),
        tensor('S55', np.array([5, 5], dtype=I64)),
        tensor('S66', np.array([6, 6], dtype=I64)),
        tensor('AX23', np.array([2, 3], dtype=I64)),
        tensor('CR0S', np.array([1, 21], dtype=I64)),
        tensor('CR0E', np.array([8, 28], dtype=I64)),
        tensor('CR1S', np.array([21, 1], dtype=I64)),
        tensor('CR1E', np.array([28, 8], dtype=I64)),
        tensor('CR2S', np.array([21, 21], dtype=I64)),
        tensor('CR2E', np.array([29, 29], dtype=I64)),
        # scalars
        tensor('ZF', np.array([0.0], dtype=F32)),
        tensor('HALF_F', np.array([0.5], dtype=F32)),
        tensor('ONEP5_F', np.array([1.5], dtype=F32)),
        tensor('Z_U8', np.array([0], dtype=U8)),
        tensor('C255_U8', np.array([255], dtype=U8)),
        tensor('C30_U8', np.array([30], dtype=U8)),
        tensor('Z_I32', np.array([0], dtype=I32)),
        tensor('ONE_I32', np.array([1], dtype=I32)),
        tensor('TWO_I32', np.array([2], dtype=I32)),
        tensor('C27_I32', np.array([27], dtype=I32)),
        tensor('M1_64', np.array([-1], dtype=I64)),
        tensor('M2_64', np.array([-2], dtype=I64)),
        tensor('ONE_64', np.array([1], dtype=I64)),
        tensor('TWO_64', np.array([2], dtype=I64)),
        tensor('C21_64', np.array([21], dtype=I64)),
        tensor('C28_64', np.array([28], dtype=I64)),
        tensor('AX0', np.array([0], dtype=I64)),
        tensor('AX2', np.array([2], dtype=I64)),
        tensor('AX3', np.array([3], dtype=I64)),
        tensor('I0', np.array([0], dtype=I64)),
        tensor('I1', np.array([1], dtype=I64)),
        tensor('I2', np.array([2], dtype=I64)),
        tensor('I3', np.array([3], dtype=I64)),
    ]
    nodes = []

    # ---- A: label read ----
    nodes += [
        n('Conv', ['input', 'W10'], ['labf'], kernel_shape=[1, 1]),
        n('Cast', ['labf'], ['lab'], to=TensorProto.UINT8),
    ]
    # ---- B: bg + box colours ----
    nodes += [
        n('Slice', ['lab', 'S00', 'S11', 'AX23'], ['bgpix']),
        n('ReduceSum', ['input', 'AX23'], ['ccnt'], keepdims=1),
        n('Equal', ['COLORS', 'bgpix'], ['isbg']),
        n('Where', ['isbg', 'ZF', 'ccnt'], ['ccnt2']),
        n('ArgMax', ['ccnt2'], ['boxi'], axis=1, keepdims=1),
        n('Cast', ['boxi'], ['boxpix'], to=TensorProto.UINT8),
    ]

    # ---- C: ref crops ----
    def crop(k, st, en, ie, rev, nm1, roff, coff):
        return [
            n('Slice', ['lab', st, en, 'AX23'], [f'cr{k}']),
            n('Equal', [f'cr{k}', 'bgpix'], [f'cb{k}']),
            n('Equal', [f'cr{k}', 'boxpix'], [f'cx{k}']),
            n('Or', [f'cb{k}', f'cx{k}'], [f'cbx{k}']),
            n('Not', [f'cbx{k}'], [f'cob{k}']),
            n('Cast', [f'cob{k}'], [f'co{k}'], to=TensorProto.UINT8),
            n('Slice', [f'co{k}', 'S22', ie, 'AX23'], [f'ci{k}']),
            n('Cast', [f'ci{k}'], [f'cii{k}'], to=TensorProto.INT32),
            n('ReduceSum', [f'cii{k}', 'AX23'], [f'cnt{k}'], keepdims=1),
            n('ReduceMax', [f'co{k}', 'AX3'], [f'rh{k}'], keepdims=1),
            n('ArgMax', [f'rh{k}'], [f'rmn{k}'], axis=2, keepdims=1),
            n('ArgMax', [f'rh{k}'], [f'rmx{k}'], axis=2, keepdims=1, select_last_index=1),
            n('Add', [f'rmn{k}', f'rmx{k}'], [f'rsm{k}']),
            n('Div', [f'rsm{k}', 'TWO_64'], [f'rcl{k}']),
            n('Add', [f'rcl{k}', roff], [f'rce{k}']),
            n('ReduceMax', [f'co{k}', 'AX2'], [f'ch{k}'], keepdims=1),
            n('ArgMax', [f'ch{k}'], [f'cmn{k}'], axis=3, keepdims=1),
            n('ArgMax', [f'ch{k}'], [f'cmx{k}'], axis=3, keepdims=1, select_last_index=1),
            n('Add', [f'cmn{k}', f'cmx{k}'], [f'csm{k}']),
            n('Div', [f'csm{k}', 'TWO_64'], [f'ccl{k}']),
            n('Add', [f'ccl{k}', coff], [f'cce{k}']),
        ]
    nodes += crop(0, 'CR0S', 'CR0E', 'S55', 'REV7', 'SIX_64', 'ONE_64', 'C21_64')
    nodes += crop(1, 'CR1S', 'CR1E', 'S55', 'REV7', 'SIX_64', 'C21_64', 'ONE_64')
    nodes += crop(2, 'CR2S', 'CR2E', 'S66', 'REV8', 'SEVEN_64', 'C21_64', 'C21_64')
    nodes += [
        n('Greater', ['cnt0', 'cnt1'], ['g01']),
        n('Where', ['g01', 'cnt0', 'cnt1'], ['n01']),
        n('Greater', ['n01', 'cnt2'], ['g012']),
        n('Where', ['g01', 'rce0', 'rce1'], ['r01']),
        n('Where', ['g01', 'cce0', 'cce1'], ['c01']),
        n('Where', ['g012', 'r01', 'rce2'], ['mr0']),
        n('Where', ['g012', 'c01', 'cce2'], ['mc0']),
        # clamp to [2,27] (safety on degenerate inputs)
        n('Cast', ['mr0'], ['mr0i'], to=TensorProto.INT32),
        n('Cast', ['mc0'], ['mc0i'], to=TensorProto.INT32),
        n('Max', ['mr0i', 'TWO_I32'], ['mr1i']),
        n('Min', ['mr1i', 'C27_I32'], ['mri']),  # [1,1,1,1] i32
        n('Max', ['mc0i', 'TWO_I32'], ['mc1i']),
        n('Min', ['mc1i', 'C27_I32'], ['mci']),
        n('Cast', ['mri'], ['mr64_4d'], to=TensorProto.INT64),
        n('Cast', ['mci'], ['mc64_4d'], to=TensorProto.INT64),
        n('Reshape', ['mr64_4d', 'SH1'], ['mr_1']),
        n('Reshape', ['mc64_4d', 'SH1'], ['mc_1']),
    ]

    # ---- D: motif scalars ----
    nodes += [
        n('Add', ['mr_1', 'M1_64'], ['mrm1']),
        n('Add', ['mr_1', 'M2_64'], ['mrm2']),
        n('Add', ['mc_1', 'M1_64'], ['mcm1']),
        n('Add', ['mc_1', 'M2_64'], ['mcm2']),
        n('Gather', ['lab', 'mr_1'], ['rowB'], axis=2),
        n('Gather', ['lab', 'mrm1'], ['rowA'], axis=2),
        n('Gather', ['lab', 'mrm2'], ['rowC'], axis=2),
        n('Gather', ['rowB', 'mc_1'], ['s0v'], axis=3),
        n('Gather', ['rowB', 'mcm1'], ['west'], axis=3),
        n('Gather', ['rowB', 'mcm2'], ['htv'], axis=3),
        n('Gather', ['rowA', 'mc_1'], ['north'], axis=3),
        n('Gather', ['rowA', 'mcm1'], ['fillv'], axis=3),
        n('Gather', ['rowC', 'mc_1'], ['vtv'], axis=3),
    ]
    for nm in ('htv', 'vtv', 'fillv'):
        nodes += [
            n('Equal', [nm, 'bgpix'], [f'{nm}_b']),
            n('Equal', [nm, 'boxpix'], [f'{nm}_x']),
            n('Or', [f'{nm}_b', f'{nm}_x'], [f'{nm}_o']),
            n('Not', [f'{nm}_o'], [f'{nm}_p']),
        ]
    nodes += [
        n('Where', ['vtv_p', 'north', 'west'], ['lcv']),
        n('Reshape', ['fillv', 'SH1'], ['fv1']),
        n('Reshape', ['fillv_p', 'SH1'], ['fp1']),
        # +-1 / +-2 neighbour scalars in i32 (for two-compare window masks)
        n('Add', ['mr_1', 'ONE_64'], ['mrp1']),
        n('Add', ['mc_1', 'ONE_64'], ['mcp1']),
        n('Add', ['mc_1', 'TWO_64'], ['mcp2']),
        n('Cast', ['mrm1'], ['mrm1i'], to=TensorProto.INT32),
        n('Cast', ['mrp1'], ['mrp1i'], to=TensorProto.INT32),
        n('Cast', ['mcm1'], ['mcm1i'], to=TensorProto.INT32),
        n('Cast', ['mcp1'], ['mcp1i'], to=TensorProto.INT32),
        n('Cast', ['mcm2'], ['mcm2i'], to=TensorProto.INT32),
        n('Cast', ['mcp2'], ['mcp2i'], to=TensorProto.INT32),
    ]

    # ---- E1: 5-row ref wipe ----
    nodes += [
        n('Add', ['mr_1', 'WOFF5'], ['rows5']),
        n('Gather', ['lab', 'rows5'], ['g5'], axis=2),
        n('Equal', ['g5', 'boxpix'], ['e5']),
        n('Where', ['e5', 'g5', 'bgpix'], ['w5']),
        n('GreaterOrEqual', ['IOTA_W', 'mcm2i'], ['cm2a']),
        n('LessOrEqual', ['IOTA_W', 'mcp2i'], ['cm2b']),
        n('And', ['cm2a', 'cm2b'], ['cm2']),
        n('Where', ['cm2', 'w5', 'g5'], ['u5']),
        n('Reshape', ['u5', 'SH5_30'], ['u5r']),
        n('Reshape', ['rows5', 'SH5_1'], ['rows5c']),
        n('Concat', ['Z52', 'rows5c'], ['idx5'], axis=1),
        n('ScatterND', ['lab', 'idx5', 'u5r'], ['cleared']),
    ]

    # ---- F: seed slots from profiles ----
    nodes += [
        n('Equal', ['COLORS', 's0v'], ['s0b']),
        n('Cast', ['s0b'], ['s0f'], to=TensorProto.FLOAT),
        n('Reshape', ['s0f', 'SH10'], ['s0f10']),
        n('Einsum', ['input', 's0f10'], ['p'], equation='bchw,c->h'),
        n('GreaterOrEqual', ['IOTA30_I32', 'mcm1i'], ['cm1a']),
        n('LessOrEqual', ['IOTA30_I32', 'mcp1i'], ['cm1b']),
        n('And', ['cm1a', 'cm1b'], ['cm1_30']),
        n('Cast', ['cm1_30'], ['cm1f'], to=TensorProto.FLOAT),
        n('Reshape', ['cm1_30', 'SH1130'], ['cm1']),
        n('Einsum', ['input', 's0f10', 'cm1f'], ['pnear'], equation='bchw,c,w->h'),
        n('GreaterOrEqual', ['IOTA30_I32', 'mrm1i'], ['nra']),
        n('LessOrEqual', ['IOTA30_I32', 'mrp1i'], ['nrbb']),
        n('And', ['nra', 'nrbb'], ['nrb']),
        n('Sub', ['p', 'pnear'], ['psub']),
        n('Where', ['nrb', 'psub', 'p'], ['pc']),
        n('Greater', ['pc', 'HALF_F'], ['sup']),
        n('Where', ['sup', 'IOTA30_U8', 'C255_U8'], ['wm1']),
        n('ReduceMin', ['wm1', 'AX0'], ['r1u'], keepdims=1),
        n('Where', ['sup', 'IOTA30_U8', 'Z_U8'], ['wm2']),
        n('ReduceMax', ['wm2', 'AX0'], ['r4u'], keepdims=1),
        n('Less', ['r1u', 'C30_U8'], ['pres1']),
        n('Where', ['pres1', 'r1u', 'Z_U8'], ['r1c']),
        n('Equal', ['IOTA30_U8', 'r1c'], ['oh1']),
        n('Equal', ['IOTA30_U8', 'r4u'], ['oh4']),
        n('Or', ['oh1', 'oh4'], ['ohh']),
        n('Greater', ['pc', 'ONEP5_F'], ['two2']),
        n('Equal', ['r1c', 'r4u'], ['eq14']),
        n('Not', ['eq14'], ['ne14']),
        n('Not', ['ohh'], ['nohh']),
        n('And', ['sup', 'nohh'], ['sA']),
        n('And', ['two2', 'ne14'], ['sB0']),
        n('And', ['ohh', 'sB0'], ['sB']),
        n('Or', ['sA', 'sB'], ['sup2']),
        n('Where', ['sup2', 'IOTA30_U8', 'C255_U8'], ['wm3']),
        n('ReduceMin', ['wm3', 'AX0'], ['r2u'], keepdims=1),
        n('Where', ['sup2', 'IOTA30_U8', 'Z_U8'], ['wm4']),
        n('ReduceMax', ['wm4', 'AX0'], ['r3u'], keepdims=1),
        n('Less', ['r2u', 'C30_U8'], ['pres2']),
        n('Where', ['pres2', 'r2u', 'Z_U8'], ['r2c']),
        n('Concat', ['r1c', 'r2c', 'r3u', 'r4u'], ['rows4u'], axis=0),
        n('Cast', ['rows4u'], ['rows4i'], to=TensorProto.INT32),
        n('Gather', ['lab', 'rows4i'], ['lrow4'], axis=2),
        n('Equal', ['lrow4', 's0v'], ['eqs0']),
        n('Reshape', ['rows4i', 'SH1141'], ['rows4r']),
        n('GreaterOrEqual', ['rows4r', 'mrm1i'], ['nr4a']),
        n('LessOrEqual', ['rows4r', 'mrp1i'], ['nr4b']),
        n('And', ['nr4a', 'nr4b'], ['nr4']),
        n('Not', ['nr4'], ['nnr4']),
        n('Not', ['cm1'], ['ncm1']),
        n('Or', ['nnr4', 'ncm1'], ['nokill']),
        n('And', ['eqs0', 'nokill'], ['mrow']),
        n('Cast', ['mrow'], ['mru8'], to=TensorProto.UINT8),
        n('ArgMax', ['mru8'], ['cf4'], axis=3, keepdims=1),
        n('ArgMax', ['mru8'], ['cl4'], axis=3, keepdims=1, select_last_index=1),
        n('Reshape', ['cf4', 'SH4'], ['cfF']),
        n('Reshape', ['cl4', 'SH4'], ['clF']),
        n('Gather', ['cfF', 'I0'], ['cf_r1'], axis=0),
        n('Gather', ['clF', 'I0'], ['cl_r1'], axis=0),
        n('Gather', ['cfF', 'I1'], ['cf_r2'], axis=0),
        n('Gather', ['clF', 'I2'], ['cl_r3'], axis=0),
        n('Gather', ['cfF', 'I3'], ['cf_r4'], axis=0),
        n('Gather', ['clF', 'I3'], ['cl_r4'], axis=0),
        n('Equal', ['r2c', 'r1c'], ['e21']),
        n('Where', ['e21', 'cl_r1', 'cf_r2'], ['c2v']),
        n('Equal', ['r3u', 'r4u'], ['e34']),
        n('Where', ['e34', 'cf_r4', 'cl_r3'], ['c3v']),
        n('Cast', ['r1c'], ['r1_64'], to=TensorProto.INT64),
        n('Cast', ['r2c'], ['r2_64'], to=TensorProto.INT64),
        n('Cast', ['r3u'], ['r3_64'], to=TensorProto.INT64),
        n('Cast', ['r4u'], ['r4_64'], to=TensorProto.INT64),
    ]
    # slot coords, muxed + clamped [1,28]
    slotdefs = [
        ('0', 'pres1', 'r1_64', 'cf_r1'),
        ('1', 'pres2', 'r2_64', 'c2v'),
        ('2', 'pres2', 'r3_64', 'c3v'),
        ('3', 'pres1', 'r4_64', 'cl_r4'),
    ]
    for j, pr, rr, cc in slotdefs:
        nodes += [
            n('Where', [pr, rr, 'mr_1'], [f'sr{j}a']),
            n('Where', [pr, cc, 'mc_1'], [f'sc{j}a']),
            n('Max', [f'sr{j}a', 'ONE_64'], [f'sr{j}b']),
            n('Min', [f'sr{j}b', 'C28_64'], [f'sr{j}']),
            n('Max', [f'sc{j}a', 'ONE_64'], [f'sc{j}b']),
            n('Min', [f'sc{j}b', 'C28_64'], [f'sc{j}']),
        ]
    nodes += [
        n('Concat', ['sr0', 'sr1', 'sr2', 'sr3'], ['srow4'], axis=0),
        n('Concat', ['sc0', 'sc1', 'sc2', 'sc3'], ['scol4'], axis=0),
        n('Concat', ['pres1', 'pres2', 'pres2', 'pres1'], ['pres4'], axis=0),
    ]

    # ---- G: E2 ring fill ----
    nodes += [
        n('Reshape', ['srow4', 'SH411'], ['srow411']),
        n('Reshape', ['scol4', 'SH411'], ['scol411']),
        n('Add', ['srow411', 'FOFFR'], ['fr']),
        n('Add', ['scol411', 'FOFFC'], ['fc']),
        n('Concat', ['Z481', 'Z481', 'fr', 'fc'], ['fidx'], axis=2),
        n('GatherND', ['cleared', 'fidx'], ['fcur']),
        n('Reshape', ['pres4', 'SH41'], ['pres41']),
        n('And', ['pres41', 'fp1'], ['fact']),
        n('Where', ['fact', 'fv1', 'fcur'], ['fupd']),
        n('ScatterND', ['cleared', 'fidx', 'fupd'], ['filled']),
    ]

    # ---- H: horizontal lines ----
    nodes += [
        n('Cast', ['srow4'], ['srow4i'], to=TensorProto.INT32),
        n('Cast', ['scol4'], ['scol4i'], to=TensorProto.INT32),
        n('Reshape', ['htv_p', 'SH1'], ['ht1']),
        n('Reshape', ['vtv_p', 'SH1'], ['vt1']),
        n('Gather', ['lab', 'srow4i'], ['hbase'], axis=2),
        n('Equal', ['hbase', 'boxpix'], ['hbox']),
        n('Reshape', ['scol4i', 'SH1141'], ['scol_r']),
        n('Equal', ['IOTA_W', 'scol_r'], ['hceq']),
        n('Or', ['hbox', 'hceq'], ['hvec']),
        n('Where', ['hvec', 'Z_I32', 'ONE_I32'], ['hgap']),
        n('CumSum', ['hgap', 'AX3'], ['hseg']),
        n('GatherElements', ['hseg', 'scol_r'], ['hsseed'], axis=3),
        n('Equal', ['hseg', 'hsseed'], ['hsame']),
        # box-cells in the seed's segment; seed centers are star-coloured so
        # they are excluded from the line mask automatically
        n('And', ['hsame', 'hbox'], ['hm1']),
        n('Reshape', ['pres4', 'SH1141'], ['pres_h']),
        n('And', ['pres_h', 'ht1'], ['hact']),
        n('And', ['hm1', 'hact'], ['hm3']),
        n('Gather', ['filled', 'srow4i'], ['hcur'], axis=2),
        n('Where', ['hm3', 'C255_U8', 'hcur'], ['hupd4']),
        n('Reshape', ['hupd4', 'SH4_30'], ['hupd']),
        n('Reshape', ['srow4', 'SH41'], ['srow41']),
        n('Concat', ['Z42', 'srow41'], ['hidx'], axis=1),
        n('ScatterND', ['filled', 'hidx', 'hupd'], ['hlined'], reduction='max'),
    ]

    # ---- V: vertical lines ----
    nodes += [
        n('Gather', ['lab', 'scol4i'], ['vbase'], axis=3),
        n('Equal', ['vbase', 'boxpix'], ['vbox']),
        n('Reshape', ['srow4i', 'SH1114'], ['srow_c']),
        n('Equal', ['IOTA_H', 'srow_c'], ['vreq']),
        n('Or', ['vbox', 'vreq'], ['vvec']),
        n('Where', ['vvec', 'Z_I32', 'ONE_I32'], ['vgap']),
        n('CumSum', ['vgap', 'AX2'], ['vseg']),
        n('GatherElements', ['vseg', 'srow_c'], ['vsseed'], axis=2),
        n('Equal', ['vseg', 'vsseed'], ['vsame']),
        n('And', ['vsame', 'vbox'], ['vm1']),
        n('Reshape', ['pres4', 'SH1114'], ['pres_v']),
        n('And', ['pres_v', 'vt1'], ['vact']),
        n('And', ['vm1', 'vact'], ['vm3']),
        n('Gather', ['hlined', 'scol4i'], ['vcur'], axis=3),
        n('Where', ['vm3', 'C255_U8', 'vcur'], ['vupd']),
        n('Reshape', ['scol4i', 'SH1114'], ['scolc']),
        n('Expand', ['scolc', 'EXPSH'], ['vidx']),
        n('ScatterElements', ['hlined', 'vidx', 'vupd'], ['outlab'], axis=3, reduction='max'),
    ]

    # ---- final free Equal with 255-adjusted colour table ----
    nodes += [
        n('Equal', ['COLORS', 'lcv'], ['iss1']),
        n('Where', ['iss1', 'C255_U8', 'COLORS'], ['colors_adj']),
        n('Equal', ['outlab', 'colors_adj'], ['output']),
    ]
    return model('task054_cand', nodes, inits, output_dtype=TensorProto.BOOL, opset=18)
