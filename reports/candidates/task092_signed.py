"""Task 092 candidate v2 — signed-channel priority overlay, fp16 endpoint math."""
import numpy as np
from onnx import TensorProto, helper
from ._exact import model, tensor

M = 1
F16 = TensorProto.FLOAT16


def build(task):
    I = np.eye(10, dtype=np.float16)
    e0row = np.zeros((1, 10), np.float16); e0row[0, 0] = 1.0
    Wh_const = (I - e0row).astype(np.float16)
    Wv_const = ((M + 1) * I - M * np.ones((10, 10), np.float16)).astype(np.float16)
    Wv_const[0] = 0.0
    Cstack = np.stack([Wh_const, Wv_const]).astype(np.float16)
    e00 = np.zeros((10, 10), np.float16); e00[0, 0] = 1.0
    chanmask = np.array([0.] + [1.] * 9, np.float32)          # zero bg channel in moments
    notslot0 = np.array([0.] + [1.] * 9, np.float16)
    e0_10 = np.zeros(10, np.float16); e0_10[0] = 1.0

    inits = [
        tensor('coords', np.arange(30, dtype=np.float32)),
        tensor('coords2', (np.arange(30, dtype=np.float32) ** 2)),
        tensor('chanmask', chanmask),
        tensor('coords_r', np.arange(30, dtype=np.float16).reshape(1, 1, 30)),
        tensor('coords_c', np.arange(30, dtype=np.float16).reshape(1, 30)),
        tensor('zero_f', np.float32(0.0)),
        tensor('half_h', np.float16(0.5)),
        tensor('two_h', np.float16(2.0)),
        tensor('one_f', np.float32(1.0)),
        tensor('notslot0', notslot0),
        tensor('e0_10', e0_10),
        tensor('shp_10_1', np.array([10, 1], np.int64)),
        tensor('shp_1_10_1', np.array([1, 10, 1], np.int64)),
        tensor('Cstack', Cstack),
        tensor('e00', e00),
    ]

    nodes = [
        # moments (bg channel masked out so fp16 cast never overflows)
        helper.make_node('Einsum', ['input', 'chanmask', 'coords'], ['sum_xf'], equation='bchw,c,w->bc'),
        helper.make_node('Einsum', ['input', 'chanmask', 'coords'], ['sum_yf'], equation='bchw,c,h->bc'),
        helper.make_node('Einsum', ['input', 'chanmask', 'coords2'], ['sum_x2f'], equation='bchw,c,w->bc'),
        helper.make_node('Einsum', ['input', 'chanmask', 'coords2'], ['sum_y2f'], equation='bchw,c,h->bc'),
        helper.make_node('Cast', ['sum_xf'], ['sum_x'], to=10),
        helper.make_node('Cast', ['sum_yf'], ['sum_y'], to=10),
        helper.make_node('Cast', ['sum_x2f'], ['sum_x2'], to=10),
        helper.make_node('Cast', ['sum_y2f'], ['sum_y2'], to=10),
        helper.make_node('Mul', ['sum_x', 'sum_x'], ['sx2']),
        helper.make_node('Mul', ['sum_x2', 'two_h'], ['sx2t']),
        helper.make_node('Sub', ['sx2t', 'sx2'], ['dx2r']),
        helper.make_node('Relu', ['dx2r'], ['dx2']),
        helper.make_node('Sqrt', ['dx2'], ['diff_x']),
        helper.make_node('Mul', ['sum_y', 'sum_y'], ['sy2']),
        helper.make_node('Mul', ['sum_y2', 'two_h'], ['sy2t']),
        helper.make_node('Sub', ['sy2t', 'sy2'], ['dy2r']),
        helper.make_node('Relu', ['dy2r'], ['dy2']),
        helper.make_node('Sqrt', ['dy2'], ['diff_y']),
        helper.make_node('Sub', ['sum_x', 'diff_x'], ['left2']),
        helper.make_node('Mul', ['left2', 'half_h'], ['left']),
        helper.make_node('Add', ['sum_x', 'diff_x'], ['right2']),
        helper.make_node('Mul', ['right2', 'half_h'], ['right']),
        helper.make_node('Sub', ['sum_y', 'diff_y'], ['top2']),
        helper.make_node('Mul', ['top2', 'half_h'], ['top']),
        helper.make_node('Add', ['sum_y', 'diff_y'], ['bot2']),
        helper.make_node('Mul', ['bot2', 'half_h'], ['bot']),
        # grid extent for bg slot 0 (fp32 reductions -> fp16)
        helper.make_node('Einsum', ['input'], ['rowany'], equation='bchw->bh'),
        helper.make_node('Einsum', ['input'], ['colany'], equation='bchw->bw'),
        helper.make_node('Greater', ['rowany', 'zero_f'], ['rowvalid_b']),
        helper.make_node('Greater', ['colany', 'zero_f'], ['colvalid_b']),
        helper.make_node('Cast', ['rowvalid_b'], ['rowvalid'], to=1),
        helper.make_node('Cast', ['colvalid_b'], ['colvalid'], to=1),
        helper.make_node('ReduceSum', ['rowvalid'], ['H_f']),
        helper.make_node('ReduceSum', ['colvalid'], ['W_f']),
        helper.make_node('Sub', ['H_f', 'one_f'], ['Hm1f']),
        helper.make_node('Sub', ['W_f', 'one_f'], ['Wm1f']),
        helper.make_node('Cast', ['Hm1f'], ['Hm1'], to=10),
        helper.make_node('Cast', ['Wm1f'], ['Wm1'], to=10),
        # slot-0 overrides
        helper.make_node('Mul', ['top', 'notslot0'], ['top_o']),
        helper.make_node('Mul', ['left', 'notslot0'], ['left_o']),
        helper.make_node('Mul', ['bot', 'notslot0'], ['bot_m']),
        helper.make_node('Mul', ['Hm1', 'e0_10'], ['bot_bg']),
        helper.make_node('Add', ['bot_m', 'bot_bg'], ['bot_o']),
        helper.make_node('Mul', ['right', 'notslot0'], ['right_m']),
        helper.make_node('Mul', ['Wm1', 'e0_10'], ['right_bg']),
        helper.make_node('Add', ['right_m', 'right_bg'], ['right_o']),
        # RS band [1,10,30]
        helper.make_node('Reshape', ['top_o', 'shp_1_10_1'], ['topT']),
        helper.make_node('Reshape', ['bot_o', 'shp_1_10_1'], ['botT']),
        helper.make_node('GreaterOrEqual', ['coords_r', 'topT'], ['geR']),
        helper.make_node('LessOrEqual', ['coords_r', 'botT'], ['leR']),
        helper.make_node('And', ['geR', 'leR'], ['RSb']),
        helper.make_node('Cast', ['RSb'], ['RS'], to=10),
        # CS band [10,30]
        helper.make_node('Reshape', ['left_o', 'shp_10_1'], ['leftT']),
        helper.make_node('Reshape', ['right_o', 'shp_10_1'], ['rightT']),
        helper.make_node('GreaterOrEqual', ['coords_c', 'leftT'], ['geC']),
        helper.make_node('LessOrEqual', ['coords_c', 'rightT'], ['leC']),
        helper.make_node('And', ['geC', 'leC'], ['CSb']),
        helper.make_node('Cast', ['CSb'], ['CS'], to=10),
        # W [10,10]
        helper.make_node('Greater', ['diff_x', 'half_h'], ['isHb']),   # diff>0 (>=1 for sticks)
        helper.make_node('Greater', ['diff_y', 'half_h'], ['isVb']),
        helper.make_node('Cast', ['isHb'], ['isH'], to=10),
        helper.make_node('Cast', ['isVb'], ['isV'], to=10),
        helper.make_node('Reshape', ['isH', 'shp_10_1'], ['isHc']),
        helper.make_node('Reshape', ['isV', 'shp_10_1'], ['isVc']),
        helper.make_node('Concat', ['isHc', 'isVc'], ['flags'], axis=1),
        helper.make_node('Einsum', ['flags', 'Cstack'], ['Wstick'], equation='qs,sqv->qv'),
        helper.make_node('Add', ['Wstick', 'e00'], ['Wfin']),
        # free final einsum
        helper.make_node('Einsum', ['RS', 'CS', 'Wfin'], ['output'], equation='bqr,qc,qv->bvrc'),
    ]
    return model('task092_signed', nodes, inits, output_dtype=F16, opset=16)
