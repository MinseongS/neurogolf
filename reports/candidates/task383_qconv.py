"""Task383 candidate — QLinearConv on the colour-index Conv, with i32 bias fold (S10 recipe).

PRODUCER_BOUND / RANKING_ONLY: Conv(input[1,10,30,30], color_weights[1,10,2,2],
color_bias) -> color_f[1,1,24,24] fp32 (2304B... scan save 1728B), dilated 2x2 kernel,
weights -10..0, bias +11, output colour label 0..11. color_f feeds only Cast -> order-only
consumers, so int8 QLinearConv is bit-exact (S10 order-preserving class). QLinearConv's
optional int32 bias (input index 8) carries color_bias exactly since all scales=1.

Same wall as task074/task080: the conv reads the free 10-channel input directly, so a
uint8 Cast of [1,10,30,30] = 9000B new plane dwarfs the freed output plane -> NET LOSS.
Refutation control.
"""
import importlib
import numpy as np
from onnx import TensorProto, helper, numpy_helper
from src.harness import load_task


def build(task):
    inc = importlib.import_module('src.custom.task383')
    m = inc.build(load_task(383))
    g = m.graph

    conv = next(n for n in g.node if n.op_type == 'Conv' and 'color_f' in n.output)
    Wn, Bn = conv.input[1], (conv.input[2] if len(conv.input) > 2 else None)
    W = [numpy_helper.to_array(i) for i in g.initializer if i.name == Wn][0]
    W_i8 = W.astype(np.int8)
    dil = None
    for a in conv.attribute:
        if a.name == 'dilations':
            dil = list(a.ints)

    new_nodes = [n for n in g.node if n is not conv]

    inits = [
        numpy_helper.from_array(np.float32(1.0), 'q_scale_1'),
        numpy_helper.from_array(np.uint8(0), 'q_zp_u8'),
        numpy_helper.from_array(np.int8(0), 'q_zp_i8'),
        numpy_helper.from_array(W_i8, 'q_W_i8'),
    ]
    qin = ['q383_x_u8', 'q_scale_1', 'q_zp_u8',
           'q_W_i8', 'q_scale_1', 'q_zp_i8',
           'q_scale_1', 'q_zp_u8']
    if Bn is not None:
        B = [numpy_helper.to_array(i) for i in g.initializer if i.name == Bn][0]
        inits.append(numpy_helper.from_array(np.round(B).astype(np.int32), 'q_B_i32'))
        qin.append('q_B_i32')
    g.initializer.extend(inits)

    kwargs = {}
    if dil:
        kwargs['dilations'] = dil
    quant_nodes = [
        helper.make_node('Cast', ['input'], ['q383_x_u8'], to=TensorProto.UINT8),
        helper.make_node('QLinearConv', qin, ['color_f'], **kwargs),  # color_f now uint8
    ]

    del g.node[:]
    g.node.extend(quant_nodes + new_nodes)
    del g.value_info[:]
    return m
