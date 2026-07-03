"""Task074 candidate — QLinearConv on the colour-index Conv (S10 recipe).

REFUTATION CONTROL. task074's `color_f` is a PRODUCER_BOUND fp32 plane (dtype_overpay
scan): Conv(input[1,10,30,30], W[1,10,1,1]) -> color_f[1,1,30,30] fp32 (3600B), a 1x1
weighted channel-sum producing the colour label 0..8. Downstream is RANKING_ONLY
(Cast -> Max/Gather -> Equal), so the S10 int8-QLinearConv-order-preserving safety
class *applies to the consumers*.

The catch this build makes concrete: the Conv reads the FREE 10-channel fp32 input
DIRECTLY. QLinearConv requires a quantized (uint8) input, so we must Cast the whole
[1,10,30,30] input to uint8 = a NEW 9000B counted plane. That dwarfs the 2700B freed
on the (now uint8) output plane. Net = +9000 - 2700 = +6300B LOSS. The S10 wins
(264/184/365) fed QLinearConv a SINGLE-channel already-integer plane, so their
quantized-input copy was tiny; here the 10x channel fan-in kills it. This is the same
wall as the fp16-recast refutation in task041_signed.py.
"""
import importlib
import numpy as np
from onnx import TensorProto, helper, numpy_helper
from src.harness import load_task


def build(task):
    inc = importlib.import_module('src.custom.task074')
    m = inc.build(load_task(74))
    g = m.graph

    W = [numpy_helper.to_array(i) for i in g.initializer if i.name == 'W'][0]
    W_i8 = W.astype(np.int8)

    # remove the fp32 Conv node
    new_nodes = [n for n in g.node if not (n.op_type == 'Conv' and 'color_f' in n.output)]

    # new quant initializers
    g.initializer.extend([
        numpy_helper.from_array(np.float32(1.0), 'q_scale_1'),
        numpy_helper.from_array(np.uint8(0), 'q_zp_u8'),
        numpy_helper.from_array(np.int8(0), 'q_zp_i8'),
        numpy_helper.from_array(W_i8, 'q_W_i8'),
    ])

    quant_nodes = [
        helper.make_node('Cast', ['input'], ['q074_x_u8'], to=TensorProto.UINT8),
        helper.make_node('QLinearConv',
                         ['q074_x_u8', 'q_scale_1', 'q_zp_u8',
                          'q_W_i8', 'q_scale_1', 'q_zp_i8',
                          'q_scale_1', 'q_zp_u8'],
                         ['color_f']),  # color_f now uint8
    ]

    del g.node[:]
    g.node.extend(quant_nodes + new_nodes)
    return m
