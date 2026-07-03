"""Task080 candidate — QLinearConv on the colour-index Conv (S10 recipe).

Same PRODUCER_BOUND / RANKING_ONLY setup as task074_qconv: Conv(input[1,10,30,30],
KW[1,10,1,1]) -> colf32[1,1,30,30] fp32 (3600B), colour label 0..9, consumers are
order-only. int8 QLinearConv is bit-exact on the values, BUT the conv reads the free
10-channel input directly, so quantizing it costs a NEW 9000B uint8 plane vs 2700B
freed on the output -> NET LOSS. Refutation control.
"""
import importlib
import numpy as np
from onnx import TensorProto, helper, numpy_helper
from src.harness import load_task


def build(task):
    inc = importlib.import_module('src.custom.task080')
    m = inc.build(load_task(80))
    g = m.graph

    KW = [numpy_helper.to_array(i) for i in g.initializer if i.name == 'KW'][0]
    W_i8 = KW.astype(np.int8)

    new_nodes = [n for n in g.node if not (n.op_type == 'Conv' and 'colf32' in n.output)]

    g.initializer.extend([
        numpy_helper.from_array(np.float32(1.0), 'q_scale_1'),
        numpy_helper.from_array(np.uint8(0), 'q_zp_u8'),
        numpy_helper.from_array(np.int8(0), 'q_zp_i8'),
        numpy_helper.from_array(W_i8, 'q_KW_i8'),
    ])

    quant_nodes = [
        helper.make_node('Cast', ['input'], ['q080_x_u8'], to=TensorProto.UINT8),
        helper.make_node('QLinearConv',
                         ['q080_x_u8', 'q_scale_1', 'q_zp_u8',
                          'q_KW_i8', 'q_scale_1', 'q_zp_i8',
                          'q_scale_1', 'q_zp_u8'],
                         ['colf32']),  # colf32 now uint8
    ]

    del g.node[:]
    g.node.extend(quant_nodes + new_nodes)
    del g.value_info[:]  # stale fp32 value_info for colf32 & descendants -> let ORT reinfer
    return m
