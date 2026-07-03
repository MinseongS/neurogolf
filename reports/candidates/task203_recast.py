"""Task 203 recast candidate — fp16 the count-arithmetic chain.

Scan flagged maxc/pair/target/target_col as U8_CANDIDATE (max 72 on bundled), but
`cnt` = per-colour pixel count can reach ~900 on a fresh 30x30 grid (one colour
filling most cells), so uint8 (<=255) would overflow -> use fp16 (integers <=2048
exact). Values here: cnt<=900, maxc<=900, pair=maxc+4<=904, target=pair-cnt in
[4,904]; all fp16-exact -> Equal / ArgMax / Gather bit-identical.

`cnt` itself is fp32 (ReduceSum consumes the free input directly = PRODUCER_BOUND),
so add one `cnt16` Cast (20B) and route the fp16 arithmetic + the Equal off it.
Net: target 40->20, target_col 40->20, maxc/pair 4->2 each, +cnt16 20 => ~-24B.
"""
import numpy as np
from onnx import TensorProto, helper
from ._exact import arr_b64, model, tensor

F16 = TensorProto.FLOAT16


def build(task):
    inits = [
        tensor('four', np.float16(4.0)),
    ]
    nodes = [
        helper.make_node('ReduceSum', ['input'], ['cnt'], axes=[2, 3], keepdims=0),
        helper.make_node('Cast', ['cnt'], ['cnt16'], to=10),
        helper.make_node('ReduceMax', ['cnt16'], ['maxc'], keepdims=1),
        helper.make_node('Add', ['maxc', 'four'], ['pair']),
        helper.make_node('Sub', ['pair', 'cnt16'], ['target']),
        helper.make_node('Transpose', ['target'], ['target_col'], perm=[1, 0]),
        helper.make_node('Equal', ['target_col', 'cnt16'], ['match_b']),
        helper.make_node('Cast', ['match_b'], ['match_u8'], to=2),
        helper.make_node('ArgMax', ['match_u8'], ['remap'], axis=1, keepdims=0),
        helper.make_node('Gather', ['input', 'remap'], ['output'], axis=1),
    ]
    return model('task203_recast', nodes, inits, output_dtype=1, opset=12)
