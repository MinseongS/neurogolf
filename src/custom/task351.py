"""Task 351 candidate — free-input einsum moment locate (drop 576B green12 Slice plane).

Rule (dc0a314f): 16x16 D2-mirrored grid, solid 5x5 GREEN(3) cutout at (row,col),
green nowhere else. Output = mirrored source grid[15-row-i][15-col-j], padded into
30x30 canvas.

Incumbent pays: green12 [1,1,12,12] fp32 Slice (576B) + two ReduceMax profiles
(48B+48B) + ArgMax/Cast/Concat chain, mem 1632 params 19.

Candidate: the marker is a single SOLID 5x5 (exactly 25 green cells, rows
row..row+4). Free-input einsum 'bchw,c,h->b' with weights w17[h] = 17-h gives
  e_r = sum_{green cells} (17-h) = 25*17 - (25*row + 50) = 25*(15-row)
exactly in fp32 (integer, <=425). Same vector on w gives 25*(15-col).
Concat with constant 25.0, Div by 25.0 (exact: integer multiple), Cast int32
-> patch_starts = [1, 15-row, 15-col]. patch_ends = starts + [9,-5,-5].
Downstream reversed Slice + Pad identical to incumbent.
"""
import numpy as np
from onnx import TensorProto, helper
from ._exact import model, tensor


def build(task):
    inits = [
        tensor('green_sel', np.eye(10, dtype=np.float32)[3]),
        tensor('w17', (17.0 - np.arange(30)).astype(np.float32)),
        tensor('k25', np.array([25.0], dtype=np.float32)),
        tensor('c25', np.array(25.0, dtype=np.float32)),
        tensor('end_delta', np.array([9, -5, -5], dtype=np.int32)),
        tensor('patch_axes', np.array([1, 2, 3], dtype=np.int32)),
        tensor('patch_steps', np.array([1, -1, -1], dtype=np.int32)),
        tensor('pads', np.array([0, 1, 0, 0, 0, 0, 25, 25], dtype=np.int64)),
    ]
    nodes = [
        helper.make_node('Einsum', ['input', 'green_sel', 'w17'], ['e_r'],
                         equation='bchw,c,h->b'),
        helper.make_node('Einsum', ['input', 'green_sel', 'w17'], ['e_c'],
                         equation='bchw,c,w->b'),
        helper.make_node('Concat', ['k25', 'e_r', 'e_c'], ['rc25'], axis=0),
        helper.make_node('Div', ['rc25', 'c25'], ['rc_f']),
        helper.make_node('Cast', ['rc_f'], ['patch_starts'], to=TensorProto.INT32),
        helper.make_node('Add', ['patch_starts', 'end_delta'], ['patch_ends']),
        helper.make_node('Slice', ['input', 'patch_starts', 'patch_ends',
                                   'patch_axes', 'patch_steps'], ['answer_patch']),
        helper.make_node('Pad', ['answer_patch', 'pads'], ['output'], mode='constant'),
    ]
    value_infos = [
        helper.make_tensor_value_info('answer_patch', TensorProto.FLOAT, [1, 9, 5, 5]),
    ]
    return model('task351_moment_slice', nodes, inits, output_dtype=TensorProto.FLOAT,
                 opset=12, value_infos=value_infos)
