"""Task 338 — memory-optimized label-space graph.

Rule (generator task_d5d6de2d): red rectangular box borders on a black grid.
Every black cell strictly interior to a box becomes green (3); everything else
stays background (0). Interior is detected by ray-casting: count horizontal red
walls above each cell (vertical cumulative sum of a 1x3 red-run detector); odd
parity => interior. Off-grid padded cells decode to all-zero via sentinel 99.

Memory optimization over the prior exact reconstruction:
  * Grid extent is thresholded on the [1,1,30,x] ReduceMax planes and cropped to
    25 in bool space (rowb30/colb30 -> rowb/colb), replacing the fp32 rowp/colp
    slices with cheaper bool tensors. This trims the peak intermediate memory
    from 10140 to 10000 bytes while leaving every train/test/arc-gen output
    bit-identical.

Note: grid detection must reduce the full fp32 `input` (not just the sliced red
channel) because in-grid background cells only light up one-hot channel 0, which
the red-only plane does not see.
"""
import numpy as np
from onnx import helper

from ._exact import arr_b64, model, tensor


def build(task):
    inits = [
        # slice red channel (index 2), spatial 0:25,0:25 -> [1,1,25,25] fp32
        tensor('r_s', np.array([2, 0, 0], dtype=np.int64)),
        tensor('r_e', np.array([3, 25, 25], dtype=np.int64)),
        tensor('r_ax', np.array([1, 2, 3], dtype=np.int64)),
        tensor('q8', np.array([0], dtype=np.uint8)),
        tensor('two8', np.array([2], dtype=np.uint8)),
        tensor('qscale', np.array([1.0], dtype=np.float32)),
        tensor('yscale5', np.array([5.0], dtype=np.float32)),
        tensor('red3_w_u8', np.ones((1, 1, 1, 3), dtype=np.uint8)),
        tensor('three8', np.array([3], dtype=np.uint8)),
        tensor('z1', np.array([0], dtype=np.int64)),
        tensor('wn', np.array([25], dtype=np.int64)),
        tensor('a2', np.array([2], dtype=np.int64)),
        tensor('a3', np.array([3], dtype=np.int64)),
        # lower-triangular ones matrix: vertical cumulative sum via matmul
        tensor('Tl', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnfHUxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEsIDI1LCAyNSksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAQEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQEBAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAEBAQEBAAAAAAAAAAAAAAAAAAAAAAAAAAABAQEBAQEAAAAAAAAAAAAAAAAAAAAAAAAAAQEBAQEBAQAAAAAAAAAAAAAAAAAAAAAAAAEBAQEBAQEBAAAAAAAAAAAAAAAAAAAAAAABAQEBAQEBAQEAAAAAAAAAAAAAAAAAAAAAAQEBAQEBAQEBAQAAAAAAAAAAAAAAAAAAAAEBAQEBAQEBAQEBAAAAAAAAAAAAAAAAAAABAQEBAQEBAQEBAQEAAAAAAAAAAAAAAAAAAQEBAQEBAQEBAQEBAQAAAAAAAAAAAAAAAAEBAQEBAQEBAQEBAQEBAAAAAAAAAAAAAAABAQEBAQEBAQEBAQEBAQEAAAAAAAAAAAAAAQEBAQEBAQEBAQEBAQEBAQAAAAAAAAAAAAEBAQEBAQEBAQEBAQEBAQEBAAAAAAAAAAABAQEBAQEBAQEBAQEBAQEBAQEAAAAAAAAAAQEBAQEBAQEBAQEBAQEBAQEBAQAAAAAAAAEBAQEBAQEBAQEBAQEBAQEBAQEBAAAAAAABAQEBAQEBAQEBAQEBAQEBAQEBAQEAAAAAAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQAAAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAAABAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEA')),
        tensor('halff', np.array([0.5], dtype=np.float32)),
        tensor('u99', np.array([99], dtype=np.uint8)),
        tensor('pads', np.array([0, 0, 0, 0, 0, 0, 5, 5], dtype=np.int64)),
        tensor('arange', np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)),
    ]
    nodes = [
        helper.make_node('Slice', ['input', 'r_s', 'r_e', 'r_ax'], ['red_f']),
        helper.make_node('Greater', ['red_f', 'halff'], ['redb']),
        helper.make_node('Cast', ['redb'], ['red8'], to=2),
        # 1x3 red-run detector: y_scale 5.0 => requant round(sum*0.2) maps
        # run-sum {0,1,2,3} -> {0,0,0,1}; conv output IS the uint8 wall plane.
        helper.make_node('QLinearConv',
                         ['red8', 'qscale', 'q8', 'red3_w_u8', 'qscale', 'q8', 'yscale5', 'q8'],
                         ['Hm8'], pads=[0, 1, 0, 1]),
        # vertical cumulative count of walls via lower-triangular matmul
        helper.make_node('QLinearMatMul',
                         ['Tl', 'qscale', 'q8', 'Hm8', 'qscale', 'q8', 'qscale', 'q8'],
                         ['cnt']),
        helper.make_node('Mod', ['cnt', 'two8'], ['par']),
        helper.make_node('Greater', ['par', 'q8'], ['encb']),
        # grid extent: any one-hot channel (incl. background ch0) set in a
        # row/col marks the in-grid region. Reduce over channels+space on the
        # fp32 input, threshold, then crop to 25 in bool to stay cheap.
        helper.make_node('ReduceMax', ['input'], ['rowp30'], axes=[1, 3], keepdims=1),
        helper.make_node('ReduceMax', ['input'], ['colp30'], axes=[1, 2], keepdims=1),
        helper.make_node('Greater', ['rowp30', 'halff'], ['rowb30']),
        helper.make_node('Greater', ['colp30', 'halff'], ['colb30']),
        helper.make_node('Slice', ['rowb30', 'z1', 'wn', 'a2'], ['rowb']),
        helper.make_node('Slice', ['colb30', 'z1', 'wn', 'a3'], ['colb']),
        helper.make_node('And', ['rowb', 'colb'], ['gridb']),
        helper.make_node('Where', ['gridb', 'q8', 'u99'], ['Lg']),
        helper.make_node('Where', ['encb', 'three8', 'Lg'], ['Lgrn']),
        helper.make_node('Where', ['redb', 'Lg', 'Lgrn'], ['L25']),
        helper.make_node('Pad', ['L25', 'pads', 'u99'], ['L30'], mode='constant'),
        helper.make_node('Equal', ['L30', 'arange'], ['output']),
    ]
    value_infos = [
        helper.make_tensor_value_info('red_f', 1, [1, 1, 25, 25]),
        helper.make_tensor_value_info('redb', 9, [1, 1, 25, 25]),
        helper.make_tensor_value_info('red8', 2, [1, 1, 25, 25]),
        helper.make_tensor_value_info('Hm8', 2, [1, 1, 25, 25]),
        helper.make_tensor_value_info('cnt', 2, [1, 1, 25, 25]),
        helper.make_tensor_value_info('par', 2, [1, 1, 25, 25]),
        helper.make_tensor_value_info('encb', 9, [1, 1, 25, 25]),
        helper.make_tensor_value_info('rowp30', 1, [1, 1, 30, 1]),
        helper.make_tensor_value_info('colp30', 1, [1, 1, 1, 30]),
        helper.make_tensor_value_info('rowb30', 9, [1, 1, 30, 1]),
        helper.make_tensor_value_info('colb30', 9, [1, 1, 1, 30]),
        helper.make_tensor_value_info('rowb', 9, [1, 1, 25, 1]),
        helper.make_tensor_value_info('colb', 9, [1, 1, 1, 25]),
        helper.make_tensor_value_info('gridb', 9, [1, 1, 25, 25]),
        helper.make_tensor_value_info('Lg', 2, [1, 1, 25, 25]),
        helper.make_tensor_value_info('Lgrn', 2, [1, 1, 25, 25]),
        helper.make_tensor_value_info('L25', 2, [1, 1, 25, 25]),
        helper.make_tensor_value_info('L30', 2, [1, 1, 30, 30]),
    ]
    return model('task338_labelspace', nodes, inits, output_dtype=9, opset=11, value_infos=value_infos)
