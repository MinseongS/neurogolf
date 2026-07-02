"""Task 182 — overlay rewrite (free passthrough + painted foreground).

Rule (ARC 776ffc46): the input contains several shapes; the one enclosed in a
gray 7x7 box is pre-colored to a target color (2 or 3).  Every blue(1) shape
congruent to that boxed template is recolored to the target color; everything
else is copied verbatim.

This graph keeps the original box-detection / template-extraction / template-
matching machinery (a 5x5 template kernel is read out of the box and correlated
against the blue plane with a QLinearConv, then dilated and intersected with the
blue mask to recover the matched footprints).  The tail is rewritten with the
overlay technique: instead of reconstructing the whole grid, we start from the
FREE "input" passthrough and only paint the matched cells with a tiny per-color
one-hot vector via Where(mask, color_onehot, input).
"""
import numpy as np
from onnx import TensorProto, helper
from ._exact import arr_b64, model, tensor


def build(task):
    inits = [
        tensor('z_u8', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnfHUxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoA')),
        tensor('one_u8', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnfHUxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoB')),
        tensor('cw', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEwLCAyLCAyKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAEBAAAAAAAAAAAAAAAAAAACAQAAAAAAAAAAAAAAAAAAAoEAAAAAAAAAAAAAAAAAAAMBAAAAAAAAAAAAAAAAAAADgQAAAAAAAAAAAAAAAAAAAAEEAAAAAAAAAAAAAAAAAABBBAAAAAAAAAAAAAAAA')),
        tensor('sc1', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoBAAAAAAAAAA==')),
        tensor('stride', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoUAAAAAAAAAA==')),
        tensor('offs', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDI1LCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoVAAAAAAAAABYAAAAAAAAAFwAAAAAAAAAYAAAAAAAAABkAAAAAAAAAKQAAAAAAAAAqAAAAAAAAACsAAAAAAAAALAAAAAAAAAAtAAAAAAAAAD0AAAAAAAAAPgAAAAAAAAA/AAAAAAAAAEAAAAAAAAAAQQAAAAAAAABRAAAAAAAAAFIAAAAAAAAAUwAAAAAAAABUAAAAAAAAAFUAAAAAAAAAZQAAAAAAAABmAAAAAAAAAGcAAAAAAAAAaAAAAAAAAABpAAAAAAAAAA==')),
        tensor('flat_shape', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAqQAQAAAAAAAA==')),
        tensor('ax0', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAA==')),
        tensor('kshape', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDQsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoBAAAAAAAAAAEAAAAAAAAABQAAAAAAAAAFAAAAAAAAAA==')),
        tensor('cen', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoMAAAAAAAAAA==')),
        tensor('yzp_i32', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAqAAAAA')),
        tensor('xsc', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAIA/')),
        tensor('yzp', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnfHUxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAqA')),
        # pad [1,1,20,20] -> [1,1,30,30] (append 10 rows/cols at the end)
        tensor('pads20', np.array([0, 0, 0, 0, 0, 0, 10, 10], dtype=np.int64)),
        # color index range [0..9] as uint8, shape [1,10,1,1]
        tensor('color_range', np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)),
    ]
    nodes = [
        # --- box detection: recover per-cell colour index on the 20x20 crop ---
        helper.make_node('Conv', ['input', 'cw'], ['cidf32'], dilations=[10, 10], kernel_shape=[2, 2], pads=[0, 0, 0, 0]),
        helper.make_node('Cast', ['cidf32'], ['cid'], to=2),
        helper.make_node('AveragePool', ['cidf32'], ['rscore'], kernel_shape=[7, 20]),
        helper.make_node('AveragePool', ['cidf32'], ['cscore'], kernel_shape=[20, 7]),
        helper.make_node('ArgMax', ['rscore'], ['ri'], axis=2, keepdims=0),
        helper.make_node('ArgMax', ['cscore'], ['ci'], axis=3, keepdims=0),
        helper.make_node('Reshape', ['ri', 'sc1'], ['rr']),
        helper.make_node('Reshape', ['ci', 'sc1'], ['cc']),
        helper.make_node('Mul', ['rr', 'stride'], ['rstride']),
        helper.make_node('Add', ['rstride', 'cc'], ['fbase']),
        helper.make_node('Add', ['fbase', 'offs'], ['tidx']),
        # --- template extraction: read the 5x5 window inside the box ---
        helper.make_node('Reshape', ['cid', 'flat_shape'], ['cflat']),
        helper.make_node('Gather', ['cflat', 'tidx'], ['tv']),
        helper.make_node('Greater', ['tv', 'z_u8'], ['tmask_b']),
        helper.make_node('Cast', ['tmask_b'], ['tmask_u8'], to=2),
        helper.make_node('Cast', ['tmask_u8'], ['tmask_i32'], to=6),
        helper.make_node('ReduceSum', ['tmask_i32', 'ax0'], ['nshape_i32'], keepdims=1),
        helper.make_node('Add', ['tmask_u8', 'tmask_u8'], ['kflat_u8']),
        helper.make_node('Reshape', ['kflat_u8', 'kshape'], ['kern']),
        helper.make_node('Gather', ['tv', 'cen'], ['enc']),
        helper.make_node('Add', ['nshape_i32', 'yzp_i32'], ['thr_i32']),
        helper.make_node('Cast', ['thr_i32'], ['thr_u8'], to=2),
        # --- template match on the blue plane ---
        helper.make_node('Equal', ['cid', 'one_u8'], ['c1_b']),
        helper.make_node('Cast', ['c1_b'], ['c1_u8'], to=2),
        helper.make_node('QLinearConv', ['c1_u8', 'xsc', 'z_u8', 'kern', 'xsc', 'one_u8', 'xsc', 'yzp'], ['score'], kernel_shape=[5, 5], pads=[2, 2, 2, 2]),
        helper.make_node('MaxPool', ['score'], ['score_dil'], kernel_shape=[5, 5], pads=[2, 2, 2, 2]),
        helper.make_node('GreaterOrEqual', ['score_dil', 'thr_u8'], ['dil_b']),
        helper.make_node('And', ['dil_b', 'c1_b'], ['recolor_b']),
        # --- overlay tail: paint only the matched cells with the target colour ---
        helper.make_node('Pad', ['recolor_b', 'pads20'], ['mask30'], mode='constant'),
        helper.make_node('Equal', ['color_range', 'enc'], ['color_oh_b']),
        helper.make_node('Cast', ['color_oh_b'], ['color_oh'], to=1),
        helper.make_node('Where', ['mask30', 'color_oh', 'input'], ['output']),
    ]
    value_infos = [
        helper.make_tensor_value_info('cidf32', 1, [1, 1, 20, 20]),
        helper.make_tensor_value_info('cid', 2, [1, 1, 20, 20]),
        helper.make_tensor_value_info('rscore', 1, [1, 1, 14, 1]),
        helper.make_tensor_value_info('cscore', 1, [1, 1, 1, 14]),
        helper.make_tensor_value_info('ri', 7, [1, 1, 1]),
        helper.make_tensor_value_info('ci', 7, [1, 1, 1]),
        helper.make_tensor_value_info('rr', 7, [1]),
        helper.make_tensor_value_info('cc', 7, [1]),
        helper.make_tensor_value_info('rstride', 7, [1]),
        helper.make_tensor_value_info('fbase', 7, [1]),
        helper.make_tensor_value_info('tidx', 7, [25]),
        helper.make_tensor_value_info('cflat', 2, [400]),
        helper.make_tensor_value_info('tv', 2, [25]),
        helper.make_tensor_value_info('tmask_b', 9, [25]),
        helper.make_tensor_value_info('tmask_u8', 2, [25]),
        helper.make_tensor_value_info('tmask_i32', 6, [25]),
        helper.make_tensor_value_info('nshape_i32', 6, [1]),
        helper.make_tensor_value_info('kflat_u8', 2, [25]),
        helper.make_tensor_value_info('kern', 2, [1, 1, 5, 5]),
        helper.make_tensor_value_info('enc', 2, [1]),
        helper.make_tensor_value_info('thr_i32', 6, [1]),
        helper.make_tensor_value_info('thr_u8', 2, [1]),
        helper.make_tensor_value_info('c1_b', 9, [1, 1, 20, 20]),
        helper.make_tensor_value_info('c1_u8', 2, [1, 1, 20, 20]),
        helper.make_tensor_value_info('score', 2, [1, 1, 20, 20]),
        helper.make_tensor_value_info('score_dil', 2, [1, 1, 20, 20]),
        helper.make_tensor_value_info('dil_b', 9, [1, 1, 20, 20]),
        helper.make_tensor_value_info('recolor_b', 9, [1, 1, 20, 20]),
        helper.make_tensor_value_info('mask30', 9, [1, 1, 30, 30]),
        helper.make_tensor_value_info('color_oh_b', 9, [1, 10, 1, 1]),
        helper.make_tensor_value_info('color_oh', 1, [1, 10, 1, 1]),
    ]
    return model('task182_overlay', nodes, inits, output_dtype=1, opset=17, value_infos=value_infos)
