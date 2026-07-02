"""Task 163 candidate — free-input scalar einsum locate (drop 484B yellow11 Slice plane).

Rule (6d0160f0): 11x11 hollywood-squares; exactly ONE yellow(4) cell in the input.
Incumbent pays: yellow11 [1,1,11,11] fp32 Slice (484B) + row_has/col_has [11] (44B each)
+ 2x ArgMax int64 (8B each) to recover the yellow position (yr, yc).

Candidate: single yellow ==> Einsum('bchw,c,h->', input, eye[4], arange30) = yr exactly
(integer <= 29, fp32-exact). Same with w for yc. Cast fp32->int32 (truncation exact).
Downstream graph (Mod/Sub/dest_bank Gathers/patch Slice/label chain) is byte-identical
to the incumbent.
"""
import numpy as np
from onnx import TensorProto, helper
from ._exact import arr_b64, model, tensor


def build(task):
    inits = [
        tensor('yellow_sel', np.eye(10, dtype=np.float32)[4]),
        tensor('arange30', np.arange(30, dtype=np.float32)),
        tensor('four_i64', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoEAAAA')),
        tensor('dest_bank', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDMsIDExKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAQAAAAIAAAAEAAAAAwAAAAMAAAADAAAABAAAAAMAAAADAAAAAwAAAAMAAAADAAAAAwAAAAQAAAAAAAAAAQAAAAIAAAAEAAAAAwAAAAMAAAADAAAAAwAAAAMAAAADAAAABAAAAAMAAAADAAAAAwAAAAQAAAAAAAAAAQAAAAIAAAA=')),
        tensor('zero1_i64', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAA')),
        tensor('one1_i64', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoBAAAA')),
        tensor('src_extents', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDQsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoBAAAACQAAAAMAAAADAAAA')),
        tensor('label_w', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDksIDEsIDEpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAIA/AAAAQAAAQEAAAIBAAACgQAAAwEAAAOBAAAAAQQAAEEE=')),
        tensor('five_u8', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnfHUxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoF')),
        tensor('gray5', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnfGIxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEsIDUsIDUpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAQAAAAABAAAAAAEAAAAAAQEBAQEB')),
        tensor('palette', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnfHUxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEwLCAxLCAxKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAQIDBAUGBwgJ')),
        tensor('ten_u8', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnfHUxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoK')),
        tensor('pad_table5', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDgsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAAAAAAAAAIAAAAAAAAA')),
        tensor('pad_output', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDgsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAATAAAAAAAAABMAAAAAAAAA')),
    ]
    nodes = [
        helper.make_node('Einsum', ['input', 'yellow_sel', 'arange30'], ['yr_f'],
                         equation='bchw,c,h->'),
        helper.make_node('Einsum', ['input', 'yellow_sel', 'arange30'], ['yc_f'],
                         equation='bchw,c,w->'),
        helper.make_node('Cast', ['yr_f'], ['row_idx'], to=6),
        helper.make_node('Cast', ['yc_f'], ['col_idx'], to=6),
        helper.make_node('Mod', ['row_idx', 'four_i64'], ['row_local'], fmod=0),
        helper.make_node('Sub', ['row_idx', 'row_local'], ['src_row0']),
        helper.make_node('Mod', ['col_idx', 'four_i64'], ['col_local'], fmod=0),
        helper.make_node('Sub', ['col_idx', 'col_local'], ['src_col0']),
        helper.make_node('Gather', ['dest_bank', 'row_local'], ['dst_rows'], axis=0),
        helper.make_node('Gather', ['dest_bank', 'col_local'], ['dst_cols'], axis=0),
        helper.make_node('Unsqueeze', ['src_row0'], ['src_row0v'], axes=[0]),
        helper.make_node('Unsqueeze', ['src_col0'], ['src_col0v'], axes=[0]),
        helper.make_node('Concat', ['zero1_i64', 'one1_i64', 'src_row0v', 'src_col0v'], ['src_starts'], axis=0),
        helper.make_node('Add', ['src_starts', 'src_extents'], ['src_ends']),
        helper.make_node('Slice', ['input', 'src_starts', 'src_ends'], ['patch_f']),
        helper.make_node('Conv', ['patch_f', 'label_w'], ['label_f'], kernel_shape=[1, 1]),
        helper.make_node('Cast', ['label_f'], ['label_patch'], to=2),
        helper.make_node('Pad', ['label_patch', 'pad_table5'], ['label_table_black'], mode='constant'),
        helper.make_node('Where', ['gray5', 'five_u8', 'label_table_black'], ['label_table']),
        helper.make_node('Gather', ['label_table', 'dst_rows'], ['label_rows'], axis=2),
        helper.make_node('Gather', ['label_rows', 'dst_cols'], ['label11'], axis=3),
        helper.make_node('Pad', ['label11', 'pad_output', 'ten_u8'], ['label30'], mode='constant'),
        helper.make_node('Equal', ['label30', 'palette'], ['output']),
    ]
    value_infos = [
        helper.make_tensor_value_info('patch_f', 1, [1, 9, 3, 3]),
        helper.make_tensor_value_info('label_f', 1, [1, 1, 3, 3]),
        helper.make_tensor_value_info('label_patch', 2, [1, 1, 3, 3]),
        helper.make_tensor_value_info('label_table_black', 2, [1, 1, 5, 5]),
        helper.make_tensor_value_info('label_table', 2, [1, 1, 5, 5]),
        helper.make_tensor_value_info('label_rows', 2, [1, 1, 11, 5]),
        helper.make_tensor_value_info('label11', 2, [1, 1, 11, 11]),
        helper.make_tensor_value_info('label30', 2, [1, 1, 30, 30]),
    ]
    return model('task163_einsum_locate', nodes, inits, output_dtype=9, opset=12, value_infos=value_infos)
