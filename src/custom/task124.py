"""Task 124 — periodic foreground mask with centered runtime-colour renderer.

The top five foreground rows are centered once as uint8 0/2 and reused for
detection, a rank-4 compact source-row bank, and rendering.  Two bounded
QLinearMatMul row fingerprints replace the ten-cell equality reduction, and
five rank-preserving dynamic Slices avoid flat row carriers.  With input
zero-point 1, centered background/foreground are -1/+1 while implicit Conv
padding is 0.  Runtime ScatterElements weights at zero-point 1 are -1 for
background, +1 for the selected colour, and 0 for every other colour, so the
padded QLinearConv writes the FREE uint8 one-hot output directly.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper
from ._exact import arr_b64, model, tensor


def build(task):
    inits = [
        tensor('ch0_starts', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDQsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==')),
        tensor('ch0_ends', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDQsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoBAAAAAAAAAAEAAAAAAAAABQAAAAAAAAAKAAAAAAAAAA==')),
        tensor('row02_idx', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAIAAAAAAAAA')),
        tensor('shift_kernel', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAr//////////wEAAAAAAAAA')),
        tensor('p3b_idx', np.asarray([4], dtype=np.int64)),
        tensor('three_i64', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoDAAAAAAAAAA==')),
        tensor('source_offsets', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDQsIDUsIDEpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoaAAAACAAAABoAAAAIAAAAGgAAABgAAAAFAAAAFwAAAAQAAAAWAAAAFgAAAAIAAAAUAAAAAAAAABIAAAAkAAAACAAAABoAAAAkAAAACAAAAA==')),
        tensor('false8', np.zeros((1, 1, 1, 8), dtype=np.uint8)),
        tensor('ng_mask_base', np.asarray([0] + [1] * 9, dtype=np.uint8).reshape(10, 1, 1, 1)),
        tensor('ng_two', np.asarray([[[[2]]]], dtype=np.uint8)),
        tensor('ng_scale', np.asarray(1.0, dtype=np.float32)),
        tensor('ng_x_zero_point', np.asarray(1, dtype=np.uint8)),
        tensor('ng_y_zero_point', np.asarray(0, dtype=np.uint8)),
        tensor('slice_start_shape', np.asarray([1], dtype=np.int64)),
        tensor('ten_i32', np.asarray([10], dtype=np.int32)),
        tensor('slice_axes', np.asarray([3], dtype=np.int32)),
        tensor('slice_steps', np.asarray([1], dtype=np.int32)),
        tensor(
            'hash_code',
            np.asarray([1, 2, 4, 8, 16, 32, 0, 0, 0, 0], dtype=np.uint8).reshape(10, 1),
        ),
    ]
    nodes = [
        helper.make_node('Slice', ['input', 'ch0_starts', 'ch0_ends'], ['ch0_first5']),
        helper.make_node('Less', ['ch0_first5', 'ng_scale'], ['fg5']),
        helper.make_node('Where', ['fg5', 'ng_two', 'ng_y_zero_point'], ['centered5']),
        helper.make_node('Gather', ['centered5', 'row02_idx'], ['fg02_u8'], axis=2),
        helper.make_node('ArgMax', ['fg02_u8'], ['left_cols'], axis=3, keepdims=0),
        helper.make_node('MatMul', ['left_cols', 'shift_kernel'], ['shift']),
        helper.make_node('Gather', ['centered5', 'slice_steps'], ['p3_rows_a'], axis=2),
        helper.make_node('Gather', ['centered5', 'p3b_idx'], ['p3_rows_b'], axis=2),
        helper.make_node(
            'QLinearMatMul',
            [
                'p3_rows_a', 'ng_scale', 'ng_y_zero_point',
                'hash_code', 'ng_scale', 'ng_y_zero_point',
                'ng_scale', 'ng_y_zero_point',
            ],
            ['p3_hash_a'],
        ),
        helper.make_node(
            'QLinearMatMul',
            [
                'p3_rows_b', 'ng_scale', 'ng_y_zero_point',
                'hash_code', 'ng_scale', 'ng_y_zero_point',
                'ng_scale', 'ng_y_zero_point',
            ],
            ['p3_hash_b'],
        ),
        helper.make_node('Equal', ['p3_hash_a', 'p3_hash_b'], ['is_p3']),
        helper.make_node('Where', ['is_p3', 'three_i64', 'shift'], ['candidate']),
        helper.make_node('Gather', ['source_offsets', 'candidate'], ['source_offset'], axis=0),
        helper.make_node(
            'Split', ['source_offset'],
            ['bottom_start4_0', 'bottom_start4_1', 'bottom_start4_2', 'bottom_start4_3', 'bottom_start4_4'],
            axis=4, split=[1, 1, 1, 1, 1],
        ),
        helper.make_node('Reshape', ['bottom_start4_0', 'slice_start_shape'], ['bottom_start_0']),
        helper.make_node('Add', ['bottom_start_0', 'ten_i32'], ['bottom_end_0']),
        helper.make_node('Reshape', ['bottom_start4_1', 'slice_start_shape'], ['bottom_start_1']),
        helper.make_node('Add', ['bottom_start_1', 'ten_i32'], ['bottom_end_1']),
        helper.make_node('Reshape', ['bottom_start4_2', 'slice_start_shape'], ['bottom_start_2']),
        helper.make_node('Add', ['bottom_start_2', 'ten_i32'], ['bottom_end_2']),
        helper.make_node('Reshape', ['bottom_start4_3', 'slice_start_shape'], ['bottom_start_3']),
        helper.make_node('Add', ['bottom_start_3', 'ten_i32'], ['bottom_end_3']),
        helper.make_node('Reshape', ['bottom_start4_4', 'slice_start_shape'], ['bottom_start_4']),
        helper.make_node('Add', ['bottom_start_4', 'ten_i32'], ['bottom_end_4']),
        helper.make_node('Split', ['fg02_u8'], ['row0_4d', 'row2_4d'], axis=2, split=[1, 1]),
        helper.make_node(
            'Concat', ['false8', 'row0_4d', 'false8', 'p3_rows_a', 'row2_4d'],
            ['fg_pad4d'], axis=3,
        ),
        helper.make_node('Slice', ['fg_pad4d', 'bottom_start_0', 'bottom_end_0', 'slice_axes', 'slice_steps'], ['bottom_row_0']),
        helper.make_node('Slice', ['fg_pad4d', 'bottom_start_1', 'bottom_end_1', 'slice_axes', 'slice_steps'], ['bottom_row_1']),
        helper.make_node('Slice', ['fg_pad4d', 'bottom_start_2', 'bottom_end_2', 'slice_axes', 'slice_steps'], ['bottom_row_2']),
        helper.make_node('Slice', ['fg_pad4d', 'bottom_start_3', 'bottom_end_3', 'slice_axes', 'slice_steps'], ['bottom_row_3']),
        helper.make_node('Slice', ['fg_pad4d', 'bottom_start_4', 'bottom_end_4', 'slice_axes', 'slice_steps'], ['bottom_row_4']),
        helper.make_node(
            'Concat', ['bottom_row_0', 'bottom_row_1', 'bottom_row_2', 'bottom_row_3', 'bottom_row_4'],
            ['bottom_fg'], axis=2,
        ),
        helper.make_node('Concat', ['centered5', 'bottom_fg'], ['ng_centered_mask'], axis=2),
        helper.make_node('ReduceMax', ['input'], ['present_f'], axes=[2, 3], keepdims=1),
        helper.make_node('ArgMax', ['present_f'], ['color_idx_i64'], axis=1, keepdims=1, select_last_index=1),
        helper.make_node(
            'ScatterElements', ['ng_mask_base', 'color_idx_i64', 'ng_two'], ['ng_wmask'],
            name='ng_wmask', axis=0,
        ),
        helper.make_node(
            'QLinearConv',
            [
                'ng_centered_mask', 'ng_scale', 'ng_x_zero_point',
                'ng_wmask', 'ng_scale', 'ng_x_zero_point',
                'ng_scale', 'ng_y_zero_point',
            ],
            ['output'],
            name='output',
            pads=[0, 0, 20, 20],
        ),
    ]
    value_infos = [
        helper.make_tensor_value_info('bottom_row_0', TensorProto.UINT8, [1, 1, 1, 10]),
        helper.make_tensor_value_info('bottom_row_1', TensorProto.UINT8, [1, 1, 1, 10]),
        helper.make_tensor_value_info('bottom_row_2', TensorProto.UINT8, [1, 1, 1, 10]),
        helper.make_tensor_value_info('bottom_row_3', TensorProto.UINT8, [1, 1, 1, 10]),
        helper.make_tensor_value_info('bottom_row_4', TensorProto.UINT8, [1, 1, 1, 10]),
        helper.make_tensor_value_info('fg_pad4d', TensorProto.UINT8, [1, 1, 1, 46]),
        helper.make_tensor_value_info('bottom_fg', TensorProto.UINT8, [1, 1, 5, 10]),
    ]
    result = model(
        'task124_live_exact', nodes, inits,
        output_dtype=TensorProto.UINT8, opset=12, value_infos=value_infos,
    )
    return onnx.shape_inference.infer_shapes(result, strict_mode=True)
