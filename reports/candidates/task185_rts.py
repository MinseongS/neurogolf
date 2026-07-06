"""Task 185 — mechanism-16 (runtime-parameterized stamp) redesign candidate.

REPORT-ONLY measurement artifact. Does NOT touch networks/ or src/custom.

Incumbent (src/custom/task185.py) materializes a 3-way candidate BANK of
single-tap strided convs over the FREE one-hot input:
  - p3: Conv grid3_kernel[1,10,4,4] stride3          -> [1,1,9,9]  (tap at 3i+2)
  - p4: Conv grid4_kernel[1,10,2,2] stride4 dil3     -> [1,1,7,7]  (tap at 4i+3)
  - p5: Conv grid4_kernel[1,10,2,2] stride5 dil4     -> [1,1,6,6]  (tap at 5i+4)
then pads p4/p5 and muxes (2x Less + 2x Where) on the global line-code scalar.
Each conv reads the free input to a TINY output = detection floor, ~200B each.

mechanism-16 says: detect the global discrete parameter p first, then assemble
ONE runtime path instead of the muxed bank. Because stride/dilation are STATIC
attrs (cannot be runtime-selected), the strided conv must be replaced by a
runtime GATHER at computed indices idx = arange*p + (p-1).  But Gather needs a
source plane to read colours from, and the only source is a FULL-RESOLUTION
colour plane Conv(input, ramp[1,10,1,1]) -> [1,1,30,30] fp32 = 3600B.  That single
plane already exceeds the ENTIRE incumbent graph (1651B mem).  This build measures
that cost to confirm the structural KILL.
"""
import numpy as np
from onnx import TensorProto, helper
from ._exact import arr_b64, model, tensor


def build(task):
    inits = [
        # --- reused incumbent constants (copied b64) ---
        tensor('zero_u8', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnfHUxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoA')),
        tensor('one_u8', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnfHUxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoB')),
        tensor('ten_f32', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAACBB')),
        tensor('twenty_f32', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAKBB')),
        tensor('idx4', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDQsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAQAAAAIAAAADAAAA')),
        tensor('axes_hw', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoCAAAAAAAAAAMAAAAAAAAA')),
        tensor('starts_tl', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAAAAAAAAAAA')),
        tensor('ends_tl', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoDAAAAAAAAAAMAAAAAAAAA')),
        tensor('starts_tr', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAEAAAAAAAAA')),
        tensor('ends_tr', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoDAAAAAAAAAAQAAAAAAAAA')),
        tensor('starts_bl', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoBAAAAAAAAAAAAAAAAAAAA')),
        tensor('ends_bl', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoEAAAAAAAAAAMAAAAAAAAA')),
        tensor('starts_br', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoBAAAAAAAAAAEAAAAAAAAA')),
        tensor('ends_br', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoEAAAAAAAAAAQAAAAAAAAA')),
        tensor('colors10_u8', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnfHUxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEwLCAxLCAxKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAQIDBAUGBwgJ')),
        tensor('pad_output', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDgsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAbAAAAAAAAABsAAAAAAAAA')),
        tensor('line_kernel', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEwLCA1LCAxKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAPwAAMEEAAKhBAAAAAAAAAAAAAABAAABAQQAAsEEAAAAAAAAAAAAAQEAAAFBBAAC4QQAAAAAAAAAAAACAQAAAYEEAAMBBAAAAAAAAAAAAAKBAAABwQQAAyEEAAAAAAAAAAAAAwEAAAIBBAADQQQAAAAAAAAAAAADgQAAAiEEAANhBAAAAAAAAAAAAAABBAACQQQAA4EEAAAAAAAAAAAAAEEEAAJhBAADoQQ==')),
        # --- new constants for the runtime path ---
        # colour ramp: channel c -> value c, a 1x1 conv -> full 30x30 colour plane
        tensor('color_ramp', np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)),
        tensor('arange10', np.arange(10, dtype=np.int64)),
        tensor('p3_i', np.array(3, dtype=np.int64)),
        tensor('p4_i', np.array(4, dtype=np.int64)),
        tensor('p5_i', np.array(5, dtype=np.int64)),
        tensor('one_i', np.array(1, dtype=np.int64)),
        tensor('clamp29', np.array(29, dtype=np.int64)),
        tensor('shape_row', np.array([1, 1, 10, 1], dtype=np.int64)),
        tensor('shape_col', np.array([1, 1, 1, 10], dtype=np.int64)),
    ]
    nodes = [
        # global line-code scalar (encodes colour in mod 10 + spacing regime in tens)
        helper.make_node('Conv', ['input', 'line_kernel'], ['line_code_f32'], kernel_shape=[5, 1], strides=[30, 30]),
        helper.make_node('ReduceMax', ['line_code_f32'], ['line_scalar'], keepdims=0),
        # detect the global discrete period p (mechanism-16 step 1)
        helper.make_node('Less', ['line_scalar', 'ten_f32'], ['is_p3']),
        helper.make_node('Less', ['line_scalar', 'twenty_f32'], ['is_le4']),
        helper.make_node('Where', ['is_le4', 'p4_i', 'p5_i'], ['p_45']),
        helper.make_node('Where', ['is_p3', 'p3_i', 'p_45'], ['p_i']),
        helper.make_node('Sub', ['p_i', 'one_i'], ['pm1_i']),
        # runtime intersection indices  idx = arange*p + (p-1)
        helper.make_node('Mul', ['arange10', 'p_i'], ['idx_raw0']),
        helper.make_node('Add', ['idx_raw0', 'pm1_i'], ['idx_raw']),
        helper.make_node('Min', ['idx_raw', 'clamp29'], ['idx']),
        helper.make_node('Greater', ['idx_raw', 'clamp29'], ['over']),
        # the unavoidable full-resolution colour plane (3600B) -- the KILL cost
        helper.make_node('Conv', ['input', 'color_ramp'], ['color_plane']),
        helper.make_node('Gather', ['color_plane', 'idx'], ['gather_rows'], axis=2),
        helper.make_node('Gather', ['gather_rows', 'idx'], ['sub_f32'], axis=3),
        helper.make_node('Cast', ['sub_f32'], ['sub_u8'], to=2),
        # line colour scalar
        helper.make_node('Mod', ['line_code_f32', 'ten_f32'], ['line_f32'], fmod=1),
        helper.make_node('Cast', ['line_f32'], ['line_u8'], to=2),
        # mask overflow cells (idx clamped) to line colour so decode zeros them
        helper.make_node('Reshape', ['over', 'shape_row'], ['row_over']),
        helper.make_node('Reshape', ['over', 'shape_col'], ['col_over']),
        helper.make_node('Or', ['row_over', 'col_over'], ['any_over']),
        helper.make_node('Where', ['any_over', 'line_u8', 'sub_u8'], ['grid_u8']),
        # ---- decode (identical to incumbent, size-agnostic) ----
        helper.make_node('Equal', ['grid_u8', 'line_u8'], ['grid_eq_line']),
        helper.make_node('Where', ['grid_eq_line', 'zero_u8', 'grid_u8'], ['marker_color']),
        helper.make_node('ReduceMax', ['marker_color'], ['row_has'], axes=[0, 1, 3], keepdims=0),
        helper.make_node('ReduceMax', ['marker_color'], ['col_has'], axes=[0, 1, 2], keepdims=0),
        helper.make_node('Min', ['row_has', 'one_u8'], ['row_binary']),
        helper.make_node('Min', ['col_has', 'one_u8'], ['col_binary']),
        helper.make_node('ArgMax', ['row_binary'], ['row_first'], axis=0, keepdims=0),
        helper.make_node('ArgMax', ['col_binary'], ['col_first'], axis=0, keepdims=0),
        helper.make_node('Cast', ['row_first'], ['row_first_i32'], to=6),
        helper.make_node('Cast', ['col_first'], ['col_first_i32'], to=6),
        helper.make_node('Add', ['row_first_i32', 'idx4'], ['row_idx']),
        helper.make_node('Add', ['col_first_i32', 'idx4'], ['col_idx']),
        helper.make_node('Gather', ['marker_color', 'row_idx'], ['patch_rows'], axis=2),
        helper.make_node('Gather', ['patch_rows', 'col_idx'], ['patch'], axis=3),
        helper.make_node('Slice', ['patch', 'starts_tl', 'ends_tl', 'axes_hw'], ['tl']),
        helper.make_node('Slice', ['patch', 'starts_tr', 'ends_tr', 'axes_hw'], ['tr']),
        helper.make_node('Slice', ['patch', 'starts_bl', 'ends_bl', 'axes_hw'], ['bl']),
        helper.make_node('Slice', ['patch', 'starts_br', 'ends_br', 'axes_hw'], ['br']),
        helper.make_node('Equal', ['tl', 'tr'], ['eq_tr']),
        helper.make_node('Equal', ['tl', 'bl'], ['eq_bl']),
        helper.make_node('Equal', ['tl', 'br'], ['eq_br']),
        helper.make_node('And', ['eq_tr', 'eq_bl'], ['eq_tb']),
        helper.make_node('And', ['eq_tb', 'eq_br'], ['all_eq']),
        helper.make_node('Where', ['all_eq', 'tl', 'zero_u8'], ['small_u8']),
        helper.make_node('Equal', ['small_u8', 'colors10_u8'], ['small_full']),
        helper.make_node('Pad', ['small_full', 'pad_output'], ['output'], mode='constant'),
    ]
    return model('task185_rts', nodes, inits, output_dtype=9, opset=13, value_infos=[])
