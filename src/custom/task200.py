"""Task 200 — exact source reconstruction of compact live graph.

Generated from inspected graph structure; does not load networks/task200.onnx.
"""
from onnx import TensorProto, helper
from ._exact import arr_b64, model, tensor

def build(task):
    inits = [
        tensor('slice_starts_bottom0', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDQsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAAAAAAAAAAACQAAAAAAAAAAAAAAAAAAAA==')),
        tensor('slice_ends_bottom0', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDQsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoBAAAAAAAAAAEAAAAAAAAACgAAAAAAAAAKAAAAAAAAAA==')),
        tensor('zero_f', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAA')),
        tensor('color_line_w', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEsIDEsIDUpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAPAA8ADwAPAA8')),
        tensor('gray_endpoint_w', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsIDEsIDEsIDEwKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAPAAAAAAAAAA8AAAAAAAAADwAAAAAAAAAPAAAAAAAAAA8AAAAAAAA')),
        tensor('w_inside10', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEsIDEsIDEwKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAPAA8ADwAPAA8ADwAPAA8ADwAPA==')),
        tensor('w_pads', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDgsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABQAAAAAAAAA')),
        tensor('Br', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDMsIDMwKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAPAA8ADwAPAA8ADwAPAA8ADwAPAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=')),
        tensor('Sr', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDQsIDMpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAPAAAAAAAPAAAAAAAAAA8AAAAAAAAADw=')),
        tensor('Sc', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDQsIDMpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAA8AAAAPADAAAAAAAC8ADwAAAC8ADw=')),
        tensor('bc_ch0', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEwKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAPAAAAAAAAAAAAAAAAAAAAAAAAA==')),
        tensor('bc_ch5', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEwKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAAAADwAAAAAAAAAAA==')),
    ]
    nodes = [
        helper.make_node('Slice', ['input', 'slice_starts_bottom0', 'slice_ends_bottom0'], ['bottom0_f']),
        helper.make_node('Equal', ['bottom0_f', 'zero_f'], ['seed_any_b']),
        helper.make_node('Cast', ['seed_any_b'], ['seed_any_h'], to=10),
        helper.make_node('Conv', ['seed_any_h', 'color_line_w'], ['color_line_h'], dilations=[1, 2], kernel_shape=[1, 5], pads=[0, 8, 0, 0]),
        helper.make_node('Conv', ['seed_any_h', 'gray_endpoint_w'], ['gray_lines_h'], kernel_shape=[1, 10], pads=[0, 9, 0, 0]),
        helper.make_node('Concat', ['w_inside10', 'color_line_h', 'gray_lines_h'], ['W10'], axis=1),
        helper.make_node('Pad', ['W10', 'w_pads'], ['W'], mode='constant'),
        helper.make_node('ReduceMax', ['input'], ['color_presence_f'], axes=[2, 3], keepdims=0),
        helper.make_node('Cast', ['color_presence_f'], ['color_presence_h'], to=10),
        helper.make_node('Concat', ['color_presence_h', 'bc_ch0', 'bc_ch5'], ['Bc'], axis=0),
        helper.make_node('Einsum', ['Sc', 'Bc', 'Sr', 'Br', 'W'], ['output'], equation='tu,uc,tv,vh,btqw->bchw'),
    ]
    return model("task200", nodes, inits, output_dtype=TensorProto.FLOAT16, opset=13)
