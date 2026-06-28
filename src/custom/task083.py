"""Task 083 — exact source reconstruction of compact live graph.

Generated from inspected graph structure; does not load networks/task083.onnx.
"""
from onnx import TensorProto, helper
from ._exact import arr_b64, model, tensor

def build(task):
    inits = [
        tensor('bg_starts', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDQsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==')),
        tensor('bg_ends', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDQsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoBAAAAAAAAAAEAAAAAAAAAAwAAAAAAAAAEAAAAAAAAAA==')),
        tensor('row_idx', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDYsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAEAAAAAAAAAAgAAAAAAAAACAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAA=')),
        tensor('col_idx', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDgsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAEAAAAAAAAAAgAAAAAAAAADAAAAAAAAAAMAAAAAAAAAAgAAAAAAAAABAAAAAAAAAAAAAAAAAAAA')),
        tensor('zero_channel_index', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEsIDEsIDEpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAA')),
        tensor('one_updates', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEsIDYsIDgpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8=')),
    ]
    nodes = [
        helper.make_node('Slice', ['input', 'bg_starts', 'bg_ends'], ['bg32']),
        helper.make_node('Cast', ['bg32'], ['bg_mask'], to=9),
        helper.make_node('Gather', ['bg_mask', 'row_idx'], ['bg_rows_mirrored'], axis=2),
        helper.make_node('Gather', ['bg_rows_mirrored', 'col_idx'], ['bg_mirrored'], axis=3),
        helper.make_node('ReduceMax', ['input'], ['color32'], axes=[2, 3], keepdims=1),
        helper.make_node('ArgMax', ['color32'], ['color_idx64'], axis=1, keepdims=1, select_last_index=1),
        helper.make_node('Cast', ['color_idx64'], ['color_idx'], to=6),
        helper.make_node('Where', ['bg_mirrored', 'zero_channel_index', 'color_idx'], ['scatter_idx']),
        helper.make_node('ScatterElements', ['input', 'scatter_idx', 'one_updates'], ['output'], axis=1),
    ]
    return model("task083", nodes, inits, output_dtype=TensorProto.FLOAT, opset=12)
