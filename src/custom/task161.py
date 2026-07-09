"""Task 161 — scalar-tail collapse into a free-output Einsum.

The old tail materialized uint8 row/column code carriers and a full 30x30
code plane before Equal.  The optimized tail keeps the detected row/column
colour scalars as float features and writes the one-hot output directly.
"""
import numpy as np
from onnx import TensorProto, helper, numpy_helper
from ._exact import arr_b64, model, tensor


def build(task):
    inits = [
        tensor('rowsel0', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDMwLCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAIA/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=')),
        tensor('ones30', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDMwLCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8=')),
        tensor('arange10i', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEwLCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAEAAAAAAAAAAgAAAAAAAAADAAAAAAAAAAQAAAAAAAAABQAAAAAAAAAGAAAAAAAAAAcAAAAAAAAACAAAAAAAAAAJAAAAAAAAAA==')),
        tensor('arange10f', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEwLCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAACAPwAAAEAAAEBAAACAQAAAoEAAAMBAAADgQAAAAEEAABBB')),
        tensor('zf', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAA')),
        tensor('arange30r', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEsIDMwLCAxKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAACAPwAAAEAAAEBAAACAQAAAoEAAAMBAAADgQAAAAEEAABBBAAAgQQAAMEEAAEBBAABQQQAAYEEAAHBBAACAQQAAiEEAAJBBAACYQQAAoEEAAKhBAACwQQAAuEEAAMBBAADIQQAA0EEAANhBAADgQQAA6EE=')),
        tensor('arange30c', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEsIDEsIDMwKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAACAPwAAAEAAAEBAAACAQAAAoEAAAMBAAADgQAAAAEEAABBBAAAgQQAAMEEAAEBBAABQQQAAYEEAAHBBAACAQQAAiEEAAJBBAACYQQAAoEEAAKhBAACwQQAAuEEAAMBBAADIQQAA0EEAANhBAADgQQAA6EE=')),
        tensor('f45', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAJBA')),
        tensor('f15', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAMA/')),
        numpy_helper.from_array(np.ones((1, 30), dtype=np.float32), 'onesRow'),
        numpy_helper.from_array(np.array([[1.0] + [0.0] * 9], dtype=np.float32), 'delta0'),
        numpy_helper.from_array(np.array([0], dtype=np.int64), 'axis0'),
        numpy_helper.from_array(np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.float32), 'S_r'),
        numpy_helper.from_array(np.array([[1, 0], [1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32), 'S_c'),
        numpy_helper.from_array(np.array([[1, 0], [-1, 0], [-1, 0], [0, 1], [0, 1]], dtype=np.float32), 'S_k'),
    ]
    nodes = [
        helper.make_node('Einsum', ['input'], ['total'], equation='bchw->bc'),
        helper.make_node('Einsum', ['input', 'rowsel0', 'ones30'], ['colcount'], equation='bchw,h,w->bc'),
        helper.make_node('Einsum', ['input', 'ones30', 'rowsel0'], ['rowcount'], equation='bchw,h,w->bc'),
        helper.make_node('Add', ['colcount', 'rowcount'], ['near']),
        helper.make_node('Greater', ['near', 'f15'], ['nearok']),
        helper.make_node('Less', ['total', 'f45'], ['l45']),
        helper.make_node('And', ['l45', 'nearok'], ['mk']),
        helper.make_node('Cast', ['mk'], ['mki'], to=2),
        helper.make_node('ArgMax', ['mki'], ['Marg'], axis=1, keepdims=0),
        helper.make_node('Equal', ['arange10i', 'Marg'], ['oMb']),
        helper.make_node('Where', ['oMb', 'arange10f', 'zf'], ['wc']),
        helper.make_node('Einsum', ['input', 'wc', 'rowsel0'], ['Rvalf'], equation='bchw,c,w->bh'),
        helper.make_node('Einsum', ['input', 'wc', 'rowsel0'], ['Cvalf'], equation='bchw,c,h->bw'),
        helper.make_node('Einsum', ['input', 'rowsel0'], ['Wsc'], equation='bchw,h->b'),
        helper.make_node('Einsum', ['input', 'rowsel0'], ['Hsc'], equation='bchw,w->b'),
        helper.make_node('Less', ['arange30r', 'Hsc'], ['rowok']),
        helper.make_node('Less', ['arange30c', 'Wsc'], ['colok']),
        helper.make_node('Cast', ['rowok'], ['rowokf'], to=1),
        helper.make_node('Cast', ['colok'], ['colokf'], to=1),
        helper.make_node('Concat', ['onesRow', 'Rvalf'], ['VR'], axis=0),
        helper.make_node('Concat', ['onesRow', 'Cvalf'], ['VC'], axis=0),
        helper.make_node('Unsqueeze', ['wc', 'axis0'], ['wc2']),
        helper.make_node('Concat', ['delta0', 'wc2'], ['VK'], axis=0),
        helper.make_node(
            'Einsum',
            ['S_r', 'VR', 'S_c', 'VC', 'S_k', 'VK', 'rowokf', 'colokf'],
            ['output'],
            equation='ta,ah,te,ew,tv,vk,bqhr,bsuw->bkhw',
        ),
    ]
    value_infos = [
        helper.make_tensor_value_info('total', 1, [1, 10]),
        helper.make_tensor_value_info('colcount', 1, [1, 10]),
        helper.make_tensor_value_info('rowcount', 1, [1, 10]),
        helper.make_tensor_value_info('near', 1, [1, 10]),
        helper.make_tensor_value_info('nearok', 9, [1, 10]),
        helper.make_tensor_value_info('l45', 9, [1, 10]),
        helper.make_tensor_value_info('mk', 9, [1, 10]),
        helper.make_tensor_value_info('mki', 2, [1, 10]),
        helper.make_tensor_value_info('Marg', 7, [1]),
        helper.make_tensor_value_info('oMb', 9, [10]),
        helper.make_tensor_value_info('wc', 1, [10]),
        helper.make_tensor_value_info('Rvalf', 1, [1, 30]),
        helper.make_tensor_value_info('Cvalf', 1, [1, 30]),
        helper.make_tensor_value_info('Wsc', 1, [1]),
        helper.make_tensor_value_info('Hsc', 1, [1]),
        helper.make_tensor_value_info('rowok', 9, [1, 1, 30, 1]),
        helper.make_tensor_value_info('colok', 9, [1, 1, 1, 30]),
        helper.make_tensor_value_info('rowokf', 1, [1, 1, 30, 1]),
        helper.make_tensor_value_info('colokf', 1, [1, 1, 1, 30]),
        helper.make_tensor_value_info('VR', 1, [2, 30]),
        helper.make_tensor_value_info('VC', 1, [2, 30]),
        helper.make_tensor_value_info('wc2', 1, [1, 10]),
        helper.make_tensor_value_info('VK', 1, [2, 10]),
    ]
    return model('task161_tail_einsum', nodes, inits, output_dtype=TensorProto.FLOAT, opset=13, value_infos=value_infos)
