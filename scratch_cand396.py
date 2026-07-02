"""Task 396 candidate — valid-Conv in-op crop (playbook #7).

The incumbent reads the colour plane with a 1x1 Conv over the full 30x30 input
(A, 3600B fp32) then Cast->uint8 full canvas (Au, 900B) then Slice to 18x18.
Grids are <=18x18 (generator width/height in [12,18]), so the content always
lives in the top-left 18x18. A single valid Conv with a 13x13 kernel whose only
nonzero tap sits at spatial (0,0) reads colour+1 AND crops to 18x18 in one op:
  out[i,j] = sum_c (c+1) * input[c, i+0, j+0]  for i,j in [0,18)
This deletes A(3600) + Au(900) = 4500B mem, replacing with one 1296B fp32 plane
(+ the same uint8 Bu). Costs +1680 params (13x13 dense kernel vs 1x1).
Everything from Bu onward is byte-identical to the incumbent.
"""
import numpy as np
from onnx import TensorProto, helper
from ._exact import arr_b64, model, tensor


def build(task):
    # valid-conv colour kernel: tap at (0,0), weight[0,c,0,0] = c+1 (colour+1)
    a13 = np.zeros((1, 10, 13, 13), np.float32)
    a13[0, :, 0, 0] = np.arange(1, 11, dtype=np.float32)

    inits = [
        tensor('o_u8', np.array(1, np.uint8)),
        tensor('z_u8', np.array(0, np.uint8)),
        tensor('a13', a13),
        tensor('b', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAAAAAAAAAAA')),
        tensor('d', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoCAAAAAAAAAAMAAAAAAAAA')),
        tensor('h', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEsIDQsIDQpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAxgA8ADwAPAA8AAAAAAAAADwAAAAAAAAAPAAAAAAAAA==')),
        tensor('i', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAA==')),
        tensor('j', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoPAAAAAAAAAA8AAAAAAAAA')),
        tensor('k', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEsIDEsIDgpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAWABUAFAATABIAEQAQAA8')),
        tensor('l', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEsIDgsIDEpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAWABUAFAATABIAEQAQAA8')),
        tensor('m', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDgsIDEsIDEpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAWABaAFuAW8Bb4FvwW/hb')),
        tensor('n', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoBAAAAAAAAAA==')),
        tensor('o', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAPA==')),
        tensor('p', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEwKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAA8AEAAQgBEAEUARgBHAEiASA==')),
        tensor('q', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEwKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAA8ADwAPAA8ADwAPAA8ADwAPA==')),
        tensor('r', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDgsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAA8AEAAQgBEAEUARgBH')),
        tensor('s', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAqILw==')),
        tensor('t', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAQA==')),
        tensor('u', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDQsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoBAAAAAAAAAAEAAAAAAAAACAAAAAAAAAABAAAAAAAAAA==')),
        tensor('v', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDQsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoBAAAAAAAAAAgAAAAAAAAAAQAAAAAAAAABAAAAAAAAAA==')),
        tensor('w', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEsIDgsIDEpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAAAAAAAAAAA')),
        tensor('x', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDgsIDEsIDEpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAAAAAAAAAAA')),
        tensor('y', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEwLCAxLCAxKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAPABAAEIARABFAEYARwBIgEgASQ==')),
        tensor('z', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDgsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWAAAAAAAAABYAAAAAAAAA')),
    ]
    nodes = [
        helper.make_node('Conv', ['input', 'a13'], ['A18']),       # valid conv: colour+1, cropped 18x18 (1296B fp32)
        helper.make_node('Cast', ['A18'], ['Bu'], to=2),           # uint8 18x18 label (324B)
        helper.make_node('Cast', ['Bu'], ['C'], to=10),            # fp16 label grid for float ops
        # --- leaner box-colour (L = J+1) detection (uint8 label space) ---
        helper.make_node('Conv', ['C', 'h'], ['F']),               # 4x4 corner conv -> 0 at uniform L-shapes
        helper.make_node('Equal', ['F', 'i'], ['G']),              # F == 0  (corner OR uniform-black)
        helper.make_node('Slice', ['Bu', 'b', 'j', 'd'], ['Hu']),  # uint8 colour value, 15x15 (225B)
        helper.make_node('Greater', ['Hu', 'o_u8'], ['I2']),       # nonzero colour (value > 1)
        helper.make_node('And', ['G', 'I2'], ['J']),               # corner AND nonzero
        helper.make_node('Where', ['J', 'Hu', 'z_u8'], ['Ku']),    # uint8 colour at corners, else 0 (225B)
        helper.make_node('ReduceMax', ['Ku'], ['Lu'], axes=[2, 3], keepdims=1),
        helper.make_node('Cast', ['Lu'], ['L'], to=10),            # fp16 scalar box colour label
        # --- unchanged from live graph ---
        helper.make_node('Equal', ['C', 'L'], ['M']),
        helper.make_node('Cast', ['M'], ['N'], to=10),
        helper.make_node('Conv', ['N', 'k'], ['O'], pads=[0, 0, 0, 7]),
        helper.make_node('Conv', ['N', 'l'], ['P'], pads=[0, 0, 7, 0]),
        helper.make_node('ReduceMax', ['O'], ['Q'], axes=[2, 3], keepdims=1),
        helper.make_node('ReduceMax', ['P'], ['R'], axes=[2, 3], keepdims=1),
        helper.make_node('GreaterOrEqual', ['Q', 'm'], ['S']),
        helper.make_node('Cast', ['S'], ['T'], to=10),
        helper.make_node('ReduceSum', ['T', 'n'], ['U'], keepdims=1),
        helper.make_node('GreaterOrEqual', ['R', 'm'], ['V']),
        helper.make_node('Cast', ['V'], ['W'], to=10),
        helper.make_node('ReduceSum', ['W', 'n'], ['X'], keepdims=1),
        helper.make_node('ReduceMax', ['O'], ['Y'], axes=[2], keepdims=0),
        helper.make_node('ArgMax', ['Y'], ['Z'], axis=2, keepdims=0),
        helper.make_node('Cast', ['Z'], ['aa'], to=10),
        helper.make_node('ReduceMax', ['P'], ['ab'], axes=[3], keepdims=0),
        helper.make_node('ArgMax', ['ab'], ['ac'], axis=2, keepdims=0),
        helper.make_node('Cast', ['ac'], ['ad'], to=10),
        helper.make_node('ReduceSum', ['input', 'd'], ['ae'], keepdims=0),
        helper.make_node('Cast', ['ae'], ['af'], to=10),
        helper.make_node('Sub', ['L', 'o'], ['ag']),
        helper.make_node('Equal', ['p', 'ag'], ['ah']),
        helper.make_node('Where', ['ah', 'i', 'af'], ['ai']),
        helper.make_node('Mul', ['ai', 'q'], ['aj']),
        helper.make_node('ArgMax', ['aj'], ['ak'], axis=3, keepdims=0),
        helper.make_node('Cast', ['ak'], ['al'], to=10),
        helper.make_node('Add', ['al', 'o'], ['am']),
        helper.make_node('Add', ['r', 'aa'], ['an']),
        helper.make_node('Mul', ['an', 's'], ['ao']),
        helper.make_node('Sub', ['ao', 'o'], ['ap']),
        helper.make_node('Less', ['r', 'U'], ['aq']),
        helper.make_node('Where', ['aq', 'ap', 't'], ['ar']),
        helper.make_node('Add', ['r', 'ad'], ['as']),
        helper.make_node('Mul', ['as', 's'], ['at']),
        helper.make_node('Sub', ['at', 'o'], ['au']),
        helper.make_node('Less', ['r', 'X'], ['av']),
        helper.make_node('Where', ['av', 'au', 't'], ['aw']),
        helper.make_node('Reshape', ['ar', 'u'], ['ax']),
        helper.make_node('Reshape', ['aw', 'v'], ['ay']),
        helper.make_node('Concat', ['ax', 'w'], ['az'], axis=3),
        helper.make_node('Concat', ['x', 'ay'], ['aA'], axis=3),
        helper.make_node('Add', ['az', 'aA'], ['aB']),
        helper.make_node('GridSample', ['C', 'aB'], ['aC'], align_corners=1, mode='nearest', padding_mode='zeros'),
        helper.make_node('Equal', ['aC', 'L'], ['aD']),
        helper.make_node('Where', ['aD', 'am', 'aC'], ['aE']),
        helper.make_node('Equal', ['aE', 'y'], ['aF']),
        helper.make_node('Pad', ['aF', 'z'], ['output']),
    ]
    value_infos = []
    return model('task396_validcrop', nodes, inits, output_dtype=9, opset=17, value_infos=value_infos)
