"""Task 278 — fixed-colour stamp routed to the free output.

Output = input with a 3-coloured halo painted on the background cells around the
qualifying red anchors (green_u = bg AND dilated-anchor).  Instead of rebuilding a
colour-index label (red + green) and Equal-expanding it, route Where(green_mask,
onehot[3], input) straight into the free output -- the input 2s and background are
preserved automatically.  Drops the rg/rg3/cguK label-combine planes.

KEY (re-verified by fork 2026-06-30): green = halo AND ch0, where ch0 is the real
background channel.  ch0 cannot be replaced by NOT(red): off-grid sentinel cells
are all-zero (ch0=0) yet NOT(red)=1 there, so a boundary olive's 3x3 halo would
spill green into the off-grid region of a sub-18 grid.  Deriving ch0 from
row/col reductions costs more plumbing than the slice, so both fp32 slices stay.
Verified 265/265 stored + OLD==NEW on 4000 random in-domain (<=18x18) grids.
mem 6084 -> 5436 (+0.11).
"""
import numpy as np

from onnx import TensorProto, helper, numpy_helper
from ._exact import arr_b64, model, tensor


def build(task):
    inits = [
        tensor('c0s', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDQsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==')),
        tensor('c0e', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDQsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoBAAAAAAAAAAEAAAAAAAAAEgAAAAAAAAASAAAAAAAAAA==')),
        tensor('c2s', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDQsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==')),
        tensor('c2e', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDQsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoBAAAAAAAAAAMAAAAAAAAAEgAAAAAAAAASAAAAAAAAAA==')),
        tensor('xs', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAIA/')),
        tensor('xz', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnfHUxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoA')),
        tensor('ys', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAABBB')),
        tensor('anchor', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnfHUxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEsIDMsIDMpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAQABBAEAAQA=')),
        tensor('pads', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDgsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMAAAAAAAAAAwAAAAAAAAA')),
        tensor('onehot3', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEwLCAxLCAxKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')),
        tensor('pad_false', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnfGIxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoA')),
    ]
    nodes = [
        helper.make_node('Slice', ['input', 'c0s', 'c0e'], ['c0f']),
        helper.make_node('Slice', ['input', 'c2s', 'c2e'], ['c2f']),
        helper.make_node('Cast', ['c0f'], ['c0u'], to=2),
        helper.make_node('Cast', ['c2f'], ['red_u'], to=2),
        helper.make_node('QLinearConv', ['red_u', 'xs', 'xz', 'anchor', 'xs', 'xz', 'ys', 'xz'], ['olive_u'], pads=[1, 1, 1, 1]),
        helper.make_node('MaxPool', ['olive_u'], ['halo_u'], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node('Min', ['c0u', 'halo_u'], ['green_u']),
        helper.make_node('Cast', ['green_u'], ['green_b'], to=9),
        helper.make_node('Pad', ['green_b', 'pads', 'pad_false'], ['green30'], mode='constant'),
        helper.make_node('Where', ['green30', 'onehot3', 'input'], ['output']),
    ]
    return model('task278_route', nodes, inits, output_dtype=1, opset=13, value_infos=[])
