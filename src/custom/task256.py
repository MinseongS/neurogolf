"""Task 256 — compact closed-form rebuild.

Rule (generator task_a65b410d): the input has a horizontal red (2) segment of
`L` cells starting at column 0 in row `R`. The output draws, per row r, a run of
`d = L + R - r` cells in columns 0..d-1, colored green (3) for r<R, red (2) for
r==R, blue (1) for r>R; every other in-grid cell is background (0). The colored
triangle always fits inside the grid.

Compression vs. the old graph: derive R and L from a tiny 8x8 red slice, recover
the real grid H,W from two cheap ReduceMax/ReduceSum reductions (no full-grid
input slice), build a uint8 *label* grid on a fixed 13x13 window (color =
Sign(R-r)+2 gated by the colored mask; background/out-of-grid gated by
row<H & col<W), zero-pad the label to 30x30 with a sentinel, then expand it to
the 10-channel one-hot with a single Equal straight into "output". No float
one-hot; the only 30x30 working tensor is the single-channel padded label.
"""
from onnx import helper
from ._exact import arr_b64, model, tensor


def build(task):
    inits = [
        tensor('rowc13', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEsIDEzLCAxKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAACAPwAAAEAAAEBAAACAQAAAoEAAAMBAAADgQAAAAEEAABBBAAAgQQAAMEEAAEBB')),
        tensor('colc13', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEsIDEsIDEzKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAACAPwAAAEAAAEBAAACAQAAAoEAAAMBAAADgQAAAAEEAABBBAAAgQQAAMEEAAEBB')),
        tensor('two_f', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEsIDEsIDEpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAABA')),
        tensor('nine_u8', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnfHUxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEsIDEsIDEpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoJ')),
        tensor('zero_u8', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnfHUxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEsIDEsIDEpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoA')),
        tensor('planes10_u8', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnfHUxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEwLCAxLCAxKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAQIDBQUFBQUF')),
        tensor('pads_lab', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDgsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAARAAAAAAAAABEAAAAAAAAA')),
        tensor('cval9', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnfHUxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoJ')),
        tensor('red_s', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDQsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==')),
        tensor('red_e', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDQsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoBAAAAAAAAAAMAAAAAAAAACAAAAAAAAAAIAAAAAAAAAA==')),
    ]
    nodes = [
        # real grid size H, W (grid is a contiguous top-left block) -> in-grid mask
        helper.make_node('ReduceMax', ['input'], ['rowany'], axes=[1, 3], keepdims=1),
        helper.make_node('ReduceMax', ['input'], ['colany'], axes=[1, 2], keepdims=1),
        helper.make_node('ReduceSum', ['rowany'], ['Hf'], axes=[1, 2, 3], keepdims=1),
        helper.make_node('ReduceSum', ['colany'], ['Wf'], axes=[1, 2, 3], keepdims=1),
        helper.make_node('Less', ['rowc13', 'Hf'], ['inH']),
        helper.make_node('Less', ['colc13', 'Wf'], ['inW']),
        # red segment: a tiny 8x8 window is enough (red row/col are both < 8)
        helper.make_node('Slice', ['input', 'red_s', 'red_e'], ['red']),
        helper.make_node('ReduceSum', ['red'], ['redcol'], axes=[3], keepdims=1),
        helper.make_node('ReduceMax', ['redcol'], ['L'], axes=[2], keepdims=1),
        helper.make_node('ArgMax', ['redcol'], ['Ri'], axis=2, keepdims=1),
        helper.make_node('Cast', ['Ri'], ['R'], to=1),   # float
        # colored mask M = (col < L + R - r)
        helper.make_node('Add', ['L', 'R'], ['LpR']),
        helper.make_node('Sub', ['LpR', 'rowc13'], ['d']),
        helper.make_node('Less', ['colc13', 'd'], ['M']),
        # per-row color code = Sign(R - r) + 2  ->  green 3 / red 2 / blue 1
        helper.make_node('Sub', ['R', 'rowc13'], ['Rmr']),
        helper.make_node('Sign', ['Rmr'], ['sgn']),
        helper.make_node('Add', ['sgn', 'two_f'], ['color_f']),
        helper.make_node('Cast', ['color_f'], ['color'], to=2),   # uint8
        # background/out-of-grid base: 0 inside the real HxW grid, sentinel 9
        # outside (built without a full-grid AND, via nested row/col selects).
        helper.make_node('Where', ['inW', 'zero_u8', 'nine_u8'], ['wsel']),
        helper.make_node('Where', ['inH', 'wsel', 'nine_u8'], ['base']),
        # label grid: colored cells (always in-grid) -> color; else the base.
        helper.make_node('Where', ['M', 'color', 'base'], ['label']),
        # zero-pad label window to 30x30 (out-of-window -> sentinel 9), then
        # expand to the 10-channel one-hot straight into "output".
        helper.make_node('Pad', ['label', 'pads_lab', 'cval9'], ['label30']),
        helper.make_node('Equal', ['label30', 'planes10_u8'], ['output']),
    ]
    return model('task256_compact', nodes, inits, output_dtype=9, opset=11)
