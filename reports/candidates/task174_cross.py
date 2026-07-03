"""Task 174 candidate — CROSSOVER (Equal-then-Pad) + fp16 recasts.

Two independent, already-scanned levers on top of the deployed task174 graph
(src/custom/task174.py, mem 6973 / params 142 / 16.130 pts):

1. CROSSOVER (carrier_crossover_scan): the tail was
     Lin[1,1,5,5]u8 -> Pad -> L[1,1,30,30]u8 (900B carrier) -> Equal(chan) -> output.
   Replaced with Equal-then-Pad:
     Equal(Lin, chan) -> onehot[1,10,5,5] bool (250B) -> Pad(bool) -> free output.
   Requires opset 13 (bool Pad). Pre-pad area 25 < 90 crossover -> win.

2. RECASTS (dtype_overpay_scan): fp16 the two clean subtrees that are produced by
   Cast/ReduceSum/Pad/Gather (all fp16-capable) and only feed Greater/Equal/Pad:
     - mismatch_f, nmis        (mismatch subtree)
     - box10_f, rowcnt, box10_p, boxf, bcolcnt, browcnt   (box subtree)
     - is_box0_f               (feeds ArgMax, fp16-ok)
   All hold exact small integers (<=10), fp16-exact.
   NOT applied (traps / net-negative, see tasklog):
     - sig30: PRODUCER_BOUND (MatMul consumes fp32 graph input).
     - sig/sigk/sigk_p/sig_sh/modr/refl: fp32-chain-bound (Slice can't emit fp16;
       adding a Cast(sig30->fp16) costs +600B > the 560B it would save).
     - uint8 on mismatch_f / box10_f REFUTED: their ReduceSum consumer rejects uint8.
"""
from onnx import TensorProto, helper
from ._exact import arr_b64, model, tensor


def build(task):
    inits = [
        tensor('rowwt', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEsIDEsIDMwKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAIA/AAAAQAAAgEAAAABBAACAQQAAAEIAAIBCAAAAQwAAgEMAAABEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=')),
        tensor('c_s', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAA==')),
        tensor('c_e', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoKAAAAAAAAAA==')),
        tensor('c_ax', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoDAAAAAAAAAA==')),
        tensor('ZERO', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAA')),
        tensor('cramp', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEsIDEsIDEwKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAA8AEAAQgBEAEUARgBHAEiASA==')),
        tensor('PB', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIArOYw==')),
        tensor('NBv', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIArO4w==')),
        tensor('CLO', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAA==')),
        tensor('CHI', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAqASA==')),
        tensor('NEG1', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAvA==')),
        tensor('shp1', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoBAAAAAAAAAA==')),
        tensor('sigpad', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDgsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAUAAAAAAAAA')),
        tensor('kramp', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDUsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAA8AEAAQgBE')),
        tensor('KHI', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoASw==')),
        tensor('p2hi', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEsIDEwLCAxKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAABAAACAQAAAAEEAAIBBAAAAQgAAgEIAAABDAACAQwAAAEQAAIBE')),
        tensor('p2lo', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEsIDEwLCAxKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAIA/AAAAQAAAgEAAAABBAACAQQAAAEIAAIBCAAAAQwAAgEMAAABE')),
        tensor('rramp', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEsIDEwLCAxKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAA8AEAAQgBEAEUARgBHAEiASA==')),
        tensor('boxpad', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDgsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFAAAAAAAAAAAAAAAAAAAA')),
        tensor('krow', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEsIDUsIDEpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAA8AEAAQgBE')),
        tensor('kcol', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEsIDEsIDUpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAA8AEAAQgBE')),
        tensor('u0', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnfHUxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoA')),
        tensor('u10', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnfHUxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoK')),
        tensor('padpads', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDgsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAZAAAAAAAAABkAAAAAAAAA')),
        tensor('chan', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnfHUxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEwLCAxLCAxKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAQIDBAUGBwgJ')),
        tensor('notch0', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnfGIxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDEwLCAxLCAxKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAQEBAQEBAQEB')),
    ]
    # New consts for the recasts / crossover.
    import numpy as np
    inits.append(tensor('ZERO16', np.zeros((1,), dtype=np.float16)))
    inits.append(tensor('false_b', np.zeros((1,), dtype=bool)))
    # opset-13 ReduceSum takes axes as an input (was an attribute at opset 11).
    inits.append(tensor('ax2', np.array([2], dtype=np.int64)))
    inits.append(tensor('ax3', np.array([3], dtype=np.int64)))
    nodes = [
        helper.make_node('MatMul', ['rowwt', 'input'], ['sig30']),
        helper.make_node('Slice', ['sig30', 'c_s', 'c_e', 'c_ax'], ['sig']),
        helper.make_node('Greater', ['sig', 'ZERO'], ['colocc']),
        helper.make_node('Where', ['colocc', 'cramp', 'PB'], ['cmin_src']),
        helper.make_node('ReduceMin', ['cmin_src'], ['cmin'], axes=[3], keepdims=1),
        helper.make_node('Where', ['colocc', 'cramp', 'NBv'], ['cmax_src']),
        helper.make_node('ReduceMax', ['cmax_src'], ['cmax'], axes=[3], keepdims=1),
        helper.make_node('Add', ['cmin', 'cmax'], ['a']),
        helper.make_node('Sub', ['a', 'cramp'], ['amc']),
        helper.make_node('Clip', ['amc', 'CLO', 'CHI'], ['amc_cl']),
        helper.make_node('Cast', ['amc_cl'], ['ridx'], to=6),
        helper.make_node('GatherElements', ['sig', 'ridx'], ['refl'], axis=3),
        helper.make_node('Equal', ['sig', 'refl'], ['eqcol']),
        helper.make_node('Less', ['cramp', 'cmin'], ['lt_min']),
        helper.make_node('Not', ['lt_min'], ['ge_min']),
        helper.make_node('Greater', ['cramp', 'cmax'], ['gt_max']),
        helper.make_node('Not', ['gt_max'], ['le_max']),
        helper.make_node('And', ['ge_min', 'le_max'], ['inspan']),
        helper.make_node('Not', ['eqcol'], ['neqcol']),
        helper.make_node('And', ['inspan', 'neqcol'], ['mismatch']),
        helper.make_node('Cast', ['mismatch'], ['mismatch_f'], to=10),          # fp16 recast
        helper.make_node('ReduceSum', ['mismatch_f', 'ax3'], ['nmis'], keepdims=1),
        helper.make_node('Equal', ['nmis', 'ZERO16'], ['is_sym']),              # fp16 zero
        helper.make_node('Greater', ['cmax', 'NEG1'], ['present']),
        helper.make_node('And', ['present', 'notch0'], ['valid']),
        helper.make_node('And', ['is_sym', 'valid'], ['is_box0']),
        helper.make_node('Cast', ['is_box0'], ['is_box0_f'], to=10),            # fp16 recast
        helper.make_node('ArgMax', ['is_box0_f'], ['k4'], axis=1, keepdims=1),
        helper.make_node('Reshape', ['k4', 'shp1'], ['k1']),
        helper.make_node('Gather', ['sig', 'k1'], ['sigk'], axis=1),
        helper.make_node('Gather', ['cmin', 'k1'], ['cmink'], axis=1),
        helper.make_node('Cast', ['k4'], ['colf'], to=1),
        helper.make_node('Cast', ['colf'], ['col_u8'], to=2),
        helper.make_node('Pad', ['sigk', 'sigpad', 'ZERO'], ['sigk_p'], mode='constant'),
        helper.make_node('Reshape', ['cmink', 'shp1'], ['cmink_s']),
        helper.make_node('Add', ['kramp', 'cmink_s'], ['cidx_f']),
        helper.make_node('Clip', ['cidx_f', 'CLO', 'KHI'], ['cidx_cl']),
        helper.make_node('Cast', ['cidx_cl'], ['cidx'], to=6),
        helper.make_node('Gather', ['sigk_p', 'cidx'], ['sig_sh'], axis=3),
        helper.make_node('Mod', ['sig_sh', 'p2hi'], ['modr'], fmod=1),
        helper.make_node('Less', ['modr', 'p2lo'], ['ltlo']),
        helper.make_node('Not', ['ltlo'], ['box10']),
        helper.make_node('Cast', ['box10'], ['box10_f'], to=10),               # fp16 recast
        helper.make_node('ReduceSum', ['box10_f', 'ax3'], ['rowcnt'], keepdims=1),
        helper.make_node('Greater', ['rowcnt', 'ZERO16'], ['rowocc']),         # fp16 zero
        helper.make_node('Where', ['rowocc', 'rramp', 'PB'], ['rmin_src']),
        helper.make_node('ReduceMin', ['rmin_src'], ['rmink'], axes=[2], keepdims=1),
        helper.make_node('Pad', ['box10_f', 'boxpad', 'ZERO16'], ['box10_p'], mode='constant'),  # fp16 pad const
        helper.make_node('Reshape', ['rmink', 'shp1'], ['rmink_s']),
        helper.make_node('Add', ['kramp', 'rmink_s'], ['ridx_f']),
        helper.make_node('Clip', ['ridx_f', 'CLO', 'KHI'], ['ridx_cl']),
        helper.make_node('Cast', ['ridx_cl'], ['ridx2'], to=6),
        helper.make_node('Gather', ['box10_p', 'ridx2'], ['boxf'], axis=2),
        helper.make_node('Greater', ['boxf', 'ZERO16'], ['boxmask']),          # fp16 zero
        helper.make_node('ReduceSum', ['boxf', 'ax2'], ['bcolcnt'], keepdims=1),
        helper.make_node('Greater', ['bcolcnt', 'ZERO16'], ['bcolocc']),       # fp16 zero
        helper.make_node('ReduceSum', ['boxf', 'ax3'], ['browcnt'], keepdims=1),
        helper.make_node('Greater', ['browcnt', 'ZERO16'], ['browocc']),       # fp16 zero
        helper.make_node('Where', ['browocc', 'krow', 'NBv'], ['Hm1_src']),
        helper.make_node('ReduceMax', ['Hm1_src'], ['Hm1'], axes=[2], keepdims=1),
        helper.make_node('Where', ['bcolocc', 'kcol', 'NBv'], ['Wm1_src']),
        helper.make_node('ReduceMax', ['Wm1_src'], ['Wm1'], axes=[3], keepdims=1),
        helper.make_node('Greater', ['krow', 'Hm1'], ['rgt']),
        helper.make_node('Not', ['rgt'], ['rin']),
        helper.make_node('Greater', ['kcol', 'Wm1'], ['cgt']),
        helper.make_node('Not', ['cgt'], ['cin']),
        helper.make_node('And', ['rin', 'cin'], ['inside']),
        helper.make_node('Where', ['boxmask', 'col_u8', 'u0'], ['Lbox']),
        helper.make_node('Where', ['inside', 'Lbox', 'u10'], ['Lin']),
        # CROSSOVER: Equal-then-Pad (was Pad(Lin)->L[1,1,30,30]u8 -> Equal -> output)
        helper.make_node('Equal', ['Lin', 'chan'], ['onehot']),                # [1,10,5,5] bool
        helper.make_node('Pad', ['onehot', 'padpads', 'false_b'], ['output'], mode='constant'),
    ]
    value_infos = [
        helper.make_tensor_value_info('sig30', 1, [1, 10, 1, 30]),
        helper.make_tensor_value_info('sig', 1, [1, 10, 1, 10]),
        helper.make_tensor_value_info('colocc', 9, [1, 10, 1, 10]),
        helper.make_tensor_value_info('cmin_src', 10, [1, 10, 1, 10]),
        helper.make_tensor_value_info('cmin', 10, [1, 10, 1, 1]),
        helper.make_tensor_value_info('cmax_src', 10, [1, 10, 1, 10]),
        helper.make_tensor_value_info('cmax', 10, [1, 10, 1, 1]),
        helper.make_tensor_value_info('a', 10, [1, 10, 1, 1]),
        helper.make_tensor_value_info('amc', 10, [1, 10, 1, 10]),
        helper.make_tensor_value_info('amc_cl', 10, [1, 10, 1, 10]),
        helper.make_tensor_value_info('ridx', 6, [1, 10, 1, 10]),
        helper.make_tensor_value_info('refl', 1, [1, 10, 1, 10]),
        helper.make_tensor_value_info('eqcol', 9, [1, 10, 1, 10]),
        helper.make_tensor_value_info('lt_min', 9, [1, 10, 1, 10]),
        helper.make_tensor_value_info('ge_min', 9, [1, 10, 1, 10]),
        helper.make_tensor_value_info('gt_max', 9, [1, 10, 1, 10]),
        helper.make_tensor_value_info('le_max', 9, [1, 10, 1, 10]),
        helper.make_tensor_value_info('inspan', 9, [1, 10, 1, 10]),
        helper.make_tensor_value_info('neqcol', 9, [1, 10, 1, 10]),
        helper.make_tensor_value_info('mismatch', 9, [1, 10, 1, 10]),
        helper.make_tensor_value_info('mismatch_f', 10, [1, 10, 1, 10]),
        helper.make_tensor_value_info('nmis', 10, [1, 10, 1, 1]),
        helper.make_tensor_value_info('is_sym', 9, [1, 10, 1, 1]),
        helper.make_tensor_value_info('present', 9, [1, 10, 1, 1]),
        helper.make_tensor_value_info('valid', 9, [1, 10, 1, 1]),
        helper.make_tensor_value_info('is_box0', 9, [1, 10, 1, 1]),
        helper.make_tensor_value_info('is_box0_f', 10, [1, 10, 1, 1]),
        helper.make_tensor_value_info('k4', 7, [1, 1, 1, 1]),
        helper.make_tensor_value_info('k1', 7, [1]),
        helper.make_tensor_value_info('sigk', 1, [1, 1, 1, 10]),
        helper.make_tensor_value_info('cmink', 10, [1, 1, 1, 1]),
        helper.make_tensor_value_info('colf', 1, [1, 1, 1, 1]),
        helper.make_tensor_value_info('col_u8', 2, [1, 1, 1, 1]),
        helper.make_tensor_value_info('sigk_p', 1, [1, 1, 1, 15]),
        helper.make_tensor_value_info('cmink_s', 10, [1]),
        helper.make_tensor_value_info('cidx_f', 10, [5]),
        helper.make_tensor_value_info('cidx_cl', 10, [5]),
        helper.make_tensor_value_info('cidx', 6, [5]),
        helper.make_tensor_value_info('sig_sh', 1, [1, 1, 1, 5]),
        helper.make_tensor_value_info('modr', 1, [1, 1, 10, 5]),
        helper.make_tensor_value_info('ltlo', 9, [1, 1, 10, 5]),
        helper.make_tensor_value_info('box10', 9, [1, 1, 10, 5]),
        helper.make_tensor_value_info('box10_f', 10, [1, 1, 10, 5]),
        helper.make_tensor_value_info('rowcnt', 10, [1, 1, 10, 1]),
        helper.make_tensor_value_info('rowocc', 9, [1, 1, 10, 1]),
        helper.make_tensor_value_info('rmin_src', 10, [1, 1, 10, 1]),
        helper.make_tensor_value_info('rmink', 10, [1, 1, 1, 1]),
        helper.make_tensor_value_info('box10_p', 10, [1, 1, 15, 5]),
        helper.make_tensor_value_info('rmink_s', 10, [1]),
        helper.make_tensor_value_info('ridx_f', 10, [5]),
        helper.make_tensor_value_info('ridx_cl', 10, [5]),
        helper.make_tensor_value_info('ridx2', 6, [5]),
        helper.make_tensor_value_info('boxf', 10, [1, 1, 5, 5]),
        helper.make_tensor_value_info('boxmask', 9, [1, 1, 5, 5]),
        helper.make_tensor_value_info('bcolcnt', 10, [1, 1, 1, 5]),
        helper.make_tensor_value_info('bcolocc', 9, [1, 1, 1, 5]),
        helper.make_tensor_value_info('browcnt', 10, [1, 1, 5, 1]),
        helper.make_tensor_value_info('browocc', 9, [1, 1, 5, 1]),
        helper.make_tensor_value_info('Hm1_src', 10, [1, 1, 5, 1]),
        helper.make_tensor_value_info('Hm1', 10, [1, 1, 1, 1]),
        helper.make_tensor_value_info('Wm1_src', 10, [1, 1, 1, 5]),
        helper.make_tensor_value_info('Wm1', 10, [1, 1, 1, 1]),
        helper.make_tensor_value_info('rgt', 9, [1, 1, 5, 1]),
        helper.make_tensor_value_info('rin', 9, [1, 1, 5, 1]),
        helper.make_tensor_value_info('cgt', 9, [1, 1, 1, 5]),
        helper.make_tensor_value_info('cin', 9, [1, 1, 1, 5]),
        helper.make_tensor_value_info('inside', 9, [1, 1, 5, 5]),
        helper.make_tensor_value_info('Lbox', 2, [1, 1, 5, 5]),
        helper.make_tensor_value_info('Lin', 2, [1, 1, 5, 5]),
        helper.make_tensor_value_info('onehot', 9, [1, 10, 5, 5]),
    ]
    return model('task174_cross', nodes, inits, output_dtype=9, opset=13, value_infos=value_infos)
