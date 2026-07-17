"""Task 389 — shared rank-2 radial-gray palette rewrite (cost 38).

All ordinary palette columns in P lie on the unit circle; gray is the lone
short-radius column.  One shared rank-2 basis reconstructs both the squared-P
gray-XOR quadratic and the signed output-colour metric.
"""
from onnx import TensorProto, helper
from ._exact import arr_b64, model, tensor


def build(task):
    inits = [
        tensor('P', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDMsIDEwKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAP30bRD/U0DE+AAAAv7KPcL8fhWs/so9wvwAAAL/U0DE+fRtEPwAAAAC7jSQ/XBx8P9ezXT9EHa8+AAAAAEQdr77Xs12/XBx8v7uNJL8=')),
        tensor('shared_rank2_basis', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsIDMpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAqLAT9DVxddw1cXXcO0dWVDYUmLw2FJi8M=')),
        tensor('shared_rank2_weight', arr_b64('k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAq8jwy8PPlSOw==')),
    ]
    nodes = [
        helper.make_node(
            'Einsum',
            ['input', 'input', 'P', 'P', 'shared_rank2_basis', 'P', 'P',
             'shared_rank2_basis', 'shared_rank2_weight', 'P', 'P',
             'shared_rank2_basis', 'shared_rank2_weight'],
            ['output'],
            equation='nihw,njrs,ai,ai,ta,bj,bj,tb,t,pj,po,xp,x->nohw',
        ),
    ]
    value_infos = [
    ]
    return model('task389_shared_rank2_metric', nodes, inits, output_dtype=1, opset=13, value_infos=value_infos)
