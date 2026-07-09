import onnx
from onnx import TensorProto, helper
from neurogolf.topk import find_unsigned_topk

def _topk_model(elem_type):
    x = helper.make_tensor_value_info("x", elem_type, [1, 8])
    k = helper.make_tensor("k", TensorProto.INT64, [1], [3])
    v = helper.make_tensor_value_info("v", elem_type, [1, 3])
    i = helper.make_tensor_value_info("i", TensorProto.INT64, [1, 3])
    node = helper.make_node("TopK", ["x", "k"], ["v", "i"])
    graph = helper.make_graph([node], "g", [x], [v, i], initializer=[k])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])

def test_uint8_topk_flagged(tmp_path):
    p = tmp_path / "bad.onnx"; onnx.save(_topk_model(TensorProto.UINT8), p)
    assert find_unsigned_topk(p)

def test_float_topk_clean(tmp_path):
    p = tmp_path / "ok.onnx"; onnx.save(_topk_model(TensorProto.FLOAT), p)
    assert find_unsigned_topk(p) == []

def test_unknown_type_topk_flagged(tmp_path):
    # TopK input "y" is a Relu output fed by an UNDEFINED-typed graph input, which
    # defeats onnx shape inference -> "y" never gets a value_info entry -> types.get
    # returns None -> must be treated as a violation (UNKNOWN also counts).
    x = helper.make_tensor_value_info("x", TensorProto.UNDEFINED, [1, 8])
    k = helper.make_tensor("k", TensorProto.INT64, [1], [3])
    relu = helper.make_node("Relu", ["x"], ["y"])
    v = helper.make_tensor_value_info("v", TensorProto.FLOAT, [1, 3])
    i = helper.make_tensor_value_info("i", TensorProto.INT64, [1, 3])
    topk = helper.make_node("TopK", ["y", "k"], ["v", "i"])
    graph = helper.make_graph([relu, topk], "g", [x], [v, i], initializer=[k])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    p = tmp_path / "unk.onnx"; onnx.save(model, p)
    assert find_unsigned_topk(p)
