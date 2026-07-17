import onnx
from onnx import TensorProto, helper

from tools import live_to_exact_source


def test_generate_preserves_initializer_tensorproto_encoding(monkeypatch, tmp_path):
    initializer = helper.make_tensor(
        "indices",
        TensorProto.INT64,
        [3],
        [1, 2, 3],
    )
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "typed_initializer",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [initializer],
    )
    original = helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 18)],
    )
    onnx.save(original, tmp_path / "task001.onnx")
    monkeypatch.setattr(live_to_exact_source, "NET_DIR", tmp_path)

    namespace = {}
    exec(live_to_exact_source.generate(1, package_relative=False), namespace)
    rebuilt = namespace["build"](None)

    assert rebuilt.graph.initializer[0].SerializeToString() == (
        original.graph.initializer[0].SerializeToString()
    )
