from collections import Counter
from pathlib import Path

import onnx

from candidates.task110.build_reordered_einsum import (
    SOURCE,
    VARIANT_ORDERS,
    reorder_model,
)


def signature(model: onnx.ModelProto) -> Counter:
    node = model.graph.node[0]
    attr = next(attribute for attribute in node.attribute if attribute.name == "equation")
    lhs = onnx.helper.get_attribute_value(attr).decode().split("->", 1)[0].split(",")
    return Counter(zip(node.input, lhs, strict=True))


def test_every_order_is_a_full_permutation():
    operand_count = len(onnx.load(SOURCE).graph.node[0].input)
    for name, order in VARIANT_ORDERS.items():
        assert tuple(sorted(order)) == tuple(range(operand_count)), name


def test_reordering_preserves_operand_term_pairs():
    original = onnx.load(SOURCE)
    for order in VARIANT_ORDERS.values():
        candidate = reorder_model(SOURCE, order)
        onnx.checker.check_model(candidate, full_check=True)
        assert signature(candidate) == signature(original)
