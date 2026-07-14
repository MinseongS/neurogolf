import numpy as np

from neurogolf.scoring import convert_to_numpy, load_task
from tools.self_einsum_search import (
    build_model,
    canonicalize,
    evaluate_query,
    to_equation,
)


def test_internal_variable_renaming_is_canonical():
    left = (("k", "r", "c"), ("u", "c", "x"))
    right = (("k", "r", "c"), ("v", "c", "y"))
    assert canonicalize(left) == canonicalize(right)


def test_task067_zero_cost_crop_query_matches_all_bundled_examples():
    query = (("k", "r", "c"), ("u", "c", "x"))
    assert to_equation(query) == "bkrc,bucx->bkrc"
    for example in load_task(67)["train"] + load_task(67)["test"] + load_task(67)["arc-gen"]:
        arrays = convert_to_numpy(example)
        if arrays is not None:
            assert np.array_equal(evaluate_query(query, arrays["input"]), arrays["output"])


def test_built_model_has_one_node_and_no_initializers():
    model = build_model((("k", "r", "c"), ("u", "c", "x")))
    assert [node.op_type for node in model.graph.node] == ["Einsum"]
    assert len(model.graph.initializer) == 0
    assert list(model.graph.node[0].input) == ["input", "input"]
