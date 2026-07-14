import numpy as np
import pytest
from onnx import TensorProto, helper

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


def test_operand_permutation_is_canonical():
    left = (
        ("k", "r", "c"),
        ("a", "p", "p"),
        ("b", "p", "q"),
    )
    right = (
        ("k", "r", "c"),
        ("b", "p", "q"),
        ("a", "p", "p"),
    )
    assert canonicalize(left) == canonicalize(right)


def test_canonicalization_is_idempotent():
    query = (
        ("k", "r", "c"),
        ("a", "p", "p"),
        ("d", "p", "q"),
        ("e", "q", "r"),
        ("f", "r", "p"),
    )
    canonical = canonicalize(query)
    assert canonicalize(canonical) == canonical


def test_query_must_be_non_empty():
    with pytest.raises(ValueError, match="non-empty"):
        canonicalize(())


@pytest.mark.parametrize(
    "query",
    [
        (("u", "r", "c"),),
        (("k", "x", "c"),),
        (("k", "r", "x"),),
    ],
)
def test_query_must_supply_every_output_label(query):
    with pytest.raises(ValueError, match="k, r, and c"):
        canonicalize(query)


@pytest.mark.parametrize(
    "query",
    [
        (("k", "r"),),
        (("k", "r", "c", "x"),),
    ],
)
def test_every_atom_must_have_arity_three(query):
    with pytest.raises(ValueError, match="exactly three labels"):
        canonicalize(query)


@pytest.mark.parametrize(
    "query",
    [
        (("k", "r", "c"), ("r", "r", "c")),
        (("k", "r", "k"),),
        (("k", "r", "c"), ("c", "r", "c")),
        (("k", "r", "c"), ("u", "u", "c")),
    ],
)
def test_reserved_and_cross_type_labels_are_rejected(query):
    with pytest.raises(ValueError, match="reserved or shared across types"):
        canonicalize(query)


@pytest.mark.parametrize(
    "query",
    [
        (("k", "r", "c"),)
        + tuple((f"color_{index}", "r", "c") for index in range(12)),
        (("k", "r", "c"),)
        + tuple(("k", f"space_{index}", "c") for index in range(12)),
    ],
)
def test_internal_label_count_must_fit_the_typed_pool(query):
    with pytest.raises(ValueError, match="label pool capacity"):
        canonicalize(query)


def test_task067_zero_cost_crop_query_matches_all_bundled_examples():
    query = (("k", "r", "c"), ("u", "c", "x"))
    assert to_equation(query) == "bkrc,bucx->bkrc"
    evaluated = 0
    for example in load_task(67)["train"] + load_task(67)["test"] + load_task(67)["arc-gen"]:
        arrays = convert_to_numpy(example)
        if arrays is not None:
            actual = evaluate_query(query, arrays["input"])
            assert actual.dtype == np.bool_
            assert actual.shape == (1, 10, 30, 30)
            assert np.array_equal(actual, arrays["output"])
            evaluated += 1
    assert evaluated > 0


def test_built_model_has_one_node_and_no_initializers():
    model = build_model((("k", "r", "c"), ("u", "c", "x")))
    assert [node.op_type for node in model.graph.node] == ["Einsum"]
    assert len(model.graph.initializer) == 0
    assert list(model.graph.node[0].input) == ["input", "input"]
    attributes = {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in model.graph.node[0].attribute
    }
    assert attributes["equation"] == b"bkrc,bucx->bkrc"

    assert [value.name for value in model.graph.output] == ["output"]
    for value in (*model.graph.input, *model.graph.output):
        tensor_type = value.type.tensor_type
        assert tensor_type.elem_type == TensorProto.FLOAT
        assert [dimension.dim_value for dimension in tensor_type.shape.dim] == [1, 10, 30, 30]
