import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from neurogolf.scoring import calculate_params, convert_to_numpy, load_task
from tools.self_einsum_search import (
    Correction,
    build_model,
    canonicalize,
    evaluate_query,
    expand_query,
    fit_small_corrections,
    grammar_reaches,
    query_loss,
    search_task,
    select_diagnostic_examples,
    to_equation,
)


def _examples(task_num):
    task = load_task(task_num)
    return [
        arrays
        for split in ("train", "test", "arc-gen")
        for example in task[split]
        if (arrays := convert_to_numpy(example)) is not None
    ]


def _assert_model_matches_bundled(task_num, model):
    examples = _examples(task_num)
    assert examples
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    official_count = len(load_task(task_num)["train"] + load_task(task_num)["test"])
    for arrays in examples[:official_count]:
        actual = session.run(None, {"input": arrays["input"]})[0] > 0
        assert np.array_equal(actual, arrays["output"] > 0)

    node = model.graph.node[0]
    equation = next(
        helper.get_attribute_value(attribute).decode()
        for attribute in node.attribute
        if attribute.name == "equation"
    )
    initializers = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    for start in range(0, len(examples), 8):
        batch = examples[start : start + 8]
        inputs = np.concatenate([arrays["input"] for arrays in batch])
        targets = np.concatenate([arrays["output"] for arrays in batch]) > 0
        operands = [
            inputs if name == "input" else initializers[name] for name in node.input
        ]
        actual = np.einsum(equation, *operands, optimize="greedy") > 0
        assert np.array_equal(actual, targets)


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
        (("k", "r", "c"), ("r", "c", "x")),
        (("k", "r", "c"), ("b", "c", "x")),
        (("k", "r", "c"), ("u", "c", "k")),
        (("k", "r", "c"), ("u", "c", "b")),
        (("k", "r", "c"), ("u", "c", "u")),
    ],
)
def test_raw_label_spelling_is_independent_between_types(query):
    ordinary = (("k", "r", "c"), ("v", "c", "y"))
    assert canonicalize(query) == canonicalize(ordinary)
    assert to_equation(query) == "bkrc,bucx->bkrc"


@pytest.mark.parametrize(
    "query",
    [
        (("k", "r", "c"),) + tuple((f"color_{index}", "r", "c") for index in range(12)),
        (("k", "r", "c"),) + tuple(("k", f"space_{index}", "c") for index in range(12)),
    ],
)
def test_internal_label_count_must_fit_the_typed_pool(query):
    with pytest.raises(ValueError, match="label pool capacity"):
        canonicalize(query)


def test_task067_zero_cost_crop_query_matches_all_bundled_examples():
    query = (("k", "r", "c"), ("u", "c", "x"))
    assert to_equation(query) == "bkrc,bucx->bkrc"
    evaluated = 0
    for example in (
        load_task(67)["train"] + load_task(67)["test"] + load_task(67)["arc-gen"]
    ):
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
        assert [dimension.dim_value for dimension in tensor_type.shape.dim] == [
            1,
            10,
            30,
            30,
        ]


def test_expansion_is_connected_and_deduplicated():
    seed = (("k", "r", "c"),)
    expanded = expand_query(seed)
    assert expanded
    assert len(expanded) == len(set(expanded))
    for query in expanded:
        assert len(query) == 2
        assert set(query[0]) & set(query[1])


def test_expansion_introduces_at_most_one_spatial_label():
    seed = (("k", "x", "y"),)
    seed_spaces = {"x", "y"}
    for query in expand_query(seed):
        query_spaces = {label for _, row, col in query for label in (row, col)}
        assert len(query_spaces) - len(seed_spaces) <= 1


def test_grammar_reaches_task017_without_identity_or_transpose_atom():
    query = (
        ("a", "r", "x"),
        ("a", "p", "x"),
        ("b", "y", "c"),
        ("b", "y", "q"),
        ("k", "p", "q"),
    )
    assert ("k", "r", "c") not in query
    assert ("k", "c", "r") not in query
    assert grammar_reaches(query)


def test_query_loss_counts_wrong_output_cells():
    raw_input = np.zeros((1, 10, 2, 2), dtype=np.float32)
    raw_input[:, 1] = 1
    target = raw_input.copy()
    target[:, 1, 0, 0] = 0
    query = (("k", "r", "c"),)
    assert query_loss(query, [(raw_input, target)]) == 1


def test_diagnostic_examples_keep_official_and_diverse_generated_cases():
    examples = []
    for index in range(8):
        raw_input = np.zeros((1, 10, 30, 30), dtype=np.float32)
        raw_input[:, : (index % 4) + 1, : index + 1, : 2 * index + 1] = 1
        examples.append((raw_input, raw_input > 0))
    selected = select_diagnostic_examples(examples, official_count=2, arc_limit=3)
    assert all(left is right for left, right in zip(selected[:2], examples[:2]))
    assert len(selected) == 5
    repeated = select_diagnostic_examples(examples, official_count=2, arc_limit=3)
    assert all(left is right for left, right in zip(selected, repeated))


def test_search_rediscovers_task067_with_two_atoms():
    hits = search_task(67, max_atoms=2, beam=500)
    assert any(hit["fail"] == 0 and hit["cost"] == 0 for hit in hits)


def test_channel_mask_completes_a_low_cost_near_hit():
    raw_input = np.ones((1, 10, 2, 2), dtype=bool)
    target = raw_input.copy()
    target[:, 1] = False
    query = (("k", "r", "c"),)
    corrections = fit_small_corrections(query, [(raw_input, target)], max_params=10)
    assert any(
        correction.channel_mask == (1, 0, 1, 1, 1, 1, 1, 1, 1, 1)
        for correction in corrections
    )


def test_task017_cost10_correction_rebuilds_live_mechanism():
    query = canonicalize(
        (
            ("a", "r", "x"),
            ("a", "p", "x"),
            ("b", "y", "c"),
            ("b", "y", "q"),
            ("k", "p", "q"),
        )
    )
    mask = (0, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    correction = Correction(channel_masks=(("k", mask), ("u", mask), ("v", mask)))
    model = build_model(query, correction)

    assert [node.op_type for node in model.graph.node] == ["Einsum"]
    assert calculate_params(model) == 10
    assert correction.param_count == 10
    _assert_model_matches_bundled(17, model)


def test_task197_cost130_correction_rebuilds_live_mechanism():
    query = canonicalize((("k", "r", "j"), ("a", "x", "c"), ("a", "y", "j")))
    row1 = tuple(int(index == 1) for index in range(30))
    mixer = np.eye(10, dtype=np.float32)
    mixer[0, 1:] = -10
    correction = Correction(
        spatial_gates=(("y", row1), ("z", row1)),
        channel_mixer=tuple(tuple(float(value) for value in row) for row in mixer),
    )
    model = build_model(query, correction)

    assert [node.op_type for node in model.graph.node] == ["Einsum"]
    assert calculate_params(model) == 130
    assert correction.param_count == 130
    equation = helper.get_attribute_value(model.graph.node[0].attribute[0]).decode()
    assert equation.endswith(",y,z,kv->bkrc")
    _assert_model_matches_bundled(197, model)


def test_fitter_recovers_task197_internal_gates_and_signed_mixer():
    query = canonicalize((("k", "r", "j"), ("a", "x", "c"), ("a", "y", "j")))
    official_count = len(load_task(197)["train"] + load_task(197)["test"])
    examples = [
        (arrays["input"], arrays["output"])
        for arrays in _examples(197)[:official_count]
    ]
    corrections = fit_small_corrections(query, examples)
    assert any(correction.param_count == 130 for correction in corrections)
