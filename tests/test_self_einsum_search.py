import json
import subprocess

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from neurogolf.scoring import calculate_params, convert_to_numpy, load_task
import tools.self_einsum_search as self_einsum_search
from tools.self_einsum_search import (
    Correction,
    build_model,
    canonicalize,
    evaluate_query,
    expand_query,
    fit_small_corrections,
    grammar_reaches,
    main,
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


def _assert_model_matches_bundled(task_num, model, *, ort_all=False):
    examples = _examples(task_num)
    assert examples
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    official_count = sum(
        convert_to_numpy(example) is not None
        for split in ("train", "test")
        for example in load_task(task_num)[split]
    )
    runtime_examples = examples if ort_all else examples[:official_count]
    for arrays in runtime_examples:
        actual = session.run(None, {"input": arrays["input"]})[0] > 0
        assert np.array_equal(actual, arrays["output"] > 0)

    # The task017 five-input contraction peaks around 7.6 GiB per ORT run and
    # takes several minutes over 266 generated cases. Keep its ORT control on
    # every official case, then cover every bundled case with the exact same
    # compiled equation in bounded NumPy batches. The cheaper task197 control
    # runs every bundled case through ORT via ``ort_all=True``.
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


def test_expansion_cache_returns_an_immutable_bounded_value():
    expanded = expand_query((("k", "r", "c"),))
    assert isinstance(expanded, frozenset)
    with pytest.raises(AttributeError):
        expanded.add((("k", "r", "c"),))
    assert self_einsum_search.expand_query.cache_info().maxsize == 8192


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
    hits = search_task(67, max_atoms=2, beam=500, correction_beam=0)
    assert any(hit["fail"] == 0 and hit["cost"] == 0 for hit in hits)


def test_search_screens_on_subset_but_reports_full_corpus_stats(monkeypatch):
    """Rows must carry full-corpus stats even though screening uses the subset.

    A query that is exact on the screening subset but wrong on the rest of the
    bundled corpus must not be reported as exact.
    """
    q1 = (("k", "r", "c"),)
    q2 = (("k", "r", "c"), ("u", "r", "c"))
    subset_only = (np.zeros((1, 10, 2, 2), dtype=np.float32), np.zeros((1, 10, 2, 2), dtype=bool))
    extra = (np.ones((1, 10, 2, 2), dtype=np.float32), np.ones((1, 10, 2, 2), dtype=bool))
    examples = [subset_only, extra]

    monkeypatch.setattr(
        self_einsum_search,
        "_task_examples_with_official_count",
        lambda task_num: (examples, 1),
    )
    monkeypatch.setattr(
        self_einsum_search,
        "select_diagnostic_examples",
        lambda examples, *, official_count, arc_limit: [subset_only],
    )
    monkeypatch.setattr(self_einsum_search, "_seed_queries", lambda: frozenset({q1}))
    monkeypatch.setattr(
        self_einsum_search,
        "expand_query",
        lambda query: frozenset({q2}) if query == q1 else frozenset(),
    )

    def fake_stats(query, received_examples, correction=None):
        # q1 looks exact on the 1-example subset but is wrong on the full corpus.
        if len(received_examples) == 1:
            return {q1: (0, 1, 0), q2: (4, 0, 1)}[query]
        return {q1: (7, 1, 1), q2: (9, 0, 2)}[query]

    monkeypatch.setattr(self_einsum_search, "_loss_stats", fake_stats)
    monkeypatch.setattr(self_einsum_search, "_runtime_validate", lambda *a, **k: True)
    monkeypatch.setattr(
        self_einsum_search, "fit_small_corrections", lambda query, ex: []
    )

    rows = search_task(1, max_atoms=2, beam=1, correction_beam=1)

    # q1 screened as exact on the subset, but the full corpus says 7 wrong cells, so it
    # is reported with full stats and never as a hit. q2 is the depth-2 near-hit.
    assert {(row["atoms"], row["wrong_cells"]) for row in rows} == {(1, 7), (2, 9)}
    assert all(row["wrong_cells"] != 0 for row in rows)


def test_search_ranks_every_bundled_example_and_continues_after_exact(monkeypatch):
    q1 = (("k", "r", "c"),)
    q2 = (("k", "r", "c"), ("u", "r", "c"))
    q3 = (("k", "r", "c"), ("u", "r", "c"), ("v", "r", "c"))
    examples = [
        (np.zeros((1, 10, 2, 2), dtype=np.float32), np.zeros((1, 10, 2, 2), dtype=bool)),
        (np.ones((1, 10, 2, 2), dtype=np.float32), np.ones((1, 10, 2, 2), dtype=bool)),
    ]
    observed_example_counts = []

    monkeypatch.setattr(
        self_einsum_search,
        "_task_examples_with_official_count",
        lambda task_num: (examples, 1),
    )
    # Screening subset == full corpus here, so the exact-hit/continue contract is
    # exercised independently of the two-stage split.
    monkeypatch.setattr(
        self_einsum_search,
        "select_diagnostic_examples",
        lambda examples, *, official_count, arc_limit: examples,
    )
    monkeypatch.setattr(self_einsum_search, "_seed_queries", lambda: frozenset({q1}))
    monkeypatch.setattr(
        self_einsum_search,
        "expand_query",
        lambda query: frozenset({q2}) if query == q1 else frozenset({q3}) if query == q2 else frozenset(),
    )

    def fake_stats(query, received_examples, correction=None):
        observed_example_counts.append(len(received_examples))
        return {q1: (3, 1, 1), q2: (0, 2, 0), q3: (5, 1, 1)}[query]

    monkeypatch.setattr(self_einsum_search, "_loss_stats", fake_stats)
    monkeypatch.setattr(self_einsum_search, "_runtime_validate", lambda *args, **kwargs: True)
    fitted_example_counts = []
    monkeypatch.setattr(
        self_einsum_search,
        "fit_small_corrections",
        lambda query, received_examples: fitted_example_counts.append(
            len(received_examples)
        )
        or [],
    )

    rows = search_task(1, max_atoms=3, beam=1, correction_beam=1)

    assert observed_example_counts and set(observed_example_counts) == {len(examples)}
    assert fitted_example_counts == [len(examples)]
    assert {(row["atoms"], row["wrong_cells"]) for row in rows} == {
        (1, 3),
        (2, 0),
        (3, 5),
    }


@pytest.mark.parametrize(
    "correction",
    [
        None,
        Correction(channel_masks=(("k", (1,) * 10),)),
    ],
)
def test_raw_and_corrected_runtime_validation_reject_timed_out_child(
    correction, monkeypatch
):
    observed = {}

    def timeout(*args, **kwargs):
        observed["timeout"] = kwargs.get("timeout")
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(self_einsum_search.subprocess, "run", timeout)
    raw = np.zeros((1, 10, 30, 30), dtype=np.float32)
    examples = [(raw, raw > 0)]
    assert not self_einsum_search._runtime_validate(
        (("k", "r", "c"),), correction, examples, timeout_seconds=0.25
    )
    assert observed["timeout"] == 0.25


def test_corrected_report_keeps_raw_equation_and_records_compiled_equation():
    query = (("k", "r", "c"),)
    correction = Correction(channel_masks=(("k", (1, 0, 1, 0, 1, 0, 1, 0, 1, 0)),))
    row = self_einsum_search._report_row(1, query, (0, 2, 0), correction)
    assert row["equation"] == "bkrc->bkrc"
    assert row["correction"]["compiled_equation"] == "bkrc,k->bkrc"


def test_resume_persists_zero_row_completion_and_rejects_config_drift(
    tmp_path, monkeypatch
):
    output = tmp_path / "hits.json"
    calls = []
    monkeypatch.setattr(
        self_einsum_search,
        "search_task",
        lambda task, **kwargs: calls.append(task) or [],
    )
    args = [
        "--tasks",
        "67",
        "--max-atoms",
        "1",
        "--beam",
        "1",
        "--output",
        str(output),
    ]
    assert main(args) == 0
    document = json.loads(output.read_text())
    assert document["schema_version"] == 1
    assert document["completed_tasks"] == [67]
    assert document["rows"] == []
    assert document["config_hash"]

    assert main(args) == 0
    assert calls == [67]
    with pytest.raises(ValueError, match="incompatible resume"):
        main([*args[:-2], "--correction-beam", "0", *args[-2:]])
    with pytest.raises(ValueError, match="incompatible resume"):
        main(
            [
                *args[:-2],
                "--shard-index",
                "1",
                "--shard-count",
                "2",
                *args[-2:],
            ]
        )
    document["config_hash"] = "corrupt"
    output.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="incompatible resume"):
        main(args)


def test_resume_rejects_malformed_persisted_report_row(tmp_path, monkeypatch):
    output = tmp_path / "hits.json"
    monkeypatch.setattr(
        self_einsum_search,
        "search_task",
        lambda task, **kwargs: [
            self_einsum_search._report_row(
                task, (("k", "r", "c"),), (0, 1, 0), None
            )
        ],
    )
    args = ["--tasks", "67", "--max-atoms", "1", "--beam", "1", "--output", str(output)]
    assert main(args) == 0
    document = json.loads(output.read_text())
    document["rows"][0].pop("equation")
    output.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="incompatible resume"):
        main(args)


@pytest.mark.parametrize(
    "atom",
    [
        ["k", "r"],
        ["k", "r", "c", "x"],
        ["k"],
    ],
)
def test_resume_rejects_report_row_whose_atom_arity_is_not_three(
    tmp_path, monkeypatch, atom
):
    """A resumed row must carry the same arity-3 atoms the grammar enforces.

    A row that survives validation is trusted verbatim: its task stays in
    completed_tasks, so it is never re-searched and the malformed row is
    re-emitted as that task's final result.
    """
    output = tmp_path / "hits.json"
    monkeypatch.setattr(
        self_einsum_search,
        "search_task",
        lambda task, **kwargs: [
            self_einsum_search._report_row(task, (("k", "r", "c"),), (0, 1, 0), None)
        ],
    )
    args = ["--tasks", "67", "--max-atoms", "1", "--beam", "1", "--output", str(output)]
    assert main(args) == 0
    document = json.loads(output.read_text())
    document["rows"][0]["query"] = [atom]
    document["rows"][0]["atoms"] = 1
    output.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="incompatible resume"):
        main(args)


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


def test_fitter_combines_arbitrary_mask_and_distinct_spatial_gates():
    rng = np.random.default_rng(7)
    query = canonicalize(
        (("k", "r", "c"), ("u", "r", "x"), ("v", "y", "c"))
    )
    mask = (1, 0, 1, 1, 0, 0, 1, 0, 1, 0)
    gate_x = tuple(int(index == 2) for index in range(30))
    gate_y = tuple(int(index == 4) for index in range(30))
    wanted = Correction(
        channel_masks=(("k", mask),),
        spatial_gates=(("x", gate_x), ("y", gate_y)),
    )
    examples = []
    for _ in range(2):
        raw = (rng.random((1, 10, 30, 30)) < 0.08).astype(np.float32)
        examples.append((raw, evaluate_query(query, raw, wanted)))
    corrections = fit_small_corrections(query, examples, max_params=70)
    assert any(
        correction.channel_masks == wanted.channel_masks
        and correction.spatial_gates == wanted.spatial_gates
        for correction in corrections
    )


def test_mixer_fits_zero_rows_permutations_and_general_signed_separators():
    patterns = np.asarray(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 1, 1],
            [2, 1, 1],
        ],
        dtype=np.float32,
    )
    raw = np.zeros((1, 10, 1, len(patterns)), dtype=np.float32)
    raw[0, :3, 0] = patterns.T
    target = np.zeros_like(raw, dtype=bool)
    target[:, 0] = raw[:, 2] > 0
    target[:, 1] = (raw[:, 0] - 2 * raw[:, 1] + raw[:, 2]) > 0
    mixer = self_einsum_search._fit_mixer([raw], [target])
    assert mixer is not None
    actual = np.einsum("bkrc,ok->borc", raw, np.asarray(mixer)) > 0
    assert np.array_equal(actual, target)
    assert mixer[2] == (0.0,) * 10


def test_task017_cost10_bounded_runtime_and_all_bundled_numpy_control():
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


def test_fitter_recovers_task017_shared_mask_from_all_bundled_examples():
    query = canonicalize(
        (
            ("a", "r", "x"),
            ("a", "p", "x"),
            ("b", "y", "c"),
            ("b", "y", "q"),
            ("k", "p", "q"),
        )
    )
    examples = [(arrays["input"], arrays["output"]) for arrays in _examples(17)]
    corrections = fit_small_corrections(query, examples, max_params=10)
    assert any(correction.param_count == 10 for correction in corrections)


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
    _assert_model_matches_bundled(197, model, ort_all=True)


def test_fitter_recovers_task197_internal_gates_and_signed_mixer():
    query = canonicalize((("k", "r", "j"), ("a", "x", "c"), ("a", "y", "j")))
    examples = [(arrays["input"], arrays["output"]) for arrays in _examples(197)]
    corrections = fit_small_corrections(query, examples)
    assert any(correction.param_count == 130 for correction in corrections)
