import math

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tools.campaign_inventory import (
    analyze_model,
    main,
    point_gain,
    rank_rows,
    render_markdown,
    risk_flags,
)


def toy_model():
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 4, 4])
    w = numpy_helper.from_array(np.ones((3, 1, 1, 1), dtype=np.float32), name="W")
    conv = helper.make_node("Conv", ["input", "W"], ["plane"], name="conv")
    reduce = helper.make_node("ReduceSum", ["plane"], ["output"], name="reduce", axes=[1], keepdims=1)
    graph = helper.make_graph([conv, reduce], "toy", [x], [y], [w])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


def test_point_gain_uses_neurogolf_log_formula():
    assert point_gain(1000, 100) == pytest.approx(math.log(1000 / 900))


def test_analyzer_counts_intermediate_bytes_but_initializer_elements():
    row = analyze_model(1, toy_model(), {"cost": 1000, "points": 18.0, "sha256": "abc"})
    assert row["largest_tensor"]["name"] == "plane"
    assert row["largest_tensor"]["saving"] == 3 * 4 * 4 * 4
    assert row["largest_param_bank"]["name"] == "W"
    assert row["largest_param_bank"]["saving"] == 3


def test_rank_rows_uses_best_single_removal_gain_then_task_number():
    rows = [
        {"task": 2, "optimistic_gain": 0.4},
        {"task": 3, "optimistic_gain": 0.8},
        {"task": 1, "optimistic_gain": 0.8},
    ]
    assert [row["task"] for row in rank_rows(rows)] == [1, 3, 2]


def test_risk_flags_protect_public_zero_repairs_and_large_einsums():
    model = toy_model()
    model.graph.node.append(
        helper.make_node(
            "Einsum",
            ["plane"] * 13,
            ["extra"],
            equation="ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab->ab",
        )
    )
    assert risk_flags(118, model) == [
        "protected-public-zero-repair",
        "runtime-heavy-einsum",
    ]


def test_render_markdown_has_required_columns_and_ranked_rows():
    rows = [
        {
            "task": 2,
            "cost": 200,
            "points": 19.7,
            "optimistic_gain": 0.4,
            "largest_tensor": {"name": "plane", "saving": 64},
            "largest_param_bank": {"name": "W", "saving": 3},
            "risk_flags": [],
        }
    ]
    markdown = render_markdown(rows, 7455.0891)
    assert "Baseline: 7455.0891" in markdown
    assert (
        "| rank | task | cost | points | optimistic_gain | largest_tensor | "
        "largest_param_bank | risk_flags |" in markdown
    )
    assert "| 1 | 002 | 200 |" in markdown


def test_cli_reports_input_errors_through_argparse(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--manifest",
                str(tmp_path / "missing.json"),
                "--nets",
                str(tmp_path),
                "--out",
                str(tmp_path / "out"),
            ]
        )
    assert exc_info.value.code == 2
