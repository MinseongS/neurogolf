from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from tools.single_node_family_search import (
    ATTENTION_UNDERFLOW_LOGIT_GAP,
    ATTENTION_DENSE_MAX,
    ATTENTION_DENSE_STEP,
    EXCLUDED_OP_TYPES,
    GRID_SHAPE,
    SCALAR_REJECTIONS,
    attention_scales,
    build_model,
    family_registry,
    layernorm_mixed_values,
    scalar_family_registry,
)


@pytest.mark.parametrize("family", family_registry(), ids=lambda family: family.name)
def test_every_registered_family_is_pinned_runtime_legal_and_same_shape(family):
    model = build_model(family)
    onnx.checker.check_model(model, full_check=True)
    assert len(model.graph.node) == 1
    assert not model.graph.initializer

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    actual = session.run(None, {"input": np.zeros(GRID_SHAPE, np.float32)})[0]
    assert list(actual.shape) == GRID_SHAPE


def test_registry_excludes_previously_searched_and_scorer_excluded_families():
    forbidden = {
        "AveragePool",
        "MaxPool",
        "LpPool",
        "CumSum",
        "ReverseSequence",
        "Trilu",
        "Hardmax",
        "Pad",
        "Slice",
        "Einsum",
        *EXCLUDED_OP_TYPES,
    }
    assert not ({family.op_type for family in family_registry()} & forbidden)


def test_transpose_is_present_only_as_an_explicit_control():
    transpose = [family for family in family_registry() if family.op_type == "Transpose"]
    assert transpose
    assert all(family.control for family in transpose)


def test_registry_covers_variadic_reduction_and_normalization_families():
    op_types = {family.op_type for family in family_registry() if not family.control}
    assert {"Add", "Mean", "Max", "Min", "Sum"} <= op_types
    assert {"LpNormalization", "MeanVarianceNormalization", "Softmax"} <= op_types


def test_registry_reuses_the_free_input_for_legal_required_tensor_operands():
    op_types = {family.op_type for family in family_registry() if not family.control}
    assert {"Attention", "CastLike", "LayerNormalization", "RMSNormalization"} <= op_types
    layer_norm_arities = {
        family.inputs for family in family_registry()
        if family.op_type == "LayerNormalization"
    }
    assert layer_norm_arities == {2, 3}


def test_attention_registry_covers_underflow_and_free_mask_regimes():
    regimes = {
        (
            family.inputs,
            family.attribute_dict["is_causal"],
            family.attribute_dict["scale"],
        )
        for family in family_registry()
        if family.op_type == "Attention"
    }
    expected = {
        (inputs, is_causal, scale)
        for inputs in (3, 4)
        for is_causal in (0, 1)
        for scale in (10.0, 50.0, 100.0, 1e6)
    }
    assert expected <= regimes


def test_attention_scale_policy_straddles_every_float32_support_threshold():
    assert 30.0 in attention_scales(inputs=3)
    assert 30.0 in attention_scales(inputs=4)
    for inputs, mask_offsets in ((3, (0,)), (4, (-1, 0, 1))):
        scales = set(attention_scales(inputs=inputs))
        for dot_gap in range(1, 31):
            for mask_offset in mask_offsets:
                threshold = np.float32(
                    (ATTENTION_UNDERFLOW_LOGIT_GAP - mask_offset) / dot_gap
                )
                below = float(np.nextafter(threshold, np.float32(0)))
                above = float(np.nextafter(threshold, np.float32(np.inf)))
                assert {below, above} <= scales


def test_attention_scale_policy_covers_pinned_kernel_quarter_grid_regressions():
    assert ATTENTION_DENSE_STEP == 0.25
    assert ATTENTION_DENSE_MAX == 120.0
    for inputs in (3, 4):
        scales = set(attention_scales(inputs=inputs))
        assert {11.0, 9.75} <= scales
        assert {
            step * ATTENTION_DENSE_STEP
            for step in range(1, int(ATTENTION_DENSE_MAX / ATTENTION_DENSE_STEP) + 1)
        } <= scales


@pytest.mark.parametrize(
    "family", scalar_family_registry(), ids=lambda family: family.name
)
def test_every_cost_one_family_is_strictly_legal_and_same_shape(family):
    model = build_model(family)
    assert sum(max(1, int(np.prod(init.dims))) for init in model.graph.initializer) == 1
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    inferred_shape = [
        dimension.dim_value
        for dimension in inferred.graph.output[0].type.tensor_type.shape.dim
    ]
    assert inferred_shape == GRID_SHAPE

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    actual = session.run(None, {"input": np.zeros(GRID_SHAPE, np.float32)})[0]
    assert list(actual.shape) == GRID_SHAPE


def test_cost_one_registry_covers_plausible_scalar_inputs_and_rejections():
    op_types = {family.op_type for family in scalar_family_registry()}
    assert {
        "Add", "Sub", "Mul", "Div", "Pow", "Mod", "Max", "Mean", "Min", "Sum",
        "Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual",
        "Clip", "PRelu", "QuantizeLinear", "Dropout", "Gather", "Split",
        "ReduceMax", "ReduceMean", "ReduceMin", "ReduceSum",
        "LayerNormalization", "RMSNormalization", "GroupNormalization",
    } <= op_types
    rejected = {row["operator"] for row in SCALAR_REJECTIONS}
    assert {"DFT", "TopK", "DequantizeLinear", "Squeeze", "Unsqueeze"} <= rejected


def test_cost_one_normalization_registry_covers_axes_signs_bias_and_groups():
    families = scalar_family_registry()
    layer = [family for family in families if family.op_type == "LayerNormalization"]
    rms = [family for family in families if family.op_type == "RMSNormalization"]
    group = [family for family in families if family.op_type == "GroupNormalization"]

    assert {family.attribute_dict["axis"] for family in layer} == set(range(-4, 4))
    assert {family.attribute_dict["axis"] for family in rms} == set(range(-4, 4))
    assert {family.scalar_value for family in layer} >= {-1.0, 0.0, 1.0}
    assert {family.scalar_value for family in rms} >= {-1.0, 0.0, 1.0}
    assert {family.inputs for family in layer} == {2, 3}
    assert all(
        family.input_pattern != ("input", "scalar", "scalar")
        or family.cost == 1
        for family in layer
    )
    assert {family.attribute_dict["num_groups"] for family in group} == {1, 2, 5, 10}
    assert all(family.input_pattern == ("input", "scalar", "scalar") for family in group)


def test_mixed_layernorm_values_straddle_every_arc_occupancy_threshold():
    for epsilon in (0.0, 1e-5, 1.0):
        bias_values = set(layernorm_mixed_values("input_scale_scalar_bias", epsilon))
        scale_values = set(layernorm_mixed_values("scalar_scale_input_bias", epsilon))
        assert -4.0 in bias_values
        for occupancy in range(1, 901):
            probability = np.float32(occupancy / 9000.0)
            denominator = np.sqrt(
                np.float32(probability * (1.0 - probability) + epsilon),
                dtype=np.float32,
            )
            normalized_one = np.float32((1.0 - probability) / denominator)
            for threshold, values in (
                (np.float32(-normalized_one), bias_values),
                (np.float32(-1.0 / normalized_one), scale_values),
            ):
                assert float(np.nextafter(threshold, np.float32(-np.inf))) in values
                assert float(np.nextafter(threshold, np.float32(np.inf))) in values


def test_mixed_layernorm_patterns_and_free_normalization_rejections_are_explicit():
    mixed = [
        family for family in scalar_family_registry()
        if family.op_type == "LayerNormalization"
        and family.input_pattern in {
            ("input", "input", "scalar"),
            ("input", "scalar", "input"),
        }
    ]
    assert {family.input_pattern for family in mixed} == {
        ("input", "input", "scalar"),
        ("input", "scalar", "input"),
    }
    assert {family.attribute_dict["axis"] for family in mixed} == {-4, 0}
    assert any(
        family.input_pattern == ("input", "input", "scalar")
        and family.scalar_value == -4.0
        for family in mixed
    )

    zero_rms = [family for family in family_registry() if family.op_type == "RMSNormalization"]
    assert {family.inputs for family in zero_rms} == {2}
    rejected_reasons = " ".join(row["reason"] for row in SCALAR_REJECTIONS)
    assert "GroupNormalization FREE scale/bias" in rejected_reasons
    assert "RMSNormalization scalar X" in rejected_reasons
