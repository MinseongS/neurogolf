"""Census zero-parameter, same-shape, single-node ONNX families.

Previously searched spatial families are deliberately absent.  Transpose is kept
only as a labelled control because tasks 179 and 241 are known zero-cost controls.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from itertools import combinations
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from neurogolf.scoring import convert_to_numpy, load_task


GRID_SHAPE = [1, 10, 30, 30]
EXCLUDED_OP_TYPES = {
    "Loop",
    "Scan",
    "NonZero",
    "Unique",
    "Script",
    "Function",
    "Compress",
}

# exp(-x), rounded to FLOAT32, becomes zero after half the smallest positive
# subnormal.  Attention logits differ by ``scale * integer_dot_gap`` and, when
# the same FREE input is reused as attn_mask, an additional {-1, 0, +1}.
ATTENTION_UNDERFLOW_LOGIT_GAP = -math.log(
    float(np.nextafter(np.float32(0), np.float32(1))) / 2.0
)
ATTENTION_DENSE_STEP = 0.25
ATTENTION_DENSE_MAX = 120.0


def attention_scales(*, inputs: int) -> tuple[float, ...]:
    """Support-complete representatives for one-hot self-Attention.

    Q=K=input makes every dot product an integer in 0..30.  These scales
    straddle every FLOAT32 exp-underflow boundary between two such logits.
    Four-input Attention also reuses input as its additive 0/1 mask, so all
    possible mask differences are included.
    """
    if inputs not in (3, 4):
        raise ValueError("Attention family supports only 3 or 4 repeated inputs")
    # Pinned ORT 1.26's CPU Attention kernel flushes a controlled gap-1 low
    # contribution between 87.50 and 87.75 (88.50/88.75 with a +1 mask gap),
    # earlier than scalar FLOAT32 exp.  Keep the complete quarter-grid through
    # 120 as empirical kernel coverage, plus the analytic nextafter boundaries.
    values = {
        step * ATTENTION_DENSE_STEP
        for step in range(1, int(ATTENTION_DENSE_MAX / ATTENTION_DENSE_STEP) + 1)
    }
    values.update((0.01, 1e6))
    mask_offsets = (0,) if inputs == 3 else (-1, 0, 1)
    for dot_gap in range(1, 31):
        for mask_offset in mask_offsets:
            threshold = np.float32(
                (ATTENTION_UNDERFLOW_LOGIT_GAP - mask_offset) / dot_gap
            )
            values.add(float(np.nextafter(threshold, np.float32(0))))
            values.add(float(np.nextafter(threshold, np.float32(np.inf))))
    return tuple(sorted(values))


@dataclass(frozen=True)
class Family:
    name: str
    op_type: str
    inputs: int = 1
    attributes: tuple[tuple[str, object], ...] = ()
    opset: int = 20
    output_dtype: int = TensorProto.FLOAT
    control: bool = False
    input_pattern: tuple[str, ...] = ()
    scalar_value: float | int | bool | None = None
    scalar_dtype: int | None = None
    scalar_shape: tuple[int, ...] = ()

    @property
    def attribute_dict(self) -> dict[str, object]:
        return dict(self.attributes)

    @property
    def cost(self) -> int:
        return 1 if self.scalar_dtype is not None else 0


def _family(
    op_type: str,
    *,
    inputs: int = 1,
    opset: int = 20,
    output_dtype: int = TensorProto.FLOAT,
    control: bool = False,
    **attributes: object,
) -> Family:
    parts = [f"inputs={inputs}"] if inputs != 1 else []
    parts.extend(f"{key}={attributes[key]}" for key in sorted(attributes))
    suffix = ",".join(parts)
    name = op_type if not suffix else f"{op_type}[{suffix}]"
    return Family(
        name=name,
        op_type=op_type,
        inputs=inputs,
        attributes=tuple(sorted(attributes.items())),
        opset=opset,
        output_dtype=output_dtype,
        control=control,
    )


def _scalar_family(
    op_type: str,
    pattern: tuple[str, ...],
    value: float | int | bool,
    *,
    scalar_dtype: int = TensorProto.FLOAT,
    scalar_shape: tuple[int, ...] = (),
    opset: int = 20,
    output_dtype: int = TensorProto.FLOAT,
    **attributes: object,
) -> Family:
    attr_text = ",".join(f"{key}={attributes[key]}" for key in sorted(attributes))
    pieces = [f"pattern={''.join('S' if item == 'scalar' else 'X' if item == 'input' else '_' for item in pattern)}",
              f"scalar={value}", f"dtype={scalar_dtype}"]
    if attr_text:
        pieces.append(attr_text)
    return Family(
        name=f"{op_type}[{','.join(pieces)}]",
        op_type=op_type,
        inputs=len(pattern),
        attributes=tuple(sorted(attributes.items())),
        opset=opset,
        output_dtype=output_dtype,
        input_pattern=pattern,
        scalar_value=value,
        scalar_dtype=scalar_dtype,
        scalar_shape=scalar_shape,
    )


SCALAR_REJECTIONS = (
    {"operator": "DFT", "reason": "FLOAT input last dimension is 30, but DFT requires real/complex width 1 or 2"},
    {"operator": "TopK", "reason": "second required indices output cannot be omitted and would be a counted full-shape tensor"},
    {"operator": "DequantizeLinear", "reason": "quantized X determines output shape; a one-element X produces a scalar, while FLOAT graph input cannot serve as X"},
    {"operator": "Squeeze", "reason": "the only singleton input axis is batch axis 0, whose removal changes rank and shape"},
    {"operator": "Unsqueeze", "reason": "every scalar axes input adds a dimension and changes rank"},
    {"operator": "Reshape", "reason": "a one-element shape tensor requests rank one, not [1,10,30,30]"},
    {"operator": "Expand", "reason": "a one-element shape tensor cannot describe the required rank-four shape"},
    {"operator": "Tile", "reason": "repeats must have one value per input dimension"},
    {"operator": "Resize", "reason": "one-element scales or sizes cannot preserve all four input dimensions"},
    {"operator": "ConstantOfShape", "reason": "one-element shape input creates rank one, not the grid shape"},
    {"operator": "Attention", "reason": "dense one-element mask cannot satisfy the required 30x30 query/key dimensions"},
    {"operator": "GroupNormalization", "reason": "GroupNormalization FREE scale/bias has incompatible full-grid dimensions; only scalar broadcasts are legal"},
    {"operator": "RMSNormalization", "reason": "RMSNormalization scalar X fixes a scalar output shape; FREE input is legal only in the scale slot"},
)


def layernorm_mixed_values(pattern: str, epsilon: float) -> tuple[float, ...]:
    """Support-complete scalar representatives for mixed axis-0 LayerNorm.

    ARC one-hot canvases contain N=1..900 ones in the fixed 9000-element tensor.
    With Scale=input and scalar B, one-cell support changes at B=-norm(1).
    With scalar Scale and B=input, it changes at Scale=-1/norm(1); zero-cell
    support additionally changes at Scale=0.
    """
    if pattern not in {"input_scale_scalar_bias", "scalar_scale_input_bias"}:
        raise ValueError("unknown mixed LayerNormalization pattern")
    if epsilon not in (0.0, 1e-5, 1.0):
        raise ValueError("unsupported epsilon regime")
    values = {-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0}
    if pattern == "scalar_scale_input_bias":
        zero = np.float32(0)
        values.add(float(np.nextafter(zero, np.float32(-np.inf))))
        values.add(float(np.nextafter(zero, np.float32(np.inf))))
    for occupancy in range(1, 901):
        probability = np.float32(occupancy / 9000.0)
        variance = np.float32(probability * (1.0 - probability))
        denominator = np.sqrt(np.float32(variance + epsilon), dtype=np.float32)
        normalized_one = np.float32((1.0 - probability) / denominator)
        threshold = np.float32(
            -normalized_one
            if pattern == "input_scale_scalar_bias"
            else -1.0 / normalized_one
        )
        values.add(float(np.nextafter(threshold, np.float32(-np.inf))))
        values.add(float(np.nextafter(threshold, np.float32(np.inf))))
    return tuple(sorted(values))


def scalar_family_registry() -> tuple[Family, ...]:
    """Schema-legal one-element-initializer, direct-output candidates."""
    families: list[Family] = []
    values = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 30.0)

    for op_type in ("Add", "Sub", "Mul", "Div", "Pow"):
        for pattern in (("input", "scalar"), ("scalar", "input")):
            families.extend(_scalar_family(op_type, pattern, value) for value in values)
    for pattern in (("input", "scalar"), ("scalar", "input")):
        families.extend(
            _scalar_family("Mod", pattern, value, fmod=1)
            for value in values if value != 0
        )
    families.extend(
        _scalar_family("PRelu", ("input", "scalar"), value)
        for value in values
    )
    for op_type in ("Max", "Mean", "Min", "Sum"):
        families.extend(
            _scalar_family(op_type, ("input", "scalar"), value)
            for value in values
        )
    for op_type in ("Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual"):
        for pattern in (("input", "scalar"), ("scalar", "input")):
            families.extend(
                _scalar_family(op_type, pattern, value, output_dtype=TensorProto.BOOL)
                for value in values
            )

    for value in values:
        families.append(_scalar_family("Clip", ("input", "scalar"), value))
        families.append(_scalar_family("Clip", ("input", "", "scalar"), value))

    families.extend(
        _scalar_family(
            "QuantizeLinear", ("input", "scalar"), scale,
            output_dtype=TensorProto.UINT8,
        )
        for scale in (0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 30.0)
    )
    families.extend(
        _scalar_family("Dropout", ("input", "scalar"), ratio, opset=20)
        for ratio in (0.0, 0.5, 0.9)
    )
    families.append(
        _scalar_family(
            "Gather", ("input", "scalar"), 0,
            scalar_dtype=TensorProto.INT64, scalar_shape=(1,), axis=0,
        )
    )
    for axis, size in enumerate(GRID_SHAPE):
        families.append(
            _scalar_family(
                "Split", ("input", "scalar"), size,
                scalar_dtype=TensorProto.INT64, scalar_shape=(1,), opset=18, axis=axis,
            )
        )

    reduce_opsets = {
        "ReduceL1": 18, "ReduceL2": 18, "ReduceLogSum": 18,
        "ReduceLogSumExp": 18, "ReduceMax": 20, "ReduceMean": 18,
        "ReduceMin": 20, "ReduceProd": 18, "ReduceSum": 13,
        "ReduceSumSquare": 18,
    }
    for op_type, opset in reduce_opsets.items():
        families.append(
            _scalar_family(
                op_type, ("input", "scalar"), 0,
                scalar_dtype=TensorProto.INT64, scalar_shape=(1,), opset=opset,
                keepdims=1,
            )
        )

    for condition in (False, True):
        families.append(
            _scalar_family(
                "Where", ("scalar", "input", "input"), condition,
                scalar_dtype=TensorProto.BOOL,
            )
        )

    normalization_values = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
    normalization_epsilons = (0.0, 1e-5, 1.0)
    for axis in range(-4, 4):
        for epsilon in normalization_epsilons:
            for value in normalization_values:
                families.append(
                    _scalar_family(
                        "LayerNormalization", ("input", "scalar"), value,
                        opset=17, axis=axis, epsilon=epsilon,
                    )
                )
                families.append(
                    _scalar_family(
                        "LayerNormalization",
                        ("input", "scalar", "scalar"), value,
                        opset=17, axis=axis, epsilon=epsilon,
                    )
                )
                families.append(
                    _scalar_family(
                        "RMSNormalization", ("input", "scalar"), value,
                        opset=23, axis=axis, epsilon=epsilon,
                    )
                )
    for num_groups in (1, 2, 5, 10):
        for epsilon in normalization_epsilons:
            for value in normalization_values:
                families.append(
                    _scalar_family(
                        "GroupNormalization",
                        ("input", "scalar", "scalar"), value,
                        opset=21, num_groups=num_groups, epsilon=epsilon,
                    )
                )

    for axis in (-4, 0):
        for epsilon in normalization_epsilons:
            for value in layernorm_mixed_values(
                "input_scale_scalar_bias", epsilon
            ):
                families.append(
                    _scalar_family(
                        "LayerNormalization",
                        ("input", "input", "scalar"), value,
                        opset=17, axis=axis, epsilon=epsilon,
                    )
                )
            for value in layernorm_mixed_values(
                "scalar_scale_input_bias", epsilon
            ):
                families.append(
                    _scalar_family(
                        "LayerNormalization",
                        ("input", "scalar", "input"), value,
                        opset=17, axis=axis, epsilon=epsilon,
                    )
                )

    unique = {family.name: family for family in families}
    return tuple(unique[name] for name in sorted(unique))


def family_registry() -> tuple[Family, ...]:
    """Concrete pinned-runtime families with no initializer and a 1x10x30x30 output."""
    families: list[Family] = []

    # Elementwise unary standard-domain operators accepting FLOAT.
    for op_type in (
        "Abs", "Acos", "Acosh", "Asin", "Asinh", "Atan", "Atanh",
        "Ceil", "Celu", "Clip", "Cos", "Cosh", "Dropout", "Elu", "Erf",
        "Exp", "Floor", "Gelu", "HardSigmoid", "HardSwish", "Identity",
        "LeakyRelu", "Log", "Mish", "Neg", "Reciprocal", "Relu", "Round",
        "Selu", "Shrink", "Sigmoid", "Sign", "Sin", "Sinh", "Softplus",
        "Softsign", "Sqrt", "Tan", "Tanh", "ThresholdedRelu",
    ):
        families.append(_family(op_type))

    # Attribute values that can change zero/one support, not just numerics.
    families.extend(_family("Celu", alpha=alpha) for alpha in (0.25, 2.0))
    families.extend(
        _family("HardSigmoid", alpha=alpha, beta=beta)
        for alpha, beta in ((0.2, 0.0), (0.2, -0.2), (-0.2, 0.2))
    )
    families.extend(_family("Shrink", lambd=lambd, bias=bias)
                    for lambd, bias in ((0.0, 0.0), (0.5, -1.0), (1.0, 1.0)))
    families.extend(_family("ThresholdedRelu", alpha=alpha)
                    for alpha in (-1.0, 0.0, 0.5, 1.0))
    families.extend(_family("LRN", size=size, alpha=alpha, beta=beta, bias=bias)
                    for size, alpha, beta, bias in ((1, 0.0001, 0.75, 1.0),
                                                    (3, 1.0, 0.5, 0.0)))

    families.extend(_family("IsInf", output_dtype=TensorProto.BOOL,
                            detect_negative=negative, detect_positive=positive)
                    for negative, positive in ((0, 0), (0, 1), (1, 0), (1, 1)))
    families.append(_family("IsNaN", output_dtype=TensorProto.BOOL))
    for to in (TensorProto.BOOL, TensorProto.FLOAT16, TensorProto.DOUBLE,
               TensorProto.INT8, TensorProto.INT16, TensorProto.INT32,
               TensorProto.INT64, TensorProto.UINT8, TensorProto.UINT16,
               TensorProto.UINT32, TensorProto.UINT64):
        families.append(_family("Cast", output_dtype=to, to=to))

    # Same-input binary/variadic arithmetic and comparisons.
    for op_type in ("Add", "Div", "Mul", "Pow", "Sub", "PRelu"):
        families.append(_family(op_type, inputs=2))
    families.append(_family("CastLike", inputs=2))
    for op_type in ("Max", "Mean", "Min", "Sum"):
        families.extend(_family(op_type, inputs=count) for count in (1, 2, 3))
    for op_type in ("Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual"):
        families.append(_family(op_type, inputs=2, output_dtype=TensorProto.BOOL))
    # FLOAT Mod is legal only in fmod mode in the pinned CPU kernel.
    families.append(_family("Mod", inputs=2, fmod=1))

    # Rank-4 self-MatMul happens to retain [1,10,30,30].
    families.append(_family("MatMul", inputs=2))
    # Opset-23 Attention accepts rank-4 Q/K/V.  Reusing the FREE input for all
    # three produces a same-shape self-attention family without parameters.
    families.extend(
        _family("Attention", inputs=inputs, opset=23,
                is_causal=is_causal, scale=scale)
        for inputs in (3, 4)
        for is_causal in (0, 1)
        for scale in attention_scales(inputs=inputs)
    )

    # Normalization axes/attributes preserve rank and can change spatial support.
    for axis in range(-4, 4):
        families.extend(_family("LpNormalization", axis=axis, p=p) for p in (1, 2))
        families.append(_family("Softmax", axis=axis))
        families.append(_family("LogSoftmax", axis=axis))
    for width in range(1, 5):
        for axes in combinations(range(4), width):
            families.append(_family("MeanVarianceNormalization", axes=list(axes)))
    families.extend(
        _family("LayerNormalization", inputs=inputs, opset=17, axis=0, epsilon=epsilon)
        for inputs in (2, 3)
        for epsilon in (0.0, 1e-5, 1.0)
    )
    families.extend(
        _family("RMSNormalization", inputs=2, opset=23, axis=axis, epsilon=epsilon)
        for axis in (-4, 0)
        for epsilon in (0.0, 1e-5, 1.0)
    )

    # Reducing the singleton batch axis keeps the exact declared shape.  Opset 11
    # is intentional: these schemas still express axes as a zero-cost attribute.
    for op_type in (
        "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceMax",
        "ReduceMean", "ReduceMin", "ReduceProd", "ReduceSum", "ReduceSumSquare",
    ):
        families.append(_family(op_type, opset=11, axes=[0], keepdims=1))
    families.extend(
        _family(op_type, opset=20, axis=0, keepdims=1, select_last_index=last,
                output_dtype=TensorProto.INT64)
        for op_type in ("ArgMax", "ArgMin")
        for last in (0, 1)
    )

    # Shape-preserving degenerate structural operators.
    families.extend(_family("Concat", axis=axis) for axis in range(-4, 4))
    families.append(_family("DepthToSpace", blocksize=1, mode="DCR"))
    families.append(_family("DepthToSpace", blocksize=1, mode="CRD"))
    families.append(_family("SpaceToDepth", blocksize=1))

    # Prior work owns this family; retain only the non-identity axis-swap control.
    families.append(_family("Transpose", perm=[0, 1, 3, 2], control=True))

    unique = {family.name: family for family in families}
    return tuple(unique[name] for name in sorted(unique))


def build_model(family: Family) -> onnx.ModelProto:
    node_inputs = (
        list(family.input_pattern)
        if family.input_pattern
        else ["input"] * family.inputs
    )
    node = helper.make_node(
        family.op_type,
        node_inputs,
        ["output"],
        name="output",
        **family.attribute_dict,
    )
    initializers = []
    if family.scalar_dtype is not None:
        initializers.append(
            helper.make_tensor(
                "scalar",
                family.scalar_dtype,
                list(family.scalar_shape),
                [family.scalar_value],
            )
        )
    graph = helper.make_graph(
        [node],
        f"single_node_{family.op_type}",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, GRID_SHAPE)],
        [helper.make_tensor_value_info("output", family.output_dtype, GRID_SHAPE)],
        initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", family.opset)]
    )
    model.ir_version = 10
    return model


def _session(family: Family) -> ort.InferenceSession:
    model = build_model(family)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def parse_tasks(values: Iterable[str]) -> list[int]:
    tasks: set[int] = set()
    for value in values:
        if "-" in value:
            start_text, end_text = value.split("-", 1)
            start, end = int(start_text), int(end_text)
            tasks.update(range(start, end + 1))
        else:
            tasks.add(int(value))
    if not tasks or min(tasks) < 1 or max(tasks) > 400:
        raise ValueError("tasks must be in 1..400")
    return sorted(tasks)


def _examples(task_num: int) -> list[dict[str, np.ndarray]]:
    task = load_task(task_num)
    return [
        arrays
        for split in ("train", "test", "arc-gen")
        for example in task.get(split, [])
        if (arrays := convert_to_numpy(example)) is not None
    ]


def scan(tasks: list[int], output: Path) -> dict:
    legality: list[dict] = []
    legal: list[tuple[Family, ort.InferenceSession]] = []
    all_families = (*family_registry(), *scalar_family_registry())
    for family in all_families:
        row = {
            "name": family.name,
            "operator": family.op_type,
            "attributes": family.attribute_dict,
            "inputs": family.inputs,
            "opset": family.opset,
            "control": family.control,
            "params": family.cost,
            "memory": 0,
            "scalar": (
                {
                    "value": family.scalar_value,
                    "dtype": family.scalar_dtype,
                    "shape": list(family.scalar_shape),
                    "input_pattern": list(family.input_pattern),
                }
                if family.cost
                else None
            ),
        }
        try:
            session = _session(family)
            actual = session.run(None, {"input": np.zeros(GRID_SHAPE, np.float32)})[0]
            if list(actual.shape) != GRID_SHAPE:
                raise ValueError(f"runtime output shape {list(actual.shape)}")
        except Exception as error:
            row.update(legal=False, error=f"{type(error).__name__}: {error}")
        else:
            row.update(legal=True, error=None)
            legal.append((family, session))
        legality.append(row)

    hits: list[dict] = []
    for task_num in tasks:
        examples = _examples(task_num)
        if not examples:
            continue
        for family, session in legal:
            exact = True
            for arrays in examples:
                try:
                    actual = session.run(None, {"input": arrays["input"]})[0]
                except Exception:
                    exact = False
                    break
                if not np.array_equal(actual > 0, arrays["output"] > 0):
                    exact = False
                    break
            if exact:
                hit = {
                    "task": task_num,
                    "name": family.name,
                    "operator": family.op_type,
                    "attributes": family.attribute_dict,
                    "opset": family.opset,
                    "control": family.control,
                    "cost": family.cost,
                    "points": 25.0,
                    "examples": len(examples),
                }
                hits.append(hit)
                print("HIT", json.dumps(hit, sort_keys=True), flush=True)
        if task_num % 25 == 0:
            print(f"scanned {task_num}/400 hits={len(hits)}", flush=True)

    report = {
        "scope": tasks,
        "registry_size": len(legality),
        "zero_cost_registry_size": len(family_registry()),
        "cost_one_registry_size": len(scalar_family_registry()),
        "legal_count": sum(row["legal"] for row in legality),
        "excluded_previous": [
            "Pool", "CumSum", "ReverseSequence", "Trilu", "Hardmax",
            "Pad", "Slice", "self-Einsum",
        ],
        "transpose_role": "control-only",
        "scorer_excluded": sorted(EXCLUDED_OP_TYPES) + ["Sequence*"],
        "cost_one_rejections": SCALAR_REJECTIONS,
        "legality": legality,
        "hits": hits,
        "new_hits": [hit for hit in hits if not hit["control"]],
        "cost_one_hits": [hit for hit in hits if hit["cost"] == 1],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["1-400"])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("candidates/direct_discovery/single_node_family_hits.json"),
    )
    args = parser.parse_args()
    report = scan(parse_tasks(args.tasks), args.output)
    print(json.dumps({
        "registry_size": report["registry_size"],
        "legal_count": report["legal_count"],
        "hits": len(report["hits"]),
        "new_hits": len(report["new_hits"]),
        "tasks": sorted({hit["task"] for hit in report["new_hits"]}),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
