from __future__ import annotations

from typing import TypeAlias

import numpy as np
import onnx
from onnx import TensorProto, helper

Query: TypeAlias = tuple[tuple[str, str, str], ...]


def canonicalize(query: Query) -> Query:
    color_map: dict[str, str] = {"k": "k"}
    space_map: dict[str, str] = {"r": "r", "c": "c"}
    # Keep typed label pools disjoint and reserve b/k/r/c for batch/output axes.
    color_names = iter("uvwlmnopqst")
    space_names = iter("xyzadefghij")
    out = []
    for color, row, col in query:
        if color not in color_map:
            color_map[color] = next(color_names)
        for value in (row, col):
            if value not in space_map:
                space_map[value] = next(space_names)
        out.append((color_map[color], space_map[row], space_map[col]))
    return tuple(sorted(out))


def to_equation(query: Query) -> str:
    operands = ["b" + "".join(atom) for atom in canonicalize(query)]
    return ",".join(operands) + "->bkrc"


def evaluate_query(query: Query, input_onehot: np.ndarray) -> np.ndarray:
    equation = to_equation(query)
    values = np.einsum(equation, *([input_onehot] * len(query)), optimize="greedy")
    return values > 0


def build_model(query: Query) -> onnx.ModelProto:
    equation = to_equation(query)
    node = helper.make_node(
        "Einsum",
        ["input"] * len(query),
        ["output"],
        name="output",
        equation=equation,
    )

    def tensor(name: str) -> onnx.ValueInfoProto:
        return helper.make_tensor_value_info(
            name, TensorProto.FLOAT, [1, 10, 30, 30]
        )

    graph = helper.make_graph([node], "self_einsum", [tensor("input")], [tensor("output")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model
