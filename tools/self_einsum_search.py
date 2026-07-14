from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import TypeAlias

import numpy as np
import onnx
from onnx import TensorProto, helper

Query: TypeAlias = tuple[tuple[str, str, str], ...]
Vertex: TypeAlias = tuple[int, str]
GraphAtom: TypeAlias = tuple[Vertex, Vertex, Vertex]

_COLOR_POOL = tuple("uvwlmnopqst")
_SPACE_POOL = tuple("xyzadefghij")
_ANCHORS: dict[Vertex, str] = {
    (0, "k"): "k",
    (1, "r"): "r",
    (1, "c"): "c",
}
_ANCHOR_RANK = {vertex: rank for rank, vertex in enumerate(_ANCHORS)}
_COLOR_RANK = {label: index for index, label in enumerate(("k", *_COLOR_POOL))}
_SPACE_RANK = {
    label: index for index, label in enumerate(("r", "c", *_SPACE_POOL))
}


def canonicalize(query: Query) -> Query:
    """Return the canonical form of a valid unordered typed query."""
    return _canonicalize_valid(_validate_query(query))


def _validate_query(query: Query) -> Query:
    if not isinstance(query, tuple) or not query:
        raise ValueError("query must be a non-empty tuple")

    colors: set[str] = set()
    spaces: set[str] = set()
    for atom in query:
        if not isinstance(atom, tuple) or len(atom) != 3:
            raise ValueError("every query atom must contain exactly three labels")
        if not all(isinstance(label, str) and label for label in atom):
            raise ValueError("query labels must be non-empty strings")
        color, row, col = atom
        colors.add(color)
        spaces.update((row, col))

    internal_colors = colors - {"k"}
    internal_spaces = spaces - {"r", "c"}
    # Raw internal names are remapped; only k/r/c carry anchored axis semantics.
    if colors & {"r", "c"} or "k" in spaces or internal_colors & internal_spaces:
        raise ValueError("query labels cannot be reserved or shared across types")
    if "k" not in colors or not {"r", "c"} <= spaces:
        raise ValueError("query operands must supply output labels k, r, and c")
    if (
        len(internal_colors) > len(_COLOR_POOL)
        or len(internal_spaces) > len(_SPACE_POOL)
    ):
        raise ValueError("query exceeds the typed label pool capacity")
    return query


def _partition(signatures: dict[Vertex, tuple]) -> dict[Vertex, int]:
    ranks = {
        signature: rank
        for rank, signature in enumerate(sorted(set(signatures.values())))
    }
    return {vertex: ranks[signature] for vertex, signature in signatures.items()}


def _refine(
    vertices: frozenset[Vertex],
    atoms: tuple[GraphAtom, ...],
    individualized: tuple[Vertex, ...],
) -> dict[Vertex, int]:
    individual_ranks = {vertex: rank for rank, vertex in enumerate(individualized)}
    base: dict[Vertex, tuple] = {}
    for vertex in vertices:
        if vertex in _ANCHORS:
            base[vertex] = (0, _ANCHOR_RANK[vertex], 0)
        elif vertex in individual_ranks:
            base[vertex] = (1, vertex[0], individual_ranks[vertex])
        else:
            base[vertex] = (2, vertex[0], 0)
    classes = _partition(base)

    while True:
        signatures: dict[Vertex, tuple] = {}
        for vertex in vertices:
            incident = []
            for atom in atoms:
                if vertex in atom:
                    incident.append(
                        tuple(
                            -1 if neighbor == vertex else classes[neighbor]
                            for neighbor in atom
                        )
                    )
            signatures[vertex] = (classes[vertex], tuple(sorted(incident)))
        refined = _partition(signatures)
        if refined == classes:
            return refined
        classes = refined


def _is_twin_cell(cell: tuple[Vertex, ...], atoms: tuple[GraphAtom, ...]) -> bool:
    atom_counts = Counter(atoms)
    for left_index, left in enumerate(cell):
        for right in cell[left_index + 1 :]:
            swapped = Counter(
                tuple(
                    right if vertex == left else left if vertex == right else vertex
                    for vertex in atom
                )
                for atom in atoms
            )
            if swapped != atom_counts:
                return False
    return True


def _query_key(query: Query) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (_COLOR_RANK[color], _SPACE_RANK[row], _SPACE_RANK[col])
        for color, row, col in query
    )


def _render(
    vertices: frozenset[Vertex],
    atoms: tuple[GraphAtom, ...],
    classes: dict[Vertex, int],
) -> Query:
    internal_colors = sorted(
        (vertex for vertex in vertices if vertex[0] == 0 and vertex not in _ANCHORS),
        key=lambda vertex: (classes[vertex], vertex[1]),
    )
    internal_spaces = sorted(
        (vertex for vertex in vertices if vertex[0] == 1 and vertex not in _ANCHORS),
        key=lambda vertex: (classes[vertex], vertex[1]),
    )
    labels = dict(_ANCHORS)
    labels.update(
        zip(internal_colors, _COLOR_POOL[: len(internal_colors)], strict=True)
    )
    labels.update(
        zip(internal_spaces, _SPACE_POOL[: len(internal_spaces)], strict=True)
    )
    rendered = tuple(
        (labels[color], labels[row], labels[col]) for color, row, col in atoms
    )
    return tuple(sorted(rendered, key=lambda atom: _query_key((atom,))[0]))


@lru_cache(maxsize=65_536)
def _canonicalize_valid(query: Query) -> Query:
    atoms: tuple[GraphAtom, ...] = tuple(
        ((0, color), (1, row), (1, col)) for color, row, col in query
    )
    vertices = frozenset(vertex for atom in atoms for vertex in atom)

    def search(individualized: tuple[Vertex, ...]) -> Query:
        classes = _refine(vertices, atoms, individualized)
        cells: dict[int, list[Vertex]] = {}
        for vertex in vertices - _ANCHORS.keys():
            cells.setdefault(classes[vertex], []).append(vertex)
        unresolved = [
            tuple(sorted(cell))
            for cell in cells.values()
            if len(cell) > 1 and not _is_twin_cell(tuple(cell), atoms)
        ]
        if not unresolved:
            return _render(vertices, atoms, classes)

        cell = min(unresolved, key=lambda values: (len(values), classes[values[0]]))
        candidates = [search((*individualized, vertex)) for vertex in cell]
        return min(candidates, key=_query_key)

    return search(())


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
