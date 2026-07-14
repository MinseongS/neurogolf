from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Iterable, Sequence, TypeAlias

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from neurogolf.scoring import convert_to_numpy, load_task

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
_SPACE_RANK = {label: index for index, label in enumerate(("r", "c", *_SPACE_POOL))}

_GRID_SHAPE = [1, 10, 30, 30]


@dataclass(frozen=True)
class Correction:
    """Small operands folded into the graph-output Einsum.

    Labels refer to the canonical raw query. Equal tensors are deliberately shared,
    so applying one mask or gate to several labels pays for it only once.
    """

    channel_masks: tuple[tuple[str, tuple[float, ...]], ...] = ()
    spatial_gates: tuple[tuple[str, tuple[float, ...]], ...] = ()
    channel_mixer: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        if len({label for label, _ in self.channel_masks}) != len(self.channel_masks):
            raise ValueError("channel mask labels must be unique")
        if len({label for label, _ in self.spatial_gates}) != len(self.spatial_gates):
            raise ValueError("spatial gate labels must be unique")
        if any(len(values) != 10 for _, values in self.channel_masks):
            raise ValueError("channel masks must have 10 elements")
        if any(len(values) != 30 for _, values in self.spatial_gates):
            raise ValueError("spatial gates must have 30 elements")
        if self.channel_mixer is not None and (
            len(self.channel_mixer) != 10
            or any(len(row) != 10 for row in self.channel_mixer)
        ):
            raise ValueError("channel mixer must have shape [10, 10]")

    @property
    def channel_mask(self) -> tuple[float, ...] | None:
        """Convenience accessor for the common single-mask correction."""
        if not self.channel_masks:
            return None
        return self.channel_masks[0][1]

    @property
    def param_count(self) -> int:
        unique: set[tuple[tuple[int, ...], bytes]] = set()
        for _, values in (*self.channel_masks, *self.spatial_gates):
            array = np.asarray(values, dtype=np.float32)
            unique.add((array.shape, array.tobytes()))
        if self.channel_mixer is not None:
            array = np.asarray(self.channel_mixer, dtype=np.float32)
            unique.add((array.shape, array.tobytes()))
        return sum(math.prod(shape) for shape, _ in unique)

    def as_json(self) -> dict:
        return {
            "channel_masks": [
                {"label": label, "values": list(values)}
                for label, values in self.channel_masks
            ],
            "spatial_gates": [
                {"label": label, "values": list(values)}
                for label, values in self.spatial_gates
            ],
            "channel_mixer": (
                [list(row) for row in self.channel_mixer]
                if self.channel_mixer is not None
                else None
            ),
            "param_count": self.param_count,
        }


@dataclass(frozen=True)
class _CompiledQuery:
    equation: str
    operand_names: tuple[str, ...]
    initializers: tuple[tuple[str, np.ndarray], ...]


def canonicalize(query: Query) -> Query:
    """Return the canonical form of a valid unordered typed query."""
    return _canonicalize_valid(_validate_query(query, require_all_anchors=True))


def _canonicalize_partial(query: Query) -> Query:
    """Canonicalize a search prefix without weakening the public API."""
    return _canonicalize_valid(_validate_query(query, require_all_anchors=False))


def _validate_query(query: Query, *, require_all_anchors: bool) -> Query:
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
    if require_all_anchors and ("k" not in colors or not {"r", "c"} <= spaces):
        raise ValueError("query operands must supply output labels k, r, and c")
    if len(internal_colors) > len(_COLOR_POOL) or len(internal_spaces) > len(
        _SPACE_POOL
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


def _compile_query(query: Query, correction: Correction | None) -> _CompiledQuery:
    canonical = canonicalize(query)
    correction = correction or Correction()
    color_labels = {atom[0] for atom in canonical}
    spatial_labels = {label for _, row, col in canonical for label in (row, col)}
    unknown_colors = {label for label, _ in correction.channel_masks} - color_labels
    unknown_spaces = {label for label, _ in correction.spatial_gates} - spatial_labels
    if unknown_colors or unknown_spaces:
        raise ValueError(
            f"correction labels absent from query: {sorted(unknown_colors | unknown_spaces)}"
        )
    if correction.param_count > 150:
        raise ValueError("correction exceeds the 150-parameter limit")

    source_label = "k"
    if correction.channel_mixer is not None:
        source_label = next(label for label in _COLOR_POOL if label not in color_labels)
        canonical = tuple(
            (source_label if color == "k" else color, row, col)
            for color, row, col in canonical
        )

    equations = ["b" + "".join(atom) for atom in canonical]
    operand_names = ["input"] * len(canonical)
    initializers: list[tuple[str, np.ndarray]] = []
    tensor_names: dict[tuple[tuple[int, ...], bytes], str] = {}

    def add_operand(label_equation: str, values: Sequence[float], stem: str) -> None:
        array = np.asarray(values, dtype=np.float32)
        key = (array.shape, array.tobytes())
        name = tensor_names.get(key)
        if name is None:
            name = f"{stem}_{len(tensor_names)}"
            tensor_names[key] = name
            initializers.append((name, array))
        equations.append(label_equation)
        operand_names.append(name)

    for label, values in correction.channel_masks:
        add_operand(source_label if label == "k" else label, values, "channel_mask")
    for label, values in correction.spatial_gates:
        add_operand(label, values, "spatial_gate")
    if correction.channel_mixer is not None:
        add_operand("k" + source_label, correction.channel_mixer, "channel_mixer")

    equation = ",".join(equations) + "->bkrc"
    if len(set(equation.replace(",", "").replace("-", "").replace(">", ""))) > 52:
        raise ValueError("equation uses more than 52 distinct labels")
    return _CompiledQuery(
        equation=equation,
        operand_names=tuple(operand_names),
        initializers=tuple(initializers),
    )


def evaluate_query(
    query: Query,
    input_onehot: np.ndarray,
    correction: Correction | None = None,
) -> np.ndarray:
    return _evaluate_query_values(query, input_onehot, correction) > 0


def _evaluate_query_values(
    query: Query,
    input_onehot: np.ndarray,
    correction: Correction | None = None,
) -> np.ndarray:
    compiled = _compile_query(query, correction)
    values = np.asarray(input_onehot, dtype=np.float32)
    initializer_values = dict(compiled.initializers)
    operands = [
        values if name == "input" else initializer_values[name]
        for name in compiled.operand_names
    ]
    return np.einsum(compiled.equation, *operands, optimize="greedy")


def _is_complete(query: Query) -> bool:
    colors = {color for color, _, _ in query}
    spaces = {label for _, row, col in query for label in (row, col)}
    return "k" in colors and {"r", "c"} <= spaces


def _evaluate_search_query(query: Query, input_onehot: np.ndarray) -> np.ndarray:
    """Score a partial prefix by filling missing output axes with FREE all-ones gates."""
    canonical = _canonicalize_partial(query)
    if _is_complete(canonical):
        return evaluate_query(canonical, input_onehot)
    values = np.asarray(input_onehot, dtype=np.float32)
    equations = ["b" + "".join(atom) for atom in canonical]
    operands: list[np.ndarray] = [values] * len(canonical)
    colors = {color for color, _, _ in canonical}
    spaces = {label for _, row, col in canonical for label in (row, col)}
    if "k" not in colors:
        equations.append("k")
        operands.append(np.ones(values.shape[1], dtype=np.float32))
    if "r" not in spaces:
        equations.append("r")
        operands.append(np.ones(values.shape[2], dtype=np.float32))
    if "c" not in spaces:
        equations.append("c")
        operands.append(np.ones(values.shape[3], dtype=np.float32))
    result = np.einsum(",".join(equations) + "->bkrc", *operands, optimize="greedy")
    return result > 0


def build_model(query: Query, correction: Correction | None = None) -> onnx.ModelProto:
    compiled = _compile_query(query, correction)
    node = helper.make_node(
        "Einsum",
        list(compiled.operand_names),
        ["output"],
        name="output",
        equation=compiled.equation,
    )

    def tensor(name: str) -> onnx.ValueInfoProto:
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, _GRID_SHAPE)

    graph = helper.make_graph(
        [node],
        "self_einsum",
        [tensor("input")],
        [tensor("output")],
        [numpy_helper.from_array(array, name) for name, array in compiled.initializers],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model


@lru_cache(maxsize=65_536)
def expand_query(query: Query) -> set[Query]:
    """Add every connected typed atom introducing at most one label per type."""
    canonical = _canonicalize_partial(query)
    existing_labels = {label for atom in canonical for label in atom}
    colors = sorted({atom[0] for atom in canonical}, key=_COLOR_RANK.__getitem__)
    spaces = sorted(
        {label for _, row, col in canonical for label in (row, col)},
        key=_SPACE_RANK.__getitem__,
    )
    new_color = next((label for label in _COLOR_POOL if label not in colors), None)
    new_space = next((label for label in _SPACE_POOL if label not in spaces), None)
    color_choices = tuple(
        sorted(
            {*colors, "k", *((new_color,) if new_color else ())},
            key=_COLOR_RANK.__getitem__,
        )
    )
    space_choices = tuple(
        sorted(
            {*spaces, "r", "c", *((new_space,) if new_space else ())},
            key=_SPACE_RANK.__getitem__,
        )
    )
    expanded: set[Query] = set()
    for color in color_choices:
        for row in space_choices:
            for col in space_choices:
                atom = (color, row, col)
                if atom in canonical or not (set(atom) & existing_labels):
                    continue
                if len({row, col} - set(spaces)) > 1:
                    continue
                expanded.add(_canonicalize_partial((*canonical, atom)))
    return expanded


@lru_cache(maxsize=1)
def _seed_queries() -> frozenset[Query]:
    """Every typed one-atom prefix with anchors or up to two internal spaces."""
    seeds = {
        _canonicalize_partial(((color, row, col),))
        for color in ("k", _COLOR_POOL[0])
        for row in ("r", "c", _SPACE_POOL[0], _SPACE_POOL[1])
        for col in ("r", "c", _SPACE_POOL[0], _SPACE_POOL[1])
    }
    return frozenset(seeds)


def grammar_reaches(query: Query) -> bool:
    """Prove reachability by finding an actual connected expansion ordering."""
    target = _validate_query(query, require_all_anchors=True)
    for ordering in permutations(target):
        prefix = _canonicalize_partial((ordering[0],))
        if prefix not in _seed_queries():
            continue
        reachable = True
        for depth in range(2, len(ordering) + 1):
            wanted = _canonicalize_partial(tuple(ordering[:depth]))
            if wanted not in expand_query(prefix):
                reachable = False
                break
            prefix = wanted
        if reachable and canonicalize(prefix) == canonicalize(target):
            return True
    return False


def _normalize_examples(
    examples: Iterable[tuple[np.ndarray, np.ndarray] | dict[str, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    normalized = []
    for example in examples:
        if isinstance(example, dict):
            raw_input, target = example["input"], example["output"]
        else:
            raw_input, target = example
        normalized.append(
            (np.asarray(raw_input, dtype=np.float32), np.asarray(target) > 0)
        )
    if not normalized:
        raise ValueError("at least one example is required")
    return normalized


def _loss_stats(
    query: Query,
    examples: list[tuple[np.ndarray, np.ndarray]],
    correction: Correction | None = None,
) -> tuple[int, int, int]:
    try:
        if correction is not None and not _is_complete(query):
            return sys.maxsize, 0, len(examples)

        def evaluate(raw_input: np.ndarray) -> np.ndarray:
            if correction is None:
                return _evaluate_search_query(query, raw_input)
            return evaluate_query(query, raw_input, correction)

        if all(
            raw_input.shape[0] == 1 and raw_input.shape[1:] == examples[0][0].shape[1:]
            for raw_input, _ in examples
        ):
            inputs = np.concatenate([raw_input for raw_input, _ in examples], axis=0)
            targets = np.concatenate([target for _, target in examples], axis=0)
            actual = evaluate(inputs)
            wrong_per_example = np.count_nonzero(actual != targets, axis=(1, 2, 3))
            failed = int(np.count_nonzero(wrong_per_example))
            return int(wrong_per_example.sum()), len(examples) - failed, failed
        wrong_per_example = []
        for raw_input, target in examples:
            actual = evaluate(raw_input)
            wrong_per_example.append(int(np.count_nonzero(actual != target)))
        failed = sum(value != 0 for value in wrong_per_example)
        return sum(wrong_per_example), len(examples) - failed, failed
    except (MemoryError, ValueError, RuntimeError):
        return sys.maxsize, 0, len(examples)


def query_loss(
    query: Query,
    examples: list[tuple[np.ndarray, np.ndarray]],
) -> int:
    return _loss_stats(canonicalize(query), _normalize_examples(examples))[0]


def _is_exact(
    query: Query,
    examples: list[tuple[np.ndarray, np.ndarray]],
    correction: Correction,
) -> bool:
    return _loss_stats(query, examples, correction)[0] == 0


def _fit_mixer(
    raw_outputs: list[np.ndarray], targets: list[np.ndarray]
) -> tuple[tuple[float, ...], ...] | None:
    x = np.concatenate(
        [np.moveaxis(raw, 1, -1).reshape(-1, 10) for raw in raw_outputs], axis=0
    ).astype(np.float32)
    y = np.concatenate(
        [np.moveaxis(target, 1, -1).reshape(-1, 10) for target in targets], axis=0
    ).astype(bool)
    rows = []
    for output_channel in range(10):
        choices: list[np.ndarray] = []
        for source_channel in range(10):
            unit = np.zeros(10, dtype=np.float32)
            unit[source_channel] = 1
            choices.append(unit)
            for penalty in (1.0, 10.0):
                signed = np.full(10, -penalty, dtype=np.float32)
                signed[source_channel] = 1
                choices.append(signed)
        exact = [
            weights
            for weights in choices
            if np.array_equal(x @ weights > 0, y[:, output_channel])
        ]
        if not exact:
            return None
        weights = min(
            exact,
            key=lambda row: (
                int(np.count_nonzero(row)),
                float(np.abs(row).sum()),
                tuple(float(value) for value in row),
            ),
        )
        rows.append(tuple(float(value) for value in weights))
    return tuple(rows)


def fit_small_corrections(
    query: Query,
    examples: list[tuple[np.ndarray, np.ndarray]],
    max_params: int = 150,
) -> list[Correction]:
    """Fit exact masks/gates/mixers as operands of the uncontracted query."""
    canonical = canonicalize(query)
    normalized = _normalize_examples(examples)
    targets = [target for _, target in normalized]
    accepted: list[Correction] = []
    seen: set[Correction] = set()

    def consider(correction: Correction) -> None:
        if (
            correction not in seen
            and correction.param_count <= max_params
            and _is_exact(canonical, normalized, correction)
        ):
            seen.add(correction)
            accepted.append(correction)

    raw_outputs = [evaluate_query(canonical, raw_input) for raw_input, _ in normalized]
    direct_mask = []
    direct_possible = True
    for channel in range(10):
        raw_planes = [raw[:, channel] for raw in raw_outputs]
        target_planes = [target[:, channel] for target in targets]
        if all(
            np.array_equal(raw, target)
            for raw, target in zip(raw_planes, target_planes, strict=True)
        ):
            direct_mask.append(1)
        elif all(not target.any() for target in target_planes):
            direct_mask.append(0)
        else:
            direct_possible = False
            break
    if direct_possible:
        consider(Correction(channel_masks=(("k", tuple(direct_mask)),)))

    color_labels = sorted({atom[0] for atom in canonical}, key=_COLOR_RANK.__getitem__)
    masks = [(0, 1, 1, 1, 1, 1, 1, 1, 1, 1)]
    masks.extend(
        tuple(int(index == selected) for index in range(10)) for selected in range(10)
    )
    label_groups: list[tuple[str, ...]] = [(label,) for label in color_labels]
    if len(color_labels) > 1:
        label_groups.append(tuple(color_labels))
        label_groups.extend(combinations(color_labels, 2))
    for labels in label_groups:
        for mask in masks:
            consider(Correction(channel_masks=tuple((label, mask) for label in labels)))

    spatial_labels = sorted(
        {label for _, row, col in canonical for label in (row, col)} - {"r", "c"},
        key=_SPACE_RANK.__getitem__,
    )
    gate_groups: list[tuple[str, ...]] = [(label,) for label in spatial_labels]
    gate_groups.extend(combinations(spatial_labels, 2))
    if len(spatial_labels) > 2:
        gate_groups.append(tuple(spatial_labels))

    bases = [Correction()]
    bases.extend(accepted)
    for labels in gate_groups:
        for selected in range(30):
            gate = tuple(int(index == selected) for index in range(30))
            base = Correction(spatial_gates=tuple((label, gate) for label in labels))
            if base.param_count <= max_params:
                bases.append(base)
                consider(base)

    for base in bases:
        if base.param_count + 100 > max_params:
            continue
        gated_outputs = [
            _evaluate_query_values(canonical, raw_input, base)
            for raw_input, _ in normalized
        ]
        mixer = _fit_mixer(gated_outputs, targets)
        if mixer is None:
            continue
        consider(
            Correction(
                channel_masks=base.channel_masks,
                spatial_gates=base.spatial_gates,
                channel_mixer=mixer,
            )
        )
    return sorted(accepted, key=lambda item: (item.param_count, repr(item)))


def select_diagnostic_examples(
    examples: list[tuple[np.ndarray, np.ndarray]],
    *,
    official_count: int,
    arc_limit: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Keep official cases and deterministic farthest-first generated cases."""
    official_count = min(max(0, official_count), len(examples))
    selected_indices = list(range(official_count))
    remaining = list(range(official_count, len(examples)))
    if arc_limit <= 0 or not remaining:
        return [examples[index] for index in selected_indices]

    def feature(example: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        raw_input, target = example
        input_bool = raw_input > 0
        target_bool = target > 0
        return np.asarray(
            [
                np.count_nonzero(input_bool),
                np.count_nonzero(target_bool),
                np.count_nonzero(input_bool.any(axis=(0, 2, 3))),
                np.count_nonzero(target_bool.any(axis=(0, 2, 3))),
                np.count_nonzero(input_bool.any(axis=(0, 1, 3))),
                np.count_nonzero(input_bool.any(axis=(0, 1, 2))),
                np.count_nonzero(target_bool.any(axis=(0, 1, 3))),
                np.count_nonzero(target_bool.any(axis=(0, 1, 2))),
            ],
            dtype=np.float64,
        )

    features = np.stack([feature(example) for example in examples])
    spans = np.ptp(features, axis=0)
    spans[spans == 0] = 1
    features = (features - features.min(axis=0)) / spans
    anchors = selected_indices[:] or [remaining[0]]
    if not selected_indices:
        selected_indices.append(remaining.pop(0))
    while remaining and len(selected_indices) < official_count + arc_limit:
        best = max(
            remaining,
            key=lambda index: (
                min(
                    float(np.abs(features[index] - features[anchor]).sum())
                    for anchor in anchors
                ),
                -index,
            ),
        )
        selected_indices.append(best)
        anchors.append(best)
        remaining.remove(best)
    return [examples[index] for index in selected_indices]


def _task_examples_with_official_count(
    task_num: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    task = load_task(task_num)
    official = []
    generated = []
    for split in ("train", "test"):
        for example in task.get(split, []):
            if (arrays := convert_to_numpy(example)) is not None:
                official.append((arrays["input"], arrays["output"]))
    for example in task.get("arc-gen", []):
        if (arrays := convert_to_numpy(example)) is not None:
            generated.append((arrays["input"], arrays["output"]))
    normalized = _normalize_examples([*official, *generated])
    return normalized, len(official)


def _task_examples(task_num: int) -> list[tuple[np.ndarray, np.ndarray]]:
    return _task_examples_with_official_count(task_num)[0]


@lru_cache(maxsize=1)
def _manifest() -> dict:
    path = Path(__file__).resolve().parents[1] / "state" / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _report_row(
    task_num: int,
    query: Query,
    stats: tuple[int, int, int],
    correction: Correction | None,
) -> dict:
    wrong_cells, passed_examples, failed_examples = stats
    cost = correction.param_count if correction else 0
    incumbent = _manifest().get(f"{task_num:03d}", {}).get("points", 0.0)
    return {
        "task": task_num,
        "atoms": len(query),
        "query": query,
        "equation": _compile_query(query, correction).equation,
        "pass": passed_examples,
        "fail": failed_examples,
        "wrong_cells": wrong_cells,
        "correction": correction.as_json() if correction else None,
        "cost": cost,
        "projected_gain": round(max(1.0, 25.0 - math.log(max(1, cost))) - incumbent, 6),
    }


def _runtime_validate(
    query: Query,
    correction: Correction | None,
    examples: list[tuple[np.ndarray, np.ndarray]],
    *,
    fresh: bool,
) -> bool:
    model = build_model(query, correction)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    for raw_input, target in examples:
        if not np.array_equal(session.run(None, {"input": raw_input})[0] > 0, target):
            return False
    if not fresh:
        return True

    with tempfile.TemporaryDirectory(prefix="self_einsum_fresh_") as temp_dir:
        model_path = Path(temp_dir) / "candidate.onnx"
        examples_path = Path(temp_dir) / "examples.npz"
        onnx.save(model, model_path)
        np.savez(
            examples_path,
            inputs=np.concatenate([item[0] for item in examples]),
            targets=np.concatenate([item[1] for item in examples]),
        )
        code = """
import sys
import numpy as np
import onnxruntime as ort
so=ort.SessionOptions(); so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
s=ort.InferenceSession(sys.argv[1], so, providers=['CPUExecutionProvider'])
d=np.load(sys.argv[2])
for raw,target in zip(d['inputs'],d['targets'],strict=True):
    actual=s.run(None,{'input':raw[None].astype(np.float32)})[0]>0
    if not np.array_equal(actual,target[None]): raise SystemExit(2)
"""
        result = subprocess.run(
            [sys.executable, "-c", code, str(model_path), str(examples_path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.returncode == 0


def search_task(
    task_num: int,
    max_atoms: int = 8,
    beam: int = 3000,
    correction_beam: int = 8,
    ranking_arc_limit: int = 3,
) -> list[dict]:
    """Beam-search connected typed self-Einsum queries for one task."""
    if not 1 <= task_num <= 400:
        raise ValueError("task number must be in 1..400")
    if max_atoms < 1 or beam < 1 or correction_beam < 0 or ranking_arc_limit < 0:
        raise ValueError("search limits must be positive")
    examples, official_count = _task_examples_with_official_count(task_num)
    ranking_examples = select_diagnostic_examples(
        examples, official_count=official_count, arc_limit=ranking_arc_limit
    )
    frontier = set(_seed_queries())
    rows: list[dict] = []
    emitted: set[tuple[Query, Correction | None]] = set()
    correction_pool: list[tuple[int, tuple, Query, tuple[int, int, int]]] = []
    found_raw_exact = False

    for depth in range(1, max_atoms + 1):
        ranked = []
        for query in sorted(frontier, key=_query_key):
            stats = _loss_stats(query, ranking_examples)
            if stats[0] != sys.maxsize:
                ranked.append((stats[0], _query_key(query), query, stats))
        ranked.sort(key=lambda item: (item[0], item[1]))
        if not ranked:
            break

        complete_ranked = [item for item in ranked if _is_complete(item[2])]
        correction_pool.extend(complete_ranked[:correction_beam])
        if complete_ranked and complete_ranked[0][0] != 0:
            best = complete_ranked[0]
            if (best[2], None) not in emitted:
                full_stats = _loss_stats(best[2], examples)
                rows.append(_report_row(task_num, best[2], full_stats, None))
                emitted.add((best[2], None))
        for _, _, query, stats in ranked:
            if stats[0] != 0 or not _is_complete(query):
                continue
            if (query, None) in emitted:
                continue
            full_stats = _loss_stats(query, examples)
            if full_stats[0] == 0 and _runtime_validate(
                query, None, examples, fresh=True
            ):
                rows.append(_report_row(task_num, query, full_stats, None))
                emitted.add((query, None))
                found_raw_exact = True

        retained_items = ranked[:beam]
        retained_queries = {item[2] for item in retained_items}
        retained_queries.update(item[2] for item in ranked if item[0] == 0)
        if found_raw_exact or depth == max_atoms:
            break
        next_frontier: set[Query] = set()
        for query in retained_queries:
            next_frontier.update(expand_query(query))
        frontier = next_frontier

    if not found_raw_exact and correction_beam:
        correction_pool.sort(key=lambda item: (item[0], item[1]))
        correction_queries = []
        seen_queries: set[Query] = set()
        for _, _, query, _ in correction_pool:
            if query not in seen_queries:
                correction_queries.append(query)
                seen_queries.add(query)
            if len(correction_queries) == correction_beam:
                break
        for query in correction_queries:
            for correction in fit_small_corrections(query, ranking_examples):
                key = (query, correction)
                if key in emitted:
                    continue
                stats = _loss_stats(query, examples, correction)
                if stats[0] == 0 and _runtime_validate(
                    query, correction, examples, fresh=False
                ):
                    rows.append(_report_row(task_num, query, stats, correction))
                    emitted.add(key)
    return rows


def _parse_tasks(tokens: Sequence[str]) -> list[int]:
    tasks: set[int] = set()
    for token in tokens:
        if "-" in token:
            start, end = (int(value) for value in token.split("-", 1))
            tasks.update(range(start, end + 1))
        else:
            tasks.add(int(token))
    if not tasks or min(tasks) < 1 or max(tasks) > 400:
        raise ValueError("tasks must be in 1..400")
    return sorted(tasks)


def _write_json_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--max-atoms", type=int, default=8)
    parser.add_argument("--beam", type=int, default=3000)
    parser.add_argument("--correction-beam", type=int, default=8)
    parser.add_argument("--ranking-arc-limit", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-index must be in [0, shard-count)")
    tasks = [
        task
        for index, task in enumerate(_parse_tasks(args.tasks))
        if index % args.shard_count == args.shard_index
    ]
    rows: list[dict] = []
    if args.output.exists() and not args.no_resume:
        rows = json.loads(args.output.read_text())
    completed = {int(row["task"]) for row in rows}
    pending = [task for task in tasks if task not in completed]
    started = time.monotonic()
    for index, task in enumerate(pending, start=1):
        task_started = time.monotonic()
        task_rows = search_task(
            task,
            max_atoms=args.max_atoms,
            beam=args.beam,
            correction_beam=args.correction_beam,
            ranking_arc_limit=args.ranking_arc_limit,
        )
        rows.extend(task_rows)
        rows.sort(
            key=lambda row: (
                row["task"],
                row["atoms"],
                row["wrong_cells"],
                row["equation"],
            )
        )
        _write_json_atomic(args.output, rows)
        elapsed = time.monotonic() - started
        rate = index / elapsed if elapsed else 0.0
        eta = (len(pending) - index) / rate if rate else None
        print(
            json.dumps(
                {
                    "task": task,
                    "seconds": round(time.monotonic() - task_started, 3),
                    "completed": index,
                    "remaining": len(pending) - index,
                    "tasks_per_second": round(rate, 6),
                    "eta_seconds": round(eta, 1) if eta is not None else None,
                    "max_atoms": args.max_atoms,
                    "beam": args.beam,
                    "correction_beam": args.correction_beam,
                    "ranking_arc_limit": args.ranking_arc_limit,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
