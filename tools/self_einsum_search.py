from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
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


def _make_model(query: Query, correction: Correction | None = None) -> onnx.ModelProto:
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
    return model


def build_model(query: Query, correction: Correction | None = None) -> onnx.ModelProto:
    model = _make_model(query, correction)
    onnx.checker.check_model(model, full_check=True)
    return model


@lru_cache(maxsize=8_192)
def expand_query(query: Query) -> frozenset[Query]:
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
    return frozenset(expanded)


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
        expected = y[:, output_channel]
        choices: list[np.ndarray] = [np.zeros(10, dtype=np.float32)]
        for source_channel in range(10):
            unit = np.zeros(10, dtype=np.float32)
            unit[source_channel] = 1
            choices.append(unit)
            for penalty in (1.0, 10.0):
                signed = np.full(10, -penalty, dtype=np.float32)
                signed[source_channel] = 1
                choices.append(signed)

        # Least squares and a deterministic homogeneous perceptron cover general
        # signed separators, while the explicit rows above preserve exact zero and
        # permutation solutions without numerical fitting. Fit on a bounded,
        # deterministic slice and always validate the result on the full corpus.
        if x.size:
            if len(x) > 20_000:
                indices = np.linspace(0, len(x) - 1, 20_000, dtype=np.int64)
                train_x = x[indices]
                train_y = expected[indices]
            else:
                train_x = x
                train_y = expected
            least_squares, *_ = np.linalg.lstsq(
                train_x.astype(np.float64), train_y.astype(np.float64), rcond=None
            )
            choices.append(least_squares.astype(np.float32))

            weights = np.zeros(10, dtype=np.float32)
            for _ in range(200):
                scores = train_x @ weights
                mistakes = np.flatnonzero(
                    (train_y & (scores <= 0)) | (~train_y & (scores > 0))
                )
                if not len(mistakes):
                    break
                for index in mistakes:
                    weights += train_x[index] if train_y[index] else -train_x[index]
            choices.append(weights)

        exact = [
            weights
            for weights in choices
            if np.array_equal(x @ weights > 0, expected)
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


def _nonempty_subsets(labels: Sequence[str]) -> list[tuple[str, ...]]:
    return [
        subset
        for size in range(1, len(labels) + 1)
        for subset in combinations(labels, size)
    ]


def _merge_corrections(left: Correction, right: Correction) -> Correction | None:
    left_channels = {label for label, _ in left.channel_masks}
    right_channels = {label for label, _ in right.channel_masks}
    left_spaces = {label for label, _ in left.spatial_gates}
    right_spaces = {label for label, _ in right.spatial_gates}
    if left_channels & right_channels or left_spaces & right_spaces:
        return None
    if left.channel_mixer is not None and right.channel_mixer is not None:
        return None
    return Correction(
        channel_masks=tuple(
            sorted(
                (*left.channel_masks, *right.channel_masks),
                key=lambda item: _COLOR_RANK[item[0]],
            )
        ),
        spatial_gates=tuple(
            sorted(
                (*left.spatial_gates, *right.spatial_gates),
                key=lambda item: _SPACE_RANK[item[0]],
            )
        ),
        channel_mixer=left.channel_mixer or right.channel_mixer,
    )


def fit_small_corrections(
    query: Query,
    examples: list[tuple[np.ndarray, np.ndarray]],
    max_params: int = 150,
) -> list[Correction]:
    """Fit bounded combinations of masks, separable gates, and signed mixers.

    Every ranking and acceptance check uses every supplied example. Channel
    masks are data-derived arbitrary keep/drop vectors applied to every
    non-empty label subset. Spatial primitives cover arbitrary target-support
    gates plus every one-hot/complement gate, and the bounded beam combines
    distinct gates, masks, and mixers without exceeding ``max_params``.
    """
    canonical = canonicalize(query)
    normalized = _normalize_examples(examples)
    targets = [target for _, target in normalized]
    accepted: list[Correction] = []
    loss_cache: dict[Correction, int] = {}

    def loss(correction: Correction) -> int:
        if correction.param_count > max_params:
            return sys.maxsize
        if correction not in loss_cache:
            loss_cache[correction] = _loss_stats(
                canonical, normalized, correction
            )[0]
            if loss_cache[correction] == 0:
                accepted.append(correction)
        return loss_cache[correction]

    def rank(
        corrections: Iterable[Correction], limit: int
    ) -> list[Correction]:
        unique = {
            correction
            for correction in corrections
            if correction.param_count <= max_params
        }
        return sorted(
            unique,
            key=lambda correction: (
                loss(correction),
                correction.param_count,
                repr(correction),
            ),
        )[:limit]

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
        direct_mask_value = tuple(direct_mask)
    else:
        direct_mask_value = None

    color_labels = sorted({atom[0] for atom in canonical}, key=_COLOR_RANK.__getitem__)
    channel_groups = _nonempty_subsets(color_labels)
    target_support = tuple(
        int(any(target[:, channel].any() for target in targets))
        for channel in range(10)
    )
    masks = {
        (0,) * 10,
        (1,) * 10,
        target_support,
        (0, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        *(tuple(int(index == selected) for index in range(10)) for selected in range(10)),
        *(
            tuple(int(index != selected) for index in range(10))
            for selected in range(10)
        ),
    }
    if direct_mask_value is not None:
        masks.add(direct_mask_value)

    mask_pool: set[Correction] = {
        Correction(channel_masks=tuple((label, mask) for label in labels))
        for labels in channel_groups
        for mask in masks
    }

    # Coordinate descent can synthesize arbitrary keep/drop patterns beyond the
    # seeded masks, without multiplying the expensive contraction by all 2^10
    # masks at every label subset.
    for labels in channel_groups:
        for start in ((0,) * 10, (1,) * 10, target_support):
            current = Correction(
                channel_masks=tuple((label, start) for label in labels)
            )
            current_loss = loss(current)
            mask_pool.add(current)
            for _ in range(10):
                values = current.channel_masks[0][1]
                neighbors = []
                for index in range(10):
                    toggled = tuple(
                        1 - value if offset == index else value
                        for offset, value in enumerate(values)
                    )
                    neighbors.append(
                        Correction(
                            channel_masks=tuple(
                                (label, toggled) for label in labels
                            )
                        )
                    )
                best = min(
                    neighbors,
                    key=lambda correction: (loss(correction), repr(correction)),
                )
                best_loss = loss(best)
                if best_loss >= current_loss:
                    break
                mask_pool.add(best)
                current, current_loss = best, best_loss
    mask_ranked = rank(mask_pool, 48)
    mask_structured = {
        correction
        for labels in channel_groups
        for correction in rank(
            (
                candidate
                for candidate in mask_pool
                if tuple(label for label, _ in candidate.channel_masks) == labels
            ),
            2,
        )
    }
    protected_masks = {
        Correction(channel_masks=tuple((label, mask) for label in labels))
        for labels in channel_groups
        for mask in (target_support, direct_mask_value)
        if mask is not None
    }
    mask_ranked = rank([*mask_ranked, *mask_structured], 96)
    mask_ranked = list(
        dict.fromkeys([*sorted(protected_masks, key=repr), *mask_ranked])
    )

    spatial_labels = sorted(
        {label for _, row, col in canonical for label in (row, col)},
        key=_SPACE_RANK.__getitem__,
    )
    spatial_groups = _nonempty_subsets(spatial_labels)
    row_support = tuple(
        int(
            any(
                index < target.shape[2] and target[:, :, index, :].any()
                for target in targets
            )
        )
        for index in range(30)
    )
    column_support = tuple(
        int(
            any(
                index < target.shape[3] and target[:, :, :, index].any()
                for target in targets
            )
        )
        for index in range(30)
    )
    one_hot_gates = {
        tuple(int(index == selected) for index in range(30))
        for selected in range(30)
    }
    gates = {
        (0,) * 30,
        (1,) * 30,
        row_support,
        column_support,
        *one_hot_gates,
        *(
            tuple(int(index != selected) for index in range(30))
            for selected in range(30)
        ),
    }
    gate_pool: set[Correction] = {
        Correction(spatial_gates=tuple((label, gate) for label in labels))
        for labels in spatial_groups
        for gate in gates
        if 30 <= max_params
    }

    # Add bounded distinct-gate combinations. Keep the best six primitives for
    # each label, then combine disjoint labels; shared-gate subsets above remain
    # available at their cheaper deduplicated cost.
    best_single_gates: dict[str, list[Correction]] = {}
    for label in spatial_labels:
        best_single_gates[label] = rank(
            (
                Correction(spatial_gates=((label, gate),))
                for gate in one_hot_gates
            ),
            6,
        )
    for left_label, right_label in combinations(spatial_labels, 2):
        for left in best_single_gates[left_label]:
            for right in best_single_gates[right_label]:
                merged = _merge_corrections(left, right)
                if merged is not None and merged.param_count <= max_params:
                    gate_pool.add(merged)
    gate_ranked = rank(gate_pool, 64)
    gate_label_sets = {
        tuple(label for label, _ in correction.spatial_gates)
        for correction in gate_pool
    }
    gate_structured = {
        correction
        for labels in gate_label_sets
        for correction in rank(
            (
                candidate
                for candidate in gate_pool
                if tuple(label for label, _ in candidate.spatial_gates) == labels
            ),
            3,
        )
    }
    gate_ranked = rank([*gate_ranked, *gate_structured], 128)

    combined_pool = []
    for channel_base in mask_ranked:
        for spatial_base in gate_ranked:
            merged = _merge_corrections(channel_base, spatial_base)
            if merged is not None and merged.param_count <= max_params:
                combined_pool.append(merged)
    combined_ranked = rank(combined_pool, 96)

    bases = rank(
        [Correction(), *mask_ranked, *gate_ranked, *combined_ranked], 128
    )
    for base in bases:
        if base.param_count + 100 > max_params:
            continue
        try:
            gated_outputs = [
                _evaluate_query_values(canonical, raw_input, base)
                for raw_input, _ in normalized
            ]
        except (MemoryError, ValueError, RuntimeError):
            continue
        mixer = _fit_mixer(gated_outputs, targets)
        if mixer is None:
            continue
        loss(
            Correction(
                channel_masks=base.channel_masks,
                spatial_gates=base.spatial_gates,
                channel_mixer=mixer,
            )
        )
    return sorted(set(accepted), key=lambda item: (item.param_count, repr(item)))


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
    correction_json = correction.as_json() if correction else None
    if correction_json is not None:
        correction_json["compiled_equation"] = _compile_query(
            query, correction
        ).equation
    return {
        "task": task_num,
        "atoms": len(query),
        "query": query,
        "equation": to_equation(query),
        "pass": passed_examples,
        "fail": failed_examples,
        "wrong_cells": wrong_cells,
        "correction": correction_json,
        "cost": cost,
        "projected_gain": round(max(1.0, 25.0 - math.log(max(1, cost))) - incumbent, 6),
    }


def _runtime_validate(
    query: Query,
    correction: Correction | None,
    examples: list[tuple[np.ndarray, np.ndarray]],
    *,
    timeout_seconds: float,
) -> bool:
    if timeout_seconds <= 0:
        raise ValueError("runtime timeout must be positive")
    # Checker, session construction, and every inference live in the bounded
    # child. The parent deliberately uses the unchecked constructor so a bad or
    # pathological candidate cannot hang the whole search before the timeout.
    try:
        model = _make_model(query, correction)
    except (MemoryError, ValueError, RuntimeError, onnx.checker.ValidationError):
        return False
    with tempfile.TemporaryDirectory(prefix="self_einsum_runtime_") as temp_dir:
        model_path = Path(temp_dir) / "candidate.onnx"
        examples_path = Path(temp_dir) / "examples.npz"
        onnx.save(model, model_path)
        payload = {"count": np.asarray(len(examples), dtype=np.int64)}
        for index, (raw_input, target) in enumerate(examples):
            payload[f"input_{index}"] = np.asarray(raw_input, dtype=np.float32)
            payload[f"target_{index}"] = np.asarray(target, dtype=bool)
        np.savez(examples_path, **payload)
        code = """
import sys
import numpy as np
import onnx
import onnxruntime as ort
model=onnx.load(sys.argv[1]); onnx.checker.check_model(model,full_check=True)
so=ort.SessionOptions(); so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
so.intra_op_num_threads=1; so.inter_op_num_threads=1
s=ort.InferenceSession(sys.argv[1], so, providers=['CPUExecutionProvider'])
d=np.load(sys.argv[2])
for i in range(int(d['count'])):
    actual=s.run(None,{'input':d[f'input_{i}'].astype(np.float32)})[0]>0
    if not np.array_equal(actual,d[f'target_{i}']): raise SystemExit(2)
"""
        try:
            result = subprocess.run(
                [sys.executable, "-c", code, str(model_path), str(examples_path)],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0


def _rank_frontier(
    frontier: Iterable[Query],
    examples: list[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[int, tuple, Query, tuple[int, int, int]]]:
    """Query-major ranking hook; callers must pass the full bundled corpus."""
    ranked = []
    for query in sorted(frontier, key=_query_key):
        stats = _loss_stats(query, examples)
        if stats[0] != sys.maxsize:
            ranked.append((stats[0], _query_key(query), query, stats))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return ranked


def search_task(
    task_num: int,
    max_atoms: int = 8,
    beam: int = 3000,
    correction_beam: int = 8,
    ranking_arc_limit: int = 3,
    runtime_timeout_seconds: float = 30.0,
) -> list[dict]:
    """Beam-search one task, ranking every query on every bundled example.

    ``ranking_arc_limit`` remains as a compatibility-only CLI argument; no
    diagnostic subset is used. Query-major multi-task drivers can reuse
    :func:`_rank_frontier` while maintaining their own per-task beams.
    """
    if not 1 <= task_num <= 400:
        raise ValueError("task number must be in 1..400")
    if (
        max_atoms < 1
        or beam < 1
        or correction_beam < 0
        or ranking_arc_limit < 0
        or runtime_timeout_seconds <= 0
    ):
        raise ValueError("search limits must be positive")
    examples, _ = _task_examples_with_official_count(task_num)
    frontier = set(_seed_queries())
    rows: list[dict] = []
    emitted: set[tuple[Query, Correction | None]] = set()
    correction_pool: list[tuple[int, tuple, Query, tuple[int, int, int]]] = []

    for depth in range(1, max_atoms + 1):
        ranked = _rank_frontier(frontier, examples)
        if not ranked:
            break

        complete_ranked = [item for item in ranked if _is_complete(item[2])]
        near_hits = [item for item in complete_ranked if item[0] != 0]
        correction_pool.extend(near_hits[:correction_beam])
        if near_hits:
            best = near_hits[0]
            if (best[2], None) not in emitted:
                rows.append(_report_row(task_num, best[2], best[3], None))
                emitted.add((best[2], None))
        for _, _, query, stats in ranked:
            if stats[0] != 0 or not _is_complete(query):
                continue
            if (query, None) in emitted:
                continue
            if stats[0] == 0 and _runtime_validate(
                query,
                None,
                examples,
                timeout_seconds=runtime_timeout_seconds,
            ):
                rows.append(_report_row(task_num, query, stats, None))
                emitted.add((query, None))

        retained_items = ranked[:beam]
        retained_queries = {item[2] for item in retained_items}
        retained_queries.update(item[2] for item in ranked if item[0] == 0)
        if depth == max_atoms:
            break
        next_frontier: set[Query] = set()
        for query in retained_queries:
            next_frontier.update(expand_query(query))
        frontier = next_frontier

    if correction_beam:
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
            for correction in fit_small_corrections(query, examples):
                key = (query, correction)
                if key in emitted:
                    continue
                stats = _loss_stats(query, examples, correction)
                if stats[0] == 0 and _runtime_validate(
                    query,
                    correction,
                    examples,
                    timeout_seconds=runtime_timeout_seconds,
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


def _write_json_atomic(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


_OUTPUT_SCHEMA_VERSION = 1


def _valid_report_row(row: object, completed_tasks: list[int]) -> bool:
    """Validate persisted rows before trusting them during resume."""
    if not isinstance(row, dict):
        return False
    required = {
        "task",
        "atoms",
        "query",
        "equation",
        "pass",
        "fail",
        "wrong_cells",
        "correction",
        "cost",
        "projected_gain",
    }
    if set(row) != required:
        return False
    integer_fields = ("task", "atoms", "pass", "fail", "wrong_cells", "cost")
    if any(type(row[field]) is not int or row[field] < 0 for field in integer_fields):
        return False
    # Mirror the grammar's own invariants: a query is non-empty and every atom is
    # a (colour, row, col) triple. A row that passes here is trusted verbatim and
    # its task is never re-searched.
    if not isinstance(row["query"], list) or not row["query"]:
        return False
    if not all(
        isinstance(atom, list)
        and len(atom) == 3
        and all(isinstance(label, str) and len(label) == 1 for label in atom)
        for atom in row["query"]
    ):
        return False
    if row["task"] not in completed_tasks or row["atoms"] != len(row["query"]):
        return False
    if not isinstance(row["equation"], str):
        return False
    if row["correction"] is not None and not isinstance(row["correction"], dict):
        return False
    return type(row["projected_gain"]) in (int, float)


def _config_hash(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--max-atoms", type=int, default=8)
    parser.add_argument("--beam", type=int, default=3000)
    parser.add_argument("--correction-beam", type=int, default=8)
    parser.add_argument("--ranking-arc-limit", type=int, default=3)
    parser.add_argument("--runtime-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-index must be in [0, shard-count)")
    requested_tasks = _parse_tasks(args.tasks)
    tasks = [
        task
        for index, task in enumerate(requested_tasks)
        if index % args.shard_count == args.shard_index
    ]
    config = {
        "tasks": requested_tasks,
        "max_atoms": args.max_atoms,
        "beam": args.beam,
        "correction_beam": args.correction_beam,
        "ranking_arc_limit_ignored": args.ranking_arc_limit,
        "ranking_semantics": "all_bundled",
        "runtime_timeout_seconds": args.runtime_timeout_seconds,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
    }
    expected_hash = _config_hash(config)
    document = {
        "schema_version": _OUTPUT_SCHEMA_VERSION,
        "config": config,
        "config_hash": expected_hash,
        "completed_tasks": [],
        "rows": [],
    }
    if args.output.exists() and not args.no_resume:
        loaded = json.loads(args.output.read_text())
        completed_tasks = loaded.get("completed_tasks") if isinstance(loaded, dict) else None
        loaded_rows = loaded.get("rows") if isinstance(loaded, dict) else None
        if (
            not isinstance(loaded, dict)
            or loaded.get("schema_version") != _OUTPUT_SCHEMA_VERSION
            or loaded.get("config") != config
            or loaded.get("config_hash") != expected_hash
            or not isinstance(completed_tasks, list)
            or not all(isinstance(task, int) for task in completed_tasks)
            or completed_tasks != sorted(set(completed_tasks))
            or not isinstance(loaded_rows, list)
            or not all(_valid_report_row(row, completed_tasks) for row in loaded_rows)
        ):
            raise ValueError(
                "incompatible resume file: schema/config/shard differs; "
                "use --no-resume or a new output path"
            )
        document = loaded
    completed = {int(task) for task in document["completed_tasks"]}
    if not completed <= set(tasks):
        raise ValueError("incompatible resume file: completed task outside shard")
    pending = [task for task in tasks if task not in completed]
    _write_json_atomic(args.output, document)
    started = time.monotonic()
    for index, task in enumerate(pending, start=1):
        task_started = time.monotonic()
        task_rows = search_task(
            task,
            max_atoms=args.max_atoms,
            beam=args.beam,
            correction_beam=args.correction_beam,
            ranking_arc_limit=args.ranking_arc_limit,
            runtime_timeout_seconds=args.runtime_timeout_seconds,
        )
        document["rows"].extend(task_rows)
        document["rows"].sort(
            key=lambda row: (
                row["task"],
                row["atoms"],
                row["wrong_cells"],
                row["equation"],
            )
        )
        document["completed_tasks"].append(task)
        document["completed_tasks"] = sorted(set(document["completed_tasks"]))
        _write_json_atomic(args.output, document)
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
                    "ranking_semantics": "all_bundled",
                    "runtime_timeout_seconds": args.runtime_timeout_seconds,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
