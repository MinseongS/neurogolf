"""Projection-filtered exhaustive search for attribute-only MaxPool graphs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper


@dataclass(frozen=True, order=True)
class AxisConfig:
    kernel: int
    dilation: int
    pad_before: int

    @property
    def effective_kernel(self) -> int:
        return (self.kernel - 1) * self.dilation + 1

    @property
    def pad_after(self) -> int:
        return self.effective_kernel - 1 - self.pad_before


def axis_configs(size: int = 30) -> list[AxisConfig]:
    configs = [AxisConfig(1, 1, 0)]
    for kernel in range(2, size + 1):
        max_dilation = (size - 1) // (kernel - 1)
        for dilation in range(1, max_dilation + 1):
            effective = (kernel - 1) * dilation + 1
            configs.extend(
                AxisConfig(kernel, dilation, pad_before)
                for pad_before in range(effective)
            )
    return configs


def pool1d(source: np.ndarray, config: AxisConfig) -> np.ndarray:
    source = np.asarray(source, dtype=bool)
    size = source.shape[-1]
    output = np.zeros_like(source)
    for tap in range(config.kernel):
        shift = config.pad_before - tap * config.dilation
        if shift >= 0:
            if shift < size:
                output[..., shift:] |= source[..., : size - shift]
        elif -shift < size:
            output[..., : size + shift] |= source[..., -shift:]
    return output


def pool2d(source: np.ndarray, row: AxisConfig, col: AxisConfig) -> np.ndarray:
    source = np.asarray(source, dtype=bool)
    vertically_pooled = pool1d(source.swapaxes(-2, -1), row).swapaxes(-2, -1)
    return pool1d(vertically_pooled, col)


def grid_to_tensor(grid: list[list[int]]) -> np.ndarray | None:
    if not grid or not grid[0] or len(grid) > 30 or len(grid[0]) > 30:
        return None
    tensor = np.zeros((10, 30, 30), dtype=bool)
    for row, values in enumerate(grid):
        for col, color in enumerate(values):
            tensor[int(color), row, col] = True
    return tensor


def build_model(row: AxisConfig, col: AxisConfig) -> onnx.ModelProto:
    node = helper.make_node(
        "MaxPool",
        ["input"],
        ["output"],
        name="output",
        kernel_shape=[row.kernel, col.kernel],
        dilations=[row.dilation, col.dilation],
        pads=[row.pad_before, col.pad_before, row.pad_after, col.pad_after],
        strides=[1, 1],
    )
    graph = helper.make_graph(
        [node],
        "attribute_only_pool",
        [helper.make_tensor_value_info("input", TP.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TP.FLOAT, [1, 10, 30, 30])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    return model


def _vector_bits(vector: np.ndarray) -> tuple[int, ...]:
    vector = np.asarray(vector, dtype=bool)
    return tuple(
        sum(int(active) << index for index, active in enumerate(channel))
        for channel in vector
    )


def _pool_bits(bits: int, size: int, config: AxisConfig) -> int:
    mask = (1 << size) - 1
    output = 0
    for tap in range(config.kernel):
        shift = config.pad_before - tap * config.dilation
        if shift >= 0:
            output |= bits << shift
        else:
            output |= bits >> -shift
    return output & mask


def _matching_axis_configs(
    pairs: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    projection_axis: int,
    size: int,
) -> list[AxisConfig]:
    signatures = []
    for source, target in pairs:
        source_projection = source.any(axis=projection_axis)
        target_projection = target.any(axis=projection_axis)
        signatures.append((_vector_bits(source_projection), _vector_bits(target_projection)))

    matches = []
    for config in axis_configs(size):
        if all(
            all(
                _pool_bits(source_bits, size, config) == target_bits
                for source_bits, target_bits in zip(source_channels, target_channels)
            )
            for source_channels, target_channels in signatures
        ):
            matches.append(config)
    return matches


def find_pool_configs(
    examples: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    max_hits: int = 32,
) -> list[tuple[AxisConfig, AxisConfig]]:
    pairs = [(np.asarray(source, dtype=bool), np.asarray(target, dtype=bool)) for source, target in examples]
    if not pairs:
        return []
    if any(source.shape != target.shape or source.ndim != 3 for source, target in pairs):
        return []
    height, width = pairs[0][0].shape[-2:]
    if any(source.shape[-2:] != (height, width) for source, _ in pairs):
        return []

    row_configs = _matching_axis_configs(
        pairs, projection_axis=-1, size=height
    )
    col_configs = _matching_axis_configs(
        pairs, projection_axis=-2, size=width
    )

    hits = []
    for row, col in product(row_configs, col_configs):
        if all(np.array_equal(pool2d(source, row, col), target) for source, target in pairs):
            hits.append((row, col))
            if len(hits) >= max_hits:
                break
    return hits


def _task_pairs(task: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    examples = task.get("train", []) + task.get("test", []) + task.get("arc-gen", [])
    pairs = []
    for example in examples:
        source = grid_to_tensor(example["input"])
        target = grid_to_tensor(example["output"])
        if source is not None and target is not None:
            pairs.append((source, target))
    return pairs


def _config_dict(config: AxisConfig) -> dict[str, int]:
    return {
        "kernel": config.kernel,
        "dilation": config.dilation,
        "pad_before": config.pad_before,
        "pad_after": config.pad_after,
    }


def scan_board(root: Path, output: Path) -> list[dict]:
    from src.harness import evaluate

    manifest = json.loads((root / "state" / "manifest.json").read_text())
    hits = []
    for task_num in range(1, 401):
        task_path = root / "data" / f"task{task_num:03d}.json"
        task = json.loads(task_path.read_text())
        configs = find_pool_configs(_task_pairs(task))
        if configs:
            verified = None
            for row, col in configs:
                model = build_model(row, col)
                result = evaluate(model, task)
                if result["ok"] and result["fail"] == 0:
                    verified = (row, col, result)
                    break
            if verified is not None:
                row, col, result = verified
                key = f"{task_num:03d}"
                current = manifest[key]["points"]
                candidate_path = output.parent / f"task{task_num:03d}.onnx"
                onnx.save(build_model(row, col), candidate_path)
                hit = {
                    "task": task_num,
                    "row": _config_dict(row),
                    "col": _config_dict(col),
                    "current_points": current,
                    "candidate_points": result["points"],
                    "gain": result["points"] - current,
                    "memory": result["memory"],
                    "params": result["params"],
                    "candidate": str(candidate_path),
                }
                hits.append(hit)
                print("HIT", json.dumps(hit, sort_keys=True))
        if task_num % 50 == 0:
            print(f"scanned {task_num}/400 hits={len(hits)}", flush=True)

    output.write_text(json.dumps({"hits": hits}, indent=2, sort_keys=True) + "\n")
    return hits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/Users/minseong/project/neurogolf"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("candidates/attr_pool_sweep/report.json"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hits = scan_board(args.root, args.output)
    print(json.dumps({"total_hits": len(hits), "tasks": [hit["task"] for hit in hits]}))


if __name__ == "__main__":
    main()
