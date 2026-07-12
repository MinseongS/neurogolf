"""Sweep single-node spatial ops whose sole scalar initializer costs one."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper, numpy_helper

from candidates.attr_pool_sweep.pool_sweep import grid_to_tensor


@dataclass(frozen=True)
class OpConfig:
    op: str
    axis: int = 1
    reverse: int = 0
    exclusive: int = 0
    upper: int = 1
    k: int = 0


def apply_config(source: np.ndarray, config: OpConfig) -> np.ndarray:
    source = np.asarray(source, dtype=bool)
    if config.op == "cumsum":
        work = np.flip(source, axis=config.axis) if config.reverse else source
        summed = np.cumsum(work, axis=config.axis)
        if config.exclusive:
            shifted = np.zeros_like(summed)
            dst = [slice(None)] * source.ndim
            src = [slice(None)] * source.ndim
            dst[config.axis] = slice(1, None)
            src[config.axis] = slice(None, -1)
            shifted[tuple(dst)] = summed[tuple(src)]
            summed = shifted
        if config.reverse:
            summed = np.flip(summed, axis=config.axis)
        return summed > 0
    if config.op == "reverse_sequence":
        return np.flip(source, axis=config.axis)
    if config.op == "trilu":
        fn = np.triu if config.upper else np.tril
        return np.stack([fn(channel, k=config.k) for channel in source])
    raise ValueError(f"unknown op: {config.op}")


def build_model(config: OpConfig) -> onnx.ModelProto:
    if config.op == "cumsum":
        initializer = numpy_helper.from_array(np.asarray(config.axis + 1, np.int64), name="scalar")
        node = helper.make_node(
            "CumSum",
            ["input", "scalar"],
            ["output"],
            name="output",
            reverse=config.reverse,
            exclusive=config.exclusive,
        )
    elif config.op == "reverse_sequence":
        initializer = numpy_helper.from_array(np.asarray([30], np.int64), name="scalar")
        node = helper.make_node(
            "ReverseSequence",
            ["input", "scalar"],
            ["output"],
            name="output",
            batch_axis=0,
            time_axis=config.axis + 1,
        )
    elif config.op == "trilu":
        initializer = numpy_helper.from_array(np.asarray(config.k, np.int64), name="scalar")
        node = helper.make_node(
            "Trilu", ["input", "scalar"], ["output"], name="output", upper=config.upper
        )
    else:
        raise ValueError(f"unknown op: {config.op}")

    graph = helper.make_graph(
        [node],
        f"scalar_{config.op}",
        [helper.make_tensor_value_info("input", TP.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TP.FLOAT, [1, 10, 30, 30])],
        [initializer],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    return model


def configs() -> list[OpConfig]:
    out = [
        OpConfig("cumsum", axis=axis, reverse=reverse, exclusive=exclusive)
        for axis in (1, 2)
        for reverse in (0, 1)
        for exclusive in (0, 1)
    ]
    out.extend(OpConfig("reverse_sequence", axis=axis) for axis in (1, 2))
    out.extend(OpConfig("trilu", upper=upper, k=k) for upper in (0, 1) for k in range(-29, 30))
    return out


def _pairs(task: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs = []
    for example in task.get("train", []) + task.get("test", []) + task.get("arc-gen", []):
        source = grid_to_tensor(example["input"])
        target = grid_to_tensor(example["output"])
        if source is not None and target is not None:
            pairs.append((source, target))
    return pairs


def scan_board(root: Path, output: Path) -> list[dict]:
    from src.harness import evaluate

    manifest = json.loads((root / "state" / "manifest.json").read_text())
    all_configs = configs()
    hits = []
    for task_num in range(1, 401):
        task = json.loads((root / "data" / f"task{task_num:03d}.json").read_text())
        pairs = _pairs(task)
        if not pairs:
            continue
        for config in all_configs:
            if not np.array_equal(apply_config(pairs[0][0], config), pairs[0][1]):
                continue
            if not all(np.array_equal(apply_config(source, config), target) for source, target in pairs[1:]):
                continue
            model = build_model(config)
            result = evaluate(model, task)
            if not result["ok"] or result["fail"]:
                continue
            candidate_path = output.parent / f"scalar_task{task_num:03d}.onnx"
            onnx.save(model, candidate_path)
            current = manifest[f"{task_num:03d}"]["points"]
            hit = {
                "task": task_num,
                "config": asdict(config),
                "current_points": current,
                "candidate_points": result["points"],
                "gain": result["points"] - current,
                "memory": result["memory"],
                "params": result["params"],
                "candidate": str(candidate_path),
            }
            hits.append(hit)
            print("HIT", json.dumps(hit, sort_keys=True))
            break
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
        default=Path("candidates/attr_pool_sweep/scalar_report.json"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hits = scan_board(args.root, args.output)
    print(json.dumps({"total_hits": len(hits), "tasks": [hit["task"] for hit in hits]}))


if __name__ == "__main__":
    main()
