#!/usr/bin/env python3
"""Scan deployed graphs for task285's redundant target-sentinel fingerprint."""

from __future__ import annotations

import json
import pathlib
import re

import onnx


ROOT = pathlib.Path(__file__).resolve().parents[2]
NETS = ROOT / "submission/overfit_nets"


def scan_model(path: pathlib.Path) -> list[dict[str, object]]:
    model = onnx.load(path)
    match = re.search(r"task(\d{3})", str(path))
    if match is None:
        raise ValueError(f"task id missing from path: {path}")
    task = int(match.group(1))
    producers = {
        output: node for node in model.graph.node for output in node.output
    }
    hits: list[dict[str, object]] = []
    for scatter in model.graph.node:
        if scatter.op_type != "ScatterElements" or len(scatter.input) < 3:
            continue
        update = producers.get(scatter.input[2])
        if update is None or update.op_type != "Min":
            continue
        for cap_name in update.input:
            add = producers.get(cap_name)
            if add is None or add.op_type != "Add":
                continue
            for mul_name in add.input:
                mul = producers.get(mul_name)
                if mul is None or mul.op_type != "Mul":
                    continue
                for gather_name in mul.input:
                    gather = producers.get(gather_name)
                    if gather is not None and gather.op_type == "Gather":
                        hits.append(
                            {
                                "task": task,
                                "scatter": scatter.output[0],
                                "update": update.output[0],
                                "target_gather": gather.output[0],
                            }
                        )
    return hits


def main() -> None:
    paths = sorted(NETS.glob("task*.onnx"))
    candidates: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for path in paths:
        try:
            candidates.extend(scan_model(path))
        except Exception as error:
            errors.append({"path": str(path), "error": str(error)})
    print(
        json.dumps(
            {"scanned": len(paths), "errors": errors, "candidates": candidates},
            indent=2,
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
