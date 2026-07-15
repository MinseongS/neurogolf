#!/usr/bin/env python3
"""Run the bounded task285 candidate once in an isolated clean process."""

from __future__ import annotations

import json
import multiprocessing as mp
import pathlib
import time

import numpy as np
import onnxruntime as ort


ROOT = pathlib.Path(__file__).resolve().parents[3]
MODEL = ROOT / "candidates/task285/bounded_free_output_renderer.onnx"
DATA = ROOT / "data/task285.json"
TIMEOUT_SECONDS = 10.0
MAX_FIRST_INFERENCE_SECONDS = 1.0


def first_input() -> np.ndarray:
    grid = json.loads(DATA.read_text())["train"][0]["input"]
    value = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for row, line in enumerate(grid):
        for col, colour in enumerate(line):
            value[0, colour, row, col] = 1.0
    return value


def run_one(queue: mp.Queue) -> None:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    load_start = time.perf_counter()
    session = ort.InferenceSession(
        str(MODEL), options, providers=["CPUExecutionProvider"]
    )
    load_seconds = time.perf_counter() - load_start
    inference_start = time.perf_counter()
    output = session.run(["output"], {"input": first_input()})[0]
    first_inference_seconds = time.perf_counter() - inference_start
    queue.put(
        {
            "status": "output",
            "load_seconds": load_seconds,
            "first_inference_seconds": first_inference_seconds,
            "shape": list(output.shape),
            "positive_count": int(np.count_nonzero(output > 0.0)),
        }
    )


def isolated_run(timeout: float = TIMEOUT_SECONDS) -> dict[str, object]:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=run_one, args=(queue,))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(2.0)
        return {
            "status": "timeout",
            "timeout_seconds": timeout,
            "exitcode": process.exitcode,
        }
    if not queue.empty():
        return {"exitcode": process.exitcode, **queue.get()}
    return {"status": "no-output", "exitcode": process.exitcode}


def main() -> None:
    report = isolated_run()
    print(json.dumps(report, indent=2, sort_keys=True))
    if report.get("status") != "output":
        raise SystemExit(1)
    if report.get("shape") != [1, 10, 30, 30]:
        raise SystemExit(2)
    if float(report["first_inference_seconds"]) >= MAX_FIRST_INFERENCE_SECONDS:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
