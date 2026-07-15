"""Fresh raw/sign/off-grid A/B verifier for task124 composite stages."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import random
import sys

import numpy as np
import onnx
import onnxruntime as ort

from neurogolf.scoring import convert_to_numpy

from build_cost686_composite import STAGES, build


ROOT = Path(__file__).resolve().parents[2]
INCUMBENT = ROOT / "submission/overfit_nets/task124.onnx"


def _session(model_or_path: onnx.ModelProto | Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    payload = (
        model_or_path.SerializeToString()
        if isinstance(model_or_path, onnx.ModelProto)
        else model_or_path.read_bytes()
    )
    return ort.InferenceSession(
        payload, options, providers=["CPUExecutionProvider"]
    )


def verify(*, stage: str, target: int, seed: int) -> dict[str, int]:
    random.seed(seed)
    np.random.seed(seed)
    arcgen = str(ROOT / "arc-gen")
    if arcgen not in sys.path:
        sys.path.insert(0, arcgen)
    generator = importlib.import_module("tasks.task_53b68214")

    incumbent = _session(INCUMBENT)
    candidate = _session(build(stage=stage))
    counters = {
        "runs": 0,
        "incumbent_fail": 0,
        "candidate_fail": 0,
        "raw_divergence": 0,
        "sign_divergence": 0,
        "off_grid_positives": 0,
    }

    while counters["runs"] < target:
        try:
            example = generator.generate()
            arrays = convert_to_numpy(example)
        except Exception:
            continue
        if arrays is None:
            continue

        expected = arrays["output"] > 0
        old_raw = incumbent.run(
            ["output"], {"input": arrays["input"]}
        )[0]
        candidate_raw = candidate.run(
            ["output"], {"input": arrays["input"]}
        )[0]
        old_sign = old_raw > 0
        candidate_sign = candidate_raw > 0

        counters["runs"] += 1
        counters["incumbent_fail"] += int(
            not np.array_equal(old_sign, expected)
        )
        counters["candidate_fail"] += int(
            not np.array_equal(candidate_sign, expected)
        )
        counters["raw_divergence"] += int(
            not np.array_equal(candidate_raw, old_raw)
        )
        counters["sign_divergence"] += int(
            not np.array_equal(candidate_sign, old_sign)
        )
        off_grid = candidate_raw.copy()
        off_grid[..., :10, :10] = 0
        counters["off_grid_positives"] += int(
            np.count_nonzero(off_grid > 0)
        )

        if counters["runs"] % 250 == 0:
            print(
                " ".join(f"{key}={value}" for key, value in counters.items()),
                flush=True,
            )

    return counters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(STAGES), default="primary")
    parser.add_argument("--n", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=124_692)
    args = parser.parse_args()

    counters = verify(stage=args.stage, target=args.n, seed=args.seed)
    print(
        f"stage={args.stage} "
        + " ".join(f"{key}={value}" for key, value in counters.items())
    )
    failures = {key: value for key, value in counters.items() if key != "runs"}
    return int(counters["runs"] != args.n or any(failures.values()))


if __name__ == "__main__":
    raise SystemExit(main())
