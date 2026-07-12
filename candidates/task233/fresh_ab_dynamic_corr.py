"""Isolated-process fresh A/B for the task233 dynamic-correlation candidate."""

import argparse
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from src.custom.task233 import build as build_baseline
from src.harness import convert_to_numpy, run_network, sanitize_model


MAIN_ROOT = Path("/Users/minseong/project/neurogolf")


def _load_generator():
    arcgen = MAIN_ROOT / "arc-gen"
    sys.path.insert(0, str(arcgen))
    path = arcgen / "tasks" / "task_97a05b5b.py"
    spec = importlib.util.spec_from_file_location("task233_fresh_generator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate


def _session(model_or_path) -> ort.InferenceSession:
    if isinstance(model_or_path, (str, Path)):
        model = onnx.load(model_or_path)
    else:
        model = onnx.ModelProto.FromString(model_or_path.SerializeToString())
    sanitized = sanitize_model(model)
    assert sanitized is not None
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--draws", type=int, required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    generate = _load_generator()
    baseline = _session(build_baseline(None))
    candidate = _session(Path("candidates/task233/cand_dynamic_corr.onnx"))

    base_fail = cand_fail = divergence = regress = recover = 0
    for _ in range(args.draws):
        example = generate()
        converted = convert_to_numpy(example)
        assert converted is not None
        expected = converted["output"]
        base_out = run_network(baseline, converted["input"])
        cand_out = run_network(candidate, converted["input"])
        base_wrong = not np.array_equal(base_out, expected)
        cand_wrong = not np.array_equal(cand_out, expected)
        base_fail += base_wrong
        cand_fail += cand_wrong
        different = not np.array_equal(base_out, cand_out)
        divergence += different
        regress += different and not base_wrong and cand_wrong
        recover += different and base_wrong and not cand_wrong

    print(
        f"seed={args.seed} draws={args.draws} base_fail={base_fail} "
        f"cand_fail={cand_fail} divergence={divergence} regress={regress} recover={recover}"
    )


if __name__ == "__main__":
    main()
