"""Regression contract for source-owned task285/task295 deployed graphs."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import onnx
import pytest

from neurogolf.scoring import evaluate, load_task


ROOT = Path(__file__).resolve().parents[2]
CASES = ((285, 15648, 265), (295, 343, 268))


@pytest.mark.parametrize("task_num,expected_cost,expected_pass", CASES)
def test_source_rebuild_is_deployed_artifact(
    task_num: int, expected_cost: int, expected_pass: int
) -> None:
    module = importlib.import_module(f"src.custom.task{task_num:03d}")
    rebuilt = module.build(load_task(task_num))
    deployed_path = ROOT / f"submission/overfit_nets/task{task_num:03d}.onnx"
    deployed_bytes = deployed_path.read_bytes()
    rebuilt_bytes = rebuilt.SerializeToString()

    assert rebuilt_bytes == deployed_bytes, {
        "rebuilt_sha": hashlib.sha256(rebuilt_bytes).hexdigest(),
        "deployed_sha": hashlib.sha256(deployed_bytes).hexdigest(),
    }

    onnx.checker.check_model(rebuilt, full_check=True)
    result = evaluate(rebuilt, load_task(task_num), keep_failures=True)
    assert result["error"] is None
    assert result["fail"] == 0, result["failures"]
    assert result["pass"] == expected_pass
    assert result["memory"] + result["params"] == expected_cost


@pytest.mark.parametrize("task_num", (285, 295))
def test_source_has_no_external_artifact_dependency(task_num: int) -> None:
    source = (ROOT / f"src/custom/task{task_num:03d}.py").read_text()
    forbidden = ("onnx.load", "candidates", "submission", "networks", ".npy")
    assert not {token for token in forbidden if token in source}
