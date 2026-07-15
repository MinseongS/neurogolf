#!/usr/bin/env python3
"""Regression contract for the task285 target-sentinel structural rescan."""

from __future__ import annotations

import importlib.util
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[4]
SCANNER = ROOT / "candidates/task285/rescan_redundant_target_sentinel.py"


def load_scanner():
    spec = importlib.util.spec_from_file_location(
        "task285_target_sentinel_scan", SCANNER
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_scanner_finds_historical_task285_and_clears_after_adoption():
    assert SCANNER.exists(), "target-sentinel scanner is not implemented"
    scanner = load_scanner()
    historical = ROOT / "candidates/task285/connectivity_fold_candidate.onnx"
    current = ROOT / "submission/overfit_nets/task285.onnx"
    assert scanner.scan_model(historical) == [
        {
            "task": 285,
            "scatter": "newg",
            "update": "v81_safe",
            "target_gather": "target_base",
        }
    ]
    assert scanner.scan_model(current) == []
