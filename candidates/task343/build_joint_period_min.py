"""Rebuild the best verified task343 candidate (cost 224)."""

from pathlib import Path
import sys

import onnx


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.custom.task343 import build  # noqa: E402


OUTPUT = Path(__file__).with_name("joint_period_min.onnx")


if __name__ == "__main__":
    onnx.save(build(None), OUTPUT)
    print(OUTPUT)
