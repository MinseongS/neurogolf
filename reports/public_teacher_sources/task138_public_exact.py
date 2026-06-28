"""Task 138 public-teacher exact source draft.

Generated from `public_candidates/lucifer_agi_circuit_20260628/extracted/task138.onnx` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
    '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'
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))
